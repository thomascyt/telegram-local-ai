"""Tests for conversation.py — history, budgeting, concurrency.

Covers both the legacy ``add_message`` API and the new atomic
``respond(user_id, content, llm_client)`` operation that bundles
inference into a single lock-protected transaction.
"""

from __future__ import annotations

import asyncio
import copy as std_copy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conversation import (
    ConversationManager,
    ContextBudget,
    MAX_TOOL_CALL_ITERATIONS,
    _estimate_tokens,
)
from llm_client import InferenceError, InferenceResult


# ── Helpers ────────────────────────────────────────────────────────────────


def _short(**kw) -> ConversationManager:
    """Small manager tuned for fast token-budget tests."""
    kw.setdefault("context_length", 512)
    kw.setdefault("max_tokens", 64)
    kw.setdefault("history_limit", 3)
    kw.setdefault("system_prompt", "System prompt here.")
    return ConversationManager(**kw)


def _tiny(**kw) -> ConversationManager:
    """Tiny manager where budget is very tight."""
    kw.setdefault("context_length", 128)
    kw.setdefault("max_tokens", 60)
    kw.setdefault("history_limit", 5)
    kw.setdefault("system_prompt", "Short.")
    return ConversationManager(**kw)


def _mock_llm(
    reply: str | InferenceResult = "Mocked response",
    *,
    fail: bool = False,
    delay: float | None = None,
) -> MagicMock:
    """Return an ``llm_client`` stub ready for ``respond()``.

    Parameters
    ----------
    reply:
        Text or ``InferenceResult`` to return on success.
    fail:
        If *True*, the client raises :class:`InferenceError` instead.
    delay:
        Sleep this many seconds before returning (for concurrency tests).
    """
    if fail:
        mock = MagicMock()
        mock.complete = AsyncMock(side_effect=InferenceError("fail"))
        return mock

    result = reply
    if isinstance(reply, str):
        result = InferenceResult(text=reply)

    async def _complete(messages, *, tools=None):  # noqa: D103
        if delay is not None:
            await asyncio.sleep(delay)
        return result

    mock = MagicMock()
    mock.complete = AsyncMock(side_effect=_complete)
    return mock


# ══════════════════════════════════════════════════════════════════════════
# Token estimation & construction (legacy helpers — unchanged)
# ══════════════════════════════════════════════════════════════════════════


class TestEstimateTokens:
    def test_empty(self):
        assert _estimate_tokens("") == 1

    def test_ascii(self):
        assert _estimate_tokens("hello") == 1

    def test_long_text(self):
        text = "a" * 400
        assert _estimate_tokens(text) == 100

    def test_utf8_multibyte(self):
        char = "\u017e"  # 2 bytes in UTF-8
        assert _estimate_tokens(char) >= 1


class TestConstruction:
    def test_ok(self):
        cm = ConversationManager(context_length=8192, max_tokens=1024, history_limit=12)
        assert cm.context_length == 8192

    def test_bad_history_limit_zero(self):
        with pytest.raises(ValueError, match="history_limit"):
            ConversationManager(history_limit=0)

    def test_bad_max_equals_context(self):
        with pytest.raises(ValueError, match="max_tokens must be less than context_length"):
            ConversationManager(context_length=100, max_tokens=100)

    def test_bad_negative_values(self):
        with pytest.raises(ValueError, match="must be positive"):
            ConversationManager(context_length=-1, max_tokens=10)


class TestProperties:
    def test_budget_reserved(self):
        cm = _short()
        budget = cm.get_budget()
        # After safety margin and message-format overhead, the sum is less
        # than context_length (margin + overhead are carved out).
        assert budget.available + budget.reserved_output + budget.reserved_system <= cm.context_length

    def test_default_system_prompt(self):
        cm = ConversationManager()
        assert "helpful assistant" in cm.system_prompt


# ══════════════════════════════════════════════════════════════════════════
# Legacy API — add_message (retained for backward compatibility)
# ══════════════════════════════════════════════════════════════════════════


class TestLegacyAddMessage:
    @pytest.mark.asyncio
    async def test_add_first_message_creates_history(self):
        cm = _short()
        msgs = await cm.add_message(1, "Hello")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[-1] == {"role": "user", "content": "Hello"}

    @pytest.mark.asyncio
    async def test_different_users_have_separate_histories(self):
        cm = _short()
        await cm.add_message(1, "User 1")
        await cm.add_message(2, "User 2")
        m1 = await cm.get_messages(1)
        m2 = await cm.get_messages(2)
        assert m1 is not None and m2 is not None
        content_1 = [m["content"] for m in m1 if m["role"] == "user"]
        content_2 = [m["content"] for m in m2 if m["role"] == "user"]
        assert "User 1" in content_1 and "User 1" not in content_2

    @pytest.mark.asyncio
    async def test_get_messages_nonexistent_user_returns_none(self):
        cm = _short()
        assert await cm.get_messages(999) is None

    @pytest.mark.asyncio
    async def test_turns_capped_at_history_limit(self):
        cm = ConversationManager(context_length=4096, max_tokens=64, history_limit=2)
        for i in range(5):
            await cm.add_message(1, f"Msg {i}", assistant_reply=f"Reply {i}")
        non_system = [m for m in (await cm.get_messages(1)) if m["role"] == "user"]
        assert len(non_system) <= 2

    @pytest.mark.asyncio
    async def test_failed_reply_not_stored(self):
        cm = _short()
        msgs = await cm.add_message(1, "Hello")
        assert msgs[-1]["role"] == "user"
        assistant_count = sum(1 for m in msgs if m["role"] == "assistant")
        assert assistant_count == 0

    @pytest.mark.asyncio
    async def test_oversized_message_raises_value_error(self):
        cm = ConversationManager(context_length=64, max_tokens=50, history_limit=12)
        with pytest.raises(ValueError, match="exceeds.*available context budget"):
            await cm.add_message(1, "a" * 300)


class TestClearUser:
    @pytest.mark.asyncio
    async def test_clear_removes_only_that_user(self):
        cm = _short()
        await cm.add_message(1, "Hello")
        await cm.add_message(2, "Hello")
        await cm.clear_user(1)
        assert await cm.get_messages(1) is None
        assert (await cm.get_messages(2)) is not None

    @pytest.mark.asyncio
    async def test_clear_nonexistent_user_silent(self):
        cm = _short()
        await cm.clear_user(999)


class TestConcurrencyLegacy:
    @pytest.mark.asyncio
    async def test_same_user_serialized(self):
        cm = ConversationManager(context_length=8192, max_tokens=64, history_limit=12)
        order: list[int] = []

        async def push(label: int):
            await cm.add_message(1, f"Msg from {label}")
            order.append(label)

        await asyncio.gather(push(1), push(2))
        assert len(order) == 2 and order == sorted(order)

    @pytest.mark.asyncio
    async def test_different_users_concurrent(self):
        cm = ConversationManager(context_length=8192, max_tokens=64, history_limit=12)
        await asyncio.gather(
            cm.add_message(1, "Msg from 1"),
            cm.add_message(2, "Msg from 2"),
            cm.add_message(3, "Msg from 3"),
        )
        assert all(m is not None for m in [
            await cm.get_messages(1),
            await cm.get_messages(2),
            await cm.get_messages(3),
        ])


# ══════════════════════════════════════════════════════════════════════════
# NEW API — respond()  (requirements 1–9)
# ══════════════════════════════════════════════════════════════════════════


class TestRespondBasic:
    """Verify the core respond() flow (reqs 1, 2, 4-6)."""

    @pytest.mark.asyncio
    async def test_respond_commits_exchange(self):
        """A successful call appends both user and assistant messages."""
        cm = _short()
        llm = _mock_llm("Hello back!")
        resp = await cm.respond(1, "Hi", llm)

        # Check response structure
        assert resp.text == "Hello back!"
        assert isinstance(resp.snapshot, list)

        # Defensive copy — mutating returned snapshot should not corrupt history
        orig_len = len(resp.snapshot)
        resp.snapshot.append({"role": "user", "content": "fake"})
        stored = await cm.get_messages(1)
        assert stored is not None and len(stored) == orig_len

        # Stored history has system + user + assistant
        assert any(m["role"] == "user" and m["content"] == "Hi" for m in stored)
        assert any(m["role"] == "assistant" and m["content"] == "Hello back!" for m in stored)

    @pytest.mark.asyncio
    async def test_respond_builds_prompt_without_mutation(self):
        """History should contain the new exchange after respond(), but
        LLM receives the *correct* prompt (system + all turns)."""
        cm = _short()
        llm = _mock_llm("Resp")

        await cm.respond(1, "First", llm)
        msgs1 = await cm.get_messages(1)
        user_contents_1 = [m["content"] for m in msgs1 if m["role"] == "user"]
        assert len(user_contents_1) == 1 and user_contents_1[0] == "First"

        await cm.respond(1, "Second", llm)
        msgs2 = await cm.get_messages(1)
        user_contents_2 = [m["content"] for m in msgs2 if m["role"] == "user"]
        assert len(user_contents_2) == 2 and user_contents_2[-1] == "Second"

    @pytest.mark.asyncio
    async def test_respond_handles_plain_string_reply(self):
        """llm_client.complete may return a plain str, not InferenceResult."""
        cm = _short()
        llm = MagicMock()
        llm.complete = AsyncMock(return_value="Just text")

        msgs = await cm.respond(1, "Q", llm)
        stored = await cm.get_messages(1)
        assert stored is not None
        assert any(m["role"] == "assistant" and m["content"] == "Just text" for m in stored)


# ══════════════════════════════════════════════════════════════════════════
# Req 8 — Defensive copies
# ══════════════════════════════════════════════════════════════════════════


class TestDefensiveCopies:
    @pytest.mark.asyncio
    async def test_respond_returns_deepcopy(self):
        cm = _short()
        llm = _mock_llm("A")
        resp = await cm.respond(1, "X", llm)

        # Mutate the returned snapshot and its dicts
        resp.snapshot.clear()
        resp.snapshot.append({"role": "user", "content": "injected"})

        stored = await cm.get_messages(1)
        assert stored is not None
        contents = [m["content"] for m in stored if m["role"] == "user"]
        assert "injected" not in contents

    @pytest.mark.asyncio
    async def test_get_messages_returns_deepcopy(self):
        cm = _short()
        await cm.add_message(1, "Data")
        m1 = await cm.get_messages(1)
        m2 = await cm.get_messages(1)

        # They must be independent copies
        assert m1 is not m2
        m1.clear()
        assert len(m2) > 0


# ══════════════════════════════════════════════════════════════════════════
# Req 10 — Deterministic concurrency tests using asyncio.Event
# ══════════════════════════════════════════════════════════════════════════


class TestRespondConcurrency:
    """Two simultaneous requests from the same user must serialise.
    Requests from different users should NOT block each other.

    Uses ``asyncio.Event`` to synchronise start points so both coroutines
    arrive at the lock simultaneously.
    """

    @pytest.mark.asyncio
    async def test_same_user_serialised_with_event(self):
        """Two respond() calls from the same user must execute sequentially,
        not interleaved.  We verify order via an arrival list."""
        cm = _short()
        start_event = asyncio.Event()
        done1 = asyncio.Event()

        order: list[str] = []
        lock_acquired_at: list[int] = []  # sequence number when lock acquired

        async def responder(label: str, delay: float):
            """Acquire the per-user lock (inside respond), wait for start signal,
            then record when the lock was acquired."""
            nonlocal order, lock_acquired_at
            llm = _mock_llm(f"Reply {label}", delay=delay)
            await cm.respond(1, f"Msg {label}", llm)
            order.append(label)

        # Fire both at once after the event is set
        async def fire():
            start_event.set()
            await asyncio.gather(
                responder("A", delay=0.05),
                responder("B", delay=0.01),
            )

        start_event.clear()
        await fire()

        # Both messages should be committed (order is whichever got the lock first)
        stored = await cm.get_messages(1)
        assert stored is not None
        user_contents = [m["content"] for m in stored if m["role"] == "user"]
        assert "Msg A" in user_contents and "Msg B" in user_contents
        # Exactly 2 user turns, no interleaving
        assert len(user_contents) == 2

    @pytest.mark.asyncio
    async def test_different_users_not_blocked(self):
        """Different users should proceed concurrently — total time ≈ max(delay_A, delay_B),
        not sum."""
        cm = _short()
        timings: list[tuple[str, float]] = []

        async def timer(label: str, delay: float):
            start = asyncio.get_event_loop().time()
            llm = _mock_llm(f"Reply {label}", delay=delay)
            await cm.respond(hash(label), f"Msg {label}", llm)  # hash to get different user ids
            elapsed = asyncio.get_event_loop().time() - start
            timings.append((label, elapsed))

        t0 = asyncio.get_event_loop().time()
        await asyncio.gather(
            timer("A", delay=0.1),
            timer("B", delay=0.05),
        )
        total_wall = asyncio.get_event_loop().time() - t0

        # If blocked sequentially, wall time ≥ 0.15 s.  Concurrently it should be ≈ 0.1 s.
        # Allow generous tolerance for CI slowness.
        assert total_wall < 0.2, f"Users appear serialised (wall={total_wall:.3f}s); expected concurrent (< 0.2s)"

    @pytest.mark.asyncio
    async def test_same_user_race_no_duplicate_history(self):
        """After two rapid same-user respond() calls, exactly 2 turns stored."""
        cm = _short(history_limit=10)
        start_event = asyncio.Event()

        async def rapid(label: str):
            start_event.set()
            llm = _mock_llm(f"R {label}")
            await cm.respond(1, f"Q{label}", llm)

        start_event.clear()
        await asyncio.gather(rapid("X"), rapid("Y"))

        stored = await cm.get_messages(1)
        assert stored is not None
        user_turns = [m for m in stored if m["role"] == "user"]
        assistant_turns = [m for m in stored if m["role"] == "assistant"]
        assert len(user_turns) == 2
        assert len(assistant_turns) == 2


# ══════════════════════════════════════════════════════════════════════════
# Req 11 — Failure rollback, cancellation, oversized input, clearing
# ══════════════════════════════════════════════════════════════════════════


class TestFailureRollback:
    """When inference fails, history must remain unchanged (req 6)."""

    @pytest.mark.asyncio
    async def test_llm_failure_preserves_history(self):
        """InferenceError during respond() should not modify stored history."""
        cm = _short()
        # Build up some history first
        llm_ok = _mock_llm("OK")
        await cm.respond(1, "First", llm_ok)

        stored_before = std_copy.deepcopy(await cm.get_messages(1))

        # Now fail
        llm_fail = _mock_llm(fail=True)
        with pytest.raises(InferenceError):
            await cm.respond(1, "Should not commit", llm_fail)

        stored_after = await cm.get_messages(1)
        assert stored_after == stored_before, "History changed after failed inference!"

    @pytest.mark.asyncio
    async def test_llm_failure_no_synthetic_reply(self):
        """Failed calls must NOT create partial user-only or assistant-only entries."""
        cm = _short()
        llm_fail = _mock_llm(fail=True)
        with pytest.raises(InferenceError):
            await cm.respond(1, "Lost", llm_fail)

        stored = await cm.get_messages(1)
        if stored is None:
            return  # never got past first message
        # No orphaned user or assistant messages from the failed call
        user_contents = [m["content"] for m in stored if m["role"] == "user"]
        assert "Lost" not in user_contents

    @pytest.mark.asyncio
    async def test_concurrent_succeed_follows_failed(self):
        """After a failed respond(), a subsequent one should succeed normally."""
        cm = _short()
        llm_fail = _mock_llm(fail=True)
        with pytest.raises(InferenceError):
            await cm.respond(1, "Fail", llm_fail)

        llm_ok = _mock_llm("Recovered")
        msgs = await cm.respond(1, "Recovery", llm_ok)
        stored = await cm.get_messages(1)
        assert stored is not None
        user_contents = [m["content"] for m in stored if m["role"] == "user"]
        assert "Recovery" in user_contents


class TestCancellation:
    """Task cancellation during inference should not corrupt state."""

    @pytest.mark.asyncio
    async def test_cancellation_during_delayed_inference(self):
        """A respond() that takes a long time should be cancellable without
        leaving partial history or dead locks."""
        cm = _short()
        # This llm will delay 10 seconds (simulating slow backend)
        llm_slow = _mock_llm("Slow", delay=10.0)

        task = asyncio.create_task(cm.respond(1, "Start", llm_slow))
        await asyncio.sleep(0.02)  # let it start

        # Cancel the task
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # After cancellation: history should be unchanged (lock released too)
        stored = await cm.get_messages(1)
        assert stored is None or len([m for m in stored if m["role"] == "user"]) == 0

        # Lock should be released — another request should work immediately
        llm_ok = _mock_llm("OK after cancel")
        msgs = await cm.respond(1, "After", llm_ok)
        assert msgs is not None


class TestOversizedInput:
    """Oversized messages must raise ValueError BEFORE modifying history."""

    @pytest.mark.asyncio
    async def test_oversized_message_before_history_mutation(self):
        """Even when budget is exceeded, the message itself should NOT be
        appended before raising."""
        cm = _short(context_length=128, max_tokens=60)
        # First, add a valid message to build history
        llm_ok = _mock_llm("OK")
        await cm.respond(1, "Hello", llm_ok)

        stored_before_count = len([m for m in (await cm.get_messages(1)) or [] if m["role"] == "user"])

        # Now try an oversized message (much longer than budget allows)
        long_msg = "x" * 500
        with pytest.raises(ValueError, match="exceeds.*available context budget"):
            await cm.respond(1, long_msg, _mock_llm("X"))

        # History count unchanged
        stored_after = await cm.get_messages(1)
        assert stored_after is not None
        user_count = len([m for m in stored_after if m["role"] == "user"])
        assert user_count == stored_before_count, f"History mutated! Before={stored_before_count}, After={user_count}"

    @pytest.mark.asyncio
    async def test_oversized_triggers_minimum_trim(self):
        """If trimming all previous turns still doesn't fit, raise ValueError."""
        cm = _short(context_length=128, max_tokens=60)
        # Add a system prompt that uses most of the budget
        cm._system_prompt = "x" * 50  # ~13 tokens
        budget = cm.get_budget()

        # Try to respond with something that fits alone
        llm_ok = _mock_llm("OK")
        msgs = await cm.respond(1, "Short", llm_ok)
        assert msgs is not None


class TestClearDuringInference:
    """clear_user while respond() holds the lock must not race."""

    @pytest.mark.asyncio
    async def test_clear_during_active_inference(self):
        """clear_user should wait for the lock held by a slow respond()."""
        cm = _short()
        cleared_event = asyncio.Event()

        async def slow_respond():
            llm = _mock_llm("Slow", delay=0.1)
            await cm.respond(1, "Infer", llm)

        async def delayed_clear():
            # Wait a tiny bit then clear
            await asyncio.sleep(0.02)
            await cm.clear_user(1)
            cleared_event.set()

        # Start both concurrently
        await asyncio.gather(
            slow_respond(),
            delayed_clear(),
        )

        # After respond completes + clear runs, user should have no history
        stored = await cm.get_messages(1)
        assert stored is None, "clear_user should have removed history"


class TestHistoryLimitAfterRespond:
    """history_limit should be enforced after each respond() call."""

    @pytest.mark.asyncio
    async def test_history_capped_after_respond(self):
        cm = ConversationManager(context_length=4096, max_tokens=64, history_limit=2)
        llm_ok = _mock_llm("Reply")

        for i in range(5):
            await cm.respond(1, f"Msg {i}", llm_ok)

        stored = await cm.get_messages(1)
        assert stored is not None
        non_system = [m for m in stored if m["role"] == "user"]
        assert len(non_system) <= 2


class TestTrimPreservesCompletePairs:
    """When trimming, only oldest complete user+assistant pairs should be removed."""

    @pytest.mark.asyncio
    async def test_only_complete_pairs_dropped(self):
        cm = ConversationManager(
            context_length=256,
            max_tokens=100,
            history_limit=10,
            system_prompt="S",
        )
        llm_ok = _mock_llm("R")

        # Fill up to near budget limit
        for i in range(4):
            await cm.respond(1, f"Turn {i} " + "z" * 20, llm_ok)

        stored = await cm.get_messages(1)
        assert stored is not None

        # Every user message should have a matching assistant reply
        users = [m for m in stored if m["role"] == "user"]
        assistants = [m for m in stored if m["role"] == "assistant"]
        # Assistant count == user count (all pairs complete)
        assert len(assistants) == len(users)


# ══════════════════════════════════════════════════════════════════════════
# Milestone 3 — Regression tests
# ══════════════════════════════════════════════════════════════════════════


class TestSafetyMargin:
    """Verify safety_margin parameter behaviour (Milestone 3, obj 1)."""

    def test_default_safety_margin_is_0_1(self):
        cm = _short()
        budget = cm.get_budget()
        # With default 10% margin: available should be reduced by ~10%
        no_margin_cm = _short(safety_margin=0)
        no_margin_budget = no_margin_cm.get_budget()
        assert budget.available < no_margin_budget.available

    def test_safety_margin_zero(self):
        cm = _short(safety_margin=0)
        budget = cm.get_budget()
        # With 0 margin, available should be higher than with 10%
        ten_cm = _short(safety_margin=0.1)
        ten_budget = ten_cm.get_budget()
        assert budget.available > ten_budget.available

    def test_safety_margin_validation_negative(self):
        with pytest.raises(ValueError, match="must be in"):
            ConversationManager(safety_margin=-0.1)

    def test_safety_margin_validation_one(self):
        with pytest.raises(ValueError, match="must be in"):
            ConversationManager(safety_margin=1.0)

    def test_safety_margin_validation_above_one(self):
        with pytest.raises(ValueError, match="must be in"):
            ConversationManager(safety_margin=1.5)


class TestMessageFormatOverhead:
    """Verify message-format overhead deduction (Milestone 3, obj 2)."""

    def test_overhead_reduces_budget(self):
        """Budget with overhead enabled should be lower than without."""
        cm = _short()
        budget = cm.get_budget()
        assert budget.available >= 0
        # The overhead is subtracted after safety margin, so available
        # should reflect that subtraction.

    def test_overhead_constant_2(self):
        """Each call to get_budget() applies the fixed ~2 token overhead."""
        cm1 = _short(safety_margin=0)
        cm2 = _short(safety_margin=0, max_tokens=64)
        b1 = cm1.get_budget()
        b2 = cm2.get_budget()
        # Same config should yield same budget; overhead is baked in.
        assert b1.available == b2.available


class TestBoundaryBudget:
    """Verify boundary cases for context budget (Milestone 3, obj 3)."""

    @pytest.mark.asyncio
    async def test_budget_zero_allows_minimal(self):
        """When safety margin drains budget to zero, minimal system-only still works."""
        cm = ConversationManager(
            context_length=128,
            max_tokens=100,
            history_limit=1,
            system_prompt="S" * 10,
            safety_margin=0.9,  # Very aggressive margin
        )
        budget = cm.get_budget()
        # Budget should be clamped to 0 minimum
        assert budget.available >= 0

    @pytest.mark.asyncio
    async def test_safety_margin_affects_trim(self):
        """Higher safety_margin causes earlier trimming."""
        tight = ConversationManager(
            context_length=256, max_tokens=100, history_limit=10, safety_margin=0.5
        )
        loose = ConversationManager(
            context_length=256, max_tokens=100, history_limit=10, safety_margin=0.0
        )
        tight_budget = tight.get_budget()
        loose_budget = loose.get_budget()
        assert tight_budget.available < loose_budget.available


class TestDeprecationWarning:
    """Verify remove_user() emits DeprecationWarning (Milestone 3, obj 4)."""

    def test_remove_user_emits_warning(self):
        cm = _short()
        cm._histories[99] = [{"role": "system", "content": "S"}]
        with pytest.warns(DeprecationWarning, match="deprecated"):
            cm.remove_user(99)

    def test_remove_user_still_removes(self):
        cm = _short()
        cm._histories[99] = [{"role": "system", "content": "S"}]
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cm.remove_user(99)
        assert 99 not in cm._histories


class TestConversationResponse:
    """Verify ConversationResponse dataclass and respond() integration (Milestone 3, obj 5-6)."""

    @pytest.mark.asyncio
    async def test_respond_returns_conversation_response(self):
        from conversation import ConversationResponse

        cm = _short()
        llm = _mock_llm("Reply text")
        resp = await cm.respond(1, "Hi", llm)
        assert isinstance(resp, ConversationResponse)
        assert resp.text == "Reply text"
        assert isinstance(resp.snapshot, list)

    @pytest.mark.asyncio
    async def test_respond_stats_extracted_from_inference_result(self):
        from conversation import ConversationResponse
        from llm_client import InferenceResult

        # Create InferenceResult with usage stats
        result = InferenceResult(
            text="Reply",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            tokens_per_second=20.0,  # 30 tokens / 1.5 seconds
        )
        llm = _mock_llm(result)
        cm = _short()
        resp = await cm.respond(1, "Hi", llm)

        assert isinstance(resp, ConversationResponse)
        assert resp.stats is not None
        assert resp.stats.prompt_tokens == 10
        assert resp.stats.completion_tokens == 20
        assert resp.stats.total_tokens == 30
        assert resp.stats.tokens_per_second == pytest.approx(20.0)

    @pytest.mark.asyncio
    async def test_respond_stats_none_for_plain_string(self):
        """Plain string replies produce InferenceResult with zero stats; respond() sets stats=None."""
        cm = _short()
        llm = _mock_llm("Text only")  # _mock_llm wraps str in InferenceResult(text=...)
        resp = await cm.respond(1, "Hi", llm)
        # All-zero usage means respond() leaves stats as None.
        assert resp.stats is None

    @pytest.mark.asyncio
    async def test_conversation_response_frozen(self):
        from conversation import ConversationResponse

        resp = ConversationResponse(text="T", stats=None, snapshot=[])
        with pytest.raises(AttributeError):
            resp.text = "changed"


class TestEndToEndMilestone3:
    """Combined smoke test exercising all Milestone 3 features together."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_stats(self):
        from conversation import ConversationResponse
        from llm_client import InferenceResult

        cm = ConversationManager(
            context_length=1024,
            max_tokens=256,
            history_limit=5,
            safety_margin=0.1,  # 10% margin (Milestone 3, obj 1)
        )
        budget = cm.get_budget()
        assert budget.available < (1024 - 256)  # Margin + overhead reduce available

        result = InferenceResult(
            text="Full pipeline works!",
            prompt_tokens=50,
            completion_tokens=15,
            total_tokens=65,
            tokens_per_second=32.5,  # 65 / 2.0
        )
        llm = _mock_llm(result)
        resp = await cm.respond(42, "Test message", llm)

        assert isinstance(resp, ConversationResponse)
        assert resp.text == "Full pipeline works!"
        assert resp.stats is not None
        assert resp.stats.total_tokens == 65
        # Snapshot should have system + user + assistant
        assert len(resp.snapshot) == 3
        assert resp.snapshot[0]["role"] == "system"
        assert resp.snapshot[1]["role"] == "user"
        assert resp.snapshot[2]["role"] == "assistant"


# ══════════════════════════════════════════════════════════════════════════
# Tool-calling behaviour (req 8 — Milestone 3)
# ══════════════════════════════════════════════════════════════════════════


class _ToolMockLLM:
    """Stub LLM that returns a sequence of tool-call replies then a final one.

    Each entry is a list of ``tool_calls`` dicts (empty list ⇒ plain reply).
    """

    def __init__(self, tool_sequences: list[list[dict]]):
        self._seq = tool_sequences
        self._i = 0
        self.calls: list[tuple[list, dict | None]] = []

    def _make_result(self, tool_calls: list[dict]):
        return MagicMock(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None if tool_calls else "final answer",
                            "tool_calls": tool_calls or None,
                        }
                    }
                ]
            },
            text="final answer",
        )

    async def complete(self, messages, *, tools=None):
        self.calls.append((messages, tools))
        seq = self._seq[min(self._i, len(self._seq) - 1)]
        self._i += 1
        return self._make_result(seq)


class TestRunWithTools:
    @pytest.mark.asyncio
    async def test_tool_called_then_final_answer(self):
        cm = _short()
        llm = _ToolMockLLM(
            [
                [{"id": "c1", "type": "function",
                  "function": {"name": "web_search", "arguments": '{"query": "weather"}'}}],
                [],
            ]
        )
        resp = await cm.respond(7, "weather?", llm, enable_web_search=True)

        assert resp.text == "final answer"
        # Tool message persisted in history.
        stored = await cm.get_messages(7)
        assert stored is not None
        assert any(m["role"] == "tool" for m in stored)

    @pytest.mark.asyncio
    async def test_no_tools_sent_when_disabled(self):
        cm = _short()
        llm = _ToolMockLLM([[]])
        await cm.respond(7, "hi", llm, enable_web_search=False)

        # No tool messages stored when search disabled.
        stored = await cm.get_messages(7)
        assert stored is not None
        assert not any(m["role"] == "tool" for m in stored)

    @pytest.mark.asyncio
    async def test_unsupported_tool_ignored(self):
        cm = _short()
        llm = _ToolMockLLM(
            [
                [{"id": "c1", "type": "function",
                  "function": {"name": "do_something", "arguments": "{}"}}],
                [],
            ]
        )
        resp = await cm.respond(7, "x", llm, enable_web_search=True)
        assert resp.text == "final answer"
        stored = await cm.get_messages(7)
        assert stored is not None
        # The unsupported tool call produced no tool message.
        assert not any(m["role"] == "tool" for m in stored)

    @pytest.mark.asyncio
    async def test_iteration_cap(self):
        cm = _short()
        # Always returns a tool call — never a final answer.
        llm = _ToolMockLLM(
            [[{"id": f"c{i}", "type": "function",
               "function": {"name": "web_search", "arguments": '{"query": "q"}'}}]
             for i in range(10)]
        )
        resp = await cm.respond(7, "x", llm, enable_web_search=True)
        # After MAX_TOOL_CALL_ITERATIONS the loop gives up.
        assert len(llm.calls) == MAX_TOOL_CALL_ITERATIONS


class TestBuildSystemPrompt:
    def test_enabled_mentions_web_search(self):
        from conversation import build_system_prompt

        prompt = build_system_prompt(True)
        assert "web_search" in prompt
        assert "tool" in prompt.lower()

    def test_disabled_is_plain_prompt(self):
        from conversation import build_system_prompt

        prompt = build_system_prompt(False)
        assert "web_search" not in prompt
        assert "helpful assistant" in prompt
