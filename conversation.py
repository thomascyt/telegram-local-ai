"""Conversation history manager with context budgeting and concurrency control.

Maintains independent conversation histories per Telegram user, enforces a
configurable turn limit, budgets tokens against the model context window,
and serialises requests for the same user via ``asyncio.Lock``.

The primary operation is ``respond()`` which atomically:
1. Acquires the per-user lock
2. Builds a prompt without modifying stored history
3. Applies one consistent context-budget calculation
4. Calls ``llm_client.complete()`` while holding the lock
5. Commits the complete user/assistant exchange only after success
6. Preserves history unchanged on failure/cancellation/oversized input
"""

from __future__ import annotations

import asyncio
import copy
import logging
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from llm_client import InferenceStats, ToolCall, _extract_tool_calls, parse_tool_arguments
from web_search import WebSearchClient, WebSearchError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = "You are a helpful assistant running locally inside a Telegram bot. Answer concisely and accurately."

# System prompt used when the web-search tool is enabled.  It tells the
# model it *may* call the tool and when that is appropriate; the tool is
# only actually sent when the user has opted in via ``/search on``.
_WEB_SEARCH_SYSTEM_PROMPT = (
    "You are a helpful assistant running locally inside a Telegram bot. "
    "You have access to a web_search tool. Use it when the user asks about "
    "current or time-sensitive information — news, today's events, live "
    "scores or results, stock prices, weather, product availability, "
    "recent software releases, or any factual question whose answer may "
    "have changed since your training. For everything else, answer directly "
    "from your knowledge. When you do use the tool, base your final answer "
    "on the results you receive. Answer concisely."
)

# Guard against runaway tool-call loops (e.g. a model that keeps calling the
# tool without ever producing a final answer).
MAX_TOOL_CALL_ITERATIONS = 2

# Default message-format overhead per message in token estimates (role tokens, metadata).
_MESSAGE_FORMAT_OVERHEAD_TOKENS = 2


@dataclass(frozen=True)
class ContextBudget:
    """Computed budget for prompt tokens given configuration knobs."""

    available: int
    """Token slots reserved for the conversation history (prompt)."""
    reserved_output: int
    """Slots carved out for maximum output tokens."""
    reserved_system: int
    """Slots carved out for the system prompt."""


@dataclass(frozen=True)
class ConversationResponse:
    """Structured result returned by :meth:`ConversationManager.respond`.

    Parameters
    ----------
    text:
        The assistant's reply text.
    stats:
        Inference statistics (token counts, speed) from the LLM, or
        ``None`` when inference is bypassed (plain string reply).
    snapshot:
        Full message list (system + history + new exchange), identical to
        the value returned by :meth:`get_messages` after the call.
    """

    text: str
    stats: InferenceStats | None
    snapshot: list[dict[str, str]]


# ---------------------------------------------------------------------------
# Helpers ─────────────────────────────────────────────────────────────────
# Token-count heuristics (no external tokenizer dependency).
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token, minimum 1.

    This heuristic is sufficient for context budgeting; exact counts come
    from the model's response usage field and are used verbatim there.
    """
    chars = len(text.encode("utf-8"))
    return max(1, chars // 4)


def build_system_prompt(enable_web_search: bool) -> str:
    """Return the system prompt for the current web-search setting.

    Parameters
    ----------
    enable_web_search:
        When ``True`` the prompt advertises the ``web_search`` tool so a
        tool-calling model knows it may invoke it.  When ``False`` the
        prompt is the plain, non-search variant (equivalent to the older
        default), and the tool is never sent to the model.
    """
    return (
        _WEB_SEARCH_SYSTEM_PROMPT if enable_web_search else _SYSTEM_PROMPT_TEMPLATE
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ConversationManager:
    """Manage per-user conversation histories with context budgeting.

    Parameters
    ----------
    context_length:
        Total token capacity of the model's context window (from config).
    max_tokens:
        Maximum output tokens reserved for each response.
    history_limit:
        Maximum number of complete user/assistant turns to retain per user.
    system_prompt:
        System prompt prepended to every conversation.  Defaults to a
        short, neutral prompt.
    safety_margin:
        Fraction of available budget reserved as a conservative cushion
        (default ``0.1``, i.e. 10 %).
    """

    def __init__(
        self,
        *,
        context_length: int = 8192,
        max_tokens: int = 4096,
        history_limit: int = 12,
        system_prompt: str = _SYSTEM_PROMPT_TEMPLATE,
        safety_margin: float = 0.1,
        brave_api_key: str | None = None,
    ) -> None:
        if history_limit < 1:
            raise ValueError("history_limit must be >= 1")
        if context_length <= 0 or max_tokens <= 0:
            raise ValueError("context_length and max_tokens must be positive")
        if max_tokens >= context_length:
            raise ValueError(
                "max_tokens must be less than context_length"
            )
        if not 0 <= safety_margin < 1:
            raise ValueError("safety_margin must be in [0, 1)")

        self._context_length = context_length
        self._max_tokens = max_tokens
        self._history_limit = history_limit
        self._system_prompt = system_prompt
        self._safety_margin = safety_margin
        self._brave_api_key = brave_api_key
        # Per-user message lists; insertion-ordered since Python 3.7+.
        self._histories: OrderedDict[int, list[dict[str, str]]] = (
            OrderedDict()
        )
        # Per-user locks — one lock key per user id.
        self._locks: dict[int, asyncio.Lock] = {}

    # ── Context budgeting ────────────────────────────────────────────

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def context_length(self) -> int:
        return self._context_length

    def get_budget(self) -> ContextBudget:
        """Return the current context budget for prompt tokens."""
        reserved_system = _estimate_tokens(self._system_prompt)
        reserved_output = self._max_tokens
        available = self._context_length - reserved_system - reserved_output
        # Apply conservative safety margin.
        available = int(available * (1 - self._safety_margin))
        # Reserve slots for message-format overhead (role tokens, metadata).
        available -= _MESSAGE_FORMAT_OVERHEAD_TOKENS
        return ContextBudget(
            available=max(0, available),
            reserved_output=reserved_output,
            reserved_system=reserved_system,
        )

    def _prompt_tokens(self, messages: list[dict[str, str]]) -> int:
        """Estimate total tokens in *messages*."""
        return sum(_estimate_tokens(str(m.get("content", ""))) for m in messages)

    # ── User access ──────────────────────────────────────────────────

    def _get_lock(self, user_id: int) -> asyncio.Lock:
        """Return (creating on demand) the lock for *user_id*."""
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    def _get_history(self, user_id: int) -> list[dict[str, str]]:
        """Return (creating on demand) the message list for *user_id*."""
        if user_id not in self._histories:
            # Prepend the system prompt as the first message.
            self._histories[user_id] = [
                {"role": "system", "content": self._system_prompt}
            ]
            self._histories.move_to_end(user_id, last=False)
        return self._histories[user_id]

    def _remove_user(self, user_id: int) -> None:
        """Permanently remove a user's history and lock."""
        self._histories.pop(user_id, None)
        self._locks.pop(user_id, None)

    # ── Tool calling ───────────────────────────────────────────────

    async def _run_with_tools(
        self,
        llm_client: Any,
        messages: list[dict[str, str]],
        enable_web_search: bool,
    ) -> Any:
        """Run inference, executing web-search tool calls as they arise.

        When ``enable_web_search`` is ``True`` and the model returns
        ``tool_calls``, the tool is executed and the results are appended as
        ``tool`` messages before re-invoking inference, up to
        :data:`MAX_TOOL_CALL_ITERATIONS` iterations.  When the model produces
        a plain text reply (or search is disabled) the final result is
        returned directly.

        The *messages* list is mutated in place with tool messages; the
        caller is responsible for discarding those before persisting history.
        """
        # The tool schema is only attached when the user opted in via
        # ``/search on``; ``/search off`` sends no tool at all.
        tools = [WebSearchClient.tool_definition()] if enable_web_search else None

        current_messages = messages
        for _ in range(MAX_TOOL_CALL_ITERATIONS):
            assistant_reply = await llm_client.complete(
                current_messages, tools=tools
            )

            # Pull the raw response to inspect for tool calls.  The reply
            # may be an InferenceResult (real client) or a MagicMock (tests);
            # both carry a ``.data`` attribute in our setup.
            raw = getattr(assistant_reply, "data", None)
            tool_calls = _extract_tool_calls(raw) if raw else []

            if not tool_calls:
                return assistant_reply

            # Execute each declared tool call and append its result.
            for call in tool_calls:
                if call.name != "web_search":
                    logger.warning(
                        "Ignoring unsupported tool call: %r", call.name
                    )
                    continue
                args = parse_tool_arguments(call.arguments)
                query = str(args.get("query", "")).strip()
                if not query:
                    logger.warning(
                        "web_search called with no query; skipping"
                    )
                    result_text = "web_search returned no query."
                else:
                    result_text = await self._execute_web_search(query)
                current_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": result_text,
                    }
                )
            # Loop: feed the tool results back to the model.

        logger.warning(
            "Tool-call iteration cap (%d) reached without a final answer; "
            "returning the last inference result",
            MAX_TOOL_CALL_ITERATIONS,
        )
        return assistant_reply

    async def _execute_web_search(self, query: str) -> str:
        """Run a Brave web search and return formatted context, or ``None``.

        Returns an empty-context ``None`` when no results are found or when
        the search fails; callers treat both as "no search context" and let
        the model fall back to a best-effort answer.
        """
        try:
            async with WebSearchClient(
                self._brave_api_key,
            ) as search_client:
                results = await search_client.search(query)
                if results:
                    context = await search_client.format_context(results)
                    logger.info(
                        "Web search: %d results, %d chars of context",
                        len(results),
                        len(context),
                    )
                    return context
                logger.info("Web search: no results for query")
        except (WebSearchError, ValueError) as exc:
            logger.warning("Web search failed: %s", exc)
        return None

    async def respond(
        self,
        user_id: int,
        content: str,
        llm_client: Any,
        *,
        search_context: str | None = None,
        enable_web_search: bool = False,
    ) -> ConversationResponse:
        """Add a user message, call LLM, and commit the exchange atomically.

        This is the primary operation for interacting with the conversation
        manager. It holds the per-user lock for the entire transaction to
        prevent races between inference and history modification.

        Parameters
        ----------
        user_id:
            The Telegram user identifier.
        content:
            The user's message text.
        llm_client:
            An ``LlmClient`` instance with a ``complete(messages, tools=...)``
            method.
        search_context:
            Optional web-search context to prepend as a system message
            (used by the ``/search query`` one-off path).
        enable_web_search:
            When ``True``, the ``web_search`` tool is advertised to a
            tool-calling model.  If the model returns a ``tool_calls`` entry,
            the tool is executed via ``llm_client`` (which must expose an
            ``execute_tool`` callable) and the results are fed back for a
            follow-up inference, up to :data:`MAX_TOOL_CALL_ITERATIONS`.
            When ``False``, the tool is never sent and no tool handling runs.

        Returns
        -------
        ConversationResponse
            A structured result containing the assistant's reply text,
            inference statistics (token counts, speed), and a snapshot
            of the full message list ready for further inference.

        Raises
        ------
        ValueError
            When the user message exceeds the available context budget.
        InferenceError
            When ``llm_client.complete()`` fails (history unchanged).
        """
        # Step 1: Acquire lock before any read/modify
        async with self._get_lock(user_id):
            history = self._get_history(user_id)

            # ── Budget check on temp copy — no mutation of stored history ──
            msg_tokens = _estimate_tokens(content)
            budget = self.get_budget()

            # Build trimmed prompt candidate (all turns + user message).
            # Work on a temp copy so trimming never persists before inference.
            temp_stored = list(history)  # fresh copy of current stored history
            self._trim_inplace(temp_stored, budget)

            # Check if this message would fit with trimmed history
            temp_history = list(temp_stored) + [
                {"role": "user", "content": content}
            ]
            temp_tokens = self._prompt_tokens(temp_history)

            if temp_tokens > budget.available:
                # Even after trimming all previous turns, message doesn't fit.
                # Try minimum (system + this user message only).
                min_history = [history[0], {"role": "user", "content": content}]
                min_tokens = self._prompt_tokens(min_history)
                if min_tokens > budget.available:
                    raise ValueError(
                        f"Message ({msg_tokens} estimated tokens) exceeds "
                        f"available context budget ({budget.available} tokens). "
                        f"Reduce your message length or increase "
                        f"CONTEXT_LENGTH / decrease MAX_TOKENS."
                    )

                # Trim to minimum on temp_stored before LLM call.
                temp_stored[:] = [history[0]]  # system only
                messages = list(temp_stored) + [{"role": "user", "content": content}]
            else:
                messages = list(temp_history)  # trimmed history + user message

            # Inject web-search context as a system message (before inference).
            if search_context:
                messages.insert(
                    1,
                    {
                        "role": "system",
                        "content": f"Web search results for your latest query:\n\n{search_context}",
                    },
                )

            # Step 2: Call LLM (with the web-search tool when enabled) and,
            # if the model asks to use it, execute the tool and re-invoke
            # until it produces a final answer or we hit the iteration cap.
            assistant_reply = await self._run_with_tools(
                llm_client, messages, enable_web_search
            )
            reply_text = assistant_reply.text if hasattr(assistant_reply, 'text') else str(assistant_reply)

            # Extract stats from the InferenceResult (which carries its token
            # counts and speed directly, with no nested ``.usage`` wrapper).
            stats = None
            if hasattr(assistant_reply, 'prompt_tokens') or hasattr(assistant_reply, 'completion_tokens'):
                # InferenceResult has token stats directly on it (no .usage wrapper).
                prompt_tokens = getattr(assistant_reply, 'prompt_tokens', None)
                completion_tokens = getattr(assistant_reply, 'completion_tokens', None)
                total_tokens = getattr(assistant_reply, 'total_tokens', None)
                if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                    total_tokens = prompt_tokens + completion_tokens
                tokens_per_second = getattr(assistant_reply, 'tokens_per_second', None)
                if (prompt_tokens or completion_tokens):  # non-trivial usage info
                    stats = InferenceStats(
                        prompt_tokens=prompt_tokens or 0,
                        completion_tokens=completion_tokens or 0,
                        total_tokens=total_tokens or 0,
                        tokens_per_second=tokens_per_second,
                    )

            # Step 3: Commit the COMPLETE exchange only after success.
            #   - Drop any ``tool`` messages ``_run_with_tools`` appended so the
            #     stored history stays well-formed (system + turns, no orphaned
            #     tool results).
            #   - Append the assistant message.
            #   - Trim again for history_limit
            committed = [m for m in messages if m.get("role") != "tool"]
            if not reply_text or not reply_text.strip():
                # The model returned an empty reply (e.g. the tool-call cap was
                # hit while it was still emitting tool calls). Keep a non-empty
                # fallback so the stored history is well-formed.
                reply_text = "I couldn't produce a final answer right now."
            self._histories[user_id] = list(committed)
            self._histories[user_id].append(
                {"role": "assistant", "content": reply_text}
            )
            self._trim_inplace(self._histories[user_id], budget)

            # Step 4: Build and return ConversationResponse (still holding lock).
            snapshot = copy.deepcopy(self._get_history(user_id))
            return ConversationResponse(text=reply_text, stats=stats, snapshot=snapshot)

    async def add_message(
        self,
        user_id: int,
        content: str,
        *,
        assistant_reply: str | None = None,
    ) -> list[dict[str, str]]:
        """Add a user message (and optionally the assistant reply) to history.

        Enforces context budgeting by trimming oldest turns before returning
        the final message list suitable for ``LlmClient.complete``.

        The *assistant_reply* is stored only when **not** ``None``, which
        means failed inferences are never persisted.

        Returns
        -------
        list[dict[str, str]]
            The full message list (system + history) ready for inference.

        Raises
        ------
        ValueError
            When a single message exceeds the available context budget.

        .. deprecated::
            Use :meth:`respond` for atomic user+assistant exchanges with
            LLM inference. This method is kept for compatibility.
        """
        async with self._get_lock(user_id):
            history = self._get_history(user_id)
            # Append user message (newest goes to the end).
            history.append({"role": "user", "content": content})

            if assistant_reply is not None:
                history.append({"role": "assistant", "content": assistant_reply})

            # Drop any ``tool`` messages that ``_run_with_tools`` appended to
            # *messages* while executing web-search calls; only the final
            # assistant text belongs in the persisted conversation history.
            history[:] = [m for m in history if m.get("role") != "tool"]

            self._trim(user_id)
            result = list(self._histories[user_id])

        return result

    async def get_messages(self, user_id: int) -> list[dict[str, str]]:
        """Return the current message list for *user_id*.

        Returns a defensive copy to prevent mutation of internal state.

        Returns
        -------
        list[dict[str, str]] or None — ``None`` if the user has no history.
        """
        if user_id not in self._histories:
            return None
        # Return a deep copy to prevent mutation of internal state
        return copy.deepcopy(self._histories[user_id])

    async def clear_user(self, user_id: int) -> None:
        """Remove only *user_id*'s conversation history.

        Lock is preserved so concurrent access remains serialised.
        """
        async with self._get_lock(user_id):
            # Only clear the history — keep the lock stable
            self._histories.pop(user_id, None)

    def _trim_inplace(self, messages: list[dict[str, str]], budget: ContextBudget) -> None:
        """Trim *messages* in-place for context budget and history limit.

        Unlike ``_trim``, this operates on an arbitrary message list rather
        than ``self._histories[user_id]``.  It does not touch locks or users.

        Steps:
        1. Oversized-message check (raise if even system + user doesn't fit)
        2. Trim oldest complete turns (user+assistant pairs) from the front
        3. Enforce history_limit by dropping oldest pairs from the front
        """
        min_messages = 2  # [system, user]

        # ── Oversized-message check ────────────────────────────────
        last_user = None
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m
                break

        if last_user is not None:
            msg_tokens = _estimate_tokens(str(last_user.get("content", "")))
            # Check if this message alone fits with system prompt
            min_history_tokens = _estimate_tokens(messages[0].get("content", "")) + msg_tokens
            if min_history_tokens > budget.available:
                raise ValueError(
                    f"Message ({msg_tokens} estimated tokens) exceeds "
                    f"available context budget ({budget.available} tokens). "
                    f"Reduce your message length or increase "
                    f"CONTEXT_LENGTH / decrease MAX_TOKENS."
                )

        # ── Trim oldest complete turns (user+assistant pairs) ───────
        while True:
            ptokens = self._prompt_tokens(messages)
            if ptokens <= budget.available:
                break

            # Find the first user turn at or after index 1 (skip system)
            idx_user = None
            for i, m in enumerate(messages):
                if m.get("role") == "user" and i >= 1:
                    idx_user = i
                    break

            if idx_user is None:
                # Only system prompt left — nothing more to trim
                break

            # Remove the user message; remove its assistant reply too if present
            messages.pop(idx_user)
            if idx_user < len(messages) and messages[idx_user].get("role") == "assistant":
                messages.pop(idx_user)

            if len(messages) <= min_messages:
                break

        # ── Enforce history_limit (drop oldest pairs from front) ────
        non_system_count = sum(
            1 for m in messages[1:]
            if m.get("role") in ("user", "assistant")
        )
        pairs = non_system_count // 2
        max_pairs = self._history_limit

        if pairs > max_pairs:
            to_remove = (pairs - max_pairs) * 2
            del messages[1 : 1 + to_remove]

    def _trim(self, user_id: int) -> None:
        """Apply context budget and history limit to the stored history.

        This method mutates ``self._histories[user_id]`` in-place.

        Steps:
        1. Check if the last user message exceeds the budget even alone
        2. Trim oldest complete turns (user+assistant pairs) from the front
        3. Enforce history_limit by dropping oldest pairs from the front
        """
        budget = self.get_budget()
        messages = self._histories[user_id]

        # We must always keep at least the system prompt + 1 user turn.
        min_messages = 2  # [system, user]

        # ── Oversized-message check ────────────────────────────────
        # Peek at the last user message (the one we just added).
        last_user = None
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m
                break

        if last_user is not None:
            msg_tokens = _estimate_tokens(str(last_user.get("content", "")))
            # Check if this message alone fits with system prompt
            min_history_tokens = _estimate_tokens(messages[0].get("content", "")) + msg_tokens
            if min_history_tokens > budget.available:
                # This should have been caught before calling _trim,
                # but guard anyway
                raise ValueError(
                    f"Message ({msg_tokens} estimated tokens) exceeds "
                    f"available context budget ({budget.available} tokens). "
                    f"Reduce your message length or increase "
                    f"CONTEXT_LENGTH / decrease MAX_TOKENS."
                )

        # ── Trim oldest complete turns (user+assistant pairs) ───────
        # Remove from the front (after system prompt at index 0).
        while True:
            ptokens = self._prompt_tokens(messages)
            if ptokens <= budget.available:
                break
            
            # Find the first user turn at or after index 1 (skip system)
            idx_user = None
            for i, m in enumerate(messages):
                if m.get("role") == "user" and i >= 1:
                    idx_user = i
                    break
            
            if idx_user is None:
                # Only system prompt left — nothing more to trim
                break

            # Remove the user message; remove its assistant reply too if present
            messages.pop(idx_user)
            if idx_user < len(messages) and messages[idx_user].get("role") == "assistant":
                messages.pop(idx_user)

            if len(messages) <= min_messages:
                break

        # ── Enforce history_limit (drop oldest pairs from front) ────
        # Count non-system messages (user+assistant pairs)
        non_system_count = sum(
            1 for m in messages[1:] 
            if m.get("role") in ("user", "assistant")
        )
        pairs = non_system_count // 2
        max_pairs = self._history_limit
        
        if pairs > max_pairs:
            # Remove from the front (after system at index 0)
            to_remove = (pairs - max_pairs) * 2
            del messages[1 : 1 + to_remove]

        logger.debug(
            "User %d: %d turns after budget trim (budget=%d)",
            user_id,
            len([m for m in messages if m.get("role") == "user"]),
            budget.available,
        )

    # ── Housekeeping ─────────────────────────────────────────────────

    def remove_user(self, user_id: int) -> None:
        """Remove a user's history and lock.

        .. deprecated::
            Use :meth:`clear_user` instead, which acquires the per-user
            lock to prevent races during active inference.

        .. warning::
            This method does NOT acquire the per-user lock. Callers should
            acquire the lock first if concurrent access is possible. For
            safe removal during active inference, use :meth:`clear_user`
            instead (which acquires the lock).
        """
        warnings.warn(
            "remove_user() is deprecated; use clear_user() for safe, "
            "lock-acquired removal",
            DeprecationWarning,
            stacklevel=2,
        )
        self._remove_user(user_id)
