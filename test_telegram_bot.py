"""Comprehensive pytest suite for ``telegram_bot.py``.

Covers every command handler, the message handler (including allow-list
enforcement and error branches), ``build_application()``, and the
``UserStats`` persistence helpers — *all offline* with mocked back-ends.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dataclass_field
from typing import Any

import pytest

from conversation import ConversationManager, ConversationResponse
from llm_client import InferenceError, InferenceResult, InferenceStats

import os

from telegram_bot import (
    TelegramBot,
    UserStats,
    _get_user_stats,
    _set_user_stats,
    main as _main_entry_point,
)
from config import Config


# ──────────── Fixtures & helpers ──────────────────────────────────────

ALLOWED_UID = 123456789

FAKE_TOKEN = "999:FAKETOKEN"

_BASE_CONFIG_KWARGS = dict(
    telegram_bot_token=FAKE_TOKEN,
    telegram_allowed_user_ids=frozenset([ALLOWED_UID]),
)


def _make_config(**kw: object) -> Config:
    """Merge *kw* into a minimal :class:`Config`."""
    base = {**_BASE_CONFIG_KWARGS, **kw}
    return Config(**base)  # type: ignore[arg-type]


@dataclass
class _MockUser:
    id: int


@dataclass
class _MockMessage:
    text: str
    user: "_MockUser"

    async def reply_text(self, text: str, **kwargs: Any):
        self._last_reply = (text, kwargs)

    @property
    def reply_text_log(self):
        return getattr(self, "_last_reply", None)

    @property
    def effective_user(self):
        return self.user


@dataclass
class _MockUpdate:
    message: "_MockMessage | None" = None
    command: str | None = None

    @property
    def effective_user(self) -> "_MockUser | None":
        if self.message is not None:
            return self.message.user
        return None


def _make_context(user_id: int = ALLOWED_UID, user_data: dict | None = None) -> Any:
    """Return a lightweight mock that satisfies ContextTypes.DEFAULT_TYPE."""
    from unittest.mock import AsyncMock, MagicMock

    ctx = MagicMock()
    ctx.user_data = user_data or {}
    ctx.args = []
    ctx.effective_user = _MockUser(id=user_id)
    ctx.bot = type('Bot', (), {'is_bot': True})()
    return ctx


def _fake_response(
    text: str = "Hello!",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    tokens_per_second: float | None = 2.5,
):
    stats = InferenceStats(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        tokens_per_second=tokens_per_second,
    )
    return ConversationResponse(text=text, stats=stats, snapshot=[])


def _mock_cm(
    respond_result: ConversationResponse | None = None,
    respond_raises: Exception | None = None,
    messages: list[dict[str, str]] | None = None,
) -> Any:
    """Return a MagicMock :class:`ConversationManager`."""
    from unittest.mock import AsyncMock

    cm = type("MockCM", (), {})()
    cm.respond = AsyncMock()
    if respond_raises is not None:
        cm.respond.side_effect = respond_raises
    elif respond_result is not None:
        cm.respond.return_value = respond_result
    else:
        cm.respond.return_value = _fake_response()

    cm.clear_user = AsyncMock()
    cm.get_messages = AsyncMock()
    if messages is not None:
        cm.get_messages.return_value = messages
    elif messages is not None or respond_result is not None or respond_raises is None:
        # default to empty list if not explicitly set and no error
        if cm.get_messages.side_effect is not None:
            pass  # error case above already set it
        else:
            cm.get_messages.return_value = []
    return cm


def _build_bot(
    config: Config | None = None,
    cm: Any | None = None,
) -> TelegramBot:
    """Helper to instantiate :class:`TelegramBot`."""
    cfg = config or _make_config()
    return TelegramBot(cfg, conversation_manager=cm)


# ──────────── UserStats helpers ──────────────────────────────────────

class TestUserStatsPersistence:
    def test_get_stats_empty_returns_defaults(self):
        ctx = _make_context(user_data={})
        stats = _get_user_stats(ctx)
        assert stats.prompt_tokens == 0
        assert stats.completion_tokens == 0
        assert stats.total_tokens == 0
        assert stats.tokens_per_second is None

    def test_get_stats_roundtrip(self):
        ctx = _make_context(user_data={})
        original = UserStats(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            tokens_per_second=3.14,
        )
        _set_user_stats(ctx, original)
        restored = _get_user_stats(ctx)
        assert restored == original

    def test_set_and_get_partial(self):
        ctx = _make_context(user_data={})
        stats = UserStats(prompt_tokens=99)
        _set_user_stats(ctx, stats)
        recovered = _get_user_stats(ctx)
        assert recovered.prompt_tokens == 99
        assert recovered.completion_tokens == 0


# ──────────── /start handler ─────────────────────────────────────────

class TestCmdStart:
    @pytest.mark.asyncio
    async def test_start_replies_with_welcome(self):
        bot = _build_bot()
        msg = _MockMessage(text="/start", user=_MockUser(ALLOWED_UID))
        upd = _MockUpdate(message=msg)
        ctx = _make_context()
        await bot._cmd_start(upd, ctx)  # type: ignore[arg-type]
        reply = msg.reply_text_log
        assert reply is not None
        text, kwargs = reply
        assert "Bot started" in text
        assert "Markdown" in kwargs.get("parse_mode", "")

    @pytest.mark.asyncio
    async def test_start_shows_model_name(self):
        cfg = _make_config(llm_model="custom-model-x")
        bot = _build_bot(config=cfg)
        msg = _MockMessage(text="/start", user=_MockUser(ALLOWED_UID))
        ctx = _make_context()
        await bot._cmd_start(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]
        assert "custom-model-x" in msg.reply_text_log[0]


# ──────────── /help handler ──────────────────────────────────────────

class TestCmdHelp:
    @pytest.mark.asyncio
    async def test_help_lists_commands(self):
        bot = _build_bot()
        msg = _MockMessage(text="/help", user=_MockUser(ALLOWED_UID))
        ctx = _make_context()
        await bot._cmd_help(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]
        text = msg.reply_text_log[0]
        for cmd in ["/start", "/help", "/new", "/status", "/model", "/stats", "/history"]:
            assert cmd in text


# ──────────── /new handler ───────────────────────────────────────────

class TestCmdNew:
    @pytest.mark.asyncio
    async def test_new_clears_history_and_stats(self):
        cm = _mock_cm()
        bot = _build_bot(cm=cm)
        msg = _MockMessage(text="/new", user=_MockUser(ALLOWED_UID))
        ctx = _make_context(user_data={"_stats": {"total_tokens": 42}})
        await bot._cmd_new(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]
        cm.clear_user.assert_awaited_once_with(ALLOWED_UID)
        assert "_stats" not in ctx.user_data

    @pytest.mark.asyncio
    async def test_new_confirms_clear(self):
        cm = _mock_cm()
        bot = _build_bot(cm=cm)
        msg = _MockMessage(text="/new", user=_MockUser(ALLOWED_UID))
        ctx = _make_context()
        await bot._cmd_new(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]
        assert "cleared" in msg.reply_text_log[0].lower()


# ──────────── /clear alias handler ───────────────────────────────────

class TestCmdClear:
    @pytest.mark.asyncio
    async def test_clear_is_alias_for_new(self):
        cm = _mock_cm()
        bot = _build_bot(cm=cm)
        msg = _MockMessage(text="/clear", user=_MockUser(ALLOWED_UID))
        ctx = _make_context(user_data={"_stats": {"total_tokens": 42}})
        # /clear is wired to the same coroutine as /new
        await bot._cmd_new(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]
        cm.clear_user.assert_awaited_once()


# ──────────── /status handler ────────────────────────────────────────

class TestCmdStatus:
    @pytest.mark.asyncio
    async def test_status_shows_config(self):
        cfg = _make_config(
            llm_model="tiny-llama",
            context_length=4096,
            max_tokens=2048,
            temperature=0.5,
            thinking_mode=True,
            history_limit=10,
            inference_timeout=30,
        )
        bot = _build_bot(config=cfg)
        msg = _MockMessage(text="/status", user=_MockUser(ALLOWED_UID))
        ctx = _make_context()
        await bot._cmd_status(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]
        text = msg.reply_text_log[0]
        for val in ["tiny-llama", "4096", "2048", "0.5", "ON", "10", "30"]:
            assert val in text


# ──────────── /model handler ─────────────────────────────────────────

class TestCmdModel:
    @pytest.mark.asyncio
    async def test_model_shows_name(self):
        cfg = _make_config(llm_model="my-model-v2")
        bot = _build_bot(config=cfg)
        msg = _MockMessage(text="/model", user=_MockUser(ALLOWED_UID))
        ctx = _make_context()
        await bot._cmd_model(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]
        assert "my-model-v2" in msg.reply_text_log[0]


# ──────────── /stats handler ─────────────────────────────────────────

class TestCmdStats:
    @pytest.mark.asyncio
    async def test_no_stats_message(self):
        bot = _build_bot()
        msg = _MockMessage(text="/stats", user=_MockUser(ALLOWED_UID))
        ctx = _make_context(user_data={})  # no stats stored
        await bot._cmd_stats(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]
        assert "No inference stats" in msg.reply_text_log[0]

    @pytest.mark.asyncio
    async def test_shows_last_stats(self):
        bot = _build_bot()
        msg = _MockMessage(text="/stats", user=_MockUser(ALLOWED_UID))
        ctx = _make_context(
            user_data={
                "_stats": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                    "tokens_per_second": 5.0,
                }
            }
        )
        await bot._cmd_stats(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]
        text = msg.reply_text_log[0]
        for val in ["10", "20", "30"]:
            assert val in text
        assert "5.0" in text


# ──────────── /history handler ───────────────────────────────────────

class TestCmdHistory:
    @pytest.mark.asyncio
    async def test_no_history_message(self):
        cm = _mock_cm(messages=[])
        bot = _build_bot(cm=cm)
        msg = _MockMessage(text="/history", user=_MockUser(ALLOWED_UID))
        ctx = _make_context()
        await bot._cmd_history(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]
        assert "No conversation history" in msg.reply_text_log[0]

    @pytest.mark.asyncio
    async def test_shows_recent_messages(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        cm = _mock_cm(messages=msgs)
        bot = _build_bot(cm=cm)
        msg = _MockMessage(text="/history", user=_MockUser(ALLOWED_UID))
        ctx = _make_context()
        await bot._cmd_history(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]
        text = msg.reply_text_log[0]
        assert "Conversation History" in text
        assert "Hello" in text

    @pytest.mark.asyncio
    async def test_truncates_long_replies(self):
        msgs = [{"role": "assistant", "content": "x" * 500}]
        cm = _mock_cm(messages=msgs)
        bot = _build_bot(cm=cm)
        msg = _MockMessage(text="/history", user=_MockUser(ALLOWED_UID))
        ctx = _make_context()
        await bot._cmd_history(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]
        # Each part is truncated to 200 chars; text content in reply should include it


# ──────────── Message handler — happy path ───────────────────────────

class TestHandleMessage:
    @pytest.mark.asyncio
    async def test_reply_with_valid_user(self):
        resp = _fake_response(text="World!", prompt_tokens=4, completion_tokens=2)
        cm = _mock_cm(respond_result=resp)
        bot = _build_bot(cm=cm)
        msg = _MockMessage(text="Hello", user=_MockUser(ALLOWED_UID))
        upd = _MockUpdate(message=msg)
        ctx = _make_context()

        # Patch LlmClient so the async-with block doesn't hit the network.
        from unittest.mock import AsyncMock, patch

        mock_llm = AsyncMock()
        mock_llm.__aenter__ = AsyncMock(return_value=mock_llm)
        mock_llm.__aexit__ = AsyncMock(return_value=None)

        with patch("telegram_bot.LlmClient", return_value=mock_llm):
            await bot._handle_message(upd, ctx)  # type: ignore[arg-type]

        cm.respond.assert_awaited_once_with(
            ALLOWED_UID,
            "Hello",
            mock_llm,
            search_context=None,
            enable_web_search=False,
        )
        assert msg.reply_text_log[0] == "World!"
        # Stats persisted in context.user_data
        stored = _get_user_stats(ctx)
        assert stored.prompt_tokens == 4
        assert stored.completion_tokens == 2

    @pytest.mark.asyncio
    async def test_respects_allow_list(self):
        bot = _build_bot(config=_make_config(telegram_allowed_user_ids=frozenset([ALLOWED_UID])))
        msg = _MockMessage(text="Hi", user=_MockUser(999))  # not allowed
        ctx = _make_context()
        await bot._handle_message(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]
        assert "not allowed" in msg.reply_text_log[0].lower()


# ──────────── Message handler — error paths ──────────────────────────

class TestHandleMessageErrors:
    @pytest.mark.asyncio
    async def test_inference_error(self):
        cm = _mock_cm(respond_raises=InferenceError("timeout"))
        bot = _build_bot(cm=cm)
        msg = _MockMessage(text="Hi", user=_MockUser(ALLOWED_UID))
        ctx = _make_context()
        await bot._handle_message(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]
        assert "Inference failed" in msg.reply_text_log[0]

    @pytest.mark.asyncio
    async def test_value_error(self):
        cm = _mock_cm(respond_raises=ValueError("bad input"))
        bot = _build_bot(cm=cm)
        msg = _MockMessage(text="Hi", user=_MockUser(ALLOWED_UID))
        ctx = _make_context()
        await bot._handle_message(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]
        assert "bad input" in msg.reply_text_log[0]

    @pytest.mark.asyncio
    async def test_unexpected_error(self):
        cm = _mock_cm(respond_raises=RuntimeError("boom"))
        bot = _build_bot(cm=cm)
        msg = _MockMessage(text="Hi", user=_MockUser(ALLOWED_UID))
        ctx = _make_context()
        await bot._handle_message(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]
        assert "Unexpected error" in msg.reply_text_log[0]


# ──────────── build_application factory ───────────────────────────────

class TestBuildApplication:
    def test_returns_application(self):
        bot = _build_bot()
        app = bot.build_application()
        assert hasattr(app, "add_handler")  # basic Application attribute check

    def test_registers_command_handlers(self):
        """Verify all command and message handlers are registered."""
        bot = _build_bot()
        app = bot.build_application()
        # app.handlers is a dict {priority: [handlers]}
        flat = [h for group in app.handlers.values() for h in group]
        registered = {h.callback.__name__ for h in flat if hasattr(h, "callback")}
        expected = {"_cmd_start", "_cmd_help", "_cmd_new", "_cmd_status", "_cmd_model", "_cmd_stats", "_cmd_search", "_cmd_history", "_handle_message"}
        assert expected.issubset(registered)

    def test_clear_alias_registered(self):
        """Ensure /clear is wired as a CommandHandler that calls _cmd_new."""
        bot = _build_bot()
        app = bot.build_application()
        flat = [h for group in app.handlers.values() for h in group]
        # CommandHandler has .commands (set), not .command
        all_commands = {cmd for h in flat if hasattr(h, 'commands') for cmd in h.commands}
        assert 'clear' in all_commands


# ──────────── /search command ─────────────────────────────────────────


class TestSearchCommand:
    @pytest.mark.asyncio
    async def test_search_disabled_message(self):
        cfg = _make_config(search_enabled=False)
        bot = _build_bot(config=cfg)
        msg = _MockMessage(text="", user=_MockUser(ALLOWED_UID))
        ctx = _make_context()
        ctx.args = []
        await bot._cmd_search(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]
        assert "not enabled" in str(msg.reply_text_log[0]).lower()

    @pytest.mark.asyncio
    async def test_search_status_off(self):
        cfg = _make_config(search_enabled=True)
        bot = _build_bot(config=cfg)
        msg = _MockMessage(text="", user=_MockUser(ALLOWED_UID))
        ctx = _make_context(user_data={"search_enabled": False})
        ctx.args = []
        await bot._cmd_search(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]
        assert "OFF" in str(msg.reply_text_log[0])

    @pytest.mark.asyncio
    async def test_search_on(self):
        cfg = _make_config(search_enabled=True)
        bot = _build_bot(config=cfg)
        msg = _MockMessage(text="", user=_MockUser(ALLOWED_UID))
        ctx = _make_context(user_data={})
        ctx.args = ["on"]
        await bot._cmd_search(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]
        assert "ON" in str(msg.reply_text_log[0])
        assert ctx.user_data.get("search_enabled") is True

    @pytest.mark.asyncio
    async def test_search_off(self):
        cfg = _make_config(search_enabled=True)
        bot = _build_bot(config=cfg)
        msg = _MockMessage(text="", user=_MockUser(ALLOWED_UID))
        ctx = _make_context(user_data={"search_enabled": True})
        ctx.args = ["off"]
        await bot._cmd_search(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]
        assert "OFF" in str(msg.reply_text_log[0])
        assert ctx.user_data.get("search_enabled") is False

    @pytest.mark.asyncio
    async def test_search_unknown_action(self):
        cfg = _make_config(search_enabled=True)
        bot = _build_bot(config=cfg)
        msg = _MockMessage(text="", user=_MockUser(ALLOWED_UID))
        ctx = _make_context(user_data={})
        ctx.args = ["maybe"]
        await bot._cmd_search(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]
        assert "Unknown action" in str(msg.reply_text_log[0])

    @pytest.mark.asyncio
    async def test_search_query_runs_search_and_reply(self, monkeypatch):
        """``/search query <q>`` runs Brave, then LLM, and sends the reply."""
        from web_search import SearchResult

        cfg = _make_config(search_enabled=True, brave_api_key="TESTKEY")
        cm = _mock_cm(respond_result=_fake_response(text="Paris, the capital!"))
        bot = _build_bot(config=cfg, cm=cm)
        msg = _MockMessage(text="", user=_MockUser(ALLOWED_UID))
        ctx = _make_context(user_data={})
        ctx.args = ["query", "what is the capital of France?"]

        class _FakeClient:
            called: dict[str, Any] = {}

            def __init__(self, api_key, *a, **k):
                self._api_key = api_key

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def search(self, query):
                type(self).called["query"] = query
                return [SearchResult("France", "https://example.com/france",
                                     "Paris is the capital of France.")]

            async def format_context(self, results):
                return results[0].snippet

        monkeypatch.setattr("telegram_bot.WebSearchClient", _FakeClient)

        await bot._cmd_search(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]

        # Brave was invoked with the exact query text.
        assert _FakeClient.called["query"] == "what is the capital of France?"
        # The LLM was called with the query + injected search context.
        assert cm.respond.await_count == 1
        call_args, call_kwargs = cm.respond.await_args
        assert call_args[1] == "what is the capital of France?"
        assert call_kwargs["search_context"] == "Paris is the capital of France."
        # The summarized answer was sent back to the user.
        assert "Paris, the capital!" in str(msg.reply_text_log[0])
        # One-off search must NOT flip the auto-search toggle.
        assert ctx.user_data.get("search_enabled") is not True

    @pytest.mark.asyncio
    async def test_search_query_no_results(self, monkeypatch):
        """Empty results -> friendly message, no LLM call."""
        cfg = _make_config(search_enabled=True, brave_api_key="TESTKEY")
        cm = _mock_cm()
        bot = _build_bot(config=cfg, cm=cm)
        msg = _MockMessage(text="", user=_MockUser(ALLOWED_UID))
        ctx = _make_context(user_data={})
        ctx.args = ["query", "zzz no results here"]

        class _FakeClient:
            def __init__(self, api_key, *a, **k):
                self._api_key = api_key

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def search(self, query):
                return []

            async def format_context(self, results):
                return ""

        monkeypatch.setattr("telegram_bot.WebSearchClient", _FakeClient)

        await bot._cmd_search(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]

        assert "No web results" in str(msg.reply_text_log[0])
        # LLM must not be consulted when there is nothing to inject.
        assert cm.respond.await_count == 0

    @pytest.mark.asyncio
    async def test_search_query_missing_text(self, monkeypatch):
        """``/search query`` with no text shows usage, no search is run."""
        cfg = _make_config(search_enabled=True, brave_api_key="TESTKEY")
        cm = _mock_cm()
        bot = _build_bot(config=cfg, cm=cm)
        msg = _MockMessage(text="", user=_MockUser(ALLOWED_UID))
        ctx = _make_context(user_data={})
        ctx.args = ["query"]

        called = {"client": False}

        class _FakeClient:
            def __init__(self, api_key, *a, **k):
                called["client"] = True
                self._api_key = api_key

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        monkeypatch.setattr("telegram_bot.WebSearchClient", _FakeClient)

        await bot._cmd_search(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]

        assert "Usage" in str(msg.reply_text_log[0])
        assert called["client"] is False
        assert cm.respond.await_count == 0


    @pytest.mark.asyncio
    async def test_search_query_web_error(self, monkeypatch):
        """A WebSearchError is swallowed and reported to the user."""
        from web_search import WebSearchError

        cfg = _make_config(search_enabled=True, brave_api_key="TESTKEY")
        cm = _mock_cm()
        bot = _build_bot(config=cfg, cm=cm)
        msg = _MockMessage(text="", user=_MockUser(ALLOWED_UID))
        ctx = _make_context(user_data={})
        ctx.args = ["query", "boom"]

        class _FakeClient:
            def __init__(self, api_key, *a, **k):
                self._api_key = api_key

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def search(self, query):
                raise WebSearchError(
                    "Brave Search API returned HTTP 429 (rate limit / monthly quota exceeded — search disabled)",
                    status_code=429,
                )

            async def format_context(self, results):
                return ""

        monkeypatch.setattr("telegram_bot.WebSearchClient", _FakeClient)

        await bot._cmd_search(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]

        assert "No web results" in str(msg.reply_text_log[0])
        assert cm.respond.await_count == 0

    @pytest.mark.asyncio
    async def test_search_query_disabled_flag(self):
        """``/search query`` with SEARCH_ENABLED=false shows the disabled msg."""
        cfg = _make_config(search_enabled=False, brave_api_key="TESTKEY")
        cm = _mock_cm()
        bot = _build_bot(config=cfg, cm=cm)
        msg = _MockMessage(text="", user=_MockUser(ALLOWED_UID))
        ctx = _make_context(user_data={})
        ctx.args = ["query", "hello"]

        await bot._cmd_search(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]

        assert "not enabled" in str(msg.reply_text_log[0]).lower()


# ──────────── TelegramBot integration ─────────────────────────────────

class TestTelegramBotIntegration:
    @pytest.mark.asyncio
    async def test_bot_creates_cm_when_none(self):
        cfg = _make_config()
        bot = TelegramBot(cfg)  # no injected CM
        assert bot._cm is not None

    @pytest.mark.asyncio
    async def test_bot_uses_injected_cm(self):
        cm = _mock_cm()
        cfg = _make_config()
        bot = TelegramBot(cfg, conversation_manager=cm)
        assert bot._cm is cm

    @pytest.mark.asyncio
    async def test_bot_wires_brave_key_into_default_cm(self):
        # When no CM is injected, the default one must receive the Brave key
        # from Config so that the web-search tool (``/search on``) actually
        # works instead of raising ``ValueError: BRAVE_API_KEY is required``.
        cfg = _make_config(brave_api_key="brave-key-123")
        bot = TelegramBot(cfg)  # no injected CM
        assert bot._cm is not None
        assert bot._cm._brave_api_key == "brave-key-123"


# ──────────── CLI entry-point regression tests ────────────────────────

import unittest.mock as mock


class TestMainEntryPoint:
    def test_main_missing_token_exits(self):
        """Without TELEGRAM_BOT_TOKEN, main() exits cleanly with code 1."""
        env = {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_ALLOWED_USER_IDS": ""}
        with mock.patch.dict("os.environ", env, clear=False):
            with pytest.raises(SystemExit) as exc_info:
                _main_entry_point()
        assert exc_info.value.code == 1

    def test_main_builds_and_runs_bot(self, monkeypatch):
        """With a token, main() builds an app and starts the bot."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:ABC")
        monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "42")

        app = mock.MagicMock()
        with mock.patch.object(
            TelegramBot, "build_application", return_value=app
        ):
            _main_entry_point()

        app.run_polling.assert_called_once()

    def test_main_empty_allowlist_warns(self, monkeypatch, caplog):
        """An empty allowlist produces a warning, not a crash."""
        import logging

        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:ABC")
        monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "")

        app = mock.MagicMock()
        with mock.patch.object(
            TelegramBot, "build_application", return_value=app
        ):
            with caplog.at_level(logging.WARNING):
                _main_entry_point()

        assert any(
            "TELEGRAM_ALLOWED_USER_IDS" in r.message for r in caplog.records
        )



# ──────────── respond() web-search wiring ───────────────────────────


class TestHandleMessageSearchWiring:
    @pytest.mark.asyncio
    async def test_enable_web_search_true_when_toggle_on(self):
        """When /search on set context.user_data['search_enabled']=True,
        respond() must be called with enable_web_search=True."""
        resp = _fake_response(text="hi")
        cm = _mock_cm(respond_result=resp)
        bot = _build_bot(cm=cm)
        msg = _MockMessage(text="hello", user=_MockUser(ALLOWED_UID))
        upd = _MockUpdate(message=msg)
        ctx = _make_context(user_data={"search_enabled": True})

        from unittest.mock import AsyncMock, patch

        mock_llm = AsyncMock()
        mock_llm.__aenter__ = AsyncMock(return_value=mock_llm)
        mock_llm.__aexit__ = AsyncMock(return_value=None)

        with patch("telegram_bot.LlmClient", return_value=mock_llm):
            await bot._handle_message(upd, ctx)  # type: ignore[arg-type]

        _, kwargs = cm.respond.await_args
        assert kwargs["enable_web_search"] is True

    @pytest.mark.asyncio
    async def test_enable_web_search_false_when_toggle_off(self):
        """When /search off, respond() must be called with enable_web_search=False."""
        resp = _fake_response(text="hi")
        cm = _mock_cm(respond_result=resp)
        bot = _build_bot(cm=cm)
        msg = _MockMessage(text="hello", user=_MockUser(ALLOWED_UID))
        upd = _MockUpdate(message=msg)
        ctx = _make_context(user_data={"search_enabled": False})

        from unittest.mock import AsyncMock, patch

        mock_llm = AsyncMock()
        mock_llm.__aenter__ = AsyncMock(return_value=mock_llm)
        mock_llm.__aexit__ = AsyncMock(return_value=None)

        with patch("telegram_bot.LlmClient", return_value=mock_llm):
            await bot._handle_message(upd, ctx)  # type: ignore[arg-type]

        _, kwargs = cm.respond.await_args
        assert kwargs["enable_web_search"] is False

    @pytest.mark.asyncio
    async def test_search_query_passes_context_and_no_tool(self, monkeypatch):
        """One-off /search query path injects context, never toggles state."""
        from web_search import SearchResult

        cfg = _make_config(search_enabled=True, brave_api_key="TESTKEY")
        cm = _mock_cm(respond_result=_fake_response(text="Paris"))
        bot = _build_bot(config=cfg, cm=cm)
        msg = _MockMessage(text="", user=_MockUser(ALLOWED_UID))
        ctx = _make_context(user_data={})
        ctx.args = ["query", "capital of France"]

        class _FakeClient:
            def __init__(self, api_key, *a, **k):
                self._api_key = api_key

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def search(self, query):
                return [SearchResult("France", "https://e.com/fr",
                                     "Paris is the capital.")]

            async def format_context(self, results):
                return results[0].snippet

        monkeypatch.setattr("telegram_bot.WebSearchClient", _FakeClient)

        await bot._cmd_search(_MockUpdate(message=msg), ctx)  # type: ignore[arg-type]

        _, kwargs = cm.respond.await_args
        assert kwargs["search_context"] == "Paris is the capital."
        assert kwargs["enable_web_search"] is False
        # State untouched.
        assert ctx.user_data.get("search_enabled") is not True
