"""Telegram bot front-end for the local LLM chatbot.

Wires ``ConversationManager`` and ``LlmClient`` (backed by a
:class:`config.Config`) to a ``python-telegram-bot`` v22 polling
application.  Supports the following slash commands:

* ``/start`` — greet the user, confirm identity
* ``/help``  — list available commands
* ``/new``   — clear the current conversation history
* ``/status`` — show model, context length, max tokens, history limit
* ``/model`` — show the currently configured LLM model
* ``/stats`` — show token usage from the last inference (if any)
* ``/search on|off`` — toggle web search for your user account
* ``/search`` — check current search status

The bot checks an allow-list of user IDs (from ``Config``) before
processing any message.  Non-allowed users receive a silent refusal.

Architecture
------------
Handlers live in :class:`TelegramBot` and are kept separate from
:start_polling: / :meth:`TelegramBot.run()` so that handlers can be
tested in isolation without starting the polling loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from conversation import ConversationManager, ConversationResponse, build_system_prompt
from llm_client import InferenceError, LlmClient
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from web_search import WebSearchClient, WebSearchError
from config import Config, load_config

logger = logging.getLogger(__name__)

# (Logging is configured in ``main()`` only; no import-time ``basicConfig`` so
# importing this module — e.g. from tests — does not touch global logging.)


# ── Per-user state helpers via context user_data ───────────────────────


@dataclass(frozen=True)
class UserStats:
    """Stored inference statistics for a single user."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float | None = None


def _get_user_stats(context: ContextTypes.DEFAULT_TYPE) -> UserStats:
    """Return the UserStats for this user, or a fresh one."""
    raw = context.user_data.get("_stats")
    if raw is None:
        return UserStats()
    return UserStats(**raw)


def _set_user_stats(context: ContextTypes.DEFAULT_TYPE, stats: UserStats) -> None:
    """Persist UserStats into context.user_data for this user."""
    context.user_data["_stats"] = {
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.completion_tokens,
        "total_tokens": stats.total_tokens,
        "tokens_per_second": stats.tokens_per_second,
    }


def _friendly_error_message(exc: InferenceError) -> str:
    """Return a user-friendly reply for an :class:`InferenceError`.

    Detects the "backend is not serving" state — HTTP 404 (endpoint not
    found) or 502/503 (service unavailable) on the ``/chat/completions``
    endpoint — which is what happens when the local Ollama server is down
    (e.g. stopped to free up VRAM).  In that case we return a friendly
    message instead of leaking a raw HTTP error to the user.  Any other
    failure falls back to the generic ``Inference failed`` reply.
    """
    status = getattr(exc, "status_code", None)
    if status in (404, 502, 503):
        return (
            "🫙 Ollama is currently down (likely stopped to free up VRAM). "
            "Restart it and I'll run inference again. 🙂"
        )
    return f"❌ Inference failed: {exc}"


# ── TelegramBot ────────────────────────────────────────────────────────


class TelegramBot:
    """High-level bot that connects :class:`Config` → ``LlmClient`` → ``ConversationManager``.

    Parameters
    ----------
    config:
        Runtime configuration (bot token, LLM endpoint, allowed users, …).
    conversation_manager:
        A :class:`ConversationManager` instance.  Created internally when
        *None* so that callers can supply a mocked one for testing.
    """

    def __init__(
        self,
        config: Config,
        *,
        conversation_manager: ConversationManager | None = None,
    ) -> None:
        self._config = config
        self._cm = conversation_manager or ConversationManager(
            context_length=config.context_length,
            max_tokens=config.max_tokens,
            history_limit=config.history_limit,
            brave_api_key=config.brave_api_key,
        )
        # Per-user web-search tool toggle (Off / On).  Default is Off so the
        # web-search tool is never sent unless the user opts in via
        # ``/search on``.  The ``/search query`` one-off path does not touch
        # this state.  Toggle state is stored in ``context.user_data["search_enabled"]``.

    # ── Commands ────────────────────────────────────────────────────────

    async def _cmd_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Greet the user and confirm identity."""
        welcome = (
            f"🤖 *Bot started!*\n\n"
            f"I'm running locally with model "
            f"``{self._config.llm_model}``.\n\n"
            f"Available commands:\n"
            f"``/help`` — list commands\n"
            f"``/new`` — clear conversation\n"
            f"``/status`` — show configuration\n"
            f"``/model`` — current model\n"
            f"``/stats`` — last inference stats\n"
            f"``/history`` — recent messages\n"
            f"``/search on|off`` — toggle web search\n"
            f"``/search query <question>`` — one-off web search"
        )
        await update.message.reply_text(welcome, parse_mode="Markdown")  # type: ignore[attr-defined]

    async def _cmd_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """List available commands."""
        help_text = (
            "``/start`` — Start a conversation\n"
            "``/help`` — Show this help message\n"
            "``/new`` — Clear the conversation history\n"
            "``/status`` — Show current settings\n"
            "``/model`` — Show the LLM model name\n"
            "``/stats`` — Show last inference statistics\n"
            "``/history`` — Show recent messages in your conversation\n"
            "``/search on|off`` — Toggle web search (needs a Brave API key)\n"
            "``/search query <your question>`` — One-off web search + LLM summary"
        )
        await update.message.reply_text(help_text, parse_mode="Markdown")  # type: ignore[attr-defined]

    async def _cmd_new(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Clear the current user's conversation history."""
        user_id = update.effective_user.id  # type: ignore[union-attr]
        await self._cm.clear_user(user_id)
        context.user_data.pop("_stats", None)
        await update.message.reply_text("✅ Conversation history cleared.")  # type: ignore[attr-defined]

    async def _cmd_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show current configuration."""
        status = (
            f"*Current Settings*\n\n"
            f"Model: ``{self._config.llm_model}``\n"
            f"Context length: {self._config.context_length}\n"
            f"Max tokens: {self._config.max_tokens}\n"
            f"Temperature: {self._config.temperature}\n"
            f"Thinking mode: {'ON' if self._config.thinking_mode else 'OFF'}\n"
            f"History limit: {self._config.history_limit} turns\n"
            f"Inference timeout: {self._config.inference_timeout}s"
        )
        await update.message.reply_text(status, parse_mode="Markdown")  # type: ignore[attr-defined]

    async def _cmd_model(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show the current model name."""
        await update.message.reply_text(  # type: ignore[attr-defined]
            f"Model: ``{self._config.llm_model}``",
            parse_mode="Markdown",
        )

    async def _cmd_stats(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show inference stats from the last response."""
        stats = _get_user_stats(context)
        if stats.total_tokens == 0:
            await update.message.reply_text(  # type: ignore[attr-defined]
                "No inference stats yet — send a message first!"
            )
            return
        summary = (
            f"*Last Inference Stats*\n\n"
            f"Prompt tokens: {stats.prompt_tokens}\n"
            f"Completion tokens: {stats.completion_tokens}\n"
            f"Total tokens: {stats.total_tokens}\n"
        )
        if stats.tokens_per_second is not None:
            summary += f"\nSpeed: {stats.tokens_per_second:.1f} tok/s"
        await update.message.reply_text(summary, parse_mode="Markdown")  # type: ignore[attr-defined]

    async def _run_web_search(
        self, query: str, user_id: int
    ) -> str | None:
        """Run a Brave web search and return formatted context, or ``None``.

        Returns an empty-context ``None`` when no results are found or when
        the search fails; callers treat both as "no search context" and fall
        back to normal inference.
        """
        try:
            async with WebSearchClient(
                self._config.brave_api_key or "",
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
        except WebSearchError as exc:
            logger.warning("Web search failed for user %d: %s", user_id, exc)
        return None

    async def _cmd_search(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Toggle, check status, or run a one-off web search.

        Forms:
          ``/search``            – show current ON/OFF status
          ``/search on``/``off`` – toggle the web-search *tool* for future
                                   messages (enables/disables model tool-calling)
          ``/search <query>``    – run a one-off search and LLM-summarize it
                                   (does not change the ON/OFF toggle)
        """
        if not self._config.search_enabled:
            await update.message.reply_text(  # type: ignore[attr-defined]
                "Web search is not enabled on this bot. "
                "Ask the admin to set ``SEARCH_ENABLED=true`` and provide "
                "a Brave Search API key (``BRAVE_API_KEY``)."
            )
            return

        args = (context.args or []) if context else []
        if not args:
            # Show current status.
            enabled = bool(context.user_data.get("search_enabled", False))  # type: ignore[union-attr]
            state = "ON" if enabled else "OFF"
            await update.message.reply_text(  # type: ignore[attr-defined]
                f"Web search *tool* is currently *{state}* for you. "
                "When *ON*, the model decides on its own whether to search "
                "for current information. Use ``/search on`` or ``/search off`` "
                "to toggle, or ``/search <query>`` for a one-off search."
            )
            return

        action = args[0].lower()
        if action == "on":
            context.user_data["search_enabled"] = True  # type: ignore[index]
            await update.message.reply_text(  # type: ignore[attr-defined]
                "Web search *tool* is now *ON* for you. The model will call "
                "the web search tool on its own when it needs current "
                "information (news, scores, prices, weather, etc.). "
                "Use ``/search off`` to disable."
            )
        elif action == "off":
            context.user_data["search_enabled"] = False  # type: ignore[index]
            await update.message.reply_text(  # type: ignore[attr-defined]
                "Web search *tool* is now *OFF* for you. The model will no "
                "longer search the web; it answers from knowledge only. "
                "Use ``/search on`` to re-enable, or ``/search <query>`` for "
                "a one-off search."
            )
        elif action == "query":
            # ``/search query <text>`` – one-off search + LLM summary.
            query = " ".join(args[1:]).strip()
            if not query:
                await update.message.reply_text(  # type: ignore[attr-defined]
                    "Usage: ``/search query <your question>``"
                )
                return
            await self._search_and_reply(update, context, query)
        else:
            await update.message.reply_text(  # type: ignore[attr-defined]
                f"Unknown action: ``{action}``. Use ``/search on``, "
                "``/search off``, ``/search`` (status), or "
                "``/search query <your question>`` for a one-off search."
            )

    async def _search_and_reply(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, query: str
    ) -> None:
        """Run a one-off web search and reply with an LLM-summarized answer.

        Injects the search results as context (identical to the auto-search
        path) but does **not** modify the ``/search on|off`` toggle.
        """
        user_id = update.effective_user.id  # type: ignore[union-attr]
        search_context = await self._run_web_search(query, user_id)
        if not search_context:
            await update.message.reply_text(  # type: ignore[attr-defined]
                "❌ No web results for that query."
            )
            return
        try:
            async with LlmClient(self._config) as client:
                resp = await self._cm.respond(
                    user_id,
                    query,
                    client,
                    search_context=search_context,
                    enable_web_search=False,
                )
            await update.message.reply_text(resp.text[:4096])  # type: ignore[attr-defined]
        except InferenceError as exc:
            logger.error("InferenceError for user %d: %s", user_id, exc)
            await update.message.reply_text(  # type: ignore[attr-defined]
                _friendly_error_message(exc)
            )
        except Exception as exc:
            logger.exception("Unhandled error for user %d", user_id)
            await update.message.reply_text(  # type: ignore[attr-defined]
                f"❌ Unexpected error: {exc}"
            )

    async def _cmd_history(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show recent conversation messages (last 10 pairs)."""
        user_id = update.effective_user.id  # type: ignore[union-attr]
        messages = await self._cm.get_messages(user_id)
        if not messages:
            await update.message.reply_text(  # type: ignore[attr-defined]
                "No conversation history yet. Say something!"
            )
            return
        # Show last 20 messages (10 user/assistant pairs).
        recent = messages[-20:] if len(messages) > 20 else messages
        parts: list[str] = []
        for m in recent:
            role = m.get("role", "?").capitalize()
            content = str(m.get("content", ""))[:200]  # truncate long replies
            parts.append(f"*{role}:*\n```\n{content}\n```")
        history_text = "\n\n".join(parts)
        await update.message.reply_text(  # type: ignore[attr-defined]
            f"*Conversation History* (last {len(recent)} messages):\n\n"
            f"{history_text[:4000]}",  # Telegram message length cap
            parse_mode="Markdown",
        )

    # ── Message handler ────────────────────────────────────────────────

    async def _handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Process a normal (non-command) user message."""
        if not update.effective_user:
            logger.warning("No effective user found")
            return
        user_id = update.effective_user.id  # type: ignore[union-attr]
        logger.info("Message from user %d", user_id)

        # --- allow-list check ---
        if self._config.telegram_allowed_user_ids and not self._config.is_user_allowed(user_id):
            logger.warning("User %d not allowed", user_id)
            await update.message.reply_text(  # type: ignore[attr-defined]
                "❌ You are not allowed to use this bot."
            )
            return

        text = update.message.text.strip() if update.message.text else ""  # type: ignore[union-attr]
        if not text:
            logger.warning("Empty message text from user %d", user_id)
            return

        logger.info(f"Starting inference for user {user_id}")

        # --- web-search tool toggle (per user) ---
        # /search on enables the web-search *tool* so the model may call it
        # on its own; /search off (default) sends no tool at all.
        web_search_on = bool(context.user_data.get("search_enabled", False))  # type: ignore[union-attr]

        # --- reply ---
        try:
            async with LlmClient(self._config) as client:
                resp = await self._cm.respond(
                    user_id,
                    text,
                    client,
                    search_context=None,
                    enable_web_search=web_search_on,
                )

            # Store stats for /stats command
            if resp.stats is not None:
                _set_user_stats(context, UserStats(
                    prompt_tokens=resp.stats.prompt_tokens,
                    completion_tokens=resp.stats.completion_tokens,
                    total_tokens=resp.stats.total_tokens,
                    tokens_per_second=resp.stats.tokens_per_second,
                ))

            # Send the reply (Telegram truncates at ~4096 chars)
            await update.message.reply_text(resp.text[:4096])  # type: ignore[attr-defined]

        except InferenceError as exc:
            error_msg = _friendly_error_message(exc)
            logger.error("InferenceError for user %d: %s", user_id, exc)
            await update.message.reply_text(error_msg)  # type: ignore[attr-defined]
        except ValueError as exc:
            await update.message.reply_text(f"❌ {exc}")  # type: ignore[attr-defined]
        except Exception as exc:
            logger.exception("Unhandled error for user %d", user_id)
            await update.message.reply_text(f"❌ Unexpected error: {exc}")  # type: ignore[attr-defined]

    # ── Application factory ────────────────────────────────────────────

    def build_application(self) -> Application:
        """Create a :class:`telegram.ext.Application` with all handlers wired.

        Returns the fully-built application ready for ``.run()`` or testing.
        """
        app = (
            ApplicationBuilder()
            .token(self._config.telegram_bot_token)
            .build()
        )

        # Command handlers
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("help", self._cmd_help))
        app.add_handler(CommandHandler("new", self._cmd_new))
        app.add_handler(CommandHandler("clear", self._cmd_new))  # alias
        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(CommandHandler("model", self._cmd_model))
        app.add_handler(CommandHandler("stats", self._cmd_stats))
        app.add_handler(CommandHandler("search", self._cmd_search))
        app.add_handler(CommandHandler("history", self._cmd_history))

        # Fallback message handler (non-command text)
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        return app

    def run(self, *, poll_interval: float = 0.5) -> None:
        """Start the bot in a blocking call (starts and stops polling).

        This is a convenience wrapper around ``build_application().run()``.
        For testing or custom lifecycle management use
        :meth:`build_application` directly.

        Parameters
        ----------
        poll_interval:
            Seconds to wait between polling the Telegram API.
        """
        app = self.build_application()
        app.run_polling(poll_interval=poll_interval)


# ── CLI entry point ────────────────────────────────────────────────────


def main() -> None:
    """Run the Telegram bot (CLI entry point)."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    )

    try:
        config = load_config()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not config.telegram_allowed_user_ids:
        logging.warning(
            "TELEGRAM_ALLOWED_USER_IDS is empty — the bot will reject all users. "
            "Set it to your Telegram user ID(s)."
        )

    bot = TelegramBot(config)
    bot.run()


if __name__ == "__main__":
    main()
