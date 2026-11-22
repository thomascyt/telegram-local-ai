"""Asynchronous LLM inference client for OpenAI-compatible endpoints.

Sends chat-completion requests to a local LLM server (e.g. Ollama) exposed
as an OpenAI-compatible ``/chat/completions`` endpoint, and returns
structured results including the assistant text, token counts, and
generation speed.
"""

from __future__ import annotations

import json as _json
import logging
from dataclasses import dataclass, field as _dataclass_field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ── Return types ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class InferenceResult:
    """Result from an LLM inference call."""
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float | None = None
    data: Any = None  # raw API response dict (used downstream to extract tool calls)


@dataclass(frozen=True)
class ToolCall:
    """A single ``tool_calls[i]`` entry from a model response.

    Mirrors the OpenAI chat-completion ``tool_call`` shape:
    ``{"id", "type": "function", "function": {"name", "arguments"}}``.

    ``arguments`` is the raw JSON string the model produced; callers parse
    it with :func:`parse_tool_arguments` (which tolerates malformed JSON).
    """
    id: str
    name: str
    arguments: str  # raw JSON string, e.g. ``{"query": "..."}``


@dataclass(frozen=True)
class InferenceStats:
    """Detailed token and timing statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float | None = None


class InferenceError(Exception):
    """Raised when an inference call fails."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code  # type: ignore[assignment]


# ── Helpers ──────────────────────────────────────────────────────────

# The client talks to a single OpenAI-compatible backend (Ollama), so no
# backend-specific request shaping is needed — the request body is a plain
# OpenAI chat-completion payload.


def parse_tool_arguments(raw: str | None) -> dict[str, Any]:
    """Parse a model-produced tool-call arguments JSON string.

    Tolerant of the messy reality of local models: malformed JSON,
    trailing prose after the JSON object, and ``None`` all degrade
    gracefully to an empty dict instead of raising.

    Parameters
    ----------
    raw:
        The raw ``arguments`` string from a ``tool_call`` (e.g.
        ``'{"query": "latest news"}'``). May be ``None`` or contain
        non-JSON prose.

    Returns
    -------
    dict[str, Any]
        The parsed arguments, or ``{}`` when nothing parseable is found.
    """
    if not raw or not isinstance(raw, str):
        return {}
    stripped = raw.strip()
    # Fast path: valid JSON.
    try:
        parsed = _json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}
    except _json.JSONDecodeError:
        pass
    # Slow path: try to salvage a JSON object embedded in prose.
    start = stripped.find("{")
    if start == -1:
        return {}
    depth = 0
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = stripped[start : i + 1]
                try:
                    parsed = _json.loads(chunk)
                    return parsed if isinstance(parsed, dict) else {}
                except _json.JSONDecodeError:
                    return {}
    return {}


def _extract_tool_calls(data: dict[str, Any]) -> list[ToolCall]:
    """Return tool calls declared on ``choices[0].message.tool_calls``.

    Returns an empty list when the model produced a plain text reply, when
    the response shape is unexpected, or when any tool_call is malformed.
    """
    if not isinstance(data, dict):
        return []
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return []
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []

    calls: list[ToolCall] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        function = tc.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        raw_args = function.get("arguments")
        calls.append(
            ToolCall(
                id=str(tc.get("id") or ""),
                name=name,
                arguments=raw_args if isinstance(raw_args, str) else "",
            )
        )
    return calls


def _build_chat_request(
    messages: list[dict[str, str]],
    model: str,
    max_tokens: int,
    temperature: float,
    *,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a plain OpenAI-compatible chat-completion request.

    The body is intentionally minimal (``model``, ``messages``,
    ``max_tokens``, ``temperature``, and optionally ``tools``).  Ollama — the
    sole backend — handles reasoning/thinking at the model level, so no
    backend-specific kwargs (such as Strata's ``chat_template_kwargs``) are
    sent.

    ``tools`` is an optional list of OpenAI-style tool/function definitions.
    When provided, the model is allowed to return ``tool_calls`` on the
    assistant message, enabling function-calling workflows (e.g. web search).
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    # Only send ``tools`` when at least one is defined; an empty list would
    # otherwise bloat every request and confuse some inference engines.
    if tools:
        body["tools"] = tools
    return body


# ── Response validation helpers ──────────────────────────────────────


def _validate_top_level(data: dict[str, Any]) -> None:
    """Raise *InferenceError* when the top-level structure is broken."""
    if not isinstance(data, dict):
        raise InferenceError(
            "API response is not a JSON object", status_code=None
        )
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        raise InferenceError(
            "Empty or missing 'choices' array in API response"
        )


def _validate_first_choice(
    first_choice: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Return ``(content, finish_reason)`` after structural checks."""
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise InferenceError(
            "'choices[0].message' is missing or not an object",
            status_code=None,
        )
    content = message.get("content")
    finish_reason: str | None = first_choice.get("finish_reason")
    return content, finish_reason


# ── Client ───────────────────────────────────────────────────────────


class LlmClient:
    """Asynchronous client for OpenAI-compatible inference endpoints (Ollama)."""

    def __init__(
        self,
        config: "Config",
        base_url: str | None = None,
    ) -> None:
        self._config = config
        url = (base_url or config.llm_base_url).rstrip("/")
        self._endpoint = f"{url}/chat/completions"
        self._client = httpx.AsyncClient(timeout=float(config.inference_timeout))

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> InferenceResult:
        """Send *messages* and return the assistant's final answer.

        Parameters
        ----------
        messages:
            The chat history (system + turns + current user message).
        tools:
            Optional list of OpenAI-style tool definitions.  When provided,
            the model may return ``tool_calls``; the caller is responsible
            for executing them and feeding the results back via a follow-up
            ``complete`` call (see :meth:`ConversationManager.respond`).

        Raises
        ------
        InferenceError
            On HTTP errors, timeouts, connection failures, or
            malformed / empty responses.
        """
        body = _build_chat_request(
            messages=messages,
            model=self._config.llm_model,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            tools=tools,
        )

        logger.debug("Requesting inference from %s", self._endpoint)

        try:
            resp = await self._client.post(self._endpoint, json=body)
        except httpx.TimeoutException:
            raise InferenceError(
                f"Inference request timed out after "
                f"{self._config.inference_timeout}s"
            ) from None
        except httpx.ConnectError as exc:
            raise InferenceError(
                f"Connection failed to {self._endpoint}",
            ) from exc
        except httpx.RequestError as exc:
            raise InferenceError(
                f"Request failed: {exc}"
            ) from exc

        # ── HTTP status check (do not leak response body) ────────
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise InferenceError(
                f"HTTP error {exc.response.status_code}",
                status_code=exc.response.status_code,
            ) from None

        # ── Parse response ────────────────────────────────────────
        try:
            data = resp.json()
        except _json.JSONDecodeError as exc:
            raise InferenceError(
                f"Response is not valid JSON (HTTP {resp.status_code})",
                status_code=resp.status_code,
            ) from exc

        # ── Structural validation ─────────────────────────────────
        try:
            _validate_top_level(data)
        except InferenceError:
            raise

        choices = data["choices"]  # validated non-empty list above
        first_choice = choices[0]

        content, finish_reason = _validate_first_choice(first_choice)

        # Distinguish output-budget exhaustion from missing content.
        if finish_reason == "length":
            raise InferenceError(
                "Response was truncated (output budget exhausted)",
                status_code=resp.status_code,
            )

        # ── Token counts ──────────────────────────────────────────
        usage = data.get("usage") if isinstance(data, dict) else None
        if not isinstance(usage, dict):
            usage = {}

        prompt_tokens = usage.get("prompt_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or 0
        total_tokens = usage.get("total_tokens") or 0

        # ── Generation speed ──────────────────────────────────────
        tokens_per_second: float | None = None

        # Primary: top-level timings.  Ollama reports `prediction_per_second`;
        # fall back to the older `predicted_per_second` name for compatibility.
        raw_timings = data.get("timings")
        if isinstance(raw_timings, dict):
            pps = (
                raw_timings.get("prediction_per_second")
                or raw_timings.get("predicted_per_second")
            )
            if isinstance(pps, (int, float)) and pps > 0.0:
                tokens_per_second = float(pps)

        # Fallback: compute from completion_tokens / predicted_time_ms
        # only when timings exist but predicted_per_second is absent.
        if tokens_per_second is None and isinstance(raw_timings, dict):
            predicted_ms = raw_timings.get("predicted_time") or raw_timings.get(
                "predicted_ms"
            )
            if predicted_ms is not None and completion_tokens > 0:
                try:
                    seconds = float(predicted_ms) / 1000.0
                    if seconds > 0.0:
                        tokens_per_second = (
                            float(completion_tokens) / seconds
                        )
                except (TypeError, ValueError):
                    pass

        # ── Tool-calling responses (content may be null when tool_calls are present) ──
        tool_calls = _extract_tool_calls(data)
        if content is None or content == "":
            if tool_calls:
                # Model emitted a tool_calls block with no content — this is the
                # normal first-turn response for tool-calling models.  Attach the
                # raw data so ``_run_with_tools`` can extract and execute the calls.
                result = InferenceResult(
                    text="",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    tokens_per_second=tokens_per_second,
                    data=data,
                )
                return result
            raise InferenceError(
                "Assistant response contains no content",
                status_code=resp.status_code,
            )

        text = str(content)

        logger.info(
            "Inference complete: %d prompt, %d completion tokens",
            prompt_tokens,
            completion_tokens,
        )

        return InferenceResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tokens_per_second=tokens_per_second,
            data=data,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "LlmClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()


# ── Convenience ──────────────────────────────────────────────────────


async def inference(
    messages: list[dict[str, str]],
    config: "Config",
    base_url: str | None = None,
) -> InferenceResult:
    """One-shot helper: create a client, run inference, clean up."""
    async with LlmClient(config, base_url=base_url) as client:
        return await client.complete(messages)
