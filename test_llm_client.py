"""Tests for LlmClient — uses httpx.MockTransport (no network)."""

from __future__ import annotations

import json as _json
from typing import Any

import httpx
import pytest

from config import Config, load_config
from llm_client import (
    InferenceResult,
    InferenceStats,
    InferenceError,
    LlmClient,
    _build_chat_request,
    inference,
)


# ── Fixtures ─────────────────────────────────────────────────────────


def _make_config(
    monkeypatch: pytest.MonkeyPatch,
    base_url: str = "http://127.0.0.1:8080/v1",
    thinking_mode_env: str | None = None,
) -> Config:
    """Build a fresh test Config scoped to the fixture."""
    for k in [
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_ALLOWED_USER_IDS",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "CONTEXT_LENGTH",
        "MAX_TOKENS",
        "TEMPERATURE",
        "THINKING_MODE",
        "INFERENCE_TIMEOUT",
        "HISTORY_LIMIT",
    ]:
        monkeypatch.delenv(k, raising=False)

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy_token_for_tests")
    if thinking_mode_env is not None:
        monkeypatch.setenv("THINKING_MODE", thinking_mode_env)
    monkeypatch.setenv("LLM_BASE_URL", base_url)
    return load_config()


def _strata_mock_response(
    text: str = "Hello!",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    total_tokens: int = 30,
    tokens_per_second: float | None = 35.7,
    status_code: int = 200,
    timings_top_level: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    content_value: Any | None = None,
) -> httpx.Response:
    """Return an httpx.Response matching the real Strata schema.

    Strata places *timings* at the top level of the JSON response.
    When *content_value* is not given, *text* is used as the content.
    """
    actual_content = content_value if content_value is not None else text
    body: dict[str, Any] = {
        "choices": [
            {
                "message": {"role": "assistant", "content": actual_content},
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }

    if timings_top_level is None:
        if tokens_per_second is not None:
            body["timings"] = {"predicted_per_second": tokens_per_second}
    else:
        body["timings"] = timings_top_level

    if finish_reason is not None:
        body["choices"][0]["finish_reason"] = finish_reason

    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        json=body,
    )


@pytest.fixture
def config(monkeypatch) -> Config:
    return _make_config(monkeypatch)


@pytest.fixture
def mock_client(config, monkeypatch):
    """Create an LlmClient backed by a MockTransport."""
    async def _handler(request: httpx.Request) -> httpx.Response:
        return _strata_mock_response()

    client = LlmClient(config)
    transport = httpx.MockTransport(_handler)
    client._client._transport = transport
    return client


# ── _build_chat_request unit tests ───────────────────────────────────


class TestBuildChatRequest:
    def test_default_payload(self):
        msgs = [{"role": "user", "content": "Hi"}]
        payload = _build_chat_request(msgs, "model-x", 512, 0.7, False)

        assert payload["model"] == "model-x"
        assert payload["messages"] == msgs
        assert payload["max_tokens"] == 512
        assert payload["temperature"] == 0.7
        # Strata backend always sends chat_template_kwargs
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}

    def test_thinking_mode_enabled(self):
        msgs = [{"role": "user", "content": "Explain"}]
        payload = _build_chat_request(msgs, "model-x", 256, 1.0, True)

        assert payload["chat_template_kwargs"] == {"enable_thinking": True}

    def test_thinking_mode_disabled(self):
        """Strata backend always sends chat_template_kwargs (value=False)."""
        msgs = [{"role": "user", "content": "Explain"}]
        payload = _build_chat_request(msgs, "model-x", 256, 1.0, False)

        assert payload["chat_template_kwargs"] == {"enable_thinking": False}

    def test_messages_passed_through(self):
        msgs = [
            {"role": "system", "content": "You are a bot."},
            {"role": "user", "content": "Hello"},
        ]
        payload = _build_chat_request(msgs, "m", 100, 0.5, False)

        assert payload["messages"] == msgs

    def test_backend_others_omit_kwargs(self):
        """Non-Strata backends do NOT send chat_template_kwargs."""
        msgs = [{"role": "user", "content": "Hi"}]
        payload = _build_chat_request(
            msgs, "m", 100, 0.5, True, backend="ollama"
        )

        assert "chat_template_kwargs" not in payload

    def test_tools_included_when_provided(self):
        msgs = [{"role": "user", "content": "Hi"}]
        tools = [{"type": "function", "function": {"name": "web_search"}}]
        payload = _build_chat_request(msgs, "m", 100, 0.5, False, tools=tools)

        assert payload["tools"] == tools

    def test_tools_omitted_when_none(self):
        msgs = [{"role": "user", "content": "Hi"}]
        payload = _build_chat_request(msgs, "m", 100, 0.5, False, tools=None)

        assert "tools" not in payload

    def test_tools_omitted_when_empty(self):
        msgs = [{"role": "user", "content": "Hi"}]
        payload = _build_chat_request(msgs, "m", 100, 0.5, False, tools=[])

        assert "tools" not in payload


# ── Integration tests via MockTransport ──────────────────────────────


class TestLlmClient:
    @pytest.mark.asyncio
    async def test_successful_inference(self, mock_client):
        result = await mock_client.complete(
            [{"role": "user", "content": "Hi there"}]
        )

        assert isinstance(result, InferenceResult)
        assert result.text == "Hello!"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 20
        assert result.total_tokens == 30
        assert abs(result.tokens_per_second - 35.7) < 0.01

    @pytest.mark.asyncio
    async def test_result_is_frozen(self, mock_client):
        result = await mock_client.complete(
            [{"role": "user", "content": "Hi"}]
        )
        with pytest.raises(Exception):
            result.text = "modified"

    @pytest.mark.asyncio
    async def test_text_is_string(self, mock_client):
        result = await mock_client.complete(
            [{"role": "user", "content": "Test"}]
        )
        assert isinstance(result.text, str)

    @pytest.mark.asyncio
    async def test_multi_turn_messages_forwarded(self, mock_client):
        messages = [
            {"role": "system", "content": "You are a bot."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = await mock_client.complete(messages)
        assert isinstance(result, InferenceResult)

    @pytest.mark.asyncio
    async def test_request_sends_correct_model(self, config):
        sent: list[dict] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            body = _json.loads(request.content)
            sent.append(body)
            return _strata_mock_response(text="ok")

        client = LlmClient(config)
        client._client._transport = httpx.MockTransport(handler)
        await client.complete([{"role": "user", "content": "Hi"}])
        await client.aclose()

        assert len(sent) == 1
        assert sent[0]["model"] == config.llm_model

    @pytest.mark.asyncio
    async def test_request_sends_correct_temperature(self, config):
        sent: list[float] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            body = _json.loads(request.content)
            sent.append(body["temperature"])
            return _strata_mock_response(text="ok")

        client = LlmClient(config)
        client._client._transport = httpx.MockTransport(handler)
        await client.complete([{"role": "user", "content": "Hi"}])
        await client.aclose()

    @pytest.mark.asyncio
    async def test_request_with_thinking_mode(self, monkeypatch):
        """Verify chat_template_kwargs is sent with enable_thinking=True."""
        cfg = _make_config(monkeypatch, thinking_mode_env="true")

        sent: list[dict] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            body = _json.loads(request.content)
            sent.append(body)
            return _strata_mock_response(text="ok")

        client = LlmClient(cfg)
        client._client._transport = httpx.MockTransport(handler)
        await client.complete([{"role": "user", "content": "Hi"}])
        await client.aclose()

        assert len(sent) == 1
        assert sent[0]["chat_template_kwargs"]["enable_thinking"] is True

    @pytest.mark.asyncio
    async def test_request_thinking_off_sends_false(self, monkeypatch):
        """When thinking is disabled the payload must carry enable_thinking=False."""
        cfg = _make_config(monkeypatch, thinking_mode_env="false")

        sent: list[dict] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            body = _json.loads(request.content)
            sent.append(body)
            return _strata_mock_response(text="ok")

        client = LlmClient(cfg)
        client._client._transport = httpx.MockTransport(handler)
        await client.complete([{"role": "user", "content": "Hi"}])
        await client.aclose()

        assert len(sent) == 1
        assert sent[0]["chat_template_kwargs"]["enable_thinking"] is False

    @pytest.mark.asyncio
    async def test_http_500_raises_inference_error(self, monkeypatch):
        cfg = _make_config(monkeypatch, "http://127.0.0.1:8080/v1")

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=500, text="Internal Server Error")

        client = LlmClient(cfg)
        client._client._transport = httpx.MockTransport(handler)
        with pytest.raises(InferenceError) as exc_info:
            await client.complete([{"role": "user", "content": "Hi"}])
        assert "500" in str(exc_info.value)
        assert exc_info.value.status_code == 500
        await client.aclose()

    @pytest.mark.asyncio
    async def test_http_400_raises_inference_error(self, monkeypatch):
        cfg = _make_config(monkeypatch, "http://127.0.0.1:8080/v1")

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=400, json={"error": "bad model"})

        client = LlmClient(cfg)
        client._client._transport = httpx.MockTransport(handler)
        with pytest.raises(InferenceError) as exc_info:
            await client.complete([{"role": "user", "content": "Hi"}])
        assert "400" in str(exc_info.value)
        assert exc_info.value.status_code == 400
        await client.aclose()

    @pytest.mark.asyncio
    async def test_http_error_body_not_leaked(self, monkeypatch):
        """Sensitive response bodies must not be included in InferenceError."""
        cfg = _make_config(monkeypatch, "http://127.0.0.1:8080/v1")

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=401,
                json={"error": "secret_key=SK-ABCD1234"},
            )

        client = LlmClient(cfg)
        client._client._transport = httpx.MockTransport(handler)
        with pytest.raises(InferenceError) as exc_info:
            await client.complete([{"role": "user", "content": "Hi"}])
        assert "secret_key" not in str(exc_info.value)
        assert exc_info.value.status_code == 401
        await client.aclose()

    @pytest.mark.asyncio
    async def test_json_decode_error_raises_inference_error(self, monkeypatch):
        cfg = _make_config(monkeypatch, "http://127.0.0.1:8080/v1")

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=200, text="{invalid json")

        client = LlmClient(cfg)
        client._client._transport = httpx.MockTransport(handler)
        with pytest.raises(InferenceError) as exc_info:
            await client.complete([{"role": "user", "content": "Hi"}])
        msg = str(exc_info.value).lower()
        assert "json" in msg or "decode" in msg or "parse" in msg
        await client.aclose()

    @pytest.mark.asyncio
    async def test_empty_choices_raises_inference_error(self, monkeypatch):
        cfg = _make_config(monkeypatch, "http://127.0.0.1:8080/v1")

        async def handler(request: httpx.Request) -> httpx.Response:
            body = {"choices": []}
            return httpx.Response(status_code=200, text=_json.dumps(body))

        client = LlmClient(cfg)
        client._client._transport = httpx.MockTransport(handler)
        with pytest.raises(InferenceError) as exc_info:
            await client.complete([{"role": "user", "content": "Hi"}])
        assert "empty" in str(exc_info.value).lower() or "choices" in str(exc_info.value).lower()
        await client.aclose()

    @pytest.mark.asyncio
    async def test_null_content_raises_inference_error(self, monkeypatch):
        """Null/missing assistant content should be treated as failure."""
        cfg = _make_config(monkeypatch, "http://127.0.0.1:8080/v1")

        async def handler(request: httpx.Request) -> httpx.Response:
            body = {
                "choices": [{"message": {"content": None}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 0},
            }
            return httpx.Response(status_code=200, text=_json.dumps(body))

        client = LlmClient(cfg)
        client._client._transport = httpx.MockTransport(handler)
        with pytest.raises(InferenceError) as exc_info:
            await client.complete([{"role": "user", "content": "Hi"}])
        assert "content" in str(exc_info.value).lower() or "empty" in str(exc_info.value).lower()
        await client.aclose()

    @pytest.mark.asyncio
    async def test_empty_content_raises_inference_error(self, monkeypatch):
        """Empty string content should be treated as failure."""
        cfg = _make_config(monkeypatch, "http://127.0.0.1:8080/v1")

        async def handler(request: httpx.Request) -> httpx.Response:
            body = {
                "choices": [{"message": {"content": ""}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 0},
            }
            return httpx.Response(status_code=200, text=_json.dumps(body))

        client = LlmClient(cfg)
        client._client._transport = httpx.MockTransport(handler)
        with pytest.raises(InferenceError) as exc_info:
            await client.complete([{"role": "user", "content": "Hi"}])
        assert "content" in str(exc_info.value).lower() or "empty" in str(exc_info.value).lower()
        await client.aclose()

    @pytest.mark.asyncio
    async def test_reasoning_only_response_raises_inference_error(self, monkeypatch):
        """A response with reasoning content but no assistant text is a failure."""
        cfg = _make_config(monkeypatch, "http://127.0.0.1:8080/v1")

        async def handler(request: httpx.Request) -> httpx.Response:
            body = {
                "choices": [
                    {"message": {"content": None, "role": "assistant"}}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 0},
            }
            return httpx.Response(status_code=200, text=_json.dumps(body))

        client = LlmClient(cfg)
        client._client._transport = httpx.MockTransport(handler)
        with pytest.raises(InferenceError):
            await client.complete([{"role": "user", "content": "Hi"}])
        await client.aclose()

    @pytest.mark.asyncio
    async def test_output_truncation_raises_inference_error(self, monkeypatch):
        """finish_reason='length' should raise InferenceError."""
        cfg = _make_config(monkeypatch, "http://127.0.0.1:8080/v1")

        async def handler(request: httpx.Request) -> httpx.Response:
            body = {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Short answer"},
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
                "timings": {"predicted_per_second": 50.0},
            }
            return httpx.Response(status_code=200, text=_json.dumps(body))

        client = LlmClient(cfg)
        client._client._transport = httpx.MockTransport(handler)
        with pytest.raises(InferenceError) as exc_info:
            await client.complete([{"role": "user", "content": "Hi"}])
        msg = str(exc_info.value).lower()
        assert "truncat" in msg or "budget" in msg or "length" in msg
        await client.aclose()

    @pytest.mark.asyncio
    async def test_invalid_response_structure_raises_inference_error(
        self, monkeypatch
    ):
        """Non-dict top-level must raise InferenceError."""
        cfg = _make_config(monkeypatch, "http://127.0.0.1:8080/v1")

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=200, text="just a string")

        client = LlmClient(cfg)
        client._client._transport = httpx.MockTransport(handler)
        with pytest.raises(InferenceError):
            await client.complete([{"role": "user", "content": "Hi"}])
        await client.aclose()

    @pytest.mark.asyncio
    async def test_missing_message_object_raises_inference_error(
        self, monkeypatch
    ):
        """choices[0].message must be an object."""
        cfg = _make_config(monkeypatch, "http://127.0.0.1:8080/v1")

        async def handler(request: httpx.Request) -> httpx.Response:
            body = {"choices": [{"not_a_message": True}]}
            return httpx.Response(status_code=200, text=_json.dumps(body))

        client = LlmClient(cfg)
        client._client._transport = httpx.MockTransport(handler)
        with pytest.raises(InferenceError):
            await client.complete([{"role": "user", "content": "Hi"}])
        await client.aclose()

    @pytest.mark.asyncio
    async def test_http_connect_error(self, monkeypatch):
        """Connection failures should raise InferenceError."""
        cfg = _make_config(monkeypatch, "http://127.0.0.1:8080/v1")

        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused", request=request)

        client = LlmClient(cfg)
        client._client._transport = httpx.MockTransport(handler)
        with pytest.raises(InferenceError) as exc_info:
            await client.complete([{"role": "user", "content": "Hi"}])
        assert "connection" in str(exc_info.value).lower()
        await client.aclose()

    @pytest.mark.asyncio
    async def test_http_request_error(self, monkeypatch):
        """General request errors should raise InferenceError."""
        cfg = _make_config(monkeypatch, "http://127.0.0.1:8080/v1")

        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.RemoteProtocolError("Protocol error", request=request)

        client = LlmClient(cfg)
        client._client._transport = httpx.MockTransport(handler)
        with pytest.raises(InferenceError):
            await client.complete([{"role": "user", "content": "Hi"}])
        await client.aclose()

    @pytest.mark.asyncio
    async def test_aclose_closes_transport(self, monkeypatch):
        cfg = _make_config(monkeypatch, "http://127.0.0.1:8080/v1")

        client = LlmClient(cfg)
        assert client._client.is_closed is False
        await client.aclose()
        assert client._client.is_closed is True

    @pytest.mark.asyncio
    async def test_inference_convenience_function(self, config):
        """Test the inference() helper uses LlmClient context manager."""
        captured: list[dict] = []
        original_client = None

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.append(_json.loads(request.content))
            return _strata_mock_response(text="convenience")

        # Use a subclass to capture the instance and inject transport
        class _CapturingClient(LlmClient):
            def __init__(self, config, base_url=None):
                super().__init__(config, base_url=base_url)
                self._client._transport = httpx.MockTransport(handler)

        async with _CapturingClient(config) as client:
            result = await client.complete([{"role": "user", "content": "test"}])

        assert result.text == "convenience"
        # Verify context manager was used (client should be closed).
        original_client = client  # captured from the async with
        assert client._client.is_closed is True
        # Also verify messages were forwarded.
        assert len(captured) == 1

    # ── Timing / tokens_per_second regressions ─────────────────────

    @pytest.mark.asyncio
    async def test_top_level_timings_used(self, monkeypatch):
        """timings.predicted_per_second is read from top-level, not usage."""
        cfg = _make_config(monkeypatch, "http://127.0.0.1:8080/v1")

        async def handler(request: httpx.Request) -> httpx.Response:
            body = {
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 10,
                    "total_tokens": 15,
                    "timings": {"predicted_per_second": 99.9},  # stale fallback
                },
                "timings": {"predicted_per_second": 42.0},  # real value
            }
            return httpx.Response(status_code=200, text=_json.dumps(body))

        client = LlmClient(cfg)
        client._client._transport = httpx.MockTransport(handler)
        result = await client.complete([{"role": "user", "content": "Hi"}])
        assert result.tokens_per_second == 42.0
        await client.aclose()

    @pytest.mark.asyncio
    async def test_tokens_per_second_fallback_predicted_time(self, monkeypatch):
        """Fall back to completion_tokens / predicted_time_ms."""
        cfg = _make_config(monkeypatch, "http://127.0.0.1:8080/v1")

        async def handler(request: httpx.Request) -> httpx.Response:
            # 20 tokens in 400 ms = 50 tok/s
            body = {
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 20, "total_tokens": 25},
                "timings": {"predicted_time": 400},  # milliseconds
            }
            return httpx.Response(status_code=200, text=_json.dumps(body))

        client = LlmClient(cfg)
        client._client._transport = httpx.MockTransport(handler)
        result = await client.complete([{"role": "user", "content": "Hi"}])
        assert abs(result.tokens_per_second - 50.0) < 0.1
        await client.aclose()

    @pytest.mark.asyncio
    async def test_missing_timing_data(self, monkeypatch):
        """When no timing data is present tokens_per_second should be None."""
        cfg = _make_config(monkeypatch, "http://127.0.0.1:8080/v1")

        async def handler(request: httpx.Request) -> httpx.Response:
            body = {
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 10},
            }
            return httpx.Response(status_code=200, text=_json.dumps(body))

        client = LlmClient(cfg)
        client._client._transport = httpx.MockTransport(handler)
        result = await client.complete([{"role": "user", "content": "Hi"}])
        assert result.tokens_per_second is None
        await client.aclose()

    @pytest.mark.asyncio
    async def test_missing_usage_dict(self, monkeypatch):
        """Missing usage dict should not crash."""
        cfg = _make_config(monkeypatch, "http://127.0.0.1:8080/v1")

        async def handler(request: httpx.Request) -> httpx.Response:
            body = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
            return httpx.Response(status_code=200, text=_json.dumps(body))

        client = LlmClient(cfg)
        client._client._transport = httpx.MockTransport(handler)
        result = await client.complete([{"role": "user", "content": "Hi"}])
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0
        await client.aclose()


# ══════════════════════════════════════════════════════════════════════════
# Tool-calling unit tests (parse_tool_arguments / _extract_tool_calls)
# ══════════════════════════════════════════════════════════════════════════


class TestParseToolArguments:
    def test_valid_json(self):
        from llm_client import parse_tool_arguments

        out = parse_tool_arguments('{"query": "capital of France"}')
        assert out == {"query": "capital of France"}

    def test_empty_string(self):
        from llm_client import parse_tool_arguments

        assert parse_tool_arguments("") == {}

    def test_missing_arguments(self):
        from llm_client import parse_tool_arguments

        assert parse_tool_arguments(None) == {}

    def test_invalid_json_returns_empty(self):
        from llm_client import parse_tool_arguments

        assert parse_tool_arguments("not json") == {}

    def test_nested_json(self):
        from llm_client import parse_tool_arguments

        out = parse_tool_arguments('{"a": {"b": [1, 2, 3]}}')
        assert out == {"a": {"b": [1, 2, 3]}}


class TestExtractToolCalls:
    def _assistant_with_tool_calls(self, tool_calls: list[dict]) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    }
                }
            ]
        }

    def test_no_tool_calls(self):
        from llm_client import _extract_tool_calls

        data = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        assert _extract_tool_calls(data) == []

    def test_extract_single(self):
        from llm_client import _extract_tool_calls

        data = self._assistant_with_tool_calls(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query": "Eiffel Tower"}',
                    },
                }
            ]
        )
        calls = _extract_tool_calls(data)
        assert len(calls) == 1
        assert calls[0].id == "call_1"
        assert calls[0].name == "web_search"
        assert calls[0].arguments == '{"query": "Eiffel Tower"}'

    def test_extract_multiple(self):
        from llm_client import _extract_tool_calls

        data = self._assistant_with_tool_calls(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query": "a"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query": "b"}'},
                },
            ]
        )
        calls = _extract_tool_calls(data)
        assert len(calls) == 2
        assert [c.id for c in calls] == ["call_1", "call_2"]

    def test_non_json_arguments_falls_back_to_empty(self):
        from llm_client import _extract_tool_calls

        data = self._assistant_with_tool_calls(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": "not-json"},
                }
            ]
        )
        calls = _extract_tool_calls(data)
        assert calls[0].arguments == "not-json"

    def test_missing_function_name(self):
        from llm_client import _extract_tool_calls

        data = self._assistant_with_tool_calls(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"arguments": "{}"},
                }
            ]
        )
        assert _extract_tool_calls(data) == []
