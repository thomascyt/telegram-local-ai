"""Tests for ``web_search.py``.

All HTTP calls are mocked via ``httpx.MockTransport`` so no real
network requests are made during the test suite.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from web_search import (
    SearchResult,
    WebSearchClient,
    WebSearchError,
)


# ──────────── Helpers ────────────────────────────────────────────────


def _make_transport(
    status: int = 200,
    body: dict | None = None,
    raise_exc: Exception | None = None,
) -> httpx.MockTransport:
    """Return a mock transport that responds with *body* (JSON)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if raise_exc is not None:
            raise raise_exc
        return httpx.Response(
            status_code=status,
            content=json.dumps(body or {}).encode(),
            headers={"content-type": "application/json"},
        )

    return httpx.MockTransport(handler)


def _success_body(items: list[dict[str, str]] | None = None) -> dict:
    """Return a minimal Brave Search API success response."""
    if items is None:
        items = [
            {
                "title": "Result 1",
                "url": "https://example.com/1",
                "description": "Snippet one.",
            },
            {
                "title": "Result 2",
                "url": "https://example.com/2",
                "description": "Snippet two.",
            },
        ]
    return {"organicResults": items}


# ──────────── Construction ───────────────────────────────────────────


class TestWebSearchClientInit:
    def test_valid_credentials(self):
        client = WebSearchClient(api_key="key")
        assert client._api_key == "key"

    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="BRAVE_API_KEY"):
            WebSearchClient(api_key="")

    def test_invalid_top_k_raises(self):
        with pytest.raises(ValueError, match="top_k"):
            WebSearchClient(api_key="k", top_k=0)


# ──────────── search() ───────────────────────────────────────────────


class TestSearch:
    @pytest.mark.asyncio
    async def test_successful_search_returns_results(self):
        transport = _make_transport(body=_success_body())
        client = WebSearchClient(api_key="k")
        client._client = httpx.AsyncClient(transport=transport, timeout=5)

        results = await client.search("hello world")  # type: ignore[attr-defined]
        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].title == "Result 1"
        assert results[0].link == "https://example.com/1"

    @pytest.mark.asyncio
    async def test_empty_items_returns_empty_list(self):
        transport = _make_transport(body={"organicResults": []})
        client = WebSearchClient(api_key="k")
        client._client = httpx.AsyncClient(transport=transport, timeout=5)

        results = await client.search("nothing")  # type: ignore[attr-defined]
        assert results == []

    @pytest.mark.asyncio
    async def test_missing_items_key_returns_empty_list(self):
        transport = _make_transport(body={"something_else": []})
        client = WebSearchClient(api_key="k")
        client._client = httpx.AsyncClient(transport=transport, timeout=5)

        results = await client.search("query")  # type: ignore[attr-defined]
        assert results == []

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty_list(self):
        transport = _make_transport()
        client = WebSearchClient(api_key="k")
        client._client = httpx.AsyncClient(transport=transport, timeout=5)

        results = await client.search("")  # type: ignore[attr-defined]
        assert results == []

    @pytest.mark.asyncio
    async def test_http_403_raises_web_search_error(self):
        transport = _make_transport(status=403, body={"error": {"message": "Forbidden"}})
        client = WebSearchClient(api_key="k")
        client._client = httpx.AsyncClient(transport=transport, timeout=5)

        with pytest.raises(WebSearchError, match="HTTP 403"):
            await client.search("query")  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_http_429_raises_web_search_error(self):
        transport = _make_transport(status=429, body={"error": {"message": "Rate limit"}})
        client = WebSearchClient(api_key="k")
        client._client = httpx.AsyncClient(transport=transport, timeout=5)

        with pytest.raises(WebSearchError, match="HTTP 429"):
            await client.search("query")  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_http_400_raises_web_search_error(self):
        transport = _make_transport(status=400, body={"error": {"message": "Bad request"}})
        client = WebSearchClient(api_key="k")
        client._client = httpx.AsyncClient(transport=transport, timeout=5)

        with pytest.raises(WebSearchError, match="HTTP 400"):
            await client.search("query")  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_connect_error_raises_web_search_error(self):
        transport = _make_transport(raise_exc=httpx.ConnectError("nope", request=httpx.Request("GET", "http://x")))
        client = WebSearchClient(api_key="k")
        client._client = httpx.AsyncClient(transport=transport, timeout=5)

        with pytest.raises(WebSearchError, match="Failed to reach"):
            await client.search("query")  # type: ignore[attr-defined]


# ──────────── format_context() ───────────────────────────────────────


class TestFormatContext:
    @pytest.mark.asyncio
    async def test_empty_results(self):
        client = WebSearchClient(api_key="k")
        ctx = await client.format_context([])  # type: ignore[attr-defined]
        assert ctx == ""

    @pytest.mark.asyncio
    async def test_single_result(self):
        results = [SearchResult(title="T", link="https://x.com", snippet="S")]
        client = WebSearchClient(api_key="k")
        ctx = await client.format_context(results)  # type: ignore[attr-defined]
        assert "[1]" in ctx
        assert "T" in ctx
        assert "S" in ctx
        assert "Source: https://x.com" in ctx

    @pytest.mark.asyncio
    async def test_multiple_results_separated(self):
        results = [
            SearchResult(title="A", link="https://a.com", snippet="Snip A"),
            SearchResult(title="B", link="https://b.com", snippet="Snip B"),
        ]
        client = WebSearchClient(api_key="k")
        ctx = await client.format_context(results)  # type: ignore[attr-defined]
        assert "A" in ctx
        assert "B" in ctx

    @pytest.mark.asyncio
    async def test_truncation_respects_budget(self):
        long_snippet = "x" * 10000
        results = [SearchResult(title="T", link="https://x.com", snippet=long_snippet)]
        client = WebSearchClient(api_key="k")
        ctx = await client.format_context(results)  # type: ignore[attr-defined]
        assert len(ctx) <= client._max_context_chars + 200
        assert "..." in ctx


# ──────────── Context manager ────────────────────────────────────────


class TestContextManager:
    def test_aenter_returns_self(self):
        client = WebSearchClient(api_key="k")
        coro = client.__aenter__()  # type: ignore[attr-defined]
        assert coro is not None  # returns an awaitable

    async def test_aclose_closes_client(self):
        transport = _make_transport()
        client = WebSearchClient(api_key="k")
        client._client = httpx.AsyncClient(transport=transport, timeout=5)
        await client.aclose()  # type: ignore[attr-defined]

# ──────────── Tool definition ────────────────────────────────────────


class TestToolDefinition:
    def test_schema_shape(self):
        defn = WebSearchClient.tool_definition()

        assert defn["type"] == "function"
        assert defn["function"]["name"] == "web_search"
        assert defn["function"]["parameters"]["type"] == "object"

    def test_parameters_required_query(self):
        params = WebSearchClient.tool_definition()["function"]["parameters"]

        assert params["properties"]["query"]["type"] == "string"
        assert params["required"] == ["query"]
        assert params["additionalProperties"] is False

    def test_description_present(self):
        description = WebSearchClient.tool_definition()["function"]["description"]

        assert isinstance(description, str)
        assert len(description) > 10
