"""Web search client for the Brave Search API.

Wraps the Brave Search API so the chatbot can fetch a small number of
up-to-date web results and inject them as context before LLM inference.
The free tier allows 1,000 queries/month (no credit card required).

The client is intentionally defensive: any failure (network error, API
error, rate limit, empty result) is surfaced as a plain exception that
callers can catch and fall back to non-search inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 5
_MAX_SEARCH_CONTEXT_CHARS = 4000

# Brave Search API endpoint. The /res/v1/web/search path is a GET request
# (query params like ?q= and ?count=), unlike the older POST-on-api pattern.
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


@dataclass(frozen=True)
class SearchResult:
    """A single web search result."""

    title: str
    link: str
    snippet: str


class WebSearchError(Exception):
    """Raised when a web search request fails."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code  # type: ignore[assignment]


class WebSearchClient:
    """Asynchronous client for the Brave Search API."""

    def __init__(
        self,
        api_key: str,
        *,
        top_k: int = _DEFAULT_TOP_K,
        max_context_chars: int = _MAX_SEARCH_CONTEXT_CHARS,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("BRAVE_API_KEY is required for web search")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        self._api_key = api_key
        self._top_k = top_k
        self._max_context_chars = max_context_chars
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "x-subscription-token": self._api_key,
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
            },
        )

    async def search(self, query: str) -> list[SearchResult]:
        """Search the web for *query* and return up to ``top_k`` results."""
        if not query or not query.strip():
            return []

        logger.info("Web search: %r", query)
        params = {"q": query.strip(), "count": self._top_k}
        try:
            resp = await self._client.get(BRAVE_SEARCH_URL, params=params)
        except httpx.TimeoutException:
            raise WebSearchError(
                f"Web search request timed out after {self._client.timeout.total_seconds()}s"
            ) from None
        except httpx.ConnectError as exc:
            raise WebSearchError(
                "Failed to reach Brave Search API"
            ) from exc
        except httpx.RequestError as exc:
            raise WebSearchError(f"Web search request failed: {exc}") from exc

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            detail = ""
            if code in (401, 403):
                detail = "(invalid or unauthorized API key)"
            elif code == 429:
                detail = "(rate limit / monthly quota exceeded — search disabled)"
            elif code == 400:
                detail = "(invalid query)"
            raise WebSearchError(
                f"Brave Search API returned HTTP {code} {detail}".rstrip(),
                status_code=code,
            ) from None

        try:
            data = resp.json()
        except ValueError as exc:
            raise WebSearchError(
                f"Web search response is not valid JSON (HTTP {resp.status_code})"
            ) from exc

        # Brave's /res/v1/web/search returns results as lists under
        # "web.results" and "discussions.results". Older endpoints used
        # "organicResults". Google CSE used "items".
        items = None
        if isinstance(data, dict):
            web = data.get("web")
            if isinstance(web, dict) and isinstance(web.get("results"), list):
                items = web["results"]
            elif isinstance(data.get("discussions"), list):
                items = data["discussions"]
            elif isinstance(data.get("organicResults"), list):
                items = data["organicResults"]
        if not isinstance(items, list):
            return []

        results: list[SearchResult] = []
        for item in items[: self._top_k]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            link = str(item.get("url", "")).strip()
            snippet = str(item.get("description", "")).strip()
            if not link:
                continue
            results.append(SearchResult(title=title, link=link, snippet=snippet))

        logger.info(
            "Web search: %d results for %r", len(results), query
        )
        return results

    async def format_context(self, results: list[SearchResult]) -> str:
        """Format search results into a compact context string."""
        if not results:
            return ""

        parts: list[str] = []
        budget = self._max_context_chars
        for i, r in enumerate(results, start=1):
            snippet = r.snippet
            overhead = len(f"[{i}] {r.link}\n")
            avail = budget - overhead
            if avail < 0:
                avail = 0
            if len(snippet) > avail:
                snippet = snippet[: max(0, avail)].rstrip() + "..."
            budget -= len(snippet) + overhead
            parts.append(f"[{i}] {r.title}\n{snippet}\nSource: {r.link}")
            if budget <= 0:
                break

        return "\n\n".join(parts)

    @classmethod
    def tool_definition(cls) -> dict[str, Any]:
        """Return an OpenAI-style tool/function definition for web search.

        This lets a tool-calling model (e.g. Qwen via Strata) decide to
        invoke the search itself, rather than the bot pre-emptively
        searching every message.  The returned schema matches the
        ``{"type": "function", "function": {...}}`` shape expected in a
        chat-completion request's ``tools`` array.
        """
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the public web for current, up-to-date information. "
                    "Use it when the user asks about recent or live topics such "
                    "as news, current events, today's sports scores or game "
                    "results, stock prices, weather, product availability, "
                    "recent software releases, or any factual question whose "
                    "answer may have changed since training. Prefer this over "
                    "answering from memory when timeliness matters. Provide a "
                    "clear, concise search query in English."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "A concise search query describing the "
                                "current information the user needs."
                            ),
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "WebSearchClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

