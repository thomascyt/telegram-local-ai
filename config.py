"""Configuration for the Telegram–LLM chatbot.

All settings are read from environment variables so that no secrets
are ever committed to source control.

Required:
    TELEGRAM_BOT_TOKEN          Bot token from @BotFather.

Optional (defaults shown):
    TELEGRAM_ALLOWED_USER_IDS   Comma-separated Telegram user IDs, e.g. "123,456".
                                Empty/unset = allow nobody (fail closed).
    LLM_BASE_URL                http://127.0.0.1:8080/v1
    LLM_MODEL                   qwen35-9b-bartowski-q8-mtp-instruct-reasoning:latest
    CONTEXT_LENGTH              8192
    MAX_TOKENS                  4096
    TEMPERATURE                 0.7
    THINKING_MODE               false
    INFERENCE_TIMEOUT           120 (seconds)
    HISTORY_LIMIT               12 conversation turns
"""

from __future__ import annotations

import math
import os
import urllib.parse
from dataclasses import dataclass, field


def _env(name: str, default: str | None = None) -> str | None:
    """Return a stripped environment variable, or *default* when unset/blank."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_required(name: str) -> str:
    value = _env(name)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _parse_int_strict(name: str, default: int) -> int:
    """Parse a positive integer from an environment variable."""
    value = _env(name)
    if value is None:
        return default
    try:
        v = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc
    if v <= 0:
        raise RuntimeError(
            f"{name} must be a positive integer, got {v}"
        )
    return v


def _parse_float_strict(name: str, default: float) -> float:
    """Parse a float from an environment variable in range [0.0, 2.0]."""
    value = _env(name)
    if value is None:
        return default
    try:
        v = float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(v):
        raise RuntimeError(
            f"{name} must be finite, got {v}"
        )
    if v < 0.0 or v > 2.0:
        raise RuntimeError(
            f"{name} must be in range [0.0, 2.0], got {v}"
        )
    return v


def _parse_bool_strict(name: str, default: bool) -> bool:
    """Strict boolean parsing: true/false|yes/no|1/0 only."""
    value = _env(name)
    if value is None:
        return default
    mapping: dict[str, bool] = {
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
        "1": True,
        "0": False,
    }
    lower = value.lower()
    if lower not in mapping:
        raise RuntimeError(
            f"{name} must be a boolean (true/false|yes/no|1/0), got {value!r}"
        )
    return mapping[lower]


def _parse_url_strict(name: str, default: str) -> str:
    """Parse and validate an HTTP(S) URL without embedded credentials."""
    value = _env(name)
    if value is None:
        value = default
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in ("http", "https"):
        raise RuntimeError(
            f"{name} must use http or https scheme, got {parsed.scheme!r}"
        )
    if not parsed.netloc:
        raise RuntimeError(
            f"{name} must include a hostname, got {value!r}"
        )
    if parsed.username is not None or parsed.password is not None:
        raise RuntimeError(
            f"{name} must not contain embedded credentials, got {value!r}"
        )
    # Validate port number if present
    hostname = parsed.hostname
    port_str = parsed.port
    if port_str is not None and (port_str < 1 or port_str > 65535):
        raise RuntimeError(
            f"{name} has invalid port number: {port_str}"
        )
    return value


def _parse_user_ids(raw: str | None) -> frozenset[int]:
    """Parse a comma-separated list of Telegram user IDs (fail closed).

    Rejects non-positive user IDs.
    """
    if not raw:
        return frozenset()
    ids = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            uid = int(part)
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid Telegram user ID in TELEGRAM_ALLOWED_USER_IDS: {part!r}"
            ) from exc
        if uid <= 0:
            raise RuntimeError(
                f"Telegram user ID must be positive, got {uid}"
            )
        ids.add(uid)
    return frozenset(ids)


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration."""

    telegram_bot_token: str = field(repr=False)
    telegram_allowed_user_ids: frozenset[int] = frozenset()
    llm_base_url: str = "http://127.0.0.1:8080/v1"
    llm_model: str = "qwen35-9b-bartowski-q8-mtp-instruct-reasoning:latest"
    context_length: int = 8192
    max_tokens: int = 4096
    temperature: float = 0.7
    thinking_mode: bool = False
    inference_timeout: int = 120
    history_limit: int = 12
    google_cse_api_key: str | None = None
    google_cse_id: str | None = None
    brave_api_key: str | None = None
    search_enabled: bool = False

    def __post_init__(self) -> None:
        """Cross-field and intra-field validation for frozen dataclass.

        Validates constraints regardless of construction path (env-loaded
        or direct dataclass instantiation).
        """
        max_tokens = object.__getattribute__(self, "max_tokens")
        context_length = object.__getattribute__(self, "context_length")
        temperature = object.__getattribute__(self, "temperature")

        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError(
                f"max_tokens must be a positive integer, got {max_tokens!r}"
            )
        if not isinstance(context_length, int) or context_length <= 0:
            raise ValueError(
                f"context_length must be a positive integer, got {context_length!r}"
            )
        if max_tokens >= context_length:
            raise ValueError(
                f"max_tokens ({max_tokens}) must be less than "
                f"context_length ({context_length})"
            )
        if not math.isfinite(temperature):
            raise ValueError(
                f"temperature must be finite, got {temperature!r}"
            )
        if temperature < 0.0 or temperature > 2.0:
            raise ValueError(
                f"temperature must be in range [0.0, 2.0], got {temperature!r}"
            )

    def is_user_allowed(self, user_id: int) -> bool:
        """Return True when *user_id* is on the Telegram allowlist."""
        return user_id in self.telegram_allowed_user_ids


def load_config() -> Config:
    """Build a :class:`Config` from the current environment."""
    return Config(
        telegram_bot_token=_env_required("TELEGRAM_BOT_TOKEN"),
        telegram_allowed_user_ids=_parse_user_ids(
            _env("TELEGRAM_ALLOWED_USER_IDS")
        ),
        llm_base_url=_parse_url_strict(
            "LLM_BASE_URL", "http://127.0.0.1:8080/v1"
        ),
        llm_model=_env("LLM_MODEL", "qwen35-9b-bartowski-q8-mtp-instruct-reasoning:latest") or "qwen35-9b-bartowski-q8-mtp-instruct-reasoning:latest",
        context_length=_parse_int_strict("CONTEXT_LENGTH", 8192),
        max_tokens=_parse_int_strict("MAX_TOKENS", 4096),
        temperature=_parse_float_strict("TEMPERATURE", 0.7),
        thinking_mode=_parse_bool_strict("THINKING_MODE", False),
        inference_timeout=_parse_int_strict("INFERENCE_TIMEOUT", 120),
        history_limit=_parse_int_strict("HISTORY_LIMIT", 12),
        google_cse_api_key=_env("GOOGLE_CSE_API_KEY"),
        google_cse_id=_env("GOOGLE_CSE_ID"),
        brave_api_key=_env("BRAVE_API_KEY"),
        search_enabled=_parse_bool_strict("SEARCH_ENABLED", False),
    )
