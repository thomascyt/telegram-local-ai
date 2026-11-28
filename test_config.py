"""Tests for config module — Milestone 1."""

from __future__ import annotations

import math

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all config-related environment variables before each test."""
    keys_to_remove = [
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
    ]
    for key in keys_to_remove:
        monkeypatch.delenv(key, raising=False)


def _load(monkeypatch: pytest.MonkeyPatch, **env_vars: str):
    """Set environment variables and return a fresh Config."""
    for k, v in env_vars.items():
        monkeypatch.setenv(k, v)
    from config import load_config

    return load_config()


# ── Valid configuration tests ────────────────────────────────────────


def test_minimal_valid_config(monkeypatch: pytest.MonkeyPatch):
    """A minimal config with only the required token succeeds."""
    cfg = _load(monkeypatch, TELEGRAM_BOT_TOKEN="123:ABC")
    assert cfg.telegram_bot_token == "123:ABC"
    assert cfg.telegram_allowed_user_ids == frozenset()
    assert cfg.llm_base_url == "http://127.0.0.1:11434/v1"
    assert cfg.llm_model == "qwen35-9b-bartowski-q8-mtp-instruct-reasoning:latest"
    assert cfg.context_length == 8192
    assert cfg.max_tokens == 4096
    assert cfg.temperature == 0.7
    assert cfg.thinking_mode is False
    assert cfg.inference_timeout == 120
    assert cfg.history_limit == 12


def test_all_overrides_valid(monkeypatch: pytest.MonkeyPatch):
    """A config with all fields overridden to valid values."""
    cfg = _load(
        monkeypatch,
        TELEGRAM_BOT_TOKEN="token_here",
        TELEGRAM_ALLOWED_USER_IDS="1,2,3",
        LLM_BASE_URL="https://api.example.com/v1",
        LLM_MODEL="my-model",
        CONTEXT_LENGTH="4096",
        MAX_TOKENS="2048",
        TEMPERATURE="1.0",
        THINKING_MODE="yes",
        INFERENCE_TIMEOUT="60",
        HISTORY_LIMIT="10",
    )
    assert cfg.telegram_bot_token == "token_here"
    assert cfg.telegram_allowed_user_ids == frozenset({1, 2, 3})
    assert cfg.llm_base_url == "https://api.example.com/v1"
    assert cfg.llm_model == "my-model"
    assert cfg.context_length == 4096
    assert cfg.max_tokens == 2048
    assert cfg.temperature == 1.0
    assert cfg.thinking_mode is True
    assert cfg.inference_timeout == 60
    assert cfg.history_limit == 10


def test_token_masked_in_repr(monkeypatch: pytest.MonkeyPatch):
    """The bot token must not appear in the repr of Config."""
    cfg = _load(monkeypatch, TELEGRAM_BOT_TOKEN="secret-token")
    r = repr(cfg)
    assert "secret-token" not in r


def test_allowlist_parsing_variations(monkeypatch: pytest.MonkeyPatch):
    """Allowlist with whitespace and single value."""
    cfg = _load(
        monkeypatch,
        TELEGRAM_BOT_TOKEN="t",
        TELEGRAM_ALLOWED_USER_IDS=" 42 , 99 ",
    )
    assert cfg.telegram_allowed_user_ids == frozenset({42, 99})


def test_is_user_allowed_true(monkeypatch: pytest.MonkeyPatch):
    """Allowed user should return True."""
    cfg = _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", TELEGRAM_ALLOWED_USER_IDS="5")
    assert cfg.is_user_allowed(5) is True


def test_is_user_allowed_false(monkeypatch: pytest.MonkeyPatch):
    """Disallowed user should return False."""
    cfg = _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", TELEGRAM_ALLOWED_USER_IDS="5")
    assert cfg.is_user_allowed(6) is False


def test_allowlist_empty_defaults_fail_closed(monkeypatch: pytest.MonkeyPatch):
    """Unset allowlist defaults to empty set (fail closed)."""
    cfg = _load(monkeypatch, TELEGRAM_BOT_TOKEN="t")
    assert cfg.is_user_allowed(12345) is False
    assert cfg.telegram_allowed_user_ids == frozenset()


# ── Invalid configuration tests ──────────────────────────────────────


def test_missing_token(monkeypatch: pytest.MonkeyPatch):
    """Missing required token raises RuntimeError."""
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        _load(monkeypatch)


def test_negative_context_length(monkeypatch: pytest.MonkeyPatch):
    """Negative CONTEXT_LENGTH raises RuntimeError."""
    with pytest.raises(RuntimeError, match="CONTEXT_LENGTH.*positive"):
        _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", CONTEXT_LENGTH="-100")


def test_zero_context_length(monkeypatch: pytest.MonkeyPatch):
    """Zero CONTEXT_LENGTH raises RuntimeError."""
    with pytest.raises(RuntimeError, match="CONTEXT_LENGTH.*positive"):
        _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", CONTEXT_LENGTH="0")


def test_negative_max_tokens(monkeypatch: pytest.MonkeyPatch):
    """Negative MAX_TOKENS raises RuntimeError."""
    with pytest.raises(RuntimeError, match="MAX_TOKENS.*positive"):
        _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", MAX_TOKENS="-5")


def test_zero_max_tokens(monkeypatch: pytest.MonkeyPatch):
    """Zero MAX_TOKENS raises RuntimeError."""
    with pytest.raises(RuntimeError, match="MAX_TOKENS.*positive"):
        _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", MAX_TOKENS="0")


def test_max_tokens_greater_than_context_length(monkeypatch: pytest.MonkeyPatch):
    """max_tokens >= context_length raises ValueError in __post_init__."""
    with pytest.raises(ValueError, match="max_tokens.*context_length"):
        _load(
            monkeypatch,
            TELEGRAM_BOT_TOKEN="t",
            CONTEXT_LENGTH="100",
            MAX_TOKENS="200",
        )


def test_max_tokens_equal_to_context_length(monkeypatch: pytest.MonkeyPatch):
    """max_tokens == context_length raises ValueError."""
    with pytest.raises(ValueError, match="max_tokens.*context_length"):
        _load(
            monkeypatch,
            TELEGRAM_BOT_TOKEN="t",
            CONTEXT_LENGTH="100",
            MAX_TOKENS="100",
        )


def test_temperature_negative(monkeypatch: pytest.MonkeyPatch):
    """Negative temperature raises RuntimeError."""
    with pytest.raises(RuntimeError, match="TEMPERATURE.*range"):
        _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", TEMPERATURE="-0.1")


def test_temperature_above_two(monkeypatch: pytest.MonkeyPatch):
    """Temperature above 2.0 raises RuntimeError."""
    with pytest.raises(RuntimeError, match="TEMPERATURE.*range"):
        _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", TEMPERATURE="2.1")


def test_temperature_inf(monkeypatch: pytest.MonkeyPatch):
    """Temperature of inf raises RuntimeError."""
    with pytest.raises(RuntimeError, match="TEMPERATURE.*finite"):
        _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", TEMPERATURE="inf")


def test_temperature_nan(monkeypatch: pytest.MonkeyPatch):
    """Temperature of nan raises RuntimeError."""
    with pytest.raises(RuntimeError, match="TEMPERATURE.*finite"):
        _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", TEMPERATURE="nan")


def test_temperature_boundary_zero(monkeypatch: pytest.MonkeyPatch):
    """Temperature exactly 0.0 is accepted."""
    cfg = _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", TEMPERATURE="0.0")
    assert cfg.temperature == 0.0


def test_temperature_boundary_two(monkeypatch: pytest.MonkeyPatch):
    """Temperature exactly 2.0 is accepted."""
    cfg = _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", TEMPERATURE="2.0")
    assert cfg.temperature == 2.0


def test_temperature_invalid_string(monkeypatch: pytest.MonkeyPatch):
    """Non-numeric temperature string raises RuntimeError."""
    with pytest.raises(RuntimeError, match="TEMPERATURE.*number"):
        _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", TEMPERATURE="maybe")


def test_thinking_mode_invalid_value(monkeypatch: pytest.MonkeyPatch):
    """Invalid thinking_mode value raises RuntimeError.

    Note: empty string is treated as unset by ``_env`` and falls back to
    the default rather than raising — that is tested separately below.
    """
    for bad_val in ("maybe", "tr ue", "2"):
        with pytest.raises(RuntimeError, match="THINKING_MODE.*boolean"):
            _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", THINKING_MODE=bad_val)


def test_thinking_mode_empty_falls_back_to_default(monkeypatch: pytest.MonkeyPatch):
    """Empty THINKING_MODE string falls back to default (False)."""
    cfg = _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", THINKING_MODE="")
    assert cfg.thinking_mode is False


def test_thinking_mode_true_values(monkeypatch: pytest.MonkeyPatch):
    """Valid thinking_mode true values."""
    for val in ("true", "True", "TRUE", "yes", "Yes", "YES", "1"):
        cfg = _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", THINKING_MODE=val)
        assert cfg.thinking_mode is True, f"Expected True for {val!r}"


def test_thinking_mode_false_values(monkeypatch: pytest.MonkeyPatch):
    """Valid thinking_mode false values."""
    for val in ("false", "False", "FALSE", "no", "No", "NO", "0"):
        cfg = _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", THINKING_MODE=val)
        assert cfg.thinking_mode is False, f"Expected False for {val!r}"


def test_thinking_mode_invalid_int_string(monkeypatch: pytest.MonkeyPatch):
    """Thinking mode with '2' (not in allowed set) raises error."""
    with pytest.raises(RuntimeError, match="THINKING_MODE.*boolean"):
        _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", THINKING_MODE="2")


def test_inference_timeout_negative(monkeypatch: pytest.MonkeyPatch):
    """Negative INFERENCE_TIMEOUT raises RuntimeError."""
    with pytest.raises(RuntimeError, match="INFERENCE_TIMEOUT.*positive"):
        _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", INFERENCE_TIMEOUT="-10")


def test_inference_timeout_zero(monkeypatch: pytest.MonkeyPatch):
    """Zero INFERENCE_TIMEOUT raises RuntimeError."""
    with pytest.raises(RuntimeError, match="INFERENCE_TIMEOUT.*positive"):
        _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", INFERENCE_TIMEOUT="0")


def test_history_limit_negative(monkeypatch: pytest.MonkeyPatch):
    """Negative HISTORY_LIMIT raises RuntimeError."""
    with pytest.raises(RuntimeError, match="HISTORY_LIMIT.*positive"):
        _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", HISTORY_LIMIT="-5")


def test_context_length_invalid_string(monkeypatch: pytest.MonkeyPatch):
    """Non-integer CONTEXT_LENGTH raises RuntimeError."""
    with pytest.raises(RuntimeError, match="CONTEXT_LENGTH.*integer"):
        _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", CONTEXT_LENGTH="abc")


def test_url_gopher_scheme(monkeypatch: pytest.MonkeyPatch):
    """gopher scheme is rejected."""
    with pytest.raises(RuntimeError, match="LLM_BASE_URL.*scheme"):
        _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", LLM_BASE_URL="gopher://example.com")


def test_url_no_hostname(monkeypatch: pytest.MonkeyPatch):
    """URL without hostname raises RuntimeError."""
    with pytest.raises(RuntimeError, match="LLM_BASE_URL.*hostname"):
        _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", LLM_BASE_URL="http:///v1")


def test_url_embedded_credentials_username(monkeypatch: pytest.MonkeyPatch):
    """URL with embedded username raises RuntimeError via parsed.username."""
    with pytest.raises(RuntimeError, match="LLM_BASE_URL.*credentials"):
        _load(
            monkeypatch,
            TELEGRAM_BOT_TOKEN="t",
            LLM_BASE_URL="http://user:pass@host/v1",
        )


def test_url_embedded_credentials_password_only(monkeypatch: pytest.MonkeyPatch):
    """URL with only password (no username) via @ raises RuntimeError."""
    # With parsed.password — a URL like http://:pass@host has password set
    with pytest.raises(RuntimeError, match="LLM_BASE_URL.*credentials"):
        _load(
            monkeypatch,
            TELEGRAM_BOT_TOKEN="t",
            LLM_BASE_URL="http://:pass@host/v1",
        )


def test_url_http_is_valid(monkeypatch: pytest.MonkeyPatch):
    """Plain http URL is accepted."""
    cfg = _load(
        monkeypatch,
        TELEGRAM_BOT_TOKEN="t",
        LLM_BASE_URL="http://localhost:11434",
    )
    assert cfg.llm_base_url == "http://localhost:11434"


def test_url_https_is_valid(monkeypatch: pytest.MonkeyPatch):
    """HTTPS URL is accepted."""
    cfg = _load(
        monkeypatch,
        TELEGRAM_BOT_TOKEN="t",
        LLM_BASE_URL="https://api.example.com/v1",
    )
    assert cfg.llm_base_url == "https://api.example.com/v1"


def test_non_positive_user_id(monkeypatch: pytest.MonkeyPatch):
    """User ID 0 in allowlist raises RuntimeError."""
    with pytest.raises(RuntimeError, match="positive"):
        _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", TELEGRAM_ALLOWED_USER_IDS="0")


def test_negative_user_id_in_allowlist(monkeypatch: pytest.MonkeyPatch):
    """Negative user ID in allowlist raises RuntimeError."""
    with pytest.raises(RuntimeError, match="positive"):
        _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", TELEGRAM_ALLOWED_USER_IDS="-5")


def test_allowlist_invalid_string(monkeypatch: pytest.MonkeyPatch):
    """Non-integer value in allowlist raises RuntimeError."""
    with pytest.raises(RuntimeError, match="user ID"):
        _load(monkeypatch, TELEGRAM_BOT_TOKEN="t", TELEGRAM_ALLOWED_USER_IDS="abc")


def test_allowlist_whitespace_only(monkeypatch: pytest.MonkeyPatch):
    """Allowlist with only whitespace/special chars results in empty frozenset."""
    cfg = _load(
        monkeypatch,
        TELEGRAM_BOT_TOKEN="t",
        TELEGRAM_ALLOWED_USER_IDS=" ,  ",
    )
    assert cfg.telegram_allowed_user_ids == frozenset()


def test_default_history_limit_is_twelve(monkeypatch: pytest.MonkeyPatch):
    """Default HISTORY_LIMIT is 12 conversation turns."""
    cfg = _load(monkeypatch, TELEGRAM_BOT_TOKEN="t")
    assert cfg.history_limit == 12
