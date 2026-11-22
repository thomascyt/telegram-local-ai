# telegram-local-ai

A Telegram bot backed by a **local LLM** (OpenAI-compatible API served by Ollama).  
Messages are routed through a conversation manager that budgets tokens, enforces history limits, and serialises per-user requests.

## Features

- **Per-user conversation history** with automatic trimming when context budget is exceeded
- **Async concurrency control** — one request at a time per user (`asyncio.Lock`)
- **Slash commands**: `/start`, `/help`, `/new` (clear), `/status`, `/model`, `/stats`, `/history`
- **User allow-list** — only authorised Telegram user IDs can interact
- **All secrets via environment variables** — nothing hardcoded

## Prerequisites

- Python 3.14+
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- A local LLM serving an OpenAI-compatible API (e.g. Ollama at `http://127.0.0.1:11434/v1`)

## Installation

```bash
cd telegram-local-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# or, if using the project package:
# pip install -e ".[dev]"
```

## Configuration (environment variables)

| Variable | Required | Default | Description |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | **Yes** | — | Bot token from @BotFather |
| `TELEGRAM_ALLOWED_USER_IDS` | No | *(empty = reject all)* | Comma-separated Telegram user IDs |
| `LLM_BASE_URL` | No | `http://127.0.0.1:8080/v1` | LLM API endpoint |
| `LLM_MODEL` | No | `qwen3.6-35b-a3b-ud-q4_k_xl` | Model name |
| `CONTEXT_LENGTH` | No | `8192` | Token capacity of the context window |
| `MAX_TOKENS` | No | `4096` | Maximum output tokens per response |
| `TEMPERATURE` | No | `0.7` | Sampling temperature (range 0–2) |
| `THINKING_MODE` | No | `false` | Enable reasoning/thinking mode (`true`/`false`) |
| `INFERENCE_TIMEOUT` | No | `120` | Seconds to wait for LLM response |
| `HISTORY_LIMIT` | No | `12` | Maximum conversation turns per user |

## Running

```bash
# Set required env vars first
export TELEGRAM_BOT_TOKEN="123456:ABCdefGHIjklMNOpqrsTUVwxyz"
export TELEGRAM_ALLOWED_USER_IDS="123456789"

# Run directly
python telegram_bot.py

# Or as a module
python -m telegram-local-ai
```

## Testing

```bash
.venv/bin/python -m pytest -q
```

All 168 tests run offline with mocked back-ends (no network required).

## Project Structure

```
config.py              # Immutable config from environment variables
llm_client.py          # Async HTTP client for LLM inference
conversation.py        # Per-user conversation manager with context budgeting
telegram_bot.py        # Telegram bot front-end (python-telegram-bot v22)
calculator.py          # Simple math utilities (test utility)
benchmark_16k.py       # Primality test benchmark
strata_agent_test.py   # Interval merging algorithm test
__main__.py            # Module entry point
requirements.txt       # Python dependencies
pyproject.toml         # Package metadata & pytest config
```

## License

Private / internal use.
