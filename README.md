# telegram-local-ai

A Telegram bot backed by a **local LLM** (OpenAI-compatible API served by Ollama).  
Messages are routed through a conversation manager that budgets tokens, enforces history limits, and serialises per-user requests.

## Features

- **Per-user conversation history** with automatic trimming when context budget is exceeded
- **Async concurrency control** — one request at a time per user (`asyncio.Lock`)
- **Slash commands**: `/start`, `/help`, `/new` (clear), `/status`, `/model`, `/stats`, `/history`
- **User allow-list** — only authorised Telegram user IDs can interact
- **Web search via tool calling** — optional Brave Search integration; the model can invoke a `web_search` tool on its own (`/search on`) or run a one-off (`/search query <question>`). Per-user, off by default
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
| `LLM_BASE_URL` | No | `http://127.0.0.1:11434/v1` | LLM API endpoint |
| `LLM_MODEL` | No | `qwen35-9b-bartowski-q8-mtp-instruct-reasoning:latest` | Model name |
| `CONTEXT_LENGTH` | No | `8192` | Token capacity of the context window |
| `MAX_TOKENS` | No | `4096` | Maximum output tokens per response |
| `TEMPERATURE` | No | `0.7` | Sampling temperature (range 0–2) |
| `THINKING_MODE` | No | `false` | Enable reasoning/thinking mode (`true`/`false`) |
| `INFERENCE_TIMEOUT` | No | `120` | Seconds to wait for LLM response |
| `HISTORY_LIMIT` | No | `12` | Maximum conversation turns per user |
| `SEARCH_ENABLED` | No | `false` | Admin master switch for web search |
| `BRAVE_API_KEY` | No | *(empty = search disabled)* | Brave Search API key (see [Web search](#web-search-brave-search)) |

## Web search (Brave Search)

The bot can answer questions using **current, up-to-date web information** through the
[Brave Search API](https://api.search.brave.com/) — see `web_search.py`. The client is a thin
wrapper around the API, so another search provider can be swapped in if you prefer a different engine.

There are two ways to use it:

- **Tool calling (`/search on`)** — when enabled, the model decides on its own whether to call the
  `web_search` tool for time-sensitive questions (news, weather, prices, sports, etc.) rather than
  answering purely from memory. This is **per-user and off by default**.
- **One-off (`/search query <your question>`)** — run a single search and reply with an
  LLM-summarized answer, regardless of the toggle.

You can check status and toggle it per account:

- `/search` — show the current ON/OFF status
- `/search on` / `/search off` — enable or disable the web-search *tool* for your account

### Enabling it

Web search is disabled until you set these in `.env`:

```bash
SEARCH_ENABLED=true
BRAVE_API_KEY=your-brave-api-key-here
```

`SEARCH_ENABLED=true` is the **admin master switch** — without it the bot replies that web search is
disabled. `BRAVE_API_KEY` is required for the search calls. Get a key at
<https://api.search.brave.com/> and add it to `.env` (never commit the real value).

> **Per-user opt-in.** Even with the env vars set, a user must run `/search on` for the model to
> call the search tool automatically; otherwise it answers from knowledge only.

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

## Deployment & Troubleshooting

### How the bot reaches Ollama (host networking)

The bot runs in a container with `network_mode: host` (see `docker-compose.yml`). On a native
Docker Engine inside Ubuntu WSL2, this shares the WSL2 network namespace, so the container's
`127.0.0.1` is the same loopback Ollama listens on. Keep Ollama bound to `127.0.0.1:11434` —
no port publishing is needed.

```yaml
services:
  telegram-bot:
    network_mode: host
```

You don't need to change this to bridge networking to enable host networking.

#### Docker Desktop caveat

Docker Desktop runs its *own* WSL2 backend, so `network_mode: host` shares Docker Desktop's
namespace — **not** your Ubuntu WSL2 one. Ollama bound to `127.0.0.1:11434` in Ubuntu WSL2 would
then be unreachable from the container.

Docker Desktop has an **Enable host networking** setting (Settings → Resources → Network →
Enable host networking → Apply & Restart; officially supported from Docker Desktop 4.34 onward),
but it does **not** share your Ubuntu WSL2 distribution's namespace either. So on Docker Desktop,
prefer one of these:

- Run a native Docker Engine inside Ubuntu WSL2 (as above) — then `network_mode: host` works.
- Point `LLM_BASE_URL` at the Windows host via `extra_hosts`:
  `extra_hosts: - "host.docker.internal:host-gateway"` and `LLM_BASE_URL=http://host.docker.internal:11434/v1`.

(Binding Ollama to `0.0.0.0` is discouraged — Ollama has no built-in authentication, so that would
expose inference to your whole network.)

To test once the networking is sorted:

```bash
cd ~/telegram-local-ai
docker compose up -d --force-recreate
docker exec telegram-bot python -c \
  'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:11434/v1/models", timeout=5).read().decode())'
```

(With Docker Desktop + `host.docker.internal`, use `http://host.docker.internal:11434/v1` above.)

### Ollama context length

Two layers control the context length, and they compose:

1. **Modelfile `num_ctx`** — the model's own configured context (per-model override).
2. **`OLLAMA_CONTEXT_LENGTH`** — a **server-wide default** applied when a model has no `num_ctx` of
   its own. This is a real, honored setting (see the [Ollama docs](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-change-the-context-length-ollama-context-length)):
   ```
   OLLAMA_CONTEXT_LENGTH=64000 ollama serve
   ```

The systemd unit runs `ollama serve`, so a drop-in override **is** applied — contrary to the old note
in this file. Verify it with:

```bash
cat /etc/systemd/system/ollama.service.d/override.conf   # [Service] Environment="OLLAMA_CONTEXT_LENGTH=..."
systemctl show ollama -p Environment                     # confirm the var is present
ollama ps                                                # CONTEXT column = effective context
```

To change it:

```bash
# 1) Edit the systemd drop-in
sudo systemctl edit ollama
# → add:
# [Service]
# Environment="OLLAMA_CONTEXT_LENGTH=16384"
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Or set it per-model in a Modelfile (this overrides the server default):

```dockerfile
FROM qwen35-9b-bartowski-q8-mtp-instruct-reasoning:latest
parameter num_ctx 16384
parameter think false
```

```bash
ollama create qwen35-16k -f Modelfile.qwen35
```

Keep `CONTEXT_LENGTH` in `.env` **aligned** with the effective `num_ctx` so the conversation manager
trims history correctly.

### Choosing a context size for your GPU

Larger context means more GPU memory for the KV cache. On an RTX 4090 (16 GB) running a ~11 GB Q8
9B model, the KV cache footprint is roughly:

| `num_ctx` | approx. KV cache | fits in 16 GB? | behaviour |
|---|---|---|---|
| 8,192 | ~2.4 GB | yes | fully on-GPU, fast |
| 16,384 | ~4.9 GB | yes (tight) | mostly on-GPU, fast |
| 32,768 | ~9.8 GB | borderline | KV cache spills to CPU, slower |
| 131,072 | ~38.6 GB | **no** | KV cache mostly **offloaded to CPU RAM**; inference works but is **CPU-bound and very slow** |

Important: a 128K context **does not OOM** — Ollama keeps the model resident and offloads the KV
cache to system memory instead. You can confirm this on this box right now:

```bash
ollama ps        # CONTEXT = 131072, PROCESSOR shows a large CPU share
nvidia-smi       # ~14.5 GiB used, GPU idle (0% util) while resident
```

So the "increase to 130K / start with 32K" advice is technically *possible* here but **not practical**
for a responsive Telegram bot: the 128K KV cache can't live in 16 GB of VRAM, so inference runs on the
CPU. For interactive use, **8,192–16,384 is the sweet spot** on this hardware. Only raise higher if you
genuinely need very long context and can tolerate slow responses. Measure empirically with
`nvidia-smi` after a real request — a high CPU share in `ollama ps` means the KV cache is offloaded.

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
web_search.py          # Brave Search API client (web-search tool)
__main__.py            # Module entry point
requirements.txt       # Python dependencies
pyproject.toml         # Package metadata & pytest config
```

## License

Private / internal use.
