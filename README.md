# Claude Code & Codex Proxy for Nebius

A Claude Code + Codex CLI → Nebius bridge. Accepts Claude `/v1/messages` and Codex `/v1/responses` requests, converts them to OpenAI-compatible calls, and converts responses back.

## Quick Start (the easy way)

One step. Run the installer:

```bash
./install.sh
```

This launches a step-by-step TUI that:
1. Checks prerequisites
2. Creates a virtual environment
3. Installs dependencies
4. Tests your Nebius API key
5. Lets you pick models from live dropdowns
6. Writes your `.env` file
7. Runs a smoke test
8. Optionally adds `claude` / `claudius` shell shortcuts
9. Optionally configures Claude Code statusline

After it finishes:

```bash
claude --proxy        # or just 'claudius' if you added the shell shortcuts
# OR
claudius
# OR
.venv/bin/python start_proxy.py
```

Then open http://localhost:8083/dashboard for the observability dashboard.

## Prerequisites

- Python 3.9+
- Nebius API key (from https://nebius.com)
- Claude Code or Codex CLI (optional; install from Anthropic / OpenAI)
- Tavily API key (optional; enables server-side web search — see `.env.example`)

## Quick Start (manual)

If you prefer not to use the TUI:

```bash
# 1. Clone and enter directory
cd claude-codex-nebius-proxy

# 2. Create venv & install deps
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env: set OPENAI_API_KEY, BIG_MODEL, etc.

# 4. Run
.venv/bin/python start_proxy.py

# 5. Use
ANTHROPIC_BASE_URL=http://localhost:8083 ANTHROPIC_AUTH_TOKEN=claude-local claude
```

## Observability

Open http://localhost:8083/dashboard for usage, latency, cost, and model routing info.

## Quick Start (Codex CLI)

To route Codex CLI through the proxy, edit `~/.codex/config.toml` (macOS) or `%APPDATA%\codex\config.toml` (Windows):

```toml
model = "nebius/moonshotai/Kimi-K2.6"
model_provider = "nebius"

[model_providers.nebius]
name = "Nebius Proxy"
base_url = "http://127.0.0.1:8083/v1"
env_key = "OPENAI_API_KEY"
wire_api = "responses"

[projects."/path/to/your/project"]
trust_level = "trusted"

[tui]
status_line = ["model-with-reasoning", "task-progress", "permissions", "approval-mode", "fast-mode"]
status_line_use_colors = true
```

Set your Nebius API key (the proxy forwards it to the backend):

```bash
export OPENAI_API_KEY="nb-..."  # macOS / Linux
```

```powershell
$env:OPENAI_API_KEY = "nb-..."  # Windows PowerShell
```

Then open a project directory and run:

```bash
cd /path/to/your/project && codex --full-auto
```

**Codex Desktop App (macOS):** The GUI app does not inherit shell exports. Set the key via `launchctl` before launching:

```bash
launchctl setenv OPENAI_API_KEY "nb-..."
open -a "Codex"
```

Codex will use the proxy for all `/v1/responses` calls, with server-side web search (if Tavily is configured) and model routing.

See [docs/codex/CODEX_STATUSLINE.md](docs/codex/CODEX_STATUSLINE.md) for full statusline configuration options.

## Documentation

For full configuration options, model details, architecture, troubleshooting, and the complete feature list, see **[docs/README.md](docs/README.md)**.

| Doc                                                              | What's inside                                 |
| ---------------------------------------------------------------- | --------------------------------------------- |
| [docs/README.md](docs/README.md)                                 | Full reference index — everything             |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)                     | System flow, key files, design                |
| [docs/TOOL_CALL_FORMAT.md](docs/TOOL_CALL_FORMAT.md)             | Claude SSE tool-call streaming                |
| [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md)                   | Dashboard configuration                       |
| [docs/SHELL_FUNCTION.md](docs/SHELL_FUNCTION.md)                 | Shell shortcut reference                      |
| [docs/BINARY_PACKAGING.md](docs/BINARY_PACKAGING.md)             | Standalone binary notes                       |
| [docs/MANUAL_SETUP.md](docs/MANUAL_SETUP.md)                     | Manual setup (skip the TUI)                   |
| [docs/codex/CODEX_STATUSLINE.md](docs/codex/CODEX_STATUSLINE.md) | Codex CLI proxy routing and statusline config |

## Features (high-level)

- Claude `/v1/messages` proxying to Nebius OpenAI-compatible endpoints
- Codex `/v1/responses` proxying to Nebius OpenAI-compatible endpoints
- Streaming SSE (Claude and Codex)
- Server-side web search via Tavily (executes `web_search` tool calls, returns results to the model)
- Automatic model routing (big / middle / small / vision)
- Token-aware long-context routing (large prompts escalate to a long-context model)
- Built-in request optimizations for Claude Code housekeeping
- Tool-call JSON repair and deduplication
- Context auto-truncation (never orphans tool results)
- Observability dashboard with cost tracking

## Scope

Designed for Nebius token factory infrastructure. Defaults and troubleshooting guidance are Nebius-centric rather than provider-agnostic.

## Alternatives

This proxy is batteries-included and Nebius-tuned. If your needs are narrower, two well-maintained, config-driven tools cover the core Anthropic/OpenAI ↔ Nebius translation with much less to run:

- **[claude-code-router](https://github.com/musistudio/claude-code-router)** — points Claude Code at any OpenAI-compatible endpoint (including Nebius Token Factory) with per-request-type model routing (`default` / `background` / `think` / `longContext`, analogous to this project's big / middle / small / vision). Install, drop Nebius into a JSON config, run `ccr code`. Best if you use **only Claude Code** and want lightweight routing that tracks upstream Claude Code updates.
- **[LiteLLM proxy](https://docs.litellm.ai/docs/anthropic_unified/)** — exposes an Anthropic `/v1/messages` surface in front of any OpenAI-compatible backend, with built-in routing, fallbacks, and cost tracking. Point Claude Code at it via `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN`. Best if you want a mature multi-provider gateway, or already run LiteLLM. (Pin to a known-good release — a couple of past versions were flagged in a supply-chain incident.)

### When to use this project instead

Reach for this proxy when you want the extras those tools don't bundle:

- **One bridge for both CLIs** — Claude Code (`/v1/messages`) *and* OpenAI Codex (`/v1/responses`) against the same Nebius backend.
- **Server-side web search** — runs `web_search` tool calls via Tavily and returns results to the model, so search works behind a non-Anthropic backend.
- **Claude Code housekeeping optimizations** — answers low-value requests (title generation, network probes, filepath extraction) locally to save tokens and latency.
- **Tool-call JSON repair, context auto-truncation, and a built-in cost/latency dashboard.**
- **Nebius-first ergonomics** — TUI installer, live model dropdowns, and `.env` defaults tuned for Nebius Token Factory.

## License

MIT
