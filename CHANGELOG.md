# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Convention:** every change-worthy PR adds an entry under `[Unreleased]` in the
> matching group (`Added` / `Changed` / `Fixed` / `Removed` / `Deprecated` / `Security`).
> On release, rename `[Unreleased]` to the new version + date and start a fresh
> `[Unreleased]` block.

## [Unreleased]

### Security
- **Bind loopback by default** (`src/core/config.py`, `.env.example`): `HOST` now defaults to `127.0.0.1` instead of `0.0.0.0`. The proxy serves a local client (Claude Code / Codex) and ignores client auth by default, so binding all interfaces exposed the inference endpoint to the whole LAN. Set `HOST=0.0.0.0` explicitly to serve other machines.
- **Prompt/error bodies no longer logged in full** (`src/core/client.py`, `src/main.py`): `_log_openai_error` now logs a truncated, secret-scrubbed summary of upstream error bodies (Token Factory errors can echo request fields, including prompt text), and the 422 handler records only the path + validation errors — not the request body.

### Fixed
- **Dockerized proxy unreachable when `.env` sets `HOST=127.0.0.1`** (`docker-compose.yml`): the proxy service loads `.env` via `env_file`, so a loopback `HOST` (useful for native host runs) made uvicorn bind `127.0.0.1` inside the container and every connection to the published port 8083 was reset — `curl http://localhost:8083/dashboard` failed with "Connection reset by peer" even though the container was Up. The compose `environment:` block now pins `HOST: 0.0.0.0` for the container (same pattern as the existing `LANGFUSE_HOST` override), leaving the `.env` value in effect for host runs.
- **statusLine stuck on startup model after a runtime switch** (`scripts/install_ui/screens.py:STATUSLINE_CMD`): the status line read `model="${NEBIUS_SESSION_MODEL:-}"` first and only queried the live `/api/observability/config` `effective_model` when that env var was empty — but `claude --proxy` always exports `NEBIUS_SESSION_MODEL` at startup, so the live fetch was skipped and the statusline never reflected a model switched from the dashboard picker. The command now queries `effective_model` via the forwarder first (the forwarder injects `x-session-name`, so it reflects any per-session runtime override set by `PUT /v1/session-model`), falling back to `NEBIUS_SESSION_MODEL` only when the proxy is unreachable. Same single config fetch as before; no new endpoint.
- **statusLine overwrite silently dropped by installer** (`scripts/install_ui/utils.py`): `safe_merge_settings` returned `action: "updated"` for an already-configured-but-different statusLine **without writing the new command**, so the "Overwrite existing statusLine?" modal in the wizard accepted the user's choice but left the stale command in `~/.claude/settings.json` untouched — upgrades to the clickable OSC 8 model link never landed on machines that already had an older statusLine. The "updated" branch now overwrites the command (a backup is already taken before this point); the identical-statusLine short-circuit and the add/create paths are unchanged.
- **Inline-text tool-call lifter for Kimi-K2.7-Code** (`src/conversion/response_converter.py`): Some Open-Chat-Completions backends (notably `moonshotai/Kimi-K2.7-Code` via Nebius) emit tool calls as control-token text inside `delta.content` (e.g. `  ...   {args}    `) instead of structured `delta.tool_calls`. The `/v1/messages` streaming converter streamed these through as a text block ending `stop_reason: end_turn`, so Claude Code had nothing to execute and the turn stalled; each retry replayed the poisoned history and the loop repeated. The converter now lifts such sections into proper Anthropic `tool_use` content blocks and overrides the turn's stop reason to `tool_use` (streaming and non-streaming paths). Kimi-K2 emits only a tool-call id where the function name should be, so the name is recovered by matching the parsed args' keys against the request's tool schemas (with a `command`->Bash backstop). Lift is split-token safe (a section opener split across SSE chunks is held back, not flushed as text), dedups duplicate (name, args) calls, and never stalls ordinary text containing a bare `<`. Format set is appendable: a new broken-emit format = one extractor function + one line in `_INLINE_TOOL_EXTRACTORS` (plus its opener in `_INLINE_SECTION_OPENERS`).
- **Premature "context limit reached" / early auto-compaction** (`src/conversion/request_converter.py`): `DEFAULT_CONTEXT_LIMIT` raised `128 000` → `200 000` to match the standard (non-1m) Claude reference window. With a 128K default against a 200K Claude window, `compute_usage_scale` amplified every reported usage figure by ~1.56×, so a session with 114K real backend tokens was reported to Claude Code as ~179K/200K (89%) and tripped its context-limit/auto-compact guard far too early. At 200K the scale is a no-op (1.0) whenever the real backend window is unknown, so reported usage tracks reality.
- **Context limit catalog sanity floor** (`src/conversion/request_converter.py`, `scripts/install_ui/utils.py`): Nebius API returns `context_length: 8000` as a placeholder for many models with 128K+ real windows (Kimi-K2.x, MiniMax-M2.5, etc.). The proxy now ignores catalog values below 16 384 and falls through to `DEFAULT_CONTEXT_LIMIT`. The installer applies the same floor when writing `MODEL_CONTEXT_LIMIT`, so `8000` becomes `0` (runtime fallback) rather than poisoning `.env`. This was the root cause of `max_tokens: 1` on new sessions after the `MODEL_CONTEXT_LIMIT=0` change.
- **Context-full floor raised** (`src/conversion/request_converter.py`): When the computed `available` tokens is below `min_tokens_limit`, the proxy now logs a warning with the exact breakdown (limit / prompt / tool_overhead / raw_available) and floors to `min_tokens_limit` instead of `1`. A `max_tokens: 1` response produces a 1-token reply that Claude Code treats as "context full" and then attempts a compaction that also fails; a meaningful floor produces a real response even when context is tight.
- **Installer now writes the correct model context limit** (`scripts/install_ui/utils.py`, `screens.py`): `fetch_nebius_models` now calls `/models?verbose=true` (was `/models`) to obtain `context_length` per model. The TUI stores the map in `InstallState.model_context_lengths` and `write_env` uses it to write accurate `MODEL_CONTEXT_LIMIT` / `VISION_MODEL_CONTEXT_LIMIT` values for whichever models the user picks. When a model's context length is unknown, `0` is written so the runtime catalog fallback in `_get_context_limit` takes over instead of using a stale hardcoded value. `.env.example` defaults changed from `204800` → `0` to match this behavior.
- **Tool-definition tokens now counted in context guard** (`src/conversion/request_converter.py`): `_estimate_prompt_tokens` only counted message tokens; tool definitions (5–10k tokens for Claude Code's full tool set) were added to the backend request *after* the estimate, causing `max_tokens` to be over-allocated and the effective prompt to exceed the model's context window. Both message trimming and `max_tokens` computation now include a `_estimate_tools_tokens` pass over the raw Claude tool list, so the backend never receives a request that exceeds its context limit and compaction has accurate room to operate.
- **Idle-stream watchdog** (`src/core/client.py`): `REQUEST_TIMEOUT` only bounded stream *setup*; once a stream started, `async for chunk` awaited the next chunk forever, so a hung Nebius stream went silent without erroring — the "response just stops" symptom. Each chunk read is now wrapped in a `STREAM_IDLE_TIMEOUT` deadline (default 120s). On timeout the proxy surfaces a typed, client-retryable `503` / `overloaded_error` (and detects client disconnect while awaiting the next chunk), instead of stalling indefinitely.
- **Oversized tool results are compacted in place** (`src/conversion/request_converter.py`): a single huge tool result (a big file read, a long `bash` dump) previously forced the trimmer to drop whole recent turn-groups. Any `role=tool` content over ~512 tokens is now shrunk in place to a head+tail of ~384 tokens with an `[… omitted N tokens …]` marker, keeping the turn. The most recent message (the in-flight tool output) is never compacted. Applied to both the Claude and Codex request paths.
- **Consecutive same-role turns are merged** (`src/conversion/request_converter.py`): some Token Factory models reject role-alternation violations, which Claude Code can produce (consecutive user/tool turns). Adjacent same-role messages (system/user/assistant text) are now coalesced. Messages carrying `tool_calls` and `role=tool` messages are never merged (their structure is significant), and content carrying any non-text block (`image_url`, `tool_result`, …) is never merged — flattening it to text would silently drop the block. Applied to both Claude and Codex paths.
- **`thinking.budget_tokens` honored for reasoning effort** (`src/conversion/request_converter.py`): clients that signal extended reasoning via `thinking: {budget_tokens: N}` (rather than `output_config.effort`) were silently dropped. Budget now maps to `reasoning_effort` (`<8k`→low, `<32k`→medium, else high); `output_config.effort` still wins when both are present.
- **`parallel_tool_calls` never forwarded to Token Factory** (`src/codex/request_converter.py`): the Codex converter accepts `parallel_tool_calls` but the upstream rejects/misbehaves on it; it is now always omitted from the outgoing request (tools and tool_choice are unaffected).
- **Inline system messages hoisted to the front** (`src/conversion/request_converter.py`): Anthropic clients may place a `role=system` turn inside `messages` rather than in the top-level `system` field. The converter previously left it inline, so the downstream Nebius backend rejected the request with `System message must be at the beginning.` All system messages are now moved to the start of the outgoing message list before the backend request is built.
- **Inline tool-call lifter misses Kimi-K2.7-Code bare-args emissions** (`src/conversion/response_converter.py`): Kimi-K2.7-Code now (2026-07) emits inline tool-call sections *without* the `  essay` token — the args JSON directly follows the bare id (`<tool_call> chatcmpl-tool-<hex>   {"file_path": ...}  </tool_call>`). The registered extractor's regex required the argument-begin token, so nothing was lifted: complete sections were swallowed with no `tool_use` emitted, unclosed sections at stream end leaked the bare id + args JSON as visible text, and every turn ended `end_turn` — Claude Code showed the tool call as a raw string and stalled. Added a second extractor for the bare-args variant to `_INLINE_TOOL_EXTRACTORS` (the appendable registry built for exactly this), and both the non-streaming path and the stream-end flush now strip whole lifted sections (id + args included) from visible text instead of only the control tokens.

### Added
- **Request body size cap** (`src/core/config.py`, `src/main.py`): request bodies over `MAX_REQUEST_BODY_BYTES` (default 8 MiB) are rejected with `413 request_too_large`, guarding uvicorn against buffering arbitrarily large client bodies.
- **Embedding/rerank models filtered from chat surfaces** (`src/core/model_catalog.py`, `src/api/endpoints.py`): Nebius returns embedding and reranking models from `/models` that cannot serve chat completions or tool calls; they are now excluded from `/v1/upstream-models`, the session picker, and `/v1/models` so a chat-incapable model can't be selected.
- **Langfuse dashboard deep-links**: the Langfuse trace id is now persisted on
  each request row and surfaced in `/api/observability/requests`. The
  observability dashboard renders a "Trace" link on every request row that
  opens the corresponding Langfuse trace (`{host}/project/{projectId}/traces/{traceId}`).
  Set `LANGFUSE_PROJECT_ID` in `.env` to enable links; the link hides cleanly
  when Langfuse is off, the project id is unset, or a row predates tracing
  — never a broken link. The `/api/observability/config` response now includes
  a `langfuse` block (`enabled` / `configured` / `host` / `project_id`).
  Zero-overhead when `LANGFUSE_ENABLED=false` (traces are never started).
- **`claude --proxy --bypass`** (and `codex`/pwsh equivalents): non-interactive
  proxy launch for agent-to-agent orchestration — skips the session/model/
  ensemble prompts and uses defaults (session `Agent2Agent`, model = `.env`
  `MODEL`, ensemble off). `--bypass` must be the second arg, after `--proxy`;
  remaining args (e.g. `--dangerously-skip-permissions`) pass through.
- **Per-session runtime model switching via the status line** (`src/core/session_settings.py`, `src/api/endpoints.py`, `src/observability/routes.py`, `src/observability/static/pick.html`, `scripts/install_ui/`): the model name in the proxy status line is now an OSC 8 terminal hyperlink that opens a per-session model picker page (`GET /dashboard/pick?session=<name>`) in your browser. Picking a model `PUT /v1/session-model` stores a per-session runtime override; the next `/v1/messages` turn uses it (the model is resolved fresh per request via `resolve_session_settings()`, where the override wins over the forwarder's `x-session-model` header, which wins over the global default). Concurrent sessions stay isolated; the override is in-memory and reverts to the session's startup model on proxy restart. The `claude --proxy` shell function now exports `NEBIUS_SESSION_NAME` so the status-line link targets the current session. Requires a hyperlink-capable terminal (iTerm2/Kitty/WezTerm/Ghostty); Terminal.app and tmux/SSH may strip the sequence — the picker is also reachable directly by URL. Re-run `./install.sh` to pick up the new shell function + statusline command.
- **Reinstall (overwrite) existing shell functions** (`scripts/install_ui/screens.py`, `scripts/install_ui/utils.py`): a new checkbox on the installer's final step, "Reinstall (overwrite) existing shell functions", makes already-configured profiles selectable. Checking one and applying backs up the rc file (`<rc>.bak.<mtime>`) and replaces the old Claude+Codex function block with the current version (idempotent — in-place swap, no duplication). Lets a `./install.sh` re-run upgrade the shell function on an already-installed machine instead of silently skipping the stale version.
- Dynamic model catalog: pricing, context limits, and model listing are fetched from the provider's `/v1/models?verbose=true` (cached, refreshed hourly). Cost is now correct for any per-session model, including ones not in MODEL_PRICES_JSON.
- Per-session model and ensemble selection: `claude --proxy` now prompts for a
  backend model and optional ensemble config (mode/judge/models), applied via
  injected `x-session-*` headers to the shared proxy.
- `GET /v1/upstream-models` returns raw upstream model ids for the session picker.
- **Ensemble Leaderboard** on the dashboard global view — per-model win/loss aggregates across all ensemble races in the selected window: races, wins, win %, **user picks**, timeout wins, average score, average latency, and errors. Backed by `fetch_ensemble_leaderboard` and `/api/observability/ensemble/leaderboard`.

- **Langfuse observability integration** (`src/langfuse_integration/`)
  - Self-hosted (docker-compose) or cloud Langfuse support. Mounts a
    `langfuse-web` + `postgres` container stack alongside the proxy.
  - `LANGFUSE_ENABLED` env switch (default `false`) — when false, all calls are
    zero-overhead no-ops.
  - Every `/v1/messages` and `/v1/responses` request creates a Langfuse trace +
    generation carrying the full input/output/usage/error context including:
    prompt text, tool calls, streaming status, backend model, latency, and
    cached-token accounting.
  - Dual-write to SQLite dashboard is preserved unchanged — Langfuse captures
    training data; SQLite stays for live ops.
  - Documented in `src/langfuse_integration/README.md` with setup steps for
    generating keys and bringing up the compose stack. Includes an end-to-end
    test walkthrough (secrets → `docker compose up` → create project →
    `curl` a request → verify the trace landed).
  - Contiguous port layout: proxy `8083`, Langfuse web `8084`, Postgres `8085`
    (host ports; internal container ports unchanged). `NEXTAUTH_URL` defaults
    to `http://localhost:8084`.
  - 36 new tests covering config, client lifecycle, and endpoint integration.
- **Ensemble Leaderboard** on the dashboard global view — per-model win/loss aggregates across all ensemble races in the selected window: races, wins, win %, **user picks** (human-chosen winners, the trust signal), timeout wins, average score, average latency, and errors. Backed by a new `fetch_ensemble_leaderboard` aggregate query and the `/api/observability/ensemble/leaderboard` endpoint.
- Multi-theme selector (5 themes): **Lime** (default, cyberpunk), **Navy & White** (corporate), **Mono** (high-contrast), **Star Wars** (Imperial/Sith), **KTM** (orange/black). Replaced binary light/dark toggle in `dashboard.css`, `dashboard.js`, `dashboard.html`.
- `/dashboard/health` endpoint — lightweight availability check returning `{"status": "ok", "version": "1.0"}`, useful for load balancers and uptime monitors.
- `Cache-Control: public, max-age=3600, must-revalidate` headers on dashboard static assets (`/dashboard/assets/*`).

### Quality of Life
- Dashboard root (`/dashboard`) content-type is now explicitly `text/html; charset=utf-8`.
- TUI installer wizard streamlined from 10 → 8 steps. Consolidates Shell, Codex CLI, and Claude Code configuration into a single "Configure Your Setup" screen with checkboxes — no more navigating four separate screens to enable what you want. Removed `ShellScreen`, `CodexConfigScreen`, and `StatuslineScreen`; new `ConfigurationScreen` groups everything together.
- Shell shortcuts now detect **all available profiles** (`.bashrc`, `.zshrc`, PowerShell `$PROFILE`, Warp) and shows a checkbox for each — user can pick individually which profiles to write to.
- Docker commands now appear on the **Done screen** (as an alternative run option), not as a config checkbox. The Smoke Test already has the proxy running, so Docker wouldn't make sense as "configure" option.

### Fixed
- **Statusline now shows the per-session model, not the global one.** Sessions can override their backend model via the `x-session-model` header (injected by the per-session forwarder). `GET /api/observability/config` now returns a top-level `effective_model` resolved from the request headers (falling back to the global `config.model` when no session header is present), and the installer-generated statusline prefers `effective_model` over `configured_models.big`. The statusline now reads the per-session model directly from the `NEBIUS_SESSION_MODEL` env var exported by the `claude --proxy` launcher, falling back to the `/api/observability/config` endpoint only when that var is unset.
- **A single malformed tool-call argument no longer kills the whole Codex session.** Models (e.g. Kimi) sometimes emit a tool call whose `arguments` aren't valid JSON — most often a large `exec_command`/heredoc writing a file with unescaped quotes or newlines (e.g. a drawio XML diagram). Replayed in conversation history, the backend fails to parse that one call and rejects the **entire** request with `400 Unterminated string …`, so the session "ran fine then stopped at the end." `convert_responses_to_openai_chat` now sanitizes replayed tool-call arguments (`_sanitize_tool_call_arguments`): any argument that isn't valid JSON is wrapped in a valid `{"_raw_arguments": "…"}` object. The call already executed (its result is preserved in history), so the conversation stays structurally valid and one bad call can't poison every subsequent turn.
- **Codex streams no longer disconnect on an upstream error, and leaked `web_search` calls no longer 400 the backend.** Two linked defects surfaced once search requests started streaming: (1) the backend rejected requests whose history contained a `web_search` tool call — when the model batches `web_search` with a client tool, the proxy can't intercept it, so it leaks to Codex ("unsupported call: web_search") and its malformed, double-escaped `arguments` come back on the next turn and 400 the backend; (2) because the 400 was raised *inside* the SSE generator after `response.created` was already sent, it became `RuntimeError: Caught handled exception, but response already started` and the client connection dropped. Fixes: `convert_responses_to_openai_chat` now strips server-owned search tool calls (and their orphaned outputs) from replayed history (`_strip_leaked_search_calls`); `convert_openai_sse_to_responses_sse` now reads the upstream defensively and, on any mid-stream error, surfaces it as assistant text and closes with a valid terminal event sequence instead of disconnecting (recorded as an error in observability/Langfuse).
- **Codex `/v1/responses` no longer freezes when server-side web search is enabled.** With `SERVER_SEARCH_ENABLED=true` (the default) and a Tavily key, every Codex turn that advertises a `web_search` tool was routed through the non-streaming `run_search_loop`, which buffered the model's *entire* generation before sending anything to the client. Short turns were tolerable, but a long generation produced a multi-second blank wait (observed: 63s) that Codex experienced as a hang/"stopped". Added `run_search_loop_streaming()`: content/reasoning tokens stream to the client live, while turns whose tool calls are *all* server-owned searches are still executed server-side and fed back invisibly (same contract as `run_search_loop`). The non-streaming client path is unchanged.
- **Langfuse generations now carry model name, token usage, AND cost in one observation.** Previously each request created two generation observations — one with model/input (orphaned), one with usage/output (no model) — because the helper `_langfuse_start()` and each call site both called `start_generation()`. Removed the duplicate call; model + input are now passed through `start_generation()` if known upfront, or via `end_generation()` when resolved later. Also wired `PricingCatalog.quote()` into `end_generation(…, cost_details=…)` so Langfuse's per-generation cost column is populated from the proxy's `MODEL_PRICES_JSON`. Codex streaming path no longer drops usage (`None` → reads from `accumulator["usage"]`).
- `pyproject.toml` `requires-python` bumped to `>=3.10` and `python_version` to `3.10` — `langfuse>=4.0.0` requires Python 3.10+. Dockerfile now installs and pins CPython 3.12 via `uv python install`.
- `tests/test_main.py` no longer calls `load_dotenv()` at module import (`.env` leaked into other tests, breaking the Langfuse noop-client tests when `LANGFUSE_ENABLED=true` was set).
- Missing `from src.core.config import config` in `tests/test_observability.py` — the `client` fixture monkeypatches `config.ignore_client_api_key` but `config` was never imported, causing `NameError` at test collection.
- Fixed 400 error "When using `tool_choice`, `tools` must be set" from Codex requests. The codex request converter was unconditionally adding `tool_choice` to OpenAI requests even when `tools` was absent, causing Nebius backend to reject the request. Now `tool_choice` is only forwarded when tools are present.
- Streaming responses now report real provider token usage instead of zeros.
  The stream converter exited at `finish_reason` before the trailing
  empty-choices usage chunk (sent by `stream_options.include_usage`) arrived;
  it now drains the stream until `[DONE]`, so `message_delta` carries actual
  `input_tokens`/`output_tokens`/`cache_read_input_tokens`.
- Cached tokens are no longer double-counted in usage. Anthropic semantics:
  `input_tokens` excludes cached tokens (clients sum
  `input + cache_read + cache_creation` for context size), while OpenAI-style
  `prompt_tokens` includes them. The proxy now reports the uncached remainder
  as `input_tokens` and stops synthesizing `cache_creation_input_tokens`.
  Observability `total_tokens` and cost estimates still bill the full prompt.

### Added
- Ensemble streaming (hedge racing): `ENSEMBLE_MODE=hedge|approval` races one
  request across `ENSEMBLE_MODELS` in parallel and returns the best response.
  Candidates are scored on tool-call validity, finish_reason, and speed; a
  single healthy model keeps the session alive even when others 429/500.
  `approval` mode holds the Claude Code stream (with pings) while the user
  picks the winner on the dashboard (`ENSEMBLE_APPROVAL_TIMEOUT_S` fallback
  to the auto-winner). Every race is recorded per candidate (output, verdict
  reasons, latency, who chose) in a new `ensemble_candidates` table, shown as
  a split view under each session in the dashboard with Continue-with-this
  buttons and a global pending-approval banner. New endpoints:
  `GET /api/observability/ensemble/runs`, `GET .../ensemble/pending`,
  `POST .../ensemble/choose`. Default `ENSEMBLE_MODE=off` — behavior is
  unchanged unless enabled. Files: `src/ensemble/`, `src/api/endpoints.py`,
  `src/observability/`.
  - The racer takes precedence over the server-side search branch and runs
    the Tavily search loop per candidate — Claude Code's main loop always
    offers WebSearch, so without this every real turn bypassed the race
    (and approval mode never held).
  - Approval holds only streamed requests that offer tools; tool-less
    housekeeping probes (title generation, quota checks) pass through
    immediately instead of stalling for the approval timeout.
  - Optional `ENSEMBLE_JUDGE_MODEL`: an LLM judge (same Token Factory key,
    any catalog model) breaks rule-score ties and records its reasoning on
    the winner's card; judge failure falls back to rules silently.
  - Winner cards always carry an explicit `decision:` reason (higher score /
    score tie + latency gap / judge verdict); dashboard cards show token
    counts, a "(no text output)" placeholder, and collapsible full output.
- Dynamic context-window mapping: live usage and `count_tokens` are reported
  in the selected Claude model's window units
  (`claude_window / backend_window`, with 1M detected from the `[1m]` model
  suffix or `anthropic-beta: context-1m-*` header). Claude Code's native
  auto-compaction now fires when the *backend's* real window is filling —
  for any Claude-model/backend pairing — instead of overflowing the backend
  (the "exceeded the 128000 output token maximum" death loop after context
  fills). `output_tokens` is intentionally not scaled
  (CLAUDE_CODE_MAX_OUTPUT_TOKENS guards that field); observability keeps raw
  backend tokens so the dashboard and statusline stay truthful.
- Forward `metadata.user_id` upstream as `user` (per-session attribution) and
  `prompt_cache_key` (prefix-cache routing affinity for parallel subagent
  requests). Verified accepted by Nebius Token Factory; skipped on Azure.
- Forward Claude `top_k` via `extra_body` (vLLM-style backends honor it;
  skipped on Azure).
- Rate-limit pacing: retry backoff now honors the upstream `Retry-After` /
  `x-ratelimit-reset-requests` hint (clamped to 30s) instead of blind
  exponential backoff, and propagated 429s carry a `retry-after` header.
- Streaming errors now surface typed Anthropic error events
  (`rate_limit_error`, `authentication_error`, `overloaded_error`, …) instead
  of a generic `api_error` or a dropped connection, so Claude Code's retry
  and backoff logic engages correctly.
- `UvicornAccessFilter` logging module that suppresses noisy 200 OK access logs for dashboard observability endpoints. At `WARNING` level all successful requests are hidden; at `INFO` level only dashboard polls are filtered, keeping errors visible. `src/core/logging.py`.
- `/v1/models` now dynamically fetches upstream provider models and appends them to the response. `OpenAIClient.list_models()` calls the backend `/v1/models`, and the endpoint merges discovered models alongside the existing Claude aliases and custom env mappings.
- Codex server-side web search (Tavily). When `TAVILY_API_KEY` is set, `web_search` built-in tools from Codex CLI are promoted to OpenAI function tools, the proxy injects `SEARCH_TOOL_SYSTEM_SUPPLEMENT` into the system prompt, and `run_search_loop()` executes the search server-side (same pattern as the Claude Code path). A new `codex_response_to_sse()` generator converts the final non-streaming response into synthetic SSE events for streaming clients. Files: `src/codex/tools_compat.py`, `src/codex/request_converter.py`, `src/codex/stream_converter.py`, `src/api/endpoints.py`.
- `docs/codex/CODEX_STATUSLINE.md` — documentation for Codex CLI proxy routing (`openai_base_url`, `model_provider`) and statusline configuration (TOML `tui.status_line`), plus a shell-prompt workaround for live context-usage display.
- Codex proxy tool compatibility (Unit 2): `CodexToolContext` and `tools_compat.py` with parsing, conversion, and remapping for string tools, custom tools (multi-suffix proxy functions), namespace tools (flatten/unflatten), built-in tools (`web_search`/`local_shell`/`computer_use`), and standard function passthrough.
- Codex proxy stream converter (Unit 5): OpenAI SSE → Responses API SSE events with state machine for text/tool buffering, event ordering, and usage accumulation.
- Codex proxy response converter (Unit 4): OpenAI Chat Completion → Responses API output items, usage mapping, and tool name remapping.
- Codex proxy request converter (Unit 3): Responses API → OpenAI Chat Completions (instructions→system, input items→messages, reasoning effort, model mapping).
- Codex proxy foundation (Unit 1): Pydantic models, config vars, model mapping.
- Codex streaming tool-call completion events (`response.function_call_arguments.done` and `response.output_item.done` for function items).
- Codex streaming session saving: the `previous_response_id` multi-turn chain now works for streaming requests (previously only supported non-streaming).
  configured, the proxy injects a short system-prompt line telling the model to
  call web search on its own turn (not batched with other tools), so it can be
  executed server-side. Scoped — only added when a search tool is present.
- Server-side web search (Tavily). When `TAVILY_API_KEY` is set, the proxy
  executes `web_search`/`WebSearch` tool calls itself in a bounded loop and
  feeds results back to the model, returning the final answer. Fixes "Did 0
  searches" — Claude Code's search can't run behind a non-Anthropic backend.
  Search tools are also forwarded with a real `{query}` schema. Only engaged
  when a search tool is present; all other requests are unchanged. Knobs:
  `SERVER_SEARCH_ENABLED`, `TAVILY_MAX_RESULTS`, `SERVER_SEARCH_MAX_ITERS`.
- Honor `thinking.display` for adaptive thinking (Opus 4.7/4.8). Thinking text
  is surfaced only when `display` is `"summarized"`; adaptive mode defaults to
  `"omitted"` (matching Anthropic), so the backend's reasoning is no longer
  shown unless asked for. Operator override via `THINKING_DISPLAY_OVERRIDE`.
- Strip `<think>…</think>` from visible text (streaming + non-streaming) whenever
  thinking is not being surfaced, so provider reasoning never leaks as assistant
  text. Fixes the "reasoning visible in output" community reports.
- Dynamic effort forwarding: the effort chosen in Claude Code (`/effort`,
  carried in `output_config.effort`) is automatically mapped to a backend
  `reasoning_effort` (xhigh/max -> high) — no configuration. If a backend
  rejects `reasoning_effort`, the proxy strips it, retries once, and remembers
  not to send it to that model again (self-healing, no repeated latency).
- Surface a model's separate reasoning channel (`reasoning_content` / `reasoning`,
  as emitted by DeepSeek-R1, Qwen, GLM-thinking, etc.) as Claude `thinking`
  content blocks — both streaming and non-streaming.
- Accept inbound `thinking` / `redacted_thinking` blocks in assistant history
  (interleaved thinking). They are parsed without error and dropped during
  conversion, since OpenAI-compatible backends cannot consume them.
- Opt-in reasoning passthrough so reasoning-capable backends actually think:
  `REASONING_EFFORT` (operator override) and `MAP_THINKING_BUDGET_TO_EFFORT`
  (bucket a client `thinking.budget_tokens` into an effort level). Both default
  to no-op, so non-reasoning backends are unaffected.
- `docs/GAP_ANALYSIS_SPEC.md` — gap analysis vs. the current Claude Code CLI and
  a roadmap for agentic/harness work.
- `tests/test_thinking_and_reasoning.py` — unit coverage for the thinking,
  reasoning, and stop-reason changes.

### Fixed
- Codex CLI tool calls are no longer stripped as "unknown tool types". The CLI sends tools with type keys like `text_editor`, `exec_command`, `apply_patch`, etc. These are now converted to OpenAI-style function tools instead of being dropped, so the backend model can correctly emit structured tool calls. `parse_codex_tools()` in `src/codex/tools_compat.py` gains a `_KNOWN_CODEX_TOOL_TYPES` set for this.
- Codex CLI embedded tool call text is now parsed into real `function_call` events. When the model emits tool calls inline within text (Codex CLI format), the proxy extracts them from the content stream via `_extract_tool_calls_from_text()` and emits proper `function_call` events. This happens in the live streaming path (`convert_openai_sse_to_responses_sse`) and the non-streaming response path (`convert_openai_to_responses`).
- Empty `tools: []` array is now dropped from the upstream request. When all Codex tools are stripped (e.g. all builtins with no Tavily config), the proxy no longer sends `tools: []` to the backend, which caused 400 Bad Request from some providers. `convert_responses_to_openai_chat()` now checks whether `tool_ctx.tools` is truthy before adding it to the dict.
- Server-search path no longer drops tool calls. The non-streamed loop result
  was being streamed back via a text-only serializer, so on turns that merely
  *offered* `WebSearch` the model's real tool call (Read/Edit/Agent/...) was
  silently dropped -> "The model's tool call could not be parsed." Added a
  tool_use/thinking-aware SSE serializer (`claude_response_to_sse`) for that path.
- Self-heal on upstream context-length 400s: the proxy now detects when a
  backend rejects a request for exceeding its context window, sheds the oldest
  messages with the existing safe trimmer (system prompt, latest turn, and tool
  pairs preserved) and retries once, instead of surfacing a hard error. If it
  still doesn't fit, a clear message points at `<ROLE>_MODEL_CONTEXT_LIMIT`.
- Strip Kimi-K2's native tool-call control tokens (`<|tool_call_begin|>`,
  `<|tool_call_argument_begin|>`, `functions.NAME:N`, ...) that leak into tool
  arguments when a tool is forwarded without a real parameter schema (e.g. the
  Anthropic `web_search` server tool). The inner JSON is now extracted so the
  client receives clean arguments instead of a token blob.
- Extended-thinking config now understands Anthropic's real wire shape
  `{"type": "enabled"|"disabled", "budget_tokens": N}` (via `is_enabled()`),
  in addition to the legacy `{"enabled": bool}`. Previously `{"type":"disabled"}`
  was ignored and thinking stayed on, and `budget_tokens` was dropped.
- Requests whose assistant history contained `thinking` blocks no longer return
  HTTP 422.
- `thinking.type` accepts any mode string (e.g. `adaptive`), not just
  `enabled`/`disabled`. A strict enum was 422'ing real Claude Code requests that
  send newer thinking modes; only `disabled` turns thinking off.
- A provider `content_filter` finish reason now maps to the Claude `refusal`
  stop reason instead of masquerading as `end_turn`.
- Codex non-streaming tool-call status set to `"completed"` instead of `"in_progress"`.
- Codex session item ordering corrected from `output_items + input_items` to `input_items + output_items` (chronological conversation history).
- Dead code branch removed from Codex request converter.

### Changed
- `MODEL_PRICES_JSON` is now an optional override/fallback rather than the primary price source.
- Collapsed `BIG_MODEL`/`MIDDLE_MODEL`/`SMALL_MODEL` into a single `MODEL`
  (and `*_MODEL_CONTEXT_LIMIT` into `MODEL_CONTEXT_LIMIT`). `VISION_MODEL`
  unchanged. `ENSEMBLE_*` env values are now per-session defaults.
- `.env.example` default `LOG_LEVEL` changed from `INFO` to `WARNING` with expanded documentation describing the four modes (`DEBUG` / `INFO` / `WARNING` / `ERROR`).
- Replaced deprecated `[tool.uv.dev-dependencies]` in `pyproject.toml` with the standard `[dependency-groups.dev]` section. Eliminates the UV deprecation warning during package builds.
- Added `refusal`, `pause_turn`, and `model_context_window_exceeded` stop-reason
  constants (only `refusal` is emitted today; the others are reserved for
  server-tool / upstream-error wiring).

### Removed
- Reverted an experimental 1M-context `betas` override. Context window size
  remains owned solely by the per-model `*_MODEL_CONTEXT_LIMIT` settings, which
  are also what the statusline's `/api/observability/context-usage` endpoint
  reads (capped at 200K to match Claude Code). To run a 1M-capable model, set
  its `*_MODEL_CONTEXT_LIMIT` accordingly.

## [1.0.0]

- Initial baseline: Claude `/v1/messages` proxy to Nebius OpenAI-compatible
  endpoints, with streaming SSE, model routing (big/middle/small/vision),
  tool-call JSON repair, local request optimizations, and the observability
  dashboard. (Pre-changelog history is in the git log.)
