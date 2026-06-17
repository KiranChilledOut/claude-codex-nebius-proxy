# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Convention:** every change-worthy PR adds an entry under `[Unreleased]` in the
> matching group (`Added` / `Changed` / `Fixed` / `Removed` / `Deprecated` / `Security`).
> On release, rename `[Unreleased]` to the new version + date and start a fresh
> `[Unreleased]` block.

## [Unreleased]

### Added
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
