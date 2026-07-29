# TODO / Backlog

## Native Responses API for the Codex path (investigate)

Nebius Token Factory now natively supports the OpenAI **Responses API**
(`client.responses.create(...)`), including **streaming** and **function/tool
calls** (verified 2026-06-26: returns native `reasoning` + `function_call`
output items and proper Responses SSE events for `zai-org/GLM-5.2`).

**Why it could help (Codex path only):** today the proxy *emulates* the
Responses API on top of `chat.completions` for Codex CLI via ~70KB of
conversion code — the fragile core is `src/codex/stream_converter.py` (27KB),
which synthesizes `function_call` events by scraping text deltas
(`_extract_tool_calls_from_text`). Native Responses support could replace most
of that with a near-passthrough, improving tool-call and reasoning-model
fidelity and unlocking Responses-native state (`previous_response_id`).

**Does NOT help** the main Claude Code path (`/v1/messages` ↔ `chat.completions`).

**The catch:** cross-cutting features are wired on the `chat.completions` leg —
per-session model mapping, observability/Langfuse, cost (dynamic catalog),
session headers, ensemble, server-side Tavily search. A naive passthrough would
regress all of these. Real work = swap the backend leg to `responses.create`
**and** re-wire those features around native Responses + a Responses-native
streaming handler.

**Options:**
1. Full switch — retire most converters; biggest payoff, biggest change.
2. Hybrid behind `CODEX_NATIVE_RESPONSES=true` — keep converter path as
   fallback, add native path, compare, flip when confident. (Recommended:
   lower risk.)

**Status:** parked. Pursue via brainstorm → spec → plan when prioritized.
