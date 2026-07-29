# Architecture

## Overview

This project exposes a Claude-compatible API surface for Claude Code and forwards requests to Nebius-hosted OpenAI-compatible models.

It acts as a standalone Claude-compatible API surface for Claude Code.

## High-Level Flow

```text
Claude Code
  └─ Claude API request -> Proxy (`POST /v1/messages`)

Proxy
  ├─ request conversion: Claude -> OpenAI-compatible payload
  ├─ model routing: text vs vision / small vs medium vs large
  └─ response conversion: OpenAI SSE -> Claude SSE

Nebius
  └─ OpenAI-compatible inference endpoint
```

## Key Files

| Path | Purpose |
| --- | --- |
| `src/main.py` | FastAPI entry point |
| `src/api/endpoints.py` | HTTP route handling |
| `src/core/config.py` | environment-driven config |
| `src/core/model_manager.py` | model selection and routing |
| `src/conversion/request_converter.py` | Claude request -> OpenAI request |
| `src/conversion/response_converter.py` | OpenAI response -> Claude SSE |
| `src/conversion/computer_use.py` | schema-less tool conversion |
| `start_proxy.py` | local convenience launcher |

## Model Configuration

Core environment variables:

```bash
OPENAI_API_KEY=<nebius-key>
OPENAI_BASE_URL=https://api.tokenfactory.nebius.com/v1
MODEL=moonshotai/Kimi-K2.6
VISION_MODEL=Qwen/Qwen2.5-VL-72B-Instruct
```

## Request Lifecycle

1. Claude Code sends a Claude-compatible request to `/v1/messages`.
2. `request_converter.py` maps the request into OpenAI chat-completions format.
3. Schema-less Claude Code tools are converted into explicit JSON-schema tools.
4. The request is sent to the configured Nebius endpoint.
5. OpenAI-format streaming chunks are received.
6. `response_converter.py` converts them into Claude SSE events.
7. Claude Code receives a native Claude-style response stream.

## Per-session model & ensemble

`claude --proxy` prompts for a session name, model, and ensemble config, then
spawns `session_forwarder.py`, which injects per-session headers on every
request to the shared proxy:

- `x-session-name` — session label (observability/dashboard grouping)
- `x-session-model` — backend model for this session (overrides the tier map)
- `x-session-ensemble-mode` — `off` | `hedge` | `approval`
- `x-session-ensemble-models` — comma-separated model ids
- `x-session-ensemble-judge` — judge model id

The proxy resolves these per request via `resolve_session_settings()`
(`src/core/session_settings.py`), falling back to the `MODEL` / `ENSEMBLE_*`
env defaults when a header is absent. Global config is never mutated, so
concurrent sessions stay isolated. Image requests always use `VISION_MODEL`.

## Model catalog

Model metadata (pricing, context length, available model ids) is fetched from
the provider's `GET /v1/models?verbose=true` at startup, then refreshed in the
background every `MODEL_CATALOG_REFRESH_SECONDS` seconds (default 3600).

The catalog is cached in memory and managed by `src/core/model_catalog.py`.
Request-path reads are non-blocking — only the background refresh task does
network I/O.

Pricing resolution order for a given model:

1. `MODEL_PRICES_JSON` override (explicit entry wins)
2. Live catalog (from the provider's verbose model list)
3. `MODEL_PRICES_JSON` `default`/`*` entry, if set; otherwise cost is reported as unknown (`None`)

Context limits follow the same order, falling back to the catalog's
`context_length` field when no `*_MODEL_CONTEXT_LIMIT` env var is set.

`GET /v1/upstream-models` is served directly from the catalog (no extra
network call per request).

If the provider is unreachable the catalog falls back gracefully: it retains
the last-good in-memory cache, then `MODEL_PRICES_JSON`, then the static
default list — requests are never blocked waiting for a catalog refresh.
