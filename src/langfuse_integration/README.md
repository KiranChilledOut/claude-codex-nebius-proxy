# Langfuse Integration

Captures every proxied LLM call (full prompt + response + usage + tool calls)
as a Langfuse trace. This is the data you'll export later as a fine-tuning
dataset — the SQLite dashboard deliberately does not store full prompt/response
text, so Langfuse fills that gap.

## When it runs

| `LANGFUSE_ENABLED` | Keys set | Behavior |
|---|---|---|
| `false` (default) | — | All calls are zero-overhead no-ops. Proxy unchanged. |
| `true` | no | Warning logged at startup; calls are no-ops. |
| `true` | yes | SDK initialized; traces sent to `LANGFUSE_HOST`. |

The Langfuse Python SDK (v4, OpenTelemetry-based) queues events internally
and flushes them on a background timer (`LANGFUSE_FLUSH_INTERVAL`), so the
proxy's request latency is unaffected.

## Setup (self-hosted, Docker)

## Port map

All stack ports live in one contiguous range:

| Service       | Host port | Container port |
|---------------|-----------|----------------|
| Proxy         | 8083      | 8083           |
| Langfuse web  | 8084      | 3000           |
| Postgres      | 8085      | 5432           |

Use **8084** for everything you reach from your browser/Terminal on the host
(`LANGFUSE_HOST`, `NEXTAUTH_URL`). The proxy container reaches Langfuse
internally on `http://langfuse-web:3000` via `LANGFUSE_HOST_DOCKER`.

## Setup (self-hosted, Docker)

1. Bring up the Langfuse stack:

   ```bash
   docker compose up -d langfuse-web postgres
   ```

2. Open `http://localhost:8084`, create an account + project, and copy the
   public + secret API keys from **Project Settings → API Keys**.

3. Add to `.env`:

   ```ini
   LANGFUSE_ENABLED=true
   LANGFUSE_HOST=http://localhost:8084           # host / browser
   LANGFUSE_HOST_DOCKER=http://langfuse-web:3000 # proxy container (set in compose)
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   # Optional — enables deep-links from the proxy dashboard to Langfuse traces.
   # Find it in the Langfuse UI: select your project, copy the id from the URL bar
   # (https://<host>/project/<project_id>/...). Unset = dashboard rows show no link.
   LANGFUSE_PROJECT_ID=
   ```

   Also generate the Langfuse container secrets:
   ```bash
   openssl rand -base64 32  # → NEXTAUTH_SECRET, SALT, ENCRYPTION_KEY
   ```

4. Restart the proxy. On startup you'll see:
   ```
   Langfuse: enabled (host=http://langfuse-web:3000, keys=configured)
   ```
   and traces will appear in the Langfuse UI at `http://localhost:8084` as
   requests flow through.

## What's captured per request

- **Trace** (id = proxy `request_id`): metadata with claude_model,
  backend_model, stream flag, session id/name, tags `["proxy", "v1-messages"]`.
- **Generation**: full input prompt (converted OpenAI messages), output
  response or stream metrics, usage (input incl. cache tokens / output / total),
  model, and error status/message on failure.
- **Event** (`ensemble-race`): when ensemble mode races candidates, an event
  records mode, chosen-by (auto/user/timeout), winner model, candidate count.

All Langfuse calls are wrapped in try/except — a Langfuse outage never breaks
a proxy request; the worst case is a missing trace.

## Architecture

The integration lives in `src/langfuse_integration/`:

- `config.py` — env-driven `LangfuseConfig`.
- `client.py` — `LangfuseClient` wrapping the v4 SDK; lazy init, no-op when
  disabled. Exposed as a process-wide singleton via `get_langfuse_client()`.
- `endpoints.py` instrumented with `_langfuse_start()` / `end_generation()`
  calls alongside the existing `observability_recorder.record_request()` paths,
  so SQLite + Langfuse both fire on every request.

## Training data export

Once you have enough volume, use Langfuse's **Datasets** feature: filter traces
by tag/model/quality score, annotate or corrected-output them, and export to
JSONL for fine-tuning. The `score_generation()` client method is the hook for
attaching automated or human quality signals.

## End-to-end testing (bring up the full stack and verify)

This is a concrete, copy-paste sequence that gets the whole stack running and
proves traces are landing:

### 1. Prepare secrets

If you don't already have the Langfuse container secrets in `.env` (or don't
have a `.env` at all), copy from the example and generate:

```bash
cp .env.example .env
```
```bash
# macOS one-liner — generates 3 fresh secrets and writes them into .env
for key in NEXTAUTH_SECRET SALT ENCRYPTION_KEY; do
  secret=$(openssl rand -base64 32)
  sed -i '' "s|^${key}=.*|${key}=${secret}|" .env
done
```

### 2. Start Langfuse + Postgres

```bash
docker compose up -d langfuse-web postgres
```

Wait for the Langfuse web container to be ready (watch `docker compose logs -f` or
just give it ~10 seconds). You can confirm it at `http://localhost:8084`.

### 3. Create a project and get API keys

1. Open `http://localhost:8084` in a browser.
2. Register a new account (first-time only — the account is local to your
   self-hosted Postgres).
3. Once logged in, go to **Settings → API Keys** (or click **"New Project"**
   and then **Project Settings → API Keys**).
4. Note down the **Public Key** (`pk-lf-...`) and **Secret Key** (`sk-lf-...`).
5. Paste them into `.env`:

   ```ini
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   ```

### 4. Start the proxy

```bash
docker compose up -d proxy
```

Check the logs for the Langfuse startup banner:
```bash
docker compose logs proxy | grep -i langfuse
# Expected: Langfuse: enabled (host=http://langfuse-web:3000, keys=configured)
```

### 5. Send a test request

```bash
curl http://localhost:8083/v1/health
# Expected: {"status": "ok"}
```

```bash
curl -s http://localhost:8083/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6",
    "max_tokens": 50,
    "messages": [{"role": "user", "content": "Say hello in one word."}]
  }' | jq .
```

### 6. Verify the trace landed in Langfuse

Go to `http://localhost:8084` → click your project → **Traces**. You should
see a trace named `proxy-request` with a generation containing the prompt,
the response, usage tokens, and model name. Expand the trace — you'll see the
full input/output as structured JSON.

### Troubleshooting

| Symptom | Check |
|---|---|
| No traces in Langfuse | `docker compose logs proxy \| grep -i langfuse` — look for "enabled" or error lines |
| "keys are empty" in logs | `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` not set in `.env` |
| Langfuse web won't start | `NEXTAUTH_SECRET`/`SALT`/`ENCRYPTION_KEY` missing — run the `openssl rand` loop |
| `400 Bad Request` from proxy | Make sure `OPENAI_API_KEY` is set to a valid Nebius Token Factory key |
| Tests pass but Langfuse SDK not installed | `docker compose up proxy` rebuilds with `langfuse>=4.0.0`; the SDK is inside the container. Host-only test runs may need `pip install langfuse>=4` |

### Running the full test suite

```bash
# Host (requires pytest + dependencies installed in the worktree):
python -m pytest tests/ -q

# If langfuse SDK isn't installed in the host venv, the SDK-absent no-op tests
# still pass (~330 tests).
```

### Tearing down (keep data)

```bash
docker compose down
```

The Postgres volume `langfuse-postgres-data` survives `down`. To start fresh:

```bash
docker compose down -v   # destroys volumes too — wipes all Langfuse data
```
