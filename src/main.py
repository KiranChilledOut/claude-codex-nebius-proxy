import asyncio
import json
import os
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.requests import Request

from src.api.endpoints import router as api_router
from src.core.config import config
from src.observability.routes import router as observability_router
from src.observability.store import observability_recorder

app = FastAPI(title="Claude-to-OpenAI API Proxy", version="1.0.0")

_catalog_refresh_task = None

app.include_router(api_router)
app.include_router(observability_router)


@app.middleware("http")
async def limit_request_body_size(request: Request, call_next):
    """Reject oversized request bodies with 413 before parsing.

    Uvicorn otherwise buffers arbitrarily large client bodies. A declared
    Content-Length over the cap is rejected immediately; chunked bodies are
    capped as they stream.
    """
    limit = config.max_request_body_bytes
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > limit:
                return JSONResponse(
                    status_code=413,
                    content={
                        "type": "error",
                        "error": {
                            "type": "request_too_large",
                            "message": f"Request body exceeds {limit} bytes",
                        },
                    },
                )
        except ValueError:
            pass
    else:
        # No declared length (chunked/streamed): count while reading.
        body = await request.body()
        if len(body) > limit:
            return JSONResponse(
                status_code=413,
                content={
                    "type": "error",
                    "error": {
                        "type": "request_too_large",
                        "message": f"Request body exceeds {limit} bytes",
                    },
                },
            )
        # Re-inject the consumed body so downstream handlers can read it.
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(request.scope, receive)
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def log_validation_error(request: Request, exc: RequestValidationError):
    """Log 422 validation errors. Body content is NOT printed — request
    bodies carry full prompts, which must not leak into logs. Only the path
    and the (metadata-only) validation errors are recorded."""
    print("=== 422 validation error ===", flush=True)
    print("path:", request.url.path, flush=True)
    try:
        print("errors:", json.dumps(exc.errors(), default=str, indent=2)[:4000], flush=True)
    except Exception as e:
        print("errors (repr):", repr(exc.errors())[:4000], flush=True)
    print("=== end 422 ===", flush=True)
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.on_event("startup")
async def startup_event():
    await observability_recorder.start()
    # Prime the model catalog (pricing/context/listing) before serving, then
    # refresh on a timer. Never crash if the provider is unreachable.
    from src.core.model_catalog import model_catalog

    if config.model_catalog_enabled:
        ok = await asyncio.to_thread(model_catalog.refresh)
        print(
            f"   Model catalog: {'loaded ' + str(len(model_catalog.model_ids())) + ' models' if ok else 'unavailable (using MODEL_PRICES_JSON fallback)'}",
            flush=True,
        )

        async def _refresh_loop():
            while True:
                await asyncio.sleep(config.model_catalog_refresh_seconds)
                await asyncio.to_thread(model_catalog.refresh)

        global _catalog_refresh_task
        _catalog_refresh_task = asyncio.create_task(_refresh_loop())
    # Initialise the Langfuse client eagerly if enabled, so the first request
    # doesn't pay the SDK warm-up cost. No-op when disabled/unconfigured.
    try:
        from src.langfuse_integration.client import get_langfuse_client

        lf = get_langfuse_client()
        lf._ensure_client()  # noqa: SLF001 — trigger lazy init
        if lf._client is not None:  # noqa: SLF001
            print(f"   Langfuse: enabled ({lf.config.host})", flush=True)
    except Exception as exc:
        print(f"   Langfuse: init failed — {exc}", flush=True)


@app.on_event("shutdown")
async def shutdown_event():
    global _catalog_refresh_task
    if _catalog_refresh_task is not None:
        _catalog_refresh_task.cancel()
        _catalog_refresh_task = None
    await observability_recorder.stop()
    # Flush + tear down Langfuse background workers so queued traces aren't lost.
    try:
        from src.langfuse_integration.client import get_langfuse_client

        get_langfuse_client().shutdown()
    except Exception:
        pass


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Claude-to-OpenAI API Proxy v1.0.0")
        print("")
        print("Usage: python start_proxy.py [--help|--selftest]")
        print("")
        print("  --selftest   Hit /test-connection in-process and exit 0 if it")
        print("               succeeds, non-zero otherwise. Intended for CI and")
        print("               install scripts.")
        print("")
        print("Required environment variables:")
        print("  OPENAI_API_KEY - Your provider API key")
        print("")
        print("Optional environment variables:")
        print("  ANTHROPIC_API_KEY - Expected Anthropic API key for client validation")
        print("                      If set, clients must provide this exact API key")
        print("  IGNORE_CLIENT_API_KEY - Ignore/drop client API key headers (default: true)")
        print(
            f"  OPENAI_BASE_URL - OpenAI-compatible API base URL (default: {config.openai_base_url})"
        )
        print(f"  MODEL - Backend model for all Claude tiers (default: {config.model})")
        print(f"  VISION_MODEL - Model for image requests (default: {config.vision_model})")
        print(f"  HOST - Server host (default: {config.host})")
        print(f"  PORT - Server port (default: {config.port})")
        print(f"  LOG_LEVEL - Logging level (default: {config.log_level})")
        print(f"  MAX_TOKENS_LIMIT - Token limit (default: {config.max_tokens_limit})")
        print(
            f"  MIN_TOKENS_LIMIT - Fallback token limit for invalid requests (default: {config.min_tokens_limit})"
        )
        print(f"  REQUEST_TIMEOUT - Request timeout in seconds (default: {config.request_timeout})")
        print(
            f"  MAX_RETRIES - Retry attempts for provider requests (default: {config.max_retries})"
        )
        print(
            "  ENABLE_REQUEST_OPTIMIZATIONS - Answer Claude Code housekeeping "
            f"requests locally (default: {config.enable_request_optimizations})"
        )
        print("")
        print("Codex options:")
        print(
            f"  CODEX_ENABLED - Enable /v1/responses endpoint (default: {config.codex_enabled})"
        )
        print(
            f"  CODEX_TOOL_COMPAT - Enable Codex custom/namespace tool conversion (default: {config.codex_tool_compat})"
        )
        print(
            f"  CODEX_SESSION_TTL_SECONDS - Session TTL in seconds (default: {config.codex_session_ttl_seconds})"
        )
        print(
            f"  CODEX_WEBSOCKET_FALLBACK - Return 426 on WebSocket upgrade attempts (default: {config.codex_websocket_fallback})"
        )
        print("")
        print("Model mapping:")
        print(f"  Claude haiku models -> {config.small_model}")
        print(f"  Claude sonnet models -> {config.middle_model}")
        print(f"  Claude opus models -> {config.big_model}")
        print(f"  Requests with images -> {config.vision_model}")
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        # In-process smoke test: invoke /test-connection via Starlette's
        # TestClient (no uvicorn, no port binding) and exit 0/1 based on
        # the result. We deliberately don't enter TestClient as a context
        # manager so FastAPI lifespan handlers (which would open the
        # observability sqlite database) do not fire.
        import json

        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/test-connection")
        try:
            result = response.json()
        except ValueError:
            status_code = response.status_code
            message = f"selftest: non-JSON response (status {status_code}): {response.text}"
            print(message, file=sys.stderr)
            sys.exit(2)
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        sys.exit(0 if result.get("status") == "success" else 1)

    # Configuration summary
    print("🚀 Claude-to-OpenAI API Proxy v1.0.0")
    print(f"✅ Configuration loaded successfully")
    print(f"   OpenAI Base URL: {config.openai_base_url}")
    print(f"   Big Model (opus): {config.big_model}")
    print(f"   Middle Model (sonnet): {config.middle_model}")
    print(f"   Small Model (haiku): {config.small_model}")
    print(f"   Vision Model (images): {config.vision_model}")
    print(f"   Max Tokens Limit: {config.max_tokens_limit}")
    print(f"   Request Timeout: {config.request_timeout}s")
    print(f"   Server: {config.host}:{config.port}")
    validation_enabled = bool(config.anthropic_api_key and not config.ignore_client_api_key)
    print(f"   Client API Key Validation: {'Enabled' if validation_enabled else 'Disabled'}")
    print(
        f"   Ignore Client API Key Headers: {'Enabled' if config.ignore_client_api_key else 'Disabled'}"
    )
    print(
        f"   Observability: {'Enabled' if config.observability_enabled else 'Disabled'} "
        f"({config.observability_db_path})"
    )
    langfuse_host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    langfuse_keys = bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
    )
    print(
        f"   Langfuse: {'Enabled' if config.langfuse_enabled else 'Disabled'} "
        f"(host={langfuse_host}, keys={'configured' if langfuse_keys else 'unconfigured'})"
    )
    print(
        "   Request Optimizations: "
        f"{'Enabled' if config.enable_request_optimizations else 'Disabled'}"
    )
    print("")

    # Parse log level - extract just the first word to handle comments
    log_level = config.log_level.split()[0].lower()

    # Validate and set default if invalid
    valid_levels = ["debug", "info", "warning", "error", "critical"]
    if log_level not in valid_levels:
        log_level = "info"

    # Start server
    uvicorn.run(
        "src.main:app",
        host=config.host,
        port=config.port,
        log_level=log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
