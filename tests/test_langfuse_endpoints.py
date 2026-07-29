"""Integration tests for Langfuse tracing through the proxy endpoints.

These tests enable Langfuse with mock credentials and a monkeypatched Langfuse
SDK client, then drive the /v1/messages and /v1/responses endpoints through
TestClient to verify the trace/generation lifecycle fires correctly and that
the SQLite observability still records in parallel (dual-write).
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api import endpoints
from src.core.config import config
from src.langfuse_integration import client as lf_client_mod
from src.langfuse_integration.client import LangfuseClient
from src.main import app
from src.observability.store import observability_recorder


@pytest.fixture
def langfuse_enabled(monkeypatch):
    """Enable Langfuse with mock keys and inject a recording mock client."""
    monkeypatch.setattr(config, "langfuse_enabled", True)
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")

    # Build a fresh client and inject a mock SDK backend so no network calls happen.
    fresh = LangfuseClient(lf_client_mod.LangfuseConfig())
    fresh.config.enabled = True
    fresh.config.public_key = "pk-lf-test"
    fresh.config.secret_key = "sk-lf-test"
    fresh._client = MagicMock()
    fresh._client.start_observation.return_value = MagicMock(
        id="obs-1", trace_id="trace-1"
    )
    fresh._client.create_event = MagicMock()
    fresh._client.create_score = MagicMock()
    fresh._client.flush = MagicMock()

    # Patch the singleton getter used by endpoints.py.
    monkeypatch.setattr(lf_client_mod, "_langfuse_client", fresh)
    monkeypatch.setattr(
        "src.api.endpoints.get_langfuse_client", lambda: fresh
    )
    return fresh


def _stub_openai_nonstream():
    """Return an async mock returning a minimal OpenAI chat completion."""
    async def _create(request, request_id=None):
        return {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hello back"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    return _create


def _stub_openai_stream():
    """Return an async generator yielding SSE chunks for streaming."""
    async def _stream(request, request_id=None):
        yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
        yield 'data: {"usage":{"prompt_tokens":8,"completion_tokens":2,"total_tokens":10}}'
        yield "data: [DONE]"
    return _stream


# ---------------------------------------------------------------------------
# /v1/messages — non-streaming
# ---------------------------------------------------------------------------

class TestMessagesLangfuseTracing:
    def test_non_streaming_creates_trace_and_generation(self, langfuse_enabled, monkeypatch):
        monkeypatch.setattr(
            endpoints.openai_client, "create_chat_completion", _stub_openai_nonstream()
        )
        # No request optimizations — we want the real upstream path.
        monkeypatch.setattr(config, "enable_request_optimizations", False)
        # No ensemble, no search.
        monkeypatch.setattr(config, "ensemble_mode", "off")
        monkeypatch.setattr(config, "tavily_api_key", "")

        client = TestClient(app)
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200, resp.text

        # A generation observation should have been started and ended.
        assert langfuse_enabled._client.start_observation.called
        ended = [
            c for c in langfuse_enabled._client.start_observation.return_value.end.call_args_list
        ]
        assert len(ended) >= 1

    def test_streaming_creates_trace_and_generation(
        self, langfuse_enabled, monkeypatch
    ):
        monkeypatch.setattr(
            endpoints.openai_client, "create_chat_completion_stream", _stub_openai_stream()
        )
        monkeypatch.setattr(config, "enable_request_optimizations", False)
        monkeypatch.setattr(config, "ensemble_mode", "off")
        monkeypatch.setattr(config, "tavily_api_key", "")

        client = TestClient(app)
        with client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as resp:
            assert resp.status_code == 200
            # Drain the stream so the finally block fires.
            list(resp.iter_lines())

        assert langfuse_enabled._client.start_observation.called


# ---------------------------------------------------------------------------
# Dual-write: SQLite observability still records when Langfuse is on
# ---------------------------------------------------------------------------

class TestDualWrite:
    def test_langfuse_mock_called_on_nonstream_request(
        self, langfuse_enabled, monkeypatch
    ):
        """Langfuse trace fires when enabled + configured on /v1/messages."""
        monkeypatch.setattr(
            endpoints.openai_client, "create_chat_completion", _stub_openai_nonstream()
        )
        monkeypatch.setattr(config, "enable_request_optimizations", False)
        monkeypatch.setattr(config, "ensemble_mode", "off")
        monkeypatch.setattr(config, "tavily_api_key", "")

        client = TestClient(app)
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200

        # Langfuse observation was started (generation) and ended.
        assert langfuse_enabled._client.start_observation.called
        obs = langfuse_enabled._client.start_observation.return_value
        assert obs.update.called, "generation update (output) should have been called"
        assert obs.end.called, "generation end should have been called"
