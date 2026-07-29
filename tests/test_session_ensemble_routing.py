"""Integration test: per-session ENSEMBLE headers drive run_hedge_race.

Verifies that x-session-ensemble-{mode,models,judge} headers, resolved by
resolve_session_settings, flow through to run_hedge_race in the /v1/messages
handler with the correct positional and keyword arguments.

No real backend call is made: run_hedge_race is monkeypatched to record its
args and raise a sentinel RuntimeError, which the handler converts to a 500.
We only care that the race was invoked with the right header-derived values.
"""

import pytest
from fastapi.testclient import TestClient

from src.api import endpoints
from src.core.config import config
from src.main import app


@pytest.fixture(autouse=True)
def _disable_local_optimizations(monkeypatch):
    """Prevent the optimization short-circuit from skipping the ensemble path."""
    monkeypatch.setattr(config, "enable_request_optimizations", False)


@pytest.fixture(autouse=True)
def _disable_langfuse(monkeypatch):
    """Suppress Langfuse so no network calls happen."""
    monkeypatch.setattr(config, "langfuse_enabled", False)


class TestSessionEnsembleRouting:
    def test_ensemble_headers_reach_run_hedge_race(self, monkeypatch):
        """Header-derived ensemble settings arrive at run_hedge_race intact."""
        captured = {}

        async def fake_run_hedge_race(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            raise RuntimeError("captured")

        monkeypatch.setattr(endpoints, "run_hedge_race", fake_run_hedge_race)
        # Disable tavily so has_search_tool stays False (keeps path clean).
        monkeypatch.setattr(config, "tavily_api_key", "")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/messages",
            headers={
                "x-session-name": "ens",
                "x-session-ensemble-mode": "hedge",
                "x-session-ensemble-models": "model-a,model-b",
                "x-session-ensemble-judge": "judge-x",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        # Handler raises our sentinel → 500; what matters is the capture.
        assert resp.status_code == 500, (
            f"Expected 500 from sentinel but got {resp.status_code}: {resp.text}"
        )
        assert captured, "run_hedge_race was never called — ensemble branch not reached"

        # Positional args: (openai_request, openai_client, request_id,
        #                    ensemble_models, ensemble_mode, ...)
        args = captured["args"]
        assert args[3] == ["model-a", "model-b"], (
            f"ensemble_models mismatch: {args[3]!r}"
        )
        assert args[4] == "hedge", f"ensemble_mode mismatch: {args[4]!r}"

        # Keyword args
        assert captured["kwargs"].get("judge_model") == "judge-x", (
            f"judge_model mismatch: {captured['kwargs']!r}"
        )
