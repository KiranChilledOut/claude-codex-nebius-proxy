"""Tests for the statusline effective_model on /api/observability/config.

The statusline curls this endpoint through the per-session forwarder, which
injects x-session-model. The endpoint must echo back the effective per-session
model so the statusline shows the session's model, not the global one.
"""

from fastapi.testclient import TestClient

from src.core.config import config
from src.main import app


def test_config_echoes_session_model_from_header():
    client = TestClient(app)
    resp = client.get(
        "/api/observability/config",
        headers={"x-session-model": "my/custom-model"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["effective_model"] == "my/custom-model"


def test_config_falls_back_to_global_model_without_header():
    client = TestClient(app)
    resp = client.get("/api/observability/config")
    assert resp.status_code == 200, resp.text
    assert resp.json()["effective_model"] == config.model
