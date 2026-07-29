"""Tests for the GET/PUT /v1/session-model runtime override endpoints.

Drives the endpoints through TestClient with a monkeypatched upstream model
list so no network calls happen, mirroring tests/test_upstream_models.py. The
override store is process-global, so each test cleans up the session it sets.
"""

from fastapi.testclient import TestClient

from src.api import endpoints
from src.core.session_settings import (
    clear_runtime_model,
    get_runtime_model,
    resolve_session_settings,
)
from src.main import app

SESSION = "endpoint-test-sess"


def _models():
    def _impl(monkeypatch):
        monkeypatch.setattr(
            endpoints.openai_client,
            "list_models",
            lambda: [{"id": "a/m1"}, {"id": "a/m2"}, {"id": "a/vision"}],
        )
        # Ensure the catalog path is empty so the upstream fallback is used.
        monkeypatch.setattr(endpoints.model_catalog, "model_ids", lambda: [])
        return TestClient(app)
    return _impl


def test_get_returns_null_when_no_override(monkeypatch):
    clear_runtime_model(SESSION)
    client = _models()(monkeypatch)
    resp = client.get(f"/v1/session-model?session={SESSION}")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"session": SESSION, "model": None}


def test_put_sets_override_visible_to_resolve_session(monkeypatch):
    clear_runtime_model(SESSION)
    client = _models()(monkeypatch)
    resp = client.put("/v1/session-model", json={"session": SESSION, "model": "a/m2"})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ok": True, "session": SESSION, "model": "a/m2"}
    # The override wins over the forwarder's x-session-model header.
    s = resolve_session_settings({"x-session-name": SESSION, "x-session-model": "a/m1"})
    assert s.model == "a/m2"
    # And the GET reflects it.
    assert get_runtime_model(SESSION) == "a/m2"
    clear_runtime_model(SESSION)


def test_put_rejects_unknown_model(monkeypatch):
    clear_runtime_model(SESSION)
    client = _models()(monkeypatch)
    resp = client.put("/v1/session-model", json={"session": SESSION, "model": "no/such"})
    assert resp.status_code == 400, resp.text
    assert get_runtime_model(SESSION) is None
    clear_runtime_model(SESSION)


def test_put_requires_session_and_model(monkeypatch):
    client = _models()(monkeypatch)
    assert client.put("/v1/session-model", json={"session": "", "model": "a/m1"}).status_code == 400
    assert client.put("/v1/session-model", json={"session": SESSION, "model": ""}).status_code == 400
    assert get_runtime_model(SESSION) is None


def test_get_requires_session_param(monkeypatch):
    client = _models()(monkeypatch)
    assert client.get("/v1/session-model").status_code == 422
