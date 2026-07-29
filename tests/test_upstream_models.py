"""Unit test for the GET /v1/upstream-models endpoint.

Drives the endpoint through TestClient with a monkeypatched upstream
``list_models`` so no network calls happen. Mirrors the self-contained
TestClient pattern used by ``tests/test_langfuse_endpoints.py`` (no skip
guard — runs in the default suite).
"""

from fastapi.testclient import TestClient

from src.api import endpoints
from src.main import app


def test_upstream_models_endpoint_returns_ids(monkeypatch):
    monkeypatch.setattr(
        endpoints.openai_client,
        "list_models",
        lambda: [{"id": "a/m1"}, {"id": "a/m2"}],
    )
    client = TestClient(app)
    resp = client.get("/v1/upstream-models")
    assert resp.status_code == 200
    assert resp.json() == {"data": ["a/m1", "a/m2"]}
