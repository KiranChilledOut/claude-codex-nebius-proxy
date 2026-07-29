from fastapi.testclient import TestClient
from src.main import app
from src.api import endpoints


def test_upstream_models_uses_catalog(monkeypatch):
    class FakeCatalog:
        def model_ids(self):
            return ["x/a", "x/b"]
    monkeypatch.setattr(endpoints, "model_catalog", FakeCatalog())
    client = TestClient(app)
    resp = client.get("/v1/upstream-models")
    assert resp.status_code == 200
    assert resp.json() == {"data": ["x/a", "x/b"]}


def test_upstream_models_falls_back_when_catalog_empty(monkeypatch):
    class FakeCatalog:
        def model_ids(self):
            return []
    monkeypatch.setattr(endpoints, "model_catalog", FakeCatalog())
    monkeypatch.setattr(
        endpoints.openai_client, "list_models", lambda: [{"id": "fb/x"}]
    )
    client = TestClient(app)
    resp = client.get("/v1/upstream-models")
    assert resp.status_code == 200
    assert resp.json() == {"data": ["fb/x"]}
