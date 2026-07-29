from src.conversion import request_converter as rc


def test_context_limit_falls_back_to_catalog(monkeypatch):
    class FakeCatalog:
        def get_context_length(self, m):
            # Use a value above the 16384 floor so it is accepted (catalog
            # placeholders like 8000 are intentionally ignored by _get_context_limit).
            return 64000 if m == "x/dyn" else None
    monkeypatch.setattr(rc, "model_catalog", FakeCatalog())
    # a model that matches no config role → catalog value used
    assert rc._get_context_limit("x/dyn") == 64000


def test_context_limit_default_when_catalog_empty(monkeypatch):
    class FakeCatalog:
        def get_context_length(self, m):
            return None
    monkeypatch.setattr(rc, "model_catalog", FakeCatalog())
    assert rc._get_context_limit("x/unknown") == rc.DEFAULT_CONTEXT_LIMIT
