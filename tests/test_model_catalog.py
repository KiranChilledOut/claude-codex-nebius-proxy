from src.core.model_catalog import ModelCatalog

SAMPLE = {
    "data": [
        {
            "id": "x/model-a",
            "context_length": 8000,
            "architecture": {"modality": "text->text"},
            "pricing": {"prompt": "0.0000006", "completion": "0.0000018"},
            "description": "A",
        },
        {
            "id": "x/model-b",
            "context_length": 204800,
            "architecture": {"modality": "text+image->text"},
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        },
    ]
}


def _catalog_with(payload):
    c = ModelCatalog("http://x/v1", "key", enabled=True)
    c._fetch = lambda: payload  # bypass network
    return c


def test_refresh_parses_pricing_and_context():
    c = _catalog_with(SAMPLE)
    assert c.refresh() is True
    p = c.get_pricing("x/model-a")
    assert p is not None
    assert round(p.input_per_1m, 6) == 0.6   # 0.0000006 * 1e6
    assert round(p.output_per_1m, 6) == 1.8
    assert c.get_context_length("x/model-a") == 8000
    assert set(c.model_ids()) == {"x/model-a", "x/model-b"}


def test_failed_refresh_keeps_last_good():
    c = _catalog_with(SAMPLE)
    assert c.refresh() is True
    def boom():
        raise RuntimeError("network down")
    c._fetch = boom
    assert c.refresh() is False
    assert c.get_pricing("x/model-a") is not None  # last-good retained


def test_unknown_model_returns_none():
    c = _catalog_with(SAMPLE)
    c.refresh()
    assert c.get_pricing("nope") is None
    assert c.get_context_length("nope") is None


def test_disabled_catalog_is_inert():
    c = ModelCatalog("http://x/v1", "key", enabled=False)
    c._fetch = lambda: SAMPLE
    assert c.refresh() is False
    assert c.model_ids() == []
