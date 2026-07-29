from src.observability.pricing import ModelPrice, PricingCatalog


class FakeDynamic:
    def __init__(self, prices):
        self._p = prices

    def get_pricing(self, model):
        return self._p.get(model)


def test_dynamic_prices_uncatalogued_model():
    dyn = FakeDynamic({"x/dyn": ModelPrice("x/dyn", 0.5, 1.0)})
    cat = PricingCatalog("{}", dynamic=dyn)
    q = cat.quote("x/dyn", 1_000_000, 1_000_000)
    assert q["estimated_cost"] == 1.5  # 0.5 + 1.0


def test_static_overrides_dynamic():
    dyn = FakeDynamic({"x/m": ModelPrice("x/m", 9.0, 9.0)})
    cat = PricingCatalog('{"x/m": {"input_per_1m": 1.0, "output_per_1m": 2.0}}', dynamic=dyn)
    q = cat.quote("x/m", 1_000_000, 1_000_000)
    assert q["estimated_cost"] == 3.0  # static wins


def test_missing_everywhere_is_none():
    cat = PricingCatalog("{}", dynamic=FakeDynamic({}))
    assert cat.quote("nope", 10, 10)["estimated_cost"] is None
