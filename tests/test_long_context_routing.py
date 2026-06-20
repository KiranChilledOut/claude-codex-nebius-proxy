"""Tests for token-aware long-context routing.

Name-based routing (opus->BIG, sonnet->MIDDLE, haiku->SMALL) ignores how large
the prompt actually is. When a text request's estimated prompt exceeds
LONG_CONTEXT_THRESHOLD tokens, it is routed to LONG_CONTEXT_MODEL instead. These
tests cover the ModelManager policy in isolation and end-to-end through
convert_claude_to_openai.
"""

from src.conversion.request_converter import convert_claude_to_openai
from src.core.config import config as global_config
from src.core.model_manager import ModelManager, model_manager
from src.models.claude import ClaudeMessage, ClaudeMessagesRequest


class _StubConfig:
    def __init__(self, long_context_model="LONG-CTX", long_context_threshold=60000):
        self.big_model = "BIG"
        self.middle_model = "MID"
        self.small_model = "SMALL"
        self.vision_model = "VIS"
        self.long_context_model = long_context_model
        self.long_context_threshold = long_context_threshold


# --- policy (ModelManager.apply_long_context / long_context_enabled) ---------


def test_enabled_when_model_and_threshold_set():
    assert ModelManager(_StubConfig()).long_context_enabled() is True


def test_disabled_without_model():
    assert ModelManager(_StubConfig(long_context_model="")).long_context_enabled() is False


def test_disabled_with_zero_threshold():
    assert ModelManager(_StubConfig(long_context_threshold=0)).long_context_enabled() is False


def test_escalates_over_threshold():
    mm = ModelManager(_StubConfig(long_context_threshold=1000))
    assert mm.apply_long_context("BIG", 2000) == "LONG-CTX"


def test_keeps_base_under_threshold():
    mm = ModelManager(_StubConfig(long_context_threshold=1000))
    assert mm.apply_long_context("BIG", 500) == "BIG"


def test_passthrough_when_disabled():
    mm = ModelManager(_StubConfig(long_context_model=""))
    assert mm.apply_long_context("BIG", 10_000_000) == "BIG"


def test_no_double_switch_when_already_long_context():
    mm = ModelManager(_StubConfig(long_context_threshold=1000))
    assert mm.apply_long_context("LONG-CTX", 999_999) == "LONG-CTX"


# --- end-to-end through convert_claude_to_openai -----------------------------


def _request(text):
    return ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=128,
        messages=[ClaudeMessage(role="user", content=text)],
    )


def test_convert_routes_large_text_prompt(monkeypatch):
    monkeypatch.setattr(model_manager.config, "long_context_model", "LONG-CTX", raising=False)
    monkeypatch.setattr(model_manager.config, "long_context_threshold", 50, raising=False)
    out = convert_claude_to_openai(_request("word " * 4000), model_manager)
    assert out["model"] == "LONG-CTX"


def test_convert_keeps_name_route_under_threshold(monkeypatch):
    monkeypatch.setattr(model_manager.config, "long_context_model", "LONG-CTX", raising=False)
    monkeypatch.setattr(model_manager.config, "long_context_threshold", 100_000, raising=False)
    out = convert_claude_to_openai(_request("hello there"), model_manager)
    # sonnet -> middle_model when the prompt is small
    assert out["model"] == global_config.middle_model


def test_convert_long_context_off_by_default(monkeypatch):
    # With no LONG_CONTEXT_MODEL configured, even a huge prompt keeps its route.
    monkeypatch.setattr(model_manager.config, "long_context_model", "", raising=False)
    out = convert_claude_to_openai(_request("word " * 4000), model_manager)
    assert out["model"] == global_config.middle_model
