"""Tests for the Langfuse integration module.

These tests verify the configuration, client no-op mode, and import chain
without requiring the ``langfuse`` SDK to be installed.
"""

import os
import sys

import pytest

from src.langfuse_integration import LangfuseConfig, get_langfuse_client
from src.langfuse_integration.config import _as_bool, _as_int
from src.langfuse_integration.client import _serialize


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestLangfuseConfig:
    def test_defaults_disabled(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_ENABLED", raising=False)
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        config = LangfuseConfig()
        assert config.enabled is False
        assert config.is_configured() is False

    def test_enabled_true(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        config = LangfuseConfig()
        assert config.enabled is True

    def test_enabled_one(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "1")
        config = LangfuseConfig()
        assert config.enabled is True

    def test_enabled_false(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "false")
        config = LangfuseConfig()
        assert config.enabled is False

    def test_configured_with_keys(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        config = LangfuseConfig()
        assert config.is_configured() is True

    def test_not_configured_without_keys(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
        config = LangfuseConfig()
        assert config.is_configured() is False

    def test_not_configured_missing_secret(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
        config = LangfuseConfig()
        assert config.is_configured() is False

    def test_default_host(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_HOST", raising=False)
        config = LangfuseConfig()
        assert config.host == "http://localhost:8084"

    def test_custom_host(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        config = LangfuseConfig()
        assert config.host == "https://cloud.langfuse.com"

    def test_flush_interval_default(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_FLUSH_INTERVAL", raising=False)
        config = LangfuseConfig()
        assert config.flush_interval == 10

    def test_flush_interval_custom(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_FLUSH_INTERVAL", "5")
        config = LangfuseConfig()
        assert config.flush_interval == 5

    def test_max_queue_size_default(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_MAX_QUEUE_SIZE", raising=False)
        config = LangfuseConfig()
        assert config.max_queue_size == 1000


class TestConfigHelpers:
    def test_as_bool_true(self):
        for val in ("1", "true", "yes", "on", "True", "YES"):
            assert _as_bool(val) is True

    def test_as_bool_false(self):
        for val in ("0", "false", "no", "off", "False", "", None):
            assert _as_bool(val) is False

    def test_as_int_valid(self):
        assert _as_int("42", 0) == 42
        assert _as_int("0", 10) == 0
        assert _as_int("-5", 0) == -5

    def test_as_int_invalid(self):
        assert _as_int("abc", 10) == 10
        assert _as_int(None, 7) == 7


# ---------------------------------------------------------------------------
# Client tests (SDK-absent, no-op mode)
# ---------------------------------------------------------------------------

class TestLangfuseClientNoop:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        # These tests exercise the no-op path (no real SDK), so ensure the
        # environment is clean — .env may have set LANGFUSE_ENABLED=true with
        # placeholder keys that would trigger a live Langfuse() init.
        # Done in a fixture (not setup_method) so it reliably runs before
        # the client is constructed, and the singleton is reset + rebuilt here.
        monkeypatch.delenv("LANGFUSE_ENABLED", raising=False)
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_HOST", raising=False)
        import src.langfuse_integration.client as _client_mod

        _client_mod._langfuse_client = None
        self.config = LangfuseConfig()
        self.client = get_langfuse_client()

    def test_start_trace_returns_id(self):
        # Noop path: returns the original id as-is (no hex validation without
        # a real Langfuse SDK — _ensure_client bails because enabled=False).
        trace_id = self.client.start_trace(id="req-1", name="test")
        assert trace_id == "req-1"

    def test_start_trace_generates_id(self):
        trace_id = self.client.start_trace(name="test")
        assert trace_id is not None
        assert len(trace_id) == 32  # UUID hex

    def test_start_generation_returns_id(self):
        gen_id = self.client.start_generation(trace_id="tr-1", name="gen")
        assert gen_id is not None
        assert len(gen_id) == 32

    def test_end_generation_noop(self):
        self.client.end_generation(generation_id="gen-1", status="success")

    def test_score_generation_noop(self):
        self.client.score_generation(
            trace_id="tr-1", generation_id="gen-1", name="quality", value=0.9
        )

    def test_create_span_noop(self):
        self.client.create_span(trace_id="tr-1", name="search")

    def test_add_event_noop(self):
        self.client.add_event_to_trace(
            trace_id="tr-1", event_name="tool-emit", metadata={"tool": "bash"}
        )

    def test_flush_noop(self):
        self.client.flush()


# ---------------------------------------------------------------------------
# Serialize helper
# ---------------------------------------------------------------------------

class TestSerialize:
    def test_dict_passthrough(self):
        assert _serialize({"key": "value"}) == {"key": "value"}

    def test_list_passthrough(self):
        assert _serialize([1, 2, 3]) == [1, 2, 3]

    def test_string_passed_through(self):
        # v4: plain strings pass through (not JSON-encoded)
        assert _serialize("hello") == "hello"

    def test_none_returns_none(self):
        # v4: None stays None rather than becoming ""
        assert _serialize(None) is None


# ---------------------------------------------------------------------------
# Import chain tests
# ---------------------------------------------------------------------------

class TestImportChain:
    def test_langfuse_integration_init(self):
        from src.langfuse_integration import LangfuseConfig, get_langfuse_client
        assert LangfuseConfig is not None
        assert get_langfuse_client is not None

    def test_observability_re_exports(self):
        from src.observability import get_langfuse_client
        assert get_langfuse_client is not None

    def test_observability_original_exports(self):
        from src.observability import ObservabilityRecorder, PricingCatalog
        assert ObservabilityRecorder is not None
        assert PricingCatalog is not None

    def test_config_has_langfuse_enabled(self):
        from src.core.config import config
        assert hasattr(config, "langfuse_enabled")

    def test_endpoints_import(self):
        from src.api.endpoints import router
        assert router is not None