import sqlite3
import time
from unittest.mock import MagicMock

import pytest

from src.core.config import config
from src.observability.pricing import PricingCatalog
from src.observability.store import ObservabilityRecorder, utc_now_iso


def test_stream_usage_falls_back_to_estimate_when_provider_usage_is_missing():
    from src.api.endpoints import _stream_usage_with_fallback

    usage = _stream_usage_with_fallback({"usage": {}, "estimated_output_tokens": 12}, 345)

    assert usage == {
        "input_tokens": 345,
        "output_tokens": 12,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "source": "estimated",
    }


def test_pricing_catalog_computes_model_cost():
    catalog = PricingCatalog(
        '{"zai-org/GLM-4.7-FP8":{"input_per_1m":0.30,"output_per_1m":1.20,"advertised_tok_s":36.8}}'
    )

    quote = catalog.quote("zai-org/GLM-4.7-FP8", 1_000_000, 500_000)

    assert quote["input_cost"] == pytest.approx(0.30)
    assert quote["output_cost"] == pytest.approx(0.60)
    assert quote["estimated_cost"] == pytest.approx(0.90)
    assert quote["advertised_tok_s"] == pytest.approx(36.8)


def test_pricing_catalog_treats_local_optimizations_as_free():
    catalog = PricingCatalog("{}")

    quote = catalog.quote("local/quota_probe", 1_000_000, 500_000)

    assert quote["input_cost"] == 0
    assert quote["output_cost"] == 0
    assert quote["estimated_cost"] == 0
    assert quote["currency"] == "USD"


@pytest.mark.asyncio
async def test_observability_recorder_persists_request_and_tool_call(tmp_path):
    db_path = tmp_path / "observability.sqlite3"
    recorder = ObservabilityRecorder(
        enabled=True,
        db_path=str(db_path),
        queue_size=10,
        pricing_catalog=PricingCatalog(
            '{"model-a":{"input_per_1m":0.50,"output_per_1m":2.00,"advertised_tok_s":40}}'
        ),
        store_tool_args=True,
    )

    await recorder.start()
    recorder.record_request(
        request_id="req_1",
        started_at=utc_now_iso(),
        started_at_unix=time.time(),
        completed_at=utc_now_iso(),
        base_url="https://api.tokenfactory.nebius.com/v1",
        claude_model="claude-sonnet",
        backend_model="model-a",
        stream=True,
        status="success",
        http_status=200,
        latency_ms=1000,
        usage={"input_tokens": 1000, "output_tokens": 500, "source": "estimated"},
        stop_reason="tool_use",
        tool_calls=[
            {
                "tool_id": "call_1",
                "tool_name": "bash",
                "arguments": {"command": "echo ok", "api_key": "secret"},
                "status": "emitted",
                "sanitized": True,
            }
        ],
    )
    await recorder.stop()

    requests = recorder.fetch_requests(limit=10)
    tool_calls = recorder.fetch_tool_calls(limit=10)

    assert len(requests) == 1
    assert requests[0]["backend_model"] == "model-a"
    assert requests[0]["estimated_cost"] == pytest.approx(0.0015)
    assert requests[0]["observed_tok_s"] == pytest.approx(500)
    assert requests[0]["tool_call_count"] == 1
    assert requests[0]["usage_source"] == "estimated"

    assert len(tool_calls) == 1
    assert tool_calls[0]["tool_name"] == "bash"
    assert "echo ok" in tool_calls[0]["arguments_preview"]
    assert "secret" not in tool_calls[0]["arguments_preview"]
    assert "[redacted]" in tool_calls[0]["arguments_preview"]


@pytest.mark.asyncio
async def test_record_request_persists_langfuse_trace_id(tmp_path):
    """The Langfuse trace id threads through record_request and surfaces in
    fetch_requests (SELECT * exposes the column), defaulting to None when
    absent so the dashboard can hide the deep-link."""
    db_path = tmp_path / "observability.sqlite3"
    recorder = ObservabilityRecorder(
        enabled=True,
        db_path=str(db_path),
        queue_size=10,
        pricing_catalog=PricingCatalog('{}'),
        store_tool_args=False,
    )
    await recorder.start()
    # One row with a trace id, one without (Langfuse was off for it).
    common = dict(
        started_at=utc_now_iso(),
        started_at_unix=time.time(),
        completed_at=utc_now_iso(),
        base_url="https://api.tokenfactory.nebius.com/v1",
        claude_model="claude-sonnet",
        backend_model="model-a",
        stream=False,
        status="success",
        http_status=200,
        latency_ms=10,
    )
    recorder.record_request(request_id="req_with_trace", **common, langfuse_trace_id="deadbeef" * 4)
    recorder.record_request(request_id="req_without_trace", **common, langfuse_trace_id=None)
    await recorder.stop()

    rows = {r["request_id"]: r for r in recorder.fetch_requests(limit=10)}
    assert rows["req_with_trace"]["langfuse_trace_id"] == "deadbeef" * 4
    assert rows["req_without_trace"]["langfuse_trace_id"] is None


def test_observability_config_exposes_langfuse_block():
    """The dashboard config endpoint exposes the Langfuse host + project id
    needed to build a trace deep-link; values mirror the client's config."""
    from fastapi.testclient import TestClient

    from src.main import app

    client = TestClient(app)
    resp = client.get("/api/observability/config")
    assert resp.status_code == 200
    body = resp.json()
    assert "langfuse" in body
    lf = body["langfuse"]
    # Mirror the LangfuseConfig contract used to build the trace URL.
    assert {"enabled", "configured", "host", "project_id"}.issubset(lf)
    assert isinstance(lf["enabled"], bool)
    assert isinstance(lf["configured"], bool)
    assert isinstance(lf["host"], str) and lf["host"]
    assert isinstance(lf["project_id"], str)


@pytest.mark.asyncio
async def test_fetch_ensemble_leaderboard_aggregates_wins_and_user_picks(tmp_path):
    from src.ensemble.engine import EnsembleCandidate

    db_path = tmp_path / "observability.sqlite3"
    recorder = ObservabilityRecorder(
        enabled=True,
        db_path=str(db_path),
        queue_size=50,
        pricing_catalog=PricingCatalog("{}"),
        store_tool_args=True,
    )
    await recorder.start()

    def cand(index, model, status, chosen_by, score, latency):
        c = EnsembleCandidate(index=index, model=model)
        c.status = status
        c.chosen_by = chosen_by
        c.score = score
        c.latency_ms = latency
        return c

    # Race 1: model-a wins automatically over model-b.
    recorder.record_ensemble(
        request_id="r1",
        session_id="s1",
        session_name="sess",
        mode="hedge",
        candidates=[
            cand(0, "model-a", "won", "auto", 4.0, 800),
            cand(1, "model-b", "lost", None, 2.0, 600),
        ],
    )
    # Race 2: the user overrides and picks model-b; model-a errored out.
    recorder.record_ensemble(
        request_id="r2",
        session_id="s1",
        session_name="sess",
        mode="approval",
        candidates=[
            cand(0, "model-a", "error", None, float("-inf"), 1200),
            cand(1, "model-b", "won", "user", 3.0, 700),
        ],
    )
    await recorder.stop()

    board = {row["model"]: row for row in recorder.fetch_ensemble_leaderboard(hours=24)}

    assert board["model-a"]["races"] == 2
    assert board["model-a"]["wins"] == 1
    assert board["model-a"]["errors"] == 1
    assert board["model-a"]["win_rate"] == pytest.approx(0.5)
    # Errored race excluded from avg latency (only the 800ms win counts).
    assert board["model-a"]["avg_latency_ms"] == pytest.approx(800)

    assert board["model-b"]["races"] == 2
    assert board["model-b"]["wins"] == 1
    assert board["model-b"]["user_picks"] == 1
    assert board["model-b"]["errors"] == 0


def test_connect_closes_connection_even_on_exception(monkeypatch):
    """_connect context manager calls conn.close() in finally when an exception is raised."""
    recorder = ObservabilityRecorder(
        enabled=True,
        db_path=":memory:",
        queue_size=10,
        pricing_catalog=PricingCatalog("{}"),
    )
    mock_conn = MagicMock()
    monkeypatch.setattr(sqlite3, "connect", lambda _path: mock_conn)

    with pytest.raises(RuntimeError):
        with recorder._connect() as _conn:
            raise RuntimeError("boom")

    mock_conn.close.assert_called_once()


def test_connect_closes_after_successful_yield(monkeypatch):
    """_connect context manager calls conn.close() after normal completion."""
    recorder = ObservabilityRecorder(
        enabled=True,
        db_path=":memory:",
        queue_size=10,
        pricing_catalog=PricingCatalog("{}"),
    )
    mock_conn = MagicMock()
    monkeypatch.setattr(sqlite3, "connect", lambda _path: mock_conn)

    with recorder._connect() as _conn:
        pass

    mock_conn.close.assert_called_once()


def test_context_usage_for_returns_latest_nonzero_tokens(tmp_path):
    """_context_usage_for returns latest request with tokens > 0."""
    db_path = tmp_path / "observability.sqlite3"
    recorder = ObservabilityRecorder(
        enabled=True,
        db_path=str(db_path),
        queue_size=10,
        pricing_catalog=PricingCatalog("{}"),
    )
    recorder._init_db()
    with recorder._connect() as conn:
        conn.execute(
            """
            INSERT INTO requests (
                request_id, started_at, started_at_unix, status,
                total_tokens, input_tokens, output_tokens,
                session_id, claude_model, backend_model,
                cache_read_input_tokens, cache_creation_input_tokens,
                stream, latency_ms, usage_source
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'provider')
            """,
            (
                "r1", "2024-01-01T00:00:00", 1, "success",
                0, 0, 0, "s1", "", "",
                0, 0, 0, 0,
            ),
        )
        conn.execute(
            """
            INSERT INTO requests (
                request_id, started_at, started_at_unix, status,
                total_tokens, input_tokens, output_tokens,
                session_id, claude_model, backend_model,
                cache_read_input_tokens, cache_creation_input_tokens,
                stream, latency_ms, usage_source
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'provider')
            """,
            (
                "r2", "2024-01-01T00:00:01", 2, "success",
                100, 50, 50, "s1", "claude-sonnet", "model-a",
                5, 0, 0, 1000,
            ),
        )
        conn.execute(
            """
            INSERT INTO requests (
                request_id, started_at, started_at_unix, status,
                total_tokens, input_tokens, output_tokens,
                session_id, claude_model, backend_model,
                cache_read_input_tokens, cache_creation_input_tokens,
                stream, latency_ms, usage_source
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'provider')
            """,
            (
                "r3", "2024-01-01T00:00:02", 3, "success",
                200, 80, 120, "s1", "claude-sonnet", "model-b",
                10, 0, 0, 1000,
            ),
        )
        result = recorder._context_usage_for(conn, "session_id", "s1")

    assert result is not None
    assert result["total_tokens"] == 200
    assert result["input_tokens"] == 80
    assert result["output_tokens"] == 120
    assert result["cache_read_input_tokens"] == 15
    assert result["request_count"] == 3


@pytest.fixture
def client(monkeypatch):
    """Dashboard test client (auth bypassed)."""
    from fastapi.testclient import TestClient
    from src.main import app

    monkeypatch.setattr(config, "ignore_client_api_key", True)
    return TestClient(app)


def test_dashboard_returns_charset_utf8(client):
    """Dashboard content-type must include charset=utf-8."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    ct = response.headers.get("content-type", "")
    assert "text/html" in ct
    assert "charset=utf-8" in ct


def test_dashboard_serves_known_assets(client):
    """CSS and JS assets must be served with correct mime-types and cache headers."""
    for asset, expected_mime in (
        ("dashboard.css", "text/css"),
        ("dashboard.js", "application/javascript"),
    ):
        response = client.get(f"/dashboard/assets/{asset}")
        assert response.status_code == 200
        ct = response.headers.get("content-type", "")
        assert expected_mime in ct
        cc = response.headers.get("Cache-Control", "")
        assert "max-age=3600" in cc
        assert "must-revalidate" in cc


def test_dashboard_rejects_traversal_attempts(client):
    """Directory-traversal-like asset names must 404."""
    for asset in ("../store.py", "foo/../bar", "../../etc/passwd"):
        response = client.get(f"/dashboard/assets/{asset}")
        assert response.status_code == 404


def test_dashboard_rejects_unknown_assets(client):
    """Asset names unrelated to existing files must 404."""
    response = client.get("/dashboard/assets/nonexistent.xyz")
    assert response.status_code == 404


def test_dashboard_health_returns_ok(client):
    """/dashboard/health must always report availability."""
    response = client.get("/dashboard/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.0"}


def test_context_usage_for_returns_none_when_no_rows(tmp_path):
    """_context_usage_for returns None when no matching session."""
    db_path = tmp_path / "observability.sqlite3"
    recorder = ObservabilityRecorder(
        enabled=True,
        db_path=str(db_path),
        queue_size=10,
        pricing_catalog=PricingCatalog("{}"),
    )
    recorder._init_db()
    with recorder._connect() as conn:
        result = recorder._context_usage_for(conn, "session_id", "nonexistent")
    assert result is None


def test_dashboard_assets_served_and_render_trace_link():
    """The dashboard JS includes the Langfuse trace link renderer, and the
    static assets are served (so a live dashboard can build deep-links)."""
    from fastapi.testclient import TestClient
    from src.main import app

    client = TestClient(app)
    js = client.get("/dashboard/assets/dashboard.js").text
    # The renderer + URL builder exist and gate on enabled+host+projectId+trace.
    assert "function renderLangfuseTraceLink" in js
    assert "function langfuseTraceUrl" in js
    assert "/project/" in js and "/traces/" in js
    # Never renders when any required piece is missing.
    assert "if (!langfuseState.enabled" in js
    html = client.get("/dashboard").text
    assert html.count("<th>Trace</th>") == 2
