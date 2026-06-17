from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import FileResponse, Response

from src.conversion.request_converter import _get_context_limit
from src.core.config import config
from src.core.model_manager import model_manager
from src.ensemble.approval import approval_store
from src.observability.store import observability_recorder

router = APIRouter()
STATIC_DIR = Path(__file__).resolve().parent / "static"


async def validate_dashboard_api_key(
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Mirror the proxy API-key policy without importing endpoint routes."""
    if config.ignore_client_api_key:
        return

    client_api_key = None
    if x_api_key:
        client_api_key = x_api_key
    elif authorization and authorization.startswith("Bearer "):
        client_api_key = authorization.replace("Bearer ", "")

    if config.anthropic_api_key and not config.validate_client_api_key(client_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key.")


@router.get("/dashboard")
async def dashboard(_: None = Depends(validate_dashboard_api_key)):
    html = (STATIC_DIR / "dashboard.html").read_text(encoding="utf-8")
    return Response(content=html, media_type="text/html; charset=utf-8")


@router.get("/dashboard/assets/{asset_name}")
async def dashboard_asset(asset_name: str, _: None = Depends(validate_dashboard_api_key)):
    allowed = {
        "dashboard.css": "text/css",
        "dashboard.js": "application/javascript",
    }
    if asset_name not in allowed:
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(
        STATIC_DIR / asset_name,
        media_type=allowed[asset_name],
        headers={"Cache-Control": "public, max-age=3600, must-revalidate"},
    )


@router.get("/dashboard/health")
async def dashboard_health(_: None = Depends(validate_dashboard_api_key)):
    """Quick endpoint for checking dashboard availability (useful for load balancers / Uptime Kuma)."""
    return {"status": "ok", "version": "1.0"}


@router.get("/api/observability/summary")
async def observability_summary(
    hours: int = Query(24, ge=1, le=8760),
    _: None = Depends(validate_dashboard_api_key),
):
    summary = observability_recorder.fetch_summary(hours=hours)
    summary["provider"] = {
        "base_url": config.openai_base_url,
        "observability_db_path": config.observability_db_path,
        "observability_enabled": config.observability_enabled,
        "store_tool_args": config.observability_store_tool_args,
    }
    summary["configured_models"] = {
        "big": config.big_model,
        "middle": config.middle_model,
        "small": config.small_model,
        "vision": config.vision_model,
    }
    summary["context_limits"] = {
        "big": config.big_model_context_limit,
        "middle": config.middle_model_context_limit,
        "small": config.small_model_context_limit,
        "vision": config.vision_model_context_limit,
    }
    summary["pricing"] = observability_recorder.pricing_catalog.as_list()
    return summary


@router.get("/api/observability/sessions")
async def observability_sessions(
    _: None = Depends(validate_dashboard_api_key),
):
    return {"data": observability_recorder.fetch_sessions()}


@router.get("/api/observability/sessions/{session_name}/summary")
async def observability_session_summary(
    session_name: str,
    _: None = Depends(validate_dashboard_api_key),
):
    return observability_recorder.fetch_session_summary(session_name)


@router.get("/api/observability/requests")
async def observability_requests(
    limit: int = Query(100, ge=1, le=500),
    session_name: Optional[str] = Query(None),
    _: None = Depends(validate_dashboard_api_key),
):
    if session_name:
        return {"data": observability_recorder.fetch_requests_by_session(session_name, limit=limit)}
    return {"data": observability_recorder.fetch_requests(limit=limit)}


@router.get("/api/observability/failures")
async def observability_failures(
    limit: int = Query(100, ge=1, le=500),
    session_name: Optional[str] = Query(None),
    _: None = Depends(validate_dashboard_api_key),
):
    if session_name:
        return {"data": observability_recorder.fetch_failures_by_session(session_name, limit=limit)}
    return {"data": observability_recorder.fetch_failures(limit=limit)}


@router.get("/api/observability/tool-calls")
async def observability_tool_calls(
    limit: int = Query(100, ge=1, le=500),
    session_name: Optional[str] = Query(None),
    _: None = Depends(validate_dashboard_api_key),
):
    if session_name:
        return {"data": observability_recorder.fetch_tool_calls_by_session(session_name, limit=limit)}
    return {"data": observability_recorder.fetch_tool_calls(limit=limit)}


@router.get("/api/observability/config")
async def observability_config(_: None = Depends(validate_dashboard_api_key)):
    return {
        "base_url": config.openai_base_url,
        "configured_models": {
            "big": config.big_model,
            "middle": config.middle_model,
            "small": config.small_model,
            "vision": config.vision_model,
        },
        "context_limits": {
            "big": config.big_model_context_limit,
            "middle": config.middle_model_context_limit,
            "small": config.small_model_context_limit,
            "vision": config.vision_model_context_limit,
        },
        "pricing": observability_recorder.pricing_catalog.as_list(),
        "observability_enabled": config.observability_enabled,
        "observability_db_path": config.observability_db_path,
        "store_tool_args": config.observability_store_tool_args,
        "routing": {
            "haiku": model_manager.config.small_model,
            "sonnet": model_manager.config.middle_model,
            "opus": model_manager.config.big_model,
            "image": model_manager.config.vision_model,
        },
    }


@router.get("/api/observability/context-usage")
async def observability_context_usage(
    session_id: Optional[str] = Header(None, alias="x-claude-code-session-id"),
    session_name: Optional[str] = Header(None, alias="x-session-name"),
    _: None = Depends(validate_dashboard_api_key),
):
    """Return per-session context-window usage for Claude Code statusline.

    Accepts either x-claude-code-session-id (from Claude Code itself) or
    x-session-name (from the session forwarder). Prefers session_name when
    both are present so that port-isolated sessions report correctly.
    """
    if not session_id and not session_name:
        return {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "request_count": 0,
            "context_limit": 0,
            "remaining_tokens": 0,
            "percentage_used": 0.0,
            "percent": 0.0,
            "model": None,
        }

    if session_name:
        usage = observability_recorder.fetch_context_usage_by_name(session_name)
    else:
        usage = observability_recorder.fetch_context_usage(session_id)
    if not usage:
        return {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "request_count": 0,
            "context_limit": 0,
            "remaining_tokens": 0,
            "percentage_used": 0.0,
            "percent": 0.0,
            "model": None,
        }

    backend = usage["backend_model"] or ""
    claude_model = usage["claude_model"] or ""
    real_total = usage["total_tokens"] or 0

    # Use the real model context limit but cap at 200K to align with Claude
    # Code's internal ceiling. The backend model may have more (256K), but
    # Claude Code caps at 200K, so the percentage must match what CC sees.
    CONTEXT_LIMIT = _get_context_limit(backend)
    CONTEXT_LIMIT = min(CONTEXT_LIMIT, 200_000) if CONTEXT_LIMIT > 0 else 200_000

    if CONTEXT_LIMIT > 0:
        raw_percentage = round(real_total / CONTEXT_LIMIT * 100, 2)
        # Apply user-configured offset so the statusline can read higher/lower
        adjusted = max(0, min(100, raw_percentage + config.statusline_percent_adjust))
        percentage = round(adjusted, 2)
        remaining = int(CONTEXT_LIMIT * (1 - percentage / 100))
    else:
        percentage = 0.0
        remaining = 0

    return {
        "total_tokens": real_total,
        "input_tokens": usage["input_tokens"] or 0,
        "output_tokens": usage["output_tokens"] or 0,
        "cache_read_input_tokens": usage["cache_read_input_tokens"] or 0,
        "cache_creation_input_tokens": usage["cache_creation_input_tokens"] or 0,
        "request_count": usage["request_count"] or 0,
        "context_limit": CONTEXT_LIMIT,
        "remaining_tokens": remaining,
        "percentage_used": percentage,
        "percent": percentage,
        "model": claude_model or backend,
    }


@router.get("/api/observability/ensemble/runs")
async def ensemble_runs(
    session_id: Optional[str] = Query(None),
    session_name: Optional[str] = Query(None),
    limit: int = Query(120, ge=1, le=500),
    _: None = Depends(validate_dashboard_api_key),
):
    """Ensemble races for a session: per-candidate outputs, verdicts, winner."""
    return {
        "mode": config.ensemble_mode,
        "models": config.ensemble_models,
        "runs": observability_recorder.fetch_ensemble_runs(
            session_id=session_id, session_name=session_name, limit=limit
        ),
    }


@router.get("/api/observability/ensemble/leaderboard")
async def ensemble_leaderboard(
    hours: int = Query(24, ge=1, le=8760),
    _: None = Depends(validate_dashboard_api_key),
):
    """Per-model win/loss aggregates across all ensemble races in the window."""
    return {
        "mode": config.ensemble_mode,
        "models": config.ensemble_models,
        "hours": hours,
        "leaderboard": observability_recorder.fetch_ensemble_leaderboard(hours=hours),
    }


@router.get("/api/observability/ensemble/pending")
async def ensemble_pending(_: None = Depends(validate_dashboard_api_key)):
    """Races currently held open waiting for the user's choice (approval mode)."""
    return {"mode": config.ensemble_mode, "pending": approval_store.list_pending()}


@router.post("/api/observability/ensemble/choose")
async def ensemble_choose(
    payload: dict,
    _: None = Depends(validate_dashboard_api_key),
):
    """Resolve a pending approval: {'request_id': ..., 'candidate_index': N}."""
    request_id = str(payload.get("request_id") or "")
    try:
        candidate_index = int(payload.get("candidate_index"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="candidate_index must be an integer")
    if not approval_store.choose(request_id, candidate_index):
        raise HTTPException(
            status_code=404,
            detail="No such pending approval (it may have timed out) or invalid candidate",
        )
    return {"ok": True, "request_id": request_id, "candidate_index": candidate_index}
