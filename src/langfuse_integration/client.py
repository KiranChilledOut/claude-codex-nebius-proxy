"""Wrapper around the Langfuse Python SDK providing a lazy no-op client.

Targets Langfuse SDK v4 (OpenTelemetry-based). When ``langfuse`` is not
installed or ``LangfuseConfig.enabled`` is ``False``, every method silently
returns a placeholder value and logs at debug level.

v4 API pattern
  Traces and generations are *observations* on an OTLP span tree. Each
  observation is created via ``client.start_observation(trace_context=...,
  as_type='generation', ...)``, which returns a context object exposing
  ``.update()``, ``.end()``, ``.id`` and ``.trace_id``. To bind an
  observation to a known trace (the proxy request id), pass a
  ``TraceContext(trace_id=...)``.

  The client is synchronous but queues everything for batched background
  upload, so callers inside asyncio handlers don't block.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    from langfuse import Langfuse
    from langfuse.types import TraceContext

    _LANGFUSE_AVAILABLE = True
except ImportError:
    Langfuse = None  # type: ignore[assignment, misc]
    TraceContext = None  # type: ignore[assignment, misc]
    _LANGFUSE_AVAILABLE = False

from src.langfuse_integration.config import LangfuseConfig

logger = logging.getLogger(__name__)


def _serialize(value: Any) -> Any:
    """Coerce a value into something Langfuse can store.

    dicts/lists pass through unchanged so the Langfuse UI renders them as
    expandable JSON trees; anything else is JSON-encoded with a ``str``
    fallback so arbitrary objects (Pydantic models, etc.) never break
    ingestion.
    """
    import json

    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


class LangfuseClient:
    """Observability client backed by the Langfuse v4 SDK.

    Intentionally *not* async — the SDK uses an internal queue with async
    batching for uploads, so direct calls are non-blocking and safe inside
    asyncio request handlers.
    """

    def __init__(self, config: LangfuseConfig) -> None:
        self.config = config
        self._client: Optional[Any] = None  # Langfuse instance or None
        self._missing_sdk_warned: bool = False
        # Maps generation_id -> observation handle, so end_generation can
        # finalize the right observation. The SDK doesn't expose a generic
        # "fetch observation by id and mutate" entrypoint in v4.
        self._observations: Dict[str, Any] = {}

    def _ensure_client(self) -> None:
        """Lazily construct the SDK client on first enabled call."""
        if self._client is not None:
            return
        if not self.config.enabled:
            return
        if not self.config.is_configured():
            logger.debug("Langfuse: enabled but keys are empty — skipping init")
            return
        if not _LANGFUSE_AVAILABLE:
            if not self._missing_sdk_warned:
                logger.warning(
                    "Langfuse: enabled but 'langfuse' package is not installed — "
                    "all calls are no-ops"
                )
                self._missing_sdk_warned = True
            return

        self._client = Langfuse(
            public_key=self.config.public_key,
            secret_key=self.config.secret_key,
            host=self.config.host,
            flush_interval=self.config.flush_interval,
            flush_at=self.config.max_queue_size,
        )
        logger.debug(
            "Langfuse client initialised (host=%s, flush_interval=%ss, "
            "flush_at=%s)",
            self.config.host,
            self.config.flush_interval,
            self.config.max_queue_size,
        )

    # ------------------------------------------------------------------
    # Trace
    # ------------------------------------------------------------------

    def start_trace(
        self,
        id: Optional[str] = None,
        name: str = "proxy-request",
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
    ) -> str:
        """Register a Langfuse trace and return its id.

        *id* may be a pre-existing string (the proxy request id); when None
        a UUID is generated. The trace id is reused by subsequent
        ``start_generation``/``create_span`` calls.

        Langfuse v4 requires trace ids to be 32-char lowercase hex. When
        *id* is provided but doesn't match that format, we generate a new
        trace id and store the custom correlation key in metadata.
        """
        self._ensure_client()
        if self._client is None:
            return id or uuid.uuid4().hex

        trace_id = uuid.uuid4().hex
        if id and self._is_valid_trace_id(id):
            trace_id = id
        elif id:
            # The proxy request_id is a uuid but contains hyphens; Langfuse
            # needs plain hex. Store the original for dashboard correlation.
            logger.debug(
                "Langfuse: id=%r is not a valid v4 trace id — generating %s", id, trace_id
            )
            if metadata is None:
                metadata = {}
            metadata["proxy_request_id"] = id

        ctx = TraceContext(trace_id=trace_id) if TraceContext else None
        # Save user/tags onto the per-trace state for the generation call.
        self._trace_user_ids = getattr(self, "_trace_user_ids", {}) or {}
        if user_id:
            self._trace_user_ids[trace_id] = user_id
        logger.debug("Langfuse: trace %s started", trace_id)
        return trace_id

    @staticmethod
    def _is_valid_trace_id(candidate: str) -> bool:
        """Langfuse v4 trace ids must be 32-char lowercase hex."""
        if len(candidate) != 32:
            return False
        try:
            int(candidate, 16)
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Generation (LLM call)
    # ------------------------------------------------------------------

    def start_generation(
        self,
        *,
        trace_id: str,
        name: str = "llm-call",
        model: Optional[str] = None,
        input_data: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> str:
        """Log the start of an LLM generation, returning its observation id."""
        self._ensure_client()
        if self._client is None:
            return uuid.uuid4().hex

        gen_id = uuid.uuid4().hex
        ctx = TraceContext(trace_id=trace_id) if TraceContext else None

        kwargs: Dict[str, Any] = {
            "trace_context": ctx,
            "name": name,
            "as_type": "generation",
            "input": _serialize(input_data) if input_data is not None else None,
        }
        if model:
            kwargs["model"] = model
        if metadata:
            kwargs["metadata"] = metadata
        if prompt_tokens or completion_tokens or total_tokens:
            kwargs["usage_details"] = {
                "input": prompt_tokens,
                "output": completion_tokens,
                "total": total_tokens,
            }

        observation = self._client.start_observation(**kwargs)
        # The SDK assigns its own observation id; we keep our handle so the
        # proxy can finalize by our own gen id without tracking SDK ids.
        self._observations[gen_id] = observation
        logger.debug(
            "Langfuse: generation %s (%s) started on trace %s",
            gen_id,
            name,
            trace_id,
        )
        return gen_id

    def end_generation(
        self,
        *,
        generation_id: str,
        output_data: Any = None,
        usage: Optional[Dict[str, Any]] = None,
        cost: Optional[Dict[str, float]] = None,
        model: Optional[str] = None,
        input_data: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        status: str = "success",
        error_message: Optional[str] = None,
    ) -> None:
        """Complete a previously started generation.

        ``usage`` is a dict like ``{"input": N, "output": N, "total": N}``.
        ``cost`` is a dict like ``{"input": f, "output": f, "total": f}``.
        ``model``/``input_data``/``metadata`` are attached here when they were
        not known at ``start_generation`` time (e.g. the real backend model is
        only resolved after request conversion). When ``status`` is ``"error"``,
        the generation is ended at warning level with *error_message* as the
        status message.
        """
        self._ensure_client()
        if self._client is None:
            return

        observation = self._observations.pop(generation_id, None)
        if observation is None:
            logger.debug(
                "Langfuse: generation %s not found (already ended?)", generation_id
            )
            return

        update_kwargs: Dict[str, Any] = {}
        if output_data is not None:
            update_kwargs["output"] = _serialize(output_data)
        if input_data is not None:
            update_kwargs["input"] = _serialize(input_data)
        if model:
            update_kwargs["model"] = model
        if metadata:
            update_kwargs["metadata"] = metadata
        if usage is not None:
            update_kwargs["usage_details"] = {
                "input": int(usage.get("input") or 0),
                "output": int(usage.get("output") or 0),
                "total": int(usage.get("total") or 0),
            }
        if cost is not None:
            # Langfuse v4 stores cost as a free-form float map; the proxy
            # writes input/output/total so the UI's per-generation cost column
            # is populated without relying on Langfuse's own pricing table.
            update_kwargs["cost_details"] = {
                "input": float(cost.get("input") or 0),
                "output": float(cost.get("output") or 0),
                "total": float(cost.get("total") or 0),
            }
        level = "ERROR" if status == "error" else "DEFAULT"
        status_message = error_message if status == "error" else None
        update_kwargs["level"] = level
        if status_message:
            update_kwargs["status_message"] = status_message

        try:
            observation.update(**update_kwargs)
            observation.end()
        except Exception as exc:  # never let observability break the request
            logger.debug("Langfuse: failed to end generation %s: %s", generation_id, exc)

        logger.debug(
            "Langfuse: generation %s ended (status=%s)", generation_id, status
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_generation(
        self,
        *,
        trace_id: str,
        generation_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
    ) -> None:
        """Attach a numeric score to a generation.

        Use this for post-hoc quality signals: response quality ratings,
        user thumbs-up/thumbs-down, automated-correctness checks, etc.
        """
        self._ensure_client()
        if self._client is None:
            return

        observation = self._observations.get(generation_id)
        observation_id = getattr(observation, "id", None) if observation else None
        try:
            self._client.create_score(
                trace_id=trace_id,
                observation_id=observation_id,
                name=name,
                value=value,
                data_type="NUMERIC",
                comment=comment,
            )
        except Exception as exc:
            logger.debug("Langfuse: failed to score generation %s: %s", generation_id, exc)
        logger.debug(
            "Langfuse: score '%s'=%s on generation %s", name, value, generation_id
        )

    # ------------------------------------------------------------------
    # Spans
    # ------------------------------------------------------------------

    def create_span(
        self,
        *,
        trace_id: str,
        name: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a completed span within a trace.

        Useful for instrumenting non-LLM steps: request conversion, web
        search loops, codex session lookups, etc.
        """
        self._ensure_client()
        if self._client is None:
            return

        ctx = TraceContext(trace_id=trace_id) if TraceContext else None
        kwargs: Dict[str, Any] = {
            "trace_context": ctx,
            "name": name,
            "as_type": "span",
        }
        if metadata:
            kwargs["metadata"] = metadata
        try:
            observation = self._client.start_observation(**kwargs)
            observation.end()
        except Exception as exc:
            logger.debug("Langfuse: failed to create span '%s': %s", name, exc)
        logger.debug("Langfuse: span '%s' on trace %s", name, trace_id)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def add_event_to_trace(
        self,
        *,
        trace_id: str,
        event_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Attach a discrete event to a trace.

        Events are lightweight markers — tool call emissions, client
        disconnects, request cancellations, etc.
        """
        self._ensure_client()
        if self._client is None:
            return

        ctx = TraceContext(trace_id=trace_id) if TraceContext else None
        kwargs: Dict[str, Any] = {"trace_context": ctx, "name": event_name}
        if metadata:
            kwargs["metadata"] = metadata
        try:
            self._client.create_event(**kwargs)
        except Exception as exc:
            logger.debug("Langfuse: failed to create event '%s': %s", event_name, exc)
        logger.debug("Langfuse: event '%s' on trace %s", event_name, trace_id)

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Force-flush all queued events to the Langfuse server."""
        self._ensure_client()
        if self._client is None:
            return

        try:
            self._client.flush()
        except Exception as exc:
            logger.debug("Langfuse: flush failed: %s", exc)
        logger.debug("Langfuse: flush complete")

    def shutdown(self) -> None:
        """Flush and shut down the SDK's background workers (call on app shutdown)."""
        self._ensure_client()
        if self._client is None:
            return
        try:
            self._client.shutdown()
        except Exception as exc:
            logger.debug("Langfuse: shutdown failed: %s", exc)


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_langfuse_client: Optional[LangfuseClient] = None


def get_langfuse_client() -> LangfuseClient:
    """Return the process-wide ``LangfuseClient`` singleton.

    On first call the client reads configuration from the environment. This
    is cheap (no network I/O — the SDK is initialised lazily on first real
    usage), so there is no penalty in calling it early.
    """
    global _langfuse_client
    if _langfuse_client is None:
        _langfuse_client = LangfuseClient(LangfuseConfig())
    return _langfuse_client
