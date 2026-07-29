"""Configuration for the Langfuse integration.

Reads its settings from environment variables so it can be enabled/disabled
without code changes. Langfuse is an optional dependency; when
``LANGFUSE_ENABLED`` is false or the SDK is not installed every client call
becomes a no-op (see ``client.py``).
"""

import os


def _as_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class LangfuseConfig:
    """Holds Langfuse connection settings sourced from the environment."""

    def __init__(self):
        # Master switch. When false the client never initializes the SDK and
        # every method becomes a no-op.
        self.enabled = _as_bool(os.environ.get("LANGFUSE_ENABLED"), default=False)

        # Self-hosted or cloud Langfuse base URL (e.g. http://localhost:8084).
        self.host = os.environ.get("LANGFUSE_HOST", "http://localhost:8084")

        # Langfuse trace URLs require a project id in the path:
        # {host}/project/{projectId}/traces/{traceId}. The dashboard uses this
        # to deep-link a request row to its trace. Empty = link hidden (degraded,
        # not broken). Find it in the Langfuse UI URL bar after selecting a project.
        self.project_id = os.environ.get("LANGFUSE_PROJECT_ID", "").strip()

        # Public/secret API keys issued by Langfuse. Required for the SDK to
        # authenticate; empty by default so the feature stays inert until set.
        self.public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        self.secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")

        # How often (seconds) the SDK flushes its in-memory queue to the server.
        self.flush_interval = _as_int(os.environ.get("LANGFUSE_FLUSH_INTERVAL"), 10)

        # Maximum number of events queued internally before the SDK drops them.
        self.max_queue_size = _as_int(os.environ.get("LANGFUSE_MAX_QUEUE_SIZE"), 1000)

    def is_configured(self):
        """True only when both API keys are present (i.e. usable)."""
        return bool(self.public_key) and bool(self.secret_key)
