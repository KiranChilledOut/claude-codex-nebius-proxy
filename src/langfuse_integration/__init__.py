"""Langfuse integration for proxy observability and training data capture.

When ``LANGFUSE_ENABLED=true`` and API keys are set, every proxied LLM call is
traced to Langfuse (self-hosted or cloud) as a trace + generation pair carrying
the full input, output, usage, and error context. This makes prompt/response
pairs available for scoring, datasets, and eventual fine-tuning — the data the
SQLite dashboard deliberately does not store in full.

When disabled (or the ``langfuse`` package is absent), every call is a no-op,
so the proxy's behavior and performance are unchanged.
"""

from src.langfuse_integration.client import get_langfuse_client
from src.langfuse_integration.config import LangfuseConfig

__all__ = ["LangfuseConfig", "get_langfuse_client"]
