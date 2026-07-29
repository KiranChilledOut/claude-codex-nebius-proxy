from __future__ import annotations

import json
import logging
import re
import ssl
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

from src.core.config import config

if TYPE_CHECKING:  # avoids a runtime circular import (pricing -> observability -> store -> here)
    from src.observability.pricing import ModelPrice

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelInfo:
    model: str
    input_per_1m: Optional[float]
    output_per_1m: Optional[float]
    context_length: Optional[int]
    modality: Optional[str]
    description: Optional[str]


class ModelCatalog:
    """In-memory cache of provider model metadata from /models?verbose=true.

    refresh() fetches + parses (blocking, called at startup and on a timer).
    All getters are non-blocking dict reads, safe on the request path.
    """

    def __init__(self, base_url: str, api_key: str, enabled: bool = True):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.enabled = enabled
        self._models: Dict[str, ModelInfo] = {}

    def _fetch(self) -> dict:
        endpoint = self.base_url + "/models?verbose=true"
        req = urllib.request.Request(
            endpoint, headers={"Authorization": f"Bearer {self.api_key}"}
        )
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            return json.load(resp)

    def refresh(self) -> bool:
        if not self.enabled:
            return False
        try:
            payload = self._fetch()
        except Exception as exc:  # noqa: BLE001 — keep last-good on any failure
            logger.warning("Model catalog refresh failed: %s", exc)
            return False
        parsed: Dict[str, ModelInfo] = {}
        for m in payload.get("data", []):
            mid = m.get("id")
            if not mid:
                continue
            pricing = m.get("pricing") or {}
            parsed[mid] = ModelInfo(
                model=mid,
                input_per_1m=_per_million(pricing.get("prompt")),
                output_per_1m=_per_million(pricing.get("completion")),
                context_length=_as_int(m.get("context_length")),
                modality=(m.get("architecture") or {}).get("modality"),
                description=m.get("description"),
            )
        if parsed:
            self._models = parsed
        return bool(parsed)

    def get_pricing(self, model: Optional[str]) -> Optional["ModelPrice"]:
        if not model:
            return None
        info = self._models.get(model)
        if info is None or info.input_per_1m is None or info.output_per_1m is None:
            return None
        from src.observability.pricing import ModelPrice  # lazy to avoid circular import
        return ModelPrice(
            model=model,
            input_per_1m=info.input_per_1m,
            output_per_1m=info.output_per_1m,
            advertised_tok_s=None,
            currency="USD",
        )

    def get_context_length(self, model: Optional[str]) -> Optional[int]:
        if not model:
            return None
        info = self._models.get(model)
        return info.context_length if info else None

    def model_ids(self) -> List[str]:
        return list(self._models.keys())

    def chat_model_ids(self) -> List[str]:
        """Model ids usable for chat/tool sessions.

        Nebius returns embedding and reranking models from /models that
        cannot serve chat completions or tool calls; the session picker
        should never offer them. Filtered by id convention (mirage-style)
        because the verbose payload doesn't reliably mark chat modality.
        """
        return [i.model for i in self._models.values() if is_chat_capable(i.model)]

    def models_detailed(self) -> List[dict]:
        return [
            {
                "id": i.model,
                "context_length": i.context_length,
                "input_per_1m": i.input_per_1m,
                "output_per_1m": i.output_per_1m,
                "modality": i.modality,
                "description": i.description,
            }
            for i in self._models.values()
        ]


# Ids that are clearly not chat/tool-capable: embeddings and rerankers.
# Mirrors mirage's exclusion of embedding/rerank models from the chat catalog.
_NON_CHAT_ID = re.compile(r"(embed|rerank|bge-m3)", re.IGNORECASE)


def is_chat_capable(model_id: Optional[str]) -> bool:
    """False for embedding/reranking models that cannot serve chat/tool calls."""
    if not model_id:
        return False
    return not _NON_CHAT_ID.search(model_id)


def _per_million(raw) -> Optional[float]:
    try:
        return float(raw) * 1_000_000 if raw is not None else None
    except (TypeError, ValueError):
        return None


def _as_int(raw) -> Optional[int]:
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


model_catalog = ModelCatalog(
    config.openai_base_url, config.openai_api_key, enabled=config.model_catalog_enabled
)
