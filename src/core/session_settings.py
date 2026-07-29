from dataclasses import dataclass
from typing import List, Mapping, Optional

from src.core.config import config


@dataclass
class EffectiveSettings:
    """Per-request settings resolved from session headers with env fallbacks.

    Built fresh for every request so concurrent sessions never share state.
    """

    model: str
    ensemble_mode: str
    ensemble_models: List[str]
    ensemble_judge: Optional[str]


# Process-local runtime model overrides keyed by session name. Lets a running
# session switch its backend model without a restart (set via PUT /v1/session-model).
# Not persisted: a proxy restart reverts each session to its forwarder startup model.
_RUNTIME_OVERRIDES: dict[str, str] = {}


def set_runtime_model(session_name: str, model: str) -> None:
    name = session_name.strip()
    if name:
        _RUNTIME_OVERRIDES[name] = model


def get_runtime_model(session_name: str) -> Optional[str]:
    name = session_name.strip()
    return _RUNTIME_OVERRIDES.get(name) if name else None


def clear_runtime_model(session_name: str) -> bool:
    name = session_name.strip()
    return _RUNTIME_OVERRIDES.pop(name, None) is not None if name else False


def resolve_session_settings(headers: Mapping[str, str]) -> EffectiveSettings:
    norm = {str(k).lower(): v for k, v in dict(headers).items()}

    def _get(name: str) -> str:
        return (norm.get(name) or "").strip()

    # Runtime override (set from the dashboard picker) wins over the forwarder's
    # x-session-model header, which in turn wins over the global config default.
    session_name = _get("x-session-name")
    runtime_model = get_runtime_model(session_name)
    model = runtime_model or _get("x-session-model") or config.model
    mode = _get("x-session-ensemble-mode").lower() or config.ensemble_mode

    models_csv = _get("x-session-ensemble-models")
    if models_csv:
        ensemble_models = [m.strip() for m in models_csv.split(",") if m.strip()]
    else:
        ensemble_models = list(config.ensemble_models)

    judge = _get("x-session-ensemble-judge") or (config.ensemble_judge_model or None)

    return EffectiveSettings(
        model=model,
        ensemble_mode=mode,
        ensemble_models=ensemble_models,
        ensemble_judge=judge,
    )
