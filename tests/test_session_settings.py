from src.core.config import config
from src.core.session_settings import (
    clear_runtime_model,
    resolve_session_settings,
    set_runtime_model,
)


def test_falls_back_to_env_defaults_when_headers_absent():
    s = resolve_session_settings({})
    assert s.model == config.model
    assert s.ensemble_mode == config.ensemble_mode
    assert s.ensemble_models == list(config.ensemble_models)
    assert s.ensemble_judge == (config.ensemble_judge_model or None)


def test_reads_per_session_headers():
    headers = {
        "x-session-model": "deepseek-ai/DeepSeek-V4-Pro",
        "x-session-ensemble-mode": "Hedge",
        "x-session-ensemble-models": "a/m1, a/m2 ,a/m3",
        "x-session-ensemble-judge": "a/judge",
    }
    s = resolve_session_settings(headers)
    assert s.model == "deepseek-ai/DeepSeek-V4-Pro"
    assert s.ensemble_mode == "hedge"  # lowercased
    assert s.ensemble_models == ["a/m1", "a/m2", "a/m3"]  # trimmed CSV
    assert s.ensemble_judge == "a/judge"


def test_header_lookup_is_case_insensitive_and_blank_judge_is_none():
    headers = {"X-Session-Model": "a/m", "x-session-ensemble-judge": "   "}
    s = resolve_session_settings(headers)
    assert s.model == "a/m"
    assert s.ensemble_judge == (config.ensemble_judge_model or None)


def test_runtime_override_wins_over_header(tmp_path):
    set_runtime_model("runtime-sess", "override/model")
    try:
        headers = {"x-session-name": "runtime-sess", "x-session-model": "header/model"}
        s = resolve_session_settings(headers)
        assert s.model == "override/model"
    finally:
        clear_runtime_model("runtime-sess")


def test_runtime_override_only_affects_named_session():
    set_runtime_model("runtime-sess", "override/model")
    try:
        # A different session name must not see the override.
        s = resolve_session_settings({"x-session-name": "other-sess", "x-session-model": "h/model"})
        assert s.model == "h/model"
        # A session with no name falls back through the header/config chain.
        s2 = resolve_session_settings({"x-session-model": "h/model"})
        assert s2.model == "h/model"
    finally:
        clear_runtime_model("runtime-sess")


def test_clear_runtime_restores_header_fallback():
    set_runtime_model("runtime-sess", "override/model")
    assert clear_runtime_model("runtime-sess")
    headers = {"x-session-name": "runtime-sess", "x-session-model": "header/model"}
    assert resolve_session_settings(headers).model == "header/model"


def test_empty_session_name_is_ignored():
    set_runtime_model("   ", "ignored/model")  # should be a no-op
    assert resolve_session_settings({"x-session-name": "  ", "x-session-model": "h/model"}).model == "h/model"
