"""Upstream rate-limit pacing: Retry-After parsing and Anthropic error typing."""

from types import SimpleNamespace

from src.conversion.response_converter import error_type_for_status
from src.core.client import _retry_after_seconds


def _error_with_headers(headers):
    return SimpleNamespace(response=SimpleNamespace(headers=headers))


def test_retry_after_parses_plain_seconds():
    assert _retry_after_seconds(_error_with_headers({"retry-after": "7"})) == 7.0


def test_retry_after_parses_token_factory_reset_suffix():
    # Token Factory reports x-ratelimit-reset-requests as e.g. "13s"
    err = _error_with_headers({"x-ratelimit-reset-requests": "13s"})
    assert _retry_after_seconds(err) == 13.0


def test_retry_after_prefers_retry_after_header():
    err = _error_with_headers({"retry-after": "2", "x-ratelimit-reset-requests": "13s"})
    assert _retry_after_seconds(err) == 2.0


def test_retry_after_is_clamped_to_30s():
    assert _retry_after_seconds(_error_with_headers({"retry-after": "120"})) == 30.0


def test_retry_after_none_without_response_or_headers():
    assert _retry_after_seconds(None) is None
    assert _retry_after_seconds(SimpleNamespace(response=None)) is None
    assert _retry_after_seconds(_error_with_headers({})) is None
    assert _retry_after_seconds(_error_with_headers({"retry-after": "soon"})) is None


def test_error_type_for_status_mapping():
    assert error_type_for_status(429) == "rate_limit_error"
    assert error_type_for_status(401) == "authentication_error"
    assert error_type_for_status(400) == "invalid_request_error"
    assert error_type_for_status(529) == "overloaded_error"
    assert error_type_for_status(500) == "api_error"
    assert error_type_for_status(None) == "api_error"
