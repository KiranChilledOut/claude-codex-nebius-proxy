import base64
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.conversion.request_converter import (
    TOKEN_ESTIMATE_BUFFER,
    _estimate_prompt_tokens,
    convert_claude_to_openai,
)
from src.core.config import config
from src.core.model_manager import model_manager
from src.models.claude import (
    ClaudeContentBlockImage,
    ClaudeContentBlockText,
    ClaudeMessage,
    ClaudeMessagesRequest,
)


def test_image_request_sets_tool_choice_none_when_no_tools():
    """Vision requests should explicitly disable tools to appease providers that reject auto mode."""
    image_data = base64.b64encode(b"fake").decode("utf-8")
    request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=256,
        messages=[
            ClaudeMessage(
                role="user",
                content=[
                    ClaudeContentBlockText(type="text", text="what's in this image ?"),
                    ClaudeContentBlockImage(
                        type="image",
                        source={
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    ),
                ],
            )
        ],
    )

    openai_request = convert_claude_to_openai(request, model_manager)

    assert openai_request["model"] == config.vision_model
    assert openai_request.get("tool_choice") == "none"
    assert "tools" not in openai_request


def test_text_request_does_not_force_tool_choice_without_tools():
    request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=64,
        messages=[ClaudeMessage(role="user", content="hello")],
    )

    openai_request = convert_claude_to_openai(request, model_manager)

    assert "tool_choice" not in openai_request
    assert "tools" not in openai_request


def test_prompt_token_estimate_can_disable_context_safety_buffer():
    messages = [{"role": "user", "content": "hello"}]

    with_buffer = _estimate_prompt_tokens(messages)
    without_buffer = _estimate_prompt_tokens(messages, include_safety_buffer=False)

    assert with_buffer - without_buffer == TOKEN_ESTIMATE_BUFFER


def test_image_request_drops_tools_and_sets_none():
    """Image requests should drop tools and force tool_choice none."""
    image_data = base64.b64encode(b"fake").decode("utf-8")
    request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=256,
        messages=[
            ClaudeMessage(
                role="user",
                content=[
                    ClaudeContentBlockText(type="text", text="what's in this image ?"),
                    ClaudeContentBlockImage(
                        type="image",
                        source={
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    ),
                ],
            )
        ],
        tools=[
            {
                "name": "dummy",
                "description": "ignore",
                "input_schema": {"type": "object"},
            }
        ],
        tool_choice={"type": "auto"},
    )

    openai_request = convert_claude_to_openai(request, model_manager)

    assert openai_request["model"] == config.vision_model
    # Should drop tools and force tool_choice none
    assert "tools" not in openai_request
    assert openai_request.get("tool_choice") == "none"


def test_followup_without_image_keeps_tools_and_model():
    """If the latest user message has no image, keep tools and default model."""
    image_data = base64.b64encode(b"fake").decode("utf-8")
    # Conversation history includes an earlier image, but latest user is text-only
    request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=128,
        messages=[
            ClaudeMessage(
                role="user",
                content=[
                    ClaudeContentBlockText(type="text", text="what's in this image ?"),
                    ClaudeContentBlockImage(
                        type="image",
                        source={
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    ),
                ],
            ),
            ClaudeMessage(role="assistant", content="ALL DONE"),
            ClaudeMessage(role="user", content="create a file called test"),
        ],
        tools=[
            {
                "name": "dummy",
                "description": "ignore",
                "input_schema": {"type": "object"},
            }
        ],
        tool_choice={"type": "auto"},
    )

    openai_request = convert_claude_to_openai(request, model_manager)

    assert openai_request["model"] == config.middle_model
    assert openai_request.get("tools")
    assert openai_request.get("tool_choice") == "auto"
    # No image content should remain
    assert not any(
        isinstance(msg.get("content"), list)
        and any(part.get("type") == "image_url" for part in msg["content"])
        for msg in openai_request["messages"]
    )


def test_requested_max_tokens_is_not_forced_up_by_min_tokens_limit(monkeypatch):
    """MIN_TOKENS_LIMIT should only be a fallback, not an enforced floor."""
    original_min = config.min_tokens_limit
    try:
        monkeypatch.setattr(config, "min_tokens_limit", 4096)
        request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=64,
            messages=[ClaudeMessage(role="user", content="hello")],
        )

        openai_request = convert_claude_to_openai(request, model_manager)
        assert openai_request["max_tokens"] <= 64
    finally:
        monkeypatch.setattr(config, "min_tokens_limit", original_min)


def test_metadata_user_id_forwarded_as_user_and_prompt_cache_key():
    """metadata.user_id should reach the backend as `user` (attribution) and
    `prompt_cache_key` (prefix-cache routing affinity for subagent fleets)."""
    request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=64,
        messages=[ClaudeMessage(role="user", content="hello")],
        metadata={"user_id": "user_abc_session_123"},
    )

    openai_request = convert_claude_to_openai(request, model_manager)

    assert openai_request["user"] == "user_abc_session_123"
    assert openai_request["prompt_cache_key"] == "user_abc_session_123"


def test_no_metadata_means_no_user_or_prompt_cache_key():
    request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=64,
        messages=[ClaudeMessage(role="user", content="hello")],
    )

    openai_request = convert_claude_to_openai(request, model_manager)

    assert "user" not in openai_request
    assert "prompt_cache_key" not in openai_request
    assert "extra_body" not in openai_request


def test_top_k_forwarded_via_extra_body():
    request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=64,
        messages=[ClaudeMessage(role="user", content="hello")],
        top_k=40,
    )

    openai_request = convert_claude_to_openai(request, model_manager)

    assert openai_request["extra_body"]["top_k"] == 40


def test_claude_context_window_detection():
    from src.conversion.request_converter import claude_context_window

    assert claude_context_window("claude-sonnet-4-6") == 200_000
    assert claude_context_window("claude-fable-5") == 200_000
    assert claude_context_window("claude-sonnet-4-6[1m]") == 1_000_000
    assert claude_context_window("claude-fable-5", "context-1m-2025-08-07") == 1_000_000


def test_compute_usage_scale_maps_windows(monkeypatch):
    from src.conversion.request_converter import compute_usage_scale

    monkeypatch.setattr(config, "big_model", "backend/big")
    monkeypatch.setattr(config, "big_model_context_limit", 100_000)
    # 200K claude window over 100K backend window -> inflate 2x
    assert compute_usage_scale("claude-opus-4", "backend/big") == 2.0
    # 1M claude window over 100K backend window -> inflate 10x
    assert compute_usage_scale("claude-opus-4[1m]", "backend/big") == 10.0
    # Windows already aligned (within 2%) -> no scaling
    monkeypatch.setattr(config, "big_model_context_limit", 201_000)
    assert compute_usage_scale("claude-opus-4", "backend/big") == 1.0
