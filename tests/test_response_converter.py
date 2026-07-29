import json

import pytest

from src.conversion.response_converter import (
    _finalize_tool_args,
    claude_response_to_sse,
    _sanitize_tool_arguments,
    convert_openai_streaming_to_claude_with_cancellation,
    convert_openai_to_claude_response,
)
from src.models.claude import ClaudeMessage, ClaudeMessagesRequest


class _DummyRequest:
    async def is_disconnected(self):
        return False


class _DummyClient:
    def cancel_request(self, _request_id):
        return True


class _DummyLogger:
    def debug(self, *_args, **_kwargs):
        pass

    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


def test_sanitize_tool_arguments_extracts_xml_payload():
    name, arguments = _sanitize_tool_arguments(
        "Bash",
        "<arg_key>command</arg_key><arg_value>ls -la</arg_value>",
    )

    assert name == "Bash"
    assert json.loads(arguments) == {"command": "ls -la"}


def test_sanitize_tool_arguments_extracts_args_embedded_in_name():
    name, arguments = _sanitize_tool_arguments('bash(command="ls -la")', "")

    assert name == "bash"
    assert json.loads(arguments) == {"command": "ls -la"}


def test_non_streaming_response_sanitizes_tool_calls():
    request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=64,
        messages=[ClaudeMessage(role="user", content="hello")],
    )
    openai_response = {
        "id": "resp_1",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": 'bash(command="ls -la")',
                                "arguments": "",
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    response = convert_openai_to_claude_response(openai_response, request)

    assert response["stop_reason"] == "tool_use"
    assert response["content"] == [
        {
            "type": "tool_use",
            "id": "call_1",
            "name": "bash",
            "input": {"command": "ls -la"},
        }
    ]


async def _fake_stream():
    # Regular text delta
    yield "data: " + json.dumps({"choices": [{"delta": {"content": "A"}, "finish_reason": None}]})
    # Completion marker chunk
    yield "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    # Unexpected chunk after finish_reason that should be ignored
    yield "data: " + json.dumps({"choices": [{"delta": {"content": "B"}, "finish_reason": None}]})
    yield "data: [DONE]"


async def _fake_tool_stream():
    yield "data: " + json.dumps(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": 'bash(command="ls -la")',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
    )
    yield "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    yield "data: [DONE]"


@pytest.mark.asyncio
async def test_streaming_stops_after_finish_reason():
    request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=64,
        messages=[ClaudeMessage(role="user", content="hello")],
        stream=True,
    )

    events = []
    async for event in convert_openai_streaming_to_claude_with_cancellation(
        _fake_stream(),
        request,
        _DummyLogger(),
        _DummyRequest(),
        _DummyClient(),
        "req_1",
    ):
        events.append(event)

    serialized = "".join(events)
    assert '"text": "A"' in serialized
    assert '"text": "B"' not in serialized
    assert "event: message_stop" in serialized


@pytest.mark.asyncio
async def test_streaming_flushes_sanitized_tool_arguments_on_finish():
    request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=64,
        messages=[ClaudeMessage(role="user", content="run ls")],
        stream=True,
    )

    events = []
    async for event in convert_openai_streaming_to_claude_with_cancellation(
        _fake_tool_stream(),
        request,
        _DummyLogger(),
        _DummyRequest(),
        _DummyClient(),
        "req_tool_1",
    ):
        events.append(event)

    serialized = "".join(events)
    assert '"type": "tool_use"' in serialized
    assert '"name": "bash"' in serialized
    assert '"partial_json": "{\\"command\\": \\"ls -la\\"}"' in serialized
    assert '"stop_reason": "tool_use"' in serialized


# --------------------------------------------------------------------------
# Kimi-K2 native control-token tool calls (leak when a tool is forwarded
# without a real parameter schema, e.g. Anthropic server tools like web_search)
# --------------------------------------------------------------------------
def test_sanitize_strips_kimi_control_tokens():
    blob = (
        ' <|tool_calls_section_begin|> <|tool_call_begin|> functions.web_search:0 '
        '<|tool_call_argument_begin|> {"query": "spacex launch date"} '
        '<|tool_call_end|> <|tool_calls_section_end|>'
    )
    name, args = _sanitize_tool_arguments("web_search", blob)
    assert name == "web_search"
    assert json.loads(args) == {"query": "spacex launch date"}


def test_sanitize_kimi_name_from_function_token():
    # Name leaks into the blob; clean it from functions.NAME:N
    blob = (
        '<|tool_call_begin|> functions.web_fetch:1 <|tool_call_argument_begin|> '
        '{"url": "https://example.com"} <|tool_call_end|>'
    )
    name, args = _sanitize_tool_arguments("functions.web_fetch:1", blob)
    assert name == "web_fetch"
    assert json.loads(args) == {"url": "https://example.com"}


def test_finalize_tool_args_kimi_roundtrip():
    blob = (
        '<|tool_call_argument_begin|> {"query": "hello world"} <|tool_call_end|>'
    )
    name, repaired, parsed = _finalize_tool_args("web_search", blob)
    assert name == "web_search"
    assert parsed == {"query": "hello world"}


def test_sanitize_clean_args_unaffected():
    # Normal clean JSON must pass through untouched.
    name, args = _sanitize_tool_arguments("WebSearch", '{"query": "x"}')
    assert name == "WebSearch"
    assert json.loads(args) == {"query": "x"}


# --------------------------------------------------------------------------
# claude_response_to_sse: synthetic streaming must preserve tool_use blocks
# (regression: optimized_response_to_sse dropped them, breaking tool calls
#  routed through the server-search loop).
# --------------------------------------------------------------------------
def test_claude_response_to_sse_emits_tool_use():
    resp = {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "x",
        "content": [
            {"type": "text", "text": "Let me read that."},
            {"type": "tool_use", "id": "tu_1", "name": "Read",
             "input": {"file_path": "/x/settings.json"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 5, "output_tokens": 7},
    }
    sse = "".join(claude_response_to_sse(resp))
    assert '"type": "tool_use"' in sse
    assert '"name": "Read"' in sse
    assert '"input_json_delta"' in sse
    assert "/x/settings.json" in sse  # the args actually made it through
    assert '"stop_reason": "tool_use"' in sse
    assert "event: message_stop" in sse


def test_claude_response_to_sse_text_only():
    resp = {"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn", "usage": {}}
    sse = "".join(claude_response_to_sse(resp))
    assert '"text": "hi"' in sse
    assert "event: message_stop" in sse


def test_claude_response_to_sse_empty_content():
    resp = {"content": [], "stop_reason": "end_turn", "usage": {}}
    sse = "".join(claude_response_to_sse(resp))
    # still emits a valid lifecycle with at least one block
    assert "content_block_start" in sse
    assert "event: message_stop" in sse


def test_usage_excludes_cached_tokens_from_input_tokens():
    """Anthropic input_tokens excludes cached tokens; OpenAI prompt_tokens
    includes them. Splitting wrong double-counts context in Claude Code."""
    from src.conversion.response_converter import _extract_usage

    usage = _extract_usage(
        {
            "prompt_tokens": 5862,
            "completion_tokens": 8,
            "prompt_tokens_details": {"cached_tokens": 5856},
        }
    )

    assert usage["input_tokens"] == 6
    assert usage["cache_read_input_tokens"] == 5856
    assert usage["cache_creation_input_tokens"] == 0
    assert usage["output_tokens"] == 8


def test_usage_without_cache_details_is_unchanged():
    from src.conversion.response_converter import _extract_usage

    usage = _extract_usage({"prompt_tokens": 100, "completion_tokens": 20})

    assert usage["input_tokens"] == 100
    assert usage["cache_read_input_tokens"] == 0
    assert usage["cache_creation_input_tokens"] == 0


def test_scale_usage_for_client_scales_input_side_only():
    from src.conversion.response_converter import scale_usage_for_client

    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_creation_input_tokens": 10,
        "cache_read_input_tokens": 1000,
    }
    scaled = scale_usage_for_client(usage, 2.0)

    assert scaled["input_tokens"] == 200
    assert scaled["cache_creation_input_tokens"] == 20
    assert scaled["cache_read_input_tokens"] == 2000
    # output_tokens must NOT scale: Claude Code enforces
    # CLAUDE_CODE_MAX_OUTPUT_TOKENS against this field.
    assert scaled["output_tokens"] == 50
    # scale 1.0 is a no-op passthrough
    assert scale_usage_for_client(usage, 1.0) is usage


# --------------------------------------------------------------------------
# Inline-text tool-call lifter: when a backend (e.g. moonshotai/Kimi-K2.7-Code)
# emits tool calls as `  ...  `
# control tokens inside `delta.content` (instead of structured `delta.tool_calls`),
# the streaming converter must lift them into proper `tool_use` content blocks
# with `stop_reason: tool_use` so Claude Code actually runs the tool.
# --------------------------------------------------------------------------

# Production-log blob (Kimi-K2.7-Code). Notably the token after ` `
# is a bare tool-call id (chatcmpl-tool-...) NOT a `functions.NAME:N` form, so the
# function name must be recovered from the args' keys.
_BASH_TOOL = {
    "name": "Bash",
    "description": "Run a bash command",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "description": {"type": "string"},
            "timeout": {"type": "number"},
        },
    },
}
_READ_TOOL = {
    "name": "Read",
    "description": "Read a file",
    "input_schema": {
        "type": "object",
        "properties": {"file_path": {"type": "string"}},
    },
}

_SECTION_BEGIN = "<" + "|tool_calls_section_begin" + "|>"
_SECTION_END = "<" + "|tool_calls_section_end" + "|>"
_CALL_BEGIN = "<" + "|tool_call_begin" + "|>"
_ARG_BEGIN = "<" + "|tool_call_argument_begin" + "|>"
_CALL_END = "<" + "|tool_call_end" + "|>"


_INLINE_KIMI_BLOB = (
    ' The clone failed. Let me verify the cert status. '
    + _SECTION_BEGIN + ' ' + _CALL_BEGIN + ' '
    + 'chatcmpl-tool-9d65acabff1f47df ' + _ARG_BEGIN + ' '
    + '{"command": "ssh-keygen -L", "description": "Check SSH cert validity"} '
    + _CALL_END + ' ' + _SECTION_END
)


def _request_with_tools(tools):
    return ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=64,
        messages=[ClaudeMessage(role="user", content="run it")],
        stream=True,
        tools=tools,
    )


async def _inline_tool_stream(text):
    yield "data: " + json.dumps(
        {"choices": [{"delta": {"content": text}, "finish_reason": None}]}
    )
    yield "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    yield "data: [DONE]"


async def _inline_tool_stream_split(part_a, part_b):
    yield "data: " + json.dumps(
        {"choices": [{"delta": {"content": part_a}, "finish_reason": None}]}
    )
    yield "data: " + json.dumps(
        {"choices": [{"delta": {"content": part_b}, "finish_reason": None}]}
    )
    yield "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    yield "data: [DONE]"


@pytest.mark.asyncio
async def test_streaming_lifts_inline_kimi_tool_call():
    request = _request_with_tools([_BASH_TOOL, _READ_TOOL])
    events = []
    async for event in convert_openai_streaming_to_claude_with_cancellation(
        _inline_tool_stream(_INLINE_KIMI_BLOB),
        request,
        _DummyLogger(),
        _DummyRequest(),
        _DummyClient(),
        "req_inline_1",
    ):
        events.append(event)
    serialized = "".join(events)

    # Preceding prose still surfaces as text.
    assert "Let me verify the cert status" in serialized
    # Raw control tokens must NOT leak into the client stream.
    assert _SECTION_BEGIN not in serialized
    assert _CALL_BEGIN not in serialized
    # A proper tool_use block is lifted, with the name recovered from the
    # `command` arg key against the request's Bash tool schema.
    assert '"type": "tool_use"' in serialized
    assert '"name": "Bash"' in serialized
    assert "ssh-keygen -L" in serialized
    # Turn must end as tool_use so Claude Code runs the tool.
    assert '"stop_reason": "tool_use"' in serialized


@pytest.mark.asyncio
async def test_streaming_lifts_inline_tool_call_split_across_chunks():
    # Split the blob right in the middle of the section-begin token so the
    # hold-back guard must buffer across the chunk boundary.
    split_at = _INLINE_KIMI_BLOB.index(_SECTION_BEGIN) + 3
    part_a, part_b = _INLINE_KIMI_BLOB[:split_at], _INLINE_KIMI_BLOB[split_at:]
    request = _request_with_tools([_BASH_TOOL, _READ_TOOL])
    events = []
    async for event in convert_openai_streaming_to_claude_with_cancellation(
        _inline_tool_stream_split(part_a, part_b),
        request,
        _DummyLogger(),
        _DummyRequest(),
        _DummyClient(),
        "req_inline_2",
    ):
        events.append(event)
    serialized = "".join(events)

    assert _SECTION_BEGIN not in serialized
    assert '"type": "tool_use"' in serialized
    assert '"name": "Bash"' in serialized
    assert "ssh-keygen -L" in serialized
    assert '"stop_reason": "tool_use"' in serialized


@pytest.mark.asyncio
async def test_streaming_inline_tool_call_unknown_name_falls_back():
    # Args keys match no tool in the request; the lifter must still emit a
    # tool_use block so the turn stays executable instead of dying as text.
    blob = (
        _SECTION_BEGIN + ' ' + _CALL_BEGIN + ' '
        + 'chatcmpl-tool-deadbeef ' + _ARG_BEGIN + ' '
        + '{"weird_key": "value"} ' + _CALL_END + ' ' + _SECTION_END
    )
    request = _request_with_tools([_READ_TOOL])
    events = []
    async for event in convert_openai_streaming_to_claude_with_cancellation(
        _inline_tool_stream(blob),
        request,
        _DummyLogger(),
        _DummyRequest(),
        _DummyClient(),
        "req_inline_3",
    ):
        events.append(event)
    serialized = "".join(events)

    assert '"type": "tool_use"' in serialized
    assert "weird_key" in serialized
    assert '"stop_reason": "tool_use"' in serialized
    assert _SECTION_BEGIN not in serialized



@pytest.mark.asyncio
async def test_nonstreaming_lifts_inline_kimi_tool_call():
    """Non-streaming convert_openai_to_claude_response also lifts inline-text
    tool calls (same backend glitch, different code path)."""
    _SEC = "<" + "|tool_calls_section_begin" + "|>"
    _END = "<" + "|tool_calls_section_end" + "|>"
    _CB = "<" + "|tool_call_begin" + "|>"
    _AB = "<" + "|tool_call_argument_begin" + "|>"
    _CE = "<" + "|tool_call_end" + "|>"
    request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=64,
        messages=[ClaudeMessage(role="user", content="run it")],
        tools=[_BASH_TOOL, _READ_TOOL],
    )
    text = (
        " The clone failed. "
        + _SEC + " " + _CB + " chatcmpl-tool-abc " + _AB + " "
        + "{\"command\": \"ssh-keygen -L\", \"description\": \"d\"} "
        + _CE + " " + _END
    )
    openai_response = {
        "id": "resp_n1",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": text, "tool_calls": []},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    resp = convert_openai_to_claude_response(openai_response, request)

    assert resp["stop_reason"] == "tool_use"
    tool_uses = [b for b in resp["content"] if b.get("type") == "tool_use"]
    assert tool_uses[0]["name"] == "Bash"
    assert tool_uses[0]["input"]["command"] == "ssh-keygen -L"
    # Raw control tokens must not survive into any content block.
    assert _SEC not in json.dumps(resp["content"])
    # Prose text still surfaced.
    texts = [b for b in resp["content"] if b.get("type") == "text"]
    assert any("clone failed" in b["text"] for b in texts)


# --------------------------------------------------------------------------
# Kimi-K2.7-Code bare-args variant (production logs.txt, 2026-07-20): the
# <|tool_call_argument_begin|> token is omitted entirely — the args JSON
# follows the bare tool-call id directly:
#   <|tool_call_begin|> chatcmpl-tool-<hex>   {"file_path": ...} <|tool_call_end|>
# --------------------------------------------------------------------------

_WRITE_TOOL = {
    "name": "Write",
    "description": "Write a file",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "content": {"type": "string"},
        },
    },
}

_INLINE_KIMI_BARE_BLOB = (
    ' Now creating the DSL file. '
    + _SECTION_BEGIN + ' ' + _CALL_BEGIN + ' '
    + 'chatcmpl-tool-a06b5f9a1e9a6e0b6   '
    + '{"file_path": "/tmp/AzureAiPlatform_Teams_ItsNagnt.kt", "content": "package foo\\n"} '
    + _CALL_END + ' ' + _SECTION_END
)


@pytest.mark.asyncio
async def test_streaming_lifts_inline_kimi_bare_args_tool_call():
    """Args JSON directly after the id (no <|tool_call_argument_begin|>) must
    still be lifted into a tool_use block, with the name recovered from the
    args keys against the Write tool schema."""
    request = _request_with_tools([_BASH_TOOL, _READ_TOOL, _WRITE_TOOL])
    events = []
    async for event in convert_openai_streaming_to_claude_with_cancellation(
        _inline_tool_stream(_INLINE_KIMI_BARE_BLOB),
        request,
        _DummyLogger(),
        _DummyRequest(),
        _DummyClient(),
        "req_inline_bare_1",
    ):
        events.append(event)
    serialized = "".join(events)

    # Preceding prose still surfaces as text.
    assert "Now creating the DSL file" in serialized
    # Raw control tokens and the bare id/args must NOT leak as text.
    assert _SECTION_BEGIN not in serialized
    assert _CALL_BEGIN not in serialized
    # Lifted tool_use with the name recovered from file_path+content keys.
    assert '"type": "tool_use"' in serialized
    assert '"name": "Write"' in serialized
    assert "AzureAiPlatform_Teams_ItsNagnt.kt" in serialized
    assert '"stop_reason": "tool_use"' in serialized


def test_nonstreaming_lifts_inline_kimi_bare_args_tool_call():
    """Non-streaming path lifts the bare-args variant and strips the whole
    section (id + args included) from visible text."""
    request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=64,
        messages=[ClaudeMessage(role="user", content="write it")],
        tools=[_BASH_TOOL, _READ_TOOL, _WRITE_TOOL],
    )
    text = (
        " Writing the file now. "
        + _SECTION_BEGIN + " " + _CALL_BEGIN + " chatcmpl-tool-abc123   "
        + '{"file_path": "/tmp/a.kt", "content": "hi"} '
        + _CALL_END + " " + _SECTION_END
    )
    openai_response = {
        "id": "resp_bare_n1",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": text, "tool_calls": []},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    resp = convert_openai_to_claude_response(openai_response, request)

    assert resp["stop_reason"] == "tool_use"
    tool_uses = [b for b in resp["content"] if b.get("type") == "tool_use"]
    assert tool_uses[0]["name"] == "Write"
    assert tool_uses[0]["input"]["file_path"] == "/tmp/a.kt"
    assert _SECTION_BEGIN not in json.dumps(resp["content"])
    texts = [b for b in resp["content"] if b.get("type") == "text"]
    assert any("Writing the file now" in b["text"] for b in texts)
    # The bare id and args JSON must not survive as visible text either.
    assert not any("chatcmpl-tool-" in b["text"] for b in texts)
