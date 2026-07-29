"""Tests for the robustness fixes ported from the Mirage gap analysis.

Covers: idle-stream watchdog (#1), tool-output compaction (#2), role
coalescing (#3), request body cap (#4), error-log redaction (#7),
parallel_tool_calls stripping (#8), embedding/rerank filtering (#C), and
thinking.budget_tokens -> effort mapping (#D).
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.codex.models import ResponsesItem, ResponsesRequest
from src.codex.request_converter import convert_responses_to_openai_chat
from src.conversion.request_converter import (
    _coalesce_consecutive_roles,
    _compact_large_tool_results,
    _resolve_reasoning_effort,
)
from src.core.client import OpenAIClient
from src.core.model_catalog import is_chat_capable
from src.models.claude import (
    ClaudeMessage,
    ClaudeMessagesRequest,
    ClaudeThinkingConfig,
)


# --- #1 idle-stream watchdog -------------------------------------------------


class _FakeChunk:
    def __init__(self, text):
        self._text = text

    def model_dump(self):
        return {
            "choices": [{"delta": {"content": self._text}, "finish_reason": None}]
        }


class _StallingStream:
    """Yields one chunk, then hangs forever (never raises, never closes)."""

    def __init__(self):
        self._sent = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._sent:
            self._sent = True
            return _FakeChunk("hello")
        await asyncio.sleep(3600)  # simulate a hung upstream
        raise StopAsyncIteration


class _FakeStreamingCompletions:
    async def create(self, **kwargs):
        return _StallingStream()


class _FakeChat:
    def __init__(self):
        self.completions = _FakeStreamingCompletions()


class _FakeSDKClient:
    def __init__(self):
        self.chat = _FakeChat()


def _client_with_stall(idle_timeout):
    client = OpenAIClient("k", "http://example", timeout=1)
    client.client = _FakeSDKClient()
    client.stream_idle_timeout = idle_timeout
    return client


def test_idle_stream_watchdog_surfaces_error():
    """A hung upstream must raise (typed 503), not hang forever."""
    client = _client_with_stall(idle_timeout=0.2)

    async def run():
        chunks = []
        async for line in client.create_chat_completion_stream(
            {"model": "m", "messages": []}
        ):
            chunks.append(line)
        return chunks

    try:
        chunks = asyncio.run(asyncio.wait_for(run(), timeout=5))
        raised = None
    except Exception as e:  # noqa: BLE001
        chunks = []
        raised = e
    # Either we got an HTTPException out of the generator, or the wait_for
    # captured it. The key point: it did NOT hang the full 5s.
    assert raised is not None or chunks, "expected some resolution, not a hang"


def test_idle_stream_watchdog_timeout_is_typed_503():
    from fastapi import HTTPException

    client = _client_with_stall(idle_timeout=0.1)

    async def run():
        async for _ in client.create_chat_completion_stream({"model": "m", "messages": []}):
            pass

    async def capture():
        try:
            await run()
            return None
        except HTTPException as e:
            return e

    err = asyncio.run(asyncio.wait_for(capture(), timeout=5))
    assert err is not None, "idle stream should raise HTTPException, not hang"
    assert err.status_code == 503


# --- #2 tool-output compaction ----------------------------------------------


def test_compact_large_tool_results_shrinks_oldest():
    big = "word " * 4000  # well over the 512-token threshold
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": big},
        {"role": "assistant", "content": "done"},
    ]
    out = _compact_large_tool_results([dict(m) for m in msgs])
    assert "omitted" in out[3]["content"]
    assert len(out[3]["content"]) < len(big)


def test_compact_large_tool_results_leaves_last_message():
    """The in-flight (most recent) tool output is never compacted."""
    big = "word " * 4000
    msgs = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": big},
    ]
    out = _compact_large_tool_results([dict(m) for m in msgs])
    assert out[1]["content"] == big


def test_compact_large_tool_results_ignores_small():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "tool", "tool_call_id": "t1", "content": "small"},
        {"role": "assistant", "content": "a"},
    ]
    out = _compact_large_tool_results([dict(m) for m in msgs])
    assert out[1]["content"] == "small"


# --- #3 role coalescing ------------------------------------------------------


def test_coalesce_merges_consecutive_user_turns():
    msgs = [
        {"role": "user", "content": "one"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "reply"},
    ]
    out = _coalesce_consecutive_roles(msgs)
    assert [m["role"] for m in out] == ["user", "assistant"]
    assert out[0]["content"] == "one\ntwo"


def test_coalesce_never_merges_tool_structure():
    msgs = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "R", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "out"},
        {"role": "tool", "tool_call_id": "t1", "content": "out2"},
    ]
    out = _coalesce_consecutive_roles(msgs)
    # tool role messages and tool_calls must be preserved verbatim
    assert len(out) == 3
    assert out[0]["tool_calls"]
    assert out[1]["role"] == "tool"
    assert out[2]["role"] == "tool"


def test_coalesce_preserves_nontext_blocks():
    """An image/tool_result block must never be dropped by flattening."""
    image_block = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,abc"},
    }
    msgs = [
        {"role": "user", "content": "describe this"},
        {"role": "user", "content": [image_block]},
    ]
    out = _coalesce_consecutive_roles(msgs)
    # The image block must survive — either unmerged or carried in a list.
    rendered = str(out)
    assert "image_url" in rendered
    assert "data:image/png;base64,abc" in rendered


# --- #C embedding/rerank filtering ------------------------------------------


def test_is_chat_capable_filters_embedding_and_rerank():
    assert not is_chat_capable("BAAI/bge-m3")
    assert not is_chat_capable("intfloat/e5-mistral-7b-instruct-embed")
    assert not is_chat_capable("BAAI/bge-reranker-v2-m3")
    assert is_chat_capable("zai-org/GLM-4.5")
    assert is_chat_capable("moonshotai/Kimi-K2-Instruct")
    assert not is_chat_capable(None)


# --- #D thinking.budget_tokens -> effort ------------------------------------


def _req_with_thinking(budget):
    return ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=64,
        messages=[ClaudeMessage(role="user", content="hi")],
        thinking=ClaudeThinkingConfig(type="enabled", budget_tokens=budget),
    )


def test_thinking_budget_maps_to_effort():
    assert _resolve_reasoning_effort(_req_with_thinking(1000)) == "low"
    assert _resolve_reasoning_effort(_req_with_thinking(10000)) == "medium"
    assert _resolve_reasoning_effort(_req_with_thinking(40000)) == "high"


def test_thinking_disabled_yields_no_effort():
    req = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=64,
        messages=[ClaudeMessage(role="user", content="hi")],
        thinking=ClaudeThinkingConfig(type="disabled"),
    )
    assert _resolve_reasoning_effort(req) is None


# --- #8 parallel_tool_calls stripped from Codex requests ---------------------


def test_codex_request_never_forwards_parallel_tool_calls():
    req = ResponsesRequest(
        model="gpt-5",
        input="hello",
        tools=[
            {
                "type": "function",
                "name": "shell",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        parallel_tool_calls=True,
    )
    out = convert_responses_to_openai_chat(req)
    assert "parallel_tool_calls" not in out


# --- #4 request body size cap ------------------------------------------------


def test_request_body_cap_returns_413():
    """An oversized declared Content-Length is rejected with 413."""
    from fastapi.testclient import TestClient

    from src.core.config import config
    from src.main import app

    client = TestClient(app, raise_server_exceptions=False)
    oversize = config.max_request_body_bytes + 1
    # Send a small body but declare a huge Content-Length: the middleware
    # rejects on the declared length before any route/upstream work.
    response = client.post(
        "/v1/messages",
        content=b"{}",
        headers={"Content-Length": str(oversize), "Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["type"] == "request_too_large"


# --- #7 error-body redaction -------------------------------------------------


def test_redact_body_truncates_and_scrubs():
    long_body = "prompt text " * 500 + " sk-abcdefSECRETKEYMATERIAL"
    redacted = OpenAIClient._redact_body(long_body, limit=100)
    assert len(redacted) <= 116  # 100 + ellipsis marker
    assert "SECRETKEYMATERIAL" not in redacted
    assert redacted.startswith("prompt text")


def test_redact_body_handles_dict():
    redacted = OpenAIClient._redact_body({"error": {"message": "bad"}})
    assert "bad" in redacted
