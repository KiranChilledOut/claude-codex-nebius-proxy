"""Server-side execution of web-search tools (Tavily-backed).

Claude Code's ``WebSearch`` (and the Anthropic ``web_search`` server tool) cannot
run behind a non-Anthropic backend — there is no search engine on the other side,
so they resolve to "0 searches". When ``TAVILY_API_KEY`` is configured, the proxy
executes the search itself and feeds the results back to the model in a bounded
loop, returning the final answer. The web tools therefore run invisibly,
mirroring how Anthropic's server tools behave.

When ``TAVILY_API_KEY`` is unset (or ``SERVER_SEARCH_ENABLED=false``) this module
is inert and the proxy behaves exactly as before.
"""

import json
from typing import Any, AsyncGenerator, Dict, List

import httpx

from src.core.config import config
from src.core.logging import logger

# Tool names (case-insensitive) the proxy executes itself.
_SEARCH_NAMES = {"web_search", "websearch"}

# Schema we force on search tools when forwarding, so the backend model emits a
# clean {"query": ...} call instead of a no-arg / control-token blob.
SEARCH_TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The search query."}
    },
    "required": ["query"],
}


SEARCH_TOOL_SYSTEM_SUPPLEMENT = (
    "Web search note: for the user's current question, if it requires current "
    "information (events after your training cutoff or real-time data), call the "
    "web search tool (web_search / WebSearch) by ITSELF in its own turn — do not "
    "batch it together with other tool calls in the same response. The proxy "
    "executes the search and returns results before you continue. (Batching with "
    "other tools prevents execution.)"
)


def is_search_tool(tool: Any) -> bool:
    """True if a tool definition is a web-search tool (by name or server type)."""
    name = (getattr(tool, "name", "") or "").lower()
    ttype = (getattr(tool, "type", "") or "").lower()
    return name in _SEARCH_NAMES or ttype.startswith("web_search")


def is_search_tool_name(name: str) -> bool:
    return (name or "").lower() in _SEARCH_NAMES


def request_has_search_tool(claude_request) -> bool:
    """True if search execution is enabled and the request offers a search tool."""
    if not config.tavily_api_key or not config.server_search_enabled:
        return False
    for tool in getattr(claude_request, "tools", None) or []:
        if is_search_tool(tool):
            return True
    return False


def _tc_name(tc: Dict[str, Any]) -> str:
    return (tc.get("function") or {}).get("name") or ""


async def tavily_search(query: str) -> str:
    """Run a Tavily search and return a compact, model-readable result string."""
    if not query:
        return "No search query was provided."
    if not config.tavily_api_key:
        return "Web search is not configured on this proxy."

    payload = {
        "query": query,
        "max_results": config.tavily_max_results,
        "include_answer": True,
        "search_depth": "basic",
    }
    headers = {"Authorization": f"Bearer {config.tavily_api_key}"}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                "https://api.tavily.com/search", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:  # network / auth / parse — never raise into the loop
        logger.warning(f"[server_tools] Tavily search failed: {exc}")
        return f"Web search error: {exc}"

    lines: List[str] = []
    if data.get("answer"):
        lines.append(f"Answer: {data['answer']}")
    for item in (data.get("results") or [])[: config.tavily_max_results]:
        title = item.get("title", "")
        url = item.get("url", "")
        content = item.get("content", "")
        lines.append(f"- {title} ({url})\n  {content}")
    return "\n".join(lines) if lines else "No results found."


async def run_search_loop(openai_request: Dict[str, Any], openai_client, request_id):
    """Run the backend, executing owned search tool calls server-side, until the
    model produces an answer (or a non-search tool call the client must handle).

    Returns a final OpenAI response dict. Any pending tool calls in it are NOT
    owned search tools, so the caller's normal conversion/passthrough applies.
    """
    # Imported lazily to keep module import order simple (response_converter
    # imports request_converter which imports this module's siblings).
    from src.conversion.response_converter import _finalize_tool_args

    req = dict(openai_request)
    req["stream"] = False
    # `stream_options` is only valid for streaming requests; strip it to avoid
    # 400 Bad Request from backends that enforce structural validity.
    req.pop("stream_options", None)
    messages = list(req.get("messages", []))
    response: Dict[str, Any] = {}

    for _ in range(max(1, config.server_search_max_iters)):
        req["messages"] = messages
        response = await openai_client.create_chat_completion(req, request_id)

        choice = (response.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return response

        search_calls = [tc for tc in tool_calls if is_search_tool_name(_tc_name(tc))]
        # Nothing to execute, or a mix with client tools we cannot run here:
        # hand the whole response back for normal handling.
        if not search_calls or len(search_calls) != len(tool_calls):
            return response

        # Record the assistant turn, then execute each search and append results.
        messages.append(
            {
                "role": "assistant",
                "content": msg.get("content"),
                "tool_calls": tool_calls,
            }
        )
        for tc in search_calls:
            raw_args = (tc.get("function") or {}).get("arguments") or "{}"
            _, _, parsed = _finalize_tool_args(_tc_name(tc), raw_args)
            parsed = parsed or {}
            query = parsed.get("query") or parsed.get("q") or ""
            result = await tavily_search(query)
            logger.info(f"[server_tools] executed web_search query={query!r}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "content": result,
                }
            )

    # Max iterations reached while still searching: force a final answer by
    # dropping the search tools so the model must respond with text.
    req["messages"] = messages
    req.pop("tools", None)
    req.pop("tool_choice", None)
    response = await openai_client.create_chat_completion(req, request_id)
    return response


async def run_search_loop_streaming(
    openai_request: Dict[str, Any], openai_client, request_id
) -> AsyncGenerator[str, None]:
    """Streaming variant of :func:`run_search_loop`.

    Yields OpenAI chat-completion SSE strings (``data: {chunk}`` ... ``data: [DONE]``)
    in the exact shape :func:`convert_openai_sse_to_responses_sse` consumes, so the
    proxy can keep streaming tokens to the client while still executing server-owned
    search tools.

    Content/reasoning deltas are forwarded the instant they arrive — the common
    no-search turn streams live with no buffering, which is what restores Codex's
    token-by-token output. Tool-call and finish chunks are withheld until the turn
    completes; only then can we apply :func:`run_search_loop`'s contract (a turn whose
    tool calls are *all* server-owned searches is executed here and fed back, never
    forwarded). Any other turn is released verbatim. Intercepted search turns emit no
    ``[DONE]``, so the whole exchange reaches the client as one continuous response.
    """
    # Imported lazily to keep module import order simple (mirrors run_search_loop).
    from src.conversion.response_converter import _finalize_tool_args

    req = dict(openai_request)
    # create_chat_completion_stream sets stream/stream_options itself.
    req.pop("stream", None)
    req.pop("stream_options", None)
    messages = list(req.get("messages", []))
    max_iters = max(1, config.server_search_max_iters)

    for iteration in range(max_iters):
        req["messages"] = messages
        assembled: Dict[int, Dict[str, Any]] = {}
        withheld: List[str] = []  # tool-call + finish/usage chunks, pending decision

        async for sse in openai_client.create_chat_completion_stream(req, request_id):
            stripped = sse.strip()
            if stripped == "data: [DONE]":
                break
            payload = stripped[len("data:"):].strip() if stripped.startswith("data:") else ""
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                yield sse  # unparseable: forward verbatim
                continue

            choices = chunk.get("choices") or []
            delta = (choices[0].get("delta") or {}) if choices else {}
            finish = choices[0].get("finish_reason") if choices else None

            tcs = delta.get("tool_calls")
            if tcs:
                for tcd in tcs:
                    idx = tcd.get("index", 0)
                    slot = assembled.setdefault(idx, {"id": None, "name": "", "arguments": ""})
                    if tcd.get("id"):
                        slot["id"] = tcd["id"]
                    fn = tcd.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
                withheld.append(sse)
                continue

            if finish is not None or not choices:
                # finish marker or usage-only chunk: withhold until we decide.
                withheld.append(sse)
                continue

            # Plain content / reasoning / role delta: stream live.
            yield sse

        tool_calls = [assembled[i] for i in sorted(assembled)]
        search_calls = [t for t in tool_calls if is_search_tool_name(t["name"])]
        pure_search = bool(tool_calls) and len(search_calls) == len(tool_calls)

        if pure_search and iteration < max_iters - 1:
            # Execute the searches and continue; the withheld search-call chunks are
            # intentionally dropped so they never reach the client.
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": t["id"],
                            "type": "function",
                            "function": {"name": t["name"], "arguments": t["arguments"]},
                        }
                        for t in tool_calls
                    ],
                }
            )
            for t in search_calls:
                _, _, parsed = _finalize_tool_args(t["name"], t["arguments"] or "{}")
                parsed = parsed or {}
                query = parsed.get("query") or parsed.get("q") or ""
                result = await tavily_search(query)
                logger.info(f"[server_tools] executed web_search query={query!r}")
                messages.append(
                    {"role": "tool", "tool_call_id": t["id"], "content": result}
                )
            continue

        # Passthrough / final turn: release withheld chunks, then terminate.
        for s in withheld:
            yield s
        yield "data: [DONE]"
        return

    # Max iterations reached while still searching: drop the search tools and stream
    # a forced final answer verbatim (its own [DONE] terminates the stream).
    req["messages"] = messages
    req.pop("tools", None)
    req.pop("tool_choice", None)
    async for sse in openai_client.create_chat_completion_stream(req, request_id):
        yield sse
