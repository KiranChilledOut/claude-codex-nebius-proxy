"""Ensemble hedge engine: race one request across N backend models.

Each candidate is a parallel NON-streaming upstream call so every full output
is available for scoring, the dashboard split view, and approval mode. The
winner is re-emitted to the client as ordinary Anthropic SSE via
claude_response_to_sse, so Claude Code sees one normal response.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from fastapi import HTTPException

logger = logging.getLogger(__name__)

_JUDGE_VERDICT_RE = re.compile(r"\{[^{}]*\"winner\"[^{}]*\}")

# Stored output is for the dashboard split view, not an archive.
PREVIEW_CHARS = 4000


@dataclass
class EnsembleCandidate:
    index: int
    model: str
    status: str = "pending"  # won | lost | error
    chosen_by: Optional[str] = None  # auto | user | timeout — winner only
    score: float = 0.0
    latency_ms: float = 0.0
    reasons: List[str] = field(default_factory=list)
    finish_reason: Optional[str] = None
    output_text: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class EnsembleRace:
    request_id: str
    mode: str
    candidates: List[EnsembleCandidate]
    winner_index: int = -1

    @property
    def winner(self) -> EnsembleCandidate:
        return self.candidates[self.winner_index]

    def set_winner(self, index: int, chosen_by: str) -> None:
        self.winner_index = index
        for cand in self.candidates:
            if cand.status != "error":
                cand.status = "lost"
                cand.chosen_by = None
        self.candidates[index].status = "won"
        self.candidates[index].chosen_by = chosen_by


def _extract_output(cand: EnsembleCandidate, response: Dict[str, Any]) -> None:
    choices = response.get("choices") or []
    if not choices:
        cand.reasons.append("no choices in response")
        return
    choice = choices[0]
    message = choice.get("message") or {}
    cand.finish_reason = choice.get("finish_reason")
    cand.output_text = (message.get("content") or "")[:PREVIEW_CHARS]
    cand.usage = response.get("usage") or {}
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        cand.tool_calls.append(
            {"name": fn.get("name"), "arguments": (fn.get("arguments") or "")[:PREVIEW_CHARS]}
        )


def _score_candidate(cand: EnsembleCandidate, offered_tools: Set[str]) -> None:
    """Rule-based verdict. Higher is better; reasons are shown on the dashboard."""
    score = 0.0
    message = ((cand.response or {}).get("choices") or [{}])[0].get("message") or {}
    text = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []

    if cand.finish_reason == "length":
        score -= 2
        cand.reasons.append("hit max_tokens (truncated output)")
    if not text.strip() and not tool_calls:
        score -= 5
        cand.reasons.append("empty response")

    valid_calls = 0
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        args = fn.get("arguments") or ""
        try:
            json.loads(args or "{}")
            score += 2
            valid_calls += 1
        except (json.JSONDecodeError, TypeError):
            score -= 3
            cand.reasons.append(f"malformed JSON arguments for {name}")
        if offered_tools and name not in offered_tools:
            score -= 2
            cand.reasons.append(f"called unknown tool '{name}'")

    if valid_calls:
        cand.reasons.append(f"{valid_calls} clean tool call(s)")
    elif text.strip():
        cand.reasons.append("text response")

    cand.score = score


def _last_user_text(openai_request: Dict[str, Any]) -> str:
    for message in reversed(openai_request.get("messages") or []):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content[:2000]
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                    return part["text"][:2000]
    return ""


async def _judge_tiebreak(
    openai_client,
    judge_model: str,
    request_id: str,
    openai_request: Dict[str, Any],
    tied: List["EnsembleCandidate"],
) -> Optional[EnsembleCandidate]:
    """Ask ENSEMBLE_JUDGE_MODEL to pick among rule-score-tied candidates.
    Returns the chosen candidate, or None on any failure (rules then decide).
    Runs on the same Token Factory key — no extra credentials needed."""
    goal = _last_user_text(openai_request)
    sections = []
    for cand in tied:
        calls = "; ".join(
            f"{tc.get('name')}({tc.get('arguments')})" for tc in cand.tool_calls
        )
        sections.append(
            f"### Candidate {cand.index} ({cand.model})\n"
            f"Tool calls: {calls or 'none'}\n"
            f"Text:\n{cand.output_text or '(empty)'}"
        )
    prompt = (
        "Two AI assistants answered the same request. Pick the response that "
        "best advances the user's goal: correct, complete, follows the request, "
        "and (if tools are called) calls the right tool with the right arguments.\n\n"
        f"## User's request\n{goal or '(not available)'}\n\n"
        + "\n\n".join(sections)
        + '\n\nReply with ONLY a JSON object: {"winner": <candidate number>, "reason": "<one short sentence>"}'
    )
    judge_request = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": "You are a strict, terse response judge."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2000,
        "temperature": 0,
    }
    started = time.monotonic()
    try:
        response = await openai_client.create_chat_completion(
            judge_request, request_id=f"{request_id}:judge"
        )
        elapsed = time.monotonic() - started
        message = ((response.get("choices") or [{}])[0].get("message")) or {}
        # Reasoning models may leave content empty and put text (including the
        # verdict) in reasoning_content — search both.
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        match = _JUDGE_VERDICT_RE.search(content) or _JUDGE_VERDICT_RE.search(reasoning)
        if not match:
            logger.warning(
                "[ENSEMBLE] judge %s returned no verdict in %.1fs (content=%r...)",
                judge_model,
                elapsed,
                content[:120],
            )
            return None
        verdict = json.loads(match.group(0))
        winner_index = int(verdict.get("winner"))
        chosen = next((c for c in tied if c.index == winner_index), None)
        if chosen is not None:
            reason = str(verdict.get("reason") or "").strip()[:300]
            chosen.reasons.append(
                f"judge ({judge_model}, {elapsed:.1f}s): {reason or 'preferred'}"
            )
            logger.info(
                "[ENSEMBLE] judge %s picked candidate %d (%s) in %.1fs: %s",
                judge_model,
                winner_index,
                chosen.model,
                elapsed,
                reason[:120],
            )
        return chosen
    except Exception as exc:
        logger.warning(
            "[ENSEMBLE] judge %s failed after %.1fs (%s); falling back to rules",
            judge_model,
            time.monotonic() - started,
            exc,
        )
        return None


async def run_hedge_race(
    openai_request: Dict[str, Any],
    openai_client,
    request_id: str,
    models: List[str],
    mode: str,
    runner: Optional[Callable[[Dict[str, Any], str], Awaitable[Dict[str, Any]]]] = None,
    judge_model: Optional[str] = None,
) -> EnsembleRace:
    """Run the same request against every candidate model in parallel and
    auto-pick a winner (score, judge tie-break, then latency). Raises 502 only
    if every candidate fails — a single healthy model keeps the session alive.

    `runner` overrides how a candidate request is executed (e.g. the
    server-side search loop); default is a plain chat completion.
    """
    offered_tools = {
        (t.get("function") or {}).get("name")
        for t in openai_request.get("tools") or []
        if t.get("function")
    }

    async def default_runner(candidate_request: Dict[str, Any], rid: str) -> Dict[str, Any]:
        return await openai_client.create_chat_completion(candidate_request, request_id=rid)

    execute = runner or default_runner

    async def run_one(index: int, model: str) -> EnsembleCandidate:
        cand = EnsembleCandidate(index=index, model=model)
        candidate_request = dict(openai_request)
        candidate_request["model"] = model
        candidate_request.pop("stream", None)
        started = time.monotonic()
        try:
            response = await execute(candidate_request, f"{request_id}:c{index}")
            cand.latency_ms = (time.monotonic() - started) * 1000
            cand.response = response
            _extract_output(cand, response)
            _score_candidate(cand, offered_tools)
        except HTTPException as exc:
            cand.latency_ms = (time.monotonic() - started) * 1000
            cand.status = "error"
            cand.score = float("-inf")
            cand.error = str(exc.detail)[:500]
            cand.reasons.append(f"upstream error (HTTP {exc.status_code})")
        except Exception as exc:
            cand.latency_ms = (time.monotonic() - started) * 1000
            cand.status = "error"
            cand.score = float("-inf")
            cand.error = str(exc)[:500]
            cand.reasons.append("upstream error")
        return cand

    candidates = list(
        await asyncio.gather(*(run_one(i, m) for i, m in enumerate(models)))
    )

    viable = [c for c in candidates if c.status != "error"]
    if not viable:
        detail = "; ".join(f"{c.model}: {c.error}" for c in candidates)
        raise HTTPException(status_code=502, detail=f"All ensemble candidates failed — {detail}")

    best = min(viable, key=lambda c: (-c.score, c.latency_ms))
    tied = [c for c in viable if c.score == best.score]
    if len(tied) >= 2 and judge_model:
        judged = await _judge_tiebreak(
            openai_client, judge_model, request_id, openai_request, tied
        )
        if judged is not None:
            best = judged
    if "judge" not in " ".join(best.reasons):
        runners_up = [c for c in viable if c.index != best.index]
        if not runners_up:
            best.reasons.append("decision: only viable candidate")
        elif len(tied) >= 2 and best in tied:
            slowest_gap = min(c.latency_ms for c in tied if c.index != best.index) - best.latency_ms
            best.reasons.append(f"decision: score tie — {max(slowest_gap, 0):.0f}ms faster")
        else:
            runner_up = max(runners_up, key=lambda c: c.score)
            best.reasons.append(
                f"decision: higher score ({best.score:.1f} vs {runner_up.score:.1f})"
            )
    race = EnsembleRace(request_id=request_id, mode=mode, candidates=candidates)
    race.set_winner(best.index, "auto")
    logger.info(
        "[ENSEMBLE] %s won (%s, score=%.1f, %.0fms) over %s",
        best.model,
        mode,
        best.score,
        best.latency_ms,
        ", ".join(f"{c.model}(score={c.score:.1f})" for c in candidates if c.index != best.index),
    )
    return race
