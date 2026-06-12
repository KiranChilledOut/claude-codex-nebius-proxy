"""Ensemble hedge engine: racing, scoring, and the approval store."""

import asyncio

import pytest
from fastapi import HTTPException

from src.ensemble.approval import ApprovalStore
from src.ensemble.engine import run_hedge_race


def _response(model, *, text="", tool_name=None, tool_args=None, finish="stop"):
    message = {"role": "assistant", "content": text}
    if tool_name is not None:
        message["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": tool_name, "arguments": tool_args},
            }
        ]
        finish = "tool_calls"
    return {
        "id": "chatcmpl-x",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


class FakeClient:
    """Returns canned responses (or raises) keyed by model name."""

    def __init__(self, outcomes):
        self.outcomes = outcomes

    async def create_chat_completion(self, request, request_id=None):
        outcome = self.outcomes[request["model"]]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


BASE_REQUEST = {
    "model": "placeholder",
    "messages": [{"role": "user", "content": "hi"}],
    "tools": [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }
    ],
}


def test_valid_tool_call_beats_malformed():
    client = FakeClient(
        {
            "model-a": _response("model-a", tool_name="get_weather", tool_args='{"city": "Par'),
            "model-b": _response("model-b", tool_name="get_weather", tool_args='{"city": "Paris"}'),
        }
    )
    race = asyncio.run(
        run_hedge_race(dict(BASE_REQUEST), client, "req-1", ["model-a", "model-b"], "hedge")
    )

    assert race.winner.model == "model-b"
    assert race.winner.status == "won"
    assert race.winner.chosen_by == "auto"
    loser = race.candidates[0]
    assert loser.status == "lost"
    assert any("malformed" in r for r in loser.reasons)


def test_unknown_tool_and_empty_response_are_penalized():
    client = FakeClient(
        {
            "model-a": _response("model-a", tool_name="not_a_tool", tool_args="{}"),
            "model-b": _response("model-b", text="Here is the answer."),
        }
    )
    race = asyncio.run(
        run_hedge_race(dict(BASE_REQUEST), client, "req-2", ["model-a", "model-b"], "hedge")
    )

    assert race.winner.model == "model-b"
    assert any("unknown tool" in r for r in race.candidates[0].reasons)


def test_single_error_does_not_kill_the_race():
    client = FakeClient(
        {
            "model-a": HTTPException(status_code=429, detail="rate limited"),
            "model-b": _response("model-b", text="ok"),
        }
    )
    race = asyncio.run(
        run_hedge_race(dict(BASE_REQUEST), client, "req-3", ["model-a", "model-b"], "hedge")
    )

    assert race.winner.model == "model-b"
    assert race.candidates[0].status == "error"
    assert "429" in race.candidates[0].reasons[0]


def test_all_errors_raise_502():
    client = FakeClient(
        {
            "model-a": HTTPException(status_code=500, detail="boom"),
            "model-b": HTTPException(status_code=429, detail="limited"),
        }
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            run_hedge_race(dict(BASE_REQUEST), client, "req-4", ["model-a", "model-b"], "hedge")
        )
    assert exc.value.status_code == 502


def test_approval_store_choose_flow():
    async def scenario():
        client = FakeClient(
            {
                "model-a": _response("model-a", text="answer A"),
                "model-b": _response("model-b", text="answer B"),
            }
        )
        race = await run_hedge_race(
            dict(BASE_REQUEST), client, "req-5", ["model-a", "model-b"], "approval"
        )
        store = ApprovalStore()
        pending = store.register("req-5", race, "sid", "sname")

        listed = store.list_pending()
        assert len(listed) == 1
        assert len(listed[0]["candidates"]) == 2
        assert listed[0]["candidates"][race.winner_index]["auto_winner"]

        # invalid choices rejected
        assert not store.choose("nope", 0)
        assert not store.choose("req-5", 9)
        # valid choice resolves the event
        assert store.choose("req-5", 1)
        assert pending.choice == 1
        assert pending.event.is_set()
        store.remove("req-5")
        assert store.list_pending() == []

    asyncio.run(scenario())


def test_custom_runner_is_used_instead_of_chat_completion():
    """Search-tool requests race via run_search_loop — engine must honor the runner."""
    calls = []

    async def runner(candidate_request, rid):
        calls.append((candidate_request["model"], rid))
        return _response(candidate_request["model"], text=f"via runner {candidate_request['model']}")

    class ExplodingClient:
        async def create_chat_completion(self, request, request_id=None):
            raise AssertionError("default client must not be called when runner is set")

    race = asyncio.run(
        run_hedge_race(
            dict(BASE_REQUEST), ExplodingClient(), "req-6", ["model-a", "model-b"], "hedge",
            runner=runner,
        )
    )
    assert sorted(m for m, _ in calls) == ["model-a", "model-b"]
    assert race.winner.status == "won"


def test_winner_carries_decision_reason():
    client = FakeClient(
        {
            "model-a": _response("model-a", tool_name="get_weather", tool_args='{"city": "Paris"}'),
            "model-b": _response("model-b", tool_name="get_weather", tool_args="{broken"),
        }
    )
    race = asyncio.run(
        run_hedge_race(dict(BASE_REQUEST), client, "req-7", ["model-a", "model-b"], "hedge")
    )
    assert race.winner.model == "model-a"
    assert any(r.startswith("decision: higher score") for r in race.winner.reasons)


def test_judge_breaks_score_ties():
    client = FakeClient(
        {
            "model-a": _response("model-a", text="Answer A"),
            "model-b": _response("model-b", text="Answer B"),
            "judge-x": _response(
                "judge-x", text='I pick {"winner": 1, "reason": "B is more precise"}'
            ),
        }
    )
    race = asyncio.run(
        run_hedge_race(
            dict(BASE_REQUEST), client, "req-8", ["model-a", "model-b"], "hedge",
            judge_model="judge-x",
        )
    )
    assert race.winner.model == "model-b"
    assert any(r.startswith("judge (judge-x") for r in race.winner.reasons)


def test_judge_failure_falls_back_to_rules():
    client = FakeClient(
        {
            "model-a": _response("model-a", text="Answer A"),
            "model-b": _response("model-b", text="Answer B"),
            "judge-x": HTTPException(status_code=500, detail="judge down"),
        }
    )
    race = asyncio.run(
        run_hedge_race(
            dict(BASE_REQUEST), client, "req-9", ["model-a", "model-b"], "hedge",
            judge_model="judge-x",
        )
    )
    # Rules decide (score tie -> faster candidate); race must still complete.
    assert race.winner.status == "won"
    assert any(r.startswith("decision: score tie") for r in race.winner.reasons)


def test_judge_verdict_parsed_from_reasoning_content():
    """Reasoning models may leave content empty and put the verdict in
    reasoning_content — the judge must find it there too."""
    judge_response = {
        "id": "chatcmpl-j",
        "model": "judge-r",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": 'Comparing both... final: {"winner": 0, "reason": "A is correct"}',
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 30},
    }
    client = FakeClient(
        {
            "model-a": _response("model-a", text="Answer A"),
            "model-b": _response("model-b", text="Answer B"),
            "judge-r": judge_response,
        }
    )
    race = asyncio.run(
        run_hedge_race(
            dict(BASE_REQUEST), client, "req-10", ["model-a", "model-b"], "hedge",
            judge_model="judge-r",
        )
    )
    assert race.winner.model == "model-a"
    assert any(r.startswith("judge (judge-r") for r in race.winner.reasons)
