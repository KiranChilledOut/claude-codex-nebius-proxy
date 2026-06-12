"""Pending-approval registry for ensemble approval mode.

While Claude Code's SSE stream is held open (with pings), the race result
sits here so the dashboard can list it and submit the user's choice. The
holding generator waits on the per-request event; on timeout the scored
auto-winner stands.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from src.ensemble.engine import EnsembleRace

LIST_PREVIEW_CHARS = 1200


class PendingApproval:
    def __init__(
        self,
        request_id: str,
        race: EnsembleRace,
        session_id: Optional[str],
        session_name: Optional[str],
    ):
        self.request_id = request_id
        self.race = race
        self.session_id = session_id
        self.session_name = session_name
        self.created_at = time.time()
        self.choice: Optional[int] = None
        self.event = asyncio.Event()


class ApprovalStore:
    def __init__(self):
        self._pending: Dict[str, PendingApproval] = {}

    def register(
        self,
        request_id: str,
        race: EnsembleRace,
        session_id: Optional[str],
        session_name: Optional[str],
    ) -> PendingApproval:
        pending = PendingApproval(request_id, race, session_id, session_name)
        self._pending[request_id] = pending
        return pending

    def choose(self, request_id: str, candidate_index: int) -> bool:
        pending = self._pending.get(request_id)
        if pending is None:
            return False
        candidates = pending.race.candidates
        if not (0 <= candidate_index < len(candidates)):
            return False
        if candidates[candidate_index].status == "error":
            return False
        pending.choice = candidate_index
        pending.event.set()
        return True

    def remove(self, request_id: str) -> None:
        self._pending.pop(request_id, None)

    def list_pending(self) -> List[Dict[str, Any]]:
        items = []
        for pending in sorted(self._pending.values(), key=lambda p: p.created_at):
            items.append(
                {
                    "request_id": pending.request_id,
                    "session_id": pending.session_id,
                    "session_name": pending.session_name,
                    "created_at": pending.created_at,
                    "waiting_seconds": round(time.time() - pending.created_at, 1),
                    "candidates": [
                        {
                            "index": c.index,
                            "model": c.model,
                            "status": c.status,
                            "score": None if c.score == float("-inf") else c.score,
                            "latency_ms": round(c.latency_ms),
                            "reasons": c.reasons,
                            "finish_reason": c.finish_reason,
                            "output_text": c.output_text[:LIST_PREVIEW_CHARS],
                            "tool_calls": c.tool_calls,
                            "error": c.error,
                            "auto_winner": c.index == pending.race.winner_index,
                        }
                        for c in pending.race.candidates
                    ],
                }
            )
        return items


approval_store = ApprovalStore()
