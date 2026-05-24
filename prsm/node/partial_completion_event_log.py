"""Sprint 798 — operator-attributable partial-completion event log.

Sprints 784/785 routed partial_completion through the settle
paths; sprint 783's SettlementDecision surfaces `should_slash`
when the receipt's marker indicates an operator-attributable
error (currently `reason="error"`; future kinds covered by the
same flag). Pre-798, the only signal was a `logger.warning` —
operators had no persistent record + no audit surface.

Sprint 798 ships a bounded ring:

  PartialCompletionEntry — timestamp + job_id + operator_node_id
    + reason + tokens_completed + tokens_requested
  PartialCompletionEventRing — deque(maxlen=...) recent + filter
    by operator_node_id. Sibling of sprint-262's SlashEventRing
    (kept separate because on-chain slashes have different
    granularity + operator-action semantics).

Sprint 784/785 settle paths call `ring.append(...)` on slash
signals; the existing `logger.warning` is preserved as a
operator-debug breadcrumb.

Sprint 799 will add:
- /admin/partial-completion-events endpoint
- `prsm node partial-completion-history` CLI

Sprint 798 ships the foundation + the settle-path integration.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional


_DEFAULT_MAX_ENTRIES = 256


@dataclass(frozen=True)
class PartialCompletionEntry:
    """One operator-attributable partial-completion event.

    All fields JSON-serializable (no bytes, no Decimal) so a
    daemon endpoint can serve recent() output verbatim."""

    timestamp: float
    job_id: str
    operator_node_id: str
    reason: str
    tokens_completed: int
    tokens_requested: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "job_id": self.job_id,
            "operator_node_id": self.operator_node_id,
            "reason": self.reason,
            "tokens_completed": self.tokens_completed,
            "tokens_requested": self.tokens_requested,
        }


class PartialCompletionEventRing:
    """Bounded in-memory ring of partial-completion slash signals.

    Mirrors SlashEventRing's shape so /admin endpoints can
    follow the same pagination convention. Persistence is
    deferred to sprint 799 (or a follow-on if operators need
    cross-restart durability for the partial-completion stream
    specifically)."""

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        if not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError(
                f"max_entries must be a positive integer, "
                f"got {max_entries!r}"
            )
        self._entries: Deque[PartialCompletionEntry] = deque(
            maxlen=max_entries,
        )

    def append(
        self,
        *,
        job_id: str,
        operator_node_id: str,
        reason: str,
        tokens_completed: int,
        tokens_requested: int,
        timestamp: Optional[float] = None,
    ) -> None:
        entry = PartialCompletionEntry(
            timestamp=(
                timestamp if timestamp is not None else time.time()
            ),
            job_id=job_id,
            operator_node_id=operator_node_id,
            reason=reason,
            tokens_completed=int(tokens_completed),
            tokens_requested=int(tokens_requested),
        )
        self._entries.append(entry)

    def recent(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        operator_node_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return entries newest-first, optionally filtered by
        operator_node_id."""
        # Newest-first: reverse the deque (which is oldest-first
        # by insertion).
        items = list(reversed(self._entries))
        if operator_node_id is not None:
            items = [
                e for e in items
                if e.operator_node_id == operator_node_id
            ]
        return [e.to_dict() for e in items[offset:offset + limit]]

    def count(self) -> int:
        return len(self._entries)
