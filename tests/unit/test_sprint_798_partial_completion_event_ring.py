"""Sprint 798 — partial-completion event ring (records slash signals).

Sprints 784/785 wired the streaming + unary settle paths to
emit a `logger.warning` line when `decision.should_slash=True`
(operator-attributable error like OOM/crash). But operators
have no persistent record — debugging "did I get slashed
today?" requires journalctl spelunking, and the signal is
operator-visible only if they happen to be tailing logs.

Sprint 798 ships persistence:

  prsm/node/partial_completion_event_log.py:
    PartialCompletionEntry dataclass
    PartialCompletionEventRing — bounded in-memory ring,
      optional persist_dir (mirrors SlashEventRing pattern).

  Integration: sprints 784/785 settle paths call
  `node._partial_completion_event_log.append(...)` when
  `decision.should_slash=True`. Existing log.warning preserved.

The ring is separate from sprint-262's SlashEventRing because:
- SlashEventRing tracks ON-CHAIN slash events (slash_id from
  contract). PartialCompletionEventRing tracks OFF-CHAIN
  operator-attributable settlement decisions. Different
  granularity, different operator-action semantics.
- Unifying would force a Slash-or-PartialCompletion union type
  and lose schema clarity.

Endpoint (/admin/partial-completion-events) + CLI (`prsm node
partial-completion-history`) are sprint 799's scope — sprint
798 ships the foundation.

Pin tests:
- Entry dataclass exists with correct shape.
- Ring exists + bounded by max_entries.
- append + recent round-trip.
- recent ordered newest-first.
- recent honors limit / offset.
- Optional provider filter.
- Source-shape: _settle_streaming_escrow appends on slash.
- Source-shape: unary settle block appends on slash.
"""
from __future__ import annotations

import inspect
from datetime import datetime, timezone


# ---- Entry + Ring shape ----------------------------------------


def test_entry_dataclass_exists():
    from prsm.node.partial_completion_event_log import (
        PartialCompletionEntry,
    )
    entry = PartialCompletionEntry(
        timestamp=1700000000.0,
        job_id="j1",
        operator_node_id="a" * 32,
        reason="error",
        tokens_completed=4,
        tokens_requested=10,
    )
    assert entry.reason == "error"
    assert entry.tokens_completed == 4


def test_ring_exists_with_max_entries():
    from prsm.node.partial_completion_event_log import (
        PartialCompletionEventRing,
    )
    ring = PartialCompletionEventRing(max_entries=8)
    assert ring is not None


def test_ring_bounded_by_max_entries():
    """Append > max → oldest dropped (deque semantics)."""
    from prsm.node.partial_completion_event_log import (
        PartialCompletionEventRing,
    )
    ring = PartialCompletionEventRing(max_entries=3)
    for i in range(5):
        ring.append(
            job_id=f"j{i}",
            operator_node_id="a" * 32,
            reason="error",
            tokens_completed=i,
            tokens_requested=10,
        )
    entries = ring.recent(limit=100)
    assert len(entries) == 3
    # Newest 3 retained → j2, j3, j4. Newest-first ordering.
    job_ids = [e["job_id"] for e in entries]
    assert job_ids == ["j4", "j3", "j2"]


def test_recent_default_ordering_newest_first():
    from prsm.node.partial_completion_event_log import (
        PartialCompletionEventRing,
    )
    ring = PartialCompletionEventRing(max_entries=10)
    ring.append(
        job_id="old", operator_node_id="a" * 32,
        reason="error", tokens_completed=1, tokens_requested=10,
        timestamp=1000.0,
    )
    ring.append(
        job_id="new", operator_node_id="a" * 32,
        reason="error", tokens_completed=2, tokens_requested=10,
        timestamp=2000.0,
    )
    entries = ring.recent(limit=10)
    assert entries[0]["job_id"] == "new"
    assert entries[1]["job_id"] == "old"


def test_recent_limit_and_offset():
    from prsm.node.partial_completion_event_log import (
        PartialCompletionEventRing,
    )
    ring = PartialCompletionEventRing(max_entries=10)
    for i in range(6):
        ring.append(
            job_id=f"j{i}", operator_node_id="a" * 32,
            reason="error", tokens_completed=i, tokens_requested=10,
            timestamp=1000.0 + i,
        )
    # 6 entries total, newest-first j5..j0
    page1 = ring.recent(limit=2, offset=0)
    assert [e["job_id"] for e in page1] == ["j5", "j4"]
    page2 = ring.recent(limit=2, offset=2)
    assert [e["job_id"] for e in page2] == ["j3", "j2"]


def test_recent_filter_by_operator_node_id():
    from prsm.node.partial_completion_event_log import (
        PartialCompletionEventRing,
    )
    ring = PartialCompletionEventRing(max_entries=10)
    ring.append(
        job_id="a-job", operator_node_id="a" * 32,
        reason="error", tokens_completed=1, tokens_requested=10,
    )
    ring.append(
        job_id="b-job", operator_node_id="b" * 32,
        reason="error", tokens_completed=2, tokens_requested=10,
    )
    a_only = ring.recent(
        limit=10, operator_node_id="a" * 32,
    )
    assert len(a_only) == 1
    assert a_only[0]["job_id"] == "a-job"


def test_recent_entry_shape():
    """Each returned entry is a dict with operator-readable
    keys (no bytes, no Decimal — JSON-serializable)."""
    from prsm.node.partial_completion_event_log import (
        PartialCompletionEventRing,
    )
    ring = PartialCompletionEventRing(max_entries=10)
    ring.append(
        job_id="j1", operator_node_id="a" * 32,
        reason="error", tokens_completed=4, tokens_requested=10,
        timestamp=12345.6,
    )
    e = ring.recent(limit=1)[0]
    assert e["timestamp"] == 12345.6
    assert e["job_id"] == "j1"
    assert e["operator_node_id"] == "a" * 32
    assert e["reason"] == "error"
    assert e["tokens_completed"] == 4
    assert e["tokens_requested"] == 10


# ---- Integration with settle paths (source-shape) --------------


def test_streaming_settle_appends_on_slash():
    """Source-shape: _settle_streaming_escrow contains a call
    to a partial_completion_event_log.append (or similar) on
    the should_slash branch."""
    from prsm.node import api as _api
    src = inspect.getsource(_api._settle_streaming_escrow)
    assert "_partial_completion_event_log" in src, (
        "Sprint 798: _settle_streaming_escrow must record "
        "partial-completion slash events to "
        "node._partial_completion_event_log"
    )


def test_unary_settle_appends_on_slash():
    """Source-shape: the /compute/inference handler contains a
    call to a partial_completion_event_log.append on the
    should_slash branch."""
    from prsm.node import api as _api
    src = inspect.getsource(_api)
    # The unary handler is a closure; check the api module source
    # contains the append site twice (streaming + unary).
    count = src.count("_partial_completion_event_log")
    # 2 settle paths each append; the api module may also init
    # the ring itself. Require >= 2 to cover both settle sites.
    assert count >= 2, (
        f"Sprint 798: expected >= 2 references to "
        f"_partial_completion_event_log in api.py; found {count}"
    )
