"""B8 pass 3 — GET /compute/status/{job_id}.

Reads from ``node._payment_escrow.get_escrow(job_id)`` and returns
the escrow lifecycle for billing reconciliation + the
``prsm_agent_status`` MCP tool.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.payment_escrow import EscrowEntry, EscrowStatus


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


def _node_with_escrow(escrows_by_job_id: dict | None = None):
    """Build a minimal node stub with a payment_escrow that returns
    pre-seeded EscrowEntry objects from get_escrow(job_id).
    Job history defaults to None so these tests exercise the
    escrow-only code path (the legacy fallback)."""
    node = MagicMock()
    node.identity.node_id = "test-node"
    if escrows_by_job_id is None:
        escrows_by_job_id = {}

    escrow_mock = MagicMock()
    escrow_mock.get_escrow = lambda jid: escrows_by_job_id.get(jid)
    node._payment_escrow = escrow_mock
    # Explicit None so MagicMock auto-attribute-spawning doesn't
    # confuse the two-tier lookup logic.
    node._job_history = None
    return node


def _client(node):
    app = create_api_app(node, enable_security=False)
    return TestClient(app)


def _entry(
    *,
    job_id: str = "forge-abc123",
    amount: float = 5.0,
    status: EscrowStatus = EscrowStatus.PENDING,
    provider_winner: str | None = None,
    completed_at: float | None = None,
) -> EscrowEntry:
    return EscrowEntry(
        escrow_id=f"esc-{job_id}",
        job_id=job_id,
        requester_id="test-requester",
        amount=amount,
        status=status,
        provider_winner=provider_winner,
        tx_lock="tx-lock-abc",
        tx_release="tx-release-xyz" if completed_at else None,
        created_at=1_700_000_000.0,
        completed_at=completed_at,
        metadata={"source": "test"},
    )


# ──────────────────────────────────────────────────────────────────────
# Happy path — known job_id
# ──────────────────────────────────────────────────────────────────────


class TestKnownJob:
    """Escrow-only code path: history is None so the response
    contains a single ``escrow`` block (no ``compute`` block)."""

    def test_pending_escrow_returns_pending_status(self):
        node = _node_with_escrow({
            "forge-job-1": _entry(job_id="forge-job-1"),
        })
        client = _client(node)
        resp = client.get("/compute/status/forge-job-1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == "forge-job-1"
        assert "escrow" in body
        assert body["escrow"]["status"] == "pending"
        assert body["escrow"]["amount_ftns"] == 5.0
        assert body["escrow"]["completed_at"] is None
        # No history wired → no compute block.
        assert "compute" not in body

    def test_released_escrow_surfaces_provider_winner_and_release_tx(self):
        node = _node_with_escrow({
            "forge-job-2": _entry(
                job_id="forge-job-2",
                status=EscrowStatus.RELEASED,
                provider_winner="provider-node-7",
                completed_at=1_700_000_500.0,
            ),
        })
        body = _client(node).get("/compute/status/forge-job-2").json()
        assert body["escrow"]["status"] == "released"
        assert body["escrow"]["provider_winner"] == "provider-node-7"
        assert body["escrow"]["tx_release"] == "tx-release-xyz"
        assert body["escrow"]["completed_at"] == 1_700_000_500.0

    def test_refunded_escrow_returns_refunded_status(self):
        node = _node_with_escrow({
            "forge-job-3": _entry(
                job_id="forge-job-3",
                status=EscrowStatus.REFUNDED,
                completed_at=1_700_000_900.0,
            ),
        })
        body = _client(node).get("/compute/status/forge-job-3").json()
        assert body["escrow"]["status"] == "refunded"

    def test_metadata_dict_surfaced_unchanged(self):
        node = _node_with_escrow({
            "forge-job-4": _entry(job_id="forge-job-4"),
        })
        body = _client(node).get("/compute/status/forge-job-4").json()
        assert body["escrow"]["metadata"] == {"source": "test"}

    def test_response_shape_contract(self):
        # Pin the escrow-block JSON keyset so MCP clients
        # (prsm_agent_status) can rely on a stable contract for the
        # legacy escrow-only path.
        node = _node_with_escrow({
            "forge-job-5": _entry(job_id="forge-job-5"),
        })
        body = _client(node).get("/compute/status/forge-job-5").json()
        assert set(body.keys()) == {"job_id", "escrow"}
        expected_escrow_keys = {
            "escrow_id",
            "requester_id",
            "amount_ftns",
            "status",
            "provider_winner",
            "tx_lock",
            "tx_release",
            "created_at",
            "completed_at",
            "metadata",
        }
        assert set(body["escrow"].keys()) == expected_escrow_keys


# ──────────────────────────────────────────────────────────────────────
# Error paths — 404 + 503
# ──────────────────────────────────────────────────────────────────────


class TestErrorPaths:
    def test_unknown_job_returns_404(self):
        node = _node_with_escrow({})  # empty
        resp = _client(node).get("/compute/status/forge-missing")
        assert resp.status_code == 404
        assert "No history or escrow record" in resp.json()["detail"]

    def test_404_message_explains_eviction_and_budget_zero(self):
        # The error message MUST cue MCP clients about the two
        # ways a job can be missing: history-evicted (LRU) OR
        # never-locked-escrow (budget=0). Pin the explanatory
        # phrasing.
        node = _node_with_escrow({})
        body = _client(node).get("/compute/status/forge-missing").json()
        detail = body["detail"]
        assert "evicted" in detail
        assert "without a locked escrow" in detail

    def test_no_payment_escrow_and_no_history_returns_503(self):
        node = MagicMock()
        node.identity.node_id = "test-node"
        node._payment_escrow = None
        node._job_history = None
        resp = _client(node).get("/compute/status/forge-x")
        assert resp.status_code == 503
        assert "Neither" in resp.json()["detail"]


# ──────────────────────────────────────────────────────────────────────
# B8 async-dispatch follow-on — JobHistory tier
# ──────────────────────────────────────────────────────────────────────


from prsm.node.job_history import (
    JobHistoryRecord,
    JobHistoryStore,
    JobStatus,
)


def _node_with_history_and_escrow(
    *, history_records: dict | None = None,
    escrows_by_job_id: dict | None = None,
):
    node = MagicMock()
    node.identity.node_id = "test-node"

    history = JobHistoryStore(max_entries=10)
    for rec in (history_records or {}).values():
        history.put(rec)
    node._job_history = history

    escrow_mock = MagicMock()
    escrow_mock.get_escrow = lambda jid: (escrows_by_job_id or {}).get(jid)
    node._payment_escrow = escrow_mock
    return node


class TestHistoryTier:
    def test_history_only_returns_compute_block(self):
        rec = JobHistoryRecord(
            job_id="forge-h1", query="q",
            status=JobStatus.COMPLETED,
            started_at=1.0, completed_at=2.0,
            route="qo_swarm", response='{"count": 7}',
            aggregator_node_id="agg-7",
        )
        node = _node_with_history_and_escrow(
            history_records={"forge-h1": rec},
            escrows_by_job_id={},  # no escrow for this job
        )
        body = _client(node).get("/compute/status/forge-h1").json()
        assert "compute" in body
        assert body["compute"]["status"] == "completed"
        assert body["compute"]["route"] == "qo_swarm"
        assert body["compute"]["response"] == '{"count": 7}'
        assert body["compute"]["aggregator_node_id"] == "agg-7"
        # No escrow → no escrow block.
        assert "escrow" not in body

    def test_history_and_escrow_compose(self):
        # Both surfaces present → both blocks in the response.
        rec = JobHistoryRecord(
            job_id="forge-c1", query="q",
            status=JobStatus.COMPLETED,
            started_at=1.0, completed_at=2.0,
            route="qo_swarm", response="ok",
        )
        escrow_entry = _entry(
            job_id="forge-c1",
            status=EscrowStatus.RELEASED,
            provider_winner="split:3",
            completed_at=2.5,
        )
        node = _node_with_history_and_escrow(
            history_records={"forge-c1": rec},
            escrows_by_job_id={"forge-c1": escrow_entry},
        )
        body = _client(node).get("/compute/status/forge-c1").json()
        assert "compute" in body
        assert "escrow" in body
        assert body["compute"]["route"] == "qo_swarm"
        assert body["escrow"]["status"] == "released"
        assert body["escrow"]["provider_winner"] == "split:3"

    def test_failed_record_carries_error(self):
        rec = JobHistoryRecord(
            job_id="forge-f1", query="q",
            status=JobStatus.FAILED,
            started_at=1.0, completed_at=2.0,
            error="aggregator timeout",
        )
        node = _node_with_history_and_escrow(
            history_records={"forge-f1": rec},
        )
        body = _client(node).get("/compute/status/forge-f1").json()
        assert body["compute"]["status"] == "failed"
        assert body["compute"]["error"] == "aggregator timeout"

    def test_in_progress_record_omits_completion_fields(self):
        rec = JobHistoryRecord(
            job_id="forge-p1", query="q",
            status=JobStatus.IN_PROGRESS,
            started_at=1.0,
        )
        node = _node_with_history_and_escrow(
            history_records={"forge-p1": rec},
        )
        body = _client(node).get("/compute/status/forge-p1").json()
        assert body["compute"]["status"] == "in_progress"
        assert body["compute"]["completed_at"] is None
        assert body["compute"]["response"] is None
        assert body["compute"]["error"] is None

    def test_history_get_failure_falls_back_to_escrow(self):
        # If JobHistoryStore.get raises, the endpoint should not
        # crash — falls back to escrow lookup.
        bad_history = MagicMock()
        bad_history.get = MagicMock(side_effect=RuntimeError("history corrupt"))
        escrow_entry = _entry(job_id="forge-x")
        node = MagicMock()
        node.identity.node_id = "test-node"
        node._job_history = bad_history
        escrow_mock = MagicMock()
        escrow_mock.get_escrow = lambda jid: escrow_entry if jid == "forge-x" else None
        node._payment_escrow = escrow_mock

        body = _client(node).get("/compute/status/forge-x").json()
        # No compute block (history raised), but escrow surfaced.
        assert "escrow" in body
        assert "compute" not in body


# ──────────────────────────────────────────────────────────────────────
# Integration: /compute/forge writes history → /compute/status reads it
# ──────────────────────────────────────────────────────────────────────


class TestForgeWritesHistoryReadByStatus:
    """Drive the full /compute/forge → JobHistoryStore →
    /compute/status round-trip. Pins the contract that an
    immediately-following status call sees the COMPLETED record
    written at end of /compute/forge."""

    def test_completed_forge_call_visible_via_status(self):
        from dataclasses import dataclass

        @dataclass
        class _AggResult:
            query_id: bytes
            payload: bytes
            aggregator_node_id: str
            contributing_shards: tuple
            participants: tuple = ()

        class _OrchStub:
            async def dispatch_query(
                self, *, query, prompter_node_id, query_id,
                requires_tee=False, governance_denylist=frozenset(),
            ):
                return _AggResult(
                    query_id=query_id,
                    payload=b'{"count": 11}',
                    aggregator_node_id="agg-int",
                    contributing_shards=("cid-x",),
                )

        node = MagicMock()
        node.identity.node_id = "prompter-1"
        node.privacy_budget = None
        node.agent_forge = _OrchStub()
        node._payment_escrow = None  # skip escrow path
        node._job_history = JobHistoryStore(max_entries=10)

        client = _client(node)
        forge_resp = client.post("/compute/forge", json={
            "query": "Count records",
            "budget_ftns": 1.0,
        })
        assert forge_resp.status_code == 200
        forge_job_id = forge_resp.json()["job_id"]

        # Now call /compute/status for that same job_id and assert
        # the history record is visible.
        status_resp = client.get(f"/compute/status/{forge_job_id}")
        assert status_resp.status_code == 200
        body = status_resp.json()
        assert "compute" in body
        assert body["compute"]["status"] == "completed"
        assert body["compute"]["route"] == "qo_swarm"
        assert body["compute"]["response"] == '{"count": 11}'
        assert body["compute"]["aggregator_node_id"] == "agg-int"
        assert body["compute"]["query"] == "Count records"
