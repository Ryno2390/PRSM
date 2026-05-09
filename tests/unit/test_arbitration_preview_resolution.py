"""POST /content/arbitration/preview-resolution — composer-only
preview of what queue.resolve() WOULD do.

Council members composing a resolution from the AI side panel
need a dry-run that surfaces:
- The full record context (inherits from the record itself)
- The proposed decision + by_council list
- Whether the proposed decision conflicts with any existing
  resolution (idempotent-equal vs conflicting-different)

The preview endpoint does NOT call queue.resolve(). It's purely
a composer-only artifact. Council members confirm intent +
sign on-chain governance proposal separately. Designed for the
case where local-resolve auth model is still pending council
ratification.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.data.dedup.arbitration import DisputedAttributionRecord


def _record(*, similarity=0.85, kind="image"):
    return DisputedAttributionRecord(
        new_cid="bafy-x",
        new_creator="0x1111",
        candidate_parent_cid="bafy-y",
        candidate_parent_creator="0x2222",
        similarity=similarity,
        fingerprint_kind=kind,
        flagged_at=1_700_000_000,
        proposal_id=None,
    )


def _node(*, queue=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node._arbitration_queue = queue
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


# ──────────────────────────────────────────────────────────────────────
# Service availability
# ──────────────────────────────────────────────────────────────────────


class TestPreviewAvailability:
    def test_503_when_queue_not_wired(self):
        node = _node()
        resp = _client(node).post(
            "/content/arbitration/preview-resolution",
            json={
                "record_id": "x",
                "decision": "upheld_parent",
                "by_council": ["0xC1"],
            },
        )
        assert resp.status_code == 503

    def test_404_when_record_unknown(self):
        queue = MagicMock()
        queue.get = AsyncMock(return_value=None)
        node = _node(queue=queue)
        resp = _client(node).post(
            "/content/arbitration/preview-resolution",
            json={
                "record_id": "missing",
                "decision": "upheld_parent",
                "by_council": ["0xC1"],
            },
        )
        assert resp.status_code == 404


# ──────────────────────────────────────────────────────────────────────
# Composer-only invariant
# ──────────────────────────────────────────────────────────────────────


class TestPreviewIsComposerOnly:
    def test_does_not_call_resolve(self):
        """Critical invariant: preview must NEVER call queue.resolve()."""
        rec = _record()
        queue = MagicMock()
        queue.get = AsyncMock(return_value=rec)
        queue.get_resolution = AsyncMock(return_value=None)
        queue.resolve = AsyncMock()  # spy
        node = _node(queue=queue)
        resp = _client(node).post(
            "/content/arbitration/preview-resolution",
            json={
                "record_id": "x",
                "decision": "upheld_parent",
                "by_council": ["0xC1"],
            },
        )
        assert resp.status_code == 200
        # resolve() must NOT have been called.
        queue.resolve.assert_not_called()

    def test_returns_dry_run_status(self):
        rec = _record()
        queue = MagicMock()
        queue.get = AsyncMock(return_value=rec)
        queue.get_resolution = AsyncMock(return_value=None)
        node = _node(queue=queue)
        resp = _client(node).post(
            "/content/arbitration/preview-resolution",
            json={
                "record_id": "x",
                "decision": "upheld_parent",
                "by_council": ["0xC1"],
            },
        )
        body = resp.json()
        assert body["status"] == "DRY_RUN"

    def test_response_includes_preview_artifact_fields(self):
        rec = _record(similarity=0.91, kind="audio")
        queue = MagicMock()
        queue.get = AsyncMock(return_value=rec)
        queue.get_resolution = AsyncMock(return_value=None)
        node = _node(queue=queue)
        resp = _client(node).post(
            "/content/arbitration/preview-resolution",
            json={
                "record_id": "x",
                "decision": "rejected_parent",
                "by_council": ["0xC1", "0xC2"],
            },
        )
        body = resp.json()
        assert body["record"]["new_cid"] == "bafy-x"
        assert body["proposed"]["decision"] == "rejected_parent"
        assert body["proposed"]["by_council"] == ["0xC1", "0xC2"]
        assert body["current_resolution"] is None
        assert body["conflict_with_existing"] is False
        assert "note" in body  # operator action hint


# ──────────────────────────────────────────────────────────────────────
# Conflict detection
# ──────────────────────────────────────────────────────────────────────


class TestConflictDetection:
    def test_proposed_matches_existing_no_conflict(self):
        """Idempotent-equal: same decision as existing resolution
        → no conflict."""
        rec = _record()
        queue = MagicMock()
        queue.get = AsyncMock(return_value=rec)
        queue.get_resolution = AsyncMock(return_value={
            "decision": "upheld_parent",
            "by_council": ["0xC1"],
        })
        node = _node(queue=queue)
        resp = _client(node).post(
            "/content/arbitration/preview-resolution",
            json={
                "record_id": "x",
                "decision": "upheld_parent",  # same as existing
                "by_council": ["0xC1"],
            },
        )
        body = resp.json()
        assert body["conflict_with_existing"] is False

    def test_proposed_differs_from_existing_conflict_flagged(self):
        """Conflicting-different: existing said upheld_parent,
        proposed says rejected_parent → conflict."""
        rec = _record()
        queue = MagicMock()
        queue.get = AsyncMock(return_value=rec)
        queue.get_resolution = AsyncMock(return_value={
            "decision": "upheld_parent",
            "by_council": ["0xC1"],
        })
        node = _node(queue=queue)
        resp = _client(node).post(
            "/content/arbitration/preview-resolution",
            json={
                "record_id": "x",
                "decision": "rejected_parent",  # conflicts!
                "by_council": ["0xC2"],
            },
        )
        body = resp.json()
        assert body["conflict_with_existing"] is True
        # current_resolution surfaced so operator sees what's already locked.
        assert body["current_resolution"]["decision"] == "upheld_parent"


# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────


class TestValidation:
    def test_invalid_decision_returns_422(self):
        rec = _record()
        queue = MagicMock()
        queue.get = AsyncMock(return_value=rec)
        node = _node(queue=queue)
        resp = _client(node).post(
            "/content/arbitration/preview-resolution",
            json={
                "record_id": "x",
                "decision": "made_up_value",
                "by_council": ["0xC1"],
            },
        )
        assert resp.status_code == 422

    def test_empty_by_council_returns_422(self):
        rec = _record()
        queue = MagicMock()
        queue.get = AsyncMock(return_value=rec)
        node = _node(queue=queue)
        resp = _client(node).post(
            "/content/arbitration/preview-resolution",
            json={
                "record_id": "x",
                "decision": "upheld_parent",
                "by_council": [],
            },
        )
        assert resp.status_code == 422
