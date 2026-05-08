"""PRSM-PROV-1 T6.5 — arbitration queue tests.

When an upload's similarity to an existing CID lands in the
``[arbitration_floor, derivative)`` band, instead of auto-attributing
the candidate parent we enqueue a ``DisputedAttributionRecord`` for
council review. This module tests the queue's two implementations
(in-memory + filesystem-persistent) and the record dataclass.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json

import pytest

from prsm.data.dedup.arbitration import (
    ArbitrationDecision,
    ArbitrationProposalSink,
    DisputedAttributionRecord,
    FilesystemArbitrationQueue,
    InMemoryArbitrationQueue,
    NullArbitrationProposalSink,
    render_arbitration_body,
)


# ──────────────────────────────────────────────────────────────────────
# Fixture
# ──────────────────────────────────────────────────────────────────────


def _record(
    *,
    new_cid: str = "cid-new",
    candidate: str = "cid-old",
    sim: float = 0.78,
    kind: str = "text-vector",
) -> DisputedAttributionRecord:
    return DisputedAttributionRecord(
        new_cid=new_cid,
        new_creator="0xnew",
        candidate_parent_cid=candidate,
        candidate_parent_creator="0xold",
        similarity=sim,
        fingerprint_kind=kind,
        flagged_at=1_700_000_000,
        proposal_id=None,
    )


# ──────────────────────────────────────────────────────────────────────
# DisputedAttributionRecord — schema + serialisation
# ──────────────────────────────────────────────────────────────────────


class TestDisputedAttributionRecord:
    def test_construction_and_equality(self):
        a = _record()
        b = _record()
        assert a == b

    def test_immutable_frozen_dataclass(self):
        rec = _record()
        with pytest.raises(dataclasses.FrozenInstanceError):
            rec.new_cid = "tamper"  # type: ignore[misc]

    def test_to_from_dict_roundtrip(self):
        rec = _record()
        rt = DisputedAttributionRecord.from_dict(rec.to_dict())
        assert rt == rec

    def test_negative_similarity_rejected(self):
        with pytest.raises(ValueError, match="similarity"):
            DisputedAttributionRecord(
                new_cid="a",
                new_creator="b",
                candidate_parent_cid="c",
                candidate_parent_creator="d",
                similarity=-0.1,
                fingerprint_kind="text-vector",
                flagged_at=0,
                proposal_id=None,
            )

    def test_similarity_above_one_rejected(self):
        with pytest.raises(ValueError, match="similarity"):
            DisputedAttributionRecord(
                new_cid="a",
                new_creator="b",
                candidate_parent_cid="c",
                candidate_parent_creator="d",
                similarity=1.5,
                fingerprint_kind="text-vector",
                flagged_at=0,
                proposal_id=None,
            )

    def test_empty_cid_rejected(self):
        with pytest.raises(ValueError, match="new_cid"):
            DisputedAttributionRecord(
                new_cid="",
                new_creator="b",
                candidate_parent_cid="c",
                candidate_parent_creator="d",
                similarity=0.5,
                fingerprint_kind="text-vector",
                flagged_at=0,
                proposal_id=None,
            )


# ──────────────────────────────────────────────────────────────────────
# InMemoryArbitrationQueue
# ──────────────────────────────────────────────────────────────────────


class TestInMemoryQueue:
    def test_enqueue_returns_unique_id(self):
        q = InMemoryArbitrationQueue()
        rid_a = asyncio.run(q.enqueue(_record(new_cid="a")))
        rid_b = asyncio.run(q.enqueue(_record(new_cid="b")))
        assert rid_a != rid_b

    def test_get_after_enqueue(self):
        q = InMemoryArbitrationQueue()
        rec = _record()
        rid = asyncio.run(q.enqueue(rec))
        recovered = asyncio.run(q.get(rid))
        assert recovered == rec

    def test_get_unknown_returns_none(self):
        q = InMemoryArbitrationQueue()
        assert asyncio.run(q.get("missing-id")) is None

    def test_list_pending_orders_by_flagged_at(self):
        q = InMemoryArbitrationQueue()
        # Flag B earlier than A — list_pending must surface the older
        # (still-unresolved) record first.
        b = dataclasses.replace(_record(new_cid="b"), flagged_at=100)
        a = dataclasses.replace(_record(new_cid="a"), flagged_at=200)
        asyncio.run(q.enqueue(a))
        asyncio.run(q.enqueue(b))
        pending = asyncio.run(q.list_pending())
        assert [r.new_cid for r in pending] == ["b", "a"]

    def test_resolve_removes_from_pending(self):
        q = InMemoryArbitrationQueue()
        rid = asyncio.run(q.enqueue(_record()))
        asyncio.run(q.resolve(
            rid,
            decision=ArbitrationDecision.UPHELD_PARENT,
            by_council=["0xcouncil-a", "0xcouncil-b"],
        ))
        assert asyncio.run(q.list_pending()) == []

    def test_resolve_unknown_id_raises(self):
        q = InMemoryArbitrationQueue()
        with pytest.raises(KeyError):
            asyncio.run(q.resolve(
                "missing",
                decision=ArbitrationDecision.UPHELD_PARENT,
                by_council=["0xa"],
            ))

    def test_resolve_twice_idempotent(self):
        # Idempotent resolve: re-resolving the same record with the same
        # decision must not raise. Defends against governance webhook
        # double-delivery.
        q = InMemoryArbitrationQueue()
        rid = asyncio.run(q.enqueue(_record()))
        asyncio.run(q.resolve(
            rid,
            decision=ArbitrationDecision.REJECTED_PARENT,
            by_council=["0xa"],
        ))
        # Second call with SAME decision → no raise.
        asyncio.run(q.resolve(
            rid,
            decision=ArbitrationDecision.REJECTED_PARENT,
            by_council=["0xa"],
        ))

    def test_resolve_with_conflicting_decision_raises(self):
        # A record resolved REJECTED cannot later flip to UPHELD without
        # an explicit reopen — guards against accidental double-resolves
        # delivering different verdicts.
        q = InMemoryArbitrationQueue()
        rid = asyncio.run(q.enqueue(_record()))
        asyncio.run(q.resolve(
            rid,
            decision=ArbitrationDecision.REJECTED_PARENT,
            by_council=["0xa"],
        ))
        with pytest.raises(ValueError, match="conflicting"):
            asyncio.run(q.resolve(
                rid,
                decision=ArbitrationDecision.UPHELD_PARENT,
                by_council=["0xa"],
            ))

    def test_set_proposal_id(self):
        q = InMemoryArbitrationQueue()
        rid = asyncio.run(q.enqueue(_record()))
        asyncio.run(q.set_proposal_id(rid, "prop-123"))
        rec = asyncio.run(q.get(rid))
        assert rec is not None and rec.proposal_id == "prop-123"


# ──────────────────────────────────────────────────────────────────────
# FilesystemArbitrationQueue — persistence across restart
# ──────────────────────────────────────────────────────────────────────


class TestFilesystemQueue:
    def test_persists_across_construction(self, tmp_path):
        q1 = FilesystemArbitrationQueue(tmp_path / "queue")
        rid = asyncio.run(q1.enqueue(_record()))
        # Reconstruct with same path — record must survive.
        q2 = FilesystemArbitrationQueue(tmp_path / "queue")
        rec = asyncio.run(q2.get(rid))
        assert rec == _record()

    def test_resolve_persists_across_restart(self, tmp_path):
        q1 = FilesystemArbitrationQueue(tmp_path / "queue")
        rid = asyncio.run(q1.enqueue(_record()))
        asyncio.run(q1.resolve(
            rid,
            decision=ArbitrationDecision.UPHELD_PARENT,
            by_council=["0xa"],
        ))
        q2 = FilesystemArbitrationQueue(tmp_path / "queue")
        assert asyncio.run(q2.list_pending()) == []
        # Resolved record still retrievable for audit history.
        rec = asyncio.run(q2.get(rid))
        assert rec is not None

    def test_corrupt_file_skipped_with_warning(self, tmp_path, caplog):
        q = FilesystemArbitrationQueue(tmp_path / "queue")
        # Write an invalid JSON file in the queue dir; reconstruction
        # must skip it (and log) rather than raising.
        (tmp_path / "queue").mkdir(exist_ok=True)
        (tmp_path / "queue" / "bogus.json").write_text(
            "{ not valid json", encoding="utf-8",
        )
        # Reload — must not raise.
        q2 = FilesystemArbitrationQueue(tmp_path / "queue")
        assert asyncio.run(q2.list_pending()) == []


# ──────────────────────────────────────────────────────────────────────
# T6.5.gov — proposal sink + body renderer
# ──────────────────────────────────────────────────────────────────────


class TestRenderArbitrationBody:
    """The body string is the deterministic representation councils
    will read AND a future on-chain arbitration contract may verify
    bytes against. Pin field order, line breaks, decimal formatting."""

    def test_includes_all_load_bearing_fields(self):
        rec = _record(
            new_cid="cid-uploader",
            candidate="cid-claimed-parent",
            sim=0.78,
            kind="image-phash",
        )
        body = render_arbitration_body(rec)
        assert "cid-uploader" in body
        assert "cid-claimed-parent" in body
        assert "image-phash" in body
        assert "0xnew" in body
        assert "0xold" in body
        # Similarity must show 6-decimal precision so two near-identical
        # disputed-band records don't render identically.
        assert "0.780000" in body
        # Flagged-at unix timestamp surfaced for audit history.
        assert str(rec.flagged_at) in body

    def test_deterministic_for_equal_records(self):
        a = _record()
        b = _record()
        assert render_arbitration_body(a) == render_arbitration_body(b)

    def test_distinguishes_records_by_similarity(self):
        a = _record(sim=0.78)
        b = _record(sim=0.79)
        assert render_arbitration_body(a) != render_arbitration_body(b)

    def test_starts_with_pinned_header(self):
        # The first line is part of the deterministic contract — a
        # future on-chain arbitration contract may sign over the
        # bytes of this body to commit the council's review target.
        body = render_arbitration_body(_record())
        assert body.startswith(
            "PRSM-PROV-1 disputed-attribution review\n"
        )


class TestNullArbitrationProposalSink:
    def test_satisfies_protocol(self):
        sink = NullArbitrationProposalSink()
        assert isinstance(sink, ArbitrationProposalSink)

    def test_returns_none_for_any_record(self):
        sink = NullArbitrationProposalSink()
        rec = _record()
        result = asyncio.run(
            sink.create_arbitration_proposal(rec, "rid-123"),
        )
        assert result is None
