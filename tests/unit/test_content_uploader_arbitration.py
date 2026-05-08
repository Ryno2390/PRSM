"""PRSM-PROV-1 Item 6 T6.3 + T6.5 — ContentUploader three-band wiring.

Closes the integration between ``ThresholdResolver`` (T6.2 surface
already shipped) and ``ContentUploader``'s embedding-path dedup
branch:

  similarity >= duplicate_threshold        → auto-attribute (warn)
  similarity >= derivative_threshold       → auto-attribute (info)
  arbitration_floor <= sim < derivative    → enqueue, NO auto-parent
  similarity < arbitration_floor           → no-op

Earlier behavior (pre-T6.5) had only the first two branches —
borderline matches auto-attributed against creators' will and
created the dispute pattern this work fixes.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from prsm.data.dedup.arbitration import (
    DisputedAttributionRecord,
    InMemoryArbitrationQueue,
    NullArbitrationProposalSink,
)
from prsm.data.dedup.thresholds import ThresholdResolver
from prsm.node.content_uploader import ContentUploader, _SemanticIndex
from prsm.node.identity import generate_node_identity


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


def _make_uploader(
    *,
    threshold_resolver=None,
    arbitration_queue=None,
    embedding_model_id=None,
    arbitration_proposal_sink=None,
):
    identity = generate_node_identity("test-node")
    # gossip.publish is awaited inside upload() — must be AsyncMock.
    gossip = MagicMock()
    gossip.publish = AsyncMock()
    ledger = MagicMock()
    return ContentUploader(
        identity=identity,
        gossip=gossip,
        ledger=ledger,
        embedding_model_id=embedding_model_id,
        threshold_resolver=threshold_resolver,
        arbitration_queue=arbitration_queue,
        arbitration_proposal_sink=arbitration_proposal_sink,
    )


# ──────────────────────────────────────────────────────────────────────
# Constructor — backwards-compat
# ──────────────────────────────────────────────────────────────────────


class TestConstructor:
    def test_defaults_preserve_legacy_behavior(self):
        uploader = _make_uploader()
        assert uploader._threshold_resolver is None
        assert uploader._arbitration_queue is None

    def test_resolver_and_queue_stored(self):
        resolver = ThresholdResolver.from_default_path()
        queue = InMemoryArbitrationQueue()
        uploader = _make_uploader(
            threshold_resolver=resolver,
            arbitration_queue=queue,
        )
        assert uploader._threshold_resolver is resolver
        assert uploader._arbitration_queue is queue


# ──────────────────────────────────────────────────────────────────────
# _resolve_text_thresholds — helper
# ──────────────────────────────────────────────────────────────────────


class TestResolveTextThresholds:
    def test_returns_none_when_resolver_unwired(self):
        uploader = _make_uploader()
        assert uploader._resolve_text_thresholds(None) is None
        assert uploader._resolve_text_thresholds({"key": "val"}) is None

    def test_returns_effective_thresholds_for_default_kind(self):
        resolver = ThresholdResolver.from_default_path()
        uploader = _make_uploader(threshold_resolver=resolver)
        eff = uploader._resolve_text_thresholds(None)
        assert eff is not None
        assert 0.0 <= eff.arbitration_floor < eff.derivative <= eff.duplicate

    def test_consults_model_id_when_set(self):
        resolver = ThresholdResolver.from_default_path()
        # MiniLM: derivative=0.85; bare text-vector: 0.92.
        bare_uploader = _make_uploader(threshold_resolver=resolver)
        minilm_uploader = _make_uploader(
            threshold_resolver=resolver,
            embedding_model_id="sentence-transformers/all-MiniLM-L6-v2",
        )
        bare_eff = bare_uploader._resolve_text_thresholds(None)
        minilm_eff = minilm_uploader._resolve_text_thresholds(None)
        assert bare_eff.derivative > minilm_eff.derivative

    def test_content_type_hint_from_metadata(self):
        resolver = ThresholdResolver.from_default_path()
        uploader = _make_uploader(threshold_resolver=resolver)
        plain = uploader._resolve_text_thresholds(None)
        hinted = uploader._resolve_text_thresholds(
            {"content_type_hint": "scientific_abstract"},
        )
        # scientific_abstract tightens derivative.
        assert hinted.derivative > plain.derivative

    def test_resolver_failure_falls_back_to_none(self):
        # A pathological resolver that raises must NOT crash uploads.
        bad_resolver = MagicMock()
        bad_resolver.resolve.side_effect = RuntimeError("boom")
        uploader = _make_uploader(threshold_resolver=bad_resolver)
        assert uploader._resolve_text_thresholds(None) is None

    def test_empty_string_hint_ignored(self):
        # An empty-string hint must be treated as "no hint" — not
        # passed verbatim to the resolver where it would miss the
        # multiplier table.
        resolver = ThresholdResolver.from_default_path()
        uploader = _make_uploader(threshold_resolver=resolver)
        eff = uploader._resolve_text_thresholds({"content_type_hint": ""})
        assert eff is not None
        assert eff.hint_applied is None


# ──────────────────────────────────────────────────────────────────────
# _enqueue_arbitration — helper
# ──────────────────────────────────────────────────────────────────────


class TestEnqueueArbitration:
    def test_record_composed_with_correct_fields(self):
        queue = InMemoryArbitrationQueue()
        uploader = _make_uploader(arbitration_queue=queue)
        uploader.creator_address = "0xnewcreator"
        pending = {
            "candidate_parent_cid": "cid-old",
            "candidate_parent_creator": "0xoldcreator",
            "similarity": 0.78,
            "fingerprint_kind": "text-vector",
        }
        asyncio.run(uploader._enqueue_arbitration("cid-new", pending))
        records = asyncio.run(queue.list_pending())
        assert len(records) == 1
        rec = records[0]
        assert rec.new_cid == "cid-new"
        assert rec.new_creator == "0xnewcreator"
        assert rec.candidate_parent_cid == "cid-old"
        assert rec.candidate_parent_creator == "0xoldcreator"
        assert rec.similarity == pytest.approx(0.78)
        assert rec.fingerprint_kind == "text-vector"
        # flagged_at must be set to current wall time, not 0.
        assert rec.flagged_at > 0

    def test_enqueue_failure_does_not_raise(self):
        # An arbitration_queue that raises on enqueue must not crash
        # the upload — the design doc's "anti-griefing-safe" property.
        bad_queue = MagicMock()
        bad_queue.enqueue = AsyncMock(side_effect=RuntimeError("boom"))
        uploader = _make_uploader(arbitration_queue=bad_queue)
        uploader.creator_address = "0x" + "11" * 20
        pending = {
            "candidate_parent_cid": "cid-old",
            "candidate_parent_creator": "0xold",
            "similarity": 0.78,
            "fingerprint_kind": "text-vector",
        }
        # Must not raise.
        asyncio.run(uploader._enqueue_arbitration("cid-new", pending))

    def test_empty_creator_address_falls_back_to_empty_string(self):
        # creator_address can be None on legacy nodes — cipher must
        # not crash.
        queue = InMemoryArbitrationQueue()
        uploader = _make_uploader(arbitration_queue=queue)
        uploader.creator_address = None
        pending = {
            "candidate_parent_cid": "cid-old",
            "candidate_parent_creator": "0xold",
            "similarity": 0.78,
            "fingerprint_kind": "text-vector",
        }
        asyncio.run(uploader._enqueue_arbitration("cid-new", pending))
        records = asyncio.run(queue.list_pending())
        assert records[0].new_creator == ""


# ──────────────────────────────────────────────────────────────────────
# End-to-end via upload() with stubbed publish + semantic index
# ──────────────────────────────────────────────────────────────────────


class TestUploadThreeBandRouting:
    """Drive ``upload()`` with stubbed dependencies + a pre-loaded
    _SemanticIndex match. Verifies that the three-band branch fires
    based on the ``find_nearest`` similarity score."""

    def _seed_uploader(
        self, *, similarity: float, threshold_resolver=None,
        arbitration_queue=None, embedding_model_id=None,
    ):
        uploader = _make_uploader(
            threshold_resolver=threshold_resolver,
            arbitration_queue=arbitration_queue,
            embedding_model_id=embedding_model_id,
        )
        uploader.creator_address = "0x" + "11" * 20
        # Stub embedding so the embedding-path branch is taken.
        async def _embedding_fn(content):
            return np.array([1.0, 0.0], dtype=np.float32)
        uploader._get_embedding = _embedding_fn
        # Stub the index so find_nearest returns our crafted similarity.
        uploader._semantic_index.find_nearest = (
            lambda emb: ("cid-old", similarity, "0xoldcreator")
        )
        # Stub publish to a deterministic CID.
        async def _publish_stub(content, filename, ph):
            return "cid-new"
        uploader._publish_content = _publish_stub
        # Stub other side-effects we don't care about here.
        uploader._register_local_embedding = MagicMock()
        uploader._register_local_content = MagicMock()
        uploader._broadcast_provenance = AsyncMock()
        # No content_publisher means _publish_content is what runs;
        # we've already stubbed that.
        return uploader

    def test_disputed_band_enqueues_no_auto_parent(self):
        resolver = ThresholdResolver.from_default_path()
        queue = InMemoryArbitrationQueue()
        # text-vector: derivative=0.92, arbitration_floor=0.82 (default).
        uploader = self._seed_uploader(
            similarity=0.86,  # in disputed band
            threshold_resolver=resolver,
            arbitration_queue=queue,
        )
        result = asyncio.run(uploader.upload(b"some content"))
        assert result is not None
        # cid-old must NOT be in parents (no auto-attribute).
        assert "cid-old" not in (result.parent_cids or [])
        # Queue length grew by one.
        records = asyncio.run(queue.list_pending())
        assert len(records) == 1
        assert records[0].similarity == pytest.approx(0.86)

    def test_above_derivative_auto_attributes(self):
        resolver = ThresholdResolver.from_default_path()
        queue = InMemoryArbitrationQueue()
        uploader = self._seed_uploader(
            similarity=0.95,  # > derivative=0.92
            threshold_resolver=resolver,
            arbitration_queue=queue,
        )
        result = asyncio.run(uploader.upload(b"some content"))
        assert result is not None
        # cid-old IS auto-prepended as parent.
        assert "cid-old" in (result.parent_cids or [])
        # No arbitration record enqueued.
        assert asyncio.run(queue.list_pending()) == []

    def test_below_floor_no_op(self):
        resolver = ThresholdResolver.from_default_path()
        queue = InMemoryArbitrationQueue()
        uploader = self._seed_uploader(
            similarity=0.50,  # < arbitration_floor=0.82
            threshold_resolver=resolver,
            arbitration_queue=queue,
        )
        result = asyncio.run(uploader.upload(b"some content"))
        assert result is not None
        assert "cid-old" not in (result.parent_cids or [])
        assert asyncio.run(queue.list_pending()) == []

    def test_no_arbitration_queue_disables_disputed_band(self):
        # Resolver wired but queue=None. The disputed band branch
        # MUST NOT fire (since there's nowhere to enqueue) — falls
        # through to no-op, preserving legacy 2-band behavior.
        resolver = ThresholdResolver.from_default_path()
        uploader = self._seed_uploader(
            similarity=0.86,  # in would-be disputed band
            threshold_resolver=resolver,
            arbitration_queue=None,
        )
        result = asyncio.run(uploader.upload(b"some content"))
        assert result is not None
        # No parent — but also no record-keeping (legacy gap).
        assert "cid-old" not in (result.parent_cids or [])

    def test_no_resolver_uses_legacy_class_constants(self):
        # Without a resolver, the 0.92 / 0.99 class-constant
        # thresholds apply. Sim=0.86 must fall into "no-op" because
        # 0.86 < 0.92, not into a disputed band.
        queue = InMemoryArbitrationQueue()
        uploader = self._seed_uploader(
            similarity=0.86,
            threshold_resolver=None,
            arbitration_queue=queue,
        )
        result = asyncio.run(uploader.upload(b"some content"))
        assert result is not None
        assert "cid-old" not in (result.parent_cids or [])
        # Even with a queue wired, no resolver = no disputed band.
        assert asyncio.run(queue.list_pending()) == []


# ──────────────────────────────────────────────────────────────────────
# T6.5.x — binary-path 3-band wiring (image-phash / audio / video)
# ──────────────────────────────────────────────────────────────────────


class TestResolveBinaryThresholds:
    def test_returns_none_when_resolver_unwired(self):
        from prsm.data.fingerprints.base import FingerprintKind
        uploader = _make_uploader()
        assert uploader._resolve_binary_thresholds(
            FingerprintKind.IMAGE_PHASH,
        ) is None

    def test_returns_effective_thresholds_for_image(self):
        from prsm.data.fingerprints.base import FingerprintKind
        resolver = ThresholdResolver.from_default_path()
        uploader = _make_uploader(threshold_resolver=resolver)
        eff = uploader._resolve_binary_thresholds(
            FingerprintKind.IMAGE_PHASH,
        )
        assert eff is not None
        # image-phash YAML: derivative=0.81, arbitration_floor defaults
        # to 0.71 (derivative - 0.10).
        assert eff.derivative == pytest.approx(0.81)
        assert eff.arbitration_floor == pytest.approx(0.71)

    def test_returns_effective_thresholds_for_audio(self):
        from prsm.data.fingerprints.base import FingerprintKind
        resolver = ThresholdResolver.from_default_path()
        uploader = _make_uploader(threshold_resolver=resolver)
        eff = uploader._resolve_binary_thresholds(
            FingerprintKind.AUDIO_CHROMAPRINT,
        )
        assert eff is not None
        assert eff.derivative == pytest.approx(0.75)

    def test_returns_effective_thresholds_for_video(self):
        from prsm.data.fingerprints.base import FingerprintKind
        resolver = ThresholdResolver.from_default_path()
        uploader = _make_uploader(threshold_resolver=resolver)
        eff = uploader._resolve_binary_thresholds(
            FingerprintKind.VIDEO_MULTIHASH,
        )
        assert eff is not None
        assert eff.derivative == pytest.approx(0.625)

    def test_resolver_failure_falls_back_to_none(self):
        from prsm.data.fingerprints.base import FingerprintKind
        bad_resolver = MagicMock()
        bad_resolver.resolve.side_effect = RuntimeError("boom")
        uploader = _make_uploader(threshold_resolver=bad_resolver)
        assert uploader._resolve_binary_thresholds(
            FingerprintKind.IMAGE_PHASH,
        ) is None


class TestUploadBinaryThreeBandRouting:
    """End-to-end: drive ``upload()`` through the binary-fingerprint
    path. Embedding stub returns None so that branch falls through to
    the bin_match logic."""

    def _seed_binary_uploader(
        self, *, kind, similarity: float, threshold_resolver=None,
        arbitration_queue=None,
    ):
        from prsm.data.fingerprints.index import FingerprintMatch
        uploader = _make_uploader(
            threshold_resolver=threshold_resolver,
            arbitration_queue=arbitration_queue,
        )
        uploader.creator_address = "0x" + "11" * 20
        # No embedding -> binary path fires.
        async def _embedding_fn(content):
            return None
        uploader._get_embedding = _embedding_fn
        # Stub the binary-fingerprint computation to a real
        # FingerprintRecord (downstream code calls .kind on it after
        # publish to register the fingerprint with the index).
        from prsm.data.fingerprints.base import FingerprintRecord
        uploader._maybe_compute_binary_fingerprint = (
            lambda content, filename: FingerprintRecord(
                kind=kind, payload=b"\x00" * 8,
            )
        )
        # Suppress the post-publish fingerprint registration side-effects
        # (we don't need them for these tests).
        uploader._fingerprint_index.store = MagicMock()
        uploader._register_local_fingerprint = MagicMock()
        match = FingerprintMatch(
            content_id="cid-bin-old",
            similarity=similarity,
            creator_id="0xbincreator",
            kind=kind,
        )
        uploader._fingerprint_index.find_nearest = lambda rec: match
        # Backstops needed when resolver is None (legacy path
        # consults the index directly).
        uploader._fingerprint_index.derivative_threshold = (
            lambda k: 0.81 if k == kind else 0.5
        )
        uploader._fingerprint_index.duplicate_threshold = (
            lambda k: 0.94 if k == kind else 0.95
        )
        async def _publish_stub(content, filename, ph):
            return "cid-bin-new"
        uploader._publish_content = _publish_stub
        uploader._register_local_embedding = MagicMock()
        uploader._register_local_content = MagicMock()
        uploader._broadcast_provenance = AsyncMock()
        return uploader

    def test_image_disputed_band_enqueues_no_auto_parent(self):
        from prsm.data.fingerprints.base import FingerprintKind
        resolver = ThresholdResolver.from_default_path()
        queue = InMemoryArbitrationQueue()
        # image-phash: derivative=0.81, arbitration_floor=0.71.
        uploader = self._seed_binary_uploader(
            kind=FingerprintKind.IMAGE_PHASH,
            similarity=0.75,  # in disputed band
            threshold_resolver=resolver,
            arbitration_queue=queue,
        )
        result = asyncio.run(uploader.upload(b"binary content"))
        assert result is not None
        assert "cid-bin-old" not in (result.parent_cids or [])
        records = asyncio.run(queue.list_pending())
        assert len(records) == 1
        rec = records[0]
        assert rec.fingerprint_kind == "image-phash"
        assert rec.candidate_parent_cid == "cid-bin-old"
        assert rec.candidate_parent_creator == "0xbincreator"
        assert rec.similarity == pytest.approx(0.75)

    def test_audio_above_derivative_auto_attributes(self):
        from prsm.data.fingerprints.base import FingerprintKind
        resolver = ThresholdResolver.from_default_path()
        queue = InMemoryArbitrationQueue()
        uploader = self._seed_binary_uploader(
            kind=FingerprintKind.AUDIO_CHROMAPRINT,
            similarity=0.80,  # > derivative=0.75
            threshold_resolver=resolver,
            arbitration_queue=queue,
        )
        # Make the fallback thresholds match the audio kind so the
        # legacy-path checks (called when resolver returns) line up.
        result = asyncio.run(uploader.upload(b"binary content"))
        assert result is not None
        assert "cid-bin-old" in (result.parent_cids or [])
        assert asyncio.run(queue.list_pending()) == []

    def test_video_below_floor_no_op(self):
        from prsm.data.fingerprints.base import FingerprintKind
        resolver = ThresholdResolver.from_default_path()
        queue = InMemoryArbitrationQueue()
        uploader = self._seed_binary_uploader(
            kind=FingerprintKind.VIDEO_MULTIHASH,
            similarity=0.30,  # < arbitration_floor=0.525
            threshold_resolver=resolver,
            arbitration_queue=queue,
        )
        result = asyncio.run(uploader.upload(b"binary content"))
        assert result is not None
        assert "cid-bin-old" not in (result.parent_cids or [])
        assert asyncio.run(queue.list_pending()) == []

    def test_no_resolver_preserves_legacy_2_band_behavior(self):
        from prsm.data.fingerprints.base import FingerprintKind
        # Without a resolver, the FingerprintIndex's built-in
        # 2-band thresholds apply (derivative=0.81 for image-phash).
        # Sim=0.75 falls below derivative → no-op (legacy).
        queue = InMemoryArbitrationQueue()
        uploader = self._seed_binary_uploader(
            kind=FingerprintKind.IMAGE_PHASH,
            similarity=0.75,
            threshold_resolver=None,
            arbitration_queue=queue,
        )
        result = asyncio.run(uploader.upload(b"binary content"))
        assert result is not None
        assert "cid-bin-old" not in (result.parent_cids or [])
        # No resolver → no disputed band, even with queue wired.
        assert asyncio.run(queue.list_pending()) == []

    def test_no_arbitration_queue_disables_binary_disputed_band(self):
        from prsm.data.fingerprints.base import FingerprintKind
        resolver = ThresholdResolver.from_default_path()
        uploader = self._seed_binary_uploader(
            kind=FingerprintKind.IMAGE_PHASH,
            similarity=0.75,  # would-be disputed band
            threshold_resolver=resolver,
            arbitration_queue=None,
        )
        result = asyncio.run(uploader.upload(b"binary content"))
        assert result is not None
        # No auto-parent (sim < derivative) and no queue to enqueue to.
        assert "cid-bin-old" not in (result.parent_cids or [])


# ──────────────────────────────────────────────────────────────────────
# T6.5.gov — ARBITRATION_DISPUTE proposal-sink hook
# ──────────────────────────────────────────────────────────────────────


class _RecordingSink:
    """Test sink that captures every (record, record_id) it sees and
    returns a synthetic proposal_id."""

    def __init__(self, proposal_id="prop-test-1"):
        self.calls: list[tuple[DisputedAttributionRecord, str]] = []
        self._proposal_id = proposal_id

    async def create_arbitration_proposal(self, record, record_id):
        self.calls.append((record, record_id))
        return self._proposal_id


class _NullReturningSink:
    """Test sink that returns None — the link step must skip cleanly
    rather than calling set_proposal_id with None."""

    def __init__(self):
        self.calls = 0

    async def create_arbitration_proposal(self, record, record_id):
        self.calls += 1
        return None


class _RaisingSink:
    """Test sink whose create_arbitration_proposal always raises.
    Used to verify the upload still completes."""

    async def create_arbitration_proposal(self, record, record_id):
        raise RuntimeError("backend down")


class TestProposalSinkConstructor:
    def test_default_sink_is_none(self):
        uploader = _make_uploader()
        assert uploader._arbitration_proposal_sink is None

    def test_null_sink_storable(self):
        sink = NullArbitrationProposalSink()
        uploader = _make_uploader(arbitration_proposal_sink=sink)
        assert uploader._arbitration_proposal_sink is sink


class TestEnqueueWithSink:
    """Direct tests for ``_enqueue_arbitration`` with sinks of varying
    behavior — covers the happy path + every failure mode."""

    def _pending(self):
        return {
            "candidate_parent_cid": "cid-old",
            "candidate_parent_creator": "0xold",
            "similarity": 0.78,
            "fingerprint_kind": "text-vector",
        }

    def test_sink_called_and_proposal_id_linked(self):
        queue = InMemoryArbitrationQueue()
        sink = _RecordingSink(proposal_id="prop-42")
        uploader = _make_uploader(
            arbitration_queue=queue,
            arbitration_proposal_sink=sink,
        )
        uploader.creator_address = "0x" + "11" * 20
        asyncio.run(uploader._enqueue_arbitration(
            "cid-new", self._pending(),
        ))
        # Sink saw exactly one record.
        assert len(sink.calls) == 1
        record, record_id = sink.calls[0]
        assert record.new_cid == "cid-new"
        # Queue's record now carries the proposal_id link.
        recovered = asyncio.run(queue.get(record_id))
        assert recovered is not None
        assert recovered.proposal_id == "prop-42"

    def test_no_sink_skips_proposal_step(self):
        queue = InMemoryArbitrationQueue()
        uploader = _make_uploader(
            arbitration_queue=queue,
            arbitration_proposal_sink=None,
        )
        uploader.creator_address = "0x" + "11" * 20
        asyncio.run(uploader._enqueue_arbitration(
            "cid-new", self._pending(),
        ))
        records = asyncio.run(queue.list_pending())
        assert len(records) == 1
        # No sink → no proposal_id link.
        assert records[0].proposal_id is None

    def test_sink_returning_none_does_not_link(self):
        queue = InMemoryArbitrationQueue()
        sink = _NullReturningSink()
        uploader = _make_uploader(
            arbitration_queue=queue,
            arbitration_proposal_sink=sink,
        )
        uploader.creator_address = "0x" + "11" * 20
        asyncio.run(uploader._enqueue_arbitration(
            "cid-new", self._pending(),
        ))
        assert sink.calls == 1
        records = asyncio.run(queue.list_pending())
        assert len(records) == 1
        assert records[0].proposal_id is None

    def test_sink_raising_does_not_break_upload(self):
        # Sink raises but the record is still queued.
        queue = InMemoryArbitrationQueue()
        sink = _RaisingSink()
        uploader = _make_uploader(
            arbitration_queue=queue,
            arbitration_proposal_sink=sink,
        )
        uploader.creator_address = "0x" + "11" * 20
        # Must not raise.
        asyncio.run(uploader._enqueue_arbitration(
            "cid-new", self._pending(),
        ))
        records = asyncio.run(queue.list_pending())
        assert len(records) == 1
        assert records[0].proposal_id is None

    def test_set_proposal_id_failure_swallowed(self):
        # The link step is also wrapped — a queue that raises on
        # set_proposal_id must NOT crash the upload.
        sink = _RecordingSink(proposal_id="prop-99")
        bad_queue = MagicMock()
        bad_queue.enqueue = AsyncMock(return_value="rid-1")
        bad_queue.set_proposal_id = AsyncMock(
            side_effect=RuntimeError("link-storage-down"),
        )
        uploader = _make_uploader(
            arbitration_queue=bad_queue,
            arbitration_proposal_sink=sink,
        )
        uploader.creator_address = "0x" + "11" * 20
        # Must not raise.
        asyncio.run(uploader._enqueue_arbitration(
            "cid-new", self._pending(),
        ))
        # Sink was still called — just couldn't link back.
        assert len(sink.calls) == 1


class TestUploadWithProposalSink:
    """End-to-end through ``upload()`` with a real
    InMemoryArbitrationQueue + recording sink. Verifies the
    disputed-band branch surfaces the proposal correctly when both
    queue and sink are wired."""

    def test_disputed_band_creates_proposal_and_links_to_record(self):
        from prsm.data.dedup.thresholds import ThresholdResolver
        resolver = ThresholdResolver.from_default_path()
        queue = InMemoryArbitrationQueue()
        sink = _RecordingSink(proposal_id="prop-from-upload")

        uploader = _make_uploader(
            threshold_resolver=resolver,
            arbitration_queue=queue,
            arbitration_proposal_sink=sink,
        )
        uploader.creator_address = "0x" + "11" * 20

        async def _embedding_fn(content):
            return np.array([1.0, 0.0], dtype=np.float32)
        uploader._get_embedding = _embedding_fn
        uploader._semantic_index.find_nearest = (
            lambda emb: ("cid-old", 0.86, "0xoldcreator")  # disputed band
        )
        async def _publish_stub(content, filename, ph):
            return "cid-new-uploaded"
        uploader._publish_content = _publish_stub
        uploader._register_local_embedding = MagicMock()
        uploader._register_local_content = MagicMock()
        uploader._broadcast_provenance = AsyncMock()

        result = asyncio.run(uploader.upload(b"some content"))
        assert result is not None
        # Sink received the record (with the post-publish CID).
        assert len(sink.calls) == 1
        record, record_id = sink.calls[0]
        assert record.new_cid == "cid-new-uploaded"
        # The record in the queue is linked back to the proposal.
        linked = asyncio.run(queue.get(record_id))
        assert linked is not None
        assert linked.proposal_id == "prop-from-upload"
