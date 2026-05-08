"""PRSM-PROV-1 T6.5.gov.next — TokenWeightedVotingProposalSink adapter.

Wraps the heavyweight ``TokenWeightedVoting.create_proposal`` API
behind the ``ArbitrationProposalSink`` Protocol so ContentUploader
can stay dedup-module-pure.

Tests use a stub voting backend (just records calls) — the actual
TokenWeightedVoting integration is exercised by the standalone
governance test suite. We verify here:
  - GovernanceProposal shape (title / description / type / metadata)
  - Proposer-id threading
  - Failure mapping (backend raise → sink returns None)
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from prsm.data.dedup.arbitration import (
    ArbitrationProposalSink,
    DisputedAttributionRecord,
    render_arbitration_body,
)
from prsm.economy.governance.arbitration_sink import (
    TokenWeightedVotingProposalSink,
)
from prsm.economy.governance.voting import ProposalCategory


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


def _record(
    *,
    new_cid: str = "cid-uploader",
    candidate: str = "cid-claimed-parent",
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


class _RecordingVotingBackend:
    """Stub for TokenWeightedVoting.create_proposal. Records every
    call and returns a synthetic UUID (configurable per-test)."""

    def __init__(self, return_uuid: UUID | None = None):
        self.calls = []
        self._return_uuid = return_uuid or uuid4()

    async def create_proposal(self, proposer_id, proposal):
        self.calls.append((proposer_id, proposal))
        return self._return_uuid


class _RaisingVotingBackend:
    """Stub backend that raises on create_proposal."""

    def __init__(self, exc):
        self._exc = exc

    async def create_proposal(self, proposer_id, proposal):
        raise self._exc


# ──────────────────────────────────────────────────────────────────────
# Protocol satisfaction + constructor validation
# ──────────────────────────────────────────────────────────────────────


class TestProtocolAndConstruction:
    def test_satisfies_arbitration_proposal_sink_protocol(self):
        sink = TokenWeightedVotingProposalSink(
            voting=_RecordingVotingBackend(),
            proposer_id="0xfoundation",
        )
        assert isinstance(sink, ArbitrationProposalSink)

    def test_empty_proposer_id_rejected(self):
        with pytest.raises(ValueError, match="proposer_id"):
            TokenWeightedVotingProposalSink(
                voting=_RecordingVotingBackend(),
                proposer_id="",
            )

    def test_voting_backend_required(self):
        with pytest.raises(ValueError, match="voting"):
            TokenWeightedVotingProposalSink(
                voting=None,
                proposer_id="0xfoundation",
            )


# ──────────────────────────────────────────────────────────────────────
# Happy path — proposal shape
# ──────────────────────────────────────────────────────────────────────


class TestProposalShape:
    def _call(
        self, *, voting=None, proposer_id="0xfoundation",
    ):
        backend = voting or _RecordingVotingBackend()
        sink = TokenWeightedVotingProposalSink(
            voting=backend,
            proposer_id=proposer_id,
        )
        rec = _record()
        result = asyncio.run(
            sink.create_arbitration_proposal(rec, "rid-abc"),
        )
        return backend, rec, result

    def test_returns_str_uuid_on_success(self):
        target_uuid = uuid4()
        backend, _, result = self._call(
            voting=_RecordingVotingBackend(return_uuid=target_uuid),
        )
        assert result == str(target_uuid)
        assert len(backend.calls) == 1

    def test_proposer_id_threaded_to_backend(self):
        backend, _, _ = self._call(proposer_id="0xfoundation-safe")
        proposer_id, _ = backend.calls[0]
        assert proposer_id == "0xfoundation-safe"

    def test_proposal_proposer_id_set_on_object(self):
        # TokenWeightedVoting.create_proposal also reads
        # proposal.proposer_id at line 224 — must match the kwarg.
        backend, _, _ = self._call(proposer_id="0xfoundation-safe")
        _, proposal = backend.calls[0]
        assert proposal.proposer_id == "0xfoundation-safe"

    def test_proposal_type_is_arbitration_dispute(self):
        backend, _, _ = self._call()
        _, proposal = backend.calls[0]
        # Matches ProposalCategory.ARBITRATION_DISPUTE.value so the
        # backend can route by enum if it wants.
        assert proposal.proposal_type == ProposalCategory.ARBITRATION_DISPUTE.value

    def test_title_includes_truncated_cids(self):
        backend, _, _ = self._call()
        _, proposal = backend.calls[0]
        # Truncate to 12 chars per design doc §"Phase 6 governance hook"
        # so titles fit in council-UI list views.
        assert "cid-uploader" in proposal.title
        assert "cid-claimed-" in proposal.title

    def test_description_is_rendered_body(self):
        # Pin: description == render_arbitration_body(record). Future
        # on-chain arbitration contract may verify body bytes.
        backend, rec, _ = self._call()
        _, proposal = backend.calls[0]
        assert proposal.description == render_arbitration_body(rec)

    def test_metadata_carries_arbitration_record_id(self):
        # Critical link: the proposal must carry the arbitration
        # record_id so a council vote can map back to the queued
        # record (and downstream resolve() can fire when the
        # proposal closes).
        backend, _, _ = self._call()
        _, proposal = backend.calls[0]
        assert proposal.metadata.get("arbitration_record_id") == "rid-abc"

    def test_metadata_carries_kind_and_similarity(self):
        backend, rec, _ = self._call()
        _, proposal = backend.calls[0]
        assert proposal.metadata.get("fingerprint_kind") == rec.fingerprint_kind
        assert proposal.metadata.get("similarity") == pytest.approx(
            rec.similarity,
        )


# ──────────────────────────────────────────────────────────────────────
# Failure mapping — Protocol contract is "never raise; return None"
# ──────────────────────────────────────────────────────────────────────


class TestFailureMapping:
    def _sink(self, exc):
        return TokenWeightedVotingProposalSink(
            voting=_RaisingVotingBackend(exc),
            proposer_id="0xfoundation",
        )

    def test_value_error_returns_none(self):
        # Insufficient FTNS / ineligible proposer raises ValueError
        # in the real backend. Per Protocol, sink swallows + returns
        # None so the upload completes without a linked proposal.
        sink = self._sink(ValueError("Insufficient FTNS balance"))
        result = asyncio.run(sink.create_arbitration_proposal(
            _record(), "rid-x",
        ))
        assert result is None

    def test_runtime_error_returns_none(self):
        sink = self._sink(RuntimeError("backend unavailable"))
        result = asyncio.run(sink.create_arbitration_proposal(
            _record(), "rid-x",
        ))
        assert result is None

    def test_unexpected_exception_returns_none(self):
        # Even un-typed exceptions must not bubble out — the sink's
        # contract is absolute.
        class _Weird(Exception):
            pass
        sink = self._sink(_Weird("???"))
        result = asyncio.run(sink.create_arbitration_proposal(
            _record(), "rid-x",
        ))
        assert result is None
