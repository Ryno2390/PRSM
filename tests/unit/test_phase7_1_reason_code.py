"""Unit tests for Phase 7.1 Task 4 — Python-side reason code wiring.

Two concerns:
  1. Module constants in stake_manager.py match the on-chain ReasonCode
     enum and are structured so future Python code (orchestrator
     challenge pipeline, Slashed-event decoders) can reference them.
  2. ReputationTracker.record_slash flows CONSENSUS_MISMATCH identically
     to DOUBLE_SPEND / INVALID_SIGNATURE — no new code path needed on the
     reputation side per design §6 Task 4.
"""
from __future__ import annotations

import pytest

from prsm.economy.web3.stake_manager import (
    REASON_CONSENSUS_MISMATCH_KECCAK,
    REASON_DOUBLE_SPEND_KECCAK,
    REASON_INVALID_SIGNATURE_KECCAK,
    ReasonCode,
    SLASHABLE_REASONS,
    _reason_keccak,
)
from prsm.marketplace.reputation import ReputationTracker


# ── ReasonCode enum values ───────────────────────────────────────────


def test_reason_code_values_match_contract():
    """Must mirror BatchSettlementRegistry.sol:ReasonCode exactly.
    Enum ordering is stable per Phase 3.1 design §7; new codes append."""
    assert ReasonCode.DOUBLE_SPEND == 0
    assert ReasonCode.INVALID_SIGNATURE == 1
    assert ReasonCode.NO_ESCROW == 2
    assert ReasonCode.EXPIRED == 3
    assert ReasonCode.MALFORMED == 4
    assert ReasonCode.CONSENSUS_MISMATCH == 5


def test_reason_code_is_int_enum():
    """Challenge tx builders pass `reason=ReasonCode.X` directly to
    contracts.functions.challengeReceipt; the web3.py ABI encoder
    expects a uint8, which IntEnum satisfies."""
    assert int(ReasonCode.CONSENSUS_MISMATCH) == 5


# ── Slashable-reasons set ────────────────────────────────────────────


def test_slashable_reasons_matches_contract_hook():
    """Contract's challengeReceipt slash hook fires only on these three
    reasons. Callers can use SLASHABLE_REASONS to decide whether to
    expect a Slashed event on a challenge receipt."""
    assert SLASHABLE_REASONS == {
        ReasonCode.DOUBLE_SPEND,
        ReasonCode.INVALID_SIGNATURE,
        ReasonCode.CONSENSUS_MISMATCH,
    }


def test_no_escrow_and_expired_not_in_slashable_set():
    """NO_ESCROW is requester-attestation (griefing risk if slash);
    EXPIRED is protocol hygiene not malice — neither triggers slash."""
    assert ReasonCode.NO_ESCROW not in SLASHABLE_REASONS
    assert ReasonCode.EXPIRED not in SLASHABLE_REASONS


# ── Keccak identifiers ───────────────────────────────────────────────


def test_keccak_constants_are_32_bytes():
    for k in (
        REASON_DOUBLE_SPEND_KECCAK,
        REASON_INVALID_SIGNATURE_KECCAK,
        REASON_CONSENSUS_MISMATCH_KECCAK,
    ):
        assert isinstance(k, bytes)
        assert len(k) == 32


def test_keccak_constants_are_distinct():
    """Each reason's keccak identifier must be unique — otherwise
    observability tools that tag events by keccak(reason_name) would
    conflate different reason codes."""
    keccaks = {
        REASON_DOUBLE_SPEND_KECCAK,
        REASON_INVALID_SIGNATURE_KECCAK,
        REASON_CONSENSUS_MISMATCH_KECCAK,
    }
    assert len(keccaks) == 3


def test_reason_keccak_helper_matches_constants():
    """The public helper produces the same bytes as the precomputed
    module constants. Lets callers rehydrate a keccak for any ReasonCode
    or even an unknown-string reason code."""
    assert _reason_keccak(ReasonCode.CONSENSUS_MISMATCH) == REASON_CONSENSUS_MISMATCH_KECCAK
    assert _reason_keccak(ReasonCode.DOUBLE_SPEND) == REASON_DOUBLE_SPEND_KECCAK


def test_reason_keccak_accepts_raw_string():
    """Future-proofing: if a new ReasonCode lands on-chain before the
    Python enum is updated, callers can still derive a keccak from the
    raw string."""
    future = _reason_keccak("FUTURE_REASON_CODE_NAME")
    assert len(future) == 32
    assert future not in {
        REASON_DOUBLE_SPEND_KECCAK,
        REASON_INVALID_SIGNATURE_KECCAK,
        REASON_CONSENSUS_MISMATCH_KECCAK,
    }


# ── ReputationTracker flow-through ───────────────────────────────────


def test_reputation_tracker_accepts_consensus_mismatch():
    """No code change on the tracker — unknown reasons are stored
    verbatim per Phase 7 Task 6. Verify the wiring end-to-end."""
    tracker = ReputationTracker()
    tracker.record_slash(
        provider_id="providerC",
        batch_id="0x" + "ab" * 32,
        slash_amount_wei=25_000 * 10**18,
        reason=ReasonCode.CONSENSUS_MISMATCH.name,
        tx_hash="0x" + "1" * 64,
    )
    assert tracker.has_been_slashed("providerC") is True
    events = tracker.get_slash_events("providerC")
    assert len(events) == 1
    assert events[0].reason == "CONSENSUS_MISMATCH"


def test_reputation_tracker_weights_consensus_mismatch_identically():
    """SLASH_WEIGHT behavior must not depend on the reason string.
    A CONSENSUS_MISMATCH slash should crush score_for exactly like a
    DOUBLE_SPEND slash would."""
    t_consensus = ReputationTracker()
    t_double = ReputationTracker()

    for _ in range(20):
        t_consensus.record_success("p", latency_ms=50.0)
        t_double.record_success("p", latency_ms=50.0)

    t_consensus.record_slash(
        "p", batch_id="0xab", slash_amount_wei=1000,
        reason="CONSENSUS_MISMATCH",
    )
    t_double.record_slash(
        "p", batch_id="0xab", slash_amount_wei=1000,
        reason="DOUBLE_SPEND",
    )

    assert t_consensus.score_for("p") == t_double.score_for("p")


def test_reputation_tracker_separates_consensus_from_double_spend_in_audit():
    """Reasons must be stored verbatim so audit tools can distinguish
    the two even though they weigh the same in score_for."""
    tracker = ReputationTracker()
    tracker.record_slash(
        "p", batch_id="0x1", slash_amount_wei=1, reason="DOUBLE_SPEND",
    )
    tracker.record_slash(
        "p", batch_id="0x2", slash_amount_wei=1, reason="CONSENSUS_MISMATCH",
    )
    events = tracker.get_slash_events("p")
    reasons = [e.reason for e in events]
    assert reasons == ["DOUBLE_SPEND", "CONSENSUS_MISMATCH"]
