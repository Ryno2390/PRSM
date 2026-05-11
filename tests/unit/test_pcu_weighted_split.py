"""Sprint 238 — PCU-weighted compute participant split.

Pre-fix: api.py:2055-2060 distributes the prompter's compute
budget UNIFORMLY across all swarm participants regardless of
how much actual compute each did. The Vision flagged this as
deferred from §4 step 6 closure: "PCU-weighted compute split
(uniform for v1)".

This sprint:
  1. Adds `pcu_consumed: float = 0.0` to PartialResult +
     ParticipantAttribution so per-shard PCU flows from agent
     dispatch through the aggregator to settlement.
  2. Extracts the split-decision into `compute_split_amounts()`
     in a new prsm/economy/split_compute.py module.
  3. Strategy: PCU-weighted when ALL participants have
     pcu_consumed > 0; uniform fallback otherwise (preserves
     v1 behavior for legacy callers that don't set PCU yet).
"""
from __future__ import annotations

import pytest

from prsm.compute.query_orchestrator.swarm_runner import (
    ParticipantAttribution,
    PartialResult,
)
from prsm.economy.split_compute import compute_split_amounts


# ── Schema: PCU field now part of carrier types ─────────────


def test_partial_result_carries_pcu():
    p = PartialResult(
        shard_cid="cid-1",
        payload=b"x",
        agent_signature=b"\x00" * 64,
        creator_id="c-1",
        dp_noise_applied=True,
        pcu_consumed=42.0,
    )
    assert p.pcu_consumed == 42.0


def test_partial_result_pcu_defaults_to_zero():
    p = PartialResult(
        shard_cid="cid-1",
        payload=b"x",
        agent_signature=b"\x00" * 64,
        creator_id="c-1",
        dp_noise_applied=True,
    )
    assert p.pcu_consumed == 0.0


def test_participant_attribution_carries_pcu():
    pa = ParticipantAttribution(
        shard_cid="cid-1",
        source_agent_pubkey=b"\x00" * 32,
        creator_id="c-1",
        pcu_consumed=7.5,
    )
    assert pa.pcu_consumed == 7.5


def test_participant_attribution_pcu_defaults_to_zero():
    pa = ParticipantAttribution(
        shard_cid="cid-1",
        source_agent_pubkey=b"\x00" * 32,
        creator_id="c-1",
    )
    assert pa.pcu_consumed == 0.0


# ── Split-decision logic ────────────────────────────────────


def _mk_participant(pubkey_hex: str, pcu: float = 0.0):
    return {
        "source_agent_pubkey_hex": pubkey_hex,
        "pcu_consumed": pcu,
    }


def test_uniform_split_when_all_pcu_zero():
    participants = [
        _mk_participant("a"),
        _mk_participant("b"),
        _mk_participant("c"),
    ]
    splits, mode = compute_split_amounts(
        participants=participants,
        aggregator_node_id="agg",
        total_budget=100.0,
        aggregator_share_bps=500,  # 5%
    )
    assert mode == "uniform"
    # Aggregator gets 5 FTNS, 95 split uniformly → ~31.667 each.
    assert splits[0] == ("agg", 5.0)
    for recipient, amount in splits[1:]:
        assert amount == pytest.approx(95.0 / 3, rel=1e-9)
    assert {r for r, _ in splits[1:]} == {"a", "b", "c"}


def test_pcu_weighted_split_when_all_nonzero():
    participants = [
        _mk_participant("a", pcu=10.0),
        _mk_participant("b", pcu=20.0),
        _mk_participant("c", pcu=70.0),
    ]
    splits, mode = compute_split_amounts(
        participants=participants,
        aggregator_node_id="agg",
        total_budget=100.0,
        aggregator_share_bps=500,
    )
    assert mode == "pcu_weighted"
    # Aggregator 5, compute total 95. PCU shares: 10/100=10%,
    # 20/100=20%, 70/100=70% of compute_total.
    amounts = dict(splits)
    assert amounts["agg"] == 5.0
    assert amounts["a"] == pytest.approx(95.0 * 0.10, rel=1e-9)
    assert amounts["b"] == pytest.approx(95.0 * 0.20, rel=1e-9)
    assert amounts["c"] == pytest.approx(95.0 * 0.70, rel=1e-9)


def test_uniform_fallback_when_any_pcu_zero():
    """Mixed PCU values fall back to uniform (sprint 238 v1
    contract). Treating zero-PCU as zero share would punish
    legacy callers mid-migration."""
    participants = [
        _mk_participant("a", pcu=10.0),
        _mk_participant("b", pcu=0.0),
        _mk_participant("c", pcu=5.0),
    ]
    splits, mode = compute_split_amounts(
        participants=participants,
        aggregator_node_id="agg",
        total_budget=100.0,
        aggregator_share_bps=500,
    )
    assert mode == "uniform"
    for recipient, amount in splits[1:]:
        assert amount == pytest.approx(95.0 / 3, rel=1e-9)


def test_empty_participants_returns_aggregator_only():
    """Legacy path: 0 participants = single aggregator entry
    (caller still has to decide whether to use this or fall
    back to release_escrow non-split)."""
    splits, mode = compute_split_amounts(
        participants=[],
        aggregator_node_id="agg",
        total_budget=100.0,
        aggregator_share_bps=500,
    )
    assert splits == [("agg", 5.0)]
    assert mode == "uniform"


def test_zero_total_budget_rejected():
    with pytest.raises(ValueError):
        compute_split_amounts(
            participants=[_mk_participant("a")],
            aggregator_node_id="agg",
            total_budget=0.0,
            aggregator_share_bps=500,
        )


def test_aggregator_bps_out_of_range_rejected():
    with pytest.raises(ValueError):
        compute_split_amounts(
            participants=[_mk_participant("a")],
            aggregator_node_id="agg",
            total_budget=100.0,
            aggregator_share_bps=10001,
        )


def test_aggregator_zero_bps_all_to_compute():
    """0% aggregator share — full budget to compute participants."""
    splits, mode = compute_split_amounts(
        participants=[_mk_participant("a"), _mk_participant("b")],
        aggregator_node_id="agg",
        total_budget=100.0,
        aggregator_share_bps=0,
    )
    # Aggregator entry omitted when share is zero (no zero-amount
    # transactions in the split list).
    amounts = dict(splits)
    assert "agg" not in amounts
    assert amounts["a"] == 50.0
    assert amounts["b"] == 50.0
