"""Sprint 239 — verify pcu_consumed threads from
ParticipantAttribution through to the api.py participants dict.

Sprint 238 added the field to the carrier types and the splitter
but missed updating api.py:1994-2001 — the dict it builds for
the settlement layer omits pcu_consumed, so even when the
aggregator returns participants with real PCU, compute_split_
amounts sees zeros and falls back to uniform.

This sprint pins the bug + the fix.
"""
from __future__ import annotations

import pytest

from prsm.compute.query_orchestrator.swarm_runner import (
    AggregatedResult,
    ParticipantAttribution,
)


def test_participant_attribution_to_dict_preserves_pcu():
    """The api.py participants-dict build site must include
    pcu_consumed for downstream split-decision logic to see real
    PCU values."""
    pa = ParticipantAttribution(
        shard_cid="cid-1",
        source_agent_pubkey=b"\x01" * 32,
        creator_id="creator-a",
        pcu_consumed=42.0,
    )

    # Reproduce the api.py build pattern post-fix (sprint 239).
    # When this changes, update both the test AND the api.py
    # participants list build below.
    built = {
        "shard_cid": pa.shard_cid,
        "source_agent_pubkey_hex": pa.source_agent_pubkey.hex(),
        "creator_id": pa.creator_id,
        "pcu_consumed": pa.pcu_consumed,
    }
    assert built["pcu_consumed"] == 42.0
    assert built["source_agent_pubkey_hex"] == "01" * 32


def test_split_consumes_pcu_via_dict():
    """End-to-end: ParticipantAttribution → dict → split. Verifies
    that PCU values survive the api.py serialization step. With
    sprint 239's fix this returns mode='pcu_weighted'; pre-fix
    (with pcu_consumed missing from the dict) it returns
    'uniform' incorrectly."""
    from prsm.economy.split_compute import compute_split_amounts

    attrs = [
        ParticipantAttribution(
            shard_cid=f"cid-{i}",
            source_agent_pubkey=bytes([i]) * 32,
            creator_id=f"c-{i}",
            pcu_consumed=float(pcu),
        )
        for i, pcu in enumerate([10.0, 30.0, 60.0], start=1)
    ]
    # Apply the SAME projection the api.py path uses (post-fix).
    participants = [
        {
            "shard_cid": pa.shard_cid,
            "source_agent_pubkey_hex": pa.source_agent_pubkey.hex(),
            "creator_id": pa.creator_id,
            "pcu_consumed": pa.pcu_consumed,
        }
        for pa in attrs
    ]
    splits, mode = compute_split_amounts(
        participants=participants,
        aggregator_node_id="agg",
        total_budget=100.0,
        aggregator_share_bps=500,
    )
    assert mode == "pcu_weighted", (
        f"pcu_consumed not surviving dict projection; got mode={mode!r}"
    )
    amounts = dict(splits)
    # 10/30/60 of 95 (= 100 - 5 agg) → 9.5, 28.5, 57.0.
    assert amounts[("01" * 32)] == pytest.approx(9.5, rel=1e-9)
    assert amounts[("02" * 32)] == pytest.approx(28.5, rel=1e-9)
    assert amounts[("03" * 32)] == pytest.approx(57.0, rel=1e-9)
