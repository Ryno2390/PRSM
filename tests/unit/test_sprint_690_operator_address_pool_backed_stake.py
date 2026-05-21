"""Sprint 690 F31 proper fix — close advisory-mode stake bypass.

Sprint 686 shipped PRSM_PARALLAX_STAKE_ELIGIBILITY=advisory as a
documented bypass because AnchorMediatedStakeLookup was buggy:
it passes the anchor's base64 pubkey to StakeManagerClient
.stake_of() which expects an ETH address — guaranteed 0 stake
for every peer, blocking the entire pool.

Sprint 690 closes the gap properly without a new on-chain
mapping (which would require redeploying the PublisherKeyAnchor
contract). Two pieces:

  Piece 1 — local hardware profile carries operator_address.
    load_local_hardware_profile() reads PRSM_OPERATOR_ADDRESS
    env var (the EOA the operator staked from) and merges it
    into the loaded profile dict. Optional — absent peers
    keep current behavior (stake_amount=0 in their
    ParallaxGPU, get filtered under enforced mode).

  Piece 2 — PoolBackedStakeLookup.
    Replaces AnchorMediatedStakeLookup in production trust
    stack. Consults the LATEST pool snapshot for the node's
    stake_amount (which sprint 683's OnChainStakeReader
    populated via operator_address → stake_of). No anchor
    indirection, no pubkey-vs-address misuse.

After 690 ships, operators advertising operator_address + having
posted real on-chain stake can run with
PRSM_PARALLAX_STAKE_ELIGIBILITY=enforced — closing the advisory
bypass for production deployment.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ── Piece 1 — hardware profile loader extension ──────────────


def test_loader_merges_operator_address_when_env_set(
    tmp_path, monkeypatch,
):
    """PRSM_OPERATOR_ADDRESS in env → field present in loaded
    profile dict alongside the hardware specs."""
    monkeypatch.setenv(
        "PRSM_OPERATOR_ADDRESS",
        "0xF7d88c943B048dAd2e5178E40DaaD545dB3311c2",
    )
    monkeypatch.delenv("PRSM_HARDWARE_PROFILE_FILE", raising=False)
    cached = {"tflops_fp16": 4.6, "ram_total_gb": 16.0}
    (tmp_path / "hardware_profile.json").write_text(json.dumps(cached))
    from prsm.node.hardware_profile_loader import (
        load_local_hardware_profile,
    )
    result = load_local_hardware_profile(cache_dir=tmp_path)
    assert result is not None
    assert result["operator_address"] == (
        "0xF7d88c943B048dAd2e5178E40DaaD545dB3311c2"
    )
    # Original hardware fields preserved
    assert result["tflops_fp16"] == 4.6


def test_loader_omits_operator_address_when_env_unset(
    tmp_path, monkeypatch,
):
    """No env → no field. Defends against accidentally
    advertising a malformed/sentinel address."""
    monkeypatch.delenv("PRSM_OPERATOR_ADDRESS", raising=False)
    monkeypatch.delenv("PRSM_HARDWARE_PROFILE_FILE", raising=False)
    cached = {"tflops_fp16": 4.6, "ram_total_gb": 16.0}
    (tmp_path / "hardware_profile.json").write_text(json.dumps(cached))
    from prsm.node.hardware_profile_loader import (
        load_local_hardware_profile,
    )
    result = load_local_hardware_profile(cache_dir=tmp_path)
    assert "operator_address" not in result


def test_loader_skips_invalid_operator_address(tmp_path, monkeypatch):
    """Address must be 0x-prefixed 42-char hex (Ethereum form).
    Garbage → warn + skip field, don't crash."""
    monkeypatch.setenv("PRSM_OPERATOR_ADDRESS", "not-an-address")
    monkeypatch.delenv("PRSM_HARDWARE_PROFILE_FILE", raising=False)
    cached = {"tflops_fp16": 4.6, "ram_total_gb": 16.0}
    (tmp_path / "hardware_profile.json").write_text(json.dumps(cached))
    from prsm.node.hardware_profile_loader import (
        load_local_hardware_profile,
    )
    result = load_local_hardware_profile(cache_dir=tmp_path)
    assert "operator_address" not in result


# ── Piece 2 — PoolBackedStakeLookup ─────────────────────────


def test_pool_backed_stake_lookup_reads_from_latest_pool():
    """get_stake(node_id) returns the stake_amount from the
    matching GPU in the latest pool snapshot."""
    from prsm.compute.parallax_scheduling.prsm_types import (
        ParallaxGPU, TIER_ATTESTATION_NONE,
    )
    from prsm.node.inference_wiring import PoolBackedStakeLookup

    gpu_a = ParallaxGPU(
        node_id="a" * 32, region="default", layer_capacity=1,
        stake_amount=5_000_000_000_000_000_000,  # 5 FTNS
        tier_attestation=TIER_ATTESTATION_NONE,
        tflops_fp16=1.0, memory_gb=8.0,
        memory_bandwidth_gbps=50.0,
    )
    gpu_b = ParallaxGPU(
        node_id="b" * 32, region="default", layer_capacity=1,
        stake_amount=0,
        tier_attestation=TIER_ATTESTATION_NONE,
        tflops_fp16=1.0, memory_gb=8.0,
        memory_bandwidth_gbps=50.0,
    )
    provider = lambda: [gpu_a, gpu_b]
    lookup = PoolBackedStakeLookup(pool_provider=provider)
    assert lookup.get_stake("a" * 32) == 5_000_000_000_000_000_000
    assert lookup.get_stake("b" * 32) == 0


def test_pool_backed_stake_lookup_returns_zero_for_unknown_node():
    """Node not in pool → 0 (conservative — same posture as
    sprint-561 AnchorMediatedStakeLookup's failure fallback)."""
    from prsm.node.inference_wiring import PoolBackedStakeLookup
    provider = lambda: []
    lookup = PoolBackedStakeLookup(pool_provider=provider)
    assert lookup.get_stake("a" * 32) == 0


def test_pool_backed_stake_lookup_handles_provider_exception():
    """Provider raises → return 0, never propagate. Defends
    against transient provider failures crashing the trust
    stack's pre-route filter."""
    from prsm.node.inference_wiring import PoolBackedStakeLookup
    def _bad():
        raise RuntimeError("provider blew up")
    lookup = PoolBackedStakeLookup(pool_provider=_bad)
    assert lookup.get_stake("a" * 32) == 0


def test_pool_backed_stake_lookup_reads_fresh_each_call():
    """Provider is called per get_stake — peers joining/leaving
    show up in the next eligibility check without rebuilding
    the lookup."""
    from prsm.compute.parallax_scheduling.prsm_types import (
        ParallaxGPU, TIER_ATTESTATION_NONE,
    )
    from prsm.node.inference_wiring import PoolBackedStakeLookup
    state = {"calls": 0, "stake": 100}
    def _provider():
        state["calls"] += 1
        return [ParallaxGPU(
            node_id="a" * 32, region="default", layer_capacity=1,
            stake_amount=state["stake"],
            tier_attestation=TIER_ATTESTATION_NONE,
            tflops_fp16=1.0, memory_gb=8.0,
            memory_bandwidth_gbps=50.0,
        )]
    lookup = PoolBackedStakeLookup(pool_provider=_provider)
    assert lookup.get_stake("a" * 32) == 100
    state["stake"] = 200  # peer re-stakes
    assert lookup.get_stake("a" * 32) == 200
    assert state["calls"] == 2
