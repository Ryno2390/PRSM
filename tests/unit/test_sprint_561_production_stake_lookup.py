"""Sprint 561 — production stake_lookup: real StakeManagerClient via anchor.

Sprint 560 wired the real anchor but left stake_lookup as a
ZeroStakeLookup placeholder. Sprint 561 replaces it with a real
on-chain stake query backed by the StakeBond contract.

Indirection problem: trust-stack's StakeLookup.get_stake takes a
32-char hex node_id (manifest publisher ID), but the on-chain
StakeBond contract's stakeOf takes an ETH address (provider).
Resolution: AnchorMediatedStakeLookup uses the anchor's
node_id → eth_address mapping (sprint 540's PublisherKeyAnchorClient
.lookup) THEN queries stake_of on the resolved address.

  PRSM_STAKE_BOND_ADDRESS → builds real StakeManagerClient.
  Missing  → fall back to ZeroStakeLookup (sprint 560 behavior)
             with named warning. Operator on partial-production.

Conservative fail-soft: anchor returns None → 0 stake. RPC raise
on stake_of → 0 stake (logged at DEBUG to avoid log spam on
transient network errors).
"""
from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pytest


# ── shared fixtures ──────────────────────────────────────


def _stub_node():
    from prsm.node.identity import generate_node_identity
    n = MagicMock()
    n.identity = generate_node_identity("test-settler")
    return n


def _v1_catalog(tmp_path):
    p = tmp_path / "catalog.json"
    p.write_text(json.dumps({
        "schema_version": "v1",
        "models": {
            "test-model": {
                "model_name": "test-model",
                "mlx_model_name": "test-model",
                "head_size": 64, "hidden_dim": 128,
                "intermediate_dim": 256,
                "num_attention_heads": 4, "num_kv_heads": 4,
                "vocab_size": 1000, "num_layers": 4,
            },
        },
    }))
    return p


def _set_env(
    monkeypatch, tmp_path,
    anchor_addr="0x" + "ab" * 20,
    stake_addr=None,
):
    monkeypatch.setenv(
        "PRSM_PARALLAX_MODEL_CATALOG_FILE",
        str(_v1_catalog(tmp_path)),
    )
    monkeypatch.setenv(
        "PRSM_PARALLAX_TRUST_STACK_KIND", "production",
    )
    monkeypatch.setenv(
        "PRSM_PARALLAX_GPU_POOL_KIND", "static-empty",
    )
    monkeypatch.setenv(
        "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", anchor_addr,
    )
    if stake_addr is None:
        monkeypatch.delenv(
            "PRSM_STAKE_BOND_ADDRESS", raising=False,
        )
    else:
        monkeypatch.setenv("PRSM_STAKE_BOND_ADDRESS", stake_addr)


# ── adapter direct tests ──────────────────────────────────


def test_anchor_mediated_lookup_returns_zero_for_unknown_node():
    """Unknown node_id → anchor returns None → 0 stake."""
    from prsm.node.inference_wiring import (
        AnchorMediatedStakeLookup,
    )
    anchor = MagicMock()
    anchor.lookup = MagicMock(return_value=None)
    stake = MagicMock()
    lookup = AnchorMediatedStakeLookup(anchor=anchor, stake_client=stake)
    assert lookup.get_stake("unknown-node-id") == 0
    # stake_of MUST NOT be called when anchor has no mapping.
    stake.stake_of.assert_not_called()


def test_anchor_mediated_lookup_queries_stake_contract():
    """Known node_id → anchor returns eth → stake_of returns
    StakeRecord with amount_wei → adapter returns amount_wei."""
    from prsm.node.inference_wiring import (
        AnchorMediatedStakeLookup,
    )
    anchor = MagicMock()
    anchor.lookup = MagicMock(return_value="0x" + "cd" * 20)

    fake_record = MagicMock()
    fake_record.amount_wei = 5_000 * 10**18
    stake = MagicMock()
    stake.stake_of = MagicMock(return_value=fake_record)

    lookup = AnchorMediatedStakeLookup(
        anchor=anchor, stake_client=stake,
    )
    assert lookup.get_stake("known-node") == 5_000 * 10**18
    anchor.lookup.assert_called_once_with("known-node")
    stake.stake_of.assert_called_once_with("0x" + "cd" * 20)


def test_anchor_mediated_lookup_swallows_stake_rpc_errors():
    """StakeManagerClient.stake_of raises (RPC outage etc.) → adapter
    returns 0 conservatively, daemon stays alive. Logged at DEBUG."""
    from prsm.node.inference_wiring import (
        AnchorMediatedStakeLookup,
    )
    anchor = MagicMock()
    anchor.lookup = MagicMock(return_value="0x" + "ef" * 20)
    stake = MagicMock()
    stake.stake_of = MagicMock(side_effect=RuntimeError("RPC down"))

    lookup = AnchorMediatedStakeLookup(
        anchor=anchor, stake_client=stake,
    )
    # Must not raise.
    assert lookup.get_stake("any-node") == 0


# ── env-var wiring ────────────────────────────────────────


def test_production_with_stake_address_uses_real_lookup(
    monkeypatch, tmp_path,
):
    """When PRSM_STAKE_BOND_ADDRESS is set alongside the anchor,
    the production trust_stack's profile_source carries a
    sprint-690 PoolBackedStakeLookup (NOT the broken sprint-561
    AnchorMediatedStakeLookup — see F31 docs). The pool-backed
    lookup reads pre-populated stake_amounts from the DHT pool
    snapshot, avoiding the pubkey-vs-ETH-address bug that made
    anchor-mediated lookup return 0 for every peer."""
    from prsm.node.inference_wiring import (
        PoolBackedStakeLookup,
        build_parallax_executor_or_none,
    )

    _set_env(
        monkeypatch, tmp_path,
        anchor_addr="0x" + "ab" * 20,
        stake_addr="0x" + "11" * 20,
    )
    fake_anchor = MagicMock()
    fake_stake = MagicMock()
    with patch(
        "prsm.security.publisher_key_anchor.client."
        "PublisherKeyAnchorClient",
        return_value=fake_anchor,
    ), patch(
        "prsm.economy.web3.stake_manager.StakeManagerClient",
        return_value=fake_stake,
    ):
        result = build_parallax_executor_or_none(_stub_node())
    assert result is not None
    stake_lookup = result._trust.profile_source.stake_lookup
    assert isinstance(stake_lookup, PoolBackedStakeLookup), (
        f"expected sprint-690 PoolBackedStakeLookup; got "
        f"{type(stake_lookup).__name__}"
    )


def test_production_without_stake_address_falls_back_to_placeholder(
    monkeypatch, tmp_path, caplog,
):
    """When PRSM_STAKE_BOND_ADDRESS is unset, the daemon does NOT
    fail the whole production wiring — it falls back to sprint-560's
    ZeroStakeLookup placeholder with a structured warning naming
    the env var. Operator on partial-production gets a working
    daemon; the stake_lookup placeholder log line names the
    missing piece."""
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )
    _set_env(
        monkeypatch, tmp_path,
        anchor_addr="0x" + "ab" * 20,
        stake_addr=None,
    )
    with patch(
        "prsm.security.publisher_key_anchor.client."
        "PublisherKeyAnchorClient",
        return_value=MagicMock(),
    ):
        with caplog.at_level(logging.INFO):
            result = build_parallax_executor_or_none(_stub_node())
    assert result is not None
    log_text = " ".join(r.message for r in caplog.records)
    assert "PRSM_STAKE_BOND_ADDRESS" in log_text, (
        "Missing stake-bond env var must be named in the log so "
        "the operator knows what's still placeholder."
    )


def test_production_log_marks_stake_lookup_as_real_when_wired(
    monkeypatch, tmp_path, caplog,
):
    """When both anchor + stake are wired, the production-kind INFO
    log MUST reflect stake_lookup=REAL (not PLACEHOLDER) so the
    sprint-560 enumeration stays honest."""
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )
    _set_env(
        monkeypatch, tmp_path,
        anchor_addr="0x" + "ab" * 20,
        stake_addr="0x" + "22" * 20,
    )
    with patch(
        "prsm.security.publisher_key_anchor.client."
        "PublisherKeyAnchorClient",
        return_value=MagicMock(),
    ), patch(
        "prsm.economy.web3.stake_manager.StakeManagerClient",
        return_value=MagicMock(),
    ):
        with caplog.at_level(logging.INFO):
            result = build_parallax_executor_or_none(_stub_node())
    assert result is not None
    log_text = " ".join(r.message for r in caplog.records)
    # The string "stake_lookup=REAL" makes it grep-able for
    # monitoring/dashboards.
    assert "stake_lookup=REAL" in log_text


# ── back-compat ──────────────────────────────────────────


def test_mock_kind_unchanged(monkeypatch, tmp_path):
    """Sprint-558 mock kind path is not affected by sprint-561."""
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )
    from prsm.compute.inference.parallax_executor import (
        ParallaxScheduledExecutor,
    )
    monkeypatch.setenv(
        "PRSM_PARALLAX_MODEL_CATALOG_FILE",
        str(_v1_catalog(tmp_path)),
    )
    monkeypatch.setenv("PRSM_PARALLAX_TRUST_STACK_KIND", "mock")
    monkeypatch.setenv(
        "PRSM_PARALLAX_GPU_POOL_KIND", "static-empty",
    )
    result = build_parallax_executor_or_none(_stub_node())
    assert isinstance(result, ParallaxScheduledExecutor)


def test_stake_client_failure_in_pool_does_not_break_trust_stack(
    monkeypatch, tmp_path, caplog,
):
    """Sprint 690 supersedes the sprint-561 stake-client-fallback
    test. The production trust stack no longer constructs a
    StakeManagerClient directly — sprint-690's PoolBackedStakeLookup
    reads stake from the pool provider's already-resolved
    ParallaxGPU.stake_amount field. StakeManagerClient construction
    happens lazily inside sprint-683's OnChainStakeReader (lazy-
    constructed by the pool provider) with its own fail-soft path
    returning 0 stake under RPC errors.

    What this test asserts now: even with the chain RPC dead, the
    production trust stack constructs successfully (no crash; the
    pool-backed lookup returns 0 stake per peer, advisory mode
    can bridge to live-attest)."""
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
        PoolBackedStakeLookup,
    )
    _set_env(
        monkeypatch, tmp_path,
        anchor_addr="0x" + "ab" * 20,
        stake_addr="0x" + "33" * 20,
    )
    with patch(
        "prsm.security.publisher_key_anchor.client."
        "PublisherKeyAnchorClient",
        return_value=MagicMock(),
    ), patch(
        "prsm.economy.web3.stake_manager.StakeManagerClient",
        side_effect=RuntimeError("rpc dead"),
    ):
        result = build_parallax_executor_or_none(_stub_node())
    assert result is not None
    assert isinstance(
        result._trust.profile_source.stake_lookup,
        PoolBackedStakeLookup,
    )
