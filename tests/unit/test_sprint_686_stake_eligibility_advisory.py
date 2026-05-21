"""Sprint 686 — PRSM_PARALLAX_STAKE_ELIGIBILITY=advisory bypass.

Live-attest of sprint 685's DHT-backed pool surfaced two real
production bugs:

  F31 — AnchorMediatedStakeLookup.get_stake passes the base64
        pubkey (from anchor.lookup) to StakeManagerClient.stake_of
        which expects an Ethereum address. The PublisherKeyAnchor
        contract stores publicKey + registeredAt; it has NO node_id
        → operator-address mapping. Every peer gets stake=0 →
        is_eligible() returns False → filter_pool returns [].

  F32 — Misleading error: when stake-eligibility rejects all GPUs,
        the error reads "no GPU passed anchor verification" — but
        the real cause is stake eligibility (or the F31 bug above).
        Operators correlate against AnchorVerifyAdapter and find no
        rejection logs, masking the real issue.

This sprint:
  1. Adds PRSM_PARALLAX_STAKE_ELIGIBILITY env (advisory|enforced;
     default enforced). Advisory mode wraps stake_lookup with a
     _PermitAllStakeLookup at the production trust stack builder.
  2. Fixes the misleading filter_pool error string.
  3. Adds `stake_eligibility` to /admin/parallax/pool/snapshot so
     operators can see which mode the daemon is running in.

F31 itself is documented but deferred — fixing it properly
requires a node_id → operator_address mapping that doesn't exist
on-chain yet. Advisory mode is the live-attest path until that
mapping ships.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_advisory_mode_permits_zero_stake_nodes(monkeypatch):
    """PRSM_PARALLAX_STAKE_ELIGIBILITY=advisory → _PermitAllStakeLookup
    wraps the real stake_lookup; is_eligible() returns True for
    every node_id including unstaked ones."""
    monkeypatch.setenv("PRSM_PARALLAX_STAKE_ELIGIBILITY", "advisory")
    from prsm.node.inference_wiring import _wrap_stake_lookup_for_eligibility
    real = MagicMock()
    real.get_stake.return_value = 0
    wrapped = _wrap_stake_lookup_for_eligibility(real)
    assert wrapped is not real  # wrapped
    assert wrapped.get_stake("some-node-id") >= 1  # passes MIN_STAKE


def test_enforced_mode_passes_real_stake_lookup_through(monkeypatch):
    """Default PRSM_PARALLAX_STAKE_ELIGIBILITY=enforced (or unset)
    returns the real stake_lookup unwrapped — zero-stake nodes
    remain filtered out."""
    monkeypatch.delenv("PRSM_PARALLAX_STAKE_ELIGIBILITY", raising=False)
    from prsm.node.inference_wiring import _wrap_stake_lookup_for_eligibility
    real = MagicMock()
    real.get_stake.return_value = 0
    wrapped = _wrap_stake_lookup_for_eligibility(real)
    assert wrapped is real  # no wrapping


def test_advisory_mode_logs_warning_at_construction(monkeypatch, caplog):
    """Advisory mode must surface a loud WARNING so operators
    don't silently run with stake-enforcement disabled in prod."""
    monkeypatch.setenv("PRSM_PARALLAX_STAKE_ELIGIBILITY", "advisory")
    from prsm.node.inference_wiring import _wrap_stake_lookup_for_eligibility
    import logging
    real = MagicMock()
    real.get_stake.return_value = 0
    with caplog.at_level(logging.WARNING):
        _wrap_stake_lookup_for_eligibility(real)
    assert any(
        "advisory" in r.message.lower()
        and "stake" in r.message.lower()
        for r in caplog.records
    ), "advisory mode must log a WARNING about disabled enforcement"


def test_invalid_value_falls_through_to_enforced(monkeypatch):
    """PRSM_PARALLAX_STAKE_ELIGIBILITY=garbage → enforced (safe
    default). Operators can't accidentally disable enforcement
    via typo."""
    monkeypatch.setenv("PRSM_PARALLAX_STAKE_ELIGIBILITY", "garbage")
    from prsm.node.inference_wiring import _wrap_stake_lookup_for_eligibility
    real = MagicMock()
    real.get_stake.return_value = 0
    wrapped = _wrap_stake_lookup_for_eligibility(real)
    assert wrapped is real


def test_filter_pool_error_message_distinguishes_anchor_vs_stake():
    """parallax_executor's failure string must distinguish anchor
    rejection from stake-eligibility rejection. Pre-sprint-686 it
    just said 'no GPU passed anchor verification' — operators chase
    anchor-side bugs that don't exist."""
    import inspect
    from prsm.compute.inference import parallax_executor
    src = inspect.getsource(parallax_executor)
    # Either both strings present (split branches), or one combined
    # string mentioning BOTH anchor + stake. The pre-686 string was
    # anchor-only.
    assert (
        ("stake" in src.lower() and "eligibility" in src.lower())
        or "anchor verification OR stake" in src
    ), "filter_pool error must mention stake eligibility, not just anchor"


def test_snapshot_endpoint_surfaces_eligibility_mode(monkeypatch):
    """Operators must be able to GET the snapshot and see whether
    the daemon is running in advisory or enforced mode — debugging
    the live-attest gap shouldn't require reading systemd-show."""
    monkeypatch.setenv("PRSM_PARALLAX_STAKE_ELIGIBILITY", "advisory")
    # Reuse the same helper sprint 685's tests use to dodge any
    # MagicMock-as-TestClient confusion when both libraries are
    # imported in the same module.
    from tests.unit.test_sprint_685_parallax_pool_snapshot_endpoint import (
        _build_app_with_node,
    )
    node = MagicMock()
    node.inference_executor = MagicMock()
    node.inference_executor._pool_provider = lambda: []
    client = _build_app_with_node(node)
    resp = client.get("/admin/parallax/pool/snapshot")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload.get("stake_eligibility") == "advisory"
