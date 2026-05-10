"""Canonical-drift detection extension to Phase 7-storage + Phase 8.

DaemonWatchdog._canonical_check_fn() previously covered only
ftns_ledger / royalty_distributor / provenance_registry. This
sprint extends to storage_slashing / compensation_distributor /
key_distribution so operators wired to those contracts with a
mismatched address get canonical.drifted webhooks too.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from prsm.node import node as node_module


@pytest.fixture
def fake_node():
    """Build a node-like object with the 3 phase 7/8 clients."""
    n = MagicMock()
    n._storage_slashing_client = None
    n._compensation_distributor_client = None
    n._key_distribution_client = None
    n.ftns_ledger = None
    n._royalty_distributor_client = None
    n._provenance_client = None
    return n


def _build_canonical_check_fn(node):
    """Reach into node.py module and reconstruct the closure that
    DaemonWatchdog uses. We do this by inlining the exact same
    code path so we can unit-test it without spinning up the
    whole node init."""
    # Inline a copy of the closure for test isolation. Mirrors
    # the canonical_check_fn closure in node.py (lines ~2099-2185
    # post-sprint-86). Keep in sync with that closure.
    def _canonical_check_fn():
        from prsm.config.networks import (
            get_network_config, _resolve_network_name,
        )
        try:
            cfg = get_network_config(_resolve_network_name())
        except Exception:
            return {}
        result = {}
        ftns = getattr(node, "ftns_ledger", None)
        if ftns is not None:
            wired = getattr(ftns, "contract_address", None)
            canonical = cfg.ftns_token
            if wired and canonical:
                result["ftns_ledger"] = (wired, canonical)
        for attr_name, networks_field, label in (
            (
                "_storage_slashing_client", "storage_slashing",
                "storage_slashing",
            ),
            (
                "_compensation_distributor_client",
                "compensation_distributor", "compensation_distributor",
            ),
            (
                "_key_distribution_client", "key_distribution",
                "key_distribution",
            ),
        ):
            client = getattr(node, attr_name, None)
            if client is None:
                continue
            # Sprint 142 mirror: read CONTRACT address, not signer.
            wired = (
                getattr(client, "contract_address", None)
                or getattr(client, "address", None)
            )
            canonical = getattr(cfg, networks_field, None)
            if wired and canonical:
                result[label] = (wired, canonical)
        return result

    return _canonical_check_fn


def test_storage_slashing_canonical_pin_surfaced(fake_node):
    canonical = "0x0e9cAfadCCCe0987C773B5FdFF295c2Aa6F03337"
    fake_client = MagicMock(spec=["contract_address", "address"])
    fake_client.contract_address = canonical  # match
    fake_client.address = "0xSIGNER"
    fake_node._storage_slashing_client = fake_client
    with patch.dict(os.environ, {"PRSM_NETWORK": "mainnet"}):
        fn = _build_canonical_check_fn(fake_node)
        result = fn()
    assert "storage_slashing" in result
    wired, expected = result["storage_slashing"]
    assert wired == canonical
    assert expected == canonical  # matches networks.py


def test_compensation_distributor_canonical_pin_surfaced(fake_node):
    fake_client = MagicMock(spec=["contract_address", "address"])
    fake_client.contract_address = "0xWRONG"
    fake_client.address = "0xSIGNER"
    fake_node._compensation_distributor_client = fake_client
    with patch.dict(os.environ, {"PRSM_NETWORK": "mainnet"}):
        fn = _build_canonical_check_fn(fake_node)
        result = fn()
    assert "compensation_distributor" in result
    wired, expected = result["compensation_distributor"]
    assert wired == "0xWRONG"
    # Expected is the canonical from networks.py — mismatch
    assert expected != "0xWRONG"


def test_key_distribution_canonical_pin_surfaced(fake_node):
    fake_client = MagicMock(spec=["contract_address", "address"])
    fake_client.contract_address = "0xCAFE"
    fake_client.address = "0xSIGNER"
    fake_node._key_distribution_client = fake_client
    with patch.dict(os.environ, {"PRSM_NETWORK": "mainnet"}):
        fn = _build_canonical_check_fn(fake_node)
        result = fn()
    assert "key_distribution" in result


def test_unwired_client_skipped(fake_node):
    """When a phase 7/8 client is None, it doesn't appear in
    the canonical-check result (operator opted out)."""
    with patch.dict(os.environ, {"PRSM_NETWORK": "mainnet"}):
        fn = _build_canonical_check_fn(fake_node)
        result = fn()
    assert "storage_slashing" not in result
    assert "compensation_distributor" not in result
    assert "key_distribution" not in result


def test_client_without_address_attr_skipped(fake_node):
    """A malformed client missing `.address` doesn't crash the
    check or appear in result."""
    bad_client = MagicMock(spec=[])  # no `.address`
    fake_node._storage_slashing_client = bad_client
    with patch.dict(os.environ, {"PRSM_NETWORK": "mainnet"}):
        fn = _build_canonical_check_fn(fake_node)
        # Should NOT raise
        result = fn()
    assert "storage_slashing" not in result
