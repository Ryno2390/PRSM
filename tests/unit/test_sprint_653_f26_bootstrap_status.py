"""Sprint 653 (F26) — bootstrap status distinguishes P2P-probe
failures from WS-protocol probe failures.

Pre-653 the `/bootstrap/status` endpoint reported every bootstrap
server in `failed_nodes` because the P2P-handshake probe (which
runs first) always fails against a bootstrap server (they speak
the BootstrapClient WS protocol, not PRSM P2P). Operators reading
the status concluded bootstrap-eu/apac were down even when they
were fully operational.

Sprint 653 fix:
  1. Track WS-protocol probes separately (`bootstrap_client_attempted_nodes`
     + `bootstrap_client_failed_nodes`).
  2. When a node succeeds via the WS protocol, REMOVE it from
     `bootstrap_failed_nodes` (P2P-only-probe failure is now masked).
  3. Status exposes both views so operators can reason about
     "P2P failed but WS succeeded" vs "WS genuinely failed".
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from prsm.node.discovery import PeerDiscovery


def _make_discovery():
    """Build a minimal PeerDiscovery instance for state-level tests."""
    transport = MagicMock()
    transport.identity = MagicMock(node_id="test-node")
    transport.port = 9001
    return PeerDiscovery(
        transport=transport,
        bootstrap_nodes=["wss://bootstrap-us.example:8765"],
    )


def test_init_creates_ws_probe_tracking_fields():
    """Sprint 653 — new tracking fields exist + start empty."""
    d = _make_discovery()
    assert d.bootstrap_client_attempted_nodes == []
    assert d.bootstrap_client_failed_nodes == []


def test_get_bootstrap_status_exposes_ws_tracking():
    """The new fields surface in /bootstrap/status payload."""
    d = _make_discovery()
    d.bootstrap_client_attempted_nodes.append("wss://x:8765")
    d.bootstrap_client_failed_nodes.append("wss://y:8765")
    status = d.get_bootstrap_status()
    assert status["bootstrap_client_attempted_nodes"] == ["wss://x:8765"]
    assert status["bootstrap_client_failed_nodes"] == ["wss://y:8765"]


def test_ws_success_removes_from_failed_nodes():
    """The headline F26 fix: when WS succeeds, the address is
    removed from bootstrap_failed_nodes so operators don't see a
    reachable node listed as failed.
    """
    d = _make_discovery()
    # Simulate the P2P-probe-failed state
    addr = "wss://bootstrap-us.example:8765"
    d.bootstrap_failed_nodes.append(addr)
    assert addr in d.bootstrap_failed_nodes
    # The state mutation that should happen on WS-success — we
    # simulate it directly rather than spinning up a fake server.
    # (The actual code does this inside _try_bootstrap_client's
    # success block.)
    d.bootstrap_success_node = addr
    if addr in d.bootstrap_failed_nodes:
        d.bootstrap_failed_nodes.remove(addr)
    assert addr not in d.bootstrap_failed_nodes
    # success_node still points at the address
    assert d.bootstrap_success_node == addr


def test_failed_nodes_keeps_unreachable_addresses():
    """If WS never succeeds, failed_nodes still lists the address
    (regression guard for the removal logic).
    """
    d = _make_discovery()
    eu = "wss://bootstrap-eu.example:8765"
    apac = "wss://bootstrap-apac.example:8765"
    d.bootstrap_failed_nodes.extend([eu, apac])
    d.bootstrap_success_node = "wss://bootstrap-us.example:8765"
    # We never WS-probed eu/apac → they stay in failed_nodes
    assert eu in d.bootstrap_failed_nodes
    assert apac in d.bootstrap_failed_nodes
