"""Sprint 322 — Thread peer capabilities through bootstrap hydration.

Pre-sprint-322 `_hydrate_peers_from_bootstrap` only populated
PeerInfo.node_id + .address + .last_seen — it discarded the
`capabilities` list the bootstrap server attached to every
peer in its peer_list response. Downstream
`find_peers_with_capability(cap)` then returned empty for
bootstrap-discovered peers regardless of what they reported,
so QueryOrchestrator-style selectors saw no candidates.

Live dogfood 2026-05-12 confirmed the canonical bootstrap
server's peer_list payload format includes `capabilities`
per peer:
  {"peer_id": "...", "address": "...", "port": ...,
   "capabilities": ["compute", "storage"], "region": null,
   "version": "..."}

Sprint 322 threads `bp.capabilities` into the constructed
PeerInfo so capability-keyed selection works end to end.
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

from prsm.bootstrap.client import BootstrapPeer
from prsm.node.libp2p_discovery import Libp2pDiscovery


def _make_discovery():
    transport = MagicMock()
    transport.identity.node_id = "self-node"
    transport.port = 9001
    transport.connect_to_peer = AsyncMock(return_value=None)
    return Libp2pDiscovery(
        transport=transport,
        bootstrap_nodes=[],
    )


def test_bootstrap_capabilities_threaded_into_peer_info():
    """A BootstrapPeer with capabilities=['compute','storage']
    must produce a PeerInfo with the same capability list."""
    d = _make_discovery()
    d._hydrate_peers_from_bootstrap([
        BootstrapPeer(
            peer_id="peer-1",
            address="10.0.0.1",
            port=9001,
            capabilities=["compute", "storage"],
        ),
    ])
    pi = d._capability_index.get("peer-1")
    assert pi is not None
    assert set(pi.capabilities) == {"compute", "storage"}


def test_find_peers_with_capability_after_bootstrap_hydration():
    """End-to-end: hydrate from bootstrap with mixed capabilities,
    confirm find_peers_with_capability returns the right
    subset."""
    d = _make_discovery()
    d._hydrate_peers_from_bootstrap([
        BootstrapPeer(peer_id="cpu-only", address="10.0.0.1",
                      port=9001, capabilities=["compute"]),
        BootstrapPeer(peer_id="storage-only", address="10.0.0.2",
                      port=9001, capabilities=["storage"]),
        BootstrapPeer(peer_id="both", address="10.0.0.3",
                      port=9001, capabilities=["compute", "storage"]),
    ])
    compute_peers = d.find_peers_with_capability("compute")
    ids = sorted(p.node_id for p in compute_peers)
    # Pre-fix this would have been [] — all PeerInfo objects
    # had empty .capabilities lists
    assert ids == ["both", "cpu-only"]


def test_peer_join_announcement_capabilities_left_empty():
    """peer_join announcements (sprint 320 path) don't carry
    capabilities — leave empty without raising. Captured here
    so a future change that DOES start receiving caps on
    announcements doesn't silently regress (this test would
    need to be updated)."""
    d = _make_discovery()
    client = MagicMock()
    client._observed_announcements = [
        {
            "announcement_type": "peer_join",
            "peer_id": "p1",
            "peer_endpoint": "10.0.0.1:9001",
        },
    ]
    d._consume_bootstrap_announcements(client)
    pi = d._capability_index.get("p1")
    assert pi is not None
    # Capability list empty — announcements don't carry caps
    assert pi.capabilities == []


def test_capabilities_overwritten_on_re_hydration():
    """If a peer's capabilities change between polls (operator
    updates set_local_capabilities + re-registers), the next
    hydration must reflect the new list — not append, not
    leave stale."""
    d = _make_discovery()
    d._hydrate_peers_from_bootstrap([
        BootstrapPeer(peer_id="p", address="10.0.0.1", port=9001,
                      capabilities=["compute"]),
    ])
    assert set(d._capability_index["p"].capabilities) == {"compute"}

    # Re-hydrate with a different cap set
    d._hydrate_peers_from_bootstrap([
        BootstrapPeer(peer_id="p", address="10.0.0.1", port=9001,
                      capabilities=["storage", "gpu"]),
    ])
    assert set(d._capability_index["p"].capabilities) == {"storage", "gpu"}


def test_missing_capabilities_field_defaults_to_empty():
    """Forward-compat: if a BootstrapPeer somehow has no
    capabilities attribute, hydration must not raise — just
    yield empty caps for that peer."""
    d = _make_discovery()
    # BootstrapPeer.capabilities has a default_factory=list, so
    # this constructs with [] not missing — but we still want to
    # confirm the behavior is the same as the explicit empty.
    d._hydrate_peers_from_bootstrap([
        BootstrapPeer(peer_id="bare", address="10.0.0.1", port=9001),
    ])
    pi = d._capability_index.get("bare")
    assert pi is not None
    assert pi.capabilities == []
