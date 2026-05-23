"""Sprint 781 — F14 fix: auto-loopback-rewrite for co-located peers.

Sprint 456 surfaced F14: a same-host multi-daemon test bench
discovers peers correctly via the bootstrap server, but
`connect_to_peer` fails because both daemons announce the
host's external IP and the OS can't loopback to its own
external IP through NAT (no NAT hairpin pinning).

Pre-781 fix candidates from sprint 456 notes:
  A) PRSM_ADVERTISE_ADDRESS env override (manual config)
  B) STUN-style NAT detection (fancy)
  C) multi-host cloud VM (eventual right answer; doesn't help dev)

Sprint 781 ships AUTOMATIC detection: when a peer announces an
address whose host portion matches our own announced address,
we MUST be co-located → rewrite the dial target to
`127.0.0.1:<port>` (keep port, swap host for loopback).

This works without operator config when both daemons share an
externally-discovered IP (the common same-host case). Multi-host
operators on different IPs see no behavior change — the rewrite
only fires on host-match.

Pin tests:
- _rewrite_co_located_address helper exists.
- Same host + different port → rewrite to 127.0.0.1.
- Different host → unchanged.
- own_advertise=None → never rewrite (no signal).
- Empty / bogus peer addr → unchanged.
- Port preserved through rewrite.
- Port-less peer addr (just IP) → return unchanged (we need a
  port to rewrite; no information to fabricate one).
- Source-shape: _auto_dial_sweep calls the rewrite helper
  BEFORE connect_to_peer.
- Source-shape: maintain_connections also calls the helper.
"""
from __future__ import annotations

import inspect


# ---- Helper function exists + behavior --------------------------


def test_helper_function_exists():
    from prsm.node.discovery import _rewrite_co_located_address
    assert callable(_rewrite_co_located_address)


def test_same_host_different_port_rewrites_to_loopback():
    """Co-located daemon case: announced IP matches ours →
    swap host portion for 127.0.0.1, keep port."""
    from prsm.node.discovery import _rewrite_co_located_address
    out = _rewrite_co_located_address(
        peer_address="146.70.202.118:9011",
        own_advertise="146.70.202.118",
    )
    assert out == "127.0.0.1:9011"


def test_different_host_unchanged():
    """Multi-host case: announced IP differs from ours →
    keep address as-is so the real network path is used."""
    from prsm.node.discovery import _rewrite_co_located_address
    out = _rewrite_co_located_address(
        peer_address="10.0.0.5:9001",
        own_advertise="146.70.202.118",
    )
    assert out == "10.0.0.5:9001"


def test_own_advertise_none_no_rewrite():
    """No signal of what our own announced IP is → can't
    detect co-location → never rewrite."""
    from prsm.node.discovery import _rewrite_co_located_address
    out = _rewrite_co_located_address(
        peer_address="146.70.202.118:9011",
        own_advertise=None,
    )
    assert out == "146.70.202.118:9011"


def test_empty_address_returned_unchanged():
    """Caller filters bogus addresses BEFORE invoking the
    helper (sprint 570 F28 defense-in-depth). Helper itself
    just leaves them alone."""
    from prsm.node.discovery import _rewrite_co_located_address
    assert _rewrite_co_located_address(
        peer_address="",
        own_advertise="146.70.202.118",
    ) == ""
    assert _rewrite_co_located_address(
        peer_address="0.0.0.0:9001",
        own_advertise="146.70.202.118",
    ) == "0.0.0.0:9001"


def test_port_preserved_through_rewrite():
    """The port is the load-bearing identifier when two daemons
    share a host — must survive the rewrite intact."""
    from prsm.node.discovery import _rewrite_co_located_address
    out = _rewrite_co_located_address(
        peer_address="146.70.202.118:9999",
        own_advertise="146.70.202.118",
    )
    assert out.endswith(":9999")


def test_port_less_peer_address_unchanged():
    """No port → no information to dial → return unchanged
    (don't fabricate a default port)."""
    from prsm.node.discovery import _rewrite_co_located_address
    out = _rewrite_co_located_address(
        peer_address="146.70.202.118",
        own_advertise="146.70.202.118",
    )
    assert out == "146.70.202.118"


def test_own_advertise_with_port_strips_for_match():
    """If own_advertise comes through with a port (legacy
    callers) the host-match comparison still works."""
    from prsm.node.discovery import _rewrite_co_located_address
    out = _rewrite_co_located_address(
        peer_address="146.70.202.118:9011",
        own_advertise="146.70.202.118:9001",
    )
    # Same host → rewrite, regardless of port suffix on advertise
    assert out == "127.0.0.1:9011"


# ---- Source-shape pins ------------------------------------------


def test_auto_dial_sweep_invokes_rewrite_helper():
    """_auto_dial_sweep must consult the rewrite helper before
    connect_to_peer to fix F14 at the auto-dial path."""
    from prsm.node.discovery import PeerDiscovery
    src = inspect.getsource(PeerDiscovery._auto_dial_sweep)
    assert "_rewrite_co_located_address" in src, (
        "Sprint 781: _auto_dial_sweep must consult the F14 "
        "loopback-rewrite helper before connect_to_peer"
    )
    rewrite_idx = src.find("_rewrite_co_located_address")
    connect_idx = src.find("connect_to_peer")
    assert rewrite_idx > 0
    assert connect_idx > 0
    assert rewrite_idx < connect_idx, (
        "rewrite must happen BEFORE connect_to_peer"
    )


def test_maintain_connections_invokes_rewrite_helper():
    """maintain_connections must also consult the rewrite
    helper so the periodic-maintain path closes F14 too."""
    from prsm.node.discovery import PeerDiscovery
    src = inspect.getsource(PeerDiscovery.maintain_connections)
    assert "_rewrite_co_located_address" in src, (
        "Sprint 781: maintain_connections must consult the F14 "
        "loopback-rewrite helper before connect_to_peer"
    )
