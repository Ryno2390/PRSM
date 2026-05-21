"""Sprint 679 — auto-redial on silent WS drop during chain-exec-ping.

Long inference waits (gpt2 cold-load on a fresh peer can take
25-30s) can outlast the peer's WS idle timeout. Mac's view of the
peer connection stays "connected" because the dead-connection
detection lags, but subsequent ``transport.send_to_peer`` calls
silently fail. The chain-exec-ping adapter would then raise
"transport.send_to_peer returned False" and abort the whole chain.

Sprint 679 fix: when send_to_peer returns False, look up the
peer's network address via _resolve_peer_address (transport.peers
or discovery.known_peers), call transport.connect_to_peer to
redial, and retry the send once. Only if the redial AND retry
also fail does the adapter raise.

Live evidence: sprint 677/678 multi-host runs hit this exact
pattern during ~25s SFO cold-load.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_resolve_peer_address_from_transport():
    """Transport's `peers` dict carries a PeerConnection with
    `.address` — first source."""
    from prsm.node.chain_executor_adapters import _resolve_peer_address
    node = MagicMock()
    peer = MagicMock()
    peer.address = "1.2.3.4:9001"
    node.transport.peers = {"abc": peer}
    assert _resolve_peer_address(node, "abc") == "1.2.3.4:9001"


def test_resolve_peer_address_from_discovery_known_peers():
    """When transport.peers doesn't have it, fall back to
    discovery.known_peers (PeerInfo.address)."""
    from prsm.node.chain_executor_adapters import _resolve_peer_address
    node = MagicMock()
    node.transport.peers = {}
    info = MagicMock()
    info.address = "5.6.7.8:9001"
    node.discovery.known_peers = {"xyz": info}
    assert _resolve_peer_address(node, "xyz") == "5.6.7.8:9001"


def test_resolve_peer_address_returns_none_when_unknown():
    """Peer truly absent → None (caller can't redial, must fail
    the send loudly)."""
    from prsm.node.chain_executor_adapters import _resolve_peer_address
    node = MagicMock()
    node.transport.peers = {}
    node.discovery.known_peers = {}
    assert _resolve_peer_address(node, "unknown") is None


def test_resolve_peer_address_tolerates_missing_attributes():
    """Mocks without the expected attribute trees → None, never
    AttributeError. Defends against test-fixture skew."""
    from prsm.node.chain_executor_adapters import _resolve_peer_address
    node = object()  # bare object, no .transport, no .discovery
    assert _resolve_peer_address(node, "anything") is None


def test_redial_logic_present_in_send_adapter():
    """Source-grep guard: the build_send_message_adapter must
    contain the sprint-679 redial sequence. Catches future
    refactors that strip the redial path.
    """
    import inspect
    from prsm.node.chain_executor_adapters import (
        build_send_message_adapter,
    )
    src = inspect.getsource(build_send_message_adapter)
    assert "_resolve_peer_address" in src
    assert "connect_to_peer" in src
    assert "Sprint 679" in src or "sprint 679" in src
