"""Sprint 6 Tranche 1: Bootstrap fallback, address validation, and telemetry tests."""

from types import SimpleNamespace

import pytest

from prsm.node.discovery import (
    DISCOVERY_PEER_REQUEST,
    PeerDiscovery,
    validate_bootstrap_address,
)
from prsm.node.transport import MSG_GOSSIP, PeerConnection


# ── Test infrastructure (same pattern as tranche 0) ──────────────


class _MockTransport:
    def __init__(self, connect_results):
        self.identity = SimpleNamespace(node_id="node-local")
        self.host = "127.0.0.1"
        self.port = 19001
        self.peers = {}
        self.peer_count = 0
        self._connect_results = list(connect_results)
        self.connect_calls = []
        self.sent = []

    def on_message(self, _msg_type, _handler):
        return None

    async def connect_to_peer(self, address):
        self.connect_calls.append(address)
        if not self._connect_results:
            return None
        return self._connect_results.pop(0)

    async def send_to_peer(self, peer_id, msg):
        self.sent.append((peer_id, msg))
        return True


def _peer(peer_id: str, address: str) -> PeerConnection:
    return PeerConnection(
        peer_id=peer_id, address=address, websocket=object(), outbound=True
    )


# ── Address validation unit tests ─────────────────────────────────


class TestValidateBootstrapAddress:
    """Unit tests for validate_bootstrap_address()."""

    def test_valid_wss_url(self):
        ok, reason = validate_bootstrap_address("wss://bootstrap.prsm-network.com")
        assert ok is True
        assert reason == ""

    def test_valid_ws_url_with_port(self):
        ok, reason = validate_bootstrap_address("ws://host.example:9001")
        assert ok is True

    def test_valid_bare_host_port(self):
        ok, reason = validate_bootstrap_address("host.example:9001")
        assert ok is True

    def test_reject_empty_string(self):
        ok, reason = validate_bootstrap_address("")
        assert ok is False
        assert "empty" in reason.lower()

    def test_reject_whitespace_only(self):
        ok, reason = validate_bootstrap_address("   ")
        assert ok is False
        assert "empty" in reason.lower()

    def test_reject_http_scheme(self):
        ok, reason = validate_bootstrap_address("http://example.com")
        assert ok is False
        assert "scheme" in reason.lower()

    def test_reject_ftp_scheme(self):
        ok, reason = validate_bootstrap_address("ftp://example.com:21")
        assert ok is False
        assert "scheme" in reason.lower()

    def test_reject_non_numeric_port(self):
        ok, reason = validate_bootstrap_address("host.example:abc")
        assert ok is False
        assert "port" in reason.lower()

    def test_reject_port_out_of_range(self):
        ok, reason = validate_bootstrap_address("host.example:99999")
        assert ok is False
        assert "port" in reason.lower()

    def test_accept_hostname_only(self):
        ok, _ = validate_bootstrap_address("bootstrap.prsm-network.com")
        assert ok is True

    def test_strips_whitespace(self):
        ok, _ = validate_bootstrap_address("  wss://example.com  ")
        assert ok is True


# ── Fallback activation tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_fallback_activated_when_all_primary_nodes_fail():
    """When all configured primaries fail, fallback nodes are tried."""
    transport = _MockTransport(
        connect_results=[None, _peer("peer-fb", "fb1:9001")]
    )
    discovery = PeerDiscovery(
        transport=transport,
        bootstrap_nodes=["primary:9001"],
        bootstrap_fallback_enabled=True,
        bootstrap_fallback_nodes=["fb1:9001"],
        bootstrap_connect_timeout=1.0,
        bootstrap_retry_attempts=1,
        bootstrap_backoff_base=0.01,
        bootstrap_backoff_max=0.02,
    )

    connected = await discovery.bootstrap()

    assert connected == 1
    assert discovery.bootstrap_success_node == "fb1:9001"
    assert not discovery.bootstrap_degraded_mode

    telemetry = discovery.get_bootstrap_telemetry()
    assert telemetry["fallback_activated"] is True
    assert telemetry["fallback_succeeded"] is True
    assert telemetry["fallback_attempted"] == 1
    assert telemetry["source_policy"] == "primary_then_fallback"

    status = discovery.get_bootstrap_status()
    assert status["fallback_activated"] is True
    assert status["fallback_succeeded"] is True


@pytest.mark.asyncio
async def test_fallback_disabled_does_not_try_fallback_nodes():
    """When fallback is disabled, only primary nodes are tried."""
    transport = _MockTransport(connect_results=[None])
    discovery = PeerDiscovery(
        transport=transport,
        bootstrap_nodes=["primary:9001"],
        bootstrap_fallback_enabled=False,
        bootstrap_fallback_nodes=["fb1:9001"],
        bootstrap_connect_timeout=1.0,
        bootstrap_retry_attempts=1,
    )

    connected = await discovery.bootstrap()

    assert connected == 0
    assert discovery.bootstrap_degraded_mode is True
    assert transport.connect_calls == ["primary:9001"]
    # Fallback node should never be attempted
    assert "fb1:9001" not in transport.connect_calls

    telemetry = discovery.get_bootstrap_telemetry()
    assert telemetry["fallback_activated"] is False
    assert telemetry["source_policy"] == "primary_only"


@pytest.mark.asyncio
async def test_primary_success_skips_fallback_entirely():
    """When the first primary succeeds, fallback nodes are not touched."""
    transport = _MockTransport(
        connect_results=[_peer("peer-a", "primary:9001")]
    )
    discovery = PeerDiscovery(
        transport=transport,
        bootstrap_nodes=["primary:9001"],
        bootstrap_fallback_enabled=True,
        bootstrap_fallback_nodes=["fb1:9001", "fb2:9001"],
        bootstrap_connect_timeout=1.0,
        bootstrap_retry_attempts=1,
    )

    connected = await discovery.bootstrap()

    assert connected == 1
    assert transport.connect_calls == ["primary:9001"]
    telemetry = discovery.get_bootstrap_telemetry()
    assert telemetry["fallback_activated"] is False
    assert telemetry["fallback_attempted"] == 0


# ── Malformed address rejection tests ──────────────────────────────


@pytest.mark.asyncio
async def test_malformed_bootstrap_address_rejected_and_skipped():
    """Malformed addresses are rejected; valid ones are still tried."""
    transport = _MockTransport(
        connect_results=[_peer("peer-valid", "valid:9001")]
    )
    discovery = PeerDiscovery(
        transport=transport,
        bootstrap_nodes=["", "http://bad-scheme.com", "valid:9001"],
        bootstrap_validate_addresses=True,
        bootstrap_connect_timeout=1.0,
        bootstrap_retry_attempts=1,
    )

    connected = await discovery.bootstrap()

    assert connected == 1
    # Only the valid address should have been attempted
    assert transport.connect_calls == ["valid:9001"]
    assert discovery.bootstrap_success_node == "valid:9001"

    telemetry = discovery.get_bootstrap_telemetry()
    assert telemetry["addresses_rejected"] == 2
    assert telemetry["addresses_validated"] >= 3


@pytest.mark.asyncio
async def test_all_addresses_malformed_enters_degraded_mode():
    """If every address is malformed, node enters degraded mode."""
    transport = _MockTransport(connect_results=[])
    discovery = PeerDiscovery(
        transport=transport,
        bootstrap_nodes=["", "ftp://bad:21"],
        bootstrap_fallback_enabled=True,
        bootstrap_fallback_nodes=["http://also-bad"],
        bootstrap_validate_addresses=True,
        bootstrap_connect_timeout=1.0,
        bootstrap_retry_attempts=1,
    )

    connected = await discovery.bootstrap()

    assert connected == 0
    assert discovery.bootstrap_degraded_mode is True
    assert transport.connect_calls == []

    telemetry = discovery.get_bootstrap_telemetry()
    assert telemetry["addresses_rejected"] == 3


@pytest.mark.asyncio
async def test_validation_disabled_passes_all_addresses():
    """When validation is off, all addresses are passed through."""
    transport = _MockTransport(
        connect_results=[None, _peer("peer-b", "http://bad:9001")]
    )
    discovery = PeerDiscovery(
        transport=transport,
        bootstrap_nodes=["", "http://bad:9001"],
        bootstrap_validate_addresses=False,
        bootstrap_connect_timeout=1.0,
        bootstrap_retry_attempts=1,
    )

    connected = await discovery.bootstrap()

    # Both addresses should have been attempted (validation disabled)
    assert len(transport.connect_calls) == 2
    telemetry = discovery.get_bootstrap_telemetry()
    assert telemetry["addresses_rejected"] == 0


# ── Deterministic success with trusted fallback tests ──────────────


@pytest.mark.asyncio
async def test_deterministic_success_when_fallback_reachable():
    """Fresh node with unreachable primary but reachable fallback
    deterministically reaches healthy state."""
    transport = _MockTransport(
        connect_results=[
            None, None,  # primary retries
            _peer("peer-fb1", "fb1:9001"),  # fallback succeeds
        ]
    )
    discovery = PeerDiscovery(
        transport=transport,
        bootstrap_nodes=["unreachable:9001"],
        bootstrap_fallback_enabled=True,
        bootstrap_fallback_nodes=["fb1:9001"],
        bootstrap_connect_timeout=1.0,
        bootstrap_retry_attempts=2,
        bootstrap_backoff_base=0.01,
        bootstrap_backoff_max=0.02,
    )

    await discovery.start()

    assert discovery._running is True
    assert discovery.bootstrap_connected_count == 1
    assert discovery.bootstrap_success_node == "fb1:9001"
    assert not discovery.bootstrap_degraded_mode

    # Peer request should have been sent
    assert len(transport.sent) == 1
    _, msg = transport.sent[0]
    assert msg.msg_type == MSG_GOSSIP
    assert msg.payload.get("subtype") == DISCOVERY_PEER_REQUEST

    await discovery.stop()


# ── Backoff telemetry tests ───────────────────────────────────────


@pytest.mark.asyncio
async def test_backoff_accumulates_in_telemetry():
    """Exponential backoff delay is tracked in telemetry."""
    transport = _MockTransport(
        connect_results=[None, None, _peer("peer-a", "a:9001")]
    )
    discovery = PeerDiscovery(
        transport=transport,
        bootstrap_nodes=["a:9001"],
        bootstrap_connect_timeout=1.0,
        bootstrap_retry_attempts=3,
        bootstrap_backoff_base=0.01,
        bootstrap_backoff_max=1.0,
    )

    await discovery.bootstrap()

    telemetry = discovery.get_bootstrap_telemetry()
    # With 3 attempts, backoff should occur after attempt 1 and attempt 2
    assert telemetry["backoff_total_seconds"] > 0


# ── Duplicate deduplication test ──────────────────────────────────


@pytest.mark.asyncio
async def test_fallback_deduplicates_addresses_already_in_primary():
    """Fallback addresses that duplicate primary addresses are skipped."""
    transport = _MockTransport(
        connect_results=[None, _peer("peer-fb2", "fb2:9001")]
    )
    discovery = PeerDiscovery(
        transport=transport,
        bootstrap_nodes=["shared:9001"],
        bootstrap_fallback_enabled=True,
        # shared:9001 duplicates primary; fb2:9001 is unique fallback
        bootstrap_fallback_nodes=["shared:9001", "fb2:9001"],
        bootstrap_connect_timeout=1.0,
        bootstrap_retry_attempts=1,
        bootstrap_backoff_base=0.01,
        bootstrap_backoff_max=0.02,
    )

    connected = await discovery.bootstrap()

    assert connected == 1
    # shared:9001 tried as primary, then fb2:9001 as fallback (not shared:9001 again)
    assert transport.connect_calls == ["shared:9001", "fb2:9001"]
    assert discovery.bootstrap_success_node == "fb2:9001"


# ── Telemetry snapshot stability test ──────────────────────────────


@pytest.mark.asyncio
async def test_bootstrap_telemetry_returns_stable_copy():
    """get_bootstrap_telemetry() returns a dict, not a reference to internal state."""
    transport = _MockTransport(connect_results=[_peer("p", "a:9001")])
    discovery = PeerDiscovery(
        transport=transport,
        bootstrap_nodes=["a:9001"],
        bootstrap_connect_timeout=1.0,
        bootstrap_retry_attempts=1,
    )

    await discovery.bootstrap()

    t1 = discovery.get_bootstrap_telemetry()
    t2 = discovery.get_bootstrap_telemetry()
    assert t1 is not t2
    assert t1 == t2
    # Mutating the returned dict should not affect internal state
    t1["addresses_validated"] = 9999
    t3 = discovery.get_bootstrap_telemetry()
    assert t3["addresses_validated"] != 9999
