"""Sprint 566 — PRSM_ADVERTISE_ADDRESS env-var support.

Sprint 565 closed bootstrap-mediated peer discovery cross-host but
surfaced the next layer: the droplet advertises `127.0.0.1:9001`
in its registration because (a) it bootstraps via loopback (per
sprint-460 invariant) and (b) the bootstrap-server records the
client's address as seen on the WS connection (`client_ip` in
`_handle_register`, line 436 pre-sprint).

Result: bootstrap-us /peers shows droplet at 127.0.0.1, which is
loopback to anyone else — peers know about the droplet but can't
connect to it. Sprint-456 deferred this as candidate A:
"PRSM_ADVERTISE_ADDRESS env override".

Sprint 566 implements candidate A end-to-end:

1. `BootstrapClient.__init__` gains `advertise_address: Optional[str]`.
   When provided, the register message carries an `address` field.

2. Bootstrap server's `_handle_register` honors a client-supplied
   `address` if present (validated as non-empty string); falls back
   to `client_ip` as before. Back-compat: clients that don't send
   `address` see no change.

3. `Libp2pDiscovery._try_bootstrap_client` (and `node/discovery.py`)
   read PRSM_ADVERTISE_ADDRESS from env and pass it through.

Operators co-located with a bootstrap-server can now bootstrap via
loopback (sprint-460 invariant) BUT advertise their external IP to
remote peers (sprint-566 fix). The droplet operator should set
PRSM_ADVERTISE_ADDRESS=159.203.129.218:9001 in its systemd unit.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, AsyncMock

import pytest


# ── BootstrapClient: register-msg shape ──────────────────


def test_bootstrap_client_accepts_advertise_address_kwarg():
    """Constructor accepts the new kwarg without raising."""
    from prsm.bootstrap.client import BootstrapClient
    client = BootstrapClient(
        bootstrap_url="wss://x/",
        node_id="n",
        port=9001,
        advertise_address="159.203.129.218:9001",
    )
    assert client.advertise_address == "159.203.129.218:9001"


def test_bootstrap_client_advertise_address_default_none():
    """Back-compat: no advertise_address → attr is None, no `address`
    field in register message."""
    from prsm.bootstrap.client import BootstrapClient
    client = BootstrapClient(
        bootstrap_url="wss://x/", node_id="n", port=9001,
    )
    assert client.advertise_address is None


@pytest.mark.asyncio
async def test_register_msg_includes_address_when_advertise_set(
    monkeypatch,
):
    """When advertise_address is set, the register message JSON sent
    on the wire contains the `address` field."""
    from prsm.bootstrap import client as client_mod

    sent_payloads = []

    class _FakeWS:
        async def send(self, msg):
            sent_payloads.append(msg)

        async def recv(self):
            # The client expects a register_ack — minimal valid response.
            return json.dumps({
                "type": "register_ack",
                "peer_id": "n",
                "peers": [],
                "heartbeat_interval": 30,
                "server_time": "now",
            })

        async def close(self):
            pass

    async def _fake_connect(url, **kwargs):
        return _FakeWS()

    monkeypatch.setattr(
        client_mod.websockets.client, "connect", _fake_connect,
    )

    client = client_mod.BootstrapClient(
        bootstrap_url="wss://example/",
        node_id="n",
        port=9001,
        advertise_address="159.203.129.218:9001",
    )
    await client.connect()

    register_msg = json.loads(sent_payloads[0])
    assert register_msg["type"] == "register"
    assert register_msg["address"] == "159.203.129.218:9001"


@pytest.mark.asyncio
async def test_register_msg_omits_address_when_advertise_unset(
    monkeypatch,
):
    """Back-compat: no advertise_address → register message has no
    `address` field. Sprint-565 servers handling old clients keep
    falling back to client_ip."""
    from prsm.bootstrap import client as client_mod

    sent_payloads = []

    class _FakeWS:
        async def send(self, msg):
            sent_payloads.append(msg)

        async def recv(self):
            return json.dumps({
                "type": "register_ack", "peer_id": "n",
                "peers": [], "heartbeat_interval": 30,
                "server_time": "now",
            })

        async def close(self):
            pass

    async def _fake_connect(url, **kwargs):
        return _FakeWS()

    monkeypatch.setattr(
        client_mod.websockets.client, "connect", _fake_connect,
    )

    client = client_mod.BootstrapClient(
        bootstrap_url="wss://example/", node_id="n", port=9001,
    )
    await client.connect()

    register_msg = json.loads(sent_payloads[0])
    assert "address" not in register_msg


# ── Server: _handle_register honors client-supplied address ──


@pytest.mark.asyncio
async def test_server_honors_client_supplied_address(tmp_path):
    """When the register message contains `address`, the server
    records THAT value, not the WS client_ip."""
    from prsm.bootstrap.server import BootstrapServer
    from prsm.bootstrap.config import BootstrapConfig

    config = BootstrapConfig(
        host="127.0.0.1", port=18765, api_port=18000,
    )
    server = BootstrapServer(config=config)

    # Stub WebSocket — captures the register_ack send.
    sent = []
    fake_ws = MagicMock()
    fake_ws.send = AsyncMock(side_effect=lambda m: sent.append(m))

    await server._handle_register(
        websocket=fake_ws,
        data={
            "peer_id": "remote-node",
            "port": 9001,
            "address": "203.0.113.42",  # client-supplied external IP
            "capabilities": ["compute"],
            "version": "1.7.0",
        },
        client_ip="10.0.0.5",  # what the server observed on the WS
    )
    # Server should record the client-supplied address.
    assert "remote-node" in server.peers
    peer = server.peers["remote-node"]
    assert peer.address == "203.0.113.42", (
        f"server should honor client-supplied address; got "
        f"{peer.address!r}"
    )


@pytest.mark.asyncio
async def test_server_falls_back_to_client_ip_when_no_address(
    tmp_path,
):
    """Back-compat: no `address` field in register → server uses
    client_ip (pre-sprint-566 behavior preserved for old clients)."""
    from prsm.bootstrap.server import BootstrapServer
    from prsm.bootstrap.config import BootstrapConfig

    config = BootstrapConfig(
        host="127.0.0.1", port=18765, api_port=18000,
    )
    server = BootstrapServer(config=config)

    fake_ws = MagicMock()
    fake_ws.send = AsyncMock()

    await server._handle_register(
        websocket=fake_ws,
        data={
            "peer_id": "legacy-node",
            "port": 9001,
            # no `address` field — legacy client
            "capabilities": [],
        },
        client_ip="10.0.0.5",
    )
    peer = server.peers["legacy-node"]
    assert peer.address == "10.0.0.5", (
        "back-compat: legacy clients without `address` field "
        "still get their client_ip recorded as before"
    )


@pytest.mark.asyncio
async def test_server_rejects_non_string_address(tmp_path):
    """Defensive: if a malicious client sends `address: 12345` or
    `address: null`, server falls back to client_ip rather than
    storing garbage."""
    from prsm.bootstrap.server import BootstrapServer
    from prsm.bootstrap.config import BootstrapConfig

    config = BootstrapConfig(
        host="127.0.0.1", port=18765, api_port=18000,
    )
    server = BootstrapServer(config=config)

    fake_ws = MagicMock()
    fake_ws.send = AsyncMock()

    for bad_value in [None, 12345, "", {"x": 1}, []]:
        await server._handle_register(
            websocket=fake_ws,
            data={
                "peer_id": f"node-{type(bad_value).__name__}",
                "port": 9001,
                "address": bad_value,
                "capabilities": [],
            },
            client_ip="10.0.0.5",
        )
        peer = server.peers[
            f"node-{type(bad_value).__name__}"
        ]
        assert peer.address == "10.0.0.5", (
            f"bad address value {bad_value!r} should fall back to "
            f"client_ip; got {peer.address!r}"
        )


# ── Libp2pDiscovery: env-var plumbing ────────────────────


def test_libp2p_discovery_reads_advertise_address_env(monkeypatch):
    """``_resolve_advertise_address()`` returns the env-var value
    when set, None when unset. This is the bridge between
    operator config and the BootstrapClient.advertise_address arg."""
    from prsm.node.libp2p_discovery import _resolve_advertise_address

    monkeypatch.delenv("PRSM_ADVERTISE_ADDRESS", raising=False)
    assert _resolve_advertise_address() is None

    monkeypatch.setenv(
        "PRSM_ADVERTISE_ADDRESS", "159.203.129.218:9001",
    )
    assert _resolve_advertise_address() == "159.203.129.218:9001"


def test_libp2p_discovery_strips_whitespace(monkeypatch):
    """Env-var values often have stray whitespace from shell
    quoting; the helper strips it."""
    from prsm.node.libp2p_discovery import _resolve_advertise_address

    monkeypatch.setenv(
        "PRSM_ADVERTISE_ADDRESS", "  159.203.129.218:9001  \n",
    )
    assert _resolve_advertise_address() == "159.203.129.218:9001"


def test_libp2p_discovery_empty_env_treated_as_unset(monkeypatch):
    """Empty string env var is treated the same as unset (some
    shells default unset env vars to empty)."""
    from prsm.node.libp2p_discovery import _resolve_advertise_address

    monkeypatch.setenv("PRSM_ADVERTISE_ADDRESS", "")
    assert _resolve_advertise_address() is None
