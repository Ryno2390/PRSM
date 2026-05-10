"""Sprint 150 — Discovery._try_bootstrap_client must report the
real package version when registering with the bootstrap server.

Pre-fix discovery.py:228 hardcoded `version="0.24.0"`. Even after
shipping v1.x.x the dogfood node identified itself as v0.24.0 to
the bootstrap server, which:

  1. Triggers compatibility-band warnings on the server side
     (server may refuse old clients or downgrade their feature
     set).
  2. Surfaces wrong client-version metric in the operator's
     own logs.
  3. Sows confusion when correlating bootstrap-server logs with
     client-side state.

The fix references `prsm.__version__` so the version reported
on the wire stays in sync with the real package version
forever-after.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import prsm
from prsm.node.discovery import PeerDiscovery


def _make_discovery():
    """Tiny PeerDiscovery enough to call _try_bootstrap_client."""
    transport = MagicMock()
    transport.identity.node_id = "test-node-id"
    transport.port = 8000
    transport.on_message = MagicMock()
    discovery = PeerDiscovery(
        transport=transport,
        bootstrap_nodes=["wss://bootstrap.example.com:8765"],
    )
    discovery._local_capabilities = ["compute"]
    return discovery


def test_bootstrap_client_uses_runtime_version():
    """Sprint 150 — version arg passed to BootstrapClient must be
    the current package version, not a stale hardcoded literal."""
    discovery = _make_discovery()

    captured: dict = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def connect(self):
            return []

        async def start_heartbeat(self):
            pass

    with patch(
        "prsm.bootstrap.client.BootstrapClient", _FakeClient,
    ):
        result = asyncio.run(discovery._try_bootstrap_client())

    assert "version" in captured
    assert captured["version"] == prsm.__version__
    # Belt-and-suspenders: the literal pre-fix value must NOT
    # be reachable through any future regression.
    assert captured["version"] != "0.24.0"
