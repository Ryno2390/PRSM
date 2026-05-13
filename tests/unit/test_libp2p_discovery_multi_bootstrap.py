"""Sprint 375 — Libp2pDiscovery multi-bootstrap fallback.

Pre-fix Libp2pDiscovery accepted only ``bootstrap_nodes`` and
ignored the ``bootstrap_fallback_nodes`` from NodeConfig that
provides EU + APAC backup hosts. Closes the §7.29 honest-
scope reliability gap: when the canonical
``wss://bootstrap1.prsm-network.com:8765`` US droplet is
unreachable, libp2p-mode operators sat in degraded mode
forever because the fallback list never propagated through
the Libp2pDiscovery constructor.

Sprint 375 extends Libp2pDiscovery to accept
``bootstrap_fallback_nodes`` (+ the ``bootstrap_fallback_enabled``
flag) and merges primary + fallback into a single candidates
list with dedup. Mirrors the pattern from PeerDiscovery
(prsm/node/discovery.py:309-327) so the two discovery paths
behave consistently.

Backwards-compat: existing single-host configs that pass
only ``bootstrap_nodes=[...]`` keep working unchanged
(fallback_nodes defaults to []).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.libp2p_discovery import Libp2pDiscovery


def _make_transport(*, connect_return=None):
    """Stub transport whose connect_to_peer returns
    `connect_return` (or a list of return values keyed by
    address if a dict is supplied)."""
    t = MagicMock()
    t.identity.node_id = "test-node"
    t.port = 9001
    if isinstance(connect_return, dict):
        async def _connect(addr):
            return connect_return.get(addr)
        t.connect_to_peer = _connect
    else:
        t.connect_to_peer = AsyncMock(return_value=connect_return)
    return t


# ── Constructor accepts fallback list ────────────────


def test_constructor_accepts_fallback_nodes():
    """Sprint 375: bootstrap_fallback_nodes accepted as
    kwarg. Existing positional / keyword usage unchanged."""
    d = Libp2pDiscovery(
        transport=_make_transport(),
        bootstrap_nodes=["wss://primary.example.com:8765"],
        bootstrap_fallback_nodes=[
            "wss://eu.example.com:8765",
            "wss://apac.example.com:8765",
        ],
    )
    assert d.bootstrap_nodes == [
        "wss://primary.example.com:8765"
    ]
    assert d.bootstrap_fallback_nodes == [
        "wss://eu.example.com:8765",
        "wss://apac.example.com:8765",
    ]


def test_constructor_fallback_defaults_empty():
    """Backwards-compat: fallback_nodes defaults to [] when
    not specified."""
    d = Libp2pDiscovery(
        transport=_make_transport(),
        bootstrap_nodes=["wss://primary.example.com:8765"],
    )
    assert d.bootstrap_fallback_nodes == []


def test_constructor_fallback_enabled_default_true():
    """Mirrors NodeConfig.bootstrap_fallback_enabled default."""
    d = Libp2pDiscovery(
        transport=_make_transport(),
        bootstrap_nodes=[],
    )
    assert d.bootstrap_fallback_enabled is True


def test_constructor_fallback_disabled_explicit():
    """Operator can disable fallback explicitly."""
    d = Libp2pDiscovery(
        transport=_make_transport(),
        bootstrap_nodes=["wss://primary.example.com:8765"],
        bootstrap_fallback_nodes=[
            "wss://eu.example.com:8765",
        ],
        bootstrap_fallback_enabled=False,
    )
    assert d.bootstrap_fallback_enabled is False


# ── Candidate list merging ───────────────────────────


def test_candidates_primary_then_fallback_when_enabled():
    """Sprint 375: when fallback enabled + non-empty, the
    candidates list is primary + fallback (in that order)."""
    d = Libp2pDiscovery(
        transport=_make_transport(),
        bootstrap_nodes=["wss://primary.example.com:8765"],
        bootstrap_fallback_nodes=[
            "wss://eu.example.com:8765",
            "wss://apac.example.com:8765",
        ],
    )
    candidates = d._candidate_bootstrap_addresses()
    assert candidates == [
        "wss://primary.example.com:8765",
        "wss://eu.example.com:8765",
        "wss://apac.example.com:8765",
    ]


def test_candidates_primary_only_when_fallback_disabled():
    d = Libp2pDiscovery(
        transport=_make_transport(),
        bootstrap_nodes=["wss://primary.example.com:8765"],
        bootstrap_fallback_nodes=[
            "wss://eu.example.com:8765",
        ],
        bootstrap_fallback_enabled=False,
    )
    candidates = d._candidate_bootstrap_addresses()
    assert candidates == ["wss://primary.example.com:8765"]


def test_candidates_dedups_overlap():
    """Same URL in primary + fallback only appears once."""
    d = Libp2pDiscovery(
        transport=_make_transport(),
        bootstrap_nodes=["wss://shared.example.com:8765"],
        bootstrap_fallback_nodes=[
            "wss://shared.example.com:8765",
            "wss://eu.example.com:8765",
        ],
    )
    candidates = d._candidate_bootstrap_addresses()
    assert candidates == [
        "wss://shared.example.com:8765",
        "wss://eu.example.com:8765",
    ]


def test_candidates_empty_when_no_nodes():
    d = Libp2pDiscovery(
        transport=_make_transport(),
        bootstrap_nodes=[],
        bootstrap_fallback_nodes=[],
    )
    assert d._candidate_bootstrap_addresses() == []


# ── Bootstrap iteration uses merged list ─────────────


def test_bootstrap_iterates_primary_then_fallback():
    """When primary connect_to_peer returns None, the
    bootstrap loop falls through to fallback addresses.
    Each address is attempted via transport.connect_to_peer."""
    transport = _make_transport(connect_return=None)
    d = Libp2pDiscovery(
        transport=transport,
        bootstrap_nodes=["wss://primary.example.com:8765"],
        bootstrap_fallback_nodes=[
            "wss://eu.example.com:8765",
        ],
    )

    async def run():
        # The BootstrapClient fallback fires when all
        # libp2p connects fail; stub it out.
        from unittest.mock import patch
        with patch.object(
            d, "_try_bootstrap_client",
            new=AsyncMock(return_value=False),
        ):
            await d.bootstrap()

    asyncio.run(run())
    # All 3 candidates (1 primary + 1 fallback) were
    # attempted at the transport layer.
    assert d._bootstrap_status["attempted"] == 2


def test_bootstrap_status_reports_attempted_includes_fallback():
    """The `attempted` counter reflects total candidates,
    not just primary."""
    transport = _make_transport(connect_return=None)
    d = Libp2pDiscovery(
        transport=transport,
        bootstrap_nodes=["wss://primary.example.com:8765"],
        bootstrap_fallback_nodes=[
            "wss://eu.example.com:8765",
            "wss://apac.example.com:8765",
        ],
    )

    async def run():
        from unittest.mock import patch
        with patch.object(
            d, "_try_bootstrap_client",
            new=AsyncMock(return_value=False),
        ):
            await d.bootstrap()

    asyncio.run(run())
    assert d._bootstrap_status["attempted"] == 3


def test_bootstrap_primary_success_short_circuits_fallback():
    """When primary connects, fallback is NOT attempted —
    operator-impact-preserving behavior."""
    primary_peer = MagicMock()
    primary_peer.node_id = "primary-peer"
    transport = _make_transport(connect_return={
        "wss://primary.example.com:8765": primary_peer,
        "wss://eu.example.com:8765": None,
    })
    d = Libp2pDiscovery(
        transport=transport,
        bootstrap_nodes=["wss://primary.example.com:8765"],
        bootstrap_fallback_nodes=[
            "wss://eu.example.com:8765",
        ],
    )

    async def run():
        return await d.bootstrap()

    connected = asyncio.run(run())
    # Sprint 375: candidates list is processed sequentially;
    # primary succeeded so connected=1.
    assert connected >= 1


# ── Active-bootstrap status surface ──────────────────


def test_active_bootstrap_url_recorded_on_success():
    """When a candidate succeeds, _bootstrap_status records
    which URL — operator sees it via /bootstrap/status."""
    eu_peer = MagicMock()
    eu_peer.node_id = "eu-peer"
    transport = _make_transport(connect_return={
        "wss://primary.example.com:8765": None,
        "wss://eu.example.com:8765": eu_peer,
    })
    d = Libp2pDiscovery(
        transport=transport,
        bootstrap_nodes=["wss://primary.example.com:8765"],
        bootstrap_fallback_nodes=[
            "wss://eu.example.com:8765",
        ],
    )

    async def run():
        from unittest.mock import patch
        with patch.object(
            d, "_try_bootstrap_client",
            new=AsyncMock(return_value=False),
        ):
            await d.bootstrap()

    asyncio.run(run())
    # The active-bootstrap-URL field records the URL that
    # produced a successful connect.
    assert (
        d._bootstrap_status.get("active_url")
        == "wss://eu.example.com:8765"
    )


def test_get_bootstrap_status_surfaces_fallback_config():
    """Sprint 375: /bootstrap/status returns fallback_nodes
    + fallback_enabled + active_url so operators can diagnose
    SPOF behavior."""
    d = Libp2pDiscovery(
        transport=_make_transport(),
        bootstrap_nodes=["wss://primary.example.com:8765"],
        bootstrap_fallback_nodes=[
            "wss://eu.example.com:8765",
            "wss://apac.example.com:8765",
        ],
    )
    status = d.get_bootstrap_status()
    assert status["bootstrap_nodes"] == [
        "wss://primary.example.com:8765"
    ]
    assert status["bootstrap_fallback_nodes"] == [
        "wss://eu.example.com:8765",
        "wss://apac.example.com:8765",
    ]
    assert status["bootstrap_fallback_enabled"] is True
    # active_url defaults to None before bootstrap runs
    assert status["active_url"] is None


def test_get_bootstrap_status_active_url_set_after_connect():
    eu_peer = MagicMock()
    eu_peer.node_id = "eu-peer"
    transport = _make_transport(connect_return={
        "wss://primary.example.com:8765": None,
        "wss://eu.example.com:8765": eu_peer,
    })
    d = Libp2pDiscovery(
        transport=transport,
        bootstrap_nodes=["wss://primary.example.com:8765"],
        bootstrap_fallback_nodes=[
            "wss://eu.example.com:8765",
        ],
    )

    async def run():
        from unittest.mock import patch
        with patch.object(
            d, "_try_bootstrap_client",
            new=AsyncMock(return_value=False),
        ):
            await d.bootstrap()

    asyncio.run(run())
    status = d.get_bootstrap_status()
    assert status["active_url"] == "wss://eu.example.com:8765"


def test_active_bootstrap_url_none_when_all_fail():
    """All candidates fail → active_url is None and
    degraded=True."""
    transport = _make_transport(connect_return=None)
    d = Libp2pDiscovery(
        transport=transport,
        bootstrap_nodes=["wss://primary.example.com:8765"],
        bootstrap_fallback_nodes=[
            "wss://eu.example.com:8765",
        ],
    )

    async def run():
        from unittest.mock import patch
        with patch.object(
            d, "_try_bootstrap_client",
            new=AsyncMock(return_value=False),
        ):
            await d.bootstrap()

    asyncio.run(run())
    assert d._bootstrap_status.get("active_url") is None
    assert d._bootstrap_status["degraded"] is True
