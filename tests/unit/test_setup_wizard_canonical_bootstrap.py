"""Sprint 148 — setup wizard's default bootstrap list must match
canonical DEFAULT_BOOTSTRAP_NODES from prsm.node.config.

Pre-fix the wizard hardcoded:
  /dns4/bootstrap1.prsm.network/tcp/9001/p2p/QmPRSM1
  /dns4/bootstrap2.prsm.network/tcp/9001/p2p/QmPRSM2

Both wrong: (a) domain is `prsm.network` instead of the live
`prsm-network.com`; (b) format is `/dns4/.../tcp/9001/p2p/...`
multiaddr instead of `wss://...:8765` URL the bootstrap server
actually serves on; (c) PeerID `QmPRSM1` is a placeholder, not
the real PeerID. Net effect: every operator running the wizard
got config pointing at a non-existent host.

Dogfooded /status on Base Sepolia 2026-05-10 surfaced the bug
through `peers.bootstrap.bootstrap_nodes` echoing the placeholder
back to the operator with attempted=2 / connected=0 / degraded=true.
"""
from __future__ import annotations

from prsm.cli_modules import setup_wizard
from prsm.node.config import DEFAULT_BOOTSTRAP_NODES


def test_setup_wizard_default_bootstrap_matches_canonical():
    """The wizard's exported default must equal the canonical list
    from prsm.node.config — the two files were drifting and the
    wizard's wrong defaults were leaking into operator configs."""
    assert hasattr(setup_wizard, "DEFAULT_BOOTSTRAP_NODES")
    assert (
        setup_wizard.DEFAULT_BOOTSTRAP_NODES
        == list(DEFAULT_BOOTSTRAP_NODES)
    )


def test_canonical_default_uses_correct_domain():
    """Belt-and-suspenders — the canonical list itself must point
    at `prsm-network.com` (not `prsm.network`) and use the wss
    URL scheme the bootstrap server actually serves on.
    """
    assert len(DEFAULT_BOOTSTRAP_NODES) >= 1
    primary = DEFAULT_BOOTSTRAP_NODES[0]
    assert "prsm-network.com" in primary
    assert primary.startswith("wss://")
    assert ":8765" in primary
