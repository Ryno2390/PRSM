"""Sprint 149 — auto-migrate stale wizard-generated bootstrap_nodes
in config.yaml to canonical DEFAULT_BOOTSTRAP_NODES.

Sprint 148 fixed the setup wizard to write canonical bootstrap
defaults. But every operator who ran the wizard BEFORE 148 has
a config.yaml pinning the broken
  /dns4/bootstrap1.prsm.network/tcp/9001/p2p/QmPRSM1
  /dns4/bootstrap2.prsm.network/tcp/9001/p2p/QmPRSM2

list — non-existent host, wrong domain (.network vs -network.com),
wrong format (multiaddr vs wss URL), placeholder PeerID. Without
a migration these operators silently keep getting the broken list
on every restart.

Migration policy: detect the EXACT broken list and replace with
canonical. Do NOT touch operator-customized lists (single
custom node, partial override, etc.).
"""
from __future__ import annotations

import yaml
from pathlib import Path

from prsm.node.config import DEFAULT_BOOTSTRAP_NODES, NodeConfig


_BROKEN_OLD_BOOTSTRAP = [
    "/dns4/bootstrap1.prsm.network/tcp/9001/p2p/QmPRSM1",
    "/dns4/bootstrap2.prsm.network/tcp/9001/p2p/QmPRSM2",
]


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_legacy_broken_defaults_migrate_to_canonical(tmp_path):
    """A config.yaml pinning the wizard's pre-148 broken defaults
    silently migrates in-memory to canonical."""
    p = _write_yaml(tmp_path, {
        "display_name": "test",
        "node_role": "full",
        "bootstrap_nodes": _BROKEN_OLD_BOOTSTRAP,
    })
    cfg = NodeConfig._load_from_yaml_path(p)
    assert cfg.bootstrap_nodes == list(DEFAULT_BOOTSTRAP_NODES)
    # Sanity: canonical list is genuinely different from the legacy
    assert cfg.bootstrap_nodes != _BROKEN_OLD_BOOTSTRAP


def test_operator_customized_bootstrap_preserved(tmp_path):
    """Operator-supplied list (anything other than the exact broken
    legacy match) flows through unchanged — never touch user data."""
    custom = ["wss://my-private-bootstrap.example.com:9999"]
    p = _write_yaml(tmp_path, {
        "display_name": "test",
        "node_role": "full",
        "bootstrap_nodes": custom,
    })
    cfg = NodeConfig._load_from_yaml_path(p)
    assert cfg.bootstrap_nodes == custom


def test_partial_legacy_match_not_migrated(tmp_path):
    """Partial-match (operator removed one of the two legacy entries
    or reordered) is treated as an opt-in custom list and preserved.
    Migration is conservative — only the EXACT pre-148 broken list
    is migrated, since that's the only one we know is wrong by
    construction."""
    partial = [_BROKEN_OLD_BOOTSTRAP[0]]  # just the first entry
    p = _write_yaml(tmp_path, {
        "display_name": "test",
        "node_role": "full",
        "bootstrap_nodes": partial,
    })
    cfg = NodeConfig._load_from_yaml_path(p)
    assert cfg.bootstrap_nodes == partial


def test_no_bootstrap_field_uses_dataclass_default(tmp_path):
    """yaml with no bootstrap_nodes key → NodeConfig dataclass
    default kicks in (canonical list, by sprint 148 source-of-truth)."""
    p = _write_yaml(tmp_path, {
        "display_name": "test",
        "node_role": "full",
    })
    cfg = NodeConfig._load_from_yaml_path(p)
    assert cfg.bootstrap_nodes == list(DEFAULT_BOOTSTRAP_NODES)


def test_canonical_list_passes_through_unchanged(tmp_path):
    """yaml ALREADY canonical (post-148 wizard run) → no-op
    migration, list preserved as-is."""
    p = _write_yaml(tmp_path, {
        "display_name": "test",
        "node_role": "full",
        "bootstrap_nodes": list(DEFAULT_BOOTSTRAP_NODES),
    })
    cfg = NodeConfig._load_from_yaml_path(p)
    assert cfg.bootstrap_nodes == list(DEFAULT_BOOTSTRAP_NODES)
