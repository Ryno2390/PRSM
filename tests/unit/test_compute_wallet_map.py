"""Sprint 240 — source_agent_pubkey → FTNS wallet resolution.

Closes the LAST Vision §13 deferred sub-item:
"source_agent_pubkey → FTNS wallet address resolution (v1 uses
hex pubkey as wallet identifier; production needs a node-id →
wallet mapping registry)".

The compute_split_amounts output passes
`source_agent_pubkey_hex` as the recipient_id straight to
PaymentEscrow.release_escrow_split, which treats that string as
a wallet ID. In production an operator running N compute agents
has ONE FTNS wallet that should receive compensation for all N
pubkeys' work — not N separate wallets indexed by raw pubkey.

This sprint adds an opt-in mapping layer:
  - New `prsm.node.compute_wallet_map.ComputeWalletMap` loads a
    JSON pubkey_hex → wallet_id map from
    PRSM_COMPUTE_WALLET_MAP_FILE.
  - `resolve(pubkey_hex)` returns the mapped wallet_id when
    present, else the raw pubkey_hex (v1 fallback preserves
    backward compat).
  - Settlement in api.py calls .resolve() on each split's
    recipient_id BEFORE handing to release_escrow_split.

Fail-soft throughout: missing file = empty map = uniform
fallback behavior. Corrupt file logs + uses empty map.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from prsm.node.compute_wallet_map import ComputeWalletMap


def test_unset_env_yields_empty_map():
    """No env var set = empty map = identity-resolve everything."""
    m = ComputeWalletMap.from_env(env={})
    assert m.resolve("abc") == "abc"
    assert m.resolve("def") == "def"


def test_resolve_mapped_pubkey(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps({
        "01" * 32: "operator-1-wallet",
        "02" * 32: "operator-1-wallet",  # same wallet, 2 agents
        "03" * 32: "operator-2-wallet",
    }))
    m = ComputeWalletMap.from_env(env={
        "PRSM_COMPUTE_WALLET_MAP_FILE": str(path),
    })
    assert m.resolve("01" * 32) == "operator-1-wallet"
    assert m.resolve("02" * 32) == "operator-1-wallet"
    assert m.resolve("03" * 32) == "operator-2-wallet"


def test_unmapped_pubkey_returns_self(tmp_path):
    """Backwards-compat: pubkey not in the map falls through."""
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"01" * 32: "wallet-a"}))
    m = ComputeWalletMap.from_env(env={
        "PRSM_COMPUTE_WALLET_MAP_FILE": str(path),
    })
    assert m.resolve("ff" * 32) == "ff" * 32


def test_missing_file_fail_soft(tmp_path):
    """Operator points env at a non-existent file → empty map,
    no crash."""
    path = tmp_path / "does-not-exist.json"
    m = ComputeWalletMap.from_env(env={
        "PRSM_COMPUTE_WALLET_MAP_FILE": str(path),
    })
    # Acts as empty map.
    assert m.resolve("abc") == "abc"


def test_corrupt_file_fail_soft(tmp_path):
    """Garbage JSON → empty map, no crash."""
    path = tmp_path / "map.json"
    path.write_text("{not valid json")
    m = ComputeWalletMap.from_env(env={
        "PRSM_COMPUTE_WALLET_MAP_FILE": str(path),
    })
    assert m.resolve("abc") == "abc"


def test_wrong_shape_fail_soft(tmp_path):
    """JSON valid but not a dict → empty map."""
    path = tmp_path / "map.json"
    path.write_text(json.dumps(["wrong", "shape"]))
    m = ComputeWalletMap.from_env(env={
        "PRSM_COMPUTE_WALLET_MAP_FILE": str(path),
    })
    assert m.resolve("abc") == "abc"


def test_non_string_values_filtered(tmp_path):
    """Mapping with non-string values is silently filtered."""
    path = tmp_path / "map.json"
    path.write_text(json.dumps({
        "01" * 32: "good-wallet",
        "02" * 32: 12345,  # ignored
        "03" * 32: None,   # ignored
    }))
    m = ComputeWalletMap.from_env(env={
        "PRSM_COMPUTE_WALLET_MAP_FILE": str(path),
    })
    assert m.resolve("01" * 32) == "good-wallet"
    assert m.resolve("02" * 32) == "02" * 32  # falls through
    assert m.resolve("03" * 32) == "03" * 32


def test_split_recipient_resolution_via_helper():
    """The api.py wrapper that resolves a splits list pre-flight
    should rewrite each recipient via .resolve()."""
    from prsm.node.compute_wallet_map import resolve_splits

    m = ComputeWalletMap.from_mapping({
        "agent-a": "wallet-1",
        "agent-b": "wallet-1",  # same operator
        "agent-c": "wallet-2",
    })
    raw_splits = [
        ("aggregator-7", 5.0),
        ("agent-a", 10.0),
        ("agent-b", 20.0),
        ("agent-c", 30.0),
        ("agent-unknown", 35.0),
    ]
    resolved = resolve_splits(raw_splits, m)
    # Recipients pass through .resolve(); amounts unchanged.
    assert resolved == [
        ("aggregator-7", 5.0),    # not in map → self
        ("wallet-1", 10.0),
        ("wallet-1", 20.0),
        ("wallet-2", 30.0),
        ("agent-unknown", 35.0),  # not in map → self
    ]
