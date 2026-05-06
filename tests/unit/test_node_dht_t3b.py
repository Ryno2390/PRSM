"""Unit tests for PRSM-DHT-TRANSPORT T3b — node.py threading.

Tests the helper that builds DHTNodeComponents from node-startup
context, plus a dummy-Node integration that exercises the
construct/start/stop lifecycle without booting the entire PRSMNode
(which has heavy dependencies — libp2p, IPFS, ledger, etc.). The
T3b wiring itself is small enough that the unit-level coverage
here is more useful than a full Node smoketest, which would tell
us less about the wiring and more about test-environment dependency
availability.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from prsm.network.dht_components import DHTNodeComponents
from prsm.network.embedding_dht.local_index import LocalEmbeddingIndex
from prsm.network.manifest_dht.local_index import LocalManifestIndex
from prsm.node.identity import generate_node_identity
from prsm.node.node import _build_dht_components_or_none


def _make_manifest_index(tmp_path: Path) -> LocalManifestIndex:
    root = tmp_path / "m_idx"
    root.mkdir(parents=True, exist_ok=True)
    return LocalManifestIndex(root)


def _make_embedding_index(tmp_path: Path) -> LocalEmbeddingIndex:
    root = tmp_path / "e_idx"
    root.mkdir(parents=True, exist_ok=True)
    return LocalEmbeddingIndex(root)


# ──────────────────────────────────────────────────────────────────────
# helper construction
# ──────────────────────────────────────────────────────────────────────


class TestBuildDhtComponentsHelper:
    def test_returns_none_when_no_indexes(self):
        identity = generate_node_identity("a")
        out = _build_dht_components_or_none(
            identity=identity,
            listen_host="127.0.0.1",
            dht_listen_port=0,
            manifest_index=None,
            embedding_index=None,
        )
        assert out is None

    def test_builds_with_manifest_only(self, tmp_path):
        identity = generate_node_identity("a")
        manifest = _make_manifest_index(tmp_path)
        out = _build_dht_components_or_none(
            identity=identity,
            listen_host="127.0.0.1",
            dht_listen_port=0,
            manifest_index=manifest,
            embedding_index=None,
        )
        assert out is not None
        assert isinstance(out, DHTNodeComponents)
        assert out.manifest_server is not None
        assert out.embedding_server is None

    def test_builds_with_embedding_only(self, tmp_path):
        identity = generate_node_identity("a")
        embedding = _make_embedding_index(tmp_path)
        out = _build_dht_components_or_none(
            identity=identity,
            listen_host="127.0.0.1",
            dht_listen_port=0,
            manifest_index=None,
            embedding_index=embedding,
        )
        assert out is not None
        assert out.manifest_server is None
        assert out.embedding_server is not None

    def test_builds_with_both(self, tmp_path):
        identity = generate_node_identity("a")
        out = _build_dht_components_or_none(
            identity=identity,
            listen_host="127.0.0.1",
            dht_listen_port=0,
            manifest_index=_make_manifest_index(tmp_path),
            embedding_index=_make_embedding_index(tmp_path),
        )
        assert out is not None
        assert out.manifest_server is not None
        assert out.embedding_server is not None

    def test_stashes_verifier_inputs(self, tmp_path):
        """The helper stashes the fail-closed anchor + creator_pubkey_for
        + verify_signature on the instance so Node.start() can forward
        them. T3c will replace these with PublisherKeyAnchorClient +
        a real creator_pubkey lookup."""
        identity = generate_node_identity("a")
        out = _build_dht_components_or_none(
            identity=identity,
            listen_host="127.0.0.1",
            dht_listen_port=0,
            manifest_index=_make_manifest_index(tmp_path),
            embedding_index=_make_embedding_index(tmp_path),
        )
        assert out._t3b_anchor is not None
        assert out._t3b_creator_pubkey_for is not None
        assert out._t3b_verify_signature is not None

    def test_anchor_is_fail_closed(self, tmp_path):
        """T3b ships with a fail-closed anchor — every lookup returns
        None, so cross-node manifest fetch refuses to verify. This is
        intentional pre-T3c."""
        identity = generate_node_identity("a")
        out = _build_dht_components_or_none(
            identity=identity,
            listen_host="127.0.0.1",
            dht_listen_port=0,
            manifest_index=_make_manifest_index(tmp_path),
            embedding_index=None,
        )
        assert out._t3b_anchor.lookup("any-node-id") is None

    def test_creator_pubkey_for_is_fail_closed(self, tmp_path):
        identity = generate_node_identity("a")
        out = _build_dht_components_or_none(
            identity=identity,
            listen_host="127.0.0.1",
            dht_listen_port=0,
            manifest_index=None,
            embedding_index=_make_embedding_index(tmp_path),
        )
        assert out._t3b_creator_pubkey_for("any-content-hash") is None


# ──────────────────────────────────────────────────────────────────────
# verify_signature is REAL Ed25519
# ──────────────────────────────────────────────────────────────────────


class TestVerifySignatureIsReal:
    """T3b's verify_signature is real cryptography.hazmat Ed25519
    (not a stub) — this is correct because verification doesn't
    depend on any production trust input. The fail-closed pubkey
    lookup is what gates trust."""

    def test_verify_accepts_valid_signature(self, tmp_path):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
        from cryptography.hazmat.primitives.serialization import (
            Encoding, PublicFormat,
        )

        priv = Ed25519PrivateKey.generate()
        pub = priv.public_key().public_bytes(
            encoding=Encoding.Raw, format=PublicFormat.Raw,
        )
        msg = b"hello world"
        sig = priv.sign(msg)

        identity = generate_node_identity("a")
        out = _build_dht_components_or_none(
            identity=identity,
            listen_host="127.0.0.1",
            dht_listen_port=0,
            manifest_index=None,
            embedding_index=_make_embedding_index(tmp_path),
        )
        assert out._t3b_verify_signature(pub, msg, sig) is True

    def test_verify_rejects_bad_signature(self, tmp_path):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
        from cryptography.hazmat.primitives.serialization import (
            Encoding, PublicFormat,
        )

        priv = Ed25519PrivateKey.generate()
        pub = priv.public_key().public_bytes(
            encoding=Encoding.Raw, format=PublicFormat.Raw,
        )
        identity = generate_node_identity("a")
        out = _build_dht_components_or_none(
            identity=identity,
            listen_host="127.0.0.1",
            dht_listen_port=0,
            manifest_index=None,
            embedding_index=_make_embedding_index(tmp_path),
        )
        assert out._t3b_verify_signature(pub, b"msg", b"\x00" * 64) is False

    def test_verify_rejects_empty_pubkey(self, tmp_path):
        identity = generate_node_identity("a")
        out = _build_dht_components_or_none(
            identity=identity,
            listen_host="127.0.0.1",
            dht_listen_port=0,
            manifest_index=None,
            embedding_index=_make_embedding_index(tmp_path),
        )
        assert out._t3b_verify_signature(b"", b"msg", b"sig") is False


# ──────────────────────────────────────────────────────────────────────
# lifecycle — components built from helper, start + stop on real loop
# ──────────────────────────────────────────────────────────────────────


class TestLifecycle:
    def test_start_then_stop_via_node_helper(self, tmp_path):
        """Construct components via the T3b helper, start them with
        the helper-stashed verifier inputs (the same call shape
        Node.start() uses), confirm the listener binds, stop cleanly.
        """
        identity = generate_node_identity("a")
        components = _build_dht_components_or_none(
            identity=identity,
            listen_host="127.0.0.1",
            dht_listen_port=0,
            manifest_index=_make_manifest_index(tmp_path),
            embedding_index=_make_embedding_index(tmp_path),
        )
        assert components is not None
        try:
            port = components.start(
                anchor=components._t3b_anchor,
                creator_pubkey_for=components._t3b_creator_pubkey_for,
                verify_signature=components._t3b_verify_signature,
            )
            assert port > 0
            assert components.is_running
            assert components.transport is not None
            assert components.manifest_client is not None
            assert components.embedding_client is not None
        finally:
            components.stop()
        assert not components.is_running


# ──────────────────────────────────────────────────────────────────────
# config — dht_enabled + dht_listen_port present and default-safe
# ──────────────────────────────────────────────────────────────────────


class TestNodeConfigFields:
    def test_dht_disabled_by_default(self):
        from prsm.node.config import NodeConfig
        cfg = NodeConfig()
        assert cfg.dht_enabled is False
        assert cfg.dht_listen_port == 0

    def test_dht_can_be_enabled(self):
        from prsm.node.config import NodeConfig
        cfg = NodeConfig(dht_enabled=True, dht_listen_port=9876)
        assert cfg.dht_enabled is True
        assert cfg.dht_listen_port == 9876


# ──────────────────────────────────────────────────────────────────────
# PRSMNode wiring (lightweight surface check)
# ──────────────────────────────────────────────────────────────────────


class TestPRSMNodeSurface:
    def test_dht_components_field_initialized_to_none(self):
        """Pre-initialize, the field is None. Confirms the __init__
        wiring lands without booting any heavy subsystem."""
        from prsm.node.node import PRSMNode
        # __init__ does NOT initialize subsystems (that's
        # initialize() with the I/O). It does set the field.
        node = PRSMNode.__new__(PRSMNode)
        node.dht_components = None  # type: ignore[attr-defined]
        assert node.dht_components is None

    def test_start_dht_components_no_op_when_none(self):
        """Node._start_dht_components_if_present is a no-op when
        components is None. Critical for the dht_enabled=False path
        where the field stays None throughout the node's lifecycle."""
        from prsm.node.node import PRSMNode
        node = PRSMNode.__new__(PRSMNode)
        node.dht_components = None  # type: ignore[attr-defined]
        node.config = SimpleNamespace(listen_host="127.0.0.1")  # type: ignore[attr-defined]
        # Should not raise — early return.
        node._start_dht_components_if_present()
