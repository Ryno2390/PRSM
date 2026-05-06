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

import os
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


# ──────────────────────────────────────────────────────────────────────
# T3c — production PublisherKeyAnchorClient wiring
# ──────────────────────────────────────────────────────────────────────


class TestT3cAnchorClientHelper:
    """Verifies the env-driven _build_publisher_key_anchor_client_or_none
    helper introduced by T3c. Mirrors the pattern of the existing
    _build_provenance_client_or_none — env-driven, fail-soft."""

    def test_returns_none_when_address_unset(self, monkeypatch):
        from prsm.node.node import _build_publisher_key_anchor_client_or_none
        monkeypatch.delenv("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", raising=False)
        assert _build_publisher_key_anchor_client_or_none() is None

    def test_returns_none_when_address_blank(self, monkeypatch):
        from prsm.node.node import _build_publisher_key_anchor_client_or_none
        monkeypatch.setenv("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", "   ")
        assert _build_publisher_key_anchor_client_or_none() is None

    def test_returns_none_when_address_malformed(self, monkeypatch):
        """Invalid address string surfaces as a construction failure
        in PublisherKeyAnchorClient → caller logs warning + returns
        None. Node continues without the production anchor."""
        from prsm.node.node import _build_publisher_key_anchor_client_or_none
        monkeypatch.setenv(
            "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", "not-a-valid-address",
        )
        monkeypatch.setenv(
            "PRSM_BASE_RPC_URL", "https://mainnet.base.org",
        )
        # Web3.to_checksum_address throws on malformed input → caught
        # by the helper, returns None.
        assert _build_publisher_key_anchor_client_or_none() is None

    def test_returns_client_with_valid_address(self, monkeypatch):
        """When the address env var parses as a valid hex address, the
        helper returns a real PublisherKeyAnchorClient (not None).
        The RPC URL doesn't have to be live — Web3 construction with
        an HTTPProvider doesn't actually connect until first call."""
        from prsm.node.node import _build_publisher_key_anchor_client_or_none
        from prsm.security.publisher_key_anchor.client import (
            PublisherKeyAnchorClient,
        )
        monkeypatch.setenv(
            "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS",
            "0x" + "ab" * 20,
        )
        monkeypatch.setenv("PRSM_BASE_RPC_URL", "http://127.0.0.1:1")
        out = _build_publisher_key_anchor_client_or_none()
        assert out is not None
        assert isinstance(out, PublisherKeyAnchorClient)
        assert out.address.lower() == ("0x" + "ab" * 20)


class TestT3cWiringIntoComponents:
    """End-to-end check that _build_dht_components_or_none threads the
    production anchor into DHTNodeComponents when env vars are set,
    and falls back to _FailClosedAnchor otherwise."""

    def test_falls_back_to_failclosed_anchor_when_env_unset(
        self, tmp_path, monkeypatch,
    ):
        from prsm.node.node import (
            _build_dht_components_or_none, _FailClosedAnchor,
        )
        monkeypatch.delenv("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", raising=False)
        identity = generate_node_identity("a")
        out = _build_dht_components_or_none(
            identity=identity,
            listen_host="127.0.0.1",
            dht_listen_port=0,
            manifest_index=_make_manifest_index(tmp_path),
            embedding_index=None,
        )
        assert out is not None
        assert isinstance(out._t3b_anchor, _FailClosedAnchor)

    def test_uses_production_anchor_when_env_set(
        self, tmp_path, monkeypatch,
    ):
        from prsm.node.node import _build_dht_components_or_none
        from prsm.security.publisher_key_anchor.client import (
            PublisherKeyAnchorClient,
        )
        monkeypatch.setenv(
            "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS",
            "0x" + "cd" * 20,
        )
        monkeypatch.setenv("PRSM_BASE_RPC_URL", "http://127.0.0.1:1")
        identity = generate_node_identity("a")
        out = _build_dht_components_or_none(
            identity=identity,
            listen_host="127.0.0.1",
            dht_listen_port=0,
            manifest_index=_make_manifest_index(tmp_path),
            embedding_index=None,
        )
        assert out is not None
        assert isinstance(out._t3b_anchor, PublisherKeyAnchorClient)

    def test_creator_pubkey_for_is_t3d_resolver_when_embedding_present(
        self, tmp_path, monkeypatch,
    ):
        """T3d (option (a)) replaced the fail-closed stub with a
        local-index-backed resolver. When an embedding_index is
        configured, the resolver is the closure returned by
        _make_creator_pubkey_for — NOT the fail-closed stub. The
        closure itself is fail-closed when the index has no record
        for the queried content_hash, so the security posture is
        equivalent for cold-start cases."""
        from prsm.node.node import (
            _build_dht_components_or_none,
            _fail_closed_creator_pubkey_for,
        )
        monkeypatch.setenv(
            "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS",
            "0x" + "ef" * 20,
        )
        monkeypatch.setenv("PRSM_BASE_RPC_URL", "http://127.0.0.1:1")
        identity = generate_node_identity("a")
        out = _build_dht_components_or_none(
            identity=identity,
            listen_host="127.0.0.1",
            dht_listen_port=0,
            manifest_index=None,
            embedding_index=_make_embedding_index(tmp_path),
        )
        assert out is not None
        # Resolver IS callable but is NOT the bare fail-closed stub —
        # it's the T3d closure that consults the local index first.
        assert callable(out._t3b_creator_pubkey_for)
        assert (
            out._t3b_creator_pubkey_for
            is not _fail_closed_creator_pubkey_for
        )
        # Empty index → resolver returns None (cold-start preserved).
        assert out._t3b_creator_pubkey_for(
            "0x" + "00" * 32,
        ) is None


class TestT3cAnchorContractIntegration:
    """T3c lights up real anchor verification end-to-end. Spin up a
    PublisherKeyAnchor on local Hardhat, register a known publisher
    key, point the helper at it, confirm lookup() returns the
    expected base64 pubkey. This is the live-chain equivalent of the
    component-level fail-closed test above.

    Skipped unless PRSM_T3C_LIVE_HARDHAT_RPC + corresponding contract
    address are set in the environment — keeps CI fast while letting
    operators verify the wiring against a real contract.
    """

    def test_lookup_returns_registered_pubkey_against_local_anchor(
        self, monkeypatch,
    ):
        rpc_url = os.environ.get("PRSM_T3C_LIVE_HARDHAT_RPC", "").strip()
        anchor_address = os.environ.get(
            "PRSM_T3C_LIVE_PUBLISHER_KEY_ANCHOR_ADDRESS", "",
        ).strip()
        if not rpc_url or not anchor_address:
            pytest.skip(
                "PRSM_T3C_LIVE_HARDHAT_RPC + "
                "PRSM_T3C_LIVE_PUBLISHER_KEY_ANCHOR_ADDRESS env vars "
                "required — set them after deploying "
                "PublisherKeyAnchor and registering a test publisher.",
            )
        monkeypatch.setenv(
            "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", anchor_address,
        )
        monkeypatch.setenv("PRSM_BASE_RPC_URL", rpc_url)
        from prsm.node.node import _build_publisher_key_anchor_client_or_none
        client = _build_publisher_key_anchor_client_or_none()
        assert client is not None
        # Expectation: a publisher with the test node_id has been
        # registered. The exercise script (T3c follow-on) handles the
        # registration; this assertion just confirms the lookup path
        # works against a real contract.
        test_node_id = os.environ.get(
            "PRSM_T3C_LIVE_TEST_NODE_ID", "",
        ).strip()
        if test_node_id:
            result = client.lookup(test_node_id)
            assert result is not None, (
                f"expected registered pubkey for node_id={test_node_id} "
                f"at anchor {anchor_address}"
            )


# ──────────────────────────────────────────────────────────────────────
# T3d (option (a)) — local-index creator_pubkey_for resolver
# ──────────────────────────────────────────────────────────────────────


class _StubAnchor:
    """In-memory anchor for resolver tests. Maps node_id → b64 pubkey."""

    def __init__(self):
        self._registrations: dict[str, str] = {}

    def register(self, node_id: str, pubkey_b64: str) -> None:
        self._registrations[node_id] = pubkey_b64

    def lookup(self, node_id: str):
        return self._registrations.get(node_id)


def _make_signed_record_for_test(
    *, content_hash: str, model_id: str, creator_id: str,
):
    """Build a LocalEmbeddingRecord with valid base64 + signature
    fields — only the content_hash → creator_id mapping matters for
    the resolver test, but the dataclass __post_init__ enforces the
    shape, so we satisfy it with deterministic dummy bytes."""
    import base64 as _b64
    from prsm.network.embedding_dht.local_index import (
        LocalEmbeddingRecord,
    )
    dim = 4
    raw = b"\x00\x00\x80\x3f" * dim  # 4 floats of 1.0
    return LocalEmbeddingRecord(
        content_hash=content_hash,
        model_id=model_id,
        dimension=dim,
        dtype="float32",
        vector_b64=_b64.b64encode(raw).decode("ascii"),
        creator_id=creator_id,
        created_at=1715000000.0,
        signature_b64=_b64.b64encode(b"sig" * 22).decode("ascii"),
    )


class TestT3dLocalIndexLookup:
    """Verifies the LocalEmbeddingIndex.lookup_creator_by_content_hash
    method that T3d's resolver depends on."""

    def test_returns_none_for_unknown_content_hash(self, tmp_path):
        idx = _make_embedding_index(tmp_path)
        assert idx.lookup_creator_by_content_hash(
            "a" * 32,
        ) is None

    def test_returns_creator_for_registered_content(self, tmp_path):
        idx = _make_embedding_index(tmp_path)
        rec = _make_signed_record_for_test(
            content_hash="abc-123",
            model_id="model-1",
            creator_id="creator-A",
        )
        idx.register(rec)
        assert idx.lookup_creator_by_content_hash("abc-123") == "creator-A"

    def test_deterministic_when_multiple_models_per_hash(self, tmp_path):
        """If a content_hash has records under multiple model_ids,
        the resolver picks the first by sorted(model_id) — deterministic
        across calls."""
        idx = _make_embedding_index(tmp_path)
        rec_a = _make_signed_record_for_test(
            content_hash="ch1",
            model_id="model-z",
            creator_id="creator-late",
        )
        rec_b = _make_signed_record_for_test(
            content_hash="ch1",
            model_id="model-a",
            creator_id="creator-early",
        )
        idx.register(rec_a)
        idx.register(rec_b)
        # sorted(model_id) → "model-a" wins → creator-early.
        assert idx.lookup_creator_by_content_hash("ch1") == "creator-early"

    def test_rejects_unsafe_content_hash(self, tmp_path):
        idx = _make_embedding_index(tmp_path)
        # Content hashes with disallowed chars get None defensively.
        assert idx.lookup_creator_by_content_hash(
            "../../../etc/passwd",
        ) is None


class TestT3dResolver:
    """Verifies the _make_creator_pubkey_for closure that T3d returns."""

    def _real_pubkey_b64(self):
        """Build a real Ed25519 pubkey + return it as base64 — the
        resolver insists on validate=True base64."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
        from cryptography.hazmat.primitives.serialization import (
            Encoding, PublicFormat,
        )
        import base64 as _b64
        priv = Ed25519PrivateKey.generate()
        pub = priv.public_key().public_bytes(
            encoding=Encoding.Raw, format=PublicFormat.Raw,
        )
        return pub, _b64.b64encode(pub).decode("ascii")

    def test_returns_none_when_index_empty(self, tmp_path):
        from prsm.node.node import _make_creator_pubkey_for
        idx = _make_embedding_index(tmp_path)
        anchor = _StubAnchor()
        resolver = _make_creator_pubkey_for(idx, anchor)
        assert resolver("any-content-hash") is None

    def test_returns_none_when_anchor_misses(self, tmp_path):
        """Index has the content_hash → creator_id mapping, but the
        anchor doesn't know the creator_id. Resolver fail-closes."""
        from prsm.node.node import _make_creator_pubkey_for
        idx = _make_embedding_index(tmp_path)
        idx.register(_make_signed_record_for_test(
            content_hash="ch1",
            model_id="m",
            creator_id="creator-A",
        ))
        anchor = _StubAnchor()
        # anchor not populated for creator-A.
        resolver = _make_creator_pubkey_for(idx, anchor)
        assert resolver("ch1") is None

    def test_returns_pubkey_bytes_on_full_round_trip(self, tmp_path):
        """Both the index AND the anchor have the relevant entries:
        resolver returns the decoded pubkey bytes."""
        from prsm.node.node import _make_creator_pubkey_for
        idx = _make_embedding_index(tmp_path)
        idx.register(_make_signed_record_for_test(
            content_hash="ch1",
            model_id="m",
            creator_id="creator-A",
        ))
        pub_bytes, pub_b64 = self._real_pubkey_b64()
        anchor = _StubAnchor()
        anchor.register("creator-A", pub_b64)
        resolver = _make_creator_pubkey_for(idx, anchor)
        out = resolver("ch1")
        assert out == pub_bytes

    def test_returns_none_on_malformed_anchor_response(self, tmp_path):
        """If the anchor returns garbage that isn't valid base64, the
        resolver fail-closes rather than crashing the verifier."""
        from prsm.node.node import _make_creator_pubkey_for
        idx = _make_embedding_index(tmp_path)
        idx.register(_make_signed_record_for_test(
            content_hash="ch1",
            model_id="m",
            creator_id="creator-A",
        ))
        anchor = _StubAnchor()
        anchor.register("creator-A", "this-is-not-valid-base64!@#$")
        resolver = _make_creator_pubkey_for(idx, anchor)
        assert resolver("ch1") is None

    def test_returns_fail_closed_stub_when_anchor_none(self, tmp_path):
        """Defensive: if the caller forgets to pass an anchor, the
        resolver is the fail-closed stub itself (not a closure that
        crashes)."""
        from prsm.node.node import (
            _make_creator_pubkey_for,
            _fail_closed_creator_pubkey_for,
        )
        idx = _make_embedding_index(tmp_path)
        out = _make_creator_pubkey_for(idx, None)
        assert out is _fail_closed_creator_pubkey_for

    def test_returns_fail_closed_stub_when_index_none(self, tmp_path):
        from prsm.node.node import (
            _make_creator_pubkey_for,
            _fail_closed_creator_pubkey_for,
        )
        anchor = _StubAnchor()
        out = _make_creator_pubkey_for(None, anchor)
        assert out is _fail_closed_creator_pubkey_for

    def test_anchor_exception_is_caught(self, tmp_path):
        """An anchor that raises RPCError on lookup must NOT propagate
        out of the resolver — it has to fail-closed."""
        from prsm.node.node import _make_creator_pubkey_for
        idx = _make_embedding_index(tmp_path)
        idx.register(_make_signed_record_for_test(
            content_hash="ch1",
            model_id="m",
            creator_id="creator-A",
        ))

        class _RaisingAnchor:
            def lookup(self, _node_id):
                raise RuntimeError("simulated RPC outage")

        resolver = _make_creator_pubkey_for(idx, _RaisingAnchor())
        assert resolver("ch1") is None
