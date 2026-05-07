"""PRSM-PROV-1 Item 4 T4.9.next5 — server-side fingerprint storage + 2-node E2E.

Three layers of test:

1. ``_register_local_fingerprint``: ContentUploader bridges
   ``FingerprintRecord`` → ``LocalFingerprintRecord`` with a creator
   Ed25519 signature, persisted via ``LocalFingerprintIndex``.
   Skip-conditions match ``_register_local_embedding``.

2. ``DHTNodeComponents.build()`` accepts and threads
   ``local_fingerprint_index`` into ``EmbeddingDHTServer`` — without
   this, the server side returns NOT_FOUND for every fingerprint
   fetch even when clients are wired.

3. Two-node E2E mirroring ``TestTwoNodeEmbeddingE2E``: Node A
   registers a signed fingerprint; Node B finds providers + fetches
   over the full SyncDHTTransport → DHTListener → DHTRequestRouter →
   EmbeddingDHTServer → wire path; signature verification passes.
"""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding, PublicFormat,
)

from prsm.data.fingerprints import FingerprintKind, FingerprintRecord
from prsm.network.dht_components import DHTNodeComponents
from prsm.network.embedding_dht.local_fingerprint_index import (
    LocalFingerprintIndex,
)
from prsm.network.embedding_dht.local_index import LocalEmbeddingIndex
from prsm.network.embedding_dht.protocol import fingerprint_signing_payload
from prsm.node.content_uploader import ContentUploader
from prsm.node.identity import generate_node_identity
from prsm.node.transport_adapter import DirectAdapter


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────


def _real_verify_signature(
    pubkey_bytes: bytes, message: bytes, signature: bytes,
) -> bool:
    if not pubkey_bytes:
        return False
    try:
        Ed25519PublicKey.from_public_bytes(pubkey_bytes).verify(
            signature, message,
        )
    except InvalidSignature:
        return False
    return True


def _make_fp_index(tmp_path: Path, sub: str) -> LocalFingerprintIndex:
    root = tmp_path / sub
    root.mkdir(parents=True, exist_ok=True)
    return LocalFingerprintIndex(root)


def _make_emb_index(tmp_path: Path, sub: str) -> LocalEmbeddingIndex:
    root = tmp_path / sub
    root.mkdir(parents=True, exist_ok=True)
    return LocalEmbeddingIndex(root)


def _build_uploader(
    *, local_fingerprint_index=None,
) -> ContentUploader:
    return ContentUploader(
        identity=generate_node_identity("fp-test-node"),
        gossip=MagicMock(),
        ledger=MagicMock(),
        local_fingerprint_index=local_fingerprint_index,
    )


# ──────────────────────────────────────────────────────────────────────
# Layer 1 — _register_local_fingerprint
# ──────────────────────────────────────────────────────────────────────


class TestRegisterLocalFingerprint:
    def test_skipped_without_local_index(self):
        """No-op when local_fingerprint_index wasn't wired."""
        uploader = _build_uploader(local_fingerprint_index=None)
        record = FingerprintRecord(
            kind=FingerprintKind.IMAGE_PHASH, payload=b"\x01" * 8,
        )
        # Doesn't raise — just returns silently.
        uploader._register_local_fingerprint(record, "0x" + "ab" * 32)

    def test_skipped_without_provenance_hash(self, tmp_path):
        """No anchor → no peer can verify our sig → MUST NOT register."""
        idx = _make_fp_index(tmp_path, "fp")
        uploader = _build_uploader(local_fingerprint_index=idx)
        record = FingerprintRecord(
            kind=FingerprintKind.IMAGE_PHASH, payload=b"\x01" * 8,
        )
        uploader._register_local_fingerprint(record, None)
        assert len(idx) == 0

    def test_skipped_with_none_record(self, tmp_path):
        idx = _make_fp_index(tmp_path, "fp")
        uploader = _build_uploader(local_fingerprint_index=idx)
        uploader._register_local_fingerprint(None, "0x" + "ab" * 32)
        assert len(idx) == 0

    def test_register_success_produces_verifiable_record(self, tmp_path):
        """Happy path: a record gets signed with the creator's Ed25519
        private key over the canonical fingerprint_signing_payload, and
        persists in the LocalFingerprintIndex with a signature that
        verifies against the creator pubkey."""
        idx = _make_fp_index(tmp_path, "fp")
        uploader = _build_uploader(local_fingerprint_index=idx)
        content_hash = "0x" + "ab" * 32
        payload = b"\xde\xad\xbe\xef" * 2
        record = FingerprintRecord(
            kind=FingerprintKind.IMAGE_PHASH, payload=payload,
        )

        uploader._register_local_fingerprint(record, content_hash)
        assert len(idx) == 1
        stored = idx.lookup(content_hash, "image-phash")
        assert stored is not None
        assert stored.content_hash == content_hash
        assert stored.fingerprint_kind == "image-phash"
        assert base64.b64decode(stored.payload_b64) == payload

        # Verify the signature is real Ed25519 over the canonical
        # signing payload — what a peer would do to validate.
        sig_message = fingerprint_signing_payload(
            content_hash=content_hash,
            fingerprint_kind="image-phash",
            payload_bytes=payload,
            created_at=stored.created_at,
        )
        sig = base64.b64decode(stored.signature_b64)
        pubkey_bytes = base64.b64decode(uploader.identity.public_key_b64)
        assert _real_verify_signature(pubkey_bytes, sig_message, sig)

    def test_register_silent_on_unexpected_error(self, tmp_path):
        """A LocalFingerprintIndex.register() that raises must not
        propagate out — upload-critical path stays unbroken."""
        idx = _make_fp_index(tmp_path, "fp")

        class BoomIndex:
            def register(self, _r):
                raise RuntimeError("disk full")

        uploader = _build_uploader(local_fingerprint_index=BoomIndex())
        record = FingerprintRecord(
            kind=FingerprintKind.IMAGE_PHASH, payload=b"\x01" * 8,
        )
        # No exception raised.
        uploader._register_local_fingerprint(record, "0x" + "ab" * 32)


# ──────────────────────────────────────────────────────────────────────
# Layer 2 — DHTNodeComponents threads local_fingerprint_index
# ──────────────────────────────────────────────────────────────────────


class TestDHTComponentsFingerprintWiring:
    def test_build_threads_index_into_embedding_server(self, tmp_path):
        """Without this, the server side serves NOT_FOUND for every
        fingerprint fetch — clients ASK but no peer ANSWERS."""
        emb_index = _make_emb_index(tmp_path, "emb")
        fp_index = _make_fp_index(tmp_path, "fp")
        identity = generate_node_identity("dht-test")

        comp = DHTNodeComponents.build(
            my_node_id=identity.node_id,
            my_host="127.0.0.1",
            dht_listen_port=0,
            transport_adapter=DirectAdapter(),
            listen_host="127.0.0.1",
            local_embedding_index=emb_index,
            local_fingerprint_index=fp_index,
            creator_pubkey_for=lambda _h: None,
            verify_signature=_real_verify_signature,
        )
        assert comp.embedding_server is not None
        # Pin the wire-through: the server's local_fingerprint_index
        # must be the SAME instance we passed (not a copy or a None
        # fallback). Reaches into the private field by design — same
        # rationale as the lockstep wiring tests.
        assert (
            comp.embedding_server._local_fingerprint_index is fp_index
        )

    def test_build_without_fingerprint_index_keeps_server_field_none(
        self, tmp_path,
    ):
        """When local_fingerprint_index isn't passed, the server still
        constructs (backwards compat) but its fingerprint storage is
        None — fetches return NOT_FOUND, which is the safe default."""
        emb_index = _make_emb_index(tmp_path, "emb")
        identity = generate_node_identity("dht-test")
        comp = DHTNodeComponents.build(
            my_node_id=identity.node_id,
            my_host="127.0.0.1",
            dht_listen_port=0,
            transport_adapter=DirectAdapter(),
            listen_host="127.0.0.1",
            local_embedding_index=emb_index,
            creator_pubkey_for=lambda _h: None,
            verify_signature=_real_verify_signature,
        )
        assert comp.embedding_server is not None
        assert comp.embedding_server._local_fingerprint_index is None


# ──────────────────────────────────────────────────────────────────────
# Layer 3 — Two-node E2E
# ──────────────────────────────────────────────────────────────────────


class TestTwoNodeFingerprintE2E:
    def test_b_fetches_fingerprint_from_a_with_real_signatures(
        self, tmp_path,
    ):
        """Real Ed25519-signed fingerprint flows over the full DHT
        stack: SyncDHTTransport → DHTListener → DHTRequestRouter →
        EmbeddingDHTServer → wire → B parses + verifies signature."""
        # Sign a fingerprint with an Ed25519 keypair we capture so the
        # creator_pubkey_for stub can hand the matching pubkey back.
        # Note: KademliaDHT.find_closest_peers does bytes.fromhex on the
        # target_id, so content_hash must be plain hex (no "0x" prefix).
        private_key = Ed25519PrivateKey.generate()
        pubkey_bytes = private_key.public_key().public_bytes(
            encoding=Encoding.Raw, format=PublicFormat.Raw,
        )
        content_hash = hashlib.sha256(b"binary-content").hexdigest()
        kind_value = "image-phash"
        payload = b"\xde\xad\xbe\xef" * 2
        created_at = 1714000000.0

        sig_message = fingerprint_signing_payload(
            content_hash=content_hash,
            fingerprint_kind=kind_value,
            payload_bytes=payload,
            created_at=created_at,
        )
        signature = private_key.sign(sig_message)

        from prsm.network.embedding_dht.local_fingerprint_index import (
            LocalFingerprintRecord,
        )
        signed_record = LocalFingerprintRecord(
            content_hash=content_hash,
            fingerprint_kind=kind_value,
            payload_b64=base64.b64encode(payload).decode("ascii"),
            creator_id="creator-node",
            created_at=created_at,
            signature_b64=base64.b64encode(signature).decode("ascii"),
        )

        # Node A: has the fingerprint registered locally.
        a_identity = generate_node_identity("fp-node-a")
        a_emb_index = _make_emb_index(tmp_path, "a_emb_idx")
        a_fp_index = _make_fp_index(tmp_path, "a_fp_idx")
        a_fp_index.register(signed_record)

        # Node B: empty fingerprint storage; will fetch from A.
        b_identity = generate_node_identity("fp-node-b")
        b_emb_index = _make_emb_index(tmp_path, "b_emb_idx")
        b_fp_index = _make_fp_index(tmp_path, "b_fp_idx")

        def _creator_pubkey_for(ch: str) -> Optional[bytes]:
            if ch == content_hash:
                return pubkey_bytes
            return None

        node_a = DHTNodeComponents.build(
            my_node_id=a_identity.node_id,
            my_host="127.0.0.1",
            dht_listen_port=0,
            transport_adapter=DirectAdapter(),
            listen_host="127.0.0.1",
            local_embedding_index=a_emb_index,
            local_fingerprint_index=a_fp_index,
            creator_pubkey_for=_creator_pubkey_for,
            verify_signature=_real_verify_signature,
        )
        node_b = DHTNodeComponents.build(
            my_node_id=b_identity.node_id,
            my_host="127.0.0.1",
            dht_listen_port=0,
            transport_adapter=DirectAdapter(),
            listen_host="127.0.0.1",
            local_embedding_index=b_emb_index,
            local_fingerprint_index=b_fp_index,
            creator_pubkey_for=_creator_pubkey_for,
            verify_signature=_real_verify_signature,
        )

        try:
            port_a = node_a.start(
                creator_pubkey_for=_creator_pubkey_for,
                verify_signature=_real_verify_signature,
            )
            node_b.start(
                creator_pubkey_for=_creator_pubkey_for,
                verify_signature=_real_verify_signature,
            )
            assert node_b.add_peer(
                a_identity.node_id, "127.0.0.1", port_a,
            )

            assert node_b.embedding_client is not None
            providers = node_b.embedding_client.find_fingerprint_providers(
                content_hash=content_hash,
                fingerprint_kind=kind_value,
            )
            assert len(providers) >= 1

            # Fetch from the first provider; signature verification
            # passes inside fetch_fingerprint via _verify_fingerprint_signature.
            resp = node_b.embedding_client.fetch_fingerprint(
                providers[0],
                content_hash=content_hash,
                fingerprint_kind=kind_value,
            )
            assert resp.content_hash == content_hash
            assert resp.fingerprint_kind == kind_value
            assert resp.payload_b64 == signed_record.payload_b64
            assert resp.signature_b64 == signed_record.signature_b64
        finally:
            node_b.stop()
            node_a.stop()
