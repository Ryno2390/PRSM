"""
Tests for prsm.node.identity — Ed25519 keypair, signing, and persistence.
"""

import json
import tempfile
from pathlib import Path

import pytest

from prsm.node.identity import (
    NodeIdentity,
    generate_node_identity,
    load_node_identity,
    save_node_identity,
    verify_signature,
)


class TestGenerateNodeIdentity:
    def test_generates_unique_ids(self):
        id1 = generate_node_identity()
        id2 = generate_node_identity()
        assert id1.node_id != id2.node_id

    def test_node_id_is_32_hex_chars(self):
        identity = generate_node_identity()
        assert len(identity.node_id) == 32
        int(identity.node_id, 16)  # should not raise

    def test_sets_display_name(self):
        identity = generate_node_identity("my-node")
        assert identity.display_name == "my-node"

    def test_default_display_name(self):
        identity = generate_node_identity()
        assert identity.display_name == "prsm-node"

    def test_has_key_bytes(self):
        identity = generate_node_identity()
        assert len(identity.public_key_bytes) == 32   # Ed25519 public key
        assert len(identity.private_key_bytes) == 32   # Ed25519 seed

    def test_public_key_b64(self):
        import base64
        identity = generate_node_identity()
        decoded = base64.b64decode(identity.public_key_b64)
        assert decoded == identity.public_key_bytes


class TestSignAndVerify:
    def test_sign_and_verify_roundtrip(self):
        identity = generate_node_identity()
        data = b"hello world"
        sig = identity.sign(data)
        assert identity.verify(data, sig)

    def test_verify_rejects_wrong_data(self):
        identity = generate_node_identity()
        sig = identity.sign(b"hello")
        assert not identity.verify(b"goodbye", sig)

    def test_verify_rejects_wrong_signature(self):
        identity = generate_node_identity()
        identity.sign(b"hello")
        assert not identity.verify(b"hello", "badsignature==")

    def test_cross_node_verification(self):
        """Verify that one node's signature can be verified using its public key."""
        signer = generate_node_identity()
        data = b"test message"
        sig = signer.sign(data)

        # Use the standalone verify_signature function
        assert verify_signature(signer.public_key_b64, data, sig)

    def test_cross_node_reject_wrong_key(self):
        signer = generate_node_identity()
        other = generate_node_identity()
        data = b"test message"
        sig = signer.sign(data)

        # Should fail with wrong public key
        assert not verify_signature(other.public_key_b64, data, sig)


class TestPersistence:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "identity.json"
            original = generate_node_identity("test-node")
            save_node_identity(original, path)

            loaded = load_node_identity(path)
            assert loaded is not None
            assert loaded.node_id == original.node_id
            assert loaded.display_name == "test-node"
            assert loaded.public_key_bytes == original.public_key_bytes
            assert loaded.private_key_bytes == original.private_key_bytes

    def test_load_nonexistent_returns_none(self):
        result = load_node_identity(Path("/tmp/nonexistent_identity.json"))
        assert result is None

    def test_saved_identity_can_sign(self):
        """Verify that a loaded identity's keys work for signing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "identity.json"
            original = generate_node_identity()
            save_node_identity(original, path)

            loaded = load_node_identity(path)
            data = b"persistence test"
            sig = loaded.sign(data)
            assert loaded.verify(data, sig)
            # Original can also verify
            assert original.verify(data, sig)

    def test_to_dict_from_dict_roundtrip(self):
        original = generate_node_identity("roundtrip")
        d = original.to_dict()
        restored = NodeIdentity.from_dict(d)
        assert restored.node_id == original.node_id
        assert restored.display_name == "roundtrip"
        assert restored.public_key_bytes == original.public_key_bytes

    def test_save_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "identity.json"
            identity = generate_node_identity()
            save_node_identity(identity, path)
            assert path.exists()
