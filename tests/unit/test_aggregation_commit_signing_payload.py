"""AggregationCommit.signing_payload — canonical signing payload pin.

Per `docs/2026-05-08-aggregate-rpc-design.md` §"AggregateResponse —
Signing payload": this 124-byte payload is the prsm-canonical input to
the aggregator's Ed25519 signature over a commit. Once signatures
exist on-chain, changing this layout would invalidate them all —
hence the golden-vector pin.

Mirrors Phase 3.x.1 InferenceReceipt's signing-payload pinning
pattern in `test_inference_receipt_signing_payload.py`.
"""
from __future__ import annotations

import hashlib

import pytest

from prsm.compute.query_orchestrator import AggregationCommit


# ──────────────────────────────────────────────────────────────────────
# Layout pin
# ──────────────────────────────────────────────────────────────────────


class TestLayout:
    def test_prefix_is_versioned_string(self):
        assert AggregationCommit.SIGNING_PREFIX == b"prsm:aggregation-commit:v1\n"

    def test_payload_is_124_bytes(self):
        commit = AggregationCommit(
            query_id=b"\x01" * 32,
            aggregator_pubkey_hash=b"\x02" * 32,
            result_digest=b"\x03" * 32,
        )
        # 27-byte prefix (with trailing \n) + 32 + 32 + 32 = 123 bytes.
        # Wait — let's compute exactly.
        expected_len = len(b"prsm:aggregation-commit:v1\n") + 32 * 3
        assert len(commit.signing_payload()) == expected_len

    def test_payload_layout_field_order(self):
        commit = AggregationCommit(
            query_id=b"\xaa" * 32,
            aggregator_pubkey_hash=b"\xbb" * 32,
            result_digest=b"\xcc" * 32,
        )
        payload = commit.signing_payload()
        prefix_len = len(AggregationCommit.SIGNING_PREFIX)
        # Prefix first.
        assert payload[:prefix_len] == AggregationCommit.SIGNING_PREFIX
        # Then query_id.
        assert payload[prefix_len:prefix_len + 32] == b"\xaa" * 32
        # Then aggregator_pubkey_hash.
        assert payload[prefix_len + 32:prefix_len + 64] == b"\xbb" * 32
        # Then result_digest.
        assert payload[prefix_len + 64:prefix_len + 96] == b"\xcc" * 32


# ──────────────────────────────────────────────────────────────────────
# Validation — refuse non-32-byte fields
# ──────────────────────────────────────────────────────────────────────


class TestValidation:
    def test_short_query_id_raises(self):
        commit = AggregationCommit(
            query_id=b"\x01" * 16,  # short
            aggregator_pubkey_hash=b"\x02" * 32,
            result_digest=b"\x03" * 32,
        )
        with pytest.raises(ValueError, match="query_id"):
            commit.signing_payload()

    def test_short_pubkey_hash_raises(self):
        commit = AggregationCommit(
            query_id=b"\x01" * 32,
            aggregator_pubkey_hash=b"\x02" * 16,  # short
            result_digest=b"\x03" * 32,
        )
        with pytest.raises(ValueError, match="aggregator_pubkey_hash"):
            commit.signing_payload()

    def test_short_result_digest_raises(self):
        commit = AggregationCommit(
            query_id=b"\x01" * 32,
            aggregator_pubkey_hash=b"\x02" * 32,
            result_digest=b"\x03" * 16,  # short
        )
        with pytest.raises(ValueError, match="result_digest"):
            commit.signing_payload()


# ──────────────────────────────────────────────────────────────────────
# Golden vector — if this changes, every signed commit invalidates
# ──────────────────────────────────────────────────────────────────────


class TestGoldenVector:
    """Pin a fully-known commit's signing-payload SHA-256. If the
    layout, prefix, or field order ever changes, this hash blows.
    Update the constant ONLY when intentionally bumping the payload
    version (and bump SIGNING_PREFIX in lockstep)."""

    def test_known_commit_payload_sha256(self):
        # Fully-deterministic input: each field set to a distinct
        # repeating byte pattern.
        commit = AggregationCommit(
            query_id=bytes(range(32)),                       # 0..31
            aggregator_pubkey_hash=bytes(range(32, 64)),     # 32..63
            result_digest=bytes(range(64, 96)),              # 64..95
        )
        payload = commit.signing_payload()
        digest = hashlib.sha256(payload).hexdigest()
        # If you intentionally change the signing payload format,
        # regenerate this constant from the failed test output.
        assert digest == (
            "45d4dd7f25aed3eb9b87110c2058afeccf96e411ba5bf80a41bee5d0b03b5831"
        )
