"""
Unit tests — Phase 3.x.4 Task 1 — PrivacyBudgetEntry dataclass + signing payload.

Acceptance per design plan §4 Task 1: entry dataclass + signing payload
pinned; tests cover all field combinations including prev_entry_hash
chain-byte sensitivity.

The signing payload is a wire format. These tests assert exact bytes
where the wire-format contract demands it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from prsm.security.privacy_budget_persistence.models import (
    ENTRY_SCHEMA_VERSION,
    ENTRY_SIGNING_DOMAIN,
    GENESIS_PREV_HASH,
    PrivacyBudgetEntry,
    PrivacyBudgetEntryType,
)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _spend_entry(
    *,
    sequence_number: int = 0,
    epsilon: float = 8.0,
    operation: str = "inference",
    model_id: str = "llama-3-8b",
    timestamp: float = 1714000000.0,
    node_id: str = "node-AAA",
    prev_entry_hash: bytes = GENESIS_PREV_HASH,
) -> PrivacyBudgetEntry:
    return PrivacyBudgetEntry(
        sequence_number=sequence_number,
        entry_type=PrivacyBudgetEntryType.SPEND,
        node_id=node_id,
        epsilon=epsilon,
        operation=operation,
        model_id=model_id,
        timestamp=timestamp,
        prev_entry_hash=prev_entry_hash,
    )


def _reset_entry(
    *,
    sequence_number: int = 1,
    timestamp: float = 1714000001.0,
    node_id: str = "node-AAA",
    prev_entry_hash: bytes = GENESIS_PREV_HASH,
) -> PrivacyBudgetEntry:
    return PrivacyBudgetEntry(
        sequence_number=sequence_number,
        entry_type=PrivacyBudgetEntryType.RESET,
        node_id=node_id,
        epsilon=0.0,
        operation="",
        model_id="",
        timestamp=timestamp,
        prev_entry_hash=prev_entry_hash,
    )


# ──────────────────────────────────────────────────────────────────────────
# Module-level constants — pinned wire-format identifiers
# ──────────────────────────────────────────────────────────────────────────


class TestModuleConstants:
    def test_schema_version_is_one(self):
        # Bumping requires a new domain string and a migration. Lock
        # the value so anyone touching it has to update this test.
        assert ENTRY_SCHEMA_VERSION == 1

    def test_signing_domain_is_pinned(self):
        # Distinct from MANIFEST_SIGNING_DOMAIN (Phase 3.x.2) and from
        # InferenceReceipt's domain (Phase 3.x.1) — cross-artifact replay
        # protection lives here.
        assert ENTRY_SIGNING_DOMAIN == b"prsm-privacy-budget-entry:v1"

    def test_genesis_prev_hash_is_32_zero_bytes(self):
        # Same width as a sha256 digest so chain-verification logic
        # has no special case for the first entry.
        assert GENESIS_PREV_HASH == b"\x00" * 32
        assert len(GENESIS_PREV_HASH) == 32


# ──────────────────────────────────────────────────────────────────────────
# Construction + invariants
# ──────────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_basic_spend_entry(self):
        e = _spend_entry()
        assert e.entry_type == PrivacyBudgetEntryType.SPEND
        assert e.epsilon == 8.0
        assert e.operation == "inference"
        assert e.model_id == "llama-3-8b"
        assert e.schema_version == 1
        assert e.signature == b""

    def test_basic_reset_entry(self):
        e = _reset_entry()
        assert e.entry_type == PrivacyBudgetEntryType.RESET
        assert e.epsilon == 0.0
        assert e.operation == ""
        assert e.model_id == ""

    def test_is_frozen(self):
        e = _spend_entry()
        with pytest.raises((AttributeError, Exception)):
            e.epsilon = 99.0  # type: ignore[misc]

    def test_entry_type_coerced_from_str(self):
        # JSON load path passes strings; __post_init__ must coerce.
        e = PrivacyBudgetEntry(
            sequence_number=0, entry_type="spend",  # type: ignore[arg-type]
            node_id="n", epsilon=1.0, operation="op",
            model_id="m", timestamp=0.0, prev_entry_hash=GENESIS_PREV_HASH,
        )
        assert e.entry_type == PrivacyBudgetEntryType.SPEND

    def test_int_float_coercion(self):
        # JSON often round-trips ints/floats imprecisely.
        e = PrivacyBudgetEntry(
            sequence_number=2.0,  # type: ignore[arg-type]
            entry_type=PrivacyBudgetEntryType.SPEND,
            node_id="n",
            epsilon=1,  # type: ignore[arg-type]
            operation="op", model_id="m",
            timestamp=42,  # type: ignore[arg-type]
            prev_entry_hash=GENESIS_PREV_HASH,
            schema_version=1.0,  # type: ignore[arg-type]
        )
        assert isinstance(e.sequence_number, int)
        assert isinstance(e.epsilon, float)
        assert isinstance(e.timestamp, float)
        assert isinstance(e.schema_version, int)

    def test_negative_sequence_number_rejected(self):
        # Sequence numbers are 0-indexed monotonic gap-free; -1 makes
        # no sense.
        with pytest.raises(ValueError, match="sequence_number"):
            _spend_entry(sequence_number=-1)

    def test_prev_entry_hash_must_be_bytes(self):
        with pytest.raises(TypeError, match="prev_entry_hash"):
            PrivacyBudgetEntry(
                sequence_number=0, entry_type=PrivacyBudgetEntryType.SPEND,
                node_id="n", epsilon=1.0, operation="op", model_id="m",
                timestamp=0.0,
                prev_entry_hash="0" * 64,  # type: ignore[arg-type] — hex string, not bytes
            )

    def test_prev_entry_hash_must_be_32_bytes(self):
        # Wider or narrower hashes break the chain-verification uniformity.
        for bad in [b"", b"\x00" * 16, b"\x00" * 31, b"\x00" * 33, b"\x00" * 64]:
            with pytest.raises(ValueError, match="32 bytes"):
                _spend_entry(prev_entry_hash=bad)

    def test_signature_must_be_bytes(self):
        with pytest.raises(TypeError, match="signature"):
            PrivacyBudgetEntry(
                sequence_number=0, entry_type=PrivacyBudgetEntryType.SPEND,
                node_id="n", epsilon=1.0, operation="op", model_id="m",
                timestamp=0.0, prev_entry_hash=GENESIS_PREV_HASH,
                signature="not bytes",  # type: ignore[arg-type]
            )


# ──────────────────────────────────────────────────────────────────────────
# Signing payload — canonical, deterministic, schema-bound
# ──────────────────────────────────────────────────────────────────────────


class TestSigningPayload:
    def test_payload_starts_with_signing_domain(self):
        # Domain separation: prevents cross-artifact replay of a manifest
        # signature against a budget-entry payload.
        e = _spend_entry()
        payload = e.signing_payload()
        assert payload.startswith(ENTRY_SIGNING_DOMAIN)

    def test_payload_includes_schema_version(self):
        # Downgrade attack: re-stamping a v2 entry as v1 must invalidate
        # the signature.
        e = _spend_entry()
        payload = e.signing_payload()
        assert f"|{ENTRY_SCHEMA_VERSION}|".encode() in payload

    def test_payload_excludes_signature(self):
        # If signing payload included the signature, signing would be
        # circular: signing produces a sig; including it changes the
        # payload; the new payload needs a new sig; ad infinitum.
        e1 = _spend_entry()
        e2 = dataclasses.replace(e1, signature=b"\xff" * 64)
        assert e1.signing_payload() == e2.signing_payload()

    def test_payload_deterministic_across_runs(self):
        # Same entry, two constructions → same bytes. No timestamp drift,
        # no dict-key ordering surprises.
        e1 = _spend_entry()
        e2 = _spend_entry()
        assert e1.signing_payload() == e2.signing_payload()

    # Per-field tamper sensitivity — every field in signing_payload
    # MUST influence the bytes. If any of these tests fail, that field
    # is missing from the payload and could be tampered without invalidating
    # the signature.

    def test_payload_sensitive_to_sequence_number(self):
        e1 = _spend_entry(sequence_number=0)
        e2 = _spend_entry(sequence_number=1)
        assert e1.signing_payload() != e2.signing_payload()

    def test_payload_sensitive_to_entry_type(self):
        spend = _spend_entry()
        reset = _reset_entry(
            sequence_number=spend.sequence_number,
            timestamp=spend.timestamp,
            node_id=spend.node_id,
            prev_entry_hash=spend.prev_entry_hash,
        )
        # Even with same sequence/timestamp/node_id, RESET vs SPEND
        # must produce different payloads.
        assert spend.signing_payload() != reset.signing_payload()

    def test_payload_sensitive_to_node_id(self):
        e1 = _spend_entry(node_id="alice")
        e2 = _spend_entry(node_id="bob")
        assert e1.signing_payload() != e2.signing_payload()

    def test_payload_sensitive_to_epsilon(self):
        e1 = _spend_entry(epsilon=8.0)
        e2 = _spend_entry(epsilon=4.0)
        assert e1.signing_payload() != e2.signing_payload()

    def test_payload_sensitive_to_epsilon_subdigit(self):
        # :.10f format means 10-decimal-place precision. A change at
        # position 10 must influence the bytes.
        e1 = _spend_entry(epsilon=1.0000000001)
        e2 = _spend_entry(epsilon=1.0000000002)
        assert e1.signing_payload() != e2.signing_payload()

    def test_payload_sensitive_to_operation(self):
        e1 = _spend_entry(operation="inference")
        e2 = _spend_entry(operation="forge_query")
        assert e1.signing_payload() != e2.signing_payload()

    def test_payload_sensitive_to_model_id(self):
        e1 = _spend_entry(model_id="llama-3-8b")
        e2 = _spend_entry(model_id="mistral-7b")
        assert e1.signing_payload() != e2.signing_payload()

    def test_payload_sensitive_to_timestamp(self):
        e1 = _spend_entry(timestamp=1.0)
        e2 = _spend_entry(timestamp=2.0)
        assert e1.signing_payload() != e2.signing_payload()

    def test_payload_sensitive_to_prev_entry_hash(self):
        # The chain link: any change in predecessor's hash must change
        # this entry's payload, breaking its signature. This is what
        # makes historical-tamper-detection work.
        e1 = _spend_entry(prev_entry_hash=GENESIS_PREV_HASH)
        e2 = _spend_entry(prev_entry_hash=hashlib.sha256(b"different").digest())
        assert e1.signing_payload() != e2.signing_payload()


# ──────────────────────────────────────────────────────────────────────────
# JSON roundtrip — used by FilesystemPrivacyBudgetStore (Task 4)
# ──────────────────────────────────────────────────────────────────────────


class TestJSONRoundtrip:
    def test_to_from_dict_preserves_spend(self):
        e1 = _spend_entry()
        e2 = PrivacyBudgetEntry.from_dict(e1.to_dict())
        assert e1 == e2

    def test_to_from_dict_preserves_reset(self):
        e1 = _reset_entry()
        e2 = PrivacyBudgetEntry.from_dict(e1.to_dict())
        assert e1 == e2

    def test_signature_hex_roundtrip(self):
        sig = bytes(range(64))
        e1 = dataclasses.replace(_spend_entry(), signature=sig)
        d = e1.to_dict()
        # Hex-encoded for JSON safety
        assert isinstance(d["signature"], str)
        assert d["signature"] == sig.hex()
        e2 = PrivacyBudgetEntry.from_dict(d)
        assert e2.signature == sig

    def test_prev_entry_hash_hex_roundtrip(self):
        h = hashlib.sha256(b"test").digest()
        e1 = _spend_entry(prev_entry_hash=h)
        d = e1.to_dict()
        assert d["prev_entry_hash"] == h.hex()
        e2 = PrivacyBudgetEntry.from_dict(d)
        assert e2.prev_entry_hash == h

    def test_entry_type_serializes_as_string(self):
        # Enums-as-strings is the JSON-friendly form. Reading back must
        # still produce a real PrivacyBudgetEntryType, not a bare str.
        e = _spend_entry()
        d = e.to_dict()
        assert d["entry_type"] == "spend"
        e2 = PrivacyBudgetEntry.from_dict(d)
        assert e2.entry_type == PrivacyBudgetEntryType.SPEND

    def test_canonical_json_serialization(self):
        # to_dict() must produce JSON-encodable values; sort_keys gives
        # a canonical byte form so two stores can compare entries by
        # bytes alone.
        e = _spend_entry()
        d = e.to_dict()
        canonical = json.dumps(d, sort_keys=True).encode("utf-8")
        # Round-trip through JSON without loss
        e_back = PrivacyBudgetEntry.from_dict(json.loads(canonical))
        assert e == e_back

    def test_from_dict_drops_unknown_keys(self):
        # Forward-compat: a v2 entry adding a field shouldn't break
        # v1 readers.
        d = {**_spend_entry().to_dict(), "future_field": "ignored"}
        e = PrivacyBudgetEntry.from_dict(d)
        assert e.sequence_number == 0


# ──────────────────────────────────────────────────────────────────────────
# Genesis vs non-genesis chain semantics
# ──────────────────────────────────────────────────────────────────────────


class TestChainSemantics:
    def test_genesis_uses_zero_prev_hash(self):
        # The first entry in any journal carries GENESIS_PREV_HASH.
        # Documented contract; test pins it so no one "optimizes" the
        # genesis entry into a special case.
        e = _spend_entry(sequence_number=0, prev_entry_hash=GENESIS_PREV_HASH)
        assert e.prev_entry_hash == b"\x00" * 32

    def test_non_genesis_uses_predecessor_payload_hash(self):
        # The chain-construction contract: prev_entry_hash equals
        # sha256(predecessor.signing_payload()). The store's append()
        # method enforces this; here we just verify the dataclass
        # doesn't reject a real predecessor hash.
        e1 = _spend_entry(sequence_number=0)
        prev_hash = hashlib.sha256(e1.signing_payload()).digest()
        e2 = _spend_entry(sequence_number=1, prev_entry_hash=prev_hash)
        assert e2.prev_entry_hash == prev_hash
        # And the resulting payload is genuinely different from a
        # genesis-style one.
        e2_genesis = _spend_entry(sequence_number=1)
        assert e2.signing_payload() != e2_genesis.signing_payload()
