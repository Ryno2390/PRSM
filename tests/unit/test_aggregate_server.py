"""B3.2 — server-side aggregate combination handler.

Pure-functional core of the aggregator's response path:

  - verify_partial_signature   — Ed25519 over each SignedPartial
  - enforce_a5_marker          — refuse-and-raise on dp_noise_applied=False
  - sum_privacy_budgets        — epsilon composition + ceiling
  - combine_partials           — per AgentOp (v1: COUNT only)
  - AggregateServer.handle     — composes the four into a full response

Per design doc `docs/2026-05-08-aggregate-rpc-design.md` §"Server-side flow".

v1 scope is intentionally narrow:
  - COUNT combination (sum of per-partial integer payloads)
  - Other AgentOps raise NotImplementedError with a clear follow-on
    pointer — canonical per-op encoding requires a separate design
    pass; better to fail loudly than guess
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Tuple

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from prsm.compute.agents.instruction_set import (
    AgentInstruction,
    AgentOp,
    InstructionManifest,
)
from prsm.compute.chain_rpc.protocol import StageErrorCode
from prsm.compute.query_orchestrator import AggregationCommit
from prsm.compute.query_orchestrator.aggregate_protocol import (
    AggregateRequest,
    AggregateResponse,
    SignedPartial,
)
from prsm.compute.query_orchestrator.aggregate_server import (
    AggregateServer,
    AggregateServerError,
    PrivacyBudgetExhaustedError,
    UnsupportedAgentOpError,
    combine_partials,
    enforce_a5_marker,
    sum_privacy_budgets,
    verify_partial_signature,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


def _make_signed_partial(
    *,
    payload: bytes,
    privacy_budget_consumed: float = 0.05,
    dp_noise_applied: bool = True,
    shard_cid: str = "prsm:shard-0",
) -> Tuple[SignedPartial, ed25519.Ed25519PublicKey]:
    """Build a signed partial with a real Ed25519 signature over its
    canonical bytes. Returns (partial, source_pubkey) so tests can
    verify against the real key."""
    privkey = ed25519.Ed25519PrivateKey.generate()
    pubkey = privkey.public_key()
    pubkey_bytes = pubkey.public_bytes_raw()

    # Build the unsigned bundle: payload + canonical metadata.
    canonical = json.dumps({
        "shard_cid": shard_cid,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "dp_noise_applied": dp_noise_applied,
        "privacy_budget_consumed": privacy_budget_consumed,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")

    sig = privkey.sign(canonical)

    partial = SignedPartial(
        shard_cid=shard_cid,
        payload=payload,
        creator_id="creator-test",
        dp_noise_applied=dp_noise_applied,
        source_agent_pubkey=pubkey_bytes,
        source_agent_signature=sig,
        privacy_budget_consumed=privacy_budget_consumed,
    )
    return partial, pubkey


# ──────────────────────────────────────────────────────────────────────
# verify_partial_signature
# ──────────────────────────────────────────────────────────────────────


class TestVerifyPartialSignature:
    def test_valid_signature_accepts(self):
        partial, _ = _make_signed_partial(payload=b"42")
        # Should not raise.
        verify_partial_signature(partial)

    def test_tampered_payload_rejects(self):
        partial, _ = _make_signed_partial(payload=b"42")
        # Substitute payload — sig was over the original.
        tampered = replace(partial, payload=b"99")
        with pytest.raises(AggregateServerError, match="INVALID_PARTIAL_SIGNATURE"):
            verify_partial_signature(tampered)

    def test_wrong_pubkey_rejects(self):
        partial, _ = _make_signed_partial(payload=b"42")
        # Substitute a different pubkey.
        wrong_pubkey = ed25519.Ed25519PrivateKey.generate().public_key().public_bytes_raw()
        tampered = replace(partial, source_agent_pubkey=wrong_pubkey)
        with pytest.raises(AggregateServerError, match="INVALID_PARTIAL_SIGNATURE"):
            verify_partial_signature(tampered)

    def test_short_pubkey_rejected_at_dataclass_layer(self):
        # Already covered by SignedPartial.__post_init__ but pin
        # the route — verify_partial_signature isn't where this
        # check should land (it's upstream).
        from prsm.compute.chain_rpc.protocol import ChainRpcMalformedError
        with pytest.raises(ChainRpcMalformedError):
            SignedPartial(
                shard_cid="x",
                payload=b"y",
                creator_id="c",
                dp_noise_applied=True,
                source_agent_pubkey=b"\xaa" * 16,
                source_agent_signature=b"\xbb" * 64,
                privacy_budget_consumed=0.0,
            )


# ──────────────────────────────────────────────────────────────────────
# enforce_a5_marker
# ──────────────────────────────────────────────────────────────────────


class TestEnforceA5Marker:
    def test_marker_set_passes(self):
        partial, _ = _make_signed_partial(payload=b"x", dp_noise_applied=True)
        enforce_a5_marker(partial)  # no raise

    def test_marker_unset_raises(self):
        partial, _ = _make_signed_partial(payload=b"x", dp_noise_applied=False)
        with pytest.raises(AggregateServerError, match="DP_NOISE_MARKER_MISSING"):
            enforce_a5_marker(partial)


# ──────────────────────────────────────────────────────────────────────
# sum_privacy_budgets
# ──────────────────────────────────────────────────────────────────────


class TestSumPrivacyBudgets:
    def test_sum_under_ceiling_returns_sum(self):
        a, _ = _make_signed_partial(payload=b"1", privacy_budget_consumed=0.1)
        b, _ = _make_signed_partial(payload=b"2", privacy_budget_consumed=0.2)
        total = sum_privacy_budgets([a, b], ceiling=1.0)
        assert total == pytest.approx(0.3)

    def test_sum_over_ceiling_raises(self):
        a, _ = _make_signed_partial(payload=b"1", privacy_budget_consumed=0.6)
        b, _ = _make_signed_partial(payload=b"2", privacy_budget_consumed=0.6)
        with pytest.raises(PrivacyBudgetExhaustedError):
            sum_privacy_budgets([a, b], ceiling=1.0)

    def test_empty_partials_returns_zero(self):
        assert sum_privacy_budgets([], ceiling=1.0) == 0.0


# ──────────────────────────────────────────────────────────────────────
# combine_partials — COUNT (v1 supported)
# ──────────────────────────────────────────────────────────────────────


class TestCombinePartialsCount:
    def test_count_sums_integer_payloads(self):
        manifest = InstructionManifest(
            query="count records",
            instructions=[AgentInstruction(op=AgentOp.COUNT)],
        )
        # Each per-shard agent emits the integer count for its shard.
        a, _ = _make_signed_partial(payload=b"3")
        b, _ = _make_signed_partial(payload=b"7")
        c, _ = _make_signed_partial(payload=b"10")
        plaintext = combine_partials(manifest, [a, b, c])
        # Canonical JSON output: {"count": 20}
        decoded = json.loads(plaintext.decode("utf-8"))
        assert decoded == {"count": 20}

    def test_count_empty_partials_returns_zero(self):
        manifest = InstructionManifest(
            query="count nothing",
            instructions=[AgentInstruction(op=AgentOp.COUNT)],
        )
        plaintext = combine_partials(manifest, [])
        assert json.loads(plaintext.decode("utf-8")) == {"count": 0}

    def test_count_rejects_non_integer_payload(self):
        manifest = InstructionManifest(
            query="count records",
            instructions=[AgentInstruction(op=AgentOp.COUNT)],
        )
        bad, _ = _make_signed_partial(payload=b"not-an-int")
        with pytest.raises(AggregateServerError, match="payload"):
            combine_partials(manifest, [bad])


# ──────────────────────────────────────────────────────────────────────
# combine_partials — unsupported AgentOps raise loudly
# ──────────────────────────────────────────────────────────────────────


class TestCombinePartialsUnsupported:
    """v1 ships COUNT only. Other AgentOps must raise loudly so the
    orchestrator's caller knows to use a different path until the
    canonical per-op encoding lands."""

    @pytest.mark.parametrize("op", [
        AgentOp.SUM,
        AgentOp.AVERAGE,
        AgentOp.GROUP_BY,
        AgentOp.SORT,
        AgentOp.FILTER,
        AgentOp.LIMIT,
        AgentOp.SELECT,
        AgentOp.COMPARE,
        AgentOp.TIME_SERIES,
        AgentOp.AGGREGATE,
    ])
    def test_unsupported_op_raises(self, op):
        manifest = InstructionManifest(
            query="anything",
            instructions=[AgentInstruction(op=op)],
        )
        a, _ = _make_signed_partial(payload=b"x")
        with pytest.raises(UnsupportedAgentOpError):
            combine_partials(manifest, [a])

    def test_no_instructions_raises(self):
        manifest = InstructionManifest(query="anything", instructions=[])
        a, _ = _make_signed_partial(payload=b"x")
        with pytest.raises(AggregateServerError, match="instruction"):
            combine_partials(manifest, [a])


# ──────────────────────────────────────────────────────────────────────
# AggregateServer.handle — full request → response
# ──────────────────────────────────────────────────────────────────────


def _make_request(
    partials: list[SignedPartial],
    prompter_pubkey: bytes | None = None,
) -> AggregateRequest:
    """Build a synthetic AggregateRequest for server tests.

    ``prompter_pubkey`` defaults to a fresh real Ed25519 pubkey so
    the server's X25519 derivation in ``encrypt_aggregate_response``
    succeeds. Tests that need a deterministic key pass it in.
    """
    if prompter_pubkey is None:
        prompter_pubkey = (
            ed25519.Ed25519PrivateKey.generate()
            .public_key()
            .public_bytes_raw()
        )
    manifest = InstructionManifest(
        query="count records",
        instructions=[AgentInstruction(op=AgentOp.COUNT)],
    )
    return AggregateRequest(
        request_id=b"\x10" * 32,
        query_id=b"\x20" * 32,
        manifest_json=manifest.to_json(),
        partials=tuple(partials),
        prompter_pubkey=prompter_pubkey,
        prompter_node_id="prompter-test",
        beacon_used=b"\xd0" * 32,
        aggregator_pubkey_hash=hashlib.sha256(b"aggregator").digest(),
        ftns_budget=1000,
        deadline_unix=2_000_000_000,
        prompter_signature=b"\xf0" * 64,
    )


class TestAggregateServerHandle:
    def test_full_round_trip(self):
        # Build the server with a deterministic Ed25519 keypair so
        # the response signature is reproducible.
        from prsm.compute.query_orchestrator.partial_result_cipher import (
            decrypt_aggregate_response,
        )

        privkey = ed25519.Ed25519PrivateKey.generate()
        server = AggregateServer(
            aggregator_privkey=privkey,
            privacy_budget_ceiling=1.0,
        )

        # Need a real prompter keypair so we can decrypt the response.
        prompter_priv = ed25519.Ed25519PrivateKey.generate()
        prompter_pub = prompter_priv.public_key().public_bytes_raw()

        a, _ = _make_signed_partial(payload=b"3", privacy_budget_consumed=0.1)
        b, _ = _make_signed_partial(payload=b"4", privacy_budget_consumed=0.1)
        request = _make_request([a, b], prompter_pubkey=prompter_pub)

        response = server.handle(request)

        # Response basics.
        assert isinstance(response, AggregateResponse)
        assert response.request_id == request.request_id
        assert response.query_id == request.query_id
        # Decrypt: ECDH against aggregator's pubkey + AAD = commit
        # signing payload (binds tampering).
        plaintext = decrypt_aggregate_response(
            prompter_ed25519_privkey=prompter_priv.private_bytes_raw(),
            aggregator_ed25519_pubkey=response.aggregator_pubkey,
            ciphertext=response.encrypted_plaintext,
            nonce=response.nonce,
            request_id=request.request_id,
            commit_aad=response.commit.signing_payload(),
        )
        assert json.loads(plaintext.decode("utf-8")) == {"count": 7}
        # Commit binds the digest.
        assert response.commit.result_digest == hashlib.sha256(plaintext).digest()
        # Privacy budget summed.
        assert response.privacy_budget_consumed == pytest.approx(0.2)
        # Contributing creators populated.
        assert "creator-test" in response.contributing_creators

    def test_a5_violation_propagates(self):
        privkey = ed25519.Ed25519PrivateKey.generate()
        server = AggregateServer(aggregator_privkey=privkey, privacy_budget_ceiling=1.0)
        bad, _ = _make_signed_partial(payload=b"3", dp_noise_applied=False)
        request = _make_request([bad])
        with pytest.raises(AggregateServerError, match="DP_NOISE_MARKER_MISSING"):
            server.handle(request)

    def test_invalid_signature_propagates(self):
        privkey = ed25519.Ed25519PrivateKey.generate()
        server = AggregateServer(aggregator_privkey=privkey, privacy_budget_ceiling=1.0)
        valid, _ = _make_signed_partial(payload=b"3")
        # Tamper.
        bad = replace(valid, payload=b"99")
        request = _make_request([bad])
        with pytest.raises(AggregateServerError, match="INVALID_PARTIAL_SIGNATURE"):
            server.handle(request)

    def test_budget_exhaustion_propagates(self):
        privkey = ed25519.Ed25519PrivateKey.generate()
        server = AggregateServer(aggregator_privkey=privkey, privacy_budget_ceiling=0.05)
        a, _ = _make_signed_partial(payload=b"3", privacy_budget_consumed=0.1)
        request = _make_request([a])
        with pytest.raises(PrivacyBudgetExhaustedError):
            server.handle(request)

    def test_commit_signature_verifies_against_aggregator_pubkey(self):
        privkey = ed25519.Ed25519PrivateKey.generate()
        pubkey = privkey.public_key()
        server = AggregateServer(aggregator_privkey=privkey, privacy_budget_ceiling=1.0)
        a, _ = _make_signed_partial(payload=b"3")
        request = _make_request([a])

        response = server.handle(request)

        # Aggregator pubkey field is the server's public key.
        assert response.aggregator_pubkey == pubkey.public_bytes_raw()
        # Commit signature verifies against it over the canonical
        # signing payload.
        pubkey.verify(response.commit_signature, response.commit.signing_payload())


# ──────────────────────────────────────────────────────────────────────
# Error code mapping pin
# ──────────────────────────────────────────────────────────────────────


class TestErrorCodeMapping:
    """The server's typed exceptions MUST map cleanly to the
    StageErrorCode values shipped in B3.1b. Pin the mapping."""

    def test_a5_error_carries_dp_noise_marker_missing_code(self):
        partial, _ = _make_signed_partial(payload=b"x", dp_noise_applied=False)
        try:
            enforce_a5_marker(partial)
        except AggregateServerError as exc:
            assert exc.code == StageErrorCode.DP_NOISE_MARKER_MISSING.value
        else:
            pytest.fail("did not raise")

    def test_signature_error_carries_invalid_partial_signature_code(self):
        partial, _ = _make_signed_partial(payload=b"x")
        tampered = replace(partial, payload=b"y")
        try:
            verify_partial_signature(tampered)
        except AggregateServerError as exc:
            assert exc.code == StageErrorCode.INVALID_PARTIAL_SIGNATURE.value
        else:
            pytest.fail("did not raise")

    def test_budget_error_carries_privacy_budget_exhausted_code(self):
        a, _ = _make_signed_partial(payload=b"x", privacy_budget_consumed=2.0)
        try:
            sum_privacy_budgets([a], ceiling=1.0)
        except PrivacyBudgetExhaustedError as exc:
            assert exc.code == StageErrorCode.PRIVACY_BUDGET_EXHAUSTED.value
        else:
            pytest.fail("did not raise")
