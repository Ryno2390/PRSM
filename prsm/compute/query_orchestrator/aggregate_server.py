"""B3.2 — server-side aggregate combination handler.

Pure-functional core of the aggregator's response path. An
`AggregateServer` instance composes four primitives:

  - `verify_partial_signature(partial)`   — Ed25519 over each SignedPartial
  - `enforce_a5_marker(partial)`          — refuse-and-raise on
                                            dp_noise_applied=False
  - `sum_privacy_budgets(partials, ceiling)` — epsilon composition + ceiling
  - `combine_partials(manifest, partials)` — per AgentOp combination

All four raise typed exceptions that carry `StageErrorCode` values
straight to the wire — the transport layer just JSON-encodes the
StageError and sends.

v1 scope is intentionally narrow:
  - `AgentOp.COUNT` correctly combines (sum of per-partial integer
    payloads) and emits canonical `{"count": N}` JSON
  - All other AgentOps raise `UnsupportedAgentOpError` —
    canonical per-op encoding requires a separate design pass; better
    to fail loudly than guess at the wire format

Per design doc `docs/2026-05-08-aggregate-rpc-design.md` §"Server-side flow".
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519

from prsm.compute.agents.instruction_set import (
    AgentOp,
    InstructionManifest,
)
from prsm.compute.chain_rpc.protocol import StageErrorCode
from prsm.compute.query_orchestrator.aggregate_protocol import (
    AggregateRequest,
    AggregateResponse,
    SignedPartial,
)
from prsm.compute.query_orchestrator.aggregator_selector import (
    AggregationCommit,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Typed exceptions
# ──────────────────────────────────────────────────────────────────────


class AggregateServerError(RuntimeError):
    """Server-side failure carrying a StageErrorCode for wire-level
    error reporting. The transport layer maps `code` to a
    `StageError` message and ships."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class PrivacyBudgetExhaustedError(AggregateServerError):
    """Combination would exceed the configured epsilon ceiling."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message, code=StageErrorCode.PRIVACY_BUDGET_EXHAUSTED.value,
        )


class UnsupportedAgentOpError(AggregateServerError):
    """v1 only supports AgentOp.COUNT. Other ops require a canonical
    per-op encoding decision that hasn't landed yet — fail loudly so
    the orchestrator's caller doesn't silently get wrong answers."""

    def __init__(self, op: AgentOp) -> None:
        super().__init__(
            f"AgentOp.{op.name} not yet supported by aggregate-server v1; "
            f"canonical per-op encoding is a follow-on (see "
            f"docs/2026-05-08-aggregate-rpc-design.md §B3.2)",
            code=StageErrorCode.MALFORMED_REQUEST.value,
        )
        self.op = op


# ──────────────────────────────────────────────────────────────────────
# Per-partial primitives
# ──────────────────────────────────────────────────────────────────────


def _partial_canonical_signing_bytes(partial: SignedPartial) -> bytes:
    """The bytes the source agent signed when emitting this partial.

    Mirrors the test fixture's canonical encoding so server-side
    verification matches what an honest source agent would have
    produced. If a future change to the source-agent signing path
    extends this set of fields, both sides must update together.
    """
    return json.dumps({
        "shard_cid": partial.shard_cid,
        "payload_sha256": hashlib.sha256(partial.payload).hexdigest(),
        "dp_noise_applied": partial.dp_noise_applied,
        "privacy_budget_consumed": partial.privacy_budget_consumed,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")


def verify_partial_signature(partial: SignedPartial) -> None:
    """Ed25519-verify the source agent's signature over the canonical
    bundle. Raises `AggregateServerError` with code
    `INVALID_PARTIAL_SIGNATURE` on failure."""
    pubkey = ed25519.Ed25519PublicKey.from_public_bytes(
        bytes(partial.source_agent_pubkey),
    )
    try:
        pubkey.verify(
            bytes(partial.source_agent_signature),
            _partial_canonical_signing_bytes(partial),
        )
    except InvalidSignature as exc:
        raise AggregateServerError(
            f"signature verify failed for shard_cid={partial.shard_cid!r}",
            code=StageErrorCode.INVALID_PARTIAL_SIGNATURE.value,
        ) from exc


def enforce_a5_marker(partial: SignedPartial) -> None:
    """Threat-model A5 mitigation 2 enforcement at the server. The
    aggregator MUST refuse to combine un-noised partials — if the
    source agent didn't apply DP noise, the aggregation would leak
    raw shard data into the combined output."""
    if not partial.dp_noise_applied:
        raise AggregateServerError(
            f"partial for shard_cid={partial.shard_cid!r} arrived without "
            f"dp_noise_applied=True — A5 violation",
            code=StageErrorCode.DP_NOISE_MARKER_MISSING.value,
        )


def sum_privacy_budgets(
    partials: Iterable[SignedPartial],
    *,
    ceiling: float,
) -> float:
    """Sum per-partial epsilon (post-Laplace composition is additive)
    and refuse-and-raise if the total would exceed `ceiling`.

    The orchestrator threads `ceiling` from the manifest's per-query
    privacy budget. v1 uses a flat ceiling — finer-grained per-AgentOp
    or per-prompter ledgers ship as follow-on against
    `prsm/security/privacy_budget.py`."""
    total = 0.0
    for p in partials:
        total += float(p.privacy_budget_consumed)
        if total > ceiling:
            raise PrivacyBudgetExhaustedError(
                f"summed epsilon {total} > ceiling {ceiling}"
            )
    return total


# ──────────────────────────────────────────────────────────────────────
# Combination
# ──────────────────────────────────────────────────────────────────────


def _combine_count(partials: Sequence[SignedPartial]) -> bytes:
    """COUNT combination: each partial.payload is a UTF-8 integer
    string (e.g. b"42"). Sum them, emit canonical
    `{"count": N}` JSON.

    The integer-string convention is the same as the source agent's
    WASM output for a COUNT shard. If that convention ever changes,
    both sides must update together (test_combine_partials_count
    pins the contract)."""
    total = 0
    for p in partials:
        try:
            total += int(p.payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise AggregateServerError(
                f"COUNT payload for shard_cid={p.shard_cid!r} is not a "
                f"UTF-8 integer string: {p.payload!r}",
                code=StageErrorCode.MALFORMED_REQUEST.value,
            ) from exc
    return json.dumps({"count": total}, separators=(",", ":")).encode("utf-8")


def combine_partials(
    manifest: InstructionManifest,
    partials: Sequence[SignedPartial],
) -> bytes:
    """Apply the manifest's first instruction to combine per-shard
    partials into a single plaintext result.

    v1 limitation: only the FIRST instruction is honored, and only
    `AgentOp.COUNT` is supported. Other ops raise
    `UnsupportedAgentOpError`. Multi-instruction pipelines (e.g.
    FILTER → COUNT) are a follow-on once the per-op canonical
    encoding lands.
    """
    if not manifest.instructions:
        raise AggregateServerError(
            "manifest has no instructions to apply",
            code=StageErrorCode.MALFORMED_REQUEST.value,
        )
    first = manifest.instructions[0]
    if first.op == AgentOp.COUNT:
        return _combine_count(partials)
    raise UnsupportedAgentOpError(first.op)


# ──────────────────────────────────────────────────────────────────────
# AggregateServer — full request → response
# ──────────────────────────────────────────────────────────────────────


@dataclass
class AggregateServer:
    """Composes the four primitives + the response-signing step.

    A single instance is bound to one aggregator identity (one Ed25519
    keypair). Production wiring instantiates this once at node startup
    with the node's stable identity key + the privacy-budget ceiling
    from policy.

    NOT thread-safe — instantiate one per worker, or wrap externally.
    The handler is async-friendly via the absence of internal state
    mutation (each `handle` call is independent given the inputs).
    """

    aggregator_privkey: ed25519.Ed25519PrivateKey
    privacy_budget_ceiling: float

    def handle(self, request: AggregateRequest) -> AggregateResponse:
        """Run the full server-side flow per design doc §"Server-side flow".

        Steps (paraphrased from the doc; rejected steps deferred to
        the wiring layer):
          5. Verify each SignedPartial's source_agent_signature
          6. Assert dp_noise_applied=True on each partial (A5)
          7. Parse manifest_json → InstructionManifest
          8. Combine per AgentOp
          9. Compute result_digest
          10-14. Sign commit + build response

        Skipped here (lives in transport-layer wrapper):
          1-4 — protocol/version/deadline/identity checks (transport)
        """
        # Step 5 + 6: per-partial sig verify + A5 enforcement.
        for p in request.partials:
            verify_partial_signature(p)
            enforce_a5_marker(p)

        # Privacy-budget ceiling — composition + refuse over budget.
        budget_total = sum_privacy_budgets(
            request.partials, ceiling=self.privacy_budget_ceiling,
        )

        # Step 7: parse the manifest. InstructionManifest.from_json
        # is the canonical reader.
        manifest = InstructionManifest.from_json(request.manifest_json)

        # Step 8: combine. Raises UnsupportedAgentOpError for
        # non-COUNT v1 ops — caller surfaces to prompter.
        plaintext = combine_partials(manifest, request.partials)

        # Step 9: commit binds query_id + our identity + result digest.
        result_digest = hashlib.sha256(plaintext).digest()
        our_pubkey = self.aggregator_privkey.public_key().public_bytes_raw()
        commit = AggregationCommit(
            query_id=request.query_id,
            aggregator_pubkey_hash=hashlib.sha256(our_pubkey).digest(),
            result_digest=result_digest,
        )

        # Step 10-12: sign the commit's canonical signing payload
        # (B3.1a established this as the binding receipt input).
        commit_signature = self.aggregator_privkey.sign(commit.signing_payload())

        # Step 13: encrypt the combined plaintext via X25519 ECDH +
        # XChaCha20-Poly1305 (see partial_result_cipher.py). The
        # aggregator's Ed25519 signing key + the prompter's pubkey
        # (already on the wire) derive a per-request AEAD key.
        # AAD binds the AggregationCommit's signing payload so any
        # commit-tampering invalidates the ciphertext.
        from prsm.compute.query_orchestrator.partial_result_cipher import (
            PartialResultCipherError,
            encrypt_aggregate_response,
        )
        try:
            encrypted_plaintext, nonce = encrypt_aggregate_response(
                aggregator_ed25519_privkey=self.aggregator_privkey.private_bytes_raw(),
                prompter_ed25519_pubkey=request.prompter_pubkey,
                plaintext=plaintext,
                request_id=request.request_id,
                commit_aad=commit.signing_payload(),
            )
        except PartialResultCipherError as exc:
            raise AggregateServerError(
                f"partial-result encryption failed: {exc}",
                code=StageErrorCode.MALFORMED_REQUEST.value,
            ) from exc

        # Step 14: assemble the response.
        return AggregateResponse(
            request_id=request.request_id,
            query_id=request.query_id,
            commit=commit,
            commit_signature=commit_signature,
            encrypted_plaintext=encrypted_plaintext,
            nonce=nonce,
            aggregator_pubkey=our_pubkey,
            privacy_budget_consumed=budget_total,
            contributing_creators=tuple(
                p.creator_id for p in request.partials
            ),
            completed_unix=int(time.time()),
        )
