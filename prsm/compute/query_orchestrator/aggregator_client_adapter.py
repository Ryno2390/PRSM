"""B5 — AggregatorClientAdapter.

Wraps the AggregateRequest / AggregateResponse wire format from
``aggregate_protocol.py`` + a Phase-6 transport handle to satisfy
``swarm_runner.AggregatorClient`` Protocol. Production wiring binds
this adapter into ``QueryOrchestrator``; tests inject a stub
transport that round-trips ``AggregateRequest → AggregateResponse``.

Per `docs/2026-05-08-aggregate-rpc-design.md` §"AggregatorClient
adapter" (10-step client-side flow). Each ``aggregate(...)`` call:

  1. Builds a fresh request_id (so retries don't replay).
  2. Converts ``PartialResult``s → ``SignedPartial``s.
  3. Constructs ``AggregateRequest`` with manifest_json, beacon,
     pubkey-hash, budget, deadline.
  4. Signs the request using the prompter's signer callable.
  5. Sends via the injected ``AggregateTransport``.
  6. Verifies response invariants (request_id echo, query_id echo,
     commit's query_id + aggregator_pubkey_hash bind to call args).
  7. Verifies commit_signature with the aggregator's pubkey.
  8. Verifies sha256(decrypted_plaintext) == commit.result_digest (A9
     defense-in-depth — orchestrator's verify_aggregation_commit
     re-checks downstream, but client-side catch is cheaper).
  9. Returns ``(plaintext, commit)`` for swarm_runner to consume.

Verification failures all map to ``AggregationCommitMismatchError``
so the swarm runner / retry-loop routes them to slash uniformly.
Transport-level errors (timeout, decode) bubble up unchanged.

================================================================
KNOWN FOLLOW-ONS (NOT shipped here — separate orchestrator-wiring
tasks tracked in `docs/2026-05-08-query-orchestrator-wiring-readiness.md`)
================================================================

1. ✅ CLOSED — ``source_agent_pubkey`` + ``privacy_budget_consumed``
   threading. ``PartialResult`` now carries both fields directly
   (defaulted to 32 zero bytes + 0.0 for backwards compatibility).
   The adapter threads these straight from PartialResult into
   SignedPartial — no more hardcoded placeholders. Source agents
   set real values when constructing PartialResult; the
   SwarmDispatcherAdapter passes the agent's reported values
   through unchanged.

2. ✅ CLOSED — Real X25519 + XChaCha20-Poly1305 encryption.
   ``partial_result_cipher.py`` now derives a per-request key from
   the prompter's Ed25519 privkey + aggregator's Ed25519 pubkey
   (Ed25519→X25519 conversion via libsodium), HKDF-SHA256 with
   ``info = "prsm:aggregate-cipher:v1\n" || request_id``, and runs
   XChaCha20-Poly1305 over plaintext with AAD = commit signing
   payload. When ``prompter_privkey`` is configured, the adapter
   decrypts; when None, it preserves legacy plaintext-passthrough
   for tests. Honest scope: no forward secrecy yet (ephemeral keys
   would require wire-format extension); marginal value over TLS;
   no replay-cache. See partial_result_cipher.py docstring §"Honest
   scope".

3. ✅ CLOSED — ``ftns_budget`` constructor override.
   ``default_ftns_budget`` is now a constructor kwarg (default
   1000) — operators set it once per deployment. Per-query budget
   threading (where each prompter passes a different budget per
   call) is a separate orchestrator-level design pass: the
   ``AggregatorClient`` Protocol in ``swarm_runner.py`` doesn't
   currently take ``ftns_budget`` as a per-call kwarg, so per-call
   variation requires the Protocol contract to change too —
   tracked as a future swarm_runner Protocol extension.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import time
from typing import Callable, Protocol, Sequence, runtime_checkable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519

from prsm.compute.agents.instruction_set import InstructionManifest
from prsm.compute.query_orchestrator.aggregate_protocol import (
    AggregateRequest,
    AggregateResponse,
    SignedPartial,
)
from prsm.compute.query_orchestrator.aggregator_selector import (
    AggregationCommit,
    AggregationCommitMismatchError,
    StakedNode,
)
from prsm.compute.query_orchestrator.swarm_runner import PartialResult




# ──────────────────────────────────────────────────────────────────────
# Transport contract
# ──────────────────────────────────────────────────────────────────────


@runtime_checkable
class AggregateTransport(Protocol):
    """Pluggable wire transport for AggregateRequest → Response.

    Production wires this against the Phase 6 TLS transport
    (``prsm/compute/chain_rpc/`` server + matching client). Tests
    inject a stub that round-trips a canned response without going
    near the network.

    The transport is responsible for:
      - Routing the request to ``aggregator_node_id`` (typically via
        a node-id → host:port resolver).
      - Wire-encoding via ``AggregateRequest.to_dict`` + JSON.
      - Honoring the timeout (raise on expiry; the adapter does not
        wrap the call in ``asyncio.wait_for`` so the transport is the
        canonical timeout enforcer).
      - Wire-decoding the response into ``AggregateResponse`` and
        validating its structural invariants (32-byte fields, etc) —
        ``AggregateResponse.from_dict`` + ``__post_init__`` already do
        this; the transport just needs to call them.

    Failures from the transport (timeout, decode error, connection
    refused) propagate UNCHANGED so the orchestrator's retry-loop
    can route on them with full fidelity.
    """

    async def send(
        self,
        aggregator_node_id: str,
        request: AggregateRequest,
        timeout_seconds: float,
    ) -> AggregateResponse: ...


# ──────────────────────────────────────────────────────────────────────
# Adapter
# ──────────────────────────────────────────────────────────────────────


class AggregatorClientAdapter:
    """Satisfies ``swarm_runner.AggregatorClient`` Protocol.

    Constructor parameters
    ----------------------
    prompter_pubkey:
        32-byte Ed25519 pubkey of the prompter. Embedded in the request
        so the aggregator can encrypt the plaintext to it (A9 — though
        v1 stub-encrypts; see module docstring §2).
    prompter_node_id:
        String node identifier — routing only; A2 binding key. Selector
        upstream uses this to enforce self-exclusion.
    prompter_signer:
        Callable that signs ``request.signing_payload()`` bytes with
        the prompter's privkey. Decoupling the signer from the adapter
        lets production wire a hardware-key-backed signer without the
        adapter touching key material.
    beacon_provider:
        Callable returning a 32-byte beacon for ``beacon_used`` in the
        request — the A6 forensic anchor. Production wires this against
        the commit-reveal beacon source; tests inject a constant.
    transport:
        Implements ``AggregateTransport`` Protocol. See its docstring.
    request_timeout_seconds:
        Per-request timeout. Threaded into the transport AND into
        ``deadline_unix`` (deadline = now + timeout). Default 60s.
    default_ftns_budget:
        FTNS allocation embedded in every ``AggregateRequest`` this
        adapter constructs. The budget bounds the aggregator's
        per-query compensation; it's enforced at the server side via
        the privacy_budget_consumed accumulator. v1 ships a flat
        per-deployment default — operators set this once at adapter
        construction time. Per-query budget threading (where each
        prompter passes a different budget per call) is a separate
        orchestrator-level design pass; the AggregatorClient Protocol
        doesn't currently take ``ftns_budget`` as a per-call kwarg
        because that would require touching swarm_runner's Protocol
        contract. Default 1000.

    The adapter's ``aggregate(...)`` is the only method the swarm
    runner calls. Each call is independent — the adapter holds no
    per-call state, so it's safe to share one instance across many
    concurrent queries.
    """

    def __init__(
        self,
        *,
        prompter_pubkey: bytes,
        prompter_node_id: str,
        prompter_signer: Callable[[bytes], bytes],
        beacon_provider: Callable[[], bytes],
        transport: AggregateTransport,
        request_timeout_seconds: float = 60.0,
        default_ftns_budget: int = 1000,
        prompter_privkey: bytes | None = None,
    ) -> None:
        if not isinstance(prompter_pubkey, (bytes, bytearray)):
            raise TypeError(
                f"prompter_pubkey must be bytes, got "
                f"{type(prompter_pubkey).__name__}"
            )
        if len(prompter_pubkey) != 32:
            raise ValueError(
                f"prompter_pubkey must be 32 bytes, got "
                f"{len(prompter_pubkey)}"
            )
        if prompter_privkey is not None:
            if not isinstance(prompter_privkey, (bytes, bytearray)):
                raise TypeError(
                    f"prompter_privkey must be bytes, got "
                    f"{type(prompter_privkey).__name__}"
                )
            if len(prompter_privkey) != 32:
                raise ValueError(
                    f"prompter_privkey must be 32 bytes (Ed25519 raw), "
                    f"got {len(prompter_privkey)}"
                )
        if not isinstance(prompter_node_id, str) or not prompter_node_id:
            raise ValueError(
                "prompter_node_id must be a non-empty string"
            )
        if not callable(prompter_signer):
            raise TypeError("prompter_signer must be callable")
        if not callable(beacon_provider):
            raise TypeError("beacon_provider must be callable")
        if not isinstance(transport, AggregateTransport):
            raise TypeError(
                "transport must satisfy AggregateTransport Protocol "
                "(define an `async def send(...)` method)"
            )
        if request_timeout_seconds <= 0:
            raise ValueError(
                f"request_timeout_seconds must be > 0, got "
                f"{request_timeout_seconds}"
            )
        if (
            not isinstance(default_ftns_budget, int)
            or isinstance(default_ftns_budget, bool)
            or default_ftns_budget < 0
        ):
            raise ValueError(
                f"default_ftns_budget must be non-negative int, got "
                f"{default_ftns_budget!r}"
            )

        self._prompter_pubkey = bytes(prompter_pubkey)
        self._prompter_node_id = prompter_node_id
        self._prompter_signer = prompter_signer
        self._beacon_provider = beacon_provider
        self._transport = transport
        self._timeout = float(request_timeout_seconds)
        self._default_ftns_budget = int(default_ftns_budget)
        self._prompter_privkey = (
            bytes(prompter_privkey) if prompter_privkey is not None else None
        )

    async def aggregate(
        self,
        aggregator: StakedNode,
        manifest: InstructionManifest,
        partials: Sequence[PartialResult],
        query_id: bytes,
    ) -> tuple[bytes, AggregationCommit]:
        """Send partials to the selected aggregator + return
        ``(plaintext, AggregationCommit)``.

        Verification order matches the design doc §"Client-side flow":
          1. response.request_id == request.request_id (echo binding)
          2. response.query_id == query_id (echo binding)
          3. commit.query_id == query_id
          4. commit.aggregator_pubkey_hash == aggregator.pubkey_hash
          5. commit_signature verifies under response.aggregator_pubkey
          6. sha256(plaintext) == commit.result_digest

        Any verification failure → ``AggregationCommitMismatchError``.
        Transport failures bubble up untouched.
        """
        if len(query_id) != 32:
            raise ValueError(
                f"query_id must be 32 bytes, got {len(query_id)}"
            )

        # Step 1: fresh request_id per call so retries can't replay.
        request_id = hashlib.sha256(
            query_id + os.urandom(16)
        ).digest()[:32]

        # Step 2: convert PartialResults → SignedPartials. As of the
        # PartialResult schema extension, source_agent_pubkey +
        # privacy_budget_consumed are real fields threaded from the
        # source agent — no more placeholders here. Defaults at the
        # PartialResult level (32 zero bytes + 0.0) keep older
        # callsites working.
        signed_partials = tuple(
            SignedPartial(
                shard_cid=p.shard_cid,
                payload=p.payload,
                creator_id=p.creator_id,
                dp_noise_applied=p.dp_noise_applied,
                source_agent_pubkey=p.source_agent_pubkey,
                source_agent_signature=p.agent_signature,
                privacy_budget_consumed=p.privacy_budget_consumed,
            )
            for p in partials
        )

        # Step 3: build the request. signing_payload doesn't depend on
        # prompter_signature itself, so we build with a placeholder
        # signature, derive the canonical signing bytes, sign them,
        # then dataclasses.replace to swap in the real signature.
        # (Pattern-lift: chain_rpc HandoffToken builds payload from
        # ingredients separately; AggregateRequest's signing_payload is
        # an instance method so we use the placeholder-then-replace
        # idiom against the frozen dataclass.)
        beacon = self._beacon_provider()
        if not isinstance(beacon, (bytes, bytearray)) or len(beacon) != 32:
            raise ValueError(
                f"beacon_provider must return 32 bytes, got "
                f"{type(beacon).__name__} len="
                f"{len(beacon) if hasattr(beacon, '__len__') else 'n/a'}"
            )

        deadline_unix = int(time.time()) + int(self._timeout)
        unsigned_request = AggregateRequest(
            request_id=request_id,
            query_id=query_id,
            manifest_json=manifest.to_json(),
            partials=signed_partials,
            prompter_pubkey=self._prompter_pubkey,
            prompter_node_id=self._prompter_node_id,
            beacon_used=bytes(beacon),
            aggregator_pubkey_hash=aggregator.pubkey_hash,
            ftns_budget=self._default_ftns_budget,
            deadline_unix=deadline_unix,
            prompter_signature=b"\x00" * 64,  # placeholder
        )
        prompter_signature = self._prompter_signer(
            unsigned_request.signing_payload()
        )
        if (
            not isinstance(prompter_signature, (bytes, bytearray))
            or len(prompter_signature) != 64
        ):
            raise ValueError(
                f"prompter_signer must return 64 bytes, got "
                f"{type(prompter_signature).__name__} len="
                f"{len(prompter_signature) if hasattr(prompter_signature, '__len__') else 'n/a'}"
            )
        request = dataclasses.replace(
            unsigned_request,
            prompter_signature=bytes(prompter_signature),
        )

        # Step 4: send via transport.
        response = await self._transport.send(
            aggregator.node_id, request, self._timeout,
        )

        # Step 5: response invariants.
        if response.request_id != request.request_id:
            raise AggregationCommitMismatchError(
                f"response.request_id mismatch (got "
                f"{response.request_id.hex()[:16]}..., expected "
                f"{request.request_id.hex()[:16]}...) — "
                f"INVALID_RESPONSE_ECHO"
            )
        if response.query_id != query_id:
            raise AggregationCommitMismatchError(
                f"response.query_id mismatch (got "
                f"{response.query_id.hex()[:16]}..., expected "
                f"{query_id.hex()[:16]}...) — "
                f"INVALID_RESPONSE_ECHO"
            )
        if response.commit.query_id != query_id:
            raise AggregationCommitMismatchError(
                f"commit.query_id mismatch (got "
                f"{response.commit.query_id.hex()[:16]}..., expected "
                f"{query_id.hex()[:16]}...) — "
                f"INVALID_COMMIT_QUERY_ID"
            )
        if response.commit.aggregator_pubkey_hash != aggregator.pubkey_hash:
            raise AggregationCommitMismatchError(
                f"commit.aggregator_pubkey_hash mismatch (got "
                f"{response.commit.aggregator_pubkey_hash.hex()[:16]}..., "
                f"expected {aggregator.pubkey_hash.hex()[:16]}...) — "
                f"INVALID_AGGREGATOR_IDENTITY"
            )

        # Step 6: verify commit_signature over commit.signing_payload()
        # using response.aggregator_pubkey. We do NOT cross-check
        # aggregator_pubkey vs aggregator.pubkey_hash here — that's the
        # downstream slash-routing's job (anchor lookup + hash compare).
        # The contract here is "the pubkey on the wire signed the
        # commit"; the upstream pubkey_hash bind catches identity
        # substitution.
        try:
            ed25519.Ed25519PublicKey.from_public_bytes(
                response.aggregator_pubkey
            ).verify(
                response.commit_signature,
                response.commit.signing_payload(),
            )
        except (InvalidSignature, ValueError) as exc:
            raise AggregationCommitMismatchError(
                f"INVALID_COMMIT_SIGNATURE — aggregator "
                f"{aggregator.node_id} signature did not verify: {exc}"
            ) from exc

        # Step 7: decrypt via X25519 ECDH + XChaCha20-Poly1305
        # (see partial_result_cipher.py). When prompter_privkey is
        # not configured, we fall back to plaintext-passthrough so
        # legacy test fixtures still round-trip; production wiring
        # always sets prompter_privkey.
        if self._prompter_privkey is not None:
            from prsm.compute.query_orchestrator.partial_result_cipher import (
                PartialResultCipherError,
                decrypt_aggregate_response,
            )
            try:
                plaintext = decrypt_aggregate_response(
                    prompter_ed25519_privkey=self._prompter_privkey,
                    aggregator_ed25519_pubkey=response.aggregator_pubkey,
                    ciphertext=bytes(response.encrypted_plaintext),
                    nonce=bytes(response.nonce),
                    request_id=request.request_id,
                    commit_aad=response.commit.signing_payload(),
                )
            except PartialResultCipherError as exc:
                raise AggregationCommitMismatchError(
                    f"DECRYPT_FAILED — aggregator "
                    f"{aggregator.node_id} ciphertext failed cipher: {exc}"
                ) from exc
        else:
            plaintext = bytes(response.encrypted_plaintext)

        # Step 8: digest check (A9 defense-in-depth).
        if hashlib.sha256(plaintext).digest() != response.commit.result_digest:
            raise AggregationCommitMismatchError(
                f"commit.result_digest "
                f"{response.commit.result_digest.hex()[:16]}... does not "
                f"match sha256(plaintext) — A9 violation by aggregator "
                f"{aggregator.node_id}"
            )

        return plaintext, response.commit
