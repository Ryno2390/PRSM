"""B5 — AggregatorClientAdapter tests.

TDD coverage for the adapter that wraps the AggregateRequest wire
format + a Phase 6 transport handle to satisfy the
``AggregatorClient`` Protocol consumed by ``swarm_runner.run_swarm``.

Per `docs/2026-05-08-aggregate-rpc-design.md` §"Client-side flow":
the adapter constructs the request, signs it, sends it via a
pluggable transport, and verifies the response's commit-signature +
result_digest before returning ``(plaintext, commit)`` to the runner.

Tests use real Ed25519 from ``cryptography.hazmat`` so the signature
path is exercised end-to-end (not mocked away).
"""
from __future__ import annotations

import asyncio
import hashlib
import time

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from prsm.compute.agents.instruction_set import (
    AgentInstruction,
    AgentOp,
    InstructionManifest,
)
from prsm.compute.query_orchestrator import (
    AggregationCommit,
    AggregationCommitMismatchError,
    AggregatorClient,
    PartialResult,
    StakedNode,
)
from prsm.compute.query_orchestrator.aggregate_protocol import (
    AggregateRequest,
    AggregateResponse,
)
from prsm.compute.query_orchestrator.aggregator_client_adapter import (
    AggregateTransport,
    AggregatorClientAdapter,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


def _aggregator_keypair() -> tuple[
    ed25519.Ed25519PrivateKey, bytes, bytes
]:
    """Return (privkey, pubkey_bytes, pubkey_hash)."""
    priv = ed25519.Ed25519PrivateKey.generate()
    pub_bytes = priv.public_key().public_bytes_raw()
    pub_hash = hashlib.sha256(pub_bytes).digest()
    return priv, pub_bytes, pub_hash


def _prompter_keypair() -> tuple[
    ed25519.Ed25519PrivateKey, bytes
]:
    priv = ed25519.Ed25519PrivateKey.generate()
    pub_bytes = priv.public_key().public_bytes_raw()
    return priv, pub_bytes


def _staked_node(pubkey_hash: bytes, *, node_id: str = "agg-1") -> StakedNode:
    return StakedNode(
        node_id=node_id,
        pubkey_hash=pubkey_hash,
        stake_amount_ftns=1_000,
        tier="T2",
        has_tee=True,
        reputation_score=1.0,
    )


def _manifest() -> InstructionManifest:
    return InstructionManifest(
        query="count records",
        instructions=[AgentInstruction(op=AgentOp.COUNT)],
    )


def _partial(*, shard_cid: str = "prsm:shard-0") -> PartialResult:
    return PartialResult(
        shard_cid=shard_cid,
        payload=b"partial-payload",
        agent_signature=b"\x00" * 64,
        creator_id="creator-1",
        dp_noise_applied=True,
    )


class _StubTransport:
    """Records the last request seen + returns a configurable response.

    Tests instantiate this with a ``response_factory`` that takes the
    inbound request and returns the AggregateResponse to send back —
    lets us synthesize valid signatures over the actual request fields.
    """

    def __init__(self, response_factory):
        self.response_factory = response_factory
        self.last_request: AggregateRequest | None = None
        self.call_count: int = 0

    async def send(
        self,
        aggregator_node_id: str,
        request: AggregateRequest,
        timeout_seconds: float,
    ) -> AggregateResponse:
        self.last_request = request
        self.call_count += 1
        return self.response_factory(request)


def _build_response(
    *,
    request: AggregateRequest,
    aggregator_priv: ed25519.Ed25519PrivateKey,
    aggregator_pub: bytes,
    aggregator_pub_hash: bytes,
    plaintext: bytes,
    override_query_id: bytes | None = None,
    override_aggregator_pubkey_hash: bytes | None = None,
    override_result_digest: bytes | None = None,
    override_commit_signature: bytes | None = None,
) -> AggregateResponse:
    """Helper: synthesize a valid AggregateResponse over a given request.

    Override hooks let tests inject one bad field at a time to exercise
    each verification branch.
    """
    commit_query_id = override_query_id or request.query_id
    commit_pkh = override_aggregator_pubkey_hash or aggregator_pub_hash
    digest = override_result_digest or hashlib.sha256(plaintext).digest()
    commit = AggregationCommit(
        query_id=commit_query_id,
        aggregator_pubkey_hash=commit_pkh,
        result_digest=digest,
    )
    if override_commit_signature is not None:
        sig = override_commit_signature
    else:
        sig = aggregator_priv.sign(commit.signing_payload())
    return AggregateResponse(
        request_id=request.request_id,
        query_id=request.query_id,
        commit=commit,
        commit_signature=sig,
        encrypted_plaintext=plaintext,  # v1: plaintext-as-marker
        nonce=b"\x00" * 24,
        aggregator_pubkey=aggregator_pub,
        privacy_budget_consumed=0.5,
        contributing_creators=("creator-1",),
        completed_unix=int(time.time()),
    )


def _make_adapter(
    *,
    transport: AggregateTransport,
    prompter_priv: ed25519.Ed25519PrivateKey,
    prompter_pub: bytes,
    beacon: bytes = b"\xa1" * 32,
) -> AggregatorClientAdapter:
    return AggregatorClientAdapter(
        prompter_pubkey=prompter_pub,
        prompter_node_id="prompter-1",
        prompter_signer=lambda payload: prompter_priv.sign(payload),
        beacon_provider=lambda: beacon,
        transport=transport,
        request_timeout_seconds=30.0,
    )


# ──────────────────────────────────────────────────────────────────────
# 1. Happy path
# ──────────────────────────────────────────────────────────────────────


def test_happy_path_returns_plaintext_and_commit():
    agg_priv, agg_pub, agg_pkh = _aggregator_keypair()
    pp_priv, pp_pub = _prompter_keypair()
    plaintext = b"combined-result-bytes"

    transport = _StubTransport(
        response_factory=lambda req: _build_response(
            request=req,
            aggregator_priv=agg_priv,
            aggregator_pub=agg_pub,
            aggregator_pub_hash=agg_pkh,
            plaintext=plaintext,
        )
    )
    adapter = _make_adapter(
        transport=transport, prompter_priv=pp_priv, prompter_pub=pp_pub,
    )

    aggregator = _staked_node(agg_pkh)
    query_id = b"\x11" * 32

    plaintext_out, commit_out = asyncio.run(
        adapter.aggregate(aggregator, _manifest(), [_partial()], query_id)
    )

    assert plaintext_out == plaintext
    assert commit_out.query_id == query_id
    assert commit_out.aggregator_pubkey_hash == agg_pkh
    assert commit_out.result_digest == hashlib.sha256(plaintext).digest()
    assert transport.call_count == 1


# ──────────────────────────────────────────────────────────────────────
# 2. Request fields populated correctly
# ──────────────────────────────────────────────────────────────────────


def test_request_fields_populated_correctly():
    agg_priv, agg_pub, agg_pkh = _aggregator_keypair()
    pp_priv, pp_pub = _prompter_keypair()
    beacon = b"\xbe" * 32

    transport = _StubTransport(
        response_factory=lambda req: _build_response(
            request=req,
            aggregator_priv=agg_priv,
            aggregator_pub=agg_pub,
            aggregator_pub_hash=agg_pkh,
            plaintext=b"x",
        )
    )
    adapter = AggregatorClientAdapter(
        prompter_pubkey=pp_pub,
        prompter_node_id="prompter-A",
        prompter_signer=lambda payload: pp_priv.sign(payload),
        beacon_provider=lambda: beacon,
        transport=transport,
    )

    aggregator = _staked_node(agg_pkh, node_id="agg-Z")
    manifest = _manifest()
    partials = [_partial(shard_cid="s0"), _partial(shard_cid="s1")]
    query_id = b"\x22" * 32

    asyncio.run(adapter.aggregate(aggregator, manifest, partials, query_id))

    req = transport.last_request
    assert req is not None
    assert req.query_id == query_id
    assert req.aggregator_pubkey_hash == agg_pkh
    assert req.prompter_pubkey == pp_pub
    assert req.prompter_node_id == "prompter-A"
    assert req.beacon_used == beacon
    assert len(req.partials) == 2
    assert req.partials[0].shard_cid == "s0"
    assert req.partials[1].shard_cid == "s1"
    # manifest_json round-trips
    rebuilt = InstructionManifest.from_json(req.manifest_json)
    assert rebuilt.query == manifest.query
    # prompter_signature verifies over signing_payload
    pp_priv.public_key().verify(req.prompter_signature, req.signing_payload())


# ──────────────────────────────────────────────────────────────────────
# 3. Request ID is fresh per call
# ──────────────────────────────────────────────────────────────────────


def test_request_id_is_fresh_per_call():
    agg_priv, agg_pub, agg_pkh = _aggregator_keypair()
    pp_priv, pp_pub = _prompter_keypair()

    seen_request_ids: list[bytes] = []

    def _factory(req: AggregateRequest) -> AggregateResponse:
        seen_request_ids.append(req.request_id)
        return _build_response(
            request=req,
            aggregator_priv=agg_priv,
            aggregator_pub=agg_pub,
            aggregator_pub_hash=agg_pkh,
            plaintext=b"y",
        )

    transport = _StubTransport(response_factory=_factory)
    adapter = _make_adapter(
        transport=transport, prompter_priv=pp_priv, prompter_pub=pp_pub,
    )

    aggregator = _staked_node(agg_pkh)
    query_id = b"\x33" * 32
    asyncio.run(adapter.aggregate(aggregator, _manifest(), [_partial()], query_id))
    asyncio.run(adapter.aggregate(aggregator, _manifest(), [_partial()], query_id))

    assert len(seen_request_ids) == 2
    assert seen_request_ids[0] != seen_request_ids[1]


# ──────────────────────────────────────────────────────────────────────
# 4. Wrong query_id in response → AggregationCommitMismatchError
# ──────────────────────────────────────────────────────────────────────


def test_response_query_id_mismatch_raises():
    agg_priv, agg_pub, agg_pkh = _aggregator_keypair()
    pp_priv, pp_pub = _prompter_keypair()
    plaintext = b"z"

    def _factory(req: AggregateRequest) -> AggregateResponse:
        # Build a response with a commit whose query_id differs from
        # what the adapter asked for. We cannot directly tamper with
        # AggregateResponse.query_id (it's bound to request_id at the
        # adapter via echo) so we tamper with the commit's query_id.
        bad_qid = b"\xee" * 32
        commit = AggregationCommit(
            query_id=bad_qid,
            aggregator_pubkey_hash=agg_pkh,
            result_digest=hashlib.sha256(plaintext).digest(),
        )
        sig = agg_priv.sign(commit.signing_payload())
        return AggregateResponse(
            request_id=req.request_id,
            query_id=req.query_id,
            commit=commit,
            commit_signature=sig,
            encrypted_plaintext=plaintext,
            nonce=b"\x00" * 24,
            aggregator_pubkey=agg_pub,
            privacy_budget_consumed=0.0,
            contributing_creators=("creator-1",),
            completed_unix=int(time.time()),
        )

    transport = _StubTransport(response_factory=_factory)
    adapter = _make_adapter(
        transport=transport, prompter_priv=pp_priv, prompter_pub=pp_pub,
    )

    aggregator = _staked_node(agg_pkh)
    query_id = b"\x44" * 32
    with pytest.raises(AggregationCommitMismatchError):
        asyncio.run(
            adapter.aggregate(aggregator, _manifest(), [_partial()], query_id)
        )


# ──────────────────────────────────────────────────────────────────────
# 5. Wrong aggregator_pubkey_hash in commit → AggregationCommitMismatchError
# ──────────────────────────────────────────────────────────────────────


def test_response_aggregator_pubkey_hash_mismatch_raises():
    agg_priv, agg_pub, agg_pkh = _aggregator_keypair()
    pp_priv, pp_pub = _prompter_keypair()
    plaintext = b"w"

    wrong_pkh = b"\xcc" * 32

    transport = _StubTransport(
        response_factory=lambda req: _build_response(
            request=req,
            aggregator_priv=agg_priv,
            aggregator_pub=agg_pub,
            aggregator_pub_hash=agg_pkh,
            plaintext=plaintext,
            override_aggregator_pubkey_hash=wrong_pkh,
        )
    )
    adapter = _make_adapter(
        transport=transport, prompter_priv=pp_priv, prompter_pub=pp_pub,
    )

    aggregator = _staked_node(agg_pkh)
    query_id = b"\x55" * 32
    with pytest.raises(AggregationCommitMismatchError):
        asyncio.run(
            adapter.aggregate(aggregator, _manifest(), [_partial()], query_id)
        )


# ──────────────────────────────────────────────────────────────────────
# 6. Bad commit signature → AggregationCommitMismatchError
# ──────────────────────────────────────────────────────────────────────


def test_response_bad_commit_signature_raises():
    agg_priv, agg_pub, agg_pkh = _aggregator_keypair()
    pp_priv, pp_pub = _prompter_keypair()
    plaintext = b"q"

    bogus_sig = b"\x00" * 64

    transport = _StubTransport(
        response_factory=lambda req: _build_response(
            request=req,
            aggregator_priv=agg_priv,
            aggregator_pub=agg_pub,
            aggregator_pub_hash=agg_pkh,
            plaintext=plaintext,
            override_commit_signature=bogus_sig,
        )
    )
    adapter = _make_adapter(
        transport=transport, prompter_priv=pp_priv, prompter_pub=pp_pub,
    )

    aggregator = _staked_node(agg_pkh)
    query_id = b"\x66" * 32
    with pytest.raises(AggregationCommitMismatchError):
        asyncio.run(
            adapter.aggregate(aggregator, _manifest(), [_partial()], query_id)
        )


# ──────────────────────────────────────────────────────────────────────
# 7. Plaintext digest mismatch → AggregationCommitMismatchError
# ──────────────────────────────────────────────────────────────────────


def test_response_plaintext_digest_mismatch_raises():
    agg_priv, agg_pub, agg_pkh = _aggregator_keypair()
    pp_priv, pp_pub = _prompter_keypair()
    plaintext = b"p"

    # commit's result_digest is over different bytes than the plaintext
    # actually delivered. Server signs the lying commit; client should
    # catch via sha256(plaintext) != commit.result_digest.
    wrong_digest = hashlib.sha256(b"different-bytes").digest()

    transport = _StubTransport(
        response_factory=lambda req: _build_response(
            request=req,
            aggregator_priv=agg_priv,
            aggregator_pub=agg_pub,
            aggregator_pub_hash=agg_pkh,
            plaintext=plaintext,
            override_result_digest=wrong_digest,
        )
    )
    adapter = _make_adapter(
        transport=transport, prompter_priv=pp_priv, prompter_pub=pp_pub,
    )

    aggregator = _staked_node(agg_pkh)
    query_id = b"\x77" * 32
    with pytest.raises(AggregationCommitMismatchError):
        asyncio.run(
            adapter.aggregate(aggregator, _manifest(), [_partial()], query_id)
        )


# ──────────────────────────────────────────────────────────────────────
# 8. Adapter satisfies AggregatorClient Protocol (runtime_checkable)
# ──────────────────────────────────────────────────────────────────────


def test_adapter_satisfies_aggregator_client_protocol():
    agg_priv, agg_pub, agg_pkh = _aggregator_keypair()
    pp_priv, pp_pub = _prompter_keypair()

    transport = _StubTransport(
        response_factory=lambda req: _build_response(
            request=req,
            aggregator_priv=agg_priv,
            aggregator_pub=agg_pub,
            aggregator_pub_hash=agg_pkh,
            plaintext=b"x",
        )
    )
    adapter = _make_adapter(
        transport=transport, prompter_priv=pp_priv, prompter_pub=pp_pub,
    )
    assert isinstance(adapter, AggregatorClient)


# ──────────────────────────────────────────────────────────────────────
# default_ftns_budget — closes the §3 placeholder follow-on
# ──────────────────────────────────────────────────────────────────────


def test_default_ftns_budget_default_is_1000():
    """Backwards compat: omit the kwarg → default 1000 (matches the
    old _FTNS_BUDGET_PLACEHOLDER value)."""
    agg_priv, agg_pub, agg_pkh = _aggregator_keypair()
    pp_priv, pp_pub = _prompter_keypair()
    captured = {}

    def factory(req):
        captured["request"] = req
        return _build_response(
            request=req, aggregator_priv=agg_priv, aggregator_pub=agg_pub,
            aggregator_pub_hash=agg_pkh, plaintext=b"x",
        )

    transport = _StubTransport(response_factory=factory)
    adapter = AggregatorClientAdapter(
        prompter_pubkey=pp_pub,
        prompter_node_id="p",
        prompter_signer=pp_priv.sign,
        beacon_provider=lambda: b"\xa1" * 32,
        transport=transport,
    )
    aggregator = StakedNode(
        node_id="agg-1",
        pubkey_hash=agg_pkh,
        stake_amount_ftns=1000,
        tier="T2",
        has_tee=False,
        reputation_score=1.0,
    )
    asyncio.run(adapter.aggregate(
        aggregator=aggregator,
        manifest=_manifest(),
        partials=[_partial(shard_cid="s-0")],
        query_id=b"q" * 32,
    ))
    assert captured["request"].ftns_budget == 1000


def test_default_ftns_budget_constructor_override_threads_through():
    """Operator sets a non-default budget; it lands in the
    AggregateRequest the adapter constructs."""
    agg_priv, agg_pub, agg_pkh = _aggregator_keypair()
    pp_priv, pp_pub = _prompter_keypair()
    captured = {}

    def factory(req):
        captured["request"] = req
        return _build_response(
            request=req, aggregator_priv=agg_priv, aggregator_pub=agg_pub,
            aggregator_pub_hash=agg_pkh, plaintext=b"x",
        )

    transport = _StubTransport(response_factory=factory)
    adapter = AggregatorClientAdapter(
        prompter_pubkey=pp_pub,
        prompter_node_id="p",
        prompter_signer=pp_priv.sign,
        beacon_provider=lambda: b"\xa1" * 32,
        transport=transport,
        default_ftns_budget=5_000,
    )
    aggregator = StakedNode(
        node_id="agg-1",
        pubkey_hash=agg_pkh,
        stake_amount_ftns=1000,
        tier="T2",
        has_tee=False,
        reputation_score=1.0,
    )
    asyncio.run(adapter.aggregate(
        aggregator=aggregator,
        manifest=_manifest(),
        partials=[_partial(shard_cid="s-0")],
        query_id=b"q" * 32,
    ))
    assert captured["request"].ftns_budget == 5_000


def test_negative_default_ftns_budget_rejected():
    pp_priv, pp_pub = _prompter_keypair()
    transport = _StubTransport(response_factory=lambda req: None)
    with pytest.raises(ValueError, match="default_ftns_budget"):
        AggregatorClientAdapter(
            prompter_pubkey=pp_pub,
            prompter_node_id="p",
            prompter_signer=pp_priv.sign,
            beacon_provider=lambda: b"\xa1" * 32,
            transport=transport,
            default_ftns_budget=-1,
        )
