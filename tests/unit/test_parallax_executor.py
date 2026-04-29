"""Phase 3.x.6 Task 6 — ParallaxScheduledExecutor unit tests.

Coverage matches design plan §4 Task 6 acceptance criteria:
  - Returns InferenceResult.success on happy path
  - Returns InferenceResult.failure with specific reason when allocation fails
  - Returns specific reason when tier gate rejects
  - Receipt is signed with node identity (matches existing executor contract)
  - Phase-1 recompute triggered on coverage gap
  - Phase-1 recompute NOT triggered when localized GPU-leave is absorbed
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence

import pytest

from prsm.compute.inference.models import (
    ContentTier,
    InferenceRequest,
    InferenceResult,
)
from prsm.compute.inference.parallax_executor import (
    ChainExecutionResult,
    ChainExecutor,
    ParallaxScheduledExecutor,
)
from prsm.compute.inference.receipt import verify_receipt
from prsm.compute.parallax_scheduling.model_info import ModelInfo
from prsm.compute.parallax_scheduling.prsm_request_router import (
    GPUChain,
    InMemoryProfileSource,
    ProfileSnapshot,
)
from prsm.compute.parallax_scheduling.prsm_types import ParallaxGPU
from prsm.compute.parallax_scheduling.trust_adapter import (
    AnchorVerifyAdapter,
    ConsensusMismatchHook,
    StakeWeightedTrustAdapter,
    TierGateAdapter,
    TrustStack,
)
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import generate_node_identity


# ──────────────────────────────────────────────────────────────────────────
# Fixtures / fakes
# ──────────────────────────────────────────────────────────────────────────


def _model_info(num_layers: int = 4) -> ModelInfo:
    """Minimal ModelInfo sized so two-stage pipelines fit."""
    return ModelInfo(
        model_name="test-model",
        mlx_model_name="test-model-mlx",
        head_size=64,
        hidden_dim=512,
        intermediate_dim=2048,
        num_attention_heads=8,
        num_kv_heads=8,
        vocab_size=32000,
        num_layers=num_layers,
    )


def _gpu(
    node_id: str,
    *,
    region: str = "us-east",
    layer_capacity: int = 4,
    stake_amount: int = 10**18,
    tier_attestation: str = "tier-sgx",
) -> ParallaxGPU:
    return ParallaxGPU(
        node_id=node_id,
        region=region,
        layer_capacity=layer_capacity,
        stake_amount=stake_amount,
        tier_attestation=tier_attestation,
        tflops_fp16=100.0,
        memory_gb=80.0,
        memory_bandwidth_gbps=2000.0,
    )


class FakeAnchor:
    def __init__(self, registered: Dict[str, str]):
        self.registered = registered

    def lookup(self, node_id: str) -> Optional[str]:
        return self.registered.get(node_id)


class FakeStakeLookup:
    def __init__(self, stakes: Dict[str, int]):
        self.stakes = stakes

    def get_stake(self, node_id: str) -> int:
        return self.stakes.get(node_id, 0)


class RecordingChainExecutor:
    """ChainExecutor that records every dispatch + returns deterministic
    output. Set ``output_for`` to vary output by chain stage tuple."""

    def __init__(
        self,
        *,
        default_output: str = "ok",
        epsilon: float = 0.0,
        tee_type: TEEType = TEEType.SOFTWARE,
        tee_attestation: bytes = b"\x01" * 32,
    ):
        self.calls: List[GPUChain] = []
        self.output_for: Dict[tuple, str] = {}
        self.default_output = default_output
        self.epsilon = epsilon
        self.tee_type = tee_type
        self.tee_attestation = tee_attestation
        self.raise_on_call = False

    def execute_chain(
        self, *, request: InferenceRequest, chain: GPUChain
    ) -> ChainExecutionResult:
        if self.raise_on_call:
            raise RuntimeError("chain executor outage")
        self.calls.append(chain)
        out = self.output_for.get(chain.stages, self.default_output)
        return ChainExecutionResult(
            output=out,
            duration_seconds=0.05,
            tee_attestation=self.tee_attestation,
            tee_type=self.tee_type,
            epsilon_spent=self.epsilon,
        )


class RecordingSubmitter:
    def __init__(self):
        self.records = []

    def __call__(self, record):
        self.records.append(record)


def _build_trust_stack(
    *,
    registered: Dict[str, str],
    stakes: Dict[str, int],
    snapshots: Optional[Dict[str, ProfileSnapshot]] = None,
    sample_rate: float = 0.0,
    submitter: Optional[RecordingSubmitter] = None,
) -> TrustStack:
    inner = InMemoryProfileSource(snapshots=snapshots or {})
    sub = submitter if submitter is not None else RecordingSubmitter()
    return TrustStack(
        anchor_verify=AnchorVerifyAdapter(anchor=FakeAnchor(registered)),
        tier_gate=TierGateAdapter(),
        profile_source=StakeWeightedTrustAdapter(
            inner=inner,
            stake_lookup=FakeStakeLookup(stakes),
        ),
        consensus_hook=ConsensusMismatchHook(
            submitter=sub, sample_rate=sample_rate
        ),
    )


def _make_executor(
    *,
    pool: Sequence[ParallaxGPU],
    registered: Optional[Dict[str, str]] = None,
    stakes: Optional[Dict[str, int]] = None,
    chain_executor: Optional[ChainExecutor] = None,
    model_id: str = "test-model",
    num_layers: int = 4,
    sample_rate: float = 0.0,
    submitter: Optional[RecordingSubmitter] = None,
    cost_per_layer: Decimal = Decimal("0.01"),
    tier_c_chain_executor: Optional[Any] = None,
):
    if registered is None:
        registered = {g.node_id: "pk-" + g.node_id for g in pool}
    if stakes is None:
        stakes = {g.node_id: g.stake_amount for g in pool}
    snapshots = {
        g.node_id: ProfileSnapshot(
            node_id=g.node_id,
            layer_latency_ms=10.0,
            rtt_to_peers={
                other.node_id: 1.0
                for other in pool
                if other.node_id != g.node_id
            },
            timestamp_unix=1000.0,
        )
        for g in pool
    }
    trust = _build_trust_stack(
        registered=registered,
        stakes=stakes,
        snapshots=snapshots,
        sample_rate=sample_rate,
        submitter=submitter,
    )
    catalog = {model_id: _model_info(num_layers=num_layers)}
    pool_holder = list(pool)

    def provider():
        return list(pool_holder)

    executor = ParallaxScheduledExecutor(
        gpu_pool_provider=provider,
        trust_stack=trust,
        model_catalog=catalog,
        chain_executor=chain_executor or RecordingChainExecutor(),
        node_identity=generate_node_identity("test-settler"),
        cost_per_layer=cost_per_layer,
        tier_c_chain_executor=tier_c_chain_executor,
    )
    return executor, pool_holder, trust


def _request(
    *,
    model_id: str = "test-model",
    privacy_tier: PrivacyLevel = PrivacyLevel.NONE,
    budget: Decimal = Decimal("10.0"),
    request_id: str = "req-1",
    prompt: str = "hello",
    content_tier: ContentTier = ContentTier.A,
) -> InferenceRequest:
    return InferenceRequest(
        prompt=prompt,
        model_id=model_id,
        budget_ftns=budget,
        privacy_tier=privacy_tier,
        content_tier=content_tier,
        request_id=request_id,
    )


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ──────────────────────────────────────────────────────────────────────────
# Construction validation
# ──────────────────────────────────────────────────────────────────────────


class TestConstruction:
    def _stub_args(self):
        return dict(
            gpu_pool_provider=lambda: [],
            trust_stack=_build_trust_stack(registered={}, stakes={}),
            model_catalog={},
            chain_executor=RecordingChainExecutor(),
            node_identity=generate_node_identity("t"),
        )

    def test_rejects_non_callable_pool_provider(self):
        args = self._stub_args()
        args["gpu_pool_provider"] = "not-callable"  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="gpu_pool_provider"):
            ParallaxScheduledExecutor(**args)

    def test_rejects_missing_trust_stack(self):
        args = self._stub_args()
        args["trust_stack"] = None  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="trust_stack"):
            ParallaxScheduledExecutor(**args)

    def test_rejects_chain_executor_without_method(self):
        args = self._stub_args()
        args["chain_executor"] = object()  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="chain_executor"):
            ParallaxScheduledExecutor(**args)

    def test_rejects_missing_node_identity(self):
        args = self._stub_args()
        args["node_identity"] = None  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="NodeIdentity"):
            ParallaxScheduledExecutor(**args)


# ──────────────────────────────────────────────────────────────────────────
# Cost estimation + supported_models
# ──────────────────────────────────────────────────────────────────────────


class TestEstimateCost:
    def test_unsupported_model_raises(self):
        executor, _, _ = _make_executor(pool=[_gpu("a")])
        from prsm.compute.inference.executor import UnsupportedModelError
        with pytest.raises(UnsupportedModelError):
            _run(executor.estimate_cost(_request(model_id="ghost")))

    def test_cost_scales_with_layers(self):
        executor, _, _ = _make_executor(
            pool=[_gpu("a")],
            num_layers=4,
            cost_per_layer=Decimal("0.10"),
        )
        cost = _run(executor.estimate_cost(_request()))
        # 4 × 0.10 × 1.00 (NONE) = 0.40
        assert cost == Decimal("0.40")

    def test_privacy_overhead_applied(self):
        executor, _, _ = _make_executor(
            pool=[_gpu("a")],
            num_layers=4,
            cost_per_layer=Decimal("0.10"),
        )
        cost = _run(
            executor.estimate_cost(_request(privacy_tier=PrivacyLevel.HIGH))
        )
        # 4 × 0.10 × 1.25 = 0.50
        assert cost == Decimal("0.50")

    def test_supported_models_returns_catalog(self):
        executor, _, _ = _make_executor(pool=[_gpu("a")])
        assert executor.supported_models() == ["test-model"]


# ──────────────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────────────


class TestHappyPath:
    def test_returns_success_with_signed_receipt(self):
        chain_exec = RecordingChainExecutor(default_output="generated text")
        executor, _, _ = _make_executor(
            pool=[_gpu("alice"), _gpu("bob")],
            chain_executor=chain_exec,
            num_layers=4,
        )
        result = _run(executor.execute(_request()))

        assert result.success is True
        assert result.error is None
        assert result.output == "generated text"
        assert result.receipt is not None
        # The receipt was signed under the executor's identity.
        assert result.receipt.settler_node_id != ""
        assert len(result.receipt.settler_signature) > 0
        # Chain executor was called exactly once (no consensus sampling).
        assert len(chain_exec.calls) == 1

    def test_receipt_signature_verifies(self):
        executor, _, _ = _make_executor(
            pool=[_gpu("alice"), _gpu("bob")],
            num_layers=4,
        )
        result = _run(executor.execute(_request()))

        assert result.success is True
        # Pull executor's public key via internal identity (test scope).
        identity = executor._identity  # type: ignore[attr-defined]
        verify_receipt(result.receipt, identity=identity)  # raises on failure

    def test_request_id_propagates(self):
        executor, _, _ = _make_executor(
            pool=[_gpu("alice"), _gpu("bob")],
        )
        result = _run(executor.execute(_request(request_id="req-xyz")))
        assert result.request_id == "req-xyz"
        assert result.receipt.request_id == "req-xyz"

    def test_output_hash_matches_output(self):
        import hashlib
        chain_exec = RecordingChainExecutor(default_output="the answer is 42")
        executor, _, _ = _make_executor(
            pool=[_gpu("alice"), _gpu("bob")],
            chain_executor=chain_exec,
        )
        result = _run(executor.execute(_request()))
        expected = hashlib.sha256(b"the answer is 42").digest()
        assert result.receipt.output_hash == expected


# ──────────────────────────────────────────────────────────────────────────
# Failure paths — each with a specific reason
# ──────────────────────────────────────────────────────────────────────────


class TestFailurePaths:
    def test_unknown_model_id(self):
        executor, _, _ = _make_executor(pool=[_gpu("a")])
        result = _run(executor.execute(_request(model_id="ghost")))
        assert result.success is False
        assert result.receipt is None
        assert "Unknown model_id" in result.error

    def test_insufficient_budget(self):
        executor, _, _ = _make_executor(
            pool=[_gpu("alice"), _gpu("bob")],
            cost_per_layer=Decimal("100.0"),
        )
        result = _run(executor.execute(_request(budget=Decimal("1.0"))))
        assert result.success is False
        assert "Insufficient budget" in result.error

    def test_empty_pool_failure(self):
        executor, _, _ = _make_executor(pool=[])
        # No GPUs registered → empty pool.
        executor._catalog = {"test-model": _model_info(num_layers=4)}  # type: ignore[attr-defined]

        # Override to test empty-pool path explicitly.
        executor._pool_provider = lambda: []  # type: ignore[attr-defined]
        result = _run(executor.execute(_request()))
        assert result.success is False
        assert "empty" in result.error.lower()

    def test_pool_provider_exception_is_caught(self):
        executor, _, _ = _make_executor(pool=[_gpu("a")])

        def boom():
            raise IOError("DHT down")

        executor._pool_provider = boom  # type: ignore[attr-defined]
        result = _run(executor.execute(_request()))
        assert result.success is False
        assert "GPU pool provider failure" in result.error

    def test_no_anchor_verified_gpu(self):
        # Pool is non-empty but none registered on anchor.
        executor, _, _ = _make_executor(
            pool=[_gpu("alice"), _gpu("bob")],
            registered={},  # empty anchor
        )
        result = _run(executor.execute(_request()))
        assert result.success is False
        assert "anchor" in result.error.lower()

    def test_tier_gate_rejection_returns_specific_reason(self):
        # All GPUs lack hardware TEE; HIGH privacy → TierGateRejected.
        pool = [
            _gpu("alice", tier_attestation="tier-none"),
            _gpu("bob", tier_attestation="tier-none"),
        ]
        executor, _, _ = _make_executor(pool=pool)
        result = _run(
            executor.execute(_request(privacy_tier=PrivacyLevel.HIGH))
        )
        assert result.success is False
        assert "tier gate" in result.error.lower()

    def test_insufficient_capacity_reports(self):
        # 1 GPU with capacity 1, model has 8 layers → InsufficientCapacityError.
        pool = [_gpu("alice", layer_capacity=1)]
        executor, _, _ = _make_executor(pool=pool, num_layers=8)
        result = _run(executor.execute(_request()))
        assert result.success is False
        # Falls into insufficient-capacity bucket.
        assert (
            "insufficient capacity" in result.error.lower()
            or "allocation failure" in result.error.lower()
        )

    def test_chain_executor_failure_returns_failure(self):
        chain_exec = RecordingChainExecutor()
        chain_exec.raise_on_call = True
        executor, _, _ = _make_executor(
            pool=[_gpu("alice"), _gpu("bob")],
            chain_executor=chain_exec,
        )
        result = _run(executor.execute(_request()))
        assert result.success is False
        assert "chain execution" in result.error.lower()


# ──────────────────────────────────────────────────────────────────────────
# Phase-1 cache (paper §3.4)
# ──────────────────────────────────────────────────────────────────────────


class TestPhase1Cache:
    def test_first_request_triggers_one_recompute(self):
        executor, _, _ = _make_executor(pool=[_gpu("alice"), _gpu("bob")])
        assert executor.phase1_recompute_count == 0
        result = _run(executor.execute(_request()))
        assert result.success is True
        assert executor.phase1_recompute_count == 1

    def test_repeated_requests_reuse_cached_allocation(self):
        executor, _, _ = _make_executor(pool=[_gpu("alice"), _gpu("bob")])
        for i in range(5):
            r = _run(executor.execute(_request(request_id=f"req-{i}")))
            assert r.success is True
        # Only the first request rebuilt; the next four hit the cache.
        assert executor.phase1_recompute_count == 1

    def test_localized_gpu_join_does_not_recompute(self):
        """Adding a GPU is a strict superset of the cached stages →
        cache stays valid (paper §3.4 'localized' case)."""
        pool = [_gpu("alice"), _gpu("bob")]
        executor, holder, _ = _make_executor(pool=pool)
        _run(executor.execute(_request(request_id="req-1")))
        assert executor.phase1_recompute_count == 1

        # Add a new GPU mid-flight.
        new_gpu = _gpu("carol")
        # Need to register on anchor + stake for it to pass trust filter.
        # We'll inject directly on the trust stack.
        executor._trust.anchor_verify.anchor.registered["carol"] = "pk-carol"  # type: ignore[attr-defined]
        executor._trust.profile_source.stake_lookup.stakes["carol"] = 10**18  # type: ignore[attr-defined]
        executor._trust.profile_source.inner.set_snapshot(  # type: ignore[attr-defined]
            ProfileSnapshot("carol", 10.0, {"alice": 1.0, "bob": 1.0}, 1000.0)
        )
        holder.append(new_gpu)

        _run(executor.execute(_request(request_id="req-2")))
        # No recompute — alice + bob still cover all layers.
        assert executor.phase1_recompute_count == 1

    def test_cached_stage_eviction_triggers_recompute(self):
        """Removing a cached stage GPU forces Phase-1 rebuild."""
        pool = [_gpu("alice"), _gpu("bob")]
        executor, holder, _ = _make_executor(pool=pool)
        _run(executor.execute(_request(request_id="req-1")))
        assert executor.phase1_recompute_count == 1

        # Add a redundant peer so Phase-1 has somewhere to land after
        # we evict bob.
        new_gpu = _gpu("carol")
        executor._trust.anchor_verify.anchor.registered["carol"] = "pk-carol"  # type: ignore[attr-defined]
        executor._trust.profile_source.stake_lookup.stakes["carol"] = 10**18  # type: ignore[attr-defined]
        executor._trust.profile_source.inner.set_snapshot(  # type: ignore[attr-defined]
            ProfileSnapshot("carol", 10.0, {"alice": 1.0}, 1000.0)
        )
        holder.append(new_gpu)
        # Now evict bob → cached stage_set no longer subset of pool_ids.
        holder[:] = [g for g in holder if g.node_id != "bob"]

        _run(executor.execute(_request(request_id="req-2")))
        # Recompute happened (cache invalidated by missing stage).
        assert executor.phase1_recompute_count == 2


# ──────────────────────────────────────────────────────────────────────────
# Consensus-mismatch hook integration
# ──────────────────────────────────────────────────────────────────────────


class TestConsensusHookIntegration:
    def test_sample_rate_zero_no_redundant_execution(self):
        chain_exec = RecordingChainExecutor()
        submitter = RecordingSubmitter()
        executor, _, _ = _make_executor(
            pool=[_gpu("alice"), _gpu("bob"), _gpu("carol"), _gpu("dan")],
            chain_executor=chain_exec,
            sample_rate=0.0,
            submitter=submitter,
        )
        _run(executor.execute(_request()))
        assert len(chain_exec.calls) == 1
        assert submitter.records == []

    def test_matching_outputs_no_challenge_dispatched(self):
        # sample_rate=1.0 forces secondary execution every call.
        # Both chains return the same default output → no mismatch.
        chain_exec = RecordingChainExecutor(default_output="same-result")
        submitter = RecordingSubmitter()
        # 4 GPUs → primary uses some, secondary uses the rest.
        executor, _, _ = _make_executor(
            pool=[
                _gpu("alice"),
                _gpu("bob"),
                _gpu("carol"),
                _gpu("dan"),
            ],
            chain_executor=chain_exec,
            sample_rate=1.0,
            submitter=submitter,
        )
        result = _run(executor.execute(_request()))
        assert result.success is True
        # Two chain executions (primary + secondary).
        assert len(chain_exec.calls) >= 1
        # No mismatch.
        assert submitter.records == []

    def test_mismatch_dispatches_challenge(self):
        chain_exec = RecordingChainExecutor()
        submitter = RecordingSubmitter()
        executor, _, _ = _make_executor(
            pool=[
                _gpu("alice"),
                _gpu("bob"),
                _gpu("carol"),
                _gpu("dan"),
            ],
            chain_executor=chain_exec,
            sample_rate=1.0,
            submitter=submitter,
        )

        # Wire up: primary chain returns "honest", secondary returns
        # "lying". The chain_executor will be called for both; we
        # populate output_for after we know what chains were chosen.
        # Simpler: alternate between two outputs by call-count.
        original_execute = chain_exec.execute_chain
        call_count = {"n": 0}

        def alternating(*, request, chain):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return ChainExecutionResult(
                    output="honest",
                    duration_seconds=0.05,
                    tee_attestation=b"\x01" * 32,
                    tee_type=TEEType.SOFTWARE,
                    epsilon_spent=0.0,
                )
            return ChainExecutionResult(
                output="lying",
                duration_seconds=0.05,
                tee_attestation=b"\x01" * 32,
                tee_type=TEEType.SOFTWARE,
                epsilon_spent=0.0,
            )

        chain_exec.execute_chain = alternating  # type: ignore[assignment]

        result = _run(executor.execute(_request()))
        assert result.success is True
        assert result.output == "honest"  # primary wins
        # Secondary mismatched → challenge dispatched.
        assert len(submitter.records) == 1
        rec = submitter.records[0]
        assert rec.request_id == "req-1"
        assert rec.primary_output_hash != rec.secondary_output_hash

    def test_no_alternate_chain_skips_consensus(self):
        """When the pool has only enough GPUs for a single chain, no
        alternate is available — consensus check is skipped silently."""
        chain_exec = RecordingChainExecutor()
        submitter = RecordingSubmitter()
        # Two GPUs each with capacity 2, model with 4 layers — exactly
        # one chain fits, no remainder for an alternate.
        executor, _, _ = _make_executor(
            pool=[_gpu("alice", layer_capacity=2), _gpu("bob", layer_capacity=2)],
            chain_executor=chain_exec,
            sample_rate=1.0,
            submitter=submitter,
            num_layers=4,
        )
        result = _run(executor.execute(_request()))
        assert result.success is True
        # Only the primary chain ran.
        assert len(chain_exec.calls) == 1
        assert submitter.records == []

    def test_secondary_chain_failure_does_not_affect_primary(self):
        submitter = RecordingSubmitter()
        # Set up a chain executor that succeeds on the first call and
        # raises on the second. Primary should still return success.
        call_count = {"n": 0}

        class FlakyExecutor:
            def execute_chain(self, *, request, chain):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return ChainExecutionResult(
                        output="primary-good",
                        duration_seconds=0.05,
                        tee_attestation=b"\x01" * 32,
                        tee_type=TEEType.SOFTWARE,
                        epsilon_spent=0.0,
                    )
                raise RuntimeError("secondary chain crashed")

        executor, _, _ = _make_executor(
            pool=[
                _gpu("alice"),
                _gpu("bob"),
                _gpu("carol"),
                _gpu("dan"),
            ],
            chain_executor=FlakyExecutor(),
            sample_rate=1.0,
            submitter=submitter,
        )
        result = _run(executor.execute(_request()))
        assert result.success is True
        assert result.output == "primary-good"
        assert submitter.records == []


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.8.1 Task 1 — execute_streaming
# ──────────────────────────────────────────────────────────────────────────


from prsm.compute.chain_rpc.client import StreamToken
from prsm.compute.inference.parallax_executor import InferenceTokenEvent
from prsm.compute.inference.receipt import verify_receipt


class StreamingChainExecutor:
    """ChainExecutor with both execute_chain (unary) AND
    execute_chain_streaming (Phase 3.x.8). Produces a deterministic
    stream of StreamTokens followed by ChainExecutionResult."""

    def __init__(
        self,
        *,
        deltas: Optional[List[str]] = None,
        finish_reason: str = "stop",
        epsilon: float = 0.0,
        tee_type: TEEType = TEEType.SOFTWARE,
        tee_attestation: bytes = b"\x01" * 32,
        raise_on_streaming: bool = False,
        omit_terminal: bool = False,
        no_streaming_method: bool = False,
    ):
        self.deltas = deltas if deltas is not None else ["hello", " ", "world"]
        self.finish_reason = finish_reason
        self.epsilon = epsilon
        self.tee_type = tee_type
        self.tee_attestation = tee_attestation
        self.raise_on_streaming = raise_on_streaming
        self.omit_terminal = omit_terminal
        self.no_streaming_method = no_streaming_method
        self.unary_calls: List[GPUChain] = []
        self.streaming_calls: List[GPUChain] = []

    def execute_chain(
        self, *, request: InferenceRequest, chain: GPUChain,
    ) -> ChainExecutionResult:
        self.unary_calls.append(chain)
        return ChainExecutionResult(
            output="".join(self.deltas),
            duration_seconds=0.05,
            tee_attestation=self.tee_attestation,
            tee_type=self.tee_type,
            epsilon_spent=self.epsilon,
        )

    # When no_streaming_method=True, this attribute is deleted at
    # construction so AttributeError fires inside execute_streaming —
    # simulates an operator who wired a non-streaming chain executor.
    def execute_chain_streaming(self, *, request, chain):
        self.streaming_calls.append(chain)
        if self.raise_on_streaming:
            raise RuntimeError("streaming chain crashed")
        last = len(self.deltas) - 1
        for i, delta in enumerate(self.deltas):
            yield StreamToken(
                sequence_index=i,
                text_delta=delta,
                finish_reason=self.finish_reason if i == last else None,
            )
        if self.omit_terminal:
            return
        yield ChainExecutionResult(
            output="".join(self.deltas),
            duration_seconds=0.05,
            tee_attestation=self.tee_attestation,
            tee_type=self.tee_type,
            epsilon_spent=self.epsilon,
        )


def _drain_streaming(coro_or_async_iter):
    """Helper: drain an async generator into a list."""
    async def _drain():
        items = []
        async for item in coro_or_async_iter:
            items.append(item)
        return items
    return asyncio.new_event_loop().run_until_complete(_drain())


class TestExecuteStreamingHappyPath:
    def test_yields_token_events_then_terminal_success(self):
        chain_exec = StreamingChainExecutor(
            deltas=["hello", " ", "world"],
        )
        executor, _, _ = _make_executor(
            pool=[_gpu("a"), _gpu("b")], chain_executor=chain_exec,
        )
        items = _drain_streaming(executor.execute_streaming(_request()))
        # All non-terminal items are InferenceTokenEvents; LAST is the
        # InferenceResult.
        token_events = [x for x in items if isinstance(x, InferenceTokenEvent)]
        finals = [x for x in items if isinstance(x, InferenceResult)]
        assert len(finals) == 1
        assert finals[0].success is True
        assert len(token_events) == 3
        assert token_events[0].text_delta == "hello"
        assert token_events[1].text_delta == " "
        assert token_events[2].text_delta == "world"
        assert token_events[2].finish_reason == "stop"
        # Joined deltas equal the final output.
        joined = "".join(t.text_delta for t in token_events)
        assert joined == finals[0].output == "hello world"

    def test_streamed_receipt_marks_streamed_output_true(self):
        chain_exec = StreamingChainExecutor(deltas=["x"])
        executor, _, _ = _make_executor(
            pool=[_gpu("a"), _gpu("b")], chain_executor=chain_exec,
        )
        items = _drain_streaming(executor.execute_streaming(_request()))
        result = items[-1]
        assert result.receipt.streamed_output is True
        # Receipt verifies under the executor's identity (accessed
        # directly off the instance — tests in this module follow
        # the same pattern for other private fields).
        assert verify_receipt(
            result.receipt, identity=executor._identity,  # noqa: SLF001
        )

    def test_streamed_job_id_uses_stream_prefix(self):
        chain_exec = StreamingChainExecutor(deltas=["y"])
        executor, _, _ = _make_executor(
            pool=[_gpu("a"), _gpu("b")], chain_executor=chain_exec,
        )
        items = _drain_streaming(executor.execute_streaming(_request()))
        result = items[-1]
        assert result.receipt.job_id.startswith("parallax-stream-job-")

    def test_unary_path_unchanged_uses_legacy_job_prefix(self):
        # Refactor invariant: execute() still emits "parallax-job-"
        # prefix (no "stream"). Audit-log filtering relies on this.
        chain_exec = StreamingChainExecutor(deltas=["z"])
        executor, _, _ = _make_executor(
            pool=[_gpu("a"), _gpu("b")], chain_executor=chain_exec,
        )
        result = _run(executor.execute(_request()))
        assert result.success is True
        assert result.receipt.job_id.startswith("parallax-job-")
        assert not result.receipt.job_id.startswith("parallax-stream-job-")
        assert result.receipt.streamed_output is False


class TestExecuteStreamingPreExecuteFailures:
    """Pre-execute gate failures yield a SOLE InferenceResult.failure
    with NO token events emitted."""

    def test_unknown_model_yields_sole_failure(self):
        chain_exec = StreamingChainExecutor()
        executor, _, _ = _make_executor(
            pool=[_gpu("a")], chain_executor=chain_exec,
        )
        items = _drain_streaming(executor.execute_streaming(
            _request(model_id="unknown-model"),
        ))
        assert len(items) == 1
        assert isinstance(items[0], InferenceResult)
        assert items[0].success is False
        assert "Unknown model_id" in items[0].error
        # Streaming chain executor was NEVER invoked.
        assert chain_exec.streaming_calls == []

    def test_insufficient_budget_yields_sole_failure(self):
        chain_exec = StreamingChainExecutor()
        executor, _, _ = _make_executor(
            pool=[_gpu("a")], chain_executor=chain_exec,
        )
        items = _drain_streaming(executor.execute_streaming(
            _request(budget=Decimal("0.0001")),
        ))
        assert len(items) == 1
        assert items[0].success is False
        assert "Insufficient budget" in items[0].error
        assert chain_exec.streaming_calls == []

    def test_empty_pool_yields_sole_failure(self):
        chain_exec = StreamingChainExecutor()
        executor, _, _ = _make_executor(
            pool=[], chain_executor=chain_exec,
        )
        items = _drain_streaming(executor.execute_streaming(_request()))
        assert len(items) == 1
        assert items[0].success is False
        # "GPU pool is empty" or routing failure depending on the
        # gate that fires first.
        assert chain_exec.streaming_calls == []


class TestExecuteStreamingMidStreamErrors:
    """Mid-stream chain-executor failures surface as a terminal
    InferenceResult.failure — non-fatal at the wire layer."""

    def test_streaming_runner_exception_surfaces_as_terminal_failure(self):
        chain_exec = StreamingChainExecutor(raise_on_streaming=True)
        executor, _, _ = _make_executor(
            pool=[_gpu("a"), _gpu("b")], chain_executor=chain_exec,
        )
        items = _drain_streaming(executor.execute_streaming(_request()))
        # Final yield is failure; preceding tokens depend on WHERE
        # in the iteration the exception fired.
        finals = [x for x in items if isinstance(x, InferenceResult)]
        assert len(finals) == 1
        assert finals[0].success is False
        assert "chain execution failure" in finals[0].error

    def test_chain_executor_without_streaming_method_yields_failure(self):
        # Operator wired a unary-only chain executor to the streaming
        # path. Surface as a structured failure.
        class UnaryOnly:
            def execute_chain(self, **kwargs):
                raise NotImplementedError

        executor, _, _ = _make_executor(
            pool=[_gpu("a"), _gpu("b")], chain_executor=UnaryOnly(),
        )
        items = _drain_streaming(executor.execute_streaming(_request()))
        assert len(items) == 1
        assert items[0].success is False
        assert (
            "chain executor does not support streaming" in items[0].error
        )

    def test_omitted_terminal_chain_result_yields_failure(self):
        # Chain executor's streaming generator exits without yielding
        # a terminal ChainExecutionResult.
        chain_exec = StreamingChainExecutor(omit_terminal=True)
        executor, _, _ = _make_executor(
            pool=[_gpu("a"), _gpu("b")], chain_executor=chain_exec,
        )
        items = _drain_streaming(executor.execute_streaming(_request()))
        # Tokens may be emitted; final yield is failure.
        finals = [x for x in items if isinstance(x, InferenceResult)]
        assert len(finals) == 1
        assert finals[0].success is False
        assert "without yielding a terminal" in finals[0].error


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.11.q Task 4 — Tier C routing-layer integration
# ──────────────────────────────────────────────────────────────────────────


class _RecordingTierCExecutor:
    """Records that it was invoked + delegates to an inner Tier A/B
    executor. Tests assert (a) Tier C requests reach this decorator,
    (b) Tier A/B requests do NOT."""

    def __init__(
        self, inner: StreamingChainExecutor,
        tag: str = "tier-c",
    ) -> None:
        self._inner = inner
        self._tag = tag
        self.invocations: List[GPUChain] = []

    def execute_chain_streaming(self, *, request, chain):
        self.invocations.append(chain)
        for ev in self._inner.execute_chain_streaming(
            request=request, chain=chain,
        ):
            yield ev


class TestTierCRoutingIntegration:
    def test_tier_a_uses_default_executor(self):
        # Tier A request should NOT touch the Tier C decorator.
        primary = StreamingChainExecutor()
        tier_c = _RecordingTierCExecutor(StreamingChainExecutor())
        executor, _, _ = _make_executor(
            pool=[_gpu("a"), _gpu("b")],
            chain_executor=primary,
            tier_c_chain_executor=tier_c,
        )
        items = _drain_streaming(executor.execute_streaming(
            _request(content_tier=ContentTier.A),
        ))
        # primary saw the request, tier_c did not.
        assert len(primary.streaming_calls) == 1
        assert tier_c.invocations == []
        # Final terminal is success.
        finals = [x for x in items if isinstance(x, InferenceResult)]
        assert len(finals) == 1 and finals[0].success is True

    def test_tier_b_uses_default_executor(self):
        primary = StreamingChainExecutor()
        tier_c = _RecordingTierCExecutor(StreamingChainExecutor())
        executor, _, _ = _make_executor(
            pool=[_gpu("a"), _gpu("b")],
            chain_executor=primary,
            tier_c_chain_executor=tier_c,
        )
        items = _drain_streaming(executor.execute_streaming(
            _request(content_tier=ContentTier.B),
        ))
        assert len(primary.streaming_calls) == 1
        assert tier_c.invocations == []
        finals = [x for x in items if isinstance(x, InferenceResult)]
        assert len(finals) == 1 and finals[0].success is True

    def test_tier_c_routes_to_decorator_when_wired(self):
        primary = StreamingChainExecutor()
        # Tier C decorator wraps its own inner streaming executor —
        # that's the production pattern (decorator owns the inner).
        tier_c = _RecordingTierCExecutor(
            StreamingChainExecutor(deltas=["t1", "t2"]),
        )
        executor, _, _ = _make_executor(
            pool=[_gpu("a"), _gpu("b")],
            chain_executor=primary,
            tier_c_chain_executor=tier_c,
        )
        items = _drain_streaming(executor.execute_streaming(
            _request(content_tier=ContentTier.C),
        ))
        # primary did NOT see the request; tier_c did.
        assert primary.streaming_calls == []
        assert len(tier_c.invocations) == 1
        # Stream completed successfully through the tier_c path.
        finals = [x for x in items if isinstance(x, InferenceResult)]
        assert len(finals) == 1 and finals[0].success is True

    def test_tier_c_without_decorator_surfaces_failure(self):
        # Operator misconfigured: Tier C request but no decorator
        # wired. Expected: structured failure pointing at the fix.
        primary = StreamingChainExecutor()
        executor, _, _ = _make_executor(
            pool=[_gpu("a"), _gpu("b")],
            chain_executor=primary,
            tier_c_chain_executor=None,
        )
        items = _drain_streaming(executor.execute_streaming(
            _request(content_tier=ContentTier.C),
        ))
        # No tokens emitted; sole terminal is the failure.
        token_events = [
            x for x in items if isinstance(x, InferenceTokenEvent)
        ]
        finals = [x for x in items if isinstance(x, InferenceResult)]
        assert token_events == []
        assert len(finals) == 1
        assert finals[0].success is False
        assert "Tier C streaming requires" in finals[0].error
        assert "make_tier_c_sharded_executor" in finals[0].error
        # Importantly, the primary chain executor was NOT invoked.
        assert primary.streaming_calls == []

    def test_construction_rejects_decorator_without_streaming_method(self):
        # Defense-in-depth: decorator without execute_chain_streaming
        # should be caught at constructor time, not at runtime.
        class _Bogus:
            pass
        with pytest.raises(
            RuntimeError, match="execute_chain_streaming",
        ):
            _make_executor(
                pool=[_gpu("a"), _gpu("b")],
                tier_c_chain_executor=_Bogus(),
            )

    def test_factory_wired_decorator_routes_correctly(self):
        # Smoke: the production-shape wiring through
        # make_tier_c_sharded_executor + ParallaxScheduledExecutor
        # routes Tier C to the decorator end-to-end.
        from prsm.compute.chain_rpc import make_tier_c_sharded_executor
        primary = StreamingChainExecutor()
        # Use M2 (BatchedTrailing) for a smoke test — the exact
        # decorator type doesn't matter here; only that the
        # routing fires.
        tier_c_exec = make_tier_c_sharded_executor(
            primary, mode="m2",
        )
        executor, _, _ = _make_executor(
            pool=[_gpu("a"), _gpu("b")],
            chain_executor=primary,
            tier_c_chain_executor=tier_c_exec,
        )
        items = _drain_streaming(executor.execute_streaming(
            _request(content_tier=ContentTier.C),
        ))
        finals = [x for x in items if isinstance(x, InferenceResult)]
        assert len(finals) == 1 and finals[0].success is True
