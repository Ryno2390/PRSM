"""
Unit tests — Phase 3.x.1 Task 4 — TensorParallelInferenceExecutor.

Acceptance per design plan §4 Task 4: end-to-end mock inference returns
correct shape + (unsigned) receipt + DP noise applied per privacy tier.

These tests build real ShardedModel objects with numpy float64 tensor
data and run the actual TensorParallelExecutor — no crypto/numerics
mocks. Per the project's testing rules: do not deprecate to simpler
testing; work through the problem with the real primitives.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal

import numpy as np
import pytest

from prsm.compute.inference import (
    InferenceRequest,
    InferenceResult,
    TensorParallelInferenceExecutor,
    UnsupportedModelError,
)
from prsm.compute.inference.models import ContentTier
from prsm.compute.model_sharding.models import ModelShard, ShardedModel
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.compute.tee.runtime import SoftwareTEERuntime, TEERuntime


# ──────────────────────────────────────────────────────────────────────────
# Test helpers — real ShardedModel with deterministic numpy tensors
# ──────────────────────────────────────────────────────────────────────────


def _make_shard(model_id: str, index: int, total: int, rows: int, cols: int) -> ModelShard:
    """Build a real ModelShard with a deterministic 2D tensor.

    Tensor is `rows × cols` filled with `index/(rows*cols)` increments;
    different shards produce different but deterministic outputs so
    all-reduce is observable.
    """
    rng = np.random.default_rng(seed=1000 + index)
    tensor = rng.standard_normal(size=(rows, cols))
    data = tensor.tobytes()
    return ModelShard(
        shard_id=f"{model_id}-shard-{index}",
        model_id=model_id,
        shard_index=index,
        total_shards=total,
        tensor_data=data,
        tensor_shape=(rows, cols),
        layer_range=(0, 0),
        size_bytes=len(data),
        checksum=hashlib.sha256(data).hexdigest(),
    )


def _make_model(model_id: str, num_shards: int = 3, rows: int = 8, cols: int = 16) -> ShardedModel:
    shards = [_make_shard(model_id, i, num_shards, rows, cols) for i in range(num_shards)]
    return ShardedModel(
        model_id=model_id,
        model_name=f"test-{model_id}",
        total_shards=num_shards,
        shards=shards,
    )


@pytest.fixture
def model():
    return _make_model("test-llama")


@pytest.fixture
def registry(model):
    return {model.model_id: model}


@pytest.fixture
def executor(registry):
    return TensorParallelInferenceExecutor(model_registry=registry)


def _make_request(
    *,
    model_id: str = "test-llama",
    prompt: str = "the quick brown fox",
    budget_ftns: Decimal = Decimal("1.0"),
    privacy_tier: PrivacyLevel = PrivacyLevel.STANDARD,
    content_tier: ContentTier = ContentTier.A,
) -> InferenceRequest:
    return InferenceRequest(
        prompt=prompt,
        model_id=model_id,
        budget_ftns=budget_ftns,
        privacy_tier=privacy_tier,
        content_tier=content_tier,
    )


# ──────────────────────────────────────────────────────────────────────────
# Construction + registry
# ──────────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_supported_models_lists_registry(self, registry):
        ex = TensorParallelInferenceExecutor(model_registry=registry)
        assert ex.supported_models() == ["test-llama"]

    def test_register_model_adds_to_registry(self, executor):
        m2 = _make_model("test-mistral", num_shards=2)
        executor.register_model(m2)
        assert "test-mistral" in executor.supported_models()

    def test_register_model_replaces_existing(self, executor, registry):
        m_new = _make_model("test-llama", num_shards=5)
        executor.register_model(m_new)
        assert executor._models["test-llama"].total_shards == 5

    def test_default_tee_runtime_is_software(self, registry):
        ex = TensorParallelInferenceExecutor(model_registry=registry)
        assert ex.tee_runtime.tee_type == TEEType.SOFTWARE

    def test_custom_tee_runtime_used(self, registry):
        class FakeHardwareTEE(TEERuntime):
            @property
            def name(self): return "sgx-fake"
            @property
            def tee_type(self): return TEEType.SGX
            @property
            def available(self): return True
            def load(self, wasm_bytes): return None
            def execute(self, module, input_data, resource_limits): raise NotImplementedError

        ex = TensorParallelInferenceExecutor(
            model_registry=registry, tee_runtime=FakeHardwareTEE()
        )
        assert ex.tee_runtime.tee_type == TEEType.SGX


# ──────────────────────────────────────────────────────────────────────────
# Cost estimation
# ──────────────────────────────────────────────────────────────────────────


class TestEstimateCost:
    @pytest.mark.asyncio
    async def test_cost_scales_with_total_shards(self, executor):
        # 3 shards × 0.05 FTNS × 1.10 (standard overhead) = 0.165
        req = _make_request(privacy_tier=PrivacyLevel.STANDARD)
        cost = await executor.estimate_cost(req)
        assert cost == Decimal("0.05") * Decimal(3) * Decimal("1.10")

    @pytest.mark.asyncio
    async def test_overhead_increases_with_privacy_tier(self, executor):
        none_cost = await executor.estimate_cost(_make_request(privacy_tier=PrivacyLevel.NONE))
        std_cost = await executor.estimate_cost(_make_request(privacy_tier=PrivacyLevel.STANDARD))
        high_cost = await executor.estimate_cost(_make_request(privacy_tier=PrivacyLevel.HIGH))
        max_cost = await executor.estimate_cost(_make_request(privacy_tier=PrivacyLevel.MAXIMUM))
        assert none_cost < std_cost < high_cost < max_cost

    @pytest.mark.asyncio
    async def test_unknown_model_raises(self, executor):
        with pytest.raises(UnsupportedModelError, match="Unknown model_id"):
            await executor.estimate_cost(_make_request(model_id="does-not-exist"))


# ──────────────────────────────────────────────────────────────────────────
# Happy-path execution
# ──────────────────────────────────────────────────────────────────────────


class TestExecuteHappyPath:
    @pytest.mark.asyncio
    async def test_tier_a_standard_privacy_returns_signed_shape(self, executor):
        req = _make_request(privacy_tier=PrivacyLevel.STANDARD, content_tier=ContentTier.A)
        result = await executor.execute(req)
        assert result.success
        assert isinstance(result, InferenceResult)
        assert result.output  # non-empty
        assert result.receipt is not None

    @pytest.mark.asyncio
    async def test_receipt_fields_populated(self, executor):
        req = _make_request()
        result = await executor.execute(req)
        receipt = result.receipt
        assert receipt.job_id.startswith("infer-job-")
        assert receipt.request_id == req.request_id
        assert receipt.model_id == "test-llama"
        assert receipt.tee_type == TEEType.SOFTWARE
        # Tee attestation is the 64-byte sha512 marker
        assert len(receipt.tee_attestation) == 64
        # Output hash matches sha256 of output text
        assert receipt.output_hash == hashlib.sha256(result.output.encode("utf-8")).digest()
        # Settler signature is empty (signed at API layer Task 5)
        assert receipt.settler_signature == b"\x00" * 64
        assert receipt.settler_node_id == ""
        # Cost matches estimate
        expected_cost = await executor.estimate_cost(req)
        assert receipt.cost_ftns == expected_cost
        # Duration is positive
        assert receipt.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_receipt_carries_request_tiers(self, executor):
        req = _make_request(
            privacy_tier=PrivacyLevel.HIGH,
            content_tier=ContentTier.A,
        )
        result = await executor.execute(req)
        assert result.receipt.privacy_tier == PrivacyLevel.HIGH
        assert result.receipt.content_tier == ContentTier.A

    @pytest.mark.asyncio
    async def test_attestation_is_job_bound(self, executor):
        # Two separate executions produce distinct attestation blobs
        # because they have distinct job_ids.
        r1 = await executor.execute(_make_request())
        r2 = await executor.execute(_make_request())
        assert r1.receipt.tee_attestation != r2.receipt.tee_attestation
        assert r1.receipt.job_id != r2.receipt.job_id


# ──────────────────────────────────────────────────────────────────────────
# DP noise injection per privacy tier
# ──────────────────────────────────────────────────────────────────────────


class TestDPNoise:
    @pytest.mark.asyncio
    async def test_privacy_none_records_zero_epsilon(self, executor):
        result = await executor.execute(_make_request(privacy_tier=PrivacyLevel.NONE))
        assert result.success
        assert result.receipt.epsilon_spent == 0.0

    @pytest.mark.asyncio
    async def test_privacy_standard_records_eps_8(self, executor):
        result = await executor.execute(_make_request(privacy_tier=PrivacyLevel.STANDARD))
        assert result.receipt.epsilon_spent == 8.0

    @pytest.mark.asyncio
    async def test_privacy_high_records_eps_4(self, executor):
        result = await executor.execute(_make_request(privacy_tier=PrivacyLevel.HIGH))
        assert result.receipt.epsilon_spent == 4.0

    @pytest.mark.asyncio
    async def test_privacy_maximum_records_eps_1(self, executor):
        result = await executor.execute(_make_request(privacy_tier=PrivacyLevel.MAXIMUM))
        assert result.receipt.epsilon_spent == 1.0

    @pytest.mark.asyncio
    async def test_dp_noise_is_actually_applied(self, executor):
        # Same prompt + model_id + content_tier under NONE vs MAXIMUM
        # privacy must produce different output strings, since DP noise
        # perturbs the aggregated tensor.
        np.random.seed(42)
        r_none = await executor.execute(_make_request(privacy_tier=PrivacyLevel.NONE))

        np.random.seed(42)
        r_max = await executor.execute(
            _make_request(privacy_tier=PrivacyLevel.MAXIMUM, prompt="the quick brown fox")
        )
        # The deterministic part of output (model_id, prompt_len, output_dim)
        # is identical across runs; the `sample=[...]` tail differs because
        # MAXIMUM applies noise.
        assert r_none.output != r_max.output


# ──────────────────────────────────────────────────────────────────────────
# Failure paths
# ──────────────────────────────────────────────────────────────────────────


class TestFailurePaths:
    @pytest.mark.asyncio
    async def test_unknown_model_returns_failure(self, executor):
        result = await executor.execute(_make_request(model_id="not-a-model"))
        assert not result.success
        assert "Unknown model_id" in (result.error or "")
        assert result.receipt is None

    @pytest.mark.asyncio
    async def test_insufficient_budget_returns_failure(self, executor):
        # 3 shards × 0.05 × 1.10 = 0.165; budget=0.10 < cost.
        result = await executor.execute(_make_request(budget_ftns=Decimal("0.10")))
        assert not result.success
        assert "Insufficient budget" in (result.error or "")

    @pytest.mark.asyncio
    async def test_software_tee_blocked_when_disallowed_for_privacy(self, registry):
        ex = TensorParallelInferenceExecutor(
            model_registry=registry,
            allow_software_tee_for_privacy=False,
        )
        result = await ex.execute(_make_request(privacy_tier=PrivacyLevel.STANDARD))
        assert not result.success
        assert "hardware TEE" in (result.error or "")

    @pytest.mark.asyncio
    async def test_software_tee_disallowed_still_passes_for_privacy_none(self, registry):
        # privacy_tier=NONE doesn't need TEE — must succeed even when
        # software-TEE is gated for privacy levels.
        ex = TensorParallelInferenceExecutor(
            model_registry=registry,
            allow_software_tee_for_privacy=False,
        )
        result = await ex.execute(_make_request(privacy_tier=PrivacyLevel.NONE))
        assert result.success

    @pytest.mark.asyncio
    async def test_empty_model_returns_failure(self, registry):
        # Construct a model with zero shards → encoding raises → graceful failure
        empty = ShardedModel(
            model_id="empty", model_name="empty", total_shards=0, shards=[]
        )
        ex = TensorParallelInferenceExecutor(model_registry={"empty": empty})
        result = await ex.execute(_make_request(model_id="empty", privacy_tier=PrivacyLevel.NONE))
        assert not result.success
        assert "Prompt encoding failed" in (result.error or "")


# ──────────────────────────────────────────────────────────────────────────
# Determinism + numerics integration
# ──────────────────────────────────────────────────────────────────────────


class TestNumerics:
    @pytest.mark.asyncio
    async def test_privacy_none_is_deterministic(self, executor):
        # privacy_tier=NONE skips DP noise → same prompt should produce
        # identical output text for the deterministic portion.
        r1 = await executor.execute(_make_request(privacy_tier=PrivacyLevel.NONE))
        r2 = await executor.execute(_make_request(privacy_tier=PrivacyLevel.NONE))
        # Job IDs differ so receipts differ; the OUTPUT text must match.
        assert r1.output == r2.output

    @pytest.mark.asyncio
    async def test_different_prompts_produce_different_output(self, executor):
        r1 = await executor.execute(
            _make_request(privacy_tier=PrivacyLevel.NONE, prompt="prompt A")
        )
        r2 = await executor.execute(
            _make_request(privacy_tier=PrivacyLevel.NONE, prompt="prompt B different length")
        )
        assert r1.output != r2.output

    @pytest.mark.asyncio
    async def test_output_format_includes_diagnostics(self, executor):
        result = await executor.execute(
            _make_request(prompt="hello", privacy_tier=PrivacyLevel.STANDARD,
                          content_tier=ContentTier.A)
        )
        assert "test-llama" in result.output
        assert "prompt_len=5" in result.output
        assert "privacy=standard" in result.output
        assert "content_tier=A" in result.output
        assert "output_dim=" in result.output
        assert "sample=" in result.output


# ──────────────────────────────────────────────────────────────────────────
# Public-API contract: drop-in for MockInferenceExecutor
# ──────────────────────────────────────────────────────────────────────────


class TestInterfaceContract:
    @pytest.mark.asyncio
    async def test_subclasses_inferenceexecutor(self, executor):
        from prsm.compute.inference.executor import InferenceExecutor
        assert isinstance(executor, InferenceExecutor)

    @pytest.mark.asyncio
    async def test_uses_request_id_from_request(self, executor):
        req = _make_request()
        result = await executor.execute(req)
        assert result.request_id == req.request_id

    @pytest.mark.asyncio
    async def test_signed_receipt_roundtrips_through_to_dict(self, executor):
        result = await executor.execute(_make_request())
        d = result.receipt.to_dict()
        # All the canonical fields are present + serializable
        for key in [
            "job_id", "request_id", "model_id", "content_tier",
            "privacy_tier", "epsilon_spent", "tee_type",
            "tee_attestation", "output_hash", "duration_seconds",
            "cost_ftns", "settler_signature", "settler_node_id",
        ]:
            assert key in d
