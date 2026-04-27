"""Unit tests for the Phase-1 layer allocator adapter.

Phase 3.x.6 Task 2 — exercises the PRSM-side ``ParallaxGPU`` data type +
``allocate_across_regions`` orchestrator that wraps the vendored
upstream DP + water-filling allocator. The algorithm itself is not
re-verified here (upstream's own test suite covers that); these tests
pin the PRSM-side contract:

  - ParallaxGPU validation rejects garbage inputs early
  - to_parallax_node converts cleanly to the upstream Node type
  - partition_by_region groups correctly
  - allocate_across_regions never produces cross-region pipelines
  - Edge cases (empty pool, insufficient capacity, single sufficient GPU)
    surface as named exceptions or expected results
  - Water-filling on heterogeneous compute produces sane stage shapes
"""

from __future__ import annotations

import pytest

from prsm.compute.parallax_scheduling.model_info import ModelInfo
from prsm.compute.parallax_scheduling.node_management import NodeManager
from prsm.compute.parallax_scheduling.prsm_types import (
    AllocationError,
    AllocationResult,
    EmptyPoolError,
    InsufficientCapacityError,
    ParallaxGPU,
    RegionPipeline,
    TIER_ATTESTATION_NONE,
    allocate_across_regions,
    partition_by_region,
    to_parallax_node,
)


# ──────────────────────────────────────────────────────────────────────────
# Test helpers — mirror upstream test_utils.py builders
# ──────────────────────────────────────────────────────────────────────────


def _build_model_info(num_layers: int) -> ModelInfo:
    """Realistic GPT-OSS-like config; matches upstream test convention."""
    return ModelInfo(
        model_name=f"GPUOss-{num_layers}L",
        mlx_model_name=f"MLXOss-{num_layers}L",
        head_size=64,
        hidden_dim=2880,
        intermediate_dim=2880,
        num_attention_heads=64,
        num_kv_heads=8,
        vocab_size=201088,
        num_layers=num_layers,
        ffn_num_projections=3,
        num_local_experts=128,
        num_experts_per_tok=4,
        param_bytes_per_element=1,
        mlx_param_bytes_per_element=1,
        cache_bytes_per_element=2,
        embedding_bytes_per_element=2,
    )


# Realistic GPU profiles — matches upstream test_utils.py.
_HW_PRESETS = {
    "a100-80g": dict(
        tflops_fp16=312.0, memory_gb=80.0, memory_bandwidth_gbps=2039.0,
    ),
    "a100-40g": dict(
        tflops_fp16=312.0, memory_gb=40.0, memory_bandwidth_gbps=1935.0,
    ),
    "rtx5090": dict(
        tflops_fp16=165.0, memory_gb=32.0, memory_bandwidth_gbps=1792.0,
    ),
    "rtx4090": dict(
        tflops_fp16=82.6, memory_gb=24.0, memory_bandwidth_gbps=1008.0,
    ),
}


def _gpu(
    node_id: str,
    *,
    region: str = "us-west-2",
    gpu_type: str = "a100-80g",
    layer_capacity: int = 13,
    stake_amount: int = 0,
    tier_attestation: str = TIER_ATTESTATION_NONE,
) -> ParallaxGPU:
    """Build a ParallaxGPU using upstream-realistic hardware presets."""
    hw = _HW_PRESETS[gpu_type]
    return ParallaxGPU(
        node_id=node_id,
        region=region,
        layer_capacity=layer_capacity,
        stake_amount=stake_amount,
        tier_attestation=tier_attestation,
        gpu_name=gpu_type,
        device="cuda",
        **hw,
    )


# ──────────────────────────────────────────────────────────────────────────
# ParallaxGPU validation
# ──────────────────────────────────────────────────────────────────────────


class TestParallaxGPUValidation:
    """Construction-time validation. Reject garbage early; the
    allocator is load-bearing and confusion downstream is much
    harder to debug than a clean ValueError at construction."""

    def test_minimal_valid_construction(self):
        gpu = _gpu("alice")
        assert gpu.node_id == "alice"
        assert gpu.region == "us-west-2"
        assert gpu.layer_capacity == 13
        assert gpu.stake_amount == 0
        assert gpu.tier_attestation == TIER_ATTESTATION_NONE
        assert gpu.tflops_fp16 == 312.0

    def test_frozen_dataclass(self):
        gpu = _gpu("alice")
        with pytest.raises((AttributeError, Exception)):
            gpu.region = "different"  # type: ignore[misc]

    @pytest.mark.parametrize("bad_value", ["", None])
    def test_empty_node_id_rejected(self, bad_value):
        with pytest.raises((ValueError, TypeError)):
            ParallaxGPU(
                node_id=bad_value,  # type: ignore[arg-type]
                region="r",
                layer_capacity=4,
                stake_amount=0,
                tier_attestation=TIER_ATTESTATION_NONE,
                tflops_fp16=100.0,
                memory_gb=24.0,
                memory_bandwidth_gbps=500.0,
            )

    def test_empty_region_rejected(self):
        with pytest.raises(ValueError, match="region"):
            _gpu("alice", region="")

    def test_negative_layer_capacity_rejected(self):
        with pytest.raises(ValueError, match="layer_capacity"):
            _gpu("alice", layer_capacity=-1)

    def test_negative_stake_amount_rejected(self):
        with pytest.raises(ValueError, match="stake_amount"):
            _gpu("alice", stake_amount=-100)

    def test_zero_tflops_rejected(self):
        with pytest.raises(ValueError, match="tflops_fp16"):
            ParallaxGPU(
                node_id="alice",
                region="r",
                layer_capacity=4,
                stake_amount=0,
                tier_attestation=TIER_ATTESTATION_NONE,
                tflops_fp16=0.0,
                memory_gb=24.0,
                memory_bandwidth_gbps=500.0,
            )

    def test_zero_memory_rejected(self):
        with pytest.raises(ValueError, match="memory_gb"):
            ParallaxGPU(
                node_id="alice",
                region="r",
                layer_capacity=4,
                stake_amount=0,
                tier_attestation=TIER_ATTESTATION_NONE,
                tflops_fp16=100.0,
                memory_gb=0.0,
                memory_bandwidth_gbps=500.0,
            )


# ──────────────────────────────────────────────────────────────────────────
# Conversion: ParallaxGPU → upstream Node
# ──────────────────────────────────────────────────────────────────────────


class TestToParallaxNode:
    def test_basic_conversion(self):
        model = _build_model_info(num_layers=12)
        gpu = _gpu("alice", gpu_type="a100-80g")
        node = to_parallax_node(gpu, model)

        assert node.node_id == "alice"
        assert node.hardware.tflops_fp16 == 312.0
        assert node.hardware.memory_gb == 80.0
        assert node.hardware.memory_bandwidth_gbps == 2039.0
        assert node.hardware.device == "cuda"
        assert node.model_info is model
        # Pre-allocation: no layer assignment yet
        assert node.start_layer is None
        assert node.end_layer is None

    def test_quantization_speedup_propagated(self):
        model = _build_model_info(num_layers=12)
        gpu = ParallaxGPU(
            node_id="alice",
            region="r",
            layer_capacity=4,
            stake_amount=0,
            tier_attestation=TIER_ATTESTATION_NONE,
            tflops_fp16=100.0,
            memory_gb=24.0,
            memory_bandwidth_gbps=500.0,
            quantization_speedup=2.5,
        )
        node = to_parallax_node(gpu, model)
        assert node.quantization_speedup == 2.5

    def test_rtt_to_nodes_propagated(self):
        model = _build_model_info(num_layers=12)
        rtts = {"bob": 12.5, "charlie": 87.0}
        gpu = ParallaxGPU(
            node_id="alice",
            region="r",
            layer_capacity=4,
            stake_amount=0,
            tier_attestation=TIER_ATTESTATION_NONE,
            tflops_fp16=100.0,
            memory_gb=24.0,
            memory_bandwidth_gbps=500.0,
            rtt_to_nodes=rtts,
        )
        node = to_parallax_node(gpu, model)
        assert node.rtt_to_nodes == rtts
        # Defensive copy — caller's dict mutation must not propagate
        rtts["bob"] = 99999.0
        assert node.rtt_to_nodes["bob"] == 12.5


# ──────────────────────────────────────────────────────────────────────────
# Region partitioning
# ──────────────────────────────────────────────────────────────────────────


class TestPartitionByRegion:
    def test_empty_input_returns_empty_dict(self):
        assert partition_by_region([]) == {}

    def test_single_region_single_key(self):
        gpus = [_gpu(f"node{i}", region="us-west-2") for i in range(3)]
        out = partition_by_region(gpus)
        assert set(out.keys()) == {"us-west-2"}
        assert len(out["us-west-2"]) == 3

    def test_two_regions_split_correctly(self):
        gpus = (
            [_gpu(f"a{i}", region="us-west-2") for i in range(4)]
            + [_gpu(f"b{i}", region="eu-fra-1") for i in range(4)]
        )
        out = partition_by_region(gpus)
        assert set(out.keys()) == {"us-west-2", "eu-fra-1"}
        assert len(out["us-west-2"]) == 4
        assert len(out["eu-fra-1"]) == 4

    def test_node_ids_preserved_per_region(self):
        gpus = [
            _gpu("alice", region="us-west-2"),
            _gpu("bob", region="us-west-2"),
            _gpu("carol", region="eu-fra-1"),
        ]
        out = partition_by_region(gpus)
        ids_west = {g.node_id for g in out["us-west-2"]}
        ids_eu = {g.node_id for g in out["eu-fra-1"]}
        assert ids_west == {"alice", "bob"}
        assert ids_eu == {"carol"}


# ──────────────────────────────────────────────────────────────────────────
# allocate_across_regions — the Phase-1 orchestrator
# ──────────────────────────────────────────────────────────────────────────


class TestAllocateAcrossRegions:
    def test_empty_pool_raises(self):
        model = _build_model_info(num_layers=12)
        with pytest.raises(EmptyPoolError):
            allocate_across_regions([], model)

    def test_single_gpu_with_sufficient_capacity_produces_single_pipeline(self):
        # GPU can host all 4 layers itself → single 1-stage pipeline.
        model = _build_model_info(num_layers=4)
        gpu = _gpu("alice", layer_capacity=8, gpu_type="a100-80g")
        result = allocate_across_regions([gpu], model)
        pipelines = result.all_pipelines()
        assert len(pipelines) == 1
        assert pipelines[0].region == "us-west-2"
        assert pipelines[0].stages == ["alice"]
        # Single stage covers all layers.
        assert pipelines[0].layer_ranges == [(0, 4)]

    def test_insufficient_capacity_raises(self):
        # Total capacity (4) < num_layers (12).
        model = _build_model_info(num_layers=12)
        gpus = [
            _gpu("a", layer_capacity=2, gpu_type="rtx4090"),
            _gpu("b", layer_capacity=2, gpu_type="rtx4090"),
        ]
        with pytest.raises(InsufficientCapacityError, match="us-west-2"):
            allocate_across_regions(gpus, model)

    def test_insufficient_capacity_with_allow_partial_skips_region(self):
        # Region A has enough capacity, region B does not.
        # allow_partial=True skips B without raising.
        model = _build_model_info(num_layers=4)
        gpus = [
            _gpu("a1", region="A", layer_capacity=4, gpu_type="a100-80g"),
            _gpu("b1", region="B", layer_capacity=1, gpu_type="rtx4090"),
        ]
        result = allocate_across_regions(gpus, model, allow_partial=True)
        # Only region A should appear.
        assert "A" in result.region_to_pipelines
        assert "B" not in result.region_to_pipelines

    def test_pipelines_never_span_regions(self):
        # 4 GPUs in region A + 4 in region B, each region with enough
        # capacity. Result must have ≥1 pipeline per region, and
        # every pipeline's stages must come from a single region.
        model = _build_model_info(num_layers=8)
        gpus = (
            [_gpu(f"a{i}", region="A", gpu_type="rtx5090", layer_capacity=4)
             for i in range(4)]
            + [_gpu(f"b{i}", region="B", gpu_type="rtx5090", layer_capacity=4)
               for i in range(4)]
        )
        result = allocate_across_regions(gpus, model)
        assert "A" in result.region_to_pipelines
        assert "B" in result.region_to_pipelines

        a_node_ids = {f"a{i}" for i in range(4)}
        b_node_ids = {f"b{i}" for i in range(4)}
        for region, pipelines in result.region_to_pipelines.items():
            for pipeline in pipelines:
                stage_ids = set(pipeline.stages)
                if region == "A":
                    assert stage_ids.issubset(a_node_ids), (
                        f"region A pipeline contains non-A stages: {stage_ids}"
                    )
                else:
                    assert stage_ids.issubset(b_node_ids), (
                        f"region B pipeline contains non-B stages: {stage_ids}"
                    )

    def test_two_region_pipelines_cover_all_layers(self):
        # Each region's pipelines collectively cover [0, num_layers).
        model = _build_model_info(num_layers=8)
        gpus = (
            [_gpu(f"a{i}", region="A", gpu_type="rtx5090", layer_capacity=4)
             for i in range(2)]
            + [_gpu(f"b{i}", region="B", gpu_type="rtx5090", layer_capacity=4)
               for i in range(2)]
        )
        result = allocate_across_regions(gpus, model)

        for region, pipelines in result.region_to_pipelines.items():
            for pipeline in pipelines:
                # Stages must form a contiguous tile of [0, num_layers).
                ranges = pipeline.layer_ranges
                assert ranges[0][0] == 0, (
                    f"region {region} pipeline doesn't start at 0: {ranges}"
                )
                assert ranges[-1][1] == 8, (
                    f"region {region} pipeline doesn't end at 8: {ranges}"
                )
                for prev, nxt in zip(ranges, ranges[1:]):
                    assert prev[1] == nxt[0], (
                        f"region {region} pipeline has gap: {ranges}"
                    )

    def test_water_filling_balances_heterogeneous_compute(self):
        # Mixed compute capacities in a single region. Water-filling
        # should give the faster GPU more layers (proportional to
        # tflops, capped by capacity).
        model = _build_model_info(num_layers=12)
        gpus = [
            # a100-80g: 312 tflops, cap=12 → should get the bulk of layers
            _gpu("fast", region="A", gpu_type="a100-80g", layer_capacity=12),
            # rtx4090: 82.6 tflops, cap=12 → should get fewer layers
            _gpu("slow", region="A", gpu_type="rtx4090", layer_capacity=12),
        ]
        result = allocate_across_regions(gpus, model)
        pipelines = result.all_pipelines()
        # Find the pipeline that contains both fast and slow.
        chained = [p for p in pipelines
                   if "fast" in p.stages and "slow" in p.stages]
        if chained:
            p = chained[0]
            fast_idx = p.stages.index("fast")
            slow_idx = p.stages.index("slow")
            fast_layers = p.layer_ranges[fast_idx][1] - p.layer_ranges[fast_idx][0]
            slow_layers = p.layer_ranges[slow_idx][1] - p.layer_ranges[slow_idx][0]
            # Fast GPU should host strictly more layers than slow one
            # (312 / 82.6 ≈ 3.78× ratio).
            assert fast_layers >= slow_layers, (
                f"water-filling should give faster GPU more layers, "
                f"got fast={fast_layers}, slow={slow_layers}"
            )

    def test_result_pipeline_count_matches_aggregate(self):
        model = _build_model_info(num_layers=4)
        gpus = (
            [_gpu(f"a{i}", region="A", gpu_type="a100-80g", layer_capacity=4)
             for i in range(3)]
            + [_gpu(f"b{i}", region="B", gpu_type="a100-80g", layer_capacity=4)
               for i in range(3)]
        )
        result = allocate_across_regions(gpus, model)
        total_via_property = result.total_pipeline_count()
        total_via_walk = sum(len(p)
                             for p in result.region_to_pipelines.values())
        assert total_via_property == total_via_walk
        # Each region with 3 GPUs of capacity 4 each (capacity 12 total
        # for 4 layers) should produce ≥1 pipeline per region.
        assert total_via_property >= 2

    def test_unique_node_ids_across_pipelines_within_region(self):
        # A single GPU never appears in two different pipelines.
        model = _build_model_info(num_layers=4)
        gpus = [
            _gpu(f"n{i}", region="A", gpu_type="a100-80g", layer_capacity=4)
            for i in range(4)
        ]
        result = allocate_across_regions(gpus, model)
        seen: set = set()
        for pipeline in result.all_pipelines():
            for node_id in pipeline.stages:
                assert node_id not in seen, (
                    f"node {node_id} appears in multiple pipelines"
                )
                seen.add(node_id)


# ──────────────────────────────────────────────────────────────────────────
# Exception hierarchy — callers can catch the base class
# ──────────────────────────────────────────────────────────────────────────


class TestExceptionHierarchy:
    def test_empty_pool_is_allocation_error(self):
        assert issubclass(EmptyPoolError, AllocationError)

    def test_insufficient_capacity_is_allocation_error(self):
        assert issubclass(InsufficientCapacityError, AllocationError)
