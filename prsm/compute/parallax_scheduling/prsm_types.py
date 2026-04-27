"""
PRSM-native data types for the Parallax-derived scheduler.

Phase 3.x.6 Task 2 — adapts the vendored DP + water-filling allocator
to consume PRSM-shaped GPU descriptors. Algorithm itself is unchanged
from upstream; this module is pure data-model + conversion + a thin
region-partitioning shell around the upstream allocator.

What this module owns:

  - ``ParallaxGPU``: the PRSM-side GPU descriptor that operators construct
    from their environment. Includes ``region`` (PRSM-original — upstream
    handles regions implicitly via RTT clustering; PRSM names them
    explicitly), ``stake_amount`` (used by Adapter C — Phase 3.x.6 Task 5),
    and ``tier_attestation`` (used by Adapter B — Phase 3.x.6 Task 5).

  - ``to_parallax_node``: pure conversion from ``ParallaxGPU`` →
    upstream ``Node``. Does NOT consult external state. The trust
    adapters (Task 5) wrap the conversion call site, not this function.

  - ``partition_by_region``: groups GPUs by region. Run BEFORE the
    upstream allocator so each region's pool is allocated independently;
    pipelines never span regions. Implements the paper's "region-based
    heuristic" §3.2 explicitly rather than relying on RTT-clustering.

  - ``allocate_across_regions``: orchestrates per-region allocation +
    aggregates results. Returns the union of pipelines across regions.

What this module does NOT own:

  - Profile DHT publishing — that's Phase 3.x.6 Task 4.
  - Trust adapters (anchor verify, tier gate, stake-weighted, consensus
    mismatch) — that's Phase 3.x.6 Task 5. Adapters operate ON
    ``ParallaxGPU`` lists BEFORE handing them to ``allocate_across_regions``.
  - Per-request routing — that's Task 3 (Phase-2 DAG sweep).
  - The allocation algorithm itself — that's vendored in
    ``layer_allocation.py`` from upstream Parallax (Apache-2.0).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from prsm.compute.parallax_scheduling.model_info import ModelInfo
from prsm.compute.parallax_scheduling.node import Node, NodeHardwareInfo
from prsm.compute.parallax_scheduling.node_management import NodeManager


# ──────────────────────────────────────────────────────────────────────────
# Tier attestation — distinguishes hardware-TEE from software-only
# ──────────────────────────────────────────────────────────────────────────


# Sentinel string for "no hardware TEE attestation available". Operators
# whose nodes run on consumer hardware (PS5, Mac, plain Linux) advertise
# this. The Phase-1 allocator does NOT discriminate on tier — that's
# Adapter B's job (Phase 3.x.6 Task 5 — pre-route filter on Phase-2).
TIER_ATTESTATION_NONE = "tier-none"


# ──────────────────────────────────────────────────────────────────────────
# ParallaxGPU — PRSM-side descriptor
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ParallaxGPU:
    """A single GPU's worth of PRSM-side scheduling input.

    Frozen so callers can hash + use as dict keys + pass through trust
    adapters without worrying about mutation between gates.

    Required fields (operator MUST supply):
      node_id            32-char hex; resolves to a manifest publisher
                         on the Phase 3.x.5 manifest DHT. Adapter A
                         filters on this before allocation.
      region             Operator-named region string. Pipelines never
                         span regions. Examples: "us-west-2", "eu-fra-1",
                         "home". No semantics enforced — equality only.
      layer_capacity     Maximum number of transformer layers this GPU
                         can hold given its VRAM. Computed by the
                         operator from `(vram_gb, layer_size_bytes)` —
                         we don't recompute here.
      stake_amount       FTNS wei staked by this node's operator. Used
                         by Adapter C to weight profile trust. 0 means
                         unstaked → effectively excluded from routing.
      tier_attestation   TEE-attestation hash, or ``TIER_ATTESTATION_NONE``.
                         Used by Adapter B at request time, not by
                         Phase-1 allocation.

    Hardware fields (used to construct the upstream ``Node``):
      tflops_fp16        Compute throughput. Drives water-filling.
      memory_gb          VRAM. Used by upstream's roofline model.
      memory_bandwidth_gbps  Memory bandwidth. Drives roofline.
      gpu_name           Cosmetic; logged for debugging.
      device             "cuda" / "mps" / "cpu". Affects upstream's
                         compute_max_batch_size dispatch.
      num_gpus           Per-node GPU count for the upstream model.
                         Default 1 — most volunteer setups have 1 GPU.
    """

    # ── PRSM-original fields ────────────────────────────────────────────
    node_id: str
    region: str
    layer_capacity: int
    stake_amount: int
    tier_attestation: str

    # ── Hardware fields (mirror NodeHardwareInfo) ───────────────────────
    tflops_fp16: float
    memory_gb: float
    memory_bandwidth_gbps: float
    gpu_name: str = ""
    device: str = "cuda"
    num_gpus: int = 1

    # ── Optional metadata (operators can override) ──────────────────────
    quantization_speedup: float = 1.0
    rtt_to_nodes: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        # Defensive validation — these are operator-supplied inputs
        # that propagate into a load-bearing scheduler. Reject obvious
        # garbage early rather than letting it surface as confusing
        # behavior in the DP search or water-filling step.
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("node_id must be a non-empty string")
        if not isinstance(self.region, str) or not self.region:
            raise ValueError("region must be a non-empty string")
        if not isinstance(self.layer_capacity, int) or self.layer_capacity < 0:
            raise ValueError(
                f"layer_capacity must be a non-negative int, got {self.layer_capacity!r}"
            )
        if not isinstance(self.stake_amount, int) or self.stake_amount < 0:
            raise ValueError(
                f"stake_amount must be a non-negative int (FTNS wei), "
                f"got {self.stake_amount!r}"
            )
        if not isinstance(self.tier_attestation, str) or not self.tier_attestation:
            raise ValueError("tier_attestation must be a non-empty string")
        if self.tflops_fp16 <= 0:
            raise ValueError(
                f"tflops_fp16 must be positive, got {self.tflops_fp16}"
            )
        if self.memory_gb <= 0:
            raise ValueError(f"memory_gb must be positive, got {self.memory_gb}")
        if self.memory_bandwidth_gbps <= 0:
            raise ValueError(
                f"memory_bandwidth_gbps must be positive, "
                f"got {self.memory_bandwidth_gbps}"
            )
        if self.num_gpus < 1:
            raise ValueError(f"num_gpus must be ≥ 1, got {self.num_gpus}")


# ──────────────────────────────────────────────────────────────────────────
# Conversion: ParallaxGPU → upstream Node
# ──────────────────────────────────────────────────────────────────────────


def to_parallax_node(gpu: ParallaxGPU, model_info: ModelInfo) -> Node:
    """Construct an upstream ``Node`` from a PRSM ``ParallaxGPU``.

    Pure function. Does NOT call into the manifest DHT, anchor, or
    stake manager — those gates are the trust adapters' job (Task 5).
    The conversion preserves the operator-supplied ``layer_capacity``
    by setting upstream's ``manual_layer_assignment=False``: the
    upstream allocator will compute capacity from VRAM, but we
    surface ``layer_capacity`` separately as a sanity check (Tasks
    2/3 will use it for the DP-allocator's per-node cap).
    """
    hw = NodeHardwareInfo(
        node_id=gpu.node_id,
        num_gpus=gpu.num_gpus,
        tflops_fp16=gpu.tflops_fp16,
        gpu_name=gpu.gpu_name,
        memory_gb=gpu.memory_gb,
        memory_bandwidth_gbps=gpu.memory_bandwidth_gbps,
        device=gpu.device,
    )
    node = Node(
        node_id=gpu.node_id,
        hardware=hw,
        model_info=model_info,
        # Force the upstream's max-concurrent-requests path to fall
        # through to a heuristic; tests + production runs that need
        # specific concurrency caps set them explicitly afterward.
        _force_max_concurrent_requests=True,
        rtt_to_nodes=dict(gpu.rtt_to_nodes) if gpu.rtt_to_nodes else None,
    )
    # Operator-supplied speedup (e.g., FP8 vs FP16 quant).
    node.quantization_speedup = gpu.quantization_speedup
    return node


# ──────────────────────────────────────────────────────────────────────────
# Region partitioning + cross-region allocation orchestrator
# ──────────────────────────────────────────────────────────────────────────


def partition_by_region(
    gpus: List[ParallaxGPU],
) -> Dict[str, List[ParallaxGPU]]:
    """Group GPUs by ``region``.

    Implements the paper §3.2 region-based heuristic at the input
    boundary: pipelines NEVER span regions because the allocator
    only sees one region's GPUs at a time. Cross-region traffic
    happens only at request-time DAG routing (Task 3), and that's
    where it pays the latency cost — not on the per-stage hot path.

    Empty input → empty mapping. Single-region input → single key.
    """
    out: Dict[str, List[ParallaxGPU]] = defaultdict(list)
    for gpu in gpus:
        out[gpu.region].append(gpu)
    return dict(out)


@dataclass
class AllocationResult:
    """Per-region allocation result.

    Pipelines are upstream's primitive (a contiguous-layer assignment
    to a sequence of nodes). We aggregate them across regions but
    record which region each pipeline belongs to so downstream
    request-routing knows the topology.
    """

    region_to_pipelines: Dict[str, List["RegionPipeline"]] = field(
        default_factory=dict
    )

    def all_pipelines(self) -> List["RegionPipeline"]:
        """Flatten across regions for callers that just need the union."""
        out: List[RegionPipeline] = []
        for pipelines in self.region_to_pipelines.values():
            out.extend(pipelines)
        return out

    def total_pipeline_count(self) -> int:
        return sum(len(p) for p in self.region_to_pipelines.values())


@dataclass(frozen=True)
class RegionPipeline:
    """A single pipeline produced by the allocator, tagged with its region.

    ``stages`` is an ordered list of node_ids, each carrying a
    contiguous slice of the model's layers. ``layer_ranges`` is the
    parallel list of (start_layer, end_layer) — half-open.
    """

    region: str
    stages: List[str]  # node_ids in pipeline order
    layer_ranges: List[tuple]  # (start, end) per stage; end exclusive


# ──────────────────────────────────────────────────────────────────────────
# Errors raised at this layer
# ──────────────────────────────────────────────────────────────────────────


class AllocationError(Exception):
    """Base error for allocation failures."""


class InsufficientCapacityError(AllocationError):
    """Total GPU layer-capacity in a region is less than model's
    layer count — no valid pipeline exists in that region."""


class EmptyPoolError(AllocationError):
    """No GPUs supplied. Caller is responsible for distinguishing
    'no GPUs at all' from 'GPUs all filtered out by trust adapters'."""


# ──────────────────────────────────────────────────────────────────────────
# Top-level orchestrator
# ──────────────────────────────────────────────────────────────────────────


def allocate_across_regions(
    gpus: List[ParallaxGPU],
    model_info: ModelInfo,
    *,
    allow_partial: bool = False,
) -> AllocationResult:
    """Run the upstream allocator independently per region; aggregate.

    Algorithm itself (DP + water-filling) is the upstream Apache-2.0
    code in ``layer_allocation.py``. This function is the PRSM-original
    shell that:

      1. Partitions ``gpus`` by region.
      2. For each region with sufficient capacity, builds an upstream
         ``NodeManager`` populated with that region's nodes and runs
         ``DynamicProgrammingLayerAllocator.allocate_from_standby()``.
      3. Aggregates the resulting pipelines tagged with their region.

    If ``allow_partial`` is False (default): regions with insufficient
    capacity raise ``InsufficientCapacityError`` immediately. If True:
    those regions are skipped with a logged warning and the result
    contains only the regions that succeeded.

    Empty input raises ``EmptyPoolError`` regardless of ``allow_partial`` —
    the empty case is unambiguously a caller bug, not a partial-success
    scenario.
    """
    # Late import to avoid a circular dep — layer_allocation.py imports
    # node_management.py imports node.py imports model_info.py; those
    # form the algorithmic core. prsm_types.py imports the algorithmic
    # core, and the algorithmic core's __init__ may eventually want to
    # re-export prsm_types. Keep the boundary clean.
    from prsm.compute.parallax_scheduling.layer_allocation import (
        DynamicProgrammingLayerAllocator,
    )

    if not gpus:
        raise EmptyPoolError("no GPUs supplied")

    by_region = partition_by_region(gpus)
    result = AllocationResult()

    for region, region_gpus in by_region.items():
        total_capacity = sum(g.layer_capacity for g in region_gpus)
        if total_capacity < model_info.num_layers:
            if allow_partial:
                continue
            raise InsufficientCapacityError(
                f"region {region!r}: total layer_capacity={total_capacity} "
                f"< num_layers={model_info.num_layers}"
            )

        # Convert to upstream Nodes + register with a fresh NodeManager.
        nodes = [to_parallax_node(g, model_info) for g in region_gpus]
        nm = NodeManager(initial_nodes=nodes)

        # Run the upstream Phase-1 algorithm.
        allocator = DynamicProgrammingLayerAllocator(
            model_info=model_info,
            node_management=nm,
        )
        allocator.allocate_from_standby()

        # Translate upstream's Node-mutation result back into
        # PRSM-side RegionPipeline dataclasses. Upstream sets
        # start_layer/end_layer on Node; pipelines are inferred by
        # walking layer_ranges in order.
        region_pipelines = _extract_pipelines(nm, region=region)
        if region_pipelines:
            result.region_to_pipelines[region] = region_pipelines

    return result


def _extract_pipelines(nm: NodeManager, *, region: str) -> List[RegionPipeline]:
    """Walk a NodeManager whose nodes have been allocated by the upstream
    DP allocator and produce ordered ``RegionPipeline`` records.

    Upstream's primitive is "node has start_layer/end_layer set". A
    pipeline is a chain of nodes whose layer ranges tile [0, L). We
    reconstruct that chain by sorting active nodes by start_layer and
    splitting whenever a layer-range gap appears (which signals a
    pipeline boundary).
    """
    # Snapshot all nodes (active+standby) that have non-None start_layer.
    # NodeManager.nodes is a @property, not a method.
    assigned: List[Node] = []
    for node in nm.nodes:
        if node.start_layer is None or node.end_layer is None:
            continue
        assigned.append(node)

    if not assigned:
        return []

    # Sort by (start_layer, node_id) — node_id breaks ties deterministically.
    assigned.sort(key=lambda n: (n.start_layer, n.node_id))

    pipelines: List[RegionPipeline] = []
    current_stages: List[str] = []
    current_ranges: List[tuple] = []
    expected_next_start: Optional[int] = None

    for node in assigned:
        # New pipeline starts when start_layer is 0 OR there's a gap
        # from the previously-extended pipeline's end_layer.
        is_pipeline_start = (
            node.start_layer == 0
            or expected_next_start is None
            or node.start_layer != expected_next_start
        )
        if is_pipeline_start and current_stages:
            pipelines.append(
                RegionPipeline(
                    region=region,
                    stages=current_stages,
                    layer_ranges=current_ranges,
                )
            )
            current_stages = []
            current_ranges = []

        current_stages.append(node.node_id)
        current_ranges.append((node.start_layer, node.end_layer))
        expected_next_start = node.end_layer

    if current_stages:
        pipelines.append(
            RegionPipeline(
                region=region,
                stages=current_stages,
                layer_ranges=current_ranges,
            )
        )

    return pipelines
