"""
ParallaxScheduledExecutor — Phase 3.x.6 Task 6.

Implements the ``InferenceExecutor`` Protocol on top of the vendored
Parallax scheduler + the four PRSM trust adapters (Phase 3.x.6 Task 5).

Composition (per design plan §3.2):

  1. Gather current GPU pool via ``gpu_pool_provider()``.
  2. ``TrustStack.filter_pool`` — anchor-verify (Adapter A).
  3. Phase-1 ``allocate_across_regions`` (cached per model+pool).
  4. Per-request: ``TrustStack.filter_for_request`` — tier gate (Adapter B).
  5. Build ``RequestRouter`` over the (allocation, profile_source, pool)
     where the profile_source is already wrapped by Adapter C.
  6. ``router.route(...)`` → ``GPUChain``.
  7. Hand the chain to an injected ``ChainExecutor`` for actual layer
     dispatch; receive ``(output, duration, tee_attestation, tee_type,
     epsilon_spent)``.
  8. ``ConsensusMismatchHook`` post-route: if sampled, route + execute
     an alternate chain and compare outputs (Adapter D).
  9. Sign ``InferenceReceipt`` under the executor's ``NodeIdentity``.

Phase-1 cache invalidation (paper §3.4):
  - Cache hit: cached allocation reused IFF every stage in the cached
    pipelines is still present in the current trust-filtered pool. This
    is the "localized GPU-leave absorbed" case.
  - Cache miss / coverage gap: rebuild the allocation, retry the route
    once. A second NoCoverageError is a routing failure, not a cache
    issue, and surfaces as ``InferenceResult.failure``.

Drop-in replacement: callers wiring the orchestrator to either
``TensorParallelInferenceExecutor`` or ``ParallaxScheduledExecutor`` get
the same ``execute / estimate_cost / supported_models`` surface.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Mapping, Optional, Protocol, Sequence, Tuple

from prsm.compute.inference.executor import (
    InferenceExecutor,
    UnsupportedModelError,
)
from prsm.compute.inference.models import (
    InferenceReceipt,
    InferenceRequest,
    InferenceResult,
)
from prsm.compute.inference.receipt import sign_receipt
from prsm.compute.parallax_scheduling.model_info import ModelInfo
from prsm.compute.parallax_scheduling.prsm_request_router import (
    BudgetExceededError,
    EmptyAllocationError,
    GPUChain,
    NoCoverageError,
    RegionNotFoundError,
    RequestRouter,
    RouteRequest,
)
from prsm.compute.parallax_scheduling.prsm_types import (
    AllocationError,
    AllocationResult,
    EmptyPoolError,
    InsufficientCapacityError,
    ParallaxGPU,
    allocate_across_regions,
)
from prsm.compute.parallax_scheduling.trust_adapter import (
    ChallengeRecord,
    TierGateRejected,
    TrustStack,
)
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import NodeIdentity


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# ChainExecutor Protocol — abstracts actual layer dispatch
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChainExecutionResult:
    """What a ``ChainExecutor`` returns to the scheduler.

    Fields:
      output            Generated text.
      duration_seconds  Wall-clock time for the chain dispatch.
      tee_attestation   Attestation bytes from the executing chain head.
      tee_type          Effective TEE type (worst case across stages).
      epsilon_spent     DP epsilon recorded by the privacy budget.
    """

    output: str
    duration_seconds: float
    tee_attestation: bytes
    tee_type: TEEType
    epsilon_spent: float


class ChainExecutor(Protocol):
    """Adapter: given a chosen ``GPUChain``, run inference end-to-end.

    Production: wraps ``TensorParallelExecutor.execute_parallel(...)``
    over the chain's stages with cross-host RPC. Tests: a fake that
    returns deterministic output without spinning real shards.
    """

    def execute_chain(
        self, *, request: InferenceRequest, chain: GPUChain
    ) -> ChainExecutionResult:
        ...


# ──────────────────────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────────────────────


GpuPoolProvider = Callable[[], Sequence[ParallaxGPU]]
"""Callable that returns the current candidate GPU pool. Called once
per ``execute()`` invocation. Production wires this to the live
``ProfileDHT`` membership snapshot."""


# ──────────────────────────────────────────────────────────────────────────
# ParallaxScheduledExecutor
# ──────────────────────────────────────────────────────────────────────────


class ParallaxScheduledExecutor(InferenceExecutor):
    """``InferenceExecutor`` Protocol implementation backed by the
    vendored Parallax scheduler + PRSM trust stack.

    Constructor args:
      gpu_pool_provider      Callable returning the current pool.
      trust_stack            Composed ``TrustStack`` (Phase 3.x.6 Task 5).
      model_catalog          Mapping ``model_id → ModelInfo`` for any
                             model this executor advertises.
      chain_executor         Concrete chain-dispatch adapter.
      node_identity          ``NodeIdentity`` used to sign receipts
                             (this node = the settling/serving node).
      cost_per_layer         Base FTNS price per model layer for the
                             roofline cost estimate. Default 0.01.
      privacy_overhead       Per-PrivacyLevel cost multipliers. Default
                             matches Phase 3.x.1 schedule (1.00 / 1.10
                             / 1.25 / 1.50 for NONE/STANDARD/HIGH/MAX).
      allow_partial_regions  Forwarded to ``allocate_across_regions``.
                             False (default) raises if any region has
                             insufficient capacity. True skips them.
    """

    DEFAULT_PRIVACY_OVERHEAD: Mapping[PrivacyLevel, Decimal] = {
        PrivacyLevel.NONE: Decimal("1.00"),
        PrivacyLevel.STANDARD: Decimal("1.10"),
        PrivacyLevel.HIGH: Decimal("1.25"),
        PrivacyLevel.MAXIMUM: Decimal("1.50"),
    }

    def __init__(
        self,
        *,
        gpu_pool_provider: GpuPoolProvider,
        trust_stack: TrustStack,
        model_catalog: Mapping[str, ModelInfo],
        chain_executor: ChainExecutor,
        node_identity: NodeIdentity,
        cost_per_layer: Decimal = Decimal("0.01"),
        privacy_overhead: Optional[Mapping[PrivacyLevel, Decimal]] = None,
        allow_partial_regions: bool = False,
    ) -> None:
        if gpu_pool_provider is None or not callable(gpu_pool_provider):
            raise RuntimeError(
                "ParallaxScheduledExecutor requires a callable gpu_pool_provider"
            )
        if trust_stack is None:
            raise RuntimeError(
                "ParallaxScheduledExecutor requires a trust_stack"
            )
        if chain_executor is None or not hasattr(chain_executor, "execute_chain"):
            raise RuntimeError(
                "ParallaxScheduledExecutor requires a chain_executor with "
                ".execute_chain(...) → ChainExecutionResult"
            )
        if node_identity is None or not hasattr(node_identity, "node_id"):
            raise RuntimeError(
                "ParallaxScheduledExecutor requires a NodeIdentity for "
                "receipt signing"
            )

        self._pool_provider = gpu_pool_provider
        self._trust = trust_stack
        self._catalog = dict(model_catalog)
        self._chain_executor = chain_executor
        self._identity = node_identity
        self._cost_per_layer = Decimal(cost_per_layer)
        self._privacy_overhead = dict(
            privacy_overhead or self.DEFAULT_PRIVACY_OVERHEAD
        )
        self._allow_partial_regions = allow_partial_regions

        # Phase-1 allocation cache:
        #   model_id → (AllocationResult, frozenset_of_stage_node_ids)
        # Localized GPU-leave is absorbed when the cached stage-set is
        # still a subset of the trust-filtered pool's node-ids; otherwise
        # the cache is invalidated and Phase-1 recomputes.
        self._alloc_cache: dict[str, Tuple[AllocationResult, frozenset]] = {}

        # Public-facing recompute counter — operators monitor this for
        # DHT churn / pool-instability signals. Tests assert on it to
        # verify §3.4 behavior.
        self._phase1_recompute_count: int = 0

    # ── InferenceExecutor surface ─────────────────────────────────────

    def supported_models(self) -> list[str]:
        return list(self._catalog.keys())

    async def estimate_cost(self, request: InferenceRequest) -> Decimal:
        """Roofline cost — scales linearly with layer count and the
        per-tier overhead. Independent of pool composition; the actual
        chain may execute faster or slower but the price is fixed at
        request time so callers can pre-validate budgets."""
        if request.model_id not in self._catalog:
            raise UnsupportedModelError(f"Unknown model_id: {request.model_id}")
        model_info = self._catalog[request.model_id]
        base = self._cost_per_layer * Decimal(model_info.num_layers)
        overhead = self._privacy_overhead.get(
            request.privacy_tier, Decimal("1.0")
        )
        return base * overhead

    async def execute(self, request: InferenceRequest) -> InferenceResult:
        """End-to-end: filter → allocate → route → execute → sign."""
        # 1. Catalog lookup.
        if request.model_id not in self._catalog:
            return InferenceResult.failure(
                request.request_id,
                f"Unknown model_id: {request.model_id}",
            )
        model_info = self._catalog[request.model_id]

        # 2. Budget gate.
        cost = await self.estimate_cost(request)
        if request.budget_ftns < cost:
            return InferenceResult.failure(
                request.request_id,
                f"Insufficient budget: {request.budget_ftns} FTNS < "
                f"required {cost} FTNS",
            )

        # 3. Pool gathering + Adapter A (anchor verify).
        try:
            raw_pool = list(self._pool_provider())
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "ParallaxScheduledExecutor: gpu_pool_provider raised"
            )
            return InferenceResult.failure(
                request.request_id,
                f"GPU pool provider failure: {exc}",
            )
        if not raw_pool:
            return InferenceResult.failure(
                request.request_id, "GPU pool is empty"
            )
        anchor_filtered = self._trust.filter_pool(raw_pool)
        if not anchor_filtered:
            return InferenceResult.failure(
                request.request_id,
                "no GPU passed anchor verification",
            )

        # 4. Adapter B — tier gate.
        try:
            tier_filtered = self._trust.filter_for_request(
                anchor_filtered, request.privacy_tier
            )
        except TierGateRejected as exc:
            return InferenceResult.failure(
                request.request_id, f"tier gate refusal: {exc}"
            )

        if not tier_filtered:
            # NONE-tier with empty pool — treated the same as no GPUs.
            return InferenceResult.failure(
                request.request_id, "no eligible GPU after tier gate"
            )

        # 5. Phase-1 allocation (cache or recompute).
        try:
            allocation = self._get_or_build_allocation(
                request.model_id, tier_filtered, model_info
            )
        except EmptyPoolError as exc:
            return InferenceResult.failure(
                request.request_id, f"empty pool: {exc}"
            )
        except InsufficientCapacityError as exc:
            return InferenceResult.failure(
                request.request_id, f"insufficient capacity: {exc}"
            )
        except AllocationError as exc:
            return InferenceResult.failure(
                request.request_id, f"allocation failure: {exc}"
            )

        # 6. Build router; route. Retry once on NoCoverageError after
        # forced Phase-1 recompute (paper §3.4).
        try:
            chain = self._route_with_retry(
                allocation=allocation,
                model_info=model_info,
                tier_filtered=tier_filtered,
                request=request,
                model_id=request.model_id,
            )
        except (NoCoverageError, EmptyAllocationError) as exc:
            return InferenceResult.failure(
                request.request_id, f"routing coverage failure: {exc}"
            )
        except RegionNotFoundError as exc:
            return InferenceResult.failure(
                request.request_id, f"region not found: {exc}"
            )
        except BudgetExceededError as exc:
            return InferenceResult.failure(
                request.request_id, f"latency budget exceeded: {exc}"
            )

        # 7. Execute primary chain.
        try:
            primary_outcome = self._chain_executor.execute_chain(
                request=request, chain=chain
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "ParallaxScheduledExecutor: primary chain executor raised"
            )
            return InferenceResult.failure(
                request.request_id, f"chain execution failure: {exc}"
            )

        # 8. Adapter D — optional redundant execution + mismatch check.
        self._maybe_run_consensus_check(
            request=request,
            allocation=allocation,
            tier_filtered=tier_filtered,
            model_info=model_info,
            primary_chain=chain,
            primary_outcome=primary_outcome,
        )

        # 9. Build + sign receipt; return InferenceResult.
        receipt = self._build_signed_receipt(
            request=request,
            cost=cost,
            outcome=primary_outcome,
        )
        return InferenceResult(
            request_id=request.request_id,
            success=True,
            output=primary_outcome.output,
            receipt=receipt,
        )

    # ── Allocation cache ──────────────────────────────────────────────

    @property
    def phase1_recompute_count(self) -> int:
        """Counter incremented every time Phase-1 alloc is rebuilt.
        Tests use this to verify §3.4 (recompute-only-on-coverage-gap)."""
        return self._phase1_recompute_count

    def _get_or_build_allocation(
        self,
        model_id: str,
        pool: Sequence[ParallaxGPU],
        model_info: ModelInfo,
    ) -> AllocationResult:
        """Cache hit when the cached stage-set is a subset of the
        current trust-filtered pool. Otherwise rebuild."""
        pool_ids = frozenset(g.node_id for g in pool)
        cached = self._alloc_cache.get(model_id)
        if cached is not None:
            cached_alloc, cached_stage_set = cached
            if cached_stage_set.issubset(pool_ids):
                return cached_alloc
        # Rebuild.
        return self._rebuild_allocation(
            model_id, pool, model_info, pool_ids
        )

    def _rebuild_allocation(
        self,
        model_id: str,
        pool: Sequence[ParallaxGPU],
        model_info: ModelInfo,
        pool_ids: frozenset,
    ) -> AllocationResult:
        allocation = allocate_across_regions(
            list(pool),
            model_info,
            allow_partial=self._allow_partial_regions,
        )
        stage_set = frozenset(
            node_id
            for pipelines in allocation.region_to_pipelines.values()
            for pipeline in pipelines
            for node_id in pipeline.stages
        )
        self._alloc_cache[model_id] = (allocation, stage_set)
        self._phase1_recompute_count += 1
        return allocation

    def _route_with_retry(
        self,
        *,
        allocation: AllocationResult,
        model_info: ModelInfo,
        tier_filtered: Sequence[ParallaxGPU],
        request: InferenceRequest,
        model_id: str,
    ) -> GPUChain:
        """Try the cached/built allocation; on coverage gap, force one
        recompute and retry. Second NoCoverageError propagates."""
        router = RequestRouter(
            allocation=allocation,
            profile_source=self._trust.profile_source,
            model_info=model_info,
            original_gpus=list(tier_filtered),
        )
        route_req = RouteRequest(request_id=request.request_id)
        try:
            return router.route(route_req)
        except NoCoverageError:
            # Force rebuild and retry — even if the cached allocation
            # had every node, the membership-changed shape may need
            # different stage assignments. Caller above wraps a second
            # NoCoverageError into InferenceResult.failure.
            pool_ids = frozenset(g.node_id for g in tier_filtered)
            fresh_alloc = self._rebuild_allocation(
                model_id, tier_filtered, model_info, pool_ids
            )
            fresh_router = RequestRouter(
                allocation=fresh_alloc,
                profile_source=self._trust.profile_source,
                model_info=model_info,
                original_gpus=list(tier_filtered),
            )
            return fresh_router.route(route_req)

    # ── Consensus-mismatch hook ───────────────────────────────────────

    def _maybe_run_consensus_check(
        self,
        *,
        request: InferenceRequest,
        allocation: AllocationResult,
        tier_filtered: Sequence[ParallaxGPU],
        model_info: ModelInfo,
        primary_chain: GPUChain,
        primary_outcome: ChainExecutionResult,
    ) -> Optional[ChallengeRecord]:
        """When sampled, attempt to route + execute an alternate chain
        and ask the consensus hook to compare. Failures inside this
        path are logged but do NOT propagate — primary already succeeded
        and the user's request is satisfied."""
        hook = self._trust.consensus_hook
        if not hook.should_sample_redundant(request.request_id):
            return None

        secondary_chain = self._try_route_alternate_chain(
            allocation=allocation,
            tier_filtered=tier_filtered,
            model_info=model_info,
            request=request,
            primary_chain=primary_chain,
        )
        if secondary_chain is None:
            logger.info(
                "ParallaxScheduledExecutor: sampled request %s for "
                "redundant execution but no alternate chain available; "
                "skipping consensus check",
                request.request_id,
            )
            return None

        try:
            secondary_outcome = self._chain_executor.execute_chain(
                request=request, chain=secondary_chain
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "ParallaxScheduledExecutor: secondary chain executor raised "
                "for request %s; skipping consensus check",
                request.request_id,
            )
            return None

        return hook.compare_and_challenge(
            request_id=request.request_id,
            primary_output=primary_outcome.output.encode("utf-8"),
            secondary_output=secondary_outcome.output.encode("utf-8"),
            primary_chain=primary_chain,
            secondary_chain=secondary_chain,
        )

    def _try_route_alternate_chain(
        self,
        *,
        allocation: AllocationResult,
        tier_filtered: Sequence[ParallaxGPU],
        model_info: ModelInfo,
        request: InferenceRequest,
        primary_chain: GPUChain,
    ) -> Optional[GPUChain]:
        """Build an alternate chain by excluding primary's stages from
        the pool and re-running Phase-1 on the remainder. Returns None
        if the remainder can't cover the model — that's a normal "not
        enough redundancy" signal, not an error."""
        primary_node_ids = set(primary_chain.stages)
        alternate_pool = [
            gpu for gpu in tier_filtered
            if gpu.node_id not in primary_node_ids
        ]
        if not alternate_pool:
            return None
        try:
            alt_allocation = allocate_across_regions(
                alternate_pool,
                model_info,
                allow_partial=True,  # alternates are best-effort
            )
        except (AllocationError, InsufficientCapacityError, EmptyPoolError):
            return None

        if alt_allocation.total_pipeline_count() == 0:
            return None

        try:
            alt_router = RequestRouter(
                allocation=alt_allocation,
                profile_source=self._trust.profile_source,
                model_info=model_info,
                original_gpus=alternate_pool,
            )
            return alt_router.route(
                RouteRequest(request_id=f"{request.request_id}-alt")
            )
        except (NoCoverageError, EmptyAllocationError, RegionNotFoundError):
            return None

    # ── Receipt assembly ──────────────────────────────────────────────

    def _build_signed_receipt(
        self,
        *,
        request: InferenceRequest,
        cost: Decimal,
        outcome: ChainExecutionResult,
    ) -> InferenceReceipt:
        """Assemble + Ed25519-sign the receipt under this node's identity."""
        output_hash = hashlib.sha256(outcome.output.encode("utf-8")).digest()
        unsigned = InferenceReceipt(
            job_id=f"parallax-job-{uuid.uuid4().hex[:12]}",
            request_id=request.request_id,
            model_id=request.model_id,
            content_tier=request.content_tier,
            privacy_tier=request.privacy_tier,
            epsilon_spent=outcome.epsilon_spent,
            tee_type=outcome.tee_type,
            tee_attestation=outcome.tee_attestation,
            output_hash=output_hash,
            duration_seconds=outcome.duration_seconds,
            cost_ftns=cost,
        )
        return sign_receipt(unsigned, self._identity)
