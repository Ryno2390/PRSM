"""Phase 7.1 Task 2: MultiShardDispatcher.

Wraps RemoteShardDispatcher for k-of-n redundant execution. Dispatches
one shard to `k` pre-selected providers in parallel via asyncio.gather,
collects their DispatchResults, and runs ShardConsensus over the
returned ShardExecutionReceipts to partition them into majority /
minority by output_hash.

Contract (per docs/2026-04-21-phase7.1-redundant-execution-design.md §3.3–3.5):
  - Pre-selection: node_ids list is authoritative. This module does NOT
    pick providers from an eligible pool — the orchestrator's
    _select_top_k ranks the pool and hands us the k it wants. Keeps
    selection policy in one place (Phase 7.1 Task 5).
  - Parallel: one asyncio.gather across all k. No retry within a single
    k-of-n dispatch — if a provider times out or errors, they just
    don't contribute to consensus. Retries are a higher-level concern
    (orchestrator may call us again with a different k-set).
  - Partial failure tolerance: honest-work failures (preemption, TEE
    missing, dispatch errors) are classified as "did not respond" and
    excluded from the consensus group. This preserves Phase 2.1 Line A's
    honest-work-failure semantics.
  - Single escrow: each single-dispatch call receives the per-provider
    share (total quote / k). The requester's aggregate escrow is the
    sum, same as Phase 3.1 batched settlement expects.

Errors:
  - InsufficientResponsesError: fewer successful responses than the
    consensus mode's threshold. Can't reach consensus regardless of
    hash agreement.
  - ConsensusFailedError: enough responses but no agreeing group of
    sufficient size (e.g., k=3 and all three returned different hashes).
    This is the case that triggers slashing downstream — a minority
    disagreement is still an agreement (majority wins); but a full
    disagreement surfaces here for the orchestrator to decide whether
    to refund or re-dispatch.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from prsm.compute.model_sharding.models import ModelShard, PipelineStakeTier
from prsm.compute.remote_dispatcher import (
    DispatchResult,
    MissingAttestationError,
    PeerNotConnectedError,
    RemoteShardDispatcher,
    ShardDispatchError,
    ShardPreemptedError,
)
from prsm.compute.shard_consensus import ConsensusOutcome, ShardConsensus
from prsm.compute.shard_receipt import ShardExecutionReceipt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConsensusShardReceipt:
    """Per-shard output of MultiShardDispatcher.

    Distinct from ShardExecutionReceipt (a single provider's signed
    output): this is the dispatcher-level container binding k signed
    receipts under a consensus decision. The orchestrator forwards
    `majority_receipts` to the settlement accumulator (payment goes to
    providers in majority) and `minority_receipts` to the challenge
    queue (payment withheld; slashing triggered via Phase 7's path).
    """
    job_id: str
    shard_index: int
    consensus_mode: str
    k: int
    responded: int
    agreed_output_hash: str
    agreed_output: np.ndarray
    majority_receipts: List[ShardExecutionReceipt]
    minority_receipts: List[ShardExecutionReceipt]
    consensus_reached_unix: int


class InsufficientResponsesError(RuntimeError):
    """Fewer successful responses than the consensus mode's threshold.

    The caller can't determine a winner with the data they have —
    usually the right response is to refund remaining escrow and either
    re-dispatch with a different k-set or surface the job failure."""

    def __init__(self, responded: int, k: int, threshold: int):
        self.responded = responded
        self.k = k
        self.threshold = threshold
        super().__init__(
            f"insufficient responses for consensus: {responded} of {k} "
            f"responded, need {threshold}"
        )


class ConsensusFailedError(RuntimeError):
    """Enough providers responded but no group reached the agreement
    threshold. Surfaces to the orchestrator so it can decide between
    refund, re-dispatch, or escalation — MultiDispatcher does not
    auto-retry because the right action depends on job-level policy."""

    def __init__(self, responded: int, k: int, unique_hashes: int):
        self.responded = responded
        self.k = k
        self.unique_hashes = unique_hashes
        super().__init__(
            f"consensus failed: {responded} of {k} responded with "
            f"{unique_hashes} distinct output hashes; no group reached "
            f"the agreement threshold"
        )


class MultiShardDispatcher:
    """k-of-n redundant dispatch over a pre-selected provider set."""

    def __init__(
        self,
        single_dispatcher: RemoteShardDispatcher,
        consensus: ShardConsensus,
    ):
        self.single_dispatcher = single_dispatcher
        self.consensus = consensus

    async def dispatch_with_consensus(
        self,
        shard: ModelShard,
        input_tensor: np.ndarray,
        node_ids: List[str],
        job_id: str,
        stake_tier: PipelineStakeTier,
        escrow_amount_per_provider_ftns: float,
        require_tee_attestation: bool = False,
    ) -> ConsensusShardReceipt:
        """Dispatch shard to all `node_ids` concurrently; return the
        consensus decision.

        Caller contract: len(node_ids) == consensus.k. We validate this
        at entry — a mismatch is a caller bug (orchestrator's _select_top_k
        didn't honor the policy's k) and would produce misleading
        threshold math if silently tolerated.
        """
        if len(node_ids) != self.consensus.k:
            raise ValueError(
                f"node_ids count {len(node_ids)} does not match "
                f"consensus.k={self.consensus.k}"
            )
        if len(set(node_ids)) != len(node_ids):
            # Same provider twice would give them two votes — always wrong.
            raise ValueError(f"node_ids contain duplicates: {node_ids}")

        # Parallel dispatch. return_exceptions=True so one bad provider
        # doesn't abort the gather; we classify exceptions after.
        coros = [
            self.single_dispatcher.dispatch_with_receipt(
                shard=shard,
                input_tensor=input_tensor,
                node_id=node_id,
                job_id=job_id,
                stake_tier=stake_tier,
                escrow_amount_ftns=escrow_amount_per_provider_ftns,
                require_tee_attestation=require_tee_attestation,
            )
            for node_id in node_ids
        ]
        raw_results = await asyncio.gather(*coros, return_exceptions=True)

        successful: List[DispatchResult] = []
        for node_id, r in zip(node_ids, raw_results):
            if isinstance(r, DispatchResult):
                # Fallback paths (size-too-large, peer-not-connected with
                # local_fallback wired) produce an empty receipt dict per
                # Phase 3.1's contract. We can't consense without a signed
                # receipt from the provider, so skip those — they don't
                # count as agreeing responses.
                if not r.receipt:
                    logger.warning(
                        f"multi-dispatch: provider {node_id[:12]}… returned "
                        f"empty receipt (fallback path); excluding from consensus"
                    )
                    continue
                successful.append(r)
            elif isinstance(r, (
                ShardPreemptedError,
                MissingAttestationError,
                ShardDispatchError,
                PeerNotConnectedError,
            )):
                # Honest-work failures + advertised-tier violations +
                # generic dispatch failures + offline providers: the
                # provider simply didn't contribute. Log and exclude
                # from consensus. PeerNotConnectedError is the Phase 7.1
                # §8.8 audit follow-up — one offline provider in a
                # k-of-n dispatch is a classic partial-response case,
                # not a gather-aborting bug.
                logger.warning(
                    f"multi-dispatch: provider {node_id[:12]}… failed: "
                    f"{type(r).__name__}: {r}"
                )
            else:
                # Unexpected exception type — bubble up. A bug we shouldn't
                # silently paper over with "partial response." (Note:
                # asyncio.CancelledError subclasses BaseException and is
                # NOT caught by `except Exception` inside RemoteShardDispatcher
                # — it correctly propagates here so outer-loop job
                # cancellation cancels the gather rather than being
                # silently swallowed.)
                raise r

        receipts = [
            ShardExecutionReceipt.from_dict(r.receipt) for r in successful
        ]

        # Consensus check.
        outcome: ConsensusOutcome = self.consensus.resolve(receipts)
        responded = len(successful)

        if not outcome.agreed:
            # Distinguish "not enough responses" from "responded but no
            # agreement." The orchestrator's response to each differs:
            # insufficient → re-dispatch with fresh pool; failed →
            # challenge pipeline (every disagreer becomes a minority).
            threshold = self._threshold_for_mode()
            if responded < threshold:
                raise InsufficientResponsesError(
                    responded=responded, k=self.consensus.k,
                    threshold=threshold,
                )
            unique = len({r.output_hash for r in receipts})
            raise ConsensusFailedError(
                responded=responded, k=self.consensus.k,
                unique_hashes=unique,
            )

        # Majority won. Pick the agreed output from any majority provider.
        # By the hash-agreement contract, every majority DispatchResult
        # carries bitwise-identical output bytes.
        agreed_provider_ids = {r.provider_id for r in outcome.majority}
        agreed_output = next(
            r.output for r in successful
            if r.receipt.get("provider_id") in agreed_provider_ids
        )

        return ConsensusShardReceipt(
            job_id=job_id,
            shard_index=shard.shard_index,
            consensus_mode=self.consensus.mode.value,
            k=self.consensus.k,
            responded=responded,
            agreed_output_hash=outcome.agreed_output_hash,
            agreed_output=agreed_output,
            majority_receipts=outcome.majority,
            minority_receipts=outcome.minority,
            consensus_reached_unix=int(time.time()),
        )

    def _threshold_for_mode(self) -> int:
        """Minimum response count at which consensus could in principle
        be reached under the current mode. Used only for classifying
        failure as InsufficientResponses vs ConsensusFailed."""
        from prsm.node.result_consensus import ConsensusMode

        if self.consensus.mode == ConsensusMode.MAJORITY:
            return (self.consensus.k // 2) + 1
        if self.consensus.mode == ConsensusMode.UNANIMOUS:
            return self.consensus.k
        if self.consensus.mode == ConsensusMode.SINGLE:
            return 1
        # BYZANTINE blocked at ShardConsensus.__init__.
        raise RuntimeError(f"unreachable mode: {self.consensus.mode}")
