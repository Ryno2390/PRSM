"""Phase 3 Task 7: MarketplaceOrchestrator.

The integration layer. One entrypoint `orchestrate_sharded_inference`
composes:

  directory (Task 2) → filter (Task 4) → randomizer (Phase 2.1 Line B)
  → price-handshake (Task 5 server + Task 7 client)
  → RemoteShardDispatcher (Phase 2)
  → ReceiptOnlyVerification + optional TEE attestation check (Phase 2.1 Line C)
  → ReputationTracker (Task 6)

Per-shard dispatch loop: try providers from the eligible pool one at a
time, moving on after quote rejection / preemption. A provider that
raises ShardDispatchError (malicious / unrecoverable failure) is
recorded as a failure and the error propagates — that's the caller's
signal to back off entirely. Preempted and quote-rejected providers
are retried on different nodes.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

import numpy as np

from prsm.compute.model_sharding.models import (
    ModelShard,
    PipelineStakeTier,
)
from prsm.compute.remote_dispatcher import (
    MissingAttestationError,
    ShardDispatchError,
    ShardPreemptedError,
)
from prsm.marketplace.directory import MarketplaceDirectory
from prsm.marketplace.errors import NoEligibleProvidersError
from prsm.marketplace.filter import EligibilityFilter
from prsm.marketplace.listing import ProviderListing
from prsm.marketplace.policy import DispatchPolicy
from prsm.marketplace.price_handshake import PriceNegotiator, PriceQuote
from prsm.marketplace.reputation import ReputationTracker

logger = logging.getLogger(__name__)


def _stake_tier_for(label: str) -> PipelineStakeTier:
    """Map a listing's stake_tier string to the PipelineStakeTier enum
    RemoteShardDispatcher expects."""
    for t in PipelineStakeTier:
        if t.label == label:
            return t
    return PipelineStakeTier.STANDARD


class MarketplaceOrchestrator:
    """Ties the full Phase 3 marketplace flow together.

    Constructor wires all dependencies; the single public entrypoint
    `orchestrate_sharded_inference` runs one inference end-to-end.
    """

    def __init__(
        self,
        identity,
        directory: MarketplaceDirectory,
        eligibility_filter: EligibilityFilter,
        reputation: ReputationTracker,
        price_negotiator: PriceNegotiator,
        remote_dispatcher,
    ):
        self.identity = identity
        self.directory = directory
        self.eligibility_filter = eligibility_filter
        self.reputation = reputation
        self.price_negotiator = price_negotiator
        self.remote_dispatcher = remote_dispatcher

    async def orchestrate_sharded_inference(
        self,
        shards: List[ModelShard],
        input_tensor: np.ndarray,
        job_id: str,
        policy: DispatchPolicy,
    ) -> np.ndarray:
        """Run one sharded inference end-to-end across the marketplace.

        Returns the concatenated per-shard outputs (axis 0 — matches
        the Phase 2 integration test's row-parallel assembly).
        """
        listings = self.directory.list_active_providers()
        eligible = self.eligibility_filter.filter(listings, policy)
        if not eligible:
            raise NoEligibleProvidersError(
                f"no eligible providers for job {job_id!r}: "
                f"directory has {len(listings)} listings, filter excluded all"
            )

        outputs: List[np.ndarray] = []
        for shard in shards:
            output = await self._dispatch_one_shard(
                shard=shard,
                input_tensor=input_tensor,
                job_id=job_id,
                eligible=eligible,
                policy=policy,
            )
            outputs.append(output)

        return np.concatenate(outputs, axis=0)

    async def _dispatch_one_shard(
        self,
        shard: ModelShard,
        input_tensor: np.ndarray,
        job_id: str,
        eligible: List[ProviderListing],
        policy: DispatchPolicy,
    ) -> np.ndarray:
        """Try providers from the eligible pool one at a time until one
        succeeds. Raises NoEligibleProvidersError if the pool is
        exhausted; raises ShardDispatchError as-is for malicious
        failures (caller decides whether to abort the job)."""
        attempted: set = set()
        remaining = [l for l in eligible if l.provider_id not in attempted]

        last_reason: Optional[str] = None
        for listing in remaining:
            if listing.provider_id in attempted:
                continue
            attempted.add(listing.provider_id)

            # Step 1: price handshake.
            quote = await self.price_negotiator.request_quote(
                listing=listing,
                shard_index=shard.shard_index,
                shard_size_bytes=len(shard.tensor_data),
                max_acceptable_price_ftns=policy.max_price_per_shard_ftns,
            )
            if quote is None:
                last_reason = "quote_timeout"
                continue
            if not isinstance(quote, PriceQuote):
                # PriceQuoteRejected — try next provider.
                last_reason = f"quote_rejected:{quote.reason}"
                continue

            # Step 2: dispatch the shard under the quoted price.
            started = time.time()
            try:
                output = await self.remote_dispatcher.dispatch(
                    shard=shard,
                    input_tensor=input_tensor,
                    node_id=listing.provider_id,
                    job_id=job_id,
                    stake_tier=_stake_tier_for(listing.stake_tier),
                    escrow_amount_ftns=quote.quoted_price_ftns,
                    require_tee_attestation=policy.require_tee,
                )
            except ShardPreemptedError:
                # Honest-work failure — no reputation penalty.
                self.reputation.record_preemption(listing.provider_id)
                last_reason = "preempted"
                continue
            except MissingAttestationError:
                # Provider lied about TEE support in the listing.
                # Treat as a failure (reputation penalty) so the
                # liar gets filtered out on subsequent inferences.
                self.reputation.record_failure(listing.provider_id)
                last_reason = "missing_attestation"
                continue
            except ShardDispatchError as exc:
                self.reputation.record_failure(listing.provider_id)
                raise

            # Success.
            latency_ms = (time.time() - started) * 1000
            self.reputation.record_success(listing.provider_id, latency_ms)
            return output

        # Exhausted the eligible pool without a success.
        raise NoEligibleProvidersError(
            f"shard {shard.shard_index} of job {job_id!r} exhausted the "
            f"eligible pool (size={len(eligible)}) without success; "
            f"last failure reason: {last_reason or 'unknown'}"
        )
