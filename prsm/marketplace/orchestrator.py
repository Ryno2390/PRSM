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
from typing import Any, Dict, List, Optional

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

# Phase 7 Task 5 — on-chain tier gating. Ordering mirrors
# EligibilityFilter._TIER_ORDER — kept in lock-step because we reuse it
# to compare the on-chain effectiveTier string against policy.min_stake_tier.
_TIER_ORDER = {"open": 0, "standard": 1, "premium": 2, "critical": 3}

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
        # Phase 3.1 (Task 7) — optional batched-settlement integration.
        # When both are wired, after each successful remote dispatch the
        # orchestrator forwards a BatchedReceipt to the settlement client.
        # When either is None, the orchestrator behaves as Phase 3 did
        # (local-ledger-only settlement via the dispatcher's escrow flow).
        batch_settlement_client=None,
        provider_address_resolver=None,
        # Phase 7 Task 5 — on-chain tier gating. When wired, the
        # orchestrator verifies StakeBond.effectiveTier(provider) meets
        # policy.min_stake_tier before dispatching. Closes the gap where
        # a provider claims a tier in their listing but hasn't actually
        # bonded stake to back it. When None, gating is skipped and the
        # orchestrator behaves as Phase 3.1 did (listing-claim-only tier).
        stake_manager_client=None,
    ):
        self.identity = identity
        self.directory = directory
        self.eligibility_filter = eligibility_filter
        self.reputation = reputation
        self.price_negotiator = price_negotiator
        self.remote_dispatcher = remote_dispatcher
        self.batch_settlement_client = batch_settlement_client
        self.provider_address_resolver = provider_address_resolver
        self.stake_manager_client = stake_manager_client

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
        last_reason: Optional[str] = None
        for listing in eligible:

            # Phase 7 Task 5: on-chain tier gate. A listing-claimed tier
            # can be a lie; StakeBond.effectiveTier is the ground truth.
            # Check BEFORE the price handshake so a cheating provider
            # doesn't get to occupy a quote slot.
            if not self._onchain_tier_meets(listing, policy):
                self.reputation.record_failure(listing.provider_id)
                last_reason = "tier_below_required_onchain"
                continue

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
            # Use dispatch_with_receipt when Phase 3.1 batched settlement
            # is wired so we can forward the verified receipt to the
            # settlement accumulator; otherwise use the Phase 2 dispatch
            # API that returns just the output.
            started = time.time()
            use_batched_settlement = (
                self.batch_settlement_client is not None
                and self.provider_address_resolver is not None
                and getattr(self.identity, "ethereum_address", None) is not None
            )
            try:
                if use_batched_settlement:
                    result = await self.remote_dispatcher.dispatch_with_receipt(
                        shard=shard,
                        input_tensor=input_tensor,
                        node_id=listing.provider_id,
                        job_id=job_id,
                        stake_tier=_stake_tier_for(listing.stake_tier),
                        escrow_amount_ftns=quote.quoted_price_ftns,
                        require_tee_attestation=policy.require_tee,
                    )
                    output = result.output
                    receipt = result.receipt
                else:
                    output = await self.remote_dispatcher.dispatch(
                        shard=shard,
                        input_tensor=input_tensor,
                        node_id=listing.provider_id,
                        job_id=job_id,
                        stake_tier=_stake_tier_for(listing.stake_tier),
                        escrow_amount_ftns=quote.quoted_price_ftns,
                        require_tee_attestation=policy.require_tee,
                    )
                    receipt = None
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

            # Phase 3.1 (Task 7): accumulate the verified receipt for
            # batched on-chain settlement, when wired. Silently skip if
            # receipt is empty (fallback path) or addresses aren't
            # resolvable — we never want to block Phase 3 success on an
            # optional settlement failure.
            if use_batched_settlement and receipt:
                await self._accumulate_settlement_receipt(
                    listing=listing,
                    job_id=job_id,
                    shard_index=shard.shard_index,
                    receipt=receipt,
                    quoted_price_ftns=quote.quoted_price_ftns,
                )

            return output

        # Exhausted the eligible pool without a success.
        raise NoEligibleProvidersError(
            f"shard {shard.shard_index} of job {job_id!r} exhausted the "
            f"eligible pool (size={len(eligible)}) without success; "
            f"last failure reason: {last_reason or 'unknown'}"
        )

    def _onchain_tier_meets(
        self,
        listing: ProviderListing,
        policy: DispatchPolicy,
    ) -> bool:
        """Phase 7 Task 5 — Return True iff the listing should be allowed
        to serve the job under the on-chain tier policy.

        Three short-circuits preserve the additive integration pattern:
          - If the policy's min_stake_tier is "open" (the default), any
            on-chain tier is acceptable; skip the RPC.
          - If stake_manager_client OR provider_address_resolver is None,
            the orchestrator is not wired for Phase 7 — fall back to the
            listing-claim tier (already enforced by EligibilityFilter).
          - If the resolver can't map the provider to an Ethereum address,
            skip the check (providers without chain identity predate
            Phase 7; the listing filter has already approved them).

        If the RPC itself raises, we fail OPEN with a loud warning.
        Closed-fail here would let an attacker DoS the StakeBond RPC to
        block all dispatch — and the listing filter plus slashing-on-
        misbehavior still protect against the cheat case. Operators
        watching the warning log can harden to fail-closed later.
        """
        if policy.min_stake_tier == "open":
            return True
        if self.stake_manager_client is None or self.provider_address_resolver is None:
            return True

        provider_eth = self.provider_address_resolver(listing.provider_id)
        if provider_eth is None:
            return True

        try:
            onchain_tier = self.stake_manager_client.effective_tier(
                provider_eth
            )
        except Exception as exc:
            logger.warning(
                f"on-chain tier check failed for provider "
                f"{listing.provider_id[:12]}…: "
                f"{type(exc).__name__}: {exc}; failing open"
            )
            return True

        required_ord = _TIER_ORDER.get(policy.min_stake_tier, -1)
        actual_ord = _TIER_ORDER.get(onchain_tier, -1)
        if actual_ord < required_ord:
            logger.info(
                f"provider {listing.provider_id[:12]}… rejected: on-chain "
                f"tier {onchain_tier!r} below required {policy.min_stake_tier!r} "
                f"(listing claimed {listing.stake_tier!r})"
            )
            return False
        return True

    async def _accumulate_settlement_receipt(
        self,
        listing: ProviderListing,
        job_id: str,
        shard_index: int,
        receipt: Dict[str, Any],
        quoted_price_ftns: float,
    ) -> None:
        """Phase 3.1 (Task 7) — build a BatchedReceipt and forward to
        the settlement accumulator. Lazy imports so callers without
        Phase 3.1 wiring don't pay the import cost.

        Never raises: if the provider address resolver returns None, or
        the settlement client's accumulate fails, we log and swallow.
        Phase 3.1 batched settlement is strictly additive; its failures
        must not block Phase 3 dispatch success."""
        try:
            from prsm.compute.shard_receipt import ShardExecutionReceipt
            from prsm.settlement.accumulator import BatchedReceipt

            provider_eth = self.provider_address_resolver(listing.provider_id)
            if provider_eth is None:
                logger.debug(
                    f"batched settlement skipped for provider "
                    f"{listing.provider_id[:12]}…: no Ethereum address resolved"
                )
                return

            # Reconstruct a ShardExecutionReceipt dataclass from the
            # verified receipt dict so the settlement layer's canonical-
            # form encoding (Task 5) can reference typed fields.
            shard_receipt = ShardExecutionReceipt(
                job_id=receipt.get("job_id", ""),
                shard_index=receipt.get("shard_index", shard_index),
                provider_id=receipt.get("provider_id", ""),
                provider_pubkey_b64=receipt.get("provider_pubkey_b64", ""),
                output_hash=receipt.get("output_hash", ""),
                executed_at_unix=receipt.get("executed_at_unix", 0),
                signature=receipt.get("signature", ""),
                tee_attestation=receipt.get("tee_attestation"),
            )

            # Value field is in FTNS base units (wei, 18 decimals). The
            # orchestrator receives the quoted price as a float FTNS
            # value per Phase 3's price-handshake semantics.
            value_ftns_wei = int(round(quoted_price_ftns * 10**18))

            br = BatchedReceipt(
                receipt=shard_receipt,
                requester_address=self.identity.ethereum_address,
                provider_address=provider_eth,
                value_ftns=value_ftns_wei,
                local_escrow_id=f"{job_id}:shard:{shard_index}",
            )
            await self.batch_settlement_client.accumulate(br)
        except Exception as exc:
            logger.warning(
                f"batched settlement accumulate failed for shard "
                f"{shard_index} of job {job_id!r}: "
                f"{type(exc).__name__}: {exc}"
            )
