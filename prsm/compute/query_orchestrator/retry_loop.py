"""QueryOrchestrator — retry-loop shell (A4 closure).

Wraps `QueryOrchestrator.dispatch_query` with the threat-model A4
mitigations:

  - **Mitigation 1**: bounded selection retries
    (`MAX_AGG_RETRIES = 3` per threat-model `§"Open governance
    questions"` item 3 — Foundation-council ratifiable).

  - **Mitigation 2**: `record_preemption(provider_id)` on dropped
    aggregator. Source-agent preemption is treated as a separate
    signal — see `PartialResultIntegrityError` handling below.

  - **Mitigation 3**: escrow refund when retries exhaust without a
    responsive aggregator (or when no candidate exists at all).

What this module DOES:
  - Distinguishes retryable from non-retryable exceptions.
  - On retryable failures, routes the right side-effect (slash for
    A9, preemption for A5) and re-attempts up to `policy.max_retries`.
  - On exhaustion, calls `escrow_client.refund(query_id, prompter)`
    and raises `QueryRetriesExhaustedError`.

What this module does NOT do:
  - Aggregator selection itself — `aggregator_selector` owns that.
    The retry loop just calls `dispatch_query` again, which re-runs
    the selector.
  - The redundancy `p_check` sample (A1 mitigation 3) — that's a
    separate orchestrator-level concern that wraps THIS module's
    output, not the other way around. R&D follow-on.

Pluggable side-effect Protocols so the orchestrator's wiring can
inject real slash/preempt/escrow clients vs. test stubs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from prsm.compute.query_orchestrator.aggregator_selector import (
    AggregationCommitMismatchError,
    InsufficientCandidatesError,
)
from prsm.compute.query_orchestrator.swarm_runner import (
    AggregatedResult,
    PartialResultIntegrityError,
)

logger = logging.getLogger(__name__)


class QueryRetriesExhaustedError(RuntimeError):
    """All retry attempts failed; escrow refunded.

    Distinct from the per-attempt typed exceptions because the response
    is different: the orchestrator's caller surfaces this to the
    prompter as a "swarm failure, refund issued" signal.
    """


@dataclass(frozen=True)
class RetryPolicy:
    """A4 mitigation 1 parameter. Default 3 per threat-model
    §"Open governance questions" — Foundation-council ratification
    target.
    """
    max_retries: int = 3


@runtime_checkable
class SlashClient(Protocol):
    """A9 mitigation: when the aggregator's commit doesn't match the
    plaintext, the orchestrator routes a slash event into the
    StakeBond contract via this client.

    Production wiring: an adapter calling
    `prsm/economy/web3/stake_manager.py::StakeManager.slash` AND
    `prsm/marketplace/reputation.py::ReputationTracker.record_slash`."""

    def slash_aggregator(
        self, pubkey_hash: bytes, batch_id: str, reason: str
    ) -> None: ...


@runtime_checkable
class PreemptionRecorder(Protocol):
    """A4 mitigation 2: source-agent preemption signal. When a
    partial fails A5 DP-noise enforcement (i.e., the source agent
    sent un-noised data), record preemption against that agent's
    provider_id so future selections discount it.

    Production wiring: an adapter calling
    `prsm/marketplace/reputation.py:133::ReputationTracker.record_preemption`."""

    def record_preemption(self, provider_id: str) -> None: ...


@runtime_checkable
class EscrowClient(Protocol):
    """A4 mitigation 3: refund the prompter's escrow on no-aggregator
    or retries-exhausted. Per-shard agent costs are NOT refunded —
    those services were rendered, even if the aggregation failed.

    Production wiring: an adapter calling the escrow contract's
    `refund` path with the query_id as the lookup key."""

    async def refund(self, query_id: bytes, prompter: str) -> Any: ...


# ──────────────────────────────────────────────────────────────────────
# Top-level retry loop
# ──────────────────────────────────────────────────────────────────────


async def dispatch_with_retries(
    *,
    orchestrator: Any,  # QueryOrchestrator (duck-typed for testability)
    query: str,
    prompter_node_id: str,
    query_id: bytes,
    requires_tee: bool = False,
    governance_denylist: frozenset[bytes] = frozenset(),
    policy: RetryPolicy = RetryPolicy(),
    slash_client: SlashClient,
    preemption_recorder: PreemptionRecorder,
    escrow_client: EscrowClient,
) -> AggregatedResult:
    """Run a query with bounded retries + slash/preempt routing.

    Retry policy:
      - `AggregationCommitMismatchError`: slash aggregator + retry
      - `PartialResultIntegrityError`: record preemption + retry
      - `InsufficientCandidatesError`: refund escrow + propagate
        (no point retrying when there's no candidate)
      - `ValueError`: programming error, no retry, no escrow refund —
        the caller decides what to do
      - any other exception: no retry, no escrow refund — fail-loud

    On `policy.max_retries` exhaustion: refund escrow, raise
    `QueryRetriesExhaustedError`. Initial attempt counts AGAINST the
    retry budget — `max_retries=3` means up to 4 total calls.

    Note on call accounting: "max_retries" names the number of RETRY
    attempts after the initial one. Total calls = 1 + max_retries.
    """
    last_exc: Exception | None = None
    for attempt in range(policy.max_retries + 1):
        try:
            return await orchestrator.dispatch_query(
                query=query,
                prompter_node_id=prompter_node_id,
                query_id=query_id,
                requires_tee=requires_tee,
                governance_denylist=governance_denylist,
            )
        except AggregationCommitMismatchError as exc:
            # A9 violation. The aggregator is the entity to slash —
            # it committed to a digest it didn't deliver.
            last_exc = exc
            # NOTE: pubkey_hash + batch_id are not on the exception
            # by design (the swarm_runner's exception is a typed
            # signal, not a forensic record). Production wiring
            # threads these through via the AggregatedResult that
            # would have been returned. For v1 we slash by the
            # aggregator-id the orchestrator's last selection
            # produced — wired into a separate slash-context object
            # in the production wiring task. Here we surface the
            # exception's repr to the slash client so the audit
            # trail captures something.
            slash_client.slash_aggregator(
                pubkey_hash=b"\x00" * 32,  # production wiring overrides
                batch_id=query_id.hex(),
                reason="A9_COMMIT_MISMATCH",
            )
            logger.info(
                "A9 commit-mismatch on attempt %d/%d for query_id=%s; "
                "slashing aggregator + retrying",
                attempt + 1, policy.max_retries + 1, query_id.hex()[:16],
            )
        except PartialResultIntegrityError as exc:
            # A5 violation. The source agent (NOT the aggregator) is
            # the entity to preempt — they sent un-noised data.
            last_exc = exc
            # Same caveat as above: production wiring threads the
            # offending agent's provider_id via a richer signal. v1
            # surfaces the query_id as the audit anchor.
            preemption_recorder.record_preemption(
                provider_id=f"unknown-source-agent-for-query-{query_id.hex()[:16]}",
            )
            logger.info(
                "A5 partial-integrity on attempt %d/%d for query_id=%s; "
                "recording preemption + retrying",
                attempt + 1, policy.max_retries + 1, query_id.hex()[:16],
            )
        except InsufficientCandidatesError:
            # Refund first so the prompter's budget isn't held
            # against an impossible selection. Then propagate the
            # original exception so the caller sees the real reason.
            await escrow_client.refund(query_id, prompter_node_id)
            raise

    # Retries exhausted.
    await escrow_client.refund(query_id, prompter_node_id)
    raise QueryRetriesExhaustedError(
        f"{policy.max_retries + 1} attempts exhausted for query_id="
        f"{query_id.hex()[:16]}... (last error: {last_exc!r})"
    )
