"""QueryOrchestrator — swarm runner.

Closes the data-query path. Composes the three earlier sub-modules:

    decompose_query  →  manifest
    find_relevant_shards  →  shards
    select_aggregator  →  aggregator
    run_swarm(manifest, shards, aggregator, ...)  →  AggregatedResult

What this module owns:
  - Per-shard agent fan-out via a pluggable `SwarmDispatcher` Protocol.
    Production wiring constructs an adapter around
    `prsm/compute/agents/dispatcher.py::AgentDispatcher`.

  - **A5 enforcement**: the threat model mandates per-shard partials
    be DP-noised by the source agent BEFORE reaching the aggregator
    (so the aggregator never sees raw shard data). The runner
    verifies the `dp_noise_applied` marker on every partial — if any
    is missing, it refuses to forward to the aggregator.

  - Aggregator handoff via a pluggable `AggregatorClient` Protocol.

  - **A9 enforcement**: verify the aggregator's pre-commit hash
    matches the plaintext they delivered. Mismatch surfaces as
    `AggregationCommitMismatchError` — the orchestrator catches it
    and routes to slash via `ReputationTracker.record_slash`.

What this module does NOT own:
  - Aggregator selection: see `aggregator_selector.py`
  - Shard discovery: see `shard_finder.py`
  - Decomposition: see `decomposer.py`
  - Settlement: see `prsm/settlement/`
  - Escrow refund on failure (A4 mitigation 3): orchestrator's retry
    loop owns this — it catches the runner's exceptions and decides
    whether to re-poll the selector or refund the prompter.

Threat-model coverage in this module: **A5** (DP noise) and **A9**
(commit-verify). A1/A2/A6/A7/A8/A10 land at `aggregator_selector.py`;
A3/A4 land at the orchestrator's retry-loop shell.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from prsm.compute.agents.instruction_set import InstructionManifest
from prsm.compute.query_orchestrator.aggregator_selector import (
    AggregationCommit,
    AggregationCommitMismatchError,
    StakedNode,
    verify_aggregation_commit,
)
from prsm.compute.query_orchestrator.shard_finder import ShardCandidate


class PartialResultIntegrityError(RuntimeError):
    """A partial result failed integrity gate — typically the
    `dp_noise_applied` marker is missing (A5 mitigation 2). Distinct
    from AggregationCommitMismatchError because the response is
    different — refuse-and-retry-with-different-agent vs.
    refuse-and-slash-aggregator.
    """


@dataclass(frozen=True)
class PartialResult:
    """Per-shard agent output, addressed to the aggregator.

    Attributes
    ----------
    shard_cid:
        The CID this partial was computed against.
    payload:
        DP-noised partial. The aggregator combines these without
        seeing raw shard data (A5).
    agent_signature:
        Ed25519 signature by the executing agent — forensic
        traceability if the aggregator later substitutes results.
    creator_id:
        Original publisher identifier — flows to RoyaltyDistributor
        at settlement.
    dp_noise_applied:
        Marker that the source agent applied `dp_noise.py` primitives
        before signing. The runner refuses to forward un-noised
        partials to the aggregator (A5 mitigation 2 enforcement).
    """
    shard_cid: str
    payload: bytes
    agent_signature: bytes
    creator_id: str
    dp_noise_applied: bool


@dataclass(frozen=True)
class AggregatedResult:
    """The final swarm output, ready for the prompter.

    Attributes
    ----------
    query_id:
        Echoes the input — binds this result to a specific query.
    payload:
        The aggregator's plaintext combined output.
    aggregator_node_id:
        The selected aggregator's `node_id`. Settlement routes the
        aggregator's compensation against this.
    contributing_shards:
        Tuple of shard CIDs whose partials fed the aggregation.
        Settlement uses these for per-creator royalty distribution.
    """
    query_id: bytes
    payload: bytes
    aggregator_node_id: str
    contributing_shards: tuple[str, ...] = field(default_factory=tuple)


@runtime_checkable
class SwarmDispatcher(Protocol):
    """Pluggable fan-out contract.

    Production wiring: an adapter around
    `prsm/compute/agents/dispatcher.py::AgentDispatcher` that takes
    a manifest + shard list and returns the resulting partials.

    The dispatcher MUST apply DP noise on each partial via
    `prsm/compute/tee/dp_noise.py` and set `dp_noise_applied=True` —
    the runner verifies this marker and rejects un-noised partials.
    """

    async def fan_out(
        self,
        manifest: InstructionManifest,
        shards: Sequence[ShardCandidate],
    ) -> list[PartialResult]: ...


@runtime_checkable
class AggregatorClient(Protocol):
    """Pluggable aggregator-handoff contract.

    The runner calls `aggregate(...)` once with the selected
    aggregator and the (DP-noised) partials. The aggregator returns
    `(plaintext_result, AggregationCommit)` — the commit binds it to
    the result's hash + the query id (A9). The runner verifies the
    commit before returning.
    """

    async def aggregate(
        self,
        aggregator: StakedNode,
        manifest: InstructionManifest,
        partials: Sequence[PartialResult],
        query_id: bytes,
    ) -> tuple[bytes, AggregationCommit]: ...


# ──────────────────────────────────────────────────────────────────────
# Top-level runner
# ──────────────────────────────────────────────────────────────────────


def _enforce_a5_dp_noise(partials: Sequence[PartialResult]) -> None:
    """Threat-model A5 mitigation 2 enforcement.

    Refuse-and-raise on any partial whose `dp_noise_applied` marker is
    not True. The orchestrator catches the exception and decides
    whether to re-dispatch with a different agent or surface to the
    prompter as a swarm failure.
    """
    for p in partials:
        if not p.dp_noise_applied:
            raise PartialResultIntegrityError(
                f"partial for shard {p.shard_cid!r} arrived without "
                f"dp_noise_applied=True — refusing to forward to aggregator "
                f"per threat-model A5 mitigation 2"
            )


async def run_swarm(
    *,
    manifest: InstructionManifest,
    shards: Sequence[ShardCandidate],
    aggregator: StakedNode,
    dispatcher: SwarmDispatcher,
    aggregator_client: AggregatorClient,
    query_id: bytes,
) -> AggregatedResult:
    """Execute the full per-query swarm round-trip.

    Steps:
      1. Validate inputs.
      2. Fan out per-shard agents via `dispatcher.fan_out`.
      3. Verify A5 DP-noise marker on every partial.
      4. Hand partials to selected aggregator via
         `aggregator_client.aggregate`.
      5. Verify A9 commit-vs-plaintext binding.
      6. Return `AggregatedResult`.

    Raises:
        ValueError: empty shard list, malformed query_id, or empty
            partials returned from dispatcher.
        PartialResultIntegrityError: a partial lacks DP-noise marker.
        AggregationCommitMismatchError: aggregator's commit does not
            match the plaintext they delivered, OR the commit's
            query_id does not match the call's.
    """
    if not shards:
        raise ValueError("run_swarm requires non-empty shards")
    if len(query_id) != 32:
        raise ValueError(
            f"query_id must be 32 bytes, got {len(query_id)} "
            f"(commits + signing payloads bind on this width — A9)"
        )

    partials = await dispatcher.fan_out(manifest, shards)
    if not partials:
        raise ValueError(
            "dispatcher returned no partials — "
            "cannot run aggregation on empty input"
        )

    # A5 enforcement: every partial must be DP-noised before reaching
    # the aggregator.
    _enforce_a5_dp_noise(partials)

    plaintext, commit = await aggregator_client.aggregate(
        aggregator, manifest, partials, query_id
    )

    # A9 enforcement: commit binds to the call's query_id AND the
    # plaintext digest. Substitute either and the verification fails.
    if commit.query_id != query_id:
        raise AggregationCommitMismatchError(
            f"aggregator {aggregator.node_id} returned commit for "
            f"query_id {commit.query_id.hex()[:16]}... but call was for "
            f"{query_id.hex()[:16]}... — A9 forensic-traceability violation"
        )
    verify_aggregation_commit(commit, plaintext)

    return AggregatedResult(
        query_id=query_id,
        payload=plaintext,
        aggregator_node_id=aggregator.node_id,
        contributing_shards=tuple(p.shard_cid for p in partials),
    )
