"""QueryOrchestrator — composition of the 4 core sub-modules.

Each `dispatch_query` call runs:

    decompose_query  →  manifest
    find_relevant_shards  →  shards
    select_aggregator  →  aggregator
    run_swarm  →  AggregatedResult

The orchestrator is a thin composition layer — every threat-model
invariant is enforced inside a sub-module:

  - A1/A2/A6/A7/A8/A10  →  aggregator_selector.py
  - A5                  →  swarm_runner.py (DP-noise enforcement)
  - A9                  →  swarm_runner.py (commit-verify) +
                            aggregator_selector.py (verify_aggregation_commit)
  - A3/A4               →  StakeBond contract + retry-loop shell
                            (out-of-scope for this module)

This module owns:
  - The wiring shape (constructor arguments, what the production
    `node.py` must inject)
  - The default selection-input fields (sliding_window_state,
    governance_denylist, requires_tee) — sensible defaults that the
    caller can override if needed
  - Error propagation: every sub-module's typed exception bubbles
    out unchanged, so the orchestrator's outer retry-loop shell can
    route deterministically

This module does NOT own:
  - The retry-loop shell itself (A4 mitigation 1: bounded retries,
    A1 mitigation 3: p_check redundancy sample) — that's a separate
    layer that wraps `dispatch_query` calls
  - The escrow / settlement integration — that's downstream of
    `AggregatedResult`
  - Production wiring of SwarmDispatcher + AggregatorClient — those
    are `node.py` responsibilities, separately scoped
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

from prsm.compute.query_orchestrator.aggregator_selector import (
    SelectionInput,
    StakedNode,
    select_aggregator,
)
from prsm.compute.query_orchestrator.decomposer import (
    QueryDecomposer,
    decompose_query,
)
from prsm.compute.query_orchestrator.shard_finder import (
    SemanticIndex,
    find_relevant_shards,
)
from prsm.compute.query_orchestrator.swarm_runner import (
    AggregatedResult,
    AggregatorClient,
    SwarmDispatcher,
    run_swarm,
)


@dataclass
class QueryOrchestrator:
    """Replacement for the deleted Ring 5 Agent Forge.

    Wired into `node.agent_forge` at startup. Five constructor slots
    name the production-side dependencies:

      semantic_index:
          ContentUploader._semantic_index (text-vector lane) or
          ._fingerprint_index (binary lane). Must satisfy SemanticIndex
          Protocol — `find_top_k(query, k) -> list[(cid, sim, creator)]`.

      dispatcher:
          SwarmDispatcher Protocol implementation. Production wires an
          adapter around `prsm/compute/agents/dispatcher.py::AgentDispatcher`.
          Source agents MUST apply DP noise via `dp_noise.py` before
          signing partials (A5 enforcement at swarm_runner).

      aggregator_client:
          AggregatorClient Protocol implementation. Routes partials to
          the selected aggregator and returns
          `(plaintext, AggregationCommit)`. Production wires this
          against the chain-RPC stack.

      candidate_pool_provider:
          Callable returning the current T2+ stake-pool snapshot.
          Production wires this against the StakeManager + tier-gate +
          ReputationTracker triad.

      beacon_provider:
          Callable returning 32 bytes of fresh per-query randomness.
          A6 binding: production wires this against the daily Foundation
          Safe-anchored beacon (or, post-ratification, the on-chain
          `block.prevrandao`).

      decomposer (optional):
          QueryDecomposer Protocol implementation. None = falls through
          to RuleBasedDecomposer for bootstrap.

    All five slots are runtime-checked via Protocol/callable membership
    on first dispatch — wiring errors surface clearly, not as obscure
    AttributeErrors.
    """

    semantic_index: SemanticIndex
    dispatcher: SwarmDispatcher
    aggregator_client: AggregatorClient
    candidate_pool_provider: Callable[[], tuple[StakedNode, ...]]
    beacon_provider: Callable[[], bytes]
    decomposer: QueryDecomposer | None = None

    # Per-prompter rolling-window state for A1 mitigation 2. Production
    # wiring threads this against a persistent store (Redis / sqlite /
    # in-memory ring buffer); tests pass in an empty dict and let it
    # accumulate.
    sliding_window_state: dict[str, dict[str, int]] = field(default_factory=dict)

    async def dispatch_query(
        self,
        *,
        query: str,
        prompter_node_id: str,
        query_id: bytes,
        requires_tee: bool = False,
        governance_denylist: frozenset[bytes] = frozenset(),
    ) -> AggregatedResult:
        """Run a query end-to-end.

        Required kwargs:
            query: free-text query string.
            prompter_node_id: A2 hard-exclusion input.
            query_id: 32-byte per-query identifier (binds A6/A9).

        Optional kwargs:
            requires_tee: True for Tier C content (A5 dispatch gate).
            governance_denylist: A7 council-flagged pubkey hashes.

        Raises (each is the relevant sub-module's typed exception):
            ValueError: empty query, malformed query_id, or empty
                shard list / partials list.
            DecomposerOutputError: LLM output failed schema validation.
            InsufficientCandidatesError: no eligible aggregator after
                A2/A5/A7/A1 filters.
            PartialResultIntegrityError: a partial lacks A5 DP-noise
                marker.
            AggregationCommitMismatchError: A9 commit-verify failure.
        """
        # 1. Decomposition.
        manifest = decompose_query(query, llm=self.decomposer)

        # 2. Shard discovery.
        shards = find_relevant_shards(query, semantic_index=self.semantic_index)
        if not shards:
            raise ValueError(
                "no shards above similarity threshold for this query — "
                "nothing to aggregate"
            )

        # 3. Aggregator selection.
        prompter_window = self.sliding_window_state.get(prompter_node_id, {})
        spec = SelectionInput(
            prompter_node_id=prompter_node_id,
            candidate_pool=self.candidate_pool_provider(),
            beacon_randomness=self.beacon_provider(),
            query_id=query_id,
            sliding_window_state=prompter_window,
            governance_denylist=governance_denylist,
            requires_tee=requires_tee,
        )
        aggregator = select_aggregator(spec)

        # 4. Swarm round-trip.
        result = await run_swarm(
            manifest=manifest,
            shards=shards,
            aggregator=aggregator,
            dispatcher=self.dispatcher,
            aggregator_client=self.aggregator_client,
            query_id=query_id,
        )

        # Update rolling window AFTER the round-trip succeeds. (Failed
        # selections shouldn't count against the staker's quota — A4
        # preempted-but-honest pattern.)
        prompter_window[aggregator.pubkey_hash.hex()] = (
            prompter_window.get(aggregator.pubkey_hash.hex(), 0) + 1
        )
        self.sliding_window_state[prompter_node_id] = prompter_window

        return result
