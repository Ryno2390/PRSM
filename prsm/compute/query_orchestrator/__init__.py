"""QueryOrchestrator — replacement for the deleted Ring 5 Agent Forge.

Orchestrates PRSM's data-query path: decompose query → DSL manifest →
discover shards → fan out WASM agents → select aggregator → combine →
deliver → settle.

See:
- docs/2026-05-07-canonical-workflow-gap-list.md (architecture)
- docs/2026-05-07-canonical-workflow-gap-list-delta.md (next-step ordering)
- docs/2026-05-07-aggregator-selector-threat-model.md (binding design)

Sub-modules (build order):
1. aggregator_selector — selects 1 T2+ stake-pool node per query
2. decomposer         — natural-language query → AgentOp DSL
3. shard_finder       — query → shard CIDs via EmbeddingDHT
4. swarm_runner       — fan-out via AgentDispatcher + AgentExecutor
"""
from prsm.compute.query_orchestrator.aggregator_selector import (
    AggregationCommit,
    AggregationCommitMismatchError,
    InsufficientCandidatesError,
    SelectionInput,
    StakedNode,
    select_aggregator,
    verify_aggregation_commit,
)
from prsm.compute.query_orchestrator.decomposer import (
    DecomposerOutputError,
    QueryDecomposer,
    RuleBasedDecomposer,
    decompose_query,
)
from prsm.compute.query_orchestrator.shard_finder import (
    SemanticIndex,
    ShardCandidate,
    find_relevant_shards,
)

__all__ = [
    # aggregator_selector
    "AggregationCommit",
    "AggregationCommitMismatchError",
    "InsufficientCandidatesError",
    "SelectionInput",
    "StakedNode",
    "select_aggregator",
    "verify_aggregation_commit",
    # decomposer
    "DecomposerOutputError",
    "QueryDecomposer",
    "RuleBasedDecomposer",
    "decompose_query",
    # shard_finder
    "SemanticIndex",
    "ShardCandidate",
    "find_relevant_shards",
]
