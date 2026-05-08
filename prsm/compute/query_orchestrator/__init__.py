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
from prsm.compute.query_orchestrator.swarm_runner import (
    AggregatedResult,
    AggregatorClient,
    PartialResult,
    PartialResultIntegrityError,
    SwarmDispatcher,
    run_swarm,
)
from prsm.compute.query_orchestrator.orchestrator import QueryOrchestrator
from prsm.compute.query_orchestrator.retry_loop import (
    EscrowClient,
    PreemptionRecorder,
    QueryRetriesExhaustedError,
    RetryPolicy,
    SlashClient,
    dispatch_with_retries,
)
from prsm.compute.query_orchestrator.semantic_index_adapter import (
    Embedder,
    SemanticIndexAdapter,
)
from prsm.compute.query_orchestrator.sentence_transformer_embedder import (
    DEFAULT_MODEL_NAME,
    SentenceTransformerEmbedder,
)
from prsm.compute.query_orchestrator.swarm_dispatcher_adapter import (
    SwarmDispatcherAdapter,
)
from prsm.compute.query_orchestrator.aggregator_client_adapter import (
    AggregateTransport,
    AggregatorClientAdapter,
)
from prsm.compute.query_orchestrator.http_aggregate_transport import (
    AggregateTransportError,
    HttpAggregateTransport,
)
from prsm.compute.query_orchestrator.aggregate_protocol import (
    AggregateRequest,
    AggregateResponse,
    SignedPartial,
)
from prsm.compute.query_orchestrator.aggregate_server import (
    AggregateServer,
    AggregateServerError,
    PrivacyBudgetExhaustedError,
    UnsupportedAgentOpError,
    combine_partials,
    enforce_a5_marker,
    sum_privacy_budgets,
    verify_partial_signature,
)
from prsm.compute.query_orchestrator.marketplace_candidate_pool_provider import (
    DEFAULT_STAKE_PER_TIER,
    MarketplaceCandidatePoolProvider,
)
from prsm.compute.query_orchestrator.foundation_beacon_provider import (
    FoundationBeaconProvider,
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
    # swarm_runner
    "AggregatedResult",
    "AggregatorClient",
    "PartialResult",
    "PartialResultIntegrityError",
    "SwarmDispatcher",
    "run_swarm",
    # orchestrator (composition class)
    "QueryOrchestrator",
    # retry-loop shell
    "EscrowClient",
    "PreemptionRecorder",
    "QueryRetriesExhaustedError",
    "RetryPolicy",
    "SlashClient",
    "dispatch_with_retries",
    # semantic_index_adapter
    "Embedder",
    "SemanticIndexAdapter",
    # sentence_transformer_embedder (B7-prep)
    "DEFAULT_MODEL_NAME",
    "SentenceTransformerEmbedder",
    # swarm_dispatcher_adapter
    "SwarmDispatcherAdapter",
    # aggregator_client_adapter (B5)
    "AggregateTransport",
    "AggregatorClientAdapter",
    # http_aggregate_transport (B7 — HTTP/TLS transport)
    "AggregateTransportError",
    "HttpAggregateTransport",
    # aggregate_protocol (B3.1b wire format)
    "AggregateRequest",
    "AggregateResponse",
    "SignedPartial",
    # aggregate_server (B3.2 server-side combination)
    "AggregateServer",
    "AggregateServerError",
    "PrivacyBudgetExhaustedError",
    "UnsupportedAgentOpError",
    "combine_partials",
    "enforce_a5_marker",
    "sum_privacy_budgets",
    "verify_partial_signature",
    # marketplace_candidate_pool_provider (B7-prep)
    "DEFAULT_STAKE_PER_TIER",
    "MarketplaceCandidatePoolProvider",
    # foundation_beacon_provider (B7-prep)
    "FoundationBeaconProvider",
]
