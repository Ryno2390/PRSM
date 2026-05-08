"""B7 — node-side QueryOrchestrator construction factory.

The factory `node.py:1277` calls. Composes the 5 dependency adapters
(B7-prep work) + the optional decomposer into a fully-wired
`QueryOrchestrator` — or returns None when disabled.

Default-disabled. The orchestrator only constructs when the operator
explicitly opts in via the `PRSM_QUERY_ORCHESTRATOR_ENABLED`
environment variable. This keeps `node.agent_forge = None` as the
default behavior — MCP gating stays on, canonical workflow remains
gated behind the wiring program until the operator verifies their
deployment is ready.

Production node startup pattern (in node.py once B7 lands):

```python
self.agent_forge = build_query_orchestrator_for_node(
    semantic_index=SemanticIndexAdapter(
        embedder=SentenceTransformerEmbedder(),
        index=self.content_uploader._semantic_index,
    ),
    dispatcher=SwarmDispatcherAdapter(
        agent_dispatcher=self.agent_dispatcher,
        per_shard_budget_ftns=BUDGET,
    ),
    aggregator_client=AggregatorClientAdapter(
        prompter_pubkey=self.identity.public_key_bytes,
        prompter_node_id=self.identity.node_id,
        prompter_signer=self.identity.sign,
        beacon_provider=beacon_provider,
        transport=HttpAggregateTransport(
            endpoint_resolver=self._resolve_aggregator_url,
        ),
    ),
    candidate_pool_provider=MarketplaceCandidatePoolProvider(
        directory=self.marketplace_directory,
        reputation=self.reputation_tracker,
    ),
    beacon_provider=FoundationBeaconProvider(
        foundation_safe_address=FOUNDATION_SAFE_ADDR,
    ),
)
```

The 5-argument signature is required-only — there are no defaults
because honest production wiring requires the operator to supply
real adapters, not stubs.

Per `docs/2026-05-08-query-orchestrator-wiring-readiness.md` B7.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Tuple

from prsm.compute.query_orchestrator.aggregator_selector import StakedNode
from prsm.compute.query_orchestrator.decomposer import QueryDecomposer
from prsm.compute.query_orchestrator.orchestrator import QueryOrchestrator
from prsm.compute.query_orchestrator.shard_finder import SemanticIndex
from prsm.compute.query_orchestrator.swarm_runner import (
    AggregatorClient,
    SwarmDispatcher,
)


# Env var the operator sets to opt in to QueryOrchestrator wiring.
# When unset / set to a falsy value, the factory returns None and
# `node.agent_forge` stays at its v1.6.0-removal-era state.
QUERY_ORCHESTRATOR_ENABLED_ENV = "PRSM_QUERY_ORCHESTRATOR_ENABLED"

_TRUTHY = frozenset({"1", "true", "yes"})


def is_query_orchestrator_enabled() -> bool:
    """Return True iff the operator has opted in via env var.

    Mirrors the same truthy-set used at `prsm/mcp_server.py:73-103`
    for `PRSM_EXPOSE_BROKEN_TOOLS` — keep them aligned so an operator
    flipping one knob can predict the other's accepted values."""
    val = os.getenv(QUERY_ORCHESTRATOR_ENABLED_ENV, "").lower()
    return val in _TRUTHY


def build_query_orchestrator_for_node(
    *,
    semantic_index: SemanticIndex,
    dispatcher: SwarmDispatcher,
    aggregator_client: AggregatorClient,
    candidate_pool_provider: Callable[[], Tuple[StakedNode, ...]],
    beacon_provider: Callable[[], bytes],
    decomposer: QueryDecomposer | None = None,
) -> QueryOrchestrator | None:
    """Build the QueryOrchestrator for `node.agent_forge` — or None
    if disabled.

    All 5 dependency arguments are REQUIRED kwargs. There are no
    default-stub fallbacks because honest production wiring requires
    the operator to supply real adapters; a default would silently
    let a half-wired deployment go live.

    Returns:
        QueryOrchestrator wired to all 5 dependencies (when env-enabled)
        OR
        None (when env-disabled — `node.agent_forge` stays None,
        canonical-workflow MCP tools remain gated by
        `BROKEN_TOOLS_HIDDEN`).

    Disabled-by-default is the safe behavior: an operator who hasn't
    explicitly opted in shouldn't accidentally route real queries
    through the orchestrator until they've verified their deployment
    can deliver the canonical workflow end-to-end.
    """
    if not is_query_orchestrator_enabled():
        return None

    return QueryOrchestrator(
        semantic_index=semantic_index,
        dispatcher=dispatcher,
        aggregator_client=aggregator_client,
        candidate_pool_provider=candidate_pool_provider,
        beacon_provider=beacon_provider,
        decomposer=decomposer,
    )
