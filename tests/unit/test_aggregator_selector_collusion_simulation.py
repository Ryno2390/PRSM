"""QueryOrchestrator aggregator-selector — Monte-Carlo collusion stress.

Per docs/2026-05-07-aggregator-selector-threat-model.md §"Test surface
(binding for implementation)" — the threat model demands a separate
stress harness beyond the cheap unit-level A1 test:

    > tests/unit/test_aggregator_selector_collusion_simulation.py:
    >     Monte-Carlo collusion stress (10k queries × varying stake
    >     fractions), pinned bounds.

This file pins selection rates across coalition stake fractions
{10%, 20%, 30%, 40%, 50%}. Each is bounded above by `coalition + 5pp`
— if a future change biases the selector, one of these blows.

Cost: ~1.5s on the dev box; left in the unit suite rather than
gated behind a slow marker so it runs on every wide regression sweep.
"""
from __future__ import annotations

import hashlib

import pytest

from prsm.compute.query_orchestrator import (
    SelectionInput,
    StakedNode,
    select_aggregator,
)


def _node(*, node_id: str, stake: int) -> StakedNode:
    return StakedNode(
        node_id=node_id,
        pubkey_hash=hashlib.sha256(node_id.encode()).digest(),
        stake_amount_ftns=stake,
        tier="T2",
        has_tee=False,
        reputation_score=1.0,
    )


def _build_pool(coalition_count: int, honest_count: int, stake_per_node: int):
    coalition = [
        _node(node_id=f"coalition-{i}", stake=stake_per_node)
        for i in range(coalition_count)
    ]
    honest = [
        _node(node_id=f"honest-{i}", stake=stake_per_node)
        for i in range(honest_count)
    ]
    return coalition + honest


def _run(pool, queries: int) -> int:
    coalition_keys = {n.node_id for n in pool if n.node_id.startswith("coalition-")}
    wins = 0
    for i in range(queries):
        spec = SelectionInput(
            prompter_node_id="alice",
            candidate_pool=tuple(pool),
            beacon_randomness=hashlib.sha256(f"beacon-{i}".encode()).digest(),
            query_id=hashlib.sha256(f"qid-{i}".encode()).digest(),
            sliding_window_state={},
            governance_denylist=frozenset(),
            requires_tee=False,
        )
        chosen = select_aggregator(spec)
        if chosen.node_id in coalition_keys:
            wins += 1
    return wins


# Coalition fraction → (coalition_count, honest_count). Equal stake per
# node, so fraction follows directly from the count split.
_PARAMS = [
    (0.10, 1, 9),
    (0.20, 2, 8),
    (0.30, 3, 7),
    (0.40, 4, 6),
    (0.50, 5, 5),
]


@pytest.mark.parametrize(
    "fraction,coalition_count,honest_count",
    _PARAMS,
    ids=[f"{int(f*100)}pct" for f, _, _ in _PARAMS],
)
def test_coalition_selection_rate_within_bound(
    fraction: float, coalition_count: int, honest_count: int
):
    """A coalition with X% of stake MUST NOT exceed X% selection rate by
    more than 5 percentage points over a 10k-query Monte-Carlo run.

    If a future selector change introduces stake-weighted bias beyond
    the sampling variance band, this test blows. The 5pp slack is
    sized for the binomial-tail at N=10k — beyond that it's a real
    bias, not noise."""
    pool = _build_pool(coalition_count, honest_count, stake_per_node=100)
    queries = 10_000
    wins = _run(pool, queries)
    rate = wins / queries
    upper = fraction + 0.05
    lower = fraction - 0.05
    assert lower <= rate <= upper, (
        f"coalition fraction={fraction:.2f}: "
        f"observed rate={rate:.3f}, expected in [{lower:.2f}, {upper:.2f}]"
    )
