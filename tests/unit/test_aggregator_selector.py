"""QueryOrchestrator aggregator-selector — binding test surface.

Per docs/2026-05-07-aggregator-selector-threat-model.md §"Test surface
(binding for implementation)" — these 12 tests cover the 10 adversary
classes plus general determinism. Each test names the adversary class
it locks down so a future regression's blast radius is obvious.

The aggregator-selector is the trickiest piece of the QueryOrchestrator
rebuild. Stake-weighted-collusion (A1), self-selection bias (A2), and
randomness biasing (A6) are easy to get wrong — these tests exist
specifically to make those failures noisy.
"""
from __future__ import annotations

import hashlib
import os
from typing import Iterable

import pytest

from prsm.compute.query_orchestrator import (
    AggregationCommit,
    AggregationCommitMismatchError,
    InsufficientCandidatesError,
    SelectionInput,
    StakedNode,
    select_aggregator,
    verify_aggregation_commit,
)


# ──────────────────────────────────────────────────────────────────────
# Fixture helpers
# ──────────────────────────────────────────────────────────────────────


def _pubkey_hash(seed: str) -> bytes:
    """Deterministic 32-byte pubkey hash for testing."""
    return hashlib.sha256(seed.encode()).digest()


def _node(
    *,
    node_id: str,
    stake: int = 1_000,
    tier: str = "T2",
    has_tee: bool = False,
    reputation: float = 0.5,
    pubkey_seed: str | None = None,
) -> StakedNode:
    """Build a StakedNode for tests. pubkey_seed defaults to node_id —
    the A8 test overrides this to verify same-key/different-id maps to
    same identity."""
    return StakedNode(
        node_id=node_id,
        pubkey_hash=_pubkey_hash(pubkey_seed or node_id),
        stake_amount_ftns=stake,
        tier=tier,
        has_tee=has_tee,
        reputation_score=reputation,
    )


def _spec(
    *,
    prompter: str = "prompter-node",
    pool: Iterable[StakedNode] = (),
    randomness: bytes = b"\x00" * 32,
    query_id: bytes = b"q" * 32,
    sliding_window_state: dict | None = None,
    denylist: Iterable[bytes] = (),
    requires_tee: bool = False,
) -> SelectionInput:
    return SelectionInput(
        prompter_node_id=prompter,
        candidate_pool=tuple(pool),
        beacon_randomness=randomness,
        query_id=query_id,
        sliding_window_state=sliding_window_state or {},
        governance_denylist=frozenset(denylist),
        requires_tee=requires_tee,
    )


# ──────────────────────────────────────────────────────────────────────
# Test 1 — A2 self-exclusion: prompter never selected
# ──────────────────────────────────────────────────────────────────────


class TestA2SelfExclusion:
    """Threat-model A2: a prompter's own node MUST NEVER be selected as
    aggregator for that prompter's query — the prompter has no honesty
    pressure when aggregating their own query."""

    def test_prompter_node_id_never_returned(self):
        prompter = _node(node_id="alice")
        peer = _node(node_id="bob")
        spec = _spec(prompter="alice", pool=[prompter, peer])
        # Try many randomness values — alice must never come back.
        for i in range(50):
            r = hashlib.sha256(f"r-{i}".encode()).digest()
            chosen = select_aggregator(
                SelectionInput(
                    prompter_node_id="alice",
                    candidate_pool=spec.candidate_pool,
                    beacon_randomness=r,
                    query_id=spec.query_id,
                    sliding_window_state={},
                    governance_denylist=frozenset(),
                    requires_tee=False,
                )
            )
            assert chosen.node_id == "bob"


# ──────────────────────────────────────────────────────────────────────
# Test 2 — A2 fail-closed: pool of size 1 = prompter raises
# ──────────────────────────────────────────────────────────────────────


class TestA2FailClosed:
    """If the only pool member is the prompter itself, the selector
    MUST raise rather than silently fall back — A2 mitigation explicitly
    requires fail-closed behavior."""

    def test_size_one_pool_with_prompter_raises(self):
        only = _node(node_id="alice")
        spec = _spec(prompter="alice", pool=[only])
        with pytest.raises(InsufficientCandidatesError):
            select_aggregator(spec)

    def test_empty_pool_raises(self):
        spec = _spec(prompter="alice", pool=[])
        with pytest.raises(InsufficientCandidatesError):
            select_aggregator(spec)


# ──────────────────────────────────────────────────────────────────────
# Test 12 — General determinism (relied on by every other behavioral test)
# ──────────────────────────────────────────────────────────────────────


class TestGeneralDeterminism:
    """Same `(beacon, query_id, pool, deny, rate_window)` → same
    selection. Required for replay verifiability + audit-trail
    reconstruction."""

    def test_identical_inputs_yield_identical_selection(self):
        pool = [_node(node_id=f"n{i}") for i in range(5)]
        spec = _spec(
            prompter="alice",
            pool=pool,
            randomness=hashlib.sha256(b"fixed-beacon").digest(),
            query_id=b"qid-deterministic" + b"\x00" * 13,
        )
        first = select_aggregator(spec)
        for _ in range(20):
            again = select_aggregator(spec)
            assert again.node_id == first.node_id
            assert again.pubkey_hash == first.pubkey_hash


# ──────────────────────────────────────────────────────────────────────
# Test 6 — A5 TEE gate: requires_tee=True filters non-TEE
# ──────────────────────────────────────────────────────────────────────


class TestA5TeeGate:
    """Threat-model A5: Tier C content requires TEE attestation. The
    selector MUST refuse to pick a non-TEE node when `requires_tee` is
    set, regardless of stake."""

    def test_non_tee_filtered_when_requires_tee_set(self):
        # Big-stake non-TEE node should still be excluded.
        non_tee = _node(node_id="big-no-tee", stake=1_000_000, has_tee=False)
        small_tee = _node(node_id="small-tee", stake=10, has_tee=True)
        spec = _spec(
            prompter="alice",
            pool=[non_tee, small_tee],
            requires_tee=True,
        )
        chosen = select_aggregator(spec)
        assert chosen.node_id == "small-tee"

    def test_non_tee_allowed_when_requires_tee_unset(self):
        non_tee = _node(node_id="cheap", has_tee=False)
        spec = _spec(prompter="alice", pool=[non_tee], requires_tee=False)
        # Should not raise — non-TEE is fine for Tier A/B content.
        assert select_aggregator(spec).node_id == "cheap"

    def test_no_tee_node_in_pool_raises_when_required(self):
        non_tee_a = _node(node_id="a", has_tee=False)
        non_tee_b = _node(node_id="b", has_tee=False)
        spec = _spec(
            prompter="alice",
            pool=[non_tee_a, non_tee_b],
            requires_tee=True,
        )
        with pytest.raises(InsufficientCandidatesError):
            select_aggregator(spec)


# ──────────────────────────────────────────────────────────────────────
# Test 7 — A6 commit-reveal: prompter cannot bias by re-submitting
# ──────────────────────────────────────────────────────────────────────


class TestA6CommitReveal:
    """Threat-model A6: a prompter MUST NOT be able to bias selection by
    re-submitting the same query. Enforced by binding selection to a
    beacon committed-to before the prompter sees the outcome.

    Unit-level: same beacon + same query_id ⇒ same selection (no
    re-roll). The orchestrator owns the commit-reveal handshake and
    feeds the selector a beacon it cannot replay."""

    def test_same_beacon_yields_same_selection_no_reroll(self):
        pool = [_node(node_id=f"n{i}", stake=1_000) for i in range(5)]
        beacon = hashlib.sha256(b"committed-beacon").digest()
        qid = b"q" * 32
        spec_a = _spec(prompter="alice", pool=pool, randomness=beacon, query_id=qid)
        # Same as spec_a — represents a re-submission.
        spec_b = _spec(prompter="alice", pool=pool, randomness=beacon, query_id=qid)
        assert select_aggregator(spec_a).pubkey_hash == select_aggregator(spec_b).pubkey_hash

    def test_different_beacon_changes_selection_distribution(self):
        # 50 nodes, equal stake. With 100 different beacons, we should
        # see at least 5 distinct selections — the seed actually feeds
        # selection (sanity check that A6 isn't a no-op).
        pool = [_node(node_id=f"n{i}", stake=1_000) for i in range(50)]
        seen = set()
        for i in range(100):
            beacon = hashlib.sha256(f"b-{i}".encode()).digest()
            spec = _spec(prompter="alice", pool=pool, randomness=beacon)
            seen.add(select_aggregator(spec).node_id)
        assert len(seen) >= 5, f"only {len(seen)} distinct selections — beacon not feeding"


# ──────────────────────────────────────────────────────────────────────
# Test 8 — A7 governance denylist filters before selection
# ──────────────────────────────────────────────────────────────────────


class TestA7GovernanceDenylist:
    """Threat-model A7: Foundation-council-flagged identities (Sybil
    correlations, persistent abuse) must be filtered out at selection
    time, regardless of their stake."""

    def test_denylisted_pubkey_hash_never_selected(self):
        evil = _node(node_id="evil", stake=10_000_000, reputation=1.0)
        good = _node(node_id="good", stake=10, reputation=1.0)
        spec = _spec(
            prompter="alice",
            pool=[evil, good],
            denylist=[evil.pubkey_hash],
        )
        for i in range(50):
            r = hashlib.sha256(f"r-{i}".encode()).digest()
            chosen = select_aggregator(
                SelectionInput(
                    prompter_node_id=spec.prompter_node_id,
                    candidate_pool=spec.candidate_pool,
                    beacon_randomness=r,
                    query_id=spec.query_id,
                    sliding_window_state={},
                    governance_denylist=spec.governance_denylist,
                    requires_tee=False,
                )
            )
            assert chosen.node_id == "good"

    def test_full_denylist_raises(self):
        a = _node(node_id="a")
        b = _node(node_id="b")
        spec = _spec(
            prompter="alice",
            pool=[a, b],
            denylist=[a.pubkey_hash, b.pubkey_hash],
        )
        with pytest.raises(InsufficientCandidatesError):
            select_aggregator(spec)


# ──────────────────────────────────────────────────────────────────────
# Test 9 — A8 pubkey-hash identity: same key ⇒ same identity
# ──────────────────────────────────────────────────────────────────────


class TestA8PubkeyHashIdentity:
    """Threat-model A8: long-range stake-hijack defense. The selector
    treats `pubkey_hash` (not the operator-supplied `node_id` string) as
    the load-bearing identity — re-keying = new identity, buying an
    old keypair = inheriting its slash history."""

    def test_same_pubkey_hash_different_node_id_treated_as_one_identity(self):
        # Same operator key, two different `node_id` strings — denying
        # by pubkey_hash must filter both. (If the implementation keyed
        # by node_id this would fail: only one of the two would be
        # filtered.)
        shared_key = "shared-operator-key"
        re_keyed_a = _node(node_id="alias-1", pubkey_seed=shared_key)
        re_keyed_b = _node(node_id="alias-2", pubkey_seed=shared_key)
        # Same pubkey_hash → set sees just one entry.
        assert re_keyed_a.pubkey_hash == re_keyed_b.pubkey_hash

        clean = _node(node_id="clean")
        spec = _spec(
            prompter="alice",
            pool=[re_keyed_a, re_keyed_b, clean],
            denylist=[re_keyed_a.pubkey_hash],
        )
        # Both aliases are filtered; only clean remains.
        for i in range(20):
            r = hashlib.sha256(f"r-{i}".encode()).digest()
            chosen = select_aggregator(
                SelectionInput(
                    prompter_node_id=spec.prompter_node_id,
                    candidate_pool=spec.candidate_pool,
                    beacon_randomness=r,
                    query_id=spec.query_id,
                    sliding_window_state={},
                    governance_denylist=spec.governance_denylist,
                    requires_tee=False,
                )
            )
            assert chosen.node_id == "clean"

    def test_pubkey_hash_field_is_thirty_two_bytes(self):
        """A8 mitigation specifies SHA-256 of the pubkey — 32 bytes.
        Pin the contract."""
        n = _node(node_id="any")
        assert len(n.pubkey_hash) == 32


# ──────────────────────────────────────────────────────────────────────
# Test 1 — A1 collusion bound: 30% stake selected ≤ 35% of the time
# ──────────────────────────────────────────────────────────────────────


class TestA1CollusionBound:
    """Threat-model A1: stake-weighted collusion. A coalition with X% of
    the staked pool MUST NOT exceed X% selection rate by more than a
    few-points-of-variance over 1000 queries.

    This test is the cheap unit-level form of the bound. The full
    Monte-Carlo stress test lives in
    `test_aggregator_selector_collusion_simulation.py`."""

    def test_thirty_percent_coalition_stays_below_thirty_five_percent(self):
        # Coalition: 3 nodes at 100 stake each (300 total).
        # Honest:    7 nodes at 100 stake each (700 total).
        # Total stake: 1000 → coalition = 30%.
        coalition_keys = {f"coalition-{i}" for i in range(3)}
        pool = [
            _node(node_id=f"coalition-{i}", stake=100, reputation=1.0)
            for i in range(3)
        ] + [
            _node(node_id=f"honest-{i}", stake=100, reputation=1.0)
            for i in range(7)
        ]
        coalition_wins = 0
        N = 1000
        for i in range(N):
            r = hashlib.sha256(f"q-{i}".encode()).digest()
            qid = hashlib.sha256(f"qid-{i}".encode()).digest()
            spec = _spec(
                prompter="alice",
                pool=pool,
                randomness=r,
                query_id=qid,
            )
            chosen = select_aggregator(spec)
            if chosen.node_id in coalition_keys:
                coalition_wins += 1
        rate = coalition_wins / N
        # Stake-weighted draw — expectation is 0.30, allow 5pp variance.
        # If rate > 0.35 the selector is biased (bug).
        assert 0.25 <= rate <= 0.35, f"30% coalition won {rate*100:.1f}% — outside bound"


# ──────────────────────────────────────────────────────────────────────
# Test 11 — A10 constant-time: pool reorder doesn't change selection
# ──────────────────────────────────────────────────────────────────────


class TestA10ConstantTime:
    """Threat-model A10: selection-process side channels. The selector
    sorts the pool internally so adversary-controlled ordering cannot
    leak via timing or affect selection."""

    def test_reordered_pool_yields_identical_selection(self):
        pool_a = tuple(_node(node_id=f"n{i}") for i in range(10))
        pool_b = tuple(reversed(pool_a))
        randomness = hashlib.sha256(b"beacon").digest()
        qid = b"q" * 32
        spec_a = SelectionInput(
            prompter_node_id="alice",
            candidate_pool=pool_a,
            beacon_randomness=randomness,
            query_id=qid,
            sliding_window_state={},
            governance_denylist=frozenset(),
            requires_tee=False,
        )
        spec_b = SelectionInput(
            prompter_node_id="alice",
            candidate_pool=pool_b,
            beacon_randomness=randomness,
            query_id=qid,
            sliding_window_state={},
            governance_denylist=frozenset(),
            requires_tee=False,
        )
        assert select_aggregator(spec_a).pubkey_hash == select_aggregator(spec_b).pubkey_hash


# ──────────────────────────────────────────────────────────────────────
# Test 1 (mitigation 2) — A1 per-staker rate limit
# ──────────────────────────────────────────────────────────────────────


class TestA1SlidingWindowRateLimit:
    """Threat-model A1 mitigation 2: bound the fraction of any prompter's
    queries that any single staker can win in a sliding window.

    Implementation: `sliding_window_state` is `{pubkey_hash_hex: count}`.
    Once a staker hits `MAX_AGG_FRACTION * total_window_queries`, the
    selector treats them as ineligible until their count rolls off the
    window. This caps coalition throughput from scaling linearly with
    stake — see threat model A1 §3 for the rationale."""

    def test_rate_limited_node_is_skipped(self):
        # Big-stake node A is over quota; small-stake B should win.
        a = _node(node_id="A", stake=1_000_000, reputation=1.0)
        b = _node(node_id="B", stake=10, reputation=1.0)
        spec = SelectionInput(
            prompter_node_id="alice",
            candidate_pool=(a, b),
            beacon_randomness=hashlib.sha256(b"r").digest(),
            query_id=b"q" * 32,
            # A has hit quota — selector must skip them.
            sliding_window_state={a.pubkey_hash.hex(): 9999},
            governance_denylist=frozenset(),
            requires_tee=False,
        )
        chosen = select_aggregator(spec)
        assert chosen.node_id == "B"

    def test_under_quota_node_still_eligible(self):
        # Same stake distribution as above but A is below quota.
        a = _node(node_id="A", stake=1_000_000, reputation=1.0)
        b = _node(node_id="B", stake=10, reputation=1.0)
        # Over many beacons, big-stake A wins (almost) all selections.
        a_wins = 0
        for i in range(50):
            spec = SelectionInput(
                prompter_node_id="alice",
                candidate_pool=(a, b),
                beacon_randomness=hashlib.sha256(f"r{i}".encode()).digest(),
                query_id=b"q" * 32,
                sliding_window_state={a.pubkey_hash.hex(): 0},
                governance_denylist=frozenset(),
                requires_tee=False,
            )
            if select_aggregator(spec).node_id == "A":
                a_wins += 1
        assert a_wins >= 45, f"big-stake won only {a_wins}/50 with empty window"


# ──────────────────────────────────────────────────────────────────────
# Test 10 — A9 commit-before-reveal verification
# ──────────────────────────────────────────────────────────────────────


class TestA9CommitBeforeReveal:
    """Threat-model A9: aggregator must commit a hash of the combined
    result on-chain (or to the orchestrator) BEFORE producing the
    plaintext result to anyone. Mismatch between commit and final
    delivery slashes the aggregator (per the orchestrator's slash
    surface; this module just verifies the binding)."""

    def test_matching_commit_verifies(self):
        plaintext = b"final-aggregated-result"
        digest = hashlib.sha256(plaintext).digest()
        commit = AggregationCommit(
            query_id=b"q" * 32,
            aggregator_pubkey_hash=hashlib.sha256(b"agg").digest(),
            result_digest=digest,
        )
        # No raise = OK.
        verify_aggregation_commit(commit, plaintext)

    def test_mismatched_commit_raises(self):
        plaintext = b"honest-result"
        wrong_digest = hashlib.sha256(b"different-result").digest()
        commit = AggregationCommit(
            query_id=b"q" * 32,
            aggregator_pubkey_hash=hashlib.sha256(b"agg").digest(),
            result_digest=wrong_digest,
        )
        with pytest.raises(AggregationCommitMismatchError):
            verify_aggregation_commit(commit, plaintext)

    def test_commit_digest_is_thirty_two_bytes(self):
        commit = AggregationCommit(
            query_id=b"q" * 32,
            aggregator_pubkey_hash=hashlib.sha256(b"agg").digest(),
            result_digest=hashlib.sha256(b"x").digest(),
        )
        assert len(commit.result_digest) == 32
        assert len(commit.aggregator_pubkey_hash) == 32


# ──────────────────────────────────────────────────────────────────────
# Tests 4 + 5 — A3 / A4 contracts (integration-scope; pin shape here)
# ──────────────────────────────────────────────────────────────────────


class TestA3A4ContractPins:
    """A3 (unbondDelay) is enforced at the StakeBond contract
    (HIGH-2 / D-01 invariant `unbondDelay >= challengeWindow`); the
    selector inherits it. A4 (preemption signal) is the orchestrator's
    contract — it calls `record_preemption` when the selected
    aggregator drops the request. Both are out-of-scope for unit tests
    on the selector itself per the threat-model "Test surface" section.

    What we CAN unit-test: the dataclass fields the orchestrator
    reads when integrating these. Pin them so renames don't silently
    break the integration."""

    def test_selection_input_carries_sliding_window(self):
        # A1 mitigation 2 + A4 preemption-discount feed via
        # sliding_window_state. Pin the field shape.
        spec = _spec(prompter="x", pool=[_node(node_id="y")])
        assert isinstance(spec.sliding_window_state, dict)

    def test_staked_node_carries_reputation_score(self):
        # ReputationTracker.score_for(...) feeds StakedNode.reputation_score;
        # preemption (A4 mitigation 2) drops this score over time.
        n = _node(node_id="x", reputation=0.42)
        assert n.reputation_score == 0.42
