"""QueryOrchestrator — aggregator selector.

Picks one T2+ stake-pool node per query to combine per-shard partial
results into a single response. The selector is the trickiest piece of
the QueryOrchestrator rebuild: stake-weighted-collusion (A1),
self-selection bias (A2), and randomness biasing (A6) are easy to get
wrong.

Binding design artifact:
    docs/2026-05-07-aggregator-selector-threat-model.md

Each adversary class A1..A10 maps to specific code paths in this module
— search "A<n>" comments to find a mitigation's implementation.

What this module does:
  - Filters the candidate pool: A2 self-exclusion, A5 TEE gate, A7
    governance denylist, A1 sliding-window per-staker rate limit.
  - Computes a stake-weighted, deterministic selection seeded by a
    beacon (A6) and the query_id, so replay verifies.
  - Excludes nodes whose pubkey-hash carries slash history (A8) — the
    selector accepts pre-filtered reputation scores and treats
    `reputation_score == 0.0` as effective hard exclusion.

What this module does NOT do:
  - Mint or revoke stake — `StakeBond.sol` owns that.
  - Persist reputation — `ReputationTracker` is in-memory; a future
    persistence layer keys by pubkey-hash per A8.
  - Implement the commit-reveal beacon — the orchestrator constructs
    `beacon_randomness` from the on-chain anchor and feeds it in.
  - Run the redundancy `p_check` re-selection — that's the
    orchestrator's retry loop (A1 mitigation 3).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import ClassVar


class InsufficientCandidatesError(RuntimeError):
    """The candidate pool, after filtering, contains zero eligible
    aggregators. Caller (orchestrator) decides whether to widen the
    pool, escalate to operator, or refund the prompter (A4).

    Distinct exception type rather than ValueError so the orchestrator
    can route on it explicitly.
    """


class AggregationCommitMismatchError(RuntimeError):
    """The aggregator's pre-commit hash does not match the plaintext
    they finally delivered (A9). Caller (orchestrator) feeds this into
    `ReputationTracker.record_slash` so the misbehavior carries
    on-chain consequences.

    Distinct from InsufficientCandidatesError because the response is
    different — slash + retry vs. refund.
    """


# Default per-prompter quota for the A1 sliding-window rate limit.
# Ratification target: Foundation council (`MAX_AGG_FRACTION`,
# threat-model "Open governance questions" item 2). Default proposed
# at 1/N where N = pool size; absent runtime sizing we hardcode an
# absolute query count that approximates "roughly fair" until the
# orchestrator threads pool size in.
#
# Concretely: if a prompter has issued more than this many queries in
# the rolling window AND a single pubkey-hash already won this many of
# them, treat that pubkey-hash as ineligible for the next selection.
A1_DEFAULT_PER_STAKER_WINDOW_LIMIT = 1000


@dataclass(frozen=True)
class StakedNode:
    """A T2+ stake-pool member eligible for aggregator selection.

    Frozen to make selection deterministic — same inputs always hash
    the same way.

    Attributes
    ----------
    node_id:
        Operator-supplied human-readable identifier; for routing only.
        NOT load-bearing for slash-history lookup — see `pubkey_hash`.
    pubkey_hash:
        32-byte SHA-256 of the operator's signing pubkey. THIS is the
        identity that carries reputation + slash history (A8). Re-keying
        = new identity (intentional, irreversible from the protocol's
        standpoint). Buying an old keypair = inheriting that keypair's
        slash history.
    stake_amount_ftns:
        Absolute stake at selection time, in FTNS base units. Selector
        uses this as the stake-weighting input for A1.
    tier:
        "T2" | "T3" | "T4" — hardware-tier classification from
        `MarketplaceOrchestrator`. Selection requires >= T2 by Vision
        §6; the candidate pool is pre-filtered upstream.
    has_tee:
        True iff the node runs in a TEE-attested environment. Required
        for Tier C content per A5.
    reputation_score:
        Score in [0.0, 1.0] from `ReputationTracker.score_for(...)`.
        The selector multiplies stake-weight by this — slashed nodes
        with score → 0 get effectively zero selection probability.
    """
    node_id: str
    pubkey_hash: bytes
    stake_amount_ftns: int
    tier: str
    has_tee: bool
    reputation_score: float


@dataclass(frozen=True)
class SelectionInput:
    """All inputs the selector consumes.

    Frozen so determinism tests can hash the spec and rely on equal
    inputs producing equal outputs. The orchestrator constructs this
    once per query, after computing `beacon_randomness` from the
    commit-reveal anchor.
    """
    prompter_node_id: str
    """REQUIRED — A2 hard exclusion. Selector raises rather than
    silently fall back if the prompter is the only candidate."""

    candidate_pool: tuple[StakedNode, ...]
    """T2+ pre-filtered, deduplicated by pubkey_hash. Order does NOT
    affect selection (A10) — the selector sorts internally."""

    beacon_randomness: bytes
    """32 bytes from the commit-reveal beacon. The prompter cannot
    bias selection by re-submitting the same query because the beacon
    is fixed at commit time (A6)."""

    query_id: bytes
    """Per-query identifier. Combined with `beacon_randomness` and
    `prompter_node_id` to seed the selection, so replays of the same
    query yield the same selection (test 12)."""

    sliding_window_state: dict
    """Per-prompter rolling-window state used by A1 mitigation 2 — the
    `MAX_AGG_FRACTION` per-staker rate limit. Concrete shape is owned
    by the orchestrator's window manager; the selector reads it.

    Schema: {pubkey_hash_hex: count_in_window}. Empty dict = no rate
    limit applies (e.g., new prompter)."""

    governance_denylist: frozenset[bytes]
    """pubkey_hashes (32-byte) explicitly disallowed by Foundation
    council action (A7). Filtered before selection. Empty by default."""

    requires_tee: bool
    """True when the query touches Tier C content. Forces the selector
    to filter out non-TEE nodes (A5) regardless of stake."""


# ──────────────────────────────────────────────────────────────────────
# Selection
# ──────────────────────────────────────────────────────────────────────


def _filter_eligible(spec: SelectionInput) -> list[StakedNode]:
    """Apply the A2 / A5 / A7 / A1 filters in order.

    Returns the list of nodes still eligible after all filters. The
    caller decides what to do if the result is empty
    (`InsufficientCandidatesError` is the canonical answer).
    """
    out: list[StakedNode] = []
    for n in spec.candidate_pool:
        # A2: prompter never selects itself.
        if n.node_id == spec.prompter_node_id:
            continue
        # A5: Tier C content requires TEE.
        if spec.requires_tee and not n.has_tee:
            continue
        # A7: Foundation-council denylist.
        if n.pubkey_hash in spec.governance_denylist:
            continue
        # A1 mitigation 2: per-staker rate limit. If the orchestrator's
        # sliding-window state shows this pubkey_hash has hit the
        # per-prompter quota, skip them. Empty/missing entry = under
        # quota (e.g., new prompter or freshly-rolled-off window).
        win_count = spec.sliding_window_state.get(n.pubkey_hash.hex(), 0)
        if win_count >= A1_DEFAULT_PER_STAKER_WINDOW_LIMIT:
            continue
        out.append(n)
    return out


def select_aggregator(spec: SelectionInput) -> StakedNode:
    """Pick exactly one aggregator from the candidate pool.

    Raises `InsufficientCandidatesError` if no candidate survives the
    A2 / A5 / A7 / A1 filters.

    The selection is deterministic in `(beacon_randomness, query_id,
    pool by pubkey_hash, denylist, prompter)` — same inputs yield the
    same selection (test 12). This is required so the orchestrator's
    audit trail can replay-verify a selection from the on-chain commit
    record (A6).
    """
    eligible = _filter_eligible(spec)
    if not eligible:
        raise InsufficientCandidatesError(
            f"no eligible aggregator after filtering "
            f"(prompter={spec.prompter_node_id}, "
            f"pool_size={len(spec.candidate_pool)}, "
            f"requires_tee={spec.requires_tee}, "
            f"denylist_size={len(spec.governance_denylist)})"
        )

    # A10: sort by pubkey_hash so pool reorder doesn't affect the
    # selection or the timing. The selector walks the same sequence
    # regardless of how the caller ordered the pool.
    eligible.sort(key=lambda n: n.pubkey_hash)

    # Deterministic seed: beacon + query_id + prompter (A6 binds this
    # to the on-chain commit). prompter inclusion means two different
    # prompters with the same query don't get the same aggregator —
    # an honest property for the load-balancing dimension.
    seed = hashlib.sha256(
        spec.beacon_randomness
        + spec.query_id
        + spec.prompter_node_id.encode()
    ).digest()
    seed_int = int.from_bytes(seed, "big")

    # Stake-weighted draw. weight_i = stake_i * reputation_i (A1 + A8
    # via reputation feeding from pubkey_hash-keyed tracker). Float
    # multiplication is fine here; precision drift below 1 LSB doesn't
    # affect determinism because seed_int is integral.
    weights = [
        max(int(n.stake_amount_ftns * n.reputation_score), 0)
        for n in eligible
    ]
    total = sum(weights)
    if total <= 0:
        # All eligible nodes have zero effective weight (e.g., all
        # have reputation_score == 0.0). Treat as insufficient — the
        # orchestrator should refund per A4 mitigation 3.
        raise InsufficientCandidatesError(
            "all eligible candidates have zero effective stake-weight"
        )

    pick = seed_int % total
    cursor = 0
    for node, w in zip(eligible, weights):
        cursor += w
        if pick < cursor:
            return node
    # Unreachable: pick < total by construction. Belt-and-suspenders.
    return eligible[-1]


# ──────────────────────────────────────────────────────────────────────
# A9 — commit-before-reveal aggregation receipt
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AggregationCommit:
    """Pre-commit a selected aggregator publishes BEFORE delivering the
    plaintext combined result (A9 mitigation 1).

    The orchestrator records this in its receipts log (and optionally
    on-chain as a periodic anchor). When the aggregator subsequently
    delivers the plaintext result, `verify_aggregation_commit` proves
    the result wasn't substituted post-commit.

    Pattern mirrors the `RollbackCacheRequest.replay_accepted_prefix`
    commit pattern from Phase 3.x.11.q.y' (Task 1) — same shape, same
    audit-trail role.

    Attributes
    ----------
    query_id:
        Binds the commit to a specific query — prevents reuse of an
        aggregator's commit across queries.
    aggregator_pubkey_hash:
        32 bytes. The selected aggregator's identity. Slash routing
        keys off this — A8.
    result_digest:
        SHA-256 of the plaintext aggregated result. 32 bytes.
        Mismatch with the actual delivered plaintext = slash event.
    """
    query_id: bytes
    aggregator_pubkey_hash: bytes
    result_digest: bytes

    # Signing payload prefix — pinned by golden-vector test in
    # test_aggregation_commit_signing_payload.py. Renaming the prefix
    # would invalidate every previously-signed commit; if a future
    # version genuinely needs a new payload shape, bump to "v2" and
    # support both during the transition window.
    SIGNING_PREFIX: ClassVar[bytes] = b"prsm:aggregation-commit:v1\n"

    def signing_payload(self) -> bytes:
        """Canonical signing payload for the aggregator's signature
        over this commit (per `docs/2026-05-08-aggregate-rpc-design.md`
        — pattern-lift from Phase 3.x.1 InferenceReceipt).

        Layout (96 bytes after prefix):
            SIGNING_PREFIX (28 bytes)
            query_id              (32 bytes)
            aggregator_pubkey_hash(32 bytes)
            result_digest         (32 bytes)

        All three fields are fixed-width 32 bytes — no length-prefixing
        needed. Total: 124 bytes.
        """
        if len(self.query_id) != 32:
            raise ValueError(
                f"query_id must be 32 bytes for signing payload, "
                f"got {len(self.query_id)}"
            )
        if len(self.aggregator_pubkey_hash) != 32:
            raise ValueError(
                f"aggregator_pubkey_hash must be 32 bytes for signing "
                f"payload, got {len(self.aggregator_pubkey_hash)}"
            )
        if len(self.result_digest) != 32:
            raise ValueError(
                f"result_digest must be 32 bytes for signing payload, "
                f"got {len(self.result_digest)}"
            )
        return (
            self.SIGNING_PREFIX
            + self.query_id
            + self.aggregator_pubkey_hash
            + self.result_digest
        )


def verify_aggregation_commit(
    commit: AggregationCommit,
    plaintext_result: bytes,
) -> None:
    """Verify the aggregator's pre-commit matches the plaintext they
    delivered. Raises `AggregationCommitMismatchError` if not.

    No-return-on-success is intentional — this is a pure assertion in
    the orchestrator's path. The orchestrator either continues delivery
    (no exception) or records a slash (catches the exception).
    """
    actual = hashlib.sha256(plaintext_result).digest()
    if actual != commit.result_digest:
        raise AggregationCommitMismatchError(
            f"aggregator {commit.aggregator_pubkey_hash.hex()[:8]} "
            f"committed digest {commit.result_digest.hex()[:16]}... "
            f"but delivered plaintext hashes to {actual.hex()[:16]}... "
            f"(query_id={commit.query_id.hex()[:16]}...)"
        )
