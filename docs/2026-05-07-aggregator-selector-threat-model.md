# QueryOrchestrator Aggregator-Selector — Threat Model

**Date:** 2026-05-07
**Track:** Pre-A1 design artifact for the QueryOrchestrator rebuild
named in `docs/2026-05-07-canonical-workflow-gap-list-delta.md`.
**Scope:** A NEW primitive that does not yet exist. This document
specifies what the primitive must defend against BEFORE code lands.

**Why this comes first:** Per the gap-list delta, the
aggregator-selector is "the trickiest piece" of the QueryOrchestrator
rebuild. Stake-weighted-collusion, self-selection bias, and
selection-process manipulation are all easy to get wrong — this
document fixes the threat model so implementation can be reviewed
against an explicit standard rather than retroactive intuition.

---

## What the aggregator-selector does

When a user submits a data query, the QueryOrchestrator decomposes
the query into a DSL manifest (existing — `AgentOp`), discovers the
relevant data shards (existing — EmbeddingDHT + ManifestDHT), fans
out WASM agents to the shard-holders (existing — AgentDispatcher +
AgentExecutor), and **selects one aggregator node** to:

1. Receive the per-shard partial results,
2. Apply DP noise (existing — `dp_noise.py`),
3. Combine into a single response,
4. Deliver to the prompter,
5. Trigger batch settlement (existing — settler registry).

Per Vision §6 the aggregator MUST be drawn from the T2+ stake pool
(`StakeBond`-backed) and SHOULD NOT be the prompter's own node.

---

## Trust-model inputs

| Primitive | Status | Provides |
|---|---|---|
| `StakeBond.sol` mainnet | ✅ live (Phase 7 §B mainnet 2026-05-07) | per-node FTNS-denominated stake, slashable |
| `ReputationTracker` | ✅ live (`prsm/marketplace/reputation.py`) | success / failure / preemption / slash event history |
| `MarketplaceOrchestrator` tier gating | ✅ live (Phase 7 Task 5) | T1/T2/T3/T4 hardware-tier classification |
| `EmbeddingDHT` content lookup | ✅ live (T4.9.next4 keystone unblock — 2026-05-07) | enumerates which nodes hold which shards |
| `dp_noise.py` Laplace + Gaussian | ✅ live (`prsm/compute/tee/dp_noise.py`) | privacy-budget-bounded noise application |
| `RoyaltyDistributor` v1 mainnet | ✅ live (Base mainnet 0x3E82…D6c2) | enforced split — settlement layer |

The aggregator-selector consumes these and produces a single
selection. It does NOT itself enforce slashing or stake — it routes
selection events into them.

---

## Adversary catalog

### A1. Stake-weighted collusion

**Goal:** A coalition controls X% of the staked T2+ pool. They want
selection probability ≥ X%, then bias aggregations toward their
preferred output.

**Capability:** Can split stake across N identities (no per-identity
upper bound assumed); can pre-coordinate offline; can return
plausible-but-wrong results that survive plausibility checks unless
cross-checked.

**Why this is the load-bearing threat:** Pure stake-weighted random
selection makes this directly proportional. If a Foundation-adjacent
operator stakes 30% of the pool, they aggregate 30% of queries —
adversary throughput scales linearly with capital.

**Mitigations:**

1. **Anti-correlation random beacon.** Selection randomness MUST be
   sourced from a beacon the prompter cannot bias and the candidate
   pool cannot pre-compute the outcome of. Recommended: `commit-reveal`
   on a query-id-derived seed, with selection finalized at reveal
   time. See A6 below for stricter form.

2. **Rate-limit per-staker selection.** Bound the fraction of any
   prompter's queries that any single stake holder can win in a
   sliding window. Concretely: a single `StakeBond` cannot serve as
   aggregator for more than `MAX_AGG_FRACTION = 1/N` of a given
   prompter's last 24h queries, where N is the size of the eligible
   pool. Prevents adversary throughput from scaling linearly with
   stake — caps the slope.

3. **Result-quality redundancy sample.** With probability `p_check`
   (e.g., 0.05), the orchestrator runs the SAME query through TWO
   independently-selected aggregators and compares aggregated
   outputs. Disagreement triggers slash via `ReputationTracker.record_slash`.
   The `p_check` rate is tunable; high enough to make
   plausible-but-wrong results expensive in expectation, low enough
   to keep query cost from doubling.

4. **Cross-shard plausibility check at the orchestrator.** The
   orchestrator already knows the per-shard outputs (it dispatched
   them). Before delivering the aggregator's result to the prompter,
   spot-check that the aggregator's output is consistent with
   per-shard evidence (e.g., aggregator's `count` ≤ sum of per-shard
   counts; aggregator's `sum` within the bound implied by per-shard
   sums). Detects naive collusion that returns a number unrelated
   to the inputs.

**Residual risk:** A coalition with > 50% of stake AND a willingness
to slash itself can still bias aggregations within plausibility. The
`p_check` parameter trades off cost vs. collusion-resistance. This
is a parameter for Foundation council ratification — propose
`p_check = 0.05` as the default; revisit at TVL > $100K.

---

### A2. Self-selection bias

**Goal:** Prompter's node selects itself as aggregator → no honesty
pressure (the prompter has no incentive to fairly aggregate the
query they themselves issued).

**Capability:** Trivial when the selector runs on the prompter's
node and the random beacon is local.

**Mitigation:** **Hard exclusion.** The selector MUST receive the
prompter's `node_id` as an explicit argument and exclude that node
from the candidate pool. This is a single-line filter, but it must
be a *required* parameter rather than optional — make the type
signature force the question. Code shape:

```python
def select_aggregator(
    *,
    prompter_node_id: str,            # REQUIRED — used for exclusion
    candidate_pool: Iterable[StakedNode],
    randomness: bytes,
    rate_limit_window: SlidingWindow,
) -> StakedNode:
    ...
```

Tests: explicit assertion that `select_aggregator` raises if
`prompter_node_id` is in the candidate pool and the pool is size 1
(no fallback).

---

### A3. Nothing-at-stake / stake-grinding

**Goal:** Adversary stakes, wins selection, returns garbage,
withdraws stake before slash lands. Or: stakes on N forks of the
network's expected state and chooses the one that maximizes payoff
post-hoc.

**Capability:** StakeBond.sol has a finite `unbondDelay` — adversary
can stake, perform N malicious aggregations within that window, then
unbond. The slash window must extend past unbond completion for the
slash to actually bite.

**Existing mitigation:** Phase 7.1 Task 47 — HIGH-2 (A-02/D-01)
remediation enforced the invariant `unbondDelay >= challengeWindow`
on the audit-bundle. Aggregator-selector inherits this — a selected
aggregator cannot fully unbond until the challenge window closes,
during which the orchestrator's `p_check` redundancy sample (A1
mitigation 3) can catch and slash.

**Specific to aggregator-selector:** The `challengeWindow` for
aggregator misbehavior MUST be set wide enough to span the
worst-case redundancy-sample latency. Recommended: 24h — exceeds
the orchestrator's `p_check` re-run + comparison window.

---

### A4. Selected-aggregator denial-of-service

**Goal:** Selected aggregator never produces output → query times
out → prompter loses budget OR re-selection burns budget for nothing.

**Capability:** Trivial. Just don't respond.

**Mitigations:**

1. **Bounded selection re-tries.** Cap aggregator selection retries
   per query at `MAX_AGG_RETRIES = 3`. Each failure records
   `record_preemption` against the dropped aggregator —
   `ReputationTracker` already supports this exact event type at
   line 133.

2. **Reputation discount on preemption.** Preempted nodes drop in
   the candidate pool's selection weight. Existing
   `score_for(provider_id)` already accounts for failure history;
   the selector reads this score.

3. **Escrow refund on no-aggregator.** If retries exhaust without a
   responsive aggregator, the prompter's budget refunds (minus
   already-paid per-shard agent costs). Requires escrow integration
   — orchestrator owns the escrow account.

---

### A5. Aggregator-as-MEV / query-content leakage

**Goal:** Aggregator sees the query plaintext + per-shard partial
results before the prompter does, can act on that information first
(front-running for trading data, leaking sensitive queries, etc.).

**Capability:** The aggregator is BY DESIGN the trusted middle layer
that combines partial results — it sees them all by construction.

**Mitigations:**

1. **Tier gate.** Vision §6 already mandates T2+ — implies the
   aggregator runs in a TEE-attested environment for Tier C content
   (existing `prsm/compute/tee/confidential_executor.py` from Phase
   3.x.1 Task 3). Selector MUST refuse to pick a non-TEE node when
   the query touches Tier C content.

2. **DP noise BEFORE combination.** The per-shard results delivered
   to the aggregator should already be DP-noised by the agent
   (existing `dp_noise.py`). The aggregator never sees raw shard
   data — it sees DP-noised partials and recombines them. Privacy
   budget tracking exists (`prsm/security/privacy_budget.py`).

3. **Receipt commitment.** The aggregator signs an
   `AggregationReceipt` similar to `InferenceReceipt` (Phase
   3.x.1 Task 2 sign/verify pattern). Forensic traceability if
   leakage is later detected on-chain.

**Residual risk:** A T2+ TEE-running but malicious operator with
hardware exfil can still leak. The threat model assumes TEE
attestation is honest. R5 (Tier C hardening) tracks the residual.

---

### A6. Selection randomness biasing

**Goal:** Adversary biases the random source feeding the selector
so they win selection more than fair stake-weight.

**Capability:** Depends on randomness source. Local RNG: trivial
(adversary owns the prompter's node). Block-hash: leak-bias
(adversary observes block hash, computes their selection probability,
withdraws stake if they're not selected and re-stakes for the next
query — at no cost since unbondDelay covers nothing here). Ed25519
signature on query_id: prompter can grind query ids to bias.

**Mitigation:** **Commit-reveal on a Foundation-Safe-anchored
beacon.**

Step 1 (commit): At query submission, prompter commits to a query
hash, paying the budget into escrow. Block height H_commit recorded
on-chain (or in a signed attestation if no on-chain trip is
acceptable for cost reasons).

Step 2 (reveal): Selection randomness =
`HMAC-SHA256(beacon_at(H_commit + Δ), query_id || prompter_node_id)`
where:
- `beacon_at(H)` is the on-chain randomness at height H — Base
  mainnet `block.prevrandao` (Beacon Chain's RANDAO trickled down
  via OP-Stack derivation; not perfectly unbiasable but
  sufficiently unpredictable for a per-query selection)
- `Δ ≥ 5 blocks` so the prompter cannot influence the beacon by
  withholding their commit until a block hash they prefer

This forces the prompter to commit BEFORE the beacon they need is
finalized, eliminating prompter-side biasing.

**Implementation note:** Going on-chain per query is too expensive
for queries pricing at < ~1¢. The cheaper alternative: a ROTATING
beacon where each Foundation Safe member signs a daily randomness
update, the orchestrator uses the most recent unrevealed beacon
+ commits on chain only periodically (e.g., every Nth query as
audit anchor). Worth a dedicated ratification call when this
implementation lands — propose the on-chain beacon with periodic
anchoring as the default and revisit at TVL > $1M.

---

### A7. Sybil-at-stake-time

**Goal:** One entity stakes via N identities, each controlling
1/N-th of the total stake. Per-identity rate-limit (A1 mitigation 2)
is bypassed because the selector sees N "different" stakers.

**Capability:** StakeBond.sol does not enforce one-stake-per-real-
entity. Anyone with > min_stake can mint a new identity.

**Mitigation:** **Out of scope for the selector.** Sybil-resistance
at stake-mint time is a Foundation-level governance problem
(KYC for high-stake bonds? per-IP rate limits? optional
proof-of-personhood?). The selector inherits whatever policy the
StakeBond admin enforces — and the existing R3 threat model already
flags this as out of scope for individual subsystems.

**What the selector CAN do:** Surface visible signals.
`ReputationTracker.record_failure` aggregated per identity; if N
identities all owned by the same operator misbehave together,
correlated failure patterns become visible to the Foundation
council, which can blacklist via a governance-controlled deny-list
on top of the candidate pool.

**Status:** Document the dependency. Add a `governance_denylist:
Set[NodeId]` parameter to `select_aggregator` so Foundation council
ratification of Sybil-flagged identities feeds directly into the
selection logic.

---

### A8. Long-range stake-hijack

**Goal:** An old node_id that was previously slashed gets re-funded
(maybe by a different operator buying the keypair), restakes, gets
selected as aggregator.

**Capability:** Slash events live in `ReputationTracker._slash_events`
keyed by `provider_id`. If `provider_id == node_id` and node_id is
re-used, the slash history persists. If a NEW node_id is minted,
slash history is fresh.

**Mitigation:** **Hash-of-pubkey identity.** Selector uses
`hash(pubkey_bytes)` rather than the operator-supplied `node_id`
string for slash-history lookup. Re-keying = new identity (correct,
irreversible from the protocol's standpoint). Buying an old keypair
+ restaking = inheriting that keypair's slash history. Aligns
incentives: if you buy a slashed keypair you bought its bad
reputation.

**Implementation:** This is a single-line change in
`ReputationTracker.record_slash` — key by pubkey hash, not by the
free-form `provider_id` string. May require a migration of the
existing reputation store to re-key. Track as a follow-on task once
reputation persistence is wired (currently in-memory per-process).

---

### A9. Reveal censorship by selected aggregator

**Goal:** Selected aggregator partially aggregates, reveals to a
preferred recipient (e.g., a competitor of the prompter) before
delivering to the prompter.

**Capability:** The aggregator holds the only copy of the combined
result for a brief window. They can leak it before delivery.

**Mitigation:**

1. **Time-locked delivery.** Aggregator must commit a hash of the
   combined result on-chain (or to the orchestrator) BEFORE
   producing the plaintext result to anyone. Pre-commit window
   forces simultaneous reveal. Mirrors the
   `RollbackCacheRequest.replay_accepted_prefix` commit pattern
   from Phase 3.x.11.q.y' (Task 1).

2. **Encryption-to-prompter at agent level.** Per-shard agents
   encrypt their partial results to the prompter's pubkey; the
   aggregator combines ciphertexts. This requires homomorphic
   aggregation primitives, which is R1 (FHE-for-inference) territory
   and out of scope for v1. Document the upgrade path; ship without
   it.

**Residual risk:** A malicious aggregator who breaks the time-lock
before delivery can still leak the hash-preimage off-band. The
mitigation is *traceable* (the on-chain commit shows when reveal
happened) but not *preventive*. Acceptable v1 — upgrade to FHE-based
aggregation when R1 promotes.

---

### A10. Selection-process side channels

**Goal:** Adversary observes selector internal state (timing,
memory) to predict future selections and pre-position.

**Capability:** Speculative-execution / timing side channels on the
selector node. R3 + R5 + Phase 3.x.10.y already cover Tier C
constant-time padding for inference dispatch — but the SELECTOR
itself is a separate code path.

**Mitigation:** Constant-time selection logic. Document the
requirement; reuse existing constant-time DP-noise primitives where
shape allows. Add to test suite:

```python
def test_select_aggregator_constant_time_under_pool_perturbation():
    # Same query, same beacon — selection time should not vary by
    # > 10% as candidate pool is reordered (this catches naive
    # early-exit on "first match").
```

---

## Code shape (binding for implementation)

```python
# prsm/compute/query_orchestrator/aggregator_selector.py

from dataclasses import dataclass
from typing import Iterable, Optional, Set

@dataclass(frozen=True)
class StakedNode:
    """A T2+ stake pool member eligible for aggregator selection."""
    node_id: str               # operator-supplied; for routing only
    pubkey_hash: bytes         # 32 bytes — load-bearing identity (A8)
    stake_amount_ftns: int     # absolute stake at selection time
    tier: str                  # "T2" | "T3" | "T4" — A5 gate input
    has_tee: bool              # for Tier C dispatch — A5
    reputation_score: float    # ReputationTracker.score_for(...)


@dataclass(frozen=True)
class SelectionInput:
    """All inputs the selector consumes. Frozen to make
    determinism tests easy."""
    prompter_node_id: str                    # REQUIRED — A2
    candidate_pool: tuple[StakedNode, ...]   # T2+ pre-filtered
    beacon_randomness: bytes                 # A6
    query_id: bytes
    sliding_window_state: dict               # A1 — per-prompter
    governance_denylist: frozenset[bytes]    # A7 — pubkey hashes
    requires_tee: bool                       # A5 — Tier C content


def select_aggregator(spec: SelectionInput) -> StakedNode:
    """Pick exactly one aggregator from the candidate pool.

    Raises ``InsufficientCandidatesError`` if the pool after
    filtering (A2 + A7 + A5 + A1) is empty.

    Constant-time under candidate-pool perturbation (A10).

    All failure modes are explicit exceptions — no silent fallback.
    The orchestrator's retry loop (A4) consumes those exceptions
    and decides whether to re-poll or refund.
    """
```

Excluded from v1, document upgrade path:
- FHE-based aggregation (A9 mitigation 2 — R1)
- Per-IP Sybil resistance at stake-mint (A7 — governance)
- Reputation persistence across process restarts (A8 — separate
  follow-on task)

---

## Test surface (binding for implementation)

`tests/unit/test_aggregator_selector.py` — required coverage:

1. **A1 collusion bound**: a coalition with 30% of stake selected
   ≤ 35% of the time over 1000 queries (5 percentage points slack
   for variance).
2. **A2 self-exclusion**: prompter's own node_id never returned.
3. **A2 fail-closed**: pool of size 1 = prompter → raises.
4. **A3 inherits unbondDelay**: out-of-scope for unit; integration
   test against StakeBond mock.
5. **A4 preemption signal**: when a node returns a "no thanks"
   response, `record_preemption` is called.
6. **A5 TEE gate**: `requires_tee=True` filters non-TEE nodes
   regardless of stake.
7. **A6 commit-reveal**: prompter cannot bias selection by
   re-submitting the same query — beacon at commit-time fixed.
8. **A7 governance denylist**: explicit deny list filters before
   selection.
9. **A8 pubkey-hash identity**: same operator key, different
   `node_id` string → same slash history.
10. **A9 commit before reveal**: aggregator's pre-commit hash
    matches their final delivery; mismatch slashes.
11. **A10 constant-time**: pool reorder doesn't change selection
    time > 10%.
12. **General determinism**: same `(beacon, query_id, pool, deny,
    rate_window)` → same selection.

Plus:
- `tests/unit/test_aggregator_selector_collusion_simulation.py`:
  Monte-Carlo collusion stress (10k queries × varying stake
  fractions), pinned bounds.

---

## Open governance questions for council ratification

1. **`p_check` redundancy rate.** Default `0.05` proposed.
2. **`MAX_AGG_FRACTION` per-prompter rate limit.** Default `1/N`
   where N = pool size.
3. **`MAX_AGG_RETRIES`** per query. Default 3.
4. **Beacon source.** On-chain `block.prevrandao` per query
   (expensive) vs. Foundation-multisig daily beacon with periodic
   on-chain anchor (cheap). **Recommendation: daily beacon + every-100th-query
   on-chain anchor at v1 launch; revisit at TVL > $1M.**
5. **Governance denylist mechanism.** Foundation Safe 2-of-3
   multisig writes to a deny-list contract; selector polls.
6. **`challengeWindow` for aggregator misbehavior.** Default 24h.

These six parameters are NOT in the threat model itself — they're
inputs the threat model demands be set. Ratification call when
`select_aggregator` lands.

---

## What this document does NOT cover

- The QueryOrchestrator's other sub-modules (decomposer,
  shard_finder, swarm_runner) — separate threat models when those
  land.
- FHE / MPC alternatives to the trusted-aggregator model — R1 / R2
  research tracks. Documented as upgrade paths (A9 mitigation 2).
- Sybil resistance at stake-mint — governance, not selector
  (documented out-of-scope at A7).
- Slashing economics (size of slash, duration of bond, etc.) —
  Phase 7 territory; reused here.

## References

- `docs/2026-05-07-canonical-workflow-gap-list-delta.md` — names
  this as A1 prerequisite for QueryOrchestrator
- `docs/2026-04-22-r3-threat-model.md` — R3 threat-modeling
  methodology this document follows
- `docs/2026-05-06-prsm-prov-1-threat-model-addendum.md` §§3.16–3.17
  — pattern for cross-node trust models
- `prsm/marketplace/reputation.py` — `ReputationTracker` (existing)
- `prsm/compute/tee/dp_noise.py` — DP noise primitives (existing)
- `prsm/economy/web3/stake_manager.py` — StakeBond client (existing)
