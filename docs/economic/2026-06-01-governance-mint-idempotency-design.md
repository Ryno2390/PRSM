# Governance system-mint durable idempotency — design scope

**Status:** IMPLEMENTED — **Design A** shipped as sprint 912
(`prsm/economy/governance/token_distribution.py` +
`prsm/economy/tokenomics/database_ftns_service.py`,
`tests/unit/test_sprint_912_governance_mint_idempotency.py`). The RFC below
is retained as the design record.
**Date:** 2026-06-01
**Trigger:** The money-path adversarial review's remaining (deferred) item.
Sprint 908 closed the *critical* part (the self-select-`CORE_TEAM` authz
hole + the shared eval-once UUID). This scopes the remaining gap: the
governance system-mint has **no durable idempotency**, so a node restart
re-mints, and a retried contribution re-pays.

---

## 1. The bug, precisely

Two system-mint paths in `prsm/economy/governance/token_distribution.py`,
both minting via `DatabaseFTNSService.create_transaction(from_user_id=None,
...)`:

- **`distribute_contribution_rewards(user_id, contribution_type,
  contribution_reference, ...)`** — mints a reward and records it with
  `reference_id=str(distribution.distribution_id)`. There is **no check**
  that this `(user, contribution_type, contribution_reference)` was already
  rewarded → calling it twice with the same contribution mints twice.
- **`activate_governance_participation(user_id, tier, ...)`** — mints the
  tier's initial allocation, guarded only by the **in-memory**
  `self.governance_activations[user_id]` dict (lines ~175, ~230). After a
  daemon restart that dict is empty, so re-activation re-mints the full
  allocation (COMMUNITY 1K … CORE_TEAM 100K FTNS).

Root cause for both: `reference_id` is `str(distribution_id)` — now
per-instance unique (sp908), but **random per call**, so it can't anchor
"have I already done this logical mint?". The only dedup is process-local
+ non-durable.

**Deployment context (bounds the fix):** minting runs in a single daemon;
the existing `_distribution_lock` / `_activation_lock` (asyncio) already
serialize same-process concurrency. The unhandled axis is **durability
across restarts** (and logical retries), not multi-process contention.

## 2. The idempotency keys

- contribution reward → `gov-contrib:{user_id}:{contribution_type.value}:{contribution_reference}`
- governance activation → `gov-activate:{user_id}`

Both deterministic + stable across retries/restarts (unlike the random
`distribution_id`).

## 3. Design A — deterministic reference + query-under-lock (RECOMMENDED)

No schema change, no migration. Leverages the existing indexed
`ftns_transactions.reference_id` column + the in-place asyncio locks.

1. Set the mint's `reference_id` to the **deterministic key** above
   (instead of `str(distribution_id)`).
2. Add `DatabaseFTNSService.find_confirmed_transaction(reference_id,
   transaction_type) -> Optional[FTNSTransaction]` (one indexed SELECT).
3. In each distributor path, **inside the existing lock**, query first; if
   a confirmed mint with that key exists → return the existing
   distribution/activation (idempotent no-op, **no second mint**).
4. Also persist a durable activation lookup so
   `activate_governance_participation` rebuilds its guard from the DB on a
   cold process (or simply relies on the pre-mint query).

**Durability:** cross-restart ✅ (the DB query finds the prior mint);
same-process concurrency ✅ (asyncio lock). **Cross-PROCESS concurrent**
mint of the identical key ❌ (a TOCTOU window between query + insert) — but
that requires two daemons minting the same governance event simultaneously,
which is not the deployment model.

**Cost:** LOW. ~1 query method + 2 distributor edits + the deterministic
`reference_id`. No model, no migration. **Effort: ~half a sprint, TDD'd.**

## 4. Design B — dedicated atomic claim table (rigorous alternative)

Fully atomic (handles multi-process), at the cost of a schema change.

1. New model `IdempotencyClaim(claim_key String(255) PRIMARY KEY,
   claimed_at DateTime)` → table `ftns_idempotency_claims` (in
   `prsm/economy/tokenomics/models.py`, picked up by `create_all` for
   dev/test).
2. **Alembic migration** to create the table in prod (the repo has 13
   versions + a migrations pipeline).
3. `create_transaction` gains optional `idempotency_key`: in the SAME
   session, `INSERT` the claim **before** the mint; on `IntegrityError`
   (already claimed) short-circuit → return the existing transaction (no
   double-mint). Claim + mint commit atomically (a crash between them
   rolls back both — no claimed-but-unminted gap). The `IntegrityError`
   catch pattern already exists in `create_wallet`.
4. Distributors pass `idempotency_key=<deterministic key>`.

**Durability:** full — atomic across processes via the PK constraint.
**Cost:** MEDIUM. New model + migration + `create_transaction` change + 2
distributor edits + tests. **Effort: ~1 sprint.**

## 5. Recommendation

**Design A.** It closes both stated bugs (restart re-mint + contribution
re-pay) for the actual single-daemon deployment, with no schema change and
low blast radius. Design B's extra guarantee (atomic multi-process claim)
addresses a contention axis that doesn't exist today; pursue it only if/when
governance minting becomes multi-process. (If the audit posture demands the
strongest possible guarantee regardless of deployment, go straight to B —
it's not much more work and is the textbook answer.)

Either way, this also lets `distribute_contribution_rewards` /
`activate_governance_participation` return the *existing* record on a
duplicate call, so callers see an idempotent success rather than a silent
double-mint or an error.

## 6. Test plan (both designs)

- `distribute_contribution_rewards` twice with the same
  `(user, type, reference)` → mints **once**; second returns the existing
  distribution; `create_transaction` called once.
- Restart simulation: clear the in-memory guard, re-call
  `activate_governance_participation(user, tier)` → **no second mint**
  (the DB query / claim catches it).
- Distinct contributions (different `reference`) each mint (no over-dedup).
- Distinct users activating the same tier each mint (no cross-user dedup).
- (Design B) concurrent claim of the same key → exactly one wins
  (IntegrityError on the loser, no double-mint).

## 7. Out of scope

- The `CORE_TEAM` self-select authz hole + shared-UUID default — already
  fixed (sp908).
- Multi-process mint contention beyond Design B's claim table.
