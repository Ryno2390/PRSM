# Phase 5: Fiat On-Ramp & KYC — Design + TDD Plan

**Date:** 2026-04-22
**Target execution:** Q1 2027 (per `docs/2026-04-10-audit-gap-roadmap.md` Phase 5). **Compliance-gated** — legal pre-work may extend the timeline.
**Status:** Combined design + TDD plan drafted ahead of execution. Follows Phase 7 / Phase 8 pattern.
**Depends on:**
- Phase 1.3 mainnet deploy (FTNS live on Base).
- Phase 4 Wallet SDK (consumer onboarding path).
- PRSM-GOV-1 Foundation formation (required for the money-services-business (MSB) analysis and for Stripe production underwriting).
- Equity-investment-architecture legal track resolved (FTNS securities classification posture) — tracked in private repo.

---

## 1. Context & Goals

The meganode ROI story and operator economics require a round trip: earn FTNS → convert to USD → bank deposit. Today `prsm/economy/payments/` has Stripe + Coinbase scaffolding but no production credentials, no KYC, and no compliance review. Phase 5 ships the production fiat path, gated on the compliance + regulatory pre-work.

Phase 5 also ships the inbound direction (USD → FTNS) for consumers who want to pay for PRSM services without first contributing compute. That inbound flow is a **conversion facility**, not a token sale — legally critical per `PRSM_Tokenomics.md` §9.

### 1.1 Non-goals for Phase 5

- **Not a token sale.** FTNS is not sold; it is conversion-exchanged at oracle rate.
- **Not an exchange.** PRSM does not run an order book or hold user funds at rest beyond transaction duration.
- **Not a credit facility.** No FTNS lending, no leverage, no derivative products.
- **Not a stablecoin.** FTNS is not pegged to USD; Phase 5 exchanges at market rate.
- **Not a new KYC vendor.** PRSM integrates a selected third-party provider (Persona / Sumsub / Onfido), does not build KYC.

### 1.2 Backwards compatibility

- Existing FTNS earning (compute contributions, content royalties, staking bounties) unchanged.
- Existing Ed25519 node identity + wallet-binding from Phase 4 unchanged.
- Users who never use the fiat path see no change.

---

## 2. Scope

### 2.1 In scope

**Compliance + legal:**
- Selected KYC vendor integrated (choice in §8.1).
- Stripe production account + PCI scope review.
- MSB analysis for jurisdictions in operation (US-federal FinCEN + state-by-state).
- Howey analysis update under the equity-investment architecture per Tokenomics §9 — recalibrate FTNS securities-status conclusion at production scale.
- Compliance review checklist committed and reviewed by counsel.
- Terms of Service + Privacy Policy updates for fiat flows.

**Backend:**
- `prsm/economy/payments/` rewritten for production. Existing scaffolding retained where usable.
- KYC verification service wrapping the chosen vendor.
- Stripe integration for USD payouts to US bank accounts.
- Coinbase Commerce / Exchange API for FTNS↔USDC swap path.
- Withdrawal workflow state machine: requested → kyc_check → oracle_quote → swap_executed → stripe_payout → completed.
- Rate-limiting + anti-abuse (per-day + per-month caps).

**Frontend:**
- Withdrawal UI in `prsm/interface/onboarding/` adjacent to Phase 4 dashboard.
- Deposit UI for USD → FTNS conversion.
- KYC flow with vendor-hosted UI.
- Status tracker for in-flight withdrawals.

**Smart contracts:**
- **None expected.** FTNS movements use existing contracts (Phase 1.3 RoyaltyDistributor and Phase 7 StakeBond.claimBounty for earnings; standard ERC-20 transfers for internal). If a specific payout path requires a new contract, surfaced in Task 1 design review.

### 2.2 Out of scope

- International withdrawals beyond SEPA (Europe) / local rails for partner countries — phased per jurisdictional legal clearance.
- Business-account onboarding (enterprise payout flows).
- Tax reporting automation (Form 1099-MISC etc.) — deferred to a Phase 5.x compliance follow-up.
- Custody of user USDC for more than the swap-transaction window.

### 2.3 Deferred

- **Non-USD currencies.** Phase 5 ships USD only. EUR / GBP / other deferred to Phase 5.x per regulatory capacity.
- **Direct fiat-to-FTNS without KYC** — infeasible under MSB regulations regardless of product design.

---

## 3. Protocol

### 3.1 Withdrawal flow (FTNS → USD bank)

```
User requests withdrawal (X FTNS → bank account)
        │
        ▼
KYC check (vendor SDK; skipped if user is already verified)
        │
        ▼
FTNS oracle quote (valid for 2 minutes)
        │
        ▼
User confirms quote
        │
        ▼
PRSM swap: FTNS → USDC via Coinbase Exchange API
        │
        ▼
USDC → USD via Coinbase Commerce or direct Stripe payout
        │
        ▼
Stripe payout to user's linked bank account
        │
        ▼
Withdrawal complete; event logged; user email confirmation
```

### 3.2 Deposit flow (USD → FTNS)

```
User submits deposit (USD amount) via Stripe card / ACH
        │
        ▼
KYC check
        │
        ▼
Stripe charge captured
        │
        ▼
FTNS oracle quote
        │
        ▼
User confirms
        │
        ▼
PRSM swap: USDC → FTNS via Coinbase Exchange
        │
        ▼
FTNS credited to user's wallet (from Phase 4 binding)
        │
        ▼
Deposit complete; event logged; user email confirmation
```

### 3.3 KYC integration

- First-time fiat-flow user triggers the KYC flow.
- Vendor SDK handles document capture (passport / state ID / driver license) + liveness check + identity verification.
- Result stored as (`kyc_status`, `kyc_vendor_id`, `kyc_completed_at`, `tier`) per user record.
- Tier 1: ≤$1,000/month aggregate in both directions (low-friction). Tier 2: $1,000-$10,000/month. Tier 3: >$10,000/month requires enhanced-due-diligence (EDD).

### 3.4 Oracle + pricing

- FTNS/USD rate from the oracle already specified in Phase 1.3 (RoyaltyDistributor uses the same). Phase 5 reads; does not set.
- Quote valid for 2 minutes; if expired, user re-quotes before proceeding.
- PRSM takes a **2% spread** on fiat exchanges to fund operations + regulatory capital. Spread is transparent to the user in the quote display.

### 3.5 Anti-abuse + rate limits

- Per-user: $2k/day withdrawal cap at Tier 1; $25k/day at Tier 2; configurable at Tier 3.
- Per-user: $5k/day deposit cap at Tier 1; $50k/day at Tier 2; configurable at Tier 3.
- Velocity checks: sudden spike → hold for manual review.
- Sanctions screening: OFAC + SDN list checks on all KYC-verified users.
- Geographic restrictions: users from prohibited jurisdictions blocked at sign-up (list maintained by Foundation compliance officer).

---

## 4. Data model

### 4.1 Withdrawal record

```python
@dataclass(frozen=True)
class WithdrawalRequest:
    request_id: str
    user_id: str
    wallet_address: str
    amount_ftns_wei: int
    quoted_usd_cents: int
    quote_expires_at_unix: int
    stripe_payout_destination: str
    kyc_tier: int
    status: str  # requested | kyc_check | oracle_quote | swap_executed | stripe_payout | completed | failed
    created_at_unix: int
    updated_at_unix: int
    failure_reason: Optional[str]
```

### 4.2 KYC record

```python
@dataclass(frozen=True)
class KycRecord:
    user_id: str
    vendor_name: str  # "persona" | "sumsub" | "onfido"
    vendor_verification_id: str
    status: str  # pending | passed | failed | expired
    tier: int  # 1, 2, 3
    completed_at_unix: int
    expires_at_unix: Optional[int]
```

### 4.3 Endpoints

```
POST /api/fiat/kyc/start              → { vendor_sdk_session_token }
POST /api/fiat/withdraw               ← { amount_ftns_wei }
                                       → { quote_usd_cents, quote_expires_at, request_id }
POST /api/fiat/withdraw/confirm       ← { request_id }
                                       → { status }
GET  /api/fiat/withdraw/{request_id}  → { full withdrawal record }

POST /api/fiat/deposit                ← { amount_usd_cents }
                                       → { stripe_payment_intent_id, quote_ftns_wei, quote_expires_at }
POST /api/fiat/deposit/confirm        ← { stripe_payment_intent_id }
                                       → { status }
```

---

## 5. Integration points

### 5.1 Phase 4 Wallet SDK

Phase 5 builds on Phase 4's wallet binding. Withdrawal destinations are the user's bound wallet address (FTNS side); deposits credit that wallet. Bank account linkage is Stripe-managed, bound to the user's Phase 4 session.

### 5.2 FTNS oracle

Read-only consumer. Oracle is shared with Phase 1.3 RoyaltyDistributor.

### 5.3 Existing `prsm/economy/payments/` scaffolding

Existing Stripe and Coinbase clients refactored for production credentials + production error handling. Test fixtures retained.

### 5.4 Foundation legal/compliance function

Phase 5 is compliance-gated in a way prior phases were not. Foundation must have:
- A named Compliance Officer (PRSM-GOV-1 §4.5 or its successor).
- MSB registration where required (FinCEN + state by state for US operations).
- A vendor-contracting function for KYC + Stripe production agreements.

If Foundation isn't sufficiently formed to hold these commitments by Phase 5 kickoff, Phase 5 waits.

---

## 6. TDD plan

**7 tasks**.

### Task 1: Compliance review + legal sign-off

- Update Howey analysis under equity-investment architecture per Tokenomics §9.
- MSB analysis: FinCEN registration + state-by-state for initial operating states.
- OFAC / sanctions compliance plan.
- Terms of Service + Privacy Policy updates.
- Legal sign-off document filed.
- **Output:** compliance sign-off memo; risk register updates.

### Task 2: KYC vendor integration

- Selected vendor (§8.1) integration.
- User-tier assignment logic.
- Sanctions screening hook.
- Tests: happy-path verification; failure cases (doc rejected, liveness failed, sanctions hit); tier upgrades; tier downgrades on expiry.
- Expected ~15 tests.

### Task 3: Stripe production integration

- Production API keys + PCI scope review.
- Payout flow (USD → bank).
- Charge flow (card → USD).
- Webhook handling for async events.
- Tests: both flows against Stripe test-mode; error paths (declined card, failed payout); refund path.
- Expected ~15 tests.

### Task 4: Coinbase Exchange FTNS↔USDC swap

- Production API integration.
- Order placement + confirmation.
- Error recovery (partial fill, order rejection).
- Rate limits + retries.
- Tests: against Coinbase sandbox; real-time quote fetching; failure injection.
- Expected ~10 tests.

### Task 5: Withdrawal + deposit state machines

- Orchestration service wiring KYC + Stripe + Coinbase + FTNS contract calls.
- State-machine persistence (Postgres or equivalent).
- Retry + recovery logic.
- Tests: happy paths end-to-end (mocked externals); failure at every state transition with correct recovery behavior.
- Expected ~20 tests.

### Task 6: Frontend + user-facing UX

- Withdrawal UI; deposit UI; KYC launcher; status tracker.
- Rate limit + velocity-check messaging.
- Email notifications.
- Tests: e2e against staging.
- Expected ~10 tests.

### Task 7: Review gate + phased production rollout

- Independent code review.
- Compliance audit sign-off.
- Staged rollout: Foundation-internal → ≤10 beta users → ≤100 beta users → open signups with tier-1 cap.
- `phase5-merge-ready-YYYYMMDD` tag + retroactive compliance review sign-off.

---

## 7. Acceptance criterion

- A KYC-verified Tier 1 user can withdraw $100 from their FTNS balance to their US bank in under 48 hours (typically <1 hour) with zero manual intervention from the Foundation side.
- A user can deposit $100 USD via card and receive FTNS credited to their wallet within 15 minutes.
- All KYC-verified users pass OFAC / SDN sanctions screening; any screening hit blocks the flow with a human-review escalation.
- The Foundation compliance officer can audit any in-flight or historical transaction end-to-end from a single internal dashboard.

---

## 8. Open issues

### 8.1 KYC vendor selection

Three candidates; criteria: coverage (document types + liveness), cost at scale, API stability, fraud-rate track record.

- **Persona** — strong US coverage, modern API, mid-range pricing.
- **Sumsub** — strong international coverage, lower cost at volume.
- **Onfido** — older player, enterprise-focused, slower roll-out.

**Tentative recommendation:** Persona for US-only launch; re-evaluate for international expansion in Phase 5.x.

### 8.2 Jurisdictional scope at launch

US-only at launch is simplest. International expansion requires per-jurisdiction MSB / PSP analysis. Likely ordering: US → Canada → UK → EU → APAC, driven by Foundation capacity.

### 8.3 Spread amount

2% is a placeholder. Actual spread covers: (a) Coinbase Exchange fees (typically 0.1-0.4%), (b) Stripe payout fees (0.25% + $0.25), (c) regulatory capital + compliance operations overhead (~1-1.5%). Audit Task 1 output for spread calibration.

### 8.4 Custody handling during swap window

Between "FTNS debited from user" and "USD arriving in Stripe" there is a ~5-30 second swap window where PRSM holds USDC. Technically this is short-term custody. MSB analysis (Task 1) must confirm the swap-window exposure is below materiality thresholds for safekeeping registration.

### 8.5 Tax reporting

US tax reporting for users' fiat-on-ramp activity is out of scope for the v1 Phase 5 deliverable but the data model (§4.1 WithdrawalRequest logs) records everything needed. A Phase 5.x deliverable can add 1099-MISC generation if volume warrants.

### 8.6 Foundation capital reserve

MSB operations require reserves; Stripe underwriting may require a liquidity buffer; compliance breach insurance is standard. These are Foundation treasury decisions, not engineering ones, but engineering scoping must surface the dollar range so treasury plans accordingly. Rough estimate: $100k-$500k locked reserve for MVP operations.

---

## 9. Dependencies + risk register

### R1 — Legal / compliance review surfaces a blocker

Howey or MSB analysis may conclude FTNS conversion is restricted in ways that invalidate the Phase 5 design. Mitigation: Task 1 runs first; design revisions are cheap at scoping stage.

### R2 — Stripe production underwriting rejection

Stripe underwrites each MSB-adjacent business individually. Rejection would block the fiat leg entirely. Mitigation: pre-application conversations with Stripe during Task 1; alternate payout provider identification (Plaid + ACH direct) as fallback.

### R3 — Coinbase Exchange API disruption

Coinbase sandbox/production disruption during launch. Mitigation: Kraken or Binance.US as alternate FTNS↔USDC venue; design Task 4 provider-abstract.

### R4 — Regulatory environment shift

US stablecoin / token regulation is active. A regulatory change between scoping and launch could require redesign. Mitigation: monitor; flag immediately when published.

### R5 — Foundation formation incomplete

Foundation entity + compliance officer required; if not formed, Phase 5 cannot launch. Mitigation: Foundation formation deadline tracked in PRSM-GOV-1 §8.4.

### R6 — KYC vendor change mid-flight

Vendor acquisition / pricing change / outage. Mitigation: design Task 2 vendor-abstract; migration cost ~1 engineer-month.

---

## 10. Estimated scope

- **7 tasks.**
- **Expected LOC:** ~1500 Python + ~1000 frontend.
- **Test footprint target:** +~70 tests unit + e2e.
- **Calendar duration:** 4-6 weeks engineering + legal lead time (unknown; potentially 4-8 weeks pre-engineering).
- **Budget:** vendor costs ($200-$2000/month at MVP scale); legal engagement ($25k-$100k); reserve capital ($100k-$500k).

---

## 11. Changelog

- **0.1 (2026-04-22):** initial design + TDD plan. Promotes Phase 5 from master-roadmap stub to partner-handoff-ready scoping.
