# Phase 4: Wallet SDK & Consumer Onboarding — Design + TDD Plan

**Date:** 2026-04-22
**Target execution:** Q4 2026 (per `docs/2026-04-10-audit-gap-roadmap.md` Phase 4). Earliest viable Q3 2026 (post-Phase-1.3-mainnet); latest acceptable Q2 2027 before Phase 5 engagement.
**Status:** Combined design + TDD plan drafted ahead of execution. Follows Phase 7 / Phase 8 pattern.
**Depends on:**
- Phase 1.3 mainnet deploy (FTNS token live on Base, hardware-gated).
- Phase 3 marketplace orchestrator.
- PRSM-GOV-1 Foundation formation (needed if embedded-wallet vendor requires a Foundation contract).

---

## 1. Context & Goals

Today's PRSM onboarding requires CLI, raw Ed25519 keys, direct FTNS contract interaction via web3.py scripts. This is defensible for developers kicking tires; it is a hard stop for any consumer-facing rollout or investor-facing demo. The Vision doc §14 crypto-UX mitigation frames the target: **a PRSM user should sign in the way they sign in to Coinbase or Stripe** — email or passkey, no seed phrase surfaced, smart-wallet abstraction handling nonce + gas.

Phase 4 ships the first production-ready consumer onboarding. It is deliberately scoped to the onboarding path only; in-app FTNS-earning flows, payout flows (Phase 5), and creator tooling are downstream.

### 1.1 Non-goals for Phase 4

- **Not a fiat on-ramp.** Phase 5 handles USD↔FTNS. Phase 4 assumes users already hold FTNS or receive it from contributions.
- **Not a new wallet product.** PRSM doesn't build a wallet. It integrates best-of-breed wallet vendors.
- **Not a custody solution.** PRSM is non-custodial; keys live in user-controlled wallets.
- **Not a mobile-app deliverable.** Web onboarding first; mobile is Phase 4.x if/when the mobile-app phase lands.
- **Not a KYC integration.** Phase 5 scope.

### 1.2 Backwards compatibility

- CLI + raw-key onboarding remains supported indefinitely for power users and CI/testing. Phase 4 adds a consumer-friendly path alongside, does not replace the existing one.
- Existing Ed25519 PRSM node identity remains the protocol-level primitive. Phase 4's wallet layer signs an authorization linking a user wallet address to a PRSM node identity; the PRSM signing path underneath is unchanged.

---

## 2. Scope

### 2.1 In scope

**Frontend:**
- Web onboarding flow at `prsm/interface/onboarding/` (rewritten).
- Wallet connection screen supporting Coinbase Wallet SDK primary + WalletConnect v2 fallback.
- Embedded-wallet vendor integration (one selected from Privy / Web3Auth / Magic.link — see §8.1).
- Identity-binding screen: connect wallet → sign binding attestation → server links wallet address to PRSM node ID.
- USD-equivalent display wrapper throughout UX; raw FTNS only on explicit hover / expand.

**Backend:**
- `prsm/interface/onboarding/wallet_binding.py` — verifies wallet-signed attestations, binds to PRSM node identity on-chain or off-chain (choice in §8.2).
- `prsm/interface/api/wallet_api.py` — REST endpoints for the onboarding flow.
- FTNS-denomination conversion utility for USD-equivalent displays.

**Smart contracts:**
- Optionally: a small `IdentityBinding.sol` contract if we choose the on-chain binding path (§8.2). Stores the (wallet_address → node_id) mapping with event for off-chain indexers.

**Configuration:**
- Wallet SDK project IDs / API keys in env config.
- Default FTNS allowance amounts for first-time users.

### 2.2 Out of scope

- In-app FTNS earning mechanisms (already in other modules; no Phase 4 change).
- Withdrawal / payout flows (Phase 5).
- Creator tooling for publishing content (separate scope).
- Mobile app.
- Account recovery flows beyond what the embedded-wallet vendor provides.

### 2.3 Deferred

- **Custom branded wallet.** If Phase 4 adoption shows vendor dependency is a problem, a Phase 4.x fork could ship a PRSM-branded wallet built on a self-hosted Privy or Magic fork. Defer until there's data.
- **Social recovery.** Beyond passkey / email recovery from the embedded-wallet vendor.

---

## 3. Protocol

### 3.1 Onboarding flow

```
User lands on web onboarding
        │
        ▼
"Sign in" screen: 3 options
  ├── Coinbase Wallet (primary)
  ├── Other wallet (WalletConnect v2)
  └── Email (embedded wallet — Privy/Magic/Web3Auth)
        │
        ▼
Wallet connects; user signs EIP-4361 (Sign-In With Ethereum) message
        │
        ▼
Server verifies signature; creates or retrieves PRSM node identity
        │
        ▼
User signs IdentityBinding attestation (one-time, binds wallet → node)
        │
        ▼
(Optional) wallet approves FTNS allowance for PRSM escrow pool
        │
        ▼
Onboarded. Landing redirects to dashboard.
```

### 3.2 Wallet-to-PRSM-identity binding

Two candidate designs (§8.2 open issue):

- **Off-chain binding:** server stores `(wallet_address, node_id_hex, wallet_signature)` tuple; verification by re-running signature check. Cheapest; no new contract; trust in the server's storage.
- **On-chain binding:** `IdentityBinding.sol` stores the mapping; anyone can query. Higher cost (~$0.01 per binding on Base); trust-minimized.

For MVP: off-chain binding. Migrate to on-chain if/when investor-due-diligence or regulatory review requires it.

### 3.3 USD-equivalent displays

PRSM's economic model is FTNS-denominated. Users don't want to reason in FTNS; they want to reason in USD. The UX convention:

- **All prices and balances display in USD** by default, with FTNS in a small subscript (`$2.50 · 0.125 FTNS`).
- **FTNS-only display** is opt-in via a user setting or an explicit "show in FTNS" toggle.
- Source of truth for FTNS→USD conversion: the same oracle the `RoyaltyDistributor` uses on-chain (Phase 1.3 already specifies this; no new oracle).

### 3.4 Embedded-wallet vendor semantics

Selected vendor (§8.1 decision) provides:

- Email / passkey login → generates a custodial or MPC-shared wallet → the user can sign transactions without ever seeing a seed phrase.
- Optional: export-to-non-custodial flow for users who later want full custody.

Trust model: user trusts the vendor's key-custody. PRSM is not in the custody path and does not see user keys. Vendor outages are a UX risk; PRSM mitigation is the Coinbase Wallet + WalletConnect alternative paths.

---

## 4. Data model

### 4.1 IdentityBinding record

```python
@dataclass(frozen=True)
class IdentityBinding:
    wallet_address: str        # 0x-prefixed Ethereum address
    node_id_hex: str           # PRSM Ed25519 node_id (32-char hex prefix)
    bound_at_unix: int
    wallet_signature: str      # EIP-4361 signature
    signing_message_hash: str  # keccak256 of the signing message
```

Stored in a new `identity_bindings` table (SQLite / Postgres, Foundation-operated).

### 4.2 Binding endpoints

```
POST /api/wallet/nonce                 → { nonce: str }
POST /api/wallet/sign-in               ← { wallet_address, signed_message }
                                        → { node_id_hex, is_new_user: bool }
POST /api/wallet/bind-identity         ← { wallet_address, node_id_hex, binding_signature }
                                        → { binding_id: str }
GET  /api/wallet/me                    → { wallet_address, node_id_hex, balance_ftns, balance_usd }
```

### 4.3 Optional `IdentityBinding.sol` (on-chain path)

```solidity
contract IdentityBinding {
    mapping(address => bytes32) public walletToNode;
    mapping(bytes32 => address) public nodeToWallet;

    function bind(bytes32 nodeId, bytes calldata signature) external {
        // verify signature over (msg.sender, nodeId) against a published signer
        walletToNode[msg.sender] = nodeId;
        nodeToWallet[nodeId] = msg.sender;
        emit Bound(msg.sender, nodeId);
    }

    event Bound(address indexed wallet, bytes32 indexed nodeId);
}
```

Only if §8.2 resolves to on-chain.

---

## 5. Integration points

### 5.1 Existing `prsm/interface/onboarding/`

Rewritten. Existing dashboard + auth modules unchanged.

### 5.2 `prsm/interface/api/`

New `wallet_api.py` module; `auth.py` extended to recognize wallet-signed sessions alongside existing session types.

### 5.3 FTNS oracle

Phase 4 reads, does not write. Oracle price feeds USD-equivalent displays.

### 5.4 PRSM node identity

Phase 4 ties wallet_address to the existing Ed25519 node_id. New-user flow generates a node_id server-side on first wallet sign-in. Returning-user flow retrieves existing binding.

---

## 6. TDD plan

**6 tasks**.

### Task 1: EIP-4361 Sign-In With Ethereum backend

- Verify wallet signatures against the published EIP-4361 format.
- Tests: valid sig ✓, expired nonce ✗, wrong-chain-id ✗, replay ✗, wallet-address mismatch ✗.
- Expected ~10 tests.

### Task 2: Identity-binding module + off-chain storage

- `prsm/interface/onboarding/wallet_binding.py`.
- Tests: new-user flow creates node_id + binding; returning-user flow fetches existing; binding is idempotent; wrong-signature rejects binding; query-by-wallet-address and query-by-node-id both work.
- Expected ~10 tests.

### Task 3: Frontend onboarding — Coinbase Wallet SDK primary

- Coinbase Wallet SDK integration with passkey login.
- Tests: playwright / end-to-end against a staging Base node.
- Expected ~5 tests.

### Task 4: WalletConnect v2 fallback + embedded-wallet vendor

- WalletConnect v2 for arbitrary external wallets.
- Selected embedded-wallet vendor integration (§8.1 decision).
- Tests: both paths complete a full onboarding without manual intervention.
- Expected ~8 tests.

### Task 5: USD-equivalent display wrapper

- Conversion utility reading from FTNS oracle.
- UX components with USD-default display.
- Tests: balance rendering at various FTNS values; price rendering for marketplace listings; FTNS-toggle persists across pages.
- Expected ~10 tests.

### Task 6: Review gate + merge-ready tag + pilot rollout

- Independent code review on the Phase 4 cumulative diff.
- `phase4-merge-ready-YYYYMMDD` tag.
- Staged rollout: internal Foundation team → early partners → public beta.
- Retirement-of-CLI-only onboarding is NOT a deliverable; CLI path stays.

---

## 7. Acceptance criterion

A consumer with zero prior crypto experience can: (a) land on PRSM's web onboarding; (b) sign in with email via the embedded wallet; (c) bind their wallet to a freshly-created PRSM node identity; (d) see their FTNS balance in USD; (e) land on the dashboard ready to earn or consume, in **under 90 seconds total onboarding time** (TTO metric).

---

## 8. Open issues

### 8.1 Embedded-wallet vendor selection

Three candidates:

- **Privy** — most polished UX, generous free tier, MPC + embedded-wallet export.
- **Web3Auth** — older, widely-integrated, non-custodial option.
- **Magic.link** — email + passkey, strong mobile support.

Selection based on: (a) non-custodial option for users who want it, (b) Base mainnet support, (c) pricing at PRSM scale, (d) SDK stability.

**Tentative recommendation:** Privy. Revisit at Task 4 kickoff.

### 8.2 Off-chain vs on-chain identity binding

- Off-chain: cheap, fast, Foundation owns the database.
- On-chain: trust-minimized, gas cost, more auditable.

MVP recommendation: off-chain. Migration path to on-chain is straightforward (the contract is ~100 lines).

### 8.3 Account recovery

Embedded-wallet vendor handles email / passkey recovery. For Coinbase Wallet + WalletConnect, recovery is the user's responsibility — PRSM shows a UX banner reminding users to back up their seed phrase.

### 8.4 EIP-4361 nonce storage

Nonces must be single-use to prevent replay. Backend keeps a short-TTL (5-minute) nonce store in Redis or equivalent. Foundation ops responsibility.

### 8.5 Smart wallet gas abstraction

Coinbase Wallet SDK supports ERC-4337 smart wallets with gas sponsorship. PRSM's initial stance: user pays their own gas. Gas sponsorship is Phase 4.x / Phase 5 scope once fiat-on-ramp economics justify subsidizing.

---

## 9. Dependencies + risk register

### R1 — Phase 1.3 mainnet deploy slippage

If FTNS is not live on Base mainnet by Q4 2026 start, Phase 4 blocks — there's no asset to display or sign for. Mitigation: Phase 1.3 is hardware-gated; tracking via project memory. Phase 4 scoping can proceed in parallel; implementation waits.

### R2 — Embedded-wallet vendor SLA

A vendor outage during launch period would block email-onboarding users. Mitigation: Coinbase Wallet + WalletConnect remain operational; UX surfaces alternative path clearly.

### R3 — Browser-wallet market share shifts

If Coinbase Wallet loses meaningful share by Q4 2026, the "primary" designation may need revisiting. Low probability; monitor.

---

## 10. Estimated scope

- **6 tasks.**
- **Expected LOC:** ~800 Python (backend) + ~1500 React/TypeScript (frontend).
- **Test footprint target:** +~40 tests across unit + e2e.
- **Calendar duration:** 3-4 weeks engineering + 1 week staged rollout.
- **Budget:** embedded-wallet vendor cost at MVP scale is typically $0-$500/month; negligible.

---

## 11. Changelog

- **0.1 (2026-04-22):** initial design + TDD plan. Promotes Phase 4 from master-roadmap stub to partner-handoff-ready scoping.
