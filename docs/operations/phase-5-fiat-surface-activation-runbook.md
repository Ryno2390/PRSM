# Phase 5 fiat-surface activation runbook

Operator-facing step-by-step for activating the Phase 5 fiat surface — the
USD → USDC → FTNS on-ramp and FTNS → USDC → USD off-ramp built across the
Phase 5 commission-ready arc (sprints 276–286) and the sp848–881 productionization
arc. Use this when Coinbase CDP credentials arrive, the Persona KYC vendor
agreement signs, the Coinbase Pay project registers, and the Foundation has
ratified (and run) the Aerodrome pool-seed ceremony.

This doc is read by operators activating real-money flows and by reviewers
doing technical diligence. It is command-first and honest about what is NOT
yet live (see the two externalities at the end: the Aerodrome pool ceremony,
and the Paymaster F23 blocker).

---

## 0 — The readiness model: commissioned vs adapter_wired vs live_exec

Before any commands, internalize the three-signal honest-readiness model
introduced in sp848. **Every Phase 5 surface reports three booleans, and they
are NOT the same thing.** An operator who reads `commissioned: true` and assumes
"ready to execute" will be wrong.

| Signal | What it proves | What it does NOT prove |
|---|---|---|
| `commissioned` | The required env vars are present and non-placeholder. Purely a static check of `os.environ`. | That the vendor SDK loaded, that the PEM parsed, or that any real call has succeeded. |
| `adapter_wired` | A vendor SDK backend was successfully dependency-injected at runtime. Env vars were present AND the SDK imported AND the key material parsed. | That a real end-to-end transaction has been observed in this deployment. |
| `live_exec` | The surface has demonstrably worked end-to-end here. | (this is the strong signal) |

Why the distinction is load-bearing: env vars can be set (`commissioned: true`)
while `adapter_wired: false` — for example when the SDK import fails, or when the
key is still a placeholder string (`REPLACE_WITH...`) that fails PEM parsing. The
adapter degrades gracefully to `PENDING_COMMISSION` instead of crashing. The two
signals let an operator distinguish "env configured" from "actually ready to
execute."

How `live_exec` is computed per surface:

- **KYC / WaaS / Onramp**: `live_exec = commissioned AND adapter_wired`.
- **Paymaster**: `live_exec = (sponsorships > 0)` — i.e. at least one real
  user-op has actually been sponsored. (Not yet achievable in production — see
  the Paymaster F23 externality.)
- **Aerodrome**: `live_exec = pool_configured` — i.e.
  `AERODROME_USDC_FTNS_POOL_ADDRESS` is set (only true post-ceremony).

The aggregate rollup across the 5 surfaces:

- `live_count == 5` → overall **READY**
- `live_count == 0` → overall **NOT_READY**
- `1 ≤ live_count < 5` → overall **PARTIAL**

`GET /wallet/phase5/status` (and `prsm node phase5-status`) returns this grid:
each surface's `commissioned` / `adapter_wired` / `live_exec` plus per-surface
detail (`vendor`, `pool_configured`), the `overall`, `live_surface_count`, and
`total_surface_count`.

---

## Pre-flight: what must be in place BEFORE running this runbook

| Gate | Owner | Evidence |
|---|---|---|
| Foundation funding decision ratified | Foundation Council | Latest PRSM-CR covering Phase 5 commission |
| Coinbase CDP account provisioned | Foundation Ops | API Key Name + Ed25519 Private Key (PEM) + (optionally) Wallet Secret in hand |
| Coinbase Pay project registered | Foundation Ops | `COINBASE_PAY_APP_ID` (public, not a secret) in hand |
| Persona KYC vendor agreement signed | Foundation Ops | API key + inquiry template ID + webhook secret in hand |
| Aerodrome USDC↔FTNS pool seeded | Foundation Safe (multi-sig) | Pool contract address from the on-chain ceremony; verify on Basescan |
| Base mainnet RPC endpoint available | Operator | Working `BASE_RPC_URL` (Alchemy / Infura / QuickNode / public) |

If any of these aren't in place, **STOP** — the surfaces will degrade to
`PENDING_COMMISSION` rather than crash, but you will get a `PARTIAL` or
`NOT_READY` overall and the corresponding flows will not execute. Run
`prsm node fiat-readiness` first (it requires no daemon) to confirm the local
env is consistent before bringing anything up.

---

## Step 1 — Canonical env-var set

Add these to `/etc/prsm/bootstrap-server.env` (or wherever your operator
config lives). All credential values are **secrets**: the file must be
`chmod 640` and owned `root:prsm` (or equivalent service group). Grouped by
surface. For each var, the "unset →" note states the documented default behavior.

```bash
# ── Coinbase CDP WaaS (MPC wallet provisioning, sp276/851) ──
# Wires CoinbaseWaaSClient + onramp session token minting.
COINBASE_CDP_API_KEY_NAME=organizations/<ORG_ID>/apiKeys/<KEY_ID>
# unset → wallets stay PENDING_COMMISSION; adapter_wired=False.
COINBASE_CDP_API_KEY_PRIVATE=-----BEGIN EC PRIVATE KEY-----
...
-----END EC PRIVATE KEY-----
# unset → wallets stay PENDING_COMMISSION. Placeholder PEM ("REPLACE_WITH...")
#   fails parse → graceful fallback to PENDING_COMMISSION.
COINBASE_CDP_WALLET_SECRET=<ECDSA P-256 PKCS8 DER, base64>
# ECDSA P-256 wallet secret from CDP "Server Wallets". Used for the
#   X-Wallet-Auth header on /platform/v2/evm/* calls.
#   OPTIONAL → unset → WaaS backend uses API-key auth alone.

# ── Coinbase CDP Paymaster (ERC-4337 gas sponsorship, sp277/850) ──
# NOTE: paymaster live_exec is currently BLOCKED — see the F23 externality.
COINBASE_CDP_PAYMASTER_ENDPOINT=https://api.developer.coinbase.com/rpc/v1/base/<TOKEN>
# unset → PaymasterClient.from_env() endpoint=None; sponsor ops stay
#   PENDING_COMMISSION (dry_run-by-default).
COINBASE_CDP_PAYMASTER_API_KEY=<paymaster-api-key>
# Kept for parity; CDP v2 embeds the token in the URL path so header-auth is
#   not load-bearing yet. unset → is_commissioned() returns False.
PRSM_PAYMASTER_POLICY_ID=<policy-uuid-from-coinbase-portal>
# OPTIONAL per-op spend caps / rate limits. unset → None; no per-op policy.
PRSM_ERC4337_ENTRY_POINT=0x0000000071727De22E5E9d8BAf0edAc6f37da032
# OPTIONAL. unset → defaults to the canonical Base-mainnet EntryPoint above.

# ── Coinbase Pay on-ramp (user-facing widget, sp853) ──
COINBASE_PAY_APP_ID=<coinbase-pay-project-id>
# PUBLIC identifier (not a secret; no server-side signing). Wires
#   build_onramp_url(). unset → build_onramp_url returns None; the endpoint
#   surfaces PENDING_COMMISSION with guidance to register a Pay project.

# ── Persona KYC vendor (sp280/849 — pick ONE vendor) ──
KYC_VENDOR=persona                       # or "onfido" | "plaid" | "mock"
# Normalized lowercase. unset → ""; adapter returns PENDING_COMMISSION.
KYC_VENDOR_API_KEY=<persona-api-bearer-token>
# unset → None; adapter returns PENDING_COMMISSION until key lands.
PERSONA_TEMPLATE_ID=<persona-inquiry-template-id>
# BASE (Tier 1) inquiry template. Required for initiate_session. unset →
#   from_env() returns None; KYCClient falls back to PENDING_COMMISSION
#   (adapter_wired=False for Persona).
PERSONA_ENHANCED_TEMPLATE_ID=<persona-enhanced-inquiry-template-id>
# OPTIONAL (sp883). ENHANCED (Tier 2/3) template collecting proof-of-address
#   + source-of-funds, for >$1k transaction limits. Routed when
#   initiate(..., level="enhanced"). unset → enhanced inquiries fall back to
#   the base template + a warning (proof-of-address NOT collected).
PERSONA_WEBHOOK_SECRET=<persona-webhook-secret>
# HMAC signing secret for inbound Persona webhook verification
#   (t=<ts>,v1=<hmac-sha256> per sp283). unset → webhook handler rejects all
#   inbound POSTs (no_auth constraint).

# ── Aerodrome (USDC↔FTNS pool quoter, sp279) ──
BASE_RPC_URL=https://mainnet.base.org   # or Alchemy / Infura / QuickNode
# Wires eth_call balance queries (sp862) + Aerodrome pool reads (sp279).
#   unset → defaults to https://mainnet.base.org (free but rate-limited).
AERODROME_USDC_FTNS_POOL_ADDRESS=0x<pool-contract-address>
# unset → is_configured() False; pool reads return None; NO swap quotes
#   until the post-ceremony pool address is pasted (see Aerodrome externality).

# ── Persistence directories (sp857/860; :memory: = in-memory only) ──
PRSM_WAAS_STORE_DIR=/var/lib/prsm/waas-wallets
# unset → ~/.prsm/waas-wallets. ":memory:" → no disk persistence.
PRSM_KYC_STORE_DIR=/var/lib/prsm/kyc
# unset → ~/.prsm/kyc-records. ":memory:" → in-memory only.
PRSM_ONRAMP_FUNNEL_DIR=/var/lib/prsm/onramp-funnel
# unset → ~/.prsm/onramp-funnel. ":memory:" → no cross-restart funnel
#   observability.
PRSM_WAAS_NETWORK=base-mainnet           # or base-sepolia
# unset → "base-mainnet".

# ── On-ramp completion webhook (outbound notifications, sp874) ──
PRSM_ONRAMP_COMPLETION_WEBHOOK_URL=https://your-system.example/hooks/onramp
# unset → notifier is a no-op (CONFIRMED transitions still happen, no side
#   effect).
PRSM_ONRAMP_COMPLETION_WEBHOOK_SECRET=<hmac-sha256-secret>
# Signs outbound POSTs as X-PRSM-Signature: t=<unix>,v1=<hex> (sp283 pattern).
#   unset → None; outbound POSTs carry no signature header.
PRSM_ONRAMP_COMPLETION_WEBHOOK_LOG_DIR=/var/lib/prsm/onramp-completion-deliveries
# unset → ~/.prsm/onramp-completion-deliveries. ":memory:" → no audit trail.

# ── Auto-sweep worker (background settlement detection, sp878) ──
PRSM_FUNNEL_AUTO_SWEEP_INTERVAL_S=300
# Float seconds. unset/empty → auto-sweep DISABLED (interval=0.0). Values
#   under 60 clamp to the 60s minimum (settlement takes minutes; faster is RPC
#   waste). Non-numeric → logs a warning and disables (safe to typo).

# ── KYC tier limits (rolling-window USD caps, sp285/884) ──
PRSM_KYC_TIER_LIMIT_BASIC_USD=1000
PRSM_KYC_TIER_LIMIT_ENHANCED_USD=10000
# Per-tier rolling-window USD ceilings. Defaults: basic $1,000, enhanced
#   $10,000 (FinCEN MSB convention). The tier is read from the user's KYC
#   record level (basic|enhanced — see PERSONA_ENHANCED_TEMPLATE_ID); the
#   rolling total comes from the sp282 compliance ring. ENFORCED on
#   /wallet/onramp/execute (sp884): an unverified user → 403 kyc_required;
#   a verified user whose requested + rolling total would exceed their tier
#   limit → 403 tier_limit_exceeded (basic users get an upgrade-to-enhanced
#   hint). The onramp/offramp QUOTES surface these as advisory flags only.
#   A raw destination_address (no PRSM identity) is NOT gated — operator
#   responsibility.
```

**DO NOT** set this in production:

```bash
PRSM_KYC_WEBHOOK_VERIFY_DISABLED=1       # bypasses inbound webhook signature check
```

It exists for **test/dev environments only**. Setting it in a commissioned
environment turns off HMAC verification on inbound KYC vendor webhooks.

**Sp888 — the webhook endpoint FAILS CLOSED.** `/wallet/kyc/webhook/{vendor}`
is the only writer that can mint a `VERIFIED` KYC record (→ auto-provision +
raised tier limits). It accepts a webhook ONLY when the signature verifies
(the vendor secret is set) OR `PRSM_KYC_WEBHOOK_VERIFY_DISABLED=1` is
explicitly set. With neither, it returns **503** and refuses to process the
unsigned webhook. In production you MUST set `PERSONA_WEBHOOK_SECRET` (or the
vendor equivalent) — there is no silent unsigned pass-through.

---

## Step 2 — Create persistence directories

```bash
sudo mkdir -p /var/lib/prsm/waas-wallets /var/lib/prsm/kyc /var/lib/prsm/onramp-funnel /var/lib/prsm/onramp-completion-deliveries && sudo chown prsm:prsm /var/lib/prsm/waas-wallets /var/lib/prsm/kyc /var/lib/prsm/onramp-funnel /var/lib/prsm/onramp-completion-deliveries && sudo chmod 750 /var/lib/prsm/waas-wallets /var/lib/prsm/kyc /var/lib/prsm/onramp-funnel /var/lib/prsm/onramp-completion-deliveries
```

These hold sensitive PII (KYC outcomes), wallet records, funnel intents, and
the webhook delivery audit trail. Regulators (AUSTRAC, FinCEN, IRS) expect
5–7yr retention — back these directories up to encrypted off-host storage on
the same cadence as `/etc/letsencrypt/`. Use the `:memory:` sentinel only in
dev: it disables cross-restart observability and leaves no audit trail.

---

## Step 3 — Restart the PRSM service

```bash
sudo systemctl restart prsm-bootstrap   # or prsm-node — whichever runs the fiat surface
```

On boot, the funnel auto-sweep worker (sp878) spins up a background asyncio
task **only if** `PRSM_FUNNEL_AUTO_SWEEP_INTERVAL_S` resolves to a positive
value. If unset/empty, auto-sweep stays disabled and you sweep manually
(Step 6 / `POST /wallet/onramp/sweep`).

---

## Step 4 — Verify readiness from the daemon

The single canonical check is `prsm node phase5-status`. It returns the full
3-signal grid across all 5 surfaces and an overall READY / PARTIAL / NOT_READY.

```bash
prsm node phase5-status --format json
```

Interpretation:

- `overall: READY` — all 5 surfaces have `live_exec: true`. (Note: this is not
  reachable today because Paymaster `live_exec` is blocked — see F23. A healthy
  pre-paymaster deployment will read `PARTIAL`.)
- `overall: PARTIAL` — some surfaces live. Read the per-surface grid to see
  which. A surface with `commissioned: true, adapter_wired: false` means the
  env is set but the SDK/key did not load — re-check the credential material.
- `overall: NOT_READY` — no surface is live; you are still pre-commission.

For a one-screen operator view that adds fleet treasury + the conversion funnel:

```bash
prsm node phase5-dashboard
```

For a pre-commission **local** check that needs no running daemon (reads env
vars directly):

```bash
prsm node fiat-readiness --format json
```

Exit code is 0 for OK / WARN-only, non-zero if any ERROR finding. Use it in
ops automation before activation.

---

## Step 5 — Smoke-test the canonical surfaces

### 5a — Aerodrome swap quote (read-only, no spend)

```bash
curl -s -X POST "http://127.0.0.1:8000/wallet/swap/quote" -H "Content-Type: application/json" -d '{"usdc_amount": "100"}' | python3 -m json.tool
```

Expected one of: `POOL_NOT_CONFIGURED` (pool address unset — normal pre-ceremony),
`POOL_UNAVAILABLE` (RPC error), or `OK` with a quote envelope.

### 5b — On-ramp pre-flight quote (composer-only, no spend)

```bash
curl -s -X POST "http://127.0.0.1:8000/wallet/onramp/quote" -H "Content-Type: application/json" -d '{"usd_amount": "100"}' | python3 -m json.tool
```

Returns the USD → USDC → FTNS artifact only. It does NOT initiate a swap or a
fiat on-ramp (composer-only `PENDING_COMMISSION`).

### 5c — WaaS wallet provisioning

```bash
curl -s -X POST "http://127.0.0.1:8000/wallet/waas/provision" -H "Content-Type: application/json" -d '{"user_id": "test-activation-001", "email": "ops@example.com"}' | python3 -m json.tool
```

Returns the wallet record dict. Returns 503 if the WaaS client is not
initialized. Provisioning is idempotent — re-calling returns the cached record.

### 5d — KYC vendor handshake

```bash
curl -s -X POST "http://127.0.0.1:8000/wallet/kyc/initiate" -H "Content-Type: application/json" -d '{"user_id": "test-activation-001", "email": "ops@example.com", "level": "basic"}' | python3 -m json.tool
```

Returns the KYC record dict (or 422 on validation error, 503 if the KYC client
is not initialized). Check current vendor wiring with
`curl -s http://127.0.0.1:8000/wallet/kyc/status`.

### 5e — On-ramp execute (builds the Coinbase Pay widget URL)

```bash
curl -s -X POST "http://127.0.0.1:8000/wallet/onramp/execute" -H "Content-Type: application/json" -d '{"user_id": "test-activation-001", "usd_amount": "1.00", "destination_address": "0x..."}' | python3 -m json.tool
```

Records the funnel intent (`INTENT_RECORDED`) and returns a Coinbase Pay
session URL. The user opens that URL, pays via Coinbase, and USDC lands in the
WaaS wallet on Base. The rest of the flow is event-driven (Step 6).

---

## The event-driven onboarding loop

Phase 5 is a clean event-driven pipeline. KYC verification triggers
auto-provisioning, which unlocks the on-ramp; the funnel state machine then
decouples from webhooks (it detects settlement by polling on-chain balances
during sweeps).

```
(1) KYC webhook arrives (Persona inquiry.completed)
      POST /wallet/kyc/webhook/persona   [HMAC-verified, replay-checked]
(2) kyc.update_status(user_id, VERIFIED)
(3) on VERIFIED transition → maybe_auto_provision_waas() (sp858)
      → waas.provision_wallet(user_id, email)   [idempotent]
(4) user calls POST /wallet/onramp/execute   → record_intent(INTENT_RECORDED)
(5) sp878 auto-sweep fires every interval  OR  operator POST /wallet/onramp/sweep
(6) sweep reads on-chain USDC; INTENT_RECORDED → PENDING_SETTLEMENT
(7) when USDC >= expected_usd * 0.95  → PENDING_SETTLEMENT → CONFIRMED (atomic)
      on_confirmed callback fires synchronously, fail-soft:
        (a) sp871: build Aerodrome USDC→FTNS swap envelope, attach to intent
        (b) persist intent to disk
        (c) sp874: POST completion webhook to PRSM_ONRAMP_COMPLETION_WEBHOOK_URL
(8) user GET /wallet/onramp/funnel/{intent_id} → retrieves swap envelope
```

### Funnel state machine

| State | Set when | Notes |
|---|---|---|
| `INTENT_RECORDED` | `/wallet/onramp/execute` records the intent | `intent_id`, `expected_usd`, `session_token`, `created_at` persisted to `PRSM_ONRAMP_FUNNEL_DIR/<intent_id>.json`. |
| `PENDING_SETTLEMENT` | first sweep iteration (even with no USDC yet) | Funnel open, awaiting on-chain settlement. |
| `CONFIRMED` | sweep sees on-chain USDC ≥ `expected_usd × 0.95` | Threshold (`_CONVERSION_THRESHOLD = 0.95`) tolerates Coinbase's ~1.5% fee + price slippage. Terminal. Fires `on_confirmed`. |
| `EXPIRED` | `now − created_at > 86400s` (24h) | User abandoned. Terminal. Skipped on future sweeps so it no longer burns RPC calls. |

Key invariant: the `on_confirmed` callback is **fail-soft**. If envelope build
(sp871), persistence, or the completion webhook (sp874) raises, the `CONFIRMED`
transition STILL STANDS — the side-effect failure is logged but never undoes
the settlement record. The swap envelope is fail-soft too: if the Aerodrome
pool is not yet seeded, `build_envelope_for_intent` returns None and the next
sweep retries.

### Completion webhook payload (sp874)

`POST` to `PRSM_ONRAMP_COMPLETION_WEBHOOK_URL` with body
`{event: "onramp.completion", intent_id, user_id, destination_address,
expected_usd, usdc_received, confirmed_at, swap_envelope}`. If
`PRSM_ONRAMP_COMPLETION_WEBHOOK_SECRET` is set, the POST carries
`X-PRSM-Signature: t=<unix>,v1=<hex>` (HMAC-SHA256). Every delivery attempt is
recorded under `PRSM_ONRAMP_COMPLETION_WEBHOOK_LOG_DIR`. Audit the delivery log
with `prsm node onramp-notifications`.

---

## Activation sequence

Step-by-step bring-up once the credentials are in hand:

1. **Wire Coinbase CDP keys** — `COINBASE_CDP_API_KEY_NAME`,
   `COINBASE_CDP_API_KEY_PRIVATE`, (optionally) `COINBASE_CDP_WALLET_SECRET`.
   Verify with `prsm node phase5-status --format json`: `waas.adapter_wired`
   should flip to `true`. If it stays `false` with `commissioned: true`, the
   PEM did not parse — check for a placeholder string.
2. **Wire Persona KYC** — `KYC_VENDOR=persona`, `KYC_VENDOR_API_KEY`,
   `PERSONA_TEMPLATE_ID`, `PERSONA_WEBHOOK_SECRET`. Configure Persona's dashboard
   to POST `inquiry.completed` to `POST /wallet/kyc/webhook/persona`. Verify
   `kyc.adapter_wired: true`.
3. **Wire Coinbase Pay** — `COINBASE_PAY_APP_ID`. This unblocks
   `/wallet/onramp/execute` producing real widget URLs.
4. **Set `BASE_RPC_URL`** to a paid RPC (Alchemy/Infura/QuickNode) — the public
   default is rate-limited and will throttle treasury + sweep reads.
5. **Enable auto-sweep** — `PRSM_FUNNEL_AUTO_SWEEP_INTERVAL_S=300` (or your
   cadence; minimum 60s). Restart so the worker spins up. Alternatively leave it
   disabled and sweep manually with `prsm node onramp-funnel --sweep`.
6. **(Externality) Run the Aerodrome pool ceremony** so
   `AERODROME_USDC_FTNS_POOL_ADDRESS` can be set — see below. Until then,
   `aerodrome.live_exec` is `false` and swap envelopes stay None.
7. **Verify** — `prsm node phase5-status` and `prsm node phase5-dashboard`.
   Expect `PARTIAL` until both the Aerodrome ceremony closes AND the Paymaster
   F23 blocker is resolved.

---

## Operator CLI reference

All `prsm node` commands below hit the running daemon at
`http://127.0.0.1:{api_port}` (default 8000) unless noted. Text format renders
Rich tables; `--format json` emits raw JSON for automation.

| Command | Sprint | Key flags | Backing endpoint(s) | Shows |
|---|---|---|---|---|
| `prsm node phase5-status` | sp861 | `--api-port`, `--format` | `GET /wallet/phase5/status` | The KYC/WaaS/Paymaster/Onramp/Aerodrome readiness grid rolled up to READY/PARTIAL/NOT_READY, with per-surface `commissioned`/`adapter_wired`/`live_exec`. |
| `prsm node phase5-dashboard` | sp856 | `--api-port`, `--format` | `GET /wallet/phase5/status`, `GET /wallet/treasury?max_wallets=100`, `GET /wallet/onramp/funnel?limit=20` | Unified one-screen view: readiness grid + fleet treasury + conversion funnel. Each surface fails independently so a partial outage still shows available data. |
| `prsm node wallet-balance <identifier>` | sp863 | `--api-port`, `--format` | `GET /wallet/balance/{user_id}` or `GET /wallet/balance/by-address/{address}` | Live Base-mainnet USDC + FTNS + ETH for one wallet. `identifier` is a WaaS user_id or `0x` address (auto-detected). Includes block number + address. |
| `prsm node treasury` | sp865 | `--api-port`, `--max-wallets` (100), `--format` | `GET /wallet/treasury` | Fleet-wide aggregated USDC/FTNS/ETH with per-wallet breakdown (`wallet_count_total`/`_with_address`/`_funded`, block number, per-wallet balances). |
| `prsm node onramp-funnel` | sp866 | `--api-port`, `--status`, `--sweep`, `--limit` (50), `--format` | `GET /wallet/onramp/funnel` (+ `POST /wallet/onramp/sweep` if `--sweep`) | Conversion funnel: status distribution counts + per-intent breakdown (user_id, address, expected_usd, usdc_received, status, age). `--sweep` triggers an on-chain sweep first. `--status` filters INTENT_RECORDED / PENDING_SETTLEMENT / CONFIRMED / EXPIRED. |
| `prsm node onramp-notifications` | sp877 | `--api-port`, `--limit` (50), `--success-only`, `--failures-only`, `--format` | `GET /wallet/onramp/notifications` | Outbound completion-webhook delivery history: timestamp, intent_id, url, status_code, signature_attached, error. Filterable by success/failure. |
| `prsm node aerodrome-ceremony` | sp876 | `--foundation-safe` (req), `--seed-usdc` (req), `--seed-ftns` (req), `--network` (mainnet), `--slippage-bps` (100), `--deadline-seconds` (3600), `--output-json/-j`, `--output-runbook/-r` | none (pure payload builders) | Generates the Safe-Transaction-Builder JSON (3-tx batch: USDC.approve + FTNS.approve + Router.addLiquidity) plus a co-signer runbook. Writes to files or stdout. Signs nothing. |
| `prsm node compliance-export` | sp873 | `--api-port`, `--since`, `--until`, `--user-id`, `--kind`, `--min-usd`, `--output/-o` | `GET /admin/fiat-compliance/export.csv` | CSV export of fiat compliance ring entries with composable filters for AUSTRAC TTR / FinCEN CTR / IRS 1099. Writes to file or stdout. |
| `prsm node fiat-readiness` | sp422 | `--format` | none (local `check_fiat_surface_health()`) | Color-coded findings table (ERROR/WARN/OK) with cause + remediation. Exit 0 for OK/WARN-only, non-zero on any ERROR. Pre-commission local check; no daemon required. |

A related read-only command, `prsm node partial-completion-history`, also
exists in the CLI for inspecting partial-completion records; it is adjacent to
the Phase 5 surface but outside the core activation path above.

---

## HTTP endpoint reference

Grouped by surface. All are tagged `["wallet"]` or `["admin"]`.

### On-ramp + funnel

| Method + path | Sprint | Purpose |
|---|---|---|
| `POST /wallet/onramp/quote` | sp852 | Pre-flight USD → USDC → FTNS quote artifact. Composer-only; does NOT initiate a swap or on-ramp. |
| `POST /wallet/onramp/execute` | sp853 | Builds the Coinbase Pay widget URL; records the funnel intent. |
| `GET /wallet/onramp/funnel` | sp857 | Funnel summary + intent list. Filters by status; `limit` caps the response. |
| `GET /wallet/onramp/funnel/{intent_id}` | sp857 | Single intent by ID (incl. swap envelope once CONFIRMED). 404 if not found. |
| `POST /wallet/onramp/sweep` | sp857 | Trigger an on-chain balance sweep across open intents; PENDING_SETTLEMENT → CONFIRMED when USDC arrives; on CONFIRMED builds the Aerodrome swap envelope (sp871). |
| `GET /wallet/onramp/notifications` | sp874 | Outbound completion-webhook delivery audit trail. |

### Swap (Aerodrome)

| Method + path | Sprint | Purpose |
|---|---|---|
| `POST /wallet/swap/quote` | sp855 | Quote a USDC↔FTNS swap. Returns POOL_NOT_CONFIGURED / POOL_UNAVAILABLE / OK. |
| `POST /wallet/swap/execute` | sp855 | Prepare an Aerodrome swap envelope for CDP-signed submission. Returns SESSION_READY once pool seeded + user has USDC. |
| `GET /wallet/pool/state` | sp279 | Aerodrome pool state. NOT_CONFIGURED if address unset, POOL_UNAVAILABLE on RPC error, state dict otherwise. |

### Off-ramp

| Method + path | Sprint | Purpose |
|---|---|---|
| `POST /wallet/offramp/quote` | sp848 | Pre-flight FTNS → USDC → USD quote artifact. V1 scope: summary only; does NOT initiate (gates on CDP commission). |

### WaaS wallets + balances + treasury

| Method + path | Sprint | Purpose |
|---|---|---|
| `POST /wallet/waas/provision` | sp858 | Provision a WaaS wallet (user_id + email). 503 if client not initialized. |
| `GET /wallet/waas/status` | sp858 | WaaS client status: commissioned, adapter_wired, network, wallet_count. 503 if not initialized. |
| `GET /wallet/balance/{user_id}` | sp862 | Live USDC + FTNS + ETH for a WaaS user (resolves user_id → address). |
| `GET /wallet/balance/by-address/{address}` | sp862 | Live balances for an explicit address (e.g. Foundation Safe). |
| `GET /wallet/treasury` | sp864 | Fleet-wide aggregated balances across PROVISIONED WaaS wallets; `max_wallets` caps RPC load. |

### KYC

| Method + path | Sprint | Purpose |
|---|---|---|
| `POST /wallet/kyc/initiate` | sp858 | Initiate KYC (user_id + email + level). 422 on validation error, 503 if not initialized. |
| `GET /wallet/kyc/status` | sp858 | KYC client status: commissioned, adapter_wired, vendor, supported_vendors, record_count. |
| `POST /wallet/kyc/webhook/{vendor}` | sp852/283 | Vendor webhook callback; reads raw body for HMAC verification. `PRSM_KYC_WEBHOOK_VERIFY_DISABLED=1` forces bypass (dev only). Triggers KYC→WaaS auto-provision on VERIFIED. |

### Paymaster + Phase 5 rollup

| Method + path | Sprint | Purpose |
|---|---|---|
| `GET /wallet/paymaster/status` | sp856 | Paymaster spend summary (`paymaster.spend_summary()`). 503 if not initialized. |
| `GET /wallet/phase5/status` | sp859 | One-shot Phase 5 readiness grid across all 5 surfaces. Powers the dashboards + CLI. |

### Compliance (admin)

| Method + path | Sprint | Purpose |
|---|---|---|
| `GET /admin/fiat-compliance` | sp848 | List compliance ring entries (pagination + kind/user_id filters). |
| `GET /admin/fiat-compliance/summary` | sp872 | `by_kind` aggregation + `total_entries`. 503 if ring not initialized. |
| `GET /admin/fiat-compliance/export.csv` | sp872 | Canonical CSV export for AUSTRAC/FinCEN/IRS. Filters: since/until (Unix), user_id, kind, min_usd. |

---

## Step 6 — Monitor + operate

- **Readiness** — `prsm node phase5-status` (grid) and `prsm node phase5-dashboard`
  (grid + treasury + funnel) on a schedule.
- **Conversion funnel** — `prsm node onramp-funnel` for intent-state
  distribution; `--sweep` to force a sweep when auto-sweep is disabled.
- **Webhook deliveries** — `prsm node onramp-notifications`
  (`--failures-only` to surface delivery problems).
- **Treasury** — `prsm node treasury` for fleet-wide USDC/FTNS/ETH.
- **Compliance exports** — `prsm node compliance-export --min-usd 10000`
  for the FinCEN CTR threshold, or `--since`/`--until` for a reporting window.

If `BASE_RPC_URL` is the public default, expect sweeps and treasury reads to
throttle under load — move to a paid RPC before going live.

---

## Rollback

If activation needs to be reverted:

```bash
sudo sed -i 's/^KYC_VENDOR=.*/KYC_VENDOR=/' /etc/prsm/bootstrap-server.env && sudo systemctl restart prsm-bootstrap
```

Setting `KYC_VENDOR=` (empty) puts the KYC surface back into the
un-commissioned state (`PENDING_COMMISSION`). To fully quiesce the on-ramp,
also unset `COINBASE_PAY_APP_ID` (execute returns PENDING_COMMISSION) and
`PRSM_FUNNEL_AUTO_SWEEP_INTERVAL_S` (disables the background sweep worker).

This is **non-destructive**: compliance-ring entries, KYC store, funnel intents,
and the webhook delivery log are all preserved (regulators still get the audit
trail), and any in-flight CONFIRMED intents keep their swap envelopes.

---

## Externality A — Aerodrome USDC↔FTNS pool-seeding ceremony (sp875/876)

This is the economically most significant single action in a PRSM deployment:
it sets the opening market price for FTNS, and the deposit is **irreversible**.
The PRSM module signs nothing — it only generates artifacts that the Foundation
Safe multi-sig co-signers upload, review, and execute at `wallet.safe.global`.
Vision target date: **2026-06-15**.

### 1. Sepolia rehearsal FIRST (≈$0, catches operator bugs)

Generate a throwaway-amount batch on Sepolia and run the full Safe flow end to
end before risking ~$50k of mainnet liquidity:

```bash
prsm node aerodrome-ceremony --network sepolia --foundation-safe 0x<safe-address> --seed-usdc 1 --seed-ftns 1 --output-json /tmp/ceremony-sepolia.json --output-runbook /tmp/ceremony-sepolia.md
```

### 2. Mainnet ceremony batch

```bash
prsm node aerodrome-ceremony --network mainnet --foundation-safe 0x<safe-address> --seed-usdc <usdc-amount> --seed-ftns <ftns-amount> --slippage-bps 100 --deadline-seconds 3600 --output-json /tmp/ceremony-mainnet.json --output-runbook /tmp/ceremony-mainnet.md
```

The 3-tx Safe batch (token_a/token_b sorted by address, `amountMin` computed
from `slippage_bps`, `deadline = now + 3600s`):

1. `USDC.approve(router, seed_usdc_units)` (USDC has 6 decimals)
2. `FTNS.approve(router, seed_ftns_units)` (FTNS has 18 decimals)
3. `Router.addLiquidity(...)`

Upload the JSON to the Safe Transaction Builder UI; hardware-wallet co-signers
sign and execute.

### 3. Post-ceremony env wiring

After execution, read the pool address on-chain (Basescan), then:

```bash
# add to the operator env, then restart
AERODROME_USDC_FTNS_POOL_ADDRESS=0x<pool-address-from-chain>
BASE_RPC_URL=https://mainnet.base.org   # or your paid RPC
```

Restart the operator daemons. The sp859 aggregator flips
`aerodrome.live_exec: true`, and sp871 starts building real USDC→FTNS swap
envelopes on funnel confirmation. Verify with `prsm node phase5-status`.

---

## Externality B — Paymaster F23 blocker + the Smart-Wallet pivot (sp867/868)

**Status: Paymaster `live_exec` is NOT achievable today.** This is the one
surface that holds the overall rollup at `PARTIAL` even when everything else
is live. Be honest about this in diligence.

### The blocker (sp867)

The CDP bundler tracer fails `failed to trace calls` on counterfactual
SimpleAccount v0.7 deploys via the Pimlico factory
`0x91E60e0613810449d098b0b5Ec8b51A0FE8c8985`. Root cause: the CDP tracer is
tuned for Coinbase Smart Wallet (WebAuthn / P-256 + magic-value validation),
not the generic Pimlico account structure.

### What sp867 verified is NOT the problem (closed gates)

- URL token placement (project ID in path, not Bearer header) — confirmed correct.
- Mainnet billing — enabled.
- AA23 dummy signature (`v=28, r=1, s=1`) — correct.
- Gas-policy cost projection ($5 per-user cap, callGasLimit=200K,
  verificationGasLimit=2M, preVerificationGas=100K) — working.
- Cap-poisoning — mitigated via `PRSM_SP867_SALT` rotation (nonce param).

So auth, billing, and gas policy all work; the failure is specifically the
bundler tracer's architectural mismatch.

### The pivot (sp868, in-tree)

Coinbase Smart Wallet scaffolding is committed: factories
`0xBA5ED110eFDBa3D005bfC882d75358ACBbB85842` (v1.1) and
`0x0BA5ED0c6AA8c49038F819E587E2633c4A9F428a` (v1) verified deployed on
2026-05-28; `createAccount(bytes[] owners, uint256 nonce)` encoding,
`execute`/`executeBatch` callData, and the counterfactual-address hook via
`EntryPoint.getSenderAddress` are all pinned with defending tests.

**What still blocks the `live_exec` flip:** the sp869+ WebAuthn signing layer
(P-256 pubkey encoding + R1 signature wrapping) is not yet implemented.

### Operator path forward

1. Land sp869+ WebAuthn signing so the CSW path can submit real user-ops; OR
2. Open a Coinbase support ticket for generic-account tracer support on the
   Pimlico factory.

Until one of those closes, paymaster gas sponsorship is unavailable and
`paymaster.live_exec` stays `false`. The rest of the on-ramp (KYC → WaaS →
Coinbase Pay → funnel → Aerodrome envelope) does not depend on the paymaster.

---

## Verification

The regression backstop for the whole Phase 5 flow is the sp881 end-to-end
integration test:

```bash
/opt/prsm/.venv/bin/python -m pytest tests/unit/test_sprint_881_phase5_e2e_integration.py -v
```

It exercises the full KYC → WaaS → Onramp → Swap → completion-webhook path. Run
it before any change that touches the Phase 5 surface. For live readiness on a
running deployment, the canonical check is `prsm node phase5-status` (per-surface
3-signal grid + overall rollup).

---

## Companion: signed-receipt audit-trail

Phase 5 doesn't produce InferenceReceipts. It produces:

- **Fiat compliance ring entries** (sp285+; CSV export via
  `GET /admin/fiat-compliance/export.csv`, sp872): every quote + execute, with
  timestamps + KYC tier + amounts + counterparties.
- **KYC handshake records** (sp280/849): vendor decision + outcome metadata,
  persisted under `PRSM_KYC_STORE_DIR`.
- **Inbound KYC webhook event log** (sp283): every inbound vendor webhook with
  its HMAC verification result.
- **Outbound completion-webhook delivery log** (sp874): one record per dispatch
  attempt under `PRSM_ONRAMP_COMPLETION_WEBHOOK_LOG_DIR` (timestamp, url,
  status_code, signature_attached, error), auditable via
  `prsm node onramp-notifications`.
- **Funnel intent records** (sp857): the conversion-funnel state machine under
  `PRSM_ONRAMP_FUNNEL_DIR`.

These are the audit-trail surface for AUSTRAC/FinCEN/IRS retention.

---

## Cross-references

- Design plan: `docs/2026-04-22-phase5-fiat-onramp-design-plan.md`
- KYC vendor decision: `docs/2026-04-27-phase5-kyc-vendor-decision.md`
- Composer-only invariant: sprint-280 R-2026-05-08-1 (every fiat write flows
  through a composer, never a raw client)
- Inbound webhook signature verification: sp283 (HMAC-SHA256; Persona/Onfido/Plaid)
- Replay protection: sp284 (timestamp window + signature-hash dedup ring)
- Tier-rolling-total enforcement: sp285 (FinCEN MSB defaults; rolling 24h)
- Audit ring: sp285 (`prsm/economy/web3/fiat_compliance_ring.py`)
- Startup health check: sp285 (`prsm/economy/web3/fiat_surface_health.py`)
- Readiness aggregator: sp848/859 (`prsm/economy/web3/phase5_status.py`)
- On-ramp funnel: sp857 (`prsm/economy/web3/onramp_funnel.py`)
- On-ramp→swap orchestrator: sp871 (`prsm/economy/web3/onramp_to_swap_orchestrator.py`)
- Completion notifier: sp874 (`prsm/economy/web3/onramp_completion_notifier.py`)
- Auto-sweep worker: sp878 (`prsm/node/funnel_auto_sweep.py`)
- Aerodrome pool ceremony: sp875/876 (`prsm/economy/web3/aerodrome_pool_ceremony.py`)
- Coinbase Smart Wallet pivot: sp868 (`prsm/economy/web3/coinbase_smart_wallet.py`)
- E2E regression test: sp881 (`tests/unit/test_sprint_881_phase5_e2e_integration.py`)

---

## Changelog

| Date | Sprint | Change |
|---|---|---|
| 2026-05-14 | 421 | Initial runbook. Closes the operator-side activation gap left after the 11-sprint Phase 5 commission-ready arc (sprints 276–286). Doc-only; no code paths shipped. |
| 2026-05-29 | 848–881 | Brought current with the Phase 5 productionization arc. Added the commissioned/adapter_wired/live_exec readiness model (sp848); full canonical env-var set incl. `COINBASE_CDP_WALLET_SECRET`, `COINBASE_PAY_APP_ID`, `PERSONA_TEMPLATE_ID`, completion-webhook + auto-sweep vars, and `:memory:` persistence sentinels; the 8-command operator CLI reference (phase5-status, phase5-dashboard, wallet-balance, treasury, onramp-funnel, onramp-notifications, aerodrome-ceremony, compliance-export, plus fiat-readiness); the grouped HTTP endpoint reference; the event-driven onboarding loop + funnel state machine (sp857/871/874/878); the activation sequence; the two remaining externalities (Aerodrome pool ceremony sp875/876, Paymaster F23 + CSW pivot sp867/868); and the sp881 E2E test as the regression backstop. Replaces the stale sp421-era endpoints (`/fiat/quote/onramp`, `/wallet/provision`, `/kyc/initiate`, `/fiat/onramp/execute`) with current `/wallet/*` routes. |