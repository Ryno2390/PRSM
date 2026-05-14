# Phase 5 fiat-surface activation runbook

Operator-facing step-by-step for activating the Phase 5 fiat surface (sprints
276–286, commission-ready as of 2026-05-12). Use this when Coinbase CDP
credentials arrive, KYC vendor agreement signs, and the Foundation has
ratified the Aerodrome pool seed.

## Pre-flight: what must be in place BEFORE running this runbook

| Gate | Owner | Evidence |
|---|---|---|
| Foundation funding decision ratified | Foundation Council | Latest PRSM-CR §X covering Phase 5 commission |
| Coinbase CDP account provisioned | Foundation Ops | API Key Name + Private Key (PEM) in hand |
| KYC vendor agreement signed | Foundation Ops | API key + webhook secret in hand |
| Aerodrome USDC↔FTNS pool seeded | Foundation Safe (2-of-3) | Pool contract address; verify on Basescan |
| Base mainnet RPC endpoint available | Operator | Working `BASE_RPC_URL` (Alchemy / Infura / public) |

If any of these aren't in place, **STOP** — activation will fail loudly via
sprint-285's startup health check (`check_fiat_surface_health()`), but better
to fail before bringing the service down.

---

## Step 1 — Canonical env-var set

Add these to `/etc/prsm/bootstrap-server.env` (or wherever your operator
config lives). All values are **secrets**: file must be `chmod 640` and owned
`root:prsm` (or equivalent service group).

```bash
# ── Coinbase WaaS (MPC wallet provisioning, sprint 276) ──
COINBASE_CDP_API_KEY_NAME=organizations/<ORG_ID>/apiKeys/<KEY_ID>
COINBASE_CDP_API_KEY_PRIVATE=-----BEGIN EC PRIVATE KEY-----
...
-----END EC PRIVATE KEY-----

# ── Coinbase CDP Paymaster (gasless FTNS transfers, sprint 278) ──
COINBASE_CDP_PAYMASTER_ENDPOINT=https://api.developer.coinbase.com/rpc/v1/base/<KEY>
COINBASE_CDP_PAYMASTER_API_KEY=<paymaster-api-key>
PRSM_PAYMASTER_POLICY_ID=<policy-uuid-from-coinbase-portal>

# ── Aerodrome (USDC↔FTNS pool quoter, sprint 280) ──
BASE_RPC_URL=https://mainnet.base.org   # or Alchemy / Infura
AERODROME_USDC_FTNS_POOL_ADDRESS=0x<pool-contract-address>

# ── KYC vendor (sprint 282 — pick ONE) ──
KYC_VENDOR=persona                       # or "onfido" or "plaid"
KYC_VENDOR_API_KEY=<vendor-api-key>

# ── KYC vendor webhook secret (sprint 283, vendor-specific) ──
PERSONA_WEBHOOK_SECRET=<webhook-secret>
# (use ONFIDO_WEBHOOK_TOKEN if vendor=onfido)
# (use PLAID_WEBHOOK_SECRET if vendor=plaid)

# ── KYC + compliance persistence (sprint 286 audit-retention) ──
PRSM_KYC_STORE_DIR=/var/lib/prsm/kyc
PRSM_FIAT_COMPLIANCE_LOG_DIR=/var/lib/prsm/fiat-compliance

# ── Jurisdiction (sprint 285 — tier-rolling-total enforcement) ──
PRSM_FIAT_JURISDICTION=US                # ISO-2 country code; routes to FinCEN
                                          # defaults ($1K basic / $10K enhanced)
```

**DO NOT** set these in production:

```bash
PRSM_FIAT_HEALTH_CHECK_BYPASS=1          # disables startup safety gate
PRSM_KYC_WEBHOOK_VERIFY_DISABLED=1       # disables webhook signature check
```

Both exist for **test/dev environments only**. Sprint 285's health check fires
an ERROR-severity finding if either is set in a commissioned environment.

---

## Step 2 — Create persistence directories

```bash
sudo mkdir -p /var/lib/prsm/kyc /var/lib/prsm/fiat-compliance
sudo chown prsm:prsm /var/lib/prsm/kyc /var/lib/prsm/fiat-compliance
sudo chmod 750 /var/lib/prsm/kyc /var/lib/prsm/fiat-compliance
```

These hold sensitive PII (KYC outcomes) and audit-trail events (compliance
ring entries). Regulators (AUSTRAC, FinCEN, IRS) expect 5–7yr retention —
back these directories up to encrypted off-host storage on the same cadence
as `/etc/letsencrypt/`.

---

## Step 3 — Restart the PRSM service

```bash
sudo systemctl restart prsm-bootstrap   # or prsm-node — whichever runs the fiat surface
```

The sprint-285 startup health check (`check_fiat_surface_health()`) fires
automatically during startup. Findings of severity ERROR cause boot to fail
LOUDLY rather than silently running with broken security.

Common ERROR findings + remediations:

| `cause` | Means | Fix |
|---|---|---|
| `kyc_commissioned_<vendor>_webhook_secret_missing` | KYC vendor configured but webhook secret unset | Set `<VENDOR>_WEBHOOK_SECRET` |
| `kyc_commissioned_webhook_verify_disabled` | KYC live but `PRSM_KYC_WEBHOOK_VERIFY_DISABLED=1` | Unset the var |
| `fiat_compliance_log_dir_unset` (WARN) | Compliance ring will be in-memory only | Set `PRSM_FIAT_COMPLIANCE_LOG_DIR` |

If the service refused to boot, fix the listed findings and retry.

---

## Step 4 — Verify the health check programmatically

From any host with the PRSM venv:

```bash
/opt/prsm/.venv/bin/python -c "
import os
from prsm.economy.web3.fiat_surface_health import check_fiat_surface_health
findings = check_fiat_surface_health(os.environ)
if not findings:
    print('OK — fiat surface is commission-ready with no findings.')
else:
    for f in findings:
        print(f'[{f.severity.value.upper()}] {f.cause}: {f.remediation}')
"
```

Empty output = fully commission-ready. Otherwise, address each finding per
the table above + repeat.

---

## Step 5 — Smoke-test the canonical surfaces

### 5a — Aerodrome quote (read-only, no spend)

```bash
curl -s "http://localhost:8000/fiat/quote/onramp?usd_amount=100" | python3 -m json.tool
```

Expected: a JSON quote payload with `ftns_amount`, `usd_amount`, `quote_ts`,
`kyc_required: <bool>`. If `kyc_required=true` and the caller hasn't run the
KYC handshake, the composer surfaces `PENDING_KYC` instead.

### 5b — Coinbase WaaS wallet provisioning (creates real MPC wallet)

```bash
# Triggers Coinbase WaaS — costs nothing but commits to the wallet existing
curl -s -X POST "http://localhost:8000/wallet/provision" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test-activation-001"}' | python3 -m json.tool
```

Expected: `{"address": "0x...", "status": "provisioned"}`. The address is
permanent — `test-activation-001` can be deleted post-test via Coinbase
portal if you want a clean state.

### 5c — KYC vendor handshake

```bash
# Initiates KYC flow (returns vendor-hosted URL for end-user)
curl -s -X POST "http://localhost:8000/kyc/initiate" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test-activation-001"}' | python3 -m json.tool
```

Expected: `{"vendor_session_url": "https://...", "status": "PENDING"}`.

### 5d — Onramp full path (composes WaaS + paymaster + KYC + quote)

```bash
# Real money flow — use the smallest legal amount ($1) and your own user
curl -s -X POST "http://localhost:8000/fiat/onramp/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test-activation-001",
    "usd_amount": "1.00"
  }' | python3 -m json.tool
```

Expected: either `{"status": "EXECUTED", ...}` (KYC was already complete) or
`{"status": "PENDING_KYC", ...}` (KYC handshake required first).

---

## Step 6 — Monitor for the first 24 hours

Fiat surface emits these signals:

- **Compliance ring entries** at `$PRSM_FIAT_COMPLIANCE_LOG_DIR/` — every
  quote + execute lands here.
- **Webhook events** in service logs — search for `kyc_webhook_received`.
- **Prometheus**: `prsm_kyc_tier_count{tier="basic"}`,
  `prsm_kyc_tier_count{tier="enhanced"}`,
  `prsm_fiat_quote_count{action="onramp"}`,
  `prsm_fiat_execute_count{action="offramp"}`.

If any 24hr cumulative metric exceeds your KYC tier's rolling-total limit
(default $1K basic / $10K enhanced FinCEN MSB), sprint-285's tier enforcement
auto-blocks further executes for that user.

---

## Rollback

If activation needs to be reverted:

```bash
# Disable the fiat surface without touching anything else
sudo sed -i 's/^KYC_VENDOR=.*/KYC_VENDOR=/' /etc/prsm/bootstrap-server.env
sudo systemctl restart prsm-bootstrap
```

Setting `KYC_VENDOR=` (empty) puts the surface back into the pre-commission
"un-commissioned" state. The startup health check no longer requires the
webhook secrets etc. — operators can debug + recommission cleanly.

This is **non-destructive**: existing compliance-ring entries and KYC store
state are preserved (regulators still get audit trail), and any in-flight
PENDING quotes remain valid until they expire.

---

## Companion: signed-receipt audit-trail

Phase 5 doesn't produce InferenceReceipts. It produces:
- **Fiat compliance ring entries** (sprint 285): every quote + execute, JSON-encoded with timestamps + KYC tier + amounts + counterparties.
- **KYC handshake records** (sprint 282): vendor decision + outcome metadata.
- **Webhook event log** (sprint 283): every inbound webhook with HMAC verification result.

These are the audit-trail surface for AUSTRAC/FinCEN/IRS retention.

---

## Cross-references

- Design plan: `docs/2026-04-22-phase5-fiat-onramp-design-plan.md`
- KYC vendor decision: `docs/2026-04-27-phase5-kyc-vendor-decision.md`
- Composer-only invariant: sprint-280 R-2026-05-08-1 (every fiat write
  flows through a composer, never a raw client)
- Webhook signature verification: sprint 283 (HMAC-SHA256, Persona + Onfido
  + Plaid)
- Replay protection: sprint 284 (timestamp window + signature-hash dedup ring)
- Tier-rolling-total enforcement: sprint 285 ($1K basic / $10K enhanced
  FinCEN MSB defaults, rolling 24h combines onramp+offramp)
- Audit ring: sprint 285 (`prsm/economy/web3/fiat_compliance_ring.py`)
- Startup health check: sprint 285 (`prsm/economy/web3/fiat_surface_health.py`)

---

## Changelog

| Date | Sprint | Change |
|---|---|---|
| 2026-05-14 | 421 | Initial runbook. Closes the operator-side activation gap left after the 11-sprint Phase 5 commission-ready arc (sprints 276–286). Doc-only; no code paths shipped. |
