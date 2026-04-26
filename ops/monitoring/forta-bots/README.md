# PRSM Forta Detection Bot

**On-chain monitoring for PRSM smart contracts.** Implements detection rules per Exploit Response Playbook §11 readiness checklist.

## What this bot does

Watches PRSM contracts on Base mainnet, Base Sepolia, Ethereum mainnet, and Ethereum Sepolia for anomalies that match known smart-contract exploit patterns or deviate from normal protocol operation. Findings emit to:

1. **Forta network** (public, persistent on-chain record)
2. **Discord war-room webhook** (private, fast)
3. **PagerDuty** (P0/P1 only, post-mainnet)
4. **Email** (P0 backup, post-mainnet)

Severity mapping — Forta to Exploit Response Playbook:

| Forta severity | Playbook | Channel |
|---|---|---|
| Critical | P0 | Discord + PagerDuty + Email |
| High | P1 | Discord + PagerDuty |
| Medium | P2 | Discord |
| Low | P3 | Discord |
| Info | n/a | Forta network only |

## Detectors

| Detector | Contract | Risk Register | Key alerts |
|---|---|---|---|
| `ftns_token` | `FTNSTokenSimple.sol` | A4 | role grants, supply-cap proximity, transfer-volume spikes, pause |
| `provenance_registry` | `ProvenanceRegistry.sol` | A3, F4 | ownership transfers, registration spam, high royalty rate |
| `royalty_distributor` | `RoyaltyDistributor.sol` | A1 | distribution failures, split-ratio anomalies, amount spikes |
| `batch_settlement` | `BatchSettlementRegistry.sol` | A2 | challenge rate, win-rate elevations |
| `escrow_pool` | `EscrowPool.sol` | A2 | amount spikes, refund-rate elevations, stuck escrows |

Detectors deliberately err toward **emit-and-let-router-decide** rather than tight false-positive control — better to alert + downgrade than miss a real incident. Severity is tunable per-detector once mainnet operational data exists.

## Setup

### Local development

```bash
cd ops/monitoring/forta-bots
npm install
npm run build

# Configure environment
export PRSM_DISCORD_WEBHOOK_URL='https://discord.com/api/webhooks/...'
export PRSM_PAGERDUTY_KEY='xxxxxxxx'
export PRSM_ALERT_EMAILS='security@prsm.foundation'

# Or test in dry-run mode (no actual webhooks)
export PRSM_ALERT_DRY_RUN=1

# Run against a specific tx for testing
npm run tx 0xabcd...

# Run against a block range
npm run range 23000000..23000010
```

### Forta network deployment

```bash
# 1. Generate keyfile (first time only)
npm run keyfile

# 2. Build + push image
npm run build
npm run push

# 3. Publish bot to Forta registry
npm run publish

# 4. View live logs
npm run logs
```

Bot will deploy to Forta scanner nodes globally. Findings automatically propagate to the Forta network's public alert feed.

## Adding a new detector

1. Create `src/detectors/<contract>.ts` following the pattern of existing detectors:
   ```typescript
   export async function detect<Contract>Anomalies(
     txEvent: TransactionEvent,
     chainId: NetworkChainId
   ): Promise<Finding[]> {
     // ...
   }
   ```
2. Add contract address + ABI events to `src/config/contracts.ts`
3. Wire detector into `src/agent.ts` `handleTransaction` Promise.allSettled
4. Add unit test in `tests/<contract>.test.ts`
5. Update Risk Register entry to reference the new detector
6. Run `npm test` then `npm run start:dev` to validate locally

## Tuning thresholds

All thresholds live in `src/config/contracts.ts` `THRESHOLDS`. Tunable parameters:
- Spike multipliers (% above rolling p99)
- Time windows (window size for rolling stats)
- Per-address rate limits (Sybil detection)
- Stuck-escrow age cutoffs

Per Q4 ratification, threshold modifications follow the council 30-day-timelock pattern post-board. Pre-board, founder applies adjustments with public commit message + rationale.

## Pre-mainnet checklist

Per Exploit Response Playbook §11:
- [ ] Forta keyfile generated + secured
- [ ] Discord webhook configured (`PRSM_DISCORD_WEBHOOK_URL`)
- [ ] PagerDuty integration key configured (post-mainnet)
- [ ] Bot pushed to Forta network in dry-run mode
- [ ] All 5 detectors validated against synthetic-attack test fixtures
- [ ] Alert latency measured: target <30s from on-chain event to Discord delivery
- [ ] Quarterly tabletop exercise §1 (active drain) passes detection within first 30 seconds
- [ ] Mainnet contract addresses populated in `contracts.ts` post-Phase-1.3 deploy

## What this bot does NOT do

- **Pause contracts.** That's a separate manual action requiring 3-of-5 council multi-sig.
- **Replace human triage.** Findings go to war-room channels for council review; bot does not auto-execute responses.
- **Cover off-chain incidents.** See `docs/security/SECURITY_RUNBOOK.md` for traditional infosec.
- **Provide false-positive immunity.** Tuning is iterative; expect false positives in first 30 days of mainnet, refined down via threshold adjustments.

## Related documents

- [Exploit Response Playbook](../../../docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md) — what to do when this bot fires
- [Vendor decision memo](../2026-04-26-vendor-decision-forta-vs-tenderly.md) — why Forta over Tenderly
- Risk Register A1-A8 (private repo) — risks this monitoring detects

## License

MIT, same as parent PRSM repo.
