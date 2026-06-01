# Canonical-Workflow Base Mainnet Trace — 2026-05-06

**Status:** ✅ EXECUTED. Path A from the bring-up runbook completed end-to-end on Base mainnet. Foundation Safe earned its first 0.2 FTNS in real network fees.

## Executive summary

This is the first end-to-end execution of the canonical PRSM user workflow (Vision §4 step 6) on Base mainnet using real on-chain contracts and real (non-zero monetary value) FTNS supply. Every step recommended in `prsm/config/networks.py:MAINNET` is exercised: content registration, ownership transfer to a deterministic ephemeral creator, FTNS approval, royalty distribution with the canonical 3-way split, and post-state verification of the settlement landing zone.

The trace also closed two pre-existing issues:

1. **FTNS treasury custody** — the full 100M supply migrated from a hot deployer EOA to the 2-of-3 hardware-wallet Foundation Safe. Closes the security gap where the entire supply lived on a single hot key in `~/.prsm/.env`. Tx `0xa128e826...d942`.
2. **Hot-key file mode** — `~/.prsm/.env` was world-readable (mode 644). Locked to mode 600 during the trace.

## On-chain artifacts

All transactions are confirmed on Base mainnet (chain id 8453).

| Step | Tx hash | Purpose |
|------|---------|---------|
| 1 | `0xa128e8268cccbe1663f75b7f4825c0934368bfc4baac782ec7e755028611d942` | Hot key → Foundation Safe: 100,000,000 FTNS custody migration |
| 2 | (Safe audit log entry, multisig) | Foundation Safe → demo payer: 100 FTNS disbursement (2-of-3 ceremony, Ledger + Trezor) |
| 3 | `0xcb6140cefd524dc8a63a1e1c7a41e236200f26c28a24552a44ed569d7ff630b1` | Demo payer registers content on ProvenanceRegistry v1 |
| 4 | `0x7f70c0460f8cbe44e7edffd57583b4b6c3a8351d9a3e2f840f83724513e9ffcc` | Demo payer transfers content ownership to ephemeral creator EOA |
| 5 | `0x73bfc9e42ab5a2ce15c6c961bef06046aa21309777a4490d39391db436fc6612` | Demo payer approves 100 FTNS to RoyaltyDistributor |
| 6 | `0xbe85b53fbae26083c39505040ebd2ae18b6f5b6796d8ec636222ff74d4fe5b20` | Demo payer calls RoyaltyDistributor.distributeRoyalty(content_hash, 10 FTNS) |

## RoyaltyPaid event — the canonical 3-way split

Decoded from log on tx `0xbe85b5...4fe5b20`:

```
payer:        0xBbEB1cb42F1D5ad05B46eE023D6e4871D813C9a0   (demo payer)
creator:      0x491DA67CBa351bD64eDA04094376be4784700440   (ephemeral)
treasury:     0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791   (Foundation Safe)
servingNode:  0x944bF83B18Be44b40dB7081e0f25ce216408Fd3c   (ephemeral)
amounts:
  creator      = 1.0000 FTNS  (10% royalty)
  network      = 0.2000 FTNS  ( 2% network fee)
  servingNode  = 8.8000 FTNS  (88% remainder)
total          = 10.0000 FTNS  (gross)
```

The 10% / 2% / 88% split is the exact canonical Vision §4 step 6 economic mechanic. Same rates landed on Sepolia 2026-05-06 — mainnet is internally consistent with the testnet trace.

## Final FTNS balance reconciliation

Read post-trace from canonical RPC after replica catch-up:

| Address | FTNS | Δ from start | Notes |
|---------|------|--------------|-------|
| Hot deployer `0x8eaA...f012` | 0.000000 | −100,000,000 | Migrated to Safe; admin role still held (deferred) |
| Foundation Safe `0x91b0...5791` | 99,999,900.200000 | +99,999,900.2 | Custody migration + 0.2 FTNS network fee |
| Demo payer `0xBbEB...C9a0` | 90.000000 | +90 | Net of 10 FTNS distributed |
| Creator EOA `0x491D...0440` | 1.000000 | +1.0 | 10% creator royalty |
| Serving node EOA `0x944b...Fd3c` | 8.800000 | +8.8 | 88% remainder |
| RoyaltyDistributor `0x3E82...D6c2` | 0.000000 | 0 | Push-payment routes directly; no escrow |
| **TOTAL** | **100,000,000** | — | Conservation: matches pre-trace 100M supply ✓ |

## Substantive finding — push vs. pull settlement

The canonical-workflow trace script was originally written against the **pull-payment** RoyaltyDistributor variant (Sepolia, deployed 2026-05-05 with the T6.2 D-04 OZ pull-payment refactor): tokens accumulate in `claimable[address]` and recipients claim() later.

**Mainnet RoyaltyDistributor at `0x3E8201B2cdC09bB1095Fc63c6DF1673fA9A4D6c2` (deployed 2026-05-04) is push-payment:** funds are routed directly to recipient EOAs in the same tx. RoyaltyDistributor never holds a balance; there is no `claimable()` function.

Both variants enforce the same 3-way split with the same bps rates. The settlement mechanic differs:

| Variant | Tx steps | Recipient action | Escrow window |
|---------|----------|------------------|---------------|
| Push (mainnet) | 1 distribute tx → 4 internal Transfers | none | none |
| Pull (Sepolia) | 1 distribute tx → 1 Transfer to RD | claim() per recipient | indefinite |

The trace script (`scripts/exercise_canonical_workflow_base_sepolia.py`) was patched in the same change as this doc to detect either variant: if `claimable()` reverts, it falls back to direct `balanceOf()` reads on the recipient EOAs.

The choice between push and pull matters for several second-order things — gas costs, recipient gas-coverage assumptions, accounting clarity. For a future PRSM Council resolution if mainnet upgrades to pull (or vice versa), the relevant trade-offs:

- **Push** is simpler, lower latency, no claim UX, but requires recipients to have a sane address pre-registered (no rescue path if address is wrong).
- **Pull** allows misconfigured-recipient recovery (admin can re-route via a roles-gated reset), and shifts gas cost to recipient.

This is a finding to flag but not a blocker — both meet Vision §4.

## Operational follow-ups (deferred)

Things this trace did *not* do, that should be ratified separately:

1. **DEFAULT_ADMIN_ROLE migration** to Foundation Safe. The hot deployer key still holds admin role on the FTNS contract. The migration is a one-way (grant + revoke) ceremony that warrants a PRSM-CR-* council resolution before execution.
2. **Hot-key zeroization** per L6c key-rotation runbook. Now that the hot key has zero balance and only deferred admin powers, the key file (`~/.prsm/.env` `FTNS_WALLET_PRIVATE_KEY`) should be cycled out of the working laptop into a secure offline backup with documented recovery procedure.
3. **Mainnet RoyaltyDistributor upgrade decision** (push vs. pull). Currently push (legacy 2026-05-04). If pull is preferred, a redeploy + migration ceremony is needed.
4. **Ephemeral demo state cleanup** — the ephemeral creator EOA and serving-node EOA hold 1 / 8.8 FTNS respectively. These are deterministically derived from sha256 seed labels, so the keys are recoverable if needed for repeat traces or as audit evidence; otherwise the dust is unreachable to the ephemeral identities and effectively burned. Decision for the council: leave as audit evidence (recommended) vs. sweep back to Safe.
5. **PRSM-PROV-1 Item 7 mainnet deploy** — V2 ProvenanceRegistry on mainnet (now authorized by PRSM-CR-2026-05-06-2; deploy script's mainnet-block guard removed in follow-on PR).

## Verification commands for auditor reproduction

```bash
# Verify all 6 transactions confirm with status=1 on Basescan:
for h in 0xa128e826... 0xcb6140ce... 0x7f70c046... 0x73bfc9e4... 0xbe85b53f...; do
  open "https://basescan.org/tx/$h"
done

# Verify final FTNS distribution matches table above:
PYTHONPATH=. .venv/bin/python3.14 -c "
from web3 import Web3
w3 = Web3(Web3.HTTPProvider('https://mainnet.base.org'))
ftns = '0x5276a3756C85f2E9e46f6D34386167a209aa16e5'
for label, addr in [
    ('safe',    '0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791'),
    ('payer',   '0xBbEB1cb42F1D5ad05B46eE023D6e4871D813C9a0'),
    ('creator', '0x491DA67CBa351bD64eDA04094376be4784700440'),
    ('node',    '0x944bF83B18Be44b40dB7081e0f25ce216408Fd3c'),
]:
    sel = Web3.keccak(text='balanceOf(address)')[:4]
    data = sel + b'\\x00'*12 + bytes.fromhex(addr[2:])
    raw = w3.eth.call({'to': Web3.to_checksum_address(ftns), 'data': data})
    print(f'{label:8s} {int.from_bytes(raw, \"big\") / 10**18:>18,.6f} FTNS')
"
```

## Sign-off

Mainnet bring-up Path A (Foundation-Safe-funded demo payer) — **completed**. Task #364 closes here. Foundation Safe is in active service and has earned its first network fee.
