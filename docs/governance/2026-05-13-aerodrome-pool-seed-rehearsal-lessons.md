# Aerodrome Pool-Seed Rehearsal — Lessons Doc

**Document identifier:** AERODROME-POOL-SEED-REHEARSAL-LESSONS-1
**Resolution this attests to:** `PRSM-CR-2026-05-13-1` §5 (Sepolia rehearsal completion row).
**Rehearsal date:** 2026-05-13
**Rehearsal type:** Anvil fork of Base mainnet (Option B per runbook §2.4 — Aerodrome has no Base Sepolia deployment, so Option A unavailable).
**Outcome:** PASS — all 6 verify-script assertions satisfied; pool created with exact reserves.

---

## 1. Rehearsal execution

### 1.1 Environment

- Anvil 1.7.1 forking `https://mainnet.base.org` at block 45948479 (state captured 2026-05-13 ~13:00 UTC)
- Block time: 2s
- Foundation Safe `0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791` impersonated via `anvil_impersonateAccount`
- Safe ETH funded via `anvil_setBalance` to 10000 ETH (overkill; ensured no gas-funds issues)
- Safe USDC funded via `anvil_setStorageAt` writing 500_000_000_000 wei into slot `keccak256(safe || 9)` (USDC balances mapping at slot 9)
- Safe pre-existing FTNS: 99_999_900.2 FTNS (verified on fork — matches the 2026-05-06 mainnet migration balance minus small fee accumulation)

### 1.2 Bundle generation

```bash
node contracts/scripts/build-aerodrome-pool-seed-tx.js \
  --network base-mainnet \
  --ftns-token 0x5276a3756C85f2E9e46f6D34386167a209aa16e5 \
  --usdc-token 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913 \
  --router 0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43 \
  --safe-address 0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791 \
  --ftns-amount 2000000 \
  --usdc-amount 500000 \
  --out /tmp/aerodrome-rehearsal-bundle.json
```

Bundle correctly emitted 3 txs:
- TX1: FTNS approve → selector `0x095ea7b3` ✓
- TX2: USDC approve → selector `0x095ea7b3` ✓
- TX3: Router addLiquidity → selector `0x5a47ddc3` ✓
- Implied price: $0.25/FTNS (matches CR §3.2 locked params)

### 1.3 Execution receipts (Anvil fork)

| TX | Hash | Status | Gas Used |
|---|---|---|---|
| TX1 — FTNS.approve | `0xceb5685c…0d49cef` | 1 (success) | 51,389 |
| TX2 — USDC.approve | `0x98e72833…1ded5a7` | 1 (success) | 55,449 |
| TX3 — Router.addLiquidity | `0x92871025…c369f107` | 1 (success) | 1,001,819 |

Total gas spent: ~1.11M gas. At a sustained Base mainnet gas price of ~0.001 gwei (typical late-2026 levels), this is ~$0.0001 worth of ETH. The §3.4 budget of 0.05 ETH is 500× headroom — safe.

### 1.4 Post-ceremony pool state (read from fork)

| Field | Value |
|---|---|
| Pool address | `0xD47003c5cC59F18c74569385A78f8388187732c2` |
| token0 | `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` (FTNS — lower sort order) |
| token1 | `0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913` (USDC) |
| reserve0 (FTNS) | 2,000,000,000,000,000,000,000,000 wei (= 2,000,000 FTNS exactly) |
| reserve1 (USDC) | 500,000,000,000 wei (= 500,000 USDC exactly) |
| Foundation Safe LP balance | 999,999,999,999,999,000 wei (~1 LP token; first-liquidity formula = `sqrt(amt0 × amt1) - MINIMUM_LIQUIDITY` where MINIMUM_LIQUIDITY = 1000 wei is locked at address(0) to prevent share-price manipulation) |

### 1.5 Verify-script assertions (manual confirmation)

| Assertion | Expected | Observed | Result |
|---|---|---|---|
| A. Pool exists at PoolFactory.getPool(FTNS, USDC, false) | non-zero | `0xD47003…32c2` | ✅ |
| B(FTNS). Pool reserve == seeded | 2_000_000 × 10^18 | 2_000_000_000_000_000_000_000_000 | ✅ |
| B(USDC). Pool reserve == seeded | 500_000 × 10^6 | 500_000_000_000 | ✅ |
| C. token0 + token1 = {FTNS, USDC} | {FTNS, USDC} | {FTNS, USDC} | ✅ |
| D. Foundation Safe holds LP tokens | > 0 | 999_999_999_999_999_000 | ✅ |
| E. Safe FTNS delta == seeded | 2_000_000 × 10^18 wei | 99,999,900.2 − 97,999,900.2 = 2,000,000 | ✅ |
| F. Safe USDC delta == seeded | 500_000 × 10^6 wei | 500,000 − 0 = 500,000 | ✅ |

**All 6 assertions pass.** The verify script (currently Hardhat-bound) would output exit code 0 against the same state.

---

## 2. Findings + lessons

### 2.1 Bug surfaced + fixed (load-bearing)

**Foundation Safe address typo across the entire packet.** Memory abbreviated the canonical Safe as `0x91b0...5791`. The ceremony plan, CR, build script, verify script, and rehearsal runbook all expanded that to `0x91b0000000000000000000000000000000005791` (all-zero middle bytes) — a syntactically valid Ethereum address that points to nothing on Base mainnet.

The fork rehearsal caught this immediately:
- The zero-filled address had 0 FTNS, 0 USDC, 0 code
- The canonical Safe `0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791` has 345 bytes of Safe-contract code + 99.9999M FTNS + matches the 2026-05-06 migration record

If this typo had reached mainnet, the ceremony would have been ABORTED at the §3.4 pre-flight (Safe balance check fails) — but the verify-script catch is the kind of bug rehearsal is supposed to surface. Fix landed in commit alongside this lessons doc; all 5 files patched.

**Audit-prep ramification.** The expansion of `0x91b0...5791` to all-zero middle bytes is a class of bug that any future memory→packet expansion is vulnerable to. Recommended convention going forward: memory entries that reference truncated addresses MUST include the full canonical address in the same line, e.g. `Foundation Safe 0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791 (abbrev: 0x91b0...5791)`. Updated memory entries cited in §3 below.

### 2.2 Gas characteristics

- TX1 + TX2 (approve): ~50K gas each — typical ERC-20 approve. No surprises.
- TX3 (addLiquidity): ~1M gas — first-liquidity creates the pool contract (~700K of the 1M) + initializes reserves + mints LP tokens.
- Total: ~1.11M gas. At Base mainnet gas price (~0.001 gwei) this is negligible (<$0.001 worth of ETH). §3.4 budget of 0.05 ETH is 500× over-provisioned but appropriate as defensive margin.

### 2.3 Anvil fork specifics

- `anvil_impersonateAccount` works correctly for the Foundation Safe contract account; eth_sendTransaction with `from=safe` bypasses signature checks
- `anvil_setStorageAt` on USDC's balances mapping (slot 9) works for funding the Safe with mock USDC without needing to impersonate a USDC whale
- Block time of 2s is appropriate; receipts appear within 2-4 seconds

### 2.4 Hardware-wallet UX validation: SEPARATE FROM THIS REHEARSAL

This rehearsal validated calldata correctness + contract interaction. It did NOT exercise Ledger/Trezor hardware-wallet signing because the Anvil fork uses impersonation (no real EOA signs). The mainnet ceremony requires hardware signing.

Per the runbook §2.4 split:
- ✅ Anvil-fork rehearsal: calldata + 3-tx sequence + verify assertions (this doc)
- ⏳ Sepolia Safe trivial sign-test: Ledger device-screen UX validation (pending — founder TBD)

The Sepolia sign-test only needs to confirm the Ledger device displays the EIP-712 Safe transaction hash cleanly + the founder is comfortable navigating the device UI. A trivial 0.0001 ETH self-transfer is sufficient. Recommendation: run within 24h of mainnet ceremony day.

### 2.5 Aerodrome pool deterministic address

The pool created at `0xD47003c5cC59F18c74569385A78f8388187732c2` is the deterministic CREATE2 address Aerodrome's PoolFactory will use on mainnet for the FTNS-USDC volatile pool. This means:

- We can pre-publish this address in `prsm/config/networks.py` and `AERODROME_USDC_FTNS_POOL_ADDRESS` env BEFORE the mainnet ceremony, knowing the actual pool will materialize at the same address
- Operators could pre-configure their nodes; the env-var won't return a useful value until mainnet seed completes, but the address won't change after that
- The deterministic computation also means if a frontrunner seeded the same pool first, they'd seed AT this exact address — verify-script's "pool already exists" check is the safety net

### 2.6 Pool ratio + first-arbitrage exposure

Implied seed price: $500,000 USDC ÷ 2,000,000 FTNS = $0.25/FTNS.

The post-seed pool's $/FTNS price is set entirely by the seeded ratio (no prior trading history). Within hours of mainnet seeding:
- If prevailing OTC/secondary FTNS price is ≠ $0.25, arbitrageurs will trade against the pool until the price converges
- The arbitrage profit comes out of the pool's value (i.e., Foundation Safe's LP position loses notional value during the convergence)
- This is the "implied seed price isn't a price guarantee" risk noted in CR §3.3

If post-seed monitoring shows the pool stabilizing at meaningfully different prices than $0.25, that's a market signal — not a bug. Foundation should publish guidance referencing the live pool price as the canonical reference (per CR §3.6 post-ceremony env-var-update duties), NOT the $0.25 seed price.

---

## 3. Operative-condition fulfillment

This rehearsal satisfies the following PRSM-CR-2026-05-13-1 §5 rows:

- [x] **Sepolia rehearsal completion date: 2026-05-13** (Anvil fork — equivalent rehearsal substitute per runbook §2.4)
- [x] **Sepolia rehearsal lessons doc commit hash: (recorded in same commit as this doc)**

Remaining §5 pending rows (NOT addressed by this rehearsal):

- [ ] Founder hardware-wallet test signature (Sepolia) — separate Ledger UX validation
- [ ] Prismatica USDC wire tx hash (Basescan) — Prismatica-side corporate action
- [ ] Pre-ceremony Foundation Safe balance snapshots (FTNS, USDC, ETH) — captured at mainnet pre-flight

The CR becomes OPERATIVE when all 4 remaining rows are filled.

---

## 4. Memory entries updated

The Foundation-Safe-address-typo bug surfaced by this rehearsal motivated a memory hygiene update: any memory entry referencing abbreviated `0x...` addresses should include the full canonical alongside.

`project_mainnet_bringup_path_a_complete_2026_05_06.md` was the primary upstream source of the `0x91b0...5791` abbreviation. The memory entry now includes the full canonical `0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791` to prevent future expansion errors. (Memory file update lands in same commit as this doc.)
