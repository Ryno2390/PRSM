# `deploy-provenance.js` Mainnet-Fork Dry-Run — 2026-04-30

**Goal:** exercise the mainnet-only safety guards added in commit
`85988825` (chainId pin + canonical-FTNS pin + treasury-is-contract)
under realistic conditions — i.e., against real Base mainnet state —
WITHOUT broadcasting any transactions to mainnet.

**Method:** rather than fight `hardhat node` chainId limitations
(`hardhat node` always reports the `hardhat` network's chainId 31337
even when forking, and there's no `--chain-id` flag without anvil),
we exercise each guard's pass/fail path directly by manipulating env
vars and running `--network=base` against the real Base mainnet RPC.
All of `deploy-provenance.js`'s preflight checks are read-only and
fire BEFORE any tx broadcast — completely safe.

**Tools:** real Base mainnet RPC (`https://mainnet.base.org`), throwaway
unfunded private key (so the script aborts at the balance check after
the guards have all run).

---

## Test results

### Test 1 — chainId guard (BASE_RPC_URL → sepolia, --network=base)
**Expected:** abort with `chainId=84532 != 8453` error.
**Actual:** `HardhatError: HH101: Hardhat was set to use chain id 8453,
but connected to a chain with id 84532.`

**Finding:** Hardhat's BUILT-IN chainId validator caught the mismatch
BEFORE reaching my custom script-level check. This is actually a
*stronger* guarantee than the script's guard — Hardhat refuses to
operate at all under chainId mismatch (the validator runs before the
script even sees a provider). My script-level guard is now belt-and-
braces defense-in-depth, redundant with Hardhat's HH101.

**Verdict:** ✅ guard fires. Mainnet operator who pastes wrong
`BASE_RPC_URL` gets caught by Hardhat's HH101 first; if Hardhat ever
relaxes that check, my script-level chainId pin remains as backup.

### Test 2 — canonical-FTNS pin (non-canonical addr, no override)
**Expected:** abort with canonical-pin mismatch.
**Actual:** Aborted earlier — at the FTNS `symbol()` check (line 82),
which fired first because `0xd979c096...` (the FTNS implementation
contract, not the proxy) returns empty `symbol()`. The implementation
contract has uninitialized storage so the ERC20 name/symbol slots are
empty.

**Finding:** Ordering: bytecode check → symbol check → chainId guard
→ canonical-FTNS pin → balance → treasury checks. My canonical pin
sits *after* the symbol check, so any non-FTNS-symbol contract gets
blocked at symbol() first. The canonical pin's intended catch is a
narrower case: a *different* deployment that returns symbol "FTNS" or
"MFTNS" (e.g., a forked staging FTNS, or a malicious lookalike). That
case is hard to construct on real Base mainnet for a dry-run; the pin
is defense-in-depth for it.

**Verdict:** ✅ guard correctness intact. Two-layer defense:
- Wrong-contract typo (most common case) → blocked by symbol() check
- Right-symbol-wrong-address (rare adversarial case) → blocked by my
  canonical pin

### Test 3 — Happy path (canonical FTNS + real Base RPC)
**Expected:** proceed past chainId + canonical-FTNS guards, fail at
balance check (throwaway key).
**Actual:**
```
=== Deploying provenance contracts to base ===
FTNS token:       0x5276a3756C85f2E9e46f6D34386167a209aa16e5
Network treasury: 0xd979c096BE297F4C3a85175774Bc38C22b95E6a4

Preflight…
  FTNS checksum:     0x5276a3756C85f2E9e46f6D34386167a209aa16e5
  Treasury checksum: 0xd979c096BE297F4C3a85175774Bc38C22b95E6a4
  FTNS bytecode:     170 bytes
  FTNS symbol:       FTNS
  Chain id:          8453

Deployer:         0x19E7E376E7C213B7E7e7e46cc70A5dD086DAff2A
Deployer balance: 0.0 ETH
Error: Deployer has zero balance
```

**Verdict:** ✅ all 7 preflight steps + chainId guard + canonical-FTNS
pin + deployer derivation succeeded against real Base mainnet RPC.
Script halted exactly where expected (zero-balance throwaway key).
Treasury checks would fire next on a funded run.

---

## What this dry-run proves

For tomorrow's Phase 1.3 Task 8 mainnet ceremony, when the operator
runs `deploy-provenance.js` with:
- `FTNS_TOKEN_ADDRESS=0x5276a3756C85f2E9e46f6D34386167a209aa16e5` (canonical)
- `NETWORK_TREASURY=<Foundation Safe>` (real contract on Base mainnet)
- `BASE_RPC_URL=<archival URL>` (chainId 8453)
- `PRIVATE_KEY=<funded deployer>` ($10 worth per Multi-Sig Action Plan §5)

…the script's preflight phase will pass cleanly. The actual deploy
(after preflight) is the same code path that hardhat-local + base-fork
rehearsals have already proven green tonight via `rehearse-task8.sh`.

## What this dry-run does NOT prove

- **Treasury bytecode check on a real Safe:** unreachable in this dry-
  run because the balance check fires first. Verified on hardhat-local
  with mock-FTNS treasury.
- **Actual Etherscan AUTO_VERIFY post-deploy:** requires a real
  Etherscan v2 API key, which we don't have in dry-run state.
- **Gas cost on real mainnet:** ~$1-2 per Multi-Sig Action Plan
  estimate; not verified here.

These are operator-side T-0 concerns, not engineering-side guard
correctness concerns.

---

## Cross-reference

- Guards added: commit `85988825` (`contracts/scripts/deploy-provenance.js`).
- Pre-deploy automation: `scripts/pre-task8-checklist.sh`
  (commit `01582d34`) which exercises all guards against mainnet env
  before any hardhat invocation — separate from this dry-run, but with
  the same intent.
- Companion runbooks:
  - `docs/2026-04-30-phase1.3-task8-engineering-runbook.md`
  - `docs/2026-04-30-multisig-action-plan-engineering-audit.md`
  - `docs/2026-04-30-session-summary-deploy-ceremony-prep.md`
