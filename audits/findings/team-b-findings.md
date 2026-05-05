# Team B — Access Control & Ownership Findings

**Audit scope:** privilege escalation, ownership transitions, role management
**Pinned commit:** `589c14d2` (HEAD of main, identical to `cumulative-audit-prep-20260504-h`)
**On-chain context verified:** Base mainnet via `mainnet.base.org` and `base.llamarpc.com`, 2026-05-04
**PoC suite:** `contracts/test/audit-team-b/B-AccessControl-PoC.test.js` — 13 tests passing
**Output template:** Severity / Contract+Line / Attack scenario / PoC / Fix

---

## Summary

| ID | Title | Severity | Status |
|----|-------|----------|--------|
| B-VERIF-1 | `verify-audit-bundle-deployment.js` calls non-existent getters (`challengeWindow`, `unbondDelay`, `ftnsToken`) | **High** | Confirmed (PoC) |
| B-FTNS-1 | On-chain: NEITHER deployer nor Foundation Safe holds `DEFAULT_ADMIN_ROLE` on live FTNS | **Critical** | Confirmed (RPC) |
| B-RENOUNCE-1 | Sole-admin renounce path bricks FTNSTokenSimple permanently (no lock-out guard in contract) | **High** | Confirmed (PoC) |
| B-CROSS-1 | `EscrowPool.setSettlementRegistry` re-pointable post-handoff → drains escrow | **High** | Confirmed (PoC) |
| B-CROSS-2 | `EscrowPool.setFtnsToken` swap strands real-token balances (no on-chain drain check) | **Medium** | Confirmed (PoC) |
| B-CROSS-3 | `StakeBond.setSlasher` accepts EOA — single multi-sig call → arbitrary slashing + bounty farm | **High** | Confirmed (PoC) |
| B-OWNABLE-1 | Single-step `Ownable` everywhere → typo bricks contract; no `Ownable2Step` | **Medium** | Confirmed (PoC) |
| B-PAUSE-1 | All-PAUSER + admin renounce while paused → permanent freeze | **Medium** | Confirmed (PoC, edge) |
| B-TREASURY-1 | `RoyaltyDistributor.networkTreasury` immutable + only "is-contract" check → wrong-Safe deploy is permanent | **Medium** | Confirmed (PoC) |

**Headline:** the on-chain FTNS role-state is **not** in the documented post-handoff configuration. Both `0x55d2...19C74` (audit-bundle deployer) and `0x91b0...5791` (Foundation Safe) return `hasRole(DEFAULT_ADMIN_ROLE, ·) = false`. This is consistent with the `2026-04-27-cumulative-audit-prep.md §`-noted "FTNSToken DEFAULT_ADMIN_ROLE handoff … pending" — but for an outside reviewer it means **Stated Invariant #1 is not currently held on mainnet**: SOMETHING (the original FTNS-token hot key, separate from the audit-bundle deployer) holds admin and that key has not been audited as part of this scope.

---

## B-VERIF-1 — Verifier script uses wrong getter names

**Severity:** High (operational; the script that is supposed to catch a corrupted deploy will itself silently fail on routine cross-wire checks)

**File:** `contracts/scripts/verify-audit-bundle-deployment.js` lines 145-160 + 197-210

**Issue.** The verifier's ABI declarations do not match the actual contract storage-getter names:

| Verifier ABI claims | Actual contract storage |
|--|--|
| `function challengeWindow() view returns (uint256)` (line 146) | `uint256 public challengeWindowSeconds` (`BatchSettlementRegistry.sol:119`) |
| `function unbondDelay() view returns (uint256)` (line 158) | `uint256 public unbondDelaySeconds` (`StakeBond.sol:58`) |
| `function ftnsToken() view returns (address)` (line 159) | `IERC20 public immutable ftns` (`StakeBond.sol:54`) |

`compare()` invokes the wrong getters at lines 198-202 (`registry.challengeWindow()`) and 205-209 (`stakeBond.unbondDelay()`), and at line 193 (`stakeBond.ftnsToken()`).

**Attack scenario.**
1. The verifier-script invariant block at lines 197-210 is **gated** on `manifest.params.challengeWindowSeconds` / `unbondDelaySeconds` being present. Both ARE produced by `deploy-audit-bundle.js` (lines 215-216), so they ALWAYS execute on the canonical handoff manifest.
2. Each of the three calls (`challengeWindow()`, `unbondDelay()`, `ftnsToken()`) reverts because no such function exists. The `try/catch` at lines 165-169 catches that, calls `fail()`, and adds 3 to `mismatches`.
3. The script exits 1 with three confusing "getter reverted" messages — but they are not the deploy-corruption signal they pretend to be. Operator either (a) panics and unwinds a clean deploy, or (b) gets desensitized to verifier failures and starts trusting a re-run that bypasses the broken assertions.
4. **Worst case** — operator side-steps these three by removing the `params` block from the manifest. Result: the verifier no longer asserts `challengeWindowSeconds`, `unbondDelaySeconds`, or `ftns` matches anything. An attacker who briefly held deployer ownership and called `setChallengeWindowSeconds(MIN)` to shrink challenge windows would slip past the verifier.

**Confirmed PoC.** `contracts/test/audit-team-b/B-AccessControl-PoC.test.js` → "VERIFIER-ABI": deploys real `BatchSettlementRegistry` + `StakeBond`, instantiates a contract with the broken ABI from the verifier script, asserts each broken getter call reverts and each correct getter returns the expected value.

**Fix.** Edit `contracts/scripts/verify-audit-bundle-deployment.js`:

```diff
-      "function challengeWindow() view returns (uint256)",
+      "function challengeWindowSeconds() view returns (uint256)",
@@
-      "function unbondDelay() view returns (uint256)",
-      "function ftnsToken() view returns (address)",
+      "function unbondDelaySeconds() view returns (uint256)",
+      "function ftns() view returns (address)",
@@
-  await compare("stakeBond.ftnsToken", () => stakeBond.ftnsToken(), expected.FTNSToken);
+  await compare("stakeBond.ftns", () => stakeBond.ftns(), expected.FTNSToken);
@@
-      () => registry.challengeWindow(),
+      () => registry.challengeWindowSeconds(),
@@
-      () => stakeBond.unbondDelay(),
+      () => stakeBond.unbondDelaySeconds(),
```

This is a script-only fix; no on-chain change required.

---

## B-FTNS-1 — Live FTNS DEFAULT_ADMIN_ROLE state diverges from documented configuration

**Severity:** Critical (informational; Stated Invariant #1 is not held on the live mainnet contract)

**Address:** FTNSTokenSimple `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` (Base mainnet)

**Issue.** Direct RPC calls to `mainnet.base.org` and `base.llamarpc.com` (verified independently across two endpoints) return:

```
hasRole(DEFAULT_ADMIN_ROLE, 0x55d2B5623426BC65534C472b5987Cbb871619C74)  =  false   # audit-bundle deployer
hasRole(DEFAULT_ADMIN_ROLE, 0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791)  =  false   # Foundation Safe
hasRole(MINTER_ROLE,        0x55d2...)                                  =  false
hasRole(MINTER_ROLE,        0x91b0...)                                  =  false
hasRole(PAUSER_ROLE,        0x55d2...)                                  =  false
hasRole(BURNER_ROLE,        0x55d2...)                                  =  false
totalSupply()                                                          =  100_000_000 × 1e18  ✓
MINTER_ROLE() public                                                   =  0x9f2d…56a6  ✓ matches keccak256("MINTER_ROLE")
```

**Implication.** Some address (the original FTNS-token hot key, distinct from the `0x55d2...` audit-bundle deployer) holds DEFAULT_ADMIN_ROLE — that address is referenced in `docs/2026-04-27-cumulative-audit-prep.md §G4-deferred` only as "the hot key the Foundation will load onto a hardware device 2026-05-01". This key is:

- not the audit-bundle deployer this audit covers
- not in any deployment manifest in `contracts/deployments/`
- the SOLE admin authority on a live mainnet contract that holds 100M FTNS

If this key is compromised before being either (a) handed to the Safe via `transfer-ftns-roles.js` or (b) renounced via the same script, the attacker can:

- `grantRole(MINTER_ROLE, attacker)` and mint up to MAX_SUPPLY (1B - 100M = **900M FTNS**)
- push a malicious UUPS upgrade via `upgradeTo` (the implementation slot at `0xd979c096...` was independently confirmed via `eth_getStorageAt` of the EIP-1967 slot)
- `pause()` the entire token, freezing all transfers indefinitely
- Burn arbitrary balances

**Mitigation status.** External; this is a Foundation operational decision per the cited deferred-item entry. Flagging here so an external auditor sees that the live configuration does not yet satisfy "MINTER_ROLE granted by DEFAULT_ADMIN_ROLE living on the 2-of-3 Foundation Safe".

**Recommendation.**
1. Either run `scripts/transfer-ftns-roles.js` immediately (using the hardware-loaded key as the deployer signer) to transfer DEFAULT_ADMIN_ROLE to the Safe and renounce all four deployer roles, OR
2. Document explicitly in `docs/2026-05-04-task8-deploy-ceremony-lessons.md` (or successor) that the FTNS proxy is currently administered by an out-of-Safe key whose disposal-status (hardware vs renounced) is on a separate timeline, and what the rollback plan is if that key is compromised before handoff.

---

## B-RENOUNCE-1 — Sole-admin renounce permanently bricks FTNSTokenSimple

**Severity:** High (operational; one bad transaction from the multi-sig produces unrecoverable state)

**File:** `contracts/contracts/FTNSTokenSimple.sol` — uses OZ `AccessControlUpgradeable` which exposes `renounceRole(bytes32, address)`

**Stated invariant attacked:** #5 — "`renounceOwnership` is either absent or unreachable in production paths."

**Issue.** OZ `AccessControl.renounceRole` is reachable on every role-holder, including the sole `DEFAULT_ADMIN_ROLE` holder, with no `if (role == DEFAULT_ADMIN_ROLE && getRoleMemberCount(role) == 1) revert` guard. After the Foundation Safe takes admin and at any later point, **a single multi-sig signature renouncing admin permanently disables**:

- `_authorizeUpgrade` (no more UUPS upgrades — the contract's storage layout is permanently locked)
- `grantRole(MINTER_ROLE, EmissionController)` (Phase 8 emission cannot be wired up if not already done)
- All future role rotations of MINTER/BURNER/PAUSER

The script-level guard in `transfer-ftns-roles.js:191-200` ("refusing to renounce DEFAULT_ADMIN_ROLE on deployer because multi-sig has not yet received it") is correct for the deploy ceremony but does NOT protect against a mistaken renounce after the Safe owns admin. Once it is the Safe's tx, no script is in the loop.

**Confirmed PoC.** `B-AccessControl-PoC.test.js` → "B4 — renounceRole reachability": deploys the production proxy via `upgrades.deployProxy`, calls `renounceRole(DEFAULT_ADMIN_ROLE, admin)` directly, then asserts `grantRole(MINTER_ROLE, …)` reverts with `AccessControlUnauthorizedAccount`.

**Fix (defense-in-depth).** Override `renounceRole` in `FTNSTokenSimple.sol`:

```solidity
function renounceRole(bytes32 role, address callerConfirmation)
    public override
{
    if (role == DEFAULT_ADMIN_ROLE) {
        revert("DEFAULT_ADMIN_ROLE renounce disabled — use revokeRole+grantRole");
    }
    super.renounceRole(role, callerConfirmation);
}
```

This blocks the foot-gun. A genuine admin handoff must `grantRole` to the new admin first, then the previous admin uses `revokeRole(DEFAULT_ADMIN_ROLE, oldAdmin)` (the Safe's own admin authority, applied to itself) which is functionally equivalent to renounce but requires two separate authority addresses to exist.

**Caveat.** This change requires a UUPS upgrade — a chicken-and-egg problem if admin is already renounced. So this fix is only valuable if applied **before** the Foundation Safe ever holds admin, OR via an immediate post-handoff upgrade.

---

## B-CROSS-1 — Owner can re-point `EscrowPool.settlementRegistry` and drain all escrow

**Severity:** High (compromised-Safe blast radius)

**File:** `contracts/contracts/EscrowPool.sol:162-166` — `function setSettlementRegistry(address newRegistry) external onlyOwner`

**Issue.** Stated Invariant #6 is "Cross-wires are immutable post-deploy". They are NOT. `setSettlementRegistry` accepts ANY address — no contract-bytecode check, no interface assertion. After Safe handoff:

1. A compromised multi-sig (one signer + bribed second; or a Safe-module exploit; or governance attack) can call `setSettlementRegistry(attacker_eoa)`.
2. Attacker EOA then calls `settleFromRequester(victim, attacker, victim.balance)` for every requester address with a balance.
3. All escrow drained, single block.

**Confirmed PoC.** `B-AccessControl-PoC.test.js` → "B5 — cross-wire mutability": owner transfers to a fake "foundation"; foundation re-points registry to attacker EOA; attacker drains a depositor's full balance to themselves. Demonstrates 0 → 100% balance theft in 3 transactions.

**Mitigations already present (from comments in the contract):**
- "Should be paired with a PRSM-GOV-1 §10.3 14-day notice in production governance" — operational only, not enforced.

**Fix options (in order of decreasing invasiveness):**

1. **Make `settlementRegistry` immutable.** Pass it as a constructor arg; remove the setter. The deploy ceremony already does a 2-step deploy-then-wire; making it immutable forces a redeploy if rotation is needed (acceptable — registry is meant to be permanent).
2. **Timelock.** Two-phase: `proposeSettlementRegistry(addr)` records `pendingRegistry + proposedAt`; `commitSettlementRegistry()` after `MIN_NOTICE` (e.g., 14 days) finalizes. Mirrors the off-chain "14-day governance notice" but enforces it.
3. **Bytecode + interface check.** Require `IBatchSettlementRegistry(addr).escrowPool() == address(this)` in `setSettlementRegistry` so only a registry that ALSO points back at this pool can be installed (still defeated by an attacker-deployed Registry; less robust than #1 or #2).

Recommend #1 or #2.

---

## B-CROSS-2 — `EscrowPool.setFtnsToken` strands balances if drain isn't run

**Severity:** Medium

**File:** `contracts/contracts/EscrowPool.sol:183-188`

**Issue.** Comment at line 175-180 explicitly acknowledges: "we do NOT track total-balance-sum cheaply, so the 'no pending balances' check is operational-policy only. Owner MUST verify via off-chain indexing before calling. Flagged for Task 3 review + potential on-chain balance-sum tracking." Task 3 came and went; the on-chain track wasn't added.

**Attack scenario.** Operator error or compromised multi-sig calls `setFtnsToken(newFakeToken)` while real-FTNS balances are pending. All depositors discover that:

- `pool.balances[me]` still shows 100 FTNS (the credit accounting persists)
- `pool.ftns` now points at `newFakeToken`, on which the pool has zero balance
- Calling `withdraw(100)` causes `ftns.transfer(me, 100)` to revert — pool can't pay
- Real FTNS is stuck at the pool's address, with no contract function exposed to recover it

**Confirmed PoC.** `B-AccessControl-PoC.test.js` → second test in "B5". Requester deposits 100 real FTNS; owner swaps token; withdraw reverts; real FTNS observable at pool address with no recovery path.

**Fix.** Add a `totalBalances` accumulator (incremented on `deposit`, decremented on `withdraw`/`settleFromRequester`) and require `totalBalances == 0` in `setFtnsToken`. Cost: one SSTORE per deposit/withdraw (~5K gas).

---

## B-CROSS-3 — `StakeBond.setSlasher` accepts arbitrary address; compromised Safe → bounty-farming

**Severity:** High (post-handoff blast radius)

**File:** `contracts/contracts/StakeBond.sol:290-294`

**Issue.** Stated Invariant #7: "Slasher role on StakeBond = BatchSettlementRegistry address only. No other address can call slash." This is enforced ONLY at slash time (`if (msg.sender != slasher) revert`); the SETTER allows ANY address, including EOAs. After Safe handoff, a compromised Safe can:

1. `setSlasher(attacker_eoa)`
2. `attacker_eoa.slash(victim_provider, attacker_eoa, fabricated_batch_id)` for every staked provider
3. Each slash credits 70% of the slashed stake to `slashedBountyPayable[attacker_eoa]`
4. Attacker calls `claimBounty()` → drains all slashed FTNS at the bounty rate

**Confirmed PoC.** `B-AccessControl-PoC.test.js` → "owner can re-point StakeBond.slasher to attacker": provider stakes 10K at standard tier (50% slash rate); owner sets slasher to attacker EOA; attacker slashes provider; attacker's claimable bounty == 3.5K FTNS as expected (10K × 50% × 70%).

**Fix options:**
1. **Immutable slasher.** Constructor-set, no setter. Forces a redeploy of StakeBond if Registry is ever upgraded — acceptable for an immutable cross-wire.
2. **Interface assertion + timelock.** `setSlasher(addr)`: require `IBatchSettlementRegistry(addr).stakeBond() == address(this)` AND require timelock per B-CROSS-1.
3. **Allow-list of pre-approved registries.** Less general but safer than current.

Recommend #1.

---

## B-OWNABLE-1 — Single-step `transferOwnership` on every Ownable contract

**Severity:** Medium

**Files:**
- `EscrowPool.sol:38` (Ownable)
- `BatchSettlementRegistry.sol:85` (Ownable)
- `StakeBond.sol:38` (Ownable)
- `EmissionController.sol:41`, `CompensationDistributor.sol:49`, `KeyDistribution.sol:69`, `StorageSlashing.sol:54` (Ownable, out of primary scope but same pattern)

**Issue.** Every ownable contract uses OZ `Ownable` (single-step) rather than `Ownable2Step`. A typo in `transfer-ownership.js`'s `FOUNDATION_MULTISIG` env var (after passing the script's "is-contract-on-mainnet" check — which only requires bytecode, not Safe-specific signature) permanently bricks the contract: ownership transfers to an address no one controls, and there is no acceptance step that would have caught the mistake.

**Mitigations in place.**
- `transfer-ownership.js:147-167` requires `FOUNDATION_MULTISIG != deployer` AND requires bytecode on mainnet AND post-transfer asserts `owner() == multisig`.
- The post-transfer assert catches the case where the transfer succeeded but didn't stick (e.g., proxy weirdness). It does NOT catch "the multi-sig address belongs to someone else who happens to deploy a Safe at a vanity address that matches a typo'd one of yours" — extremely contrived but theoretically possible given checksum-only validation.

**Confirmed PoC.** `B-AccessControl-PoC.test.js` → "B2 — single-step transferOwnership brick risk": EscrowPool transferred to `0x...dEaD`; original deployer can no longer call owner-only setters; owner is permanently `0x...dEaD`.

**Fix.** Replace `Ownable` with `Ownable2Step` on all 7 contracts. The new owner must call `acceptOwnership()` to take over — a typo'd target address cannot accept and ownership stays with the deployer until the operator re-runs the transfer with the correct address.

The cost is a one-tx-per-contract acceptance step in the multi-sig, which adds ~10 minutes to the ceremony but eliminates the brick risk entirely.

---

## B-PAUSE-1 — All-PAUSER renounce while paused → permanent transfer freeze

**Severity:** Medium (multi-step foot-gun; requires admin renounce + every PAUSER renouncing)

**File:** `contracts/contracts/FTNSTokenSimple.sol:87-96`

**Issue.** The pause/unpause both gate on `PAUSER_ROLE`. `_update` reverts on `whenNotPaused`. If:

1. Admin renounces `DEFAULT_ADMIN_ROLE` (no future PAUSER grants possible)
2. Token is currently paused
3. Every `PAUSER_ROLE` holder renounces

…then no address can ever unpause. All transfers (including those by holders to anyone) are permanently bricked.

**Confirmed PoC.** `B-AccessControl-PoC.test.js` → "CONFIRMED edge case: if DEFAULT_ADMIN renounces AND every PAUSER renounces while paused, transfers are permanently frozen". Demonstrates `EnforcedPause` revert with no recovery.

**Fix.** Apply the same `renounceRole` override from B-RENOUNCE-1 — block renouncing `PAUSER_ROLE` if it would result in zero pausers AND the contract is paused. The contract-side check is the only safe place; an off-chain script can't see all role-holders cheaply.

---

## B-TREASURY-1 — `RoyaltyDistributor.networkTreasury` immutable + only "is-contract" check

**Severity:** Medium (one-shot, not after-deploy)

**File:** `contracts/contracts/RoyaltyDistributor.sol:32` (`address public immutable networkTreasury`); `contracts/scripts/deploy-provenance.js:155-165`

**Issue.** The deploy script verifies treasury has bytecode but does not verify it is the SPECIFIC Foundation Safe. Any contract address would pass the check.

```javascript
const treasuryCode = await hre.ethers.provider.getCode(treasuryChecksum);
if (treasuryCode === "0x" || treasuryCode === "0x0") { … }
console.log(`  Treasury bytecode: ${(treasuryCode.length / 2 - 1)} bytes (contract ✓)`);
```

A typo'd address that happens to point at, say, an unrelated ERC-20 contract or a Safe owned by a different organization, would pass. `networkTreasury` is then permanently routed to the wrong destination — 2% of ALL royalty volume forever, with no setter.

**Mitigation in place.** Operator review at signing time + immutable design as a feature, not a bug.

**Confirmed PoC.** `B-AccessControl-PoC.test.js` → "CONFIRMED secondary: networkTreasury IS immutable, but a wrong-but-nonzero address at deploy is permanent".

**Fix (defense-in-depth).** Add to `deploy-provenance.js`:

```javascript
// Verify treasury is the canonical Safe address. Override with
// FORCE_NONCANONICAL_TREASURY=1 if intentionally deploying to a different Safe.
const CANONICAL_FOUNDATION_SAFE = "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791";
if (isMainnet
    && treasuryChecksum.toLowerCase() !== CANONICAL_FOUNDATION_SAFE.toLowerCase()
    && process.env.FORCE_NONCANONICAL_TREASURY !== "1") {
  throw new Error(`NETWORK_TREASURY does not match canonical Foundation Safe`);
}
```

Mirrors the already-present `CANONICAL_FTNS_BASE_MAINNET` pin pattern. Live state for `RoyaltyDistributor` (`0x3E82...D6c2`) confirms `networkTreasury() == 0x91b0...5791` ✓ — so the current deploy was correct, but the next one needs this guardrail.

---

## Vectors evaluated and cleared

For each B1-B10, the explicit verdict:

- **B1 — Pre-handoff backdoor.** **Cleared (deploy-script side); flagged on-chain.** The audit-bundle deploy script (`deploy-audit-bundle.js`) does not grant any roles besides what the contract constructors do. `FTNSTokenSimple.initialize` grants all four roles to `initialOwner` (the deployer); `transfer-ftns-roles.js` step 2-5 then renounces all four. There is no API path to grant a hidden third address. **HOWEVER**, see B-FTNS-1: on the LIVE FTNS contract, the role state is not consistent with either deployer-holds or Safe-holds. An out-of-scope hot key holds admin. That key was not auditable by this team.

- **B2 — One-step `transferOwnership` brick.** **Confirmed.** See B-OWNABLE-1.

- **B3 — Role-graph cycles.** **Cleared.** OZ AccessControl's role-admin-of-admin chain bottoms out at `DEFAULT_ADMIN_ROLE`, which is the admin of itself by default. No subroles redefine `_setRoleAdmin` in any contract in scope (`grep -n setRoleAdmin contracts/contracts/*.sol` returns nothing). The chain cannot loop because `DEFAULT_ADMIN_ROLE`'s admin is itself, terminating evaluation.

- **B4 — `renounceRole` reachability.** **Confirmed.** See B-RENOUNCE-1.

- **B5 — Cross-wire mutability.** **Confirmed.** See B-CROSS-1, B-CROSS-2, B-CROSS-3. Stated Invariant #6 ("cross-wires are immutable post-deploy") is **not** held by current contract code.

- **B6 — Pauser locking.** **Cleared in normal operation; edge case confirmed.** See B-PAUSE-1.

- **B7 — Initializer re-entry.** **Cleared.** `FTNSTokenSimple` calls `_disableInitializers()` in its constructor (line 38), and `initialize` carries the `initializer` modifier. PoC verifies both the implementation cannot be initialized AND the proxy cannot be re-initialized. `BridgeSecurity` and `FTNSBridge` also use the same pattern (out of primary scope but same correct pattern). No non-upgradeable contract uses `initialize()` at all.

- **B8 — Slasher acceptance.** **Confirmed.** See B-CROSS-3. `StakeBond.setSlasher` accepts an arbitrary address with no interface check.

- **B9 — Foundation Safe owners + threshold.** **Verified on-chain ✓.** Direct RPC call: `Safe.getThreshold() = 2`, `Safe.getOwners() = [0x7e824c2c...e1ba, 0x0d39032a...f180, 0x1623ceba...023c]`. Three distinct owner addresses, threshold 2. The hardware-wallet attestation file referenced in MEMORY.md (`~/.prsm-foundation-private/Multi-Sig_Addresses.txt`) is out of repo scope; if those three addresses match the documented Ledger/Trezor/OneKey set, B9 is fully cleared. Auditor should compare the three addresses above to that file.

- **B10 — Constructor-arg poisoning.** **Cleared for zero-address case; confirmed for wrong-but-nonzero.** See B-TREASURY-1. `RoyaltyDistributor` rejects zero addresses (PoC). `EscrowPool` rejects zero `ftnsAddress` but accepts zero `initialRegistry` by design (registry deployed second). `BatchSettlementRegistry` validates the challenge-window range. `StakeBond` validates ftnsAddress non-zero and unbond-delay range.

---

## Special focus — deploy-ceremony attack surface

### transfer-ownership.js (read line by line)

**Line 91-95** — idempotency check: skip if owner already == multi-sig. Robust against re-runs.
**Line 97-103** — refuses transfer if current owner is neither deployer nor multi-sig. Robust against post-deploy state corruption.
**Line 107-117** — calls `transferOwnership(multisig)`, waits, re-reads, asserts. Robust modulo single-step risk (B-OWNABLE-1).
**Line 144-152** — refuses on mainnet if `multisig == deployer` (would defeat the safety property).
**Line 159-168** — refuses if multi-sig is an EOA on mainnet (Safe must have bytecode). **Weakness:** "is contract" ≠ "is the right Safe". Auditor should verify `getThreshold()` and `getOwners()` post-handoff (script does not).

**Recommendation.** Add to `transferOne()` post-transfer check:

```javascript
const safeAbi = ["function getThreshold() view returns (uint256)", "function getOwners() view returns (address[])"];
const safe = new hre.ethers.Contract(multisig, safeAbi, deployer);
const threshold = await safe.getThreshold();
const owners = await safe.getOwners();
if (threshold < 2n) throw new Error(`Safe threshold ${threshold} < 2`);
if (owners.length < 3) throw new Error(`Safe has ${owners.length} owners < 3`);
console.log(`     ✓ Safe verified: ${owners.length} owners, ${threshold}-of-N`);
```

### transfer-ftns-roles.js (read line by line)

**Line 188-200** — refuses to renounce DEFAULT_ADMIN on deployer if multi-sig hasn't received it yet. Excellent guard. **Limitation:** only protects ceremony-time. Post-ceremony, the multi-sig can still renounce. See B-RENOUNCE-1.
**Line 161-167** — confirms deployer holds admin BEFORE attempting grant. Robust.
**Line 173-174** — verifies grant stuck. Robust.
**Line 205-206** — verifies renounce stuck. Robust.
**Final invariant 211-225** — comprehensive post-state check. **Strength:** fails loudly if any role still held by deployer or admin not held by multi-sig.

**No new issue identified** in this script beyond the FTNS-side observation in B-FTNS-1 (which is about the live state, not the script).

### deploy-audit-bundle.js (read line by line)

**Line 78-85** — refuses MockSignatureVerifier on mainnet. Good.
**Line 88-94** — refuses Foundation wallet == deployer on mainnet. Good.
**Line 192-204** — post-deploy invariant checks all 6 cross-wires. Good. **But:** this only checks at-deploy state; post-handoff mutations are not detected by this script (see B-CROSS-1/2/3).

**Recommendation.** None new; covered by B-VERIF-1.

### L4 (key-paste in chat) — verification

The lessons file (`docs/2026-05-04-task8-deploy-ceremony-lessons.md` §L4) asserts that the contracts in question "ENFORCE that the deployer can do no harm post-deploy" because ProvenanceRegistry + RoyaltyDistributor are non-Ownable + non-upgradeable. **Verified ✓**:

- `ProvenanceRegistry.sol`: no `Ownable`, no `Initializable`, no roles. owner() reverts on chain (confirmed via RPC). The only privileged op is per-content `transferContentOwnership`, which only the current creator can call.
- `RoyaltyDistributor.sol`: no `Ownable`, no `Initializable`. `networkTreasury` immutable, `ftns` immutable, `registry` immutable. Stateless.

The L4 lesson is sound: a deployer-key paste of THESE contract's deployer was containable. **However**, the FTNS proxy at `0x5276a3...` is UUPS-upgradeable, and ITS deployer (the separate hot key — see B-FTNS-1) holds DEFAULT_ADMIN_ROLE which controls `_authorizeUpgrade`. If THAT key is ever pasted in chat, it is not containable: an attacker can push a malicious implementation that drains 100M existing FTNS or mints up to MAX_SUPPLY. The lessons doc is honest about this risk in §L4 paragraph 4 ("Hardware wallets aren't a convenience, they're the primary defense") — but the operational state (B-FTNS-1) implies that hardening is not yet complete on the live contract.

---

## On-chain state verification (Base mainnet)

| Address | Result | Status |
|---------|--------|--------|
| `ProvenanceRegistry 0xdF47…9915` | `owner()` reverts (not Ownable) | ✓ as designed |
| `RoyaltyDistributor 0x3E82…D6c2` `networkTreasury()` | `0x91b0…5791` (Foundation Safe) | ✓ matches manifest |
| `RoyaltyDistributor` `registry()` | `0xdF47…9915` | ✓ matches manifest |
| `RoyaltyDistributor` `ftns()` | `0x5276…16e5` | ✓ matches manifest |
| `RoyaltyDistributor` `owner()` reverts | not Ownable | ✓ as designed |
| `FTNSTokenSimple 0x5276…16e5` | `MINTER_ROLE() == 0x9f2d…56a6` | ✓ keccak("MINTER_ROLE") |
| `FTNSTokenSimple` total supply | `100_000_000 × 1e18` | ✓ matches initial |
| `FTNSTokenSimple` `hasRole(ADMIN, 0x55d2…)` | `false` | **B-FTNS-1** |
| `FTNSTokenSimple` `hasRole(ADMIN, 0x91b0…)` | `false` | **B-FTNS-1** |
| `FTNSTokenSimple` `hasRole(MINTER, 0x55d2…)` | `false` | **B-FTNS-1** |
| `FTNSTokenSimple` `hasRole(MINTER, 0x91b0…)` | `false` | **B-FTNS-1** |
| `Safe 0x91b0…5791` `getThreshold()` | `2` | ✓ |
| `Safe 0x91b0…5791` `getOwners()` | 3 owners (`0x7e82…e1ba`, `0x0d39…f180`, `0x1623…023c`) | ✓ external verification needed against MEMORY.md |
| `Safe 0x91b0…5791` bytecode | 172 bytes (Safe proxy) | ✓ |

---

## Out of scope / for other teams

- **Economic** (Team A): bounty-amount math in StakeBond, fee composition, supply caps.
- **Signature/cryptographic** (Team C): Ed25519Verifier semantics, signature-replay across batches, MerkleProof correctness.
- **State machine / composition** (Team D): batch lifecycle, challenge-window race conditions, EscrowPool ↔ Registry settlement-finalization atomicity. (Note: B-CROSS-1 has overlap — drained-via-malicious-registry is a state-composition concern but the *cause* is access-control, hence reported here.)

---

## Files produced

- `audits/findings/team-b-findings.md` — this document
- `contracts/test/audit-team-b/B-AccessControl-PoC.test.js` — 13 PoC tests, all passing under `npx hardhat test contracts/test/audit-team-b/`
