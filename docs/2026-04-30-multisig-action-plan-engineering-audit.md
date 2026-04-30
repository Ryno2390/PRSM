# Multi-Sig Action Plan — Engineering Audit

**Audit target:** `Multi-Sig_Action_Plan.md` in the Foundation Obsidian
vault (created 2026-04-20). The operator-side runbook for hardware-
device initialization, Safe deployment, and Phase 1.3 Task 8 deploy
handoff.

**Audit date:** 2026-04-30, ahead of multi-sig hardware arrival 2026-05-01.

**Auditor lens:** what would an engineer catch reading this doc that an
operator-focused author might miss? The Action Plan is well-written for
operational discipline (seed handling, storage, threshold theory). This
audit looks specifically for engineering-side gaps: places where the
deploy script's behavior, the post-deploy verification, or the ops
integration won't match operator expectations.

**Output:** per-finding severity + concrete fix. The Action Plan itself
stays in the operator's vault unmodified; this doc is the engineer-side
companion the operator should read alongside it.

---

## Findings

### F1 — Action Plan predates 2026-04-30 mainnet hardening **(HIGH)**

The reference deploy command in Phase 6 (Action Plan §6 step 4) was
written 2026-04-20 against an earlier `deploy-provenance.js`. Tonight
(2026-04-30) the script gained three new mainnet-only guards:

- **chainId pin:** `--network=base` AND RPC reports chainId != 8453 →
  hard-fail before any tx. Catches `BASE_RPC_URL` typo'd to a sepolia URL.
- **Canonical FTNS pin:** refuse `FTNS_TOKEN_ADDRESS != 0x5276a3756C85f2E9e46f6D34386167a209aa16e5`
  on mainnet unless `FORCE_NONCANONICAL_FTNS=1`. Catches address typos.
- **Treasury-is-contract:** `NETWORK_TREASURY` must have bytecode (Safe)
  on mainnet, not be an EOA. Catches "operator pasted MetaMask address
  instead of Safe address" mistakes.

**Risk:** if any guard fires unexpectedly, the operator sees a new error
message and may panic-abort assuming something is wrong with the
infrastructure.

**Fix:** add a §6 sub-section to the Action Plan reading:

> The deploy script will refuse on mainnet if any of these are wrong:
> - chainId from RPC != 8453 (Base mainnet)
> - FTNS_TOKEN_ADDRESS != 0x5276a3756C85f2E9e46f6D34386167a209aa16e5
>   (the canonical Base mainnet FTNS — verified 2026-04-30)
> - NETWORK_TREASURY is an EOA rather than a Safe contract
>
> If any of these fire, STOP and read the error message. The script
> aborts before any tx is submitted; no mainnet state has changed.

### F2 — No programmatic post-deploy verification step **(HIGH)**

Action Plan §6 step 6 says "Claude Code verifies artifacts, Basescan
verification, and on-chain state match expectations." This is a manual
process — the operator pastes the deploy log to Claude Code, who
eyeballs the addresses.

**Risk:** RoyaltyDistributor's constructor args are immutable. If the
deploy-time wiring is wrong (e.g., a chain-reorg replaced the
RoyaltyDistributor address with one that has a different `networkTreasury`
than the operator intended), there's no automated check. The eyeball-
the-log workflow can miss subtle mismatches.

**Fix:** as of `92b41123` (this branch), `verify-provenance-deployment.js`
provides a single-command on-chain state check. Add to Action Plan §6:

> After step 4 (the deploy) and BEFORE step 5 (handoff log to Claude
> Code), run:
> ```bash
> PROVENANCE_MANIFEST=contracts/deployments/provenance-base-<ts>.json \
>   npx hardhat run scripts/verify-provenance-deployment.js --network base
> ```
> Exits 0 if all immutable getters (`ftns()`, `registry()`,
> `networkTreasury()`, `NETWORK_FEE_BPS()`) match the manifest. Exits 1
> with explicit ❌ lines if any don't. **Do not proceed to step 5 if
> verification fails.**

### F3 — Post-deploy ops-integration handoff not enumerated **(MEDIUM)**

Action Plan §6 step 6 says "Task 8 complete" once Claude Code verifies.
But the new ProvenanceRegistry + RoyaltyDistributor addresses need to
flow into:

1. **Production-config files** in the PRSM repo (any hard-coded testnet
   addresses or env-var defaults pointing at testnet contracts).
2. **Forta detection bots** (task #137 done, but bots are pointed at
   testnet addresses — need to be re-targeted to mainnet).
3. **Pause-tx templates** (task #138 done; templates reference contract
   names — need address swap-in for the new mainnet RoyaltyDistributor).
4. **Cumulative audit-prep doc** (`docs/2026-04-27-cumulative-audit-prep.md`
   may reference deploy addresses; refresh + retag).
5. **MEMORY.md** (project-memory entry recording the production
   addresses + deploy tx hashes for future-session reference).
6. **Public docs** (anywhere `0x...` placeholders exist for these contracts).

**Risk:** monitoring + ops infrastructure stays pointed at testnet,
silently missing mainnet incidents. Worst case: a real exploit happens
and the pause-tx template doesn't address the right contract.

**Fix:** add a Phase 7 to the Action Plan (or a §6 follow-up):

> ### Phase 7 — Ops integration (T+0 to T+24h)
>
> Within 24h of Task 8 deploy completion:
> 1. Search the repo for any hard-coded testnet addresses of
>    ProvenanceRegistry / RoyaltyDistributor and replace with mainnet.
> 2. Re-target Forta detection bots at mainnet addresses (per the
>    on-chain monitoring deployment doc).
> 3. Update pause-tx templates with mainnet RoyaltyDistributor address.
> 4. Refresh `docs/2026-04-27-cumulative-audit-prep.md` deploy-address
>    section and re-tag (`cumulative-audit-prep-mainnet-task8-<YYYYMMDD>`).
> 5. Save a project-memory entry recording mainnet addresses + deploy
>    tx hashes.

### F4 — Capture deploy output to log file not explicit **(MEDIUM)**

Action Plan §6 step 4 shows the deploy command without `tee` or output
redirection. Step 5 says "you paste the log back to Claude Code" —
implying the operator should capture the output, but doesn't show how.

**Risk:** operator runs the command, sees the manifest path on screen,
but the deploy log scrolls past. If anything went wrong mid-deploy
(e.g., a non-fatal verification warning), it's lost.

**Fix:** change Action Plan §6 step 4 reference command to:

```bash
npx hardhat run scripts/deploy-provenance.js --network base 2>&1 \
  | tee /tmp/mainnet-deploy-$(date +%s).log
```

Then step 5 references `/tmp/mainnet-deploy-<ts>.log` explicitly as the
file to paste from.

### F5 — No pre-T-0 env-var verification checklist **(MEDIUM)**

Action Plan §6 lists env vars to set in the deploy command, but doesn't
have a "verify all values are correct BEFORE running deploy" step. The
operator could `export FTNS_TOKEN_ADDRESS=...` with a typo and only see
the new canonical-FTNS-pin guard fire after invoking hardhat (which is
fine, but adds a needless cycle).

**Fix:** add a §6 pre-flight checklist immediately before the deploy
command:

```bash
# Pre-flight: confirm all env vars are correct.
echo "FTNS_TOKEN_ADDRESS=$FTNS_TOKEN_ADDRESS"   # expect 0x5276a3756C85f2E9e46f6D34386167a209aa16e5
echo "NETWORK_TREASURY=$NETWORK_TREASURY"       # expect SAFE_ADDR from Phase 3.3
echo "BASE_RPC_URL=$BASE_RPC_URL"               # archival URL recommended
echo "ETHERSCAN_API_KEY length: ${#ETHERSCAN_API_KEY}"  # expect ~34 chars
echo "PRIVATE_KEY length: ${#PRIVATE_KEY}"      # expect 66 (0x + 64 hex)
echo "AUTO_VERIFY=$AUTO_VERIFY"                 # expect 1
# Confirm RPC reachable + reports chainId 8453:
curl -s -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' \
  "$BASE_RPC_URL"
# expect: {"jsonrpc":"2.0","result":"0x2105","id":1}   (0x2105 = 8453)
```

### F6 — Phase 4 doesn't explicitly test threshold rejection **(MEDIUM)**

Action Plan §4.2 + §4.3 test that 2-of-3 signatures DO execute. The
inverse — that 1-of-3 does NOT — is implicit (after one signature, the
"Execute" button is greyed out / refuses).

**Risk:** if Safe creation in §3.2 was somehow misconfigured to
threshold=1 (UI mistake), the doc's tests would all pass without
catching the misconfiguration. A 1-of-3 Safe is operationally
equivalent to a single hot key — defeats the entire ceremony purpose.

**Fix:** add an explicit step in §4.2:

> After signing once with Ledger, **before** signing with Trezor,
> click "Execute" and confirm Safe refuses ("Need 1 more signature").
> If Safe shows execute as available with only 1 signature, **STOP** —
> threshold is misconfigured and the Safe must be re-deployed.

### F7 — Deployer-key generation echoes private key to terminal **(MEDIUM)**

Action Plan §5 generates the deployer key with:
```python
print('PRIVATE_KEY:', acct.key.hex())
```
The output ends up in terminal scrollback. `unset HISTFILE` prevents
shell-history persistence but does NOT clear the terminal scrollback —
the key is visible if the operator scrolls up (or if the terminal
session leaks via screenshot, OS clipboard history, accessibility tools,
etc.).

**Risk:** for a $10 disposable deployer key, the blast radius is small.
For the same pattern reused on a larger ceremony (post-audit deploy),
the risk grows. Worth fixing here as a habit.

**Fix:** alternative pattern that avoids screen exposure:

```python
import os
from eth_account import Account
acct = Account.create()
print('ADDRESS:', acct.address)
# Write key to a tmpfile with restrictive perms; never echo to screen.
key_path = f"/tmp/deployer-key-{os.getpid()}.hex"
with open(key_path, 'w') as f:
    f.write(acct.key.hex())
os.chmod(key_path, 0o600)
print(f"PRIVATE_KEY saved to: {key_path} (chmod 600)")
print("Source it before deploy: export PRIVATE_KEY=$(cat $key_path)")
print("Delete after deploy: shred -u $key_path")
```

This keeps the key off the screen, lets the operator source it
directly into the deploy command, and provides a clean shred path.

### F8 — Phase 5 deployer key recovery has stranded-fund risk **(LOW)**

§5 says "Closing the window loses the key; you'd have to regenerate
and re-fund." That phrasing understates the cost: regenerating means
the OLD funded deployer's $10 ETH is permanently stranded (no key, no
recovery, no path to sweep).

**Fix:** F7's tmpfile pattern fixes this — the key persists across
shell invocations and survives terminal-window close. Operator can
explicitly `shred -u` after Task 8 succeeds.

### F9 — No verification that Etherscan API key works pre-T-0 **(LOW)**

§6 sets `ETHERSCAN_API_KEY` for `AUTO_VERIFY=1` post-deploy verification.
If the key is wrong/expired, the deploy succeeds but verification
fails non-fatally (script logs warning, operator does it manually
later).

**Risk:** post-deploy contracts show "unverified bytecode" on Basescan
until manual verification, which auditors / users may notice and flag.

**Fix:** add a pre-flight check (alongside F5):

```bash
curl -s "https://api.etherscan.io/v2/api?chainid=8453&module=stats&action=ethsupply&apikey=${ETHERSCAN_API_KEY}" \
  | grep -q '"status":"1"' \
  && echo "Etherscan key OK" \
  || echo "❌ Etherscan key invalid or rate-limited"
```

### F10 — `BASE_RPC_URL` defaults to public endpoint **(LOW)**

Action Plan §6 reference command uses `https://mainnet.base.org` as
default. For a 2-tx Task 8 deploy, public RPC is acceptable. But:
- Public RPC is rate-limited; under bursty conditions a tx broadcast
  may drop and force re-tries.
- Public RPC doesn't expose archival data, so post-deploy state
  verification (`eth_getStorageAt` for state slots) may not be reliable.

**Fix:** soft-recommend an archival provider (Alchemy / QuickNode /
Infura) in the env-var documentation. My Task 8 engineering runbook
§1.7 already does this; sync language across docs.

---

## Cross-references between this audit and existing PRSM repo docs

| Finding | PRSM repo doc that addresses it |
|---|---|
| F1 (mainnet hardening) | `contracts/scripts/deploy-provenance.js` HEAD; commit `85988825` |
| F2 (post-deploy verification) | `contracts/scripts/verify-provenance-deployment.js`; commit `92b41123` |
| F3 (ops integration) | `docs/2026-04-30-phase1.3-task8-engineering-runbook.md` §6 |
| F4-F9 (operator-process gaps) | This audit doc — operator patches Action Plan in vault |
| F10 (RPC) | `docs/2026-04-30-phase1.3-task8-engineering-runbook.md` §1.7 |

---

## What this audit does NOT cover

Out of scope:
- Hardware-device tamper inspection (operational discipline; Action Plan
  Drop-Dead Rules cover this).
- Physical-storage choice between specific banks/SDBs (operational).
- Hardware-firmware security advisories (vendor-specific; covered in
  Action Plan §"Failure modes to watch for over time").
- Legal questions about treasury jurisdiction (separate from engineering).
- Post-Task-8 ceremony for the audit-bundle stack (covered by
  `docs/2026-04-30-post-audit-deploy-ceremony-runbook.md`, which has
  its own audit at `docs/2026-04-30-deploy-ceremony-dry-run-audit.md`).

---

## Recommended Action Plan patches (operator-side)

If the operator wants to apply the high+medium findings to the Action
Plan in their vault before T-0, here's a minimum-viable patch list:

1. **Phase 4.2** — add explicit threshold-rejection test (F6).
2. **Phase 5** — switch to tmpfile-based deployer key generation (F7+F8).
3. **Phase 6** — add pre-flight env-var checklist (F5+F9), reference the
   new mainnet hardening guards (F1), reference `verify-provenance-deployment.js`
   (F2), and capture deploy output via `tee` (F4).
4. **New Phase 7** — ops integration enumeration (F3).

The low-severity findings (F10) can ride along or get picked up later.

---

## Document provenance

- Authored 2026-04-30 ahead of multi-sig hardware arrival 2026-05-01.
- References Multi-Sig Action Plan in Foundation vault, version dated
  2026-04-20.
- References PRSM repo state at HEAD of branch `main`, commits `85988825`
  + `92b41123` (deploy-provenance hardening + verify-provenance-deployment
  rehearsal infrastructure).
- Companion to:
  - `docs/2026-04-30-phase1.3-task8-engineering-runbook.md` (engineering
    runbook for Task 8 itself)
  - `docs/2026-04-30-deploy-ceremony-dry-run-audit.md` (parallel audit
    for the post-audit ceremony)
- The Action Plan stays canonical for operator-side procedure; this
  audit doc is the engineering-side companion.
