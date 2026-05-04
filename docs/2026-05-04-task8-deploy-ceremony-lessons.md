# Phase 1.3 Task 8 Deploy Ceremony — Operational Lessons

**Date:** 2026-05-04
**Ceremony tag:** `phase1.3-task8-complete-20260504`
**Manifest commit:** `2daeafec`

This doc captures friction points that came up during the live Phase 1.3 Task 8 ceremony — things the existing Multi-Sig Action Plan and runbooks didn't anticipate. Reading priority for the next ceremonies (Phase 7 Task 9 / Phase 7.1 Task 9 post-auditor): high. Each lesson includes the symptom, root cause, fix that worked, and a one-line preventive action for next time.

---

## L1. `eth_account` private key missing `0x` prefix

**Symptom.** Deploy script preflight passed all 6 sanity checks (chainId, FTNS symbol, treasury bytecode), then crashed at line 119 with:

```
TypeError: Cannot read properties of undefined (reading 'address')
    at main (.../deploy-provenance.js:119:47)
```

**Root cause.** The Multi-Sig Action Plan §5 keygen snippet uses `eth_account.Account.create()` then `acct.key.hex()`. Some `eth_account` versions return the hex without `0x` prefix. Hardhat's `pkAccounts()` helper in `hardhat.config.js` requires the canonical 0x-prefixed 64-hex format and silently returns `[]` if the format is wrong:

```javascript
const PK_RE = /^0x[0-9a-fA-F]{64}$/;
function pkAccounts() {
  const pk = process.env.PRIVATE_KEY;
  if (!pk) return [];
  if (!PK_RE.test(pk)) return [];  // ← silent failure here
  return [pk];
}
```

When `pkAccounts()` returns `[]`, hardhat connects to the network with no signer. `getSigners()` returns empty array → `deployer = undefined` → reading `.address` fails. The error message doesn't indicate that the private key was malformed.

**Fix that worked.**
```bash
# Diagnostic
echo "PK length: ${#PRIVATE_KEY}"   # should be 66 (0x + 64)
[[ "$PRIVATE_KEY" == 0x* ]] && echo "✓" || echo "✗ missing 0x"

# Fix if missing prefix
export PRIVATE_KEY="0x$PRIVATE_KEY"
```

**Preventive action for next ceremony.** Update Multi-Sig Action Plan §5 keygen to print with explicit prefix check, AND add the diagnostic above as the first step of §6 handoff. Also worth: `pkAccounts()` should `console.warn` (not silent fail) when PRIVATE_KEY is set but malformed — proposed contracts/hardhat.config.js patch in §appendix.

---

## L2. Alchemy URL is network-specific (caught by chainId pin)

**Symptom.** Deploy script's chainId check failed:

```
HardhatError: HH101: Hardhat was set to use chain id 8453,
but connected to a chain with id 1.
```

**Root cause.** Alchemy issues separate URLs per network:
- Ethereum mainnet: `https://eth-mainnet.g.alchemy.com/v2/<key>`
- Base mainnet: `https://base-mainnet.g.alchemy.com/v2/<key>` ← needed
- Base Sepolia: `https://base-sepolia.g.alchemy.com/v2/<key>`

Operator copied the ETH mainnet URL by reflex (Alchemy dashboard's first-listed app). Hardhat config + the deploy script's chainId pin both caught it before any gas burned. Defense-in-depth worked exactly as designed.

**Fix that worked.**
```bash
# Re-export with correct URL
export BASE_RPC_URL="https://base-mainnet.g.alchemy.com/v2/<key>"

# Verify
curl -s -X POST "$BASE_RPC_URL" -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}'
# Must return: {"jsonrpc":"2.0","result":"0x2105","id":1}    (= 8453)
```

**Preventive action for next ceremony.** Add to `pre-task8-checklist.sh` (already does this) AND add a one-line check at top of every operator-facing deploy script: "Step 0 — paste the Alchemy URL into the curl one-liner above; verify chainId 0x2105 before continuing." Catches this BEFORE the operator burns 30 min of state setup.

---

## L3. Base sweep needs L1 data fee buffer (web3.py `gas_price` only returns L2 component)

**Symptom.** Sweep script (post-deploy hygiene per action plan §6) failed twice with:

```
web3.exceptions.Web3RPCError: {'code': -32003, 'message':
  'insufficient funds for gas * price + value:
   have 3992722281076644 want 3992728926499092'}
```

**Root cause.** Base is an OP Stack rollup. Total tx cost on Base = L2 execution gas + **L1 data fee** (cost to post calldata to Ethereum L1, often the dominant cost on L2s). `w3.eth.gas_price` returns ONLY the L2 component. The action plan's sweep script computes `fee = 21000 * gas_price` which excludes the L1 portion.

For ETH transfers on Base, the L1 data fee runs ~6 billion wei (~$0.000016 at current ETH price) — small in absolute terms, but enough to make `bal - fee` insufficient when sweeping the entire balance.

**Fix that worked.** Replace `fee = 21000 * gas_price` with a fixed safety buffer that covers L1 data fee jitter:

```python
SAFETY_BUFFER_WEI = 5 * 10**12  # 5e12 wei = ~$0.012 at current ETH price
gas_price = int(w3.eth.gas_price * 2)  # 2x safety on L2 gas
send_amt = bal - SAFETY_BUFFER_WEI
# Tx will use whatever fee the node demands; buffer absorbs L1 jitter.
```

The buffer left ~0.000005 ETH (~$0.012) abandoned at the deployer dust address. Acceptable cost for guaranteed sweep success.

**Preventive action for next ceremony.** Update the action plan §6 sweep snippet to use the buffer pattern. Optionally, more accurately: query the Optimism portal contract for L1 data fee estimate (`gasOracle.getL1Fee(rlp_encoded_tx)` on Base predeploy `0x420000000000000000000000000000000000000F`). The buffer pattern is simpler and the cost difference is rounding error — buffer it is.

---

## L4. 🚨 Private key paste in AI chat — security incident

**Symptom.** Mid-ceremony, operator pasted the deployer private key directly into the AI chat conversation while debugging the L3 sweep failure. The key appeared in chat as part of a syntax-error log message:

```
File "<string>", line 1
    from web3 import Web3; ... k = '0x925d9eba63751dfd8c07eaf...';
```

**Root cause.** Stress + rushed copy-paste. The Multi-Sig Action Plan §5 emphasized "do not screenshot the key" and "do not paste into chat", but when troubleshooting a stuck terminal command, the natural instinct is to paste the failing command verbatim — and the failing command had the key inlined as a Python literal.

**The single mitigation that mattered:** the deployer key in question had **no ongoing privileges**. ProvenanceRegistry + RoyaltyDistributor are non-Ownable + non-upgradeable; admin authority lives entirely with the Foundation Safe via the immutable `networkTreasury` constructor arg. The key controlled only the leftover ~$10 ETH at the deployer address. The mitigation:

1. Sweep funds out of compromised key immediately (using AI's tool access via `Bash` — operator authorized)
2. Confirmed sweep landed via Basescan
3. Deleted the temp script file (`shred -uz /tmp/sweep_deployer.py`)
4. Closed deployer terminal → key dies with shell

**The why this was containable:** the contracts ENFORCE that the deployer can do no harm post-deploy. If the same key paste had happened with a Foundation Safe owner key (Ledger / Trezor / OneKey hardware), the impact would be catastrophic — that key signs all Foundation treasury operations forever. Hardware-wallet keys NEVER leave the device → impossible to paste → impossible to leak by this mechanism. **Hardware wallets aren't a convenience, they're the primary defense against this exact failure mode.**

**Preventive actions for next ceremony.**

1. **Multi-Sig Action Plan §5 keygen output should NEVER print the key inlined with code.** Print key on its own line with explicit `[KEY ON THIS LINE — DO NOT COPY THIS LINE]` markers. If it must be a Python value, never as a `-c` one-liner; always via `os.environ['PRIVATE_KEY']` pulled from terminal env.

2. **§6 sweep script should be in the repo as a file** (not pasted as a one-liner). Then the user runs `python3 scripts/sweep_deployer.py` which reads the key from env var. The temp-file sweep script we built today is a starting point for committing.

3. **All operator-facing scripts that touch keys should print to stderr a banner:** "DO NOT paste this terminal's contents into any chat, document, or screenshot. The PRIVATE_KEY env var is in this shell's memory."

4. **The biggest lesson for higher-stakes keys:** hardware wallets exist for exactly this reason. The deployer key was disposable by design — that disposability was load-bearing for today's mitigation. For Phase 7 / Phase 7.1 deploy ceremonies (if they ever require an EOA deployer rather than a Safe-direct deploy), the same disposable-key pattern applies. For ANY operation where the key controls ongoing authority (Foundation Safe owner roles, FTNS DEFAULT_ADMIN, etc.) — those go through hardware ONLY, never an EOA-with-software-key path.

---

## Quick-reference: ceremony pre-flight additions for next time

The existing `scripts/pre-task8-checklist.sh` is comprehensive but missed these. Add for next ceremony:

```bash
# Verify PRIVATE_KEY format BEFORE running deploy
[[ "$PRIVATE_KEY" =~ ^0x[0-9a-fA-F]{64}$ ]] || {
  echo "FAIL: PRIVATE_KEY must be 0x-prefixed 64 hex chars (got ${#PRIVATE_KEY} chars)"
  exit 1
}

# Verify BASE_RPC_URL points to Base mainnet
chainid=$(curl -s -X POST "$BASE_RPC_URL" -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' \
  | python3 -c "import sys,json; print(int(json.load(sys.stdin)['result'], 16))")
[[ "$chainid" == "8453" ]] || {
  echo "FAIL: BASE_RPC_URL points to chainId $chainid, expected 8453 (Base mainnet)"
  exit 1
}

# Print one-time security banner before deploy
cat << 'BANNER'
⚠️  SECURITY BANNER ⚠️
Your deployer PRIVATE_KEY is in this shell's environment.
DO NOT:
  - Paste this terminal's contents into chat / docs / screenshots
  - Send the env var contents anywhere
  - Run untrusted commands that might exfiltrate env vars
After ceremony: 'exit' kills this shell; key dies with it.
BANNER
```

---

## Cross-references

- Multi-Sig Action Plan: `docs/Multi-Sig_Action_Plan.md` (operator side, in iCloud Vault)
- Phase 1.3 Task 8 engineering runbook: `docs/2026-04-30-phase1.3-task8-engineering-runbook.md`
- Cumulative audit-prep §7.16: `docs/2026-04-27-cumulative-audit-prep.md` (ceremony executed entry at top of §7.16)
- Deployment manifest: `contracts/deployments/provenance-base-1777917793612.json`
- Today's commits: `git log phase1.3-task8-complete-20260504~5..phase1.3-task8-complete-20260504`
