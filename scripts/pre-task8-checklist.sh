#!/usr/bin/env bash
#
# Phase 1.3 Task 8 — pre-deploy mainnet checklist.
#
# Runs every verification that deploy-provenance.js will do at runtime,
# BEFORE the operator burns gas on a doomed deploy. Each check prints
# green ✓ or red ❌. Exits 0 only if all checks pass.
#
# Wires findings F1 / F5 / F9 from
# docs/2026-04-30-multisig-action-plan-engineering-audit.md into
# executable pre-flight infrastructure: rather than ask the operator
# to remember to check 8 things by hand at T-0, run this script.
#
# Required env vars (mainnet):
#   PRIVATE_KEY          - deployer key (0x + 64 hex chars). Used ONLY
#                           to derive the public address; never echoed.
#   FTNS_TOKEN_ADDRESS   - expected = 0x5276a3756C85f2E9e46f6D34386167a209aa16e5
#   NETWORK_TREASURY     - Foundation Safe address from Multi-Sig Action Plan §3.3
#   BASE_RPC_URL         - archival RPC endpoint
#   ETHERSCAN_API_KEY    - Etherscan v2 unified key (for AUTO_VERIFY)
#
# Optional:
#   FORCE_NONCANONICAL_FTNS=1  - allow FTNS_TOKEN_ADDRESS != 0x5276…
#                                (matches deploy-provenance.js semantics)
#
# Usage:
#   source .env.mainnet  # or export each var
#   ./scripts/pre-task8-checklist.sh
#
# Exit codes:
#   0 = all green; safe to proceed to deploy
#   1 = at least one ❌; investigate before touching mainnet

set -uo pipefail
# NOTE: not -e — we want to run all checks even if some fail, so the
# operator gets the full picture in one pass instead of fixing-then-
# rerunning serially.

CANONICAL_FTNS="0x5276a3756C85f2E9e46f6D34386167a209aa16e5"
EXPECTED_CHAIN_ID="0x2105"   # 8453 hex
MIN_DEPLOYER_BAL_WEI="3000000000000000"  # 0.003 ETH (~$6 at $2K ETH;
                                          # generous for 2-tx deploy + sweep)

PASS=0
FAIL=0

green() { printf "  \033[32m✓\033[0m %s\n" "$1"; PASS=$((PASS+1)); }
red()   { printf "  \033[31m❌\033[0m %s\n" "$1"; FAIL=$((FAIL+1)); }
hint()  { printf "  \033[31m  %s\033[0m\n" "$1"; }   # red text, no counter bump
hdr()   { printf "\n\033[1m%s\033[0m\n" "$1"; }

echo "=== Phase 1.3 Task 8 pre-deploy checklist ==="

# ── 1. Required env vars present ────────────────────────────────────────
hdr "[1] Required environment variables"
for var in PRIVATE_KEY FTNS_TOKEN_ADDRESS NETWORK_TREASURY BASE_RPC_URL ETHERSCAN_API_KEY; do
  if [[ -z "${!var:-}" ]]; then
    red "${var} is unset"
  else
    green "${var} present"
  fi
done

# Stop early if PRIVATE_KEY is missing — the rest of the checks depend
# on deriving the deployer address from it.
if [[ -z "${PRIVATE_KEY:-}" ]]; then
  red "Cannot derive deployer address without PRIVATE_KEY. Aborting remaining checks."
  printf "\n\033[1mResult: %d passed, %d failed\033[0m\n" $PASS $FAIL
  exit 1
fi

# ── 2. PRIVATE_KEY format ───────────────────────────────────────────────
hdr "[2] PRIVATE_KEY format"
if [[ ! "${PRIVATE_KEY}" =~ ^0x[0-9a-fA-F]{64}$ ]]; then
  red "PRIVATE_KEY must be 0x + 64 hex chars (got length ${#PRIVATE_KEY})"
else
  green "PRIVATE_KEY format valid (0x + 64 hex chars)"
fi

# ── 3. Derive deployer address ──────────────────────────────────────────
# Use ethers (already a project dep) to derive the address. Stash key
# on stdin to avoid argv exposure.
hdr "[3] Derive deployer address"
DEPLOYER_ADDR=""
if command -v node >/dev/null 2>&1; then
  DEPLOYER_ADDR="$(
    node -e '
      const { Wallet } = require("ethers");
      try {
        const pk = process.env.PRIVATE_KEY;
        const w = new Wallet(pk);
        console.log(w.address);
      } catch (e) {
        process.exit(2);
      }
    ' --experimental-vm-modules 2>/dev/null \
    | tr -d '[:space:]'
  )" || true
fi
if [[ -z "${DEPLOYER_ADDR}" ]] || [[ ! "${DEPLOYER_ADDR}" =~ ^0x[0-9a-fA-F]{40}$ ]]; then
  # Fallback: try via the contracts/ npm install if the top-level fails.
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  DEPLOYER_ADDR="$(
    cd "${REPO_ROOT}/contracts" && node -e '
      const { Wallet } = require("ethers");
      try {
        const w = new Wallet(process.env.PRIVATE_KEY);
        console.log(w.address);
      } catch (e) { process.exit(2); }
    ' 2>/dev/null | tr -d '[:space:]'
  )" || true
fi
if [[ -n "${DEPLOYER_ADDR}" ]] && [[ "${DEPLOYER_ADDR}" =~ ^0x[0-9a-fA-F]{40}$ ]]; then
  green "Deployer address derived: ${DEPLOYER_ADDR}"
else
  red "Could not derive deployer address from PRIVATE_KEY"
  DEPLOYER_ADDR=""
fi

# ── 4. RPC reachability + chainId ───────────────────────────────────────
hdr "[4] BASE_RPC_URL reachability + chainId"
RPC_RESP="$(curl -s -m 10 -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' \
  "${BASE_RPC_URL}" 2>/dev/null || true)"
RPC_CHAIN_ID="$(echo "${RPC_RESP}" | grep -oE '"result":"0x[0-9a-fA-F]+"' | head -1 | sed 's/.*"\(0x[0-9a-fA-F]*\)"/\1/')"
if [[ -z "${RPC_CHAIN_ID}" ]]; then
  red "RPC unreachable or no chainId in response (response: ${RPC_RESP:0:120})"
elif [[ "${RPC_CHAIN_ID}" != "${EXPECTED_CHAIN_ID}" ]]; then
  red "RPC chainId=${RPC_CHAIN_ID}, expected ${EXPECTED_CHAIN_ID} (Base mainnet 8453)"
  hint "→ BASE_RPC_URL likely points at the wrong network. Check the URL."
else
  green "RPC reachable; chainId=${RPC_CHAIN_ID} (Base mainnet 8453)"
fi

# ── 5. Etherscan API key validity ───────────────────────────────────────
hdr "[5] Etherscan v2 API key"
ES_RESP="$(curl -s -m 10 \
  "https://api.etherscan.io/v2/api?chainid=8453&module=stats&action=ethsupply&apikey=${ETHERSCAN_API_KEY}" \
  2>/dev/null || true)"
if echo "${ES_RESP}" | grep -q '"status":"1"'; then
  green "Etherscan key works (Base mainnet API reachable)"
elif echo "${ES_RESP}" | grep -q "Invalid API Key"; then
  red "Etherscan key invalid (API rejected)"
elif echo "${ES_RESP}" | grep -q "rate limit"; then
  red "Etherscan rate-limited (key may work but too many requests)"
else
  red "Etherscan response unexpected: ${ES_RESP:0:120}"
fi

# ── 6. FTNS_TOKEN_ADDRESS canonical pin ─────────────────────────────────
hdr "[6] FTNS_TOKEN_ADDRESS canonical pin"
FTNS_LOWER="$(echo "${FTNS_TOKEN_ADDRESS}" | tr '[:upper:]' '[:lower:]')"
CANONICAL_LOWER="$(echo "${CANONICAL_FTNS}" | tr '[:upper:]' '[:lower:]')"
if [[ "${FTNS_LOWER}" == "${CANONICAL_LOWER}" ]]; then
  green "FTNS_TOKEN_ADDRESS matches canonical (${CANONICAL_FTNS})"
elif [[ "${FORCE_NONCANONICAL_FTNS:-0}" == "1" ]]; then
  green "FTNS_TOKEN_ADDRESS is non-canonical, but FORCE_NONCANONICAL_FTNS=1 (operator opt-in)"
else
  red "FTNS_TOKEN_ADDRESS=${FTNS_TOKEN_ADDRESS} != canonical ${CANONICAL_FTNS}"
  hint "→ if intentional (new production token), set FORCE_NONCANONICAL_FTNS=1"
fi

# ── 7. FTNS on-chain sanity (bytecode + symbol + totalSupply) ──────────
hdr "[7] FTNS on-chain state"
FTNS_CODE_RESP="$(curl -s -m 10 -X POST -H "Content-Type: application/json" \
  --data "{\"jsonrpc\":\"2.0\",\"method\":\"eth_getCode\",\"params\":[\"${FTNS_TOKEN_ADDRESS}\",\"latest\"],\"id\":1}" \
  "${BASE_RPC_URL}" 2>/dev/null || true)"
FTNS_CODE="$(echo "${FTNS_CODE_RESP}" | grep -oE '"result":"0x[0-9a-fA-F]*"' | head -1 | sed 's/.*"\(0x[0-9a-fA-F]*\)"/\1/')"
if [[ -z "${FTNS_CODE}" ]] || [[ "${FTNS_CODE}" == "0x" ]] || [[ "${FTNS_CODE}" == "0x0" ]]; then
  red "FTNS_TOKEN_ADDRESS has no bytecode on Base mainnet"
else
  green "FTNS bytecode present ($((${#FTNS_CODE} / 2 - 1)) bytes)"
fi
# symbol() = 0x95d89b41
SYM_RESP="$(curl -s -m 10 -X POST -H "Content-Type: application/json" \
  --data "{\"jsonrpc\":\"2.0\",\"method\":\"eth_call\",\"params\":[{\"to\":\"${FTNS_TOKEN_ADDRESS}\",\"data\":\"0x95d89b41\"},\"latest\"],\"id\":1}" \
  "${BASE_RPC_URL}" 2>/dev/null || true)"
# Decode last 64 hex chars before any trailing zeroes — quick and dirty.
# "FTNS" hex = 46544e53 → expect those 8 chars in the response.
if echo "${SYM_RESP}" | grep -qi '46544e53'; then
  green "FTNS.symbol() returns 'FTNS'"
else
  red "FTNS.symbol() did not return 'FTNS'. Response: ${SYM_RESP:0:120}"
fi

# ── 8. NETWORK_TREASURY != deployer ────────────────────────────────────
hdr "[8] NETWORK_TREASURY != deployer"
if [[ -z "${DEPLOYER_ADDR}" ]]; then
  red "Skipping (deployer address unavailable)"
else
  TREAS_LOWER="$(echo "${NETWORK_TREASURY}" | tr '[:upper:]' '[:lower:]')"
  DEP_LOWER="$(echo "${DEPLOYER_ADDR}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${TREAS_LOWER}" == "${DEP_LOWER}" ]]; then
    red "NETWORK_TREASURY == deployer (${DEPLOYER_ADDR}). Use the Foundation Safe."
  else
    green "NETWORK_TREASURY (${NETWORK_TREASURY}) != deployer (${DEPLOYER_ADDR})"
  fi
fi

# ── 9. NETWORK_TREASURY has bytecode (Safe sanity) ─────────────────────
hdr "[9] NETWORK_TREASURY is a contract (Safe sanity)"
TREAS_CODE_RESP="$(curl -s -m 10 -X POST -H "Content-Type: application/json" \
  --data "{\"jsonrpc\":\"2.0\",\"method\":\"eth_getCode\",\"params\":[\"${NETWORK_TREASURY}\",\"latest\"],\"id\":1}" \
  "${BASE_RPC_URL}" 2>/dev/null || true)"
TREAS_CODE="$(echo "${TREAS_CODE_RESP}" | grep -oE '"result":"0x[0-9a-fA-F]*"' | head -1 | sed 's/.*"\(0x[0-9a-fA-F]*\)"/\1/')"
if [[ -z "${TREAS_CODE}" ]] || [[ "${TREAS_CODE}" == "0x" ]] || [[ "${TREAS_CODE}" == "0x0" ]]; then
  red "NETWORK_TREASURY ${NETWORK_TREASURY} is an EOA (no bytecode)"
  hint "→ expected a deployed Safe contract. Hot wallets cannot replace 2-of-3 multi-sig."
else
  green "NETWORK_TREASURY has bytecode ($((${#TREAS_CODE} / 2 - 1)) bytes; Safe ✓)"
fi

# ── 10. Deployer balance ────────────────────────────────────────────────
hdr "[10] Deployer balance"
if [[ -z "${DEPLOYER_ADDR}" ]]; then
  red "Skipping (deployer address unavailable)"
else
  BAL_RESP="$(curl -s -m 10 -X POST -H "Content-Type: application/json" \
    --data "{\"jsonrpc\":\"2.0\",\"method\":\"eth_getBalance\",\"params\":[\"${DEPLOYER_ADDR}\",\"latest\"],\"id\":1}" \
    "${BASE_RPC_URL}" 2>/dev/null || true)"
  BAL_HEX="$(echo "${BAL_RESP}" | grep -oE '"result":"0x[0-9a-fA-F]*"' | head -1 | sed 's/.*"\(0x[0-9a-fA-F]*\)"/\1/')"
  if [[ -z "${BAL_HEX}" ]]; then
    red "Could not query deployer balance"
  else
    BAL_DEC="$(printf '%d' "${BAL_HEX}" 2>/dev/null || echo 0)"
    if [[ "${BAL_DEC}" -lt "${MIN_DEPLOYER_BAL_WEI}" ]]; then
      red "Deployer balance ${BAL_DEC} wei < minimum ${MIN_DEPLOYER_BAL_WEI} wei (~0.003 ETH)"
      hint "→ Multi-Sig Action Plan §5 funds with \$10; current balance suggests funding step skipped"
    else
      # Format as ether for human readability (whole + decimal).
      BAL_ETH="$(node -e "console.log((BigInt('${BAL_HEX}') * 1000n / (10n ** 18n))/1000)" 2>/dev/null || echo "?")"
      green "Deployer balance ${BAL_DEC} wei (≈ ${BAL_ETH} ETH; sufficient for deploy)"
    fi
  fi
fi

# ── Final summary ───────────────────────────────────────────────────────
echo
echo "=== Pre-deploy checklist summary ==="
printf "\033[1m%d passed, %d failed\033[0m\n" "${PASS}" "${FAIL}"
if [[ "${FAIL}" -gt 0 ]]; then
  printf "\n\033[31m❌ DO NOT PROCEED to deploy. Investigate failures above.\033[0m\n"
  exit 1
fi
printf "\n\033[32m✅ All checks green. Safe to proceed with:\033[0m\n"
echo "    npx hardhat run scripts/deploy-provenance.js --network base 2>&1 \\"
echo "      | tee /tmp/mainnet-deploy-\$(date +%s).log"
exit 0
