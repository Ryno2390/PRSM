#!/usr/bin/env bash
#
# End-to-end deploy rehearsal for the PRSM mainnet launch.
#
# What it does:
#   1. Starts a local hardhat node (or uses an existing one if HARDHAT_URL set).
#   2. Deploys MockERC20 as stand-in FTNS token.
#   3. Deploys the Phase 3.1+7+7.1 audit bundle (EscrowPool +
#      BatchSettlementRegistry + MockSignatureVerifier + StakeBond +
#      cross-wire).
#   4. Deploys Phase 8 emission (EmissionController + CompensationDistributor +
#      cross-wire).
#   5. Deploys Phase 7-storage (StorageSlashing + KeyDistribution).
#   6. Prints the full deployment manifest chain.
#
# Why it exists:
#   Mainnet day is a single deploy ceremony against Base with hardware
#   multi-sig as deployer. Running this script against a local hardhat node
#   catches constructor-arg typos, cross-wire ordering bugs, and invariant
#   violations BEFORE burning mainnet gas and BEFORE the multi-sig ceremony
#   eats wall-clock hours per tx.
#
# Usage:
#   # Local hardhat (default):
#   ./scripts/rehearse-deploy.sh
#
#   # Base Sepolia (broadcasts real testnet txs):
#   NETWORK=base-sepolia PRIVATE_KEY=0x... FTNS_TOKEN_ADDRESS=0x... \
#       ./scripts/rehearse-deploy.sh
#
# Exit codes:
#   0 = all deploys + invariant checks passed
#   1 = any step failed (deploy / cross-wire / invariant)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTRACTS="${REPO_ROOT}/contracts"
NETWORK="${NETWORK:-hardhat-local}"

echo "=== PRSM mainnet deploy rehearsal ==="
echo "Network: ${NETWORK}"
echo "Repo:    ${REPO_ROOT}"
echo

cd "${CONTRACTS}"

NODE_PID=""
cleanup() {
  if [[ -n "${NODE_PID}" ]] && kill -0 "${NODE_PID}" 2>/dev/null; then
    echo
    echo "Stopping hardhat node (pid ${NODE_PID})…"
    kill "${NODE_PID}" 2>/dev/null || true
    wait "${NODE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ── Launch local node if rehearsing against hardhat-local ──────────────
if [[ "${NETWORK}" == "hardhat-local" ]]; then
  echo "[0/5] Starting hardhat node in background…"
  npx hardhat node --hostname 127.0.0.1 --port 8545 > /tmp/prsm-rehearsal-node.log 2>&1 &
  NODE_PID=$!

  # Wait for node to come up
  for i in $(seq 1 30); do
    if curl -s -X POST -H "Content-Type: application/json" \
         --data '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' \
         http://127.0.0.1:8545 > /dev/null 2>&1; then
      break
    fi
    sleep 0.5
  done
  echo "   hardhat node up at http://127.0.0.1:8545 (pid ${NODE_PID})"
  HARDHAT_NETWORK_FLAG="--network localhost"
else
  HARDHAT_NETWORK_FLAG="--network ${NETWORK}"
fi

# ── 1. Mock FTNS (testnet/local only) ──────────────────────────────────
if [[ -z "${FTNS_TOKEN_ADDRESS:-}" ]]; then
  if [[ "${NETWORK}" == "base" || "${NETWORK}" == "mainnet" ]]; then
    echo "❌ FTNS_TOKEN_ADDRESS required on mainnet." >&2
    exit 1
  fi
  echo
  echo "[1/5] Deploying MockERC20 (MFTNS stand-in)…"
  MOCK_OUT="$(npx hardhat run scripts/deploy-mock-ftns.js ${HARDHAT_NETWORK_FLAG})"
  echo "${MOCK_OUT}"
  export FTNS_TOKEN_ADDRESS="$(echo "${MOCK_OUT}" | grep -oE 'MockERC20: 0x[0-9a-fA-F]{40}' | head -1 | awk '{print $2}')"
  if [[ -z "${FTNS_TOKEN_ADDRESS}" ]]; then
    echo "❌ could not parse MockERC20 address from deploy output" >&2
    exit 1
  fi
  echo "   FTNS_TOKEN_ADDRESS=${FTNS_TOKEN_ADDRESS}"
else
  echo "[1/5] Using pre-existing FTNS token at ${FTNS_TOKEN_ADDRESS}"
fi

# ── 2. Audit bundle (Phase 3.1 + 7 + 7.1) ─────────────────────────────
echo
echo "[2/5] Deploying audit bundle (Phase 3.1 + 7 + 7.1)…"
: "${FOUNDATION_RESERVE_WALLET:=0x000000000000000000000000000000000000dEaD}"
export FOUNDATION_RESERVE_WALLET
BUNDLE_OUT="$(npx hardhat run scripts/deploy-audit-bundle.js ${HARDHAT_NETWORK_FLAG})"
echo "${BUNDLE_OUT}"
STAKE_BOND_ADDRESS="$(echo "${BUNDLE_OUT}" | grep -oE 'StakeBond:[[:space:]]+0x[0-9a-fA-F]{40}' | head -1 | awk '{print $2}')"
if [[ -z "${STAKE_BOND_ADDRESS}" ]]; then
  echo "❌ could not parse StakeBond address" >&2
  exit 1
fi
export STAKE_BOND_ADDRESS
echo "   StakeBond parsed → ${STAKE_BOND_ADDRESS}"

# ── 3. Phase 8 emission ───────────────────────────────────────────────
echo
echo "[3/5] Deploying Phase 8 emission…"
: "${CREATOR_POOL:=0x00000000000000000000000000000000000c7ea0}"
: "${OPERATOR_POOL:=0x0000000000000000000000000000000000000fee}"
: "${GRANT_POOL:=0x00000000000000000000000000000000000c07a0}"
export CREATOR_POOL OPERATOR_POOL GRANT_POOL
npx hardhat run scripts/deploy-phase8-emission.js ${HARDHAT_NETWORK_FLAG}

# ── 4. Phase 7-storage ────────────────────────────────────────────────
echo
echo "[4/5] Deploying Phase 7-storage…"
: "${AUTHORIZED_VERIFIER:=${FOUNDATION_RESERVE_WALLET}}"
export AUTHORIZED_VERIFIER
npx hardhat run scripts/deploy-phase7-storage.js ${HARDHAT_NETWORK_FLAG}

# ── 5. Summary ────────────────────────────────────────────────────────
echo
echo "[5/5] Rehearsal complete."
echo
echo "Manifests:"
ls -1t "${CONTRACTS}/deployments/" | head -10 | while read -r f; do
  echo "   ${CONTRACTS}/deployments/${f}"
done

echo
echo "✅ All deploys + invariant checks passed on ${NETWORK}."
