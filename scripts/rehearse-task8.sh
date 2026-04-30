#!/usr/bin/env bash
#
# Phase 1.3 Task 8 — deploy-ceremony rehearsal.
#
# Mirrors what scripts/rehearse-deploy.sh does for the post-audit
# audit-bundle ceremony, but for the much-narrower Task 8 scope
# (ProvenanceRegistry + RoyaltyDistributor with Foundation Safe as
# NETWORK_TREASURY).
#
# What it does:
#   1. Spins up a local hardhat node (or uses HARDHAT_URL if set).
#   2. Deploys MockERC20 (MFTNS stand-in).
#   3. Deploys ProvenanceRegistry + RoyaltyDistributor against the mock,
#      with hardhat default account #1 as a stub treasury.
#   4. Runs verify-provenance-deployment.js to confirm all immutable
#      getters match the manifest.
#
# Why it exists:
#   Task 8 mainnet is a single deploy ceremony with a disposable hot
#   key. RoyaltyDistributor's constructor args (ftns, registry, treasury)
#   are IMMUTABLE — there is no upgrade path, no transferOwnership step.
#   A typo permanently routes the 2% royalty fee to the wrong address.
#   Running this rehearsal locally catches:
#     - env-var typos
#     - hardhat-config drift after dependency bumps
#     - new mainnet hardening guards in deploy-provenance.js firing
#       incorrectly on testnet
#     - getter-name regressions (ftns/registry/networkTreasury)
#
# Usage:
#   # Local hardhat (default):
#   ./scripts/rehearse-task8.sh
#
#   # Base Sepolia (broadcasts real testnet txs; requires PRIVATE_KEY +
#   # FTNS_TOKEN_ADDRESS pointing at a base-sepolia FTNS):
#   NETWORK=base-sepolia PRIVATE_KEY=0x... \
#       FTNS_TOKEN_ADDRESS=0x... NETWORK_TREASURY=0x... \
#       ./scripts/rehearse-task8.sh
#
# Exit codes:
#   0 = deploy + verify both passed
#   1 = any step failed

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTRACTS="${REPO_ROOT}/contracts"
NETWORK="${NETWORK:-hardhat-local}"

echo "=== Phase 1.3 Task 8 deploy-ceremony rehearsal ==="
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
  echo "[0/3] Starting hardhat node in background…"
  npx hardhat node --hostname 127.0.0.1 --port 8545 > /tmp/prsm-task8-rehearsal-node.log 2>&1 &
  NODE_PID=$!

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

# ── 1. FTNS token resolution ──────────────────────────────────────────
if [[ -z "${FTNS_TOKEN_ADDRESS:-}" ]]; then
  if [[ "${NETWORK}" == "base" || "${NETWORK}" == "mainnet" ]]; then
    echo "❌ FTNS_TOKEN_ADDRESS required on ${NETWORK} (use canonical 0x5276…)" >&2
    exit 1
  fi
  echo
  echo "[1/3] Deploying MockERC20 (MFTNS stand-in)…"
  MOCK_OUT="$(npx hardhat run scripts/deploy-mock-ftns.js ${HARDHAT_NETWORK_FLAG})"
  echo "${MOCK_OUT}"
  export FTNS_TOKEN_ADDRESS="$(echo "${MOCK_OUT}" | grep -oE 'MockERC20: 0x[0-9a-fA-F]{40}' | head -1 | awk '{print $2}')"
  if [[ -z "${FTNS_TOKEN_ADDRESS}" ]]; then
    echo "❌ could not parse MockERC20 address from deploy output" >&2
    exit 1
  fi
  echo "   FTNS_TOKEN_ADDRESS=${FTNS_TOKEN_ADDRESS}"
else
  echo "[1/3] Using pre-existing FTNS token at ${FTNS_TOKEN_ADDRESS}"
fi

# ── 2. NETWORK_TREASURY resolution ────────────────────────────────────
# On mainnet/testnet, operator passes NETWORK_TREASURY explicitly.
# On hardhat-local, default to hardhat default account #1 (well-known dev
# key, never used in real deploys). This MUST NOT equal the deployer
# (account #0) — deploy-provenance.js refuses if they match.
if [[ -z "${NETWORK_TREASURY:-}" ]]; then
  if [[ "${NETWORK}" == "hardhat-local" ]]; then
    NETWORK_TREASURY="0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
    echo "   Defaulting NETWORK_TREASURY=${NETWORK_TREASURY} (hardhat account #1) for rehearsal."
  else
    echo "❌ NETWORK_TREASURY required on ${NETWORK}" >&2
    exit 1
  fi
fi
export NETWORK_TREASURY

# ── 3. Deploy provenance ──────────────────────────────────────────────
echo
echo "[2/3] Deploying ProvenanceRegistry + RoyaltyDistributor…"
DEPLOY_OUT="$(npx hardhat run scripts/deploy-provenance.js ${HARDHAT_NETWORK_FLAG})"
echo "${DEPLOY_OUT}"
MANIFEST="$(echo "${DEPLOY_OUT}" | grep -oE 'Manifest saved → [^ ]+\.json' | head -1 | sed 's/Manifest saved → //')"
if [[ -z "${MANIFEST}" ]] || [[ ! -f "${MANIFEST}" ]]; then
  echo "❌ could not locate manifest from deploy output" >&2
  exit 1
fi
echo "   Manifest: ${MANIFEST}"

# ── 4. Verify on-chain state matches manifest ─────────────────────────
echo
echo "[3/3] Verifying on-chain state matches manifest…"
PROVENANCE_MANIFEST="${MANIFEST}" \
  npx hardhat run scripts/verify-provenance-deployment.js ${HARDHAT_NETWORK_FLAG}

echo
echo "✅ Task 8 rehearsal complete on ${NETWORK}."
echo "   Manifest: ${MANIFEST}"
