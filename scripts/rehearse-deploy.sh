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
# FOUNDATION_RESERVE_WALLET is wired into multiple Phase-3.1/7/7.1
# constructors. On hardhat-local default to a burn address for rehearsal
# convenience; on testnet/mainnet REQUIRE explicit operator input. A
# silent fallback to 0x...dEaD on mainnet would route every emission +
# escrow refund into a burn address with no recovery.
if [[ -z "${FOUNDATION_RESERVE_WALLET:-}" ]]; then
  if [[ "${NETWORK}" == "hardhat-local" ]]; then
    FOUNDATION_RESERVE_WALLET="0x000000000000000000000000000000000000dEaD"
    echo "   Defaulting FOUNDATION_RESERVE_WALLET=${FOUNDATION_RESERVE_WALLET} for hardhat-local rehearsal."
  else
    echo "❌ FOUNDATION_RESERVE_WALLET required on ${NETWORK}." >&2
    echo "   This wires into EscrowPool / StakeBond / etc. constructors;" >&2
    echo "   defaulting to 0x...dEaD on mainnet would burn every routed payment." >&2
    echo "   Set explicitly to the Foundation 2-of-3 multi-sig (or treasury)." >&2
    exit 1
  fi
fi
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
# CREATOR_POOL / OPERATOR_POOL / GRANT_POOL are the three CompensationDistributor
# pool sinks. On hardhat-local default to vanity-byte placeholders; on
# testnet/mainnet REQUIRE explicit operator input — silent fallback would
# route every emission tx into addresses no one controls.
if [[ "${NETWORK}" == "hardhat-local" ]]; then
  : "${CREATOR_POOL:=0x00000000000000000000000000000000000c7ea0}"
  : "${OPERATOR_POOL:=0x0000000000000000000000000000000000000fee}"
  : "${GRANT_POOL:=0x00000000000000000000000000000000000c07a0}"
else
  for var in CREATOR_POOL OPERATOR_POOL GRANT_POOL; do
    if [[ -z "${!var:-}" ]]; then
      echo "❌ ${var} required on ${NETWORK}." >&2
      echo "   This is a CompensationDistributor pool sink; defaulting to a" >&2
      echo "   placeholder on mainnet would route emission to an unowned address." >&2
      echo "   Set explicitly to the Foundation-managed pool address." >&2
      exit 1
    fi
  done
fi
export CREATOR_POOL OPERATOR_POOL GRANT_POOL
npx hardhat run scripts/deploy-phase8-emission.js ${HARDHAT_NETWORK_FLAG}

# ── 4. Phase 7-storage ────────────────────────────────────────────────
echo
echo "[4/5] Deploying Phase 7-storage…"
# AUTHORIZED_VERIFIER is the Phase 7-storage proof verifier — distinct
# from FOUNDATION_RESERVE_WALLET. On testnet/mainnet operators MUST pass
# the dedicated off-chain prover EOA (or eventually a verifier contract).
# Defaulting to FOUNDATION_RESERVE_WALLET would silently misconfigure
# StorageSlashing.authorizedVerifier on mainnet.
if [[ -z "${AUTHORIZED_VERIFIER:-}" ]]; then
  if [[ "${NETWORK}" == "hardhat-local" ]]; then
    AUTHORIZED_VERIFIER="${FOUNDATION_RESERVE_WALLET}"
    echo "   Defaulting AUTHORIZED_VERIFIER=${AUTHORIZED_VERIFIER} for hardhat-local rehearsal."
  else
    echo "❌ AUTHORIZED_VERIFIER required on ${NETWORK} (the off-chain prover EOA;" >&2
    echo "   defaulting to FOUNDATION_RESERVE_WALLET would silently misconfigure" >&2
    echo "   StorageSlashing.authorizedVerifier). Set explicitly." >&2
    exit 1
  fi
fi
export AUTHORIZED_VERIFIER
npx hardhat run scripts/deploy-phase7-storage.js ${HARDHAT_NETWORK_FLAG}

# ── 5. Ownership transfer rehearsal (hardhat-local only by default) ───
# Exercises the two-phase deploy model: hot-deployer does the cross-wire,
# then transferOwnership(MULTISIG) hands all 7 Ownable contracts over.
# On testnet/mainnet the operator runs transfer-ownership.js separately
# with the real Foundation 2-of-3 multi-sig address; here we stub with
# the second hardhat default account to verify the script's invariant
# checks fire end-to-end.
if [[ "${NETWORK}" == "hardhat-local" ]] && [[ "${SKIP_TRANSFER:-0}" != "1" ]]; then
  echo
  echo "[5/6] Rehearsing ownership transfer to stub multi-sig…"
  # Hardhat default account #1 (well-known dev key, never used in real deploys).
  STUB_MULTISIG="0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
  # Newest manifests created by this rehearsal run.
  AB="$(ls -1t "${CONTRACTS}/deployments/audit-bundle-localhost-"*.json 2>/dev/null | head -1 || true)"
  P8="$(ls -1t "${CONTRACTS}/deployments/phase8-emission-localhost-"*.json 2>/dev/null | head -1 || true)"
  P7S="$(ls -1t "${CONTRACTS}/deployments/phase7-storage-localhost-"*.json 2>/dev/null | head -1 || true)"
  if [[ -z "${AB}" ]]; then
    echo "❌ no audit-bundle manifest found for transfer rehearsal" >&2
    exit 1
  fi
  FOUNDATION_MULTISIG="${STUB_MULTISIG}" \
    AUDIT_BUNDLE_MANIFEST="${AB}" \
    PHASE8_MANIFEST="${P8}" \
    PHASE7_STORAGE_MANIFEST="${P7S}" \
    npx hardhat run scripts/transfer-ownership.js ${HARDHAT_NETWORK_FLAG}

  # Idempotency check: re-run against the same manifests. Every contract
  # is already owned by the stub multi-sig; the script must skip cleanly
  # without attempting any transfer txs. Catches a re-run after a partial
  # ceremony from clobbering already-transferred contracts.
  echo
  echo "   Verifying idempotency (re-run must skip all 7)…"
  RERUN_OUT="$(FOUNDATION_MULTISIG="${STUB_MULTISIG}" \
    AUDIT_BUNDLE_MANIFEST="${AB}" \
    PHASE8_MANIFEST="${P8}" \
    PHASE7_STORAGE_MANIFEST="${P7S}" \
    npx hardhat run scripts/transfer-ownership.js ${HARDHAT_NETWORK_FLAG})"
  echo "${RERUN_OUT}" | grep -E "(already owned by multi-sig|Ownership transferred:)" || true
  if ! echo "${RERUN_OUT}" | grep -q "Ownership transferred: 0 contracts; skipped (already-multi-sig): 7"; then
    echo "❌ idempotency check failed: re-run did not skip all 7 contracts" >&2
    echo "${RERUN_OUT}" >&2
    exit 1
  fi
  echo "   ✅ idempotency holds (7/7 skipped on re-run)"
fi

# ── 6. Summary ────────────────────────────────────────────────────────
echo
echo "[6/6] Rehearsal complete."
echo
echo "Manifests:"
ls -1t "${CONTRACTS}/deployments/" | head -10 | while read -r f; do
  echo "   ${CONTRACTS}/deployments/${f}"
done

echo
echo "✅ All deploys + invariant checks passed on ${NETWORK}."
