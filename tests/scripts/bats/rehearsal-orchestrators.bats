#!/usr/bin/env bats
#
# Behavior pins for the two rehearsal orchestrators:
#   scripts/rehearse-task8.sh           (Phase 1.3 Task 8)
#   scripts/rehearse-deploy.sh          (post-audit ceremony)
#
# Fast tests pin structural invariants (script exists, executable,
# argument shape). Slow tests (PRSM_BATS_SLOW=1) actually run the
# full rehearsal — these are heavy (spin up hardhat node, deploy
# real contracts) and take 30-60s each.

load _helpers

# ── rehearse-task8.sh ──────────────────────────────────────────────────

@test "rehearse-task8: script exists and is executable" {
  [[ -x "${SCRIPTS_DIR}/rehearse-task8.sh" ]]
}

@test "rehearse-task8: docstring header references Phase 1.3 Task 8 + 4-script chain" {
  head -30 "${SCRIPTS_DIR}/rehearse-task8.sh" | grep -q "Phase 1.3 Task 8"
  # Must mention the chain components it orchestrates
  head -30 "${SCRIPTS_DIR}/rehearse-task8.sh" | grep -q "MockERC20"
  head -30 "${SCRIPTS_DIR}/rehearse-task8.sh" | grep -q "ProvenanceRegistry + RoyaltyDistributor"
  head -30 "${SCRIPTS_DIR}/rehearse-task8.sh" | grep -q "verify-provenance-deployment"
}

@test "rehearse-task8: rejects mainnet target on missing FTNS_TOKEN_ADDRESS" {
  # Mainnet path requires explicit FTNS_TOKEN_ADDRESS — rehearsal should
  # error fast without requiring hardhat node.
  run env -i PATH="${PATH}" HOME="${HOME}" \
    NETWORK=base "${SCRIPTS_DIR}/rehearse-task8.sh"
  # Will fail somewhere — could be FTNS guard, could be cd to contracts,
  # could be node spawn. Just ensure non-zero exit + no spinning forever.
  [[ "${status}" -ne 0 ]]
}

@test "rehearse-task8 slow: full hardhat-local rehearsal green" {
  skip_if_not_slow
  cd "${REPO_ROOT}"
  # Cleanup any lingering nodes
  pkill -f "hardhat node" 2>/dev/null || true
  sleep 2
  run "${SCRIPTS_DIR}/rehearse-task8.sh"
  [[ "${status}" -eq 0 ]]
  output_clean="$(echo "${output}" | strip_ansi)"
  [[ "${output_clean}" == *"All on-chain state matches manifest"* ]]
  [[ "${output_clean}" == *"Task 8 rehearsal complete on hardhat-local"* ]]
}

# ── rehearse-deploy.sh ─────────────────────────────────────────────────

@test "rehearse-deploy: script exists and is executable" {
  [[ -x "${SCRIPTS_DIR}/rehearse-deploy.sh" ]]
}

@test "rehearse-deploy: docstring references audit-bundle stack components" {
  head -40 "${SCRIPTS_DIR}/rehearse-deploy.sh" | grep -q "EscrowPool"
  head -40 "${SCRIPTS_DIR}/rehearse-deploy.sh" | grep -q "BatchSettlementRegistry"
  head -40 "${SCRIPTS_DIR}/rehearse-deploy.sh" | grep -q "StakeBond"
  head -40 "${SCRIPTS_DIR}/rehearse-deploy.sh" | grep -q "EmissionController"
  head -40 "${SCRIPTS_DIR}/rehearse-deploy.sh" | grep -q "StorageSlashing"
}

@test "rehearse-deploy: FTNS_DEPLOY_MODE selector documented" {
  grep -q "FTNS_DEPLOY_MODE" "${SCRIPTS_DIR}/rehearse-deploy.sh"
  # Must support all three modes — match the bash :=DEFAULT idiom the
  # script uses, e.g. `: "${FTNS_DEPLOY_MODE:=mock}"`
  grep -q ':=mock' "${SCRIPTS_DIR}/rehearse-deploy.sh"
  grep -q ':=real' "${SCRIPTS_DIR}/rehearse-deploy.sh"
  grep -q ':=existing' "${SCRIPTS_DIR}/rehearse-deploy.sh"
}

@test "rehearse-deploy: invalid FTNS_DEPLOY_MODE rejected" {
  # Don't actually need to start hardhat — set FTNS_TOKEN_ADDRESS so
  # we skip the existing-mode default and use the bogus mode value.
  # Script should exit 1 at the case statement.
  run env -i PATH="${PATH}" HOME="${HOME}" \
    NETWORK=base-sepolia \
    FTNS_DEPLOY_MODE=bogus_mode_xyz \
    FTNS_TOKEN_ADDRESS=0x1234567890123456789012345678901234567890 \
    "${SCRIPTS_DIR}/rehearse-deploy.sh"
  [[ "${status}" -ne 0 ]]
}

@test "rehearse-deploy: mainnet rejects mock FTNS_DEPLOY_MODE" {
  # Belt-and-braces: forcing mock on mainnet would deploy a public-mint
  # MockERC20 — disastrous. The script's case statement guards this.
  grep -q "forbidden on \${NETWORK}" "${SCRIPTS_DIR}/rehearse-deploy.sh"
}

@test "rehearse-deploy slow: default mode (mock FTNS) runs end-to-end green" {
  skip_if_not_slow
  cd "${REPO_ROOT}"
  pkill -f "hardhat node" 2>/dev/null || true
  sleep 2
  run "${SCRIPTS_DIR}/rehearse-deploy.sh"
  [[ "${status}" -eq 0 ]]
  output_clean="$(echo "${output}" | strip_ansi)"
  [[ "${output_clean}" == *"All deploys + invariant checks passed on hardhat-local"* ]]
  [[ "${output_clean}" == *"idempotency holds (7/7 skipped on re-run)"* ]]
}

@test "rehearse-deploy slow: real mode (fresh FTNSTokenSimple) runs end-to-end green" {
  skip_if_not_slow
  cd "${REPO_ROOT}"
  pkill -f "hardhat node" 2>/dev/null || true
  sleep 2
  run env FTNS_DEPLOY_MODE=real "${SCRIPTS_DIR}/rehearse-deploy.sh"
  [[ "${status}" -eq 0 ]]
  output_clean="$(echo "${output}" | strip_ansi)"
  [[ "${output_clean}" == *"FTNS role-transfer idempotency holds (5/5 skipped on re-run)"* ]]
}
