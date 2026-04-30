#!/usr/bin/env bats
#
# Behavior pins for scripts/post-task8-handoff-checklist.sh.
# Pins the network-aware artifact-template fix from commit 7ddf87b3
# (caught during the chain test) — generated PR body + memory stub
# must reflect the actual manifest's network, not hardcode "Base mainnet".

load _helpers

setup() {
  SCRIPT="${SCRIPTS_DIR}/post-task8-handoff-checklist.sh"
  [[ -x "${SCRIPT}" ]] || skip "post-task8-handoff-checklist.sh not executable"
  TEST_TMP="$(mktemp -d)"
}

teardown() {
  rm -rf "${TEST_TMP}"
}

# Helper: write a synthetic manifest JSON to a tmpfile
write_manifest() {
  local network="${1}"
  local chain_id="${2}"
  cat > "${TEST_TMP}/manifest.json" <<EOF
{
  "network": "${network}",
  "chainId": "${chain_id}",
  "timestamp": "2026-04-30T12:00:00.000Z",
  "deployer": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
  "contracts": {
    "ProvenanceRegistry": "0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0",
    "RoyaltyDistributor": "0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9",
    "FTNSToken": "0x5276a3756C85f2E9e46f6D34386167a209aa16e5",
    "NetworkTreasury": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
  }
}
EOF
  echo "${TEST_TMP}/manifest.json"
}

# ── Structural pins ────────────────────────────────────────────────────

@test "post-task8: missing MAINNET_MANIFEST exits 1" {
  run env -i PATH="${PATH}" HOME="${HOME}" "${SCRIPT}"
  [[ "${status}" -eq 1 ]]
  [[ "${output}" == *"MAINNET_MANIFEST env var must point at"* ]]
}

@test "post-task8: nonexistent manifest exits 1" {
  run env -i PATH="${PATH}" HOME="${HOME}" \
    MAINNET_MANIFEST=/nonexistent/path.json "${SCRIPT}"
  [[ "${status}" -eq 1 ]]
}

@test "post-task8: base manifest produces 'Base mainnet' label (NOT redundant chainId)" {
  manifest="$(write_manifest "base" "8453")"
  cd "${REPO_ROOT}"
  run env PATH="${PATH}" HOME="${HOME}" MAINNET_MANIFEST="${manifest}" "${SCRIPT}"
  [[ "${status}" -eq 0 ]]
  # Find the most recently generated artifacts
  pr_path="$(ls -1t /tmp/task8-mainnet-handoff-pr-body-*.md 2>/dev/null | head -1)"
  mem_path="$(ls -1t /tmp/task8-mainnet-deploy-memory-stub-*.md 2>/dev/null | head -1)"
  [[ -n "${pr_path}" ]] && [[ -f "${pr_path}" ]]
  [[ -n "${mem_path}" ]] && [[ -f "${mem_path}" ]]
  # PR body title should say "Base mainnet" (not "base" or "(Base mainnet (chainId 8453))")
  pr_head="$(head -1 "${pr_path}")"
  [[ "${pr_head}" == *"Base mainnet"* ]]
  [[ "${pr_head}" != *"chainId 8453"* ]]
  # Memory stub deploy line should say "on Base mainnet via" (no duplicate chainId)
  mem_body="$(grep "Deploy completed" "${mem_path}")"
  [[ "${mem_body}" == *"on Base mainnet via"* ]]
  [[ "${mem_body}" != *"(chainId 8453) (chainId 8453)"* ]]
}

@test "post-task8: base-sepolia manifest produces 'Base Sepolia testnet' label" {
  manifest="$(write_manifest "base-sepolia" "84532")"
  cd "${REPO_ROOT}"
  run env PATH="${PATH}" HOME="${HOME}" MAINNET_MANIFEST="${manifest}" "${SCRIPT}"
  [[ "${status}" -eq 0 ]]
  pr_path="$(ls -1t /tmp/task8-mainnet-handoff-pr-body-*.md 2>/dev/null | head -1)"
  pr_head="$(head -1 "${pr_path}")"
  [[ "${pr_head}" == *"Base Sepolia testnet"* ]]
}

@test "post-task8: localhost/non-canonical manifest produces (network (chainId N)) format" {
  manifest="$(write_manifest "localhost" "31337")"
  cd "${REPO_ROOT}"
  run env PATH="${PATH}" HOME="${HOME}" MAINNET_MANIFEST="${manifest}" "${SCRIPT}"
  [[ "${status}" -eq 0 ]]
  pr_path="$(ls -1t /tmp/task8-mainnet-handoff-pr-body-*.md 2>/dev/null | head -1)"
  pr_head="$(head -1 "${pr_path}")"
  [[ "${pr_head}" == *"localhost"* ]]
  [[ "${pr_head}" == *"chainId 31337"* ]]
  # Header warning should fire on non-base
  output_clean="$(echo "${output}" | strip_ansi)"
  [[ "${output_clean}" == *"manifest network=localhost (not 'base')"* ]]
}

@test "post-task8: PR body draft includes verification command with correct network" {
  manifest="$(write_manifest "base" "8453")"
  cd "${REPO_ROOT}"
  run env PATH="${PATH}" HOME="${HOME}" MAINNET_MANIFEST="${manifest}" "${SCRIPT}"
  pr_path="$(ls -1t /tmp/task8-mainnet-handoff-pr-body-*.md 2>/dev/null | head -1)"
  pr_body="$(cat "${pr_path}")"
  [[ "${pr_body}" == *"--network base"* ]]
  [[ "${pr_body}" == *"verify-provenance-deployment.js"* ]]
}

@test "post-task8: integration touchpoints section enumerates 7 [2x] items" {
  manifest="$(write_manifest "base" "8453")"
  cd "${REPO_ROOT}"
  run env PATH="${PATH}" HOME="${HOME}" MAINNET_MANIFEST="${manifest}" "${SCRIPT}"
  output_clean="$(echo "${output}" | strip_ansi)"
  for tag in "2a" "2b" "2c" "2d" "2e" "2f" "2g"; do
    [[ "${output_clean}" == *"[${tag}]"* ]] \
      || (echo "missing [${tag}] section" && false)
  done
}

@test "post-task8: artifacts saved with stable timestamp suffix pattern" {
  manifest="$(write_manifest "base" "8453")"
  cd "${REPO_ROOT}"
  run env PATH="${PATH}" HOME="${HOME}" MAINNET_MANIFEST="${manifest}" "${SCRIPT}"
  output_clean="$(echo "${output}" | strip_ansi)"
  # Match Unix-timestamp-style filenames
  [[ "${output_clean}" =~ task8-mainnet-handoff-pr-body-[0-9]+\.md ]]
  [[ "${output_clean}" =~ task8-mainnet-deploy-memory-stub-[0-9]+\.md ]]
}
