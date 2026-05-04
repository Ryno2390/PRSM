#!/usr/bin/env bats
#
# Behavior pins for scripts/sweep-deployer.py.
#
# These tests exercise input validation paths (env var checks, format
# validation, chain/address mismatches). They never broadcast a tx —
# the script always errors out before reaching send_raw_transaction.
# Slow tests (real-network sweep) gated by PRSM_BATS_SLOW=1; skipped here
# to avoid spending real ETH.

load _helpers

setup() {
  SCRIPT="${REPO_ROOT}/scripts/sweep-deployer.py"
  PYTHON="${REPO_ROOT}/.venv/bin/python3"
  [[ -x "${SCRIPT}" ]] || skip "sweep-deployer.py not executable"
  [[ -x "${PYTHON}" ]] || skip "venv python not present"
}

# ── Env-var input validation ────────────────────────────────────────────

@test "sweep-deployer: missing PRIVATE_KEY exits 2" {
  run env -i PATH="${PATH}" HOME="${HOME}" "${PYTHON}" "${SCRIPT}"
  [[ "${status}" -eq 2 ]]
  [[ "${output}" == *"PRIVATE_KEY env var is not set"* ]]
}

@test "sweep-deployer: empty PRIVATE_KEY exits 2" {
  run env -i PATH="${PATH}" HOME="${HOME}" PRIVATE_KEY="" "${PYTHON}" "${SCRIPT}"
  [[ "${status}" -eq 2 ]]
  [[ "${output}" == *"PRIVATE_KEY env var is not set"* ]]
}

@test "sweep-deployer: PRIVATE_KEY missing 0x prefix gets auto-prepended (L1 fix)" {
  # Strip 0x from throwaway key; script should auto-fix.
  no_prefix_pk="${THROWAWAY_PK#0x}"
  run env -i PATH="${PATH}" HOME="${HOME}" \
    PRIVATE_KEY="${no_prefix_pk}" \
    "${PYTHON}" "${SCRIPT}"
  # Should fail later (RECOVERY_ADDR missing) but past the 0x check
  [[ "${status}" -eq 2 ]]
  [[ "${output}" == *"prepending automatically"* ]]
  [[ "${output}" == *"RECOVERY_ADDR env var is not set"* ]]
}

@test "sweep-deployer: malformed PRIVATE_KEY (wrong length) exits 2" {
  run env -i PATH="${PATH}" HOME="${HOME}" \
    PRIVATE_KEY="0xabc123" \
    "${PYTHON}" "${SCRIPT}"
  [[ "${status}" -eq 2 ]]
  [[ "${output}" == *"format invalid"* ]]
}

@test "sweep-deployer: missing RECOVERY_ADDR exits 2" {
  run env -i PATH="${PATH}" HOME="${HOME}" \
    PRIVATE_KEY="${THROWAWAY_PK}" \
    "${PYTHON}" "${SCRIPT}"
  [[ "${status}" -eq 2 ]]
  [[ "${output}" == *"RECOVERY_ADDR env var is not set"* ]]
}

@test "sweep-deployer: invalid RECOVERY_ADDR exits 2" {
  run env -i PATH="${PATH}" HOME="${HOME}" \
    PRIVATE_KEY="${THROWAWAY_PK}" \
    RECOVERY_ADDR="not-an-address" \
    "${PYTHON}" "${SCRIPT}"
  [[ "${status}" -eq 2 ]]
  [[ "${output}" == *"is not a valid address"* ]]
}

@test "sweep-deployer: chainId mismatch refuses to send (slow)" {
  skip_if_not_slow
  # Point at Ethereum mainnet RPC; script expects Base (8453).
  run env -i PATH="${PATH}" HOME="${HOME}" \
    PRIVATE_KEY="${THROWAWAY_PK}" \
    RECOVERY_ADDR="${STUB_TREASURY}" \
    BASE_RPC_URL="https://eth.llamarpc.com" \
    "${PYTHON}" "${SCRIPT}"
  [[ "${status}" -eq 3 ]]
  [[ "${output}" == *"reports chainId=1, expected 8453"* ]]
}
