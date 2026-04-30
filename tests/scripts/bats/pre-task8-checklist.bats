#!/usr/bin/env bats
#
# Behavior pins for scripts/pre-task8-checklist.sh.
# This is the load-bearing T-0 mainnet gate — operator runs it before
# burning gas. We pin its 10-check structure so future cosmetic edits
# don't accidentally delete a guard or change exit semantics.

load _helpers

setup() {
  SCRIPT="${SCRIPTS_DIR}/pre-task8-checklist.sh"
  [[ -x "${SCRIPT}" ]] || skip "pre-task8-checklist.sh not executable"
}

# ── Structural pins ─────────────────────────────────────────────────────

@test "pre-task8: script exists and is executable" {
  [[ -x "${SCRIPT}" ]]
}

@test "pre-task8: empty env exits 1" {
  run env -i PATH="${PATH}" HOME="${HOME}" "${SCRIPT}"
  [[ "${status}" -eq 1 ]]
}

@test "pre-task8: empty env reports all 5 missing env vars" {
  run env -i PATH="${PATH}" HOME="${HOME}" "${SCRIPT}"
  output_clean="$(echo "${output}" | strip_ansi)"
  [[ "${output_clean}" == *"PRIVATE_KEY is unset"* ]]
  [[ "${output_clean}" == *"FTNS_TOKEN_ADDRESS is unset"* ]]
  [[ "${output_clean}" == *"NETWORK_TREASURY is unset"* ]]
  [[ "${output_clean}" == *"BASE_RPC_URL is unset"* ]]
  [[ "${output_clean}" == *"ETHERSCAN_API_KEY is unset"* ]]
}

@test "pre-task8: empty env aborts before remaining checks" {
  run env -i PATH="${PATH}" HOME="${HOME}" "${SCRIPT}"
  output_clean="$(echo "${output}" | strip_ansi)"
  [[ "${output_clean}" == *"Cannot derive deployer address without PRIVATE_KEY"* ]]
  # Should NOT have run later checks
  [[ "${output_clean}" != *"chainId="* ]]
}

@test "pre-task8: malformed PRIVATE_KEY rejected by format check" {
  run env -i PATH="${PATH}" HOME="${HOME}" \
    PRIVATE_KEY="0xnothex" \
    FTNS_TOKEN_ADDRESS="${CANONICAL_FTNS}" \
    NETWORK_TREASURY="${STUB_TREASURY}" \
    BASE_RPC_URL="https://mainnet.base.org" \
    ETHERSCAN_API_KEY="bogus" \
    "${SCRIPT}"
  output_clean="$(echo "${output}" | strip_ansi)"
  [[ "${output_clean}" == *"PRIVATE_KEY must be 0x + 64 hex chars"* ]]
}

@test "pre-task8: PRIVATE_KEY is never echoed to stdout/stderr" {
  run env -i PATH="${PATH}" HOME="${HOME}" \
    PRIVATE_KEY="${THROWAWAY_PK}" \
    FTNS_TOKEN_ADDRESS="${CANONICAL_FTNS}" \
    NETWORK_TREASURY="${STUB_TREASURY}" \
    BASE_RPC_URL="https://mainnet.base.org" \
    ETHERSCAN_API_KEY="bogus" \
    "${SCRIPT}"
  # The key value itself must not appear anywhere in output
  [[ "${output}" != *"${THROWAWAY_PK}"* ]]
  # But the truncated/format-validated marker should
  output_clean="$(echo "${output}" | strip_ansi)"
  [[ "${output_clean}" == *"PRIVATE_KEY format valid"* ]]
}

@test "pre-task8: 10 checks structure (numbered headers)" {
  run env -i PATH="${PATH}" HOME="${HOME}" \
    PRIVATE_KEY="${THROWAWAY_PK}" \
    FTNS_TOKEN_ADDRESS="${CANONICAL_FTNS}" \
    NETWORK_TREASURY="${STUB_TREASURY}" \
    BASE_RPC_URL="https://mainnet.base.org" \
    ETHERSCAN_API_KEY="bogus" \
    "${SCRIPT}"
  output_clean="$(echo "${output}" | strip_ansi)"
  for n in 1 2 3 4 5 6 7 8 9 10; do
    [[ "${output_clean}" == *"[${n}]"* ]] \
      || (echo "missing [${n}] header in output" && false)
  done
}

@test "pre-task8: success-path check labels present" {
  run env -i PATH="${PATH}" HOME="${HOME}" \
    PRIVATE_KEY="${THROWAWAY_PK}" \
    FTNS_TOKEN_ADDRESS="${CANONICAL_FTNS}" \
    NETWORK_TREASURY="${STUB_TREASURY}" \
    BASE_RPC_URL="https://mainnet.base.org" \
    ETHERSCAN_API_KEY="bogus" \
    "${SCRIPT}"
  output_clean="$(echo "${output}" | strip_ansi)"
  # Each labeled section must print its name
  [[ "${output_clean}" == *"Required environment variables"* ]]
  [[ "${output_clean}" == *"PRIVATE_KEY format"* ]]
  [[ "${output_clean}" == *"Derive deployer address"* ]]
  [[ "${output_clean}" == *"BASE_RPC_URL reachability"* ]]
  [[ "${output_clean}" == *"Etherscan v2 API key"* ]]
  [[ "${output_clean}" == *"FTNS_TOKEN_ADDRESS canonical pin"* ]]
  [[ "${output_clean}" == *"FTNS on-chain state"* ]]
  [[ "${output_clean}" == *"NETWORK_TREASURY != deployer"* ]]
  [[ "${output_clean}" == *"NETWORK_TREASURY is a contract"* ]]
  [[ "${output_clean}" == *"Deployer balance"* ]]
}

@test "pre-task8: summary includes pass/fail counts" {
  run env -i PATH="${PATH}" HOME="${HOME}" "${SCRIPT}"
  output_clean="$(echo "${output}" | strip_ansi)"
  [[ "${output_clean}" =~ [0-9]+\ passed,\ [0-9]+\ failed ]]
}

@test "pre-task8: canonical-FTNS pin rejects non-canonical without override" {
  # We can test this without RPC by checking the pin logic — the
  # script's case-insensitive comparison will fail when address differs.
  run env -i PATH="${PATH}" HOME="${HOME}" \
    PRIVATE_KEY="${THROWAWAY_PK}" \
    FTNS_TOKEN_ADDRESS="0x1234567890123456789012345678901234567890" \
    NETWORK_TREASURY="${STUB_TREASURY}" \
    BASE_RPC_URL="https://mainnet.base.org" \
    ETHERSCAN_API_KEY="bogus" \
    "${SCRIPT}"
  output_clean="$(echo "${output}" | strip_ansi)"
  [[ "${output_clean}" == *"!= canonical"* ]]
  [[ "${output_clean}" == *"FORCE_NONCANONICAL_FTNS=1"* ]]
}

@test "pre-task8: FORCE_NONCANONICAL_FTNS=1 allows non-canonical address" {
  run env -i PATH="${PATH}" HOME="${HOME}" \
    PRIVATE_KEY="${THROWAWAY_PK}" \
    FTNS_TOKEN_ADDRESS="0x1234567890123456789012345678901234567890" \
    NETWORK_TREASURY="${STUB_TREASURY}" \
    BASE_RPC_URL="https://mainnet.base.org" \
    ETHERSCAN_API_KEY="bogus" \
    FORCE_NONCANONICAL_FTNS=1 \
    "${SCRIPT}"
  output_clean="$(echo "${output}" | strip_ansi)"
  [[ "${output_clean}" == *"FORCE_NONCANONICAL_FTNS=1 (operator opt-in)"* ]]
}

# ── Slow tests (real RPC) ─────────────────────────────────────────────

@test "pre-task8 slow: happy path against real Base mainnet RPC" {
  skip_if_not_slow
  run env -i PATH="${PATH}" HOME="${HOME}" \
    PRIVATE_KEY="${THROWAWAY_PK}" \
    FTNS_TOKEN_ADDRESS="${CANONICAL_FTNS}" \
    NETWORK_TREASURY="${STUB_TREASURY}" \
    BASE_RPC_URL="https://mainnet.base.org" \
    ETHERSCAN_API_KEY="bogus" \
    "${SCRIPT}"
  output_clean="$(echo "${output}" | strip_ansi)"
  # Throwaway key has zero balance, so at minimum balance check fails
  [[ "${status}" -eq 1 ]]
  # But chainId, FTNS state, treasury checks should all pass against real Base
  [[ "${output_clean}" == *"chainId=0x2105"* ]]
  [[ "${output_clean}" == *"FTNS bytecode present"* ]]
  [[ "${output_clean}" == *"FTNS.symbol() returns 'FTNS'"* ]]
}
