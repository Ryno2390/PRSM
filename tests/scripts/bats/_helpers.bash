# Shared bats helpers for PRSM shell-script tests.

REPO_ROOT="$(cd "$(dirname "${BATS_TEST_FILENAME}")/../../.." && pwd)"
SCRIPTS_DIR="${REPO_ROOT}/scripts"

# A throwaway PRIVATE_KEY (NOT real, NEVER funded; well-formed 0x + 64 hex)
THROWAWAY_PK="0x1111111111111111111111111111111111111111111111111111111111111111"

# Canonical Base mainnet FTNS pinned in deploy-provenance.js
CANONICAL_FTNS="0x5276a3756C85f2E9e46f6D34386167a209aa16e5"

# Hardhat default account #1 (well-known dev address; never used in real deploys)
STUB_TREASURY="0x70997970C51812dc3A010C7d01b50e0d17dc79C8"

# Skip a test when slow integration tests are not enabled.
skip_if_not_slow() {
  if [[ "${PRSM_BATS_SLOW:-0}" != "1" ]]; then
    skip "slow test (set PRSM_BATS_SLOW=1 to run)"
  fi
}

# Strip ANSI color codes from output (lets tests grep on text without
# worrying about \033[32m and friends).
strip_ansi() {
  sed -E 's/\x1b\[[0-9;]*[a-zA-Z]//g'
}
