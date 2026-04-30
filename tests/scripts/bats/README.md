# Shell-script bats tests

Behavior pins for the four deploy-ceremony shell scripts in `scripts/`.
Catches regressions in argument validation, error paths, and output
format invariants without requiring a hardhat node for the fast tests.

## Coverage

| Script | Test file | Fast tests | Slow tests |
|---|---|---:|---:|
| `pre-task8-checklist.sh` | `pre-task8-checklist.bats` | 11 | 1 |
| `post-task8-handoff-checklist.sh` | `post-task8-handoff-checklist.bats` | 8 | 0 |
| `rehearse-task8.sh` + `rehearse-deploy.sh` | `rehearsal-orchestrators.bats` | 8 | 3 |
| **Total** | | **27** | **4** |

## Run

```bash
# Fast tests (~30s, no hardhat node needed)
make test-shell

# All tests including slow integration (spins hardhat node, ~3 min)
make test-shell-slow
```

Or invoke directly:
```bash
bats tests/scripts/bats/                                 # fast
PRSM_BATS_SLOW=1 bats tests/scripts/bats/                # all
bats tests/scripts/bats/pre-task8-checklist.bats         # one file
```

## Slow-test gating

Tests tagged "slow" are gated by `PRSM_BATS_SLOW=1`. They:
- Spin up a hardhat node (port 8545)
- Deploy real (mock) contracts
- Run the full rehearsal end-to-end
- Take 30-60s each

Use them for pre-merge validation; skip them for tight loops during
development.

## What the fast tests pin

- **Required env vars present + format-validated** (5 vars on `pre-task8-checklist`).
- **Error paths exit non-zero** (missing manifest, malformed key, bad mode).
- **PRIVATE_KEY never echoed to output** (security pin).
- **Numbered-section structure** (10 checks on pre-task8 checklist; 7 touchpoints on post-task8).
- **Network-aware artifact templates** (the chain-test seam-bug fix from commit `7ddf87b3`).
- **Mode rejection** (`FTNS_DEPLOY_MODE=mock` forbidden on mainnet; invalid modes rejected).

## What the slow tests pin

- **Full rehearsal succeeds end-to-end** (deploy + verify + idempotent re-run).
- **Real Base mainnet RPC connectivity** (`pre-task8-checklist` against canonical FTNS).
- **Both `FTNS_DEPLOY_MODE=mock` and `=real` paths green on hardhat-local**.

## Adding new tests

Each test file `load _helpers` for shared variables and helpers
(`THROWAWAY_PK`, `CANONICAL_FTNS`, `STUB_TREASURY`, `strip_ansi`,
`skip_if_not_slow`).

Keep fast tests under 1s each. If a test needs hardhat or RPC, gate
it with `skip_if_not_slow`.
