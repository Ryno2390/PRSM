# PRSM symbolic-proofs lane

**Purpose.** Halmos-compatible Solidity test contracts that
formally prove protocol invariants. Complements the runtime
probe in `prsm/economy/web3/formal_invariants.py`:

| Layer | What it answers | Where |
|-------|-----------------|-------|
| Runtime probe | Is the CURRENT live state in spec? | `/admin/formal-verification/check?contract=…` |
| Symbolic proof | Can ANY reachable state break the spec? | This directory + `prsm/economy/web3/halmos_runner.py` |

Vision §14 item 4 commits to "formal verification on the
highest-value contracts." The runtime probe handles the
audit-time + ops-alerting use cases; the symbolic proofs
handle the threat-model "what if every input is adversarial"
use case.

## Layout

```
symbolic-proofs/
├── foundry.toml         # Foundry config (solc 0.8.22, FS allowlist)
├── README.md            # This file
├── src/                 # Spec contracts (currently empty)
├── test/                # Halmos check_* contracts
│   └── FTNSTokenSupplyCap.t.sol
└── out/                 # forge build output (gitignored)
```

## Running

**Prerequisites.** `halmos` (`pip install halmos`) and
`forge` (via [foundryup](https://getfoundry.sh)).

```bash
cd contracts/symbolic-proofs
halmos --contract FTNSSupplyCapSpec
```

Expected output:

```
Running 3 tests for test/FTNSTokenSupplyCap.t.sol:FTNSSupplyCapSpec
[PASS] check_max_supply_constant_value() (paths: 1, time: 0.00s)
[PASS] check_mint_preserves_cap(address,uint256) (paths: 5, time: 0.02s)
[PASS] check_post_construction_in_cap() (paths: 1, time: 0.00s)
Symbolic test result: 3 passed; 0 failed
```

The `(paths: 5, …)` line is halmos reporting that it
explored 5 symbolic execution paths through `mintReward`
without finding a counterexample to the supply-cap
invariant.

## Python integration

`prsm/economy/web3/halmos_runner.py` provides:

- `HalmosRunner` — invokes halmos via subprocess + parses
  output. Fail-soft if halmos / forge aren't installed
  (returns `SKIPPED` suite, never raises).
- `SymbolicProofResult` / `SymbolicProofSuite` — typed
  outcomes serializable via `to_dict()`.
- `SYMBOLIC_PROOF_CATALOG` — registry of available proofs
  keyed by Solidity contract name, with cross-references
  to the runtime-invariant IDs they mirror.

Example:

```python
from prsm.economy.web3.halmos_runner import HalmosRunner

runner = HalmosRunner()
if runner.is_available():
    suite = runner.run("FTNSSupplyCapSpec")
    print(suite.to_dict())
else:
    print(f"missing tools: {runner.missing_tools()}")
```

## Pattern for adding new proofs

1. **Identify a runtime invariant** in the
   `INVARIANT_REGISTRY` (in `formal_invariants.py`) that
   benefits from "across-all-inputs" verification — supply
   caps, solvency invariants, anti-confiscation rate
   bounds, etc.
2. **Write a minimal Solidity contract** that mirrors the
   production logic structurally. Avoid UUPS proxy /
   OpenZeppelin import noise where it doesn't affect the
   invariant — symbolic execution scales poorly with
   irrelevant complexity.
3. **Add `check_*` functions** taking symbolic args. Each
   function should be small (one invariant) and assert the
   post-state property.
4. **Add to `SYMBOLIC_PROOF_CATALOG`** in `halmos_runner.py`
   so operators / CI know to invoke it.
5. **Document the structural-equivalence claim** (which
   lines of the canonical production source the spec
   mirrors). This is the audit-visible link between
   "halmos proved spec X" and "production contract Y is
   safe."

## Current proofs

| Spec contract | Mirrors | Invariants | Paths (headline) |
|--------------|---------|-----------|------------------|
| `FTNSSupplyCapSpec` | `FTNSTokenSimple.mintReward` (lines 70-73) | INV-FT-1, INV-FT-2 | 5 paths on `check_mint_preserves_cap` |
| `RoyaltyDistributorSolvencySpec` | `RoyaltyDistributor.{distributeRoyalty,claim,recoverStranded}` (lines 111-193) | INV-RD-1, INV-RD-4 | 23 paths on `check_distributeRoyalty_preserves_solvency`; 35 paths on `check_claim_preserves_solvency` |
| `AdminBoundedSettersSpec` | Bounded admin setters across StorageSlashing / StakeBond / CompensationDistributor / EmissionController | INV-SS-1+2, INV-SB-1+2+3, INV-CD-1, INV-EC-1+2 | 3 paths each on `check_*_always_in_range` bounded-setter proofs |
| `EscrowPoolSolvencySpec` | `EscrowPool.{deposit,withdraw,settleFromRequester}` (lines 119-192) | INV-EP-1 | 5 paths on `check_settle_preserves_solvency`; 5 paths on `check_withdraw_preserves_solvency` |
| `RoleDisarmAccessControlSpec` | OZ AccessControl `{grantRole,revokeRole}` admin-gating | INV-FT-3, INV-FT-4, INV-FT-5 | 3 paths on `check_grant_role_admin_gated`; 4 paths on `check_revoke_role_admin_gated` |
| `SpeculationRollbackMathSpec` | `prsm/compute/chain_rpc/client.py` rollback math (line 1431) + adaptive-K transition (lines 1463-1466) | (off-chain — no runtime invariant counterpart; audit-prep §7.11+§7.12) | 2 paths on `check_rollback_math_post_fix_bounded`; 5 paths on `check_adaptive_k_stays_in_range` |
| `ChunkStreamingBoundsSpec` | `prsm/compute/chain_rpc/server.py:1960-1994` H1 bounded-iterator + per-chunk request_id binding | (off-chain — audit-prep §7.3 Phase 3.x.7.1) | 65 paths on `check_accepted_bounded_by_expected` with `--loop 32`; 111 total |
| `M2ResponseSizePaddingSpec` | `prsm/compute/chain_rpc/tier_c_sharded_executors.py:208-253` `_pad_or_truncate_utf8` | (off-chain — audit-prep §7.15 Phase 3.x.11.q.x) | 5 paths on `check_output_length_equals_pad_target`; 14 total |
| `M1CadenceDrivenYieldSpec` | `prsm/compute/chain_rpc/tier_c_sharded_executors.py:328-351` `FixedRateShardedExecutor.execute_chain_streaming` | (off-chain — audit-prep §7.13 Phase 3.x.11.q) | 6 PASS / 12 paths; headline `check_inter_emission_gap_bounded` |
| `EncryptedProbsCoSetSpec` | `prsm/compute/chain_rpc/protocol.py:956-998` encrypted-probs validators | (off-chain — audit-prep §7.14 Phase 3.x.11.q.y) | 9 PASS / 36 paths; mutual exclusion + co-set + length cap |

## CI integration via `requires_halmos` marker (sprint 366)

The project-wide `tests/conftest.py:602` autouse fixture
mocks `subprocess.run` globally for external-connection
safety. Halmos legitimately needs `subprocess.run` to
invoke its own binary — so sprint 366 carved a per-test
marker that surgically restores the real subprocess for
marked tests only:

```python
import pytest
from prsm.economy.web3.halmos_runner import HalmosRunner

@pytest.mark.requires_halmos
def test_my_proof():
    runner = HalmosRunner()
    if not runner.is_available():
        pytest.skip("halmos/forge not installed")
    suite = runner.run("MySpec")
    assert suite.status.value == "passed", suite.to_dict()
```

Marker semantics:
- Tests WITHOUT the marker keep the session-wide mock
  (default — preserves existing test infrastructure).
- Tests WITH the marker get real `subprocess.run` for
  their duration; mock is re-applied after the test.
- Marker auto-skips when halmos / forge aren't on PATH
  (runner.is_available() returns False) — CI environments
  without the tools still pass cleanly.

To run halmos-marked tests in CI, install both tools:

```
pip install halmos && curl -L https://foundry.paradigm.xyz | bash && foundryup
```

Then ensure they're on `$PATH` when invoking pytest.

## Honest scope

- All 20 runtime invariants from §14 item 4 mirrored in
  symbolic specs — 16 done, 4 deferred (state-dependent;
  runtime probe IS the right layer for those).
- Symbolic proofs against the actual canonical contracts
  (FTNSTokenSimple, RoyaltyDistributor, etc.) rather than
  structurally-equivalent minimal mocks. Doable but
  requires modeling IERC20 + IProvenanceRegistry as
  symbolic backing contracts; deferred.
- Source-identity CI parity check between spec contracts
  + canonical source (would catch silent drift if
  someone modifies one without the other); deferred.

## References

- Halmos: https://github.com/a16z/halmos
- a16z post: https://a16zcrypto.com/posts/article/symbolic-testing-with-halmos-leveraging-existing-tests-for-formal-verification/
- PRSM Vision §14 item 4
- PRSM audit-prep §7.X (sprint 360 entry pending)
