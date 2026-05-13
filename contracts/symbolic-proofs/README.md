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

## Honest scope

This is **sprint 360 proof-of-concept**. Mature symbolic-
proof coverage requires:

- All 20 runtime invariants from §14 item 4 mirrored in
  symbolic specs (incremental sprints).
- CI gate that runs halmos on every PR touching the spec
  contracts. Currently the live `subprocess.run` is
  mocked in pytest by `tests/conftest.py` so CI doesn't
  invoke halmos directly — proofs are verified manually
  during development.
- Symbolic proofs against the actual canonical contracts
  (FTNSTokenSimple, RoyaltyDistributor, etc.) rather than
  structurally-equivalent minimal mocks. Doable but
  requires modeling IERC20 + IProvenanceRegistry as
  symbolic backing contracts; deferred.

## References

- Halmos: https://github.com/a16z/halmos
- a16z post: https://a16zcrypto.com/posts/article/symbolic-testing-with-halmos-leveraging-existing-tests-for-formal-verification/
- PRSM Vision §14 item 4
- PRSM audit-prep §7.X (sprint 360 entry pending)
