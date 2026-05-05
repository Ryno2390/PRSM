# L1 — Static + Symbolic Tooling

**Layer:** L1 of the 11-layer defense-in-depth audit plan
(`audits/AUDIT_PLAN.md` §5).
**Status:** Wired 2026-05-05.
**Workflow:** `.github/workflows/solidity-static-analysis.yml`
**Scope:** Solidity smart contracts under `contracts/contracts/`.

---

## 1. What L1 covers

L1 is the lowest-cost, highest-frequency layer of the audit stack: continuous
automated analysis of every PR + scheduled deeper sweeps. It catches the
common smell-classes that other layers (L2 AI multi-team review, L3
cryptographic specialist, L4 public contest) shouldn't have to spend their
budget on. Tools wired:

| Tool       | Class                       | Tier       | Trigger                     |
|------------|-----------------------------|------------|-----------------------------|
| **Slither**  | SAST (pattern-based)        | Fast       | Per-PR + push to main       |
| **Aderyn**   | SAST (Rust-based, complementary) | Fast  | Per-PR + push to main       |
| **Mythril**  | Symbolic execution          | Slow       | Weekly + workflow_dispatch  |
| **Halmos**   | Symbolic verification       | Slow       | Weekly + workflow_dispatch  |
| **Echidna**  | Property-based fuzzing      | Slow       | Weekly + workflow_dispatch  |

Fast tier MUST pass before merge to main. Slow tier soft-fails with
artifacts uploaded for review.

---

## 2. Slither

**What it is:** Trail of Bits' Python-based SAST. Pattern-matches against
~90 detectors (reentrancy, uninitialized storage, dangerous delegatecall,
unchecked low-level calls, etc.).

**Config:** `contracts/slither.config.json`
**Threshold:** Fails CI on `high` severity. Surfaces medium/low/info as
artifacts.

**Filter rationale (`detectors_to_exclude`):**

- `naming-convention`: tests use snake_case fixtures intentionally
- `pragma`: bundle is intentionally pinned at 0.8.22
- `solc-version`: same — explicit pin
- `assembly`: Ed25519Lib + Sha512 use assembly for curve math; reviewed
  separately under L3 (cryptographic specialist)
- `low-level-calls`: EscrowPool/StakeBond use bool-returning IERC20
  transfers; checks are explicit
- `timestamp`: Phase 7 uses `block.timestamp` for unbond delays;
  documented assumption

**Local run:**
```bash
cd contracts
pip install slither-analyzer
slither . --config-file slither.config.json --fail-on high
```

**Suppressions inside source:** prefer `// slither-disable-next-line
<detector>` over global config additions. Each in-source suppression
should have a brief comment justifying it.

---

## 3. Aderyn

**What it is:** Cyfrin's Rust-based AST-based SAST. Faster than Slither
for large codebases; runs ~50 detectors, complementary coverage.

**Config:** none (uses defaults)
**Threshold:** Soft-fail (we collect findings as artifacts); enforce
high severity once we've validated the per-PR signal.

**Local run:**
```bash
# Install (one-time):
curl -L https://raw.githubusercontent.com/Cyfrin/aderyn/dev/cyfrinup/install | bash
export PATH="$HOME/.cyfrin/bin:$PATH"

cd contracts
aderyn .
# Output: report.md in CWD
```

---

## 4. Mythril

**What it is:** ConsenSys' symbolic-execution engine. Explores reachable
program states to find vulnerabilities that pattern-matching SAST misses
(integer over/underflow under specific paths, predictable randomness,
external-call-after-state-change reentrancy variants).

**Why slow tier:** symbolic execution is exponentially slower than
pattern-matching. A single contract can take minutes; the audit-bundle
takes ~30 minutes total at our chosen `--execution-timeout 300
--max-depth 50`.

**Targets** (from workflow):
```
EscrowPool, StakeBond, BatchSettlementRegistry, RoyaltyDistributor,
FTNSTokenSimple, EmissionController, CompensationDistributor,
StorageSlashing, KeyDistribution
```

Mocks, libraries (Ed25519Lib + Sha512), and test harnesses are skipped —
those are L3-scope.

**Local run:**
```bash
pip install mythril
cd contracts
npx hardhat compile
myth analyze contracts/EscrowPool.sol --solv 0.8.22 \
  --execution-timeout 300 --max-depth 50 -o markdown
```

---

## 5. Halmos

**What it is:** a16z's symbolic test runner for Foundry-style property
tests. Treats `test_*` / `check_*` functions as theorem statements and
proves (or refutes) them via SMT solving over symbolic inputs.

**Current status:** **No-op.** PRSM's contract test suite is hardhat-
based (`test/*.test.js`), not foundry. Halmos requires `.t.sol` files
with a specific test interface to do anything useful.

**To enable:**
1. Create `contracts/test/halmos/` with `.t.sol` files defining
   property-shaped functions (e.g., `function check_invariant_X(...)`).
2. The workflow's discovery probe will detect them and run halmos
   automatically.
3. Recommended properties to write first (high-value invariants):
   - **EscrowPool.totalEscrowedBalance == sum(balances)** — already
     enforced in code, but a halmos property would prove it across all
     reachable states.
   - **StakeBond.unbond_eligible_at >= challengeWindowSeconds**
     post-requestUnbond — the L2-audit HIGH-2 invariant.
   - **BatchSettlementRegistry.b.invalidatedValueFTNS <=
     b.totalValueFTNS** — math safety.
   - **RoyaltyDistributor: sum(claimable) == address(this).balance**
     in FTNS terms — the pull-payment escrow invariant from
     MEDIUM D-04.

**Local run (after properties exist):**
```bash
pip install halmos
cd contracts
halmos
```

---

## 6. Echidna

**What it is:** Trail of Bits' coverage-guided property-based fuzzer for
EVM. Runs a bounded number of randomized transaction sequences against
invariant-shaped functions and reports any sequence that breaks an
invariant.

**Current status:** **No-op.** Echidna requires
`Echidna<ContractName>.sol` invariant contracts that wrap the target
contracts and expose `echidna_*` functions returning `bool`.

**To enable:**
1. Create `contracts/contracts/EchidnaEscrowPool.sol` (etc.) wrapping
   EscrowPool with `echidna_balance_invariant() returns (bool)` style
   functions.
2. The workflow's discovery probe will detect them.
3. Same priority list as Halmos §5 above.

**Local run (after invariants exist):**
```bash
# Install Echidna binary from
# https://github.com/crytic/echidna/releases (or via Homebrew/Nix)
cd contracts
echidna contracts/EchidnaEscrowPool.sol --contract EchidnaEscrowPool \
  --test-limit 50000
```

---

## 7. Suppression policy

When L1 surfaces a false positive, prefer suppression in this order:

1. **Fix the code** — if the finding is a real smell even if not a
   bug, fixing it is best.
2. **In-source comment** — `// slither-disable-next-line <detector>` (or
   the equivalent in other tools) directly above the line. Always
   include a brief comment explaining why.
3. **Config-level exclusion** — add to `detectors_to_exclude` in
   `slither.config.json` only if the detector is uniformly noisy
   across the codebase. Document the exclusion in §2 above.

Never silence findings via blanket `// solhint-disable` or by deleting
the workflow steps.

---

## 8. CI integration

The workflow is `.github/workflows/solidity-static-analysis.yml`.
It triggers on:

- **Push to main** — fast tier runs.
- **PR to main** — fast tier runs; SARIF uploads to GitHub
  code-scanning so findings appear inline in the PR diff.
- **Weekly cron (Mon 06:00 UTC)** — slow tier runs; artifacts
  uploaded with 90-day retention.
- **workflow_dispatch** — manual trigger; opt-in to slow tier via
  `run_slow_tier=true` input.

Path filter: only triggers when `contracts/contracts/`,
`contracts/scripts/`, `contracts/hardhat.config.js`, or the workflow
file itself changes. Other PRs (Python/docs only) skip the static
analysis to save CI minutes.

---

## 9. Findings disposition

L1 findings are triaged like any other audit finding:

1. **CRITICAL / HIGH** — fix in-PR before merge. Workflow blocks
   merge on `high` severity in the fast tier.
2. **MEDIUM / LOW** — file as `audits/findings/L1-static/<id>.md`,
   triage in the next sprint review.
3. **False positive** — add in-source suppression with comment per §7.
4. **Out of scope** — annotate in the report and pass to the relevant
   layer (L3 for crypto, L4 for protocol logic, etc.).

---

## 10. Roadmap

| Item                          | Status      | Notes |
|-------------------------------|-------------|-------|
| Slither per-PR                | ✅ live     | This commit |
| Aderyn per-PR                 | ✅ live     | This commit |
| Mythril weekly                | ✅ live     | This commit |
| Halmos discovery + scaffold   | ✅ live     | No-op until `.t.sol` properties exist |
| Echidna discovery + scaffold  | ✅ live     | No-op until `Echidna*.sol` invariants exist |
| Halmos property suite         | ☐ planned   | 4 invariants in §5 priority list |
| Echidna invariant suite       | ☐ planned   | Same 4 as Halmos |
| SARIF dashboarding            | ☐ planned   | GitHub code-scanning is wired; consider exporting to a separate dashboard for the foundation council |

Properties / invariants authoring is a follow-up sprint after
external L3/L4 engagements close. The current L1 wiring catches the
~80% of pattern-matchable smells; symbolic verification is sharpest
when paired with auditor-defined properties, not as a standalone
sweep.
