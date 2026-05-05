# PRSM Audit — Layer L2 (AI Multi-Team Adversarial Review)

> **This is a sub-document.** The umbrella plan covering all eleven audit
> layers (L0–L11) and all nine attack surfaces (3.1–3.9) is
> [`AUDIT_PLAN.md`](./AUDIT_PLAN.md). Read that first.

**Pinned target:** `cumulative-audit-prep-20260504-h` (commit `e7e4cdb3`)

**Status:** Phase 1.3 Task 8 (ProvenanceRegistry + RoyaltyDistributor) live on Base
mainnet 2026-05-04. Audit-bundle stack (EscrowPool + BatchSettlementRegistry +
SignatureVerifier + StakeBond) frozen pending sign-off (#31, #40).

## What this layer is

L2 in the master plan: four parallel AI agent teams attack the on-chain Solidity
surface from orthogonal lenses. This catches what static tooling can't (creative
adversarial scenarios) but is not a substitute for L3 (cryptographic spec-level
review of Ed25519Lib) or L4 (paid human contest). See `AUDIT_PLAN.md` §4 for the
full layer matrix and §6 for the gating sequence.

## Three internal phases of L2

L2 itself is structured in three sequential phases. Each phase catches what the
others miss; skipping is risk explicitly accepted, not a substitute.

### Phase A — AI multi-team adversarial review (this directory)

Four parallel AI agent teams each attack the codebase from an orthogonal lens.
Findings written to `audits/findings/team-{a,b,c,d}-findings.md`. A fifth
synthesis pass deduplicates and ranks → `audits/findings/consolidated.md`.

| Team | Lens | Primary scope |
|------|------|---------------|
| **A** | Economic value extraction | RoyaltyDistributor, FTNSTokenSimple, EscrowPool value flows, payment splits, MEV |
| **B** | Access control + ownership | Ownable transitions, AccessControl roles, MINTER/PAUSER/BURNER matrix, ownership-window attacks |
| **C** | Signature + cryptographic surface | EIP-712 domain separators, Ed25519 verifier, signature replay across chains/contracts/nonces, HandoffToken |
| **D** | State-machine composition | Challenge windows, unbonding delays, escrow lifecycles, reentrancy across contract boundaries, pause-state coverage |

Wall-clock: ~1 day. Cost envelope: ~$30–80 in API.

### Phase B — Static + symbolic tooling

Industry-standard automated analysis over the same pinned commit. Findings
appended to `audits/findings/tooling/`.

- **Slither** — pattern-based static analysis (free, fast)
- **Aderyn** — Rust-native static analyzer (free, complementary patterns)
- **Mythril** — symbolic execution (free, deeper)
- **Halmos** — bounded symbolic verification of invariants (free)
- **Echidna** — property-based fuzzing of stated invariants (free)

Wall-clock: ~4 hours. Cost: $0.

### Phase C — Targeted human review

Use Phase A + B findings as the brief for a paid human pass. Options:

- **Code4rena contest** ($20–40K, ~2 weeks) — broad crowd of vetted auditors
- **Sherlock contest** ($30–50K, ~3 weeks) — fixed-fee with judge protocol
- **Solo firm** (Trail of Bits / OpenZeppelin / ConsenSys Diligence) — $80K+, 4–6 weeks

Skipping this for a live treasury layer is a conscious risk decision, not the
default.

## Scope

### In scope (Solidity, ~3,400 LoC)

Live mainnet (Phase 1.3 Task 8):
- `contracts/contracts/FTNSTokenSimple.sol` (128 LoC)
- `contracts/contracts/ProvenanceRegistry.sol` (104 LoC)
- `contracts/contracts/RoyaltyDistributor.sol` (125 LoC)

Audit-bundle stack (Phase 7 + 7.1, gating #31/#40):
- `contracts/contracts/EscrowPool.sol` (196 LoC)
- `contracts/contracts/BatchSettlementRegistry.sol` (788 LoC)
- `contracts/contracts/StakeBond.sol` (412 LoC)
- `contracts/contracts/Ed25519Verifier.sol` (59 LoC)
- `contracts/contracts/lib/Ed25519Lib.sol` (887 LoC)
- `contracts/contracts/lib/Sha512.sol` (328 LoC)

### Out of scope for v1

- Off-chain Python (`prsm/inference/`, `prsm/chain_rpc/`, `prsm/streaming/`) —
  separate ML-supply-chain audit, different specialist skill set.
- Phase 8 contracts (`EmissionController.sol`, `CompensationDistributor.sol`) —
  not yet deployed; included in Phase C but not Phase A.
- Bridge contracts (`FTNSBridge.sol`, `BridgeSecurity.sol`) — out of audit
  bundle, future phase.

## Required reading for every team

Each team prompt instructs the agent to read these before attacking:

- `docs/2026-04-22-r3-threat-model.md` — top-level threat model (R1–R7)
- `docs/2026-04-30-phase3.x.11-threat-model-addendum.md` — streaming-inference addendum
- `docs/2026-04-27-cumulative-audit-prep.md` — the auditor brief (current state)
- `docs/2026-04-21-audit-bundle-coordinator.md` — bundle composition + invariants
- `docs/Tokenomics.md` (if present) — economic invariants + payment-split spec

## Severity rubric

Findings classified per Code4rena conventions:

| Severity | Meaning |
|----------|---------|
| **High** | Direct loss of user funds, treasury, or breaks core invariant |
| **Medium** | Conditional fund loss, requires specific market conditions |
| **Low** | Best-practice violation, no direct exploit path |
| **Gas** | Optimization without behavioral change |
| **Informational** | Code quality, docs, naming |

Every finding **must** include: severity, contract+line, attack scenario,
proof-of-concept (test or trace), recommended fix.

## Coordination

- Each team is dispatched as an independent agent with a single prompt file.
- Teams write to their own findings file; no shared state.
- Synthesis pass runs only after all 4 teams complete.
- Consolidated report is the artifact that gates moving to Phase B.
