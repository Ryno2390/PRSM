# Auditor Shortlist + RFP Template (PRSM Bundled Engagement)

**Date:** 2026-04-23
**Owner:** engineering lead (RFP send-off delegated to Foundation officers once provisioned)
**Status:** Shortlist research complete; RFP template ready to send
**Depends on:** Foundation 2-of-3 multi-sig provisioning + treasury funding for retainer

---

## 0. What this is

The PRSM Phase 3.1 + 7 + 7.1 + 7-storage + 8 mainnet audit is a single bundled
engagement covering ~2500 LoC Solidity across 10 contracts plus the Python
off-chain challenger-submitter stack. This doc is what you use the day the
multi-sig is ready:

1. Pick ≤3 firms from §2 that best fit PRSM's scope.
2. Copy the email template from §4.
3. Paste in the URLs + merge-ready tag per firm; hit Send.

Nothing in this doc is speculative about vendor capability — every claim has a
citation, and nothing is assumed about pricing or lead times because no firm
publishes either. We ask both in the RFP.

---

## 1. Scope summary (for the RFP reader)

| Dimension | Value |
|-----------|-------|
| Solidity LoC | ~2500 across 10 contracts |
| Solidity version | 0.8.22 |
| Contract patterns | OZ Ownable, OZ ReentrancyGuard, OZ upgradeable (UUPS proxy for FTNSTokenSimple), custom errors throughout |
| Target chain | Base mainnet (L2, chainId 8453) |
| Novel mechanisms | Challenge/slash economics with three reason codes (DOUBLE_SPEND, INVALID_SIGNATURE, CONSENSUS_MISMATCH), 70/30 challenger/Foundation slash split with self-slash 100%-to-Foundation, emission halving via right-shift, StakeBond with MIN_SLASH_GAS floor |
| Python-adjacent | ConsensusChallengeSubmitter service + SQLite-backed queue persistence (supports on-chain challenge path but is itself off-chain) |
| Tests in tree | 142 Solidity (Hardhat) + 283 Python unit + 2 Python E2E = 427 passing |
| Merge-ready tag | `phase7.1x-audit-prep-20260422-2` |
| Audit-coordinator doc | `docs/2026-04-21-audit-bundle-coordinator.md` |
| Pre-audit hardening landed | Six review findings resolved pre-audit (§8.6, §8.7, §8.8 from Phase 7 and Phase 7.1 per-phase bundles) |

Auditor-facing entry point is the bundle coordinator; per-phase bundles
linked from §2 of that doc.

---

## 2. Shortlist (six firms)

Data verified 2026-04-23 from public sources. Pricing and lead-time rows are
intentionally blank because no firm publishes them — these are the columns
you fill in *from* the RFP responses.

### 2.1 Comparison matrix

| Firm | Base L2 exp. | OZ-upgradeable exp. | Novel-mechanism exp. | Python/Rust adjacent | Engagement model |
|------|-------------|---------------------|----------------------|----------------------|------------------|
| Trail of Bits | Moderate (via Offchain Labs / Arbitrum, Scroll) | Yes | **Strong** (sequencer liveness, custom-fee bridges, owns Slither + Echidna) | **Strong** (Rust via Scroll zkEVM, Aleo, wasmCloud; Python via PyPI review) | Team-based, structured SDLC, weekly syncs, Slack channel, PM, fix-review appendix |
| OpenZeppelin | Yes (listed in nav) | **Strongest** (they write the library) | Moderate (Uniswap 9×, 1inch 19×, Synthetix 4×) | Moderate (Linea, Polkadot, Stellar, Starknet) | Team-based, long-running retention (>95%) |
| Spearbit / Cantina | **Strongest** (4+ Coinbase Base Bridge audits in 2025: Aug, Sept, Oct, Dec + basenames + Base Flywheel) | Yes (via Morpho Vault v2, Uniswap bounty) | **Strong** (Sky Avalanche SkyLink Bridge, Morpho vault v2, Aztec) | Rust (OpenVM, Aztec Barretenberg, Fluent); off-chain submitter work less visible | **Three SKUs**: Spearbit Guild (4-5 top researchers), Cantina Reviews (solo researcher team), Cantina Competitions (contest model) |
| Zellic | Not in public portfolio | Yes | **Strong** (Scroll zkEVM, LayerZero EigenLayer DVN, Lido Fixed Income, Astria bridge) | **Strong Rust** (Solana Anza BPF, Audius Solana, Astria Geth, Move ecosystem) | Team-based, daily client comms; founders ex-world-#1 CTF team |
| Consensys Diligence | **Weak** (Linea-centric, no public Base audit) | Yes | Moderate (Aligned Layer, Lido V3, 1inch Cross Chain Swap v2) | Rust via zk tooling; Python not advertised | Boutique team + AI agents (Super Chonky / tintinweb); pre-audit, full audit, continuous scans, SNAP, incident response, fuzzing |
| ChainSecurity | **Weak** (Ethereum L1-heavy; WBTC Solana Bridge is only cross-L2 recent work) | Yes | **Strong** (Aave V4 Feb 2026, Pendle Boros, Sky Diamond, WBTC Bridge) | Weak; Solidity/EVM focused | Team-based; ETH Zurich spinout; academic rigor |

### 2.2 Recent public audits per firm (signal, not full portfolio)

- **Trail of Bits (2025):** Offchain Labs (Arbitrum) sequencer-liveness, Scroll Euclid phases 1-2, Lagrange LA Token, Franklin Templeton Benji, Near MPC, Automata DCAP Attestation, ZetaChain Solana Gateway. Publications: `github.com/trailofbits/publications/tree/master/reviews`.
- **OpenZeppelin:** >$110B TVL secured (self-reported), >1M LoC reviewed, 700+ C/H findings. Continuous Uniswap + 1inch + Synthetix engagements. Recent featured: OpenAI EVMBench.
- **Spearbit / Cantina (2025-2026 Q1):** Coinbase AggregateVerifier, Coinbase NitroEnclaveVerifier, Morpho wETH Loan Policy + Spend Router + Multiproof + Vault v2, Sky Avalanche SkyLink Bridge, Cow Protocol, Optimism PBC U18, Base Bridge Referral Fees, Zora Coinbase Creator Coin. Bounty programs live: Uniswap $15.5M, Reserve $10M, Euler $7.5M, Polymarket $5M, Coinbase $5M.
- **Zellic (self-reported 2025):** 338 reviews, 247 critical + 308 high-impact findings, 45% of codebases had serious issues. Portfolio: Scroll zkEVM, LayerZero EigenLayer DVN, Lido Fixed Income, Axiom, Anza (Solana), Astria bridge/geth/sequencer.
- **Consensys Diligence (2025-2026 Q1):** Linea Rollup Update (Feb 2026), Linea Poseidon2 (Jan 2026), Linea Yield Manager (Dec 2025), MetaMask Delegator Safe Module, Lido V3, 1inch Cross Chain Swap v2, Aligned Layer, Intuition TRUST token.
- **ChainSecurity (2026 Q1):** Sky NFAT Facility + Diamond PAU + Capped Oracle Feed, Spark Savings Intents, **Aave V4 (Feb 2026)**, Pendle Boros Markets, Gateway Smart Contracts, WBTC Solana Bridge.

### 2.3 Contact channels

| Firm | URL | Intake |
|------|-----|--------|
| Trail of Bits | `trailofbits.com/services/software-assurance/` | Web form — "Schedule a call" / "Bootstrap your project" |
| OpenZeppelin | `openzeppelin.com/security-audits` | Web form — "Request a Security Audit" |
| Spearbit / Cantina | `cantina.xyz/welcome` | Dashboard signup; Spearbit Guild intake via Cantina; Competitions/Bounties self-serve |
| Zellic | `zellic.io` | "Request a quote" CTA |
| Consensys Diligence | `diligence.consensys.io` | "Request Audit" form |
| ChainSecurity | `chainsecurity.com` | "Request Smart Contract Audit" / "Audit Contract Now" form |

---

## 3. Decision criteria tuned to PRSM

When RFP responses come back, score each firm on these six axes (higher is
better, 1-5 scale):

| Axis | Why it matters for PRSM |
|------|------------------------|
| **Base L2 experience** | PRSM deploys to Base. Auditor fluency with OP Stack gas dynamics, sequencer failure modes, and L1→L2 message semantics matters. |
| **Challenge/slash novelty match** | PRSM's 3-reason-code challenge dispatch + 70/30 split is not a stock DeFi pattern. Prior work on sequencer liveness, custom fee bridges, or optimistic-rollup challenger games transfers well. |
| **OZ-upgradeable fluency** | FTNSTokenSimple is UUPS; remaining contracts use OZ Ownable + ReentrancyGuard. Auditor should be able to catch proxy-init footguns on sight. |
| **Off-chain submitter review** | `ConsensusChallengeSubmitter` and SQLite queue sit adjacent to the contracts; a firm willing to read Python helps catch coupling bugs. |
| **Turnaround + communication cadence** | Foundation launch timeline is tight; a firm that responds within a week and closes within 4 is better than one with a 6-month queue. |
| **Remediation cycle included** | Must confirm the quote includes one round of remediation + re-review, not two separate engagements. |

**Recommended top-3 from the matrix**, ordered:

1. **Spearbit / Cantina (Spearbit Guild)** — strongest Base experience, offers
   marketplace-based engagement that can parallelize faster than big-firm queue.
2. **Trail of Bits** — strongest novel-mechanism + Python-adjacent combination.
   Owns Slither + Echidna. Structured engagement with visible fix-review cycle.
3. **OpenZeppelin** — strongest OZ-upgradeable fluency, highest retention signal.

Send RFP to all three in parallel. Compare responses on duration, cost,
communication cadence, remediation terms, and researcher experience.

**Secondary pool (fourth RFP only if top-3 all decline / over-quoted):**
Zellic (strong technical bench, weak Base exp.), ChainSecurity (strong DeFi
rigor via Aave V4, weak Base/off-chain coverage).

**Drop:** Consensys Diligence — no visible Base mainnet work; Linea-centric
portfolio is a poor match for a Base-deploying protocol.

---

## 4. RFP email template (copy/paste)

Replace `{{PLACEHOLDER}}` values before sending. Everything in `{{ }}` must be
resolved — one missing substitution and the firm wastes a reply cycle.

### 4.1 Subject line

```
RFP: PRSM mainnet audit — ~2500 LoC Solidity + bundled 3-phase engagement (Base L2)
```

### 4.2 Body

```
Hi {{FIRM_CONTACT_NAME}},

I'm Ryne Schultz, engineering lead on PRSM (Protocol for Recursive Scientific
Modeling). We're preparing a bundled mainnet audit engagement and {{FIRM_NAME}}
is on our shortlist given your {{CITE_SPECIFIC_RECENT_AUDIT}} work.

=== Scope ===

- ~2500 LoC Solidity 0.8.22 across 10 contracts
- OpenZeppelin upgradeable (UUPS) + Ownable + ReentrancyGuard patterns
- Target: Base mainnet (chainId 8453)
- Novel mechanisms: challenge/slash economics with three reason codes
  (DOUBLE_SPEND, INVALID_SIGNATURE, CONSENSUS_MISMATCH), 70/30 challenger/
  Foundation slash split with self-slash 100%-to-Foundation, emission
  halving via right-shift, StakeBond with MIN_SLASH_GAS floor
- Adjacent off-chain: Python ConsensusChallengeSubmitter + SQLite queue —
  optional in scope

=== What exists going in ===

- 427 passing tests (142 Solidity via Hardhat + 283 Python unit + 2 Python E2E)
- Merge-ready freeze tag: phase7.1x-audit-prep-20260422-2
- Pre-audit hardening complete: six review findings resolved pre-audit
  (documented in the bundle coordinator, §5)
- Local hardhat deploy rehearsal with 15 invariant checks green end-to-end;
  Base Sepolia dress rehearsal planned pre-mainnet

=== Starting point for your review ===

Single auditor entry point:
{{REPO_URL}}/docs/2026-04-21-audit-bundle-coordinator.md

Per-phase bundles:
- {{REPO_URL}}/docs/2026-04-22-phase7.1x-audit-prep.md  (most recent; covers full hardening arc)
- {{REPO_URL}}/docs/2026-04-21-phase7-audit-prep.md    (Phase 7 StakeBond + slash-hook)
- {{REPO_URL}}/docs/2026-04-21-phase7.1-audit-prep.md  (Phase 7.1 consensus extension)

Operator runbook with on-chain invariants:
{{REPO_URL}}/docs/OPERATOR_GUIDE.md

=== What we're asking ===

1. Confirm availability + earliest engagement start date.
2. Quote:
   - Fixed fee or time-and-materials (we prefer fixed)
   - Expected duration (initial review + remediation cycle + re-review)
   - Team composition (names + seniority if possible)
   - Whether the quote includes one remediation round or is per-round billed
3. Confirm Base L2 / OP Stack fluency.
4. Confirm willingness to review the Python-adjacent challenger-submitter
   stack (optional add-on; flag separately if scoped as additional).
5. Sample of a prior report involving comparable novel challenge/slash
   economics (if public).

Our timeline: we'd like to start the engagement within {{EARLIEST_START_WEEKS}}
weeks and wrap the first review cycle within {{WRAP_WEEKS}} weeks. Foundation
multi-sig is provisioned and treasury is funded for retainer; contract
execution is Foundation-side.

Happy to jump on a 30-minute intro call to walk through the bundle
coordinator and the three cross-phase seams (§3 of that doc).

Best,
Ryne Schultz
schultzryne@gmail.com
{{GITHUB_PROFILE_URL}}
{{OPTIONAL_FOUNDATION_SIGNATURE_BLOCK}}
```

### 4.3 Placeholder checklist

Before sending, confirm every placeholder is resolved:

- [ ] `{{FIRM_CONTACT_NAME}}` — look up on LinkedIn / firm site
- [ ] `{{FIRM_NAME}}` — firm display name
- [ ] `{{CITE_SPECIFIC_RECENT_AUDIT}}` — reference something from §2.2 above. E.g. for Cantina: "Coinbase Base Bridge" or "Morpho Vault v2"; for Trail of Bits: "Offchain Labs sequencer-liveness"; for OpenZeppelin: "continuous Uniswap engagements since 2020".
- [ ] `{{REPO_URL}}` — public PRSM repo URL (once publicly accessible to auditor)
- [ ] `{{EARLIEST_START_WEEKS}}` — your availability (typically 2-4 weeks)
- [ ] `{{WRAP_WEEKS}}` — desired wrap (typically 6-10 weeks post-start)
- [ ] `{{GITHUB_PROFILE_URL}}` — GitHub profile
- [ ] `{{OPTIONAL_FOUNDATION_SIGNATURE_BLOCK}}` — Foundation name + address
  once entity formation is complete; remove if pre-formation

### 4.4 When NOT to send this template verbatim

- **Spearbit / Cantina** — use their Cantina Dashboard intake instead of cold
  email. Attach this email body as a scope summary in the dashboard form.
- **If pre-Foundation-formation** — remove the final Foundation signature
  block and mention treasury funding is "in progress" rather than "funded."
  Delay RFP if the firm asks for counterparty entity information you can't
  yet provide.

---

## 5. Response-tracking table

Once RFP responses come in, track them in this table and decide. Leave rows
blank when the firm hasn't responded; target 3 responses before deciding.

| Firm | Contact name | RFP sent | First response | Quote (fixed / T&M) | Duration | Start ETA | Remediation incl.? | Score (1-5) | Notes |
|------|--------------|----------|----------------|---------------------|----------|-----------|---------------------|-------------|-------|
| Spearbit/Cantina | | | | | | | | | |
| Trail of Bits | | | | | | | | | |
| OpenZeppelin | | | | | | | | | |
| *(Zellic — backup)* | | | | | | | | | |
| *(ChainSecurity — backup)* | | | | | | | | | |

Decision rule: pick the highest-scoring firm that can start within our
`{{EARLIEST_START_WEEKS}}` window. If two score within 1 point, prefer the
one with stronger Base L2 experience.

---

## 6. What this depends on

Send-off preconditions — every one of these must be true before RFPs go out,
or the RFP asks questions the sender can't answer:

- [ ] Foundation entity is formed OR the RFP explicitly flags that Foundation
  formation is in progress (adjust email template §4.2 accordingly).
- [ ] Multi-sig provisioned and treasury funded with the retainer amount
  likely required (industry range for ~2500 LoC: $50K-$250K).
- [ ] Repo is accessible to the auditor — either public, or access-gated
  invite flow is ready.
- [ ] The six review findings flagged as "RESOLVED pre-audit" in
  `docs/2026-04-22-phase7.1x-audit-prep.md` are actually landed in the tree
  referenced by the freeze tag.
- [ ] At least one deploy rehearsal against Base Sepolia has run green
  (tracked in `docs/2026-04-23-testnet-rehearsal-plan.md` §5 Step 1).

---

## 7. Related documentation

- `docs/2026-04-21-audit-bundle-coordinator.md` — the auditor's entry point.
- `docs/2026-04-22-phase7.1x-audit-prep.md` — most recent per-phase bundle.
- `docs/2026-04-21-phase7-audit-prep.md` — Phase 7 StakeBond bundle.
- `docs/2026-04-21-phase7.1-audit-prep.md` — Phase 7.1 consensus bundle.
- `docs/2026-04-23-testnet-rehearsal-plan.md` — hardware-day deploy rehearsal.
- `docs/OPERATOR_GUIDE.md` — operational invariants (§On-chain Keypairs, §Redundant-Execution Dispatch Tier B).

---

## 8. Research provenance

All vendor facts in §2 were verified from public sources on 2026-04-23. No
vendor was contacted during research — the shortlist exists to prepare for
outreach, not to substitute for it. Sources (all URLs as of 2026-04-23):

- Trail of Bits — `trailofbits.com/services/software-assurance/`,
  `github.com/trailofbits/publications/tree/master/reviews`
- OpenZeppelin — `openzeppelin.com/security-audits`
- Cantina (Spearbit) — `cantina.xyz/portfolio`, `cantina.xyz/welcome`,
  `cantina.xyz/bounties`, `cantina.xyz/competitions`
- Zellic — `zellic.io`, `github.com/Zellic/publications`
- Consensys Diligence — `diligence.consensys.io`, `diligence.consensys.io/audits/`
- ChainSecurity — `chainsecurity.com`, `chainsecurity.com/audits`

Pricing and lead-time data are intentionally absent because no firm
publishes them. Those columns in §5 fill in from the RFP responses, not
from research.
