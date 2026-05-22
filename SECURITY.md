# Security Policy

**Last updated:** 2026-05-22
**Related documents:** [`audits/AUDIT_PLAN.md`](audits/AUDIT_PLAN.md) (master audit plan, §12 disclosure policy) · [`docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md`](docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md) (incident response) · [`docs/2026-05-22-parallax-inference-audit-readiness.md`](docs/2026-05-22-parallax-inference-audit-readiness.md) (wire-protocol hardening audit, F30-F65)

## What we protect

PRSM (Protocol for Research, Storage, and Modeling) defends nine attack
surfaces, listed in `audits/AUDIT_PLAN.md` §3. The on-chain attack surface
includes live treasury infrastructure on Base mainnet:

| Contract | Address | Role |
|----------|---------|------|
| Foundation Safe (2-of-3 Gnosis Safe) | `0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791` | Treasury custodian |
| ProvenanceRegistry | `0xdF470BFa9eF310B196801D5105468515d0069915` | Content registration |
| RoyaltyDistributor | `0x3E8201B2cdC09bB1095Fc63c6DF1673fA9A4D6c2` | Payment-split executor |
| FTNSTokenSimple | `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` | Native protocol token |

Off-chain we protect: bootstrap and relay nodes, the inference pipeline
(`prsm/inference/`, `prsm/chain_rpc/`, `prsm/streaming/`), the marketplace
listing surface, and the storage/proofs layer.

The audit-bundle stack (EscrowPool, BatchSettlementRegistry, StakeBond,
SignatureVerifier) is **pre-mainnet** pending external sign-off; reports
against pre-deploy code are still in scope.

## Reporting a vulnerability

### Channels

- **GitHub Security Advisory** (preferred): [Report a vulnerability](https://github.com/prsm-network/PRSM/security/advisories/new)
- **Email** (for time-sensitive issues): `security@prsm-network.com`
  - **PGP key:** *Pending publication.* Until the foundation PGP key is
    published, encrypt sensitive details with the GitHub Security Advisory
    flow (which is end-to-end encrypted between reporter and maintainers).
- **DO NOT** open a public GitHub issue for security vulnerabilities.

### What to include

- Affected component (contract address, file path, or service).
- Attack scenario — what does the attacker control, what's the sequence
  of calls, what does the attacker gain?
- Proof of concept — runnable test, transaction trace, or call log.
- Impact assessment — funds at risk, scope of compromise, severity estimate.
- Suggested remediation if known.

## Disclosure timeline (post-mainnet)

We follow a **90-day coordinated disclosure window** as defined in
`audits/AUDIT_PLAN.md` §12.

| Day | Phase | Action |
|-----|-------|--------|
| 0 | Receipt | Reporter submits via GHSA or email |
| 0–2 | Triage | Acknowledgment + severity assignment |
| 2–14 | Reproduction | Foundation engineering reproduces and confirms |
| 14–60 | Remediation | Fix development + audit of the fix |
| 60–75 | Coordinated deploy | Fix deployed; integrators privately notified ≥72 hours pre-disclosure |
| 75–90 | Public disclosure | GHSA + blog post + bounty payout |
| 90 | Hard cap | Public disclosure even if remediation is incomplete |

### Critical-severity exception

For findings with **active exploitation** OR **>50% TVL at imminent risk**:

- Hour 0–2: Pause-transaction templates from
  [`docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md`](docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md)
  executed by the Foundation Safe.
- Hour 2–24: Private notification to known integrators + Forta alert
  subscribers.
- Hour 24–48: Public disclosure on Foundation channels + GHSA.
- Day 7+: Post-mortem published.

The 90-day window does **not** apply to actively-exploited bugs.

## Pre-mainnet (current state)

Until `audits/AUDIT_PLAN.md` Gate B clears (audit-bundle stack + Phase 8
contracts + L4 contest + L3 specialist crypto audit), the disclosure model is:

- Findings on **live mainnet contracts** (Phase 1.3 Task 8): full 90-day
  policy above + critical-severity exception applies.
- Findings on **audit-bundle pre-deploy code** or **off-chain Python**:
  reported confidentially, fixed pre-deploy, disclosed at Gate B clearance + 30
  days alongside the consolidated audit report.
- **Self-discovered findings** (internal review L0 / static-analysis L1):
  triaged internally; no external disclosure required pre-mainnet.

## Bounty program

Pre-mainnet bounty program design is at `audits/AUDIT_PLAN.md` §5 L9
(pending; cannot launch until L4 contest first-pass complete).

When live, payout scale follows Immunefi conventions:

| Severity | Range | Examples |
|----------|-------|----------|
| Critical | $50K–$1M | Funds drainable, mint forgery, total treasury compromise |
| High | $10K–$50K | Conditional fund loss, signature forgery on challenge path |
| Medium | $2K–$10K | Best-practice violation with conditional exploit, griefing |
| Low | $500–$2K | Minor invariant breach, no direct exploit path |
| Informational | $0–$500 | Code quality, docs, naming |

Payout in USDC from Foundation Safe within 14 days of fix deployment.
Researchers credited (with consent) in `audits/hall-of-fame.md` at L9 launch.

### Duplicate handling

First valid report wins, determined by submission timestamp on
`security@prsm-network.com` inbox or GHSA timestamp. Subsequent identical
reports receive acknowledgment but no bounty.

## Scope

### In scope

- Smart contracts under `contracts/contracts/` (live mainnet + audit-bundle
  pre-deploy).
- Deploy ceremony scripts under `contracts/scripts/` (deploy-provenance.js,
  transfer-ownership.js, verify-audit-bundle-deployment.js, sweep-deployer.py).
- Off-chain Python services in `prsm/inference/`, `prsm/chain_rpc/`,
  `prsm/streaming/`, `prsm/node/`, `prsm/storage/`.
- The bootstrap node infrastructure
  (`wss://bootstrap1.prsm-network.com:8765`).
- The on-chain monitoring + alert routing
  (`ops/monitoring/forta-bots/`).
- Foundation Safe operational security — phishing surface targeting signers,
  ceremony manipulation paths, etc. (responsible-disclosure paths only;
  social-engineering tests against signers are **NOT** authorized).

### Out of scope

- Issues already documented in `audits/findings/` or in the threat models
  (`docs/2026-04-22-r3-threat-model.md`,
  `docs/2026-04-30-phase3.x.11-threat-model-addendum.md`).
- Deprecated branches (`v1.5.x` and earlier).
- Spam, low-impact informational issues already known.
- Issues requiring physical access to a hardware wallet, an active social
  engineering attack against a Foundation signer, or compromise of a third-
  party service we use (these are non-PRSM attack surfaces).
- Best-practice / code-quality suggestions without an exploit (these are
  welcome as PRs or discussions, but not bounty-eligible).

## Defense-in-depth posture

The full defense matrix is in `audits/AUDIT_PLAN.md`. Eleven audit layers
cover the nine attack surfaces. Key continuously-active layers:

- **L0** — internal review gates (per-phase merge-ready tags).
- **L1** — static + symbolic tooling (Slither, Aderyn, Mythril, Halmos,
  Echidna; CI integration in progress).
- **L10a** — Forta detection bots monitoring live mainnet contracts for
  anomalous transfers, role grants, pause events, and distribution failures.
- **L10b** — exploit-response playbook with pre-staged pause transactions.

Pending external layers (per `audits/AUDIT_PLAN.md` §6 Gate B):
- **L3** — specialist Ed25519 crypto audit
- **L4** — external smart-contract audit (Code4rena + solo firm)
- **L5** — off-chain ML supply-chain audit
- **L6f** — network infrastructure pen-test
- **L7** — economic / game-theory audit
- **L8** — legal / regulatory review
- **L9** — public bug bounty (post-L4)

## Recent hardening — wire-protocol audit arc (2026-05-22)

Between sprints 711 and 734, an iterative audit of the
parallax-inference wire protocols (remote token-stream + unary
chain-executor RPC) closed **35 F-class production-blockers
(F30 through F65)** across 10 audit dimensions. Detailed
write-up + per-fix evidence in
[`docs/2026-05-22-parallax-inference-audit-readiness.md`](docs/2026-05-22-parallax-inference-audit-readiness.md).

Summary of the security-grade fixes:

- **F50 / F51 / F52 / F55 / F56 / F57 / F58 / F59 / F61 / F62**
  — wire-protocol DoS hardening (back-pressure races, identical-
  request collisions, payload-size limits, per-peer concurrency
  caps, hang-defense execution timeouts) on both streaming and
  unary paths with parity.
- **F53 / F60** — stream + unary response hijack protection via
  sender-binding. Pre-fix, a peered third party that learned a
  victim's stream_id or request_id could forge frames/responses
  into the victim's pending queue/future. Now `pending[]` stores
  `(state, expected_sender)` tuples + handlers verify
  `msg.sender_id` before routing.
- **F54** — server-side resource leak: server kept iterating
  inference generator after requester disconnect, burning GPU
  on tokens nobody would receive. Now reads send_to_peer return
  value + closes inner generator on disconnect (releases KV
  cache promptly).
- **F63 / F64** — *foundation fix.* Transport verified
  cryptographic signatures only at handshake; subsequent
  `msg.sender_id` was wire-trusted but not crypto-bound. A peer
  with valid handshake could spoof sender_id to defeat per-peer
  caps (F56/F59) AND forge responses (F53/F60) — undermining
  several preceding fixes. Sprint 730 bound sender_id at the
  chain-executor dispatch wrappers; sprint 731 generalized at
  the transport layer (`WebSocketTransport._dispatch`) so EVERY
  `MSG_DIRECT` handler in the codebase — including
  `ledger_sync` FTNS transfers, `compute_provider`,
  `storage_provider`, `content_provider`, `agent_registry` —
  now sees the handshake-authenticated peer identity.
- **F65** — `/admin/*` endpoints were unauthenticated
  (`tags=["admin"]` was a swagger grouping, not access control).
  Sprint 722's observability endpoint exposed live stream
  metadata to any network client. Middleware now restricts
  `/admin/*` to loopback by default; `PRSM_ADMIN_REMOTE_ALLOWED=1`
  opt-in for remote (must be behind reverse-proxy auth or VPN).

Operator-facing consequences are documented in
[`docs/operations/parallax-inference-deploy.md`](docs/operations/parallax-inference-deploy.md).
**Existing operators upgrading past sprint 734 should expect
`/admin/*` to default-deny non-loopback access** — set
`PRSM_ADMIN_REMOTE_ALLOWED=1` and add a real auth layer in
front if remote admin access is required.

## Supported versions

| Version | Status | Security support |
|---------|--------|------------------|
| `main` (current) | Active development | Full |
| `v1.6.x` (latest tagged) | Active maintenance | Full |
| `v1.5.x` and earlier | Superseded | None |

The on-chain treasury layer (deployed 2026-05-04, Phase 1.3 Task 8) is
non-Ownable and non-upgradeable; "version" semantics for those contracts
are governed by the manifest at
`contracts/deployments/provenance-base-1777917793612.json` and the milestone
tag `phase1.3-task8-complete-20260504`.

## Authorization clarification

This policy authorizes good-faith security research within scope. It does
**not** authorize:

- Attacks against the live infrastructure that disrupt service (DoS testing
  against bootstrap nodes, etc.) — please coordinate first.
- Social engineering against Foundation employees, signers, or contributors.
- Physical attacks against any party.
- Testing on user systems without explicit authorization.

Out-of-scope research conducted in good faith and reported responsibly will
not be pursued legally; coordinate with us before testing edge cases.

## Acknowledgments

PRSM thanks the security research community. Researchers who report valid
findings are credited (with consent) in `audits/hall-of-fame.md` after fix
deployment. Significant findings receive bounty payout per the scale above
once the L9 program launches.

---

## Operationalization status (operator-facing notes)

The following are pending engineering/operations actions, tracked in the
PRSM task list:

- [ ] `security@prsm-network.com` inbox setup (DNS/MX + email provider).
- [ ] Foundation PGP keypair generation + publication.
- [ ] `audits/hall-of-fame.md` initial commit (empty roster).
- [ ] L9 Immunefi listing draft (post-L4 first-pass clearance).
- [ ] L10a alert routing wired to PagerDuty + Foundation council Slack.

Until these complete, the GHSA channel is the primary reporting path and
all operational claims above should be read as "policy-target" rather than
"already-operational."
