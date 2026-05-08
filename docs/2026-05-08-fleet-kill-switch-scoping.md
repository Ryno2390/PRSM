# Fleet Kill-Switch Design — Scoping

**Document identifier:** FLEET-KILL-SWITCH-SCOPING-1
**Version:** 0.1 Draft
**Status:** Architectural scoping. Specifies the design space for promoting today's per-node env-var kill switches to a coordinated fleet-wide push mechanism. Captures threat model, three candidate architectures, security model, named promotion triggers, and operator opt-in posture. **NOT an execution plan.** **NOT a commitment to ship.** **NOT a replacement for the per-node mechanism.**
**Date:** 2026-05-08
**Drafting authority:** PRSM founder
**Promotes:** the open §6.2 readiness item from `docs/security/EXPLOIT_RESPONSE_PLAYBOOK_ANNEX_2026_05.md`. The annex landed 2026-05-08 with three deferred items in §6.2; the first two (Phase 7-storage + Phase 8 Python clients, daemons) closed same-day across commits e75ccd9a / 3b73a60e / 25da8b69 / 321de20c. The third — coordinated env-var push for fleet-wide kill switches — is structurally larger and is scoped here.

**Related documents:**
- `docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md` v0.1 — parent playbook (Phase 1 Detect / Phase 2 Contain / Phase 3 Communicate / Phase 4 Recover / Phase 5 Post-mortem). Fleet kill-switch is a Phase 2 (Contain) primitive.
- `docs/security/EXPLOIT_RESPONSE_PLAYBOOK_ANNEX_2026_05.md` §3-§4 — per-node env-var disable surfaces. Fleet kill-switch builds above these, never below.
- `docs/2026-05-08-r10-cross-node-arbitration-scoping.md` §5 — architectural principles (per-node mechanism remains load-bearing; no new governance authority; coordination best-effort under partition; operator opt-in remains visible). Same principles apply here.
- `docs/2026-04-21-prsm-gov-1-foundation-governance-charter.md` §4 — Foundation council authority. Fleet kill-switch issuance authority maps to council-multisig per Q6 ratification.
- `prsm/config/networks.py` MAINNET block — canonical contract addresses. A new `FleetKillSwitch.sol` would be added here under Architecture A.

---

## 1. Purpose

The question raised: **how does the Foundation propagate a kill-switch signal across the operator fleet in seconds, when an active incident requires disabling a specific subsystem (e.g., QueryOrchestrator, settlement-split, KeyDistribution Tier C release) on every node, without resorting to manual per-operator coordination on Discord that takes minutes-to-hours?**

Today's surface (per `EXPLOIT_RESPONSE_PLAYBOOK_ANNEX_2026_05.md` §4):

- Each operator node has 7 env-var-based kill switches (`PRSM_QUERY_ORCHESTRATOR_ENABLED`, `PRSM_AGGREGATOR_SHARE_BPS`, `PRSM_ARBITRATION_PROPOSER_ID`, `PRSM_DHT_ENABLED`, `PRSM_KEY_DISTRIBUTION_ADDRESS`, `PRSM_ONCHAIN_PROVENANCE`, JobHistoryStore capacity).
- Operator-side disable: unset the env var, restart the node.
- Time to apply per node: seconds.
- Time to apply across the fleet: linear in operator count, plus operator-attention latency (during business hours: minutes; off-hours: hours).

This is correct and shippable for early bake-in (low operator count, founder-known-Discord coordination). **It is not what a healthy mature network looks like.** Specifically:

- A 50-operator deployment cannot rely on individual operator attention during a critical incident.
- A "operator out for the night" leaves the kill switch un-applied on their node.
- There is no way for the Foundation to verify which operators have actually applied a coordination directive.
- An operator who forgets to unset the env var post-incident has the kill switch stuck on indefinitely.

The fleet kill-switch must address all four gaps **without** introducing:

- Foundation-controlled remote shutdown (operator sovereignty must remain intact — operators opt INTO accepting fleet directives, not out of them).
- A new attack surface (a kill switch that an attacker can spoof becomes a denial-of-service vector against the entire fleet).
- A single point of failure (the kill-switch infrastructure itself must not be the chokepoint).

**Document scope.** This doc answers six framing questions:

1. What incident classes warrant a fleet kill-switch, and what is the latency target? (§2)
2. What design constraints must be preserved? (§3)
3. What are the candidate architectures, and what are their tradeoffs? (§4)
4. What is the recommended architecture, and why? (§5)
5. What is the security model — who can issue, what authority is required, how does operator opt-in work? (§6)
6. Under what named conditions should this be promoted from research-track to execution? (§7)

**This doc is NOT:**
- An implementation plan (execution gates on §7 triggers + governance ratification).
- A commitment to any specific architecture (three are scoped; final choice is execution-phase).
- A replacement for the per-node mechanism (which remains the canonical operator-side disable).
- A precondition for any current PRSM milestone (the per-node mechanism is sufficient for the current 1-5 operator scale).

---

## 2. Threat model — incidents that warrant fleet kill-switch

Each incident class below is mapped to (a) what subsystem must be disabled, (b) the latency target, (c) why per-node manual coordination is insufficient.

### 2.1 Active settlement-split bug (P0)

**Symptom:** anomalous `PaymentEscrow.release_escrow_split` events route funds to wrong addresses (per annex §5.1 detection scenario).

**Subsystem to disable:** force `PRSM_AGGREGATOR_SHARE_BPS=0` or `=10000` to halt the split logic; revert to legacy single-recipient escrow release.

**Latency target:** **< 60 seconds across the fleet.** Each new settlement after detection burns more user funds.

**Why per-node insufficient:** with 50 operators, manual coordination during off-hours can take 30-60 minutes; in that window, hundreds of settlement transactions can leak.

### 2.2 KeyDistribution release-without-payment bypass (P0)

**Symptom:** `KeyReleasedEvent` emitted without corresponding payment-escrow release (per annex §5.4).

**Subsystem to disable:** unset `PRSM_KEY_DISTRIBUTION_ADDRESS` across operator fleet.

**Latency target:** **< 60 seconds.** Tier C trust depends on payment-verification guarantee; a bypass means encrypted content is being released without payment.

**Why per-node insufficient:** an attacker exploiting the bypass can extract many keys per minute; coordination latency directly translates to extracted-content count.

### 2.3 QueryOrchestrator anomaly (P1)

**Symptom:** `/compute/forge` returning malformed `AggregatedResult`, wrong settlement participants, or query results that diverge from expected aggregation.

**Subsystem to disable:** unset `PRSM_QUERY_ORCHESTRATOR_ENABLED` across fleet; falls back to legacy AgentForge dispatch.

**Latency target:** **< 5 minutes.** Less critical than P0 because the failure mode is correctness, not direct fund-loss.

**Why per-node insufficient:** correctness divergence across the fleet creates a Byzantine-fault scenario where some nodes return correct results and others return wrong ones; partial fleet disable is worse than uniform disable.

### 2.4 DHT eclipse attack (P1)

**Symptom:** cross-node fingerprint-dedup queries returning systematically wrong answers (per annex §5.5).

**Subsystem to disable:** unset `PRSM_DHT_ENABLED` across fleet; degrades to per-node-only dedup, regains correctness.

**Latency target:** **< 5 minutes.**

**Why per-node insufficient:** an eclipsed node and a non-eclipsed node will produce different dedup outcomes, fragmenting the network's content provenance.

### 2.5 On-chain ingest panic (P0/P1)

**Symptom:** suspected ProvenanceRegistry / RoyaltyDistributor compromise where every additional on-chain registration adds fuel.

**Subsystem to disable:** unset `PRSM_ONCHAIN_PROVENANCE` across fleet; new uploads stop registering on-chain.

**Latency target:** **< 60 seconds for P0, < 5 minutes for P1.**

**Why per-node insufficient:** every node that hasn't yet flipped the switch continues to add to the compromised state.

### 2.6 Latency target summary

| Incident class | Severity | Latency target across fleet |
|---|---|---|
| Settlement-split bug | P0 | < 60 seconds |
| KeyDistribution release bypass | P0 | < 60 seconds |
| On-chain ingest panic (P0) | P0 | < 60 seconds |
| QueryOrchestrator anomaly | P1 | < 5 minutes |
| DHT eclipse attack | P1 | < 5 minutes |
| On-chain ingest panic (P1) | P1 | < 5 minutes |

The < 60s target for P0 is the binding constraint on architecture choice. < 5 min is achievable by polling-based architectures; < 60s requires either push-based or aggressive polling (≤ 30s interval).

---

## 3. Design constraints

Six constraints any candidate architecture must satisfy.

### 3.1 Operator opt-in remains visible

Every operator must explicitly opt INTO accepting fleet kill-switch directives. The opt-in is via env var (e.g., `PRSM_FLEET_KILL_SWITCH_ENABLED=1`); without it, the node ignores all fleet-side signals and continues to honor only its local env-var configuration. **No silent default change on upgrade.**

This preserves operator sovereignty: the Foundation cannot remotely disable subsystems on a node whose operator has not opted in. The trade is that opt-out operators retain full responsibility for incident response on their own node.

### 3.2 Per-node mechanism remains load-bearing

The fleet kill-switch is a coordination layer ABOVE the per-node env-var mechanism. If the fleet layer fails (signal not propagated, signature verification fails, RPC unreachable), the per-node mechanism stays available. Operators can always manually disable a subsystem regardless of what the fleet layer says.

### 3.3 No new governance authority

The fleet kill-switch does not create a new decision-making body. Issuance authority maps to existing Foundation Safe (2-of-3 hardware multisig per Q6 ratification). Specifically:

- **P0 fleet kill-switch issuance** — 3-of-5 council multi-sig (operational threshold per parent playbook §5; mirrors emergency pause authorization).
- **P1 fleet kill-switch issuance** — Security lead OR council member (mirrors P1 incident declaration authority).

This is the same authority structure that already governs per-contract pause invocation. The fleet kill-switch is a Phase 2 (Contain) primitive, not a governance primitive.

### 3.4 Coordination is best-effort under partition

A fleet kill-switch directive must degrade cleanly under network partition. A node disconnected from the fleet-coordination plane (RPC unreachable, DHT eclipsed, HTTPS endpoint down) MUST continue to operate (uploads green, node functional) — it just doesn't apply the fleet directive until reconnected.

This is identical to R10 §5.3 — partition-resilience is non-negotiable.

### 3.5 Cryptographic verification, not address-based trust

Every fleet directive must be cryptographically signed by Foundation Safe authority. Operator nodes verify the signature before applying the directive. **A spoofed directive must be detectable.**

Without this property, the fleet kill-switch becomes a denial-of-service attack surface — an attacker who can publish unsigned directives can disable the entire fleet at will.

### 3.6 Granular subsystem targeting

A directive targets a specific subsystem (e.g., "disable QueryOrchestrator"), not the whole node. Foundation cannot issue a "shut down all PRSM nodes" directive — the directive surface is narrowed to the same set of env vars that exist as per-node kill switches today.

This bounds the blast radius of either Foundation misjudgment or compromised Foundation-multisig key.

---

## 4. Candidate architectures

Three candidates. Each is internally coherent; each has tradeoffs.

### 4.1 Architecture A — On-chain `FleetKillSwitch.sol`

Deploy a new Solidity contract owned by Foundation Safe:

```solidity
struct KillSwitch {
    bytes32 subsystemId;        // e.g., keccak256("PRSM_QUERY_ORCHESTRATOR_ENABLED")
    bool active;                 // true = subsystem disabled
    uint64 activatedAt;
    bytes32 reasonHash;          // keccak256 of human-readable rationale
}
mapping(bytes32 => KillSwitch) public switches;

function activate(bytes32 subsystemId, bytes32 reasonHash) external onlyOwner;
function deactivate(bytes32 subsystemId) external onlyOwner;
event SwitchActivated(bytes32 indexed subsystemId, bytes32 reasonHash, uint64 at);
event SwitchDeactivated(bytes32 indexed subsystemId, uint64 at);
```

**Operator-side polling loop:**

```python
# Pseudocode running every poll_interval (default 30s):
for subsystem_id in monitored_subsystems:
    state = contract.switches(subsystem_id)
    if state.active and not _local_state[subsystem_id]:
        _disable_subsystem(subsystem_id)
        _local_state[subsystem_id] = True
    elif not state.active and _local_state[subsystem_id]:
        _re_enable_subsystem(subsystem_id)
        _local_state[subsystem_id] = False
```

**Pros:**
- Cryptographic auth for free — Foundation Safe is sole writer; any other caller's tx reverts.
- Auditable — every activation + deactivation leaves a Basescan trail.
- Composes with existing on-chain governance.
- Operator nodes already have RPC connections (used for FTNS / Provenance / etc.), so no new infrastructure.

**Cons:**
- Latency floor = block time (~2s on Base) + poll interval. With 30s poll, P0 < 60s target is met but barely.
- Gas cost per activation/deactivation (modest on Base, but non-zero).
- Chain dependency — if RPC is down for the fleet, kill switch can't propagate.
- Foundation Safe multisig signing cycle (cosigner reachability) gates on the same 3-of-5 reachability that the per-contract pause already requires; the kill switch doesn't add coordination overhead beyond what already exists.

**Effort estimate:** 4-6 weeks. Contract draft + Hardhat tests + audit + deploy ceremony + Python client + node-side polling loop integration. Audit is the long pole.

### 4.2 Architecture B — Signed-payload broadcast over DHT

Foundation publishes signed `KillSwitchEnvelope` to the DHT under a known topic. Operator nodes subscribe.

```python
@dataclass(frozen=True)
class KillSwitchEnvelope:
    subsystem_id: str
    active: bool
    timestamp: int  # unix
    reason_hash: bytes  # 32 bytes
    signature: bytes  # Ed25519 over (subsystem_id || active || timestamp || reason_hash)
```

Foundation Safe holds the kill-switch signing key (separate from the multisig signers — a delegated signing key under Foundation custody, rotation policy TBD). Verification reuses the existing `PublisherKeyAnchor` pattern.

**Operator-side subscription:**

```python
# Pseudocode — runs on the existing DHT event loop:
async def on_kill_switch_envelope(env: KillSwitchEnvelope):
    if not _verify_signature(env, FOUNDATION_KILL_SWITCH_PUBKEY):
        return  # spoofed envelope, ignore
    if env.timestamp < _last_processed_timestamp[env.subsystem_id]:
        return  # replay attempt or out-of-order
    _last_processed_timestamp[env.subsystem_id] = env.timestamp
    if env.active and not _local_state[env.subsystem_id]:
        _disable_subsystem(env.subsystem_id)
    elif not env.active and _local_state[env.subsystem_id]:
        _re_enable_subsystem(env.subsystem_id)
```

**Pros:**
- Latency: sub-second propagation under healthy DHT. P0 < 60s target easily met.
- No gas cost.
- Reuses existing DHT transport infrastructure (`prsm/network/dht/*` — shipped 2026-05-06).
- No new contract audit.

**Cons:**
- DHT eclipse attack surface — an attacker controlling a target node's DHT view can suppress kill-switch envelopes (causing target to keep running a compromised subsystem) or inject stale envelopes (causing target to apply old directives).
- Audit trail off-chain — kill-switch events not visible on Basescan; operators need an off-chain log aggregator.
- Foundation custody of an additional signing key — separate from the multisig signers; introduces a new key-management burden.
- Subscribers need a stable subscription topic and re-subscribe-on-restart logic.

**Effort estimate:** 2-3 weeks. Builds on shipped DHT primitives; main work is signed-envelope schema, signature verification path, replay/order protection, and the subscription wiring in Node.start.

### 4.3 Architecture C — HTTPS-pull from Foundation-operated endpoint

Foundation operates a static endpoint (S3 bucket, Cloudflare Worker, or similar) serving a signed JSON payload. Operator nodes poll periodically.

```json
{
  "version": 1,
  "issued_at": 1735689600,
  "signature": "0x...",
  "switches": {
    "PRSM_QUERY_ORCHESTRATOR_ENABLED": { "active": false, "reason_hash": "0x..." },
    "PRSM_KEY_DISTRIBUTION_ADDRESS":   { "active": true,  "reason_hash": "0x..." },
    ...
  }
}
```

The `signature` is an EIP-191 signature by the Foundation Safe kill-switch key (or a Foundation-Safe-multisig-issued signature in a more complex variant) over the canonical-JSON-serialized `(version, issued_at, switches)` triple.

**Operator-side polling:**

```python
# Pseudocode every poll_interval (default 30s):
resp = httpx.get(FLEET_KILL_SWITCH_URL, timeout=10.0)
payload = resp.json()
if not _verify_signature(payload, FOUNDATION_KILL_SWITCH_PUBKEY):
    return
if payload["issued_at"] < _last_processed_at:
    return  # stale
for subsystem_id, state in payload["switches"].items():
    _apply_state(subsystem_id, state)
_last_processed_at = payload["issued_at"]
```

**Pros:**
- Simplest; lowest latency floor (HTTPS + DNS + CDN edge cache = sub-second).
- No on-chain dependency, no DHT dependency.
- Foundation can update the payload faster than block time.
- Familiar HTTPS auth pattern; no new on-chain audit needed.
- Endpoint can be CDN-backed for high availability.

**Cons:**
- Centralization — the endpoint is a single point of failure (DDoS-able, takedown-able by hosting provider). Doesn't fit the decentralization story cleanly.
- Foundation operates HTTPS infrastructure (TLS cert renewal, hosting bill, monitoring).
- Requires HTTPS connectivity from every operator node (which the nodes already have for npm package downloads, but worth noting).
- Audit trail off-chain — same as Architecture B.

**Effort estimate:** 1-2 weeks. Endpoint deploy + signing-key custody + Python polling client + Node integration. Smallest of the three.

### 4.4 Comparison summary

| Property | A (on-chain) | B (DHT) | C (HTTPS-pull) |
|---|---|---|---|
| P0 latency target (<60s) | Met (30s poll + 2s block) | Easily met (sub-second) | Easily met (sub-second) |
| Gas cost per directive | Modest on Base | $0 | $0 |
| Audit trail | On-chain (Basescan) | Off-chain logs | Off-chain logs |
| Decentralization | Strong (chain ground truth) | Medium (DHT eclipse risk) | Weak (single endpoint) |
| New contract audit | Yes (long pole) | No | No |
| New infrastructure | None (uses existing RPC) | None (uses existing DHT) | Foundation-operated HTTPS |
| Fits §3.1-§3.6 constraints | All six | All six | §3.6 (decentralization) is weak |
| Effort | 4-6 weeks | 2-3 weeks | 1-2 weeks |
| Recommended when | Production-grade, audited | Bake-in / interim | First ship / urgent |

---

## 5. Recommended architecture: phased C → A migration

The recommendation is **NOT** a single architecture, but a **phased rollout** that exploits each architecture's strengths in sequence.

### 5.1 Phase 1: Architecture C (HTTPS-pull) — first ship

Deploy as the bootstrap fleet kill-switch. Properties:

- Ships fastest (1-2 weeks); closes the §6.2 readiness item without waiting on a contract audit.
- Foundation operates a Cloudflare Worker (or equivalent) at a stable URL (e.g., `https://kill-switch.prsm-network.com/v1/switches.json`).
- Payload signed by a delegated Foundation kill-switch key (Ed25519, separate from multisig hardware-wallet keys).
- Operator nodes opt in via `PRSM_FLEET_KILL_SWITCH_ENABLED=1` + `PRSM_FLEET_KILL_SWITCH_URL=<url>` (default URL embedded in code; override allowed for testnet / development).
- Polling cadence: 30s default, operator-tunable via `PRSM_FLEET_KILL_SWITCH_POLL_SECONDS` in `[10, 300]` band.
- Applies only the 7 currently-existing per-node kill switches; adding new ones requires Phase-2 design review.

Phase 1 closes the §6.2 readiness item and provides immediate operational value while the Phase 2 contract audit proceeds in parallel.

### 5.2 Phase 2: Architecture A (on-chain) — production target

Once `FleetKillSwitch.sol` clears external audit, deploy as the canonical kill-switch surface. Operator nodes are upgraded to read on-chain state preferentially, falling back to HTTPS-pull only if RPC is unreachable.

Properties:

- Cryptographic auth via Foundation Safe sole-owner enforcement (no separate signing key custody burden).
- On-chain audit trail for every directive — every Basescan reader can verify what the Foundation has activated and when.
- Removes Foundation HTTPS infrastructure dependency once HTTPS-pull is fully retired.

Phase 2 migration is gated on:
- External Solidity audit of `FleetKillSwitch.sol`.
- Sepolia bake-in for ≥ 30 days with synthetic kill-switch activations to validate the polling loop.
- Council ratification of the on-chain authority (3-of-5 multisig for activation; same for deactivation).

### 5.3 Why not Architecture B (DHT)

Architecture B is technically attractive (sub-second propagation, no gas cost) but loses to A and C on three dimensions:

- **Eclipse-attack surface** — a kill-switch envelope is exactly the kind of high-value coordination signal that an attacker would target via eclipse. DHT lookups (Kademlia α-of-k) are eclipsable with realistic budgets.
- **Audit-trail visibility** — Basescan readers see Architecture A directives natively; Architecture C directives are a single signed JSON payload anyone can fetch; Architecture B directives are scattered across DHT logs and require operator-side aggregation.
- **Key-management overhead** — Architecture B requires Foundation custody of a delegated signing key with no clear benefit over either A's multisig-native auth or C's simpler delegated-key pattern.

Architecture B is documented for completeness but not recommended for either phase.

---

## 6. Security model

### 6.1 Issuance authority

| Directive type | Authority | Authorization threshold |
|---|---|---|
| P0 activation (immediate disable) | Foundation council multisig | 3-of-5 (operational) |
| P1 activation | Security lead OR council member | 1 |
| Any deactivation | Foundation council multisig | 3-of-5 (operational) |
| Adding a new subsystem to the targeted-list | Foundation council multisig | 4-of-5 (governance, with auditor sign-off) |

Asymmetry note: P1 single-actor declaration mirrors the parent playbook's Phase 1 (Detect) authority — anyone with credible suspicion can trigger; downgrading later is acceptable. P0 requires 3-of-5 because the blast radius is greater.

### 6.2 Cryptographic verification

**Architecture A:** Foundation Safe is sole writer; chain enforces. No additional verification needed at the operator node beyond reading state.

**Architecture C:** payload signed by a delegated Ed25519 key under Foundation custody. Public key embedded in PRSM source code (operators verify against the embedded pubkey, not against an external authority). Key rotation requires a PRSM software release, which is a feature: rotation is auditable via PRSM's release process.

The delegated key is NOT the multisig hardware-wallet key. It is a Foundation-custodied Ed25519 key with a documented rotation policy:

- Rotated annually OR on suspected compromise.
- Stored in Foundation Safe-controlled HSM (Cloudflare HSM, AWS CloudHSM, or equivalent).
- Activation requires 3-of-5 council multisig to authorize the HSM signing operation.

This is a new operational primitive but not a new auth model — it's the same shape as how DNS root-zone signing keys work in DNSSEC.

### 6.3 Replay protection

Every directive carries `issued_at` (unix timestamp). Operator nodes track `last_processed_at` per subsystem. Directives with `issued_at <= last_processed_at` are rejected as stale or replay attempts.

Architecture A has implicit replay protection via on-chain state (the contract's `activatedAt` is monotonically increasing per subsystem).

### 6.4 Granularity bound

Operator nodes only honor directives targeting subsystems in their **monitored list** — initially the 7 per-node kill switches catalogued in `EXPLOIT_RESPONSE_PLAYBOOK_ANNEX_2026_05.md` §4. Adding a subsystem to this list requires a PRSM software release (auditable via release commit + tag).

This means even a compromised Foundation key cannot expand the kill-switch authority beyond what operators ratified at install time.

### 6.5 Audit logging

Every applied directive is logged at WARNING level by the operator node:

```
WARNING fleet_kill_switch: applied directive subsystem=PRSM_QUERY_ORCHESTRATOR_ENABLED active=False issued_at=2026-05-08T14:30:00Z reason_hash=0xabc...
```

Operators are encouraged (but not required) to forward these logs to a centralized aggregator.

### 6.6 Reversibility

A fleet kill-switch directive is reversible by issuing a deactivation directive. Operator-node behavior on deactivation:

- Re-enable the subsystem (revert env var to original state).
- Log the deactivation at INFO level.
- Resume normal operation; no restart required for most subsystems.

Subsystems that cannot be re-enabled without a node restart (TBD per subsystem) require an operator-initiated restart after the deactivation; the directive cannot force a restart.

---

## 7. Promotion triggers

The fleet kill-switch stays in research-track until **at least one** of the following is observed.

### 7.1 T1 — Active P0 incident requires fleet coordination

A real P0 incident occurs that requires fleet-wide subsystem disable, and manual coordination on Discord takes > 5 minutes. This is the gateway evidence that per-node coordination has crossed the cost/benefit boundary.

### 7.2 T2 — Operator fleet exceeds 10 distinct operator addresses

Per parent playbook §5, council reachability check is monthly. At 10+ operators, manual coordination during business-hours-spanning incidents is no longer feasible; fleet automation becomes load-bearing.

### 7.3 T3 — TVL crosses $50,000

Per PRSM-POL-2's audit-revisit trigger schedule, TVL > $50K is an automatic full-stack security re-evaluation. Fleet kill-switch should be on that re-evaluation's docket.

### 7.4 T4 — VC term sheet specifying fleet operability

A Series A or seed-extension term sheet conditioning investment on fleet-management automation. Direct market signal that fleet coordination is institutional-investor-visible.

### 7.5 T5 — External auditor flag

External security review flags lack of fleet coordination as a P1+ operational risk. Auditor's specific finding determines which architecture phase is appropriate.

### 7.6 T6 — Fleet kill-switch mentioned by ≥ 2 operators in 30 days

Operator-driven demand signal: if multiple operators independently request fleet kill-switch infrastructure, operator UX is the binding constraint.

---

## 8. Integration phases (conditional on promotion)

### 8.1 Phase F1 — Architecture C deployment (1-2 weeks)

- Cloudflare Worker (or equivalent) deployed at `kill-switch.prsm-network.com/v1/switches.json`.
- Foundation kill-switch Ed25519 key generated; pubkey hardcoded in `prsm/security/fleet_kill_switch.py`.
- Operator-side polling client + Node.start integration.
- Three operator env vars: `PRSM_FLEET_KILL_SWITCH_ENABLED=1`, `PRSM_FLEET_KILL_SWITCH_URL=<override>`, `PRSM_FLEET_KILL_SWITCH_POLL_SECONDS=30`.
- 30-50 unit tests covering signature verification, replay protection, subsystem allowlist enforcement, partition-resilience.

### 8.2 Phase F2 — `FleetKillSwitch.sol` design + audit (4-6 weeks)

- Solidity contract draft + Hardhat tests.
- External audit (target: 1 firm; ~$15K-$30K). Audit scope similar to today's RoyaltyDistributor v2.
- Sepolia deploy + 30-day bake-in.

### 8.3 Phase F3 — On-chain migration (2-3 weeks)

- Mainnet deploy after audit + bake-in.
- Operator client gains on-chain reader; HTTPS-pull becomes fallback when RPC unreachable.
- Foundation HTTPS endpoint deprecated 90 days after mainnet on-chain activation.

### 8.4 Phase F4 — Audit-prep refresh (1 week)

- New §7.21 in cumulative audit-prep.
- Threat-model addendum for fleet-kill-switch-specific adversaries.
- Operator runbook update.

---

## 9. NOT in scope (explicit non-goals)

- **Foundation-controlled remote shutdown.** Fleet kill-switch only disables specific subsystems opt-in operators have ratified; it cannot take down a node, kill its process, or revoke its identity.
- **Gradual rollout / canary deployment.** A directive applies to all opted-in operators simultaneously. Phased rollout (e.g., "disable on 10% first, then 50%") is a future enhancement, not part of this sprint's scope.
- **Operator-side challenge mechanism.** Operators cannot vote against a directive in real-time. If an operator disagrees with a Foundation directive, they can opt out by unsetting `PRSM_FLEET_KILL_SWITCH_ENABLED` (which excludes their node from all future directives, not just the disputed one).
- **Cross-jurisdictional operator gating.** Directives apply uniformly to all opted-in operators globally; no regional gating.
- **Time-locked directives.** A directive applies as soon as it's signed and propagated. Time-locked activation (e.g., "this directive applies in 24 hours") is a future enhancement.
- **Partial-subsystem directives.** A directive either fully disables or fully enables a subsystem. Partial states (e.g., "disable Tier C key release for content-CIDs in this list") are out of scope; the existing kill switches are coarse-grained by design.
- **Operator economic compensation for false-positive directives.** If the Foundation incorrectly issues a kill-switch and an opted-in operator loses revenue during the disable, the Foundation may compensate via post-mortem governance proposal but does NOT pre-commit to compensation. Operators opt in with this risk visible.

---

## 10. Open questions

These are deliberately left unresolved at scoping time. Each will be addressed by the corresponding execution phase.

- **Q1.** What is the canonical key-rotation cadence for the Architecture C delegated signing key? Annual is suggested, but rotation requires PRSM software release (or a key-list-versioning scheme). TBD at Phase F1.
- **Q2.** How does the operator client behave during a network partition that lasts longer than the cache-staleness budget? Default proposal: continue using the last-known directive set until reconnected, with WARNING-level logs every poll interval. Final answer at Phase F1.
- **Q3.** Should Phase F2 architecture-A migration be opt-in or default? Default proposal: default opt-in for new installs, opt-in for upgraded installs (avoid surprise behavior change on upgrade). Final at Phase F3.
- **Q4.** What is the maximum payload size for the Phase F1 HTTPS endpoint? Default proposal: 4 KB hard cap to defend against payload-bloat denial-of-service. Final at Phase F1.
- **Q5.** Should the Phase F2 contract emit the human-readable reason text as well as the hash? Trade: reason text adds calldata cost but improves on-chain audit-trail richness; reason hash + off-chain reason document is the alternative. Final at Phase F2 design review.

---

## 11. Conclusion

The fleet kill-switch closes the open §6.2 readiness item from the 2026-05 exploit-response annex by promoting today's per-node manual coordination to a Foundation-issued, cryptographically-verifiable, operator-opt-in directive plane. The recommended path is phased — Architecture C (HTTPS-pull) ships first to provide immediate operational value while the Architecture A (on-chain) contract audit proceeds in parallel.

The named promotion triggers in §7 are designed to be **objective signals**, not judgment calls. Until a trigger fires, the per-node manual coordination remains canonical, and the fleet kill-switch sits in design-only state at this scoping doc.

Until then, the per-node mechanism (annex §4) remains the canonical operator-side disable, and §11.x of `EXPLOIT_RESPONSE_PLAYBOOK_ANNEX_2026_05.md` §6.2 remains an OPEN readiness item with this scoping doc as the pre-built escalation route.

---

**End of document.**
