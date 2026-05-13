# Fleet Kill-Switch — Operator Runbook

**Status:** Operator-facing runbook for the **per-node mechanism** that exists today + the optional **fleet-coordination layer** described in `docs/2026-05-08-fleet-kill-switch-scoping.md`. Sprint 379 closes the §7.21 audit-prep honest-scope item.

**Audience:** Anyone running a PRSM node. **Read this BEFORE an incident** — incident time is not training time.

**Scope:**
- §1 names the 7 per-node kill switches that exist today and how to use them
- §2 covers the fleet-coordination layer's operator-opt-in model (when implemented per the §7 promotion triggers)
- §3 walks through "what do I do when a directive fires"
- §4 covers verification + dispute + appeals
- §5 covers post-incident recovery
- §6 documents the audit-trail expectations

**Out of scope:**
- Foundation-side issuance procedures (lives in `docs/security/EXPLOIT_RESPONSE_PLAYBOOK_ANNEX_2026_05.md`)
- Implementation internals (lives in the scoping doc)
- Governance authority schedule for issuance (locked by `PRSM-CR-2026-05-08` Resolution 7 + `FLEET-KILL-SWITCH-SCOPING-1` §6.1; see §2.2 below)

---

## 1. Per-node kill switches (available today)

Seven env-var-gated subsystems are operator-disablable WITHOUT any Foundation coordination. These are the load-bearing mechanism; the fleet-coordination layer (§2) sits on top of them, never around them.

### 1.1 The seven kill switches

| Env var | Subsystem | Disables |
|---|---|---|
| `PRSM_QUERY_ORCHESTRATOR_ENABLED=0` | QueryOrchestrator | New `/compute/forge` dispatch; in-flight jobs unaffected |
| `PRSM_AGGREGATOR_SHARE_BPS=0` | Settlement split | Aggregator coordination-fee zero; recipients still paid |
| `PRSM_ARBITRATION_PROPOSER_ID=` (empty) | PRSM-PROV-1 Item 6 | Dispute-routing disabled; auto-attribute still works |
| `PRSM_DHT_ENABLED=0` | Phase 7-storage DHT | Manifest + Embedding DHT listeners + sync clients down |
| `PRSM_KEY_DISTRIBUTION_ADDRESS=` (empty) | Tier C key release | KeyDistributionClient builder returns None; no on-chain release calls |
| `PRSM_ONCHAIN_PROVENANCE=0` | ProvenanceRegistry writes | Content uploads skip on-chain registration; gossip + IPFS unchanged |
| `PRSM_JOB_HISTORY_DIR=` (empty) | JobHistoryStore filesystem persistence | Memory-only mode (LRU 1024-bound) |

### 1.2 How to apply

**Via env var (canonical):**

```bash
# Stop the node
sudo systemctl stop prsm-node

# Edit the systemd env file (or .env, or whatever your
# deployment uses)
sudo vi /etc/prsm/node.env
# Add or modify the line:
#   PRSM_QUERY_ORCHESTRATOR_ENABLED=0

# Restart
sudo systemctl start prsm-node

# Verify
curl -s http://localhost:8000/health/detailed | jq \
  '.subsystems.query_orchestrator'
# Expect: {"available": false, "status": "not_wired"}
```

**Verification path:**
- `/health/detailed` reports the subsystem as `not_wired` (= opt-out, NOT degradation)
- `/info` reports the env-var-derived configuration
- `prsm node status` CLI surfaces the same fields

### 1.3 Reversing the change (post-incident)

```bash
sudo vi /etc/prsm/node.env
# Remove or set:
#   PRSM_QUERY_ORCHESTRATOR_ENABLED=1
sudo systemctl restart prsm-node
curl -s http://localhost:8000/health/detailed | jq \
  '.subsystems.query_orchestrator'
# Expect: {"available": true, "status": "ok"}
```

**Common mistake:** forgetting to remove the kill-switch env var after the incident is resolved, leaving the subsystem disabled indefinitely. Set a calendar reminder when you apply a kill switch; re-verify on the next scheduled-restart cadence.

---

## 2. Fleet-coordination layer (when implemented)

The fleet-coordination layer is **design-only as of 2026-05-13**. Implementation is gated on the §7 promotion triggers (T1 active P0 incident / T2 fleet > 10 operators / T3 TVL > $50K / T4 VC term sheet / T5 external auditor flag / T6 ≥ 2 operator requests). None have fired as of this writing.

When it ships, the contract below is what operators commit to.

### 2.1 Operator opt-in

**You must explicitly opt IN. There is no default-on behavior. Upgrade never silently enables the fleet layer.**

```bash
# Opt in
PRSM_FLEET_KILL_SWITCH_ENABLED=1

# Optional: pin the trusted-issuer key set (otherwise
# defaults to the Foundation council canonical 5-of-N
# multisig)
PRSM_FLEET_KILL_SWITCH_TRUSTED_KEYS=key1.pub,key2.pub,...

# Optional: bootstrap source for the directive feed
# (HTTPS-pull during Phase 1; DHT-broadcast under
# consideration for Phase 2 per the scoping doc)
PRSM_FLEET_KILL_SWITCH_FEED_URL=https://kill-switch.prsm-network.com/feed
```

**Opt-out posture (default):** the node ignores all fleet-side signals. Per-node kill switches (§1) remain fully functional and are your sole responsibility during incidents. Foundation cannot remotely disable subsystems on your node.

### 2.2 Authority schedule per directive type

Locked by PRSM-CR-2026-05-08 Resolution 7 + FLEET-KILL-SWITCH-SCOPING-1 §6.1:

| Directive | Authority | Issuance window |
|---|---|---|
| P0 emergency disable (single subsystem) | 3-of-5 council multisig | Hours-from-trigger |
| P1 advisory disable | Single-actor (council-member, post-audit-trail) | Same-day |
| Deactivation (re-enable subsystem) | 3-of-5 council multisig | Same posture as P0 — re-enable requires same approval threshold to prevent unilateral re-arm |
| List-expansion (add new disablable subsystem) | 4-of-5 council multisig + external auditor co-sign | PRSM software release boundary; never a runtime directive |

**What this means for you as operator:**
- You can verify the directive's signature set BEFORE applying it (see §2.3)
- A directive that doesn't carry the right authority set is a P0 fraud event — surface it (see §4)
- The Foundation cannot expand the list of disablable subsystems at runtime; that requires a software release

### 2.3 What happens when a directive fires

Mechanics (per the scoping doc's Architecture C, expected Phase 1 implementation):

1. Foundation council assembles signatures (3-of-5 for P0 disable)
2. Signed directive published to the canonical feed URL
3. Your node's fleet-kill-switch client polls the feed (configurable interval, default 30s)
4. Signature verification against `PRSM_FLEET_KILL_SWITCH_TRUSTED_KEYS`
5. Directive applied — the targeted subsystem switches off as if you'd set the per-node env var
6. Status surfaced via `/health/detailed` (new `fleet_directive_applied` field) + `/info` (sticky log of applied directives this process-lifetime)
7. Audit log entry written via `/audit/recent` (sprint 70 audit ring; existing surface)

**Crucial:** the fleet-applied disable is **runtime-only**. It does NOT modify your env file. Process restart re-evaluates the directive against the feed. This is by design — operators who want a directive sticky-across-restart apply it to their env file too.

### 2.4 Latency targets (per scoping doc §2.6)

| Scenario | Target apply-time across the fleet |
|---|---|
| P0 settlement bug | < 5 minutes from issuance |
| P0 key-release bypass | < 5 minutes from issuance |
| P1 advisory | < 30 minutes from issuance |
| Deactivation | < 30 minutes (intentionally not faster — give operators time to verify before re-enable) |

---

## 3. What to do when a directive fires

### 3.1 First 5 minutes

1. **Don't panic.** The directive was applied by your node automatically (if opted in). Your subsystem is now disabled at runtime.
2. **Verify the directive is genuine.**

```bash
curl -s http://localhost:8000/health/detailed | jq \
  '.fleet_directive_applied'
# Expect a directive object with:
#   directive_id, severity, target_subsystem,
#   signers_keyids (3+ for P0), issued_at,
#   signature_verified: true
```

3. **Confirm via independent channel** — check `https://kill-switch.prsm-network.com/feed` directly + cross-reference the directive_id with the Foundation's incident-response Discord/Telegram announcement.

4. **If signature_verified is FALSE or signers don't match the canonical council set** — see §4. This is a P0 fraud event.

### 3.2 What's affected on your node

The targeted subsystem is now in the same state as if you'd set the corresponding per-node env var. Cross-reference with §1.1:

- Compute jobs in-flight: continue to completion
- New compute jobs dispatched after directive: rejected per the subsystem's own gating
- Heartbeat: continues (unaffected)
- Wallet: unaffected
- P2P discovery: unaffected unless DHT is the directive target

### 3.3 What to communicate

If you operate publicly-visible compute services on this node:

1. **Acknowledge the directive** in your own status page within 15 minutes
2. **Reference the Foundation's incident announcement** (not yours) for impact details
3. **Set a follow-up commitment** — "we expect re-enable within X hours per Foundation's published timeline"

If you don't operate public compute services, no operator-side comms required.

---

## 4. Verification, dispute, and appeals

### 4.1 Verifying a directive

Every directive carries:
- `directive_id` — UUID4 per-directive
- `severity` — `p0` / `p1` / `deactivation` / `list-expansion`
- `target_subsystem` — one of the 7 canonical names (or expanded list per §2.2)
- `signers_keyids` — list of council keyids that signed
- `signature` — Ed25519 signature over `(directive_id, severity, target_subsystem, issued_at, expires_at)`

**The list of disablable subsystems is HARDCODED in your PRSM software release.** A directive targeting a subsystem not in your local list is rejected at the client side BEFORE signature verification. This is the structural defense against list-expansion-via-directive (§2.2 row 4).

### 4.2 Disputing a directive

If you believe a directive is invalid (wrong target / wrong signers / outside Foundation's authority):

1. **Per-node override** — set the corresponding per-node kill switch's env var to `1` (re-enable). The per-node mechanism overrides the fleet directive on your node. Document why in your operator log.
2. **Surface to the Foundation council** via the council's published intake (the same channel used for `prsm_disclosure` per sprint 300)
3. **DO NOT silently ignore.** A directive that fired without proper authority is a P0 fraud event — other operators need to know.

### 4.3 Appeals process

Locked by FLEET-KILL-SWITCH-SCOPING-1 §6.1 + cross-references to PRSM-POL-2 (Resource-Constrained Audit Strategy):

| Stage | Authority | Timeline |
|---|---|---|
| Initial review | Foundation council majority | 7 days from operator filing |
| External audit | Audit firm engagement (PRSM-POL-2 trigger) | If majority deadlocked OR if ≥ 2 operators file the same dispute |
| Final decision | Council 4-of-5 supermajority post-audit | Audit + 14 days |

Disputes that succeed result in: directive retracted, council issues `deactivation` directive (3-of-5), full audit-trail published in `docs/governance/`.

---

## 5. Post-incident recovery

### 5.1 When the Foundation issues deactivation

Mirror of §3 but inverted:

1. Your node receives the `deactivation` directive (3-of-5 council multisig)
2. Subsystem re-enabled at runtime
3. `/health/detailed` returns to pre-incident state for that subsystem
4. Cross-reference with the Foundation's incident-resolution announcement

### 5.2 If your node was opt-OUT during the incident

Per-node kill switches (§1) are your sole responsibility. After the Foundation announces "all clear":

1. Verify the incident is genuinely resolved via the post-mortem document
2. Reverse your per-node env-var change (§1.3)
3. Confirm subsystem health via `/health/detailed`

### 5.3 Audit-trail expectations

- Your node's `/audit/recent` shows the directive-application event with `directive_id`, `applied_at`, `signers_keyids`
- The Foundation publishes the directive + post-mortem in `docs/governance/INCIDENT-YYYY-MM-DD-N.md`
- The CR ratifying the incident response lands in `docs/governance/PRSM-CR-YYYY-MM-DD-N.md`

You should be able to reconstruct what happened, when, who signed, and what your node did — fully from local audit logs + public Foundation docs. If you can't, that's a process bug; file via §4.2.

---

## 6. Quick-reference card

Print this if you operate a node. Pin it near your incident-response Discord channel.

```
PRSM Fleet Kill-Switch — Quick Reference
─────────────────────────────────────────

Verify directive applied:
  curl :8000/health/detailed | jq .fleet_directive_applied

Per-node override (if directive disputed):
  Set <ENV_VAR>=1 in /etc/prsm/node.env + restart

Seven local kill switches:
  PRSM_QUERY_ORCHESTRATOR_ENABLED         compute/forge
  PRSM_AGGREGATOR_SHARE_BPS=0             settlement split
  PRSM_ARBITRATION_PROPOSER_ID=           dispute routing
  PRSM_DHT_ENABLED=0                      Phase 7-storage DHT
  PRSM_KEY_DISTRIBUTION_ADDRESS=          Tier C key release
  PRSM_ONCHAIN_PROVENANCE=0               ProvenanceRegistry
  PRSM_JOB_HISTORY_DIR=                   JobHistory FS persist

Authority thresholds:
  P0 disable           3-of-5 council
  P1 advisory          1 council member
  Deactivation         3-of-5 council
  List-expansion       4-of-5 + auditor co-sign

Latency targets (from issuance):
  P0           < 5 min
  P1           < 30 min
  Deactivate   < 30 min

Dispute channel:
  prsm_disclosure MCP / /admin/disclosure/submit

Status surfaces:
  /health/detailed             current subsystem state
  /info                        sticky log of applied directives
  /audit/recent                directive-application audit ring
  prsm_node_health MCP         AI-assisted triage
```

---

## 7. Cross-references

- `docs/2026-05-08-fleet-kill-switch-scoping.md` — architectural scoping (full design space)
- `docs/security/EXPLOIT_RESPONSE_PLAYBOOK_ANNEX_2026_05.md` — Foundation-side incident response procedure
- `docs/governance/PRSM-CR-2026-05-08.md` — council resolution locking the authority schedule
- `docs/governance/2026-05-06-resource-constrained-audit-strategy.md` (PRSM-POL-2) — audit-trigger conditions for the appeals process
- `docs/2026-04-27-cumulative-audit-prep.md` §7.21 — auditor reading path for the fleet kill-switch scope

---

## 8. Changelog

| Date | Sprint | Change |
|---|---|---|
| 2026-05-13 | 379 | Initial runbook. Closes §7.21 design-doc honest-scope ("operator runbook not yet drafted"). Captures current per-node kill switches (§1) + the fleet-coordination layer's operator-opt-in posture (§2) per the existing scoping doc. Fleet layer remains design-only until §7 promotion triggers fire. |
