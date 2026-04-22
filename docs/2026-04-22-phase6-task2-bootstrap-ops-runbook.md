# Phase 6 Task 2 — Foundation Bootstrap-Node Ops Runbook

**Document identifier:** PHASE6-TASK2-BOOTSTRAP-OPS-1
**Version:** 0.1 Draft
**Status:** Partner-handoff-ready scoping doc. Prescribes the architecture, deploy sequence, signing-key custody, publication flow, monitoring, rotation, and failure-mode runbooks for the Foundation-operated bootstrap-peer infrastructure. Not an IaC repository — specifies what the IaC must achieve; the actual Terraform / Ansible / Pulumi templates are Foundation DevRel execution artifacts authored against this scope.
**Date:** 2026-04-22
**Drafting authority:** PRSM founder
**Related documents:**
- `docs/2026-04-22-phase6-p2p-hardening-design-plan.md` §3.1, §5.3, §6 Task 2 — origin of this runbook.
- `prsm/node/bootstrap.py` — Phase 6 Task 1 implementation (SHIPPED). Signed bootstrap list + HTTPS / DNS fetch orchestration. This runbook specifies the Foundation-side infrastructure that PUBLISHES the signed lists consumed by that module.
- `docs/2026-04-22-phase6-task4-dht-tuning-plan.md` (PHASE6-TASK4-DHT-TUNING-1) — follow-on measurement task that runs AGAINST the bootstrap infrastructure this runbook stands up.
- `PRSM_Vision.md` §6 — supply-architecture framing for geographic distribution.
- `docs/2026-04-22-prsm-supply-1-supply-diversity-standard.md` (PRSM-SUPPLY-1) — geographic-diversity target baseline.

---

## 1. Purpose

PRSM clients locate the P2P network via signed bootstrap peer lists fetched from (1) an HTTPS endpoint and (2) a DNS TXT fallback. This document specifies the Foundation-side infrastructure that:

1. **Runs the three bootstrap nodes** themselves (libp2p peers with long-lived identities, publicly reachable).
2. **Publishes the signed list** pointing at those nodes, rotated quarterly.
3. **Rotates signing keys and bootstrap-node identities** on a documented cadence.
4. **Monitors liveness + signature-cache staleness + fetch success rates** from the client population.
5. **Recovers** from outages, key compromise, and regional failures.

The engineering surface (signed-list verification, HTTPS / DNS fetcher Protocol, fallback orchestration) shipped in Phase 6 Task 1 at `prsm/node/bootstrap.py`. This runbook is the operational counterpart — what a Foundation DevRel role does to make the engineering actually work in production.

---

## 2. Non-goals

- **Not running the DHT itself.** Bootstrap nodes are discovery entry points, not authority. The DHT runs across all PRSM peers; bootstrap nodes seed membership.
- **Not a full PRSM node.** Bootstrap nodes run the libp2p + transport layer only. They do NOT participate in compute marketplace, storage, content royalties, or FTNS settlement.
- **Not a permanent address book.** Bootstrap peer identities rotate quarterly; consumers always re-fetch the signed list.
- **Not a CDN / anycast.** The signed list is small (<10 KB) and cacheable; plain HTTPS + DNS suffice. Anycast + global CDN are Phase 6.x scope if bootstrap-fetch traffic ever exceeds what a single origin can serve.

---

## 3. Architecture

### 3.1 Three bootstrap nodes, three regions

Minimum deployment per plan §2.1: three Foundation-operated bootstrap nodes in three distinct regions.

| Node | Region | Cloud / Provider | Public address form |
|------|--------|------------------|---------------------|
| `bootstrap-0` | us-east | AWS / GCP / Hetzner (Foundation choice) | `/ip4/<public-ip>/tcp/4001/p2p/<peer-id>` |
| `bootstrap-1` | us-west or eu-west | Different provider from bootstrap-0 | Same form |
| `bootstrap-2` | eu-west or ap-south | Different provider from both 0 and 1 | Same form |

**Why three:** Phase 1 bootstrap per §3.1 connects clients to 3 random peers from the list. With only two bootstrap nodes, a single-region outage halves the discovery surface. Three nodes across three providers yields single-region-outage tolerance under the Phase 6 Task 4 DHT tuning plan's ≥99% lookup-success target.

**Why different providers:** cloud-provider correlated outages (AWS us-east going down 2021, Fastly 2021, etc.) demonstrate that same-provider redundancy across regions still correlates. Three providers breaks correlation.

### 3.2 Bootstrap-node software surface

Each bootstrap node runs:

- **libp2p** (same version pinned in Phase 6 Task 1 + Phase 6 Task 4 measurement).
- **The bootstrap-node helper script** (`scripts/run_bootstrap_node.py`, not yet shipped — Task 2 engineering deliverable).
- **Prometheus node exporter** for host-level metrics.
- **Node-specific Prometheus exporter** emitting:
  - `p2p_connected_peers` (gauge)
  - `bootstrap_fetch_requests_per_minute` (counter — proxied from nginx if HTTPS lives on the same host)
  - `libp2p_incoming_connections_total` (counter)
- **systemd service** for process supervision + auto-restart.
- **Hardened SSH** (ed25519 keys only, no password auth, fail2ban).

### 3.3 Bootstrap list publication

The signed bootstrap list lives in two places per plan §3.1:

1. **HTTPS primary.** `https://bootstrap.prsm.ai/bootstrap-nodes.json` (or equivalent domain). Static JSON file; served by nginx on a Foundation-operated host. TLS via Let's Encrypt with auto-renewal.
2. **DNS TXT fallback.** `_prsm-bootstrap.prsm.ai`. Signed-list JSON split across 255-byte TXT chunks. DNS authoritative server under Foundation control (route53 / cloudflare DNS / etc. — DNS provider need not match the HTTPS host provider).

**Why both:** HTTPS primary is easy to update + reason about; DNS TXT fallback is the censorship-resistant path when `bootstrap.prsm.ai` is blocked by a network operator or DNS-over-HTTPS policy. `prsm/node/bootstrap.py` orchestrates primary→fallback per Phase 6 Task 1's design.

### 3.4 Signing-key custody

The Ed25519 keypair that signs the bootstrap list is a **load-bearing Foundation secret** — anyone with it can redirect the entire PRSM client population to a hostile bootstrap set. Custody requirements:

- **Hardware-backed.** YubiKey (OpenPGP applet) or equivalent. Key never extractable from hardware; signatures produced on-device.
- **Multi-sig custody policy.** At least 2-of-3 Foundation officers must consent before a new signed list can be produced. Enforced socially (signing-ceremony meetings) until §6 rotation automation can enforce it technically.
- **Public-key pinned in client binary.** Plan §3.1 commitment: client ships with the pubkey compiled in. Rotation requires a client release.

**Key rotation window:** 18-month keypair lifetime. New keypair published in the release BEFORE the old one expires, giving clients one release-cycle of dual-signing overlap (both signatures accepted) before the old pubkey is removed from the binary.

### 3.5 Bootstrap-node identity rotation

Separate from the signing key, each bootstrap node has its own Ed25519 peer_id (libp2p identity). Unlike the signing key, these rotate on a quarterly cadence:

- Old node remains in the signed list for a 14-day overlap window after rotation.
- New node joins the list with a fresh peer_id + same public IP or a new one.
- Clients fetch the updated list on next startup; long-running clients refresh on the 30-day expiry per plan §3.1.

---

## 4. Deploy sequence

### 4.1 Infrastructure-as-Code layout

Recommend Terraform for cloud provisioning + Ansible for node configuration. One repo (`prsm-bootstrap-infra`) under Foundation DevRel control, separate from the main PRSM monorepo so rotation cadence is independent.

```
prsm-bootstrap-infra/
├── terraform/
│   ├── us-east/          # Provider A (AWS or GCP)
│   ├── us-west-or-eu/    # Provider B (different)
│   ├── eu-or-apsouth/    # Provider C (third)
│   └── shared/           # DNS zones, TLS cert automation
├── ansible/
│   ├── roles/
│   │   ├── libp2p_bootstrap/
│   │   ├── nginx_bootstrap_list/
│   │   ├── prometheus_exporter/
│   │   └── hardening/
│   └── playbooks/
│       ├── deploy_bootstrap_node.yml
│       ├── rotate_bootstrap_identity.yml
│       └── rotate_signing_key.yml
├── scripts/
│   ├── sign_bootstrap_list.py     # Signs via YubiKey; used by Foundation officers.
│   └── publish_signed_list.py     # HTTPS upload + DNS TXT update.
└── docs/
    ├── runbook.md                 # Runbook references THIS doc.
    └── signing-ceremony-checklist.md
```

### 4.2 Initial deploy checklist

One-time, at Foundation formation:

1. **Procure YubiKeys.** 3 officers, one key each. Initialise with a fresh Ed25519 keypair (never imported from elsewhere).
2. **Publish pubkey.** Compile the pubkey into `prsm/node/bootstrap.py`'s `PINNED_FOUNDATION_PUBKEY` constant. Ship in the next PRSM release.
3. **Provision three VMs.** Terraform apply per region. Confirm:
   - Public IPv4 + IPv6 (dual-stack).
   - Inbound tcp/4001 (libp2p).
   - Outbound egress unrestricted.
   - SSH accessible only from Foundation bastion host with hardware-key auth.
4. **Generate bootstrap-node identities.** On each node, run `prsm-node bootstrap init` to produce a fresh Ed25519 peer_id. Peer_ids go in the signed list.
5. **Install + start bootstrap helper.** Ansible run deploys the libp2p bootstrap process, systemd service, nginx + TLS cert, Prometheus exporter.
6. **First signing ceremony.** 2-of-3 officers with YubiKeys present; run `sign_bootstrap_list.py` against the three peer_ids collected in (4). Produces the initial `bootstrap-nodes.json`.
7. **Publish signed list.** `publish_signed_list.py` uploads to `https://bootstrap.prsm.ai/bootstrap-nodes.json` AND updates `_prsm-bootstrap.prsm.ai` TXT records. Both take effect within DNS-propagation bounds (~5 min).
8. **Smoke test.** From a client machine, run `prsm node bootstrap verify` — fetches the list from both HTTPS + DNS, verifies signature, confirms all 3 peer_ids dial.

### 4.3 Ongoing deploy cadence

- **Daily:** systemd auto-restart handles process crashes.
- **Weekly:** review Prometheus dashboards for anomaly (see §6).
- **Monthly:** apply OS security updates via Ansible. 3-region staggered rollout (one region at a time, 4-hour soak between).
- **Quarterly:** rotate bootstrap-node identities per §3.5.
- **Annually / 18-month:** rotate signing keypair per §3.4.

---

## 5. Signing-list lifecycle

### 5.1 Normal rotation

Every 30 days (per plan §3.1 expiry), a new list is published:

1. Officers schedule a signing ceremony (2-of-3 quorum).
2. `sign_bootstrap_list.py` accepts the current node-identity + multiaddr list, current date, and 30-day `expires_at`. Produces canonical JSON body. Each YubiKey signs the body; signatures aggregate (multi-sig scheme — see §5.3 open issue).
3. `publish_signed_list.py` uploads to HTTPS + DNS.
4. Old list remains valid until its `expires_at`; both coexist during transition. Clients may fetch either depending on which endpoint responds first.

### 5.2 Emergency rotation

If a key-compromise signal fires (§6.4), officers can produce a new list within 4 hours of ceremony start:

1. Rotate every bootstrap-node identity simultaneously (throw out all three `peer_id`s, generate fresh ones).
2. Officer quorum signs the new list with the existing signing key.
3. Publish. Old list still verifies (signature valid, not yet expired), but peer_ids in old list no longer reach reachable nodes → clients fall through to re-fetch.
4. Within 30 days, the old list expires and is purged.

If the SIGNING KEY ITSELF is compromised, emergency response requires a client release with a new pinned pubkey — 2-4 week turnaround. For this reason, §3.4 mandates hardware-backed custody + multi-sig consent.

### 5.3 Open issue — multi-sig scheme

Plan §3.1 specifies Ed25519 single-signer. True 2-of-3 multi-sig over Ed25519 requires either:

- **Pairwise signatures.** Each officer signs independently; verifier checks at least 2 signatures against 3 pinned pubkeys. Larger signature surface (~192 bytes vs 64), but uses off-the-shelf Ed25519.
- **Threshold Ed25519.** FROST / similar threshold scheme produces a single aggregated signature. Cleaner wire format, requires specialised tooling not yet in the Foundation stack.

Recommendation for initial deploy: **pairwise signatures.** Upgrade to FROST when and if Foundation tooling matures. This requires a minor `prsm/node/bootstrap.py` change to accept 1-3 signatures from a 3-pubkey set; ship alongside the next client release.

---

## 6. Monitoring + alerts

### 6.1 Dashboards

Foundation DevRel maintains two Grafana dashboards:

**Bootstrap node health:**
- Per-node: uptime, connected peer count, libp2p incoming connections/min, CPU, memory, disk.
- Aggregate: bootstrap-fetch requests per minute (HTTPS + DNS combined), TLS cert expiry countdown, DNS TXT record TTL countdown.

**Client-side adoption telemetry:**
- Successful bootstrap-list fetches / 15-min window, per source (HTTPS / DNS).
- Signature-verification failures / 15-min (alert on >0.5% rate).
- Client bootstrap attempts that fall through to both sources unavailable.

### 6.2 Alert routing

| Alert | Trigger | Escalation |
|-------|---------|------------|
| `BootstrapNodeDown` | any node's `p2p_connected_peers` = 0 for ≥10 min | PagerDuty primary |
| `BootstrapFetchFailing` | HTTPS 5xx rate > 2% over 5 min | PagerDuty primary |
| `BootstrapListExpirySoon` | `expires_at` < now + 7 days | Foundation DevRel lead, email |
| `SignatureVerificationSpike` | client-side verify failures > 0.5% / 15 min | PagerDuty primary + Foundation security |
| `TLSCertExpirySoon` | cert `notAfter` < now + 14 days | Foundation DevRel lead, email |
| `DNSTXTMismatch` | DNS TXT record hash ≠ HTTPS payload hash | PagerDuty primary |
| `NodeIdentityDuplicatedInList` | two entries in signed list share a peer_id | Foundation security, PagerDuty primary |

### 6.3 Data retention

- Prometheus node metrics: 90 days.
- Client-side fetch telemetry: 30 days (aggregated only; no per-client identifiers).
- Signing-ceremony audit log: **permanent, immutable**. Each ceremony logs: officers present, YubiKey serial numbers used, signed body hash, publish timestamp.

### 6.4 Key-compromise signals

Any of the following triggers emergency rotation (§5.2):

- A YubiKey reported lost / stolen by an officer.
- `SignatureVerificationSpike` alert firing with no pending list rotation.
- A signed list discovered in public distribution that did NOT come from an officer-ceremony log entry.
- Post-compromise attribution by Foundation security (e.g., after supply-chain audit finding).

---

## 7. Rotation procedures

### 7.1 Quarterly bootstrap-identity rotation

- **T-14 days:** DevRel lead drafts rotation plan, schedules signing ceremony.
- **T-7 days:** provision new bootstrap nodes in parallel to old (so the list can temporarily include both sets).
- **T=0 (ceremony):** officers sign new list containing both old and new peer_ids.
- **T+14 days:** issue a second signing ceremony; new list omits old peer_ids. Decommission old nodes.

Total 28-day rolling window, full overlap for 14 days, operator-cost 2× nodes for 7 days (acceptable at bootstrap scale).

### 7.2 18-month signing-key rotation

- **T-6 months:** Foundation publishes rotation schedule + new pubkey. Client release (`vN.M`) includes BOTH old and new pubkeys as trusted. Verifier accepts lists signed by either.
- **T-3 months:** switch signing ceremonies to use new keypair. Lists signed with new key, verified by new pubkey (also accepted by old clients that have upgraded).
- **T=0:** client release (`vN.M+1`) removes old pubkey. Clients not yet upgraded can still verify lists signed with old key until their ship release propagates.
- **T+3 months:** all clients on `vN.M+1`. Old pubkey fully retired.

### 7.3 Emergency signing-key rotation

If key compromise confirmed:

1. **Within 4 hours:** Foundation officers regenerate signing keypair on fresh YubiKeys. Security disposal of compromised keys.
2. **Same day:** emergency client release with new pubkey. Released via the usual channels + explicit warning to operators not running auto-update.
3. **Within 7 days:** all active PRSM node versions contain the new pubkey. Old pubkey removed. Old lists continue to verify until their 30-day expiry; no new lists signed with the compromised key.

Operational cost: one emergency release + one week of aggressive operator outreach. The 18-month lifetime + multi-sig + hardware-backed custody make this path very unlikely.

---

## 8. Failure-mode runbooks

### 8.1 Single bootstrap node down

- Alert: `BootstrapNodeDown`.
- Impact: reduced discovery surface. Clients fall through to remaining 2 nodes (still within 30-day list expiry window).
- Response: DevRel on-call diagnoses + restarts or re-provisions within 1 hour. No signing-list change needed unless node is gone for >7 days.
- Re-publish list only if node identity changes during recovery.

### 8.2 HTTPS primary unreachable

- Alert: `BootstrapFetchFailing`.
- Impact: clients fall through to DNS TXT fallback. `prsm/node/bootstrap.py`'s orchestration handles this gracefully.
- Response: DevRel diagnoses TLS / nginx / DNS / cloud-provider issue. SLA target: 2-hour restoration.

### 8.3 Both HTTPS and DNS unreachable

- Alert: `BootstrapFetchFailing` + client-side telemetry shows fall-through exhaustion.
- Impact: new clients cannot discover the network. Existing long-running clients unaffected until next list fetch.
- Response: both paths down simultaneously is a major incident. DevRel lead + Foundation security paged. Likely indicates correlated provider failure OR active DNS hijack — §6.4 key-compromise signals overlap here.

### 8.4 Signing key compromise confirmed

- Alert: `SignatureVerificationSpike` or independent attribution.
- Response: §5.2 + §7.3 emergency rotation.
- Post-mortem required. Ship within 14 days.

### 8.5 Regional provider outage

- Alert: `BootstrapNodeDown` for one node; others healthy.
- Impact: reduced but not degraded discovery.
- Response: if outage projected >6 hours, temporarily re-provision that node with a different provider (new peer_id). Sign a fresh list with the new peer_id if rotation window permits; otherwise tolerate the 2-node period until planned rotation.

---

## 9. Promotion triggers (the "when to fund this" test)

Task 2 is a Foundation-ops deliverable, not an engineering-track deliverable. It moves from "scoped" to "funded" when **all** of:

### T1: Phase 6 Task 1 SHIPPED

Already satisfied (`phase6-merge-ready-20260422` tag).

### T2: Foundation entity operational

Requires PRSM-GOV-1 Foundation formation (jurisdiction selected, entity formed, officers named, YubiKeys procured, bank-account funding for cloud provider bills). Target per PRSM-GOV-1 §8.4: Year-1 milestone.

### T3: Foundation DevRel role filled

Ops runbooks need a maintainer. Rotation cadences (monthly OS updates, quarterly identity rotation, 18-month key rotation) require a human with named responsibility. Not an intern / volunteer role.

### T4: Mainnet + pre-Phase-4-onboarding scale

Bootstrap infrastructure matters most once real users start hitting the network. Phase 1.3 mainnet deploy + Phase 4 wallet onboarding together represent the "real users" inflection. Target window: Q3-Q4 2026.

**Operational reading:** Phase 6 Task 2 moves to execution when (T2) Foundation is formed + (T3) DevRel role hired + (T4) pre-Phase-4 readiness. Likely same quarter as Phase 4 Tasks 3-4 frontend launch.

---

## 10. Relationship to neighbouring work

- **Phase 6 Task 1** (SHIPPED) — client-side fetch + verify. This runbook specifies the Foundation-side infrastructure Task 1 consumes.
- **Phase 6 Task 4 DHT tuning** — measurement task that runs AGAINST the infrastructure this runbook stands up. Task 4 is a downstream dependency.
- **PRSM-GOV-1 Foundation formation** — T2 trigger. The signing-ceremony multi-sig policy assumes Foundation officers exist with named roles.
- **PRSM-SUPPLY-1 geographic diversity** — bootstrap-node regional placement aligns with SUPPLY-1's diversity targets. Bootstrap nodes themselves don't count toward compute diversity metrics (they don't provide compute) but are a structural signal of Foundation's regional-diversity commitment.
- **Phase 4 Wallet SDK** — frontend onboarding requires bootstrap to be live. Task 2 is a soft dependency for Phase 4 frontend launch.

---

## 11. What a partner DevRel role receives

Day 1 handoff package:

- This scoping doc (PHASE6-TASK2-BOOTSTRAP-OPS-1).
- `docs/2026-04-22-phase6-p2p-hardening-design-plan.md` — source Phase 6 plan.
- `prsm/node/bootstrap.py` + `tests/unit/test_bootstrap_discovery.py` — client-side implementation + test suite to validate against.
- Repo skeleton (`prsm-bootstrap-infra`, Foundation-controlled) — empty but structured per §4.1.
- Budget allocation for 3 cloud VMs + DNS + TLS cert + YubiKey procurement.
- Foundation officer roster (required for signing ceremonies).

First-milestone deliverable: initial three-region deploy + signing ceremony + published list that `prsm node bootstrap verify` successfully validates against the real infrastructure. 4-week ramp.

---

## 12. Success criteria

Task 2 is considered successful when:

1. **Three bootstrap nodes live** across three providers / three regions, each reachable on `tcp/4001` from the public internet.
2. **A signed bootstrap list published** at both HTTPS primary and DNS TXT fallback, verifiable by `prsm/node/bootstrap.py` clients using the pinned Foundation pubkey.
3. **Monitoring + alerts live** per §6.2, paging DevRel on-call within 5 min of any listed trigger.
4. **Rotation cadence proven** — at least one successful quarterly identity rotation executed end-to-end.
5. **Audit-trail retained** for all signing ceremonies per §6.3.
6. **Runbook tested** against §8 failure modes in a non-production drill.

---

## 13. Risk register

### R1: Foundation delay

T2 delay blocks T3 blocks Task 2 delivery. Mitigation: the Phase 6 engineering tasks run independent of bootstrap infra (state machines are CI-tested in chaos harness). Bootstrap absence blocks real user onboarding but not engineering progress.

### R2: Signing-key compromise early in bootstrap life

If a key is compromised within 6 months of deploy (before the first 18-month rotation even completes), emergency rotation happens at the worst possible time (pre-mainnet scale, limited client-upgrade velocity). Mitigation: §3.4 hardware custody + multi-sig ceremony discipline from day 1. Cheap to set up right once.

### R3: DNS provider acquisition / policy change

DNS TXT fallback depends on Foundation's DNS provider remaining available + supporting TXT record updates via API. If a provider changes policy, DevRel has 24-48h to migrate. Mitigation: cross-check DNS provider's TXT-update SLA annually.

### R4: TLS provider migration

Let's Encrypt (or chosen CA) remaining available + free is assumed. If the model changes, DevRel migrates to another free CA or accepts a commercial cost. Low probability, contained blast radius.

### R5: Over-engineering before scale

Three nodes + manual signing ceremonies are sufficient for the first 10,000 clients. Building out anycast, automated multi-sig, per-region redundancy BEFORE that scale is premature. Mitigation: this runbook caps initial scope at the minimum viable; re-review at 10,000-client scale.

---

## 14. Changelog

- **0.1 (2026-04-22):** initial scoping doc. Promotes Phase 6 Task 2 from design-doc bullet points to partner-handoff-ready DevRel ops runbook. Closes Phase 6 planning — Tasks 1, 3, 5, 6a, 6b, 7 engineering shipped; Tasks 2 + 4 both now at partner-handoff-ready scoping-doc status. Execution of Task 2 pending T1-T4 triggers per §9.
