# Phase 6: P2P Network Hardening — Design + TDD Plan

**Date:** 2026-04-22
**Target execution:** Q2 2027 (per `docs/2026-04-10-audit-gap-roadmap.md` Phase 6).
**Status:** Combined design + TDD plan drafted ahead of execution. Follows Phase 7 / Phase 8 pattern.
**Depends on:**
- Phase 2 remote-compute dispatch shipped (existing libp2p path is the substrate to harden).
- Phase 4 wallet onboarding (user-visible error surfaces and reporting flows).
- Foundation infrastructure budget for bootstrap-node operation (~3 geographically distributed nodes).

---

## 1. Context & Goals

PRSM's P2P layer today is a libp2p prototype that works in dev and staging topologies but has not been load-tested under real-world adversarial conditions: asymmetric NAT, bootstrap-node failure, DHT poisoning, high-churn operator pools. Once Phases 1-5 attract real users and operators, these become production blockers.

Phase 6 hardens the P2P layer without changing the protocol semantics. It is the infrastructure-scaling deliverable that sits between "marketplace works on 3-10 nodes in a trusted topology" and "marketplace works with thousands of nodes in adversarial conditions including malicious participants."

### 1.1 Non-goals for Phase 6

- **Not a protocol redesign.** Wire format + application-level messages unchanged.
- **Not a new transport.** libp2p stays; this phase tunes + hardens it.
- **Not a centralization layer.** Bootstrap nodes are discovery hints, not authority.
- **Not compute / storage / content scope.** Those are Phase 2 / Phase 7 / Phase 1.3.
- **Not an observability product.** Foundation-internal dashboard only; operator-facing dashboards are out of scope.

### 1.2 Backwards compatibility

- Existing deployed nodes continue to work; the upgrade path is additive.
- Bootstrap-node list is updated via signed configuration; nodes fetching the list get new bootstrap peers on next restart.
- Protocol version unchanged.

---

## 2. Scope

### 2.1 In scope

**Infrastructure:**
- Bootstrap node infrastructure: ≥3 geographically distributed (US-east, US-west, EU or APAC). Foundation-operated or contracted.
- Signed bootstrap-peer list published via HTTPS + fallback DNS. Clients verify signature against a published Foundation key.

**libp2p hardening:**
- ICE/STUN/TURN integration for NAT traversal (essential for residential T1/T2 operators).
- Kademlia DHT parameter tuning: replication factor, bucket sizes, query timeout.
- Connection liveness + automatic peer eviction on dead connections.
- Rate limiting on DHT queries per peer (anti-amplification).

**Observability:**
- Internal dashboard tracking: connected peers per node, DHT query latency p50/p95, bootstrap-node reachability, NAT-traversal success rates, churn events.
- Metrics export (Prometheus / OpenTelemetry).
- Alert hooks for critical events (bootstrap node down, DHT convergence failure).

**Transport upgrade:**
- gRPC-streaming alternative path for Phase 2 compute dispatch at >10 MB shards. Phase 2 deferred shards >10 MB; Phase 6 closes that gap.

**Testing:**
- Chaos testing framework: 100-node simulated network with 30% churn, geographic-diversity simulation, adversarial-peer injection.
- Fault injection library: drop packets, partition network, delay messages, restart peers.

### 2.2 Out of scope

- **Custom transport / protocol design.** Use libp2p as-is with configuration tuning.
- **End-to-end encrypted transport.** libp2p provides this already (Noise protocol).
- **Consensus-layer hardening.** Phase 7 / 7.1 / 7.1x covers the consensus path; Phase 6 hardens the transport underneath.
- **Anti-DDoS beyond basic rate limits.** DDoS mitigation beyond rate limits is ops-side (Cloudflare / similar) not P2P.
- **Mobile support.** libp2p mobile clients are out of scope for Phase 6; defer to Phase 6.x if mobile rollout proceeds.

### 2.3 Deferred

- **Multi-region bootstrap failover.** Initial bootstrap-node deploy is static; automatic failover between regions deferred to Phase 6.x.
- **libp2p relay hosting.** Running libp2p relays (for unreachable peers) adds operational burden; defer until volume justifies.

---

## 3. Protocol

### 3.1 Bootstrap discovery

```
Node starts up
        │
        ▼
Fetches signed bootstrap list from:
   - https://bootstrap.prsm.ai/bootstrap-nodes.json (primary)
   - DNS TXT record _prsm-bootstrap.prsm.ai (fallback)
        │
        ▼
Verifies list signature against published Foundation pubkey
        │
        ▼
Connects to N random bootstrap peers (N=3 default)
        │
        ▼
Performs initial DHT bootstrap query
        │
        ▼
Ready.
```

Bootstrap list signature: Foundation Ed25519 key; pubkey committed in client binary. Expiry: 30 days. Rotation: next list published ≥14 days before expiry.

### 3.2 NAT traversal

Sequence for a new peer:

1. Node detects its NAT type via STUN to a Foundation-operated server.
2. If fully cone NAT or better → direct dial possible.
3. If restricted cone / port-restricted → uses ICE with STUN to coordinate.
4. If symmetric NAT → falls back to TURN relay (Foundation-operated or contracted).
5. If TURN unavailable → marks peer as inbound-only; outbound dials only.

libp2p's AutoNAT + AutoRelay do most of this; Phase 6 ensures they're configured correctly.

### 3.3 DHT tuning

Kademlia parameters:
- Replication factor `k`: currently default 20; tune to 10 based on network size vs lookup success rate.
- Bucket size: 20 (default).
- Alpha (concurrent queries): 3 (default).
- Query timeout: 30 seconds (default) → tune to 10 seconds based on measurements.
- Republish interval: 22 hours (default) → keep.

Empirical calibration in Task 4 testnet exercise.

### 3.4 Connection liveness + eviction

- Ping interval: 30 seconds.
- Dead-connection threshold: 3 consecutive missed pings.
- On detection: close connection + emit event.
- Connection manager keeps `low_water` and `high_water` marks (default 20 / 100); new connections accepted until `high_water`, then oldest idle connection evicted on accept.

### 3.5 gRPC streaming upgrade

For shards >10 MB (Phase 2 §line-item-B deferred these):

- Fallback to gRPC-over-TLS direct between operator + dispatcher.
- gRPC streaming chunks payload; libp2p used only for discovery + coordination.
- Receipt signing path unchanged — gRPC stream and libp2p stream both produce the same `ShardExecutionReceipt`.

### 3.6 Rate limiting

Per-peer limits:
- DHT queries: 100 per minute.
- Direct messages: 500 per minute.
- Shard dispatch requests: 50 per hour (marketplace-layer, not P2P-layer, but surfaced here for completeness).

Violations: peer transitions to rate-limited-for-60-seconds state; sustained violations trigger local ban for 1 hour.

---

## 4. Data model

### 4.1 Bootstrap list format

```json
{
  "version": 1,
  "expires_at": "2027-05-01T00:00:00Z",
  "bootstrap_peers": [
    {
      "peer_id": "12D3KooW...",
      "multiaddrs": ["/ip4/1.2.3.4/tcp/4001", "/ip6/2001:db8::1/tcp/4001"],
      "region": "us-east",
      "operator": "PRSM Foundation"
    }
  ],
  "signature": "base64-signed-blob"
}
```

### 4.2 Observability metrics

```python
@dataclass(frozen=True)
class P2PMetrics:
    connected_peers: int
    inbound_connections: int
    outbound_connections: int
    dht_query_p50_ms: float
    dht_query_p95_ms: float
    dht_successful_lookups_per_sec: float
    bootstrap_reachable_count: int
    nat_type: str  # "cone" | "restricted" | "port-restricted" | "symmetric"
    nat_traversal_success_rate: float
    peer_churn_events_per_hour: int
    timestamp_unix: int
```

### 4.3 Chaos test spec

```python
class ChaosScenario:
    node_count: int = 100
    churn_rate: float = 0.30   # 30% of nodes cycle per hour
    geographic_regions: int = 4
    adversarial_peers: int = 5  # nodes that return invalid responses
    duration_minutes: int = 60

    def run(self) -> ChaosReport:
        # spin up simulated network
        # inject churn + adversarial behavior
        # measure: DHT lookup success rate, shard dispatch success rate,
        #          bootstrap-node load, NAT traversal success rate
        ...
```

---

## 5. Integration points

### 5.1 `prsm/node/` modules

libp2p wrapper modules gain configuration hooks for:
- Bootstrap peer list source.
- NAT-traversal strategy selection.
- Rate-limiter hooks.
- Observability emitter.

### 5.2 Phase 2 RemoteShardDispatcher

Accepts optional gRPC transport for >10 MB shards. Existing libp2p path remains default for ≤10 MB.

### 5.3 Foundation ops surface

- Bootstrap node deploy playbook.
- DNS TXT record management.
- Signing-key management for bootstrap list.
- Dashboard access for Foundation staff.

### 5.4 Observability stack

Prometheus / OpenTelemetry export from every PRSM node. Foundation dashboard scrapes; operators get their own node's metrics locally without requiring external telemetry upload.

---

## 6. TDD plan

**7 tasks**.

### Task 1: Signed bootstrap list + discovery

- Bootstrap list signing + verification.
- Fetch with HTTPS primary + DNS fallback.
- List rotation on expiry.
- Tests: signature verification (valid ✓, tampered ✗, expired ✗); fallback to DNS when HTTPS fails; graceful degradation when both sources unavailable.
- Expected ~12 tests.

### Task 2: Bootstrap node infrastructure deploy

- Foundation-operated deploy playbook (Terraform / Ansible / similar).
- 3 geographically distributed nodes.
- Monitoring + auto-restart.
- Rotation procedure.
- **Output:** operational runbook + 3 live bootstrap nodes.

### Task 3: NAT traversal integration

- libp2p AutoNAT + AutoRelay configuration.
- STUN server (Foundation-operated or contracted).
- TURN relay fallback (Foundation-operated or contracted).
- Tests: NAT type detection; connection success per NAT type; relay fallback when direct fails.
- Expected ~15 tests.

### Task 4: DHT tuning via testnet measurement

**Full scoping doc:** [`2026-04-22-phase6-task4-dht-tuning-plan.md`](./2026-04-22-phase6-task4-dht-tuning-plan.md) (PHASE6-TASK4-DHT-TUNING-1).

- 50-100 node testnet deployed.
- Parameter sweep: replication factor, query timeout, bucket size, alpha, republish interval.
- Choose values minimizing p95 lookup latency subject to ≥99% lookup success.
- Five preregistered hypotheses (H1-H5) with pass/fail/null gates. Phased 5-week execution (prep → H1 sweep → H2+H3+H4 parallel → H5 + final validation → commit).
- Promotion triggers: T1 Phase 6 Tasks 1+3+5+6+7 shipped (DONE, `phase6-merge-ready-20260422`), T2 Foundation bootstrap infra live (Task 2), T3 Foundation SRE capacity, T4 pre-mainnet-scale interest.
- **Output:** tuned parameter set committed to `prsm/node/libp2p_*` defaults + publishable measurement report.

### Task 5: Connection liveness + rate limiting

- Ping-based liveness detection.
- Peer eviction on dead connection.
- Per-peer rate limits with violation tracking.
- Tests: dead-peer eviction timing; rate-limit enforcement; auto-ban on sustained violations; recovery after ban.
- Expected ~15 tests.

### Task 6: Observability dashboard + gRPC streaming

- Prometheus / OTel emission from every node.
- Foundation dashboard (Grafana or similar).
- gRPC transport for >10 MB shards.
- Tests: metrics export correctness; dashboard queries; gRPC round trip ≥ 100 MB payload.
- Expected ~15 tests.

### Task 7: Chaos-test harness + merge-ready tag

- `tests/chaos/` harness per §4.3.
- 100-node simulated network, 30% churn, 1-hour run.
- Pass criteria: ≥95% DHT lookup success; ≥90% shard dispatch success (Phase 2 baseline).
- Adversarial-peer injection verifies malicious responses are dropped without degrading network.
- `phase6-merge-ready-YYYYMMDD` tag.

---

## 7. Acceptance criterion

- **Bootstrap reachability:** new nodes reach ≥2 of 3 bootstrap peers within 10 seconds of startup, 99% of the time across 100 trials.
- **NAT traversal:** ≥95% success rate for consumer-residential NAT (measured on at least 10 real residential networks across regions).
- **DHT lookup:** p95 latency ≤500 ms; ≥99% success rate under ≤50% churn.
- **Chaos test:** 100-node / 30%-churn / 60-minute scenario maintains ≥95% DHT lookup success and ≥90% shard dispatch success.
- **>10 MB shards:** Phase 2 scenarios that previously hit the 10 MB ceiling now run via gRPC without change in correctness semantics.

---

## 8. Open issues

### 8.1 Bootstrap node hosting model

Three candidates:
- Foundation-operated on Foundation-rented cloud (AWS / GCP / Azure).
- Contracted operation to a T4 meganode partner (e.g., Prismatica).
- Volunteer-operated via a Foundation-vetted list.

**Tentative recommendation:** Foundation-operated on 3 different cloud providers (one per region) to avoid single-vendor correlation with PRSM-SUPPLY-1 concentration concerns.

### 8.2 TURN relay hosting

TURN is bandwidth-heavy. Operating at MVP scale is affordable; at production scale requires CDN-level capacity. Candidates: Cloudflare Realtime SFU, commercial TURN-as-a-service (Twilio, Metered), self-hosted coTURN.

**Tentative recommendation:** coTURN on Foundation infrastructure for MVP; re-evaluate at 10k-user threshold.

### 8.3 Observability data retention

Metrics retained for 30 days by default; 90 days for high-value operators. Long-term aggregates for Foundation transparency reporting. Precise retention policy depends on Foundation storage budget.

### 8.4 Rate-limit tuning

Initial limits (§3.6) are conservative. Real-operator behavior under peak demand may justify relaxation; high-spam periods may justify tightening. Task 7 chaos-test runs inform initial calibration.

### 8.5 Relationship with PRSM-SUPPLY-1 diversity

Bootstrap nodes being 3-of-3 hosted by one cloud provider would count as a SUPPLY-1 concentration concern if the network itself is provider-diverse. Mitigation: specify bootstrap hosting as 1-per-3-providers explicitly (§8.1). Not required, but good hygiene.

### 8.6 gRPC TLS certificate management

gRPC transport requires TLS certificates per node. Let's Encrypt + cert-manager for Foundation infrastructure; node-level certificate issuance for operators needs an automated path. Defer to Task 6 implementation review.

### 8.7 Chaos-test harness maintainability

100-node simulation consumes CI resources. Likely runs nightly on a dedicated runner, not per-commit. Acceptable given Phase 6's production-infrastructure role.

---

## 9. Dependencies + risk register

### R1 — libp2p breaking changes

libp2p releases breaking-change minor versions more frequently than stable protocols. Mitigation: pin to a specific libp2p version during Phase 6 development; migrate under controlled conditions.

### R2 — Bootstrap node compromise

A compromised bootstrap node could serve malicious peer lists. Mitigation: signing verification on bootstrap list; operator responsibility on Foundation side; incident-response playbook.

### R3 — Residential NAT unsolved cases

Some residential NATs (especially carrier-grade NAT / CGNAT) cannot be traversed even with TURN. Affected users become inbound-only peers. Mitigation: document the limitation; operators requiring full peer participation can use port forwarding.

### R4 — Churn beyond 30% in practice

Chaos test targets 30% churn/hour. Real networks may exceed this during major events (cloud provider outage, regulatory action). Mitigation: observability alerts on extreme churn; design degrades gracefully rather than failing hard.

### R5 — Bandwidth costs for bootstrap nodes + TURN relays

Bandwidth-heavy services can run up Foundation cloud bills. Mitigation: budget line in Foundation treasury; usage-based scaling + TURN-relay sponsorship partnerships if needed.

### R6 — Observability-stack vendor lock-in

Grafana Cloud + Prometheus or equivalent — if vendor changes pricing or offering, migration is possible but non-trivial. Mitigation: use standard OpenTelemetry where possible; dashboard definitions stored as code.

---

## 10. Estimated scope

- **7 tasks.**
- **Expected LOC:** ~1200 Python + ~500 infrastructure-as-code.
- **Test footprint target:** +~60 unit + ~15 chaos-scenario integration tests.
- **Calendar duration:** 4-5 weeks engineering + 1 week bootstrap deploy.
- **Budget:** bootstrap node infrastructure ~$500-$1500/month; TURN relay bandwidth costs scale with traffic; observability stack ~$100-$500/month at MVP.

---

## 11. Changelog

- **0.1 (2026-04-22):** initial design + TDD plan. Promotes Phase 6 from master-roadmap stub to partner-handoff-ready scoping.
