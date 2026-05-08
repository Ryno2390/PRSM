# AggregateRequest / AggregateResponse — RPC design

**Date:** 2026-05-08
**Track:** B3 from `docs/2026-05-08-query-orchestrator-wiring-readiness.md`
**Pattern-lift:** Phase 3.x.7 `RunLayerSliceRequest` (signed-handoff
chain RPC) + Phase 3.x.1 `InferenceReceipt` (signed receipt with
canonical signing payload).

---

## What this RPC does

The QueryOrchestrator's `AggregatorClient` Protocol takes a selected
T2+ aggregator + the swarm's DP-noised partials and gets back the
combined plaintext + a signed `AggregationCommit`. This document
specifies the wire format that satisfies the Protocol contract.

Per the threat model: **the aggregator MUST commit BEFORE delivering
plaintext** (A9). This is enforced by the response shape — the
commit and plaintext arrive in the same response, and the orchestrator
verifies the commit's `result_digest == sha256(plaintext)` before
accepting.

## Message types

Two new entries in `ChainRpcMessageType` (extends Phase 3.x.7
existing protocol):

```python
AGGREGATE_REQUEST = "aggregate_request"
AGGREGATE_RESPONSE = "aggregate_response"
```

No protocol-version bump required — additive enum extension follows
the established pattern (Phase 3.x.10 `TOKEN_FRAME` was added the
same way at v2).

## AggregateRequest

```python
@dataclass(frozen=True)
class AggregateRequest:
    # Routing
    protocol_version: int                  # currently 2
    request_id: bytes                      # 32 bytes — distinct per (query_id, attempt)
    query_id: bytes                        # 32 bytes — A9 binding key
    
    # Workload
    manifest_json: str                     # InstructionManifest.to_json()
    partials: tuple[SignedPartial, ...]    # the DP-noised per-shard outputs
    
    # Routing back to prompter
    prompter_pubkey: bytes                 # 32 bytes Ed25519 — for plaintext encryption
    prompter_node_id: str                  # routing only; A2 binding key
    
    # Selection-context (forensic anchor for A6)
    beacon_used: bytes                     # 32 bytes — the beacon at selection time
    aggregator_pubkey_hash: bytes          # 32 bytes — A8 identity binding
    
    # Budget
    ftns_budget: int                       # base units; aggregator cannot consume more
    deadline_unix: int                     # request expiry
    
    # Sender attestation
    prompter_signature: bytes              # 64 bytes Ed25519 over signing_payload()
```

### `SignedPartial`

```python
@dataclass(frozen=True)
class SignedPartial:
    shard_cid: str
    payload: bytes                         # DP-noised partial
    creator_id: str                        # for royalty routing
    dp_noise_applied: bool                 # A5 marker — verified server-side
    source_agent_pubkey: bytes             # 32 bytes Ed25519
    source_agent_signature: bytes          # 64 bytes Ed25519 over partial
    privacy_budget_consumed: float         # epsilon spent on this partial
```

### Signing payload

`AggregateRequest.signing_payload()` → bytes for prompter to sign. Mirrors
`HandoffToken.signing_payload()` from Phase 3.x.7:

```
"prsm:aggregate:v1\n"
+ protocol_version (uvarint)
+ request_id (32 bytes)
+ query_id (32 bytes)
+ sha256(manifest_json) (32 bytes)
+ len(partials) (uvarint) + per-partial sha256(serialize(partial)) concatenated
+ prompter_pubkey (32 bytes)
+ prompter_node_id (utf8 length-prefixed)
+ beacon_used (32 bytes)
+ aggregator_pubkey_hash (32 bytes)
+ ftns_budget (uvarint)
+ deadline_unix (uvarint)
```

## AggregateResponse

```python
@dataclass(frozen=True)
class AggregateResponse:
    protocol_version: int                  # 2
    request_id: bytes                      # echoed from request
    query_id: bytes                        # echoed
    
    # Pre-commit (A9 — committed BEFORE plaintext delivery)
    commit: AggregationCommit              # query_id + aggregator_pubkey_hash + result_digest
    commit_signature: bytes                # 64 bytes Ed25519 over commit
    
    # Encrypted plaintext result
    encrypted_plaintext: bytes             # x25519+chacha20-poly1305 to prompter_pubkey
    nonce: bytes                           # 24 bytes
    
    # Forensic
    aggregator_pubkey: bytes               # 32 bytes Ed25519 — verifier resolves to pubkey_hash
    privacy_budget_consumed: float         # sum across partials, post-combination
    contributing_creators: tuple[str, ...] # for RoyaltyDistributor
    completed_unix: int                    # response timestamp
```

### Signing payload

`AggregationCommit.signing_payload()`:

```
"prsm:aggregation-commit:v1\n"
+ query_id (32 bytes)
+ aggregator_pubkey_hash (32 bytes)
+ result_digest (32 bytes)
```

The `commit_signature` covers this payload. The aggregator computes
`result_digest = sha256(plaintext_before_encryption)` — the prompter
decrypts the response, hashes the plaintext, and verifies it matches
`commit.result_digest`. Mismatch = `AggregationCommitMismatchError`
= slash route.

## Server-side flow

```
on AggregateRequest:
    1. verify protocol_version supported
    2. verify deadline_unix not yet passed
    3. verify prompter_signature over signing_payload()
    4. resolve aggregator_pubkey_hash to local node identity (refuse
       if not addressed to us)
    5. verify each SignedPartial's source_agent_signature
    6. for each partial:
         - assert dp_noise_applied == True
         - assert privacy_budget_consumed > 0
       (A5 enforcement at server side, mirrors swarm_runner client-side)
    7. parse manifest_json → InstructionManifest
    8. consume each partial.payload through DP-budget-bounded
       combination via the manifest's aggregate ops:
         - COUNT/SUM: integer-add per group
         - AVERAGE: sum + count tracked separately, divided post-combine
         - SORT/LIMIT: top-k merge of per-shard sorted outputs
         - GROUP_BY: per-key aggregation
       (re-uses prsm/security/privacy_budget.py for budget ledger)
    9. plaintext = combined output (canonical JSON encoding)
   10. result_digest = sha256(plaintext)
   11. commit = AggregationCommit(query_id, our_pubkey_hash, result_digest)
   12. commit_signature = ed25519_sign(commit.signing_payload(), our_privkey)
   13. encrypted_plaintext = x25519_chacha20_seal(plaintext, prompter_pubkey, nonce)
   14. return AggregateResponse(...)
```

Failure modes (return `StageError` per existing pattern):
  - `MALFORMED_REQUEST` — bad signature or missing fields
  - `UNSUPPORTED_VERSION` — protocol_version mismatch
  - `INVALID_TOKEN` — prompter_signature doesn't verify
  - `EXPIRED_DEADLINE` (NEW code) — deadline_unix passed
  - `DP_NOISE_MARKER_MISSING` (NEW code) — partial without A5 marker
  - `PRIVACY_BUDGET_EXHAUSTED` (NEW code) — combination would exceed
    epsilon ceiling
  - `INVALID_PARTIAL_SIGNATURE` (NEW code) — source agent's sig fails

## Client-side flow

`AggregatorClient.aggregate(...)` adapter:

```
1. construct AggregateRequest from QueryOrchestrator's call args:
     - request_id = sha256(query_id || attempt_index || nonce16)
     - manifest_json = manifest.to_json()
     - partials = [SignedPartial(...)] from swarm_runner's PartialResult list
     - prompter_pubkey = node identity pubkey
     - prompter_node_id, beacon_used, aggregator_pubkey_hash from spec
     - ftns_budget = orchestrator's per-query budget allocation
     - deadline_unix = now + timeout (default 60s)
2. sign with prompter privkey
3. wire-encode (existing canonical encoding from
   chain_rpc/protocol.py — same as RunLayerSliceRequest)
4. send via Phase 6 transport (existing — TLS to aggregator's
   listener)
5. receive AggregateResponse
6. verify commit_signature over commit.signing_payload()
7. decrypt encrypted_plaintext with our privkey + their pubkey + nonce
8. plaintext_digest = sha256(plaintext)
9. assert plaintext_digest == commit.result_digest
   (A9 — orchestrator's verify_aggregation_commit re-checks this
   downstream, but client-side sanity is cheaper than passing a
   bogus tuple all the way to swarm_runner)
10. return (plaintext, commit) → swarm_runner consumes
```

## Threat-model coverage

| Adversary | Mitigation in this RPC |
|---|---|
| A1 collusion | aggregator_pubkey_hash bound in signing payload — A8 identity |
| A2 self-selection | prompter_node_id surfaced; selector enforces upstream |
| A3 unbondDelay | aggregator slash window — StakeBond contract; unrelated to wire format |
| A4 retries | Each retry generates a fresh request_id; deadline_unix bounds reattempts |
| A5 DP-noise | `dp_noise_applied` per-partial marker; server verifies before combination |
| A6 commit-reveal | beacon_used in signing payload — forensic |
| A7 governance denylist | enforced upstream at selector; no wire-format change |
| A8 pubkey-hash identity | aggregator_pubkey_hash bound; aggregator's reply pubkey resolves to same hash |
| A9 commit-before-reveal | response shape — commit signed BEFORE plaintext encrypt; client re-verifies digest |
| A10 constant-time selection | upstream concern; no wire-format binding |

## Privacy-budget integration

Each `SignedPartial` carries `privacy_budget_consumed: float` — the
epsilon the source agent spent applying DP noise. The server's
combination logic adds these (post-Laplace-DP composition: epsilon
sums) and refuses to emit a response if the sum exceeds the
manifest's per-query epsilon ceiling.

The existing `prsm/security/privacy_budget.py` provides the ledger
shape — server wires it identically to the inference path.

## Implementation tasks (B3 + B4 + B5)

### B3.1 — Wire format
- [ ] Add `AGGREGATE_REQUEST` + `AGGREGATE_RESPONSE` to
      `ChainRpcMessageType`
- [ ] `AggregateRequest` + `AggregateResponse` + `SignedPartial`
      dataclasses in `prsm/compute/chain_rpc/protocol.py`
- [ ] `signing_payload()` methods + canonical encoding/decoding
- [ ] 4 new `StageErrorCode` values
- [ ] Unit tests: roundtrip encode/decode, signing payload golden
      vectors, version-mismatch handling

### B3.2 — Server handler
- [ ] `AggregateServer` class in `prsm/compute/chain_rpc/server.py`
      (or sibling file) — receives requests, runs combination,
      signs commit, returns response
- [ ] Combination logic per AgentOp (re-use existing Phase 3.x.7
      patterns for per-stage execution)
- [ ] Privacy-budget ledger integration
- [ ] Unit tests: each AgentOp's combination correctness, A5 marker
      enforcement, signature failures, budget exhaustion

### B4 — SwarmDispatcher adapter
- [ ] Around `prsm/compute/agents/dispatcher.py::AgentDispatcher`
- [ ] Constructs `MobileAgent` per shard with manifest + ftns_budget
- [ ] Awaits per-shard results via `wait_for_result`
- [ ] Converts to `PartialResult` (with `dp_noise_applied=True`
      marker — source agent applies DP via existing `dp_noise.py`
      before signing)
- [ ] Returns `list[PartialResult]` to swarm_runner

### B5 — AggregatorClient adapter
- [ ] Wraps the AggregateRequest construction + transport send +
      AggregateResponse decode
- [ ] Returns `(plaintext, AggregationCommit)` to swarm_runner
- [ ] Re-verifies digest client-side as defense-in-depth

## Estimated effort

| Task | Estimate |
|---|---|
| B3.1 wire format | 1–1.5 days |
| B3.2 server handler | 2.5–3.5 days (combination logic per AgentOp + privacy budget) |
| B4 SwarmDispatcher adapter | 1.5–2 days |
| B5 AggregatorClient adapter | 1 day |
| 3-node E2E test | 0.5–1 day |
| **Total** | **6.5–9 days** (~1.5 weeks) |

Combine with B7 (node.py:1277 wiring, ~half-day) + B8 (MCP unhide,
~few hours) for the complete mainnet path.

## Open questions for follow-on

1. **Aggregator-side TEE attestation receipt.** Should the
   `AggregateResponse` include an attestation quote for Tier C
   queries (mirrors Phase 3.x.7 Task 5)? Adds ~200 bytes per response
   but matches the inference path. Recommend **yes** for symmetry.

2. **Streaming aggregation.** For very large per-shard outputs (>1MB),
   should partials chunk like `ActivationChunk` from Phase 3.x.7.1?
   For v1: hard 1MB-per-partial cap, no streaming. Revisit at TVL > $1M.

3. **Re-aggregation requests.** A4 retry sends a fresh `request_id`
   but the same `query_id` to (potentially) a new aggregator. Should
   the aggregator have visibility into `attempt_index` (e.g., for
   logging)? Recommend **no** — leak surface for nothing.

4. **Aggregator-side privacy budget ledger persistence.** Cross-query
   composition tracks accumulated epsilon per prompter. Should the
   ledger persist across server restarts? Phase 3.x.4
   (`PersistentPrivacyBudgetTracker`) already provides this — wire it.

## References

- `prsm/compute/chain_rpc/protocol.py:113-125` — message-type enum
- `prsm/compute/chain_rpc/protocol.py:587` — RunLayerSliceRequest pattern
- `prsm/compute/inference/inference_receipt.py` — Phase 3.x.1 signing payload pattern
- `prsm/security/privacy_budget.py` — DP-budget ledger
- `docs/2026-05-07-aggregator-selector-threat-model.md` — A1–A10 catalog
- `docs/2026-05-08-query-orchestrator-wiring-readiness.md` — B3 placement in the
  larger wiring program
