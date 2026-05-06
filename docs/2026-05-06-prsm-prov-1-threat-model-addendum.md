# PRSM-PROV-1 Threat Model Addendum (§3.16)

**Date:** 2026-05-06
**Track:** PRSM-PROV-1 Content Provenance Correctness (Items 3–7 from the
2026-05-05 embedding audit)
**Cross-references:**
- Plan: [`docs/2026-05-06-content-provenance-correctness-plan.md`](2026-05-06-content-provenance-correctness-plan.md)
- Companion §3.16 in the master threat model surface
- Audit-prep §7.16 (paired with this doc)

This addendum captures the threat model for Item 3 (EmbeddingDHT) — the
cross-node embedding gossip layer that closes the "Alice@A and Bob@B
both register near-duplicate as original" gap. Items 4/6/7 will get
their own addenda when those tasks ship.

---

## §3.16.1 Trust boundaries

The EmbeddingDHT introduces a **new trust boundary** between:

- **Producer node** (the original creator of a piece of content). Signs
  the embedding with its Ed25519 private key. Has authority over what
  vector represents its content.
- **Relay node** (any peer that received the signed record via DHT
  replication or pull-through caching). Stores and serves the record
  verbatim. Has NO authority — the signature it relays cannot be forged.
- **Consumer node** (a peer running dedup against an in-progress
  upload). Pulls candidate records via the DHT, verifies each against
  the on-chain anchor, only feeds verified vectors into its
  `_SemanticIndex`.

The on-chain anchor (`PublisherKeyAnchor`, Phase 3.x.3) is the root of
trust. Without an anchor entry for a creator, no peer can verify their
signature, and the consumer rejects.

## §3.16.2 Adversaries in scope

| ID | Adversary | Capability |
|---|---|---|
| A1 | **Embedding poisoner** | Operates a peer node; serves vectors that don't match the content they claim to represent (random direction, semantic decoy). |
| A2 | **Eclipse / sybil cluster** | Operates multiple peer nodes; surrounds the keyspace for a target `content_hash` so all queries route to attacker-controlled servers. |
| A3 | **Network passive** | Observes DHT traffic but cannot modify it. Goal: extract content from embeddings ("vector inversion"). |
| A4 | **Compromised legitimate node** | Has lost its Ed25519 private key to the attacker, but the public key remains anchored on-chain. |
| A5 | **Storage exhaustion adversary** | Floods peers with bogus `LocalEmbeddingRecord` payloads to fill their local index. |

Out of scope (handled by other tracks):
- On-chain anchor compromise (Phase 3.x.3 §3.x territory).
- Content-store provider impersonation (Phase 4 / IPFS surface).
- Solidity-level governance attacks on `ProvenanceRegistry` (Item 7
  scope when it lands).

## §3.16.3 Attack mitigations

### A1 — Embedding poisoning

**Vector:** Malicious peer responds to `FETCH_EMBEDDING` with a
crafted vector that doesn't represent the content. Bob's
`_SemanticIndex` ingests it, dedup misses Alice's near-duplicate, both
register as originals, royalty math splits incorrectly.

**Mitigation:** `EmbeddingResponse.signature_b64` covers the canonical
payload `(content_hash, model_id, dimension, dtype, vector_bytes,
created_at)` (length-prefixed, domain-tagged with
`PRSM-PROV-1/EmbeddingResponse/v1`). The signature is by the
**creator** of the content — not the serving node. A relay cannot forge
this. The consumer (`EmbeddingDHTClient._verify_creator_signature`)
fetches the creator's pubkey via the on-chain anchor and verifies
before returning the vector. **Verifier is mandatory** —
`EmbeddingDHTClient` constructor raises `RuntimeError` if either
`creator_pubkey_for` or `verify_signature` is None. There is no
"trust the network" mode.

**Unit-test coverage** (`tests/unit/test_embedding_dht_server_client.py`,
4 poisoning scenarios):
- Tampered vector with original signature → rejected.
- Response missing on-chain anchor entry → rejected.
- Response signed under wrong publisher pubkey → rejected.
- Server swaps `(content_hash, model_id)` to mis-key the cache →
  rejected (anti-swap check at `dht_client.py:251-261`).

**Residual risk:** A creator who poisons their **own** content's
embedding can defeat dedup. This is captured at the policy layer:
poisoning your own embedding is no worse than refusing to embed at
all, and the on-chain provenance trail still binds the (incorrect)
record to the creator. Item 7 (on-chain embedding commitment) makes
this auditable post-hoc.

### A2 — Eclipse / sybil cluster

**Vector:** Adversary operates N peer nodes whose `node_id`s cluster
around the keyspace for a target `content_hash`. Honest queries route
exclusively to attacker peers, who serve a uniformly poisoned
response. Single-peer signature verification doesn't help — every
response is signature-INVALID, the consumer falls through with no
result, dedup silently fails open ("no near-duplicate found").

**Mitigation:**
- `EmbeddingDHTClient.find_providers` deduplicates providers across
  multiple closest-peer queries (`dht_client.py:159-192`), so a sybil
  cluster contributing duplicate `node_id`s gets collapsed.
- `_SemanticIndex._fetch_one_embedding` (T3.5) tries providers in
  order and breaks on the first signature-verified response, so an
  honest provider's record wins even if the sybil pool is larger.
- Consumer fall-through-on-failure: when ALL providers return
  poisoned responses, dedup gracefully degrades to "no match found"
  rather than accepting a bad vector. The upload still succeeds; the
  dedup miss is a false-negative, not a false-positive. This is the
  intended degradation mode — the alternative (block uploads on DHT
  failure) is worse for liveness.

**Residual risk:** Sustained eclipse against a specific `content_hash`
is not detected at the protocol layer. **Detection is reputational
and offline** — Phase 7 ReputationTracker can flag peers that
repeatedly produce signature-invalid records. The PRSM-PROV-1
in-protocol mitigation accepts that an eclipsed query degrades to
local-only dedup for that query; the broader trust system catches
the attacker over time.

### A3 — Privacy / vector inversion

**Vector:** A passive observer sees an `EmbeddingResponse` for some
`content_hash` they don't have access to. They run a vector-inversion
attack to reconstruct (some of) the original content's semantics.

**Mitigation:** Embedding gossip is **opt-in per content tier**:
- **Tier A (public)**: gossip enabled by default. The embedding
  reveals no more than the content itself, which is already public.
- **Tier B/C (private)**: gossip OFF by default. Set
  `--embedding-gossip-tier-b=true` at node start to opt in. T3.6
  wiring respects this — `_register_local_embedding` skips when no
  on-chain `provenance_hash` is available, which is the natural gate
  for tier-restricted content (no on-chain anchor → no public dedup).

**Residual risk:** State-of-the-art vector-inversion attacks (e.g.,
*Vec2Text*, NeurIPS 2023) recover 30–50% of content from public
embeddings of common models. For Tier A content this is acceptable
because the content is already public. For any node operator who
opts Tier B/C content into gossip, the policy memo
([`docs/governance/embedding-gossip-tier-policy.md`] — to be drafted
if/when Tier B/C gossip is enabled) flags this as the operator's
informed risk acceptance. **Not enabled by default and not
recommended.**

### A4 — Compromised legitimate creator

**Vector:** Attacker has stolen creator C's private key. They publish
a signed embedding for content_hash X that they didn't create. Honest
peers verify the signature successfully against C's still-anchored
public key.

**Mitigation:** This is fundamentally an on-chain key-rotation
problem, not an embedding-DHT problem. The path forward is the
existing `PublisherKeyAnchor.rotate(...)` flow (Phase 3.x.3) —
rotating C's key invalidates all signatures by the compromised key
because the verifier looks up the **current** anchored key.

The PRSM-PROV-1 layer also bounds the impact: a compromised key can
poison embeddings only for content already registered to C, since the
DHT key is `(content_hash, model_id)` and `content_hash` is the
on-chain identifier. The attacker cannot impersonate C for content
created by a different creator.

### A5 — Storage exhaustion

**Vector:** Attacker floods peer P with `EmbeddingResponse` records
for fabricated `content_hash` values, attempting to fill P's
`LocalEmbeddingIndex` with ten million junk entries.

**Mitigation:**
- `LocalEmbeddingIndex.register` is server-controlled, not
  network-driven. P only registers records when:
  1. P uploaded the content itself (T3.6 path) — bounded by P's
     own upload rate, capped by IPFS storage gates.
  2. P pulled the record via DHT in T3.5's `_pull_remote_embeddings`
     — bounded by `max_remote_pulls_per_query` (default 32 per
     `find_nearest()` call) and gated on `peer_candidates_fn`
     yielding only records present in the gossip-driven
     `ContentIndex._records` (also LRU-bounded to 10K entries by
     existing Phase 4 wiring).
- Records pulled via T3.5 only stick in P's index when their
  signature verifies against the on-chain anchor. Fabricated
  `content_hash` values have no anchor entry → rejected at the
  client → never reach the index.

**Residual risk:** Coordinated upload of many tiny pieces of legitimate
content against one peer remains possible (the `_local_content` table
in `ContentProvider` would also fill). That's covered by the existing
Phase 4 storage-incentive gate, not PRSM-PROV-1.

## §3.16.4 Cross-cutting invariants

These hold across all of A1–A5 and are tested in unit tests:

1. **Verifier-required client construction:** `EmbeddingDHTClient`
   refuses to construct without `(creator_pubkey_for,
   verify_signature)`. Tested at
   `tests/unit/test_embedding_dht_server_client.py::test_client_refuses_without_verifier`.
2. **Cross-model partition end-to-end:** wire-format keys by
   `(content_hash, model_id)`; `LocalEmbeddingIndex` keys by the same
   tuple; `_SemanticIndex.find_nearest` only escalates DHT queries
   under the local node's `model_id`. Tested at
   `tests/unit/test_semantic_index_dht_escalation.py::test_dht_call_includes_correct_model_id`.
3. **Local-side advertise gate:** `_register_local_embedding` skips
   when no `provenance_hash_hex` available — no on-chain anchor
   means no peer can verify; we MUST NOT advertise. Tested at
   `tests/unit/test_content_uploader_register_local_embedding.py::test_skips_when_no_provenance_hash`.
4. **Failure paths never raise:** every error in T3.5 escalation
   (peer_candidates_fn raises, find_providers raises, all providers
   poisoned, NOT_FOUND, malformed response) and every error in T3.6
   registration (disk full, malformed shape, registry import error)
   is logged and swallowed. Upload critical path is never blocked
   by DHT-layer failures.

## §3.16.5 Audit-prep §7.16 (paired)

Auditor walkthrough talking points (to be expanded into the
cumulative audit-prep doc at the next refresh):

- Wire format walkthrough (`prsm/network/embedding_dht/protocol.py`).
- Signing payload determinism + length-prefix injection-resistance
  (41 protocol unit tests, including
  `test_canonical_signing_payload_is_byte_deterministic` and
  `test_canonical_payload_resists_length_prefix_attack`).
- Real Ed25519 verifier tests (no mocks) — see
  `tests/unit/test_embedding_dht_server_client.py::_real_verify_signature`.
- 4 poisoning scenarios — concrete test cases against the consumer
  rejection path.
- Cross-model isolation tests — vectors from MiniLM (384-dim) and
  OpenAI (1536-dim) cannot collide because the keyspace partitions.
- Gate hierarchy: T3.5 escalation gated on (model_id + dht_client +
  peer_candidates_fn) all wired; T3.6 advertise gated on
  (embedding_index + model_id + provenance_hash) all present;
  EmbeddingDHTClient construction gated on (creator_pubkey_for +
  verify_signature) being callables.

## §3.16.6 Status

**Drafted:** 2026-05-06 alongside T3.6 commit `8d0fff42`.
**Auditor review:** deferred to L4 vendor engagement.
**Next addendum:** Item 4 (BinaryFingerprint) §3.17 when image/audio/
video/structural-data perceptual hashing lands.
