# Content Provenance Correctness — Items 3-7 Completion Plan

**Date:** 2026-05-06
**Status:** Draft
**Predecessor work:** Items 1, 2, 5 shipped (sentence-transformers
local fallback; mock auto-fallback removed; chunked embedding for long
documents)
**Track ID:** PRSM-PROV-1
**Tag prefix:** `prov-1-task-N-merge-ready-{date}` per task; cumulative
audit-prep refresh after each item ships.

---

## 1. Goal

Move PRSM content-provenance dedup from "node-local cosmetic" to a
load-bearing network-level provenance signal. The Items 1/2/5 work
fixed silent-mock and long-document gaps; Items 3-7 fix the four
remaining correctness gaps documented in the 2026-05-05 embedding
audit:

| # | Gap | What breaks today |
|---|---|---|
| 3 | Embeddings are **node-local only** | Alice on Node A and Bob on Node B both register the same paper as original |
| 4 | **Binary content** is never embedded | Re-encoded JPEG or HDF5 dataset always looks "new" |
| 6 | Thresholds are **hard-coded class constants** | Code, prose, scientific-text similarity distributions get the same 0.92 / 0.99 cutoffs |
| 7 | Embeddings **never reach the chain** | Provenance disputes have no semantic on-chain record |

Items 3 and 7 are coupled (you want vectors flowing before committing
their hash). Item 4 is independent. Item 6 needs corpus data we don't
yet have — it's a calibration task, deferrable until testnet-deploy
T-series upload trail is meaningful.

**Sequencing:** Item 3 → Item 4 (in parallel after T3.1) → Item 7 (depends on Item 3) → Item 6 (calibration after corpus exists)

```
W0 ───────── W2 ───────── W4 ───────── W6 ───────── W8 ─────────
T3 (DHT) ───────────────►
                T4 (perceptual hashes) ─────────────►
                                T7 (on-chain) ────────────────►
                                                T6 (calibration) ──►
```

---

## 2. Item 3 — EmbeddingDHT (cross-node embedding gossip)

### 2.1 Why

`prsm/node/content_index.py:47` already gossips
`embedding_id: Optional[str]` (a label like `emb:<cid>`) but never the
vector itself. The `_SemanticIndex` in `prsm/node/content_uploader.py`
loads only embeddings from local uploads. So when:

1. Alice uploads `paper-v1.pdf` to Node A — it embeds, registers.
2. Bob uploads near-identical `paper-v1-corrected.pdf` to Node B —
   Node B's `_SemanticIndex` is empty for this content, dedup says
   "novel", Bob registers as a new original.

The royalty-split in `RoyaltyDistributor` then pays both creators full
rate, which is exactly the case the dedup feature exists to prevent.

### 2.2 Approach

Build an `EmbeddingDHT` modeled on the working
`prsm/network/manifest_dht/` pattern (Phase 3.x.5). The DHT stores
mappings of `content_hash` (already on-chain via `provenance_hash`) →
serialized embedding vector, keyed by the deterministic content hash
so any node can ask "do you have an embedding for content X?"

**Why DHT not gossip:** Vectors are larger than current
advertisement payloads (1536-dim float32 = 6.1 KB per OpenAI vector;
384-dim = 1.5 KB per MiniLM vector). Pushing every embedding through
gossip-flood would multiply network bandwidth by ~6-25× per
advertisement. DHT lets each vector sit at the closest peers only,
fetched on demand at dedup time.

**Wire format:** New protocol module
`prsm/network/embedding_dht/protocol.py` — copy of `manifest_dht`
protocol with new message types:

```python
class EmbeddingMessageType(str, Enum):
    FIND_EMBEDDING = "find_embedding"          # who has this content_hash?
    EMBEDDING_PROVIDERS = "embedding_providers"
    FETCH_EMBEDDING = "fetch_embedding"        # give me the vector
    EMBEDDING_RESPONSE = "embedding_response"
    ERROR = "error"

@dataclass(frozen=True)
class EmbeddingResponse:
    content_hash: str          # 0x-prefixed hex
    model_id: str              # "openai/text-embedding-ada-002" | "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int             # 1536 | 384
    dtype: str                 # "float32"
    vector_b64: str            # base64-encoded raw bytes (dim * 4 bytes for float32)
    creator_id: str            # for provenance attribution
    created_at: float
    signature_b64: str         # creator's Ed25519 sig over canonical payload
```

**Critical constraint — model_id matching:** Cosine similarity
between vectors from different models is meaningless. The DHT must
key by `(content_hash, model_id)`, not just `content_hash`. A node
running MiniLM cannot dedup against an OpenAI vector — it gets back
"no embedding under your model" and falls through to upload-as-new.

Long term, an alignment layer (Procrustes / linear projection
between embedding spaces calibrated on a shared corpus) could let
cross-model dedup work; that's deferred research, not in PRSM-PROV-1.

### 2.3 Tasks

| Task | Surface | Lines (est.) |
|---|---|---|
| T3.1 | Wire-format protocol — `prsm/network/embedding_dht/protocol.py` (5 message types + dataclasses + encode/parse + tests) | ~500 LOC + ~400 test LOC |
| T3.2 | `LocalEmbeddingIndex` (RAM-backed, persists to JSON like `_SemanticIndex` does today) — `prsm/network/embedding_dht/local_index.py` | ~300 LOC + ~250 test LOC |
| T3.3 | `EmbeddingDHTServer` — handle FIND_EMBEDDING / FETCH_EMBEDDING; uses Phase 6 transport | ~250 LOC + ~250 test LOC |
| T3.4 | `EmbeddingDHTClient` — `find_embedding(content_hash, model_id) → providers`, `fetch_embedding(provider, content_hash, model_id) → vector` | ~400 LOC + ~300 test LOC |
| T3.5 | Integrate into `_SemanticIndex.find_nearest()` — when local search misses or returns low-similarity, query DHT, mean-pool the local + remote candidates, re-test threshold | ~150 LOC in `content_uploader.py` + ~200 test LOC |
| T3.6 | Wire `EmbeddingDHT` into `node.py` startup alongside `ContentIndex`; advertise local embeddings on upload | ~80 LOC + ~150 test LOC |
| T3.7 | Module exports + smoke tests | ~50 LOC + ~80 test LOC |
| T3.8 | E2E integration test — 3 simulated nodes, Alice@A and Bob@B both upload near-duplicate paper, assert Bob's record has `near_dup_cid` set to Alice's CID | ~250 LOC |
| T3.9 | Threat model addendum + audit-prep §7.16 | ~100 LOC |
| T3.10 | Code review + merge-ready tag | — |

**Estimated total: ~1,980 source LOC + ~1,880 test LOC ≈ 12-16h focused work**

### 2.4 Threat model — what to call out

- **Embedding poisoning:** Malicious node serves a vector that's a
  random direction, not the true embedding of the content. Bob
  uploads paper, malicious peer responds with a random vector that
  doesn't match Alice's, so dedup misses. Mitigation: signature in
  `EmbeddingResponse` is over the (content_hash, model_id, vector
  bytes) tuple by the original *creator*, not the serving node. A
  node serving someone else's embedding cannot forge the signature.
  Verifier checks signature against the on-chain creator pubkey
  (existing `PublisherKeyAnchor` from Phase 3.x.3).
- **DHT eclipse / sybil:** Standard DHT attack — adversary
  surrounds the keyspace for `content_hash X` and serves only their
  poisoned response. Mitigation: DHT client queries N≥3 peers and
  rejects if signatures disagree. Echoed from Phase 3.x.5
  ManifestDHT threat model §3.
- **Privacy:** A vector leaks information about the content. For
  Tier B/C content, embedding gossip is opt-in (default off) — set
  `--embedding-gossip-tier-b=false` at node start. For Tier A
  (public) content, the embedding is no more sensitive than the
  content itself. New §3.16 addendum to `THREAT_MODEL.md`.

### 2.5 Acceptance criteria

- [ ] 3-node E2E: Alice uploads paper to Node A, Bob uploads
      near-duplicate to Node B. Bob's `ContentRecord.near_duplicate_of
      == Alice's CID` and `parent_cids` includes Alice's CID.
- [ ] Royalty math: when Charlie purchases Bob's near-duplicate, both
      Alice and Bob receive royalty (split per `royalty_rate`).
- [ ] Cross-model isolation: Node A running OpenAI and Node B running
      MiniLM both upload the same paper. Each registers as original
      under its own `model_id` keyspace. Documented as a known
      limitation, not a bug.
- [ ] Embedding-poisoning test: malicious node serves random vector;
      DHT client rejects unsigned/wrong-signed response; falls through
      to next provider.

---

## 3. Item 4 — Perceptual hashes for binary content

### 3.1 Why

`ContentUploader._get_embedding` does
`content.decode("utf-8", errors="ignore").strip()` then bails if
`len < 50`. So:

- A 50KB JPEG decodes to garbage UTF-8 (or empty), fails the
  50-char check, gets `embedding = None`.
- The provenance path then falls back to byte-exact
  `provenance_hash = keccak256(creator || sha3_256(file_bytes))`.
- Re-encoding the JPEG to PNG produces different bytes → different
  hash → registered as new original.
- Same problem for audio (re-encoded MP3), video (re-muxed MP4),
  scientific binary formats (HDF5, Parquet, Arrow, NumPy `.npy`).

A meaningful fraction of PRSM's intended content (datasets, images,
audio) cannot dedup at all today.

### 3.2 Approach

Add a `BinaryFingerprint` plug-in path that runs *before* text
embedding. Content-type detection via `python-magic` (already a
declared transitive dep via several upload paths) → dispatch to one
of:

| MIME type prefix | Library | Output |
|---|---|---|
| `image/*` | `imagehash` (pHash, dHash, wHash) | 64-bit perceptual hash |
| `audio/*` | `chromaprint` (acoustid fingerprint) | variable-length integer fingerprint |
| `video/*` | sample N=8 keyframes via `pyav` (ffmpeg-python wrapper) → image pHash on each → 8×64-bit array | multi-hash |
| `application/x-hdf5`, `application/octet-stream` (Parquet/Arrow/npy) | structural hash: dtype + shape + descriptor (column names for Parquet, dataset names for HDF5) + sampled-content hash | structural-hash |
| (anything else binary) | fall back to byte-hash (current behavior) | — |
| text-like | fall through to existing chunked text embedding | (Items 1+5) |

**These are not embedding vectors** — they're hashes with their own
distance functions (Hamming distance for pHash, edit-distance for
Chromaprint, structural-hash equality for binary scientific data).
The semantic index needs to grow:

```python
@dataclass
class FingerprintEntry:
    cid: str
    creator_id: str
    fingerprint_kind: str   # "text-vector" | "image-phash" | "audio-chromaprint" | "video-multihash" | "structural"
    fingerprint_payload: bytes
    threshold: float        # kind-specific (already exists for text, new for binary)
```

The dedup query becomes a per-kind dispatch:
- `text-vector` → cosine similarity (existing)
- `image-phash` → Hamming distance / 64 ≤ 0.10 (≈ 6 differing bits)
- `audio-chromaprint` → fingerprint-overlap ≥ 0.85
- `video-multihash` → quorum: ≥ 5 of 8 keyframe pHashes match within 6 bits
- `structural` → exact match (HDF5 datasets with same name + shape + dtype + content hash are duplicates)

### 3.3 Tasks

| Task | Surface | Lines (est.) |
|---|---|---|
| T4.1 | `prsm/data/fingerprints/__init__.py` + `BinaryFingerprint` ABC + content-type detection helper | ~150 LOC + ~100 test LOC |
| T4.2 | `ImageFingerprint` (pHash via `imagehash`) | ~100 LOC + ~150 test LOC (with real test images) |
| T4.3 | `AudioFingerprint` (Chromaprint via `pyacoustid`) | ~120 LOC + ~150 test LOC |
| T4.4 | `VideoFingerprint` (8-keyframe sampling) | ~180 LOC + ~200 test LOC |
| T4.5 | `StructuralFingerprint` (HDF5/Parquet/Arrow/npy) | ~200 LOC + ~200 test LOC |
| T4.6 | `_SemanticIndex` → `_FingerprintIndex`: per-kind storage, per-kind nearest-search | ~250 LOC in `content_uploader.py` + ~300 test LOC |
| T4.7 | Wire into upload path: detect content type → fingerprint → dedup-check → embed (text path only) | ~80 LOC + ~150 test LOC |
| T4.8 | E2E test: re-encoded JPEG dedup-matches; Parquet column-rename dedup-matches; mp3-bitrate-change dedup-matches | ~200 LOC |
| T4.9 | DHT integration — `BinaryFingerprint` flows through `EmbeddingDHT` (rename to `FingerprintDHT`?) with fingerprint_kind in the wire format | ~150 LOC + ~150 test LOC |
| T4.10 | Threat model + audit-prep §7.17 + code review tag | ~100 LOC |

**Estimated total: ~1,530 source LOC + ~1,400 test LOC ≈ 8-12h.**
**Dependency notes:**
- New deps to add: `imagehash`, `pyacoustid`, `pyav`. All BSD/LGPL.
  Audit-prep §7.1 third-party-derived-components needs an update.
- `pyacoustid` requires the system `chromaprint` library
  (`brew install chromaprint` / `apt install libchromaprint-tools`).
  Document in README.

### 3.4 Decision needed before T4.3

Chromaprint is the only mainstream open-source acoustic fingerprint;
the alternative is rolling our own MFCC-based fingerprint, which is
research-grade work, not 12h. Going with Chromaprint means accepting
a system-library dependency for audio dedup. **Default
recommendation:** ship with Chromaprint; if it's unavailable on the
host, audio falls back to byte-hash with a `logger.warning` at upload
time.

---

## 4. Item 7 — On-chain commitment to embeddings

### 4.1 Why

Today only `provenance_hash = keccak256(creator || sha3_256(file_bytes))`
lands on-chain. If two creators dispute provenance:

- Whoever called `ProvenanceRegistry.registerContent` first wins.
- There's no chain-level evidence of *what the content actually was*.
- Off-chain-only embeddings can be silently rotated by a malicious
  node operator without anyone noticing.

### 4.2 Approach

Extend `ProvenanceRegistry.sol` with an optional embedding commitment.
Two design options — pick one:

#### Option A: Per-content commitment (simple)

```solidity
struct ContentRecord {
    address creator;
    bytes32 contentHash;
    uint256 registeredAt;
    bytes32 embeddingCommitment;   // NEW — keccak256(model_id || dim || vector_bytes); zero if none
    bytes32 fingerprintKind;       // NEW — keccak256("text-vector" | "image-phash" | etc.); zero if byte-hash only
}
```

- Registration takes one extra `bytes32` arg (or zero).
- Dispute resolution: "show me the vector that hashes to
  `embeddingCommitment`" — challenger and defender each post a
  vector; on-chain only verifies the commitment match; off-chain
  arbitration (or future on-chain arbitration contract) decides
  which vector represents the content better.
- Storage cost: 64 bytes extra per registration (~$0.05 on Base
  mainnet at 0.001 gwei).

#### Option B: HNSW-root commitment (per-node)

- Nodes commit a Merkle/HNSW root of all their indexed embeddings
  periodically (e.g., every 100 uploads or every hour).
- Disputes verify a Merkle proof against the most recent root.
- Storage cost: ~$0.01 per root commitment, amortized over many
  uploads.
- Significantly more complex; deferred research per R7 KV
  compression scoping doc.

**Recommendation:** Ship **Option A** in PRSM-PROV-1. Option B is a
v2 optimization once node-operator volume justifies the engineering
cost.

### 4.3 Tasks

| Task | Surface | Lines (est.) |
|---|---|---|
| T7.1 | `ProvenanceRegistry.sol` extension — add `embeddingCommitment` + `fingerprintKind` to `ContentRecord` struct + `registerContent` signature; emit them in `ContentRegistered` event | ~80 LOC Solidity + ~150 LOC Hardhat tests |
| T7.2 | OZ upgrade-safety check: ensure storage layout extension is append-only (existing records stay valid) | tests in T7.1 |
| T7.3 | Foundation Safe → mainnet upgrade transaction (gated by L4 audit firm review of the upgrade) — **deferred until Phase 7+L4 ships** | runbook only |
| T7.4 | Python `ProvenanceClient.register_content_v2(...)` — adds the two new args; default to zero for byte-hash-only content | ~80 LOC + ~120 test LOC |
| T7.5 | Wire into `ContentUploader._register_on_chain` — pass `embedding_commitment = keccak256(model_id || dim_be || vector_bytes)` when an embedding exists | ~50 LOC + ~150 test LOC |
| T7.6 | Dispute-arbitration helper `dispute_provenance(cid, my_vector_bytes) → bool` — verifies caller's vector matches the on-chain commitment | ~100 LOC + ~150 test LOC |
| T7.7 | E2E test on Base Sepolia: register → query → dispute path verified | ~200 LOC |
| T7.8 | Threat model addendum §3.18 + audit-prep §7.18 | ~100 LOC |
| T7.9 | Code review + merge-ready tag | — |

**Estimated total: ~600 source LOC + ~870 test LOC ≈ 6-8h Python work + audit gate before mainnet upgrade**

### 4.4 Backwards compatibility

- Existing on-chain `ContentRecord`s have no `embeddingCommitment` —
  the field is zero, meaning "no commitment registered". Disputes
  fall back to byte-hash. No legacy data is invalidated.
- Off-chain `_SemanticIndex` keeps working without on-chain
  commitment if the content registration predated this upgrade.

### 4.5 Critical L7 gate

**Mainnet deploy of the `ProvenanceRegistry` upgrade is blocked
behind L4 audit firm review.** This is an immutable-contract upgrade
on a contract that is currently running mainnet at
`0xdF47...9915`. We cannot ship Option A to mainnet without:

1. Trail of Bits / Spearbit / OpenZeppelin sign-off on the
   storage-layout safety
2. Foundation council resolution authorizing the upgrade per
   PRSM-POL-1 §5 (this falls in the > $500K risk-reserve tier
   only if it would deplete reserves, which it won't — but we
   still publish the upgrade plan with 30-day notice per §5)

T7.1-T7.2 can ship to Sepolia immediately for testing. T7.3 is
operator-gated.

---

## 5. Item 6 — Per-content-type thresholds + arbitration

### 5.1 Why

`DERIVATIVE_THRESHOLD = 0.92` and `DUPLICATE_THRESHOLD = 0.99` are
class constants (`prsm/node/content_uploader.py:80-82` area).
Cosine-similarity distributions vary by content type:

- Code (especially Python): tighter distribution, 0.92 might match
  unrelated files just because they share boilerplate
- Scientific abstracts: cluster *very* tightly even for unrelated
  papers in the same field; 0.92 produces false-positive
  derivative-claims
- Multilingual text: variable, model-dependent
- Image pHash: completely different distance function, current
  thresholds don't apply at all

### 5.2 Approach

Two-layer system:

1. **Per-fingerprint-kind thresholds** (calibrated, in-code defaults):
   - `text-vector / openai-ada-002`: derivative=0.92, duplicate=0.99
     (current values; these were tuned for general text)
   - `text-vector / sentence-transformers-MiniLM`: derivative=0.85,
     duplicate=0.97 (MiniLM has slightly looser distribution)
   - `image-phash`: derivative=≤12 bits diff, duplicate=≤4 bits diff
   - `audio-chromaprint`: derivative=≥0.75, duplicate=≥0.92
   - `video-multihash`: derivative=≥5/8 keyframes match,
     duplicate=≥7/8 match
   - `structural`: duplicate-only (exact match), no derivative tier

2. **Per-content-type override** (advisory, registered at upload):
   - `metadata: {"content_type_hint": "scientific_abstract" | "code" | "prose" | ...}`
   - Maps to a multiplier on the kind-default (configured in YAML
     `prsm/data/dedup_thresholds.yaml`)
   - Override is *advisory* — can't tighten beyond the kind
     default's published floor (prevents griefing where a malicious
     uploader claims "code" to slip past dedup)

3. **Arbitration mechanism for disputed matches** (ratification gate):
   - When a dedup hit lands in `[derivative_floor, derivative_threshold)`
     band, the upload completes but is flagged for **community
     review** rather than auto-attributing.
   - Review surface: the existing
     `prsm/community/governance/proposal.py` voting layer (Phase 6
     governance) — content owners + flagged-uploader argue, council
     adjudicates.
   - This is the foundation for the future on-chain arbitration
     contract.

### 5.3 Tasks

| Task | Surface | Lines (est.) |
|---|---|---|
| T6.1 | `prsm/data/dedup_thresholds.yaml` — kind defaults + content-type-hint multipliers + floors | ~50 LOC YAML |
| T6.2 | `ThresholdResolver` class — load YAML, compute effective threshold given (kind, content_type_hint) | ~150 LOC + ~200 test LOC |
| T6.3 | Wire into `_FingerprintIndex.find_nearest()` — replace hardcoded constants | ~80 LOC in `content_uploader.py` + ~150 test LOC |
| T6.4 | Calibration corpus — 10K samples per kind from public datasets, similarity histogram + threshold-sweep ROC | ~300 LOC offline analysis script + dedup_thresholds.yaml updates |
| T6.5 | Disputed-band flag → arbitration queue (uses Phase 6 governance) | ~200 LOC + ~250 test LOC |
| T6.6 | E2E test — upload code that's 0.93-similar to existing code; assert it lands in arbitration queue, not auto-attributed | ~150 LOC |
| T6.7 | Audit-prep §7.19 + code review tag | ~100 LOC |

**Estimated total: ~830 source LOC + ~1,150 test LOC ≈ 6-8h, but
gated on T6.4 corpus access and at least 30 days of testnet upload
trail to validate the thresholds against real PRSM traffic.**

### 5.4 Why this is last

- Item 6 is calibration, not infrastructure. Without items 3 and 4
  shipped, there's no embedding/fingerprint network to calibrate
  against.
- The most valuable calibration data is *PRSM uploads*, not generic
  corpora. So we need testnet T-series upload traffic flowing
  before T6.4 produces useful threshold values.
- Until then, the kind-defaults from §5.2.1 are reasonable
  conservative starting points that don't actively harm dedup quality.

---

## 6. Cross-cutting work

### 6.1 Audit-prep document

After each item lands, refresh `cumulative-audit-prep-N` with:
- §7.16 EmbeddingDHT (after Item 3)
- §7.17 BinaryFingerprint (after Item 4)
- §7.18 On-chain embedding commitment (after Item 7)
- §7.19 Per-content-type thresholds (after Item 6)

### 6.2 Threat model

`THREAT_MODEL.md` gets four new sections:
- §3.16 EmbeddingDHT — poisoning, eclipse, privacy leak
- §3.17 Binary fingerprints — adversarial perturbations (image
  pHash is robust to compression but not to crafted noise; document
  the bound)
- §3.18 On-chain embedding commitment — dispute-resolution UX
- §3.19 Per-content-type thresholds — griefing prevention

### 6.3 Investor materials

Sprint summary at end of Items 3 + 4 (after the user-visible
correctness fix lands): "PRSM dedup now works across the network and
across content types — same content fingerprinted on Node A is
recognized at Node B; same image re-encoded as PNG matches its JPEG
ancestor."

### 6.4 Testing methodology

Per `CLAUDE.md` and `CLAUDE.local.md`: real corpora, no mocked
fingerprint libraries. Tests download the small open
`imagenet-mini` (~1GB) and the
`librispeech-test-clean` (~350MB) on first run, cached in
`~/.prsm/test-corpora/`. Audio dedup tests use real Chromaprint via
`pyacoustid`; image dedup tests use real `imagehash`. **No mock-and-pretend.**

---

## 7. Out of scope (deferred research)

The following came up while drafting this plan but are not in
PRSM-PROV-1:

- **Cross-model embedding alignment** (Procrustes / linear projection
  between OpenAI ada-002 and MiniLM spaces) — open research; punt to
  R10
- **HNSW root commitment on-chain** (Item 7 Option B) — needs node
  volume to justify; punt to R11
- **Self-supervised PRSM-native embedding model** trained on PRSM
  corpus — needs corpus first; punt to R12
- **Watermark detection** (DALL-E / Stable Diffusion image watermark
  recognition for AI-generated content provenance) — separate
  privacy/policy track; punt to PRSM-POLICY-WATERMARK-1
- **Encrypted embedding gossip** (homomorphic similarity computation
  so Tier B/C content can dedup without revealing the vector) —
  R1 FHE scoping doc covers this; not blocked on PRSM-PROV-1

---

## 8. Estimated total effort

| Item | Source LOC | Test LOC | Focused-work hours | Calendar |
|---|---|---|---|---|
| 3 EmbeddingDHT | ~1,980 | ~1,880 | 12-16h | W0-W2 |
| 4 BinaryFingerprint | ~1,530 | ~1,400 | 8-12h | W2-W4 |
| 7 On-chain commitment | ~600 | ~870 | 6-8h Python + audit gate | W4-W6 |
| 6 Per-kind thresholds | ~830 | ~1,150 | 6-8h + 30d corpus wait | W6-W8 |
| **Total** | **~4,940** | **~5,300** | **32-44h focused work** | **~8 weeks calendar** |

---

## 9. Decision points before kickoff

1. **Item 3 — model_id keyspace lock-in:** confirm we're explicit
   that "OpenAI vector ≠ MiniLM vector" is a hard partition. The
   alternative (eager alignment) is research-grade and would push
   Item 3 from 12-16h to multi-week. **Recommended: lock in.**

2. **Item 4 — Chromaprint dependency:** are we OK with a system-lib
   dep (`brew install chromaprint`) for audio dedup, with byte-hash
   fallback when missing? **Recommended: yes.**

3. **Item 7 — Option A vs Option B:** ship the simple per-content
   commitment now, defer HNSW-root commitment. **Recommended: A.**

4. **Item 6 — calibration timing:** we wait 30+ days of testnet
   upload traffic before T6.4 produces final thresholds. Use
   conservative kind-defaults until then. **Recommended: yes.**

If you ratify these four decisions I'll start on Item 3 Task 1
(wire-format protocol).
