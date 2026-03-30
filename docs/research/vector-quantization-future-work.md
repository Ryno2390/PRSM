# Vector Quantization — Future Work

**Filed:** 2026-03-30  
**Status:** Deferred — relevant at scale, not current bottleneck  
**Flagged by:** Isaac (PRSM Manager agent)

---

## Background

Google Research recently published **TurboQuant** ([blog post](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)), along with two underlying papers:

- **QJL** (Zandieh et al., 2024) — 1-bit Quantized Johnson-Lindenstrauss transform for KV cache compression. Achieves >5x memory reduction with zero accuracy loss by applying a random Gaussian projection before sign-bit quantization — eliminating per-block normalization overhead entirely.
- **PolarQuant** (Han et al., 2025) — Converts KV embeddings to polar coordinates after random preconditioning. The preconditioning causes angles to concentrate into an analytically computable distribution, enabling 4.2x compression with best-in-class quality scores on long-context tasks.
- **TurboQuant** (Zandieh et al., 2025) — Combines MSE-optimal Lloyd-Max scalar quantization on rotated vectors with a 1-bit QJL residual pass. Achieves near-optimal information-theoretic distortion bounds (within ~2.7x of Shannon limit) at 2.5–3.5 bits/channel.

**Core insight shared across all three:** Apply a random rotation (Johnson-Lindenstrauss transform) to embedding vectors before quantization. This spreads information evenly across dimensions, eliminates the need for per-block normalization constants, and enables aggressive bit-width reduction while preserving inner product accuracy.

---

## Why This Doesn't Apply to PRSM Today

Our current `prsm/compute/nwtn/corpus/code_index.py` indexes ~3,000 Python symbols from the PRSM codebase into ChromaDB (~4MB at 384 dimensions). ChromaDB handles HNSW indexing efficiently at this scale. Quantization would add complexity with zero measurable benefit.

---

## Where This Becomes Highly Relevant at Scale

### 1. Academic Corpus Indexing (highest priority)
**Location:** `prsm/compute/nwtn/corpus/`  
When PRSM indexes millions of scientific papers for retrieval-augmented reasoning, the embedding store will grow to GB+ scale. TurboQuant's 4x compression would directly reduce storage costs and query latency at that volume. The online (data-oblivious) property of QJL/TurboQuant is essential here — corpus entries arrive as a stream and can't wait for offline codebook training.

### 2. NWTN Whiteboard at Long Context
**Location:** `prsm/compute/nwtn/whiteboard/`  
Long NWTN sessions accumulate many whiteboard entries that need semantic similarity retrieval. As session counts scale, embedding-based retrieval becomes a bottleneck. Quantized embeddings would allow more history to fit in memory without eviction.

### 3. FTNS Marketplace Matching
**Location:** `prsm/economy/marketplace/`  
Semantic similarity matching between compute providers and requesters could involve millions of vectors at production scale. Inner product preservation under quantization (QJL's primary guarantee) is exactly what marketplace recommendation needs.

---

## Recommended Implementation Path (when ready)

1. **Instrument first** — add latency/memory metrics to `CodebaseIndex` and `vector_db.py` to establish when quantization becomes necessary (suggested threshold: >100k vectors or >500ms query latency)
2. **Start with ChromaDB's built-in quantization** — ChromaDB supports scalar quantization natively; enable it before implementing custom solutions
3. **Custom QJL/TurboQuant** — implement as a preprocessing step on top of ChromaDB if built-in quantization is insufficient. The random rotation matrix `S` can be fixed per-collection and stored cheaply.
4. **Target:** 3-4 bits/dimension (vs current 32-bit floats) → ~8-10x storage reduction with <1% recall degradation

---

## References

- QJL paper: https://arxiv.org/abs/2406.03482
- PolarQuant paper: https://arxiv.org/abs/2502.02617
- TurboQuant paper: https://arxiv.org/abs/2504.19874
- Google Research blog: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
