# PRSM Glossary — Disambiguation Reference

> **Scope:** This is not a dictionary. It is a disambiguation reference for terms that are ambiguous, reused, or have meanings that have shifted during PRSM's architectural evolution. If a term has a single unambiguous meaning, it is not here — read the Vision doc Appendix C for one-liner definitions of those.
>
> **Audience:** engineers, contributors, subagents writing plans. Prefer fully-qualified forms from this doc in code comments, commit messages, plan documents, and review feedback.
>
> **Maintenance:** add an entry only when a naming collision or ambiguity has caused actual re-reading or confusion. Grow on demand, not prophylactically.

---

## The three Tier systems (high-collision hazard)

PRSM uses the word "Tier" for **three structurally independent classification schemes**. They are orthogonal — a single node can be characterized on all three axes simultaneously — but because the labels overlap, readers who encounter "Tier B" in one doc and "Tier B" in another will misread unless they know which system is being referenced.

| System | Label set | What it classifies | Where used |
|---|---|---|---|
| **Hardware supply tiers** | T1 / T2 / T3 / T4 | Node hardware capability and supply-side role | `PRSM_Vision.md` §6, `Prismatica_Vision.md` |
| **Compute verification tiers** | Tier A / Tier B / Tier C | Verification-strength spectrum for remote compute dispatch | Phase 2 plan, `prsm/compute/shard_receipt.py`, roadmap Phase 7 |
| **Content confidentiality tiers** | Tier A / Tier B / Tier C | Confidentiality-strength spectrum for stored content | `PRSM_Vision.md` §2 and §7, roadmap Phase 7 |

**Rule when writing:** use fully-qualified forms — "hardware supply tier T3," "compute verification Tier A," "content confidentiality Tier B" — unless the surrounding paragraph has already established context within the last few lines.

**Rule when reading:** if you encounter a bare "Tier X" in a doc you opened cold, scroll up to find which of the three systems the doc is using. Do not assume.

### Hardware supply tiers (T1-T4)

Classification of nodes by compute capability and supply-side economic role. All nodes on the network fall into exactly one supply tier.

- **T1 — Consumer edge.** Smartphones, laptops, consumer NPUs. Low-latency retrieval, tiny jobs at massive scale, edge locality. Cannot handle heavy jobs; high churn.
- **T2 — Prosumer edge.** Gaming PCs, PS5/Xbox consoles, Mac Studios. Medium jobs, edge locality, consistent uptime. Sustained ~1-2 TFLOPS per node.
- **T3 — Professional arbitrage.** Operators running PRSM nodes on rented cloud GPUs (RunPod, Lambda, CoreWeave). Elastic heavy compute, H100-class, spins up on demand. Subject to cloud provider TOS, currency risk.
- **T4 — Meganodes.** Prismatica and other strategic operators running dedicated infrastructure. Guaranteed baseline, SLA commitments, TEE-attested private-inference tier.

### Compute verification tiers (A/B/C)

Verification-strength spectrum applied to remote compute dispatch. Selected per-inference based on the workload's trust requirements.

- **Compute verification Tier A — Receipt-only.** Provider signs an Ed25519 receipt attesting to the output hash; requester verifies the signature. Current Phase 2 scope. Cheap; trusts the provider modulo signature.
- **Compute verification Tier B — Redundant execution consensus.** Multiple providers execute the same shard; outputs compared for consensus. Deferred to Phase 7. Expensive; protects against unilateral provider cheating.
- **Compute verification Tier C — Stake-slash verification.** Providers stake FTNS; misbehavior slashed via on-chain `ComputeSlashing.sol`. Deferred to Phase 7. Strongest; requires the slashing contract to ship.

### Content confidentiality tiers (A/B/C)

Confidentiality-strength spectrum applied to stored content. Selected by the publisher at upload time.

- **Content confidentiality Tier A — Public content.** BitTorrent-native sharding; pieces contain plaintext byte ranges. Current Phase 1 scope. Integrity and provenance enforced; confidentiality is not a goal. Appropriate for commons data.
- **Content confidentiality Tier B — Encryption-before-sharding.** Publisher encrypts the file (AES-256-GCM) before sharding. Shards contain ciphertext only. Key released via on-chain key-distribution contract triggered by verified royalty payment. Deferred to Phase 7.
- **Content confidentiality Tier C — Zero-knowledge content.** Encryption + Reed-Solomon erasure coding (K-of-N) + Shamir-split decryption keys (M-of-N). Reconstruction requires crossing both thresholds. Deferred to Phase 7.

### Why the two A/B/C tier systems are orthogonal (not aliased)

Phase 7 delivers both compute verification Tier B/C **and** content confidentiality Tier B/C simultaneously because both depend on similar cryptographic and consensus infrastructure (stake management, on-chain proofs, erasure coding). **They do not share code paths.** They are independent subsystems co-located in the same phase delivery for scheduling convenience, not logical coupling. A content-confidentiality-Tier-B upload served by a compute-verification-Tier-A inference is a valid, sensible configuration.

---

## Ring vs Phase (numbering epochs)

Two numbering schemes exist in the codebase and docs. Both are correct; they apply to different eras.

- **Rings 1-10** — pre-v1.6 organization (April 2026 and earlier). Ring plans archived at [`archive/`](./archive/) (see `archive/README.md`). Rings 7-10 shipped as confidential compute in v0.35.0. Ring 8's tensor-parallel sharding shipped for single-machine execution; its remote-dispatch completion is current Phase 2.
- **Phases 1-8** — post-v1.6 organization (April 2026+). Master roadmap at [`2026-04-10-audit-gap-roadmap.md`](./2026-04-10-audit-gap-roadmap.md). Current authoritative numbering.

**Rule:** new work uses Phase numbering exclusively. Referencing "Ring 8" is valid when describing earlier shipped work (e.g., Phase 2 docs refer to "Ring 8 tensor-parallel sharding" because that's what shipped earlier and what Phase 2 completes). Do not invent new Ring numbers.

---

## Prismatica vs Foundation (two-entity structure)

PRSM involves two legally distinct entities that are routinely conflated:

- **The PRSM Foundation.** Nonprofit (501(c)(3) or offshore equivalent). Protocol steward. Holds the 100M genesis FTNS allocation and distributes it as compensation. Operates POL reserve (discretionary, not a peg). Neutral with respect to all protocol participants. Does not sell FTNS to anyone.
- **Prismatica.** For-profit operating company (Delaware C-corp or PBC). First-class network participant that builds and operates T4 meganodes, curates commons data, commissions proprietary datasets, trains domain models, and runs a venture fund. Raises bootstrap capital via Reg D 506(c) equity to accredited investors. Pays dividends to the Foundation per the Foundation's equity stake (proposed 15-25%, targeting 25%).

**Alignment:** Foundation's equity stake + Prismatica's on-chain royalty flow on commons and commissioned data. **Separation:** distinct legal entities, distinct boards (≤1 overlapping liaison), arms-length transactions, no protocol-level favoritism for Prismatica content (enforced at smart-contract level).

**Rule when writing:** when referring to "who does X," name the specific entity. "The Foundation sets halving rates during the operational period." "Prismatica accumulates FTNS by operating T4 nodes." Never say "PRSM does X" when the subject is one of the two entities specifically.

See: `Prismatica_Vision.md` §1 (case for separate entity), `PRSM_Tokenomics.md` §11 (hybrid revenue model).

---

## Compensation-only distribution vs bonding curve (pivot flag)

FTNS is currently distributed **only as compensation** for services rendered to the network (creator royalties, node operator compensation, contributor grants, foundation operational distributions). The Foundation does not sell FTNS to any party in exchange for other currency.

Earlier drafts of the tokenomics described a **bonding-curve sale** where investors purchased FTNS directly from a curve with explicit appreciation expectations. **This has been superseded.** Bootstrap capital is now raised via Prismatica equity instead; FTNS is never sold by the Foundation.

**Rule when reading archived docs:** if you encounter any of these in `docs/archive/` or older versions of `PRSM_Vision.md` / `PRSM_Tokenomics.md`, treat as legacy:
- "bonding curve" in the context of FTNS sale
- "SAFT" structure for FTNS pre-sale
- "ICO" or "public token sale event"
- "investor FTNS compensation" or similar

Current posture: `PRSM_Tokenomics.md` §3 (distribution model), §9 (Howey analysis). Regulatory risk reduced from ~20-40% (bonding-curve era) to ~5-10% (compensation-only era) for FTNS-as-security determination.

---

## Acronyms

Terms redefined on first use in almost every doc. Single authoritative definition here.

| Term | Full form | Quick disambiguation |
|---|---|---|
| **FTNS** | "Photons" | ERC-20 utility token on Base. The economic medium of PRSM. Pronounced "photons." |
| **SPRK** | Sharded Processing & Remote Kernel | ~50KB WASM module executed in Wasmtime. Used for (a) move-code-to-data analysis jobs and (b) Megatron-style sharded inference activation-streaming stages. Pronounced "spark." |
| **PCU** | PRSM Compute Unit | Composite billing unit: `(tflops × seconds) + (memory_gb × seconds) + egress_mb`. Captures compute, memory, bandwidth in one figure. |
| **POL** | Protocol-Owned Liquidity | Foundation-held USDC + FTNS reserve deployed defensively during severe market stress. **Discretionary, not a peg.** Governed by §3.6 of `PRSM_Tokenomics.md`. |
| **MCP** | Model Context Protocol | Standard for LLM tool invocation (`modelcontextprotocol.io`). PRSM's Phase 3 exposes its gateway as an MCP server so Claude Desktop / ChatGPT Desktop / Gemini / etc. can invoke PRSM without users switching interfaces. |
| **TEE** | Trusted Execution Environment | Hardware-isolated enclave (Intel SGX / TDX, AMD SEV-SNP, ARM TrustZone, Apple Secure Enclave). Security boundary, **not a sealed vault** — see `PRSM_Vision.md` §7 honest limits. |
| **PoR** | Proof of Retrievability | Challenge-response protocol proving a storage node actually holds the data it claims to seed. Phase 7 storage hardening. |
| **Base** | Ethereum L2 by Coinbase | Host chain for FTNS and PRSM's on-chain contracts. Chain ID 8453. PRSM inherits Base's security, which inherits Ethereum's. |
| **Reg D 506(c)** | SEC exemption | Permits accredited-investor-only private placement of securities. The exemption Prismatica equity is raised under. |
| **Aerodrome** | Base-native DEX | Decentralized exchange where organic secondary FTNS liquidity may emerge. Foundation does not seed or operate pools on Aerodrome. |
| **Fuel metering** | Wasmtime feature | Deterministic instruction counter. Replaces wall-clock timeouts for sandbox resource enforcement. Per-module fuel budget (default 30B units). |

---

## Related references

- `PRSM_Vision.md` Appendix C — one-liner definitions of every term used in the Vision doc (broader scope than this glossary; less disambiguation depth)
- `docs/2026-04-10-audit-gap-roadmap.md` — master execution roadmap; uses terms from this glossary
- `docs/archive/README.md` — explains what's in the archive directory and the superseded status of pre-pivot artifacts
