# PRSM-PROV-1 Threat Model Addendum — §3.18 (Item 7)

**Track:** PRSM-PROV-1 Item 7 (On-chain embedding commitment).
**Surface:** `contracts/contracts/ProvenanceRegistryV2.sol` +
`prsm/economy/web3/provenance_registry_v2.py`.
**Status:** Sepolia-ready. Mainnet deployment gated behind L4 audit firm
review per plan §4.5 + PRSM-POL-1 §5 (council resolution required for
new immutable contract on production chain).

This addendum extends the existing PRSM-PROV-1 threat model
(`docs/2026-05-06-prsm-prov-1-threat-model-addendum.md` §§3.16-3.17) with
the on-chain commitment surface introduced by Item 7.

---

## §3.18 — Adversaries against on-chain embedding commitment

### A1. Substitution attacker (cross-model spoofing)

**Goal:** Win a provenance dispute by claiming a different model's
embedding for the same content.

**Capability:** Has access to the file bytes; can compute embeddings
under any model.

**Mitigation:** `compute_embedding_commitment(model_id, dim,
vector_bytes)` binds the model identifier and dimension into the
keccak256 input. A vector under a different model produces a different
commitment, so `verifyEmbeddingCommitment` returns false. Tested in
`test_dispute_changes_with_model_id` and
`test_commitment_changes_with_dim`.

**Residual risk:** None on-chain. Off-chain, an attacker who controls
the `model_id` registry could publish a fake model that produces a
collision — but the model_id is itself canonical text (the
sentence-transformers / OpenAI ID), and the publisher-key anchor (Phase
3.x.3) prevents identity spoofing at the off-chain layer.

### A2. Truncation / extension attacker

**Goal:** Submit a vector of different length that hashes to the same
commitment.

**Mitigation:** `dim` is included in the keccak256 input as a fixed-width
big-endian uint32 prefix. Any length difference produces a different
hash. keccak256 has no known second-preimage attacks at this domain
size.

**Residual risk:** None.

### A3. Squatter (race condition on registration)

**Goal:** A malicious node sees a creator's upload begin off-chain and
front-runs `registerContent` with the same `contentHash` but a different
`embeddingCommitment` they computed from a fake vector.

**Mitigation:** `contentHash` itself is bound to the creator's address
via the v1 inheritance (`compute_content_hash(creator_address,
raw_content_bytes) = keccak256(creator_address_20bytes ||
sha3_256(raw_content_bytes))`). A different sender produces a different
contentHash, so the squatter's tx registers a *different* on-chain
record, not a competing one.

**Residual risk:** None at the `contentHash` level. But: a squatter who
also controls a separate creator key can register fake provenance for
the *same raw content bytes under their own address*. This is the same
v1 squatter problem already documented in §3.16 of the prior addendum,
not a v2 regression.

### A4. Dispute-time vector forgery (zero-commitment edge case)

**Goal:** Win a dispute against legacy / byte-hash-only content (where
`embeddingCommitment == 0`) by submitting `claimed = bytes32(0)` and
hoping the contract returns true on a degenerate match.

**Mitigation:** `verifyEmbeddingCommitment` explicitly returns false
when the on-chain commitment is zero, *regardless of the claim*:

```solidity
if (onChain == bytes32(0)) {
    return false;
}
```

Tested in
`test_dispute_returns_false_for_unregistered_content` and the Hardhat
`returns false for content with zero commitment, regardless of claim`.

**Residual risk:** None.

### A5. Storage-layout corruption (upgrade attacker)

**Goal:** Persuade Foundation governance to upgrade v1 → v2 in-place,
leveraging a storage layout difference to corrupt existing v1 records.

**Mitigation:** v1 ProvenanceRegistry is *not upgradeable* (no proxy
pattern, no UUPS, no transparent proxy). v2 is a separate contract with
its own storage. There is no in-place upgrade path. New uploads register
on v2; old uploads stay on v1. Off-chain code that reads provenance
must check both registries (caller responsibility — documented in the
v2 client docstring).

**Residual risk:** None on-chain. Off-chain coordination risk: nodes
that fail to read v1 will mistakenly think pre-v2 content is
unregistered. Mitigated by RoyaltyDistributor's read-both-registries
contract (out of scope for this addendum; tracked separately in the
mainnet upgrade runbook when the Foundation council ratifies v2 deploy
per PRSM-POL-1 §5).

### A6. Privacy regression — vector exposure on dispute

**Goal:** Force a creator to reveal their embedding vector in public
on-chain calldata when responding to a dispute.

**Mitigation:** `verifyEmbeddingCommitment` is a `view` function, so
the *caller* burns the gas, not the contract. The dispute helper is
typically called from off-chain code (a node validating a dispute
challenge), so the vector flows through a single RPC node, not the
public mempool. If a creator wants to dispute on-chain in calldata, the
existing v1 advice applies: only commit to embeddings the creator is
willing to publish (since the commitment is public on-chain anyway, the
vector is implicitly committed-to but not revealed).

**Residual risk:** Low. A creator who registers an embedding
commitment has implicitly chosen to make that vector verifiable; they
must accept that any party with the file can recompute and prove they
have it. This is the *intended* property — the entire point of the
commitment is to enable disputes.

---

## §3.18.1 — Cross-cutting invariants

| Invariant | Why it holds | Test |
|---|---|---|
| commitment binds (model_id, dim, vector_bytes) jointly | keccak256 second-preimage resistance + fixed-width dim prefix | `test_commitment_changes_with_*` |
| zero commitment never matches any claim | Solidity guard `if (onChain == 0) return false` | Hardhat `returns false for content with zero commitment` |
| Python helper agrees byte-exact with Hardhat helper | Both compute `keccak256(model_id_utf8 \|\| uint32_be(dim) \|\| vector_bytes)` | `test_commitment_canonical_format_matches_hardhat` |
| Python kind tag agrees byte-exact with Hardhat tag | Both compute `keccak256(kind_label_utf8)` | `test_kind_tag_canonical_value_*` |
| MAX_ROYALTY_RATE_BPS python mirror = contract value | Hardcoded `9800` in both surfaces with cross-check test | `test_max_royalty_rate_constant_matches_contract` |

---

## §3.18.2 — Audit-prep talking points (L4 review)

When the L4 audit firm engages on this contract, the following are the
non-obvious correctness arguments they should validate:

1. **Storage layout**: v2 is a fresh deployment, NOT an upgrade. There
   is no risk of slot collision with v1. The audit firm should still
   validate that the struct ordering inside v2 is sensible (creator
   /uint16/uint64/bytes32/bytes32/string) — note the unbounded `string
   metadataUri` is at the end so squatters can't grief storage of the
   indexed fields.

2. **Zero-commitment semantics**: Verify the explicit
   `if (onChain == bytes32(0)) return false` short-circuit is in place
   and survives optimizer passes. Without it, a bug where solc folds
   the constant comparison could allow zero-claim spoofing against
   byte-hash-only records.

3. **Event emission**: ContentRegistered emits both new fields
   non-indexed; Foundation indexers should reindex against this when
   v2 deploys. No on-chain logic depends on event emission, but
   off-chain dispute resolution does.

4. **Re-registration immunity**: `transferContentOwnership` does NOT
   modify `embeddingCommitment` or `fingerprintKind` (verified in
   Hardhat test `transfer leaves embeddingCommitment untouched`). If
   the audit firm wants this changed (e.g. to allow new owners to
   commit to a new vector), that's a v3 design discussion, not a v2
   bug.

5. **No payable functions**: Both new fields are bytes32, no value-bearing
   surface. No reentrancy concerns.

---

## §3.18.3 — Out-of-scope for this addendum

- Wiring v2 into ContentUploader's `_register_on_chain` path (T7.5).
  Plan defers this until ratification of which content types should
  publish embedding commitments by default (privacy implication: every
  upload reveals at least the *existence* of the creator's chosen
  embedding model).
- Sepolia deploy script (T7.3-Sepolia subset). Pattern is identical to
  PublisherKeyAnchor's `scripts/deploy_publisher_key_anchor.js` —
  defer until Item 6 calibration is closer to settling.
- Mainnet upgrade runbook + council resolution (T7.3-mainnet subset).
  Gated behind L4 audit per plan §4.5.
- E2E dispute flow against a live Sepolia contract (T7.7). Will land
  alongside the Sepolia deploy when bandwidth allows.

---

**Cross-references:**
- Plan: `docs/2026-05-06-content-provenance-correctness-plan.md` §4
- v1 threat model: `docs/2026-05-06-prsm-prov-1-threat-model-addendum.md`
  §§3.16-3.17
- Treasury policy: PRSM-POL-1 §5 (council resolution requirements)
- Audit gating: AUDIT_PLAN.md L4 (smart contract specialist firm)
