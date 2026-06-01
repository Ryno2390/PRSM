# PRSM-PROV-1 Item 6 — Operator Activation Runbook

**Document identifier:** PROV-1-ITEM-6-ACTIVATION-1
**Version:** 0.1 (initial)
**Status:** Production-ready. Item 6 is shipped end-to-end through tag `prov-1-item-6-t6-5-gov-next2-merge-ready-20260508`; this runbook is the operator-facing activation guide.
**Date:** 2026-05-08
**Drafting authority:** PRSM founder
**Related documents:**
- Plan: `docs/2026-05-06-content-provenance-correctness-plan.md` §5
- T6.5 design: `docs/2026-05-07-PRSM-PROV-1-T6.5-arbitration-queue-design.md`
- Audit-prep: `docs/2026-04-27-cumulative-audit-prep.md` §7.19
- Threat model: `docs/2026-05-08-prsm-prov-1-threat-model-addendum-item-6.md` (§3.18)

---

## 1. Purpose

Pre-Item-6, every dedup hit at `sim ≥ DERIVATIVE_THRESHOLD` (a single class constant of 0.92) auto-prepended the matching CID as a parent of a new upload, splitting royalties under the 70/25/5 schedule. This created auto-attribution disputes whenever borderline-similar content (scientific abstracts in the same field, code sharing boilerplate) triggered the threshold without genuinely being a derivative work.

Item 6 introduces a **disputed band** between `arbitration_floor` and `derivative_threshold` — borderline matches route to council review rather than auto-attributing. This runbook covers:

1. **Activation** — env vars + dependencies for production nodes.
2. **Verification** — log lines + queue inspection commands proving the activation worked.
3. **Monitoring** — what to alert on (FTNS-drain, queue-depth, sink-failure pattern).
4. **Council resolution flow** — how disputed records become governance proposals + how verdicts close them.
5. **Rollback** — how to disable Item 6 if needed.
6. **Troubleshooting** — common issues + fixes.

---

## 2. Activation

### 2.1 Prerequisites

| Item | Check |
|---|---|
| Node running tag ≥ `prov-1-item-6-t6-5-gov-next2-merge-ready-20260508` | `git describe` |
| `prsm/data/dedup_thresholds.yaml` is on disk + readable | `python -c "from prsm.data.dedup.thresholds import ThresholdResolver; ThresholdResolver.from_default_path()"` |
| `~/.prsm/` directory writable (for arbitration queue persistence) | `[ -w ~/.prsm ]` |
| **(For governance proposals)** A Foundation-controlled FTNS-bearing address available as system proposer | Operator-side custody decision |
| **(For governance proposals)** That address has FTNS sufficient to cover `PROPOSAL_SUBMISSION_FEE` per disputed-band record | `ftns_service.get_balance(<addr>)` |

### 2.2 Activation modes

There are three valid activation tiers. Operators pick one:

#### Tier 0 — Default (no env vars)

**Configuration:** No env vars set.

**Behavior:** `_build_threshold_resolver_or_none` loads the canonical YAML and returns a resolver (not env-gated). `_build_arbitration_queue_or_none` constructs a `FilesystemArbitrationQueue` at `~/.prsm/arbitration_queue/` (not env-gated). The only env-gated component is the proposal sink, which is None at this tier.

Net behavior: 3-band routing fires, disputed-band records persist to disk, but no governance proposals auto-create. Councils review by inspecting the queue dir directly (see §5.2 — Path B).

This is the recommended starting tier for testnet and early-mainnet operation. Records accumulate; councils author proposals manually if/when they want to act on a record. To get true pre-Item-6 binary auto-attribute behavior bit-identically, see §6.3 (rollback).

#### Tier 1 — Auto-proposal creation

**Configuration:**
```bash
export PRSM_ARBITRATION_PROPOSER_ID="0x<foundation-safe-or-delegate-address>"
```

**Behavior:** All of Tier 0 PLUS `TokenWeightedVotingProposalSink` is wired. Disputed-band records auto-create `ProposalCategory.ARBITRATION_DISPUTE` proposals via `TokenWeightedVoting.create_proposal`. The returned UUID is linked back to the queued record via `set_proposal_id`.

**FTNS cost:** Each disputed-band record costs `PROPOSAL_SUBMISSION_FEE` FTNS deducted from the proposer address. Configure proposer to hold at least `100 × PROPOSAL_SUBMISSION_FEE` to absorb a day of moderate dispute traffic (operator-tunable based on observed volume).

#### Tier 2 — Auto-proposal + custom calibration

**Configuration:**
```bash
export PRSM_ARBITRATION_PROPOSER_ID="0x<addr>"
# AND replace prsm/data/dedup_thresholds.yaml with your operator-tuned version
# OR mount a different YAML at the same path before node startup
```

**Behavior:** Tier 1 PLUS operator-supplied threshold values (e.g. once T6.4 calibration corpus is run, replace conservative defaults with ROC-tuned per-kind values).

**Defer Tier 2 until T6.4 calibration runs.** v1 conservative defaults are fine for the first 30+ days of testnet operation.

### 2.3 Restart sequence

1. Stop the node:
   ```bash
   systemctl stop prsm-node   # or your supervisor equivalent
   ```

2. Set env vars in your supervisor's environment file (NOT in shell — env vars must persist across restarts):
   ```bash
   # /etc/prsm/node.env (example for systemd EnvironmentFile=)
   PRSM_ARBITRATION_PROPOSER_ID=0x<addr>
   ```

3. Start the node + watch logs for the activation messages (see §3.1).

4. Verify queue dir was created:
   ```bash
   ls ~/.prsm/arbitration_queue/   # empty initially; populates on first disputed-band upload
   ```

---

## 3. Verification

### 3.1 Startup log lines

Look for these in the node startup log. Each appears at INFO level when the corresponding component wires successfully:

```
TokenWeightedVotingProposalSink wired with proposer_id=0x<addr> (disputed-band records will surface as ARBITRATION_DISPUTE proposals)
```

Absence of this line means one of:
- `PRSM_ARBITRATION_PROPOSER_ID` not set → Tier 0 mode (queue runs, sink doesn't).
- Sink construction failed (FTNS service unavailable, etc.) → check WARN logs for the corresponding `failed to construct TokenWeightedVotingProposalSink` line.

### 3.2 First disputed-band upload (smoke test)

To confirm activation end-to-end, post two near-but-not-quite-identical pieces of content:

1. **Upload A:** original content, say a 500-word scientific abstract. Lands as ordinary upload; CID = `<cid_A>`.

2. **Upload B:** the same abstract with a few sentences paraphrased. Engineered to hit similarity ~0.86 against `<cid_A>` (in the disputed band of text-vector defaults: `arbitration_floor=0.82, derivative=0.92`).

Expected log line on Upload B:
```
Disputed-band similarity for '<filename>': CID <cid_A>... (similarity=0.86xx, floor=0.8200, derivative=0.9200). Flagged for arbitration; no auto-parent.
```

Then verify the queue:
```bash
ls ~/.prsm/arbitration_queue/
# Expected: one .json file per disputed-band record
```

```bash
cat ~/.prsm/arbitration_queue/<record_id>.json
# Expected fields: record { new_cid, new_creator, candidate_parent_cid, candidate_parent_creator, similarity, fingerprint_kind, flagged_at, proposal_id }, resolution: null
```

If `proposal_id` is non-null on Tier 1, the sink fired successfully. If `proposal_id` is `null`, check WARN logs for sink-failure messages.

### 3.3 Queue health

Operators should periodically inspect:

```bash
ls ~/.prsm/arbitration_queue/ | wc -l    # total records (resolved + pending)
```

```python
# pending count via Python
import asyncio
from pathlib import Path
from prsm.data.dedup.arbitration import FilesystemArbitrationQueue
q = FilesystemArbitrationQueue(Path.home() / ".prsm" / "arbitration_queue")
print(len(asyncio.run(q.list_pending())))   # unresolved records only
```

---

## 4. Monitoring

Add the following alerts to your logs pipeline (Promtail / Vector / Filebeat):

### 4.1 Sink failure pattern (WARN-level alert)

**Log substring:** `TokenWeightedVotingProposalSink: backend rejected arbitration proposal`

**Threshold:** ≥3 occurrences within a 5-minute window → page on-call.

**Common cause:** Foundation Safe FTNS balance depleted. Refill the proposer address; sink will recover automatically (no node restart needed).

### 4.2 Queue-depth drift (operator dashboard)

**Metric:** `len(asyncio.run(q.list_pending()))` exposed via a periodic health-probe endpoint.

**Threshold:** unresolved-records ≥ 50 sustained over 7 days → councils not keeping up with dispute volume; consider auto-defer or rate-limit policy.

**Note:** v1 has no built-in counter; operator wraps the queue in their own metric scrape.

### 4.3 Arbitration enqueue failure (WARN-level alert)

**Log substring:** `arbitration enqueue failed for cid=`

**Threshold:** ANY occurrence → investigate. The queue dir is on local disk; failures here suggest disk-full or filesystem-corruption.

### 4.4 Conflicting verdict (ERROR-level alert)

**Log substring:** `conflicting resolve for` (raised from `ArbitrationQueue.resolve`)

**Threshold:** ANY occurrence → investigate. Indicates the governance backend's webhook delivered conflicting verdicts for the same record. v1 retains the *first* verdict; operator must reconcile manually.

### 4.5 FTNS balance drain (financial alert)

**Metric:** `ftns_service.get_balance(PRSM_ARBITRATION_PROPOSER_ID)`

**Threshold:** balance < `10 × PROPOSAL_SUBMISSION_FEE` → refill within 24h to avoid sink-failure pattern.

---

## 5. Council resolution flow

When a disputed-band record exists, a council member or DAO voter resolves it through one of three paths:

### 5.1 Path A — Off-chain governance vote (Tier 1)

1. The proposal exists in `TokenWeightedVoting.proposals` keyed by the UUID returned by `create_proposal`.
2. Council members vote via `TokenWeightedVoting.cast_vote(proposal_id, voter_id, vote_choice)`.
3. When voting closes (per the `VotingPeriod` configured for the category), backend computes results.
4. Backend's webhook (operator-implemented) calls:
   ```python
   await arbitration_queue.resolve(
       record_id,
       decision=ArbitrationDecision.UPHELD_PARENT,    # or REJECTED_PARENT
       by_council=[<voter_addresses>],
   )
   ```
5. Queue record's resolution is now persisted; subsequent `list_pending()` excludes it.

### 5.2 Path B — Manual council inspection (Tier 0)

For Tier 0 deployments without sink wiring:

1. Council member reads the queue file directly:
   ```bash
   cat ~/.prsm/arbitration_queue/<record_id>.json
   ```
2. Council deliberates off-platform (Discord, Signal, etc.).
3. Council member calls `arbitration_queue.resolve(...)` directly via an operator-provided CLI tool (not in v1; bring your own).

### 5.3 Path C — Future on-chain arbitration contract

Long-term per design doc §5.2.3. The byte-deterministic `render_arbitration_body` is the load-bearing primitive — a future contract will sign over the body bytes and councils' on-chain vote will close the dispute.

### 5.4 Verdict semantics

| Verdict | Effect |
|---|---|
| `UPHELD_PARENT` | Council confirms derivative claim. Operator-implemented downstream layer SHOULD now back-attribute the candidate parent (mutate provenance + retroactively split royalties). v1 has no built-in retro-attribution; operator implements per their royalty-distribution architecture. |
| `REJECTED_PARENT` | Council rejects derivative claim. No back-attribution; uploader keeps full original-creator royalty share. |
| `INSUFFICIENT` | Not enough info; record stays open for re-review. `list_pending()` continues to surface it. |

---

## 6. Rollback

If Item 6 needs to be disabled (governance backend outage, calibration mistake producing pathological thresholds, etc.):

### 6.1 Disable proposal sink only (keep queue running)

```bash
# In supervisor env file
unset PRSM_ARBITRATION_PROPOSER_ID
# Or:
PRSM_ARBITRATION_PROPOSER_ID=
```

Restart node. Queue continues persisting records; no governance proposals auto-create. Tier 1 → Tier 0 transition.

### 6.2 Disable disputed-band branch entirely (legacy 2-band behavior)

Move or rename the queue dir before restart:

```bash
mv ~/.prsm/arbitration_queue ~/.prsm/arbitration_queue.disabled
```

The `_build_arbitration_queue_or_none` helper will see the missing dir, attempt creation, succeed (creates new empty dir), and behavior continues as Tier 1 — i.e. this does NOT disable the queue.

To actually disable: monkey-patch `_build_arbitration_queue_or_none` to return None, OR pass `arbitration_queue=None` explicitly to ContentUploader, OR temporarily comment-out the construction at `prsm/node/node.py:738` lines `_arbitration_queue = _build_arbitration_queue_or_none()`. There is no env var for queue-disable in v1 (deferred follow-on if operational need emerges).

### 6.3 Disable resolver (full pre-Item-6 behavior)

Remove `prsm/data/dedup_thresholds.yaml` (move it aside):

```bash
mv prsm/data/dedup_thresholds.yaml prsm/data/dedup_thresholds.yaml.disabled
```

`_build_threshold_resolver_or_none` will fail to load the YAML, return None, and `ContentUploader._resolve_text_thresholds` falls back to legacy class-constant thresholds (0.92 / 0.99). Combined with §6.2, this restores pre-Item-6 binary auto-attribute behavior bit-identically.

---

## 7. Troubleshooting

### 7.1 "Disputed-band similarity for '<file>'..." log line never appears

Check in order:

1. Is the upload's similarity actually in the disputed band? Use `prsm_query_similarity <cid_A> <cid_B>` (operator tool) to confirm `arbitration_floor ≤ sim < derivative_threshold`.

2. Is `_arbitration_queue` actually wired? Look for the absence of the warning:
   ```
   failed to construct FilesystemArbitrationQueue: ... — uploads will not record disputed-band records
   ```

3. Is the resolver actually wired? Look for the absence of:
   ```
   failed to load ThresholdResolver: ... — uploads will use legacy 2-band class-constant thresholds
   ```

4. Is the upload taking the binary-fingerprint path? T6.5.x covers binary lanes; verify the FingerprintMatch.kind is `image-phash` / `audio-chromaprint` / `video-multihash` (not `byte-hash` which has no backend).

### 7.2 Queue records with `proposal_id: null` after Tier 1 activation

Sink failed. Check WARN logs for:
```
TokenWeightedVotingProposalSink: backend rejected arbitration proposal (record_id=<id>): <reason>
```

Common causes:
- **Insufficient FTNS balance.** Refill `PRSM_ARBITRATION_PROPOSER_ID` address.
- **Proposer not eligible.** TokenWeightedVoting validates proposer eligibility (FTNS balance, reputation). Configure proposer accordingly.
- **FTNS service unavailable.** Voting backend depends on `get_atomic_ftns_service()`; check service health.

After fixing, NEW records will get linked. Existing un-linked records can be relinked via operator-supplied CLI (not in v1).

### 7.3 `conflicting resolve for <record_id>` raised

Backend webhook delivered different verdicts for the same record. v1 retains the *first* verdict; the conflicting later call is rejected.

To reconcile:
1. Investigate which verdict is correct (by examining vote results + voter addresses).
2. If the FIRST verdict was correct: no action needed; the queue already retains it.
3. If the SECOND verdict was correct: there is no `reopen` operation in v1. Document the record as a known-bad-resolve in operator notes; future v2 may add reopen.

### 7.4 Queue dir grows unbounded

Resolved records remain in the dir for audit history (intentional; supports retroactive analysis). To prune resolved records older than N days, operator implements a periodic cleanup script — not in v1.

### 7.5 Threshold YAML schema-validation errors at startup

```
failed to load ThresholdResolver: defaults[<kind>] missing required tier 'derivative'
```

YAML edit broke schema. Restore from git or copy the canonical YAML:
```bash
git checkout prsm/data/dedup_thresholds.yaml
```

---

## 8. Honest scope (what's NOT in this runbook)

- **T6.4 calibration corpus.** YAML thresholds are conservative defaults. T6.4 (deferred) replaces with empirically-tuned values once 30+ days of testnet upload traffic provide a real similarity-distribution dataset.
- **Cross-node arbitration consistency.** v1 is node-local. A creator on node A flagging node B's upload is R10 territory.
- **On-chain arbitration contract.** Long-term per design doc §5.2.3.
- **Per-creator daily flag cap.** Anti-griefing v2 hardening; deferred until first observed griefing pattern.
- **Built-in retro-attribution on UPHELD_PARENT verdict.** Operator implements per their royalty-distribution architecture.
- **`reopen` operation for resolved records.** v2 if operational need emerges.
- **Built-in queue-depth metric.** Operator wraps with their own scrape.
- **Built-in resolved-record cleanup script.** Operator implements per retention policy.

---

## 9. Cross-references

| Doc | What it covers |
|---|---|
| `docs/2026-05-06-content-provenance-correctness-plan.md` §5 | Item 6 plan (T6.1 → T6.7 task breakdown + rationale). |
| `docs/2026-05-07-PRSM-PROV-1-T6.5-arbitration-queue-design.md` | T6.5 design (disputed-band routing, queue Protocol, governance hook integration model). |
| `docs/2026-04-27-cumulative-audit-prep.md` §7.19 | Audit-prep entry for external review (12 headline guarantees + 6 trust seams). |
| `docs/2026-05-08-prsm-prov-1-threat-model-addendum-item-6.md` | Threat model §3.18 (8 adversary classes A1-A8 with vectors + mitigations + test pins). |
| `prsm/data/dedup_thresholds.yaml` | Per-kind thresholds + content-type-hint multipliers + per-kind floors. |
| `prsm/data/dedup/thresholds.py` | `ThresholdResolver` + `EffectiveThresholds`. |
| `prsm/data/dedup/arbitration.py` | `ArbitrationQueue` Protocol + In-memory + Filesystem impls + `ArbitrationProposalSink` + `render_arbitration_body`. |
| `prsm/economy/governance/arbitration_sink.py` | `TokenWeightedVotingProposalSink` (production binding to TokenWeightedVoting). |
| `prsm/node/content_uploader.py` (search `T6.5`) | Three-band routing in upload path + `_enqueue_arbitration` three-tier failure isolation. |
| `prsm/node/node.py` (search `_build_threshold_resolver_or_none`) | Node-startup wiring for the three components. |

Tag pinning the surface this runbook activates: `prov-1-item-6-t6-5-gov-next2-merge-ready-20260508`.
