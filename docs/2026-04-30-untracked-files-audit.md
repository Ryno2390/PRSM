# Untracked Files Audit — 2026-04-30

**Goal:** the repo had 168 untracked files when this audit started. CLAUDE.md
mandates the repo stay clean for external audit. This document categorizes
the surface and proposes per-category actions. Phase 1 (immediate
deletions, ~4 files) was actioned in the same commit that produces this
doc; Phase 2 (the larger triage of ~164 remaining files) is captured
here for multi-session decision-making since most files require user
intent that I shouldn't unilaterally encode.

---

## Phase 1 — actioned in same commit

Four confirmed-stale files deleted:

| Path | Why |
|---|---|
| `contracts/test/FTNSToken.test.js` | Tests `FTNSToken` contract which doesn't exist (only `FTNSTokenSimple`). 33 ethers v5 idiom uses (`ethers.utils.parseEther`). Same era as the deleted `deploy.js` / `verify-deployment.js`. |
| `docs/audit/INVESTOR_AUDIT_GUIDE.md` | Investor material per `feedback_repo_scope_prsm_vs_prismatica.md` — investor-facing docs go to a separate private Prismatica surface, not the public PRSM dev repo. |
| `docs/research/NWTN_Critical_Analysis_and_Improvement_Recommendations.md` | Dated August 2025; NWTN-era legacy product. Streaming-inference subsystem at Phase 3.x.* is the canonical inference path now. |
| `docs/security/SECURITY_IMPLEMENTATION_STATUS.md` | Dated June 2025; superseded by `docs/2026-04-27-cumulative-audit-prep.md` and the threat-model addendum. |

Untracked count: 168 → 164.

---

## Phase 2 — needs user decision

164 remaining untracked files across 5 buckets:

| Bucket | Count | Phase 2 recommendation |
|---|---:|---|
| `.github/workflows/*.yml` | 12 | NEEDS USER DECISION — see §A |
| `prsm/compute/nwtn/` | 39 → 30 | PARTIAL: 9 .md/.json deleted (this commit). 30 .py + subdirs deferred — see §B |
| `prsm/compute/collaboration/` | 16 | LIKELY LEGACY — see §B |
| `prsm/compute/{federation,agents,chronos,others}/` | 28 (`agents/` expands to 26 .py) | MIXED — see §B. **Pattern confirmed across 3 subdirs:** chronos/ + agents/ + federation/ all show v1.6-plan re-introduction signature. Bulk-delete recommended for all three pending operator confirmation. |
| `tests/{integration,nwtn,scripts_integration,...}/` | 58 | LIKELY LEGACY — see §C |
| `docs/api/PHASE_7_API_REFERENCE.md` | 1 | NEEDS REVIEW — see §D |
| `config/nginx/ipfs-proxy.conf` | 1 | DEPRECATED INFRA — see §E |
| (other small misc) | ~9 | see git status |

---

## §A — `.github/workflows/*.yml` (12 files, ~4,000 lines)

```
cd.yml                           329 lines
ci-cd-pipeline.yml               651 lines
ci-comprehensive.yml             339 lines
deploy-production.yml            516 lines
infrastructure-tests.yml         356 lines
neural-regression.yml             30 lines
performance-regression.yml       343 lines
production-deploy.yml            440 lines
security-validation.yml          465 lines
security.yml                      64 lines
test-coverage.yml                135 lines
validation-pipeline.yml          439 lines
```

**Tracked workflows already in `.github/workflows/`:** `auto-label-issues.yml`, `changelog.yml`, `ci.yml`, `deploy.yml`, `release.yml`, `security-audit.yml`. These are the actual CI surface today.

**No staleness markers found in the 12 untracked workflows** (no Polygon Mumbai, no `FTNSToken`/`FTNSMarketplace`/`FTNSGovernance` references, no references to the deleted `deploy.js`/`verify-deployment.js`/`deploy-simple.js`).

**Possible explanations:**
- Aspirational CI infrastructure from a planning sprint that never got wired in
- Parallel CI that was abandoned in favor of the 6 currently-tracked workflows
- Pending check-in from a previous session that got interrupted

**Recommended action paths:**
1. **Review-and-add:** read each of the 12 to confirm they're useful + non-overlapping with the 6 tracked workflows; add the live ones, delete the rest. ~2-3 hours of focused review.
2. **Bulk delete:** if these are abandoned drafts, delete all 12. ~5 minutes.
3. **Defer:** leave as untracked clutter. Not recommended because external auditors / contributors will see them and wonder.

The right call depends on **operator intent** that I can't determine from the files alone. Path (2) is safer if there's any doubt — re-creating workflows from scratch is cheaper than reviewing 4K lines of YAML for production-readiness.

## §B — `prsm/compute/` (83 untracked files)

### NWTN subdir (39 → 30 files; .md/.json sweep actioned 2026-04-30)

NWTN is the legacy conceptual product per project memory. Streaming-inference subsystem (Phase 3.x.*, with roadmap cap reached at q.x today) is the canonical inference path.

**HOWEVER:** spot-check shows `prsm/compute/nwtn/__init__.py` + `prsm/compute/nwtn/training/__init__.py` ARE tracked. The directory is mixed — some files committed, some untracked. So bulk-delete on `.py` files is unsafe.

**Phase 1 actioned (2026-04-30):** the 9 lowest-risk `.md` + `.json` files at the top of `prsm/compute/nwtn/` deleted. All confirmed dated July-August 2025 (~9 months old) with NWTN-era branding ("Neural Web for Transformation Networking", "9-Step Reasoning Pipeline", "Voicebox", "Breakthrough Pipeline") superseded by the streaming-inference subsystem. Verified zero references from tracked Python files; package import (`import prsm.compute.nwtn`) still succeeds post-deletion.

Files deleted:
- `README.md` (1073 lines; old "9-Step AI Reasoning Pipeline" framing)
- `BREAKTHROUGH_PIPELINE_DOCUMENTATION.md` (Last Updated: July 31, 2025)
- `CRITICAL_FIXES_IMPLEMENTATION_SUMMARY.md` (Date: August 11, 2025)
- `ETHICAL_DATA_INGESTION_ROADMAP.md` (no date; "52 diverse sources" data ingestion strategy from old data layer)
- `NWTN_LOCAL_INTEGRATION_COMPLETE.md` (Generated: August 7, 2025; GPT-OSS+Gemma+Ollama integration era)
- `NWTN_OPTIMIZATION_ROADMAP.md` (730 lines; neuro-symbolic reasoning roadmap)
- `NWTN_OPTIMIZATION_VALIDATION_COMPLETE.md` (candidate-deduplication validation note)
- `NWTN_VOICEBOX_OPTIMIZATION_COMPLETE.md` (Qwen2.5:7B integration note)
- `nwtn_complete_pipeline_results_1754921011.json` (Aug 11 2025 pipeline-results dump)

**Phase 2 deferred:** the 30 remaining untracked items in `prsm/compute/nwtn/` (mostly `.py` files in subdirs: `architectures/`, `backends/`, `bsc/`, `core/`, `corpus/`, `engines/`, `experiments/`, `openclaw/`, `processing/`, `reasoning/`, `security/`, `synthesis/`, `team/`, `test_prompts/`, `whiteboard/` plus a few top-level `.py` files like `breakthrough_modes.py`, `complete_system.py`, `context_manager.py`, etc.). These need import-graph analysis before deletion (tracked files DO import from `prsm.compute.nwtn` — but which specific submodules?). Recommend a dedicated session.

### collaboration/ subdir (16 files)

```
prsm/compute/collaboration/README.md
prsm/compute/collaboration/academic/
prsm/compute/collaboration/containers/
prsm/compute/collaboration/datascience/
prsm/compute/collaboration/design/
prsm/compute/collaboration/development/
prsm/compute/collaboration/enterprise/
prsm/compute/collaboration/grants/
prsm/compute/collaboration/jupyter/
prsm/compute/collaboration/latex/
prsm/compute/collaboration/models.py
prsm/compute/collaboration/references/
prsm/compute/collaboration/specialized/
prsm/compute/collaboration/state_sync.py
prsm/compute/collaboration/tech_transfer/
prsm/compute/collaboration/university_industry/
```

`prsm/compute/collaboration/p2p/__init__.py` + `prsm/compute/collaboration/security/__init__.py` ARE tracked. So this is also a mixed-state subdir.

The list above (academic/containers/datascience/...) reads like a **product catalog** of collaboration verticals — which is a NWTN-era product framing. Current PRSM canonical scope is "Research, Storage, and Modeling" per `project_prsm_scope_2026.md` — these specific verticals don't appear in current architecture.

**Recommended action:** dedicated collaboration-cleanup session. Probably most are legacy.

### Other compute/ subdirs (28 files)

- `federation/` (6) — distributed_evolution / distributed_model_registry / distributed_rlt_network / knowledge_transfer / model_registry / phase5_demo
- `agents/` (6 top-level entries; 26 .py files when subdirs expanded) — see agents-specific findings below
- `chronos/` (4) — see chronos-specific findings below
- Others (12) — validation / teachers / students / network / improvement / evolution / evaluation / distillation / data / candidates / benchmarking / ai_orchestration (1 file each)

Mixed state again — `prsm/compute/agents/__init__.py` + others tracked, but new dirs are not.

**Recommended action:** dedicated compute-cleanup sessions per subdir. Same triage pattern: spot-check imports, commit live files, archive/delete legacy.

#### agents/ specific finding — v1.6 deletion confirmed by tracked-code comments (2026-04-30)

The 6 untracked entries in `prsm/compute/agents/` expand to 26 .py files:
- `base.py` (top-level "PRSM Base Agent Framework / 5-layer PRSM architecture")
- `architects/` (1 file: hierarchical_architect.py)
- `compilers/` (2 files: rlt_enhanced_compiler.py, hierarchical_compiler.py)
- `executors/` (10 files: ollama_client / openrouter_client / hybrid_router / model_executor / api_clients / etc.)
- `prompters/` (1 file: prompt_optimizer.py)
- `routers/` (5 files: rlt_enhanced_router / marketplace_integration / tool_router / performance_tracker / model_router)

**Tracked `__init__.py` describes a TOTALLY DIFFERENT scope** — "PRSM WASM Mobile Agent Runtime" (dispatcher/executor/instruction_set/models/data_processor; Ring 9 Courier pattern). It explicitly does NOT import any of the untracked subdirs. The two architectures (NWTN-era "5-layer" vs current "WASM Mobile Agent") share a directory but not a design.

**Strongest evidence yet that v1.6 deletion was intentional:** all 3 tracked Python files that reference these untracked subdirs do so via `try/except ImportError` with EXPLICIT INLINE COMMENTS confirming the deletion was by-design:

- `prsm/data/context/enhanced_context_compression.py:30`:
  ```python
  # ModelExecutor removed (agents/executors/ deleted in v1.6.0 scope alignment)
  try:
      from prsm.compute.agents.executors.model_executor import ModelExecutor
  except ImportError:
      ModelExecutor = None  # type: ignore[assignment,misc]
  ```
- `prsm/data/context/reasoning_trace_sharing.py:33`: identical pattern
- `prsm/compute/performance/performance_benchmarks.py:114`: function-level `try` import (not module-level)

The tracked code is DESIGNED to work without these files. The re-introduction does NOT activate them in any tracked code path — the soft-import fallback (`ModelExecutor = None`) handles it.

**Disposition recommendation:** same as chronos but with stronger evidence. The case for bulk-delete here is more clear-cut because the v1.6 design intent is documented in tracked-code comments (not just an archive plan doc). All 26 .py files (plus the 5 subdirs and `base.py`) can be removed with no code-path breakage; the soft-import shims would just continue to fall back to `None` (which they already do today since import paths fail when `__init__.py` doesn't exist for the subdirs).

**Verified:** the subdirs ARE the issue — `prsm/compute/agents/executors/__init__.py` exists in the untracked subdir, but Python's import system needs that file to be reachable. With the subdir untracked + Python looking up the path, the imports DO succeed locally (because the files exist on disk), but they would fail in any clean checkout. So the soft-import shims are already "live insurance" against deletion — exactly the v1.6 design.

**Operator decision needed:** same matrix as chronos:
- If v1.6 deletion was correct + re-introduction unintentional → bulk-delete all 26 .py + 5 subdirs + `base.py`
- If re-introduction was intentional (Phase X agent-system revival) → commit them + remove the v1.6 deletion comments + remove the soft-import shims (the comments are now lying)
- If unclear → defer

**My recommendation:** bulk-delete is strongly correct here. The tracked-code comments explicitly document the design intent. Keeping the untracked re-introduction creates a "phantom dependency" — code locally works because files exist, but the same code on a clean checkout would silently route through the `None` fallback. That divergence between local and clean-checkout behavior is exactly what the v1.6 sprint was trying to eliminate.

#### chronos/ specific finding — v1.6-plan re-introduction puzzle (2026-04-30)

The 4 untracked files in `prsm/compute/chronos/` are:
- `cashout_api.py` (137 lines)
- `enterprise_sdk.py` (632 lines)
- `staking_integration.py` (467 lines)
- `treasury_provider.py` (501 lines)

Total: 1,737 LoC of "FTNS↔USD/USDT enterprise treasury" code (MicroStrategy/Coinbase Custody integration scope).

**The puzzle:** `docs/archive/2026-04-09-v1.6-scope-alignment.md` explicitly slated 3 of these (`enterprise_sdk.py`, `staking_integration.py`, `treasury_provider.py`) plus 2 currently-tracked files (`hub_spoke_router.py`, `exchange_router.py`) for `git rm` deletion in the v1.6 sprint. The v1.6 sprint shipped to PyPI 2026-04-09 per project memory. So:
- The 3 untracked files WERE deleted by v1.6, then re-introduced (status now: present-as-untracked, never re-committed).
- The 2 supposedly-deleted files (`hub_spoke_router.py`, `exchange_router.py`) are still tracked — the v1.6 plan was modified mid-execution.

**Dependency direction:** the 4 untracked files import FROM tracked chronos code (`models.py`, `clearing_engine.py`, `hub_spoke_router.py`, `price_oracles.py`). NO tracked code imports the untracked 4. So they're downstream consumers — bulk-deleting them does NOT break any tracked code path. (Confirmed via `git ls-files | xargs grep` for the four module names.)

**Operator decision needed:**
- **If the v1.6 deletion was correct + the re-introduction was unintentional** (e.g., resurrected from a backup branch by accident): bulk-delete all 4. Repo aligns with v1.6 plan.
- **If the re-introduction was intentional** (e.g., enterprise treasury integration is being revived as a Phase X feature): commit them. Update `docs/archive/2026-04-09-v1.6-scope-alignment.md` to note the reversal, or move that doc out of `archive/` since its conclusions no longer hold.
- **If unclear:** keep as-is, but the audit-blocker remains until resolved.

**My recommendation:** the v1.6 sprint had a clear scope-alignment thesis (memory: "deleted ~210K LoC legacy"), and the canonical PRSM scope today is "Research, Storage, and Modeling" — enterprise FTNS↔USD treasury integration doesn't appear in current architecture docs, threat model, or Phase 1.3/Phase 7/Phase 7.1 surfaces. Bulk-delete is more likely the correct call. But this needs operator confirmation since 1,737 LoC carries real cost-to-recreate if the intent was preservation.

**Smaller mystery:** `cashout_api.py` (137 lines) is NOT in the v1.6 deletion list, BUT it imports from `hub_spoke_router.py` (which v1.6 slated for deletion). So it was probably co-introduced with the deletion-plan files. Same disposition recommendation.

Both `hub_spoke_router.py` and `exchange_router.py` are currently tracked but were also slated for v1.6 deletion. Their fate is logically coupled with the 4 untracked files: if the operator decides the enterprise-treasury layer was correctly deprecated by v1.6, all 6 files should go (4 deletions + 2 `git rm` of currently-tracked files). If preservation was intentional, all 6 stay.

#### federation/ specific finding — third subdir matching v1.6 re-introduction signature (2026-04-30)

The 6 untracked files in `prsm/compute/federation/` are:
- `distributed_evolution.py`
- `distributed_model_registry.py`
- `distributed_rlt_network.py`
- `knowledge_transfer.py`
- `model_registry.py`
- `phase5_demo.py`

17 tracked files coexist in the subdir (consensus / p2p_network / fault_tolerance / etc.) — same mixed-state pattern as chronos/ + agents/.

**v1.6 plan evidence:** the same `docs/archive/2026-04-09-v1.6-scope-alignment.md` plan doc explicitly slated 4 of the 6 untracked files for `git rm` deletion: `distributed_evolution.py`, `distributed_rlt_network.py`, `knowledge_transfer.py`, `phase5_demo.py`. The other 2 (`distributed_model_registry.py`, `model_registry.py`) are not in the named deletion list but follow the same re-introduction-after-deletion pattern.

**Soft-import shim pattern matches agents/ exactly:** all 3 tracked test files that reference `distributed_rlt_network` import it via `try/except ImportError` with a `FEDERATION_AVAILABLE = False` fallback flag:

- `tests/integration/test_complete_prsm_system.py:505` — try/except wrapping import
- `tests/integration/test_prsm_core_integration.py:54` — try/except + `FEDERATION_AVAILABLE = True` on success
- `tests/integration/test_prsm_fixed_components.py:70` — explicit `FEDERATION_AVAILABLE = False` initialization, set True only on import success

This is the same "code designed to work without these files" signature documented for agents/ — the soft-import shim handles missing modules by routing through the `FEDERATION_AVAILABLE = False` branch. The tracked tests would NOT break if the 6 untracked files were deleted; they'd just skip the federation-specific paths.

**Disposition recommendation:** same as chronos + agents. Bulk-delete is most likely correct. The case here is supported by THREE pieces of converging evidence:
1. v1.6 plan doc explicitly names 4 of 6 files for deletion
2. Tracked tests use soft-import + `FEDERATION_AVAILABLE` fallback (same pattern as agents/)
3. Files are downstream consumers — no tracked code depends on them being present

**Operator decision needed:** same matrix:
- If v1.6 deletion correct + re-introduction unintentional → bulk-delete all 6
- If re-introduction intentional → commit all 6 + remove `FEDERATION_AVAILABLE = False` fallback in tests
- If unclear → defer

#### Meta-pattern across §B subdirs

Three subdirs (`chronos/`, `agents/`, `federation/`) now show the same v1.6-deletion-then-reintroduction signature. This raises a meta-question worth surfacing to the operator:

- **Was there a single resurrection event after v1.6** (e.g., a stash unstash, a backup-branch merge, a `git restore` from an old ref) that brought back multiple subsystems at once? If so, all three are candidates for the same `git clean -fd` scope.
- **Or were these independent re-introductions** across different sessions, each with different intent? If so, each subdir needs separate operator decision.

The fact that all three subdirs share the soft-import / `*_AVAILABLE = False` fallback pattern suggests the v1.6 sprint anticipated optional re-introduction (otherwise tracked code would just import directly and break on absence). So at minimum the deletion was deliberate; the re-introduction may or may not have been.

## §C — `tests/` (58 untracked files)

```
19 integration
16 nwtn
10 scripts_integration
 4 unit
 3 new_integration_tests
 2 security
 2 benchmarks
 1 performance
 1 load
```

**Important context:** task #89 in the task list is "Repo audit — relocate 109 legacy test files to `tests/_legacy/`" (completed). So there's an established pattern for legacy test relocation. These 58 untracked tests likely missed the #89 sweep.

**Recommended action:**
1. **Bulk-relocate to `tests/_legacy/`** (matches existing pattern from #89), OR
2. **Spot-check + delete obvious legacy** (e.g. `tests/nwtn/*` is almost certainly NWTN-era), OR
3. **Run the test suite to identify broken tests** — if a test fails to import or collect, it's stale.

Path (3) is the most rigorous but slowest. Path (1) is the safest quick win and matches prior precedent.

## §D — `docs/api/PHASE_7_API_REFERENCE.md` (740 lines)

Comprehensive API reference for Phase 7. Contains: Overview, Authentication, Base URLs, API Components, Error Handling, Rate Limiting, Webhooks, SDK Libraries.

Phase 7 is `[in_progress]` in the task list (hardware-gated for mainnet deploy). The doc may or may not match the current Phase 7 contract surface (which has gone through Tasks 1-9 with §8.x hardening passes since this doc was written).

**Recommended action:**
1. **Review for accuracy** against current `prsm/economy/web3/stake_manager.py` + the audit-bundle contracts. If accurate → commit. If stale → either rewrite or delete.
2. **Defer until post-audit ceremony** since the API surface may stabilize differently after auditor review.

## §E — `config/nginx/ipfs-proxy.conf` (1 file)

Referenced from `docs/native-storage-design.md` and `docs/plans/native-storage-migration.md`. The "native-storage-migration" doc title indicates PRSM is migrating AWAY from IPFS to native storage. Only 1 tracked Python file imports IPFS-related modules (vs many for current inference/storage code paths) — confirming IPFS is being phased out.

**Recommended action:**
1. Wait for native-storage migration to land, then delete the conf.
2. Or commit now as documentation of legacy infrastructure.

Currently low priority since the file isn't broken or load-bearing.

---

## Summary

- ✅ **Phase 1 actioned (4 files):** confirmed stale or out-of-scope material deleted.
- ⏳ **Phase 2 deferred (~164 files):** decisions captured in this doc per category, awaiting operator intent + multi-session triage.

The repo is materially cleaner post-Phase 1 than before this audit — every file we deleted was either testing something that doesn't exist (FTNSToken.test.js), an investor-scope violation (INVESTOR_AUDIT_GUIDE.md), or a 9-10 month-old NWTN/security doc superseded by current audit-prep docs.

**The bigger Phase 2 surface (workflows, NWTN, collaboration, tests/) is not blocking hardware-day or the post-audit ceremony.** It's a "before external audit packet ships" cleanup. This audit doc gives the operator the categorized roadmap to drive that cleanup over multiple sessions.

---

## Quick-action one-liners

If the operator wants to take aggressive action without per-file review:

```bash
# Bulk-delete 12 unwired CI workflows (Path A.2)
rm .github/workflows/{cd,ci-cd-pipeline,ci-comprehensive,deploy-production,infrastructure-tests,neural-regression,performance-regression,production-deploy,security-validation,security,test-coverage,validation-pipeline}.yml

# Bulk-relocate untracked tests to tests/_legacy/ (Path C.1; matches #89 precedent)
mkdir -p tests/_legacy/2026-04-30-untracked-sweep
git status --short | grep '^??' | awk '{print $2}' | grep '^tests/' | while read f; do
  mkdir -p "tests/_legacy/2026-04-30-untracked-sweep/$(dirname "${f#tests/}")"
  mv "$f" "tests/_legacy/2026-04-30-untracked-sweep/${f#tests/}"
done

# Spot-check NWTN .md files for deletion
ls prsm/compute/nwtn/*.md
```

Don't run these blindly — the audit report is the deliberation, not the action.
