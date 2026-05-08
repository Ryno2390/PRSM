# Parallel-lane coordination postmortem — 2026-05-08

**Track:** QueryOrchestrator B7-prep wiring sprint
**Affected commits:** `0d891c82`, `e476de20`, `866d619c` (B5 rescue)
**Severity:** annoying-but-functional — code correct, commit attribution scrambled

---

## What happened

Three parallel `agent-teams` lanes (lead + 2 teammates) running concurrent
commit cycles into a shared `__init__.py` produced commits with **misleading
commit messages**: each commit's title described the author's intent, but
the file contents pulled in unrelated files from the staged-but-uncommitted
work of other lanes.

Specific scrambles:

| Commit | Message described | Actually shipped |
|---|---|---|
| `866d619c` | "AggregatorClientAdapter (rescue commit)" | accurate — explicit rescue of B5 teammate's uncommitted files |
| `0d891c82` | "B7-factory — node_wiring.py" | `http_aggregate_transport.py` + test (B7-prep #4) |
| `e476de20` | "B7-prep — SentenceTransformerEmbedder" | Embedder + my `node_wiring.py` + test (5 files) |

**Code is correct.** All adapters work, all 63 tests pass, all imports
resolve. The damage is purely in the git log — anyone tracing
"where did node_wiring come from?" will find it under the Embedder
commit, which is wrong.

---

## Root cause

`git add` from concurrent lanes accumulates in the same `.git/index`. When
the lead ran `git commit`, the index contained:

1. Files the lead had explicitly added
2. Files teammate-1 had `git add`-ed but not yet committed
3. Files teammate-2 had `git add`-ed but not yet committed
4. `__init__.py` edits from the most recent teammate to touch it

The lead's commit captured **all four**, even though `git status --short`
showed `A` markers that looked like they were "the lead's staged files".

Behavior amplified by:

- **Test files** — teammates ran `pytest` mid-implementation, which
  doesn't touch git but DOES leave files on disk. When followed by
  `git add`, the index picks up the file regardless of which lane
  authored it.

- **`__init__.py` is a coordination point** — every adapter touches
  `prsm/compute/query_orchestrator/__init__.py` to add its export.
  Concurrent edits race; the `git checkout HEAD -- __init__.py` reset
  reverts to the index version, which may itself be a teammate's
  staged but uncommitted edit.

- **Staged-but-uncommitted state persists across `git reset --hard`
  scope** — `git reset HEAD -- <file>` only unstages the named file.
  Other staged files from other lanes stay staged. The lead's
  `git add` on top adds to that pile.

---

## Recovery (what worked)

### Lead
1. `git checkout HEAD -- __init__.py` — revert teammate's pending edits
2. `git reset HEAD -- <teammate-files>` — unstage teammate work
3. `git add <my-files-only>`
4. Verify `git status --short` shows ONLY my files staged
5. Commit + tag + push

### HTTP Transport teammate (cad9f37 abandoned)
1. Realized their commit would ship sentence_transformer files
2. `git show --stat HEAD` to verify the mismatch
3. Soft-reset their commit
4. Discovered their actual files had already been shipped under a
   different commit message — re-tagged the existing commit instead
   of pushing a duplicate
5. Net: zero duplicate work on origin

### Tag remapping (post-merge cleanup)
- `git tag -d <tag> && git push origin :refs/tags/<tag>` — delete
  misplaced tag locally + remotely
- `git tag -a <tag> <correct-commit>` — re-create at the right commit
- `git push origin <tag>` — re-publish

---

## Lessons (for future agent-teams parallel lanes)

### 1. `__init__.py` as a coordination point is a footgun

Multiple lanes touching the same module's `__init__.py` produce the
exact race observed here. Mitigations:

**Best:** Don't touch `__init__.py` in parallel lanes. The LEAD adds
all exports in a separate post-merge commit after both teammates
have shipped their non-`__init__.py` work.

**Acceptable:** Each lane's work is in a sibling submodule with its
own `__init__.py`. The top-level `__init__.py` re-exports via
`from . import <submodule>` — non-conflicting.

**Avoid:** Each lane edits the top-level `__init__.py` and tries to
`git pull --rebase` to merge — the index race documented above
strikes despite the rebase.

### 2. Briefing prompts MUST explicitly say "stage only your own files"

Even with TDD discipline, teammates running `pytest` mid-loop leave
files on disk that get accidentally absorbed into the lead's commit.
Future briefings should include:

> Before `git commit`, run `git status --short` and verify ONLY the
> files you authored show as `A`. Use `git reset HEAD -- <other-file>`
> to unstage anything you didn't write. If `__init__.py` is staged
> from another lane's edits, `git checkout HEAD -- __init__.py` to
> revert and add ONLY your own export hunk.

### 3. Commit-message-vs-content drift is hard to detect at push time

`git push` doesn't validate the commit message against the file
contents. The mismatch only surfaces at `git log` review time. The
HTTP Transport teammate caught it via `git show --stat HEAD` — that
should be a mandatory step in every parallel-lane briefing:

> After `git commit` and BEFORE `git push`, run
> `git show --stat HEAD`. If the file list doesn't match what you
> actually wrote, soft-reset and re-stage cleanly.

### 4. Tag remapping is fine when caught early

Tags are cheap to delete + recreate locally + remotely. As long as
no external consumer has cloned a tag-pointing-at-wrong-commit
yet, fixing it is harmless. **Verify tag → commit → contents
alignment before announcing merge-ready externally.**

---

## What actually shipped (for the record)

| Adapter | File | Tag → commit | Tests |
|---|---|---|---|
| MarketplaceCandidatePoolProvider | `marketplace_candidate_pool_provider.py` | `…cppool…` → `b0aa68c1` | 13 |
| FoundationBeaconProvider | `foundation_beacon_provider.py` | `…beacon…` → `a91c8b79` | 10 |
| SentenceTransformerEmbedder | `sentence_transformer_embedder.py` | `…embedder…` → `e476de20` | 13 |
| HttpAggregateTransport | `http_aggregate_transport.py` | `…http-transport…` → `0d891c82` | 12 |
| node_wiring factory | `node_wiring.py` | `…b7-factory…` → `e476de20` | 18 (split across 2 classes) |

All 4 B7-prep supporting adapters + the B7 factory are functional in
origin/main as of this postmortem. 63/63 tests green across the new
surface; 246-test wide regression sweep across the QueryOrchestrator
program also green.

Remaining work: B7 (node.py:1277 swap to use the factory) +
B8 (lift MCP `BROKEN_TOOLS_HIDDEN` after operator verification).
Both are sequential single-lane tasks. No further parallel
coordination needed.
