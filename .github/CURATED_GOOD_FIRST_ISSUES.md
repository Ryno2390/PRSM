# Curated Good-First-Issues for PRSM

This document specifies 10 concrete first-contributor issues for the PRSM founder (or future maintainers) to post on GitHub. Each issue is **fully scoped** with file paths, expected behavior, and acceptance criteria — eliminating the "where do I start" friction that kills most first contributions.

**Usage:** Open each as a separate GitHub Issue using the `🌟 Good First Issue` template at `.github/ISSUE_TEMPLATE/good_first_issue.md`. Apply labels: `good first issue`, `help wanted`, plus the relevant component label (e.g., `documentation`, `python-sdk`, `mcp`, etc.).

**Rationale for the curation:** First contributors land on a project, look for a way in, find either no issues labeled `good first issue` (project feels closed) or vague stub-issues (give up after 30 minutes of reading code). 10 well-scoped issues with clear acceptance criteria converts that funnel.

The 10 issues span: **3 documentation**, **3 testing**, **2 minor features**, **2 community/tooling**. Difficulty rated ⭐ (easiest, ~2 hr) to ⭐⭐⭐ (most complex, ~8-10 hr). All have working acceptance criteria that can be verified locally before PR submission.

---

## Issue 1 — Add MCP tool example to README quickstart

**Difficulty:** ⭐ · **Estimated time:** 2-3 hours · **Skills:** Markdown, basic Python

**Description:**
The README quickstart (lines ~37-49) shows installation commands but doesn't show an actual MCP-mediated query. Add a worked example showing:
1. User asks Claude Code (or any MCP client) a question
2. Claude routes to PRSM via MCP
3. PRSM returns a result

The example doesn't need to be runnable end-to-end (mainnet pending) — it needs to communicate the **shape** of the user experience.

**Files to modify:**
- `README.md` — add new "Your First Query" subsection after Quick Start

**Acceptance criteria:**
- [ ] New subsection ≤200 words showing realistic LLM-PRSM interaction
- [ ] Mentions both `prsm_quote` (free cost estimate) and `prsm_analyze` (full execution)
- [ ] Doesn't promise functionality not currently shipped (read `docs/IMPLEMENTATION_STATUS.md` for what's live)
- [ ] PR title: `docs: add first-query example to README quickstart`

**Reference:** `PRSM_Vision.md` §4 worked example covers this in detail; condense to ~150 words for the README.

---

## Issue 2 — Document the 16 MCP tools in a single reference page

**Difficulty:** ⭐⭐ · **Estimated time:** 4-6 hours · **Skills:** Markdown, code reading

**Description:**
The README has a 16-row MCP tools table. The full tool definitions live in `prsm/mcp_server.py` (1070 LOC). For developers building on PRSM via MCP, there's no single "reference page" with each tool's full inputSchema, expected output shape, error modes.

Build that reference page.

**Files to create:**
- `docs/MCP_TOOLS_REFERENCE.md` — one section per tool with full schema + examples

**Acceptance criteria:**
- [ ] All 16 tools documented (one section each)
- [ ] Each section: tool name, description, full inputSchema (JSON), example invocation, example output, common errors
- [ ] Linked from `docs/INDEX.md`
- [ ] Linked from README MCP Integration section
- [ ] PR title: `docs: add MCP tools reference page`

**Reference:** `prsm/mcp_server.py` `TOOLS = [...]` array (lines 42-389). Each Tool() definition has the schema you need.

---

## Issue 3 — "Run a node in 10 minutes" guide

**Difficulty:** ⭐⭐ · **Estimated time:** 3-4 hours · **Skills:** Markdown, hands-on testing of `prsm node start`

**Description:**
The README quickstart shows `prsm node start` as a one-liner. But "running a node" has nuance — bootstrap connectivity, hardware tier classification, FTNS welcome grant, monitoring earnings. New node operators need a focused walkthrough.

**Files to create:**
- `docs/RUN_A_NODE_IN_10_MINUTES.md` — single-page operator quickstart

**Acceptance criteria:**
- [ ] Covers: install → benchmark → start → verify connection → check earnings (5 sections)
- [ ] Realistic 10-minute timeline broken down by section
- [ ] Common-issues troubleshooting at the bottom (5+ common errors with fixes)
- [ ] Tested by running the steps yourself on a fresh machine and confirming each works
- [ ] Links to `SUPPORT.md` for further help
- [ ] PR title: `docs: add run-a-node-in-10-minutes operator guide`

**Reference:** Existing `docs/GETTING_STARTED.md` is more comprehensive but slower; this is the fast path.

---

## Issue 4 — Add unit tests for `RevenueSplitEngine` edge cases

**Difficulty:** ⭐⭐ · **Estimated time:** 4-6 hours · **Skills:** Python, pytest, decimal arithmetic

**Description:**
`prsm/economy/pricing/revenue_split.py` implements the 80/15/5 default split (and the 95/5 no-data variant). Existing tests cover the happy path. Edge cases needed:
1. Zero-amount payment (rounds to 0; should not error)
2. Single compute provider with 100% allocation
3. Unequal compute-provider allocations summing to 100% (e.g., 60/30/10)
4. Decimal precision at the lower bound (0.000001 FTNS)
5. Maximum-cap allocation (provider with 99% gets correct share)

**Files to modify:**
- `tests/economy/pricing/test_revenue_split.py` — add 5 new test cases

**Acceptance criteria:**
- [ ] 5 new test cases covering the listed edge cases
- [ ] All assertions use `pytest.approx` for Decimal comparisons (no exact float equality)
- [ ] Each test has a docstring explaining the scenario
- [ ] All tests pass: `pytest tests/economy/pricing/test_revenue_split.py -v`
- [ ] PR title: `test: revenue split engine edge cases`

**Reference:** `prsm/economy/pricing/revenue_split.py` for the implementation.

---

## Issue 5 — Improve error messages on `prsm_analyze` budget validation

**Difficulty:** ⭐ · **Estimated time:** 2-3 hours · **Skills:** Python, MCP server familiarity

**Description:**
`prsm/mcp_server.py:handle_prsm_analyze` (line 429) checks `budget_ftns <= 0` and returns a generic error message. The error doesn't tell the user what budget would actually work for their query. Improve by:
1. If budget is 0 or negative, suggest running `prsm_quote` first
2. If budget is too small (< minimum), show the minimum AND the typical range for the query type
3. Include a helpful one-liner about FTNS economics ("FTNS pays creators / compute providers / network — minimum 0.01")

**Files to modify:**
- `prsm/mcp_server.py:handle_prsm_analyze` and `handle_prsm_create_agent` (similar patterns)

**Acceptance criteria:**
- [ ] Error messages reference the actual minimum (`MINIMUM_BUDGET_FTNS = 0.01`)
- [ ] Error messages explicitly suggest the next tool to call (`prsm_quote`)
- [ ] No regression on existing tests: `pytest tests/mcp/ -v`
- [ ] PR title: `feat(mcp): improve budget-validation error messages`

---

## Issue 6 — Add hardware-benchmark output to `prsm_node_status` MCP tool

**Difficulty:** ⭐⭐ · **Estimated time:** 4-5 hours · **Skills:** Python, MCP server, slight refactoring

**Description:**
`prsm_node_status` returns ring health but doesn't surface the hardware tier or TFLOPS. New users running the tool from their LLM want a one-shot view that includes "what tier am I serving" — currently they have to invoke two separate tools.

**Files to modify:**
- `prsm/mcp_server.py:handle_prsm_node_status` — extend response to include hardware summary
- `prsm/node/api.py:/rings/status` endpoint — add hardware tier to the response payload (optional; alternative: query `/hardware` endpoint internally)

**Acceptance criteria:**
- [ ] `prsm_node_status` response includes: tier (T1-T4), TFLOPS, TEE backend
- [ ] Tests added covering the new fields
- [ ] No breaking changes to the existing schema (additive only)
- [ ] PR title: `feat(mcp): include hardware tier in prsm_node_status`

---

## Issue 7 — Add `--dry-run` flag to `prsm storage upload` CLI

**Difficulty:** ⭐⭐ · **Estimated time:** 4-6 hours · **Skills:** Python, Click CLI

**Description:**
`prsm storage upload` immediately publishes data on first invocation. Users who are uncertain about parameters (royalty rate, replica count) want to see what *would* happen before committing. Add `--dry-run` flag that:
1. Computes the manifest hash that would be registered
2. Shows the CIDs that would be generated
3. Estimates registration gas cost
4. Skips actual on-chain registration + content upload

**Files to modify:**
- `prsm/cli_modules/storage.py` (or wherever the upload subcommand lives — search if unclear)

**Acceptance criteria:**
- [ ] `--dry-run` flag accepts no value (boolean)
- [ ] In dry-run mode, no on-chain transaction is sent
- [ ] In dry-run mode, no content is actually uploaded to peers
- [ ] Output clearly labeled "[DRY RUN]" in every line
- [ ] Existing behavior unchanged when flag absent
- [ ] Test added covering dry-run path
- [ ] PR title: `feat(cli): add --dry-run flag to storage upload`

---

## Issue 8 — Add `prsm doctor` health-check CLI command

**Difficulty:** ⭐⭐⭐ · **Estimated time:** 6-8 hours · **Skills:** Python, system introspection, error reporting

**Description:**
New users run into a variety of setup issues (Python version, missing dependencies, network access to bootstrap, RPC connectivity, hardware-detection failures). A single `prsm doctor` command that diagnoses common issues would make support easier.

**Files to modify/create:**
- `prsm/cli_modules/doctor.py` — new module
- `prsm/cli.py` — register the new command

**Checks to implement** (start with these 8; expand as you go):
1. Python version ≥ 3.10
2. `prsm-network` package installed and version matches expected
3. Bootstrap node reachable (TCP connect to `bootstrap1.prsm-network.com:8765`)
4. Configured RPC URL responds to `eth_chainId`
5. Hardware benchmark completes without error
6. Local node directory exists and has correct permissions
7. Sufficient disk space (≥10GB free)
8. Optional: TEE detection successful

**Acceptance criteria:**
- [ ] `prsm doctor` runs all checks and prints PASS/FAIL/WARN per check
- [ ] Each FAIL includes a remediation hint
- [ ] Exit code 0 on all-pass, 1 on any-fail
- [ ] Tests cover at least 4 of the 8 checks (mocking external state)
- [ ] PR title: `feat(cli): add prsm doctor health-check command`

---

## Issue 9 — Improve typo / grammar in `docs/INDEX.md`

**Difficulty:** ⭐ · **Estimated time:** 2-3 hours · **Skills:** Careful reading, basic Markdown

**Description:**
`docs/INDEX.md` is the navigable map of PRSM's documentation surface. It's been edited many times and likely has typos, grammar issues, broken-but-not-yet-noticed cross-references, or outdated section descriptions. Read carefully and fix.

**Files to modify:**
- `docs/INDEX.md`

**Acceptance criteria:**
- [ ] At least 5 typo / grammar fixes
- [ ] At least 1 broken-link fix (verify all `[text](path)` links resolve)
- [ ] At least 1 outdated description updated to match the actual current file
- [ ] No new content additions in this PR (focus on cleanup, not expansion)
- [ ] PR title: `docs: typo + grammar pass on docs/INDEX.md`

**Tip:** Use a tool like `markdown-link-check` or just manually verify links — the goal is correctness, not feature additions.

---

## Issue 10 — Add Discord welcome bot configuration

**Difficulty:** ⭐⭐⭐ · **Estimated time:** 6-10 hours · **Skills:** Discord API, JavaScript / Python, basic ops

**Description:**
The PRSM Discord (https://discord.gg/R8dhCBCUp3) currently has no welcome automation. New members join and don't know which channel to post in, what role to ask for, or how to start contributing. Build:

1. A welcome message posted automatically when a new member joins
2. Self-assign role workflow (creator / node-operator / dev / governance)
3. A pinned `#start-here` post linking to README, SUPPORT, CONTRIBUTING, good-first-issues

This is technical-but-low-stakes work — Discord bot APIs are well-documented and the failure mode is low (no production data at risk).

**Files to create:**
- `ops/community/discord-bot/` — new directory
- `ops/community/discord-bot/README.md` — setup instructions
- `ops/community/discord-bot/welcome.js` (or .py) — the welcome bot

**Acceptance criteria:**
- [ ] Documented setup process (Discord developer portal → bot token → invite)
- [ ] Welcome message posts automatically on new-member join
- [ ] Role-self-assignment via reaction or slash command
- [ ] Bot deployable to a small VPS (~$5/mo) — dependencies and runtime documented
- [ ] PR title: `feat(community): Discord welcome bot + role assignment`

**Note:** Coordinate with the founder before deploying — the bot needs a Discord application + token managed by foundation operations.

---

## Difficulty distribution

| Stars | Issues | Total time |
|---|---|---|
| ⭐ (easiest) | #1, #5, #9 | ~7-9 hours |
| ⭐⭐ (medium) | #2, #3, #4, #6, #7 | ~21-28 hours |
| ⭐⭐⭐ (most complex) | #8, #10 | ~12-18 hours |
| **Total** | **10 issues** | **~40-55 hours** |

A single dedicated contributor can clear all 10 in 4-6 weeks of part-time work. Opening all 10 simultaneously is fine — contributors will self-select to what matches their skills/interests.

## Posting checklist (for the founder)

When ready to post these on GitHub:

- [ ] Apply labels: `good first issue`, `help wanted`, plus relevant component label
- [ ] Set milestone if appropriate (e.g., `pre-mainnet polish`, `community-launch`)
- [ ] Add `mentor` field with the founder's GitHub username (`@Ryno2390`) until others are designated
- [ ] Cross-post a single Discord message in `#general` and `#dev` listing all 10 with links
- [ ] Update `README.md` "Contributing" section to mention "10 good-first-issues live — pick one and ping in Discord"

## After landing each contribution

- [ ] Thank the contributor publicly in Discord `#general`
- [ ] Add them to `CONTRIBUTORS.md` if not already (create the file if missing)
- [ ] Tag them in the next release notes
- [ ] Promote interesting contributions in PRSM's Twitter / blog

---

## Maintaining this list

When an issue is closed:
- [ ] Mark it as `[CLOSED]` here with link to merged PR
- [ ] Replace with a new good-first-issue at similar difficulty + similar topic area
- [ ] Goal: maintain ~10 open `good first issue`-labeled issues at any given time

When a topic becomes saturated (e.g., 5 contributors finished doc tasks; need more code tasks):
- [ ] Rotate categories — bring in fresh tooling / SDK / governance tasks
- [ ] Avoid `good-first-issue` inflation: if it would take > 10 hours for a senior dev, it's not first-issue material

---

## Versioning

- **0.1 (2026-04-26):** Initial 10-issue curation. Refresh quarterly post-mainnet.
