---
name: agent-teams
description: |
  Orchestrate Claude Code agent teams for PRSM phased development work. Use when a
  PRSM task spans multiple independent workstreams (contract module + test author +
  auditor in parallel; WASM runtime + scheduler + TEE specialist on a phase kickoff;
  architecture shotgun across perspectives; multi-file refactor with natural shards).
  Skip for single-file edits, linear bug fixes, daily bake-in check-ins, and
  documentation tweaks — those are cheaper as a single session. Use when asked to
  "spin up a team", "parallelize this phase", "fan out the implementation", or
  "get multiple perspectives on X". Proactively suggest at the start of a new PRSM
  phase or when a plan doc has 3+ independent line items.
allowed-tools:
  - Bash
  - Read
  - Edit
  - Write
  - Grep
  - Glob
  - Agent
  - AskUserQuestion
  - TaskCreate
  - TaskUpdate
  - TaskList
  - TaskGet
---

# agent-teams — PRSM phased development orchestration

Claude Code's agent-teams feature (experimental, v2.1.32+) spawns multiple
independent Claude sessions coordinated by a lead. This skill codifies **when** and
**how** to use it for PRSM's phased roadmap (Phase 2 remote compute dispatch
onward), with the guardrails that keep us from burning tokens on work a single
session would do better.

## When to use (decision rule)

Spin up an agent team **only if** the work satisfies all three:

1. **Parallelizable.** The subtasks do not share in-memory state and do not need
   to wait on each other. Good: "implement the scheduler, the WASM host-call
   shim, and the TEE attestation probe." Bad: "fix this bug, then verify, then
   deploy" — that's linear.
2. **Independently verifiable.** Each teammate's output can be judged on its own
   (tests pass, contract compiles, audit report produced). If everything funnels
   into one PR that must be reviewed holistically anyway, a single session with
   subagent delegations (`Agent` tool) is usually cheaper.
3. **Scope justifies the tokens.** Agent teams cost N× a single session's tokens
   where N = teammate count. Worth it for phase kickoffs, multi-module
   implementations, and architecture shotguns. Not worth it for <2 hours of work.

**Default: single session + subagent delegations.** Agent teams are the escalation,
not the baseline.

## Red-flag thoughts (don't spin up a team for these)

| Thought | Reality |
|---------|---------|
| "It'll be faster with 5 teammates" | Coordination tax usually eats the speedup under 4-hour workloads |
| "I want a second opinion" | Use `/codex` or the `code-reviewer` subagent, not a whole team |
| "The plan has lots of sections" | Count *independent* sections. Sequential sections = single session |
| "I'll parallelize the bake-in daily checks" | One-command cron job. No team needed |
| "Let's have teammates argue perspectives" | One session running structured red-team prompts is usually cleaner |

## PRSM-specific team patterns

Each pattern lists the composition Claude should suggest when the user invokes
this skill for that kind of work. The user can override teammate count, model,
and roles.

### Pattern: Phase kickoff (new phase from roadmap)

Use when starting Phase 2, 3, 4, etc. per `docs/2026-04-10-audit-gap-roadmap.md`.

- **Lead** (Opus, this session): reads the phase plan, decomposes into
  independent tracks, spawns teammates, owns task list.
- **Architect teammate** (Opus, plan-approval required): drafts module-level
  design for the phase's core component (e.g., WASM dispatcher for Phase 2).
  Plan must pass lead review before implementation.
- **Implementer teammate(s)** (Sonnet, 1-3 of them): one per independent
  module/contract/service in the phase plan.
- **Test-author teammate** (Sonnet): writes forge tests for contract work OR
  pytest integration tests for Python work. Runs in parallel with implementers,
  consumes their interfaces from the architect's plan.
- **Reviewer teammate** (Opus, `code-reviewer` subagent type): final review
  against the original plan before lead merges.

Cost: ~4-5× single-session. Suitable for a phase that will consume 1-2 weeks of
real calendar time.

### Pattern: Smart-contract workstream

Use for any `contracts/` changes (ProvenanceRegistry upgrades, new contracts for
Phase 2/3/7/8, mainnet deploy prep).

- **Lead**: owns deployment script, Basescan verification, exit-criteria doc.
- **Solidity author** (Opus, plan-approval required): writes/edits contracts.
  Plan approval mandatory — smart contracts are irreversible on mainnet.
- **Forge test author** (Sonnet): writes unit + invariant tests against the
  author's interface.
- **Auditor teammate** (Opus, adversarial prompt): reviews for reentrancy,
  access control, overflow, oracle trust, MEV. Does NOT write code. Output:
  findings doc.
- **Gas-optimizer teammate** (Sonnet, optional, only for hot paths): measures
  baseline, proposes specific optimizations with before/after gas numbers.

Lead reviews auditor findings against author's code **before** merging. Cost is
worth it because mainnet rollback = new deploy + migration.

### Pattern: Tokenomics simulation

Use for Epoch 1 FTNS rate simulation, halving-curve validation, supply-cap stress
tests, equity-pivot value-trajectory modeling.

- **Lead**: defines scenarios, owns results doc.
- **Simulator teammate** (Sonnet): implements the model (Python, numpy/pandas).
  No economic judgment — just math.
- **Economic-adversary teammate** (Opus): proposes attack scenarios (whale
  dump, validator collusion, burn-address grief, staking cartel). Writes
  scenarios as config inputs for the simulator.
- **Report-writer teammate** (Sonnet): synthesizes simulator output + adversary
  scenarios into a clean memo.

Keep report writer separate from simulator so numbers aren't laundered through
narrative bias.

### Pattern: P2P / libp2p hardening (Phase 6)

- **Lead**: scopes to a specific attack surface (eclipse, Sybil, NAT traversal,
  pubsub flood).
- **Protocol specialist** (Opus): instruments the relevant libp2p subsystem.
- **Fuzz-test teammate** (Sonnet): writes property-based tests for the
  hardening work.
- **Observability teammate** (Sonnet): adds metrics/traces so we can see the
  hardening working in the wild.

### Pattern: Mainnet deployment prep

This one has a hard rule: **every teammate runs with plan-approval required.**

- **Lead**: owns go/no-go gate.
- **Deployer teammate** (Opus, plan-approval): drafts deploy scripts, dry-run
  output, fork simulation against Base mainnet state.
- **Monitoring teammate** (Sonnet, plan-approval): sets up Basescan webhook,
  on-chain event alerts, gas-price tripwires.
- **Rollback-plan teammate** (Opus, plan-approval): writes the "what if X
  breaks" runbook with concrete mitigation commands.

Lead approves no plan that lacks a tested rollback path.

## How to spin up a team

### Step 1 — Pre-flight

Before the user's first prompt reaches the lead:

```bash
# Verify flag is set
test "$CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS" = "1" || {
  echo "Agent teams not enabled. Restart session after setting the env var." >&2
  exit 1
}

# Verify version ≥ 2.1.32
claude --version
```

Also recommend the user **lower auto-compaction** to 100k-256k tokens before
spinning up a team. Token usage scales linearly with teammate count, and the
default 1M threshold means each teammate can run away before compaction kicks
in. Set via `/config auto-compact <threshold>` or settings.json.

### Step 2 — Team-creation prompt

When the user asks for a team, the lead (this session) drafts a spawn plan:

1. Identify the applicable PRSM pattern (above) or build ad-hoc.
2. Propose the team composition to the user before calling `TeamCreate`:
   role, model, plan-approval flag, subagent type. Use `AskUserQuestion` if the
   user left the composition open.
3. Define the shared task list up front. Include dependencies (architect →
   implementers → reviewer). Set plan-approval on any teammate touching
   contracts, mainnet ops, tokenomics parameters, or irreversible state.
4. On user confirmation, `TeamCreate`, then spawn teammates, then assign
   initial tasks.

**Naming convention.** Give every teammate a human-readable name tied to its
role (`architect`, `contracts-author`, `forge-tests`, `auditor`). This makes
`Shift+Down` cycling and targeted `SendMessage` calls legible.

### Step 3 — During execution

- **Lead stays read-heavy.** Don't write code as lead if a teammate is assigned
  to that track. Lead's job is coordination, review, and merge.
- **Check in on idle teammates.** When a teammate goes idle, read its output
  before it's wiped from the pane. If the task was supposed to be bigger, send
  a follow-up via `SendMessage`.
- **Resolve plan-approval requests promptly.** Do not leave a teammate stuck in
  read-only plan mode. Either approve, reject with specific feedback, or
  shut it down.
- **Watch the task-list health.** If 3+ tasks are pending with no unblocked
  claims, something is wrong in the dependency graph. Fix it before spawning
  more teammates.

### Step 4 — Cleanup

- Shut down each teammate explicitly ("ask the architect teammate to shut
  down"). Wait for acknowledgement.
- Only after all teammates confirm shutdown, run `TeamDelete` from the lead.
  Never `TeamDelete` from a teammate — team context may not resolve correctly.
- Confirm `~/.claude/teams/{team-name}/` and `~/.claude/tasks/{team-name}/` are
  cleaned up.

## Hard rules for PRSM agent-team work

These apply to every team the lead spawns, regardless of pattern:

1. **Never deprecate tests to clear a hurdle.** PRSM's CLAUDE.md forbids
   simplifying tests to pass. When a teammate reports a test failure, the lead
   must require investigation, not a test rewrite. If a teammate proposes
   "simplify the test," reject the plan and require a root-cause analysis.
2. **Edit existing files; don't fork dependencies.** Spawned teammates inherit
   PRSM's CLAUDE.md, but the lead should reinforce this in the spawn prompt.
   Watch for teammates creating parallel config/util files when they should
   edit the canonical one.
3. **Keep the root directory clean.** No teammate writes `.md` reports,
   scratch files, or intermediate artifacts to the repo root. Use `docs/`,
   `docs/archive/`, or `/tmp/`. Lead audits before any commit/push.
4. **SSH for every push.** Teammates that run `git push` must use the SSH
   remote (key SHA256:7X+8TXzSjBE88FP41JQClH1uHWjLbKP/LEAVTKSIapQ). The lead
   verifies the remote URL before authorizing push.
5. **Plan-approval is mandatory for:**
   - Any `contracts/` edit destined for mainnet
   - Any `.gitignore`, `settings.json`, or CI-workflow change
   - Any tokenomics parameter (supply cap, halving schedule, burn %,
     staking yield) change
   - Any deploy script or environment-variable rotation
   - Any deletion of historical docs (archive instead of delete)
6. **Never skip commit hooks** (`--no-verify`, `--no-gpg-sign`). If a hook
   fails, the teammate investigates and fixes. Lead rejects any plan that
   proposes bypassing.
7. **Bake-in log is lead-only.** The 7-day Sepolia bake-in log
   (`docs/2026-04-11-phase1.3-sepolia-bakein-log.md`) is updated by the lead
   after reviewing the daily cron output. Teammates do not touch this file.

## Caveman mode — teammates only, not the lead

**Asymmetric policy:** teammates run caveman for token efficiency; the lead
runs normal English so user-facing communication stays clear and natural.

### Lead: normal English always

The caveman plugin's `SessionStart` hook auto-activates caveman in every
Claude Code session — including this one. As part of pre-flight (before
`TeamCreate`), the lead **must deactivate caveman for its own session**:

```
(lead sends to itself, or user confirms)
stop caveman
```

The lead then speaks full sentences to the user, writes status updates in
normal prose, and drafts commit messages / PR bodies in normal English.
This is non-negotiable — caveman at the lead layer produces fragmented
hand-offs that cost clarity more than they save tokens.

`SendMessage` payloads from lead to teammates MAY be caveman (teammates
are in caveman mode anyway), but the lead's own reasoning and user-facing
output stays normal.

### Teammates: caveman mandatory

Every PRSM teammate runs caveman by default — the plugin's `SessionStart`
hook activates `full` mode automatically on spawn, and the
`UserPromptSubmit` hook keeps it tracked. Lead enforces these rules in
spawn prompts so teammates don't drift:

1. **Stay in caveman** for all prose, status updates, task descriptions,
   mailbox messages, and explanations. Drop articles, filler, hedging.
   Fragments OK. `[thing] [action] [reason]` pattern.
2. **Switch to normal English** automatically for: code, code comments,
   git commit messages, PR descriptions, CHANGELOG entries, smart-contract
   NatSpec, security warnings, irreversible-action confirmations,
   multi-step runbooks where order must not be misread.
3. **Level guidance by role:**
   - `architect`, `auditor`, `reviewer`: `lite` (keep articles + full
     sentences; plan quality > token savings)
   - `wasm-runtime`, `scheduler`, `forge-tests`, implementers: `full`
     (default — fragments OK, caveman register)
   - `report-writer`, daily status teammates: `ultra` (abbreviate,
     arrows for causality, one word when one word enough)

Paste-in phrase for spawn prompts:

> "Run in caveman `full` mode (lite for plan/review work, ultra for status
> reports). Drop articles/filler/hedging. Keep technical substance exact.
> Switch to normal English for code, commits, security warnings, and
> runbooks. Do not ask the lead to drop caveman for itself — lead runs
> normal English intentionally."

If a teammate drifts out of caveman in prose (noticed via `Shift+Down`
cycle or mailbox output), lead sends: `"stay caveman"` and they snap back.
Do not let teammates argue this — the token math is non-negotiable at
team scale.

### Token math with asymmetric policy

6-teammate Phase 2 kickoff:
- Lead (normal English): baseline tokens — no reduction, but lead
  generates far less prose than teammates.
- 6 teammates (caveman full/lite/ultra mix): ~70% reduction each.

Full-day estimate: lead ~500k + 6 × 200k × 0.3 = ~860k tokens
(vs. ~1.7M with lead also in caveman, or ~5M with no caveman anywhere).
Asymmetric policy costs ~5% more than "caveman everywhere" but reclaims
100% of lead-user communication quality.

## Cost-awareness checklist

Before spawning a team, the lead runs this mental check:

- [ ] Estimated wall-clock for this work as a single session: `____` hours
- [ ] Estimated wall-clock with the proposed team: `____` hours
- [ ] Teammate count × avg context usage per teammate: `____` tokens
- [ ] Is the speedup >2× AND the total work >4 hours? If no, prefer single
      session + subagents.
- [ ] Auto-compaction set to ≤256k? If no, set it first.
- [ ] Model selection: Opus only where judgment is load-bearing (architect,
      auditor, reviewer). Sonnet for implementation, test writing, doc work.

## Hooks (optional, set per-project in settings.json)

Three hook types protect against drift:

- **TeammateIdle** — runs when a teammate goes idle. Exit code 2 sends feedback
  and keeps the teammate working. Useful: "did you run the tests after
  implementing?" — fail the idle transition until the teammate confirms.
- **TaskCreated** — runs before a task is added to the shared list. Exit code
  2 prevents creation. Useful: reject tasks that touch forbidden paths
  (e.g., `contracts/` without plan-approval, root-dir writes).
- **TaskCompleted** — runs before a task is marked complete. Exit code 2
  prevents completion. Useful: require that a test suite passed and linked
  output is attached to the task before closing.

PRSM doesn't ship default hooks yet. When we do, they go in project
`.claude/settings.json` and are documented here.

## Subagent reuse (preferred over ad-hoc teammate definitions)

Where PRSM already has subagent types in `.claude/agents/` (or picks up from
the user scope — `claude-code-guide`, `Explore`, `Plan`, `code-reviewer`,
`general-purpose`), spawn teammates using those types rather than writing
inline role prompts. Consistency wins. Example:

> "Spawn a reviewer teammate using the `code-reviewer` subagent type.
> Name it `reviewer`. It reviews the architect's plan before implementers
> start, and the implementer diff before lead merges."

Teammates spawned from subagent definitions honor the definition's tool
allowlist and model; team-coordination tools (`SendMessage`, task management)
are always available regardless of the allowlist.

Caveat: `skills` and `mcpServers` frontmatter fields in subagent definitions
do **not** apply when the definition runs as a teammate. Teammates load skills
and MCP servers from project + user settings, same as a regular session.

## Skill inheritance (superpowers, caveman, etc.)

Every teammate spawns with the same skill catalog a fresh `claude` session would
get: project `.claude/skills/`, user `~/.claude/skills/`, and every plugin
enabled in `~/.claude/settings.json` (superpowers, caveman, gstack). That means
by default, teammates can and will reach for:

- **superpowers**: `brainstorming`, `writing-plans`, `executing-plans`,
  `test-driven-development`, `systematic-debugging`,
  `verification-before-completion`, `subagent-driven-development`,
  `dispatching-parallel-agents`, `using-git-worktrees`,
  `requesting-code-review`, `receiving-code-review`, `writing-skills`.
  The `using-superpowers` skill itself enforces "invoke skills before any
  response" inside each teammate's context — expect proactive skill use.
- **caveman**: terse-communication modes. Caveman's plugin registers both a
  `SessionStart` hook (`caveman-activate.js`) and a `UserPromptSubmit` hook
  (`caveman-mode-tracker.js`). Both fire in every teammate's context
  automatically — no per-teammate setup needed. **This is the primary
  token-bloat defense for agent teams.** With 6 teammates × ~75% token
  reduction, a full-day Phase 2 kickoff goes from ~5M tokens to ~1.25M.
- **gstack**: all `/ship`, `/review`, `/qa`, `/investigate`, `/codex`, etc.

**Gotchas:**

1. **Restrictive `tools` allowlists block the `Skill` tool.** A teammate
   spawned from a subagent definition whose frontmatter omits `Skill` has the
   skill files on disk but cannot invoke them. Only team-coordination tools
   (`SendMessage`, `TaskCreate/Update/List/Get`) are always-on regardless of
   the allowlist. If a teammate's job requires skills (TDD, debugging,
   brainstorming), either spawn inline (no subagent type) or pick a subagent
   definition that includes `Skill` in its tools.
2. **Skill content tokens cost per teammate.** Three implementers each
   loading `test-driven-development` pays the skill-content tokens 3×. Usually
   fine; flag when spawning 5+ teammates that will each reach for the same
   heavy skills.
3. **SessionStart hooks multiply.** Caveman boot-up chatter fires N times.
   Mostly harmless, occasionally noisy.

For PRSM work specifically: lean into superpowers in teammate prompts. Tell
architects to use `brainstorming` + `writing-plans`, implementers to use
`test-driven-development` + `verification-before-completion`, reviewers to
use `requesting-code-review` / `receiving-code-review`. Explicit references
in the spawn prompt make the skill invocation more reliable than relying on
the teammate to notice them.

## Current context (2026-04-17)

- Phase 1.3 Sepolia bake-in: Day 6 of 7. Exit-criteria evaluation
  **2026-04-19 (Sunday)**. Do not spin up agent teams for the bake-in itself
  — it's a cron job.
- Next phase kickoff: **Phase 2 remote compute dispatch** (post-bake-in,
  assuming exit criteria pass). Plan doc:
  `docs/2026-04-12-phase2-remote-compute-plan.md`. Suggested pattern: Phase
  Kickoff (above). Reference R7 addendum for compression-related work items.
- Research-track items (R1-R7) are deferred. Do not spin up teams for them
  until their named promotion triggers fire.

## References

- Official docs: https://code.claude.com/docs/en/agent-teams
- PRSM master roadmap: `docs/2026-04-10-audit-gap-roadmap.md`
- Phase 2 plan: `docs/2026-04-12-phase2-remote-compute-plan.md`
- Glossary (Tier systems, Ring vs Phase): `docs/glossary.md`
- Research track R1-R7: `docs/2026-04-14-phase4plus-research-track.md`
- Risk register (G1-G6): `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/My Vault/Agent-Shared/Risk_Register.md`
