## gstack

Use the `/browse` skill from gstack for all web browsing. Never use `mcp__claude-in-chrome__*` tools.

Available gstack skills:
- /office-hours
- /plan-ceo-review
- /plan-eng-review
- /plan-design-review
- /design-consultation
- /design-shotgun
- /design-html
- /review
- /ship
- /land-and-deploy
- /canary
- /benchmark
- /browse
- /connect-chrome
- /qa
- /qa-only
- /design-review
- /setup-browser-cookies
- /setup-deploy
- /retro
- /investigate
- /document-release
- /codex
- /cso
- /autoplan
- /careful
- /freeze
- /guard
- /unfreeze
- /gstack-upgrade
- /learn

## Testing Methodology

- When performing tests do not deprecate to simpler testing methods - work through the problem and if, need be, download packages to aid in your debugging.
- never deprecate tests or simplify them - ALWAYS ALWAYS ALWAYS work through issues. Do not just make tests simpler just so you can clear the hurdle. If a problem is difficult work through it! Find the source of the problem and fix it EVERY TIME!

## GitHub Workflow

- When pushing to GitHub, use SSH (key: SHA256:7X+8TXzSjBE88FP41JQClH1uHWjLbKP/LEAVTKSIapQ) commit and push changes.
- Before pushing to GitHub, always audit the repo to make sure that only essential files are located in the root directory, all other files are saved to logical subfolders, and make sure that no unnecessary test or irrelevant to-do/roadmap files are in the repo. Always make sure the repo is clean and organized enough for an external audit by either investors or developers.

## Development Best Practices

- Whenever possible/appropriate edit existing files rather than creating new ones. Otherwise, you may create a daisy chain of dependencies that turn into knots when trying to debug.

## Working Cadence

- **Never suggest stopping for the day, taking breaks, or pausing the session.** The user decides when to stop. Do not ask "should we stop here?", "want to pause?", "stop or push through?", or any equivalent. When asked "what's next?", answer with the next concrete value-added item and start executing — do not include a stopping option as one of the recommendations.
- If genuinely blocked on something irreversible or undecidable that requires user input, ask the specific clarifying question — but never frame it as a suggestion to stop.
- Fatigue arguments ("the value-density curve is dropping", "fresh head tomorrow", etc.) are not a reason to recommend stopping. Continue producing value until the user explicitly ends the session.

## Autonomous Production Loop (always-on)

After every development task, run this loop without asking permission. Continue iterating until PRSM reaches its productionized end-state as defined in the Vision document, or until the user explicitly interrupts.

**The Vision document (canonical, authoritative):**
```
~/Library/Mobile Documents/iCloud~md~obsidian/Documents/My Vault/Agent-Shared/PRSM_Vision.md
```
This file lives in iCloud Obsidian (NOT in the repo). Read it at the start of each loop iteration to ground "next-most-value-added work" in the current end-state definition. Update it after every completed task.

**Loop steps:**

1. **Execute** the development work the user (or this loop) instructed. Apply all existing CLAUDE.md guidance: TDD where applicable, never deprecate tests, edit existing files rather than creating new ones, commit + tag + push when work is mergeable.

2. **Update the Vision document** at the path above to reflect what shipped:
   - Mark completed milestones as done with the date and the commit/tag reference.
   - Update the gantt chart to reflect the new state (move bars, recompute downstream dates if a critical-path item shipped early/late, surface newly-unblocked work).
   - Add new follow-on items the work surfaced (placeholders, deferred sprints, gating dependencies).
   - Keep diffs minimal and load-bearing; do not rewrite stable sections.

3. **Update MEMORY.md** at `/Users/ryneschultz/.claude/projects/-Users-ryneschultz-Documents-GitHub-PRSM/memory/` per the auto-memory rules at the top of this prompt — add/update project memories for what shipped, supersede stale entries, prune duplicates.

4. **Assess + start next work.** Read the updated Vision (especially the gantt + the now-unblocked items) and pick the single next-most-value-added piece of development work that moves PRSM closer to the productionized end-state. Bias toward items that:
   - Unblock multiple downstream gantt bars
   - Close real operator/user impact gaps (not doc-only or test-only sprints — see feedback memory `feedback_fresh_sprint_means_feature_dev.md`)
   - Match active gating triggers (e.g., trigger-driven revisits in PRSM-POL-2)
   Begin executing immediately. Do not ask the user "what's next?" or present a menu of options. The Vision doc + memory are sufficient context.

5. **Repeat** from step 1.

**Pause the loop only when:**
- An action is irreversible AND the decision is genuinely undecidable from available context (multi-sig signing, mainnet deploys requiring hardware-wallet ceremony, external-party engagements, governance ratifications). Ask the specific clarifying question, then resume the loop on answer.
- The user explicitly interrupts with a different instruction. Honor the new instruction, then resume the loop on its completion.
- The Vision document declares PRSM productionized end-state reached. At that point, surface the milestone to the user and await direction.

**Never:**
- Skip step 2 or 3 — Vision + MEMORY drift is the failure mode this loop is designed to prevent.
- Treat "fresh sprint" or "what's next" as questions to answer with a menu — they're triggers to execute step 4.
- Interpret a one-off task ("fix this bug") as opt-out from the loop — complete the task in step 1, then proceed to step 2.