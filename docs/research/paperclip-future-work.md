# Paperclip — Future Orchestration Layer

**Filed:** 2026-03-30  
**Status:** Deferred — relevant when Prismatica + PRSM Foundation are both running  
**Repo:** https://github.com/paperclipai/paperclip

---

## What It Is

Paperclip is an open-source Node.js + React orchestration layer for running AI agent teams as companies. Tagline: "If OpenClaw is an employee, Paperclip is the company."

It provides:
- **Org charts** — roles, reporting lines, job descriptions for each agent
- **Goal ancestry** — every task traces back to the company mission; agents know the "why"
- **Per-agent budgets** — monthly token cost caps with automatic throttling
- **Approval gates** — governance with rollback on config changes
- **Heartbeat scheduling** — agents wake on schedule, check work, act
- **Multi-company isolation** — one deployment, multiple companies with separate data + audit trails
- **Audit log** — immutable tool-call tracing and decision history
- **Clipmart** *(coming soon)* — download/publish entire company templates (org + agents + skills)

Explicitly supports OpenClaw as a worker agent.

---

## Why We're Deferring

At current scale (one company, one human, ~6 agents coordinated via AGENTS.md), Paperclip adds infrastructure complexity with no payoff. Their own README: "If you have one agent, you probably don't need Paperclip. If you have twenty — you definitely do."

---

## Trigger Conditions to Revisit

1. **Prismatica Inc. is legally formed** — distinct from PRSM Foundation, needs separate budget tracking and governance
2. **PRSM Foundation is formed** — two entities running in parallel benefits from multi-company isolation
3. **Agent team exceeds ~10 concurrent workstreams** — cost runaway protection becomes critical
4. **Series A due diligence** — investors will want audit logs and governance documentation

---

## High-Value Features at Scale

| Feature | Why it matters for PRSM/Prismatica |
|---|---|
| Multi-company isolation | Run PRSM Foundation + Prismatica Inc. as separate entities with separate budgets |
| Per-agent budget caps | Prevent runaway spend in production multi-agent workflows |
| Goal ancestry propagation | Every sub-agent task traces to company mission — critical for investor-facing governance |
| Audit log + tool-call tracing | Series A due diligence + regulatory compliance |
| Clipmart company template | Export "Prismatica mega-node operator" config for others to fork |

---

## Quick Start (when ready)

```bash
npx paperclipai onboard --yes
# Starts API server at http://localhost:3100
# Embedded PostgreSQL — no setup required
```

Requirements: Node.js 20+, pnpm 9.15+
