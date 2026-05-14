# 2026-05-14 — user-perspective dogfood findings

## Context

After today's 36-sprint cycle (sprints 388–423) closed the operator/audit
surface, ran an end-to-end test pretending to be a brand-new user following
the canonical workflow in `docs/PARTICIPANT_GUIDE.md`. Goal: validate that
PRSM does what `PRSM_Vision.md` §4 claims a user can do.

## Headline

**Vision §4 describes the IDEAL workflow; the DEFAULT user experience hits
multiple walls before the first successful query.** This finding is not a
regression — most of the gaps have always existed — but they are visible
now in a way that's worth surfacing as the network moves toward genuinely
inviting new users.

## Friction points (in order of user encounter)

### F1 — `prsm daemon` is deprecated but PARTICIPANT_GUIDE still uses it

**Symptom.** Following Step 3 in PARTICIPANT_GUIDE, a user runs
`prsm daemon start`. The CLI emits a deprecation warning steering them to
`prsm node start --background`. The daemon DOES start (backward-compat),
but the user sees a yellow warning on their FIRST canonical command.

**Severity.** Documentation rot. The user surface still works; the doc
needs updating.

**Fix shipped this sprint.** PARTICIPANT_GUIDE Step 3 updated to use
`prsm node start --background` + `prsm node status` + `prsm node logs -f`.

### F2 — `PRSM_QUERY_ORCHESTRATOR_ENABLED=1` is required but undocumented

**Symptom.** User submits a query (via `prsm_analyze` MCP tool, or directly
to `/compute/forge`). Hits:

```
{"detail": "Agent forge not initialized. Check LLM backend configuration."}
```

The actual cause is that the §7 QueryOrchestrator is default-disabled
(sprint 173 / Vision §13 commit). Operator must set
`PRSM_QUERY_ORCHESTRATOR_ENABLED=1` in the env BEFORE starting the
daemon. The error message says "Check LLM backend configuration" which
is misleading — the LLM is fine; the orchestrator wasn't constructed.

**Severity.** High UX impact. A user following PARTICIPANT_GUIDE cannot
submit ANY query without setting an env var that isn't mentioned anywhere
in the user-facing docs.

**Fix shipped this sprint.** PARTICIPANT_GUIDE Step 3 + new
"First-time-user reality check" section documents the env var explicitly.

### F3 — First query fails with "no content shards" on default install

**Symptom.** With orchestrator enabled, query reaches the gateway and the
quote endpoint (`POST /compute/forge/quote`) works correctly — but the
execution endpoint (`POST /compute/forge`) returns:

```
{"detail": "No content shards above the similarity threshold for this query.
            Upload relevant content to this node or refine the query."}
```

The error message correctly tells the user what to do, but the underlying
issue is fundamental: a brand-new node has no content of its own, and the
peer-discovery counter on a default install shows
`discovered_peer_count: 0` — there are no peers with content either.

**Severity.** Chicken-and-egg for new users. Vision §4 describes the ideal
case where content already exists on the network; in practice, a fresh
install hits zero content unless the user also uploads content themselves
(which has its own friction, see F4).

**Fix shipped this sprint.** New PARTICIPANT_GUIDE section "First-time-user
reality check" sets expectations honestly: a fresh install starts empty;
joining a content-populated network or self-seeding is required.

### F4 — Content upload fails on default node ("content_publisher unwired")

**Symptom.** Attempting to upload content to unblock F3:

```
POST /content/upload
{"detail": "Upload failed — upload_text returned None. Common causes:
            content_publisher unwired, _publish_content raised + was swallowed
            (check logs for 'Content publish failed'), or BitTorrent layer
            crashed mid-upload."}
```

The default node's `content_publisher` primitive isn't wired up. The
operator needs to take additional steps beyond `prsm node start` to enable
content upload. The error message is honest but the wiring path isn't
documented in PARTICIPANT_GUIDE.

**Severity.** Wiring-level gap. Not fixable with documentation alone —
needs either a default wiring change OR an explicit "enable content
upload" runbook for operators who want to host content.

**Status.** Documented as honest-scope in PARTICIPANT_GUIDE; the underlying
wiring fix is a separate sprint candidate.

### F5 — Quote endpoint path is `/compute/forge/quote`, not `/compute/quote`

**Symptom.** Following the "MCP tools" hint in PARTICIPANT_GUIDE, a user
probing the API directly tries `POST /compute/quote` (matching the tool
name `prsm_quote`) and gets a 404. The actual canonical path is
`POST /compute/forge/quote`.

**Severity.** Discoverability gap. MCP tools work fine (they have the
correct path internally); only operators probing the raw HTTP surface
hit this.

**Fix shipped this sprint.** Documented in the new "First-time-user
reality check" section's "Direct HTTP probes" sub-section.

### F6 — `/onboarding/` URL advertised in node-start log but returns 404

**Symptom.** Node startup logs print:

```
Node onboarding UI available
url=http://127.0.0.1:8000/onboarding/
```

But hitting `http://127.0.0.1:8000/onboarding/` returns HTTP 404.

**Severity.** Minor — a log line claims something that doesn't actually
work. The user's worst-case is "I followed the link in the log and got a
404; weird; moving on."

**Status.** Tracked here; deferred. The onboarding flow exists (sprint X
shipped onboarding endpoints) but the trailing-slash or root path
behaves differently than advertised. Either fix the log line or fix the
route.

---

## What's working (positive findings)

- ✅ `prsm setup --minimal` completes cleanly on existing config (re-prompt
  flow is unambiguous)
- ✅ `prsm node start` produces a clear status panel showing FTNS balance,
  P2P + API addresses, dashboard URL
- ✅ Bootstrap connection succeeds against the canonical fleet (sprints
  388–411 work paid off — `discovered_peer_count: 0` but `connected: 1,
  degraded: false`)
- ✅ `GET /balance` works end-to-end (read-side of the user-facing
  workflow is reliable)
- ✅ `GET /info` is rich and operator-useful — shows canonical addresses
  for all 12 on-chain contracts, network/chain_id, RPC host, bootstrap
  digest, query orchestrator state, agent forge wiring status
- ✅ `POST /compute/forge/quote` works for cost estimation (this is the
  critical "show user expected cost before they spend" path; even when
  execution fails for F3 reasons, the user can still see what they
  would have paid)
- ✅ Dashboard at `/` serves a real HTML page
- ✅ Sprint 408+'s observability stack is fully online on the node side

---

## What I'd recommend prioritizing

In order of user-impact-per-effort:

1. **PARTICIPANT_GUIDE Step 3 update** — fixes F1 + F2 + F3 in one
   doc-only change (this sprint).
2. **Default-wire content_publisher** — closes F4. Operators who follow
   the guide and want to host content shouldn't need a separate runbook.
3. **`/compute/quote` alias** — fix F5 by adding a route alias so the
   intuitive path works. 5-line code change.
4. **Onboarding URL** — fix F6 by either suppressing the log line or
   wiring the `/onboarding/` route properly.
5. **Vision §4 honest-scope note** — Vision §4 paints the ideal end-to-end
   workflow; consider adding a footnote ("Available once a node has
   content or has joined a content-populated peer network — see
   PARTICIPANT_GUIDE 'First-time-user reality check' for the bootstrap
   journey") so investors reading Vision aren't surprised by the gap
   between vision and default-install reality.

---

## Test environment

- macOS, repo at `/Users/ryneschultz/Documents/GitHub/PRSM`
- Existing identity at `~/.prsm/identity.json` (created 2026-02-18)
- FTNS balance starting at 173.05
- Bootstrap fleet 3/3 reachable per sprint 407 smoke test
- Node started via `python3 -m prsm.cli daemon start`
- Daemon listening on `127.0.0.1:8000`

---

## What this finding-set isn't

- Not a regression report. None of these break a workflow that previously
  worked end-to-end with this configuration.
- Not a blocker for shipped Phase 5 / §7 work. Those operate at a
  different layer and are independently verified.
- Not a claim that PRSM doesn't work. It works as designed; the
  default-install user experience just doesn't match the maximalist
  framing in PARTICIPANT_GUIDE without operator-level configuration.

The point of this report is to make that gap legible + actionable.
