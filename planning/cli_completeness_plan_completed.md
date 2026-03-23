# CLI Completeness Plan

## Overview

The PRSM CLI (`prsm/cli.py`) is built with Click and communicates with the FastAPI backend over
HTTP via httpx. It has solid coverage of auth, FTNS, IPFS storage, governance, marketplace, compute
jobs, and node management. This plan closes the remaining gaps — most critically BitTorrent (backend
just implemented, zero CLI surface) and the reputation system (central to PRSM's provenance royalty
model), followed by NWTN enhancements, model distillation, and monitoring.

All new commands follow the existing patterns exactly:
- HTTP calls via `httpx.Client`, never direct imports of backend services
- Rich tables/panels for output
- `✅ / ❌ / ⚠️` status indicators
- `click.prompt()` / `click.confirm()` for interactive flows
- Bearer token auth from `~/.prsm/credentials.json`
- Consistent error handling (401 → login hint, 402 → balance hint, 503 → service hint)

---

## Gap Summary

| Command Group | Current State | Priority |
|---------------|--------------|----------|
| `prsm torrent` | ❌ Zero coverage — backend fully built | **P1** |
| `prsm reputation` | ❌ Zero coverage — core to royalty system | **P1** |
| `prsm nwtn` | ⚠️ Only `prsm query` — needs reasoning traces, model mgmt | **P2** |
| `prsm distill` | ❌ Zero coverage — knowledge transfer is a key feature | **P2** |
| `prsm teacher` | ⚠️ Missing `eval`, `delete`, `list-runs` | **P2** |
| `prsm monitor` | ❌ Zero coverage — operators need observability | **P3** |
| `prsm workflow` | ❌ Zero coverage — workflow scheduling API exists | **P3** |
| `prsm team` | ❌ Zero coverage — collaboration API exists | **P3** |

---

## Phase 1 — BitTorrent Command Group (`prsm torrent`)

### Rationale

The BitTorrent backend (Phase 1–7 of the previous plan) is fully implemented and registered at
`/api/v1/torrents`. There is currently no way for a user to interact with it from the CLI at all.
This is the highest-priority gap.

### New command group: `prsm torrent`

Add a `torrent` Click group immediately below the existing `storage` group in `cli.py`.

---

#### `prsm torrent create <path>`

Create a new torrent from a local file or directory, begin seeding it, and register it on the PRSM
network.

```
Arguments:
  path                  Path to file or directory to torrent

Options:
  --name TEXT           Human-readable name (default: filename)
  --piece-length INT    Piece size in bytes [default: 262144]
  --provenance-id TEXT  Link to existing PRSM provenance record
  --no-seed             Create torrent file only, do not start seeding
  --output-torrent PATH Save .torrent file to disk at this path

Output:
  Panel showing:
    Infohash: <40-char hex>
    Name:     <name>
    Size:     <human-readable>
    Pieces:   <N> × <piece_length>KB
    Magnet:   magnet:?xt=urn:btih:<infohash>&dn=<name>
    Status:   Seeding ✅  (or Created only)

API: POST /api/v1/torrents/create
```

---

#### `prsm torrent add <source>`

Add an existing torrent to the local node from a magnet URI or a `.torrent` file path.

```
Arguments:
  source                Magnet URI ("magnet:?xt=...") or path to .torrent file

Options:
  --save-path PATH      Where to save downloaded files [default: ~/.prsm/torrents]
  --seed-mode           Skip downloading — assume data already present, just seed
  --download            Begin downloading immediately (default: add only)

Output:
  ✅ Torrent added: <infohash[:16]>...
  Name:   <name>
  Status: Added / Downloading / Seeding

API: POST /api/v1/torrents/add
```

---

#### `prsm torrent list`

List all torrents known to this node — seeding, downloading, and available on the network.

```
Options:
  --seeding             Show only torrents this node is seeding
  --available           Show only network-announced torrents (from gossip)
  --limit INT           Max results [default: 50]

Output:
  Rich table:
    Infohash (16 chars)  |  Name  |  Size  |  State  |  Progress  |  Peers  |  ↑ Rate  |  ↓ Rate

API: GET /api/v1/torrents
```

---

#### `prsm torrent status <infohash>`

Show detailed live status for a specific torrent.

```
Arguments:
  infohash              Full or prefix infohash

Output:
  Panel:
    Infohash:     <full 40 chars>
    Name:         <name>
    Size:         <human-readable>
    State:        Seeding / Downloading / Paused / Error
    Progress:     [████████░░] 80%   (progress bar)
    Download:     <rate> KB/s  (ETA: <time>)
    Upload:       <rate> KB/s
    Peers:        <N> connected
    Uploaded:     <total human-readable>
    Downloaded:   <total human-readable>
    Magnet URI:   magnet:?xt=...
    IPFS CID:     <cid if .torrent pinned to IPFS>

API: GET /api/v1/torrents/{infohash}
```

---

#### `prsm torrent seed <infohash>`

Start seeding a torrent that this node has data for but is not currently seeding.

```
Arguments:
  infohash              Torrent to start seeding

Output:
  ✅ Now seeding: <infohash[:16]>...
  Announced to PRSM network.

API: POST /api/v1/torrents/{infohash}/seed
```

---

#### `prsm torrent unseed <infohash>`

Stop seeding a torrent (withdraw from swarm and stop FTNS reward accrual).

```
Arguments:
  infohash              Torrent to stop seeding

Options:
  --delete-files        Also delete local data files [requires --yes]
  --yes                 Skip confirmation prompt

Output:
  ✅ Stopped seeding: <infohash[:16]>...
  ⚠️  Files retained at <path>  (or deleted)

API: DELETE /api/v1/torrents/{infohash}/seed
```

---

#### `prsm torrent download <infohash>`

Download a torrent's content. Polls progress until completion or timeout.

```
Arguments:
  infohash              Torrent to download

Options:
  --save-path PATH      Destination directory [default: ~/.prsm/torrents]
  --timeout FLOAT       Max seconds to wait [default: 3600]
  --no-wait             Fire-and-forget (return request_id immediately)

Output (interactive):
  Resolving torrent... ✅
  Starting download...
  [████████████░░░░░░░░] 60%  4.2 MB/s  ETA: 1m 23s
  ✅ Download complete: <path>
  Downloaded: 1.2 GB in 4m 37s

API: POST /api/v1/torrents/{infohash}/download
     GET  /api/v1/torrents/{infohash}/download/{request_id}  (polling)
```

---

#### `prsm torrent peers <infohash>`

List peers currently connected for a torrent.

```
Arguments:
  infohash              Torrent to inspect

Output:
  Rich table:
    IP:Port  |  Client  |  Downloaded  |  Uploaded  |  Seed?

API: GET /api/v1/torrents/{infohash}/peers
```

---

#### `prsm torrent stats`

Show aggregate BitTorrent stats for this node (total seeding, upload/download bandwidth, FTNS
earned from seeding).

```
Output:
  Panel:
    Active torrents:   <N> seeding, <M> downloading
    Total uploaded:    <human-readable>
    Total downloaded:  <human-readable>
    Upload rate:       <KB/s>
    Download rate:     <KB/s>
    FTNS earned (seeding): <amount>

API: GET /api/v1/torrents/stats
```

---

## Phase 2 — Reputation Command Group (`prsm reputation`)

### Rationale

Reputation is core to PRSM's trust model — it determines provenance weight, royalty multipliers,
and governance influence. There is currently zero CLI surface for it.

### New command group: `prsm reputation`

---

#### `prsm reputation score [user-id]`

Show reputation score and breakdown for a user (defaults to current authenticated user).

```
Arguments:
  user-id               Optional: target user (default: current user)

Output:
  Panel:
    User:              <user-id>
    Overall Score:     87.3 / 100
    ──────────────────────────────
    Contribution:      92.1   (data/model uploads, citations)
    Reliability:       88.4   (uptime, proof-of-storage pass rate)
    Governance:        79.0   (proposal quality, voting participation)
    Marketplace:       91.2   (buyer/seller ratings)
    ──────────────────────────────
    Tier:              Contributor
    Percentile:        Top 12%

API: GET /api/v1/reputation/{user_id}
```

---

#### `prsm reputation leaderboard`

Show the top contributors on the network by reputation score.

```
Options:
  --category TEXT       Filter by: contribution, reliability, governance, marketplace
  --limit INT           Number of entries [default: 20]

Output:
  Rich table:
    Rank  |  User ID  |  Score  |  Tier  |  Uploads  |  Uptime  |  Since

API: GET /api/v1/reputation/leaderboard
```

---

#### `prsm reputation history [user-id]`

Show reputation change history — what actions caused score changes.

```
Arguments:
  user-id               Optional: target user (default: current user)

Options:
  --limit INT           [default: 25]

Output:
  Rich table:
    Date  |  Event  |  Category  |  Δ Score  |  Running Total

API: GET /api/v1/reputation/{user_id}/history
```

---

#### `prsm reputation report <user-id>`

File a reputation report against a node (e.g. fake seeder, bad data, governance abuse).

```
Arguments:
  user-id               Node to report

Options:
  --reason TEXT         One of: fake_seeder, bad_data, spam, governance_abuse, other
  --evidence TEXT       CID or transaction ID supporting the report
  --description TEXT    Free-form explanation

Output:
  Confirmation prompt → ✅ Report filed. Report ID: <id>
  PRSM governance will review within 72 hours.

API: POST /api/v1/reputation/reports
```

---

#### `prsm reputation reports`

List reputation reports filed by or against the current user.

```
Options:
  --filed               Reports I filed (default)
  --received            Reports filed against me
  --status TEXT         Filter: pending, reviewed, dismissed, upheld
  --limit INT           [default: 20]

Output:
  Rich table:
    Report ID  |  Target/Reporter  |  Reason  |  Status  |  Filed At

API: GET /api/v1/reputation/reports
```

---

## Phase 3 — NWTN Enhancements (`prsm nwtn`)

### Rationale

`prsm query` exists but only submits a basic query. NWTN is the core reasoning orchestrator and
deserves a dedicated command group that exposes reasoning traces, model selection, and pipeline
health.

### New command group: `prsm nwtn`

The existing `prsm query` command should remain for backward compatibility. The new group adds
depth.

---

#### `prsm nwtn query <query>`

Enhanced replacement for `prsm query` with more control options.

```
Arguments:
  query                 The question or task

Options:
  --model TEXT          Model backend: auto, gpt-4o, claude-3-5-sonnet, ollama/<name>
  --context TEXT        Additional context JSON or plain text
  --user-id TEXT        Override user ID
  --trace               Show full reasoning trace after response
  --stream              Stream output token by token (if API supports it)
  --budget FLOAT        Max FTNS to spend on this query
  --format TEXT         Output format: text (default), json, markdown

Output:
  Spinner while processing...
  ─────────── Response ───────────
  <response text>
  ────────────────────────────────
  Tokens used: 1,247   Cost: 0.031 FTNS   Time: 2.4s

  (with --trace):
  ─────── Reasoning Trace ────────
  Step 1: [retrieval] Searched IPFS index for <query_terms>
  Step 2: [reasoning] Applied deductive chain...
  Step 3: [synthesis] Combined 3 sources...

API: POST /api/v1/nwtn/query  (or existing sessions/query endpoint)
```

---

#### `prsm nwtn models`

List available model backends and their status.

```
Options:
  --available           Show only currently reachable backends
  --detail              Include model metadata (context window, cost/token, etc.)

Output:
  Rich table:
    Backend  |  Model  |  Status  |  Latency  |  Cost/1k tokens  |  Context Window

API: GET /api/v1/nwtn/models  (or compute model_manager endpoint)
```

---

#### `prsm nwtn health`

Show NWTN pipeline health — all five reasoning layers and their status.

```
Output:
  Panel:
    Layer 1 - Retrieval:       ✅ Healthy  (avg 120ms)
    Layer 2 - Reasoning:       ✅ Healthy  (avg 340ms)
    Layer 3 - Synthesis:       ⚠️  Degraded (avg 1.2s — 2× normal)
    Layer 4 - Validation:      ✅ Healthy
    Layer 5 - Output:          ✅ Healthy
    ──────────────────────────────────────
    Pipeline overall:          ⚠️  Degraded

API: GET /api/v1/nwtn/health  (or monitoring/health endpoint)
```

---

#### `prsm nwtn sessions`

List recent NWTN query sessions for the current user.

```
Options:
  --limit INT           [default: 20]
  --include-cost        Show FTNS cost per session

Output:
  Rich table:
    Session ID  |  Query (truncated)  |  Model  |  Tokens  |  Cost  |  Time  |  Date

API: GET /api/v1/sessions  (existing session_api)
```

---

## Phase 4 — Model Distillation (`prsm distill`)

### Rationale

Model distillation (teacher → student knowledge transfer) is a key PRSM feature with a fully
implemented backend. No CLI surface exists.

### New command group: `prsm distill`

---

#### `prsm distill run <teacher-id>`

Start a distillation job from a teacher model.

```
Arguments:
  teacher-id            Source teacher model ID

Options:
  --student-arch TEXT   Target architecture (default: auto-select smallest viable)
  --dataset-cid TEXT    IPFS CID of training dataset (overrides teacher's default)
  --epochs INT          Training epochs [default: 3]
  --output-name TEXT    Name for the resulting student model
  --budget FLOAT        Max FTNS to spend
  --follow              Stream progress logs until completion

Output:
  ✅ Distillation job started: <job_id>
  Teacher:  <teacher-id>
  Student:  <student-arch>
  Dataset:  <cid>
  Use: prsm distill status <job_id>  to monitor

API: POST /api/v1/distillation/jobs
```

---

#### `prsm distill status <job-id>`

Poll distillation job status.

```
Arguments:
  job-id                Distillation job ID

Output:
  Panel:
    Job ID:       <id>
    Status:       Running / Completed / Failed
    Progress:     [██████░░░░] 60%   Epoch 2/3
    Teacher:      <teacher-id>
    Student:      <student-arch>
    Loss:         0.234 (↓ from 0.412)
    Started:      <timestamp>
    ETA:          ~12 minutes

API: GET /api/v1/distillation/jobs/{job_id}
```

---

#### `prsm distill list`

List distillation jobs for the current user.

```
Options:
  --status TEXT         Filter: running, completed, failed [default: all]
  --limit INT           [default: 20]

Output:
  Rich table:
    Job ID  |  Teacher  |  Student Arch  |  Status  |  Progress  |  Cost  |  Date

API: GET /api/v1/distillation/jobs
```

---

#### `prsm distill cancel <job-id>`

Cancel a running distillation job.

```
Arguments:
  job-id                Job to cancel

Options:
  --yes                 Skip confirmation

API: DELETE /api/v1/distillation/jobs/{job_id}
```

---

#### `prsm distill models`

List student models produced by completed distillation jobs.

```
Options:
  --limit INT           [default: 20]

Output:
  Rich table:
    Model ID  |  Name  |  Teacher Source  |  Architecture  |  Size  |  Quality Score  |  Date

API: GET /api/v1/distillation/models
```

---

## Phase 5 — Teacher Model Improvements

The existing `prsm teacher` group is missing three commands.

### `prsm teacher delete <teacher-id>`

```
Arguments:
  teacher-id            Teacher model to delete

Options:
  --yes                 Skip confirmation

Output:
  ✅ Teacher model <id> deleted.
  ⚠️  Note: Active distillation jobs referencing this teacher will be cancelled.

API: DELETE /api/v1/teachers/{teacher_id}
```

---

### `prsm teacher list-runs <teacher-id>`

List all training runs for a teacher model.

```
Arguments:
  teacher-id            Teacher to inspect

Options:
  --limit INT           [default: 20]

Output:
  Rich table:
    Run ID  |  Status  |  Epochs  |  Loss  |  Duration  |  Date

API: GET /api/v1/teachers/{teacher_id}/runs
```

---

### `prsm teacher eval <teacher-id>`

Run a quick benchmark evaluation against a teacher model.

```
Arguments:
  teacher-id            Teacher to evaluate

Options:
  --benchmark TEXT      Benchmark suite: default, math, code, science [default: default]
  --samples INT         Number of eval samples [default: 100]

Output:
  Running evaluation... (spinner)
  ──── Evaluation Results ─────
  Teacher:     <teacher-id>
  Benchmark:   default
  Score:       82.4 / 100
  Accuracy:    84.1%
  Latency:     avg 310ms
  Cost:        0.15 FTNS
  Rank:        Top 23% in domain

API: POST /api/v1/teachers/{teacher_id}/eval
```

---

## Phase 6 — Monitoring (`prsm monitor`)

### Rationale

Node operators and network participants need visibility into system health without needing to access
Grafana or the Streamlit dashboard.

### New command group: `prsm monitor`

---

#### `prsm monitor health`

Full health check across all PRSM subsystems.

```
Options:
  --watch               Refresh every 5 seconds (like `watch`)
  --json                Output as JSON for scripting

Output:
  Panel:
    API Server:       ✅ Healthy    (v0.2.1)
    Database:         ✅ Healthy    (42ms)
    IPFS:             ✅ Healthy    (3 peers)
    Redis:            ✅ Healthy
    BitTorrent:       ✅ Healthy    (8 active torrents)
    NWTN Pipeline:    ⚠️  Degraded  (Layer 3 slow)
    P2P Network:      ✅ Healthy    (14 peers)
    FTNS Service:     ✅ Healthy

API: GET /api/v1/health  (existing health_api.py)
     GET /api/v1/health/liveness
     GET /api/v1/health/readiness
```

---

#### `prsm monitor metrics`

Show key performance metrics for the local node.

```
Options:
  --period TEXT         Time window: 1h, 6h, 24h, 7d [default: 1h]
  --watch               Auto-refresh every 10 seconds

Output:
  Panel:
    Period: Last 1h
    ────────── Compute ──────────
    Queries served:      147
    Avg query latency:   280ms
    CPU usage:           34%
    Memory usage:        2.1 GB / 8 GB
    ────────── Network ──────────
    P2P bandwidth ↑:     14.2 MB
    P2P bandwidth ↓:     8.7 MB
    BT seeded:           230 MB
    BT downloaded:       45 MB
    ────────── Economy ──────────
    FTNS earned:         +1.42
    FTNS spent:          -0.31

API: GET /api/v1/monitoring/metrics
```

---

#### `prsm monitor logs`

Stream or retrieve recent structured log output.

```
Options:
  --level TEXT          Filter: debug, info, warning, error [default: info]
  --service TEXT        Filter by service: api, nwtn, ipfs, bittorrent, node
  --limit INT           Lines to show [default: 50]
  --follow              Stream new log entries (Ctrl+C to stop)

Output:
  Formatted log lines with timestamps, levels, and service tags

API: GET /api/v1/monitoring/logs  (or existing security_logging_api)
```

---

## Phase 7 — Workflow Scheduling (`prsm workflow`)

### Rationale

The workflow scheduling API (`workflow_scheduling_api.py`) is fully implemented but has no CLI
surface. Researchers need to automate recurring jobs (e.g., nightly dataset ingestion → distillation
→ provenance registration).

### New command group: `prsm workflow`

---

#### `prsm workflow create`

Create a new automated workflow.

```
Options:
  --name TEXT           Workflow name
  --description TEXT    Description
  --trigger TEXT        cron expression or "manual" [e.g. "0 2 * * *" = 2am daily]
  --steps-file PATH     JSON file describing workflow steps
  --budget FLOAT        Max FTNS per run

Output:
  ✅ Workflow created: <workflow_id>
  Next run: <timestamp>  (if cron trigger)

API: POST /api/v1/workflows
```

---

#### `prsm workflow list`

List all workflows for the current user.

```
Options:
  --status TEXT         active, paused, failed [default: all]
  --limit INT           [default: 20]

Output:
  Rich table:
    ID  |  Name  |  Trigger  |  Status  |  Last Run  |  Next Run  |  Runs

API: GET /api/v1/workflows
```

---

#### `prsm workflow run <workflow-id>`

Manually trigger a workflow run.

```
Arguments:
  workflow-id           Workflow to run

Options:
  --follow              Stream execution logs until completion

API: POST /api/v1/workflows/{workflow_id}/run
```

---

#### `prsm workflow logs <workflow-id>`

Show execution logs for a workflow.

```
Arguments:
  workflow-id           Workflow to inspect

Options:
  --run-id TEXT         Specific run (default: most recent)
  --limit INT           Log lines [default: 100]

API: GET /api/v1/workflows/{workflow_id}/runs/{run_id}/logs
```

---

## Implementation Notes

### File to modify: `prsm/cli.py`

All new commands are added directly to the existing `cli.py`. Do not create separate files — the
existing pattern is a single `cli.py` with all groups defined inline. Creating separate modules
would break the current import structure and require restructuring the entry point.

### Authentication pattern (copy exactly from existing commands)

```python
def _get_auth_headers(api_url: str) -> Dict[str, str]:
    """Load credentials and return auth headers."""
    creds_path = Path("~/.prsm/credentials.json").expanduser()
    if not creds_path.exists():
        console.print("[red]Not logged in. Run: prsm login[/red]")
        raise SystemExit(1)
    creds = json.loads(creds_path.read_text())
    return {"Authorization": f"Bearer {creds['token']}"}
```

### HTTP error handling pattern (copy exactly from existing commands)

```python
if response.status_code == 401:
    console.print("[red]Session expired. Run: prsm login[/red]")
    raise SystemExit(1)
elif response.status_code == 402:
    console.print("[yellow]Insufficient FTNS balance.[/yellow]")
    raise SystemExit(1)
elif response.status_code == 503:
    console.print("[yellow]Service unavailable. Is the relevant backend running?[/yellow]")
    raise SystemExit(1)
elif not response.is_success:
    console.print(f"[red]Error {response.status_code}: {response.text}[/red]")
    raise SystemExit(1)
```

### Progress polling pattern (for download, distill, train)

```python
with Progress() as progress:
    task = progress.add_task("Downloading...", total=100)
    while True:
        r = client.get(f"{api_url}/api/v1/torrents/{infohash}/download/{request_id}", ...)
        data = r.json()
        progress.update(task, completed=data["progress"] * 100)
        if data["status"] in ("completed", "failed"):
            break
        time.sleep(2)
```

### Output sizing conventions

```python
# Human-readable bytes
def _fmt_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"

# Truncated IDs for tables
infohash[:16] + "..."
```

---

## New Commands Summary

| Group | Command | API Endpoint |
|-------|---------|--------------|
| `torrent` | `create` | POST /api/v1/torrents/create |
| `torrent` | `add` | POST /api/v1/torrents/add |
| `torrent` | `list` | GET /api/v1/torrents |
| `torrent` | `status` | GET /api/v1/torrents/{infohash} |
| `torrent` | `seed` | POST /api/v1/torrents/{infohash}/seed |
| `torrent` | `unseed` | DELETE /api/v1/torrents/{infohash}/seed |
| `torrent` | `download` | POST /api/v1/torrents/{infohash}/download |
| `torrent` | `peers` | GET /api/v1/torrents/{infohash}/peers |
| `torrent` | `stats` | GET /api/v1/torrents/stats |
| `reputation` | `score` | GET /api/v1/reputation/{user_id} |
| `reputation` | `leaderboard` | GET /api/v1/reputation/leaderboard |
| `reputation` | `history` | GET /api/v1/reputation/{user_id}/history |
| `reputation` | `report` | POST /api/v1/reputation/reports |
| `reputation` | `reports` | GET /api/v1/reputation/reports |
| `nwtn` | `query` | POST /api/v1/nwtn/query |
| `nwtn` | `models` | GET /api/v1/nwtn/models |
| `nwtn` | `health` | GET /api/v1/nwtn/health |
| `nwtn` | `sessions` | GET /api/v1/sessions |
| `distill` | `run` | POST /api/v1/distillation/jobs |
| `distill` | `status` | GET /api/v1/distillation/jobs/{id} |
| `distill` | `list` | GET /api/v1/distillation/jobs |
| `distill` | `cancel` | DELETE /api/v1/distillation/jobs/{id} |
| `distill` | `models` | GET /api/v1/distillation/models |
| `teacher` | `delete` | DELETE /api/v1/teachers/{id} |
| `teacher` | `list-runs` | GET /api/v1/teachers/{id}/runs |
| `teacher` | `eval` | POST /api/v1/teachers/{id}/eval |
| `monitor` | `health` | GET /api/v1/health |
| `monitor` | `metrics` | GET /api/v1/monitoring/metrics |
| `monitor` | `logs` | GET /api/v1/monitoring/logs |
| `workflow` | `create` | POST /api/v1/workflows |
| `workflow` | `list` | GET /api/v1/workflows |
| `workflow` | `run` | POST /api/v1/workflows/{id}/run |
| `workflow` | `logs` | GET /api/v1/workflows/{id}/runs/{run_id}/logs |

**Total new commands: 33**

---

## Implementation Order

```
Phase 1: prsm torrent group (9 commands)         — closes BitTorrent CLI gap immediately
Phase 2: prsm reputation group (5 commands)      — core to provenance/royalty UX
Phase 3: prsm nwtn group (4 commands)            — elevates core value prop
Phase 4: prsm distill group (5 commands)         — exposes key PRSM feature
Phase 5: prsm teacher improvements (3 commands)  — completes existing group
Phase 6: prsm monitor group (3 commands)         — operator observability
Phase 7: prsm workflow group (4 commands)        — automation/scheduling
```

Each phase adds a self-contained command group and can be tested independently.
A phase is complete when all commands reach the API and return well-formatted output for both
success and all documented error cases (401, 402, 404, 503).

---

## Implementation Completion Summary

**Completion Date:** 2026-03-23

### Work Completed

All 7 phases implemented, adding 33 new commands to the PRSM CLI.

#### Phase 1 - BitTorrent Commands (`prsm torrent`) ✅
Added 9 commands to `prsm/cli.py`:
- `prsm torrent create <path>` - Create torrent from file/directory and seed
- `prsm torrent add <source>` - Add existing torrent from magnet or .torrent file
- `prsm torrent list` - List all torrents (seeding, downloading, available)
- `prsm torrent status <infohash>` - Detailed status with progress bar
- `prsm torrent seed <infohash>` - Start seeding a torrent
- `prsm torrent unseed <infohash>` - Stop seeding (with optional file deletion)
- `prsm torrent download <infohash>` - Download with progress polling
- `prsm torrent peers <infohash>` - List connected peers
- `prsm torrent stats` - Aggregate BitTorrent stats for node

#### Phase 2 - Reputation Commands (`prsm reputation`) ✅
Added 5 commands:
- `prsm reputation score [user-id]` - Show reputation breakdown with tier/percentile
- `prsm reputation leaderboard` - Top contributors by score
- `prsm reputation history [user-id]` - Reputation change events
- `prsm reputation report <user-id>` - File a reputation report
- `prsm reputation reports` - List reports filed/received

#### Phase 3 - NWTN Commands (`prsm nwtn`) ✅
Added 4 commands:
- `prsm nwtn query <query>` - Enhanced query with model/budget/trace options
- `prsm nwtn models` - List available model backends
- `prsm nwtn health` - Pipeline health across 5 reasoning layers
- `prsm nwtn sessions` - Recent query sessions

#### Phase 4 - Distillation Commands (`prsm distill`) ✅
Added 5 commands:
- `prsm distill run <teacher-id>` - Start distillation job
- `prsm distill status <job-id>` - Poll job progress with progress bar
- `prsm distill list` - List distillation jobs
- `prsm distill cancel <job-id>` - Cancel running job
- `prsm distill models` - List distilled student models

#### Phase 5 - Teacher Command Additions ✅
Added 3 commands to existing `prsm teacher` group:
- `prsm teacher delete <teacher-id>` - Delete a teacher model
- `prsm teacher list-runs <teacher-id>` - Training run history
- `prsm teacher eval <teacher-id>` - Benchmark evaluation

#### Phase 6 - Monitoring Commands (`prsm monitor`) ✅
Added 3 commands:
- `prsm monitor health` - Full health check with `--watch` option
- `prsm monitor metrics` - Node performance metrics with `--watch` option
- `prsm monitor logs` - Structured log output with `--follow` streaming

#### Phase 7 - Workflow Commands (`prsm workflow`) ✅
Added 4 commands:
- `prsm workflow create` - Create automated workflow with cron trigger
- `prsm workflow list` - List workflows
- `prsm workflow run <workflow-id>` - Manually trigger workflow
- `prsm workflow logs <workflow-id>` - Execution logs

### Helper Functions Added
- `_fmt_bytes(n: int) -> str` - Human-readable byte formatting
- `_fmt_rate(n: float) -> str` - Human-readable rate formatting

### Design Patterns Followed
- HTTP calls via `httpx.Client`, never direct backend imports
- Rich tables/panels for output
- `✅ / ❌ / ⚠️` status indicators
- `click.prompt()` / `click.confirm()` for interactive flows
- Bearer token auth from `~/.prsm/credentials.json`
- Consistent error handling (401 → login hint, 402 → balance hint, 503 → service hint)

### Files Modified
- `prsm/cli.py` - Added all new command groups and helper functions

### Verification
```
Available command groups:
  - compute, dashboard, db-*, distill, ftns, governance, init, login, logout
  - marketplace, monitor, node, nwtn, query, reputation, serve, status
  - storage, teacher, torrent, workflow

torrent commands: ['create', 'add', 'list', 'status', 'seed', 'unseed', 'download', 'peers', 'stats']
reputation commands: ['score', 'leaderboard', 'history', 'report', 'reports']
nwtn commands: ['query', 'models', 'health', 'sessions']
distill commands: ['run', 'status', 'list', 'cancel', 'models']
monitor commands: ['health', 'metrics', 'logs']
workflow commands: ['create', 'list', 'run', 'logs']
teacher commands: ['list', 'create', 'train', 'status', 'cancel-training', 'delete', 'list-runs', 'eval']
```

### Total New Commands: 33

### Status
All 7 phases complete. The CLI now has full coverage of BitTorrent, Reputation, NWTN, Distillation, Monitoring, and Workflow features.
