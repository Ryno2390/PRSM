# PRSM Setup Experience — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build a polished, Hermes/OpenClaw-quality interactive setup experience for PRSM that takes a user from `pip install prsm-network` through full node configuration, background daemon operation, and AI-native skill integration.

**Architecture:** A modular setup system with a shared UI toolkit (Rich-based theming, ASCII art, interactive prompts), a unified `prsm setup` first-run wizard, a `prsm config` management command, daemon lifecycle management, and a user-facing skill package system. Each component is a separate module under `prsm/cli/` to keep the 6400-line `cli.py` from growing further.

**Tech Stack:** Python 3.10+, Rich (Console, Panel, Table, Text, Progress, Prompt), click (CLI framework — already in use), prompt_toolkit (for advanced multi-select/arrow-key menus), asyncio (daemon), psutil (resource detection — already in use).

**Art Direction:** Black and white monochromatic design — clean, minimal, cutting-edge ASCII art. Match the polish level of Hermes and OpenClaw's setup wizards but with PRSM's own identity: geometric, isometric, bold strokes. The PRSM logo features a left-pointing chevron, a 3D rectangular form, and a bottom accent line — all sharp angles, no curves. Reference logos at docs/assets/PRSM_Logo_Light.png and PRSM_Logo_Dark.png. No color palette — use white/gray on dark terminals, black/gray on light terminals. Status indicators (✓/⚠/✗) use semantic colors only where functionally necessary.

---

## Phase 0: UI Toolkit & Brand Identity

### Task 0.1: Create PRSM CLI Package Structure

**Objective:** Extract CLI into a proper package so we can add modules without bloating cli.py further.

**Files:**
- Create: `prsm/cli/__init__.py`
- Create: `prsm/cli/theme.py`
- Create: `prsm/cli/ui.py`
- Create: `prsm/cli/banner.py`

**Details:**

`prsm/cli_modules/theme.py` — PRSM's visual identity (monochromatic):
```python
"""PRSM CLI Theme — Black & white monochromatic design."""
from dataclasses import dataclass

@dataclass(frozen=True)
class PRSMTheme:
    """Monochromatic theme — geometric, minimal, sharp."""
    # Primary palette — black & white with grays
    primary: str = "bold white"
    secondary: str = "white"
    accent: str = "bold"
    
    # Text hierarchy
    heading: str = "bold white"
    subheading: str = "white"
    body: str = "default"
    muted: str = "dim"
    dim: str = "dim italic"
    
    # Semantic — color used ONLY for functional status
    success: str = "green"
    warning: str = "yellow"
    error: str = "red"
    info: str = "dim"
    
    # Interactive elements
    prompt: str = "bold"
    selected: str = "bold white"
    unselected: str = "dim"
    
    # Borders & lines
    border: str = "dim white"
    rule: str = "dim"

THEME = PRSMTheme()

# Status indicators — minimal, geometric
ICONS = {
    "success": "✓",
    "warning": "⚠",
    "error": "✗",
    "info": "→",
    "bullet": "▸",
    "arrow": "›",
    "check": "■",
    "uncheck": "□",
    "step": "◆",
    "network": "⬡",
    "compute": "⚡",
    "storage": "▪",
    "prism": "◇",
    "bar_full": "█",
    "bar_empty": "░",
    "line": "─",
    "corner": "┐",
}
```

`prsm/cli_modules/banner.py` — ASCII art and branding (monochromatic, geometric):
```python
"""PRSM ASCII banner — monochromatic geometric design.

The PRSM logo features a left-pointing chevron, a 3D rectangular form,
and a bottom accent line. All sharp angles, no curves. Isometric feel.
Reference: docs/assets/PRSM_Logo_Dark.png / PRSM_Logo_Light.png
"""

# Geometric isometric logo mark — inspired by the actual PRSM logo
# Left chevron + right 3D block + bottom accent
PRSM_ICON = r"""
       ╱╲       ┌──┐
      ╱  ╲      │  │
     ╱    ╲─────┤  │
    ╱     ╱     │  │
   ╱     ╱      └──┘
  ╱     ╱
 ╱     ╱
╱─────╱
"""

# Full banner: icon + wordmark
PRSM_BANNER = r"""
       ╱╲       ┌──┐
      ╱  ╲      │  │
     ╱    ╲─────┤  │
    ╱     ╱     │  │
   ╱     ╱      └──┘
  ╱     ╱
 ╱     ╱
╱─────╱

  P  R  S  M
"""

# Compact version for narrow terminals (<60 cols)
PRSM_BANNER_COMPACT = "◇ P R S M"

# Minimal one-liner
PRSM_INLINE = "◇ PRSM"

TAGLINES = [
    "Decentralized AI infrastructure.",
    "Your compute. Your data. Your network.",
    "Intelligence, distributed.",
    "The open AI network.",
    "Compute without borders.",
]

# Thin geometric separator
RULE = "─" * 48
```

`prsm/cli_modules/ui.py` — Shared UI components (monochromatic):
```python
"""PRSM CLI UI toolkit — monochromatic, geometric, minimal."""
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
import click
from .theme import THEME, ICONS
from .banner import PRSM_BANNER, PRSM_BANNER_COMPACT, TAGLINES, RULE
import random, shutil

console = Console()

def show_banner(version: str = "0.1.0"):
    """Display the PRSM monochromatic banner."""
    width = shutil.get_terminal_size().columns
    if width < 60:
        banner_text = PRSM_BANNER_COMPACT
    else:
        banner_text = PRSM_BANNER
    
    styled = Text()
    styled.append(banner_text.strip(), style=THEME.primary)
    
    tagline = random.choice(TAGLINES)
    styled.append(f"\n\n  {tagline}", style=THEME.muted)
    styled.append(f"\n  v{version}", style=THEME.dim)
    
    console.print(Panel(
        styled,
        border_style=THEME.border,
        padding=(1, 2),
    ))

def section(title: str, icon: str = "bullet"):
    """Print a section header — clean, geometric."""
    ico = ICONS.get(icon, icon)
    console.print(f"\n  {ico} {title}", style=THEME.heading)
    console.print(f"    {'─' * len(title)}", style=THEME.rule)

def step_header(step: int, total: int, title: str):
    """Step progress: ◆ Step 1/7 ━━━━━━░░░░░ Title"""
    filled = int((step / total) * 20)
    bar = "━" * filled + "░" * (20 - filled)
    console.print(f"\n  ◆ Step {step}/{total}  {bar}", style=THEME.primary)
    console.print(f"    {title}", style=THEME.heading)
    console.print(f"    {'─' * len(title)}", style=THEME.rule)

def success(msg: str):
    console.print(f"    {ICONS['success']} {msg}", style=THEME.success)

def warning(msg: str):
    console.print(f"    {ICONS['warning']} {msg}", style=THEME.warning)

def error(msg: str):
    console.print(f"    {ICONS['error']} {msg}", style=THEME.error)

def info(msg: str):
    console.print(f"    {ICONS['info']} {msg}", style=THEME.muted)

def muted(msg: str):
    console.print(f"      {msg}", style=THEME.dim)

def prompt_text(message: str, default=None, **kwargs) -> str:
    """Clean text prompt — no color noise."""
    return click.prompt(
        f"    {ICONS['arrow']} {message}",
        default=default, **kwargs
    )

def prompt_choice(message: str, choices: list[dict], default: int = 0) -> str:
    """Selection menu — choices listed, then click.Choice input."""
    console.print(f"\n    {message}", style=THEME.heading)
    for i, c in enumerate(choices):
        marker = ICONS["check"] if i == default else ICONS["uncheck"]
        label = c["label"]
        hint = c.get("hint", "")
        if hint:
            console.print(f"      {marker} {label}", style=THEME.primary)
            console.print(f"        {hint}", style=THEME.dim)
        else:
            console.print(f"      {marker} {label}", style=THEME.primary)
    
    valid = [c["value"] for c in choices]
    return click.prompt(
        f"    {ICONS['arrow']} Choose",
        type=click.Choice(valid, case_sensitive=False),
        default=valid[default]
    )

def prompt_confirm(message: str, default: bool = True) -> bool:
    return click.confirm(f"    {ICONS['arrow']} {message}", default=default)

def prompt_number(message: str, default: int = 50, min_val: int = 0, max_val: int = 100) -> int:
    return click.prompt(
        f"    {ICONS['arrow']} {message}",
        type=click.IntRange(min_val, max_val),
        default=default
    )

def summary_panel(title: str, items: dict):
    """Display a summary panel — monochromatic with dim border."""
    table = Table.grid(padding=(0, 3))
    table.add_column(style=THEME.muted)
    table.add_column(style=THEME.primary)
    for k, v in items.items():
        table.add_row(k, str(v))
    console.print(Panel(
        table,
        title=f"  {ICONS['success']} {title}  ",
        title_align="left",
        border_style=THEME.border,
        padding=(1, 2),
    ))

def resource_bar(label: str, value: int, max_val: int = 100, unit: str = "%"):
    """Visual resource bar — monochrome with semantic color only at threshold."""
    bar_width = 30
    filled = int((value / max_val) * bar_width)
    bar = ICONS["bar_full"] * filled + ICONS["bar_empty"] * (bar_width - filled)
    # Only use color when dangerously high
    if value >= 90:
        style = THEME.error
    elif value >= 75:
        style = THEME.warning
    else:
        style = THEME.body
    console.print(f"      {label:>12}  [{style}]{bar}[/]  {value}{unit}")

def progress_context(description: str = "Working..."):
    """Rich progress with minimal styling."""
    return Progress(
        SpinnerColumn("dots", style=THEME.muted),
        TextColumn(f"  {description}", style=THEME.body),
        TimeElapsedColumn(),
        console=console,
    )
```

**Verification:** `python -c "from prsm.cli.theme import THEME; print(THEME.brand_primary)"`

---

### Task 0.2: Wire New CLI Package into Existing cli.py

**Objective:** Make cli.py import from the new package without breaking anything.

**Files:**
- Modify: `prsm/cli/__init__.py` (export main, console)
- Modify: `prsm/cli.py` → keep as entry point, import from `prsm/cli/` modules

**Details:**
- `prsm/cli/__init__.py` should re-export `main` from the existing `prsm/cli.py` (renamed to `prsm/cli/main.py`) so the pyproject.toml entry point `prsm.cli:main` still works
- OR: simpler approach — keep `prsm/cli.py` as-is, have `prsm/cli/` as `prsm/cli_modules/` to avoid the naming conflict
- Decision: Use `prsm/cli_modules/` to avoid breaking the existing entry point
- Update imports in cli.py: `from prsm.cli_modules.ui import show_banner, section, success, ...`

**Verification:** `cd /Users/ryneschultz/Documents/GitHub/PRSM && python -c "from prsm.cli_modules.ui import show_banner; show_banner()"`

---

## Phase 1: Unified `prsm setup` First-Run Wizard

### Task 1.1: Create Setup Wizard Module

**Objective:** Build the main `prsm setup` command that orchestrates the entire first-run experience.

**Files:**
- Create: `prsm/cli_modules/setup_wizard.py`

**Details:**

The setup wizard has 7 steps:

```
Step 1: Welcome + System Detection
  - Show PRSM banner with prismatic gradient
  - Detect OS, CPU, RAM, GPU, disk space
  - Show system summary in a Rich Table
  - Check prerequisites (Python version, IPFS availability, network)
  
Step 2: Intent / Role Selection
  - "What would you like to do with PRSM?"
  - Options:
    a) Contribute Resources (earn FTNS by sharing compute/storage)
    b) Use the Network (consume AI services, curate data)
    c) Both (full node — contribute and consume)
  - Brief explanation panel for each option
  
Step 3: Resource Allocation
  - Only shown if contributing resources
  - Visual resource bars showing current system usage
  - Interactive sliders/prompts for:
    * CPU allocation % (with visual bar)
    * Memory allocation % (with visual bar)
    * GPU allocation % (if GPU detected)
    * Storage pledge (GB, with disk space context)
    * Max concurrent jobs
  - "Smart defaults" button: auto-configure based on system
  - Active hours/days scheduling (optional advanced)
  
Step 4: API Keys & Wallet
  - FTNS wallet setup (generate new or import existing)
  - API key configuration:
    * OpenAI key (optional, for embeddings)
    * Anthropic key (optional, for Claude)
    * HuggingFace token (optional, for model access)
  - Each key validated on entry with ✓/✗ feedback
  - Keys stored in ~/.prsm/.env (gitignored)
  
Step 5: Network Configuration
  - P2P port (default 9001, with port availability check)
  - API port (default 8000, with port availability check)
  - Bootstrap nodes (pre-populated with known nodes)
  - Advanced: custom bootstrap, relay mode, NAT traversal
  
Step 6: AI Integration
  - "Would you like to enable AI assistant integration?"
  - If yes: configure MCP server exposure on localhost
  - Port selection for MCP server (default 9100)
  - Show example of how to add PRSM to Hermes/OpenClaw config
  - Generate MCP config snippet that user can copy
  
Step 7: Review & Launch
  - Summary panel showing ALL configured settings
  - Resource allocation visual bars
  - Network config summary
  - "Everything look good?" confirmation
  - Save config to ~/.prsm/config.yaml
  - Option to start node immediately or just save config
  - Show next steps: "prsm start", "prsm config", "prsm status"
```

Each step should:
- Use `section()` header with step number: "◆ Step 1 of 7 — Welcome"
- Show a progress indicator: "Step 1/7 ═══════░░░░░░"
- Allow going back to previous step (press 'b')
- Allow skipping optional steps (press 's')
- Save progress incrementally (crash-resilient)

**Verification:** `prsm setup --dry-run` should walk through all prompts without saving

---

### Task 1.2: Wire `prsm setup` Command into CLI

**Objective:** Add the `prsm setup` command to the main CLI.

**Files:**
- Modify: `prsm/cli.py` — add `setup` command
- Create: `prsm/cli_modules/setup_wizard.py` (the actual wizard logic)

**Details:**
```python
@main.command()
@click.option("--dry-run", is_flag=True, help="Walk through setup without saving")
@click.option("--minimal", is_flag=True, help="Quick setup with smart defaults")
@click.option("--reset", is_flag=True, help="Reset all settings and re-run setup")
def setup(dry_run, minimal, reset):
    """Interactive first-run setup wizard for PRSM."""
    from prsm.cli_modules.setup_wizard import run_setup_wizard
    run_setup_wizard(dry_run=dry_run, minimal=minimal, reset=reset)
```

**Verification:** `prsm setup --help` shows the command with options

---

### Task 1.3: First-Run Auto-Detection

**Objective:** Automatically trigger setup wizard on first run if no config exists.

**Files:**
- Modify: `prsm/cli.py` — add first-run check to main group

**Details:**
- On any `prsm` command (except `setup`, `--help`, `--version`), check if `~/.prsm/config.yaml` exists
- If not, show: "Welcome to PRSM! Let's get you set up." and offer to run setup
- Don't force it — user can skip with `--no-setup` or by creating a minimal config

---

## Phase 2: `prsm config` Management Command

### Task 2.1: Create Config Management Module

**Objective:** Build a top-level `prsm config` command for viewing and editing settings.

**Files:**
- Create: `prsm/cli_modules/config_manager.py`
- Modify: `prsm/cli.py` — add `config` command group

**Details:**

Commands:
```
prsm config              → Re-enter interactive setup wizard (all steps)
prsm config show         → Display current config in styled Rich panels
prsm config set KEY VAL  → Set a specific config value
prsm config get KEY      → Get a specific config value
prsm config reset        → Reset to defaults (with confirmation)
prsm config export       → Export config as YAML to stdout
prsm config import FILE  → Import config from YAML file
prsm config wizard       → Run just the setup wizard again
```

`prsm config show` output should be a beautiful Rich display:
```
┌─── ◇ PRSM Configuration ───────────────────────┐
│                                                   │
│  ◆ Node Identity                                  │
│    Display Name: ryno-node-01                     │
│    Node ID:      prsm_abc123...                   │
│    Role:         Full Node (Compute + Storage)    │
│                                                   │
│  ◆ Resource Allocation                            │
│    CPU:     ████████████████░░░░░░░░░░░░░░ 50%   │
│    Memory:  ████████████████░░░░░░░░░░░░░░ 50%   │
│    GPU:     ████████████████████████░░░░░░ 80%   │
│    Storage: 10.0 GB pledged (245 GB available)    │
│    Jobs:    3 concurrent max                      │
│                                                   │
│  ◆ Network                                        │
│    P2P Port:  9001 ✓                              │
│    API Port:  8000 ✓                              │
│    Bootstrap: 2 nodes configured                  │
│                                                   │
│  ◆ AI Integration                                 │
│    MCP Server: enabled (localhost:9100)            │
│    Skills:     3 packages installed               │
│                                                   │
│  Config: ~/.prsm/config.yaml                      │
│  Run 'prsm config' to modify settings             │
└───────────────────────────────────────────────────┘
```

---

### Task 2.2: Unified Config Schema

**Objective:** Create a single Pydantic config model that encompasses all setup wizard settings.

**Files:**
- Create: `prsm/cli_modules/config_schema.py`

**Details:**
```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from pathlib import Path

class NodeRole(str, Enum):
    CONTRIBUTOR = "contributor"
    CONSUMER = "consumer"  
    FULL = "full"

class PRSMConfig(BaseModel):
    """Unified PRSM configuration — the single source of truth."""
    # Node identity
    display_name: str = "prsm-node"
    node_role: NodeRole = NodeRole.FULL
    
    # Resource allocation
    cpu_pct: int = Field(50, ge=10, le=90)
    memory_pct: int = Field(50, ge=10, le=90)
    gpu_pct: int = Field(80, ge=0, le=100)
    storage_gb: float = Field(10.0, ge=1.0)
    max_concurrent_jobs: int = Field(3, ge=1, le=20)
    upload_mbps_limit: float = Field(0.0, ge=0)  # 0 = unlimited
    active_hours_start: Optional[int] = None  # 0-23
    active_hours_end: Optional[int] = None
    active_days: list[str] = Field(default_factory=list)
    
    # Network
    p2p_port: int = Field(9001, ge=1024, le=65535)
    api_port: int = Field(8000, ge=1024, le=65535)
    bootstrap_nodes: list[str] = Field(default_factory=list)
    
    # API Keys (stored separately in .env but referenced here)
    has_openai_key: bool = False
    has_anthropic_key: bool = False
    has_huggingface_token: bool = False
    
    # FTNS Wallet
    wallet_address: Optional[str] = None
    
    # AI Integration
    mcp_server_enabled: bool = True
    mcp_server_port: int = Field(9100, ge=1024, le=65535)
    
    # Meta
    setup_completed: bool = False
    setup_version: str = "1.0.0"
    
    @classmethod
    def config_path(cls) -> Path:
        return Path.home() / ".prsm" / "config.yaml"
    
    def save(self):
        import yaml
        self.config_path().parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path(), "w") as f:
            yaml.safe_dump(self.model_dump(), f, default_flow_style=False)
    
    @classmethod
    def load(cls) -> "PRSMConfig":
        import yaml
        path = cls.config_path()
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        return cls()
```

This replaces/supersedes the existing `NodeConfig` dataclass for user-facing config while maintaining backward compatibility.

**Verification:** Round-trip test — save config, load it, compare values.

---

## Phase 3: Background Daemon Mode

### Task 3.1: Create Daemon Manager Module

**Objective:** Enable `prsm start --daemon` to run the PRSM node as a background service.

**Files:**
- Create: `prsm/cli_modules/daemon.py`
- Modify: `prsm/cli.py` — add/modify `start` and `stop` commands

**Details:**

Approach: Use `subprocess.Popen` with stdout/stderr redirected to log files, write PID to `~/.prsm/prsm.pid`. Cross-platform (no systemd dependency).

Commands:
```
prsm start              → Start node in foreground (existing behavior)
prsm start --daemon     → Start node as background process
prsm stop               → Stop the running daemon (send SIGTERM, wait, SIGKILL)
prsm restart            → Stop + start daemon
prsm status             → Show node status (running/stopped, uptime, resource usage)
prsm logs               → Tail the daemon log file
prsm logs --follow      → Live tail with Rich formatting
```

`prsm status` output:
```
◇ PRSM Node Status
  ─────────────────

  Status:     🟢 Running (PID 12345)
  Uptime:     2h 34m 12s
  Role:       Full Node

  Resources:
    CPU:      ████████░░░░░░░░ 32% of 50% allocated
    Memory:   ██████░░░░░░░░░░ 24% of 50% allocated  
    Storage:  ██░░░░░░░░░░░░░░ 1.2 GB of 10 GB used
    
  Network:
    Peers:    7 connected
    P2P:      :9001 ✓
    API:      :8000 ✓
    
  Economy:
    FTNS Earned:  142.5 (today: 12.3)
    Jobs Done:    47 (today: 8)
    
  Logs: ~/.prsm/logs/prsm.log
```

---

### Task 3.2: Platform Service Integration (Optional/Advanced)

**Objective:** Generate systemd/launchd service files for auto-start on boot.

**Files:**
- Create: `prsm/cli_modules/service.py`

**Details:**
```
prsm service install     → Generate and install systemd/launchd service
prsm service uninstall   → Remove service
prsm service status      → Show service status
```

- macOS: Generate launchd plist at ~/Library/LaunchAgents/com.prsm.node.plist
- Linux: Generate systemd user unit at ~/.config/systemd/user/prsm-node.service
- Both: KeepAlive/Restart=always, log rotation, environment from ~/.prsm/.env

---

## Phase 4: AI-Native Skill Packages

### Task 4.1: Define Skill Package Format

**Objective:** Create the standard format for PRSM skill packages that any AI can ingest.

**Files:**
- Create: `prsm/skills/__init__.py`
- Create: `prsm/skills/schema.py`
- Create: `prsm/skills/loader.py`
- Create: `prsm/skills/registry.py`

**Details:**

A PRSM skill package is a directory or installable package containing:
```
prsm-skill-datasets/
├── SKILL.yaml           # Manifest (name, version, description, capabilities)
├── README.md            # Human-readable documentation
├── tools.json           # MCP-compatible tool definitions
├── prompts/             # System prompts for AI agents
│   ├── curator.md       # "You are a dataset curator on the PRSM network..."
│   └── analyst.md       # "You are a data quality analyst..."
├── workflows/           # Multi-step workflow definitions
│   └── curate.yaml      # Step-by-step curation workflow
└── examples/            # Usage examples
    └── curate_code_data.md
```

`SKILL.yaml` format:
```yaml
name: prsm-datasets
version: 1.0.0
description: Dataset curation and management on the PRSM network
author: PRSM Core Team
capabilities:
  - dataset_search
  - dataset_curate
  - dataset_validate
  - dataset_publish
requires:
  - prsm-network >= 0.2.0
tools:
  - name: prsm_search_datasets
    description: Search the PRSM network for datasets matching criteria
    parameters:
      query: {type: string, description: "Search query"}
      domain: {type: string, description: "Domain filter", optional: true}
      min_quality: {type: number, description: "Minimum quality score 0-1", optional: true}
  - name: prsm_curate_dataset
    description: Curate a new dataset from network sources
    parameters:
      name: {type: string}
      sources: {type: array, items: {type: string}}
      filters: {type: object, optional: true}
  - name: prsm_publish_dataset
    description: Publish a curated dataset to the PRSM network
    parameters:
      dataset_id: {type: string}
      metadata: {type: object}
      ftns_price: {type: number, optional: true}
```

This format is designed to be:
1. Human-readable (YAML + Markdown)
2. AI-ingestible (structured tool definitions that map to MCP)
3. Hermes-compatible (similar to SKILL.md format)
4. Installable via `prsm skills install`

---

### Task 4.2: Built-in Skill Packages

**Objective:** Ship PRSM with 3 core skill packages out of the box.

**Files:**
- Create: `prsm/skills/builtins/datasets/` (dataset curation)
- Create: `prsm/skills/builtins/compute/` (compute job management)
- Create: `prsm/skills/builtins/network/` (network operations)

**Details:**

1. **prsm-datasets** — Search, curate, validate, publish datasets
2. **prsm-compute** — Submit compute jobs, check status, manage queue
3. **prsm-network** — Node management, peer discovery, network stats

Each package follows the schema from Task 4.1.

---

### Task 4.3: `prsm skills` CLI Commands

**Objective:** Build the skill management CLI.

**Files:**
- Create: `prsm/cli_modules/skills_cli.py`
- Modify: `prsm/cli.py` — add `skills` command group

**Details:**
```
prsm skills list              → List installed skill packages
prsm skills search QUERY      → Search available skills on network
prsm skills install PACKAGE   → Install a skill package
prsm skills remove PACKAGE    → Remove a skill package
prsm skills info PACKAGE      → Show detailed skill info
prsm skills export PACKAGE    → Export skill as MCP tool manifest
```

`prsm skills list` output:
```
◇ Installed PRSM Skills
  ──────────────────────

  ◆ prsm-datasets v1.0.0
    Dataset curation and management
    Tools: search, curate, validate, publish
    
  ◆ prsm-compute v1.0.0
    Compute job management
    Tools: submit, status, cancel, queue
    
  ◆ prsm-network v1.0.0
    Network operations and monitoring
    Tools: peers, stats, health, discover

  3 packages installed (12 tools available)
  Run 'prsm skills info <package>' for details
```

---

### Task 4.4: MCP Server for AI Integration

**Objective:** Expose PRSM skills as an MCP server that any AI assistant can connect to.

**Files:**
- Create: `prsm/cli_modules/mcp_server.py`
- Modify: `prsm/cli.py` — add `mcp` command

**Details:**

When `prsm start` runs with MCP enabled (default), it starts a local MCP server that:
1. Reads all installed skill packages
2. Exposes their tools as MCP-compatible endpoints
3. Listens on the configured port (default 9100)

A user adds this to their Hermes config:
```yaml
mcp_servers:
  prsm:
    transport: streamable-http
    url: http://localhost:9100/mcp
```

Then from Hermes/Claude:
```
"Search PRSM for transformer training datasets focused on code generation"
→ Hermes calls prsm_search_datasets via MCP
→ PRSM node queries the network
→ Results returned to the AI
```

Commands:
```
prsm mcp start           → Start MCP server standalone
prsm mcp status          → Check MCP server status
prsm mcp config-snippet  → Print config snippet for Hermes/OpenClaw
```

---

## Phase 5: Integration & Polish

### Task 5.1: Migrate Existing Wizard to New UI

**Objective:** Replace the existing `_run_node_wizard()` and `_run_interactive_configure()` with calls to the new setup wizard, preserving all functionality.

**Files:**
- Modify: `prsm/cli.py` — update `node start --wizard` and `node configure`

**Details:**
- `prsm node start --wizard` → redirect to `prsm setup`
- `prsm node configure` → redirect to `prsm config`
- Keep backward compatibility: old commands still work, just delegate to new ones
- Deprecation notice: "Tip: use 'prsm config' for the full configuration experience"

---

### Task 5.2: End-to-End Testing

**Objective:** Test the complete flow from install to running node.

**Files:**
- Create: `tests/cli/test_setup_wizard.py`
- Create: `tests/cli/test_config_manager.py`
- Create: `tests/cli/test_daemon.py`
- Create: `tests/cli/test_skills.py`

**Details:**
- Test each wizard step independently (mock click prompts)
- Test config round-trip (save/load/show)
- Test daemon PID management (start/stop/status)
- Test skill loading and MCP tool generation
- All tests should run without network access

---

### Task 5.3: Documentation

**Objective:** Update README and create getting-started guide.

**Files:**
- Modify: `README.md` — update quick-start section
- Create: `docs/getting-started.md`
- Create: `docs/configuration.md`
- Create: `docs/ai-integration.md`

---

## Execution Order & Dependencies

```
Phase 0 (Foundation)     → No dependencies
  Task 0.1: CLI package + theme + UI toolkit
  Task 0.2: Wire into existing cli.py

Phase 1 (Setup Wizard)  → Depends on Phase 0
  Task 1.1: Setup wizard module
  Task 1.2: Wire prsm setup command
  Task 1.3: First-run auto-detection

Phase 2 (Config)         → Depends on Phase 0
  Task 2.1: Config management CLI
  Task 2.2: Unified config schema

Phase 3 (Daemon)         → Depends on Phase 2
  Task 3.1: Daemon manager
  Task 3.2: Platform service integration

Phase 4 (Skills)         → Depends on Phase 0
  Task 4.1: Skill package format
  Task 4.2: Built-in skill packages
  Task 4.3: Skills CLI
  Task 4.4: MCP server

Phase 5 (Polish)         → Depends on all above
  Task 5.1: Migrate existing wizard
  Task 5.2: E2E testing
  Task 5.3: Documentation
```

Phases 1, 2, and 4 can run in parallel after Phase 0 completes.

---

## Delegation Strategy

| Phase | Owner | Rationale |
|-------|-------|-----------|
| Phase 0 | Hermes (subagent) | Foundation — needs careful architecture |
| Phase 1 | Hermes (subagent) | Core UX — needs art direction attention |
| Phase 2 | Isaac (OpenClaw) | Config schema + CLI — well-defined scope |
| Phase 3 | Isaac (OpenClaw) | Daemon management — systems work |
| Phase 4 | Hermes (subagent) | AI-native skills — needs MCP expertise |
| Phase 5 | Both | Integration testing + docs |

---

## Risks & Mitigations

1. **cli.py is 6400 lines** — Mitigated by putting all new code in `prsm/cli_modules/` and only adding thin wrappers in cli.py
2. **Naming conflict: prsm/cli.py vs prsm/cli/` package** — Mitigated by using `prsm/cli_modules/` directory name
3. **Config migration** — Old `~/.prsm/node_config.json` users need migration path. New unified config in `~/.prsm/config.yaml` with migration function.
4. **MCP server stability** — Start simple (FastMCP), iterate. Don't over-engineer the protocol layer.
5. **Skill package discovery** — V1 is local-only (built-in + installed). Network discovery is Phase 2 of a future plan.
