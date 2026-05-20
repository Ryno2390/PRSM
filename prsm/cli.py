"""
PRSM Command Line Interface
Main entry point for PRSM CLI commands
"""

import asyncio
import socket
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # Forward references for type annotations — the actual NodeConfig class
    # is imported inside functions that need it at runtime (setup wizard /
    # configure flow) to avoid loading the full node stack at module import.
    from prsm.node.config import NodeConfig  # noqa: F401
from urllib.parse import urlparse

import os

import click
import uvicorn
from rich.console import Console
from rich.table import Table


def detect_available_backends() -> dict:
    """
    Check which LLM backends have valid API keys configured.

    Inlined in v1.6.0 from the removed ``prsm.compute.nwtn.backends.config``
    module — the legacy backends subsystem was deleted as part of the scope
    alignment refactor (PR 2). PRSM users now rely on third-party LLMs
    (local or via OAuth/API); this helper only needs to inspect environment
    variables to decide whether a real backend is available.
    """
    anthropic_available = bool(os.environ.get("ANTHROPIC_API_KEY"))
    openai_available = bool(os.environ.get("OPENAI_API_KEY"))
    local_available = bool(os.environ.get("PRSM_LOCAL_MODEL_URL"))
    return {
        "anthropic": anthropic_available,
        "openai": openai_available,
        "local": local_available,
        "any_real_backend": anthropic_available or openai_available or local_available,
    }


console = Console()

# ---------------------------------------------------------------------------
# AI-agent-centric output helpers (Sub-phase 10.6)
# ---------------------------------------------------------------------------

def _agent_output(data: dict, format: str = "json") -> None:
    """
    Write structured output to stdout.

    In ``json`` mode (the default for agent callers) the output is a single
    valid JSON document — no colour, no Rich formatting, no spinners.
    In ``text`` mode the caller should render with Rich as normal.

    All agent-callable commands should call this helper so the same command
    works for both human inspection and programmatic consumption.

    Conventions:
    - Success:  exit 0,  ``{"ok": true, ...}``
    - Failure:  exit 1,  ``{"ok": false, "error": "..."}``
    """
    import json
    sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")
    sys.stdout.flush()


def _agent_error(message: str, code: int = 1) -> None:
    """Write a JSON error to stdout and exit with *code*."""
    import json
    sys.stdout.write(json.dumps({"ok": False, "error": message}) + "\n")
    sys.stdout.flush()
    raise SystemExit(code)


def _run_async(coro):
    """
    Run *coro* and return its result regardless of whether an event loop is
    already running in the current thread (e.g. inside pytest-asyncio).

    Strategy:
    - Try ``asyncio.run()`` in the current thread first (fast path, works in
      production where no loop is running).
    - If a loop is already running, spawn a daemon thread with a fresh loop
      and block until completion (safe, no nesting required).
    """
    import threading

    try:
        asyncio.get_running_loop()
        # A loop IS running — use a separate thread
        result_box: list = [None]
        exc_box:    list = [None]

        def _worker():
            try:
                result_box[0] = asyncio.run(coro)
            except Exception as e:
                exc_box[0] = e

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join()
        if exc_box[0]:
            raise exc_box[0]
        return result_box[0]

    except RuntimeError:
        # No loop running — safe to call asyncio.run() directly
        return asyncio.run(coro)


PREFLIGHT_PASS = "PASS"
PREFLIGHT_WARN = "WARN"
PREFLIGHT_FAIL = "FAIL"
PREFLIGHT_BOOTSTRAP_TIMEOUT_SECONDS = 1.5


# ── Credential storage ──────────────────────────────────────────────────────
# Credentials are stored at ~/.prsm/credentials.json (chmod 600).
# They contain the JWT access token obtained by `prsm login`.

_CREDENTIALS_FILE = Path.home() / ".prsm" / "credentials.json"


def _load_credentials() -> Optional[dict]:
    """Return stored credentials dict, or None if not logged in."""
    if _CREDENTIALS_FILE.exists():
        try:
            import json
            return json.loads(_CREDENTIALS_FILE.read_text())
        except Exception:
            return None
    return None


def _save_credentials(data: dict) -> None:
    """Write credentials to disk with restricted permissions (owner-only read)."""
    import json
    _CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CREDENTIALS_FILE.write_text(json.dumps(data, indent=2))
    _CREDENTIALS_FILE.chmod(0o600)


def _clear_credentials() -> None:
    """Remove stored credentials file."""
    if _CREDENTIALS_FILE.exists():
        _CREDENTIALS_FILE.unlink()


def _auth_headers() -> dict:
    """
    Return a dict with Authorization header, or empty dict if not logged in.

    Used by all commands that call authenticated API endpoints. Empty dict
    causes the request to proceed without auth (resulting in a 401 which
    the command then surfaces as "Session expired. Run: prsm login").
    """
    creds = _load_credentials()
    if creds and creds.get("access_token"):
        return {"Authorization": f"Bearer {creds['access_token']}"}
    return {}


def _api_url_from_creds(override: Optional[str]) -> str:
    """
    Return the API URL to use: explicit override → stored credential → default.
    """
    if override:
        return override.rstrip("/")
    creds = _load_credentials()
    if creds and creds.get("api_url"):
        return creds["api_url"].rstrip("/")
    return "http://127.0.0.1:8000"


@dataclass
class PreflightCheckResult:
    """Node startup preflight check result."""

    name: str
    status: str
    required: bool
    details: str
    remediation: str


def _probe_tcp_endpoint(host: str, port: int, timeout_seconds: float) -> tuple[bool, str]:
    """Attempt a bounded TCP connection probe to a host:port endpoint."""
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True, f"reachable at {host}:{port}"
    except Exception as exc:
        return False, f"unreachable at {host}:{port} ({exc})"


def _parse_endpoint(address: str, default_port: int) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """Parse host/port endpoint from either URL or host:port style strings."""
    if not address or not address.strip():
        return None, None, "empty address"

    address = address.strip()

    try:
        if "://" in address:
            parsed = urlparse(address)
            if not parsed.hostname:
                return None, None, f"invalid URL: {address}"
            return parsed.hostname, int(parsed.port or default_port), None

        host, sep, port_text = address.rpartition(":")
        if sep and host:
            return host, int(port_text), None

        return address, default_port, None
    except Exception as exc:
        return None, None, f"unable to parse '{address}' ({exc})"


def parse_active_hours(value: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """Parse active hours string like '22-8' or 'off'.
    
    Returns:
        Tuple of (start_hour, end_hour) where None values mean "always on".
    
    Raises:
        click.BadParameter: If the format is invalid.
    """
    if value is None:
        return None, None
    if value.lower() == "off":
        return None, None  # Always on
    try:
        parts = value.split("-")
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            if 0 <= start <= 23 and 0 <= end <= 23:
                return start, end
    except (ValueError, AttributeError):
        pass
    raise click.BadParameter(f"Invalid active hours format: {value}. Use '22-8' or 'off'")


def parse_active_days(value: Optional[str]) -> List[int]:
    """Parse active days string like 'mon,tue,wed' or 'weekdays'.
    
    Returns:
        List of day numbers (0=Mon, 6=Sun). Empty list means every day.
    """
    if value is None:
        return []
    value = value.lower()
    if value == "weekdays":
        return [0, 1, 2, 3, 4]  # Mon-Fri
    if value == "weekends":
        return [5, 6]  # Sat-Sun
    day_map = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    days = []
    for part in value.split(","):
        part = part.strip().lower()[:3]  # Take first 3 chars
        if part in day_map:
            days.append(day_map[part])
    return sorted(days)


def _node_preflight_diagnostics(config: "NodeConfig") -> List[PreflightCheckResult]:
    """Run non-breaking node startup diagnostics with required/optional classification."""

    checks: List[PreflightCheckResult] = []

    # Required: Python runtime basics.
    py_ok = sys.version_info >= (3, 10)
    checks.append(
        PreflightCheckResult(
            name="Python runtime",
            status=PREFLIGHT_PASS if py_ok else PREFLIGHT_FAIL,
            required=True,
            details=f"detected {sys.version.split()[0]}",
            remediation="Install Python 3.10+ and re-run 'prsm node start'.",
        )
    )

    # Required but non-blocking by itself: config presence (defaults are supported).
    config_exists = config.config_path.exists()
    checks.append(
        PreflightCheckResult(
            name="Node config file",
            status=PREFLIGHT_PASS if config_exists else PREFLIGHT_WARN,
            required=False,
            details=(
                f"found at {config.config_path}"
                if config_exists
                else f"missing at {config.config_path}; defaults will be used"
            ),
            remediation=(
                "Run 'prsm node start --wizard' to generate explicit node settings."
                if not config_exists
                else "None"
            ),
        )
    )

    # Required hard precondition: local API bind availability.
    api_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    api_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        api_sock.bind(("127.0.0.1", config.api_port))
        api_status = PREFLIGHT_PASS
        api_details = f"127.0.0.1:{config.api_port} is available"
    except Exception as exc:
        api_status = PREFLIGHT_FAIL
        api_details = f"127.0.0.1:{config.api_port} unavailable ({exc})"
    finally:
        api_sock.close()

    checks.append(
        PreflightCheckResult(
            name="Local API bind",
            status=api_status,
            required=True,
            details=api_details,
            remediation="Choose a free --api-port or stop the process using this port.",
        )
    )

    # Optional: quick bootstrap reachability probe (bounded timeout).
    bootstrap_nodes = list(config.bootstrap_nodes or [])
    if not bootstrap_nodes:
        checks.append(
            PreflightCheckResult(
                name="Bootstrap target reachability",
                status=PREFLIGHT_WARN,
                required=False,
                details="no bootstrap targets configured",
                remediation="Configure --bootstrap or node_config bootstrap_nodes for peer discovery.",
            )
        )
    else:
        any_reachable = False
        probe_notes: List[str] = []
        for address in bootstrap_nodes:
            host, port, parse_error = _parse_endpoint(address, default_port=config.p2p_port)
            if parse_error:
                probe_notes.append(f"{address}: {parse_error}")
                continue
            ok, detail = _probe_tcp_endpoint(host, port, PREFLIGHT_BOOTSTRAP_TIMEOUT_SECONDS)
            probe_notes.append(f"{address}: {detail}")
            if ok:
                any_reachable = True
                break

        checks.append(
            PreflightCheckResult(
                name="Bootstrap target reachability",
                status=PREFLIGHT_PASS if any_reachable else PREFLIGHT_WARN,
                required=False,
                details=(
                    "at least one target reachable"
                    if any_reachable
                    else "; ".join(probe_notes[:2]) or "no reachable targets"
                ),
                remediation=(
                    "None"
                    if any_reachable
                    else "Startup will continue in degraded mode; verify DNS/network or update bootstrap targets."
                ),
            )
        )

    # Optional dependency: ContentStore availability.
    try:
        from prsm.storage import get_content_store
        store_available = get_content_store() is not None
    except Exception:
        store_available = False
    checks.append(
        PreflightCheckResult(
            name="ContentStore (optional)",
            status=PREFLIGHT_PASS if store_available else PREFLIGHT_WARN,
            required=False,
            details="Native ContentStore initialized" if store_available else "ContentStore not initialized",
            remediation=(
                "None"
                if store_available
                else "ContentStore will be auto-initialized on node start."
            ),
        )
    )

    # Wallet config diagnostic. Operators see derived on-chain
    # address pre-startup so they can confirm they're running with
    # the wallet they expect. Catches stale env / wrong-key
    # misconfigurations before any on-chain action.
    pk = os.environ.get("FTNS_WALLET_PRIVATE_KEY", "").strip()
    if not pk:
        checks.append(
            PreflightCheckResult(
                name="Wallet config (optional)",
                status=PREFLIGHT_WARN,
                required=False,
                details="FTNS_WALLET_PRIVATE_KEY not set",
                remediation=(
                    "Set FTNS_WALLET_PRIVATE_KEY for on-chain "
                    "operations (royalty claim, heartbeat trigger, "
                    "distribution trigger). Read-only paths work "
                    "without it."
                ),
            )
        )
    else:
        try:
            from prsm.node.operator_address import (
                resolve_operator_address,
            )
            addr = resolve_operator_address()
            if addr:
                checks.append(
                    PreflightCheckResult(
                        name="Wallet config (optional)",
                        status=PREFLIGHT_PASS,
                        required=False,
                        details=f"Operator address: {addr[:8]}...{addr[-6:]}",
                        remediation="None",
                    )
                )
            else:
                checks.append(
                    PreflightCheckResult(
                        name="Wallet config (optional)",
                        status=PREFLIGHT_WARN,
                        required=False,
                        details="FTNS_WALLET_PRIVATE_KEY set but malformed",
                        remediation=(
                            "Verify private key is 0x-prefixed 32 bytes "
                            "of hex. Bad key won't crash startup but "
                            "all on-chain operations will fail."
                        ),
                    )
                )
        except Exception as exc:
            checks.append(
                PreflightCheckResult(
                    name="Wallet config (optional)",
                    status=PREFLIGHT_WARN,
                    required=False,
                    details=f"Address derivation failed: {exc}",
                    remediation="Check FTNS_WALLET_PRIVATE_KEY format.",
                )
            )

    return checks


def _render_preflight_summary(results: List[PreflightCheckResult]) -> None:
    """Render startup diagnostics summary with remediation hints."""
    table = Table(title="Node Startup Preflight Diagnostics")
    table.add_column("Check", style="cyan")
    table.add_column("Req", style="magenta")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="green")
    table.add_column("Remediation", style="yellow")

    for result in results:
        status_style = {
            PREFLIGHT_PASS: "green",
            PREFLIGHT_WARN: "yellow",
            PREFLIGHT_FAIL: "red",
        }.get(result.status, "white")
        table.add_row(
            result.name,
            "required" if result.required else "optional",
            f"[{status_style}]{result.status}[/{status_style}]",
            result.details,
            result.remediation,
        )

    console.print()
    console.print(table)
    console.print()


def _has_hard_preflight_failures(results: List[PreflightCheckResult]) -> bool:
    """True when a required preflight check has failed."""
    return any(r.required and r.status == PREFLIGHT_FAIL for r in results)


def _should_announce_degraded_mode(results: List[PreflightCheckResult]) -> bool:
    """Determine whether startup should explicitly mention degraded mode continuation."""
    for result in results:
        if result.name == "Bootstrap target reachability" and result.status in (PREFLIGHT_WARN, PREFLIGHT_FAIL):
            return True
    return False


def _init_config():
    """Initialize config manager with defaults so get_settings() works.

    This must be called before starting uvicorn so that when the app
    module is imported, get_settings()/get_config() return valid objects.
    """
    from prsm.core.config import ConfigManager
    manager = ConfigManager()
    if manager.get_config() is None:
        manager.load_config()


def _get_debug() -> bool:
    """Get debug setting, defaulting to False if config unavailable."""
    try:
        _init_config()
        from prsm.core.config import get_settings
        s = get_settings()
        return getattr(s, 'debug', False) if s else False
    except Exception:
        return False


def _get_version():
    """Read version from package metadata (single source of truth: pyproject.toml).

    Sprint 150 — fallback now reads `prsm.__version__` instead of a
    stale literal. The hardcoded "0.24.0" leaked into bootstrap
    server registrations + assorted UA strings on every importlib
    metadata-miss path.
    """
    try:
        from importlib.metadata import version as pkg_version
        return pkg_version("prsm-network")
    except Exception:
        try:
            import prsm as _prsm_pkg
            return _prsm_pkg.__version__
        except Exception:
            return "unknown"


@click.group()
@click.version_option(version=_get_version(), prog_name="PRSM")
@click.pass_context
def main(ctx):
    """
    PRSM: Protocol for Research, Storage, and Modeling

    A peer-to-peer protocol unifying data, compute, and economic layers.
    Frontier LLMs (Claude, GPT, Gemini, or local) call PRSM via MCP as
    a retrieval + heavy-compute substrate. Contributors earn FTNS for
    sharing storage, compute, and data.
    """
    # Auto-migrate old node_config.json -> config.yaml for existing users
    try:
        from prsm.cli_modules.migration import migrate_if_needed
        migrate_if_needed()
    except Exception:
        pass  # never block CLI startup on migration issues

    # First-run auto-detection: nudge unconfigured users toward setup
    config_path = Path.home() / ".prsm" / "config.yaml"
    if not config_path.exists():
        # Don't nag on setup, help, or version invocations
        invoked = ctx.invoked_subcommand or ""
        safe_commands = {"setup", None}  # None = no subcommand (shows help)
        if invoked not in safe_commands:
            click.echo("  ◇ PRSM is not configured yet. Run: prsm setup")
            click.echo()


# ── Theme + icons (used by config/mcp subcommands below) ─────────────
from prsm.cli_modules.theme import THEME, ICONS  # noqa: E402, F401

# ── Skills CLI (Phase 4.3) ───────────────────────────────────────────
from prsm.cli_modules.skills_cli import skills as skills_group  # noqa: E402
main.add_command(skills_group, "skills")


# ── On-chain Provenance CLI (Phase 1) ────────────────────────────────
from prsm.cli_modules.provenance import provenance as provenance_group  # noqa: E402
main.add_command(provenance_group, "provenance")


@main.command()
def version():
    """Print the installed PRSM package version.

    Reads from importlib.metadata.version("prsm-network") so
    it stays in sync with pyproject.toml across releases.
    """
    try:
        from importlib.metadata import version as _pkg_version
        v = _pkg_version("prsm-network")
        console.print(f"PRSM {v}")
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[yellow]Could not resolve PRSM version: {exc}[/yellow]\n"
            f"[dim]Install via `pip install -e .` or check "
            f"`pip show prsm-network`[/dim]"
        )
        sys.exit(1)


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to (default: localhost for security)")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@click.option("--workers", default=1, help="Number of worker processes")
def serve(host: str, port: int, reload: bool, workers: int):
    """Start the PRSM API server.

    DEPRECATED: Use `prsm node start` for full P2P connectivity,
    or `prsm daemon start` for background operation.
    """
    console.print()
    console.print("  prsm serve is deprecated.", style="yellow")
    console.print("  Use one of these instead:", style="yellow")
    console.print("    prsm node start         -- Full P2P node with dashboard", style="dim")
    console.print("    prsm node start --no-dashboard  -- Full P2P, static output", style="dim")
    console.print("    prsm daemon start       -- Background daemon mode", style="dim")
    console.print()

    console.print(f"🚀 Starting PRSM server on {host}:{port}", style="bold green")
    _init_config()

    if reload and workers > 1:
        console.print("⚠️  Cannot use --reload with multiple workers", style="yellow")
        workers = 1

    uvicorn.run(
        "prsm.interface.api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="info" if not _get_debug() else "debug"
    )


@main.command()
@click.option("--format", "output_format",
              type=click.Choice(["text", "json"]), default="text",
              help="Output format: 'text' (human) or 'json' (agent-parseable)")
def status(output_format: str):
    """Show PRSM system status.

    Use --format json for machine-readable output (AI agent consumption).
    """
    _init_config()
    from prsm.core.config import get_settings
    settings = get_settings()

    # Sprint 533 F53 fix: when ~/.prsm/config.yaml doesn't exist,
    # the global "PRSM is not configured" nudge fires above, but
    # then the table below showed "Configuration: ✅ Loaded" using
    # defaults. Contradictory signals confuse new users. Detect
    # the unconfigured state + reflect it in the table.
    config_path = Path.home() / ".prsm" / "config.yaml"
    config_status = "loaded" if config_path.exists() else "not_configured"

    if settings:
        data = {
            "ok": True,
            "components": {
                "configuration": {
                    "status": config_status,
                    "environment": getattr(settings, "environment", "unknown"),
                },
                "database": {
                    "status": "configured",
                    "url": str(getattr(settings, "database_url", "sqlite (default)")),
                },
                "storage": {
                    "status": "configured",
                    "data_dir": getattr(settings, "storage_data_dir", "~/.prsm/storage"),
                },
                "nwtn": {
                    "enabled": getattr(settings, "nwtn_enabled", True),
                    "model": getattr(settings, "nwtn_default_model", "default"),
                },
                "ftns": {
                    "enabled": getattr(settings, "ftns_enabled", True),
                    "initial_grant": getattr(settings, "ftns_initial_grant", 100),
                },
            },
        }
    else:
        data = {
            "ok": True,
            "components": {
                "configuration": {"status": "defaults", "note": "No .env file found"},
                "nwtn": {"enabled": True, "model": "default"},
                "ftns": {"enabled": True, "initial_grant": 100},
            },
        }

    if output_format == "json":
        _agent_output(data)
        return

    table = Table(title="PRSM System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Details", style="green")
    for name, info in data["components"].items():
        enabled = info.get("enabled", True)
        status_text = "✅ Loaded" if info.get("status") == "loaded" else \
                      "⚠️ Defaults" if info.get("status") == "defaults" else \
                      "⚠️ Not configured" if info.get("status") == "not_configured" else \
                      "✅ Enabled" if enabled else "❌ Disabled"
        detail = info.get("url", info.get("model", info.get("note", "")))
        table.add_row(name.capitalize(), status_text, str(detail))
    console.print(table)


@main.command()
@click.option("--port", default=8501, help="Port for the Streamlit dashboard")
@click.option("--api-port", default=8000, help="Port for the PRSM API")
def dashboard(port: int, api_port: int):
    """Launch the high-fidelity PRSM Dashboard and API"""
    import subprocess
    import time
    import socket

    _init_config()

    console.print("🚀 Starting PRSM Command Center...", style="bold green")

    # Sprint 537 F67 fix: detect if a daemon is already listening on
    # api_port. If yes, reuse it (the dashboard talks to /rings/status
    # etc. which the daemon's prsm.node.api serves). If no, spawn the
    # interface API as a fallback. Pre-fix unconditionally spawned a
    # second uvicorn that conflicted with the daemon on :8000 and died
    # silently, leaving the dashboard talking to the wrong API surface.
    api_process = None
    api_already_running = False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            api_already_running = (
                s.connect_ex(("127.0.0.1", api_port)) == 0
            )
    except Exception:
        api_already_running = False

    if api_already_running:
        console.print(
            f"  ✓ API already running on port {api_port} — reusing existing daemon",
            style="dim",
        )
    else:
        console.print(
            f"📡 No daemon detected — launching fallback API on port {api_port}...",
            style="dim",
        )
        api_process = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "prsm.interface.api.main:app",
                "--port", str(api_port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Give the API a moment to spin up
        time.sleep(2)
    
    # 2. Launch Streamlit
    console.print(f"🎨 Launching Dashboard UI on port {port}...", style="dim")
    
    # Find the correct Python executable (use venv if available)
    # Try python3.14 first, then python, then sys.executable
    venv_base = Path(__file__).parent.parent / ".venv" / "bin"
    for py_name in ["python3.14", "python"]:
        venv_python = venv_base / py_name
        if venv_python.exists():
            python_exe = str(venv_python)
            break
    else:
        python_exe = sys.executable
    
    dashboard_path = Path(__file__).parent / "interface" / "dashboard" / "streamlit_app.py"
    streamlit_cmd = [
        python_exe, "-m", "streamlit", "run", 
        str(dashboard_path),
        "--server.port", str(port),
        "--server.headless", "false"
    ]
    
    try:
        subprocess.run(streamlit_cmd)
    except KeyboardInterrupt:
        console.print("\n👋 Closing Command Center...", style="bold yellow")
    finally:
        # Sprint 537 F67 fix: only kill the API process if WE spawned
        # it. Pre-fix: api_process.terminate() always ran — when a
        # daemon was already running and we reused it, this line
        # would AttributeError on None (post-fix) or wrongly kill the
        # operator's daemon (pre-fix never had this branch).
        if api_process is not None:
            api_process.terminate()


@main.command()
@click.argument("query", required=True)
@click.option("--context", "-c", default=100, help="FTNS context allocation")
@click.option("--user-id", default="cli-user", help="User ID for the query")
@click.option("--api-url", default="http://127.0.0.1:8000", help="PRSM API URL")
def query(query: str, context: int, user_id: str, api_url: str):
    """Submit a query to a running PRSM node (dev / testing convenience).

    The recommended end-user flow is: run ``prsm mcp-server`` and configure
    your LLM client (Claude Desktop / Gemini CLI / local MCP-compatible
    tool) to point at it. Your LLM will then invoke PRSM tools directly
    whenever a query needs retrieval, heavy compute, or provenance.

    This ``query`` command hits the legacy ``/query`` REST endpoint
    directly without going through an LLM. Useful for development and
    integration testing; not the primary user flow.
    """
    import httpx
    from rich.panel import Panel

    console.print(f"Submitting query to PRSM node at {api_url}...", style="bold blue")
    console.print(f"Query: {query}")
    console.print(f"Context allocation: {context} FTNS")

    payload = {
        "prompt": query,
        "context_allocation": str(context),
        "user_id": user_id
    }

    try:
        with console.status("[bold green]PRSM node processing query..."):
            with httpx.Client(timeout=120.0) as client:
                response = client.post(f"{api_url}/query", json=payload)

        if response.status_code == 200:
            data = response.json()
            answer = data.get("final_answer", "No answer provided.")

            console.print("\n")
            console.print(Panel(answer, title="[bold green]PRSM Response[/bold green]", border_style="green"))

            trace = data.get("reasoning_trace", [])
            if trace:
                console.print("\n[bold cyan]Reasoning Trace:[/bold cyan]")
                for i, step in enumerate(trace, 1):
                    action = step.get("input_data", {}).get("action", str(step.get("agent_type", "Process")))
                    console.print(f"  [dim]{i}.[/dim] {action}")

            conf = data.get('confidence_score', 0)
            ctx = data.get('context_used', 0)
            console.print(f"\n[dim]Confidence Score: {conf:.2f} | Context Used: {ctx} tokens[/dim]")
        elif response.status_code == 404:
            # Sprint 533 F47 fix: /query was retired in favor of
            # /compute/forge. Surface actionable hint instead of
            # opaque "Not Found".
            console.print(
                "\n[bold yellow]⚠️  Legacy `/query` endpoint not found.[/bold yellow]"
            )
            console.print(
                "The `/query` route was retired. Use one of:\n"
                "  • [cyan]prsm compute run --query \"...\"[/cyan]  — full forge pipeline (Rings 1-10)\n"
                "  • [cyan]prsm compute submit --prompt \"...\"[/cyan]  — single-shot inference\n"
                "  • [cyan]prsm mcp-server[/cyan] + configure your LLM (Claude/Gemini)\n"
                "  • Direct: [cyan]curl -X POST {api}/compute/forge[/cyan]"
                .format(api=api_url)
            )
            raise SystemExit(1)
        else:
            console.print(f"\n[bold red]Error ({response.status_code}):[/bold red] {response.text}")
            raise SystemExit(1)

    except httpx.RequestError as e:
        console.print(f"\n[bold red]Connection Error:[/bold red] Could not connect to {api_url}.")
        console.print("Make sure the PRSM API server is running (`prsm serve`).")
        console.print(f"[dim]Details: {e}[/dim]")
        raise SystemExit(1)


@main.command()
def init():
    """Initialize PRSM configuration and database.

    DEPRECATED: Use `prsm setup` for the full interactive setup experience.
    """
    console.print()
    console.print("  prsm init is deprecated.", style="yellow")
    console.print("  Use one of these instead:", style="yellow")
    console.print("    prsm setup              -- Interactive setup wizard", style="dim")
    console.print("    prsm setup --minimal    -- Quick setup with defaults", style="dim")
    console.print("    prsm daemon start       -- Start the daemon directly", style="dim")
    console.print()

    console.print("🔧 Initializing PRSM...", style="bold blue")

    # Check if .env exists
    env_file = Path(".env")
    if not env_file.exists():
        env_example = Path(".env.example")
        if env_example.exists():
            console.print("📄 Copying .env.example to .env")
            env_file.write_text(env_example.read_text())
        else:
            console.print("❌ No .env.example found", style="red")
            return

    console.print("✅ Configuration ready")
    console.print("Next steps:")
    console.print("1. Edit .env with your configuration")
    console.print("2. Run: prsm db-upgrade")
    console.print("3. Run: prsm daemon start")


@main.command()
@click.option("--username", "-u", prompt="Username", help="PRSM account username")
@click.option(
    "--password", "-p",
    prompt="Password", hide_input=True,
    help="PRSM account password"
)
@click.option(
    "--api-url",
    default="http://127.0.0.1:8000",
    show_default=True,
    help="PRSM API base URL"
)
def login(username: str, password: str, api_url: str) -> None:
    """
    Log in to PRSM and save credentials for subsequent commands.

    Credentials (JWT access token) are stored at ~/.prsm/credentials.json
    with owner-only read permissions (chmod 600). Run `prsm logout` to
    remove them.
    """
    import httpx

    api_url = api_url.rstrip("/")
    console.print(f"🔑 Logging in to {api_url} as '{username}'...", style="bold blue")

    try:
        response = httpx.post(
            f"{api_url}/api/v1/auth/login",
            json={"username": username, "password": password},
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {api_url}", style="red")
        console.print("💡 Make sure the PRSM server is running: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        _save_credentials({
            "access_token":  data["access_token"],
            "refresh_token": data["refresh_token"],
            "api_url":       api_url,
            "username":      username,
        })
        expires_min = data.get("expires_in", 1800) // 60
        console.print(f"✅ Logged in as '{username}'", style="bold green")
        console.print(
            f"   Credentials saved to {_CREDENTIALS_FILE} "
            f"(token expires in {expires_min} min)",
            style="dim"
        )
    elif response.status_code == 401:
        console.print("❌ Invalid username or password", style="red")
        raise SystemExit(1)
    else:
        console.print(
            f"❌ Login failed: HTTP {response.status_code}",
            style="red"
        )
        console.print(f"   {response.text[:200]}")
        raise SystemExit(1)


@main.command()
def logout() -> None:
    """Remove stored PRSM credentials."""
    creds = _load_credentials()
    if creds:
        username = creds.get("username", "unknown")
        _clear_credentials()
        console.print(f"✅ Logged out (was: '{username}')", style="green")
        console.print(f"   Removed {_CREDENTIALS_FILE}", style="dim")
    else:
        console.print("Not currently logged in.", style="dim")


def _find_alembic_ini() -> Optional[Path]:
    """
    Locate alembic.ini, checking two locations in priority order:
    1. Parent of the directory containing cli.py (works for editable installs
       and direct invocation from the source tree)
    2. Current working directory (works when running `prsm` from the project root
       after a standard `pip install`)
    Returns the Path if found, None otherwise.
    """
    candidates = [
        Path(__file__).parent.parent / "alembic.ini",  # source tree
        Path.cwd() / "alembic.ini",                    # cwd fallback
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


@main.command()
@click.option(
    "--revision",
    default="head",
    show_default=True,
    help="Target revision label or ID. Use 'head' for latest, '+1'/'-1' for relative steps.",
)
def db_upgrade(revision: str) -> None:
    """Upgrade database schema to the target revision (default: head)."""
    console.print(f"🗄️   Upgrading database schema → '{revision}'...", style="bold blue")

    alembic_ini = _find_alembic_ini()
    if alembic_ini is None:
        console.print(
            "❌ alembic.ini not found. Run this command from the PRSM project root.",
            style="red",
        )
        raise SystemExit(1)

    try:
        from alembic.config import Config
        from alembic import command as alembic_command

        cfg = Config(str(alembic_ini))
        alembic_command.upgrade(cfg, revision)
        console.print(f"✅ Schema upgraded to '{revision}' successfully.", style="green")
    except Exception as exc:
        console.print(f"❌ Migration failed: {exc}", style="red")
        console.print(
            "💡 Verify PRSM_DATABASE_URL is set and the database is reachable.",
            style="yellow",
        )
        raise SystemExit(1)


@main.command()
@click.option(
    "--revision",
    default="-1",
    show_default=True,
    help="Target revision. Use '-1' for one step back, 'base' to revert everything.",
)
def db_downgrade(revision: str) -> None:
    """Downgrade database schema (default: one step back).

    \b
    WARNING: downgrading drops columns and tables. Back up your data first.
    Common values:
      -1              one revision back (default)
      -2              two revisions back
      base            revert all migrations
      <revision-id>   specific revision ID from `prsm db-status`
    """
    console.print(f"🗄️   Downgrading database schema → '{revision}'...", style="bold blue")
    console.print("⚠️   This will drop columns/tables. Ensure you have a backup.", style="yellow")

    alembic_ini = _find_alembic_ini()
    if alembic_ini is None:
        console.print(
            "❌ alembic.ini not found. Run this command from the PRSM project root.",
            style="red",
        )
        raise SystemExit(1)

    try:
        from alembic.config import Config
        from alembic import command as alembic_command

        cfg = Config(str(alembic_ini))
        alembic_command.downgrade(cfg, revision)
        console.print(f"✅ Schema downgraded to '{revision}' successfully.", style="green")
    except Exception as exc:
        console.print(f"❌ Migration failed: {exc}", style="red")
        console.print(
            "💡 Verify PRSM_DATABASE_URL is set and the database is reachable.",
            style="yellow",
        )
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Version management — update command
# ---------------------------------------------------------------------------


def _is_installed_via_pipx() -> bool:
    """Check if prsm-network was installed via pipx."""
    import subprocess as _sub
    result = _sub.run(
        ["pipx", "list", "--json"], capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        return False
    import json
    try:
        data = json.loads(result.stdout)
        return "prsm-network" in data.get("venvs", {})
    except (json.JSONDecodeError, KeyError):
        return False


def _run_pipx_upgrade() -> None:
    """Upgrade prsm-network via pipx."""
    import subprocess as _sub
    console.print("[dim]Detected pipx installation. Upgrading via pipx...[/dim]")
    result = _sub.run(
        ["pipx", "upgrade", "prsm-network"],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode == 0:
        console.print("[green]✓ PRSM upgraded successfully.[/green]")
        if result.stdout.strip():
            console.print(f"[dim]{result.stdout.strip()}[/dim]")
    else:
        console.print(f"[red]Upgrade failed:[/red]")
        console.print(f"[dim]{result.stderr or result.stdout}[/dim]")
        raise SystemExit(1)


def _run_pip_upgrade() -> None:
    """Upgrade prsm-network via pip."""
    import subprocess as _sub
    console.print("[dim]Detected pip installation. Upgrading via pip...[/dim]")
    result = _sub.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "prsm-network"],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode == 0:
        console.print("[green]✓ PRSM upgraded successfully.[/green]")
    else:
        console.print(f"[red]Upgrade failed:[/red]")
        console.print(f"[dim]{result.stderr or result.stdout}[/dim]")
        raise SystemExit(1)


@main.command()
@click.option("--dry-run", is_flag=True, help="Show what would be updated without installing")
def update(dry_run: bool):
    """Check for and install PRSM updates.

    Automatically detects how PRSM was installed (pipx vs pip) and
    upgrades accordingly.
    """
    import importlib.metadata
    from packaging.version import Version as _V
    from packaging.version import InvalidVersion

    # Get installed version
    try:
        current_ver = importlib.metadata.version("prsm-network")
    except importlib.metadata.PackageNotFoundError:
        console.print("[red]PRSM is not installed as a package.[/red]")
        console.print("Install with:  pip install prsm-network")
        raise SystemExit(1)

    # Fetch latest from PyPI
    try:
        import httpx
        resp = httpx.get("https://pypi.org/pypi/prsm-network/json", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        latest_ver = data["info"]["version"]
    except Exception as e:
        console.print(f"[red]Could not check for updates: {e}[/red]")
        raise SystemExit(1)

    try:
        current = _V(current_ver)
        latest = _V(latest_ver)
    except InvalidVersion:
        console.print(f"[yellow]Cannot compare versions: installed={current_ver}, latest={latest_ver}[/yellow]")
        raise SystemExit(1)

    if current >= latest:
        console.print(f"[green]✓ PRSM is up to date (v{current_ver})[/green]")
        return

    console.print(f"Update available: v{current_ver} → v{latest_ver}")

    if dry_run:
        console.print("[dim]Dry run — no changes will be made.[/dim]")
        return

    # Detect install method and upgrade
    is_pipx = _is_installed_via_pipx()
    if is_pipx:
        _run_pipx_upgrade()
    else:
        _run_pip_upgrade()


# ---------------------------------------------------------------------------
# Database management
# ---------------------------------------------------------------------------


@main.command()
def db_status() -> None:
    """Show current database migration revision and pending migrations."""
    alembic_ini = _find_alembic_ini()
    if alembic_ini is None:
        console.print(
            "❌ alembic.ini not found. Run this command from the PRSM project root.",
            style="red",
        )
        raise SystemExit(1)

    try:
        from alembic.config import Config
        from alembic import command as alembic_command

        cfg = Config(str(alembic_ini))
        console.print("🗄️   Current database revision:", style="bold blue")
        alembic_command.current(cfg, verbose=True)
        console.print("\n📋 Migration history (latest first):", style="bold blue")
        alembic_command.history(cfg, indicate_current=True)
    except Exception as exc:
        console.print(f"❌ Could not check migration status: {exc}", style="red")
        console.print(
            "💡 Verify PRSM_DATABASE_URL is set and the database is reachable.",
            style="yellow",
        )
        raise SystemExit(1)


@main.command()
@click.option("--dry-run", is_flag=True, help="Walk through setup without saving")
@click.option("--minimal", is_flag=True, help="Quick setup with smart defaults")
@click.option("--reset", is_flag=True, help="Reset all settings and re-run setup")
def setup(dry_run, minimal, reset):
    """Interactive first-run setup wizard for PRSM."""
    from prsm.cli_modules.setup_wizard import run_setup_wizard
    run_setup_wizard(dry_run=dry_run, minimal=minimal, reset=reset)


@main.group()
def node():
    """P2P node management commands"""
    pass


def _run_node_wizard() -> "NodeConfig":
    """Deprecated: redirects to the new `prsm setup` wizard.

    Kept for backward compatibility with `prsm node start --wizard`.
    Saves a minimal NodeConfig so the existing node start flow continues
    to work with any CLI overrides (ports, resources, etc.).
    """
    from prsm.node.config import NodeConfig

    console.print()
    console.print("  prsm node start --wizard is deprecated.")
    console.print("  Redirecting to the new setup wizard.", style="yellow")
    console.print("  Tip: Run `prsm setup` directly for the full experience.", style="dim")

    # Run the new setup wizard
    from prsm.cli_modules.setup_wizard import run_setup_wizard
    run_setup_wizard(minimal=False)

    # Also try to run migration so the legacy NodeConfig picks up the
    # settings written by the new wizard (~/.prsm/config.yaml → node_config.json).
    try:
        from prsm.cli_modules.migration import migrate_if_needed
        migrate_if_needed()
    except Exception:
        pass

    # Return a loaded NodeConfig so the caller (node start) continues normally
    return NodeConfig.load()


@node.command()
@click.option("--wizard", is_flag=True, help="Run interactive setup wizard")
@click.option("--background", "-b", is_flag=True, help="Start as background daemon")
@click.option("--p2p-port", default=None, type=int, help="P2P listen port (default: 9001)")
@click.option("--api-port", default=None, type=int, help="API listen port (default: 8000)")
@click.option("--bootstrap", default=None, help="Bootstrap node address (host:port)")
@click.option("--no-dashboard", is_flag=True, help="Disable live dashboard (static output)")
@click.option("--cpu", default=None, type=int, help="CPU allocation % (10-90)")
@click.option("--memory", default=None, type=int, help="RAM allocation % (10-90)")
@click.option("--storage", default=None, type=float, help="Storage to pledge in GB")
@click.option("--jobs", default=None, type=int, help="Max concurrent compute jobs")
def start(wizard: bool, background: bool, p2p_port: int, api_port: int, bootstrap: str, no_dashboard: bool,
          cpu: Optional[int], memory: Optional[int], storage: Optional[float], jobs: Optional[int]):
    """Start a PRSM network node with real P2P connectivity.

    Runs in the foreground by default. Use --background to run as a
    background daemon (equivalent to 'prsm daemon start').
    """
    from prsm.node.config import NodeConfig

    host = "127.0.0.1"
    port = 8000

    # Background mode — route to daemon
    if background:
        from prsm.cli_modules.daemon import daemon_start as _dstart
        _dstart(host=host, port=api_port or 8000)
        return

    if wizard:
        config = _run_node_wizard()
    else:
        # Load existing config or use defaults
        config = NodeConfig.load()

    # CLI overrides
    if p2p_port is not None:
        config.p2p_port = p2p_port
    if api_port is not None:
        config.api_port = api_port
    if bootstrap:
        config.bootstrap_nodes = [b.strip() for b in bootstrap.split(",")]
    
    # Resource CLI overrides
    if cpu is not None:
        if not 10 <= cpu <= 90:
            raise click.BadParameter("CPU allocation must be between 10-90%")
        config.cpu_allocation_pct = cpu
    if memory is not None:
        if not 10 <= memory <= 90:
            raise click.BadParameter("Memory allocation must be between 10-90%")
        config.memory_allocation_pct = memory
    if storage is not None:
        if storage <= 0:
            raise click.BadParameter("Storage must be a positive value in GB")
        config.storage_gb = storage
    if jobs is not None:
        if jobs < 1:
            raise click.BadParameter("Max concurrent jobs must be at least 1")
        config.max_concurrent_jobs = jobs
    
    # Save config if any resource overrides were provided
    if any(v is not None for v in [cpu, memory, storage, jobs]):
        config.save()

    # Run preflight diagnostics before starting the node
    results = _node_preflight_diagnostics(config)
    _render_preflight_summary(results)

    if _has_hard_preflight_failures(results):
        print("\n❌ Preflight checks failed. Cannot start node.")
        print("   Fix the issues above and try again.")
        sys.exit(1)

    # When using the live dashboard, suppress startup noise so the
    # terminal is clean.  The dashboard captures prsm.node.* activity
    # via its own log handler.
    if not no_dashboard:
        import logging as _logging
        import warnings as _warnings

        # Suppress Python logging below ERROR during startup
        _logging.root.setLevel(_logging.ERROR)
        # Suppress structlog info messages (MCP tool registrations)
        _logging.getLogger("prsm.compute").setLevel(_logging.ERROR)
        _logging.getLogger("prsm.core").setLevel(_logging.ERROR)
        # Suppress Python warnings (optional dependency notices)
        _warnings.filterwarnings("ignore")

        # Suppress structlog output (uses its own rendering pipeline)
        try:
            import structlog
            structlog.configure(
                wrapper_class=structlog.make_filtering_bound_logger(_logging.ERROR),
            )
        except ImportError:
            pass

        # Suppress uvicorn's CancelledError traceback on shutdown
        _logging.getLogger("uvicorn.error").setLevel(_logging.CRITICAL)

        console.print()
        console.print("  Starting PRSM Node...", style="bold green")
        console.print(f"  🖥️   Dashboard:  http://localhost:{config.api_port}/", style="bold cyan")

    else:
        console.print()
        console.print("=" * 60, style="bold green")
        console.print("  Starting PRSM Node", style="bold green")
        console.print("=" * 60, style="bold green")

    async def _run():
        import logging as _logging
        from prsm.node.node import PRSMNode

        prsm_node = PRSMNode(config)
        await prsm_node.initialize()
        await prsm_node.start()

        # Check for LLM backends and warn if only mock is available
        backends = detect_available_backends()
        if not backends.get("any_real_backend"):
            print()
            print("⚠️   No LLM API keys detected — inference will return mock responses.")
            print("    To enable real AI inference:")
            print("      1. cp .env.example .env")
            print("      2. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")
            print("      3. Restart the node")
            print()

        try:
            if no_dashboard:
                # Static table + sleep loop (original behavior)
                status = await prsm_node.get_status()
                console.print()
                table = Table(title="PRSM Node Status")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="green")
                table.add_row("Node ID", status["node_id"])
                table.add_row("Display Name", status["display_name"])
                table.add_row("Roles", ", ".join(status["roles"]))
                table.add_row("P2P Address", status["p2p_address"])
                table.add_row("API Address", status["api_address"])
                table.add_row("Dashboard", f"http://127.0.0.1:{config.api_port}/")
                table.add_row("FTNS Balance", f"{status['ftns_balance']:.2f}")
                bootstrap = status.get("peers", {}).get("bootstrap", {})
                if bootstrap.get("degraded_mode"):
                    table.add_row("Bootstrap", "DEGRADED local mode")
                    table.add_row(
                        "Limited Features",
                        "Remote peer discovery/collaboration may be unavailable until peers connect",
                    )
                elif bootstrap.get("success_node"):
                    table.add_row("Bootstrap", f"connected via {bootstrap['success_node']}")
                elif bootstrap.get("configured_nodes") == []:
                    table.add_row("Bootstrap", "none configured (first node/local mode)")
                if status.get("compute"):
                    res = status["compute"]["resources"]
                    table.add_row("CPU", f"{res['cpu_count']} cores")
                    table.add_row("RAM", f"{res['memory_total_gb']} GB")
                    table.add_row("GPU", res.get("gpu_name", "none") if res.get("gpu_available") else "none")
                if status.get("storage"):
                    st = status["storage"]
                    table.add_row("Storage", "connected" if st.get("storage_available", False) else "not available")
                    table.add_row("Storage Pledged", f"{st['pledged_gb']} GB")
                console.print(table)
                console.print()
                console.print("Node is running. Press Ctrl+C to stop.", style="bold")
                console.print()
                while True:
                    await asyncio.sleep(1)
            else:
                # Restore logging for prsm.node.* so the dashboard
                # activity log captures events during operation.
                _logging.getLogger("prsm.node").setLevel(_logging.INFO)

                # Clear startup messages before showing dashboard
                console.clear()

                from prsm.node.dashboard import NodeDashboard
                dashboard = NodeDashboard(prsm_node)
                await dashboard.run()
        except asyncio.CancelledError:
            pass
        finally:
            await prsm_node.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("\nNode stopped.", style="bold")


@node.command("stop")
@click.option("--timeout", default=10, help="Seconds to wait for graceful shutdown")
def node_stop(timeout: int):
    """Stop the background node daemon."""
    from prsm.cli_modules.daemon import daemon_stop as _stop
    _stop(timeout=timeout)


@node.command("restart")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--timeout", default=10, help="Seconds to wait for graceful shutdown")
def node_restart(host: str, port: int, timeout: int):
    """Restart the background node daemon."""
    from prsm.cli_modules.daemon import daemon_restart as _restart
    _restart(host=host, port=port, timeout=timeout)


@node.command("status")
@click.option("--format", "output_format",
              type=click.Choice(["text", "json"]), default="text",
              help="Output format")
def node_status(output_format: str):
    """Show background node daemon status."""
    from prsm.cli_modules.daemon import daemon_status as _status
    _status(output_format=output_format)


@node.command("logs")
@click.option("--lines", "-n", default=50, help="Number of lines to show")
@click.option("--follow", "-f", is_flag=True, help="Follow log output (tail -f)")
def node_logs(lines: int, follow: bool):
    """Show background node daemon logs."""
    from prsm.cli_modules.daemon import daemon_logs as _logs
    _logs(lines=lines, follow=follow)


@node.command("earnings")
@click.option(
    "--api-port", default=8000, type=int,
    help="Local API port (default 8000)",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def node_earnings(api_port: int, output_format: str):
    """Show this node's earnings dashboard.

    Aggregates 3 streams: royalty (claimable_wei) + heartbeat
    status + distribution timing. Backed by GET
    /admin/earnings-summary on the running node daemon.
    """
    import json
    import httpx

    url = f"http://127.0.0.1:{api_port}/admin/earnings-summary"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
    except httpx.RequestError as exc:
        console.print(
            f"[red]Cannot reach PRSM node at {url}[/red]\n"
            f"[dim]Start with: prsm node start[/dim]\n"
            f"[dim]Details: {exc}[/dim]"
        )
        sys.exit(2)

    if resp.status_code != 200:
        console.print(
            f"[red]/admin/earnings-summary returned "
            f"{resp.status_code}[/red]: {resp.text}"
        )
        sys.exit(1)

    body = resp.json()

    if output_format == "json":
        console.print(json.dumps(body, indent=2))
        return

    op = body.get("operator_address") or "(PRSM_OPERATOR_ADDRESS unset)"
    console.print(f"[bold]PRSM Operator Earnings[/bold]")
    console.print(f"  Operator: {op}")
    console.print()

    royalty = body.get("royalty", {})
    if royalty.get("available"):
        wei = royalty.get("claimable_wei", 0)
        ftns = wei / 1e18
        console.print(
            f"  [green]Royalty:[/green]      {ftns:.6f} FTNS claimable"
        )
    else:
        err = royalty.get("error")
        suffix = f" — error: {err}" if err else ""
        console.print(f"  [yellow]Royalty:[/yellow]      not wired{suffix}")

    hb = body.get("heartbeat", {})
    if hb.get("available"):
        if hb.get("never_recorded"):
            console.print(
                f"  [red]Heartbeat:[/red]    never recorded — "
                f"slashing imminent"
            )
        elif hb.get("expired"):
            console.print(
                f"  [red]Heartbeat:[/red]    EXPIRED — slashing window open"
            )
        elif hb.get("at_risk"):
            console.print(
                f"  [yellow]Heartbeat:[/yellow]    at-risk — "
                f"{hb['grace_remaining']}s grace remaining"
            )
        else:
            console.print(
                f"  [green]Heartbeat:[/green]    ok — "
                f"{hb['grace_remaining']}s grace "
                f"(of {hb['grace_seconds']}s)"
            )
    else:
        err = hb.get("error")
        suffix = f" — error: {err}" if err else ""
        console.print(f"  [yellow]Heartbeat:[/yellow]    not wired{suffix}")

    dist = body.get("distribution", {})
    if dist.get("available"):
        if dist.get("never_distributed"):
            console.print(f"  [yellow]Distribution:[/yellow] never run yet")
        else:
            secs = dist.get("seconds_since", 0)
            hrs = secs // 3600
            console.print(
                f"  [green]Distribution:[/green] last run {hrs}h ago"
            )
    else:
        err = dist.get("error")
        suffix = f" — error: {err}" if err else ""
        console.print(
            f"  [yellow]Distribution:[/yellow] not wired{suffix}"
        )


def _node_admin_history(
    *, api_port: int, path: str, label: str,
    output_format: str, limit: int = 20,
    row_renderer=None,
):
    """Shared helper for the 4 admin-history CLI commands.

    `row_renderer` is a callable(entry: dict) -> str that
    each command supplies for typed rendering. Without it the
    raw entry dict is printed (debug fallback).
    """
    import json
    import datetime
    import httpx

    url = (
        f"http://127.0.0.1:{api_port}{path}?limit={limit}"
    )
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
    except httpx.RequestError as exc:
        console.print(
            f"[red]Cannot reach PRSM node at {url}[/red]\n"
            f"[dim]Start with: prsm node start[/dim]\n"
            f"[dim]Details: {exc}[/dim]"
        )
        sys.exit(2)

    if resp.status_code != 200:
        if resp.status_code == 503:
            console.print(
                f"[yellow]{label} log not configured.[/yellow]\n"
                f"[dim]{resp.json().get('detail', 'unknown')}[/dim]"
            )
            sys.exit(0)  # 503 is not an error — just unwired
        console.print(
            f"[red]{path} returned {resp.status_code}[/red]: "
            f"{resp.text}"
        )
        sys.exit(1)

    body = resp.json()
    if output_format == "json":
        console.print(json.dumps(body, indent=2))
        return

    entries = body.get("entries", [])
    total = body.get("total", 0)
    console.print(
        f"[bold]PRSM {label}[/bold] (showing {len(entries)} of {total}):"
    )
    if not entries:
        console.print(f"  [dim]No entries[/dim]")
        return
    for e in entries:
        ts = e.get("timestamp", 0)
        try:
            t = datetime.datetime.fromtimestamp(ts).strftime(
                "%H:%M:%S",
            )
        except Exception:
            t = "????"
        if row_renderer is not None:
            console.print(f"  {t}  {row_renderer(e)}")
        else:
            console.print(f"  {t}  {e}")


def _short_addr(addr: str, *, head: int = 8, tail: int = 6) -> str:
    """Truncate 0x... addresses for column display."""
    if not addr or len(addr) <= head + tail + 2:
        return addr or "?"
    return f"{addr[:head]}..{addr[-tail:]}"


def _render_webhook_row(e: dict) -> str:
    success = e.get("success")
    # Escape brackets so rich doesn't interpret as markup tags
    marker = r"\[ok]" if success else r"\[!]"
    code = e.get("status_code", "?")
    detail = (
        "delivered" if success
        else (e.get("error") or "no error msg")
    )
    return (
        f"{marker} {e.get('event', '?'):<28}  "
        f"status={code:<5} {detail}"
    )


def _render_slash_row(e: dict) -> str:
    return (
        f"{e.get('kind', '?'):<28}  "
        f"provider={_short_addr(e.get('provider', '?'))}  "
        f"slash_id={e.get('slash_id', '?')[:14]}..."
    )


def _render_heartbeat_row(e: dict) -> str:
    on_ts = e.get("onchain_timestamp", 0)
    return (
        f"provider={_short_addr(e.get('provider', '?'))}  "
        f"onchain_ts={on_ts}"
    )


def _render_distribution_row(e: dict) -> str:
    creator = e.get("to_creator", 0) / 1e18
    operator = e.get("to_operator", 0) / 1e18
    grant = e.get("to_grant", 0) / 1e18
    return (
        f"creator={creator:.4f} operator={operator:.4f} "
        f"grant={grant:.4f} FTNS"
    )


@node.command("slash-history")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
)
@click.option("--limit", default=20, type=int)
def node_slash_history(api_port, output_format, limit):
    """Show recent on-chain slash events."""
    _node_admin_history(
        api_port=api_port,
        path="/admin/slash-history",
        label="Slash Events",
        output_format=output_format,
        limit=limit,
        row_renderer=_render_slash_row,
    )


@node.command("heartbeats")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
)
@click.option("--limit", default=20, type=int)
def node_heartbeats(api_port, output_format, limit):
    """Show recent on-chain heartbeats."""
    _node_admin_history(
        api_port=api_port,
        path="/admin/heartbeat-history",
        label="Heartbeats",
        output_format=output_format,
        limit=limit,
        row_renderer=_render_heartbeat_row,
    )


@node.command("distributions")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
)
@click.option("--limit", default=20, type=int)
def node_distributions(api_port, output_format, limit):
    """Show recent on-chain Distributed events."""
    _node_admin_history(
        api_port=api_port,
        path="/admin/distribution-history",
        label="Distributions",
        output_format=output_format,
        limit=limit,
        row_renderer=_render_distribution_row,
    )


@node.command("webhooks")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
)
@click.option("--limit", default=20, type=int)
def node_webhooks(api_port, output_format, limit):
    """Show recent webhook dispatch attempts."""
    _node_admin_history(
        api_port=api_port,
        path="/admin/webhook-history",
        label="Webhooks",
        output_format=output_format,
        limit=limit,
        row_renderer=_render_webhook_row,
    )


@node.command("bootstrap")
@click.option(
    "--api-port", default=8000, type=int,
    help="Local API port (default 8000)",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def node_bootstrap(api_port: int, output_format: str):
    """Show P2P bootstrap status: primary, fallback,
    active URL, SPOF posture.

    Sprint 380 — third surface for the sprint-375 multi-
    bootstrap fields. Same data as /bootstrap/status JSON +
    prsm_bootstrap_status MCP tool; the CLI completes the
    operator-trifecta (REST / MCP / shell).
    """
    import json
    import httpx

    url = f"http://127.0.0.1:{api_port}/bootstrap/status"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
    except httpx.RequestError as exc:
        console.print(
            f"[red]Cannot reach PRSM node at "
            f"{url}[/red]\n"
            f"[dim]Start with: prsm node start[/dim]\n"
            f"[dim]Details: {exc}[/dim]"
        )
        sys.exit(2)

    if resp.status_code == 503:
        detail = "(not wired)"
        try:
            detail = resp.json().get("detail", detail)
        except Exception:  # noqa: BLE001
            pass
        console.print(
            f"[yellow]Bootstrap discovery not wired.[/yellow]"
            f"\n[dim]{detail}[/dim]"
        )
        sys.exit(1)
    if resp.status_code != 200:
        console.print(
            f"[red]/bootstrap/status returned "
            f"{resp.status_code}[/red]: {resp.text}"
        )
        sys.exit(1)

    body = resp.json()

    if output_format == "json":
        console.print(json.dumps(body, indent=2))
        return

    # Health summary marker — operators triage by color.
    connected = int(body.get("connected", 0) or 0)
    degraded = bool(body.get("degraded", False))
    client_state = body.get("client_state", "?")
    if connected > 0 and not degraded and (
        client_state == "connected"
    ):
        marker = "[green]✓ healthy[/green]"
    elif degraded or client_state == "dead":
        marker = "[red]⚠ degraded[/red]"
    else:
        marker = "[yellow]⚠ disconnected[/yellow]"

    console.print(f"[bold]PRSM Bootstrap Status[/bold] — {marker}")
    console.print(f"  client_state:           {client_state}")
    console.print(f"  connected:              {connected}")
    console.print(f"  attempted:              {body.get('attempted', 0)}")
    console.print(
        f"  discovered peers:       "
        f"{body.get('discovered_peer_count', 0)}"
    )

    # Sprint 375/376 — active_url + fallback config
    active_url = body.get("active_url")
    if active_url:
        # Strip scheme for compact rendering — same pattern
        # as the prsm_node_health MCP wrapper.
        short = active_url
        if "://" in short:
            short = short.split("://", 1)[1]
        console.print(
            f"  [bold]active URL:[/bold]             {short}"
        )
    else:
        console.print(
            "  [bold]active URL:[/bold]             "
            "[dim](none — all candidates failed)[/dim]"
        )

    fb_enabled = body.get("bootstrap_fallback_enabled")
    if fb_enabled is not None:
        if fb_enabled:
            console.print(
                "  fallback enabled:       [green]yes[/green]"
            )
        else:
            console.print(
                "  fallback enabled:       "
                "[yellow]no (single-host posture)[/yellow]"
            )

    # Counter snapshot (sprint 324)
    console.print()
    console.print(
        "  peer_join_events:       "
        f"{body.get('peer_join_events', 0)}"
    )
    console.print(
        "  peer_leave_events:      "
        f"{body.get('peer_leave_events', 0)}"
    )
    console.print(
        "  stale_evictions:        "
        f"{body.get('stale_evictions', 0)}"
    )
    console.print(
        "  reconnect_attempts:     "
        f"{body.get('reconnect_attempts', 0)}"
    )
    console.print(
        "  reconnect_successes:    "
        f"{body.get('reconnect_successes', 0)}"
    )

    # Candidate URLs (primary + fallback)
    bnodes = body.get("bootstrap_nodes") or []
    if bnodes:
        console.print()
        console.print(
            f"  bootstrap_nodes ({len(bnodes)} primary):"
        )
        for n in bnodes:
            marker = (
                "[green]●[/green]" if n == active_url
                else "[dim]○[/dim]"
            )
            console.print(f"    {marker} {n}")

    fb_nodes = body.get("bootstrap_fallback_nodes") or []
    if fb_nodes:
        console.print(
            f"  fallback_nodes ({len(fb_nodes)}):"
        )
        for n in fb_nodes:
            marker = (
                "[green]●[/green]" if n == active_url
                else "[dim]○[/dim]"
            )
            console.print(f"    {marker} {n}")


@node.command("bootstrap-test")
@click.option(
    "--url", "urls", multiple=True,
    help=(
        "Bootstrap URL(s) to test. Repeatable. When unset, "
        "tests the canonical fleet (US + EU + APAC defaults "
        "from prsm/node/config.py)."
    ),
)
@click.option(
    "--timeout", default=10.0, type=float,
    help="Per-host probe timeout in seconds (default 10)",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
)
def node_bootstrap_test(urls, timeout, output_format):
    """Probe canonical bootstrap fleet from this machine.

    Sprint 385 — operator-trifecta complement to
    `prsm node bootstrap` (sprint 380). That one shows
    THIS node's bootstrap-registration state. This one
    probes ALL canonical bootstraps from wherever the
    operator is standing and reports TCP / TLS / WSS
    health for each. Diagnostic for "is my regional
    bootstrap up, or is something local broken?"

    Doesn't require a running PRSM node.
    """
    import asyncio
    import json
    from prsm.cli_helpers.bootstrap_probe import (
        ProbeStatus,
        canonical_bootstrap_urls,
        probe_fleet,
    )

    target_urls = list(urls) if urls else (
        canonical_bootstrap_urls()
    )
    if not target_urls:
        console.print(
            "[red]No bootstrap URLs to test.[/red]\n"
            "[dim]Pass --url <wss://...> or check your "
            "BOOTSTRAP_PRIMARY env vars.[/dim]"
        )
        sys.exit(2)

    fleet = asyncio.run(
        probe_fleet(target_urls, timeout_seconds=timeout),
    )

    if output_format == "json":
        console.print(json.dumps(fleet.to_dict(), indent=2))
        sys.exit(0 if fleet.all_healthy else 1)

    # Header marker
    if fleet.all_healthy:
        marker = "[green]✓ all healthy[/green]"
    elif fleet.any_healthy:
        marker = "[yellow]⚠ partial[/yellow]"
    else:
        marker = "[red]⚠ all degraded[/red]"
    console.print(
        f"[bold]PRSM Bootstrap Fleet Probe[/bold] — "
        f"{marker} "
        f"({fleet.healthy_count}/{fleet.total_count} reachable)"
    )
    console.print()

    for h in fleet.hosts:
        if h.status == ProbeStatus.OK:
            status_str = "[green]✓ ok[/green]"
        elif h.status == ProbeStatus.TIMEOUT:
            status_str = "[yellow]⚠ timeout[/yellow]"
        else:
            status_str = (
                f"[red]✗ {h.status.value}[/red]"
            )
        url_short = h.url
        if "://" in url_short:
            url_short = url_short.split("://", 1)[1]
        latency_str = (
            f"{h.latency_ms:.0f}ms"
            if h.latency_ms is not None else "-"
        )
        console.print(
            f"  {status_str:<22}  {url_short:<50}  "
            f"{latency_str}"
        )
        # Per-layer detail line
        layer_marks = []
        for label, ok in (
            ("TCP", h.tcp_ok),
            ("TLS", h.tls_ok),
            ("WSS", h.wss_ok),
        ):
            if ok:
                layer_marks.append(
                    f"[green]{label}[/green]"
                )
            else:
                layer_marks.append(f"[dim]{label}[/dim]")
        console.print(
            "    " + " · ".join(layer_marks)
            + (
                f"  ([dim]cert: {h.cert_subject} "
                f"issued by {h.cert_issuer}[/dim])"
                if h.cert_subject else ""
            )
        )
        # Sprint 591 — surface SAN-mismatch warning in text output
        # (sprint 590 already populates h.san_mismatch + JSON; text
        # output had only the subject CN before this sprint).
        if getattr(h, "san_mismatch", False):
            san_list = ", ".join(h.cert_san_dns) if h.cert_san_dns else "<empty>"
            console.print(
                f"    [yellow]⚠ SAN mismatch[/yellow] — "
                f"cert covers [{san_list}], "
                f"not [cyan]{h.host}[/cyan]. "
                f"Strict-TLS clients will fail."
            )
        if h.error:
            console.print(f"    [red]error:[/red] {h.error}")

    # Exit code mirrors the CI-friendly contract:
    #   0 = all reachable
    #   1 = some degraded
    #   2 = all degraded
    if fleet.all_healthy:
        sys.exit(0)
    elif fleet.any_healthy:
        sys.exit(1)
    else:
        sys.exit(2)


def _node_admin_trigger(*, api_port: int, path: str, label: str):
    """Shared helper for action triggers — POSTs to admin
    endpoints, renders tx_hash + status."""
    import httpx

    url = f"http://127.0.0.1:{api_port}{path}"
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url)
    except httpx.RequestError as exc:
        console.print(
            f"[red]Cannot reach PRSM node at {url}[/red]\n"
            f"[dim]Details: {exc}[/dim]"
        )
        sys.exit(2)

    if resp.status_code == 503:
        detail = resp.json().get("detail", "not wired")
        console.print(
            f"[yellow]{label} unavailable.[/yellow]\n"
            f"[dim]{detail}[/dim]"
        )
        sys.exit(1)
    if resp.status_code == 502:
        detail = resp.json().get("detail", "chain error")
        console.print(
            f"[red]{label} failed on-chain.[/red]\n"
            f"[dim]{detail}[/dim]"
        )
        sys.exit(1)
    if resp.status_code != 200:
        console.print(
            f"[red]{path} returned {resp.status_code}[/red]: "
            f"{resp.text}"
        )
        sys.exit(1)

    body = resp.json()
    tx_hash = body.get("tx_hash", "?")
    status = body.get("status", "?")
    console.print(
        f"[green]{label} submitted on-chain.[/green]\n"
        f"  tx_hash: {tx_hash}\n"
        f"  status:  {status}"
    )


@node.command("trigger-heartbeat")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--yes", "-y", is_flag=True,
    help="Skip confirmation prompt",
)
def node_trigger_heartbeat(api_port, yes):
    """Manually record a heartbeat on-chain.

    Use when the HeartbeatScheduler crashed/paused and you
    want to avoid the slashing window opening. Caller pays gas.
    """
    if not yes:
        click.confirm(
            "This will submit an on-chain transaction. Continue?",
            abort=True,
        )
    _node_admin_trigger(
        api_port=api_port,
        path="/admin/heartbeat/trigger",
        label="Heartbeat",
    )


@node.command("trigger-distribution")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--yes", "-y", is_flag=True,
    help="Skip confirmation prompt",
)
def node_trigger_distribution(api_port, yes):
    """Manually invoke pull_and_distribute on-chain.

    Use when the PullAndDistributeScheduler crashed/paused or
    to force an emission round before the next cadence tick.
    Caller pays gas.
    """
    if not yes:
        click.confirm(
            "This will submit an on-chain transaction. Continue?",
            abort=True,
        )
    _node_admin_trigger(
        api_port=api_port,
        path="/admin/distribution/trigger",
        label="Distribution",
    )


@node.command("claim-royalty")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--execute", is_flag=True,
    help="Execute the claim on-chain (default: dry-run only)",
)
def node_claim_royalty(api_port, execute):
    """Claim accumulated royalties from RoyaltyDistributor.

    Default: dry-run that shows claimable amount without
    on-chain action. Pass --execute to send the tx.
    """
    import json
    import httpx

    url = f"http://127.0.0.1:{api_port}/wallet/royalty/claim"
    payload = {"dry_run": not execute}
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload)
    except httpx.RequestError as exc:
        console.print(
            f"[red]Cannot reach PRSM node at {url}[/red]\n"
            f"[dim]Details: {exc}[/dim]"
        )
        sys.exit(2)

    if resp.status_code == 503:
        detail = resp.json().get("detail", "not wired")
        console.print(
            f"[yellow]RoyaltyDistributor not wired.[/yellow]\n"
            f"[dim]{detail}[/dim]"
        )
        sys.exit(1)

    if resp.status_code != 200:
        console.print(
            f"[red]/wallet/royalty/claim returned "
            f"{resp.status_code}[/red]: {resp.text}"
        )
        sys.exit(1)

    body = resp.json()
    status = body.get("status", "?")
    if status == "DRY_RUN":
        wei = body.get("claimable_wei", 0)
        ftns = wei / 1e18
        console.print(
            f"[bold]Dry run — no on-chain action[/bold]\n"
            f"  Claimable: {ftns:.6f} FTNS ({wei} wei)\n"
            f"  Re-run with --execute to claim on-chain."
        )
    elif status == "SKIPPED_ZERO":
        console.print(
            "[yellow]Nothing to claim — claimable balance is 0.[/yellow]"
        )
    elif status == "EXECUTED":
        tx_hash = body.get("tx_hash", "?")
        wei = body.get("amount_claimed_wei", 0)
        ftns = wei / 1e18
        console.print(
            f"[green]Royalty claim submitted on-chain.[/green]\n"
            f"  Amount:  {ftns:.6f} FTNS\n"
            f"  tx_hash: {tx_hash}"
        )
    else:
        console.print(json.dumps(body, indent=2))


@node.command("install")
@click.option("--dry-run", is_flag=True, help="Print service file without installing")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def node_install(dry_run: bool, host: str, port: int):
    """Install node as a system service (launchd / systemd)."""
    from prsm.cli_modules.daemon import daemon_service_install as _install
    _install(dry_run=dry_run, host=host, port=port)


@node.command("uninstall")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def node_uninstall(yes: bool):
    """Remove the node system service."""
    from prsm.cli_modules.daemon import daemon_service_uninstall as _uninstall
    _uninstall(yes=yes)


@node.command()
def peers():
    """List connected + known peers (sprint 574 enhanced)."""
    from prsm.node.config import NodeConfig
    from prsm.node.identity import load_node_identity

    config = NodeConfig.load()
    identity = load_node_identity(config.identity_path)

    if not identity:
        console.print("No node identity found. Run 'prsm setup' first.", style="yellow")
        return

    console.print(f"Node ID: {identity.node_id}", style="bold cyan")
    console.print(f"P2P address: ws://{config.listen_host}:{config.p2p_port}", style="cyan")
    console.print()

    # Try connecting to the running node's API to get live peer data
    # Sprint 574: honor PRSM_API_PORT env override (operators on
    # non-default ports e.g. droplet at 8002)
    try:
        import httpx, os as _os
        _env_port = (_os.environ.get("PRSM_API_PORT", "") or "").strip()
        try:
            _api_port = int(_env_port) if _env_port else config.api_port
        except ValueError:
            _api_port = config.api_port
        resp = httpx.get(f"http://127.0.0.1:{_api_port}/peers", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            connected = data.get("connected", [])
            known = data.get("known", [])

            connected_ids = {p["peer_id"] for p in connected}

            if connected:
                table = Table(title="Connected Peers")
                table.add_column("Peer ID", style="cyan")
                table.add_column("Address", style="green")
                table.add_column("Name", style="magenta")
                table.add_column("Direction", style="blue")
                for p in connected:
                    table.add_row(
                        p["peer_id"][:16] + "...",
                        p["address"],
                        p.get("display_name", ""),
                        "outbound" if p.get("outbound") else "inbound",
                    )
                console.print(table)
            else:
                console.print("No peers connected.", style="dim")

            # Sprint 574 — also show known-but-unconnected so operators
            # see the gap between bootstrap-discovered and actually-dialed
            known_only = [
                k for k in known if k.get("node_id") not in connected_ids
            ]
            if known_only:
                console.print()
                tbl2 = Table(title="Known (not connected)")
                tbl2.add_column("Peer ID", style="cyan")
                tbl2.add_column("Address", style="green")
                tbl2.add_column("Name", style="magenta")
                tbl2.add_column("Capabilities", style="dim")
                for k in known_only:
                    tbl2.add_row(
                        (k.get("node_id", "") or "")[:16] + "...",
                        k.get("address", ""),
                        k.get("display_name", "") or "",
                        ", ".join(k.get("capabilities", []) or []),
                    )
                console.print(tbl2)

            console.print(f"\nConnected: {data.get('connected_count', 0)}  "
                          f"Known: {data.get('known_count', 0)}")
            return
    except Exception:
        pass

    console.print("Node is not running. Start it with 'prsm node start'.", style="yellow")


# ── Sprint 585 — §7 readiness aggregate ──────────────────────────


def _probe_anchor_outcome():
    """Returns (outcome, error_str_or_None) for anchor construction."""
    import os as _os
    addr = (_os.environ.get("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", "") or "").strip()
    rpc = _os.environ.get(
        "PRSM_BASE_RPC_URL", "https://mainnet.base.org",
    )
    if not addr:
        return "unset", None
    try:
        from prsm.security.publisher_key_anchor.client import (
            PublisherKeyAnchorClient,
        )
        PublisherKeyAnchorClient(contract_address=addr, rpc_url=rpc)
        return "ok", None
    except Exception as exc:  # noqa: BLE001
        return "construction_failed", f"{type(exc).__name__}: {exc}"


def _probe_stake_bond_outcome():
    import os as _os
    addr = (_os.environ.get("PRSM_STAKE_BOND_ADDRESS", "") or "").strip()
    rpc = _os.environ.get(
        "PRSM_BASE_RPC_URL", "https://mainnet.base.org",
    )
    if not addr:
        return "unset", None
    try:
        from prsm.economy.web3.stake_manager import StakeManagerClient
        StakeManagerClient(contract_address=addr, rpc_url=rpc)
        return "ok", None
    except Exception as exc:  # noqa: BLE001
        return "construction_failed", f"{type(exc).__name__}: {exc}"


def _probe_rpc_outcome():
    import os as _os
    import httpx as _httpx
    rpc = _os.environ.get(
        "PRSM_BASE_RPC_URL", "https://mainnet.base.org",
    )
    try:
        resp = _httpx.post(
            rpc,
            json={
                "jsonrpc": "2.0", "method": "eth_chainId",
                "params": [], "id": 1,
            },
            timeout=10.0,
        )
        if resp.status_code != 200:
            return "error", f"HTTP {resp.status_code}"
        body = resp.json()
        if body.get("result") is None:
            return "error", f"no result: {body!r}"[:200]
        return "ok", None
    except _httpx.HTTPError as exc:
        return "unreachable", f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return "error", f"{type(exc).__name__}: {exc}"


@node.command("section7-readiness")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def section7_readiness_cli(output_format: str):
    """Aggregate §7 production-readiness check.

    Sprint 585 — runs anchor-probe + stake-bond-probe + rpc-probe
    in one shot, reports overall readiness. Exit 0 only when ALL
    three probes return ok (suitable for CI gating).
    """
    import json
    components = {
        "anchor": dict(zip(("outcome", "error"), _probe_anchor_outcome())),
        "stake_bond": dict(zip(("outcome", "error"), _probe_stake_bond_outcome())),
        "rpc": dict(zip(("outcome", "error"), _probe_rpc_outcome())),
    }
    overall = (
        "ready"
        if all(c["outcome"] == "ok" for c in components.values())
        else "not_ready"
    )
    payload = {"overall": overall, "components": components}

    if output_format == "json":
        click.echo(json.dumps(payload, indent=2))
        raise SystemExit(0 if overall == "ready" else 1)

    table = Table(title="§7 production-readiness (sprint 585)")
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Outcome", style="green", no_wrap=True)
    table.add_column("Error", style="red", no_wrap=False)
    for name, c in components.items():
        out = c["outcome"]
        out_style = "[green]✓ ok[/green]" if out == "ok" else f"[red]{out}[/red]"
        table.add_row(name, out_style, c.get("error") or "")
    console.print(table)
    if overall == "ready":
        console.print(
            "\n[green]✓ ready[/green] — all three §7 components "
            "pass preflight. Safe to set "
            "PRSM_PARALLAX_TRUST_STACK_KIND=production."
        )
    else:
        console.print(
            "\n[yellow]not_ready[/yellow] — fix the failing "
            "component(s) above before flipping production."
        )
        raise SystemExit(1)


# ── Sprint 584 — RPC probe ───────────────────────────────────────


@node.command("rpc-probe")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def rpc_probe_cli(output_format: str):
    """Probe whether PRSM_BASE_RPC_URL is reachable + responds.

    Sprint 584 — completes the §7 production preflight trifecta
    with sprints 581 (anchor) + 583 (stake-bond). Tests RPC
    reachability via eth_chainId JSON-RPC call so operators can
    distinguish "wrong contract address" from "unreachable RPC"
    when 581/583 fail with construction_failed.
    """
    import json
    import os as _os
    import httpx as _httpx

    rpc_url = _os.environ.get(
        "PRSM_BASE_RPC_URL", "https://mainnet.base.org",
    )
    outcome = "ok"
    error = None
    chain_id_hex = None
    try:
        resp = _httpx.post(
            rpc_url,
            json={
                "jsonrpc": "2.0",
                "method": "eth_chainId",
                "params": [],
                "id": 1,
            },
            timeout=10.0,
        )
        if resp.status_code != 200:
            outcome = "error"
            error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        else:
            try:
                body = resp.json()
            except ValueError as exc:
                outcome = "error"
                error = f"non-JSON response: {exc}"
            else:
                chain_id_hex = body.get("result")
                if chain_id_hex is None:
                    outcome = "error"
                    error = f"no 'result' field: {body!r}"[:200]
    except _httpx.HTTPError as exc:
        outcome = "unreachable"
        error = f"{type(exc).__name__}: {exc}"

    payload = {
        "PRSM_BASE_RPC_URL": rpc_url,
        "outcome": outcome,
        "chain_id_hex": chain_id_hex,
        "error": error,
    }
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2))
        raise SystemExit(0 if outcome == "ok" else 1)

    if outcome == "ok":
        console.print(
            f"[green]✓ ok[/green] — RPC at [cyan]{rpc_url}[/cyan] "
            f"responded with chain_id=[magenta]{chain_id_hex}[/magenta]"
        )
        return
    console.print(
        f"[red]✗ {outcome}[/red]: {error}\n"
        f"[dim]rpc_url={rpc_url!r}[/dim]\n"
        f"[dim]Check the URL is reachable (try `curl -s {rpc_url}`) "
        f"and that any auth tokens (Infura/Alchemy key) are valid.[/dim]"
    )
    raise SystemExit(1)


# ── Sprint 581 — anchor probe ────────────────────────────────────


@node.command("anchor-probe")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def anchor_probe_cli(output_format: str):
    """Probe whether PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS wires a
    working PublisherKeyAnchorClient.

    Operator preflight before flipping PRSM_PARALLAX_TRUST_STACK_KIND
    =production. Surfaces:
      - The env value (or <unset>)
      - Configured RPC URL
      - Construction outcome: ok / unset / construction_failed
      - Error detail when construction fails

    Sprint 581 — closes the slow "set env, restart daemon, grep
    logs" feedback loop with a one-shot operator-side check.
    """
    import json
    import os as _os
    anchor_addr = (
        _os.environ.get("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", "") or ""
    ).strip()
    rpc_url = _os.environ.get(
        "PRSM_BASE_RPC_URL", "https://mainnet.base.org",
    )
    outcome = "ok"
    error = None
    if not anchor_addr:
        outcome = "unset"
    else:
        try:
            from prsm.security.publisher_key_anchor.client import (
                PublisherKeyAnchorClient,
            )
            PublisherKeyAnchorClient(
                contract_address=anchor_addr,
                rpc_url=rpc_url,
            )
        except Exception as exc:  # noqa: BLE001
            outcome = "construction_failed"
            error = f"{type(exc).__name__}: {exc}"

    payload = {
        "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS": anchor_addr or None,
        "PRSM_BASE_RPC_URL": rpc_url,
        "outcome": outcome,
        "error": error,
    }
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2))
        raise SystemExit(0 if outcome == "ok" else 1)

    if outcome == "ok":
        console.print(
            f"[green]✓ ok[/green] — PublisherKeyAnchorClient constructed "
            f"against [cyan]{anchor_addr}[/cyan] on "
            f"[magenta]{rpc_url}[/magenta]"
        )
        return
    if outcome == "unset":
        console.print(
            "[yellow]PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS is unset[/yellow] "
            "— required to flip PRSM_PARALLAX_TRUST_STACK_KIND=production. "
            "Set it to the deployed Phase-3.x.3 anchor contract address."
        )
        raise SystemExit(1)
    # construction_failed
    console.print(
        f"[red]✗ construction_failed[/red]: {error}\n"
        f"[dim]anchor_addr={anchor_addr!r}, rpc_url={rpc_url!r}[/dim]\n"
        f"[dim]Check the address resolves to a deployed contract on "
        f"the RPC chain, and that the RPC endpoint is reachable.[/dim]"
    )
    raise SystemExit(1)


# ── Sprint 583 — stake-bond probe ────────────────────────────────


@node.command("stake-bond-probe")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format",
)
def stake_bond_probe_cli(output_format: str):
    """Probe whether PRSM_STAKE_BOND_ADDRESS wires a working
    StakeManagerClient.

    Sprint 583 — mirror of sprint-581 anchor-probe for the
    sibling §7 production env var (sprint 561). Operator
    preflight before flipping PRSM_PARALLAX_TRUST_STACK_KIND
    =production with stake-weighted trust.
    """
    import json
    import os as _os
    stake_addr = (
        _os.environ.get("PRSM_STAKE_BOND_ADDRESS", "") or ""
    ).strip()
    rpc_url = _os.environ.get(
        "PRSM_BASE_RPC_URL", "https://mainnet.base.org",
    )
    outcome = "ok"
    error = None
    if not stake_addr:
        outcome = "unset"
    else:
        try:
            from prsm.economy.web3.stake_manager import (
                StakeManagerClient,
            )
            StakeManagerClient(
                contract_address=stake_addr,
                rpc_url=rpc_url,
            )
        except Exception as exc:  # noqa: BLE001
            outcome = "construction_failed"
            error = f"{type(exc).__name__}: {exc}"

    payload = {
        "PRSM_STAKE_BOND_ADDRESS": stake_addr or None,
        "PRSM_BASE_RPC_URL": rpc_url,
        "outcome": outcome,
        "error": error,
    }
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2))
        raise SystemExit(0 if outcome == "ok" else 1)

    if outcome == "ok":
        console.print(
            f"[green]✓ ok[/green] — StakeManagerClient constructed "
            f"against [cyan]{stake_addr}[/cyan] on "
            f"[magenta]{rpc_url}[/magenta]"
        )
        return
    if outcome == "unset":
        console.print(
            "[yellow]PRSM_STAKE_BOND_ADDRESS is unset[/yellow] "
            "— required for sprint-561 real stake-weighted trust. "
            "Without it, production trust-stack falls back to "
            "ZeroStakeLookup placeholder (every node treated as "
            "zero stake)."
        )
        raise SystemExit(1)
    console.print(
        f"[red]✗ construction_failed[/red]: {error}\n"
        f"[dim]stake_addr={stake_addr!r}, rpc_url={rpc_url!r}[/dim]\n"
        f"[dim]Check the address resolves to a deployed StakeBond "
        f"contract and the RPC endpoint is reachable.[/dim]"
    )
    raise SystemExit(1)


# ── Sprint 579 — trust-stack observability ──────────────────────


_TRUST_STACK_ENV_VARS = [
    ("PRSM_PARALLAX_TRUST_STACK_KIND",
     ("mock", "production"), "mock",
     "Top-level trust-stack kind. mock=permissive (dev); "
     "production=4-component verification (sprints 558-562)."),
    ("PRSM_PARALLAX_PROFILE_SOURCE_KIND",
     ("in_memory", "dht"), "in_memory",
     "Inner ProfileSource (sprint 576). dht hooks Phase 2 "
     "ProfileDHT integration; Phase 1 falls back to in_memory."),
    ("PRSM_PARALLAX_CONSENSUS_SUBMITTER_KIND",
     ("logging", "onchain"), "logging",
     "Consensus mismatch submitter (sprint 577). onchain hooks "
     "Phase 2 ChallengeRecord → Phase 7.1x ABI dispatch; Phase 1 "
     "falls back to logging."),
    ("PRSM_PARALLAX_CHAIN_EXECUTOR_KIND",
     ("stub", "rpc"), "stub",
     "Inference chain executor (sprint 578). rpc hooks Phase 2 "
     "make_rpc_chain_executor wiring; Phase 1 falls back to stub."),
]


def _resolve_trust_stack_entry(name, valid, default):
    """Return (kind, env_value, status) for one trust-stack env var."""
    import os as _os
    raw = (_os.environ.get(name, "") or "").strip()
    env_value = raw if raw else None
    val = (raw or default).lower()
    if val in valid:
        # Phase 1: only the FIRST valid kind is fully active; the
        # second one is the Phase 2 hook that currently falls back.
        if val == default:
            status = "active (default)" if env_value is None else "active"
        else:
            # Phase 2 hook kinds — surface as "phase 2 pending"
            # so operators don't assume they got the real thing.
            status = "phase 2 pending (falls back to default)"
        return val, env_value, status
    return default, env_value, f"unknown_falls_back_to_{default}"


@node.command("trust-stack")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    help="Output format: 'text' (Rich table) or 'json' (agent-parseable)",
)
def trust_stack_cli(output_format: str):
    """Inspect §7 trust-stack Phase-1 env config.

    Reports the 4 ParallaxScheduledExecutor component env vars
    + their effective kinds. Useful for verifying which Phase 2
    hooks an operator has opted into.

    Sprint 579 closes the observability gap left by sprints
    558-562/576/577/578 — operators no longer need to spelunk
    daemon startup logs to see effective trust-stack config.
    """
    import json
    entries = {}
    for (name, valid, default, descr) in _TRUST_STACK_ENV_VARS:
        kind, env_value, status = _resolve_trust_stack_entry(
            name, valid, default,
        )
        entries[name] = {
            "kind": kind,
            "env_value": env_value,
            "status": status,
            "valid": list(valid),
            "default": default,
            "description": descr,
        }

    if output_format == "json":
        click.echo(json.dumps(entries, indent=2))
        return

    # no_wrap on all columns + console.width override prevents Rich
    # auto-truncation in narrow terminals or test harness output.
    from rich.console import Console as _RichConsole
    _wide = _RichConsole(width=140)
    table = Table(title="Trust-stack Phase-1 env config (sprint 579)")
    table.add_column("Env var", style="cyan", no_wrap=True)
    table.add_column("Effective kind", style="green", no_wrap=True)
    table.add_column("Env value", style="magenta", no_wrap=True)
    table.add_column("Status", style="blue", no_wrap=True)
    for name, e in entries.items():
        table.add_row(
            name,
            e["kind"],
            (e["env_value"] or "<unset>"),
            e["status"],
        )
    _wide.print(table)
    _wide.print(
        "\n[dim]Sprints 558-562/576/577/578 plumb four Phase-1 "
        "hooks for §7 trust-stack. Phase 2 lands real impl per "
        "component additively.[/dim]"
    )


# ── Sprint 574 — fleet-ops CLI quartet ───────────────────────────


def _api_base() -> str:
    """Resolve the local daemon's API base URL.

    Sprint 574 — honor PRSM_API_PORT env var FIRST so operators
    running daemons on non-default ports (e.g., bootstrap-us
    droplet at 8002 alongside bootstrap-server-v2) get the right
    target without editing NodeConfig. Falls back to NodeConfig
    loaded value (default 8000) when env unset.
    """
    import os
    env_port = (os.environ.get("PRSM_API_PORT", "") or "").strip()
    if env_port:
        try:
            port = int(env_port)
        except ValueError:
            port = 0
        if port > 0:
            return f"http://127.0.0.1:{port}"
    from prsm.node.config import NodeConfig
    cfg = NodeConfig.load()
    return f"http://127.0.0.1:{cfg.api_port}"


def _daemon_down_message():
    console.print(
        "Could not reach the daemon. Is it running? "
        "Start it with [cyan]prsm node start[/cyan].",
        style="yellow",
    )


@node.command()
@click.argument("address")
def dial(address: str):
    """Dial a peer by address (host:port or ws://host:port).

    Sprint 574 — wraps POST /peers/connect. Useful when auto-dial
    sweep (sprint 573) failed or when joining a peer not in the
    bootstrap registry.
    """
    import httpx
    try:
        resp = httpx.post(
            f"{_api_base()}/peers/connect",
            json={"address": address},
            timeout=30.0,
        )
    except httpx.HTTPError:
        _daemon_down_message()
        raise SystemExit(1)

    if resp.status_code == 200:
        data = resp.json()
        console.print(
            f"[green]✓ Connected[/green] to "
            f"[cyan]{data.get('peer_id', '?')[:16]}...[/cyan] "
            f"at [magenta]{data.get('address', address)}[/magenta]"
        )
        return

    if resp.status_code == 502:
        console.print(
            f"[yellow]Could not connect to {address}[/yellow] "
            f"(502: transport returned None — peer unreachable, "
            f"firewall, or handshake failure).",
        )
    elif resp.status_code == 503:
        console.print(
            "[yellow]Daemon transport not initialized yet "
            "(503).[/yellow] Try again in a moment."
        )
    else:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        console.print(
            f"[red]dial failed[/red] (HTTP {resp.status_code}): {detail}"
        )
    raise SystemExit(1)


@node.command()
@click.argument("cid")
@click.option(
    "--output", "-o", "output_path", default=None,
    help="Write decoded content to this file instead of stdout.",
)
def fetch(cid: str, output_path):
    """Fetch content by CID from the network.

    Sprint 574 — wraps GET /content/retrieve. Base64-decodes the
    response and writes to stdout (default) or --output file.
    """
    import base64
    import httpx
    try:
        resp = httpx.get(
            f"{_api_base()}/content/retrieve/{cid}",
            timeout=60.0,
        )
    except httpx.HTTPError:
        _daemon_down_message()
        raise SystemExit(1)

    if resp.status_code != 200:
        console.print(
            f"[red]fetch failed[/red] (HTTP {resp.status_code})"
        )
        raise SystemExit(1)

    data = resp.json()
    status = data.get("status")
    if status != "success":
        err = data.get("error") or status or "unknown"
        console.print(f"[yellow]{status or 'not_found'}:[/yellow] {err}")
        raise SystemExit(1)

    try:
        payload = base64.b64decode(data.get("data", ""))
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]base64 decode failed[/red]: {exc}")
        raise SystemExit(1)

    if output_path:
        with open(output_path, "wb") as fh:
            fh.write(payload)
        console.print(
            f"[green]✓ Wrote {len(payload)} bytes[/green] to "
            f"[cyan]{output_path}[/cyan] "
            f"(filename={data.get('filename', '?')})"
        )
    else:
        # Render as text if it decodes cleanly; raw bytes otherwise
        try:
            console.print(payload.decode("utf-8"))
        except UnicodeDecodeError:
            console.print(f"[dim]{len(payload)} bytes (binary)[/dim]")
            click.echo(payload, nl=False)


@node.command()
@click.argument("file_path")
def share(file_path: str):
    """Upload a file's text contents to the network + print the CID.

    Sprint 574 — wraps POST /content/upload. Output is the CID so
    operators can pipe to ``prsm node dial`` / share workflows.
    """
    import httpx
    try:
        if file_path == "-":
            import sys
            text = sys.stdin.read()
        else:
            with open(file_path, "r", encoding="utf-8") as fh:
                text = fh.read()
    except OSError as exc:
        console.print(f"[red]cannot read[/red] {file_path}: {exc}")
        raise SystemExit(1)

    try:
        resp = httpx.post(
            f"{_api_base()}/content/upload",
            json={"text": text},
            timeout=60.0,
        )
    except httpx.HTTPError:
        _daemon_down_message()
        raise SystemExit(1)

    if resp.status_code != 200:
        console.print(
            f"[red]share failed[/red] (HTTP {resp.status_code}): "
            f"{resp.text}"
        )
        raise SystemExit(1)

    data = resp.json()
    cid = data.get("cid", "")
    console.print(
        f"[green]✓ Shared[/green] [cyan]{cid}[/cyan] "
        f"({data.get('size_bytes', 0)} bytes, "
        f"filename={data.get('filename', '?')})"
    )


@node.command()
@click.option("--format", "output_format",
              type=click.Choice(["text", "json"]), default="text",
              help="Output format: 'text' (human) or 'json' (agent-parseable)")
def info(output_format: str):
    """Show node identity and configuration.

    Use --format json for machine-readable output (AI agent consumption).
    """
    from prsm.node.config import NodeConfig
    from prsm.node.identity import load_node_identity

    config = NodeConfig.load()
    identity = load_node_identity(config.identity_path)

    if not identity:
        if output_format == "json":
            _agent_error("No node identity found. Run: prsm setup")
        console.print("No node identity found. Run 'prsm setup' first.", style="yellow")
        return

    # Sprint 534 F58 fix: surface BOTH primary + fallback
    # bootstrap lists so operators see the full failover chain
    # (not just the single dead bootstrap1 that pre-sprint-534
    # configs persisted).
    fallback_nodes = list(
        getattr(config, "bootstrap_fallback_nodes", []) or []
    )
    data = {
        "ok": True,
        "node_id": identity.node_id,
        "display_name": config.display_name,
        "public_key_b64": identity.public_key_b64,
        "roles": [r.value for r in config.roles],
        "p2p_port": config.p2p_port,
        "api_port": config.api_port,
        "data_dir": config.data_dir,
        "bootstrap_nodes": config.bootstrap_nodes or [],
        "bootstrap_fallback_nodes": fallback_nodes,
    }

    if output_format == "json":
        _agent_output(data)
        return

    table = Table(title="PRSM Node Info")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Node ID", data["node_id"])
    table.add_row("Display Name", data["display_name"])
    table.add_row("Public Key", data["public_key_b64"][:32] + "...")
    table.add_row("Roles", ", ".join(data["roles"]))
    table.add_row("P2P Port", str(data["p2p_port"]))
    table.add_row("API Port", str(data["api_port"]))
    table.add_row("Data Dir", data["data_dir"])
    table.add_row(
        "Bootstrap Primary",
        ", ".join(data["bootstrap_nodes"]) or "none",
    )
    if fallback_nodes:
        table.add_row(
            "Bootstrap Fallback",
            ", ".join(fallback_nodes),
        )
    console.print(table)


@node.command("configure")
@click.option("--show", is_flag=True, help="Show current configuration and exit")
@click.option("--cpu", default=None, type=int, help="CPU allocation % (10-90)")
@click.option("--memory", default=None, type=int, help="RAM allocation % (10-90)")
@click.option("--storage", default=None, type=float, help="Storage to pledge in GB")
@click.option("--jobs", default=None, type=int, help="Max concurrent compute jobs")
@click.option("--gpu-pct", default=None, type=int, help="GPU allocation % (10-100)")
@click.option("--upload-limit", default=None, type=float, help="Upload bandwidth limit in Mbps (0=unlimited)")
@click.option("--active-hours", default=None, type=str, help="Active hours range (e.g., '22-8' for 10pm-8am, 'off' for always on)")
@click.option("--active-days", default=None, type=str, help="Active days (e.g., 'mon,tue,wed' or 'weekdays' or 'weekends')")
def configure(show: bool, cpu: Optional[int], memory: Optional[int], storage: Optional[float],
              jobs: Optional[int], gpu_pct: Optional[int], upload_limit: Optional[float],
              active_hours: Optional[str], active_days: Optional[str]):
    """Configure node resource settings.
    
    Without arguments, runs interactively to prompt for settings.
    With --show, prints current configuration.
    With specific flags, updates only those settings.
    """
    from prsm.node.config import NodeConfig
    
    config = NodeConfig.load()
    
    # Handle --show flag — redirect to new prsm config system
    if show:
        # Try the new PRSMConfig system first, fall back to legacy display
        try:
            from prsm.cli_modules.config_schema import PRSMConfig
            cfg = PRSMConfig.load()
            console.print()
            console.print("  PRSM Configuration", style="bold cyan")
            console.print("  [dim]Note: Also run `prsm config show` for the full display", style="dim")
            console.print(f"  Role:      {cfg.node_role.value}  (new config)")
            console.print(f"  CPU:       {cfg.cpu_pct}%")
            console.print(f"  Memory:    {cfg.memory_pct}%")
            console.print(f"  Storage:   {cfg.storage_gb} GB")
            console.print(f"  P2P Port:  {cfg.p2p_port}")
            console.print(f"  API Port:  {cfg.api_port}")
            console.print(f"  MCP:       {'enabled' if cfg.mcp_server_enabled else 'disabled'}")
            console.print(f"  Config:    {PRSMConfig.config_path()}")
            console.print()
            return
        except Exception:
            pass
        _show_configuration(config)
        return
    
    # Check if any flags were provided
    has_flags = any(v is not None for v in [cpu, memory, storage, jobs, gpu_pct, upload_limit, active_hours, active_days])
    
    if has_flags:
        # Update only specified fields
        changes = []
        
        if cpu is not None:
            if not 10 <= cpu <= 90:
                raise click.BadParameter("CPU allocation must be between 10-90%")
            old_val = config.cpu_allocation_pct
            config.cpu_allocation_pct = cpu
            changes.append(f"CPU allocation: {old_val}% → {cpu}%")
        
        if memory is not None:
            if not 10 <= memory <= 90:
                raise click.BadParameter("Memory allocation must be between 10-90%")
            old_val = config.memory_allocation_pct
            config.memory_allocation_pct = memory
            changes.append(f"Memory allocation: {old_val}% → {memory}%")
        
        if storage is not None:
            if storage <= 0:
                raise click.BadParameter("Storage must be a positive value in GB")
            old_val = config.storage_gb
            config.storage_gb = storage
            changes.append(f"Storage pledged: {old_val} GB → {storage} GB")
        
        if jobs is not None:
            if jobs < 1:
                raise click.BadParameter("Max concurrent jobs must be at least 1")
            old_val = config.max_concurrent_jobs
            config.max_concurrent_jobs = jobs
            changes.append(f"Max concurrent jobs: {old_val} → {jobs}")
        
        if gpu_pct is not None:
            if not 10 <= gpu_pct <= 100:
                raise click.BadParameter("GPU allocation must be between 10-100%")
            old_val = config.gpu_allocation_pct
            config.gpu_allocation_pct = gpu_pct
            changes.append(f"GPU allocation: {old_val}% → {gpu_pct}%")
        
        if upload_limit is not None:
            if upload_limit < 0:
                raise click.BadParameter("Upload limit cannot be negative")
            old_val = config.upload_mbps_limit
            config.upload_mbps_limit = upload_limit
            old_str = f"{old_val} Mbps" if old_val > 0 else "unlimited"
            new_str = f"{upload_limit} Mbps" if upload_limit > 0 else "unlimited"
            changes.append(f"Upload limit: {old_str} → {new_str}")
        
        if active_hours is not None:
            start, end = parse_active_hours(active_hours)
            old_start, old_end = config.active_hours_start, config.active_hours_end
            config.active_hours_start = start
            config.active_hours_end = end
            old_str = f"{old_start:02d}-{old_end:02d}" if old_start is not None and old_end is not None else "always on"
            new_str = f"{start:02d}-{end:02d}" if start is not None and end is not None else "always on"
            changes.append(f"Active hours: {old_str} → {new_str}")
        
        if active_days is not None:
            days = parse_active_days(active_days)
            old_days = config.active_days
            config.active_days = days
            day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            old_str = ", ".join(day_names[d] for d in old_days) if old_days else "every day"
            new_str = ", ".join(day_names[d] for d in days) if days else "every day"
            changes.append(f"Active days: {old_str} → {new_str}")
        
        config.save()
        
        console.print("✅ Configuration updated:", style="bold green")
        for change in changes:
            console.print(f"   {change}")
        console.print(f"\n   Config saved to {config.config_path}")
        
    else:
        # Interactive mode
        _run_interactive_configure(config)


def _show_configuration(config: "NodeConfig") -> None:
    """Display current node configuration in human-readable format."""
    from prsm.node.compute_provider import detect_resources
    
    # Detect actual system resources
    resources = detect_resources()
    
    # Format role display
    role_display = " + ".join(r.value for r in config.roles)
    
    console.print()
    console.print("PRSM Node Resource Configuration", style="bold magenta")
    console.print("=" * 50)
    console.print(f"  Role:             {role_display}", style="cyan")
    
    # CPU allocation
    cpu_offered = round(resources.cpu_count * config.cpu_allocation_pct / 100, 1)
    console.print(f"  CPU allocation:   {config.cpu_allocation_pct}% of {resources.cpu_count} cores → {cpu_offered:.1f} cores offered")
    
    # Memory allocation
    mem_offered = round(resources.memory_total_gb * config.memory_allocation_pct / 100, 1)
    console.print(f"  RAM allocation:   {config.memory_allocation_pct}% of {resources.memory_total_gb:.1f} GB → {mem_offered:.1f} GB offered")
    
    # Concurrent jobs
    console.print(f"  Concurrent jobs:  {config.max_concurrent_jobs} slots")
    
    # GPU info
    if resources.gpu_available:
        gpu_offered = round(resources.gpu_memory_gb * config.gpu_allocation_pct / 100, 1)
        console.print(f"  GPU:              {resources.gpu_name} ({resources.gpu_memory_gb:.1f} GB) — {config.gpu_allocation_pct}% → {gpu_offered:.1f} GB offered")
    else:
        console.print(f"  GPU:              not detected")
    
    # Storage
    console.print(f"  Storage pledged:  {config.storage_gb:.1f} GB")
    
    # Upload limit
    if config.upload_mbps_limit > 0:
        console.print(f"  Upload limit:     {config.upload_mbps_limit:.1f} Mbps")
    else:
        console.print(f"  Upload limit:     unlimited")
    
    # Active hours
    if config.active_hours_start is not None and config.active_hours_end is not None:
        console.print(f"  Active hours:     {config.active_hours_start:02d}:00 - {config.active_hours_end:02d}:00")
    else:
        console.print(f"  Active hours:     always on")
    
    # Active days
    if config.active_days:
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        days_str = ", ".join(day_names[d] for d in config.active_days)
        console.print(f"  Active days:      {days_str}")
    else:
        console.print(f"  Active days:      every day")
    
    console.print()


def _run_interactive_configure(config: "NodeConfig") -> None:
    """Deprecated: redirects to the new `prsm config` system.

    Runs the legacy config wizard to update the old NodeConfig (kept for
    backward compat), then also runs the new PRSMConfig wizard so both
    configs stay in sync.
    """
    from prsm.node.compute_provider import detect_resources

    console.print()
    console.print("  NOTE: prsm node configure (interactive) is deprecated.", style="yellow")
    console.print("  Tip: Use `prsm config` for the full experience.", style="dim")
    console.print()

    # Run the legacy interactive prompts to update NodeConfig
    resources = detect_resources()
    console.print("  System: {} cores, {:.1f} GB RAM".format(
        resources.cpu_count, resources.memory_total_gb), style="dim")
    console.print()

    cpu = click.prompt("  CPU allocation for compute jobs (%)", default=config.cpu_allocation_pct, type=int)
    config.cpu_allocation_pct = max(10, min(90, cpu))
    mem = click.prompt("  Memory allocation for compute jobs (%)", default=config.memory_allocation_pct, type=int)
    config.memory_allocation_pct = max(10, min(90, mem))
    storage = click.prompt("  Storage to pledge (GB)", default=config.storage_gb, type=float)
    config.storage_gb = storage
    jobs = click.prompt("  Max concurrent compute jobs", default=config.max_concurrent_jobs, type=int)
    config.max_concurrent_jobs = max(1, jobs)
    if resources.gpu_available:
        gpu = click.prompt("  GPU allocation %", default=config.gpu_allocation_pct, type=int)
        config.gpu_allocation_pct = max(10, min(100, gpu))

    config.save()
    console.print(f"\n  ✅ Saved to {config.config_path}", style="green")

    # Also sync to the new config system
    try:
        from prsm.cli_modules.migration import migrate_if_needed
        migrate_if_needed()
    except Exception:
        pass


@node.command()
def benchmark():
    """Run hardware profiler and display compute tier classification."""
    from prsm.compute.wasm.profiler import HardwareProfiler
    profiler = HardwareProfiler()
    profile = profiler.detect()
    click.echo(f"Hardware Profile:")
    # Sprint 533 F51 fix: omit MHz if reading was junk (0).
    # Apple Silicon and some platforms don't expose real freq;
    # showing "0 MHz" or "4 MHz" is worse than just core count.
    if profile.cpu_freq_mhz and profile.cpu_freq_mhz >= 100:
        click.echo(
            f"  CPU: {profile.cpu_cores} cores @ "
            f"{profile.cpu_freq_mhz:.0f} MHz"
        )
    else:
        click.echo(f"  CPU: {profile.cpu_cores} cores")
    click.echo(f"  GPU: {profile.gpu_name or 'None detected'}")
    if profile.gpu_vram_gb > 0:
        click.echo(f"  VRAM: {profile.gpu_vram_gb:.1f} GB")
    click.echo(f"  RAM: {profile.ram_total_gb:.1f} GB total, {profile.ram_available_gb:.1f} GB available")
    click.echo(f"  TFLOPS: {profile.tflops_fp32:.2f} FP32")
    click.echo(f"  Compute Tier: {profile.compute_tier.value.upper()}")
    click.echo(f"  Thermal: {profile.thermal_class.value}")



# ============================================================================
# COMPUTE COMMANDS
# ============================================================================

@main.group()
def compute():
    """Compute job management commands"""
    pass


@compute.command()
@click.option('--prompt', required=True, help='Prompt to process')
@click.option('--model', default='nwtn', help='Model to use (default: nwtn)')
@click.option('--max-tokens', default=1000, type=int, help='Maximum tokens in response')
@click.option('--budget', type=float, help='Maximum FTNS to spend')
def submit(prompt: str, model: str, max_tokens: int, budget: Optional[float]):
    """Submit a compute job to the local node."""
    from prsm.compute.jobs_store import create_job

    job = create_job(prompt=prompt, model=model, max_tokens=max_tokens, budget=budget)

    console.print(f"[bold green]✅ Job submitted[/bold green]")
    console.print(f"   Job ID:  {job['job_id']}")
    console.print(f"   Model:   {job['model']}")
    console.print(f"   Status:  {job['status']}")
    if job.get('budget'):
        console.print(f"   Budget:  {job['budget']} FTNS")
    console.print()
    console.print("  Jobs are processed by your local P2P node.", style="dim")
    console.print("  Check status with:  prsm compute status " + job['job_id'], style="dim")


@compute.command()
@click.argument('job_id')
def status(job_id: str):
    """Get status of a compute job."""
    from prsm.compute.jobs_store import get_job

    job = get_job(job_id)
    if not job:
        console.print(f"[red]Job not found: {job_id}[/red]")
        raise SystemExit(1)

    console.print(f"\n[bold]Job Status: {job_id}[/bold]\n")
    table = Table(show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Status", job.get("status", "unknown"))
    table.add_row("Model", job.get("model", "—"))
    table.add_row("Prompt", (job.get("prompt", "") or "")[:80])
    if job.get("result"):
        table.add_row("Result", str(job["result"])[:120])
    if job.get("error"):
        table.add_row("Error", job["error"])
    if job.get("budget"):
        table.add_row("Budget", f"{job['budget']} FTNS")
    console.print(table)


@compute.command()
@click.argument('job_id')
def result(job_id: str):
    """Get result of a completed compute job."""
    from prsm.compute.jobs_store import get_job

    job = get_job(job_id)
    if not job:
        console.print(f"[red]Job not found: {job_id}[/red]")
        raise SystemExit(1)

    if job.get("status") == "completed" and job.get("result"):
        console.print("[bold green]📄 Job Result:[/bold green]")
        console.print(job["result"])
    else:
        console.print(f"[yellow]Job is {job.get('status', 'unknown')} — no result yet.[/yellow]")
        if job.get("error"):
            console.print(f"[red]Error: {job['error']}[/red]")


@compute.command()
@click.argument('job_id')
def cancel(job_id: str):
    """Cancel a running compute job."""
    from prsm.compute.jobs_store import update_job, get_job

    job = get_job(job_id)
    if not job:
        console.print(f"[red]Job not found: {job_id}[/red]")
        raise SystemExit(1)

    if job.get("status") in ("completed", "failed"):
        console.print(f"[yellow]Job is already {job['status']} — cannot cancel.[/yellow]")
        return

    update_job(job_id, status="cancelled")
    console.print(f"[green]✅ Job {job_id} cancelled[/green]")


@compute.command("list")
@click.option('--limit', default=10, type=int, help='Maximum number of jobs to list')
def list_compute_jobs(limit: int):
    """List recent compute jobs."""
    from prsm.compute.jobs_store import list_jobs as _list

    jobs = _list(limit=limit)
    if not jobs:
        console.print("No compute jobs yet.", style="dim")
        console.print("Submit one with:  prsm compute submit --prompt 'your question'", style="dim")
        return

    table = Table(title="Recent Compute Jobs")
    table.add_column("Job ID", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Model", style="blue")
    table.add_column("Created", style="green")
    for job in jobs:
        created = job.get("created_at", "?")
        if isinstance(created, (int, float)):
            import datetime
            created = datetime.datetime.fromtimestamp(created).isoformat()[:19]
        table.add_row(
            (job.get("job_id") or "?")[:16],
            job.get("status", "?"),
            job.get("model", "—"),
            str(created)[:19],
        )
    console.print(table)


@compute.command("run")
@click.option('--prompt', default=None, help='Prompt to process (legacy NWTN path)')
@click.option('--query', default=None, help='Query for full forge pipeline (Rings 1-10)')
@click.option('--budget', default=10.0, type=float, help='FTNS budget for the job')
@click.option('--privacy', default='standard', type=click.Choice(['none', 'standard', 'high', 'maximum']), help='Privacy level for confidential compute')
@click.option('--api', default='http://127.0.0.1:8000', help='Node API URL')
def compute_run(prompt: str, query: str, budget: float, privacy: str, api: str):
    """Submit a compute job to your running daemon.

    Two modes:

      --prompt: Legacy path — routes directly to NWTN orchestrator.

      --query:  Full forge pipeline (Rings 1-10) — decomposes query via LLM,
                dispatches WASM mobile agents to edge nodes, aggregates results,
                settles FTNS payments, applies differential privacy.

    Requires a running daemon (`prsm node start`).
    """
    import httpx

    if not prompt and not query:
        console.print("[red]Either --prompt or --query is required[/red]")
        raise SystemExit(1)

    # Determine which path to use
    if query:
        # Enforce minimum budget for forge pipeline
        if budget <= 0:
            console.print("[red]FTNS budget is required for forge pipeline execution.[/red]")
            console.print("  PRSM's distributed compute network requires FTNS tokens to pay")
            console.print("  compute providers and data owners for their resources.")
            console.print()
            console.print("  [dim]Get a cost estimate first:[/dim]")
            console.print(f'    prsm compute quote "{query}"')
            console.print()
            console.print("  [dim]Then run with a budget:[/dim]")
            console.print(f'    prsm compute run --query "{query}" --budget 1.0')
            raise SystemExit(1)

        # Full forge pipeline (Rings 1-10)
        endpoint = f"{api}/compute/forge"
        payload = {
            "query": query,
            "budget_ftns": budget,
            "privacy_level": privacy,
        }
        console.print(f"[bold]Forge Pipeline[/bold] (Rings 1-10)")
        console.print(f"  Query: {query}")
        console.print(f"  Budget: {budget:.2f} FTNS")
        console.print(f"  Privacy: {privacy}")
        console.print()
    else:
        # Legacy NWTN path
        endpoint = f"{api}/compute/query"
        payload = {"prompt": prompt, "budget": budget}

    try:
        resp = httpx.post(endpoint, json=payload, timeout=120.0)
    except httpx.ConnectError:
        console.print(f"[red]Cannot connect to node API at {api}[/red]")
        console.print("  [dim]Start your daemon first: prsm node start[/dim]")
        raise SystemExit(1)

    if resp.status_code == 200:
        data = resp.json()

        if query:
            # Forge result display
            route = data.get("route", "unknown")
            console.print(f"[bold green]Result[/bold green] (route: {route}):")
            console.print(f"  {data.get('response', str(data.get('result', data)))}")
            console.print()
            console.print(f"  [dim]Job ID: {data.get('job_id', '?')}[/dim]")
            console.print(f"  [dim]Route: {route}[/dim]")
            console.print(f"  [dim]Budget: {data.get('budget_ftns', 0):.2f} FTNS[/dim]")
            traces = data.get("traces_collected", 0)
            if traces:
                console.print(f"  [dim]Training traces collected: {traces}[/dim]")
        else:
            # Legacy result display
            console.print(f"[bold green]Result:[/bold green]")
            text = data.get("response", str(data.get("result", data)))
            console.print(f"  {text}")
            cost = data.get("ftns_charged", data.get("ftns_cost"))
            if cost is not None:
                console.print(f"  [dim]Cost: {cost:.6f} FTNS[/dim]")
            job_id = data.get("job_id")
            if job_id:
                console.print(f"  [dim]Job ID: {job_id}[/dim]")

    elif resp.status_code == 503:
        console.print(f"[red]Service not available[/red]")
        detail = resp.json().get("detail", "") if resp.headers.get("content-type", "").startswith("application/json") else resp.text
        console.print(f"  [dim]{detail[:300]}[/dim]")
        raise SystemExit(1)
    elif resp.status_code == 404:
        console.print(f"[red]API endpoint not found on daemon[/red]")
        console.print("  [dim]Upgrade: pip install --upgrade prsm-network[/dim]")
        raise SystemExit(1)
    else:
        console.print(f"[red]Request failed: HTTP {resp.status_code}[/red]")
        if resp.text:
            console.print(f"  [dim]{resp.text[:300]}[/dim]")
        raise SystemExit(1)


@compute.command("quote")
@click.argument("query")
@click.option("--shards", default=3, help="Estimated number of data shards")
@click.option("--tier", default="t2", help="Hardware tier (t1-t4)")
def compute_quote(query, shards, tier):
    """Get a cost estimate for a compute query."""
    from prsm.economy.pricing.engine import PricingEngine

    engine = PricingEngine()
    quote = engine.quote_swarm_job(
        shard_count=shards,
        hardware_tier=tier,
        estimated_pcu_per_shard=50.0,
    )

    click.echo(f"Cost Quote for: {query}")
    click.echo(f"  Compute: {quote.compute_cost} FTNS")
    click.echo(f"  Data: {quote.data_cost} FTNS")
    click.echo(f"  Network Fee: {quote.network_fee} FTNS")
    click.echo(f"  Total: {quote.total} FTNS")


@main.command("demo")
def demo():
    """DEPRECATED: Ring 1-10 demo removed in v1.6.0.

    Use `prsm node start` to launch a real daemon, or `prsm demo-multinode`
    for a multi-node P2P walkthrough.
    """
    # Sprint 533 F46 fix: previously the help-text promised a Ring 1-10
    # demo that the command no longer runs. First-impression UX
    # blocker for new users. Updated docstring + actionable error +
    # nonzero exit so misuse is signaled clearly.
    console.print(
        "[yellow]⚠️  `prsm demo` was removed in v1.6.0 scope alignment.[/yellow]\n"
        "[bold]Try instead:[/bold]\n"
        "  • [cyan]prsm node start[/cyan]        — launch a real daemon\n"
        "  • [cyan]prsm demo-multinode[/cyan]    — multi-node P2P walkthrough\n"
        "  • [cyan]prsm setup[/cyan]             — first-run config wizard\n"
    )
    raise SystemExit(1)


@main.command("mcp-server")
def mcp_server_cmd():
    """Start the PRSM MCP server for LLM tool access.

    Exposes 17 PRSM tools to any MCP-compatible LLM (Claude, Gemini, etc.)
    via stdio protocol.

    Configure in Claude Desktop (~/.claude/claude_desktop_config.json):

        {"mcpServers": {"prsm": {"command": "python", "args": ["-m", "prsm.mcp_entry"]}}}
    """
    import subprocess, sys
    # Use the clean entry point to avoid stdout noise
    proc = subprocess.run([sys.executable, "-m", "prsm.mcp_entry"])
    raise SystemExit(proc.returncode)


@main.command("demo-multinode")
@click.option("--nodes", default=3, type=int, help="Number of nodes to spawn")
def compute_demo(nodes: int):
    """Run a multi-node P2P demonstration.

    Spawns local nodes and demonstrates:
    1. Escrow creation (FTNS locked before job runs)
    2. Job offer broadcast via gossip
    3. Cross-node job acceptance and execution
    4. Result consensus (multiple providers must agree)
    5. Payment release to winning provider
    """
    import asyncio
    from prsm.node.multinode_demo import MultiNodeDemo

    async def _run():
        demo = MultiNodeDemo()
        await demo.run()

    asyncio.run(_run())


# ============================================================================
# FTNS COMMANDS
# ============================================================================

@main.group()
def ftns():
    """FTNS token management commands"""
    pass


@ftns.command()
@click.option("--api-url", default=None, help="PRSM API URL (default: from stored credentials)")
def balance(api_url: str) -> None:
    """Show your FTNS token balance."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    try:
        response = httpx.get(
            f"{url}/api/v1/ftns/balance",
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        table = Table(title="FTNS Balance")
        table.add_column("Type", style="cyan")
        table.add_column("Amount (FTNS)", style="green", justify="right")
        table.add_row("Total",     f"{data.get('balance', 0):.6f}")
        table.add_row("Available", f"{data.get('available_balance', 0):.6f}")
        table.add_row("Locked",    f"{data.get('locked_balance', 0):.6f}")
        console.print(table)
        console.print(f"\n[dim]User ID: {data.get('user_id', '?')}[/dim]")
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@ftns.command()
@click.option("--to",          required=True,             help="Recipient user ID")
@click.option("--amount",      required=True, type=float, help="Amount in FTNS")
@click.option("--description", default="",                help="Optional transfer note")
@click.option("--api-url",     default=None,              help="PRSM API URL (default: from stored credentials)")
def transfer(to: str, amount: float, description: str, api_url: str) -> None:
    """Transfer FTNS tokens to another user."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    if amount <= 0:
        console.print("❌ Amount must be positive", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)
    console.print(f"💸 Transferring {amount:.6f} FTNS → {to}...", style="bold blue")

    try:
        response = httpx.post(
            f"{url}/api/v1/ftns/transfer",
            # Correct request body: 'recipient' and 'description', not 'to_address'/'memo'
            json={"recipient": to, "amount": amount, "description": description},
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        console.print("✅ Transfer successful!", style="bold green")
        # Correct response fields: 'status', 'amount', 'recipient' (no 'transaction_id'/'fee')
        console.print(f"   Recipient : {data.get('recipient')}")
        console.print(f"   Amount    : {data.get('amount', 0):.6f} FTNS")
        console.print(f"   Status    : {data.get('status')}")
    elif response.status_code == 402:
        console.print("❌ Insufficient FTNS balance", style="red")
        raise SystemExit(1)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Transfer failed: HTTP {response.status_code}", style="red")
        console.print(f"   {response.text[:200]}")
        raise SystemExit(1)


@ftns.command("transfer-onchain")
@click.option("--to",      required=True,             help="Recipient Ethereum address (0x…)")
@click.option("--amount",  required=True, type=float, help="Amount in FTNS")
@click.option("--api-url", default=None,              help="PRSM API URL (default: from stored credentials)")
def transfer_onchain(to: str, amount: float, api_url: str) -> None:
    """Transfer FTNS tokens on-chain to an Ethereum address.

    Unlike `prsm ftns transfer` (which moves FTNS between user IDs
    on the off-chain DAG ledger), this command broadcasts a real
    ERC-20 Transfer on Base mainnet using the daemon's loaded
    FTNS_WALLET_PRIVATE_KEY. Requires daemon to be started with
    PRSM_ONCHAIN_FTNS=1 + FTNS_WALLET_PRIVATE_KEY set.
    """
    import httpx

    if amount <= 0:
        console.print("❌ Amount must be positive (> 0)", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)
    console.print(
        f"🔗 Broadcasting on-chain transfer: {amount:.6f} FTNS → {to}...",
        style="bold blue",
    )

    try:
        response = httpx.post(
            f"{url}/wallet/transfer/onchain",
            json={"to_address": to, "amount_ftns": amount},
            timeout=90.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        console.print("✅ Transfer confirmed on-chain!", style="bold green")
        console.print(f"   tx_hash      : 0x{data.get('tx_hash', '').lstrip('0x')}")
        console.print(f"   block_number : {data.get('block_number')}")
        console.print(f"   status       : {data.get('status')}")
        console.print(f"   from         : {data.get('from_address')}")
        console.print(f"   to           : {data.get('to_address')}")
        console.print(f"   amount_ftns  : {data.get('amount_ftns')}")
    elif response.status_code == 503:
        detail = ""
        try:
            detail = response.json().get("detail", "")
        except Exception:
            detail = response.text[:300]
        console.print("❌ On-chain ledger not available:", style="red")
        console.print(f"   {detail}")
        raise SystemExit(1)
    elif response.status_code == 422:
        detail = ""
        try:
            detail = response.json().get("detail", "")
        except Exception:
            detail = response.text[:300]
        console.print(f"❌ Invalid input: {detail}", style="red")
        raise SystemExit(1)
    else:
        console.print(
            f"❌ Transfer failed: HTTP {response.status_code}", style="red",
        )
        console.print(f"   {response.text[:300]}")
        raise SystemExit(1)


@ftns.command()
@click.option("--amount",      required=True, type=float, help="FTNS to stake")
@click.option("--lock-days",   default=30,    type=int,   help="Lock duration in days (default: 30)")
@click.option("--api-url",     default=None,              help="PRSM API URL (default: from stored credentials)")
def stake(amount: float, lock_days: int, api_url: str) -> None:
    """
    Stake FTNS tokens for governance voting power.

    Staked tokens are locked for the specified duration and grant
    proportional voting power on governance proposals.
    """
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)
    console.print(f"🔒 Staking {amount:.6f} FTNS for {lock_days} days...", style="bold blue")

    try:
        # Correct endpoint: /api/v1/governance/stake (not /api/v1/ftns/stake)
        # Correct param name: lock_duration_days (not lock_period)
        response = httpx.post(
            f"{url}/api/v1/governance/stake",
            json={"amount": amount, "lock_duration_days": lock_days},
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        staking = data.get("data", {}).get("staking", {})
        console.print("✅ Staking successful!", style="bold green")
        console.print(f"   Staked        : {staking.get('staked_amount', amount)} FTNS")
        console.print(f"   Voting power  : {staking.get('voting_power', '?')}")
        console.print(f"   Unlock date   : {staking.get('unlock_date', '?')[:10]}")
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    elif response.status_code == 400:
        detail = response.json().get("detail", response.text)
        console.print(f"❌ Staking failed: {detail}", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Staking failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@ftns.command()
@click.option("--limit",   default=20, type=int, help="Transactions to show (max 100)")
@click.option("--search",  default=None,          help="Filter by description or transaction ID")
@click.option("--onchain", is_flag=True, default=False, help="Show on-chain TX (broadcast by daemon) instead of off-chain DAG ledger")
@click.option("--stats", is_flag=True, default=False, help="With --onchain: print aggregate stats instead of full TX list")
@click.option("--inbound", is_flag=True, default=False, help="With --onchain: show INBOUND Transfer events (FTNS sent TO this wallet) — sprint 512")
@click.option("--lookback-blocks", default=100000, type=int, help="With --onchain --inbound: how many blocks back to scan (default 100000 ~= 56hrs on Base)")
@click.option("--api-url", default=None,           help="PRSM API URL (default: from stored credentials)")
def history(limit: int, search: Optional[str], onchain: bool, stats: bool, inbound: bool, lookback_blocks: int, api_url: str) -> None:
    """Show your FTNS transaction history.

    Default: off-chain DAG ledger (user-to-user FTNS moves).
    With --onchain: real Base mainnet TX broadcast by this daemon's
    loaded FTNS_WALLET_PRIVATE_KEY (sprint-498 endpoint).
    With --onchain --stats: compact aggregate summary.
    With --onchain --inbound: FTNS received from external parties.
    """
    import httpx

    if onchain and inbound and stats:
        url = _api_url_from_creds(api_url)
        try:
            r = httpx.get(
                f"{url}/wallet/transactions/onchain/inbound/stats",
                params={"lookback_blocks": lookback_blocks},
                timeout=30.0,
            )
        except httpx.ConnectError:
            console.print(f"❌ Cannot connect to {url}", style="red")
            raise SystemExit(1)
        if r.status_code == 503:
            try:
                detail = r.json().get("detail", "")
            except Exception:
                detail = r.text[:300]
            console.print("❌ Inbound stats unavailable:", style="red")
            console.print(f"   {detail}")
            raise SystemExit(1)
        if r.status_code != 200:
            console.print(f"❌ HTTP {r.status_code}", style="red")
            raise SystemExit(1)
        d = r.json()
        console.print(f"\n[bold]Inbound FTNS stats[/bold] for {d.get('recipient')}")
        console.print(f"  count             : {d.get('count')}")
        console.print(f"  total received    : [green]{d.get('total_inbound_ftns', 0):.6f}[/green] FTNS")
        console.print(f"  first inbound blk : {d.get('first_inbound_block', '—')}")
        console.print(f"  last inbound blk  : {d.get('last_inbound_block', '—')}")
        console.print(f"  scan window       : blocks {d.get('from_block')}-{d.get('to_block')}\n")
        return

    if onchain and inbound:
        url = _api_url_from_creds(api_url)
        try:
            r = httpx.get(
                f"{url}/wallet/transactions/onchain/inbound",
                params={"lookback_blocks": lookback_blocks},
                timeout=30.0,
            )
        except httpx.ConnectError:
            console.print(f"❌ Cannot connect to {url}", style="red")
            raise SystemExit(1)
        if r.status_code == 503:
            try:
                detail = r.json().get("detail", "")
            except Exception:
                detail = r.text[:300]
            console.print("❌ Inbound scan unavailable:", style="red")
            console.print(f"   {detail}")
            raise SystemExit(1)
        if r.status_code != 200:
            console.print(f"❌ HTTP {r.status_code}: {r.text[:200]}", style="red")
            raise SystemExit(1)
        d = r.json()
        transfers = d.get("transfers", [])
        count = d.get("count", 0)
        console.print(
            f"\n🔗 Inbound FTNS for [bold]{d.get('recipient')}[/bold]  "
            f"[dim]({count} in blocks {d.get('from_block')}-{d.get('to_block')})[/dim]",
        )
        if not transfers:
            console.print("No inbound transfers in this window.", style="dim")
            return
        table = Table(title=f"Inbound FTNS  (count={count})")
        table.add_column("tx_hash",  style="dim",    max_width=18)
        table.add_column("block",    style="cyan",   justify="right")
        table.add_column("from",     style="white",  max_width=14)
        table.add_column("amount",   style="green",  justify="right")
        for t in transfers[: min(limit, 100)]:
            tx_hash = t.get("tx_hash") or ""
            table.add_row(
                (tx_hash[:16] + "…") if tx_hash else "—",
                str(t.get("block_number") or "—"),
                (t.get("from_address") or "—")[:14],
                f"{t.get('amount_ftns', 0):.6f}",
            )
        console.print(table)
        return

    if onchain and stats:
        url = _api_url_from_creds(api_url)
        try:
            r = httpx.get(
                f"{url}/wallet/transactions/onchain/stats",
                timeout=10.0,
            )
        except httpx.ConnectError:
            console.print(f"❌ Cannot connect to {url}", style="red")
            raise SystemExit(1)
        if r.status_code == 503:
            try:
                detail = r.json().get("detail", "")
            except Exception:
                detail = r.text[:300]
            console.print("❌ On-chain ledger not available:", style="red")
            console.print(f"   {detail}")
            raise SystemExit(1)
        if r.status_code != 200:
            console.print(f"❌ HTTP {r.status_code}", style="red")
            raise SystemExit(1)
        d = r.json()
        import datetime as _dt

        def _fmt_ts(ts):
            if ts is None:
                return "—"
            return _dt.datetime.fromtimestamp(ts).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        console.print(f"\n[bold]On-chain TX stats[/bold] for {d.get('address')}")
        console.print(f"  total       : {d.get('total_count')}")
        console.print(f"  confirmed   : [green]{d.get('confirmed_count')}[/green]")
        console.print(f"  pending     : [yellow]{d.get('pending_count')}[/yellow]")
        console.print(f"  rejected    : [red]{d.get('rejected_count')}[/red]")
        console.print(f"  total sent  : {d.get('total_ftns_sent', 0):.6f} FTNS  (confirmed only)")
        console.print(f"  first tx    : {_fmt_ts(d.get('first_tx_at'))}")
        console.print(f"  last tx     : {_fmt_ts(d.get('last_tx_at'))}")
        console.print(f"  [dim]scope:[/dim] {d.get('scope', '')}\n")
        return

    if onchain:
        url = _api_url_from_creds(api_url)
        try:
            response = httpx.get(
                f"{url}/wallet/transactions/onchain",
                timeout=10.0,
            )
        except httpx.ConnectError:
            console.print(f"❌ Cannot connect to {url}", style="red")
            raise SystemExit(1)

        if response.status_code == 503:
            detail = ""
            try:
                detail = response.json().get("detail", "")
            except Exception:
                detail = response.text[:300]
            console.print("❌ On-chain ledger not available:", style="red")
            console.print(f"   {detail}")
            raise SystemExit(1)
        if response.status_code != 200:
            console.print(f"❌ Failed: HTTP {response.status_code}", style="red")
            raise SystemExit(1)

        data = response.json()
        txs = data.get("transactions", [])
        count = data.get("count", 0)
        addr = data.get("connected_address", "?")
        scope = data.get("scope", "")

        console.print(
            f"🔗 On-chain TX for [bold]{addr}[/bold]  "
            f"[dim]({count} total — {scope})[/dim]",
        )
        if not txs:
            console.print(
                "No on-chain transactions in this session.",
                style="dim",
            )
            return

        table = Table(title=f"On-chain FTNS TX  (count={count})")
        table.add_column("tx_hash",   style="dim",     max_width=18)
        table.add_column("block",     style="cyan",    justify="right")
        table.add_column("from",      style="white",   max_width=14)
        table.add_column("to",        style="white",   max_width=14)
        table.add_column("amount",    style="green",   justify="right")
        table.add_column("status",    style="magenta")

        for tx in txs[: min(limit, 100)]:
            tx_hash = (tx.get("tx_hash") or "")
            table.add_row(
                (tx_hash[:16] + "…") if tx_hash else "—",
                str(tx.get("block_number") or "—"),
                (tx.get("from_address") or "—")[:14],
                (tx.get("to_address") or "—")[:14],
                f"{tx.get('amount_ftns', 0):.6f}",
                tx.get("status", "?"),
            )
        console.print(table)
        return

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)
    params: dict = {"limit": min(limit, 100)}
    if search:
        params["search"] = search

    try:
        # Correct endpoint: /transactions, not /history
        response = httpx.get(
            f"{url}/api/v1/ftns/transactions",
            params=params,
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        raise SystemExit(1)

    if response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    if response.status_code != 200:
        console.print(f"❌ Failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)

    data = response.json()
    transactions = data.get("transactions", [])

    if not transactions:
        console.print("No transactions found.", style="dim")
        return

    table = Table(title=f"FTNS History — {data.get('user_id', '?')}")
    table.add_column("ID",          style="dim",     max_width=14)
    table.add_column("Type",        style="cyan")
    table.add_column("Amount",      style="green",   justify="right")
    table.add_column("Description", style="white",   max_width=36)
    table.add_column("Status",      style="magenta")
    table.add_column("Time",        style="blue")

    for tx in transactions:
        amount = float(tx.get("amount", 0))
        table.add_row(
            str(tx.get("transaction_id", ""))[:12] + "…",
            tx.get("transaction_type", "?"),
            f"{amount:.6f}",
            (tx.get("description") or "")[:36],
            tx.get("status", "?"),
            str(tx.get("timestamp", ""))[:19],
        )
    console.print(table)


@ftns.command("yield-estimate")
@click.option("--hours", default=8, help="Hours per day available for compute")
@click.option("--stake", default=0, type=float, help="FTNS staked")
def ftns_yield_estimate(hours, stake):
    """Estimate daily/monthly FTNS earnings based on your hardware."""
    from prsm.compute.wasm.profiler import HardwareProfiler
    from prsm.economy.pricing.engine import PricingEngine
    from prsm.economy.pricing.models import ProsumerTier

    profiler = HardwareProfiler()
    profile = profiler.detect()
    tier = ProsumerTier.from_stake(stake)
    engine = PricingEngine()

    estimate = engine.yield_estimate(
        hardware_tier=profile.compute_tier.value,
        tflops=profile.tflops_fp32,
        hours_per_day=hours,
        prosumer_tier=tier,
    )

    click.echo(f"Yield Estimate:")
    click.echo(f"  Hardware: {profile.compute_tier.value.upper()} ({profile.tflops_fp32:.1f} TFLOPS)")
    click.echo(f"  Stake: {stake:.0f} FTNS ({tier.label})")
    click.echo(f"  Yield Boost: {estimate['yield_boost']}x")
    click.echo(f"  Daily: {float(estimate['daily_ftns']):.2f} FTNS")
    click.echo(f"  Monthly: {float(estimate['monthly_ftns']):.2f} FTNS")


# ============================================================================
# SETTLEMENT COMMANDS (under ftns group)
# ============================================================================

def _get_api_url() -> str:
    """Return the PRSM API URL — env var override > loopback default.

    Settlement subcommands read this when the caller omits --api-url.
    Uses PRSM_API_URL env var if set, else 127.0.0.1:8000 which matches
    the default in other CLI subcommands (e.g. `start --api-port 8000`).
    """
    return os.environ.get("PRSM_API_URL", "http://127.0.0.1:8000")


@ftns.group()
def settle():
    """On-chain batch settlement commands."""
    pass


@settle.command("status")
@click.option("--api-url", default=None, help="PRSM API URL")
def settle_status(api_url: str) -> None:
    """Show batch settlement queue status."""
    import httpx
    from rich.console import Console
    from rich.table import Table
    console = Console()
    url = api_url or _get_api_url()
    try:
        resp = httpx.get(f"{url}/settlement/stats", timeout=10)
        data = resp.json()
    except Exception as e:
        console.print(f"  [red]Error:[/red] {e}")
        return

    table = Table(title="Batch Settlement Status", show_header=True)
    table.add_column("Property", style="bold")
    table.add_column("Value")
    table.add_row("Mode", str(data.get("mode", "unknown")))
    table.add_row("Queue Size", str(data.get("queue_size", 0)))
    table.add_row("Pending FTNS", f"{data.get('pending_amount', 0):.6f}")
    table.add_row("Flush Interval", f"{data.get('flush_interval', 0)}s")
    table.add_row("Flush Threshold", f"{data.get('flush_threshold', 0)} FTNS")
    table.add_row("Total Settled", str(data.get("total_settled", 0)))
    table.add_row("Gas Txs Saved", str(data.get("gas_txs_saved", 0)))
    ago = data.get("last_flush_ago")
    table.add_row("Last Flush", f"{ago:.0f}s ago" if ago else "never")
    console.print(table)


@settle.command("pending")
@click.option("--api-url", default=None, help="PRSM API URL")
def settle_pending(api_url: str) -> None:
    """List pending un-settled transfers."""
    import httpx
    from rich.console import Console
    from rich.table import Table
    console = Console()
    url = api_url or _get_api_url()
    try:
        resp = httpx.get(f"{url}/settlement/pending", timeout=10)
        data = resp.json()
    except Exception as e:
        console.print(f"  [red]Error:[/red] {e}")
        return

    pending = data.get("pending", [])
    if not pending:
        console.print("  No pending transfers.")
        return

    table = Table(title=f"Pending Transfers ({len(pending)})", show_header=True)
    table.add_column("TX ID")
    table.add_column("To")
    table.add_column("Amount")
    table.add_column("Age")
    for p in pending:
        table.add_row(
            str(p.get("tx_id", "")),
            str(p.get("to", "")),
            f"{p.get('amount', 0):.6f}",
            f"{p.get('age_seconds', 0):.0f}s",
        )
    console.print(table)


@settle.command("flush")
@click.option("--api-url", default=None, help="PRSM API URL")
def settle_flush(api_url: str) -> None:
    """Manually trigger batch settlement (flush all pending transfers on-chain)."""
    import httpx
    from rich.console import Console
    console = Console()
    url = api_url or _get_api_url()
    console.print("  Flushing settlement queue...")
    try:
        resp = httpx.post(f"{url}/settlement/flush", timeout=120)
        data = resp.json()
    except Exception as e:
        console.print(f"  [red]Error:[/red] {e}")
        return

    settled = data.get("settled_count", 0)
    net = data.get("net_transfers", 0)
    amount = data.get("total_amount", 0)
    duration = data.get("duration_seconds", 0)
    hashes = data.get("tx_hashes", [])
    errors = data.get("errors", [])

    if settled == 0:
        console.print("  Nothing to settle.")
        return

    console.print(f"  Settled {settled} transfers → {net} on-chain txs")
    console.print(f"  Total: {amount:.6f} FTNS in {duration:.1f}s")
    if hashes:
        for h in hashes:
            console.print(f"  TX: {h}")
    if errors:
        for e in errors:
            console.print(f"  [red]Error:[/red] {e}")


@settle.command("history")
@click.option("--api-url", default=None, help="PRSM API URL")
@click.option("--limit", default=10, help="Number of records")
def settle_history(api_url: str, limit: int) -> None:
    """Show recent settlement history."""
    import httpx
    from rich.console import Console
    from rich.table import Table
    console = Console()
    url = api_url or _get_api_url()
    try:
        resp = httpx.get(f"{url}/settlement/history", params={"limit": limit}, timeout=10)
        data = resp.json()
    except Exception as e:
        console.print(f"  [red]Error:[/red] {e}")
        return

    history = data.get("history", [])
    if not history:
        console.print("  No settlement history.")
        return

    table = Table(title="Settlement History", show_header=True)
    table.add_column("Settled")
    table.add_column("Net TXs")
    table.add_column("Amount")
    table.add_column("Duration")
    table.add_column("Errors")
    for h in history:
        table.add_row(
            str(h.get("settled_count", 0)),
            str(h.get("net_transfers", 0)),
            f"{h.get('total_amount', 0):.6f}",
            f"{h.get('duration_seconds', 0):.1f}s",
            str(len(h.get("errors", []))),
        )
    console.print(table)


# ============================================================================
# BRIDGE COMMANDS (under ftns group)
# ============================================================================

@ftns.group()
def bridge():
    """FTNS token bridge commands for cross-chain transfers."""
    pass


@bridge.command()
@click.option('--amount', required=True, type=float, help='Amount of FTNS to deposit')
@click.option('--address', required=True, help='Destination on-chain address')
@click.option('--chain', default=137, type=int, help='Destination chain ID (default: 137 for Polygon)')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def deposit(amount: float, address: str, chain: int, api_url: str):
    """
    Deposit FTNS tokens from local balance to external chain.
    
    Burns local FTNS and initiates bridge transfer to mint tokens on the destination chain.
    The transaction will go through validation, processing, and confirmation stages.
    """
    import httpx
    
    console.print(f"🌉 Depositing {amount} FTNS to chain {chain}...", style="bold blue")
    console.print(f"   Destination address: {address}")
    
    try:
        response = httpx.post(
            f"{api_url}/bridge/deposit",
            json={
                "amount": amount,
                "chain_address": address,
                "destination_chain": chain
            },
            timeout=30.0
        )
        
        if response.status_code == 200:
            data = response.json()
            tx = data.get('transaction', {})
            console.print(f"✅ Bridge deposit initiated!", style="bold green")
            console.print(f"   Transaction ID: {tx.get('transaction_id')}")
            console.print(f"   Status: {tx.get('status')}")
            console.print(f"   Amount: {int(tx.get('amount', 0)) / 10**18:.4f} FTNS")
            console.print(f"   Fee: {int(tx.get('fee_amount', 0)) / 10**18:.4f} FTNS")
            console.print(f"   Created: {tx.get('created_at', 'N/A')[:19]}")
            
            if tx.get('status') == 'completed':
                console.print(f"   🎉 Transaction completed!", style="green")
            elif tx.get('status') == 'pending':
                console.print(f"   ⏳ Transaction pending - check status with: prsm ftns bridge status", style="yellow")
        else:
            error_detail = response.json().get('detail', response.text)
            console.print(f"❌ Bridge deposit failed: {response.status_code}", style="red")
            console.print(f"   {error_detail}")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@bridge.command()
@click.option('--amount', required=True, type=float, help='Amount of FTNS to withdraw')
@click.option('--address', required=True, help='Source on-chain address')
@click.option('--chain', default=137, type=int, help='Source chain ID (default: 137 for Polygon)')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def withdraw(amount: float, address: str, chain: int, api_url: str):
    """
    Withdraw FTNS tokens from external chain to local balance.
    
    Locks on-chain FTNS and initiates bridge transfer to mint local FTNS to your account.
    The transaction will go through validation, processing, and confirmation stages.
    """
    import httpx
    
    console.print(f"🌉 Withdrawing {amount} FTNS from chain {chain}...", style="bold blue")
    console.print(f"   Source address: {address}")
    
    try:
        response = httpx.post(
            f"{api_url}/bridge/withdraw",
            json={
                "amount": amount,
                "chain_address": address,
                "source_chain": chain
            },
            timeout=30.0
        )
        
        if response.status_code == 200:
            data = response.json()
            tx = data.get('transaction', {})
            console.print(f"✅ Bridge withdraw initiated!", style="bold green")
            console.print(f"   Transaction ID: {tx.get('transaction_id')}")
            console.print(f"   Status: {tx.get('status')}")
            console.print(f"   Amount: {int(tx.get('amount', 0)) / 10**18:.4f} FTNS")
            console.print(f"   Fee: {int(tx.get('fee_amount', 0)) / 10**18:.4f} FTNS")
            console.print(f"   Created: {tx.get('created_at', 'N/A')[:19]}")
            
            if tx.get('status') == 'completed':
                console.print(f"   🎉 Transaction completed!", style="green")
            elif tx.get('status') == 'pending':
                console.print(f"   ⏳ Transaction pending - check status with: prsm ftns bridge status", style="yellow")
        else:
            error_detail = response.json().get('detail', response.text)
            console.print(f"❌ Bridge withdraw failed: {response.status_code}", style="red")
            console.print(f"   {error_detail}")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@bridge.command()
@click.option('--tx-id', help='Specific transaction ID to look up')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def status(tx_id: str, api_url: str):
    """
    Get bridge status and pending operations.
    
    Without --tx-id: Shows overall bridge statistics and pending transactions.
    With --tx-id: Shows status of a specific bridge transaction.
    """
    import httpx
    
    try:
        if tx_id:
            # Get specific transaction status
            response = httpx.get(f"{api_url}/bridge/transactions/{tx_id}", timeout=10.0)
            
            if response.status_code == 200:
                data = response.json()
                tx = data.get('transaction', {})
                
                table = Table(title=f"Bridge Transaction: {tx_id[:20]}...")
                table.add_column("Field", style="cyan")
                table.add_column("Value", style="green")
                
                table.add_row("Transaction ID", tx.get('transaction_id', 'N/A'))
                table.add_row("Direction", tx.get('direction', 'N/A').upper())
                table.add_row("Status", tx.get('status', 'N/A'))
                table.add_row("Amount", f"{int(tx.get('amount', 0)) / 10**18:.4f} FTNS")
                table.add_row("Fee", f"{int(tx.get('fee_amount', 0)) / 10**18:.4f} FTNS")
                table.add_row("Chain Address", tx.get('chain_address', 'N/A'))
                table.add_row("Source Chain", str(tx.get('source_chain', 'N/A')))
                table.add_row("Dest Chain", str(tx.get('destination_chain', 'N/A')))
                table.add_row("Created", tx.get('created_at', 'N/A')[:19] if tx.get('created_at') else 'N/A')
                table.add_row("Updated", tx.get('updated_at', 'N/A')[:19] if tx.get('updated_at') else 'N/A')
                
                if tx.get('completed_at'):
                    table.add_row("Completed", tx.get('completed_at')[:19])
                if tx.get('source_tx_hash'):
                    table.add_row("Source TX Hash", tx.get('source_tx_hash')[:20] + "...")
                if tx.get('destination_tx_hash'):
                    table.add_row("Dest TX Hash", tx.get('destination_tx_hash')[:20] + "...")
                if tx.get('error_message'):
                    table.add_row("Error", tx.get('error_message'), style="red")
                
                console.print(table)
            elif response.status_code == 404:
                console.print(f"❌ Transaction not found: {tx_id}", style="red")
            else:
                console.print(f"❌ Failed to get transaction: {response.status_code}", style="red")
        else:
            # Get overall bridge status
            response = httpx.get(f"{api_url}/bridge/status", timeout=10.0)
            
            if response.status_code == 200:
                data = response.json()
                stats = data.get('stats', {})
                limits = data.get('limits', {})
                pending = data.get('pending_transactions', [])
                
                # Stats table
                stats_table = Table(title="Bridge Statistics")
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Value", style="green")
                
                stats_table.add_row("Total Deposited", f"{int(stats.get('total_deposited', 0)) / 10**18:.4f} FTNS")
                stats_table.add_row("Total Withdrawn", f"{int(stats.get('total_withdrawn', 0)) / 10**18:.4f} FTNS")
                stats_table.add_row("Total Fees Collected", f"{int(stats.get('total_fees_collected', 0)) / 10**18:.4f} FTNS")
                stats_table.add_row("Pending Transactions", str(stats.get('pending_transactions', 0)))
                stats_table.add_row("Completed Transactions", str(stats.get('completed_transactions', 0)))
                stats_table.add_row("Failed Transactions", str(stats.get('failed_transactions', 0)))
                
                console.print(stats_table)
                
                # Limits table
                if limits:
                    limits_table = Table(title="Bridge Limits")
                    limits_table.add_column("Limit", style="cyan")
                    limits_table.add_column("Value", style="magenta")
                    
                    limits_table.add_row("Minimum Amount", f"{int(limits.get('min_amount', 0)) / 10**18:.4f} FTNS")
                    limits_table.add_row("Maximum Amount", f"{int(limits.get('max_amount', 0)) / 10**18:.4f} FTNS")
                    limits_table.add_row("Daily Limit", f"{int(limits.get('daily_limit', 0)) / 10**18:.4f} FTNS")
                    limits_table.add_row("Fee (BPS)", str(limits.get('fee_bps', 0)))
                    
                    console.print(limits_table)
                
                # Pending transactions
                if pending:
                    pending_table = Table(title=f"Pending Transactions ({len(pending)})")
                    pending_table.add_column("TX ID", style="dim")
                    pending_table.add_column("Direction", style="cyan")
                    pending_table.add_column("Amount", style="green")
                    pending_table.add_column("Status", style="magenta")
                    pending_table.add_column("Created", style="blue")
                    
                    for tx in pending[:10]:  # Show max 10
                        pending_table.add_row(
                            tx.get('transaction_id', 'N/A')[:20] + "...",
                            tx.get('direction', 'N/A').upper(),
                            f"{int(tx.get('amount', 0)) / 10**18:.4f}",
                            tx.get('status', 'N/A'),
                            tx.get('created_at', 'N/A')[:19]
                        )
                    
                    console.print(pending_table)
                    if len(pending) > 10:
                        console.print(f"   ... and {len(pending) - 10} more pending transactions", style="dim")
            else:
                console.print(f"❌ Failed to get bridge status: {response.status_code}", style="red")
                
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@bridge.command()
@click.option('--limit', default=20, type=int, help='Maximum transactions to show')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def history(limit: int, api_url: str):
    """Show bridge transaction history for the current user."""
    import httpx
    
    try:
        response = httpx.get(f"{api_url}/bridge/transactions?limit={limit}", timeout=10.0)
        
        if response.status_code == 200:
            data = response.json()
            transactions = data.get('transactions', [])
            
            if not transactions:
                console.print("No bridge transactions found.", style="dim")
                return
            
            table = Table(title="Bridge Transaction History")
            table.add_column("TX ID", style="dim")
            table.add_column("Direction", style="cyan")
            table.add_column("Amount", style="green")
            table.add_column("Fee", style="yellow")
            table.add_column("Status", style="magenta")
            table.add_column("Created", style="blue")
            
            for tx in transactions:
                table.add_row(
                    tx.get('transaction_id', 'N/A')[:16] + "...",
                    tx.get('direction', 'N/A').upper(),
                    f"{int(tx.get('amount', 0)) / 10**18:.4f}",
                    f"{int(tx.get('fee_amount', 0)) / 10**18:.4f}",
                    tx.get('status', 'N/A'),
                    tx.get('created_at', 'N/A')[:19]
                )
            
            console.print(table)
        else:
            console.print(f"❌ Failed to get bridge history: {response.status_code}", style="red")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")




# ============================================================================
# STORAGE COMMANDS
# ============================================================================

@main.group()
def storage():
    """Content storage management commands"""
    pass


@storage.command()
@click.argument("file-path", type=click.Path(exists=True, readable=True))
@click.option("--description",   default="",    help="Content description")
@click.option(
    "--royalty-rate",
    default=0.01, type=float,
    help="FTNS earned per access (0.001–0.1, default: 0.01)"
)
@click.option(
    "--parent-cids",
    default="",
    help="Comma-separated CIDs of content this file derives from"
)
@click.option("--replicas",   default=3, type=int, help="Replication factor (1–10)")
@click.option("--api-url",    default=None,         help="PRSM API URL (default: from stored credentials)")
@click.option("--semantic-shard", is_flag=True, default=False, help="Semantically shard the dataset by content similarity before uploading")
def upload(
    file_path: str,
    description: str,
    royalty_rate: float,
    parent_cids: str,
    replicas: int,
    api_url: str,
    semantic_shard: bool,
) -> None:
    """
    Upload a file to ContentStore and register provenance for royalty collection.

    The file is stored in the native ContentStore and a provenance record is
    created in the platform database. When other users access this content,
    they pay the configured royalty rate to your FTNS balance.

    \b
    Examples:
        prsm storage upload model_weights.pt --royalty-rate 0.05
        prsm storage upload dataset.csv --description "Training data Q1 2026"
        prsm storage upload paper.pdf --parent-cids QmAbc...,QmDef...
    """
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)
    file_path_obj = Path(file_path)
    file_size = file_path_obj.stat().st_size

    if semantic_shard:
        console.print(f"[bold]Semantic sharding enabled[/bold]")
        from prsm.data.shard_models import SemanticShard, SemanticShardManifest

        # Read file and create simple shards based on line count
        # (Real implementation would use embeddings for clustering)
        with open(file_path, 'rb') as f:
            content = f.read()

        file_size = len(content)
        # Split into chunks of ~1MB
        chunk_size = 1024 * 1024  # 1MB
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunks.append(content[i:i + chunk_size])

        if not chunks:
            chunks = [content]

        shards = []
        for i, chunk in enumerate(chunks):
            shard = SemanticShard(
                shard_id=f"shard-{i:04d}",
                parent_dataset=file_path_obj.name,
                cid=f"pending-upload-{i}",
                centroid=[float(i) / max(len(chunks), 1)],  # Placeholder centroid
                record_count=len(chunk),
                size_bytes=len(chunk),
                keywords=[file_path_obj.stem, f"shard-{i}"],
            )
            shards.append(shard)

        manifest = SemanticShardManifest(
            dataset_id=file_path_obj.stem,
            total_records=file_size,
            total_size_bytes=file_size,
            shards=shards,
        )

        console.print(f"  Shards: {len(shards)}")
        console.print(f"  Total size: {file_size:,} bytes")
        console.print(f"  Manifest: {manifest.dataset_id}")
        console.print()

    console.print(
        f"📤 Uploading {file_path_obj.name} ({file_size:,} bytes)...",
        style="bold blue"
    )
    if parent_cids:
        console.print(f"   Derivative of: {parent_cids}", style="dim")

    try:
        with open(file_path, "rb") as fh:
            files = {"file": (file_path_obj.name, fh, "application/octet-stream")}
            data = {
                "description": description,
                "royalty_rate": str(royalty_rate),
                "parent_cids": parent_cids,
                "replicas": str(replicas),
            }
            with console.status("[bold green]Uploading to ContentStore..."):
                response = httpx.post(
                    f"{url}/api/v1/content/upload",
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=120.0,
                )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        result = response.json()

        table = Table(title="Upload Result")
        table.add_column("Property",  style="cyan")
        table.add_column("Value",     style="green")
        table.add_row("CID",          result.get("cid", "?"))
        table.add_row("Filename",     result.get("filename", "?"))
        table.add_row("Size",         f"{result.get('size_bytes', 0):,} bytes")
        table.add_row("Royalty Rate", f"{result.get('royalty_rate', 0):.4f} FTNS/access")
        table.add_row(
            "Provenance",
            "✅ Registered" if result.get("provenance_registered") else "⚠️  Not persisted"
        )
        table.add_row("Access URL",   result.get("access_url", "?"))
        console.print(table)

        if result.get("parent_cids"):
            console.print(
                f"\n[dim]Derivative of: {', '.join(result['parent_cids'])}[/dim]"
            )

        # Emit CID on its own line — easy to capture in scripts
        console.print(f"\n[bold]CID: {result.get('cid')}[/bold]")

    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    elif response.status_code == 503:
        console.print("❌ Storage service unavailable", style="red")
        console.print("💡 Ensure ContentStore is initialized (prsm serve will initialize it)",
                      style="yellow")
        raise SystemExit(1)
    else:
        console.print(f"❌ Upload failed: HTTP {response.status_code}", style="red")
        console.print(f"   {response.text[:200]}")
        raise SystemExit(1)


@storage.command()
@click.argument('cid')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def download(cid: str, output: str, api_url: str):
    """Download content from ContentStore by content ID."""
    import httpx
    
    console.print(f"📥 Downloading {cid}...", style="bold blue")
    
    try:
        response = httpx.get(f"{api_url}/api/v1/storage/{cid}/download", timeout=60.0)
        
        if response.status_code == 200:
            if output:
                with open(output, 'wb') as f:
                    f.write(response.content)
                console.print(f"✅ Downloaded to {output}", style="green")
            else:
                # Print to console if no output file
                try:
                    console.print(response.content.decode('utf-8'))
                except UnicodeDecodeError:
                    console.print(f"Binary content ({len(response.content)} bytes)", style="dim")
        else:
            console.print(f"❌ Download failed: {response.status_code}", style="red")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@storage.command()
@click.argument('cid')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def info(cid: str, api_url: str):
    """Get information about stored content."""
    import httpx
    
    try:
        response = httpx.get(f"{api_url}/api/v1/storage/{cid}", timeout=10.0)
        
        if response.status_code == 200:
            data = response.json()
            table = Table(title=f"Storage Info: {cid}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("CID", data.get('cid', 'N/A'))
            table.add_row("Content Type", data.get('content_type', 'N/A'))
            table.add_row("Size", f"{data.get('size', 0)} bytes")
            table.add_row("Status", data.get('status', 'N/A'))
            table.add_row("Pinned", "Yes" if data.get('is_pinned') else "No")
            table.add_row("Public", "Yes" if data.get('is_public') else "No")
            if data.get('filename'):
                table.add_row("Filename", data.get('filename'))
            console.print(table)
        elif response.status_code == 404:
            console.print(f"❌ Content not found: {cid}", style="red")
        else:
            console.print(f"❌ Failed to get info: {response.status_code}", style="red")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@storage.command()
@click.argument('cid')
@click.option('--replication', default=3, type=int, help='Replication factor')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def pin(cid: str, replication: int, api_url: str):
    """Pin content for persistent storage."""
    import httpx
    
    console.print(f"📌 Pinning {cid}...", style="bold blue")
    
    try:
        response = httpx.post(
            f"{api_url}/api/v1/storage/{cid}/pin",
            json={"replication": replication},
            timeout=10.0
        )
        
        if response.status_code == 200:
            data = response.json()
            console.print(f"✅ Content pinned!", style="bold green")
            console.print(f"   Replication: {data.get('replication')}")
            console.print(f"   Monthly cost: {data.get('monthly_cost', 0):.4f} FTNS")
        else:
            console.print(f"❌ Pinning failed: {response.status_code}", style="red")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@storage.command()
@click.option('--limit', default=20, type=int, help='Maximum results')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def pins(limit: int, api_url: str):
    """List pinned content."""
    import httpx
    
    try:
        response = httpx.get(f"{api_url}/api/v1/storage/pins?limit={limit}", timeout=10.0)
        
        if response.status_code == 200:
            data = response.json()
            pins = data.get('pins', [])
            
            if not pins:
                console.print("No pinned content.", style="dim")
                return
            
            table = Table(title="Pinned Content")
            table.add_column("CID", style="cyan")
            table.add_column("Size", style="green")
            table.add_column("Replication", style="magenta")
            table.add_column("Monthly Cost", style="blue")
            
            for pin in pins:
                table.add_row(
                    pin.get('cid', 'N/A')[:20] + "...",
                    f"{pin.get('size', 0)} bytes",
                    str(pin.get('replication', 0)),
                    f"{pin.get('monthly_cost', 0):.4f} FTNS"
                )
            console.print(table)
        else:
            console.print(f"❌ Failed to list pins: {response.status_code}", style="red")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


# ============================================================================
# GOVERNANCE COMMANDS
# ============================================================================

@main.group()
def governance():
    """Governance and voting commands"""
    pass


@governance.command()
@click.option('--limit', default=10, type=int, help='Maximum proposals to show')
@click.option('--status', type=click.Choice(['draft', 'active', 'voting', 'passed', 'rejected', 'executed']), help='Filter by status')
@click.option('--api-url', default=None, help='PRSM API URL')
def proposals(limit: int, status: str, api_url: str):
    """List governance proposals."""
    import httpx
    
    api_url = _api_url_from_creds(api_url)
    headers = _auth_headers()
    
    params = {"limit": limit}
    if status:
        params["status_filter"] = status
    
    try:
        response = httpx.get(
            f"{api_url}/api/v1/governance/proposals",
            params=params,
            headers=headers,
            timeout=10.0
        )
        
        if response.status_code == 200:
            outer = response.json()
            proposals = outer.get("data", {}).get("proposals", [])
            
            if not proposals:
                console.print("No proposals found.", style="dim")
                return
            
            table = Table(title="Governance Proposals")
            table.add_column("ID", style="dim")
            table.add_column("Title", style="cyan")
            table.add_column("Status", style="magenta")
            table.add_column("Type", style="blue")
            table.add_column("Votes", style="green")
            
            for prop in proposals:
                votes = f"✓{prop.get('votes_for', 0):.0f} ✗{prop.get('votes_against', 0):.0f}"
                table.add_row(
                    prop.get('proposal_id', 'N/A')[:12] + "...",
                    prop.get('title', 'N/A')[:30],
                    prop.get('status', 'N/A'),
                    prop.get('proposal_type', 'N/A'),
                    votes
                )
            console.print(table)
        elif response.status_code == 401:
            console.print("❌ Session expired. Run: prsm login", style="red")
        elif response.status_code == 404:
            # Sprint 533 F49 fix: surface that the governance route
            # isn't wired on this daemon. Operators wanting governance
            # today use /admin/upgrade/* (sprint-394-era surface).
            console.print(
                "[yellow]⚠️  Governance endpoint not wired on this daemon.[/yellow]\n"
                "Current governance surface lives at `/admin/upgrade/*`. Use:\n"
                "  • [cyan]curl /admin/upgrade/{proposal_id}[/cyan]  — get proposal\n"
                "  • [cyan]curl /admin/upgrade/{id}/compose-upgrade[/cyan]\n"
                "  • [cyan]curl /admin/upgrade/{id}/compose-rollback[/cyan]\n"
                "The `prsm governance ...` CLI commands wrap a future "
                "proposal-list endpoint that isn't deployed in this build."
            )
        else:
            console.print(f"❌ Failed to list proposals: {response.status_code}", style="red")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@governance.command()
@click.argument('proposal-id')
@click.option('--api-url', default=None, help='PRSM API URL')
def proposal(proposal_id: str, api_url: str):
    """Get details of a specific proposal."""
    import httpx
    
    api_url = _api_url_from_creds(api_url)
    headers = _auth_headers()
    
    try:
        response = httpx.get(
            f"{api_url}/api/v1/governance/proposals/{proposal_id}",
            headers=headers,
            timeout=10.0
        )
        
        if response.status_code == 200:
            outer = response.json()
            data = outer.get("data", {}).get("proposal", {})
            results = outer.get("data", {}).get("results")
            
            console.print(f"\n📋 {data.get('title')}", style="bold cyan")
            console.print(f"   ID: {data.get('proposal_id')}")
            console.print(f"   Status: {data.get('status')}")
            console.print(f"   Type: {data.get('proposal_type')}")
            console.print(f"   Proposer: {data.get('proposer_id', 'N/A')[:16]}...")
            console.print()
            console.print("Description:", style="bold")
            console.print(data.get('description', 'N/A'))
            console.print()
            
            table = Table(title="Voting Results")
            table.add_column("Choice", style="cyan")
            table.add_column("Votes (FTNS)", style="green")
            table.add_row("For", f"{data.get('votes_for', 0):.2f}")
            table.add_row("Against", f"{data.get('votes_against', 0):.2f}")
            console.print(table)
            
            if results:
                console.print(f"\n   Results:", style="bold")
                console.print(f"   - Passed: {results.get('passed', False)}")
                console.print(f"   - Quorum met: {results.get('quorum_met', False)}")
            
            console.print(f"\n   Quorum required: {data.get('quorum', 0) * 100:.1f}%")
            console.print(f"   Threshold: {data.get('threshold', 0) * 100:.1f}%")
            console.print(f"   Voting ends: {data.get('voting_ends', 'N/A')}")
        elif response.status_code == 401:
            console.print("❌ Session expired. Run: prsm login", style="red")
        elif response.status_code == 404:
            console.print(f"❌ Proposal not found: {proposal_id}", style="red")
        else:
            console.print(f"❌ Failed to get proposal: {response.status_code}", style="red")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@governance.command()
@click.argument('proposal-id')
@click.option('--choice', required=True, type=click.Choice(['for', 'against']), help='Vote choice')
@click.option('--rationale', help='Rationale for your vote')
@click.option('--api-url', default=None, help='PRSM API URL')
def vote(proposal_id: str, choice: str, rationale: str, api_url: str):
    """Cast a vote on a proposal."""
    import httpx
    
    api_url = _api_url_from_creds(api_url)
    headers = _auth_headers()
    
    # Convert choice to boolean vote_choice
    vote_choice = (choice == "for")
    
    console.print(f"🗳️  Casting vote '{choice}' on proposal {proposal_id}...", style="bold blue")
    
    try:
        response = httpx.post(
            f"{api_url}/api/v1/governance/vote",
            json={
                "proposal_id": proposal_id,
                "vote_choice": vote_choice,
                "rationale": rationale
            },
            headers=headers,
            timeout=10.0
        )
        
        if response.status_code == 200:
            outer = response.json()
            data = outer.get("data", {}).get("vote", {})
            console.print(f"✅ Vote cast successfully!", style="bold green")
            console.print(f"   Proposal ID: {data.get('proposal_id')}")
            console.print(f"   Vote choice: {'For' if data.get('vote_choice') else 'Against'}")
            console.print(f"   Cast at: {data.get('cast_at', 'N/A')}")
        elif response.status_code == 401:
            console.print("❌ Session expired. Run: prsm login", style="red")
        else:
            console.print(f"❌ Failed to cast vote: {response.status_code}", style="red")
            console.print(f"   {response.text}")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@governance.command()
@click.option('--title', required=True, help='Proposal title (min 10 characters)')
@click.option('--description', required=True, help='Proposal description (min 100 characters)')
@click.option('--type', 'proposal_type', required=True,
              type=click.Choice(['safety', 'economic', 'technical', 'governance',
                                'parameter_change', 'constitutional', 'emergency',
                                'operational', 'community']),
              help='Type of proposal')
@click.option('--api-url', default=None, help='PRSM API URL')
def create_proposal(title: str, description: str, proposal_type: str, api_url: str):
    """Create a new governance proposal."""
    import httpx
    
    # Client-side validation
    if len(title) < 10:
        console.print("❌ Title must be at least 10 characters", style="red")
        return
    
    if len(description) < 100:
        console.print("❌ Description must be at least 100 characters", style="red")
        return
    
    api_url = _api_url_from_creds(api_url)
    headers = _auth_headers()
    
    console.print(f"📝 Creating proposal: {title}...", style="bold blue")
    
    try:
        response = httpx.post(
            f"{api_url}/api/v1/governance/proposals",
            json={
                "title": title,
                "description": description,
                "proposal_type": proposal_type
            },
            headers=headers,
            timeout=10.0
        )
        
        if response.status_code == 200:
            outer = response.json()
            data = outer.get("data", {}).get("proposal", {})
            console.print(f"✅ Proposal created!", style="bold green")
            console.print(f"   Proposal ID: {data.get('proposal_id')}")
            console.print(f"   Title: {data.get('title')}")
            console.print(f"   Status: {data.get('status')}")
            console.print(f"   Voting ends: {data.get('voting_ends')}")
            console.print(f"   Created at: {data.get('created_at')}")
        elif response.status_code == 401:
            console.print("❌ Session expired. Run: prsm login", style="red")
        elif response.status_code == 422:
            console.print("❌ Validation error:", style="red")
            error_data = response.json()
            console.print(f"   {error_data.get('detail', 'Unknown validation error')}")
        else:
            console.print(f"❌ Failed to create proposal: {response.status_code}", style="red")
            console.print(f"   {response.text}")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@governance.command()
@click.option('--api-url', default=None, help='PRSM API URL')
def voting_power(api_url: str):
    """Show your voting power and governance status."""
    import httpx
    
    api_url = _api_url_from_creds(api_url)
    headers = _auth_headers()
    
    try:
        response = httpx.get(
            f"{api_url}/api/v1/governance/status",
            headers=headers,
            timeout=10.0
        )
        
        if response.status_code == 200:
            outer = response.json()
            data = outer.get("data", {}).get("governance_status", {})
            
            if not data:
                console.print("❌ Governance not activated. Run: prsm governance activate", style="yellow")
                return
            
            console.print(f"🗳️  Your governance status:", style="bold green")
            console.print(f"   Voting power: {data.get('voting_power', 0):.2f} FTNS")
            console.print(f"   Participant tier: {data.get('participant_tier', 'N/A')}")
            console.print(f"   Active: {data.get('is_active', False)}")
        elif response.status_code == 401:
            console.print("❌ Session expired. Run: prsm login", style="red")
        else:
            console.print(f"❌ Failed to get governance status: {response.status_code}", style="red")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@governance.command()
@click.option('--tier', required=True,
              type=click.Choice(['community', 'contributor', 'expert', 'delegate', 'council_member', 'core_team']),
              help='Participant tier for governance')
@click.option('--api-url', default=None, help='PRSM API URL')
def activate(tier: str, api_url: str):
    """Activate governance participation."""
    import httpx
    
    api_url = _api_url_from_creds(api_url)
    headers = _auth_headers()
    
    console.print(f"🗳️  Activating governance participation as {tier}...", style="bold blue")
    
    try:
        response = httpx.post(
            f"{api_url}/api/v1/governance/activate",
            json={
                "participant_tier": tier,
                "auto_stake_percentage": 0.5
            },
            headers=headers,
            timeout=10.0
        )
        
        if response.status_code == 200:
            outer = response.json()
            data = outer.get("data", {}).get("activation", {})
            console.print(f"✅ Governance activated!", style="bold green")
            console.print(f"   Participant tier: {data.get('participant_tier')}")
            console.print(f"   Voting power: {data.get('voting_power', 0):.2f} FTNS")
            console.print(f"   Auto-stake: {data.get('auto_stake_percentage', 0) * 100:.0f}%")
        elif response.status_code == 401:
            console.print("❌ Session expired. Run: prsm login", style="red")
        else:
            console.print(f"❌ Failed to activate governance: {response.status_code}", style="red")
            console.print(f"   {response.text}")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")



# ============================================================================
# TORRENT COMMANDS (Phase 1)
# ============================================================================

def _fmt_bytes(n: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _fmt_rate(n: float) -> str:
    """Format bytes per second as human-readable rate."""
    return f"{_fmt_bytes(int(n))}/s"


@main.group()
def torrent():
    """BitTorrent P2P distribution commands"""
    pass


@torrent.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--name", help="Human-readable name (default: filename)")
@click.option("--piece-length", type=int, default=262144, help="Piece size in bytes")
@click.option("--provenance-id", help="Link to existing PRSM provenance record")
@click.option("--no-seed", is_flag=True, help="Create torrent file only, do not start seeding")
@click.option("--output-torrent", type=click.Path(), help="Save .torrent file to disk")
@click.option("--api-url", default=None, help="PRSM API URL")
def create(path: str, name: Optional[str], piece_length: int, provenance_id: Optional[str],
           no_seed: bool, output_torrent: Optional[str], api_url: str):
    """Create a new torrent from a local file or directory and begin seeding."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)
    path_obj = Path(path)

    console.print(f"🔗 Creating torrent from {path_obj.name}...", style="bold blue")

    payload = {
        "content_path": str(path_obj.absolute()),
        "piece_length": piece_length,
    }
    if name:
        payload["name"] = name
    if provenance_id:
        payload["provenance_id"] = provenance_id

    try:
        response = httpx.post(
            f"{url}/api/v1/torrents/create",
            json=payload,
            headers=headers,
            timeout=60.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        infohash = data.get("infohash", "")

        from rich.panel import Panel
        content = f"""Infohash: {infohash}
Name:     {data.get('name', 'N/A')}
Size:     {_fmt_bytes(data.get('size_bytes', 0))}
Pieces:   {data.get('num_pieces', 0)} × {_fmt_bytes(data.get('piece_length', piece_length))}
Magnet:   magnet:?xt=urn:btih:{infohash}&dn={data.get('name', '')}
Status:   {'Created only' if no_seed else 'Seeding ✅'}"""

        console.print(Panel(content, title="🔗 Torrent Created", border_style="green"))

        if output_torrent:
            console.print(f"   Torrent file: {output_torrent}", style="dim")
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    elif response.status_code == 503:
        console.print("⚠️  BitTorrent service unavailable", style="yellow")
        raise SystemExit(1)
    else:
        console.print(f"❌ Create failed: HTTP {response.status_code}", style="red")
        console.print(f"   {response.text[:200]}")
        raise SystemExit(1)


@torrent.command()
@click.argument("source")
@click.option("--save-path", type=click.Path(), help="Where to save downloaded files")
@click.option("--seed-mode", is_flag=True, help="Skip downloading — assume data already present")
@click.option("--download", is_flag=True, help="Begin downloading immediately")
@click.option("--api-url", default=None, help="PRSM API URL")
def add(source: str, save_path: Optional[str], seed_mode: bool, download: bool, api_url: str):
    """Add an existing torrent from magnet URI or .torrent file path."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    console.print("🔗 Adding torrent...", style="bold blue")

    payload = {
        "source": source,
        "seed_mode": seed_mode,
    }
    if save_path:
        payload["save_path"] = save_path

    try:
        response = httpx.post(
            f"{url}/api/v1/torrents/add",
            json=payload,
            headers=headers,
            timeout=30.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        infohash = data.get("infohash", "")[:16]

        console.print(f"✅ Torrent added: {infohash}...", style="bold green")
        console.print(f"   Name:   {data.get('name', 'N/A')}")
        console.print(f"   Status: {data.get('state', 'Added')}")
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    elif response.status_code == 503:
        console.print("⚠️  BitTorrent service unavailable", style="yellow")
        raise SystemExit(1)
    else:
        console.print(f"❌ Add failed: HTTP {response.status_code}", style="red")
        console.print(f"   {response.text[:200]}")
        raise SystemExit(1)


@torrent.command("list")
@click.option("--seeding", is_flag=True, help="Show only torrents this node is seeding")
@click.option("--available", is_flag=True, help="Show only network-announced torrents")
@click.option("--limit", type=int, default=50, help="Max results")
@click.option("--api-url", default=None, help="PRSM API URL")
def list_torrents(seeding: bool, available: bool, limit: int, api_url: str):
    """List all torrents known to this node."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    try:
        response = httpx.get(
            f"{url}/api/v1/torrents",
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        torrents = data.get("torrents", [])

        if not torrents:
            console.print("No torrents found.", style="yellow")
            return

        table = Table(title="Torrents")
        table.add_column("Infohash", style="cyan", no_wrap=True)
        table.add_column("Name", style="green")
        table.add_column("Size", style="blue")
        table.add_column("State", style="magenta")
        table.add_column("Progress", style="yellow")
        table.add_column("Peers", style="dim")
        table.add_column("↑ Rate", style="red")
        table.add_column("↓ Rate", style="blue")

        for t in torrents[:limit]:
            infohash = t.get("infohash", "")
            short_hash = infohash[:16] + "..." if len(infohash) > 16 else infohash
            progress = t.get("progress", 0) * 100

            table.add_row(
                short_hash,
                t.get("name", "N/A")[:30],
                _fmt_bytes(t.get("size_bytes", 0)),
                t.get("state", "?"),
                f"{progress:.1f}%",
                str(t.get("seeders", 0) + t.get("leechers", 0)),
                _fmt_rate(t.get("upload_rate", 0)),
                _fmt_rate(t.get("download_rate", 0)),
            )

        console.print(table)
        console.print(f"\n📊 Total: {data.get('total', len(torrents))} torrents", style="blue")
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ List failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@torrent.command()
@click.argument("infohash")
@click.option("--api-url", default=None, help="PRSM API URL")
def status(infohash: str, api_url: str):
    """Show detailed live status for a specific torrent."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    try:
        response = httpx.get(
            f"{url}/api/v1/torrents/{infohash}",
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        progress = data.get("progress", 0) * 100

        from rich.panel import Panel

        # Build progress bar
        progress_bar = "█" * int(progress / 5) + "░" * (20 - int(progress / 5))

        eta_str = ""
        eta = data.get("eta_seconds", 0)
        if eta > 0:
            mins, secs = divmod(int(eta), 60)
            eta_str = f"  (ETA: {mins}m {secs}s)" if mins else f"  (ETA: {secs}s)"

        content = f"""Infohash:     {data.get('infohash', 'N/A')}
Name:         {data.get('name', 'N/A')}
Size:         {_fmt_bytes(data.get('size_bytes', 0))}
State:        {data.get('state', 'Unknown')}
Progress:     [{progress_bar}] {progress:.1f}%{eta_str}
Download:     {_fmt_rate(data.get('download_rate', 0))}
Upload:       {_fmt_rate(data.get('upload_rate', 0))}
Peers:        {data.get('seeders', 0) + data.get('leechers', 0)} connected
Uploaded:     {_fmt_bytes(data.get('bytes_uploaded', 0))}
Downloaded:   {_fmt_bytes(data.get('bytes_downloaded', 0))}
Magnet URI:   magnet:?xt=urn:btih:{data.get('infohash', '')}"""

        if data.get("error"):
            content += f"\nError:        ⚠️ {data.get('error')}"

        console.print(Panel(content, title="🔗 Torrent Status", border_style="blue"))
    elif response.status_code == 404:
        console.print(f"❌ Torrent not found: {infohash}", style="red")
        raise SystemExit(1)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Status failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@torrent.command()
@click.argument("infohash")
@click.option("--api-url", default=None, help="PRSM API URL")
def seed(infohash: str, api_url: str):
    """Start seeding a torrent."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    try:
        response = httpx.post(
            f"{url}/api/v1/torrents/{infohash}/seed",
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        console.print(f"✅ Now seeding: {infohash[:16]}...", style="bold green")
        console.print("   Announced to PRSM network.", style="dim")
    elif response.status_code == 501:
        console.print("⚠️  Seeding control not yet implemented", style="yellow")
        raise SystemExit(1)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Seed failed: HTTP {response.status_code}", style="red")
        console.print(f"   {response.text[:200]}")
        raise SystemExit(1)


@torrent.command()
@click.argument("infohash")
@click.option("--delete-files", is_flag=True, help="Also delete local data files")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
@click.option("--api-url", default=None, help="PRSM API URL")
def unseed(infohash: str, delete_files: bool, yes: bool, api_url: str):
    """Stop seeding a torrent."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    if delete_files and not yes:
        if not click.confirm("This will delete local files. Continue?", default=False):
            console.print("Cancelled.", style="yellow")
            raise SystemExit(0)

    try:
        response = httpx.delete(
            f"{url}/api/v1/torrents/{infohash}/seed",
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        console.print(f"✅ Stopped seeding: {infohash[:16]}...", style="bold green")
        if delete_files:
            console.print("   Files deleted.", style="dim")
        else:
            console.print("   ⚠️  Files retained at download location", style="yellow")
    elif response.status_code == 400:
        console.print("⚠️  Could not stop seeding (not found or minimum seed time not met)", style="yellow")
        raise SystemExit(1)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Unseed failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@torrent.command()
@click.argument("infohash")
@click.option("--save-path", type=click.Path(), help="Destination directory")
@click.option("--timeout", type=float, default=3600.0, help="Max seconds to wait")
@click.option("--no-wait", is_flag=True, help="Fire-and-forget (return request_id immediately)")
@click.option("--api-url", default=None, help="PRSM API URL")
def download(infohash: str, save_path: Optional[str], timeout: float, no_wait: bool, api_url: str):
    """Download a torrent's content."""
    import httpx
    import time

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    payload = {
        "infohash": infohash,
        "save_path": save_path or str(Path.home() / ".prsm" / "torrents"),
        "timeout": timeout,
    }

    console.print("🔗 Starting download...", style="bold blue")

    try:
        response = httpx.post(
            f"{url}/api/v1/torrents/{infohash}/download",
            json=payload,
            headers=headers,
            timeout=30.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        request_id = data.get("request_id", "")

        if no_wait or data.get("status") == "completed":
            console.print(f"✅ Download request: {request_id}", style="bold green")
            if data.get("status") == "completed":
                console.print(f"   Downloaded: {_fmt_bytes(data.get('bytes_downloaded', 0))}")
            return

        # Poll for progress
        console.print(f"   Request ID: {request_id}")
        console.print("   Polling progress...", style="dim")

        start_time = time.time()
        with console.status("[bold green]Downloading...") as status:
            while time.time() - start_time < timeout:
                poll = httpx.get(
                    f"{url}/api/v1/torrents/{infohash}/download/{request_id}",
                    headers=headers,
                    timeout=10.0,
                )
                if poll.status_code == 200:
                    poll_data = poll.json()
                    progress = poll_data.get("progress", 0) * 100
                    status.update(f"[bold green]Downloading... {progress:.1f}%")

                    if poll_data.get("status") == "completed":
                        console.print(f"\n✅ Download complete!", style="bold green")
                        console.print(f"   Downloaded: {_fmt_bytes(poll_data.get('bytes_downloaded', 0))}")
                        return
                    elif poll_data.get("status") == "failed":
                        console.print(f"\n❌ Download failed: {poll_data.get('error', 'Unknown error')}", style="red")
                        raise SystemExit(1)

                time.sleep(2)

        console.print("\n⚠️  Download timed out", style="yellow")
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    elif response.status_code == 503:
        console.print("⚠️  BitTorrent requester not available", style="yellow")
        raise SystemExit(1)
    else:
        console.print(f"❌ Download failed: HTTP {response.status_code}", style="red")
        console.print(f"   {response.text[:200]}")
        raise SystemExit(1)


@torrent.command()
@click.argument("infohash")
@click.option("--api-url", default=None, help="PRSM API URL")
def peers(infohash: str, api_url: str):
    """List peers currently connected for a torrent."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    try:
        response = httpx.get(
            f"{url}/api/v1/torrents/{infohash}/peers",
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        peers_data = response.json()

        if not peers_data:
            console.print("No peers connected.", style="yellow")
            return

        table = Table(title="Connected Peers")
        table.add_column("IP:Port", style="cyan")
        table.add_column("Client", style="green")
        table.add_column("Downloaded", style="blue")
        table.add_column("Uploaded", style="magenta")
        table.add_column("Seed?", style="yellow")

        for p in peers_data:
            table.add_row(
                f"{p.get('ip', '?')}:{p.get('port', '?')}",
                p.get("client", "?")[:15],
                _fmt_bytes(p.get("downloaded", 0)),
                _fmt_bytes(p.get("uploaded", 0)),
                "✅" if p.get("is_seed") else "",
            )

        console.print(table)
        console.print(f"\n📊 Total: {len(peers_data)} peers", style="blue")
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Peers failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@torrent.command()
@click.option("--api-url", default=None, help="PRSM API URL")
def stats(api_url: str):
    """Show aggregate BitTorrent stats for this node."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    try:
        response = httpx.get(
            f"{url}/api/v1/torrents/stats",
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()

        if not data.get("available"):
            console.print("⚠️  BitTorrent client not available", style="yellow")
            return

        from rich.panel import Panel

        provider = data.get("provider", {})
        requester = data.get("requester", {})

        content = f"""Active torrents:   {provider.get('active_torrents', 0)} seeding, {requester.get('active_downloads', 0)} downloading
Total uploaded:   {_fmt_bytes(provider.get('total_uploaded_bytes', 0))}
Total downloaded: {_fmt_bytes(requester.get('total_downloaded_bytes', 0))}
Upload rate:       {_fmt_rate(provider.get('upload_rate', 0))}
Download rate:     {_fmt_rate(requester.get('download_rate', 0))}
FTNS earned:       {provider.get('total_rewards', 0):.4f}"""

        console.print(Panel(content, title="🔗 BitTorrent Stats", border_style="green"))
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Stats failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


# ============================================================================

# ============================================================================
# MONITORING COMMANDS (Phase 6)
# ============================================================================

@main.group()
def monitor():
    """System monitoring commands"""
    pass


@monitor.command()
@click.option("--watch", is_flag=True, help="Refresh every 5 seconds")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON for scripting")
@click.option("--api-url", default=None, help="PRSM API URL")
def health_cmd(watch: bool, json_output: bool, api_url: str):
    """Full health check across all PRSM subsystems."""
    import httpx
    import time

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    def fetch_health():
        try:
            return httpx.get(
                f"{url}/api/v1/health",
                headers=headers,
                timeout=10.0,
            )
        except httpx.ConnectError:
            return None

    def display_health(response):
        if response is None:
            console.print("❌ Cannot connect to API", style="red")
            return

        if response.status_code == 200:
            data = response.json()

            if json_output:
                import json
                console.print(json.dumps(data, indent=2))
                return

            from rich.panel import Panel

            status_icons = {
                "healthy": "✅",
                "degraded": "⚠️ ",
                "unhealthy": "❌",
            }

            services = data.get("services", {})
            lines = []
            for name, info in services.items():
                status = info.get("status", "unknown")
                icon = status_icons.get(status, "?")
                detail = info.get("detail", "")
                line = f"{name:15s}: {icon} {status.capitalize()}"
                if detail:
                    line += f"  ({detail})"
                lines.append(line)

            overall = data.get("overall", "unknown")
            overall_icon = status_icons.get(overall, "?")

            content = "\n".join(lines) + f"""
──────────────────────────────────────
Pipeline overall:  {overall_icon} {overall.capitalize()}"""

            console.print(Panel(content, title="🏥 System Health", border_style="green"))
        elif response.status_code == 401:
            console.print("❌ Session expired. Run: prsm login", style="red")
        else:
            console.print(f"❌ Health check failed: HTTP {response.status_code}", style="red")

    if watch:
        try:
            while True:
                console.clear()
                display_health(fetch_health())
                console.print("\n[dim]Press Ctrl+C to exit[/dim]")
                time.sleep(5)
        except KeyboardInterrupt:
            console.print("\n👋 Monitoring stopped", style="yellow")
    else:
        display_health(fetch_health())


@monitor.command()
@click.option("--period", type=click.Choice(["1h", "6h", "24h", "7d"]), default="1h", help="Time window")
@click.option("--watch", is_flag=True, help="Auto-refresh every 10 seconds")
@click.option("--api-url", default=None, help="PRSM API URL")
def metrics(period: str, watch: bool, api_url: str):
    """Show key performance metrics for the local node."""
    import httpx
    import time

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    def fetch_metrics():
        try:
            return httpx.get(
                f"{url}/api/v1/monitoring/metrics",
                params={"period": period},
                headers=headers,
                timeout=10.0,
            )
        except httpx.ConnectError:
            return None

    def display_metrics(response):
        if response is None:
            console.print("❌ Cannot connect to API", style="red")
            return

        if response.status_code == 200:
            data = response.json()

            from rich.panel import Panel

            compute = data.get("compute", {})
            network = data.get("network", {})
            economy = data.get("economy", {})

            content = f"""Period: Last {period}
────────── Compute ──────────
Queries served:      {compute.get('queries_served', 0)}
Avg query latency:   {compute.get('avg_latency_ms', 0)}ms
CPU usage:           {compute.get('cpu_percent', 0):.1f}%
Memory usage:        {compute.get('memory_used_gb', 0):.1f} GB / {compute.get('memory_total_gb', 0):.1f} GB
────────── Network ──────────
P2P bandwidth ↑:     {_fmt_bytes(network.get('p2p_upload_bytes', 0))}
P2P bandwidth ↓:     {_fmt_bytes(network.get('p2p_download_bytes', 0))}
BT seeded:           {_fmt_bytes(network.get('bt_seeded_bytes', 0))}
BT downloaded:       {_fmt_bytes(network.get('bt_downloaded_bytes', 0))}
────────── Economy ──────────
FTNS earned:         +{economy.get('ftns_earned', 0):.4f}
FTNS spent:           -{economy.get('ftns_spent', 0):.4f}"""

            console.print(Panel(content, title="📊 Node Metrics", border_style="blue"))
        elif response.status_code == 401:
            console.print("❌ Session expired. Run: prsm login", style="red")
        else:
            console.print(f"❌ Metrics failed: HTTP {response.status_code}", style="red")

    if watch:
        try:
            while True:
                console.clear()
                display_metrics(fetch_metrics())
                console.print("\n[dim]Press Ctrl+C to exit[/dim]")
                time.sleep(10)
        except KeyboardInterrupt:
            console.print("\n👋 Monitoring stopped", style="yellow")
    else:
        display_metrics(fetch_metrics())


@monitor.command()
@click.option("--level", type=click.Choice(["debug", "info", "warning", "error"]), default="info",
              help="Log level filter")
@click.option("--service", help="Filter by service: api, nwtn, storage, bittorrent, node")
@click.option("--limit", type=int, default=50, help="Lines to show")
@click.option("--follow", "-f", is_flag=True, help="Stream new log entries")
@click.option("--api-url", default=None, help="PRSM API URL")
def logs(level: str, service: Optional[str], limit: int, follow: bool, api_url: str):
    """Stream or retrieve recent structured log output."""
    import httpx
    import time

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    params = {"level": level, "limit": limit}
    if service:
        params["service"] = service

    def fetch_logs():
        try:
            return httpx.get(
                f"{url}/api/v1/monitoring/logs",
                params=params,
                headers=headers,
                timeout=10.0,
            )
        except httpx.ConnectError:
            return None

    def display_logs(response):
        if response is None:
            console.print("❌ Cannot connect to API", style="red")
            return

        if response.status_code == 200:
            data = response.json()
            logs_list = data.get("logs", [])

            for log in logs_list:
                timestamp = log.get("timestamp", "")[:19]
                log_level = log.get("level", "info").upper()
                log_service = log.get("service", "?")
                message = log.get("message", "")

                level_colors = {
                    "DEBUG": "dim",
                    "INFO": "blue",
                    "WARNING": "yellow",
                    "ERROR": "red",
                }
                level_color = level_colors.get(log_level, "white")

                console.print(f"[dim]{timestamp}[/dim] [{level_color}]{log_level:8s}[/{level_color}] [{log_service}] {message}")
        elif response.status_code == 401:
            console.print("❌ Session expired. Run: prsm login", style="red")
        else:
            console.print(f"❌ Logs failed: HTTP {response.status_code}", style="red")

    if follow:
        try:
            while True:
                console.clear()
                display_logs(fetch_logs())
                console.print("\n[dim]Press Ctrl+C to exit[/dim]")
                time.sleep(2)
        except KeyboardInterrupt:
            console.print("\n👋 Log streaming stopped", style="yellow")
    else:
        display_logs(fetch_logs())


# ============================================================================
# WORKFLOW COMMANDS (Phase 7)
# ============================================================================

@main.group()
def workflow():
    """Workflow scheduling commands"""
    pass


@workflow.command()
@click.option("--name", required=True, help="Workflow name")
@click.option("--description", help="Description")
@click.option("--trigger", required=True, help="Cron expression or 'manual'")
@click.option("--steps-file", type=click.Path(exists=True), help="JSON file describing workflow steps")
@click.option("--budget", type=float, help="Max FTNS per run")
@click.option("--api-url", default=None, help="PRSM API URL")
def create_cmd(name: str, description: Optional[str], trigger: str, steps_file: Optional[str],
               budget: Optional[float], api_url: str):
    """Create a new automated workflow."""
    import httpx
    import json

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    payload = {
        "name": name,
        "trigger": trigger,
    }
    if description:
        payload["description"] = description
    if budget:
        payload["max_budget"] = budget

    if steps_file:
        steps_path = Path(steps_file)
        try:
            steps = json.loads(steps_path.read_text())
            payload["steps"] = steps
        except json.JSONDecodeError:
            console.print("❌ Invalid JSON in steps file", style="red")
            raise SystemExit(1)

    console.print(f"📋 Creating workflow '{name}'...", style="bold blue")

    try:
        response = httpx.post(
            f"{url}/api/v1/workflows",
            json=payload,
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        console.print(f"✅ Workflow created: {data.get('workflow_id', 'N/A')}", style="bold green")
        if trigger != "manual":
            console.print(f"   Next run: {data.get('next_run', 'N/A')}", style="dim")
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Create failed: HTTP {response.status_code}", style="red")
        console.print(f"   {response.text[:200]}")
        raise SystemExit(1)


@workflow.command("list")
@click.option("--status", "status_filter", help="Filter: active, paused, failed")
@click.option("--limit", type=int, default=20, help="Number of entries")
@click.option("--api-url", default=None, help="PRSM API URL")
def list_workflows(status_filter: Optional[str], limit: int, api_url: str):
    """List all workflows for the current user."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    params = {"limit": limit}
    if status_filter:
        params["status"] = status_filter

    try:
        response = httpx.get(
            f"{url}/api/v1/workflows",
            params=params,
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        workflows = data.get("workflows", [])

        if not workflows:
            console.print("No workflows found.", style="yellow")
            return

        table = Table(title="📋 Workflows")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Trigger", style="magenta")
        table.add_column("Status", style="yellow")
        table.add_column("Last Run", style="dim")
        table.add_column("Next Run", style="blue")
        table.add_column("Runs", style="dim", justify="right")

        for w in workflows:
            workflow_id = w.get("workflow_id", "?")[:12]
            table.add_row(
                workflow_id,
                w.get("name", "?")[:20],
                w.get("trigger", "?"),
                w.get("status", "?"),
                w.get("last_run", "Never")[:10] if w.get("last_run") else "Never",
                w.get("next_run", "N/A")[:10] if w.get("next_run") else "N/A",
                str(w.get("run_count", 0)),
            )

        console.print(table)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ List failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@workflow.command()
@click.argument("workflow-id")
@click.option("--follow", "-f", is_flag=True, help="Stream execution logs until completion")
@click.option("--api-url", default=None, help="PRSM API URL")
def run(workflow_id: str, follow: bool, api_url: str):
    """Manually trigger a workflow run."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    console.print(f"▶️  Starting workflow {workflow_id[:12]}...", style="bold blue")

    try:
        response = httpx.post(
            f"{url}/api/v1/workflows/{workflow_id}/run",
            headers=headers,
            timeout=30.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        console.print(f"✅ Workflow run started: {data.get('run_id', 'N/A')}", style="bold green")
        console.print(f"   Status: {data.get('status', 'running')}")
    elif response.status_code == 404:
        console.print(f"❌ Workflow not found: {workflow_id}", style="red")
        raise SystemExit(1)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Run failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@workflow.command()
@click.argument("workflow-id")
@click.option("--run-id", help="Specific run (default: most recent)")
@click.option("--limit", type=int, default=100, help="Log lines")
@click.option("--api-url", default=None, help="PRSM API URL")
def logs_cmd(workflow_id: str, run_id: Optional[str], limit: int, api_url: str):
    """Show execution logs for a workflow."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    # If no run_id, get the most recent run
    if not run_id:
        run_id = "latest"

    params = {"limit": limit}

    try:
        response = httpx.get(
            f"{url}/api/v1/workflows/{workflow_id}/runs/{run_id}/logs",
            params=params,
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        logs_list = data.get("logs", [])

        if not logs_list:
            console.print("No logs found.", style="yellow")
            return

        console.print(f"📋 Workflow {workflow_id[:12]} - Run {run_id[:12] if run_id != 'latest' else 'latest'}",
                     style="bold blue")

        for log in logs_list:
            timestamp = log.get("timestamp", "")[:19]
            level = log.get("level", "info").upper()
            message = log.get("message", "")
            step = log.get("step", "")

            level_colors = {
                "DEBUG": "dim",
                "INFO": "blue",
                "WARNING": "yellow",
                "ERROR": "red",
            }
            level_color = level_colors.get(level, "white")

            step_str = f"[{step}] " if step else ""
            console.print(f"[dim]{timestamp}[/dim] [{level_color}]{level:8s}[/{level_color}] {step_str}{message}")
    elif response.status_code == 404:
        console.print(f"❌ Workflow or run not found: {workflow_id}/{run_id}", style="red")
        raise SystemExit(1)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Logs failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)



@main.group()
def mcp():
    """MCP server for AI agent integration."""
    pass


@mcp.command("start")
@click.option("--host", default="localhost", show_default=True, help="Host to bind the MCP server")
@click.option("--port", default=9100, show_default=True, help="Port for the MCP server")
def mcp_start(host: str, port: int):
    """Start the PRSM MCP server for AI agent integration.

    Exposes all installed PRSM skill tools as MCP-compatible endpoints
    that AI agents (Hermes, OpenClaw, Claude Desktop) can discover and invoke.
    """
    from prsm.cli_modules.mcp_server import is_fastmcp_available, start_mcp_server

    if not is_fastmcp_available():
        # Sprint 538 F70 fix: point operators to the PRSM-canonical
        # install path (`.[mcp]` extra) instead of generic
        # `pip install fastmcp`. The `[mcp]` extra also pulls the
        # official `mcp>=1.27.0` SDK that fastmcp wraps + ensures
        # version compat. Generic install can mismatch.
        # F70b: escape `[mcp]` for Rich (square brackets = markup).
        console.print(
            "✗ fastmcp is required to run the MCP server.\n"
            "  Install via PRSM extra (recommended — pins compatible\n"
            "  versions of mcp + fastmcp):\n"
            "    pip install -e '.\\[mcp]'\n"
            "  Or standalone (may not match PRSM's pinned versions):\n"
            "    pip install 'fastmcp>=2.0.0' 'mcp>=1.27.0'",
            style="red",
        )
        raise SystemExit(1)

    console.print(f"◇ Starting PRSM MCP server on {host}:{port}...", style="bold")
    console.print(f"  Endpoint: http://{host}:{port}/mcp", style="dim")
    console.print("  Press Ctrl+C to stop.\n", style="dim")
    start_mcp_server(host=host, port=port)


@mcp.command("config-snippet")
@click.option("--host", default="localhost", show_default=True, help="Host for the snippet")
@click.option("--port", default=9100, show_default=True, help="Port for the snippet")
def mcp_config_snippet(host: str, port: int):
    """Print a YAML config snippet for Hermes/OpenClaw MCP client setup.

    Add the printed snippet to your AI client's configuration file
    to connect it to the PRSM MCP server.
    """
    from prsm.cli_modules.mcp_server import get_config_snippet

    console.print("Add this to your Hermes or OpenClaw config:\n", style="dim")
    snippet = get_config_snippet(host=host, port=port)
    console.print(snippet)


@mcp.command("status")
@click.option("--format", "output_format",
              type=click.Choice(["text", "json"]), default="text",
              help="Output format")
def mcp_status_cmd(output_format: str):
    """Show MCP server status.

    Reports whether the MCP server is running based on config AND
    actual port-bind state.
    """
    from prsm.cli_modules.config_schema import PRSMConfig

    cfg = PRSMConfig.load()
    enabled = cfg.mcp_server_enabled
    port = cfg.mcp_server_port

    # Sprint 534 F60 fix: probe the actual port instead of trusting
    # config-says-enabled. Pre-fix: `prsm mcp status` reported
    # "enabled (:9100)" while the port was closed — misleading
    # for operators trying to integrate Claude Desktop / Gemini CLI.
    running = False
    if enabled:
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                running = s.connect_ex(("127.0.0.1", port)) == 0
        except Exception:
            running = False

    data = {
        "ok": True,
        "mcp_enabled": enabled,
        "mcp_running": running,
        "mcp_port": port,
        "config_path": str(PRSMConfig.config_path()),
    }

    if output_format == "json":
        _agent_output(data)
        return

    if enabled and running:
        console.print(
            f"\n  {ICONS['success']} MCP Server: running (:{port})",
        )
        console.print(
            f"  Config:  {PRSMConfig.config_path()}",
            style=THEME.dim,
        )
    elif enabled and not running:
        console.print(
            f"\n  ⚠️  MCP Server: enabled in config but NOT listening on :{port}",
            style="yellow",
        )
        console.print(
            f"  Start:   prsm mcp start", style=THEME.dim,
        )
        console.print(
            f"  Config:  {PRSMConfig.config_path()}",
            style=THEME.dim,
        )
    else:
        console.print(f"\n  {ICONS['info']} MCP Server: disabled")
        console.print(
            f"  Enable:  prsm config set mcp_server_enabled true",
            style=THEME.dim,
        )


# ---------------------------------------------------------------------------
# config CLI group (Phase 2)
# ---------------------------------------------------------------------------

@main.group()
def config():
    """Configuration management commands."""
    pass


@config.command("show")
@click.option("--format", "output_format",
              type=click.Choice(["text", "json"]), default="text",
              help="Output format: 'text' (human) or 'json' (agent-parseable)")
def config_show(output_format: str):
    """Display current PRSM configuration.

    Shows all settings from ~/.prsm/config.yaml grouped by category.
    Use --format json for machine-readable output.
    """
    from prsm.cli_modules.config_schema import PRSMConfig

    cfg = PRSMConfig.load()

    data = {
        "ok": True,
        "config_path": str(PRSMConfig.config_path()),
        "config": cfg.model_dump(),
    }

    if output_format == "json":
        _agent_output(data)
        return

    # Rich table output
    console.print(f"\n[bold]PRSM Configuration[/bold]", style="cyan")
    console.print(f"[dim]Source: {PRSMConfig.config_path()}[/dim]\n")

    # Node Identity
    table = Table(title="Node Identity", show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("display_name", cfg.display_name)
    table.add_row("node_role", cfg.node_role.value)
    console.print(table)

    # Resources
    table = Table(title="Resources", show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("cpu_pct", f"{cfg.cpu_pct}%")
    table.add_row("memory_pct", f"{cfg.memory_pct}%")
    table.add_row("gpu_pct", f"{cfg.gpu_pct}%")
    table.add_row("storage_gb", f"{cfg.storage_gb} GB")
    table.add_row("max_concurrent_jobs", str(cfg.max_concurrent_jobs))
    if cfg.upload_mbps_limit > 0:
        table.add_row("upload_mbps_limit", f"{cfg.upload_mbps_limit} Mbps")
    else:
        table.add_row("upload_mbps_limit", "unlimited")
    if cfg.active_hours_start is not None and cfg.active_hours_end is not None:
        table.add_row("active_hours", f"{cfg.active_hours_start:02d}:00 - {cfg.active_hours_end:02d}:00")
    if cfg.active_days:
        table.add_row("active_days", ", ".join(str(d) for d in cfg.active_days))
    console.print(table)

    # Network
    table = Table(title="Network", show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("p2p_port", str(cfg.p2p_port))
    table.add_row("api_port", str(cfg.api_port))
    # Sprint 533 F52 fix: detect known-stale bootstrap_nodes
    # placeholder patterns and surface a migration hint. The
    # `/dns4/.../p2p/QmPRSM1` pattern was a pre-sprint-148 default
    # using fake PeerIDs; canonical is now `wss://bootstrap-eu.
    # prsm-network.com:8765` (sprint-385 DNS fleet).
    _stale = bool(
        cfg.bootstrap_nodes
        and any(
            "QmPRSM" in b or "bootstrap1.prsm.network" in b
            for b in cfg.bootstrap_nodes
        )
    )
    if cfg.bootstrap_nodes:
        table.add_row("bootstrap_nodes", ", ".join(cfg.bootstrap_nodes))
    else:
        table.add_row("bootstrap_nodes", "[dim]none configured[/dim]")
    console.print(table)
    if _stale:
        console.print(
            "[yellow]⚠️  bootstrap_nodes contains pre-sprint-148 "
            "placeholder values (QmPRSM1 / bootstrap1.prsm.network).[/yellow]\n"
            "[dim]Run `prsm setup --reset` or `prsm config reset` to "
            "refresh from canonical fleet (wss://bootstrap-eu/apac.prsm-network.com:8765).[/dim]"
        )

    # API Keys
    table = Table(title="API Keys", show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("has_openai_key", "✓" if cfg.has_openai_key else "✗")
    table.add_row("has_anthropic_key", "✓" if cfg.has_anthropic_key else "✗")
    table.add_row("has_huggingface_token", "✓" if cfg.has_huggingface_token else "✗")
    console.print(table)

    # AI Integration
    table = Table(title="AI Integration", show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("mcp_server_enabled", "✓" if cfg.mcp_server_enabled else "✗")
    table.add_row("mcp_server_port", str(cfg.mcp_server_port))
    console.print(table)

    # FTNS Wallet
    if cfg.wallet_address:
        table = Table(title="FTNS Wallet", show_header=False, box=None)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("wallet_address", cfg.wallet_address)
        console.print(table)

    # Meta
    console.print(f"\n[dim]setup_completed: {cfg.setup_completed} | setup_version: {cfg.setup_version}[/dim]")


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str):
    """Set a configuration value.

    Updates ~/.prsm/config.yaml with the new value.
    Example: prsm config set cpu_pct 70
    """
    from prsm.cli_modules.config_schema import PRSMConfig, NodeRole

    cfg = PRSMConfig.load()

    # Check if key exists
    valid_fields = list(cfg.model_dump().keys())
    if key not in valid_fields:
        console.print(f"[red]Error: Unknown configuration key '{key}'[/red]")
        console.print(f"\n[dim]Valid keys:[/dim]")
        for f in sorted(valid_fields):
            console.print(f"  • {f}")
        raise SystemExit(1)

    # Get old value
    old_value = getattr(cfg, key)

    # Parse and set new value based on field type
    try:
        # Get the field type from the model
        field_info = type(cfg).model_fields.get(key)
        if field_info:
            field_type = field_info.annotation
            # Handle Optional types
            if hasattr(field_type, "__origin__") and field_type.__origin__ is type(None) | type(str):
                # This is Optional[X], extract X
                import typing
                args = typing.get_args(field_type)
                if args:
                    field_type = args[0]

            # Convert value based on type
            if field_type == bool or (hasattr(field_type, "__origin__") and field_type.__origin__ is bool):
                # Boolean parsing
                if value.lower() in ("true", "1", "yes", "on"):
                    new_value = True
                elif value.lower() in ("false", "0", "no", "off"):
                    new_value = False
                else:
                    raise ValueError(f"Cannot convert '{value}' to boolean")
            elif field_type == int or (isinstance(field_type, type) and issubclass(field_type, int)):
                new_value = int(value)
            elif field_type == float or (isinstance(field_type, type) and issubclass(field_type, float)):
                new_value = float(value)
            elif field_type == NodeRole:
                new_value = NodeRole(value.lower())
            elif field_type == list or (hasattr(field_type, "__origin__") and field_type.__origin__ is list):
                # Parse comma-separated list
                new_value = [v.strip() for v in value.split(",") if v.strip()]
            else:
                # Default to string
                new_value = value
        else:
            new_value = value

        setattr(cfg, key, new_value)

        # Validate via Pydantic
        cfg.model_validate(cfg.model_dump())

        # Save
        cfg.save()

        console.print(f"[green]✓ Updated {key}[/green]")
        console.print(f"  Before: {old_value!r}")
        console.print(f"  After:  {new_value!r}")

    except ValueError as e:
        console.print(f"[red]Error: Invalid value for '{key}': {e}[/red]")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Error setting configuration: {e}[/red]")
        raise SystemExit(1)


@config.command("reset")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def config_reset(yes: bool):
    """Delete the configuration file.

    Removes ~/.prsm/config.yaml. Run 'prsm setup' to reconfigure.
    """
    from prsm.cli_modules.config_schema import PRSMConfig

    config_path = PRSMConfig.config_path()

    if not config_path.exists():
        console.print("[dim]No configuration file to delete.[/dim]")
        return

    if not yes:
        if not click.confirm(f"Delete {config_path}?"):
            console.print("[dim]Cancelled.[/dim]")
            return

    config_path.unlink()
    console.print(f"[green]✓ Deleted {config_path}[/green]")


@config.command("validate")
@click.option("--format", "output_format",
              type=click.Choice(["text", "json"]), default="text",
              help="Output format: 'text' (human) or 'json' (agent-parseable)")
def config_validate(output_format: str):
    """Validate configuration and check port availability.

    Runs Pydantic validation and checks that p2p_port and api_port are available.
    Exit code: 0 if all pass/warn, 1 if any fail.
    """
    from prsm.cli_modules.config_schema import PRSMConfig

    cfg = PRSMConfig.load()
    checks = []
    any_fail = False

    # 1. Pydantic validation
    try:
        cfg.model_validate(cfg.model_dump())
        checks.append({
            "name": "Pydantic validation",
            "status": "PASS",
            "details": "All fields valid",
        })
    except Exception as e:
        checks.append({
            "name": "Pydantic validation",
            "status": "FAIL",
            "details": str(e),
        })
        any_fail = True

    # 2. P2P port availability
    p2p_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    p2p_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        p2p_sock.bind(("127.0.0.1", cfg.p2p_port))
        checks.append({
            "name": "P2P port availability",
            "status": "PASS",
            "details": f"Port {cfg.p2p_port} is available",
        })
    except Exception as e:
        checks.append({
            "name": "P2P port availability",
            "status": "WARN",
            "details": f"Port {cfg.p2p_port} may be in use: {e}",
        })
    finally:
        p2p_sock.close()

    # 3. API port availability
    api_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    api_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        api_sock.bind(("127.0.0.1", cfg.api_port))
        checks.append({
            "name": "API port availability",
            "status": "PASS",
            "details": f"Port {cfg.api_port} is available",
        })
    except Exception as e:
        checks.append({
            "name": "API port availability",
            "status": "WARN",
            "details": f"Port {cfg.api_port} may be in use: {e}",
        })
    finally:
        api_sock.close()

    # 4. MCP port availability (if enabled)
    if cfg.mcp_server_enabled:
        mcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        mcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            mcp_sock.bind(("127.0.0.1", cfg.mcp_server_port))
            checks.append({
                "name": "MCP port availability",
                "status": "PASS",
                "details": f"Port {cfg.mcp_server_port} is available",
            })
        except Exception as e:
            checks.append({
                "name": "MCP port availability",
                "status": "WARN",
                "details": f"Port {cfg.mcp_server_port} may be in use: {e}",
            })
        finally:
            mcp_sock.close()

    # 5. Config file exists
    checks.append({
        "name": "Config file exists",
        "status": "PASS" if PRSMConfig.exists() else "WARN",
        "details": str(PRSMConfig.config_path()) if PRSMConfig.exists() else "Run 'prsm setup' to create",
    })

    # Output
    result = {
        "ok": not any_fail,
        "config_path": str(PRSMConfig.config_path()),
        "checks": checks,
    }

    if output_format == "json":
        _agent_output(result)
        raise SystemExit(1 if any_fail else 0)

    # Rich output
    console.print("\n[bold]Configuration Validation[/bold]\n")
    table = Table(show_header=True)
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="dim")

    for check in checks:
        status = check["status"]
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(status, status)
        status_style = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}.get(status, "white")
        table.add_row(check["name"], f"[{status_style}]{icon} {status}[/{status_style}]", check["details"])

    console.print(table)

    if any_fail:
        console.print("\n[red]Validation failed. Fix errors above.[/red]")
        raise SystemExit(1)
    else:
        console.print("\n[green]All checks passed or warned.[/green]")


@config.command("path")
def config_path_cmd():
    """Print the configuration file path.

    Useful for scripts and automation.
    """
    from prsm.cli_modules.config_schema import PRSMConfig
    console.print(str(PRSMConfig.config_path()))


@config.command("get")
@click.argument("key")
def config_get(key: str):
    """Get a single config value printed to stdout.

    Useful for shell scripts: VALUE=$(prsm config get cpu_pct)
    """
    from prsm.cli_modules.config_schema import PRSMConfig
    import enum

    cfg = PRSMConfig.load()
    if not hasattr(cfg, key):
        console.print(f"[red]Unknown config key: {key}[/red]")
        console.print(f"Available keys: {', '.join(sorted(k for k in cfg.model_fields if not k.startswith('_')))}")
        raise SystemExit(1)

    val = getattr(cfg, key)
    if isinstance(val, enum.Enum):
        print(val.value)
    elif isinstance(val, list):
        print(",".join(str(v) for v in val))
    elif val is None:
        print("")
    else:
        print(val)


@config.command("export")
def config_export():
    """Export current configuration as YAML to stdout.

    Useful for backups or transferring config between nodes.
    """
    from prsm.cli_modules.config_schema import PRSMConfig
    import yaml

    cfg = PRSMConfig.load()
    data = cfg.model_dump(mode='json')
    yaml.safe_dump(data, sys.stdout, default_flow_style=False, sort_keys=False)


@config.command("import")
@click.argument("filepath", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def config_import(filepath: Path, yes: bool):
    """Import configuration from a YAML file.

    Overwrites the current config at ~/.prsm/config.yaml.
    """
    from prsm.cli_modules.config_schema import PRSMConfig
    import yaml

    if not yes:
        if not click.confirm(f"Import config from {filepath}? This will overwrite current settings."):
            console.print("[dim]Cancelled.[/dim]")
            return

    try:
        data = yaml.safe_load(filepath.read_text()) or {}
        cfg = PRSMConfig(**data)
        cfg.save()
        console.print(f"[green]{ICONS['success']} Config imported from {filepath}[/green]")
        console.print(f"  Saved to {PRSMConfig.config_path()}")
    except Exception as e:
        console.print(f"[red]{ICONS['error']} Import failed: {e}[/red]")
        raise SystemExit(1)


@config.command("wizard")
def config_wizard():
    """Re-enter the full interactive setup wizard.

    Alias for `prsm setup`.
    """
    from prsm.cli_modules.setup_wizard import run_setup_wizard
    run_setup_wizard()


# ---------------------------------------------------------------------------
# daemon CLI group — deprecated, delegates to `prsm node`
# ---------------------------------------------------------------------------

@main.group()
def daemon():
    """DEPRECATED: use `prsm node` instead.

    All daemon commands are now available under `prsm node`:
      prsm node start --background   # start in background
      prsm node stop                 # stop background node
      prsm node restart              # restart background node
      prsm node status               # show status
      prsm node logs -f              # follow logs
      prsm node install              # system service
    """
    console.print()
    console.print("  prsm daemon is deprecated.", style="yellow")
    console.print("  Use `prsm node` commands instead:", style="yellow")
    console.print("    prsm node start --background  -- Start in background", style="dim")
    console.print("    prsm node stop                -- Stop background node", style="dim")
    console.print("    prsm node restart             -- Restart background node", style="dim")
    console.print("    prsm node status              -- Show node status", style="dim")
    console.print("    prsm node logs -f             -- Follow logs live", style="dim")
    console.print()


@daemon.command("start")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def daemon_start(host: str, port: int):
    """DEPRECATED: use `prsm node start --background`."""
    from prsm.cli_modules.daemon import daemon_start as _start
    _start(host=host, port=port)


@daemon.command("stop")
@click.option("--timeout", default=10, help="Seconds to wait for graceful shutdown")
def daemon_stop(timeout: int):
    """DEPRECATED: use `prsm node stop`."""
    from prsm.cli_modules.daemon import daemon_stop as _stop
    _stop(timeout=timeout)


@daemon.command("restart")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--timeout", default=10, help="Seconds to wait for graceful shutdown")
def daemon_restart(host: str, port: int, timeout: int):
    """DEPRECATED: use `prsm node restart`."""
    from prsm.cli_modules.daemon import daemon_restart as _restart
    _restart(host=host, port=port, timeout=timeout)


@daemon.command("status")
@click.option("--format", "output_format",
              type=click.Choice(["text", "json"]), default="text",
              help="Output format")
def daemon_status(output_format: str):
    """DEPRECATED: use `prsm node status`."""
    from prsm.cli_modules.daemon import daemon_status as _status
    _status(output_format=output_format)


@daemon.command("logs")
@click.option("--lines", "-n", default=50, help="Number of lines to show")
@click.option("--follow", "-f", is_flag=True, help="Follow log output (tail -f)")
def daemon_logs(lines: int, follow: bool):
    """DEPRECATED: use `prsm node logs`."""
    from prsm.cli_modules.daemon import daemon_logs as _logs
    _logs(lines=lines, follow=follow)


@daemon.command("install")
@click.option("--dry-run", is_flag=True, help="Print service file without installing")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def daemon_install(dry_run: bool, host: str, port: int):
    """DEPRECATED: use `prsm node install`."""
    from prsm.cli_modules.daemon import daemon_service_install as _install
    _install(dry_run=dry_run, host=host, port=port)


@daemon.command("uninstall")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def daemon_uninstall(yes: bool):
    """DEPRECATED: use `prsm node uninstall`."""
    from prsm.cli_modules.daemon import daemon_service_uninstall as _uninstall
    _uninstall(yes=yes)


# ---------------------------------------------------------------------------
# wallet CLI group (T6.2)
#
# Surface real on-chain state to users: FTNS balance + claimable royalties +
# claim-flow. Reads from prsm/config/networks.py for per-network contract
# addresses; reads PRIVATE_KEY from env (typically loaded from
# ~/.prsm/<network>-deployer.env). Without PRIVATE_KEY, view-only mode:
# can show balance + claimable for any address but can't claim.
# ---------------------------------------------------------------------------


def _wallet_load_signer(network_name: str) -> dict:
    """Resolve wallet identity for the wallet CLI."""
    import os
    from prsm.config.networks import get_network_config

    cfg = get_network_config(network_name)
    rpc_url = (os.environ.get("BASE_SEPOLIA_RPC_URL")
               if network_name == "testnet"
               else os.environ.get("PRSM_BASE_RPC_URL"))
    if not rpc_url:
        rpc_url = cfg.rpc_url_default

    pk = (os.environ.get("PRIVATE_KEY")
          or os.environ.get("FTNS_WALLET_PRIVATE_KEY")
          or "").strip() or None
    address = None
    if pk:
        try:
            from eth_account import Account
            address = Account.from_key(pk).address
        except Exception as exc:
            console.print(
                f"⚠️  could not derive address from PRIVATE_KEY: {exc}",
                style="yellow")

    return {
        "network": cfg,
        "address": address,
        "private_key": pk,
        "rpc_url": rpc_url,
    }


def _wallet_read_balance_wei(rpc_url: str, ftns_token: str, address: str) -> int:
    """Read FTNS balanceOf(address) using a minimal Web3 call."""
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    erc20_abi = [{
        "inputs": [{"name": "owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    }]
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(ftns_token), abi=erc20_abi)
    return int(contract.functions.balanceOf(
        Web3.to_checksum_address(address)).call())


def _wallet_read_eth_balance_wei(rpc_url: str, address: str) -> int:
    """Sprint 508: native ETH balance for gas-runway visibility."""
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    return int(w3.eth.get_balance(Web3.to_checksum_address(address)))


def _wallet_read_inbound_count(
    rpc_url: str,
    ftns_token: str,
    address: str,
    lookback_blocks: int = 10000,
) -> tuple:
    """Sprint 518: aggregate inbound count + total in a
    window using the sprint-512 scan helper. RPC-direct
    so it works without a running daemon.

    Returns (count, total_ftns).
    """
    from web3 import Web3
    from prsm.economy.ftns_onchain import (
        _ERC20_ABI, scan_inbound_transfers,
    )
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(ftns_token),
        abi=_ERC20_ABI,
    )
    latest = w3.eth.block_number
    from_block = max(0, latest - lookback_blocks)
    transfers = scan_inbound_transfers(
        contract,
        recipient=address,
        from_block=from_block,
        to_block=latest,
    )
    total = sum(t.get("amount_ftns", 0.0) for t in transfers)
    return (len(transfers), total)


@main.group()
def content():
    """Content publishing — view uploads, royalties accrued."""
    pass


@content.command("mine")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
)
@click.option("--limit", default=20, type=int)
def content_mine(api_port, output_format, limit):
    """List content this node has uploaded.

    Each entry shows filename, size, royalty rate, access count,
    accumulated FTNS royalties, and on-chain provenance status.
    """
    import json
    import httpx

    url = f"http://127.0.0.1:{api_port}/content/mine?limit={limit}"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
    except httpx.RequestError as exc:
        console.print(
            f"[red]Cannot reach PRSM node at {url}[/red]\n"
            f"[dim]Details: {exc}[/dim]"
        )
        sys.exit(2)

    if resp.status_code == 503:
        console.print(
            f"[yellow]ContentUploader not configured.[/yellow]\n"
            f"[dim]{resp.json().get('detail', 'unknown')}[/dim]"
        )
        sys.exit(0)
    if resp.status_code != 200:
        console.print(
            f"[red]/content/mine returned "
            f"{resp.status_code}[/red]: {resp.text}"
        )
        sys.exit(1)

    body = resp.json()
    if output_format == "json":
        console.print(json.dumps(body, indent=2))
        return

    entries = body.get("entries", [])
    total = body.get("total", 0)
    console.print(
        f"[bold]My Uploaded Content[/bold] "
        f"(showing {len(entries)} of {total}):"
    )
    if not entries:
        console.print(
            f"  [dim]No uploads yet. POST to /content/upload "
            f"or /content/upload/shard.[/dim]"
        )
        return
    for e in entries:
        cid = e.get("content_id", "?")
        if len(cid) > 22:
            cid = cid[:14] + ".." + cid[-6:]
        fn = e.get("filename", "?")
        if len(fn) > 18:
            fn = fn[:15] + "..."
        royalties = e.get("total_royalties", 0.0)
        hits = e.get("access_count", 0)
        size = e.get("size_bytes", 0)
        # Escape brackets so rich doesn't interpret as markup tags
        prov = (
            r"\[chain]" if e.get("provenance_tx_hash")
            else r"\[off]"
        )
        console.print(
            f"  {cid:<22}  {fn:<18}  {size:>8}b  "
            f"{royalties:>9.6f} FTNS  hits={hits:<4}  {prov}"
        )


@main.group()
def wallet():
    """On-chain wallet — view balance, claim royalties."""
    pass


@wallet.command("info")
@click.option(
    "--network", "network_name",
    default="testnet",
    type=click.Choice(["mainnet", "testnet"]),
    help="Network to query (default: testnet)",
)
@click.option(
    "--address",
    default=None,
    help="Override wallet address (default: derive from PRIVATE_KEY env)",
)
def wallet_info(network_name: str, address):
    """Show on-chain wallet state: address, FTNS balance, claimable royalties.

    \b
    Reads from:
      - PRIVATE_KEY env (or FTNS_WALLET_PRIVATE_KEY) — derives address
      - prsm/config/networks.py — for contract addresses
      - BASE_SEPOLIA_RPC_URL (testnet) or PRSM_BASE_RPC_URL (mainnet)

    \b
    Example:
        prsm wallet info --network testnet
        prsm wallet info --network testnet --address 0xabc...
    """
    ctx = _wallet_load_signer(network_name)
    cfg = ctx["network"]

    addr = address or ctx["address"]
    if not addr:
        console.print(
            "❌ no address available — set PRIVATE_KEY env var "
            "or pass --address", style="red")
        raise SystemExit(1)
    if not cfg.ftns_token:
        console.print(
            f"❌ {network_name} FTNS token address not configured in "
            f"prsm/config/networks.py", style="red")
        raise SystemExit(1)

    console.print(f"\n[bold]Wallet on {cfg.name}[/bold] (chainId {cfg.chain_id})")
    console.print(f"Address:        {addr}")
    console.print(f"Explorer:       {cfg.explorer_url}/address/{addr}")
    console.print(f"RPC:            {ctx['rpc_url']}")
    console.print()

    # FTNS balance
    try:
        bal_wei = _wallet_read_balance_wei(ctx['rpc_url'], cfg.ftns_token, addr)
        bal = bal_wei / 1e18
        console.print(f"FTNS balance:   [bold]{bal:,.6f}[/bold] FTNS  "
                      f"({cfg.ftns_token[:10]}…)")
    except Exception as exc:
        console.print(f"FTNS balance:   ⚠️  read failed: {exc}", style="yellow")

    # Sprint 508 — native ETH balance (gas runway).
    try:
        eth_wei = _wallet_read_eth_balance_wei(ctx['rpc_url'], addr)
        eth = eth_wei / 1e18
        if eth < 0.0001:
            status_color = "bold red"
            status_label = "CRITICAL"
        elif eth < 0.0005:
            status_color = "yellow"
            status_label = "LOW"
        else:
            status_color = "bold green"
            status_label = "ok"
        console.print(
            f"ETH balance:    [bold]{eth:.10f}[/bold] ETH  "
            f"[[{status_color}]{status_label}[/{status_color}]]"
        )
        if status_label == "CRITICAL":
            console.print(
                "  ⚠️  Top up ETH now — on-chain TX will start failing.",
                style="bold red",
            )
        elif status_label == "LOW":
            console.print(
                "  ⚠️  Gas is low — plan to top up soon.",
                style="yellow",
            )
    except Exception as exc:
        console.print(f"ETH balance:    ⚠️  read failed: {exc}", style="yellow")

    # Sprint 518 — recent inbound FTNS count (scan Transfer
    # events for `to == address`). Uses sprint-512 helper.
    try:
        inb_count, inb_total = _wallet_read_inbound_count(
            ctx['rpc_url'], cfg.ftns_token, addr,
        )
        if inb_count > 0:
            console.print(
                f"Inbound:        [bold]{inb_count}[/bold] receipts  "
                f"([green]{inb_total:.6f}[/green] FTNS total, last ~10k blocks)"
            )
        else:
            console.print(
                f"Inbound:        0 receipts in last ~10k blocks",
                style="dim",
            )
    except Exception as exc:
        console.print(f"Inbound:        ⚠️  read failed: {exc}", style="yellow")

    # Claimable royalties
    if cfg.royalty_distributor:
        try:
            from prsm.economy.web3.royalty_distributor import (
                RoyaltyDistributorClient,
            )
            client = RoyaltyDistributorClient(
                rpc_url=ctx['rpc_url'],
                distributor_address=cfg.royalty_distributor,
                ftns_token_address=cfg.ftns_token,
                private_key=None,
            )
            claimable_wei = client.claimable(addr)
            claimable = claimable_wei / 1e18
            color = "bold green" if claimable > 0 else "dim"
            console.print(f"Claimable:      [{color}]{claimable:,.6f}[/{color}] "
                          f"FTNS  ({cfg.royalty_distributor[:10]}…)")
            if claimable > 0:
                console.print(f"\n  → run [bold]prsm wallet claim --network "
                              f"{network_name}[/bold] to withdraw")
        except Exception as exc:
            console.print(f"Claimable:      ⚠️  read failed: {exc}",
                          style="yellow")
    else:
        console.print(
            f"Claimable:      (RoyaltyDistributor not configured "
            f"for {network_name})", style="dim")

    console.print()
    if network_name == "testnet":
        for note in cfg.notes:
            console.print(f"  ℹ {note}", style="dim")
    console.print()


@wallet.command("gas-status")
@click.option("--api-url", default=None, help="PRSM daemon API URL")
def wallet_gas_status(api_url: str) -> None:
    """Show ETH gas balance for the daemon's loaded wallet.

    Queries /wallet/gas-status on the running daemon. Surfaces
    status=ok|low|critical so operators can top up before
    on-chain TX start failing with cryptic "insufficient funds
    for gas" errors.
    """
    import httpx
    url = _api_url_from_creds(api_url)
    try:
        r = httpx.get(f"{url}/wallet/gas-status", timeout=10.0)
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        raise SystemExit(1)
    if r.status_code == 503:
        try:
            detail = r.json().get("detail", "")
        except Exception:
            detail = r.text[:300]
        console.print("❌ Gas status unavailable:", style="red")
        console.print(f"   {detail}")
        raise SystemExit(1)
    if r.status_code != 200:
        console.print(f"❌ HTTP {r.status_code}: {r.text[:200]}", style="red")
        raise SystemExit(1)
    d = r.json()
    status_color = {
        "ok": "bold green",
        "low": "yellow",
        "critical": "bold red",
        "unavailable": "dim",
    }.get(d.get("status"), "white")
    console.print(f"\n[bold]Operator gas balance[/bold]")
    console.print(f"  address       : {d.get('address')}")
    if d.get("eth_balance") is not None:
        console.print(f"  ETH balance   : {d['eth_balance']:.10f} ETH")
    else:
        console.print(f"  ETH balance   : unavailable")
    console.print(f"  status        : [{status_color}]{d.get('status', '?').upper()}[/{status_color}]")
    console.print(f"  low threshold : {d.get('low_threshold_eth')} ETH")
    console.print(f"  critical thr. : {d.get('critical_threshold_eth')} ETH")
    if d.get("status") == "critical":
        console.print(
            "\n⚠️  Top up ETH now — broadcasts will start failing soon.",
            style="bold red",
        )
    elif d.get("status") == "low":
        console.print(
            "\n⚠️  Plan to top up ETH soon to avoid mid-job failures.",
            style="yellow",
        )
    console.print()


@wallet.command("deposit-info")
@click.option("--api-url", default=None, help="PRSM daemon API URL")
def wallet_deposit_info(api_url: str) -> None:
    """Show bridge deposit escrow address + linkage status.

    Sprint 540 Pattern A: daemon-mediated bridge. Deposit on-chain
    FTNS into your off-chain PRSM balance by:
      1. Link your sending ETH address (prsm wallet link-address)
      2. Sign an on-chain ERC-20 transfer to the escrow address
      3. Daemon credits your off-chain balance after detected inbound
    """
    import httpx
    url = _api_url_from_creds(api_url)
    try:
        r = httpx.get(f"{url}/wallet/deposit/info", timeout=10.0)
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        raise SystemExit(1)
    if r.status_code == 503:
        try:
            detail = r.json().get("detail", "")
        except Exception:
            detail = r.text[:300]
        console.print("❌ Deposit flow unavailable:", style="red")
        console.print(f"   {detail}")
        raise SystemExit(1)
    if r.status_code != 200:
        console.print(f"❌ HTTP {r.status_code}", style="red")
        raise SystemExit(1)
    d = r.json()
    console.print("\n[bold]Bridge deposit info[/bold]")
    console.print(f"  escrow address  : [bold]{d.get('escrow_address')}[/bold]")
    console.print(f"  wallet_id       : {d.get('wallet_id')}")
    linked = d.get("linked_eth_address")
    if linked:
        console.print(
            f"  linked ETH addr : [green]{linked}[/green]"
        )
    else:
        console.print(
            "  linked ETH addr : [yellow]NOT LINKED[/yellow]"
        )
        console.print(
            "\n  → Link an address first:",
            style="dim",
        )
        console.print(
            f"    prsm wallet link-address --eth-address 0x...",
            style="cyan",
        )
    console.print(f"  FTNS contract   : {d.get('ftns_token_contract')}")
    console.print(f"  chain_id        : {d.get('chain_id')}")
    console.print()
    console.print(f"[dim]{d.get('instructions', '')}[/dim]")
    console.print()


@wallet.command("link-address")
@click.option(
    "--eth-address", required=True,
    help="0x-prefixed ETH address to link to your wallet_id",
)
@click.option(
    "--wallet-id", default=None,
    help="Wallet to link (default: this node's identity)",
)
@click.option("--api-url", default=None, help="PRSM daemon API URL")
def wallet_link_address(
    eth_address: str, wallet_id: str, api_url: str,
) -> None:
    """Link an ETH address for bridge deposits.

    Inbound on-chain FTNS transfers FROM this address will credit
    your off-chain wallet balance automatically (sprint 540).
    """
    import httpx
    url = _api_url_from_creds(api_url)
    # Default wallet_id to this node's identity if unset
    if not wallet_id:
        try:
            info = httpx.get(
                f"{url}/wallet/deposit/info", timeout=5.0,
            ).json()
            wallet_id = info.get("wallet_id")
        except Exception:
            console.print(
                "❌ Could not auto-resolve wallet_id; pass "
                "--wallet-id explicitly", style="red",
            )
            raise SystemExit(1)
    try:
        r = httpx.post(
            f"{url}/wallet/deposit/link",
            json={
                "wallet_id": wallet_id,
                "eth_address": eth_address,
            },
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        raise SystemExit(1)
    if r.status_code == 200:
        d = r.json()
        console.print(
            f"✅ Linked [bold]{d['eth_address']}[/bold] → "
            f"wallet_id [bold]{d['wallet_id']}[/bold]",
            style="green",
        )
    elif r.status_code == 422:
        console.print(
            f"❌ Invalid input: {r.json().get('detail', '?')}",
            style="red",
        )
        raise SystemExit(1)
    elif r.status_code == 503:
        console.print(
            f"❌ Service unavailable: {r.json().get('detail', '?')}",
            style="red",
        )
        raise SystemExit(1)
    else:
        console.print(
            f"❌ HTTP {r.status_code}: {r.text[:200]}", style="red",
        )
        raise SystemExit(1)


@wallet.command("withdraw")
@click.option(
    "--amount", required=True, type=float,
    help="FTNS to withdraw from off-chain balance to on-chain",
)
@click.option(
    "--to", "to_eth_address", default=None,
    help="Recipient ETH address (default: this wallet's linked addr)",
)
@click.option(
    "--wallet-id", default=None,
    help="Wallet to debit (default: this node's identity)",
)
@click.option(
    "--yes", "-y", is_flag=True, default=False,
    help="Skip confirmation prompt",
)
@click.option("--api-url", default=None, help="PRSM daemon API URL")
def wallet_withdraw(
    amount: float, to_eth_address: str, wallet_id: str,
    yes: bool, api_url: str,
) -> None:
    """Withdraw off-chain FTNS to on-chain (Pattern A bridge).

    Debits your off-chain wallet, then signs an on-chain ERC-20
    transfer from the operator escrow to your recipient address.
    Atomicity: if the broadcast fails, the daemon credits a refund
    so your off-chain balance stays whole.
    """
    import httpx
    url = _api_url_from_creds(api_url)
    payload = {"amount_ftns": amount}
    if to_eth_address:
        payload["to_eth_address"] = to_eth_address
    if wallet_id:
        payload["wallet_id"] = wallet_id

    if not yes:
        console.print(
            f"\n[bold]About to withdraw {amount:.6f} FTNS[/bold]"
        )
        if to_eth_address:
            console.print(f"  → to {to_eth_address}")
        else:
            console.print(
                f"  → to (linked address — daemon resolves)",
                style="dim",
            )
        console.print(
            "[yellow]This will broadcast a real on-chain TX. "
            "Continue?[/yellow] (y/N): ",
            end="",
        )
        ans = input().strip().lower()
        if ans not in ("y", "yes"):
            console.print("Cancelled.", style="dim")
            raise SystemExit(0)

    try:
        r = httpx.post(
            f"{url}/wallet/withdraw", json=payload, timeout=90.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        raise SystemExit(1)

    if r.status_code == 200:
        d = r.json()
        console.print(
            "✅ Withdraw confirmed on-chain!", style="bold green",
        )
        console.print(f"   tx_hash      : 0x{d.get('tx_hash', '').lstrip('0x')}")
        console.print(f"   block        : {d.get('block_number')}")
        console.print(f"   amount       : {d.get('amount_ftns')} FTNS")
        console.print(f"   to           : {d.get('to_eth_address')}")
        console.print(f"   wallet_id    : {d.get('wallet_id')}")
        console.print(f"   debit_tx_id  : {d.get('debit_tx_id')}")
        return
    detail = ""
    try:
        detail = r.json().get("detail", "")
    except Exception:
        detail = r.text[:300]
    if r.status_code == 402:
        console.print(f"❌ Insufficient off-chain balance:", style="red")
        console.print(f"   {detail}")
    elif r.status_code == 400:
        console.print(f"❌ Invalid request:", style="red")
        console.print(f"   {detail}")
    elif r.status_code == 422:
        console.print(f"❌ Validation error:", style="red")
        console.print(f"   {detail}")
    elif r.status_code == 502:
        console.print(
            f"⚠️  Broadcast failed; off-chain refund issued:",
            style="yellow",
        )
        console.print(f"   {detail}")
    elif r.status_code == 503:
        console.print(f"❌ Service unavailable:", style="red")
        console.print(f"   {detail}")
    else:
        console.print(f"❌ HTTP {r.status_code}: {detail}", style="red")
    raise SystemExit(1)


# ──────────────────────────────────────────────────────────────────
# Sprint 557 — signed withdraw helper for sprint-556 enforcement.
# Operators with requires_user_signature=True on their wallet use
# this command instead of `wallet withdraw` to drive the full
# signed-flow happy path without writing Python.
# ──────────────────────────────────────────────────────────────────


def _build_signed_withdraw_body(
    *,
    amount_ftns: float,
    wallet_id: str,
    to_eth_address: str,
    nonce: int,
    private_key,
    expiry_unix=None,
) -> dict:
    """Build the JSON body for POST /wallet/withdraw with a signed
    EIP-712 payload. Extracted from the CLI command body so tests
    can pin the signing path without Click invocation.

    `expiry_unix` defaults to now + 300 (5-minute window per sprint-
    554 user input).
    """
    import time as _time
    from prsm.economy.withdraw_signature import (
        sign_withdraw_payload,
    )
    if expiry_unix is None:
        expiry_unix = int(_time.time()) + 300
    amount_wei = int(amount_ftns * 1e18)
    payload = {
        "wallet_id": wallet_id,
        "amount_ftns_wei": amount_wei,
        "to_eth_address": to_eth_address,
        "nonce": int(nonce),
        "expiry_unix": int(expiry_unix),
    }
    sig = sign_withdraw_payload(payload, private_key)
    return {
        "amount_ftns": amount_ftns,
        "wallet_id": wallet_id,
        "to_eth_address": to_eth_address,
        "signature": "0x" + sig.hex(),
        "nonce": int(nonce),
        "expiry_unix": int(expiry_unix),
    }


@wallet.command("sign-withdraw")
@click.option(
    "--amount", required=True, type=float,
    help="FTNS to withdraw from off-chain balance to on-chain",
)
@click.option(
    "--to", "to_eth_address", default=None,
    help="Recipient ETH address (default: this wallet's linked addr)",
)
@click.option(
    "--wallet-id", default=None,
    help="Wallet to debit (default: this node's identity)",
)
@click.option(
    "--nonce", default=None, type=int,
    help="Nonce to use (default: fetched from /wallet/deposit/info)",
)
@click.option(
    "--expiry-seconds", default=300, type=int,
    help="Seconds until signature expires (default: 300)",
)
@click.option(
    "--private-key", default=None,
    help=(
        "User's ECDSA private key (default: env PRSM_USER_SIGNING_KEY). "
        "Must recover to the wallet's linked eth_address."
    ),
)
@click.option(
    "--yes", "-y", is_flag=True, default=False,
    help="Skip confirmation prompt",
)
@click.option("--api-url", default=None, help="PRSM daemon API URL")
def wallet_sign_withdraw(
    amount: float, to_eth_address: str, wallet_id: str,
    nonce, expiry_seconds: int, private_key, yes: bool,
    api_url: str,
) -> None:
    """Withdraw off-chain FTNS to on-chain WITH a user EIP-712 signature.

    Required when the wallet has requires_user_signature=True (toggle
    via POST /wallet/require-signature). Sprint-556 enforcement at
    /wallet/withdraw verifies the signature recovers to the wallet's
    linked eth_address (sprint 540). Use the same private key you
    used when linking your eth_address.
    """
    import os
    import time as _time
    import httpx

    pk = private_key or os.environ.get("PRSM_USER_SIGNING_KEY", "")
    pk = pk.strip()
    if not pk:
        console.print(
            "❌ No signing key. Pass --private-key or set "
            "PRSM_USER_SIGNING_KEY in env.",
            style="red",
        )
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    # Fetch wallet_id / linked / nonce from /wallet/deposit/info.
    try:
        r = httpx.get(f"{url}/wallet/deposit/info", timeout=15.0)
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        raise SystemExit(1)
    if r.status_code != 200:
        console.print(
            f"❌ /wallet/deposit/info returned HTTP {r.status_code}: "
            f"{r.text[:200]}",
            style="red",
        )
        raise SystemExit(1)
    info = r.json()
    resolved_wallet = wallet_id or info.get("wallet_id")
    resolved_to = to_eth_address or info.get("linked_eth_address")
    resolved_nonce = (
        nonce if nonce is not None else info.get("next_withdraw_nonce", 0)
    )
    if not resolved_to:
        console.print(
            "❌ No recipient: pass --to or link an eth_address via "
            "`prsm wallet link-address` first.",
            style="red",
        )
        raise SystemExit(1)

    expiry = int(_time.time()) + int(expiry_seconds)
    body = _build_signed_withdraw_body(
        amount_ftns=amount,
        wallet_id=resolved_wallet,
        to_eth_address=resolved_to,
        nonce=int(resolved_nonce),
        private_key=pk,
        expiry_unix=expiry,
    )

    if not yes:
        console.print(
            f"\n[bold]About to sign + withdraw {amount:.6f} FTNS[/bold]"
        )
        console.print(f"  → to        : {resolved_to}")
        console.print(f"  → wallet_id : {resolved_wallet}")
        console.print(f"  → nonce     : {body['nonce']}")
        console.print(
            f"  → expires   : {expiry} ({expiry_seconds}s window)"
        )
        console.print(
            "[yellow]This will broadcast a real on-chain TX. "
            "Continue?[/yellow] (y/N): ",
            end="",
        )
        ans = input().strip().lower()
        if ans not in ("y", "yes"):
            console.print("Cancelled.", style="dim")
            raise SystemExit(0)

    try:
        r = httpx.post(
            f"{url}/wallet/withdraw", json=body, timeout=90.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        raise SystemExit(1)

    if r.status_code == 200:
        d = r.json()
        console.print(
            "✅ Signed withdraw confirmed on-chain!",
            style="bold green",
        )
        console.print(f"   tx_hash        : {d.get('tx_hash')}")
        console.print(f"   block          : {d.get('block_number')}")
        console.print(f"   amount         : {d.get('amount_ftns')} FTNS")
        console.print(f"   to             : {d.get('to_eth_address')}")
        console.print(f"   wallet_id      : {d.get('wallet_id')}")
        console.print(f"   nonce_consumed : {d.get('nonce_consumed')}")
        console.print(f"   debit_tx_id    : {d.get('debit_tx_id')}")
        return

    detail = ""
    try:
        detail = r.json().get("detail", "")
    except Exception:
        detail = r.text[:300]
    if r.status_code == 401:
        console.print(
            "❌ Signature rejected (401):", style="red",
        )
        console.print(f"   {detail}")
        console.print(
            "[dim]Hint: re-run `prsm wallet deposit-info` to read the "
            "current nonce + check that your linked eth_address "
            "matches the key you're signing with.[/dim]"
        )
    elif r.status_code == 402:
        console.print(f"❌ Insufficient off-chain balance: {detail}",
                      style="red")
    elif r.status_code == 502:
        console.print(
            "⚠️  Broadcast failed; off-chain refund issued; "
            "nonce stays consumed (replay safety):",
            style="yellow",
        )
        console.print(f"   {detail}")
    else:
        console.print(
            f"❌ HTTP {r.status_code}: {detail}", style="red",
        )
    raise SystemExit(1)


@wallet.command("claim")
@click.option(
    "--network", "network_name",
    default="testnet",
    type=click.Choice(["mainnet", "testnet"]),
    help="Network to claim from (default: testnet)",
)
@click.confirmation_option(
    prompt="Submit RoyaltyDistributor.claim() transaction?",
    help="Skip confirmation prompt with --yes",
)
def wallet_claim(network_name: str):
    """Withdraw accumulated FTNS royalties via RoyaltyDistributor.claim().

    \b
    The signer's full claimable[address] balance is transferred to their
    address as FTNS, and the mapping entry is zeroed. Reverts on-chain
    if claimable is 0.

    \b
    Required env:
      PRIVATE_KEY or FTNS_WALLET_PRIVATE_KEY — signs the tx
    """
    ctx = _wallet_load_signer(network_name)
    cfg = ctx["network"]

    if not ctx['private_key']:
        console.print("❌ PRIVATE_KEY env var required for claim", style="red")
        raise SystemExit(1)
    if not cfg.royalty_distributor or not cfg.ftns_token:
        console.print(
            f"❌ {network_name}: RoyaltyDistributor or FTNS token "
            f"address missing in prsm/config/networks.py", style="red")
        raise SystemExit(1)

    addr = ctx['address']
    console.print(f"\n[bold]Claiming on {cfg.name}[/bold]")
    console.print(f"From:           {addr}")

    try:
        from prsm.economy.web3.royalty_distributor import (
            RoyaltyDistributorClient,
        )
        client = RoyaltyDistributorClient(
            rpc_url=ctx['rpc_url'],
            distributor_address=cfg.royalty_distributor,
            ftns_token_address=cfg.ftns_token,
            private_key=ctx['private_key'],
        )
        # Pre-flight: check claimable so we surface a clean message
        # instead of an opaque on-chain revert.
        claimable_wei = client.claimable(addr)
        if claimable_wei == 0:
            console.print(
                f"⚠️  claimable[{addr}] is 0 — nothing to claim "
                f"(would revert on-chain). Earn royalties first.",
                style="yellow")
            raise SystemExit(0)
        console.print(f"Claimable:      {claimable_wei / 1e18:,.6f} FTNS")
        console.print()
        console.print("Submitting claim transaction…")
        tx_hash, status = client.claim()
        console.print(f"  Tx hash:       {tx_hash}")
        console.print(f"  Status:        {status}")
        console.print(f"  Explorer:      {cfg.explorer_url}/tx/{tx_hash}")
        console.print(f"\n[bold green]✓ Claim submitted[/bold green]\n")
    except SystemExit:
        raise
    except Exception as exc:
        console.print(
            f"❌ claim failed: {type(exc).__name__}: {exc}", style="red")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# join-testnet command (T5)
#
# One-command onboarding for new testnet users. Generates a fresh burner
# keypair, persists it to ~/.prsm/testnet-deployer.env (chmod 600), and
# prints next-step guidance for funding + running.
# ---------------------------------------------------------------------------


@main.command("join-testnet")
@click.option(
    "--force", is_flag=True, default=False,
    help="Overwrite existing ~/.prsm/testnet-deployer.env",
)
@click.option(
    "--no-print-key", is_flag=True, default=False,
    help="Don't echo the private key to stdout (still saved to file)",
)
def join_testnet(force: bool, no_print_key: bool):
    """Generate a fresh testnet burner wallet + onboarding env file.

    \b
    Creates ~/.prsm/testnet-deployer.env with:
      - PRIVATE_KEY     (freshly generated, 64-char hex)
      - BASE_SEPOLIA_RPC_URL  (default: https://sepolia.base.org)
      - PRSM_NETWORK    (testnet)
      - PRSM_PROVENANCE_REGISTRY_ADDRESS  (from networks.py)
      - FTNS_TOKEN_ADDRESS  (from networks.py)
      - PRSM_ROYALTY_DISTRIBUTOR_ADDRESS  (from networks.py)
      - PRSM_ONCHAIN_PROVENANCE=1

    \b
    The wallet starts with zero balance. Fund it with Base Sepolia ETH
    (Coinbase Developer Platform faucet, etc.) and request testnet-FTNS
    via the project's #testnet-faucet channel before running.

    \b
    Example:
        prsm join-testnet
        # then fund the printed address...
        source ~/.prsm/testnet-deployer.env
        prsm wallet info --network testnet
    """
    import os
    import stat
    from pathlib import Path
    from eth_account import Account
    from prsm.config.networks import get_network_config

    cfg = get_network_config("testnet")

    env_dir = Path.home() / ".prsm"
    env_dir.mkdir(exist_ok=True)
    env_path = env_dir / "testnet-deployer.env"

    if env_path.exists() and not force:
        console.print(
            f"❌ {env_path} already exists. Pass --force to overwrite.",
            style="red")
        console.print(
            "  (or just `source ~/.prsm/testnet-deployer.env` to reuse "
            "your existing burner)", style="dim")
        raise SystemExit(1)

    # Generate fresh burner
    account = Account.create()
    private_key = account.key.hex()
    if not private_key.startswith("0x"):
        private_key = "0x" + private_key
    address = account.address

    # Build env file content
    contract_lines = []
    if cfg.ftns_token:
        contract_lines.append(f"export FTNS_TOKEN_ADDRESS={cfg.ftns_token}")
    if cfg.provenance_registry:
        contract_lines.append(
            f"export PRSM_PROVENANCE_REGISTRY_ADDRESS={cfg.provenance_registry}")
    if cfg.royalty_distributor:
        contract_lines.append(
            f"export PRSM_ROYALTY_DISTRIBUTOR_ADDRESS={cfg.royalty_distributor}")

    env_content = "\n".join([
        "# PRSM testnet burner — generated by `prsm join-testnet`",
        "# DO NOT commit. DO NOT use for mainnet. Testnet has zero monetary value.",
        f"# Address: {address}",
        "",
        f"export PRIVATE_KEY={private_key}",
        f"export FTNS_WALLET_PRIVATE_KEY={private_key}",
        f"export BASE_SEPOLIA_RPC_URL={cfg.rpc_url_default}",
        "export PRSM_NETWORK=testnet",
        "export PRSM_ONCHAIN_PROVENANCE=1",
        *contract_lines,
        "",
    ])

    env_path.write_text(env_content)
    env_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600

    console.print(f"\n[bold green]✓ Testnet wallet created[/bold green]")
    console.print(f"\nAddress:    [bold]{address}[/bold]")
    if not no_print_key:
        console.print(f"Private key: {private_key}", style="dim")
    console.print(f"Env file:   {env_path} (mode 600)")
    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print("  1. Fund the wallet with Base Sepolia ETH (gas):")
    console.print(f"     • Coinbase faucet: https://portal.cdp.coinbase.com/faucet")
    console.print(f"     • Verify at: {cfg.explorer_url}/address/{address}")
    console.print()
    console.print("  2. Request testnet-FTNS (ask in project Discord channel)")
    console.print()
    console.print("  3. Activate the env + check status:")
    console.print(f"     source ~/.prsm/testnet-deployer.env")
    console.print(f"     prsm wallet info --network testnet")
    console.print()
    console.print("  4. Start a node + upload content / earn:")
    console.print(f"     prsm node start --network testnet  (T4 — bootstrap "
                  "wiring still pending)")
    console.print(f"     prsm storage upload <file>")
    console.print()
    if cfg.notes:
        console.print("[dim]Reminders:[/dim]")
        for note in cfg.notes:
            console.print(f"  ℹ {note}", style="dim")
    console.print()


@main.group("bootstrap-server")
def bootstrap_server():
    """Manage / probe a bootstrap server you are running.

    Sprint 390 — operator-trifecta third corner.
    Complements `prsm node bootstrap` (this node's
    registration state) and `prsm node bootstrap-test`
    (probes canonical fleet from your perspective) with
    a probe of your OWN bootstrap droplet's HTTP control
    surface.
    """
    pass


@bootstrap_server.command("status")
@click.option(
    "--host", default="127.0.0.1", show_default=True,
    help="Bootstrap server host (default localhost; pass a "
         "hostname / IP for remote probes).",
)
@click.option(
    "--port", default=8000, type=int, show_default=True,
    help="Bootstrap server API port (BootstrapConfig.api_port).",
)
@click.option(
    "--timeout", default=5.0, type=float, show_default=True,
    help="HTTP timeout in seconds.",
)
@click.option(
    "--detailed", is_flag=True,
    help="Also fetch /health/detailed (sprint 392 "
         "per-subsystem readiness probe) and render the "
         "subsystem table.",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
def bootstrap_server_status(host, port, timeout, detailed, output_format):
    """One-screen ops summary of a running bootstrap server.

    Hits /health and /metrics on the bootstrap server's
    HTTP API and renders a color-coded summary. Defaults
    to localhost:8000 — SSH to your droplet and run this.
    Pass --host for remote probes.
    """
    import asyncio
    import json as _json

    # Late-import so click pulls in nothing at import time
    from prsm.cli_helpers import bootstrap_server_probe as bsp_module

    probe = asyncio.run(
        bsp_module.fetch_server_status(
            host=host, port=port, timeout_seconds=timeout,
            include_subsystems=detailed,
        )
    )

    if output_format == "json":
        console.print(_json.dumps(probe.to_dict(), indent=2))
        sys.exit(0 if probe.status == bsp_module.ProbeStatus.OK else 1)

    # ── Text rendering ────────────────────────────────────
    status_markers = {
        bsp_module.ProbeStatus.OK: "[green]✓ healthy[/green]",
        bsp_module.ProbeStatus.PARTIAL: "[yellow]⚠ partial — metrics unavailable[/yellow]",
        bsp_module.ProbeStatus.CONNECT_FAIL: "[red]✗ connect refused[/red]",
        bsp_module.ProbeStatus.TIMEOUT: "[red]✗ timeout[/red]",
        bsp_module.ProbeStatus.HTTP_ERROR: "[red]✗ http error[/red]",
        bsp_module.ProbeStatus.UNKNOWN: "[red]✗ unknown[/red]",
    }
    marker = status_markers.get(probe.status, "[red]?[/red]")
    console.print(
        f"[bold]PRSM Bootstrap Server Status[/bold] — {marker}"
    )
    console.print(
        f"  target: [cyan]{probe.host}:{probe.port}[/cyan]"
    )
    if probe.error:
        console.print(f"  error:  [red]{probe.error}[/red]")

    if probe.health:
        console.print()
        console.print("[bold]Health[/bold]")
        for k, v in probe.health.items():
            console.print(f"  {k}: {v}")

    if probe.metrics:
        console.print()
        console.print("[bold]Metrics[/bold]")
        # Render flat scalars first, label-dicts last
        flat = {k: v for k, v in probe.metrics.items()
                if not isinstance(v, dict)}
        labeled = {k: v for k, v in probe.metrics.items()
                   if isinstance(v, dict)}
        for k, v in flat.items():
            console.print(f"  {k}: {v}")
        for k, label_dict in labeled.items():
            if not label_dict:
                continue
            console.print(f"  {k}:")
            for label, value in label_dict.items():
                console.print(f"    {label}: {value}")

    if probe.health_detailed:
        console.print()
        agg = probe.health_detailed.get("status", "?")
        agg_color = {
            "healthy": "green",
            "degraded": "yellow",
            "unhealthy": "red",
        }.get(agg, "white")
        console.print(
            f"[bold]Subsystems[/bold] — aggregate: "
            f"[{agg_color}]{agg}[/{agg_color}]"
        )
        for sub_name, sub_data in (
            probe.health_detailed.get("subsystems") or {}
        ).items():
            sub_status = sub_data.get("status", "?")
            sub_color = {
                "healthy": "green",
                "degraded": "yellow",
                "stale": "red",
            }.get(sub_status, "white")
            age = sub_data.get("last_heartbeat_age_seconds")
            age_str = (
                f"{age:.0f}s" if isinstance(age, (int, float))
                else "—"
            )
            console.print(
                f"  {sub_name}: "
                f"[{sub_color}]{sub_status}[/{sub_color}] "
                f"(age {age_str})"
            )

    sys.exit(0 if probe.status == bsp_module.ProbeStatus.OK else 1)


@node.command("fiat-readiness")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
def node_fiat_readiness(output_format):
    """Probe Phase 5 fiat-surface activation readiness.

    Sprint 422 — wraps sprint-285's
    `check_fiat_surface_health()` for operator CLI use.
    Run this BEFORE attempting Phase 5 activation (see
    `docs/operations/phase-5-fiat-surface-activation-
    runbook.md`) to verify your env is ready.

    Exit code: 0 = OK or WARN-only findings; non-zero =
    at least one ERROR finding (activation will fail).

    Default `text` format renders a color-coded findings
    table with remediation hints. `json` format for ops
    automation (parseable on both success + failure).
    """
    import json as _json
    import os

    from prsm.economy.web3.fiat_surface_health import (
        check_fiat_surface_health,
    )

    findings = check_fiat_surface_health(os.environ)

    has_error = any(
        getattr(f.severity, "value", str(f.severity)).lower()
        == "error"
        for f in findings
    )
    has_warn = any(
        getattr(f.severity, "value", str(f.severity)).lower()
        == "warn"
        for f in findings
    )
    if has_error:
        overall = "error"
    elif has_warn:
        overall = "warn"
    else:
        overall = "ok"

    if output_format == "json":
        payload = {
            "overall_status": overall,
            "findings": [
                {
                    "severity": getattr(
                        f.severity, "value", str(f.severity),
                    ),
                    "cause": f.cause,
                    "remediation": f.remediation,
                }
                for f in findings
            ],
        }
        # Plain stdout — Rich's console.print injects ANSI
        # control chars that break JSON parsers downstream.
        click.echo(_json.dumps(payload, indent=2))
        sys.exit(0 if not has_error else 1)

    # Text rendering
    if overall == "ok":
        console.print(
            "[green]✓ Phase 5 fiat surface ready — OK[/green] "
            "[dim](no findings)[/dim]"
        )
        sys.exit(0)

    marker_map = {
        "error": "[red]✗ ERROR[/red]",
        "warn": "[yellow]⚠ WARN[/yellow]",
    }
    overall_marker = marker_map.get(
        overall, "[white]?[/white]"
    )
    console.print(
        f"[bold]Phase 5 fiat-readiness[/bold] — "
        f"{overall_marker}"
    )
    console.print()
    for f in findings:
        sev = getattr(f.severity, "value", str(f.severity)).lower()
        marker = marker_map.get(sev, "[white]?[/white]")
        console.print(f"{marker} [bold]{f.cause}[/bold]")
        console.print(f"  [dim]{f.remediation}[/dim]")
        console.print()

    sys.exit(0 if not has_error else 1)


# ──────────────────────────────────────────────────────────────
# Sprint 638 — `prsm node models list` discovers what's registered.
# Pre-638 operators using `prsm node infer` had to know a model_id
# ahead of time; nothing surfaced the local registry contents from
# the CLI. This command reads PRSM_MODEL_REGISTRY_ROOT, enumerates
# every model + dumps manifest metadata (publisher, shard count,
# published_at, layer ranges). Local-only for now — peer-advertised
# models will come in a follow-on sprint when there's a protocol-
# level "what do you serve?" message.
# ──────────────────────────────────────────────────────────────


@node.command("models")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
@click.option(
    "--registry-root", default=None,
    help="Override PRSM_MODEL_REGISTRY_ROOT for one-off probes.",
)
def node_models_list_cli(output_format: str, registry_root: Optional[str]):
    """List models in the local FilesystemModelRegistry.

    Sprint 638 — closes the discovery gap from sprint 633's
    `prsm node infer`. Operators previously had to know a model_id
    ahead of time; this command surfaces every registered model
    with its publisher identity, shard count, and per-shard layer
    coverage so operators can pick a target for inference.

    Default registry root is read from PRSM_MODEL_REGISTRY_ROOT
    (the same path the daemon uses at runtime); override with
    --registry-root for one-off probes against alternate roots.

    Exit code: 0 when at least one model is present; 1 when the
    registry is empty or unreachable so scripts can branch on it.
    """
    import json as _json
    import os as _os
    import sys as _sys

    from prsm.compute.model_registry.registry import (
        FilesystemModelRegistry,
    )

    root = registry_root or _os.environ.get("PRSM_MODEL_REGISTRY_ROOT", "")
    if not root:
        if output_format == "json":
            click.echo(_json.dumps({
                "error": "registry_root_unset",
                "models": [],
            }, indent=2))
        else:
            console.print(
                "[red]✗ Registry root not configured[/red]\n"
                "[dim]Set PRSM_MODEL_REGISTRY_ROOT or pass "
                "--registry-root to point at a FilesystemModelRegistry "
                "directory.[/dim]"
            )
        _sys.exit(1)

    try:
        registry = FilesystemModelRegistry(root=root)
    except Exception as exc:  # noqa: BLE001
        if output_format == "json":
            click.echo(_json.dumps({
                "error": f"{type(exc).__name__}: {exc}",
                "registry_root": root,
                "models": [],
            }, indent=2))
        else:
            console.print(
                f"[red]✗ Failed to open registry at {root!r}[/red]: "
                f"{type(exc).__name__}: {exc}"
            )
        _sys.exit(1)

    model_ids = registry.list_models()
    summaries = []
    for model_id in model_ids:
        try:
            manifest = registry.get_manifest(model_id)
        except Exception as exc:  # noqa: BLE001
            summaries.append({
                "model_id": model_id,
                "error": f"manifest load failed: "
                         f"{type(exc).__name__}: {exc}",
            })
            continue
        layer_ranges = []
        for s in manifest.shards:
            layer_ranges.append(list(s.layer_range))
        summaries.append({
            "model_id": manifest.model_id,
            "model_name": manifest.model_name,
            "publisher_node_id": manifest.publisher_node_id,
            "total_shards": manifest.total_shards,
            "published_at": manifest.published_at,
            "schema_version": manifest.schema_version,
            "layer_ranges": layer_ranges,
        })

    if output_format == "json":
        click.echo(_json.dumps({
            "registry_root": root,
            "total": len(summaries),
            "models": summaries,
        }, indent=2))
        _sys.exit(0 if summaries else 1)

    if not summaries:
        console.print(
            f"[yellow]No models registered[/yellow] at "
            f"[cyan]{root}[/cyan]\n"
            f"[dim]Register a model via the registry API or "
            f"copy a published-manifest directory into this root.[/dim]"
        )
        _sys.exit(1)

    console.print(
        f"[dim]Registry root: {root}[/dim]\n"
        f"[bold]{len(summaries)} model(s) registered:[/bold]"
    )
    for s in summaries:
        if "error" in s:
            console.print(
                f"  [red]✗ {s['model_id']}[/red]: {s['error']}"
            )
            continue
        # Defense: layer_ranges may all be (0,0) for legacy
        # manifests pre-sprint 627; render compactly when sentinel.
        non_sentinel = [r for r in s["layer_ranges"] if r != [0, 0]]
        layer_summary = (
            ", ".join(f"[{r[0]}, {r[1]})" for r in non_sentinel)
            if non_sentinel
            else "(layer ranges unset — pre-sprint-627 manifest)"
        )
        console.print(
            f"  [green]●[/green] [bold]{s['model_id']}[/bold] "
            f"([dim]name={s['model_name']!r}[/dim])\n"
            f"    publisher: [cyan]{s['publisher_node_id'][:16]}...[/cyan]\n"
            f"    shards: {s['total_shards']}  "
            f"layers: {layer_summary}\n"
            f"    schema_v{s['schema_version']}, "
            f"published_at {s['published_at']:.0f}"
        )
    _sys.exit(0)


# ──────────────────────────────────────────────────────────────
# Sprint 633 — `prsm node infer` operator CLI for the live P2P
# inference path (sprint 628's multi-token GPT-2 demo wrapped as
# a first-class operator command). Closes the dogfood gap where
# the headline demo required hand-running a hardcoded Python
# script — now any operator with a running fleet + registered
# model can drive the full mainnet-anchor-verified inference
# loop directly from the CLI.
# ──────────────────────────────────────────────────────────────


@node.command("infer")
@click.option(
    "--prompt", required=True, type=str,
    help="Initial prompt text",
)
@click.option(
    "--model", default="gpt2", show_default=True,
    help="Model id (must be registered + available on at least one peer)",
)
@click.option(
    "--max-tokens", "-n", type=int, default=10, show_default=True,
    help="Number of tokens to generate",
)
@click.option(
    "--stage-peer-id", default=None,
    help="Peer ID of the layer-stage server. If omitted, uses the "
    "first connected peer.",
)
@click.option(
    "--api", "api_url", default="http://127.0.0.1:8000", show_default=True,
    help="Local PRSM daemon API URL",
)
@click.option(
    "--timeout", type=float, default=120.0, show_default=True,
    help="Per-token request timeout (seconds)",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
@click.option(
    "--save-receipts", "save_receipts_path", type=click.Path(),
    default=None,
    help="Append one JSON-line per token to this file. Each line "
    "carries the stage signature, stage_node_id, request_id, and "
    "activation hash — enough for offline verification against the "
    "PublisherKeyAnchor.",
)
@click.option(
    "--temperature", type=float, default=None,
    help="Sprint 639 — sampling temperature. Omit (default) for "
    "greedy argmax (deterministic, audit-chain-friendly). Values "
    "between 0 and ~2 produce increasingly random output. The "
    "receipt's `sampling_mode` field records the choice so the "
    "downstream `verify-receipts --check-chain` knows to skip "
    "the argmax-vs-next_token_id invariant for non-greedy runs.",
)
@click.option(
    "--top-k", type=int, default=None,
    help="Sprint 639 — restrict sampling to the top-K tokens by "
    "logit magnitude. Pairs with --temperature; combining the two "
    "is the typical 'creative but not chaotic' setting. Ignored "
    "when --temperature is unset.",
)
@click.option(
    "--seed", type=int, default=None,
    help="Sprint 639 — RNG seed for sampling. Recorded in the "
    "receipt so a verifier with the same seed can re-derive the "
    "sample deterministically (a partial mitigation for the "
    "audit-chain weakening that comes with non-greedy sampling).",
)
def node_infer_cli(
    prompt: str,
    model: str,
    max_tokens: int,
    stage_peer_id: Optional[str],
    api_url: str,
    timeout: float,
    output_format: str,
    save_receipts_path: Optional[str],
    temperature: Optional[float],
    top_k: Optional[int],
    seed: Optional[int],
):
    """Generate tokens via the live PRSM P2P inference path.

    Wraps the sprint-628 demo: prompt → tokenize+embed locally →
    sign + ship each forward to a stage peer → receive logits →
    argmax → append → repeat. Every dispatch verifies the
    settler's pubkey against the live PublisherKeyAnchor on Base
    mainnet (sprint 621 deploy).

    Required prerequisites (this command does NOT bootstrap them):
      • Local daemon running with PRSM_PARALLAX_TRUST_STACK_KIND=production
      • At least one peer connected that serves the requested model
        via LayerStageServer (PRSM_PARALLAX_LAYER_SLICE_RUNNER_KIND=huggingface)
      • The model registered on both sides (filesystem registry)

    Example:
        prsm node infer --prompt "The capital of France is" -n 10
    """
    import base64
    import json as _json
    import sys as _sys
    import time as _time

    import httpx as _httpx
    import numpy as _np

    from prsm.compute.chain_rpc.protocol import (
        ContentTier, HandoffToken, PrivacyLevel, RunLayerSliceRequest,
        encode_message, parse_message,
    )
    from prsm.node.config import NodeConfig
    from prsm.node.identity import load_node_identity

    # ── Resolve stage peer ──
    try:
        peers_resp = _httpx.get(f"{api_url}/peers", timeout=5.0)
        peers_resp.raise_for_status()
    except Exception as exc:
        console.print(
            f"[red]✗ Failed to reach local daemon at {api_url}[/red]: "
            f"{type(exc).__name__}: {exc}\n"
            f"[dim]Start the daemon first: prsm node start[/dim]"
        )
        _sys.exit(1)
    peers_data = peers_resp.json()
    connected = peers_data.get("connected", [])
    if stage_peer_id is None:
        if not connected:
            console.print(
                "[red]✗ No connected peers[/red]. Need at least one "
                "peer to serve the inference stage.\n"
                "[dim]Check fleet symmetry with `prsm node info` or "
                "wait for bootstrap-mediated discovery.[/dim]"
            )
            _sys.exit(1)
        stage_peer_id = connected[0].get("peer_id")
        if not stage_peer_id:
            console.print("[red]✗ First connected peer has no peer_id[/red]")
            _sys.exit(1)

    # ── Load settler identity ──
    cfg = NodeConfig.load()
    settler = load_node_identity(cfg.identity_path)

    # ── Load tokenizer + embedding layer ──
    # Lazy import — operators without HF installed should still see a
    # clean error rather than a top-level ModuleNotFoundError at CLI
    # boot.
    try:
        import torch as _torch
        from transformers import (
            AutoModelForCausalLM as _AutoModel,
            AutoTokenizer as _AutoTok,
        )
    except ImportError as exc:
        console.print(
            f"[red]✗ HuggingFace deps missing[/red]: {exc}\n"
            f"[dim]Install with: pip install transformers torch[/dim]"
        )
        _sys.exit(1)

    if output_format == "text":
        console.print(
            f"[dim]Loading {model} for tokenize + embed...[/dim]"
        )
    try:
        tok = _AutoTok.from_pretrained(model)
        hf_model = _AutoModel.from_pretrained(
            model, torch_dtype=_torch.float32,
        ).eval()
    except Exception as exc:
        console.print(
            f"[red]✗ Failed to load HF model {model!r}[/red]: "
            f"{type(exc).__name__}: {exc}"
        )
        _sys.exit(1)

    # ── Receipt sink (sprint 634) ──
    # When --save-receipts is set, open the file in append-mode at
    # the top of the loop. Each per-token write is a single JSON
    # line so the file is friendly to `jq` + grep-based post-
    # processing. Append rather than truncate so multi-run audit
    # trails accumulate naturally.
    import hashlib as _hashlib
    import os as _os_for_dir
    from pathlib import Path as _Path
    receipts_fh = None
    if save_receipts_path:
        save_p = _Path(save_receipts_path)
        try:
            save_p.parent.mkdir(parents=True, exist_ok=True)
            receipts_fh = save_p.open("a", encoding="utf-8")
        except OSError as exc:
            console.print(
                f"[red]✗ Cannot open receipts file "
                f"{save_receipts_path!r}[/red]: {exc}"
            )
            _sys.exit(1)
        if output_format == "text":
            console.print(
                f"[dim]Recording per-token receipts to "
                f"{save_receipts_path}[/dim]"
            )

    # ── Generation loop ──
    text = prompt
    per_token_records = []
    overall_t0 = _time.time()
    for step in range(max_tokens):
        step_t0 = _time.time()
        input_ids = tok.encode(text, return_tensors="pt")
        with _torch.no_grad():
            te = hf_model.transformer.wte(input_ids)
            pe = hf_model.transformer.wpe(
                _torch.arange(input_ids.shape[-1]).unsqueeze(0),
            )
            activation = (te + pe).numpy()

        request_id = f"prsm-cli-infer-step{step}-{int(_time.time())}"
        deadline = _time.time() + timeout
        ho_token = HandoffToken.sign(
            identity=settler, request_id=request_id,
            chain_stage_index=0, chain_total_stages=1,
            deadline_unix=deadline,
        )
        # Layer range = (0, num_layers) so the stage runs the full
        # stack. For gpt2 num_layers=12; the catalog drives this.
        # Defensively read from hf_model.config when available so
        # CLI works against any model on the registry.
        n_layers = getattr(
            getattr(hf_model, "config", None), "num_hidden_layers",
            getattr(hf_model.config, "n_layer", 12),
        )
        request = RunLayerSliceRequest(
            request_id=request_id, model_id=model,
            layer_range=(0, n_layers),
            privacy_tier=PrivacyLevel.NONE,
            content_tier=ContentTier.A,
            activation_blob=activation.tobytes(),
            activation_shape=tuple(activation.shape),
            activation_dtype=str(activation.dtype),
            upstream_token=ho_token, deadline_unix=deadline,
        )
        req_bytes = encode_message(request)

        try:
            r = _httpx.post(
                f"{api_url}/admin/chain-exec-ping",
                json={
                    "peer_id": stage_peer_id,
                    "payload_b64": base64.b64encode(req_bytes).decode("ascii"),
                    "timeout": timeout - 5.0,
                },
                timeout=timeout,
            )
        except Exception as exc:
            console.print(
                f"[red]✗ step {step}: chain-exec-ping failed[/red]: "
                f"{type(exc).__name__}: {exc}"
            )
            _sys.exit(1)
        if r.status_code != 200:
            console.print(
                f"[red]✗ step {step}: HTTP {r.status_code}[/red]: "
                f"{r.text[:300]}"
            )
            _sys.exit(1)
        resp_bytes = base64.b64decode(r.json()["response_b64"])
        try:
            resp = parse_message(resp_bytes)
        except Exception as exc:
            console.print(
                f"[red]✗ step {step}: parse_message failed[/red]: "
                f"{type(exc).__name__}: {exc}"
            )
            _sys.exit(1)
        if not hasattr(resp, "activation_blob"):
            msg = getattr(resp, "message", "?")
            code = getattr(resp, "code", "?")
            console.print(
                f"[red]✗ step {step}: StageError[/red] code={code} "
                f"message={msg}"
            )
            _sys.exit(1)
        logits = _np.frombuffer(
            resp.activation_blob, dtype=resp.activation_dtype,
        ).reshape(resp.activation_shape)
        # Sprint 639 — sampling. The `sampling_mode` string is also
        # written into the receipt so verify-receipts --check-chain
        # knows whether to apply the argmax↔next_token_id invariant
        # (C5). Greedy mode commits the full chain-of-custody chain;
        # temperature/top-k weakens it to "stage signed over these
        # logits + operator recorded a sample drawn from this
        # distribution" — verifier with the seed can re-derive.
        # Sprint 641 — single sampling helper used by both CLI infer
        # and verify-receipts replay. Drift-free by construction.
        from prsm.cli_modules.sampling import (
            sample_token_from_logits, format_sampling_mode,
        )
        last_logits = logits[0, -1, :]
        next_id = sample_token_from_logits(
            last_logits,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
            step=step,
        )
        sampling_mode = format_sampling_mode(
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        )
        next_token = tok.decode([next_id])
        text += next_token
        step_dt = _time.time() - step_t0

        # Sprint 634 — record signed receipt for this token. The
        # stage_signature is over the canonical signing payload
        # (chain_rpc.protocol.RunLayerSliceResponse.signing_payload);
        # any party with the stage_node_id's pubkey from the
        # PublisherKeyAnchor can rebuild that payload + verify
        # offline. We also hash the activation_blob (logits) so
        # the receipt commits to the exact output bytes the
        # token was sampled from.
        if receipts_fh is not None:
            logits_sha256 = _hashlib.sha256(
                resp.activation_blob,
            ).hexdigest()
            # Sprint 635: include the bytes the stage_signature was
            # COMPUTED over, so receipts are self-verifying offline.
            # Without these, the activation_sha256 is observation-
            # only (proves the operator saw these specific bytes
            # come back, but can't independently reconstruct the
            # signing payload). With them, anyone can rebuild the
            # canonical signing payload + verify Ed25519 against
            # the stage_node_id's pubkey (from PublisherKeyAnchor).
            #
            # Cost: gpt2 logits are ~1.3MB per token → ~13MB per
            # 10-token run base64-encoded. Operators with size
            # concerns can post-process (gzip the file → ~50%).
            receipt_record = {
                "step": step,
                "wall_unix": _time.time(),
                "request_id": resp.request_id,
                "settler_node_id": settler.node_id,
                "stage_node_id": resp.stage_node_id,
                "stage_signature_b64": resp.stage_signature_b64,
                "model_id": model,
                "layer_range": [0, int(n_layers)],
                "activation_shape": list(resp.activation_shape),
                "activation_dtype": resp.activation_dtype,
                "activation_sha256": logits_sha256,
                "activation_blob_b64": base64.b64encode(
                    bytes(resp.activation_blob),
                ).decode("ascii"),
                "tee_attestation_b64": base64.b64encode(
                    bytes(resp.tee_attestation),
                ).decode("ascii"),
                "duration_seconds": resp.duration_seconds,
                "epsilon_spent": resp.epsilon_spent,
                "tee_type": getattr(
                    resp.tee_type, "value",
                    str(resp.tee_type),
                ),
                "protocol_version": resp.protocol_version,
                "next_token_id": next_id,
                "next_token_text": next_token,
                # Sprint 639 — record sampling mode so the
                # verify-receipts --check-chain (sprint 637) can
                # decide whether the C5 argmax-vs-next_token_id
                # invariant applies. Greedy mode lets C5 fire;
                # non-greedy modes skip C5 (with the seed in the
                # mode string, a verifier could re-derive but that's
                # out of sprint 639's scope).
                "sampling_mode": sampling_mode,
            }
            try:
                receipts_fh.write(_json.dumps(receipt_record) + "\n")
                receipts_fh.flush()
            except OSError as exc:
                # Don't kill the run for an audit-write failure; just
                # surface + carry on. Operator can decide on retry.
                console.print(
                    f"[yellow]⚠ step {step}: receipt write failed: "
                    f"{exc}[/yellow]"
                )
        per_token_records.append({
            "step": step, "token_id": next_id,
            "token_text": next_token, "elapsed_s": step_dt,
            "seq_len_before": input_ids.shape[-1],
        })
        if output_format == "text":
            console.print(
                f"  [dim][step {step:2d}][/dim] +token "
                f"{next_id:6d} [cyan]{next_token!r:>14s}[/cyan]  "
                f"([dim]{step_dt:.1f}s[/dim])"
            )

    overall_dt = _time.time() - overall_t0
    # Sprint 634 — close receipts sink cleanly. Failure here is
    # already-fsync'd so it doesn't lose tokens; just log.
    if receipts_fh is not None:
        try:
            receipts_fh.close()
        except OSError as exc:
            console.print(
                f"[yellow]⚠ receipts file close failed: {exc}[/yellow]"
            )
    if output_format == "json":
        click.echo(_json.dumps({
            "prompt": prompt,
            "model": model,
            "stage_peer_id": stage_peer_id,
            "max_tokens": max_tokens,
            "generated_text": text,
            "elapsed_s": overall_dt,
            "per_token": per_token_records,
            "receipts_path": save_receipts_path,
        }, indent=2))
        return
    console.print()
    console.print(
        f"[green]🎯 Generated[/green] "
        f"({max_tokens} tokens in {overall_dt:.1f}s):"
    )
    console.print(f"  [bold]{text!r}[/bold]")
    if save_receipts_path:
        console.print(
            f"[dim]{max_tokens} signed receipt(s) saved to "
            f"{save_receipts_path}[/dim]"
        )


@node.command("verify-receipts")
@click.argument(
    "receipts_path", type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
@click.option(
    "--check-chain", is_flag=True, default=False,
    help="Sprint 637 — also assert chain-of-custody invariants "
    "across receipts: settler/model consistency, request_id "
    "uniqueness, wall_unix monotonicity, next_token_id matches "
    "argmax of activation_blob. Defends against post-generation "
    "tampering that doesn't invalidate per-token signatures.",
)
def node_verify_receipts_cli(
    receipts_path: str, output_format: str, check_chain: bool,
):
    """Verify per-token signed receipts written by `prsm node infer
    --save-receipts`.

    Sprint 635 — closes the Vision §7 truth-surfacing loop. Each
    receipt's stage_signature_b64 was produced by the stage node
    over the canonical signing payload (chain_rpc.protocol
    RunLayerSliceResponse.signing_payload). This command:

      1. Reads each JSON line from `receipts_path`.
      2. Looks up the stage_node_id's pubkey via the live
         PublisherKeyAnchor (using `_build_anchor_or_none()`).
      3. Reconstructs RunLayerSliceResponse from the receipt's
         fields (activation_blob_b64, tee_attestation_b64, etc.).
      4. Calls verify_with_anchor with expected_stage_node_id =
         the receipt's stage_node_id.
      5. Reports pass/fail per line + an aggregate summary.

    Exit code: 0 if every receipt verified, non-zero otherwise.

    Requires sprint 635+ receipts (activation_blob_b64 field).
    Pre-sprint-635 receipts (sha256 only) will be marked
    UNVERIFIABLE — the receipt proved observation but the
    activation bytes weren't persisted for signature reconstruction.
    """
    import json as _json
    import sys as _sys

    from prsm.cli_modules.receipt_verify import verify_receipts_file
    from prsm.node.inference_wiring import _build_anchor_or_none

    anchor = _build_anchor_or_none()
    if anchor is None:
        console.print(
            "[red]✗ No anchor available[/red]: cannot resolve "
            "stage pubkeys for signature verification.\n"
            "[dim]Set PRSM_NETWORK=mainnet (sprint 629 default "
            "honors networks.py) or pass an explicit "
            "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS.[/dim]"
        )
        _sys.exit(2)

    # Sprint 636 — verification core lives in cli_modules.receipt_verify
    # for direct unit testing. CLI is a thin renderer over the results.
    results = verify_receipts_file(
        receipts_path, anchor=anchor, check_chain=check_chain,
    )

    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_total = sum(1 for r in results if r["status"] != "CHAIN_ONLY")
    # Sprint 637 — chain_findings is attached to the last result;
    # extract for rendering + exit-code computation.
    chain_findings = []
    for r in results:
        if "chain_findings" in r:
            chain_findings = r["chain_findings"]
            break
    chain_ok = len(chain_findings) == 0
    overall_ok = (n_ok == n_total) and (chain_ok if check_chain else True)

    if output_format == "json":
        click.echo(_json.dumps({
            "receipts_path": receipts_path,
            "total": n_total,
            "verified": n_ok,
            "check_chain": check_chain,
            "chain_ok": chain_ok if check_chain else None,
            "chain_findings": chain_findings if check_chain else None,
            "results": results,
        }, indent=2))
        _sys.exit(0 if overall_ok else 1)

    for r in results:
        if r.get("status") == "CHAIN_ONLY":
            continue
        if r["status"] == "OK":
            console.print(
                f"  [green]✓[/green] line {r['line']:3d}: "
                f"[dim]{r['stage_node_id'][:16]}...[/dim] "
                f"token=[cyan]{r.get('next_token_text', '?')!r}[/cyan]"
            )
        else:
            console.print(
                f"  [red]✗[/red] line {r['line']:3d}: "
                f"[bold]{r['status']}[/bold] — "
                f"{r.get('reason', '?')}"
            )
    console.print()
    if n_ok == n_total:
        console.print(
            f"[green]🎯 {n_ok}/{n_total} receipts verified[/green] "
            f"against the live anchor."
        )
    else:
        console.print(
            f"[red]✗ {n_ok}/{n_total} receipts verified[/red]; "
            f"{n_total - n_ok} failed."
        )
    # Sprint 637 — render chain findings
    if check_chain:
        if chain_ok:
            console.print(
                "[green]🔗 chain-of-custody invariants OK[/green] — "
                "settler/model consistency, request_id uniqueness, "
                "wall_unix monotonicity, argmax↔next_token_id match"
            )
        else:
            console.print(
                f"[red]✗ {len(chain_findings)} chain-of-custody "
                f"invariant(s) violated:[/red]"
            )
            for f in chain_findings:
                console.print(
                    f"  [red]●[/red] [bold]{f['kind']}[/bold]: "
                    f"{f['message']} "
                    f"[dim](line(s): {f['line_indices']})[/dim]"
                )
    _sys.exit(0 if overall_ok else 1)


# ──────────────────────────────────────────────────────────────
# Sprint 434 — `prsm node incident ...` CLI trifecta gap
#
# The /admin/incident/* REST surface (sprint <pre-roadmap>) +
# `prsm_incident` MCP tool exist; the CLI lane was the gap per
# PRSM_Testing.md §13 "Operator-trifecta gaps". This block adds
# the read-only triage commands an operator needs at incident
# time: list active incidents, view one in detail, print the
# canonical playbook. The mutating commands (open / advance /
# log-event) need more thought about input-parameter UX and
# stay deferred — operators can hit those via the REST surface
# or `prsm_incident` MCP.
# ──────────────────────────────────────────────────────────────


@node.group("incident", invoke_without_command=False)
def node_incident_group():
    """Incident-response triage commands (read-only).

    Maps to /admin/incident/* REST endpoints. For the
    mutating actions (open / advance / log-event), use
    `prsm_incident` MCP or the REST surface directly.
    """


@node_incident_group.command("list")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
@click.option(
    "--severity", default=None,
    help=(
        "Filter by severity: s0 (catastrophic), s1 "
        "(critical), s2 (high), s3 (low)."
    ),
)
@click.option(
    "--phase", default=None,
    help=(
        "Filter by phase: detected, triaged, contained, "
        "mitigated, postmortem."
    ),
)
def node_incident_list(api_port, output_format, severity, phase):
    """List active incidents on this node.

    Wraps GET /admin/incident. Color-codes by severity in
    text mode; emits JSON-stable payload for ops automation.
    """
    import json as _json
    import urllib.parse
    import urllib.request

    params = {}
    if severity:
        params["severity"] = severity
    if phase:
        params["phase"] = phase
    qs = ("?" + urllib.parse.urlencode(params)) if params else ""
    url = f"http://127.0.0.1:{api_port}/admin/incident{qs}"

    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            payload = _json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        click.echo(
            f"Failed to fetch incidents: {exc}", err=True,
        )
        sys.exit(2)

    records = payload.get("records", [])
    if output_format == "json":
        click.echo(_json.dumps(payload, indent=2))
        return

    count = payload.get("count", len(records))
    if count == 0:
        console.print(
            "[green]✓ No active incidents.[/green]"
        )
        return

    sev_color = {
        "s0": "red",          # catastrophic
        "s1": "red",          # critical
        "s2": "yellow",       # high
        "s3": "cyan",         # low
    }
    console.print(
        f"[bold]Active incidents ({count}):[/bold]\n"
    )
    for r in records:
        sev = (r.get("severity") or "").lower()
        color = sev_color.get(sev, "white")
        console.print(
            f"[{color}]{sev.upper():<8}[/{color}] "
            f"[bold]{r.get('incident_id', '?')}[/bold] "
            f"phase=[dim]{r.get('phase', '?')}[/dim] "
            f"kind=[dim]{r.get('kind', '?')}[/dim]"
        )
        title = r.get("title") or r.get("description")
        if title:
            console.print(f"  [dim]{title}[/dim]")


@node_incident_group.command("details")
@click.argument("incident_id")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
def node_incident_details(incident_id, api_port, output_format):
    """Show full detail for a single incident.

    Wraps GET /admin/incident/{incident_id}.
    """
    import json as _json
    import urllib.error
    import urllib.request

    url = (
        f"http://127.0.0.1:{api_port}/admin/incident/"
        f"{incident_id}"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            payload = _json.loads(r.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            click.echo(
                f"Incident {incident_id!r} not found.", err=True,
            )
            sys.exit(1)
        click.echo(
            f"Failed: HTTP {exc.code}: {exc.read().decode()[:200]}",
            err=True,
        )
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Failed to fetch: {exc}", err=True)
        sys.exit(2)

    if output_format == "json":
        click.echo(_json.dumps(payload, indent=2))
        return

    console.print(
        f"[bold]Incident {payload.get('incident_id', '?')}[/bold]"
    )
    console.print(
        f"  severity:  {payload.get('severity', '?')}"
    )
    console.print(
        f"  phase:     {payload.get('phase', '?')}"
    )
    console.print(
        f"  kind:      {payload.get('kind', '?')}"
    )
    if payload.get("title"):
        console.print(f"  title:     {payload['title']}")
    if payload.get("description"):
        console.print(
            f"  desc:      {payload['description']}"
        )
    events = payload.get("events") or []
    if events:
        console.print(f"\n  Events ({len(events)}):")
        for e in events:
            console.print(
                f"    - [dim]{e.get('timestamp', '?')}[/dim] "
                f"{e.get('event_type', '?')}: "
                f"{e.get('description', '')}"
            )


@node_incident_group.command("playbook")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
@click.option(
    "--severity", default=None,
    help="Filter playbook to one severity level.",
)
def node_incident_playbook(api_port, output_format, severity):
    """Show the canonical incident-response playbook.

    Wraps GET /admin/incident/playbook. Per Vision §14:
    the playbook is published BEFORE any incident — this
    command makes it readable from the operator terminal.
    """
    import json as _json
    import urllib.request

    url = (
        f"http://127.0.0.1:{api_port}/admin/incident/playbook"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            payload = _json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Failed to fetch: {exc}", err=True)
        sys.exit(2)

    if output_format == "json":
        click.echo(_json.dumps(payload, indent=2))
        return

    decision_tree = payload.get("decision_tree", [])
    if severity:
        decision_tree = [
            d for d in decision_tree
            if d.get("severity", "").lower() == severity.lower()
        ]
    console.print("[bold]Incident Response Playbook[/bold]\n")
    console.print(
        f"[bold]Decision tree ({len(decision_tree)} entries):"
        "[/bold]"
    )
    for d in decision_tree:
        sev = d.get("severity", "?")
        phase = d.get("phase", "?")
        recs = d.get("recommendations", [])
        console.print(
            f"\n  [bold]{sev.upper()}[/bold] / [dim]{phase}"
            f"[/dim]"
        )
        for rec in recs:
            console.print(f"    • {rec}")


# ──────────────────────────────────────────────────────────────
# Sprint 435 — `prsm node insurance ...` CLI trifecta gap
#
# Two endpoints on the REST surface:
#   GET  /admin/insurance-fund/status — current fund state
#   POST /admin/insurance-fund/compose-recovery — produce a
#        multi-sig-uploadable transfer tx payload
#
# Both safe in CLI form: status is read-only; compose-recovery
# only PRODUCES the tx bytes (does NOT execute — Foundation Safe
# holds the actual transfer privilege per Vision §14). Operator
# pipes the JSON output into the multi-sig signing tool of choice.
# ──────────────────────────────────────────────────────────────


@node.group("insurance", invoke_without_command=False)
def node_insurance_group():
    """Insurance-fund triage + recovery-tx composition.

    Maps to /admin/insurance-fund/* REST endpoints.
    """


@node_insurance_group.command("status")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
def node_insurance_status(api_port, output_format):
    """Show current insurance-fund status (balance, recent
    recoveries, etc.).

    Wraps GET /admin/insurance-fund/status. 503 if the
    tracker isn't wired on this node.
    """
    import json as _json
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{api_port}/admin/insurance-fund/status"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            payload = _json.loads(r.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            click.echo(
                "Insurance fund tracker not initialized on "
                "this node.",
                err=True,
            )
            sys.exit(1)
        click.echo(
            f"Failed: HTTP {exc.code}: {exc.read().decode()[:200]}",
            err=True,
        )
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Failed to fetch: {exc}", err=True)
        sys.exit(2)

    if output_format == "json":
        click.echo(_json.dumps(payload, indent=2))
        return

    console.print("[bold]Insurance Fund Status[/bold]")
    for k, v in payload.items():
        console.print(f"  {k}: [cyan]{v}[/cyan]")


@node_insurance_group.command("compose-recovery")
@click.option(
    "--recipient", required=True,
    help="0x-prefixed Ethereum address to receive funds.",
)
@click.option(
    "--amount-wei", required=True, type=int,
    help="Amount to transfer, in wei (integer).",
)
@click.option(
    "--reason", required=True,
    help=(
        "Recovery reason (logged + included in the audit "
        "trail; e.g., 'Sprint 435 user-X reimbursement')."
    ),
)
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="json",
    show_default=True,
    help=(
        "Default JSON so operators can pipe directly into "
        "multi-sig signing tools."
    ),
)
def node_insurance_compose_recovery(
    recipient, amount_wei, reason, api_port, output_format,
):
    """Compose a multi-sig-uploadable insurance-fund recovery tx.

    Does NOT execute — produces the tx-payload bytes that the
    Foundation Safe operator uploads to their multi-sig
    signing tool. Per Vision §14 invariant: PRSM never executes
    the transfer directly; Foundation Safe holds the privilege.

    Output defaults to JSON so the payload can be piped into
    `safe-cli` or similar. Use --format text for human-readable
    summary.
    """
    import json as _json
    import urllib.error
    import urllib.request

    body = {
        "recipient": recipient,
        "amount_wei": amount_wei,
        "reason": reason,
    }
    url = (
        f"http://127.0.0.1:{api_port}/admin/insurance-fund/"
        f"compose-recovery"
    )
    req = urllib.request.Request(
        url, data=_json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            tx = _json.loads(r.read())
    except urllib.error.HTTPError as exc:
        if exc.code in (422, 503):
            click.echo(
                f"Compose failed: {exc.read().decode()[:300]}",
                err=True,
            )
            sys.exit(1)
        click.echo(
            f"Failed: HTTP {exc.code}: {exc.read().decode()[:200]}",
            err=True,
        )
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Failed: {exc}", err=True)
        sys.exit(2)

    if output_format == "json":
        # Default JSON output → operator can pipe to safe-cli.
        # click.echo (not console.print) keeps ANSI clean.
        click.echo(_json.dumps(tx, indent=2))
        return

    console.print(
        "[bold]Recovery TX composed[/bold] "
        "[dim](not executed — multi-sig must sign)[/dim]\n"
    )
    for k, v in tx.items():
        console.print(f"  {k}: [cyan]{v}[/cyan]")


# ──────────────────────────────────────────────────────────────
# Sprint 436 — `prsm node tee ...` CLI trifecta gap closure
#
# Two endpoints on the REST surface:
#   GET  /admin/tee-policy/node-status — this node's own
#        attestation tier (operators use this to pre-screen
#        before dispatching workloads)
#   POST /admin/tee-policy/evaluate — evaluate an
#        attestation blob against a policy
# ──────────────────────────────────────────────────────────────


@node.group("tee", invoke_without_command=False)
def node_tee_group():
    """TEE attestation + policy evaluation commands.

    Maps to /admin/tee-policy/* REST endpoints.
    """


@node_tee_group.command("status")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
def node_tee_status(api_port, output_format):
    """Show this node's TEE attestation tier.

    Wraps GET /admin/tee-policy/node-status. Enterprises
    use this to pre-screen which nodes are eligible to
    participate in a given workload before dispatching.
    """
    import json as _json
    import urllib.error
    import urllib.request

    url = (
        f"http://127.0.0.1:{api_port}/admin/tee-policy/"
        f"node-status"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            payload = _json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Failed: {exc}", err=True)
        sys.exit(2)

    if output_format == "json":
        click.echo(_json.dumps(payload, indent=2))
        return

    tier = payload.get("effective_tier", "?")
    vendor = payload.get("vendor", "?")
    verified = payload.get("vendor_verified", False)
    tier_color = {
        "tee-hardware": "green",
        "tee-software": "yellow",
        "none": "red",
    }.get(tier, "white")
    verified_marker = (
        "[green]✓ vendor-verified[/green]" if verified
        else "[yellow]⚠ not vendor-verified[/yellow]"
    )
    console.print("[bold]TEE Node Status[/bold]")
    console.print(
        f"  effective_tier: [{tier_color}]{tier}[/{tier_color}]"
    )
    console.print(f"  vendor: [cyan]{vendor}[/cyan]")
    console.print(f"  {verified_marker}")
    if payload.get("diagnostic"):
        console.print(
            f"  diagnostic: [dim]{payload['diagnostic']}[/dim]"
        )


@node_tee_group.command("evaluate")
@click.option(
    "--policy-file", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Path to a JSON file containing the TEEPolicy "
        "(min_tier, allowed_vendors, require_vendor_verified, "
        "etc.). See prsm.enterprise.tee_policy.TEEPolicy for "
        "the full schema."
    ),
)
@click.option(
    "--attestation-b64", default=None,
    help=(
        "Base64-encoded attestation blob to evaluate. If "
        "omitted, the policy is evaluated against a missing-"
        "attestation case (useful for pre-flight policy "
        "validation)."
    ),
)
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
def node_tee_evaluate(
    policy_file, attestation_b64, api_port, output_format,
):
    """Evaluate an attestation blob against a TEE policy.

    Wraps POST /admin/tee-policy/evaluate. Returns the
    evaluation result with effective_tier, policy_passed,
    reasons, etc.
    """
    import json as _json
    import urllib.error
    import urllib.request

    try:
        with open(policy_file) as f:
            policy = _json.load(f)
    except _json.JSONDecodeError as exc:
        click.echo(
            f"Policy file is not valid JSON: {exc}", err=True,
        )
        sys.exit(1)

    body = {"policy": policy}
    if attestation_b64:
        body["attestation_b64"] = attestation_b64

    url = (
        f"http://127.0.0.1:{api_port}/admin/tee-policy/evaluate"
    )
    req = urllib.request.Request(
        url, data=_json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            result = _json.loads(r.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 422:
            click.echo(
                f"Invalid policy or attestation: "
                f"{exc.read().decode()[:300]}",
                err=True,
            )
            sys.exit(1)
        click.echo(
            f"Failed: HTTP {exc.code}: "
            f"{exc.read().decode()[:200]}",
            err=True,
        )
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Failed: {exc}", err=True)
        sys.exit(2)

    if output_format == "json":
        click.echo(_json.dumps(result, indent=2))
        return

    passed = result.get("policy_passed", False)
    marker = (
        "[green]✓ POLICY PASSED[/green]" if passed
        else "[red]✗ POLICY FAILED[/red]"
    )
    console.print(f"[bold]TEE Policy Evaluation[/bold] — {marker}\n")
    console.print(
        f"  effective_tier: "
        f"[cyan]{result.get('effective_tier', '?')}[/cyan]"
    )
    if result.get("vendor"):
        console.print(f"  vendor: {result['vendor']}")
    reasons = result.get("reasons") or []
    if reasons:
        console.print("\n  Reasons:")
        for r in reasons:
            console.print(f"    • {r}")


# ──────────────────────────────────────────────────────────────
# Sprint 437 — `prsm node federated ...` + `prsm node pipeline ...`
# CLI trifecta gap closure (read-only admin triage).
#
# Both surfaces are deep (~7 endpoints each — list/details/execute/
# round/aggregate/issue-round/update). This sprint closes the
# read-only triage path (list + details) — the operator subset
# needed at incident time. Mutating commands stay deferred per
# the sprint-434 incident-CLI pattern.
# ──────────────────────────────────────────────────────────────


def _node_admin_list_details(
    *, group_name, list_path, details_path_template,
    api_port, output_format, filter_status,
    json_records_key,
    record_id_field,
):
    """Shared shape between federated + pipeline list commands.

    Both follow: GET /admin/<group>/job[?status=X] returns a
    {jobs|records: [...]} envelope; details endpoint takes a
    job_id path param.
    """
    import json as _json
    import urllib.parse
    import urllib.request

    qs = ""
    if filter_status:
        qs = "?" + urllib.parse.urlencode(
            {"status": filter_status},
        )
    url = f"http://127.0.0.1:{api_port}{list_path}{qs}"

    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            payload = _json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        click.echo(
            f"Failed to fetch {group_name} jobs: {exc}", err=True,
        )
        sys.exit(2)

    records = payload.get(json_records_key, [])
    if output_format == "json":
        click.echo(_json.dumps(payload, indent=2))
        return

    if not records:
        console.print(
            f"[green]✓ No active {group_name} jobs.[/green]"
        )
        return

    console.print(
        f"[bold]{group_name.capitalize()} jobs "
        f"({len(records)}):[/bold]\n"
    )
    for r in records:
        rid = r.get(record_id_field, "?")
        status = r.get("status", "?")
        console.print(
            f"  [bold]{rid}[/bold] "
            f"status=[cyan]{status}[/cyan]"
        )


@node.group("federated", invoke_without_command=False)
def node_federated_group():
    """Federated-learning admin triage commands (read-only).

    Maps to /admin/federated/* REST endpoints. Mutating
    commands (issue-round, aggregate, update) deferred —
    operators use the REST surface or `prsm_federated_learning`
    MCP for those.
    """


@node_federated_group.command("list")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
@click.option(
    "--status", default=None,
    help="Filter by JobStatus (pending, active, completed, ...).",
)
def node_federated_list(api_port, output_format, status):
    """List federated-learning jobs on this node."""
    _node_admin_list_details(
        group_name="federated",
        list_path="/admin/federated/job",
        details_path_template="/admin/federated/job/{}",
        api_port=api_port, output_format=output_format,
        filter_status=status,
        json_records_key="jobs",
        record_id_field="job_id",
    )


@node_federated_group.command("details")
@click.argument("job_id")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
def node_federated_details(job_id, api_port, output_format):
    """Show details for one federated-learning job."""
    import json as _json
    import urllib.error
    import urllib.request

    url = (
        f"http://127.0.0.1:{api_port}/admin/federated/"
        f"job/{job_id}"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            payload = _json.loads(r.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            click.echo(
                f"Federated job {job_id!r} not found.", err=True,
            )
            sys.exit(1)
        click.echo(
            f"Failed: HTTP {exc.code}: "
            f"{exc.read().decode()[:200]}",
            err=True,
        )
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Failed: {exc}", err=True)
        sys.exit(2)

    if output_format == "json":
        click.echo(_json.dumps(payload, indent=2))
        return

    console.print(
        f"[bold]Federated job {payload.get('job_id', '?')}[/bold]"
    )
    for k, v in payload.items():
        if k == "job_id":
            continue
        if isinstance(v, (list, dict)):
            v_str = _json.dumps(v, indent=2)[:200]
        else:
            v_str = str(v)
        console.print(f"  {k}: [cyan]{v_str}[/cyan]")


@node.group("pipeline", invoke_without_command=False)
def node_pipeline_group():
    """Pipeline-inference admin triage commands (read-only).

    Maps to /admin/inference/pipeline/* REST endpoints.
    """


@node_pipeline_group.command("list")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
def node_pipeline_list(api_port, output_format):
    """List pipeline-inference jobs on this node."""
    _node_admin_list_details(
        group_name="pipeline",
        list_path="/admin/inference/pipeline/job",
        details_path_template="/admin/inference/pipeline/job/{}",
        api_port=api_port, output_format=output_format,
        filter_status=None,  # endpoint doesn't accept filter
        json_records_key="jobs",
        record_id_field="job_id",
    )


@node_pipeline_group.command("details")
@click.argument("job_id")
@click.option("--api-port", default=8000, type=int)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]), default="text",
    show_default=True,
)
def node_pipeline_details(job_id, api_port, output_format):
    """Show details for one pipeline-inference job."""
    import json as _json
    import urllib.error
    import urllib.request

    url = (
        f"http://127.0.0.1:{api_port}/admin/inference/"
        f"pipeline/job/{job_id}"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            payload = _json.loads(r.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            click.echo(
                f"Pipeline job {job_id!r} not found.", err=True,
            )
            sys.exit(1)
        click.echo(
            f"Failed: HTTP {exc.code}: "
            f"{exc.read().decode()[:200]}",
            err=True,
        )
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Failed: {exc}", err=True)
        sys.exit(2)

    if output_format == "json":
        click.echo(_json.dumps(payload, indent=2))
        return

    console.print(
        f"[bold]Pipeline job {payload.get('job_id', '?')}[/bold]"
    )
    for k, v in payload.items():
        if k == "job_id":
            continue
        if isinstance(v, (list, dict)):
            v_str = _json.dumps(v, indent=2)[:200]
        else:
            v_str = str(v)
        console.print(f"  {k}: [cyan]{v_str}[/cyan]")


if __name__ == "__main__":
    main()
