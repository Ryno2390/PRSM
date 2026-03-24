"""
PRSM Command Line Interface
Main entry point for PRSM CLI commands
"""

import asyncio
import socket
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import click
import uvicorn
from rich.console import Console
from rich.table import Table

from prsm.compute.nwtn.backends.config import detect_available_backends

console = Console()


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
    from prsm.node.config import NodeRole

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

    # Optional dependency: IPFS daemon availability.
    ipfs_host, ipfs_port, ipfs_parse_error = _parse_endpoint(config.ipfs_api_url, default_port=5001)
    ipfs_role_expected = any(role in (NodeRole.FULL, NodeRole.STORAGE) for role in config.roles)
    if ipfs_parse_error or not ipfs_host or not ipfs_port:
        checks.append(
            PreflightCheckResult(
                name="IPFS dependency (optional)",
                status=PREFLIGHT_WARN,
                required=False,
                details=f"invalid ipfs_api_url '{config.ipfs_api_url}'",
                remediation="Set a valid ipfs_api_url or install/start IPFS if storage features are needed.",
            )
        )
    else:
        ipfs_ok, ipfs_detail = _probe_tcp_endpoint(ipfs_host, ipfs_port, timeout_seconds=1.0)
        checks.append(
            PreflightCheckResult(
                name="IPFS dependency (optional)",
                status=PREFLIGHT_PASS if ipfs_ok else PREFLIGHT_WARN,
                required=False,
                details=(
                    f"{ipfs_detail}; storage role active"
                    if ipfs_role_expected
                    else f"{ipfs_detail}; storage role not active"
                ),
                remediation=(
                    "None"
                    if ipfs_ok
                    else "Install/start IPFS for storage features; compute/routing startup remains available."
                ),
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
    from prsm.core.config.schemas import PRSMConfig
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


@click.group()
@click.version_option(version="0.2.1", prog_name="PRSM")
def main():
    """
    PRSM: Protocol for Recursive Scientific Modeling
    
    A decentralized AI framework for scientific discovery.
    """
    pass


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to (default: localhost for security)")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@click.option("--workers", default=1, help="Number of worker processes")
def serve(host: str, port: int, reload: bool, workers: int):
    """Start the PRSM API server"""
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
def status():
    """Show PRSM system status"""
    _init_config()
    from prsm.core.config import get_settings
    settings = get_settings()
    table = Table(title="PRSM System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Details", style="green")

    if settings:
        table.add_row("Configuration", "✅ Loaded", f"Environment: {getattr(settings, 'environment', 'unknown')}")
        table.add_row("Database", "🔄 Checking...", str(getattr(settings, 'database_url', 'sqlite (default)')))
        table.add_row("IPFS", "🔄 Checking...", f"{getattr(settings, 'ipfs_host', 'localhost')}:{getattr(settings, 'ipfs_port', 5001)}")
        table.add_row("NWTN", "✅ Enabled" if getattr(settings, 'nwtn_enabled', True) else "❌ Disabled",
                      f"Model: {getattr(settings, 'nwtn_default_model', 'default')}")
        table.add_row("FTNS", "✅ Enabled" if getattr(settings, 'ftns_enabled', True) else "❌ Disabled",
                      f"Initial grant: {getattr(settings, 'ftns_initial_grant', 100)}")
    else:
        table.add_row("Configuration", "⚠️ Using defaults", "No .env file found")
        table.add_row("NWTN", "✅ Enabled", "Default model")
        table.add_row("FTNS", "✅ Enabled", "Default settings")

    console.print(table)


@main.command()
@click.option("--port", default=8501, help="Port for the Streamlit dashboard")
@click.option("--api-port", default=8000, help="Port for the PRSM API")
def dashboard(port: int, api_port: int):
    """Launch the high-fidelity PRSM Dashboard and API"""
    import subprocess
    import time
    import os

    _init_config()
    
    console.print("🚀 Starting PRSM Command Center...", style="bold green")
    
    # 1. Start the API Server in the background
    console.print(f"📡 Launching API Server on port {api_port}...", style="dim")
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "prsm.interface.api.main:app", "--port", str(api_port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
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
        api_process.terminate()


@main.command()
@click.argument("query", required=True)
@click.option("--context", "-c", default=100, help="FTNS context allocation")
@click.option("--user-id", default="cli-user", help="User ID for the query")
@click.option("--api-url", default="http://127.0.0.1:8000", help="PRSM API URL")
def query(query: str, context: int, user_id: str, api_url: str):
    """Submit a query to NWTN (requires running server)"""
    import httpx
    from rich.panel import Panel
    
    console.print(f"🧠 Submitting query to NWTN...", style="bold blue")
    console.print(f"Query: {query}")
    console.print(f"Context allocation: {context} FTNS")
    
    payload = {
        "prompt": query,
        "context_allocation": str(context),
        "user_id": user_id
    }
    
    try:
        with console.status("[bold green]NWTN Orchestrator is reasoning..."):
            with httpx.Client(timeout=120.0) as client:
                response = client.post(f"{api_url}/query", json=payload)
                
        if response.status_code == 200:
            data = response.json()
            answer = data.get("final_answer", "No answer provided.")
            
            console.print("\n")
            console.print(Panel(answer, title="[bold green]NWTN Synthesis[/bold green]", border_style="green"))
            
            trace = data.get("reasoning_trace", [])
            if trace:
                console.print("\n[bold cyan]Reasoning Trace:[/bold cyan]")
                for i, step in enumerate(trace, 1):
                    action = step.get("input_data", {}).get("action", str(step.get("agent_type", "Process")))
                    console.print(f"  [dim]{i}.[/dim] {action}")
                    
            conf = data.get('confidence_score', 0)
            ctx = data.get('context_used', 0)
            console.print(f"\n[dim]Confidence Score: {conf:.2f} | Context Used: {ctx} tokens[/dim]")
        else:
            console.print(f"\n[bold red]Error ({response.status_code}):[/bold red] {response.text}")
            
    except httpx.RequestError as e:
        console.print(f"\n[bold red]Connection Error:[/bold red] Could not connect to {api_url}.")
        console.print("Make sure the PRSM API server is running (`prsm serve`).")
        console.print(f"[dim]Details: {e}[/dim]")


@main.command()
def init():
    """Initialize PRSM configuration and database"""
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
    console.print("3. Run: prsm serve")


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


@main.group()
def node():
    """P2P node management commands"""
    pass


def _run_node_wizard() -> "NodeConfig":
    """Interactive wizard that configures a real PRSM node."""
    from prsm.node.config import NodeConfig, NodeRole
    from prsm.node.compute_provider import detect_resources

    config = NodeConfig()

    console.print("=" * 60, style="bold magenta")
    console.print("  PRSM Node Setup Wizard", style="bold magenta")
    console.print("=" * 60, style="bold magenta")
    console.print()

    # 1. Display name
    name = click.prompt("  Node display name", default="prsm-node")
    config.display_name = name

    # 2. Role selection
    console.print()
    console.print("  Node roles:", style="bold")
    console.print("    full    - Compute + storage + routing (recommended)")
    console.print("    compute - Compute jobs only")
    console.print("    storage - Storage contribution only")
    role_str = click.prompt(
        "  Choose your node role",
        type=click.Choice(["full", "compute", "storage"]),
        default="full",
    )
    config.roles = [NodeRole(role_str)]

    # 3. Resource detection
    console.print()
    console.print("  Detecting system resources...", style="dim")
    resources = detect_resources()
    console.print(f"    CPUs: {resources.cpu_count}", style="green")
    console.print(f"    RAM:  {resources.memory_total_gb:.1f} GB", style="green")
    if resources.gpu_available:
        console.print(f"    GPU:  {resources.gpu_name} ({resources.gpu_memory_gb:.1f} GB)", style="green")
    else:
        console.print("    GPU:  not detected", style="dim")

    if NodeRole(role_str) in (NodeRole.FULL, NodeRole.COMPUTE):
        cpu_pct = click.prompt("  CPU allocation for compute jobs (%)", default=50, type=int)
        config.cpu_allocation_pct = max(10, min(90, cpu_pct))
        mem_pct = click.prompt("  Memory allocation for compute jobs (%)", default=50, type=int)
        config.memory_allocation_pct = max(10, min(90, mem_pct))

    # 4. IPFS detection
    if NodeRole(role_str) in (NodeRole.FULL, NodeRole.STORAGE):
        console.print()
        console.print("  Checking for IPFS daemon...", style="dim")
        ipfs_ok = False
        try:
            import subprocess
            result = subprocess.run(
                ["ipfs", "id"], capture_output=True, timeout=5
            )
            ipfs_ok = result.returncode == 0
        except Exception:
            pass

        if ipfs_ok:
            console.print("    IPFS daemon detected!", style="green")
            gb = click.prompt("  Storage to pledge (GB)", default=10.0, type=float)
            config.storage_gb = gb
        else:
            console.print("    IPFS not detected. Storage features will be disabled.", style="yellow")
            console.print("    Install: https://docs.ipfs.tech/install/", style="dim")

    # 5. Network configuration
    console.print()
    p2p_port = click.prompt("  P2P port", default=9001, type=int)
    config.p2p_port = p2p_port
    api_port = click.prompt("  API port", default=8000, type=int)
    config.api_port = api_port

    bootstrap = click.prompt(
        "  Bootstrap node (host:port, or empty for none)",
        default="wss://bootstrap.prsm-network.com",
    )
    if bootstrap.strip():
        config.bootstrap_nodes = [b.strip() for b in bootstrap.split(",")]
    else:
        config.bootstrap_nodes = []

    # Save config
    config.save()
    console.print()
    console.print("  Configuration saved to ~/.prsm/node_config.json", style="green")
    return config


@node.command()
@click.option("--wizard", is_flag=True, help="Run interactive setup wizard")
@click.option("--p2p-port", default=None, type=int, help="P2P listen port (default: 9001)")
@click.option("--api-port", default=None, type=int, help="API listen port (default: 8000)")
@click.option("--bootstrap", default=None, help="Bootstrap node address (host:port)")
@click.option("--no-dashboard", is_flag=True, help="Disable live dashboard (static output)")
@click.option("--cpu", default=None, type=int, help="CPU allocation % (10-90)")
@click.option("--memory", default=None, type=int, help="RAM allocation % (10-90)")
@click.option("--storage", default=None, type=float, help="Storage to pledge in GB")
@click.option("--jobs", default=None, type=int, help="Max concurrent compute jobs")
def start(wizard: bool, p2p_port: int, api_port: int, bootstrap: str, no_dashboard: bool,
          cpu: Optional[int], memory: Optional[int], storage: Optional[float], jobs: Optional[int]):
    """Start a PRSM network node with real P2P connectivity."""
    from prsm.node.config import NodeConfig

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
                    table.add_row("IPFS", "connected" if st["ipfs_available"] else "not available")
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


@node.command()
def peers():
    """List connected peers"""
    from prsm.node.config import NodeConfig
    from prsm.node.identity import load_node_identity
    from pathlib import Path

    config = NodeConfig.load()
    identity = load_node_identity(config.identity_path)

    if not identity:
        console.print("No node identity found. Run 'prsm node start --wizard' first.", style="yellow")
        return

    console.print(f"Node ID: {identity.node_id}", style="bold cyan")
    console.print(f"P2P address: ws://{config.listen_host}:{config.p2p_port}", style="cyan")
    console.print()

    # Try connecting to the running node's API to get live peer data
    try:
        import httpx
        resp = httpx.get(f"http://127.0.0.1:{config.api_port}/peers", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            connected = data.get("connected", [])
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

            console.print(f"\nKnown peers: {data.get('known_count', 0)}")
            return
    except Exception:
        pass

    console.print("Node is not running. Start it with 'prsm node start'.", style="yellow")


@node.command()
def info():
    """Show node identity and configuration"""
    from prsm.node.config import NodeConfig
    from prsm.node.identity import load_node_identity

    config = NodeConfig.load()
    identity = load_node_identity(config.identity_path)

    if not identity:
        console.print("No node identity found. Run 'prsm node start --wizard' first.", style="yellow")
        return

    table = Table(title="PRSM Node Info")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Node ID", identity.node_id)
    table.add_row("Display Name", config.display_name)
    table.add_row("Public Key", identity.public_key_b64[:32] + "...")
    table.add_row("Roles", ", ".join(r.value for r in config.roles))
    table.add_row("P2P Port", str(config.p2p_port))
    table.add_row("API Port", str(config.api_port))
    table.add_row("Data Dir", config.data_dir)
    table.add_row("Bootstrap Nodes", ", ".join(config.bootstrap_nodes) if config.bootstrap_nodes else "none")
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
    from prsm.node.config import NodeConfig, NodeRole
    from prsm.node.compute_provider import detect_resources
    
    config = NodeConfig.load()
    
    # Handle --show flag
    if show:
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
    from prsm.node.config import NodeRole
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
    """Run interactive configuration wizard for resource settings."""
    from prsm.node.config import NodeRole
    from prsm.node.compute_provider import detect_resources
    
    resources = detect_resources()
    
    console.print()
    console.print("=" * 60, style="bold magenta")
    console.print("  PRSM Node Resource Configuration", style="bold magenta")
    console.print("=" * 60, style="bold magenta")
    console.print()
    
    # Show current settings
    console.print("  Current settings:", style="bold")
    console.print(f"    CPU allocation:    {config.cpu_allocation_pct}%")
    console.print(f"    Memory allocation: {config.memory_allocation_pct}%")
    console.print(f"    Storage pledged:   {config.storage_gb} GB")
    console.print(f"    Max concurrent jobs: {config.max_concurrent_jobs}")
    console.print(f"    GPU allocation:    {config.gpu_allocation_pct}%")
    console.print(f"    Upload limit:      {config.upload_mbps_limit} Mbps" if config.upload_mbps_limit > 0 else "    Upload limit:      unlimited")
    if config.active_hours_start is not None:
        console.print(f"    Active hours:      {config.active_hours_start:02d}-{config.active_hours_end:02d}")
    else:
        console.print(f"    Active hours:      always on")
    console.print()
    
    # Prompt for each setting
    console.print("  Detected system resources:", style="bold")
    console.print(f"    CPUs: {resources.cpu_count}")
    console.print(f"    RAM:  {resources.memory_total_gb:.1f} GB")
    if resources.gpu_available:
        console.print(f"    GPU:  {resources.gpu_name} ({resources.gpu_memory_gb:.1f} GB)")
    console.print()
    
    # CPU allocation
    cpu = click.prompt("  CPU allocation for compute jobs (%)", default=config.cpu_allocation_pct, type=int)
    config.cpu_allocation_pct = max(10, min(90, cpu))
    
    # Memory allocation
    mem = click.prompt("  Memory allocation for compute jobs (%)", default=config.memory_allocation_pct, type=int)
    config.memory_allocation_pct = max(10, min(90, mem))
    
    # Storage
    storage = click.prompt("  Storage to pledge (GB)", default=config.storage_gb, type=float)
    config.storage_gb = storage
    
    # Max concurrent jobs
    jobs = click.prompt("  Max concurrent compute jobs", default=config.max_concurrent_jobs, type=int)
    config.max_concurrent_jobs = max(1, jobs)
    
    # GPU allocation (if GPU available)
    if resources.gpu_available:
        gpu = click.prompt("  GPU allocation %", default=config.gpu_allocation_pct, type=int)
        config.gpu_allocation_pct = max(10, min(100, gpu))
    
    # Upload limit
    upload_default = str(config.upload_mbps_limit) if config.upload_mbps_limit > 0 else "0"
    upload_str = click.prompt("  Upload bandwidth limit in Mbps (0=unlimited)", default=upload_default)
    config.upload_mbps_limit = float(upload_str)
    
    # Active hours
    current_hours = f"{config.active_hours_start:02d}-{config.active_hours_end:02d}" if config.active_hours_start is not None else "off"
    hours_input = click.prompt("  Active hours (e.g., '22-8' or 'off' for always on)", default=current_hours)
    if hours_input.lower() != "off":
        start, end = parse_active_hours(hours_input)
        config.active_hours_start = start
        config.active_hours_end = end
    else:
        config.active_hours_start = None
        config.active_hours_end = None
    
    # Active days
    day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    current_days = "weekdays" if config.active_days == [0, 1, 2, 3, 4] else \
                   "weekends" if config.active_days == [5, 6] else \
                   ",".join(day_names[d] for d in config.active_days) if config.active_days else "everyday"
    days_input = click.prompt("  Active days (e.g., 'weekdays', 'weekends', 'mon,wed,fri', or 'everyday')", default=current_days)
    if days_input.lower() not in ("everyday", ""):
        config.active_days = parse_active_days(days_input)
    else:
        config.active_days = []
    
    # Save configuration
    config.save()
    console.print()
    console.print(f"  ✅ Configuration saved to {config.config_path}", style="bold green")


@main.group()
def teacher():
    """Teacher model management commands"""
    pass


@teacher.command("list")
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def list_teachers(api_url: str):
    """List available teacher models"""
    import httpx
    
    console.print("🎓 Fetching teacher models...", style="bold blue")
    
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{api_url}/teacher/list")
            
            if response.status_code == 200:
                data = response.json()
                teachers = data.get("teachers", [])
                
                if not teachers:
                    console.print("No teacher models found.", style="yellow")
                    console.print("💡 Create one with: prsm teacher create <specialization>", style="blue")
                    return
                
                table = Table(title="Teacher Models")
                table.add_column("ID", style="cyan", no_wrap=True)
                table.add_column("Name", style="green")
                table.add_column("Specialization", style="magenta")
                table.add_column("Domain", style="blue")
                table.add_column("Status", style="yellow")
                
                for t in teachers:
                    # Truncate ID for display
                    teacher_id = t.get("teacher_id", "unknown")
                    short_id = teacher_id[:8] + "..." if len(teacher_id) > 8 else teacher_id
                    table.add_row(
                        short_id,
                        t.get("name", "N/A"),
                        t.get("specialization", "N/A"),
                        t.get("domain", "N/A"),
                        t.get("status", "unknown")
                    )
                
                console.print(table)
                console.print(f"\n📊 Total: {data.get('count', 0)} teacher models", style="blue")
            else:
                console.print(f"❌ Failed to fetch teachers: {response.status_code}", style="red")
                console.print(response.text, style="dim")
                
    except httpx.ConnectError:
        console.print("❌ Could not connect to PRSM API", style="red")
        console.print(f"💡 Make sure the PRSM node is running at {api_url}", style="yellow")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@teacher.command()
@click.argument("specialization")
@click.option('--domain', help='Sub-domain (defaults to specialization)')
@click.option('--use-real/--no-real', default=True, help='Use PyTorch backend if available')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def create(specialization: str, domain: Optional[str], use_real: bool, api_url: str):
    """Create a new teacher model"""
    import httpx
    
    console.print(f"🎓 Creating teacher model for {specialization}...", style="bold green")
    
    payload = {
        "specialization": specialization,
        "domain": domain,
        "use_real_implementation": use_real
    }
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{api_url}/teacher/create", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                console.print("✅ Teacher model created successfully!", style="bold green")
                console.print(f"   ID: {data.get('teacher_id', 'N/A')}", style="cyan")
                console.print(f"   Name: {data.get('name', 'N/A')}", style="green")
                console.print(f"   Specialization: {data.get('specialization', 'N/A')}", style="magenta")
                console.print(f"   Domain: {data.get('domain', 'N/A')}", style="blue")
                if data.get('reward_ftns'):
                    console.print(f"   Reward: {data.get('reward_ftns')} FTNS", style="yellow")
            else:
                console.print(f"❌ Failed to create teacher: {response.status_code}", style="red")
                console.print(response.text, style="dim")
                
    except httpx.ConnectError:
        console.print("❌ Could not connect to PRSM API", style="red")
        console.print(f"💡 Make sure the PRSM node is running at {api_url}", style="yellow")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@teacher.command()
@click.argument("teacher_id")
@click.option('--epochs', type=int, help='Number of training epochs (1-100)')
@click.option('--learning-rate', type=float, help='Learning rate for training')
@click.option('--training-data-cid', help='IPFS CID of custom training data')
@click.option('--follow', '-f', is_flag=True, help='Poll until training completes')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def train(teacher_id: str, epochs: Optional[int], learning_rate: Optional[float],
          training_data_cid: Optional[str], follow: bool, api_url: str):
    """Start training for a teacher model. Use -f to follow progress until completion."""
    import httpx
    
    payload = {}
    if epochs is not None:
        payload["epochs"] = epochs
    if learning_rate is not None:
        payload["learning_rate"] = learning_rate
    if training_data_cid is not None:
        payload["training_data_cid"] = training_data_cid
    
    try:
        r = httpx.post(f"{api_url}/teacher/{teacher_id}/train", json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()
        run_id = data["run_id"]
        poll_url = f"{api_url}{data['poll_url']}"

        console.print(f"🎓  Training started", style="bold green")
        console.print(f"    Run ID:        {run_id}")
        console.print(f"    Total epochs:  {data.get('total_epochs', 'unknown')}")
        console.print(f"    FTNS charged:  {data['cost_ftns']}")
        console.print(f"    Poll:          prsm teacher status {teacher_id} {run_id}", style="dim")

        if follow:
            console.print()
            _poll_training_status(poll_url, run_id)

    except httpx.ConnectError:
        console.print("❌  Node not running. Start with: prsm node start", style="red")
    except httpx.HTTPStatusError as e:
        console.print(f"❌  {e.response.json().get('detail', str(e))}", style="red")


def _poll_training_status(poll_url: str, run_id: str) -> None:
    """Block and poll training status until a terminal state is reached."""
    import httpx
    import time
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Training…", total=100)
        while True:
            try:
                r = httpx.get(poll_url, timeout=10)
                data = r.json()
                status = data["status"]

                if "progress" in data:
                    pct = data["progress"]["progress_pct"]
                    epoch = data["progress"]["current_epoch"]
                    total = data["progress"]["total_epochs"]
                    progress.update(task,
                        completed=pct,
                        description=f"Epoch {epoch}/{total}")

                if status == "completed":
                    progress.update(task, completed=100)
                    console.print("✅  Training complete!", style="bold green")
                    if "result" in data:
                        r = data["result"]
                        console.print(f"    Epochs:     {r.get('total_epochs')}")
                        loss_val = r.get('final_train_loss', 'N/A')
                        if isinstance(loss_val, (int, float)):
                            console.print(f"    Final loss: {loss_val:.4f}")
                        else:
                            console.print(f"    Final loss: {loss_val}")
                        if r.get("best_checkpoint_path"):
                            console.print(f"    Checkpoint: {r['best_checkpoint_path']}", style="dim")
                    return

                if status == "failed":
                    console.print(f"❌  Training failed: {data.get('error', 'unknown')}", style="red")
                    return

                if status == "cancelled":
                    console.print("⚠️   Training was cancelled.", style="yellow")
                    return

            except Exception as e:
                console.print(f"⚠️   Poll error: {e}", style="yellow")

            time.sleep(3)


@teacher.command()
@click.argument("teacher_id")
@click.argument("run_id")
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def status(teacher_id: str, run_id: str, api_url: str):
    """Check the status of a training run."""
    import httpx
    try:
        r = httpx.get(f"{api_url}/teacher/{teacher_id}/training/{run_id}", timeout=10)
        r.raise_for_status()
        data = r.json()

        status_color = {
            "pending":   "yellow", "running": "cyan",
            "completed": "green",  "failed":  "red",
            "cancelled": "dim",
        }.get(data["status"], "white")

        console.print(f"  Status:  [{status_color}]{data['status']}[/{status_color}]")
        if "progress" in data:
            p = data["progress"]
            console.print(f"  Epoch:   {p['current_epoch']}/{p['total_epochs']} "
                          f"({p['progress_pct']}%)")
            console.print(f"  Elapsed: {p['elapsed_seconds']}s")
        if data.get("error"):
            console.print(f"  Error:   {data['error']}", style="red")
        if data.get("result"):
            r = data["result"]
            console.print(f"  Loss:    {r.get('final_train_loss', 'N/A')}")
            console.print(f"  Checkpoint: {r.get('best_checkpoint_path', 'none')}", style="dim")
    except httpx.ConnectError:
        console.print("❌  Node not running.", style="red")


@teacher.command("cancel-training")
@click.argument("teacher_id")
@click.argument("run_id")
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def cancel_training(teacher_id: str, run_id: str, api_url: str):
    """Cancel a pending or running training job."""
    import httpx
    try:
        r = httpx.delete(f"{api_url}/teacher/{teacher_id}/training/{run_id}", timeout=10)
        r.raise_for_status()
        console.print(f"✅  Cancellation requested for run {run_id}", style="yellow")
        console.print("    Poll with: prsm teacher status <teacher_id> <run_id>", style="dim")
    except httpx.HTTPStatusError as e:
        console.print(f"❌  {e.response.json().get('detail', str(e))}", style="red")


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
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def submit(prompt: str, model: str, max_tokens: int, budget: float, api_url: str):
    """Submit a compute job to the PRSM network."""
    import httpx
    
    console.print(f"🚀 Submitting compute job...", style="bold blue")
    console.print(f"   Model: {model}")
    console.print(f"   Max tokens: {max_tokens}")
    if budget:
        console.print(f"   Budget: {budget} FTNS")
    
    try:
        response = httpx.post(
            f"{api_url}/api/v1/compute/jobs",
            json={
                "prompt": prompt,
                "model": model,
                "max_tokens": max_tokens,
                "budget": budget
            },
            timeout=30.0
        )
        
        if response.status_code == 200:
            data = response.json()
            console.print(f"✅ Job submitted successfully!", style="bold green")
            console.print(f"   Job ID: {data.get('job_id')}")
            console.print(f"   Status: {data.get('status')}")
            console.print(f"   Estimated cost: {data.get('estimated_cost', 0):.4f} FTNS")
        else:
            console.print(f"❌ Failed to submit job: {response.status_code}", style="red")
            console.print(f"   {response.text}")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
        console.print("   Make sure the server is running: prsm serve")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@compute.command()
@click.argument('job-id')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def status(job_id: str, api_url: str):
    """Get status of a compute job."""
    import httpx
    
    try:
        response = httpx.get(f"{api_url}/api/v1/compute/jobs/{job_id}", timeout=10.0)
        
        if response.status_code == 200:
            data = response.json()
            table = Table(title=f"Job Status: {job_id}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Job ID", data.get('job_id', 'N/A'))
            table.add_row("Status", data.get('status', 'N/A'))
            table.add_row("Progress", f"{data.get('progress', 0) * 100:.1f}%")
            table.add_row("Model", data.get('model', 'N/A'))
            if data.get('result'):
                table.add_row("Result", "Available")
            console.print(table)
        elif response.status_code == 404:
            console.print(f"❌ Job not found: {job_id}", style="red")
        else:
            console.print(f"❌ Failed to get job status: {response.status_code}", style="red")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@compute.command()
@click.argument('job-id')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def result(job_id: str, api_url: str):
    """Get result of a completed compute job."""
    import httpx
    
    try:
        response = httpx.get(f"{api_url}/api/v1/compute/jobs/{job_id}/result", timeout=10.0)
        
        if response.status_code == 200:
            data = response.json()
            console.print("📄 Job Result:", style="bold green")
            console.print(data.get('content', 'No content'))
            console.print()
            console.print(f"💰 Cost: {data.get('ftns_cost', 0):.4f} FTNS")
            console.print(f"⏱️  Execution time: {data.get('execution_time', 0):.2f}s")
        elif response.status_code == 404:
            console.print(f"❌ Job not found or not complete: {job_id}", style="red")
        else:
            console.print(f"❌ Failed to get result: {response.status_code}", style="red")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@compute.command()
@click.argument('job-id')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def cancel(job_id: str, api_url: str):
    """Cancel a running compute job."""
    import httpx
    
    try:
        response = httpx.post(f"{api_url}/api/v1/compute/jobs/{job_id}/cancel", timeout=10.0)
        
        if response.status_code == 200:
            console.print(f"✅ Job {job_id} cancelled", style="green")
        else:
            console.print(f"❌ Failed to cancel job: {response.status_code}", style="red")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@compute.command("list")
@click.option('--limit', default=10, type=int, help='Maximum number of jobs to list')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def list_compute_jobs(limit: int, api_url: str):
    """List recent compute jobs."""
    import httpx
    
    try:
        response = httpx.get(f"{api_url}/api/v1/compute/jobs?limit={limit}", timeout=10.0)
        
        if response.status_code == 200:
            data = response.json()
            jobs = data.get('jobs', [])
            
            if not jobs:
                console.print("No jobs found.", style="dim")
                return
            
            table = Table(title="Recent Compute Jobs")
            table.add_column("Job ID", style="cyan")
            table.add_column("Status", style="magenta")
            table.add_column("Model", style="blue")
            table.add_column("Created", style="green")
            
            for job in jobs:
                table.add_row(
                    job.get('job_id', 'N/A')[:16] + "...",
                    job.get('status', 'N/A'),
                    job.get('model', 'N/A'),
                    job.get('created_at', 'N/A')[:19]
                )
            console.print(table)
        else:
            console.print(f"❌ Failed to list jobs: {response.status_code}", style="red")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


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
@click.option("--api-url", default=None,           help="PRSM API URL (default: from stored credentials)")
def history(limit: int, search: Optional[str], api_url: str) -> None:
    """Show your FTNS transaction history."""
    import httpx

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
    """IPFS storage management commands"""
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
def upload(
    file_path: str,
    description: str,
    royalty_rate: float,
    parent_cids: str,
    replicas: int,
    api_url: str,
) -> None:
    """
    Upload a file to IPFS and register provenance for royalty collection.

    The file is stored on IPFS and a provenance record is created in the
    platform database. When other users access this content, they pay the
    configured royalty rate to your FTNS balance.

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
            with console.status("[bold green]Uploading to IPFS..."):
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
        console.print("❌ IPFS service unavailable", style="red")
        console.print("💡 Ensure an IPFS daemon is running at http://127.0.0.1:5001",
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
    """Download content from IPFS by CID."""
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
# MARKETPLACE COMMANDS
# ============================================================================

@main.group()
def marketplace() -> None:
    """Browse, purchase, and publish resources on the PRSM marketplace.

    \b
    Supported resource types:
      ai_model         AI/ML models (language, vision, multimodal)
      dataset          Training and evaluation datasets
      agent_workflow   AI agent configurations and pipelines
      tool             AI utilities and integrations
      compute_resource GPU instances and cloud compute
      knowledge_base   Documentation and embeddings
      evaluation_metric  Model benchmarks and evaluators
      training_dataset Specialized training data
      safety_dataset   AI safety and alignment datasets
    """
    pass


@marketplace.command("list")
@click.option("--type", "resource_type", default=None, help="Filter by resource type (ai_model, dataset, etc.)")
@click.option("--query", default=None, help="Search query for name/description")
@click.option("--quality", default=None, type=float, help="Minimum quality score (0.0-1.0)")
@click.option("--pricing", default=None, type=click.Choice(["free", "paid", "freemium"]), help="Filter by pricing model")
@click.option("--tags", default=None, help="Comma-separated tags to filter by")
@click.option("--sort", default="relevance", type=click.Choice(["relevance", "quality", "price_low", "price_high", "downloads", "rating"]), help="Sort order")
@click.option("--page", default=1, type=int, help="Page number for pagination")
@click.option("--limit", default=20, type=int, help="Results per page (max 100)")
@click.option("--api-url", default=None, help="PRSM API URL (default: from stored credentials)")
def marketplace_list(resource_type: Optional[str], query: Optional[str], quality: Optional[float],
                     pricing: Optional[str], tags: Optional[str], sort: str, page: int, limit: int,
                     api_url: Optional[str]) -> None:
    """List and search marketplace resources.

    \b
    Examples:
        prsm marketplace list
        prsm marketplace list --type ai_model --sort quality
        prsm marketplace list --query "gpt" --pricing free
        prsm marketplace list --tags nlp,transformers --limit 50
    """
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    # Build query parameters
    params: dict = {"page": page, "limit": min(limit, 100)}
    if resource_type:
        params["resource_type"] = resource_type
    if query:
        params["query"] = query
    if quality is not None:
        params["min_quality"] = quality
    if pricing:
        params["pricing_model"] = pricing
    if tags:
        params["tags"] = tags
    if sort:
        params["sort"] = sort

    console.print("🛒 Searching marketplace...", style="bold blue")

    try:
        response = httpx.get(
            f"{url}/api/v1/marketplace/resources",
            params=params,
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    elif response.status_code != 200:
        console.print(f"❌ Failed: HTTP {response.status_code}", style="red")
        console.print(f"   {response.text[:200]}")
        raise SystemExit(1)

    data = response.json()
    resources = data.get("resources", [])
    total = data.get("total", 0)

    if not resources:
        console.print("No resources found.", style="dim")
        console.print("💡 Try adjusting your filters or search query", style="yellow")
        return

    table = Table(title=f"Marketplace Resources ({total} total)")
    table.add_column("ID", style="dim", max_width=12)
    table.add_column("Name", style="cyan", max_width=28)
    table.add_column("Type", style="blue", max_width=14)
    table.add_column("Quality", style="green", justify="right")
    table.add_column("Price", style="yellow", justify="right")
    table.add_column("Rating", style="magenta", justify="right")
    table.add_column("Tags", style="white", max_width=20)

    for r in resources:
        # Format price
        price = r.get("price", 0)
        pricing_model = r.get("pricing_model", "free")
        if pricing_model == "free":
            price_str = "Free"
        elif pricing_model == "freemium":
            price_str = f"Free+{price:.2f}"
        else:
            price_str = f"{price:.2f} FTNS"

        # Format tags
        resource_tags = r.get("tags", [])
        tags_str = ", ".join(resource_tags[:3])
        if len(resource_tags) > 3:
            tags_str += "..."

        # Format rating
        rating = r.get("average_rating", 0)
        rating_str = f"⭐ {rating:.1f}" if rating > 0 else "-"

        # Format quality
        quality_score = r.get("quality_score", 0)
        quality_str = f"{quality_score:.2f}" if quality_score else "-"

        table.add_row(
            str(r.get("resource_id", ""))[:10] + "...",
            r.get("name", "")[:28],
            r.get("resource_type", "")[:14],
            quality_str,
            price_str,
            rating_str,
            tags_str,
        )

    console.print(table)

    # Show pagination info
    current_page = data.get("page", page)
    total_pages = data.get("total_pages", 1)
    if total_pages > 1:
        console.print(f"\n[dim]Page {current_page} of {total_pages}. Use --page to navigate.[/dim]")


@marketplace.command()
@click.argument("resource-id")
@click.option("--api-url", default=None, help="PRSM API URL (default: from stored credentials)")
def show(resource_id: str, api_url: Optional[str]) -> None:
    """Show detailed information about a marketplace resource.

    \b
    Example:
        prsm marketplace show abc123-def456
    """
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    try:
        response = httpx.get(
            f"{url}/api/v1/marketplace/resources/{resource_id}",
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 404:
        console.print(f"❌ Resource not found: {resource_id}", style="red")
        raise SystemExit(1)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    elif response.status_code != 200:
        console.print(f"❌ Failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)

    r = response.json()

    table = Table(title=f"Resource: {r.get('name', 'Unknown')}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    # Format price
    price = r.get("price", 0)
    pricing_model = r.get("pricing_model", "free")
    if pricing_model == "free":
        price_str = "Free"
    elif pricing_model == "freemium":
        price_str = f"Freemium (Premium: {price:.2f} FTNS)"
    else:
        price_str = f"{price:.2f} FTNS"

    # Format rating
    rating = r.get("average_rating", 0)
    rating_str = f"⭐ {rating:.2f}/5.0" if rating > 0 else "No ratings"

    # Format tags
    resource_tags = r.get("tags", [])
    tags_str = ", ".join(resource_tags) if resource_tags else "None"

    # Format dates
    created = r.get("created_at", "")
    created_str = created[:19] if created else "N/A"

    table.add_row("ID", str(r.get("resource_id", "N/A")))
    table.add_row("Name", r.get("name", "N/A"))
    table.add_row("Type", r.get("resource_type", "N/A"))
    table.add_row("Quality Score", f"{r.get('quality_score', 0):.2f}")
    table.add_row("Status", r.get("status", "N/A"))
    table.add_row("Provider", str(r.get("provider_id", "N/A"))[:20] + "...")
    table.add_row("Price", price_str)
    table.add_row("Pricing Model", pricing_model.title())
    table.add_row("License", r.get("license", "N/A"))
    table.add_row("Rating", rating_str)
    table.add_row("Downloads", str(r.get("download_count", 0)))
    table.add_row("Tags", tags_str)
    table.add_row("Description", (r.get("description") or "No description")[:60] + "...")
    table.add_row("Docs URL", r.get("documentation_url") or "N/A")
    table.add_row("Source URL", r.get("source_url") or "N/A")
    table.add_row("Created", created_str)

    console.print(table)

    # Print full description below
    description = r.get("description")
    if description and len(description) > 60:
        console.print(f"\n[bold]Full Description:[/bold]")
        console.print(description)


@marketplace.command()
@click.argument("resource-id")
@click.option("--order-type", default="purchase", type=click.Choice(["purchase", "rent", "subscription"]), help="Order type")
@click.option("--quantity", default=1, type=int, help="Quantity to purchase")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
@click.option("--api-url", default=None, help="PRSM API URL (default: from stored credentials)")
def buy(resource_id: str, order_type: str, quantity: int, yes: bool, api_url: Optional[str]) -> None:
    """Purchase a marketplace resource.

    \b
    Examples:
        prsm marketplace buy abc123-def456
        prsm marketplace buy abc123-def456 --order-type subscription --yes
        prsm marketplace buy abc123-def456 --quantity 5
    """
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    # First, fetch resource details for confirmation
    try:
        detail_response = httpx.get(
            f"{url}/api/v1/marketplace/resources/{resource_id}",
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if detail_response.status_code == 404:
        console.print(f"❌ Resource not found: {resource_id}", style="red")
        raise SystemExit(1)
    elif detail_response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    elif detail_response.status_code != 200:
        console.print(f"❌ Failed to fetch resource: HTTP {detail_response.status_code}", style="red")
        raise SystemExit(1)

    resource = detail_response.json()
    resource_name = resource.get("name", "Unknown")
    resource_type = resource.get("resource_type", "Unknown")
    price = resource.get("price", 0)
    pricing_model = resource.get("pricing_model", "paid")

    # Calculate total cost
    if pricing_model == "free":
        total_cost = 0
    else:
        total_cost = price * quantity

    # Show confirmation prompt
    if not yes:
        console.print(f"\n📦 Purchase Confirmation:", style="bold cyan")
        console.print(f"   Resource : {resource_name}")
        console.print(f"   Type     : {resource_type}")
        console.print(f"   Order    : {order_type}")
        console.print(f"   Quantity : {quantity}")
        console.print(f"   Cost     : {total_cost:.2f} FTNS" if total_cost > 0 else "   Cost     : Free")
        console.print()

        if not click.confirm("Proceed with purchase?", default=True):
            console.print("Purchase cancelled.", style="yellow")
            raise SystemExit(0)

    # Place the order
    console.print(f"🛒 Placing order...", style="bold blue")

    try:
        response = httpx.post(
            f"{url}/api/v1/marketplace/orders",
            json={
                "resource_id": resource_id,
                "order_type": order_type,
                "quantity": quantity,
            },
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        console.print("✅ Purchase successful!", style="bold green")
        console.print(f"   Order ID: {data.get('order_id', 'N/A')}")
        console.print(f"   Status  : {data.get('status', 'N/A')}")
        console.print(f"   Total   : {data.get('total_cost', 0):.2f} FTNS")
        if data.get("access_url"):
            console.print(f"   Access  : {data.get('access_url')}")
    elif response.status_code == 402:
        console.print("❌ Insufficient FTNS balance", style="red")
        console.print("💡 Check your balance: prsm ftns balance", style="yellow")
        raise SystemExit(1)
    elif response.status_code == 404:
        console.print(f"❌ Resource not found: {resource_id}", style="red")
        raise SystemExit(1)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Purchase failed: HTTP {response.status_code}", style="red")
        console.print(f"   {response.text[:200]}")
        raise SystemExit(1)


@marketplace.command()
@click.option("--name", required=True, help="Resource name")
@click.option("--description", required=True, help="Resource description")
@click.option("--type", "resource_type", required=True, help="Resource type (ai_model, dataset, etc.)")
@click.option("--price", default=0.0, type=float, help="Price in FTNS (default: 0 for free)")
@click.option("--quality", default=None, type=float, help="Quality score (0.0-1.0)")
@click.option("--pricing-model", default="free", type=click.Choice(["free", "paid", "freemium"]), help="Pricing model")
@click.option("--tags", default=None, help="Comma-separated tags")
@click.option("--license", default="MIT", help="License identifier (default: MIT)")
@click.option("--docs-url", default=None, help="Documentation URL")
@click.option("--source-url", default=None, help="Source code/repository URL")
@click.option("--api-url", default=None, help="PRSM API URL (default: from stored credentials)")
def publish(name: str, description: str, resource_type: str, price: float, quality: Optional[float],
            pricing_model: str, tags: Optional[str], license: str, docs_url: Optional[str],
            source_url: Optional[str], api_url: Optional[str]) -> None:
    """Publish a new resource to the marketplace.

    \b
    Examples:
        prsm marketplace publish --name "My Model" --description "A great model" --type ai_model
        prsm marketplace publish --name "Dataset" --description "Training data" --type dataset --price 10.0
        prsm marketplace publish --name "Tool" --description "AI utility" --type tool --tags nlp,python
    """
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    # Build request body
    body: dict = {
        "name": name,
        "description": description,
        "resource_type": resource_type,
        "price": price,
        "pricing_model": pricing_model,
        "license": license,
    }

    if quality is not None:
        body["quality_score"] = quality
    if tags:
        body["tags"] = [t.strip() for t in tags.split(",")]
    if docs_url:
        body["documentation_url"] = docs_url
    if source_url:
        body["source_url"] = source_url

    console.print(f"📤 Publishing '{name}' to marketplace...", style="bold blue")

    try:
        response = httpx.post(
            f"{url}/api/v1/marketplace/resources",
            json=body,
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        console.print("✅ Resource published successfully!", style="bold green")
        console.print(f"   Resource ID: {data.get('resource_id', 'N/A')}")
        console.print(f"   Name       : {data.get('name', name)}")
        console.print(f"   Type       : {data.get('resource_type', resource_type)}")
        console.print(f"   Status     : {data.get('status', 'pending')}")
        console.print(f"   Price      : {price:.2f} FTNS" if price > 0 else "   Price      : Free")
    elif response.status_code == 403:
        console.print("❌ Permission denied. You need enhanced create permissions.", style="red")
        raise SystemExit(1)
    elif response.status_code == 400:
        detail = response.json().get("detail", response.text)
        console.print(f"❌ Validation error: {detail}", style="red")
        raise SystemExit(1)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Publish failed: HTTP {response.status_code}", style="red")
        console.print(f"   {response.text[:200]}")
        raise SystemExit(1)


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
        from rich.progress import Progress, BarColumn

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
# REPUTATION COMMANDS (Phase 2)
# ============================================================================

@main.group()
def reputation():
    """Reputation and trust score commands"""
    pass


@reputation.command()
@click.argument("user-id", required=False)
@click.option("--api-url", default=None, help="PRSM API URL")
def score(user_id: Optional[str], api_url: str):
    """Show reputation score and breakdown for a user."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    # If no user_id provided, get current user's score
    target = user_id or "me"

    try:
        response = httpx.get(
            f"{url}/api/v1/reputation/{target}",
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()

        from rich.panel import Panel

        content = f"""User:              {data.get('user_id', 'N/A')}
Overall Score:     {data.get('overall_score', 0):.1f} / 100
──────────────────────────────
Contribution:      {data.get('scores', {}).get('contribution', 0):.1f}   (data/model uploads, citations)
Reliability:       {data.get('scores', {}).get('reliability', 0):.1f}   (uptime, proof-of-storage pass rate)
Governance:        {data.get('scores', {}).get('governance', 0):.1f}   (proposal quality, voting participation)
Marketplace:       {data.get('scores', {}).get('marketplace', 0):.1f}   (buyer/seller ratings)
──────────────────────────────
Tier:              {data.get('tier', 'Unknown')}
Percentile:        Top {data.get('percentile', 100)}%"""

        console.print(Panel(content, title="⭐ Reputation Score", border_style="yellow"))
    elif response.status_code == 404:
        console.print(f"❌ User not found: {target}", style="red")
        raise SystemExit(1)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Score failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@reputation.command()
@click.option("--category", help="Filter by: contribution, reliability, governance, marketplace")
@click.option("--limit", type=int, default=20, help="Number of entries")
@click.option("--api-url", default=None, help="PRSM API URL")
def leaderboard(category: Optional[str], limit: int, api_url: str):
    """Show the top contributors on the network by reputation score."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    params = {"limit": limit}
    if category:
        params["category"] = category

    try:
        response = httpx.get(
            f"{url}/api/v1/reputation/leaderboard",
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
        entries = data.get("leaderboard", [])

        if not entries:
            console.print("No entries found.", style="yellow")
            return

        table = Table(title="🏆 Reputation Leaderboard")
        table.add_column("Rank", style="yellow", justify="right")
        table.add_column("User ID", style="cyan")
        table.add_column("Score", style="green", justify="right")
        table.add_column("Tier", style="magenta")
        table.add_column("Uploads", style="blue", justify="right")
        table.add_column("Uptime", style="dim")
        table.add_column("Since", style="dim")

        for i, e in enumerate(entries, 1):
            user_id = e.get("user_id", "unknown")
            short_id = user_id[:12] + "..." if len(user_id) > 12 else user_id
            table.add_row(
                str(i),
                short_id,
                f"{e.get('score', 0):.1f}",
                e.get("tier", "?"),
                str(e.get("uploads", 0)),
                f"{e.get('uptime', 0):.1f}%",
                e.get("member_since", "?")[:10] if e.get("member_since") else "N/A",
            )

        console.print(table)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Leaderboard failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@reputation.command()
@click.argument("user-id", required=False)
@click.option("--limit", type=int, default=25, help="Number of entries")
@click.option("--api-url", default=None, help="PRSM API URL")
def history(user_id: Optional[str], limit: int, api_url: str):
    """Show reputation change history."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)
    target = user_id or "me"

    try:
        response = httpx.get(
            f"{url}/api/v1/reputation/{target}/history",
            params={"limit": limit},
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        events = data.get("history", [])

        if not events:
            console.print("No history found.", style="yellow")
            return

        table = Table(title="📜 Reputation History")
        table.add_column("Date", style="dim")
        table.add_column("Event", style="green")
        table.add_column("Category", style="magenta")
        table.add_column("Δ Score", style="yellow", justify="right")
        table.add_column("Running Total", style="blue", justify="right")

        for e in events:
            delta = e.get("delta", 0)
            delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
            table.add_row(
                e.get("date", "?")[:10],
                e.get("event", "?")[:30],
                e.get("category", "?"),
                delta_str,
                f"{e.get('running_total', 0):.1f}",
            )

        console.print(table)
    elif response.status_code == 404:
        console.print(f"❌ User not found: {target}", style="red")
        raise SystemExit(1)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ History failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@reputation.command()
@click.argument("user-id")
@click.option("--reason", type=click.Choice(["fake_seeder", "bad_data", "spam", "governance_abuse", "other"]),
              required=True, help="Reason for report")
@click.option("--evidence", help="CID or transaction ID supporting the report")
@click.option("--description", help="Free-form explanation")
@click.option("--api-url", default=None, help="PRSM API URL")
def report(user_id: str, reason: str, evidence: Optional[str], description: Optional[str], api_url: str):
    """File a reputation report against a node."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    payload = {
        "target_user_id": user_id,
        "reason": reason,
    }
    if evidence:
        payload["evidence"] = evidence
    if description:
        payload["description"] = description

    console.print(f"📝 Filing report against {user_id[:12]}...", style="bold blue")

    if not click.confirm("Submit this report?", default=True):
        console.print("Cancelled.", style="yellow")
        raise SystemExit(0)

    try:
        response = httpx.post(
            f"{url}/api/v1/reputation/reports",
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
        console.print(f"✅ Report filed. Report ID: {data.get('report_id', 'N/A')}", style="bold green")
        console.print("   PRSM governance will review within 72 hours.", style="dim")
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    elif response.status_code == 404:
        console.print(f"❌ User not found: {user_id}", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Report failed: HTTP {response.status_code}", style="red")
        console.print(f"   {response.text[:200]}")
        raise SystemExit(1)


@reputation.command()
@click.option("--filed", is_flag=True, help="Reports I filed (default)")
@click.option("--received", is_flag=True, help="Reports filed against me")
@click.option("--status-filter", "status", help="Filter: pending, reviewed, dismissed, upheld")
@click.option("--limit", type=int, default=20, help="Number of entries")
@click.option("--api-url", default=None, help="PRSM API URL")
def reports(filed: bool, received: bool, status: Optional[str], limit: int, api_url: str):
    """List reputation reports filed by or against the current user."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    params = {"limit": limit}
    if received:
        params["type"] = "received"
    else:
        params["type"] = "filed"
    if status:
        params["status"] = status

    try:
        response = httpx.get(
            f"{url}/api/v1/reputation/reports",
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
        reports_list = data.get("reports", [])

        if not reports_list:
            console.print("No reports found.", style="yellow")
            return

        table = Table(title="📋 Reputation Reports")
        table.add_column("Report ID", style="cyan")
        table.add_column("Target/Reporter", style="green")
        table.add_column("Reason", style="magenta")
        table.add_column("Status", style="yellow")
        table.add_column("Filed At", style="dim")

        for r in reports_list:
            report_id = r.get("report_id", "?")[:12]
            other_party = r.get("target_user_id") if filed else r.get("reporter_user_id")
            short_other = other_party[:12] + "..." if other_party and len(other_party) > 12 else other_party
            table.add_row(
                report_id,
                short_other or "?",
                r.get("reason", "?"),
                r.get("status", "?"),
                r.get("filed_at", "?")[:10] if r.get("filed_at") else "?",
            )

        console.print(table)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Reports failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


# ============================================================================
# NWTN COMMANDS (Phase 3)
# ============================================================================

@main.group()
def nwtn():
    """NWTN reasoning orchestrator commands"""
    pass


@nwtn.command()
@click.argument("query")
@click.option("--model", help="Model backend: auto, gpt-4o, claude-3-5-sonnet, ollama/<name>")
@click.option("--context", help="Additional context JSON or plain text")
@click.option("--user-id", help="Override user ID")
@click.option("--trace", is_flag=True, help="Show full reasoning trace after response")
@click.option("--budget", type=float, help="Max FTNS to spend on this query")
@click.option("--format", "output_format", type=click.Choice(["text", "json", "markdown"]), default="text",
               help="Output format")
@click.option("--api-url", default=None, help="PRSM API URL")
def query_cmd(query: str, model: Optional[str], context: Optional[str], user_id: Optional[str],
              trace: bool, budget: Optional[float], output_format: str, api_url: str):
    """Submit a query to NWTN with enhanced control options."""
    import httpx
    import time

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    payload = {"query": query}
    if model:
        payload["model"] = model
    if context:
        payload["context"] = context
    if user_id:
        payload["user_id"] = user_id
    if budget:
        payload["max_budget"] = budget
    payload["output_format"] = output_format

    start_time = time.time()

    try:
        with console.status("[bold green]Processing query..."):
            response = httpx.post(
                f"{url}/api/v1/nwtn/query",
                json=payload,
                headers=headers,
                timeout=120.0,
            )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    elapsed = time.time() - start_time

    if response.status_code == 200:
        data = response.json()

        console.print("\n─────────── Response ───────────", style="bold blue")
        console.print(data.get("response", "No response"))
        console.print("────────────────────────────────", style="dim")
        console.print(f"Tokens used: {data.get('tokens_used', 'N/A')}   "
                     f"Cost: {data.get('cost_ftns', 0):.4f} FTNS   "
                     f"Time: {elapsed:.1f}s", style="dim")

        if trace and data.get("reasoning_trace"):
            console.print("\n─────── Reasoning Trace ─────────", style="bold yellow")
            for i, step in enumerate(data["reasoning_trace"], 1):
                console.print(f"Step {i}: [{step.get('type', '?')}] {step.get('description', '')}")
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    elif response.status_code == 402:
        console.print("❌ Insufficient FTNS balance", style="red")
        console.print("💡 Check your balance: prsm ftns balance", style="yellow")
        raise SystemExit(1)
    else:
        console.print(f"❌ Query failed: HTTP {response.status_code}", style="red")
        console.print(f"   {response.text[:200]}")
        raise SystemExit(1)


@nwtn.command()
@click.option("--available", is_flag=True, help="Show only currently reachable backends")
@click.option("--detail", is_flag=True, help="Include model metadata")
@click.option("--api-url", default=None, help="PRSM API URL")
def models(available: bool, detail: bool, api_url: str):
    """List available model backends and their status."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    params = {}
    if available:
        params["available_only"] = "true"
    if detail:
        params["include_metadata"] = "true"

    try:
        response = httpx.get(
            f"{url}/api/v1/nwtn/models",
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
        models_list = data.get("models", [])

        if not models_list:
            console.print("No models found.", style="yellow")
            return

        table = Table(title="🤖 Available Models")
        table.add_column("Backend", style="cyan")
        table.add_column("Model", style="green")
        table.add_column("Status", style="magenta")
        table.add_column("Latency", style="yellow")
        table.add_column("Cost/1k tokens", style="blue")
        table.add_column("Context Window", style="dim")

        for m in models_list:
            status = "✅" if m.get("available") else "❌"
            table.add_row(
                m.get("backend", "?"),
                m.get("model", "?")[:25],
                status,
                f"{m.get('latency_ms', 0)}ms" if m.get("latency_ms") else "N/A",
                f"{m.get('cost_per_1k', 0):.4f}" if m.get("cost_per_1k") else "N/A",
                str(m.get("context_window", "N/A")),
            )

        console.print(table)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Models failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@nwtn.command()
@click.option("--api-url", default=None, help="PRSM API URL")
def health(api_url: str):
    """Show NWTN pipeline health — all five reasoning layers."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    try:
        response = httpx.get(
            f"{url}/api/v1/nwtn/health",
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()

        from rich.panel import Panel

        layers = data.get("layers", {})
        layer_lines = []
        for i, (name, info) in enumerate(layers.items(), 1):
            status_icon = "✅" if info.get("healthy") else "⚠️ "
            status_text = "Healthy" if info.get("healthy") else info.get("status", "Degraded")
            latency = info.get("avg_latency_ms", 0)
            layer_lines.append(f"Layer {i} - {name:12s}: {status_icon} {status_text}  (avg {latency}ms)")

        overall_icon = "✅" if data.get("overall_healthy") else "⚠️ "
        overall_text = "Healthy" if data.get("overall_healthy") else "Degraded"

        content = "\n".join(layer_lines) + f"""
──────────────────────────────────────
Pipeline overall:          {overall_icon} {overall_text}"""

        console.print(Panel(content, title="🧠 NWTN Pipeline Health", border_style="blue"))
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Health check failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@nwtn.command()
@click.option("--limit", type=int, default=20, help="Number of sessions")
@click.option("--include-cost", is_flag=True, help="Show FTNS cost per session")
@click.option("--api-url", default=None, help="PRSM API URL")
def sessions(limit: int, include_cost: bool, api_url: str):
    """List recent NWTN query sessions."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    try:
        response = httpx.get(
            f"{url}/api/v1/sessions",
            params={"limit": limit},
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        sessions_list = data.get("sessions", [])

        if not sessions_list:
            console.print("No sessions found.", style="yellow")
            return

        table = Table(title="📜 NWTN Sessions")
        table.add_column("Session ID", style="cyan")
        table.add_column("Query", style="green")
        table.add_column("Model", style="magenta")
        table.add_column("Tokens", style="blue", justify="right")
        if include_cost:
            table.add_column("Cost", style="yellow", justify="right")
        table.add_column("Time", style="dim")
        table.add_column("Date", style="dim")

        for s in sessions_list:
            session_id = s.get("session_id", "?")[:12]
            query_text = s.get("query", "?")[:25] + "..." if len(s.get("query", "?")) > 25 else s.get("query", "?")
            table.add_row(
                session_id,
                query_text,
                s.get("model", "?")[:15],
                str(s.get("tokens_used", 0)),
                *([f"{s.get('cost_ftns', 0):.4f}"] if include_cost else []),
                f"{s.get('duration_seconds', 0):.1f}s",
                s.get("created_at", "?")[:10] if s.get("created_at") else "?",
            )

        console.print(table)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Sessions failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


# ============================================================================
# DISTILLATION COMMANDS (Phase 4)
# ============================================================================

@main.group()
def distill():
    """Model distillation commands"""
    pass


@distill.command("run")
@click.argument("teacher-id")
@click.option("--student-arch", help="Target architecture")
@click.option("--dataset-cid", help="IPFS CID of training dataset")
@click.option("--epochs", type=int, default=3, help="Training epochs")
@click.option("--output-name", help="Name for the resulting student model")
@click.option("--budget", type=float, help="Max FTNS to spend")
@click.option("--follow", "-f", is_flag=True, help="Stream progress logs until completion")
@click.option("--api-url", default=None, help="PRSM API URL")
def distill_run(teacher_id: str, student_arch: Optional[str], dataset_cid: Optional[str],
                epochs: int, output_name: Optional[str], budget: Optional[float],
                follow: bool, api_url: str):
    """Start a distillation job from a teacher model."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    payload = {
        "teacher_id": teacher_id,
        "epochs": epochs,
    }
    if student_arch:
        payload["student_architecture"] = student_arch
    if dataset_cid:
        payload["dataset_cid"] = dataset_cid
    if output_name:
        payload["output_name"] = output_name
    if budget:
        payload["max_budget"] = budget

    console.print(f"🔬 Starting distillation from {teacher_id[:12]}...", style="bold blue")

    try:
        response = httpx.post(
            f"{url}/api/v1/distillation/jobs",
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
        job_id = data.get("job_id", "")

        console.print(f"✅ Distillation job started: {job_id}", style="bold green")
        console.print(f"   Teacher:  {teacher_id}")
        console.print(f"   Student:  {data.get('student_architecture', 'auto')}")
        console.print(f"   Dataset:  {dataset_cid or 'default'}")
        console.print(f"   Use: prsm distill status {job_id}  to monitor", style="dim")
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    elif response.status_code == 402:
        console.print("❌ Insufficient FTNS balance", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Distill failed: HTTP {response.status_code}", style="red")
        console.print(f"   {response.text[:200]}")
        raise SystemExit(1)


@distill.command()
@click.argument("job-id")
@click.option("--api-url", default=None, help="PRSM API URL")
def status(job_id: str, api_url: str):
    """Poll distillation job status."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    try:
        response = httpx.get(
            f"{url}/api/v1/distillation/jobs/{job_id}",
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()

        from rich.panel import Panel

        progress = data.get("progress", 0) * 100
        progress_bar = "█" * int(progress / 5) + "░" * (20 - int(progress / 5))
        epoch_info = f"Epoch {data.get('current_epoch', 0)}/{data.get('total_epochs', 0)}"

        content = f"""Job ID:       {job_id}
Status:       {data.get('status', 'Unknown')}
Progress:     [{progress_bar}] {progress:.1f}%   {epoch_info}
Teacher:      {data.get('teacher_id', 'N/A')}
Student:      {data.get('student_architecture', 'N/A')}
Loss:         {data.get('current_loss', 'N/A')}
Started:      {data.get('started_at', 'N/A')}
ETA:          {data.get('eta_minutes', 'calculating...')}"""

        console.print(Panel(content, title="🔬 Distillation Job", border_style="green"))
    elif response.status_code == 404:
        console.print(f"❌ Job not found: {job_id}", style="red")
        raise SystemExit(1)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Status failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@distill.command("list")
@click.option("--status-filter", "status", help="Filter: running, completed, failed")
@click.option("--limit", type=int, default=20, help="Number of entries")
@click.option("--api-url", default=None, help="PRSM API URL")
def distill_list(status: Optional[str], limit: int, api_url: str):
    """List distillation jobs for the current user."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    params = {"limit": limit}
    if status:
        params["status"] = status

    try:
        response = httpx.get(
            f"{url}/api/v1/distillation/jobs",
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
        jobs = data.get("jobs", [])

        if not jobs:
            console.print("No distillation jobs found.", style="yellow")
            return

        table = Table(title="🔬 Distillation Jobs")
        table.add_column("Job ID", style="cyan")
        table.add_column("Teacher", style="green")
        table.add_column("Student Arch", style="magenta")
        table.add_column("Status", style="yellow")
        table.add_column("Progress", style="blue")
        table.add_column("Cost", style="dim")
        table.add_column("Date", style="dim")

        for j in jobs:
            job_id = j.get("job_id", "?")[:12]
            teacher = j.get("teacher_id", "?")[:12]
            progress = j.get("progress", 0) * 100
            table.add_row(
                job_id,
                teacher,
                j.get("student_architecture", "?")[:15],
                j.get("status", "?"),
                f"{progress:.0f}%",
                f"{j.get('cost_ftns', 0):.4f}",
                j.get("created_at", "?")[:10] if j.get("created_at") else "?",
            )

        console.print(table)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ List failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@distill.command()
@click.argument("job-id")
@click.option("--yes", is_flag=True, help="Skip confirmation")
@click.option("--api-url", default=None, help="PRSM API URL")
def cancel(job_id: str, yes: bool, api_url: str):
    """Cancel a running distillation job."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    if not yes:
        if not click.confirm(f"Cancel distillation job {job_id[:12]}?", default=False):
            console.print("Cancelled.", style="yellow")
            raise SystemExit(0)

    try:
        response = httpx.delete(
            f"{url}/api/v1/distillation/jobs/{job_id}",
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        console.print(f"✅ Distillation job cancelled: {job_id[:12]}", style="bold green")
    elif response.status_code == 404:
        console.print(f"❌ Job not found: {job_id}", style="red")
        raise SystemExit(1)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Cancel failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@distill.command()
@click.option("--limit", type=int, default=20, help="Number of entries")
@click.option("--api-url", default=None, help="PRSM API URL")
def models(limit: int, api_url: str):
    """List student models produced by completed distillation jobs."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    try:
        response = httpx.get(
            f"{url}/api/v1/distillation/models",
            params={"limit": limit},
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        models_list = data.get("models", [])

        if not models_list:
            console.print("No distilled models found.", style="yellow")
            return

        table = Table(title="🎓 Distilled Models")
        table.add_column("Model ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Teacher Source", style="magenta")
        table.add_column("Architecture", style="blue")
        table.add_column("Size", style="yellow")
        table.add_column("Quality", style="dim")
        table.add_column("Date", style="dim")

        for m in models_list:
            model_id = m.get("model_id", "?")[:12]
            teacher = m.get("teacher_source", "?")[:12]
            table.add_row(
                model_id,
                m.get("name", "?")[:20],
                teacher,
                m.get("architecture", "?")[:15],
                _fmt_bytes(m.get("size_bytes", 0)),
                f"{m.get('quality_score', 0):.1f}",
                m.get("created_at", "?")[:10] if m.get("created_at") else "?",
            )

        console.print(table)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Models failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


# ============================================================================
# TEACHER COMMAND ADDITIONS (Phase 5)
# ============================================================================

@teacher.command()
@click.argument("teacher-id")
@click.option("--yes", is_flag=True, help="Skip confirmation")
@click.option("--api-url", default=None, help="PRSM API URL")
def delete(teacher_id: str, yes: bool, api_url: str):
    """Delete a teacher model."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    if not yes:
        if not click.confirm(f"Delete teacher model {teacher_id[:12]}?", default=False):
            console.print("Cancelled.", style="yellow")
            raise SystemExit(0)

    try:
        response = httpx.delete(
            f"{url}/api/v1/teachers/{teacher_id}",
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        console.print(f"✅ Teacher model {teacher_id[:12]} deleted.", style="bold green")
        console.print("   ⚠️  Note: Active distillation jobs referencing this teacher will be cancelled.", style="yellow")
    elif response.status_code == 404:
        console.print(f"❌ Teacher not found: {teacher_id}", style="red")
        raise SystemExit(1)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Delete failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@teacher.command("list-runs")
@click.argument("teacher-id")
@click.option("--limit", type=int, default=20, help="Number of entries")
@click.option("--api-url", default=None, help="PRSM API URL")
def list_runs(teacher_id: str, limit: int, api_url: str):
    """List all training runs for a teacher model."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    try:
        response = httpx.get(
            f"{url}/api/v1/teachers/{teacher_id}/runs",
            params={"limit": limit},
            headers=headers,
            timeout=10.0,
        )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()
        runs = data.get("runs", [])

        if not runs:
            console.print(f"No training runs found for teacher {teacher_id[:12]}", style="yellow")
            return

        table = Table(title=f"Training Runs for {teacher_id[:12]}")
        table.add_column("Run ID", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Epochs", style="magenta")
        table.add_column("Loss", style="yellow")
        table.add_column("Duration", style="blue")
        table.add_column("Date", style="dim")

        for r in runs:
            run_id = r.get("run_id", "?")[:12]
            table.add_row(
                run_id,
                r.get("status", "?"),
                f"{r.get('completed_epochs', 0)}/{r.get('total_epochs', 0)}",
                f"{r.get('final_loss', 0):.4f}" if r.get("final_loss") else "N/A",
                r.get("duration", "N/A"),
                r.get("started_at", "?")[:10] if r.get("started_at") else "?",
            )

        console.print(table)
    elif response.status_code == 404:
        console.print(f"❌ Teacher not found: {teacher_id}", style="red")
        raise SystemExit(1)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ List-runs failed: HTTP {response.status_code}", style="red")
        raise SystemExit(1)


@teacher.command()
@click.argument("teacher-id")
@click.option("--benchmark", type=click.Choice(["default", "math", "code", "science"]), default="default",
              help="Benchmark suite")
@click.option("--samples", type=int, default=100, help="Number of eval samples")
@click.option("--api-url", default=None, help="PRSM API URL")
def eval(teacher_id: str, benchmark: str, samples: int, api_url: str):
    """Run a quick benchmark evaluation against a teacher model."""
    import httpx

    headers = _auth_headers()
    if not headers:
        console.print("❌ Not logged in. Run: prsm login", style="red")
        raise SystemExit(1)

    url = _api_url_from_creds(api_url)

    payload = {
        "benchmark": benchmark,
        "samples": samples,
    }

    console.print(f"📊 Evaluating teacher {teacher_id[:12]}...", style="bold blue")

    try:
        with console.status("[bold green]Running evaluation..."):
            response = httpx.post(
                f"{url}/api/v1/teachers/{teacher_id}/eval",
                json=payload,
                headers=headers,
                timeout=120.0,
            )
    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to {url}", style="red")
        console.print("💡 Start the server: prsm serve", style="yellow")
        raise SystemExit(1)

    if response.status_code == 200:
        data = response.json()

        from rich.panel import Panel

        content = f"""Teacher:     {teacher_id}
Benchmark:   {benchmark}
Score:       {data.get('score', 0):.1f} / 100
Accuracy:    {data.get('accuracy', 0):.1f}%
Latency:     avg {data.get('latency_ms', 0)}ms
Cost:        {data.get('cost_ftns', 0):.4f} FTNS
Rank:        Top {data.get('percentile', 100)}% in domain"""

        console.print(Panel(content, title="📊 Evaluation Results", border_style="green"))
    elif response.status_code == 404:
        console.print(f"❌ Teacher not found: {teacher_id}", style="red")
        raise SystemExit(1)
    elif response.status_code == 401:
        console.print("❌ Session expired. Run: prsm login", style="red")
        raise SystemExit(1)
    else:
        console.print(f"❌ Eval failed: HTTP {response.status_code}", style="red")
        console.print(f"   {response.text[:200]}")
        raise SystemExit(1)


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
@click.option("--service", help="Filter by service: api, nwtn, ipfs, bittorrent, node")
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


if __name__ == "__main__":
    main()
