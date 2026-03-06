"""
PRSM Command Line Interface
Main entry point for PRSM CLI commands
"""

import asyncio
import socket
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
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
def query(query: str, context: int, user_id: str):
    """Submit a query to NWTN (requires running server)"""
    console.print(f"🧠 Submitting query to NWTN...", style="bold blue")
    console.print(f"Query: {query}")
    console.print(f"Context allocation: {context} FTNS")
    
    # This would connect to the running PRSM server
    console.print("🚧 Coming Soon - Full NWTN orchestration will be available in v0.2.0", style="yellow")
    console.print("💡 For now, you can:", style="blue")
    console.print("   • Start the API server: prsm server start", style="blue")
    console.print("   • Test endpoints: curl http://localhost:8000/health", style="blue")


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
def db_upgrade():
    """Upgrade database to latest schema"""
    console.print("🗄️  Upgrading database schema...", style="bold blue")
    console.print("🚧 Database migrations coming in v0.2.0", style="yellow")
    console.print("💡 Current setup uses automatic table creation", style="blue")


@main.command()
def db_downgrade():
    """Downgrade database schema by one version"""
    console.print("🗄️  Downgrading database schema...", style="bold blue")
    console.print("🚧 Database migrations coming in v0.2.0", style="yellow")
    console.print("💡 Current setup uses automatic table creation", style="blue")


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
def start(wizard: bool, p2p_port: int, api_port: int, bootstrap: str, no_dashboard: bool):
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


@main.group()
def teacher():
    """Teacher model management commands"""
    pass


@teacher.command()
def list():
    """List available teacher models"""
    console.print("🎓 Available teacher models:", style="bold blue")
    console.print("🚧 CLI teacher management coming in v0.2.0", style="yellow")
    console.print("💡 Teacher models available via API endpoints", style="blue")


@teacher.command()
@click.argument("specialization")
def create(specialization: str):
    """Create a new teacher model"""
    console.print(f"🎓 Creating teacher model for {specialization}...", style="bold green")
    console.print("🚧 CLI teacher management coming in v0.2.0", style="yellow")
    console.print("💡 Teacher models available via API endpoints", style="blue")


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


@compute.command()
@click.option('--limit', default=10, type=int, help='Maximum number of jobs to list')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def list(limit: int, api_url: str):
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
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def balance(api_url: str):
    """Show FTNS token balance."""
    import httpx
    
    try:
        response = httpx.get(f"{api_url}/api/v1/ftns/balance", timeout=10.0)
        
        if response.status_code == 200:
            data = response.json()
            table = Table(title="FTNS Balance")
            table.add_column("Type", style="cyan")
            table.add_column("Amount", style="green")
            table.add_row("Total", f"{data.get('total_balance', 0):.4f}")
            table.add_row("Available", f"{data.get('available_balance', 0):.4f}")
            table.add_row("Reserved", f"{data.get('reserved_balance', 0):.4f}")
            if 'staked_balance' in data:
                table.add_row("Staked", f"{data.get('staked_balance', 0):.4f}")
            console.print(table)
        else:
            console.print(f"❌ Failed to get balance: {response.status_code}", style="red")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@ftns.command()
@click.option('--to', required=True, help='Recipient address')
@click.option('--amount', required=True, type=float, help='Amount to transfer')
@click.option('--memo', help='Transaction memo')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def transfer(to: str, amount: float, memo: str, api_url: str):
    """Transfer FTNS tokens to another address."""
    import httpx
    
    console.print(f"💸 Transferring {amount} FTNS to {to}...", style="bold blue")
    
    try:
        response = httpx.post(
            f"{api_url}/api/v1/ftns/transfer",
            json={"to_address": to, "amount": amount, "memo": memo},
            timeout=10.0
        )
        
        if response.status_code == 200:
            data = response.json()
            console.print(f"✅ Transfer successful!", style="bold green")
            console.print(f"   Transaction ID: {data.get('transaction_id')}")
            console.print(f"   Amount: {data.get('amount')} FTNS")
            console.print(f"   Fee: {data.get('fee', 0):.4f} FTNS")
        else:
            console.print(f"❌ Transfer failed: {response.status_code}", style="red")
            console.print(f"   {response.text}")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@ftns.command()
@click.option('--amount', required=True, type=float, help='Amount to stake')
@click.option('--lock-period', default=30, type=int, help='Lock period in days')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def stake(amount: float, lock_period: int, api_url: str):
    """Stake FTNS tokens for network participation."""
    import httpx
    
    console.print(f"🔒 Staking {amount} FTNS for {lock_period} days...", style="bold blue")
    
    try:
        response = httpx.post(
            f"{api_url}/api/v1/ftns/stake",
            json={"amount": amount, "lock_period": lock_period},
            timeout=10.0
        )
        
        if response.status_code == 200:
            data = response.json()
            console.print(f"✅ Staking successful!", style="bold green")
            console.print(f"   Staked: {data.get('staked_amount')} FTNS")
            console.print(f"   APY: {data.get('apy', 0) * 100:.1f}%")
            console.print(f"   Rewards earned: {data.get('rewards_earned', 0):.4f} FTNS")
        else:
            console.print(f"❌ Staking failed: {response.status_code}", style="red")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@ftns.command()
@click.option('--limit', default=20, type=int, help='Maximum transactions to show')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def history(limit: int, api_url: str):
    """Show FTNS transaction history."""
    import httpx
    
    try:
        response = httpx.get(f"{api_url}/api/v1/ftns/history?limit={limit}", timeout=10.0)
        
        if response.status_code == 200:
            data = response.json()
            transactions = data.get('transactions', [])
            
            if not transactions:
                console.print("No transactions found.", style="dim")
                return
            
            table = Table(title="FTNS Transaction History")
            table.add_column("ID", style="dim")
            table.add_column("Type", style="cyan")
            table.add_column("Amount", style="green")
            table.add_column("Status", style="magenta")
            table.add_column("Time", style="blue")
            
            for tx in transactions:
                table.add_row(
                    tx.get('transaction_id', 'N/A')[:12] + "...",
                    tx.get('transaction_type', 'N/A'),
                    f"{tx.get('amount', 0):.4f}",
                    tx.get('status', 'N/A'),
                    tx.get('timestamp', 'N/A')[:19]
                )
            console.print(table)
        else:
            console.print(f"❌ Failed to get history: {response.status_code}", style="red")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


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
@click.argument('file-path', type=click.Path(exists=True))
@click.option('--description', help='Content description')
@click.option('--tags', help='Comma-separated tags')
@click.option('--public', is_flag=True, help='Make content public')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def upload(file_path: str, description: str, tags: str, public: bool, api_url: str):
    """Upload a file to IPFS storage."""
    import httpx
    
    console.print(f"📤 Uploading {file_path}...", style="bold blue")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f)}
            data = {}
            if description:
                data['description'] = description
            if tags:
                data['tags'] = tags
            if public:
                data['is_public'] = 'true'
            
            response = httpx.post(
                f"{api_url}/api/v1/storage/upload",
                files=files,
                data=data,
                timeout=60.0
            )
        
        if response.status_code == 200:
            data = response.json()
            console.print(f"✅ Upload successful!", style="bold green")
            console.print(f"   CID: {data.get('cid')}")
            console.print(f"   Size: {data.get('size')} bytes")
            console.print(f"   Cost: {data.get('ftns_cost', 0):.4f} FTNS")
            console.print(f"   Gateway: {data.get('gateway_url')}")
        else:
            console.print(f"❌ Upload failed: {response.status_code}", style="red")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


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
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def proposals(limit: int, status: str, api_url: str):
    """List governance proposals."""
    import httpx
    
    params = f"limit={limit}"
    if status:
        params += f"&status={status}"
    
    try:
        response = httpx.get(f"{api_url}/api/v1/governance/proposals?{params}", timeout=10.0)
        
        if response.status_code == 200:
            data = response.json()
            proposals = data.get('proposals', [])
            
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
                votes = f"✓{prop.get('votes_yes', 0):.0f} ✗{prop.get('votes_no', 0):.0f}"
                table.add_row(
                    prop.get('proposal_id', 'N/A')[:12] + "...",
                    prop.get('title', 'N/A')[:30],
                    prop.get('status', 'N/A'),
                    prop.get('proposal_type', 'N/A'),
                    votes
                )
            console.print(table)
        else:
            console.print(f"❌ Failed to list proposals: {response.status_code}", style="red")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@governance.command()
@click.argument('proposal-id')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def proposal(proposal_id: str, api_url: str):
    """Get details of a specific proposal."""
    import httpx
    
    try:
        response = httpx.get(f"{api_url}/api/v1/governance/proposals/{proposal_id}", timeout=10.0)
        
        if response.status_code == 200:
            data = response.json()
            console.print(f"\n📋 {data.get('title')}", style="bold cyan")
            console.print(f"   ID: {data.get('proposal_id')}")
            console.print(f"   Status: {data.get('status')}")
            console.print(f"   Type: {data.get('proposal_type')}")
            console.print(f"   Proposer: {data.get('proposer', 'N/A')[:16]}...")
            console.print()
            console.print("Description:", style="bold")
            console.print(data.get('description', 'N/A'))
            console.print()
            
            table = Table(title="Voting Results")
            table.add_column("Choice", style="cyan")
            table.add_column("Votes (FTNS)", style="green")
            table.add_row("Yes", f"{data.get('votes_yes', 0):.2f}")
            table.add_row("No", f"{data.get('votes_no', 0):.2f}")
            table.add_row("Abstain", f"{data.get('votes_abstain', 0):.2f}")
            console.print(table)
            
            console.print(f"\n   Quorum required: {data.get('quorum', 0) * 100:.1f}%")
            console.print(f"   Threshold: {data.get('threshold', 0) * 100:.1f}%")
            console.print(f"   Voting ends: {data.get('voting_ends', 'N/A')}")
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
@click.option('--choice', required=True, type=click.Choice(['yes', 'no', 'abstain']), help='Vote choice')
@click.option('--reason', help='Reason for your vote')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def vote(proposal_id: str, choice: str, reason: str, api_url: str):
    """Cast a vote on a proposal."""
    import httpx
    
    console.print(f"🗳️  Casting vote '{choice}' on proposal {proposal_id}...", style="bold blue")
    
    try:
        response = httpx.post(
            f"{api_url}/api/v1/governance/proposals/{proposal_id}/vote",
            json={"choice": choice, "reason": reason},
            timeout=10.0
        )
        
        if response.status_code == 200:
            data = response.json()
            console.print(f"✅ Vote cast successfully!", style="bold green")
            console.print(f"   Vote ID: {data.get('vote_id')}")
            console.print(f"   Voting power: {data.get('voting_power', 0):.2f} FTNS")
        else:
            console.print(f"❌ Failed to cast vote: {response.status_code}", style="red")
            console.print(f"   {response.text}")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@governance.command()
@click.option('--title', required=True, help='Proposal title')
@click.option('--description', required=True, help='Proposal description')
@click.option('--type', 'proposal_type', required=True,
              type=click.Choice(['parameter_change', 'protocol_upgrade', 'treasury_spend',
                                'model_addition', 'model_removal', 'fee_adjustment',
                                'governance_change', 'other']),
              help='Type of proposal')
@click.option('--duration', default=7, type=int, help='Voting duration in days')
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def create_proposal(title: str, description: str, proposal_type: str, duration: int, api_url: str):
    """Create a new governance proposal."""
    import httpx
    
    console.print(f"📝 Creating proposal: {title}...", style="bold blue")
    
    try:
        response = httpx.post(
            f"{api_url}/api/v1/governance/proposals",
            json={
                "title": title,
                "description": description,
                "proposal_type": proposal_type,
                "duration_days": duration
            },
            timeout=10.0
        )
        
        if response.status_code == 200:
            data = response.json()
            console.print(f"✅ Proposal created!", style="bold green")
            console.print(f"   Proposal ID: {data.get('proposal_id')}")
            console.print(f"   Status: {data.get('status')}")
            console.print(f"   Voting starts: {data.get('voting_starts')}")
            console.print(f"   Voting ends: {data.get('voting_ends')}")
        else:
            console.print(f"❌ Failed to create proposal: {response.status_code}", style="red")
            console.print(f"   {response.text}")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@governance.command()
@click.option('--api-url', default='http://localhost:8000', help='PRSM API URL')
def voting_power(api_url: str):
    """Show your voting power."""
    import httpx
    
    try:
        response = httpx.get(f"{api_url}/api/v1/governance/voting-power", timeout=10.0)
        
        if response.status_code == 200:
            data = response.json()
            console.print(f"🗳️  Your voting power: {data.get('voting_power', 0):.2f} FTNS", style="bold green")
        else:
            console.print(f"❌ Failed to get voting power: {response.status_code}", style="red")
    except httpx.ConnectError:
        console.print("❌ Cannot connect to PRSM server", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


if __name__ == "__main__":
    main()
