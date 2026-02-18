"""
PRSM Command Line Interface
Main entry point for PRSM CLI commands
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
import uvicorn
from rich.console import Console
from rich.table import Table

console = Console()


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
@click.version_option(version="0.1.0", prog_name="PRSM")
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
        default="100.83.80.91:9001",
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
def start(wizard: bool, p2p_port: int, api_port: int, bootstrap: str):
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

    # Start the node
    console.print()
    console.print("=" * 60, style="bold green")
    console.print("  Starting PRSM Node", style="bold green")
    console.print("=" * 60, style="bold green")

    async def _run():
        from prsm.node.node import PRSMNode

        prsm_node = PRSMNode(config)
        await prsm_node.initialize()

        # Print status dashboard
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

        await prsm_node.start()

        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(1)
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


if __name__ == "__main__":
    main()