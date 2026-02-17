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


@node.command()
@click.option("--wizard", is_flag=True, help="Run interactive setup wizard")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def start(wizard: bool, host: str, port: int):
    """Start a PRSM node (single-node mode)"""
    if wizard:
        console.print("🧙 Welcome to the PRSM Node Setup Wizard", style="bold magenta")
        console.print("This will configure and start a PRSM node in single-node mode.\n")
        console.print("Note: P2P networking is in development. Your node runs locally for now.\n", style="dim")

        # 1. Identity
        use_sro = click.confirm("🔗 Would you like to link your ORCID for identity?")
        if use_sro:
            orcid = click.prompt("   Enter your ORCID ID")
            console.print(f"   ✅ ORCID noted: {orcid} (will be used when P2P launches)", style="green")
        else:
            console.print("   👤 Starting as anonymous user.", style="blue")

        # 2. Contribution
        mode = click.prompt(
            "🧠 Choose your node role",
            type=click.Choice(["full", "compute", "verify"]),
            default="full"
        )
        console.print(f"   ✅ Configured as {mode} node.", style="green")

        console.print(f"\n✅ Configuration complete. Starting PRSM API server on {host}:{port}...", style="bold green")
        console.print("   P2P peer discovery will be enabled in a future release.", style="dim")
    else:
        console.print(f"🌐 Starting PRSM node in single-node mode on {host}:{port}...", style="bold green")

    # Initialize config and start the server
    _init_config()
    uvicorn.run(
        "prsm.interface.api.main:app",
        host=host,
        port=port,
        log_level="info" if not _get_debug() else "debug"
    )


@node.command()
def peers():
    """List connected peers"""
    console.print("🌐 Connected peers:", style="bold blue")
    console.print("🚧 P2P networking coming in v0.3.0", style="yellow")
    console.print("💡 Currently operates in single-node mode", style="blue")


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