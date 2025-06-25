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

from prsm.core.config import get_settings


console = Console()
settings = get_settings()


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
    
    if reload and workers > 1:
        console.print("⚠️  Cannot use --reload with multiple workers", style="yellow")
        workers = 1
    
    uvicorn.run(
        "prsm.api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="info" if not settings.debug else "debug"
    )


@main.command()
def status():
    """Show PRSM system status"""
    table = Table(title="PRSM System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Details", style="green")
    
    # Check basic configuration
    table.add_row("Configuration", "✅ Loaded", f"Environment: {settings.environment}")
    table.add_row("Database", "🔄 Checking...", settings.database_url)
    table.add_row("IPFS", "🔄 Checking...", f"{settings.ipfs_host}:{settings.ipfs_port}")
    table.add_row("NWTN", "✅ Enabled" if settings.nwtn_enabled else "❌ Disabled", 
                  f"Model: {settings.nwtn_default_model}")
    table.add_row("FTNS", "✅ Enabled" if settings.ftns_enabled else "❌ Disabled",
                  f"Initial grant: {settings.ftns_initial_grant}")
    
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
def start():
    """Start P2P node"""
    console.print("🌐 Starting P2P node...", style="bold green")
    console.print("🚧 P2P networking coming in v0.3.0", style="yellow")
    console.print("💡 Currently operates in single-node mode", style="blue")


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