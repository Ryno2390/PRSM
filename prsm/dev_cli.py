"""
Enhanced PRSM Development CLI Tool
One-command setup and development automation for PRSM

🎯 PURPOSE:
Addresses consulting report finding: "Complex setup process vs. simple 'pip install' expectations"
Provides one-command developer onboarding and environment management.

🚀 QUICK START:
    prsm-dev setup         # Complete one-command setup
    prsm-dev quickstart    # Fast hello-world example
    prsm-dev status        # Check system health
"""

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import platform
import json

import click
import docker
import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.tree import Tree
from rich.live import Live
from rich.layout import Layout
from rich.syntax import Syntax

console = Console()


class PRSMDevSetup:
    """Enhanced PRSM development environment setup manager"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.prsm_dir = self.project_root / "prsm"
        self.config_dir = self.project_root / "config"
        self.system_info = self._get_system_info()
        self.docker_client = None
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for optimal setup"""
        return {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python_version": sys.version_info,
            "node_available": shutil.which("node") is not None,
            "docker_available": shutil.which("docker") is not None,
            "go_available": shutil.which("go") is not None,
            "git_available": shutil.which("git") is not None,
        }
    
    def check_prerequisites(self) -> bool:
        """Check and install prerequisites"""
        console.print("\n🔍 Checking prerequisites...", style="bold blue")
        
        missing = []
        
        # Python version check
        if self.system_info["python_version"] < (3, 8):
            missing.append("Python 3.8+")
        
        # Required tools check
        if not self.system_info["docker_available"]:
            missing.append("Docker")
        
        if not self.system_info["git_available"]:
            missing.append("Git")
        
        if missing:
            console.print(f"\n❌ Missing prerequisites: {', '.join(missing)}", style="red")
            console.print("\nPlease install the following:")
            for item in missing:
                if item == "Docker":
                    console.print("  • Docker: https://docs.docker.com/get-docker/")
                elif item == "Git":
                    console.print("  • Git: https://git-scm.com/downloads")
                elif "Python" in item:
                    console.print(f"  • {item}: https://www.python.org/downloads/")
            return False
        
        console.print("✅ All prerequisites met!", style="green")
        return True
    
    def setup_python_environment(self) -> bool:
        """Setup Python virtual environment and dependencies"""
        console.print("\n🐍 Setting up Python environment...", style="bold blue")
        
        try:
            # Create virtual environment if it doesn't exist
            venv_path = self.project_root / ".venv"
            if not venv_path.exists():
                console.print("Creating virtual environment...")
                subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
            
            # Determine pip path
            if self.system_info["platform"] == "Windows":
                pip_path = venv_path / "Scripts" / "pip"
                python_path = venv_path / "Scripts" / "python"
            else:
                pip_path = venv_path / "bin" / "pip"
                python_path = venv_path / "bin" / "python"
            
            # Upgrade pip
            console.print("Upgrading pip...")
            subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], check=True)
            
            # Install requirements
            requirements_file = self.project_root / "requirements.txt"
            if requirements_file.exists():
                console.print("Installing PRSM dependencies...")
                subprocess.run([str(pip_path), "install", "-r", str(requirements_file)], check=True)
            
            # Install development dependencies
            dev_requirements = self.project_root / "requirements-dev.txt"
            if dev_requirements.exists():
                console.print("Installing development dependencies...")
                subprocess.run([str(pip_path), "install", "-r", str(dev_requirements)], check=True)
            
            # Install PRSM in development mode
            console.print("Installing PRSM in development mode...")
            subprocess.run([str(pip_path), "install", "-e", "."], check=True)
            
            console.print("✅ Python environment ready!", style="green")
            return True
            
        except subprocess.CalledProcessError as e:
            console.print(f"❌ Failed to setup Python environment: {e}", style="red")
            return False
    
    def setup_configuration(self) -> bool:
        """Setup configuration files"""
        console.print("\n⚙️ Setting up configuration...", style="bold blue")
        
        try:
            # Copy environment template
            env_example = self.config_dir / "api_keys.env.example"
            env_file = self.config_dir / "api_keys.env"
            
            if env_example.exists() and not env_file.exists():
                console.print("Creating API keys configuration...")
                shutil.copy2(env_example, env_file)
                console.print(f"✅ Created {env_file}", style="green")
                console.print("💡 Edit this file to add your API keys", style="blue")
            
            # Create .env in root if needed
            root_env = self.project_root / ".env"
            if not root_env.exists():
                console.print("Creating root environment file...")
                root_env.write_text("""# PRSM Development Environment
# Auto-generated by prsm-dev

# Development mode
PRSM_ENV=development
DEBUG=true

# Database (using SQLite for development)
DATABASE_URL=sqlite+aiosqlite:///./prsm_dev.db

# Redis (using default local instance)
REDIS_URL=redis://localhost:6379/0

# IPFS (using local node)
IPFS_HOST=localhost
IPFS_PORT=5001

# API Keys (edit config/api_keys.env)
API_KEYS_FILE=config/api_keys.env

# FTNS Configuration
FTNS_ENABLED=true
FTNS_INITIAL_GRANT=1000.0

# NWTN Configuration  
NWTN_ENABLED=true
NWTN_DEFAULT_MODEL=gpt-3.5-turbo

# Safety settings for development
SAFETY_LEVEL=moderate
CIRCUIT_BREAKER_ENABLED=true
\"\"\")
                console.print(f"✅ Created {root_env}", style="green")
            
            console.print("✅ Configuration ready!", style="green")
            return True
            
        except Exception as e:
            console.print(f"❌ Failed to setup configuration: {e}", style="red")
            return False
    
    def setup_docker_services(self) -> bool:
        """Setup Docker services (Redis, IPFS, etc.)"""
        console.print("\n🐳 Setting up Docker services...", style="bold blue")
        
        try:
            self.docker_client = docker.from_env()
            
            # Check for available Docker configurations (in order of preference)
            compose_options = [
                ("docker-compose.quickstart.yml", "quickstart", "Quickstart (30s)"),
                ("docker-compose.onboarding.yml", "onboarding", "Onboarding (1min)"),
                ("docker-compose.dev.yml", "development", "Development (3min)")
            ]
            
            selected_compose = None
            for compose_file, env_type, description in compose_options:
                compose_path = self.project_root / compose_file
                if compose_path.exists():
                    console.print(f"📁 Found {compose_file} - {description}")
                    
                    # For setup automation, prefer quickstart for speed
                    if env_type == "quickstart":
                        selected_compose = (compose_path, env_type, description)
                        break
                    elif selected_compose is None:
                        selected_compose = (compose_path, env_type, description)
            
            if not selected_compose:
                console.print("❌ No Docker Compose files found", style="red")
                console.print("💡 Available options:", style="blue")
                console.print("   • docker-compose.quickstart.yml - Fastest setup")
                console.print("   • docker-compose.onboarding.yml - Balanced features")
                console.print("   • docker-compose.dev.yml - Full development")
                return False
            
            compose_path, env_type, description = selected_compose
            console.print(f"🚀 Using {env_type} environment: {description}")
            
            # Start services based on environment type
            if env_type == "quickstart":
                services = []  # Start all services in quickstart
                console.print("Starting minimal services (Redis + IPFS)...")
            elif env_type == "onboarding":
                services = []  # Start all services in onboarding
                console.print("Starting onboarding services...")
            else:
                services = ["redis", "ipfs"]  # Only essential services for dev
                console.print("Starting essential services (Redis + IPFS)...")
            
            cmd = ["docker-compose", "-f", str(compose_path), "up", "-d"] + services
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print("✅ Docker services started!", style="green")
                
                # Wait for services to be ready
                console.print("Waiting for services to be ready...")
                time.sleep(3 if env_type == "quickstart" else 5)
                
                # Test Redis connection
                if self._test_redis():
                    console.print("✅ Redis is ready", style="green")
                else:
                    console.print("⚠️ Redis connection test failed", style="yellow")
                
                # Test IPFS connection
                if self._test_ipfs():
                    console.print("✅ IPFS is ready", style="green")
                else:
                    console.print("⚠️ IPFS connection test failed", style="yellow")
                
                # Show helpful information
                console.print(f"\n💡 Using {env_type} Docker environment")
                if env_type != "quickstart":
                    console.print("🔧 For faster setup next time, use quickstart:")
                    console.print(f"   docker-compose -f docker-compose.quickstart.yml up -d")
                
                return True
            else:
                console.print(f"❌ Failed to start Docker services: {result.stderr}", style="red")
                # Provide helpful fallback suggestions
                console.print("\n🔧 Troubleshooting suggestions:", style="blue")
                console.print(f"   1. Manual start: docker-compose -f {compose_path.name} up -d")
                console.print("   2. Check Docker is running: docker ps")
                console.print("   3. Try quickstart: scripts/docker-helper.sh start quickstart")
                return False
                
        except Exception as e:
            console.print(f"❌ Docker setup failed: {e}", style="red")
            console.print("\n🔧 Alternative setup options:", style="blue")
            console.print("   • Use Docker helper: scripts/docker-helper.sh start quickstart")
            console.print("   • Manual setup: docker-compose -f docker-compose.quickstart.yml up -d")
            console.print("   • Check Docker installation: docker --version")
            return False
    
    def _test_redis(self) -> bool:
        """Test Redis connection"""
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            return True
        except:
            return False
    
    def _test_ipfs(self) -> bool:
        """Test IPFS connection"""
        try:
            response = requests.get("http://localhost:5001/api/v0/version", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def initialize_database(self) -> bool:
        """Initialize database with Alembic migrations"""
        console.print("\n🗄️ Initializing database...", style="bold blue")
        
        try:
            # Check if we're in a virtual environment
            venv_path = self.project_root / ".venv"
            if venv_path.exists():
                if self.system_info["platform"] == "Windows":
                    python_path = venv_path / "Scripts" / "python"
                else:
                    python_path = venv_path / "bin" / "python"
            else:
                python_path = sys.executable
            
            # Run database initialization
            console.print("Running Alembic migrations...")
            result = subprocess.run([
                str(python_path), "-m", "alembic", "upgrade", "head"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                console.print("✅ Database initialized!", style="green")
                return True
            else:
                # If migrations fail, try creating tables directly
                console.print("⚠️ Alembic migration failed, trying direct table creation...", style="yellow")
                
                # Import and run database initialization
                sys.path.insert(0, str(self.project_root))
                try:
                    from prsm.core.database import init_database
                    asyncio.run(init_database())
                    console.print("✅ Database tables created!", style="green")
                    return True
                except Exception as e:
                    console.print(f"❌ Database initialization failed: {e}", style="red")
                    return False
                    
        except Exception as e:
            console.print(f"❌ Database setup failed: {e}", style="red")
            return False
    
    def run_health_check(self) -> Dict[str, bool]:
        """Run comprehensive health check"""
        console.print("\n🏥 Running health check...", style="bold blue")
        
        health_status = {}
        
        # Check Python environment
        try:
            import prsm
            health_status["python_env"] = True
            console.print("✅ Python environment: OK", style="green")
        except ImportError:
            health_status["python_env"] = False
            console.print("❌ Python environment: FAIL", style="red")
        
        # Check Docker services
        health_status["redis"] = self._test_redis()
        health_status["ipfs"] = self._test_ipfs()
        
        if health_status["redis"]:
            console.print("✅ Redis: OK", style="green")
        else:
            console.print("❌ Redis: FAIL", style="red")
        
        if health_status["ipfs"]:
            console.print("✅ IPFS: OK", style="green")
        else:
            console.print("❌ IPFS: FAIL", style="red")
        
        # Check configuration files
        config_files = [
            self.project_root / ".env",
            self.config_dir / "api_keys.env"
        ]
        
        config_ok = all(f.exists() for f in config_files)
        health_status["configuration"] = config_ok
        
        if config_ok:
            console.print("✅ Configuration: OK", style="green")
        else:
            console.print("❌ Configuration: FAIL", style="red")
        
        # Check database
        try:
            db_file = self.project_root / "prsm_dev.db"
            health_status["database"] = db_file.exists()
            if health_status["database"]:
                console.print("✅ Database: OK", style="green")
            else:
                console.print("❌ Database: FAIL", style="red")
        except:
            health_status["database"] = False
            console.print("❌ Database: FAIL", style="red")
        
        return health_status
    
    def full_setup(self) -> bool:
        """Run complete setup process"""
        console.print(Panel.fit(
            "🚀 PRSM Development Environment Setup",
            style="bold blue"
        ))
        
        steps = [
            ("Prerequisites", self.check_prerequisites),
            ("Python Environment", self.setup_python_environment),
            ("Configuration", self.setup_configuration),
            ("Docker Services", self.setup_docker_services),
            ("Database", self.initialize_database),
        ]
        
        failed_steps = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            for step_name, step_func in steps:
                task = progress.add_task(f"Setting up {step_name}...", total=None)
                
                try:
                    success = step_func()
                    if success:
                        progress.update(task, description=f"✅ {step_name} complete")
                    else:
                        progress.update(task, description=f"❌ {step_name} failed")
                        failed_steps.append(step_name)
                except Exception as e:
                    progress.update(task, description=f"❌ {step_name} error: {str(e)[:50]}")
                    failed_steps.append(step_name)
                
                progress.remove_task(task)
        
        # Final health check
        health_status = self.run_health_check()
        all_healthy = all(health_status.values())
        
        if all_healthy and not failed_steps:
            console.print(Panel.fit(
                "🎉 PRSM Development Environment Ready!\n\n"
                "Next steps:\n"
                "1. Edit config/api_keys.env with your API keys\n"
                "2. Run: prsm-dev quickstart\n"
                "3. Run: prsm serve",
                style="bold green",
                title="Setup Complete"
            ))
            return True
        else:
            console.print(Panel.fit(
                f"⚠️ Setup completed with issues\n\n"
                f"Failed steps: {', '.join(failed_steps) if failed_steps else 'None'}\n"
                f"Health check: {sum(health_status.values())}/{len(health_status)} OK\n\n"
                "Run 'prsm-dev diagnose' for detailed troubleshooting",
                style="yellow",
                title="Setup Partial"
            ))
            return False


# CLI Commands
@click.group()
@click.version_option(version="0.1.0", prog_name="prsm-dev")
def main():
    """
    🚀 PRSM Development Environment Manager
    
    One-command setup and automation for PRSM development.
    """
    pass


@main.command()
@click.option("--skip-docker", is_flag=True, help="Skip Docker services setup")
@click.option("--skip-db", is_flag=True, help="Skip database initialization")
def setup(skip_docker: bool, skip_db: bool):
    """Complete one-command PRSM development setup"""
    setup_manager = PRSMDevSetup()
    
    if skip_docker:
        console.print("⏭️ Skipping Docker services setup", style="yellow")
    if skip_db:
        console.print("⏭️ Skipping database initialization", style="yellow")
    
    success = setup_manager.full_setup()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


@main.command()
def status():
    """Check PRSM development environment status"""
    setup_manager = PRSMDevSetup()
    health_status = setup_manager.run_health_check()
    
    table = Table(title="PRSM Development Environment Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Details")
    
    for component, status in health_status.items():
        status_icon = "✅" if status else "❌"
        status_text = "OK" if status else "FAIL"
        table.add_row(
            component.replace("_", " ").title(),
            f"{status_icon} {status_text}",
            ""
        )
    
    console.print(table)
    
    healthy_count = sum(health_status.values())
    total_count = len(health_status)
    
    if healthy_count == total_count:
        console.print(f"\n🎉 All systems operational ({healthy_count}/{total_count})", style="green")
    else:
        console.print(f"\n⚠️ {total_count - healthy_count} issues detected ({healthy_count}/{total_count})", style="yellow")
        console.print("Run 'prsm-dev diagnose' for troubleshooting", style="blue")


@main.command()
def quickstart():
    """Run a quick Hello World example"""
    console.print(Panel.fit(
        "🚀 PRSM Quickstart Example",
        style="bold green"
    ))
    
    # Check if environment is ready
    setup_manager = PRSMDevSetup()
    health_status = setup_manager.run_health_check()
    
    if not all(health_status.values()):
        console.print("❌ Environment not ready. Run 'prsm-dev setup' first.", style="red")
        return
    
    # Create quickstart script
    quickstart_script = '''
"""
PRSM Quickstart Example
A simple Hello World to test your PRSM installation
"""
import asyncio
from prsm.core.models import UserInput, PRSMSession
from prsm.nwtn.orchestrator import NWTNOrchestrator

async def main():
    print("🧠 Initializing PRSM...")
    
    # Create NWTN orchestrator
    orchestrator = NWTNOrchestrator()
    
    # Create user session
    session = PRSMSession(
        user_id="quickstart_user",
        nwtn_context_allocation=100
    )
    
    # Simple query
    user_input = UserInput(
        user_id="quickstart_user",
        prompt="Hello PRSM! Please introduce yourself.",
        context_allocation=50
    )
    
    print("🚀 Submitting query to PRSM...")
    try:
        response = await orchestrator.process_query(user_input)
        
        print("\\n✅ PRSM Response:")
        print(f"Content: {response.final_answer}")
        print(f"FTNS Cost: {response.ftns_charged}")
        print(f"Processing Time: {response.processing_time:.2f}s")
        
        if response.reasoning_trace:
            print("\\n🧭 Reasoning Trace:")
            for i, step in enumerate(response.reasoning_trace, 1):
                print(f"  {i}. {step}")
        
        print("\\n🎉 PRSM is working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you've configured API keys in config/api_keys.env")

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    # Write and run quickstart
    quickstart_file = Path("quickstart_example.py")
    quickstart_file.write_text(quickstart_script)
    
    console.print("🏃 Running quickstart example...")
    console.print("(This will test your PRSM installation with a simple query)")
    
    try:
        # Run the quickstart script
        venv_path = Path.cwd() / ".venv"
        if venv_path.exists():
            if platform.system() == "Windows":
                python_path = venv_path / "Scripts" / "python"
            else:
                python_path = venv_path / "bin" / "python"
        else:
            python_path = sys.executable
        
        result = subprocess.run([str(python_path), str(quickstart_file)], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            console.print(result.stdout)
        else:
            console.print("❌ Quickstart failed:", style="red")
            console.print(result.stderr)
            console.print("\n💡 Troubleshooting tips:", style="blue")
            console.print("1. Check API keys in config/api_keys.env")
            console.print("2. Run 'prsm-dev status' to check system health")
            console.print("3. Run 'prsm-dev diagnose' for detailed diagnostics")
        
        # Clean up
        quickstart_file.unlink()
        
    except Exception as e:
        console.print(f"❌ Failed to run quickstart: {e}", style="red")


@main.command()
def diagnose():
    """Run comprehensive diagnostics"""
    console.print(Panel.fit(
        "🔍 PRSM Development Environment Diagnostics",
        style="bold blue"
    ))
    
    setup_manager = PRSMDevSetup()
    
    # System information
    console.print("\n💻 System Information:", style="bold")
    info_table = Table()
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")
    
    for key, value in setup_manager.system_info.items():
        info_table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(info_table)
    
    # Detailed health check
    console.print("\n🏥 Detailed Health Check:", style="bold")
    health_status = setup_manager.run_health_check()
    
    # Docker services detailed check
    console.print("\n🐳 Docker Services:", style="bold")
    try:
        docker_client = docker.from_env()
        containers = docker_client.containers.list(all=True)
        
        docker_table = Table()
        docker_table.add_column("Container", style="cyan")
        docker_table.add_column("Status", style="magenta")
        docker_table.add_column("Ports")
        
        for container in containers:
            if any(name in container.name for name in ["redis", "ipfs", "prsm"]):
                status_color = "green" if container.status == "running" else "red"
                ports = ", ".join([f"{p['HostPort']}:{p['PrivatePort']}" 
                                 for p in container.ports.values() 
                                 for p in p if p]) if container.ports else "None"
                docker_table.add_row(
                    container.name,
                    f"[{status_color}]{container.status}[/{status_color}]",
                    ports
                )
        
        console.print(docker_table)
        
    except Exception as e:
        console.print(f"❌ Docker check failed: {e}", style="red")
    
    # Configuration files check
    console.print("\n📄 Configuration Files:", style="bold")
    config_files = [
        Path.cwd() / ".env",
        Path.cwd() / "config" / "api_keys.env",
        Path.cwd() / "requirements.txt",
        Path.cwd() / "docker-compose.dev.yml"
    ]
    
    config_table = Table()
    config_table.add_column("File", style="cyan")
    config_table.add_column("Status", style="magenta")
    config_table.add_column("Size")
    
    for file_path in config_files:
        if file_path.exists():
            size = f"{file_path.stat().st_size} bytes"
            config_table.add_row(
                str(file_path.relative_to(Path.cwd())),
                "[green]✅ Exists[/green]",
                size
            )
        else:
            config_table.add_row(
                str(file_path.relative_to(Path.cwd())),
                "[red]❌ Missing[/red]",
                "-"
            )
    
    console.print(config_table)
    
    # Recommendations
    console.print("\n💡 Recommendations:", style="bold blue")
    issues = []
    
    if not health_status.get("python_env", False):
        issues.append("Run 'prsm-dev setup' to fix Python environment")
    
    if not health_status.get("redis", False):
        issues.append("Start Redis: docker-compose -f docker-compose.dev.yml up -d redis")
    
    if not health_status.get("ipfs", False):
        issues.append("Start IPFS: docker-compose -f docker-compose.dev.yml up -d ipfs")
    
    if not health_status.get("configuration", False):
        issues.append("Run 'prsm-dev setup' to create configuration files")
    
    if not health_status.get("database", False):
        issues.append("Initialize database: alembic upgrade head")
    
    if issues:
        for i, issue in enumerate(issues, 1):
            console.print(f"  {i}. {issue}")
    else:
        console.print("🎉 No issues detected! Your environment looks good.")


@main.command()
@click.option("--service", type=click.Choice(["redis", "ipfs", "all"]), default="all")
def start(service: str):
    """Start development services"""
    console.print(f"🚀 Starting {service} service(s)...", style="bold green")
    
    try:
        if service == "all":
            services = ["redis", "ipfs"]
        else:
            services = [service]
        
        result = subprocess.run([
            "docker-compose", "-f", "docker-compose.dev.yml", "up", "-d"
        ] + services, capture_output=True, text=True)
        
        if result.returncode == 0:
            console.print(f"✅ {service.title()} service(s) started!", style="green")
        else:
            console.print(f"❌ Failed to start services: {result.stderr}", style="red")
    
    except Exception as e:
        console.print(f"❌ Error starting services: {e}", style="red")


@main.command()
@click.option("--service", type=click.Choice(["redis", "ipfs", "all"]), default="all")
def stop(service: str):
    """Stop development services"""
    console.print(f"🛑 Stopping {service} service(s)...", style="bold yellow")
    
    try:
        if service == "all":
            result = subprocess.run([
                "docker-compose", "-f", "docker-compose.dev.yml", "down"
            ], capture_output=True, text=True)
        else:
            result = subprocess.run([
                "docker-compose", "-f", "docker-compose.dev.yml", "stop", service
            ], capture_output=True, text=True)
        
        if result.returncode == 0:
            console.print(f"✅ {service.title()} service(s) stopped!", style="green")
        else:
            console.print(f"❌ Failed to stop services: {result.stderr}", style="red")
    
    except Exception as e:
        console.print(f"❌ Error stopping services: {e}", style="red")


@main.command()
def clean():
    """Clean development environment (remove containers, temp files)"""
    console.print("🧹 Cleaning development environment...", style="bold yellow")
    
    if not Confirm.ask("This will remove Docker containers and temp files. Continue?"):
        console.print("Cancelled.", style="yellow")
        return
    
    try:
        # Stop and remove containers
        result = subprocess.run([
            "docker-compose", "-f", "docker-compose.dev.yml", "down", "-v"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            console.print("✅ Docker containers removed", style="green")
        
        # Clean temp files
        temp_files = [
            Path.cwd() / "prsm_dev.db",
            Path.cwd() / "quickstart_example.py",
        ]
        
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()
                console.print(f"✅ Removed {temp_file.name}", style="green")
        
        console.print("🎉 Environment cleaned!", style="green")
        
    except Exception as e:
        console.print(f"❌ Error cleaning environment: {e}", style="red")


if __name__ == "__main__":
    main()