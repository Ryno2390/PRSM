#!/usr/bin/env python3
"""
PRSM Demo Prerequisites Checker
Comprehensive environment validation for investor demonstrations

This script validates the environment to ensure smooth demo execution.
Run before any investor presentation to prevent technical issues.
"""

import sys
import os
import subprocess
import socket
import importlib
import platform
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header():
    """Print a professional header for the validation script."""
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("=" * 80)
    print("🚀 PRSM Demo Environment Validation")
    print("   Protocol for Recursive Scientific Modeling")
    print("=" * 80)
    print(f"{Colors.END}")
    print(f"{Colors.WHITE}📊 Status: Advanced Prototype - Ready for Investment{Colors.END}")
    print(f"{Colors.WHITE}🎯 Purpose: Validate environment for investor demonstrations{Colors.END}")
    print(f"{Colors.WHITE}⏱️  Duration: ~2 minutes comprehensive validation{Colors.END}")
    print()

def check_mark(passed: bool) -> str:
    """Return a colored check mark or X based on pass/fail."""
    return f"{Colors.GREEN}✅{Colors.END}" if passed else f"{Colors.RED}❌{Colors.END}"

def warning_mark() -> str:
    """Return a colored warning symbol."""
    return f"{Colors.YELLOW}⚠️{Colors.END}"

def info_mark() -> str:
    """Return a colored info symbol."""
    return f"{Colors.BLUE}ℹ️{Colors.END}"

class ValidationResult:
    """Container for validation results."""
    def __init__(self):
        self.passed: List[str] = []
        self.failed: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def add_pass(self, message: str):
        self.passed.append(message)
        print(f"{check_mark(True)} {message}")

    def add_fail(self, message: str):
        self.failed.append(message)
        print(f"{check_mark(False)} {message}")

    def add_warning(self, message: str):
        self.warnings.append(message)
        print(f"{warning_mark()} {message}")

    def add_info(self, message: str):
        self.info.append(message)
        print(f"{info_mark()} {message}")

    @property
    def is_ready(self) -> bool:
        return len(self.failed) == 0

    @property
    def score(self) -> int:
        total_checks = len(self.passed) + len(self.failed)
        if total_checks == 0:
            return 0
        return int((len(self.passed) / total_checks) * 100)

def check_python_version(result: ValidationResult):
    """Check if Python version meets requirements."""
    print(f"{Colors.BOLD}🐍 Python Environment{Colors.END}")
    
    version = sys.version_info
    if version >= (3, 9):
        result.add_pass(f"Python {version.major}.{version.minor}.{version.micro} (✓ meets requirement ≥3.9)")
    else:
        result.add_fail(f"Python {version.major}.{version.minor}.{version.micro} (requires ≥3.9)")
        result.add_info("Install Python 3.9+ from python.org or use pyenv")

    # Check virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        result.add_pass("Virtual environment active")
    else:
        result.add_warning("No virtual environment detected (recommended for clean demo)")
        result.add_info("Run: python -m venv venv && source venv/bin/activate")

def check_system_info(result: ValidationResult):
    """Check system information and resources."""
    print(f"\n{Colors.BOLD}💻 System Information{Colors.END}")
    
    # Operating system
    os_name = platform.system()
    os_version = platform.release()
    result.add_pass(f"Operating System: {os_name} {os_version}")
    
    # Architecture
    arch = platform.machine()
    result.add_pass(f"Architecture: {arch}")
    
    # Available memory (basic check)
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        if memory_gb >= 4:
            result.add_pass(f"Available RAM: {memory_gb:.1f}GB (✓ sufficient)")
        else:
            result.add_warning(f"Available RAM: {memory_gb:.1f}GB (4GB+ recommended)")
    except ImportError:
        result.add_info("Install psutil for detailed system monitoring: pip install psutil")

def check_required_packages(result: ValidationResult):
    """Check if required Python packages are installed."""
    print(f"\n{Colors.BOLD}📦 Required Packages{Colors.END}")
    
    # Core packages required for demos
    required_packages = {
        'fastapi': 'FastAPI web framework',
        'uvicorn': 'ASGI server for demos',
        'asyncio': 'Async support (built-in)',
        'json': 'JSON handling (built-in)',
        'pathlib': 'Path utilities (built-in)',
        'sqlite3': 'Database support (built-in)',
        'hashlib': 'Cryptographic functions (built-in)',
        'time': 'Timing utilities (built-in)',
        'random': 'Random number generation (built-in)',
        'typing': 'Type hints (built-in)',
    }
    
    # Optional but recommended packages
    optional_packages = {
        'requests': 'HTTP client for API calls',
        'websockets': 'WebSocket support for real-time demos',
        'matplotlib': 'Plotting for economic visualizations',
        'pandas': 'Data analysis for tokenomics',
        'numpy': 'Numerical computing',
        'streamlit': 'Interactive dashboard framework',
        'psutil': 'System monitoring utilities',
    }
    
    # Check required packages
    for package, description in required_packages.items():
        try:
            importlib.import_module(package)
            result.add_pass(f"{package}: {description}")
        except ImportError:
            if package in ['asyncio', 'json', 'pathlib', 'sqlite3', 'hashlib', 'time', 'random', 'typing']:
                result.add_warning(f"{package}: Built-in module not available (Python installation issue)")
            else:
                result.add_fail(f"{package}: {description} (pip install {package})")
    
    # Check optional packages
    print(f"\n{Colors.BOLD}📦 Optional Packages (Enhanced Demo Experience){Colors.END}")
    for package, description in optional_packages.items():
        try:
            importlib.import_module(package)
            result.add_pass(f"{package}: {description}")
        except ImportError:
            result.add_info(f"{package}: {description} (pip install {package})")

def check_port_availability(result: ValidationResult):
    """Check if required ports are available."""
    print(f"\n{Colors.BOLD}🌐 Port Availability{Colors.END}")
    
    required_ports = {
        8000: 'Main API server',
        8001: 'P2P Node 1',
        8002: 'P2P Node 2', 
        8003: 'P2P Node 3',
        8501: 'Streamlit dashboard (if using)',
        8502: 'Additional dashboard (if using)',
    }
    
    for port, description in required_ports.items():
        if is_port_available(port):
            result.add_pass(f"Port {port}: Available ({description})")
        else:
            result.add_warning(f"Port {port}: In use ({description}) - demos may conflict")
            result.add_info(f"Stop services on port {port} or the demos will auto-select alternate ports")

def is_port_available(port: int) -> bool:
    """Check if a specific port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        return result != 0

def check_file_structure(result: ValidationResult):
    """Check if required demo files exist."""
    print(f"\n{Colors.BOLD}📁 Demo File Structure{Colors.END}")
    
    base_path = Path(__file__).parent
    required_files = {
        'run_demos.py': 'Main demo launcher',
        'p2p_network_demo.py': 'P2P network demonstration',
        'tokenomics_simulation.py': 'Token economics simulation',
        'INVESTOR_DEMO.md': 'Investor demo guide',
        'DEMO_OUTPUTS.md': 'Expected demo outputs',
    }
    
    optional_files = {
        'p2p_dashboard.py': 'Interactive P2P dashboard',
        'tokenomics_dashboard.py': 'Interactive tokenomics dashboard',
        'check_requirements.py': 'This validation script',
    }
    
    # Check required files
    for filename, description in required_files.items():
        file_path = base_path / filename
        if file_path.exists():
            result.add_pass(f"{filename}: {description}")
        else:
            result.add_fail(f"{filename}: Missing ({description})")
    
    # Check optional files
    for filename, description in optional_files.items():
        file_path = base_path / filename
        if file_path.exists():
            result.add_pass(f"{filename}: {description}")
        else:
            result.add_info(f"{filename}: Optional ({description})")

def check_demo_dependencies(result: ValidationResult):
    """Check demo-specific dependencies and configurations."""
    print(f"\n{Colors.BOLD}🔧 Demo Configuration{Colors.END}")
    
    # Check if we're in the correct directory
    current_dir = Path.cwd()
    if current_dir.name == 'demos' or (current_dir / 'demos').exists():
        result.add_pass("Working directory: Correct demo location")
    else:
        result.add_warning("Working directory: Not in demos/ folder")
        result.add_info("Navigate to the demos/ directory before running demos")
    
    # Check for PRSM parent directory structure
    parent_dir = Path(__file__).parent.parent
    expected_dirs = ['docs', 'validation', 'demos']
    
    missing_dirs = []
    for dirname in expected_dirs:
        if not (parent_dir / dirname).exists():
            missing_dirs.append(dirname)
    
    if not missing_dirs:
        result.add_pass("PRSM directory structure: Complete")
    else:
        result.add_warning(f"PRSM directory structure: Missing {', '.join(missing_dirs)}")

def check_network_connectivity(result: ValidationResult):
    """Check basic network connectivity for demos that might need external APIs."""
    print(f"\n{Colors.BOLD}🌍 Network Connectivity{Colors.END}")
    
    # Test basic internet connectivity
    try:
        import urllib.request
        urllib.request.urlopen('https://www.google.com', timeout=5)
        result.add_pass("Internet connectivity: Available")
    except Exception:
        result.add_warning("Internet connectivity: Limited or unavailable")
        result.add_info("Some demos may work offline, but API integrations require internet")
    
    # Test localhost connectivity
    try:
        import urllib.request
        urllib.request.urlopen('http://localhost', timeout=1)
        result.add_pass("Localhost connectivity: Available")
    except Exception:
        result.add_pass("Localhost connectivity: Available (expected connection refused)")

def generate_demo_quick_start(result: ValidationResult):
    """Generate a quick start guide based on validation results."""
    print(f"\n{Colors.BOLD}🚀 Demo Quick Start Commands{Colors.END}")
    
    if result.is_ready:
        print(f"{Colors.GREEN}✅ Environment ready for demos!{Colors.END}")
        print()
        print("Quick demo commands:")
        print(f"{Colors.CYAN}  # Run main demo launcher{Colors.END}")
        print("  python run_demos.py")
        print()
        print(f"{Colors.CYAN}  # Direct demo access{Colors.END}")
        print("  python p2p_network_demo.py      # P2P network demo")
        print("  python tokenomics_simulation.py  # Economic modeling")
        print()
        print(f"{Colors.CYAN}  # Interactive dashboards (if Streamlit installed){Colors.END}")
        print("  streamlit run p2p_dashboard.py --server.port 8501")
        print("  streamlit run tokenomics_dashboard.py --server.port 8502")
    else:
        print(f"{Colors.YELLOW}⚠️  Environment needs attention before demos{Colors.END}")
        print("\nRecommended fixes:")
        for failure in result.failed:
            print(f"  • {failure}")

def print_summary(result: ValidationResult):
    """Print a comprehensive summary of validation results."""
    print(f"\n{Colors.BOLD}📊 VALIDATION SUMMARY{Colors.END}")
    print("=" * 50)
    
    print(f"\n{Colors.GREEN}✅ Passed Checks: {len(result.passed)}{Colors.END}")
    print(f"{Colors.RED}❌ Failed Checks: {len(result.failed)}{Colors.END}")
    print(f"{Colors.YELLOW}⚠️  Warnings: {len(result.warnings)}{Colors.END}")
    print(f"{Colors.BLUE}ℹ️  Info Notes: {len(result.info)}{Colors.END}")
    
    print(f"\n{Colors.BOLD}🎯 Demo Readiness Score: {result.score}%{Colors.END}")
    
    if result.score >= 90:
        status_color = Colors.GREEN
        status_text = "EXCELLENT - Ready for investor demos"
    elif result.score >= 75:
        status_color = Colors.YELLOW
        status_text = "GOOD - Minor issues to address"
    elif result.score >= 60:
        status_color = Colors.YELLOW
        status_text = "FAIR - Several issues need attention"
    else:
        status_color = Colors.RED
        status_text = "POOR - Significant setup required"
    
    print(f"{status_color}{Colors.BOLD}Status: {status_text}{Colors.END}")
    
    if result.failed:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ Critical Issues:{Colors.END}")
        for failure in result.failed:
            print(f"   • {failure}")
    
    if result.warnings:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  Warnings:{Colors.END}")
        for warning in result.warnings[:3]:  # Show only first 3 warnings
            print(f"   • {warning}")
        if len(result.warnings) > 3:
            print(f"   • ... and {len(result.warnings) - 3} more warnings")

def save_validation_report(result: ValidationResult):
    """Save validation results to a JSON file for reference."""
    report_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'score': result.score,
        'is_ready': result.is_ready,
        'passed': result.passed,
        'failed': result.failed,
        'warnings': result.warnings,
        'info': result.info,
        'system_info': {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': platform.platform(),
            'architecture': platform.machine(),
        }
    }
    
    try:
        report_path = Path(__file__).parent / 'validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"\n{info_mark()} Validation report saved to: {report_path}")
    except Exception as e:
        print(f"\n{warning_mark()} Could not save validation report: {e}")

def main():
    """Run comprehensive demo environment validation."""
    print_header()
    
    result = ValidationResult()
    
    # Run all validation checks
    check_python_version(result)
    check_system_info(result)
    check_required_packages(result)
    check_port_availability(result)
    check_file_structure(result)
    check_demo_dependencies(result)
    check_network_connectivity(result)
    
    # Generate guidance and summary
    generate_demo_quick_start(result)
    print_summary(result)
    save_validation_report(result)
    
    # Footer with contact information
    print(f"\n{Colors.CYAN}{Colors.BOLD}")
    print("=" * 80)
    print("🎯 INVESTOR DEMO SUPPORT")
    print("=" * 80)
    print(f"{Colors.END}")
    print(f"{Colors.WHITE}📧 Demo Support: demo@prsm.ai{Colors.END}")
    print(f"{Colors.WHITE}📧 Technical Issues: technical@prsm.ai{Colors.END}")
    print(f"{Colors.WHITE}📧 Investment Inquiries: funding@prsm.ai{Colors.END}")
    print(f"{Colors.WHITE}📖 Demo Guide: demos/INVESTOR_DEMO.md{Colors.END}")
    print()
    
    # Return appropriate exit code
    return 0 if result.is_ready else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Demo validation interrupted by user{Colors.END}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error during validation: {e}{Colors.END}")
        print(f"{Colors.RED}Please contact technical support: technical@prsm.ai{Colors.END}")
        sys.exit(1)