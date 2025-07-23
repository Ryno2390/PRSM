"""
PRSM Test Runner
================

Comprehensive test runner with support for different test categories,
coverage reporting, performance benchmarking, and CI/CD integration.
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class PRSMTestRunner:
    """Main test runner for PRSM test suite"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.test_dir = self.project_root / "tests"
        self.results = {}
        
    def run_unit_tests(self, verbose: bool = False, coverage: bool = True) -> Dict[str, Any]:
        """Run unit tests"""
        print("🧪 Running unit tests...")
        
        cmd = [sys.executable, "-m", "pytest", "tests/unit/"]
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend([
                "--cov=prsm",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/unit",
                "--cov-report=xml:coverage_unit.xml"
            ])
        
        cmd.extend([
            "--tb=short",
            "--durations=10",
            "-m", "not slow"  # Exclude slow tests by default
        ])
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        end_time = time.time()
        
        return {
            "category": "unit",
            "success": result.returncode == 0,
            "duration": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    
    def run_integration_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run integration tests"""
        print("🔗 Running integration tests...")
        
        cmd = [sys.executable, "-m", "pytest", "tests/integration/"]
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend([
            "--tb=short",
            "--durations=10",
            "-m", "integration"
        ])
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        end_time = time.time()
        
        return {
            "category": "integration",
            "success": result.returncode == 0,
            "duration": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    
    def run_performance_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run performance tests and benchmarks"""
        print("⚡ Running performance tests...")
        
        cmd = [sys.executable, "-m", "pytest", "tests/performance/"]
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend([
            "--tb=short",
            "--durations=20",
            "-m", "performance or benchmark",
            "--benchmark-only",
            "--benchmark-json=benchmark_results.json"
        ])
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        end_time = time.time()
        
        return {
            "category": "performance",
            "success": result.returncode == 0,
            "duration": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    
    def run_security_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run security tests"""
        print("🔒 Running security tests...")
        
        cmd = [sys.executable, "-m", "pytest"]
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend([
            "--tb=short",
            "-m", "security",
            "tests/"
        ])
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        end_time = time.time()
        
        return {
            "category": "security",
            "success": result.returncode == 0,
            "duration": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    
    def run_api_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run API tests"""
        print("🌐 Running API tests...")
        
        cmd = [sys.executable, "-m", "pytest"]
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend([
            "--tb=short",
            "-m", "api",
            "tests/"
        ])
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        end_time = time.time()
        
        return {
            "category": "api",
            "success": result.returncode == 0,
            "duration": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    
    def run_load_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run load/stress tests"""
        print("🚀 Running load tests...")
        
        cmd = [sys.executable, "-m", "pytest"]
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend([
            "--tb=short",
            "-m", "load or stress",
            "tests/",
            "--durations=0"  # Show all durations for load tests
        ])
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        end_time = time.time()
        
        return {
            "category": "load",
            "success": result.returncode == 0,
            "duration": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    
    def run_all_tests(self, verbose: bool = False, coverage: bool = True) -> Dict[str, Any]:
        """Run all test categories"""
        print("🎯 Running complete test suite...")
        
        cmd = [sys.executable, "-m", "pytest", "tests/"]
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend([
                "--cov=prsm",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/complete",
                "--cov-report=xml:coverage_complete.xml"
            ])
        
        cmd.extend([
            "--tb=short",
            "--durations=20",
            "-m", "not slow"  # Exclude slow tests unless specifically requested
        ])
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        end_time = time.time()
        
        return {
            "category": "complete",
            "success": result.returncode == 0,
            "duration": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    
    def run_quick_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run quick smoke tests"""
        print("💨 Running quick smoke tests...")
        
        cmd = [sys.executable, "-m", "pytest"]
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend([
            "--tb=line",
            "-x",  # Stop on first failure
            "-m", "not slow and not load and not stress",
            "tests/unit/core/",  # Only core unit tests
            "--maxfail=5"  # Stop after 5 failures
        ])
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        end_time = time.time()
        
        return {
            "category": "quick",
            "success": result.returncode == 0,
            "duration": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    
    def check_test_environment(self) -> Dict[str, Any]:
        """Check if test environment is properly set up"""
        print("🔍 Checking test environment...")
        
        checks = {
            "python_version": sys.version,
            "pytest_available": False,
            "project_root_exists": self.project_root.exists(),
            "test_dir_exists": self.test_dir.exists(),
            "dependencies_installed": False,
            "environment_variables": {}
        }
        
        # Check pytest availability
        try:
            result = subprocess.run([sys.executable, "-m", "pytest", "--version"], 
                                  capture_output=True, text=True)
            checks["pytest_available"] = result.returncode == 0
            checks["pytest_version"] = result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            pass
        
        # Check key dependencies
        required_packages = ["pydantic", "fastapi", "sqlalchemy", "redis"]
        installed_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                installed_packages.append(package)
            except ImportError:
                pass
        
        checks["dependencies_installed"] = len(installed_packages) >= len(required_packages) * 0.8
        checks["installed_packages"] = installed_packages
        
        # Check environment variables
        env_vars = ["PRSM_ENVIRONMENT", "PRSM_DATABASE_URL", "PRSM_LOG_LEVEL"]
        for var in env_vars:
            checks["environment_variables"][var] = os.environ.get(var, "Not set")
        
        return checks
    
    def generate_test_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive test report"""
        report_lines = [
            "PRSM Test Suite Report",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Summary
        total_duration = sum(r.get("duration", 0) for r in results)
        successful_tests = sum(1 for r in results if r.get("success", False))
        total_tests = len(results)
        
        report_lines.extend([
            "SUMMARY",
            "-" * 20,
            f"Total test categories run: {total_tests}",
            f"Successful: {successful_tests}",
            f"Failed: {total_tests - successful_tests}",
            f"Total duration: {total_duration:.2f} seconds",
            f"Success rate: {(successful_tests/total_tests*100):.1f}%" if total_tests > 0 else "N/A",
            ""
        ])
        
        # Detailed results
        report_lines.extend([
            "DETAILED RESULTS",
            "-" * 30
        ])
        
        for result in results:
            category = result.get("category", "unknown")
            success = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
            duration = result.get("duration", 0)
            
            report_lines.extend([
                f"{category.upper()}: {success} ({duration:.2f}s)",
                ""
            ])
            
            if not result.get("success", False) and result.get("stderr"):
                report_lines.extend([
                    "Error details:",
                    result["stderr"][:500] + "..." if len(result["stderr"]) > 500 else result["stderr"],
                    ""
                ])
        
        # Performance metrics (if available)
        perf_results = [r for r in results if r.get("category") == "performance"]
        if perf_results:
            report_lines.extend([
                "PERFORMANCE METRICS",
                "-" * 30,
                "See benchmark_results.json for detailed performance data",
                ""
            ])
        
        # Coverage information
        coverage_results = [r for r in results if "coverage" in r.get("stdout", "")]
        if coverage_results:
            report_lines.extend([
                "COVERAGE INFORMATION",
                "-" * 30,
                "Coverage reports generated in htmlcov/ directory",
                "XML reports: coverage_*.xml",
                ""
            ])
        
        return "\n".join(report_lines)
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str = "test_results.json"):
        """Save test results to JSON file"""
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_categories": len(results),
                "successful": sum(1 for r in results if r.get("success", False)),
                "total_duration": sum(r.get("duration", 0) for r in results)
            },
            "results": results
        }
        
        output_path = self.project_root / output_file
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"📊 Test results saved to {output_path}")


def main():
    """Main entry point for test runner"""
    parser = argparse.ArgumentParser(description="PRSM Test Runner")
    
    parser.add_argument(
        "category",
        nargs="?",
        choices=["unit", "integration", "performance", "security", "api", "load", "all", "quick"],
        default="quick",
        help="Test category to run (default: quick)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting"
    )
    
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check test environment setup"
    )
    
    parser.add_argument(
        "--report",
        type=str,
        default="test_report.txt",
        help="Output file for test report"
    )
    
    parser.add_argument(
        "--results",
        type=str,
        default="test_results.json",
        help="Output file for test results JSON"
    )
    
    args = parser.parse_args()
    
    runner = PRSMTestRunner()
    results = []
    
    # Check environment if requested
    if args.check_env:
        print("🔍 Environment Check Results:")
        env_check = runner.check_test_environment()
        for key, value in env_check.items():
            print(f"  {key}: {value}")
        print()
    
    # Run tests based on category
    coverage = not args.no_coverage
    
    if args.category == "unit":
        results.append(runner.run_unit_tests(args.verbose, coverage))
    elif args.category == "integration":
        results.append(runner.run_integration_tests(args.verbose))
    elif args.category == "performance":
        results.append(runner.run_performance_tests(args.verbose))
    elif args.category == "security":
        results.append(runner.run_security_tests(args.verbose))
    elif args.category == "api":
        results.append(runner.run_api_tests(args.verbose))
    elif args.category == "load":
        results.append(runner.run_load_tests(args.verbose))
    elif args.category == "all":
        results.append(runner.run_all_tests(args.verbose, coverage))
    elif args.category == "quick":
        results.append(runner.run_quick_tests(args.verbose))
    
    # Generate and save report
    if results:
        report = runner.generate_test_report(results)
        
        # Save report to file
        report_path = runner.project_root / args.report
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save results to JSON
        runner.save_results(results, args.results)
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST EXECUTION SUMMARY")
        print("=" * 60)
        
        for result in results:
            category = result.get("category", "unknown").upper()
            status = "PASSED ✅" if result.get("success", False) else "FAILED ❌"
            duration = result.get("duration", 0)
            print(f"{category:15} {status:12} ({duration:.2f}s)")
        
        print(f"\n📋 Full report saved to: {report_path}")
        print(f"📊 Results data saved to: {runner.project_root / args.results}")
        
        # Exit with error code if any tests failed
        if not all(r.get("success", False) for r in results):
            print("\n❌ Some tests failed!")
            sys.exit(1)
        else:
            print("\n✅ All tests passed!")
            sys.exit(0)
    else:
        print("❓ No tests were run.")
        sys.exit(1)


if __name__ == "__main__":
    main()