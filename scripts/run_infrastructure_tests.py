#!/usr/bin/env python3
"""
PRSM Infrastructure Test Runner
==============================

Orchestrates comprehensive infrastructure testing for production readiness validation.
Executes full-stack integration tests and generates detailed compliance reports.

Features:
- Pre-test environment validation
- Parallel test execution for performance
- Real-time progress monitoring
- Comprehensive reporting and remediation guidance
- CI/CD integration support
- Production-safe testing modes

Usage:
    python scripts/run_infrastructure_tests.py --mode development
    python scripts/run_infrastructure_tests.py --mode production --generate-report
    python scripts/run_infrastructure_tests.py --category security --verbose
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InfrastructureTestRunner:
    """Orchestrates infrastructure testing and reporting"""
    
    def __init__(self, mode: str = "development", verbose: bool = False):
        self.mode = mode
        self.verbose = verbose
        self.project_root = Path(__file__).parent.parent
        self.reports_dir = self.project_root / "infrastructure-test-reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Test categories and their criticality
        self.test_categories = {
            "core_infrastructure": {"critical": True, "weight": 25},
            "application_layer": {"critical": True, "weight": 20},
            "data_layer": {"critical": True, "weight": 15},
            "security_layer": {"critical": True, "weight": 20},
            "performance_monitoring": {"critical": False, "weight": 10},
            "business_logic": {"critical": True, "weight": 10},
            "external_integrations": {"critical": False, "weight": 5},
            "disaster_recovery": {"critical": True, "weight": 5},
            "multi_cloud_readiness": {"critical": False, "weight": 3},
            "end_to_end_workflows": {"critical": True, "weight": 7}
        }
        
        logger.info(f"🧪 Infrastructure Test Runner initialized (mode: {mode})")
    
    async def run_tests(self, categories: Optional[List[str]] = None) -> Dict:
        """Run infrastructure tests"""
        logger.info("🚀 Starting Infrastructure Integration Tests")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Step 1: Validate test environment
            env_validation = await self._validate_test_environment()
            if not env_validation["valid"]:
                logger.error("❌ Test environment validation failed")
                return {"status": "FAILED", "reason": "environment_validation", "details": env_validation}
            
            logger.info("✅ Test environment validation passed")
            
            # Step 2: Import and run the full test suite
            from tests.infrastructure.test_full_stack_integration import run_infrastructure_tests
            
            # Execute tests
            logger.info("🔧 Executing full-stack integration tests...")
            test_results = await run_infrastructure_tests(production_mode=(self.mode == "production"))
            
            # Step 3: Generate comprehensive report
            report_data = await self._generate_comprehensive_report(test_results)
            
            execution_time = time.time() - start_time
            
            # Step 4: Determine overall status
            overall_status = self._determine_overall_status(test_results)
            
            logger.info(f"✅ Infrastructure tests completed in {execution_time:.2f} seconds")
            logger.info(f"📊 Overall Status: {overall_status}")
            logger.info(f"🏥 Health Score: {test_results.overall_health_score:.1f}/100")
            
            return {
                "status": overall_status,
                "health_score": test_results.overall_health_score,
                "execution_time": execution_time,
                "test_results": test_results,
                "report_file": report_data["report_file"],
                "summary": {
                    "total_tests": test_results.total_tests,
                    "passed_tests": test_results.passed_tests,
                    "failed_tests": test_results.failed_tests,
                    "critical_issues": len(test_results.critical_issues),
                    "warnings": len(test_results.warnings)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Infrastructure testing failed: {e}")
            return {
                "status": "ERROR",
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    async def _validate_test_environment(self) -> Dict:
        """Validate the test environment before running tests"""
        validation_result = {
            "valid": True,
            "checks": [],
            "warnings": []
        }
        
        # Check 1: Python environment
        python_version = sys.version_info
        if python_version >= (3, 8):
            validation_result["checks"].append({
                "check": "python_version",
                "status": "PASS",
                "details": f"Python {python_version.major}.{python_version.minor}.{python_version.micro}"
            })
        else:
            validation_result["checks"].append({
                "check": "python_version",
                "status": "FAIL",
                "details": f"Python {python_version.major}.{python_version.minor} < 3.8"
            })
            validation_result["valid"] = False
        
        # Check 2: Required packages
        required_packages = ["structlog", "pytest"]
        for package in required_packages:
            try:
                __import__(package)
                validation_result["checks"].append({
                    "check": f"package_{package}",
                    "status": "PASS",
                    "details": f"{package} available"
                })
            except ImportError:
                validation_result["checks"].append({
                    "check": f"package_{package}",
                    "status": "FAIL",
                    "details": f"{package} not installed"
                })
                validation_result["valid"] = False
        
        # Check 3: Project structure
        required_paths = [
            "prsm",
            "tests",
            "deploy",
            "scripts"
        ]
        
        for path in required_paths:
            path_obj = self.project_root / path
            if path_obj.exists():
                validation_result["checks"].append({
                    "check": f"path_{path}",
                    "status": "PASS",
                    "details": f"{path} directory exists"
                })
            else:
                validation_result["checks"].append({
                    "check": f"path_{path}",
                    "status": "FAIL",
                    "details": f"{path} directory missing"
                })
                validation_result["valid"] = False
        
        # Check 4: Test mode validation
        if self.mode == "production":
            validation_result["warnings"].append("Running in PRODUCTION mode - using production infrastructure")
        else:
            validation_result["warnings"].append("Running in DEVELOPMENT mode - using mocked infrastructure")
        
        return validation_result
    
    async def _generate_comprehensive_report(self, test_results) -> Dict:
        """Generate comprehensive infrastructure test report"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        # Import the report generator
        from tests.infrastructure.test_full_stack_integration import generate_test_report
        
        # Generate the main report
        report_content = generate_test_report(test_results)
        
        # Add executive summary and recommendations
        enhanced_report = self._enhance_report_with_analysis(report_content, test_results)
        
        # Save report
        report_filename = f"infrastructure_integration_report_{timestamp}.md"
        report_path = self.reports_dir / report_filename
        
        with open(report_path, 'w') as f:
            f.write(enhanced_report)
        
        # Generate JSON summary for CI/CD integration
        json_summary = {
            "test_run": {
                "timestamp": test_results.start_time.isoformat(),
                "mode": self.mode,
                "health_score": test_results.overall_health_score,
                "total_tests": test_results.total_tests,
                "passed_tests": test_results.passed_tests,
                "failed_tests": test_results.failed_tests,
                "critical_issues_count": len(test_results.critical_issues),
                "warnings_count": len(test_results.warnings)
            },
            "categories": {
                category: {
                    "success_rate": (data["passed"] / max(1, data["total"])) * 100,
                    "total": data["total"],
                    "passed": data["passed"],
                    "failed": data["failed"],
                    "critical": self.test_categories.get(category, {}).get("critical", False)
                } for category, data in test_results.categories.items()
            },
            "critical_issues": test_results.critical_issues,
            "warnings": test_results.warnings,
            "recommendations": self._generate_recommendations(test_results)
        }
        
        json_path = self.reports_dir / f"infrastructure_test_summary_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(json_summary, f, indent=2)
        
        logger.info(f"📄 Test report saved: {report_path}")
        logger.info(f"📋 JSON summary saved: {json_path}")
        
        return {
            "report_file": str(report_path),
            "json_summary": str(json_path),
            "health_score": test_results.overall_health_score
        }
    
    def _enhance_report_with_analysis(self, base_report: str, test_results) -> str:
        """Enhance the base report with additional analysis"""
        
        # Calculate category scores
        category_analysis = ""
        for category, data in test_results.categories.items():
            success_rate = (data["passed"] / max(1, data["total"])) * 100
            criticality = self.test_categories.get(category, {}).get("critical", False)
            weight = self.test_categories.get(category, {}).get("weight", 0)
            
            status = "🔴 CRITICAL" if criticality and success_rate < 80 else "🟡 WARNING" if success_rate < 90 else "🟢 HEALTHY"
            
            category_analysis += f"""
### {category.replace('_', ' ').title()}
- **Success Rate:** {success_rate:.1f}%
- **Weight:** {weight}%
- **Critical:** {'Yes' if criticality else 'No'}
- **Status:** {status}
"""
        
        # Production readiness assessment
        production_readiness = self._assess_production_readiness(test_results)
        
        enhanced_report = f"""# 🏗️ PRSM Infrastructure Integration Test Report
## Comprehensive Production Readiness Assessment

{base_report}

## 📊 Detailed Category Analysis
{category_analysis}

## 🚀 Production Readiness Assessment

{production_readiness}

## 🔧 Remediation Roadmap

{self._generate_remediation_roadmap(test_results)}

---
*Enhanced report generated by PRSM Infrastructure Test Runner*
*Mode: {self.mode.upper()} | Health Score: {test_results.overall_health_score:.1f}/100*
"""
        
        return enhanced_report
    
    def _assess_production_readiness(self, test_results) -> str:
        """Assess production readiness based on test results"""
        health_score = test_results.overall_health_score
        critical_issues = len(test_results.critical_issues)
        
        if health_score >= 95 and critical_issues == 0:
            return """
### ✅ PRODUCTION READY
- All critical systems operational
- Security posture excellent
- Performance within acceptable thresholds
- **Recommendation:** Proceed with Series A production deployment

**Action Items:**
- Monitor warnings and address proactively
- Schedule regular health checks
- Implement continuous monitoring
"""
        elif health_score >= 85 and critical_issues <= 2:
            return """
### 🟡 NEARLY PRODUCTION READY
- Core systems operational with minor issues
- Address critical issues before full deployment
- **Recommendation:** Fix critical issues, then proceed with staged deployment

**Action Items:**
- Resolve all critical issues immediately
- Implement additional monitoring for warnings
- Plan staged rollout strategy
"""
        elif health_score >= 70:
            return """
### 🟠 REQUIRES IMPROVEMENT
- Infrastructure needs significant work before production
- Multiple critical issues detected
- **Recommendation:** Address all critical issues before considering production deployment

**Action Items:**
- Create detailed remediation plan
- Allocate engineering resources for fixes
- Re-run tests after remediation
"""
        else:
            return """
### 🔴 NOT PRODUCTION READY
- Critical infrastructure failures detected
- Significant security or stability concerns
- **Recommendation:** Do not deploy to production until all issues resolved

**Action Items:**
- Emergency remediation required
- Full infrastructure review needed
- Consider rolling back recent changes
"""
    
    def _generate_remediation_roadmap(self, test_results) -> str:
        """Generate a prioritized remediation roadmap"""
        roadmap = "### Priority 1: Critical Issues (Immediate Action Required)\n\n"
        
        # Group critical issues by category
        critical_by_category = {}
        for issue in test_results.critical_issues:
            category = issue.get("category", "unknown")
            if category not in critical_by_category:
                critical_by_category[category] = []
            critical_by_category[category].append(issue)
        
        for category, issues in critical_by_category.items():
            roadmap += f"**{category.replace('_', ' ').title()}:**\n"
            for issue in issues:
                roadmap += f"- {issue['issue']} (Impact: {issue.get('impact', 'HIGH')})\n"
            roadmap += "\n"
        
        roadmap += "### Priority 2: Warnings (Address Within 1 Week)\n\n"
        
        # Group warnings by category
        warnings_by_category = {}
        for warning in test_results.warnings:
            category = warning.get("category", "unknown")
            if category not in warnings_by_category:
                warnings_by_category[category] = []
            warnings_by_category[category].append(warning)
        
        for category, warnings in warnings_by_category.items():
            roadmap += f"**{category.replace('_', ' ').title()}:**\n"
            for warning in warnings:
                roadmap += f"- {warning['warning']}\n"
            roadmap += "\n"
        
        roadmap += "### Priority 3: Optimization (Ongoing Improvement)\n\n"
        roadmap += "- Implement performance monitoring dashboards\n"
        roadmap += "- Establish automated testing in CI/CD pipeline\n"
        roadmap += "- Schedule regular infrastructure health assessments\n"
        roadmap += "- Document operational procedures\n"
        roadmap += "- Plan capacity scaling strategies\n"
        
        return roadmap
    
    def _generate_recommendations(self, test_results) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        health_score = test_results.overall_health_score
        critical_issues = len(test_results.critical_issues)
        
        if critical_issues > 0:
            recommendations.append("Immediately address all critical infrastructure issues")
        
        if health_score < 90:
            recommendations.append("Implement comprehensive monitoring and alerting")
        
        # Category-specific recommendations
        for category, data in test_results.categories.items():
            success_rate = (data["passed"] / max(1, data["total"])) * 100
            if success_rate < 80 and self.test_categories.get(category, {}).get("critical", False):
                recommendations.append(f"Critical: Fix {category.replace('_', ' ')} infrastructure")
            elif success_rate < 90:
                recommendations.append(f"Improve {category.replace('_', ' ')} reliability")
        
        if health_score >= 90:
            recommendations.append("Infrastructure ready for production deployment")
            recommendations.append("Implement continuous monitoring for ongoing health")
        
        return recommendations
    
    def _determine_overall_status(self, test_results) -> str:
        """Determine overall test status"""
        health_score = test_results.overall_health_score
        critical_issues = len(test_results.critical_issues)
        
        if health_score >= 95 and critical_issues == 0:
            return "EXCELLENT"
        elif health_score >= 85 and critical_issues <= 2:
            return "GOOD"
        elif health_score >= 70:
            return "NEEDS_IMPROVEMENT"
        else:
            return "CRITICAL"

async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="PRSM Infrastructure Test Runner")
    parser.add_argument("--mode", choices=["development", "production"], default="development",
                       help="Test mode (default: development)")
    parser.add_argument("--category", nargs="+", help="Specific test categories to run")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--generate-report", action="store_true", help="Generate detailed report")
    parser.add_argument("--output-dir", help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = InfrastructureTestRunner(mode=args.mode, verbose=args.verbose)
    
    if args.output_dir:
        runner.reports_dir = Path(args.output_dir)
        runner.reports_dir.mkdir(exist_ok=True)
    
    logger.info("🧪 PRSM Infrastructure Integration Test Runner")
    logger.info("=" * 60)
    
    # Run tests
    results = await runner.run_tests(categories=args.category)
    
    # Display summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST EXECUTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Status: {results['status']}")
    
    if "health_score" in results:
        logger.info(f"Health Score: {results['health_score']:.1f}/100")
        logger.info(f"Execution Time: {results['execution_time']:.2f} seconds")
        
        summary = results.get("summary", {})
        logger.info(f"Tests: {summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)} passed")
        logger.info(f"Critical Issues: {summary.get('critical_issues', 0)}")
        logger.info(f"Warnings: {summary.get('warnings', 0)}")
        
        if "report_file" in results:
            logger.info(f"Report: {results['report_file']}")
    
    # Exit with appropriate code
    if results["status"] in ["EXCELLENT", "GOOD"]:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())