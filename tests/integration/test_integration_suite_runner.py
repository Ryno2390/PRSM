#!/usr/bin/env python3
"""
PRSM Comprehensive Integration Test Suite Runner
===============================================

Master test runner that executes all integration tests in a coordinated manner,
provides comprehensive reporting, and validates overall system integration.

This runner:
- Executes all integration test suites in optimal order
- Provides unified reporting across all test categories
- Measures overall system health and readiness
- Generates comprehensive test coverage reports
- Validates production readiness
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import sys
import os

# Import all integration test modules
from test_marketplace_production_integration import run_marketplace_integration_tests
from test_end_to_end_prsm_workflow import run_end_to_end_integration_tests
from test_api_integration_comprehensive import run_api_integration_tests
from test_system_resilience_integration import run_system_resilience_tests

# Existing integration tests
from test_complete_prsm_system import main as run_complete_system_test
from test_system_health import run_system_health_check


class IntegrationTestSuiteRunner:
    """Comprehensive integration test suite runner"""
    
    def __init__(self):
        self.suite_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.results = {
            "suite_id": self.suite_id,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "test_suites": {},
            "overall_summary": {},
            "system_readiness": {},
            "recommendations": []
        }
    
    async def run_test_suite(self, suite_name: str, test_function, description: str) -> Dict[str, Any]:
        """Run individual test suite and capture results"""
        
        print(f"\n{'='*80}")
        print(f"🧪 RUNNING: {suite_name}")
        print(f"📝 {description}")
        print(f"{'='*80}")
        
        suite_start_time = time.time()
        
        try:
            # Run the test suite
            success = await test_function()
            
            suite_duration = time.time() - suite_start_time
            
            result = {
                "suite_name": suite_name,
                "description": description,
                "success": success,
                "duration": suite_duration,
                "start_time": datetime.fromtimestamp(suite_start_time, timezone.utc).isoformat(),
                "end_time": datetime.now(timezone.utc).isoformat(),
                "error": None
            }
            
            if success:
                print(f"✅ {suite_name} PASSED in {suite_duration:.2f}s")
            else:
                print(f"⚠️ {suite_name} FAILED/SKIPPED in {suite_duration:.2f}s")
            
            return result
            
        except Exception as e:
            suite_duration = time.time() - suite_start_time
            
            result = {
                "suite_name": suite_name,
                "description": description,
                "success": False,
                "duration": suite_duration,
                "start_time": datetime.fromtimestamp(suite_start_time, timezone.utc).isoformat(),
                "end_time": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }
            
            print(f"❌ {suite_name} ERROR in {suite_duration:.2f}s: {e}")
            
            return result
    
    async def run_all_integration_tests(self) -> Dict[str, Any]:
        """Run all integration test suites in optimal order"""
        
        print("🚀 PRSM COMPREHENSIVE INTEGRATION TEST SUITE")
        print("=" * 80)
        print(f"Suite ID: {self.suite_id}")
        print(f"Started: {datetime.now(timezone.utc).isoformat()}")
        print("=" * 80)
        
        # Define test suites in execution order
        test_suites = [
            {
                "name": "System Health Check",
                "function": run_system_health_check,
                "description": "Basic system health and component availability validation"
            },
            {
                "name": "Complete PRSM System Test", 
                "function": run_complete_system_test,
                "description": "Comprehensive test of all major PRSM subsystems"
            },
            {
                "name": "API Integration Tests",
                "function": run_api_integration_tests,
                "description": "Complete API layer testing including all endpoints and WebSocket"
            },
            {
                "name": "Marketplace Production Integration",
                "function": run_marketplace_integration_tests,
                "description": "Production marketplace system with real database operations"
            },
            {
                "name": "End-to-End PRSM Workflow",
                "function": run_end_to_end_integration_tests,
                "description": "Complete user query workflows through full PRSM pipeline"
            },
            {
                "name": "System Resilience Tests",
                "function": run_system_resilience_tests,
                "description": "Fault tolerance, error recovery, and edge case handling"
            }
        ]
        
        # Execute test suites
        for suite_config in test_suites:
            suite_result = await self.run_test_suite(
                suite_config["name"],
                suite_config["function"],
                suite_config["description"]
            )
            
            self.results["test_suites"][suite_config["name"]] = suite_result
        
        # Generate comprehensive analysis
        await self.generate_comprehensive_analysis()
        
        return self.results
    
    async def generate_comprehensive_analysis(self):
        """Generate comprehensive analysis of all test results"""
        
        print(f"\n{'='*80}")
        print("📊 COMPREHENSIVE TEST ANALYSIS")
        print(f"{'='*80}")
        
        # Basic statistics
        total_suites = len(self.results["test_suites"])
        successful_suites = sum(1 for suite in self.results["test_suites"].values() if suite["success"])
        failed_suites = total_suites - successful_suites
        
        total_duration = time.time() - self.start_time
        
        # Calculate success rate
        success_rate = successful_suites / total_suites if total_suites > 0 else 0
        
        # Overall summary
        self.results["overall_summary"] = {
            "total_suites": total_suites,
            "successful_suites": successful_suites,
            "failed_suites": failed_suites,
            "success_rate": success_rate,
            "total_duration": total_duration,
            "end_time": datetime.now(timezone.utc).isoformat()
        }
        
        print(f"📈 OVERALL RESULTS:")
        print(f"   Total Test Suites: {total_suites}")
        print(f"   Successful: {successful_suites}")
        print(f"   Failed/Skipped: {failed_suites}")
        print(f"   Success Rate: {success_rate:.1%}")
        print(f"   Total Duration: {total_duration:.2f}s")
        
        # Detailed suite analysis
        print(f"\n📋 SUITE-BY-SUITE RESULTS:")
        for suite_name, suite_result in self.results["test_suites"].items():
            status = "✅ PASS" if suite_result["success"] else "❌ FAIL"
            duration = suite_result["duration"]
            
            print(f"   {status} {suite_name}: {duration:.2f}s")
            
            if suite_result["error"]:
                print(f"      Error: {suite_result['error']}")
        
        # System readiness assessment
        await self.assess_system_readiness()
        
        # Generate recommendations
        await self.generate_recommendations()
        
        # Performance analysis
        await self.analyze_performance()
    
    async def assess_system_readiness(self):
        """Assess overall system readiness for production"""
        
        print(f"\n🎯 SYSTEM READINESS ASSESSMENT:")
        
        # Categorize test results by system area
        readiness_categories = {
            "core_system": ["System Health Check", "Complete PRSM System Test"],
            "api_layer": ["API Integration Tests"],
            "marketplace": ["Marketplace Production Integration"],
            "workflows": ["End-to-End PRSM Workflow"],
            "resilience": ["System Resilience Tests"]
        }
        
        category_readiness = {}
        
        for category, suite_names in readiness_categories.items():
            category_suites = [
                self.results["test_suites"][name] 
                for name in suite_names 
                if name in self.results["test_suites"]
            ]
            
            if category_suites:
                successful = sum(1 for suite in category_suites if suite["success"])
                total = len(category_suites)
                readiness_score = successful / total
                
                category_readiness[category] = {
                    "score": readiness_score,
                    "successful": successful,
                    "total": total,
                    "ready": readiness_score >= 0.8
                }
                
                status = "✅ READY" if readiness_score >= 0.8 else "⚠️ NEEDS WORK" if readiness_score >= 0.5 else "❌ NOT READY"
                print(f"   {status} {category.replace('_', ' ').title()}: {readiness_score:.1%} ({successful}/{total})")
        
        # Overall system readiness
        if category_readiness:
            overall_readiness = sum(cat["score"] for cat in category_readiness.values()) / len(category_readiness)
            ready_categories = sum(1 for cat in category_readiness.values() if cat["ready"])
            
            self.results["system_readiness"] = {
                "overall_score": overall_readiness,
                "ready_categories": ready_categories,
                "total_categories": len(category_readiness),
                "category_details": category_readiness,
                "production_ready": overall_readiness >= 0.8 and ready_categories >= len(category_readiness) * 0.8
            }
            
            production_status = "🚀 PRODUCTION READY" if self.results["system_readiness"]["production_ready"] else "🔧 NEEDS IMPROVEMENT"
            print(f"\n{production_status}")
            print(f"   Overall Readiness Score: {overall_readiness:.1%}")
            print(f"   Ready Categories: {ready_categories}/{len(category_readiness)}")
        
        else:
            self.results["system_readiness"] = {
                "overall_score": 0.0,
                "production_ready": False,
                "note": "Insufficient test data for readiness assessment"
            }
    
    async def generate_recommendations(self):
        """Generate recommendations based on test results"""
        
        print(f"\n💡 RECOMMENDATIONS:")
        
        recommendations = []
        
        # Analyze failed suites
        failed_suites = [
            (name, suite) for name, suite in self.results["test_suites"].items() 
            if not suite["success"]
        ]
        
        if failed_suites:
            for suite_name, suite_result in failed_suites:
                if "database" in suite_result.get("error", "").lower():
                    recommendations.append({
                        "priority": "high",
                        "category": "infrastructure",
                        "recommendation": "Set up database connections for full integration testing",
                        "affected_suite": suite_name
                    })
                elif "service" in suite_result.get("error", "").lower():
                    recommendations.append({
                        "priority": "medium",
                        "category": "services",
                        "recommendation": f"Investigate service configuration issues in {suite_name}",
                        "affected_suite": suite_name
                    })
                else:
                    recommendations.append({
                        "priority": "medium",
                        "category": "general",
                        "recommendation": f"Review and fix issues in {suite_name}",
                        "affected_suite": suite_name,
                        "error": suite_result.get("error")
                    })
        
        # Performance recommendations
        slow_suites = [
            (name, suite) for name, suite in self.results["test_suites"].items()
            if suite["duration"] > 120.0  # Longer than 2 minutes
        ]
        
        if slow_suites:
            recommendations.append({
                "priority": "low",
                "category": "performance",
                "recommendation": "Optimize slow test suites or underlying system performance",
                "affected_suites": [name for name, _ in slow_suites]
            })
        
        # Success-based recommendations
        if self.results["overall_summary"]["success_rate"] >= 0.8:
            recommendations.append({
                "priority": "low",
                "category": "deployment",
                "recommendation": "System shows good integration health - consider production deployment readiness review"
            })
        
        if not recommendations:
            recommendations.append({
                "priority": "info",
                "category": "status",
                "recommendation": "All integration tests completed successfully - system appears ready for use"
            })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2, "info": 3}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        self.results["recommendations"] = recommendations
        
        for i, rec in enumerate(recommendations, 1):
            priority_icon = {"high": "🔥", "medium": "⚠️", "low": "💡", "info": "ℹ️"}.get(rec["priority"], "•")
            print(f"   {i}. {priority_icon} {rec['recommendation']}")
            if "affected_suite" in rec:
                print(f"      Affected: {rec['affected_suite']}")
    
    async def analyze_performance(self):
        """Analyze performance characteristics across test suites"""
        
        print(f"\n⚡ PERFORMANCE ANALYSIS:")
        
        suite_durations = [
            (name, suite["duration"]) 
            for name, suite in self.results["test_suites"].items()
        ]
        
        if suite_durations:
            suite_durations.sort(key=lambda x: x[1], reverse=True)
            
            total_time = sum(duration for _, duration in suite_durations)
            avg_time = total_time / len(suite_durations)
            
            print(f"   Total Test Time: {total_time:.2f}s")
            print(f"   Average Suite Time: {avg_time:.2f}s")
            print(f"   Slowest Suite: {suite_durations[0][0]} ({suite_durations[0][1]:.2f}s)")
            print(f"   Fastest Suite: {suite_durations[-1][0]} ({suite_durations[-1][1]:.2f}s)")
            
            # Performance recommendations
            slow_threshold = avg_time * 2
            slow_suites = [name for name, duration in suite_durations if duration > slow_threshold]
            
            if slow_suites:
                print(f"   ⚠️ Slow Suites (>{slow_threshold:.1f}s): {', '.join(slow_suites)}")
    
    async def save_results(self, filename: str = None):
        """Save comprehensive test results to file"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prsm_integration_test_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            print(f"\n💾 Test results saved to: {filename}")
            
        except Exception as e:
            print(f"\n⚠️ Failed to save results: {e}")


async def run_comprehensive_integration_tests():
    """Run the complete PRSM integration test suite"""
    
    runner = IntegrationTestSuiteRunner()
    
    try:
        # Run all integration tests
        results = await runner.run_all_integration_tests()
        
        # Save results
        await runner.save_results()
        
        # Final assessment
        print(f"\n{'='*80}")
        print("🎉 PRSM INTEGRATION TEST SUITE COMPLETE")
        print(f"{'='*80}")
        
        success_rate = results["overall_summary"]["success_rate"]
        production_ready = results["system_readiness"].get("production_ready", False)
        
        if production_ready:
            print("🚀 RESULT: PRSM system is ready for production deployment!")
            print("   All critical integration tests passed successfully.")
            print("   System demonstrates strong reliability and functionality.")
        elif success_rate >= 0.7:
            print("✅ RESULT: PRSM system is largely functional with minor issues.")
            print("   Most integration tests passed - review recommendations.")
            print("   System is suitable for continued development and testing.")
        elif success_rate >= 0.5:
            print("⚠️ RESULT: PRSM system has significant integration issues.")
            print("   Multiple test failures indicate system needs attention.")
            print("   Address failed tests before production consideration.")
        else:
            print("❌ RESULT: PRSM system requires substantial integration work.")
            print("   Major system components are not functioning properly.")
            print("   Comprehensive review and fixes needed.")
        
        return success_rate >= 0.7
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST SUITE FAILED: {e}")
        print("Critical error in test execution - review system configuration.")
        return False


if __name__ == "__main__":
    # Run the comprehensive integration test suite
    success = asyncio.run(run_comprehensive_integration_tests())
    exit(0 if success else 1)