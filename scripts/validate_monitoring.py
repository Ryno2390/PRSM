#!/usr/bin/env python3
"""
PRSM Monitoring Validation Script
================================

Comprehensive validation script for PRSM monitoring systems.
Tests dashboard functionality, metrics accuracy, and system performance under load.
"""

import asyncio
import sys
import time
import json
import concurrent.futures
from pathlib import Path

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prsm.monitoring import (
    MetricsCollector, DashboardManager, AlertManager,
    ValidationSuite, create_monitoring_validation_suite,
    PerformanceProfiler
)
from prsm.monitoring.validators import (
    MetricsValidationTest, DashboardValidationTest,
    AlertValidationTest, PerformanceValidationTest
)


class LoadTestRunner:
    """Runs load tests against the monitoring system"""
    
    def __init__(self, dashboard_url: str = "http://localhost:3000"):
        self.dashboard_url = dashboard_url
        self.metrics_collector = None
        self.dashboard_manager = None
        self.alert_manager = None
    
    async def setup_monitoring_stack(self):
        """Set up the complete monitoring stack"""
        print("🔧 Setting up monitoring stack...")
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.dashboard_manager = DashboardManager(
            metrics_collector=self.metrics_collector,
            port=3000
        )
        self.alert_manager = AlertManager()
        
        # Start metrics collection
        await self.metrics_collector.start_collection()
        print("✅ Metrics collection started")
        
        # Start dashboard (non-blocking)
        dashboard_task = asyncio.create_task(
            self.dashboard_manager.start_server()
        )
        
        # Wait a moment for dashboard to start
        await asyncio.sleep(2)
        print("✅ Dashboard server started")
        
        return dashboard_task
    
    async def run_load_test(self, duration: int = 60, concurrent_requests: int = 10):
        """Run load test against the dashboard"""
        print(f"🚀 Starting load test: {concurrent_requests} concurrent requests for {duration}s")
        
        import aiohttp
        
        async def make_request(session, endpoint):
            """Make a single request to the dashboard"""
            try:
                url = f"{self.dashboard_url}{endpoint}"
                async with session.get(url, timeout=5) as response:
                    await response.text()
                    return response.status, len(await response.text())
            except Exception as e:
                return None, str(e)
        
        # Test endpoints
        endpoints = [
            "/",
            "/api/metrics",
            "/api/health", 
            "/api/system-info"
        ]
        
        results = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'status_codes': {},
            'errors': []
        }
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < duration:
                # Create concurrent requests
                tasks = []
                for _ in range(concurrent_requests):
                    endpoint = endpoints[results['total_requests'] % len(endpoints)]
                    task = make_request(session, endpoint)
                    tasks.append(task)
                
                # Execute requests concurrently
                request_start = time.time()
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                request_time = time.time() - request_start
                
                # Process results
                for response in responses:
                    results['total_requests'] += 1
                    
                    if isinstance(response, tuple) and response[0] is not None:
                        status_code, content_length = response
                        results['successful_requests'] += 1
                        results['status_codes'][status_code] = results['status_codes'].get(status_code, 0) + 1
                        results['response_times'].append(request_time / concurrent_requests)
                    else:
                        results['failed_requests'] += 1
                        error = response[1] if isinstance(response, tuple) else str(response)
                        results['errors'].append(error)
                
                # Brief pause between batches
                await asyncio.sleep(0.1)
        
        return results
    
    async def validate_metrics_accuracy(self):
        """Validate that metrics are accurate and consistent"""
        print("🔍 Validating metrics accuracy...")
        
        if not self.metrics_collector:
            return {"status": "failed", "message": "Metrics collector not initialized"}
        
        # Collect metrics multiple times and check consistency
        metric_samples = []
        for i in range(5):
            metrics = await self.metrics_collector.get_metrics()
            metric_samples.append(metrics)
            await asyncio.sleep(1)
        
        # Analyze consistency
        inconsistencies = []
        
        # Check if metrics are reasonable
        for sample in metric_samples:
            if 'cpu_usage' in sample:
                if sample['cpu_usage'] < 0 or sample['cpu_usage'] > 100:
                    inconsistencies.append(f"Invalid CPU usage: {sample['cpu_usage']}")
            
            if 'memory_usage' in sample:
                if sample['memory_usage'] < 0:
                    inconsistencies.append(f"Invalid memory usage: {sample['memory_usage']}")
        
        return {
            "status": "passed" if not inconsistencies else "warning",
            "message": "Metrics validation completed",
            "inconsistencies": inconsistencies,
            "sample_count": len(metric_samples)
        }
    
    async def run_comprehensive_validation(self):
        """Run the comprehensive validation suite"""
        print("🧪 Running comprehensive validation suite...")
        
        # Create validation suite
        suite = create_monitoring_validation_suite(
            metrics_collector=self.metrics_collector,
            dashboard_manager=self.dashboard_manager,
            alert_manager=self.alert_manager
        )
        
        # Add custom load test validation
        class LoadTestValidation:
            def __init__(self, load_results):
                self.load_results = load_results
            
            async def run(self, context):
                if self.load_results['failed_requests'] > self.load_results['total_requests'] * 0.1:
                    return {
                        'test_name': 'load_test_validation',
                        'status': 'failed',
                        'message': f"High failure rate: {self.load_results['failed_requests']}/{self.load_results['total_requests']}",
                        'execution_time': 0,
                        'timestamp': time.time()
                    }
                else:
                    return {
                        'test_name': 'load_test_validation', 
                        'status': 'passed',
                        'message': f"Load test passed: {self.load_results['successful_requests']}/{self.load_results['total_requests']} successful",
                        'execution_time': 0,
                        'timestamp': time.time()
                    }
        
        # Run validation suite
        results = await suite.run_all()
        
        return results
    
    async def generate_report(self, load_results, validation_results, metrics_validation):
        """Generate comprehensive validation report"""
        report = {
            'timestamp': time.time(),
            'validation_summary': {
                'total_tests': validation_results.total_tests,
                'passed': validation_results.passed,
                'failed': validation_results.failed,
                'warnings': validation_results.warnings,
                'execution_time': validation_results.execution_time
            },
            'load_test_results': {
                'total_requests': load_results['total_requests'],
                'successful_requests': load_results['successful_requests'],
                'failed_requests': load_results['failed_requests'],
                'success_rate': load_results['successful_requests'] / max(load_results['total_requests'], 1) * 100,
                'average_response_time': sum(load_results['response_times']) / max(len(load_results['response_times']), 1) if load_results['response_times'] else 0,
                'status_codes': load_results['status_codes'],
                'error_count': len(load_results['errors'])
            },
            'metrics_validation': metrics_validation,
            'detailed_results': [
                {
                    'test_name': r.test_name,
                    'status': r.status.value,
                    'message': r.message,
                    'execution_time': r.execution_time
                }
                for r in validation_results.results
            ]
        }
        
        return report
    
    def print_report(self, report):
        """Print formatted validation report"""
        print("\n" + "="*60)
        print("🎯 PRSM MONITORING VALIDATION REPORT")
        print("="*60)
        
        # Summary
        summary = report['validation_summary']
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   ✅ Passed: {summary['passed']}")
        print(f"   ❌ Failed: {summary['failed']}")
        print(f"   ⚠️  Warnings: {summary['warnings']}")
        print(f"   ⏱️  Execution Time: {summary['execution_time']:.2f}s")
        
        # Load test results
        load = report['load_test_results']
        print(f"\n🚀 LOAD TEST RESULTS:")
        print(f"   Total Requests: {load['total_requests']}")
        print(f"   Success Rate: {load['success_rate']:.1f}%")
        print(f"   Average Response Time: {load['average_response_time']:.3f}s")
        print(f"   Errors: {load['error_count']}")
        
        # Metrics validation
        metrics = report['metrics_validation']
        print(f"\n🔍 METRICS VALIDATION:")
        print(f"   Status: {metrics['status'].upper()}")
        print(f"   Message: {metrics['message']}")
        if metrics.get('inconsistencies'):
            print(f"   Issues: {len(metrics['inconsistencies'])}")
        
        # Detailed results
        print(f"\n📋 DETAILED TEST RESULTS:")
        for result in report['detailed_results']:
            status_emoji = {"passed": "✅", "failed": "❌", "warning": "⚠️", "skipped": "⏭️"}
            emoji = status_emoji.get(result['status'], "❓")
            print(f"   {emoji} {result['test_name']}: {result['message']} ({result['execution_time']:.2f}s)")
        
        print("\n" + "="*60)
        
        # Overall assessment
        if summary['failed'] == 0:
            if summary['warnings'] == 0:
                print("🎉 ALL TESTS PASSED! Monitoring system is production-ready.")
            else:
                print("⚠️  TESTS PASSED WITH WARNINGS. Review warnings before production deployment.")
        else:
            print("❌ SOME TESTS FAILED. Address failures before production deployment.")
        
        print("="*60)


async def main():
    """Main validation runner"""
    print("🎯 PRSM Monitoring Validation System")
    print("====================================\n")
    
    runner = LoadTestRunner()
    
    try:
        # Set up monitoring stack
        dashboard_task = await runner.setup_monitoring_stack()
        
        # Run load test
        load_results = await runner.run_load_test(duration=30, concurrent_requests=5)
        
        # Validate metrics accuracy
        metrics_validation = await runner.validate_metrics_accuracy()
        
        # Run comprehensive validation
        validation_results = await runner.run_comprehensive_validation()
        
        # Generate and print report
        report = await runner.generate_report(load_results, validation_results, metrics_validation)
        runner.print_report(report)
        
        # Save report to file
        report_file = Path(__file__).parent.parent / "monitoring_validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Full report saved to: {report_file}")
        
        # Clean shutdown
        dashboard_task.cancel()
        
    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())