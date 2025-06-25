"""
PRSM Validation Suite
====================

Comprehensive validation framework for PRSM systems.
Provides automated testing, validation, and quality assurance
for monitoring, metrics, and system health.
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import psutil
import aiohttp
from pathlib import Path
import logging


class ValidationStatus(Enum):
    """Status of validation test"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationResult:
    """Result of a single validation test"""
    test_name: str
    status: ValidationStatus
    message: str
    execution_time: float
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ValidationSuiteResult:
    """Results from entire validation suite"""
    suite_name: str
    total_tests: int
    passed: int
    failed: int
    warnings: int
    skipped: int
    execution_time: float
    timestamp: float
    results: List[ValidationResult] = field(default_factory=list)


class ValidationTest:
    """Base class for validation tests"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    async def run(self, context: Dict[str, Any] = None) -> ValidationResult:
        """Run the validation test"""
        start_time = time.time()
        try:
            result = await self._execute(context or {})
            execution_time = time.time() - start_time
            
            return ValidationResult(
                test_name=self.name,
                status=result.get('status', ValidationStatus.PASSED),
                message=result.get('message', 'Test passed'),
                execution_time=execution_time,
                timestamp=start_time,
                details=result.get('details', {}),
                error=result.get('error')
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ValidationResult(
                test_name=self.name,
                status=ValidationStatus.FAILED,
                message=f"Test failed with exception: {str(e)}",
                execution_time=execution_time,
                timestamp=start_time,
                error=str(e)
            )
    
    async def _execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Override this method to implement test logic"""
        raise NotImplementedError("Subclasses must implement _execute method")


class MetricsValidationTest(ValidationTest):
    """Validate metrics collection and accuracy"""
    
    def __init__(self, metrics_collector):
        super().__init__("metrics_validation", "Validate metrics collection")
        self.metrics_collector = metrics_collector
    
    async def _execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Test metrics collection
        if not hasattr(self.metrics_collector, 'get_metrics'):
            return {
                'status': ValidationStatus.FAILED,
                'message': 'Metrics collector missing get_metrics method'
            }
        
        # Get metrics
        try:
            metrics = await self.metrics_collector.get_metrics()
            if not metrics:
                return {
                    'status': ValidationStatus.WARNING,
                    'message': 'No metrics collected yet'
                }
            
            # Validate metric values
            issues = []
            if 'cpu_usage' in metrics and (metrics['cpu_usage'] < 0 or metrics['cpu_usage'] > 100):
                issues.append("Invalid CPU usage value")
            
            if 'memory_usage' in metrics and metrics['memory_usage'] < 0:
                issues.append("Invalid memory usage value")
            
            if issues:
                return {
                    'status': ValidationStatus.WARNING,
                    'message': f"Metric validation issues: {', '.join(issues)}",
                    'details': {'issues': issues, 'metrics': metrics}
                }
            
            return {
                'status': ValidationStatus.PASSED,
                'message': 'Metrics validation successful',
                'details': {'metric_count': len(metrics)}
            }
        
        except Exception as e:
            return {
                'status': ValidationStatus.FAILED,
                'message': f'Failed to get metrics: {str(e)}',
                'error': str(e)
            }


class DashboardValidationTest(ValidationTest):
    """Validate dashboard functionality"""
    
    def __init__(self, dashboard_url: str = "http://localhost:3000"):
        super().__init__("dashboard_validation", "Validate dashboard accessibility")
        self.dashboard_url = dashboard_url
    
    async def _execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with aiohttp.ClientSession() as session:
                # Test dashboard endpoint
                async with session.get(self.dashboard_url, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Basic content validation
                        if 'PRSM' in content and 'dashboard' in content.lower():
                            return {
                                'status': ValidationStatus.PASSED,
                                'message': 'Dashboard is accessible and working',
                                'details': {'status_code': response.status, 'content_length': len(content)}
                            }
                        else:
                            return {
                                'status': ValidationStatus.WARNING,
                                'message': 'Dashboard accessible but content may be incomplete',
                                'details': {'status_code': response.status}
                            }
                    else:
                        return {
                            'status': ValidationStatus.FAILED,
                            'message': f'Dashboard returned status code {response.status}'
                        }
        
        except asyncio.TimeoutError:
            return {
                'status': ValidationStatus.FAILED,
                'message': 'Dashboard request timed out'
            }
        except aiohttp.ClientError as e:
            return {
                'status': ValidationStatus.FAILED,
                'message': f'Failed to connect to dashboard: {str(e)}'
            }


class AlertValidationTest(ValidationTest):
    """Validate alert system functionality"""
    
    def __init__(self, alert_manager):
        super().__init__("alert_validation", "Validate alert system")
        self.alert_manager = alert_manager
    
    async def _execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not hasattr(self.alert_manager, 'send_alert'):
            return {
                'status': ValidationStatus.FAILED,
                'message': 'Alert manager missing send_alert method'
            }
        
        # Test alert configuration
        try:
            # Check if alert channels are configured
            if hasattr(self.alert_manager, 'channels') and self.alert_manager.channels:
                channel_count = len(self.alert_manager.channels)
                return {
                    'status': ValidationStatus.PASSED,
                    'message': f'Alert system configured with {channel_count} channels',
                    'details': {'channel_count': channel_count}
                }
            else:
                return {
                    'status': ValidationStatus.WARNING,
                    'message': 'Alert system available but no channels configured'
                }
        
        except Exception as e:
            return {
                'status': ValidationStatus.FAILED,
                'message': f'Alert validation failed: {str(e)}',
                'error': str(e)
            }


class PerformanceValidationTest(ValidationTest):
    """Validate system performance under load"""
    
    def __init__(self, load_duration: int = 10):
        super().__init__("performance_validation", "Validate system performance")
        self.load_duration = load_duration
    
    async def _execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Monitor system performance for specified duration
        start_time = time.time()
        cpu_samples = []
        memory_samples = []
        
        while time.time() - start_time < self.load_duration:
            cpu_samples.append(psutil.cpu_percent())
            memory_samples.append(psutil.virtual_memory().percent)
            await asyncio.sleep(1)
        
        # Calculate averages
        avg_cpu = sum(cpu_samples) / len(cpu_samples)
        avg_memory = sum(memory_samples) / len(memory_samples)
        max_cpu = max(cpu_samples)
        max_memory = max(memory_samples)
        
        # Evaluate performance
        issues = []
        if avg_cpu > 80:
            issues.append(f"High average CPU usage: {avg_cpu:.1f}%")
        if max_cpu > 95:
            issues.append(f"CPU spike detected: {max_cpu:.1f}%")
        if avg_memory > 85:
            issues.append(f"High average memory usage: {avg_memory:.1f}%")
        if max_memory > 95:
            issues.append(f"Memory spike detected: {max_memory:.1f}%")
        
        if issues:
            return {
                'status': ValidationStatus.WARNING,
                'message': f"Performance issues detected: {', '.join(issues)}",
                'details': {
                    'avg_cpu': avg_cpu,
                    'max_cpu': max_cpu,
                    'avg_memory': avg_memory,
                    'max_memory': max_memory,
                    'duration': self.load_duration
                }
            }
        else:
            return {
                'status': ValidationStatus.PASSED,
                'message': 'System performance within acceptable limits',
                'details': {
                    'avg_cpu': avg_cpu,
                    'max_cpu': max_cpu,
                    'avg_memory': avg_memory,
                    'max_memory': max_memory,
                    'duration': self.load_duration
                }
            }


class ValidationSuite:
    """
    Comprehensive validation suite for PRSM monitoring systems
    """
    
    def __init__(self, name: str = "PRSM Validation Suite"):
        self.name = name
        self.tests: List[ValidationTest] = []
        self.context: Dict[str, Any] = {}
    
    def add_test(self, test: ValidationTest):
        """Add a validation test to the suite"""
        self.tests.append(test)
    
    def add_context(self, key: str, value: Any):
        """Add context data for tests"""
        self.context[key] = value
    
    async def run_all(self, parallel: bool = True) -> ValidationSuiteResult:
        """Run all validation tests"""
        start_time = time.time()
        
        if parallel:
            # Run tests in parallel
            tasks = [test.run(self.context) for test in self.tests]
            results = await asyncio.gather(*tasks)
        else:
            # Run tests sequentially
            results = []
            for test in self.tests:
                result = await test.run(self.context)
                results.append(result)
        
        execution_time = time.time() - start_time
        
        # Calculate summary statistics
        passed = sum(1 for r in results if r.status == ValidationStatus.PASSED)
        failed = sum(1 for r in results if r.status == ValidationStatus.FAILED)
        warnings = sum(1 for r in results if r.status == ValidationStatus.WARNING)
        skipped = sum(1 for r in results if r.status == ValidationStatus.SKIPPED)
        
        return ValidationSuiteResult(
            suite_name=self.name,
            total_tests=len(self.tests),
            passed=passed,
            failed=failed,
            warnings=warnings,
            skipped=skipped,
            execution_time=execution_time,
            timestamp=start_time,
            results=results
        )
    
    async def run_specific(self, test_names: List[str]) -> ValidationSuiteResult:
        """Run specific validation tests"""
        filtered_tests = [test for test in self.tests if test.name in test_names]
        
        # Temporarily replace tests list
        original_tests = self.tests
        self.tests = filtered_tests
        
        try:
            return await self.run_all()
        finally:
            self.tests = original_tests
    
    def get_test_names(self) -> List[str]:
        """Get list of available test names"""
        return [test.name for test in self.tests]
    
    def export_results(self, results: ValidationSuiteResult, filename: str):
        """Export validation results to JSON file"""
        data = {
            'suite_name': results.suite_name,
            'total_tests': results.total_tests,
            'passed': results.passed,
            'failed': results.failed,
            'warnings': results.warnings,
            'skipped': results.skipped,
            'execution_time': results.execution_time,
            'timestamp': results.timestamp,
            'results': [
                {
                    'test_name': r.test_name,
                    'status': r.status.value,
                    'message': r.message,
                    'execution_time': r.execution_time,
                    'timestamp': r.timestamp,
                    'details': r.details,
                    'error': r.error
                }
                for r in results.results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)


# Pre-configured validation suites
def create_monitoring_validation_suite(metrics_collector=None, dashboard_manager=None, alert_manager=None) -> ValidationSuite:
    """Create a comprehensive monitoring validation suite"""
    suite = ValidationSuite("PRSM Monitoring Validation Suite")
    
    if metrics_collector:
        suite.add_test(MetricsValidationTest(metrics_collector))
    
    if dashboard_manager:
        # Assume dashboard runs on port 3000 by default
        suite.add_test(DashboardValidationTest("http://localhost:3000"))
    
    if alert_manager:
        suite.add_test(AlertValidationTest(alert_manager))
    
    # Always add performance validation
    suite.add_test(PerformanceValidationTest())
    
    return suite