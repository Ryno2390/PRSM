"""
Regression Detection Tests
=========================

Automated regression testing framework for PRSM to detect breaking changes,
performance regressions, and API compatibility issues across system updates.

Regression Categories:
- Functional Regressions: Breaking changes in core functionality
- Performance Regressions: Degradation in system performance
- API Regressions: Breaking changes in API contracts
- Data Integrity Regressions: Issues with data consistency
- Security Regressions: Weakening of security measures
- Integration Regressions: Issues with external service integration
"""

import pytest
import asyncio
import json
import time
import hashlib
from decimal import Decimal
from typing import Dict, List, Any, Optional, Tuple, Set
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
import uuid
import statistics

try:
    from prsm.core.models import UserInput, PRSMResponse, AgentType, TaskStatus
    from prsm.nwtn.orchestrator import NWTNOrchestrator
    from prsm.tokenomics.ftns_service import FTNSService
    from prsm.api.main import app
    from prsm.core.database import DatabaseManager
    from prsm.auth.auth_manager import AuthManager
    from fastapi.testclient import TestClient
    from httpx import AsyncClient
except ImportError:
    # Create mocks if imports fail
    UserInput = Mock
    PRSMResponse = Mock
    AgentType = Mock
    TaskStatus = Mock
    NWTNOrchestrator = Mock
    FTNSService = Mock
    app = Mock()
    DatabaseManager = Mock
    AuthManager = Mock
    TestClient = Mock
    AsyncClient = Mock


@dataclass
class RegressionTestResult:
    """Result from a regression test"""
    test_name: str
    test_category: str
    passed: bool
    current_result: Any
    baseline_result: Any
    regression_detected: bool
    regression_severity: str  # "none", "low", "medium", "high", "critical"
    regression_details: Dict[str, Any]
    execution_time: float
    timestamp: str
    
    def __post_init__(self):
        if not hasattr(self, 'timestamp') or self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class RegressionDetector:
    """Core regression detection framework"""
    
    def __init__(self, baseline_file: str = "regression_baseline.json"):
        self.baseline_file = baseline_file
        self.baseline_data: Dict[str, Any] = {}
        self.current_results: Dict[str, Any] = {}
        self.regression_results: List[RegressionTestResult] = []
        
        self.load_baseline()
    
    def load_baseline(self):
        """Load baseline test results for comparison"""
        try:
            with open(self.baseline_file, 'r') as f:
                self.baseline_data = json.load(f)
            print(f"✅ Loaded baseline from {self.baseline_file}")
        except FileNotFoundError:
            print(f"⚠️  No baseline file found at {self.baseline_file}, will create new baseline")
            self.baseline_data = {}
    
    def save_baseline(self):
        """Save current results as new baseline"""
        with open(self.baseline_file, 'w') as f:
            json.dump(self.current_results, f, indent=2, default=str)
        print(f"💾 Saved new baseline to {self.baseline_file}")
    
    def detect_functional_regression(
        self, 
        test_name: str, 
        current_result: Any, 
        expected_keys: List[str] = None,
        tolerance: float = 0.01
    ) -> RegressionTestResult:
        """Detect functional regressions by comparing structured results"""
        
        start_time = time.perf_counter()
        baseline_result = self.baseline_data.get(test_name)
        execution_time = time.perf_counter() - start_time
        
        regression_detected = False
        regression_severity = "none"
        regression_details = {}
        
        if baseline_result is None:
            # No baseline - this is the first run
            self.current_results[test_name] = current_result
            regression_details = {"reason": "no_baseline", "message": "First run - establishing baseline"}
        else:
            # Compare with baseline
            if isinstance(current_result, dict) and isinstance(baseline_result, dict):
                regression_details = self._compare_dict_results(current_result, baseline_result, tolerance)
            elif isinstance(current_result, (int, float)) and isinstance(baseline_result, (int, float)):
                regression_details = self._compare_numeric_results(current_result, baseline_result, tolerance)
            elif isinstance(current_result, str) and isinstance(baseline_result, str):
                regression_details = self._compare_string_results(current_result, baseline_result)
            else:
                regression_details = self._compare_generic_results(current_result, baseline_result)
            
            # Determine regression severity
            if regression_details.get("differences"):
                regression_detected = True
                severity_score = len(regression_details["differences"])
                
                if severity_score >= 5:
                    regression_severity = "critical"
                elif severity_score >= 3:
                    regression_severity = "high" 
                elif severity_score >= 2:
                    regression_severity = "medium"
                else:
                    regression_severity = "low"
            
            self.current_results[test_name] = current_result
        
        # Check for expected keys if provided
        if expected_keys and isinstance(current_result, dict):
            missing_keys = [key for key in expected_keys if key not in current_result]
            if missing_keys:
                regression_detected = True
                regression_severity = "high"
                regression_details["missing_keys"] = missing_keys
        
        test_passed = not regression_detected or regression_severity in ["none", "low"]
        
        result = RegressionTestResult(
            test_name=test_name,
            test_category="functional",
            passed=test_passed,
            current_result=current_result,
            baseline_result=baseline_result,
            regression_detected=regression_detected,
            regression_severity=regression_severity,
            regression_details=regression_details,
            execution_time=execution_time
        )
        
        self.regression_results.append(result)
        return result
    
    def detect_performance_regression(
        self,
        test_name: str,
        current_time: float,
        acceptable_degradation: float = 0.20  # 20% slower is acceptable
    ) -> RegressionTestResult:
        """Detect performance regressions by comparing execution times"""
        
        baseline_time = self.baseline_data.get(f"{test_name}_performance")
        regression_detected = False
        regression_severity = "none"
        regression_details = {}
        
        if baseline_time is None:
            # No baseline performance data
            self.current_results[f"{test_name}_performance"] = current_time
            regression_details = {"reason": "no_baseline", "current_time": current_time}
        else:
            # Calculate performance change
            if baseline_time > 0:
                performance_change = (current_time - baseline_time) / baseline_time
                regression_details = {
                    "current_time": current_time,
                    "baseline_time": baseline_time,
                    "performance_change_percentage": performance_change * 100,
                    "acceptable_threshold": acceptable_degradation * 100
                }
                
                if performance_change > acceptable_degradation:
                    regression_detected = True
                    
                    if performance_change > 1.0:  # 100% slower
                        regression_severity = "critical"
                    elif performance_change > 0.5:  # 50% slower
                        regression_severity = "high"
                    elif performance_change > 0.3:  # 30% slower
                        regression_severity = "medium"
                    else:
                        regression_severity = "low"
                
                # Also detect significant improvements (might indicate test issues)
                elif performance_change < -0.5:  # 50% faster (suspicious)
                    regression_details["warning"] = "Significant performance improvement detected - verify test validity"
            
            self.current_results[f"{test_name}_performance"] = current_time
        
        test_passed = not regression_detected or regression_severity == "low"
        
        result = RegressionTestResult(
            test_name=f"{test_name}_performance",
            test_category="performance",
            passed=test_passed,
            current_result=current_time,
            baseline_result=baseline_time,
            regression_detected=regression_detected,
            regression_severity=regression_severity,
            regression_details=regression_details,
            execution_time=0.0  # Performance test itself is instantaneous
        )
        
        self.regression_results.append(result)
        return result
    
    def detect_api_regression(
        self,
        test_name: str,
        api_response: Dict[str, Any],
        expected_status_code: int = 200,
        required_fields: List[str] = None
    ) -> RegressionTestResult:
        """Detect API contract regressions"""
        
        start_time = time.perf_counter()
        baseline_response = self.baseline_data.get(f"{test_name}_api")
        execution_time = time.perf_counter() - start_time
        
        regression_detected = False
        regression_severity = "none"
        regression_details = {}
        
        # Check status code
        current_status = api_response.get("status_code", 0)
        if current_status != expected_status_code:
            regression_detected = True
            regression_severity = "high"
            regression_details["status_code_mismatch"] = {
                "expected": expected_status_code,
                "actual": current_status
            }
        
        # Check required fields
        if required_fields:
            response_data = api_response.get("data", {})
            missing_fields = [field for field in required_fields if field not in response_data]
            if missing_fields:
                regression_detected = True
                regression_severity = "high" if len(missing_fields) > 1 else "medium"
                regression_details["missing_required_fields"] = missing_fields
        
        # Compare with baseline API structure
        if baseline_response:
            baseline_data = baseline_response.get("data", {})
            current_data = api_response.get("data", {})
            
            if isinstance(baseline_data, dict) and isinstance(current_data, dict):
                # Check for removed fields (breaking change)
                removed_fields = set(baseline_data.keys()) - set(current_data.keys())
                if removed_fields:
                    regression_detected = True
                    regression_severity = "high"
                    regression_details["removed_fields"] = list(removed_fields)
                
                # Check for changed field types
                type_changes = []
                for key in baseline_data.keys():
                    if key in current_data:
                        baseline_type = type(baseline_data[key]).__name__
                        current_type = type(current_data[key]).__name__
                        if baseline_type != current_type:
                            type_changes.append({
                                "field": key,
                                "baseline_type": baseline_type,
                                "current_type": current_type
                            })
                
                if type_changes:
                    regression_detected = True
                    regression_severity = "medium"
                    regression_details["type_changes"] = type_changes
        
        self.current_results[f"{test_name}_api"] = api_response
        test_passed = not regression_detected or regression_severity == "low"
        
        result = RegressionTestResult(
            test_name=f"{test_name}_api",
            test_category="api",
            passed=test_passed,
            current_result=api_response,
            baseline_result=baseline_response,
            regression_detected=regression_detected,
            regression_severity=regression_severity,
            regression_details=regression_details,
            execution_time=execution_time
        )
        
        self.regression_results.append(result)
        return result
    
    def _compare_dict_results(self, current: Dict, baseline: Dict, tolerance: float) -> Dict[str, Any]:
        """Compare dictionary results for differences"""
        differences = []
        
        # Check for missing keys
        missing_keys = set(baseline.keys()) - set(current.keys())
        if missing_keys:
            differences.append(f"Missing keys: {list(missing_keys)}")
        
        # Check for extra keys
        extra_keys = set(current.keys()) - set(baseline.keys())
        if extra_keys:
            differences.append(f"Extra keys: {list(extra_keys)}")
        
        # Check for value differences
        for key in baseline.keys():
            if key in current:
                baseline_val = baseline[key]
                current_val = current[key]
                
                if isinstance(baseline_val, (int, float)) and isinstance(current_val, (int, float)):
                    if abs(current_val - baseline_val) > tolerance * abs(baseline_val):
                        differences.append(f"Value change in '{key}': {baseline_val} → {current_val}")
                elif baseline_val != current_val:
                    differences.append(f"Value change in '{key}': {baseline_val} → {current_val}")
        
        return {"differences": differences}
    
    def _compare_numeric_results(self, current: float, baseline: float, tolerance: float) -> Dict[str, Any]:
        """Compare numeric results"""
        if abs(current - baseline) > tolerance * abs(baseline):
            return {
                "differences": [f"Numeric value changed: {baseline} → {current}"],
                "change_percentage": ((current - baseline) / baseline) * 100 if baseline != 0 else float('inf')
            }
        return {"differences": []}
    
    def _compare_string_results(self, current: str, baseline: str) -> Dict[str, Any]:
        """Compare string results"""
        if current != baseline:
            # Calculate similarity
            current_hash = hashlib.md5(current.encode()).hexdigest()
            baseline_hash = hashlib.md5(baseline.encode()).hexdigest()
            
            return {
                "differences": [f"String content changed"],
                "current_length": len(current),
                "baseline_length": len(baseline),
                "content_hash_changed": current_hash != baseline_hash
            }
        return {"differences": []}
    
    def _compare_generic_results(self, current: Any, baseline: Any) -> Dict[str, Any]:
        """Compare generic results"""
        if current != baseline:
            return {
                "differences": [f"Generic value changed: {baseline} → {current}"],
                "current_type": type(current).__name__,
                "baseline_type": type(baseline).__name__
            }
        return {"differences": []}
    
    def generate_regression_report(self) -> Dict[str, Any]:
        """Generate comprehensive regression test report"""
        
        total_tests = len(self.regression_results)
        passed_tests = sum(1 for r in self.regression_results if r.passed)
        failed_tests = total_tests - passed_tests
        
        regressions_by_severity = {}
        for severity in ["critical", "high", "medium", "low", "none"]:
            regressions_by_severity[severity] = [
                r for r in self.regression_results 
                if r.regression_severity == severity
            ]
        
        regressions_by_category = {}
        for category in ["functional", "performance", "api", "data_integrity", "security", "integration"]:
            regressions_by_category[category] = [
                r for r in self.regression_results
                if r.test_category == category
            ]
        
        return {
            "summary": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "regression_rate": (total_tests - passed_tests) / total_tests if total_tests > 0 else 0,
                "overall_status": self._determine_overall_status()
            },
            "regressions_by_severity": {
                severity: len(regressions) for severity, regressions in regressions_by_severity.items()
            },
            "regressions_by_category": {
                category: len(regressions) for category, regressions in regressions_by_category.items()
            },
            "critical_regressions": [
                {
                    "test_name": r.test_name,
                    "category": r.test_category,
                    "details": r.regression_details
                }
                for r in regressions_by_severity["critical"]
            ],
            "detailed_results": [asdict(r) for r in self.regression_results],
            "recommendations": self._generate_recommendations()
        }
    
    def _determine_overall_status(self) -> str:
        """Determine overall regression test status"""
        critical_count = sum(1 for r in self.regression_results if r.regression_severity == "critical")
        high_count = sum(1 for r in self.regression_results if r.regression_severity == "high")
        failed_count = sum(1 for r in self.regression_results if not r.passed)
        
        if critical_count > 0:
            return "critical_regressions"
        elif high_count > 2:
            return "high_regressions"
        elif failed_count > len(self.regression_results) * 0.2:  # >20% failures
            return "multiple_regressions"
        elif failed_count > 0:
            return "minor_regressions"
        else:
            return "no_regressions"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on regression results"""
        recommendations = []
        
        critical_regressions = [r for r in self.regression_results if r.regression_severity == "critical"]
        high_regressions = [r for r in self.regression_results if r.regression_severity == "high"]
        performance_regressions = [r for r in self.regression_results if r.test_category == "performance" and r.regression_detected]
        api_regressions = [r for r in self.regression_results if r.test_category == "api" and r.regression_detected]
        
        if critical_regressions:
            recommendations.append(f"URGENT: Fix {len(critical_regressions)} critical regressions before deployment")
        
        if high_regressions:
            recommendations.append(f"HIGH PRIORITY: Address {len(high_regressions)} high-severity regressions")
        
        if performance_regressions:
            recommendations.append(f"Review performance optimizations - {len(performance_regressions)} performance regressions detected")
        
        if api_regressions:
            recommendations.append(f"Update API documentation and review breaking changes - {len(api_regressions)} API regressions detected")
        
        # Check for patterns
        functional_issues = sum(1 for r in self.regression_results if r.test_category == "functional" and r.regression_detected)
        if functional_issues > 3:
            recommendations.append("Multiple functional regressions suggest core logic changes - comprehensive testing recommended")
        
        return recommendations


@pytest.mark.regression
@pytest.mark.slow
class TestNWTNRegressionDetection:
    """Regression tests for NWTN orchestrator functionality"""
    
    @pytest.fixture
    def regression_detector(self):
        return RegressionDetector("nwtn_regression_baseline.json")
    
    @pytest.fixture
    def mock_nwtn_orchestrator(self):
        orchestrator = Mock(spec=NWTNOrchestrator)
        
        async def mock_process_query(user_input):
            # Consistent mock response for regression testing
            return {
                "session_id": "test_session_123",
                "final_answer": f"NWTN processed query: {user_input.prompt}",
                "reasoning_trace": [
                    {
                        "step_id": "step_1",
                        "agent_type": "architect",
                        "execution_time": 0.1,
                        "confidence_score": 0.9
                    },
                    {
                        "step_id": "step_2", 
                        "agent_type": "executor",
                        "execution_time": 0.3,
                        "confidence_score": 0.85
                    }
                ],
                "confidence_score": 0.87,
                "context_used": len(user_input.prompt),
                "processing_time": 0.4
            }
        
        orchestrator.process_query = mock_process_query
        return orchestrator
    
    async def test_nwtn_query_response_structure_regression(self, regression_detector, mock_nwtn_orchestrator):
        """Test for regressions in NWTN query response structure"""
        
        user_input = UserInput(
            user_id="regression_test_user",
            prompt="Test query for regression detection",
            context_allocation=100
        )
        
        start_time = time.perf_counter()
        result = await mock_nwtn_orchestrator.process_query(user_input)
        execution_time = time.perf_counter() - start_time
        
        # Test functional regression
        expected_keys = ["session_id", "final_answer", "reasoning_trace", "confidence_score", "context_used"]
        functional_result = regression_detector.detect_functional_regression(
            test_name="nwtn_query_response_structure",
            current_result=result,
            expected_keys=expected_keys
        )
        
        # Test performance regression
        performance_result = regression_detector.detect_performance_regression(
            test_name="nwtn_query_processing",
            current_time=execution_time,
            acceptable_degradation=0.25  # 25% slower is acceptable
        )
        
        # Assertions
        assert functional_result.passed, f"Functional regression detected: {functional_result.regression_details}"
        assert performance_result.passed, f"Performance regression detected: {performance_result.regression_details}"
        
        return functional_result, performance_result
    
    async def test_nwtn_reasoning_trace_regression(self, regression_detector, mock_nwtn_orchestrator):
        """Test for regressions in reasoning trace structure and content"""
        
        user_input = UserInput(
            user_id="trace_regression_user",
            prompt="Complex query requiring detailed reasoning trace",
            context_allocation=200
        )
        
        result = await mock_nwtn_orchestrator.process_query(user_input)
        reasoning_trace = result.get("reasoning_trace", [])
        
        # Analyze reasoning trace structure
        trace_analysis = {
            "trace_length": len(reasoning_trace),
            "agent_types": [step.get("agent_type") for step in reasoning_trace],
            "total_execution_time": sum(step.get("execution_time", 0) for step in reasoning_trace),
            "average_confidence": statistics.mean([step.get("confidence_score", 0) for step in reasoning_trace]) if reasoning_trace else 0,
            "has_required_fields": all(
                all(field in step for field in ["step_id", "agent_type", "execution_time"])
                for step in reasoning_trace
            )
        }
        
        regression_result = regression_detector.detect_functional_regression(
            test_name="nwtn_reasoning_trace_analysis",
            current_result=trace_analysis,
            expected_keys=["trace_length", "agent_types", "total_execution_time", "average_confidence"]
        )
        
        assert regression_result.passed, f"Reasoning trace regression: {regression_result.regression_details}"
        assert trace_analysis["has_required_fields"], "Reasoning trace missing required fields"
        
        return regression_result
    
    async def test_nwtn_confidence_scoring_regression(self, regression_detector, mock_nwtn_orchestrator):
        """Test for regressions in confidence scoring algorithms"""
        
        test_queries = [
            "Simple factual question",
            "Complex analytical question requiring multiple reasoning steps",
            "Ambiguous question with multiple possible interpretations"
        ]
        
        confidence_scores = []
        
        for query in test_queries:
            user_input = UserInput(
                user_id="confidence_test_user",
                prompt=query,
                context_allocation=150
            )
            
            result = await mock_nwtn_orchestrator.process_query(user_input)
            confidence_scores.append(result.get("confidence_score", 0))
        
        confidence_analysis = {
            "scores": confidence_scores,
            "average_confidence": statistics.mean(confidence_scores),
            "confidence_range": max(confidence_scores) - min(confidence_scores),
            "all_scores_valid": all(0 <= score <= 1 for score in confidence_scores),
            "score_distribution": {
                "high_confidence": sum(1 for score in confidence_scores if score > 0.8),
                "medium_confidence": sum(1 for score in confidence_scores if 0.5 < score <= 0.8),
                "low_confidence": sum(1 for score in confidence_scores if score <= 0.5)
            }
        }
        
        regression_result = regression_detector.detect_functional_regression(
            test_name="nwtn_confidence_scoring",
            current_result=confidence_analysis,
            tolerance=0.1  # 10% tolerance for confidence scores
        )
        
        assert regression_result.passed, f"Confidence scoring regression: {regression_result.regression_details}"
        assert confidence_analysis["all_scores_valid"], "Invalid confidence scores detected"
        
        return regression_result


@pytest.mark.regression
@pytest.mark.slow
class TestFTNSRegressionDetection:
    """Regression tests for FTNS tokenomics functionality"""
    
    @pytest.fixture
    def regression_detector(self):
        return RegressionDetector("ftns_regression_baseline.json")
    
    @pytest.fixture
    def mock_ftns_service(self):
        service = Mock(spec=FTNSService)
        
        def mock_get_balance(user_id):
            return {
                "total_balance": Decimal("100.00"),
                "available_balance": Decimal("85.50"),
                "reserved_balance": Decimal("14.50"),
                "last_transaction_id": "tx_12345",
                "balance_timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        def mock_create_transaction(from_user, to_user, amount, transaction_type):
            return {
                "transaction_id": "tx_67890",
                "success": True,
                "amount": amount,
                "transaction_type": transaction_type,
                "fee": amount * 0.01,  # 1% fee
                "new_balance": Decimal("90.00"),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        service.get_balance = mock_get_balance
        service.create_transaction = mock_create_transaction
        return service
    
    def test_ftns_balance_calculation_regression(self, regression_detector, mock_ftns_service):
        """Test for regressions in FTNS balance calculations"""
        
        start_time = time.perf_counter()
        balance_result = mock_ftns_service.get_balance("regression_test_user")
        execution_time = time.perf_counter() - start_time
        
        # Normalize balance for comparison (convert Decimal to float)
        normalized_balance = {
            "total_balance": float(balance_result["total_balance"]),
            "available_balance": float(balance_result["available_balance"]),
            "reserved_balance": float(balance_result["reserved_balance"]),
            "balance_components_sum": float(balance_result["available_balance"]) + float(balance_result["reserved_balance"]),
            "has_transaction_id": "last_transaction_id" in balance_result,
            "has_timestamp": "balance_timestamp" in balance_result
        }
        
        # Check balance calculation consistency
        balance_sum_matches = abs(normalized_balance["total_balance"] - normalized_balance["balance_components_sum"]) < 0.01
        normalized_balance["balance_calculation_consistent"] = balance_sum_matches
        
        functional_result = regression_detector.detect_functional_regression(
            test_name="ftns_balance_calculation",
            current_result=normalized_balance,
            expected_keys=["total_balance", "available_balance", "reserved_balance"]
        )
        
        performance_result = regression_detector.detect_performance_regression(
            test_name="ftns_balance_lookup",
            current_time=execution_time,
            acceptable_degradation=0.30
        )
        
        assert functional_result.passed, f"Balance calculation regression: {functional_result.regression_details}"
        assert performance_result.passed, f"Balance lookup performance regression: {performance_result.regression_details}"
        assert balance_sum_matches, "Balance components don't sum to total balance"
        
        return functional_result, performance_result
    
    def test_ftns_transaction_processing_regression(self, regression_detector, mock_ftns_service):
        """Test for regressions in transaction processing logic"""
        
        start_time = time.perf_counter()
        transaction_result = mock_ftns_service.create_transaction(
            from_user="user_a",
            to_user="user_b", 
            amount=10.0,
            transaction_type="transfer"
        )
        execution_time = time.perf_counter() - start_time
        
        # Normalize transaction result
        normalized_transaction = {
            "success": transaction_result["success"],
            "amount": transaction_result["amount"],
            "transaction_type": transaction_result["transaction_type"],
            "fee": transaction_result["fee"],
            "fee_percentage": (transaction_result["fee"] / transaction_result["amount"]) * 100,
            "new_balance": float(transaction_result["new_balance"]),
            "has_transaction_id": "transaction_id" in transaction_result,
            "has_timestamp": "timestamp" in transaction_result,
            "fee_calculation_correct": abs(transaction_result["fee"] - (transaction_result["amount"] * 0.01)) < 0.001
        }
        
        functional_result = regression_detector.detect_functional_regression(
            test_name="ftns_transaction_processing",
            current_result=normalized_transaction,
            expected_keys=["success", "amount", "transaction_type", "fee", "new_balance"]
        )
        
        performance_result = regression_detector.detect_performance_regression(
            test_name="ftns_transaction_creation",
            current_time=execution_time,
            acceptable_degradation=0.25
        )
        
        assert functional_result.passed, f"Transaction processing regression: {functional_result.regression_details}"
        assert performance_result.passed, f"Transaction performance regression: {performance_result.regression_details}"
        assert normalized_transaction["fee_calculation_correct"], "Transaction fee calculation incorrect"
        
        return functional_result, performance_result
    
    def test_ftns_economic_model_regression(self, regression_detector, mock_ftns_service):
        """Test for regressions in overall economic model consistency"""
        
        # Simulate a series of transactions to test economic model
        initial_balance = mock_ftns_service.get_balance("economic_test_user")
        
        transactions = []
        for i in range(5):
            transaction = mock_ftns_service.create_transaction(
                from_user="economic_test_user",
                to_user=f"recipient_{i}",
                amount=5.0,
                transaction_type="payment"
            )
            transactions.append(transaction)
        
        final_balance = mock_ftns_service.get_balance("economic_test_user")
        
        # Economic model analysis
        economic_analysis = {
            "initial_balance": float(initial_balance["total_balance"]),
            "final_balance": float(final_balance["total_balance"]),
            "total_transactions": len(transactions),
            "total_amount_sent": sum(tx["amount"] for tx in transactions),
            "total_fees_paid": sum(tx["fee"] for tx in transactions),
            "expected_final_balance": float(initial_balance["total_balance"]) - sum(tx["amount"] + tx["fee"] for tx in transactions),
            "balance_consistency": abs(
                float(final_balance["total_balance"]) - 
                (float(initial_balance["total_balance"]) - sum(tx["amount"] + tx["fee"] for tx in transactions))
            ) < 0.01,
            "average_fee_percentage": statistics.mean([(tx["fee"] / tx["amount"]) * 100 for tx in transactions]),
            "all_transactions_successful": all(tx["success"] for tx in transactions)
        }
        
        regression_result = regression_detector.detect_functional_regression(
            test_name="ftns_economic_model_consistency",
            current_result=economic_analysis,
            tolerance=0.05  # 5% tolerance for economic calculations
        )
        
        assert regression_result.passed, f"Economic model regression: {regression_result.regression_details}"
        assert economic_analysis["balance_consistency"], "Economic model balance calculations inconsistent"
        assert economic_analysis["all_transactions_successful"], "Some transactions failed unexpectedly"
        
        return regression_result


@pytest.mark.regression 
@pytest.mark.api
class TestAPIRegressionDetection:
    """Regression tests for API contract compliance"""
    
    @pytest.fixture
    def regression_detector(self):
        return RegressionDetector("api_regression_baseline.json")
    
    async def test_nwtn_query_api_regression(self, regression_detector):
        """Test for regressions in NWTN query API contract"""
        
        # Mock API response
        api_response = {
            "status_code": 200,
            "data": {
                "session_id": "api_test_session_123",
                "final_answer": "API test response",
                "reasoning_trace": [
                    {"step_id": "1", "agent_type": "architect", "execution_time": 0.1}
                ],
                "confidence_score": 0.85,
                "context_used": 50,
                "processing_time": 0.4,
                "metadata": {
                    "api_version": "v1",
                    "model_version": "nwtn_v2.1"
                }
            },
            "headers": {
                "Content-Type": "application/json",
                "X-Response-Time": "400ms"
            }
        }
        
        regression_result = regression_detector.detect_api_regression(
            test_name="nwtn_query_endpoint",
            api_response=api_response,
            expected_status_code=200,
            required_fields=["session_id", "final_answer", "confidence_score", "context_used"]
        )
        
        assert regression_result.passed, f"NWTN API regression detected: {regression_result.regression_details}"
        
        return regression_result
    
    async def test_ftns_balance_api_regression(self, regression_detector):
        """Test for regressions in FTNS balance API contract"""
        
        api_response = {
            "status_code": 200,
            "data": {
                "total_balance": 100.00,
                "available_balance": 85.50,
                "reserved_balance": 14.50,
                "currency": "FTNS",
                "last_transaction": {
                    "transaction_id": "tx_12345",
                    "timestamp": "2025-01-23T10:30:00Z",
                    "amount": -5.0,
                    "type": "charge"
                },
                "balance_history_available": True
            },
            "headers": {
                "Content-Type": "application/json",
                "Cache-Control": "no-cache"
            }
        }
        
        regression_result = regression_detector.detect_api_regression(
            test_name="ftns_balance_endpoint",
            api_response=api_response,
            expected_status_code=200,
            required_fields=["total_balance", "available_balance", "currency"]
        )
        
        assert regression_result.passed, f"FTNS balance API regression detected: {regression_result.regression_details}"
        
        return regression_result
    
    async def test_authentication_api_regression(self, regression_detector):
        """Test for regressions in authentication API contract"""
        
        # Test successful authentication response
        auth_response = {
            "status_code": 200,
            "data": {
                "access_token": "jwt_token_abc123",
                "token_type": "bearer",
                "expires_in": 3600,
                "refresh_token": "refresh_token_def456",
                "user_info": {
                    "user_id": "auth_test_user",
                    "username": "testuser",
                    "role": "researcher"
                },
                "permissions": ["query", "balance_view", "transaction_create"]
            },
            "headers": {
                "Content-Type": "application/json",
                "Set-Cookie": "session_id=abc123; HttpOnly; Secure"
            }
        }
        
        regression_result = regression_detector.detect_api_regression(
            test_name="authentication_endpoint",
            api_response=auth_response,
            expected_status_code=200,
            required_fields=["access_token", "token_type", "expires_in", "user_info"]
        )
        
        assert regression_result.passed, f"Authentication API regression detected: {regression_result.regression_details}"
        
        return regression_result


@pytest.mark.regression 
@pytest.mark.integration
class TestDataIntegrityRegressionDetection:
    """Regression tests for data integrity and consistency"""
    
    @pytest.fixture
    def regression_detector(self):
        return RegressionDetector("data_integrity_regression_baseline.json")
    
    def test_database_schema_regression(self, regression_detector):
        """Test for regressions in database schema and constraints"""
        
        # Mock database schema information
        schema_info = {
            "tables": {
                "prsm_sessions": {
                    "columns": ["session_id", "user_id", "nwtn_context_allocation", "context_used", "status", "created_at"],
                    "primary_key": "session_id",
                    "foreign_keys": ["user_id"],
                    "indexes": ["user_id", "created_at", "status"],
                    "constraints": ["unique_session_id", "non_negative_context"]
                },
                "ftns_transactions": {
                    "columns": ["transaction_id", "from_user", "to_user", "amount", "transaction_type", "created_at"],
                    "primary_key": "transaction_id", 
                    "foreign_keys": ["from_user", "to_user"],
                    "indexes": ["from_user", "to_user", "created_at"],
                    "constraints": ["positive_amount", "valid_transaction_type"]
                },
                "users": {
                    "columns": ["user_id", "username", "email", "role", "is_active", "created_at"],
                    "primary_key": "user_id",
                    "foreign_keys": [],
                    "indexes": ["username", "email"],
                    "constraints": ["unique_username", "unique_email", "valid_email_format"]
                }
            },
            "total_tables": 3,
            "total_columns": 19,
            "total_constraints": 7,
            "schema_version": "v2.1.0"
        }
        
        regression_result = regression_detector.detect_functional_regression(
            test_name="database_schema_structure",
            current_result=schema_info,
            expected_keys=["tables", "total_tables", "schema_version"]
        )
        
        assert regression_result.passed, f"Database schema regression: {regression_result.regression_details}"
        
        return regression_result
    
    def test_data_validation_regression(self, regression_detector):
        """Test for regressions in data validation rules"""
        
        # Test various data validation scenarios
        validation_tests = {
            "user_input_validation": {
                "valid_prompts": ["Valid question", "Another valid query"],
                "invalid_prompts": ["", None, "x" * 10001],  # Empty, None, too long
                "context_allocation_valid": [100, 500, 1000],
                "context_allocation_invalid": [-10, 0, 10001],
                "validation_pass_rate": 0.75  # 75% should pass
            },
            "ftns_amount_validation": {
                "valid_amounts": [0.01, 1.0, 100.50, 999.99],
                "invalid_amounts": [-1.0, 0.0, 10000.0, "not_a_number"],
                "precision_test": [1.123456789, 0.001, 999.999],
                "validation_rules": ["positive", "max_precision_2", "max_value_9999"]
            },
            "session_status_validation": {
                "valid_statuses": ["pending", "in_progress", "completed", "failed", "cancelled"],
                "invalid_statuses": ["unknown", "", None, "invalid_status"],
                "status_transitions": {
                    "pending": ["in_progress", "cancelled"],
                    "in_progress": ["completed", "failed", "cancelled"],
                    "completed": [],
                    "failed": [],
                    "cancelled": []
                }
            }
        }
        
        regression_result = regression_detector.detect_functional_regression(
            test_name="data_validation_rules", 
            current_result=validation_tests,
            tolerance=0.1  # 10% tolerance for validation pass rates
        )
        
        assert regression_result.passed, f"Data validation regression: {regression_result.regression_details}"
        
        return regression_result
    
    def test_data_consistency_regression(self, regression_detector):
        """Test for regressions in cross-system data consistency"""
        
        # Mock data consistency checks
        consistency_checks = {
            "user_balance_consistency": {
                "users_checked": 100,
                "consistent_balances": 98,
                "inconsistencies_found": 2,
                "consistency_rate": 0.98,
                "max_discrepancy": 0.05  # $0.05 max discrepancy
            },
            "session_context_consistency": {
                "sessions_checked": 250,
                "context_allocation_matches": 248,
                "context_usage_within_allocation": 250,
                "consistency_rate": 0.992,
                "average_utilization": 0.75
            },
            "transaction_ledger_consistency": {
                "transactions_verified": 500,
                "double_entry_balanced": 499,
                "fee_calculations_correct": 500,
                "timestamp_ordering_correct": 498,
                "overall_consistency": 0.996
            },
            "referential_integrity": {
                "foreign_key_violations": 0,
                "orphaned_records": 1,
                "circular_references": 0,
                "integrity_score": 0.999
            }
        }
        
        regression_result = regression_detector.detect_functional_regression(
            test_name="cross_system_data_consistency",
            current_result=consistency_checks,
            tolerance=0.02  # 2% tolerance for consistency rates
        )
        
        assert regression_result.passed, f"Data consistency regression: {regression_result.regression_details}"
        
        # Additional assertions for critical consistency requirements
        assert consistency_checks["user_balance_consistency"]["consistency_rate"] > 0.95, "User balance consistency below threshold"
        assert consistency_checks["referential_integrity"]["foreign_key_violations"] == 0, "Foreign key violations detected"
        
        return regression_result


@pytest.mark.regression
class TestComprehensiveRegressionSuite:
    """Comprehensive regression test suite runner"""
    
    async def test_full_regression_suite(self):
        """Run complete regression test suite and generate report"""
        
        print("🔍 Starting PRSM Comprehensive Regression Test Suite...")
        print("=" * 70)
        
        # Initialize regression detectors for different categories
        detectors = {
            "nwtn": RegressionDetector("nwtn_regression_baseline.json"),
            "ftns": RegressionDetector("ftns_regression_baseline.json"), 
            "api": RegressionDetector("api_regression_baseline.json"),
            "data": RegressionDetector("data_integrity_regression_baseline.json")
        }
        
        all_results = []
        
        try:
            # NWTN Regression Tests
            print("🎭 Running NWTN Regression Tests...")
            nwtn_tests = TestNWTNRegressionDetection()
            
            functional_result, performance_result = await nwtn_tests.test_nwtn_query_response_structure_regression(
                detectors["nwtn"], nwtn_tests.mock_nwtn_orchestrator()
            )
            all_results.extend([functional_result, performance_result])
            print(f"  ✅ Query Structure: {'PASS' if functional_result.passed else 'FAIL'}")
            
            trace_result = await nwtn_tests.test_nwtn_reasoning_trace_regression(
                detectors["nwtn"], nwtn_tests.mock_nwtn_orchestrator()
            )
            all_results.append(trace_result)
            print(f"  ✅ Reasoning Trace: {'PASS' if trace_result.passed else 'FAIL'}")
            
            confidence_result = await nwtn_tests.test_nwtn_confidence_scoring_regression(
                detectors["nwtn"], nwtn_tests.mock_nwtn_orchestrator()
            )
            all_results.append(confidence_result)
            print(f"  ✅ Confidence Scoring: {'PASS' if confidence_result.passed else 'FAIL'}")
            
        except Exception as e:
            print(f"  ❌ NWTN regression tests failed: {e}")
        
        try:
            # FTNS Regression Tests
            print("\n💰 Running FTNS Regression Tests...")
            ftns_tests = TestFTNSRegressionDetection()
            
            balance_func, balance_perf = ftns_tests.test_ftns_balance_calculation_regression(
                detectors["ftns"], ftns_tests.mock_ftns_service()
            )
            all_results.extend([balance_func, balance_perf])
            print(f"  ✅ Balance Calculation: {'PASS' if balance_func.passed else 'FAIL'}")
            
            tx_func, tx_perf = ftns_tests.test_ftns_transaction_processing_regression(
                detectors["ftns"], ftns_tests.mock_ftns_service()
            )
            all_results.extend([tx_func, tx_perf])
            print(f"  ✅ Transaction Processing: {'PASS' if tx_func.passed else 'FAIL'}")
            
            economic_result = ftns_tests.test_ftns_economic_model_regression(
                detectors["ftns"], ftns_tests.mock_ftns_service()
            )
            all_results.append(economic_result)
            print(f"  ✅ Economic Model: {'PASS' if economic_result.passed else 'FAIL'}")
            
        except Exception as e:
            print(f"  ❌ FTNS regression tests failed: {e}")
        
        try:
            # API Regression Tests
            print("\n🌐 Running API Regression Tests...")
            api_tests = TestAPIRegressionDetection()
            
            nwtn_api_result = await api_tests.test_nwtn_query_api_regression(detectors["api"])
            all_results.append(nwtn_api_result)
            print(f"  ✅ NWTN Query API: {'PASS' if nwtn_api_result.passed else 'FAIL'}")
            
            balance_api_result = await api_tests.test_ftns_balance_api_regression(detectors["api"])
            all_results.append(balance_api_result)
            print(f"  ✅ FTNS Balance API: {'PASS' if balance_api_result.passed else 'FAIL'}")
            
            auth_api_result = await api_tests.test_authentication_api_regression(detectors["api"])
            all_results.append(auth_api_result)
            print(f"  ✅ Authentication API: {'PASS' if auth_api_result.passed else 'FAIL'}")
            
        except Exception as e:
            print(f"  ❌ API regression tests failed: {e}")
        
        try:
            # Data Integrity Regression Tests
            print("\n🗄️  Running Data Integrity Regression Tests...")
            data_tests = TestDataIntegrityRegressionDetection()
            
            schema_result = data_tests.test_database_schema_regression(detectors["data"])
            all_results.append(schema_result)
            print(f"  ✅ Database Schema: {'PASS' if schema_result.passed else 'FAIL'}")
            
            validation_result = data_tests.test_data_validation_regression(detectors["data"])
            all_results.append(validation_result)
            print(f"  ✅ Data Validation: {'PASS' if validation_result.passed else 'FAIL'}")
            
            consistency_result = data_tests.test_data_consistency_regression(detectors["data"])
            all_results.append(consistency_result)
            print(f"  ✅ Data Consistency: {'PASS' if consistency_result.passed else 'FAIL'}")
            
        except Exception as e:
            print(f"  ❌ Data integrity regression tests failed: {e}")
        
        # Generate comprehensive regression report
        print(f"\n📊 Generating Comprehensive Regression Report...")
        
        # Combine all regression results
        combined_detector = RegressionDetector("combined_regression_baseline.json")
        combined_detector.regression_results = all_results
        
        regression_report = combined_detector.generate_regression_report()
        
        # Save detailed report
        with open("comprehensive_regression_report.json", 'w') as f:
            json.dump(regression_report, f, indent=2, default=str)
        
        # Print summary
        print("=" * 70)
        print("📋 COMPREHENSIVE REGRESSION TEST SUMMARY")
        print("=" * 70)
        
        summary = regression_report["summary"]
        print(f"🎯 Overall Status: {summary['overall_status'].upper().replace('_', ' ')}")
        print(f"📊 Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed_tests']}")
        print(f"❌ Failed: {summary['failed_tests']}")
        print(f"📉 Regression Rate: {summary['regression_rate']:.1%}")
        
        severity_counts = regression_report["regressions_by_severity"]
        print(f"\n🚨 Regressions by Severity:")
        print(f"  🔴 Critical: {severity_counts['critical']}")
        print(f"  🟠 High: {severity_counts['high']}")
        print(f"  🟡 Medium: {severity_counts['medium']}")
        print(f"  🟢 Low: {severity_counts['low']}")
        
        if regression_report["critical_regressions"]:
            print(f"\n🚨 CRITICAL REGRESSIONS:")
            for regression in regression_report["critical_regressions"]:
                print(f"  ❌ {regression['test_name']} ({regression['category']})")
                print(f"     {regression['details']}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for recommendation in regression_report["recommendations"]:
            print(f"  • {recommendation}")
        
        print(f"\n📄 Detailed report saved: comprehensive_regression_report.json")
        
        # Save updated baselines if no critical regressions
        if severity_counts['critical'] == 0:
            for detector in detectors.values():
                detector.save_baseline()
            print("💾 Updated regression baselines saved")
        
        # Final assertions
        assert len(all_results) > 0, "No regression tests completed successfully"
        assert severity_counts['critical'] == 0, f"Critical regressions detected: {regression_report['critical_regressions']}"
        assert summary['regression_rate'] < 0.3, f"Regression rate too high: {summary['regression_rate']:.1%}"
        
        print(f"\n🎉 Regression test suite completed successfully!")
        
        return regression_report