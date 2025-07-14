#!/usr/bin/env python3
"""
NWTN Multi-Modal Reasoning Validation Framework
Comprehensive validation and testing framework for the multi-modal reasoning system

This module provides a complete validation framework to ensure the reliability,
accuracy, and effectiveness of NWTN's multi-modal reasoning capabilities.

Key Components:
1. Individual Engine Validation - Test each reasoning engine independently
2. Network Validation Testing - Validate the network validation system
3. Integration Testing - Test multi-modal reasoning integration
4. Performance Benchmarking - Measure system performance
5. Accuracy Assessment - Evaluate reasoning accuracy
6. Breakthrough Discovery Validation - Test discovery capabilities

Validation Methods:
- Unit tests for individual reasoning engines
- Integration tests for multi-modal reasoning
- Network validation system tests
- Performance benchmarks
- Accuracy metrics
- Comparative analysis with baseline systems

Usage:
    from prsm.nwtn.validation_framework import ValidationFramework
    
    validator = ValidationFramework()
    results = await validator.run_comprehensive_validation()
"""

import asyncio
import json
import math
import statistics
import time
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4
from datetime import datetime, timezone
from collections import defaultdict

import structlog
from pydantic import BaseModel, Field

from prsm.nwtn.multi_modal_reasoning_engine import MultiModalReasoningEngine
from prsm.nwtn.network_validation_engine import NetworkValidationEngine, ValidationMethod
from prsm.nwtn.analogical_breakthrough_engine import AnalogicalBreakthroughEngine
from prsm.nwtn.deductive_reasoning_engine import DeductiveReasoningEngine
from prsm.nwtn.inductive_reasoning_engine import InductiveReasoningEngine
from prsm.nwtn.abductive_reasoning_engine import AbductiveReasoningEngine
from prsm.nwtn.causal_reasoning_engine import CausalReasoningEngine
from prsm.nwtn.probabilistic_reasoning_engine import ProbabilisticReasoningEngine
from prsm.nwtn.counterfactual_reasoning_engine import CounterfactualReasoningEngine
from prsm.agents.executors.model_executor import ModelExecutor

logger = structlog.get_logger(__name__)


class ValidationLevel(str, Enum):
    """Levels of validation testing"""
    UNIT = "unit"                       # Individual engine tests
    INTEGRATION = "integration"         # Multi-modal integration tests
    NETWORK = "network"                 # Network validation tests
    PERFORMANCE = "performance"         # Performance benchmarks
    ACCURACY = "accuracy"               # Accuracy assessment
    COMPREHENSIVE = "comprehensive"     # All validation levels


class TestCategory(str, Enum):
    """Categories of validation tests"""
    FUNCTIONAL = "functional"           # Functional correctness
    PERFORMANCE = "performance"         # Performance metrics
    ACCURACY = "accuracy"               # Accuracy metrics
    RELIABILITY = "reliability"         # Reliability and consistency
    SCALABILITY = "scalability"         # Scalability tests
    INTEGRATION = "integration"         # Integration tests


class TestStatus(str, Enum):
    """Status of validation tests"""
    PENDING = "pending"                 # Not yet run
    RUNNING = "running"                 # Currently running
    PASSED = "passed"                   # Test passed
    FAILED = "failed"                   # Test failed
    SKIPPED = "skipped"                 # Test skipped


@dataclass
class ValidationTest:
    """A single validation test"""
    
    id: str
    name: str
    description: str
    category: TestCategory
    level: ValidationLevel
    
    # Test configuration
    test_function: str
    test_parameters: Dict[str, Any] = field(default_factory=dict)
    expected_result: Any = None
    
    # Test execution
    status: TestStatus = TestStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: float = 0.0
    
    # Test results
    actual_result: Any = None
    success: bool = False
    error_message: Optional[str] = None
    
    # Metrics
    accuracy_score: float = 0.0
    performance_score: float = 0.0
    reliability_score: float = 0.0
    
    # Test data
    test_data: Dict[str, Any] = field(default_factory=dict)
    validation_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Results of validation testing"""
    
    id: str
    validation_level: ValidationLevel
    
    # Test summary
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    success_rate: float
    
    # Performance metrics
    total_execution_time: float
    average_test_time: float
    fastest_test_time: float
    slowest_test_time: float
    
    # Accuracy metrics
    overall_accuracy: float
    engine_accuracies: Dict[str, float]
    
    # Test results
    test_results: List[ValidationTest]
    failed_test_details: List[str]
    
    # Insights
    validation_insights: List[str]
    performance_insights: List[str]
    recommendations: List[str]
    
    # Metadata
    validation_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    system_info: Dict[str, Any] = field(default_factory=dict)


class ValidationFramework:
    """
    Comprehensive validation framework for multi-modal reasoning system
    
    This framework provides extensive testing capabilities to ensure the
    reliability, accuracy, and effectiveness of NWTN's reasoning capabilities.
    """
    
    def __init__(self):
        self.model_executor = ModelExecutor(agent_id="validation_framework")
        
        # Initialize all engines for testing
        self.multi_modal_engine = MultiModalReasoningEngine()
        self.network_validator = NetworkValidationEngine()
        self.analogical_engine = AnalogicalBreakthroughEngine()
        self.deductive_engine = DeductiveReasoningEngine()
        self.inductive_engine = InductiveReasoningEngine()
        self.abductive_engine = AbductiveReasoningEngine()
        self.causal_engine = CausalReasoningEngine()
        self.probabilistic_engine = ProbabilisticReasoningEngine()
        self.counterfactual_engine = CounterfactualReasoningEngine()
        
        # Validation storage
        self.validation_results: List[ValidationResult] = []
        self.test_registry: Dict[str, ValidationTest] = {}
        
        # Test configurations
        self.test_timeout = 60  # seconds
        self.accuracy_threshold = 0.7
        self.performance_threshold = 5.0  # seconds
        self.reliability_threshold = 0.8
        
        # Test data
        self.test_queries = self._load_test_queries()
        self.benchmark_data = self._load_benchmark_data()
        
        logger.info("Initialized Validation Framework")
    
    def _load_test_queries(self) -> Dict[str, List[str]]:
        """Load test queries for validation"""
        
        return {
            "deductive": [
                "All mammals are warm-blooded. Dogs are mammals. Therefore, dogs are warm-blooded.",
                "If it rains, then the ground gets wet. It is raining. What can we conclude?",
                "No birds are mammals. Penguins are birds. Are penguins mammals?"
            ],
            "inductive": [
                "The sun has risen every day for the past 10,000 years. What can we predict about tomorrow?",
                "In 100 coin flips, 52 came up heads. What is the likely bias of the coin?",
                "Every swan I've seen is white. What can we conclude about swan colors?"
            ],
            "abductive": [
                "The grass is wet. What is the most likely explanation?",
                "The patient has a fever, headache, and body aches. What might be the diagnosis?",
                "The computer won't turn on. What could be the problem?"
            ],
            "analogical": [
                "How is the structure of an atom similar to the solar system?",
                "What can we learn about managing a company from ant colonies?",
                "How is DNA like a blueprint for a building?"
            ],
            "causal": [
                "What causes inflation in an economy?",
                "How does exercise affect cardiovascular health?",
                "What are the causal factors in climate change?"
            ],
            "probabilistic": [
                "What is the probability of rolling two sixes with fair dice?",
                "Given that it's cloudy, what's the chance of rain?",
                "What are the odds of a new drug being effective?"
            ],
            "counterfactual": [
                "What would have happened if the internet was never invented?",
                "If gravity were twice as strong, how would life be different?",
                "What if humans had evolved from reptiles instead of mammals?"
            ]
        }
    
    def _load_benchmark_data(self) -> Dict[str, Any]:
        """Load benchmark data for performance testing"""
        
        return {
            "response_time_targets": {
                "deductive": 1.0,
                "inductive": 2.0,
                "abductive": 1.5,
                "analogical": 3.0,
                "causal": 2.5,
                "probabilistic": 2.0,
                "counterfactual": 3.0
            },
            "accuracy_targets": {
                "deductive": 0.95,
                "inductive": 0.80,
                "abductive": 0.75,
                "analogical": 0.70,
                "causal": 0.80,
                "probabilistic": 0.85,
                "counterfactual": 0.70
            },
            "network_validation_targets": {
                "consensus_rate": 0.75,
                "validation_efficiency": 0.80,
                "result_quality": 0.75
            }
        }
    
    async def run_comprehensive_validation(self) -> ValidationResult:
        """Run comprehensive validation across all levels"""
        
        logger.info("Starting comprehensive validation")
        
        # Run all validation levels
        unit_results = await self.run_unit_validation()
        integration_results = await self.run_integration_validation()
        network_results = await self.run_network_validation()
        performance_results = await self.run_performance_validation()
        accuracy_results = await self.run_accuracy_validation()
        
        # Combine results
        all_tests = (
            unit_results.test_results +
            integration_results.test_results +
            network_results.test_results +
            performance_results.test_results +
            accuracy_results.test_results
        )
        
        # Calculate overall metrics
        total_tests = len(all_tests)
        passed_tests = sum(1 for test in all_tests if test.status == TestStatus.PASSED)
        failed_tests = sum(1 for test in all_tests if test.status == TestStatus.FAILED)
        skipped_tests = sum(1 for test in all_tests if test.status == TestStatus.SKIPPED)
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        
        # Performance metrics
        execution_times = [test.execution_time for test in all_tests if test.execution_time > 0]
        total_execution_time = sum(execution_times)
        average_test_time = statistics.mean(execution_times) if execution_times else 0.0
        fastest_test_time = min(execution_times) if execution_times else 0.0
        slowest_test_time = max(execution_times) if execution_times else 0.0
        
        # Accuracy metrics
        accuracy_scores = [test.accuracy_score for test in all_tests if test.accuracy_score > 0]
        overall_accuracy = statistics.mean(accuracy_scores) if accuracy_scores else 0.0
        
        # Engine-specific accuracies
        engine_accuracies = {}
        for engine in ["deductive", "inductive", "abductive", "analogical", "causal", "probabilistic", "counterfactual"]:
            engine_tests = [test for test in all_tests if engine in test.name.lower()]
            engine_scores = [test.accuracy_score for test in engine_tests if test.accuracy_score > 0]
            engine_accuracies[engine] = statistics.mean(engine_scores) if engine_scores else 0.0
        
        # Generate insights
        validation_insights = await self._generate_validation_insights(all_tests)
        performance_insights = await self._generate_performance_insights(all_tests)
        recommendations = await self._generate_recommendations(all_tests)
        
        # Create comprehensive result
        result = ValidationResult(
            id=str(uuid4()),
            validation_level=ValidationLevel.COMPREHENSIVE,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            success_rate=success_rate,
            total_execution_time=total_execution_time,
            average_test_time=average_test_time,
            fastest_test_time=fastest_test_time,
            slowest_test_time=slowest_test_time,
            overall_accuracy=overall_accuracy,
            engine_accuracies=engine_accuracies,
            test_results=all_tests,
            failed_test_details=[test.error_message for test in all_tests if test.status == TestStatus.FAILED],
            validation_insights=validation_insights,
            performance_insights=performance_insights,
            recommendations=recommendations
        )
        
        self.validation_results.append(result)
        
        logger.info(
            "Comprehensive validation complete",
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            success_rate=success_rate,
            overall_accuracy=overall_accuracy
        )
        
        return result
    
    async def run_unit_validation(self) -> ValidationResult:
        """Run unit validation tests for individual engines"""
        
        logger.info("Starting unit validation")
        
        tests = []
        
        # Test each reasoning engine
        for engine_name, queries in self.test_queries.items():
            for i, query in enumerate(queries):
                test = ValidationTest(
                    id=f"unit_{engine_name}_{i+1}",
                    name=f"Unit Test - {engine_name.title()} Engine",
                    description=f"Test {engine_name} reasoning with: {query[:50]}...",
                    category=TestCategory.FUNCTIONAL,
                    level=ValidationLevel.UNIT,
                    test_function=f"test_{engine_name}_engine",
                    test_parameters={"query": query, "engine": engine_name}
                )
                tests.append(test)
        
        # Execute tests
        executed_tests = []
        for test in tests:
            executed_test = await self._execute_unit_test(test)
            executed_tests.append(executed_test)
        
        # Calculate metrics
        return await self._calculate_validation_metrics(executed_tests, ValidationLevel.UNIT)
    
    async def run_integration_validation(self) -> ValidationResult:
        """Run integration validation tests"""
        
        logger.info("Starting integration validation")
        
        tests = []
        
        # Test multi-modal reasoning integration
        integration_queries = [
            "What is the most promising approach to develop sustainable energy solutions?",
            "How can we improve urban transportation systems?",
            "What are the key factors in successful team management?"
        ]
        
        for i, query in enumerate(integration_queries):
            test = ValidationTest(
                id=f"integration_{i+1}",
                name=f"Integration Test - Multi-Modal Reasoning",
                description=f"Test multi-modal integration with: {query[:50]}...",
                category=TestCategory.INTEGRATION,
                level=ValidationLevel.INTEGRATION,
                test_function="test_multi_modal_integration",
                test_parameters={"query": query}
            )
            tests.append(test)
        
        # Execute tests
        executed_tests = []
        for test in tests:
            executed_test = await self._execute_integration_test(test)
            executed_tests.append(executed_test)
        
        return await self._calculate_validation_metrics(executed_tests, ValidationLevel.INTEGRATION)
    
    async def run_network_validation(self) -> ValidationResult:
        """Run network validation tests"""
        
        logger.info("Starting network validation")
        
        tests = []
        
        # Test network validation system
        network_queries = [
            "What are the top 5 most promising experiments to reduce inflammation without side effects?",
            "What innovative approaches could improve online education effectiveness?",
            "How can we design more efficient renewable energy storage systems?"
        ]
        
        for i, query in enumerate(network_queries):
            test = ValidationTest(
                id=f"network_{i+1}",
                name=f"Network Validation Test",
                description=f"Test network validation with: {query[:50]}...",
                category=TestCategory.FUNCTIONAL,
                level=ValidationLevel.NETWORK,
                test_function="test_network_validation",
                test_parameters={"query": query}
            )
            tests.append(test)
        
        # Execute tests
        executed_tests = []
        for test in tests:
            executed_test = await self._execute_network_test(test)
            executed_tests.append(executed_test)
        
        return await self._calculate_validation_metrics(executed_tests, ValidationLevel.NETWORK)
    
    async def run_performance_validation(self) -> ValidationResult:
        """Run performance validation tests"""
        
        logger.info("Starting performance validation")
        
        tests = []
        
        # Create performance tests for each engine
        for engine_name in self.test_queries.keys():
            test = ValidationTest(
                id=f"performance_{engine_name}",
                name=f"Performance Test - {engine_name.title()} Engine",
                description=f"Test {engine_name} engine performance",
                category=TestCategory.PERFORMANCE,
                level=ValidationLevel.PERFORMANCE,
                test_function=f"test_{engine_name}_performance",
                test_parameters={"engine": engine_name}
            )
            tests.append(test)
        
        # Execute tests
        executed_tests = []
        for test in tests:
            executed_test = await self._execute_performance_test(test)
            executed_tests.append(executed_test)
        
        return await self._calculate_validation_metrics(executed_tests, ValidationLevel.PERFORMANCE)
    
    async def run_accuracy_validation(self) -> ValidationResult:
        """Run accuracy validation tests"""
        
        logger.info("Starting accuracy validation")
        
        tests = []
        
        # Create accuracy tests
        accuracy_queries = [
            ("What is 2 + 2?", "4", "deductive"),
            ("What causes rain?", "water_cycle", "causal"),
            ("What is the probability of a fair coin landing heads?", "0.5", "probabilistic")
        ]
        
        for i, (query, expected, engine) in enumerate(accuracy_queries):
            test = ValidationTest(
                id=f"accuracy_{i+1}",
                name=f"Accuracy Test - {engine.title()}",
                description=f"Test accuracy with: {query}",
                category=TestCategory.ACCURACY,
                level=ValidationLevel.ACCURACY,
                test_function=f"test_accuracy",
                test_parameters={"query": query, "expected": expected, "engine": engine},
                expected_result=expected
            )
            tests.append(test)
        
        # Execute tests
        executed_tests = []
        for test in tests:
            executed_test = await self._execute_accuracy_test(test)
            executed_tests.append(executed_test)
        
        return await self._calculate_validation_metrics(executed_tests, ValidationLevel.ACCURACY)
    
    async def _execute_unit_test(self, test: ValidationTest) -> ValidationTest:
        """Execute a unit test"""
        
        test.status = TestStatus.RUNNING
        test.start_time = datetime.now(timezone.utc)
        
        try:
            # Get engine and query
            engine_name = test.test_parameters["engine"]
            query = test.test_parameters["query"]
            
            # Execute based on engine type
            if engine_name == "deductive":
                result = await self.deductive_engine.deduce_conclusion(
                    premises=[query],
                    conclusion_type="logical"
                )
                test.actual_result = result.conclusion
                test.accuracy_score = result.logical_validity
                
            elif engine_name == "inductive":
                result = await self.inductive_engine.induce_pattern(
                    observations=[query]
                )
                test.actual_result = result.conclusion_statement
                test.accuracy_score = result.probability
                
            elif engine_name == "abductive":
                result = await self.abductive_engine.generate_best_explanation(
                    observations=[query]
                )
                test.actual_result = result.best_explanation.hypothesis
                test.accuracy_score = result.best_explanation.plausibility
                
            elif engine_name == "analogical":
                result = await self.analogical_engine.discover_cross_domain_insights(
                    source_domain="general",
                    target_domain="general",
                    focus_area=query
                )
                test.actual_result = result[0].description if result else "No insights"
                test.accuracy_score = result[0].confidence_score if result else 0.0
                
            elif engine_name == "causal":
                result = await self.causal_engine.analyze_causal_relationships(
                    observations=[query]
                )
                test.actual_result = result.causal_summary
                test.accuracy_score = result.overall_confidence
                
            elif engine_name == "probabilistic":
                result = await self.probabilistic_engine.probabilistic_inference(
                    evidence=[query],
                    hypothesis=query
                )
                test.actual_result = result.inference_result
                test.accuracy_score = result.overall_confidence
                
            elif engine_name == "counterfactual":
                result = await self.counterfactual_engine.evaluate_counterfactual(
                    query=query
                )
                test.actual_result = result.comparison.preferable_scenario
                test.accuracy_score = result.overall_probability
            
            test.status = TestStatus.PASSED
            test.success = True
            
        except Exception as e:
            test.status = TestStatus.FAILED
            test.error_message = str(e)
            test.success = False
            logger.error(f"Unit test failed: {test.id}", error=str(e))
        
        test.end_time = datetime.now(timezone.utc)
        test.execution_time = (test.end_time - test.start_time).total_seconds()
        
        return test
    
    async def _execute_integration_test(self, test: ValidationTest) -> ValidationTest:
        """Execute an integration test"""
        
        test.status = TestStatus.RUNNING
        test.start_time = datetime.now(timezone.utc)
        
        try:
            query = test.test_parameters["query"]
            
            # Test multi-modal reasoning
            result = await self.multi_modal_engine.process_query(query)
            
            test.actual_result = result.integrated_conclusion
            test.accuracy_score = result.overall_confidence
            test.success = result.overall_confidence > self.accuracy_threshold
            test.status = TestStatus.PASSED if test.success else TestStatus.FAILED
            
        except Exception as e:
            test.status = TestStatus.FAILED
            test.error_message = str(e)
            test.success = False
            logger.error(f"Integration test failed: {test.id}", error=str(e))
        
        test.end_time = datetime.now(timezone.utc)
        test.execution_time = (test.end_time - test.start_time).total_seconds()
        
        return test
    
    async def _execute_network_test(self, test: ValidationTest) -> ValidationTest:
        """Execute a network validation test"""
        
        test.status = TestStatus.RUNNING
        test.start_time = datetime.now(timezone.utc)
        
        try:
            query = test.test_parameters["query"]
            
            # Test network validation
            result = await self.network_validator.validate_candidates(
                query=query,
                domain="general"
            )
            
            test.actual_result = {
                "approved_candidates": len(result.approved_candidates),
                "consensus_rate": result.consensus_rate,
                "validation_efficiency": result.validation_efficiency
            }
            
            # Check against benchmarks
            targets = self.benchmark_data["network_validation_targets"]
            test.accuracy_score = (
                (result.consensus_rate >= targets["consensus_rate"]) +
                (result.validation_efficiency >= targets["validation_efficiency"]) +
                (result.result_quality >= targets["result_quality"])
            ) / 3.0
            
            test.success = test.accuracy_score >= 0.67  # 2/3 targets met
            test.status = TestStatus.PASSED if test.success else TestStatus.FAILED
            
        except Exception as e:
            test.status = TestStatus.FAILED
            test.error_message = str(e)
            test.success = False
            logger.error(f"Network test failed: {test.id}", error=str(e))
        
        test.end_time = datetime.now(timezone.utc)
        test.execution_time = (test.end_time - test.start_time).total_seconds()
        
        return test
    
    async def _execute_performance_test(self, test: ValidationTest) -> ValidationTest:
        """Execute a performance test"""
        
        test.status = TestStatus.RUNNING
        test.start_time = datetime.now(timezone.utc)
        
        try:
            engine_name = test.test_parameters["engine"]
            queries = self.test_queries[engine_name]
            
            # Test performance with multiple queries
            execution_times = []
            
            for query in queries:
                start_time = time.time()
                
                # Execute query based on engine
                if engine_name == "deductive":
                    await self.deductive_engine.deduce_conclusion([query])
                elif engine_name == "inductive":
                    await self.inductive_engine.induce_pattern([query])
                elif engine_name == "abductive":
                    await self.abductive_engine.generate_best_explanation([query])
                elif engine_name == "analogical":
                    await self.analogical_engine.discover_cross_domain_insights("general", "general", query)
                elif engine_name == "causal":
                    await self.causal_engine.analyze_causal_relationships([query])
                elif engine_name == "probabilistic":
                    await self.probabilistic_engine.probabilistic_inference([query], query)
                elif engine_name == "counterfactual":
                    await self.counterfactual_engine.evaluate_counterfactual(query)
                
                execution_times.append(time.time() - start_time)
            
            # Calculate performance metrics
            avg_time = statistics.mean(execution_times)
            target_time = self.benchmark_data["response_time_targets"][engine_name]
            
            test.actual_result = {
                "average_time": avg_time,
                "target_time": target_time,
                "queries_tested": len(queries)
            }
            
            test.performance_score = min(1.0, target_time / avg_time)  # Higher is better
            test.success = avg_time <= target_time
            test.status = TestStatus.PASSED if test.success else TestStatus.FAILED
            
        except Exception as e:
            test.status = TestStatus.FAILED
            test.error_message = str(e)
            test.success = False
            logger.error(f"Performance test failed: {test.id}", error=str(e))
        
        test.end_time = datetime.now(timezone.utc)
        test.execution_time = (test.end_time - test.start_time).total_seconds()
        
        return test
    
    async def _execute_accuracy_test(self, test: ValidationTest) -> ValidationTest:
        """Execute an accuracy test"""
        
        test.status = TestStatus.RUNNING
        test.start_time = datetime.now(timezone.utc)
        
        try:
            query = test.test_parameters["query"]
            expected = test.test_parameters["expected"]
            engine = test.test_parameters["engine"]
            
            # Execute query and compare result
            if engine == "deductive":
                result = await self.deductive_engine.deduce_conclusion([query])
                actual = result.conclusion
            elif engine == "causal":
                result = await self.causal_engine.analyze_causal_relationships([query])
                actual = result.causal_summary
            elif engine == "probabilistic":
                result = await self.probabilistic_engine.probabilistic_inference([query], query)
                actual = result.inference_result
            else:
                actual = "Unknown engine"
            
            test.actual_result = actual
            
            # Simple accuracy check (would be more sophisticated in production)
            if expected in str(actual).lower() or str(expected).lower() in str(actual).lower():
                test.accuracy_score = 1.0
                test.success = True
            else:
                test.accuracy_score = 0.0
                test.success = False
            
            test.status = TestStatus.PASSED if test.success else TestStatus.FAILED
            
        except Exception as e:
            test.status = TestStatus.FAILED
            test.error_message = str(e)
            test.success = False
            logger.error(f"Accuracy test failed: {test.id}", error=str(e))
        
        test.end_time = datetime.now(timezone.utc)
        test.execution_time = (test.end_time - test.start_time).total_seconds()
        
        return test
    
    async def _calculate_validation_metrics(self, tests: List[ValidationTest], level: ValidationLevel) -> ValidationResult:
        """Calculate validation metrics for a set of tests"""
        
        total_tests = len(tests)
        passed_tests = sum(1 for test in tests if test.status == TestStatus.PASSED)
        failed_tests = sum(1 for test in tests if test.status == TestStatus.FAILED)
        skipped_tests = sum(1 for test in tests if test.status == TestStatus.SKIPPED)
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        
        # Performance metrics
        execution_times = [test.execution_time for test in tests if test.execution_time > 0]
        total_execution_time = sum(execution_times)
        average_test_time = statistics.mean(execution_times) if execution_times else 0.0
        fastest_test_time = min(execution_times) if execution_times else 0.0
        slowest_test_time = max(execution_times) if execution_times else 0.0
        
        # Accuracy metrics
        accuracy_scores = [test.accuracy_score for test in tests if test.accuracy_score > 0]
        overall_accuracy = statistics.mean(accuracy_scores) if accuracy_scores else 0.0
        
        return ValidationResult(
            id=str(uuid4()),
            validation_level=level,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            success_rate=success_rate,
            total_execution_time=total_execution_time,
            average_test_time=average_test_time,
            fastest_test_time=fastest_test_time,
            slowest_test_time=slowest_test_time,
            overall_accuracy=overall_accuracy,
            engine_accuracies={},
            test_results=tests,
            failed_test_details=[test.error_message for test in tests if test.status == TestStatus.FAILED],
            validation_insights=[],
            performance_insights=[],
            recommendations=[]
        )
    
    async def _generate_validation_insights(self, tests: List[ValidationTest]) -> List[str]:
        """Generate validation insights from test results"""
        
        insights = []
        
        # Overall performance
        total_tests = len(tests)
        passed_tests = sum(1 for test in tests if test.status == TestStatus.PASSED)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        
        insights.append(f"Overall validation success rate: {success_rate:.1%}")
        
        # Engine performance
        engine_results = defaultdict(list)
        for test in tests:
            for engine in ["deductive", "inductive", "abductive", "analogical", "causal", "probabilistic", "counterfactual"]:
                if engine in test.name.lower():
                    engine_results[engine].append(test.success)
        
        for engine, results in engine_results.items():
            if results:
                engine_success_rate = sum(results) / len(results)
                insights.append(f"{engine.title()} engine success rate: {engine_success_rate:.1%}")
        
        # Test categories
        category_results = defaultdict(list)
        for test in tests:
            category_results[test.category].append(test.success)
        
        for category, results in category_results.items():
            if results:
                category_success_rate = sum(results) / len(results)
                insights.append(f"{category.value.title()} tests success rate: {category_success_rate:.1%}")
        
        return insights
    
    async def _generate_performance_insights(self, tests: List[ValidationTest]) -> List[str]:
        """Generate performance insights from test results"""
        
        insights = []
        
        # Execution time analysis
        execution_times = [test.execution_time for test in tests if test.execution_time > 0]
        if execution_times:
            avg_time = statistics.mean(execution_times)
            insights.append(f"Average test execution time: {avg_time:.2f} seconds")
            
            # Slowest tests
            slowest_tests = sorted(tests, key=lambda x: x.execution_time, reverse=True)[:3]
            insights.append(f"Slowest tests: {', '.join(test.name for test in slowest_tests)}")
        
        # Performance scores
        performance_scores = [test.performance_score for test in tests if test.performance_score > 0]
        if performance_scores:
            avg_performance = statistics.mean(performance_scores)
            insights.append(f"Average performance score: {avg_performance:.2f}")
        
        return insights
    
    async def _generate_recommendations(self, tests: List[ValidationTest]) -> List[str]:
        """Generate recommendations based on test results"""
        
        recommendations = []
        
        # Failed tests
        failed_tests = [test for test in tests if test.status == TestStatus.FAILED]
        if failed_tests:
            recommendations.append(f"Address {len(failed_tests)} failed tests to improve system reliability")
            
            # Common failure patterns
            error_patterns = defaultdict(int)
            for test in failed_tests:
                if test.error_message:
                    # Simple pattern matching
                    if "timeout" in test.error_message.lower():
                        error_patterns["timeout"] += 1
                    elif "error" in test.error_message.lower():
                        error_patterns["error"] += 1
            
            for pattern, count in error_patterns.items():
                recommendations.append(f"Investigate {count} tests with {pattern} issues")
        
        # Performance recommendations
        slow_tests = [test for test in tests if test.execution_time > self.performance_threshold]
        if slow_tests:
            recommendations.append(f"Optimize {len(slow_tests)} slow-performing tests")
        
        # Accuracy recommendations
        low_accuracy_tests = [test for test in tests if test.accuracy_score < self.accuracy_threshold]
        if low_accuracy_tests:
            recommendations.append(f"Improve accuracy for {len(low_accuracy_tests)} underperforming tests")
        
        return recommendations
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results"""
        
        if not self.validation_results:
            return {"message": "No validation results available"}
        
        # Latest comprehensive result
        latest_result = self.validation_results[-1]
        
        return {
            "latest_validation": {
                "timestamp": latest_result.validation_timestamp.isoformat(),
                "level": latest_result.validation_level.value,
                "total_tests": latest_result.total_tests,
                "success_rate": latest_result.success_rate,
                "overall_accuracy": latest_result.overall_accuracy,
                "total_execution_time": latest_result.total_execution_time
            },
            "engine_performance": latest_result.engine_accuracies,
            "key_insights": latest_result.validation_insights,
            "recommendations": latest_result.recommendations,
            "historical_results": len(self.validation_results),
            "system_health": "Good" if latest_result.success_rate >= 0.8 else "Needs Attention"
        }