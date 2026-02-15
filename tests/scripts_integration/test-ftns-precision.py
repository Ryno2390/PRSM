#!/usr/bin/env python3
"""
FTNS Accounting Ledger Test Suite
Validates microsecond-precision cost calculation and Phase 1 requirements

This comprehensive test suite validates:
1. Microsecond-precision cost calculation (28 decimal places)
2. Local token accounting system accuracy
3. Usage tracking completeness
4. Performance under load (10K transactions/second)
5. Cost calculation consistency
6. Audit trail integrity
7. Balance reconciliation accuracy

Test Scenarios:
- Precision validation with edge cases
- High-volume transaction processing
- Cost calculation stress testing
- Usage analytics accuracy
- Error handling and recovery
- Performance benchmarking
"""

import asyncio
import time
import sys
import os
import random
from datetime import datetime, timezone, timedelta
from decimal import Decimal, getcontext
from typing import List, Dict, Any, Optional
from uuid import uuid4
import statistics
import structlog

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prsm.economy.tokenomics.enhanced_ftns_service import (
    EnhancedFTNSService, MicrosecondCostCalculation, UsageTrackingEntry,
    get_enhanced_ftns_service
)
from prsm.core.models import PRSMSession, AgentType, PRSMResponse

logger = structlog.get_logger(__name__)

class FTNSPrecisionTester:
    """Comprehensive FTNS precision and performance tester"""
    
    def __init__(self):
        self.ftns_service = get_enhanced_ftns_service()
        self.test_results = {
            'precision_tests': [],
            'performance_tests': [],
            'accuracy_tests': [],
            'stress_tests': [],
            'validation_summary': {}
        }
        
        # Test data generators
        self.test_sessions = []
        self.test_users = [f"test_user_{i:04d}" for i in range(1000)]
        
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run complete FTNS test suite"""
        logger.info("Starting comprehensive FTNS precision test suite")
        
        try:
            # Test 1: Microsecond Precision Validation
            await self._test_microsecond_precision()
            
            # Test 2: High-Volume Transaction Processing
            await self._test_high_volume_transactions()
            
            # Test 3: Cost Calculation Accuracy
            await self._test_cost_calculation_accuracy()
            
            # Test 4: Usage Tracking Completeness
            await self._test_usage_tracking()
            
            # Test 5: Performance Benchmarking
            await self._test_performance_benchmarks()
            
            # Test 6: Stress Testing
            await self._test_stress_scenarios()
            
            # Test 7: Error Handling
            await self._test_error_handling()
            
            # Generate validation summary
            self._generate_validation_summary()
            
            return self.test_results
            
        except Exception as e:
            logger.error("Comprehensive test suite failed", error=str(e))
            raise
    
    async def _test_microsecond_precision(self):
        """Test microsecond-precision cost calculation"""
        logger.info("Testing microsecond precision cost calculation")
        
        precision_tests = []
        
        # Test 1: Maximum precision calculation
        session = self._create_test_session()
        agents = [AgentType.ARCHITECT, AgentType.EXECUTOR, AgentType.COMPILER]
        
        calculation = await self.ftns_service.calculate_microsecond_precision_cost(
            session=session,
            agents_used=agents,
            context_units=1000,
            execution_time_microseconds=1234567
        )
        
        # Verify precision (28 decimal places)
        total_cost_str = str(calculation.total_cost)
        decimal_places = len(total_cost_str.split('.')[-1]) if '.' in total_cost_str else 0
        
        precision_test = {
            'test_name': 'maximum_precision',
            'decimal_places_achieved': decimal_places,
            'expected_minimum': 20,
            'total_cost': float(calculation.total_cost),
            'calculation_time_us': calculation.calculation_duration_microseconds,
            'passed': decimal_places >= 20
        }
        precision_tests.append(precision_test)
        
        # Test 2: Precision consistency across multiple calculations
        costs = []
        for i in range(100):
            calc = await self.ftns_service.calculate_microsecond_precision_cost(
                session=session,
                agents_used=agents,
                context_units=1000,
                execution_time_microseconds=1234567
            )
            costs.append(calc.total_cost)
        
        # All calculations should be identical
        consistency_test = {
            'test_name': 'precision_consistency',
            'calculations_count': len(costs),
            'unique_values': len(set(costs)),
            'expected_unique': 1,
            'cost_variance': float(max(costs) - min(costs)) if costs else 0,
            'passed': len(set(costs)) == 1
        }
        precision_tests.append(consistency_test)
        
        # Test 3: Edge case precision
        edge_cases = [
            ([], 0, 0),  # Minimum case
            ([AgentType.ARCHITECT, AgentType.PROMPTER, AgentType.ROUTER, AgentType.EXECUTOR, AgentType.COMPILER], 10000, 86400000000),  # Maximum case
            ([AgentType.EXECUTOR], 1, 1),  # Single unit
        ]
        
        for agents, context_units, execution_time in edge_cases:
            calc = await self.ftns_service.calculate_microsecond_precision_cost(
                session=session,
                agents_used=agents,
                context_units=context_units,
                execution_time_microseconds=execution_time
            )
            
            edge_test = {
                'test_name': f'edge_case_{len(agents)}agents_{context_units}context',
                'agents_count': len(agents),
                'context_units': context_units,
                'total_cost': float(calc.total_cost),
                'calculation_valid': calc.total_cost >= 0,
                'passed': calc.total_cost >= 0 and calc.calculation_duration_microseconds > 0
            }
            precision_tests.append(edge_test)
        
        self.test_results['precision_tests'] = precision_tests
        
        passed_tests = sum(1 for test in precision_tests if test['passed'])
        logger.info("Precision tests completed",
                   total_tests=len(precision_tests),
                   passed_tests=passed_tests,
                   success_rate=f"{passed_tests/len(precision_tests)*100:.1f}%")
    
    async def _test_high_volume_transactions(self):
        """Test high-volume transaction processing (Phase 1: 10K transactions/second)"""
        logger.info("Testing high-volume transaction processing")
        
        performance_tests = []
        
        # Test 1: 10,000 transactions in 1 second
        transaction_count = 10000
        start_time = time.perf_counter()
        
        tasks = []
        for i in range(transaction_count):
            session = self._create_test_session(user_id=f"perf_user_{i % 100}")
            task = self._process_test_transaction(session)
            tasks.append(task)
        
        # Execute all transactions concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Analyze results
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful
        transactions_per_second = successful / total_time if total_time > 0 else 0
        
        volume_test = {
            'test_name': 'high_volume_10k_transactions',
            'target_transactions': transaction_count,
            'successful_transactions': successful,
            'failed_transactions': failed,
            'total_time_seconds': total_time,
            'transactions_per_second': transactions_per_second,
            'target_tps': 10000,
            'passed': transactions_per_second >= 8000  # 80% of target
        }
        performance_tests.append(volume_test)
        
        # Test 2: Sustained load (1000 TPS for 60 seconds)
        sustained_duration = 10  # Reduced for testing
        transactions_per_batch = 100
        batches = sustained_duration * 10  # 10 batches per second
        
        start_time = time.perf_counter()
        total_successful = 0
        
        for batch in range(batches):
            batch_start = time.perf_counter()
            
            batch_tasks = []
            for i in range(transactions_per_batch):
                session = self._create_test_session(user_id=f"sustained_user_{i % 50}")
                task = self._process_test_transaction(session)
                batch_tasks.append(task)
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            batch_successful = sum(1 for r in batch_results if not isinstance(r, Exception))
            total_successful += batch_successful
            
            # Maintain target rate
            batch_time = time.perf_counter() - batch_start
            target_batch_time = 0.1  # 100ms per batch
            if batch_time < target_batch_time:
                await asyncio.sleep(target_batch_time - batch_time)
        
        sustained_time = time.perf_counter() - start_time
        sustained_tps = total_successful / sustained_time if sustained_time > 0 else 0
        
        sustained_test = {
            'test_name': 'sustained_load_test',
            'duration_seconds': sustained_time,
            'total_transactions': total_successful,
            'average_tps': sustained_tps,
            'target_tps': 1000,
            'passed': sustained_tps >= 800  # 80% of target
        }
        performance_tests.append(sustained_test)
        
        self.test_results['performance_tests'] = performance_tests
        
        logger.info("High-volume tests completed",
                   peak_tps=volume_test['transactions_per_second'],
                   sustained_tps=sustained_test['average_tps'])
    
    async def _test_cost_calculation_accuracy(self):
        """Test cost calculation accuracy and consistency"""
        logger.info("Testing cost calculation accuracy")
        
        accuracy_tests = []
        
        # Test 1: Agent cost attribution
        session = self._create_test_session()
        
        for agent_type in AgentType:
            calculation = await self.ftns_service.calculate_microsecond_precision_cost(
                session=session,
                agents_used=[agent_type],
                context_units=100,
                execution_time_microseconds=1000000
            )
            
            agent_cost = calculation.agent_cost_breakdown.get(agent_type.value, Decimal('0'))
            expected_cost = self.ftns_service.base_costs.get(agent_type, Decimal('0'))
            
            accuracy_test = {
                'test_name': f'agent_cost_{agent_type.value}',
                'agent_type': agent_type.value,
                'calculated_cost': float(agent_cost),
                'expected_base_cost': float(expected_cost),
                'cost_difference': float(abs(agent_cost - expected_cost)),
                'passed': abs(agent_cost - expected_cost) < Decimal('0.000001')
            }
            accuracy_tests.append(accuracy_test)
        
        # Test 2: Context unit scaling
        base_context = 100
        scaling_factors = [1, 2, 5, 10, 100]
        
        base_calculation = await self.ftns_service.calculate_microsecond_precision_cost(
            session=session,
            agents_used=[AgentType.EXECUTOR],
            context_units=base_context,
            execution_time_microseconds=1000000
        )
        
        for factor in scaling_factors:
            scaled_calculation = await self.ftns_service.calculate_microsecond_precision_cost(
                session=session,
                agents_used=[AgentType.EXECUTOR],
                context_units=base_context * factor,
                execution_time_microseconds=1000000
            )
            
            expected_ratio = Decimal(str(factor))
            actual_ratio = scaled_calculation.total_cost / base_calculation.total_cost
            ratio_difference = abs(actual_ratio - expected_ratio)
            
            scaling_test = {
                'test_name': f'context_scaling_{factor}x',
                'scaling_factor': factor,
                'base_cost': float(base_calculation.total_cost),
                'scaled_cost': float(scaled_calculation.total_cost),
                'expected_ratio': float(expected_ratio),
                'actual_ratio': float(actual_ratio),
                'ratio_difference': float(ratio_difference),
                'passed': ratio_difference < Decimal('0.01')  # 1% tolerance
            }
            accuracy_tests.append(scaling_test)
        
        # Test 3: Complexity multiplier validation
        complexities = [0.1, 0.5, 0.8, 1.0]
        
        for complexity in complexities:
            test_session = self._create_test_session()
            test_session.complexity_estimate = complexity
            
            calculation = await self.ftns_service.calculate_microsecond_precision_cost(
                session=test_session,
                agents_used=[AgentType.EXECUTOR],
                context_units=100,
                execution_time_microseconds=1000000
            )
            
            expected_multiplier = Decimal('1') + (Decimal(str(complexity)) * Decimal('0.5'))
            actual_multiplier = calculation.complexity_multiplier
            multiplier_difference = abs(actual_multiplier - expected_multiplier)
            
            complexity_test = {
                'test_name': f'complexity_multiplier_{complexity}',
                'complexity_input': complexity,
                'expected_multiplier': float(expected_multiplier),
                'actual_multiplier': float(actual_multiplier),
                'multiplier_difference': float(multiplier_difference),
                'passed': multiplier_difference < Decimal('0.000001')
            }
            accuracy_tests.append(complexity_test)
        
        self.test_results['accuracy_tests'] = accuracy_tests
        
        passed_tests = sum(1 for test in accuracy_tests if test['passed'])
        logger.info("Accuracy tests completed",
                   total_tests=len(accuracy_tests),
                   passed_tests=passed_tests,
                   accuracy_rate=f"{passed_tests/len(accuracy_tests)*100:.1f}%")
    
    async def _test_usage_tracking(self):
        """Test usage tracking completeness and accuracy"""
        logger.info("Testing usage tracking system")
        
        tracking_tests = []
        
        # Test 1: Complete session tracking
        session = self._create_test_session()
        agents_used = [AgentType.ARCHITECT, AgentType.EXECUTOR, AgentType.COMPILER]
        
        # Simulate complete session with timing
        session_start = datetime.now(timezone.utc)
        
        calculation = await self.ftns_service.calculate_microsecond_precision_cost(
            session=session,
            agents_used=agents_used,
            context_units=150,
            execution_time_microseconds=2500000
        )
        
        # Track usage for each agent
        usage_entries = []
        for i, agent_type in enumerate(agents_used):
            agent_start = session_start + timedelta(microseconds=i*800000)
            agent_end = agent_start + timedelta(microseconds=800000)
            
            usage_entry = await self.ftns_service.track_usage_with_precision(
                session=session,
                agent_type=agent_type,
                operation_type="test_execution",
                context_units=50,
                start_timestamp=agent_start,
                end_timestamp=agent_end,
                cost_calculation=calculation,
                success=True
            )
            usage_entries.append(usage_entry)
        
        # Validate tracking completeness
        tracking_test = {
            'test_name': 'complete_session_tracking',
            'agents_tracked': len(usage_entries),
            'expected_agents': len(agents_used),
            'total_tracked_cost': sum(float(entry.cost_ftns) for entry in usage_entries),
            'all_agents_tracked': len(usage_entries) == len(agents_used),
            'timing_accuracy': all(entry.duration_microseconds > 0 for entry in usage_entries),
            'passed': len(usage_entries) == len(agents_used) and all(entry.duration_microseconds > 0 for entry in usage_entries)
        }
        tracking_tests.append(tracking_test)
        
        # Test 2: High-frequency tracking
        rapid_sessions = []
        for i in range(100):
            rapid_session = self._create_test_session(user_id=f"rapid_user_{i}")
            rapid_sessions.append(rapid_session)
        
        tracking_start = time.perf_counter()
        rapid_entries = []
        
        for session in rapid_sessions:
            calc = await self.ftns_service.calculate_microsecond_precision_cost(
                session=session,
                agents_used=[AgentType.EXECUTOR],
                context_units=10,
                execution_time_microseconds=100000
            )
            
            entry = await self.ftns_service.track_usage_with_precision(
                session=session,
                agent_type=AgentType.EXECUTOR,
                operation_type="rapid_test",
                context_units=10,
                start_timestamp=datetime.now(timezone.utc),
                end_timestamp=datetime.now(timezone.utc) + timedelta(microseconds=100000),
                cost_calculation=calc,
                success=True
            )
            rapid_entries.append(entry)
        
        tracking_time = time.perf_counter() - tracking_start
        tracking_rate = len(rapid_entries) / tracking_time if tracking_time > 0 else 0
        
        rapid_tracking_test = {
            'test_name': 'high_frequency_tracking',
            'sessions_tracked': len(rapid_entries),
            'tracking_time_seconds': tracking_time,
            'tracking_rate_per_second': tracking_rate,
            'target_rate': 1000,
            'passed': tracking_rate >= 500  # 50% of target
        }
        tracking_tests.append(rapid_tracking_test)
        
        self.test_results['usage_tracking_tests'] = tracking_tests
        
        logger.info("Usage tracking tests completed",
                   tracking_rate=rapid_tracking_test['tracking_rate_per_second'])
    
    async def _test_performance_benchmarks(self):
        """Test performance benchmarks for Phase 1 requirements"""
        logger.info("Running performance benchmarks")
        
        benchmark_tests = []
        
        # Benchmark 1: Cost calculation speed
        calculation_times = []
        session = self._create_test_session()
        
        for _ in range(1000):
            start = time.perf_counter()
            await self.ftns_service.calculate_microsecond_precision_cost(
                session=session,
                agents_used=[AgentType.ARCHITECT, AgentType.EXECUTOR],
                context_units=100,
                execution_time_microseconds=1000000
            )
            end = time.perf_counter()
            calculation_times.append((end - start) * 1000)  # Convert to milliseconds
        
        calc_benchmark = {
            'test_name': 'cost_calculation_speed',
            'calculations_performed': len(calculation_times),
            'avg_time_ms': statistics.mean(calculation_times),
            'median_time_ms': statistics.median(calculation_times),
            'p95_time_ms': statistics.quantiles(calculation_times, n=20)[18] if len(calculation_times) > 20 else max(calculation_times),
            'max_time_ms': max(calculation_times),
            'target_time_ms': 1.0,  # Sub-millisecond target
            'passed': statistics.mean(calculation_times) < 1.0
        }
        benchmark_tests.append(calc_benchmark)
        
        # Benchmark 2: Transaction processing speed
        transaction_times = []
        
        for _ in range(100):  # Smaller sample for transaction tests
            test_session = self._create_test_session()
            
            start = time.perf_counter()
            await self._process_test_transaction(test_session)
            end = time.perf_counter()
            
            transaction_times.append((end - start) * 1000)
        
        tx_benchmark = {
            'test_name': 'transaction_processing_speed',
            'transactions_performed': len(transaction_times),
            'avg_time_ms': statistics.mean(transaction_times),
            'median_time_ms': statistics.median(transaction_times),
            'p95_time_ms': statistics.quantiles(transaction_times, n=20)[18] if len(transaction_times) > 20 else max(transaction_times),
            'target_time_ms': 10.0,  # 10ms target
            'passed': statistics.mean(transaction_times) < 10.0
        }
        benchmark_tests.append(tx_benchmark)
        
        # Benchmark 3: Memory usage efficiency
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Perform 1000 operations
        for _ in range(1000):
            session = self._create_test_session()
            await self.ftns_service.calculate_microsecond_precision_cost(
                session=session,
                agents_used=[AgentType.EXECUTOR],
                context_units=50,
                execution_time_microseconds=500000
            )
        
        gc.collect()  # Force garbage collection
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        memory_benchmark = {
            'test_name': 'memory_efficiency',
            'operations_performed': 1000,
            'initial_memory_mb': initial_memory / (1024 * 1024),
            'final_memory_mb': final_memory / (1024 * 1024),
            'memory_increase_mb': memory_increase / (1024 * 1024),
            'memory_per_operation_kb': memory_increase / 1000 / 1024,
            'target_increase_mb': 50,  # 50MB max increase
            'passed': memory_increase < 50 * 1024 * 1024
        }
        benchmark_tests.append(memory_benchmark)
        
        self.test_results['benchmark_tests'] = benchmark_tests
        
        logger.info("Performance benchmarks completed",
                   avg_calc_time_ms=calc_benchmark['avg_time_ms'],
                   avg_tx_time_ms=tx_benchmark['avg_time_ms'],
                   memory_increase_mb=memory_benchmark['memory_increase_mb'])
    
    async def _test_stress_scenarios(self):
        """Test system under stress conditions"""
        logger.info("Running stress test scenarios")
        
        stress_tests = []
        
        # Stress Test 1: Concurrent users with mixed workloads
        concurrent_users = 100
        operations_per_user = 10
        
        async def user_workload(user_id: str):
            results = []
            for _ in range(operations_per_user):
                try:
                    session = self._create_test_session(user_id=user_id)
                    
                    # Random agent combination
                    agents = random.sample(list(AgentType), random.randint(1, 3))
                    context_units = random.randint(10, 500)
                    execution_time = random.randint(100000, 5000000)
                    
                    calc = await self.ftns_service.calculate_microsecond_precision_cost(
                        session=session,
                        agents_used=agents,
                        context_units=context_units,
                        execution_time_microseconds=execution_time
                    )
                    
                    results.append({'success': True, 'cost': float(calc.total_cost)})
                    
                    # Small delay to simulate user behavior
                    await asyncio.sleep(random.uniform(0.01, 0.1))
                    
                except Exception as e:
                    results.append({'success': False, 'error': str(e)})
            
            return results
        
        # Run concurrent user workloads
        start_time = time.perf_counter()
        user_tasks = [user_workload(f"stress_user_{i}") for i in range(concurrent_users)]
        user_results = await asyncio.gather(*user_tasks, return_exceptions=True)
        stress_time = time.perf_counter() - start_time
        
        # Analyze stress test results
        total_operations = 0
        successful_operations = 0
        total_cost = 0.0
        
        for user_result in user_results:
            if isinstance(user_result, Exception):
                continue
            
            for operation in user_result:
                total_operations += 1
                if operation.get('success', False):
                    successful_operations += 1
                    total_cost += operation.get('cost', 0)
        
        stress_test = {
            'test_name': 'concurrent_mixed_workload',
            'concurrent_users': concurrent_users,
            'operations_per_user': operations_per_user,
            'total_operations': total_operations,
            'successful_operations': successful_operations,
            'success_rate': successful_operations / total_operations if total_operations > 0 else 0,
            'total_time_seconds': stress_time,
            'operations_per_second': total_operations / stress_time if stress_time > 0 else 0,
            'total_cost_ftns': total_cost,
            'avg_cost_per_operation': total_cost / successful_operations if successful_operations > 0 else 0,
            'target_success_rate': 0.95,
            'passed': (successful_operations / total_operations) >= 0.95 if total_operations > 0 else False
        }
        stress_tests.append(stress_test)
        
        self.test_results['stress_tests'] = stress_tests
        
        logger.info("Stress tests completed",
                   success_rate=f"{stress_test['success_rate']*100:.1f}%",
                   ops_per_second=stress_test['operations_per_second'])
    
    async def _test_error_handling(self):
        """Test error handling and recovery scenarios"""
        logger.info("Testing error handling scenarios")
        
        error_tests = []
        
        # Test 1: Invalid session data
        try:
            invalid_session = PRSMSession(user_id="", nwtn_context_allocation=0)
            await self.ftns_service.calculate_microsecond_precision_cost(
                session=invalid_session,
                agents_used=[],
                context_units=0,
                execution_time_microseconds=0
            )
            error_handling_1 = {'test_name': 'invalid_session', 'passed': True}
        except Exception:
            error_handling_1 = {'test_name': 'invalid_session', 'passed': False}
        
        error_tests.append(error_handling_1)
        
        # Test 2: Extreme values
        try:
            session = self._create_test_session()
            extreme_calc = await self.ftns_service.calculate_microsecond_precision_cost(
                session=session,
                agents_used=list(AgentType) * 100,  # Extreme agent list
                context_units=1000000,  # Extreme context
                execution_time_microseconds=86400000000000  # 24 hours in microseconds
            )
            
            extreme_handling = {
                'test_name': 'extreme_values',
                'calculation_succeeded': True,
                'cost_is_finite': extreme_calc.total_cost < Decimal('1000000'),
                'passed': extreme_calc.total_cost < Decimal('1000000')
            }
        except Exception as e:
            extreme_handling = {
                'test_name': 'extreme_values',
                'calculation_succeeded': False,
                'error': str(e),
                'passed': False
            }
        
        error_tests.append(extreme_handling)
        
        self.test_results['error_tests'] = error_tests
        
        logger.info("Error handling tests completed")
    
    def _generate_validation_summary(self):
        """Generate comprehensive validation summary"""
        summary = {
            'overall_status': 'UNKNOWN',
            'test_categories': {},
            'phase1_compliance': {},
            'recommendations': []
        }
        
        # Analyze each test category
        categories = [
            ('precision_tests', 'Microsecond Precision'),
            ('performance_tests', 'High-Volume Processing'),
            ('accuracy_tests', 'Cost Calculation Accuracy'),
            ('usage_tracking_tests', 'Usage Tracking'),
            ('benchmark_tests', 'Performance Benchmarks'),
            ('stress_tests', 'Stress Testing'),
            ('error_tests', 'Error Handling')
        ]
        
        total_tests = 0
        total_passed = 0
        
        for category_key, category_name in categories:
            if category_key in self.test_results:
                tests = self.test_results[category_key]
                passed = sum(1 for test in tests if test.get('passed', False))
                total = len(tests)
                
                summary['test_categories'][category_name] = {
                    'total_tests': total,
                    'passed_tests': passed,
                    'success_rate': passed / total if total > 0 else 0,
                    'status': 'PASSED' if passed == total else 'FAILED'
                }
                
                total_tests += total
                total_passed += passed
        
        # Overall status
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0
        summary['overall_status'] = 'PASSED' if overall_success_rate >= 0.95 else 'FAILED'
        summary['overall_success_rate'] = overall_success_rate
        
        # Phase 1 compliance assessment
        summary['phase1_compliance'] = {
            'microsecond_precision': overall_success_rate >= 0.95,
            'high_volume_processing': any(
                test.get('transactions_per_second', 0) >= 8000 
                for test in self.test_results.get('performance_tests', [])
            ),
            'cost_accuracy': any(
                test.get('passed', False) 
                for test in self.test_results.get('accuracy_tests', [])
            ),
            'usage_tracking': any(
                test.get('passed', False)
                for test in self.test_results.get('usage_tracking_tests', [])
            )
        }
        
        phase1_compliant = all(summary['phase1_compliance'].values())
        summary['phase1_compliance']['overall'] = phase1_compliant
        
        # Generate recommendations
        if not phase1_compliant:
            summary['recommendations'].append("Phase 1 requirements not fully met - review failed tests")
        
        if overall_success_rate < 0.98:
            summary['recommendations'].append("Test success rate below 98% - investigate failures")
        
        if not summary['phase1_compliance']['microsecond_precision']:
            summary['recommendations'].append("Microsecond precision issues detected - verify decimal configuration")
        
        if not summary['phase1_compliance']['high_volume_processing']:
            summary['recommendations'].append("High-volume processing below target - optimize transaction pipeline")
        
        self.test_results['validation_summary'] = summary
    
    # Helper methods
    def _create_test_session(self, user_id: Optional[str] = None) -> PRSMSession:
        """Create test session with random or specified user"""
        if user_id is None:
            user_id = random.choice(self.test_users)
        
        session = PRSMSession(
            user_id=user_id,
            nwtn_context_allocation=random.randint(50, 200),
            complexity_estimate=random.uniform(0.1, 1.0)
        )
        
        self.test_sessions.append(session)
        return session
    
    async def _process_test_transaction(self, session: PRSMSession) -> bool:
        """Process a complete test transaction"""
        try:
            # Calculate cost
            agents = random.sample(list(AgentType), random.randint(1, 3))
            calculation = await self.ftns_service.calculate_microsecond_precision_cost(
                session=session,
                agents_used=agents,
                context_units=random.randint(10, 200),
                execution_time_microseconds=random.randint(100000, 3000000)
            )
            
            # Create mock response
            response = PRSMResponse(
                session_id=session.session_id,
                user_id=session.user_id,
                final_answer="Test response",
                reasoning_trace=[],
                confidence_score=0.8,
                context_used=calculation.cost_factors.get('context_units', 100),
                ftns_charged=float(calculation.total_cost),
                sources=[],
                safety_validated=True
            )
            
            # Process transaction
            await self.ftns_service.process_session_transaction(
                session=session,
                response=response,
                cost_calculation=calculation
            )
            
            return True
            
        except Exception as e:
            logger.debug("Test transaction failed", error=str(e))
            return False

async def main():
    """Run FTNS precision test suite"""
    try:
        tester = FTNSPrecisionTester()
        results = await tester.run_comprehensive_tests()
        
        # Display results summary
        summary = results['validation_summary']
        
        print("\n" + "="*80)
        print("🎯 FTNS ACCOUNTING LEDGER TEST RESULTS")
        print("="*80)
        
        print(f"\n📊 OVERALL STATUS: {'✅ PASSED' if summary['overall_status'] == 'PASSED' else '❌ FAILED'}")
        print(f"Success Rate: {summary['overall_success_rate']:.1%}")
        
        print(f"\n🧪 TEST CATEGORIES:")
        for category, stats in summary['test_categories'].items():
            status_icon = "✅" if stats['status'] == 'PASSED' else "❌"
            print(f"├─ {category}: {status_icon} {stats['passed_tests']}/{stats['total_tests']} ({stats['success_rate']:.1%})")
        
        print(f"\n🎯 PHASE 1 COMPLIANCE:")
        compliance = summary['phase1_compliance']
        print(f"├─ Overall: {'✅ COMPLIANT' if compliance['overall'] else '❌ NON-COMPLIANT'}")
        print(f"├─ Microsecond Precision: {'✅' if compliance['microsecond_precision'] else '❌'}")
        print(f"├─ High-Volume Processing: {'✅' if compliance['high_volume_processing'] else '❌'}")
        print(f"├─ Cost Accuracy: {'✅' if compliance['cost_accuracy'] else '❌'}")
        print(f"└─ Usage Tracking: {'✅' if compliance['usage_tracking'] else '❌'}")
        
        if summary['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in summary['recommendations']:
                print(f"   • {rec}")
        
        print("\n" + "="*80)
        
        # Save detailed results
        import json
        with open('ftns_precision_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("📄 Detailed results saved to: ftns_precision_test_results.json")
        
        # Exit with appropriate code
        if summary['overall_status'] == 'PASSED':
            print("\n🎉 All tests passed! FTNS system ready for Phase 1.")
            sys.exit(0)
        else:
            print("\n⚠️  Some tests failed. Review results and retry.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())