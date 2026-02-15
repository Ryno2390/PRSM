#!/usr/bin/env python3
"""
Circuit Breaker Testing Suite
Comprehensive testing framework for failure handling and system resilience

This test suite validates circuit breaker functionality under various failure conditions:
1. Agent pipeline failures and recovery
2. FTNS service overload protection
3. IPFS storage disruption handling
4. Database connection failure management
5. External API rate limiting and timeouts
6. Cascading failure prevention

Key Features:
- Simulated failure scenarios for comprehensive testing
- Real-time circuit state monitoring and validation
- Graceful degradation and recovery verification
- Performance impact analysis during failures
- Concurrent failure handling validation
"""

import asyncio
import time
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from uuid import uuid4
from dataclasses import dataclass
import structlog

from prsm.core.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState, FailureType,
    get_circuit_breaker, protected_call, get_all_circuit_stats,
    CircuitBreakerOpenException,
    AGENT_CIRCUIT_CONFIG, FTNS_CIRCUIT_CONFIG, IPFS_CIRCUIT_CONFIG,
    DATABASE_CIRCUIT_CONFIG, EXTERNAL_API_CIRCUIT_CONFIG
)
from prsm.compute.nwtn.orchestrator import get_nwtn_orchestrator
from prsm.economy.tokenomics.enhanced_ftns_service import get_enhanced_ftns_service
from prsm.data.data_layer.enhanced_ipfs import get_ipfs_client

logger = structlog.get_logger(__name__)

@dataclass
class FailureScenario:
    """Defines a failure testing scenario"""
    name: str
    description: str
    failure_rate: float  # 0.0 to 1.0
    failure_delay: float  # seconds
    recovery_time: float  # seconds
    concurrent_calls: int
    expected_state: CircuitState

class CircuitBreakerTester:
    """
    Comprehensive circuit breaker testing framework
    
    Provides systematic testing of circuit breaker behavior under
    various failure conditions and load patterns.
    """
    
    def __init__(self):
        self.orchestrator = get_nwtn_orchestrator()
        self.ftns_service = get_enhanced_ftns_service()
        self.ipfs_client = get_ipfs_client()
        
        # Test tracking
        self.test_results: List[Dict[str, Any]] = []
        self.failure_scenarios: List[FailureScenario] = []
        
        # Initialize failure scenarios
        self._initialize_scenarios()
        
        logger.info("Circuit Breaker Tester initialized")
    
    def _initialize_scenarios(self):
        """Initialize comprehensive failure scenarios"""
        
        self.failure_scenarios = [
            FailureScenario(
                name="agent_timeout_cascade",
                description="Agent pipeline with cascading timeouts",
                failure_rate=0.8,
                failure_delay=5.0,
                recovery_time=30.0,
                concurrent_calls=50,
                expected_state=CircuitState.OPEN
            ),
            FailureScenario(
                name="ftns_overload",
                description="FTNS service overload with rate limiting",
                failure_rate=0.6,
                failure_delay=0.1,
                recovery_time=60.0,
                concurrent_calls=100,
                expected_state=CircuitState.OPEN
            ),
            FailureScenario(
                name="ipfs_intermittent",
                description="IPFS intermittent failures with recovery",
                failure_rate=0.4,
                failure_delay=2.0,
                recovery_time=45.0,
                concurrent_calls=25,
                expected_state=CircuitState.HALF_OPEN
            ),
            FailureScenario(
                name="database_connection_loss",
                description="Database connection failures",
                failure_rate=0.9,
                failure_delay=1.0,
                recovery_time=30.0,
                concurrent_calls=75,
                expected_state=CircuitState.OPEN
            ),
            FailureScenario(
                name="external_api_rate_limit",
                description="External API rate limiting",
                failure_rate=0.7,
                failure_delay=0.5,
                recovery_time=120.0,
                concurrent_calls=60,
                expected_state=CircuitState.OPEN
            ),
            FailureScenario(
                name="partial_failure_recovery",
                description="Partial failures that should not open circuit",
                failure_rate=0.2,
                failure_delay=1.0,
                recovery_time=10.0,
                concurrent_calls=30,
                expected_state=CircuitState.CLOSED
            )
        ]
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive circuit breaker test suite"""
        logger.info("Starting comprehensive circuit breaker test suite")
        start_time = time.perf_counter()
        
        test_results = {
            "test_suite_id": str(uuid4()),
            "start_time": datetime.now(timezone.utc),
            "scenarios_tested": len(self.failure_scenarios),
            "scenario_results": [],
            "overall_results": {},
            "recommendations": []
        }
        
        # Test each failure scenario
        for scenario in self.failure_scenarios:
            logger.info(f"Testing scenario: {scenario.name}")
            
            scenario_result = await self._test_failure_scenario(scenario)
            test_results["scenario_results"].append(scenario_result)
            
            # Short pause between scenarios
            await asyncio.sleep(2)
        
        # Test concurrent multi-circuit failures
        multi_circuit_result = await self._test_multi_circuit_failures()
        test_results["scenario_results"].append(multi_circuit_result)
        
        # Test recovery behavior
        recovery_result = await self._test_recovery_behavior()
        test_results["scenario_results"].append(recovery_result)
        
        # Calculate overall results
        total_duration = time.perf_counter() - start_time
        test_results["total_duration_seconds"] = total_duration
        test_results["end_time"] = datetime.now(timezone.utc)
        
        # Analyze results
        test_results["overall_results"] = self._analyze_test_results(test_results["scenario_results"])
        test_results["recommendations"] = self._generate_recommendations(test_results["overall_results"])
        
        logger.info("Circuit breaker test suite completed",
                   duration=total_duration,
                   scenarios=len(self.failure_scenarios),
                   success_rate=test_results["overall_results"]["success_rate"])
        
        return test_results
    
    async def _test_failure_scenario(self, scenario: FailureScenario) -> Dict[str, Any]:
        """Test specific failure scenario"""
        logger.info(f"Testing failure scenario: {scenario.name}",
                   failure_rate=scenario.failure_rate,
                   concurrent_calls=scenario.concurrent_calls)
        
        start_time = time.perf_counter()
        
        # Create circuit breaker for this test
        circuit_name = f"test_{scenario.name}"
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=scenario.recovery_time,
            timeout_seconds=scenario.failure_delay + 1.0
        )
        
        circuit = get_circuit_breaker(circuit_name, config)
        
        # Track results
        successful_calls = 0
        failed_calls = 0
        rejected_calls = 0
        state_changes = []
        
        # Monitor state changes
        def track_state_change(old_state: CircuitState, new_state: CircuitState):
            state_changes.append({
                "timestamp": datetime.now(timezone.utc),
                "from_state": old_state.value,
                "to_state": new_state.value
            })
        
        circuit.add_state_change_callback(track_state_change)
        
        # Create failing function
        async def failing_function():
            if random.random() < scenario.failure_rate:
                await asyncio.sleep(scenario.failure_delay)
                raise Exception(f"Simulated failure for {scenario.name}")
            return f"Success from {scenario.name}"
        
        # Execute concurrent calls
        tasks = []
        for i in range(scenario.concurrent_calls):
            task = self._execute_protected_call(circuit, failing_function, scenario.name)
            tasks.append(task)
        
        # Wait for all calls to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        for result in results:
            if isinstance(result, Exception):
                if isinstance(result, CircuitBreakerOpenException):
                    rejected_calls += 1
                else:
                    failed_calls += 1
            else:
                successful_calls += 1
        
        # Check final circuit state
        final_stats = circuit.get_stats()
        duration = time.perf_counter() - start_time
        
        # Verify expected behavior
        state_correct = final_stats.current_state == scenario.expected_state
        
        result = {
            "scenario_name": scenario.name,
            "description": scenario.description,
            "expected_state": scenario.expected_state.value,
            "actual_state": final_stats.current_state.value,
            "state_correct": state_correct,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "rejected_calls": rejected_calls,
            "total_calls": scenario.concurrent_calls,
            "success_rate": successful_calls / scenario.concurrent_calls,
            "failure_rate": failed_calls / scenario.concurrent_calls,
            "rejection_rate": rejected_calls / scenario.concurrent_calls,
            "state_changes": state_changes,
            "duration_seconds": duration,
            "circuit_stats": {
                "consecutive_failures": final_stats.consecutive_failures,
                "consecutive_successes": final_stats.consecutive_successes,
                "avg_response_time": final_stats.avg_response_time,
                "failure_rate": final_stats.failure_rate
            }
        }
        
        logger.info(f"Scenario '{scenario.name}' completed",
                   state_correct=state_correct,
                   success_rate=result["success_rate"],
                   final_state=final_stats.current_state.value)
        
        return result
    
    async def _execute_protected_call(self, circuit: CircuitBreaker, func: Callable, scenario_name: str) -> Any:
        """Execute function with circuit breaker protection"""
        try:
            return await circuit.call(func)
        except Exception as e:
            # Re-raise for proper tracking
            raise e
    
    async def _test_multi_circuit_failures(self) -> Dict[str, Any]:
        """Test concurrent failures across multiple circuits"""
        logger.info("Testing multi-circuit failure handling")
        start_time = time.perf_counter()
        
        # Create multiple circuits
        circuits = {
            "agent": get_circuit_breaker("test_agent", AGENT_CIRCUIT_CONFIG),
            "ftns": get_circuit_breaker("test_ftns", FTNS_CIRCUIT_CONFIG),
            "ipfs": get_circuit_breaker("test_ipfs", IPFS_CIRCUIT_CONFIG),
            "database": get_circuit_breaker("test_database", DATABASE_CIRCUIT_CONFIG)
        }
        
        # Create different failure functions for each circuit
        async def agent_failure():
            if random.random() < 0.7:
                await asyncio.sleep(2.0)
                raise Exception("Agent timeout")
            return "Agent success"
        
        async def ftns_failure():
            if random.random() < 0.6:
                raise Exception("FTNS overload")
            return "FTNS success"
        
        async def ipfs_failure():
            if random.random() < 0.5:
                await asyncio.sleep(1.0)
                raise Exception("IPFS connection error")
            return "IPFS success"
        
        async def database_failure():
            if random.random() < 0.8:
                raise Exception("Database connection lost")
            return "Database success"
        
        failure_functions = {
            "agent": agent_failure,
            "ftns": ftns_failure,
            "ipfs": ipfs_failure,
            "database": database_failure
        }
        
        # Execute concurrent calls across all circuits
        tasks = []
        calls_per_circuit = 20
        
        for circuit_name, circuit in circuits.items():
            for i in range(calls_per_circuit):
                task = self._execute_protected_call(
                    circuit, 
                    failure_functions[circuit_name], 
                    f"multi_{circuit_name}"
                )
                tasks.append((circuit_name, task))
        
        # Wait for all calls
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        # Analyze results by circuit
        circuit_results = {}
        for i, (circuit_name, _) in enumerate(tasks):
            if circuit_name not in circuit_results:
                circuit_results[circuit_name] = {
                    "successful": 0,
                    "failed": 0,
                    "rejected": 0
                }
            
            result = results[i]
            if isinstance(result, Exception):
                if isinstance(result, CircuitBreakerOpenException):
                    circuit_results[circuit_name]["rejected"] += 1
                else:
                    circuit_results[circuit_name]["failed"] += 1
            else:
                circuit_results[circuit_name]["successful"] += 1
        
        # Get final states
        final_states = {name: circuit.get_stats().current_state.value 
                       for name, circuit in circuits.items()}
        
        duration = time.perf_counter() - start_time
        
        result = {
            "scenario_name": "multi_circuit_failures",
            "description": "Concurrent failures across multiple circuits",
            "circuit_results": circuit_results,
            "final_states": final_states,
            "duration_seconds": duration,
            "total_calls": len(tasks),
            "calls_per_circuit": calls_per_circuit
        }
        
        logger.info("Multi-circuit failure test completed",
                   duration=duration,
                   final_states=final_states)
        
        return result
    
    async def _test_recovery_behavior(self) -> Dict[str, Any]:
        """Test circuit breaker recovery behavior"""
        logger.info("Testing circuit breaker recovery behavior")
        start_time = time.perf_counter()
        
        # Create circuit for recovery testing
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=10.0,  # Short recovery time for testing
            success_threshold=2
        )
        
        circuit = get_circuit_breaker("test_recovery", config)
        
        # Track state transitions
        state_transitions = []
        def track_recovery(old_state: CircuitState, new_state: CircuitState):
            state_transitions.append({
                "timestamp": datetime.now(timezone.utc),
                "from": old_state.value,
                "to": new_state.value
            })
        
        circuit.add_state_change_callback(track_recovery)
        
        # Phase 1: Force circuit to open
        async def always_fail():
            raise Exception("Forced failure")
        
        # Execute failing calls to open circuit
        for i in range(5):
            try:
                await circuit.call(always_fail)
            except:
                pass
        
        phase1_state = circuit.get_stats().current_state
        
        # Phase 2: Wait for recovery timeout
        await asyncio.sleep(12)  # Wait longer than recovery timeout
        
        # Phase 3: Test recovery with successful calls
        async def always_succeed():
            return "Recovery success"
        
        recovery_results = []
        for i in range(5):
            try:
                result = await circuit.call(always_succeed)
                recovery_results.append(True)
            except CircuitBreakerOpenException:
                recovery_results.append(False)
            except Exception as e:
                recovery_results.append(False)
        
        final_state = circuit.get_stats().current_state
        duration = time.perf_counter() - start_time
        
        # Analyze recovery
        successful_recovery = final_state == CircuitState.CLOSED
        recovery_call_success_rate = sum(recovery_results) / len(recovery_results)
        
        result = {
            "scenario_name": "recovery_behavior",
            "description": "Circuit breaker recovery from open to closed state",
            "phase1_state": phase1_state.value,
            "final_state": final_state.value,
            "successful_recovery": successful_recovery,
            "recovery_call_success_rate": recovery_call_success_rate,
            "state_transitions": state_transitions,
            "duration_seconds": duration,
            "recovery_calls": len(recovery_results)
        }
        
        logger.info("Recovery behavior test completed",
                   successful_recovery=successful_recovery,
                   final_state=final_state.value,
                   transitions=len(state_transitions))
        
        return result
    
    def _analyze_test_results(self, scenario_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall test results"""
        
        total_scenarios = len(scenario_results)
        successful_scenarios = sum(1 for r in scenario_results 
                                 if r.get("state_correct", False) or r.get("successful_recovery", False))
        
        # Calculate aggregate metrics
        total_calls = sum(r.get("total_calls", 0) for r in scenario_results)
        total_successful = sum(r.get("successful_calls", 0) for r in scenario_results)
        total_failed = sum(r.get("failed_calls", 0) for r in scenario_results)
        total_rejected = sum(r.get("rejected_calls", 0) for r in scenario_results)
        
        # State distribution
        final_states = [r.get("actual_state") or r.get("final_state") 
                       for r in scenario_results if r.get("actual_state") or r.get("final_state")]
        state_distribution = {}
        for state in final_states:
            if state:
                state_distribution[state] = state_distribution.get(state, 0) + 1
        
        return {
            "total_scenarios": total_scenarios,
            "successful_scenarios": successful_scenarios,
            "success_rate": successful_scenarios / total_scenarios if total_scenarios > 0 else 0,
            "total_calls": total_calls,
            "call_success_rate": total_successful / total_calls if total_calls > 0 else 0,
            "call_failure_rate": total_failed / total_calls if total_calls > 0 else 0,
            "call_rejection_rate": total_rejected / total_calls if total_calls > 0 else 0,
            "state_distribution": state_distribution,
            "circuit_protection_effective": total_rejected > 0,  # Circuit breakers did reject calls
            "failure_isolation_working": successful_scenarios >= total_scenarios * 0.7  # 70% scenarios should work
        }
    
    def _generate_recommendations(self, overall_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        success_rate = overall_results["success_rate"]
        if success_rate < 0.8:
            recommendations.append(
                f"Circuit breaker success rate ({success_rate:.1%}) below target 80%. "
                "Review failure thresholds and recovery timeouts."
            )
        
        rejection_rate = overall_results["call_rejection_rate"]
        if rejection_rate < 0.1:
            recommendations.append(
                "Circuit breakers may not be protecting adequately - very low rejection rate. "
                "Consider lowering failure thresholds."
            )
        elif rejection_rate > 0.5:
            recommendations.append(
                "Circuit breakers may be too aggressive - high rejection rate. "
                "Consider increasing failure thresholds or reducing recovery timeout."
            )
        
        if not overall_results["circuit_protection_effective"]:
            recommendations.append(
                "Circuit breakers did not reject any calls during failures. "
                "Verify circuit breaker configuration and integration."
            )
        
        if not overall_results["failure_isolation_working"]:
            recommendations.append(
                "Failure isolation not working effectively. "
                "Review cascading failure prevention and circuit breaker placement."
            )
        
        # State-specific recommendations
        state_dist = overall_results["state_distribution"]
        open_circuits = state_dist.get("open", 0)
        if open_circuits > len(overall_results) * 0.5:
            recommendations.append(
                "Too many circuits remained open after testing. "
                "Consider shorter recovery timeouts or better fallback mechanisms."
            )
        
        return recommendations
    
    async def test_specific_component(self, component: str, failure_rate: float = 0.5, calls: int = 50) -> Dict[str, Any]:
        """Test circuit breaker for specific PRSM component"""
        logger.info(f"Testing {component} circuit breaker",
                   failure_rate=failure_rate,
                   calls=calls)
        
        if component == "nwtn":
            return await self._test_nwtn_circuit(failure_rate, calls)
        elif component == "ftns":
            return await self._test_ftns_circuit(failure_rate, calls)
        elif component == "ipfs":
            return await self._test_ipfs_circuit(failure_rate, calls)
        else:
            raise ValueError(f"Unknown component: {component}")
    
    async def _test_nwtn_circuit(self, failure_rate: float, calls: int) -> Dict[str, Any]:
        """Test NWTN orchestrator circuit breaker"""
        circuit = get_circuit_breaker("nwtn_agent_pipeline", AGENT_CIRCUIT_CONFIG)
        
        async def simulate_agent_call():
            if random.random() < failure_rate:
                await asyncio.sleep(2.0)  # Simulate timeout
                raise Exception("Agent pipeline timeout")
            return {"response": "Agent pipeline success", "agents_used": ["architect", "prompter"]}
        
        return await self._execute_component_test("nwtn", circuit, simulate_agent_call, calls)
    
    async def _test_ftns_circuit(self, failure_rate: float, calls: int) -> Dict[str, Any]:
        """Test FTNS service circuit breaker"""
        circuit = get_circuit_breaker("ftns_service", FTNS_CIRCUIT_CONFIG)
        
        async def simulate_ftns_call():
            if random.random() < failure_rate:
                raise Exception("FTNS service overload")
            return {"cost": 0.05, "balance": 100.0}
        
        return await self._execute_component_test("ftns", circuit, simulate_ftns_call, calls)
    
    async def _test_ipfs_circuit(self, failure_rate: float, calls: int) -> Dict[str, Any]:
        """Test IPFS client circuit breaker"""
        circuit = get_circuit_breaker("ipfs_client", IPFS_CIRCUIT_CONFIG)
        
        async def simulate_ipfs_call():
            if random.random() < failure_rate:
                await asyncio.sleep(1.5)
                raise Exception("IPFS connection timeout")
            return {"cid": "QmXXXXX", "size": 1024}
        
        return await self._execute_component_test("ipfs", circuit, simulate_ipfs_call, calls)
    
    async def _execute_component_test(self, component: str, circuit: CircuitBreaker, func: Callable, calls: int) -> Dict[str, Any]:
        """Execute test for specific component"""
        start_time = time.perf_counter()
        
        successful = 0
        failed = 0
        rejected = 0
        
        initial_stats = circuit.get_stats()
        
        # Execute calls
        for i in range(calls):
            try:
                await circuit.call(func)
                successful += 1
            except CircuitBreakerOpenException:
                rejected += 1
            except Exception:
                failed += 1
        
        final_stats = circuit.get_stats()
        duration = time.perf_counter() - start_time
        
        return {
            "component": component,
            "calls": calls,
            "successful": successful,
            "failed": failed,
            "rejected": rejected,
            "success_rate": successful / calls,
            "failure_rate": failed / calls,
            "rejection_rate": rejected / calls,
            "initial_state": initial_stats.current_state.value,
            "final_state": final_stats.current_state.value,
            "duration_seconds": duration,
            "circuit_responded_correctly": final_stats.current_state != CircuitState.CLOSED if failed > 5 else True
        }


# === Test Execution Functions ===

async def run_quick_circuit_test():
    """Run quick circuit breaker test"""
    tester = CircuitBreakerTester()
    
    print("🔧 Running Quick Circuit Breaker Test")
    
    # Test a few key scenarios
    scenarios_to_test = ["agent_timeout_cascade", "ftns_overload", "partial_failure_recovery"]
    
    results = []
    for scenario_name in scenarios_to_test:
        scenario = next(s for s in tester.failure_scenarios if s.name == scenario_name)
        result = await tester._test_failure_scenario(scenario)
        results.append(result)
        
        print(f"  {scenario.name}: {'✅' if result['state_correct'] else '❌'} "
              f"(Success: {result['success_rate']:.1%}, State: {result['actual_state']})")
    
    # Test recovery
    recovery_result = await tester._test_recovery_behavior()
    print(f"  Recovery: {'✅' if recovery_result['successful_recovery'] else '❌'} "
          f"(Final: {recovery_result['final_state']})")
    
    overall_success = all(r["state_correct"] for r in results) and recovery_result["successful_recovery"]
    
    print(f"\n{'✅' if overall_success else '❌'} Quick circuit breaker test {'PASSED' if overall_success else 'FAILED'}")
    
    return overall_success


async def run_comprehensive_circuit_test():
    """Run comprehensive circuit breaker test suite"""
    tester = CircuitBreakerTester()
    
    print("🧪 Running Comprehensive Circuit Breaker Test Suite")
    print("This will test all failure scenarios and recovery behavior...")
    
    results = await tester.run_comprehensive_test_suite()
    
    print(f"\n=== Circuit Breaker Test Results ===")
    print(f"Test Suite ID: {results['test_suite_id']}")
    print(f"Duration: {results['total_duration_seconds']:.2f}s")
    print(f"Scenarios Tested: {results['scenarios_tested']}")
    
    overall = results["overall_results"]
    print(f"\nOverall Results:")
    print(f"  Scenario Success Rate: {overall['success_rate']:.1%}")
    print(f"  Call Success Rate: {overall['call_success_rate']:.1%}")
    print(f"  Call Rejection Rate: {overall['call_rejection_rate']:.1%}")
    print(f"  Protection Effective: {'✅' if overall['circuit_protection_effective'] else '❌'}")
    print(f"  Failure Isolation: {'✅' if overall['failure_isolation_working'] else '❌'}")
    
    print(f"\nScenario Results:")
    for scenario in results["scenario_results"]:
        name = scenario.get("scenario_name", "unknown")
        state_correct = scenario.get("state_correct", scenario.get("successful_recovery", False))
        success_rate = scenario.get("success_rate", scenario.get("recovery_call_success_rate", 0))
        
        print(f"  {name}: {'✅' if state_correct else '❌'} (Success: {success_rate:.1%})")
    
    if results["recommendations"]:
        print(f"\nRecommendations:")
        for i, rec in enumerate(results["recommendations"], 1):
            print(f"  {i}. {rec}")
    
    return results


async def test_component_circuits():
    """Test circuit breakers for specific PRSM components"""
    tester = CircuitBreakerTester()
    
    print("🔧 Testing Component-Specific Circuit Breakers")
    
    components = ["nwtn", "ftns", "ipfs"]
    results = {}
    
    for component in components:
        print(f"\nTesting {component.upper()} circuit breaker...")
        result = await tester.test_specific_component(component, failure_rate=0.6, calls=25)
        results[component] = result
        
        success = result["circuit_responded_correctly"]
        print(f"  {component}: {'✅' if success else '❌'} "
              f"(Success: {result['success_rate']:.1%}, "
              f"Rejected: {result['rejection_rate']:.1%}, "
              f"State: {result['final_state']})")
    
    all_passed = all(r["circuit_responded_correctly"] for r in results.values())
    print(f"\n{'✅' if all_passed else '❌'} Component circuit tests {'PASSED' if all_passed else 'FAILED'}")
    
    return results


# === Main Test Execution ===

if __name__ == "__main__":
    import sys
    
    async def run_tests():
        """Run circuit breaker tests"""
        if len(sys.argv) > 1:
            test_type = sys.argv[1]
            
            if test_type == "quick":
                return await run_quick_circuit_test()
            elif test_type == "comprehensive":
                return await run_comprehensive_circuit_test()
            elif test_type == "components":
                return await test_component_circuits()
            else:
                print("Usage: python test-circuit-breakers.py [quick|comprehensive|components]")
                return False
        else:
            # Default to quick test
            return await run_quick_circuit_test()
    
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)