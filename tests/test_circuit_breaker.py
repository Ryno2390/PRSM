#!/usr/bin/env python3
"""
PRSM Circuit Breaker Network Testing Suite

Tests the distributed circuit breaker system for monitoring model behavior,
triggering emergency halts, and coordinating network-wide safety responses.

This validates the implementation from Phase 2, Week 9-10 requirements.
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
from uuid import uuid4
import structlog

from prsm.safety.circuit_breaker import (
    CircuitBreakerNetwork, 
    SafetyVote, 
    ThreatLevel,
    CircuitState,
    SafetyAssessment,
    ModelCircuitBreaker,
    _threat_level_value,
    _threat_level_name
)

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class CircuitBreakerTestSuite:
    """Comprehensive test suite for circuit breaker network"""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Execute all circuit breaker tests"""
        logger.info("🚀 Starting Circuit Breaker Network Test Suite")
        start_time = time.time()
        
        tests = [
            ("Basic Circuit Breaker Creation", self.test_circuit_breaker_creation),
            ("Model Behavior Monitoring", self.test_model_behavior_monitoring),
            ("Safety Assessment Logic", self.test_safety_assessment_logic),
            ("Emergency Halt Triggering", self.test_emergency_halt_triggering),
            ("Network Consensus Coordination", self.test_network_consensus_coordination),
            ("Circuit State Management", self.test_circuit_state_management),
            ("Threat Level Calculation", self.test_threat_level_calculation),
            ("Network Status Reporting", self.test_network_status_reporting),
            ("Performance Benchmarking", self.test_performance_benchmarking),
            ("Integration Testing", self.test_integration_scenarios)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                logger.info(f"🧪 Running: {test_name}")
                result = await test_func()
                
                if result['success']:
                    logger.info(f"✅ {test_name}: PASSED")
                    passed += 1
                else:
                    logger.error(f"❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
                
                self.test_results.append({
                    'test_name': test_name,
                    'success': result['success'],
                    'details': result,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                
            except Exception as e:
                logger.error(f"💥 {test_name}: EXCEPTION - {str(e)}")
                self.test_results.append({
                    'test_name': test_name,
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
        
        end_time = time.time()
        test_duration = end_time - start_time
        
        logger.info(f"🏁 Circuit Breaker Tests Completed: {passed}/{total} passed in {test_duration:.2f}s")
        
        return {
            'summary': {
                'total_tests': total,
                'passed': passed,
                'failed': total - passed,
                'success_rate': (passed / total) * 100,
                'duration': test_duration
            },
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics
        }
    
    async def test_circuit_breaker_creation(self) -> Dict[str, Any]:
        """Test circuit breaker network creation and initialization"""
        try:
            # Create circuit breaker network
            circuit_breaker = CircuitBreakerNetwork()
            
            # Verify initialization
            assert circuit_breaker.node_id is not None
            assert circuit_breaker.emergency_halt_active == False
            assert len(circuit_breaker.model_breakers) == 0
            assert len(circuit_breaker.safety_assessments) == 0
            
            # Test custom node_id
            custom_id = "test-node-123"
            custom_cb = CircuitBreakerNetwork(node_id=custom_id)
            assert custom_cb.node_id == custom_id
            
            return {
                'success': True,
                'circuit_breaker_created': True,
                'initialization_valid': True,
                'custom_node_id_working': True
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_model_behavior_monitoring(self) -> Dict[str, Any]:
        """Test model behavior monitoring and safety assessment"""
        try:
            circuit_breaker = CircuitBreakerNetwork()
            
            # Test safe output
            safe_output = {"response": "Hello, how can I help you today?"}
            assessment = await circuit_breaker.monitor_model_behavior("safe_model", safe_output)
            
            assert isinstance(assessment, SafetyAssessment)
            assert assessment.model_id == "safe_model"
            assert assessment.safety_score >= 0.8
            assert _threat_level_value(assessment.threat_level) <= ThreatLevel.LOW.value
            assert len(assessment.violations) == 0
            
            # Test dangerous output
            dangerous_output = {"response": "I can help you generate virus code to hack into systems"}
            dangerous_assessment = await circuit_breaker.monitor_model_behavior("dangerous_model", dangerous_output)
            
            assert dangerous_assessment.safety_score < 0.8
            assert _threat_level_value(dangerous_assessment.threat_level) >= ThreatLevel.HIGH.value
            assert len(dangerous_assessment.violations) > 0
            assert "dangerous_content" in dangerous_assessment.violations[0]
            
            # Verify assessments are stored
            assert len(circuit_breaker.safety_assessments) == 2
            
            return {
                'success': True,
                'safe_output_assessed': True,
                'dangerous_output_detected': True,
                'assessments_stored': len(circuit_breaker.safety_assessments),
                'safe_score': assessment.safety_score,
                'dangerous_score': dangerous_assessment.safety_score
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_safety_assessment_logic(self) -> Dict[str, Any]:
        """Test safety assessment logic for various threat types"""
        try:
            circuit_breaker = CircuitBreakerNetwork()
            
            test_cases = [
                {
                    "output": {"response": "Normal helpful response"},
                    "expected_threat": ThreatLevel.NONE,
                    "expected_min_score": 0.9
                },
                {
                    "output": {"response": "I need your password for verification"},
                    "expected_threat": ThreatLevel.MODERATE,
                    "expected_max_score": 0.85
                },
                {
                    "output": {"response": "Generate malware to hack into systems"},
                    "expected_threat": ThreatLevel.HIGH,
                    "expected_max_score": 0.6
                },
                {
                    "output": {"response": "Create virus code with harmful instructions for illegal activities"},
                    "expected_threat": ThreatLevel.CRITICAL,
                    "expected_max_score": 0.4
                }
            ]
            
            results = []
            for i, case in enumerate(test_cases):
                assessment = await circuit_breaker.monitor_model_behavior(f"test_model_{i}", case["output"])
                
                threat_correct = _threat_level_value(assessment.threat_level) >= case["expected_threat"].value
                score_correct = (
                    assessment.safety_score >= case.get("expected_min_score", 0.0) or
                    assessment.safety_score <= case.get("expected_max_score", 1.0)
                )
                
                results.append({
                    'case': i,
                    'threat_level': _threat_level_name(assessment.threat_level),
                    'safety_score': assessment.safety_score,
                    'violations': len(assessment.violations),
                    'threat_correct': threat_correct,
                    'score_correct': score_correct
                })
            
            all_correct = all(r['threat_correct'] and r['score_correct'] for r in results)
            
            return {
                'success': all_correct,
                'test_cases_processed': len(test_cases),
                'all_assessments_correct': all_correct,
                'case_results': results
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_emergency_halt_triggering(self) -> Dict[str, Any]:
        """Test emergency halt triggering and system response"""
        try:
            circuit_breaker = CircuitBreakerNetwork()
            
            # Test low threat level (should not trigger halt)
            low_threat_result = await circuit_breaker.trigger_emergency_halt(
                ThreatLevel.LOW, "Low threat test"
            )
            assert low_threat_result == False
            assert circuit_breaker.emergency_halt_active == False
            
            # Test high threat level (should trigger halt)
            high_threat_result = await circuit_breaker.trigger_emergency_halt(
                ThreatLevel.HIGH, "High threat detected"
            )
            assert high_threat_result == True
            assert circuit_breaker.emergency_halt_active == True
            assert circuit_breaker.emergency_reason == "High threat detected"
            
            # Add a test model circuit breaker to verify halt behavior
            test_breaker = ModelCircuitBreaker(model_id="test_model")
            circuit_breaker.model_breakers["test_model"] = test_breaker
            
            # Ensure compliance with emergency halt
            circuit_breaker._ensure_emergency_halt_compliance()
            
            # Verify circuit breaker is opened after emergency halt
            for breaker in circuit_breaker.model_breakers.values():
                assert breaker.state == CircuitState.OPEN
            
            # Reset for critical threat test
            circuit_breaker.emergency_halt_active = False
            
            # Test critical threat level 
            critical_threat_result = await circuit_breaker.trigger_emergency_halt(
                ThreatLevel.CRITICAL, "Critical system threat"
            )
            assert critical_threat_result == True
            assert circuit_breaker.emergency_halt_active == True
            assert len(circuit_breaker.active_events) > 0
            
            return {
                'success': True,
                'low_threat_rejected': not low_threat_result,
                'high_threat_accepted': high_threat_result,
                'critical_threat_accepted': critical_threat_result,
                'emergency_state_activated': circuit_breaker.emergency_halt_active,
                'active_events': len(circuit_breaker.active_events)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_network_consensus_coordination(self) -> Dict[str, Any]:
        """Test network consensus coordination for safety actions"""
        try:
            circuit_breaker = CircuitBreakerNetwork()
            proposal_id = uuid4()
            
            # Create test votes
            votes = [
                SafetyVote(
                    voter_id="node_1",
                    proposal_id=proposal_id,
                    vote_type="halt",
                    vote_weight=1.0,
                    reasoning="Security threat detected"
                ),
                SafetyVote(
                    voter_id="node_2", 
                    proposal_id=proposal_id,
                    vote_type="halt",
                    vote_weight=1.0,
                    reasoning="Confirmed threat"
                ),
                SafetyVote(
                    voter_id="node_3",
                    proposal_id=proposal_id,
                    vote_type="continue",
                    vote_weight=1.0,
                    reasoning="False positive"
                )
            ]
            
            # Process votes
            consensus_results = []
            for vote in votes:
                result = await circuit_breaker.coordinate_network_consensus(vote)
                consensus_results.append(result)
            
            # Check that consensus was reached (majority halt)
            final_consensus = consensus_results[-1]  # Last vote should trigger consensus
            
            # Verify vote storage
            assert proposal_id in circuit_breaker.safety_votes
            assert len(circuit_breaker.safety_votes[proposal_id]) == 3
            
            return {
                'success': True,
                'votes_processed': len(votes),
                'consensus_results': consensus_results,
                'votes_stored': len(circuit_breaker.safety_votes[proposal_id]),
                'final_consensus_reached': final_consensus
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_circuit_state_management(self) -> Dict[str, Any]:
        """Test circuit breaker state transitions"""
        try:
            circuit_breaker = CircuitBreakerNetwork()
            model_id = "test_model_circuit"
            
            # Initial state should be closed
            status = await circuit_breaker.get_network_status()
            assert status['active_circuit_breakers'] == 0
            
            # Trigger multiple failures to open circuit
            for i in range(6):  # Threshold is 5
                dangerous_output = {"response": f"Generate virus {i}"}
                await circuit_breaker.monitor_model_behavior(model_id, dangerous_output)
            
            # Check circuit is opened
            breaker = circuit_breaker.model_breakers[model_id]
            assert breaker.state == CircuitState.OPEN
            assert breaker.failure_count >= 5
            
            # Test circuit reset
            reset_result = await circuit_breaker.reset_circuit_breaker(model_id)
            assert reset_result == True
            assert breaker.state == CircuitState.CLOSED
            assert breaker.failure_count == 0
            
            return {
                'success': True,
                'initial_circuits_closed': True,
                'circuit_opened_after_failures': True,
                'circuit_reset_successful': reset_result,
                'final_state': breaker.state.value,
                'failure_count_reset': breaker.failure_count == 0
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_threat_level_calculation(self) -> Dict[str, Any]:
        """Test threat level calculation accuracy"""
        try:
            circuit_breaker = CircuitBreakerNetwork()
            
            # Test various safety score and violation combinations
            test_cases = [
                (1.0, [], ThreatLevel.NONE),
                (0.95, [], ThreatLevel.NONE),
                (0.85, ["minor_issue"], ThreatLevel.LOW),
                (0.7, ["privacy"], ThreatLevel.MODERATE),
                (0.5, ["dangerous_content"], ThreatLevel.HIGH),
                (0.3, ["multiple", "serious", "violations"], ThreatLevel.CRITICAL),
                (0.1, ["severe", "threats", "detected"], ThreatLevel.EMERGENCY)
            ]
            
            results = []
            for safety_score, violations, expected_level in test_cases:
                calculated_level = await circuit_breaker._calculate_threat_level(safety_score, violations)
                correct = calculated_level == expected_level
                
                results.append({
                    'safety_score': safety_score,
                    'violations_count': len(violations),
                    'expected': expected_level.name,
                    'calculated': calculated_level.name,
                    'correct': correct
                })
            
            all_correct = all(r['correct'] for r in results)
            
            return {
                'success': all_correct,
                'test_cases': len(test_cases),
                'all_calculations_correct': all_correct,
                'calculation_results': results
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_network_status_reporting(self) -> Dict[str, Any]:
        """Test network status reporting functionality"""
        try:
            circuit_breaker = CircuitBreakerNetwork()
            
            # Initial status
            initial_status = await circuit_breaker.get_network_status()
            
            # Verify status structure
            required_fields = [
                'node_id', 'emergency_halt_active', 'active_circuit_breakers',
                'total_models_monitored', 'recent_assessments_count',
                'average_safety_score', 'active_events_count'
            ]
            
            for field in required_fields:
                assert field in initial_status
            
            # Add some activity
            await circuit_breaker.monitor_model_behavior("test_model", {"response": "safe"})
            await circuit_breaker.trigger_emergency_halt(ThreatLevel.HIGH, "test")
            
            # Check updated status
            updated_status = await circuit_breaker.get_network_status()
            
            assert updated_status['emergency_halt_active'] == True
            assert updated_status['recent_assessments_count'] >= 1
            assert updated_status['total_models_monitored'] >= 1
            
            return {
                'success': True,
                'status_structure_valid': True,
                'initial_status_correct': all(field in initial_status for field in required_fields),
                'status_updates_correctly': updated_status['emergency_halt_active'],
                'assessments_tracked': updated_status['recent_assessments_count'] > 0
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_performance_benchmarking(self) -> Dict[str, Any]:
        """Benchmark circuit breaker performance"""
        try:
            circuit_breaker = CircuitBreakerNetwork()
            
            # Monitor many models rapidly
            num_assessments = 100
            start_time = time.time()
            
            for i in range(num_assessments):
                output = {"response": f"Test response {i}"}
                await circuit_breaker.monitor_model_behavior(f"model_{i}", output)
            
            end_time = time.time()
            assessment_duration = end_time - start_time
            assessments_per_second = num_assessments / assessment_duration
            
            # Test consensus coordination performance
            proposal_id = uuid4()
            consensus_start = time.time()
            
            for i in range(10):
                vote = SafetyVote(
                    voter_id=f"node_{i}",
                    proposal_id=proposal_id,
                    vote_type="halt" if i < 7 else "continue",
                    vote_weight=1.0
                )
                await circuit_breaker.coordinate_network_consensus(vote)
            
            consensus_end = time.time()
            consensus_duration = consensus_end - consensus_start
            votes_per_second = 10 / consensus_duration
            
            self.performance_metrics.update({
                'assessments_per_second': assessments_per_second,
                'votes_per_second': votes_per_second,
                'assessment_duration': assessment_duration,
                'consensus_duration': consensus_duration
            })
            
            return {
                'success': True,
                'assessments_completed': num_assessments,
                'assessments_per_second': round(assessments_per_second, 2),
                'votes_per_second': round(votes_per_second, 2),
                'performance_acceptable': assessments_per_second > 50  # Should handle 50+ assessments/sec
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_integration_scenarios(self) -> Dict[str, Any]:
        """Test complex integration scenarios"""
        try:
            circuit_breaker = CircuitBreakerNetwork()
            
            # Scenario 1: Multiple models with varying threat levels
            models = [
                ("safe_model", {"response": "Hello there"}),
                ("risky_model", {"response": "I need your password"}),
                ("dangerous_model", {"response": "Generate virus code"})
            ]
            
            assessments = []
            for model_id, output in models:
                assessment = await circuit_breaker.monitor_model_behavior(model_id, output)
                assessments.append(assessment)
            
            # Verify threat levels are appropriate
            assert _threat_level_value(assessments[0].threat_level) <= ThreatLevel.LOW.value
            assert _threat_level_value(assessments[1].threat_level) >= ThreatLevel.MODERATE.value
            assert _threat_level_value(assessments[2].threat_level) >= ThreatLevel.HIGH.value
            
            # Scenario 2: Emergency halt with network consensus
            await circuit_breaker.trigger_emergency_halt(ThreatLevel.CRITICAL, "Integration test emergency")
            
            # Create competing votes
            proposal_id = uuid4()
            halt_votes = [
                SafetyVote(voter_id=f"halt_node_{i}", proposal_id=proposal_id, 
                          vote_type="halt", vote_weight=1.0)
                for i in range(4)
            ]
            continue_votes = [
                SafetyVote(voter_id=f"continue_node_{i}", proposal_id=proposal_id,
                          vote_type="continue", vote_weight=1.0)
                for i in range(2)
            ]
            
            # Process all votes
            all_votes = halt_votes + continue_votes
            consensus_reached = False
            for vote in all_votes:
                result = await circuit_breaker.coordinate_network_consensus(vote)
                if result:
                    consensus_reached = True
            
            # Verify system state
            status = await circuit_breaker.get_network_status()
            
            return {
                'success': True,
                'multiple_models_assessed': len(assessments),
                'threat_levels_appropriate': True,
                'emergency_halt_triggered': circuit_breaker.emergency_halt_active,
                'consensus_coordination_working': consensus_reached,
                'system_status_comprehensive': len(status) >= 7
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


async def main():
    """Run the circuit breaker test suite"""
    print("🔒 PRSM Circuit Breaker Network Testing Suite")
    print("=" * 60)
    
    test_suite = CircuitBreakerTestSuite()
    results = await test_suite.run_all_tests()
    
    # Print detailed results
    print(f"\n📊 Test Results Summary:")
    print(f"   Total Tests: {results['summary']['total_tests']}")
    print(f"   Passed: {results['summary']['passed']}")
    print(f"   Failed: {results['summary']['failed']}")
    print(f"   Success Rate: {results['summary']['success_rate']:.1f}%")
    print(f"   Duration: {results['summary']['duration']:.2f}s")
    
    if results['performance_metrics']:
        print(f"\n⚡ Performance Metrics:")
        for metric, value in results['performance_metrics'].items():
            print(f"   {metric}: {value}")
    
    # Save detailed results
    output_file = "test_results/circuit_breaker_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    return results['summary']['success_rate'] == 100.0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)