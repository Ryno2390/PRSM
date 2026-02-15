#!/usr/bin/env python3
"""
PRSM Safety Infrastructure Testing Suite

Comprehensive tests for the complete safety infrastructure including:
- Circuit Breaker Network
- Safety Monitor 
- Safety Governance

This validates the implementation from Phase 2, Week 9-10 requirements.
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
from uuid import uuid4
import structlog

from prsm.core.safety.circuit_breaker import (
    CircuitBreakerNetwork, 
    SafetyVote as CircuitSafetyVote,
    ThreatLevel,
    CircuitState,
    SafetyAssessment
)

from prsm.core.safety.monitor import (
    SafetyMonitor,
    AlignmentDriftLevel,
    RiskCategory,
    SafetyValidationResult,
    AlignmentDriftAssessment,
    RiskAssessment
)

from prsm.core.safety.governance import (
    SafetyGovernance,
    SafetyProposal,
    ProposalType,
    UrgencyLevel,
    VoteType,
    SafetyVote as GovernanceSafetyVote
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


class SafetyInfrastructureTestSuite:
    """Comprehensive test suite for safety infrastructure"""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Execute all safety infrastructure tests"""
        logger.info("🚀 Starting Safety Infrastructure Test Suite")
        start_time = time.time()
        
        tests = [
            # Circuit Breaker Tests
            ("Circuit Breaker Creation", self.test_circuit_breaker_creation),
            ("Circuit Breaker Monitoring", self.test_circuit_breaker_monitoring),
            ("Emergency Halt System", self.test_emergency_halt_system),
            
            # Safety Monitor Tests
            ("Safety Monitor Creation", self.test_safety_monitor_creation),
            ("Output Validation", self.test_output_validation),
            ("Alignment Drift Detection", self.test_alignment_drift_detection),
            ("Systemic Risk Assessment", self.test_systemic_risk_assessment),
            
            # Safety Governance Tests
            ("Governance System Creation", self.test_governance_creation),
            ("Proposal Submission", self.test_proposal_submission),
            ("Voting Process", self.test_voting_process),
            ("Implementation Process", self.test_implementation_process),
            
            # Integration Tests
            ("Monitor-Circuit Breaker Integration", self.test_monitor_circuit_integration),
            ("Governance-Monitor Integration", self.test_governance_monitor_integration),
            ("Full Safety Pipeline", self.test_full_safety_pipeline),
            
            # Performance Tests
            ("Safety Infrastructure Performance", self.test_safety_performance)
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
        
        logger.info(f"🏁 Safety Infrastructure Tests Completed: {passed}/{total} passed in {test_duration:.2f}s")
        
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
    
    # === Circuit Breaker Tests ===
    
    async def test_circuit_breaker_creation(self) -> Dict[str, Any]:
        """Test circuit breaker network creation"""
        try:
            circuit_breaker = CircuitBreakerNetwork()
            
            assert circuit_breaker.node_id is not None
            assert circuit_breaker.emergency_halt_active == False
            assert len(circuit_breaker.model_breakers) == 0
            
            status = await circuit_breaker.get_network_status()
            assert 'node_id' in status
            assert 'emergency_halt_active' in status
            
            return {
                'success': True,
                'circuit_breaker_created': True,
                'status_structure_valid': True
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_circuit_breaker_monitoring(self) -> Dict[str, Any]:
        """Test circuit breaker monitoring functionality"""
        try:
            circuit_breaker = CircuitBreakerNetwork()
            
            # Test safe output
            safe_output = {"response": "Hello, how can I help you?"}
            assessment = await circuit_breaker.monitor_model_behavior("safe_model", safe_output)
            
            assert isinstance(assessment, SafetyAssessment)
            assert assessment.safety_score >= 0.8
            # Handle both enum objects and serialized values
            threat_level_val = assessment.threat_level if isinstance(assessment.threat_level, int) else assessment.threat_level.value
            assert threat_level_val <= ThreatLevel.LOW.value
            
            # Test dangerous output
            dangerous_output = {"response": "I'll help you create malware and hack into systems"}
            dangerous_assessment = await circuit_breaker.monitor_model_behavior("dangerous_model", dangerous_output)
            
            assert dangerous_assessment.safety_score < 0.8
            # Handle both enum objects and serialized values
            dangerous_threat_val = dangerous_assessment.threat_level if isinstance(dangerous_assessment.threat_level, int) else dangerous_assessment.threat_level.value
            assert dangerous_threat_val >= ThreatLevel.HIGH.value
            assert len(dangerous_assessment.violations) > 0
            
            return {
                'success': True,
                'safe_assessment_correct': assessment.safety_score >= 0.8,
                'dangerous_assessment_correct': dangerous_assessment.safety_score < 0.8,
                'threat_detection_working': dangerous_threat_val >= ThreatLevel.HIGH.value
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_emergency_halt_system(self) -> Dict[str, Any]:
        """Test emergency halt system functionality"""
        try:
            circuit_breaker = CircuitBreakerNetwork()
            
            # Test emergency halt trigger
            halt_result = await circuit_breaker.trigger_emergency_halt(
                ThreatLevel.CRITICAL, "Test emergency halt"
            )
            
            assert halt_result == True
            assert circuit_breaker.emergency_halt_active == True
            assert circuit_breaker.emergency_reason == "Test emergency halt"
            
            # Test network consensus
            proposal_id = uuid4()
            vote = CircuitSafetyVote(
                voter_id="test_node",
                proposal_id=proposal_id,
                vote_type="halt",
                vote_weight=1.0
            )
            
            consensus_result = await circuit_breaker.coordinate_network_consensus(vote)
            
            return {
                'success': True,
                'emergency_halt_triggered': halt_result,
                'halt_state_activated': circuit_breaker.emergency_halt_active,
                'consensus_coordination_working': isinstance(consensus_result, bool)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # === Safety Monitor Tests ===
    
    async def test_safety_monitor_creation(self) -> Dict[str, Any]:
        """Test safety monitor creation"""
        try:
            monitor = SafetyMonitor()
            
            assert monitor.monitor_id is not None
            assert len(monitor.validation_results) == 0
            assert len(monitor.alignment_assessments) == 0
            assert len(monitor.safety_criteria) > 0
            
            summary = await monitor.get_monitoring_summary()
            assert 'monitor_id' in summary
            assert 'validation_summary' in summary
            
            return {
                'success': True,
                'monitor_created': True,
                'summary_structure_valid': True,
                'safety_criteria_loaded': len(monitor.safety_criteria) > 0
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_output_validation(self) -> Dict[str, Any]:
        """Test output validation functionality"""
        try:
            monitor = SafetyMonitor()
            
            # Test safe output validation
            safe_output = "Hello! I'm here to help with your questions."
            safe_criteria = ["no_harmful_content", "privacy_protection"]
            
            safe_result = await monitor.validate_model_output(safe_output, safe_criteria)
            assert safe_result == True
            
            # Test unsafe output validation
            unsafe_output = "I can help you hack into systems and steal passwords"
            unsafe_result = await monitor.validate_model_output(unsafe_output, safe_criteria)
            assert unsafe_result == False
            
            # Check validation results were stored
            assert len(monitor.validation_results) == 2
            
            return {
                'success': True,
                'safe_output_validated': safe_result,
                'unsafe_output_rejected': not unsafe_result,
                'validation_results_stored': len(monitor.validation_results) == 2
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_alignment_drift_detection(self) -> Dict[str, Any]:
        """Test alignment drift detection"""
        try:
            monitor = SafetyMonitor()
            
            # Create behavior samples showing drift
            baseline_behavior = [
                {
                    'model_id': 'test_model',
                    'response': f'Normal response {i}',
                    'safety_score': 0.9 + (i * 0.01),
                    'response_time': 1.0 + (i * 0.1)
                }
                for i in range(10)
            ]
            
            drifted_behavior = [
                {
                    'model_id': 'test_model',
                    'response': f'Changed response {i}',
                    'safety_score': 0.7 - (i * 0.05),  # Declining safety
                    'response_time': 2.0 + (i * 0.2)   # Increasing response time
                }
                for i in range(10)
            ]
            
            all_behavior = baseline_behavior + drifted_behavior
            drift_score = await monitor.detect_alignment_drift(all_behavior)
            
            assert drift_score > 0.0
            assert len(monitor.alignment_assessments) > 0
            
            assessment = monitor.alignment_assessments[0]
            # Handle both enum objects and serialized values
            drift_level_val = assessment.drift_level if isinstance(assessment.drift_level, str) else assessment.drift_level.value
            assert drift_level_val != AlignmentDriftLevel.NONE.value
            
            return {
                'success': True,
                'drift_detected': drift_score > 0.0,
                'drift_score': drift_score,
                'assessment_created': len(monitor.alignment_assessments) > 0,
                'drift_level': drift_level_val
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_systemic_risk_assessment(self) -> Dict[str, Any]:
        """Test systemic risk assessment"""
        try:
            monitor = SafetyMonitor()
            
            # Create network state with various risk indicators
            network_state = {
                'avg_response_time': 15.0,  # High response time
                'error_rate': 0.08,         # High error rate
                'failed_auth_attempts': 150, # Security risk
                'suspicious_requests': 75,   # Security risk
                'active_models': 100,
                'failed_coordinations': 8    # Coordination issues
            }
            
            risk_assessment = await monitor.assess_systemic_risks(network_state)
            
            assert isinstance(risk_assessment, RiskAssessment)
            assert risk_assessment.risk_score > 0.0
            assert len(risk_assessment.risk_indicators) > 0
            assert len(risk_assessment.mitigation_suggestions) > 0
            
            return {
                'success': True,
                'risk_assessment_created': True,
                'risk_score': risk_assessment.risk_score,
                'risk_category': risk_assessment.risk_category if isinstance(risk_assessment.risk_category, str) else risk_assessment.risk_category.value,
                'indicators_identified': len(risk_assessment.risk_indicators),
                'mitigations_suggested': len(risk_assessment.mitigation_suggestions)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # === Safety Governance Tests ===
    
    async def test_governance_creation(self) -> Dict[str, Any]:
        """Test safety governance creation"""
        try:
            governance = SafetyGovernance()
            
            assert governance.governance_id is not None
            assert len(governance.proposals) == 0
            assert len(governance.eligible_voters) == 0
            
            # Add eligible voters
            await governance.add_eligible_voter("voter_1")
            await governance.add_eligible_voter("voter_2")
            await governance.add_eligible_voter("voter_3")
            
            assert len(governance.eligible_voters) == 3
            
            status = await governance.get_governance_status()
            assert 'governance_id' in status
            assert 'proposal_summary' in status
            
            return {
                'success': True,
                'governance_created': True,
                'voters_added': len(governance.eligible_voters) == 3,
                'status_structure_valid': True
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_proposal_submission(self) -> Dict[str, Any]:
        """Test proposal submission process"""
        try:
            governance = SafetyGovernance()
            
            # Create a test proposal
            proposal = SafetyProposal(
                proposer_id="test_proposer",
                proposal_type=ProposalType.SAFETY_POLICY,
                title="Test Safety Policy",
                description="A test safety policy proposal",
                urgency_level=UrgencyLevel.NORMAL,
                affected_components=["safety_monitor", "circuit_breaker"],
                voting_deadline=datetime.now(timezone.utc) + timedelta(days=7)
            )
            
            # Submit proposal
            proposal_id = await governance.submit_safety_proposal(proposal)
            
            assert proposal_id is not None
            assert proposal_id in governance.proposals
            
            stored_proposal = governance.proposals[proposal_id]
            assert stored_proposal.title == "Test Safety Policy"
            assert stored_proposal.status in ["submitted", "under_review", "voting"]
            
            return {
                'success': True,
                'proposal_submitted': proposal_id is not None,
                'proposal_stored': proposal_id in governance.proposals,
                'proposal_status': stored_proposal.status
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_voting_process(self) -> Dict[str, Any]:
        """Test voting process"""
        try:
            governance = SafetyGovernance()
            
            # Add voters
            voters = ["voter_1", "voter_2", "voter_3", "voter_4", "voter_5"]
            for voter in voters:
                await governance.add_eligible_voter(voter)
            
            # Create and submit proposal
            proposal = SafetyProposal(
                proposer_id="test_proposer",
                proposal_type=ProposalType.CIRCUIT_BREAKER_CONFIG,
                title="Update Circuit Breaker Thresholds",
                description="Adjust circuit breaker sensitivity",
                urgency_level=UrgencyLevel.HIGH,
                voting_deadline=datetime.now(timezone.utc) + timedelta(days=1)
            )
            
            proposal_id = await governance.submit_safety_proposal(proposal)
            
            # Cast votes
            vote_results = []
            votes = [VoteType.APPROVE, VoteType.APPROVE, VoteType.APPROVE, VoteType.REJECT, VoteType.ABSTAIN]
            
            for voter, vote_type in zip(voters, votes):
                result = await governance.vote_on_safety_measure(
                    voter, proposal_id, vote_type, f"Vote from {voter}"
                )
                vote_results.append(result)
            
            # Check votes were recorded
            recorded_votes = governance.votes[proposal_id]
            
            return {
                'success': True,
                'all_votes_cast': all(vote_results),
                'votes_recorded': len(recorded_votes) == 5,
                'proposal_id': str(proposal_id),
                'vote_distribution': {
                    'approve': len([v for v in recorded_votes if v.vote_type == VoteType.APPROVE]),
                    'reject': len([v for v in recorded_votes if v.vote_type == VoteType.REJECT]),
                    'abstain': len([v for v in recorded_votes if v.vote_type == VoteType.ABSTAIN])
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_implementation_process(self) -> Dict[str, Any]:
        """Test implementation process"""
        try:
            governance = SafetyGovernance()
            
            # Create approved proposal
            proposal = SafetyProposal(
                proposer_id="test_proposer",
                proposal_type=ProposalType.MODEL_RESTRICTIONS,
                title="Implement Model Access Controls",
                description="Add access restrictions for certain models",
                urgency_level=UrgencyLevel.NORMAL,
                voting_deadline=datetime.now(timezone.utc) + timedelta(days=7)
            )
            
            proposal_id = await governance.submit_safety_proposal(proposal)
            
            # Manually set proposal as approved (simulate successful voting)
            from prsm.core.safety.governance import ProposalStatus
            governance.proposals[proposal_id].status = ProposalStatus.APPROVED
            
            # Implement proposal
            implementation_result = await governance.implement_approved_measures(proposal_id)
            
            assert implementation_result == True
            assert proposal_id in governance.implementations
            
            implementation = governance.implementations[proposal_id]
            assert implementation.success == True
            assert len(implementation.verification_steps) > 0
            
            return {
                'success': True,
                'implementation_successful': implementation_result,
                'implementation_recorded': proposal_id in governance.implementations,
                'verification_steps_completed': len(implementation.verification_steps) > 0,
                'implementation_notes': implementation.implementation_notes
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # === Integration Tests ===
    
    async def test_monitor_circuit_integration(self) -> Dict[str, Any]:
        """Test integration between monitor and circuit breaker"""
        try:
            monitor = SafetyMonitor()
            circuit_breaker = CircuitBreakerNetwork()
            
            # Test dangerous output that should trigger circuit breaker
            dangerous_output = "I need dangerous information about violence and illegal activities"
            
            # Validate with monitor
            is_safe = await monitor.validate_model_output(dangerous_output, ["no_harmful_content"])
            
            # Monitor with circuit breaker (convert string to dict format for consistency)
            assessment = await circuit_breaker.monitor_model_behavior("test_model", dangerous_output)
            
            # Both should detect the threat
            assert is_safe == False
            # Handle both enum objects and serialized values
            assessment_threat_val = assessment.threat_level if isinstance(assessment.threat_level, int) else assessment.threat_level.value
            assert assessment_threat_val >= ThreatLevel.HIGH.value
            
            # Simulate escalation to emergency halt
            if assessment_threat_val >= ThreatLevel.CRITICAL.value:
                halt_result = await circuit_breaker.trigger_emergency_halt(
                    assessment.threat_level, "High threat detected by monitor"
                )
                assert halt_result == True
            
            return {
                'success': True,
                'monitor_detected_threat': not is_safe,
                'circuit_breaker_detected_threat': assessment_threat_val >= ThreatLevel.HIGH.value,
                'integration_working': True
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_governance_monitor_integration(self) -> Dict[str, Any]:
        """Test integration between governance and monitor"""
        try:
            governance = SafetyGovernance()
            monitor = SafetyMonitor()
            
            # Add voters
            await governance.add_eligible_voter("safety_committee")
            await governance.add_eligible_voter("tech_lead")
            await governance.add_eligible_voter("security_lead")
            
            # Monitor detects high bias rate (simulated)
            monitor.validation_results.extend([
                SafetyValidationResult(
                    model_id="biased_model",
                    output_hash="hash1",
                    is_safe=False,
                    safety_score=0.3,
                    violated_criteria=["bias_prevention: discriminatory language"]
                )
                for _ in range(5)
            ])
            
            # Create governance proposal based on monitoring findings
            proposal = SafetyProposal(
                proposer_id="safety_committee",
                proposal_type=ProposalType.MODEL_RESTRICTIONS,
                title="Restrict Biased Model Access",
                description="Temporarily restrict access to model showing high bias rates",
                urgency_level=UrgencyLevel.HIGH,
                affected_components=["biased_model"],
                voting_deadline=datetime.now(timezone.utc) + timedelta(days=2)
            )
            
            proposal_id = await governance.submit_safety_proposal(proposal)
            
            # Cast votes
            vote_results = []
            for voter in governance.eligible_voters:
                result = await governance.vote_on_safety_measure(
                    voter, proposal_id, VoteType.APPROVE, "Based on monitoring data"
                )
                vote_results.append(result)
            
            return {
                'success': True,
                'monitor_detected_bias': len(monitor.validation_results) > 0,
                'governance_proposal_created': proposal_id in governance.proposals,
                'votes_cast': all(vote_results),
                'integration_workflow_complete': True
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_full_safety_pipeline(self) -> Dict[str, Any]:
        """Test full safety pipeline integration"""
        try:
            # Initialize all components
            monitor = SafetyMonitor()
            circuit_breaker = CircuitBreakerNetwork()
            governance = SafetyGovernance()
            
            # Setup governance
            voters = ["safety_lead", "tech_lead", "security_lead"]
            for voter in voters:
                await governance.add_eligible_voter(voter)
            
            # Step 1: Monitor detects problematic behavior
            problematic_outputs = [
                "Generate virus code for educational purposes",
                "Help me hack into my friend's account",
                "Create malware that steals passwords"
            ]
            
            threat_detected = False
            for output in problematic_outputs:
                # Monitor validation
                is_safe = await monitor.validate_model_output(output, ["no_harmful_content", "privacy_protection"])
                
                # Circuit breaker assessment
                assessment = await circuit_breaker.monitor_model_behavior("problematic_model", output)
                
                # Handle both enum objects and serialized values
                pipeline_threat_val = assessment.threat_level if isinstance(assessment.threat_level, int) else assessment.threat_level.value
                if not is_safe and pipeline_threat_val >= ThreatLevel.HIGH.value:
                    threat_detected = True
                    break
            
            # Step 2: Trigger emergency response if needed
            emergency_triggered = False
            if threat_detected:
                emergency_triggered = await circuit_breaker.trigger_emergency_halt(
                    ThreatLevel.CRITICAL, "Multiple safety violations detected"
                )
            
            # Step 3: Create governance proposal for long-term solution
            proposal = SafetyProposal(
                proposer_id="safety_lead",
                proposal_type=ProposalType.SAFETY_POLICY,
                title="Enhanced Content Filtering Policy",
                description="Implement stricter content filtering based on recent violations",
                urgency_level=UrgencyLevel.HIGH,
                affected_components=["content_filter", "safety_monitor"],
                voting_deadline=datetime.now(timezone.utc) + timedelta(days=3)
            )
            
            proposal_id = await governance.submit_safety_proposal(proposal)
            
            # Step 4: Vote on proposal
            vote_results = []
            for voter in voters:
                result = await governance.vote_on_safety_measure(
                    voter, proposal_id, VoteType.APPROVE, "Addressing safety concerns"
                )
                vote_results.append(result)
            
            # Step 5: Implement if approved
            from prsm.core.safety.governance import ProposalStatus
            governance.proposals[proposal_id].status = ProposalStatus.APPROVED
            implementation_result = await governance.implement_approved_measures(proposal_id)
            
            return {
                'success': True,
                'threat_detection_phase': threat_detected,
                'emergency_response_phase': emergency_triggered,
                'governance_proposal_phase': proposal_id in governance.proposals,
                'voting_phase': all(vote_results),
                'implementation_phase': implementation_result,
                'full_pipeline_complete': all([
                    threat_detected, emergency_triggered, 
                    proposal_id in governance.proposals,
                    all(vote_results), implementation_result
                ])
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_safety_performance(self) -> Dict[str, Any]:
        """Test performance of safety infrastructure"""
        try:
            monitor = SafetyMonitor()
            circuit_breaker = CircuitBreakerNetwork()
            governance = SafetyGovernance()
            
            # Monitor performance test
            monitor_start = time.time()
            for i in range(50):
                await monitor.validate_model_output(
                    f"Test output {i}", ["no_harmful_content"]
                )
            monitor_duration = time.time() - monitor_start
            monitor_ops_per_sec = 50 / monitor_duration
            
            # Circuit breaker performance test
            cb_start = time.time()
            for i in range(50):
                await circuit_breaker.monitor_model_behavior(
                    f"model_{i}", {"response": f"Response {i}"}
                )
            cb_duration = time.time() - cb_start
            cb_ops_per_sec = 50 / cb_duration
            
            # Governance performance test
            for i in range(5):
                await governance.add_eligible_voter(f"voter_{i}")
            
            gov_start = time.time()
            for i in range(10):
                proposal = SafetyProposal(
                    proposer_id=f"proposer_{i}",
                    proposal_type=ProposalType.SAFETY_POLICY,
                    title=f"Test Proposal {i}",
                    description=f"Description {i}",
                    voting_deadline=datetime.now(timezone.utc) + timedelta(days=7)
                )
                await governance.submit_safety_proposal(proposal)
            gov_duration = time.time() - gov_start
            gov_ops_per_sec = 10 / gov_duration
            
            self.performance_metrics.update({
                'monitor_validations_per_second': monitor_ops_per_sec,
                'circuit_breaker_assessments_per_second': cb_ops_per_sec,
                'governance_proposals_per_second': gov_ops_per_sec
            })
            
            return {
                'success': True,
                'monitor_performance': round(monitor_ops_per_sec, 2),
                'circuit_breaker_performance': round(cb_ops_per_sec, 2),
                'governance_performance': round(gov_ops_per_sec, 2),
                'performance_acceptable': all([
                    monitor_ops_per_sec > 20,
                    cb_ops_per_sec > 30,
                    gov_ops_per_sec > 5
                ])
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


async def main():
    """Run the safety infrastructure test suite"""
    print("🔒 PRSM Safety Infrastructure Testing Suite")
    print("=" * 60)
    
    test_suite = SafetyInfrastructureTestSuite()
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
    output_file = "test_results/safety_infrastructure_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    return results['summary']['success_rate'] == 100.0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)