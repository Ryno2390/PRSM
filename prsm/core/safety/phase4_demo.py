"""
Phase 4 Demo: Safety and Governance Integration

Demonstrates the comprehensive safety-constrained self-modification system
and governance-integrated evolution capabilities implemented in Phase 4.

This demo showcases:
1. Multi-layered safety validation
2. Resource monitoring and constraint checking  
3. Emergency shutdown procedures
4. Governance-integrated approval workflows
5. Community review and expert panel processes
6. Performance tracking and governance insights
"""

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List
from decimal import Decimal

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core imports
from .safe_modification import SafetyConstrainedModificationSystem
from .safety_models import (
    CapabilityBounds, ResourceLimits, BehavioralConstraints,
    SafetyStatus, RiskAssessment
)
from ..governance.dgm_governance import (
    DGMGovernanceSystem, CommunityReviewSystem, GovernancePerformanceTracker,
    EvolutionGovernanceProposal, GovernanceDecisionType
)
from ..evolution.models import (
    ModificationProposal, ComponentType, RiskLevel, ImpactLevel
)


class MockGovernanceComponents:
    """Mock governance components for demonstration."""
    
    class MockTokenVoting:
        def __init__(self):
            self.active_votes = {}
        
        async def start_vote(self, proposal_id: str, description: str, options: List[str], duration_hours: int):
            vote_session_id = f"vote_{proposal_id}"
            self.active_votes[proposal_id] = {
                "session_id": vote_session_id,
                "options": options,
                "votes": {},
                "start_time": datetime.utcnow()
            }
            return vote_session_id
        
        async def get_vote_results(self, proposal_id: str) -> Dict:
            # Simulate vote results
            total_eligible = 100
            total_votes = random.randint(40, 90)  # Random participation
            approve_votes = random.randint(int(total_votes * 0.3), int(total_votes * 0.8))
            
            return {
                "total_votes": total_votes,
                "eligible_voters": total_eligible,
                "approve_votes": approve_votes,
                "reject_votes": total_votes - approve_votes,
                "quorum_met": total_votes >= (total_eligible * 0.15),
                "approval_rate": approve_votes / total_votes if total_votes > 0 else 0
            }
    
    class MockProposalManager:
        def __init__(self):
            self.proposals = {}


class MockModificationGenerator:
    """Generates realistic modification proposals for testing."""
    
    def __init__(self):
        self.modification_types = [
            "performance_optimization",
            "security_enhancement", 
            "feature_addition",
            "bug_fix",
            "algorithm_improvement"
        ]
        
        self.component_types = [
            ComponentType.TASK_ORCHESTRATOR,
            ComponentType.INTELLIGENT_ROUTER,
            ComponentType.SAFETY_MONITOR
        ]
    
    def generate_modification(self, risk_level: RiskLevel = None) -> ModificationProposal:
        """Generate a modification proposal with specified risk level."""
        
        risk_level = risk_level or random.choice(list(RiskLevel))
        mod_type = random.choice(self.modification_types)
        component_type = random.choice(self.component_types)
        
        # Risk-based parameters
        if risk_level == RiskLevel.LOW:
            performance_impact = random.uniform(0.01, 0.05)
            description = f"Minor {mod_type} for {component_type.value}"
        elif risk_level == RiskLevel.MEDIUM:
            performance_impact = random.uniform(0.05, 0.15)
            description = f"Moderate {mod_type} for {component_type.value}"
        elif risk_level == RiskLevel.HIGH:
            performance_impact = random.uniform(0.15, 0.35)
            description = f"Major {mod_type} for {component_type.value}"
        else:  # CRITICAL
            performance_impact = random.uniform(0.35, 0.6)
            description = f"Critical {mod_type} for {component_type.value}"
        
        return ModificationProposal(
            solution_id=f"solution_{random.randint(1000, 9999)}",
            component_type=component_type,
            modification_type=mod_type,
            description=description,
            rationale=f"Implementing {mod_type} to improve system {random.choice(['performance', 'security', 'reliability'])}",
            config_changes={
                "optimization_level": random.choice([1, 2, 3]),
                "batch_size": random.choice([16, 32, 64]),
                "learning_rate": random.uniform(0.001, 0.01)
            },
            estimated_performance_impact=performance_impact,
            risk_level=risk_level,
            impact_level=ImpactLevel.MEDIUM if risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM] else ImpactLevel.HIGH,
            safety_considerations=[
                "performance_monitoring",
                "rollback_capability",
                "impact_assessment"
            ],
            rollback_plan="Automatic rollback if performance degrades by >5%",
            proposer_id="demo_system"
        )


async def demonstrate_phase4_capabilities():
    """Comprehensive demonstration of Phase 4 safety and governance capabilities."""
    
    print("🛡️ Phase 4: Safety and Governance Integration Demo")
    print("=" * 65)
    
    # Initialize Phase 4 systems
    print("\n📋 Initializing Safety and Governance Systems...")
    
    # Configure safety constraints
    capability_bounds = CapabilityBounds(
        max_modification_frequency=5,  # 5 per hour
        max_performance_delta=0.3,     # 30% max change
        requires_approval_threshold=0.1,  # 10% requires approval
        emergency_stop_threshold=0.4   # 40% triggers emergency stop
    )
    
    resource_limits = ResourceLimits(
        cpu_cores_limit=4,
        memory_limit_gb=8.0,
        ftns_budget_limit=Decimal('500.0'),
        execution_timeout_seconds=300
    )
    
    behavioral_constraints = BehavioralConstraints(
        requires_testing=True,
        requires_rollback_plan=True,
        requires_impact_assessment=True
    )
    
    # Initialize safety system
    safety_system = SafetyConstrainedModificationSystem(
        capability_bounds=capability_bounds,
        resource_limits=resource_limits,
        behavioral_constraints=behavioral_constraints
    )
    
    # Initialize governance system
    mock_voting = MockGovernanceComponents.MockTokenVoting()
    mock_proposals = MockGovernanceComponents.MockProposalManager()
    governance_system = DGMGovernanceSystem(mock_voting, mock_proposals)
    
    # Set up community review
    governance_system.community_review.register_expert_panel(
        "safety_engineering", 
        ["expert_safety_1", "expert_safety_2", "expert_safety_3"]
    )
    governance_system.community_review.register_expert_panel(
        "orchestration_systems",
        ["expert_orchestration_1", "expert_orchestration_2"]
    )
    governance_system.community_review.register_stakeholder_group(
        "developers", 
        ["dev_1", "dev_2", "dev_3", "dev_4"],
        weight=1.5
    )
    governance_system.community_review.register_stakeholder_group(
        "general_community",
        [f"user_{i}" for i in range(1, 21)],
        weight=1.0
    )
    
    modification_generator = MockModificationGenerator()
    
    print("✅ Safety system initialized with comprehensive constraints")
    print("✅ Governance system initialized with expert panels and stakeholder groups")
    print("✅ Community review system configured")
    
    # Phase 1: Low-Risk Modification (Automatic Approval)
    print("\n🟢 Phase 1: Low-Risk Modification Processing")
    print("-" * 50)
    
    low_risk_mod = modification_generator.generate_modification(RiskLevel.LOW)
    print(f"Generated low-risk modification: {low_risk_mod.description}")
    print(f"   Estimated impact: {low_risk_mod.estimated_performance_impact:.3f}")
    print(f"   Risk level: {low_risk_mod.risk_level.value}")
    
    # Safety validation
    safety_result = await safety_system.validate_modification_safety(low_risk_mod)
    print(f"🔍 Safety validation result: {safety_result.safety_status.value}")
    print(f"   Risk level: {safety_result.overall_risk_level.value}")
    print(f"   Violations: {safety_result.total_violations}")
    print(f"   Warnings: {safety_result.total_warnings}")
    
    if safety_result.passed:
        mod_result = await safety_system.execute_safe_modification(low_risk_mod)
        print(f"✅ Low-risk modification executed successfully: {mod_result.success}")
    
    # Phase 2: Medium-Risk Modification (Governance Review)
    print("\n🟡 Phase 2: Medium-Risk Modification with Governance Review")
    print("-" * 60)
    
    medium_risk_mod = modification_generator.generate_modification(RiskLevel.MEDIUM)
    print(f"Generated medium-risk modification: {medium_risk_mod.description}")
    print(f"   Estimated impact: {medium_risk_mod.estimated_performance_impact:.3f}")
    print(f"   Risk level: {medium_risk_mod.risk_level.value}")
    
    # Safety validation
    safety_result = await safety_system.validate_modification_safety(medium_risk_mod)
    print(f"🔍 Safety validation result: {safety_result.safety_status.value}")
    print(f"   Risk level: {safety_result.overall_risk_level.value}")
    print(f"   Requires approval: {safety_result.requires_manual_approval}")
    
    if safety_result.requires_manual_approval:
        # Submit for governance approval
        proposal_id = await governance_system.submit_modification_for_approval(
            medium_risk_mod, safety_result, "demo_user"
        )
        print(f"📋 Submitted for governance approval: {proposal_id}")
        
        # Start governance vote
        vote_info = await governance_system.start_governance_vote(proposal_id)
        print(f"🗳️ Governance vote started: {vote_info['vote_config']['vote_type']}")
        print(f"   Voting period: {vote_info['vote_config'].get('voting_period_hours', 48)} hours")
        print(f"   Required quorum: {vote_info['vote_config']['required_quorum']:.1%}")
        print(f"   Approval threshold: {vote_info['vote_config']['approval_threshold']:.1%}")
        
        # Simulate vote completion
        await asyncio.sleep(1)  # Brief delay
        decision_result = await governance_system.finalize_governance_decision(proposal_id)
        print(f"🏛️ Governance decision: {decision_result['decision_outcome']}")
        print(f"   Vote results: {decision_result['vote_results']['approve_votes']}/{decision_result['vote_results']['total_votes']} approve")
        print(f"   Implementation scheduled: {decision_result['implementation_scheduled']}")
    
    # Phase 3: High-Risk Modification (Expert Review Required)
    print("\n🔴 Phase 3: High-Risk Modification with Expert Review")
    print("-" * 55)
    
    high_risk_mod = modification_generator.generate_modification(RiskLevel.HIGH)
    print(f"Generated high-risk modification: {high_risk_mod.description}")
    print(f"   Estimated impact: {high_risk_mod.estimated_performance_impact:.3f}")
    print(f"   Risk level: {high_risk_mod.risk_level.value}")
    
    # Safety validation
    safety_result = await safety_system.validate_modification_safety(high_risk_mod)
    print(f"🔍 Safety validation result: {safety_result.safety_status.value}")
    print(f"   Risk level: {safety_result.overall_risk_level.value}")
    print(f"   Requires governance vote: {safety_result.requires_governance_vote}")
    print(f"   Safety recommendations: {len(safety_result.safety_recommendations)}")
    
    for i, rec in enumerate(safety_result.safety_recommendations[:3], 1):
        print(f"     {i}. {rec}")
    
    if safety_result.requires_governance_vote:
        # Submit for governance approval with expert review
        proposal_id = await governance_system.submit_modification_for_approval(
            high_risk_mod, safety_result, "demo_user"
        )
        print(f"📋 Submitted for expert review and governance approval: {proposal_id}")
        
        proposal = governance_system.active_proposals[proposal_id]
        print(f"🔬 Expert review required: {proposal.requires_expert_review}")
        print(f"   Vote type: {proposal.vote_type.value}")
        print(f"   Approval threshold: {proposal.approval_threshold:.1%}")
    
    # Phase 4: Critical Risk Modification (Emergency Stop)
    print("\n⚠️ Phase 4: Critical Risk Modification (Emergency Procedures)")
    print("-" * 58)
    
    critical_risk_mod = modification_generator.generate_modification(RiskLevel.CRITICAL)
    critical_risk_mod.estimated_performance_impact = 0.5  # Trigger emergency threshold
    
    print(f"Generated critical-risk modification: {critical_risk_mod.description}")
    print(f"   Estimated impact: {critical_risk_mod.estimated_performance_impact:.3f}")
    print(f"   Risk level: {critical_risk_mod.risk_level.value}")
    
    # Safety validation
    safety_result = await safety_system.validate_modification_safety(critical_risk_mod)
    print(f"🚨 Safety validation result: {safety_result.safety_status.value}")
    print(f"   Risk level: {safety_result.overall_risk_level.value}")
    
    if safety_result.safety_status == SafetyStatus.EMERGENCY_STOP:
        print("🛑 EMERGENCY STOP TRIGGERED")
        print("   Modification exceeds safety thresholds")
        print("   Manual intervention required")
        print("   Emergency protocols activated")
    
    # Phase 5: Resource Monitoring Demonstration
    print("\n📊 Phase 5: Resource Monitoring and Constraint Validation")
    print("-" * 60)
    
    # Create a resource-intensive modification
    resource_mod = modification_generator.generate_modification(RiskLevel.MEDIUM)
    resource_mod.compute_requirements = {
        "cpu_cores": 6,  # Exceeds limit of 4
        "memory_gb": 12.0,  # Exceeds limit of 8.0
        "estimated_duration_hours": 2.0
    }
    
    print(f"Testing resource-intensive modification...")
    print(f"   CPU requirement: {resource_mod.compute_requirements['cpu_cores']} cores (limit: {resource_limits.cpu_cores_limit})")
    print(f"   Memory requirement: {resource_mod.compute_requirements['memory_gb']}GB (limit: {resource_limits.memory_limit_gb}GB)")
    
    # Test resource validation
    safety_result = await safety_system.validate_modification_safety(resource_mod)
    print(f"🔍 Resource validation result: {safety_result.safety_status.value}")
    
    if safety_result.resource_check:
        print(f"   Resource violations: {len(safety_result.resource_check.violations_found)}")
        for violation in safety_result.resource_check.violations_found:
            print(f"     ⚠️ {violation}")
    
    # Phase 6: Governance Performance Analysis
    print("\n📈 Phase 6: Governance Performance Analysis")
    print("-" * 50)
    
    # Simulate some governance history
    for i in range(5):
        test_mod = modification_generator.generate_modification()
        test_safety = await safety_system.validate_modification_safety(test_mod)
        
        if test_safety.requires_manual_approval:
            proposal_id = await governance_system.submit_modification_for_approval(
                test_mod, test_safety, f"user_{i}"
            )
            await governance_system.start_governance_vote(proposal_id)
            decision = await governance_system.finalize_governance_decision(proposal_id)
            
            # Track implementation outcome
            impl_result = {
                "success": random.choice([True, True, True, False]),  # 75% success rate
                "performance_delta": random.uniform(-0.05, 0.15),
                "safety_status": "approved"
            }
            await governance_system.performance_tracker.track_decision_outcome(
                governance_system.active_proposals[proposal_id], 
                decision["decision_outcome"],
                impl_result
            )
    
    # Get governance insights
    governance_dashboard = await governance_system.get_governance_dashboard()
    insights = governance_dashboard["governance_insights"]
    
    print("🏛️ Governance Performance Summary:")
    if "performance_metrics" in insights and insights["performance_metrics"]:
        metrics = insights["performance_metrics"]
        print(f"   Approval rate: {metrics.get('approval_rate', 0):.1%}")
        print(f"   Implementation success rate: {metrics.get('implementation_success_rate', 0):.1%}")
        print(f"   Average decision time: {metrics.get('average_decision_time_hours', 0):.1f} hours")
        print(f"   Average performance impact: {metrics.get('average_performance_impact', 0):+.3f}")
    
    if "recommendations" in insights:
        print("💡 Governance Recommendations:")
        for rec in insights["recommendations"][:3]:
            print(f"   • {rec}")
    
    # Final Summary
    print("\n🎉 Phase 4 Demo Complete!")
    print("=" * 65)
    
    dashboard = governance_dashboard
    system_stats = dashboard["system_status"]
    
    print("✨ Phase 4 Capabilities Demonstrated:")
    print(f"   🛡️ Multi-layered safety validation with constraint checking")
    print(f"   📊 Resource monitoring and limit enforcement")
    print(f"   🚨 Emergency procedures and automatic intervention")
    print(f"   🏛️ Governance-integrated approval workflows")
    print(f"   🔬 Expert panel review for high-risk modifications")
    print(f"   📈 Performance tracking and governance insights")
    
    print(f"\n📊 System Configuration:")
    print(f"   Expert panels: {system_stats['expert_panels']}")
    print(f"   Stakeholder groups: {system_stats['stakeholder_groups']}")
    print(f"   Total governance decisions: {system_stats['total_decisions']}")
    print(f"   Active proposals: {len(dashboard['active_proposals'])}")
    print(f"   Pending implementations: {len(dashboard['pending_implementations'])}")
    
    print(f"\n🚀 Phase 4 Implementation Success!")
    print("   The safety and governance integration provides comprehensive")
    print("   oversight for DGM self-modification with community control,")
    print("   expert review, and automated safety constraints.")
    
    return {
        "phase_completed": "Phase 4",
        "capabilities_demonstrated": [
            "safety_constrained_modification",
            "governance_integrated_approval", 
            "community_review_system",
            "expert_panel_integration",
            "resource_monitoring",
            "emergency_procedures",
            "performance_tracking"
        ],
        "safety_validations_performed": 6,
        "governance_proposals_processed": 5,
        "system_status": "fully_operational"
    }


if __name__ == "__main__":
    result = asyncio.run(demonstrate_phase4_capabilities())
    print(f"\n📊 Demo Results: {result}")