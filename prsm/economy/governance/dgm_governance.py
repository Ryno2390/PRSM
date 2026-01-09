"""
DGM-Enhanced Governance System

Integrates Darwin Gödel Machine evolution capabilities with PRSM's governance
system, enabling community oversight of self-modifications and evolutionary
governance policy optimization.

Implements Phase 4.2 of the DGM roadmap: Governance-Integrated Evolution.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from decimal import Decimal
import uuid

from .voting import TokenWeightedVoting
from .proposals import ProposalManager
from prsm.compute.evolution.models import (
    ModificationProposal, EvaluationResult, ComponentType, RiskLevel
)
from prsm.core.safety.safety_models import (
    SafetyValidationResult, SafetyStatus
)

logger = logging.getLogger(__name__)


class GovernanceDecisionType(str, Enum):
    """Types of governance decisions for DGM evolution."""
    MODIFICATION_APPROVAL = "MODIFICATION_APPROVAL"
    SAFETY_POLICY_UPDATE = "SAFETY_POLICY_UPDATE"
    EVOLUTION_PARAMETER_CHANGE = "EVOLUTION_PARAMETER_CHANGE"
    EMERGENCY_INTERVENTION = "EMERGENCY_INTERVENTION"
    ARCHIVE_MANAGEMENT = "ARCHIVE_MANAGEMENT"
    PERFORMANCE_THRESHOLD_UPDATE = "PERFORMANCE_THRESHOLD_UPDATE"


class GovernanceVoteType(str, Enum):
    """Types of voting mechanisms for different decisions."""
    SIMPLE_MAJORITY = "SIMPLE_MAJORITY"
    SUPERMAJORITY = "SUPERMAJORITY"
    EXPERT_PANEL = "EXPERT_PANEL"
    STAKEHOLDER_WEIGHTED = "STAKEHOLDER_WEIGHTED"
    EMERGENCY_COMMITTEE = "EMERGENCY_COMMITTEE"


class EvolutionGovernanceProposal:
    """Governance proposal specifically for DGM evolution decisions."""
    
    def __init__(
        self,
        proposal_id: str,
        decision_type: GovernanceDecisionType,
        title: str,
        description: str,
        proposer_id: str,
        modification_proposal: Optional[ModificationProposal] = None,
        safety_validation: Optional[SafetyValidationResult] = None,
        impact_assessment: Optional[Dict[str, Any]] = None
    ):
        self.proposal_id = proposal_id
        self.decision_type = decision_type
        self.title = title
        self.description = description
        self.proposer_id = proposer_id
        self.modification_proposal = modification_proposal
        self.safety_validation = safety_validation
        self.impact_assessment = impact_assessment or {}
        
        # Voting configuration
        self.vote_type = self._determine_vote_type()
        self.voting_period_hours = self._determine_voting_period()
        self.required_quorum = self._determine_required_quorum()
        self.approval_threshold = self._determine_approval_threshold()
        
        # Metadata
        self.created_at = datetime.utcnow()
        self.voting_start_time: Optional[datetime] = None
        self.voting_end_time: Optional[datetime] = None
        self.status = "draft"  # draft, voting, approved, rejected, expired
        
        # Expert review
        self.requires_expert_review = self._requires_expert_review()
        self.expert_reviews: List[Dict[str, Any]] = []
        
        # Community feedback
        self.community_comments: List[Dict[str, Any]] = []
        self.stakeholder_notifications_sent = False
    
    def _determine_vote_type(self) -> GovernanceVoteType:
        """Determine appropriate voting mechanism based on decision type and risk."""
        
        if self.decision_type == GovernanceDecisionType.EMERGENCY_INTERVENTION:
            return GovernanceVoteType.EMERGENCY_COMMITTEE
        
        if self.modification_proposal and self.safety_validation:
            if self.safety_validation.overall_risk_level == RiskLevel.CRITICAL:
                return GovernanceVoteType.EXPERT_PANEL
            elif self.safety_validation.overall_risk_level == RiskLevel.HIGH:
                return GovernanceVoteType.SUPERMAJORITY
            else:
                return GovernanceVoteType.STAKEHOLDER_WEIGHTED
        
        if self.decision_type in [
            GovernanceDecisionType.SAFETY_POLICY_UPDATE,
            GovernanceDecisionType.EVOLUTION_PARAMETER_CHANGE
        ]:
            return GovernanceVoteType.SUPERMAJORITY
        
        return GovernanceVoteType.SIMPLE_MAJORITY
    
    def _determine_voting_period(self) -> int:
        """Determine voting period based on decision type and urgency."""
        
        if self.decision_type == GovernanceDecisionType.EMERGENCY_INTERVENTION:
            return 4  # 4 hours for emergencies
        
        if self.modification_proposal and self.safety_validation:
            if self.safety_validation.overall_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                return 72  # 3 days for high-risk modifications
            else:
                return 48  # 2 days for normal modifications
        
        return 168  # 1 week for policy changes
    
    def _determine_required_quorum(self) -> float:
        """Determine required quorum percentage."""
        
        if self.vote_type == GovernanceVoteType.EMERGENCY_COMMITTEE:
            return 0.6  # 60% of emergency committee
        elif self.vote_type == GovernanceVoteType.EXPERT_PANEL:
            return 0.7  # 70% of expert panel
        elif self.vote_type == GovernanceVoteType.SUPERMAJORITY:
            return 0.3  # 30% of all stakeholders
        else:
            return 0.15  # 15% of all stakeholders
    
    def _determine_approval_threshold(self) -> float:
        """Determine approval threshold percentage."""
        
        if self.vote_type == GovernanceVoteType.SUPERMAJORITY:
            return 0.67  # 67% approval required
        elif self.vote_type == GovernanceVoteType.EXPERT_PANEL:
            return 0.75  # 75% expert approval
        elif self.vote_type == GovernanceVoteType.EMERGENCY_COMMITTEE:
            return 0.6   # 60% emergency committee approval
        else:
            return 0.51  # Simple majority
    
    def _requires_expert_review(self) -> bool:
        """Determine if expert review is required."""
        
        if self.decision_type == GovernanceDecisionType.SAFETY_POLICY_UPDATE:
            return True
        
        if self.modification_proposal and self.safety_validation:
            return self.safety_validation.overall_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        
        return False


class CommunityReviewSystem:
    """Manages community review process for DGM evolution decisions."""
    
    def __init__(self):
        self.expert_panels = {}
        self.stakeholder_groups = {}
        self.notification_handlers = []
        self.review_history = []
    
    def register_expert_panel(self, domain: str, expert_ids: List[str]):
        """Register expert panel for specific domain."""
        self.expert_panels[domain] = {
            "expert_ids": expert_ids,
            "created_at": datetime.utcnow(),
            "active": True
        }
        logger.info(f"Registered expert panel for {domain} with {len(expert_ids)} experts")
    
    def register_stakeholder_group(self, group_name: str, stakeholder_ids: List[str], weight: float = 1.0):
        """Register stakeholder group with voting weight."""
        self.stakeholder_groups[group_name] = {
            "stakeholder_ids": stakeholder_ids,
            "weight": weight,
            "created_at": datetime.utcnow(),
            "active": True
        }
        logger.info(f"Registered stakeholder group {group_name} with {len(stakeholder_ids)} members")
    
    async def initiate_community_review(self, proposal: EvolutionGovernanceProposal) -> Dict[str, Any]:
        """Initiate community review process for a proposal."""
        
        logger.info(f"Initiating community review for proposal {proposal.proposal_id}")
        
        # Send stakeholder notifications
        if not proposal.stakeholder_notifications_sent:
            await self._send_stakeholder_notifications(proposal)
            proposal.stakeholder_notifications_sent = True
        
        # Request expert reviews if required
        expert_review_requests = []
        if proposal.requires_expert_review:
            expert_review_requests = await self._request_expert_reviews(proposal)
        
        # Set up community comment period
        comment_period_start = datetime.utcnow()
        comment_period_end = comment_period_start + timedelta(hours=proposal.voting_period_hours // 2)
        
        review_session = {
            "proposal_id": proposal.proposal_id,
            "review_start": comment_period_start,
            "review_end": comment_period_end,
            "expert_reviews_requested": len(expert_review_requests),
            "stakeholder_notifications_sent": len(self.stakeholder_groups),
            "public_comment_enabled": True
        }
        
        self.review_history.append(review_session)
        
        return review_session
    
    async def _send_stakeholder_notifications(self, proposal: EvolutionGovernanceProposal):
        """Send notifications to relevant stakeholders."""
        
        # Determine relevant stakeholder groups
        relevant_groups = self._identify_relevant_stakeholders(proposal)
        
        notification_count = 0
        for group_name in relevant_groups:
            if group_name in self.stakeholder_groups:
                group = self.stakeholder_groups[group_name]
                for stakeholder_id in group["stakeholder_ids"]:
                    await self._send_notification(
                        stakeholder_id,
                        proposal,
                        f"New governance proposal requires your attention: {proposal.title}"
                    )
                    notification_count += 1
        
        logger.info(f"Sent {notification_count} stakeholder notifications for proposal {proposal.proposal_id}")
    
    async def _request_expert_reviews(self, proposal: EvolutionGovernanceProposal) -> List[str]:
        """Request expert reviews for the proposal."""
        
        # Determine relevant expert domains
        relevant_domains = self._identify_relevant_expert_domains(proposal)
        
        review_requests = []
        for domain in relevant_domains:
            if domain in self.expert_panels:
                panel = self.expert_panels[domain]
                for expert_id in panel["expert_ids"]:
                    request_id = str(uuid.uuid4())
                    await self._send_expert_review_request(expert_id, proposal, request_id)
                    review_requests.append(request_id)
        
        logger.info(f"Requested {len(review_requests)} expert reviews for proposal {proposal.proposal_id}")
        return review_requests
    
    def _identify_relevant_stakeholders(self, proposal: EvolutionGovernanceProposal) -> List[str]:
        """Identify stakeholder groups relevant to the proposal."""
        
        relevant_groups = ["general_community"]  # Always include general community
        
        if proposal.decision_type == GovernanceDecisionType.MODIFICATION_APPROVAL:
            if proposal.modification_proposal:
                if proposal.modification_proposal.component_type == ComponentType.TASK_ORCHESTRATOR:
                    relevant_groups.extend(["orchestrator_users", "developers"])
                elif proposal.modification_proposal.component_type == ComponentType.INTELLIGENT_ROUTER:
                    relevant_groups.extend(["router_users", "infrastructure_operators"])
        
        elif proposal.decision_type == GovernanceDecisionType.SAFETY_POLICY_UPDATE:
            relevant_groups.extend(["security_experts", "system_administrators", "developers"])
        
        elif proposal.decision_type == GovernanceDecisionType.EVOLUTION_PARAMETER_CHANGE:
            relevant_groups.extend(["researchers", "developers", "power_users"])
        
        return relevant_groups
    
    def _identify_relevant_expert_domains(self, proposal: EvolutionGovernanceProposal) -> List[str]:
        """Identify expert domains relevant to the proposal."""
        
        domains = []
        
        if proposal.decision_type == GovernanceDecisionType.MODIFICATION_APPROVAL:
            if proposal.safety_validation:
                if proposal.safety_validation.overall_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    domains.extend(["safety_engineering", "security"])
                
                if proposal.modification_proposal:
                    if proposal.modification_proposal.component_type == ComponentType.TASK_ORCHESTRATOR:
                        domains.append("orchestration_systems")
                    elif proposal.modification_proposal.component_type == ComponentType.INTELLIGENT_ROUTER:
                        domains.append("routing_systems")
        
        elif proposal.decision_type == GovernanceDecisionType.SAFETY_POLICY_UPDATE:
            domains.extend(["safety_engineering", "security", "governance"])
        
        elif proposal.decision_type == GovernanceDecisionType.EVOLUTION_PARAMETER_CHANGE:
            domains.extend(["machine_learning", "evolutionary_systems", "performance_optimization"])
        
        return domains
    
    async def _send_notification(self, stakeholder_id: str, proposal: EvolutionGovernanceProposal, message: str):
        """Send notification to a stakeholder."""
        # This would integrate with actual notification system
        logger.debug(f"Notification sent to {stakeholder_id}: {message}")
    
    async def _send_expert_review_request(self, expert_id: str, proposal: EvolutionGovernanceProposal, request_id: str):
        """Send expert review request."""
        # This would integrate with actual expert review system
        logger.debug(f"Expert review request {request_id} sent to {expert_id}")


class GovernancePerformanceTracker:
    """Tracks and analyzes governance performance and decision quality."""
    
    def __init__(self):
        self.decision_history = []
        self.performance_metrics = {}
        self.efficiency_tracker = {}
    
    async def track_decision_outcome(
        self,
        proposal: EvolutionGovernanceProposal,
        decision_outcome: str,
        implementation_result: Optional[Dict[str, Any]] = None
    ):
        """Track the outcome and performance of a governance decision."""
        
        decision_record = {
            "proposal_id": proposal.proposal_id,
            "decision_type": proposal.decision_type.value,
            "decision_outcome": decision_outcome,  # approved, rejected, expired
            "vote_type": proposal.vote_type.value,
            "voting_duration_hours": (
                (proposal.voting_end_time - proposal.voting_start_time).total_seconds() / 3600
                if proposal.voting_end_time and proposal.voting_start_time else None
            ),
            "implementation_result": implementation_result,
            "timestamp": datetime.utcnow()
        }
        
        if implementation_result:
            # Track implementation success
            decision_record["implementation_success"] = implementation_result.get("success", False)
            decision_record["performance_impact"] = implementation_result.get("performance_delta", 0.0)
            decision_record["safety_outcome"] = implementation_result.get("safety_status", "unknown")
        
        self.decision_history.append(decision_record)
        await self._update_performance_metrics()
        
        logger.info(f"Tracked governance decision outcome for {proposal.proposal_id}: {decision_outcome}")
    
    async def _update_performance_metrics(self):
        """Update governance performance metrics based on decision history."""
        
        if not self.decision_history:
            return
        
        recent_decisions = [
            d for d in self.decision_history
            if (datetime.utcnow() - d["timestamp"]).days <= 30
        ]
        
        # Decision efficiency metrics
        approval_rate = len([d for d in recent_decisions if d["decision_outcome"] == "approved"]) / len(recent_decisions)
        
        # Implementation success rate
        implemented_decisions = [d for d in recent_decisions if d.get("implementation_result")]
        implementation_success_rate = (
            len([d for d in implemented_decisions if d.get("implementation_success", False)]) / len(implemented_decisions)
            if implemented_decisions else 0
        )
        
        # Average decision time
        timed_decisions = [d for d in recent_decisions if d.get("voting_duration_hours")]
        avg_decision_time = (
            sum(d["voting_duration_hours"] for d in timed_decisions) / len(timed_decisions)
            if timed_decisions else 0
        )
        
        # Performance impact analysis
        impact_decisions = [d for d in recent_decisions if d.get("performance_impact") is not None]
        avg_performance_impact = (
            sum(d["performance_impact"] for d in impact_decisions) / len(impact_decisions)
            if impact_decisions else 0
        )
        
        self.performance_metrics = {
            "approval_rate": approval_rate,
            "implementation_success_rate": implementation_success_rate,
            "average_decision_time_hours": avg_decision_time,
            "average_performance_impact": avg_performance_impact,
            "total_decisions": len(recent_decisions),
            "last_updated": datetime.utcnow()
        }
    
    def get_governance_insights(self) -> Dict[str, Any]:
        """Get comprehensive governance performance insights."""
        
        insights = {
            "performance_metrics": self.performance_metrics,
            "decision_trends": self._analyze_decision_trends(),
            "efficiency_analysis": self._analyze_governance_efficiency(),
            "recommendations": self._generate_governance_recommendations()
        }
        
        return insights
    
    def _analyze_decision_trends(self) -> Dict[str, Any]:
        """Analyze trends in governance decisions."""
        
        if len(self.decision_history) < 5:
            return {"insufficient_data": True}
        
        # Analyze decision types over time
        decision_type_counts = {}
        for decision in self.decision_history[-20:]:  # Last 20 decisions
            decision_type = decision["decision_type"]
            decision_type_counts[decision_type] = decision_type_counts.get(decision_type, 0) + 1
        
        # Analyze approval trends
        recent_approvals = [
            d["decision_outcome"] == "approved" 
            for d in self.decision_history[-10:]
        ]
        approval_trend = "increasing" if sum(recent_approvals[-5:]) > sum(recent_approvals[:5]) else "stable"
        
        return {
            "decision_type_distribution": decision_type_counts,
            "approval_trend": approval_trend,
            "recent_approval_rate": sum(recent_approvals) / len(recent_approvals)
        }
    
    def _analyze_governance_efficiency(self) -> Dict[str, Any]:
        """Analyze governance system efficiency."""
        
        if not self.performance_metrics:
            return {"no_metrics_available": True}
        
        metrics = self.performance_metrics
        efficiency_score = (
            metrics.get("implementation_success_rate", 0) * 0.4 +
            (1 - min(metrics.get("average_decision_time_hours", 168) / 168, 1)) * 0.3 +
            metrics.get("approval_rate", 0) * 0.3
        )
        
        return {
            "overall_efficiency_score": efficiency_score,
            "decision_speed": "fast" if metrics.get("average_decision_time_hours", 168) < 48 else "slow",
            "implementation_reliability": "high" if metrics.get("implementation_success_rate", 0) > 0.8 else "medium"
        }
    
    def _generate_governance_recommendations(self) -> List[str]:
        """Generate recommendations for governance improvement."""
        
        recommendations = []
        
        if not self.performance_metrics:
            return ["Insufficient data for recommendations"]
        
        metrics = self.performance_metrics
        
        if metrics.get("approval_rate", 0) < 0.3:
            recommendations.append("Low approval rate - consider improving proposal quality or review process")
        
        if metrics.get("implementation_success_rate", 0) < 0.7:
            recommendations.append("Low implementation success - strengthen technical review and testing")
        
        if metrics.get("average_decision_time_hours", 0) > 120:
            recommendations.append("Long decision times - consider streamlining review process")
        
        if metrics.get("average_performance_impact", 0) < 0:
            recommendations.append("Negative average performance impact - improve impact assessment")
        
        if not recommendations:
            recommendations.append("Governance performance is satisfactory - maintain current processes")
        
        return recommendations


class DGMGovernanceSystem:
    """
    DGM-Enhanced Governance System that integrates evolutionary capabilities
    with community oversight and democratic decision-making.
    """
    
    def __init__(self, token_voting: TokenWeightedVoting, proposal_manager: ProposalManager):
        self.token_voting = token_voting
        self.proposal_manager = proposal_manager
        self.community_review = CommunityReviewSystem()
        self.performance_tracker = GovernancePerformanceTracker()
        
        # Active proposals and decisions
        self.active_proposals = {}
        self.pending_implementations = {}
        
        logger.info("DGM Governance System initialized")
    
    async def submit_modification_for_approval(
        self,
        modification_proposal: ModificationProposal,
        safety_validation: SafetyValidationResult,
        submitter_id: str
    ) -> str:
        """Submit a modification proposal for governance approval."""
        
        # Create governance proposal
        governance_proposal = EvolutionGovernanceProposal(
            proposal_id=str(uuid.uuid4()),
            decision_type=GovernanceDecisionType.MODIFICATION_APPROVAL,
            title=f"Approve Modification: {modification_proposal.description[:50]}...",
            description=self._create_modification_description(modification_proposal, safety_validation),
            proposer_id=submitter_id,
            modification_proposal=modification_proposal,
            safety_validation=safety_validation
        )
        
        # Initiate community review
        await self.community_review.initiate_community_review(governance_proposal)
        
        # Store active proposal
        self.active_proposals[governance_proposal.proposal_id] = governance_proposal
        
        logger.info(f"Submitted modification {modification_proposal.id} for governance approval as {governance_proposal.proposal_id}")
        
        return governance_proposal.proposal_id
    
    async def start_governance_vote(self, proposal_id: str) -> Dict[str, Any]:
        """Start the governance voting process for a proposal."""
        
        if proposal_id not in self.active_proposals:
            raise ValueError(f"Proposal {proposal_id} not found")
        
        proposal = self.active_proposals[proposal_id]
        
        # Set voting period
        proposal.voting_start_time = datetime.utcnow()
        proposal.voting_end_time = proposal.voting_start_time + timedelta(hours=proposal.voting_period_hours)
        proposal.status = "voting"
        
        # Configure voting parameters based on proposal type
        vote_config = {
            "proposal_id": proposal_id,
            "vote_type": proposal.vote_type.value,
            "required_quorum": proposal.required_quorum,
            "approval_threshold": proposal.approval_threshold,
            "voting_end_time": proposal.voting_end_time
        }
        
        # Start the actual vote through the token voting system
        voting_session = await self.token_voting.start_vote(
            proposal_id=proposal_id,
            description=proposal.description,
            options=["approve", "reject"],
            duration_hours=proposal.voting_period_hours
        )
        
        logger.info(f"Started governance vote for proposal {proposal_id} ({proposal.vote_type.value})")
        
        return {
            "voting_session_id": voting_session,
            "vote_config": vote_config,
            "proposal_summary": {
                "title": proposal.title,
                "decision_type": proposal.decision_type.value,
                "voting_period_hours": proposal.voting_period_hours,
                "requires_expert_review": proposal.requires_expert_review
            }
        }
    
    async def finalize_governance_decision(self, proposal_id: str) -> Dict[str, Any]:
        """Finalize governance decision and trigger implementation if approved."""
        
        if proposal_id not in self.active_proposals:
            raise ValueError(f"Proposal {proposal_id} not found")
        
        proposal = self.active_proposals[proposal_id]
        
        # Get voting results
        vote_results = await self.token_voting.get_vote_results(proposal_id)
        
        # Determine decision outcome
        decision_outcome = self._determine_decision_outcome(proposal, vote_results)
        proposal.status = decision_outcome
        
        # Track governance performance
        await self.performance_tracker.track_decision_outcome(proposal, decision_outcome)
        
        # Schedule implementation if approved
        implementation_scheduled = False
        if decision_outcome == "approved" and proposal.modification_proposal:
            implementation_scheduled = await self._schedule_modification_implementation(proposal)
        
        result = {
            "proposal_id": proposal_id,
            "decision_outcome": decision_outcome,
            "vote_results": vote_results,
            "implementation_scheduled": implementation_scheduled
        }
        
        logger.info(f"Finalized governance decision for {proposal_id}: {decision_outcome}")
        
        return result
    
    def _create_modification_description(
        self,
        modification: ModificationProposal,
        safety_validation: SafetyValidationResult
    ) -> str:
        """Create detailed description for modification proposal."""
        
        description = f"""
        Modification Proposal: {modification.description}
        
        Component: {modification.component_type.value}
        Risk Level: {modification.risk_level.value}
        Estimated Performance Impact: {modification.estimated_performance_impact:+.3f}
        
        Safety Validation Summary:
        - Overall Risk: {safety_validation.overall_risk_level.value}
        - Safety Status: {safety_validation.safety_status.value}
        - Total Violations: {safety_validation.total_violations}
        - Total Warnings: {safety_validation.total_warnings}
        
        Rationale: {modification.rationale}
        
        Safety Recommendations:
        {chr(10).join(f"- {rec}" for rec in safety_validation.safety_recommendations)}
        
        Rollback Plan: {modification.rollback_plan}
        """
        
        return description.strip()
    
    def _determine_decision_outcome(self, proposal: EvolutionGovernanceProposal, vote_results: Dict[str, Any]) -> str:
        """Determine decision outcome based on voting results and requirements."""
        
        # Check if quorum was met
        total_votes = vote_results.get("total_votes", 0)
        eligible_voters = vote_results.get("eligible_voters", 1)
        quorum_met = (total_votes / eligible_voters) >= proposal.required_quorum
        
        if not quorum_met:
            return "rejected"  # Quorum not met
        
        # Check approval threshold
        approve_votes = vote_results.get("approve_votes", 0)
        approval_rate = approve_votes / total_votes if total_votes > 0 else 0
        
        if approval_rate >= proposal.approval_threshold:
            return "approved"
        else:
            return "rejected"
    
    async def _schedule_modification_implementation(self, proposal: EvolutionGovernanceProposal) -> bool:
        """Schedule implementation of approved modification."""
        
        if not proposal.modification_proposal:
            return False
        
        # Add to pending implementations
        implementation_id = str(uuid.uuid4())
        self.pending_implementations[implementation_id] = {
            "proposal_id": proposal.proposal_id,
            "modification_proposal": proposal.modification_proposal,
            "safety_validation": proposal.safety_validation,
            "scheduled_at": datetime.utcnow(),
            "status": "pending"
        }
        
        logger.info(f"Scheduled implementation {implementation_id} for approved proposal {proposal.proposal_id}")
        
        return True
    
    async def get_governance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive governance system dashboard."""
        
        # Get performance insights
        governance_insights = self.performance_tracker.get_governance_insights()
        
        # Get active proposal summaries
        active_proposal_summaries = []
        for proposal_id, proposal in self.active_proposals.items():
            if proposal.status in ["draft", "voting"]:
                active_proposal_summaries.append({
                    "proposal_id": proposal_id,
                    "title": proposal.title,
                    "decision_type": proposal.decision_type.value,
                    "status": proposal.status,
                    "voting_end_time": proposal.voting_end_time.isoformat() if proposal.voting_end_time else None
                })
        
        # Get pending implementations
        pending_impl_summaries = []
        for impl_id, impl in self.pending_implementations.items():
            if impl["status"] == "pending":
                pending_impl_summaries.append({
                    "implementation_id": impl_id,
                    "proposal_id": impl["proposal_id"],
                    "scheduled_at": impl["scheduled_at"].isoformat(),
                    "modification_type": impl["modification_proposal"].modification_type
                })
        
        dashboard = {
            "governance_insights": governance_insights,
            "active_proposals": active_proposal_summaries,
            "pending_implementations": pending_impl_summaries,
            "system_status": {
                "expert_panels": len(self.community_review.expert_panels),
                "stakeholder_groups": len(self.community_review.stakeholder_groups),
                "total_decisions": len(self.performance_tracker.decision_history)
            }
        }
        
        return dashboard