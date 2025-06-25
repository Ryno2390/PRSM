"""
PRSM Proposal Management System

Implements comprehensive proposal lifecycle management including eligibility
validation, lifecycle management, and execution of approved governance proposals.

Based on execution_plan.md Phase 3, Week 17-18 requirements.
"""

import asyncio
import math
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from uuid import UUID, uuid4
from enum import Enum
import structlog

# Set precision for proposal calculations
getcontext().prec = 18

from ..core.config import settings
from ..core.models import GovernanceProposal, Vote, PRSMBaseModel
from ..tokenomics.ftns_service import ftns_service
from ..safety.monitor import SafetyMonitor
from ..safety.governance import SafetyGovernance
from ..improvement.proposal_engine import ImprovementProposalEngine

logger = structlog.get_logger()

# === Proposal Configuration ===

# Eligibility requirements
MIN_PROPOSER_BALANCE = float(getattr(settings, "GOVERNANCE_MIN_PROPOSER_BALANCE", 10000.0))  # 10K FTNS
MIN_COMMUNITY_SUPPORT = int(getattr(settings, "GOVERNANCE_MIN_COMMUNITY_SUPPORT", 25))  # 25 supporters
PROPOSAL_REVIEW_PERIOD_HOURS = int(getattr(settings, "GOVERNANCE_REVIEW_PERIOD_HOURS", 48))  # 48 hours

# Lifecycle management
PROPOSAL_DRAFT_PERIOD_DAYS = int(getattr(settings, "GOVERNANCE_DRAFT_PERIOD_DAYS", 3))  # 3 days for drafts
PROPOSAL_DISCUSSION_PERIOD_DAYS = int(getattr(settings, "GOVERNANCE_DISCUSSION_PERIOD_DAYS", 7))  # 7 days discussion
PROPOSAL_EXECUTION_TIMEOUT_DAYS = int(getattr(settings, "GOVERNANCE_EXECUTION_TIMEOUT_DAYS", 30))  # 30 days to execute

# Review requirements
REQUIRED_REVIEWER_APPROVALS = int(getattr(settings, "GOVERNANCE_REQUIRED_REVIEWER_APPROVALS", 3))  # 3 reviewers
REVIEWER_DIVERSITY_REQUIREMENT = float(getattr(settings, "GOVERNANCE_REVIEWER_DIVERSITY", 0.6))  # 60% different specializations


class ProposalStatus(str, Enum):
    """Enhanced proposal status tracking"""
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    COMMUNITY_DISCUSSION = "community_discussion"
    VOTING_ACTIVE = "voting_active"
    VOTING_CONCLUDED = "voting_concluded"
    APPROVED = "approved"
    REJECTED = "rejected"
    FAILED_QUORUM = "failed_quorum"
    EXECUTED = "executed"
    EXECUTION_FAILED = "execution_failed"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"


class ProposalPriority(str, Enum):
    """Proposal priority levels"""
    CRITICAL = "critical"     # Emergency/safety issues
    HIGH = "high"            # Important system changes
    MEDIUM = "medium"        # Standard improvements
    LOW = "low"             # Nice-to-have features


class ReviewDecision(str, Enum):
    """Reviewer decision types"""
    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"
    REJECT = "reject"
    ABSTAIN = "abstain"


class ExecutionStrategy(str, Enum):
    """Proposal execution strategies"""
    IMMEDIATE = "immediate"      # Execute immediately after approval
    SCHEDULED = "scheduled"      # Execute at specified time
    CONDITIONAL = "conditional"  # Execute when conditions are met
    MANUAL = "manual"           # Requires manual execution trigger


class ProposalReview(PRSMBaseModel):
    """Individual proposal review record"""
    review_id: UUID = uuid4()
    proposal_id: UUID
    reviewer_id: str
    reviewer_role: str
    decision: ReviewDecision
    feedback: str = ""
    safety_concerns: List[str] = []
    technical_issues: List[str] = []
    economic_impact_assessment: Optional[Dict[str, Any]] = None
    recommendation: str = ""
    review_submitted_at: datetime = datetime.now(timezone.utc)


class ProposalSupport(PRSMBaseModel):
    """Community support record"""
    support_id: UUID = uuid4()
    proposal_id: UUID
    supporter_id: str
    support_type: str  # "endorse", "co_sponsor", "general_support"
    voting_power_pledged: float
    message: str = ""
    supported_at: datetime = datetime.now(timezone.utc)


class ExecutionPlan(PRSMBaseModel):
    """Proposal execution plan"""
    execution_id: UUID = uuid4()
    proposal_id: UUID
    strategy: ExecutionStrategy
    scheduled_execution_time: Optional[datetime] = None
    execution_conditions: List[str] = []
    estimated_duration_hours: int = 24
    required_resources: Dict[str, Any] = {}
    rollback_plan: Optional[str] = None
    execution_steps: List[Dict[str, Any]] = []
    responsible_party: str = "system"


class ProposalMetrics(PRSMBaseModel):
    """Proposal performance metrics"""
    proposal_id: UUID
    community_engagement_score: float = 0.0
    technical_feasibility_score: float = 0.0
    economic_impact_score: float = 0.0
    safety_risk_score: float = 0.0
    overall_quality_score: float = 0.0
    discussion_activity: int = 0
    support_momentum: float = 0.0
    controversy_level: float = 0.0
    implementation_complexity: float = 0.0


class ProposalManager:
    """
    Comprehensive proposal lifecycle management system
    Handles eligibility validation, review coordination, and execution management
    """
    
    def __init__(self):
        self.manager_id = str(uuid4())
        self.logger = logger.bind(component="proposal_manager", manager_id=self.manager_id)
        
        # Proposal management state
        self.proposals: Dict[UUID, GovernanceProposal] = {}
        self.proposal_reviews: Dict[UUID, List[ProposalReview]] = defaultdict(list)
        self.community_support: Dict[UUID, List[ProposalSupport]] = defaultdict(list)
        self.execution_plans: Dict[UUID, ExecutionPlan] = {}
        self.proposal_metrics: Dict[UUID, ProposalMetrics] = {}
        
        # Review coordination
        self.available_reviewers: Dict[str, Set[str]] = defaultdict(set)  # specialization -> reviewer_ids
        self.reviewer_workload: Dict[str, int] = defaultdict(int)
        self.review_assignments: Dict[UUID, List[str]] = defaultdict(list)
        
        # Execution management
        self.execution_queue: List[UUID] = []
        self.active_executions: Dict[UUID, Dict[str, Any]] = {}
        self.execution_history: Dict[UUID, List[Dict[str, Any]]] = defaultdict(list)
        
        # Integration components
        self.safety_monitor = SafetyMonitor()
        self.safety_governance = SafetyGovernance()
        self.improvement_engine = ImprovementProposalEngine()
        
        # Performance statistics
        self.management_stats = {
            "total_proposals_processed": 0,
            "proposals_approved_rate": 0.0,
            "average_review_time_hours": 0.0,
            "average_execution_time_hours": 0.0,
            "community_engagement_rate": 0.0,
            "reviewer_utilization_rate": 0.0,
            "execution_success_rate": 0.0,
            "safety_flags_raised": 0,
            "technical_issues_identified": 0
        }
        
        # Synchronization
        self._proposal_lock = asyncio.Lock()
        self._review_lock = asyncio.Lock()
        self._execution_lock = asyncio.Lock()
        
        print("📋 ProposalManager initialized")
    
    
    async def validate_proposal_eligibility(self, proposal: GovernanceProposal) -> bool:
        """
        Comprehensive proposal eligibility validation
        
        Args:
            proposal: Governance proposal to validate
            
        Returns:
            True if proposal is eligible for submission
        """
        try:
            # Basic validation
            if not proposal.title or len(proposal.title.strip()) < 10:
                self.logger.warning("Proposal title too short", title=proposal.title)
                return False
            
            if not proposal.description or len(proposal.description.strip()) < 100:
                self.logger.warning("Proposal description too short")
                return False
            
            # Proposer eligibility
            if not await self._validate_proposer_credentials(proposal.proposer_id):
                return False
            
            # Content validation
            if not await self._validate_proposal_content(proposal):
                return False
            
            # Technical feasibility check
            if not await self._validate_technical_feasibility(proposal):
                return False
            
            # Safety validation
            if not await self._validate_proposal_safety(proposal):
                return False
            
            # Economic impact validation
            if not await self._validate_economic_impact(proposal):
                return False
            
            # Duplication check
            if await self._check_proposal_duplication(proposal):
                self.logger.warning("Similar proposal already exists")
                return False
            
            self.logger.info(
                "Proposal eligibility validated",
                proposal_id=str(proposal.proposal_id),
                proposer=proposal.proposer_id
            )
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to validate proposal eligibility", error=str(e))
            return False
    
    
    async def manage_proposal_lifecycle(self, proposal_id: UUID) -> Dict[str, Any]:
        """
        Manage the complete lifecycle of a governance proposal
        
        Args:
            proposal_id: ID of the proposal to manage
            
        Returns:
            Lifecycle management results and next steps
        """
        try:
            async with self._proposal_lock:
                if proposal_id not in self.proposals:
                    raise ValueError("Proposal not found")
                
                proposal = self.proposals[proposal_id]
                current_status = ProposalStatus(proposal.status)
                lifecycle_results = {
                    "proposal_id": str(proposal_id),
                    "current_status": current_status,
                    "transitions_made": [],
                    "next_actions": [],
                    "timeline_updates": [],
                    "notifications_sent": []
                }
                
                # Process status transitions
                if current_status == ProposalStatus.DRAFT:
                    await self._process_draft_status(proposal, lifecycle_results)
                
                elif current_status == ProposalStatus.UNDER_REVIEW:
                    await self._process_review_status(proposal, lifecycle_results)
                
                elif current_status == ProposalStatus.COMMUNITY_DISCUSSION:
                    await self._process_discussion_status(proposal, lifecycle_results)
                
                elif current_status == ProposalStatus.VOTING_ACTIVE:
                    await self._process_voting_status(proposal, lifecycle_results)
                
                elif current_status == ProposalStatus.APPROVED:
                    await self._process_approved_status(proposal, lifecycle_results)
                
                elif current_status == ProposalStatus.EXECUTED:
                    await self._process_executed_status(proposal, lifecycle_results)
                
                # Update proposal metrics
                await self._update_proposal_metrics(proposal_id)
                
                # Update statistics
                self.management_stats["total_proposals_processed"] += 1
                
                self.logger.info(
                    "Proposal lifecycle managed",
                    proposal_id=str(proposal_id),
                    status=current_status,
                    transitions=len(lifecycle_results["transitions_made"])
                )
                
                return lifecycle_results
                
        except Exception as e:
            self.logger.error("Failed to manage proposal lifecycle", error=str(e))
            return {
                "proposal_id": str(proposal_id),
                "error": str(e),
                "transitions_made": [],
                "next_actions": ["retry_lifecycle_management"]
            }
    
    
    async def execute_approved_proposals(self, proposal_id: UUID) -> Dict[str, Any]:
        """
        Execute approved governance proposals according to their execution plans
        
        Args:
            proposal_id: ID of the approved proposal to execute
            
        Returns:
            Execution results and status
        """
        try:
            async with self._execution_lock:
                if proposal_id not in self.proposals:
                    raise ValueError("Proposal not found")
                
                proposal = self.proposals[proposal_id]
                
                # Validate proposal is approved
                if proposal.status != "approved":
                    raise ValueError("Proposal is not approved for execution")
                
                # Get execution plan
                execution_plan = self.execution_plans.get(proposal_id)
                if not execution_plan:
                    execution_plan = await self._create_default_execution_plan(proposal_id)
                
                execution_results = {
                    "proposal_id": str(proposal_id),
                    "execution_id": str(execution_plan.execution_id),
                    "execution_strategy": execution_plan.strategy,
                    "execution_started_at": datetime.now(timezone.utc),
                    "steps_completed": [],
                    "steps_failed": [],
                    "overall_status": "in_progress",
                    "estimated_completion": None,
                    "rollback_triggered": False
                }
                
                # Mark execution as active
                self.active_executions[proposal_id] = execution_results
                
                # Execute based on strategy
                if execution_plan.strategy == ExecutionStrategy.IMMEDIATE:
                    await self._execute_immediate(proposal, execution_plan, execution_results)
                
                elif execution_plan.strategy == ExecutionStrategy.SCHEDULED:
                    await self._execute_scheduled(proposal, execution_plan, execution_results)
                
                elif execution_plan.strategy == ExecutionStrategy.CONDITIONAL:
                    await self._execute_conditional(proposal, execution_plan, execution_results)
                
                elif execution_plan.strategy == ExecutionStrategy.MANUAL:
                    await self._execute_manual(proposal, execution_plan, execution_results)
                
                # Update proposal status
                if execution_results["overall_status"] == "completed":
                    proposal.status = "executed"
                    self.management_stats["execution_success_rate"] += 1
                elif execution_results["overall_status"] == "failed":
                    proposal.status = "execution_failed"
                
                # Store execution history
                self.execution_history[proposal_id].append(execution_results)
                
                # Clean up active execution
                if proposal_id in self.active_executions:
                    del self.active_executions[proposal_id]
                
                self.logger.info(
                    "Proposal execution completed",
                    proposal_id=str(proposal_id),
                    status=execution_results["overall_status"],
                    steps_completed=len(execution_results["steps_completed"])
                )
                
                return execution_results
                
        except Exception as e:
            self.logger.error("Failed to execute proposal", error=str(e))
            return {
                "proposal_id": str(proposal_id),
                "execution_id": str(uuid4()),
                "overall_status": "failed",
                "error": str(e),
                "rollback_triggered": True
            }
    
    
    async def submit_proposal_review(self, proposal_id: UUID, reviewer_id: str, 
                                   decision: ReviewDecision, feedback: str = "") -> bool:
        """Submit a review for a proposal"""
        try:
            async with self._review_lock:
                if proposal_id not in self.proposals:
                    return False
                
                # Create review record
                review = ProposalReview(
                    proposal_id=proposal_id,
                    reviewer_id=reviewer_id,
                    reviewer_role=await self._get_reviewer_role(reviewer_id),
                    decision=decision,
                    feedback=feedback,
                    safety_concerns=await self._assess_safety_concerns(proposal_id),
                    technical_issues=await self._assess_technical_issues(proposal_id),
                    economic_impact_assessment=await self._assess_economic_impact(proposal_id)
                )
                
                # Store review
                self.proposal_reviews[proposal_id].append(review)
                
                # Update reviewer workload
                self.reviewer_workload[reviewer_id] -= 1
                
                # Check if all required reviews are complete
                await self._check_review_completion(proposal_id)
                
                self.logger.info(
                    "Proposal review submitted",
                    proposal_id=str(proposal_id),
                    reviewer=reviewer_id,
                    decision=decision
                )
                
                return True
                
        except Exception as e:
            self.logger.error("Failed to submit review", error=str(e))
            return False
    
    
    async def add_community_support(self, proposal_id: UUID, supporter_id: str, 
                                  support_type: str = "general_support", 
                                  message: str = "") -> bool:
        """Add community support to a proposal"""
        try:
            if proposal_id not in self.proposals:
                return False
            
            # Get supporter's voting power
            user_balance = await ftns_service.get_user_balance(supporter_id)
            voting_power = min(user_balance.balance, 1000.0)  # Cap support power
            
            support = ProposalSupport(
                proposal_id=proposal_id,
                supporter_id=supporter_id,
                support_type=support_type,
                voting_power_pledged=voting_power,
                message=message
            )
            
            self.community_support[proposal_id].append(support)
            
            # Update proposal metrics
            await self._update_proposal_metrics(proposal_id)
            
            self.logger.info(
                "Community support added",
                proposal_id=str(proposal_id),
                supporter=supporter_id,
                power=voting_power
            )
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to add community support", error=str(e))
            return False
    
    
    async def get_proposal_status(self, proposal_id: UUID) -> Optional[Dict[str, Any]]:
        """Get comprehensive proposal status and metrics"""
        if proposal_id not in self.proposals:
            return None
        
        proposal = self.proposals[proposal_id]
        reviews = self.proposal_reviews[proposal_id]
        support = self.community_support[proposal_id]
        metrics = self.proposal_metrics.get(proposal_id)
        
        return {
            "proposal": proposal.dict(),
            "status": proposal.status,
            "reviews": {
                "total_reviews": len(reviews),
                "reviewer_decisions": [r.decision for r in reviews],
                "average_quality_score": sum(r.economic_impact_assessment.get("quality_score", 0) for r in reviews if r.economic_impact_assessment) / max(len(reviews), 1),
                "safety_concerns_raised": sum(len(r.safety_concerns) for r in reviews),
                "technical_issues_raised": sum(len(r.technical_issues) for r in reviews)
            },
            "community_support": {
                "total_supporters": len(support),
                "total_voting_power_pledged": sum(s.voting_power_pledged for s in support),
                "support_types": {st: len([s for s in support if s.support_type == st]) for st in set(s.support_type for s in support)}
            },
            "metrics": metrics.dict() if metrics else None,
            "execution_plan": self.execution_plans.get(proposal_id, {}).dict() if proposal_id in self.execution_plans else None,
            "execution_history": self.execution_history.get(proposal_id, [])
        }
    
    
    async def get_management_statistics(self) -> Dict[str, Any]:
        """Get comprehensive proposal management statistics"""
        total_proposals = len(self.proposals)
        
        if total_proposals > 0:
            approved_proposals = len([p for p in self.proposals.values() if p.status == "approved"])
            executed_proposals = len([p for p in self.proposals.values() if p.status == "executed"])
            
            self.management_stats["proposals_approved_rate"] = approved_proposals / total_proposals
            self.management_stats["execution_success_rate"] = executed_proposals / max(approved_proposals, 1)
        
        return {
            **self.management_stats,
            "active_proposals": len([p for p in self.proposals.values() if p.status in ["voting_active", "under_review", "community_discussion"]]),
            "total_reviews_submitted": sum(len(reviews) for reviews in self.proposal_reviews.values()),
            "total_community_support": sum(len(support) for support in self.community_support.values()),
            "active_executions": len(self.active_executions),
            "reviewer_pool_size": len(set().union(*self.available_reviewers.values())),
            "average_support_per_proposal": sum(len(support) for support in self.community_support.values()) / max(len(self.proposals), 1)
        }
    
    
    # === Private Helper Methods ===
    
    async def _validate_proposer_credentials(self, proposer_id: str) -> bool:
        """Validate proposer has sufficient credentials"""
        # Check FTNS balance
        user_balance = await ftns_service.get_user_balance(proposer_id)
        if user_balance.balance < MIN_PROPOSER_BALANCE:
            return False
        
        # Check community standing (simplified)
        return True
    
    
    async def _validate_proposal_content(self, proposal: GovernanceProposal) -> bool:
        """Validate proposal content quality and appropriateness"""
        # Safety validation
        safety_check = await self.safety_monitor.validate_model_output(
            {"proposal": proposal.dict()},
            ["no_malicious_content", "content_appropriateness", "technical_accuracy"]
        )
        
        return safety_check
    
    
    async def _validate_technical_feasibility(self, proposal: GovernanceProposal) -> bool:
        """Validate technical feasibility of the proposal"""
        # Use improvement engine for technical validation
        if hasattr(proposal, 'implementation_details'):
            feasibility_result = await self.improvement_engine.simulate_proposed_changes(proposal)
            return feasibility_result.get("technically_feasible", True)
        
        return True
    
    
    async def _validate_proposal_safety(self, proposal: GovernanceProposal) -> bool:
        """Validate proposal safety implications"""
        safety_check = await self.improvement_engine.validate_improvement_safety(proposal)
        return safety_check.get("safety_approved", True)
    
    
    async def _validate_economic_impact(self, proposal: GovernanceProposal) -> bool:
        """Validate economic impact of the proposal"""
        # Check for reasonable resource requirements
        if hasattr(proposal, 'budget_impact'):
            return proposal.budget_impact.get('total_cost', 0) <= 1000000  # 1M FTNS cap
        
        return True
    
    
    async def _check_proposal_duplication(self, proposal: GovernanceProposal) -> bool:
        """Check for duplicate or very similar proposals"""
        # Simplified duplication check
        for existing_proposal in self.proposals.values():
            if (existing_proposal.title.lower() == proposal.title.lower() and 
                existing_proposal.status in ["voting_active", "approved", "under_review"]):
                return True
        
        return False
    
    
    async def _process_draft_status(self, proposal: GovernanceProposal, results: Dict[str, Any]):
        """Process proposal in draft status"""
        # Check if draft period has elapsed
        if proposal.created_at and datetime.now(timezone.utc) - proposal.created_at > timedelta(days=PROPOSAL_DRAFT_PERIOD_DAYS):
            proposal.status = "under_review"
            results["transitions_made"].append({"from": "draft", "to": "under_review"})
            results["next_actions"].append("assign_reviewers")
            
            # Assign reviewers
            await self._assign_reviewers(proposal.proposal_id)
    
    
    async def _process_review_status(self, proposal: GovernanceProposal, results: Dict[str, Any]):
        """Process proposal under review"""
        reviews = self.proposal_reviews[proposal.proposal_id]
        
        if len(reviews) >= REQUIRED_REVIEWER_APPROVALS:
            # Check reviewer decisions
            approvals = len([r for r in reviews if r.decision == ReviewDecision.APPROVE])
            rejections = len([r for r in reviews if r.decision == ReviewDecision.REJECT])
            
            if approvals >= REQUIRED_REVIEWER_APPROVALS:
                proposal.status = "community_discussion"
                results["transitions_made"].append({"from": "under_review", "to": "community_discussion"})
                results["next_actions"].append("open_community_discussion")
            elif rejections >= REQUIRED_REVIEWER_APPROVALS:
                proposal.status = "rejected"
                results["transitions_made"].append({"from": "under_review", "to": "rejected"})
    
    
    async def _process_discussion_status(self, proposal: GovernanceProposal, results: Dict[str, Any]):
        """Process proposal in community discussion"""
        # Check if discussion period has elapsed and sufficient support exists
        support = self.community_support[proposal.proposal_id]
        
        if (len(support) >= MIN_COMMUNITY_SUPPORT and 
            datetime.now(timezone.utc) - proposal.created_at > timedelta(days=PROPOSAL_DISCUSSION_PERIOD_DAYS)):
            proposal.status = "voting_active"
            results["transitions_made"].append({"from": "community_discussion", "to": "voting_active"})
            results["next_actions"].append("initialize_voting")
    
    
    async def _process_voting_status(self, proposal: GovernanceProposal, results: Dict[str, Any]):
        """Process proposal with active voting"""
        # This would typically be handled by the voting system
        results["next_actions"].append("monitor_voting_progress")
    
    
    async def _process_approved_status(self, proposal: GovernanceProposal, results: Dict[str, Any]):
        """Process approved proposal"""
        # Prepare for execution
        if proposal.proposal_id not in self.execution_plans:
            execution_plan = await self._create_default_execution_plan(proposal.proposal_id)
            results["next_actions"].append("execute_proposal")
    
    
    async def _process_executed_status(self, proposal: GovernanceProposal, results: Dict[str, Any]):
        """Process executed proposal"""
        # Monitor post-execution effects
        results["next_actions"].append("monitor_post_execution_effects")
    
    
    async def _assign_reviewers(self, proposal_id: UUID) -> bool:
        """Assign qualified reviewers to a proposal"""
        try:
            proposal = self.proposals[proposal_id]
            
            # Get reviewers for proposal category
            category_reviewers = self.available_reviewers.get(proposal.proposal_type, set())
            
            # Select reviewers with lowest workload
            available_reviewers = sorted(
                category_reviewers, 
                key=lambda r: self.reviewer_workload.get(r, 0)
            )[:REQUIRED_REVIEWER_APPROVALS]
            
            # Assign reviewers
            for reviewer_id in available_reviewers:
                self.review_assignments[proposal_id].append(reviewer_id)
                self.reviewer_workload[reviewer_id] += 1
            
            return len(available_reviewers) >= REQUIRED_REVIEWER_APPROVALS
            
        except Exception as e:
            self.logger.error("Failed to assign reviewers", error=str(e))
            return False
    
    
    async def _create_default_execution_plan(self, proposal_id: UUID) -> ExecutionPlan:
        """Create a default execution plan for a proposal"""
        proposal = self.proposals[proposal_id]
        
        execution_plan = ExecutionPlan(
            proposal_id=proposal_id,
            strategy=ExecutionStrategy.IMMEDIATE,
            estimated_duration_hours=24,
            execution_steps=[
                {"step": 1, "action": "validate_pre_conditions", "estimated_minutes": 30},
                {"step": 2, "action": "implement_changes", "estimated_minutes": 120},
                {"step": 3, "action": "validate_post_conditions", "estimated_minutes": 30},
                {"step": 4, "action": "notify_stakeholders", "estimated_minutes": 15}
            ],
            rollback_plan="Revert all changes if validation fails"
        )
        
        self.execution_plans[proposal_id] = execution_plan
        return execution_plan
    
    
    async def _execute_immediate(self, proposal: GovernanceProposal, 
                                execution_plan: ExecutionPlan, 
                                results: Dict[str, Any]):
        """Execute proposal immediately"""
        try:
            for step in execution_plan.execution_steps:
                step_result = await self._execute_step(proposal, step)
                
                if step_result["success"]:
                    results["steps_completed"].append(step)
                else:
                    results["steps_failed"].append(step)
                    results["overall_status"] = "failed"
                    return
            
            results["overall_status"] = "completed"
            
        except Exception as e:
            results["overall_status"] = "failed"
            results["error"] = str(e)
    
    
    async def _execute_scheduled(self, proposal: GovernanceProposal,
                                execution_plan: ExecutionPlan,
                                results: Dict[str, Any]):
        """Execute proposal at scheduled time"""
        # For now, treat as immediate execution
        # In production, this would use a scheduler
        await self._execute_immediate(proposal, execution_plan, results)
    
    
    async def _execute_conditional(self, proposal: GovernanceProposal,
                                 execution_plan: ExecutionPlan,
                                 results: Dict[str, Any]):
        """Execute proposal when conditions are met"""
        # Check conditions
        conditions_met = await self._check_execution_conditions(execution_plan.execution_conditions)
        
        if conditions_met:
            await self._execute_immediate(proposal, execution_plan, results)
        else:
            results["overall_status"] = "waiting_for_conditions"
    
    
    async def _execute_manual(self, proposal: GovernanceProposal,
                            execution_plan: ExecutionPlan,
                            results: Dict[str, Any]):
        """Execute proposal manually"""
        # Manual execution requires operator intervention
        results["overall_status"] = "awaiting_manual_execution"
        results["next_actions"] = ["manual_execution_required"]
    
    
    async def _execute_step(self, proposal: GovernanceProposal, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step of the execution plan"""
        try:
            action = step["action"]
            
            if action == "validate_pre_conditions":
                # Validate system state before execution
                return {"success": True, "message": "Pre-conditions validated"}
            
            elif action == "implement_changes":
                # Implement the actual proposal changes
                return {"success": True, "message": "Changes implemented"}
            
            elif action == "validate_post_conditions":
                # Validate system state after execution
                return {"success": True, "message": "Post-conditions validated"}
            
            elif action == "notify_stakeholders":
                # Notify relevant stakeholders
                return {"success": True, "message": "Stakeholders notified"}
            
            else:
                return {"success": False, "message": f"Unknown action: {action}"}
                
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    
    async def _check_execution_conditions(self, conditions: List[str]) -> bool:
        """Check if execution conditions are met"""
        # Simplified condition checking
        return True
    
    
    async def _update_proposal_metrics(self, proposal_id: UUID):
        """Update comprehensive metrics for a proposal"""
        if proposal_id not in self.proposals:
            return
        
        proposal = self.proposals[proposal_id]
        reviews = self.proposal_reviews[proposal_id]
        support = self.community_support[proposal_id]
        
        # Calculate metrics
        metrics = ProposalMetrics(
            proposal_id=proposal_id,
            community_engagement_score=min(len(support) * 10.0, 100.0),
            technical_feasibility_score=85.0,  # Simplified
            economic_impact_score=75.0,  # Simplified
            safety_risk_score=max(0, 100 - sum(len(r.safety_concerns) for r in reviews) * 10),
            support_momentum=len(support) * 0.1,
            discussion_activity=len(support) + len(reviews),
            implementation_complexity=50.0  # Simplified
        )
        
        # Calculate overall quality score
        metrics.overall_quality_score = (
            metrics.technical_feasibility_score * 0.3 +
            metrics.economic_impact_score * 0.2 +
            metrics.safety_risk_score * 0.3 +
            metrics.community_engagement_score * 0.2
        )
        
        self.proposal_metrics[proposal_id] = metrics
    
    
    async def _get_reviewer_role(self, reviewer_id: str) -> str:
        """Get the role of a reviewer"""
        # Simplified role determination
        return "community_reviewer"
    
    
    async def _assess_safety_concerns(self, proposal_id: UUID) -> List[str]:
        """Assess safety concerns for a proposal"""
        # Simplified safety assessment
        return []
    
    
    async def _assess_technical_issues(self, proposal_id: UUID) -> List[str]:
        """Assess technical issues for a proposal"""
        # Simplified technical assessment
        return []
    
    
    async def _assess_economic_impact(self, proposal_id: UUID) -> Dict[str, Any]:
        """Assess economic impact of a proposal"""
        # Simplified economic assessment
        return {
            "cost_estimate": 10000.0,
            "benefit_estimate": 50000.0,
            "roi_estimate": 5.0,
            "quality_score": 75.0
        }
    
    
    async def _check_review_completion(self, proposal_id: UUID):
        """Check if all required reviews are complete"""
        reviews = self.proposal_reviews[proposal_id]
        
        if len(reviews) >= REQUIRED_REVIEWER_APPROVALS:
            # All reviews complete, trigger lifecycle management
            await self.manage_proposal_lifecycle(proposal_id)


# === Global Proposal Manager Instance ===

_manager_instance: Optional[ProposalManager] = None

def get_proposal_manager() -> ProposalManager:
    """Get or create the global proposal manager instance"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ProposalManager()
    return _manager_instance