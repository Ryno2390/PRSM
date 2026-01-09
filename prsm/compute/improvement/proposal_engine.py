"""
PRSM Improvement Proposal Engine
Generates architecture proposals, simulates changes, and validates safety
"""

import asyncio
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4

from prsm.core.config import settings
from prsm.core.models import (
    ImprovementProposal, ImprovementType, ProposalStatus,
    SimulationResult, SafetyCheck, ImprovementOpportunity,
    PerformanceMetric, MetricType
)
from prsm.core.safety.monitor import SafetyMonitor
from prsm.core.safety.circuit_breaker import CircuitBreakerNetwork
from prsm.economy.tokenomics.ftns_service import get_ftns_service


# --- Proposal Engine Configuration ---

# Proposal generation settings
MIN_CONFIDENCE_FOR_PROPOSAL = float(getattr(settings, "PRSM_MIN_PROPOSAL_CONFIDENCE", 0.6))
MAX_PROPOSALS_PER_ANALYSIS = int(getattr(settings, "PRSM_MAX_PROPOSALS", 5))
PROPOSAL_PRIORITY_THRESHOLD = float(getattr(settings, "PRSM_PROPOSAL_PRIORITY_THRESHOLD", 0.5))

# Simulation settings
SIMULATION_DURATION_HOURS = float(getattr(settings, "PRSM_SIMULATION_DURATION", 24.0))
SIMULATION_CONFIDENCE_THRESHOLD = float(getattr(settings, "PRSM_SIMULATION_CONFIDENCE", 0.7))
PERFORMANCE_VARIANCE_FACTOR = float(getattr(settings, "PRSM_PERFORMANCE_VARIANCE", 0.1))

# Safety validation settings
SAFETY_SCORE_THRESHOLD = float(getattr(settings, "PRSM_SAFETY_THRESHOLD", 0.8))
HIGH_RISK_REJECTION_THRESHOLD = float(getattr(settings, "PRSM_HIGH_RISK_THRESHOLD", 0.3))
GOVERNANCE_APPROVAL_THRESHOLD = float(getattr(settings, "PRSM_GOVERNANCE_THRESHOLD", 0.9))


class ImprovementProposalEngine:
    """
    Engine for generating, simulating, and validating system improvement proposals
    Analyzes weaknesses and proposes targeted architectural and operational improvements
    """
    
    def __init__(self):
        # Component integration
        self.safety_monitor = SafetyMonitor()
        self.circuit_breaker = CircuitBreakerNetwork()
        self.ftns_service = get_ftns_service()
        
        # Proposal storage
        self.proposals: Dict[UUID, ImprovementProposal] = {}
        self.proposal_templates: Dict[ImprovementType, Dict[str, Any]] = {}
        self.weakness_patterns: Dict[str, List[str]] = {}
        
        # Performance tracking
        self.engine_stats = {
            "proposals_generated": 0,
            "simulations_run": 0,
            "safety_checks_performed": 0,
            "approved_proposals": 0,
            "rejected_proposals": 0,
            "implementation_successes": 0
        }
        
        # Initialize proposal templates
        self._initialize_proposal_templates()
        self._initialize_weakness_patterns()
        
        # Synchronization
        self._proposals_lock = asyncio.Lock()
        
        print("🔧 ImprovementProposalEngine initialized")
    
    
    async def generate_architecture_proposals(self, weakness_analysis: Dict[str, Any]) -> List[ImprovementProposal]:
        """
        Generate architecture improvement proposals based on weakness analysis
        
        Args:
            weakness_analysis: Analysis of current system weaknesses and bottlenecks
            
        Returns:
            List of generated improvement proposals
        """
        try:
            proposals = []
            
            # Extract key weaknesses
            weaknesses = weakness_analysis.get("identified_weaknesses", [])
            performance_issues = weakness_analysis.get("performance_bottlenecks", {})
            resource_constraints = weakness_analysis.get("resource_constraints", {})
            
            # Generate proposals for each weakness category
            for weakness in weaknesses:
                category_proposals = await self._generate_proposals_for_weakness(
                    weakness, performance_issues, resource_constraints
                )
                proposals.extend(category_proposals)
            
            # Generate proposals for performance bottlenecks
            for component, issues in performance_issues.items():
                performance_proposals = await self._generate_performance_proposals(
                    component, issues, weakness_analysis
                )
                proposals.extend(performance_proposals)
            
            # Generate proposals for resource constraints
            for resource, constraint_data in resource_constraints.items():
                resource_proposals = await self._generate_resource_proposals(
                    resource, constraint_data, weakness_analysis
                )
                proposals.extend(resource_proposals)
            
            # Sort by priority and limit count
            proposals.sort(key=lambda p: p.priority_score, reverse=True)
            proposals = proposals[:MAX_PROPOSALS_PER_ANALYSIS]
            
            # Store proposals
            async with self._proposals_lock:
                for proposal in proposals:
                    self.proposals[proposal.proposal_id] = proposal
            
            # Update stats
            self.engine_stats["proposals_generated"] += len(proposals)
            
            print(f"🎯 Generated {len(proposals)} architecture proposals")
            
            return proposals
            
        except Exception as e:
            print(f"❌ Error generating architecture proposals: {str(e)}")
            return []
    
    
    async def simulate_proposed_changes(self, proposal: ImprovementProposal) -> SimulationResult:
        """
        Simulate the effects of a proposed improvement
        
        Args:
            proposal: The improvement proposal to simulate
            
        Returns:
            Simulation results including performance predictions
        """
        try:
            start_time = datetime.now(timezone.utc)
            
            # Get baseline performance for comparison
            baseline_performance = await self._get_baseline_performance(proposal.target_component)
            
            # Simulate the proposed changes
            predicted_changes = await self._simulate_performance_changes(proposal, baseline_performance)
            
            # Calculate resource requirements
            resource_requirements = await self._calculate_resource_requirements(proposal)
            
            # Assess implementation risks
            risk_assessment = await self._assess_implementation_risks(proposal)
            
            # Calculate confidence score
            confidence_score = await self._calculate_simulation_confidence(
                proposal, predicted_changes, baseline_performance
            )
            
            # Perform validation checks
            validation_metrics = await self._validate_simulation_results(
                proposal, predicted_changes, resource_requirements
            )
            
            # Calculate simulation duration
            end_time = datetime.now(timezone.utc)
            simulation_duration = (end_time - start_time).total_seconds()
            
            # Create simulation result
            simulation_result = SimulationResult(
                proposal_id=proposal.proposal_id,
                simulation_duration=simulation_duration,
                predicted_performance_change=predicted_changes,
                resource_requirements=resource_requirements,
                risk_assessment=risk_assessment,
                confidence_score=confidence_score,
                validation_metrics=validation_metrics
            )
            
            # Update proposal with simulation results
            proposal.simulation_result = simulation_result
            
            # Update stats
            self.engine_stats["simulations_run"] += 1
            
            print(f"🔬 Simulated proposal {proposal.proposal_id}: {confidence_score:.2f} confidence")
            
            return simulation_result
            
        except Exception as e:
            print(f"❌ Error simulating proposal {proposal.proposal_id}: {str(e)}")
            return SimulationResult(
                proposal_id=proposal.proposal_id,
                simulation_duration=0.0,
                predicted_performance_change={},
                resource_requirements={},
                risk_assessment={"error": 1.0},
                confidence_score=0.0,
                validation_metrics={"error": str(e)}
            )
    
    
    async def validate_improvement_safety(self, proposal: ImprovementProposal) -> SafetyCheck:
        """
        Validate the safety implications of an improvement proposal
        
        Args:
            proposal: The improvement proposal to validate
            
        Returns:
            Safety check results with risk assessment
        """
        try:
            # Analyze potential safety risks
            potential_risks = await self._analyze_safety_risks(proposal)
            
            # Generate risk mitigation strategies
            mitigation_strategies = await self._generate_mitigation_strategies(
                proposal, potential_risks
            )
            
            # Assess circuit breaker impact
            circuit_breaker_impact = await self._assess_circuit_breaker_impact(proposal)
            
            # Determine governance requirements
            governance_requirements = await self._determine_governance_requirements(
                proposal, potential_risks
            )
            
            # Calculate overall safety score
            safety_score = await self._calculate_safety_score(
                proposal, potential_risks, circuit_breaker_impact
            )
            
            # Determine if approval is required
            approval_required = (
                safety_score < SAFETY_SCORE_THRESHOLD or
                len(governance_requirements) > 0 or
                proposal.implementation_cost > GOVERNANCE_APPROVAL_THRESHOLD
            )
            
            # Create safety check result
            safety_check = SafetyCheck(
                proposal_id=proposal.proposal_id,
                safety_score=safety_score,
                potential_risks=potential_risks,
                risk_mitigation_strategies=mitigation_strategies,
                circuit_breaker_impact=circuit_breaker_impact,
                governance_requirements=governance_requirements,
                approval_required=approval_required
            )
            
            # Update proposal with safety check
            proposal.safety_check = safety_check
            
            # Update stats
            self.engine_stats["safety_checks_performed"] += 1
            
            print(f"🛡️ Safety check for {proposal.proposal_id}: {safety_score:.2f} score")
            
            return safety_check
            
        except Exception as e:
            print(f"❌ Error validating safety for proposal {proposal.proposal_id}: {str(e)}")
            return SafetyCheck(
                proposal_id=proposal.proposal_id,
                safety_score=0.0,
                potential_risks=[f"Safety validation error: {str(e)}"],
                risk_mitigation_strategies=["Manual review required"],
                circuit_breaker_impact={"error": True},
                governance_requirements=["Emergency review"],
                approval_required=True
            )
    
    
    async def get_proposal(self, proposal_id: UUID) -> Optional[ImprovementProposal]:
        """Get a specific proposal by ID"""
        async with self._proposals_lock:
            return self.proposals.get(proposal_id)
    
    
    async def list_proposals(self, status: Optional[ProposalStatus] = None, 
                           improvement_type: Optional[ImprovementType] = None) -> List[ImprovementProposal]:
        """List proposals with optional filtering"""
        async with self._proposals_lock:
            proposals = list(self.proposals.values())
            
            if status:
                proposals = [p for p in proposals if p.status == status]
            
            if improvement_type:
                proposals = [p for p in proposals if p.improvement_type == improvement_type]
            
            return sorted(proposals, key=lambda p: p.priority_score, reverse=True)
    
    
    async def update_proposal_status(self, proposal_id: UUID, new_status: ProposalStatus, 
                                   notes: str = "") -> bool:
        """Update the status of a proposal"""
        try:
            async with self._proposals_lock:
                if proposal_id in self.proposals:
                    proposal = self.proposals[proposal_id]
                    old_status = proposal.status
                    proposal.status = new_status
                    proposal.updated_at = datetime.now(timezone.utc)
                    
                    # Add to approval history
                    proposal.approval_history.append({
                        "status_change": f"{old_status.value} -> {new_status.value}",
                        "timestamp": proposal.updated_at.isoformat(),
                        "notes": notes
                    })
                    
                    # Update stats
                    if new_status == ProposalStatus.APPROVED:
                        self.engine_stats["approved_proposals"] += 1
                    elif new_status == ProposalStatus.REJECTED:
                        self.engine_stats["rejected_proposals"] += 1
                    elif new_status == ProposalStatus.COMPLETED:
                        self.engine_stats["implementation_successes"] += 1
                    
                    print(f"📝 Updated proposal {proposal_id} status: {old_status.value} -> {new_status.value}")
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error updating proposal status: {str(e)}")
            return False
    
    
    async def get_engine_statistics(self) -> Dict[str, Any]:
        """Get current engine statistics"""
        async with self._proposals_lock:
            proposal_count_by_status = defaultdict(int)
            proposal_count_by_type = defaultdict(int)
            
            for proposal in self.proposals.values():
                proposal_count_by_status[proposal.status.value] += 1
                proposal_count_by_type[proposal.improvement_type.value] += 1
            
            return {
                **self.engine_stats,
                "total_proposals": len(self.proposals),
                "proposals_by_status": dict(proposal_count_by_status),
                "proposals_by_type": dict(proposal_count_by_type),
                "success_rate": (
                    self.engine_stats["implementation_successes"] / 
                    max(1, self.engine_stats["approved_proposals"])
                ),
                "configuration": {
                    "min_confidence": MIN_CONFIDENCE_FOR_PROPOSAL,
                    "max_proposals": MAX_PROPOSALS_PER_ANALYSIS,
                    "safety_threshold": SAFETY_SCORE_THRESHOLD,
                    "simulation_duration": SIMULATION_DURATION_HOURS
                }
            }
    
    
    # === Private Helper Methods ===
    
    def _initialize_proposal_templates(self):
        """Initialize templates for different types of improvement proposals"""
        self.proposal_templates = {
            ImprovementType.ARCHITECTURE: {
                "expected_benefits": {"performance": 0.15, "scalability": 0.20, "maintainability": 0.10},
                "implementation_cost": 0.7,
                "timeline_estimate": 14,
                "risk_factors": ["complexity", "compatibility", "performance_regression"]
            },
            ImprovementType.HYPERPARAMETER: {
                "expected_benefits": {"accuracy": 0.05, "training_speed": 0.10, "convergence": 0.08},
                "implementation_cost": 0.2,
                "timeline_estimate": 3,
                "risk_factors": ["overfitting", "training_instability"]
            },
            ImprovementType.TRAINING_DATA: {
                "expected_benefits": {"accuracy": 0.12, "robustness": 0.15, "generalization": 0.10},
                "implementation_cost": 0.5,
                "timeline_estimate": 7,
                "risk_factors": ["data_quality", "bias_introduction", "privacy"]
            },
            ImprovementType.MODEL_SIZE: {
                "expected_benefits": {"accuracy": 0.08, "memory_efficiency": 0.25, "inference_speed": 0.15},
                "implementation_cost": 0.4,
                "timeline_estimate": 5,
                "risk_factors": ["accuracy_loss", "compatibility_issues"]
            },
            ImprovementType.OPTIMIZATION: {
                "expected_benefits": {"performance": 0.20, "resource_usage": 0.15, "throughput": 0.18},
                "implementation_cost": 0.3,
                "timeline_estimate": 5,
                "risk_factors": ["regression", "edge_cases", "maintenance_overhead"]
            },
            ImprovementType.SAFETY_ENHANCEMENT: {
                "expected_benefits": {"safety_score": 0.25, "reliability": 0.20, "transparency": 0.15},
                "implementation_cost": 0.6,
                "timeline_estimate": 10,
                "risk_factors": ["performance_impact", "complexity_increase"]
            }
        }
    
    
    def _initialize_weakness_patterns(self):
        """Initialize patterns for identifying weakness types"""
        self.weakness_patterns = {
            "latency_issues": ["high_response_time", "slow_processing", "bottleneck_detected"],
            "accuracy_problems": ["low_precision", "poor_recall", "classification_errors"],
            "resource_constraints": ["memory_pressure", "cpu_saturation", "storage_limits"],
            "scalability_limits": ["concurrent_user_limit", "throughput_plateau", "load_balancing_issues"],
            "safety_concerns": ["alignment_drift", "unexpected_behavior", "safety_violations"],
            "maintenance_issues": ["code_complexity", "technical_debt", "documentation_gaps"]
        }
    
    
    async def _generate_proposals_for_weakness(self, weakness: str, performance_issues: Dict[str, Any],
                                             resource_constraints: Dict[str, Any]) -> List[ImprovementProposal]:
        """Generate proposals targeted at a specific weakness"""
        proposals = []
        
        try:
            # Determine improvement types based on weakness
            relevant_types = await self._map_weakness_to_improvement_types(weakness)
            
            for improvement_type in relevant_types:
                template = self.proposal_templates.get(improvement_type, {})
                
                # Create proposal based on template and weakness
                proposal = ImprovementProposal(
                    improvement_type=improvement_type,
                    target_component=weakness.get("component", "system"),
                    title=f"{improvement_type.value.title()} improvement for {weakness.get('category', 'unknown')}",
                    description=await self._generate_proposal_description(weakness, improvement_type),
                    technical_details=await self._generate_technical_details(weakness, improvement_type),
                    expected_benefits=template.get("expected_benefits", {}),
                    implementation_cost=template.get("implementation_cost", 0.5),
                    timeline_estimate=template.get("timeline_estimate", 7),
                    priority_score=await self._calculate_proposal_priority(weakness, improvement_type),
                    weakness_analysis={"target_weakness": weakness},
                    proposed_by="improvement_engine"
                )
                
                if proposal.priority_score >= PROPOSAL_PRIORITY_THRESHOLD:
                    proposals.append(proposal)
            
        except Exception as e:
            print(f"⚠️ Error generating proposals for weakness: {str(e)}")
        
        return proposals
    
    
    async def _generate_performance_proposals(self, component: str, issues: Dict[str, Any],
                                            analysis: Dict[str, Any]) -> List[ImprovementProposal]:
        """Generate proposals for performance bottlenecks"""
        proposals = []
        
        try:
            # Analyze the type of performance issues
            issue_types = issues.get("types", [])
            severity = issues.get("severity", 0.5)
            
            # Generate optimization proposals
            if "latency" in issue_types:
                proposal = ImprovementProposal(
                    improvement_type=ImprovementType.OPTIMIZATION,
                    target_component=component,
                    title=f"Latency optimization for {component}",
                    description=f"Optimize {component} to reduce response latency by implementing caching, query optimization, and async processing",
                    technical_details={
                        "optimization_targets": ["response_time", "processing_efficiency"],
                        "proposed_techniques": ["caching", "async_processing", "query_optimization"],
                        "estimated_improvement": f"{min(30, severity * 50):.1f}%"
                    },
                    expected_benefits={"latency": -0.2, "throughput": 0.15, "user_satisfaction": 0.1},
                    implementation_cost=0.4,
                    timeline_estimate=7,
                    priority_score=severity * 0.8,
                    weakness_analysis={"performance_issue": issues},
                    proposed_by="improvement_engine"
                )
                proposals.append(proposal)
            
            if "throughput" in issue_types:
                proposal = ImprovementProposal(
                    improvement_type=ImprovementType.ARCHITECTURE,
                    target_component=component,
                    title=f"Throughput enhancement for {component}",
                    description=f"Architectural improvements to increase {component} throughput through load balancing and parallel processing",
                    technical_details={
                        "architecture_changes": ["load_balancing", "parallel_processing", "resource_pooling"],
                        "scalability_targets": ["concurrent_requests", "processing_capacity"],
                        "estimated_improvement": f"{min(50, severity * 80):.1f}%"
                    },
                    expected_benefits={"throughput": 0.25, "scalability": 0.3, "resource_efficiency": 0.15},
                    implementation_cost=0.6,
                    timeline_estimate=14,
                    priority_score=severity * 0.9,
                    weakness_analysis={"performance_issue": issues},
                    proposed_by="improvement_engine"
                )
                proposals.append(proposal)
            
        except Exception as e:
            print(f"⚠️ Error generating performance proposals: {str(e)}")
        
        return proposals
    
    
    async def _generate_resource_proposals(self, resource: str, constraint_data: Dict[str, Any],
                                         analysis: Dict[str, Any]) -> List[ImprovementProposal]:
        """Generate proposals for resource constraints"""
        proposals = []
        
        try:
            utilization = constraint_data.get("utilization", 0.5)
            limit_reached = constraint_data.get("limit_reached", False)
            
            if limit_reached or utilization > 0.8:
                proposal = ImprovementProposal(
                    improvement_type=ImprovementType.OPTIMIZATION,
                    target_component=f"{resource}_management",
                    title=f"{resource.title()} optimization and efficiency improvement",
                    description=f"Optimize {resource} usage through better allocation, cleanup, and efficiency improvements",
                    technical_details={
                        "resource_type": resource,
                        "current_utilization": f"{utilization:.2%}",
                        "optimization_techniques": ["allocation_optimization", "cleanup_automation", "usage_monitoring"],
                        "target_reduction": f"{min(25, (utilization - 0.6) * 50):.1f}%"
                    },
                    expected_benefits={f"{resource}_efficiency": 0.2, "cost_reduction": 0.15, "stability": 0.1},
                    implementation_cost=0.3,
                    timeline_estimate=5,
                    priority_score=utilization * 0.8,
                    weakness_analysis={"resource_constraint": constraint_data},
                    proposed_by="improvement_engine"
                )
                proposals.append(proposal)
        
        except Exception as e:
            print(f"⚠️ Error generating resource proposals: {str(e)}")
        
        return proposals
    
    
    async def _map_weakness_to_improvement_types(self, weakness: Any) -> List[ImprovementType]:
        """Map weakness to relevant improvement types"""
        # Handle both dict and string weaknesses
        if isinstance(weakness, dict):
            weakness_category = weakness.get("category", "unknown")
        else:
            weakness_category = str(weakness)
        
        mapping = {
            "latency_issues": [ImprovementType.OPTIMIZATION, ImprovementType.ARCHITECTURE],
            "accuracy_problems": [ImprovementType.TRAINING_DATA, ImprovementType.HYPERPARAMETER],
            "resource_constraints": [ImprovementType.OPTIMIZATION, ImprovementType.MODEL_SIZE],
            "scalability_limits": [ImprovementType.ARCHITECTURE, ImprovementType.OPTIMIZATION],
            "safety_concerns": [ImprovementType.SAFETY_ENHANCEMENT, ImprovementType.ARCHITECTURE],
            "maintenance_issues": [ImprovementType.ARCHITECTURE, ImprovementType.OPTIMIZATION]
        }
        
        return mapping.get(weakness_category, [ImprovementType.OPTIMIZATION])
    
    
    async def _generate_proposal_description(self, weakness: Any, improvement_type: ImprovementType) -> str:
        """Generate detailed description for a proposal"""
        if isinstance(weakness, dict):
            weakness_desc = weakness.get("description", str(weakness))
        else:
            weakness_desc = str(weakness)
        
        descriptions = {
            ImprovementType.ARCHITECTURE: f"Architectural refactoring to address {weakness_desc} through improved design patterns and system structure",
            ImprovementType.HYPERPARAMETER: f"Hyperparameter optimization to resolve {weakness_desc} with better model configuration",
            ImprovementType.TRAINING_DATA: f"Training data enhancement to improve {weakness_desc} through better data quality and coverage",
            ImprovementType.MODEL_SIZE: f"Model size optimization to address {weakness_desc} while maintaining performance",
            ImprovementType.OPTIMIZATION: f"Performance optimization to eliminate {weakness_desc} through algorithmic improvements",
            ImprovementType.SAFETY_ENHANCEMENT: f"Safety enhancement to mitigate {weakness_desc} with better monitoring and controls"
        }
        
        return descriptions.get(improvement_type, f"Improvement to address {weakness_desc}")
    
    
    async def _generate_technical_details(self, weakness: Any, improvement_type: ImprovementType) -> Dict[str, Any]:
        """Generate technical implementation details"""
        base_details = {
            "improvement_approach": improvement_type.value,
            "target_weakness": str(weakness),
            "implementation_strategy": "incremental_rollout",
            "testing_requirements": ["unit_tests", "integration_tests", "performance_tests"],
            "rollback_plan": "automated_rollback_on_failure"
        }
        
        type_specific = {
            ImprovementType.ARCHITECTURE: {
                "design_patterns": ["microservices", "event_driven", "caching_layers"],
                "refactoring_scope": "module_level"
            },
            ImprovementType.HYPERPARAMETER: {
                "optimization_method": "bayesian_optimization",
                "parameter_space": "learning_rate_batch_size_architecture"
            },
            ImprovementType.TRAINING_DATA: {
                "data_augmentation": ["synthetic_generation", "noise_injection"],
                "quality_improvements": ["deduplication", "bias_reduction"]
            },
            ImprovementType.OPTIMIZATION: {
                "optimization_targets": ["algorithmic_complexity", "memory_usage", "io_operations"],
                "profiling_required": True
            }
        }
        
        base_details.update(type_specific.get(improvement_type, {}))
        return base_details
    
    
    async def _calculate_proposal_priority(self, weakness: Any, improvement_type: ImprovementType) -> float:
        """Calculate priority score for a proposal"""
        # Base priority on weakness severity
        if isinstance(weakness, dict):
            severity = weakness.get("severity", 0.5)
            impact = weakness.get("impact", 0.5)
        else:
            severity = 0.5
            impact = 0.5
        
        # Adjust based on improvement type complexity
        type_multipliers = {
            ImprovementType.HYPERPARAMETER: 1.2,  # Quick wins
            ImprovementType.OPTIMIZATION: 1.1,
            ImprovementType.MODEL_SIZE: 1.0,
            ImprovementType.TRAINING_DATA: 0.9,
            ImprovementType.ARCHITECTURE: 0.8,    # More complex
            ImprovementType.SAFETY_ENHANCEMENT: 0.9
        }
        
        multiplier = type_multipliers.get(improvement_type, 1.0)
        priority = (severity * 0.6 + impact * 0.4) * multiplier
        
        return min(1.0, max(0.0, priority))
    
    
    async def _get_baseline_performance(self, component: str) -> Dict[str, float]:
        """Get baseline performance metrics for a component"""
        # Simulate baseline performance data
        # In a real system, this would fetch actual metrics from the performance monitor
        return {
            "latency": random.uniform(100, 500),  # ms
            "throughput": random.uniform(10, 100),  # requests/sec
            "accuracy": random.uniform(0.7, 0.95),  # percentage
            "resource_usage": random.uniform(0.3, 0.8),  # percentage
            "error_rate": random.uniform(0.01, 0.1)  # percentage
        }
    
    
    async def _simulate_performance_changes(self, proposal: ImprovementProposal, 
                                          baseline: Dict[str, float]) -> Dict[str, float]:
        """Simulate performance changes from implementing a proposal"""
        changes = {}
        
        # Apply expected benefits with some variance
        for metric, benefit in proposal.expected_benefits.items():
            if metric in baseline:
                # Add realistic variance to predictions
                variance = random.uniform(-PERFORMANCE_VARIANCE_FACTOR, PERFORMANCE_VARIANCE_FACTOR)
                predicted_change = benefit + variance
                
                # Calculate absolute change
                if metric in ["latency", "error_rate", "resource_usage"]:
                    # Lower is better for these metrics
                    changes[metric] = baseline[metric] * (1 + predicted_change)
                else:
                    # Higher is better
                    changes[metric] = baseline[metric] * (1 + predicted_change)
            else:
                # New metric introduced by the improvement
                changes[metric] = benefit
        
        return changes
    
    
    async def _calculate_resource_requirements(self, proposal: ImprovementProposal) -> Dict[str, float]:
        """Calculate resource requirements for implementing a proposal"""
        base_requirements = {
            "cpu_hours": proposal.implementation_cost * 100,
            "memory_gb": proposal.implementation_cost * 10,
            "storage_gb": proposal.implementation_cost * 5,
            "network_bandwidth": proposal.implementation_cost * 1.0,
            "developer_hours": proposal.timeline_estimate * 8
        }
        
        # Adjust based on improvement type
        type_multipliers = {
            ImprovementType.ARCHITECTURE: 1.5,
            ImprovementType.SAFETY_ENHANCEMENT: 1.3,
            ImprovementType.TRAINING_DATA: 1.2,
            ImprovementType.OPTIMIZATION: 0.8,
            ImprovementType.HYPERPARAMETER: 0.6,
            ImprovementType.MODEL_SIZE: 0.7
        }
        
        multiplier = type_multipliers.get(proposal.improvement_type, 1.0)
        
        return {k: v * multiplier for k, v in base_requirements.items()}
    
    
    async def _assess_implementation_risks(self, proposal: ImprovementProposal) -> Dict[str, float]:
        """Assess risks associated with implementing a proposal"""
        base_risks = {
            "performance_regression": 0.2,
            "compatibility_issues": 0.15,
            "implementation_complexity": proposal.implementation_cost,
            "timeline_risk": min(0.5, proposal.timeline_estimate / 30),
            "resource_constraints": 0.1,
            "rollback_difficulty": 0.2
        }
        
        # Adjust based on improvement type
        type_risk_adjustments = {
            ImprovementType.ARCHITECTURE: {"implementation_complexity": 0.3, "rollback_difficulty": 0.4},
            ImprovementType.SAFETY_ENHANCEMENT: {"compatibility_issues": 0.2, "performance_regression": 0.1},
            ImprovementType.TRAINING_DATA: {"timeline_risk": 0.2, "resource_constraints": 0.3},
            ImprovementType.OPTIMIZATION: {"performance_regression": 0.3},
            ImprovementType.HYPERPARAMETER: {"implementation_complexity": -0.2, "timeline_risk": -0.1}
        }
        
        adjustments = type_risk_adjustments.get(proposal.improvement_type, {})
        for risk, adjustment in adjustments.items():
            if risk in base_risks:
                base_risks[risk] = max(0.0, min(1.0, base_risks[risk] + adjustment))
        
        return base_risks
    
    
    async def _calculate_simulation_confidence(self, proposal: ImprovementProposal,
                                             predicted_changes: Dict[str, float],
                                             baseline: Dict[str, float]) -> float:
        """Calculate confidence in simulation results"""
        confidence_factors = []
        
        # Factor 1: Proposal implementation cost (lower cost = higher confidence)
        cost_confidence = 1.0 - proposal.implementation_cost
        confidence_factors.append(cost_confidence)
        
        # Factor 2: Number of predicted metrics (more metrics = more uncertainty)
        metrics_confidence = max(0.5, 1.0 - len(predicted_changes) * 0.1)
        confidence_factors.append(metrics_confidence)
        
        # Factor 3: Magnitude of predicted changes (smaller changes = higher confidence)
        if predicted_changes and baseline:
            change_magnitudes = []
            for metric, predicted in predicted_changes.items():
                if metric in baseline and baseline[metric] != 0:
                    change_magnitude = abs((predicted - baseline[metric]) / baseline[metric])
                    change_magnitudes.append(change_magnitude)
            
            if change_magnitudes:
                avg_change = statistics.mean(change_magnitudes)
                change_confidence = max(0.3, 1.0 - avg_change)
                confidence_factors.append(change_confidence)
        
        # Factor 4: Improvement type reliability
        type_confidence = {
            ImprovementType.HYPERPARAMETER: 0.9,
            ImprovementType.OPTIMIZATION: 0.8,
            ImprovementType.MODEL_SIZE: 0.8,
            ImprovementType.TRAINING_DATA: 0.7,
            ImprovementType.SAFETY_ENHANCEMENT: 0.7,
            ImprovementType.ARCHITECTURE: 0.6
        }
        confidence_factors.append(type_confidence.get(proposal.improvement_type, 0.7))
        
        return statistics.mean(confidence_factors)
    
    
    async def _validate_simulation_results(self, proposal: ImprovementProposal,
                                         predicted_changes: Dict[str, float],
                                         resource_requirements: Dict[str, float]) -> Dict[str, Any]:
        """Validate simulation results for consistency and feasibility"""
        validation_results = {
            "consistency_check": True,
            "feasibility_check": True,
            "resource_validation": True,
            "performance_validation": True,
            "warnings": [],
            "recommendations": []
        }
        
        # Check for unrealistic performance improvements
        for metric, value in predicted_changes.items():
            if metric in ["accuracy", "throughput"] and value > 2.0:  # 100% improvement
                validation_results["warnings"].append(f"Unrealistic {metric} improvement: {value:.1%}")
                validation_results["performance_validation"] = False
        
        # Check resource requirements reasonableness
        if resource_requirements.get("developer_hours", 0) > 1000:
            validation_results["warnings"].append("High developer hour requirement may indicate scope creep")
            validation_results["resource_validation"] = False
        
        # Check implementation timeline
        if proposal.timeline_estimate > 30:  # More than 30 days
            validation_results["recommendations"].append("Consider breaking down into smaller phases")
        
        return validation_results
    
    
    async def _analyze_safety_risks(self, proposal: ImprovementProposal) -> List[str]:
        """Analyze potential safety risks of a proposal"""
        risks = []
        
        # Common risks for all improvements
        base_risks = [
            "Potential for unintended behavioral changes",
            "Risk of introducing new failure modes"
        ]
        
        # Type-specific risks
        type_risks = {
            ImprovementType.ARCHITECTURE: [
                "System instability during transition",
                "Data consistency issues during migration",
                "Service availability impact"
            ],
            ImprovementType.TRAINING_DATA: [
                "Introduction of data bias",
                "Model behavior drift",
                "Privacy and security concerns"
            ],
            ImprovementType.HYPERPARAMETER: [
                "Training instability",
                "Model convergence issues",
                "Performance degradation"
            ],
            ImprovementType.OPTIMIZATION: [
                "Edge case handling failures",
                "Performance regression in specific scenarios",
                "Resource exhaustion under load"
            ],
            ImprovementType.SAFETY_ENHANCEMENT: [
                "Over-restrictive behavior",
                "Performance impact from safety checks",
                "False positive safety alerts"
            ],
            ImprovementType.MODEL_SIZE: [
                "Accuracy degradation",
                "Compatibility issues with existing systems",
                "Inference quality reduction"
            ]
        }
        
        risks.extend(base_risks)
        risks.extend(type_risks.get(proposal.improvement_type, []))
        
        # Add risks based on implementation cost
        if proposal.implementation_cost > 0.7:
            risks.append("High complexity increases risk of implementation errors")
        
        # Add risks based on timeline
        if proposal.timeline_estimate > 14:
            risks.append("Extended timeline increases risk of scope drift")
        
        return risks
    
    
    async def _generate_mitigation_strategies(self, proposal: ImprovementProposal, 
                                            risks: List[str]) -> List[str]:
        """Generate risk mitigation strategies"""
        strategies = []
        
        # General mitigation strategies
        strategies.extend([
            "Implement comprehensive testing suite",
            "Use gradual rollout with monitoring",
            "Prepare automated rollback procedures",
            "Conduct thorough code review process"
        ])
        
        # Risk-specific strategies
        for risk in risks:
            if "instability" in risk.lower():
                strategies.append("Implement canary deployments")
            elif "bias" in risk.lower():
                strategies.append("Conduct bias detection and fairness testing")
            elif "performance" in risk.lower():
                strategies.append("Establish performance monitoring and alerting")
            elif "convergence" in risk.lower():
                strategies.append("Use validation metrics to monitor training stability")
            elif "compatibility" in risk.lower():
                strategies.append("Maintain backward compatibility layers")
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(strategies))
    
    
    async def _assess_circuit_breaker_impact(self, proposal: ImprovementProposal) -> Dict[str, Any]:
        """Assess how the proposal might impact circuit breaker systems"""
        impact = {
            "affected_systems": [],
            "threshold_adjustments_needed": False,
            "new_monitoring_required": False,
            "safety_score_impact": 0.0
        }
        
        # Determine which systems might be affected
        if proposal.improvement_type == ImprovementType.ARCHITECTURE:
            impact["affected_systems"] = ["routing", "load_balancing", "service_mesh"]
            impact["threshold_adjustments_needed"] = True
        elif proposal.improvement_type == ImprovementType.SAFETY_ENHANCEMENT:
            impact["affected_systems"] = ["safety_monitoring", "threat_detection"]
            impact["new_monitoring_required"] = True
            impact["safety_score_impact"] = 0.1
        elif proposal.improvement_type == ImprovementType.OPTIMIZATION:
            impact["affected_systems"] = ["performance_monitoring"]
            impact["threshold_adjustments_needed"] = True
        
        # High-cost implementations affect more systems
        if proposal.implementation_cost > 0.6:
            impact["new_monitoring_required"] = True
        
        return impact
    
    
    async def _determine_governance_requirements(self, proposal: ImprovementProposal, 
                                               risks: List[str]) -> List[str]:
        """Determine governance and approval requirements"""
        requirements = []
        
        # Requirements based on improvement type
        if proposal.improvement_type == ImprovementType.SAFETY_ENHANCEMENT:
            requirements.append("Safety committee review")
        
        if proposal.improvement_type == ImprovementType.ARCHITECTURE:
            requirements.append("Architecture review board approval")
        
        # Requirements based on cost and timeline
        if proposal.implementation_cost > 0.7:
            requirements.append("Resource allocation approval")
        
        if proposal.timeline_estimate > 14:
            requirements.append("Project management oversight")
        
        # Requirements based on risks
        high_risk_indicators = ["instability", "security", "privacy", "safety"]
        for risk in risks:
            if any(indicator in risk.lower() for indicator in high_risk_indicators):
                requirements.append("Security and safety review")
                break
        
        # Requirements based on target component
        critical_components = ["safety", "consensus", "authentication", "tokenomics"]
        if any(component in proposal.target_component.lower() for component in critical_components):
            requirements.append("Critical system change approval")
        
        return list(set(requirements))  # Remove duplicates
    
    
    async def _calculate_safety_score(self, proposal: ImprovementProposal, 
                                    risks: List[str], circuit_breaker_impact: Dict[str, Any]) -> float:
        """Calculate overall safety score for a proposal"""
        safety_factors = []
        
        # Factor 1: Number of identified risks (fewer = safer)
        risk_factor = max(0.0, 1.0 - len(risks) * 0.1)
        safety_factors.append(risk_factor)
        
        # Factor 2: Implementation cost (lower = safer)
        cost_factor = 1.0 - proposal.implementation_cost
        safety_factors.append(cost_factor)
        
        # Factor 3: Improvement type safety
        type_safety = {
            ImprovementType.HYPERPARAMETER: 0.9,
            ImprovementType.OPTIMIZATION: 0.8,
            ImprovementType.MODEL_SIZE: 0.8,
            ImprovementType.TRAINING_DATA: 0.7,
            ImprovementType.SAFETY_ENHANCEMENT: 0.9,  # Paradoxically safe
            ImprovementType.ARCHITECTURE: 0.5
        }
        safety_factors.append(type_safety.get(proposal.improvement_type, 0.7))
        
        # Factor 4: Circuit breaker impact
        cb_impact = circuit_breaker_impact.get("safety_score_impact", 0.0)
        cb_factor = 0.8 + cb_impact  # Baseline 0.8, adjusted by impact
        safety_factors.append(max(0.0, min(1.0, cb_factor)))
        
        # Factor 5: Target component criticality
        critical_components = ["safety", "consensus", "authentication"]
        if any(comp in proposal.target_component.lower() for comp in critical_components):
            safety_factors.append(0.6)  # More scrutiny for critical components
        else:
            safety_factors.append(0.8)
        
        return statistics.mean(safety_factors)


# === Global Proposal Engine Instance ===

_proposal_engine_instance: Optional[ImprovementProposalEngine] = None

def get_proposal_engine() -> ImprovementProposalEngine:
    """Get or create the global proposal engine instance"""
    global _proposal_engine_instance
    if _proposal_engine_instance is None:
        _proposal_engine_instance = ImprovementProposalEngine()
    return _proposal_engine_instance