"""
Liquidity-Provenance Enforcement
===============================

Prevents "model squatting" and provenance hoarding by requiring actual usage
and accessibility. Ensures that models earning royalties are genuinely 
contributing to the ecosystem, not just sitting idle for exclusivity.

Key Features:
- Time-locked access that decays if models aren't used
- Open distillation mandates for FTNS eligibility  
- Model usage audits with verifiable logs
- Anti-hoarding penalties and forced availability
- Liquidity requirements for maintaining royalty streams
"""

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass

from pydantic import BaseModel, Field


class AccessibilityStatus(str, Enum):
    """Model accessibility status"""
    FULLY_OPEN = "fully_open"          # Completely open access
    API_ACCESSIBLE = "api_accessible"   # Available via API endpoints
    RESTRICTED_ACCESS = "restricted_access"  # Limited access with approval
    EMBARGOED = "embargoed"            # Temporarily restricted for competitive reasons
    HOARDED = "hoarded"                # Flagged as being artificially restricted


class UsageRequirement(str, Enum):
    """Types of usage requirements for maintaining royalties"""
    MINIMUM_API_CALLS = "minimum_api_calls"        # Minimum API usage per period
    DISTILLATION_AVAILABILITY = "distillation_availability"  # Must allow distillation
    EMBEDDING_ACCESS = "embedding_access"          # Must provide embedding access
    COMMUNITY_USAGE = "community_usage"           # Community must be able to use it
    EDUCATIONAL_ACCESS = "educational_access"      # Educational institutions must have access


@dataclass
class UsageMetrics:
    """Metrics tracking model usage and accessibility"""
    total_api_calls: int
    unique_users: int
    distillation_requests: int
    embedding_accesses: int
    community_usage_hours: float
    educational_usage_sessions: int
    last_usage_timestamp: datetime
    
    # Accessibility metrics
    api_uptime_percentage: float
    average_response_time_ms: float
    access_denial_rate: float


class LiquidityRequirement(BaseModel):
    """Liquidity requirements for maintaining royalty eligibility"""
    model_id: UUID
    requirement_type: UsageRequirement
    
    # Thresholds
    minimum_daily_usage: int = 0
    minimum_weekly_unique_users: int = 0
    minimum_monthly_distillations: int = 0
    maximum_access_denial_rate: float = 0.1  # 10% max denial rate
    
    # Grace periods
    grace_period_days: int = 30
    warning_period_days: int = 7
    
    # Penalties
    royalty_reduction_percentage: float = 0.0
    forced_availability_duration_days: int = 0


class ModelAvailabilityAudit(BaseModel):
    """Audit record for model availability and usage"""
    audit_id: UUID = Field(default_factory=uuid4)
    model_id: UUID
    audit_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Availability metrics
    accessibility_status: AccessibilityStatus
    usage_metrics: UsageMetrics
    
    # Compliance assessment
    requirements_met: List[UsageRequirement] = Field(default_factory=list)
    violations_found: List[str] = Field(default_factory=list)
    
    # Actions taken
    penalties_applied: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class LiquidityProvenanceEnforcement:
    """
    Enforcement system that prevents model hoarding and ensures genuine
    accessibility for models earning provenance royalties.
    
    Implements the principle: "If you want to earn from the commons,
    you must contribute to the commons."
    """
    
    def __init__(self):
        # Model tracking
        self.registered_models: Dict[UUID, Dict[str, Any]] = {}
        self.liquidity_requirements: Dict[UUID, List[LiquidityRequirement]] = {}
        self.usage_tracking: Dict[UUID, List[UsageMetrics]] = {}
        self.audit_history: List[ModelAvailabilityAudit] = []
        
        # Default requirements
        self.default_requirements = {
            UsageRequirement.MINIMUM_API_CALLS: {
                "daily_threshold": 10,
                "grace_period_days": 30
            },
            UsageRequirement.DISTILLATION_AVAILABILITY: {
                "monthly_threshold": 1,
                "grace_period_days": 60
            },
            UsageRequirement.COMMUNITY_USAGE: {
                "weekly_hours_threshold": 5,
                "grace_period_days": 14
            }
        }
        
        # Penalty schedules
        self.penalty_schedule = {
            "first_violation": 0.1,    # 10% royalty reduction
            "second_violation": 0.25,  # 25% royalty reduction
            "third_violation": 0.5,    # 50% royalty reduction
            "chronic_violation": 0.9   # 90% royalty reduction
        }
        
        print("🔄 Liquidity-Provenance Enforcement initialized")
        print("   - Anti-hoarding mechanisms active")
        print("   - Usage auditing enabled")
        print("   - Forced availability protocols ready")
    
    async def register_model_for_tracking(self,
                                        model_id: UUID,
                                        model_metadata: Dict[str, Any],
                                        custom_requirements: List[LiquidityRequirement] = None) -> Dict[str, Any]:
        """
        Register a model for liquidity-provenance tracking with usage requirements.
        """
        
        # Set up default requirements if not provided
        requirements = custom_requirements or self._generate_default_requirements(model_id, model_metadata)
        
        # Register model
        registration = {
            "model_id": model_id,
            "metadata": model_metadata,
            "registration_timestamp": datetime.now(timezone.utc),
            "status": "active",
            "violation_count": 0,
            "last_audit": None
        }
        
        self.registered_models[model_id] = registration
        self.liquidity_requirements[model_id] = requirements
        self.usage_tracking[model_id] = []
        
        print(f"📊 Model registered for liquidity tracking: {model_id}")
        print(f"   - Requirements: {len(requirements)}")
        print(f"   - Next audit in: 24 hours")
        
        return {
            "registration_successful": True,
            "requirements": requirements,
            "grace_period_ends": datetime.now(timezone.utc) + timedelta(days=30),
            "next_audit_date": datetime.now(timezone.utc) + timedelta(days=1)
        }
    
    async def record_model_usage(self,
                               model_id: UUID,
                               usage_metrics: UsageMetrics) -> bool:
        """
        Record usage metrics for a model to track liquidity compliance.
        """
        
        if model_id not in self.registered_models:
            print(f"⚠️ Attempted to record usage for unregistered model: {model_id}")
            return False
        
        # Add to usage tracking
        self.usage_tracking[model_id].append(usage_metrics)
        
        # Keep only last 90 days of usage data
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=90)
        self.usage_tracking[model_id] = [
            metrics for metrics in self.usage_tracking[model_id]
            if metrics.last_usage_timestamp >= cutoff_date
        ]
        
        return True
    
    async def perform_liquidity_audit(self, model_id: UUID) -> ModelAvailabilityAudit:
        """
        Perform comprehensive liquidity and availability audit for a model.
        """
        
        if model_id not in self.registered_models:
            raise ValueError(f"Model {model_id} not registered for tracking")
        
        model_info = self.registered_models[model_id]
        requirements = self.liquidity_requirements[model_id]
        usage_history = self.usage_tracking[model_id]
        
        # Assess current accessibility status
        accessibility_status = await self._assess_accessibility_status(model_id, usage_history)
        
        # Check compliance with each requirement
        requirements_met = []
        violations_found = []
        
        for requirement in requirements:
            is_compliant = await self._check_requirement_compliance(
                requirement, usage_history
            )
            
            if is_compliant:
                requirements_met.append(requirement.requirement_type)
            else:
                violations_found.append(
                    f"Failed {requirement.requirement_type}: below threshold"
                )
        
        # Determine penalties and actions
        penalties_applied = []
        recommendations = []
        
        if violations_found:
            penalties = await self._calculate_penalties(model_id, violations_found)
            penalties_applied.extend(penalties)
            
            # Generate recommendations
            recommendations = await self._generate_compliance_recommendations(
                model_id, violations_found, usage_history
            )
        
        # Create audit record
        latest_usage = usage_history[-1] if usage_history else self._create_empty_usage_metrics()
        
        audit = ModelAvailabilityAudit(
            model_id=model_id,
            accessibility_status=accessibility_status,
            usage_metrics=latest_usage,
            requirements_met=requirements_met,
            violations_found=violations_found,
            penalties_applied=penalties_applied,
            recommendations=recommendations
        )
        
        # Update model status
        model_info["last_audit"] = audit.audit_timestamp
        if violations_found:
            model_info["violation_count"] += 1
        
        # Store audit
        self.audit_history.append(audit)
        
        print(f"🔍 Liquidity audit completed: {model_id}")
        print(f"   - Status: {accessibility_status}")
        print(f"   - Requirements met: {len(requirements_met)}/{len(requirements)}")
        print(f"   - Violations: {len(violations_found)}")
        
        return audit
    
    async def enforce_availability_requirements(self,
                                              model_id: UUID,
                                              violations: List[str]) -> Dict[str, Any]:
        """
        Enforce availability requirements for models with violations.
        """
        
        enforcement_actions = []
        
        # Determine severity of violations
        violation_count = self.registered_models[model_id]["violation_count"]
        
        if violation_count >= 3:  # Chronic violations
            # Force full availability for 90 days
            enforcement_actions.append({
                "action": "forced_full_availability",
                "duration_days": 90,
                "royalty_reduction": self.penalty_schedule["chronic_violation"]
            })
        elif violation_count >= 2:  # Repeat violations
            # Force API availability for 60 days
            enforcement_actions.append({
                "action": "forced_api_availability", 
                "duration_days": 60,
                "royalty_reduction": self.penalty_schedule["second_violation"]
            })
        else:  # First violation
            # Warning with minor penalty
            enforcement_actions.append({
                "action": "warning_with_penalty",
                "duration_days": 30,
                "royalty_reduction": self.penalty_schedule["first_violation"]
            })
        
        # Apply enforcement actions
        for action in enforcement_actions:
            await self._apply_enforcement_action(model_id, action)
        
        return {
            "model_id": model_id,
            "enforcement_actions": enforcement_actions,
            "effective_immediately": True,
            "appeal_deadline": datetime.now(timezone.utc) + timedelta(days=14)
        }
    
    async def analyze_ecosystem_liquidity(self) -> Dict[str, Any]:
        """
        Analyze overall ecosystem liquidity and identify hoarding patterns.
        """
        
        total_models = len(self.registered_models)
        
        # Status distribution
        status_distribution = {}
        violation_distribution = {}
        
        for model_info in self.registered_models.values():
            # Get latest audit
            model_id = model_info["model_id"]
            latest_audit = next(
                (audit for audit in reversed(self.audit_history) if audit.model_id == model_id),
                None
            )
            
            if latest_audit:
                status = latest_audit.accessibility_status
                status_distribution[status] = status_distribution.get(status, 0) + 1
                
                violation_count = len(latest_audit.violations_found)
                violation_distribution[violation_count] = violation_distribution.get(violation_count, 0) + 1
        
        # Calculate liquidity health metrics
        healthy_models = (
            status_distribution.get(AccessibilityStatus.FULLY_OPEN, 0) +
            status_distribution.get(AccessibilityStatus.API_ACCESSIBLE, 0)
        )
        
        liquidity_health_score = healthy_models / total_models if total_models > 0 else 0
        
        # Identify hoarding patterns
        hoarded_models = status_distribution.get(AccessibilityStatus.HOARDED, 0)
        hoarding_risk_score = hoarded_models / total_models if total_models > 0 else 0
        
        return {
            "total_models_tracked": total_models,
            "status_distribution": status_distribution,
            "violation_distribution": violation_distribution,
            "liquidity_health_score": liquidity_health_score,
            "hoarding_risk_score": hoarding_risk_score,
            "enforcement_actions_last_30_days": len([
                audit for audit in self.audit_history
                if audit.audit_timestamp >= datetime.now(timezone.utc) - timedelta(days=30)
                and audit.penalties_applied
            ]),
            "recommendations": self._generate_ecosystem_recommendations(
                liquidity_health_score, hoarding_risk_score
            )
        }
    
    def _generate_default_requirements(self,
                                     model_id: UUID,
                                     metadata: Dict[str, Any]) -> List[LiquidityRequirement]:
        """Generate default liquidity requirements based on model type and scale"""
        
        requirements = []
        
        # Minimum API availability requirement
        requirements.append(LiquidityRequirement(
            model_id=model_id,
            requirement_type=UsageRequirement.MINIMUM_API_CALLS,
            minimum_daily_usage=10,
            grace_period_days=30,
            royalty_reduction_percentage=10.0
        ))
        
        # Distillation availability for large models
        if metadata.get("parameter_count", 0) > 1_000_000_000:  # 1B+ parameters
            requirements.append(LiquidityRequirement(
                model_id=model_id,
                requirement_type=UsageRequirement.DISTILLATION_AVAILABILITY,
                minimum_monthly_distillations=1,
                grace_period_days=60,
                royalty_reduction_percentage=25.0
            ))
        
        # Community usage requirement
        requirements.append(LiquidityRequirement(
            model_id=model_id,
            requirement_type=UsageRequirement.COMMUNITY_USAGE,
            minimum_weekly_unique_users=5,
            grace_period_days=14,
            royalty_reduction_percentage=15.0
        ))
        
        return requirements
    
    async def _assess_accessibility_status(self,
                                         model_id: UUID,
                                         usage_history: List[UsageMetrics]) -> AccessibilityStatus:
        """Assess current accessibility status of a model"""
        
        if not usage_history:
            return AccessibilityStatus.HOARDED
        
        latest_usage = usage_history[-1]
        
        # Check if model is being actively used
        days_since_last_use = (datetime.now(timezone.utc) - latest_usage.last_usage_timestamp).days
        
        if days_since_last_use > 30:
            return AccessibilityStatus.HOARDED
        
        # Check access patterns
        if latest_usage.access_denial_rate > 0.5:  # High denial rate
            return AccessibilityStatus.RESTRICTED_ACCESS
        elif latest_usage.api_uptime_percentage > 0.95 and latest_usage.access_denial_rate < 0.1:
            return AccessibilityStatus.FULLY_OPEN
        elif latest_usage.api_uptime_percentage > 0.8:
            return AccessibilityStatus.API_ACCESSIBLE
        else:
            return AccessibilityStatus.RESTRICTED_ACCESS
    
    async def _check_requirement_compliance(self,
                                          requirement: LiquidityRequirement,
                                          usage_history: List[UsageMetrics]) -> bool:
        """Check if a model meets a specific liquidity requirement"""
        
        if not usage_history:
            return False
        
        # Check based on requirement type
        if requirement.requirement_type == UsageRequirement.MINIMUM_API_CALLS:
            # Check daily API calls over last week
            recent_usage = [
                metrics for metrics in usage_history
                if (datetime.now(timezone.utc) - metrics.last_usage_timestamp).days <= 7
            ]
            
            if not recent_usage:
                return False
            
            avg_daily_calls = sum(metrics.total_api_calls for metrics in recent_usage) / len(recent_usage)
            return avg_daily_calls >= requirement.minimum_daily_usage
        
        elif requirement.requirement_type == UsageRequirement.DISTILLATION_AVAILABILITY:
            # Check monthly distillation requests
            monthly_usage = [
                metrics for metrics in usage_history
                if (datetime.now(timezone.utc) - metrics.last_usage_timestamp).days <= 30
            ]
            
            if not monthly_usage:
                return False
            
            total_distillations = sum(metrics.distillation_requests for metrics in monthly_usage)
            return total_distillations >= requirement.minimum_monthly_distillations
        
        elif requirement.requirement_type == UsageRequirement.COMMUNITY_USAGE:
            # Check weekly unique users
            weekly_usage = [
                metrics for metrics in usage_history
                if (datetime.now(timezone.utc) - metrics.last_usage_timestamp).days <= 7
            ]
            
            if not weekly_usage:
                return False
            
            max_weekly_users = max(metrics.unique_users for metrics in weekly_usage)
            return max_weekly_users >= requirement.minimum_weekly_unique_users
        
        return False
    
    async def _calculate_penalties(self,
                                 model_id: UUID,
                                 violations: List[str]) -> List[str]:
        """Calculate penalties for liquidity violations"""
        
        penalties = []
        violation_count = self.registered_models[model_id]["violation_count"]
        
        # Royalty reduction
        if violation_count == 0:
            reduction = self.penalty_schedule["first_violation"]
            penalties.append(f"10% royalty reduction for {len(violations)} violations")
        elif violation_count == 1:
            reduction = self.penalty_schedule["second_violation"]
            penalties.append(f"25% royalty reduction for repeat violations")
        elif violation_count >= 2:
            reduction = self.penalty_schedule["chronic_violation"]
            penalties.append(f"90% royalty reduction for chronic violations")
        
        # Forced availability
        if violation_count >= 1:
            penalties.append("Forced API availability for 30 days")
        
        if violation_count >= 2:
            penalties.append("Forced full availability for 60 days")
        
        return penalties
    
    async def _generate_compliance_recommendations(self,
                                                 model_id: UUID,
                                                 violations: List[str],
                                                 usage_history: List[UsageMetrics]) -> List[str]:
        """Generate recommendations for improving compliance"""
        
        recommendations = []
        
        if "minimum_api_calls" in str(violations):
            recommendations.append("Increase API availability and reduce access restrictions")
            recommendations.append("Implement automatic scaling to handle more requests")
        
        if "distillation_availability" in str(violations):
            recommendations.append("Enable distillation API endpoints")
            recommendations.append("Provide embedding access for research purposes")
        
        if "community_usage" in str(violations):
            recommendations.append("Create educational access programs")
            recommendations.append("Reduce barriers to community usage")
            recommendations.append("Implement tiered access with free community tier")
        
        # General recommendations
        recommendations.append("Monitor usage metrics daily")
        recommendations.append("Set up automated alerts for low usage periods")
        
        return recommendations
    
    def _create_empty_usage_metrics(self) -> UsageMetrics:
        """Create empty usage metrics for models with no usage data"""
        return UsageMetrics(
            total_api_calls=0,
            unique_users=0,
            distillation_requests=0,
            embedding_accesses=0,
            community_usage_hours=0.0,
            educational_usage_sessions=0,
            last_usage_timestamp=datetime.now(timezone.utc) - timedelta(days=365),
            api_uptime_percentage=0.0,
            average_response_time_ms=0.0,
            access_denial_rate=1.0
        )
    
    async def _apply_enforcement_action(self,
                                      model_id: UUID,
                                      action: Dict[str, Any]) -> bool:
        """Apply enforcement action to a model"""
        
        # In a real implementation, this would integrate with model hosting infrastructure
        # to actually enforce availability requirements
        
        print(f"🚨 Applying enforcement action: {action['action']} for model {model_id}")
        print(f"   - Duration: {action['duration_days']} days")
        print(f"   - Royalty reduction: {action['royalty_reduction']*100:.1f}%")
        
        return True
    
    def _generate_ecosystem_recommendations(self,
                                          liquidity_health_score: float,
                                          hoarding_risk_score: float) -> List[str]:
        """Generate ecosystem-level recommendations"""
        
        recommendations = []
        
        if liquidity_health_score < 0.7:
            recommendations.append("Strengthen liquidity requirements and enforcement")
            recommendations.append("Increase penalties for low accessibility")
        
        if hoarding_risk_score > 0.1:
            recommendations.append("Implement immediate availability mandates for hoarded models")
            recommendations.append("Review and tighten anti-hoarding policies")
        
        if liquidity_health_score > 0.9:
            recommendations.append("Consider relaxing some requirements for high-performing models")
        
        return recommendations


# Global liquidity-provenance enforcement instance
liquidity_provenance = LiquidityProvenanceEnforcement()