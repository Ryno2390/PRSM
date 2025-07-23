"""
PRSM Enterprise Cost Optimization & Resource Monitoring
Advanced cost optimization with real-time monitoring, budget management, and intelligent recommendations
"""

from typing import Dict, Any, List, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import json
import time
import logging
import uuid
import statistics
from collections import defaultdict, deque
import redis.asyncio as aioredis
from contextlib import asynccontextmanager

# Financial calculations
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    np = None

logger = logging.getLogger(__name__)


class CostCategory(Enum):
    """Cost categories for tracking"""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"
    MONITORING = "monitoring"
    SECURITY = "security"
    BACKUP = "backup"
    OTHER = "other"


class OptimizationType(Enum):
    """Types of cost optimizations"""
    RIGHTSIZING = "rightsizing"
    RESERVED_INSTANCES = "reserved_instances"
    SPOT_INSTANCES = "spot_instances"
    STORAGE_OPTIMIZATION = "storage_optimization"
    NETWORK_OPTIMIZATION = "network_optimization"
    RESOURCE_CONSOLIDATION = "resource_consolidation"
    SCHEDULED_SHUTDOWN = "scheduled_shutdown"
    REGION_OPTIMIZATION = "region_optimization"
    WASTE_ELIMINATION = "waste_elimination"


class BudgetPeriod(Enum):
    """Budget tracking periods"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"


class AlertSeverity(Enum):
    """Cost alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class CostMetrics:
    """Cost and resource metrics"""
    timestamp: datetime
    
    # Cost breakdown
    compute_cost: float = 0.0
    storage_cost: float = 0.0
    network_cost: float = 0.0
    database_cost: float = 0.0
    cache_cost: float = 0.0
    monitoring_cost: float = 0.0
    security_cost: float = 0.0
    backup_cost: float = 0.0
    other_cost: float = 0.0
    
    # Resource utilization
    compute_utilization_percent: float = 0.0
    storage_utilization_percent: float = 0.0
    network_utilization_percent: float = 0.0
    
    # Efficiency metrics
    cost_per_request: float = 0.0
    cost_per_user: float = 0.0
    cost_per_transaction: float = 0.0
    
    # Performance correlation
    requests_per_second: float = 0.0
    active_users: int = 0
    transactions_per_minute: float = 0.0
    
    @property
    def total_cost(self) -> float:
        """Calculate total cost"""
        return (self.compute_cost + self.storage_cost + self.network_cost +
                self.database_cost + self.cache_cost + self.monitoring_cost +
                self.security_cost + self.backup_cost + self.other_cost)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_cost": self.total_cost,
            "compute_cost": self.compute_cost,
            "storage_cost": self.storage_cost,
            "network_cost": self.network_cost,
            "database_cost": self.database_cost,
            "cache_cost": self.cache_cost,
            "monitoring_cost": self.monitoring_cost,
            "security_cost": self.security_cost,
            "backup_cost": self.backup_cost,
            "other_cost": self.other_cost,
            "compute_utilization_percent": self.compute_utilization_percent,
            "storage_utilization_percent": self.storage_utilization_percent,
            "network_utilization_percent": self.network_utilization_percent,
            "cost_per_request": self.cost_per_request,
            "cost_per_user": self.cost_per_user,
            "cost_per_transaction": self.cost_per_transaction,
            "requests_per_second": self.requests_per_second,
            "active_users": self.active_users,
            "transactions_per_minute": self.transactions_per_minute
        }


@dataclass
class BudgetLimit:
    """Budget limit configuration"""
    budget_id: str
    name: str
    description: str
    
    # Budget settings
    limit_amount: float
    period: BudgetPeriod
    categories: List[CostCategory] = field(default_factory=list)
    
    # Alert thresholds (percentages)
    warning_threshold: float = 75.0
    critical_threshold: float = 90.0
    emergency_threshold: float = 100.0
    
    # Enforcement
    enforce_limit: bool = False
    auto_shutdown_enabled: bool = False
    
    # Tracking
    current_spend: float = 0.0
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    enabled: bool = True
    
    @property
    def utilization_percent(self) -> float:
        """Calculate budget utilization percentage"""
        if self.limit_amount <= 0:
            return 0.0
        return (self.current_spend / self.limit_amount) * 100
    
    @property
    def remaining_budget(self) -> float:
        """Calculate remaining budget"""
        return max(0, self.limit_amount - self.current_spend)


@dataclass
class CostOptimizationRecommendation:
    """Cost optimization recommendation"""
    recommendation_id: str
    optimization_type: OptimizationType
    title: str
    description: str
    
    # Impact analysis
    estimated_savings_monthly: float
    estimated_savings_annual: float
    confidence: float  # 0.0-1.0
    
    # Implementation details
    effort_level: str  # "low", "medium", "high"
    risk_level: str    # "low", "medium", "high"
    implementation_time: str  # "immediate", "1-2 weeks", "1-2 months"
    
    # Resources involved
    affected_resources: List[str] = field(default_factory=list)
    categories: List[CostCategory] = field(default_factory=list)
    
    # Requirements
    prerequisites: List[str] = field(default_factory=list)
    potential_risks: List[str] = field(default_factory=list)
    
    # Tracking
    status: str = "pending"  # "pending", "approved", "implemented", "rejected"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Cost-benefit analysis
    implementation_cost: float = 0.0
    payback_period_months: Optional[float] = None
    
    def calculate_payback_period(self):
        """Calculate payback period in months"""
        if self.estimated_savings_monthly > 0 and self.implementation_cost > 0:
            self.payback_period_months = self.implementation_cost / self.estimated_savings_monthly
        else:
            self.payback_period_months = None


@dataclass
class CostAlert:
    """Cost alert definition"""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    
    # Alert details
    current_value: float
    threshold_value: float
    category: Optional[CostCategory] = None
    budget_id: Optional[str] = None
    
    # Timing
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Check if alert is still active"""
        return self.resolved_at is None


class ResourceTracker:
    """Track resource usage and costs"""
    
    def __init__(self):
        # Resource tracking
        self.resources: Dict[str, Dict[str, Any]] = {}
        self.resource_costs: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.utilization_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Cost tracking
        self.cost_per_resource_type = {
            "ec2_instance": {"base_rate": 0.1, "utilization_factor": True},
            "rds_instance": {"base_rate": 0.15, "utilization_factor": False},
            "s3_storage": {"base_rate": 0.023, "per_gb": True},
            "cloudfront": {"base_rate": 0.085, "per_gb": True},
            "redis_cache": {"base_rate": 0.063, "utilization_factor": False},
            "load_balancer": {"base_rate": 0.025, "utilization_factor": False}
        }
    
    def register_resource(self, resource_id: str, resource_type: str, 
                         specifications: Dict[str, Any]):
        """Register a resource for tracking"""
        self.resources[resource_id] = {
            "type": resource_type,
            "specs": specifications,
            "created_at": datetime.now(timezone.utc),
            "last_updated": datetime.now(timezone.utc)
        }
    
    def update_resource_utilization(self, resource_id: str, utilization_metrics: Dict[str, float]):
        """Update resource utilization metrics"""
        if resource_id not in self.resources:
            return
        
        # Store utilization data
        self.utilization_history[resource_id].append({
            "timestamp": datetime.now(timezone.utc),
            "metrics": utilization_metrics
        })
        
        # Update resource info
        self.resources[resource_id]["last_updated"] = datetime.now(timezone.utc)
        self.resources[resource_id]["current_utilization"] = utilization_metrics
    
    def calculate_resource_cost(self, resource_id: str, hours: float = 1.0) -> float:
        """Calculate cost for a resource"""
        if resource_id not in self.resources:
            return 0.0
        
        resource = self.resources[resource_id]
        resource_type = resource["type"]
        specs = resource["specs"]
        
        if resource_type not in self.cost_per_resource_type:
            return 0.0
        
        cost_config = self.cost_per_resource_type[resource_type]
        base_rate = cost_config["base_rate"]
        
        # Calculate base cost
        if cost_config.get("per_gb", False):
            # Storage-based pricing
            size_gb = specs.get("size_gb", 1)
            cost = base_rate * size_gb * hours / 24  # Convert to hourly
        else:
            # Instance-based pricing
            cost = base_rate * hours
        
        # Apply utilization factor if relevant
        if cost_config.get("utilization_factor", False):
            current_util = resource.get("current_utilization", {})
            cpu_util = current_util.get("cpu_percent", 50) / 100
            memory_util = current_util.get("memory_percent", 50) / 100
            avg_util = (cpu_util + memory_util) / 2
            
            # Adjust cost based on utilization (unused capacity still costs money)
            utilization_multiplier = 0.7 + (avg_util * 0.3)  # 70% base + 30% utilization-based
            cost *= utilization_multiplier
        
        return cost
    
    def get_underutilized_resources(self, utilization_threshold: float = 30.0) -> List[Dict[str, Any]]:
        """Find underutilized resources"""
        underutilized = []
        
        for resource_id, resource in self.resources.items():
            if resource_id not in self.utilization_history:
                continue
            
            # Calculate average utilization over recent period
            recent_data = list(self.utilization_history[resource_id])[-24:]  # Last 24 readings
            
            if not recent_data:
                continue
            
            total_cpu = sum(data["metrics"].get("cpu_percent", 0) for data in recent_data)
            total_memory = sum(data["metrics"].get("memory_percent", 0) for data in recent_data)
            
            avg_cpu = total_cpu / len(recent_data)
            avg_memory = total_memory / len(recent_data)
            avg_utilization = (avg_cpu + avg_memory) / 2
            
            if avg_utilization < utilization_threshold:
                current_cost = self.calculate_resource_cost(resource_id, 24)  # Daily cost
                
                underutilized.append({
                    "resource_id": resource_id,
                    "resource_type": resource["type"],
                    "avg_utilization_percent": avg_utilization,
                    "daily_cost": current_cost,
                    "potential_savings": current_cost * 0.5,  # Assume 50% savings potential
                    "recommendation": "Consider rightsizing or consolidation"
                })
        
        return sorted(underutilized, key=lambda x: x["potential_savings"], reverse=True)
    
    def get_cost_breakdown(self) -> Dict[str, float]:
        """Get cost breakdown by resource type"""
        breakdown = defaultdict(float)
        
        for resource_id, resource in self.resources.items():
            resource_type = resource["type"]
            cost = self.calculate_resource_cost(resource_id, 1)  # Hourly cost
            breakdown[resource_type] += cost
        
        return dict(breakdown)


class BudgetManager:
    """Manage budgets and spending limits"""
    
    def __init__(self):
        self.budgets: Dict[str, BudgetLimit] = {}
        self.spending_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.active_alerts: Dict[str, CostAlert] = {}
        
    def create_budget(self, budget: BudgetLimit):
        """Create a new budget"""
        self.budgets[budget.budget_id] = budget
        logger.info(f"Created budget: {budget.name} (${budget.limit_amount:.2f}/{budget.period.value})")
    
    def update_budget_spending(self, budget_id: str, amount: float, category: Optional[CostCategory] = None):
        """Update spending for a budget"""
        if budget_id not in self.budgets:
            return
        
        budget = self.budgets[budget_id]
        
        # Check if spending applies to this budget's categories
        if budget.categories and category and category not in budget.categories:
            return
        
        # Reset budget if new period started
        self._check_and_reset_budget_period(budget)
        
        # Update spending
        budget.current_spend += amount
        
        # Record spending history
        self.spending_history[budget_id].append({
            "timestamp": datetime.now(timezone.utc),
            "amount": amount,
            "category": category.value if category else None,
            "total_spend": budget.current_spend
        })
        
        # Check for alerts
        self._check_budget_alerts(budget)
    
    def _check_and_reset_budget_period(self, budget: BudgetLimit):
        """Check if budget period has reset"""
        current_time = datetime.now(timezone.utc)
        
        if budget.period == BudgetPeriod.DAILY:
            period_duration = timedelta(days=1)
        elif budget.period == BudgetPeriod.WEEKLY:
            period_duration = timedelta(weeks=1)
        elif budget.period == BudgetPeriod.MONTHLY:
            period_duration = timedelta(days=30)
        elif budget.period == BudgetPeriod.QUARTERLY:
            period_duration = timedelta(days=90)
        else:  # ANNUALLY
            period_duration = timedelta(days=365)
        
        if current_time >= budget.period_start + period_duration:
            # Reset budget for new period
            budget.current_spend = 0.0
            budget.period_start = current_time
            
            # Clear resolved alerts for this budget
            alerts_to_remove = [
                alert_id for alert_id, alert in self.active_alerts.items()
                if alert.budget_id == budget.budget_id and alert.resolved_at is not None
            ]
            
            for alert_id in alerts_to_remove:
                del self.active_alerts[alert_id]
    
    def _check_budget_alerts(self, budget: BudgetLimit):
        """Check for budget threshold violations"""
        utilization = budget.utilization_percent
        
        # Determine alert level
        if utilization >= budget.emergency_threshold:
            severity = AlertSeverity.EMERGENCY
            alert_type = "budget_exceeded"
        elif utilization >= budget.critical_threshold:
            severity = AlertSeverity.CRITICAL
            alert_type = "budget_critical"
        elif utilization >= budget.warning_threshold:
            severity = AlertSeverity.WARNING
            alert_type = "budget_warning"
        else:
            return  # No alert needed
        
        # Check if similar alert already exists
        existing_alert = None
        for alert in self.active_alerts.values():
            if (alert.budget_id == budget.budget_id and 
                alert.alert_type == alert_type and 
                alert.is_active):
                existing_alert = alert
                break
        
        if existing_alert:
            # Update existing alert
            existing_alert.current_value = utilization
            existing_alert.triggered_at = datetime.now(timezone.utc)
        else:
            # Create new alert
            alert = CostAlert(
                alert_id=str(uuid.uuid4()),
                alert_type=alert_type,
                severity=severity,
                message=f"Budget '{budget.name}' is at {utilization:.1f}% ({budget.current_spend:.2f}/${budget.limit_amount:.2f})",
                current_value=utilization,
                threshold_value=getattr(budget, f"{severity.value}_threshold"),
                budget_id=budget.budget_id
            )
            
            self.active_alerts[alert.alert_id] = alert
            logger.warning(f"Budget alert: {alert.message}")
            
            # Auto-shutdown if enabled and emergency threshold reached
            if (severity == AlertSeverity.EMERGENCY and 
                budget.auto_shutdown_enabled):
                self._trigger_emergency_shutdown(budget)
    
    def _trigger_emergency_shutdown(self, budget: BudgetLimit):
        """Trigger emergency resource shutdown"""
        logger.critical(f"Emergency shutdown triggered for budget: {budget.name}")
        # Implementation would integrate with resource management systems
        # to automatically shut down or scale down resources
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get status of all budgets"""
        status = {
            "total_budgets": len(self.budgets),
            "active_alerts": len([a for a in self.active_alerts.values() if a.is_active]),
            "budgets": []
        }
        
        for budget in self.budgets.values():
            budget_info = {
                "budget_id": budget.budget_id,
                "name": budget.name,
                "limit_amount": budget.limit_amount,
                "current_spend": budget.current_spend,
                "utilization_percent": budget.utilization_percent,
                "remaining_budget": budget.remaining_budget,
                "period": budget.period.value,
                "status": "ok"
            }
            
            # Determine status
            if budget.utilization_percent >= budget.emergency_threshold:
                budget_info["status"] = "emergency"
            elif budget.utilization_percent >= budget.critical_threshold:
                budget_info["status"] = "critical"
            elif budget.utilization_percent >= budget.warning_threshold:
                budget_info["status"] = "warning"
            
            status["budgets"].append(budget_info)
        
        return status


class CostOptimizationEngine:
    """Generate cost optimization recommendations"""
    
    def __init__(self, resource_tracker: ResourceTracker):
        self.resource_tracker = resource_tracker
        self.recommendations: Dict[str, CostOptimizationRecommendation] = {}
        
        # Optimization rules
        self.optimization_rules = {
            OptimizationType.RIGHTSIZING: self._analyze_rightsizing,
            OptimizationType.SPOT_INSTANCES: self._analyze_spot_instances,
            OptimizationType.RESERVED_INSTANCES: self._analyze_reserved_instances,
            OptimizationType.STORAGE_OPTIMIZATION: self._analyze_storage_optimization,
            OptimizationType.WASTE_ELIMINATION: self._analyze_waste_elimination,
            OptimizationType.RESOURCE_CONSOLIDATION: self._analyze_resource_consolidation
        }
    
    async def generate_recommendations(self) -> List[CostOptimizationRecommendation]:
        """Generate all cost optimization recommendations"""
        recommendations = []
        
        for optimization_type, analyzer in self.optimization_rules.items():
            try:
                type_recommendations = await analyzer()
                recommendations.extend(type_recommendations)
            except Exception as e:
                logger.error(f"Error generating {optimization_type.value} recommendations: {e}")
        
        # Sort by potential savings
        recommendations.sort(key=lambda r: r.estimated_savings_annual, reverse=True)
        
        # Store recommendations
        for rec in recommendations:
            self.recommendations[rec.recommendation_id] = rec
        
        return recommendations
    
    async def _analyze_rightsizing(self) -> List[CostOptimizationRecommendation]:
        """Analyze rightsizing opportunities"""
        recommendations = []
        underutilized = self.resource_tracker.get_underutilized_resources(30.0)
        
        for resource in underutilized:
            if resource["avg_utilization_percent"] < 20:
                # Severe underutilization - recommend significant downsizing
                savings_percent = 0.5
                effort = "medium"
                risk = "low"
            else:
                # Moderate underutilization - recommend minor downsizing
                savings_percent = 0.25
                effort = "low"
                risk = "low"
            
            annual_savings = resource["daily_cost"] * 365 * savings_percent
            
            recommendation = CostOptimizationRecommendation(
                recommendation_id=str(uuid.uuid4()),
                optimization_type=OptimizationType.RIGHTSIZING,
                title=f"Rightsize {resource['resource_type']} instance",
                description=f"Resource {resource['resource_id']} is underutilized at {resource['avg_utilization_percent']:.1f}%. Consider downsizing to reduce costs.",
                estimated_savings_monthly=annual_savings / 12,
                estimated_savings_annual=annual_savings,
                confidence=0.8,
                effort_level=effort,
                risk_level=risk,
                implementation_time="1-2 weeks",
                affected_resources=[resource["resource_id"]],
                categories=[CostCategory.COMPUTE]
            )
            
            recommendation.calculate_payback_period()
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _analyze_spot_instances(self) -> List[CostOptimizationRecommendation]:
        """Analyze spot instance opportunities"""
        recommendations = []
        
        compute_resources = [
            (rid, resource) for rid, resource in self.resource_tracker.resources.items()
            if resource["type"] == "ec2_instance"
        ]
        
        if len(compute_resources) >= 3:  # Minimum for spot instance strategy
            total_compute_cost = sum(
                self.resource_tracker.calculate_resource_cost(rid, 24)
                for rid, _ in compute_resources
            )
            
            # Estimate 60% savings with spot instances
            annual_savings = total_compute_cost * 365 * 0.6
            
            recommendation = CostOptimizationRecommendation(
                recommendation_id=str(uuid.uuid4()),
                optimization_type=OptimizationType.SPOT_INSTANCES,
                title="Implement spot instance strategy",
                description="Use spot instances for fault-tolerant workloads to achieve significant cost savings.",
                estimated_savings_monthly=annual_savings / 12,
                estimated_savings_annual=annual_savings,
                confidence=0.7,
                effort_level="high",
                risk_level="medium",
                implementation_time="1-2 months",
                affected_resources=[rid for rid, _ in compute_resources],
                categories=[CostCategory.COMPUTE],
                potential_risks=[
                    "Spot instances can be terminated with short notice",
                    "Requires fault-tolerant application architecture"
                ]
            )
            
            recommendation.calculate_payback_period()
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _analyze_reserved_instances(self) -> List[CostOptimizationRecommendation]:
        """Analyze reserved instance opportunities"""
        recommendations = []
        
        # Find stable, long-running instances
        stable_instances = []
        for rid, resource in self.resource_tracker.resources.items():
            if resource["type"] in ["ec2_instance", "rds_instance"]:
                # Check if instance has been running for a while (simplified)
                age_days = (datetime.now(timezone.utc) - resource["created_at"]).days
                if age_days > 30:  # Running for more than 30 days
                    stable_instances.append((rid, resource))
        
        if len(stable_instances) >= 2:
            total_stable_cost = sum(
                self.resource_tracker.calculate_resource_cost(rid, 24)
                for rid, _ in stable_instances
            )
            
            # Estimate 30% savings with 1-year reserved instances
            annual_savings = total_stable_cost * 365 * 0.3
            
            recommendation = CostOptimizationRecommendation(
                recommendation_id=str(uuid.uuid4()),
                optimization_type=OptimizationType.RESERVED_INSTANCES,
                title="Purchase reserved instances",
                description="Purchase reserved instances for stable workloads to achieve significant cost savings.",
                estimated_savings_monthly=annual_savings / 12,
                estimated_savings_annual=annual_savings,
                confidence=0.9,
                effort_level="low",
                risk_level="low",
                implementation_time="immediate",
                affected_resources=[rid for rid, _ in stable_instances],
                categories=[CostCategory.COMPUTE, CostCategory.DATABASE],
                prerequisites=["Commit to 1-3 year usage"],
                implementation_cost=total_stable_cost * 365 * 0.7  # Upfront payment
            )
            
            recommendation.calculate_payback_period()
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _analyze_storage_optimization(self) -> List[CostOptimizationRecommendation]:
        """Analyze storage optimization opportunities"""
        recommendations = []
        
        storage_resources = [
            (rid, resource) for rid, resource in self.resource_tracker.resources.items()
            if resource["type"] == "s3_storage"
        ]
        
        if storage_resources:
            total_storage_cost = sum(
                self.resource_tracker.calculate_resource_cost(rid, 24)
                for rid, _ in storage_resources
            )
            
            # Estimate 40% savings with intelligent tiering and lifecycle policies
            annual_savings = total_storage_cost * 365 * 0.4
            
            recommendation = CostOptimizationRecommendation(
                recommendation_id=str(uuid.uuid4()),
                optimization_type=OptimizationType.STORAGE_OPTIMIZATION,
                title="Optimize storage with intelligent tiering",
                description="Implement intelligent storage tiering and lifecycle policies to reduce storage costs.",
                estimated_savings_monthly=annual_savings / 12,
                estimated_savings_annual=annual_savings,
                confidence=0.8,
                effort_level="medium",
                risk_level="low",
                implementation_time="1-2 weeks",
                affected_resources=[rid for rid, _ in storage_resources],
                categories=[CostCategory.STORAGE]
            )
            
            recommendation.calculate_payback_period()
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _analyze_waste_elimination(self) -> List[CostOptimizationRecommendation]:
        """Analyze waste elimination opportunities"""
        recommendations = []
        
        # Find idle resources (simplified detection)
        idle_resources = []
        for rid, resource in self.resource_tracker.resources.items():
            if rid in self.resource_tracker.utilization_history:
                recent_data = list(self.resource_tracker.utilization_history[rid])[-24:]
                
                if recent_data:
                    # Check if resource has been consistently idle
                    avg_utilization = sum(
                        data["metrics"].get("cpu_percent", 0) + data["metrics"].get("memory_percent", 0)
                        for data in recent_data
                    ) / (len(recent_data) * 2)
                    
                    if avg_utilization < 5:  # Less than 5% utilization
                        idle_resources.append((rid, resource))
        
        if idle_resources:
            total_waste_cost = sum(
                self.resource_tracker.calculate_resource_cost(rid, 24)
                for rid, _ in idle_resources
            )
            
            annual_savings = total_waste_cost * 365
            
            recommendation = CostOptimizationRecommendation(
                recommendation_id=str(uuid.uuid4()),
                optimization_type=OptimizationType.WASTE_ELIMINATION,
                title="Terminate idle resources",
                description=f"Found {len(idle_resources)} idle resources that can be safely terminated.",
                estimated_savings_monthly=annual_savings / 12,
                estimated_savings_annual=annual_savings,
                confidence=0.9,
                effort_level="low",
                risk_level="low",
                implementation_time="immediate",
                affected_resources=[rid for rid, _ in idle_resources],
                categories=[CostCategory.COMPUTE, CostCategory.STORAGE]
            )
            
            recommendation.calculate_payback_period()
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _analyze_resource_consolidation(self) -> List[CostOptimizationRecommendation]:
        """Analyze resource consolidation opportunities"""
        recommendations = []
        
        # Find lightly utilized resources that could be consolidated
        light_resources = self.resource_tracker.get_underutilized_resources(50.0)
        
        if len(light_resources) >= 3:
            # Group by resource type
            resource_groups = defaultdict(list)
            for resource in light_resources:
                resource_groups[resource["resource_type"]].append(resource)
            
            for resource_type, resources in resource_groups.items():
                if len(resources) >= 3:
                    total_cost = sum(r["daily_cost"] for r in resources)
                    
                    # Estimate 40% savings through consolidation
                    annual_savings = total_cost * 365 * 0.4
                    
                    recommendation = CostOptimizationRecommendation(
                        recommendation_id=str(uuid.uuid4()),
                        optimization_type=OptimizationType.RESOURCE_CONSOLIDATION,
                        title=f"Consolidate {resource_type} instances",
                        description=f"Consolidate {len(resources)} lightly utilized {resource_type} instances into fewer, more efficiently utilized instances.",
                        estimated_savings_monthly=annual_savings / 12,
                        estimated_savings_annual=annual_savings,
                        confidence=0.7,
                        effort_level="high",
                        risk_level="medium",
                        implementation_time="1-2 months",
                        affected_resources=[r["resource_id"] for r in resources],
                        categories=[CostCategory.COMPUTE],
                        potential_risks=[
                            "Requires application architecture changes",
                            "May impact fault tolerance"
                        ]
                    )
                    
                    recommendation.calculate_payback_period()
                    recommendations.append(recommendation)
        
        return recommendations


class CostMonitoringDashboard:
    """Real-time cost monitoring and alerting"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.resource_tracker = ResourceTracker()
        self.budget_manager = BudgetManager()
        self.optimization_engine = CostOptimizationEngine(self.resource_tracker)
        
        # Metrics storage
        self.cost_history = deque(maxlen=10000)
        self.alert_history = deque(maxlen=1000)
        
        # Statistics
        self.stats = {
            "total_cost_tracked": 0.0,
            "total_savings_identified": 0.0,
            "active_optimizations": 0,
            "budget_violations": 0,
            "cost_trends": {}
        }
    
    async def add_cost_metrics(self, metrics: CostMetrics):
        """Add cost metrics for tracking"""
        try:
            # Store metrics
            self.cost_history.append(metrics)
            
            # Update budgets
            for category in CostCategory:
                category_cost = getattr(metrics, f"{category.value}_cost", 0.0)
                if category_cost > 0:
                    # Update all relevant budgets
                    for budget in self.budget_manager.budgets.values():
                        if not budget.categories or category in budget.categories:
                            self.budget_manager.update_budget_spending(
                                budget.budget_id, category_cost, category
                            )
            
            # Store in Redis
            await self.redis.lpush(
                "cost_monitoring:metrics",
                json.dumps(metrics.to_dict())
            )
            await self.redis.ltrim("cost_monitoring:metrics", 0, 9999)
            
            # Update statistics
            self.stats["total_cost_tracked"] += metrics.total_cost
            
        except Exception as e:
            logger.error(f"Error adding cost metrics: {e}")
    
    async def generate_cost_report(self, period_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive cost report"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=period_hours)
            recent_metrics = [
                m for m in self.cost_history 
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return {"error": "No metrics available for the specified period"}
            
            # Calculate totals and averages
            total_cost = sum(m.total_cost for m in recent_metrics)
            avg_cost_per_hour = total_cost / period_hours if period_hours > 0 else 0
            
            # Cost breakdown
            cost_breakdown = {}
            for category in CostCategory:
                category_total = sum(
                    getattr(m, f"{category.value}_cost", 0.0) 
                    for m in recent_metrics
                )
                cost_breakdown[category.value] = category_total
            
            # Trend analysis
            if len(recent_metrics) > 1:
                first_half = recent_metrics[:len(recent_metrics)//2]
                second_half = recent_metrics[len(recent_metrics)//2:]
                
                first_avg = sum(m.total_cost for m in first_half) / len(first_half)
                second_avg = sum(m.total_cost for m in second_half) / len(second_half)
                
                trend_percent = ((second_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
            else:
                trend_percent = 0
            
            # Efficiency metrics
            if recent_metrics:
                latest_metrics = recent_metrics[-1]
                efficiency = {
                    "cost_per_request": latest_metrics.cost_per_request,
                    "cost_per_user": latest_metrics.cost_per_user,
                    "cost_per_transaction": latest_metrics.cost_per_transaction
                }
            else:
                efficiency = {}
            
            # Get optimizations
            recommendations = await self.optimization_engine.generate_recommendations()
            total_potential_savings = sum(r.estimated_savings_annual for r in recommendations)
            
            # Budget status
            budget_status = self.budget_manager.get_budget_status()
            
            return {
                "report_period": {
                    "hours": period_hours,
                    "start_time": cutoff_time.isoformat(),
                    "end_time": datetime.now(timezone.utc).isoformat()
                },
                "cost_summary": {
                    "total_cost": total_cost,
                    "avg_cost_per_hour": avg_cost_per_hour,
                    "projected_monthly": avg_cost_per_hour * 24 * 30,
                    "trend_percent": trend_percent
                },
                "cost_breakdown": cost_breakdown,
                "efficiency_metrics": efficiency,
                "optimization_opportunities": {
                    "total_recommendations": len(recommendations),
                    "potential_annual_savings": total_potential_savings,
                    "top_recommendations": [
                        {
                            "title": r.title,
                            "type": r.optimization_type.value,
                            "annual_savings": r.estimated_savings_annual,
                            "confidence": r.confidence,
                            "effort": r.effort_level
                        }
                        for r in recommendations[:5]
                    ]
                },
                "budget_status": budget_status,
                "resource_utilization": {
                    "total_resources": len(self.resource_tracker.resources),
                    "underutilized_resources": len(
                        self.resource_tracker.get_underutilized_resources()
                    )
                },
                "statistics": self.stats.copy()
            }
            
        except Exception as e:
            logger.error(f"Error generating cost report: {e}")
            return {"error": str(e)}
    
    async def get_real_time_dashboard(self) -> Dict[str, Any]:
        """Get real-time dashboard data"""
        try:
            # Latest metrics
            latest_metrics = self.cost_history[-1] if self.cost_history else None
            
            # Active alerts
            active_alerts = [
                {
                    "alert_id": alert.alert_id,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "triggered_at": alert.triggered_at.isoformat()
                }
                for alert in self.budget_manager.active_alerts.values()
                if alert.is_active
            ]
            
            # Recent cost trend (last 24 hours)
            last_24h = [
                m for m in self.cost_history
                if m.timestamp >= datetime.now(timezone.utc) - timedelta(hours=24)
            ]
            
            cost_trend = [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "total_cost": m.total_cost,
                    "compute_cost": m.compute_cost,
                    "storage_cost": m.storage_cost
                }
                for m in last_24h[-50:]  # Last 50 data points
            ]
            
            # Resource breakdown
            resource_breakdown = self.resource_tracker.get_cost_breakdown()
            
            return {
                "current_status": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "current_hourly_cost": latest_metrics.total_cost if latest_metrics else 0,
                    "projected_daily": (latest_metrics.total_cost * 24) if latest_metrics else 0,
                    "projected_monthly": (latest_metrics.total_cost * 24 * 30) if latest_metrics else 0
                },
                "alerts": {
                    "active_count": len(active_alerts),
                    "alerts": active_alerts
                },
                "cost_trend": cost_trend,
                "resource_breakdown": resource_breakdown,
                "budget_summary": {
                    "total_budgets": len(self.budget_manager.budgets),
                    "budgets_over_threshold": len([
                        b for b in self.budget_manager.budgets.values()
                        if b.utilization_percent > b.warning_threshold
                    ])
                },
                "optimization_summary": {
                    "active_recommendations": len(self.optimization_engine.recommendations),
                    "potential_savings": self.stats.get("total_savings_identified", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {"error": str(e)}


# Global cost monitoring system
cost_monitoring_system: Optional[CostMonitoringDashboard] = None


def initialize_cost_monitoring(redis_client: aioredis.Redis) -> CostMonitoringDashboard:
    """Initialize the cost monitoring system"""
    global cost_monitoring_system
    
    cost_monitoring_system = CostMonitoringDashboard(redis_client)
    logger.info("✅ Cost monitoring system initialized")
    return cost_monitoring_system


def get_cost_monitoring_system() -> CostMonitoringDashboard:
    """Get the global cost monitoring system"""
    if cost_monitoring_system is None:
        raise RuntimeError("Cost monitoring system not initialized")
    return cost_monitoring_system


# Convenience functions

async def track_resource_cost(resource_id: str, resource_type: str, 
                            specifications: Dict[str, Any], 
                            utilization_metrics: Dict[str, float]):
    """Track a resource's cost and utilization"""
    if cost_monitoring_system:
        cost_monitoring_system.resource_tracker.register_resource(
            resource_id, resource_type, specifications
        )
        cost_monitoring_system.resource_tracker.update_resource_utilization(
            resource_id, utilization_metrics
        )


async def add_cost_data(metrics: CostMetrics):
    """Add cost metrics to the monitoring system"""
    if cost_monitoring_system:
        await cost_monitoring_system.add_cost_metrics(metrics)


async def get_cost_dashboard() -> Dict[str, Any]:
    """Get real-time cost dashboard"""
    if cost_monitoring_system:
        return await cost_monitoring_system.get_real_time_dashboard()
    return {"error": "Cost monitoring not initialized"}


def create_budget(name: str, limit_amount: float, period: BudgetPeriod, 
                 categories: List[CostCategory] = None) -> str:
    """Create a new budget"""
    if not cost_monitoring_system:
        return ""
    
    budget_id = str(uuid.uuid4())
    budget = BudgetLimit(
        budget_id=budget_id,
        name=name,
        description=f"Budget for {name}",
        limit_amount=limit_amount,
        period=period,
        categories=categories or []
    )
    
    cost_monitoring_system.budget_manager.create_budget(budget)
    return budget_id