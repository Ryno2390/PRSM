"""
PRSM Resource Management API
===========================

User-friendly API endpoints for managing distributed resource contributions.
Enables users to easily configure, monitor, and optimize their node's
resource contributions to the PRSM network.

Features:
- Intuitive resource contribution settings
- Real-time resource monitoring and verification
- Economic optimization recommendations
- Performance analytics and insights
- Easy onboarding for new contributors
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from uuid import UUID
from pydantic import BaseModel, Field

from ..federation.distributed_resource_manager import (
    ResourceContributionSettings, ResourceType, ResourceMeasurement,
    distributed_resource_manager, register_node_resources, 
    update_node_resource_settings, get_node_resource_profile,
    get_network_resource_summary, allocate_resources_for_computation
)
from prsm.core.models import UserInput
from prsm.economy.tokenomics.ftns_service import get_ftns_service
import structlog

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1/resources", tags=["Resource Management"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ResourceConfigurationRequest(BaseModel):
    """Request model for configuring resource contributions"""
    # Resource allocation percentages (0-100)
    cpu_allocation_percent: float = Field(default=50.0, ge=0.0, le=100.0, description="CPU allocation percentage")
    gpu_allocation_percent: float = Field(default=50.0, ge=0.0, le=100.0, description="GPU allocation percentage")
    storage_allocation_percent: float = Field(default=30.0, ge=0.0, le=100.0, description="Storage allocation percentage")
    memory_allocation_percent: float = Field(default=40.0, ge=0.0, le=100.0, description="Memory allocation percentage")
    bandwidth_allocation_percent: float = Field(default=60.0, ge=0.0, le=100.0, description="Bandwidth allocation percentage")
    
    # Operational settings
    max_cpu_temperature: float = Field(default=80.0, description="Maximum CPU temperature (Celsius)")
    max_power_consumption: float = Field(default=100.0, description="Maximum power consumption (Watts)")
    priority_level: int = Field(default=5, ge=1, le=10, description="Priority level (1=lowest, 10=highest)")
    
    # Economic settings
    minimum_hourly_rate: float = Field(default=0.1, ge=0.0, description="Minimum hourly rate in FTNS")
    automatic_scaling: bool = Field(default=True, description="Enable automatic scaling")
    market_participation: bool = Field(default=True, description="Participate in resource marketplace")
    
    # Reliability settings
    uptime_commitment: float = Field(default=95.0, ge=50.0, le=99.9, description="Uptime commitment percentage")
    geographic_restrictions: List[str] = Field(default_factory=list, description="Geographic restrictions")
    data_retention_days: int = Field(default=30, ge=1, le=365, description="Data retention period (days)")
    
    # Advanced settings
    specialized_capabilities: List[str] = Field(default_factory=list, description="Specialized capabilities")
    security_level: str = Field(default="standard", description="Security level (standard, high, maximum)")


class ResourceStatusResponse(BaseModel):
    """Response model for resource status"""
    node_id: str
    user_id: str
    node_type: str
    geographic_region: str
    total_resources: Dict[str, Dict[str, Any]]
    current_allocation: Dict[str, float]
    verification_status: Dict[str, str]
    reputation_score: float
    estimated_earnings: Dict[str, float]
    performance_metrics: Dict[str, float]
    last_updated: datetime


class NetworkSummaryResponse(BaseModel):
    """Response model for network resource summary"""
    total_nodes: int
    total_capacity: Dict[str, float]
    current_utilization: Dict[str, float]
    average_reputation: float
    geographic_distribution: Dict[str, int]
    resource_prices: Dict[str, float]
    network_health: str


class OptimizationRecommendation(BaseModel):
    """Optimization recommendation for resource contributions"""
    recommendation_type: str
    title: str
    description: str
    potential_benefit: str
    implementation_difficulty: str
    estimated_ftns_increase: float


class ResourceOptimizationResponse(BaseModel):
    """Response with optimization recommendations"""
    current_efficiency: float
    potential_efficiency: float
    recommendations: List[OptimizationRecommendation]
    market_opportunities: List[Dict[str, Any]]


# ============================================================================
# RESOURCE CONFIGURATION ENDPOINTS
# ============================================================================

@router.post("/configure", response_model=Dict[str, str])
async def configure_resource_contribution(
    config: ResourceConfigurationRequest,
    user_id: str,
    background_tasks: BackgroundTasks
):
    """Configure or update resource contribution settings for a user's node"""
    try:
        # Convert percentage to decimal for internal use
        settings = ResourceContributionSettings(
            cpu_allocation_percentage=config.cpu_allocation_percent / 100.0,
            gpu_allocation_percentage=config.gpu_allocation_percent / 100.0,
            storage_allocation_percentage=config.storage_allocation_percent / 100.0,
            memory_allocation_percentage=config.memory_allocation_percent / 100.0,
            bandwidth_allocation_percentage=config.bandwidth_allocation_percent / 100.0,
            max_cpu_temperature=config.max_cpu_temperature,
            max_power_consumption=config.max_power_consumption,
            priority_level=config.priority_level,
            minimum_hourly_rate=Decimal(str(config.minimum_hourly_rate)),
            automatic_scaling=config.automatic_scaling,
            market_participation=config.market_participation,
            uptime_commitment=config.uptime_commitment / 100.0,
            geographic_restrictions=config.geographic_restrictions,
            data_retention_days=config.data_retention_days,
            specialized_capabilities=config.specialized_capabilities,
            security_level=config.security_level
        )
        
        # Check if user already has a node
        existing_node_id = await _get_user_node_id(user_id)
        
        if existing_node_id:
            # Update existing node
            success = await update_node_resource_settings(existing_node_id, settings)
            if not success:
                raise HTTPException(status_code=400, detail="Failed to update node settings")
            node_id = existing_node_id
            action = "updated"
        else:
            # Register new node
            node_id = await register_node_resources(user_id, settings)
            action = "registered"
        
        # Start background verification
        background_tasks.add_task(_verify_node_resources, node_id)
        
        logger.info("Resource configuration updated", 
                   user_id=user_id, 
                   node_id=node_id, 
                   action=action)
        
        return {
            "status": "success",
            "message": f"Node {action} successfully",
            "node_id": node_id,
            "action": action
        }
        
    except Exception as e:
        logger.error("Resource configuration failed", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")


@router.get("/status/{user_id}", response_model=ResourceStatusResponse)
async def get_resource_status(user_id: str):
    """Get current resource status and contribution details for a user"""
    try:
        node_id = await _get_user_node_id(user_id)
        if not node_id:
            raise HTTPException(status_code=404, detail="No node found for user")
        
        node_profile = get_node_resource_profile(node_id)
        if not node_profile:
            raise HTTPException(status_code=404, detail="Node profile not found")
        
        # Calculate current allocation and status
        total_resources = {}
        current_allocation = {}
        verification_status = {}
        
        for resource_type, resource_spec in node_profile.resources.items():
            total_resources[resource_type.value] = {
                "total_capacity": resource_spec.total_capacity,
                "allocated_capacity": resource_spec.allocated_capacity,
                "available_capacity": resource_spec.total_capacity - resource_spec.allocated_capacity,
                "measurement_unit": resource_spec.measurement_unit.value,
                "quality_metrics": resource_spec.quality_metrics
            }
            
            current_allocation[resource_type.value] = (
                resource_spec.allocated_capacity / resource_spec.total_capacity * 100
                if resource_spec.total_capacity > 0 else 0
            )
            
            verification_status[resource_type.value] = (
                "verified" if resource_spec.verification_score > 0.8 else
                "pending" if resource_spec.verification_score > 0.5 else "failed"
            )
        
        # Calculate estimated earnings
        estimated_earnings = await _calculate_estimated_earnings(node_profile)
        
        # Get performance metrics
        performance_metrics = await _get_performance_metrics(node_id)
        
        return ResourceStatusResponse(
            node_id=node_profile.node_id,
            user_id=node_profile.user_id,
            node_type=node_profile.node_type,
            geographic_region=node_profile.geographic_region,
            total_resources=total_resources,
            current_allocation=current_allocation,
            verification_status=verification_status,
            reputation_score=node_profile.reputation_score,
            estimated_earnings=estimated_earnings,
            performance_metrics=performance_metrics,
            last_updated=node_profile.last_updated
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get resource status", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/network/summary", response_model=NetworkSummaryResponse)
async def get_network_summary():
    """Get summary of total network resources and utilization"""
    try:
        network_data = get_network_resource_summary()
        
        # Calculate utilization percentages
        total_capacity = {}
        current_utilization = {}
        
        for resource_type, data in network_data["resource_summary"].items():
            total_capacity[resource_type] = data["total_capacity"]
            utilization_pct = (
                data["allocated_capacity"] / data["total_capacity"] * 100
                if data["total_capacity"] > 0 else 0
            )
            current_utilization[resource_type] = utilization_pct
        
        # Get geographic distribution
        geographic_distribution = await _get_geographic_distribution()
        
        # Get current resource prices
        resource_prices = await _get_current_resource_prices()
        
        return NetworkSummaryResponse(
            total_nodes=network_data["total_nodes"],
            total_capacity=total_capacity,
            current_utilization=current_utilization,
            average_reputation=network_data["average_node_reputation"],
            geographic_distribution=geographic_distribution,
            resource_prices=resource_prices,
            network_health=network_data["network_health"]
        )
        
    except Exception as e:
        logger.error("Failed to get network summary", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get network summary: {str(e)}")


# ============================================================================
# OPTIMIZATION AND RECOMMENDATIONS
# ============================================================================

@router.get("/optimize/{user_id}", response_model=ResourceOptimizationResponse)
async def get_optimization_recommendations(user_id: str):
    """Get personalized optimization recommendations for resource contributions"""
    try:
        node_id = await _get_user_node_id(user_id)
        if not node_id:
            raise HTTPException(status_code=404, detail="No node found for user")
        
        node_profile = get_node_resource_profile(node_id)
        if not node_profile:
            raise HTTPException(status_code=404, detail="Node profile not found")
        
        # Calculate current efficiency
        current_efficiency = await _calculate_resource_efficiency(node_profile)
        
        # Generate optimization recommendations
        recommendations = await _generate_optimization_recommendations(node_profile)
        
        # Find market opportunities
        market_opportunities = await _find_market_opportunities(node_profile)
        
        # Calculate potential efficiency with recommendations
        potential_efficiency = await _calculate_potential_efficiency(node_profile, recommendations)
        
        return ResourceOptimizationResponse(
            current_efficiency=current_efficiency,
            potential_efficiency=potential_efficiency,
            recommendations=recommendations,
            market_opportunities=market_opportunities
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to generate optimization recommendations", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.post("/auto-optimize/{user_id}")
async def auto_optimize_resources(user_id: str, background_tasks: BackgroundTasks):
    """Automatically optimize resource allocation based on current market conditions"""
    try:
        node_id = await _get_user_node_id(user_id)
        if not node_id:
            raise HTTPException(status_code=404, detail="No node found for user")
        
        node_profile = get_node_resource_profile(node_id)
        if not node_profile:
            raise HTTPException(status_code=404, detail="Node profile not found")
        
        # Check if auto-optimization is enabled
        settings = node_profile.contribution_settings
        if not settings.get("automatic_scaling", True):
            raise HTTPException(status_code=400, detail="Auto-optimization is disabled for this node")
        
        # Generate optimized settings
        optimized_settings = await _generate_optimized_settings(node_profile)
        
        # Apply optimization
        success = await update_node_resource_settings(node_id, optimized_settings)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to apply optimization")
        
        # Schedule background verification
        background_tasks.add_task(_verify_node_resources, node_id)
        
        logger.info("Auto-optimization applied", user_id=user_id, node_id=node_id)
        
        return {
            "status": "success",
            "message": "Resource allocation automatically optimized",
            "optimizations_applied": await _summarize_optimizations(node_profile, optimized_settings)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Auto-optimization failed", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Auto-optimization failed: {str(e)}")


# ============================================================================
# MONITORING AND ANALYTICS
# ============================================================================

@router.get("/analytics/{user_id}")
async def get_resource_analytics(
    user_id: str, 
    days: int = 30,
    resource_type: Optional[str] = None
):
    """Get detailed analytics for resource contributions and earnings"""
    try:
        node_id = await _get_user_node_id(user_id)
        if not node_id:
            raise HTTPException(status_code=404, detail="No node found for user")
        
        # Get analytics data
        analytics = await _get_resource_analytics(node_id, days, resource_type)
        
        return {
            "node_id": node_id,
            "period_days": days,
            "resource_filter": resource_type,
            "analytics": analytics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get analytics", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")


@router.get("/earnings/{user_id}")
async def get_earnings_report(user_id: str, period: str = "30d"):
    """Get detailed earnings report for resource contributions"""
    try:
        node_id = await _get_user_node_id(user_id)
        if not node_id:
            raise HTTPException(status_code=404, detail="No node found for user")
        
        # Parse period
        days = _parse_period_to_days(period)
        
        # Get earnings data
        earnings_report = await _get_earnings_report(node_id, days)
        
        return earnings_report
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get earnings report", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Earnings report failed: {str(e)}")


# ============================================================================
# RESOURCE VERIFICATION AND TESTING
# ============================================================================

@router.post("/verify/{user_id}")
async def verify_node_resources(user_id: str, background_tasks: BackgroundTasks):
    """Manually trigger resource verification for a node"""
    try:
        node_id = await _get_user_node_id(user_id)
        if not node_id:
            raise HTTPException(status_code=404, detail="No node found for user")
        
        # Start background verification
        background_tasks.add_task(_verify_node_resources, node_id)
        
        return {
            "status": "success",
            "message": "Resource verification initiated",
            "node_id": node_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Manual verification failed", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@router.post("/benchmark/{user_id}")
async def run_performance_benchmark(user_id: str, resource_type: str, background_tasks: BackgroundTasks):
    """Run performance benchmark for specific resource type"""
    try:
        node_id = await _get_user_node_id(user_id)
        if not node_id:
            raise HTTPException(status_code=404, detail="No node found for user")
        
        # Validate resource type
        try:
            ResourceType(resource_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid resource type: {resource_type}")
        
        # Start background benchmark
        background_tasks.add_task(_run_resource_benchmark, node_id, resource_type)
        
        return {
            "status": "success",
            "message": f"Benchmark initiated for {resource_type}",
            "node_id": node_id,
            "resource_type": resource_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Benchmark failed", user_id=user_id, resource_type=resource_type, error=str(e))
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def _get_user_node_id(user_id: str) -> Optional[str]:
    """Get node ID for a user"""
    # Search through registered nodes to find user's node
    for node_id, node_profile in distributed_resource_manager.node_registry.items():
        if node_profile.user_id == user_id:
            return node_id
    return None


async def _verify_node_resources(node_id: str):
    """Background task to verify node resources"""
    try:
        await distributed_resource_manager._verify_node_resources(node_id)
        logger.info("Background resource verification completed", node_id=node_id)
    except Exception as e:
        logger.error("Background verification failed", node_id=node_id, error=str(e))


async def _calculate_estimated_earnings(node_profile) -> Dict[str, float]:
    """Calculate estimated earnings for different time periods"""
    # Simplified earnings calculation
    base_hourly_rate = 0.1  # FTNS per hour per resource unit
    
    total_capacity = sum(
        spec.total_capacity for spec in node_profile.resources.values()
    )
    
    hourly_earnings = total_capacity * base_hourly_rate * node_profile.reputation_score
    
    return {
        "hourly": hourly_earnings,
        "daily": hourly_earnings * 24,
        "weekly": hourly_earnings * 24 * 7,
        "monthly": hourly_earnings * 24 * 30
    }


async def _get_performance_metrics(node_id: str) -> Dict[str, float]:
    """Get performance metrics for a node"""
    # Would query actual performance data
    return {
        "cpu_utilization": 65.5,
        "memory_utilization": 78.2,
        "network_latency": 15.3,
        "uptime_percentage": 99.2,
        "task_completion_rate": 95.8
    }


async def _get_geographic_distribution() -> Dict[str, int]:
    """Get geographic distribution of nodes"""
    distribution = {}
    for node_profile in distributed_resource_manager.node_registry.values():
        region = node_profile.geographic_region
        distribution[region] = distribution.get(region, 0) + 1
    return distribution


async def _get_current_resource_prices() -> Dict[str, float]:
    """Get current market prices for different resource types"""
    # Would query actual market data
    return {
        "compute_cpu": 0.05,      # FTNS per core-hour
        "compute_gpu": 0.25,      # FTNS per GPU-hour
        "storage_persistent": 0.01, # FTNS per GB-hour
        "storage_memory": 0.03,   # FTNS per GB-hour
        "bandwidth_ingress": 0.02, # FTNS per Mbps-hour
        "bandwidth_egress": 0.02   # FTNS per Mbps-hour
    }


async def _calculate_resource_efficiency(node_profile) -> float:
    """Calculate current resource utilization efficiency"""
    total_capacity = 0.0
    total_allocated = 0.0
    
    for resource_spec in node_profile.resources.values():
        total_capacity += resource_spec.total_capacity
        total_allocated += resource_spec.allocated_capacity
    
    return (total_allocated / total_capacity * 100) if total_capacity > 0 else 0.0


async def _generate_optimization_recommendations(node_profile) -> List[OptimizationRecommendation]:
    """Generate personalized optimization recommendations"""
    recommendations = []
    
    # Check for underutilized resources
    for resource_type, resource_spec in node_profile.resources.items():
        utilization = resource_spec.allocated_capacity / resource_spec.total_capacity
        
        if utilization < 0.3:  # Less than 30% utilized
            recommendations.append(OptimizationRecommendation(
                recommendation_type="increase_allocation",
                title=f"Increase {resource_type.value} allocation",
                description=f"Your {resource_type.value} is only {utilization*100:.1f}% utilized. Consider increasing allocation to earn more FTNS.",
                potential_benefit=f"+{(0.7 - utilization) * resource_spec.total_capacity * 0.1:.2f} FTNS/hour",
                implementation_difficulty="Easy",
                estimated_ftns_increase=(0.7 - utilization) * resource_spec.total_capacity * 0.1 * 24
            ))
    
    # Check reputation score
    if node_profile.reputation_score < 0.8:
        recommendations.append(OptimizationRecommendation(
            recommendation_type="improve_reliability",
            title="Improve node reliability",
            description="Your reputation score is below optimal. Ensure consistent uptime and performance to increase earnings.",
            potential_benefit=f"+{(0.9 - node_profile.reputation_score) * 100:.1f}% earnings multiplier",
            implementation_difficulty="Medium",
            estimated_ftns_increase=5.0  # Estimated daily increase
        ))
    
    # Check for market opportunities
    if node_profile.node_type == "micro" and len(node_profile.resources) > 3:
        recommendations.append(OptimizationRecommendation(
            recommendation_type="micro_node_bonus",
            title="Claim micro-node empowerment bonus",
            description="Your diverse resource portfolio qualifies for micro-node empowerment bonuses. Ensure your node is properly classified.",
            potential_benefit="2x earnings multiplier",
            implementation_difficulty="Easy",
            estimated_ftns_increase=10.0
        ))
    
    return recommendations


async def _find_market_opportunities(node_profile) -> List[Dict[str, Any]]:
    """Find current market opportunities for the node"""
    opportunities = []
    
    # High-demand resource types
    high_demand_resources = ["compute_gpu", "specialized_quantum", "storage_memory"]
    
    for resource_type, resource_spec in node_profile.resources.items():
        if resource_type.value in high_demand_resources:
            opportunities.append({
                "type": "high_demand_resource",
                "resource_type": resource_type.value,
                "description": f"{resource_type.value} is in high demand",
                "current_price": 0.25,  # Would get from market
                "projected_price": 0.35,
                "opportunity_score": 8.5
            })
    
    # Geographic arbitrage opportunities
    if node_profile.geographic_region in ["asia", "europe"]:
        opportunities.append({
            "type": "geographic_arbitrage",
            "description": "High demand for compute in your region during US off-hours",
            "time_window": "18:00-06:00 local time",
            "price_premium": "25%",
            "opportunity_score": 7.0
        })
    
    return opportunities


async def _calculate_potential_efficiency(node_profile, recommendations) -> float:
    """Calculate potential efficiency if recommendations are implemented"""
    current_efficiency = await _calculate_resource_efficiency(node_profile)
    
    # Estimate improvement from recommendations
    potential_improvement = sum(rec.estimated_ftns_increase for rec in recommendations) / 10
    
    return min(current_efficiency + potential_improvement, 95.0)  # Cap at 95%


async def _generate_optimized_settings(node_profile) -> ResourceContributionSettings:
    """Generate optimized resource contribution settings"""
    current_settings = node_profile.contribution_settings
    
    # Start with current settings
    optimized = ResourceContributionSettings(**current_settings)
    
    # Increase allocation for underutilized resources
    for resource_type, resource_spec in node_profile.resources.items():
        utilization = resource_spec.allocated_capacity / resource_spec.total_capacity
        
        if utilization < 0.5:  # Increase allocation for underutilized resources
            if resource_type == ResourceType.COMPUTE_CPU:
                optimized.cpu_allocation_percentage = min(optimized.cpu_allocation_percentage + 0.2, 0.8)
            elif resource_type == ResourceType.COMPUTE_GPU:
                optimized.gpu_allocation_percentage = min(optimized.gpu_allocation_percentage + 0.2, 0.9)
            elif resource_type == ResourceType.STORAGE_PERSISTENT:
                optimized.storage_allocation_percentage = min(optimized.storage_allocation_percentage + 0.1, 0.6)
    
    return optimized


async def _summarize_optimizations(old_profile, new_settings) -> List[str]:
    """Summarize applied optimizations"""
    optimizations = []
    
    old_settings = old_profile.contribution_settings
    
    if new_settings.cpu_allocation_percentage > old_settings.get("cpu_allocation_percentage", 0.5):
        optimizations.append("Increased CPU allocation")
    
    if new_settings.gpu_allocation_percentage > old_settings.get("gpu_allocation_percentage", 0.5):
        optimizations.append("Increased GPU allocation")
    
    if new_settings.storage_allocation_percentage > old_settings.get("storage_allocation_percentage", 0.3):
        optimizations.append("Increased storage allocation")
    
    return optimizations


async def _get_resource_analytics(node_id: str, days: int, resource_type: Optional[str]) -> Dict[str, Any]:
    """Get detailed analytics for resource usage"""
    # Would query actual analytics data
    return {
        "utilization_trend": [65, 72, 68, 75, 71, 78, 74],  # Last 7 days
        "earnings_trend": [2.5, 3.1, 2.8, 3.4, 3.0, 3.7, 3.2],  # FTNS per day
        "reputation_trend": [0.85, 0.86, 0.87, 0.88, 0.87, 0.89, 0.88],
        "task_completion_rate": 96.5,
        "average_response_time": 1.2,  # seconds
        "total_tasks_completed": 1247,
        "total_ftns_earned": 89.7
    }


def _parse_period_to_days(period: str) -> int:
    """Parse period string to number of days"""
    period_map = {
        "7d": 7, "1w": 7,
        "30d": 30, "1m": 30,
        "90d": 90, "3m": 90,
        "365d": 365, "1y": 365
    }
    return period_map.get(period, 30)


async def _get_earnings_report(node_id: str, days: int) -> Dict[str, Any]:
    """Get detailed earnings report"""
    # Would query actual earnings data
    total_earnings = 89.7 * (days / 30)  # Scale by period
    
    return {
        "period_days": days,
        "total_ftns_earned": total_earnings,
        "average_daily_earnings": total_earnings / days,
        "earnings_by_resource": {
            "compute_cpu": total_earnings * 0.4,
            "compute_gpu": total_earnings * 0.3,
            "storage_persistent": total_earnings * 0.15,
            "bandwidth": total_earnings * 0.15
        },
        "performance_bonuses": total_earnings * 0.1,
        "reliability_bonuses": total_earnings * 0.05,
        "projected_monthly_earnings": total_earnings * (30 / days) if days < 30 else total_earnings,
        "earnings_rank": "Top 25%",  # Would calculate actual rank
        "optimization_potential": "12% increase possible"
    }


async def _run_resource_benchmark(node_id: str, resource_type: str):
    """Background task to run resource benchmark"""
    try:
        # Would run actual benchmark
        logger.info("Benchmark initiated", node_id=node_id, resource_type=resource_type)
        
        # Simulate benchmark execution
        await asyncio.sleep(60)  # Simulate 1-minute benchmark
        
        logger.info("Benchmark completed", node_id=node_id, resource_type=resource_type)
        
    except Exception as e:
        logger.error("Benchmark failed", node_id=node_id, resource_type=resource_type, error=str(e))


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health")
async def health_check():
    """Health check endpoint for resource management service"""
    try:
        network_summary = get_network_resource_summary()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc),
            "nodes_online": network_summary["total_nodes"],
            "network_health": network_summary["network_health"],
            "service_version": "1.0.0"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc)
            }
        )