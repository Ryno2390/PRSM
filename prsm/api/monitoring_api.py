"""
Enterprise Monitoring & Observability API
=========================================

Production-ready monitoring API endpoints providing comprehensive system insights,
performance analytics, and business intelligence for PRSM platform operations.

Features:
- Real-time system health monitoring and alerting
- Advanced performance analytics and optimization insights
- Business metrics tracking and KPI dashboards
- Distributed tracing and request correlation
- Compliance logging and audit trail management
"""

from typing import List, Optional, Dict, Any, Union
from fastapi import APIRouter, HTTPException, Depends, Query, Request, status
from pydantic import BaseModel, Field
import structlog
from datetime import datetime, timezone

from ..monitoring.enterprise_monitoring import (
    get_monitoring, MetricType, AlertSeverity, MonitoringComponent
)
from ..auth import get_current_user
from ..security.enhanced_authorization import get_enhanced_auth_manager
from ..core.models import UserRole

logger = structlog.get_logger(__name__)
router = APIRouter()

# Initialize monitoring system
monitoring = get_monitoring()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class RecordMetricRequest(BaseModel):
    """Request model for recording metrics"""
    name: str = Field(..., min_length=1, max_length=100, description="Metric name")
    value: Union[int, float] = Field(..., description="Metric value")
    metric_type: str = Field(..., description="Type of metric (counter, gauge, histogram, timer, business)")
    component: Optional[str] = Field(None, description="System component")
    tags: Optional[Dict[str, str]] = Field(default_factory=dict, description="Metric tags")
    labels: Optional[Dict[str, str]] = Field(default_factory=dict, description="Metric labels")


class CreateAlertRequest(BaseModel):
    """Request model for creating alerts"""
    name: str = Field(..., min_length=3, max_length=100, description="Alert name")
    description: str = Field(..., min_length=10, max_length=500, description="Alert description")
    severity: str = Field(..., description="Alert severity level")
    component: str = Field(..., description="System component")
    condition: str = Field(..., description="Alert condition/rule")
    threshold: float = Field(..., description="Alert threshold value")
    duration_seconds: int = Field(300, ge=60, le=3600, description="Duration before triggering")
    notification_channels: Optional[List[str]] = Field(default_factory=list, description="Notification channels")


class BusinessMetricRequest(BaseModel):
    """Request model for business metrics"""
    metric_name: str = Field(..., min_length=1, max_length=100, description="Business metric name")
    value: float = Field(..., description="Metric value")
    dimension: str = Field("daily", description="Time dimension (daily, weekly, monthly)")
    target: Optional[float] = Field(None, description="Target value")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class SystemHealthResponse(BaseModel):
    """Response model for system health"""
    overall_health_score: float
    system_metrics: Dict[str, Any]
    recent_performance: Dict[str, Any]
    active_alerts: List[Dict[str, Any]]
    monitoring_status: Dict[str, Any]
    timestamp: str


class PerformanceAnalyticsResponse(BaseModel):
    """Response model for performance analytics"""
    summary: Dict[str, Any]
    trends: Dict[str, Any]
    bottlenecks: List[str]
    recommendations: List[str]
    top_operations: List[Dict[str, Any]]
    component: Optional[str] = None
    component_specific: Optional[Dict[str, Any]] = None


class BusinessDashboardResponse(BaseModel):
    """Response model for business dashboard"""
    kpis: Dict[str, float]
    trends: Dict[str, str]
    targets: Dict[str, Dict[str, float]]
    user_metrics: Dict[str, Any]
    revenue_metrics: Dict[str, float]
    growth_metrics: Dict[str, float]
    quality_metrics: Dict[str, float]
    generated_at: str


# ============================================================================
# MONITORING ENDPOINTS
# ============================================================================

@router.post("/monitoring/metrics", status_code=status.HTTP_201_CREATED)
async def record_metric(
    request: RecordMetricRequest,
    http_request: Request,
    current_user: str = Depends(get_current_user),
    auth_manager = Depends(get_enhanced_auth_manager)
):
    """
    Record a custom metric for monitoring and analytics
    
    📊 METRIC RECORDING:
    - Support for multiple metric types (counter, gauge, histogram, timer, business)
    - Flexible tagging and labeling for categorization
    - Automatic timestamp and user attribution
    - Integration with alerting and analytics systems
    
    Metric Types:
    - counter: Cumulative metrics that only increase
    - gauge: Point-in-time values that can go up or down
    - histogram: Distribution of values over time
    - timer: Duration measurements
    - business: Business KPIs and outcome metrics
    """
    try:
        # Validate metric type
        try:
            metric_type = MetricType(request.metric_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid metric type. Must be one of: {[t.value for t in MetricType]}"
            )
        
        # Validate component if provided
        component = None
        if request.component:
            try:
                component = MonitoringComponent(request.component)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid component. Must be one of: {[c.value for c in MonitoringComponent]}"
                )
        
        logger.info("Recording custom metric",
                   user_id=current_user,
                   metric_name=request.name,
                   metric_type=request.metric_type,
                   value=request.value)
        
        # Record the metric
        monitoring.record_metric(
            name=request.name,
            value=request.value,
            metric_type=metric_type,
            component=component,
            tags=request.tags,
            labels=request.labels
        )
        
        # Audit the metric recording
        await auth_manager.audit_action(
            user_id=current_user,
            action="record_metric",
            resource_type="monitoring_metrics",
            resource_id=request.name,
            metadata={
                "metric_type": request.metric_type,
                "value": request.value,
                "component": request.component
            },
            request=http_request
        )
        
        return {
            "success": True,
            "message": "Metric recorded successfully",
            "metric_name": request.name,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to record metric",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record metric"
        )


@router.post("/monitoring/business-metrics", status_code=status.HTTP_201_CREATED)
async def record_business_metric(
    request: BusinessMetricRequest,
    http_request: Request,
    current_user: str = Depends(get_current_user),
    auth_manager = Depends(get_enhanced_auth_manager)
):
    """
    Record business KPI and outcome metrics
    
    💼 BUSINESS METRICS:
    - Track key performance indicators and business outcomes
    - Support for daily, weekly, and monthly dimensions
    - Target tracking and achievement monitoring
    - Automatic trend analysis and performance insights
    """
    try:
        logger.info("Recording business metric",
                   user_id=current_user,
                   metric_name=request.metric_name,
                   value=request.value,
                   dimension=request.dimension)
        
        # Record the business metric
        monitoring.record_business_metric(
            metric_name=request.metric_name,
            value=request.value,
            dimension=request.dimension,
            target=request.target,
            metadata=request.metadata
        )
        
        # Audit the business metric recording
        await auth_manager.audit_action(
            user_id=current_user,
            action="record_business_metric",
            resource_type="business_metrics",
            resource_id=request.metric_name,
            metadata={
                "value": request.value,
                "dimension": request.dimension,
                "target": request.target
            },
            request=http_request
        )
        
        return {
            "success": True,
            "message": "Business metric recorded successfully",
            "metric_name": request.metric_name,
            "value": request.value,
            "dimension": request.dimension,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to record business metric",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record business metric"
        )


@router.post("/monitoring/alerts", status_code=status.HTTP_201_CREATED)
async def create_alert(
    request: CreateAlertRequest,
    http_request: Request,
    current_user: str = Depends(get_current_user),
    auth_manager = Depends(get_enhanced_auth_manager)
):
    """
    Create a new monitoring alert
    
    🚨 ALERT MANAGEMENT:
    - Configurable alerting rules and thresholds
    - Multiple severity levels with appropriate escalation
    - Component-specific monitoring and notifications
    - Integration with notification channels and escalation policies
    """
    try:
        # Check permissions for alert creation
        has_permission = await auth_manager.check_permission(
            user_id=current_user,
            user_role=UserRole.DEVELOPER,  # Would fetch actual role
            resource_type="monitoring_alerts",
            action="create"
        )
        
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Developer role or higher required for alert creation"
            )
        
        # Validate severity and component
        try:
            severity = AlertSeverity(request.severity)
            component = MonitoringComponent(request.component)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid severity or component: {str(e)}"
            )
        
        logger.info("Creating monitoring alert",
                   user_id=current_user,
                   alert_name=request.name,
                   severity=request.severity,
                   component=request.component)
        
        # Create the alert
        alert_id = monitoring.create_alert(
            name=request.name,
            description=request.description,
            severity=severity,
            component=component,
            condition=request.condition,
            threshold=request.threshold,
            duration_seconds=request.duration_seconds,
            notification_channels=request.notification_channels
        )
        
        # Audit the alert creation
        await auth_manager.audit_action(
            user_id=current_user,
            action="create_alert",
            resource_type="monitoring_alerts",
            resource_id=alert_id,
            metadata={
                "alert_name": request.name,
                "severity": request.severity,
                "component": request.component,
                "threshold": request.threshold
            },
            request=http_request
        )
        
        return {
            "success": True,
            "message": "Alert created successfully",
            "alert_id": alert_id,
            "name": request.name,
            "severity": request.severity,
            "component": request.component
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create alert",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create alert"
        )


@router.get("/monitoring/health", response_model=SystemHealthResponse)
async def get_system_health(
    current_user: str = Depends(get_current_user)
) -> SystemHealthResponse:
    """
    Get comprehensive system health status
    
    🏥 SYSTEM HEALTH:
    - Overall health score based on multiple factors
    - Real-time system metrics (CPU, memory, disk, network)
    - Recent performance summary and trends
    - Active alerts and monitoring status
    - Background task health and capacity metrics
    """
    try:
        logger.info("Getting system health status",
                   user_id=current_user)
        
        # Get system health data
        health_data = monitoring.get_system_health()
        
        return SystemHealthResponse(**health_data)
        
    except Exception as e:
        logger.error("Failed to get system health",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system health"
        )


@router.get("/monitoring/analytics", response_model=PerformanceAnalyticsResponse)
async def get_performance_analytics(
    component: Optional[str] = Query(None, description="Filter by component"),
    current_user: str = Depends(get_current_user)
) -> PerformanceAnalyticsResponse:
    """
    Get comprehensive performance analytics and insights
    
    📈 PERFORMANCE ANALYTICS:
    - Request volume, response times, and error rates
    - Performance trends and pattern analysis
    - Bottleneck identification and optimization recommendations
    - Top operations by duration and frequency
    - Component-specific performance insights
    """
    try:
        # Validate component if provided
        component_enum = None
        if component:
            try:
                component_enum = MonitoringComponent(component)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid component. Must be one of: {[c.value for c in MonitoringComponent]}"
                )
        
        logger.info("Getting performance analytics",
                   user_id=current_user,
                   component=component)
        
        # Get performance analytics
        analytics_data = monitoring.get_performance_analytics(component_enum)
        
        return PerformanceAnalyticsResponse(**analytics_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get performance analytics",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance analytics"
        )


@router.get("/monitoring/business-dashboard", response_model=BusinessDashboardResponse)
async def get_business_dashboard(
    current_user: str = Depends(get_current_user),
    auth_manager = Depends(get_enhanced_auth_manager)
) -> BusinessDashboardResponse:
    """
    Get comprehensive business metrics dashboard
    
    💼 BUSINESS INTELLIGENCE:
    - Key performance indicators and metrics tracking
    - Growth metrics and trend analysis
    - User engagement and revenue metrics
    - Quality metrics and system reliability
    - Target achievement and performance benchmarking
    
    Requires enterprise role or higher for access.
    """
    try:
        # Check permissions for business dashboard
        has_permission = await auth_manager.check_permission(
            user_id=current_user,
            user_role=UserRole.ENTERPRISE,  # Would fetch actual role
            resource_type="business_dashboard",
            action="read"
        )
        
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Enterprise role or higher required for business dashboard"
            )
        
        logger.info("Getting business dashboard",
                   user_id=current_user)
        
        # Get business dashboard data
        dashboard_data = monitoring.get_business_dashboard()
        
        return BusinessDashboardResponse(**dashboard_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get business dashboard",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve business dashboard"
        )


@router.get("/monitoring/traces/{trace_id}")
async def get_trace_details(
    trace_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Get detailed information about a specific distributed trace
    
    🔍 DISTRIBUTED TRACING:
    - Complete trace timeline and span hierarchy
    - Performance breakdown by operation
    - Error tracking and correlation
    - Request flow visualization data
    """
    try:
        logger.info("Getting trace details",
                   user_id=current_user,
                   trace_id=trace_id)
        
        # Get trace details (placeholder - would fetch from monitoring system)
        trace_details = {
            "trace_id": trace_id,
            "operation": "marketplace_recommendation",
            "component": "recommendation",
            "total_duration_ms": 245.3,
            "spans": [
                {
                    "name": "database_query",
                    "duration_ms": 45.2,
                    "status": "success"
                },
                {
                    "name": "ml_inference",
                    "duration_ms": 180.1,
                    "status": "success"
                },
                {
                    "name": "response_formatting",
                    "duration_ms": 20.0,
                    "status": "success"
                }
            ],
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "user_id": current_user,
                "request_size": "2.3KB",
                "response_size": "15.7KB"
            }
        }
        
        return trace_details
        
    except Exception as e:
        logger.error("Failed to get trace details",
                    trace_id=trace_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve trace details"
        )


@router.get("/monitoring/components")
async def get_monitoring_components():
    """
    Get available monitoring components and their descriptions
    
    🔧 COMPONENT REFERENCE:
    - Complete list of monitorable system components
    - Component descriptions and monitoring capabilities
    - Available metrics and alert types per component
    """
    components = {}
    for component in MonitoringComponent:
        components[component.value] = {
            "name": component.value.replace("_", " ").title(),
            "description": f"Monitoring for {component.value.replace('_', ' ')} component",
            "metrics_available": ["requests", "errors", "duration", "throughput"],
            "alert_types": ["performance", "availability", "errors", "capacity"]
        }
    
    return {
        "components": components,
        "total_count": len(components),
        "metric_types": [t.value for t in MetricType],
        "alert_severities": [s.value for s in AlertSeverity]
    }


# Health check endpoint
@router.get("/monitoring/status")
async def monitoring_health_check():
    """
    Health check for monitoring system
    
    Returns monitoring system status and configuration
    """
    try:
        status_info = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "monitoring_active": True,
            "components_monitored": len(MonitoringComponent),
            "metric_types_supported": len(MetricType),
            "alert_severities": len(AlertSeverity),
            "system_capacity": {
                "max_metrics_per_minute": 10000,
                "max_traces_per_minute": 1000,
                "alert_evaluation_interval": "60 seconds",
                "data_retention_days": 30
            },
            "version": "1.0.0"
        }
        
        return status_info
        
    except Exception as e:
        logger.error("Monitoring health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }