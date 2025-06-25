"""
Security Logging Management API
===============================

REST API endpoints for managing and monitoring the comprehensive security
logging system with real-time metrics and alert management.
"""

import structlog
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Depends, status, Query
from pydantic import BaseModel, Field

from prsm.auth import get_current_user
from prsm.auth.models import UserRole
from prsm.auth.auth_manager import auth_manager
from prsm.security.comprehensive_logging import (
    get_security_logger, 
    LogLevel, 
    EventCategory, 
    AlertSeverity
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1/security/logging", tags=["security-logging"])


# === Request/Response Models ===

class LogSecurityEventRequest(BaseModel):
    """Request to log a security event"""
    event_type: str = Field(description="Type of security event")
    level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    category: EventCategory = Field(description="Event category")
    message: str = Field(description="Event message")
    user_id: Optional[str] = Field(default=None, description="User ID associated with event")
    ip_address: Optional[str] = Field(default=None, description="IP address")
    risk_score: int = Field(default=0, ge=0, le=100, description="Risk score (0-100)")
    threat_indicators: List[str] = Field(default_factory=list, description="List of threat indicators")
    compliance_flags: List[str] = Field(default_factory=list, description="Compliance flags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CreateAlertRuleRequest(BaseModel):
    """Request to create a security alert rule"""
    rule_id: str = Field(description="Unique rule identifier")
    description: str = Field(description="Rule description")
    condition_config: Dict[str, Any] = Field(description="Alert condition configuration")
    severity: AlertSeverity = Field(description="Alert severity")
    notification_channels: List[str] = Field(description="Notification channels")
    enabled: bool = Field(default=True, description="Whether rule is enabled")


class SecurityMetricsQuery(BaseModel):
    """Query parameters for security metrics"""
    hours: int = Field(default=24, ge=1, le=168, description="Time period in hours")
    category: Optional[EventCategory] = Field(default=None, description="Filter by category")
    level: Optional[LogLevel] = Field(default=None, description="Filter by log level")
    user_id: Optional[str] = Field(default=None, description="Filter by user ID")


class SecurityLogResponse(BaseModel):
    """Standard security logging response"""
    success: bool
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# === Security Event Logging ===

@router.post("/events", response_model=SecurityLogResponse)
async def log_security_event(
    request: LogSecurityEventRequest,
    current_user: str = Depends(get_current_user)
) -> SecurityLogResponse:
    """
    Log a security event to the comprehensive logging system
    
    🔐 SECURITY EVENT LOGGING:
    - Records security events with comprehensive metadata
    - Applies risk scoring and threat indicator tracking
    - Triggers real-time alerts based on configured rules
    - Maintains audit trails for compliance requirements
    - Supports multiple event categories and severity levels
    """
    try:
        security_logger = await get_security_logger()
        
        # Log the security event
        await security_logger.log_security_event(
            event_type=request.event_type,
            level=request.level,
            category=request.category,
            message=request.message,
            user_id=request.user_id or current_user,
            ip_address=request.ip_address,
            risk_score=request.risk_score,
            threat_indicators=request.threat_indicators,
            compliance_flags=request.compliance_flags,
            metadata=request.metadata,
            component="api_request"
        )
        
        return SecurityLogResponse(
            success=True,
            message="✅ Security event logged successfully",
            data={
                "event_type": request.event_type,
                "level": request.level.value,
                "category": request.category.value,
                "risk_score": request.risk_score,
                "logged_by": current_user
            }
        )
        
    except Exception as e:
        logger.error("Failed to log security event",
                    user_id=current_user,
                    event_type=request.event_type,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to log security event"
        )


@router.post("/events/authentication", response_model=SecurityLogResponse)
async def log_authentication_event(
    event_type: str,
    success: bool,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    current_user: str = Depends(get_current_user)
) -> SecurityLogResponse:
    """
    Log authentication-specific events
    
    🔑 AUTHENTICATION LOGGING:
    - Specialized logging for login, logout, and authentication events
    - Automatic risk scoring based on success/failure patterns
    - IP address and user agent tracking
    - Integration with threat detection systems
    """
    try:
        security_logger = await get_security_logger()
        
        await security_logger.log_authentication_event(
            event_type=event_type,
            user_id=current_user,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            component="authentication_api"
        )
        
        return SecurityLogResponse(
            success=True,
            message=f"✅ Authentication event logged: {event_type}",
            data={
                "event_type": event_type,
                "success": success,
                "user_id": current_user,
                "ip_address": ip_address
            }
        )
        
    except Exception as e:
        logger.error("Failed to log authentication event",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to log authentication event"
        )


@router.post("/events/web3", response_model=SecurityLogResponse)
async def log_web3_event(
    tx_hash: str,
    contract_address: str,
    method: str,
    success: bool,
    gas_used: Optional[int] = None,
    current_user: str = Depends(get_current_user)
) -> SecurityLogResponse:
    """
    Log Web3 transaction events
    
    ⛓️ WEB3 TRANSACTION LOGGING:
    - Comprehensive Web3 transaction tracking
    - Smart contract interaction monitoring
    - Gas usage and cost analysis
    - Blockchain event correlation
    """
    try:
        security_logger = await get_security_logger()
        
        await security_logger.log_web3_transaction(
            tx_hash=tx_hash,
            user_id=current_user,
            contract_address=contract_address,
            method=method,
            success=success,
            metadata={
                "gas_used": gas_used
            },
            component="web3_api"
        )
        
        return SecurityLogResponse(
            success=True,
            message="✅ Web3 transaction event logged",
            data={
                "tx_hash": tx_hash,
                "contract_address": contract_address,
                "method": method,
                "success": success,
                "user_id": current_user
            }
        )
        
    except Exception as e:
        logger.error("Failed to log Web3 event",
                    user_id=current_user,
                    tx_hash=tx_hash,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to log Web3 event"
        )


# === Security Metrics and Monitoring ===

@router.get("/metrics", response_model=SecurityLogResponse)
async def get_security_metrics(
    hours: int = Query(default=24, ge=1, le=168, description="Time period in hours"),
    category: Optional[EventCategory] = Query(default=None, description="Filter by category"),
    current_user: str = Depends(get_current_user)
) -> SecurityLogResponse:
    """
    Get comprehensive security metrics and statistics
    
    📊 SECURITY METRICS:
    - Event volume and distribution by category
    - Risk score trends and anomaly detection
    - Alert frequency and response metrics
    - User activity patterns and threat indicators
    - System performance and logging statistics
    """
    try:
        security_logger = await get_security_logger()
        
        metrics = await security_logger.get_security_metrics(hours=hours)
        
        return SecurityLogResponse(
            success=True,
            message="📊 Security metrics retrieved successfully",
            data={
                "metrics": metrics,
                "query_parameters": {
                    "hours": hours,
                    "category": category.value if category else None
                },
                "retrieved_by": current_user
            }
        )
        
    except Exception as e:
        logger.error("Failed to get security metrics",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security metrics"
        )


@router.get("/alerts/recent", response_model=SecurityLogResponse)
async def get_recent_alerts(
    limit: int = Query(default=50, ge=1, le=500, description="Maximum number of alerts"),
    severity: Optional[AlertSeverity] = Query(default=None, description="Filter by severity"),
    current_user: str = Depends(get_current_user)
) -> SecurityLogResponse:
    """
    Get recent security alerts
    
    🚨 SECURITY ALERTS:
    - Recent security alerts with severity filtering
    - Alert frequency and pattern analysis
    - Threat progression tracking
    - Response status and resolution tracking
    """
    try:
        # Check admin permissions for security alerts
        user = await auth_manager.get_user_by_id(current_user)
        if not user or user.role not in [UserRole.ADMIN, UserRole.MODERATOR]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin permissions required to view security alerts"
            )
        
        security_logger = await get_security_logger()
        
        # Get recent alerts from alert manager
        all_alerts = security_logger.alert_manager.alert_history
        
        # Filter by severity if specified
        if severity:
            filtered_alerts = [
                alert for alert in all_alerts
                if alert["severity"] == severity.value
            ]
        else:
            filtered_alerts = all_alerts
        
        # Sort by timestamp and limit
        recent_alerts = sorted(
            filtered_alerts,
            key=lambda x: x["triggered_at"],
            reverse=True
        )[:limit]
        
        return SecurityLogResponse(
            success=True,
            message=f"🚨 Retrieved {len(recent_alerts)} recent alerts",
            data={
                "alerts": recent_alerts,
                "total_alerts": len(all_alerts),
                "filtered_count": len(filtered_alerts),
                "query_parameters": {
                    "limit": limit,
                    "severity": severity.value if severity else None
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get recent alerts",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve recent alerts"
        )


# === Alert Management ===

@router.post("/alerts/rules", response_model=SecurityLogResponse)
async def create_alert_rule(
    request: CreateAlertRuleRequest,
    current_user: str = Depends(get_current_user)
) -> SecurityLogResponse:
    """
    Create a new security alert rule (Admin only)
    
    ⚙️ ALERT RULE CREATION:
    - Custom alert rule configuration
    - Condition-based triggering
    - Multi-channel notification support
    - Rule enablement and management
    """
    try:
        # Check admin permissions
        user = await auth_manager.get_user_by_id(current_user)
        if not user or user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin permissions required to create alert rules"
            )
        
        security_logger = await get_security_logger()
        
        # Create a simple condition function based on config
        # In production, this would be more sophisticated
        def condition_func(log_entry):
            config = request.condition_config
            
            # Check category match
            if "category" in config and log_entry.category.value != config["category"]:
                return False
            
            # Check level match
            if "level" in config and log_entry.level.value != config["level"]:
                return False
            
            # Check risk score threshold
            if "min_risk_score" in config and log_entry.risk_score < config["min_risk_score"]:
                return False
            
            # Check for keywords in message
            if "keywords" in config:
                keywords = config["keywords"]
                message_lower = log_entry.message.lower()
                if not any(keyword.lower() in message_lower for keyword in keywords):
                    return False
            
            return True
        
        # Add the alert rule
        security_logger.alert_manager.add_alert_rule(
            rule_id=request.rule_id,
            condition=condition_func,
            severity=request.severity,
            notification_channels=request.notification_channels
        )
        
        return SecurityLogResponse(
            success=True,
            message=f"✅ Alert rule '{request.rule_id}' created successfully",
            data={
                "rule_id": request.rule_id,
                "description": request.description,
                "severity": request.severity.value,
                "channels": request.notification_channels,
                "enabled": request.enabled,
                "created_by": current_user
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create alert rule",
                    admin_user=current_user,
                    rule_id=request.rule_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create alert rule"
        )


@router.get("/alerts/rules", response_model=SecurityLogResponse)
async def list_alert_rules(
    current_user: str = Depends(get_current_user)
) -> SecurityLogResponse:
    """
    List all configured alert rules (Admin only)
    
    📋 ALERT RULES:
    - List of all configured security alert rules
    - Rule status and trigger statistics
    - Configuration details and conditions
    - Performance metrics for each rule
    """
    try:
        # Check admin permissions
        user = await auth_manager.get_user_by_id(current_user)
        if not user or user.role not in [UserRole.ADMIN, UserRole.MODERATOR]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin permissions required to view alert rules"
            )
        
        security_logger = await get_security_logger()
        
        # Get alert rules with statistics
        rules_info = []
        for rule_id, rule_data in security_logger.alert_manager.alert_rules.items():
            rules_info.append({
                "rule_id": rule_id,
                "severity": rule_data["severity"].value,
                "notification_channels": rule_data["channels"],
                "last_triggered": rule_data["last_triggered"].isoformat() if rule_data["last_triggered"] else None,
                "trigger_count": rule_data["trigger_count"]
            })
        
        return SecurityLogResponse(
            success=True,
            message="📋 Alert rules retrieved successfully",
            data={
                "rules": rules_info,
                "total_rules": len(rules_info)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list alert rules",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve alert rules"
        )


# === System Status and Health ===

@router.get("/status", response_model=SecurityLogResponse)
async def get_logging_system_status(
    current_user: str = Depends(get_current_user)
) -> SecurityLogResponse:
    """
    Get security logging system status and health
    
    🏥 SYSTEM STATUS:
    - Logging system health and performance
    - Queue status and processing metrics
    - Disk usage and log rotation status
    - Alert system functionality
    - Component status and diagnostics
    """
    try:
        security_logger = await get_security_logger()
        
        system_status = {
            "status": "healthy",
            "uptime_hours": (datetime.now(timezone.utc) - security_logger.stats["start_time"]).total_seconds() / 3600,
            "logging_queue": {
                "current_size": security_logger.log_queue.qsize(),
                "max_size": security_logger.log_queue.maxsize,
                "utilization_percent": (security_logger.log_queue.qsize() / security_logger.log_queue.maxsize) * 100
            },
            "statistics": security_logger.stats,
            "configuration": {
                "log_directory": str(security_logger.log_dir),
                "max_log_size_mb": security_logger.config["max_log_size_mb"],
                "retention_days": security_logger.config["retention_days"],
                "alerts_enabled": security_logger.config["enable_real_time_alerts"],
                "metrics_enabled": security_logger.config["enable_metrics"]
            },
            "alert_system": {
                "total_rules": len(security_logger.alert_manager.alert_rules),
                "total_alerts": len(security_logger.alert_manager.alert_history),
                "recent_alerts_24h": len([
                    alert for alert in security_logger.alert_manager.alert_history
                    if datetime.fromisoformat(alert["triggered_at"]) > datetime.now(timezone.utc) - timedelta(hours=24)
                ])
            }
        }
        
        return SecurityLogResponse(
            success=True,
            message="🏥 Security logging system status retrieved",
            data={"system_status": system_status}
        )
        
    except Exception as e:
        logger.error("Failed to get logging system status",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system status"
        )