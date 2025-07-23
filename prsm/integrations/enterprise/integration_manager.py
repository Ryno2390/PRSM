#!/usr/bin/env python3
"""
Enterprise Integration Manager
==============================

Comprehensive integration monitoring, management, and orchestration system
for enterprise data connectivity and processing workflows.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable
import uuid
from pathlib import Path

from .data_connectors import DataConnector, ConnectionConfig, ConnectionStatus
from .etl_pipeline import PipelineManager, ETLPipeline, PipelineStatus
from .transformation_engine import TransformationEngine
from .sync_manager import DataSyncManager, SyncConfiguration

logger = logging.getLogger(__name__)


class IntegrationType(Enum):
    """Types of integrations"""
    DATA_CONNECTOR = "data_connector"
    ETL_PIPELINE = "etl_pipeline"
    DATA_SYNC = "data_sync"
    API_INTEGRATION = "api_integration"
    WEBHOOK = "webhook"
    STREAM_PROCESSING = "stream_processing"


class IntegrationStatus(Enum):
    """Integration status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive" 
    ERROR = "error"
    CONFIGURING = "configuring"
    TESTING = "testing"
    DEPRECATED = "deprecated"


class HealthStatus(Enum):
    """Health check status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class Integration:
    """Integration record"""
    integration_id: str
    name: str
    integration_type: IntegrationType
    description: str = ""
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Status and health
    status: IntegrationStatus = IntegrationStatus.INACTIVE
    health_status: HealthStatus = HealthStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    
    # Monitoring
    monitoring_enabled: bool = True
    alert_thresholds: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Error tracking
    error_count: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "integration_id": self.integration_id,
            "name": self.name,
            "integration_type": self.integration_type.value,
            "description": self.description,
            "config": self.config,
            "status": self.status.value,
            "health_status": self.health_status.value,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "dependencies": self.dependencies,
            "dependents": self.dependents,
            "monitoring_enabled": self.monitoring_enabled,
            "alert_thresholds": self.alert_thresholds,
            "metrics": self.metrics,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "tags": self.tags
        }


@dataclass
class HealthCheckResult:
    """Result of integration health check"""
    integration_id: str
    status: HealthStatus
    timestamp: datetime
    response_time_ms: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "integration_id": self.integration_id,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "response_time_ms": self.response_time_ms,
            "message": self.message,
            "details": self.details
        }


@dataclass
class Alert:
    """Integration alert"""
    alert_id: str
    integration_id: str
    alert_type: str
    severity: str  # low, medium, high, critical
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "alert_id": self.alert_id,
            "integration_id": self.integration_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


class IntegrationMonitor:
    """Health monitoring and alerting system"""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheckResult] = {}
        self.alerts: Dict[str, Alert] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        
        # Alert handlers
        self.alert_handlers: List[Callable] = []
        
        # Statistics
        self.stats = {
            "total_health_checks": 0,
            "healthy_integrations": 0,
            "unhealthy_integrations": 0,
            "total_alerts": 0,
            "active_alerts": 0,
            "resolved_alerts": 0
        }
    
    def add_alert_handler(self, handler: Callable):
        """Add alert handler"""
        self.alert_handlers.append(handler)
        logger.info(f"Added alert handler: {handler.__name__}")
    
    async def start_monitoring(self, integration: Integration):
        """Start monitoring an integration"""
        if integration.integration_id in self.monitoring_tasks:
            logger.warning(f"Already monitoring integration: {integration.name}")
            return
        
        if not integration.monitoring_enabled:
            logger.info(f"Monitoring disabled for integration: {integration.name}")
            return
        
        # Start monitoring task
        task = asyncio.create_task(self._monitor_integration(integration))
        self.monitoring_tasks[integration.integration_id] = task
        
        logger.info(f"Started monitoring integration: {integration.name}")
    
    async def stop_monitoring(self, integration_id: str):
        """Stop monitoring an integration"""
        if integration_id not in self.monitoring_tasks:
            return
        
        task = self.monitoring_tasks[integration_id]
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        del self.monitoring_tasks[integration_id]
        logger.info(f"Stopped monitoring integration: {integration_id}")
    
    async def _monitor_integration(self, integration: Integration):
        """Monitor integration health"""
        check_interval = integration.config.get("health_check_interval", 60)  # seconds
        
        while True:
            try:
                # Perform health check
                health_result = await self._perform_health_check(integration)
                
                # Store result
                self.health_checks[integration.integration_id] = health_result
                
                # Update statistics
                self.stats["total_health_checks"] += 1
                if health_result.status == HealthStatus.HEALTHY:
                    self.stats["healthy_integrations"] += 1
                else:
                    self.stats["unhealthy_integrations"] += 1
                
                # Check for alerts
                await self._check_alert_conditions(integration, health_result)
                
                # Wait before next check
                await asyncio.sleep(check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error for {integration.name}: {e}")
                await asyncio.sleep(check_interval)
    
    async def _perform_health_check(self, integration: Integration) -> HealthCheckResult:
        """Perform health check on integration"""
        start_time = datetime.now()
        
        try:
            if integration.integration_type == IntegrationType.DATA_CONNECTOR:
                return await self._check_connector_health(integration)
            elif integration.integration_type == IntegrationType.ETL_PIPELINE:
                return await self._check_pipeline_health(integration)
            elif integration.integration_type == IntegrationType.DATA_SYNC:
                return await self._check_sync_health(integration)
            elif integration.integration_type == IntegrationType.API_INTEGRATION:
                return await self._check_api_health(integration)
            else:
                return HealthCheckResult(
                    integration_id=integration.integration_id,
                    status=HealthStatus.UNKNOWN,
                    timestamp=start_time,
                    message=f"Unknown integration type: {integration.integration_type}"
                )
                
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return HealthCheckResult(
                integration_id=integration.integration_id,
                status=HealthStatus.CRITICAL,
                timestamp=start_time,
                response_time_ms=response_time,
                message=f"Health check failed: {e}"
            )
    
    async def _check_connector_health(self, integration: Integration) -> HealthCheckResult:
        """Check data connector health"""
        start_time = datetime.now()
        
        # This would check actual connector health
        # For now, simulate a basic check
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return HealthCheckResult(
            integration_id=integration.integration_id,
            status=HealthStatus.HEALTHY,
            timestamp=start_time,
            response_time_ms=response_time,
            message="Connector is responsive"
        )
    
    async def _check_pipeline_health(self, integration: Integration) -> HealthCheckResult:
        """Check ETL pipeline health"""
        start_time = datetime.now()
        
        # Check pipeline status, recent executions, error rates
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return HealthCheckResult(
            integration_id=integration.integration_id,
            status=HealthStatus.HEALTHY,
            timestamp=start_time,
            response_time_ms=response_time,
            message="Pipeline is running normally"
        )
    
    async def _check_sync_health(self, integration: Integration) -> HealthCheckResult:
        """Check data sync health"""
        start_time = datetime.now()
        
        # Check sync status, lag, error rates
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return HealthCheckResult(
            integration_id=integration.integration_id,
            status=HealthStatus.HEALTHY,
            timestamp=start_time,
            response_time_ms=response_time,
            message="Sync is operating normally"
        )
    
    async def _check_api_health(self, integration: Integration) -> HealthCheckResult:
        """Check API integration health"""
        start_time = datetime.now()
        
        # Make test API call
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return HealthCheckResult(
            integration_id=integration.integration_id,
            status=HealthStatus.HEALTHY,
            timestamp=start_time,
            response_time_ms=response_time,
            message="API is accessible"
        )
    
    async def _check_alert_conditions(self, integration: Integration, 
                                     health_result: HealthCheckResult):
        """Check if alert conditions are met"""
        # Check response time threshold
        response_time_threshold = integration.alert_thresholds.get("response_time_ms", 5000)
        if health_result.response_time_ms > response_time_threshold:
            await self._create_alert(
                integration.integration_id,
                "high_response_time",
                "medium",
                f"Response time {health_result.response_time_ms:.2f}ms exceeds threshold {response_time_threshold}ms",
                {"response_time_ms": health_result.response_time_ms, "threshold": response_time_threshold}
            )
        
        # Check health status
        if health_result.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
            severity = "critical" if health_result.status == HealthStatus.CRITICAL else "medium"
            await self._create_alert(
                integration.integration_id,
                "health_check_failure",
                severity,
                f"Integration health check failed: {health_result.message}",
                {"health_status": health_result.status.value}
            )
        
        # Check error rate (would need actual error tracking)
        error_rate_threshold = integration.alert_thresholds.get("error_rate_percent", 5.0)
        current_error_rate = integration.metrics.get("error_rate_percent", 0.0)
        
        if current_error_rate > error_rate_threshold:
            await self._create_alert(
                integration.integration_id,
                "high_error_rate",
                "high",
                f"Error rate {current_error_rate:.2f}% exceeds threshold {error_rate_threshold}%",
                {"error_rate_percent": current_error_rate, "threshold": error_rate_threshold}
            )
    
    async def _create_alert(self, integration_id: str, alert_type: str, 
                           severity: str, message: str, details: Dict[str, Any]):
        """Create a new alert"""
        alert_id = f"alert_{uuid.uuid4().hex[:8]}"
        
        alert = Alert(
            alert_id=alert_id,
            integration_id=integration_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            details=details
        )
        
        self.alerts[alert_id] = alert
        self.stats["total_alerts"] += 1
        self.stats["active_alerts"] += 1
        
        # Notify alert handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
        logger.warning(f"Alert created: {alert_type} for integration {integration_id}")
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now(timezone.utc)
            
            logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
    
    async def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            
            if not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now(timezone.utc)
                
                self.stats["active_alerts"] -= 1
                self.stats["resolved_alerts"] += 1
                
                logger.info(f"Alert resolved: {alert_id}")
    
    def get_integration_health(self, integration_id: str) -> Optional[HealthCheckResult]:
        """Get latest health check result for integration"""
        return self.health_checks.get(integration_id)
    
    def get_active_alerts(self, integration_id: Optional[str] = None) -> List[Alert]:
        """Get active alerts"""
        alerts = [alert for alert in self.alerts.values() if not alert.resolved]
        
        if integration_id:
            alerts = [alert for alert in alerts if alert.integration_id == integration_id]
        
        return alerts
    
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            **self.stats,
            "monitored_integrations": len(self.monitoring_tasks),
            "alert_handlers": len(self.alert_handlers)
        }


class EnterpriseIntegrationManager:
    """Main enterprise integration manager"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./integration_data")
        self.storage_path.mkdir(exist_ok=True)
        
        # Component managers
        self.pipeline_manager = PipelineManager()
        self.sync_manager = DataSyncManager()
        self.transformation_engine = TransformationEngine()
        self.monitor = IntegrationMonitor()
        
        # Integration registry
        self.integrations: Dict[str, Integration] = {}
        self.connectors: Dict[str, DataConnector] = {}
        
        # Dependency graph
        self.dependency_graph: Dict[str, List[str]] = {}
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Statistics
        self.manager_stats = {
            "total_integrations": 0,
            "active_integrations": 0,
            "healthy_integrations": 0,
            "error_integrations": 0,
            "total_data_processed": 0,
            "uptime_start": datetime.now(timezone.utc)
        }
        
        # Setup alert handlers
        self.monitor.add_alert_handler(self._handle_integration_alert)
        
        logger.info("Enterprise Integration Manager initialized")
    
    def register_integration(self, integration: Integration):
        """Register a new integration"""
        self.integrations[integration.integration_id] = integration
        self.manager_stats["total_integrations"] += 1
        
        # Build dependency graph
        self.dependency_graph[integration.integration_id] = integration.dependencies.copy()
        
        # Update dependents
        for dep_id in integration.dependencies:
            if dep_id in self.integrations:
                if integration.integration_id not in self.integrations[dep_id].dependents:
                    self.integrations[dep_id].dependents.append(integration.integration_id)
        
        # Start monitoring if enabled
        if integration.monitoring_enabled:
            asyncio.create_task(self.monitor.start_monitoring(integration))
        
        logger.info(f"Registered integration: {integration.name}")
        
        # Emit event
        asyncio.create_task(self._emit_event("integration_registered", integration.to_dict()))
    
    def register_connector(self, connector: DataConnector):
        """Register a data connector"""
        self.connectors[connector.config.connector_id] = connector
        
        # Register with sub-managers
        self.pipeline_manager.register_connector(connector)
        self.sync_manager.register_connector(connector)
        
        logger.info(f"Registered connector: {connector.config.name}")
    
    async def activate_integration(self, integration_id: str) -> bool:
        """Activate an integration"""
        if integration_id not in self.integrations:
            logger.error(f"Integration not found: {integration_id}")
            return False
        
        integration = self.integrations[integration_id]
        
        try:
            # Check dependencies are active
            for dep_id in integration.dependencies:
                if dep_id in self.integrations:
                    dep_integration = self.integrations[dep_id]
                    if dep_integration.status != IntegrationStatus.ACTIVE:
                        logger.error(f"Dependency {dep_id} is not active for integration {integration_id}")
                        return False
            
            # Activate based on type
            if integration.integration_type == IntegrationType.DATA_CONNECTOR:
                success = await self._activate_connector(integration)
            elif integration.integration_type == IntegrationType.ETL_PIPELINE:
                success = await self._activate_pipeline(integration)
            elif integration.integration_type == IntegrationType.DATA_SYNC:
                success = await self._activate_sync(integration)
            else:
                success = await self._activate_generic(integration)
            
            if success:
                integration.status = IntegrationStatus.ACTIVE
                integration.updated_at = datetime.now(timezone.utc)
                self.manager_stats["active_integrations"] += 1
                
                # Start monitoring
                if integration.monitoring_enabled:
                    await self.monitor.start_monitoring(integration)
                
                logger.info(f"Activated integration: {integration.name}")
                
                # Emit event
                await self._emit_event("integration_activated", integration.to_dict())
            
            return success
            
        except Exception as e:
            integration.status = IntegrationStatus.ERROR
            integration.last_error = str(e)
            integration.last_error_time = datetime.now(timezone.utc)
            integration.error_count += 1
            
            logger.error(f"Failed to activate integration {integration.name}: {e}")
            return False
    
    async def deactivate_integration(self, integration_id: str) -> bool:
        """Deactivate an integration"""
        if integration_id not in self.integrations:
            logger.error(f"Integration not found: {integration_id}")
            return False
        
        integration = self.integrations[integration_id]
        
        try:
            # Check dependents
            active_dependents = [
                dep_id for dep_id in integration.dependents
                if dep_id in self.integrations and 
                self.integrations[dep_id].status == IntegrationStatus.ACTIVE
            ]
            
            if active_dependents:
                logger.warning(f"Integration {integration_id} has active dependents: {active_dependents}")
                # Could force deactivate or return False based on policy
            
            # Deactivate based on type
            if integration.integration_type == IntegrationType.DATA_CONNECTOR:
                success = await self._deactivate_connector(integration)
            elif integration.integration_type == IntegrationType.ETL_PIPELINE:
                success = await self._deactivate_pipeline(integration)
            elif integration.integration_type == IntegrationType.DATA_SYNC:
                success = await self._deactivate_sync(integration)
            else:
                success = await self._deactivate_generic(integration)
            
            if success:
                integration.status = IntegrationStatus.INACTIVE
                integration.updated_at = datetime.now(timezone.utc)
                self.manager_stats["active_integrations"] -= 1
                
                # Stop monitoring
                await self.monitor.stop_monitoring(integration_id)
                
                logger.info(f"Deactivated integration: {integration.name}")
                
                # Emit event
                await self._emit_event("integration_deactivated", integration.to_dict())
            
            return success
            
        except Exception as e:
            integration.status = IntegrationStatus.ERROR
            integration.last_error = str(e)
            integration.last_error_time = datetime.now(timezone.utc)
            integration.error_count += 1
            
            logger.error(f"Failed to deactivate integration {integration.name}: {e}")
            return False
    
    async def _activate_connector(self, integration: Integration) -> bool:
        """Activate data connector"""
        connector_id = integration.config.get("connector_id")
        if connector_id and connector_id in self.connectors:
            connector = self.connectors[connector_id]
            return await connector.connect()
        return False
    
    async def _deactivate_connector(self, integration: Integration) -> bool:
        """Deactivate data connector"""
        connector_id = integration.config.get("connector_id")
        if connector_id and connector_id in self.connectors:
            connector = self.connectors[connector_id]
            return await connector.disconnect()
        return True
    
    async def _activate_pipeline(self, integration: Integration) -> bool:
        """Activate ETL pipeline"""
        pipeline_id = integration.config.get("pipeline_id")
        if pipeline_id:
            # Start pipeline scheduler or execution
            await self.pipeline_manager.start_scheduler()
            return True
        return False
    
    async def _deactivate_pipeline(self, integration: Integration) -> bool:
        """Deactivate ETL pipeline"""
        pipeline_id = integration.config.get("pipeline_id")
        if pipeline_id:
            # Stop pipeline execution
            return True
        return True
    
    async def _activate_sync(self, integration: Integration) -> bool:
        """Activate data sync"""
        sync_id = integration.config.get("sync_id")
        if sync_id:
            sync_config = integration.config.get("sync_config")
            if sync_config and sync_config.get("sync_type") == "real_time":
                return await self.sync_manager.start_realtime_sync(sync_id)
            else:
                # For batch sync, just register - execution is triggered separately
                return True
        return False
    
    async def _deactivate_sync(self, integration: Integration) -> bool:
        """Deactivate data sync"""
        sync_id = integration.config.get("sync_id")
        if sync_id:
            return await self.sync_manager.stop_realtime_sync(sync_id)
        return True
    
    async def _activate_generic(self, integration: Integration) -> bool:
        """Activate generic integration"""
        # Generic activation logic
        logger.info(f"Activating generic integration: {integration.name}")
        return True
    
    async def _deactivate_generic(self, integration: Integration) -> bool:
        """Deactivate generic integration"""
        # Generic deactivation logic
        logger.info(f"Deactivating generic integration: {integration.name}")
        return True
    
    async def test_integration(self, integration_id: str) -> Dict[str, Any]:
        """Test integration functionality"""
        if integration_id not in self.integrations:
            return {"success": False, "error": "Integration not found"}
        
        integration = self.integrations[integration_id]
        integration.status = IntegrationStatus.TESTING
        
        try:
            # Perform health check
            health_result = await self.monitor._perform_health_check(integration)
            
            # Additional testing based on integration type
            test_results = {}
            
            if integration.integration_type == IntegrationType.DATA_CONNECTOR:
                test_results = await self._test_connector(integration)
            elif integration.integration_type == IntegrationType.ETL_PIPELINE:
                test_results = await self._test_pipeline(integration)
            elif integration.integration_type == IntegrationType.DATA_SYNC:
                test_results = await self._test_sync(integration)
            
            return {
                "success": True,
                "health_check": health_result.to_dict(),
                "test_results": test_results,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        finally:
            integration.status = IntegrationStatus.INACTIVE
    
    async def _test_connector(self, integration: Integration) -> Dict[str, Any]:
        """Test data connector"""
        connector_id = integration.config.get("connector_id")
        if connector_id and connector_id in self.connectors:
            connector = self.connectors[connector_id]
            
            # Test connection
            can_connect = await connector.test_connection()
            
            return {
                "connection_test": can_connect,
                "connector_status": connector.get_status()
            }
        
        return {"error": "Connector not found"}
    
    async def _test_pipeline(self, integration: Integration) -> Dict[str, Any]:
        """Test ETL pipeline"""
        pipeline_id = integration.config.get("pipeline_id")
        
        # Get pipeline status from manager
        pipeline_status = self.pipeline_manager.get_pipeline_status(pipeline_id)
        
        return {
            "pipeline_status": pipeline_status
        }
    
    async def _test_sync(self, integration: Integration) -> Dict[str, Any]:
        """Test data sync"""
        sync_id = integration.config.get("sync_id")
        
        # Get sync status from manager
        sync_status = self.sync_manager.get_sync_status(sync_id)
        
        return {
            "sync_status": sync_status
        }
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        logger.info(f"Added event handler for {event_type}: {handler.__name__}")
    
    async def _emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit integration event"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    await handler(event_data)
                except Exception as e:
                    logger.error(f"Event handler error for {event_type}: {e}")
    
    async def _handle_integration_alert(self, alert: Alert):
        """Handle integration alert"""
        if alert.integration_id in self.integrations:
            integration = self.integrations[alert.integration_id]
            
            # Update integration metrics
            integration.metrics["last_alert"] = alert.timestamp.isoformat()
            integration.metrics["alert_count"] = integration.metrics.get("alert_count", 0) + 1
            
            # Update manager statistics
            if alert.severity == "critical":
                self.manager_stats["error_integrations"] += 1
            
            logger.warning(f"Integration alert: {alert.message} for {integration.name}")
    
    def get_integration(self, integration_id: str) -> Optional[Integration]:
        """Get integration by ID"""
        return self.integrations.get(integration_id)
    
    def list_integrations(self, status: Optional[IntegrationStatus] = None,
                         integration_type: Optional[IntegrationType] = None) -> List[Integration]:
        """List integrations with optional filtering"""
        integrations = list(self.integrations.values())
        
        if status:
            integrations = [i for i in integrations if i.status == status]
        
        if integration_type:
            integrations = [i for i in integrations if i.integration_type == integration_type]
        
        return integrations
    
    def get_integration_dependencies(self, integration_id: str) -> Dict[str, List[str]]:
        """Get integration dependency information"""
        if integration_id not in self.integrations:
            return {}
        
        integration = self.integrations[integration_id]
        
        return {
            "dependencies": integration.dependencies,
            "dependents": integration.dependents,
            "dependency_status": {
                dep_id: self.integrations[dep_id].status.value
                for dep_id in integration.dependencies
                if dep_id in self.integrations
            }
        }
    
    def get_integration_metrics(self, integration_id: str) -> Dict[str, Any]:
        """Get comprehensive integration metrics"""
        if integration_id not in self.integrations:
            return {}
        
        integration = self.integrations[integration_id]
        
        # Get health status
        health_result = self.monitor.get_integration_health(integration_id)
        
        # Get active alerts
        active_alerts = self.monitor.get_active_alerts(integration_id)
        
        return {
            "integration": integration.to_dict(),
            "health": health_result.to_dict() if health_result else None,
            "active_alerts": [alert.to_dict() for alert in active_alerts],
            "dependencies": self.get_integration_dependencies(integration_id)
        }
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get system-wide integration overview"""
        # Count integrations by status
        status_counts = {}
        type_counts = {}
        
        for integration in self.integrations.values():
            status_counts[integration.status.value] = status_counts.get(integration.status.value, 0) + 1
            type_counts[integration.integration_type.value] = type_counts.get(integration.integration_type.value, 0) + 1
        
        # Get health overview
        healthy_count = sum(1 for i in self.integrations.values() 
                          if self.monitor.get_integration_health(i.integration_id) and
                          self.monitor.get_integration_health(i.integration_id).status == HealthStatus.HEALTHY)
        
        # Get alert overview
        active_alerts = self.monitor.get_active_alerts()
        alert_severity_counts = {}
        for alert in active_alerts:
            alert_severity_counts[alert.severity] = alert_severity_counts.get(alert.severity, 0) + 1
        
        uptime = datetime.now(timezone.utc) - self.manager_stats["uptime_start"]
        
        return {
            "total_integrations": len(self.integrations),
            "status_breakdown": status_counts,
            "type_breakdown": type_counts,
            "health_summary": {
                "healthy": healthy_count,
                "unhealthy": len(self.integrations) - healthy_count
            },
            "alert_summary": {
                "total_active": len(active_alerts),
                "by_severity": alert_severity_counts
            },
            "system_uptime_seconds": uptime.total_seconds(),
            "manager_statistics": self.manager_stats,
            "monitoring_statistics": self.monitor.get_monitoring_statistics(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def perform_system_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        health_results = {}
        
        for integration_id, integration in self.integrations.items():
            if integration.status == IntegrationStatus.ACTIVE:
                health_result = await self.monitor._perform_health_check(integration)
                health_results[integration_id] = health_result.to_dict()
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_checked": len(health_results),
            "results": health_results
        }
    
    async def shutdown(self):
        """Graceful shutdown of integration manager"""
        logger.info("Shutting down Enterprise Integration Manager")
        
        # Stop all monitoring
        for integration_id in list(self.monitor.monitoring_tasks.keys()):
            await self.monitor.stop_monitoring(integration_id)
        
        # Deactivate all active integrations
        active_integrations = [
            i for i in self.integrations.values() 
            if i.status == IntegrationStatus.ACTIVE
        ]
        
        for integration in active_integrations:
            await self.deactivate_integration(integration.integration_id)
        
        # Stop sub-managers
        await self.pipeline_manager.stop_scheduler()
        await self.sync_manager.stop_scheduler()
        
        logger.info("Enterprise Integration Manager shutdown complete")


# Export main classes
__all__ = [
    'IntegrationType',
    'IntegrationStatus',
    'HealthStatus',
    'Integration',
    'HealthCheckResult',
    'Alert',
    'IntegrationMonitor',
    'EnterpriseIntegrationManager'
]