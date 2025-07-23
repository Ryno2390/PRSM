"""
PRSM Observability Integration
Complete integration of monitoring, tracing, metrics, logging, alerting, and dashboard systems
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import asyncio
import logging
import redis.asyncio as aioredis
from contextlib import asynccontextmanager

# Import all observability components
from .tracing import (
    initialize_tracing, get_tracer, TracingConfig, TraceLevel,
    start_async_span, trace_function
)
from .metrics import (
    initialize_metrics, get_metrics_collector, start_metrics_collection,
    MetricsConfig, MetricDefinition, MetricType, increment_counter,
    set_gauge, observe_histogram, time_operation
)
from .structured_logging import (
    initialize_logging, get_logger, LoggingConfig, LogLevel,
    correlation_context, with_correlation, log_errors
)
from .monitoring_dashboard import (
    initialize_dashboard, get_dashboard, start_dashboard_monitoring,
    MonitoringDashboard, create_dashboard_app
)
from .alerting import (
    initialize_alerting, get_alerting_system, start_alerting,
    AlertingConfig, AlertRule, AlertSeverity, NotificationChannel,
    create_cpu_alert_rule, create_memory_alert_rule, create_disk_alert_rule,
    create_error_rate_alert_rule
)

logger = logging.getLogger(__name__)


@dataclass
class ObservabilityConfig:
    """Complete observability system configuration"""
    service_name: str = "prsm-api"
    service_version: str = "1.0.0"
    environment: str = "production"
    
    # Redis configuration
    redis_url: str = "redis://localhost:6379"
    
    # Component configurations
    tracing_config: TracingConfig = None
    metrics_config: MetricsConfig = None
    logging_config: LoggingConfig = None
    alerting_config: AlertingConfig = None
    
    # Integration settings
    enable_tracing: bool = True
    enable_metrics: bool = True
    enable_logging: bool = True
    enable_dashboard: bool = True
    enable_alerting: bool = True
    
    # Dashboard settings
    dashboard_port: int = 8080
    enable_dashboard_api: bool = True
    
    def __post_init__(self):
        """Initialize component configurations if not provided"""
        if self.tracing_config is None:
            self.tracing_config = TracingConfig(
                service_name=self.service_name,
                service_version=self.service_version,
                environment=self.environment,
                sampling_rate=1.0,
                trace_level=TraceLevel.NORMAL,
                store_spans_in_redis=True,
                auto_instrument_redis=True,
                auto_instrument_database=True,
                auto_instrument_fastapi=True
            )
        
        if self.metrics_config is None:
            self.metrics_config = MetricsConfig(
                service_name=self.service_name,
                service_version=self.service_version,
                environment=self.environment,
                collection_interval=30,
                enable_prometheus=True,
                prometheus_port=9090,
                store_in_redis=True,
                collect_system_metrics=True,
                collect_process_metrics=True,
                collect_runtime_metrics=True
            )
        
        if self.logging_config is None:
            self.logging_config = LoggingConfig(
                service_name=self.service_name,
                service_version=self.service_version,
                environment=self.environment,
                log_level=LogLevel.INFO,
                enable_correlation_ids=True,
                enable_trace_integration=True,
                store_in_redis=True,
                filter_sensitive_data=True
            )
        
        if self.alerting_config is None:
            self.alerting_config = AlertingConfig(
                service_name=self.service_name,
                store_alerts_in_redis=True,
                default_notification_channels=[
                    NotificationChannel.CONSOLE,
                    NotificationChannel.EMAIL
                ]
            )


class ObservabilitySystem:
    """Integrated observability system manager"""
    
    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self.redis_client: Optional[aioredis.Redis] = None
        self.dashboard_app = None
        self.running = False
        
        # Component references
        self.tracer = None
        self.metrics_collector = None
        self.structured_logger = None
        self.dashboard = None
        self.alerting_system = None
        
        # Statistics
        self.stats = {
            "system_started_at": None,
            "components_initialized": 0,
            "components_started": 0,
            "total_errors": 0
        }
    
    async def initialize(self):
        """Initialize all observability components"""
        try:
            logger.info("🚀 Initializing PRSM Observability System...")
            
            # Initialize Redis connection
            await self._initialize_redis()
            
            # Initialize components based on configuration
            if self.config.enable_tracing:
                await self._initialize_tracing()
            
            if self.config.enable_metrics:
                await self._initialize_metrics()
            
            if self.config.enable_logging:
                await self._initialize_logging()
            
            if self.config.enable_dashboard:
                await self._initialize_dashboard()
            
            if self.config.enable_alerting:
                await self._initialize_alerting()
            
            # Set up default alert rules
            await self._setup_default_alerts()
            
            # Set up component integrations
            await self._setup_integrations()
            
            logger.info(f"✅ Observability system initialized with {self.stats['components_initialized']} components")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize observability system: {e}")
            self.stats["total_errors"] += 1
            raise
    
    async def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = aioredis.from_url(self.config.redis_url)
            # Test connection
            await self.redis_client.ping()
            logger.info("✅ Redis connection established")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
    
    async def _initialize_tracing(self):
        """Initialize distributed tracing"""
        try:
            initialize_tracing(self.config.tracing_config, self.redis_client)
            self.tracer = get_tracer()
            self.stats["components_initialized"] += 1
            logger.info("✅ Distributed tracing initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize tracing: {e}")
            self.stats["total_errors"] += 1
    
    async def _initialize_metrics(self):
        """Initialize metrics collection"""
        try:
            initialize_metrics(self.config.metrics_config, self.redis_client)
            self.metrics_collector = get_metrics_collector()
            self.stats["components_initialized"] += 1
            logger.info("✅ Metrics collection initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize metrics: {e}")
            self.stats["total_errors"] += 1
    
    async def _initialize_logging(self):
        """Initialize structured logging"""
        try:
            initialize_logging(self.config.logging_config, self.redis_client)
            self.structured_logger = get_logger()
            self.stats["components_initialized"] += 1
            logger.info("✅ Structured logging initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize logging: {e}")
            self.stats["total_errors"] += 1
    
    async def _initialize_dashboard(self):
        """Initialize monitoring dashboard"""
        try:
            self.dashboard = initialize_dashboard(self.redis_client)
            
            if self.config.enable_dashboard_api:
                from .monitoring_dashboard import create_dashboard_app
                self.dashboard_app = create_dashboard_app(self.redis_client)
            
            self.stats["components_initialized"] += 1
            logger.info("✅ Monitoring dashboard initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize dashboard: {e}")
            self.stats["total_errors"] += 1
    
    async def _initialize_alerting(self):
        """Initialize alerting system"""
        try:
            initialize_alerting(self.config.alerting_config, self.redis_client)
            self.alerting_system = get_alerting_system()
            self.stats["components_initialized"] += 1
            logger.info("✅ Alerting system initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize alerting: {e}")
            self.stats["total_errors"] += 1
    
    async def _setup_default_alerts(self):
        """Set up default alert rules"""
        try:
            if not self.alerting_system:
                return
            
            # Create default alert rules
            default_rules = [
                create_cpu_alert_rule(threshold=80.0, severity=AlertSeverity.WARNING),
                create_memory_alert_rule(threshold=85.0, severity=AlertSeverity.WARNING),
                create_disk_alert_rule(threshold=90.0, severity=AlertSeverity.ERROR),
                create_error_rate_alert_rule(threshold=5.0, severity=AlertSeverity.ERROR)
            ]
            
            for rule in default_rules:
                await self.alerting_system.create_rule(rule)
            
            logger.info(f"✅ Created {len(default_rules)} default alert rules")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup default alerts: {e}")
            self.stats["total_errors"] += 1
    
    async def _setup_integrations(self):
        """Set up integrations between components"""
        try:
            # Register custom metrics for observability system itself
            if self.metrics_collector:
                observability_metrics = [
                    MetricDefinition(
                        name="observability_components_active",
                        metric_type=MetricType.GAUGE,
                        description="Number of active observability components"
                    ),
                    MetricDefinition(
                        name="observability_errors_total",
                        metric_type=MetricType.COUNTER,
                        description="Total observability system errors"
                    ),
                    MetricDefinition(
                        name="observability_alerts_active",
                        metric_type=MetricType.GAUGE,
                        description="Number of active alerts"
                    )
                ]
                
                for metric_def in observability_metrics:
                    self.metrics_collector.register_metric(metric_def)
            
            logger.info("✅ Component integrations configured")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup integrations: {e}")
            self.stats["total_errors"] += 1
    
    async def start(self):
        """Start all observability components"""
        if self.running:
            logger.warning("Observability system is already running")
            return
        
        try:
            logger.info("🚀 Starting PRSM Observability System...")
            
            # Start metrics collection
            if self.metrics_collector:
                await start_metrics_collection()
                self.stats["components_started"] += 1
            
            # Start dashboard monitoring
            if self.dashboard:
                await start_dashboard_monitoring()
                self.stats["components_started"] += 1
            
            # Start alerting
            if self.alerting_system:
                await start_alerting()
                self.stats["components_started"] += 1
            
            self.running = True
            self.stats["system_started_at"] = asyncio.get_event_loop().time()
            
            # Update system metrics
            await self._update_system_metrics()
            
            logger.info(f"✅ Observability system started with {self.stats['components_started']} active components")
            
        except Exception as e:
            logger.error(f"❌ Failed to start observability system: {e}")
            self.stats["total_errors"] += 1
            raise
    
    async def stop(self):
        """Stop all observability components"""
        if not self.running:
            return
        
        try:
            logger.info("🛑 Stopping PRSM Observability System...")
            
            # Stop alerting
            if self.alerting_system:
                await self.alerting_system.stop()
            
            # Stop dashboard monitoring
            if self.dashboard:
                from .monitoring_dashboard import stop_dashboard_monitoring
                await stop_dashboard_monitoring()
            
            # Stop metrics collection
            if self.metrics_collector:
                from .metrics import stop_metrics_collection
                await stop_metrics_collection()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            self.running = False
            logger.info("✅ Observability system stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping observability system: {e}")
            self.stats["total_errors"] += 1
    
    async def _update_system_metrics(self):
        """Update observability system metrics"""
        try:
            if not self.metrics_collector:
                return
            
            # Count active components
            active_components = sum([
                1 if self.tracer else 0,
                1 if self.metrics_collector else 0,
                1 if self.structured_logger else 0,
                1 if self.dashboard else 0,
                1 if self.alerting_system else 0
            ])
            
            # Update metrics
            set_gauge("observability_components_active", active_components)
            set_gauge("observability_errors_total", self.stats["total_errors"])
            
            if self.alerting_system:
                summary = await self.alerting_system.get_alert_summary()
                set_gauge("observability_alerts_active", summary.get("active_alerts", 0))
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    @asynccontextmanager
    async def observability_context(self, operation_name: str, **tags):
        """Context manager that provides full observability for an operation"""
        correlation_id = None
        
        try:
            # Start structured logging context
            if self.structured_logger:
                async with correlation_context() as log_context:
                    correlation_id = log_context.correlation_id
                    
                    # Start distributed tracing
                    if self.tracer:
                        async with start_async_span(
                            operation_name,
                            tags={**tags, "correlation_id": correlation_id}
                        ) as span:
                            
                            # Time the operation
                            if self.metrics_collector:
                                async with time_operation("operation_duration_seconds", 
                                                         operation=operation_name, **tags):
                                    yield {
                                        "correlation_id": correlation_id,
                                        "span": span,
                                        "logger": self.structured_logger
                                    }
                            else:
                                yield {
                                    "correlation_id": correlation_id,
                                    "span": span,
                                    "logger": self.structured_logger
                                }
                    else:
                        # No tracing, just logging and metrics
                        if self.metrics_collector:
                            async with time_operation("operation_duration_seconds",
                                                     operation=operation_name, **tags):
                                yield {
                                    "correlation_id": correlation_id,
                                    "logger": self.structured_logger
                                }
                        else:
                            yield {
                                "correlation_id": correlation_id,
                                "logger": self.structured_logger
                            }
            else:
                # Minimal context without structured logging
                yield {
                    "correlation_id": "no-logging",
                    "logger": None
                }
                
        except Exception as e:
            # Record error in all systems
            if self.structured_logger:
                await self.structured_logger.error(
                    f"Error in {operation_name}",
                    exception=e,
                    operation=operation_name,
                    **tags
                )
            
            if self.metrics_collector:
                increment_counter("operation_errors_total", 
                                operation=operation_name, **tags)
            
            self.stats["total_errors"] += 1
            raise
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information"""
        try:
            health = {
                "status": "healthy" if self.running else "stopped",
                "service_name": self.config.service_name,
                "service_version": self.config.service_version,
                "environment": self.config.environment,
                "uptime_seconds": None,
                "components": {},
                "statistics": self.stats.copy()
            }
            
            if self.running and self.stats["system_started_at"]:
                health["uptime_seconds"] = int(asyncio.get_event_loop().time() - self.stats["system_started_at"])
            
            # Component health
            if self.tracer:
                health["components"]["tracing"] = await self.tracer.get_system_stats()
            
            if self.metrics_collector:
                health["components"]["metrics"] = await self.metrics_collector.get_system_overview()
            
            if self.structured_logger:
                health["components"]["logging"] = self.structured_logger.get_stats()
            
            if self.dashboard:
                health["components"]["dashboard"] = await self.dashboard.get_dashboard_stats()
            
            if self.alerting_system:
                health["components"]["alerting"] = await self.alerting_system.get_alert_summary()
            
            return health
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                "status": "error",
                "error": str(e),
                "statistics": self.stats.copy()
            }
    
    def get_dashboard_app(self):
        """Get the dashboard FastAPI app for integration"""
        return self.dashboard_app


# Global observability system instance
observability_system: Optional[ObservabilitySystem] = None


async def initialize_observability(config: ObservabilityConfig) -> ObservabilitySystem:
    """Initialize the complete observability system"""
    global observability_system
    
    observability_system = ObservabilitySystem(config)
    await observability_system.initialize()
    
    logger.info("✅ PRSM Observability System fully initialized")
    return observability_system


def get_observability_system() -> ObservabilitySystem:
    """Get the global observability system instance"""
    if observability_system is None:
        raise RuntimeError("Observability system not initialized")
    return observability_system


async def start_observability():
    """Start the observability system"""
    if observability_system:
        await observability_system.start()


async def stop_observability():
    """Stop the observability system"""
    if observability_system:
        await observability_system.stop()


@asynccontextmanager
async def observe_operation(operation_name: str, **tags):
    """Context manager for full observability of an operation"""
    if observability_system:
        async with observability_system.observability_context(operation_name, **tags) as context:
            yield context
    else:
        yield {"correlation_id": "no-observability", "logger": None}


# Decorators for easy integration

def observe_function(operation_name: Optional[str] = None, **tags):
    """Decorator to add full observability to a function"""
    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"
        
        @trace_function(operation_name, tags=tags)
        @with_correlation()
        @log_errors()
        async def async_wrapper(*args, **kwargs):
            async with observe_operation(operation_name, **tags):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we can't use async context managers
            # but we can still apply tracing and logging decorators
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Configuration helpers

def create_development_config(service_name: str = "prsm-api") -> ObservabilityConfig:
    """Create observability configuration for development environment"""
    return ObservabilityConfig(
        service_name=service_name,
        environment="development",
        tracing_config=TracingConfig(
            service_name=service_name,
            environment="development",
            sampling_rate=1.0,
            trace_level=TraceLevel.DEBUG,
            export_to_console=True
        ),
        metrics_config=MetricsConfig(
            service_name=service_name,
            environment="development",
            collection_interval=15,
            enable_prometheus=True
        ),
        logging_config=LoggingConfig(
            service_name=service_name,
            environment="development",
            log_level=LogLevel.DEBUG,
            console_output=True
        ),
        dashboard_port=8080
    )


def create_production_config(service_name: str = "prsm-api") -> ObservabilityConfig:
    """Create observability configuration for production environment"""
    return ObservabilityConfig(
        service_name=service_name,
        environment="production",
        tracing_config=TracingConfig(
            service_name=service_name,
            environment="production",
            sampling_rate=0.1,  # 10% sampling in production
            trace_level=TraceLevel.NORMAL,
            export_to_jaeger=True,
            jaeger_endpoint="http://jaeger:14268"
        ),
        metrics_config=MetricsConfig(
            service_name=service_name,
            environment="production",
            collection_interval=30,
            enable_prometheus=True,
            prometheus_pushgateway_url="http://pushgateway:9091"
        ),
        logging_config=LoggingConfig(
            service_name=service_name,
            environment="production",
            log_level=LogLevel.INFO,
            console_output=False,
            file_output=True,
            file_path="/var/log/prsm/app.log"
        ),
        alerting_config=AlertingConfig(
            service_name=service_name,
            notification_rate_limit=50,
            default_notification_channels=[
                NotificationChannel.EMAIL,
                NotificationChannel.SLACK,
                NotificationChannel.PAGERDUTY
            ]
        ),
        dashboard_port=8080
    )