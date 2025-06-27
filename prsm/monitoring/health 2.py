"""
PRSM Health Monitoring
=====================

Comprehensive health checking system for PRSM components.
Monitors system health, component status, and operational readiness.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthCheck(ABC):
    """Base class for health checks"""
    
    def __init__(self, name: str, description: str, timeout: float = 10.0):
        self.name = name
        self.description = description
        self.timeout = timeout
        self.last_result: Optional[HealthCheckResult] = None
        self.check_count = 0
        self.failure_count = 0
    
    @abstractmethod
    async def check(self) -> HealthCheckResult:
        """Perform the health check"""
        pass
    
    async def run_check(self) -> HealthCheckResult:
        """Run the health check with timeout and error handling"""
        start_time = time.time()
        self.check_count += 1
        
        try:
            result = await asyncio.wait_for(self.check(), timeout=self.timeout)
            response_time = (time.time() - start_time) * 1000
            result.response_time_ms = response_time
            
            if result.status != HealthStatus.HEALTHY:
                self.failure_count += 1
            
            self.last_result = result
            return result
            
        except asyncio.TimeoutError:
            self.failure_count += 1
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check timed out after {self.timeout}s",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000
            )
            self.last_result = result
            return result
            
        except Exception as e:
            self.failure_count += 1
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                details={"error": str(e), "error_type": type(e).__name__}
            )
            self.last_result = result
            return result
    
    def get_failure_rate(self) -> float:
        """Get the failure rate as a percentage"""
        if self.check_count == 0:
            return 0.0
        return (self.failure_count / self.check_count) * 100


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connectivity"""
    
    def __init__(self, database_url: str, query: str = "SELECT 1"):
        super().__init__("database", "Database connectivity check")
        self.database_url = database_url
        self.query = query
    
    async def check(self) -> HealthCheckResult:
        """Check database connectivity"""
        try:
            # This would use actual database connection
            # For demo purposes, we'll simulate the check
            await asyncio.sleep(0.1)  # Simulate connection time
            
            # Simulate successful connection
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                timestamp=datetime.now(),
                response_time_ms=0,  # Will be set by run_check
                details={
                    "query": self.query,
                    "database_type": "postgresql",
                    "connection_pool_size": 10
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Database connection failed: {str(e)}",
                timestamp=datetime.now(),
                response_time_ms=0,
                details={"error": str(e)}
            )


class RedisHealthCheck(HealthCheck):
    """Health check for Redis connectivity"""
    
    def __init__(self, redis_url: str):
        super().__init__("redis", "Redis connectivity check")
        self.redis_url = redis_url
    
    async def check(self) -> HealthCheckResult:
        """Check Redis connectivity"""
        try:
            # Simulate Redis ping
            await asyncio.sleep(0.05)
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Redis connection successful",
                timestamp=datetime.now(),
                response_time_ms=0,
                details={
                    "redis_version": "7.0.0",
                    "memory_usage": "45MB",
                    "connected_clients": 12
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Redis connection failed: {str(e)}",
                timestamp=datetime.now(),
                response_time_ms=0,
                details={"error": str(e)}
            )


class PRSMCoreHealthCheck(HealthCheck):
    """Health check for PRSM core components"""
    
    def __init__(self, prsm_client):
        super().__init__("prsm_core", "PRSM core components health check")
        self.prsm_client = prsm_client
    
    async def check(self) -> HealthCheckResult:
        """Check PRSM core health"""
        try:
            # Check if PRSM client is available and responsive
            if hasattr(self.prsm_client, 'ping'):
                healthy = await self.prsm_client.ping()
                if healthy:
                    return HealthCheckResult(
                        name=self.name,
                        status=HealthStatus.HEALTHY,
                        message="PRSM core is responsive",
                        timestamp=datetime.now(),
                        response_time_ms=0,
                        details={
                            "nwtn_status": "active",
                            "teachers_available": 3,
                            "agents_active": 5
                        }
                    )
                else:
                    return HealthCheckResult(
                        name=self.name,
                        status=HealthStatus.WARNING,
                        message="PRSM core is not fully responsive",
                        timestamp=datetime.now(),
                        response_time_ms=0
                    )
            else:
                # Simulate health check
                await asyncio.sleep(0.1)
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="PRSM core simulation - healthy",
                    timestamp=datetime.now(),
                    response_time_ms=0
                )
                
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"PRSM core health check failed: {str(e)}",
                timestamp=datetime.now(),
                response_time_ms=0,
                details={"error": str(e)}
            )


class SystemResourcesHealthCheck(HealthCheck):
    """Health check for system resources (CPU, memory, disk)"""
    
    def __init__(self, cpu_threshold: float = 80.0, memory_threshold: float = 85.0, 
                 disk_threshold: float = 90.0):
        super().__init__("system_resources", "System resources health check")
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
    
    async def check(self) -> HealthCheckResult:
        """Check system resources"""
        try:
            # Simulate system resource checks
            # In a real implementation, this would use psutil or similar
            import random
            
            cpu_usage = random.uniform(10, 60)  # Simulate CPU usage
            memory_usage = random.uniform(30, 70)  # Simulate memory usage
            disk_usage = random.uniform(40, 80)  # Simulate disk usage
            
            status = HealthStatus.HEALTHY
            messages = []
            
            if cpu_usage > self.cpu_threshold:
                status = HealthStatus.WARNING
                messages.append(f"High CPU usage: {cpu_usage:.1f}%")
            
            if memory_usage > self.memory_threshold:
                status = HealthStatus.WARNING
                messages.append(f"High memory usage: {memory_usage:.1f}%")
            
            if disk_usage > self.disk_threshold:
                status = HealthStatus.CRITICAL
                messages.append(f"High disk usage: {disk_usage:.1f}%")
            
            if not messages:
                message = "System resources are within normal limits"
            else:
                message = "; ".join(messages)
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                response_time_ms=0,
                details={
                    "cpu_usage_percent": cpu_usage,
                    "memory_usage_percent": memory_usage,
                    "disk_usage_percent": disk_usage,
                    "thresholds": {
                        "cpu": self.cpu_threshold,
                        "memory": self.memory_threshold,
                        "disk": self.disk_threshold
                    }
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {str(e)}",
                timestamp=datetime.now(),
                response_time_ms=0,
                details={"error": str(e)}
            )


class ExternalAPIHealthCheck(HealthCheck):
    """Health check for external API dependencies"""
    
    def __init__(self, api_name: str, api_url: str, timeout: float = 5.0):
        super().__init__(f"external_api_{api_name}", f"External API {api_name} health check", timeout)
        self.api_name = api_name
        self.api_url = api_url
    
    async def check(self) -> HealthCheckResult:
        """Check external API health"""
        try:
            # Simulate external API check
            await asyncio.sleep(0.2)  # Simulate network delay
            
            # Simulate successful API response
            import random
            if random.random() > 0.1:  # 90% success rate
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message=f"External API {self.api_name} is responsive",
                    timestamp=datetime.now(),
                    response_time_ms=0,
                    details={
                        "api_url": self.api_url,
                        "status_code": 200,
                        "response_size": "1.2KB"
                    }
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.WARNING,
                    message=f"External API {self.api_name} returned error",
                    timestamp=datetime.now(),
                    response_time_ms=0,
                    details={
                        "api_url": self.api_url,
                        "status_code": 503,
                        "error": "Service temporarily unavailable"
                    }
                )
                
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"External API {self.api_name} check failed: {str(e)}",
                timestamp=datetime.now(),
                response_time_ms=0,
                details={"error": str(e)}
            )


class HealthChecker:
    """Main health checking system"""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.health_checks: Dict[str, HealthCheck] = {}
        self.is_monitoring = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self.health_history: List[Dict[str, HealthCheckResult]] = []
        self.max_history = 1000
    
    def add_health_check(self, health_check: HealthCheck) -> None:
        """Add a health check"""
        self.health_checks[health_check.name] = health_check
        logger.info(f"Added health check: {health_check.name}")
    
    def remove_health_check(self, name: str) -> None:
        """Remove a health check"""
        if name in self.health_checks:
            del self.health_checks[name]
            logger.info(f"Removed health check: {name}")
    
    def get_health_check(self, name: str) -> Optional[HealthCheck]:
        """Get a health check by name"""
        return self.health_checks.get(name)
    
    def list_health_checks(self) -> List[str]:
        """List all health check names"""
        return list(self.health_checks.keys())
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks"""
        results = {}
        
        # Run all checks concurrently
        check_tasks = {
            name: check.run_check() 
            for name, check in self.health_checks.items()
        }
        
        completed_checks = await asyncio.gather(
            *check_tasks.values(), 
            return_exceptions=True
        )
        
        # Collect results
        for (name, _), result in zip(check_tasks.items(), completed_checks):
            if isinstance(result, Exception):
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check execution failed: {str(result)}",
                    timestamp=datetime.now(),
                    response_time_ms=0,
                    details={"error": str(result)}
                )
            else:
                results[name] = result
        
        # Store in history
        self.health_history.append(results)
        if len(self.health_history) > self.max_history:
            self.health_history.pop(0)
        
        return results
    
    async def run_single_check(self, name: str) -> Optional[HealthCheckResult]:
        """Run a single health check"""
        health_check = self.health_checks.get(name)
        if not health_check:
            return None
        
        return await health_check.run_check()
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status"""
        if not self.health_checks:
            return HealthStatus.UNKNOWN
        
        statuses = [check.last_result.status for check in self.health_checks.values() 
                   if check.last_result]
        
        if not statuses:
            return HealthStatus.UNKNOWN
        
        # Determine overall status
        if any(status == HealthStatus.CRITICAL for status in statuses):
            return HealthStatus.CRITICAL
        elif any(status == HealthStatus.WARNING for status in statuses):
            return HealthStatus.WARNING
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary"""
        overall_status = self.get_overall_health()
        
        check_summary = {}
        for name, check in self.health_checks.items():
            if check.last_result:
                check_summary[name] = {
                    "status": check.last_result.status.value,
                    "message": check.last_result.message,
                    "last_check": check.last_result.timestamp.isoformat(),
                    "response_time_ms": check.last_result.response_time_ms,
                    "failure_rate": check.get_failure_rate()
                }
            else:
                check_summary[name] = {
                    "status": "not_run",
                    "message": "Health check has not been run yet",
                    "failure_rate": 0.0
                }
        
        return {
            "overall_status": overall_status.value,
            "total_checks": len(self.health_checks),
            "monitoring_active": self.is_monitoring,
            "check_interval": self.check_interval,
            "checks": check_summary,
            "last_update": datetime.now().isoformat()
        }
    
    async def start_monitoring(self) -> None:
        """Start continuous health monitoring"""
        if self.is_monitoring:
            logger.warning("Health monitoring already started")
            return
        
        self.is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"Started health monitoring with {self.check_interval}s interval")
    
    async def stop_monitoring(self) -> None:
        """Stop continuous health monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped health monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                await self.run_all_checks()
                logger.debug(f"Completed health check cycle")
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
            
            try:
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
    
    def get_health_history(self, minutes: int = 60) -> List[Dict[str, HealthCheckResult]]:
        """Get health check history for the specified time period"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        filtered_history = []
        for check_results in self.health_history:
            # Check if any result in this check cycle is within the time window
            if any(result.timestamp >= cutoff_time for result in check_results.values()):
                filtered_history.append(check_results)
        
        return filtered_history
    
    def setup_default_checks(self, prsm_client=None, database_url: str = None, 
                            redis_url: str = None) -> None:
        """Setup default health checks"""
        logger.info("Setting up default health checks")
        
        # System resources check
        self.add_health_check(SystemResourcesHealthCheck())
        
        # PRSM core check
        if prsm_client:
            self.add_health_check(PRSMCoreHealthCheck(prsm_client))
        
        # Database check
        if database_url:
            self.add_health_check(DatabaseHealthCheck(database_url))
        
        # Redis check  
        if redis_url:
            self.add_health_check(RedisHealthCheck(redis_url))
        
        # Example external API checks
        self.add_health_check(ExternalAPIHealthCheck("openai", "https://api.openai.com/v1/models"))
        
        logger.info(f"Setup {len(self.health_checks)} default health checks")
