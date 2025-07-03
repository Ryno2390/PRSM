from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import psutil
# import aioredis  # Temporarily commented out for Python 3.13 compatibility
import asyncpg
import httpx
import time
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

from ..core.database import get_db_session
from ..core.redis_client import get_redis_client
from ..auth import get_current_user
from ..core.config import settings

router = APIRouter(prefix="/health", tags=["health"])

# =============================================================================
# Health Check Models
# =============================================================================

class ComponentHealth(BaseModel):
    """Health status for individual component"""
    status: str  # "healthy", "degraded", "unhealthy"
    response_time_ms: Optional[float] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    last_check: datetime

class SystemHealth(BaseModel):
    """Overall system health status"""
    status: str  # "healthy", "degraded", "unhealthy"
    version: str
    uptime_seconds: float
    environment: str
    timestamp: datetime
    components: Dict[str, ComponentHealth]
    metrics: Optional[Dict[str, Any]] = None

class DetailedStatus(BaseModel):
    """Detailed system status information"""
    system: SystemHealth
    resources: Dict[str, Any]
    dependencies: Dict[str, ComponentHealth]
    performance: Dict[str, Any]
    security: Dict[str, Any]

# =============================================================================
# Health Check Functions
# =============================================================================

class HealthChecker:
    """Comprehensive health checking system"""
    
    def __init__(self):
        self.start_time = time.time()
        self.version = "1.0.0"  # This should come from package version
        
    async def check_database(self) -> ComponentHealth:
        """Check PostgreSQL database health"""
        start_time = time.time()
        try:
            # Test database connection
            conn = await asyncpg.connect(
                host=settings.DB_HOST,
                port=settings.DB_PORT,
                user=settings.DB_USER,
                password=settings.DB_PASSWORD,
                database=settings.DB_NAME,
                timeout=5.0
            )
            
            # Test basic query
            result = await conn.fetchval("SELECT 1")
            await conn.close()
            
            response_time = (time.time() - start_time) * 1000
            
            # Get additional database metrics
            details = {
                "query_result": result,
                "connection_successful": True
            }
            
            return ComponentHealth(
                status="healthy",
                response_time_ms=response_time,
                details=details,
                last_check=datetime.utcnow()
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ComponentHealth(
                status="unhealthy",
                response_time_ms=response_time,
                error=str(e),
                last_check=datetime.utcnow()
            )
    
    async def check_redis(self) -> ComponentHealth:
        """Check Redis cache health"""
        start_time = time.time()
        try:
            # Test Redis connection
            redis = aioredis.from_url(
                f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
                password=settings.REDIS_PASSWORD,
                socket_timeout=5.0
            )
            
            # Test basic operations
            test_key = "health_check_test"
            await redis.set(test_key, "test_value", ex=60)
            result = await redis.get(test_key)
            await redis.delete(test_key)
            await redis.close()
            
            response_time = (time.time() - start_time) * 1000
            
            details = {
                "test_operation_successful": True,
                "test_value_retrieved": result.decode() == "test_value" if result else False
            }
            
            return ComponentHealth(
                status="healthy",
                response_time_ms=response_time,
                details=details,
                last_check=datetime.utcnow()
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ComponentHealth(
                status="unhealthy",
                response_time_ms=response_time,
                error=str(e),
                last_check=datetime.utcnow()
            )
    
    async def check_ipfs(self) -> ComponentHealth:
        """Check IPFS node health"""
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://{settings.IPFS_HOST}:{settings.IPFS_API_PORT}/api/v0/id")
                response.raise_for_status()
                
                data = response.json()
                response_time = (time.time() - start_time) * 1000
                
                details = {
                    "node_id": data.get("ID"),
                    "agent_version": data.get("AgentVersion"),
                    "protocol_version": data.get("ProtocolVersion")
                }
                
                return ComponentHealth(
                    status="healthy",
                    response_time_ms=response_time,
                    details=details,
                    last_check=datetime.utcnow()
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ComponentHealth(
                status="unhealthy",
                response_time_ms=response_time,
                error=str(e),
                last_check=datetime.utcnow()
            )
    
    async def check_external_apis(self) -> ComponentHealth:
        """Check external API dependencies"""
        start_time = time.time()
        external_services = {
            "openai": "https://api.openai.com/v1/models",
            "anthropic": "https://api.anthropic.com/v1/messages",
            "huggingface": "https://api-inference.huggingface.co/models"
        }
        
        healthy_services = 0
        total_services = len(external_services)
        service_results = {}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for service_name, url in external_services.items():
                try:
                    # Just check if the endpoint is reachable (don't need valid API calls)
                    response = await client.head(url)
                    if response.status_code < 500:  # Accept 4xx as "reachable"
                        healthy_services += 1
                        service_results[service_name] = "reachable"
                    else:
                        service_results[service_name] = f"error_{response.status_code}"
                except Exception as e:
                    service_results[service_name] = f"error: {str(e)[:50]}"
        
        response_time = (time.time() - start_time) * 1000
        health_ratio = healthy_services / total_services
        
        if health_ratio >= 0.8:
            status = "healthy"
        elif health_ratio >= 0.5:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return ComponentHealth(
            status=status,
            response_time_ms=response_time,
            details={
                "healthy_services": healthy_services,
                "total_services": total_services,
                "health_ratio": health_ratio,
                "service_results": service_results
            },
            last_check=datetime.utcnow()
        )
    
    def get_system_resources(self) -> Dict[str, Any]:
        """Get system resource information"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count(),
                    "count_logical": psutil.cpu_count(logical=True)
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                    "free": memory.free
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                },
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        uptime = time.time() - self.start_time
        
        return {
            "uptime_seconds": uptime,
            "uptime_human": self._format_uptime(uptime),
            "process_id": psutil.Process().pid,
            "threads": psutil.Process().num_threads(),
            "memory_rss": psutil.Process().memory_info().rss,
            "memory_vms": psutil.Process().memory_info().vms,
            "cpu_times": psutil.Process().cpu_times()._asdict()
        }
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human readable format"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m {secs}s"
        elif hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    async def get_full_health_check(self) -> SystemHealth:
        """Perform comprehensive health check"""
        # Run all health checks concurrently
        db_check, redis_check, ipfs_check, external_check = await asyncio.gather(
            self.check_database(),
            self.check_redis(),
            self.check_ipfs(),
            self.check_external_apis(),
            return_exceptions=True
        )
        
        components = {
            "database": db_check if isinstance(db_check, ComponentHealth) else ComponentHealth(
                status="unhealthy", error=str(db_check), last_check=datetime.utcnow()
            ),
            "redis": redis_check if isinstance(redis_check, ComponentHealth) else ComponentHealth(
                status="unhealthy", error=str(redis_check), last_check=datetime.utcnow()
            ),
            "ipfs": ipfs_check if isinstance(ipfs_check, ComponentHealth) else ComponentHealth(
                status="unhealthy", error=str(ipfs_check), last_check=datetime.utcnow()
            ),
            "external_apis": external_check if isinstance(external_check, ComponentHealth) else ComponentHealth(
                status="unhealthy", error=str(external_check), last_check=datetime.utcnow()
            )
        }
        
        # Determine overall system health
        unhealthy_count = sum(1 for comp in components.values() if comp.status == "unhealthy")
        degraded_count = sum(1 for comp in components.values() if comp.status == "degraded")
        
        if unhealthy_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return SystemHealth(
            status=overall_status,
            version=self.version,
            uptime_seconds=time.time() - self.start_time,
            environment=settings.ENVIRONMENT,
            timestamp=datetime.utcnow(),
            components=components
        )

# Global health checker instance
health_checker = HealthChecker()

# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/", response_model=SystemHealth)
async def health_check():
    """Basic health check endpoint"""
    return await health_checker.get_full_health_check()

@router.get("/liveness")
async def liveness_probe():
    """Kubernetes liveness probe - simple check"""
    return {"status": "alive", "timestamp": datetime.utcnow()}

@router.get("/readiness")
async def readiness_probe():
    """Kubernetes readiness probe - check if ready to serve traffic"""
    health = await health_checker.get_full_health_check()
    
    if health.status == "unhealthy":
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "status": "ready",
        "health_status": health.status,
        "timestamp": datetime.utcnow()
    }

@router.get("/detailed", response_model=DetailedStatus)
async def detailed_status(current_user = Depends(get_current_user)):
    """Detailed system status - requires authentication"""
    health = await health_checker.get_full_health_check()
    resources = health_checker.get_system_resources()
    performance = health_checker.get_performance_metrics()
    
    # Security information (basic)
    security_info = {
        "authenticated_user": current_user.username if current_user else None,
        "auth_required": True,
        "security_headers_enabled": True,
        "rate_limiting_enabled": True
    }
    
    return DetailedStatus(
        system=health,
        resources=resources,
        dependencies=health.components,
        performance=performance,
        security=security_info
    )

@router.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    metrics = generate_latest()
    return JSONResponse(
        content=metrics.decode('utf-8'),
        media_type=CONTENT_TYPE_LATEST
    )

@router.get("/version")
async def version_info():
    """Get version and build information"""
    return {
        "version": health_checker.version,
        "environment": settings.ENVIRONMENT,
        "build_time": "2025-06-11T00:00:00Z",  # This should be set during build
        "git_commit": "latest",  # This should be set during build
        "python_version": psutil.Process().exe,
        "uptime_seconds": time.time() - health_checker.start_time
    }

@router.get("/ping")
async def ping():
    """Simple ping endpoint"""
    return {"message": "pong", "timestamp": datetime.utcnow()}

@router.get("/status/components")
async def component_status():
    """Get status of individual components"""
    health = await health_checker.get_full_health_check()
    return {
        "components": {
            name: {
                "status": comp.status,
                "response_time_ms": comp.response_time_ms,
                "last_check": comp.last_check,
                "healthy": comp.status == "healthy"
            }
            for name, comp in health.components.items()
        },
        "overall_status": health.status,
        "healthy_components": sum(1 for comp in health.components.values() if comp.status == "healthy"),
        "total_components": len(health.components)
    }

@router.get("/status/resources")
async def resource_status():
    """Get system resource status"""
    resources = health_checker.get_system_resources()
    
    # Add resource health assessment
    health_assessment = {
        "cpu_health": "healthy" if resources.get("cpu", {}).get("percent", 0) < 80 else "degraded",
        "memory_health": "healthy" if resources.get("memory", {}).get("percent", 0) < 85 else "degraded",
        "disk_health": "healthy" if resources.get("disk", {}).get("percent", 0) < 90 else "degraded"
    }
    
    return {
        "resources": resources,
        "health_assessment": health_assessment,
        "timestamp": datetime.utcnow()
    }