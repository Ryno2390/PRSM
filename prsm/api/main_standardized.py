"""
PRSM FastAPI Application - Standardized Version
Demonstrates API standardization improvements and best practices
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from prsm.core.config import get_settings

# Import standardization modules
from .standards import APIConfig, HealthCheckResponse
from .exceptions import register_exception_handlers
from .middleware import setup_middleware
from .dependencies import add_request_metadata


logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with standardized startup/shutdown"""
    
    # Startup
    logger.info("🚀 Starting PRSM API server", environment=settings.environment)
    
    try:
        # Initialize all subsystems with standardized error handling
        from prsm.core.database import init_database
        from prsm.core.redis_client import init_redis
        from prsm.core.vector_db import init_vector_databases
        from prsm.core.ipfs_client import init_ipfs
        
        await init_database()
        logger.info("✅ Database connections established")
        
        await init_redis()
        logger.info("✅ Redis caching initialized")
        
        await init_vector_databases()
        logger.info("✅ Vector databases initialized")
        
        await init_ipfs()
        logger.info("✅ IPFS distributed storage initialized")
        
        logger.info("🎉 PRSM API server startup completed successfully")
        
    except Exception as e:
        logger.error("❌ Failed to start PRSM API server", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down PRSM API server")
    
    try:
        from prsm.core.database import close_database
        from prsm.core.redis_client import close_redis
        from prsm.core.vector_db import close_vector_databases
        from prsm.core.ipfs_client import close_ipfs
        
        await close_database()
        await close_redis()
        await close_vector_databases()
        await close_ipfs()
        
        logger.info("✅ PRSM API server shutdown completed")
        
    except Exception as e:
        logger.error("❌ Error during shutdown", error=str(e))


# Create FastAPI application with standardized configuration
app = FastAPI(
    title="PRSM API",
    description="Protocol for Recursive Scientific Modeling - Standardized API",
    version="1.0.0",
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    openapi_tags=APIConfig.OPENAPI_TAGS,
    lifespan=lifespan
)

# Setup standardized middleware (order matters!)
setup_middleware(app)

# Register standardized exception handlers
register_exception_handlers(app)

# Add standard request metadata dependency to all routes
app.dependency_overrides[add_request_metadata] = add_request_metadata


# === Standardized Endpoints ===

@app.get(
    "/",
    response_model=dict,
    summary="API Root",
    description="Get API information and feature flags"
)
async def root():
    """Root endpoint with standardized response format"""
    return {
        "name": "PRSM API",
        "version": "1.0.0",
        "description": "Protocol for Recursive Scientific Modeling",
        "environment": settings.environment.value,
        "api_version": APIConfig.API_VERSION,
        "status": "operational",
        "features": {
            "nwtn_enabled": settings.nwtn_enabled,
            "ftns_enabled": settings.ftns_enabled,
            "p2p_enabled": settings.p2p_enabled,
            "governance_enabled": settings.governance_enabled,
        },
        "documentation": {
            "openapi": "/openapi.json",
            "docs": "/docs" if not settings.is_production else None,
            "redoc": "/redoc" if not settings.is_production else None
        }
    }


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Comprehensive system health check"
)
async def health_check():
    """Standardized health check endpoint"""
    from datetime import datetime
    
    health_status = HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        checks={}
    )
    
    # Database health check
    try:
        from prsm.core.database import db_manager
        db_healthy = await db_manager.health_check()
        health_status.checks["database"] = {
            "status": "healthy" if db_healthy else "unhealthy",
            "response_time_ms": 5 if db_healthy else None
        }
        if not db_healthy:
            health_status.status = "degraded"
    except Exception as e:
        health_status.checks["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status.status = "unhealthy"
    
    # Redis health check
    try:
        from prsm.core.redis_client import redis_manager
        redis_healthy = await redis_manager.health_check()
        health_status.checks["redis"] = {
            "status": "healthy" if redis_healthy else "unhealthy",
            "response_time_ms": 2 if redis_healthy else None
        }
        if not redis_healthy:
            health_status.status = "degraded"
    except Exception as e:
        health_status.checks["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status.status = "unhealthy"
    
    # IPFS health check
    try:
        from prsm.core.ipfs_client import get_ipfs_client
        ipfs_client = get_ipfs_client()
        ipfs_healthy = await ipfs_client.health_check() > 0
        health_status.checks["ipfs"] = {
            "status": "healthy" if ipfs_healthy else "unhealthy",
            "response_time_ms": 150 if ipfs_healthy else None
        }
        if not ipfs_healthy:
            health_status.status = "degraded"
    except Exception as e:
        health_status.checks["ipfs"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status.status = "unhealthy"
    
    return health_status


# === Include Standardized Routers ===

def include_standardized_routers():
    """Include all API routers with standardized configuration"""
    
    # Authentication (no prefix for backward compatibility)
    try:
        from prsm.api.auth_api import router as auth_router
        from .standards import get_router_config
        
        app.include_router(
            auth_router,
            **get_router_config("auth", "User authentication and authorization")
        )
        logger.info("✅ Authentication API endpoints registered")
    except ImportError as e:
        logger.warning(f"⚠️ Authentication API not available: {e}")
    
    # Models
    try:
        from prsm.api.models_api import router as models_router
        app.include_router(
            models_router,
            **get_router_config("models", "AI model operations and management")
        )
        logger.info("✅ Models API endpoints registered")
    except ImportError as e:
        logger.warning(f"⚠️ Models API not available: {e}")
    
    # Marketplace
    try:
        from prsm.api.real_marketplace_api import router as marketplace_router
        app.include_router(
            marketplace_router,
            **get_router_config("marketplace", "FTNS token marketplace operations")
        )
        logger.info("✅ Marketplace API endpoints registered")
    except ImportError as e:
        logger.warning(f"⚠️ Marketplace API not available: {e}")
    
    # Budget Management
    try:
        from prsm.api.budget_api import router as budget_router
        app.include_router(
            budget_router,
            **get_router_config("budget", "Budget management and cost tracking")
        )
        logger.info("✅ Budget API endpoints registered")
    except ImportError as e:
        logger.warning(f"⚠️ Budget API not available: {e}")
    
    # Tasks
    try:
        from prsm.api.task_api import router as task_router
        app.include_router(
            task_router,
            **get_router_config("tasks", "Task management and execution tracking")
        )
        logger.info("✅ Tasks API endpoints registered")
    except ImportError as e:
        logger.warning(f"⚠️ Tasks API not available: {e}")
    
    # Teams
    try:
        from prsm.api.teams_api import router as teams_router
        app.include_router(
            teams_router,
            **get_router_config("teams", "Team collaboration and project management")
        )
        logger.info("✅ Teams API endpoints registered")
    except ImportError as e:
        logger.warning(f"⚠️ Teams API not available: {e}")
    
    # Analytics
    try:
        from prsm.api.analytics_api import router as analytics_router
        app.include_router(
            analytics_router,
            **get_router_config("analytics", "Usage analytics and performance metrics")
        )
        logger.info("✅ Analytics API endpoints registered")
    except ImportError as e:
        logger.warning(f"⚠️ Analytics API not available: {e}")
    
    # Safety
    try:
        from prsm.api.safety_api import router as safety_router
        app.include_router(
            safety_router,
            **get_router_config("safety", "Safety monitoring and circuit breaker systems")
        )
        logger.info("✅ Safety API endpoints registered")
    except ImportError as e:
        logger.warning(f"⚠️ Safety API not available: {e}")
    
    # Governance
    try:
        from prsm.api.governance_api import router as governance_router
        app.include_router(
            governance_router,
            **get_router_config("governance", "Platform governance and voting mechanisms")
        )
        logger.info("✅ Governance API endpoints registered")
    except ImportError as e:
        logger.warning(f"⚠️ Governance API not available: {e}")


# Include all routers with standardized configuration
include_standardized_routers()


# === Development Server ===

if __name__ == "__main__":
    import uvicorn
    import os
    
    host = os.getenv("PRSM_API_HOST", "127.0.0.1")
    port = int(os.getenv("PRSM_API_PORT", "8000"))
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=not settings.is_production,
        log_level="info"
    )