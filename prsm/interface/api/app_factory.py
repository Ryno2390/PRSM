"""
FastAPI Application Factory
===========================

Creates and configures the FastAPI application instance using the factory pattern.
This allows for better testability and configuration management.

Usage:
    from prsm.interface.api.app_factory import create_app

    app = create_app()  # Uses default configuration
    app = create_app(environment="testing")  # Test configuration
"""

from typing import Callable, Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from prsm.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def default_lifespan(app: FastAPI):
    """
    Default application lifespan context manager.

    Handles startup and shutdown sequences for the PRSM API.
    """
    from prsm.interface.api.lifecycle import startup_sequence, shutdown_sequence

    try:
        await startup_sequence(app)
        yield
    finally:
        await shutdown_sequence(app)


def create_app(
    lifespan: Optional[Callable] = None,
    title: str = "PRSM API",
    version: str = "0.1.0",
    include_middleware: bool = True,
    include_routers: bool = True,
    include_core_endpoints: bool = True
) -> FastAPI:
    """
    Create and configure FastAPI application.

    This factory function creates a fully configured FastAPI application
    with all middleware, routers, and settings applied.

    Args:
        lifespan: Async context manager for startup/shutdown. If None, uses default_lifespan.
        title: API title
        version: API version
        include_middleware: Whether to add middleware stack
        include_routers: Whether to include API routers
        include_core_endpoints: Whether to include core endpoints (root, health, query, etc.)

    Returns:
        Configured FastAPI application
    """
    # Import here to avoid circular imports
    from prsm.interface.api.openapi_config import custom_openapi_schema, API_TAGS_METADATA

    # Use default lifespan if none provided
    effective_lifespan = lifespan if lifespan is not None else default_lifespan

    app = FastAPI(
        title=title,
        description=_get_api_description(),
        version=version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=effective_lifespan,
        openapi_tags=API_TAGS_METADATA,
        contact={
            "name": "PRSM API Support",
            "email": "api-support@prsm.org",
            "url": "https://developers.prsm.org"
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT"
        },
        servers=_get_servers()
    )

    # Apply custom OpenAPI schema
    app.openapi = lambda: custom_openapi_schema(app)

    # Add middleware if requested
    if include_middleware:
        _configure_middleware(app)

    # Include routers if requested
    if include_routers:
        _include_routers(app)

    # Include core endpoints if requested
    if include_core_endpoints:
        _register_core_endpoints(app)

    # Add exception handlers
    _configure_exception_handlers(app)

    logger.info(
        "FastAPI application created",
        title=title,
        version=version,
        environment=settings.environment.value
    )

    return app


def _get_api_description() -> str:
    """Get API description based on environment."""
    base_description = """
# Protocol for Recursive Scientific Modeling (PRSM)

API for decentralized AI collaboration and knowledge synthesis.

## Features

- **NWTN Orchestration**: Neuro-symbolic reasoning engine
- **FTNS Token Economy**: Fungible tokens for computational resources
- **Distributed Storage**: IPFS-based content addressing
- **Governance**: Decentralized decision making
- **Safety Systems**: Circuit breakers and monitoring

## Authentication

Most endpoints require JWT authentication. Obtain tokens via `/api/v1/auth/login`.

## Rate Limiting

API calls are rate-limited based on your tier:
- Anonymous: 10 req/min
- Free: 60 req/min
- Pro: 300 req/min
- Enterprise: 1000 req/min
"""
    if settings.is_development:
        base_description += "\n\n**Development Mode Active**"

    return base_description


def _get_servers() -> List[Dict[str, str]]:
    """Get server list based on environment."""
    servers = []

    if settings.is_production:
        servers.extend([
            {"url": "https://api.prsm.org", "description": "Production server"},
            {"url": "https://staging-api.prsm.org", "description": "Staging server"},
        ])

    # Always include localhost for development
    servers.append(
        {"url": "http://localhost:8000", "description": "Development server"}
    )

    return servers


def _configure_middleware(app: FastAPI) -> None:
    """Configure middleware stack for the application."""
    from prsm.interface.api.middleware import configure_middleware_stack
    configure_middleware_stack(app)


def _include_routers(app: FastAPI) -> None:
    """Include all API routers."""
    from prsm.interface.api.router_registry import include_all_routers
    include_all_routers(app)


def _register_core_endpoints(app: FastAPI) -> None:
    """Register core endpoints (root, health, query, etc.)."""
    from prsm.interface.api.core_endpoints import register_core_endpoints
    register_core_endpoints(app)


def _configure_exception_handlers(app: FastAPI) -> None:
    """Configure global exception handlers."""
    from fastapi.responses import JSONResponse
    from fastapi import Request

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler for unhandled errors."""
        logger.error(
            "Unhandled exception",
            exc_info=exc,
            path=request.url.path,
            method=request.method
        )

        # Don't expose internal errors in production
        if settings.is_production:
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error",
                    "error": str(exc),
                    "type": type(exc).__name__
                }
            )
