"""
Application Startup Sequence
============================

Handles all initialization tasks during application startup.
"""

import structlog
from fastapi import FastAPI

from prsm.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


async def startup_sequence(app: FastAPI) -> None:
    """
    Execute application startup sequence.

    Order of initialization:
    1. Validate configuration
    2. Initialize database connections
    3. Initialize Redis caching
    4. Initialize vector databases
    5. Initialize IPFS
    6. Initialize security systems
    7. Initialize authentication
    8. Initialize secure credentials

    Args:
        app: FastAPI application instance
    """
    logger.info("Starting PRSM API server", environment=settings.environment)

    # Step 1: Validate configuration
    await _validate_configuration()

    # Step 2: Initialize database
    await _init_database()

    # Step 3: Initialize Redis
    await _init_redis()

    # Step 4: Initialize vector databases
    await _init_vector_databases()

    # Step 5: Initialize IPFS
    await _init_ipfs()

    # Step 6: Initialize secure credentials
    await _init_secure_credentials()

    # Step 7: Initialize security systems
    await _init_security_systems()

    # Step 8: Initialize authentication
    await _init_authentication()

    logger.info("PRSM API server startup completed successfully")


async def _validate_configuration() -> None:
    """Validate required configuration."""
    missing_config = settings.validate_required_config()

    if missing_config:
        logger.warning("Missing configuration detected", missing_items=missing_config)

        if settings.is_production and missing_config:
            raise RuntimeError(
                f"Missing required production configuration: {', '.join(missing_config)}"
            )
        else:
            logger.info("Continuing startup - some features may be limited")
    else:
        logger.info("Configuration validation passed")


async def _init_database() -> None:
    """Initialize database connections."""
    try:
        from prsm.core.database import init_database
        await init_database()
        logger.info("Database connections established")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        if settings.is_production:
            raise


async def _init_redis() -> None:
    """Initialize Redis caching and session management."""
    try:
        from prsm.core.redis_client import init_redis
        await init_redis()
        logger.info("Redis caching initialized")
    except Exception as e:
        logger.warning("Redis initialization failed - caching disabled", error=str(e))


async def _init_vector_databases() -> None:
    """Initialize vector databases for semantic search."""
    try:
        from prsm.core.vector_db import init_vector_databases
        await init_vector_databases()
        logger.info("Vector databases initialized")
    except Exception as e:
        logger.warning("Vector database initialization failed", error=str(e))


async def _init_ipfs() -> None:
    """Initialize IPFS distributed storage."""
    try:
        from prsm.core.ipfs_client import init_ipfs
        await init_ipfs()
        logger.info("IPFS distributed storage initialized")
    except Exception as e:
        logger.warning("IPFS initialization failed", error=str(e))


async def _init_secure_credentials() -> None:
    """Initialize secure credential management."""
    try:
        from prsm.core.integrations.security.secure_config_manager import initialize_secure_configuration
        success = await initialize_secure_configuration()
        if success:
            logger.info("Secure credential management initialized")
        else:
            logger.warning("Secure credential management initialization incomplete")
    except Exception as e:
        logger.warning("Secure credential initialization failed", error=str(e))


async def _init_security_systems() -> None:
    """Initialize security monitoring and rate limiting."""
    try:
        from prsm.core.redis_client import redis_manager

        # Initialize enhanced authentication
        from prsm.core.security.enhanced_authentication import initialize_auth_system
        jwt_secret = settings.jwt_secret or settings.secret_key
        await initialize_auth_system(redis_manager.client, jwt_secret)
        logger.info("Enhanced authentication system initialized")

        # Initialize rate limiting
        from prsm.core.security.advanced_rate_limiting import initialize_rate_limiter
        await initialize_rate_limiter(redis_manager.client)
        logger.info("Rate limiting system initialized")

        # Initialize security monitoring
        from prsm.core.security.security_monitoring import initialize_security_monitor
        await initialize_security_monitor(redis_manager.client)
        logger.info("Security monitoring initialized")

        # Initialize security analytics
        from prsm.core.security.security_analytics import initialize_security_analytics
        await initialize_security_analytics(redis_manager.client)
        logger.info("Security analytics initialized")

    except Exception as e:
        logger.warning("Some security systems failed to initialize", error=str(e))
        if settings.is_production:
            raise


async def _init_authentication() -> None:
    """Initialize authentication manager."""
    try:
        from prsm.core.auth.auth_manager import auth_manager
        await auth_manager.initialize()
        logger.info("Authentication system initialized")
    except Exception as e:
        logger.error("Failed to initialize auth system", error=str(e))
