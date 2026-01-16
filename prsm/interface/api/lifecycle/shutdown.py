"""
Application Shutdown Sequence
=============================

Handles graceful shutdown and cleanup.
"""

import structlog
from fastapi import FastAPI

logger = structlog.get_logger(__name__)


async def shutdown_sequence(app: FastAPI) -> None:
    """
    Execute application shutdown sequence.

    Order of cleanup:
    1. Close database connections
    2. Close Redis connections
    3. Close vector database connections
    4. Close IPFS connections
    5. Shutdown security systems

    Args:
        app: FastAPI application instance
    """
    logger.info("Shutting down PRSM API server")

    # Step 1: Close database
    await _close_database()

    # Step 2: Close Redis
    await _close_redis()

    # Step 3: Close vector databases
    await _close_vector_databases()

    # Step 4: Close IPFS
    await _close_ipfs()

    # Step 5: Shutdown security systems
    await _shutdown_security_systems()

    logger.info("PRSM API server shutdown completed")


async def _close_database() -> None:
    """Close database connections."""
    try:
        from prsm.core.database import close_database
        await close_database()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error("Error closing database", error=str(e))


async def _close_redis() -> None:
    """Close Redis connections."""
    try:
        from prsm.core.redis_client import close_redis
        await close_redis()
        logger.info("Redis connections closed")
    except Exception as e:
        logger.error("Error closing Redis", error=str(e))


async def _close_vector_databases() -> None:
    """Close vector database connections."""
    try:
        from prsm.core.vector_db import close_vector_databases
        await close_vector_databases()
        logger.info("Vector database connections closed")
    except Exception as e:
        logger.error("Error closing vector databases", error=str(e))


async def _close_ipfs() -> None:
    """Close IPFS connections."""
    try:
        from prsm.core.ipfs_client import close_ipfs
        await close_ipfs()
        logger.info("IPFS connections closed")
    except Exception as e:
        logger.error("Error closing IPFS", error=str(e))


async def _shutdown_security_systems() -> None:
    """Shutdown security systems."""
    try:
        from prsm.core.security.security_monitoring import shutdown_security_monitor
        from prsm.core.security.security_analytics import shutdown_security_analytics

        await shutdown_security_monitor()
        await shutdown_security_analytics()
        logger.info("Security systems shutdown completed")
    except Exception as e:
        logger.error("Error shutting down security systems", error=str(e))
