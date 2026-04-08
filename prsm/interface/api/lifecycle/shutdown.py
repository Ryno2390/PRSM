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

    # Step 6: Stop metrics collection
    await _shutdown_observability()

    # Step 7: Stop analytics processor
    await _shutdown_analytics()

    # Step 8: Stop Web3 event monitoring
    await _shutdown_web3()

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
    """Close content storage."""
    try:
        from prsm.storage import close_content_store
        close_content_store()
        logger.info("Content storage closed")
    except Exception as e:
        logger.error("Error closing content storage", error=str(e))


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


async def _shutdown_observability() -> None:
    """Stop metrics collection gracefully."""
    try:
        from prsm.compute.performance.metrics import stop_metrics_collection
        await stop_metrics_collection()
        logger.info("Metrics collection stopped")
    except Exception as e:
        logger.error("Error stopping metrics collection", error=str(e))


async def _shutdown_analytics() -> None:
    """Stop the real-time analytics stream processor."""
    try:
        from prsm.data.analytics.real_time_processor import get_real_time_processor
        processor = get_real_time_processor()
        await processor.stop()
        logger.info("Real-time analytics processor stopped")
    except RuntimeError:
        pass  # Not initialized — nothing to stop
    except Exception as e:
        logger.error("Error stopping analytics processor", error=str(e))


async def _shutdown_web3() -> None:
    """Stop Web3 event monitoring and clean up blockchain connections."""
    try:
        from prsm.economy.web3.web3_service import cleanup_web3_services
        await cleanup_web3_services()
        logger.info("Web3 event monitoring stopped")
    except ImportError:
        pass  # web3 not installed — nothing to stop
    except Exception as e:
        logger.error("Error stopping Web3 event monitoring", error=str(e))
