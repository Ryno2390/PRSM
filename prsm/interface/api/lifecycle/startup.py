"""
Application Startup Sequence
============================

Handles all initialization tasks during application startup.
"""

import structlog
from fastapi import FastAPI

from prsm.core.config import get_settings

logger = structlog.get_logger(__name__)


def _get_settings():
    """Get settings, ensuring config is loaded."""
    s = get_settings()
    if s is None:
        from prsm.core.config import load_config
        try:
            s = load_config()
        except Exception:
            pass  # Config loading failed, return None (settings are optional at import time)
    return s


# Module-level reference (may be None at import time, refreshed at startup)
settings = _get_settings()


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
    global settings
    settings = _get_settings()
    logger.info("Starting PRSM API server", environment=getattr(settings, 'environment', 'unknown'))

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

    # Step 9: Initialize observability (metrics collection + tracing)
    await _init_observability()

    # Step 10: Initialize real-time analytics
    await _init_analytics()

    # Step 11: Initialize Web3 event monitoring
    await _init_web3_monitoring()

    logger.info("PRSM API server startup completed successfully")


async def _validate_configuration() -> None:
    """Validate required configuration."""
    if settings is None:
        logger.warning("No configuration loaded - using defaults")
        return
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
        import os

        # === Pre-flight JWT secret enforcement ===
        jwt_secret = settings.jwt_secret or settings.secret_key
        env = os.getenv("PRSM_ENV", "development").lower()

        _WEAK_DEFAULTS = {
            "test-secret-key-at-least-32-characters-long",
            "change-me-to-a-random-string-at-least-32-chars",
        }

        if not jwt_secret:
            raise RuntimeError(
                "FATAL: No JWT secret configured. "
                "Set PRSM_SECRET_KEY in your environment."
            )

        if env == "production":
            if jwt_secret in _WEAK_DEFAULTS or jwt_secret.startswith("change-me") or jwt_secret.startswith("test-"):
                raise RuntimeError(
                    "FATAL: JWT secret is a known-weak placeholder. "
                    "Set PRSM_SECRET_KEY to a random value: openssl rand -hex 32"
                )
            if len(jwt_secret) < 64:
                raise RuntimeError(
                    f"FATAL: JWT secret is {len(jwt_secret)} characters. "
                    "Production requires at least 64 characters. "
                    "Generate: openssl rand -hex 32"
                )

        # === Existing initialization (unchanged) ===
        from prsm.core.security.enhanced_authentication import initialize_auth_system
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


async def _init_observability() -> None:
    """Initialize Prometheus metrics collection and OpenTelemetry tracing."""
    import os

    if os.getenv("PRSM_METRICS_ENABLED", "true").lower() != "true":
        logger.info("Metrics collection disabled via PRSM_METRICS_ENABLED")
        return

    try:
        from prsm.compute.performance.metrics import (
            initialize_metrics, start_metrics_collection, MetricsConfig
        )
        from prsm.core.redis_client import redis_manager

        config = MetricsConfig(
            service_name="prsm-api",
            service_version=getattr(settings, 'version', '0.2.1'),
            environment=getattr(settings, 'environment', 'production'),
            collection_interval=30,
            enable_prometheus=True,
            prometheus_port=None,  # Metrics served through /health/metrics
            store_in_redis=getattr(redis_manager, 'client', None) is not None,
            collect_system_metrics=True,
            collect_process_metrics=True,
            collect_runtime_metrics=True,
        )

        redis_client = getattr(redis_manager, 'client', None)
        initialize_metrics(config, redis_client)
        await start_metrics_collection()
        logger.info("Metrics collection initialized and running")

    except Exception as e:
        logger.warning("Metrics initialization failed — monitoring disabled", error=str(e))

    # === OpenTelemetry Tracing Initialization ===
    # Configure distributed tracing for observability
    # Supports: console (development), jaeger, otlp (production)
    try:
        await _init_tracing()
    except Exception as e:
        logger.warning("Tracing initialization failed — tracing disabled", error=str(e))


async def _init_tracing() -> None:
    """
    Initialize OpenTelemetry tracing with environment-appropriate exporter.

    Configuration via environment variables:
    - OTEL_EXPORTER: Exporter type (console|jaeger|otlp), default: console
    - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint URL
    - JAEGER_HOST: Jaeger agent host (default: localhost)
    - JAEGER_PORT: Jaeger agent port (default: 6831)
    """
    import os

    if os.getenv("OTEL_TRACING_ENABLED", "true").lower() != "true":
        logger.info("OpenTelemetry tracing disabled via OTEL_TRACING_ENABLED")
        return

    try:
        from prsm.compute.performance.tracing import TracingManager

        exporter_type = os.getenv("OTEL_EXPORTER", "console")
        service_name = os.getenv("OTEL_SERVICE_NAME", "prsm-node")

        manager = TracingManager(
            service_name=service_name,
            exporter_type=exporter_type,
            jaeger_host=os.getenv("JAEGER_HOST", "localhost"),
            jaeger_port=int(os.getenv("JAEGER_PORT", "6831")),
            otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", ""),
        )

        await manager.initialize()

        logger.info(
            "OpenTelemetry tracing initialized",
            exporter=exporter_type,
            service_name=service_name
        )

        # Store reference for shutdown
        import prsm.interface.api.lifecycle.startup as startup_module
        startup_module._tracing_manager = manager

    except ImportError:
        logger.info(
            "OpenTelemetry not available — tracing disabled. "
            "Install with: pip install opentelemetry-api opentelemetry-sdk"
        )
    except Exception as e:
        logger.warning("Tracing initialization failed", error=str(e))


async def _init_analytics() -> None:
    """Initialize the real-time analytics stream processor."""
    import os

    if os.getenv("PRSM_ANALYTICS_ENABLED", "true").lower() != "true":
        logger.info("Real-time analytics disabled via PRSM_ANALYTICS_ENABLED")
        return

    try:
        from prsm.data.analytics.real_time_processor import (
            initialize_real_time_processor
        )

        processor = initialize_real_time_processor(
            buffer_size=int(os.getenv("PRSM_ANALYTICS_BUFFER_SIZE", "100000"))
        )
        await processor.start()
        logger.info("Real-time analytics processor started")

    except Exception as e:
        logger.warning(
            "Real-time analytics initialization failed — analytics disabled",
            error=str(e)
        )
        # Non-fatal: never block startup due to analytics


async def _init_web3_monitoring() -> None:
    """
    Initialize blockchain event monitoring for FTNS token.

    Only activates when all three conditions are met:
    1. WEB3_MONITORING_ENABLED=true
    2. FTNS_TOKEN_ADDRESS is set (contract deployed)
    3. web3 library is installed

    Never blocks startup — logs a warning and continues on failure.
    """
    import os

    if os.getenv("WEB3_MONITORING_ENABLED", "true").lower() != "true":
        logger.info("Web3 event monitoring disabled via WEB3_MONITORING_ENABLED")
        return

    if not os.getenv("FTNS_TOKEN_ADDRESS", "").strip():
        logger.info(
            "Web3 event monitoring skipped — FTNS_TOKEN_ADDRESS not configured. "
            "Set this once the FTNS contract is deployed."
        )
        return

    try:
        from prsm.economy.web3.web3_service import initialize_web3_services

        network = os.getenv("WEB3_NETWORK", "polygon_mumbai")
        private_key = os.getenv("WALLET_PRIVATE_KEY", "")

        success = await initialize_web3_services(
            network=network,
            private_key=private_key if private_key else None
        )

        if success:
            logger.info(
                "Web3 event monitoring started",
                extra={"network": network}
            )
        else:
            logger.warning(
                "Web3 event monitoring failed to initialize — "
                "blockchain events will not be tracked. "
                "Check RPC URL and contract addresses."
            )

    except ImportError:
        logger.info(
            "Web3 event monitoring skipped — web3 library not installed. "
            "Install with: pip install 'prsm[blockchain]'"
        )
    except Exception as e:
        logger.warning(
            "Web3 event monitoring initialization failed — monitoring disabled",
            error=str(e)
        )
        # Never block startup due to blockchain connectivity issues
