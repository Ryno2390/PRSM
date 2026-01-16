"""
Router Registry
================

Centralized router management for PRSM API.
All routers are registered here for clean separation of concerns.
"""

import structlog
from fastapi import FastAPI

logger = structlog.get_logger(__name__)


def include_all_routers(app: FastAPI) -> None:
    """
    Include all API routers in the application.

    This function centralizes router registration for:
    - Better testability (can skip routers in tests)
    - Clear visibility of all API endpoints
    - Consistent prefix and tag management

    Args:
        app: FastAPI application instance
    """
    # === Core API Routers ===
    _include_core_routers(app)

    # === Marketplace Routers ===
    _include_marketplace_routers(app)

    # === Advanced Feature Routers ===
    _include_advanced_routers(app)

    # === Specialized Service Routers ===
    _include_service_routers(app)

    # === Integration Layer ===
    _include_integration_routers(app)

    # === Documentation & Migration ===
    _include_docs_and_migration(app)

    logger.info("All API routers registered successfully")


def _include_core_routers(app: FastAPI) -> None:
    """Include core API routers."""
    try:
        from prsm.interface.api.auth_api import router as auth_router
        app.include_router(auth_router)  # Has /api/v1/auth prefix internally
        logger.debug("Auth router registered")
    except ImportError as e:
        logger.warning(f"Auth router not available: {e}")

    try:
        from prsm.interface.api.teams_api import router as teams_router
        app.include_router(teams_router, tags=["Teams"])
        logger.debug("Teams router registered")
    except ImportError as e:
        logger.warning(f"Teams router not available: {e}")

    try:
        from prsm.interface.api.credential_api import router as credential_router
        app.include_router(credential_router, tags=["Credentials"])
        logger.debug("Credential router registered")
    except ImportError as e:
        logger.warning(f"Credential router not available: {e}")

    try:
        from prsm.interface.api.security_status_api import router as security_router
        app.include_router(security_router, tags=["Security"])
        logger.debug("Security status router registered")
    except ImportError as e:
        logger.warning(f"Security status router not available: {e}")

    try:
        from prsm.interface.api.security_logging_api import router as security_logging_router
        app.include_router(security_logging_router, tags=["Security Logging"])
        logger.debug("Security logging router registered")
    except ImportError as e:
        logger.warning(f"Security logging router not available: {e}")

    try:
        from prsm.interface.api.payment_api import router as payment_router
        app.include_router(payment_router, tags=["Payments"])
        logger.debug("Payment router registered")
    except ImportError as e:
        logger.warning(f"Payment router not available: {e}")

    try:
        from prsm.interface.api.cryptography_api import router as crypto_router
        app.include_router(crypto_router, tags=["Cryptography"])
        logger.debug("Cryptography router registered")
    except ImportError as e:
        logger.warning(f"Cryptography router not available: {e}")

    try:
        from prsm.interface.api.governance_api import router as governance_router
        app.include_router(governance_router, tags=["Governance"])
        logger.debug("Governance router registered")
    except ImportError as e:
        logger.warning(f"Governance router not available: {e}")

    try:
        from prsm.interface.api.budget_api import router as budget_router
        app.include_router(budget_router, tags=["Budget Management"])
        logger.debug("Budget router registered")
    except ImportError as e:
        logger.warning(f"Budget router not available: {e}")

    logger.info("Core API routers registered")


def _include_marketplace_routers(app: FastAPI) -> None:
    """Include marketplace-related routers."""
    try:
        from prsm.interface.api.real_marketplace_api import router as marketplace_router
        app.include_router(marketplace_router, prefix="/api/v1/marketplace", tags=["Marketplace"])
        logger.debug("Marketplace router registered")
    except ImportError as e:
        logger.warning(f"Marketplace router not available: {e}")

    try:
        from prsm.interface.api.recommendation_api import router as recommendation_router
        app.include_router(recommendation_router, prefix="/api/v1/marketplace", tags=["Recommendations"])
        logger.debug("Recommendation router registered")
    except ImportError as e:
        logger.warning(f"Recommendation router not available: {e}")

    try:
        from prsm.interface.api.reputation_api import router as reputation_router
        app.include_router(reputation_router, prefix="/api/v1", tags=["Reputation"])
        logger.debug("Reputation router registered")
    except ImportError as e:
        logger.warning(f"Reputation router not available: {e}")

    logger.info("Marketplace routers registered")


def _include_advanced_routers(app: FastAPI) -> None:
    """Include advanced feature routers."""
    try:
        from prsm.interface.api.distillation_api import router as distillation_router
        app.include_router(distillation_router, prefix="/api/v1", tags=["Distillation"])
        logger.debug("Distillation router registered")
    except ImportError as e:
        logger.warning(f"Distillation router not available: {e}")

    try:
        from prsm.interface.api.monitoring_api import router as monitoring_router
        app.include_router(monitoring_router, prefix="/api/v1", tags=["Monitoring"])
        logger.debug("Monitoring router registered")
    except ImportError as e:
        logger.warning(f"Monitoring router not available: {e}")

    try:
        from prsm.interface.api.compliance_api import router as compliance_router
        app.include_router(compliance_router, prefix="/api/v1", tags=["Compliance"])
        logger.debug("Compliance router registered")
    except ImportError as e:
        logger.warning(f"Compliance router not available: {e}")

    try:
        from prsm.interface.api.contributor_api import router as contributor_router
        app.include_router(contributor_router, prefix="/api/v1/contributors", tags=["Contributors"])
        logger.debug("Contributor router registered")
    except ImportError as e:
        logger.warning(f"Contributor router not available: {e}")

    try:
        from prsm.interface.api.mainnet_deployment_api import router as mainnet_router
        app.include_router(mainnet_router, prefix="/api/v1", tags=["Mainnet Deployment"])
        logger.debug("Mainnet deployment router registered")
    except ImportError as e:
        logger.warning(f"Mainnet deployment router not available: {e}")

    try:
        from prsm.economy.web3.frontend_integration import router as web3_router
        app.include_router(web3_router, prefix="/api/v1", tags=["Web3"])
        logger.debug("Web3 router registered")
    except ImportError as e:
        logger.warning(f"Web3 router not available: {e}")

    try:
        from prsm.compute.chronos.api import router as chronos_router
        app.include_router(chronos_router, prefix="/api/v1", tags=["CHRONOS"])
        logger.debug("CHRONOS router registered")
    except ImportError as e:
        logger.warning(f"CHRONOS router not available: {e}")

    logger.info("Advanced feature routers registered")


def _include_service_routers(app: FastAPI) -> None:
    """Include specialized service routers."""
    try:
        from prsm.interface.api.health_api import router as health_router
        app.include_router(health_router, prefix="/api/v1", tags=["Health"])
        logger.debug("Health router registered")
    except ImportError as e:
        logger.warning(f"Health router not available: {e}")

    try:
        from prsm.interface.api.session_api import router as session_router
        app.include_router(session_router, prefix="/api/v1", tags=["Sessions"])
        logger.debug("Session router registered")
    except ImportError as e:
        logger.warning(f"Session router not available: {e}")

    try:
        from prsm.interface.api.task_api import router as task_router
        app.include_router(task_router, prefix="/api/v1", tags=["Tasks"])
        logger.debug("Task router registered")
    except ImportError as e:
        logger.warning(f"Task router not available: {e}")

    try:
        from prsm.interface.api.ui_api import router as ui_router
        app.include_router(ui_router, prefix="/ui", tags=["UI"])
        logger.debug("UI router registered")
    except ImportError as e:
        logger.warning(f"UI router not available: {e}")

    try:
        from prsm.interface.api.ipfs_api import router as ipfs_router
        app.include_router(ipfs_router, prefix="/ipfs", tags=["IPFS"])
        logger.debug("IPFS router registered")
    except ImportError as e:
        logger.warning(f"IPFS router not available: {e}")

    logger.info("Service routers registered")


def _include_integration_routers(app: FastAPI) -> None:
    """Include integration layer routers."""
    try:
        from prsm.core.integrations.api.integration_api import integration_router
        from prsm.core.integrations.api.config_api import config_router
        from prsm.core.integrations.api.security_api import security_router as int_security_router

        app.include_router(integration_router, prefix="/integrations", tags=["Integrations"])
        app.include_router(config_router, prefix="/integrations/config", tags=["Configuration"])
        app.include_router(int_security_router, prefix="/integrations/security", tags=["Security"])
        logger.info("Integration layer routers registered")
    except ImportError as e:
        logger.warning(f"Integration layer not available: {e}")


def _include_docs_and_migration(app: FastAPI) -> None:
    """Include documentation and migration endpoints."""
    try:
        from prsm.interface.api.docs_ui import create_enhanced_docs_ui
        create_enhanced_docs_ui(app)
        logger.debug("Enhanced docs UI registered")
    except ImportError as e:
        logger.warning(f"Enhanced docs UI not available: {e}")

    try:
        from prsm.interface.api.version_schemas import create_version_specific_docs_endpoints
        create_version_specific_docs_endpoints(app)
        logger.debug("Version-specific docs endpoints registered")
    except ImportError as e:
        logger.warning(f"Version-specific docs not available: {e}")

    try:
        from prsm.interface.api.migration import create_migration_endpoints
        create_migration_endpoints(app)
        logger.debug("Migration endpoints registered")
    except ImportError as e:
        logger.warning(f"Migration endpoints not available: {e}")

    try:
        from prsm.core.security.security_analytics import create_security_analytics_endpoints
        create_security_analytics_endpoints(app)
        logger.debug("Security analytics endpoints registered")
    except ImportError as e:
        logger.warning(f"Security analytics endpoints not available: {e}")

    logger.info("Documentation and migration endpoints registered")


def get_router_summary() -> dict:
    """
    Get a summary of all registered routers.

    Returns:
        Dictionary with router categories and their endpoints
    """
    return {
        "core": [
            "/api/v1/auth",
            "/api/v1/teams",
            "/api/v1/credentials",
            "/api/v1/security",
            "/api/v1/security/logging",
            "/api/v1/payments",
            "/api/v1/crypto",
            "/api/v1/governance",
            "/api/v1/budget"
        ],
        "marketplace": [
            "/api/v1/marketplace",
            "/api/v1/marketplace/recommendations",
            "/api/v1/reputation"
        ],
        "advanced": [
            "/api/v1/distillation",
            "/api/v1/monitoring",
            "/api/v1/compliance",
            "/api/v1/contributors",
            "/api/v1/mainnet",
            "/api/v1/web3",
            "/api/v1/chronos"
        ],
        "services": [
            "/api/v1/health",
            "/api/v1/sessions",
            "/api/v1/tasks",
            "/ui",
            "/ipfs"
        ],
        "integrations": [
            "/integrations",
            "/integrations/config",
            "/integrations/security"
        ]
    }
