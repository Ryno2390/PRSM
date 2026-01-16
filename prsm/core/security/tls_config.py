"""
TLS and Production Security Configuration
==========================================

Production-grade TLS configuration for PRSM including:
- HTTPS/TLS server configuration
- Database connection TLS
- Redis connection TLS
- Certificate management utilities

Addresses audit finding: "TLS/Certificate Management - Not implemented"

Usage:
    from prsm.core.security.tls_config import (
        get_tls_config,
        get_database_ssl_config,
        get_redis_ssl_config,
        TLSMode
    )
"""

import os
import ssl
import logging
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


class TLSMode(str, Enum):
    """TLS operation modes"""
    DISABLED = "disabled"      # No TLS (development only)
    OPPORTUNISTIC = "prefer"   # Use TLS if available
    REQUIRED = "require"       # Require TLS, skip verification
    VERIFY_CA = "verify-ca"    # Require TLS, verify CA
    VERIFY_FULL = "verify-full"  # Require TLS, verify CA and hostname


class TLSVersion(str, Enum):
    """Minimum TLS version requirements"""
    TLS_1_2 = "TLSv1.2"
    TLS_1_3 = "TLSv1.3"


@dataclass
class TLSConfig:
    """
    TLS configuration settings for production deployment.

    Security Requirements (from audit):
    - TLS 1.2+ required in production
    - Strong cipher suites only
    - Certificate verification enabled
    - HSTS header configured
    """

    # TLS Mode
    mode: TLSMode = TLSMode.VERIFY_FULL

    # Minimum TLS version (1.2+ required for production)
    min_version: TLSVersion = TLSVersion.TLS_1_2

    # Certificate paths
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    ca_file: Optional[str] = None

    # Certificate verification
    verify_certificates: bool = True
    check_hostname: bool = True

    # Cipher configuration (strong ciphers only)
    ciphers: str = field(default_factory=lambda: ":".join([
        "ECDHE+AESGCM",
        "ECDHE+CHACHA20",
        "DHE+AESGCM",
        "DHE+CHACHA20",
        "!aNULL",
        "!MD5",
        "!DSS",
        "!RC4",
        "!3DES"
    ]))

    # HSTS configuration
    hsts_enabled: bool = True
    hsts_max_age: int = 31536000  # 1 year
    hsts_include_subdomains: bool = True
    hsts_preload: bool = False

    def validate(self) -> bool:
        """Validate TLS configuration for production readiness."""
        errors = []

        if self.mode in [TLSMode.VERIFY_CA, TLSMode.VERIFY_FULL]:
            if not self.ca_file or not Path(self.ca_file).exists():
                errors.append("CA certificate file required for certificate verification")

        if self.mode == TLSMode.VERIFY_FULL and not self.check_hostname:
            errors.append("Hostname verification required for verify-full mode")

        if self.min_version == TLSVersion.TLS_1_2:
            logger.info("TLS 1.2 minimum version configured")
        elif self.min_version == TLSVersion.TLS_1_3:
            logger.info("TLS 1.3 minimum version configured (recommended)")

        if errors:
            for error in errors:
                logger.error(f"TLS configuration error: {error}")
            return False

        return True


def get_tls_config() -> TLSConfig:
    """
    Get TLS configuration from environment variables.

    Environment Variables:
        PRSM_TLS_MODE: TLS mode (disabled, prefer, require, verify-ca, verify-full)
        PRSM_TLS_MIN_VERSION: Minimum TLS version (TLSv1.2, TLSv1.3)
        PRSM_TLS_CERT_FILE: Path to certificate file
        PRSM_TLS_KEY_FILE: Path to private key file
        PRSM_TLS_CA_FILE: Path to CA certificate file
        PRSM_TLS_VERIFY: Enable certificate verification (true/false)
        PRSM_TLS_CHECK_HOSTNAME: Enable hostname verification (true/false)
    """
    mode_str = os.environ.get("PRSM_TLS_MODE", "verify-full")
    try:
        mode = TLSMode(mode_str)
    except ValueError:
        logger.warning(f"Invalid TLS mode '{mode_str}', defaulting to verify-full")
        mode = TLSMode.VERIFY_FULL

    min_version_str = os.environ.get("PRSM_TLS_MIN_VERSION", "TLSv1.2")
    try:
        min_version = TLSVersion(min_version_str)
    except ValueError:
        logger.warning(f"Invalid TLS version '{min_version_str}', defaulting to TLSv1.2")
        min_version = TLSVersion.TLS_1_2

    return TLSConfig(
        mode=mode,
        min_version=min_version,
        cert_file=os.environ.get("PRSM_TLS_CERT_FILE"),
        key_file=os.environ.get("PRSM_TLS_KEY_FILE"),
        ca_file=os.environ.get("PRSM_TLS_CA_FILE"),
        verify_certificates=os.environ.get("PRSM_TLS_VERIFY", "true").lower() == "true",
        check_hostname=os.environ.get("PRSM_TLS_CHECK_HOSTNAME", "true").lower() == "true",
        hsts_enabled=os.environ.get("PRSM_HSTS_ENABLED", "true").lower() == "true",
        hsts_max_age=int(os.environ.get("PRSM_HSTS_MAX_AGE", "31536000")),
        hsts_include_subdomains=os.environ.get("PRSM_HSTS_SUBDOMAINS", "true").lower() == "true",
        hsts_preload=os.environ.get("PRSM_HSTS_PRELOAD", "false").lower() == "true"
    )


def create_ssl_context(config: TLSConfig) -> Optional[ssl.SSLContext]:
    """
    Create SSL context for server/client connections.

    Args:
        config: TLS configuration

    Returns:
        Configured SSLContext or None if TLS disabled
    """
    if config.mode == TLSMode.DISABLED:
        logger.warning("TLS disabled - NOT RECOMMENDED FOR PRODUCTION")
        return None

    # Create context with appropriate protocol
    if config.min_version == TLSVersion.TLS_1_3:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = ssl.TLSVersion.TLSv1_3
    else:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = ssl.TLSVersion.TLSv1_2

    # Set cipher suites
    context.set_ciphers(config.ciphers)

    # Configure certificate verification
    if config.mode in [TLSMode.VERIFY_CA, TLSMode.VERIFY_FULL]:
        context.verify_mode = ssl.CERT_REQUIRED
        if config.ca_file:
            context.load_verify_locations(config.ca_file)
        else:
            context.load_default_certs()
    elif config.mode == TLSMode.REQUIRED:
        context.verify_mode = ssl.CERT_NONE
    else:
        context.verify_mode = ssl.CERT_OPTIONAL

    # Hostname checking
    context.check_hostname = config.check_hostname and config.mode == TLSMode.VERIFY_FULL

    # Load client certificate if provided
    if config.cert_file and config.key_file:
        context.load_cert_chain(config.cert_file, config.key_file)

    logger.info(
        f"SSL context created: mode={config.mode.value}, "
        f"min_version={config.min_version.value}, "
        f"verify={context.verify_mode.name}"
    )

    return context


def get_database_ssl_config(config: Optional[TLSConfig] = None) -> Dict[str, Any]:
    """
    Get PostgreSQL SSL connection configuration.

    Returns:
        Dictionary of SSL parameters for asyncpg/psycopg2

    Example usage with asyncpg:
        ssl_config = get_database_ssl_config()
        conn = await asyncpg.connect(
            database_url,
            ssl=ssl_config.get('ssl_context')
        )

    Example usage with SQLAlchemy:
        ssl_config = get_database_ssl_config()
        engine = create_async_engine(
            database_url,
            connect_args=ssl_config
        )
    """
    if config is None:
        config = get_tls_config()

    if config.mode == TLSMode.DISABLED:
        return {"ssl": False}

    ssl_config = {}

    # PostgreSQL sslmode mapping
    mode_mapping = {
        TLSMode.OPPORTUNISTIC: "prefer",
        TLSMode.REQUIRED: "require",
        TLSMode.VERIFY_CA: "verify-ca",
        TLSMode.VERIFY_FULL: "verify-full"
    }

    ssl_config["ssl"] = mode_mapping.get(config.mode, "require")

    # Certificate paths for PostgreSQL
    if config.ca_file:
        ssl_config["sslrootcert"] = config.ca_file
    if config.cert_file:
        ssl_config["sslcert"] = config.cert_file
    if config.key_file:
        ssl_config["sslkey"] = config.key_file

    # For asyncpg, create SSL context
    ssl_context = create_ssl_context(config)
    if ssl_context:
        ssl_config["ssl_context"] = ssl_context

    logger.info(f"Database SSL config: mode={config.mode.value}")

    return ssl_config


def get_redis_ssl_config(config: Optional[TLSConfig] = None) -> Dict[str, Any]:
    """
    Get Redis SSL connection configuration.

    Returns:
        Dictionary of SSL parameters for redis-py/aioredis

    Example usage with redis-py:
        ssl_config = get_redis_ssl_config()
        client = redis.Redis(
            host='localhost',
            port=6379,
            **ssl_config
        )

    Example usage with aioredis:
        ssl_config = get_redis_ssl_config()
        client = await aioredis.from_url(
            "rediss://localhost:6379",  # Note: rediss:// for TLS
            **ssl_config
        )
    """
    if config is None:
        config = get_tls_config()

    if config.mode == TLSMode.DISABLED:
        return {"ssl": False}

    ssl_context = create_ssl_context(config)

    redis_config = {
        "ssl": True,
        "ssl_cert_reqs": (
            ssl.CERT_REQUIRED if config.mode in [TLSMode.VERIFY_CA, TLSMode.VERIFY_FULL]
            else ssl.CERT_NONE
        ),
    }

    if ssl_context:
        redis_config["ssl_context"] = ssl_context

    if config.ca_file:
        redis_config["ssl_ca_certs"] = config.ca_file
    if config.cert_file:
        redis_config["ssl_certfile"] = config.cert_file
    if config.key_file:
        redis_config["ssl_keyfile"] = config.key_file

    # Hostname verification
    redis_config["ssl_check_hostname"] = config.check_hostname and config.mode == TLSMode.VERIFY_FULL

    logger.info(f"Redis SSL config: mode={config.mode.value}")

    return redis_config


def get_enhanced_security_headers(config: Optional[TLSConfig] = None) -> Dict[str, str]:
    """
    Get enhanced security headers including HSTS and CSP.

    Returns:
        Dictionary of security headers for HTTP responses
    """
    if config is None:
        config = get_tls_config()

    headers = {
        # Prevent MIME type sniffing
        "X-Content-Type-Options": "nosniff",

        # Prevent clickjacking
        "X-Frame-Options": "DENY",

        # XSS protection (legacy, but still useful)
        "X-XSS-Protection": "1; mode=block",

        # Referrer policy
        "Referrer-Policy": "strict-origin-when-cross-origin",

        # Permissions policy (replaces Feature-Policy)
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",

        # Content Security Policy
        "Content-Security-Policy": "; ".join([
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",  # Adjust based on needs
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "font-src 'self'",
            "connect-src 'self' wss: https:",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'"
        ]),

        # Prevent caching of sensitive data
        "Cache-Control": "no-store, max-age=0",

        # Cross-Origin policies
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Resource-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp"
    }

    # HSTS header
    if config.hsts_enabled:
        hsts_value = f"max-age={config.hsts_max_age}"
        if config.hsts_include_subdomains:
            hsts_value += "; includeSubDomains"
        if config.hsts_preload:
            hsts_value += "; preload"
        headers["Strict-Transport-Security"] = hsts_value

    return headers


def get_uvicorn_ssl_config(config: Optional[TLSConfig] = None) -> Dict[str, Any]:
    """
    Get Uvicorn SSL configuration for HTTPS server.

    Returns:
        Dictionary of SSL parameters for Uvicorn

    Example usage:
        ssl_config = get_uvicorn_ssl_config()
        uvicorn.run(app, **ssl_config)
    """
    if config is None:
        config = get_tls_config()

    if config.mode == TLSMode.DISABLED:
        return {}

    if not config.cert_file or not config.key_file:
        logger.warning("SSL certificate and key files required for HTTPS")
        return {}

    uvicorn_config = {
        "ssl_certfile": config.cert_file,
        "ssl_keyfile": config.key_file,
    }

    if config.ca_file:
        uvicorn_config["ssl_ca_certs"] = config.ca_file

    # SSL version (Uvicorn uses ssl module constants)
    if config.min_version == TLSVersion.TLS_1_3:
        uvicorn_config["ssl_version"] = ssl.TLSVersion.TLSv1_3
    else:
        uvicorn_config["ssl_version"] = ssl.TLSVersion.TLSv1_2

    uvicorn_config["ssl_ciphers"] = config.ciphers

    logger.info(f"Uvicorn SSL config: cert={config.cert_file}")

    return uvicorn_config


def validate_production_tls() -> Dict[str, Any]:
    """
    Validate TLS configuration for production readiness.

    Returns:
        Dictionary with validation results and recommendations
    """
    config = get_tls_config()
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "recommendations": []
    }

    # Check TLS mode
    if config.mode == TLSMode.DISABLED:
        results["valid"] = False
        results["errors"].append("TLS is disabled - CRITICAL for production")
    elif config.mode == TLSMode.OPPORTUNISTIC:
        results["warnings"].append("TLS mode 'prefer' may fall back to unencrypted")
        results["recommendations"].append("Use 'verify-full' mode for production")
    elif config.mode == TLSMode.REQUIRED:
        results["warnings"].append("Certificate verification disabled")
        results["recommendations"].append("Enable certificate verification with 'verify-ca' or 'verify-full'")

    # Check TLS version
    if config.min_version == TLSVersion.TLS_1_2:
        results["recommendations"].append("Consider upgrading to TLS 1.3 minimum for enhanced security")

    # Check HSTS
    if not config.hsts_enabled:
        results["warnings"].append("HSTS is disabled")
        results["recommendations"].append("Enable HSTS for production")
    elif config.hsts_max_age < 31536000:
        results["recommendations"].append("HSTS max-age should be at least 1 year (31536000 seconds)")

    # Check certificate files
    if config.mode != TLSMode.DISABLED:
        if not config.cert_file:
            results["errors"].append("No certificate file configured")
            results["valid"] = False
        elif not Path(config.cert_file).exists():
            results["errors"].append(f"Certificate file not found: {config.cert_file}")
            results["valid"] = False

        if not config.key_file:
            results["errors"].append("No key file configured")
            results["valid"] = False
        elif not Path(config.key_file).exists():
            results["errors"].append(f"Key file not found: {config.key_file}")
            results["valid"] = False

    return results


# Production security checklist
PRODUCTION_SECURITY_CHECKLIST = """
PRSM Production Security Checklist
===================================

TLS/HTTPS Configuration:
[ ] TLS 1.2+ enabled (PRSM_TLS_MIN_VERSION=TLSv1.2)
[ ] Certificate verification enabled (PRSM_TLS_MODE=verify-full)
[ ] Valid SSL certificate installed (PRSM_TLS_CERT_FILE)
[ ] Private key secured (PRSM_TLS_KEY_FILE, chmod 600)
[ ] CA certificate chain configured (PRSM_TLS_CA_FILE)

HSTS Configuration:
[ ] HSTS enabled (PRSM_HSTS_ENABLED=true)
[ ] max-age >= 1 year (PRSM_HSTS_MAX_AGE=31536000)
[ ] includeSubDomains enabled (PRSM_HSTS_SUBDOMAINS=true)
[ ] Consider HSTS preload submission

Database Security:
[ ] PostgreSQL SSL enabled (sslmode=verify-full)
[ ] Database connection uses SSL
[ ] Database credentials in environment variables
[ ] Connection string not logged

Redis Security:
[ ] Redis TLS enabled (rediss:// protocol)
[ ] Redis authentication configured
[ ] Redis connection uses SSL

API Security:
[ ] All security headers configured
[ ] CORS properly restricted
[ ] Rate limiting enabled
[ ] JWT verification enabled
[ ] Token revocation working

Environment:
[ ] Debug mode disabled
[ ] Secret key rotated from development
[ ] Error messages sanitized
[ ] Logging configured (no sensitive data)
"""


def print_security_checklist():
    """Print the production security checklist."""
    print(PRODUCTION_SECURITY_CHECKLIST)


if __name__ == "__main__":
    # Validate configuration when run directly
    print("Validating TLS configuration...")
    results = validate_production_tls()

    if results["errors"]:
        print("\nERRORS:")
        for error in results["errors"]:
            print(f"  - {error}")

    if results["warnings"]:
        print("\nWARNINGS:")
        for warning in results["warnings"]:
            print(f"  - {warning}")

    if results["recommendations"]:
        print("\nRECOMMENDATIONS:")
        for rec in results["recommendations"]:
            print(f"  - {rec}")

    print(f"\nConfiguration valid: {results['valid']}")
    print("\n" + "=" * 50)
    print_security_checklist()
