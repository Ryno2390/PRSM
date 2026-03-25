"""
Centralized Secrets Management for PRSM

Provides a unified interface for loading secrets from environment variables,
with support for future extension to HashiCorp Vault, AWS Secrets Manager,
or other secret backends.

🎯 PURPOSE IN PRSM:
- Centralized secret loading and validation
- Clear error messages when secrets are missing
- Documentation of required vs optional secrets
- Abstraction for future secret backend migration

🔧 USAGE:
    from prsm.core.secrets import get_secrets, MissingSecretError

    secrets = get_secrets()
    api_key = secrets.get("ANTHROPIC_API_KEY", required=True)

🚀 FEATURES:
- Environment variable loading (default)
- Required vs optional secret distinction
- Startup validation of all required secrets
- Clear error messages with documentation pointers
"""

import os
from typing import Optional, Dict, Any, List
from functools import lru_cache
import structlog

logger = structlog.get_logger(__name__)


class MissingSecretError(Exception):
    """
    Raised when a required secret is not available.

    Includes helpful context about where to find the secret
    in the configuration template.
    """
    def __init__(self, secret_name: str, message: str = None):
        self.secret_name = secret_name
        self.message = message or (
            f"Required secret '{secret_name}' is not set. "
            f"Add it to your .env file or environment. "
            f"See config/secure.env.template for documentation."
        )
        super().__init__(self.message)


class SecretsManager:
    """
    Centralized secrets management for PRSM.

    Loads secrets from the environment by default. The interface is designed
    to be easily extended to support other backends (HashiCorp Vault,
    AWS Secrets Manager, etc.) without changing calling code.

    Usage:
        secrets = get_secrets()
        api_key = secrets.get("ANTHROPIC_API_KEY", required=True)
        optional_key = secrets.get("OPTIONAL_KEY", default="default_value")

    Attributes:
        REQUIRED_SECRETS: List of secrets that must be set for the node to start
        OPTIONAL_SECRETS_WITH_DOCS: Dictionary of optional secrets with documentation
    """

    # Secrets that are required for node startup
    REQUIRED_SECRETS: List[str] = [
        # Authentication
        "JWT_SECRET_KEY",  # Or SECRET_KEY as fallback

        # Database
        # DATABASE_URL is often set but has a default in some environments
    ]

    # Optional secrets with documentation
    OPTIONAL_SECRETS_WITH_DOCS: Dict[str, str] = {
        # AI Provider API Keys
        "ANTHROPIC_API_KEY": "Required for Anthropic AI backend (Claude models)",
        "OPENAI_API_KEY": "Required for OpenAI AI backend (GPT models)",

        # Payment Processing
        "STRIPE_API_KEY": "Required for fiat gateway (Stripe payments)",
        "STRIPE_WEBHOOK_SECRET": "Required for Stripe webhook validation",
        "PAYPAL_CLIENT_ID": "Required for fiat gateway (PayPal payments)",
        "PAYPAL_CLIENT_SECRET": "Required for PayPal API authentication",

        # Infrastructure
        "IPFS_GATEWAY_URL": "Custom IPFS gateway URL (defaults to public gateway)",
        "REDIS_URL": "Redis connection URL for caching and rate limiting",

        # Security
        "ENCRYPTION_KEY": "Key for encrypting sensitive data at rest",

        # External Services
        "COINMARKETCAP_API_KEY": "For cryptocurrency price fetching",
        "COINGECKO_API_KEY": "For cryptocurrency price fetching",

        # Monitoring
        "SENTRY_DSN": "Sentry error tracking DSN",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "OpenTelemetry exporter endpoint",

        # Blockchain
        "WEB3_PROVIDER_URL": "Ethereum/Web3 provider URL",
        "POLYGON_RPC_URL": "Polygon network RPC URL",
    }

    def __init__(self, env: Dict[str, str] = None):
        """
        Initialize the secrets manager.

        Args:
            env: Environment dictionary to use (defaults to os.environ)
        """
        self._env = env if env is not None else dict(os.environ)
        self._cache: Dict[str, Optional[str]] = {}

    def get(
        self,
        key: str,
        required: bool = False,
        default: Optional[str] = None
    ) -> Optional[str]:
        """
        Get a secret value by key.

        Args:
            key: The secret name/environment variable
            required: If True, raises MissingSecretError when not found
            default: Default value if secret is not found

        Returns:
            The secret value, or default/None if not found

        Raises:
            MissingSecretError: If required=True and secret is not set
        """
        # Check cache first
        if key in self._cache:
            return self._cache[key] if self._cache[key] is not None else default

        # Look up the secret
        value = self._env.get(key, default)

        # Handle required secrets
        if required and not value:
            raise MissingSecretError(key)

        # Cache the result
        self._cache[key] = value

        return value

    def get_int(
        self,
        key: str,
        required: bool = False,
        default: int = 0
    ) -> int:
        """
        Get a secret as an integer.

        Args:
            key: The secret name
            required: If True, raises error when not found
            default: Default integer value

        Returns:
            The secret as an integer

        Raises:
            ValueError: If the secret cannot be parsed as an integer
        """
        value = self.get(key, required=required)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"Secret '{key}' is not a valid integer: {value}")

    def get_float(
        self,
        key: str,
        required: bool = False,
        default: float = 0.0
    ) -> float:
        """
        Get a secret as a float.

        Args:
            key: The secret name
            required: If True, raises error when not found
            default: Default float value

        Returns:
            The secret as a float

        Raises:
            ValueError: If the secret cannot be parsed as a float
        """
        value = self.get(key, required=required)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Secret '{key}' is not a valid float: {value}")

    def get_bool(
        self,
        key: str,
        required: bool = False,
        default: bool = False
    ) -> bool:
        """
        Get a secret as a boolean.

        Recognizes: "true", "1", "yes", "on" (case-insensitive) as True
        Everything else as False

        Args:
            key: The secret name
            required: If True, raises error when not found
            default: Default boolean value

        Returns:
            The secret as a boolean
        """
        value = self.get(key, required=required)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")

    def validate_all_required(self) -> List[str]:
        """
        Validate that all required secrets are set.

        This should be called at application startup to fail fast
        if critical configuration is missing.

        Returns:
            List of missing required secret names (empty if all present)

        Raises:
            MissingSecretError: If any required secrets are missing
        """
        missing = []

        for key in self.REQUIRED_SECRETS:
            # Handle fallback keys (e.g., JWT_SECRET_KEY or SECRET_KEY)
            if key == "JWT_SECRET_KEY":
                if not (self._env.get("JWT_SECRET_KEY") or self._env.get("SECRET_KEY")):
                    missing.append("JWT_SECRET_KEY (or SECRET_KEY)")
            elif not self._env.get(key):
                missing.append(key)

        if missing:
            raise MissingSecretError(
                ", ".join(missing),
                f"Missing required secrets: {', '.join(missing)}. "
                f"See config/secure.env.template for documentation."
            )

        return []

    def check_optional_secrets(self) -> Dict[str, bool]:
        """
        Check which optional secrets are configured.

        Useful for startup logging and feature availability checks.

        Returns:
            Dictionary mapping secret names to whether they're configured
        """
        status = {}
        for key, doc in self.OPTIONAL_SECRETS_WITH_DOCS.items():
            status[key] = bool(self._env.get(key))
        return status

    def get_missing_optional(self) -> List[str]:
        """
        Get list of optional secrets that are not configured.

        Returns:
            List of secret names that are not set
        """
        return [
            key for key, doc in self.OPTIONAL_SECRETS_WITH_DOCS.items()
            if not self._env.get(key)
        ]

    def log_secret_status(self):
        """Log the status of all secrets at startup."""
        # Check required secrets
        try:
            self.validate_all_required()
            logger.info("All required secrets are configured")
        except MissingSecretError as e:
            logger.warning("Missing required secrets", missing=e.secret_name)

        # Check optional secrets
        status = self.check_optional_secrets()
        configured = [k for k, v in status.items() if v]
        missing = [k for k, v in status.items() if not v]

        if configured:
            logger.info(
                "Optional secrets configured",
                secrets=configured
            )

        if missing:
            logger.info(
                "Optional secrets not configured (features may be limited)",
                missing=missing
            )

    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()


# Module-level singleton
@lru_cache(maxsize=1)
def get_secrets() -> SecretsManager:
    """
    Get the global SecretsManager instance.

    Uses lru_cache for singleton pattern - returns the same instance
    on every call within the same process.

    Returns:
        SecretsManager: The global secrets manager instance
    """
    return SecretsManager()


def reset_secrets():
    """
    Reset the secrets manager singleton.

    Useful for testing or when environment changes need to be picked up.
    """
    get_secrets.cache_clear()


# Convenience functions for common patterns
def get_api_key(provider: str) -> Optional[str]:
    """
    Get API key for a specific provider.

    Args:
        provider: Provider name (anthropic, openai, stripe, etc.)

    Returns:
        API key if configured, None otherwise
    """
    secrets = get_secrets()
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "stripe": "STRIPE_API_KEY",
        "paypal_client_id": "PAYPAL_CLIENT_ID",
        "paypal_client_secret": "PAYPAL_CLIENT_SECRET",
        "coinmarketcap": "COINMARKETCAP_API_KEY",
        "coingecko": "COINGECKO_API_KEY",
    }

    env_key = key_map.get(provider.lower())
    if not env_key:
        logger.warning(f"Unknown provider for API key: {provider}")
        return None

    return secrets.get(env_key)


def require_api_key(provider: str) -> str:
    """
    Get API key for a provider, raising error if not configured.

    Args:
        provider: Provider name

    Returns:
        API key

    Raises:
        MissingSecretError: If API key is not configured
    """
    secrets = get_secrets()
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "stripe": "STRIPE_API_KEY",
        "paypal_client_id": "PAYPAL_CLIENT_ID",
        "paypal_client_secret": "PAYPAL_CLIENT_SECRET",
    }

    env_key = key_map.get(provider.lower())
    if not env_key:
        raise ValueError(f"Unknown provider for API key: {provider}")

    return secrets.get(env_key, required=True)
