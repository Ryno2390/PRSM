"""
Secure Environment Configuration

Secure environment configuration for PRSM.
Validates and manages environment variables and secrets.
"""

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
import structlog

logger = structlog.get_logger(__name__)


class SecretStrength(Enum):
    """Strength levels for secrets"""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class SecretStrengthResult:
    """
    Result of secret strength assessment.
    
    Attributes:
        key: Secret key
        exists: Whether the secret exists
        length: Length of the secret
        strength: Strength level
        score: Strength score (0-100)
        issues: List of issues found
        recommendations: List of recommendations
    """
    key: str
    exists: bool = False
    length: int = 0
    strength: SecretStrength = SecretStrength.VERY_WEAK
    score: int = 0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "key": self.key,
            "exists": self.exists,
            "length": self.length,
            "strength": self.strength.value,
            "score": self.score,
            "issues": self.issues,
            "recommendations": self.recommendations,
        }


@dataclass
class EnvironmentValidationResult:
    """
    Result of environment validation.
    
    Attributes:
        valid: Whether the environment is valid
        missing_required: List of missing required variables
        missing_recommended: List of missing recommended variables
        weak_secrets: List of weak secrets
        warnings: List of warnings
        errors: List of errors
        timestamp: When the validation was run
    """
    valid: bool = True
    missing_required: List[str] = field(default_factory=list)
    missing_recommended: List[str] = field(default_factory=list)
    weak_secrets: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "valid": self.valid,
            "missing_required": self.missing_required,
            "missing_recommended": self.missing_recommended,
            "weak_secrets": self.weak_secrets,
            "warnings": self.warnings,
            "errors": self.errors,
            "timestamp": self.timestamp.isoformat(),
        }


class SecureEnvironment:
    """
    Secure environment configuration for PRSM.
    
    Provides:
    - Environment variable validation
    - Secret strength checking
    - Configuration management
    - Security recommendations
    """
    
    # Required secrets for PRSM
    REQUIRED_SECRETS = {
        "JWT_SECRET_KEY": {
            "min_length": 32,
            "description": "Secret key for JWT token signing",
            "recommendation": "Use a cryptographically random string of at least 32 characters",
        },
        "DATABASE_URL": {
            "min_length": 10,
            "description": "Database connection URL",
            "recommendation": "Use environment-specific connection strings",
        },
        "ENCRYPTION_KEY": {
            "min_length": 32,
            "description": "Key for data encryption",
            "recommendation": "Use a cryptographically random string of at least 32 characters",
        },
    }
    
    # Recommended secrets
    RECOMMENDED_SECRETS = {
        "REDIS_URL": {
            "min_length": 10,
            "description": "Redis connection URL",
            "recommendation": "Use for caching and session management",
        },
        "API_KEY": {
            "min_length": 32,
            "description": "API key for external services",
            "recommendation": "Use a cryptographically random string",
        },
        "ADMIN_PASSWORD": {
            "min_length": 16,
            "description": "Admin user password",
            "recommendation": "Use a strong password with uppercase, lowercase, digits, and special characters",
        },
        "SMTP_PASSWORD": {
            "min_length": 8,
            "description": "SMTP server password for email",
            "recommendation": "Use app-specific password if available",
        },
    }
    
    # Required configuration
    REQUIRED_CONFIG = {
        "APP_ENV": {
            "allowed_values": ["development", "staging", "production"],
            "default": "development",
            "description": "Application environment",
        },
        "APP_NAME": {
            "default": "PRSM",
            "description": "Application name",
        },
    }
    
    # Security-related configuration
    SECURITY_CONFIG = {
        "COOKIE_SECURE": {
            "default": "true",
            "description": "Set Secure flag on cookies",
        },
        "COOKIE_HTTPONLY": {
            "default": "true",
            "description": "Set HttpOnly flag on cookies",
        },
        "COOKIE_SAMESITE": {
            "default": "lax",
            "allowed_values": ["strict", "lax", "none"],
            "description": "SameSite attribute for cookies",
        },
        "CORS_ORIGINS": {
            "default": "",
            "description": "Allowed CORS origins (comma-separated)",
        },
        "SESSION_TIMEOUT_MINUTES": {
            "default": "30",
            "description": "Session timeout in minutes",
        },
        "MAX_LOGIN_ATTEMPTS": {
            "default": "5",
            "description": "Maximum login attempts before lockout",
        },
        "LOCKOUT_DURATION_MINUTES": {
            "default": "15",
            "description": "Account lockout duration in minutes",
        },
    }
    
    # Common weak secrets
    WEAK_SECRETS = {
        "password",
        "secret",
        "admin",
        "root",
        "123456",
        "password123",
        "changeme",
        "default",
        "test",
        "development",
        "example",
        "sample",
        "demo",
        "temp",
        "temporary",
    }
    
    # Patterns for detecting secrets in code
    SECRET_PATTERNS = [
        r'(?i)(password|passwd|pwd)\s*[=:]\s*["\'](?!\s*["\'])[^"\']{3,}["\']',
        r'(?i)(secret|api_key|apikey)\s*[=:]\s*["\'](?!\s*["\'])[^"\']{8,}["\']',
        r'(?i)(token|auth)\s*[=:]\s*["\'](?!\s*["\'])[^"\']{8,}["\']',
        r'-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----',
        r'(?i)aws_access_key_id\s*[=:]\s*["\']?AKIA[0-9A-Z]{16}["\']?',
    ]
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize the secure environment.
        
        Args:
            env_file: Optional path to .env file to load
        """
        self.env_file = env_file
        self._loaded = False
        self._validation_cache: Optional[EnvironmentValidationResult] = None
        
        if env_file:
            self._load_env_file(env_file)
    
    def _load_env_file(self, env_file: str) -> None:
        """Load environment variables from a .env file"""
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse KEY=VALUE
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        
                        # Set environment variable if not already set
                        if key not in os.environ:
                            os.environ[key] = value
            
            self._loaded = True
            logger.info("Environment file loaded", file=env_file)
        except FileNotFoundError:
            logger.warning("Environment file not found", file=env_file)
        except Exception as e:
            logger.error("Failed to load environment file", file=env_file, error=str(e))
    
    def validate_environment(self, check_secrets: bool = True) -> EnvironmentValidationResult:
        """
        Validate the environment configuration.
        
        Args:
            check_secrets: Whether to check secret strength
            
        Returns:
            EnvironmentValidationResult with validation status
        """
        result = EnvironmentValidationResult()
        
        # Check required secrets
        for key, config in self.REQUIRED_SECRETS.items():
            value = os.getenv(key)
            
            if not value:
                result.missing_required.append(key)
                result.errors.append(f"Required secret '{key}' is not set")
            elif check_secrets:
                strength = self.check_secret_strength(key, value)
                if strength.strength in [SecretStrength.VERY_WEAK, SecretStrength.WEAK]:
                    result.weak_secrets.append(key)
                    result.warnings.append(
                        f"Secret '{key}' is weak: {', '.join(strength.issues)}"
                    )
        
        # Check recommended secrets
        for key, config in self.RECOMMENDED_SECRETS.items():
            value = os.getenv(key)
            
            if not value:
                result.missing_recommended.append(key)
                result.warnings.append(f"Recommended secret '{key}' is not set")
            elif check_secrets:
                strength = self.check_secret_strength(key, value)
                if strength.strength in [SecretStrength.VERY_WEAK, SecretStrength.WEAK]:
                    result.weak_secrets.append(key)
                    result.warnings.append(
                        f"Secret '{key}' is weak: {', '.join(strength.issues)}"
                    )
        
        # Check required configuration
        for key, config in self.REQUIRED_CONFIG.items():
            value = os.getenv(key)
            
            if not value:
                if "default" in config:
                    os.environ[key] = config["default"]
                    logger.debug("Using default value for config", key=key, value=config["default"])
                else:
                    result.errors.append(f"Required config '{key}' is not set")
        
        # Check security configuration
        for key, config in self.SECURITY_CONFIG.items():
            value = os.getenv(key)
            
            if not value and "default" in config:
                os.environ[key] = config["default"]
                logger.debug("Using default value for security config", key=key, value=config["default"])
            
            # Validate allowed values
            if "allowed_values" in config:
                value = os.getenv(key)
                if value and value not in config["allowed_values"]:
                    result.warnings.append(
                        f"Security config '{key}' has invalid value '{value}'. "
                        f"Allowed: {', '.join(config['allowed_values'])}"
                    )
        
        # Check for development settings in production
        app_env = os.getenv("APP_ENV", "development")
        if app_env == "production":
            # Check for development-only settings
            debug = os.getenv("DEBUG", "false").lower()
            if debug == "true":
                result.errors.append("DEBUG should not be true in production")
            
            cors_origins = os.getenv("CORS_ORIGINS", "")
            if cors_origins == "*":
                raise EnvironmentError(
                    "FATAL: CORS_ORIGINS='*' is not allowed in production. "
                    "Set CORS_ORIGINS to a comma-separated list of specific allowed origins "
                    "(e.g., 'https://prsm.ai,https://app.prsm.ai')."
                )
        
        # Determine if environment is valid
        result.valid = len(result.errors) == 0 and len(result.missing_required) == 0
        
        self._validation_cache = result
        
        logger.info(
            "Environment validation completed",
            valid=result.valid,
            errors=len(result.errors),
            warnings=len(result.warnings),
        )
        
        return result
    
    def check_secret_strength(
        self,
        key: str,
        value: Optional[str] = None,
    ) -> SecretStrengthResult:
        """
        Check the strength of a secret.
        
        Args:
            key: Secret key
            value: Secret value (if None, will get from environment)
            
        Returns:
            SecretStrengthResult with strength assessment
        """
        if value is None:
            value = os.getenv(key)
        
        result = SecretStrengthResult(key=key)
        
        if not value:
            result.exists = False
            result.issues.append("Secret is not set")
            result.recommendations.append(f"Set the {key} environment variable")
            return result
        
        result.exists = True
        result.length = len(value)
        
        # Check minimum length
        config = self.REQUIRED_SECRETS.get(key) or self.RECOMMENDED_SECRETS.get(key) or {}
        min_length = config.get("min_length", 8)
        
        if result.length < min_length:
            result.issues.append(f"Length ({result.length}) is below minimum ({min_length})")
            result.recommendations.append(f"Increase length to at least {min_length} characters")
        
        # Check for weak values
        if value.lower() in self.WEAK_SECRETS:
            result.issues.append("Secret is a common weak value")
            result.recommendations.append("Use a cryptographically random value")
        
        # Check character composition
        has_uppercase = any(c.isupper() for c in value)
        has_lowercase = any(c.islower() for c in value)
        has_digits = any(c.isdigit() for c in value)
        has_special = any(not c.isalnum() for c in value)
        
        if not has_uppercase:
            result.issues.append("No uppercase characters")
            result.recommendations.append("Add uppercase characters")
        
        if not has_lowercase:
            result.issues.append("No lowercase characters")
            result.recommendations.append("Add lowercase characters")
        
        if not has_digits:
            result.issues.append("No digit characters")
            result.recommendations.append("Add digit characters")
        
        if not has_special:
            result.issues.append("No special characters")
            result.recommendations.append("Add special characters")
        
        # Check for patterns
        if re.match(r'^[a-z]+$', value):
            result.issues.append("Only lowercase letters")
            result.recommendations.append("Add variety to character types")
        
        if re.match(r'^[0-9]+$', value):
            result.issues.append("Only digits")
            result.recommendations.append("Add letters and special characters")
        
        # Check for repeated characters
        if len(set(value)) < len(value) * 0.5:
            result.issues.append("Many repeated characters")
            result.recommendations.append("Use more unique characters")
        
        # Check for sequential characters
        for i in range(len(value) - 2):
            if ord(value[i]) + 1 == ord(value[i+1]) == ord(value[i+2]) - 1:
                result.issues.append("Sequential characters detected")
                result.recommendations.append("Avoid sequential characters")
                break
        
        # Calculate score
        score = 0
        
        # Length score
        if result.length >= 8:
            score += 10
        if result.length >= 12:
            score += 10
        if result.length >= 16:
            score += 10
        if result.length >= 24:
            score += 10
        if result.length >= 32:
            score += 10
        
        # Character variety score
        if has_uppercase:
            score += 10
        if has_lowercase:
            score += 10
        if has_digits:
            score += 10
        if has_special:
            score += 15
        
        # Entropy bonus
        unique_chars = len(set(value))
        if unique_chars >= result.length * 0.7:
            score += 15
        
        # Deductions
        if value.lower() in self.WEAK_SECRETS:
            score -= 50
        
        if len(result.issues) > 2:
            score -= len(result.issues) * 5
        
        # Normalize score to 0-100
        score = max(0, min(100, score))
        result.score = score
        
        # Determine strength
        if score >= 80:
            result.strength = SecretStrength.VERY_STRONG
        elif score >= 60:
            result.strength = SecretStrength.STRONG
        elif score >= 40:
            result.strength = SecretStrength.MEDIUM
        elif score >= 20:
            result.strength = SecretStrength.WEAK
        else:
            result.strength = SecretStrength.VERY_WEAK
        
        return result
    
    def get_required_secrets(self) -> Dict[str, str]:
        """
        Get all required secrets.
        
        Returns:
            Dictionary of secret keys to values
        """
        return {key: os.getenv(key, "") for key in self.REQUIRED_SECRETS}
    
    def get_recommended_secrets(self) -> Dict[str, str]:
        """
        Get all recommended secrets.
        
        Returns:
            Dictionary of secret keys to values
        """
        return {key: os.getenv(key, "") for key in self.RECOMMENDED_SECRETS}
    
    def get_security_config(self) -> Dict[str, str]:
        """
        Get all security configuration.
        
        Returns:
            Dictionary of config keys to values
        """
        config = {}
        for key in self.SECURITY_CONFIG:
            config[key] = os.getenv(key, self.SECURITY_CONFIG[key].get("default", ""))
        return config
    
    def get_app_config(self) -> Dict[str, str]:
        """
        Get all application configuration.
        
        Returns:
            Dictionary of config keys to values
        """
        config = {}
        for key in self.REQUIRED_CONFIG:
            config[key] = os.getenv(key, self.REQUIRED_CONFIG[key].get("default", ""))
        return config
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return os.getenv("APP_ENV", "development").lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return os.getenv("APP_ENV", "development").lower() == "development"
    
    def is_staging(self) -> bool:
        """Check if running in staging environment"""
        return os.getenv("APP_ENV", "development").lower() == "staging"
    
    def get_environment(self) -> str:
        """Get the current environment"""
        return os.getenv("APP_ENV", "development")
    
    def set_environment(self, env: str) -> None:
        """
        Set the environment.
        
        Args:
            env: Environment name (development, staging, production)
        """
        os.environ["APP_ENV"] = env
        logger.info("Environment set", environment=env)
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get an environment variable.
        
        Args:
            key: Environment variable key
            default: Default value if not set
            
        Returns:
            Environment variable value or default
        """
        return os.getenv(key, default)
    
    def get_int(self, key: str, default: int = 0) -> int:
        """
        Get an environment variable as an integer.
        
        Args:
            key: Environment variable key
            default: Default value if not set
            
        Returns:
            Environment variable value as integer
        """
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """
        Get an environment variable as a boolean.
        
        Args:
            key: Environment variable key
            default: Default value if not set
            
        Returns:
            Environment variable value as boolean
        """
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")
    
    def get_list(self, key: str, separator: str = ",", default: Optional[List[str]] = None) -> List[str]:
        """
        Get an environment variable as a list.
        
        Args:
            key: Environment variable key
            separator: List separator
            default: Default value if not set
            
        Returns:
            Environment variable value as list
        """
        value = os.getenv(key)
        if value is None:
            return default or []
        return [v.strip() for v in value.split(separator) if v.strip()]
    
    def set(self, key: str, value: str) -> None:
        """
        Set an environment variable.
        
        Args:
            key: Environment variable key
            value: Environment variable value
        """
        os.environ[key] = value
        logger.debug("Environment variable set", key=key)
    
    def unset(self, key: str) -> None:
        """
        Unset an environment variable.
        
        Args:
            key: Environment variable key
        """
        if key in os.environ:
            del os.environ[key]
            logger.debug("Environment variable unset", key=key)
    
    def scan_for_secrets(self, directory: str) -> List[Dict[str, Any]]:
        """
        Scan a directory for potential secret leaks.
        
        Args:
            directory: Directory to scan
            
        Returns:
            List of potential secret leaks
        """
        import pathlib
        
        leaks = []
        dir_path = pathlib.Path(directory)
        
        if not dir_path.exists():
            logger.warning("Directory does not exist", directory=directory)
            return leaks
        
        # Files to exclude
        excluded_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", "env"}
        excluded_files = {".env", ".env.local", ".env.production", "credentials.json", "secrets.json"}
        
        for file_path in dir_path.rglob("*"):
            # Skip excluded directories
            if any(excluded in file_path.parts for excluded in excluded_dirs):
                continue
            
            # Skip excluded files
            if file_path.name in excluded_files:
                leaks.append({
                    "file": str(file_path.relative_to(dir_path)),
                    "line": 0,
                    "type": "excluded_file",
                    "message": f"Excluded file: {file_path.name}",
                })
                continue
            
            # Only scan text files
            if not file_path.is_file():
                continue
            
            try:
                content = file_path.read_text()
                lines = content.splitlines()
                
                for i, line in enumerate(lines, 1):
                    for pattern in self.SECRET_PATTERNS:
                        if re.search(pattern, line):
                            # Mask the potential secret
                            masked_line = re.sub(
                                pattern,
                                lambda m: m.group(0)[:10] + "...[MASKED]",
                                line
                            )
                            
                            leaks.append({
                                "file": str(file_path.relative_to(dir_path)),
                                "line": i,
                                "type": "potential_secret",
                                "message": f"Potential secret detected",
                                "snippet": masked_line[:100],
                            })
                            break
            except (UnicodeDecodeError, IOError):
                # Skip binary files
                pass
        
        return leaks
    
    def generate_env_template(self) -> str:
        """
        Generate a template .env file.
        
        Returns:
            Template .env file content
        """
        lines = [
            "# PRSM Environment Configuration",
            "# Generated by SecureEnvironment",
            "",
            "# Application Configuration",
        ]
        
        # Add required config
        for key, config in self.REQUIRED_CONFIG.items():
            default = config.get("default", "")
            description = config.get("description", "")
            allowed = config.get("allowed_values", [])
            
            lines.append(f"# {description}")
            if allowed:
                lines.append(f"# Allowed values: {', '.join(allowed)}")
            lines.append(f"{key}={default}")
            lines.append("")
        
        lines.append("# Security Configuration")
        
        # Add security config
        for key, config in self.SECURITY_CONFIG.items():
            default = config.get("default", "")
            description = config.get("description", "")
            allowed = config.get("allowed_values", [])
            
            lines.append(f"# {description}")
            if allowed:
                lines.append(f"# Allowed values: {', '.join(allowed)}")
            lines.append(f"{key}={default}")
            lines.append("")
        
        lines.append("# Required Secrets")
        
        # Add required secrets
        for key, config in self.REQUIRED_SECRETS.items():
            min_length = config.get("min_length", 8)
            description = config.get("description", "")
            recommendation = config.get("recommendation", "")
            
            lines.append(f"# {description}")
            lines.append(f"# Minimum length: {min_length}")
            lines.append(f"# {recommendation}")
            lines.append(f"{key}=<generate-secure-secret>")
            lines.append("")
        
        lines.append("# Recommended Secrets")
        
        # Add recommended secrets
        for key, config in self.RECOMMENDED_SECRETS.items():
            min_length = config.get("min_length", 8)
            description = config.get("description", "")
            recommendation = config.get("recommendation", "")
            
            lines.append(f"# {description}")
            lines.append(f"# Minimum length: {min_length}")
            lines.append(f"# {recommendation}")
            lines.append(f"{key}=")
            lines.append("")
        
        return "\n".join(lines)
    
    def export_validation_report(self) -> str:
        """
        Export validation report as a string.
        
        Returns:
            Validation report
        """
        if self._validation_cache is None:
            self.validate_environment()
        
        result = self._validation_cache
        
        lines = [
            "# PRSM Environment Validation Report",
            f"Generated: {result.timestamp.isoformat()}",
            "",
            f"## Status: {'✅ VALID' if result.valid else '❌ INVALID'}",
            "",
        ]
        
        if result.missing_required:
            lines.append("## Missing Required Secrets")
            for key in result.missing_required:
                lines.append(f"- {key}")
            lines.append("")
        
        if result.missing_recommended:
            lines.append("## Missing Recommended Secrets")
            for key in result.missing_recommended:
                lines.append(f"- {key}")
            lines.append("")
        
        if result.weak_secrets:
            lines.append("## Weak Secrets")
            for key in result.weak_secrets:
                lines.append(f"- {key}")
            lines.append("")
        
        if result.errors:
            lines.append("## Errors")
            for error in result.errors:
                lines.append(f"- ❌ {error}")
            lines.append("")
        
        if result.warnings:
            lines.append("## Warnings")
            for warning in result.warnings:
                lines.append(f"- ⚠️ {warning}")
            lines.append("")
        
        return "\n".join(lines)


# Global secure environment instance
_secure_environment: Optional[SecureEnvironment] = None


def get_secure_environment() -> SecureEnvironment:
    """Get the global secure environment instance"""
    global _secure_environment
    
    if _secure_environment is None:
        _secure_environment = SecureEnvironment()
    
    return _secure_environment


def validate_environment(check_secrets: bool = True) -> EnvironmentValidationResult:
    """
    Convenience function to validate the environment.
    
    Args:
        check_secrets: Whether to check secret strength
        
    Returns:
        EnvironmentValidationResult
    """
    env = get_secure_environment()
    return env.validate_environment(check_secrets=check_secrets)


def get_config(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Convenience function to get a configuration value.
    
    Args:
        key: Configuration key
        default: Default value
        
    Returns:
        Configuration value
    """
    env = get_secure_environment()
    return env.get(key, default)