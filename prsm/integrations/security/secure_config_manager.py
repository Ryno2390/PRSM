"""
Secure Configuration Manager
============================

Manages secure configuration and credential initialization for PRSM.
Handles migration from insecure environment variables to encrypted
credential storage while maintaining backward compatibility.
"""

import asyncio
import os
import secrets
import structlog
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path

from .secure_api_client_factory import SecureClientType, secure_client_factory
from ..config.credential_manager import CredentialManager, CredentialType
from ..models.integration_models import IntegrationPlatform
from .audit_logger import audit_logger
from ...core.config import settings

logger = structlog.get_logger(__name__)


class SecureConfigManager:
    """
    Secure Configuration Manager
    
    Features:
    - Migrates environment variables to encrypted storage
    - Manages system-level credentials
    - Provides secure configuration templates
    - Handles credential rotation and validation
    - Maintains audit trail of configuration changes
    """
    
    def __init__(self):
        self.credential_manager = CredentialManager()
        self.system_user_id = "system"
        self._migration_completed = False
        
    async def initialize_secure_configuration(self) -> bool:
        """
        Initialize secure configuration system
        
        This method:
        1. Migrates environment variables to encrypted storage
        2. Validates existing credentials
        3. Sets up system credentials
        4. Creates secure configuration templates
        
        Returns:
            True if initialization succeeded, False otherwise
        """
        try:
            logger.info("Initializing secure configuration system")
            
            # Step 1: Migrate environment variables to secure storage
            migration_success = await self._migrate_environment_credentials()
            
            # Step 2: Validate all credentials
            validation_success = await self._validate_all_credentials()
            
            # Step 3: Generate secure system secrets
            secret_generation_success = await self._generate_secure_system_secrets()
            
            # Step 4: Create configuration templates
            template_success = await self._create_secure_config_templates()
            
            # Log initialization result
            success = migration_success and validation_success and secret_generation_success and template_success
            
            await self._log_initialization_event(success, {
                "migration_success": migration_success,
                "validation_success": validation_success,
                "secret_generation_success": secret_generation_success,
                "template_success": template_success
            })
            
            if success:
                self._migration_completed = True
                logger.info("Secure configuration system initialized successfully")
            else:
                logger.error("Failed to initialize secure configuration system")
            
            return success
            
        except Exception as e:
            logger.error("Error initializing secure configuration", error=str(e))
            await self._log_initialization_event(False, {"error": str(e)})
            return False
    
    async def register_api_credentials(
        self,
        platform: str,
        credentials: Dict[str, Any],
        user_id: Optional[str] = None,
        expires_at: Optional[datetime] = None
    ) -> bool:
        """
        Register API credentials for a platform
        
        Args:
            platform: Platform name (openai, anthropic, huggingface, etc.)
            credentials: Credential data
            user_id: User ID (defaults to system)
            expires_at: Optional expiration time
            
        Returns:
            True if registration succeeded, False otherwise
        """
        try:
            if user_id is None:
                user_id = self.system_user_id
            
            # Map platform name to client type
            client_type = self._get_client_type_for_platform(platform)
            if not client_type:
                logger.error("Unknown platform for credential registration", platform=platform)
                return False
            
            # Register credentials using secure factory
            success = await secure_client_factory.register_user_credentials(
                client_type=client_type,
                user_id=user_id,
                credentials=credentials,
                expires_at=expires_at
            )
            
            if success:
                logger.info("API credentials registered successfully",
                           platform=platform,
                           user_id=user_id)
            else:
                logger.error("Failed to register API credentials",
                           platform=platform,
                           user_id=user_id)
            
            return success
            
        except Exception as e:
            logger.error("Error registering API credentials",
                        platform=platform,
                        user_id=user_id,
                        error=str(e))
            return False
    
    async def get_secure_configuration_status(self) -> Dict[str, Any]:
        """
        Get status of secure configuration system
        
        Returns:
            Dictionary with configuration status information
        """
        try:
            # Check credential availability for each platform
            platforms = [
                SecureClientType.OPENAI,
                SecureClientType.ANTHROPIC,
                SecureClientType.HUGGINGFACE,
                SecureClientType.GITHUB,
                SecureClientType.PINECONE,
                SecureClientType.WEAVIATE,
                SecureClientType.OLLAMA
            ]
            
            platform_status = {}
            for platform in platforms:
                has_credentials = await secure_client_factory.validate_client_credentials(
                    platform, self.system_user_id
                )
                platform_status[platform] = {
                    "credentials_available": has_credentials,
                    "client_type": platform
                }
            
            # Check system secrets
            system_secrets_secure = await self._check_system_secrets_security()
            
            return {
                "migration_completed": self._migration_completed,
                "system_secrets_secure": system_secrets_secure,
                "platform_credentials": platform_status,
                "credential_manager_available": self.credential_manager is not None,
                "system_user_id": self.system_user_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error("Error getting secure configuration status", error=str(e))
            return {
                "error": str(e),
                "migration_completed": False,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def rotate_platform_credentials(
        self,
        platform: str,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Rotate credentials for a specific platform
        
        Args:
            platform: Platform name
            user_id: User ID (defaults to system)
            
        Returns:
            True if rotation succeeded, False otherwise
        """
        try:
            if user_id is None:
                user_id = self.system_user_id
            
            client_type = self._get_client_type_for_platform(platform)
            if not client_type:
                return False
            
            success = await secure_client_factory.rotate_credentials(client_type, user_id)
            
            if success:
                logger.info("Platform credentials rotated successfully",
                           platform=platform,
                           user_id=user_id)
            else:
                logger.info("Platform credential rotation not implemented",
                           platform=platform,
                           user_id=user_id)
            
            return success
            
        except Exception as e:
            logger.error("Error rotating platform credentials",
                        platform=platform,
                        user_id=user_id,
                        error=str(e))
            return False
    
    async def _migrate_environment_credentials(self) -> bool:
        """
        Migrate credentials from environment variables to encrypted storage
        """
        try:
            logger.info("Starting credential migration from environment variables")
            
            # Define environment variable mappings
            env_mappings = {
                'OPENAI_API_KEY': SecureClientType.OPENAI,
                'ANTHROPIC_API_KEY': SecureClientType.ANTHROPIC,
                'HUGGINGFACE_API_KEY': SecureClientType.HUGGINGFACE,
                'GITHUB_ACCESS_TOKEN': SecureClientType.GITHUB,
                'PINECONE_API_KEY': SecureClientType.PINECONE,
                'WEAVIATE_API_KEY': SecureClientType.WEAVIATE,
                'OLLAMA_API_KEY': SecureClientType.OLLAMA
            }
            
            migration_count = 0
            for env_var, client_type in env_mappings.items():
                value = os.getenv(env_var)
                if value:
                    # Check if credentials already exist in secure storage
                    has_secure_creds = await secure_client_factory.validate_client_credentials(
                        client_type, self.system_user_id
                    )
                    
                    if not has_secure_creds:
                        # Migrate to secure storage
                        credentials = self._format_credentials_for_client_type(client_type, value)
                        
                        success = await secure_client_factory.register_user_credentials(
                            client_type=client_type,
                            user_id=self.system_user_id,
                            credentials=credentials
                        )
                        
                        if success:
                            migration_count += 1
                            logger.info("Migrated credential from environment",
                                       env_var=env_var,
                                       client_type=client_type)
                        else:
                            logger.error("Failed to migrate credential",
                                        env_var=env_var,
                                        client_type=client_type)
                    else:
                        logger.debug("Secure credentials already exist, skipping migration",
                                   client_type=client_type)
            
            logger.info("Credential migration completed", migrated_count=migration_count)
            return True
            
        except Exception as e:
            logger.error("Error during credential migration", error=str(e))
            return False
    
    async def _validate_all_credentials(self) -> bool:
        """
        Validate all stored credentials
        """
        try:
            logger.info("Validating all stored credentials")
            
            validation_results = {}
            platforms = [
                SecureClientType.OPENAI,
                SecureClientType.ANTHROPIC,
                SecureClientType.HUGGINGFACE,
                SecureClientType.GITHUB,
                SecureClientType.PINECONE,
                SecureClientType.WEAVIATE,
                SecureClientType.OLLAMA
            ]
            
            for platform in platforms:
                is_valid = await secure_client_factory.validate_client_credentials(
                    platform, self.system_user_id
                )
                validation_results[platform] = is_valid
                
                if is_valid:
                    logger.debug("Credentials valid", platform=platform)
                else:
                    logger.debug("Credentials not available or invalid", platform=platform)
            
            # Return True if validation process completed (not all credentials need to be valid)
            logger.info("Credential validation completed", results=validation_results)
            return True
            
        except Exception as e:
            logger.error("Error during credential validation", error=str(e))
            return False
    
    async def _generate_secure_system_secrets(self) -> bool:
        """
        Generate secure system secrets (JWT keys, encryption keys, etc.)
        """
        try:
            logger.info("Generating secure system secrets")
            
            # Check if JWT secret is secure
            current_secret = settings.secret_key
            if not current_secret or len(current_secret) < 32 or current_secret.startswith("test-"):
                logger.warning("Insecure JWT secret detected, generating secure secret")
                
                # Generate a secure secret
                secure_secret = secrets.token_urlsafe(64)
                
                # Update settings with secure secret through secure configuration system
                secret_update_result = await self._update_system_secret("jwt_secret_key", secure_secret)
                
                if secret_update_result:
                    logger.info("Secure JWT secret generated and applied to configuration system")
                    
                    # Store backup in secure storage
                    await self._backup_secure_secret("jwt_secret_key", secure_secret)
                    
                    # Update runtime configuration if possible
                    await self._update_runtime_configuration("secret_key", secure_secret)
                else:
                    logger.warning("Secure JWT secret generated but configuration update failed - manual update required")
                    logger.info("Secure JWT secret generated (manual configuration update required)",
                               secret_length=len(secure_secret))
            
            return True
            
        except Exception as e:
            logger.error("Error generating secure system secrets", error=str(e))
            return False
    
    async def _create_secure_config_templates(self) -> bool:
        """
        Create secure configuration templates for deployment
        """
        try:
            logger.info("Creating secure configuration templates")
            
            # Create secure .env template
            secure_env_template = self._generate_secure_env_template()
            
            # Write template to config directory
            config_dir = Path("config")
            config_dir.mkdir(exist_ok=True)
            
            template_path = config_dir / "secure.env.template"
            with open(template_path, 'w') as f:
                f.write(secure_env_template)
            
            logger.info("Secure configuration template created", path=str(template_path))
            return True
            
        except Exception as e:
            logger.error("Error creating secure configuration templates", error=str(e))
            return False
    
    async def _check_system_secrets_security(self) -> bool:
        """
        Check if system secrets are secure
        """
        try:
            # Check JWT secret
            secret = settings.secret_key
            if not secret or len(secret) < 32 or secret.startswith("test-"):
                return False
            
            # Add other secret checks here
            return True
            
        except Exception:
            return False
    
    def _get_client_type_for_platform(self, platform: str) -> Optional[SecureClientType]:
        """
        Map platform name to SecureClientType
        """
        mapping = {
            'openai': SecureClientType.OPENAI,
            'anthropic': SecureClientType.ANTHROPIC,
            'huggingface': SecureClientType.HUGGINGFACE,
            'github': SecureClientType.GITHUB,
            'pinecone': SecureClientType.PINECONE,
            'weaviate': SecureClientType.WEAVIATE,
            'ollama': SecureClientType.OLLAMA
        }
        return mapping.get(platform.lower())
    
    def _format_credentials_for_client_type(self, client_type: SecureClientType, value: str) -> Dict[str, Any]:
        """
        Format credential value for specific client type
        """
        if client_type == SecureClientType.GITHUB:
            return {"access_token": value}
        elif client_type in [SecureClientType.PINECONE]:
            return {"api_key": value, "environment": "us-west1-gcp"}
        elif client_type in [SecureClientType.WEAVIATE]:
            return {"api_key": value, "url": "http://localhost:8080"}
        elif client_type == SecureClientType.OLLAMA:
            return {"base_url": value or "http://localhost:11434"}
        else:
            return {"api_key": value}
    
    def _generate_secure_env_template(self) -> str:
        """
        Generate secure environment template
        """
        template = """# PRSM Secure Configuration Template
# This template shows how to configure PRSM with secure credential management
# 
# IMPORTANT: Do NOT put actual credentials in this file!
# Use the credential management API to register credentials securely.

# JWT Security (REQUIRED - Generate a secure random string)
SECRET_KEY=GENERATE_SECURE_RANDOM_STRING_64_CHARS_MINIMUM

# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/prsm
REDIS_URL=redis://localhost:6379/0

# Optional: Local model paths
PRSM_LOCAL_MODELS_PATH=/models

# Web3 Configuration (if using blockchain features)
WEB3_NETWORK=polygon_mumbai
POLYGON_MUMBAI_RPC_URL=https://rpc-mumbai.maticvigil.com

# SECURITY NOTES:
# 1. API keys should be registered via the credential management API
# 2. Use strong, unique passwords for database connections
# 3. Generate a cryptographically secure SECRET_KEY
# 4. Never commit actual credentials to version control
# 5. Use environment-specific configuration files

# Example credential registration (use PRSM API):
# POST /api/v1/integrations/credentials
# {
#   "platform": "openai",
#   "credentials": {"api_key": "your-secure-key"},
#   "expires_at": "2024-12-31T23:59:59Z"
# }
"""
        return template
    
    async def _log_initialization_event(self, success: bool, details: Dict[str, Any]):
        """
        Log secure configuration initialization event
        """
        try:
            await audit_logger.log_security_event(
                event_type="secure_config_initialization",
                user_id=self.system_user_id,
                details={
                    "success": success,
                    **details
                },
                security_level="info" if success else "error"
            )
        except Exception as e:
            logger.error("Failed to log initialization event", error=str(e))


# Global secure configuration manager
secure_config_manager = SecureConfigManager()


async def initialize_secure_configuration() -> bool:
    """
    Initialize the secure configuration system
    
    This should be called during application startup to migrate
    environment variables to encrypted storage and set up secure
    configuration management.
    """
    return await secure_config_manager.initialize_secure_configuration()


async def get_secure_configuration_status() -> Dict[str, Any]:
    """
    Get the current status of secure configuration
    """
    return await secure_config_manager.get_secure_configuration_status()


# Add secure secret management helper methods to SecureConfigManager class
class SecureConfigManagerExtensions:
    """Extension methods for secure secret management"""
    
    async def _update_system_secret(self, secret_key: str, secret_value: str) -> bool:
        """Update system secret in secure configuration"""
        try:
            # Store in encrypted configuration
            encrypted_secret = self.fernet.encrypt(secret_value.encode())
            
            # Create or update secure secrets file
            secrets_file = self._get_config_file_path("secrets")
            secrets_data = {}
            
            if os.path.exists(secrets_file):
                try:
                    with open(secrets_file, 'r') as f:
                        secrets_data = json.load(f)
                except:
                    secrets_data = {}
            
            secrets_data[secret_key] = {
                "value": encrypted_secret.decode(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "key_length": len(secret_value),
                "algorithm": "fernet"
            }
            
            with open(secrets_file, 'w') as f:
                json.dump(secrets_data, f, indent=2)
            
            logger.info("System secret updated in secure configuration",
                       secret_key=secret_key)
            return True
            
        except Exception as e:
            logger.error("Failed to update system secret",
                        secret_key=secret_key,
                        error=str(e))
            return False
    
    async def _backup_secure_secret(self, secret_key: str, secret_value: str):
        """Create backup of secure secret"""
        try:
            backup_dir = Path("secure_backups")
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"{secret_key}_backup_{timestamp}.enc"
            
            # Encrypt and store backup
            encrypted_backup = self.fernet.encrypt(secret_value.encode())
            
            with open(backup_file, 'wb') as f:
                f.write(encrypted_backup)
            
            logger.info("Secure secret backup created",
                       secret_key=secret_key,
                       backup_file=str(backup_file))
            
        except Exception as e:
            logger.error("Failed to create secure secret backup",
                        secret_key=secret_key,
                        error=str(e))
    
    async def _update_runtime_configuration(self, config_key: str, config_value: str):
        """Update runtime configuration if possible"""
        try:
            # Attempt to update runtime settings
            from prsm.core.config import get_settings
            
            settings = get_settings()
            if hasattr(settings, config_key):
                # Update the setting if it's mutable
                setattr(settings, config_key, config_value)
                logger.info("Runtime configuration updated",
                           config_key=config_key)
            else:
                logger.warning("Runtime configuration key not found or not mutable",
                             config_key=config_key)
                
        except Exception as e:
            logger.warning("Failed to update runtime configuration",
                          config_key=config_key,
                          error=str(e))


# Extend SecureConfigManager with new methods
for method_name in dir(SecureConfigManagerExtensions):
    if not method_name.startswith('_') or method_name.startswith('_update') or method_name.startswith('_backup'):
        method = getattr(SecureConfigManagerExtensions, method_name)
        if callable(method):
            setattr(SecureConfigManager, method_name, method)