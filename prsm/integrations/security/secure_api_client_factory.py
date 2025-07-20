"""
Secure API Client Factory
=========================

Factory for creating secure API clients that use encrypted credential management
instead of direct environment variable access. Provides centralized credential
management and security validation for all external service integrations.
"""

import asyncio
import structlog
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Union
from enum import Enum

from ..config.credential_manager import CredentialManager, CredentialType, CredentialData
from ..models.integration_models import IntegrationPlatform
from ...core.config import settings
from .audit_logger import audit_logger

logger = structlog.get_logger(__name__)


class SecureClientType(str, Enum):
    """Types of secure API clients - OpenAI removed, NWTN uses Claude only"""
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    GITHUB = "github"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    OLLAMA = "ollama"


class SecureAPIClientFactory:
    """
    Factory for creating API clients with secure credential management
    
    Features:
    - Uses encrypted credential manager instead of environment variables
    - Automatic credential validation and health checking
    - Audit logging for all credential access
    - Fallback strategies for credential failures
    - Automatic credential rotation support
    """
    
    def __init__(self):
        self.credential_manager = CredentialManager()
        self._client_cache: Dict[str, Any] = {}
        self._last_validation: Dict[str, datetime] = {}
        self.validation_interval = timedelta(hours=1)  # Validate credentials every hour
        
    async def get_secure_client(
        self,
        client_type: SecureClientType,
        user_id: str,
        force_refresh: bool = False
    ) -> Optional[Any]:
        """
        Get a secure API client with encrypted credential management
        
        Args:
            client_type: Type of client to create
            user_id: User ID for credential isolation
            force_refresh: Force credential refresh even if cached
            
        Returns:
            Configured API client instance or None if credentials unavailable
        """
        try:
            cache_key = f"{client_type}:{user_id}"
            
            # Check if we need to refresh credentials
            if not force_refresh and cache_key in self._client_cache:
                last_validation = self._last_validation.get(cache_key, datetime.min.replace(tzinfo=timezone.utc))
                if datetime.now(timezone.utc) - last_validation < self.validation_interval:
                    return self._client_cache[cache_key]
            
            # Get credentials from secure storage
            credentials = await self._get_secure_credentials(client_type, user_id)
            if not credentials:
                await self._log_credential_event(
                    "credential_not_found",
                    client_type,
                    user_id,
                    {"error": "No credentials available"}
                )
                return None
            
            # Validate credentials are not expired
            if not await self._validate_credentials(credentials, client_type):
                await self._log_credential_event(
                    "credential_validation_failed",
                    client_type,
                    user_id,
                    {"error": "Credential validation failed"}
                )
                return None
            
            # Create secure client
            client = await self._create_client(client_type, credentials)
            if client:
                # Cache the client and update validation time
                self._client_cache[cache_key] = client
                self._last_validation[cache_key] = datetime.now(timezone.utc)
                
                await self._log_credential_event(
                    "secure_client_created",
                    client_type,
                    user_id,
                    {"success": True}
                )
            
            return client
            
        except Exception as e:
            logger.error("Failed to create secure API client",
                        client_type=client_type,
                        user_id=user_id,
                        error=str(e))
            
            await self._log_credential_event(
                "secure_client_creation_failed",
                client_type,
                user_id,
                {"error": str(e)}
            )
            return None
    
    async def get_system_client(
        self,
        client_type: SecureClientType,
        force_refresh: bool = False
    ) -> Optional[Any]:
        """
        Get a system-level API client for background operations
        
        Uses system credentials instead of user-specific ones.
        Useful for background tasks, system operations, etc.
        """
        return await self.get_secure_client(
            client_type, 
            "system",  # System user ID
            force_refresh
        )
    
    async def validate_client_credentials(
        self,
        client_type: SecureClientType,
        user_id: str
    ) -> bool:
        """
        Validate that credentials for a client type are available and valid
        
        Args:
            client_type: Type of client to validate
            user_id: User ID for credential lookup
            
        Returns:
            True if credentials are valid, False otherwise
        """
        try:
            credentials = await self._get_secure_credentials(client_type, user_id)
            if not credentials:
                return False
            
            return await self._validate_credentials(credentials, client_type)
            
        except Exception as e:
            logger.error("Failed to validate client credentials",
                        client_type=client_type,
                        user_id=user_id,
                        error=str(e))
            return False
    
    async def register_user_credentials(
        self,
        client_type: SecureClientType,
        user_id: str,
        credentials: Dict[str, Any],
        expires_at: Optional[datetime] = None
    ) -> bool:
        """
        Register new credentials for a user and client type
        
        Args:
            client_type: Type of client
            user_id: User ID
            credentials: Credential data (API keys, tokens, etc.)
            expires_at: Optional expiration time
            
        Returns:
            True if registration succeeded, False otherwise
        """
        try:
            platform = self._get_platform_for_client_type(client_type)
            credential_type = self._get_credential_type_for_client_type(client_type)
            
            # Convert credentials dict to CredentialData
            credential_data = CredentialData(**credentials)
            
            # Calculate expires_in_days if expires_at is provided
            expires_in_days = None
            if expires_at:
                days_until_expiry = (expires_at - datetime.now(timezone.utc)).days
                expires_in_days = max(1, days_until_expiry)  # At least 1 day
            
            # Store credentials securely
            credential_id = self.credential_manager.store_credential(
                user_id=user_id,
                platform=platform,
                credential_data=credential_data,
                credential_type=credential_type,
                expires_in_days=expires_in_days
            )
            success = bool(credential_id)
            
            if success:
                # Clear cache to force refresh
                cache_key = f"{client_type}:{user_id}"
                self._client_cache.pop(cache_key, None)
                self._last_validation.pop(cache_key, None)
                
                await self._log_credential_event(
                    "credentials_registered",
                    client_type,
                    user_id,
                    {"success": True}
                )
            
            return success
            
        except Exception as e:
            logger.error("Failed to register user credentials",
                        client_type=client_type,
                        user_id=user_id,
                        error=str(e))
            
            await self._log_credential_event(
                "credential_registration_failed",
                client_type,
                user_id,
                {"error": str(e)}
            )
            return False
    
    async def rotate_credentials(
        self,
        client_type: SecureClientType,
        user_id: str
    ) -> bool:
        """
        Rotate credentials for a client type (where supported)
        
        Args:
            client_type: Type of client
            user_id: User ID
            
        Returns:
            True if rotation succeeded, False otherwise
        """
        try:
            # This would implement credential rotation logic for each platform
            # For now, log the rotation attempt
            await self._log_credential_event(
                "credential_rotation_requested",
                client_type,
                user_id,
                {"status": "not_implemented"}
            )
            
            logger.info("Credential rotation requested",
                       client_type=client_type,
                       user_id=user_id)
            
            # TODO: Implement platform-specific rotation logic
            return False
            
        except Exception as e:
            logger.error("Failed to rotate credentials",
                        client_type=client_type,
                        user_id=user_id,
                        error=str(e))
            return False
    
    async def _get_secure_credentials(
        self,
        client_type: SecureClientType,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get credentials from secure storage"""
        try:
            platform = self._get_platform_for_client_type(client_type)
            
            # Try user-specific credentials first
            credentials = self.credential_manager.get_credential(user_id, platform)
            
            # Fall back to system credentials if user credentials not available
            if not credentials and user_id != "system":
                credentials = self.credential_manager.get_credential("system", platform)
                
                if credentials:
                    logger.debug("Using system credentials as fallback",
                               client_type=client_type,
                               user_id=user_id)
            
            # Convert CredentialData object to dictionary if needed
            if credentials:
                if hasattr(credentials, 'model_dump'):
                    # Pydantic model - convert to dict
                    cred_dict = credentials.model_dump()
                elif hasattr(credentials, 'dict'):
                    # Legacy Pydantic - convert to dict
                    cred_dict = credentials.dict()
                else:
                    # Already a dict
                    cred_dict = credentials
                
                # Filter out None values and convert SecretStr to string
                result = {}
                for key, value in cred_dict.items():
                    if value is not None:
                        # Handle SecretStr objects
                        if hasattr(value, 'get_secret_value'):
                            result[key] = value.get_secret_value()
                        else:
                            result[key] = value
                
                return result
            
            return None
            
        except Exception as e:
            logger.error("Failed to get secure credentials",
                        client_type=client_type,
                        user_id=user_id,
                        error=str(e))
            return None
    
    async def _validate_credentials(
        self,
        credentials: Dict[str, Any],
        client_type: SecureClientType
    ) -> bool:
        """Validate that credentials are still valid"""
        try:
            # Check if credentials have required fields
            required_fields = self._get_required_fields_for_client_type(client_type)
            
            for field in required_fields:
                if field not in credentials:
                    logger.warning("Missing required credential field",
                                  client_type=client_type,
                                  field=field)
                    return False
            
            # TODO: Add platform-specific validation (API calls to test credentials)
            # For now, just check basic structure
            return True
            
        except Exception as e:
            logger.error("Failed to validate credentials",
                        client_type=client_type,
                        error=str(e))
            return False
    
    async def _create_client(
        self,
        client_type: SecureClientType,
        credentials: Dict[str, Any]
    ) -> Optional[Any]:
        """Create API client with secured credentials"""
        try:
            if client_type == SecureClientType.ANTHROPIC:
                return await self._create_anthropic_client(credentials)
            elif client_type == SecureClientType.HUGGINGFACE:
                return await self._create_huggingface_client(credentials)
            elif client_type == SecureClientType.GITHUB:
                return await self._create_github_client(credentials)
            elif client_type == SecureClientType.PINECONE:
                return await self._create_pinecone_client(credentials)
            elif client_type == SecureClientType.WEAVIATE:
                return await self._create_weaviate_client(credentials)
            elif client_type == SecureClientType.OLLAMA:
                return await self._create_ollama_client(credentials)
            else:
                logger.error("Unknown client type", client_type=client_type)
                return None
                
        except Exception as e:
            logger.error("Failed to create API client",
                        client_type=client_type,
                        error=str(e))
            return None
    
    async def _create_openai_client(self, credentials: Dict[str, Any]) -> Optional[Any]:
        """Create OpenAI client with secure credentials"""
        try:
            # Import OpenAI client dynamically to avoid import errors if not installed
            import openai
            
            api_key = credentials.get("api_key")
            if not api_key:
                return None
            
            # Create OpenAI client with secure credentials
            client = openai.OpenAI(api_key=api_key)
            
            logger.debug("OpenAI client created successfully")
            return client
            
        except ImportError:
            logger.warning("OpenAI library not installed")
            return None
        except Exception as e:
            logger.error("Failed to create OpenAI client", error=str(e))
            return None
    
    async def _create_anthropic_client(self, credentials: Dict[str, Any]) -> Optional[Any]:
        """Create Anthropic client with secure credentials"""
        try:
            # Import Anthropic client dynamically
            import anthropic
            
            api_key = credentials.get("api_key")
            if not api_key:
                return None
            
            # Create Anthropic client with secure credentials
            client = anthropic.Anthropic(api_key=api_key)
            
            logger.debug("Anthropic client created successfully")
            return client
            
        except ImportError:
            logger.warning("Anthropic library not installed")
            return None
        except Exception as e:
            logger.error("Failed to create Anthropic client", error=str(e))
            return None
    
    async def _create_huggingface_client(self, credentials: Dict[str, Any]) -> Optional[Any]:
        """Create Hugging Face client with secure credentials"""
        try:
            # Create a simple client object for HuggingFace
            # This can be expanded based on actual HuggingFace client needs
            api_token = credentials.get("api_token") or credentials.get("api_key")
            if not api_token:
                return None
            
            # Return client configuration for HuggingFace
            client_config = {
                "api_token": api_token,
                "base_url": "https://api-inference.huggingface.co"
            }
            
            logger.debug("HuggingFace client configuration created successfully")
            return client_config
            
        except Exception as e:
            logger.error("Failed to create HuggingFace client", error=str(e))
            return None
    
    async def _create_github_client(self, credentials: Dict[str, Any]) -> Optional[Any]:
        """Create GitHub client with secure credentials"""
        try:
            # Create GitHub client configuration
            access_token = credentials.get("access_token")
            if not access_token:
                return None
            
            # Return client configuration for GitHub
            client_config = {
                "access_token": access_token,
                "base_url": "https://api.github.com"
            }
            
            logger.debug("GitHub client configuration created successfully")
            return client_config
            
        except Exception as e:
            logger.error("Failed to create GitHub client", error=str(e))
            return None
    
    async def _create_pinecone_client(self, credentials: Dict[str, Any]) -> Optional[Any]:
        """Create Pinecone client with secure credentials"""
        try:
            api_key = credentials.get("api_key")
            environment = credentials.get("environment", "us-west1-gcp")
            
            if not api_key:
                return None
            
            # Return client configuration for Pinecone
            client_config = {
                "api_key": api_key,
                "environment": environment
            }
            
            logger.debug("Pinecone client configuration created successfully")
            return client_config
            
        except Exception as e:
            logger.error("Failed to create Pinecone client", error=str(e))
            return None
    
    async def _create_weaviate_client(self, credentials: Dict[str, Any]) -> Optional[Any]:
        """Create Weaviate client with secure credentials"""
        try:
            url = credentials.get("url", "http://localhost:8080")
            api_key = credentials.get("api_key")
            
            # Return client configuration for Weaviate
            client_config = {
                "url": url,
                "api_key": api_key
            }
            
            logger.debug("Weaviate client configuration created successfully")
            return client_config
            
        except Exception as e:
            logger.error("Failed to create Weaviate client", error=str(e))
            return None
    
    async def _create_ollama_client(self, credentials: Dict[str, Any]) -> Optional[Any]:
        """Create Ollama client with secure credentials"""
        try:
            base_url = credentials.get("base_url", "http://localhost:11434")
            api_key = credentials.get("api_key")  # Optional for Ollama
            
            # Return client configuration for Ollama
            client_config = {
                "base_url": base_url,
                "api_key": api_key
            }
            
            logger.debug("Ollama client configuration created successfully")
            return client_config
            
        except Exception as e:
            logger.error("Failed to create Ollama client", error=str(e))
            return None
    
    def _get_platform_for_client_type(self, client_type: SecureClientType) -> IntegrationPlatform:
        """Map client type to integration platform"""
        mapping = {
            SecureClientType.ANTHROPIC: IntegrationPlatform.ANTHROPIC,
            SecureClientType.HUGGINGFACE: IntegrationPlatform.HUGGINGFACE,
            SecureClientType.GITHUB: IntegrationPlatform.GITHUB,
            SecureClientType.PINECONE: IntegrationPlatform.CUSTOM,
            SecureClientType.WEAVIATE: IntegrationPlatform.CUSTOM,
            SecureClientType.OLLAMA: IntegrationPlatform.CUSTOM
        }
        return mapping.get(client_type, IntegrationPlatform.CUSTOM)
    
    def _get_credential_type_for_client_type(self, client_type: SecureClientType) -> str:
        """Map client type to credential type"""
        if client_type == SecureClientType.GITHUB:
            return CredentialType.OAUTH_TOKEN
        else:
            return CredentialType.API_KEY
    
    def _get_required_fields_for_client_type(self, client_type: SecureClientType) -> list[str]:
        """Get required credential fields for client type"""
        if client_type == SecureClientType.GITHUB:
            return ["access_token"]
        elif client_type in [SecureClientType.PINECONE]:
            return ["api_key", "environment"]
        elif client_type in [SecureClientType.WEAVIATE, SecureClientType.OLLAMA]:
            return ["url"]
        else:
            return ["api_key"]
    
    async def _log_credential_event(
        self,
        event_type: str,
        client_type: SecureClientType,
        user_id: str,
        details: Dict[str, Any]
    ):
        """Log credential-related security events"""
        try:
            await audit_logger.log_security_event(
                event_type=f"credential_{event_type}",
                user_id=user_id,
                details={
                    "client_type": client_type,
                    **details
                },
                security_level="info" if details.get("success") else "warning"
            )
        except Exception as e:
            logger.error("Failed to log credential event", error=str(e))


# Global secure client factory instance
secure_client_factory = SecureAPIClientFactory()


async def get_secure_api_client(
    client_type: SecureClientType,
    user_id: str,
    force_refresh: bool = False
) -> Optional[Any]:
    """
    Convenience function to get a secure API client
    
    Usage:
        openai_client = await get_secure_api_client(SecureClientType.OPENAI, user_id)
        if openai_client:
            response = await openai_client.chat.completions.create(...)
    """
    return await secure_client_factory.get_secure_client(client_type, user_id, force_refresh)


async def get_system_api_client(
    client_type: SecureClientType,
    force_refresh: bool = False
) -> Optional[Any]:
    """
    Convenience function to get a system-level API client
    
    Usage:
        openai_client = await get_system_api_client(SecureClientType.OPENAI)
    """
    return await secure_client_factory.get_system_client(client_type, force_refresh)