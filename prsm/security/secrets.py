"""
Secrets Management

Secure secrets management for PRSM.
Supports multiple backends: environment variables, HashiCorp Vault, AWS Secrets Manager, GCP Secret Manager.
"""

import os
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import structlog

logger = structlog.get_logger(__name__)


class SecretBackend(Enum):
    """Supported secret backends"""
    ENV = "env"  # Environment variables
    FILE = "file"  # File-based secrets
    VAULT = "vault"  # HashiCorp Vault
    AWS_SECRETS = "aws_secrets"  # AWS Secrets Manager
    GCP_SECRETS = "gcp_secrets"  # GCP Secret Manager
    AZURE_KEYVAULT = "azure_keyvault"  # Azure Key Vault


@dataclass
class SecretMetadata:
    """
    Metadata for a secret.
    
    Attributes:
        key: Secret key/name
        backend: Backend storing the secret
        created_at: When the secret was created
        updated_at: When the secret was last updated
        expires_at: When the secret expires (if applicable)
        version: Secret version (if applicable)
        rotation_enabled: Whether rotation is enabled
        last_rotation: When the secret was last rotated
        tags: Additional metadata tags
    """
    key: str
    backend: SecretBackend
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    version: Optional[str] = None
    rotation_enabled: bool = False
    last_rotation: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "key": self.key,
            "backend": self.backend.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "version": self.version,
            "rotation_enabled": self.rotation_enabled,
            "last_rotation": self.last_rotation.isoformat() if self.last_rotation else None,
            "tags": self.tags,
        }


@dataclass
class SecretValue:
    """
    Container for a secret value with metadata.
    
    Attributes:
        value: The secret value
        metadata: Secret metadata
        checksum: SHA256 checksum of the value
    """
    value: str
    metadata: SecretMetadata
    checksum: str = ""
    
    def __post_init__(self):
        """Calculate checksum after initialization"""
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate SHA256 checksum of the value"""
        return hashlib.sha256(self.value.encode()).hexdigest()[:16]
    
    def is_expired(self) -> bool:
        """Check if the secret is expired"""
        if self.metadata.expires_at:
            return datetime.now(timezone.utc) > self.metadata.expires_at
        return False
    
    def needs_rotation(self, max_age_days: int = 90) -> bool:
        """Check if the secret needs rotation"""
        if not self.metadata.last_rotation:
            return True
        
        rotation_age = datetime.now(timezone.utc) - self.metadata.last_rotation
        return rotation_age > timedelta(days=max_age_days)


class SecretBackendInterface(ABC):
    """Abstract interface for secret backends"""
    
    @abstractmethod
    async def get_secret(self, key: str) -> Optional[str]:
        """Get a secret value"""
        pass
    
    @abstractmethod
    async def set_secret(self, key: str, value: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Set a secret value"""
        pass
    
    @abstractmethod
    async def delete_secret(self, key: str) -> bool:
        """Delete a secret"""
        pass
    
    @abstractmethod
    async def list_secrets(self) -> List[str]:
        """List all secret keys"""
        pass
    
    @abstractmethod
    async def rotate_secret(self, key: str) -> Optional[str]:
        """Rotate a secret"""
        pass


class EnvironmentBackend(SecretBackendInterface):
    """Environment variable-based secret backend"""
    
    def __init__(self, prefix: str = ""):
        """
        Initialize environment backend.
        
        Args:
            prefix: Prefix for environment variables
        """
        self.prefix = prefix
        self._cache: Dict[str, SecretValue] = {}
    
    async def get_secret(self, key: str) -> Optional[str]:
        """Get a secret from environment variables"""
        env_key = f"{self.prefix}{key}" if self.prefix else key
        value = os.getenv(env_key)
        
        if value:
            metadata = SecretMetadata(
                key=key,
                backend=SecretBackend.ENV,
            )
            self._cache[key] = SecretValue(value=value, metadata=metadata)
            return value
        
        return None
    
    async def set_secret(self, key: str, value: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Set a secret in environment (not recommended for production)"""
        # Environment variables cannot be set at runtime in a portable way
        # This is primarily for testing
        env_key = f"{self.prefix}{key}" if self.prefix else key
        os.environ[env_key] = value
        
        metadata_obj = SecretMetadata(
            key=key,
            backend=SecretBackend.ENV,
        )
        self._cache[key] = SecretValue(value=value, metadata=metadata_obj)
        
        logger.warning("Secret set in environment - not recommended for production", key=key)
        return True
    
    async def delete_secret(self, key: str) -> bool:
        """Delete a secret from environment"""
        env_key = f"{self.prefix}{key}" if self.prefix else key
        
        if env_key in os.environ:
            del os.environ[env_key]
            if key in self._cache:
                del self._cache[key]
            return True
        
        return False
    
    async def list_secrets(self) -> List[str]:
        """List all secret keys"""
        if self.prefix:
            return [k[len(self.prefix):] for k in os.environ if k.startswith(self.prefix)]
        return list(os.environ.keys())
    
    async def rotate_secret(self, key: str) -> Optional[str]:
        """Rotate a secret (not supported for environment backend)"""
        logger.warning("Secret rotation not supported for environment backend", key=key)
        return None


class FileBackend(SecretBackendInterface):
    """File-based secret backend"""
    
    def __init__(self, secrets_dir: str = "/run/secrets"):
        """
        Initialize file backend.
        
        Args:
            secrets_dir: Directory containing secret files
        """
        self.secrets_dir = secrets_dir
        self._cache: Dict[str, SecretValue] = {}
    
    async def get_secret(self, key: str) -> Optional[str]:
        """Get a secret from a file"""
        import pathlib
        
        secret_path = pathlib.Path(self.secrets_dir) / key
        
        try:
            if secret_path.exists():
                value = secret_path.read_text().strip()
                metadata = SecretMetadata(
                    key=key,
                    backend=SecretBackend.FILE,
                )
                self._cache[key] = SecretValue(value=value, metadata=metadata)
                return value
        except Exception as e:
            logger.error("Failed to read secret file", key=key, error=str(e))
        
        return None
    
    async def set_secret(self, key: str, value: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Set a secret in a file"""
        import pathlib
        
        secret_path = pathlib.Path(self.secrets_dir) / key
        
        try:
            secret_path.parent.mkdir(parents=True, exist_ok=True)
            secret_path.write_text(value)
            secret_path.chmod(0o600)  # Owner read/write only
            
            metadata_obj = SecretMetadata(
                key=key,
                backend=SecretBackend.FILE,
            )
            self._cache[key] = SecretValue(value=value, metadata=metadata_obj)
            
            return True
        except Exception as e:
            logger.error("Failed to write secret file", key=key, error=str(e))
            return False
    
    async def delete_secret(self, key: str) -> bool:
        """Delete a secret file"""
        import pathlib
        
        secret_path = pathlib.Path(self.secrets_dir) / key
        
        try:
            if secret_path.exists():
                secret_path.unlink()
                if key in self._cache:
                    del self._cache[key]
                return True
        except Exception as e:
            logger.error("Failed to delete secret file", key=key, error=str(e))
        
        return False
    
    async def list_secrets(self) -> List[str]:
        """List all secret files"""
        import pathlib
        
        secrets_dir = pathlib.Path(self.secrets_dir)
        
        if secrets_dir.exists():
            return [f.name for f in secrets_dir.iterdir() if f.is_file()]
        
        return []
    
    async def rotate_secret(self, key: str) -> Optional[str]:
        """Rotate a secret (requires external rotation mechanism)"""
        logger.warning("Secret rotation requires external mechanism for file backend", key=key)
        return None


class VaultBackend(SecretBackendInterface):
    """HashiCorp Vault secret backend"""
    
    def __init__(
        self,
        url: str = "http://localhost:8200",
        token: Optional[str] = None,
        mount_point: str = "secret",
    ):
        """
        Initialize Vault backend.
        
        Args:
            url: Vault server URL
            token: Vault token
            mount_point: Secret engine mount point
        """
        self.url = url
        self.token = token or os.getenv("VAULT_TOKEN")
        self.mount_point = mount_point
        self._client = None
        self._cache: Dict[str, SecretValue] = {}
    
    async def _get_client(self):
        """Get or create Vault client"""
        if self._client is None:
            try:
                import hvac
                
                self._client = hvac.Client(url=self.url, token=self.token)
                
                if not self._client.is_authenticated():
                    raise Exception("Vault authentication failed")
            except ImportError:
                raise Exception("hvac package not installed. Install with: pip install hvac")
        
        return self._client
    
    async def get_secret(self, key: str) -> Optional[str]:
        """Get a secret from Vault"""
        try:
            client = await self._get_client()
            
            response = client.secrets.kv.v2.read_secret_version(
                path=key,
                mount_point=self.mount_point,
            )
            
            value = response.get("data", {}).get("data", {}).get("value")
            
            if value:
                metadata = SecretMetadata(
                    key=key,
                    backend=SecretBackend.VAULT,
                    version=response.get("data", {}).get("metadata", {}).get("version"),
                )
                self._cache[key] = SecretValue(value=value, metadata=metadata)
                return value
        except Exception as e:
            logger.error("Failed to get secret from Vault", key=key, error=str(e))
        
        return None
    
    async def set_secret(self, key: str, value: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Set a secret in Vault"""
        try:
            client = await self._get_client()
            
            client.secrets.kv.v2.create_or_update_secret(
                path=key,
                secret={"value": value},
                mount_point=self.mount_point,
            )
            
            metadata_obj = SecretMetadata(
                key=key,
                backend=SecretBackend.VAULT,
            )
            self._cache[key] = SecretValue(value=value, metadata=metadata_obj)
            
            return True
        except Exception as e:
            logger.error("Failed to set secret in Vault", key=key, error=str(e))
            return False
    
    async def delete_secret(self, key: str) -> bool:
        """Delete a secret from Vault"""
        try:
            client = await self._get_client()
            
            client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=key,
                mount_point=self.mount_point,
            )
            
            if key in self._cache:
                del self._cache[key]
            
            return True
        except Exception as e:
            logger.error("Failed to delete secret from Vault", key=key, error=str(e))
            return False
    
    async def list_secrets(self) -> List[str]:
        """List all secrets in Vault"""
        try:
            client = await self._get_client()
            
            response = client.secrets.kv.v2.list_secrets(
                mount_point=self.mount_point,
            )
            
            return response.get("data", {}).get("keys", [])
        except Exception as e:
            logger.error("Failed to list secrets from Vault", error=str(e))
            return []
    
    async def rotate_secret(self, key: str) -> Optional[str]:
        """Rotate a secret in Vault"""
        # Generate a new random secret
        import secrets
        import string
        
        new_value = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))
        
        if await self.set_secret(key, new_value):
            return new_value
        
        return None


class AWSSecretsBackend(SecretBackendInterface):
    """AWS Secrets Manager backend"""
    
    def __init__(self, region: str = "us-east-1"):
        """
        Initialize AWS Secrets Manager backend.
        
        Args:
            region: AWS region
        """
        self.region = region
        self._client = None
        self._cache: Dict[str, SecretValue] = {}
    
    async def _get_client(self):
        """Get or create AWS Secrets Manager client"""
        if self._client is None:
            try:
                import boto3
                
                self._client = boto3.client("secretsmanager", region_name=self.region)
            except ImportError:
                raise Exception("boto3 package not installed. Install with: pip install boto3")
        
        return self._client
    
    async def get_secret(self, key: str) -> Optional[str]:
        """Get a secret from AWS Secrets Manager"""
        try:
            client = await self._get_client()
            
            response = client.get_secret_value(SecretId=key)
            
            value = response.get("SecretString")
            
            if value:
                metadata = SecretMetadata(
                    key=key,
                    backend=SecretBackend.AWS_SECRETS,
                    version=response.get("VersionId"),
                )
                self._cache[key] = SecretValue(value=value, metadata=metadata)
                return value
        except Exception as e:
            logger.error("Failed to get secret from AWS", key=key, error=str(e))
        
        return None
    
    async def set_secret(self, key: str, value: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Set a secret in AWS Secrets Manager"""
        try:
            client = await self._get_client()
            
            try:
                client.create_secret(Name=key, SecretString=value)
            except client.exceptions.ResourceExistsException:
                client.put_secret_value(SecretId=key, SecretString=value)
            
            metadata_obj = SecretMetadata(
                key=key,
                backend=SecretBackend.AWS_SECRETS,
            )
            self._cache[key] = SecretValue(value=value, metadata=metadata_obj)
            
            return True
        except Exception as e:
            logger.error("Failed to set secret in AWS", key=key, error=str(e))
            return False
    
    async def delete_secret(self, key: str) -> bool:
        """Delete a secret from AWS Secrets Manager"""
        try:
            client = await self._get_client()
            
            client.delete_secret(
                SecretId=key,
                ForceDeleteWithoutRecovery=True,
            )
            
            if key in self._cache:
                del self._cache[key]
            
            return True
        except Exception as e:
            logger.error("Failed to delete secret from AWS", key=key, error=str(e))
            return False
    
    async def list_secrets(self) -> List[str]:
        """List all secrets in AWS Secrets Manager"""
        try:
            client = await self._get_client()
            
            secrets = []
            response = client.list_secrets()
            
            for secret in response.get("SecretList", []):
                secrets.append(secret["Name"])
            
            return secrets
        except Exception as e:
            logger.error("Failed to list secrets from AWS", error=str(e))
            return []
    
    async def rotate_secret(self, key: str) -> Optional[str]:
        """Rotate a secret in AWS Secrets Manager"""
        # Generate a new random secret
        import secrets
        import string
        
        new_value = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))
        
        if await self.set_secret(key, new_value):
            return new_value
        
        return None


class GCPSecretsBackend(SecretBackendInterface):
    """GCP Secret Manager backend"""
    
    def __init__(self, project_id: str = ""):
        """
        Initialize GCP Secret Manager backend.
        
        Args:
            project_id: GCP project ID
        """
        self.project_id = project_id or os.getenv("GCP_PROJECT", "")
        self._client = None
        self._cache: Dict[str, SecretValue] = {}
    
    async def _get_client(self):
        """Get or create GCP Secret Manager client"""
        if self._client is None:
            try:
                from google.cloud import secretmanager
                
                self._client = secretmanager.SecretManagerServiceClient()
            except ImportError:
                raise Exception("google-cloud-secret-manager package not installed. Install with: pip install google-cloud-secret-manager")
        
        return self._client
    
    async def get_secret(self, key: str) -> Optional[str]:
        """Get a secret from GCP Secret Manager"""
        try:
            client = await self._get_client()
            
            name = f"projects/{self.project_id}/secrets/{key}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            
            value = response.payload.data.decode("UTF-8")
            
            if value:
                metadata = SecretMetadata(
                    key=key,
                    backend=SecretBackend.GCP_SECRETS,
                )
                self._cache[key] = SecretValue(value=value, metadata=metadata)
                return value
        except Exception as e:
            logger.error("Failed to get secret from GCP", key=key, error=str(e))
        
        return None
    
    async def set_secret(self, key: str, value: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Set a secret in GCP Secret Manager"""
        try:
            client = await self._get_client()
            
            parent = f"projects/{self.project_id}"
            
            # Create secret if it doesn't exist
            try:
                client.create_secret(
                    request={
                        "parent": parent,
                        "secret_id": key,
                        "secret": {"replication": {"automatic": {}}},
                    }
                )
            except Exception:
                pass  # Secret already exists
            
            # Add secret version
            client.add_secret_version(
                request={
                    "parent": f"projects/{self.project_id}/secrets/{key}",
                    "payload": {"data": value.encode("UTF-8")},
                }
            )
            
            metadata_obj = SecretMetadata(
                key=key,
                backend=SecretBackend.GCP_SECRETS,
            )
            self._cache[key] = SecretValue(value=value, metadata=metadata_obj)
            
            return True
        except Exception as e:
            logger.error("Failed to set secret in GCP", key=key, error=str(e))
            return False
    
    async def delete_secret(self, key: str) -> bool:
        """Delete a secret from GCP Secret Manager"""
        try:
            client = await self._get_client()
            
            name = f"projects/{self.project_id}/secrets/{key}"
            client.delete_secret(request={"name": name})
            
            if key in self._cache:
                del self._cache[key]
            
            return True
        except Exception as e:
            logger.error("Failed to delete secret from GCP", key=key, error=str(e))
            return False
    
    async def list_secrets(self) -> List[str]:
        """List all secrets in GCP Secret Manager"""
        try:
            client = await self._get_client()
            
            parent = f"projects/{self.project_id}"
            response = client.list_secrets(request={"parent": parent})
            
            return [secret.name.split("/")[-1] for secret in response]
        except Exception as e:
            logger.error("Failed to list secrets from GCP", error=str(e))
            return []
    
    async def rotate_secret(self, key: str) -> Optional[str]:
        """Rotate a secret in GCP Secret Manager"""
        # Generate a new random secret
        import secrets
        import string
        
        new_value = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))
        
        if await self.set_secret(key, new_value):
            return new_value
        
        return None


class SecretsManager:
    """
    Secure secrets management for PRSM.
    
    Provides a unified interface for secrets management across multiple backends:
    - Environment variables (development)
    - File-based secrets (Docker secrets)
    - HashiCorp Vault (production)
    - AWS Secrets Manager (AWS deployments)
    - GCP Secret Manager (GCP deployments)
    """
    
    # Required secrets for PRSM
    REQUIRED_SECRETS = [
        "JWT_SECRET_KEY",
        "DATABASE_URL",
        "ENCRYPTION_KEY",
    ]
    
    # Recommended secrets
    RECOMMENDED_SECRETS = [
        "REDIS_URL",
        "API_KEY",
        "ADMIN_PASSWORD",
        "SMTP_PASSWORD",
    ]
    
    def __init__(
        self,
        backend: SecretBackend = SecretBackend.ENV,
        **kwargs,
    ):
        """
        Initialize the secrets manager.
        
        Args:
            backend: Secret backend to use
            **kwargs: Backend-specific configuration
        """
        self.backend = backend
        self._backend_impl: Optional[SecretBackendInterface] = None
        self._kwargs = kwargs
        self._cache: Dict[str, SecretValue] = {}
        self._rotation_callbacks: Dict[str, Callable] = {}
    
    async def initialize(self) -> None:
        """Initialize the secrets manager"""
        self._backend_impl = await self._create_backend()
        logger.info("Secrets manager initialized", backend=self.backend.value)
    
    async def _create_backend(self) -> SecretBackendInterface:
        """Create the appropriate backend implementation"""
        if self.backend == SecretBackend.ENV:
            return EnvironmentBackend(prefix=self._kwargs.get("prefix", ""))
        elif self.backend == SecretBackend.FILE:
            return FileBackend(secrets_dir=self._kwargs.get("secrets_dir", "/run/secrets"))
        elif self.backend == SecretBackend.VAULT:
            return VaultBackend(
                url=self._kwargs.get("url", os.getenv("VAULT_ADDR", "http://localhost:8200")),
                token=self._kwargs.get("token"),
                mount_point=self._kwargs.get("mount_point", "secret"),
            )
        elif self.backend == SecretBackend.AWS_SECRETS:
            return AWSSecretsBackend(
                region=self._kwargs.get("region", os.getenv("AWS_REGION", "us-east-1"))
            )
        elif self.backend == SecretBackend.GCP_SECRETS:
            return GCPSecretsBackend(
                project_id=self._kwargs.get("project_id", os.getenv("GCP_PROJECT", ""))
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _ensure_backend(self) -> SecretBackendInterface:
        """Ensure backend is initialized"""
        if self._backend_impl is None:
            raise RuntimeError("Secrets manager not initialized. Call initialize() first.")
        return self._backend_impl
    
    async def get_secret(self, key: str, use_cache: bool = True) -> Optional[str]:
        """
        Get a secret value.
        
        Args:
            key: Secret key
            use_cache: Whether to use cached value
            
        Returns:
            Secret value or None if not found
        """
        # Check cache first
        if use_cache and key in self._cache:
            cached = self._cache[key]
            if not cached.is_expired():
                return cached.value
        
        backend = self._ensure_backend()
        value = await backend.get_secret(key)
        
        if value:
            # Update cache
            if key in self._cache:
                self._cache[key].value = value
                self._cache[key].metadata.updated_at = datetime.now(timezone.utc)
            else:
                # Create metadata
                metadata = SecretMetadata(key=key, backend=self.backend)
                self._cache[key] = SecretValue(value=value, metadata=metadata)
        
        return value
    
    async def get_secret_or_raise(self, key: str) -> str:
        """
        Get a secret value or raise an exception.
        
        Args:
            key: Secret key
            
        Returns:
            Secret value
            
        Raises:
            ValueError: If secret is not found
        """
        value = await self.get_secret(key)
        if value is None:
            raise ValueError(f"Secret not found: {key}")
        return value
    
    async def set_secret(self, key: str, value: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Set a secret value.
        
        Args:
            key: Secret key
            value: Secret value
            metadata: Optional metadata
            
        Returns:
            True if successful
        """
        backend = self._ensure_backend()
        success = await backend.set_secret(key, value, metadata)
        
        if success:
            # Update cache
            if key in self._cache:
                self._cache[key].value = value
                self._cache[key].metadata.updated_at = datetime.now(timezone.utc)
            else:
                meta = SecretMetadata(key=key, backend=self.backend)
                self._cache[key] = SecretValue(value=value, metadata=meta)
            
            logger.info("Secret set", key=key)
        
        return success
    
    async def delete_secret(self, key: str) -> bool:
        """
        Delete a secret.
        
        Args:
            key: Secret key
            
        Returns:
            True if successful
        """
        backend = self._ensure_backend()
        success = await backend.delete_secret(key)
        
        if success and key in self._cache:
            del self._cache[key]
            logger.info("Secret deleted", key=key)
        
        return success
    
    async def list_secrets(self) -> List[str]:
        """
        List all secret keys.
        
        Returns:
            List of secret keys
        """
        backend = self._ensure_backend()
        return await backend.list_secrets()
    
    async def rotate_secret(self, key: str) -> Optional[str]:
        """
        Rotate a secret.
        
        Args:
            key: Secret key
            
        Returns:
            New secret value or None if rotation failed
        """
        backend = self._ensure_backend()
        new_value = await backend.rotate_secret(key)
        
        if new_value:
            # Update cache
            if key in self._cache:
                self._cache[key].value = new_value
                self._cache[key].metadata.last_rotation = datetime.now(timezone.utc)
            
            # Call rotation callback if registered
            if key in self._rotation_callbacks:
                try:
                    await self._rotation_callbacks[key](key, new_value)
                except Exception as e:
                    logger.error("Rotation callback failed", key=key, error=str(e))
            
            logger.info("Secret rotated", key=key)
        
        return new_value
    
    def register_rotation_callback(self, key: str, callback: Callable) -> None:
        """
        Register a callback to be called when a secret is rotated.
        
        Args:
            key: Secret key
            callback: Async callback function
        """
        self._rotation_callbacks[key] = callback
    
    def unregister_rotation_callback(self, key: str) -> None:
        """
        Unregister a rotation callback.
        
        Args:
            key: Secret key
        """
        if key in self._rotation_callbacks:
            del self._rotation_callbacks[key]
    
    async def validate_secrets(self) -> Dict[str, bool]:
        """
        Validate that all required secrets are set.
        
        Returns:
            Dictionary mapping secret keys to whether they are set
        """
        validation = {}
        
        for secret in self.REQUIRED_SECRETS:
            value = await self.get_secret(secret)
            validation[secret] = value is not None and len(value) > 0
        
        return validation
    
    async def check_secret_strength(self, key: str, value: Optional[str] = None) -> Dict[str, Any]:
        """
        Check the strength of a secret.
        
        Args:
            key: Secret key
            value: Secret value (if None, will retrieve from backend)
            
        Returns:
            Strength assessment
        """
        if value is None:
            value = await self.get_secret(key)
        
        if value is None:
            return {"key": key, "exists": False, "strength": "none"}
        
        assessment = {
            "key": key,
            "exists": True,
            "length": len(value),
            "has_uppercase": any(c.isupper() for c in value),
            "has_lowercase": any(c.islower() for c in value),
            "has_digits": any(c.isdigit() for c in value),
            "has_special": any(not c.isalnum() for c in value),
            "is_common": self._is_common_secret(value),
        }
        
        # Calculate strength score
        score = 0
        if assessment["length"] >= 8:
            score += 1
        if assessment["length"] >= 16:
            score += 1
        if assessment["length"] >= 32:
            score += 1
        if assessment["has_uppercase"]:
            score += 1
        if assessment["has_lowercase"]:
            score += 1
        if assessment["has_digits"]:
            score += 1
        if assessment["has_special"]:
            score += 1
        if not assessment["is_common"]:
            score += 1
        
        # Determine strength level
        if score >= 7:
            assessment["strength"] = "strong"
        elif score >= 5:
            assessment["strength"] = "medium"
        elif score >= 3:
            assessment["strength"] = "weak"
        else:
            assessment["strength"] = "very_weak"
        
        return assessment
    
    def _is_common_secret(self, value: str) -> bool:
        """Check if a secret is a common/weak value"""
        common_secrets = {
            "password",
            "secret",
            "admin",
            "root",
            "123456",
            "password123",
            "changeme",
            "default",
            "test",
        }
        
        return value.lower() in common_secrets
    
    async def get_required_secrets_status(self) -> Dict[str, Any]:
        """
        Get status of all required secrets.
        
        Returns:
            Status of required secrets
        """
        status = {
            "required": {},
            "recommended": {},
            "missing_required": [],
            "missing_recommended": [],
        }
        
        # Check required secrets
        for secret in self.REQUIRED_SECRETS:
            strength = await self.check_secret_strength(secret)
            status["required"][secret] = strength
            
            if not strength["exists"]:
                status["missing_required"].append(secret)
        
        # Check recommended secrets
        for secret in self.RECOMMENDED_SECRETS:
            strength = await self.check_secret_strength(secret)
            status["recommended"][secret] = strength
            
            if not strength["exists"]:
                status["missing_recommended"].append(secret)
        
        return status
    
    async def generate_secret(
        self,
        length: int = 32,
        include_uppercase: bool = True,
        include_lowercase: bool = True,
        include_digits: bool = True,
        include_special: bool = True,
    ) -> str:
        """
        Generate a random secret.
        
        Args:
            length: Secret length
            include_uppercase: Include uppercase letters
            include_lowercase: Include lowercase letters
            include_digits: Include digits
            include_special: Include special characters
            
        Returns:
            Generated secret
        """
        import secrets
        import string
        
        chars = ""
        
        if include_uppercase:
            chars += string.ascii_uppercase
        if include_lowercase:
            chars += string.ascii_lowercase
        if include_digits:
            chars += string.digits
        if include_special:
            chars += "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        if not chars:
            chars = string.ascii_letters + string.digits
        
        return ''.join(secrets.choice(chars) for _ in range(length))
    
    def get_cached_secret(self, key: str) -> Optional[str]:
        """
        Get a secret from cache only (no backend call).
        
        Args:
            key: Secret key
            
        Returns:
            Cached secret value or None
        """
        if key in self._cache:
            return self._cache[key].value
        return None
    
    def clear_cache(self) -> None:
        """Clear the secret cache"""
        self._cache.clear()
        logger.info("Secret cache cleared")
    
    def get_secret_metadata(self, key: str) -> Optional[SecretMetadata]:
        """
        Get metadata for a secret.
        
        Args:
            key: Secret key
            
        Returns:
            Secret metadata or None
        """
        if key in self._cache:
            return self._cache[key].metadata
        return None


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


async def get_secrets_manager() -> SecretsManager:
    """Get the global secrets manager instance"""
    global _secrets_manager
    
    if _secrets_manager is None:
        # Determine backend from environment
        backend_str = os.getenv("SECRETS_BACKEND", "env").lower()
        backend_map = {
            "env": SecretBackend.ENV,
            "file": SecretBackend.FILE,
            "vault": SecretBackend.VAULT,
            "aws": SecretBackend.AWS_SECRETS,
            "aws_secrets": SecretBackend.AWS_SECRETS,
            "gcp": SecretBackend.GCP_SECRETS,
            "gcp_secrets": SecretBackend.GCP_SECRETS,
        }
        backend = backend_map.get(backend_str, SecretBackend.ENV)
        
        _secrets_manager = SecretsManager(backend=backend)
        await _secrets_manager.initialize()
    
    return _secrets_manager


async def get_secret(key: str) -> Optional[str]:
    """
    Convenience function to get a secret.
    
    Args:
        key: Secret key
        
    Returns:
        Secret value or None
    """
    manager = await get_secrets_manager()
    return await manager.get_secret(key)


async def get_secret_or_raise(key: str) -> str:
    """
    Convenience function to get a secret or raise an exception.
    
    Args:
        key: Secret key
        
    Returns:
        Secret value
        
    Raises:
        ValueError: If secret is not found
    """
    manager = await get_secrets_manager()
    return await manager.get_secret_or_raise(key)