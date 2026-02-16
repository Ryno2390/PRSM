"""
Integration Configuration
=========================

Configuration management system for PRSM integration layer settings,
preferences, and platform-specific configurations.

Features:
- User-specific integration preferences
- Platform configuration templates
- Security and compliance settings
- Performance and rate limiting configuration
- Configuration validation and defaults
"""

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4

from pydantic import BaseModel, Field, validator, SecretStr

from ..models.integration_models import IntegrationPlatform
from prsm.core.config import settings


class SecurityLevel(str, Enum):
    """Security levels for integration operations"""
    STRICT = "strict"      # Maximum security, thorough validation
    STANDARD = "standard"  # Balanced security and performance
    PERMISSIVE = "permissive"  # Minimal security for development


class RateLimitConfig(BaseModel):
    """Rate limiting configuration"""
    requests_per_minute: int = Field(default=60, ge=1, le=10000)
    burst_limit: int = Field(default=10, ge=1, le=100)
    backoff_factor: float = Field(default=1.5, ge=1.0, le=5.0)
    max_retry_attempts: int = Field(default=3, ge=0, le=10)


class SecurityConfig(BaseModel):
    """Security configuration for integrations"""
    security_level: SecurityLevel = SecurityLevel.STANDARD
    require_license_validation: bool = True
    require_vulnerability_scan: bool = True
    allow_copyleft_licenses: bool = False
    allow_unknown_licenses: bool = False
    max_file_size_mb: int = Field(default=100, ge=1, le=1000)
    allowed_file_extensions: List[str] = Field(default_factory=lambda: [
        '.py', '.js', '.ts', '.json', '.yaml', '.yml', '.md', '.txt', '.ipynb'
    ])
    blocked_file_patterns: List[str] = Field(default_factory=lambda: [
        '*.exe', '*.dll', '*.so', '*.dylib', '*.bin'
    ])


class PlatformConfig(BaseModel):
    """Platform-specific configuration"""
    platform: IntegrationPlatform
    enabled: bool = True
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    custom_settings: Dict[str, Any] = Field(default_factory=dict)
    
    # Platform-specific settings
    api_base_url: Optional[str] = None
    api_version: Optional[str] = None
    timeout_seconds: int = Field(default=30, ge=5, le=300)
    max_concurrent_requests: int = Field(default=5, ge=1, le=20)
    
    # OAuth/Authentication settings
    oauth_scopes: List[str] = Field(default_factory=list)
    auth_flow_timeout: int = Field(default=300, ge=60, le=1800)
    
    # Content preferences
    default_import_location: Optional[str] = None
    auto_organize_imports: bool = True
    preserve_folder_structure: bool = True


class IntegrationPreferences(BaseModel):
    """User preferences for integration behavior"""
    auto_connect_on_startup: bool = False
    show_notifications: bool = True
    notification_level: str = Field(default="info")  # debug, info, warning, error
    
    # Import preferences
    auto_scan_security: bool = True
    auto_validate_licenses: bool = True
    auto_reward_creators: bool = True
    confirm_before_import: bool = True
    
    # UI preferences
    default_search_limit: int = Field(default=10, ge=1, le=100)
    remember_search_filters: bool = True
    show_advanced_options: bool = False
    
    # Performance preferences
    enable_caching: bool = True
    cache_duration_hours: int = Field(default=24, ge=1, le=168)
    enable_background_sync: bool = True


class UserIntegrationConfig(BaseModel):
    """Complete integration configuration for a user"""
    config_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Configuration sections
    preferences: IntegrationPreferences = Field(default_factory=IntegrationPreferences)
    platforms: Dict[str, PlatformConfig] = Field(default_factory=dict)
    
    # Global settings
    global_security: SecurityConfig = Field(default_factory=SecurityConfig)
    global_rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    
    # Metadata
    version: str = "1.0"
    is_active: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_platform_config(self, platform: IntegrationPlatform) -> PlatformConfig:
        """Get configuration for specific platform"""
        platform_key = platform.value
        
        if platform_key not in self.platforms:
            # Create default configuration
            self.platforms[platform_key] = self._create_default_platform_config(platform)
            
        return self.platforms[platform_key]
    
    def _create_default_platform_config(self, platform: IntegrationPlatform) -> PlatformConfig:
        """Create default configuration for platform"""
        
        # Platform-specific defaults
        defaults = {
            IntegrationPlatform.GITHUB: {
                'api_base_url': 'https://api.github.com',
                'api_version': '2022-11-28',
                'oauth_scopes': ['public_repo', 'read:user'],
                'rate_limit': RateLimitConfig(requests_per_minute=5000, burst_limit=100),  # GitHub has high limits
                'timeout_seconds': 30
            },
            IntegrationPlatform.HUGGINGFACE: {
                'api_base_url': 'https://huggingface.co',
                'api_version': 'v1',
                'oauth_scopes': ['read-repos'],
                'rate_limit': RateLimitConfig(requests_per_minute=1000, burst_limit=50),
                'timeout_seconds': 45  # Model downloads can be slow
            },
            IntegrationPlatform.OLLAMA: {
                'api_base_url': 'http://localhost:11434',
                'api_version': 'v1',
                'oauth_scopes': [],  # No auth needed
                'rate_limit': RateLimitConfig(requests_per_minute=120, burst_limit=20),  # Local limits
                'timeout_seconds': 60  # Model operations can take time
            }
        }
        
        platform_defaults = defaults.get(platform, {})
        
        return PlatformConfig(
            platform=platform,
            **platform_defaults
        )


class ConfigurationManager:
    """
    Configuration management system for integration layer
    
    Manages user-specific configurations, platform settings,
    and system-wide integration preferences.
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize configuration manager"""
        
        # Storage setup
        self.storage_dir = Path(storage_dir or getattr(settings, "config_storage_dir", "~/.prsm/config")).expanduser()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = self.storage_dir / "integration_config.json"
        
        # In-memory storage
        self.user_configs: Dict[str, UserIntegrationConfig] = {}
        
        # Load existing configurations
        self._load_configurations()
        
        print(f"⚙️ Configuration manager initialized with storage at {self.storage_dir}")
    
    def _load_configurations(self):
        """Load configurations from storage"""
        try:
            if not self.config_file.exists():
                return
            
            import json
            with open(self.config_file, 'r') as f:
                data = json.load(f)
            
            for user_id, config_data in data.get('user_configs', {}).items():
                try:
                    # Parse dates
                    for date_field in ['created_at', 'updated_at']:
                        if config_data.get(date_field):
                            config_data[date_field] = datetime.fromisoformat(
                                config_data[date_field].replace('Z', '+00:00')
                            )
                    
                    config = UserIntegrationConfig(**config_data)
                    self.user_configs[user_id] = config
                    
                except Exception as e:
                    print(f"⚠️ Failed to load config for user {user_id}: {e}")
            
            print(f"📋 Loaded configurations for {len(self.user_configs)} users")
            
        except Exception as e:
            print(f"⚠️ Failed to load configurations: {e}")
    
    def _save_configurations(self):
        """Save configurations to storage"""
        try:
            import json
            
            data = {
                'version': '1.0',
                'user_configs': {}
            }
            
            for user_id, config in self.user_configs.items():
                config_data = config.model_dump()
                
                # Convert dates to ISO format
                for date_field in ['created_at', 'updated_at']:
                    if config_data.get(date_field):
                        config_data[date_field] = config_data[date_field].isoformat()
                
                data['user_configs'][user_id] = config_data
            
            # Write atomically
            temp_file = self.config_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            temp_file.replace(self.config_file)
            
        except Exception as e:
            print(f"❌ Failed to save configurations: {e}")
            raise
    
    # === Public API ===
    
    def get_user_config(self, user_id: str) -> UserIntegrationConfig:
        """
        Get configuration for user (creates default if not exists)
        
        Args:
            user_id: User identifier
            
        Returns:
            User's integration configuration
        """
        if user_id not in self.user_configs:
            # Create default configuration
            config = UserIntegrationConfig(user_id=user_id)
            self.user_configs[user_id] = config
            self._save_configurations()
            print(f"⚙️ Created default configuration for user {user_id}")
        
        return self.user_configs[user_id]
    
    def update_user_config(
        self,
        user_id: str,
        preferences: Optional[IntegrationPreferences] = None,
        platform_configs: Optional[Dict[str, PlatformConfig]] = None,
        global_security: Optional[SecurityConfig] = None,
        global_rate_limit: Optional[RateLimitConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update user configuration
        
        Args:
            user_id: User identifier
            preferences: Updated preferences
            platform_configs: Updated platform configurations
            global_security: Updated global security settings
            global_rate_limit: Updated global rate limiting
            metadata: Additional metadata
            
        Returns:
            True if updated successfully
        """
        try:
            config = self.get_user_config(user_id)
            
            # Update fields if provided
            if preferences:
                config.preferences = preferences
            
            if platform_configs:
                config.platforms.update(platform_configs)
            
            if global_security:
                config.global_security = global_security
            
            if global_rate_limit:
                config.global_rate_limit = global_rate_limit
            
            if metadata:
                config.metadata.update(metadata)
            
            # Update timestamp
            config.updated_at = datetime.now(timezone.utc)
            
            # Save changes
            self._save_configurations()
            
            print(f"⚙️ Updated configuration for user {user_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to update user configuration: {e}")
            return False
    
    def get_platform_config(
        self,
        user_id: str,
        platform: IntegrationPlatform
    ) -> PlatformConfig:
        """
        Get platform-specific configuration for user
        
        Args:
            user_id: User identifier
            platform: Integration platform
            
        Returns:
            Platform configuration
        """
        config = self.get_user_config(user_id)
        return config.get_platform_config(platform)
    
    def update_platform_config(
        self,
        user_id: str,
        platform: IntegrationPlatform,
        platform_config: PlatformConfig
    ) -> bool:
        """
        Update platform-specific configuration
        
        Args:
            user_id: User identifier
            platform: Integration platform
            platform_config: New platform configuration
            
        Returns:
            True if updated successfully
        """
        try:
            config = self.get_user_config(user_id)
            config.platforms[platform.value] = platform_config
            config.updated_at = datetime.now(timezone.utc)
            
            self._save_configurations()
            
            print(f"⚙️ Updated {platform.value} configuration for user {user_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to update platform configuration: {e}")
            return False
    
    def validate_configuration(
        self,
        user_id: str,
        platform: Optional[IntegrationPlatform] = None
    ) -> Dict[str, Any]:
        """
        Validate user configuration
        
        Args:
            user_id: User identifier
            platform: Optional specific platform to validate
            
        Returns:
            Validation results
        """
        try:
            config = self.get_user_config(user_id)
            issues = []
            warnings = []
            
            # Validate global settings
            if config.global_rate_limit.requests_per_minute > 10000:
                warnings.append("Very high rate limit may cause API issues")
            
            if config.global_security.max_file_size_mb > 500:
                warnings.append("Large file size limit may impact performance")
            
            # Validate platform configurations
            platforms_to_check = [platform] if platform else list(IntegrationPlatform)
            
            for plat in platforms_to_check:
                if isinstance(plat, str):
                    continue  # Skip string values in enum iteration
                    
                plat_config = config.get_platform_config(plat)
                
                # Validate URLs
                if plat_config.api_base_url:
                    if not plat_config.api_base_url.startswith(('http://', 'https://')):
                        issues.append(f"{plat.value}: Invalid API base URL")
                
                # Validate timeouts
                if plat_config.timeout_seconds < 5:
                    warnings.append(f"{plat.value}: Very short timeout may cause failures")
                
                # Validate rate limits
                if plat_config.rate_limit.requests_per_minute > 5000 and plat == IntegrationPlatform.GITHUB:
                    # GitHub has high limits, this is OK
                    pass
                elif plat_config.rate_limit.requests_per_minute > 1000:
                    warnings.append(f"{plat.value}: High rate limit may exceed API limits")
            
            # Validate preferences
            if config.preferences.default_search_limit > 100:
                warnings.append("High search limit may impact performance")
            
            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'warnings': warnings,
                'config_version': config.version,
                'last_updated': config.updated_at.isoformat()
            }
            
        except Exception as e:
            return {
                'valid': False,
                'issues': [f"Validation failed: {str(e)}"],
                'warnings': [],
                'config_version': 'unknown',
                'last_updated': None
            }
    
    def export_user_config(self, user_id: str) -> Dict[str, Any]:
        """
        Export user configuration for backup/sharing
        
        Args:
            user_id: User identifier
            
        Returns:
            Exportable configuration data
        """
        try:
            config = self.get_user_config(user_id)
            
            # Export without sensitive data
            export_data = config.model_dump()
            
            # Remove user-specific identifiers
            export_data.pop('config_id', None)
            export_data.pop('user_id', None)
            export_data.pop('created_at', None)
            export_data.pop('updated_at', None)
            
            # Add export metadata
            export_data['export_metadata'] = {
                'exported_at': datetime.now(timezone.utc).isoformat(),
                'export_version': '1.0',
                'source': 'prsm_integration_layer'
            }
            
            return export_data
            
        except Exception as e:
            print(f"❌ Failed to export configuration: {e}")
            return {}
    
    def import_user_config(
        self,
        user_id: str,
        config_data: Dict[str, Any],
        merge: bool = True
    ) -> bool:
        """
        Import user configuration from backup/share
        
        Args:
            user_id: User identifier
            config_data: Configuration data to import
            merge: Whether to merge with existing config or replace
            
        Returns:
            True if imported successfully
        """
        try:
            # Remove export metadata
            config_data.pop('export_metadata', None)
            
            if merge:
                # Merge with existing configuration
                existing_config = self.get_user_config(user_id)
                
                # Update specific sections
                if 'preferences' in config_data:
                    existing_config.preferences = IntegrationPreferences(**config_data['preferences'])
                
                if 'platforms' in config_data:
                    for platform_key, platform_data in config_data['platforms'].items():
                        platform_data['platform'] = platform_key  # Ensure platform is set
                        existing_config.platforms[platform_key] = PlatformConfig(**platform_data)
                
                if 'global_security' in config_data:
                    existing_config.global_security = SecurityConfig(**config_data['global_security'])
                
                if 'global_rate_limit' in config_data:
                    existing_config.global_rate_limit = RateLimitConfig(**config_data['global_rate_limit'])
                
                existing_config.updated_at = datetime.now(timezone.utc)
                
            else:
                # Replace entire configuration
                config_data['user_id'] = user_id
                config_data['config_id'] = str(uuid4())
                config_data['created_at'] = datetime.now(timezone.utc)
                config_data['updated_at'] = datetime.now(timezone.utc)
                
                new_config = UserIntegrationConfig(**config_data)
                self.user_configs[user_id] = new_config
            
            # Save changes
            self._save_configurations()
            
            print(f"📥 Imported configuration for user {user_id} (merge={merge})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to import configuration: {e}")
            return False
    
    def reset_user_config(self, user_id: str) -> bool:
        """
        Reset user configuration to defaults
        
        Args:
            user_id: User identifier
            
        Returns:
            True if reset successfully
        """
        try:
            # Create new default configuration
            new_config = UserIntegrationConfig(user_id=user_id)
            self.user_configs[user_id] = new_config
            
            # Save changes
            self._save_configurations()
            
            print(f"🔄 Reset configuration for user {user_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to reset configuration: {e}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get configuration system statistics"""
        try:
            total_users = len(self.user_configs)
            active_configs = sum(1 for config in self.user_configs.values() if config.is_active)
            
            platform_usage = {}
            security_levels = {}
            
            for config in self.user_configs.values():
                # Count platform configurations
                for platform_key in config.platforms:
                    platform_usage[platform_key] = platform_usage.get(platform_key, 0) + 1
                
                # Count security levels
                level = config.global_security.security_level
                security_levels[level] = security_levels.get(level, 0) + 1
            
            return {
                'total_users': total_users,
                'active_configs': active_configs,
                'platform_usage': platform_usage,
                'security_levels': security_levels,
                'storage_path': str(self.storage_dir)
            }
            
        except Exception as e:
            print(f"❌ Failed to get system stats: {e}")
            return {}


# Global configuration manager instance
config_manager = ConfigurationManager()