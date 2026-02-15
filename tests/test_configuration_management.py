"""
Configuration Management Test Suite
===================================

Comprehensive test suite for PRSM integration layer configuration management,
covering credential storage, user preferences, and platform configurations.

Test Categories:
- Credential storage and encryption tests
- Configuration management tests
- API endpoint tests
- Security and validation tests
- Performance and reliability tests
"""

import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from prsm.core.integrations.config.credential_manager import (
    CredentialManager, CredentialData, CredentialType, StoredCredential
)
from prsm.core.integrations.config.integration_config import (
    ConfigurationManager, UserIntegrationConfig, IntegrationPreferences,
    PlatformConfig, SecurityConfig, RateLimitConfig, SecurityLevel
)
from prsm.core.integrations.models.integration_models import IntegrationPlatform


# === Test Fixtures ===

@pytest.fixture
def temp_storage_dir():
    """Create temporary storage directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def credential_manager(temp_storage_dir):
    """Create credential manager with temporary storage"""
    return CredentialManager(storage_dir=temp_storage_dir)


@pytest.fixture
def config_manager(temp_storage_dir):
    """Create configuration manager with temporary storage"""
    return ConfigurationManager(storage_dir=temp_storage_dir)


@pytest.fixture
def sample_credential_data():
    """Sample credential data for testing"""
    return CredentialData(
        api_key="test_api_key_123",
        access_token="test_access_token_456",
        client_id="test_client_id",
        custom_fields={"scope": "read:repo"}
    )


@pytest.fixture
def sample_platform_config():
    """Sample platform configuration"""
    return PlatformConfig(
        platform=IntegrationPlatform.GITHUB,
        enabled=True,
        api_base_url="https://api.github.com",
        timeout_seconds=30,
        rate_limit=RateLimitConfig(requests_per_minute=5000),
        security=SecurityConfig(security_level=SecurityLevel.STANDARD)
    )


# === Credential Manager Tests ===

class TestCredentialManager:
    """Test credential storage and encryption functionality"""
    
    def test_initialization(self, temp_storage_dir):
        """Test credential manager initialization"""
        manager = CredentialManager(storage_dir=temp_storage_dir)
        
        assert manager.storage_dir == Path(temp_storage_dir)
        assert manager.master_key_file.exists()
        assert manager.cipher is not None
        assert len(manager.active_credentials) == 0
    
    def test_encryption_decryption(self, credential_manager, sample_credential_data):
        """Test credential encryption and decryption"""
        # Encrypt data
        encrypted_data = credential_manager._encrypt_credential_data(sample_credential_data)
        assert isinstance(encrypted_data, bytes)
        assert len(encrypted_data) > 0
        
        # Decrypt data
        decrypted_data = credential_manager._decrypt_credential_data(encrypted_data)
        assert isinstance(decrypted_data, CredentialData)
        assert decrypted_data.api_key.get_secret_value() == "test_api_key_123"
        assert decrypted_data.access_token.get_secret_value() == "test_access_token_456"
        assert decrypted_data.client_id == "test_client_id"
    
    def test_store_credential(self, credential_manager, sample_credential_data):
        """Test storing credentials"""
        credential_id = credential_manager.store_credential(
            user_id="test_user",
            platform=IntegrationPlatform.GITHUB,
            credential_data=sample_credential_data,
            credential_type=CredentialType.API_KEY,
            expires_in_days=30
        )
        
        assert credential_id is not None
        assert credential_id in credential_manager.active_credentials
        
        stored_credential = credential_manager.active_credentials[credential_id]
        assert stored_credential.user_id == "test_user"
        assert stored_credential.platform == IntegrationPlatform.GITHUB
        assert stored_credential.credential_type == CredentialType.API_KEY
        assert stored_credential.expires_at is not None
    
    def test_retrieve_credential(self, credential_manager, sample_credential_data):
        """Test retrieving stored credentials"""
        # Store credential
        credential_id = credential_manager.store_credential(
            user_id="test_user",
            platform=IntegrationPlatform.GITHUB,
            credential_data=sample_credential_data
        )
        
        # Retrieve credential
        retrieved_data = credential_manager.get_credential(
            user_id="test_user",
            platform=IntegrationPlatform.GITHUB
        )
        
        assert retrieved_data is not None
        assert retrieved_data.api_key.get_secret_value() == "test_api_key_123"
        assert retrieved_data.access_token.get_secret_value() == "test_access_token_456"
    
    def test_credential_not_found(self, credential_manager):
        """Test retrieving non-existent credential"""
        retrieved_data = credential_manager.get_credential(
            user_id="nonexistent_user",
            platform=IntegrationPlatform.GITHUB
        )
        
        assert retrieved_data is None
    
    def test_list_credentials(self, credential_manager, sample_credential_data):
        """Test listing user credentials"""
        # Store multiple credentials
        github_id = credential_manager.store_credential(
            user_id="test_user",
            platform=IntegrationPlatform.GITHUB,
            credential_data=sample_credential_data
        )
        
        hf_data = CredentialData(api_key="hf_test_key")
        hf_id = credential_manager.store_credential(
            user_id="test_user",
            platform=IntegrationPlatform.HUGGINGFACE,
            credential_data=hf_data
        )
        
        # List all credentials
        credentials = credential_manager.list_credentials("test_user")
        assert len(credentials) == 2
        
        # List platform-specific credentials
        github_creds = credential_manager.list_credentials(
            "test_user", 
            platform=IntegrationPlatform.GITHUB
        )
        assert len(github_creds) == 1
        assert github_creds[0]['platform'] == 'github'
    
    def test_update_credential(self, credential_manager, sample_credential_data):
        """Test updating stored credentials"""
        # Store credential
        credential_id = credential_manager.store_credential(
            user_id="test_user",
            platform=IntegrationPlatform.GITHUB,
            credential_data=sample_credential_data
        )
        
        # Update credential data
        updated_data = CredentialData(api_key="updated_api_key")
        success = credential_manager.update_credential(
            credential_id=credential_id,
            user_id="test_user",
            credential_data=updated_data,
            metadata={"updated": True}
        )
        
        assert success is True
        
        # Verify update
        retrieved_data = credential_manager.get_credential(
            user_id="test_user",
            platform=IntegrationPlatform.GITHUB
        )
        assert retrieved_data.api_key.get_secret_value() == "updated_api_key"
    
    def test_delete_credential(self, credential_manager, sample_credential_data):
        """Test deleting credentials"""
        # Store credential
        credential_id = credential_manager.store_credential(
            user_id="test_user",
            platform=IntegrationPlatform.GITHUB,
            credential_data=sample_credential_data
        )
        
        # Delete credential
        success = credential_manager.delete_credential(
            credential_id=credential_id,
            user_id="test_user"
        )
        
        assert success is True
        assert credential_id not in credential_manager.active_credentials
        
        # Verify deletion
        retrieved_data = credential_manager.get_credential(
            user_id="test_user",
            platform=IntegrationPlatform.GITHUB
        )
        assert retrieved_data is None
    
    def test_credential_validation(self, credential_manager, sample_credential_data):
        """Test credential validation"""
        # Store valid credential
        credential_manager.store_credential(
            user_id="test_user",
            platform=IntegrationPlatform.GITHUB,
            credential_data=sample_credential_data,
            expires_in_days=30
        )
        
        # Validate credential
        validation_result = credential_manager.validate_credential(
            user_id="test_user",
            platform=IntegrationPlatform.GITHUB
        )
        
        assert validation_result['valid'] is True
        assert validation_result['status'] == 'valid'
    
    def test_expired_credential_validation(self, credential_manager, sample_credential_data):
        """Test validation of expired credentials"""
        # Store expired credential (manually set expiration)
        credential_id = credential_manager.store_credential(
            user_id="test_user",
            platform=IntegrationPlatform.GITHUB,
            credential_data=sample_credential_data
        )
        
        # Manually expire the credential
        stored_credential = credential_manager.active_credentials[credential_id]
        stored_credential.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
        
        # Validate expired credential
        validation_result = credential_manager.validate_credential(
            user_id="test_user",
            platform=IntegrationPlatform.GITHUB
        )
        
        assert validation_result['valid'] is False
        assert validation_result['status'] == 'expired'
    
    def test_cleanup_expired_credentials(self, credential_manager, sample_credential_data):
        """Test cleanup of expired credentials"""
        # Store multiple credentials with different expirations
        valid_id = credential_manager.store_credential(
            user_id="test_user",
            platform=IntegrationPlatform.GITHUB,
            credential_data=sample_credential_data,
            expires_in_days=30
        )
        
        expired_id = credential_manager.store_credential(
            user_id="test_user",
            platform=IntegrationPlatform.HUGGINGFACE,
            credential_data=sample_credential_data
        )
        
        # Manually expire one credential
        stored_credential = credential_manager.active_credentials[expired_id]
        stored_credential.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
        
        # Run cleanup
        removed_count = credential_manager.cleanup_expired_credentials()
        
        assert removed_count == 1
        assert valid_id in credential_manager.active_credentials
        assert expired_id not in credential_manager.active_credentials
    
    def test_storage_persistence(self, temp_storage_dir, sample_credential_data):
        """Test credential persistence across manager instances"""
        # Store credential with first manager instance
        manager1 = CredentialManager(storage_dir=temp_storage_dir)
        credential_id = manager1.store_credential(
            user_id="test_user",
            platform=IntegrationPlatform.GITHUB,
            credential_data=sample_credential_data
        )
        
        # Create new manager instance with same storage
        manager2 = CredentialManager(storage_dir=temp_storage_dir)
        
        # Verify credential is loaded
        assert credential_id in manager2.active_credentials
        
        # Verify credential can be retrieved and decrypted
        retrieved_data = manager2.get_credential(
            user_id="test_user",
            platform=IntegrationPlatform.GITHUB
        )
        
        assert retrieved_data is not None
        assert retrieved_data.api_key.get_secret_value() == "test_api_key_123"


# === Configuration Manager Tests ===

class TestConfigurationManager:
    """Test configuration management functionality"""
    
    def test_initialization(self, temp_storage_dir):
        """Test configuration manager initialization"""
        manager = ConfigurationManager(storage_dir=temp_storage_dir)
        
        assert manager.storage_dir == Path(temp_storage_dir)
        assert len(manager.user_configs) == 0
    
    def test_get_user_config_default(self, config_manager):
        """Test getting default user configuration"""
        config = config_manager.get_user_config("test_user")
        
        assert config.user_id == "test_user"
        assert isinstance(config.preferences, IntegrationPreferences)
        assert len(config.platforms) == 0  # Empty by default
        assert isinstance(config.global_security, SecurityConfig)
        assert isinstance(config.global_rate_limit, RateLimitConfig)
    
    def test_platform_config_creation(self, config_manager):
        """Test automatic platform configuration creation"""
        config = config_manager.get_user_config("test_user")
        
        # Get GitHub config (should be created automatically)
        github_config = config.get_platform_config(IntegrationPlatform.GITHUB)
        
        assert github_config.platform == IntegrationPlatform.GITHUB
        assert github_config.api_base_url == "https://api.github.com"
        assert github_config.enabled is True
        assert github_config.rate_limit.requests_per_minute == 5000
    
    def test_update_user_preferences(self, config_manager):
        """Test updating user preferences"""
        # Update preferences
        new_preferences = IntegrationPreferences(
            auto_connect_on_startup=True,
            show_notifications=False,
            default_search_limit=20
        )
        
        success = config_manager.update_user_config(
            user_id="test_user",
            preferences=new_preferences
        )
        
        assert success is True
        
        # Verify update
        config = config_manager.get_user_config("test_user")
        assert config.preferences.auto_connect_on_startup is True
        assert config.preferences.show_notifications is False
        assert config.preferences.default_search_limit == 20
    
    def test_update_platform_config(self, config_manager, sample_platform_config):
        """Test updating platform-specific configuration"""
        success = config_manager.update_platform_config(
            user_id="test_user",
            platform=IntegrationPlatform.GITHUB,
            platform_config=sample_platform_config
        )
        
        assert success is True
        
        # Verify update
        config = config_manager.get_platform_config("test_user", IntegrationPlatform.GITHUB)
        assert config.api_base_url == "https://api.github.com"
        assert config.timeout_seconds == 30
    
    def test_configuration_validation(self, config_manager):
        """Test configuration validation"""
        # Get default config
        config_manager.get_user_config("test_user")
        
        # Validate configuration
        validation_result = config_manager.validate_configuration("test_user")
        
        assert validation_result['valid'] is True
        assert isinstance(validation_result['issues'], list)
        assert isinstance(validation_result['warnings'], list)
    
    def test_configuration_export_import(self, config_manager):
        """Test configuration export and import"""
        # Set up custom configuration
        config_manager.get_user_config("test_user")
        
        # Update with custom preferences
        new_preferences = IntegrationPreferences(
            auto_connect_on_startup=True,
            default_search_limit=50
        )
        config_manager.update_user_config("test_user", preferences=new_preferences)
        
        # Export configuration
        exported_config = config_manager.export_user_config("test_user")
        
        assert 'preferences' in exported_config
        assert 'export_metadata' in exported_config
        assert exported_config['preferences']['auto_connect_on_startup'] is True
        
        # Import to new user
        success = config_manager.import_user_config(
            user_id="test_user_2",
            config_data=exported_config,
            merge=False
        )
        
        assert success is True
        
        # Verify import
        imported_config = config_manager.get_user_config("test_user_2")
        assert imported_config.preferences.auto_connect_on_startup is True
        assert imported_config.preferences.default_search_limit == 50
    
    def test_configuration_reset(self, config_manager):
        """Test configuration reset to defaults"""
        # Set up custom configuration
        new_preferences = IntegrationPreferences(auto_connect_on_startup=True)
        config_manager.update_user_config("test_user", preferences=new_preferences)
        
        # Verify custom setting
        config = config_manager.get_user_config("test_user")
        assert config.preferences.auto_connect_on_startup is True
        
        # Reset configuration
        success = config_manager.reset_user_config("test_user")
        assert success is True
        
        # Verify reset
        reset_config = config_manager.get_user_config("test_user")
        assert reset_config.preferences.auto_connect_on_startup is False  # Default
    
    def test_configuration_persistence(self, temp_storage_dir):
        """Test configuration persistence across manager instances"""
        # Create first manager and set configuration
        manager1 = ConfigurationManager(storage_dir=temp_storage_dir)
        new_preferences = IntegrationPreferences(auto_connect_on_startup=True)
        manager1.update_user_config("test_user", preferences=new_preferences)
        
        # Create second manager with same storage
        manager2 = ConfigurationManager(storage_dir=temp_storage_dir)
        
        # Verify configuration is loaded
        config = manager2.get_user_config("test_user")
        assert config.preferences.auto_connect_on_startup is True


# === Integration Tests ===

class TestConfigurationIntegration:
    """Test complete configuration workflows"""
    
    def test_complete_credential_workflow(self, credential_manager, sample_credential_data):
        """Test complete credential management workflow"""
        user_id = "integration_test_user"
        platform = IntegrationPlatform.GITHUB
        
        # Store credential
        credential_id = credential_manager.store_credential(
            user_id=user_id,
            platform=platform,
            credential_data=sample_credential_data,
            expires_in_days=30,
            metadata={"source": "integration_test"}
        )
        
        # Validate credential
        validation = credential_manager.validate_credential(user_id, platform)
        assert validation['valid'] is True
        
        # List credentials
        credentials = credential_manager.list_credentials(user_id)
        assert len(credentials) == 1
        assert credentials[0]['credential_id'] == credential_id
        
        # Update credential
        updated_data = CredentialData(api_key="updated_key")
        update_success = credential_manager.update_credential(
            credential_id, user_id, updated_data
        )
        assert update_success is True
        
        # Verify update
        retrieved = credential_manager.get_credential(user_id, platform)
        assert retrieved.api_key.get_secret_value() == "updated_key"
        
        # Delete credential
        delete_success = credential_manager.delete_credential(credential_id, user_id)
        assert delete_success is True
        
        # Verify deletion
        final_credentials = credential_manager.list_credentials(user_id)
        assert len(final_credentials) == 0
    
    def test_complete_configuration_workflow(self, config_manager):
        """Test complete configuration management workflow"""
        user_id = "config_test_user"
        
        # Get default configuration
        config = config_manager.get_user_config(user_id)
        assert config.user_id == user_id
        
        # Update preferences
        new_preferences = IntegrationPreferences(
            auto_connect_on_startup=True,
            show_notifications=False,
            auto_scan_security=False,
            default_search_limit=25
        )
        
        update_success = config_manager.update_user_config(
            user_id, preferences=new_preferences
        )
        assert update_success is True
        
        # Update platform configuration
        platform_config = PlatformConfig(
            platform=IntegrationPlatform.GITHUB,
            enabled=True,
            api_base_url="https://api.github.com",
            timeout_seconds=45,
            rate_limit=RateLimitConfig(requests_per_minute=3000)
        )
        
        platform_update_success = config_manager.update_platform_config(
            user_id, IntegrationPlatform.GITHUB, platform_config
        )
        assert platform_update_success is True
        
        # Validate configuration
        validation = config_manager.validate_configuration(user_id)
        assert validation['valid'] is True
        
        # Export configuration
        exported = config_manager.export_user_config(user_id)
        assert 'preferences' in exported
        assert 'platforms' in exported
        
        # Test import to new user
        import_success = config_manager.import_user_config(
            "imported_user", exported, merge=False
        )
        assert import_success is True
        
        # Verify imported configuration
        imported_config = config_manager.get_user_config("imported_user")
        assert imported_config.preferences.auto_connect_on_startup is True
        assert imported_config.preferences.default_search_limit == 25


# === Security Tests ===

class TestSecurity:
    """Test security aspects of configuration management"""
    
    def test_credential_encryption_strength(self, credential_manager):
        """Test credential encryption security"""
        # Test that encrypted data is different each time
        credential_data = CredentialData(api_key="test_key")
        
        encrypted1 = credential_manager._encrypt_credential_data(credential_data)
        encrypted2 = credential_manager._encrypt_credential_data(credential_data)
        
        # Should be different due to Fernet's built-in randomization
        assert encrypted1 != encrypted2
        
        # But both should decrypt to the same data
        decrypted1 = credential_manager._decrypt_credential_data(encrypted1)
        decrypted2 = credential_manager._decrypt_credential_data(encrypted2)
        
        assert decrypted1.api_key.get_secret_value() == decrypted2.api_key.get_secret_value()
    
    def test_user_isolation(self, credential_manager, sample_credential_data):
        """Test that users cannot access each other's credentials"""
        # Store credential for user1
        user1_id = credential_manager.store_credential(
            user_id="user1",
            platform=IntegrationPlatform.GITHUB,
            credential_data=sample_credential_data
        )
        
        # Try to access as user2
        user2_credential = credential_manager.get_credential(
            user_id="user2",
            platform=IntegrationPlatform.GITHUB
        )
        
        assert user2_credential is None
        
        # Try to update as user2
        update_success = credential_manager.update_credential(
            credential_id=user1_id,
            user_id="user2",
            credential_data=sample_credential_data
        )
        
        assert update_success is False
        
        # Try to delete as user2
        delete_success = credential_manager.delete_credential(
            credential_id=user1_id,
            user_id="user2"
        )
        
        assert delete_success is False
    
    def test_invalid_decryption_handling(self, temp_storage_dir):
        """Test handling of corrupted encrypted data"""
        manager = CredentialManager(storage_dir=temp_storage_dir)
        
        # Test with invalid encrypted data
        with pytest.raises(ValueError, match="Failed to decrypt"):
            manager._decrypt_credential_data(b"invalid_encrypted_data")


# === Performance Tests ===

class TestPerformance:
    """Test performance characteristics of configuration management"""
    
    def test_large_number_of_credentials(self, credential_manager):
        """Test handling large numbers of credentials"""
        # Store many credentials
        credential_ids = []
        for i in range(100):
            credential_data = CredentialData(api_key=f"key_{i}")
            credential_id = credential_manager.store_credential(
                user_id="perf_test_user",
                platform=IntegrationPlatform.GITHUB,
                credential_data=credential_data,
                metadata={"index": i}
            )
            credential_ids.append(credential_id)
        
        # Test listing performance
        credentials = credential_manager.list_credentials("perf_test_user")
        assert len(credentials) == 100
        
        # Test retrieval performance
        retrieved = credential_manager.get_credential(
            "perf_test_user", IntegrationPlatform.GITHUB
        )
        assert retrieved is not None
    
    def test_concurrent_access_simulation(self, credential_manager, sample_credential_data):
        """Test simulated concurrent access patterns"""
        # Store initial credential
        credential_id = credential_manager.store_credential(
            user_id="concurrent_user",
            platform=IntegrationPlatform.GITHUB,
            credential_data=sample_credential_data
        )
        
        # Simulate multiple rapid operations
        for i in range(10):
            # Update metadata
            credential_manager.update_credential(
                credential_id=credential_id,
                user_id="concurrent_user",
                metadata={"operation": i}
            )
            
            # Retrieve credential
            retrieved = credential_manager.get_credential(
                "concurrent_user", IntegrationPlatform.GITHUB
            )
            assert retrieved is not None
            
            # Validate credential
            validation = credential_manager.validate_credential(
                "concurrent_user", IntegrationPlatform.GITHUB
            )
            assert validation['valid'] is True


# === Test Configuration ===

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])