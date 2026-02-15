"""
Integration Layer Test Suite
===========================

Comprehensive test suite for PRSM's integration layer, covering:
- GitHub and Hugging Face connectors
- Integration manager functionality
- Security sandbox operations
- API endpoints
- Provenance tracking and FTNS rewards

Test Categories:
- Unit tests for individual components
- Integration tests for end-to-end workflows
- Mock tests for external API interactions
- Security tests for sandbox and compliance
"""

import asyncio
import json
import tempfile
import os
from datetime import datetime, timezone
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import aiohttp
from fastapi.testclient import TestClient

# Import integration layer components
from prsm.core.integrations.core.integration_manager import IntegrationManager
from prsm.core.integrations.core.base_connector import BaseConnector, ConnectorStatus
from prsm.core.integrations.connectors.github_connector import GitHubConnector
from prsm.core.integrations.connectors.huggingface_connector import HuggingFaceConnector
from prsm.core.integrations.connectors.ollama_connector import OllamaConnector
from prsm.core.integrations.security.sandbox_manager import SandboxManager
from prsm.core.integrations.models.integration_models import (
    IntegrationPlatform, ConnectorConfig, IntegrationSource,
    ImportRequest, ImportStatus, SecurityRisk, LicenseType
)


# === Test Fixtures ===

@pytest.fixture
def mock_github_config():
    """Mock GitHub connector configuration"""
    return ConnectorConfig(
        platform=IntegrationPlatform.GITHUB,
        user_id="test_user",
        oauth_credentials={"access_token": "test_token"}
    )


@pytest.fixture
def mock_hf_config():
    """Mock Hugging Face connector configuration"""
    return ConnectorConfig(
        platform=IntegrationPlatform.HUGGINGFACE,
        user_id="test_user",
        api_key="test_api_key"
    )


@pytest.fixture
def mock_ollama_config():
    """Mock Ollama connector configuration"""
    return ConnectorConfig(
        platform=IntegrationPlatform.OLLAMA,
        user_id="test_user",
        custom_settings={"base_url": "http://localhost:11434"}
    )


@pytest.fixture
def sample_github_repo():
    """Sample GitHub repository metadata"""
    return {
        "id": 123456,
        "full_name": "microsoft/vscode",
        "name": "vscode",
        "description": "Visual Studio Code",
        "owner": {"login": "microsoft"},
        "html_url": "https://github.com/microsoft/vscode",
        "clone_url": "https://github.com/microsoft/vscode.git",
        "size": 50000,
        "stargazers_count": 140000,
        "forks_count": 25000,
        "license": {"key": "mit", "name": "MIT License"},
        "topics": ["editor", "typescript", "electron"],
        "default_branch": "main",
        "created_at": "2015-09-03T20:23:34Z",
        "updated_at": "2024-01-15T10:30:00Z"
    }


@pytest.fixture
def sample_hf_model():
    """Sample Hugging Face model metadata"""
    return {
        "id": "microsoft/DialoGPT-medium",
        "sha": "abc123def456",
        "pipeline_tag": "conversational",
        "library_name": "transformers",
        "tags": ["pytorch", "gpt2", "conversational", "license:mit"],
        "downloads": 50000,
        "likes": 250,
        "createdAt": "2020-05-15T10:00:00Z",
        "lastModified": "2023-12-01T14:30:00Z",
        "cardData": {"license": "mit"},
        "config": {"model_type": "gpt2"}
    }


@pytest.fixture
def sample_ollama_model():
    """Sample Ollama model metadata"""
    return {
        "name": "llama2:7b",
        "size": 3825819519,
        "digest": "sha256:8fdf52f7",
        "modified_at": "2024-01-15T10:30:00Z",
        "details": {
            "family": "llama",
            "families": ["llama"],
            "format": "gguf",
            "parameter_size": "7B",
            "quantization_level": "Q4_0"
        }
    }


# === Unit Tests - Base Connector ===

class TestBaseConnector:
    """Test the BaseConnector abstract class functionality"""
    
    @pytest.mark.asyncio
    async def test_connector_initialization(self, mock_github_config):
        """Test connector initialization"""
        connector = GitHubConnector(mock_github_config)
        
        assert connector.platform == IntegrationPlatform.GITHUB
        assert connector.user_id == "test_user"
        assert connector.status == ConnectorStatus.INITIALIZING
        assert connector.error_count == 0
    
    @pytest.mark.asyncio 
    async def test_connector_metrics(self, mock_github_config):
        """Test connector metrics collection"""
        connector = GitHubConnector(mock_github_config)
        
        # Simulate some activity
        connector.total_requests = 10
        connector.successful_requests = 8
        connector.failed_requests = 2
        connector.average_response_time = 1.5
        
        metrics = connector.get_metrics()
        
        assert metrics["platform"] == "github"
        assert metrics["total_requests"] == 10
        assert metrics["error_rate"] == 0.2
        assert metrics["average_response_time"] == 1.5


# === Unit Tests - GitHub Connector ===

class TestGitHubConnector:
    """Test GitHub connector functionality"""
    
    @pytest.mark.asyncio
    async def test_authentication_success(self, mock_github_config):
        """Test successful GitHub authentication"""
        connector = GitHubConnector(mock_github_config)
        
        # Mock successful API response
        mock_user_info = {"login": "test_user", "id": 12345}
        
        with patch.object(connector, '_make_api_request', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_user_info
            
            result = await connector.authenticate()
            
            assert result is True
            assert connector.authenticated_user == "test_user"
            mock_api.assert_called_once_with("/user")
    
    @pytest.mark.asyncio
    async def test_authentication_failure(self, mock_github_config):
        """Test failed GitHub authentication"""
        connector = GitHubConnector(mock_github_config)
        
        with patch.object(connector, '_make_api_request', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = None
            
            result = await connector.authenticate()
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_search_repositories(self, mock_github_config, sample_github_repo):
        """Test GitHub repository search"""
        connector = GitHubConnector(mock_github_config)
        
        # Mock search response
        mock_response = {"items": [sample_github_repo]}
        
        with patch.object(connector, '_make_api_request', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_response
            
            results = await connector.search_content("vscode", "repository", limit=1)
            
            assert len(results) == 1
            assert results[0].platform == IntegrationPlatform.GITHUB
            assert results[0].external_id == "microsoft/vscode"
            assert results[0].display_name == "vscode"
    
    @pytest.mark.asyncio
    async def test_get_repository_metadata(self, mock_github_config, sample_github_repo):
        """Test getting detailed repository metadata"""
        connector = GitHubConnector(mock_github_config)
        
        with patch.object(connector, '_make_api_request', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = sample_github_repo
            
            # Mock additional API calls
            with patch.object(connector, '_get_repository_languages', new_callable=AsyncMock) as mock_langs:
                mock_langs.return_value = {"TypeScript": 70, "JavaScript": 30}
                
                metadata = await connector.get_content_metadata("microsoft/vscode")
                
                assert metadata["type"] == "repository"
                assert metadata["full_name"] == "microsoft/vscode"
                assert metadata["creator"] == "microsoft"
                assert metadata["license"]["key"] == "mit"
    
    @pytest.mark.asyncio
    async def test_license_validation_permissive(self, mock_github_config, sample_github_repo):
        """Test license validation for permissive license"""
        connector = GitHubConnector(mock_github_config)
        
        with patch.object(connector, '_get_repository_metadata', new_callable=AsyncMock) as mock_meta:
            mock_meta.return_value = {
                "license": {"key": "mit", "name": "MIT License"},
                "type": "repository"
            }
            
            license_info = await connector.validate_license("microsoft/vscode")
            
            assert license_info["type"] == "permissive"
            assert license_info["compliant"] is True
            assert len(license_info["issues"]) == 0
    
    @pytest.mark.asyncio
    async def test_license_validation_copyleft(self, mock_github_config):
        """Test license validation for copyleft license"""
        connector = GitHubConnector(mock_github_config)
        
        with patch.object(connector, '_get_repository_metadata', new_callable=AsyncMock) as mock_meta:
            mock_meta.return_value = {
                "license": {"key": "gpl-3.0", "name": "GNU General Public License v3.0"},
                "type": "repository"
            }
            
            license_info = await connector.validate_license("some/gpl-repo")
            
            assert license_info["type"] == "copyleft"
            assert license_info["compliant"] is False
            assert len(license_info["issues"]) > 0


# === Unit Tests - Ollama Connector ===

class TestOllamaConnector:
    """Test Ollama connector functionality"""
    
    @pytest.mark.asyncio
    async def test_authentication_local(self, mock_ollama_config):
        """Test Ollama local authentication"""
        connector = OllamaConnector(mock_ollama_config)
        
        with patch.object(connector, '_test_connection', new_callable=AsyncMock) as mock_test:
            mock_test.return_value = True
            
            result = await connector.authenticate()
            
            assert result is True
            assert connector.authenticated_user == "local"
    
    @pytest.mark.asyncio
    async def test_search_local_models(self, mock_ollama_config, sample_ollama_model):
        """Test Ollama local model search"""
        connector = OllamaConnector(mock_ollama_config)
        
        # Mock model data
        from prsm.core.integrations.connectors.ollama_connector import OllamaModelInfo
        connector.available_models = [
            OllamaModelInfo(
                name="llama2:7b",
                tag="7b",
                size=3825819519,
                digest="sha256:8fdf52f7",
                details={"parameter_size": "7B"}
            )
        ]
        
        with patch.object(connector, '_refresh_available_models', new_callable=AsyncMock):
            results = await connector.search_content("llama", "model", limit=1)
            
            assert len(results) == 1
            assert results[0].platform == IntegrationPlatform.OLLAMA
            assert results[0].external_id == "llama2:7b"
            assert results[0].display_name == "llama2"
            assert results[0].owner_id == "local"
    
    @pytest.mark.asyncio
    async def test_get_model_metadata_local(self, mock_ollama_config):
        """Test getting local model metadata"""
        connector = OllamaConnector(mock_ollama_config)
        
        mock_response = {
            "template": "{{ .System }}\n\n{{ .Prompt }}",
            "parameters": {"temperature": 0.8},
            "details": {"parameter_size": "7B"},
            "license": "Apache License 2.0"
        }
        
        with patch.object(connector, '_make_api_request', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_response
            
            # Add model to available models
            from prsm.core.integrations.connectors.ollama_connector import OllamaModelInfo
            connector.available_models = [
                OllamaModelInfo(name="llama2:7b", tag="7b", size=3825819519)
            ]
            
            metadata = await connector.get_content_metadata("llama2:7b")
            
            assert metadata["type"] == "local_model"
            assert metadata["name"] == "llama2:7b"
            assert metadata["platform"] == "ollama"
            assert metadata["size_gb"] == 3.57
    
    @pytest.mark.asyncio
    async def test_license_validation_local_model(self, mock_ollama_config):
        """Test license validation for local Ollama model"""
        connector = OllamaConnector(mock_ollama_config)
        
        with patch.object(connector, 'get_content_metadata', new_callable=AsyncMock) as mock_meta:
            mock_meta.return_value = {
                "license": "Apache License 2.0"
            }
            
            license_info = await connector.validate_license("llama2:7b")
            
            assert license_info["type"] == "permissive"
            assert license_info["compliant"] is True
            assert "Local Ollama model" in license_info["note"]


# === Unit Tests - Hugging Face Connector ===

class TestHuggingFaceConnector:
    """Test Hugging Face connector functionality"""
    
    @pytest.mark.asyncio
    async def test_authentication_with_token(self, mock_hf_config):
        """Test HF authentication with API token"""
        connector = HuggingFaceConnector(mock_hf_config)
        
        mock_user_info = {"name": "test_user", "fullname": "Test User"}
        
        with patch.object(connector, '_make_api_request', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_user_info
            
            result = await connector.authenticate()
            
            assert result is True
            assert connector.authenticated_user == "test_user"
    
    @pytest.mark.asyncio
    async def test_authentication_anonymous(self):
        """Test HF authentication in anonymous mode"""
        config = ConnectorConfig(
            platform=IntegrationPlatform.HUGGINGFACE,
            user_id="test_user"
        )
        connector = HuggingFaceConnector(config)
        
        result = await connector.authenticate()
        
        assert result is True
        assert connector.authenticated_user == "anonymous"
    
    @pytest.mark.asyncio
    async def test_search_models(self, mock_hf_config, sample_hf_model):
        """Test HF model search"""
        connector = HuggingFaceConnector(mock_hf_config)
        
        with patch.object(connector, '_make_api_request', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = [sample_hf_model]
            
            results = await connector.search_content("DialoGPT", "model", limit=1)
            
            assert len(results) == 1
            assert results[0].platform == IntegrationPlatform.HUGGINGFACE
            assert results[0].external_id == "microsoft/DialoGPT-medium"
    
    @pytest.mark.asyncio
    async def test_get_model_metadata(self, mock_hf_config, sample_hf_model):
        """Test getting detailed model metadata"""
        connector = HuggingFaceConnector(mock_hf_config)
        
        with patch.object(connector, '_make_api_request', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = sample_hf_model
            
            # Mock additional calls
            with patch.object(connector, '_get_model_card', new_callable=AsyncMock) as mock_card:
                mock_card.return_value = "# DialoGPT Model\nThis is a conversational AI model..."
                
                metadata = await connector.get_content_metadata("microsoft/DialoGPT-medium")
                
                assert metadata["type"] == "model"
                assert metadata["id"] == "microsoft/DialoGPT-medium"
                assert metadata["creator"] == "microsoft"
                assert metadata["pipeline_tag"] == "conversational"


# === Unit Tests - Security Sandbox ===

class TestSandboxManager:
    """Test security sandbox functionality"""
    
    @pytest.fixture
    def sandbox_manager(self):
        """Create sandbox manager instance"""
        return SandboxManager()
    
    @pytest.mark.asyncio
    async def test_basic_file_validation(self, sandbox_manager):
        """Test basic file validation"""
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("print('Hello, World!')")
            temp_path = f.name
        
        try:
            validation = await sandbox_manager._perform_basic_validation(temp_path)
            
            assert validation["passed"] is True
            assert len(validation["issues"]) == 0
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_suspicious_file_detection(self, sandbox_manager):
        """Test detection of suspicious files"""
        # Create temporary executable file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.exe', delete=False) as f:
            f.write("fake executable content")
            temp_path = f.name
        
        try:
            validation = await sandbox_manager._perform_basic_validation(temp_path)
            
            assert validation["passed"] is False
            assert any("suspicious" in issue.lower() for issue in validation["issues"])
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_license_compliance_scan(self, sandbox_manager):
        """Test license compliance scanning"""
        metadata = {
            "license": {"key": "mit", "name": "MIT License"}
        }
        
        result = await sandbox_manager._scan_license_compliance("dummy_path", metadata)
        
        assert result["type"] == LicenseType.PERMISSIVE
        assert result["compliant"] is True
    
    @pytest.mark.asyncio
    async def test_vulnerability_pattern_detection(self, sandbox_manager):
        """Test vulnerability pattern detection"""
        # Create file with potential SQL injection
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("query = 'SELECT * FROM users WHERE id = ' + user_input")
            temp_path = f.name
        
        try:
            vulnerabilities = []
            await sandbox_manager._pattern_based_vuln_scan(temp_path, vulnerabilities)
            
            # Should detect potential SQL injection
            assert len(vulnerabilities) > 0
        finally:
            os.unlink(temp_path)


# === Unit Tests - Integration Manager ===

class TestIntegrationManager:
    """Test integration manager functionality"""
    
    @pytest.fixture
    def integration_manager(self):
        """Create integration manager instance"""
        return IntegrationManager()
    
    @pytest.mark.asyncio
    async def test_connector_registration(self, integration_manager, mock_github_config):
        """Test connector registration"""
        with patch.object(GitHubConnector, 'initialize', new_callable=AsyncMock) as mock_init:
            mock_init.return_value = True
            
            success = await integration_manager.register_connector(GitHubConnector, mock_github_config)
            
            assert success is True
            assert IntegrationPlatform.GITHUB in integration_manager.connectors
            assert integration_manager.stats.platforms_connected == 1
    
    @pytest.mark.asyncio
    async def test_import_request_submission(self, integration_manager):
        """Test import request submission"""
        # Create mock connector
        mock_connector = MagicMock()
        mock_connector.is_healthy.return_value = True
        integration_manager.connectors[IntegrationPlatform.GITHUB] = mock_connector
        
        # Create import request
        source = IntegrationSource(
            platform=IntegrationPlatform.GITHUB,
            external_id="microsoft/vscode",
            display_name="vscode",
            owner_id="microsoft",
            url="https://github.com/microsoft/vscode"
        )
        
        request = ImportRequest(
            user_id="test_user",
            source=source,
            import_type="repository"
        )
        
        with patch.object(integration_manager, '_validate_import_request', new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = True
            
            request_id = await integration_manager.submit_import_request(request)
            
            assert request_id is not None
            assert request_id in integration_manager.active_imports
    
    @pytest.mark.asyncio
    async def test_health_check_all_connectors(self, integration_manager):
        """Test health check on all connectors"""
        # Add mock connectors
        mock_github = MagicMock()
        mock_github.health_check = AsyncMock(return_value=MagicMock(status="healthy"))
        integration_manager.connectors[IntegrationPlatform.GITHUB] = mock_github
        
        mock_hf = MagicMock()
        mock_hf.health_check = AsyncMock(return_value=MagicMock(status="degraded"))
        integration_manager.connectors[IntegrationPlatform.HUGGINGFACE] = mock_hf
        
        health_results = await integration_manager.health_check_all_connectors()
        
        assert len(health_results) == 2
        assert IntegrationPlatform.GITHUB in health_results
        assert IntegrationPlatform.HUGGINGFACE in health_results


# === Integration Tests ===

class TestIntegrationWorkflows:
    """Test end-to-end integration workflows"""
    
    @pytest.mark.asyncio
    async def test_github_repository_import_workflow(self, mock_github_config, sample_github_repo):
        """Test complete GitHub repository import workflow"""
        # Initialize components
        manager = IntegrationManager()
        
        # Mock GitHub connector responses
        with patch.object(GitHubConnector, 'authenticate', new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = True
            
            with patch.object(GitHubConnector, 'get_content_metadata', new_callable=AsyncMock) as mock_meta:
                mock_meta.return_value = {
                    "type": "repository",
                    "creator": "microsoft",
                    "license": {"key": "mit", "name": "MIT License"}
                }
                
                with patch.object(GitHubConnector, 'validate_license', new_callable=AsyncMock) as mock_license:
                    mock_license.return_value = {"type": "permissive", "compliant": True}
                    
                    with patch.object(GitHubConnector, 'download_content', new_callable=AsyncMock) as mock_download:
                        mock_download.return_value = True
                        
                        # Register connector
                        await manager.register_connector(GitHubConnector, mock_github_config)
                        
                        # Create import request
                        source = IntegrationSource(
                            platform=IntegrationPlatform.GITHUB,
                            external_id="microsoft/vscode",
                            display_name="vscode",
                            owner_id="microsoft",
                            url="https://github.com/microsoft/vscode"
                        )
                        
                        request = ImportRequest(
                            user_id="test_user",
                            source=source,
                            import_type="repository"
                        )
                        
                        # Submit import request
                        request_id = await manager.submit_import_request(request)
                        
                        # Wait for completion (mock execution)
                        await asyncio.sleep(0.1)
                        
                        # Check final status
                        status = await manager.get_import_status(request_id)
                        assert status is not None
    
    @pytest.mark.asyncio
    async def test_huggingface_model_import_workflow(self, mock_hf_config, sample_hf_model):
        """Test complete Hugging Face model import workflow"""
        manager = IntegrationManager()
        
        with patch.object(HuggingFaceConnector, 'authenticate', new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = True
            
            with patch.object(HuggingFaceConnector, 'get_content_metadata', new_callable=AsyncMock) as mock_meta:
                mock_meta.return_value = {
                    "type": "model",
                    "creator": "microsoft",
                    "tags": ["license:mit"]
                }
                
                with patch.object(HuggingFaceConnector, 'validate_license', new_callable=AsyncMock) as mock_license:
                    mock_license.return_value = {"type": "permissive", "compliant": True}
                    
                    with patch.object(HuggingFaceConnector, 'download_content', new_callable=AsyncMock) as mock_download:
                        mock_download.return_value = True
                        
                        # Register connector
                        await manager.register_connector(HuggingFaceConnector, mock_hf_config)
                        
                        # Create import request
                        source = IntegrationSource(
                            platform=IntegrationPlatform.HUGGINGFACE,
                            external_id="microsoft/DialoGPT-medium",
                            display_name="DialoGPT-medium",
                            owner_id="microsoft",
                            url="https://huggingface.co/microsoft/DialoGPT-medium"
                        )
                        
                        request = ImportRequest(
                            user_id="test_user",
                            source=source,
                            import_type="model"
                        )
                        
                        # Submit import request
                        request_id = await manager.submit_import_request(request)
                        
                        assert request_id is not None


# === Performance Tests ===

class TestPerformance:
    """Test performance and scalability"""
    
    @pytest.mark.asyncio
    async def test_concurrent_searches(self, mock_github_config):
        """Test concurrent search operations"""
        connector = GitHubConnector(mock_github_config)
        
        # Mock search response
        mock_response = {"items": []}
        
        with patch.object(connector, '_make_api_request', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_response
            
            # Execute multiple concurrent searches
            search_tasks = [
                connector.search_content(f"query{i}", "repository", limit=5)
                for i in range(10)
            ]
            
            results = await asyncio.gather(*search_tasks)
            
            assert len(results) == 10
            assert all(isinstance(result, list) for result in results)
    
    @pytest.mark.asyncio
    async def test_rate_limiting_handling(self, mock_github_config):
        """Test rate limiting handling"""
        connector = GitHubConnector(mock_github_config)
        
        # Simulate rate limit exceeded
        connector.rate_limit_remaining = 0
        connector.status = ConnectorStatus.RATE_LIMITED
        
        # Should handle gracefully
        result = await connector.search_content("test", "repository")
        
        assert result == []
        assert connector.status == ConnectorStatus.RATE_LIMITED


# === API Tests ===

class TestIntegrationAPI:
    """Test integration API endpoints"""
    
    @pytest.fixture
    def test_client(self):
        """Create test client for API endpoints"""
        from fastapi import FastAPI
        from prsm.core.integrations.api.integration_api import integration_router
        
        app = FastAPI()
        app.include_router(integration_router)
        
        return TestClient(app)
    
    def test_health_endpoint(self, test_client):
        """Test health endpoint"""
        with patch('prsm.integrations.core.integration_manager.integration_manager') as mock_manager:
            mock_manager.get_system_health = AsyncMock(return_value={
                "overall_status": "healthy",
                "health_percentage": 95.0,
                "connectors": {"total": 2, "healthy": 2},
                "imports": {"active": 0, "total": 10},
                "last_health_check": None,
                "sandbox_status": {"status": "idle"}
            })
            
            response = test_client.get("/integrations/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["overall_status"] == "healthy"
            assert data["health_percentage"] == 95.0
    
    def test_search_endpoint(self, test_client):
        """Test content search endpoint"""
        with patch('prsm.integrations.core.integration_manager.integration_manager') as mock_manager:
            mock_source = IntegrationSource(
                platform=IntegrationPlatform.GITHUB,
                external_id="test/repo",
                display_name="test repo",
                owner_id="test",
                url="https://github.com/test/repo"
            )
            
            mock_manager.search_content = AsyncMock(return_value=[mock_source])
            
            search_data = {
                "query": "test",
                "content_type": "repository",
                "limit": 10
            }
            
            response = test_client.post("/integrations/search", json=search_data)
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["external_id"] == "test/repo"


# === Test Configuration ===

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])