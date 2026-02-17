"""
Ollama Connector Test Suite
===========================

Comprehensive test suite for the Ollama connector, covering:
- Local Ollama instance connection
- Model discovery and management
- Content search and metadata retrieval
- Model pulling and removal operations
- Health monitoring and status tracking

Test Categories:
- Unit tests for core functionality
- Integration tests with mock Ollama API
- Error handling and edge cases
- Performance and reliability testing
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

try:
    import aiohttp
    from aioresponses import aioresponses

    from prsm.core.integrations.connectors.ollama_connector import OllamaConnector, OllamaModelInfo
    from prsm.core.integrations.core.base_connector import ConnectorStatus
    from prsm.core.integrations.models.integration_models import (
        IntegrationPlatform, ConnectorConfig, IntegrationSource
    )
except (ImportError, ModuleNotFoundError) as e:
    pytest.skip("Module 'aioresponses' not yet available", allow_module_level=True)


# === Test Fixtures ===

@pytest.fixture
def ollama_config():
    """Mock Ollama connector configuration"""
    return ConnectorConfig(
        platform=IntegrationPlatform.OLLAMA,
        user_id="test_user",
        custom_settings={
            "base_url": "http://localhost:11434",
            "timeout": 30
        }
    )


@pytest.fixture
def sample_ollama_models():
    """Sample Ollama model list response"""
    return {
        "models": [
            {
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
            },
            {
                "name": "codellama:13b",
                "size": 7365960935,
                "digest": "sha256:9a110e10",
                "modified_at": "2024-01-14T15:45:00Z",
                "details": {
                    "family": "llama",
                    "families": ["llama"],
                    "format": "gguf",
                    "parameter_size": "13B",
                    "quantization_level": "Q4_0"
                }
            }
        ]
    }


@pytest.fixture
def sample_model_info():
    """Sample model info response"""
    return {
        "template": "{{ .System }}\n\n{{ .Prompt }}",
        "parameters": {
            "stop": ["<|im_end|>"],
            "temperature": 0.8,
            "top_p": 0.9
        },
        "model_info": {
            "general.architecture": "llama",
            "general.parameter_count": 6738415616,
            "general.quantization_version": 2,
            "general.file_type": 2
        },
        "details": {
            "family": "llama",
            "families": ["llama"],
            "format": "gguf",
            "parameter_size": "7B",
            "quantization_level": "Q4_0"
        },
        "license": "Apache License 2.0\n\nCopyright (c) Meta Platforms, Inc. and affiliates.",
        "system": "You are a helpful assistant."
    }


@pytest.fixture
def mock_version_response():
    """Mock Ollama version response"""
    return {"version": "0.1.17"}


# === Unit Tests - Core Functionality ===

class TestOllamaConnector:
    """Test Ollama connector core functionality"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, ollama_config):
        """Test connector initialization"""
        connector = OllamaConnector(ollama_config)
        
        assert connector.platform == IntegrationPlatform.OLLAMA
        assert connector.user_id == "test_user"
        assert connector.base_url == "http://localhost:11434"
        assert connector.timeout == 30
        assert connector.status == ConnectorStatus.INITIALIZING
    
    @pytest.mark.asyncio
    async def test_connection_success(self, ollama_config, mock_version_response):
        """Test successful connection to Ollama"""
        connector = OllamaConnector(ollama_config)

        with patch.object(connector, '_test_connection', new_callable=AsyncMock) as mock_conn:
            mock_conn.return_value = True
            with patch.object(connector, '_load_initial_data', new_callable=AsyncMock):
                # Simulate what _test_connection sets on success
                async def test_conn_side_effect():
                    connector.ollama_version = "0.1.17"
                    return True
                mock_conn.side_effect = test_conn_side_effect

                success = await connector.initialize()

                assert success is True
                assert connector.status == ConnectorStatus.CONNECTED
                assert connector.ollama_version == "0.1.17"
    
    @pytest.mark.asyncio
    async def test_connection_failure(self, ollama_config):
        """Test failed connection to Ollama"""
        connector = OllamaConnector(ollama_config)
        
        with aioresponses() as m:
            m.get(f"{connector.base_url}/api/version", status=500)
            
            success = await connector.initialize()
            
            assert success is False
            assert connector.status == ConnectorStatus.DISCONNECTED
    
    @pytest.mark.asyncio
    async def test_authentication_success(self, ollama_config, mock_version_response):
        """Test successful authentication (no auth required)"""
        connector = OllamaConnector(ollama_config)

        with patch.object(connector, '_test_connection', new_callable=AsyncMock) as mock_conn:
            mock_conn.return_value = True

            result = await connector.authenticate()

            assert result is True
            assert connector.authenticated_user == "local"
    
    @pytest.mark.asyncio
    async def test_model_list_parsing(self, ollama_config, sample_ollama_models):
        """Test parsing of Ollama model list"""
        connector = OllamaConnector(ollama_config)
        connector.session = MagicMock()
        
        with patch.object(connector, '_make_api_request', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = sample_ollama_models
            
            await connector._refresh_available_models()
            
            assert len(connector.available_models) == 2
            
            # Check first model
            model1 = connector.available_models[0]
            assert model1.name == "llama2:7b"
            assert model1.tag == "7b"
            assert model1.size == 3825819519
            assert model1.details["parameter_size"] == "7B"
            
            # Check second model
            model2 = connector.available_models[1]
            assert model2.name == "codellama:13b"
            assert model2.tag == "13b"
            assert model2.size == 7365960935


# === Content Discovery Tests ===

class TestOllamaSearch:
    """Test Ollama content search functionality"""
    
    @pytest.mark.asyncio
    async def test_search_models(self, ollama_config, sample_ollama_models):
        """Test searching for models"""
        connector = OllamaConnector(ollama_config)
        
        with patch.object(connector, '_refresh_available_models', new_callable=AsyncMock) as mock_refresh:
            # Set up model data
            connector.available_models = [
                OllamaModelInfo(
                    name="llama2:7b",
                    tag="7b",
                    size=3825819519,
                    digest="sha256:8fdf52f7",
                    details={"parameter_size": "7B", "family": "llama"}
                ),
                OllamaModelInfo(
                    name="codellama:13b", 
                    tag="13b",
                    size=7365960935,
                    digest="sha256:9a110e10",
                    details={"parameter_size": "13B", "family": "llama"}
                )
            ]
            
            # Search for "llama" models
            results = await connector.search_content("llama", "model", 10)
            
            assert len(results) == 2
            
            # Check result structure
            result1 = results[0]
            assert result1.platform == IntegrationPlatform.OLLAMA
            assert result1.external_id == "llama2:7b"
            assert result1.display_name == "llama2"
            assert result1.owner_id == "local"
            assert result1.metadata["size_gb"] == 3.56  # ~3.56 GB
    
    @pytest.mark.asyncio
    async def test_search_no_results(self, ollama_config):
        """Test search with no matching results"""
        connector = OllamaConnector(ollama_config)
        
        with patch.object(connector, '_refresh_available_models', new_callable=AsyncMock):
            connector.available_models = []
            
            results = await connector.search_content("nonexistent", "model", 10)
            
            assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_search_with_limit(self, ollama_config):
        """Test search with result limit"""
        connector = OllamaConnector(ollama_config)
        
        with patch.object(connector, '_refresh_available_models', new_callable=AsyncMock):
            # Create 5 models
            connector.available_models = [
                OllamaModelInfo(name=f"model{i}:latest", tag="latest") 
                for i in range(5)
            ]
            
            results = await connector.search_content("model", "model", 3)
            
            assert len(results) == 3


# === Model Management Tests ===

class TestOllamaModelManagement:
    """Test Ollama model management operations"""
    
    @pytest.mark.asyncio
    async def test_get_model_metadata(self, ollama_config, sample_model_info):
        """Test getting detailed model metadata"""
        connector = OllamaConnector(ollama_config)
        
        with patch.object(connector, '_make_api_request', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = sample_model_info
            
            # Mock available models
            connector.available_models = [
                OllamaModelInfo(
                    name="llama2:7b",
                    tag="7b", 
                    size=3825819519,
                    digest="sha256:8fdf52f7"
                )
            ]
            
            metadata = await connector.get_content_metadata("llama2:7b")
            
            assert metadata["type"] == "local_model"
            assert metadata["name"] == "llama2:7b"
            assert metadata["platform"] == "ollama"
            assert metadata["size_gb"] == 3.56
            assert "Apache License 2.0" in metadata["license"]
            assert metadata["parameters"]["temperature"] == 0.8
    
    @pytest.mark.asyncio
    async def test_model_pull_success(self, ollama_config):
        """Test successful model pull operation"""
        connector = OllamaConnector(ollama_config)
        connector.session = MagicMock()
        
        # Mock successful pull
        with patch.object(connector, '_stream_pull_request', new_callable=AsyncMock) as mock_pull:
            mock_pull.return_value = True
            
            with patch.object(connector, '_refresh_available_models', new_callable=AsyncMock) as mock_refresh:
                # First call returns no models, second call returns the pulled model
                connector.available_models = []
                
                async def mock_refresh_side_effect():
                    if mock_refresh.call_count == 2:
                        connector.available_models = [
                            OllamaModelInfo(name="new-model:latest", tag="latest")
                        ]
                
                mock_refresh.side_effect = mock_refresh_side_effect
                
                result = await connector.download_content("new-model:latest", "")
                
                assert result is True
    
    @pytest.mark.asyncio
    async def test_model_already_exists(self, ollama_config):
        """Test download when model already exists"""
        connector = OllamaConnector(ollama_config)
        
        with patch.object(connector, '_refresh_available_models', new_callable=AsyncMock):
            connector.available_models = [
                OllamaModelInfo(name="existing-model:latest", tag="latest")
            ]
            
            result = await connector.download_content("existing-model:latest", "")
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_model_removal(self, ollama_config):
        """Test model removal operation"""
        connector = OllamaConnector(ollama_config)
        
        with patch.object(connector, '_make_api_request', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {}  # Empty response indicates success
            
            with patch.object(connector, '_refresh_available_models', new_callable=AsyncMock):
                result = await connector.remove_model("old-model:latest")
                
                assert result is True
                mock_api.assert_called_once_with(
                    "/api/delete",
                    method="DELETE",
                    data={"name": "old-model:latest"}
                )


# === License Validation Tests ===

class TestOllamaLicenseValidation:
    """Test license validation for Ollama models"""
    
    @pytest.mark.asyncio
    async def test_license_validation_with_license(self, ollama_config):
        """Test license validation when license is available"""
        connector = OllamaConnector(ollama_config)
        
        with patch.object(connector, 'get_content_metadata', new_callable=AsyncMock) as mock_meta:
            mock_meta.return_value = {
                "license": "MIT License\n\nPermission is hereby granted..."
            }
            
            license_info = await connector.validate_license("mit-model:latest")
            
            assert license_info["type"] == "permissive"
            assert license_info["compliant"] is True
            assert len(license_info["issues"]) == 0
    
    @pytest.mark.asyncio
    async def test_license_validation_gpl(self, ollama_config):
        """Test license validation for GPL license"""
        connector = OllamaConnector(ollama_config)
        
        with patch.object(connector, 'get_content_metadata', new_callable=AsyncMock) as mock_meta:
            mock_meta.return_value = {
                "license": "GNU General Public License v3.0"
            }
            
            license_info = await connector.validate_license("gpl-model:latest")
            
            assert license_info["type"] == "copyleft"
            assert license_info["compliant"] is False
            assert len(license_info["issues"]) > 0
    
    @pytest.mark.asyncio
    async def test_license_validation_no_license(self, ollama_config):
        """Test license validation when no license info available"""
        connector = OllamaConnector(ollama_config)
        
        with patch.object(connector, 'get_content_metadata', new_callable=AsyncMock) as mock_meta:
            mock_meta.return_value = {"license": ""}
            
            license_info = await connector.validate_license("no-license-model:latest")
            
            assert license_info["type"] == "unknown"
            assert license_info["compliant"] is True  # Local models assumed OK
            assert "No license information available" in license_info["issues"][0]


# === Health Monitoring Tests ===

class TestOllamaHealthMonitoring:
    """Test health monitoring functionality"""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, ollama_config, mock_version_response):
        """Test successful health check"""
        connector = OllamaConnector(ollama_config)
        
        with patch.object(connector, '_make_api_request', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_version_response
            
            with patch.object(connector, '_refresh_available_models', new_callable=AsyncMock):
                connector.available_models = [MagicMock(), MagicMock()]  # 2 models
                
                with patch.object(connector, '_refresh_running_models', new_callable=AsyncMock):
                    connector.running_models = [MagicMock()]  # 1 running
                    
                    health = await connector.health_check()
                    
                    assert health["status"] == "healthy"
                    assert health["platform"] == "ollama"
                    assert health["available_models"] == 2
                    assert health["running_models"] == 1
                    assert health["ollama_version"] == "0.1.17"
                    assert "local_inference" in health["capabilities"]
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, ollama_config):
        """Test health check failure"""
        connector = OllamaConnector(ollama_config)
        
        with patch.object(connector, '_make_api_request', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = None  # Connection failure
            
            health = await connector.health_check()
            
            assert health["status"] == "unhealthy"
            assert health["platform"] == "ollama"
            assert "Failed to connect" in health["error"]
    
    @pytest.mark.asyncio
    async def test_health_check_exception(self, ollama_config):
        """Test health check with exception"""
        connector = OllamaConnector(ollama_config)
        
        with patch.object(connector, '_make_api_request', new_callable=AsyncMock) as mock_api:
            mock_api.side_effect = Exception("Connection timeout")
            
            health = await connector.health_check()
            
            assert health["status"] == "error"
            assert health["platform"] == "ollama"
            assert "Connection timeout" in health["error"]
            assert connector.error_count > 0


# === Integration Tests ===

class TestOllamaIntegration:
    """Test complete Ollama integration workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_model_discovery_workflow(self, ollama_config, sample_ollama_models, sample_model_info):
        """Test complete model discovery and metadata retrieval"""
        connector = OllamaConnector(ollama_config)

        with patch.object(connector, '_test_connection', new_callable=AsyncMock) as mock_conn:
            async def test_conn_side_effect():
                connector.ollama_version = "0.1.17"
                return True
            mock_conn.side_effect = test_conn_side_effect

            with patch.object(connector, '_load_initial_data', new_callable=AsyncMock) as mock_load:
                async def load_side_effect():
                    # Parse models from sample data
                    connector.available_models = [
                        OllamaModelInfo(
                            name=m["name"],
                            tag=m["name"].split(":")[-1] if ":" in m["name"] else "latest",
                            size=m.get("size", 0),
                            digest=m.get("digest", "")
                        )
                        for m in sample_ollama_models.get("models", [])
                    ]
                mock_load.side_effect = load_side_effect

                # Initialize connector
                await connector.initialize()

                # Search for models
                results = await connector.search_content("llama", "model")
                assert len(results) == 2

                with patch.object(connector, '_make_api_request', new_callable=AsyncMock) as mock_api:
                    mock_api.return_value = sample_model_info

                    # Get metadata for first model
                    metadata = await connector.get_content_metadata("llama2:7b")
                    assert metadata["type"] == "local_model"
                    assert metadata["name"] == "llama2:7b"

                    # Validate license
                    license_info = await connector.validate_license("llama2:7b")
                    assert license_info["type"] == "permissive"  # Apache license
    
    @pytest.mark.asyncio
    async def test_model_pull_workflow(self, ollama_config):
        """Test complete model pull workflow"""
        connector = OllamaConnector(ollama_config)
        connector.session = MagicMock()
        
        # Mock model pull stream
        pull_responses = [
            '{"status": "pulling manifest"}\n',
            '{"status": "downloading", "completed": 1000, "total": 10000}\n',
            '{"status": "downloading", "completed": 5000, "total": 10000}\n',
            '{"status": "downloading", "completed": 10000, "total": 10000}\n',
            '{"status": "success"}\n'
        ]
        
        class MockResponse:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            @property
            def content(self):
                class MockContent:
                    async def __aiter__(self_inner):
                        for line in pull_responses:
                            yield line.encode()
                return MockContent()

        def mock_post(*args, **kwargs):
            return MockResponse()
        
        progress_updates = []
        
        def progress_callback(progress, status):
            progress_updates.append((progress, status))
        
        with patch.object(connector.session, 'post', side_effect=mock_post):
            with patch.object(connector, '_refresh_available_models', new_callable=AsyncMock):
                connector.available_models = []  # Initially empty
                
                # After pull, model should be available
                async def refresh_side_effect():
                    if len(progress_updates) > 3:  # After some progress
                        connector.available_models = [
                            OllamaModelInfo(name="new-model:latest", tag="latest")
                        ]
                
                connector._refresh_available_models.side_effect = refresh_side_effect
                
                result = await connector.download_content("new-model:latest", "", progress_callback)
                
                assert result is True
                assert len(progress_updates) > 0
                # Should have progress updates from 10% to 100%
                assert any(update[0] == 100 for update in progress_updates)


# === Performance Tests ===

class TestOllamaPerformance:
    """Test Ollama connector performance characteristics"""
    
    @pytest.mark.asyncio
    async def test_concurrent_model_queries(self, ollama_config):
        """Test concurrent model metadata queries"""
        connector = OllamaConnector(ollama_config)
        
        with patch.object(connector, '_make_api_request', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"name": "test-model", "license": "MIT"}
            
            # Execute multiple concurrent metadata requests
            tasks = [
                connector.get_content_metadata(f"model-{i}:latest")
                for i in range(10)
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 10
            assert all("name" in result for result in results)
    
    @pytest.mark.asyncio
    async def test_large_model_list_handling(self, ollama_config):
        """Test handling of large model lists"""
        connector = OllamaConnector(ollama_config)
        
        # Create large model list
        large_model_list = {
            "models": [
                {
                    "name": f"model-{i}:latest",
                    "size": 1000000 * i,
                    "digest": f"sha256:hash{i}",
                    "modified_at": "2024-01-15T10:30:00Z",
                    "details": {"parameter_size": f"{i}B"}
                }
                for i in range(100)  # 100 models
            ]
        }
        
        with patch.object(connector, '_make_api_request', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = large_model_list
            
            await connector._refresh_available_models()
            
            assert len(connector.available_models) == 100
            
            # Test search performance
            results = await connector.search_content("model", "model", 50)
            assert len(results) == 50


# === Error Handling Tests ===

class TestOllamaErrorHandling:
    """Test error handling in various scenarios"""
    
    @pytest.mark.asyncio
    async def test_network_timeout_handling(self, ollama_config):
        """Test handling of network timeouts"""
        connector = OllamaConnector(ollama_config)
        connector.session = MagicMock()
        
        with patch.object(connector, '_make_api_request', new_callable=AsyncMock) as mock_api:
            mock_api.side_effect = asyncio.TimeoutError("Request timeout")
            
            results = await connector.search_content("test", "model")
            
            assert results == []
    
    @pytest.mark.asyncio
    async def test_invalid_response_handling(self, ollama_config):
        """Test handling of invalid API responses"""
        connector = OllamaConnector(ollama_config)
        
        with patch.object(connector, '_make_api_request', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"invalid": "response"}  # Missing expected fields
            
            # Should handle gracefully without crashing
            await connector._refresh_available_models()
            
            assert len(connector.available_models) == 0
    
    @pytest.mark.asyncio
    async def test_model_not_found_handling(self, ollama_config):
        """Test handling when requested model is not found"""
        connector = OllamaConnector(ollama_config)
        
        with patch.object(connector, '_make_api_request', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = None  # Model not found
            
            metadata = await connector.get_content_metadata("nonexistent:latest")
            
            assert "error" in metadata
            assert "not found" in metadata["error"]


# === Test Configuration ===

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])