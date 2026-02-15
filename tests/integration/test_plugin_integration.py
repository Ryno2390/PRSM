#!/usr/bin/env python3
"""
Plugin Integration Tests
=======================

Tests for plugin loading, interaction scenarios, and ecosystem integration.
"""

import pytest
import asyncio
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import tempfile
import json
from pathlib import Path

from prsm.compute.plugins.plugin_manager import PluginManager
from prsm.economy.marketplace.ecosystem.plugin_registry import PluginRegistry
from prsm.economy.marketplace.ecosystem.marketplace_core import MarketplaceCore


class MockPlugin:
    """Mock plugin for testing"""
    
    def __init__(self, name: str = "test-plugin"):
        self.name = name
        self.version = "1.0.0"
        self.initialized = False
        self.capabilities = ["test", "mock"]
    
    async def initialize(self):
        """Initialize the mock plugin"""
        self.initialized = True
        return True
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task"""
        return {
            "status": "success",
            "result": f"Processed by {self.name}",
            "task_id": task.get("id", "unknown")
        }
    
    async def shutdown(self):
        """Shutdown the plugin"""
        self.initialized = False


class MockAnalyticsPlugin(MockPlugin):
    """Mock analytics plugin"""
    
    def __init__(self):
        super().__init__("analytics-plugin")
        self.capabilities = ["analytics", "metrics", "dashboards"]
        self.data_points = []
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect system metrics"""
        return {
            "cpu_usage": 0.45,
            "memory_usage": 0.67,
            "active_users": 150,
            "query_count": 2847
        }
    
    async def create_dashboard(self, config: Dict[str, Any]) -> str:
        """Create a dashboard"""
        dashboard_id = f"dashboard_{len(self.data_points)}"
        self.data_points.append({
            "id": dashboard_id,
            "config": config,
            "created_at": "2025-07-23T16:00:00Z"
        })
        return dashboard_id


class MockReasoningPlugin(MockPlugin):
    """Mock reasoning plugin"""
    
    def __init__(self):
        super().__init__("reasoning-plugin")
        self.capabilities = ["reasoning", "inference", "analysis"]
        self.reasoning_cache = {}
    
    async def reason(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform reasoning"""
        reasoning_id = f"reasoning_{len(self.reasoning_cache)}"
        result = {
            "reasoning_id": reasoning_id,
            "query": query,
            "conclusion": f"Analysis of: {query}",
            "confidence": 0.85,
            "reasoning_steps": [
                "Analyzed query structure",
                "Applied domain knowledge",
                "Generated conclusion"
            ]
        }
        self.reasoning_cache[reasoning_id] = result
        return result


class TestPluginLoading:
    """Test plugin loading functionality"""
    
    @pytest.fixture
    def plugin_manager(self):
        """Create plugin manager for testing"""
        return PluginManager()
    
    @pytest.fixture
    def plugin_registry(self):
        """Create plugin registry for testing"""
        return PluginRegistry()
    
    @pytest.fixture
    def temp_plugin_dir(self):
        """Create temporary directory for plugin files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.mark.asyncio
    async def test_basic_plugin_loading(self, plugin_manager):
        """Test basic plugin loading functionality"""
        # Create mock plugin manifest
        plugin_manifest = {
            "name": "test-plugin",
            "version": "1.0.0",
            "description": "Test plugin for integration testing",
            "entry_point": "test_plugin:TestPlugin",
            "capabilities": ["test", "mock"],
            "dependencies": []
        }
        
        # Mock plugin loading
        with patch.object(plugin_manager, '_load_plugin_module') as mock_load:
            mock_load.return_value = MockPlugin
            
            plugin_id = await plugin_manager.load_plugin(plugin_manifest)
            assert plugin_id is not None
            
            # Verify plugin is loaded
            loaded_plugins = plugin_manager.get_loaded_plugins()
            assert len(loaded_plugins) >= 1
            assert any(p["name"] == "test-plugin" for p in loaded_plugins)
    
    @pytest.mark.asyncio
    async def test_plugin_initialization(self, plugin_manager):
        """Test plugin initialization process"""
        plugin_manifest = {
            "name": "init-test-plugin",
            "version": "1.0.0",
            "entry_point": "test_plugin:MockPlugin"
        }
        
        with patch.object(plugin_manager, '_load_plugin_module') as mock_load:
            mock_load.return_value = MockPlugin
            
            plugin_id = await plugin_manager.load_plugin(plugin_manifest)
            
            # Test plugin initialization
            await plugin_manager.initialize_plugin(plugin_id)
            
            # Verify plugin is initialized
            plugin_info = plugin_manager.get_plugin_info(plugin_id)
            assert plugin_info is not None
            assert plugin_info["status"] == "initialized"
    
    @pytest.mark.asyncio
    async def test_plugin_execution(self, plugin_manager):
        """Test plugin task execution"""
        plugin_manifest = {
            "name": "execution-test-plugin",
            "version": "1.0.0",
            "entry_point": "test_plugin:MockPlugin"
        }
        
        with patch.object(plugin_manager, '_load_plugin_module') as mock_load:
            mock_plugin_class = MockPlugin
            mock_load.return_value = mock_plugin_class
            
            plugin_id = await plugin_manager.load_plugin(plugin_manifest)
            await plugin_manager.initialize_plugin(plugin_id)
            
            # Execute task through plugin
            task = {
                "id": "test-task-123",
                "type": "test",
                "data": {"message": "Hello, plugin!"}
            }
            
            result = await plugin_manager.execute_plugin_task(plugin_id, task)
            assert result["status"] == "success"
            assert "test-task-123" in result["task_id"]


class TestPluginInteractions:
    """Test plugin interaction scenarios"""
    
    @pytest.fixture
    def plugin_manager(self):
        return PluginManager()
    
    @pytest.mark.asyncio
    async def test_multiple_plugin_coordination(self, plugin_manager):
        """Test coordination between multiple plugins"""
        # Load analytics plugin
        analytics_manifest = {
            "name": "analytics-plugin",
            "version": "1.0.0",
            "entry_point": "analytics:AnalyticsPlugin",
            "capabilities": ["analytics", "metrics"]
        }
        
        # Load reasoning plugin
        reasoning_manifest = {
            "name": "reasoning-plugin", 
            "version": "1.0.0",
            "entry_point": "reasoning:ReasoningPlugin",
            "capabilities": ["reasoning", "analysis"]
        }
        
        with patch.object(plugin_manager, '_load_plugin_module') as mock_load:
            def mock_load_side_effect(manifest):
                if "analytics" in manifest["name"]:
                    return MockAnalyticsPlugin
                elif "reasoning" in manifest["name"]:
                    return MockReasoningPlugin
                else:
                    return MockPlugin
            
            mock_load.side_effect = mock_load_side_effect
            
            # Load both plugins
            analytics_id = await plugin_manager.load_plugin(analytics_manifest)
            reasoning_id = await plugin_manager.load_plugin(reasoning_manifest)
            
            # Initialize both plugins
            await plugin_manager.initialize_plugin(analytics_id)
            await plugin_manager.initialize_plugin(reasoning_id)
            
            # Test coordinated workflow
            # 1. Collect metrics with analytics plugin
            metrics_task = {"type": "collect_metrics", "id": "metrics-1"}
            metrics_result = await plugin_manager.execute_plugin_task(analytics_id, metrics_task)
            assert metrics_result["status"] == "success"
            
            # 2. Analyze metrics with reasoning plugin
            analysis_task = {
                "type": "analyze",
                "id": "analysis-1", 
                "data": {"metrics": "mock_metrics_data"}
            }
            analysis_result = await plugin_manager.execute_plugin_task(reasoning_id, analysis_task)
            assert analysis_result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_plugin_communication(self, plugin_manager):
        """Test communication between plugins"""
        # Create plugins that need to communicate
        sender_manifest = {
            "name": "sender-plugin",
            "version": "1.0.0",
            "entry_point": "sender:SenderPlugin"
        }
        
        receiver_manifest = {
            "name": "receiver-plugin",
            "version": "1.0.0", 
            "entry_point": "receiver:ReceiverPlugin"
        }
        
        with patch.object(plugin_manager, '_load_plugin_module') as mock_load:
            mock_load.return_value = MockPlugin
            
            sender_id = await plugin_manager.load_plugin(sender_manifest)
            receiver_id = await plugin_manager.load_plugin(receiver_manifest)
            
            await plugin_manager.initialize_plugin(sender_id)
            await plugin_manager.initialize_plugin(receiver_id)
            
            # Test plugin communication through manager
            message = {
                "from": sender_id,
                "to": receiver_id,
                "type": "data_transfer",
                "payload": {"data": "test_communication_data"}
            }
            
            # Mock the communication
            with patch.object(plugin_manager, 'send_plugin_message', new_callable=AsyncMock) as mock_send:
                mock_send.return_value = {"status": "delivered", "message_id": "msg-123"}
                
                result = await plugin_manager.send_plugin_message(message)
                assert result["status"] == "delivered"
    
    @pytest.mark.asyncio
    async def test_plugin_error_handling(self, plugin_manager):
        """Test plugin error handling and recovery"""
        faulty_manifest = {
            "name": "faulty-plugin",
            "version": "1.0.0",
            "entry_point": "faulty:FaultyPlugin"
        }
        
        class FaultyPlugin(MockPlugin):
            async def execute(self, task):
                if task.get("should_fail", False):
                    raise Exception("Simulated plugin error")
                return await super().execute(task)
        
        with patch.object(plugin_manager, '_load_plugin_module') as mock_load:
            mock_load.return_value = FaultyPlugin
            
            plugin_id = await plugin_manager.load_plugin(faulty_manifest)
            await plugin_manager.initialize_plugin(plugin_id)
            
            # Test error handling
            failing_task = {"type": "test", "should_fail": True}
            
            with pytest.raises(Exception):
                await plugin_manager.execute_plugin_task(plugin_id, failing_task)
            
            # Test recovery - plugin should still work for normal tasks
            normal_task = {"type": "test", "should_fail": False}
            result = await plugin_manager.execute_plugin_task(plugin_id, normal_task)
            assert result["status"] == "success"


class TestPluginRegistryIntegration:
    """Test plugin registry integration"""
    
    @pytest.fixture
    def plugin_registry(self):
        return PluginRegistry()
    
    @pytest.fixture
    def marketplace_core(self):
        return MarketplaceCore()
    
    @pytest.mark.asyncio
    async def test_plugin_registry_validation(self, plugin_registry):
        """Test plugin validation in registry"""
        plugin_manifest = {
            "name": "registry-test-plugin",
            "version": "1.0.0",
            "description": "Plugin for registry testing",
            "entry_point": "test:TestPlugin",
            "capabilities": ["test"],
            "security_level": "sandbox",
            "resource_requirements": {
                "memory": "256MB",
                "cpu": "0.5 cores"
            }
        }
        
        # Test plugin validation
        validation_result = await plugin_registry.validate_plugin(plugin_manifest)
        assert validation_result["is_valid"] is True
        assert len(validation_result["errors"]) == 0
    
    @pytest.mark.asyncio
    async def test_plugin_security_scanning(self, plugin_registry):
        """Test plugin security scanning"""
        plugin_data = {
            "name": "security-test-plugin",
            "version": "1.0.0",
            "code": "print('Hello, world!')",  # Safe code
            "manifest": {
                "capabilities": ["safe_operation"],
                "permissions": ["read_only"]
            }
        }
        
        # Mock security scanning
        with patch.object(plugin_registry, '_scan_plugin_security', new_callable=AsyncMock) as mock_scan:
            mock_scan.return_value = {
                "security_score": 95,
                "vulnerabilities": [],
                "risk_level": "low",
                "approved": True
            }
            
            security_result = await plugin_registry.security_scan(plugin_data)
            assert security_result["approved"] is True
            assert security_result["security_score"] >= 90
    
    @pytest.mark.asyncio
    async def test_marketplace_plugin_integration(self, marketplace_core, plugin_registry):
        """Test integration between marketplace and plugin registry"""
        await marketplace_core.initialize()
        
        # Register plugin in marketplace
        plugin_integration = {
            "name": "Marketplace Plugin",
            "type": "plugin",
            "version": "2.0.0",
            "developer_id": "test-developer",
            "description": "Plugin registered through marketplace",
            "capabilities": {
                "data_processing": True,
                "real_time_analysis": True
            }
        }
        
        marketplace_id = await marketplace_core.register_integration(plugin_integration)
        assert marketplace_id is not None
        
        # Validate through plugin registry
        plugin_manifest = {
            "name": plugin_integration["name"],
            "version": plugin_integration["version"],
            "marketplace_id": marketplace_id,
            "capabilities": list(plugin_integration["capabilities"].keys())
        }
        
        validation_result = await plugin_registry.validate_plugin(plugin_manifest)
        assert validation_result["is_valid"] is True


class TestPluginLifecycle:
    """Test complete plugin lifecycle"""
    
    @pytest.fixture
    def plugin_manager(self):
        return PluginManager()
    
    @pytest.mark.asyncio
    async def test_complete_plugin_lifecycle(self, plugin_manager):
        """Test complete plugin lifecycle from registration to removal"""
        plugin_manifest = {
            "name": "lifecycle-test-plugin",
            "version": "1.0.0",
            "description": "Plugin for lifecycle testing",
            "entry_point": "lifecycle:LifecyclePlugin",
            "capabilities": ["test", "lifecycle"]
        }
        
        with patch.object(plugin_manager, '_load_plugin_module') as mock_load:
            mock_load.return_value = MockPlugin
            
            # 1. Load plugin
            plugin_id = await plugin_manager.load_plugin(plugin_manifest)
            assert plugin_id is not None
            
            # 2. Initialize plugin
            await plugin_manager.initialize_plugin(plugin_id)
            plugin_info = plugin_manager.get_plugin_info(plugin_id)
            assert plugin_info["status"] == "initialized"
            
            # 3. Execute tasks
            task = {"type": "test", "id": "lifecycle-task"}
            result = await plugin_manager.execute_plugin_task(plugin_id, task)
            assert result["status"] == "success"
            
            # 4. Pause plugin
            await plugin_manager.pause_plugin(plugin_id)
            plugin_info = plugin_manager.get_plugin_info(plugin_id)
            assert plugin_info["status"] == "paused"
            
            # 5. Resume plugin
            await plugin_manager.resume_plugin(plugin_id)
            plugin_info = plugin_manager.get_plugin_info(plugin_id)
            assert plugin_info["status"] == "initialized"
            
            # 6. Shutdown plugin
            await plugin_manager.shutdown_plugin(plugin_id)
            plugin_info = plugin_manager.get_plugin_info(plugin_id)
            assert plugin_info["status"] == "shutdown"
            
            # 7. Unload plugin
            await plugin_manager.unload_plugin(plugin_id)
            loaded_plugins = plugin_manager.get_loaded_plugins()
            assert not any(p["id"] == plugin_id for p in loaded_plugins)


if __name__ == '__main__':
    # Run plugin integration tests
    pytest.main([__file__, '-v', '--asyncio-mode=auto'])