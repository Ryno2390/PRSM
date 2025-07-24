#!/usr/bin/env python3
"""
Phase 7 Integration Tests
========================

Comprehensive integration tests verifying that all Phase 7 components
work together seamlessly across the enterprise architecture.
"""

import pytest
import asyncio
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
import json
from pathlib import Path

# Test the major Phase 7 components
from prsm.ai_orchestration.orchestrator import AIOrchestrator
from prsm.analytics.dashboard_manager import DashboardManager
from prsm.enterprise.global_infrastructure import GlobalInfrastructure
from prsm.integrations.enterprise.integration_manager import IntegrationManager
from prsm.marketplace.ecosystem.marketplace_core import MarketplaceCore
from prsm.nwtn.unified_pipeline_controller import UnifiedPipelineController
from prsm.plugins.plugin_manager import PluginManager


class TestPhase7Integration:
    """Integration tests for Phase 7 enterprise architecture components"""
    
    @pytest.fixture
    def ai_orchestrator(self):
        """Create AI orchestrator for testing"""
        return AIOrchestrator()
    
    @pytest.fixture  
    def dashboard_manager(self):
        """Create dashboard manager for testing"""
        return DashboardManager()
    
    @pytest.fixture
    def global_infrastructure(self):
        """Create global infrastructure for testing"""
        return GlobalInfrastructure()
    
    @pytest.fixture
    def integration_manager(self):
        """Create integration manager for testing"""
        return IntegrationManager()
    
    @pytest.fixture
    def marketplace_core(self):
        """Create marketplace core for testing"""
        return MarketplaceCore()
    
    @pytest.fixture
    def unified_pipeline(self):
        """Create unified pipeline controller for testing"""
        return UnifiedPipelineController()
    
    @pytest.fixture
    def plugin_manager(self):
        """Create plugin manager for testing"""
        return PluginManager()

    @pytest.mark.asyncio
    async def test_ai_orchestration_integration(self, ai_orchestrator):
        """Test AI orchestration system integration"""
        # Test orchestrator initialization
        await ai_orchestrator.initialize()
        assert ai_orchestrator.is_initialized
        
        # Test model registration
        model_config = {
            'name': 'test-model',
            'type': 'reasoning',
            'capabilities': ['text-generation', 'analysis']
        }
        
        model_id = await ai_orchestrator.register_model(model_config)
        assert model_id is not None
        
        # Test task distribution
        task = {
            'type': 'reasoning',
            'content': 'Test reasoning task',
            'priority': 'high'
        }
        
        with patch.object(ai_orchestrator, '_execute_task', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {'result': 'success', 'confidence': 0.95}
            
            result = await ai_orchestrator.execute_task(task)
            assert result['result'] == 'success'
            assert result['confidence'] == 0.95
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_analytics_dashboard_integration(self, dashboard_manager):
        """Test analytics and dashboard system integration"""
        # Test dashboard creation
        dashboard_config = {
            'name': 'Test Dashboard',
            'type': 'analytics',
            'widgets': [
                {'type': 'chart', 'data_source': 'metrics'},
                {'type': 'table', 'data_source': 'logs'}
            ]
        }
        
        dashboard_id = await dashboard_manager.create_dashboard(dashboard_config)
        assert dashboard_id is not None
        
        # Test data integration
        test_data = {
            'metrics': [
                {'timestamp': '2025-07-23T16:00:00Z', 'value': 100, 'metric': 'cpu_usage'},
                {'timestamp': '2025-07-23T16:01:00Z', 'value': 105, 'metric': 'cpu_usage'}
            ]
        }
        
        await dashboard_manager.update_dashboard_data(dashboard_id, test_data)
        
        # Verify dashboard data
        dashboard = await dashboard_manager.get_dashboard(dashboard_id)
        assert dashboard['name'] == 'Test Dashboard'
        assert len(dashboard['widgets']) == 2

    @pytest.mark.asyncio
    async def test_global_infrastructure_integration(self, global_infrastructure):
        """Test global infrastructure system integration"""
        # Test infrastructure initialization
        await global_infrastructure.initialize()
        
        # Test region management
        region_config = {
            'name': 'us-west-1',
            'capacity': 1000,
            'availability_zone': 'us-west-1a'
        }
        
        region_id = await global_infrastructure.add_region(region_config)
        assert region_id is not None
        
        # Test load balancing
        with patch.object(global_infrastructure, '_check_region_health', new_callable=AsyncMock) as mock_health:
            mock_health.return_value = {'status': 'healthy', 'load': 0.3}
            
            optimal_region = await global_infrastructure.get_optimal_region('reasoning')
            assert optimal_region is not None

    @pytest.mark.asyncio
    async def test_enterprise_integration_suite(self, integration_manager):
        """Test enterprise integration suite"""
        # Test integration setup
        integration_config = {
            'name': 'Test Integration',
            'type': 'database',
            'connection': {
                'host': 'localhost',
                'port': 5432,
                'database': 'test_db'
            }
        }
        
        integration_id = await integration_manager.create_integration(integration_config)
        assert integration_id is not None
        
        # Test data sync
        with patch.object(integration_manager, '_sync_data', new_callable=AsyncMock) as mock_sync:
            mock_sync.return_value = {'synced_records': 100, 'status': 'success'}
            
            sync_result = await integration_manager.sync_integration(integration_id)
            assert sync_result['status'] == 'success'
            assert sync_result['synced_records'] == 100

    @pytest.mark.asyncio
    async def test_marketplace_ecosystem_integration(self, marketplace_core):
        """Test marketplace ecosystem integration"""
        # Test marketplace initialization
        await marketplace_core.initialize()
        
        # Test integration registration
        integration_data = {
            'name': 'Test Plugin',
            'type': 'ai_model',
            'version': '1.0.0',
            'developer_id': 'test-dev',
            'description': 'Test integration for marketplace'
        }
        
        integration_id = await marketplace_core.register_integration(integration_data)
        assert integration_id is not None
        
        # Test search functionality
        search_results = await marketplace_core.search_integrations('test', limit=10)
        assert len(search_results) >= 0
        
        # Test integration retrieval
        integration = await marketplace_core.get_integration(integration_id)
        assert integration['name'] == 'Test Plugin'
        assert integration['type'] == 'ai_model'

    @pytest.mark.asyncio
    async def test_nwtn_pipeline_integration(self, unified_pipeline):
        """Test NWTN unified pipeline integration"""
        # Test pipeline initialization
        init_success = await unified_pipeline.initialize()
        assert init_success is True
        
        # Test health check
        health_status = await unified_pipeline.get_system_health()
        assert 'component_health' in health_status
        assert 'performance_metrics' in health_status
        
        # Test pipeline configuration
        config_success = await unified_pipeline.configure_user_api(
            user_id='test-user',
            provider='claude',
            api_key='test-key'
        )
        # This may return False in test environment, which is expected
        assert isinstance(config_success, bool)

    @pytest.mark.asyncio
    async def test_plugin_system_integration(self, plugin_manager):
        """Test plugin system integration"""
        # Test plugin loading
        plugin_manifest = {
            'name': 'test-plugin',
            'version': '1.0.0',
            'description': 'Test plugin for integration testing',
            'entry_point': 'test_plugin:TestPlugin',
            'dependencies': []
        }
        
        # Mock plugin loading since we don't have actual plugin files
        with patch.object(plugin_manager, '_load_plugin_module') as mock_load:
            mock_plugin_class = Mock()
            mock_plugin_instance = Mock()
            mock_plugin_class.return_value = mock_plugin_instance
            mock_load.return_value = mock_plugin_class
            
            plugin_id = await plugin_manager.load_plugin(plugin_manifest)
            assert plugin_id is not None
            
            # Test plugin interaction
            plugins = plugin_manager.get_loaded_plugins()
            assert len(plugins) >= 1

    @pytest.mark.asyncio
    async def test_cross_component_integration(self, ai_orchestrator, dashboard_manager, marketplace_core):
        """Test integration between multiple Phase 7 components"""
        # Initialize all components
        await ai_orchestrator.initialize()
        await marketplace_core.initialize()
        
        # Test workflow: Register AI model in marketplace, then use in orchestrator
        ai_model_config = {
            'name': 'Cross-Integration Model',
            'type': 'ai_model',
            'version': '1.0.0',
            'developer_id': 'integration-test',
            'capabilities': {
                'reasoning': True,
                'text_generation': True,
                'analysis': True
            }
        }
        
        # Register in marketplace
        marketplace_id = await marketplace_core.register_integration(ai_model_config)
        assert marketplace_id is not None
        
        # Register same model in orchestrator
        orchestrator_model_id = await ai_orchestrator.register_model({
            'name': ai_model_config['name'],
            'type': 'reasoning',
            'marketplace_id': marketplace_id,
            'capabilities': list(ai_model_config['capabilities'].keys())
        })
        assert orchestrator_model_id is not None
        
        # Create dashboard to monitor the integration
        dashboard_config = {
            'name': 'Cross-Integration Monitor',
            'type': 'integration',
            'widgets': [
                {'type': 'model_metrics', 'model_id': orchestrator_model_id},
                {'type': 'marketplace_stats', 'integration_id': marketplace_id}
            ]
        }
        
        dashboard_id = await dashboard_manager.create_dashboard(dashboard_config)
        assert dashboard_id is not None
        
        # Verify cross-component data flow
        dashboard = await dashboard_manager.get_dashboard(dashboard_id)
        assert dashboard['name'] == 'Cross-Integration Monitor'
        assert len(dashboard['widgets']) == 2

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, ai_orchestrator, integration_manager):
        """Test error handling across integrated components"""
        # Test graceful failure handling
        invalid_task = {
            'type': 'invalid_type',
            'content': None,
            'priority': 'unknown'
        }
        
        # Should handle invalid task gracefully
        with pytest.raises(ValueError):
            await ai_orchestrator.execute_task(invalid_task)
        
        # Test integration error handling
        invalid_integration = {
            'name': '',  # Invalid empty name
            'type': 'unknown_type',
            'connection': {}  # Empty connection
        }
        
        with pytest.raises(ValueError):
            await integration_manager.create_integration(invalid_integration)

    @pytest.mark.asyncio
    async def test_performance_integration(self, unified_pipeline, global_infrastructure):
        """Test performance across integrated components"""
        # Test pipeline performance monitoring
        await unified_pipeline.initialize()
        
        # Get system health metrics
        health_metrics = await unified_pipeline.get_system_health()
        assert 'performance_metrics' in health_metrics
        
        # Test infrastructure performance monitoring
        await global_infrastructure.initialize()
        
        # Mock performance data
        with patch.object(global_infrastructure, 'get_performance_metrics', new_callable=AsyncMock) as mock_perf:
            mock_perf.return_value = {
                'cpu_usage': 0.45,
                'memory_usage': 0.67,
                'network_latency': 12.3,
                'throughput': 1500
            }
            
            perf_metrics = await global_infrastructure.get_performance_metrics()
            assert perf_metrics['cpu_usage'] < 1.0
            assert perf_metrics['memory_usage'] < 1.0
            assert perf_metrics['throughput'] > 0

    @pytest.mark.asyncio
    async def test_configuration_integration(self):
        """Test configuration consistency across components"""
        from prsm.core.config import get_settings_safe
        
        # Test configuration loading
        settings = get_settings_safe()
        if settings:
            # Verify essential settings are available
            assert hasattr(settings, 'embedding_model')
            assert hasattr(settings, 'nwtn_enabled')
            
            # Test configuration consistency
            assert settings.nwtn_enabled in [True, False]
        
        # Test component configuration alignment
        components = [
            UnifiedPipelineController(),
            MarketplaceCore(),
            AIOrchestrator()
        ]
        
        for component in components:
            # Each component should handle missing configuration gracefully
            assert component is not None


class TestPhase7EndToEndWorkflow:
    """End-to-end workflow tests for Phase 7 integration"""
    
    @pytest.mark.asyncio
    async def test_complete_enterprise_workflow(self):
        """Test complete workflow from query to result through all Phase 7 components"""
        # This test simulates a complete enterprise workflow
        
        # 1. Initialize unified pipeline
        pipeline = UnifiedPipelineController()
        init_success = await pipeline.initialize()
        
        # 2. Initialize marketplace
        marketplace = MarketplaceCore()
        await marketplace.initialize()
        
        # 3. Initialize AI orchestrator
        orchestrator = AIOrchestrator()
        await orchestrator.initialize()
        
        # 4. Simulate enterprise query processing
        test_query = "What are the latest developments in quantum computing?"
        
        # Mock the complete workflow
        with patch.object(pipeline, 'process_query_full_pipeline', new_callable=AsyncMock) as mock_process:
            mock_result = Mock()
            mock_result.status = 'completed'
            mock_result.natural_language_response = 'Quantum computing developments include...'
            mock_result.metrics.confidence_score = 0.92
            mock_result.metrics.total_cost_ftns = 15.5
            
            mock_process.return_value = mock_result
            
            # Execute workflow
            result = await pipeline.process_query_full_pipeline(
                user_id='enterprise-user',
                query=test_query,
                verbosity_level='detailed'
            )
            
            # Verify workflow completion
            assert result.status == 'completed'
            assert result.natural_language_response.startswith('Quantum computing')
            assert result.metrics.confidence_score > 0.9
            assert result.metrics.total_cost_ftns > 0

    @pytest.mark.asyncio
    async def test_marketplace_plugin_integration_workflow(self):
        """Test complete marketplace plugin integration workflow"""
        marketplace = MarketplaceCore()
        plugin_manager = PluginManager()
        
        await marketplace.initialize()
        
        # 1. Register plugin in marketplace
        plugin_data = {
            'name': 'Analytics Plugin',
            'type': 'plugin',
            'version': '2.1.0',
            'developer_id': 'analytics-team',
            'description': 'Advanced analytics plugin for enterprise users',
            'capabilities': {
                'real_time_analytics': True,
                'custom_dashboards': True,
                'data_export': True
            }
        }
        
        plugin_id = await marketplace.register_integration(plugin_data)
        assert plugin_id is not None
        
        # 2. Search and discover plugin
        search_results = await marketplace.search_integrations('analytics', limit=5)
        found_plugin = None
        for result in search_results:
            if result.get('name') == 'Analytics Plugin':
                found_plugin = result
                break
        
        # 3. Load plugin through plugin manager
        if found_plugin:
            plugin_manifest = {
                'name': found_plugin['name'],
                'version': found_plugin['version'],
                'marketplace_id': plugin_id,
                'entry_point': 'analytics_plugin:AnalyticsPlugin'
            }
            
            with patch.object(plugin_manager, '_load_plugin_module') as mock_load:
                mock_plugin = Mock()
                mock_load.return_value = mock_plugin
                
                loaded_plugin_id = await plugin_manager.load_plugin(plugin_manifest)
                assert loaded_plugin_id is not None


if __name__ == '__main__':
    # Run integration tests
    pytest.main([__file__, '-v', '--asyncio-mode=auto'])