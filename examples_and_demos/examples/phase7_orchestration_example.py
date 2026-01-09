#!/usr/bin/env python3
"""
PRSM Phase 7 AI Orchestration Example
====================================

Demonstrates how to use PRSM's advanced AI orchestration platform
for multi-model coordination and intelligent task distribution.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from prsm.ai_orchestration.orchestrator import AIOrchestrator
from prsm.nwtn.unified_pipeline_controller import UnifiedPipelineController
from prsm.marketplace.ecosystem.marketplace_core import MarketplaceCore
from prsm.core.config import get_settings_safe


class OrchestrationExample:
    """Example class demonstrating Phase 7 AI orchestration features"""
    
    def __init__(self):
        self.orchestrator = AIOrchestrator()
        self.pipeline_controller = UnifiedPipelineController()
        self.marketplace = MarketplaceCore()
        
    async def initialize(self):
        """Initialize orchestration components"""
        print("🚀 Initializing PRSM AI Orchestration System...")
        
        # Initialize components with safe configuration
        settings = get_settings_safe()
        if settings:
            print("✅ Configuration loaded successfully")
        else:
            print("⚠️  Using fallback configuration")
            
        await self.orchestrator.initialize()
        await self.pipeline_controller.initialize()
        await self.marketplace.initialize()
        
        print("✅ AI Orchestration system initialized")

    async def register_ai_models(self) -> List[str]:
        """Register various AI models for orchestration"""
        print("\n🤖 Registering AI Models...")
        
        models_to_register = [
            {
                'name': 'GPT-4 Reasoning Engine',
                'type': 'reasoning',
                'provider': 'openai',
                'model_id': 'gpt-4',
                'capabilities': ['reasoning', 'analysis', 'synthesis', 'problem_solving'],
                'configuration': {
                    'temperature': 0.7,
                    'max_tokens': 4000,
                    'timeout': 30
                },
                'cost_per_token': 0.00003,
                'performance_metrics': {
                    'accuracy': 0.95,
                    'speed': 0.8,
                    'reliability': 0.99
                }
            },
            {
                'name': 'Claude-3.5-Sonnet Analysis',
                'type': 'analysis',
                'provider': 'anthropic',
                'model_id': 'claude-3.5-sonnet',
                'capabilities': ['analysis', 'reasoning', 'research', 'writing'],
                'configuration': {
                    'temperature': 0.3,
                    'max_tokens': 4000,
                    'timeout': 25
                },
                'cost_per_token': 0.000015,
                'performance_metrics': {
                    'accuracy': 0.97,
                    'speed': 0.85,
                    'reliability': 0.98
                }
            },
            {
                'name': 'Specialized Math Solver',
                'type': 'mathematical',
                'provider': 'custom',
                'model_id': 'wolfram-alpha-integration',
                'capabilities': ['mathematics', 'computation', 'symbolic_analysis'],
                'configuration': {
                    'precision': 'high',
                    'timeout': 15
                },
                'cost_per_token': 0.00001,
                'performance_metrics': {
                    'accuracy': 0.99,
                    'speed': 0.9,
                    'reliability': 0.97
                }
            },
            {
                'name': 'Code Generation Specialist',
                'type': 'code_generation',
                'provider': 'github',
                'model_id': 'copilot-enterprise',
                'capabilities': ['code_generation', 'debugging', 'optimization', 'testing'],
                'configuration': {
                    'language_focus': ['python', 'javascript', 'rust'],
                    'timeout': 20
                },
                'cost_per_token': 0.00002,
                'performance_metrics': {
                    'accuracy': 0.92,
                    'speed': 0.95,
                    'reliability': 0.96
                }
            }
        ]
        
        registered_models = []
        for model_config in models_to_register:
            model_id = await self.orchestrator.register_model(model_config)
            registered_models.append(model_id)
            print(f"  ✅ Registered: {model_config['name']} -> {model_id}")
            
        print(f"✅ {len(registered_models)} AI models registered")
        return registered_models

    async def demonstrate_intelligent_routing(self):
        """Demonstrate intelligent task routing to optimal models"""
        print("\n🎯 Demonstrating Intelligent Task Routing...")
        
        # Define various types of tasks
        tasks = [
            {
                'id': 'task_reasoning_001',
                'type': 'reasoning',
                'content': 'Analyze the implications of quantum computing on cybersecurity',
                'priority': 'high',
                'requirements': {
                    'reasoning_depth': 'deep',
                    'domain_expertise': ['quantum_physics', 'cybersecurity'],
                    'max_cost': 500,
                    'preferred_accuracy': 0.95
                }
            },
            {
                'id': 'task_math_001',
                'type': 'mathematical',
                'content': 'Solve the differential equation: dy/dx = x^2 + y^2',
                'priority': 'medium',
                'requirements': {
                    'precision': 'high',
                    'symbolic_manipulation': True,
                    'max_time': 15
                }
            },
            {
                'id': 'task_code_001',
                'type': 'code_generation',
                'content': 'Create a distributed caching system in Python with Redis',
                'priority': 'high',
                'requirements': {
                    'language': 'python',
                    'patterns': ['distributed_systems', 'caching'],
                    'testing': True,
                    'documentation': True
                }
            },
            {
                'id': 'task_analysis_001',
                'type': 'analysis',
                'content': 'Analyze market trends for renewable energy investments',
                'priority': 'medium',
                'requirements': {
                    'data_sources': ['financial_markets', 'energy_sector'],
                    'time_horizon': '5_years',
                    'confidence_intervals': True
                }
            }
        ]
        
        # Execute tasks with intelligent routing
        results = []
        for task in tasks:
            print(f"\n  🎯 Processing {task['id']}...")
            print(f"     Task: {task['content'][:50]}...")
            
            # Execute task through orchestrator
            result = await self.orchestrator.execute_task(task)
            results.append(result)
            
            print(f"     ✅ Completed in {result.get('processing_time', 0):.1f}s")
            print(f"     🤖 Model: {result.get('selected_model', 'unknown')}")
            print(f"     📊 Confidence: {result.get('confidence', 0):.1%}")
            print(f"     💰 Cost: {result.get('cost_ftns', 0)} FTNS")
        
        print(f"\n✅ All {len(tasks)} tasks completed successfully")
        return results

    async def demonstrate_multi_model_collaboration(self):
        """Demonstrate multiple models working together on complex tasks"""
        print("\n🤝 Demonstrating Multi-Model Collaboration...")
        
        # Complex task requiring multiple AI capabilities
        complex_task = {
            'id': 'complex_analysis_001',
            'name': 'Comprehensive Business Analysis',
            'description': 'Analyze a startup idea from multiple perspectives',
            'subtasks': [
                {
                    'type': 'market_analysis',
                    'content': 'Analyze the market potential for AI-powered code review tools',
                    'required_capabilities': ['analysis', 'reasoning', 'research']
                },
                {
                    'type': 'technical_feasibility',
                    'content': 'Assess technical feasibility and implementation challenges',
                    'required_capabilities': ['code_generation', 'reasoning', 'analysis']
                },
                {
                    'type': 'financial_modeling',
                    'content': 'Create financial projections and business model analysis',
                    'required_capabilities': ['mathematical', 'analysis', 'reasoning']
                },
                {
                    'type': 'risk_assessment',
                    'content': 'Identify potential risks and mitigation strategies',
                    'required_capabilities': ['reasoning', 'analysis']
                }
            ]
        }
        
        print(f"  📋 Task: {complex_task['name']}")
        print(f"  📝 Subtasks: {len(complex_task['subtasks'])}")
        
        # Execute subtasks with optimal model selection
        subtask_results = []
        for i, subtask in enumerate(complex_task['subtasks'], 1):
            print(f"\n    {i}. {subtask['type'].replace('_', ' ').title()}")
            print(f"       Content: {subtask['content']}")
            
            # Find optimal model for this subtask
            task_config = {
                'type': subtask['type'],
                'content': subtask['content'],
                'required_capabilities': subtask['required_capabilities'],
                'priority': 'high'
            }
            
            result = await self.orchestrator.execute_task(task_config)
            subtask_results.append({
                'subtask': subtask['type'],
                'result': result,
                'model_used': result.get('selected_model'),
                'confidence': result.get('confidence', 0)
            })
            
            print(f"       ✅ Model: {result.get('selected_model', 'unknown')}")
            print(f"       📊 Confidence: {result.get('confidence', 0):.1%}")
        
        # Synthesize results from multiple models
        print(f"\n  🔗 Synthesizing results from {len(subtask_results)} models...")
        
        synthesis_task = {
            'type': 'synthesis',
            'content': 'Synthesize the market, technical, financial, and risk analyses into a comprehensive business recommendation',
            'context': {
                'subtask_results': [r['result'] for r in subtask_results],
                'collaboration_models': [r['model_used'] for r in subtask_results]
            },
            'requirements': {
                'synthesis_depth': 'comprehensive',
                'include_confidence_intervals': True,
                'format': 'executive_summary'
            }
        }
        
        final_result = await self.orchestrator.execute_task(synthesis_task)
        
        print(f"  ✅ Multi-model collaboration completed")
        print(f"  📊 Overall confidence: {final_result.get('confidence', 0):.1%}")
        print(f"  🤖 Models involved: {len(set(r['model_used'] for r in subtask_results))}")
        print(f"  💰 Total cost: {sum(r['result'].get('cost_ftns', 0) for r in subtask_results)} FTNS")
        
        return final_result

    async def demonstrate_unified_pipeline_integration(self):
        """Demonstrate integration with NWTN unified pipeline"""
        print("\n🔗 Demonstrating Unified Pipeline Integration...")
        
        # Complex query that uses the full NWTN pipeline
        query = "What are the most promising approaches to achieve artificial general intelligence, and what are the timeline estimates for each approach?"
        
        print(f"  🔍 Query: {query}")
        print(f"  ⚙️  Processing through 7-stage NWTN pipeline...")
        
        # Process through unified pipeline with detailed tracing
        pipeline_result = await self.pipeline_controller.process_query_full_pipeline(
            user_id='orchestration_demo_user',
            query=query,
            verbosity_level='detailed',
            enable_detailed_tracing=True
        )
        
        print(f"\n  ✅ Pipeline processing completed:")
        print(f"     📊 Status: {pipeline_result.status}")
        print(f"     ⏱️  Processing Time: {pipeline_result.metrics.processing_time:.1f}s")
        print(f"     🎯 Confidence: {pipeline_result.metrics.confidence_score:.1%}")
        print(f"     💰 Cost: {pipeline_result.metrics.total_cost_ftns} FTNS")
        print(f"     📚 Sources: {len(pipeline_result.content_sources)} papers analyzed")
        print(f"     🧠 Reasoning: {len(pipeline_result.reasoning_results)} engines used")
        
        # Show first 200 characters of response
        response_preview = pipeline_result.natural_language_response[:200] + "..."
        print(f"     📝 Response Preview: {response_preview}")
        
        return pipeline_result

    async def demonstrate_performance_optimization(self):
        """Demonstrate performance optimization features"""
        print("\n⚡ Demonstrating Performance Optimization...")
        
        # Configure performance optimization settings
        optimization_config = {
            'load_balancing': {
                'strategy': 'least_response_time',
                'health_check_interval': 30,
                'failover_threshold': 3
            },
            'caching': {
                'enable_result_caching': True,
                'cache_duration': 3600,  # 1 hour
                'cache_similarity_threshold': 0.95
            },
            'resource_allocation': {
                'adaptive_scaling': True,
                'max_concurrent_tasks': 10,
                'queue_timeout': 60
            },
            'model_selection': {
                'cost_optimization': True,
                'accuracy_threshold': 0.90,
                'speed_preference': 0.7  # Balance between speed and accuracy
            }
        }
        
        await self.orchestrator.update_configuration(optimization_config)
        print("  ✅ Performance optimization configured")
        
        # Run performance benchmark
        print("  🏃 Running performance benchmark...")
        
        benchmark_tasks = [
            {
                'type': 'reasoning',
                'content': f'Analyze the impact of AI on task {i}',
                'priority': 'medium'
            } for i in range(5)
        ]
        
        start_time = datetime.utcnow()
        
        # Execute tasks concurrently
        benchmark_results = await asyncio.gather(*[
            self.orchestrator.execute_task(task) 
            for task in benchmark_tasks
        ])
        
        end_time = datetime.utcnow()
        total_time = (end_time - start_time).total_seconds()
        
        # Calculate performance metrics
        avg_response_time = sum(r.get('processing_time', 0) for r in benchmark_results) / len(benchmark_results)
        total_cost = sum(r.get('cost_ftns', 0) for r in benchmark_results)
        avg_confidence = sum(r.get('confidence', 0) for r in benchmark_results) / len(benchmark_results)
        
        print(f"  📊 Benchmark Results:")
        print(f"     Tasks: {len(benchmark_tasks)}")
        print(f"     Total Time: {total_time:.1f}s")
        print(f"     Avg Response Time: {avg_response_time:.1f}s")
        print(f"     Total Cost: {total_cost} FTNS")
        print(f"     Avg Confidence: {avg_confidence:.1%}")
        print(f"     Throughput: {len(benchmark_tasks)/total_time:.1f} tasks/sec")
        
        return {
            'total_time': total_time,
            'avg_response_time': avg_response_time,
            'total_cost': total_cost,
            'avg_confidence': avg_confidence,
            'throughput': len(benchmark_tasks)/total_time
        }

    async def demonstrate_marketplace_integration(self):
        """Demonstrate integration with marketplace AI models"""
        print("\n🏪 Demonstrating Marketplace Integration...")
        
        # Search for AI models in marketplace
        print("  🔍 Searching marketplace for AI models...")
        
        search_results = await self.marketplace.search_integrations(
            query='AI model reasoning analysis',
            integration_type='ai_model',
            limit=5
        )
        
        print(f"  📦 Found {len(search_results)} marketplace AI models")
        
        # Register a marketplace model for orchestration
        if search_results:
            marketplace_model = search_results[0]
            print(f"  📥 Integrating: {marketplace_model.get('name', 'Unknown Model')}")
            
            # Register marketplace model in orchestrator
            model_config = {
                'name': marketplace_model.get('name', 'Marketplace Model'),
                'type': 'marketplace_ai',
                'provider': 'marketplace',
                'marketplace_id': marketplace_model.get('id'),
                'capabilities': marketplace_model.get('capabilities', []),
                'cost_per_token': marketplace_model.get('pricing', {}).get('per_token', 0.00001)
            }
            
            marketplace_model_id = await self.orchestrator.register_model(model_config)
            print(f"  ✅ Marketplace model registered: {marketplace_model_id}")
            
            # Test marketplace model with a task
            test_task = {
                'type': 'marketplace_test',
                'content': 'Test the capabilities of this marketplace AI model',
                'preferred_models': [marketplace_model_id],
                'priority': 'low'
            }
            
            result = await self.orchestrator.execute_task(test_task)
            print(f"  🧪 Test result: {result.get('status', 'unknown')}")
            print(f"  📊 Performance: {result.get('confidence', 0):.1%} confidence")
            
        else:
            print("  ℹ️  No marketplace models found for this demo")

    async def run_complete_orchestration_demo(self):
        """Run complete orchestration demonstration"""
        print("🎯 Starting Complete PRSM AI Orchestration Demo")
        print("=" * 55)
        
        try:
            # Initialize system
            await self.initialize()
            
            # Register AI models
            registered_models = await self.register_ai_models()
            
            # Demonstrate intelligent routing
            routing_results = await self.demonstrate_intelligent_routing()
            
            # Demonstrate multi-model collaboration
            collaboration_result = await self.demonstrate_multi_model_collaboration()
            
            # Demonstrate unified pipeline integration
            pipeline_result = await self.demonstrate_unified_pipeline_integration()
            
            # Demonstrate performance optimization
            performance_metrics = await self.demonstrate_performance_optimization()
            
            # Demonstrate marketplace integration
            await self.demonstrate_marketplace_integration()
            
            print("\n🎉 AI Orchestration demo completed successfully!")
            print(f"🤖 Models Registered: {len(registered_models)}")
            print(f"🎯 Tasks Executed: {len(routing_results) + 1}")  # +1 for collaboration
            print(f"⚡ Avg Performance: {performance_metrics['throughput']:.1f} tasks/sec")
            print(f"📊 Pipeline Confidence: {pipeline_result.metrics.confidence_score:.1%}")
            
        except Exception as e:
            print(f"❌ Demo failed: {str(e)}")
            raise


async def main():
    """Main function to run the orchestration example"""
    example = OrchestrationExample()
    await example.run_complete_orchestration_demo()


if __name__ == "__main__":
    # Run the complete orchestration demonstration
    asyncio.run(main())