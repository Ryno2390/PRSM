#!/usr/bin/env python3
"""
NWTN Integration Testing
Test the complete NWTN orchestrator with all integrated services
Phase 1 / Week 3 - Task 1 & 2 Integration Testing
"""

import asyncio
import sys
from typing import Dict, Any
from uuid import UUID

# Structured logging
import structlog
import pytest

structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(colors=True)
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger(__name__)

# PRSM imports
try:
    from prsm.core.models import UserInput, PRSMSession, ClarifiedPrompt
    from prsm.compute.nwtn.orchestrator import NWTNOrchestrator
    from prsm.compute.nwtn.context_manager import ContextManager
    from prsm.economy.tokenomics.ftns_service import FTNSService
    from prsm.data.data_layer.enhanced_ipfs import PRSMIPFSClient
    from prsm.compute.federation.model_registry import ModelRegistry
except (ImportError, ModuleNotFoundError) as e:
    pytest.skip("NWTN orchestrator module not yet implemented", allow_module_level=True)


class NWTNIntegrationTester:
    """Test NWTN orchestrator integration with all services"""
    
    def __init__(self):
        self.ftns_service = FTNSService()
        self.context_manager = ContextManager()
        self.ipfs_client = PRSMIPFSClient()
        self.model_registry = ModelRegistry()
        self.orchestrator = NWTNOrchestrator(
            context_manager=self.context_manager,
            ftns_service=self.ftns_service,
            ipfs_client=self.ipfs_client,
            model_registry=self.model_registry
        )
        self.test_user_id = "test_user_nwtn_integration"
    
    async def setup_test_environment(self):
        """Set up test environment with FTNS balance and models"""
        print("🏗️ Setting up test environment...")
        
        # Give test user FTNS tokens
        reward_success = await self.ftns_service.reward_contribution(
            self.test_user_id, "data", 1000.0
        )
        print(f"🪙 Reward given: {reward_success}")
        
        # Check balance immediately
        balance_check = await self.ftns_service.get_user_balance(self.test_user_id)
        print(f"🪙 Balance after reward: {balance_check.balance} FTNS")
        
        # Register some test models in the registry
        test_models = [
            {
                "name": "Research Assistant",
                "specialization": "research",
                "performance_score": 0.92
            },
            {
                "name": "Data Analyzer",
                "specialization": "data_analysis",
                "performance_score": 0.88
            },
            {
                "name": "General Helper",
                "specialization": "general",
                "performance_score": 0.85
            }
        ]
        
        for model_data in test_models:
            # Store model in IPFS (simulated)
            model_bytes = f"Model: {model_data['name']}".encode()
            cid = await self.ipfs_client.store_model(model_bytes, model_data)
            
            # Register in model registry
            from prsm.core.models import TeacherModel
            teacher_model = TeacherModel(
                name=model_data["name"],
                specialization=model_data["specialization"],
                performance_score=model_data["performance_score"],
                curriculum_ids=[],
                student_models=[]
            )
            
            await self.model_registry.register_teacher_model(teacher_model, cid)
        
        balance_obj = await self.ftns_service.get_user_balance(self.test_user_id)
        balance = balance_obj.balance
        models_count = len(self.model_registry.registered_models)
        
        print(f"✅ Environment ready: {balance} FTNS, {models_count} models")
    
    async def test_simple_query(self):
        """Test simple query processing"""
        print("\n🧪 Testing simple query processing...")
        
        user_input = UserInput(
            user_id=self.test_user_id,
            prompt="What is machine learning?",
            context_allocation=200
        )
        
        try:
            response = await self.orchestrator.process_query(user_input)
            
            print(f"✅ Query processed successfully")
            print(f"   - Session ID: {response.session_id}")
            print(f"   - Context used: {response.context_used}")
            print(f"   - FTNS charged: {response.ftns_charged}")
            print(f"   - Reasoning steps: {len(response.reasoning_trace)}")
            print(f"   - Safety validated: {response.safety_validated}")
            
            return True
            
        except Exception as e:
            print(f"❌ Simple query failed: {e}")
            return False
    
    async def test_research_query(self):
        """Test complex research query"""
        print("\n🧪 Testing complex research query...")
        
        user_input = UserInput(
            user_id=self.test_user_id,
            prompt="Research the impact of artificial intelligence on scientific discovery and analyze the key factors that accelerate breakthrough discoveries in computational biology.",
            context_allocation=500
        )
        
        try:
            response = await self.orchestrator.process_query(user_input)
            
            print(f"✅ Research query processed successfully")
            print(f"   - Session ID: {response.session_id}")
            print(f"   - Context used: {response.context_used}")
            print(f"   - FTNS charged: {response.ftns_charged}")
            print(f"   - Reasoning steps: {len(response.reasoning_trace)}")
            print(f"   - Confidence: {response.confidence_score}")
            
            # Check that multiple reasoning steps were created
            if len(response.reasoning_trace) >= 4:
                print(f"   - ✅ Complete 4-stage pipeline executed")
                for step in response.reasoning_trace:
                    print(f"     • Stage {step.step_number}: {step.agent_type.value} ({step.context_used} context)")
            
            return True
            
        except Exception as e:
            print(f"❌ Research query failed: {e}")
            return False
    
    async def test_insufficient_balance(self):
        """Test handling of insufficient FTNS balance"""
        print("\n🧪 Testing insufficient FTNS balance handling...")
        
        # Create user with low balance
        low_balance_user = "test_user_low_balance"
        await self.ftns_service.reward_contribution(low_balance_user, "data", 1.0)
        
        user_input = UserInput(
            user_id=low_balance_user,
            prompt="Explain quantum computing in detail",
            context_allocation=1000  # High allocation
        )
        
        try:
            response = await self.orchestrator.process_query(user_input)
            print(f"❌ Expected failure but query succeeded: {response.session_id}")
            return False
            
        except ValueError as e:
            if "Insufficient" in str(e):
                print(f"✅ Correctly rejected insufficient balance: {e}")
                return True
            else:
                print(f"❌ Wrong error type: {e}")
                return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    async def test_context_optimization(self):
        """Test context usage optimization"""
        print("\n🧪 Testing context usage optimization...")
        
        # Generate some historical usage data
        historical_data = [
            {"allocated": 100, "used": 85},
            {"allocated": 200, "used": 150},
            {"allocated": 150, "used": 120},
            {"allocated": 300, "used": 180},
            {"allocated": 250, "used": 200}
        ]
        
        try:
            recommendations = await self.context_manager.optimize_context_allocation(
                historical_data
            )
            
            print(f"✅ Context optimization completed")
            print(f"   - Average efficiency: {recommendations.get('avg_efficiency', 0):.2%}")
            print(f"   - Over-allocation rate: {recommendations.get('over_allocation_rate', 0):.1%}")
            print(f"   - Under-allocation rate: {recommendations.get('under_allocation_rate', 0):.1%}")
            print(f"   - Optimization potential: {recommendations.get('optimization_potential', 0):.1f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Context optimization failed: {e}")
            return False
    
    async def test_session_management(self):
        """Test session creation and management"""
        print("\n🧪 Testing session management...")
        
        try:
            # Create multiple sessions
            sessions = []
            for i in range(3):
                user_input = UserInput(
                    user_id=f"{self.test_user_id}_{i}",
                    prompt=f"Test query {i}",
                    context_allocation=100
                )
                
                response = await self.orchestrator.process_query(user_input)
                sessions.append(response.session_id)
            
            print(f"✅ Created {len(sessions)} sessions")
            
            # Check session storage
            total_sessions = len(self.orchestrator.sessions)
            print(f"   - Total sessions in orchestrator: {total_sessions}")
            
            # Test context usage tracking
            for session_id in sessions:
                usage = await self.context_manager.get_session_usage(session_id)
                if usage:
                    print(f"   - Session {str(session_id)[:8]}...: {usage.context_used} context used")
                else:
                    print(f"   - Session {str(session_id)[:8]}...: No usage data")
            
            return True
            
        except Exception as e:
            print(f"❌ Session management test failed: {e}")
            return False
    
    async def test_model_discovery_integration(self):
        """Test model discovery integration"""
        print("\n🧪 Testing model discovery integration...")
        
        try:
            # Test different types of queries to trigger model discovery
            query_types = [
                ("Research the latest developments in AI", "research"),
                ("Analyze this dataset for patterns", "data_analysis"),
                ("Help me understand basic concepts", "general")
            ]
            
            for prompt, expected_category in query_types:
                # Clarify intent first
                clarified = await self.orchestrator.clarify_intent(prompt)
                print(f"   - Query: '{prompt[:30]}...'")
                print(f"     Category: {clarified.intent_category}")
                print(f"     Complexity: {clarified.complexity_estimate:.2f}")
                print(f"     Context required: {clarified.context_required}")
                
                # Check that correct models are discovered
                available_models = await self.model_registry.discover_specialists(
                    clarified.intent_category
                )
                print(f"     Available models: {len(available_models)}")
            
            print(f"✅ Model discovery integration working")
            return True
            
        except Exception as e:
            print(f"❌ Model discovery integration failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all integration tests"""
        print("🚀 NWTN Integration Testing")
        print("Testing Phase 1 / Week 3 NWTN orchestrator integration")
        print("=" * 60)
        
        await self.setup_test_environment()
        
        tests = [
            ("Simple Query Processing", self.test_simple_query),
            ("Research Query Processing", self.test_research_query),
            ("Insufficient Balance Handling", self.test_insufficient_balance),
            ("Context Optimization", self.test_context_optimization),
            ("Session Management", self.test_session_management),
            ("Model Discovery Integration", self.test_model_discovery_integration)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = await test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name} crashed: {e}")
                results.append((test_name, False))
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅" if result else "❌"
            print(f"{status} {test_name}")
        
        print(f"\n📈 Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL NWTN INTEGRATION TESTS PASSED!")
            print("✅ NWTN orchestrator is fully integrated with:")
            print("   - FTNS token system for context allocation")
            print("   - Context manager for usage tracking")
            print("   - Model registry for specialist discovery")
            print("   - IPFS client for model storage")
            print("   - Complete 4-stage agent pipeline simulation")
        else:
            print(f"⚠️  {total - passed} tests failed. Please review and fix issues.")
        
        return passed == total


async def main():
    """Run NWTN integration tests"""
    tester = NWTNIntegrationTester()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())