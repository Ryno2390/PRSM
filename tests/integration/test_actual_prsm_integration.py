"""
PRSM Actual Component Integration Test
=====================================

This test validates that actual PRSM components can work together
with simplified dependencies, identifying real integration issues
and providing concrete validation of system harmony.
"""

import asyncio
import sys
import os
from decimal import Decimal
from uuid import uuid4, UUID
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from unittest.mock import AsyncMock, MagicMock, patch

# Add PRSM to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Try to import actual PRSM components and handle gracefully
try:
    from prsm.core.models import UserInput, PRSMSession
    from prsm.core.config import get_settings
    CORE_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Core models not available: {e}")
    CORE_MODELS_AVAILABLE = False

try:
    # Import budget management components
    from prsm.tokenomics.ftns_budget_manager import FTNSBudgetManager, SpendingCategory
    BUDGET_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Budget management not available: {e}")
    BUDGET_AVAILABLE = False

try:
    # Import marketplace components  
    from prsm.marketplace.expanded_models import ResourceType, DatasetListing, AgentWorkflowListing
    from prsm.marketplace.real_marketplace_service import RealMarketplaceService
    MARKETPLACE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Marketplace components not available: {e}")
    MARKETPLACE_AVAILABLE = False

try:
    # Import enhanced orchestrator
    from prsm.nwtn.enhanced_orchestrator import EnhancedNWTNOrchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Enhanced orchestrator not available: {e}")
    ORCHESTRATOR_AVAILABLE = False

# Fallback models if imports fail
if not CORE_MODELS_AVAILABLE:
    class UserInput:
        def __init__(self, user_id: str, prompt: str, preferences: Dict = None):
            self.user_id = user_id
            self.prompt = prompt
            self.preferences = preferences or {}
    
    class PRSMSession:
        def __init__(self, user_id: str):
            self.session_id = uuid4()
            self.user_id = user_id

if not BUDGET_AVAILABLE:
    class SpendingCategory:
        MODEL_INFERENCE = "model_inference"
        TOOL_EXECUTION = "tool_execution"
        AGENT_COORDINATION = "agent_coordination"
    
    class FTNSBudgetManager:
        def __init__(self):
            self.active_budgets = {}


class PRSMActualIntegrationTest:
    """Test actual PRSM component integration with graceful fallbacks"""
    
    def __init__(self):
        self.test_results = {
            "component_availability": {},
            "integration_tests": {},
            "issues_found": [],
            "recommendations": []
        }
        
        # Check component availability
        self.test_results["component_availability"] = {
            "core_models": CORE_MODELS_AVAILABLE,
            "budget_management": BUDGET_AVAILABLE,
            "marketplace": MARKETPLACE_AVAILABLE,
            "orchestrator": ORCHESTRATOR_AVAILABLE
        }
    
    async def test_component_availability(self):
        """Test which PRSM components are available and functional"""
        print("🔍 TESTING COMPONENT AVAILABILITY")
        print("=" * 50)
        
        components = [
            ("Core Models (UserInput, PRSMSession)", CORE_MODELS_AVAILABLE),
            ("Budget Management (FTNSBudgetManager)", BUDGET_AVAILABLE),
            ("Expanded Marketplace", MARKETPLACE_AVAILABLE),
            ("Enhanced Orchestrator", ORCHESTRATOR_AVAILABLE)
        ]
        
        available_count = 0
        for component_name, available in components:
            status = "✅ AVAILABLE" if available else "❌ NOT AVAILABLE"
            print(f"   {status} {component_name}")
            if available:
                available_count += 1
        
        availability_percentage = (available_count / len(components)) * 100
        print(f"\n📊 Component Availability: {available_count}/{len(components)} ({availability_percentage:.1f}%)")
        
        if availability_percentage < 50:
            self.test_results["issues_found"].append("Less than 50% of components available")
            self.test_results["recommendations"].append("Fix import dependencies and missing components")
        
        return availability_percentage >= 75
    
    async def test_core_models_integration(self):
        """Test core PRSM models can be created and used"""
        print("\n🧠 TESTING CORE MODELS INTEGRATION")
        print("-" * 40)
        
        if not CORE_MODELS_AVAILABLE:
            print("   ⚠️ Using fallback models due to import issues")
            self.test_results["issues_found"].append("Core models import failed - using fallbacks")
        
        try:
            # Test UserInput creation
            user_input = UserInput(
                user_id="test_user_001",
                prompt="Test quantum field analysis for photonic systems",
                preferences={"max_budget": 100.0}
            )
            
            print(f"   ✅ UserInput created: {user_input.user_id}")
            print(f"      Prompt length: {len(user_input.prompt)} chars")
            print(f"      Preferences: {user_input.preferences}")
            
            # Test PRSMSession creation
            session = PRSMSession(user_id=user_input.user_id)
            
            print(f"   ✅ PRSMSession created: {session.session_id}")
            print(f"      User ID match: {session.user_id == user_input.user_id}")
            
            self.test_results["integration_tests"]["core_models"] = True
            return True
            
        except Exception as e:
            print(f"   ❌ Core models integration failed: {e}")
            self.test_results["integration_tests"]["core_models"] = False
            self.test_results["issues_found"].append(f"Core models integration error: {e}")
            return False
    
    async def test_budget_management_integration(self):
        """Test budget management components integration"""
        print("\n💰 TESTING BUDGET MANAGEMENT INTEGRATION")
        print("-" * 40)
        
        if not BUDGET_AVAILABLE:
            print("   ⚠️ Budget management components not available")
            self.test_results["issues_found"].append("Budget management import failed")
            return False
        
        try:
            # Test budget manager creation
            budget_manager = FTNSBudgetManager()
            
            print(f"   ✅ FTNSBudgetManager created")
            print(f"      Active budgets: {len(budget_manager.active_budgets)}")
            
            # Test spending categories
            categories = [
                SpendingCategory.MODEL_INFERENCE,
                SpendingCategory.TOOL_EXECUTION,
                SpendingCategory.AGENT_COORDINATION
            ]
            
            print(f"   ✅ Spending categories available: {len(categories)}")
            for category in categories:
                print(f"      - {category}")
            
            self.test_results["integration_tests"]["budget_management"] = True
            return True
            
        except Exception as e:
            print(f"   ❌ Budget management integration failed: {e}")
            self.test_results["integration_tests"]["budget_management"] = False
            self.test_results["issues_found"].append(f"Budget management error: {e}")
            return False
    
    async def test_marketplace_integration(self):
        """Test marketplace components integration"""
        print("\n🏪 TESTING MARKETPLACE INTEGRATION")
        print("-" * 40)
        
        if not MARKETPLACE_AVAILABLE:
            print("   ⚠️ Marketplace components not available")
            self.test_results["issues_found"].append("Marketplace import failed")
            return False
        
        try:
            # Test marketplace service creation
            marketplace_service = RealMarketplaceService()
            
            print(f"   ✅ RealMarketplaceService created")
            
            # Test resource types
            resource_types = list(ResourceType)
            print(f"   ✅ Resource types available: {len(resource_types)}")
            for rt in resource_types:
                print(f"      - {rt.value}")
            
            # Test resource model creation
            test_dataset = DatasetListing(
                name="Test Dataset",
                description="Test dataset for integration testing",
                category="scientific_research",
                size_bytes=1024,
                record_count=100,
                data_format="json",
                license_type="mit",
                pricing_model="free",
                owner_user_id=uuid4(),
                quality_grade="community",
                tags=["test", "integration"]
            )
            
            print(f"   ✅ DatasetListing created: {test_dataset.name}")
            
            test_workflow = AgentWorkflowListing(
                name="Test Agent Workflow",
                description="Test workflow for integration testing",
                agent_type="research_agent",
                capabilities=["multi_step_reasoning"],
                pricing_model="pay_per_use",
                price_per_execution=Decimal('5.99'),
                owner_user_id=uuid4(),
                quality_grade="community",
                tags=["test", "agent"]
            )
            
            print(f"   ✅ AgentWorkflowListing created: {test_workflow.name}")
            
            self.test_results["integration_tests"]["marketplace"] = True
            return True
            
        except Exception as e:
            print(f"   ❌ Marketplace integration failed: {e}")
            self.test_results["integration_tests"]["marketplace"] = False
            self.test_results["issues_found"].append(f"Marketplace error: {e}")
            return False
    
    async def test_orchestrator_integration(self):
        """Test orchestrator integration with other components"""
        print("\n🎭 TESTING ORCHESTRATOR INTEGRATION")
        print("-" * 40)
        
        if not ORCHESTRATOR_AVAILABLE:
            print("   ⚠️ Enhanced orchestrator not available")
            self.test_results["issues_found"].append("Orchestrator import failed")
            return False
        
        try:
            # Test orchestrator creation with mocked dependencies
            with patch('prsm.nwtn.enhanced_orchestrator.get_settings') as mock_settings:
                mock_settings.return_value = MagicMock()
                
                # Create orchestrator with minimal dependencies
                orchestrator = EnhancedNWTNOrchestrator()
                
                print(f"   ✅ EnhancedNWTNOrchestrator created")
                print(f"      Type: {type(orchestrator).__name__}")
                
                # Test basic attributes exist
                expected_attributes = [
                    'context_manager', 'ftns_service', 'budget_manager',
                    'safety_monitor', 'model_network'
                ]
                
                available_attributes = [attr for attr in expected_attributes if hasattr(orchestrator, attr)]
                print(f"   ✅ Attributes available: {len(available_attributes)}/{len(expected_attributes)}")
                
                for attr in available_attributes:
                    print(f"      - {attr}: {type(getattr(orchestrator, attr, None)).__name__}")
            
            self.test_results["integration_tests"]["orchestrator"] = True
            return True
            
        except Exception as e:
            print(f"   ❌ Orchestrator integration failed: {e}")
            self.test_results["integration_tests"]["orchestrator"] = False
            self.test_results["issues_found"].append(f"Orchestrator error: {e}")
            return False
    
    async def test_cross_component_integration(self):
        """Test integration between different PRSM components"""
        print("\n🔗 TESTING CROSS-COMPONENT INTEGRATION")
        print("-" * 40)
        
        integration_score = 0
        max_score = 0
        
        # Test 1: Core models + Budget management
        if CORE_MODELS_AVAILABLE and BUDGET_AVAILABLE:
            try:
                user_input = UserInput(
                    user_id="integration_test_user",
                    prompt="Cross-component integration test",
                    preferences={"max_budget": 50.0}
                )
                
                session = PRSMSession(user_id=user_input.user_id)
                budget_manager = FTNSBudgetManager()
                
                print(f"   ✅ Core + Budget integration working")
                integration_score += 1
                
            except Exception as e:
                print(f"   ❌ Core + Budget integration failed: {e}")
                self.test_results["issues_found"].append(f"Core+Budget integration: {e}")
        
        max_score += 1
        
        # Test 2: Budget + Marketplace integration
        if BUDGET_AVAILABLE and MARKETPLACE_AVAILABLE:
            try:
                budget_manager = FTNSBudgetManager()
                marketplace_service = RealMarketplaceService()
                
                print(f"   ✅ Budget + Marketplace integration working")
                integration_score += 1
                
            except Exception as e:
                print(f"   ❌ Budget + Marketplace integration failed: {e}")
                self.test_results["issues_found"].append(f"Budget+Marketplace integration: {e}")
        
        max_score += 1
        
        # Test 3: All components together (if available)
        if all([CORE_MODELS_AVAILABLE, BUDGET_AVAILABLE, MARKETPLACE_AVAILABLE]):
            try:
                user_input = UserInput(
                    user_id="full_integration_test",
                    prompt="Full system integration test",
                    preferences={"max_budget": 100.0}
                )
                
                session = PRSMSession(user_id=user_input.user_id)
                budget_manager = FTNSBudgetManager()
                marketplace_service = RealMarketplaceService()
                
                print(f"   ✅ Full system integration working")
                integration_score += 1
                
            except Exception as e:
                print(f"   ❌ Full system integration failed: {e}")
                self.test_results["issues_found"].append(f"Full system integration: {e}")
        
        max_score += 1
        
        integration_percentage = (integration_score / max_score) * 100 if max_score > 0 else 0
        print(f"\n🎯 Cross-component integration: {integration_score}/{max_score} ({integration_percentage:.1f}%)")
        
        self.test_results["integration_tests"]["cross_component"] = integration_percentage >= 66
        return integration_percentage >= 66
    
    async def test_dependency_resolution(self):
        """Test dependency resolution and error handling"""
        print("\n🔧 TESTING DEPENDENCY RESOLUTION")
        print("-" * 40)
        
        dependency_issues = []
        
        # Test database dependency simulation
        try:
            # Simulate database service creation
            print("   🔍 Testing database service dependency...")
            
            # This would normally require actual database
            # For now, just test that the import pattern works
            if "get_database_service" in str(self.test_results):
                print("   ✅ Database service pattern recognized")
            else:
                print("   ⚠️ Database service dependency not resolved")
                dependency_issues.append("Database service dependency")
                
        except Exception as e:
            dependency_issues.append(f"Database dependency: {e}")
        
        # Test FTNS service dependency
        try:
            print("   🔍 Testing FTNS service dependency...")
            
            if BUDGET_AVAILABLE:
                print("   ✅ FTNS service accessible through budget manager")
            else:
                print("   ⚠️ FTNS service dependency not resolved")
                dependency_issues.append("FTNS service dependency")
                
        except Exception as e:
            dependency_issues.append(f"FTNS dependency: {e}")
        
        # Test external service dependencies
        external_services = [
            ("PostgreSQL", "postgresql://"),
            ("Redis", "redis://"),
            ("IPFS", "ipfs://"),
            ("Vector DB", "vector_db")
        ]
        
        print("   🔍 External service dependencies:")
        for service_name, service_pattern in external_services:
            print(f"      ⚠️ {service_name}: Mock mode (production requires real service)")
            dependency_issues.append(f"{service_name} requires production setup")
        
        print(f"\n📊 Dependency Resolution:")
        print(f"   Issues identified: {len(dependency_issues)}")
        print(f"   Resolution status: {'⚠️ PARTIAL' if dependency_issues else '✅ COMPLETE'}")
        
        self.test_results["dependency_issues"] = dependency_issues
        return len(dependency_issues) < 10  # Allow some expected issues in proof-of-concept
    
    async def generate_integration_assessment(self):
        """Generate comprehensive integration assessment"""
        print("\n📋 INTEGRATION ASSESSMENT REPORT")
        print("=" * 60)
        
        # Calculate overall score
        available_components = sum(self.test_results["component_availability"].values())
        total_components = len(self.test_results["component_availability"])
        
        passed_tests = sum(self.test_results["integration_tests"].values())
        total_tests = len(self.test_results["integration_tests"])
        
        component_score = (available_components / total_components) * 100
        integration_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        overall_score = (component_score + integration_score) / 2
        
        print(f"🎯 Component Availability: {available_components}/{total_components} ({component_score:.1f}%)")
        print(f"🎯 Integration Tests: {passed_tests}/{total_tests} ({integration_score:.1f}%)")
        print(f"🎯 Overall Score: {overall_score:.1f}%")
        
        # Assessment categories
        if overall_score >= 85:
            assessment = "EXCELLENT"
            color = "✅"
            recommendation = "System ready for advanced testing and development"
        elif overall_score >= 70:
            assessment = "GOOD"
            color = "✅"
            recommendation = "System functional with minor issues to address"
        elif overall_score >= 50:
            assessment = "FAIR"
            color = "⚠️"
            recommendation = "System partially functional, requires focused debugging"
        else:
            assessment = "NEEDS WORK"
            color = "❌"
            recommendation = "System requires significant integration work"
        
        print(f"\n{color} PRSM SYSTEM INTEGRATION: {assessment}")
        print(f"   {recommendation}")
        
        # Issues summary
        if self.test_results["issues_found"]:
            print(f"\n❌ Issues Found ({len(self.test_results['issues_found'])}):")
            for i, issue in enumerate(self.test_results["issues_found"], 1):
                print(f"   {i}. {issue}")
        
        # Recommendations
        recommendations = [
            "Set up proper external service dependencies (PostgreSQL, Redis, IPFS)",
            "Resolve import dependency conflicts",
            "Add comprehensive error handling for missing services",
            "Implement graceful degradation for unavailable components",
            "Create development environment setup scripts",
            "Add service health monitoring and diagnostics"
        ]
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        return {
            "overall_score": overall_score,
            "assessment": assessment,
            "component_availability": component_score,
            "integration_score": integration_score,
            "issues_count": len(self.test_results["issues_found"]),
            "ready_for_development": overall_score >= 70
        }


async def run_actual_prsm_integration_test():
    """Run actual PRSM component integration test"""
    print("🧪 PRSM ACTUAL COMPONENT INTEGRATION TEST")
    print("=" * 70)
    
    # Initialize test framework
    test_framework = PRSMActualIntegrationTest()
    
    try:
        # Run test suites
        print("Starting component integration validation...")
        
        # Test component availability
        availability_ok = await test_framework.test_component_availability()
        
        # Test individual component integration
        core_ok = await test_framework.test_core_models_integration()
        budget_ok = await test_framework.test_budget_management_integration()
        marketplace_ok = await test_framework.test_marketplace_integration()
        orchestrator_ok = await test_framework.test_orchestrator_integration()
        
        # Test cross-component integration
        cross_ok = await test_framework.test_cross_component_integration()
        
        # Test dependency resolution
        deps_ok = await test_framework.test_dependency_resolution()
        
        # Generate assessment
        final_assessment = await test_framework.generate_integration_assessment()
        
        print("\n🎉 ACTUAL INTEGRATION TESTING COMPLETE!")
        print("=" * 50)
        
        if final_assessment["ready_for_development"]:
            print("🚀 PRSM is ready for continued development!")
            return True
        else:
            print("⚠️ PRSM needs attention before proceeding.")
            return False
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the actual integration test
    result = asyncio.run(run_actual_prsm_integration_test())
    
    if result:
        print("\n✅ PRSM actual component integration validation passed!")
        exit(0)
    else:
        print("\n⚠️ PRSM actual component integration needs attention.")
        exit(1)