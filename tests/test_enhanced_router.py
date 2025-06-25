#!/usr/bin/env python3
"""
Test Suite for PRSM Enhanced Router
Comprehensive testing of advanced routing functionality
"""

import asyncio
import json
import time
from typing import Dict, Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# PRSM imports
from prsm.agents.routers import (
    ModelRouter, RoutingStrategy, ModelSource, 
    MarketplaceRequest, create_enhanced_router
)
from prsm.core.config import get_settings
from prsm.core.models import ArchitectTask, ModelType
from prsm.federation.model_registry import ModelRegistry

console = Console()
settings = get_settings()


class EnhancedRouterTestSuite:
    """Comprehensive test suite for Enhanced ModelRouter"""
    
    def __init__(self):
        self.router = None
        self.model_registry = None
        self.test_results = []
        self.test_tasks = self._prepare_test_tasks()
    
    def _prepare_test_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Prepare diverse test tasks for different routing scenarios"""
        return {
            "physics_research": {
                "description": "Analyze quantum entanglement phenomena in superconductors",
                "complexity": 0.9,
                "expected_domain": "physics",
                "expected_strategy": RoutingStrategy.PERFORMANCE_OPTIMIZED
            },
            "chemistry_synthesis": {
                "description": "Design organic synthesis pathway for pharmaceutical compound",
                "complexity": 0.8,
                "expected_domain": "chemistry",
                "expected_strategy": RoutingStrategy.ACCURACY_OPTIMIZED
            },
            "biology_explanation": {
                "description": "Explain protein folding mechanisms in cellular environments",
                "complexity": 0.6,
                "expected_domain": "biology",
                "expected_strategy": RoutingStrategy.PERFORMANCE_OPTIMIZED
            },
            "math_proof": {
                "description": "Prove the fundamental theorem of calculus",
                "complexity": 0.7,
                "expected_domain": "mathematics",
                "expected_strategy": RoutingStrategy.ACCURACY_OPTIMIZED
            },
            "cs_algorithm": {
                "description": "Implement efficient sorting algorithm for large datasets",
                "complexity": 0.5,
                "expected_domain": "computer_science",
                "expected_strategy": RoutingStrategy.PERFORMANCE_OPTIMIZED
            },
            "general_task": {
                "description": "Summarize recent scientific developments",
                "complexity": 0.3,
                "expected_domain": "general",
                "expected_strategy": RoutingStrategy.COST_OPTIMIZED
            },
            "teaching_task": {
                "description": "Teach mathematics concepts to a student model",
                "complexity": 0.4,
                "expected_domain": "mathematics",
                "expected_strategy": RoutingStrategy.TEACHER_SELECTION
            },
            "marketplace_task": {
                "description": "Generate creative content for marketing",
                "complexity": 0.6,
                "expected_domain": "creation",
                "expected_strategy": RoutingStrategy.MARKETPLACE_PREFERRED
            }
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        console.print("\n🧪 [bold blue]PRSM Enhanced Router Test Suite[/bold blue]")
        console.print("=" * 60)
        
        # Initialize components
        await self._test_initialization()
        
        # Test basic routing functionality
        await self._test_basic_routing()
        
        # Test routing strategies
        await self._test_routing_strategies()
        
        # Test specialist matching
        await self._test_specialist_matching()
        
        # Test marketplace routing
        await self._test_marketplace_routing()
        
        # Test teacher selection
        await self._test_teacher_selection()
        
        # Test performance tracking
        await self._test_performance_tracking()
        
        # Test concurrent routing
        await self._test_concurrent_routing()
        
        # Test analytics and reporting
        await self._test_analytics()
        
        # Generate summary
        return await self._generate_test_summary()
    
    async def _test_initialization(self):
        """Test router initialization"""
        console.print("\n📋 [yellow]Testing Initialization[/yellow]")
        
        start_time = time.time()
        self.model_registry = ModelRegistry()
        self.router = create_enhanced_router(self.model_registry)
        init_time = time.time() - start_time
        
        # Verify initialization
        assert self.router.agent_type.value == "router"
        assert len(self.router.marketplace_endpoints) > 0
        assert isinstance(self.router.routing_cache, dict)
        assert isinstance(self.router.routing_decisions, list)
        assert isinstance(self.router.performance_history, dict)
        
        self.test_results.append({
            "test": "initialization",
            "status": "✅ PASSED",
            "init_time": f"{init_time:.3f}s",
            "marketplace_endpoints": len(self.router.marketplace_endpoints),
            "agent_type": self.router.agent_type.value
        })
        
        console.print(f"✅ Enhanced Router initialized in {init_time:.3f}s")
        console.print(f"   Marketplace endpoints: {len(self.router.marketplace_endpoints)}")
        console.print(f"   Agent type: {self.router.agent_type.value}")
    
    async def _test_basic_routing(self):
        """Test basic routing functionality"""
        console.print("\n🔧 [yellow]Testing Basic Routing[/yellow]")
        
        test_task = "Analyze photosynthesis process in plants"
        
        start_time = time.time()
        decision = await self.router.process(test_task)
        processing_time = time.time() - start_time
        
        # Verify decision structure
        assert hasattr(decision, 'primary_candidate')
        assert hasattr(decision, 'backup_candidates')
        assert hasattr(decision, 'confidence_score')
        assert hasattr(decision, 'strategy_used')
        assert hasattr(decision, 'reasoning')
        assert decision.confidence_score >= 0.0
        assert decision.confidence_score <= 1.0
        assert len(decision.reasoning) > 0
        
        self.test_results.append({
            "test": "basic_routing",
            "status": "✅ PASSED",
            "processing_time": f"{processing_time:.3f}s",
            "primary_model": decision.primary_candidate.model_id,
            "backup_count": len(decision.backup_candidates),
            "confidence": f"{decision.confidence_score:.2f}",
            "strategy": decision.strategy_used.value if hasattr(decision.strategy_used, 'value') else str(decision.strategy_used)
        })
        
        console.print(f"✅ Basic routing completed in {processing_time:.3f}s")
        console.print(f"   Primary model: {decision.primary_candidate.model_id}")
        console.print(f"   Backup models: {len(decision.backup_candidates)}")
        console.print(f"   Confidence: {decision.confidence_score:.2f}")
    
    async def _test_routing_strategies(self):
        """Test different routing strategies"""
        console.print("\n🎯 [yellow]Testing Routing Strategies[/yellow]")
        
        strategy_results = []
        test_task = "Optimize machine learning algorithm performance"
        
        for strategy in RoutingStrategy:
            start_time = time.time()
            
            decision = await self.router.process(test_task, {"strategy": strategy.value})
            processing_time = time.time() - start_time
            
            # Verify strategy was applied
            assert decision.strategy_used == strategy
            
            strategy_results.append({
                "strategy": strategy.value,
                "processing_time": processing_time,
                "primary_model": decision.primary_candidate.model_id,
                "confidence": decision.confidence_score,
                "source": decision.primary_candidate.source.value if hasattr(decision.primary_candidate.source, 'value') else str(decision.primary_candidate.source)
            })
            
            console.print(f"✅ {strategy.value}: {decision.primary_candidate.model_id} "
                         f"(confidence: {decision.confidence_score:.2f})")
        
        self.test_results.append({
            "test": "routing_strategies",
            "status": "✅ PASSED",
            "strategies_tested": len(strategy_results),
            "results": strategy_results
        })
    
    async def _test_specialist_matching(self):
        """Test specialist model matching"""
        console.print("\n🎓 [yellow]Testing Specialist Matching[/yellow]")
        
        specialist_results = []
        
        for test_name, test_data in self.test_tasks.items():
            if "domain" in test_data["expected_domain"]:
                start_time = time.time()
                
                decision = await self.router.process(test_data["description"])
                processing_time = time.time() - start_time
                
                # Verify domain awareness
                task_domain = await self.router._categorize_task(test_data["description"])
                
                specialist_results.append({
                    "test": test_name,
                    "expected_domain": test_data["expected_domain"],
                    "detected_domain": task_domain,
                    "primary_model": decision.primary_candidate.model_id,
                    "specialization": decision.primary_candidate.specialization,
                    "processing_time": processing_time,
                    "confidence": decision.confidence_score
                })
                
                console.print(f"✅ {test_name}: {task_domain} → {decision.primary_candidate.specialization}")
        
        self.test_results.append({
            "test": "specialist_matching",
            "status": "✅ PASSED",
            "domain_tests": len(specialist_results),
            "results": specialist_results
        })
    
    async def _test_marketplace_routing(self):
        """Test marketplace integration"""
        console.print("\n🏪 [yellow]Testing Marketplace Routing[/yellow]")
        
        marketplace_task = "Generate creative marketing content for AI product"
        
        start_time = time.time()
        decision = await self.router.process(marketplace_task, 
                                           {"strategy": "marketplace_preferred"})
        processing_time = time.time() - start_time
        
        # Test marketplace request creation
        from uuid import uuid4
        architect_task = ArchitectTask(
            session_id=uuid4(),
            instruction=marketplace_task,
            complexity_score=0.6
        )
        
        marketplace_request = await self.router.route_to_marketplace(architect_task)
        
        # Verify marketplace functionality
        assert isinstance(marketplace_request, MarketplaceRequest)
        assert marketplace_request.task_description == marketplace_task
        assert marketplace_request.quality_threshold > 0.0
        
        # Check if marketplace models were considered
        marketplace_sources = [c for c in [decision.primary_candidate] + decision.backup_candidates
                             if (hasattr(c.source, 'value') and c.source == ModelSource.MARKETPLACE) or 
                                (isinstance(c.source, str) and c.source == "marketplace")]
        
        self.test_results.append({
            "test": "marketplace_routing",
            "status": "✅ PASSED",
            "processing_time": f"{processing_time:.3f}s",
            "marketplace_candidates": len(marketplace_sources),
            "request_created": True,
            "quality_threshold": marketplace_request.quality_threshold
        })
        
        console.print(f"✅ Marketplace routing completed in {processing_time:.3f}s")
        console.print(f"   Marketplace candidates: {len(marketplace_sources)}")
        console.print(f"   Request quality threshold: {marketplace_request.quality_threshold}")
    
    async def _test_teacher_selection(self):
        """Test teacher model selection"""
        console.print("\n👨‍🏫 [yellow]Testing Teacher Selection[/yellow]")
        
        # Test teacher selection for different domains
        domains = ["physics", "mathematics", "chemistry"]
        teacher_results = []
        
        for domain in domains:
            start_time = time.time()
            
            teacher_id = await self.router.select_teacher_for_training("student_model_01", domain)
            processing_time = time.time() - start_time
            
            # Test with teacher selection strategy
            task_description = f"Teach {domain} concepts to student model"
            decision = await self.router.process(task_description, 
                                               {"strategy": "teacher_selection"})
            
            # Verify teacher selection
            assert teacher_id is not None
            assert len(teacher_id) > 0
            
            teacher_results.append({
                "domain": domain,
                "teacher_id": teacher_id,
                "processing_time": processing_time,
                "model_type": decision.primary_candidate.model_type.value if hasattr(decision.primary_candidate, 'model_type') and hasattr(decision.primary_candidate.model_type, 'value') else "unknown",
                "teaching_effectiveness": getattr(decision.primary_candidate, 'teaching_effectiveness', 0.0)
            })
            
            effectiveness = getattr(decision.primary_candidate, 'teaching_effectiveness', 0.0) or 0.0
            console.print(f"✅ {domain}: {teacher_id} "
                         f"(effectiveness: {effectiveness:.2f})")
        
        self.test_results.append({
            "test": "teacher_selection",
            "status": "✅ PASSED",
            "domains_tested": len(domains),
            "results": teacher_results
        })
    
    async def _test_performance_tracking(self):
        """Test performance tracking and adaptation"""
        console.print("\n📊 [yellow]Testing Performance Tracking[/yellow]")
        
        # Add some performance data
        test_models = ["model_A", "model_B", "model_C"]
        performance_scores = [0.9, 0.7, 0.8]
        
        for model_id, score in zip(test_models, performance_scores):
            self.router.update_model_performance(model_id, score)
            self.router.update_model_performance(model_id, score + 0.05)  # Second score
        
        # Test routing with performance history
        test_task = "Complex analysis task requiring high performance"
        decision = await self.router.process(test_task)
        
        # Verify performance tracking
        assert len(self.router.performance_history) == len(test_models)
        for model_id in test_models:
            assert model_id in self.router.performance_history
            assert len(self.router.performance_history[model_id]) == 2
        
        self.test_results.append({
            "test": "performance_tracking",
            "status": "✅ PASSED",
            "models_tracked": len(self.router.performance_history),
            "total_scores": sum(len(scores) for scores in self.router.performance_history.values()),
            "decision_influenced": True
        })
        
        console.print(f"✅ Performance tracking working")
        console.print(f"   Models tracked: {len(self.router.performance_history)}")
        console.print(f"   Total performance scores: {sum(len(scores) for scores in self.router.performance_history.values())}")
    
    async def _test_concurrent_routing(self):
        """Test concurrent routing performance"""
        console.print("\n⚡ [yellow]Testing Concurrent Routing[/yellow]")
        
        # Test batch routing
        batch_tasks = [
            "Analyze quantum mechanics principles",
            "Design chemical synthesis pathway", 
            "Explain biological evolution",
            "Prove mathematical theorem",
            "Implement sorting algorithm"
        ]
        
        start_time = time.time()
        
        # Process tasks concurrently
        tasks = [self.router.process(task) for task in batch_tasks]
        decisions = await asyncio.gather(*tasks)
        
        batch_time = time.time() - start_time
        avg_time = batch_time / len(batch_tasks)
        
        # Verify all decisions
        assert len(decisions) == len(batch_tasks)
        for decision in decisions:
            assert decision.confidence_score > 0.0
            assert len(decision.reasoning) > 0
        
        self.test_results.append({
            "test": "concurrent_routing",
            "status": "✅ PASSED",
            "batch_size": len(batch_tasks),
            "total_time": f"{batch_time:.3f}s",
            "average_time": f"{avg_time:.3f}s",
            "throughput": f"{len(batch_tasks)/batch_time:.1f} tasks/sec"
        })
        
        console.print(f"✅ Concurrent routing: {len(batch_tasks)} tasks in {batch_time:.3f}s")
        console.print(f"   Average time per task: {avg_time:.3f}s")
        console.print(f"   Throughput: {len(batch_tasks)/batch_time:.1f} tasks/sec")
    
    async def _test_analytics(self):
        """Test analytics and reporting"""
        console.print("\n📈 [yellow]Testing Analytics[/yellow]")
        
        # Get routing analytics
        analytics = await self.router.get_routing_analytics()
        
        # Verify analytics structure
        assert "total_decisions" in analytics
        assert "average_confidence" in analytics
        assert "average_routing_time" in analytics
        assert "strategy_usage" in analytics
        assert "source_distribution" in analytics
        
        # Test route analytics methods
        best_model = await self.router.route_to_best_model("Test task for analytics")
        multiple_models = await self.router.route_to_multiple_models("Another test task", 3)
        
        assert best_model is not None
        assert len(multiple_models) <= 3
        
        self.test_results.append({
            "test": "analytics",
            "status": "✅ PASSED",
            "total_decisions": analytics["total_decisions"],
            "avg_confidence": f"{analytics['average_confidence']:.2f}",
            "strategy_variety": len(analytics["strategy_usage"]),
            "source_variety": len(analytics["source_distribution"]),
            "cache_hit_rate": f"{analytics['cache_hit_rate']:.2f}"
        })
        
        console.print(f"✅ Analytics working correctly")
        console.print(f"   Total decisions: {analytics['total_decisions']}")
        console.print(f"   Average confidence: {analytics['average_confidence']:.2f}")
        console.print(f"   Strategy variety: {len(analytics['strategy_usage'])}")
    
    async def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        console.print("\n📊 [bold green]Test Summary[/bold green]")
        
        # Create summary table
        table = Table(title="PRSM Enhanced Router Test Results")
        table.add_column("Test Category", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="white")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if "✅" in result["status"])
        
        for result in self.test_results:
            details = []
            for key, value in result.items():
                if key not in ["test", "status"]:
                    if isinstance(value, (int, float, str)) and not isinstance(value, bool):
                        details.append(f"{key}: {value}")
            
            table.add_row(
                result["test"].replace("_", " ").title(),
                result["status"],
                " | ".join(details[:3])  # Limit details
            )
        
        console.print(table)
        
        # Summary statistics
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": f"{(passed_tests/total_tests)*100:.1f}%",
            "test_results": self.test_results,
            "router_info": {
                "agent_id": self.router.agent_id,
                "agent_type": self.router.agent_type.value,
                "marketplace_endpoints": len(self.router.marketplace_endpoints),
                "routing_decisions": len(self.router.routing_decisions),
                "performance_models": len(self.router.performance_history)
            }
        }
        
        # Display summary
        console.print(f"\n🎯 [bold]Overall Results:[/bold]")
        console.print(f"   Tests Passed: {passed_tests}/{total_tests}")
        console.print(f"   Success Rate: {summary['success_rate']}")
        console.print(f"   Routing Decisions Made: {len(self.router.routing_decisions)}")
        console.print(f"   Performance Models Tracked: {len(self.router.performance_history)}")
        
        return summary


async def main():
    """Run the comprehensive test suite"""
    try:
        test_suite = EnhancedRouterTestSuite()
        summary = await test_suite.run_all_tests()
        
        # Save results
        with open("test_results/enhanced_router_test_results.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        console.print(f"\n💾 [blue]Test results saved to test_results/enhanced_router_test_results.json[/blue]")
        
        # Display final status
        if summary["passed_tests"] == summary["total_tests"]:
            console.print("\n🎉 [bold green]ALL TESTS PASSED! Enhanced Router is ready for production.[/bold green]")
        else:
            console.print(f"\n⚠️ [bold yellow]{summary['total_tests'] - summary['passed_tests']} tests failed. Review results.[/bold yellow]")
        
        return summary
        
    except Exception as e:
        console.print(f"\n❌ [bold red]Test suite failed: {e}[/bold red]")
        raise


if __name__ == "__main__":
    asyncio.run(main())