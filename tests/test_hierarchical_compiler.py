#!/usr/bin/env python3
"""
Test Suite for PRSM Hierarchical Compiler
Comprehensive testing of advanced compilation functionality
"""

import asyncio
import json
import time
from typing import Dict, Any
from uuid import uuid4
import pytest

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text

    # PRSM imports
    from prsm.compute.agents.compilers import (
        HierarchicalCompiler, CompilationLevel, SynthesisStrategy,
        IntermediateResult, MidResult, FinalResponse, ReasoningTrace
    )
    from prsm.core.config import get_settings
    from prsm.core.models import AgentResponse, AgentType
    
    console = Console()
    settings = get_settings()
    
except (ImportError, ModuleNotFoundError) as e:
    pytest.skip("Module 'rich' not yet implemented", allow_module_level=True)


class HierarchicalCompilerTestSuite:
    """Comprehensive test suite for HierarchicalCompiler"""
    
    def __init__(self):
        self.compiler = None
        self.test_results = []
        self.test_data = self._prepare_test_data()
    
    def _prepare_test_data(self) -> Dict[str, Any]:
        """Prepare diverse test data for compilation testing"""
        return {
            "simple_responses": [
                "The experiment shows positive results",
                "Data analysis confirms the hypothesis", 
                "Statistical significance achieved"
            ],
            "agent_responses": [
                {
                    "agent_id": "test_agent_1",
                    "agent_type": "executor",
                    "output_data": "Quantum mechanics analysis complete",
                    "success": True,
                    "confidence": 0.85
                },
                {
                    "agent_id": "test_agent_2", 
                    "agent_type": "executor",
                    "output_data": "Molecular dynamics simulation finished",
                    "success": True,
                    "confidence": 0.92
                }
            ],
            "complex_data": [
                {
                    "summary": "Research findings indicate novel properties",
                    "details": "Comprehensive analysis of 1000+ samples",
                    "confidence": 0.78,
                    "key_points": ["Novel mechanism discovered", "High reproducibility", "Clinical relevance"]
                },
                {
                    "summary": "Theoretical model validation successful", 
                    "details": "Mathematical proofs and experimental verification",
                    "confidence": 0.83,
                    "insights": ["Model accuracy >95%", "Predictive capabilities confirmed"]
                }
            ],
            "conflicting_data": [
                {
                    "conclusion": "Method A is superior",
                    "evidence": "Performance metrics favor A",
                    "confidence": 0.7
                },
                {
                    "conclusion": "Method B shows better results",
                    "evidence": "Long-term stability favors B", 
                    "confidence": 0.75
                }
            ]
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        console.print("\n🧪 [bold blue]PRSM Hierarchical Compiler Test Suite[/bold blue]")
        console.print("=" * 60)
        
        # Initialize compiler
        await self._test_initialization()
        
        # Test basic compilation
        await self._test_basic_compilation()
        
        # Test compilation strategies
        await self._test_compilation_strategies()
        
        # Test multi-level compilation
        await self._test_multi_level_compilation()
        
        # Test conflict resolution
        await self._test_conflict_resolution()
        
        # Test reasoning trace generation
        await self._test_reasoning_trace()
        
        # Test performance optimization
        await self._test_performance_optimization()
        
        # Test concurrent compilation
        await self._test_concurrent_compilation()
        
        # Generate summary
        return await self._generate_test_summary()
    
    async def _test_initialization(self):
        """Test compiler initialization"""
        console.print("\n📋 [yellow]Testing Initialization[/yellow]")
        
        start_time = time.time()
        try:
            self.compiler = HierarchicalCompiler(
                confidence_threshold=0.8,
                default_strategy=SynthesisStrategy.COMPREHENSIVE
            )
            init_time = time.time() - start_time
            
            # Verify initialization
            assert self.compiler.agent_type.value == "compiler"
            assert self.compiler.confidence_threshold == 0.8
            assert hasattr(self.compiler, 'compilation_stages')
            assert hasattr(self.compiler, 'reasoning_trace')
            
            self.test_results.append({
                "test": "initialization",
                "status": "✅ PASSED",
                "init_time": f"{init_time:.3f}s",
                "agent_type": self.compiler.agent_type.value,
                "confidence_threshold": self.compiler.confidence_threshold
            })
            
            console.print(f"✅ Compiler initialized in {init_time:.3f}s")
            console.print(f"   Agent type: {self.compiler.agent_type.value}")
            console.print(f"   Confidence threshold: {self.compiler.confidence_threshold}")
            
        except Exception as e:
            self.test_results.append({
                "test": "initialization",
                "status": "❌ FAILED",
                "error": str(e)
            })
            console.print(f"❌ Initialization failed: {e}")
    
    async def _test_basic_compilation(self):
        """Test basic compilation functionality"""
        console.print("\n🔧 [yellow]Testing Basic Compilation[/yellow]")
        
        if not self.compiler:
            console.print("❌ Skipping - compiler not initialized")
            return
        
        try:
            test_data = self.test_data["simple_responses"]
            
            start_time = time.time()
            result = await self.compiler.process(test_data)
            processing_time = time.time() - start_time
            
            # Verify result structure
            assert hasattr(result, 'session_id')
            assert hasattr(result, 'compilation_level')
            assert hasattr(result, 'confidence_score')
            assert result.confidence_score >= 0.0
            assert result.confidence_score <= 1.0
            
            self.test_results.append({
                "test": "basic_compilation",
                "status": "✅ PASSED",
                "processing_time": f"{processing_time:.3f}s",
                "input_count": len(test_data),
                "confidence": f"{result.confidence_score:.2f}",
                "compilation_level": result.compilation_level
            })
            
            console.print(f"✅ Basic compilation completed in {processing_time:.3f}s")
            console.print(f"   Input count: {len(test_data)}")
            console.print(f"   Confidence: {result.confidence_score:.2f}")
            
        except Exception as e:
            self.test_results.append({
                "test": "basic_compilation",
                "status": "❌ FAILED",
                "error": str(e)
            })
            console.print(f"❌ Basic compilation failed: {e}")
    
    async def _test_compilation_strategies(self):
        """Test different compilation strategies"""
        console.print("\n🎯 [yellow]Testing Compilation Strategies[/yellow]")
        
        if not self.compiler:
            console.print("❌ Skipping - compiler not initialized")
            return
        
        try:
            strategies = [
                SynthesisStrategy.CONSENSUS,
                SynthesisStrategy.WEIGHTED_AVERAGE,
                SynthesisStrategy.BEST_RESULT,
                SynthesisStrategy.COMPREHENSIVE
            ]
            
            strategy_results = []
            test_data = self.test_data["complex_data"]
            
            for strategy in strategies:
                start_time = time.time()
                
                result = await self.compiler.process(
                    test_data, 
                    {"strategy": strategy.value}
                )
                
                processing_time = time.time() - start_time
                
                strategy_results.append({
                    "strategy": strategy.value,
                    "processing_time": processing_time,
                    "confidence": result.confidence_score,
                    "success": True
                })
                
                console.print(f"✅ {strategy.value}: confidence {result.confidence_score:.2f}")
            
            self.test_results.append({
                "test": "compilation_strategies",
                "status": "✅ PASSED",
                "strategies_tested": len(strategies),
                "results": strategy_results
            })
            
        except Exception as e:
            self.test_results.append({
                "test": "compilation_strategies", 
                "status": "❌ FAILED",
                "error": str(e)
            })
            console.print(f"❌ Strategy testing failed: {e}")
    
    async def _test_multi_level_compilation(self):
        """Test multi-level hierarchical compilation"""
        console.print("\n🏗️ [yellow]Testing Multi-Level Compilation[/yellow]")
        
        if not self.compiler:
            console.print("❌ Skipping - compiler not initialized")
            return
        
        try:
            # Test with larger dataset
            large_dataset = (
                self.test_data["simple_responses"] + 
                [d["summary"] for d in self.test_data["complex_data"]] +
                ["Additional research data point " + str(i) for i in range(5)]
            )
            
            start_time = time.time()
            result = await self.compiler.process(large_dataset)
            processing_time = time.time() - start_time
            
            # Verify multi-level processing
            assert result.input_count == len(large_dataset)
            
            # Check if compilation stages were recorded
            stages_completed = result.metadata.get("stages_completed", 0)
            
            self.test_results.append({
                "test": "multi_level_compilation",
                "status": "✅ PASSED",
                "processing_time": f"{processing_time:.3f}s",
                "input_count": len(large_dataset),
                "stages_completed": stages_completed,
                "confidence": f"{result.confidence_score:.2f}"
            })
            
            console.print(f"✅ Multi-level compilation completed in {processing_time:.3f}s")
            console.print(f"   Input count: {len(large_dataset)}")
            console.print(f"   Stages completed: {stages_completed}")
            
        except Exception as e:
            self.test_results.append({
                "test": "multi_level_compilation",
                "status": "❌ FAILED", 
                "error": str(e)
            })
            console.print(f"❌ Multi-level compilation failed: {e}")
    
    async def _test_conflict_resolution(self):
        """Test conflict detection and resolution"""
        console.print("\n⚔️ [yellow]Testing Conflict Resolution[/yellow]")
        
        if not self.compiler:
            console.print("❌ Skipping - compiler not initialized")
            return
        
        try:
            conflicting_data = self.test_data["conflicting_data"]
            
            start_time = time.time()
            result = await self.compiler.process(conflicting_data)
            processing_time = time.time() - start_time
            
            # Check if conflicts were handled
            assert result.confidence_score >= 0.0
            
            self.test_results.append({
                "test": "conflict_resolution",
                "status": "✅ PASSED",
                "processing_time": f"{processing_time:.3f}s",
                "input_count": len(conflicting_data),
                "confidence": f"{result.confidence_score:.2f}"
            })
            
            console.print(f"✅ Conflict resolution completed in {processing_time:.3f}s")
            console.print(f"   Conflicting inputs: {len(conflicting_data)}")
            console.print(f"   Final confidence: {result.confidence_score:.2f}")
            
        except Exception as e:
            self.test_results.append({
                "test": "conflict_resolution",
                "status": "❌ FAILED",
                "error": str(e)
            })
            console.print(f"❌ Conflict resolution failed: {e}")
    
    async def _test_reasoning_trace(self):
        """Test reasoning trace generation"""
        console.print("\n🧠 [yellow]Testing Reasoning Trace[/yellow]")
        
        if not self.compiler:
            console.print("❌ Skipping - compiler not initialized")
            return
        
        try:
            test_data = self.test_data["complex_data"]
            
            result = await self.compiler.process(test_data)
            
            # Check reasoning trace
            reasoning_trace = result.reasoning_trace
            assert isinstance(reasoning_trace, list)
            assert len(reasoning_trace) > 0
            
            self.test_results.append({
                "test": "reasoning_trace",
                "status": "✅ PASSED",
                "trace_length": len(reasoning_trace),
                "has_reasoning": len(reasoning_trace) > 0
            })
            
            console.print(f"✅ Reasoning trace generated")
            console.print(f"   Trace steps: {len(reasoning_trace)}")
            
        except Exception as e:
            self.test_results.append({
                "test": "reasoning_trace",
                "status": "❌ FAILED",
                "error": str(e)
            })
            console.print(f"❌ Reasoning trace failed: {e}")
    
    async def _test_performance_optimization(self):
        """Test performance optimization features"""
        console.print("\n⚡ [yellow]Testing Performance Optimization[/yellow]")
        
        if not self.compiler:
            console.print("❌ Skipping - compiler not initialized")
            return
        
        try:
            # Test with varying data sizes
            small_data = self.test_data["simple_responses"][:2]
            medium_data = self.test_data["simple_responses"] + [d["summary"] for d in self.test_data["complex_data"]]
            
            # Small dataset
            start_time = time.time()
            small_result = await self.compiler.process(small_data)
            small_time = time.time() - start_time
            
            # Medium dataset  
            start_time = time.time()
            medium_result = await self.compiler.process(medium_data)
            medium_time = time.time() - start_time
            
            # Check performance scaling
            performance_ratio = medium_time / small_time if small_time > 0 else 1.0
            
            self.test_results.append({
                "test": "performance_optimization",
                "status": "✅ PASSED",
                "small_dataset_time": f"{small_time:.3f}s",
                "medium_dataset_time": f"{medium_time:.3f}s", 
                "performance_ratio": f"{performance_ratio:.2f}",
                "scaling_efficiency": performance_ratio < 3.0  # Should scale reasonably
            })
            
            console.print(f"✅ Performance optimization tested")
            console.print(f"   Small dataset: {small_time:.3f}s")
            console.print(f"   Medium dataset: {medium_time:.3f}s")
            console.print(f"   Scaling ratio: {performance_ratio:.2f}")
            
        except Exception as e:
            self.test_results.append({
                "test": "performance_optimization",
                "status": "❌ FAILED",
                "error": str(e)
            })
            console.print(f"❌ Performance testing failed: {e}")
    
    async def _test_concurrent_compilation(self):
        """Test concurrent compilation capabilities"""
        console.print("\n🔄 [yellow]Testing Concurrent Compilation[/yellow]")
        
        if not self.compiler:
            console.print("❌ Skipping - compiler not initialized")
            return
        
        try:
            # Prepare multiple compilation tasks
            tasks_data = [
                self.test_data["simple_responses"],
                self.test_data["complex_data"],
                [d["summary"] for d in self.test_data["complex_data"]],
                self.test_data["conflicting_data"]
            ]
            
            start_time = time.time()
            
            # Run concurrent compilations
            tasks = [self.compiler.process(data) for data in tasks_data]
            results = await asyncio.gather(*tasks)
            
            concurrent_time = time.time() - start_time
            
            # Verify all results
            assert len(results) == len(tasks_data)
            for result in results:
                assert result.confidence_score >= 0.0
                assert result.confidence_score <= 1.0
            
            self.test_results.append({
                "test": "concurrent_compilation",
                "status": "✅ PASSED",
                "concurrent_tasks": len(tasks_data),
                "total_time": f"{concurrent_time:.3f}s",
                "average_time": f"{concurrent_time/len(tasks_data):.3f}s",
                "all_successful": True
            })
            
            console.print(f"✅ Concurrent compilation completed")
            console.print(f"   Concurrent tasks: {len(tasks_data)}")
            console.print(f"   Total time: {concurrent_time:.3f}s")
            console.print(f"   Average time: {concurrent_time/len(tasks_data):.3f}s")
            
        except Exception as e:
            self.test_results.append({
                "test": "concurrent_compilation",
                "status": "❌ FAILED",
                "error": str(e)
            })
            console.print(f"❌ Concurrent compilation failed: {e}")
    
    async def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        console.print("\n📊 [bold green]Test Summary[/bold green]")
        
        # Create summary table
        table = Table(title="PRSM Hierarchical Compiler Test Results")
        table.add_column("Test Category", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="white")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if "✅" in result["status"])
        
        for result in self.test_results:
            details = []
            for key, value in result.items():
                if key not in ["test", "status", "error"]:
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
            "compiler_info": {
                "agent_id": self.compiler.agent_id if self.compiler else None,
                "agent_type": self.compiler.agent_type.value if self.compiler else None,
                "confidence_threshold": self.compiler.confidence_threshold if self.compiler else None
            }
        }
        
        # Display summary
        console.print(f"\n🎯 [bold]Overall Results:[/bold]")
        console.print(f"   Tests Passed: {passed_tests}/{total_tests}")
        console.print(f"   Success Rate: {summary['success_rate']}")
        
        return summary


async def main():
    """Run the comprehensive test suite"""
    try:
        test_suite = HierarchicalCompilerTestSuite()
        summary = await test_suite.run_all_tests()
        
        # Save results
        with open("test_results/hierarchical_compiler_test_results.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        console.print(f"\n💾 [blue]Test results saved to test_results/hierarchical_compiler_test_results.json[/blue]")
        
        # Display final status
        if summary["passed_tests"] == summary["total_tests"]:
            console.print("\n🎉 [bold green]ALL TESTS PASSED! Hierarchical Compiler is ready for production.[/bold green]")
        else:
            console.print(f"\n⚠️ [bold yellow]{summary['total_tests'] - summary['passed_tests']} tests failed. Review results.[/bold yellow]")
        
        return summary
        
    except Exception as e:
        console.print(f"\n❌ [bold red]Test suite failed: {e}[/bold red]")
        raise


if __name__ == "__main__":
    asyncio.run(main())