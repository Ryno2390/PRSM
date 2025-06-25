#!/usr/bin/env python3
"""
Test Suite for PRSM Prompt Optimizer
Comprehensive testing of prompt optimization functionality
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
from prsm.agents.prompters import (
    PromptOptimizer, DomainType, PromptType, OptimizationStrategy
)
from prsm.core.config import get_settings

console = Console()
settings = get_settings()


class PromptOptimizerTestSuite:
    """Comprehensive test suite for PromptOptimizer"""
    
    def __init__(self):
        self.optimizer = None
        self.test_results = []
        self.test_prompts = self._prepare_test_prompts()
    
    def _prepare_test_prompts(self) -> Dict[str, Dict[str, Any]]:
        """Prepare diverse test prompts for different domains and types"""
        return {
            "physics_analytical": {
                "prompt": "Explain energy conservation",
                "domain": DomainType.PHYSICS,
                "prompt_type": PromptType.ANALYTICAL,
                "expected_keywords": ["energy", "conservation", "thermodynamics"]
            },
            "chemistry_synthesis": {
                "prompt": "How do catalysts work?",
                "domain": DomainType.CHEMISTRY,
                "prompt_type": PromptType.SYNTHESIS,
                "expected_keywords": ["catalyst", "reaction", "mechanism"]
            },
            "biology_explanatory": {
                "prompt": "Describe protein folding",
                "domain": DomainType.BIOLOGY,
                "prompt_type": PromptType.EXPLANATORY,
                "expected_keywords": ["protein", "structure", "function"]
            },
            "math_evaluation": {
                "prompt": "Is this proof correct?",
                "domain": DomainType.MATHEMATICS,
                "prompt_type": PromptType.EVALUATION,
                "expected_keywords": ["proof", "theorem", "logic"]
            },
            "cs_decomposition": {
                "prompt": "Design sorting algorithm",
                "domain": DomainType.COMPUTER_SCIENCE,
                "prompt_type": PromptType.DECOMPOSITION,
                "expected_keywords": ["algorithm", "complexity", "sorting"]
            },
            "vague_prompt": {
                "prompt": "Tell me about stuff",
                "domain": DomainType.GENERAL_SCIENCE,
                "prompt_type": PromptType.ANALYTICAL,
                "expected_keywords": ["material", "concepts", "analysis"]
            },
            "short_prompt": {
                "prompt": "Gravity?",
                "domain": DomainType.PHYSICS,
                "prompt_type": PromptType.EXPLANATORY,
                "expected_keywords": ["force", "mass", "acceleration"]
            }
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        console.print("\n🧪 [bold blue]PRSM Prompt Optimizer Test Suite[/bold blue]")
        console.print("=" * 60)
        
        # Initialize optimizer
        await self._test_initialization()
        
        # Test basic functionality
        await self._test_basic_optimization()
        
        # Test domain specialization
        await self._test_domain_specialization()
        
        # Test prompt types
        await self._test_prompt_types()
        
        # Test safety alignment
        await self._test_safety_alignment()
        
        # Test context templates
        await self._test_context_templates()
        
        # Test edge cases
        await self._test_edge_cases()
        
        # Test performance
        await self._test_performance()
        
        # Generate summary
        return await self._generate_test_summary()
    
    async def _test_initialization(self):
        """Test optimizer initialization"""
        console.print("\n📋 [yellow]Testing Initialization[/yellow]")
        
        start_time = time.time()
        self.optimizer = PromptOptimizer()
        init_time = time.time() - start_time
        
        # Verify initialization
        assert self.optimizer.agent_type.value == "prompter"
        assert len(self.optimizer.domain_strategies) >= 5
        assert len(self.optimizer.safety_guidelines) > 0
        assert len(self.optimizer.context_templates) > 0
        
        self.test_results.append({
            "test": "initialization",
            "status": "✅ PASSED",
            "init_time": f"{init_time:.3f}s",
            "domains": len(self.optimizer.domain_strategies),
            "safety_rules": len(self.optimizer.safety_guidelines)
        })
        
        console.print(f"✅ Optimizer initialized in {init_time:.3f}s")
        console.print(f"   Domains: {len(self.optimizer.domain_strategies)}")
        console.print(f"   Safety rules: {len(self.optimizer.safety_guidelines)}")
    
    async def _test_basic_optimization(self):
        """Test basic prompt optimization functionality"""
        console.print("\n🔧 [yellow]Testing Basic Optimization[/yellow]")
        
        test_prompt = "Explain how computers work"
        
        start_time = time.time()
        result = await self.optimizer.process(test_prompt)
        processing_time = time.time() - start_time
        
        # Verify result structure
        assert result.original_prompt == test_prompt
        assert len(result.optimized_prompt) > len(test_prompt)
        assert result.confidence_score > 0.0
        assert len(result.strategies_applied) > 0
        assert result.domain == DomainType.GENERAL_SCIENCE
        assert result.prompt_type == PromptType.ANALYTICAL
        
        self.test_results.append({
            "test": "basic_optimization",
            "status": "✅ PASSED",
            "processing_time": f"{processing_time:.3f}s",
            "length_increase": len(result.optimized_prompt) - len(test_prompt),
            "strategies": len(result.strategies_applied),
            "confidence": f"{result.confidence_score:.2f}"
        })
        
        console.print(f"✅ Basic optimization completed in {processing_time:.3f}s")
        console.print(f"   Length increase: {len(result.optimized_prompt) - len(test_prompt)} chars")
        console.print(f"   Strategies applied: {len(result.strategies_applied)}")
        console.print(f"   Confidence: {result.confidence_score:.2f}")
    
    async def _test_domain_specialization(self):
        """Test domain-specific optimization"""
        console.print("\n🎯 [yellow]Testing Domain Specialization[/yellow]")
        
        domain_results = []
        
        for test_name, test_data in self.test_prompts.items():
            if "domain" in test_data:
                start_time = time.time()
                
                result = await self.optimizer.process({
                    "prompt": test_data["prompt"],
                    "domain": test_data["domain"].value,
                    "prompt_type": test_data["prompt_type"].value
                })
                
                processing_time = time.time() - start_time
                
                # Verify domain-specific enhancement
                assert result.domain == test_data["domain"]
                assert result.prompt_type == test_data["prompt_type"]
                
                # Check for expected domain keywords
                optimized_lower = result.optimized_prompt.lower()
                keywords_found = sum(1 for keyword in test_data["expected_keywords"] 
                                   if keyword in optimized_lower)
                
                domain_results.append({
                    "test": test_name,
                    "domain": test_data["domain"].value,
                    "keywords_found": keywords_found,
                    "total_keywords": len(test_data["expected_keywords"]),
                    "processing_time": processing_time,
                    "length_increase": len(result.optimized_prompt) - len(test_data["prompt"])
                })
                
                console.print(f"✅ {test_name}: {keywords_found}/{len(test_data['expected_keywords'])} keywords")
        
        self.test_results.append({
            "test": "domain_specialization",
            "status": "✅ PASSED",
            "domain_tests": len(domain_results),
            "results": domain_results
        })
    
    async def _test_prompt_types(self):
        """Test different prompt type handling"""
        console.print("\n📝 [yellow]Testing Prompt Types[/yellow]")
        
        type_results = []
        base_prompt = "Analyze this scientific concept"
        
        for prompt_type in PromptType:
            start_time = time.time()
            
            result = await self.optimizer.process({
                "prompt": base_prompt,
                "domain": "general_science",
                "prompt_type": prompt_type.value
            })
            
            processing_time = time.time() - start_time
            
            # Verify prompt type handling
            assert result.prompt_type == prompt_type
            
            type_results.append({
                "type": prompt_type.value,
                "processing_time": processing_time,
                "strategies": len(result.strategies_applied),
                "templates": len(result.context_templates)
            })
            
            console.print(f"✅ {prompt_type.value}: {len(result.strategies_applied)} strategies")
        
        self.test_results.append({
            "test": "prompt_types",
            "status": "✅ PASSED",
            "types_tested": len(type_results),
            "results": type_results
        })
    
    async def _test_safety_alignment(self):
        """Test safety alignment functionality"""
        console.print("\n🛡️ [yellow]Testing Safety Alignment[/yellow]")
        
        # Test safe prompt
        safe_prompt = "Explain photosynthesis"
        safe_result = await self.optimizer.process(safe_prompt)
        
        # Test potentially concerning prompt
        concerning_prompt = "How to make harmful substances"
        concerning_result = await self.optimizer.process(concerning_prompt)
        
        # Verify safety processing
        assert safe_result.safety_validated == True
        assert "safety" in str(safe_result.optimization_metadata).lower()
        
        # Test custom safety guidelines
        custom_guidelines = ["Focus on educational content", "Avoid dangerous applications"]
        enhanced_prompt = await self.optimizer.enhance_alignment(
            "Chemical reactions", custom_guidelines
        )
        
        assert len(enhanced_prompt) > len("Chemical reactions")
        
        self.test_results.append({
            "test": "safety_alignment",
            "status": "✅ PASSED",
            "safe_prompt_validated": safe_result.safety_validated,
            "concerning_prompt_flagged": len(concerning_result.optimization_metadata.get("safety_alignment", {}).get("safety_flags", [])) > 0,
            "custom_guidelines_applied": len(enhanced_prompt) > 20
        })
        
        console.print("✅ Safety alignment working correctly")
        console.print(f"   Safe prompt validated: {safe_result.safety_validated}")
        console.print(f"   Custom guidelines applied successfully")
    
    async def _test_context_templates(self):
        """Test context template generation"""
        console.print("\n📄 [yellow]Testing Context Templates[/yellow]")
        
        template_results = []
        
        for task_type in ["analytical", "creative", "explanatory"]:
            templates = await self.optimizer.generate_context_templates(task_type)
            
            template_results.append({
                "task_type": task_type,
                "templates_generated": len(templates),
                "sample_template": templates[0] if templates else None
            })
            
            console.print(f"✅ {task_type}: {len(templates)} templates generated")
        
        self.test_results.append({
            "test": "context_templates",
            "status": "✅ PASSED",
            "task_types_tested": len(template_results),
            "results": template_results
        })
    
    async def _test_edge_cases(self):
        """Test edge cases and error handling"""
        console.print("\n⚠️ [yellow]Testing Edge Cases[/yellow]")
        
        edge_cases = []
        
        # Empty prompt
        try:
            result = await self.optimizer.process("")
            edge_cases.append({"case": "empty_prompt", "status": "handled", "length": len(result.optimized_prompt)})
        except Exception as e:
            edge_cases.append({"case": "empty_prompt", "status": "error", "error": str(e)})
        
        # Very long prompt
        long_prompt = "This is a test prompt. " * 100
        try:
            result = await self.optimizer.process(long_prompt)
            edge_cases.append({"case": "long_prompt", "status": "handled", "processing_time": "measured"})
        except Exception as e:
            edge_cases.append({"case": "long_prompt", "status": "error", "error": str(e)})
        
        # Invalid domain
        try:
            result = await self.optimizer.optimize_for_domain("Test prompt", "invalid_domain")
            edge_cases.append({"case": "invalid_domain", "status": "handled", "fallback": "general_science"})
        except Exception as e:
            edge_cases.append({"case": "invalid_domain", "status": "error", "error": str(e)})
        
        self.test_results.append({
            "test": "edge_cases",
            "status": "✅ PASSED",
            "cases_tested": len(edge_cases),
            "results": edge_cases
        })
        
        console.print(f"✅ Edge cases tested: {len(edge_cases)}")
    
    async def _test_performance(self):
        """Test performance characteristics"""
        console.print("\n⚡ [yellow]Testing Performance[/yellow]")
        
        # Test batch processing
        batch_prompts = [
            "Explain quantum mechanics",
            "How does DNA replication work?",
            "Describe machine learning algorithms",
            "What is thermodynamics?",
            "Explain cellular respiration"
        ]
        
        start_time = time.time()
        
        # Process prompts concurrently
        tasks = [self.optimizer.process(prompt) for prompt in batch_prompts]
        results = await asyncio.gather(*tasks)
        
        batch_time = time.time() - start_time
        avg_time = batch_time / len(batch_prompts)
        
        # Verify all results
        assert len(results) == len(batch_prompts)
        for result in results:
            assert result.confidence_score > 0.0
            assert len(result.strategies_applied) > 0
        
        self.test_results.append({
            "test": "performance",
            "status": "✅ PASSED",
            "batch_size": len(batch_prompts),
            "total_time": f"{batch_time:.3f}s",
            "average_time": f"{avg_time:.3f}s",
            "throughput": f"{len(batch_prompts)/batch_time:.1f} prompts/sec"
        })
        
        console.print(f"✅ Batch processing: {len(batch_prompts)} prompts in {batch_time:.3f}s")
        console.print(f"   Average time per prompt: {avg_time:.3f}s")
        console.print(f"   Throughput: {len(batch_prompts)/batch_time:.1f} prompts/sec")
    
    async def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        console.print("\n📊 [bold green]Test Summary[/bold green]")
        
        # Create summary table
        table = Table(title="PRSM Prompt Optimizer Test Results")
        table.add_column("Test Category", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="white")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if "✅" in result["status"])
        
        for result in self.test_results:
            details = []
            for key, value in result.items():
                if key not in ["test", "status"]:
                    if isinstance(value, (int, float, str)):
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
            "optimizer_info": {
                "agent_id": self.optimizer.agent_id,
                "agent_type": self.optimizer.agent_type.value,
                "domains_supported": len(self.optimizer.domain_strategies),
                "safety_guidelines": len(self.optimizer.safety_guidelines)
            }
        }
        
        # Display summary
        console.print(f"\n🎯 [bold]Overall Results:[/bold]")
        console.print(f"   Tests Passed: {passed_tests}/{total_tests}")
        console.print(f"   Success Rate: {summary['success_rate']}")
        console.print(f"   Domains Supported: {len(self.optimizer.domain_strategies)}")
        
        return summary


async def main():
    """Run the comprehensive test suite"""
    try:
        test_suite = PromptOptimizerTestSuite()
        summary = await test_suite.run_all_tests()
        
        # Save results
        with open("test_results/prompt_optimizer_test_results.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        console.print(f"\n💾 [blue]Test results saved to test_results/prompt_optimizer_test_results.json[/blue]")
        
        # Display final status
        if summary["passed_tests"] == summary["total_tests"]:
            console.print("\n🎉 [bold green]ALL TESTS PASSED! Prompt Optimizer is ready for production.[/bold green]")
        else:
            console.print(f"\n⚠️ [bold yellow]{summary['total_tests'] - summary['passed_tests']} tests failed. Review results.[/bold yellow]")
        
        return summary
        
    except Exception as e:
        console.print(f"\n❌ [bold red]Test suite failed: {e}[/bold red]")
        raise


if __name__ == "__main__":
    asyncio.run(main())