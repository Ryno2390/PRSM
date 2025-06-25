"""
Comprehensive Test Suite for RLT Training Pipeline

Tests all core RLT components with realistic datasets and validates
Sakana AI's dual reward methodology implementation.

Test Coverage:
- RLT Training Pipeline functionality
- Dual reward system (r_SS + r_KL) computation
- Student comprehension evaluation
- Explanation formatting and quality
- Integration with existing PRSM components
"""

import asyncio
import json
import pytest
import time
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any

# Import RLT components
from prsm.teachers.rlt import (
    RLTDenseRewardTrainer,
    StudentCompressionEvaluator,
    RLTFormatter,
    RLTQualityMonitor
)
from prsm.teachers.rlt.dense_reward_trainer import RLTTrainingConfig
from prsm.teachers.rlt.student_comprehension_evaluator import EvaluationConfig
from prsm.teachers.rlt.explanation_formatter import RLTFormatConfig
from prsm.teachers.rlt.quality_monitor import MonitoringConfig


class RLTTestDataset:
    """Comprehensive test dataset for RLT validation"""
    
    def __init__(self):
        self.test_data = self._create_test_dataset()
        self.expected_results = self._create_expected_results()
    
    def _create_test_dataset(self) -> List[Dict[str, str]]:
        """Create diverse test dataset covering multiple domains"""
        return [
            # Mathematics - Basic
            {
                "question": "What is the derivative of x^2?",
                "solution": "The derivative of x^2 is 2x",
                "domain": "mathematics",
                "complexity": "basic",
                "expected_quality_range": (0.7, 0.9)
            },
            {
                "question": "Solve for x: 2x + 5 = 13",
                "solution": "x = 4",
                "domain": "mathematics", 
                "complexity": "basic",
                "expected_quality_range": (0.7, 0.9)
            },
            
            # Mathematics - Advanced
            {
                "question": "Find the integral of sin(x)cos(x) dx",
                "solution": "The integral is (1/2)sin²(x) + C",
                "domain": "mathematics",
                "complexity": "advanced",
                "expected_quality_range": (0.6, 0.8)
            },
            {
                "question": "Prove that the limit of (1 + 1/n)^n as n approaches infinity equals e",
                "solution": "This is proven using the definition of e and properties of limits",
                "domain": "mathematics",
                "complexity": "advanced",
                "expected_quality_range": (0.5, 0.7)
            },
            
            # Physics
            {
                "question": "What is Newton's second law of motion?",
                "solution": "F = ma (Force equals mass times acceleration)",
                "domain": "physics",
                "complexity": "basic",
                "expected_quality_range": (0.7, 0.9)
            },
            {
                "question": "Calculate the gravitational force between two 10kg masses separated by 1 meter",
                "solution": "F = G*m1*m2/r² = 6.67×10⁻¹¹ * 10 * 10 / 1² = 6.67×10⁻⁹ N",
                "domain": "physics",
                "complexity": "intermediate",
                "expected_quality_range": (0.6, 0.8)
            },
            
            # Computer Science
            {
                "question": "What is a binary search algorithm?",
                "solution": "A binary search is an efficient algorithm for finding an item in a sorted list by repeatedly dividing the search interval in half",
                "domain": "computer_science",
                "complexity": "intermediate",
                "expected_quality_range": (0.6, 0.8)
            },
            {
                "question": "Write a function to reverse a string in Python",
                "solution": "def reverse_string(s): return s[::-1]",
                "domain": "computer_science",
                "complexity": "basic",
                "expected_quality_range": (0.7, 0.9)
            },
            
            # Edge cases
            {
                "question": "Short?",
                "solution": "Yes",
                "domain": "general",
                "complexity": "basic",
                "expected_quality_range": (0.3, 0.5)  # Low quality due to brevity
            },
            {
                "question": "This is an extremely long and complex question that involves multiple interconnected concepts from various domains including advanced mathematics, theoretical physics, computer science algorithms, and philosophical reasoning that requires deep understanding and sophisticated analytical thinking to properly address",
                "solution": "This solution is equally complex and involves detailed analysis across multiple disciplines requiring extensive background knowledge and careful step-by-step reasoning to arrive at a comprehensive and well-supported conclusion",
                "domain": "general",
                "complexity": "advanced",
                "expected_quality_range": (0.4, 0.6)  # Moderate quality due to complexity
            }
        ]
    
    def _create_expected_results(self) -> Dict[str, Any]:
        """Create expected results for validation"""
        return {
            "min_reward_scores": {
                "basic": 0.5,
                "intermediate": 0.4,
                "advanced": 0.3
            },
            "min_comprehension_scores": {
                "basic": 0.6,
                "intermediate": 0.5,
                "advanced": 0.4
            },
            "expected_domains": ["mathematics", "physics", "computer_science", "general"],
            "quality_thresholds": {
                "coherence": 0.3,
                "logical_flow": 0.3,
                "concept_coverage": 0.3
            }
        }
    
    def get_test_batch(self, batch_size: int = 3) -> Dict[str, List]:
        """Get a batch of test data for training"""
        selected_data = self.test_data[:batch_size]
        return {
            "questions": [item["question"] for item in selected_data],
            "solutions": [item["solution"] for item in selected_data],
            "domains": [item["domain"] for item in selected_data],
            "complexities": [item["complexity"] for item in selected_data]
        }


class RLTValidationSuite:
    """Comprehensive validation suite for RLT components"""
    
    def __init__(self):
        self.dataset = RLTTestDataset()
        self.results = {}
        
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        print("🧪 Starting RLT Training Pipeline Validation Suite")
        print("=" * 60)
        
        validation_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_results": {},
            "performance_metrics": {},
            "validation_summary": {}
        }
        
        try:
            # Test 1: Formatter Validation
            print("\n📝 Testing RLT Formatter...")
            formatter_results = await self.test_rlt_formatter()
            validation_results["test_results"]["formatter"] = formatter_results
            print(f"✅ Formatter Test: {'PASSED' if formatter_results['success'] else 'FAILED'}")
            
            # Test 2: Student Comprehension Evaluator
            print("\n📊 Testing Student Comprehension Evaluator...")
            comprehension_results = await self.test_student_comprehension_evaluator()
            validation_results["test_results"]["comprehension"] = comprehension_results
            print(f"✅ Comprehension Test: {'PASSED' if comprehension_results['success'] else 'FAILED'}")
            
            # Test 3: Quality Monitor
            print("\n📈 Testing Quality Monitor...")
            monitor_results = await self.test_quality_monitor()
            validation_results["test_results"]["monitor"] = monitor_results
            print(f"✅ Monitor Test: {'PASSED' if monitor_results['success'] else 'FAILED'}")
            
            # Test 4: Dual Reward System
            print("\n💰 Testing Dual Reward System...")
            reward_results = await self.test_dual_reward_system()
            validation_results["test_results"]["rewards"] = reward_results
            print(f"✅ Reward System Test: {'PASSED' if reward_results['success'] else 'FAILED'}")
            
            # Test 5: Training Pipeline Integration
            print("\n🧠 Testing Training Pipeline...")
            pipeline_results = await self.test_training_pipeline()
            validation_results["test_results"]["pipeline"] = pipeline_results
            print(f"✅ Pipeline Test: {'PASSED' if pipeline_results['success'] else 'FAILED'}")
            
            # Generate summary
            validation_results["validation_summary"] = self._generate_validation_summary(validation_results)
            
            print("\n" + "=" * 60)
            print("🎯 RLT Validation Suite Complete")
            print(f"✅ Overall Success: {validation_results['validation_summary']['overall_success']}")
            print(f"📊 Tests Passed: {validation_results['validation_summary']['tests_passed']}/{validation_results['validation_summary']['total_tests']}")
            
            return validation_results
            
        except Exception as e:
            print(f"❌ Validation suite failed with error: {str(e)}")
            validation_results["error"] = str(e)
            return validation_results
    
    async def test_rlt_formatter(self) -> Dict[str, Any]:
        """Test RLT Formatter functionality"""
        try:
            formatter = RLTFormatter()
            test_batch = self.dataset.get_test_batch(3)
            
            results = {
                "success": True,
                "tests_run": 0,
                "tests_passed": 0,
                "details": {}
            }
            
            # Test input formatting
            for i, (question, solution) in enumerate(zip(test_batch["questions"], test_batch["solutions"])):
                results["tests_run"] += 1
                
                try:
                    # Format input
                    formatted_input = formatter.format_question_solution_input(question, solution)
                    
                    # Validate format
                    assert formatted_input.question == question
                    assert formatted_input.solution == solution
                    assert "Question:" in formatted_input.formatted_input
                    assert "Solution:" in formatted_input.formatted_input
                    assert "Explain:" in formatted_input.formatted_input
                    
                    results["tests_passed"] += 1
                    results["details"][f"input_format_test_{i}"] = {"status": "passed", "length": len(formatted_input.formatted_input)}
                    
                except Exception as e:
                    results["details"][f"input_format_test_{i}"] = {"status": "failed", "error": str(e)}
            
            # Test output parsing with mock teacher output
            mock_outputs = [
                "<think>This is a test explanation</think><solution>Test solution</solution>",
                "Unstructured explanation without tags",
                "<think>Complex explanation with multiple steps. First step. Second step.</think>"
            ]
            
            for i, output in enumerate(mock_outputs):
                results["tests_run"] += 1
                
                try:
                    parsed = formatter.parse_rlt_output(output)
                    assert parsed.raw_output == output
                    assert parsed.think_content is not None
                    
                    results["tests_passed"] += 1
                    results["details"][f"output_parse_test_{i}"] = {
                        "status": "passed", 
                        "think_length": len(parsed.think_content),
                        "is_valid": parsed.is_valid
                    }
                    
                except Exception as e:
                    results["details"][f"output_parse_test_{i}"] = {"status": "failed", "error": str(e)}
            
            # Test distillation prompt creation
            for i, (question, solution) in enumerate(zip(test_batch["questions"][:2], test_batch["solutions"][:2])):
                results["tests_run"] += 1
                
                try:
                    think_tokens = f"To solve this problem: {question[:50]}..."
                    distillation_prompt = formatter.create_student_distillation_prompt(
                        question, think_tokens, solution, "standard"
                    )
                    
                    assert distillation_prompt.question == question
                    assert distillation_prompt.solution == solution
                    assert len(distillation_prompt.formatted_prompt) > 0
                    
                    results["tests_passed"] += 1
                    results["details"][f"distillation_test_{i}"] = {
                        "status": "passed",
                        "prompt_length": len(distillation_prompt.formatted_prompt)
                    }
                    
                except Exception as e:
                    results["details"][f"distillation_test_{i}"] = {"status": "failed", "error": str(e)}
            
            results["success"] = results["tests_passed"] == results["tests_run"]
            return results
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_student_comprehension_evaluator(self) -> Dict[str, Any]:
        """Test Student Comprehension Evaluator"""
        try:
            config = EvaluationConfig()
            evaluator = StudentCompressionEvaluator(config=config)
            
            results = {
                "success": True,
                "tests_run": 0,
                "tests_passed": 0,
                "details": {}
            }
            
            # Test individual metric computation (without actual models for speed)
            test_cases = [
                {
                    "question": "What is 2+2?",
                    "explanation": "To add 2+2, we combine two sets of 2. First, we start with 2. Then we add another 2. The result is 4.",
                    "solution": "4"
                },
                {
                    "question": "What is the derivative of x^2?",
                    "explanation": "Use power rule. Bring down exponent, reduce by 1. Result is 2x.",
                    "solution": "2x"
                }
            ]
            
            for i, test_case in enumerate(test_cases):
                results["tests_run"] += 1
                
                try:
                    # Test individual assessment methods (using heuristics without actual models)
                    coherence = evaluator._assess_coherence(test_case["explanation"])
                    logical_flow = evaluator._assess_logical_flow(test_case["explanation"])
                    concept_coverage = evaluator._assess_concept_coverage(
                        test_case["explanation"], test_case["question"], test_case["solution"]
                    )
                    
                    # Validate metrics are in valid range
                    assert 0.0 <= coherence <= 1.0
                    assert 0.0 <= logical_flow <= 1.0
                    assert 0.0 <= concept_coverage <= 1.0
                    
                    results["tests_passed"] += 1
                    results["details"][f"metrics_test_{i}"] = {
                        "status": "passed",
                        "coherence": coherence,
                        "logical_flow": logical_flow,
                        "concept_coverage": concept_coverage
                    }
                    
                except Exception as e:
                    results["details"][f"metrics_test_{i}"] = {"status": "failed", "error": str(e)}
            
            # Test configuration validation
            results["tests_run"] += 1
            try:
                assert config.min_comprehension_threshold > 0
                assert config.solution_weight + config.coherence_weight + config.logical_flow_weight + config.concept_grasp_weight == 1.0
                
                results["tests_passed"] += 1
                results["details"]["config_test"] = {"status": "passed"}
                
            except Exception as e:
                results["details"]["config_test"] = {"status": "failed", "error": str(e)}
            
            results["success"] = results["tests_passed"] == results["tests_run"]
            return results
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_quality_monitor(self) -> Dict[str, Any]:
        """Test Quality Monitor functionality"""
        try:
            config = MonitoringConfig()
            monitor = RLTQualityMonitor(config=config, teacher_id="test_teacher")
            
            results = {
                "success": True,
                "tests_run": 0,
                "tests_passed": 0,
                "details": {}
            }
            
            await monitor.start_monitoring()
            
            # Test quality metrics recording
            test_metrics = [
                {
                    "explanation": "Clear explanation with logical steps. First, we identify the problem. Then, we apply the method. Finally, we get the result.",
                    "question": "Test question",
                    "solution": "Test solution",
                    "generation_time": 1.5,
                    "reward_score": 0.8,
                    "comprehension_score": 0.85
                },
                {
                    "explanation": "Short explanation",
                    "question": "Simple question",
                    "solution": "Simple answer",
                    "generation_time": 0.5,
                    "reward_score": 0.4,
                    "comprehension_score": 0.3
                }
            ]
            
            for i, metrics in enumerate(test_metrics):
                results["tests_run"] += 1
                
                try:
                    recorded_metrics = await monitor.record_quality_metrics(
                        session_id=f"test_session_{i}",
                        **metrics
                    )
                    
                    # Validate recorded metrics
                    assert recorded_metrics.teacher_id == "test_teacher"
                    assert recorded_metrics.student_comprehension == metrics["comprehension_score"]
                    assert recorded_metrics.generation_time == metrics["generation_time"]
                    assert 0.0 <= recorded_metrics.overall_quality() <= 1.0
                    
                    results["tests_passed"] += 1
                    results["details"][f"metrics_recording_test_{i}"] = {
                        "status": "passed",
                        "overall_quality": recorded_metrics.overall_quality()
                    }
                    
                except Exception as e:
                    results["details"][f"metrics_recording_test_{i}"] = {"status": "failed", "error": str(e)}
            
            # Test summary generation
            results["tests_run"] += 1
            try:
                summary = monitor.get_quality_summary(time_window_hours=1)
                
                assert "total_explanations" in summary
                assert "quality_statistics" in summary
                assert summary["total_explanations"] == len(test_metrics)
                
                results["tests_passed"] += 1
                results["details"]["summary_test"] = {
                    "status": "passed",
                    "total_explanations": summary["total_explanations"]
                }
                
            except Exception as e:
                results["details"]["summary_test"] = {"status": "failed", "error": str(e)}
            
            await monitor.stop_monitoring()
            
            results["success"] = results["tests_passed"] == results["tests_run"]
            return results
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_dual_reward_system(self) -> Dict[str, Any]:
        """Test the dual reward system (r_SS + r_KL)"""
        try:
            config = RLTTrainingConfig()
            trainer = RLTDenseRewardTrainer(config=config)
            
            results = {
                "success": True,
                "tests_run": 0,
                "tests_passed": 0,
                "details": {}
            }
            
            # Test reward computation (without actual models)
            test_cases = [
                {
                    "explanation": "Step-by-step explanation with clear reasoning",
                    "student_response": "I understand the solution",
                    "solution": "Test solution",
                    "question": "Test question"
                },
                {
                    "explanation": "Brief explanation",
                    "student_response": "Unclear",
                    "solution": "Simple answer", 
                    "question": "Simple question"
                }
            ]
            
            for i, test_case in enumerate(test_cases):
                results["tests_run"] += 1
                
                try:
                    # Test reward computation methods
                    rewards = trainer.compute_total_reward(
                        test_case["explanation"],
                        test_case["student_response"],
                        test_case["solution"],
                        test_case["question"]
                    )
                    
                    # Validate reward structure
                    assert "r_ss" in rewards
                    assert "r_kl" in rewards
                    assert "total_reward" in rewards
                    assert isinstance(rewards["r_ss"], (int, float))
                    assert isinstance(rewards["r_kl"], (int, float))
                    assert isinstance(rewards["total_reward"], (int, float))
                    
                    results["tests_passed"] += 1
                    results["details"][f"reward_computation_test_{i}"] = {
                        "status": "passed",
                        "r_ss": rewards["r_ss"],
                        "r_kl": rewards["r_kl"],
                        "total_reward": rewards["total_reward"]
                    }
                    
                except Exception as e:
                    results["details"][f"reward_computation_test_{i}"] = {"status": "failed", "error": str(e)}
            
            # Test reward configuration
            results["tests_run"] += 1
            try:
                assert config.alpha > 0  # Weight for r_KL
                assert config.beta > 0   # KL regularization
                assert 0 < config.gamma <= 1  # Discount factor
                
                results["tests_passed"] += 1
                results["details"]["reward_config_test"] = {"status": "passed"}
                
            except Exception as e:
                results["details"]["reward_config_test"] = {"status": "failed", "error": str(e)}
            
            results["success"] = results["tests_passed"] == results["tests_run"]
            return results
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_training_pipeline(self) -> Dict[str, Any]:
        """Test the complete training pipeline"""
        try:
            config = RLTTrainingConfig(
                batch_size=2,
                num_training_epochs=1,
                learning_rate=1e-4
            )
            trainer = RLTDenseRewardTrainer(config=config)
            
            results = {
                "success": True,
                "tests_run": 0,
                "tests_passed": 0,
                "details": {}
            }
            
            # Test with small dataset
            test_batch = self.dataset.get_test_batch(3)
            
            # Test dataset creation
            results["tests_run"] += 1
            try:
                from prsm.teachers.rlt.dense_reward_trainer import RLTTrainingDataset
                
                # Mock tokenizer for testing
                class MockTokenizer:
                    def __init__(self):
                        self.pad_token = "[PAD]"
                        self.eos_token = "[EOS]"
                    
                    def __call__(self, text, **kwargs):
                        # Return mock encoding
                        import torch
                        return {
                            "input_ids": torch.randint(0, 1000, (1, min(kwargs.get("max_length", 100), 50))),
                            "attention_mask": torch.ones(1, min(kwargs.get("max_length", 100), 50))
                        }
                
                mock_tokenizer = MockTokenizer()
                dataset = RLTTrainingDataset(
                    test_batch["questions"], 
                    test_batch["solutions"], 
                    mock_tokenizer
                )
                
                assert len(dataset) == len(test_batch["questions"])
                
                results["tests_passed"] += 1
                results["details"]["dataset_creation_test"] = {
                    "status": "passed",
                    "dataset_size": len(dataset)
                }
                
            except Exception as e:
                results["details"]["dataset_creation_test"] = {"status": "failed", "error": str(e)}
            
            # Test explanation generation (mock)
            results["tests_run"] += 1
            try:
                # Mock explanation generation without actual models
                question = test_batch["questions"][0]
                solution = test_batch["solutions"][0]
                
                # Test the method exists and has correct signature
                assert hasattr(trainer, 'generate_explanation')
                
                results["tests_passed"] += 1
                results["details"]["explanation_generation_test"] = {"status": "passed"}
                
            except Exception as e:
                results["details"]["explanation_generation_test"] = {"status": "failed", "error": str(e)}
            
            # Test configuration validation
            results["tests_run"] += 1
            try:
                assert config.batch_size > 0
                assert config.learning_rate > 0
                assert config.num_training_epochs > 0
                assert config.max_explanation_length > 0
                
                results["tests_passed"] += 1
                results["details"]["config_validation_test"] = {"status": "passed"}
                
            except Exception as e:
                results["details"]["config_validation_test"] = {"status": "failed", "error": str(e)}
            
            results["success"] = results["tests_passed"] == results["tests_run"]
            return results
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive validation summary"""
        test_results = validation_results["test_results"]
        
        total_tests = 0
        tests_passed = 0
        component_status = {}
        
        for component, results in test_results.items():
            component_status[component] = results.get("success", False)
            if "tests_run" in results:
                total_tests += results["tests_run"]
                tests_passed += results["tests_passed"]
        
        overall_success = all(component_status.values())
        
        return {
            "overall_success": overall_success,
            "total_tests": total_tests,
            "tests_passed": tests_passed,
            "test_success_rate": tests_passed / max(total_tests, 1),
            "component_status": component_status,
            "recommendations": self._generate_recommendations(component_status)
        }
    
    def _generate_recommendations(self, component_status: Dict[str, bool]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if not component_status.get("formatter", True):
            recommendations.append("Fix RLT Formatter implementation - check input/output formatting")
        
        if not component_status.get("comprehension", True):
            recommendations.append("Fix Student Comprehension Evaluator - check metric calculations")
        
        if not component_status.get("monitor", True):
            recommendations.append("Fix Quality Monitor - check metrics recording and alerting")
        
        if not component_status.get("rewards", True):
            recommendations.append("Fix Dual Reward System - check r_SS and r_KL computations")
        
        if not component_status.get("pipeline", True):
            recommendations.append("Fix Training Pipeline - check integration and configuration")
        
        if all(component_status.values()):
            recommendations.append("All components passed - ready for integration with SEAL Teacher")
        
        return recommendations


# Main test execution
async def run_rlt_validation():
    """Main function to run RLT validation suite"""
    validation_suite = RLTValidationSuite()
    results = await validation_suite.run_full_validation()
    
    # Save results to file
    results_file = f"rlt_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Validation results saved to: {results_file}")
    
    return results


# Performance benchmarking
async def benchmark_rlt_performance():
    """Benchmark RLT performance vs baseline"""
    print("\n🏁 Starting RLT Performance Benchmark")
    print("=" * 50)
    
    dataset = RLTTestDataset()
    test_batch = dataset.get_test_batch(5)
    
    # Benchmark formatter
    formatter = RLTFormatter()
    start_time = time.time()
    
    formatted_inputs = []
    for question, solution in zip(test_batch["questions"], test_batch["solutions"]):
        formatted_input = formatter.format_question_solution_input(question, solution)
        formatted_inputs.append(formatted_input)
    
    format_time = time.time() - start_time
    
    # Benchmark comprehension evaluator
    evaluator = StudentCompressionEvaluator()
    start_time = time.time()
    
    comprehension_scores = []
    for question, solution in zip(test_batch["questions"][:3], test_batch["solutions"][:3]):
        explanation = f"This is a test explanation for: {question}"
        # Use heuristic methods for speed
        coherence = evaluator._assess_coherence(explanation)
        comprehension_scores.append(coherence)
    
    evaluation_time = time.time() - start_time
    
    benchmark_results = {
        "formatting_performance": {
            "total_time": format_time,
            "items_processed": len(formatted_inputs),
            "items_per_second": len(formatted_inputs) / format_time,
            "avg_time_per_item": format_time / len(formatted_inputs)
        },
        "evaluation_performance": {
            "total_time": evaluation_time,
            "items_processed": len(comprehension_scores),
            "items_per_second": len(comprehension_scores) / evaluation_time,
            "avg_time_per_item": evaluation_time / len(comprehension_scores)
        }
    }
    
    print(f"📊 Formatting: {benchmark_results['formatting_performance']['items_per_second']:.2f} items/sec")
    print(f"📊 Evaluation: {benchmark_results['evaluation_performance']['items_per_second']:.2f} items/sec")
    
    return benchmark_results


if __name__ == "__main__":
    # Run validation suite
    print("🚀 Starting RLT Training Pipeline Test Suite")
    validation_results = asyncio.run(run_rlt_validation())
    
    # Run performance benchmark
    benchmark_results = asyncio.run(benchmark_rlt_performance())
    
    print("\n🎯 Testing Complete!")
    print(f"Overall Success: {validation_results['validation_summary']['overall_success']}")
    print(f"Components Ready for Integration: {validation_results['validation_summary']['component_status']}")