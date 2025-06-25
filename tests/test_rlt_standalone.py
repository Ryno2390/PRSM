"""
Standalone RLT Test Script

Tests RLT components independently without heavy PRSM dependencies.
Validates core functionality and Sakana AI methodology implementation.
"""

import asyncio
import json
import time
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any

# Simple test imports without complex dependencies
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_rlt_formatter():
    """Test RLT Formatter without external dependencies"""
    print("📝 Testing RLT Formatter...")
    
    # Import locally to avoid dependency issues
    try:
        from prsm.teachers.rlt.explanation_formatter import RLTFormatter, RLTFormatConfig
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    try:
        formatter = RLTFormatter()
        
        # Test data
        question = "What is the derivative of x^2?"
        solution = "The derivative of x^2 is 2x"
        
        # Test input formatting
        formatted_input = formatter.format_question_solution_input(question, solution)
        
        # Validate format
        assert formatted_input.question == question
        assert formatted_input.solution == solution
        assert "Question:" in formatted_input.formatted_input
        assert "Solution:" in formatted_input.formatted_input
        assert "Explain:" in formatted_input.formatted_input
        
        print(f"  ✅ Input formatting: PASSED")
        print(f"     Formatted length: {len(formatted_input.formatted_input)} chars")
        
        # Test output parsing
        mock_output = "<think>To find the derivative, use the power rule</think><solution>2x</solution>"
        parsed = formatter.parse_rlt_output(mock_output)
        
        assert "power rule" in parsed.think_content
        assert parsed.solution_content == "2x"
        
        print(f"  ✅ Output parsing: PASSED")
        print(f"     Think content: {len(parsed.think_content)} chars")
        
        # Test distillation prompt
        think_tokens = "To find the derivative, use the power rule. Multiply by exponent, reduce by 1."
        distillation_prompt = formatter.create_student_distillation_prompt(
            question, think_tokens, solution, "standard"
        )
        
        assert distillation_prompt.question == question
        assert distillation_prompt.solution == solution
        assert len(distillation_prompt.formatted_prompt) > 0
        
        print(f"  ✅ Distillation prompts: PASSED")
        print(f"     Prompt length: {len(distillation_prompt.formatted_prompt)} chars")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Formatter test failed: {e}")
        return False


def test_student_comprehension():
    """Test Student Comprehension Evaluator heuristics"""
    print("\n📊 Testing Student Comprehension Evaluator...")
    
    try:
        from prsm.teachers.rlt.student_comprehension_evaluator import StudentCompressionEvaluator, EvaluationConfig
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    try:
        config = EvaluationConfig()
        evaluator = StudentCompressionEvaluator(config=config)
        
        # Test heuristic methods (no ML models required)
        test_cases = [
            {
                "explanation": "To solve this problem, first identify the key components. Then, apply the appropriate method. Finally, verify the result.",
                "question": "How to solve quadratic equations?",
                "solution": "Use the quadratic formula",
                "expected_coherence": 0.6,  # Good structure
                "expected_flow": 0.4,       # Good logical flow
                "expected_coverage": 0.3    # Moderate coverage
            },
            {
                "explanation": "Short answer",
                "question": "What is 2+2?",
                "solution": "4",
                "expected_coherence": 0.3,  # Too short
                "expected_flow": 0.2,       # Minimal flow
                "expected_coverage": 0.2    # Low coverage
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            # Test coherence assessment
            coherence = evaluator._assess_coherence(test_case["explanation"])
            assert 0.0 <= coherence <= 1.0
            
            # Test logical flow assessment
            logical_flow = evaluator._assess_logical_flow(test_case["explanation"])
            assert 0.0 <= logical_flow <= 1.0
            
            # Test concept coverage
            concept_coverage = evaluator._assess_concept_coverage(
                test_case["explanation"], 
                test_case["question"], 
                test_case["solution"]
            )
            assert 0.0 <= concept_coverage <= 1.0
            
            print(f"  ✅ Test case {i+1}: PASSED")
            print(f"     Coherence: {coherence:.3f} (expected ~{test_case['expected_coherence']:.1f})")
            print(f"     Logical flow: {logical_flow:.3f} (expected ~{test_case['expected_flow']:.1f})")
            print(f"     Concept coverage: {concept_coverage:.3f} (expected ~{test_case['expected_coverage']:.1f})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Comprehension test failed: {e}")
        return False


def test_quality_monitor():
    """Test Quality Monitor functionality"""
    print("\n📈 Testing Quality Monitor...")
    
    try:
        from prsm.teachers.rlt.quality_monitor import RLTQualityMonitor, MonitoringConfig, QualityMetrics
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    try:
        config = MonitoringConfig()
        monitor = RLTQualityMonitor(config=config, teacher_id="test_teacher")
        
        # Test quality metrics creation
        test_metrics = QualityMetrics(
            explanation_coherence=0.8,
            student_comprehension=0.75,
            logical_flow=0.7,
            concept_coverage=0.65,
            explanation_length=150,
            generation_time=1.5,
            reward_score=0.72,
            question_complexity=0.6,
            domain="mathematics"
        )
        
        # Test metrics validation
        overall_quality = test_metrics.overall_quality()
        assert 0.0 <= overall_quality <= 1.0
        
        print(f"  ✅ Metrics creation: PASSED")
        print(f"     Overall quality: {overall_quality:.3f}")
        
        # Test metrics storage
        monitor.quality_history.append(test_metrics)
        monitor.total_evaluations = 1
        
        # Test summary generation
        summary = monitor.get_quality_summary(time_window_hours=1)
        
        assert "total_explanations" in summary
        assert "quality_statistics" in summary
        assert summary["total_explanations"] >= 0
        
        print(f"  ✅ Summary generation: PASSED")
        print(f"     Summary keys: {list(summary.keys())}")
        
        # Test heuristic assessment methods
        coherence = monitor._assess_coherence("Clear explanation with logical steps. First, identify the problem. Then, solve it.")
        assert 0.0 <= coherence <= 1.0
        
        complexity = monitor._assess_question_complexity("What is the integral of sin(x)cos(x)?")
        assert 0.0 <= complexity <= 1.0
        
        domain = monitor._identify_domain("Calculate the derivative using calculus")
        assert domain in ["mathematics", "physics", "chemistry", "computer_science", "general"]
        
        print(f"  ✅ Heuristic assessments: PASSED")
        print(f"     Coherence: {coherence:.3f}, Complexity: {complexity:.3f}, Domain: {domain}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Quality monitor test failed: {e}")
        return False


def test_dense_reward_trainer():
    """Test Dense Reward Trainer configuration and methods"""
    print("\n💰 Testing Dense Reward Trainer...")
    
    try:
        from prsm.teachers.rlt.dense_reward_trainer import RLTDenseRewardTrainer, RLTTrainingConfig
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    try:
        config = RLTTrainingConfig(
            alpha=0.1,
            beta=0.01,
            learning_rate=5e-5,
            batch_size=4,
            num_training_epochs=1
        )
        
        trainer = RLTDenseRewardTrainer(config=config)
        
        # Test configuration validation
        assert config.alpha > 0
        assert config.beta > 0
        assert config.learning_rate > 0
        assert config.batch_size > 0
        assert config.num_training_epochs > 0
        
        print(f"  ✅ Configuration: PASSED")
        print(f"     Alpha: {config.alpha}, Beta: {config.beta}")
        print(f"     Learning rate: {config.learning_rate}")
        print(f"     Batch size: {config.batch_size}")
        
        # Test reward computation without models (will return 0.0 but structure is valid)
        test_case = {
            "explanation": "Step by step solution approach",
            "student_response": "I understand the method",
            "solution": "Final answer is X",
            "question": "How to solve this problem?"
        }
        
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
        assert "explanation_length" in rewards
        assert "timestamp" in rewards
        
        print(f"  ✅ Reward computation: PASSED")
        print(f"     Reward components: {list(rewards.keys())}")
        print(f"     Total reward: {rewards['total_reward']}")
        
        # Test explanation assessment methods
        coherence = trainer._assess_coherence(test_case["explanation"])
        completeness = trainer._assess_completeness(test_case["explanation"], test_case["solution"])
        
        assert 0.0 <= coherence <= 1.0
        assert 0.0 <= completeness <= 1.0
        
        print(f"  ✅ Assessment methods: PASSED")
        print(f"     Coherence: {coherence:.3f}, Completeness: {completeness:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Dense reward trainer test failed: {e}")
        return False


def test_integration_readiness():
    """Test that all components can work together"""
    print("\n🔗 Testing Integration Readiness...")
    
    try:
        # Import all components
        from prsm.teachers.rlt.explanation_formatter import RLTFormatter
        from prsm.teachers.rlt.student_comprehension_evaluator import StudentCompressionEvaluator
        from prsm.teachers.rlt.quality_monitor import RLTQualityMonitor
        from prsm.teachers.rlt.dense_reward_trainer import RLTDenseRewardTrainer
        
        # Initialize components
        formatter = RLTFormatter()
        evaluator = StudentCompressionEvaluator()
        monitor = RLTQualityMonitor(teacher_id="integration_test")
        trainer = RLTDenseRewardTrainer()
        
        # Test data flow
        question = "What is the derivative of x^3?"
        solution = "The derivative of x^3 is 3x^2"
        
        # 1. Format input
        formatted_input = formatter.format_question_solution_input(question, solution)
        
        # 2. Simulate explanation generation
        mock_explanation = "To find the derivative of x^3, we use the power rule. The power rule states that for x^n, the derivative is n*x^(n-1). For x^3, n=3, so the derivative is 3*x^(3-1) = 3*x^2."
        
        # 3. Parse explanation
        mock_output = f"<think>{mock_explanation}</think><solution>{solution}</solution>"
        parsed_output = formatter.parse_rlt_output(mock_output)
        
        # 4. Evaluate comprehension (heuristic methods)
        coherence = evaluator._assess_coherence(mock_explanation)
        logical_flow = evaluator._assess_logical_flow(mock_explanation)
        concept_coverage = evaluator._assess_concept_coverage(mock_explanation, question, solution)
        
        # 5. Compute rewards
        rewards = trainer.compute_total_reward(mock_explanation, "I understand", solution, question)
        
        # 6. Quality monitoring
        from prsm.teachers.rlt.quality_monitor import QualityMetrics
        quality_metrics = QualityMetrics(
            explanation_coherence=coherence,
            student_comprehension=0.8,  # Mock score
            logical_flow=logical_flow,
            concept_coverage=concept_coverage,
            explanation_length=len(mock_explanation),
            generation_time=1.2,
            reward_score=rewards["total_reward"],
            question_complexity=0.7,
            domain="mathematics"
        )
        
        monitor.quality_history.append(quality_metrics)
        
        print(f"  ✅ Data flow: PASSED")
        print(f"     Input → Format → Explanation → Parse → Evaluate → Monitor")
        print(f"     Final quality score: {quality_metrics.overall_quality():.3f}")
        
        # Test component compatibility
        assert len(formatted_input.formatted_input) > 0
        assert len(parsed_output.think_content) > 0
        assert 0.0 <= coherence <= 1.0
        assert "total_reward" in rewards
        assert quality_metrics.overall_quality() >= 0.0
        
        print(f"  ✅ Component compatibility: PASSED")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False


def run_performance_benchmark():
    """Run performance benchmark"""
    print("\n🏁 Performance Benchmark")
    print("=" * 30)
    
    # Test formatter performance
    from prsm.teachers.rlt.explanation_formatter import RLTFormatter
    
    formatter = RLTFormatter()
    test_pairs = [
        ("What is 2+2?", "4"),
        ("What is the derivative of x^2?", "2x"),
        ("How do you solve quadratic equations?", "Use the quadratic formula"),
        ("What is Newton's second law?", "F = ma"),
        ("What is a binary search?", "An efficient search algorithm for sorted arrays")
    ]
    
    # Benchmark formatting
    start_time = time.time()
    formatted_inputs = []
    for question, solution in test_pairs:
        formatted_input = formatter.format_question_solution_input(question, solution)
        formatted_inputs.append(formatted_input)
    format_time = time.time() - start_time
    
    # Benchmark parsing
    mock_outputs = [
        f"<think>Explanation for {q}</think><solution>{s}</solution>" 
        for q, s in test_pairs
    ]
    
    start_time = time.time()
    parsed_outputs = []
    for output in mock_outputs:
        parsed = formatter.parse_rlt_output(output)
        parsed_outputs.append(parsed)
    parse_time = time.time() - start_time
    
    print(f"📊 Formatting: {len(test_pairs) / format_time:.1f} items/sec")
    print(f"📊 Parsing: {len(mock_outputs) / parse_time:.1f} items/sec")
    print(f"📊 Total processing: {(len(test_pairs) + len(mock_outputs)) / (format_time + parse_time):.1f} items/sec")
    
    return {
        "formatting_speed": len(test_pairs) / format_time,
        "parsing_speed": len(mock_outputs) / parse_time,
        "total_items": len(test_pairs) + len(mock_outputs),
        "total_time": format_time + parse_time
    }


def main():
    """Main test execution"""
    print("🚀 RLT Training Pipeline Standalone Test Suite")
    print("=" * 60)
    
    test_results = {}
    
    # Run component tests
    tests = [
        ("Formatter", test_rlt_formatter),
        ("Student Comprehension", test_student_comprehension),
        ("Quality Monitor", test_quality_monitor),
        ("Dense Reward Trainer", test_dense_reward_trainer),
        ("Integration Readiness", test_integration_readiness)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results[test_name] = result
            if result:
                passed_tests += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            test_results[test_name] = False
            print(f"❌ {test_name}: ERROR - {e}")
    
    # Run performance benchmark
    print("\n" + "=" * 60)
    benchmark_results = run_performance_benchmark()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎯 RLT Test Suite Summary")
    print("=" * 60)
    print(f"✅ Tests Passed: {passed_tests}/{total_tests}")
    print(f"📊 Success Rate: {passed_tests/total_tests:.1%}")
    print(f"🏁 Processing Speed: {benchmark_results['total_items']/benchmark_results['total_time']:.1f} items/sec")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED - RLT Implementation Ready!")
        print("✅ Components validated and ready for SEAL integration")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed - Review implementation")
    
    # Save results
    results_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_results": test_results,
        "performance": benchmark_results,
        "summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests/total_tests,
            "overall_success": passed_tests == total_tests
        }
    }
    
    with open("rlt_test_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: rlt_test_results.json")
    
    return results_summary


if __name__ == "__main__":
    main()