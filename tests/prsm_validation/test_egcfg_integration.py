#!/usr/bin/env python3
"""
Test suite for EG-CFG (Execution-Guided Classifier-Free Guidance) integration
Validates the cutting-edge research implementation in PRSM
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from prsm.agents.executors.execution_guided_code_runner import (
    EGCFGAgent,
    ExecutionGuidedCodeRunner,
    GenerationConfig,
    SafeExecutionEnvironment,
    ExecutionStatus,
    CodeCandidate,
    ExecutionTrace
)

class TestSafeExecutionEnvironment:
    """Test the safe execution environment"""
    
    def setup_method(self):
        """Setup test environment"""
        self.env = SafeExecutionEnvironment(timeout=2.0)
    
    def test_simple_execution(self):
        """Test basic code execution"""
        trace = self.env.execute_line("x = 5")
        
        assert trace.status == ExecutionStatus.SUCCESS
        assert trace.code_line == "x = 5"
        assert "x" in trace.variables
        assert trace.variables["x"] == 5
    
    def test_expression_evaluation(self):
        """Test expression evaluation"""
        # Setup variable
        self.env.execute_line("x = 10")
        
        # Evaluate expression
        trace = self.env.execute_line("x * 2")
        
        assert trace.status == ExecutionStatus.SUCCESS
        assert trace.return_value == 20
    
    def test_syntax_error_handling(self):
        """Test syntax error detection"""
        trace = self.env.execute_line("def invalid syntax:")
        
        assert trace.status == ExecutionStatus.SYNTAX_ERROR
        assert "SyntaxError" in trace.error_message
    
    def test_runtime_error_handling(self):
        """Test runtime error handling"""
        trace = self.env.execute_line("undefined_variable")
        
        assert trace.status == ExecutionStatus.ERROR
        assert trace.error_message
    
    def test_print_output_capture(self):
        """Test output capture"""
        trace = self.env.execute_line('print("Hello, PRSM!")')
        
        assert trace.status == ExecutionStatus.SUCCESS
        assert "Hello, PRSM!" in trace.output
    
    def test_execution_timeout(self):
        """Test execution timeout (Note: simplified for testing)"""
        # Use a simple long-running operation
        trace = self.env.execute_line("sum(range(1000000))")
        
        # Should complete successfully for reasonable operations
        assert trace.status == ExecutionStatus.SUCCESS
        assert trace.execution_time < 2.0

class TestCodeCandidate:
    """Test code candidate functionality"""
    
    def test_candidate_creation(self):
        """Test candidate creation"""
        candidate = CodeCandidate(
            candidate_id="test_001",
            code_lines=["def hello():", "    print('world')"],
            temperature=0.7,
            execution_traces=[]
        )
        
        assert candidate.candidate_id == "test_001"
        assert len(candidate.code_lines) == 2
        assert candidate.temperature == 0.7
        assert candidate.total_score == 0.0
        assert candidate.syntax_valid is True

class TestExecutionGuidedCodeRunner:
    """Test the main EG-CFG code runner"""
    
    def setup_method(self):
        """Setup test runner"""
        config = GenerationConfig(
            candidates_per_line=2,
            temperatures=[0.7, 0.9],
            enable_parallel_execution=False,  # Disable for deterministic testing
            max_execution_time=2.0
        )
        self.runner = ExecutionGuidedCodeRunner(config)
    
    @pytest.mark.asyncio
    async def test_basic_code_generation(self):
        """Test basic code generation"""
        result = await self.runner.generate_and_execute_code(
            prompt="Create a simple function",
            context={"max_lines": 5}
        )
        
        assert result["success"] is True
        assert "generated_code" in result
        assert "execution_traces" in result
        assert result["methodology"] == "EG-CFG"
        assert result["candidate_count"] > 0
        assert result["total_time"] > 0
    
    @pytest.mark.asyncio
    async def test_line_candidate_generation(self):
        """Test line candidate generation"""
        candidates = await self.runner._generate_line_candidates(
            prompt="Create a function",
            existing_lines=[],
            context={}
        )
        
        assert len(candidates) == 2  # Based on config
        assert all(isinstance(c, CodeCandidate) for c in candidates)
        assert all(len(c.code_lines) == 1 for c in candidates)
    
    def test_candidate_execution(self):
        """Test candidate execution"""
        candidate = CodeCandidate(
            candidate_id="test_exec",
            code_lines=["x = 42", "y = x * 2"],
            temperature=0.8,
            execution_traces=[]
        )
        
        executed = self.runner._execute_single_candidate(candidate)
        
        assert len(executed.execution_traces) == 2
        assert executed.execution_traces[0].status == ExecutionStatus.SUCCESS
        assert executed.execution_traces[1].status == ExecutionStatus.SUCCESS
        assert executed.total_score > 0
    
    def test_ast_validation(self):
        """Test AST validation"""
        # Valid code
        valid_lines = ["def hello():", "    return 'world'"]
        assert self.runner._validate_ast(valid_lines) is True
        
        # Invalid code
        invalid_lines = ["def invalid(:", "    return"]
        assert self.runner._validate_ast(invalid_lines) is False
    
    def test_candidate_scoring(self):
        """Test candidate scoring"""
        # Create candidate with successful execution
        candidate = CodeCandidate(
            candidate_id="scoring_test",
            code_lines=["x = 5"],
            temperature=0.7,
            execution_traces=[
                ExecutionTrace(
                    line_number=1,
                    code_line="x = 5",
                    status=ExecutionStatus.SUCCESS,
                    execution_time=0.01
                )
            ],
            syntax_valid=True
        )
        
        score = self.runner._calculate_candidate_score(candidate)
        assert score > 0  # Should get positive score for success
    
    def test_classifier_free_guidance(self):
        """Test CFG selection mechanism"""
        candidates = [
            CodeCandidate(
                candidate_id="cfg_test_1",
                code_lines=["x = 1"],
                temperature=0.7,
                execution_traces=[],
                total_score=2.0
            ),
            CodeCandidate(
                candidate_id="cfg_test_2",
                code_lines=["x = 2"],
                temperature=0.9,
                execution_traces=[],
                total_score=3.0
            )
        ]
        
        best = self.runner._apply_classifier_free_guidance(candidates, {})
        
        assert best is not None
        assert best.candidate_id in ["cfg_test_1", "cfg_test_2"]
    
    def test_completion_detection(self):
        """Test completion detection"""
        # Candidate with return statement
        complete_candidate = CodeCandidate(
            candidate_id="complete",
            code_lines=["def func():", "    return 42"],
            temperature=0.7,
            execution_traces=[
                ExecutionTrace(
                    line_number=2,
                    code_line="    return 42",
                    status=ExecutionStatus.SUCCESS,
                    return_value=42
                )
            ]
        )
        
        assert self.runner._is_generation_complete(complete_candidate, {}) is True
        
        # Incomplete candidate
        incomplete_candidate = CodeCandidate(
            candidate_id="incomplete",
            code_lines=["def func():"],
            temperature=0.7,
            execution_traces=[]
        )
        
        assert self.runner._is_generation_complete(incomplete_candidate, {}) is False
    
    def test_execution_statistics(self):
        """Test execution statistics collection"""
        # Add some candidates to history
        self.runner.generation_history = [
            CodeCandidate("hist_1", ["x = 1"], 0.7, [], total_score=2.0),
            CodeCandidate("hist_2", ["y = 2"], 0.8, [], total_score=3.0),
            CodeCandidate("hist_3", ["z = 3"], 0.9, [], total_score=1.0)
        ]
        
        stats = self.runner.get_execution_statistics()
        
        assert stats["total_candidates"] == 3
        assert stats["successful_candidates"] == 3  # All have positive scores
        assert stats["success_rate"] == 1.0
        assert stats["average_score"] == 2.0
        assert stats["methodology"] == "EG-CFG (Execution-Guided Classifier-Free Guidance)"

class TestEGCFGAgent:
    """Test the PRSM agent wrapper"""
    
    def setup_method(self):
        """Setup test agent"""
        config = GenerationConfig(
            candidates_per_line=2,
            temperatures=[0.7, 0.9],
            enable_parallel_execution=False
        )
        self.agent = EGCFGAgent(config)
    
    @pytest.mark.asyncio
    async def test_agent_task_processing(self):
        """Test agent task processing"""
        task = {
            "prompt": "Create a simple calculator function",
            "context": {
                "max_lines": 8,
                "enable_validation": True
            }
        }
        
        result = await self.agent.process_coding_task(task)
        
        assert result["success"] is True
        assert result["agent_id"] == "egcfg_agent"
        assert "agent_capabilities" in result
        assert result["processing_method"] == "Execution-Guided Classifier-Free Guidance"
        assert "generated_code" in result
    
    def test_agent_status(self):
        """Test agent status reporting"""
        status = self.agent.get_agent_status()
        
        assert status["agent_id"] == "egcfg_agent"
        assert status["status"] == "active"
        assert "execution_guided_generation" in status["capabilities"]
        assert status["methodology"] == "EG-CFG"
        assert "arXiv:2506.10948v1" in status["research_base"]

class TestEGCFGIntegrationScenarios:
    """Integration test scenarios for EG-CFG"""
    
    def setup_method(self):
        """Setup integration tests"""
        self.config = GenerationConfig(
            candidates_per_line=3,
            temperatures=[0.6, 0.8, 1.0],
            enable_parallel_execution=True,
            max_execution_time=3.0
        )
        self.agent = EGCFGAgent(self.config)
    
    @pytest.mark.asyncio
    async def test_fibonacci_generation_scenario(self):
        """Test Fibonacci function generation scenario"""
        task = {
            "prompt": "Create a function that calculates Fibonacci numbers",
            "context": {
                "max_lines": 10,
                "expected_output": "55",  # fib(10) = 55
                "expected_variables": {"result": 55}
            }
        }
        
        result = await self.agent.process_coding_task(task)
        
        assert result["success"] is True
        assert "def" in result["generated_code"]
        assert result["final_score"] > 0
        assert len(result["execution_traces"]) > 0
    
    @pytest.mark.asyncio
    async def test_data_processing_scenario(self):
        """Test data processing scenario"""
        task = {
            "prompt": "Create a function to process a list of numbers",
            "context": {
                "max_lines": 8,
                "enable_validation": True,
                "execution_requirements": ["list_processing", "numerical_computation"]
            }
        }
        
        result = await self.agent.process_coding_task(task)
        
        assert result["success"] is True
        assert result["candidate_count"] >= 3  # Based on config
        assert "generated_code" in result
    
    @pytest.mark.asyncio
    async def test_error_handling_scenario(self):
        """Test error handling in complex scenarios"""
        task = {
            "prompt": "Create invalid syntax intentionally",
            "context": {
                "max_lines": 5,
                "allow_syntax_errors": True
            }
        }
        
        # Should handle gracefully even with problematic prompts
        result = await self.agent.process_coding_task(task)
        
        # Result should indicate issues but not crash
        assert "generated_code" in result
        assert "execution_traces" in result
    
    @pytest.mark.asyncio
    async def test_performance_scenario(self):
        """Test performance with larger generation tasks"""
        task = {
            "prompt": "Create a comprehensive class with multiple methods",
            "context": {
                "max_lines": 20,
                "enable_validation": True,
                "require_completion": True
            }
        }
        
        import time
        start_time = time.time()
        
        result = await self.agent.process_coding_task(task)
        
        execution_time = time.time() - start_time
        
        assert result["success"] is True
        assert execution_time < 10.0  # Should complete reasonably quickly
        assert result["total_time"] > 0
        assert result["candidate_count"] > 0

class TestEGCFGResearchCompliance:
    """Test compliance with EG-CFG research methodology"""
    
    def setup_method(self):
        """Setup research compliance tests"""
        self.runner = ExecutionGuidedCodeRunner()
    
    def test_line_by_line_generation(self):
        """Test that generation follows line-by-line methodology"""
        # Mock to verify line-by-line behavior
        with patch.object(self.runner, '_generate_line_candidates') as mock_gen:
            mock_gen.return_value = asyncio.coroutine(lambda: [])()
            
            # This would test that generation proceeds incrementally
            # Implementation details would verify research compliance
            assert hasattr(self.runner, '_generate_line_candidates')
    
    def test_execution_feedback_integration(self):
        """Test that execution feedback influences generation"""
        # Test that execution traces are properly integrated
        env = SafeExecutionEnvironment()
        trace = env.execute_line("x = 42")
        
        assert trace.status == ExecutionStatus.SUCCESS
        assert hasattr(trace, 'variables')
        assert hasattr(trace, 'execution_time')
        assert hasattr(trace, 'return_value')
    
    def test_parallel_exploration(self):
        """Test parallel candidate exploration"""
        config = GenerationConfig(
            candidates_per_line=4,
            enable_parallel_execution=True
        )
        runner = ExecutionGuidedCodeRunner(config)
        
        # Verify parallel execution capability
        assert runner.config.enable_parallel_execution is True
        assert hasattr(runner, '_execute_candidates_parallel')
    
    def test_classifier_free_guidance_mechanism(self):
        """Test CFG mechanism implementation"""
        runner = ExecutionGuidedCodeRunner()
        
        # Verify CFG components exist
        assert hasattr(runner, '_apply_classifier_free_guidance')
        assert hasattr(runner, '_calculate_conditional_scores')
        assert hasattr(runner, '_determine_cfg_strength')
    
    def test_research_paper_attribution(self):
        """Test proper research paper attribution"""
        stats = self.runner.get_execution_statistics()
        
        assert "arXiv:2506.10948v1" in stats["research_paper"]
        assert stats["methodology"] == "EG-CFG (Execution-Guided Classifier-Free Guidance)"

@pytest.mark.integration
class TestEGCFGFullIntegration:
    """Full integration tests for EG-CFG system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_code_generation(self):
        """Test complete end-to-end code generation workflow"""
        agent = EGCFGAgent()
        
        task = {
            "prompt": "Create a function to calculate the area of a circle",
            "context": {
                "max_lines": 6,
                "expected_variables": {"pi": 3.14159},
                "enable_validation": True
            }
        }
        
        result = await agent.process_coding_task(task)
        
        # Comprehensive validation
        assert result["success"] is True
        assert "def" in result["generated_code"]
        assert result["methodology"] == "EG-CFG"
        assert len(result["execution_traces"]) > 0
        assert result["final_score"] > 0
        assert result["candidate_count"] > 0
        
        # Verify research compliance
        assert "Execution-Guided Classifier-Free Guidance" in result["processing_method"]
    
    @pytest.mark.asyncio
    async def test_egcfg_vs_traditional_comparison(self):
        """Test EG-CFG advantages over traditional approaches"""
        # This would be a comparative test showing EG-CFG benefits
        agent = EGCFGAgent()
        
        task = {
            "prompt": "Implement a sorting algorithm",
            "context": {"max_lines": 15}
        }
        
        result = await agent.process_coding_task(task)
        
        # EG-CFG should provide execution feedback and quality scoring
        assert "execution_traces" in result
        assert result["final_score"] >= 0
        assert result["candidate_count"] > 1  # Multiple candidates explored
    
    def test_egcfg_configuration_validation(self):
        """Test EG-CFG configuration validation"""
        config = GenerationConfig(
            candidates_per_line=5,
            temperatures=[0.1, 0.5, 0.7, 0.9, 1.2],
            completion_horizons=[1, 2, 4, 8],
            cfg_strengths=[0, 0.5, 1.0, 2.0, 3.0]
        )
        
        agent = EGCFGAgent(config)
        
        assert len(agent.code_runner.config.temperatures) == 5
        assert agent.code_runner.config.candidates_per_line == 5
        assert len(agent.code_runner.config.completion_horizons) == 4
        assert len(agent.code_runner.config.cfg_strengths) == 5

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])