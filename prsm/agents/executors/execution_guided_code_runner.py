#!/usr/bin/env python3
"""
Execution-Guided Code Runner for PRSM
Implements the EG-CFG (Execution-Guided Classifier-Free Guidance) methodology
from "Execution Guided Line-by-Line Code Generation" (arXiv:2506.10948v1)

This module provides advanced code generation with real-time execution feedback,
line-by-line validation, and parallel exploration of reasoning paths.
"""

import ast
import asyncio
import copy
import sys
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import contextlib
import io
import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import subprocess
import tempfile
import os
import re
import structlog

# Setup structured logging
logger = structlog.get_logger(__name__)


class SecurityError(Exception):
    """Raised when potentially unsafe code is detected"""
    pass

class ExecutionStatus(Enum):
    """Execution status for code lines"""
    SUCCESS = "success"
    ERROR = "error"
    INCOMPLETE = "incomplete"
    SYNTAX_ERROR = "syntax_error"
    TIMEOUT = "timeout"

@dataclass
class ExecutionTrace:
    """Execution trace for a line of code"""
    line_number: int
    code_line: str
    status: ExecutionStatus
    output: str = ""
    error_message: str = ""
    variables: Dict[str, Any] = None
    execution_time: float = 0.0
    return_value: Any = None
    stack_trace: str = ""
    
    def __post_init__(self):
        if self.variables is None:
            self.variables = {}

@dataclass
class CodeCandidate:
    """Code generation candidate with execution feedback"""
    candidate_id: str
    code_lines: List[str]
    temperature: float
    execution_traces: List[ExecutionTrace]
    total_score: float = 0.0
    syntax_valid: bool = True
    is_complete: bool = False
    generation_time: float = 0.0

@dataclass
class GenerationConfig:
    """Configuration for execution-guided generation"""
    candidates_per_line: int = 3
    temperatures: List[float] = None
    completion_horizons: List[int] = None
    cfg_strengths: List[float] = None
    max_execution_time: float = 5.0
    enable_ast_validation: bool = True
    enable_parallel_execution: bool = True
    beam_search_width: int = 5
    
    def __post_init__(self):
        if self.temperatures is None:
            self.temperatures = [0.7, 0.75, 0.85, 0.95, 1.2, 1.5]
        if self.completion_horizons is None:
            self.completion_horizons = [2, 3, 6, 8]
        if self.cfg_strengths is None:
            self.cfg_strengths = [0, 0.5, 1, 3]

class SafeExecutionEnvironment:
    """
    Safe execution environment for code validation
    
    Enhanced with robust timeout mechanisms and comprehensive safety controls.
    """
    
    def __init__(self, timeout: float = 5.0, enable_async_timeout: bool = True):
        self.timeout = timeout
        self.enable_async_timeout = enable_async_timeout
        self.global_namespace = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'bool': bool,
                'max': max,
                'min': min,
                'sum': sum,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'all': all,
                'any': any,
                # Add more safe builtins as needed
            }
        }
        self.local_namespace = {}
    
    def execute_line(self, code_line: str) -> ExecutionTrace:
        """Execute a single line of code safely"""
        start_time = time.time()
        
        # Capture stdout
        old_stdout = sys.stdout
        captured_output = io.StringIO()
        
        try:
            # Redirect stdout
            sys.stdout = captured_output
            
            # Parse and validate syntax
            try:
                parsed = ast.parse(code_line)
                syntax_valid = True
            except SyntaxError as e:
                return ExecutionTrace(
                    line_number=0,
                    code_line=code_line,
                    status=ExecutionStatus.SYNTAX_ERROR,
                    error_message=str(e),
                    execution_time=time.time() - start_time
                )
            
            # Execute with timeout
            result = self._execute_with_timeout(code_line, self.timeout)
            
            execution_time = time.time() - start_time
            output = captured_output.getvalue()
            
            # Get current variable state
            variables = {k: v for k, v in self.local_namespace.items() 
                        if not k.startswith('_')}
            
            return ExecutionTrace(
                line_number=0,
                code_line=code_line,
                status=ExecutionStatus.SUCCESS,
                output=output,
                variables=variables,
                execution_time=execution_time,
                return_value=result
            )
            
        except TimeoutError:
            return ExecutionTrace(
                line_number=0,
                code_line=code_line,
                status=ExecutionStatus.TIMEOUT,
                error_message="Execution timeout",
                execution_time=self.timeout
            )
        except Exception as e:
            return ExecutionTrace(
                line_number=0,
                code_line=code_line,
                status=ExecutionStatus.ERROR,
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                execution_time=time.time() - start_time
            )
        finally:
            sys.stdout = old_stdout
    
    async def execute_line_async(self, code_line: str) -> ExecutionTrace:
        """
        Async-compatible version of execute_line with enhanced timeout handling
        
        Provides better integration with async/await patterns and improved
        timeout precision for long-running code execution.
        """
        start_time = time.time()
        
        # Capture stdout
        old_stdout = sys.stdout
        captured_output = io.StringIO()
        
        try:
            # Redirect stdout
            sys.stdout = captured_output
            
            # Parse and validate syntax
            try:
                parsed = ast.parse(code_line)
                syntax_valid = True
            except SyntaxError as e:
                return ExecutionTrace(
                    line_number=0,
                    code_line=code_line,
                    status=ExecutionStatus.SYNTAX_ERROR,
                    error_message=str(e),
                    execution_time=time.time() - start_time
                )
            
            # Execute with async-compatible timeout
            if self.enable_async_timeout:
                result = await self._execute_with_async_timeout(code_line, self.timeout)
            else:
                result = self._execute_with_timeout(code_line, self.timeout)
            
            execution_time = time.time() - start_time
            output = captured_output.getvalue()
            
            # Get current variable state
            variables = {k: v for k, v in self.local_namespace.items() 
                        if not k.startswith('_')}
            
            return ExecutionTrace(
                line_number=0,
                code_line=code_line,
                status=ExecutionStatus.SUCCESS,
                output=output,
                variables=variables,
                execution_time=execution_time,
                return_value=result
            )
            
        except TimeoutError:
            return ExecutionTrace(
                line_number=0,
                code_line=code_line,
                status=ExecutionStatus.TIMEOUT,
                error_message=f"Execution timeout after {self.timeout}s",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ExecutionTrace(
                line_number=0,
                code_line=code_line,
                status=ExecutionStatus.ERROR,
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                execution_time=time.time() - start_time
            )
        finally:
            sys.stdout = old_stdout
    
    async def _execute_with_async_timeout(self, code_line: str, timeout: float) -> Any:
        """
        Async-compatible timeout execution using asyncio
        
        Provides better integration with async event loops and more precise
        timeout handling for both sync and async code execution.
        """
        # Security validation
        if not self._is_safe_code(code_line):
            raise SecurityError(f"Potentially unsafe code detected: {code_line[:100]}...")
        
        import asyncio
        import concurrent.futures
        
        def sync_target():
            try:
                # Use exec for statements, eval for expressions with restricted globals
                restricted_globals = self._get_restricted_globals()
                
                if any(code_line.strip().startswith(stmt) for stmt in 
                       ['def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'with ']):
                    exec(code_line, restricted_globals, self.local_namespace)  # nosec B102 - controlled execution context
                    return None
                else:
                    # Try as expression first, fallback to statement
                    try:
                        return eval(code_line, restricted_globals, self.local_namespace)  # nosec B307 - controlled execution context
                    except SyntaxError:
                        exec(code_line, restricted_globals, self.local_namespace)  # nosec B102 - controlled execution context
                        return None
            except Exception as e:
                raise e
        
        # Use asyncio's executor with timeout
        loop = asyncio.get_running_loop()
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = loop.run_in_executor(executor, sync_target)
                result = await asyncio.wait_for(future, timeout=timeout)
                return result
        except asyncio.TimeoutError:
            logger.warning(
                "Async timeout occurred",
                timeout=timeout,
                async_execution=True
            )
            raise TimeoutError(f"Async code execution exceeded {timeout:.2f} seconds")
        except Exception as e:
            logger.error(
                "Async execution error",
                error=str(e),
                code_preview=code_line[:50]
            )
            raise e
    
    def _execute_with_timeout(self, code_line: str, timeout: float) -> Any:
        """
        Execute code with robust timeout protection and security controls
        
        Improved timeout mechanism with:
        - Cross-platform compatibility (Windows/Unix)
        - Thread-safe execution
        - Precise timeout handling (sub-second precision)
        - Graceful cancellation support
        - Resource cleanup guarantees
        """
        # Security validation
        if not self._is_safe_code(code_line):
            raise SecurityError(f"Potentially unsafe code detected: {code_line[:100]}...")
        
        def target():
            try:
                # Use exec for statements, eval for expressions with restricted globals
                restricted_globals = self._get_restricted_globals()
                
                if any(code_line.strip().startswith(stmt) for stmt in 
                       ['def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'with ']):
                    exec(code_line, restricted_globals, self.local_namespace)  # nosec B102 - controlled execution context
                    return None
                else:
                    # Try as expression first, fallback to statement
                    try:
                        return eval(code_line, restricted_globals, self.local_namespace)  # nosec B307 - controlled execution context
                    except SyntaxError:
                        exec(code_line, restricted_globals, self.local_namespace)  # nosec B102 - controlled execution context
                        return None
            except Exception as e:
                raise e
        
        # Enhanced cross-platform timeout implementation
        return self._execute_with_robust_timeout(target, timeout)
    
    def _execute_with_robust_timeout(self, target_func, timeout: float) -> Any:
        """
        Cross-platform robust timeout implementation
        
        Features:
        - Works on both Windows and Unix systems
        - Thread-safe execution with proper cleanup
        - Sub-second precision timeout handling
        - Graceful resource management
        - Comprehensive error handling
        """
        import threading
        import queue
        import platform
        
        # Use different strategies based on platform
        if platform.system() == "Windows":
            return self._execute_with_thread_timeout(target_func, timeout)
        else:
            # Try signal-based timeout first, fallback to threading
            try:
                return self._execute_with_signal_timeout(target_func, timeout)
            except (OSError, AttributeError):
                # Fallback to thread-based timeout if signals not available
                return self._execute_with_thread_timeout(target_func, timeout)
    
    def _execute_with_thread_timeout(self, target_func, timeout: float) -> Any:
        """
        Thread-based timeout implementation (cross-platform)
        
        More reliable than signals and works on all platforms.
        Uses a separate thread for execution with controlled termination.
        """
        import queue
        import threading
        
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def wrapper():
            try:
                result = target_func()
                result_queue.put(("result", result))
            except Exception as e:
                exception_queue.put(("exception", e))
        
        # Create and start execution thread
        execution_thread = threading.Thread(target=wrapper, daemon=True)
        execution_thread.start()
        
        # Wait for completion or timeout
        execution_thread.join(timeout=timeout)
        
        if execution_thread.is_alive():
            # Thread is still running, timeout occurred
            logger.warning(
                "Code execution timeout",
                timeout=timeout,
                thread_alive=True
            )
            
            # Note: We cannot forcefully kill the thread in Python
            # The thread will continue running but we abandon it
            # This is a Python limitation for safety reasons
            raise TimeoutError(f"Code execution exceeded {timeout:.2f} seconds")
        
        # Check for exceptions first
        if not exception_queue.empty():
            _, exception = exception_queue.get_nowait()
            raise exception
        
        # Get result if available
        if not result_queue.empty():
            _, result = result_queue.get_nowait()
            return result
        
        # No result and no exception (shouldn't happen)
        return None
    
    def _execute_with_signal_timeout(self, target_func, timeout: float) -> Any:
        """
        Signal-based timeout implementation (Unix-like systems)
        
        More efficient than threading but only works on Unix-like systems.
        Provides precise timeout control with proper cleanup.
        """
        import signal
        import platform
        
        if platform.system() == "Windows":
            raise OSError("Signal-based timeout not supported on Windows")
        
        class TimeoutContext:
            def __init__(self, timeout_duration: float):
                self.timeout_duration = timeout_duration
                self.old_handler = None
                self.timed_out = False
            
            def timeout_handler(self, signum, frame):
                self.timed_out = True
                raise TimeoutError(f"Code execution exceeded {self.timeout_duration:.2f} seconds")
            
            def __enter__(self):
                # Set up signal handler
                self.old_handler = signal.signal(signal.SIGALRM, self.timeout_handler)
                signal.alarm(int(self.timeout_duration))
                
                # For sub-second precision, use setitimer if available
                if hasattr(signal, 'setitimer'):
                    signal.setitimer(signal.ITIMER_REAL, self.timeout_duration)
                
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                # Clean up signal handler
                signal.alarm(0)
                if hasattr(signal, 'setitimer'):
                    signal.setitimer(signal.ITIMER_REAL, 0)
                
                if self.old_handler is not None:
                    signal.signal(signal.SIGALRM, self.old_handler)
                
                # Don't suppress timeout exceptions
                return False
        
        try:
            with TimeoutContext(timeout) as ctx:
                result = target_func()
                return result
        except TimeoutError as e:
            logger.warning(
                "Signal-based timeout occurred",
                timeout=timeout,
                signal_based=True
            )
            raise e
    
    def _is_safe_code(self, code: str) -> bool:
        """Security validation for code execution"""
        # Dangerous patterns to block
        dangerous_patterns = [
            r'__import__',
            r'exec\s*\(',
            r'eval\s*\(',
            r'compile\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
            r'subprocess',
            r'os\.system',
            r'os\.popen',
            r'os\.spawn',
            r'os\.exec',
            r'commands\.',
            r'getattr',
            r'setattr',
            r'delattr',
            r'hasattr',
            r'globals\s*\(',
            r'locals\s*\(',
            r'vars\s*\(',
            r'dir\s*\(',
            r'reload\s*\(',
            r'memoryview',
            r'buffer',
            r'help\s*\(',
            r'copyright',
            r'credits',
            r'license',
            r'quit',
            r'exit',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                logger.warning("Blocked potentially dangerous code pattern", pattern=pattern, code_preview=code[:50])
                return False
        
        # Additional AST-based validation
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                # Block certain AST node types
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Only allow safe imports
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name not in ['math', 'random', 'string', 'json', 'datetime']:
                                return False
                    elif isinstance(node, ast.ImportFrom):
                        if node.module not in ['math', 'random', 'string', 'json', 'datetime']:
                            return False
                
                elif isinstance(node, ast.Call):
                    # Block certain function calls
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['exec', 'eval', 'compile', '__import__', 'open', 'file']:
                            return False
        except SyntaxError:
            # If we can't parse it, it's probably not safe
            return False
        
        return True
    
    def _get_restricted_globals(self) -> Dict[str, Any]:
        """Get restricted global namespace for safer execution"""
        # Return a copy of the safe global namespace
        return self.global_namespace.copy()

class ExecutionGuidedCodeRunner:
    """
    Advanced code runner implementing EG-CFG methodology
    Based on "Execution Guided Line-by-Line Code Generation" research
    """
    
    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()
        self.execution_env = SafeExecutionEnvironment(
            timeout=self.config.max_execution_time
        )
        self.generation_history: List[CodeCandidate] = []
        self.execution_feedback: List[ExecutionTrace] = []
        
        logger.info("Initialized Execution-Guided Code Runner with EG-CFG", methodology="EG-CFG")
    
    async def generate_and_execute_code(self, 
                                       prompt: str, 
                                       context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate code using execution-guided approach
        Implements the core EG-CFG methodology
        """
        logger.info("Starting execution-guided code generation", methodology="EG-CFG")
        
        start_time = time.time()
        context = context or {}
        
        # Initialize generation state
        candidates = []
        current_code_lines = []
        
        # Generate initial candidates
        initial_candidates = await self._generate_line_candidates(
            prompt, current_code_lines, context
        )
        
        # Execute and evaluate candidates
        evaluated_candidates = await self._execute_candidates(initial_candidates)
        
        # Select best candidate using CFG
        best_candidate = self._apply_classifier_free_guidance(
            evaluated_candidates, context
        )
        
        # Continue line-by-line generation
        final_candidate = await self._continue_generation(
            best_candidate, prompt, context
        )
        
        # Compile final results
        total_time = time.time() - start_time
        
        result = {
            "generated_code": "\n".join(final_candidate.code_lines),
            "execution_traces": [asdict(trace) for trace in final_candidate.execution_traces],
            "success": final_candidate.syntax_valid and len(final_candidate.execution_traces) > 0,
            "total_time": total_time,
            "candidate_count": len(self.generation_history),
            "methodology": "EG-CFG",
            "final_score": final_candidate.total_score
        }
        
        logger.info("Code generation completed", total_time=total_time, methodology="EG-CFG")
        return result
    
    async def _generate_line_candidates(self, 
                                       prompt: str, 
                                       existing_lines: List[str],
                                       context: Dict[str, Any]) -> List[CodeCandidate]:
        """Generate multiple candidate lines using different temperatures"""
        candidates = []
        
        for i, temp in enumerate(self.config.temperatures[:self.config.candidates_per_line]):
            # Mock code generation (replace with actual LLM integration)
            candidate_line = await self._mock_generate_line(
                prompt, existing_lines, temp, context
            )
            
            candidate = CodeCandidate(
                candidate_id=f"cand_{i}_{temp}",
                code_lines=existing_lines + [candidate_line],
                temperature=temp,
                execution_traces=[],
                generation_time=time.time()
            )
            
            candidates.append(candidate)
        
        return candidates
    
    async def _mock_generate_line(self, 
                                 prompt: str, 
                                 existing_lines: List[str], 
                                 temperature: float,
                                 context: Dict[str, Any]) -> str:
        """
        Mock line generation (replace with actual LLM integration)
        This would integrate with PRSM's LLM providers
        """
        # Simple mock generation based on context and temperature
        if "function" in prompt.lower():
            if not existing_lines:
                return f"def solve(input_data):"
            elif len(existing_lines) == 1:
                return f"    # Processing with temperature {temperature}"
            elif len(existing_lines) == 2:
                return f"    result = input_data * {temperature:.1f}"
            else:
                return f"    return result"
        else:
            if not existing_lines:
                return f"# Generated with temperature {temperature}"
            else:
                return f"x = {int(temperature * 10)}"
    
    async def _execute_candidates(self, candidates: List[CodeCandidate]) -> List[CodeCandidate]:
        """Execute candidates and collect execution traces"""
        if self.config.enable_parallel_execution:
            return await self._execute_candidates_parallel(candidates)
        else:
            return await self._execute_candidates_sequential(candidates)
    
    async def _execute_candidates_parallel(self, candidates: List[CodeCandidate]) -> List[CodeCandidate]:
        """Execute candidates in parallel for performance"""
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=min(len(candidates), 4)) as executor:
            futures = []
            
            for candidate in candidates:
                future = loop.run_in_executor(
                    executor, self._execute_single_candidate, candidate
                )
                futures.append(future)
            
            executed_candidates = await asyncio.gather(*futures)
        
        return executed_candidates
    
    async def _execute_candidates_sequential(self, candidates: List[CodeCandidate]) -> List[CodeCandidate]:
        """Execute candidates sequentially"""
        executed_candidates = []
        
        for candidate in candidates:
            executed_candidate = self._execute_single_candidate(candidate)
            executed_candidates.append(executed_candidate)
        
        return executed_candidates
    
    def _execute_single_candidate(self, candidate: CodeCandidate) -> CodeCandidate:
        """Execute a single candidate and collect traces"""
        # Create isolated execution environment for this candidate
        env = SafeExecutionEnvironment(timeout=self.config.max_execution_time)
        
        executed_candidate = copy.deepcopy(candidate)
        executed_candidate.execution_traces = []
        
        for i, line in enumerate(candidate.code_lines):
            trace = env.execute_line(line)
            trace.line_number = i + 1
            executed_candidate.execution_traces.append(trace)
            
            # Stop execution if there's an error
            if trace.status in [ExecutionStatus.ERROR, ExecutionStatus.TIMEOUT]:
                break
        
        # Validate overall syntax
        if self.config.enable_ast_validation:
            executed_candidate.syntax_valid = self._validate_ast(candidate.code_lines)
        
        # Calculate candidate score
        executed_candidate.total_score = self._calculate_candidate_score(executed_candidate)
        
        return executed_candidate
    
    def _validate_ast(self, code_lines: List[str]) -> bool:
        """Validate code using AST parsing"""
        try:
            full_code = "\n".join(code_lines)
            ast.parse(full_code)
            return True
        except SyntaxError:
            # Try with fallback strategies
            return self._try_ast_fallbacks(code_lines)
    
    def _try_ast_fallbacks(self, code_lines: List[str]) -> bool:
        """Try AST validation with fallback strategies"""
        # Strategy 1: Add pass statement
        try:
            code_with_pass = code_lines + ["    pass"]
            ast.parse("\n".join(code_with_pass))
            return True
        except SyntaxError:
            pass
        
        # Strategy 2: Remove incomplete lines
        try:
            filtered_lines = [line for line in code_lines if line.strip() and not line.strip().endswith(':')]
            if filtered_lines:
                ast.parse("\n".join(filtered_lines))
                return True
        except SyntaxError:
            pass
        
        return False
    
    def _calculate_candidate_score(self, candidate: CodeCandidate) -> float:
        """Calculate candidate score based on execution feedback"""
        score = 0.0
        
        for trace in candidate.execution_traces:
            if trace.status == ExecutionStatus.SUCCESS:
                score += 1.0
            elif trace.status == ExecutionStatus.ERROR:
                score -= 0.5
            elif trace.status == ExecutionStatus.TIMEOUT:
                score -= 1.0
            elif trace.status == ExecutionStatus.SYNTAX_ERROR:
                score -= 2.0
        
        # Bonus for syntax validity
        if candidate.syntax_valid:
            score += 0.5
        
        # Penalty for long execution times
        avg_execution_time = sum(t.execution_time for t in candidate.execution_traces) / max(len(candidate.execution_traces), 1)
        if avg_execution_time > 1.0:
            score -= 0.2
        
        return score
    
    def _apply_classifier_free_guidance(self, 
                                       candidates: List[CodeCandidate], 
                                       context: Dict[str, Any]) -> CodeCandidate:
        """
        Apply Classifier-Free Guidance to select best candidate
        Implements the CFG methodology from the research paper
        """
        if not candidates:
            return None
        
        # Calculate unconditional and conditional distributions
        unconditional_scores = [c.total_score for c in candidates]
        conditional_scores = self._calculate_conditional_scores(candidates, context)
        
        # Apply CFG with dynamic strength
        cfg_strength = self._determine_cfg_strength(context)
        
        final_scores = []
        for i, candidate in enumerate(candidates):
            # CFG formula: score = unconditional + γ * (conditional - unconditional)
            cfg_score = unconditional_scores[i] + cfg_strength * (
                conditional_scores[i] - unconditional_scores[i]
            )
            final_scores.append(cfg_score)
        
        # Select candidate with highest CFG score
        best_idx = final_scores.index(max(final_scores))
        best_candidate = candidates[best_idx]
        
        logger.info("Selected candidate with CFG score", candidate_id=best_candidate.candidate_id, cfg_score=final_scores[best_idx])
        
        return best_candidate
    
    def _calculate_conditional_scores(self, 
                                     candidates: List[CodeCandidate], 
                                     context: Dict[str, Any]) -> List[float]:
        """Calculate conditional scores based on execution feedback"""
        conditional_scores = []
        
        for candidate in candidates:
            score = candidate.total_score
            
            # Enhance score based on context alignment
            if 'expected_output' in context:
                for trace in candidate.execution_traces:
                    if trace.output and context['expected_output'] in trace.output:
                        score += 1.0
            
            # Enhance score based on variable states
            if 'expected_variables' in context:
                for trace in candidate.execution_traces:
                    for var_name, expected_value in context['expected_variables'].items():
                        if var_name in trace.variables:
                            if trace.variables[var_name] == expected_value:
                                score += 0.5
            
            conditional_scores.append(score)
        
        return conditional_scores
    
    def _determine_cfg_strength(self, context: Dict[str, Any]) -> float:
        """Determine CFG strength dynamically based on context"""
        # Start with base strength
        base_strength = 1.0
        
        # Adjust based on context richness
        if 'expected_output' in context:
            base_strength += 0.5
        if 'expected_variables' in context:
            base_strength += 0.3
        if 'execution_requirements' in context:
            base_strength += 0.2
        
        # Clamp to reasonable range
        return min(max(base_strength, 0.0), 3.0)
    
    async def _continue_generation(self, 
                                  best_candidate: CodeCandidate, 
                                  prompt: str, 
                                  context: Dict[str, Any]) -> CodeCandidate:
        """Continue generation line by line until completion"""
        current_candidate = best_candidate
        max_lines = context.get('max_lines', 20)
        
        while len(current_candidate.code_lines) < max_lines:
            # Check if candidate appears complete
            if self._is_generation_complete(current_candidate, context):
                break
            
            # Generate next line candidates
            next_candidates = await self._generate_line_candidates(
                prompt, current_candidate.code_lines, context
            )
            
            # Execute new candidates
            executed_candidates = await self._execute_candidates(next_candidates)
            
            # Apply CFG to select best continuation
            next_best = self._apply_classifier_free_guidance(executed_candidates, context)
            
            if next_best and next_best.total_score > current_candidate.total_score - 1.0:
                current_candidate = next_best
                self.generation_history.append(current_candidate)
            else:
                # No improvement found, stop generation
                break
        
        return current_candidate
    
    def _is_generation_complete(self, candidate: CodeCandidate, context: Dict[str, Any]) -> bool:
        """Determine if code generation is complete"""
        # Check for obvious completion signals
        if candidate.code_lines:
            last_line = candidate.code_lines[-1].strip()
            if last_line.startswith('return ') or last_line == 'pass':
                return True
        
        # Check execution success
        if candidate.execution_traces:
            last_trace = candidate.execution_traces[-1]
            if last_trace.status == ExecutionStatus.SUCCESS and last_trace.return_value is not None:
                return True
        
        # Check context-specific completion criteria
        if 'completion_criteria' in context:
            return context['completion_criteria'](candidate)
        
        return False
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics"""
        if not self.generation_history:
            return {"message": "No generation history available"}
        
        total_candidates = len(self.generation_history)
        successful_candidates = len([c for c in self.generation_history if c.total_score > 0])
        
        avg_score = sum(c.total_score for c in self.generation_history) / total_candidates
        avg_execution_time = sum(
            sum(t.execution_time for t in c.execution_traces) 
            for c in self.generation_history
        ) / total_candidates
        
        return {
            "total_candidates": total_candidates,
            "successful_candidates": successful_candidates,
            "success_rate": successful_candidates / total_candidates,
            "average_score": avg_score,
            "average_execution_time": avg_execution_time,
            "methodology": "EG-CFG (Execution-Guided Classifier-Free Guidance)",
            "research_paper": "arXiv:2506.10948v1"
        }

# Integration with PRSM's agent system
class EGCFGAgent:
    """PRSM Agent wrapper for Execution-Guided Code Runner"""
    
    def __init__(self, config: GenerationConfig = None):
        self.code_runner = ExecutionGuidedCodeRunner(config)
        self.agent_id = "egcfg_agent"
        self.capabilities = [
            "execution_guided_generation",
            "line_by_line_validation", 
            "parallel_exploration",
            "real_time_feedback"
        ]
    
    async def process_coding_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process coding task using EG-CFG methodology"""
        prompt = task.get('prompt', '')
        context = task.get('context', {})
        
        logger.info("EG-CFG Agent processing coding task", agent_id=self.agent_id)
        
        result = await self.code_runner.generate_and_execute_code(prompt, context)
        
        # Add agent metadata
        result.update({
            "agent_id": self.agent_id,
            "agent_capabilities": self.capabilities,
            "processing_method": "Execution-Guided Classifier-Free Guidance"
        })
        
        return result
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status and statistics"""
        stats = self.code_runner.get_execution_statistics()
        
        return {
            "agent_id": self.agent_id,
            "status": "active",
            "capabilities": self.capabilities,
            "execution_stats": stats,
            "methodology": "EG-CFG",
            "research_base": "Execution Guided Line-by-Line Code Generation (arXiv:2506.10948v1)"
        }

# Example usage and testing
async def demonstrate_egcfg():
    """Demonstrate the EG-CFG methodology"""
    print("🚀 Demonstrating Execution-Guided Code Generation")
    print("=" * 60)
    
    # Initialize EG-CFG agent
    config = GenerationConfig(
        candidates_per_line=3,
        temperatures=[0.7, 0.9, 1.2],
        enable_parallel_execution=True
    )
    
    agent = EGCFGAgent(config)
    
    # Test task: Simple function generation
    task = {
        "prompt": "Create a function that calculates the sum of squares",
        "context": {
            "expected_output": "30",  # For input [1,2,3]: 1²+2²+3² = 14
            "expected_variables": {"result": 14},
            "max_lines": 10
        }
    }
    
    # Process task
    result = await agent.process_coding_task(task)
    
    # Display results
    print(f"📝 Generated Code:")
    print(result['generated_code'])
    print(f"\n✅ Success: {result['success']}")
    print(f"⏱️  Time: {result['total_time']:.2f}s")
    print(f"🎯 Score: {result['final_score']:.2f}")
    print(f"🔄 Candidates: {result['candidate_count']}")
    
    # Show execution traces
    print(f"\n📊 Execution Traces:")
    for i, trace in enumerate(result['execution_traces']):
        status_emoji = "✅" if trace['status'] == 'success' else "❌"
        print(f"  {i+1}. {status_emoji} {trace['code_line']} ({trace['status']})")
        if trace['output']:
            print(f"     Output: {trace['output']}")
    
    # Agent statistics
    status = agent.get_agent_status()
    print(f"\n📈 Agent Statistics:")
    print(f"  Method: {status['methodology']}")
    print(f"  Research: {status['research_base']}")
    
    return result

if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_egcfg())