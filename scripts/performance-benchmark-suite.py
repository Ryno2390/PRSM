#!/usr/bin/env python3
"""
PRSM Performance Benchmark Suite
Phase 1 validation comparing PRSM vs GPT-4/Claude on standardized tasks

This comprehensive benchmark suite validates Phase 1 requirements:
1. <2s average query latency on benchmark tasks
2. 1000 concurrent requests handled successfully
3. 95% output quality parity with GPT-4 on evaluation suite
4. 99.9% uptime on test network

Key Features:
- Standardized task evaluation across multiple domains
- Real-time performance metrics collection
- Quality assessment using automated scoring
- Concurrent load testing with realistic user patterns
- Comprehensive reporting and analysis
"""

import asyncio
import json
import time
import statistics
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4
from dataclasses import dataclass, field
from enum import Enum
import structlog
from decimal import Decimal
import aiohttp
import numpy as np
from pathlib import Path

from prsm.core.config import get_settings
from prsm.nwtn.orchestrator import get_nwtn_orchestrator
from prsm.tokenomics.enhanced_ftns_service import get_enhanced_ftns_service
from prsm.core.database_service import get_database_service

logger = structlog.get_logger(__name__)
settings = get_settings()

class BenchmarkTaskType(Enum):
    """Types of benchmark tasks"""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    REASONING = "reasoning"
    CREATIVE_WRITING = "creative_writing"
    ANALYSIS = "analysis"
    TRANSLATION = "translation"

class PerformanceMetric(Enum):
    """Performance metrics to track"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    SUCCESS_RATE = "success_rate"
    QUALITY_SCORE = "quality_score"
    COST_EFFICIENCY = "cost_efficiency"
    RESOURCE_USAGE = "resource_usage"

@dataclass
class BenchmarkTask:
    """Individual benchmark task definition"""
    task_id: str
    task_type: BenchmarkTaskType
    prompt: str
    expected_output: Optional[str] = None
    evaluation_criteria: Dict[str, Any] = field(default_factory=dict)
    difficulty_level: int = 1  # 1-5 scale
    max_tokens: int = 1000
    timeout_seconds: int = 30

@dataclass
class BenchmarkResult:
    """Result from a single benchmark task execution"""
    task_id: str
    platform: str  # "prsm", "gpt4", "claude"
    response: str
    latency_ms: float
    success: bool
    quality_score: float
    cost_ftns: Optional[Decimal] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report"""
    benchmark_id: str
    total_tasks: int
    total_duration_seconds: float
    platform_results: Dict[str, Dict[str, Any]]
    comparative_analysis: Dict[str, Any]
    phase1_compliance: Dict[str, bool]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class PerformanceBenchmarkSuite:
    """
    Comprehensive performance benchmark suite for Phase 1 validation
    
    Provides standardized testing framework to validate PRSM performance
    against established AI platforms with detailed quality and performance metrics.
    """
    
    def __init__(self):
        self.orchestrator = get_nwtn_orchestrator()
        self.ftns_service = get_enhanced_ftns_service()
        self.database_service = get_database_service()
        
        # Benchmark configuration
        self.benchmark_tasks: List[BenchmarkTask] = []
        self.results: List[BenchmarkResult] = []
        self.concurrent_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Performance targets (Phase 1 requirements)
        self.target_latency_ms = 2000  # <2s average latency
        self.target_concurrent_users = 1000  # 1000 concurrent requests
        self.target_success_rate = 0.999  # 99.9% uptime
        self.target_quality_parity = 0.95  # 95% quality parity with GPT-4
        
        # External API configuration (for comparison)
        self.external_apis = {
            "gpt4": {
                "endpoint": "https://api.openai.com/v1/chat/completions",
                "headers": {"Authorization": f"Bearer {settings.openai_api_key}"},
                "model": "gpt-4"
            },
            "claude": {
                "endpoint": "https://api.anthropic.com/v1/messages",
                "headers": {"X-API-Key": settings.anthropic_api_key},
                "model": "claude-3-sonnet-20240229"
            }
        }
        
        # Quality evaluation models
        self.quality_evaluators = {
            "semantic_similarity": self._evaluate_semantic_similarity,
            "task_completion": self._evaluate_task_completion,
            "coherence": self._evaluate_coherence,
            "factual_accuracy": self._evaluate_factual_accuracy,
            "creativity": self._evaluate_creativity
        }
        
        logger.info("Performance Benchmark Suite initialized")
    
    async def initialize_benchmark_tasks(self) -> None:
        """Initialize comprehensive benchmark task suite"""
        
        # Text Generation Tasks
        self.benchmark_tasks.extend([
            BenchmarkTask(
                task_id="text_gen_1",
                task_type=BenchmarkTaskType.TEXT_GENERATION,
                prompt="Write a professional email explaining a project delay to stakeholders.",
                evaluation_criteria={"coherence": 0.8, "professionalism": 0.9, "completeness": 0.85},
                difficulty_level=2,
                max_tokens=300
            ),
            BenchmarkTask(
                task_id="text_gen_2",
                task_type=BenchmarkTaskType.TEXT_GENERATION,
                prompt="Create a technical specification for a REST API endpoint that manages user authentication.",
                evaluation_criteria={"technical_accuracy": 0.9, "completeness": 0.85, "clarity": 0.8},
                difficulty_level=3,
                max_tokens=500
            )
        ])
        
        # Code Generation Tasks
        self.benchmark_tasks.extend([
            BenchmarkTask(
                task_id="code_gen_1",
                task_type=BenchmarkTaskType.CODE_GENERATION,
                prompt="Write a Python function that implements binary search on a sorted array.",
                expected_output="def binary_search(arr, target):",
                evaluation_criteria={"correctness": 0.95, "efficiency": 0.8, "style": 0.7},
                difficulty_level=2,
                max_tokens=200
            ),
            BenchmarkTask(
                task_id="code_gen_2",
                task_type=BenchmarkTaskType.CODE_GENERATION,
                prompt="Implement a React component for a data table with sorting and filtering capabilities.",
                evaluation_criteria={"functionality": 0.9, "best_practices": 0.8, "completeness": 0.85},
                difficulty_level=4,
                max_tokens=600
            )
        ])
        
        # Question Answering Tasks
        self.benchmark_tasks.extend([
            BenchmarkTask(
                task_id="qa_1",
                task_type=BenchmarkTaskType.QUESTION_ANSWERING,
                prompt="What are the key differences between microservices and monolithic architecture?",
                evaluation_criteria={"accuracy": 0.9, "comprehensiveness": 0.85, "clarity": 0.8},
                difficulty_level=3,
                max_tokens=400
            ),
            BenchmarkTask(
                task_id="qa_2",
                task_type=BenchmarkTaskType.QUESTION_ANSWERING,
                prompt="Explain the concept of blockchain consensus mechanisms and compare Proof of Work vs Proof of Stake.",
                evaluation_criteria={"technical_accuracy": 0.95, "depth": 0.8, "comparison_quality": 0.85},
                difficulty_level=4,
                max_tokens=500
            )
        ])
        
        # Reasoning Tasks
        self.benchmark_tasks.extend([
            BenchmarkTask(
                task_id="reasoning_1",
                task_type=BenchmarkTaskType.REASONING,
                prompt="A company's revenue increased by 20% year-over-year, but profits decreased by 10%. What are three possible explanations for this scenario?",
                evaluation_criteria={"logical_reasoning": 0.9, "business_understanding": 0.8, "creativity": 0.7},
                difficulty_level=3,
                max_tokens=300
            ),
            BenchmarkTask(
                task_id="reasoning_2",
                task_type=BenchmarkTaskType.REASONING,
                prompt="If you could travel back in time but only once, and you had to choose between preventing a historical disaster or meeting a historical figure, which would you choose and why?",
                evaluation_criteria={"logical_argumentation": 0.8, "ethical_reasoning": 0.85, "depth": 0.75},
                difficulty_level=2,
                max_tokens=400
            )
        ])
        
        # Creative Writing Tasks
        self.benchmark_tasks.extend([
            BenchmarkTask(
                task_id="creative_1",
                task_type=BenchmarkTaskType.CREATIVE_WRITING,
                prompt="Write the opening paragraph of a science fiction story set in a world where AI has solved climate change.",
                evaluation_criteria={"creativity": 0.9, "narrative_structure": 0.8, "engagement": 0.85},
                difficulty_level=3,
                max_tokens=200
            ),
            BenchmarkTask(
                task_id="creative_2",
                task_type=BenchmarkTaskType.CREATIVE_WRITING,
                prompt="Create a haiku about the intersection of technology and nature.",
                evaluation_criteria={"creativity": 0.85, "poetic_structure": 0.95, "thematic_coherence": 0.8},
                difficulty_level=2,
                max_tokens=50
            )
        ])
        
        logger.info(f"Initialized {len(self.benchmark_tasks)} benchmark tasks across {len(set(task.task_type for task in self.benchmark_tasks))} categories")
    
    async def run_prsm_benchmark(self, task: BenchmarkTask) -> BenchmarkResult:
        """Execute benchmark task using PRSM system"""
        start_time = time.perf_counter()
        session_id = str(uuid4())
        
        try:
            # Create PRSM session
            session = await self.orchestrator.create_session(
                user_id="benchmark_system",
                session_type="performance_benchmark",
                metadata={
                    "task_id": task.task_id,
                    "task_type": task.task_type.value,
                    "benchmark_run": True
                }
            )
            
            # Execute task through NWTN orchestrator
            response = await self.orchestrator.process_query(
                session_id=session.session_id,
                query=task.prompt,
                max_tokens=task.max_tokens,
                timeout=task.timeout_seconds
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Calculate quality score
            quality_score = await self._evaluate_response_quality(task, response.response)
            
            # Get FTNS cost
            cost_calculation = await self.ftns_service.get_session_cost_summary(session.session_id)
            cost_ftns = cost_calculation.total_cost if cost_calculation else Decimal('0')
            
            return BenchmarkResult(
                task_id=task.task_id,
                platform="prsm",
                response=response.response,
                latency_ms=latency_ms,
                success=True,
                quality_score=quality_score,
                cost_ftns=cost_ftns,
                session_id=str(session.session_id),
                metadata={
                    "agents_used": response.metadata.get("agents_used", []),
                    "total_processing_time": response.metadata.get("total_processing_time", 0),
                    "cache_hits": response.metadata.get("cache_hits", 0)
                }
            )
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"PRSM benchmark task failed", task_id=task.task_id, error=str(e))
            
            return BenchmarkResult(
                task_id=task.task_id,
                platform="prsm",
                response="",
                latency_ms=latency_ms,
                success=False,
                quality_score=0.0,
                error_message=str(e),
                session_id=session_id
            )
    
    async def run_external_benchmark(self, task: BenchmarkTask, platform: str) -> BenchmarkResult:
        """Execute benchmark task using external API (GPT-4 or Claude)"""
        start_time = time.perf_counter()
        
        if platform not in self.external_apis:
            raise ValueError(f"Unknown platform: {platform}")
        
        api_config = self.external_apis[platform]
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=task.timeout_seconds)) as session:
                if platform == "gpt4":
                    payload = {
                        "model": api_config["model"],
                        "messages": [{"role": "user", "content": task.prompt}],
                        "max_tokens": task.max_tokens,
                        "temperature": 0.7
                    }
                elif platform == "claude":
                    payload = {
                        "model": api_config["model"],
                        "max_tokens": task.max_tokens,
                        "messages": [{"role": "user", "content": task.prompt}]
                    }
                
                async with session.post(
                    api_config["endpoint"],
                    headers=api_config["headers"],
                    json=payload
                ) as response:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        if platform == "gpt4":
                            response_text = data["choices"][0]["message"]["content"]
                        elif platform == "claude":
                            response_text = data["content"][0]["text"]
                        
                        quality_score = await self._evaluate_response_quality(task, response_text)
                        
                        return BenchmarkResult(
                            task_id=task.task_id,
                            platform=platform,
                            response=response_text,
                            latency_ms=latency_ms,
                            success=True,
                            quality_score=quality_score,
                            metadata={"api_response": data}
                        )
                    else:
                        error_text = await response.text()
                        return BenchmarkResult(
                            task_id=task.task_id,
                            platform=platform,
                            response="",
                            latency_ms=latency_ms,
                            success=False,
                            quality_score=0.0,
                            error_message=f"API error {response.status}: {error_text}"
                        )
        
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"{platform} benchmark task failed", task_id=task.task_id, error=str(e))
            
            return BenchmarkResult(
                task_id=task.task_id,
                platform=platform,
                response="",
                latency_ms=latency_ms,
                success=False,
                quality_score=0.0,
                error_message=str(e)
            )
    
    async def run_concurrent_load_test(self, concurrent_users: int = 1000, duration_seconds: int = 300) -> Dict[str, Any]:
        """
        Execute concurrent load test to validate 1000 concurrent users requirement
        
        Args:
            concurrent_users: Number of concurrent users to simulate
            duration_seconds: Duration of the load test
            
        Returns:
            Load test results and performance metrics
        """
        logger.info(f"Starting concurrent load test", users=concurrent_users, duration=duration_seconds)
        
        # Select subset of tasks for load testing
        load_test_tasks = self.benchmark_tasks[:4]  # Use first 4 tasks for load testing
        
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        
        # Results tracking
        completed_requests = []
        failed_requests = []
        active_sessions = set()
        
        async def simulate_user_session():
            """Simulate a single user session"""
            session_id = str(uuid4())
            active_sessions.add(session_id)
            
            try:
                while time.perf_counter() < end_time:
                    # Select random task
                    task = np.random.choice(load_test_tasks)
                    
                    # Execute task
                    result = await self.run_prsm_benchmark(task)
                    
                    if result.success:
                        completed_requests.append(result)
                    else:
                        failed_requests.append(result)
                    
                    # Variable delay between requests (1-5 seconds)
                    await asyncio.sleep(np.random.uniform(1, 5))
                    
            except Exception as e:
                logger.error(f"User session failed", session_id=session_id, error=str(e))
            finally:
                active_sessions.discard(session_id)
        
        # Launch concurrent user sessions
        tasks = [simulate_user_session() for _ in range(concurrent_users)]
        
        try:
            # Run with timeout
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=duration_seconds + 60  # Extra buffer
            )
        except asyncio.TimeoutError:
            logger.warning("Load test timed out, collecting partial results")
        
        total_duration = time.perf_counter() - start_time
        total_requests = len(completed_requests) + len(failed_requests)
        success_rate = len(completed_requests) / total_requests if total_requests > 0 else 0
        
        # Calculate performance metrics
        if completed_requests:
            latencies = [r.latency_ms for r in completed_requests]
            avg_latency = statistics.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
        else:
            avg_latency = p95_latency = p99_latency = 0
        
        throughput = total_requests / total_duration if total_duration > 0 else 0
        
        load_test_results = {
            "test_config": {
                "concurrent_users": concurrent_users,
                "duration_seconds": duration_seconds,
                "task_count": len(load_test_tasks)
            },
            "performance_metrics": {
                "total_requests": total_requests,
                "successful_requests": len(completed_requests),
                "failed_requests": len(failed_requests),
                "success_rate": success_rate,
                "avg_latency_ms": avg_latency,
                "p95_latency_ms": p95_latency,
                "p99_latency_ms": p99_latency,
                "throughput_rps": throughput,
                "peak_concurrent_sessions": len(active_sessions)
            },
            "phase1_compliance": {
                "concurrent_users_target": concurrent_users >= self.target_concurrent_users,
                "latency_target": avg_latency <= self.target_latency_ms,
                "success_rate_target": success_rate >= self.target_success_rate
            },
            "recommendations": []
        }
        
        # Add recommendations based on results
        if success_rate < self.target_success_rate:
            load_test_results["recommendations"].append(
                f"Success rate {success_rate:.3f} below target {self.target_success_rate}. Investigate error patterns and improve system reliability."
            )
        
        if avg_latency > self.target_latency_ms:
            load_test_results["recommendations"].append(
                f"Average latency {avg_latency:.1f}ms exceeds target {self.target_latency_ms}ms. Optimize query processing pipeline."
            )
        
        logger.info("Concurrent load test completed",
                   total_requests=total_requests,
                   success_rate=success_rate,
                   avg_latency=avg_latency,
                   throughput=throughput)
        
        return load_test_results
    
    async def run_comprehensive_benchmark(self, include_external: bool = True) -> BenchmarkReport:
        """
        Execute comprehensive benchmark comparing PRSM against external platforms
        
        Args:
            include_external: Whether to include external API comparisons
            
        Returns:
            Comprehensive benchmark report with analysis
        """
        logger.info("Starting comprehensive benchmark suite")
        start_time = time.perf_counter()
        benchmark_id = str(uuid4())
        
        # Ensure tasks are initialized
        if not self.benchmark_tasks:
            await self.initialize_benchmark_tasks()
        
        # Execute PRSM benchmarks
        logger.info("Running PRSM benchmarks")
        prsm_tasks = [self.run_prsm_benchmark(task) for task in self.benchmark_tasks]
        prsm_results = await asyncio.gather(*prsm_tasks, return_exceptions=True)
        
        # Filter successful results
        prsm_results = [r for r in prsm_results if isinstance(r, BenchmarkResult)]
        
        # Execute external benchmarks if requested
        external_results = {}
        if include_external:
            for platform in ["gpt4", "claude"]:
                if platform in self.external_apis:
                    logger.info(f"Running {platform} benchmarks")
                    platform_tasks = [self.run_external_benchmark(task, platform) for task in self.benchmark_tasks]
                    platform_results = await asyncio.gather(*platform_tasks, return_exceptions=True)
                    external_results[platform] = [r for r in platform_results if isinstance(r, BenchmarkResult)]
        
        # Aggregate results
        all_results = {
            "prsm": prsm_results,
            **external_results
        }
        
        # Calculate platform statistics
        platform_stats = {}
        for platform, results in all_results.items():
            if results:
                successful_results = [r for r in results if r.success]
                
                latencies = [r.latency_ms for r in successful_results]
                quality_scores = [r.quality_score for r in successful_results]
                
                platform_stats[platform] = {
                    "total_tasks": len(results),
                    "successful_tasks": len(successful_results),
                    "success_rate": len(successful_results) / len(results) if results else 0,
                    "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
                    "median_latency_ms": statistics.median(latencies) if latencies else 0,
                    "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
                    "avg_quality_score": statistics.mean(quality_scores) if quality_scores else 0,
                    "median_quality_score": statistics.median(quality_scores) if quality_scores else 0,
                    "total_cost_ftns": sum(r.cost_ftns or Decimal('0') for r in successful_results),
                    "avg_cost_per_task": statistics.mean([float(r.cost_ftns or 0) for r in successful_results]) if successful_results else 0
                }
        
        # Comparative analysis
        comparative_analysis = await self._generate_comparative_analysis(platform_stats, all_results)
        
        # Phase 1 compliance check
        prsm_stats = platform_stats.get("prsm", {})
        phase1_compliance = {
            "latency_target": prsm_stats.get("avg_latency_ms", float('inf')) <= self.target_latency_ms,
            "quality_target": prsm_stats.get("avg_quality_score", 0) >= self.target_quality_parity,
            "success_rate_target": prsm_stats.get("success_rate", 0) >= 0.95,  # 95% success rate for individual tasks
            "overall_compliance": False
        }
        
        phase1_compliance["overall_compliance"] = all(phase1_compliance.values())
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(platform_stats, phase1_compliance)
        
        total_duration = time.perf_counter() - start_time
        
        # Create comprehensive report
        report = BenchmarkReport(
            benchmark_id=benchmark_id,
            total_tasks=len(self.benchmark_tasks),
            total_duration_seconds=total_duration,
            platform_results=platform_stats,
            comparative_analysis=comparative_analysis,
            phase1_compliance=phase1_compliance,
            recommendations=recommendations
        )
        
        # Store results
        self.results.extend([r for results_list in all_results.values() for r in results_list])
        
        logger.info("Comprehensive benchmark completed",
                   benchmark_id=benchmark_id,
                   total_tasks=len(self.benchmark_tasks),
                   duration=total_duration,
                   phase1_compliant=phase1_compliance["overall_compliance"])
        
        return report
    
    async def _evaluate_response_quality(self, task: BenchmarkTask, response: str) -> float:
        """Evaluate response quality using multiple criteria"""
        quality_scores = []
        
        for criterion, evaluator in self.quality_evaluators.items():
            if criterion in task.evaluation_criteria:
                score = await evaluator(task, response)
                weighted_score = score * task.evaluation_criteria[criterion]
                quality_scores.append(weighted_score)
        
        # If no specific criteria, use general quality evaluation
        if not quality_scores:
            quality_scores = [
                await self._evaluate_semantic_similarity(task, response),
                await self._evaluate_task_completion(task, response),
                await self._evaluate_coherence(task, response)
            ]
        
        return statistics.mean(quality_scores) if quality_scores else 0.0
    
    async def _evaluate_semantic_similarity(self, task: BenchmarkTask, response: str) -> float:
        """Evaluate semantic similarity to expected output"""
        # Simplified heuristic-based evaluation
        # In production, this would use embedding models or specialized metrics
        
        if not response or len(response.strip()) == 0:
            return 0.0
        
        # Basic checks
        length_score = min(len(response) / max(task.max_tokens * 0.5, 50), 1.0)  # Reasonable length
        
        # Task-specific evaluation
        if task.task_type == BenchmarkTaskType.CODE_GENERATION:
            # Check for code-like structures
            code_indicators = ["def ", "function ", "class ", "{", "}", "return ", "if ", "for "]
            code_score = sum(1 for indicator in code_indicators if indicator in response) / len(code_indicators)
            return (length_score + code_score) / 2
        
        elif task.task_type == BenchmarkTaskType.QUESTION_ANSWERING:
            # Check for structured, informative response
            structure_score = 1.0 if len(response.split('.')) >= 3 else 0.7  # Multiple sentences
            return (length_score + structure_score) / 2
        
        # General evaluation
        return length_score * 0.8  # Conservative baseline
    
    async def _evaluate_task_completion(self, task: BenchmarkTask, response: str) -> float:
        """Evaluate how well the task was completed"""
        if not response or len(response.strip()) == 0:
            return 0.0
        
        # Check if response addresses the prompt
        prompt_words = set(task.prompt.lower().split())
        response_words = set(response.lower().split())
        
        # Calculate word overlap (simple metric)
        overlap = len(prompt_words.intersection(response_words))
        overlap_score = min(overlap / max(len(prompt_words) * 0.3, 1), 1.0)
        
        # Length appropriateness
        expected_length = task.max_tokens * 0.6  # Reasonable target
        actual_length = len(response.split())
        length_ratio = min(actual_length / expected_length, 1.5)  # Cap at 1.5x expected
        length_score = 1.0 - abs(1.0 - length_ratio) if length_ratio <= 1.5 else 0.5
        
        return (overlap_score + length_score) / 2
    
    async def _evaluate_coherence(self, task: BenchmarkTask, response: str) -> float:
        """Evaluate response coherence and readability"""
        if not response or len(response.strip()) == 0:
            return 0.0
        
        # Basic coherence metrics
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        # Sentence structure
        if len(sentences) == 0:
            return 0.0
        
        # Average sentence length (reasonable range)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        length_score = 1.0 if 5 <= avg_sentence_length <= 25 else 0.7
        
        # Repetition check
        words = response.lower().split()
        unique_words = set(words)
        repetition_score = len(unique_words) / max(len(words), 1)
        
        return (length_score + repetition_score) / 2
    
    async def _evaluate_factual_accuracy(self, task: BenchmarkTask, response: str) -> float:
        """Evaluate factual accuracy (simplified heuristic)"""
        # This would require external knowledge bases in production
        # Using basic heuristics for now
        
        if not response or len(response.strip()) == 0:
            return 0.0
        
        # Check for confident assertions vs hedged language
        confident_indicators = ["is", "are", "will", "must", "always", "never"]
        hedged_indicators = ["might", "could", "possibly", "likely", "probably", "may"]
        
        confident_count = sum(1 for word in confident_indicators if word in response.lower())
        hedged_count = sum(1 for word in hedged_indicators if word in response.lower())
        
        # Balanced confidence is good
        if confident_count + hedged_count == 0:
            return 0.8  # Neutral
        
        confidence_ratio = confident_count / (confident_count + hedged_count)
        
        # Prefer moderate confidence (not too confident, not too hedged)
        if 0.3 <= confidence_ratio <= 0.7:
            return 0.9
        elif 0.1 <= confidence_ratio <= 0.9:
            return 0.8
        else:
            return 0.6
    
    async def _evaluate_creativity(self, task: BenchmarkTask, response: str) -> float:
        """Evaluate creativity and originality"""
        if not response or len(response.strip()) == 0:
            return 0.0
        
        # Vocabulary diversity
        words = response.lower().split()
        unique_words = set(words)
        diversity_score = len(unique_words) / max(len(words), 1)
        
        # Use of interesting language features
        creative_indicators = ["metaphor", "analogy", "imagine", "envision", "creative", "innovative", "unique"]
        creativity_count = sum(1 for indicator in creative_indicators if indicator in response.lower())
        creativity_score = min(creativity_count / 2, 1.0)  # Cap at 1.0
        
        return (diversity_score + creativity_score) / 2
    
    async def _generate_comparative_analysis(self, platform_stats: Dict[str, Dict[str, Any]], all_results: Dict[str, List[BenchmarkResult]]) -> Dict[str, Any]:
        """Generate comparative analysis between platforms"""
        
        analysis = {
            "performance_comparison": {},
            "quality_comparison": {},
            "cost_efficiency": {},
            "strengths_weaknesses": {}
        }
        
        # Performance comparison
        if "prsm" in platform_stats:
            prsm_stats = platform_stats["prsm"]
            
            for platform, stats in platform_stats.items():
                if platform != "prsm":
                    analysis["performance_comparison"][f"prsm_vs_{platform}"] = {
                        "latency_ratio": prsm_stats["avg_latency_ms"] / max(stats["avg_latency_ms"], 1),
                        "quality_ratio": prsm_stats["avg_quality_score"] / max(stats["avg_quality_score"], 0.1),
                        "success_rate_diff": prsm_stats["success_rate"] - stats["success_rate"]
                    }
        
        # Quality comparison by task type
        quality_by_task_type = {}
        for task_type in BenchmarkTaskType:
            task_type_results = {}
            for platform, results in all_results.items():
                type_results = [r for r in results if r.success and any(t.task_type == task_type for t in self.benchmark_tasks if t.task_id == r.task_id)]
                if type_results:
                    task_type_results[platform] = statistics.mean(r.quality_score for r in type_results)
            
            if task_type_results:
                quality_by_task_type[task_type.value] = task_type_results
        
        analysis["quality_comparison"]["by_task_type"] = quality_by_task_type
        
        # Cost efficiency (PRSM only, as we don't have cost data for external APIs)
        if "prsm" in platform_stats:
            prsm_stats = platform_stats["prsm"]
            analysis["cost_efficiency"]["prsm"] = {
                "avg_cost_per_task": prsm_stats["avg_cost_per_task"],
                "cost_per_quality_point": prsm_stats["avg_cost_per_task"] / max(prsm_stats["avg_quality_score"], 0.1),
                "total_operational_cost": float(prsm_stats["total_cost_ftns"])
            }
        
        return analysis
    
    async def _generate_recommendations(self, platform_stats: Dict[str, Dict[str, Any]], phase1_compliance: Dict[str, bool]) -> List[str]:
        """Generate actionable recommendations based on benchmark results"""
        recommendations = []
        
        if "prsm" not in platform_stats:
            recommendations.append("PRSM benchmark execution failed. Investigate system stability and error handling.")
            return recommendations
        
        prsm_stats = platform_stats["prsm"]
        
        # Latency recommendations
        if not phase1_compliance["latency_target"]:
            avg_latency = prsm_stats["avg_latency_ms"]
            recommendations.append(
                f"Average latency ({avg_latency:.1f}ms) exceeds Phase 1 target ({self.target_latency_ms}ms). "
                "Consider: 1) Agent pipeline optimization, 2) Caching strategies, 3) Load balancing improvements."
            )
        
        # Quality recommendations
        if not phase1_compliance["quality_target"]:
            avg_quality = prsm_stats["avg_quality_score"]
            recommendations.append(
                f"Quality score ({avg_quality:.3f}) below Phase 1 target ({self.target_quality_parity}). "
                "Consider: 1) Agent model improvements, 2) Prompt engineering, 3) Output post-processing."
            )
        
        # Success rate recommendations
        if not phase1_compliance["success_rate_target"]:
            success_rate = prsm_stats["success_rate"]
            recommendations.append(
                f"Success rate ({success_rate:.3f}) below target. "
                "Investigate: 1) Error patterns, 2) Timeout handling, 3) System reliability improvements."
            )
        
        # Performance optimization recommendations
        p95_latency = prsm_stats.get("p95_latency_ms", 0)
        if p95_latency > self.target_latency_ms * 1.5:
            recommendations.append(
                f"P95 latency ({p95_latency:.1f}ms) indicates performance inconsistency. "
                "Implement: 1) Better resource allocation, 2) Query prioritization, 3) Circuit breakers."
            )
        
        # Cost efficiency recommendations
        avg_cost = prsm_stats.get("avg_cost_per_task", 0)
        if avg_cost > 1.0:  # Arbitrary threshold
            recommendations.append(
                f"Average cost per task ({avg_cost:.3f} FTNS) may be high for production use. "
                "Optimize: 1) Agent selection, 2) Token usage, 3) Processing efficiency."
            )
        
        return recommendations
    
    async def save_benchmark_results(self, report: BenchmarkReport, output_dir: Path = None) -> None:
        """Save benchmark results to files"""
        if output_dir is None:
            output_dir = Path("benchmark_results")
        
        output_dir.mkdir(exist_ok=True)
        
        # Save main report
        report_file = output_dir / f"benchmark_report_{report.benchmark_id}.json"
        with open(report_file, 'w') as f:
            json.dump({
                "benchmark_id": report.benchmark_id,
                "timestamp": report.timestamp.isoformat(),
                "total_tasks": report.total_tasks,
                "total_duration_seconds": report.total_duration_seconds,
                "platform_results": report.platform_results,
                "comparative_analysis": report.comparative_analysis,
                "phase1_compliance": report.phase1_compliance,
                "recommendations": report.recommendations
            }, f, indent=2, default=str)
        
        # Save detailed results
        detailed_results_file = output_dir / f"detailed_results_{report.benchmark_id}.json"
        with open(detailed_results_file, 'w') as f:
            json.dump([{
                "task_id": r.task_id,
                "platform": r.platform,
                "response": r.response,
                "latency_ms": r.latency_ms,
                "success": r.success,
                "quality_score": r.quality_score,
                "cost_ftns": float(r.cost_ftns) if r.cost_ftns else None,
                "timestamp": r.timestamp.isoformat(),
                "error_message": r.error_message,
                "metadata": r.metadata
            } for r in self.results], f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {output_dir}")


# === Benchmark Execution Functions ===

async def run_quick_benchmark():
    """Run a quick benchmark for development testing"""
    suite = PerformanceBenchmarkSuite()
    await suite.initialize_benchmark_tasks()
    
    # Run only PRSM benchmark (no external APIs)
    report = await suite.run_comprehensive_benchmark(include_external=False)
    
    print(f"\n=== Quick Benchmark Results ===")
    print(f"Benchmark ID: {report.benchmark_id}")
    print(f"Total Tasks: {report.total_tasks}")
    print(f"Duration: {report.total_duration_seconds:.2f}s")
    
    if "prsm" in report.platform_results:
        prsm_stats = report.platform_results["prsm"]
        print(f"\nPRSM Performance:")
        print(f"  Success Rate: {prsm_stats['success_rate']:.3f}")
        print(f"  Avg Latency: {prsm_stats['avg_latency_ms']:.1f}ms")
        print(f"  Avg Quality: {prsm_stats['avg_quality_score']:.3f}")
        print(f"  Total Cost: {prsm_stats['total_cost_ftns']} FTNS")
    
    print(f"\nPhase 1 Compliance:")
    for metric, compliant in report.phase1_compliance.items():
        status = "✅" if compliant else "❌"
        print(f"  {metric}: {status}")
    
    if report.recommendations:
        print(f"\nRecommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
    
    await suite.save_benchmark_results(report)
    return report


async def run_full_benchmark():
    """Run comprehensive benchmark including external API comparisons"""
    suite = PerformanceBenchmarkSuite()
    await suite.initialize_benchmark_tasks()
    
    print("Starting comprehensive benchmark suite...")
    print("This will compare PRSM against GPT-4 and Claude.")
    print("Estimated duration: 5-10 minutes")
    
    # Run comprehensive benchmark
    report = await suite.run_comprehensive_benchmark(include_external=True)
    
    print(f"\n=== Comprehensive Benchmark Results ===")
    print(f"Benchmark ID: {report.benchmark_id}")
    print(f"Total Tasks: {report.total_tasks}")
    print(f"Duration: {report.total_duration_seconds:.2f}s")
    
    # Print platform comparison
    for platform, stats in report.platform_results.items():
        print(f"\n{platform.upper()} Performance:")
        print(f"  Success Rate: {stats['success_rate']:.3f}")
        print(f"  Avg Latency: {stats['avg_latency_ms']:.1f}ms")
        print(f"  P95 Latency: {stats['p95_latency_ms']:.1f}ms")
        print(f"  Avg Quality: {stats['avg_quality_score']:.3f}")
        if platform == "prsm":
            print(f"  Total Cost: {stats['total_cost_ftns']} FTNS")
    
    print(f"\nPhase 1 Compliance:")
    for metric, compliant in report.phase1_compliance.items():
        status = "✅" if compliant else "❌"
        print(f"  {metric}: {status}")
    
    if report.recommendations:
        print(f"\nRecommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
    
    await suite.save_benchmark_results(report)
    return report


async def run_load_test():
    """Run concurrent load test for Phase 1 validation"""
    suite = PerformanceBenchmarkSuite()
    await suite.initialize_benchmark_tasks()
    
    print("Starting load test: 1000 concurrent users for 5 minutes")
    
    load_results = await suite.run_concurrent_load_test(
        concurrent_users=1000,
        duration_seconds=300
    )
    
    print(f"\n=== Load Test Results ===")
    config = load_results["test_config"]
    metrics = load_results["performance_metrics"]
    compliance = load_results["phase1_compliance"]
    
    print(f"Test Configuration:")
    print(f"  Concurrent Users: {config['concurrent_users']}")
    print(f"  Duration: {config['duration_seconds']}s")
    
    print(f"\nPerformance Metrics:")
    print(f"  Total Requests: {metrics['total_requests']}")
    print(f"  Success Rate: {metrics['success_rate']:.3f}")
    print(f"  Avg Latency: {metrics['avg_latency_ms']:.1f}ms")
    print(f"  P95 Latency: {metrics['p95_latency_ms']:.1f}ms")
    print(f"  Throughput: {metrics['throughput_rps']:.1f} req/s")
    
    print(f"\nPhase 1 Compliance:")
    for metric, compliant in compliance.items():
        if metric != "recommendations":
            status = "✅" if compliant else "❌"
            print(f"  {metric}: {status}")
    
    if load_results["recommendations"]:
        print(f"\nRecommendations:")
        for i, rec in enumerate(load_results["recommendations"], 1):
            print(f"  {i}. {rec}")
    
    return load_results


# === Global benchmark suite instance ===
performance_benchmark_suite = None

def get_performance_benchmark_suite() -> PerformanceBenchmarkSuite:
    """Get or create global benchmark suite instance"""
    global performance_benchmark_suite
    if performance_benchmark_suite is None:
        performance_benchmark_suite = PerformanceBenchmarkSuite()
    return performance_benchmark_suite


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            asyncio.run(run_quick_benchmark())
        elif sys.argv[1] == "full":
            asyncio.run(run_full_benchmark())
        elif sys.argv[1] == "load":
            asyncio.run(run_load_test())
        else:
            print("Usage: python performance-benchmark-suite.py [quick|full|load]")
    else:
        # Default to quick benchmark
        asyncio.run(run_quick_benchmark())