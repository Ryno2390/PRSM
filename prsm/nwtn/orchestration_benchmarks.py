"""
Orchestration Pattern Benchmarks

Comprehensive benchmarking suite for evaluating and comparing
orchestration patterns in the DGM-enhanced NWTN orchestrator.

Implements staged evaluation (quick → comprehensive → production)
following the DGM roadmap Phase 2.1 requirements.
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import statistics
import random

# PRSM imports
from prsm.core.models import (
    UserInput, PRSMSession, TaskStatus, AgentType
)

# DGM imports
from ..evolution.models import (
    ComponentType, EvaluationResult, PerformanceStats
)
from .dgm_orchestrator import (
    DGMEnhancedNWTNOrchestrator, OrchestrationPattern, OrchestrationMetrics
)

logger = logging.getLogger(__name__)


class BenchmarkTier(str, Enum):
    """Benchmark evaluation tiers."""
    QUICK = "quick"
    COMPREHENSIVE = "comprehensive"
    PRODUCTION = "production"


class TaskComplexity(str, Enum):
    """Task complexity levels for benchmarking."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass
class BenchmarkTask:
    """Individual benchmark task definition."""
    
    task_id: str
    name: str
    description: str
    complexity: TaskComplexity
    
    # Input specification
    user_query: str
    expected_agent_types: List[AgentType]
    context_requirements: Dict[str, Any]
    
    # Performance expectations
    expected_latency_ms: float
    expected_quality_threshold: float
    max_acceptable_latency_ms: float
    
    # Success criteria
    required_reasoning_steps: int
    success_indicators: List[str]
    
    # Resource constraints
    max_context_tokens: int
    max_ftns_cost: float
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = f"task_{uuid.uuid4().hex[:8]}"


@dataclass
class BenchmarkResult:
    """Result from executing a benchmark task."""
    
    task_id: str
    orchestrator_id: str
    pattern_id: str
    
    # Execution metrics
    start_time: datetime
    end_time: datetime
    success: bool
    
    # Performance metrics
    total_latency_ms: float
    processing_latency_ms: float
    quality_score: float
    
    # Resource usage
    context_tokens_used: int
    ftns_cost: float
    agents_invoked: int
    
    # Error tracking
    errors_encountered: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> float:
        """Calculate total duration in milliseconds."""
        return (self.end_time - self.start_time).total_seconds() * 1000
    
    @property
    def meets_latency_requirement(self) -> bool:
        """Check if result meets latency requirements."""
        return self.total_latency_ms <= self.expected_latency_ms
    
    @property
    def meets_quality_requirement(self) -> bool:
        """Check if result meets quality requirements."""
        return self.quality_score >= self.expected_quality_threshold


class OrchestrationBenchmarkSuite:
    """
    Comprehensive benchmark suite for orchestration pattern evaluation.
    
    Provides tiered evaluation system with quick, comprehensive, and
    production-level benchmarks for systematic pattern assessment.
    """
    
    def __init__(self):
        self.benchmark_tasks = self._initialize_benchmark_tasks()
        self.tier_configurations = {
            BenchmarkTier.QUICK: {
                "task_count": 10,
                "timeout_seconds": 60,
                "complexity_filter": [TaskComplexity.SIMPLE, TaskComplexity.MODERATE]
            },
            BenchmarkTier.COMPREHENSIVE: {
                "task_count": 50,
                "timeout_seconds": 300,
                "complexity_filter": [TaskComplexity.SIMPLE, TaskComplexity.MODERATE, TaskComplexity.COMPLEX]
            },
            BenchmarkTier.PRODUCTION: {
                "task_count": 200,
                "timeout_seconds": 1800,
                "complexity_filter": [TaskComplexity.SIMPLE, TaskComplexity.MODERATE, 
                                    TaskComplexity.COMPLEX, TaskComplexity.EXPERT]
            }
        }
    
    def _initialize_benchmark_tasks(self) -> List[BenchmarkTask]:
        """Initialize comprehensive benchmark task suite."""
        
        tasks = []
        
        # Simple tasks
        tasks.extend([
            BenchmarkTask(
                task_id="simple_query_001",
                name="Basic Information Query",
                description="Simple factual question requiring single agent response",
                complexity=TaskComplexity.SIMPLE,
                user_query="What is the capital of France?",
                expected_agent_types=[AgentType.ROUTER, AgentType.EXECUTOR],
                context_requirements={"knowledge_base": "general"},
                expected_latency_ms=500,
                expected_quality_threshold=0.9,
                max_acceptable_latency_ms=1000,
                required_reasoning_steps=2,
                success_indicators=["correct_answer", "clear_response"],
                max_context_tokens=512,
                max_ftns_cost=10.0
            ),
            BenchmarkTask(
                task_id="simple_calculation_001",
                name="Basic Calculation",
                description="Simple mathematical calculation",
                complexity=TaskComplexity.SIMPLE,
                user_query="Calculate 15% tip on a $42.50 bill",
                expected_agent_types=[AgentType.ROUTER, AgentType.EXECUTOR],
                context_requirements={"tools": ["calculator"]},
                expected_latency_ms=300,
                expected_quality_threshold=0.95,
                max_acceptable_latency_ms=800,
                required_reasoning_steps=2,
                success_indicators=["correct_calculation", "clear_explanation"],
                max_context_tokens=256,
                max_ftns_cost=5.0
            ),
            BenchmarkTask(
                task_id="simple_translation_001",
                name="Basic Translation",
                description="Simple text translation task",
                complexity=TaskComplexity.SIMPLE,
                user_query="Translate 'Hello, how are you?' to Spanish",
                expected_agent_types=[AgentType.ROUTER, AgentType.EXECUTOR],
                context_requirements={"language_models": ["translation"]},
                expected_latency_ms=400,
                expected_quality_threshold=0.9,
                max_acceptable_latency_ms=1000,
                required_reasoning_steps=2,
                success_indicators=["accurate_translation", "proper_grammar"],
                max_context_tokens=512,
                max_ftns_cost=8.0
            )
        ])
        
        # Moderate complexity tasks
        tasks.extend([
            BenchmarkTask(
                task_id="moderate_analysis_001",
                name="Text Analysis",
                description="Analyze sentiment and extract key themes from text",
                complexity=TaskComplexity.MODERATE,
                user_query="Analyze the sentiment and main themes in this customer review: 'The product arrived quickly but the quality was disappointing. Customer service was helpful though.'",
                expected_agent_types=[AgentType.ARCHITECT, AgentType.ROUTER, AgentType.EXECUTOR, AgentType.COMPILER],
                context_requirements={"nlp_models": ["sentiment", "topic_extraction"]},
                expected_latency_ms=1200,
                expected_quality_threshold=0.8,
                max_acceptable_latency_ms=2500,
                required_reasoning_steps=4,
                success_indicators=["sentiment_analysis", "theme_extraction", "structured_output"],
                max_context_tokens=1024,
                max_ftns_cost=25.0
            ),
            BenchmarkTask(
                task_id="moderate_planning_001",
                name="Travel Planning",
                description="Create a travel itinerary with multiple constraints",
                complexity=TaskComplexity.MODERATE,
                user_query="Plan a 3-day weekend trip to San Francisco for 2 people, budget $800, interested in food and museums",
                expected_agent_types=[AgentType.ARCHITECT, AgentType.ROUTER, AgentType.EXECUTOR, AgentType.COMPILER],
                context_requirements={"apis": ["maps", "restaurants", "attractions"], "tools": ["budget_calculator"]},
                expected_latency_ms=2000,
                expected_quality_threshold=0.75,
                max_acceptable_latency_ms=4000,
                required_reasoning_steps=6,
                success_indicators=["itinerary_created", "budget_respected", "preferences_incorporated"],
                max_context_tokens=2048,
                max_ftns_cost=50.0
            ),
            BenchmarkTask(
                task_id="moderate_comparison_001",
                name="Product Comparison",
                description="Compare multiple products across various criteria",
                complexity=TaskComplexity.MODERATE,
                user_query="Compare iPhone 15, Samsung Galaxy S24, and Google Pixel 8 for photography, battery life, and price",
                expected_agent_types=[AgentType.ARCHITECT, AgentType.PROMPTER, AgentType.ROUTER, AgentType.EXECUTOR, AgentType.COMPILER],
                context_requirements={"databases": ["product_specs", "reviews"], "tools": ["comparison_matrix"]},
                expected_latency_ms=1800,
                expected_quality_threshold=0.8,
                max_acceptable_latency_ms=3500,
                required_reasoning_steps=5,
                success_indicators=["comprehensive_comparison", "structured_table", "recommendation"],
                max_context_tokens=1536,
                max_ftns_cost=40.0
            )
        ])
        
        # Complex tasks
        tasks.extend([
            BenchmarkTask(
                task_id="complex_research_001",
                name="Multi-source Research",
                description="Research project requiring multiple information sources and synthesis",
                complexity=TaskComplexity.COMPLEX,
                user_query="Research the impact of artificial intelligence on employment in the healthcare sector, including current trends, future projections, and policy recommendations",
                expected_agent_types=[AgentType.ARCHITECT, AgentType.PROMPTER, AgentType.ROUTER, AgentType.EXECUTOR, AgentType.COMPILER],
                context_requirements={"databases": ["academic_papers", "industry_reports"], "tools": ["web_search", "citation_manager"]},
                expected_latency_ms=5000,
                expected_quality_threshold=0.75,
                max_acceptable_latency_ms=10000,
                required_reasoning_steps=8,
                success_indicators=["comprehensive_research", "multiple_sources", "policy_recommendations", "proper_citations"],
                max_context_tokens=4096,
                max_ftns_cost=150.0
            ),
            BenchmarkTask(
                task_id="complex_programming_001",
                name="Software Architecture Design",
                description="Design a software system with multiple requirements and constraints",
                complexity=TaskComplexity.COMPLEX,
                user_query="Design a microservices architecture for an e-commerce platform that needs to handle 100k users, support real-time inventory, payment processing, and recommendation engine",
                expected_agent_types=[AgentType.ARCHITECT, AgentType.PROMPTER, AgentType.ROUTER, AgentType.EXECUTOR, AgentType.COMPILER],
                context_requirements={"knowledge_bases": ["software_patterns", "cloud_services"], "tools": ["diagram_generator", "capacity_calculator"]},
                expected_latency_ms=6000,
                expected_quality_threshold=0.7,
                max_acceptable_latency_ms=12000,
                required_reasoning_steps=10,
                success_indicators=["architecture_diagram", "service_definitions", "scalability_analysis", "technology_recommendations"],
                max_context_tokens=6144,
                max_ftns_cost=200.0
            )
        ])
        
        # Expert-level tasks
        tasks.extend([
            BenchmarkTask(
                task_id="expert_analysis_001",
                name="Complex Financial Analysis",
                description="Multi-dimensional financial analysis with forecasting and risk assessment",
                complexity=TaskComplexity.EXPERT,
                user_query="Analyze Tesla's financial performance over the last 5 years, project growth for next 3 years considering EV market trends, regulatory changes, and competition. Include risk assessment and investment recommendation.",
                expected_agent_types=[AgentType.ARCHITECT, AgentType.PROMPTER, AgentType.ROUTER, AgentType.EXECUTOR, AgentType.COMPILER],
                context_requirements={"databases": ["financial_data", "market_research"], "tools": ["financial_modeling", "risk_calculator", "charting"]},
                expected_latency_ms=10000,
                expected_quality_threshold=0.65,
                max_acceptable_latency_ms=20000,
                required_reasoning_steps=12,
                success_indicators=["historical_analysis", "growth_projections", "risk_assessment", "investment_recommendation", "supporting_charts"],
                max_context_tokens=8192,
                max_ftns_cost=400.0
            )
        ])
        
        return tasks
    
    async def evaluate_orchestrator(
        self,
        orchestrator: DGMEnhancedNWTNOrchestrator,
        tier: BenchmarkTier = BenchmarkTier.COMPREHENSIVE,
        pattern_id: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate orchestrator performance using specified benchmark tier.
        
        Args:
            orchestrator: The orchestrator to evaluate
            tier: Evaluation tier (quick/comprehensive/production)
            pattern_id: Optional pattern identifier for tracking
            
        Returns:
            EvaluationResult with comprehensive performance metrics
        """
        
        logger.info(f"Starting {tier.value} orchestration evaluation")
        
        tier_config = self.tier_configurations[tier]
        
        # Select tasks for this tier
        available_tasks = [
            task for task in self.benchmark_tasks
            if task.complexity in tier_config["complexity_filter"]
        ]
        
        selected_tasks = self._select_benchmark_tasks(
            available_tasks, 
            tier_config["task_count"]
        )
        
        # Execute benchmark tasks
        start_time = datetime.utcnow()
        results = []
        
        for task in selected_tasks:
            try:
                result = await self._execute_benchmark_task(
                    orchestrator,
                    task,
                    tier_config["timeout_seconds"],
                    pattern_id
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Benchmark task {task.task_id} failed: {e}")
                # Create failed result
                failed_result = BenchmarkResult(
                    task_id=task.task_id,
                    orchestrator_id=orchestrator.component_id,
                    pattern_id=pattern_id or "unknown",
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    success=False,
                    total_latency_ms=tier_config["timeout_seconds"] * 1000,
                    processing_latency_ms=0,
                    quality_score=0.0,
                    context_tokens_used=0,
                    ftns_cost=0.0,
                    agents_invoked=0,
                    errors_encountered=[str(e)]
                )
                results.append(failed_result)
        
        end_time = datetime.utcnow()
        
        # Analyze results and create evaluation
        return self._analyze_benchmark_results(
            results, 
            orchestrator.component_id,
            pattern_id,
            tier,
            start_time,
            end_time
        )
    
    def _select_benchmark_tasks(
        self, 
        available_tasks: List[BenchmarkTask], 
        count: int
    ) -> List[BenchmarkTask]:
        """Select balanced set of benchmark tasks."""
        
        if len(available_tasks) <= count:
            return available_tasks
        
        # Ensure balanced complexity distribution
        complexity_groups = {}
        for task in available_tasks:
            if task.complexity not in complexity_groups:
                complexity_groups[task.complexity] = []
            complexity_groups[task.complexity].append(task)
        
        selected = []
        tasks_per_complexity = count // len(complexity_groups)
        remaining = count % len(complexity_groups)
        
        for complexity, tasks in complexity_groups.items():
            task_count = tasks_per_complexity
            if remaining > 0:
                task_count += 1
                remaining -= 1
            
            # Randomly select tasks from this complexity group
            selected.extend(random.sample(tasks, min(task_count, len(tasks))))
        
        return selected[:count]
    
    async def _execute_benchmark_task(
        self,
        orchestrator: DGMEnhancedNWTNOrchestrator,
        task: BenchmarkTask,
        timeout_seconds: int,
        pattern_id: Optional[str]
    ) -> BenchmarkResult:
        """Execute individual benchmark task."""
        
        start_time = datetime.utcnow()
        
        # Create user input
        session_id = str(uuid.uuid4())
        user_input = UserInput(
            user_id=f"benchmark_user_{uuid.uuid4().hex[:8]}",
            prompt=task.user_query,
            session_id=session_id,
            context_requirements=task.context_requirements
        )
        
        # Create session
        session = PRSMSession(
            session_id=user_input.session_id,
            user_id=user_input.user_id,
            created_at=start_time,
            context_budget=task.max_context_tokens,
            ftns_budget=task.max_ftns_cost
        )
        
        try:
            # Execute orchestration with timeout
            response = await asyncio.wait_for(
                orchestrator.orchestrate_session(user_input, session),
                timeout=timeout_seconds
            )
            
            end_time = datetime.utcnow()
            
            # Calculate performance metrics
            total_latency_ms = (end_time - start_time).total_seconds() * 1000
            success = response.status == TaskStatus.COMPLETED
            quality_score = self._evaluate_response_quality(response, task)
            
            # Extract resource usage
            context_used = response.context_usage.tokens_used if response.context_usage else 0
            ftns_cost = float(response.context_usage.cost_ftns) if response.context_usage else 0.0
            agents_invoked = len([
                step for step in response.reasoning_trace
                if step.step_type in ['ARCHITECT', 'PROMPTER', 'ROUTER', 'EXECUTOR', 'COMPILER']
            ])
            
            return BenchmarkResult(
                task_id=task.task_id,
                orchestrator_id=orchestrator.component_id,
                pattern_id=pattern_id or "unknown",
                start_time=start_time,
                end_time=end_time,
                success=success,
                total_latency_ms=total_latency_ms,
                processing_latency_ms=total_latency_ms,  # Simplified for benchmark
                quality_score=quality_score,
                context_tokens_used=context_used,
                ftns_cost=ftns_cost,
                agents_invoked=agents_invoked
            )
            
        except asyncio.TimeoutError:
            end_time = datetime.utcnow()
            return BenchmarkResult(
                task_id=task.task_id,
                orchestrator_id=orchestrator.component_id,
                pattern_id=pattern_id or "unknown",
                start_time=start_time,
                end_time=end_time,
                success=False,
                total_latency_ms=timeout_seconds * 1000,
                processing_latency_ms=timeout_seconds * 1000,
                quality_score=0.0,
                context_tokens_used=0,
                ftns_cost=0.0,
                agents_invoked=0,
                errors_encountered=["timeout"]
            )
        
        except Exception as e:
            end_time = datetime.utcnow()
            return BenchmarkResult(
                task_id=task.task_id,
                orchestrator_id=orchestrator.component_id,
                pattern_id=pattern_id or "unknown",
                start_time=start_time,
                end_time=end_time,
                success=False,
                total_latency_ms=(end_time - start_time).total_seconds() * 1000,
                processing_latency_ms=0,
                quality_score=0.0,
                context_tokens_used=0,
                ftns_cost=0.0,
                agents_invoked=0,
                errors_encountered=[str(e)]
            )
    
    def _evaluate_response_quality(
        self, 
        response: Any,  # PRSMResponse
        task: BenchmarkTask
    ) -> float:
        """Evaluate quality of orchestration response."""
        
        quality_score = 0.0
        
        # Base score from completion status
        if response.status == TaskStatus.COMPLETED:
            quality_score = 0.6
        elif response.status == TaskStatus.PARTIAL:
            quality_score = 0.3
        else:
            quality_score = 0.0
        
        # Bonus for meeting reasoning step requirements
        reasoning_steps = len(response.reasoning_trace)
        if reasoning_steps >= task.required_reasoning_steps:
            quality_score += 0.2
        elif reasoning_steps >= task.required_reasoning_steps * 0.7:
            quality_score += 0.1
        
        # Bonus for response completeness
        if hasattr(response, 'content') and response.content:
            if len(response.content) > 100:  # Substantial response
                quality_score += 0.1
        
        # Penalty for safety violations
        safety_violations = sum(
            1 for step in response.reasoning_trace
            for flag in getattr(step, 'safety_flags', [])
            if flag.severity == "HIGH"
        )
        quality_score -= safety_violations * 0.15
        
        # Bonus for efficiency (meeting latency targets)
        if hasattr(response, 'processing_time_ms'):
            if response.processing_time_ms <= task.expected_latency_ms:
                quality_score += 0.1
        
        return max(0.0, min(1.0, quality_score))
    
    def _analyze_benchmark_results(
        self,
        results: List[BenchmarkResult],
        orchestrator_id: str,
        pattern_id: Optional[str],
        tier: BenchmarkTier,
        start_time: datetime,
        end_time: datetime
    ) -> EvaluationResult:
        """Analyze benchmark results and create evaluation."""
        
        if not results:
            return EvaluationResult(
                solution_id=pattern_id or "unknown",
                component_type=ComponentType.TASK_ORCHESTRATOR,
                performance_score=0.0,
                task_success_rate=0.0,
                tasks_evaluated=0,
                tasks_successful=0,
                evaluation_duration_seconds=0.0,
                evaluation_tier=tier.value,
                evaluator_version="1.0",
                benchmark_suite="orchestration_patterns"
            )
        
        # Calculate basic metrics
        successful_tasks = sum(1 for r in results if r.success)
        success_rate = successful_tasks / len(results)
        
        # Calculate latency metrics
        latencies = [r.total_latency_ms for r in results if r.success]
        avg_latency = statistics.mean(latencies) if latencies else 0
        median_latency = statistics.median(latencies) if latencies else 0
        
        # Calculate quality metrics
        quality_scores = [r.quality_score for r in results if r.success]
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0
        
        # Calculate throughput
        total_duration = (end_time - start_time).total_seconds()
        throughput = len(results) / total_duration if total_duration > 0 else 0
        
        # Calculate composite performance score
        # Weighted combination of success rate, quality, and efficiency
        latency_score = max(0, 1 - (avg_latency / 10000))  # Normalize to 10s max
        efficiency_score = min(1, throughput / 2)  # Normalize to 2 tasks/s max
        
        performance_score = (
            success_rate * 0.4 +
            avg_quality * 0.3 +
            latency_score * 0.2 +
            efficiency_score * 0.1
        )
        
        # Calculate statistical metrics
        performance_stats = None
        if quality_scores:
            std_dev = statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0
            confidence_interval = self._calculate_confidence_interval(quality_scores)
            
            performance_stats = PerformanceStats(
                mean=avg_quality,
                median=statistics.median(quality_scores),
                std_dev=std_dev,
                min_value=min(quality_scores),
                max_value=max(quality_scores),
                confidence_interval=confidence_interval,
                sample_size=len(quality_scores),
                statistical_significance=self._calculate_significance(quality_scores),
                noise_level=std_dev / avg_quality if avg_quality > 0 else 0
            )
        
        return EvaluationResult(
            solution_id=pattern_id or "unknown",
            component_type=ComponentType.TASK_ORCHESTRATOR,
            performance_score=performance_score,
            task_success_rate=success_rate,
            latency_ms=avg_latency,
            throughput_rps=throughput,
            tasks_evaluated=len(results),
            tasks_successful=successful_tasks,
            evaluation_duration_seconds=total_duration,
            performance_stats=performance_stats,
            evaluation_tier=tier.value,
            evaluator_version="1.0",
            benchmark_suite="orchestration_patterns"
        )
    
    def _calculate_confidence_interval(self, values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for performance values."""
        if len(values) < 2:
            mean_val = values[0] if values else 0
            return (mean_val, mean_val)
        
        mean_val = statistics.mean(values)
        std_dev = statistics.stdev(values)
        
        # Use t-distribution for small samples, normal for large
        if len(values) < 30:
            # Simplified t-distribution approximation
            t_value = 2.0  # Approximate t-value for 95% confidence
        else:
            t_value = 1.96  # Z-value for 95% confidence
        
        margin_error = t_value * (std_dev / (len(values) ** 0.5))
        
        return (mean_val - margin_error, mean_val + margin_error)
    
    def _calculate_significance(self, values: List[float]) -> float:
        """Calculate statistical significance of results."""
        if len(values) < 5:
            return 0.0
        
        # Simplified significance calculation
        # In real implementation, would use proper statistical tests
        std_dev = statistics.stdev(values)
        mean_val = statistics.mean(values)
        
        if mean_val == 0:
            return 0.0
        
        # Coefficient of variation as proxy for significance
        cv = std_dev / mean_val
        significance = max(0, min(1, 1 - cv))
        
        return significance


class OrchestrationPatternComparator:
    """
    Utility for comparing orchestration patterns and their performance.
    Provides statistical analysis and recommendations for pattern selection.
    """
    
    def __init__(self):
        self.comparison_cache: Dict[str, Any] = {}
    
    async def compare_patterns(
        self,
        pattern_evaluations: Dict[str, EvaluationResult],
        significance_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Compare multiple orchestration patterns statistically.
        
        Args:
            pattern_evaluations: Dict mapping pattern_id to EvaluationResult
            significance_threshold: Threshold for statistical significance
            
        Returns:
            Comprehensive comparison analysis
        """
        
        if len(pattern_evaluations) < 2:
            return {"error": "At least 2 patterns required for comparison"}
        
        # Rank patterns by performance
        ranked_patterns = sorted(
            pattern_evaluations.items(),
            key=lambda x: x[1].performance_score,
            reverse=True
        )
        
        best_pattern_id, best_eval = ranked_patterns[0]
        
        # Calculate performance deltas
        performance_deltas = {}
        for pattern_id, evaluation in pattern_evaluations.items():
            if pattern_id != best_pattern_id:
                delta = best_eval.performance_score - evaluation.performance_score
                performance_deltas[pattern_id] = delta
        
        # Identify significant improvements
        significant_improvements = {}
        for pattern_id, delta in performance_deltas.items():
            if delta > significance_threshold:
                significant_improvements[pattern_id] = {
                    "performance_delta": delta,
                    "relative_improvement": delta / pattern_evaluations[pattern_id].performance_score,
                    "confidence": self._calculate_improvement_confidence(
                        best_eval, pattern_evaluations[pattern_id]
                    )
                }
        
        # Analyze performance dimensions
        dimension_analysis = self._analyze_performance_dimensions(pattern_evaluations)
        
        # Generate recommendations
        recommendations = self._generate_pattern_recommendations(
            ranked_patterns,
            significant_improvements,
            dimension_analysis
        )
        
        return {
            "best_pattern": {
                "pattern_id": best_pattern_id,
                "performance_score": best_eval.performance_score,
                "success_rate": best_eval.task_success_rate,
                "average_latency_ms": best_eval.latency_ms,
                "throughput_rps": best_eval.throughput_rps
            },
            "pattern_rankings": [
                {
                    "pattern_id": pid,
                    "rank": i + 1,
                    "performance_score": eval_result.performance_score,
                    "relative_performance": eval_result.performance_score / best_eval.performance_score
                }
                for i, (pid, eval_result) in enumerate(ranked_patterns)
            ],
            "significant_improvements": significant_improvements,
            "dimension_analysis": dimension_analysis,
            "recommendations": recommendations,
            "statistical_summary": {
                "patterns_compared": len(pattern_evaluations),
                "best_performance": best_eval.performance_score,
                "performance_range": max(e.performance_score for e in pattern_evaluations.values()) - 
                                  min(e.performance_score for e in pattern_evaluations.values()),
                "average_performance": statistics.mean(e.performance_score for e in pattern_evaluations.values())
            }
        }
    
    def _calculate_improvement_confidence(
        self, 
        best_eval: EvaluationResult, 
        other_eval: EvaluationResult
    ) -> float:
        """Calculate confidence level in performance improvement."""
        
        # Simplified confidence calculation
        # In real implementation, would use proper statistical tests (t-test, Mann-Whitney U, etc.)
        
        if not (best_eval.performance_stats and other_eval.performance_stats):
            return 0.5  # Low confidence without detailed stats
        
        best_stats = best_eval.performance_stats
        other_stats = other_eval.performance_stats
        
        # Calculate effect size (Cohen's d approximation)
        pooled_std = ((best_stats.std_dev ** 2 + other_stats.std_dev ** 2) / 2) ** 0.5
        if pooled_std == 0:
            return 0.9 if best_stats.mean > other_stats.mean else 0.1
        
        effect_size = abs(best_stats.mean - other_stats.mean) / pooled_std
        
        # Convert effect size to confidence (simplified)
        confidence = min(0.95, 0.5 + (effect_size * 0.3))
        
        return confidence
    
    def _analyze_performance_dimensions(
        self, 
        pattern_evaluations: Dict[str, EvaluationResult]
    ) -> Dict[str, Any]:
        """Analyze performance across different dimensions."""
        
        dimensions = {
            "success_rate": [e.task_success_rate for e in pattern_evaluations.values()],
            "latency": [e.latency_ms or 0 for e in pattern_evaluations.values()],
            "throughput": [e.throughput_rps or 0 for e in pattern_evaluations.values()],
            "overall_performance": [e.performance_score for e in pattern_evaluations.values()]
        }
        
        analysis = {}
        
        for dimension, values in dimensions.items():
            if values:
                analysis[dimension] = {
                    "mean": statistics.mean(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "range": max(values) - min(values),
                    "coefficient_of_variation": statistics.stdev(values) / statistics.mean(values) 
                                             if len(values) > 1 and statistics.mean(values) > 0 else 0
                }
        
        return analysis
    
    def _generate_pattern_recommendations(
        self,
        ranked_patterns: List[Tuple[str, EvaluationResult]],
        significant_improvements: Dict[str, Any],
        dimension_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations for pattern selection."""
        
        recommendations = []
        
        best_pattern_id, best_eval = ranked_patterns[0]
        
        # Primary recommendation
        recommendations.append(
            f"Adopt pattern '{best_pattern_id}' as primary orchestration strategy "
            f"(performance score: {best_eval.performance_score:.3f})"
        )
        
        # Performance improvement recommendations
        if significant_improvements:
            max_improvement = max(
                improvement["performance_delta"] 
                for improvement in significant_improvements.values()
            )
            recommendations.append(
                f"Switching from worst-performing pattern could improve "
                f"performance by up to {max_improvement:.1%}"
            )
        
        # Dimension-specific recommendations
        if dimension_analysis.get("latency", {}).get("coefficient_of_variation", 0) > 0.3:
            recommendations.append(
                "High latency variability detected - consider patterns optimized for consistency"
            )
        
        if dimension_analysis.get("success_rate", {}).get("mean", 0) < 0.8:
            recommendations.append(
                "Low overall success rate - prioritize reliability-focused patterns"
            )
        
        # Statistical recommendations
        if len(ranked_patterns) > 3:
            top_3_performance = statistics.mean([
                eval_result.performance_score 
                for _, eval_result in ranked_patterns[:3]
            ])
            all_performance = statistics.mean([
                eval_result.performance_score 
                for _, eval_result in ranked_patterns
            ])
            
            if top_3_performance > all_performance * 1.1:
                recommendations.append(
                    "Focus evolution efforts on top 3 performing patterns for best results"
                )
        
        return recommendations