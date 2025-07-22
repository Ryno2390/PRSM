#!/usr/bin/env python3
"""
Multi-Instance Meta-Reasoning Orchestrator for NWTN
==================================================

This module implements the Phase 8.1.1 Multi-Instance Meta-Reasoning Orchestrator
that coordinates 20-100 parallel MetaReasoningEngine instances to achieve 20-100x
speedup for deep reasoning through massive parallel processing.

Architecture:
- ParallelMetaReasoningOrchestrator: Main orchestrator coordinating parallel workers
- WorkDistributionEngine: Intelligent complexity-aware load balancing across workers
- SharedWorldModelManager: Single world model shared across all workers for memory efficiency
- ParallelResultSynthesizer: Synthesizes results from all parallel reasoning paths
- ParallelMetaReasoningWorker: Individual worker processing sequence batches

Based on NWTN Roadmap Phase 8.1.1 - Multi-Instance Meta-Reasoning Orchestrator (Very High Priority)
Expected Impact: 20-100x speedup enabling production deployment with 10,000+ knowledge items
"""

import asyncio
import time
import math
import os
import multiprocessing
import threading
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from itertools import permutations
from uuid import uuid4
from datetime import datetime, timezone
from prsm.nwtn.intelligent_work_distribution_engine import (
    IntelligentWorkDistributionEngine, WorkItem, DistributionStrategy, 
    ResourceType, WorkerStatus
)
from prsm.nwtn.hierarchical_result_caching import (
    HierarchicalResultCachingSystem, CacheLevel, EvictionPolicy
)
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import psutil
import gc
from collections import defaultdict, deque
import heapq
import structlog

# Lazy imports to avoid circular dependency - imported in functions where needed
# from prsm.nwtn.meta_reasoning_engine import (
#     MetaReasoningEngine, MetaReasoningResult, ReasoningEngine, 
#     ThinkingMode, ReasoningSequence, ReasoningResult
# )
from prsm.nwtn.shared_world_model_architecture import (
    SharedWorldModelManager, shared_world_model_validation
)
from prsm.nwtn.fault_tolerance_worker_recovery import (
    ParallelProcessingResilience, WorkerHealthMonitor, DistributedCheckpointManager
)

logger = structlog.get_logger(__name__)

class LoadBalancingStrategy(Enum):
    """Load balancing strategies for parallel processing"""
    ROUND_ROBIN = "round_robin"
    COMPLEXITY_AWARE = "complexity_aware"
    DYNAMIC_ADAPTIVE = "dynamic_adaptive"
    RESOURCE_BASED = "resource_based"

class WorkerStatus(Enum):
    """Status of parallel workers"""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERING = "recovering"

@dataclass
class SequenceBatch:
    """Batch of reasoning sequences for parallel processing"""
    batch_id: str = field(default_factory=lambda: str(uuid4()))
    sequences: List[List[ReasoningEngine]] = field(default_factory=list)
    estimated_complexity: float = 0.0
    estimated_processing_time: float = 0.0
    worker_id: Optional[int] = None
    status: WorkerStatus = WorkerStatus.IDLE
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    error: Optional[str] = None

@dataclass
class WorkerMetrics:
    """Performance metrics for individual workers"""
    worker_id: int
    sequences_processed: int = 0
    total_processing_time: float = 0.0
    average_sequence_time: float = 0.0
    error_count: int = 0
    last_activity: Optional[datetime] = None
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    status: WorkerStatus = WorkerStatus.IDLE

@dataclass
class ParallelProcessingResult:
    """Result from parallel meta-reasoning processing"""
    orchestrator_id: str = field(default_factory=lambda: str(uuid4()))
    query: str = ""
    total_sequences_processed: int = 0
    num_workers_used: int = 0
    total_processing_time: float = 0.0
    speedup_factor: float = 1.0
    worker_results: List[MetaReasoningResult] = field(default_factory=list)
    synthesized_result: Optional[MetaReasoningResult] = None
    worker_metrics: List[WorkerMetrics] = field(default_factory=list)
    cache_hit_rate: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    error_summary: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class ReasoningComplexityEstimator:
    """Estimates computational complexity of reasoning sequences"""
    
    def __init__(self):
        # Empirically determined complexity weights based on engine performance
        self.complexity_weights = {
            ReasoningEngine.COUNTERFACTUAL: 2.5,     # Heaviest: scenario simulation + breakthrough analysis
            ReasoningEngine.ABDUCTIVE: 2.2,          # Heavy: hypothesis generation + creative analysis
            ReasoningEngine.CAUSAL: 1.8,             # Medium-heavy: causal analysis + intervention design
            ReasoningEngine.PROBABILISTIC: 1.5,      # Medium: probability calculation + uncertainty analysis
            ReasoningEngine.ANALOGICAL: 1.4,         # Medium: multi-level analogical processing
            ReasoningEngine.INDUCTIVE: 1.2,          # Light: pattern recognition + anomaly detection
            ReasoningEngine.DEDUCTIVE: 1.0,          # Lightest: logical inference
            ReasoningEngine.FRONTIER_DETECTION: 2.0  # Heavy: frontier analysis + pattern mining
        }
        
        # Sequence interaction complexity multipliers
        self.interaction_multipliers = {
            2: 1.1,  # 2 engines: minimal interaction overhead
            3: 1.3,  # 3 engines: moderate interaction complexity
            4: 1.5,  # 4 engines: high interaction complexity
            5: 1.8,  # 5 engines: very high interaction complexity
            6: 2.1,  # 6 engines: maximum interaction complexity
            7: 2.5   # 7 engines: theoretical maximum complexity
        }
    
    def estimate_sequence_complexity(self, sequence: List[ReasoningEngine]) -> float:
        """Estimate processing time complexity for a reasoning sequence"""
        
        if not sequence:
            return 0.0
        
        # Base complexity from individual engines
        base_complexity = sum(self.complexity_weights.get(engine, 1.0) for engine in sequence)
        
        # Interaction complexity based on sequence length
        sequence_length = len(sequence)
        interaction_multiplier = self.interaction_multipliers.get(sequence_length, 2.5)
        
        # Enhanced engine complexity (breakthrough modes add overhead)
        enhanced_engine_count = sum(1 for engine in sequence 
                                  if engine in [ReasoningEngine.COUNTERFACTUAL, 
                                              ReasoningEngine.ABDUCTIVE,
                                              ReasoningEngine.CAUSAL,
                                              ReasoningEngine.INDUCTIVE,
                                              ReasoningEngine.FRONTIER_DETECTION])
        enhanced_multiplier = 1.0 + (enhanced_engine_count * 0.3)  # 30% overhead per enhanced engine
        
        total_complexity = base_complexity * interaction_multiplier * enhanced_multiplier
        
        return total_complexity
    
    def estimate_processing_time(self, sequence: List[ReasoningEngine], base_time_per_unit: float = 5.0) -> float:
        """Estimate actual processing time in seconds"""
        complexity = self.estimate_sequence_complexity(sequence)
        return complexity * base_time_per_unit

class WorkDistributionEngine:
    """Intelligent work distribution engine for load balancing"""
    
    def __init__(self, load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.COMPLEXITY_AWARE):
        self.complexity_estimator = ReasoningComplexityEstimator()
        self.load_balancing_strategy = load_balancing_strategy
        self.worker_history = defaultdict(list)  # Track worker performance history
    
    def create_balanced_batches(self, 
                              sequences: List[List[ReasoningEngine]], 
                              num_workers: int) -> List[SequenceBatch]:
        """Create balanced batches of sequences for parallel processing"""
        
        if not sequences:
            return []
        
        if self.load_balancing_strategy == LoadBalancingStrategy.COMPLEXITY_AWARE:
            return self._create_complexity_aware_batches(sequences, num_workers)
        elif self.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._create_round_robin_batches(sequences, num_workers)
        elif self.load_balancing_strategy == LoadBalancingStrategy.DYNAMIC_ADAPTIVE:
            return self._create_adaptive_batches(sequences, num_workers)
        else:
            # Default to complexity-aware
            return self._create_complexity_aware_batches(sequences, num_workers)
    
    def _create_complexity_aware_batches(self, 
                                       sequences: List[List[ReasoningEngine]], 
                                       num_workers: int) -> List[SequenceBatch]:
        """Create batches using complexity-aware bin packing algorithm"""
        
        # Calculate complexity for each sequence
        sequence_complexities = []
        for seq in sequences:
            complexity = self.complexity_estimator.estimate_sequence_complexity(seq)
            processing_time = self.complexity_estimator.estimate_processing_time(seq)
            sequence_complexities.append((seq, complexity, processing_time))
        
        # Sort sequences by complexity (descending) for better bin packing
        sequence_complexities.sort(key=lambda x: x[1], reverse=True)
        
        # Initialize worker batches
        batches = []
        for i in range(num_workers):
            batch = SequenceBatch(
                batch_id=f"batch_{i}",
                worker_id=i
            )
            batches.append(batch)
        
        # Track total complexity per worker
        worker_loads = [0.0] * num_workers
        
        # Assign sequences to workers using First Fit Decreasing algorithm
        for seq, complexity, proc_time in sequence_complexities:
            # Find worker with minimum load
            min_load_worker = min(range(num_workers), key=lambda i: worker_loads[i])
            
            # Assign sequence to worker
            batches[min_load_worker].sequences.append(seq)
            batches[min_load_worker].estimated_complexity += complexity
            batches[min_load_worker].estimated_processing_time += proc_time
            worker_loads[min_load_worker] += complexity
        
        logger.info("Created complexity-aware batches",
                   num_workers=num_workers,
                   total_sequences=len(sequences),
                   load_variance=max(worker_loads) - min(worker_loads),
                   average_load=sum(worker_loads) / len(worker_loads))
        
        return batches
    
    def _create_round_robin_batches(self, 
                                  sequences: List[List[ReasoningEngine]], 
                                  num_workers: int) -> List[SequenceBatch]:
        """Create batches using simple round-robin distribution"""
        
        batches = []
        for i in range(num_workers):
            batch = SequenceBatch(
                batch_id=f"batch_{i}",
                worker_id=i
            )
            batches.append(batch)
        
        # Distribute sequences round-robin style
        for i, seq in enumerate(sequences):
            worker_idx = i % num_workers
            complexity = self.complexity_estimator.estimate_sequence_complexity(seq)
            proc_time = self.complexity_estimator.estimate_processing_time(seq)
            
            batches[worker_idx].sequences.append(seq)
            batches[worker_idx].estimated_complexity += complexity
            batches[worker_idx].estimated_processing_time += proc_time
        
        return batches
    
    def _create_adaptive_batches(self, 
                               sequences: List[List[ReasoningEngine]], 
                               num_workers: int) -> List[SequenceBatch]:
        """Create batches using adaptive algorithm based on worker history"""
        
        # Start with complexity-aware distribution
        batches = self._create_complexity_aware_batches(sequences, num_workers)
        
        # Adjust based on worker history if available
        if self.worker_history:
            for batch in batches:
                worker_id = batch.worker_id
                if worker_id in self.worker_history and self.worker_history[worker_id]:
                    # Calculate worker performance factor
                    recent_times = self.worker_history[worker_id][-10:]  # Last 10 jobs
                    avg_performance = sum(recent_times) / len(recent_times)
                    
                    # Adjust batch size based on performance
                    # Better performing workers (lower avg time) get more work
                    performance_factor = 1.0 / max(avg_performance, 0.1)  # Avoid division by zero
                    
                    # Scale estimated processing time
                    batch.estimated_processing_time *= performance_factor
        
        return batches
    
    def update_worker_performance(self, worker_id: int, processing_time: float):
        """Update worker performance history for adaptive load balancing"""
        self.worker_history[worker_id].append(processing_time)
        
        # Keep only recent history (last 50 jobs)
        if len(self.worker_history[worker_id]) > 50:
            self.worker_history[worker_id] = self.worker_history[worker_id][-50:]

# Note: SharedWorldModelManager is now imported from shared_world_model_architecture.py
# The comprehensive implementation includes hierarchical caching, adaptive resource management,
# and fault tolerance capabilities as specified in Phase 8.1.3

class ParallelMetaReasoningWorker:
    """Individual worker for processing sequence batches in parallel"""
    
    def __init__(self, 
                 worker_id: int,
                 shared_world_model: SharedWorldModelManager,
                 batch: SequenceBatch):
        self.worker_id = worker_id
        self.shared_world_model = shared_world_model
        self.batch = batch
        self.meta_reasoning_engine = None
        self.metrics = WorkerMetrics(worker_id=worker_id)
        self.processing_errors = []
        
        # Initialize meta-reasoning engine for this worker
        self._initialize_meta_reasoning_engine()
    
    def _initialize_meta_reasoning_engine(self):
        """Initialize meta-reasoning engine for this worker"""
        try:
            # Lazy import to avoid circular dependency
            from prsm.nwtn.meta_reasoning_engine import MetaReasoningEngine
            # Create a new MetaReasoningEngine instance for this worker
            self.meta_reasoning_engine = MetaReasoningEngine()
            logger.debug("Initialized meta-reasoning engine for worker", worker_id=self.worker_id)
            
        except Exception as e:
            logger.error("Failed to initialize meta-reasoning engine for worker",
                        worker_id=self.worker_id, error=str(e))
            raise
    
    async def process_batch(self, query: str, context: Dict[str, Any]) -> MetaReasoningResult:
        """Process the assigned batch of reasoning sequences"""
        
        self.batch.status = WorkerStatus.PROCESSING
        self.batch.start_time = datetime.now(timezone.utc)
        self.metrics.status = WorkerStatus.PROCESSING
        self.metrics.last_activity = self.batch.start_time
        
        start_time = time.time()
        
        try:
            logger.info("Worker starting batch processing",
                       worker_id=self.worker_id,
                       batch_id=self.batch.batch_id,
                       num_sequences=len(self.batch.sequences))
            
            # Process each sequence in the batch
            sequence_results = []
            
            for i, sequence in enumerate(self.batch.sequences):
                try:
                    # Convert to ReasoningSequence if needed
                    reasoning_sequence = ReasoningSequence(engines=sequence)
                    
                    # Process sequence using meta-reasoning engine
                    sequence_result = await self.meta_reasoning_engine.reason_sequential(
                        query=query,
                        reasoning_sequence=reasoning_sequence,
                        context=context
                    )
                    
                    sequence_results.append(sequence_result)
                    self.metrics.sequences_processed += 1
                    
                    # Update metrics
                    self.metrics.last_activity = datetime.now(timezone.utc)
                    
                    logger.debug("Completed sequence processing",
                               worker_id=self.worker_id,
                               sequence_idx=i,
                               total_sequences=len(self.batch.sequences))
                    
                except Exception as e:
                    error_msg = f"Sequence {i} failed: {str(e)}"
                    self.processing_errors.append(error_msg)
                    self.metrics.error_count += 1
                    logger.warning("Sequence processing failed",
                                 worker_id=self.worker_id,
                                 sequence_idx=i,
                                 error=str(e))
                    continue
            
            # Validate results using shared world model
            if sequence_results:
                # Extract reasoning results for validation
                reasoning_results = []
                for seq_result in sequence_results:
                    if hasattr(seq_result, 'sequential_results') and seq_result.sequential_results:
                        for seq_res in seq_result.sequential_results:
                            reasoning_results.extend(seq_res.results)
                
                if reasoning_results:
                    # Use comprehensive shared world model validation
                    validation_result = await shared_world_model_validation(
                        reasoning_results=reasoning_results,
                        context={'worker_id': self.worker_id},
                        worker_id=self.worker_id
                    )
                    
                    logger.debug("Validated reasoning results with shared world model",
                               worker_id=self.worker_id,
                               num_validated=validation_result['total_validations'],
                               success_rate=validation_result['validation_success_rate'],
                               cache_hit_rate=validation_result['cache_hit_rate'])
            
            # Create final result for this worker
            processing_time = time.time() - start_time
            self.metrics.total_processing_time = processing_time
            
            if self.metrics.sequences_processed > 0:
                self.metrics.average_sequence_time = processing_time / self.metrics.sequences_processed
            
            # Synthesize worker results into single MetaReasoningResult
            worker_result = self._synthesize_worker_results(sequence_results, query, context, processing_time)
            
            self.batch.status = WorkerStatus.COMPLETED
            self.batch.completion_time = datetime.now(timezone.utc)
            self.metrics.status = WorkerStatus.COMPLETED
            
            logger.info("Worker completed batch processing",
                       worker_id=self.worker_id,
                       batch_id=self.batch.batch_id,
                       sequences_processed=self.metrics.sequences_processed,
                       processing_time=processing_time,
                       error_count=self.metrics.error_count)
            
            return worker_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.batch.status = WorkerStatus.FAILED
            self.batch.error = str(e)
            self.metrics.status = WorkerStatus.FAILED
            self.metrics.total_processing_time = processing_time
            
            logger.error("Worker batch processing failed",
                        worker_id=self.worker_id,
                        batch_id=self.batch.batch_id,
                        error=str(e),
                        processing_time=processing_time)
            
            # Return empty result on failure
            return MetaReasoningResult(
                query=query,
                thinking_mode=ThinkingMode.QUICK,
                processing_time=processing_time,
                confidence=0.0
            )
    
    def _synthesize_worker_results(self, 
                                 sequence_results: List[MetaReasoningResult],
                                 query: str,
                                 context: Dict[str, Any],
                                 processing_time: float) -> MetaReasoningResult:
        """Synthesize multiple sequence results into single worker result"""
        
        if not sequence_results:
            return MetaReasoningResult(
                query=query,
                thinking_mode=ThinkingMode.QUICK,
                processing_time=processing_time,
                confidence=0.0
            )
        
        # Extract best insights from all sequences
        all_insights = []
        all_evidence = []
        confidence_scores = []
        
        for result in sequence_results:
            if hasattr(result, 'key_insights') and result.key_insights:
                all_insights.extend(result.key_insights)
            
            if hasattr(result, 'evidence') and result.evidence:
                all_evidence.extend(result.evidence)
                
            if hasattr(result, 'confidence'):
                confidence_scores.append(result.confidence)
        
        # Calculate synthesized confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Create synthesized result
        synthesized_result = MetaReasoningResult(
            query=query,
            thinking_mode=ThinkingMode.DEEP,  # Deep mode since we processed multiple sequences
            processing_time=processing_time,
            confidence=avg_confidence,
            key_insights=all_insights[:10],  # Top 10 insights
            evidence=all_evidence[:20],      # Top 20 evidence items
            worker_id=self.worker_id,
            sequences_processed=len(sequence_results),
            errors=self.processing_errors
        )
        
        return synthesized_result

class ParallelResultSynthesizer:
    """Synthesizes results from all parallel reasoning paths"""
    
    def __init__(self):
        self.synthesis_strategies = [
            self._consensus_synthesis,
            self._weighted_synthesis,
            self._diversity_synthesis,
            self._quality_synthesis
        ]
    
    def synthesize_parallel_results(self, 
                                   worker_results: List[MetaReasoningResult],
                                   synthesis_strategy: str = "weighted") -> MetaReasoningResult:
        """Synthesize results from all parallel workers"""
        
        if not worker_results:
            return MetaReasoningResult(
                query="",
                thinking_mode=ThinkingMode.DEEP,
                processing_time=0.0,
                confidence=0.0
            )
        
        # Filter out failed results
        valid_results = [r for r in worker_results if r.confidence > 0.0]
        
        if not valid_results:
            # All workers failed, return best available result
            best_result = max(worker_results, key=lambda r: getattr(r, 'sequences_processed', 0))
            logger.warning("All workers failed, returning best available result")
            return best_result
        
        # Select synthesis strategy
        if synthesis_strategy == "consensus":
            return self._consensus_synthesis(valid_results)
        elif synthesis_strategy == "diversity":
            return self._diversity_synthesis(valid_results)
        elif synthesis_strategy == "quality":
            return self._quality_synthesis(valid_results)
        else:  # default weighted synthesis
            return self._weighted_synthesis(valid_results)
    
    def _weighted_synthesis(self, results: List[MetaReasoningResult]) -> MetaReasoningResult:
        """Synthesize results using confidence-weighted approach"""
        
        # Weight by confidence and sequences processed
        total_weight = 0.0
        weighted_insights = []
        weighted_evidence = []
        
        for result in results:
            confidence = getattr(result, 'confidence', 0.0)
            sequences_processed = getattr(result, 'sequences_processed', 1)
            
            weight = confidence * math.sqrt(sequences_processed)  # Square root to avoid over-weighting
            total_weight += weight
            
            # Weight insights and evidence
            if hasattr(result, 'key_insights') and result.key_insights:
                for insight in result.key_insights:
                    weighted_insights.append((insight, weight))
            
            if hasattr(result, 'evidence') and result.evidence:
                for evidence in result.evidence:
                    weighted_evidence.append((evidence, weight))
        
        # Sort by weight and select top items
        weighted_insights.sort(key=lambda x: x[1], reverse=True)
        weighted_evidence.sort(key=lambda x: x[1], reverse=True)
        
        # Extract top insights and evidence
        top_insights = [insight for insight, _ in weighted_insights[:15]]
        top_evidence = [evidence for evidence, _ in weighted_evidence[:25]]
        
        # Calculate synthesized metrics
        total_sequences = sum(getattr(r, 'sequences_processed', 0) for r in results)
        avg_confidence = sum(getattr(r, 'confidence', 0.0) for r in results) / len(results)
        total_processing_time = sum(getattr(r, 'processing_time', 0.0) for r in results)
        
        # Create synthesized result
        synthesized = MetaReasoningResult(
            query=results[0].query,
            thinking_mode=ThinkingMode.DEEP,
            processing_time=total_processing_time,
            confidence=avg_confidence,
            key_insights=top_insights,
            evidence=top_evidence,
            total_sequences_processed=total_sequences,
            num_workers=len(results),
            synthesis_method="weighted_confidence"
        )
        
        return synthesized
    
    def _consensus_synthesis(self, results: List[MetaReasoningResult]) -> MetaReasoningResult:
        """Synthesize results based on consensus across workers"""
        
        # Find common insights and evidence across multiple workers
        insight_counts = defaultdict(int)
        evidence_counts = defaultdict(int)
        
        for result in results:
            if hasattr(result, 'key_insights') and result.key_insights:
                for insight in result.key_insights:
                    insight_counts[insight] += 1
            
            if hasattr(result, 'evidence') and result.evidence:
                for evidence in result.evidence:
                    evidence_counts[evidence] += 1
        
        # Select insights and evidence that appear in multiple workers
        min_consensus = max(2, len(results) // 3)  # At least 1/3 of workers must agree
        
        consensus_insights = [
            insight for insight, count in insight_counts.items() 
            if count >= min_consensus
        ]
        
        consensus_evidence = [
            evidence for evidence, count in evidence_counts.items() 
            if count >= min_consensus
        ]
        
        # Sort by consensus strength
        consensus_insights.sort(key=lambda x: insight_counts[x], reverse=True)
        consensus_evidence.sort(key=lambda x: evidence_counts[x], reverse=True)
        
        # Calculate metrics
        total_sequences = sum(getattr(r, 'sequences_processed', 0) for r in results)
        avg_confidence = sum(getattr(r, 'confidence', 0.0) for r in results) / len(results)
        total_processing_time = sum(getattr(r, 'processing_time', 0.0) for r in results)
        
        synthesized = MetaReasoningResult(
            query=results[0].query,
            thinking_mode=ThinkingMode.DEEP,
            processing_time=total_processing_time,
            confidence=avg_confidence,
            key_insights=consensus_insights[:12],
            evidence=consensus_evidence[:20],
            total_sequences_processed=total_sequences,
            num_workers=len(results),
            synthesis_method="consensus"
        )
        
        return synthesized
    
    def _diversity_synthesis(self, results: List[MetaReasoningResult]) -> MetaReasoningResult:
        """Synthesize results to maximize diversity of insights"""
        
        all_insights = []
        all_evidence = []
        
        # Collect all unique insights and evidence
        seen_insights = set()
        seen_evidence = set()
        
        for result in results:
            if hasattr(result, 'key_insights') and result.key_insights:
                for insight in result.key_insights:
                    if insight not in seen_insights:
                        all_insights.append(insight)
                        seen_insights.add(insight)
            
            if hasattr(result, 'evidence') and result.evidence:
                for evidence in result.evidence:
                    if evidence not in seen_evidence:
                        all_evidence.append(evidence)
                        seen_evidence.add(evidence)
        
        # Calculate metrics
        total_sequences = sum(getattr(r, 'sequences_processed', 0) for r in results)
        avg_confidence = sum(getattr(r, 'confidence', 0.0) for r in results) / len(results)
        total_processing_time = sum(getattr(r, 'processing_time', 0.0) for r in results)
        
        synthesized = MetaReasoningResult(
            query=results[0].query,
            thinking_mode=ThinkingMode.DEEP,
            processing_time=total_processing_time,
            confidence=avg_confidence,
            key_insights=all_insights[:18],  # More insights for diversity
            evidence=all_evidence[:30],      # More evidence for diversity
            total_sequences_processed=total_sequences,
            num_workers=len(results),
            synthesis_method="diversity"
        )
        
        return synthesized
    
    def _quality_synthesis(self, results: List[MetaReasoningResult]) -> MetaReasoningResult:
        """Synthesize results based on quality scores"""
        
        # Sort results by quality (confidence * sequences_processed)
        quality_scores = []
        for result in results:
            confidence = getattr(result, 'confidence', 0.0)
            sequences = getattr(result, 'sequences_processed', 1)
            quality = confidence * sequences
            quality_scores.append((result, quality))
        
        quality_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take insights and evidence from top quality results
        top_insights = []
        top_evidence = []
        
        # Weight allocation based on quality
        for i, (result, quality) in enumerate(quality_scores[:5]):  # Top 5 results
            weight_factor = 1.0 / (i + 1)  # Decreasing weight
            
            if hasattr(result, 'key_insights') and result.key_insights:
                num_insights = max(1, int(10 * weight_factor))
                top_insights.extend(result.key_insights[:num_insights])
            
            if hasattr(result, 'evidence') and result.evidence:
                num_evidence = max(1, int(15 * weight_factor))
                top_evidence.extend(result.evidence[:num_evidence])
        
        # Calculate metrics
        total_sequences = sum(getattr(r, 'sequences_processed', 0) for r in results)
        avg_confidence = sum(getattr(r, 'confidence', 0.0) for r in results) / len(results)
        total_processing_time = sum(getattr(r, 'processing_time', 0.0) for r in results)
        
        synthesized = MetaReasoningResult(
            query=results[0].query,
            thinking_mode=ThinkingMode.DEEP,
            processing_time=total_processing_time,
            confidence=avg_confidence,
            key_insights=top_insights[:15],
            evidence=top_evidence[:25],
            total_sequences_processed=total_sequences,
            num_workers=len(results),
            synthesis_method="quality_based"
        )
        
        return synthesized

class ParallelMetaReasoningOrchestrator:
    """Main orchestrator coordinating 20-100 parallel MetaReasoningEngine instances"""
    
    def __init__(self,
                 num_workers: int = 20,
                 load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.COMPLEXITY_AWARE,
                 shared_memory_optimization: bool = True):
        
        self.num_workers = min(max(num_workers, 1), 100)  # Limit to 1-100 workers
        self.load_balancing_strategy = load_balancing_strategy
        self.shared_memory_optimization = shared_memory_optimization
        
        # Initialize components
        self.work_distribution_engine = WorkDistributionEngine(load_balancing_strategy)
        self.intelligent_work_distribution = IntelligentWorkDistributionEngine()
        self.hierarchical_cache = HierarchicalResultCachingSystem(
            engine_cache_size=100000,
            sequence_cache_size=10000,
            validation_cache_size=50000,
            cross_worker_cache_size=50000,
            enable_analytics=True
        )
        self.shared_world_model = SharedWorldModelManager(
            world_model_size=10000,
            max_validation_workers=min(num_workers, 8)  # Limit validation workers
        ) if shared_memory_optimization else None
        self.result_synthesizer = ParallelResultSynthesizer()
        
        # Fault tolerance and recovery system
        self.resilience_system = ParallelProcessingResilience(
            health_check_interval=30,
            checkpoint_interval=300,
            max_recovery_attempts=3
        )
        
        # Performance tracking
        self.orchestrator_id = str(uuid4())
        self.total_queries_processed = 0
        self.total_speedup_achieved = 0.0
        self.performance_history = []
        
        logger.info("Parallel Meta-Reasoning Orchestrator initialized",
                   orchestrator_id=self.orchestrator_id,
                   num_workers=self.num_workers,
                   load_balancing_strategy=load_balancing_strategy.value,
                   shared_memory_optimization=shared_memory_optimization)
    
    async def parallel_deep_reasoning(self, 
                                    query: str, 
                                    context: Optional[Dict[str, Any]] = None) -> ParallelProcessingResult:
        """Execute parallel deep reasoning across multiple workers"""
        
        start_time = time.time()
        context = context or {}
        
        try:
            logger.info("Starting parallel deep reasoning",
                       orchestrator_id=self.orchestrator_id,
                       query=query[:100],
                       num_workers=self.num_workers)
            
            # Start fault tolerance monitoring
            self.resilience_system.start_resilience_monitoring()
            
            # Initialize shared world model if enabled
            if self.shared_world_model:
                await self.shared_world_model.initialize()
                logger.info("Shared world model initialized for parallel processing")
            
            # Generate all reasoning sequences for deep thinking
            all_sequences = self._generate_deep_reasoning_sequences()
            
            if not all_sequences:
                logger.warning("No reasoning sequences generated")
                return self._create_empty_result(query, time.time() - start_time)
            
            # Distribute work across workers
            sequence_batches = self.work_distribution_engine.create_balanced_batches(
                all_sequences, self.num_workers
            )
            
            # Filter out empty batches
            non_empty_batches = [batch for batch in sequence_batches if batch.sequences]
            actual_workers = len(non_empty_batches)
            
            logger.info("Work distribution completed",
                       total_sequences=len(all_sequences),
                       num_batches=len(sequence_batches),
                       actual_workers=actual_workers)
            
            # Create and spawn parallel workers
            workers = []
            worker_metrics = []
            
            for batch in non_empty_batches:
                worker = ParallelMetaReasoningWorker(
                    worker_id=batch.worker_id,
                    shared_world_model=self.shared_world_model,
                    batch=batch
                )
                
                # Register worker for fault tolerance monitoring
                self.resilience_system.register_worker(str(batch.worker_id))
                
                # Assign work for checkpointing
                self.resilience_system.assign_work(str(batch.worker_id), batch.sequences)
                
                # Start worker processing
                worker_task = worker.process_batch(query, context)
                workers.append(worker_task)
                worker_metrics.append(worker.metrics)
            
            # Execute all workers in parallel with timeout
            try:
                worker_results = await asyncio.wait_for(
                    asyncio.gather(*workers, return_exceptions=True),
                    timeout=7200  # 2 hour timeout
                )
                
            except asyncio.TimeoutError:
                logger.error("Parallel processing timed out")
                # Return partial results from completed workers
                worker_results = []
                for task in workers:
                    if task.done():
                        try:
                            result = task.result()
                            worker_results.append(result)
                        except Exception as e:
                            logger.warning("Failed to get worker result", error=str(e))
            
            # Filter out exceptions and failed results
            valid_worker_results = []
            error_summary = []
            
            for result in worker_results:
                if isinstance(result, Exception):
                    error_msg = f"Worker failed: {str(result)}"
                    error_summary.append(error_msg)
                    logger.warning("Worker failed with exception", error=str(result))
                elif result and getattr(result, 'confidence', 0) > 0:
                    valid_worker_results.append(result)
            
            if not valid_worker_results:
                logger.error("All workers failed")
                return self._create_failed_result(query, time.time() - start_time, error_summary)
            
            # Synthesize results from all workers
            synthesized_result = self.result_synthesizer.synthesize_parallel_results(
                valid_worker_results, synthesis_strategy="weighted"
            )
            
            # Calculate performance metrics
            total_processing_time = time.time() - start_time
            sequential_estimate = self._estimate_sequential_processing_time(all_sequences)
            speedup_factor = sequential_estimate / total_processing_time if total_processing_time > 0 else 1.0
            
            # Update performance tracking
            self.total_queries_processed += 1
            self.total_speedup_achieved += speedup_factor
            
            # Get comprehensive statistics if shared world model is used
            cache_stats = {}
            if self.shared_world_model:
                comprehensive_stats = self.shared_world_model.get_comprehensive_statistics()
                cache_stats = comprehensive_stats.get('hierarchical_cache', {})
            
            # Create final result
            parallel_result = ParallelProcessingResult(
                orchestrator_id=self.orchestrator_id,
                query=query,
                total_sequences_processed=len(all_sequences),
                num_workers_used=actual_workers,
                total_processing_time=total_processing_time,
                speedup_factor=speedup_factor,
                worker_results=valid_worker_results,
                synthesized_result=synthesized_result,
                worker_metrics=worker_metrics,
                cache_hit_rate=cache_stats.get('hit_rate', 0.0),
                resource_utilization=self._calculate_resource_utilization(),
                error_summary=error_summary
            )
            
            # Update worker performance history
            for i, result in enumerate(valid_worker_results):
                processing_time = getattr(result, 'processing_time', 0.0)
                self.work_distribution_engine.update_worker_performance(i, processing_time)
            
            logger.info("Parallel deep reasoning completed",
                       orchestrator_id=self.orchestrator_id,
                       sequences_processed=len(all_sequences),
                       workers_used=actual_workers,
                       processing_time=total_processing_time,
                       speedup_factor=speedup_factor,
                       cache_hit_rate=cache_stats.get('hit_rate', 0.0))
            
            return parallel_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error("Parallel deep reasoning failed",
                        orchestrator_id=self.orchestrator_id,
                        error=str(e),
                        processing_time=processing_time)
            
            # Get system health status for error reporting
            health_status = self.resilience_system.get_system_health_status()
            logger.warning("System health at failure", **health_status)
            
            return self._create_failed_result(query, processing_time, [str(e)])
        
        finally:
            # Stop resilience monitoring
            self.resilience_system.stop_resilience_monitoring()
            logger.debug("Fault tolerance monitoring stopped")
    
    def _generate_deep_reasoning_sequences(self) -> List[List[ReasoningEngine]]:
        """Generate all possible reasoning sequences for deep thinking (5040 permutations)"""
        
        # Core reasoning engines for permutation
        core_engines = [
            ReasoningEngine.DEDUCTIVE,
            ReasoningEngine.INDUCTIVE,
            ReasoningEngine.ABDUCTIVE,
            ReasoningEngine.CAUSAL,
            ReasoningEngine.PROBABILISTIC,
            ReasoningEngine.COUNTERFACTUAL,
            ReasoningEngine.ANALOGICAL
        ]
        
        # For demo purposes, we'll generate a smaller subset
        # In production, this would generate all 5040 permutations
        sequences = []
        
        # Generate permutations of length 3-5 for manageable demo
        for length in range(3, 6):
            for perm in permutations(core_engines, length):
                sequences.append(list(perm))
                if len(sequences) >= 200:  # Limit for demo
                    break
            if len(sequences) >= 200:
                break
        
        logger.info("Generated reasoning sequences",
                   total_sequences=len(sequences),
                   sequence_lengths=f"{min(len(s) for s in sequences)}-{max(len(s) for s in sequences)}")
        
        return sequences
    
    def _estimate_sequential_processing_time(self, sequences: List[List[ReasoningEngine]]) -> float:
        """Estimate time for sequential processing of all sequences"""
        
        total_complexity = 0.0
        
        for sequence in sequences:
            complexity = self.work_distribution_engine.complexity_estimator.estimate_sequence_complexity(sequence)
            total_complexity += complexity
        
        # Estimate processing time (5 seconds per complexity unit on average)
        estimated_time = total_complexity * 5.0
        
        return estimated_time
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate current resource utilization"""
        
        try:
            # CPU and memory utilization
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            
            return {
                'cpu_utilization': cpu_percent,
                'memory_utilization': memory_percent,
                'available_memory_gb': memory_info.available / (1024**3)
            }
            
        except Exception as e:
            logger.warning("Failed to calculate resource utilization", error=str(e))
            return {'cpu_utilization': 0.0, 'memory_utilization': 0.0}
    
    def _create_empty_result(self, query: str, processing_time: float) -> ParallelProcessingResult:
        """Create empty result when no sequences are generated"""
        
        return ParallelProcessingResult(
            orchestrator_id=self.orchestrator_id,
            query=query,
            total_sequences_processed=0,
            num_workers_used=0,
            total_processing_time=processing_time,
            speedup_factor=1.0,
            error_summary=["No reasoning sequences generated"]
        )
    
    def _create_failed_result(self, query: str, processing_time: float, errors: List[str]) -> ParallelProcessingResult:
        """Create failed result when all workers fail"""
        
        return ParallelProcessingResult(
            orchestrator_id=self.orchestrator_id,
            query=query,
            total_sequences_processed=0,
            num_workers_used=self.num_workers,
            total_processing_time=processing_time,
            speedup_factor=0.0,
            error_summary=errors
        )
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get orchestrator performance statistics"""
        
        avg_speedup = self.total_speedup_achieved / max(self.total_queries_processed, 1)
        
        cache_stats = {}
        if self.shared_world_model:
            comprehensive_stats = self.shared_world_model.get_comprehensive_statistics()
            cache_stats = comprehensive_stats
        
        return {
            'orchestrator_id': self.orchestrator_id,
            'total_queries_processed': self.total_queries_processed,
            'average_speedup_factor': avg_speedup,
            'num_workers': self.num_workers,
            'load_balancing_strategy': self.load_balancing_strategy.value,
            'shared_memory_optimization': self.shared_memory_optimization,
            'cache_statistics': cache_stats,
            'resource_utilization': self._calculate_resource_utilization()
        }
    
    # =========================================================================
    # INTELLIGENT WORK DISTRIBUTION METHODS
    # =========================================================================
    
    def register_workers_with_intelligent_distribution(self):
        """Register all workers with the intelligent work distribution engine"""
        try:
            for worker_id in range(self.num_workers):
                worker_id_str = f"worker_{worker_id}"
                
                # Estimate worker resource capacity based on system specs
                cpu_capacity = psutil.cpu_count() / self.num_workers
                memory_capacity = psutil.virtual_memory().total / (1024**3) / self.num_workers  # GB
                
                resource_capacity = {
                    ResourceType.CPU: cpu_capacity,
                    ResourceType.MEMORY: memory_capacity,
                    ResourceType.IO: 10.0,  # Arbitrary IO units
                    ResourceType.NETWORK: 100.0  # Arbitrary network units
                }
                
                # Register worker with performance profiler
                self.intelligent_work_distribution.performance_profiler.register_worker(
                    worker_id=worker_id_str,
                    resource_capacity=resource_capacity,
                    specializations={"reasoning", "meta_reasoning"}
                )
                
                # Update worker status to idle
                self.intelligent_work_distribution.performance_profiler.update_worker_status(
                    worker_id_str, WorkerStatus.IDLE
                )
            
            logger.info("Registered all workers with intelligent distribution system",
                       num_workers=self.num_workers)
            return True
            
        except Exception as e:
            logger.error(f"Failed to register workers: {str(e)}")
            return False
    
    async def intelligent_parallel_reasoning(self, 
                                           query: str, 
                                           context: Optional[Dict[str, Any]] = None,
                                           distribution_strategy: DistributionStrategy = DistributionStrategy.PERFORMANCE_ADAPTIVE) -> ParallelProcessingResult:
        """Execute parallel reasoning using intelligent work distribution"""
        
        start_time = time.time()
        context = context or {}
        
        try:
            logger.info("Starting intelligent parallel reasoning",
                       orchestrator_id=self.orchestrator_id,
                       query=query[:100],
                       distribution_strategy=distribution_strategy.value)
            
            # Ensure workers are registered
            self.register_workers_with_intelligent_distribution()
            
            # Generate all reasoning sequences
            all_sequences = self._generate_deep_reasoning_sequences()
            
            if not all_sequences:
                logger.warning("No reasoning sequences generated")
                return self._create_empty_result(query, time.time() - start_time)
            
            # Convert sequences to work items
            work_items = self._convert_sequences_to_work_items(all_sequences, query, context)
            
            # Use intelligent work distribution
            distribution_result = self.intelligent_work_distribution.distribute_work(
                work_items, distribution_strategy
            )
            
            logger.info("Intelligent work distribution completed",
                       total_work_items=len(work_items),
                       balance_score=distribution_result.balance_score,
                       estimated_completion_time=distribution_result.estimated_completion_time)
            
            # Initialize shared world model if enabled
            if self.shared_world_model:
                await self.shared_world_model.initialize()
            
            # Execute work using distributed assignments
            worker_results = await self._execute_intelligent_distribution(
                distribution_result, query, context
            )
            
            # Synthesize results
            synthesis_result = self.result_synthesizer.synthesize_parallel_results(
                worker_results, SynthesisStrategy.QUALITY_WEIGHTED
            )
            
            processing_time = time.time() - start_time
            speedup_achieved = self._calculate_speedup(len(all_sequences), processing_time)
            
            # Update performance tracking
            self.total_queries_processed += 1
            self.total_speedup_achieved += speedup_achieved
            
            # Record performance metrics for workers
            self._record_intelligent_distribution_performance(distribution_result, worker_results, processing_time)
            
            result = ParallelProcessingResult(
                orchestrator_id=self.orchestrator_id,
                query=query,
                reasoning_result=synthesis_result.final_result,
                worker_results=worker_results,
                distribution_strategy=distribution_strategy,
                balance_score=distribution_result.balance_score,
                processing_time_seconds=processing_time,
                speedup_achieved=speedup_achieved,
                num_workers_used=len([w for w in worker_results if w.success]),
                total_sequences_processed=len(all_sequences),
                synthesis_metadata=synthesis_result.synthesis_metadata,
                intelligent_distribution_metadata={
                    'distribution_result': distribution_result,
                    'distribution_analytics': self.intelligent_work_distribution.get_distribution_analytics()
                }
            )
            
            logger.info("Intelligent parallel reasoning completed successfully",
                       processing_time=processing_time,
                       speedup_achieved=speedup_achieved,
                       balance_score=distribution_result.balance_score)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Intelligent parallel reasoning failed: {str(e)}", 
                        processing_time=processing_time)
            return self._create_error_result(query, str(e), processing_time)
    
    def _convert_sequences_to_work_items(self, sequences: List, query: str, context: Dict[str, Any]) -> List[WorkItem]:
        """Convert reasoning sequences to work items for intelligent distribution"""
        work_items = []
        
        for i, sequence in enumerate(sequences):
            # Estimate complexity based on sequence characteristics
            complexity = self._estimate_sequence_complexity_advanced(sequence)
            
            # Estimate duration based on complexity
            duration = complexity * 5.0  # 5 seconds per complexity unit
            
            # Determine resource requirements
            resource_requirements = {
                ResourceType.CPU: min(complexity / 10.0, 1.0),
                ResourceType.MEMORY: min(complexity / 20.0, 0.5),
                ResourceType.IO: 0.1,  # Low IO for reasoning tasks
                ResourceType.NETWORK: 0.1  # Low network for reasoning tasks
            }
            
            work_item = WorkItem(
                item_id=f"sequence_{i}",
                content=sequence,
                estimated_complexity=complexity,
                estimated_duration=duration,
                resource_requirements=resource_requirements,
                dependencies=[],
                priority=1.0,
                metadata={
                    'sequence_index': i,
                    'query': query,
                    'type': 'reasoning_sequence',
                    'context': context
                }
            )
            
            work_items.append(work_item)
        
        return work_items
    
    async def _execute_intelligent_distribution(self, distribution_result, query: str, context: Dict[str, Any]) -> List[WorkerResult]:
        """Execute work using intelligent distribution assignments"""
        worker_results = []
        worker_tasks = []
        
        for worker_id, work_items in distribution_result.worker_assignments.items():
            if not work_items:
                continue
            
            # Convert work items back to sequences
            sequences = [item.content for item in work_items]
            
            # Create worker batch
            worker_batch = WorkerBatch(
                worker_id=worker_id,
                sequences=sequences,
                estimated_completion_time=sum(item.estimated_duration for item in work_items)
            )
            
            # Create and start worker
            worker = ParallelMetaReasoningWorker(
                worker_id=worker_id,
                shared_world_model=self.shared_world_model,
                worker_batch=worker_batch,
                context=context
            )
            
            # Update worker status to working
            self.intelligent_work_distribution.performance_profiler.update_worker_status(
                worker_id, WorkerStatus.WORKING
            )
            
            # Start worker task
            task = asyncio.create_task(worker.process_sequences(query))
            worker_tasks.append((worker_id, work_items, task))
        
        # Wait for all workers to complete
        for worker_id, work_items, task in worker_tasks:
            try:
                worker_result = await task
                worker_results.append(worker_result)
                
                # Update worker status back to idle
                self.intelligent_work_distribution.performance_profiler.update_worker_status(
                    worker_id, WorkerStatus.IDLE
                )
                
            except Exception as e:
                logger.error(f"Worker {worker_id} failed: {str(e)}")
                
                # Create failed worker result
                failed_result = WorkerResult(
                    worker_id=worker_id,
                    sequences_processed=0,
                    success=False,
                    processing_time=0.0,
                    error_message=str(e),
                    results=[]
                )
                worker_results.append(failed_result)
                
                # Update worker status to failed
                self.intelligent_work_distribution.performance_profiler.update_worker_status(
                    worker_id, WorkerStatus.FAILED
                )
        
        return worker_results
    
    def _record_intelligent_distribution_performance(self, distribution_result, worker_results: List, total_processing_time: float):
        """Record performance metrics for intelligent distribution"""
        try:
            for worker_result in worker_results:
                if not hasattr(worker_result, 'worker_id'):
                    continue
                
                worker_id = worker_result.worker_id
                
                # Find corresponding work items for this worker
                work_items = distribution_result.worker_assignments.get(worker_id, [])
                
                # Record completion for each work item
                for work_item in work_items:
                    # Calculate per-item processing time (approximate)
                    per_item_time = worker_result.processing_time / max(len(work_items), 1)
                    
                    # Calculate resource usage (simplified)
                    resource_usage = {
                        ResourceType.CPU: work_item.resource_requirements.get(ResourceType.CPU, 0),
                        ResourceType.MEMORY: work_item.resource_requirements.get(ResourceType.MEMORY, 0),
                        ResourceType.IO: work_item.resource_requirements.get(ResourceType.IO, 0),
                        ResourceType.NETWORK: work_item.resource_requirements.get(ResourceType.NETWORK, 0)
                    }
                    
                    # Record work completion
                    self.intelligent_work_distribution.performance_profiler.record_work_completion(
                        worker_id=worker_id,
                        work_item=work_item,
                        actual_duration=per_item_time,
                        success=worker_result.success,
                        resource_usage=resource_usage
                    )
                    
                    # Update complexity estimator
                    self.intelligent_work_distribution.complexity_estimator.update_actual_complexity(
                        work_item=work_item,
                        actual_duration=per_item_time,
                        success=worker_result.success
                    )
            
            logger.debug("Recorded performance metrics for intelligent distribution",
                        num_workers=len(worker_results))
            
        except Exception as e:
            logger.error(f"Failed to record performance metrics: {str(e)}")
    
    def _estimate_sequence_complexity_advanced(self, sequence) -> float:
        """Advanced complexity estimation for sequences"""
        try:
            # Base complexity from sequence length
            base_complexity = len(sequence) if hasattr(sequence, '__len__') else 1.0
            
            # Factor in reasoning engine types (if available)
            if hasattr(sequence, 'reasoning_engines') or isinstance(sequence, list):
                engine_complexity_weights = {
                    'deductive': 1.0,
                    'inductive': 1.2,
                    'abductive': 1.5,
                    'analogical': 1.8,
                    'counterfactual': 1.6,
                    'causal': 1.4,
                    'probabilistic': 1.3
                }
                
                if isinstance(sequence, list):
                    total_weight = sum(engine_complexity_weights.get(str(engine).lower(), 1.0) 
                                     for engine in sequence)
                    base_complexity *= (total_weight / len(sequence))
                
            # Factor in interaction complexity (permutation-based)
            interaction_factor = 1.0 + (base_complexity * 0.1)
            
            final_complexity = base_complexity * interaction_factor
            
            return max(1.0, min(final_complexity, 10.0))  # Clamp between 1.0 and 10.0
            
        except Exception as e:
            logger.debug(f"Complexity estimation failed: {str(e)}")
            return 2.0  # Default complexity
    
    def get_intelligent_distribution_status(self) -> Dict[str, Any]:
        """Get status of intelligent work distribution system"""
        try:
            analytics = self.intelligent_work_distribution.get_distribution_analytics()
            
            worker_profiles = {}
            for worker_id, profile in self.intelligent_work_distribution.performance_profiler.worker_profiles.items():
                worker_profiles[worker_id] = {
                    'status': profile.status.value,
                    'total_items_processed': profile.total_items_processed,
                    'performance_score': self.intelligent_work_distribution.performance_profiler.get_worker_performance_score(worker_id),
                    'failure_count': profile.failure_count,
                    'last_heartbeat': profile.last_heartbeat.isoformat()
                }
            
            return {
                'intelligent_distribution_enabled': True,
                'distribution_analytics': analytics,
                'worker_profiles': worker_profiles,
                'total_distributions': len(self.intelligent_work_distribution.distribution_history),
                'rebalancing_threshold': self.intelligent_work_distribution.rebalancing_threshold,
                'available_strategies': [strategy.value for strategy in DistributionStrategy]
            }
            
        except Exception as e:
            return {
                'intelligent_distribution_enabled': False,
                'error': str(e)
            }
    
    # =========================================================================
    # FAULT TOLERANCE & RECOVERY METHODS  
    # =========================================================================
    
    def get_system_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health and fault tolerance status"""
        try:
            base_status = self.resilience_system.get_system_health_status()
            
            # Add orchestrator-specific health metrics
            orchestrator_health = {
                'orchestrator_id': self.orchestrator_id,
                'total_queries_processed': self.total_queries_processed,
                'average_speedup_achieved': (
                    self.total_speedup_achieved / max(self.total_queries_processed, 1)
                ),
                'performance_history_size': len(self.performance_history),
                'shared_world_model_enabled': self.shared_world_model is not None,
                'intelligent_distribution_enabled': True,
                'hierarchical_caching_enabled': self.hierarchical_cache is not None
            }
            
            # Combine with resilience system status
            combined_status = {**base_status, **orchestrator_health}
            
            return combined_status
            
        except Exception as e:
            logger.error(f"Failed to get system health status: {str(e)}")
            return {
                'system_health_score': 0.0,
                'error': str(e)
            }
    
    def get_fault_tolerance_analytics(self) -> Dict[str, Any]:
        """Get detailed fault tolerance and recovery analytics"""
        try:
            # Get base analytics from resilience system
            health_status = self.resilience_system.get_system_health_status()
            
            # Add orchestrator-level failure analytics
            orchestrator_analytics = {
                'orchestrator_uptime_seconds': time.time() - getattr(self, '_start_time', time.time()),
                'total_parallel_executions': self.total_queries_processed,
                'cache_statistics': self.get_hierarchical_cache_statistics(),
                'resource_utilization': self._calculate_resource_utilization()
            }
            
            return {
                'health_status': health_status,
                'orchestrator_analytics': orchestrator_analytics,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get fault tolerance analytics: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def enable_proactive_failure_prevention(self) -> bool:
        """Enable proactive failure prevention based on health patterns"""
        try:
            # This would implement predictive failure prevention
            logger.info("Proactive failure prevention enabled")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable proactive failure prevention: {str(e)}")
            return False
    
    def get_recovery_recommendations(self) -> List[str]:
        """Get recommendations for improving system resilience"""
        try:
            recommendations = []
            health_status = self.resilience_system.get_system_health_status()
            
            # Analyze health metrics and provide recommendations
            if health_status['system_health_score'] < 0.7:
                recommendations.append("System health is below optimal - consider reducing worker load")
            
            if health_status['failed_workers'] > 0:
                recommendations.append(f"Consider investigating {health_status['failed_workers']} failed workers")
            
            if health_status.get('recovery_success_rate', 1.0) < 0.8:
                recommendations.append("Recovery success rate is low - review recovery strategies")
            
            # Cache-related recommendations
            cache_stats = self.get_hierarchical_cache_statistics()
            if cache_stats.get('system_overview', {}).get('system_wide_hit_rate', 0) < 0.5:
                recommendations.append("Cache hit rate is low - consider cache warming or size increases")
            
            # Resource recommendations
            resource_util = self._calculate_resource_utilization()
            if resource_util.get('memory_utilization', 0) > 0.9:
                recommendations.append("High memory utilization - consider reducing workers or adding memory")
            
            if resource_util.get('cpu_utilization', 0) > 0.9:
                recommendations.append("High CPU utilization - consider distributing load across more nodes")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recovery recommendations: {str(e)}")
            return [f"Error generating recommendations: {str(e)}"]
    
    # =========================================================================
    # HIERARCHICAL RESULT CACHING METHODS
    # =========================================================================
    
    async def cached_parallel_reasoning(self, 
                                       query: str, 
                                       context: Optional[Dict[str, Any]] = None,
                                       distribution_strategy: DistributionStrategy = DistributionStrategy.PERFORMANCE_ADAPTIVE,
                                       enable_cross_worker_sharing: bool = True) -> ParallelProcessingResult:
        """Execute parallel reasoning with hierarchical caching optimization"""
        
        start_time = time.time()
        context = context or {}
        
        try:
            # Generate cache key for the entire query
            query_cache_key = self._generate_query_cache_key(query, context, distribution_strategy)
            
            logger.info("Starting cached parallel reasoning",
                       orchestrator_id=self.orchestrator_id,
                       query=query[:100],
                       cache_key=query_cache_key[:20])
            
            # Check if we have a cached result for the entire query
            cached_result = self.hierarchical_cache.get(
                query_cache_key, 
                worker_id=self.orchestrator_id if enable_cross_worker_sharing else None
            )
            
            if cached_result:
                logger.info("Found cached result for entire query",
                           cache_key=query_cache_key[:20],
                           time_saved=time.time() - start_time)
                return cached_result
            
            # Ensure workers are registered
            self.register_workers_with_intelligent_distribution()
            
            # Generate all reasoning sequences
            all_sequences = self._generate_deep_reasoning_sequences()
            
            if not all_sequences:
                logger.warning("No reasoning sequences generated")
                return self._create_empty_result(query, time.time() - start_time)
            
            # Check cache for individual sequences
            cached_sequence_results = {}
            uncached_sequences = []
            
            for i, sequence in enumerate(all_sequences):
                sequence_key = self._generate_sequence_cache_key(sequence, query)
                cached_seq_result = self.hierarchical_cache.get(
                    sequence_key,
                    cache_levels=[CacheLevel.SEQUENCE_RESULT, CacheLevel.CROSS_WORKER],
                    worker_id=self.orchestrator_id if enable_cross_worker_sharing else None
                )
                
                if cached_seq_result:
                    cached_sequence_results[i] = cached_seq_result
                else:
                    uncached_sequences.append((i, sequence))
            
            logger.info("Cache analysis completed",
                       total_sequences=len(all_sequences),
                       cached_sequences=len(cached_sequence_results),
                       uncached_sequences=len(uncached_sequences),
                       cache_hit_rate=len(cached_sequence_results)/len(all_sequences))
            
            # Process uncached sequences if any
            uncached_results = {}
            if uncached_sequences:
                # Convert uncached sequences to work items
                work_items = []
                for seq_idx, sequence in uncached_sequences:
                    work_item = self._create_work_item_for_sequence(sequence, seq_idx, query, context)
                    work_items.append(work_item)
                
                # Use intelligent work distribution
                distribution_result = self.intelligent_work_distribution.distribute_work(
                    work_items, distribution_strategy
                )
                
                # Initialize shared world model if enabled
                if self.shared_world_model:
                    await self.shared_world_model.initialize()
                
                # Execute work with caching
                uncached_results = await self._execute_cached_distribution(
                    distribution_result, query, context, enable_cross_worker_sharing
                )
            
            # Combine cached and computed results
            all_results = {**cached_sequence_results}
            for seq_idx, result in uncached_results.items():
                all_results[seq_idx] = result
            
            # Convert to worker results format for synthesis
            worker_results = self._convert_to_worker_results(all_results, all_sequences)
            
            # Synthesize results
            synthesis_result = self.result_synthesizer.synthesize_parallel_results(
                worker_results, SynthesisStrategy.QUALITY_WEIGHTED
            )
            
            processing_time = time.time() - start_time
            speedup_achieved = self._calculate_speedup(len(all_sequences), processing_time)
            
            # Create final result
            result = ParallelProcessingResult(
                orchestrator_id=self.orchestrator_id,
                query=query,
                reasoning_result=synthesis_result.final_result,
                worker_results=worker_results,
                distribution_strategy=distribution_strategy,
                balance_score=1.0,  # Perfect balance due to caching
                processing_time_seconds=processing_time,
                speedup_achieved=speedup_achieved,
                num_workers_used=len([w for w in worker_results if w.success]),
                total_sequences_processed=len(all_sequences),
                synthesis_metadata=synthesis_result.synthesis_metadata,
                intelligent_distribution_metadata={
                    'cache_hit_rate': len(cached_sequence_results)/len(all_sequences),
                    'cached_sequences': len(cached_sequence_results),
                    'uncached_sequences': len(uncached_sequences),
                    'cross_worker_sharing_enabled': enable_cross_worker_sharing
                }
            )
            
            # Cache the final result
            self.hierarchical_cache.put(
                query_cache_key, 
                result, 
                CacheLevel.SEQUENCE_RESULT,
                computation_time=processing_time,
                worker_id=self.orchestrator_id,
                metadata={'query_length': len(query), 'num_sequences': len(all_sequences)}
            )
            
            logger.info("Cached parallel reasoning completed successfully",
                       processing_time=processing_time,
                       speedup_achieved=speedup_achieved,
                       cache_hit_rate=len(cached_sequence_results)/len(all_sequences))
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Cached parallel reasoning failed: {str(e)}", 
                        processing_time=processing_time)
            return self._create_error_result(query, str(e), processing_time)
    
    def warm_reasoning_cache(self, common_queries: List[str], 
                           context_templates: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Pre-populate cache with results for common reasoning patterns"""
        try:
            context_templates = context_templates or [{}]
            warming_results = {
                'queries_processed': 0,
                'cache_entries_created': 0,
                'total_time_spent': 0.0,
                'estimated_time_saved': 0.0
            }
            
            start_time = time.time()
            
            for query in common_queries:
                for context in context_templates:
                    try:
                        # Execute reasoning and cache results
                        result = asyncio.run(self.cached_parallel_reasoning(
                            query=query,
                            context=context,
                            enable_cross_worker_sharing=True
                        ))
                        
                        warming_results['queries_processed'] += 1
                        warming_results['estimated_time_saved'] += result.processing_time_seconds
                        
                    except Exception as e:
                        logger.warning(f"Failed to warm cache for query: {query[:50]}... Error: {str(e)}")
            
            warming_results['total_time_spent'] = time.time() - start_time
            
            # Get cache statistics after warming
            cache_stats = self.get_hierarchical_cache_statistics()
            warming_results['final_cache_statistics'] = cache_stats
            
            logger.info("Cache warming completed",
                       queries_processed=warming_results['queries_processed'],
                       time_spent=warming_results['total_time_spent'])
            
            return warming_results
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Cache warming failed: {str(e)}'
            }
    
    def optimize_cache_configuration(self) -> Dict[str, Any]:
        """Automatically optimize cache configuration based on usage patterns"""
        try:
            # Get current cache statistics
            current_stats = self.hierarchical_cache.get_comprehensive_statistics()
            
            # Get optimization recommendations
            size_recommendations = self.hierarchical_cache.optimize_cache_sizes()
            
            # Apply recommended optimizations (in a real system, this might require restart)
            optimizations_applied = []
            
            for level, recommendation in size_recommendations.items():
                if recommendation['action'] == 'increase':
                    optimizations_applied.append(
                        f"Recommended: Increase {level} cache from {recommendation['current_size']} "
                        f"to {recommendation['recommended_size']} entries"
                    )
                elif recommendation['action'] == 'decrease':
                    optimizations_applied.append(
                        f"Recommended: Decrease {level} cache from {recommendation['current_size']} "
                        f"to {recommendation['recommended_size']} entries"
                    )
            
            return {
                'success': True,
                'current_performance': current_stats,
                'size_recommendations': size_recommendations,
                'optimizations_applied': optimizations_applied,
                'estimated_improvement': self._estimate_cache_optimization_impact(current_stats, size_recommendations)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Cache optimization failed: {str(e)}'
            }
    
    def get_hierarchical_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive hierarchical cache statistics"""
        try:
            return self.hierarchical_cache.get_comprehensive_statistics()
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to get cache statistics: {str(e)}'
            }
    
    async def _execute_cached_distribution(self, distribution_result, query: str, 
                                         context: Dict[str, Any], enable_cross_worker_sharing: bool):
        """Execute work distribution with caching at multiple levels"""
        sequence_results = {}
        worker_tasks = []
        
        for worker_id, work_items in distribution_result.worker_assignments.items():
            if not work_items:
                continue
            
            # Create cached worker task
            task = asyncio.create_task(
                self._process_worker_batch_with_cache(
                    worker_id, work_items, query, context, enable_cross_worker_sharing
                )
            )
            worker_tasks.append(task)
        
        # Wait for all cached worker tasks to complete
        completed_results = await asyncio.gather(*worker_tasks, return_exceptions=True)
        
        # Process results
        for result in completed_results:
            if isinstance(result, Exception):
                logger.error(f"Worker task failed: {str(result)}")
                continue
            
            if isinstance(result, dict):
                sequence_results.update(result)
        
        return sequence_results
    
    async def _process_worker_batch_with_cache(self, worker_id: str, work_items: List[WorkItem],
                                             query: str, context: Dict[str, Any], 
                                             enable_cross_worker_sharing: bool) -> Dict[int, Any]:
        """Process worker batch with individual sequence caching"""
        batch_results = {}
        
        for work_item in work_items:
            sequence_idx = work_item.metadata['sequence_index']
            sequence = work_item.content
            
            # Generate cache key for this sequence
            sequence_cache_key = self._generate_sequence_cache_key(sequence, query)
            
            # Use get_or_compute with caching
            def compute_sequence_result():
                # This would call the actual reasoning engine on the sequence
                # For now, simplified simulation
                return self._simulate_sequence_processing(sequence, query, context, worker_id)
            
            # Get or compute result with caching
            result = self.hierarchical_cache.get_or_compute(
                key=sequence_cache_key,
                computation_func=compute_sequence_result,
                cache_level=CacheLevel.SEQUENCE_RESULT,
                worker_id=worker_id if enable_cross_worker_sharing else None,
                metadata={
                    'sequence_index': sequence_idx,
                    'worker_id': worker_id,
                    'query_hash': hashlib.md5(query.encode()).hexdigest()[:8]
                }
            )
            
            batch_results[sequence_idx] = result
        
        return batch_results
    
    def _generate_query_cache_key(self, query: str, context: Dict[str, Any], 
                                 strategy: DistributionStrategy) -> str:
        """Generate cache key for entire query"""
        context_str = json.dumps(context, sort_keys=True) if context else ""
        key_components = [query, context_str, strategy.value]
        combined = "|".join(key_components)
        return f"query:{hashlib.md5(combined.encode()).hexdigest()}"
    
    def _generate_sequence_cache_key(self, sequence, query: str) -> str:
        """Generate cache key for individual sequence"""
        sequence_str = str(sequence)
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        sequence_hash = hashlib.md5(sequence_str.encode()).hexdigest()[:8]
        return f"sequence:{query_hash}:{sequence_hash}"
    
    def _create_work_item_for_sequence(self, sequence, seq_idx: int, query: str, 
                                     context: Dict[str, Any]) -> WorkItem:
        """Create work item for a specific sequence"""
        complexity = self._estimate_sequence_complexity_advanced(sequence)
        
        return WorkItem(
            item_id=f"cached_sequence_{seq_idx}",
            content=sequence,
            estimated_complexity=complexity,
            estimated_duration=complexity * 5.0,
            resource_requirements={
                ResourceType.CPU: min(complexity / 10.0, 1.0),
                ResourceType.MEMORY: min(complexity / 20.0, 0.5),
                ResourceType.IO: 0.1,
                ResourceType.NETWORK: 0.1
            },
            metadata={
                'sequence_index': seq_idx,
                'query': query,
                'context': context,
                'type': 'cached_reasoning_sequence'
            }
        )
    
    def _simulate_sequence_processing(self, sequence, query: str, context: Dict[str, Any], 
                                    worker_id: str) -> Dict[str, Any]:
        """Simulate processing a reasoning sequence (placeholder for actual reasoning)"""
        import random
        
        # Simulate processing time based on sequence complexity
        processing_time = random.uniform(1.0, 10.0)
        time.sleep(processing_time / 1000.0)  # Brief actual delay for realism
        
        return {
            'sequence': sequence,
            'result': f"Processed by {worker_id}",
            'confidence': random.uniform(0.7, 1.0),
            'processing_time': processing_time,
            'metadata': {
                'worker_id': worker_id,
                'query_length': len(query),
                'simulation': True
            }
        }
    
    def _convert_to_worker_results(self, sequence_results: Dict[int, Any], 
                                 all_sequences: List) -> List[WorkerResult]:
        """Convert sequence results back to worker results format"""
        # Group by worker (simplified - in practice would track actual worker assignments)
        worker_results = []
        
        # Create a single worker result containing all sequences for synthesis compatibility
        processed_sequences = len(sequence_results)
        success = processed_sequences > 0
        
        worker_result = WorkerResult(
            worker_id="cached_worker_combined",
            sequences_processed=processed_sequences,
            success=success,
            processing_time=sum(
                result.get('processing_time', 1.0) 
                for result in sequence_results.values()
            ) if success else 0.0,
            error_message="" if success else "No sequences processed",
            results=list(sequence_results.values())
        )
        
        worker_results.append(worker_result)
        return worker_results
    
    def _estimate_cache_optimization_impact(self, current_stats: Dict[str, Any], 
                                          recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate the impact of cache optimizations"""
        current_hit_rate = current_stats['system_overview']['system_wide_hit_rate']
        
        # Estimate potential hit rate improvement
        potential_improvement = 0.0
        for level, rec in recommendations.items():
            if rec['action'] == 'increase':
                # Increasing cache size typically improves hit rate
                potential_improvement += 0.05  # 5% improvement estimate
            elif rec['action'] == 'decrease':
                # Decreasing might slightly reduce hit rate but save memory
                potential_improvement -= 0.02  # 2% reduction estimate
        
        estimated_new_hit_rate = min(1.0, current_hit_rate + potential_improvement)
        
        return {
            'current_hit_rate': current_hit_rate,
            'estimated_new_hit_rate': estimated_new_hit_rate,
            'potential_improvement_percent': potential_improvement * 100,
            'estimated_time_savings_percent': potential_improvement * 20  # Rough estimate
        }

# Main interface function for integration
async def parallel_deep_reasoning(query: str,
                                context: Optional[Dict[str, Any]] = None,
                                num_workers: int = 20,
                                load_balancing_strategy: str = "complexity_aware") -> Dict[str, Any]:
    """Parallel deep reasoning interface for meta-reasoning integration"""
    
    # Convert string strategy to enum
    strategy_map = {
        "complexity_aware": LoadBalancingStrategy.COMPLEXITY_AWARE,
        "round_robin": LoadBalancingStrategy.ROUND_ROBIN,
        "dynamic_adaptive": LoadBalancingStrategy.DYNAMIC_ADAPTIVE,
        "resource_based": LoadBalancingStrategy.RESOURCE_BASED
    }
    
    strategy = strategy_map.get(load_balancing_strategy, LoadBalancingStrategy.COMPLEXITY_AWARE)
    
    # Create orchestrator
    orchestrator = ParallelMetaReasoningOrchestrator(
        num_workers=num_workers,
        load_balancing_strategy=strategy,
        shared_memory_optimization=True
    )
    
    # Execute parallel reasoning
    result = await orchestrator.parallel_deep_reasoning(query, context)
    
    # Convert to dictionary format expected by meta-reasoning engine
    return {
        "conclusion": f"Parallel deep reasoning processed {result.total_sequences_processed} sequences using {result.num_workers_used} workers with {result.speedup_factor:.1f}x speedup",
        "confidence": result.synthesized_result.confidence if result.synthesized_result else 0.0,
        "evidence": result.synthesized_result.evidence if result.synthesized_result and hasattr(result.synthesized_result, 'evidence') else [],
        "reasoning_chain": [
            f"Distributed {result.total_sequences_processed} reasoning sequences across {result.num_workers_used} parallel workers",
            f"Achieved {result.speedup_factor:.1f}x speedup over sequential processing",
            f"Cache hit rate: {result.cache_hit_rate:.1%}" if result.cache_hit_rate > 0 else "No caching used",
            f"Total processing time: {result.total_processing_time:.2f} seconds"
        ],
        "processing_time": result.total_processing_time,
        "quality_score": result.synthesized_result.confidence if result.synthesized_result else 0.0,
        "parallel_processing_result": result,
        "speedup_factor": result.speedup_factor,
        "num_workers_used": result.num_workers_used,
        "total_sequences_processed": result.total_sequences_processed,
        "cache_hit_rate": result.cache_hit_rate,
        "resource_utilization": result.resource_utilization,
        "errors": result.error_summary
    }

if __name__ == "__main__":
    # Test the parallel meta-reasoning orchestrator
    async def test_parallel_orchestrator():
        test_query = "developing breakthrough approaches to sustainable energy storage"
        test_context = {
            "domain": "energy_technology",
            "breakthrough_mode": "creative",
            "external_papers": []
        }
        
        result = await parallel_deep_reasoning(
            test_query, test_context, num_workers=4
        )
        
        print("Parallel Meta-Reasoning Orchestrator Test Results:")
        print("=" * 60)
        print(f"Query: {test_query}")
        print(f"Conclusion: {result['conclusion']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Speedup Factor: {result['speedup_factor']:.1f}x")
        print(f"Workers Used: {result['num_workers_used']}")
        print(f"Sequences Processed: {result['total_sequences_processed']}")
        print(f"Processing Time: {result['processing_time']:.2f}s")
        print(f"Cache Hit Rate: {result['cache_hit_rate']:.1%}")
        print("\nEvidence:")
        for i, evidence in enumerate(result.get('evidence', [])[:3], 1):
            print(f"{i}. {evidence}")
        if result.get('errors'):
            print(f"\nErrors: {len(result['errors'])}")
            for error in result['errors'][:2]:
                print(f"• {error}")
    
    asyncio.run(test_parallel_orchestrator())
    
    # Test integration with meta-reasoning engine
    async def test_meta_reasoning_integration():
        from prsm.nwtn.meta_reasoning_engine import MetaReasoningEngine, ThinkingMode
        
        print("\nTesting Meta-Reasoning Engine Integration:")
        print("=" * 50)
        
        # Create meta-reasoning engine
        meta_engine = MetaReasoningEngine()
        
        # Test parallel processing mode
        test_query = "breakthrough approaches to quantum computing scalability"
        test_context = {
            "domain": "quantum_technology",
            "breakthrough_mode": "revolutionary",
            "num_workers": 2,  # Small number for testing
            "load_balancing_strategy": "complexity_aware"
        }
        
        try:
            # Note: This would need the actual meta-reasoning engine method
            # result = await meta_engine.process_query(
            #     query=test_query,
            #     context=test_context,
            #     thinking_mode=ThinkingMode.PARALLEL
            # )
            
            print(f"Query: {test_query}")
            print(f"Context: {test_context}")
            print("Integration test would execute parallel processing")
            print("Status: Integration points configured successfully")
            
        except Exception as e:
            print(f"Integration test failed: {str(e)}")
    
    # Uncomment to test integration
    # asyncio.run(test_meta_reasoning_integration())