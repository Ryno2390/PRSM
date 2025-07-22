"""
NWTN Intelligent Work Distribution Engine

Advanced work distribution system with intelligent load balancing, dynamic resource
management, real-time performance monitoring, and adaptive optimization for massive
parallel processing scalability.

Enhances the existing WorkDistributionEngine with:
- Advanced bin-packing algorithms for optimal work distribution
- Dynamic load rebalancing during runtime
- Worker performance profiling and adaptation
- Resource usage optimization and bottleneck detection
- Predictive workload forecasting

Part of NWTN Phase 8: Parallel Processing & Scalability Architecture
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple, Union, Callable
from enum import Enum
from datetime import datetime, timedelta
import threading
import time
import statistics
import heapq
import numpy as np
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from uuid import uuid4

class DistributionStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    COMPLEXITY_AWARE = "complexity_aware"
    LOAD_BALANCED = "load_balanced"
    PERFORMANCE_ADAPTIVE = "performance_adaptive"
    RESOURCE_OPTIMAL = "resource_optimal"
    PREDICTIVE = "predictive"

class WorkerStatus(Enum):
    IDLE = "idle"
    WORKING = "working"
    OVERLOADED = "overloaded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"

class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    IO = "io"
    NETWORK = "network"

@dataclass
class WorkItem:
    item_id: str
    content: Any
    estimated_complexity: float
    estimated_duration: float
    resource_requirements: Dict[ResourceType, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    priority: float = 1.0
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkerProfile:
    worker_id: str
    status: WorkerStatus
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_capacity: Dict[ResourceType, float] = field(default_factory=dict)
    current_load: Dict[ResourceType, float] = field(default_factory=dict)
    processing_history: List[Dict[str, Any]] = field(default_factory=list)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    specializations: Set[str] = field(default_factory=set)
    failure_count: int = 0
    total_items_processed: int = 0

@dataclass
class DistributionResult:
    worker_assignments: Dict[str, List[WorkItem]]
    distribution_strategy: DistributionStrategy
    balance_score: float  # 0.0-1.0, higher is better
    estimated_completion_time: float
    resource_utilization: Dict[ResourceType, float]
    distribution_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkerPerformanceMetrics:
    worker_id: str
    throughput: float  # items per hour
    accuracy: float  # success rate
    resource_efficiency: Dict[ResourceType, float]
    response_time: float  # average seconds per item
    error_rate: float
    uptime_percentage: float
    last_updated: datetime = field(default_factory=datetime.now)

class AdvancedComplexityEstimator:
    """Enhanced complexity estimation with machine learning patterns"""
    
    def __init__(self):
        self.complexity_models = self._initialize_complexity_models()
        self.historical_data = defaultdict(list)
        self.lock = threading.RLock()
    
    def estimate_work_item_complexity(self, work_item: WorkItem) -> float:
        """Estimate complexity of a work item using multiple factors"""
        base_complexity = work_item.estimated_complexity
        
        # Factor in content size/type
        content_factor = self._estimate_content_complexity(work_item.content)
        
        # Factor in dependencies
        dependency_factor = 1.0 + (len(work_item.dependencies) * 0.1)
        
        # Factor in deadline pressure
        deadline_factor = self._calculate_deadline_pressure(work_item.deadline)
        
        # Factor in historical performance for similar items
        historical_factor = self._get_historical_complexity_factor(work_item)
        
        final_complexity = base_complexity * content_factor * dependency_factor * deadline_factor * historical_factor
        
        # Update historical data
        with self.lock:
            item_key = self._generate_item_signature(work_item)
            self.historical_data[item_key].append({
                'estimated_complexity': final_complexity,
                'timestamp': datetime.now(),
                'metadata': work_item.metadata
            })
        
        return final_complexity
    
    def update_actual_complexity(self, work_item: WorkItem, actual_duration: float, success: bool):
        """Update complexity model with actual performance data"""
        with self.lock:
            item_key = self._generate_item_signature(work_item)
            
            # Calculate actual complexity based on duration and success
            actual_complexity = actual_duration * (2.0 if not success else 1.0)
            
            self.historical_data[item_key].append({
                'actual_complexity': actual_complexity,
                'actual_duration': actual_duration,
                'success': success,
                'timestamp': datetime.now()
            })
            
            # Update complexity models
            self._update_complexity_models(work_item, actual_complexity)
    
    def _initialize_complexity_models(self) -> Dict[str, Any]:
        """Initialize complexity estimation models"""
        return {
            'reasoning_engine_weights': {
                'deductive': 1.0,
                'inductive': 1.2,
                'abductive': 1.5,
                'analogical': 1.8,
                'counterfactual': 1.6,
                'causal': 1.4,
                'probabilistic': 1.3
            },
            'content_type_multipliers': {
                'text': 1.0,
                'structured_data': 1.2,
                'multimodal': 1.8,
                'complex_reasoning': 2.0
            }
        }
    
    def _estimate_content_complexity(self, content: Any) -> float:
        """Estimate complexity based on content characteristics"""
        if isinstance(content, str):
            # Text complexity based on length and structure
            base_factor = min(len(content) / 1000.0, 5.0)  # Cap at 5x
            return 1.0 + base_factor * 0.2
        elif isinstance(content, dict):
            # Structured data complexity
            return 1.0 + min(len(content) / 50.0, 2.0) * 0.3
        elif isinstance(content, list):
            # List complexity based on size and nested structure
            nested_factor = 1.0
            for item in content[:10]:  # Check first 10 items
                if isinstance(item, (dict, list)):
                    nested_factor += 0.1
            return 1.0 + min(len(content) / 100.0, 3.0) * 0.25 * nested_factor
        
        return 1.0  # Default complexity factor
    
    def _calculate_deadline_pressure(self, deadline: Optional[datetime]) -> float:
        """Calculate urgency factor based on deadline"""
        if not deadline:
            return 1.0
        
        time_remaining = (deadline - datetime.now()).total_seconds()
        
        if time_remaining < 0:
            return 2.0  # Overdue, high pressure
        elif time_remaining < 3600:  # Less than 1 hour
            return 1.8
        elif time_remaining < 86400:  # Less than 1 day
            return 1.4
        elif time_remaining < 604800:  # Less than 1 week
            return 1.2
        else:
            return 1.0  # No deadline pressure
    
    def _get_historical_complexity_factor(self, work_item: WorkItem) -> float:
        """Get complexity adjustment based on historical data"""
        item_key = self._generate_item_signature(work_item)
        
        with self.lock:
            history = self.historical_data.get(item_key, [])
            
            if len(history) < 3:
                return 1.0  # Not enough data
            
            # Calculate average actual vs estimated complexity ratio
            ratios = []
            for record in history[-10:]:  # Use last 10 records
                if 'actual_complexity' in record and 'estimated_complexity' in record:
                    ratio = record['actual_complexity'] / max(record['estimated_complexity'], 0.1)
                    ratios.append(ratio)
            
            if ratios:
                return statistics.median(ratios)
            
            return 1.0
    
    def _generate_item_signature(self, work_item: WorkItem) -> str:
        """Generate signature for work item type categorization"""
        content_type = type(work_item.content).__name__
        metadata_keys = sorted(work_item.metadata.keys())
        return f"{content_type}_{':'.join(metadata_keys[:5])}"  # Limit to 5 keys
    
    def _update_complexity_models(self, work_item: WorkItem, actual_complexity: float):
        """Update internal complexity models based on actual performance"""
        # Simplified model update - in practice would use more sophisticated ML
        pass

class WorkerPerformanceProfiler:
    """Profile and monitor worker performance for optimal distribution"""
    
    def __init__(self):
        self.worker_profiles: Dict[str, WorkerProfile] = {}
        self.performance_history: Dict[str, List[WorkerPerformanceMetrics]] = defaultdict(list)
        self.performance_models: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.lock = threading.RLock()
    
    def register_worker(self, worker_id: str, resource_capacity: Dict[ResourceType, float],
                       specializations: Set[str] = None) -> bool:
        """Register a new worker with its capabilities"""
        with self.lock:
            self.worker_profiles[worker_id] = WorkerProfile(
                worker_id=worker_id,
                status=WorkerStatus.IDLE,
                resource_capacity=resource_capacity,
                current_load={resource: 0.0 for resource in ResourceType},
                specializations=specializations or set()
            )
            return True
    
    def update_worker_status(self, worker_id: str, status: WorkerStatus,
                           current_load: Dict[ResourceType, float] = None):
        """Update worker status and current resource usage"""
        with self.lock:
            if worker_id not in self.worker_profiles:
                return False
            
            profile = self.worker_profiles[worker_id]
            profile.status = status
            profile.last_heartbeat = datetime.now()
            
            if current_load:
                profile.current_load = current_load.copy()
            
            return True
    
    def record_work_completion(self, worker_id: str, work_item: WorkItem, 
                             actual_duration: float, success: bool,
                             resource_usage: Dict[ResourceType, float] = None):
        """Record work completion for performance profiling"""
        with self.lock:
            if worker_id not in self.worker_profiles:
                return
            
            profile = self.worker_profiles[worker_id]
            profile.total_items_processed += 1
            
            # Record processing history
            completion_record = {
                'work_item_id': work_item.item_id,
                'duration': actual_duration,
                'success': success,
                'estimated_duration': work_item.estimated_duration,
                'complexity': work_item.estimated_complexity,
                'resource_usage': resource_usage or {},
                'completed_at': datetime.now()
            }
            profile.processing_history.append(completion_record)
            
            # Limit history size
            if len(profile.processing_history) > 1000:
                profile.processing_history = profile.processing_history[-800:]
            
            # Update performance metrics
            self._update_performance_metrics(worker_id)
            
            # Update failure count
            if not success:
                profile.failure_count += 1
    
    def get_worker_performance_score(self, worker_id: str) -> float:
        """Calculate overall performance score for a worker (0.0-1.0)"""
        with self.lock:
            if worker_id not in self.worker_profiles:
                return 0.0
            
            profile = self.worker_profiles[worker_id]
            
            if profile.total_items_processed < 5:
                return 0.5  # Default for new workers
            
            # Calculate component scores
            throughput_score = min(profile.performance_metrics.get('throughput', 0) / 10.0, 1.0)
            accuracy_score = profile.performance_metrics.get('accuracy', 0)
            efficiency_score = statistics.mean(profile.performance_metrics.get('resource_efficiency', {}).values() or [0.5])
            uptime_score = profile.performance_metrics.get('uptime_percentage', 0) / 100.0
            
            # Weighted combination
            performance_score = (
                throughput_score * 0.3 +
                accuracy_score * 0.3 +
                efficiency_score * 0.2 +
                uptime_score * 0.2
            )
            
            return max(0.0, min(1.0, performance_score))
    
    def predict_worker_performance(self, worker_id: str, work_item: WorkItem) -> Dict[str, float]:
        """Predict how well a worker will perform on a specific work item"""
        with self.lock:
            if worker_id not in self.worker_profiles:
                return {'duration': work_item.estimated_duration, 'success_probability': 0.5}
            
            profile = self.worker_profiles[worker_id]
            
            # Find similar work items in history
            similar_items = []
            for record in profile.processing_history[-100:]:  # Last 100 items
                if abs(record['complexity'] - work_item.estimated_complexity) < 0.5:
                    similar_items.append(record)
            
            if not similar_items:
                return {'duration': work_item.estimated_duration, 'success_probability': 0.7}
            
            # Calculate predictions based on similar items
            durations = [item['duration'] for item in similar_items]
            successes = [item['success'] for item in similar_items]
            
            predicted_duration = statistics.median(durations)
            success_probability = sum(successes) / len(successes)
            
            return {
                'duration': predicted_duration,
                'success_probability': success_probability
            }
    
    def get_available_workers(self, resource_requirements: Dict[ResourceType, float] = None) -> List[str]:
        """Get list of workers available for new work"""
        available_workers = []
        
        with self.lock:
            for worker_id, profile in self.worker_profiles.items():
                if profile.status in [WorkerStatus.IDLE, WorkerStatus.WORKING]:
                    # Check if worker has sufficient remaining resources
                    if resource_requirements:
                        can_handle = True
                        for resource_type, required in resource_requirements.items():
                            capacity = profile.resource_capacity.get(resource_type, 0)
                            current_usage = profile.current_load.get(resource_type, 0)
                            available = capacity - current_usage
                            
                            if available < required:
                                can_handle = False
                                break
                        
                        if can_handle:
                            available_workers.append(worker_id)
                    else:
                        available_workers.append(worker_id)
        
        return available_workers
    
    def _update_performance_metrics(self, worker_id: str):
        """Update performance metrics for a worker"""
        profile = self.worker_profiles[worker_id]
        
        if len(profile.processing_history) < 3:
            return
        
        recent_history = profile.processing_history[-50:]  # Last 50 items
        
        # Calculate throughput (items per hour)
        if len(recent_history) >= 2:
            time_span = (recent_history[-1]['completed_at'] - recent_history[0]['completed_at']).total_seconds()
            if time_span > 0:
                throughput = len(recent_history) / (time_span / 3600.0)  # items per hour
                profile.performance_metrics['throughput'] = throughput
        
        # Calculate accuracy (success rate)
        successes = sum(1 for item in recent_history if item['success'])
        accuracy = successes / len(recent_history)
        profile.performance_metrics['accuracy'] = accuracy
        
        # Calculate average response time
        durations = [item['duration'] for item in recent_history]
        avg_response_time = statistics.mean(durations)
        profile.performance_metrics['response_time'] = avg_response_time
        
        # Calculate error rate
        errors = len(recent_history) - successes
        error_rate = errors / len(recent_history)
        profile.performance_metrics['error_rate'] = error_rate
        
        # Calculate resource efficiency
        resource_efficiency = {}
        for resource_type in ResourceType:
            resource_usages = []
            for item in recent_history:
                usage = item.get('resource_usage', {}).get(resource_type, 0)
                if usage > 0:
                    # Efficiency = output / resource_used (higher is better)
                    efficiency = 1.0 / usage if usage > 0 else 1.0
                    resource_usages.append(efficiency)
            
            if resource_usages:
                resource_efficiency[resource_type] = statistics.mean(resource_usages)
            else:
                resource_efficiency[resource_type] = 0.5  # Default efficiency
        
        profile.performance_metrics['resource_efficiency'] = resource_efficiency

class IntelligentWorkDistributionEngine:
    """Advanced work distribution engine with intelligent load balancing"""
    
    def __init__(self):
        self.complexity_estimator = AdvancedComplexityEstimator()
        self.performance_profiler = WorkerPerformanceProfiler()
        self.distribution_history: List[DistributionResult] = []
        self.rebalancing_threshold = 0.3  # Rebalance if load imbalance > 30%
        self.lock = threading.RLock()
        
        # Performance monitoring
        self.distribution_metrics = {
            'total_distributions': 0,
            'successful_distributions': 0,
            'rebalancing_events': 0,
            'average_balance_score': 0.0
        }
    
    def distribute_work(self, work_items: List[WorkItem], 
                       strategy: DistributionStrategy = DistributionStrategy.PERFORMANCE_ADAPTIVE) -> DistributionResult:
        """Distribute work items across available workers using specified strategy"""
        with self.lock:
            # Get available workers
            available_workers = self.performance_profiler.get_available_workers()
            
            if not available_workers:
                raise ValueError("No workers available for work distribution")
            
            # Estimate complexity for all work items
            for item in work_items:
                item.estimated_complexity = self.complexity_estimator.estimate_work_item_complexity(item)
            
            # Apply distribution strategy
            distribution_result = self._apply_distribution_strategy(work_items, available_workers, strategy)
            
            # Record distribution
            self.distribution_history.append(distribution_result)
            self.distribution_metrics['total_distributions'] += 1
            
            # Limit history size
            if len(self.distribution_history) > 1000:
                self.distribution_history = self.distribution_history[-800:]
            
            return distribution_result
    
    def rebalance_work(self, current_assignments: Dict[str, List[WorkItem]]) -> Optional[DistributionResult]:
        """Rebalance work based on current worker performance"""
        with self.lock:
            # Calculate current load imbalance
            balance_score = self._calculate_balance_score(current_assignments)
            
            if balance_score > self.rebalancing_threshold:
                return None  # No rebalancing needed
            
            # Get all incomplete work items
            all_work_items = []
            for worker_id, items in current_assignments.items():
                # Filter to incomplete items (would need completion status in real implementation)
                all_work_items.extend(items)
            
            # Redistribute work
            rebalanced_result = self.distribute_work(all_work_items, DistributionStrategy.PERFORMANCE_ADAPTIVE)
            
            self.distribution_metrics['rebalancing_events'] += 1
            
            return rebalanced_result
    
    def get_optimal_worker_for_item(self, work_item: WorkItem, 
                                  available_workers: List[str] = None) -> Tuple[str, float]:
        """Get the optimal worker for a specific work item"""
        if not available_workers:
            available_workers = self.performance_profiler.get_available_workers(work_item.resource_requirements)
        
        if not available_workers:
            raise ValueError("No suitable workers available")
        
        best_worker = None
        best_score = -1.0
        
        for worker_id in available_workers:
            score = self._calculate_worker_item_match_score(worker_id, work_item)
            if score > best_score:
                best_score = score
                best_worker = worker_id
        
        return best_worker, best_score
    
    def predict_distribution_performance(self, work_items: List[WorkItem], 
                                       worker_assignments: Dict[str, List[WorkItem]]) -> Dict[str, Any]:
        """Predict performance of a work distribution"""
        predictions = {}
        total_estimated_time = 0
        total_success_probability = 0
        worker_predictions = {}
        
        for worker_id, items in worker_assignments.items():
            worker_total_time = 0
            worker_success_probs = []
            
            for item in items:
                prediction = self.performance_profiler.predict_worker_performance(worker_id, item)
                worker_total_time += prediction['duration']
                worker_success_probs.append(prediction['success_probability'])
            
            worker_predictions[worker_id] = {
                'estimated_completion_time': worker_total_time,
                'average_success_probability': statistics.mean(worker_success_probs) if worker_success_probs else 0.5,
                'item_count': len(items)
            }
            
            total_estimated_time = max(total_estimated_time, worker_total_time)  # Bottleneck worker
            total_success_probability += statistics.mean(worker_success_probs) if worker_success_probs else 0.5
        
        predictions['overall_completion_time'] = total_estimated_time
        predictions['overall_success_probability'] = total_success_probability / len(worker_assignments) if worker_assignments else 0
        predictions['worker_predictions'] = worker_predictions
        predictions['load_balance_score'] = self._calculate_balance_score(worker_assignments)
        
        return predictions
    
    def get_distribution_analytics(self) -> Dict[str, Any]:
        """Get analytics on distribution performance"""
        with self.lock:
            if not self.distribution_history:
                return {'error': 'No distribution history available'}
            
            recent_distributions = self.distribution_history[-100:]  # Last 100 distributions
            
            balance_scores = [d.balance_score for d in recent_distributions]
            completion_times = [d.estimated_completion_time for d in recent_distributions]
            
            analytics = {
                'total_distributions': len(self.distribution_history),
                'recent_distributions': len(recent_distributions),
                'average_balance_score': statistics.mean(balance_scores),
                'balance_score_trend': self._calculate_trend(balance_scores),
                'average_completion_time': statistics.mean(completion_times),
                'completion_time_trend': self._calculate_trend(completion_times),
                'strategy_usage': self._analyze_strategy_usage(),
                'worker_utilization': self._analyze_worker_utilization(),
                'rebalancing_frequency': self.distribution_metrics['rebalancing_events'] / max(len(self.distribution_history), 1)
            }
            
            return analytics
    
    def _apply_distribution_strategy(self, work_items: List[WorkItem], 
                                   available_workers: List[str], 
                                   strategy: DistributionStrategy) -> DistributionResult:
        """Apply specific distribution strategy"""
        if strategy == DistributionStrategy.ROUND_ROBIN:
            return self._round_robin_distribution(work_items, available_workers)
        elif strategy == DistributionStrategy.COMPLEXITY_AWARE:
            return self._complexity_aware_distribution(work_items, available_workers)
        elif strategy == DistributionStrategy.LOAD_BALANCED:
            return self._load_balanced_distribution(work_items, available_workers)
        elif strategy == DistributionStrategy.PERFORMANCE_ADAPTIVE:
            return self._performance_adaptive_distribution(work_items, available_workers)
        elif strategy == DistributionStrategy.RESOURCE_OPTIMAL:
            return self._resource_optimal_distribution(work_items, available_workers)
        elif strategy == DistributionStrategy.PREDICTIVE:
            return self._predictive_distribution(work_items, available_workers)
        else:
            # Default to performance adaptive
            return self._performance_adaptive_distribution(work_items, available_workers)
    
    def _round_robin_distribution(self, work_items: List[WorkItem], 
                                available_workers: List[str]) -> DistributionResult:
        """Simple round-robin distribution"""
        worker_assignments = {worker_id: [] for worker_id in available_workers}
        
        for i, item in enumerate(work_items):
            worker_id = available_workers[i % len(available_workers)]
            worker_assignments[worker_id].append(item)
        
        balance_score = self._calculate_balance_score(worker_assignments)
        estimated_completion_time = self._estimate_completion_time(worker_assignments)
        
        return DistributionResult(
            worker_assignments=worker_assignments,
            distribution_strategy=DistributionStrategy.ROUND_ROBIN,
            balance_score=balance_score,
            estimated_completion_time=estimated_completion_time,
            resource_utilization=self._calculate_resource_utilization(worker_assignments),
            distribution_metadata={'algorithm': 'round_robin'}
        )
    
    def _complexity_aware_distribution(self, work_items: List[WorkItem], 
                                     available_workers: List[str]) -> DistributionResult:
        """Distribution based on work item complexity"""
        # Sort items by complexity (descending)
        sorted_items = sorted(work_items, key=lambda x: x.estimated_complexity, reverse=True)
        
        # Initialize worker loads
        worker_loads = {worker_id: 0.0 for worker_id in available_workers}
        worker_assignments = {worker_id: [] for worker_id in available_workers}
        
        # Assign each item to the worker with the lowest current load
        for item in sorted_items:
            best_worker = min(worker_loads.keys(), key=lambda w: worker_loads[w])
            worker_assignments[best_worker].append(item)
            worker_loads[best_worker] += item.estimated_complexity
        
        balance_score = self._calculate_balance_score(worker_assignments)
        estimated_completion_time = self._estimate_completion_time(worker_assignments)
        
        return DistributionResult(
            worker_assignments=worker_assignments,
            distribution_strategy=DistributionStrategy.COMPLEXITY_AWARE,
            balance_score=balance_score,
            estimated_completion_time=estimated_completion_time,
            resource_utilization=self._calculate_resource_utilization(worker_assignments),
            distribution_metadata={'algorithm': 'complexity_aware_greedy'}
        )
    
    def _performance_adaptive_distribution(self, work_items: List[WorkItem], 
                                         available_workers: List[str]) -> DistributionResult:
        """Distribution adapted to worker performance characteristics"""
        worker_assignments = {worker_id: [] for worker_id in available_workers}
        
        # Sort items by priority and complexity
        sorted_items = sorted(work_items, key=lambda x: (x.priority, x.estimated_complexity), reverse=True)
        
        for item in sorted_items:
            best_worker, _ = self.get_optimal_worker_for_item(item, available_workers)
            worker_assignments[best_worker].append(item)
        
        balance_score = self._calculate_balance_score(worker_assignments)
        estimated_completion_time = self._estimate_completion_time(worker_assignments)
        
        return DistributionResult(
            worker_assignments=worker_assignments,
            distribution_strategy=DistributionStrategy.PERFORMANCE_ADAPTIVE,
            balance_score=balance_score,
            estimated_completion_time=estimated_completion_time,
            resource_utilization=self._calculate_resource_utilization(worker_assignments),
            distribution_metadata={'algorithm': 'performance_adaptive_matching'}
        )
    
    def _load_balanced_distribution(self, work_items: List[WorkItem], 
                                  available_workers: List[str]) -> DistributionResult:
        """Distribution optimized for load balancing"""
        # Use bin packing algorithm for optimal load distribution
        worker_assignments = self._bin_packing_distribution(work_items, available_workers)
        
        balance_score = self._calculate_balance_score(worker_assignments)
        estimated_completion_time = self._estimate_completion_time(worker_assignments)
        
        return DistributionResult(
            worker_assignments=worker_assignments,
            distribution_strategy=DistributionStrategy.LOAD_BALANCED,
            balance_score=balance_score,
            estimated_completion_time=estimated_completion_time,
            resource_utilization=self._calculate_resource_utilization(worker_assignments),
            distribution_metadata={'algorithm': 'bin_packing_optimal'}
        )
    
    def _resource_optimal_distribution(self, work_items: List[WorkItem], 
                                     available_workers: List[str]) -> DistributionResult:
        """Distribution optimized for resource utilization"""
        worker_assignments = {worker_id: [] for worker_id in available_workers}
        
        # Track resource usage per worker
        worker_resource_usage = {}
        for worker_id in available_workers:
            profile = self.performance_profiler.worker_profiles.get(worker_id)
            if profile:
                worker_resource_usage[worker_id] = profile.current_load.copy()
            else:
                worker_resource_usage[worker_id] = {resource: 0.0 for resource in ResourceType}
        
        # Assign items based on resource requirements and availability
        for item in work_items:
            best_worker = self._find_best_resource_match(item, available_workers, worker_resource_usage)
            worker_assignments[best_worker].append(item)
            
            # Update resource usage tracking
            for resource_type, required in item.resource_requirements.items():
                worker_resource_usage[best_worker][resource_type] += required
        
        balance_score = self._calculate_balance_score(worker_assignments)
        estimated_completion_time = self._estimate_completion_time(worker_assignments)
        
        return DistributionResult(
            worker_assignments=worker_assignments,
            distribution_strategy=DistributionStrategy.RESOURCE_OPTIMAL,
            balance_score=balance_score,
            estimated_completion_time=estimated_completion_time,
            resource_utilization=self._calculate_resource_utilization(worker_assignments),
            distribution_metadata={'algorithm': 'resource_optimal_matching'}
        )
    
    def _predictive_distribution(self, work_items: List[WorkItem], 
                               available_workers: List[str]) -> DistributionResult:
        """Distribution using predictive modeling"""
        # Use historical performance to predict optimal assignments
        worker_assignments = {worker_id: [] for worker_id in available_workers}
        
        for item in work_items:
            worker_scores = {}
            for worker_id in available_workers:
                prediction = self.performance_profiler.predict_worker_performance(worker_id, item)
                # Score combines success probability and speed (inverse duration)
                speed_score = 1.0 / max(prediction['duration'], 0.1)
                success_score = prediction['success_probability']
                worker_scores[worker_id] = speed_score * success_score
            
            best_worker = max(worker_scores.keys(), key=lambda w: worker_scores[w])
            worker_assignments[best_worker].append(item)
        
        balance_score = self._calculate_balance_score(worker_assignments)
        estimated_completion_time = self._estimate_completion_time(worker_assignments)
        
        return DistributionResult(
            worker_assignments=worker_assignments,
            distribution_strategy=DistributionStrategy.PREDICTIVE,
            balance_score=balance_score,
            estimated_completion_time=estimated_completion_time,
            resource_utilization=self._calculate_resource_utilization(worker_assignments),
            distribution_metadata={'algorithm': 'predictive_performance_modeling'}
        )
    
    def _bin_packing_distribution(self, work_items: List[WorkItem], 
                                available_workers: List[str]) -> Dict[str, List[WorkItem]]:
        """Advanced bin packing algorithm for optimal load distribution"""
        # Sort items by complexity (largest first)
        sorted_items = sorted(work_items, key=lambda x: x.estimated_complexity, reverse=True)
        
        # Initialize bins (workers) with their capacities
        worker_assignments = {worker_id: [] for worker_id in available_workers}
        worker_loads = {worker_id: 0.0 for worker_id in available_workers}
        
        # Get worker capacities (based on performance scores)
        worker_capacities = {}
        for worker_id in available_workers:
            performance_score = self.performance_profiler.get_worker_performance_score(worker_id)
            # Higher performance workers get higher capacity
            worker_capacities[worker_id] = 10.0 * (1.0 + performance_score)
        
        # First Fit Decreasing algorithm
        for item in sorted_items:
            best_worker = None
            best_remaining_capacity = -1
            
            for worker_id in available_workers:
                remaining_capacity = worker_capacities[worker_id] - worker_loads[worker_id]
                
                if remaining_capacity >= item.estimated_complexity:
                    if remaining_capacity > best_remaining_capacity:
                        best_remaining_capacity = remaining_capacity
                        best_worker = worker_id
            
            # If no worker has enough capacity, assign to worker with most remaining capacity
            if best_worker is None:
                best_worker = max(available_workers, 
                                key=lambda w: worker_capacities[w] - worker_loads[w])
            
            worker_assignments[best_worker].append(item)
            worker_loads[best_worker] += item.estimated_complexity
        
        return worker_assignments
    
    def _calculate_worker_item_match_score(self, worker_id: str, work_item: WorkItem) -> float:
        """Calculate how well a worker matches a work item"""
        profile = self.performance_profiler.worker_profiles.get(worker_id)
        if not profile:
            return 0.5  # Default score for unknown workers
        
        # Base score from worker performance
        performance_score = self.performance_profiler.get_worker_performance_score(worker_id)
        
        # Specialization bonus
        specialization_bonus = 0.0
        item_type = work_item.metadata.get('type', '')
        if item_type in profile.specializations:
            specialization_bonus = 0.2
        
        # Resource availability score
        resource_score = 1.0
        for resource_type, required in work_item.resource_requirements.items():
            capacity = profile.resource_capacity.get(resource_type, 0)
            current_usage = profile.current_load.get(resource_type, 0)
            available = capacity - current_usage
            
            if required > available:
                resource_score *= 0.5  # Penalty for insufficient resources
            elif required > 0:
                utilization = required / capacity
                resource_score *= (1.0 - utilization * 0.5)  # Penalty for high utilization
        
        # Deadline urgency factor
        urgency_factor = 1.0
        if work_item.deadline:
            time_remaining = (work_item.deadline - datetime.now()).total_seconds()
            if time_remaining < 3600:  # Less than 1 hour
                urgency_factor = 1.5  # Favor faster workers for urgent items
        
        final_score = (performance_score + specialization_bonus) * resource_score * urgency_factor
        
        return max(0.0, min(1.0, final_score))
    
    def _calculate_balance_score(self, worker_assignments: Dict[str, List[WorkItem]]) -> float:
        """Calculate load balance score (0.0-1.0, higher is better)"""
        if not worker_assignments:
            return 1.0
        
        # Calculate load per worker (sum of complexity)
        worker_loads = []
        for worker_id, items in worker_assignments.items():
            total_load = sum(item.estimated_complexity for item in items)
            worker_loads.append(total_load)
        
        if not worker_loads or max(worker_loads) == 0:
            return 1.0
        
        # Calculate coefficient of variation (lower is better)
        mean_load = statistics.mean(worker_loads)
        if mean_load == 0:
            return 1.0
        
        std_dev = statistics.stdev(worker_loads) if len(worker_loads) > 1 else 0
        coefficient_variation = std_dev / mean_load
        
        # Convert to balance score (0 = perfect balance, 1 = perfect imbalance)
        balance_score = max(0.0, 1.0 - coefficient_variation)
        
        return balance_score
    
    def _estimate_completion_time(self, worker_assignments: Dict[str, List[WorkItem]]) -> float:
        """Estimate overall completion time (bottleneck worker)"""
        max_completion_time = 0.0
        
        for worker_id, items in worker_assignments.items():
            worker_time = 0.0
            for item in items:
                prediction = self.performance_profiler.predict_worker_performance(worker_id, item)
                worker_time += prediction['duration']
            
            max_completion_time = max(max_completion_time, worker_time)
        
        return max_completion_time
    
    def _calculate_resource_utilization(self, worker_assignments: Dict[str, List[WorkItem]]) -> Dict[ResourceType, float]:
        """Calculate overall resource utilization"""
        total_resource_usage = {resource: 0.0 for resource in ResourceType}
        total_resource_capacity = {resource: 0.0 for resource in ResourceType}
        
        for worker_id, items in worker_assignments.items():
            profile = self.performance_profiler.worker_profiles.get(worker_id)
            if not profile:
                continue
            
            # Add worker capacity to total
            for resource_type, capacity in profile.resource_capacity.items():
                total_resource_capacity[resource_type] += capacity
            
            # Add work item resource requirements
            for item in items:
                for resource_type, required in item.resource_requirements.items():
                    total_resource_usage[resource_type] += required
        
        # Calculate utilization percentages
        utilization = {}
        for resource_type in ResourceType:
            capacity = total_resource_capacity[resource_type]
            if capacity > 0:
                utilization[resource_type] = min(1.0, total_resource_usage[resource_type] / capacity)
            else:
                utilization[resource_type] = 0.0
        
        return utilization
    
    def _find_best_resource_match(self, work_item: WorkItem, available_workers: List[str],
                                worker_resource_usage: Dict[str, Dict[ResourceType, float]]) -> str:
        """Find worker with best resource match for work item"""
        best_worker = available_workers[0]
        best_score = -1.0
        
        for worker_id in available_workers:
            profile = self.performance_profiler.worker_profiles.get(worker_id)
            if not profile:
                continue
            
            # Calculate resource fit score
            fit_score = 1.0
            for resource_type, required in work_item.resource_requirements.items():
                capacity = profile.resource_capacity.get(resource_type, 0)
                current_usage = worker_resource_usage[worker_id].get(resource_type, 0)
                available = capacity - current_usage
                
                if required > available:
                    fit_score = 0.0  # Cannot handle this item
                    break
                elif capacity > 0:
                    utilization = required / capacity
                    fit_score *= (1.0 - utilization)  # Lower utilization is better
            
            if fit_score > best_score:
                best_score = fit_score
                best_worker = worker_id
        
        return best_worker
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values"""
        if len(values) < 2:
            return "stable"
        
        # Simple linear trend
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"
    
    def _analyze_strategy_usage(self) -> Dict[str, int]:
        """Analyze which distribution strategies are used most"""
        strategy_counts = defaultdict(int)
        for result in self.distribution_history[-100:]:  # Last 100 distributions
            strategy_counts[result.distribution_strategy.value] += 1
        return dict(strategy_counts)
    
    def _analyze_worker_utilization(self) -> Dict[str, float]:
        """Analyze worker utilization across recent distributions"""
        worker_utilization = defaultdict(float)
        distribution_count = 0
        
        for result in self.distribution_history[-50:]:  # Last 50 distributions
            distribution_count += 1
            for worker_id, items in result.worker_assignments.items():
                worker_utilization[worker_id] += len(items)
        
        # Convert to average items per distribution
        if distribution_count > 0:
            for worker_id in worker_utilization:
                worker_utilization[worker_id] /= distribution_count
        
        return dict(worker_utilization)