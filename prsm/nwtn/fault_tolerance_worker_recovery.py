"""
NWTN Fault Tolerance & Worker Recovery System

Advanced fault tolerance and worker recovery system for production-ready parallel processing
with automatic failure detection, work redistribution, and distributed checkpointing.

Implements comprehensive resilience architecture:
1. Worker Health Monitoring - Real-time health detection and failure prediction
2. Work Redistribution Engine - Intelligent work redistribution across healthy workers
3. Distributed Checkpoint Manager - Fault-tolerant progress preservation
4. Recovery Coordination - Automated recovery orchestration and resource rebalancing
5. Failure Analytics - Pattern analysis for proactive failure prevention

Features:
- Real-time worker health monitoring with predictive failure detection
- Intelligent work redistribution with load balancing optimization
- Distributed checkpointing with corruption detection and recovery
- Automatic worker recovery and resource reallocation
- Comprehensive failure analytics and prevention strategies
- Production-grade monitoring with alerting and reporting

Part of NWTN Phase 8: Parallel Processing & Scalability Architecture
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple, Union, Callable, AsyncCallable
from enum import Enum
from datetime import datetime, timedelta
import threading
import time
import asyncio
import json
import pickle
import gzip
import statistics
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import weakref
from uuid import uuid4
import psutil
import hashlib
import logging
from pathlib import Path

class WorkerStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    FAILING = "failing"
    FAILED = "failed"
    RECOVERING = "recovering"
    UNAVAILABLE = "unavailable"

class FailureType(Enum):
    TIMEOUT = "timeout"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    CPU_OVERLOAD = "cpu_overload"
    NETWORK_FAILURE = "network_failure"
    PROCESS_CRASH = "process_crash"
    RESOURCE_CONTENTION = "resource_contention"
    UNHANDLED_EXCEPTION = "unhandled_exception"

class RecoveryStrategy(Enum):
    RESTART_WORKER = "restart_worker"
    REDISTRIBUTE_WORK = "redistribute_work"
    SCALE_RESOURCES = "scale_resources"
    ISOLATE_FAILURE = "isolate_failure"
    GRACEFUL_DEGRADATION = "graceful_degradation"

class CheckpointType(Enum):
    INCREMENTAL = "incremental"
    FULL = "full"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"

@dataclass
class WorkerHealthMetrics:
    worker_id: str
    status: WorkerStatus
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_latency: float = 0.0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    response_time: float = 0.0
    error_rate: float = 0.0
    health_score: float = 1.0
    predicted_failure_probability: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    performance_degradation: float = 0.0

@dataclass
class WorkerFailure:
    worker_id: str
    failure_type: FailureType
    failure_time: datetime
    failure_details: str
    incomplete_work: List[Any] = field(default_factory=list)
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_attempts: int = 0
    recovery_success: bool = False
    impact_assessment: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DistributedCheckpoint:
    checkpoint_id: str
    checkpoint_type: CheckpointType
    timestamp: datetime
    worker_states: Dict[str, Any]
    work_distribution: Dict[str, List[Any]]
    progress_metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    compression_enabled: bool = True
    recovery_information: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecoveryOperation:
    operation_id: str
    failed_workers: List[str]
    recovery_strategy: RecoveryStrategy
    start_time: datetime
    estimated_completion_time: Optional[datetime] = None
    affected_work: List[Any] = field(default_factory=list)
    new_work_distribution: Dict[str, List[Any]] = field(default_factory=dict)
    recovery_progress: float = 0.0
    success_probability: float = 0.0
    rollback_plan: Optional[Dict[str, Any]] = None

class WorkerHealthMonitor:
    """Advanced worker health monitoring with predictive failure detection"""
    
    def __init__(self, 
                 health_check_interval: int = 30,
                 failure_threshold: float = 0.8,
                 prediction_window: int = 300):
        self.health_check_interval = health_check_interval
        self.failure_threshold = failure_threshold
        self.prediction_window = prediction_window
        
        # Worker tracking
        self.worker_metrics: Dict[str, WorkerHealthMetrics] = {}
        self.worker_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Health monitoring
        self.health_patterns: Dict[str, List[float]] = defaultdict(list)
        self.failure_predictors: Dict[str, float] = {}
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        self.lock = threading.RLock()
        
        # Analytics
        self.failure_statistics = {
            'total_failures': 0,
            'failure_types': defaultdict(int),
            'recovery_success_rate': 0.0,
            'mean_time_to_failure': 0.0,
            'mean_time_to_recovery': 0.0
        }
        
    def start_monitoring(self):
        """Start continuous worker health monitoring"""
        with self.lock:
            if not self.monitoring_active:
                self.monitoring_active = True
                self.monitoring_thread = threading.Thread(
                    target=self._monitoring_loop, daemon=True
                )
                self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop worker health monitoring"""
        with self.lock:
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5.0)
    
    def register_worker(self, worker_id: str) -> WorkerHealthMetrics:
        """Register a new worker for monitoring"""
        with self.lock:
            metrics = WorkerHealthMetrics(
                worker_id=worker_id,
                status=WorkerStatus.HEALTHY
            )
            self.worker_metrics[worker_id] = metrics
            return metrics
    
    def update_worker_metrics(self, worker_id: str, **metrics) -> bool:
        """Update worker health metrics"""
        with self.lock:
            if worker_id not in self.worker_metrics:
                return False
            
            worker_metrics = self.worker_metrics[worker_id]
            
            # Update metrics
            for key, value in metrics.items():
                if hasattr(worker_metrics, key):
                    setattr(worker_metrics, key, value)
            
            # Update last heartbeat
            worker_metrics.last_heartbeat = datetime.now()
            
            # Calculate health score
            worker_metrics.health_score = self._calculate_health_score(worker_metrics)
            
            # Predict failure probability
            worker_metrics.predicted_failure_probability = self._predict_failure_probability(worker_id)
            
            # Update status based on health score
            worker_metrics.status = self._determine_worker_status(worker_metrics)
            
            # Store historical data
            self.worker_history[worker_id].append({
                'timestamp': datetime.now(),
                'health_score': worker_metrics.health_score,
                'cpu_usage': worker_metrics.cpu_usage,
                'memory_usage': worker_metrics.memory_usage,
                'error_rate': worker_metrics.error_rate
            })
            
            return True
    
    def detect_failures(self) -> List[str]:
        """Detect failed workers"""
        with self.lock:
            failed_workers = []
            current_time = datetime.now()
            
            for worker_id, metrics in self.worker_metrics.items():
                # Check for timeout
                time_since_heartbeat = (current_time - metrics.last_heartbeat).total_seconds()
                if time_since_heartbeat > 120:  # 2 minutes timeout
                    metrics.status = WorkerStatus.FAILED
                    failed_workers.append(worker_id)
                    continue
                
                # Check health score
                if metrics.health_score < self.failure_threshold:
                    if metrics.status != WorkerStatus.FAILED:
                        metrics.status = WorkerStatus.FAILING
                        if metrics.health_score < 0.3:  # Critical threshold
                            metrics.status = WorkerStatus.FAILED
                            failed_workers.append(worker_id)
                
                # Check predicted failure probability
                if metrics.predicted_failure_probability > 0.9:
                    metrics.status = WorkerStatus.FAILING
                    failed_workers.append(worker_id)
            
            return failed_workers
    
    def get_healthy_workers(self) -> List[str]:
        """Get list of healthy worker IDs"""
        with self.lock:
            return [
                worker_id for worker_id, metrics in self.worker_metrics.items()
                if metrics.status in [WorkerStatus.HEALTHY, WorkerStatus.WARNING]
            ]
    
    def get_worker_capacity(self, worker_id: str) -> float:
        """Get worker capacity score (0.0-1.0)"""
        with self.lock:
            if worker_id not in self.worker_metrics:
                return 0.0
            
            metrics = self.worker_metrics[worker_id]
            
            # Calculate capacity based on resource utilization
            cpu_capacity = max(0.0, 1.0 - metrics.cpu_usage)
            memory_capacity = max(0.0, 1.0 - metrics.memory_usage)
            load_capacity = max(0.0, 1.0 - (metrics.active_tasks / 10.0))  # Assume max 10 tasks
            
            return (cpu_capacity + memory_capacity + load_capacity) / 3.0
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Update worker metrics from system
                self._update_system_metrics()
                
                # Detect patterns and update predictors
                self._update_failure_predictors()
                
                # Sleep until next check
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logging.error(f"Error in worker health monitoring: {e}")
                time.sleep(5)  # Brief pause before retrying
    
    def _update_system_metrics(self):
        """Update system-level metrics for all workers"""
        with self.lock:
            for worker_id in self.worker_metrics:
                # This would typically query actual worker processes
                # For now, we'll simulate with system metrics
                try:
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory_info = psutil.virtual_memory()
                    
                    self.update_worker_metrics(
                        worker_id,
                        cpu_usage=min(cpu_percent / 100.0, 1.0),
                        memory_usage=memory_info.percent / 100.0,
                        resource_utilization={
                            'cpu': cpu_percent / 100.0,
                            'memory': memory_info.percent / 100.0,
                            'disk': psutil.disk_usage('/').percent / 100.0
                        }
                    )
                except Exception as e:
                    logging.warning(f"Failed to update metrics for worker {worker_id}: {e}")
    
    def _calculate_health_score(self, metrics: WorkerHealthMetrics) -> float:
        """Calculate overall worker health score"""
        # Base score from resource utilization
        cpu_score = max(0.0, 1.0 - metrics.cpu_usage)
        memory_score = max(0.0, 1.0 - metrics.memory_usage)
        
        # Performance scores
        error_score = max(0.0, 1.0 - metrics.error_rate)
        response_score = max(0.0, 1.0 - min(metrics.response_time / 10.0, 1.0))
        
        # Task completion rate
        total_tasks = metrics.completed_tasks + metrics.failed_tasks
        completion_score = metrics.completed_tasks / max(total_tasks, 1)
        
        # Weighted combination
        health_score = (
            cpu_score * 0.2 +
            memory_score * 0.2 +
            error_score * 0.3 +
            response_score * 0.15 +
            completion_score * 0.15
        )
        
        return max(0.0, min(1.0, health_score))
    
    def _predict_failure_probability(self, worker_id: str) -> float:
        """Predict probability of worker failure"""
        if worker_id not in self.worker_history:
            return 0.0
        
        history = list(self.worker_history[worker_id])
        if len(history) < 5:
            return 0.0
        
        # Analyze trends in health scores
        recent_scores = [h['health_score'] for h in history[-10:]]
        
        # Calculate trend (negative = declining health)
        if len(recent_scores) >= 2:
            trend = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
        else:
            trend = 0.0
        
        # Calculate volatility
        volatility = statistics.stdev(recent_scores) if len(recent_scores) > 1 else 0.0
        
        # Current health score
        current_score = recent_scores[-1]
        
        # Failure probability based on trend, volatility, and current health
        base_probability = 1.0 - current_score
        trend_factor = max(0.0, -trend * 2)  # Negative trend increases failure probability
        volatility_factor = volatility
        
        failure_probability = min(1.0, base_probability + trend_factor + volatility_factor)
        
        return failure_probability
    
    def _determine_worker_status(self, metrics: WorkerHealthMetrics) -> WorkerStatus:
        """Determine worker status based on metrics"""
        if metrics.health_score >= 0.8:
            return WorkerStatus.HEALTHY
        elif metrics.health_score >= 0.6:
            return WorkerStatus.WARNING
        elif metrics.health_score >= 0.3:
            return WorkerStatus.FAILING
        else:
            return WorkerStatus.FAILED
    
    def _update_failure_predictors(self):
        """Update failure prediction models"""
        with self.lock:
            for worker_id in self.worker_metrics:
                # Simple pattern-based predictor
                # In production, this would use more sophisticated ML models
                if worker_id in self.worker_history:
                    history = list(self.worker_history[worker_id])
                    if len(history) >= 10:
                        # Look for declining patterns
                        recent_trend = self._calculate_health_trend(history[-10:])
                        self.failure_predictors[worker_id] = max(0.0, -recent_trend)
    
    def _calculate_health_trend(self, history: List[Dict]) -> float:
        """Calculate health trend from history"""
        if len(history) < 2:
            return 0.0
        
        scores = [h['health_score'] for h in history]
        x_values = list(range(len(scores)))
        
        # Simple linear regression
        n = len(scores)
        sum_x = sum(x_values)
        sum_y = sum(scores)
        sum_xy = sum(x * y for x, y in zip(x_values, scores))
        sum_x2 = sum(x * x for x in x_values)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope

class WorkRedistributionEngine:
    """Intelligent work redistribution across healthy workers"""
    
    def __init__(self, health_monitor: WorkerHealthMonitor):
        self.health_monitor = health_monitor
        self.redistribution_strategies = {
            'capacity_based': self._capacity_based_redistribution,
            'round_robin': self._round_robin_redistribution,
            'load_balanced': self._load_balanced_redistribution,
            'priority_aware': self._priority_aware_redistribution
        }
        self.lock = threading.RLock()
    
    def redistribute(self, 
                    incomplete_work: List[Any], 
                    healthy_workers: List[str],
                    strategy: str = 'capacity_based') -> Dict[str, List[Any]]:
        """Redistribute work across healthy workers"""
        with self.lock:
            if not healthy_workers:
                return {}
            
            if strategy not in self.redistribution_strategies:
                strategy = 'capacity_based'
            
            return self.redistribution_strategies[strategy](incomplete_work, healthy_workers)
    
    def _capacity_based_redistribution(self, 
                                     work: List[Any], 
                                     workers: List[str]) -> Dict[str, List[Any]]:
        """Redistribute based on worker capacity"""
        # Calculate worker capacities
        worker_capacities = {}
        total_capacity = 0.0
        
        for worker_id in workers:
            capacity = self.health_monitor.get_worker_capacity(worker_id)
            worker_capacities[worker_id] = capacity
            total_capacity += capacity
        
        if total_capacity == 0:
            return self._round_robin_redistribution(work, workers)
        
        # Distribute work proportionally
        distribution = {worker_id: [] for worker_id in workers}
        work_per_capacity = len(work) / total_capacity
        
        work_index = 0
        for worker_id in workers:
            worker_work_count = int(worker_capacities[worker_id] * work_per_capacity)
            worker_work_count = min(worker_work_count, len(work) - work_index)
            
            for _ in range(worker_work_count):
                if work_index < len(work):
                    distribution[worker_id].append(work[work_index])
                    work_index += 1
        
        # Distribute remaining work round-robin
        worker_cycle = iter(workers * ((len(work) - work_index) // len(workers) + 1))
        while work_index < len(work):
            worker_id = next(worker_cycle)
            distribution[worker_id].append(work[work_index])
            work_index += 1
        
        return distribution
    
    def _round_robin_redistribution(self, 
                                  work: List[Any], 
                                  workers: List[str]) -> Dict[str, List[Any]]:
        """Simple round-robin redistribution"""
        distribution = {worker_id: [] for worker_id in workers}
        
        for i, work_item in enumerate(work):
            worker_id = workers[i % len(workers)]
            distribution[worker_id].append(work_item)
        
        return distribution
    
    def _load_balanced_redistribution(self, 
                                    work: List[Any], 
                                    workers: List[str]) -> Dict[str, List[Any]]:
        """Redistribute based on current load"""
        # Get current active tasks for each worker
        worker_loads = {}
        for worker_id in workers:
            if worker_id in self.health_monitor.worker_metrics:
                worker_loads[worker_id] = self.health_monitor.worker_metrics[worker_id].active_tasks
            else:
                worker_loads[worker_id] = 0
        
        distribution = {worker_id: [] for worker_id in workers}
        
        # Assign work to least loaded workers first
        for work_item in work:
            # Find worker with minimum load
            min_load_worker = min(worker_loads.keys(), key=lambda w: worker_loads[w])
            distribution[min_load_worker].append(work_item)
            worker_loads[min_load_worker] += 1
        
        return distribution
    
    def _priority_aware_redistribution(self, 
                                     work: List[Any], 
                                     workers: List[str]) -> Dict[str, List[Any]]:
        """Redistribute based on work priority and worker capabilities"""
        # Sort work by priority (assuming work items have priority attribute)
        try:
            sorted_work = sorted(work, key=lambda item: getattr(item, 'priority', 0), reverse=True)
        except:
            sorted_work = work
        
        # Sort workers by health score (best workers get high priority work)
        worker_scores = []
        for worker_id in workers:
            if worker_id in self.health_monitor.worker_metrics:
                score = self.health_monitor.worker_metrics[worker_id].health_score
            else:
                score = 0.5
            worker_scores.append((worker_id, score))
        
        sorted_workers = [w[0] for w in sorted(worker_scores, key=lambda x: x[1], reverse=True)]
        
        # Distribute high priority work to best workers
        distribution = {worker_id: [] for worker_id in workers}
        
        for i, work_item in enumerate(sorted_work):
            # Assign to best available worker based on current load
            best_worker = None
            min_load = float('inf')
            
            for worker_id in sorted_workers:
                current_load = len(distribution[worker_id])
                if current_load < min_load:
                    min_load = current_load
                    best_worker = worker_id
            
            if best_worker:
                distribution[best_worker].append(work_item)
        
        return distribution

class DistributedCheckpointManager:
    """Distributed checkpoint management with fault tolerance"""
    
    def __init__(self, 
                 checkpoint_dir: str = "./checkpoints",
                 checkpoint_interval: int = 300,
                 max_checkpoints: int = 10):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        
        # Checkpoint storage
        self.checkpoints: Dict[str, DistributedCheckpoint] = {}
        self.checkpoint_history: List[str] = []
        
        # Worker state tracking
        self.worker_states: Dict[str, Any] = {}
        self.work_assignments: Dict[str, List[Any]] = {}
        
        # Checkpointing thread
        self.checkpointing_active = False
        self.checkpointing_thread = None
        self.lock = threading.RLock()
        
        # Load existing checkpoints
        self._load_existing_checkpoints()
    
    def start_checkpointing(self):
        """Start automatic checkpoint creation"""
        with self.lock:
            if not self.checkpointing_active:
                self.checkpointing_active = True
                self.checkpointing_thread = threading.Thread(
                    target=self._checkpointing_loop, daemon=True
                )
                self.checkpointing_thread.start()
    
    def stop_checkpointing(self):
        """Stop automatic checkpoint creation"""
        with self.lock:
            self.checkpointing_active = False
            if self.checkpointing_thread:
                self.checkpointing_thread.join(timeout=5.0)
    
    def create_checkpoint(self, 
                         checkpoint_type: CheckpointType = CheckpointType.INCREMENTAL) -> str:
        """Create a distributed checkpoint"""
        with self.lock:
            checkpoint_id = f"checkpoint_{int(time.time())}_{uuid4().hex[:8]}"
            
            checkpoint = DistributedCheckpoint(
                checkpoint_id=checkpoint_id,
                checkpoint_type=checkpoint_type,
                timestamp=datetime.now(),
                worker_states=self.worker_states.copy(),
                work_distribution=self.work_assignments.copy(),
                progress_metrics=self._calculate_progress_metrics(),
                metadata={
                    'total_workers': len(self.worker_states),
                    'active_work_items': sum(len(work) for work in self.work_assignments.values()),
                    'system_state': 'active'
                }
            )
            
            # Calculate checksum
            checkpoint.checksum = self._calculate_checksum(checkpoint)
            
            # Save checkpoint
            if self._save_checkpoint(checkpoint):
                self.checkpoints[checkpoint_id] = checkpoint
                self.checkpoint_history.append(checkpoint_id)
                
                # Cleanup old checkpoints
                self._cleanup_old_checkpoints()
                
                return checkpoint_id
            
            return ""
    
    def restore_from_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore system state from checkpoint"""
        with self.lock:
            if checkpoint_id not in self.checkpoints:
                # Try to load from disk
                if not self._load_checkpoint(checkpoint_id):
                    return False
            
            checkpoint = self.checkpoints[checkpoint_id]
            
            # Verify checkpoint integrity
            if not self._verify_checkpoint_integrity(checkpoint):
                logging.error(f"Checkpoint {checkpoint_id} integrity check failed")
                return False
            
            # Restore state
            self.worker_states = checkpoint.worker_states.copy()
            self.work_assignments = checkpoint.work_distribution.copy()
            
            logging.info(f"Successfully restored from checkpoint {checkpoint_id}")
            return True
    
    def get_incomplete_work(self, worker_id: str) -> List[Any]:
        """Get incomplete work for a failed worker"""
        with self.lock:
            return self.work_assignments.get(worker_id, [])
    
    def update_worker_state(self, worker_id: str, state: Any):
        """Update worker state for checkpointing"""
        with self.lock:
            self.worker_states[worker_id] = state
    
    def update_work_assignment(self, worker_id: str, work_items: List[Any]):
        """Update work assignment for a worker"""
        with self.lock:
            self.work_assignments[worker_id] = work_items
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the latest checkpoint ID"""
        with self.lock:
            if self.checkpoint_history:
                return self.checkpoint_history[-1]
            return None
    
    def _checkpointing_loop(self):
        """Automatic checkpoint creation loop"""
        while self.checkpointing_active:
            try:
                self.create_checkpoint(CheckpointType.INCREMENTAL)
                time.sleep(self.checkpoint_interval)
            except Exception as e:
                logging.error(f"Error in automatic checkpointing: {e}")
                time.sleep(30)  # Pause before retrying
    
    def _calculate_progress_metrics(self) -> Dict[str, Any]:
        """Calculate current progress metrics"""
        total_work_items = sum(len(work) for work in self.work_assignments.values())
        active_workers = len([w for w in self.worker_states if self.worker_states[w].get('active', False)])
        
        return {
            'total_work_items': total_work_items,
            'active_workers': active_workers,
            'checkpoint_time': time.time(),
            'system_health': self._calculate_system_health()
        }
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health score"""
        if not self.worker_states:
            return 0.0
        
        healthy_workers = sum(1 for state in self.worker_states.values() 
                            if state.get('health_score', 0) > 0.5)
        
        return healthy_workers / len(self.worker_states)
    
    def _calculate_checksum(self, checkpoint: DistributedCheckpoint) -> str:
        """Calculate checkpoint checksum for integrity verification"""
        # Create deterministic representation
        checkpoint_data = {
            'checkpoint_id': checkpoint.checkpoint_id,
            'timestamp': checkpoint.timestamp.isoformat(),
            'worker_states': checkpoint.worker_states,
            'work_distribution': checkpoint.work_distribution,
            'progress_metrics': checkpoint.progress_metrics
        }
        
        # Calculate SHA-256 hash
        checkpoint_json = json.dumps(checkpoint_data, sort_keys=True, default=str)
        return hashlib.sha256(checkpoint_json.encode()).hexdigest()
    
    def _save_checkpoint(self, checkpoint: DistributedCheckpoint) -> bool:
        """Save checkpoint to disk"""
        try:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint.checkpoint_id}.pkl"
            
            checkpoint_data = checkpoint
            if checkpoint.compression_enabled:
                # Compress checkpoint data
                serialized_data = pickle.dumps(checkpoint_data)
                compressed_data = gzip.compress(serialized_data)
                
                with open(checkpoint_file, 'wb') as f:
                    f.write(compressed_data)
            else:
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to save checkpoint {checkpoint.checkpoint_id}: {e}")
            return False
    
    def _load_checkpoint(self, checkpoint_id: str) -> bool:
        """Load checkpoint from disk"""
        try:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"
            
            if not checkpoint_file.exists():
                return False
            
            with open(checkpoint_file, 'rb') as f:
                file_data = f.read()
            
            try:
                # Try to decompress first
                decompressed_data = gzip.decompress(file_data)
                checkpoint = pickle.loads(decompressed_data)
            except:
                # If decompression fails, try direct pickle load
                checkpoint = pickle.loads(file_data)
            
            # Verify checkpoint integrity
            if self._verify_checkpoint_integrity(checkpoint):
                self.checkpoints[checkpoint_id] = checkpoint
                return True
            else:
                logging.error(f"Checkpoint {checkpoint_id} failed integrity check")
                return False
                
        except Exception as e:
            logging.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return False
    
    def _verify_checkpoint_integrity(self, checkpoint: DistributedCheckpoint) -> bool:
        """Verify checkpoint integrity using checksum"""
        expected_checksum = self._calculate_checksum(checkpoint)
        return expected_checksum == checkpoint.checksum
    
    def _load_existing_checkpoints(self):
        """Load existing checkpoints from disk"""
        try:
            for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.pkl"):
                checkpoint_id = checkpoint_file.stem
                if self._load_checkpoint(checkpoint_id):
                    self.checkpoint_history.append(checkpoint_id)
            
            # Sort by creation time
            self.checkpoint_history.sort(key=lambda cid: self.checkpoints[cid].timestamp)
            
        except Exception as e:
            logging.error(f"Error loading existing checkpoints: {e}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond the maximum limit"""
        while len(self.checkpoint_history) > self.max_checkpoints:
            old_checkpoint_id = self.checkpoint_history.pop(0)
            
            # Remove from memory
            if old_checkpoint_id in self.checkpoints:
                del self.checkpoints[old_checkpoint_id]
            
            # Remove from disk
            try:
                checkpoint_file = self.checkpoint_dir / f"{old_checkpoint_id}.pkl"
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
            except Exception as e:
                logging.warning(f"Failed to remove old checkpoint file: {e}")

class ParallelProcessingResilience:
    """Main fault tolerance and recovery orchestration system"""
    
    def __init__(self,
                 health_check_interval: int = 30,
                 checkpoint_interval: int = 300,
                 max_recovery_attempts: int = 3):
        
        # Core components
        self.health_monitor = WorkerHealthMonitor(
            health_check_interval=health_check_interval
        )
        self.work_redistributor = WorkRedistributionEngine(self.health_monitor)
        self.checkpoint_manager = DistributedCheckpointManager(
            checkpoint_interval=checkpoint_interval
        )
        
        # Recovery settings
        self.max_recovery_attempts = max_recovery_attempts
        self.recovery_operations: Dict[str, RecoveryOperation] = {}
        
        # System state
        self.processing_active = False
        self.resilience_thread = None
        self.lock = threading.RLock()
        
        # Failure analytics
        self.failure_history: List[WorkerFailure] = []
        self.recovery_statistics = {
            'total_failures': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'average_recovery_time': 0.0,
            'failure_patterns': defaultdict(int)
        }
    
    def start_resilience_monitoring(self):
        """Start fault tolerance and recovery monitoring"""
        with self.lock:
            if not self.processing_active:
                self.processing_active = True
                
                # Start component services
                self.health_monitor.start_monitoring()
                self.checkpoint_manager.start_checkpointing()
                
                # Start main resilience loop
                self.resilience_thread = threading.Thread(
                    target=self._resilience_monitoring_loop, daemon=True
                )
                self.resilience_thread.start()
    
    def stop_resilience_monitoring(self):
        """Stop fault tolerance and recovery monitoring"""
        with self.lock:
            self.processing_active = False
            
            # Stop component services
            self.health_monitor.stop_monitoring()
            self.checkpoint_manager.stop_checkpointing()
            
            # Stop main thread
            if self.resilience_thread:
                self.resilience_thread.join(timeout=10.0)
    
    def register_worker(self, worker_id: str):
        """Register a worker for monitoring"""
        self.health_monitor.register_worker(worker_id)
        self.checkpoint_manager.update_worker_state(worker_id, {
            'registered_at': datetime.now(),
            'active': True,
            'health_score': 1.0
        })
    
    def update_worker_status(self, worker_id: str, **metrics):
        """Update worker status and metrics"""
        # Update health monitoring
        self.health_monitor.update_worker_metrics(worker_id, **metrics)
        
        # Update checkpoint state
        worker_state = {
            'last_update': datetime.now(),
            'active': True,
            **metrics
        }
        self.checkpoint_manager.update_worker_state(worker_id, worker_state)
    
    def assign_work(self, worker_id: str, work_items: List[Any]):
        """Assign work to a worker"""
        self.checkpoint_manager.update_work_assignment(worker_id, work_items)
    
    async def handle_worker_failure(self, failed_workers: List[str]) -> bool:
        """Handle worker failure with recovery orchestration"""
        if not failed_workers:
            return True
        
        logging.warning(f"Handling failure of workers: {failed_workers}")
        
        # Create recovery operation
        recovery_operation = RecoveryOperation(
            operation_id=f"recovery_{int(time.time())}_{uuid4().hex[:8]}",
            failed_workers=failed_workers,
            recovery_strategy=RecoveryStrategy.REDISTRIBUTE_WORK,
            start_time=datetime.now()
        )
        
        try:
            # Analyze failure and determine strategy
            recovery_operation.recovery_strategy = self._determine_recovery_strategy(failed_workers)
            
            # Execute recovery strategy
            success = await self._execute_recovery_strategy(recovery_operation)
            
            # Update statistics
            self.recovery_statistics['total_failures'] += len(failed_workers)
            if success:
                self.recovery_statistics['successful_recoveries'] += 1
            else:
                self.recovery_statistics['failed_recoveries'] += 1
            
            # Record failure details
            for worker_id in failed_workers:
                failure = WorkerFailure(
                    worker_id=worker_id,
                    failure_type=self._classify_failure_type(worker_id),
                    failure_time=datetime.now(),
                    failure_details=f"Worker failed during operation {recovery_operation.operation_id}",
                    incomplete_work=self.checkpoint_manager.get_incomplete_work(worker_id),
                    recovery_strategy=recovery_operation.recovery_strategy,
                    recovery_success=success
                )
                self.failure_history.append(failure)
            
            return success
            
        except Exception as e:
            logging.error(f"Error in recovery operation {recovery_operation.operation_id}: {e}")
            return False
    
    def get_system_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        with self.lock:
            healthy_workers = self.health_monitor.get_healthy_workers()
            all_workers = list(self.health_monitor.worker_metrics.keys())
            
            # Calculate overall health
            if all_workers:
                total_health = sum(
                    self.health_monitor.worker_metrics[w].health_score 
                    for w in all_workers
                )
                average_health = total_health / len(all_workers)
            else:
                average_health = 0.0
            
            return {
                'system_health_score': average_health,
                'total_workers': len(all_workers),
                'healthy_workers': len(healthy_workers),
                'failed_workers': len(all_workers) - len(healthy_workers),
                'active_recovery_operations': len(self.recovery_operations),
                'total_failures_handled': self.recovery_statistics['total_failures'],
                'recovery_success_rate': (
                    self.recovery_statistics['successful_recoveries'] / 
                    max(self.recovery_statistics['total_failures'], 1)
                ),
                'last_checkpoint': self.checkpoint_manager.get_latest_checkpoint(),
                'failure_patterns': dict(self.recovery_statistics['failure_patterns'])
            }
    
    def _resilience_monitoring_loop(self):
        """Main resilience monitoring and recovery loop"""
        while self.processing_active:
            try:
                # Detect failed workers
                failed_workers = self.health_monitor.detect_failures()
                
                if failed_workers:
                    # Handle failures asynchronously
                    asyncio.create_task(self.handle_worker_failure(failed_workers))
                
                # Check recovery operations
                self._check_recovery_operations()
                
                # Sleep until next check
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logging.error(f"Error in resilience monitoring: {e}")
                time.sleep(10)  # Brief pause before retrying
    
    def _determine_recovery_strategy(self, failed_workers: List[str]) -> RecoveryStrategy:
        """Determine the best recovery strategy for failed workers"""
        # Simple strategy selection based on failure patterns
        total_workers = len(self.health_monitor.worker_metrics)
        failed_count = len(failed_workers)
        healthy_workers = self.health_monitor.get_healthy_workers()
        
        # If most workers failed, try graceful degradation
        if failed_count / total_workers > 0.7:
            return RecoveryStrategy.GRACEFUL_DEGRADATION
        
        # If few healthy workers remain, try to restart failed workers
        elif len(healthy_workers) < 3:
            return RecoveryStrategy.RESTART_WORKER
        
        # If we have healthy workers, redistribute work
        elif len(healthy_workers) >= failed_count:
            return RecoveryStrategy.REDISTRIBUTE_WORK
        
        # Default to work redistribution
        else:
            return RecoveryStrategy.REDISTRIBUTE_WORK
    
    async def _execute_recovery_strategy(self, recovery_operation: RecoveryOperation) -> bool:
        """Execute the chosen recovery strategy"""
        strategy = recovery_operation.recovery_strategy
        failed_workers = recovery_operation.failed_workers
        
        try:
            if strategy == RecoveryStrategy.REDISTRIBUTE_WORK:
                return await self._redistribute_work_recovery(recovery_operation)
            
            elif strategy == RecoveryStrategy.RESTART_WORKER:
                return await self._restart_worker_recovery(recovery_operation)
            
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                return await self._graceful_degradation_recovery(recovery_operation)
            
            elif strategy == RecoveryStrategy.ISOLATE_FAILURE:
                return await self._isolate_failure_recovery(recovery_operation)
            
            else:
                logging.warning(f"Unknown recovery strategy: {strategy}")
                return False
                
        except Exception as e:
            logging.error(f"Error executing recovery strategy {strategy}: {e}")
            return False
    
    async def _redistribute_work_recovery(self, recovery_operation: RecoveryOperation) -> bool:
        """Recover by redistributing work to healthy workers"""
        failed_workers = recovery_operation.failed_workers
        healthy_workers = self.health_monitor.get_healthy_workers()
        
        if not healthy_workers:
            logging.error("No healthy workers available for redistribution")
            return False
        
        # Collect incomplete work from failed workers
        all_incomplete_work = []
        for worker_id in failed_workers:
            incomplete_work = self.checkpoint_manager.get_incomplete_work(worker_id)
            all_incomplete_work.extend(incomplete_work)
            recovery_operation.affected_work.extend(incomplete_work)
        
        if not all_incomplete_work:
            logging.info("No incomplete work to redistribute")
            return True
        
        # Redistribute work
        new_distribution = self.work_redistributor.redistribute(
            all_incomplete_work, healthy_workers, 'capacity_based'
        )
        
        # Update work assignments
        for worker_id, work_items in new_distribution.items():
            current_work = self.checkpoint_manager.get_incomplete_work(worker_id)
            updated_work = current_work + work_items
            self.checkpoint_manager.update_work_assignment(worker_id, updated_work)
        
        recovery_operation.new_work_distribution = new_distribution
        recovery_operation.recovery_progress = 1.0
        
        logging.info(f"Successfully redistributed {len(all_incomplete_work)} work items to {len(healthy_workers)} workers")
        return True
    
    async def _restart_worker_recovery(self, recovery_operation: RecoveryOperation) -> bool:
        """Recover by restarting failed workers"""
        # This would typically involve actual worker process management
        # For now, we simulate worker restart
        failed_workers = recovery_operation.failed_workers
        
        restarted_workers = []
        for worker_id in failed_workers:
            # Simulate worker restart (in production, this would restart actual processes)
            logging.info(f"Attempting to restart worker {worker_id}")
            
            # Reset worker state
            self.health_monitor.update_worker_metrics(
                worker_id,
                cpu_usage=0.1,
                memory_usage=0.1,
                error_rate=0.0,
                active_tasks=0
            )
            
            # Mark as healthy
            if worker_id in self.health_monitor.worker_metrics:
                self.health_monitor.worker_metrics[worker_id].status = WorkerStatus.HEALTHY
                restarted_workers.append(worker_id)
        
        recovery_operation.recovery_progress = len(restarted_workers) / len(failed_workers)
        
        logging.info(f"Successfully restarted {len(restarted_workers)}/{len(failed_workers)} workers")
        return len(restarted_workers) > 0
    
    async def _graceful_degradation_recovery(self, recovery_operation: RecoveryOperation) -> bool:
        """Recover by gracefully degrading system performance"""
        # Reduce system load and continue with available workers
        healthy_workers = self.health_monitor.get_healthy_workers()
        
        if not healthy_workers:
            logging.error("No healthy workers available for graceful degradation")
            return False
        
        # Collect all incomplete work
        all_work = []
        for worker_id in self.checkpoint_manager.work_assignments:
            work = self.checkpoint_manager.get_incomplete_work(worker_id)
            all_work.extend(work)
        
        # Prioritize most important work (assuming work has priority)
        try:
            prioritized_work = sorted(all_work, key=lambda item: getattr(item, 'priority', 0), reverse=True)
            # Take only top 50% of work for degraded mode
            reduced_work = prioritized_work[:len(prioritized_work) // 2]
        except:
            # If no priority, take first half
            reduced_work = all_work[:len(all_work) // 2]
        
        # Redistribute reduced work
        new_distribution = self.work_redistributor.redistribute(
            reduced_work, healthy_workers, 'capacity_based'
        )
        
        # Update work assignments
        for worker_id in self.checkpoint_manager.work_assignments:
            self.checkpoint_manager.update_work_assignment(worker_id, [])
        
        for worker_id, work_items in new_distribution.items():
            self.checkpoint_manager.update_work_assignment(worker_id, work_items)
        
        recovery_operation.new_work_distribution = new_distribution
        recovery_operation.recovery_progress = 1.0
        
        logging.info(f"Graceful degradation: reduced to {len(reduced_work)} priority work items on {len(healthy_workers)} workers")
        return True
    
    async def _isolate_failure_recovery(self, recovery_operation: RecoveryOperation) -> bool:
        """Recover by isolating failed components"""
        failed_workers = recovery_operation.failed_workers
        
        # Mark failed workers as isolated (remove from active pool)
        for worker_id in failed_workers:
            if worker_id in self.health_monitor.worker_metrics:
                self.health_monitor.worker_metrics[worker_id].status = WorkerStatus.UNAVAILABLE
            
            # Clear their work assignments
            self.checkpoint_manager.update_work_assignment(worker_id, [])
        
        # Continue with remaining healthy workers
        healthy_workers = self.health_monitor.get_healthy_workers()
        
        logging.info(f"Isolated {len(failed_workers)} failed workers, continuing with {len(healthy_workers)} healthy workers")
        recovery_operation.recovery_progress = 1.0
        
        return len(healthy_workers) > 0
    
    def _classify_failure_type(self, worker_id: str) -> FailureType:
        """Classify the type of worker failure"""
        if worker_id not in self.health_monitor.worker_metrics:
            return FailureType.PROCESS_CRASH
        
        metrics = self.health_monitor.worker_metrics[worker_id]
        
        # Check timeout
        time_since_heartbeat = (datetime.now() - metrics.last_heartbeat).total_seconds()
        if time_since_heartbeat > 120:
            return FailureType.TIMEOUT
        
        # Check resource issues
        if metrics.memory_usage > 0.95:
            return FailureType.MEMORY_EXHAUSTION
        
        if metrics.cpu_usage > 0.98:
            return FailureType.CPU_OVERLOAD
        
        if metrics.error_rate > 0.5:
            return FailureType.UNHANDLED_EXCEPTION
        
        # Default classification
        return FailureType.PROCESS_CRASH
    
    def _check_recovery_operations(self):
        """Check status of active recovery operations"""
        with self.lock:
            completed_operations = []
            
            for op_id, operation in self.recovery_operations.items():
                # Check if operation should timeout
                elapsed_time = (datetime.now() - operation.start_time).total_seconds()
                
                if elapsed_time > 300:  # 5 minute timeout
                    logging.warning(f"Recovery operation {op_id} timed out")
                    completed_operations.append(op_id)
                elif operation.recovery_progress >= 1.0:
                    logging.info(f"Recovery operation {op_id} completed successfully")
                    completed_operations.append(op_id)
            
            # Remove completed operations
            for op_id in completed_operations:
                del self.recovery_operations[op_id]