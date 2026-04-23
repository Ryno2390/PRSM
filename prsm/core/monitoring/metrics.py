"""
PRSM Metrics Collection and Management
=====================================

Comprehensive metrics collection system for PRSM components including
core system metrics, custom business metrics, and performance tracking.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
from abc import ABC, abstractmethod

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        start_http_server
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = Gauge = Summary = Info = None
    CollectorRegistry = generate_latest = CONTENT_TYPE_LATEST = None

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics supported"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class MetricDefinition:
    """Definition of a metric"""
    name: str
    type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    unit: Optional[str] = None
    buckets: Optional[List[float]] = None  # For histograms


@dataclass
class MetricValue:
    """A metric value with timestamp and labels"""
    name: str
    value: Union[int, float, str]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


class CustomMetric(ABC):
    """Base class for custom metrics"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.created_at = datetime.now()
    
    @abstractmethod
    async def collect(self) -> List[MetricValue]:
        """Collect metric values"""
        pass


class PRSMSystemMetrics(CustomMetric):
    """PRSM core system metrics"""
    
    def __init__(self):
        super().__init__("prsm_system", "PRSM core system metrics")
        self._query_count = 0
        self._total_ftns_used = 0.0
        self._active_sessions = 0
        self._error_count = 0
        self._start_time = time.time()
    
    def record_query(self, ftns_used: float, processing_time: float, quality_score: int):
        """Record a query execution"""
        self._query_count += 1
        self._total_ftns_used += ftns_used
    
    def record_session_start(self):
        """Record session start"""
        self._active_sessions += 1
    
    def record_session_end(self):
        """Record session end"""
        self._active_sessions = max(0, self._active_sessions - 1)
    
    def record_error(self):
        """Record an error"""
        self._error_count += 1
    
    async def collect(self) -> List[MetricValue]:
        """Collect system metrics"""
        now = datetime.now()
        uptime = time.time() - self._start_time
        
        return [
            MetricValue("prsm_queries_total", self._query_count, now),
            MetricValue("prsm_ftns_used_total", self._total_ftns_used, now, unit="ftns"),
            MetricValue("prsm_active_sessions", self._active_sessions, now),
            MetricValue("prsm_errors_total", self._error_count, now),
            MetricValue("prsm_uptime_seconds", uptime, now, unit="seconds"),
            MetricValue(
                "prsm_average_ftns_per_query", 
                self._total_ftns_used / max(1, self._query_count), 
                now, 
                unit="ftns"
            ),
        ]


class TeacherModelMetrics(CustomMetric):
    """Metrics for teacher model performance"""
    
    def __init__(self, model_name: str):
        super().__init__(f"teacher_model_{model_name}", f"Metrics for teacher model {model_name}")
        self.model_name = model_name
        self._inference_times = deque(maxlen=1000)
        self._accuracy_scores = deque(maxlen=1000)
        self._request_count = 0
        self._error_count = 0
    
    def record_inference(self, inference_time: float, accuracy_score: Optional[float] = None):
        """Record an inference"""
        self._request_count += 1
        self._inference_times.append(inference_time)
        if accuracy_score is not None:
            self._accuracy_scores.append(accuracy_score)
    
    def record_error(self):
        """Record an error"""
        self._error_count += 1
    
    async def collect(self) -> List[MetricValue]:
        """Collect teacher model metrics"""
        now = datetime.now()
        labels = {"model": self.model_name}
        
        metrics = [
            MetricValue("teacher_requests_total", self._request_count, now, labels),
            MetricValue("teacher_errors_total", self._error_count, now, labels),
        ]
        
        if self._inference_times:
            avg_time = sum(self._inference_times) / len(self._inference_times)
            max_time = max(self._inference_times)
            min_time = min(self._inference_times)
            
            metrics.extend([
                MetricValue("teacher_inference_time_avg", avg_time, now, labels, "seconds"),
                MetricValue("teacher_inference_time_max", max_time, now, labels, "seconds"),
                MetricValue("teacher_inference_time_min", min_time, now, labels, "seconds"),
            ])
        
        if self._accuracy_scores:
            avg_accuracy = sum(self._accuracy_scores) / len(self._accuracy_scores)
            metrics.append(
                MetricValue("teacher_accuracy_avg", avg_accuracy, now, labels, "percentage")
            )
        
        return metrics


class NWTNOrchestratorMetrics(CustomMetric):
    """Metrics for NWTN orchestrator"""
    
    def __init__(self):
        super().__init__("nwtn_orchestrator", "NWTN orchestrator metrics")
        self._task_queue_size = 0
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._processing_times = deque(maxlen=1000)
        self._agent_utilization = defaultdict(int)
    
    def update_queue_size(self, size: int):
        """Update task queue size"""
        self._task_queue_size = size
    
    def record_task_completion(self, processing_time: float, agent_id: str):
        """Record task completion"""
        self._completed_tasks += 1
        self._processing_times.append(processing_time)
        self._agent_utilization[agent_id] += 1
    
    def record_task_failure(self):
        """Record task failure"""
        self._failed_tasks += 1
    
    async def collect(self) -> List[MetricValue]:
        """Collect orchestrator metrics"""
        now = datetime.now()
        
        metrics = [
            MetricValue("nwtn_task_queue_size", self._task_queue_size, now),
            MetricValue("nwtn_tasks_completed_total", self._completed_tasks, now),
            MetricValue("nwtn_tasks_failed_total", self._failed_tasks, now),
        ]
        
        if self._processing_times:
            avg_time = sum(self._processing_times) / len(self._processing_times)
            metrics.append(
                MetricValue("nwtn_task_processing_time_avg", avg_time, now, unit="seconds")
            )
        
        # Agent utilization metrics
        for agent_id, count in self._agent_utilization.items():
            metrics.append(
                MetricValue(
                    "nwtn_agent_utilization", 
                    count, 
                    now, 
                    labels={"agent_id": agent_id}
                )
            )
        
        return metrics


class CollaborationTelemetryMetrics(CustomMetric):
    """Operational metrics bridge for collaboration/trust telemetry.

    This adapter converts bounded in-memory telemetry snapshots from
    transport/gossip/collaboration components into canonical metric values
    consumable by the existing monitoring + alerting stack.
    """

    def __init__(
        self,
        transport_provider: Optional[Callable[[], Any]] = None,
        gossip_provider: Optional[Callable[[], Any]] = None,
        agent_collab_provider: Optional[Callable[[], Any]] = None,
        collab_manager_provider: Optional[Callable[[], Any]] = None,
    ):
        super().__init__("collaboration_telemetry", "Collaboration and trust-path telemetry bridge")
        self._transport_provider = transport_provider
        self._gossip_provider = gossip_provider
        self._agent_collab_provider = agent_collab_provider
        self._collab_manager_provider = collab_manager_provider
        self._last_replay_nonce_total = 0

        # Bounded reason taxonomies for dashboard/query stability.
        self._allowed_handshake_failure_reasons = {
            "replay_nonce",
            "missing_public_key",
            "missing_signature",
            "invalid_signature",
            "timestamp_skew",
            "expired_challenge",
            "peer_validation_failed",
        }
        self._allowed_dispatch_failure_reasons = {
            "handler_exception",
            "missing_handler",
            "serialization_error",
            "permission_denied",
            "timeout",
        }
        self._allowed_gossip_drop_reasons = {
            "ttl_exhausted",
            "missing_subtype",
            "validation_failed",
            "deduplicated",
            "queue_overflow",
        }

    @staticmethod
    def _snapshot(component: Any) -> Dict[str, Any]:
        if component is None or not hasattr(component, "get_telemetry_snapshot"):
            return {}
        try:
            snapshot = component.get_telemetry_snapshot()
            if isinstance(snapshot, dict):
                return snapshot
        except Exception:
            pass
        return {}

    @staticmethod
    def _ratio(numerator: float, denominator: float) -> float:
        return float(numerator) / float(denominator) if denominator > 0 else 0.0

    @staticmethod
    def _bounded_reason_counts(raw: Dict[str, Any], allowed: Set[str]) -> Dict[str, int]:
        """Return bounded reason counts with unknown values grouped into `other`."""
        bounded: Dict[str, int] = defaultdict(int)
        for reason, count in (raw or {}).items():
            label = str(reason)
            if label not in allowed:
                label = "other"
            try:
                bounded[label] += int(count)
            except (TypeError, ValueError):
                bounded[label] += 0
        return dict(bounded)

    async def collect(self) -> List[MetricValue]:
        now = datetime.now()
        metrics: List[MetricValue] = []

        transport = self._snapshot(self._transport_provider() if self._transport_provider else None)
        gossip = self._snapshot(self._gossip_provider() if self._gossip_provider else None)
        collab = self._snapshot(self._agent_collab_provider() if self._agent_collab_provider else None)
        manager = self._snapshot(self._collab_manager_provider() if self._collab_manager_provider else None)

        # Transport metrics
        hs_success = int(transport.get("handshake_success_total", 0))
        hs_failure = int(transport.get("handshake_failure_total", 0))
        hs_total = hs_success + hs_failure
        dispatch_success = int(transport.get("dispatch_success_total", 0))
        dispatch_failure = int(transport.get("dispatch_failure_total", 0))
        dispatch_total = dispatch_success + dispatch_failure

        handshake_reasons = self._bounded_reason_counts(
            transport.get("handshake_failure_reasons", {}) or {},
            self._allowed_handshake_failure_reasons,
        )
        replay_nonce_total = int(handshake_reasons.get("replay_nonce", 0))
        replay_nonce_delta = max(0, replay_nonce_total - self._last_replay_nonce_total)
        self._last_replay_nonce_total = replay_nonce_total

        metrics.extend(
            [
                MetricValue("collab_transport_handshake_failure_total", hs_failure, now),
                MetricValue(
                    "collab_transport_handshake_failure_rate",
                    self._ratio(hs_failure, hs_total),
                    now,
                ),
                MetricValue("collab_transport_handshake_replay_nonce_total", replay_nonce_total, now),
                MetricValue("collab_transport_handshake_replay_nonce_delta", replay_nonce_delta, now),
                MetricValue("collab_transport_dispatch_failure_total", dispatch_failure, now),
                MetricValue(
                    "collab_transport_dispatch_failure_rate",
                    self._ratio(dispatch_failure, dispatch_total),
                    now,
                ),
            ]
        )

        for reason, count in handshake_reasons.items():
            metrics.append(
                MetricValue(
                    "collab_transport_handshake_failures_by_reason_total",
                    int(count),
                    now,
                    labels={"reason": str(reason)},
                )
            )

        dispatch_reasons = self._bounded_reason_counts(
            transport.get("dispatch_failure_reasons", {}) or {},
            self._allowed_dispatch_failure_reasons,
        )
        for reason, count in dispatch_reasons.items():
            metrics.append(
                MetricValue(
                    "collab_transport_dispatch_failures_by_reason_total",
                    int(count),
                    now,
                    labels={"reason": str(reason)},
                )
            )

        # Gossip metrics
        gossip_publish = int(gossip.get("publish_total", 0))
        gossip_forward = int(gossip.get("forward_total", 0))
        gossip_drop = int(gossip.get("drop_total", 0))
        metrics.extend(
            [
                MetricValue("collab_gossip_publish_total", gossip_publish, now),
                MetricValue("collab_gossip_forward_total", gossip_forward, now),
                MetricValue("collab_gossip_drop_total", gossip_drop, now),
                MetricValue(
                    "collab_gossip_drop_rate",
                    self._ratio(gossip_drop, gossip_drop + gossip_forward),
                    now,
                ),
            ]
        )

        gossip_drop_reasons = self._bounded_reason_counts(
            gossip.get("drop_by_reason", {}) or {},
            self._allowed_gossip_drop_reasons,
        )
        for reason, count in gossip_drop_reasons.items():
            metrics.append(
                MetricValue(
                    "collab_gossip_drop_by_reason_total",
                    int(count),
                    now,
                    labels={"reason": str(reason)},
                )
            )

        # Agent collaboration protocol metrics
        protocol_transition_total = int(collab.get("protocol_transition_total", 0))
        terminal_outcome_total = int(collab.get("terminal_outcome_total", 0))
        stalled_total = max(0, protocol_transition_total - terminal_outcome_total)
        metrics.extend(
            [
                MetricValue("collab_protocol_transition_total", protocol_transition_total, now),
                MetricValue("collab_protocol_terminal_outcome_total", terminal_outcome_total, now),
                MetricValue("collab_protocol_stalled_total", stalled_total, now),
                MetricValue(
                    "collab_protocol_stalled_ratio",
                    self._ratio(stalled_total, protocol_transition_total),
                    now,
                ),
            ]
        )

        # Collaboration manager dispatch metrics
        mgr_success = int(manager.get("dispatch_success_total", 0))
        mgr_failure = int(manager.get("dispatch_failure_total", 0))
        mgr_total = mgr_success + mgr_failure
        metrics.extend(
            [
                MetricValue("collab_manager_dispatch_success_total", mgr_success, now),
                MetricValue("collab_manager_dispatch_failure_total", mgr_failure, now),
                MetricValue(
                    "collab_manager_dispatch_failure_rate",
                    self._ratio(mgr_failure, mgr_total),
                    now,
                ),
            ]
        )

        return metrics


class MetricsRegistry:
    """Registry for managing metrics"""
    
    def __init__(self):
        self._metrics: Dict[str, CustomMetric] = {}
        self._prometheus_metrics: Dict[str, Any] = {}
        self._prometheus_registry = None
        
        if PROMETHEUS_AVAILABLE:
            self._prometheus_registry = CollectorRegistry()
        
        # Initialize core PRSM metrics
        self.register_metric(PRSMSystemMetrics())
        self.register_metric(NWTNOrchestratorMetrics())
    
    def register_metric(self, metric: CustomMetric) -> None:
        """Register a custom metric"""
        self._metrics[metric.name] = metric
        logger.info(f"Registered metric: {metric.name}")
    
    def register_teacher_model(self, model_name: str) -> TeacherModelMetrics:
        """Register metrics for a teacher model"""
        metric = TeacherModelMetrics(model_name)
        self.register_metric(metric)
        return metric

    def register_collaboration_telemetry(
        self,
        transport_provider: Optional[Callable[[], Any]] = None,
        gossip_provider: Optional[Callable[[], Any]] = None,
        agent_collab_provider: Optional[Callable[[], Any]] = None,
        collab_manager_provider: Optional[Callable[[], Any]] = None,
    ) -> CollaborationTelemetryMetrics:
        """Register collaboration telemetry bridge metrics."""
        metric = CollaborationTelemetryMetrics(
            transport_provider=transport_provider,
            gossip_provider=gossip_provider,
            agent_collab_provider=agent_collab_provider,
            collab_manager_provider=collab_manager_provider,
        )
        self.register_metric(metric)
        return metric
    
    def get_metric(self, name: str) -> Optional[CustomMetric]:
        """Get a metric by name"""
        return self._metrics.get(name)
    
    def list_metrics(self) -> List[str]:
        """List all registered metric names"""
        return list(self._metrics.keys())
    
    async def collect_all(self) -> List[MetricValue]:
        """Collect all metric values"""
        all_values = []
        
        for metric in self._metrics.values():
            try:
                values = await metric.collect()
                all_values.extend(values)
            except Exception as e:
                logger.error(f"Failed to collect metric {metric.name}: {e}")
        
        return all_values
    
    def create_prometheus_metric(self, definition: MetricDefinition) -> Optional[Any]:
        """Create a Prometheus metric"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available")
            return None
        
        try:
            if definition.type == MetricType.COUNTER:
                metric = Counter(
                    definition.name,
                    definition.description,
                    definition.labels,
                    registry=self._prometheus_registry
                )
            elif definition.type == MetricType.GAUGE:
                metric = Gauge(
                    definition.name,
                    definition.description,
                    definition.labels,
                    registry=self._prometheus_registry
                )
            elif definition.type == MetricType.HISTOGRAM:
                metric = Histogram(
                    definition.name,
                    definition.description,
                    definition.labels,
                    buckets=definition.buckets,
                    registry=self._prometheus_registry
                )
            elif definition.type == MetricType.SUMMARY:
                metric = Summary(
                    definition.name,
                    definition.description,
                    definition.labels,
                    registry=self._prometheus_registry
                )
            elif definition.type == MetricType.INFO:
                metric = Info(
                    definition.name,
                    definition.description,
                    registry=self._prometheus_registry
                )
            else:
                logger.error(f"Unknown metric type: {definition.type}")
                return None
            
            self._prometheus_metrics[definition.name] = metric
            return metric
            
        except Exception as e:
            logger.error(f"Failed to create Prometheus metric {definition.name}: {e}")
            return None
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus formatted metrics"""
        if not PROMETHEUS_AVAILABLE or not self._prometheus_registry:
            return "# Prometheus client not available\n"
        
        try:
            return generate_latest(self._prometheus_registry).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to generate Prometheus metrics: {e}")
            return f"# Error generating metrics: {e}\n"

    def observe_metric_values(self, values: List[MetricValue]) -> None:
        """Bridge collected metric values into Prometheus exporter registry.

        This keeps custom metrics (including collaboration telemetry) visible on the
        shared `/metrics` exporter path without changing runtime behavior when
        Prometheus is unavailable.
        """
        if not PROMETHEUS_AVAILABLE:
            return

        for metric_value in values:
            try:
                label_keys = tuple(sorted((metric_value.labels or {}).keys()))
                metric = self._prometheus_metrics.get(metric_value.name)

                if metric is None:
                    metric = Gauge(
                        metric_value.name,
                        f"PRSM custom metric bridge for {metric_value.name}",
                        list(label_keys),
                        registry=self._prometheus_registry,
                    )
                    self._prometheus_metrics[metric_value.name] = metric

                if label_keys:
                    label_values = [str((metric_value.labels or {}).get(k, "none")) for k in label_keys]
                    metric.labels(*label_values).set(float(metric_value.value))
                else:
                    metric.set(float(metric_value.value))
            except Exception as exc:
                logger.debug("Failed to bridge metric '%s' into Prometheus: %s", metric_value.name, exc)


class MetricsCollector:
    """Main metrics collection and management system"""
    
    def __init__(self, collection_interval: float = 10.0):
        self.collection_interval = collection_interval
        self.registry = MetricsRegistry()
        self.is_collecting = False
        self._collection_task: Optional[asyncio.Task] = None
        self._metrics_history: deque = deque(maxlen=10000)
        self._prometheus_server_port: Optional[int] = None
        
        # Initialize core Prometheus metrics
        self._init_prometheus_metrics()
    
    def _init_prometheus_metrics(self):
        """Initialize core Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Core PRSM metrics
        core_metrics = [
            MetricDefinition(
                "prsm_queries_total",
                MetricType.COUNTER,
                "Total number of queries processed",
                ["user_id", "status"]
            ),
            MetricDefinition(
                "prsm_query_duration_seconds",
                MetricType.HISTOGRAM,
                "Query processing duration in seconds",
                ["user_id"],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
            ),
            MetricDefinition(
                "prsm_ftns_used",
                MetricType.HISTOGRAM,
                "FTNS tokens used per query",
                ["user_id"],
                buckets=[10, 25, 50, 100, 200, 500, 1000]
            ),
            MetricDefinition(
                "prsm_quality_score",
                MetricType.HISTOGRAM,
                "Quality score of responses",
                ["user_id"],
                buckets=[50, 60, 70, 80, 85, 90, 95, 100]
            ),
            MetricDefinition(
                "prsm_active_sessions",
                MetricType.GAUGE,
                "Number of active sessions"
            ),
            MetricDefinition(
                "prsm_errors_total",
                MetricType.COUNTER,
                "Total number of errors",
                ["error_type", "component"]
            ),
        ]
        
        for metric_def in core_metrics:
            self.registry.create_prometheus_metric(metric_def)

        # Collaboration bridge metrics (P3 operational observability).
        collaboration_metrics = [
            MetricDefinition(
                "collab_transport_handshake_failure_rate",
                MetricType.GAUGE,
                "Transport handshake failure rate",
            ),
            MetricDefinition(
                "collab_transport_handshake_replay_nonce_delta",
                MetricType.GAUGE,
                "Delta of replay-nonce rejections between collection intervals",
            ),
            MetricDefinition(
                "collab_transport_dispatch_failure_rate",
                MetricType.GAUGE,
                "Transport dispatch failure rate",
            ),
            MetricDefinition(
                "collab_protocol_stalled_total",
                MetricType.GAUGE,
                "Derived count of protocol transitions lacking terminal outcomes",
            ),
            MetricDefinition(
                "collab_protocol_stalled_ratio",
                MetricType.GAUGE,
                "Derived ratio of stalled protocol transitions",
            ),
            MetricDefinition(
                "collab_transport_handshake_failures_by_reason_total",
                MetricType.GAUGE,
                "Bounded handshake failure reason totals",
                ["reason"],
            ),
            MetricDefinition(
                "collab_transport_dispatch_failures_by_reason_total",
                MetricType.GAUGE,
                "Bounded dispatch failure reason totals",
                ["reason"],
            ),
            MetricDefinition(
                "collab_gossip_drop_by_reason_total",
                MetricType.GAUGE,
                "Bounded gossip drop reason totals",
                ["reason"],
            ),
        ]
        for metric_def in collaboration_metrics:
            self.registry.create_prometheus_metric(metric_def)

    async def collect_once(self) -> List[MetricValue]:
        """Collect one metrics snapshot and bridge it to Prometheus export state."""
        metrics = await self.registry.collect_all()
        for metric in metrics:
            self._metrics_history.append(metric)
        self.registry.observe_metric_values(metrics)
        return metrics
    
    async def start_collection(self) -> None:
        """Start metrics collection"""
        if self.is_collecting:
            logger.warning("Metrics collection already started")
            return
        
        self.is_collecting = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info(f"Started metrics collection with {self.collection_interval}s interval")
    
    async def stop_collection(self) -> None:
        """Stop metrics collection"""
        if not self.is_collecting:
            return
        
        self.is_collecting = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped metrics collection")
    
    async def _collection_loop(self) -> None:
        """Main collection loop"""
        while self.is_collecting:
            try:
                metrics = await self.collect_once()
                
                logger.debug(f"Collected {len(metrics)} metric values")
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
            
            try:
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
    
    def record_query(self, user_id: str, processing_time: float, ftns_used: float, 
                    quality_score: int, status: str = "success") -> None:
        """Record a query execution"""
        # Update custom metrics
        system_metric = self.registry.get_metric("prsm_system")
        if isinstance(system_metric, PRSMSystemMetrics):
            system_metric.record_query(ftns_used, processing_time, quality_score)
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE and "prsm_queries_total" in self.registry._prometheus_metrics:
            self.registry._prometheus_metrics["prsm_queries_total"].labels(
                user_id=user_id, status=status
            ).inc()
            
            self.registry._prometheus_metrics["prsm_query_duration_seconds"].labels(
                user_id=user_id
            ).observe(processing_time)
            
            self.registry._prometheus_metrics["prsm_ftns_used"].labels(
                user_id=user_id
            ).observe(ftns_used)
            
            self.registry._prometheus_metrics["prsm_quality_score"].labels(
                user_id=user_id
            ).observe(quality_score)
    
    def record_error(self, error_type: str, component: str) -> None:
        """Record an error"""
        # Update custom metrics
        system_metric = self.registry.get_metric("prsm_system")
        if isinstance(system_metric, PRSMSystemMetrics):
            system_metric.record_error()
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE and "prsm_errors_total" in self.registry._prometheus_metrics:
            self.registry._prometheus_metrics["prsm_errors_total"].labels(
                error_type=error_type, component=component
            ).inc()

    def register_collaboration_telemetry(
        self,
        transport_provider: Optional[Callable[[], Any]] = None,
        gossip_provider: Optional[Callable[[], Any]] = None,
        agent_collab_provider: Optional[Callable[[], Any]] = None,
        collab_manager_provider: Optional[Callable[[], Any]] = None,
    ) -> CollaborationTelemetryMetrics:
        """Register collaboration telemetry bridge metrics.

        Providers are zero-arg callables returning component instances exposing
        `get_telemetry_snapshot()`.
        """
        return self.registry.register_collaboration_telemetry(
            transport_provider=transport_provider,
            gossip_provider=gossip_provider,
            agent_collab_provider=agent_collab_provider,
            collab_manager_provider=collab_manager_provider,
        )
    
    def get_recent_metrics(self, minutes: int = 10) -> List[MetricValue]:
        """Get recent metrics within specified time window"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            metric for metric in self._metrics_history 
            if metric.timestamp >= cutoff_time
        ]
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics"""
        recent_metrics = self.get_recent_metrics(60)  # Last hour
        
        summary = {
            "total_metrics_collected": len(self._metrics_history),
            "recent_metrics_count": len(recent_metrics),
            "collection_active": self.is_collecting,
            "collection_interval": self.collection_interval,
            "registered_metrics": self.registry.list_metrics(),
            "prometheus_available": PROMETHEUS_AVAILABLE,
        }
        
        if recent_metrics:
            # Calculate some basic statistics
            query_metrics = [m for m in recent_metrics if "queries" in m.name]
            if query_metrics:
                summary["recent_query_count"] = len(query_metrics)
        
        return summary
    
    def start_prometheus_server(self, port: int = 8000) -> None:
        """Start Prometheus metrics server"""
        if not PROMETHEUS_AVAILABLE:
            logger.error("Prometheus client not available")
            return
        
        try:
            start_http_server(port, registry=self.registry._prometheus_registry)
            self._prometheus_server_port = port
            logger.info(f"Started Prometheus metrics server on port {port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus formatted metrics"""
        return self.registry.get_prometheus_metrics()


# Aliases for backward compatibility
SystemMetrics = PRSMSystemMetrics
