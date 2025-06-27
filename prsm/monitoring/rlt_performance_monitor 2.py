"""
RLT Performance Monitoring System

Comprehensive performance monitoring for RLT (Reinforcement Learning Teachers) 
integration with PRSM's existing monitoring infrastructure. Provides real-time
tracking of explanation quality, student progress, teaching effectiveness, and
resource utilization.

Key Features:
- Real-time quality metrics tracking
- Student progress monitoring and analytics
- Teaching effectiveness evaluation
- Performance optimization recommendations
- Integration with existing PRSM monitoring stack
- Prometheus metrics export
- Dashboard data collection
- Alert system integration
"""

import asyncio
import time
import json
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from uuid import UUID, uuid4
import threading
from concurrent.futures import ThreadPoolExecutor

import structlog

from ..teachers.rlt.quality_monitor import QualityMonitor, QualityMetrics, MonitoringConfig
from ..teachers.rlt.student_comprehension_evaluator import ComprehensionMetrics, EvaluationConfig
from ..teachers.rlt.dense_reward_trainer import RLTTrainingConfig
from .metrics import MetricsCollector, SystemMetrics
from ..safety.monitor import SafetyMonitor
from ..safety.circuit_breaker import CircuitBreakerNetwork

logger = structlog.get_logger(__name__)


@dataclass
class RLTMetrics:
    """Real-time RLT performance metrics"""
    timestamp: float
    teacher_id: str
    session_id: str
    explanation_quality: float
    logical_coherence: float
    student_comprehension: float
    concept_coverage: float
    generation_time: float
    reward_score: float
    domain: str
    complexity: float
    student_id: Optional[str] = None
    question_id: Optional[str] = None


@dataclass
class StudentProgressMetrics:
    """Student learning progress tracking"""
    student_id: str
    domain: str
    initial_capability: float
    current_capability: float
    improvement_rate: float
    session_count: int
    total_learning_time: float
    mastery_progression: List[float]
    difficulty_progression: List[float]
    last_updated: datetime


@dataclass
class TeachingEffectivenessMetrics:
    """Teaching effectiveness evaluation metrics"""
    teacher_id: str
    domain: str
    total_sessions: int
    successful_sessions: int
    average_quality: float
    average_student_improvement: float
    average_generation_time: float
    consistency_score: float
    adaptation_success_rate: float
    cost_effectiveness: float
    last_evaluated: datetime


@dataclass
class RLTPerformanceSummary:
    """Comprehensive RLT performance summary"""
    time_window_start: datetime
    time_window_end: datetime
    total_explanations: int
    active_teachers: int
    active_students: int
    quality_statistics: Dict[str, float]
    effectiveness_statistics: Dict[str, float]
    performance_statistics: Dict[str, float]
    alerts_generated: int
    recommendations: List[str]


class RLTPerformanceAlert:
    """RLT performance alert management"""
    
    def __init__(
        self,
        alert_id: str,
        severity: str,
        message: str,
        teacher_id: Optional[str] = None,
        student_id: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        self.alert_id = alert_id
        self.severity = severity  # 'low', 'medium', 'high', 'critical'
        self.message = message
        self.teacher_id = teacher_id
        self.student_id = student_id
        self.metrics = metrics or {}
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.acknowledged = False
        self.resolved = False


class RLTPerformanceAnalyzer:
    """Advanced analytics for RLT performance optimization"""
    
    def __init__(self, monitoring_config: MonitoringConfig):
        self.config = monitoring_config
        self.analysis_cache = {}
        self.trend_models = {}
        
    def analyze_quality_trends(
        self, 
        metrics: List[RLTMetrics], 
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Analyze quality trends over time"""
        if not metrics:
            return {"trend": "no_data", "confidence": 0.0}
        
        # Filter to time window
        cutoff_time = time.time() - (time_window_hours * 3600)
        recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        if len(recent_metrics) < 2:
            return {"trend": "insufficient_data", "confidence": 0.0}
        
        # Calculate quality trend
        times = [m.timestamp for m in recent_metrics]
        qualities = [m.explanation_quality for m in recent_metrics]
        
        # Simple linear regression for trend
        n = len(qualities)
        sum_x = sum(times)
        sum_y = sum(qualities)
        sum_xy = sum(t * q for t, q in zip(times, qualities))
        sum_x2 = sum(t * t for t in times)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Determine trend
        if abs(slope) < 1e-6:
            trend = "stable"
        elif slope > 0:
            trend = "improving"
        else:
            trend = "declining"
        
        # Calculate confidence based on data consistency
        quality_variance = np.var(qualities)
        confidence = max(0.0, min(1.0, 1.0 - quality_variance))
        
        return {
            "trend": trend,
            "slope": slope,
            "confidence": confidence,
            "data_points": n,
            "quality_variance": quality_variance,
            "mean_quality": np.mean(qualities)
        }
    
    def detect_performance_anomalies(
        self, 
        current_metrics: RLTMetrics,
        historical_metrics: List[RLTMetrics]
    ) -> List[Dict[str, Any]]:
        """Detect performance anomalies using statistical analysis"""
        anomalies = []
        
        if len(historical_metrics) < 10:
            return anomalies  # Need sufficient history
        
        # Quality anomaly detection
        historical_qualities = [m.explanation_quality for m in historical_metrics]
        mean_quality = np.mean(historical_qualities)
        std_quality = np.std(historical_qualities)
        
        if abs(current_metrics.explanation_quality - mean_quality) > 2 * std_quality:
            anomalies.append({
                "type": "quality_anomaly",
                "severity": "high" if abs(current_metrics.explanation_quality - mean_quality) > 3 * std_quality else "medium",
                "message": f"Quality score {current_metrics.explanation_quality:.3f} deviates significantly from historical mean {mean_quality:.3f}",
                "metric": "explanation_quality",
                "current_value": current_metrics.explanation_quality,
                "expected_range": (mean_quality - 2 * std_quality, mean_quality + 2 * std_quality)
            })
        
        # Generation time anomaly detection
        historical_times = [m.generation_time for m in historical_metrics]
        mean_time = np.mean(historical_times)
        std_time = np.std(historical_times)
        
        if current_metrics.generation_time > mean_time + 2 * std_time:
            anomalies.append({
                "type": "performance_anomaly",
                "severity": "medium",
                "message": f"Generation time {current_metrics.generation_time:.3f}s significantly higher than historical average {mean_time:.3f}s",
                "metric": "generation_time",
                "current_value": current_metrics.generation_time,
                "expected_range": (0, mean_time + 2 * std_time)
            })
        
        return anomalies
    
    def generate_optimization_recommendations(
        self, 
        teacher_metrics: List[TeachingEffectivenessMetrics],
        student_progress: List[StudentProgressMetrics]
    ) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        if not teacher_metrics:
            return recommendations
        
        # Analyze teacher performance
        avg_quality = np.mean([tm.average_quality for tm in teacher_metrics])
        avg_effectiveness = np.mean([tm.cost_effectiveness for tm in teacher_metrics])
        
        if avg_quality < 0.7:
            recommendations.append(
                "Consider retraining teachers with quality scores below 0.7 using enhanced RLT methodology"
            )
        
        if avg_effectiveness < 0.6:
            recommendations.append(
                "Optimize teacher selection algorithms to prioritize cost-effective high-quality teachers"
            )
        
        # Analyze student progress
        if student_progress:
            avg_improvement = np.mean([sp.improvement_rate for sp in student_progress])
            
            if avg_improvement < 0.1:
                recommendations.append(
                    "Implement adaptive difficulty adjustment to improve student learning rates"
                )
        
        # Resource utilization analysis
        high_time_teachers = [tm for tm in teacher_metrics if tm.average_generation_time > 2.0]
        if len(high_time_teachers) > len(teacher_metrics) * 0.3:
            recommendations.append(
                "Optimize model inference for teachers with generation times > 2s"
            )
        
        # Consistency analysis
        inconsistent_teachers = [tm for tm in teacher_metrics if tm.consistency_score < 0.6]
        if inconsistent_teachers:
            recommendations.append(
                f"Review {len(inconsistent_teachers)} teachers with low consistency scores for potential retraining"
            )
        
        return recommendations


class RLTPerformanceMonitor:
    """
    Comprehensive RLT Performance Monitoring System
    
    Integrates with PRSM's existing monitoring infrastructure to provide:
    - Real-time quality metrics tracking
    - Student progress monitoring
    - Teaching effectiveness evaluation
    - Performance optimization recommendations
    - Alert generation and management
    - Dashboard data collection
    - Prometheus metrics export
    """
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        safety_monitor: Optional[SafetyMonitor] = None,
        circuit_breaker: Optional[CircuitBreakerNetwork] = None,
        monitoring_config: Optional[MonitoringConfig] = None
    ):
        # Core infrastructure integration
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.safety_monitor = safety_monitor or SafetyMonitor()
        self.circuit_breaker = circuit_breaker or CircuitBreakerNetwork()
        self.config = monitoring_config or MonitoringConfig()
        
        # RLT-specific monitoring components
        self.quality_monitors: Dict[str, QualityMonitor] = {}
        self.analyzer = RLTPerformanceAnalyzer(self.config)
        
        # Data storage
        self.rlt_metrics: deque = deque(maxlen=10000)  # Recent metrics
        self.student_progress: Dict[str, StudentProgressMetrics] = {}
        self.teacher_effectiveness: Dict[str, TeachingEffectivenessMetrics] = {}
        self.active_alerts: Dict[str, RLTPerformanceAlert] = {}
        
        # Performance tracking
        self.monitoring_start_time = time.time()
        self.total_explanations_monitored = 0
        self.total_alerts_generated = 0
        
        # Threading for background monitoring
        self.monitoring_thread = None
        self.shutdown_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[RLTPerformanceAlert], None]] = []
        
        logger.info(
            "RLT Performance Monitor initialized",
            config=asdict(self.config),
            integration_components=[
                "metrics_collector",
                "safety_monitor", 
                "circuit_breaker"
            ]
        )
    
    def register_teacher(self, teacher_id: str, domain: str = "general") -> QualityMonitor:
        """Register a new RLT teacher for monitoring"""
        if teacher_id not in self.quality_monitors:
            quality_monitor = QualityMonitor(
                teacher_id=teacher_id,
                config=self.config
            )
            self.quality_monitors[teacher_id] = quality_monitor
            
            # Initialize teacher effectiveness metrics
            self.teacher_effectiveness[teacher_id] = TeachingEffectivenessMetrics(
                teacher_id=teacher_id,
                domain=domain,
                total_sessions=0,
                successful_sessions=0,
                average_quality=0.0,
                average_student_improvement=0.0,
                average_generation_time=0.0,
                consistency_score=1.0,
                adaptation_success_rate=0.0,
                cost_effectiveness=0.0,
                last_evaluated=datetime.now(timezone.utc)
            )
            
            logger.info(f"Registered RLT teacher for monitoring: {teacher_id}")
        
        return self.quality_monitors[teacher_id]
    
    def register_student(self, student_id: str, domain: str, initial_capability: float = 0.5):
        """Register a new student for progress monitoring"""
        if student_id not in self.student_progress:
            self.student_progress[student_id] = StudentProgressMetrics(
                student_id=student_id,
                domain=domain,
                initial_capability=initial_capability,
                current_capability=initial_capability,
                improvement_rate=0.0,
                session_count=0,
                total_learning_time=0.0,
                mastery_progression=[initial_capability],
                difficulty_progression=[0.5],
                last_updated=datetime.now(timezone.utc)
            )
            
            logger.debug(f"Registered student for progress monitoring: {student_id}")
    
    async def record_explanation_metrics(
        self,
        teacher_id: str,
        session_id: str,
        quality_metrics: QualityMetrics,
        generation_time: float,
        student_id: Optional[str] = None,
        question_id: Optional[str] = None
    ):
        """Record metrics for an RLT explanation"""
        # Create RLT metrics record
        rlt_metrics = RLTMetrics(
            timestamp=time.time(),
            teacher_id=teacher_id,
            session_id=session_id,
            explanation_quality=quality_metrics.explanation_coherence,
            logical_coherence=quality_metrics.logical_flow,
            student_comprehension=quality_metrics.student_comprehension,
            concept_coverage=quality_metrics.concept_coverage,
            generation_time=generation_time,
            reward_score=quality_metrics.reward_score,
            domain=quality_metrics.domain,
            complexity=quality_metrics.question_complexity,
            student_id=student_id,
            question_id=question_id
        )
        
        # Store metrics
        self.rlt_metrics.append(rlt_metrics)
        self.total_explanations_monitored += 1
        
        # Update quality monitor
        if teacher_id in self.quality_monitors:
            self.quality_monitors[teacher_id].track_quality(quality_metrics)
        
        # Update teacher effectiveness
        await self._update_teacher_effectiveness(teacher_id, rlt_metrics)
        
        # Check for anomalies
        await self._check_performance_anomalies(rlt_metrics)
        
        # Update system metrics
        await self._update_system_metrics(rlt_metrics)
        
        logger.debug(
            "Recorded RLT explanation metrics",
            teacher_id=teacher_id,
            session_id=session_id,
            quality=rlt_metrics.explanation_quality,
            generation_time=generation_time
        )
    
    async def update_student_progress(
        self,
        student_id: str,
        new_capability: float,
        learning_time: float,
        difficulty: float
    ):
        """Update student learning progress"""
        if student_id not in self.student_progress:
            self.register_student(student_id, "general", 0.5)
        
        progress = self.student_progress[student_id]
        
        # Calculate improvement rate
        previous_capability = progress.current_capability
        improvement = new_capability - previous_capability
        
        # Update metrics
        progress.current_capability = new_capability
        progress.session_count += 1
        progress.total_learning_time += learning_time
        progress.mastery_progression.append(new_capability)
        progress.difficulty_progression.append(difficulty)
        progress.last_updated = datetime.now(timezone.utc)
        
        # Calculate improvement rate (smoothed)
        if len(progress.mastery_progression) > 1:
            recent_improvements = [
                progress.mastery_progression[i] - progress.mastery_progression[i-1]
                for i in range(max(1, len(progress.mastery_progression) - 5), 
                             len(progress.mastery_progression))
            ]
            progress.improvement_rate = np.mean(recent_improvements)
        
        logger.debug(
            "Updated student progress",
            student_id=student_id,
            new_capability=new_capability,
            improvement=improvement,
            session_count=progress.session_count
        )
    
    async def _update_teacher_effectiveness(self, teacher_id: str, metrics: RLTMetrics):
        """Update teacher effectiveness metrics"""
        if teacher_id not in self.teacher_effectiveness:
            return
        
        effectiveness = self.teacher_effectiveness[teacher_id]
        
        # Update session counts
        effectiveness.total_sessions += 1
        if metrics.explanation_quality > 0.7:  # Threshold for successful session
            effectiveness.successful_sessions += 1
        
        # Update running averages
        n = effectiveness.total_sessions
        effectiveness.average_quality = (
            (effectiveness.average_quality * (n - 1) + metrics.explanation_quality) / n
        )
        effectiveness.average_generation_time = (
            (effectiveness.average_generation_time * (n - 1) + metrics.generation_time) / n
        )
        
        # Calculate consistency score (1 - variance of recent quality scores)
        teacher_metrics = [m for m in self.rlt_metrics if m.teacher_id == teacher_id]
        if len(teacher_metrics) >= 5:
            recent_qualities = [m.explanation_quality for m in teacher_metrics[-10:]]
            effectiveness.consistency_score = max(0.0, 1.0 - np.var(recent_qualities))
        
        effectiveness.last_evaluated = datetime.now(timezone.utc)
    
    async def _check_performance_anomalies(self, current_metrics: RLTMetrics):
        """Check for performance anomalies and generate alerts"""
        teacher_id = current_metrics.teacher_id
        historical_metrics = [
            m for m in self.rlt_metrics 
            if m.teacher_id == teacher_id and m.timestamp < current_metrics.timestamp
        ]
        
        anomalies = self.analyzer.detect_performance_anomalies(
            current_metrics, historical_metrics
        )
        
        for anomaly in anomalies:
            alert_id = f"{teacher_id}_{anomaly['type']}_{int(time.time())}"
            alert = RLTPerformanceAlert(
                alert_id=alert_id,
                severity=anomaly['severity'],
                message=anomaly['message'],
                teacher_id=teacher_id,
                metrics=anomaly
            )
            
            await self._generate_alert(alert)
    
    async def _generate_alert(self, alert: RLTPerformanceAlert):
        """Generate and process performance alert"""
        self.active_alerts[alert.alert_id] = alert
        self.total_alerts_generated += 1
        
        # Integrate with safety monitor if critical
        if alert.severity == "critical" and alert.teacher_id:
            await self._escalate_to_safety_monitor(alert)
        
        # Trigger alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        logger.warning(
            "RLT performance alert generated",
            alert_id=alert.alert_id,
            severity=alert.severity,
            message=alert.message,
            teacher_id=alert.teacher_id
        )
    
    async def _escalate_to_safety_monitor(self, alert: RLTPerformanceAlert):
        """Escalate critical alerts to safety monitoring system"""
        try:
            # Create safety flag for critical RLT issues
            safety_context = {
                "alert_type": "rlt_performance_critical",
                "teacher_id": alert.teacher_id,
                "message": alert.message,
                "metrics": alert.metrics
            }
            
            # This would integrate with the actual SafetyMonitor
            logger.critical(
                "Escalating RLT alert to safety monitor",
                alert_id=alert.alert_id,
                teacher_id=alert.teacher_id,
                context=safety_context
            )
            
        except Exception as e:
            logger.error(f"Failed to escalate alert to safety monitor: {e}")
    
    async def _update_system_metrics(self, metrics: RLTMetrics):
        """Update system-wide metrics"""
        try:
            # Update metrics collector with RLT-specific data
            system_metrics = SystemMetrics(
                timestamp=metrics.timestamp,
                cpu_usage=0.0,  # Would be populated from actual system monitoring
                memory_usage=0.0,
                network_io=0.0,
                disk_io=0.0,
                active_connections=len(self.quality_monitors),
                custom_metrics={
                    "rlt_explanation_quality": metrics.explanation_quality,
                    "rlt_generation_time": metrics.generation_time,
                    "rlt_active_teachers": len(self.quality_monitors),
                    "rlt_total_explanations": self.total_explanations_monitored
                }
            )
            
            # This would integrate with the actual MetricsCollector
            logger.debug("Updated system metrics with RLT data")
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def add_alert_callback(self, callback: Callable[[RLTPerformanceAlert], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def get_performance_summary(
        self, 
        time_window_hours: int = 24
    ) -> RLTPerformanceSummary:
        """Get comprehensive performance summary"""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=time_window_hours)
        cutoff_timestamp = start_time.timestamp()
        
        # Filter metrics to time window
        window_metrics = [m for m in self.rlt_metrics if m.timestamp >= cutoff_timestamp]
        
        if not window_metrics:
            return RLTPerformanceSummary(
                time_window_start=start_time,
                time_window_end=end_time,
                total_explanations=0,
                active_teachers=0,
                active_students=0,
                quality_statistics={},
                effectiveness_statistics={},
                performance_statistics={},
                alerts_generated=0,
                recommendations=[]
            )
        
        # Calculate quality statistics
        qualities = [m.explanation_quality for m in window_metrics]
        quality_stats = {
            "mean": np.mean(qualities),
            "median": np.median(qualities),
            "std": np.std(qualities),
            "min": np.min(qualities),
            "max": np.max(qualities),
            "percentile_90": np.percentile(qualities, 90),
            "percentile_95": np.percentile(qualities, 95)
        }
        
        # Calculate effectiveness statistics
        generation_times = [m.generation_time for m in window_metrics]
        reward_scores = [m.reward_score for m in window_metrics]
        effectiveness_stats = {
            "avg_generation_time": np.mean(generation_times),
            "avg_reward_score": np.mean(reward_scores),
            "throughput": len(window_metrics) / time_window_hours,
            "success_rate": len([q for q in qualities if q > 0.7]) / len(qualities)
        }
        
        # Calculate performance statistics
        active_teachers = len(set(m.teacher_id for m in window_metrics))
        active_students = len(set(m.student_id for m in window_metrics if m.student_id))
        
        performance_stats = {
            "active_teachers": active_teachers,
            "active_students": active_students,
            "explanations_per_teacher": len(window_metrics) / max(active_teachers, 1),
            "avg_complexity": np.mean([m.complexity for m in window_metrics])
        }
        
        # Count recent alerts
        recent_alerts = len([
            alert for alert in self.active_alerts.values()
            if alert.timestamp >= start_time
        ])
        
        # Generate recommendations
        teacher_metrics = list(self.teacher_effectiveness.values())
        student_metrics = list(self.student_progress.values())
        recommendations = self.analyzer.generate_optimization_recommendations(
            teacher_metrics, student_metrics
        )
        
        return RLTPerformanceSummary(
            time_window_start=start_time,
            time_window_end=end_time,
            total_explanations=len(window_metrics),
            active_teachers=active_teachers,
            active_students=active_students,
            quality_statistics=quality_stats,
            effectiveness_statistics=effectiveness_stats,
            performance_statistics=performance_stats,
            alerts_generated=recent_alerts,
            recommendations=recommendations
        )
    
    def get_teacher_performance(self, teacher_id: str) -> Optional[TeachingEffectivenessMetrics]:
        """Get performance metrics for specific teacher"""
        return self.teacher_effectiveness.get(teacher_id)
    
    def get_student_progress_data(self, student_id: str) -> Optional[StudentProgressMetrics]:
        """Get progress data for specific student"""
        return self.student_progress.get(student_id)
    
    def get_active_alerts(self, severity_filter: Optional[str] = None) -> List[RLTPerformanceAlert]:
        """Get current active alerts"""
        alerts = list(self.active_alerts.values())
        if severity_filter:
            alerts = [a for a in alerts if a.severity == severity_filter]
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert acknowledged: {alert_id}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            logger.info(f"Alert resolved: {alert_id}")
            return True
        return False
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for real-time monitoring dashboard"""
        summary = self.get_performance_summary(time_window_hours=1)
        
        # Recent metrics for charts
        recent_metrics = list(self.rlt_metrics)[-100:]  # Last 100 explanations
        
        dashboard_data = {
            "timestamp": time.time(),
            "summary": asdict(summary),
            "recent_metrics": [asdict(m) for m in recent_metrics],
            "active_alerts": [asdict(a) for a in self.get_active_alerts()],
            "teacher_count": len(self.quality_monitors),
            "student_count": len(self.student_progress),
            "uptime_hours": (time.time() - self.monitoring_start_time) / 3600,
            "total_explanations": self.total_explanations_monitored,
            "total_alerts": self.total_alerts_generated
        }
        
        return dashboard_data
    
    def get_prometheus_metrics(self) -> Dict[str, Any]:
        """Get metrics in Prometheus format"""
        summary = self.get_performance_summary()
        
        prometheus_metrics = {
            # Quality metrics
            "rlt_explanation_quality_mean": summary.quality_statistics.get("mean", 0.0),
            "rlt_explanation_quality_p95": summary.quality_statistics.get("percentile_95", 0.0),
            "rlt_explanation_quality_std": summary.quality_statistics.get("std", 0.0),
            
            # Performance metrics
            "rlt_generation_time_avg": summary.effectiveness_statistics.get("avg_generation_time", 0.0),
            "rlt_throughput": summary.effectiveness_statistics.get("throughput", 0.0),
            "rlt_success_rate": summary.effectiveness_statistics.get("success_rate", 0.0),
            
            # System metrics
            "rlt_active_teachers": summary.active_teachers,
            "rlt_active_students": summary.active_students,
            "rlt_total_explanations": self.total_explanations_monitored,
            "rlt_active_alerts": len(self.active_alerts),
            
            # Uptime and health
            "rlt_monitor_uptime_seconds": time.time() - self.monitoring_start_time,
            "rlt_monitor_health": 1.0 if len(self.active_alerts) == 0 else 0.5
        }
        
        return prometheus_metrics
    
    async def start_monitoring(self):
        """Start background monitoring processes"""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.shutdown_event.clear()
            self.monitoring_thread = threading.Thread(
                target=self._background_monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
            logger.info("RLT performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring processes"""
        self.shutdown_event.set()
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        self.executor.shutdown(wait=True)
        logger.info("RLT performance monitoring stopped")
    
    def _background_monitoring_loop(self):
        """Background monitoring loop for periodic tasks"""
        while not self.shutdown_event.is_set():
            try:
                # Periodic cleanup of old alerts
                self._cleanup_old_alerts()
                
                # Periodic effectiveness recalculation
                self._recalculate_teacher_effectiveness()
                
                # Sleep before next iteration
                self.shutdown_event.wait(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in background monitoring loop: {e}")
                self.shutdown_event.wait(10)  # Wait before retrying
    
    def _cleanup_old_alerts(self):
        """Clean up old resolved alerts"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        alerts_to_remove = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if alert.resolved and alert.timestamp < cutoff_time
        ]
        
        for alert_id in alerts_to_remove:
            del self.active_alerts[alert_id]
        
        if alerts_to_remove:
            logger.debug(f"Cleaned up {len(alerts_to_remove)} old alerts")
    
    def _recalculate_teacher_effectiveness(self):
        """Recalculate teacher effectiveness metrics"""
        for teacher_id in self.teacher_effectiveness:
            teacher_metrics = [m for m in self.rlt_metrics if m.teacher_id == teacher_id]
            
            if len(teacher_metrics) >= 10:  # Minimum for meaningful calculation
                # Update cost effectiveness based on quality vs generation time
                recent_metrics = teacher_metrics[-20:]  # Last 20 explanations
                avg_quality = np.mean([m.explanation_quality for m in recent_metrics])
                avg_time = np.mean([m.generation_time for m in recent_metrics])
                
                # Cost effectiveness: quality per unit time
                cost_effectiveness = avg_quality / max(avg_time, 0.1)
                self.teacher_effectiveness[teacher_id].cost_effectiveness = cost_effectiveness


# Global RLT Performance Monitor instance
_rlt_performance_monitor: Optional[RLTPerformanceMonitor] = None


def get_rlt_performance_monitor(
    metrics_collector: Optional[MetricsCollector] = None,
    safety_monitor: Optional[SafetyMonitor] = None,
    circuit_breaker: Optional[CircuitBreakerNetwork] = None,
    monitoring_config: Optional[MonitoringConfig] = None
) -> RLTPerformanceMonitor:
    """Get or create the global RLT Performance Monitor instance"""
    global _rlt_performance_monitor
    
    if _rlt_performance_monitor is None:
        _rlt_performance_monitor = RLTPerformanceMonitor(
            metrics_collector=metrics_collector,
            safety_monitor=safety_monitor,
            circuit_breaker=circuit_breaker,
            monitoring_config=monitoring_config
        )
    
    return _rlt_performance_monitor