"""
RLT Quality Monitor

Real-time monitoring and assessment of RLT teacher explanation quality.
Tracks performance metrics, detects degradation, and provides optimization
recommendations for continuous improvement.

Key Features:
- Real-time quality scoring
- Performance trend analysis
- Degradation detection and alerts
- Optimization recommendations
- Integration with PRSM monitoring stack
"""

import asyncio
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Deque
from uuid import UUID, uuid4
import structlog

from prsm.core.models import PRSMBaseModel

logger = structlog.get_logger()


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for RLT explanations"""
    # Core metrics
    explanation_coherence: float
    student_comprehension: float
    logical_flow: float
    concept_coverage: float
    
    # Performance metrics
    explanation_length: int
    generation_time: float
    reward_score: float
    
    # Context
    question_complexity: float
    domain: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    teacher_id: str = ""
    session_id: str = ""
    
    def overall_quality(self) -> float:
        """Compute weighted overall quality score"""
        return (
            self.explanation_coherence * 0.25 +
            self.student_comprehension * 0.35 +
            self.logical_flow * 0.20 +
            self.concept_coverage * 0.20
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "explanation_coherence": self.explanation_coherence,
            "student_comprehension": self.student_comprehension,
            "logical_flow": self.logical_flow,
            "concept_coverage": self.concept_coverage,
            "explanation_length": self.explanation_length,
            "generation_time": self.generation_time,
            "reward_score": self.reward_score,
            "question_complexity": self.question_complexity,
            "domain": self.domain,
            "overall_quality": self.overall_quality(),
            "timestamp": self.timestamp.isoformat(),
            "teacher_id": self.teacher_id,
            "session_id": self.session_id
        }


@dataclass
class QualityAlert:
    """Alert for quality degradation or issues"""
    alert_type: str  # "degradation", "threshold", "anomaly"
    severity: str    # "low", "medium", "high", "critical"
    message: str
    metrics: Dict[str, float]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    teacher_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "metrics": self.metrics,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
            "teacher_id": self.teacher_id
        }


@dataclass
class MonitoringConfig:
    """Configuration for RLT quality monitoring"""
    # Quality thresholds
    min_coherence_threshold: float = 0.6
    min_comprehension_threshold: float = 0.7
    min_overall_quality_threshold: float = 0.65
    
    # Performance thresholds
    max_generation_time: float = 10.0  # seconds
    max_explanation_length: int = 2048
    min_explanation_length: int = 50
    
    # Monitoring parameters
    sliding_window_size: int = 100
    trend_analysis_window: int = 50
    alert_cooldown_minutes: int = 15
    
    # Degradation detection
    degradation_threshold: float = 0.1  # 10% drop in quality
    consecutive_failures_threshold: int = 5
    anomaly_detection_sensitivity: float = 2.0  # standard deviations
    
    # Reporting
    report_frequency_minutes: int = 60
    detailed_analysis_frequency_hours: int = 24


class RLTQualityMonitor:
    """
    Real-time quality monitoring system for RLT teacher explanations.
    
    Provides continuous assessment, trend analysis, and optimization
    recommendations for maintaining high-quality explanations.
    """
    
    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
        teacher_id: str = ""
    ):
        self.config = config or MonitoringConfig()
        self.teacher_id = teacher_id
        self.logger = logger.bind(component="RLTQualityMonitor", teacher_id=teacher_id)
        
        # Metrics storage
        self.quality_history: Deque[QualityMetrics] = deque(
            maxlen=self.config.sliding_window_size
        )
        self.alerts_history: List[QualityAlert] = []
        
        # Performance tracking
        self.session_start = datetime.now(timezone.utc)
        self.total_evaluations = 0
        self.quality_trend = 0.0
        self.last_alert_time = {}
        
        # Statistical tracking
        self.running_stats = {
            "mean_quality": 0.0,
            "std_quality": 0.0,
            "mean_generation_time": 0.0,
            "total_explanations": 0,
            "quality_trend_7d": 0.0,
            "alert_count_24h": 0
        }
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task = None
    
    async def record_quality_metrics(
        self,
        explanation: str,
        question: str,
        solution: str,
        generation_time: float,
        reward_score: float,
        comprehension_score: float,
        session_id: str = ""
    ) -> QualityMetrics:
        """
        Record and analyze quality metrics for an explanation.
        
        Args:
            explanation: Generated explanation text
            question: Original question
            solution: Ground truth solution
            generation_time: Time taken to generate explanation
            reward_score: RLT reward score (r_SS + r_KL)
            comprehension_score: Student comprehension score
            session_id: Session identifier
            
        Returns:
            QualityMetrics object with computed scores
        """
        try:
            # Compute quality metrics
            coherence = self._assess_coherence(explanation)
            logical_flow = self._assess_logical_flow(explanation)
            concept_coverage = self._assess_concept_coverage(explanation, question, solution)
            question_complexity = self._assess_question_complexity(question)
            domain = self._identify_domain(question)
            
            # Create metrics object
            metrics = QualityMetrics(
                explanation_coherence=coherence,
                student_comprehension=comprehension_score,
                logical_flow=logical_flow,
                concept_coverage=concept_coverage,
                explanation_length=len(explanation),
                generation_time=generation_time,
                reward_score=reward_score,
                question_complexity=question_complexity,
                domain=domain,
                teacher_id=self.teacher_id,
                session_id=session_id
            )
            
            # Store metrics
            self.quality_history.append(metrics)
            self.total_evaluations += 1
            
            # Update running statistics
            self._update_running_stats(metrics)
            
            # Check for quality issues
            await self._check_quality_thresholds(metrics)
            await self._detect_quality_degradation()
            await self._detect_anomalies(metrics)
            
            self.logger.info(
                "Quality metrics recorded",
                overall_quality=metrics.overall_quality(),
                coherence=coherence,
                comprehension=comprehension_score,
                generation_time=generation_time
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error("Error recording quality metrics", error=str(e))
            raise
    
    async def _check_quality_thresholds(self, metrics: QualityMetrics):
        """Check if metrics fall below quality thresholds"""
        issues = []
        
        if metrics.explanation_coherence < self.config.min_coherence_threshold:
            issues.append(f"Low coherence: {metrics.explanation_coherence:.3f}")
        
        if metrics.student_comprehension < self.config.min_comprehension_threshold:
            issues.append(f"Low comprehension: {metrics.student_comprehension:.3f}")
        
        if metrics.overall_quality() < self.config.min_overall_quality_threshold:
            issues.append(f"Low overall quality: {metrics.overall_quality():.3f}")
        
        if metrics.generation_time > self.config.max_generation_time:
            issues.append(f"Slow generation: {metrics.generation_time:.2f}s")
        
        if metrics.explanation_length < self.config.min_explanation_length:
            issues.append(f"Explanation too short: {metrics.explanation_length} chars")
        
        if metrics.explanation_length > self.config.max_explanation_length:
            issues.append(f"Explanation too long: {metrics.explanation_length} chars")
        
        if issues:
            await self._create_alert(
                alert_type="threshold",
                severity="medium" if len(issues) < 3 else "high",
                message=f"Quality threshold violations: {'; '.join(issues)}",
                metrics=metrics.to_dict(),
                recommendations=self._generate_threshold_recommendations(issues)
            )
    
    async def _detect_quality_degradation(self):
        """Detect if quality is degrading over time"""
        if len(self.quality_history) < self.config.trend_analysis_window:
            return
        
        # Get recent quality scores
        recent_scores = [m.overall_quality() for m in list(self.quality_history)[-self.config.trend_analysis_window:]]
        
        # Split into earlier and later halves
        mid_point = len(recent_scores) // 2
        earlier_scores = recent_scores[:mid_point]
        later_scores = recent_scores[mid_point:]
        
        # Check for significant degradation
        earlier_mean = np.mean(earlier_scores)
        later_mean = np.mean(later_scores)
        degradation = earlier_mean - later_mean
        
        if degradation > self.config.degradation_threshold:
            await self._create_alert(
                alert_type="degradation",
                severity="high",
                message=f"Quality degradation detected: {degradation:.3f} drop over {len(recent_scores)} explanations",
                metrics={
                    "earlier_quality": earlier_mean,
                    "later_quality": later_mean,
                    "degradation": degradation,
                    "window_size": len(recent_scores)
                },
                recommendations=self._generate_degradation_recommendations(degradation)
            )
    
    async def _detect_anomalies(self, metrics: QualityMetrics):
        """Detect anomalous quality scores"""
        if len(self.quality_history) < 10:  # Need minimum history
            return
        
        # Get historical quality scores
        historical_scores = [m.overall_quality() for m in list(self.quality_history)[:-1]]
        
        if len(historical_scores) < 5:
            return
        
        # Compute statistics
        mean_quality = np.mean(historical_scores)
        std_quality = np.std(historical_scores)
        
        # Check for anomaly
        current_quality = metrics.overall_quality()
        z_score = abs(current_quality - mean_quality) / max(std_quality, 0.01)
        
        if z_score > self.config.anomaly_detection_sensitivity:
            await self._create_alert(
                alert_type="anomaly",
                severity="medium",
                message=f"Anomalous quality score detected: {current_quality:.3f} (z-score: {z_score:.2f})",
                metrics={
                    "current_quality": current_quality,
                    "historical_mean": mean_quality,
                    "historical_std": std_quality,
                    "z_score": z_score
                },
                recommendations=self._generate_anomaly_recommendations(z_score, current_quality < mean_quality)
            )
    
    async def _create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        metrics: Dict[str, Any],
        recommendations: List[str]
    ):
        """Create and store a quality alert"""
        # Check cooldown
        cooldown_key = f"{alert_type}_{severity}"
        now = datetime.now(timezone.utc)
        
        if cooldown_key in self.last_alert_time:
            time_since_last = now - self.last_alert_time[cooldown_key]
            if time_since_last < timedelta(minutes=self.config.alert_cooldown_minutes):
                return  # Skip alert due to cooldown
        
        # Create alert
        alert = QualityAlert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            metrics=metrics,
            recommendations=recommendations,
            teacher_id=self.teacher_id
        )
        
        self.alerts_history.append(alert)
        self.last_alert_time[cooldown_key] = now
        
        # Log alert
        self.logger.warning(
            "Quality alert generated",
            alert_type=alert_type,
            severity=severity,
            message=message
        )
        
        # Trigger alert handler if configured
        await self._handle_alert(alert)
    
    async def _handle_alert(self, alert: QualityAlert):
        """Handle a quality alert (can be extended for integrations)"""
        # This is where you could integrate with external monitoring systems
        # For now, we just log the alert
        pass
    
    def _update_running_stats(self, metrics: QualityMetrics):
        """Update running statistics with new metrics"""
        total = self.total_evaluations
        
        # Update mean quality
        old_mean = self.running_stats["mean_quality"]
        new_mean = (old_mean * (total - 1) + metrics.overall_quality()) / total
        self.running_stats["mean_quality"] = new_mean
        
        # Update variance (for std calculation)
        if total > 1:
            old_var = self.running_stats["std_quality"] ** 2
            new_var = ((total - 2) * old_var + (metrics.overall_quality() - old_mean) * (metrics.overall_quality() - new_mean)) / (total - 1)
            self.running_stats["std_quality"] = np.sqrt(max(0, new_var))
        
        # Update generation time
        old_time_mean = self.running_stats["mean_generation_time"]
        new_time_mean = (old_time_mean * (total - 1) + metrics.generation_time) / total
        self.running_stats["mean_generation_time"] = new_time_mean
        
        self.running_stats["total_explanations"] = total
    
    def _assess_coherence(self, explanation: str) -> float:
        """Assess explanation coherence using heuristics"""
        if not explanation:
            return 0.0
        
        coherence_score = 0.0
        
        # Length factor
        if 50 <= len(explanation) <= 1000:
            coherence_score += 0.3
        
        # Sentence structure
        sentences = [s.strip() for s in explanation.split('.') if s.strip()]
        if 2 <= len(sentences) <= 8:
            coherence_score += 0.2
        
        # Logical connectors
        connectors = ['because', 'therefore', 'since', 'thus', 'hence', 'so', 'then', 'first', 'next', 'finally']
        connector_count = sum(1 for conn in connectors if conn.lower() in explanation.lower())
        coherence_score += min(0.3, connector_count * 0.05)
        
        # Avoid repetition
        words = explanation.lower().split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        coherence_score += unique_ratio * 0.2
        
        return min(1.0, coherence_score)
    
    def _assess_logical_flow(self, explanation: str) -> float:
        """Assess logical flow of explanation"""
        if not explanation:
            return 0.0
        
        flow_score = 0.0
        sentences = [s.strip() for s in explanation.split('.') if s.strip()]
        
        # Sequential indicators
        sequence_words = ['first', 'second', 'third', 'then', 'next', 'after', 'finally', 'lastly']
        sequence_count = sum(1 for sent in sentences for word in sequence_words if word in sent.lower())
        flow_score += min(0.4, sequence_count * 0.1)
        
        # Causal relationships
        causal_words = ['because', 'since', 'therefore', 'thus', 'so', 'as a result']
        causal_count = sum(1 for sent in sentences for word in causal_words if word in sent.lower())
        flow_score += min(0.3, causal_count * 0.1)
        
        # Question to answer progression
        if any(word in explanation.lower() for word in ['to solve', 'to find', 'we need', 'let us']):
            flow_score += 0.3
        
        return min(1.0, flow_score)
    
    def _assess_concept_coverage(self, explanation: str, question: str, solution: str) -> float:
        """Assess how well explanation covers key concepts"""
        if not explanation or not question:
            return 0.0
        
        # Extract key terms from question and solution
        question_words = set(word.lower() for word in question.split() if len(word) > 3)
        solution_words = set(word.lower() for word in solution.split() if len(word) > 3)
        explanation_words = set(word.lower() for word in explanation.split())
        
        # Key concepts are important words from question/solution
        key_concepts = question_words.union(solution_words)
        
        if not key_concepts:
            return 1.0
        
        # Check coverage
        covered_concepts = key_concepts.intersection(explanation_words)
        coverage_ratio = len(covered_concepts) / len(key_concepts)
        
        return coverage_ratio
    
    def _assess_question_complexity(self, question: str) -> float:
        """Assess the complexity of a question"""
        complexity_score = 0.0
        
        # Length factor
        if len(question) > 100:
            complexity_score += 0.3
        
        # Mathematical terms
        math_terms = ['integral', 'derivative', 'equation', 'theorem', 'proof', 'calculate', 'solve']
        math_count = sum(1 for term in math_terms if term.lower() in question.lower())
        complexity_score += min(0.4, math_count * 0.1)
        
        # Multi-part questions
        if any(indicator in question.lower() for indicator in ['and', 'also', 'furthermore', 'additionally']):
            complexity_score += 0.2
        
        # Technical language
        if any(char in question for char in ['=', '+', '-', '*', '/', '^', '²', '³']):
            complexity_score += 0.1
        
        return min(1.0, complexity_score)
    
    def _identify_domain(self, question: str) -> str:
        """Identify the domain/subject of a question"""
        question_lower = question.lower()
        
        # Simple domain classification
        if any(term in question_lower for term in ['derivative', 'integral', 'calculus', 'function']):
            return "mathematics"
        elif any(term in question_lower for term in ['physics', 'force', 'energy', 'velocity']):
            return "physics"
        elif any(term in question_lower for term in ['chemistry', 'molecule', 'reaction', 'bond']):
            return "chemistry"
        elif any(term in question_lower for term in ['code', 'program', 'algorithm', 'function']):
            return "computer_science"
        else:
            return "general"
    
    def _generate_threshold_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations for threshold violations"""
        recommendations = []
        
        for issue in issues:
            if "coherence" in issue:
                recommendations.append("Improve explanation structure with clear logical flow")
                recommendations.append("Use more transitional phrases and connectors")
            elif "comprehension" in issue:
                recommendations.append("Simplify language and provide more detailed steps")
                recommendations.append("Include more examples or analogies")
            elif "generation" in issue:
                recommendations.append("Optimize model inference or reduce complexity")
            elif "short" in issue:
                recommendations.append("Provide more detailed explanations with additional context")
            elif "long" in issue:
                recommendations.append("Make explanations more concise and focused")
        
        return recommendations
    
    def _generate_degradation_recommendations(self, degradation: float) -> List[str]:
        """Generate recommendations for quality degradation"""
        recommendations = [
            "Review recent training data for quality issues",
            "Consider retraining with higher quality examples",
            "Analyze student feedback for improvement areas"
        ]
        
        if degradation > 0.2:
            recommendations.extend([
                "Consider reverting to previous model checkpoint",
                "Implement immediate quality control measures"
            ])
        
        return recommendations
    
    def _generate_anomaly_recommendations(self, z_score: float, is_low: bool) -> List[str]:
        """Generate recommendations for anomalous scores"""
        recommendations = []
        
        if is_low:
            recommendations.extend([
                "Investigate specific case that caused low quality",
                "Check if input was outside training distribution",
                "Consider additional training for similar cases"
            ])
        else:
            recommendations.extend([
                "Analyze what made this explanation exceptionally good",
                "Consider using this as a positive training example"
            ])
        
        return recommendations
    
    def get_quality_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get quality summary for specified time window"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
        
        # Filter metrics by time window
        recent_metrics = [m for m in self.quality_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"message": "No metrics in specified time window"}
        
        # Compute summary statistics
        quality_scores = [m.overall_quality() for m in recent_metrics]
        coherence_scores = [m.explanation_coherence for m in recent_metrics]
        comprehension_scores = [m.student_comprehension for m in recent_metrics]
        generation_times = [m.generation_time for m in recent_metrics]
        
        # Recent alerts
        recent_alerts = [a for a in self.alerts_history if a.timestamp >= cutoff_time]
        
        summary = {
            "time_window_hours": time_window_hours,
            "total_explanations": len(recent_metrics),
            "quality_statistics": {
                "mean": np.mean(quality_scores),
                "std": np.std(quality_scores),
                "min": np.min(quality_scores),
                "max": np.max(quality_scores),
                "median": np.median(quality_scores)
            },
            "component_averages": {
                "coherence": np.mean(coherence_scores),
                "comprehension": np.mean(comprehension_scores),
                "generation_time": np.mean(generation_times)
            },
            "alerts": {
                "total_alerts": len(recent_alerts),
                "alert_types": {},
                "severity_counts": {}
            },
            "running_stats": self.running_stats,
            "session_info": {
                "session_start": self.session_start.isoformat(),
                "total_evaluations": self.total_evaluations,
                "teacher_id": self.teacher_id
            }
        }
        
        # Alert breakdown
        for alert in recent_alerts:
            alert_type = alert.alert_type
            severity = alert.severity
            
            summary["alerts"]["alert_types"][alert_type] = summary["alerts"]["alert_types"].get(alert_type, 0) + 1
            summary["alerts"]["severity_counts"][severity] = summary["alerts"]["severity_counts"].get(severity, 0) + 1
        
        return summary
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        self.is_monitoring = True
        self.logger.info("Starting RLT quality monitoring")
        
        # Could add periodic tasks here (e.g., generate reports)
        # For now, monitoring is event-driven through record_quality_metrics
    
    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.is_monitoring = False
        self.logger.info("Stopped RLT quality monitoring")
        
        if self.monitoring_task:
            self.monitoring_task.cancel()


# Example usage and testing
async def test_quality_monitor():
    """Test function for RLT quality monitor"""
    monitor = RLTQualityMonitor(teacher_id="test_teacher")
    
    await monitor.start_monitoring()
    
    # Simulate some quality recordings
    test_cases = [
        {
            "explanation": "To find the derivative of x^2, we use the power rule. First, identify the exponent (2). Then multiply by the exponent and reduce it by 1: 2*x^(2-1) = 2x.",
            "question": "What is the derivative of x^2?",
            "solution": "2x",
            "generation_time": 1.2,
            "reward_score": 0.85,
            "comprehension_score": 0.9
        },
        {
            "explanation": "Derivative is 2x",
            "question": "What is the derivative of x^2?", 
            "solution": "2x",
            "generation_time": 0.3,
            "reward_score": 0.4,
            "comprehension_score": 0.5
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        metrics = await monitor.record_quality_metrics(
            session_id=f"test_session_{i}",
            **test_case
        )
        print(f"Test case {i+1} - Overall quality: {metrics.overall_quality():.3f}")
    
    # Get summary
    summary = monitor.get_quality_summary(time_window_hours=1)
    print("\nQuality Summary:")
    print(f"Mean quality: {summary['quality_statistics']['mean']:.3f}")
    print(f"Total alerts: {summary['alerts']['total_alerts']}")
    
    await monitor.stop_monitoring()
    return monitor


# Alias for backward compatibility
QualityMonitor = RLTQualityMonitor


if __name__ == "__main__":
    asyncio.run(test_quality_monitor())