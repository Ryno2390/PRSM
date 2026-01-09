"""
Model Performance Tracking and Ranking System
Real-time performance monitoring and adaptive model ranking for PRSM
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import structlog

from prsm.core.database import get_database_service
from prsm.core.redis_client import get_redis_client
from prsm.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class MetricType(str, Enum):
    """Types of performance metrics"""
    RESPONSE_TIME = "response_time"
    ACCURACY = "accuracy"  
    SUCCESS_RATE = "success_rate"
    TOKEN_EFFICIENCY = "token_efficiency"
    USER_SATISFACTION = "user_satisfaction"
    COST_EFFECTIVENESS = "cost_effectiveness"
    RELIABILITY = "reliability"
    AVAILABILITY = "availability"


class PerformanceGrade(str, Enum):
    """Performance grade categories"""
    EXCELLENT = "excellent"  # Top 10%
    GOOD = "good"           # Top 25%
    AVERAGE = "average"     # Top 50%
    POOR = "poor"          # Bottom 50%
    CRITICAL = "critical"   # Bottom 10%


@dataclass
class PerformanceMetric:
    """Individual performance metric recording"""
    model_id: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    task_id: Optional[str] = None
    user_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPerformanceProfile:
    """Comprehensive performance profile for a model"""
    model_id: str
    model_name: str
    provider: str
    
    # Core metrics (time-windowed averages)
    response_time_avg: float = 0.0
    accuracy_score: float = 0.0
    success_rate: float = 0.0
    token_efficiency: float = 0.0
    user_satisfaction: float = 0.0
    cost_effectiveness: float = 0.0
    reliability_score: float = 0.0
    availability_score: float = 0.0
    
    # Performance trends
    response_time_trend: float = 0.0  # Positive = getting slower
    accuracy_trend: float = 0.0       # Positive = improving
    usage_trend: float = 0.0          # Positive = more usage
    
    # Usage statistics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_used: Optional[datetime] = None
    
    # Ranking information
    overall_rank: int = 0
    category_rank: int = 0
    performance_grade: PerformanceGrade = PerformanceGrade.AVERAGE
    
    # Time tracking
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PerformanceTracker:
    """
    Advanced performance tracking system for model evaluation and ranking
    
    Features:
    - Real-time metric collection
    - Time-windowed performance analysis
    - Adaptive ranking with trend analysis
    - Performance degradation detection
    - Usage pattern analysis
    """
    
    def __init__(self):
        self.db_service = None
        self.redis_client = None
        
        # In-memory performance data (with TTL)
        self.model_profiles: Dict[str, ModelPerformanceProfile] = {}
        self.recent_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.global_rankings: List[str] = []  # Model IDs ranked by performance
        self.category_rankings: Dict[str, List[str]] = defaultdict(list)
        
        # Performance tracking configuration
        self.metric_window_hours = 24  # Time window for metric averaging
        self.trend_window_hours = 168  # 1 week for trend analysis
        self.ranking_update_interval = 300  # 5 minutes
        self.last_ranking_update = datetime.now(timezone.utc)
        
        # Performance thresholds
        self.thresholds = {
            MetricType.RESPONSE_TIME: {"excellent": 1.0, "good": 2.0, "average": 5.0, "poor": 10.0},
            MetricType.ACCURACY: {"excellent": 0.95, "good": 0.85, "average": 0.7, "poor": 0.5},
            MetricType.SUCCESS_RATE: {"excellent": 0.98, "good": 0.9, "average": 0.8, "poor": 0.6},
            MetricType.USER_SATISFACTION: {"excellent": 4.5, "good": 4.0, "average": 3.5, "poor": 2.5}
        }
        
    async def initialize(self):
        """Initialize database and Redis connections"""
        try:
            self.db_service = get_database_service()
            self.redis_client = get_redis_client()
            
            # Load existing performance profiles from database
            await self._load_performance_profiles()
            
            logger.info("Performance tracker initialized",
                       models_loaded=len(self.model_profiles))
                       
        except Exception as e:
            logger.error("Failed to initialize performance tracker", error=str(e))
            
    async def record_metric(self, model_id: str, metric_type: MetricType, 
                          value: float, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Record a performance metric for a model
        
        Args:
            model_id: ID of the model
            metric_type: Type of metric being recorded
            value: Metric value
            context: Additional context (task_id, user_id, etc.)
            
        Returns:
            True if metric was recorded successfully
        """
        try:
            metric = PerformanceMetric(
                model_id=model_id,
                metric_type=metric_type,
                value=value,
                timestamp=datetime.now(timezone.utc),
                task_id=context.get("task_id") if context else None,
                user_id=context.get("user_id") if context else None,
                context=context or {}
            )
            
            # Store in recent metrics buffer
            self.recent_metrics[model_id].append(metric)
            
            # Update model profile
            await self._update_model_profile(model_id, metric)
            
            # Store in database for persistence
            await self._persist_metric(metric)
            
            # Check if ranking update is needed
            await self._maybe_update_rankings()
            
            logger.debug("Performance metric recorded",
                        model_id=model_id,
                        metric_type=metric_type.value,
                        value=value)
            
            return True
            
        except Exception as e:
            logger.error("Failed to record performance metric",
                        model_id=model_id,
                        metric_type=metric_type.value,
                        error=str(e))
            return False
    
    async def get_model_performance(self, model_id: str) -> Optional[ModelPerformanceProfile]:
        """Get comprehensive performance profile for a model"""
        if model_id not in self.model_profiles:
            # Try to load from database
            await self._load_model_profile(model_id)
        
        return self.model_profiles.get(model_id)
    
    async def get_top_models(self, category: Optional[str] = None, 
                           limit: int = 10) -> List[ModelPerformanceProfile]:
        """
        Get top performing models overall or in a specific category
        
        Args:
            category: Optional category filter (e.g., "code_generation", "general")
            limit: Maximum number of models to return
            
        Returns:
            List of top performing model profiles
        """
        # Ensure rankings are up to date
        await self._update_rankings_if_needed()
        
        if category:
            ranked_ids = self.category_rankings.get(category, [])
        else:
            ranked_ids = self.global_rankings
        
        # Get profiles for top models
        top_profiles = []
        for model_id in ranked_ids[:limit]:
            profile = self.model_profiles.get(model_id)
            if profile:
                top_profiles.append(profile)
        
        return top_profiles
    
    async def get_performance_trends(self, model_id: str, 
                                   hours: int = 24) -> Dict[MetricType, List[Tuple[datetime, float]]]:
        """Get performance trends for a model over time"""
        trends = {}
        
        # Get recent metrics for the model
        model_metrics = self.recent_metrics.get(model_id, deque())
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        # Group metrics by type
        metric_groups = defaultdict(list)
        for metric in model_metrics:
            if metric.timestamp > cutoff_time:
                metric_groups[metric.metric_type].append((metric.timestamp, metric.value))
        
        # Sort by timestamp for each metric type
        for metric_type, values in metric_groups.items():
            trends[metric_type] = sorted(values, key=lambda x: x[0])
        
        return trends
    
    async def detect_performance_issues(self, model_id: str) -> List[str]:
        """Detect potential performance issues for a model"""
        issues = []
        profile = await self.get_model_performance(model_id)
        
        if not profile:
            return ["Model profile not found"]
        
        # Check for critical performance issues
        if profile.success_rate < 0.5:
            issues.append(f"Critical success rate: {profile.success_rate:.2%}")
        
        if profile.response_time_avg > 10.0:
            issues.append(f"High response time: {profile.response_time_avg:.2f}s")
        
        if profile.availability_score < 0.8:
            issues.append(f"Low availability: {profile.availability_score:.2%}")
        
        # Check for negative trends
        if profile.response_time_trend > 0.2:
            issues.append("Response time degrading")
        
        if profile.accuracy_trend < -0.1:
            issues.append("Accuracy declining")
        
        # Check for underperformance compared to peers
        global_rank = profile.overall_rank
        total_models = len(self.model_profiles)
        if total_models > 0 and global_rank > total_models * 0.8:
            issues.append(f"Poor overall ranking: {global_rank}/{total_models}")
        
        return issues
    
    async def get_performance_analytics(self) -> Dict[str, Any]:
        """Get system-wide performance analytics"""
        total_models = len(self.model_profiles)
        
        if total_models == 0:
            return {"total_models": 0}
        
        # Calculate aggregate statistics
        response_times = [p.response_time_avg for p in self.model_profiles.values() if p.response_time_avg > 0]
        success_rates = [p.success_rate for p in self.model_profiles.values()]
        accuracy_scores = [p.accuracy_score for p in self.model_profiles.values() if p.accuracy_score > 0]
        
        analytics = {
            "total_models": total_models,
            "models_with_data": len([p for p in self.model_profiles.values() if p.total_requests > 0]),
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "avg_success_rate": statistics.mean(success_rates) if success_rates else 0,
            "avg_accuracy": statistics.mean(accuracy_scores) if accuracy_scores else 0,
            "performance_distribution": {
                grade.value: len([p for p in self.model_profiles.values() if p.performance_grade == grade])
                for grade in PerformanceGrade
            },
            "category_stats": {}
        }
        
        # Calculate per-category statistics
        categories = set()
        for profile in self.model_profiles.values():
            # Extract category from model profile (would need to be stored)
            # For now, use a simple heuristic
            categories.add("general")  # Placeholder
        
        for category in categories:
            category_models = [p for p in self.model_profiles.values()]  # Filter by category
            analytics["category_stats"][category] = {
                "total_models": len(category_models),
                "avg_performance": statistics.mean([p.accuracy_score for p in category_models]) if category_models else 0
            }
        
        return analytics
    
    # Private helper methods
    
    async def _load_performance_profiles(self):
        """Load existing performance profiles from database"""
        try:
            if not self.db_service:
                return
            
            # Initialize with common models if database not available
            default_models = [
                ("gpt-4", "general", 0.92),
                ("gpt-3.5-turbo", "general", 0.85),
                ("claude-3-sonnet", "reasoning", 0.89),
                ("claude-3-haiku", "general", 0.83),
                ("gpt-4-turbo", "general", 0.94)
            ]
            
            for model_id, specialization, base_score in default_models:
                profile = ModelPerformanceProfile(
                    model_id=model_id,
                    provider="unknown",
                    specialization=specialization,
                    overall_score=base_score,
                    success_rate=base_score * 0.98,
                    avg_latency=1.5,
                    cost_efficiency=0.8,
                    quality_score=base_score,
                    reliability_score=base_score * 0.95,
                    last_updated=datetime.now(timezone.utc)
                )
                self.model_profiles[model_id] = profile
            
        except Exception as e:
            logger.error("Failed to load performance profiles", error=str(e))
    
    async def _load_model_profile(self, model_id: str):
        """Load specific model profile from database"""
        try:
            if not self.db_service:
                return
            
            # Would load specific model profile
            # For now, create new profile
            self.model_profiles[model_id] = ModelPerformanceProfile(
                model_id=model_id,
                model_name=f"Model {model_id}",
                provider="unknown"
            )
            
        except Exception as e:
            logger.error("Failed to load model profile", model_id=model_id, error=str(e))
    
    async def _update_model_profile(self, model_id: str, metric: PerformanceMetric):
        """Update model profile with new metric"""
        if model_id not in self.model_profiles:
            await self._load_model_profile(model_id)
        
        profile = self.model_profiles[model_id]
        
        # Update profile based on metric type
        if metric.metric_type == MetricType.RESPONSE_TIME:
            profile.response_time_avg = await self._calculate_windowed_average(
                model_id, MetricType.RESPONSE_TIME
            )
        elif metric.metric_type == MetricType.ACCURACY:
            profile.accuracy_score = await self._calculate_windowed_average(
                model_id, MetricType.ACCURACY
            )
        elif metric.metric_type == MetricType.SUCCESS_RATE:
            profile.success_rate = await self._calculate_windowed_average(
                model_id, MetricType.SUCCESS_RATE
            )
        
        # Update usage statistics
        profile.total_requests += 1
        if metric.metric_type == MetricType.SUCCESS_RATE and metric.value > 0.5:
            profile.successful_requests += 1
        else:
            profile.failed_requests += 1
        
        profile.last_used = metric.timestamp
        profile.updated_at = datetime.now(timezone.utc)
        
        # Calculate performance grade
        profile.performance_grade = self._calculate_performance_grade(profile)
    
    async def _calculate_windowed_average(self, model_id: str, metric_type: MetricType) -> float:
        """Calculate time-windowed average for a metric"""
        model_metrics = self.recent_metrics.get(model_id, deque())
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.metric_window_hours)
        
        values = [
            metric.value for metric in model_metrics
            if metric.metric_type == metric_type and metric.timestamp > cutoff_time
        ]
        
        return statistics.mean(values) if values else 0.0
    
    def _calculate_performance_grade(self, profile: ModelPerformanceProfile) -> PerformanceGrade:
        """Calculate overall performance grade for a model"""
        # Composite score based on key metrics
        score = 0.0
        weight_sum = 0.0
        
        # Response time (lower is better)
        if profile.response_time_avg > 0:
            response_score = max(0, 1.0 - (profile.response_time_avg / 10.0))
            score += response_score * 0.2
            weight_sum += 0.2
        
        # Accuracy
        if profile.accuracy_score > 0:
            score += profile.accuracy_score * 0.3
            weight_sum += 0.3
        
        # Success rate
        score += profile.success_rate * 0.3
        weight_sum += 0.3
        
        # Reliability
        score += profile.reliability_score * 0.2
        weight_sum += 0.2
        
        if weight_sum > 0:
            final_score = score / weight_sum
        else:
            final_score = 0.5
        
        # Map to performance grades
        if final_score >= 0.9:
            return PerformanceGrade.EXCELLENT
        elif final_score >= 0.75:
            return PerformanceGrade.GOOD
        elif final_score >= 0.5:
            return PerformanceGrade.AVERAGE
        elif final_score >= 0.25:
            return PerformanceGrade.POOR
        else:
            return PerformanceGrade.CRITICAL
    
    async def _persist_metric(self, metric: PerformanceMetric):
        """Persist metric to database"""
        try:
            if not self.db_service:
                return
            
            # Would save to database table
            # For now, just log
            logger.debug("Persisting metric", model_id=metric.model_id, type=metric.metric_type.value)
            
        except Exception as e:
            logger.error("Failed to persist metric", error=str(e))
    
    async def _maybe_update_rankings(self):
        """Update rankings if enough time has passed"""
        now = datetime.now(timezone.utc)
        if (now - self.last_ranking_update).total_seconds() > self.ranking_update_interval:
            await self._update_rankings()
    
    async def _update_rankings_if_needed(self):
        """Update rankings if they're stale"""
        await self._maybe_update_rankings()
    
    async def _update_rankings(self):
        """Update global and category-based model rankings"""
        try:
            # Calculate composite scores for all models
            model_scores = []
            
            for model_id, profile in self.model_profiles.items():
                if profile.total_requests > 0:  # Only rank models with usage data
                    composite_score = self._calculate_composite_score(profile)
                    model_scores.append((model_id, composite_score))
            
            # Sort by composite score (highest first)
            model_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Update global rankings
            self.global_rankings = [model_id for model_id, _ in model_scores]
            
            # Update ranks in profiles
            for rank, (model_id, _) in enumerate(model_scores, 1):
                self.model_profiles[model_id].overall_rank = rank
            
            # Update category rankings (placeholder - would categorize models)
            self.category_rankings["general"] = self.global_rankings.copy()
            
            self.last_ranking_update = datetime.now(timezone.utc)
            
            logger.info("Model rankings updated",
                       total_ranked=len(model_scores),
                       top_model=model_scores[0][0] if model_scores else None)
            
        except Exception as e:
            logger.error("Failed to update rankings", error=str(e))
    
    def _calculate_composite_score(self, profile: ModelPerformanceProfile) -> float:
        """Calculate composite performance score for ranking"""
        score = 0.0
        
        # Accuracy (30%)
        score += profile.accuracy_score * 0.3
        
        # Success rate (25%)
        score += profile.success_rate * 0.25
        
        # Response time (20%, inverted)
        if profile.response_time_avg > 0:
            response_score = max(0, 1.0 - (profile.response_time_avg / 10.0))
            score += response_score * 0.2
        
        # Reliability (15%)
        score += profile.reliability_score * 0.15
        
        # Usage popularity bonus (10%)
        if profile.total_requests > 100:
            popularity_score = min(1.0, profile.total_requests / 1000.0)
            score += popularity_score * 0.1
        
        return score


# Global instance
performance_tracker = PerformanceTracker()