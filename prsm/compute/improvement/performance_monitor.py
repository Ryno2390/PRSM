"""
PRSM Performance Monitor
Tracks model metrics, identifies improvement opportunities, and benchmarks against baselines
"""

import asyncio
import statistics
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4

from prsm.core.config import settings
from prsm.core.models import (
    PerformanceMetric, MetricType, ComparisonReport, 
    ImprovementOpportunity, ImprovementType, PerformanceAnalysis
)
from prsm.core.safety.monitor import SafetyMonitor
from prsm.economy.tokenomics.ftns_service import get_ftns_service


# === Performance Monitoring Configuration ===

# Monitoring settings
METRIC_RETENTION_DAYS = int(getattr(settings, "PRSM_METRIC_RETENTION_DAYS", 30))
ANALYSIS_WINDOW_HOURS = int(getattr(settings, "PRSM_ANALYSIS_WINDOW_HOURS", 24))
MIN_SAMPLES_FOR_ANALYSIS = int(getattr(settings, "PRSM_MIN_SAMPLES_ANALYSIS", 10))
IMPROVEMENT_THRESHOLD = float(getattr(settings, "PRSM_IMPROVEMENT_THRESHOLD", 0.05))  # 5%

# Baseline settings  
BASELINE_CONFIDENCE_THRESHOLD = float(getattr(settings, "PRSM_BASELINE_CONFIDENCE", 0.8))
STATISTICAL_SIGNIFICANCE_LEVEL = float(getattr(settings, "PRSM_SIGNIFICANCE_LEVEL", 0.05))
ANOMALY_DETECTION_THRESHOLD = float(getattr(settings, "PRSM_ANOMALY_THRESHOLD", 2.0))  # Standard deviations

# Improvement opportunity settings
HIGH_PRIORITY_THRESHOLD = float(getattr(settings, "PRSM_HIGH_PRIORITY_THRESHOLD", 0.8))
MEDIUM_PRIORITY_THRESHOLD = float(getattr(settings, "PRSM_MEDIUM_PRIORITY_THRESHOLD", 0.5))


class PerformanceMonitor:
    """
    Performance monitoring system for PRSM models and components
    Tracks metrics, identifies improvement opportunities, and benchmarks performance
    """
    
    def __init__(self):
        # Metric storage
        self.metrics_store: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.baselines: Dict[str, Dict[MetricType, float]] = {}
        self.analysis_cache: Dict[str, PerformanceAnalysis] = {}
        
        # Safety integration
        self.safety_monitor = SafetyMonitor()
        
        # Performance tracking
        self.monitoring_stats = {
            "total_metrics_tracked": 0,
            "models_monitored": set(),
            "improvement_opportunities_identified": 0,
            "analyses_performed": 0,
            "baseline_comparisons": 0
        }
        
        # Synchronization
        self._metrics_lock = asyncio.Lock()
        self._baselines_lock = asyncio.Lock()
        
        print("🔍 PerformanceMonitor initialized")
    
    
    async def track_model_metrics(self, model_id: str, performance_data: Dict[str, Any]) -> bool:
        """
        Track performance metrics for a specific model
        
        Args:
            model_id: Identifier for the model being tracked
            performance_data: Dictionary containing metric values
            
        Returns:
            True if metrics were successfully tracked
        """
        try:
            async with self._metrics_lock:
                metrics_created = []
                
                # Process each metric in the performance data
                for metric_name, metric_value in performance_data.items():
                    # Map metric names to types
                    metric_type = self._map_metric_name_to_type(metric_name)
                    
                    if metric_type and isinstance(metric_value, (int, float)):
                        # Get baseline for comparison
                        baseline_value = await self._get_baseline_value(model_id, metric_type)
                        
                        # Set baseline if this is the first metric of this type
                        if baseline_value is None:
                            await self._set_baseline_value(model_id, metric_type, float(metric_value))
                            baseline_value = float(metric_value)
                        
                        # Calculate improvement percentage
                        improvement_pct = None
                        if baseline_value is not None:
                            improvement_pct = ((metric_value - baseline_value) / baseline_value) * 100
                        
                        # Create performance metric
                        metric = PerformanceMetric(
                            model_id=model_id,
                            metric_type=metric_type,
                            value=float(metric_value),
                            unit=self._get_metric_unit(metric_type),
                            baseline_value=baseline_value,
                            improvement_percentage=improvement_pct,
                            context=performance_data.get("context", {})
                        )
                        
                        # Store metric
                        self.metrics_store[model_id].append(metric)
                        metrics_created.append(metric)
                        
                        # Update monitoring stats
                        self.monitoring_stats["total_metrics_tracked"] += 1
                        self.monitoring_stats["models_monitored"].add(model_id)
                
                print(f"📊 Tracked {len(metrics_created)} metrics for model {model_id}")
                
                # Trigger improvement opportunity analysis if enough metrics
                if len(self.metrics_store[model_id]) >= MIN_SAMPLES_FOR_ANALYSIS:
                    await self._trigger_opportunity_analysis(model_id)
                
                return True
                
        except Exception as e:
            print(f"❌ Error tracking metrics for model {model_id}: {str(e)}")
            return False
    
    
    async def identify_improvement_opportunities(self, historical_data: List[Dict[str, Any]]) -> List[ImprovementOpportunity]:
        """
        Identify improvement opportunities based on historical performance data
        
        Args:
            historical_data: List of historical performance data points
            
        Returns:
            List of identified improvement opportunities
        """
        try:
            opportunities = []
            
            # Group data by model
            model_data = defaultdict(list)
            for data_point in historical_data:
                model_id = data_point.get("model_id")
                if model_id:
                    model_data[model_id].append(data_point)
            
            # Analyze each model's data
            for model_id, data_points in model_data.items():
                # Standard metric-based opportunities
                model_opportunities = await self._analyze_model_opportunities(model_id, data_points)
                opportunities.extend(model_opportunities)

                # Economic efficiency-based opportunities
                # Extract all metrics for this model across data points
                model_metrics = defaultdict(list)
                for dp in data_points:
                    dp_metrics = dp.get("metrics", {})
                    for k, v in dp_metrics.items():
                        model_metrics[k].append(v)
                
                econ_opportunity = await self._analyze_economic_efficiency(model_id, model_metrics)
                if econ_opportunity:
                    opportunities.append(econ_opportunity)
            
            # Sort by priority score
            opportunities.sort(key=lambda x: x.priority_score, reverse=True)
            
            # Update stats
            self.monitoring_stats["improvement_opportunities_identified"] += len(opportunities)
            
            print(f"🎯 Identified {len(opportunities)} improvement opportunities")
            
            return opportunities
            
        except Exception as e:
            print(f"❌ Error identifying improvement opportunities: {str(e)}")
            return []
    
    
    async def benchmark_against_baselines(self, model_id: str) -> ComparisonReport:
        """
        Benchmark model performance against established baselines
        
        Args:
            model_id: Model to benchmark
            
        Returns:
            Comprehensive comparison report
        """
        try:
            # Get recent metrics for the model
            recent_metrics = await self._get_recent_metrics(model_id, hours=ANALYSIS_WINDOW_HOURS)
            
            if not recent_metrics:
                raise ValueError(f"No recent metrics found for model {model_id}")
            
            # Get baseline model for comparison
            baseline_model_id = await self._get_baseline_model(model_id)
            
            # Compare metrics
            comparison_metrics = []
            significant_improvements = []
            regressions = []
            improvement_scores = []
            
            # Group metrics by type
            metrics_by_type = defaultdict(list)
            for metric in recent_metrics:
                metrics_by_type[metric.metric_type].append(metric)
            
            # Compare each metric type
            for metric_type, metrics in metrics_by_type.items():
                # Calculate average performance
                avg_value = statistics.mean([m.value for m in metrics])
                
                # Get baseline value
                baseline_value = await self._get_baseline_value(model_id, metric_type)
                
                if baseline_value is not None:
                    # Calculate improvement
                    improvement_pct = ((avg_value - baseline_value) / baseline_value) * 100
                    
                    # Create comparison metric
                    comparison_metric = PerformanceMetric(
                        model_id=model_id,
                        metric_type=metric_type,
                        value=avg_value,
                        unit=self._get_metric_unit(metric_type),
                        baseline_value=baseline_value,
                        improvement_percentage=improvement_pct
                    )
                    comparison_metrics.append(comparison_metric)
                    
                    # Categorize improvement
                    if improvement_pct > IMPROVEMENT_THRESHOLD * 100:
                        significant_improvements.append(f"{metric_type.value}: +{improvement_pct:.1f}%")
                        improvement_scores.append(improvement_pct / 100)
                    elif improvement_pct < -IMPROVEMENT_THRESHOLD * 100:
                        regressions.append(f"{metric_type.value}: {improvement_pct:.1f}%")
                        improvement_scores.append(improvement_pct / 100)
                    else:
                        improvement_scores.append(0.0)
            
            # Calculate overall improvement
            overall_improvement = statistics.mean(improvement_scores) if improvement_scores else 0.0
            
            # Generate recommendation
            recommendation = await self._generate_recommendation(
                overall_improvement, significant_improvements, regressions
            )
            
            # Calculate confidence score
            confidence_score = min(1.0, len(comparison_metrics) / 5.0)  # More metrics = higher confidence
            
            # Create comparison report
            report = ComparisonReport(
                model_id=model_id,
                baseline_model_id=baseline_model_id,
                comparison_metrics=comparison_metrics,
                overall_improvement=overall_improvement,
                significant_improvements=significant_improvements,
                regressions=regressions,
                recommendation=recommendation,
                confidence_score=confidence_score
            )
            
            # Update stats
            self.monitoring_stats["baseline_comparisons"] += 1
            
            print(f"📈 Generated benchmark report for {model_id}: {overall_improvement:.1%} overall improvement")
            
            return report
            
        except Exception as e:
            print(f"❌ Error benchmarking model {model_id}: {str(e)}")
            # Return empty report on error
            return ComparisonReport(
                model_id=model_id,
                baseline_model_id="unknown",
                comparison_metrics=[],
                overall_improvement=0.0,
                significant_improvements=[],
                regressions=[],
                recommendation="Error generating benchmark report",
                confidence_score=0.0
            )
    
    
    async def generate_performance_analysis(self, model_id: str, period_hours: int = 24) -> PerformanceAnalysis:
        """
        Generate comprehensive performance analysis for a model
        
        Args:
            model_id: Model to analyze
            period_hours: Analysis period in hours
            
        Returns:
            Comprehensive performance analysis
        """
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=period_hours)
            
            # Get metrics in the analysis period
            metrics = await self._get_metrics_in_period(model_id, start_time, end_time)
            
            if not metrics:
                raise ValueError(f"No metrics found for model {model_id} in the specified period")
            
            # Calculate trends
            trends = await self._calculate_trends(metrics)
            
            # Detect anomalies
            anomalies = await self._detect_anomalies(metrics)
            
            # Identify improvement opportunities
            historical_data = [
                {
                    "model_id": model_id,
                    "metrics": {m.metric_type.value: m.value for m in metrics},
                    "timestamp": max([m.timestamp for m in metrics])
                }
            ]
            improvement_opportunities = await self.identify_improvement_opportunities(historical_data)
            
            # Calculate overall health score
            health_score = await self._calculate_health_score(metrics, trends, anomalies)
            
            # Generate recommendations
            recommendations = await self._generate_analysis_recommendations(
                trends, anomalies, improvement_opportunities
            )
            
            # Create analysis
            analysis = PerformanceAnalysis(
                model_id=model_id,
                analysis_period_start=start_time,
                analysis_period_end=end_time,
                metrics_analyzed=metrics,
                trends=trends,
                anomalies=anomalies,
                improvement_opportunities=improvement_opportunities,
                overall_health_score=health_score,
                recommendations=recommendations
            )
            
            # Cache analysis
            self.analysis_cache[model_id] = analysis
            
            # Update stats
            self.monitoring_stats["analyses_performed"] += 1
            
            print(f"🔍 Generated performance analysis for {model_id}: {health_score:.2f} health score")
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error generating performance analysis for {model_id}: {str(e)}")
            return PerformanceAnalysis(
                model_id=model_id,
                analysis_period_start=start_time,
                analysis_period_end=end_time,
                metrics_analyzed=[],
                trends={},
                anomalies=[],
                improvement_opportunities=[],
                overall_health_score=0.0,
                recommendations=["Error generating analysis"]
            )
    
    
    async def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get current monitoring statistics"""
        return {
            **self.monitoring_stats,
            "models_monitored_count": len(self.monitoring_stats["models_monitored"]),
            "models_monitored": list(self.monitoring_stats["models_monitored"]),
            "metrics_store_size": sum(len(deque_obj) for deque_obj in self.metrics_store.values()),
            "baselines_count": len(self.baselines),
            "cached_analyses": len(self.analysis_cache),
            "configuration": {
                "retention_days": METRIC_RETENTION_DAYS,
                "analysis_window_hours": ANALYSIS_WINDOW_HOURS,
                "min_samples": MIN_SAMPLES_FOR_ANALYSIS,
                "improvement_threshold": IMPROVEMENT_THRESHOLD
            }
        }
    
    
    # === Private Helper Methods ===
    
    def _map_metric_name_to_type(self, metric_name: str) -> Optional[MetricType]:
        """Map string metric names to MetricType enum values"""
        mapping = {
            "accuracy": MetricType.ACCURACY,
            "precision": MetricType.ACCURACY,
            "recall": MetricType.ACCURACY,
            "f1_score": MetricType.ACCURACY,
            "latency": MetricType.LATENCY,
            "response_time": MetricType.LATENCY,
            "processing_time": MetricType.LATENCY,
            "throughput": MetricType.THROUGHPUT,
            "requests_per_second": MetricType.THROUGHPUT,
            "qps": MetricType.THROUGHPUT,
            "memory_usage": MetricType.RESOURCE_USAGE,
            "cpu_usage": MetricType.RESOURCE_USAGE,
            "disk_usage": MetricType.RESOURCE_USAGE,
            "error_rate": MetricType.ERROR_RATE,
            "failure_rate": MetricType.ERROR_RATE,
            "success_rate": MetricType.ERROR_RATE,
            "user_satisfaction": MetricType.USER_SATISFACTION,
            "rating": MetricType.USER_SATISFACTION,
            "cost": MetricType.COST_EFFICIENCY,
            "cost_per_request": MetricType.COST_EFFICIENCY,
            "efficiency": MetricType.COST_EFFICIENCY,
            "photon_cost": MetricType.COST_EFFICIENCY,
            "economic_efficiency": MetricType.ECONOMIC_EFFICIENCY
        }
        return mapping.get(metric_name.lower())
    
    
    def _get_metric_unit(self, metric_type: MetricType) -> str:
        """Get appropriate unit for metric type"""
        units = {
            MetricType.ACCURACY: "percentage",
            MetricType.LATENCY: "milliseconds", 
            MetricType.THROUGHPUT: "requests/second",
            MetricType.RESOURCE_USAGE: "percentage",
            MetricType.ERROR_RATE: "percentage",
            MetricType.USER_SATISFACTION: "score",
            MetricType.COST_EFFICIENCY: "photons",
            MetricType.ECONOMIC_EFFICIENCY: "photons/inference"
        }
        return units.get(metric_type, "units")

    async def _analyze_economic_efficiency(self, model_id: str, metrics: Dict[str, List[float]]) -> Optional[ImprovementOpportunity]:
        """
        Analyze the economic efficiency of a model.
        Detects if a model is too expensive (in Photons/FTNS) for its performance.
        """
        try:
            # We need accuracy, latency, and cost to calculate efficiency
            acc_list = metrics.get("accuracy", [])
            lat_list = metrics.get("latency", [])
            cost_list = metrics.get("cost", metrics.get("photon_cost", []))

            if not (acc_list and lat_list and cost_list):
                return None

            latest_acc = acc_list[-1]
            latest_lat = lat_list[-1]
            latest_cost = cost_list[-1]

            # Efficiency = Accuracy / (Cost * Latency)
            # We normalize Latency to seconds for the denominator
            efficiency = latest_acc / (max(0.001, latest_cost) * max(0.001, latest_lat / 1000.0))
            
            # Threshold for efficiency alert (photons are expensive!)
            # If cost > 50 photons per 90% accurate inference, consider it a regression
            EFFICIENCY_THRESHOLD = 0.05 
            
            if efficiency < EFFICIENCY_THRESHOLD:
                return ImprovementOpportunity(
                    improvement_type=ImprovementType.ARCHITECTURE,
                    target_component=model_id,
                    current_performance=efficiency,
                    expected_improvement=0.5, # Distillation to SSM usually gives 50%+ efficiency boost
                    confidence=0.85,
                    implementation_cost=0.3,
                    priority_score=0.9, # High priority: saving Photons is key to the network economy
                    description=f"Economic Efficiency regression detected ({efficiency:.4f}). High Photon cost per inference. Recommend autonomous distillation to SSM/Mamba architecture.",
                    supporting_data={
                        "efficiency_score": efficiency,
                        "photon_cost": latest_cost,
                        "accuracy": latest_acc,
                        "latency_ms": latest_lat,
                        "recommendation": "ssm_distillation"
                    }
                )
            
            return None
        except Exception as e:
            print(f"⚠️ Error analyzing economic efficiency: {e}")
            return None
    
    
    async def _get_baseline_value(self, model_id: str, metric_type: MetricType) -> Optional[float]:
        """Get baseline value for a model and metric type"""
        async with self._baselines_lock:
            model_baselines = self.baselines.get(model_id, {})
            return model_baselines.get(metric_type)
    
    
    async def _set_baseline_value(self, model_id: str, metric_type: MetricType, value: float):
        """Set baseline value for a model and metric type"""
        async with self._baselines_lock:
            if model_id not in self.baselines:
                self.baselines[model_id] = {}
            self.baselines[model_id][metric_type] = value
    
    
    async def _trigger_opportunity_analysis(self, model_id: str):
        """Trigger improvement opportunity analysis for a model"""
        try:
            # Get recent metrics for analysis
            recent_metrics = list(self.metrics_store[model_id])[-MIN_SAMPLES_FOR_ANALYSIS:]
            
            # Convert to analysis format
            historical_data = [{
                "model_id": model_id,
                "metrics": {m.metric_type.value: m.value for m in recent_metrics},
                "timestamp": recent_metrics[-1].timestamp if recent_metrics else datetime.now(timezone.utc)
            }]
            
            # Identify opportunities (async, don't wait)
            asyncio.create_task(self.identify_improvement_opportunities(historical_data))
            
        except Exception as e:
            print(f"⚠️ Error triggering opportunity analysis for {model_id}: {str(e)}")
    
    
    async def _analyze_model_opportunities(self, model_id: str, data_points: List[Dict[str, Any]]) -> List[ImprovementOpportunity]:
        """Analyze improvement opportunities for a specific model"""
        opportunities = []
        
        try:
            # Collect all metrics
            all_metrics = defaultdict(list)
            for data_point in data_points:
                metrics = data_point.get("metrics", {})
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        all_metrics[metric_name].append(value)
            
            # Analyze each metric for improvement opportunities
            for metric_name, values in all_metrics.items():
                if len(values) >= 3:  # Need minimum data points
                    # Calculate trend
                    trend = self._calculate_simple_trend(values)
                    
                    # Identify opportunity based on trend and current performance
                    opportunity = await self._identify_metric_opportunity(
                        model_id, metric_name, values, trend
                    )
                    
                    if opportunity:
                        opportunities.append(opportunity)
            
        except Exception as e:
            print(f"⚠️ Error analyzing opportunities for model {model_id}: {str(e)}")
        
        return opportunities
    
    
    def _calculate_simple_trend(self, values: List[float]) -> float:
        """Calculate simple linear trend for a list of values"""
        if len(values) < 2:
            return 0.0
        
        # Simple slope calculation
        n = len(values)
        x = list(range(n))
        
        # Calculate slope using least squares
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    
    async def _identify_metric_opportunity(self, model_id: str, metric_name: str, 
                                         values: List[float], trend: float) -> Optional[ImprovementOpportunity]:
        """Identify improvement opportunity for a specific metric"""
        try:
            current_performance = values[-1]  # Latest value
            avg_performance = statistics.mean(values)
            std_performance = statistics.stdev(values) if len(values) > 1 else 0.0
            
            # Determine if improvement is needed
            improvement_needed = False
            improvement_type = ImprovementType.OPTIMIZATION
            expected_improvement = 0.0
            description = ""
            
            # Analyze based on metric characteristics
            if metric_name in ["accuracy", "precision", "recall", "f1_score", "throughput"]:
                # Higher is better
                if trend < 0 or current_performance < avg_performance - std_performance:
                    improvement_needed = True
                    expected_improvement = abs(trend) * 10 + 0.1  # Estimate improvement
                    description = f"Declining {metric_name} trend detected, optimization recommended"
                    
            elif metric_name in ["latency", "error_rate", "cost", "memory_usage", "cpu_usage"]:
                # Lower is better
                if trend > 0 or current_performance > avg_performance + std_performance:
                    improvement_needed = True
                    expected_improvement = abs(trend) * 10 + 0.1  # Estimate improvement
                    description = f"Increasing {metric_name} trend detected, optimization recommended"
            
            if improvement_needed:
                # Calculate priority score
                priority_score = self._calculate_priority_score(
                    current_performance, avg_performance, std_performance, abs(trend)
                )
                
                # Determine improvement type based on metric
                if metric_name in ["accuracy", "precision", "recall", "f1_score"]:
                    improvement_type = ImprovementType.TRAINING_DATA
                elif metric_name in ["latency", "throughput"]:
                    improvement_type = ImprovementType.OPTIMIZATION
                elif metric_name in ["memory_usage", "cpu_usage"]:
                    improvement_type = ImprovementType.ARCHITECTURE
                
                return ImprovementOpportunity(
                    improvement_type=improvement_type,
                    target_component=f"{model_id}:{metric_name}",
                    current_performance=current_performance,
                    expected_improvement=expected_improvement,
                    confidence=min(1.0, len(values) / 10.0),  # More data = higher confidence
                    implementation_cost=0.5,  # Default medium cost
                    priority_score=priority_score,
                    description=description,
                    supporting_data={
                        "trend": trend,
                        "avg_performance": avg_performance,
                        "std_deviation": std_performance,
                        "sample_count": len(values)
                    }
                )
            
            return None
            
        except Exception as e:
            print(f"⚠️ Error identifying opportunity for {metric_name}: {str(e)}")
            return None
    
    
    def _calculate_priority_score(self, current: float, avg: float, std: float, trend_magnitude: float) -> float:
        """Calculate priority score for an improvement opportunity"""
        # Deviation from average (normalized)
        deviation_score = abs(current - avg) / (std + 0.001)  # Avoid division by zero
        
        # Trend magnitude
        trend_score = min(1.0, trend_magnitude)
        
        # Combined score
        priority = (deviation_score * 0.6 + trend_score * 0.4)
        
        # Normalize to 0-1 range
        return min(1.0, priority)
    
    
    async def _get_recent_metrics(self, model_id: str, hours: int = 24) -> List[PerformanceMetric]:
        """Get recent metrics for a model within specified hours"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        if model_id not in self.metrics_store:
            return []
        
        recent_metrics = []
        for metric in self.metrics_store[model_id]:
            if metric.timestamp >= cutoff_time:
                recent_metrics.append(metric)
        
        return recent_metrics
    
    
    async def _get_baseline_model(self, model_id: str) -> str:
        """Get baseline model for comparison (could be a previous version or reference model)"""
        # For now, use the same model as its own baseline
        # In a real system, this would return a reference baseline model
        return f"{model_id}_baseline"
    
    
    async def _generate_recommendation(self, overall_improvement: float, 
                                     improvements: List[str], regressions: List[str]) -> str:
        """Generate recommendation based on performance comparison"""
        if overall_improvement > IMPROVEMENT_THRESHOLD:
            return f"Model shows strong improvement ({overall_improvement:.1%}). Continue current approach."
        elif overall_improvement < -IMPROVEMENT_THRESHOLD:
            return f"Model shows performance decline ({overall_improvement:.1%}). Investigation recommended."
        elif regressions:
            return f"Mixed results with {len(regressions)} regressions. Focus on addressing: {', '.join(regressions[:2])}"
        else:
            return "Performance is stable. Consider optimization for further gains."
    
    
    async def _get_metrics_in_period(self, model_id: str, start_time: datetime, end_time: datetime) -> List[PerformanceMetric]:
        """Get metrics for a model within a specific time period"""
        if model_id not in self.metrics_store:
            return []
        
        period_metrics = []
        for metric in self.metrics_store[model_id]:
            if start_time <= metric.timestamp <= end_time:
                period_metrics.append(metric)
        
        return period_metrics
    
    
    async def _calculate_trends(self, metrics: List[PerformanceMetric]) -> Dict[str, float]:
        """Calculate performance trends for different metric types"""
        trends = {}
        
        # Group by metric type
        metrics_by_type = defaultdict(list)
        for metric in metrics:
            metrics_by_type[metric.metric_type].append(metric)
        
        # Calculate trend for each type
        for metric_type, type_metrics in metrics_by_type.items():
            if len(type_metrics) >= 2:
                # Sort by timestamp
                type_metrics.sort(key=lambda x: x.timestamp)
                values = [m.value for m in type_metrics]
                trend = self._calculate_simple_trend(values)
                trends[metric_type.value] = trend
        
        return trends
    
    
    async def _detect_anomalies(self, metrics: List[PerformanceMetric]) -> List[Dict[str, Any]]:
        """Detect anomalies in performance metrics"""
        anomalies = []
        
        # Group by metric type
        metrics_by_type = defaultdict(list)
        for metric in metrics:
            metrics_by_type[metric.metric_type].append(metric)
        
        # Detect anomalies for each type
        for metric_type, type_metrics in metrics_by_type.items():
            if len(type_metrics) >= MIN_SAMPLES_FOR_ANALYSIS:
                values = [m.value for m in type_metrics]
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0.0
                
                # Find outliers (values beyond threshold standard deviations)
                for metric in type_metrics:
                    if std_val > 0:
                        z_score = abs((metric.value - mean_val) / std_val)
                        if z_score > ANOMALY_DETECTION_THRESHOLD:
                            anomalies.append({
                                "metric_type": metric_type.value,
                                "value": metric.value,
                                "expected_range": [mean_val - 2*std_val, mean_val + 2*std_val],
                                "z_score": z_score,
                                "timestamp": metric.timestamp.isoformat(),
                                "severity": "high" if z_score > 3.0 else "medium"
                            })
        
        return anomalies
    
    
    async def _calculate_health_score(self, metrics: List[PerformanceMetric], 
                                    trends: Dict[str, float], anomalies: List[Dict[str, Any]]) -> float:
        """Calculate overall health score for the model"""
        if not metrics:
            return 0.0
        
        health_factors = []
        
        # Factor 1: Recent performance vs baseline
        baseline_comparison = 0.0
        baseline_count = 0
        for metric in metrics:
            if metric.baseline_value is not None and metric.improvement_percentage is not None:
                # Normalize improvement percentage to 0-1 score
                improvement_score = max(0.0, min(1.0, metric.improvement_percentage / 100 + 0.5))
                baseline_comparison += improvement_score
                baseline_count += 1
        
        if baseline_count > 0:
            health_factors.append(baseline_comparison / baseline_count)
        
        # Factor 2: Trend stability (good trends = higher score)
        trend_score = 0.0
        if trends:
            # Negative trends for latency/errors are good, positive for throughput/accuracy are good
            positive_metrics = ["accuracy", "throughput", "user_satisfaction"]
            trend_scores = []
            
            for metric_name, trend in trends.items():
                if metric_name in positive_metrics:
                    # Positive trend is good
                    trend_scores.append(max(0.0, min(1.0, trend + 0.5)))
                else:
                    # Negative trend is good (lower latency, fewer errors)
                    trend_scores.append(max(0.0, min(1.0, -trend + 0.5)))
            
            if trend_scores:
                trend_score = statistics.mean(trend_scores)
                health_factors.append(trend_score)
        
        # Factor 3: Anomaly penalty
        anomaly_penalty = len(anomalies) * 0.1  # Each anomaly reduces score by 10%
        anomaly_score = max(0.0, 1.0 - anomaly_penalty)
        health_factors.append(anomaly_score)
        
        # Factor 4: Data availability (more metrics = higher confidence)
        data_score = min(1.0, len(metrics) / 20.0)  # 20 metrics = full score
        health_factors.append(data_score)
        
        # Calculate weighted average
        if health_factors:
            return statistics.mean(health_factors)
        else:
            return 0.5  # Neutral score if no data
    
    
    async def _generate_analysis_recommendations(self, trends: Dict[str, float], 
                                               anomalies: List[Dict[str, Any]], 
                                               opportunities: List[ImprovementOpportunity]) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Trend-based recommendations
        negative_trends = [metric for metric, trend in trends.items() if trend < -0.1]
        if negative_trends:
            recommendations.append(f"Address declining trends in: {', '.join(negative_trends)}")
        
        # Anomaly-based recommendations
        high_severity_anomalies = [a for a in anomalies if a.get("severity") == "high"]
        if high_severity_anomalies:
            recommendations.append(f"Investigate {len(high_severity_anomalies)} high-severity anomalies")
        
        # Opportunity-based recommendations
        high_priority_opportunities = [o for o in opportunities if o.priority_score > HIGH_PRIORITY_THRESHOLD]
        if high_priority_opportunities:
            top_opportunity = max(high_priority_opportunities, key=lambda x: x.priority_score)
            recommendations.append(f"High priority: {top_opportunity.description}")
        
        # Default recommendation
        if not recommendations:
            recommendations.append("Continue monitoring. Performance appears stable.")
        
        return recommendations


# === Global Performance Monitor Instance ===

_performance_monitor_instance: Optional[PerformanceMonitor] = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor instance"""
    global _performance_monitor_instance
    if _performance_monitor_instance is None:
        _performance_monitor_instance = PerformanceMonitor()
    return _performance_monitor_instance