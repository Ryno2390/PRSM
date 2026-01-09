#!/usr/bin/env python3
"""
PRSM Benchmark Results Comparator
Comprehensive framework for persisting, comparing, and analyzing benchmark results over time

Features:
- Historical benchmark data persistence
- Performance regression detection
- Trend analysis and visualization
- Comparative analysis between configurations
- Automated performance reporting
- Integration with dashboard and scaling tests
"""

import json
import csv
import sqlite3
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import numpy as np
import sys

# Add PRSM to path
PRSM_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PRSM_ROOT))


class TrendDirection(str, Enum):
    """Performance trend directions"""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    VOLATILE = "volatile"


class RegressionSeverity(str, Enum):
    """Performance regression severity levels"""
    NONE = "none"
    MINOR = "minor"          # 5-15% degradation
    MODERATE = "moderate"    # 15-30% degradation
    MAJOR = "major"          # 30-50% degradation
    CRITICAL = "critical"    # >50% degradation


@dataclass
class BenchmarkComparison:
    """Comparison between two benchmark results"""
    baseline_id: str
    comparison_id: str
    baseline_timestamp: datetime
    comparison_timestamp: datetime
    
    # Performance changes
    throughput_change_percent: float
    latency_change_percent: float
    success_rate_change_percent: float
    efficiency_change_percent: float
    
    # Trend analysis
    trend_direction: TrendDirection
    regression_severity: RegressionSeverity
    
    # Detailed changes
    metrics_comparison: Dict[str, Dict[str, float]]  # metric -> {baseline, comparison, change_percent}
    
    # Summary
    overall_performance_change: float  # Weighted performance score change
    is_regression: bool
    is_improvement: bool
    summary: str
    recommendations: List[str] = field(default_factory=list)


@dataclass
class TrendAnalysis:
    """Trend analysis for a specific metric over time"""
    metric_name: str
    time_range_days: int
    data_points: int
    
    # Statistical analysis
    trend_direction: TrendDirection
    slope: float  # Rate of change per day
    correlation: float  # Strength of trend (-1 to 1)
    variance: float  # Volatility measure
    
    # Performance characteristics
    mean_value: float
    std_deviation: float
    min_value: float
    max_value: float
    recent_average: float  # Last 7 days average
    
    # Projections
    projected_30_day_value: float
    confidence_interval: Tuple[float, float]
    
    # Analysis
    is_stable: bool
    has_regression: bool
    quality_score: float  # 0-1 quality rating


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    report_id: str
    generated_at: datetime
    time_range_days: int
    
    # Summary statistics
    total_benchmarks: int
    unique_configurations: int
    avg_throughput: float
    avg_latency: float
    avg_success_rate: float
    
    # Trend analysis
    trends: Dict[str, TrendAnalysis]  # metric_name -> TrendAnalysis
    
    # Regression analysis
    regressions_detected: List[BenchmarkComparison]
    improvements_detected: List[BenchmarkComparison]
    
    # Configuration analysis
    best_performing_configs: List[Dict[str, Any]]
    worst_performing_configs: List[Dict[str, Any]]
    
    # Recommendations
    performance_recommendations: List[str]
    optimization_opportunities: List[str]
    stability_concerns: List[str]
    
    # Raw data summary
    benchmark_count_by_type: Dict[str, int]
    configuration_performance_summary: Dict[str, Dict[str, float]]


class BenchmarkDatabase:
    """SQLite database for benchmark results persistence"""
    
    def __init__(self, db_path: str = "benchmark_results.db"):
        self.db_path = Path(db_path)
        self.init_database()
    
    def init_database(self):
        """Initialize the database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Benchmark runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    timestamp DATETIME NOT NULL,
                    benchmark_type TEXT NOT NULL,
                    configuration_hash TEXT NOT NULL,
                    node_count INTEGER,
                    duration_seconds INTEGER,
                    network_condition TEXT,
                    resource_profile TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Benchmark metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_type TEXT DEFAULT 'performance',
                    FOREIGN KEY (run_id) REFERENCES benchmark_runs (run_id)
                )
            """)
            
            # Scaling results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scaling_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    node_count INTEGER NOT NULL,
                    operations_per_second REAL,
                    mean_latency_ms REAL,
                    p95_latency_ms REAL,
                    success_rate REAL,
                    scaling_efficiency REAL,
                    cpu_usage_estimate REAL,
                    memory_usage_mb REAL,
                    FOREIGN KEY (run_id) REFERENCES benchmark_runs (run_id)
                )
            """)
            
            # Performance trends table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    date DATE NOT NULL,
                    value REAL NOT NULL,
                    benchmark_type TEXT,
                    node_count INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def store_benchmark_result(self, result_data: Dict[str, Any]) -> str:
        """Store a benchmark result in the database"""
        run_id = f"{result_data.get('config', {}).get('name', 'unknown')}_{int(datetime.now().timestamp())}"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Store benchmark run
            cursor.execute("""
                INSERT OR REPLACE INTO benchmark_runs 
                (run_id, timestamp, benchmark_type, configuration_hash, node_count, 
                 duration_seconds, network_condition, resource_profile)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                result_data.get('timing', {}).get('start_time', datetime.now().isoformat()),
                result_data.get('config', {}).get('benchmark_type', 'unknown'),
                str(hash(str(result_data.get('config', {})))),
                result_data.get('config', {}).get('node_count', 0),
                result_data.get('config', {}).get('duration_seconds', 0),
                result_data.get('config', {}).get('network_condition', 'unknown'),
                result_data.get('config', {}).get('resource_profile', 'unknown')
            ))
            
            # Store metrics
            metrics = result_data.get('metrics', {})
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    cursor.execute("""
                        INSERT INTO benchmark_metrics (run_id, metric_name, metric_value)
                        VALUES (?, ?, ?)
                    """, (run_id, metric_name, value))
            
            # Store scaling results if available
            if 'performance' in result_data and 'node_performance' in result_data['performance']:
                node_performance = result_data['performance']['node_performance']
                scaling_efficiency = result_data['performance'].get('scaling_efficiency', {})
                resource_usage = result_data.get('resources', {}).get('resource_usage', {})
                
                for node_count, perf_data in node_performance.items():
                    cursor.execute("""
                        INSERT INTO scaling_results 
                        (run_id, node_count, operations_per_second, mean_latency_ms, 
                         p95_latency_ms, success_rate, scaling_efficiency, 
                         cpu_usage_estimate, memory_usage_mb)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        run_id,
                        int(node_count),
                        perf_data.get('operations_per_second', 0),
                        perf_data.get('mean_latency_ms', 0),
                        perf_data.get('p95_latency_ms', 0),
                        perf_data.get('success_rate', 0),
                        scaling_efficiency.get(node_count, 0),
                        resource_usage.get(node_count, {}).get('cpu_usage_estimate', 0),
                        resource_usage.get(node_count, {}).get('memory_usage_mb', 0)
                    ))
            
            conn.commit()
        
        return run_id
    
    def get_benchmark_history(self, 
                            days: int = 30, 
                            benchmark_type: Optional[str] = None,
                            metric_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get benchmark history from the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT br.*, bm.metric_name, bm.metric_value
                FROM benchmark_runs br
                LEFT JOIN benchmark_metrics bm ON br.run_id = bm.run_id
                WHERE br.timestamp >= datetime('now', '-? days')
            """
            
            params = [str(days)]  # Add days parameter for parameterized query
            if benchmark_type:
                query += " AND br.benchmark_type = ?"
                params.append(benchmark_type)
            
            if metric_name:
                query += " AND bm.metric_name = ?"
                params.append(metric_name)
            
            query += " ORDER BY br.timestamp DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to dictionary format
            columns = [desc[0] for desc in cursor.description]
            results = []
            for row in rows:
                result = dict(zip(columns, row))
                results.append(result)
            
            return results


class BenchmarkComparator:
    """Main class for comparing and analyzing benchmark results"""
    
    def __init__(self, database: Optional[BenchmarkDatabase] = None):
        self.db = database or BenchmarkDatabase()
        self.comparison_cache = {}
    
    def load_benchmark_results_from_files(self, results_dir: str = "benchmark_results") -> List[Dict[str, Any]]:
        """Load benchmark results from JSON files"""
        results = []
        results_path = Path(results_dir)
        
        if not results_path.exists():
            return results
        
        # Load from comprehensive benchmark files
        for json_file in results_path.glob("benchmark_results_*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if 'results' in data:
                        for result in data['results']:
                            # Store in database
                            run_id = self.db.store_benchmark_result(result)
                            result['run_id'] = run_id
                            results.append(result)
            except Exception as e:
                print(f"⚠️ Could not load {json_file}: {e}")
        
        # Load from scaling test files
        scaling_dir = Path("scaling_test_results")
        if scaling_dir.exists():
            for json_file in scaling_dir.glob("scaling_test_*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        # Store in database
                        run_id = self.db.store_benchmark_result(data)
                        data['run_id'] = run_id
                        results.append(data)
                except Exception as e:
                    print(f"⚠️ Could not load {json_file}: {e}")
        
        return results
    
    def compare_benchmarks(self, 
                          baseline_result: Dict[str, Any], 
                          comparison_result: Dict[str, Any]) -> BenchmarkComparison:
        """Compare two benchmark results"""
        
        # Extract metrics
        baseline_metrics = baseline_result.get('metrics', {})
        comparison_metrics = comparison_result.get('metrics', {})
        
        # Calculate percentage changes
        def safe_percent_change(baseline, comparison):
            if baseline == 0:
                return 100.0 if comparison > 0 else 0.0
            return ((comparison - baseline) / baseline) * 100.0
        
        throughput_change = safe_percent_change(
            baseline_metrics.get('operations_per_second', 0),
            comparison_metrics.get('operations_per_second', 0)
        )
        
        latency_change = safe_percent_change(
            baseline_metrics.get('mean_latency_ms', 0),
            comparison_metrics.get('mean_latency_ms', 0)
        )
        
        success_rate_change = safe_percent_change(
            baseline_metrics.get('consensus_success_rate', 0),
            comparison_metrics.get('consensus_success_rate', 0)
        )
        
        efficiency_change = safe_percent_change(
            baseline_metrics.get('operations_per_node_per_second', 0),
            comparison_metrics.get('operations_per_node_per_second', 0)
        )
        
        # Detailed metrics comparison
        metrics_comparison = {}
        all_metrics = set(baseline_metrics.keys()) | set(comparison_metrics.keys())
        
        for metric in all_metrics:
            baseline_val = baseline_metrics.get(metric, 0)
            comparison_val = comparison_metrics.get(metric, 0)
            change_pct = safe_percent_change(baseline_val, comparison_val)
            
            metrics_comparison[metric] = {
                'baseline': baseline_val,
                'comparison': comparison_val,
                'change_percent': change_pct
            }
        
        # Calculate overall performance change (weighted)
        weights = {
            'throughput': 0.4,
            'latency': -0.3,  # Negative because lower is better
            'success_rate': 0.2,
            'efficiency': 0.1
        }
        
        overall_change = (
            weights['throughput'] * throughput_change +
            weights['latency'] * (-latency_change) +  # Invert latency (lower is better)
            weights['success_rate'] * success_rate_change +
            weights['efficiency'] * efficiency_change
        )
        
        # Determine trend direction
        if overall_change > 10:
            trend = TrendDirection.IMPROVING
        elif overall_change < -10:
            trend = TrendDirection.DEGRADING
        elif abs(overall_change) < 5:
            trend = TrendDirection.STABLE
        else:
            trend = TrendDirection.VOLATILE
        
        # Determine regression severity
        if overall_change >= 0:
            severity = RegressionSeverity.NONE
        elif overall_change > -15:
            severity = RegressionSeverity.MINOR
        elif overall_change > -30:
            severity = RegressionSeverity.MODERATE
        elif overall_change > -50:
            severity = RegressionSeverity.MAJOR
        else:
            severity = RegressionSeverity.CRITICAL
        
        # Generate summary and recommendations
        summary = self._generate_comparison_summary(overall_change, throughput_change, latency_change)
        recommendations = self._generate_comparison_recommendations(severity, metrics_comparison)
        
        return BenchmarkComparison(
            baseline_id=baseline_result.get('run_id', 'unknown'),
            comparison_id=comparison_result.get('run_id', 'unknown'),
            baseline_timestamp=datetime.fromisoformat(baseline_result.get('timing', {}).get('start_time', datetime.now().isoformat())),
            comparison_timestamp=datetime.fromisoformat(comparison_result.get('timing', {}).get('start_time', datetime.now().isoformat())),
            throughput_change_percent=throughput_change,
            latency_change_percent=latency_change,
            success_rate_change_percent=success_rate_change,
            efficiency_change_percent=efficiency_change,
            trend_direction=trend,
            regression_severity=severity,
            metrics_comparison=metrics_comparison,
            overall_performance_change=overall_change,
            is_regression=overall_change < -5,
            is_improvement=overall_change > 5,
            summary=summary,
            recommendations=recommendations
        )
    
    def analyze_trends(self, 
                      metric_name: str, 
                      days: int = 30, 
                      benchmark_type: Optional[str] = None) -> TrendAnalysis:
        """Analyze trends for a specific metric over time"""
        
        # Get historical data
        history = self.db.get_benchmark_history(days=days, benchmark_type=benchmark_type, metric_name=metric_name)
        
        if len(history) < 2:
            # Not enough data for trend analysis
            return TrendAnalysis(
                metric_name=metric_name,
                time_range_days=days,
                data_points=len(history),
                trend_direction=TrendDirection.STABLE,
                slope=0.0,
                correlation=0.0,
                variance=0.0,
                mean_value=history[0]['metric_value'] if history else 0.0,
                std_deviation=0.0,
                min_value=history[0]['metric_value'] if history else 0.0,
                max_value=history[0]['metric_value'] if history else 0.0,
                recent_average=history[0]['metric_value'] if history else 0.0,
                projected_30_day_value=history[0]['metric_value'] if history else 0.0,
                confidence_interval=(0.0, 0.0),
                is_stable=True,
                has_regression=False,
                quality_score=0.5
            )
        
        # Extract values and timestamps
        values = [row['metric_value'] for row in history if row['metric_value'] is not None]
        timestamps = [datetime.fromisoformat(row['timestamp']) for row in history if row['metric_value'] is not None]
        
        if len(values) < 2:
            return TrendAnalysis(
                metric_name=metric_name,
                time_range_days=days,
                data_points=0,
                trend_direction=TrendDirection.STABLE,
                slope=0.0,
                correlation=0.0,
                variance=0.0,
                mean_value=0.0,
                std_deviation=0.0,
                min_value=0.0,
                max_value=0.0,
                recent_average=0.0,
                projected_30_day_value=0.0,
                confidence_interval=(0.0, 0.0),
                is_stable=True,
                has_regression=False,
                quality_score=0.0
            )
        
        # Convert timestamps to days since first measurement
        base_time = min(timestamps)
        time_deltas = [(ts - base_time).total_seconds() / 86400 for ts in timestamps]  # Days
        
        # Statistical analysis
        mean_value = statistics.mean(values)
        std_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        min_value = min(values)
        max_value = max(values)
        variance = std_deviation ** 2
        
        # Recent average (last 7 days)
        recent_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        recent_values = [v for i, v in enumerate(values) if timestamps[i] >= recent_cutoff]
        recent_average = statistics.mean(recent_values) if recent_values else mean_value
        
        # Trend analysis using linear regression
        if len(time_deltas) >= 2:
            # Simple linear regression
            n = len(time_deltas)
            sum_x = sum(time_deltas)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(time_deltas, values))
            sum_x2 = sum(x * x for x in time_deltas)
            
            # Slope (rate of change per day)
            denominator = n * sum_x2 - sum_x * sum_x
            if abs(denominator) > 1e-10:
                slope = (n * sum_xy - sum_x * sum_y) / denominator
            else:
                slope = 0.0
            
            # Correlation coefficient
            if std_deviation > 0 and len(time_deltas) > 1:
                time_std = statistics.stdev(time_deltas)
                if time_std > 0:
                    covariance = sum((x - statistics.mean(time_deltas)) * (y - mean_value) for x, y in zip(time_deltas, values)) / (n - 1)
                    correlation = covariance / (time_std * std_deviation)
                else:
                    correlation = 0.0
            else:
                correlation = 0.0
        else:
            slope = 0.0
            correlation = 0.0
        
        # Determine trend direction
        if abs(slope) < std_deviation * 0.1:  # Slope is small relative to variance
            trend_direction = TrendDirection.STABLE
        elif slope > 0:
            trend_direction = TrendDirection.IMPROVING if metric_name in ['operations_per_second', 'consensus_success_rate'] else TrendDirection.DEGRADING
        else:
            trend_direction = TrendDirection.DEGRADING if metric_name in ['operations_per_second', 'consensus_success_rate'] else TrendDirection.IMPROVING
        
        # Check if volatile
        if variance > mean_value * 0.25:  # High variance relative to mean
            trend_direction = TrendDirection.VOLATILE
        
        # Project 30-day value
        current_time_delta = max(time_deltas) if time_deltas else 0
        projected_30_day_value = values[-1] + slope * (current_time_delta + 30) if values else 0.0
        
        # Confidence interval (simple approximation)
        margin_of_error = 1.96 * std_deviation  # 95% confidence
        confidence_interval = (projected_30_day_value - margin_of_error, projected_30_day_value + margin_of_error)
        
        # Quality assessment
        is_stable = trend_direction == TrendDirection.STABLE
        has_regression = trend_direction == TrendDirection.DEGRADING
        
        # Quality score based on data consistency and trend strength
        quality_score = min(1.0, max(0.0, (
            0.4 * min(1.0, len(values) / 20) +  # Data quantity
            0.3 * min(1.0, abs(correlation)) +   # Trend strength
            0.3 * (1.0 - min(1.0, variance / (mean_value + 1e-10)))  # Consistency
        )))
        
        return TrendAnalysis(
            metric_name=metric_name,
            time_range_days=days,
            data_points=len(values),
            trend_direction=trend_direction,
            slope=slope,
            correlation=correlation,
            variance=variance,
            mean_value=mean_value,
            std_deviation=std_deviation,
            min_value=min_value,
            max_value=max_value,
            recent_average=recent_average,
            projected_30_day_value=projected_30_day_value,
            confidence_interval=confidence_interval,
            is_stable=is_stable,
            has_regression=has_regression,
            quality_score=quality_score
        )
    
    def detect_regressions(self, days: int = 7) -> List[BenchmarkComparison]:
        """Detect performance regressions in recent benchmarks"""
        regressions = []
        
        # Get recent benchmark results
        recent_results = self.load_benchmark_results_from_files()
        
        if len(recent_results) < 2:
            return regressions
        
        # Sort by timestamp
        recent_results.sort(key=lambda x: x.get('timing', {}).get('start_time', ''))
        
        # Compare each result with the previous one
        for i in range(1, len(recent_results)):
            baseline = recent_results[i-1]
            current = recent_results[i]
            
            comparison = self.compare_benchmarks(baseline, current)
            
            if comparison.is_regression and comparison.regression_severity != RegressionSeverity.NONE:
                regressions.append(comparison)
        
        return regressions
    
    def generate_performance_report(self, days: int = 30) -> PerformanceReport:
        """Generate a comprehensive performance report"""
        
        report_id = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Load all benchmark data
        all_results = self.load_benchmark_results_from_files()
        
        # Filter by time range
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        filtered_results = []
        for result in all_results:
            timestamp_str = result.get('timing', {}).get('start_time', datetime.now().isoformat())
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if timestamp >= cutoff_date:
                    filtered_results.append(result)
            except ValueError:
                continue
        
        if not filtered_results:
            # Return empty report
            return PerformanceReport(
                report_id=report_id,
                generated_at=datetime.now(timezone.utc),
                time_range_days=days,
                total_benchmarks=0,
                unique_configurations=0,
                avg_throughput=0.0,
                avg_latency=0.0,
                avg_success_rate=0.0,
                trends={},
                regressions_detected=[],
                improvements_detected=[],
                best_performing_configs=[],
                worst_performing_configs=[],
                performance_recommendations=[],
                optimization_opportunities=[],
                stability_concerns=[],
                benchmark_count_by_type={},
                configuration_performance_summary={}
            )
        
        # Calculate summary statistics
        total_benchmarks = len(filtered_results)
        unique_configurations = len(set(str(r.get('config', {})) for r in filtered_results))
        
        throughputs = [r.get('metrics', {}).get('operations_per_second', 0) for r in filtered_results]
        latencies = [r.get('metrics', {}).get('mean_latency_ms', 0) for r in filtered_results]
        success_rates = [r.get('metrics', {}).get('consensus_success_rate', 0) for r in filtered_results]
        
        avg_throughput = statistics.mean([t for t in throughputs if t > 0]) if throughputs else 0.0
        avg_latency = statistics.mean([l for l in latencies if l > 0]) if latencies else 0.0
        avg_success_rate = statistics.mean([s for s in success_rates if s > 0]) if success_rates else 0.0
        
        # Trend analysis
        key_metrics = ['operations_per_second', 'mean_latency_ms', 'consensus_success_rate', 'operations_per_node_per_second']
        trends = {}
        for metric in key_metrics:
            trends[metric] = self.analyze_trends(metric, days=days)
        
        # Regression and improvement detection
        regressions_detected = self.detect_regressions(days=days)
        improvements_detected = [comp for comp in regressions_detected if comp.is_improvement]
        regressions_detected = [comp for comp in regressions_detected if comp.is_regression]
        
        # Configuration analysis
        config_performance = {}
        for result in filtered_results:
            config_name = result.get('config', {}).get('name', 'unknown')
            metrics = result.get('metrics', {})
            
            if config_name not in config_performance:
                config_performance[config_name] = {
                    'throughput': [],
                    'latency': [],
                    'success_rate': []
                }
            
            config_performance[config_name]['throughput'].append(metrics.get('operations_per_second', 0))
            config_performance[config_name]['latency'].append(metrics.get('mean_latency_ms', 0))
            config_performance[config_name]['success_rate'].append(metrics.get('consensus_success_rate', 0))
        
        # Best and worst performing configs
        config_scores = {}
        for config_name, perf_data in config_performance.items():
            if perf_data['throughput']:
                # Weighted performance score
                avg_throughput = statistics.mean(perf_data['throughput'])
                avg_latency = statistics.mean(perf_data['latency'])
                avg_success = statistics.mean(perf_data['success_rate'])
                
                # Normalize and weight (higher is better)
                score = (avg_throughput * 0.4) + ((1000 / max(avg_latency, 1)) * 0.3) + (avg_success * 100 * 0.3)
                config_scores[config_name] = score
        
        sorted_configs = sorted(config_scores.items(), key=lambda x: x[1], reverse=True)
        best_performing_configs = [{'name': name, 'score': score} for name, score in sorted_configs[:3]]
        worst_performing_configs = [{'name': name, 'score': score} for name, score in sorted_configs[-3:]]
        
        # Generate recommendations
        performance_recommendations = self._generate_performance_recommendations(trends, regressions_detected)
        optimization_opportunities = self._generate_optimization_opportunities(config_performance, trends)
        stability_concerns = self._identify_stability_concerns(trends, regressions_detected)
        
        # Benchmark count by type
        benchmark_count_by_type = {}
        for result in filtered_results:
            bench_type = result.get('config', {}).get('benchmark_type', 'unknown')
            benchmark_count_by_type[bench_type] = benchmark_count_by_type.get(bench_type, 0) + 1
        
        # Configuration performance summary
        configuration_performance_summary = {}
        for config_name, perf_data in config_performance.items():
            configuration_performance_summary[config_name] = {
                'avg_throughput': statistics.mean(perf_data['throughput']) if perf_data['throughput'] else 0.0,
                'avg_latency': statistics.mean(perf_data['latency']) if perf_data['latency'] else 0.0,
                'avg_success_rate': statistics.mean(perf_data['success_rate']) if perf_data['success_rate'] else 0.0,
                'stability': statistics.stdev(perf_data['throughput']) if len(perf_data['throughput']) > 1 else 0.0
            }
        
        return PerformanceReport(
            report_id=report_id,
            generated_at=datetime.now(timezone.utc),
            time_range_days=days,
            total_benchmarks=total_benchmarks,
            unique_configurations=unique_configurations,
            avg_throughput=avg_throughput,
            avg_latency=avg_latency,
            avg_success_rate=avg_success_rate,
            trends=trends,
            regressions_detected=regressions_detected,
            improvements_detected=improvements_detected,
            best_performing_configs=best_performing_configs,
            worst_performing_configs=worst_performing_configs,
            performance_recommendations=performance_recommendations,
            optimization_opportunities=optimization_opportunities,
            stability_concerns=stability_concerns,
            benchmark_count_by_type=benchmark_count_by_type,
            configuration_performance_summary=configuration_performance_summary
        )
    
    def _generate_comparison_summary(self, overall_change: float, throughput_change: float, latency_change: float) -> str:
        """Generate a summary for benchmark comparison"""
        if overall_change > 10:
            return f"Significant improvement: {overall_change:.1f}% overall performance gain"
        elif overall_change > 5:
            return f"Moderate improvement: {overall_change:.1f}% overall performance gain"
        elif overall_change > -5:
            return f"Stable performance: {abs(overall_change):.1f}% variation"
        elif overall_change > -15:
            return f"Minor regression: {abs(overall_change):.1f}% performance decrease"
        elif overall_change > -30:
            return f"Moderate regression: {abs(overall_change):.1f}% performance decrease"
        else:
            return f"Major regression: {abs(overall_change):.1f}% performance decrease"
    
    def _generate_comparison_recommendations(self, severity: RegressionSeverity, metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate recommendations based on comparison results"""
        recommendations = []
        
        if severity == RegressionSeverity.CRITICAL:
            recommendations.append("URGENT: Critical performance regression detected - immediate investigation required")
            recommendations.append("Consider rolling back recent changes")
            recommendations.append("Run comprehensive diagnostics")
        elif severity == RegressionSeverity.MAJOR:
            recommendations.append("Major performance regression - prioritize investigation")
            recommendations.append("Review recent code changes and configuration updates")
        elif severity == RegressionSeverity.MODERATE:
            recommendations.append("Moderate performance regression - schedule investigation")
            recommendations.append("Monitor closely for further degradation")
        elif severity == RegressionSeverity.MINOR:
            recommendations.append("Minor performance regression - monitor trends")
        
        # Specific metric recommendations
        for metric, data in metrics.items():
            change = data['change_percent']
            if abs(change) > 20:
                if 'latency' in metric and change > 0:
                    recommendations.append(f"High latency increase in {metric} - check network conditions")
                elif 'throughput' in metric and change < 0:
                    recommendations.append(f"Significant throughput decrease in {metric} - investigate bottlenecks")
        
        return recommendations
    
    def _generate_performance_recommendations(self, trends: Dict[str, TrendAnalysis], regressions: List[BenchmarkComparison]) -> List[str]:
        """Generate performance recommendations based on trends and regressions"""
        recommendations = []
        
        # Trend-based recommendations
        degrading_trends = [name for name, trend in trends.items() if trend.trend_direction == TrendDirection.DEGRADING]
        if degrading_trends:
            recommendations.append(f"Performance degradation detected in: {', '.join(degrading_trends)}")
            recommendations.append("Consider implementing performance optimization measures")
        
        volatile_trends = [name for name, trend in trends.items() if trend.trend_direction == TrendDirection.VOLATILE]
        if volatile_trends:
            recommendations.append(f"High performance variability in: {', '.join(volatile_trends)}")
            recommendations.append("Investigate causes of performance instability")
        
        # Regression-based recommendations
        if len(regressions) > 3:
            recommendations.append("Multiple performance regressions detected - comprehensive review needed")
        elif regressions:
            recommendations.append("Recent performance regressions require attention")
        
        return recommendations
    
    def _generate_optimization_opportunities(self, config_performance: Dict[str, Dict[str, List[float]]], trends: Dict[str, TrendAnalysis]) -> List[str]:
        """Generate optimization opportunities"""
        opportunities = []
        
        # Configuration-based opportunities
        if len(config_performance) > 1:
            # Find best and worst configurations
            config_scores = {}
            for config_name, perf_data in config_performance.items():
                if perf_data['throughput']:
                    avg_throughput = statistics.mean(perf_data['throughput'])
                    config_scores[config_name] = avg_throughput
            
            if config_scores:
                best_config = max(config_scores, key=config_scores.get)
                worst_config = min(config_scores, key=config_scores.get)
                
                best_score = config_scores[best_config]
                worst_score = config_scores[worst_config]
                
                if best_score > worst_score * 1.5:
                    opportunities.append(f"Configuration optimization potential: {best_config} performs {best_score/worst_score:.1f}x better than {worst_config}")
        
        # Trend-based opportunities
        for metric_name, trend in trends.items():
            if trend.quality_score < 0.6:
                opportunities.append(f"Improve {metric_name} consistency - current quality score: {trend.quality_score:.2f}")
        
        return opportunities
    
    def _identify_stability_concerns(self, trends: Dict[str, TrendAnalysis], regressions: List[BenchmarkComparison]) -> List[str]:
        """Identify stability concerns"""
        concerns = []
        
        # High variance trends
        for metric_name, trend in trends.items():
            if trend.variance > trend.mean_value * 0.5:  # Variance > 50% of mean
                concerns.append(f"High variability in {metric_name} - coefficient of variation: {(trend.std_deviation/trend.mean_value)*100:.1f}%")
        
        # Frequent regressions
        if len(regressions) > 2:
            concerns.append(f"Frequent performance regressions detected: {len(regressions)} in recent period")
        
        # Low quality trends
        low_quality_trends = [name for name, trend in trends.items() if trend.quality_score < 0.4]
        if low_quality_trends:
            concerns.append(f"Poor data quality for metrics: {', '.join(low_quality_trends)}")
        
        return concerns
    
    def save_report(self, report: PerformanceReport, output_dir: str = "performance_reports") -> str:
        """Save performance report to file"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        filename = f"{report.report_id}.json"
        filepath = output_path / filename
        
        # Convert to JSON-serializable format
        report_data = {
            'report_id': report.report_id,
            'generated_at': report.generated_at.isoformat(),
            'time_range_days': report.time_range_days,
            'summary': {
                'total_benchmarks': report.total_benchmarks,
                'unique_configurations': report.unique_configurations,
                'avg_throughput': report.avg_throughput,
                'avg_latency': report.avg_latency,
                'avg_success_rate': report.avg_success_rate
            },
            'trends': {name: asdict(trend) for name, trend in report.trends.items()},
            'regressions_detected': [asdict(reg) for reg in report.regressions_detected],
            'improvements_detected': [asdict(imp) for imp in report.improvements_detected],
            'analysis': {
                'best_performing_configs': report.best_performing_configs,
                'worst_performing_configs': report.worst_performing_configs,
                'performance_recommendations': report.performance_recommendations,
                'optimization_opportunities': report.optimization_opportunities,
                'stability_concerns': report.stability_concerns
            },
            'data_summary': {
                'benchmark_count_by_type': report.benchmark_count_by_type,
                'configuration_performance_summary': report.configuration_performance_summary
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"📊 Performance report saved: {filepath}")
        return str(filepath)


# Demo functions
def demo_benchmark_comparator():
    """Demonstrate benchmark comparison capabilities"""
    print("📊 PRSM Benchmark Comparator - Demo")
    print("=" * 60)
    
    comparator = BenchmarkComparator()
    
    # Load existing benchmark results
    print("1. Loading benchmark results...")
    results = comparator.load_benchmark_results_from_files()
    print(f"   ✅ Loaded {len(results)} benchmark results")
    
    if len(results) >= 2:
        # Compare two results
        print("\n2. Comparing benchmark results...")
        comparison = comparator.compare_benchmarks(results[0], results[1])
        print(f"   ✅ Comparison completed")
        print(f"   Summary: {comparison.summary}")
        print(f"   Overall change: {comparison.overall_performance_change:.1f}%")
        print(f"   Regression severity: {comparison.regression_severity.value}")
    
    # Generate performance report
    print("\n3. Generating performance report...")
    report = comparator.generate_performance_report(days=30)
    print(f"   ✅ Report generated")
    print(f"   Total benchmarks analyzed: {report.total_benchmarks}")
    print(f"   Average throughput: {report.avg_throughput:.2f} ops/s")
    print(f"   Trends analyzed: {len(report.trends)}")
    print(f"   Regressions detected: {len(report.regressions_detected)}")
    
    # Save report
    report_file = comparator.save_report(report)
    print(f"   ✅ Report saved: {report_file}")
    
    print(f"\n✅ Benchmark comparator demo completed!")
    return True


if __name__ == "__main__":
    demo_benchmark_comparator()