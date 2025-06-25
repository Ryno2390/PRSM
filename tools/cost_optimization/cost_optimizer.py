#!/usr/bin/env python3
"""
PRSM Advanced Cost Optimizer

Intelligent cost optimization system that automatically analyzes usage patterns,
identifies cost-saving opportunities, and provides actionable recommendations
for PRSM deployments.

Features:
- Real-time cost monitoring and analysis
- Automated resource right-sizing recommendations
- Provider cost comparison and switching suggestions
- Performance vs cost optimization
- Predictive cost modeling
- Budget alerts and cost controls

Author: PRSM Platform Team
"""

import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict, deque


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Cost optimization strategies"""
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_PERFORMANCE = "maximize_performance"
    BALANCED = "balanced"
    SUSTAINABILITY = "sustainability"


class AlertLevel(Enum):
    """Cost alert levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ResourceUtilization:
    """Resource utilization metrics"""
    cpu_utilization: float = 0.0  # 0-100%
    memory_utilization: float = 0.0  # 0-100%
    gpu_utilization: float = 0.0  # 0-100%
    storage_utilization: float = 0.0  # 0-100%
    network_utilization: float = 0.0  # 0-100%
    
    # Performance metrics
    avg_response_time: float = 0.0  # seconds
    error_rate: float = 0.0  # 0-100%
    throughput: float = 0.0  # requests per second


@dataclass
class CostAlert:
    """Cost monitoring alert"""
    alert_id: str
    level: AlertLevel
    message: str
    current_value: float
    threshold: float
    recommendation: str
    timestamp: datetime


@dataclass
class OptimizationRecommendation:
    """Cost optimization recommendation"""
    recommendation_id: str
    title: str
    description: str
    potential_savings: float  # USD per month
    effort_level: str  # low, medium, high
    implementation_time: str  # estimate
    risk_level: str  # low, medium, high
    category: str  # infrastructure, ai_models, usage_patterns
    specific_actions: List[str]
    impact_score: float  # 0-100


class CostOptimizer:
    """Advanced cost optimization engine"""
    
    def __init__(self, monitoring_window_hours: int = 24):
        self.monitoring_window = monitoring_window_hours
        self.usage_history = deque(maxlen=monitoring_window_hours * 60)  # Store per minute
        self.cost_history = deque(maxlen=monitoring_window_hours)
        self.alerts = []
        
        # Optimization thresholds
        self.thresholds = {
            "cpu_low_utilization": 20.0,
            "memory_low_utilization": 30.0,
            "gpu_low_utilization": 40.0,
            "high_cost_per_request": 1.0,
            "budget_warning": 0.8,  # 80% of budget
            "budget_critical": 0.95  # 95% of budget
        }
    
    def add_usage_sample(self, utilization: ResourceUtilization, cost_data: Dict[str, float]):
        """Add usage and cost sample for analysis"""
        timestamp = datetime.now()
        
        sample = {
            "timestamp": timestamp,
            "utilization": asdict(utilization),
            "costs": cost_data,
            "total_cost": sum(cost_data.values())
        }
        
        self.usage_history.append(sample)
        
        # Aggregate hourly cost data
        if len(self.cost_history) == 0 or \
           (timestamp - self.cost_history[-1]["timestamp"]).total_seconds() >= 3600:
            self.cost_history.append({
                "timestamp": timestamp,
                "hourly_cost": sum(cost_data.values()),
                "cost_breakdown": cost_data.copy()
            })
    
    def analyze_utilization_patterns(self) -> Dict[str, Any]:
        """Analyze resource utilization patterns"""
        if len(self.usage_history) < 10:
            return {"error": "Insufficient data for analysis"}
        
        # Extract utilization data
        cpu_data = [sample["utilization"]["cpu_utilization"] for sample in self.usage_history]
        memory_data = [sample["utilization"]["memory_utilization"] for sample in self.usage_history]
        gpu_data = [sample["utilization"]["gpu_utilization"] for sample in self.usage_history]
        
        # Calculate statistics
        analysis = {
            "cpu": {
                "avg": np.mean(cpu_data),
                "max": np.max(cpu_data),
                "min": np.min(cpu_data),
                "std": np.std(cpu_data),
                "p95": np.percentile(cpu_data, 95),
                "p99": np.percentile(cpu_data, 99)
            },
            "memory": {
                "avg": np.mean(memory_data),
                "max": np.max(memory_data),
                "min": np.min(memory_data),
                "std": np.std(memory_data),
                "p95": np.percentile(memory_data, 95),
                "p99": np.percentile(memory_data, 99)
            },
            "gpu": {
                "avg": np.mean(gpu_data),
                "max": np.max(gpu_data),
                "min": np.min(gpu_data),
                "std": np.std(gpu_data),
                "p95": np.percentile(gpu_data, 95),
                "p99": np.percentile(gpu_data, 99)
            }
        }
        
        # Identify patterns
        patterns = {
            "low_cpu_utilization": analysis["cpu"]["avg"] < self.thresholds["cpu_low_utilization"],
            "low_memory_utilization": analysis["memory"]["avg"] < self.thresholds["memory_low_utilization"],
            "low_gpu_utilization": analysis["gpu"]["avg"] < self.thresholds["gpu_low_utilization"],
            "high_variability": any(stats["std"] > 40 for stats in analysis.values()),
            "consistent_low_usage": all(stats["p95"] < 50 for stats in analysis.values())
        }
        
        return {
            "statistics": analysis,
            "patterns": patterns,
            "sample_count": len(self.usage_history)
        }
    
    def analyze_cost_trends(self) -> Dict[str, Any]:
        """Analyze cost trends and anomalies"""
        if len(self.cost_history) < 5:
            return {"error": "Insufficient cost data for trend analysis"}
        
        # Extract cost data
        hourly_costs = [sample["hourly_cost"] for sample in self.cost_history]
        timestamps = [sample["timestamp"] for sample in self.cost_history]
        
        # Calculate trends
        cost_trend = np.polyfit(range(len(hourly_costs)), hourly_costs, 1)[0]  # Linear trend
        
        # Cost statistics
        avg_cost = np.mean(hourly_costs)
        cost_volatility = np.std(hourly_costs) / avg_cost if avg_cost > 0 else 0
        
        # Detect anomalies (costs > 2 standard deviations from mean)
        threshold = avg_cost + 2 * np.std(hourly_costs)
        anomalies = [
            {"timestamp": timestamps[i], "cost": cost, "threshold": threshold}
            for i, cost in enumerate(hourly_costs)
            if cost > threshold
        ]
        
        # Cost breakdown analysis
        cost_categories = defaultdict(list)
        for sample in self.cost_history:
            for category, cost in sample["cost_breakdown"].items():
                cost_categories[category].append(cost)
        
        category_analysis = {}
        for category, costs in cost_categories.items():
            category_analysis[category] = {
                "avg": np.mean(costs),
                "trend": np.polyfit(range(len(costs)), costs, 1)[0],
                "percentage_of_total": np.mean(costs) / avg_cost * 100 if avg_cost > 0 else 0
            }
        
        return {
            "hourly_costs": hourly_costs,
            "average_hourly_cost": avg_cost,
            "cost_trend": cost_trend,
            "cost_volatility": cost_volatility,
            "anomalies": anomalies,
            "category_breakdown": category_analysis,
            "projection_24h": avg_cost * 24,
            "projection_monthly": avg_cost * 24 * 30
        }
    
    def generate_recommendations(self, strategy: OptimizationStrategy = OptimizationStrategy.BALANCED) -> List[OptimizationRecommendation]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        utilization_analysis = self.analyze_utilization_patterns()
        cost_analysis = self.analyze_cost_trends()
        
        if "error" in utilization_analysis or "error" in cost_analysis:
            return recommendations
        
        patterns = utilization_analysis["patterns"]
        category_costs = cost_analysis["category_breakdown"]
        
        # Right-sizing recommendations
        if patterns["low_cpu_utilization"]:
            recommendations.append(OptimizationRecommendation(
                recommendation_id="rightsizing_cpu",
                title="Reduce CPU Resources",
                description=f"CPU utilization averaging {utilization_analysis['statistics']['cpu']['avg']:.1f}% - consider downsizing",
                potential_savings=200.0,  # Estimate
                effort_level="low",
                implementation_time="1-2 hours",
                risk_level="low",
                category="infrastructure",
                specific_actions=[
                    "Reduce CPU cores from current allocation",
                    "Monitor performance during transition",
                    "Consider auto-scaling for variable workloads"
                ],
                impact_score=75.0
            ))
        
        if patterns["low_memory_utilization"]:
            recommendations.append(OptimizationRecommendation(
                recommendation_id="rightsizing_memory",
                title="Optimize Memory Allocation",
                description=f"Memory utilization averaging {utilization_analysis['statistics']['memory']['avg']:.1f}% - reduce allocation",
                potential_savings=150.0,
                effort_level="low",
                implementation_time="1 hour",
                risk_level="low",
                category="infrastructure",
                specific_actions=[
                    "Reduce memory allocation by 20-30%",
                    "Set up memory monitoring alerts",
                    "Implement memory optimization in applications"
                ],
                impact_score=65.0
            ))
        
        if patterns["low_gpu_utilization"]:
            recommendations.append(OptimizationRecommendation(
                recommendation_id="gpu_optimization",
                title="GPU Resource Optimization",
                description=f"GPU utilization averaging {utilization_analysis['statistics']['gpu']['avg']:.1f}% - optimize or reduce",
                potential_savings=800.0,
                effort_level="medium",
                implementation_time="4-6 hours",
                risk_level="medium",
                category="infrastructure",
                specific_actions=[
                    "Implement GPU sharing for multiple workloads",
                    "Consider spot/preemptible GPU instances",
                    "Optimize AI model inference batching",
                    "Evaluate GPU-less alternatives for some workloads"
                ],
                impact_score=90.0
            ))
        
        # AI model cost optimizations
        if "ai_models" in category_costs and category_costs["ai_models"]["percentage_of_total"] > 60:
            recommendations.append(OptimizationRecommendation(
                recommendation_id="ai_model_optimization",
                title="AI Model Cost Optimization",
                description=f"AI model costs are {category_costs['ai_models']['percentage_of_total']:.1f}% of total - optimize provider mix",
                potential_savings=500.0,
                effort_level="medium",
                implementation_time="1-2 days",
                risk_level="medium",
                category="ai_models",
                specific_actions=[
                    "Implement intelligent model routing",
                    "Use cheaper models for simple tasks",
                    "Increase self-hosted model usage",
                    "Optimize prompt engineering to reduce token usage",
                    "Implement response caching"
                ],
                impact_score=85.0
            ))
        
        # Usage pattern optimizations
        if patterns["high_variability"]:
            recommendations.append(OptimizationRecommendation(
                recommendation_id="auto_scaling",
                title="Implement Auto-Scaling",
                description="High resource usage variability detected - implement dynamic scaling",
                potential_savings=300.0,
                effort_level="high",
                implementation_time="1-2 weeks",
                risk_level="medium",
                category="usage_patterns",
                specific_actions=[
                    "Implement horizontal auto-scaling",
                    "Set up predictive scaling based on usage patterns",
                    "Configure automatic shutdown during low usage periods",
                    "Implement request queueing for peak periods"
                ],
                impact_score=80.0
            ))
        
        # Provider optimization based on strategy
        if strategy == OptimizationStrategy.MINIMIZE_COST:
            recommendations.append(OptimizationRecommendation(
                recommendation_id="cost_optimized_providers",
                title="Switch to Cost-Optimized Providers",
                description="Prioritize lowest cost providers and self-hosted solutions",
                potential_savings=400.0,
                effort_level="medium",
                implementation_time="3-5 days",
                risk_level="low",
                category="ai_models",
                specific_actions=[
                    "Increase PRSM Network usage to 80%",
                    "Use self-hosted models for 60% of simple tasks",
                    "Reserve premium APIs only for complex tasks",
                    "Implement aggressive caching strategies"
                ],
                impact_score=88.0
            ))
        
        elif strategy == OptimizationStrategy.SUSTAINABILITY:
            recommendations.append(OptimizationRecommendation(
                recommendation_id="green_optimization",
                title="Sustainability-Focused Optimization",
                description="Optimize for environmental impact and long-term sustainability",
                potential_savings=250.0,
                effort_level="high",
                implementation_time="2-3 weeks",
                risk_level="low",
                category="infrastructure",
                specific_actions=[
                    "Migrate to carbon-neutral cloud providers",
                    "Implement energy-efficient computing practices",
                    "Use renewable energy powered data centers",
                    "Optimize algorithms for energy efficiency",
                    "Participate in PRSM Network for distributed efficiency"
                ],
                impact_score=70.0
            ))
        
        # Sort recommendations by impact score
        recommendations.sort(key=lambda r: r.impact_score, reverse=True)
        
        return recommendations
    
    def monitor_budget(self, monthly_budget: float, current_date: datetime = None) -> List[CostAlert]:
        """Monitor budget and generate alerts"""
        if current_date is None:
            current_date = datetime.now()
        
        alerts = []
        
        # Calculate current month spending
        month_start = current_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_costs = [
            sample["hourly_cost"] for sample in self.cost_history
            if sample["timestamp"] >= month_start
        ]
        
        if not month_costs:
            return alerts
        
        current_spending = sum(month_costs)
        days_in_month = (current_date.replace(month=current_date.month % 12 + 1, day=1) - timedelta(days=1)).day
        days_elapsed = current_date.day
        projected_spending = current_spending / days_elapsed * days_in_month if days_elapsed > 0 else 0
        
        budget_utilization = current_spending / monthly_budget
        projected_budget_utilization = projected_spending / monthly_budget
        
        # Generate budget alerts
        if budget_utilization >= self.thresholds["budget_critical"]:
            alerts.append(CostAlert(
                alert_id=f"budget_critical_{int(time.time())}",
                level=AlertLevel.CRITICAL,
                message=f"Budget critically exceeded: {budget_utilization:.1%} of monthly budget used",
                current_value=current_spending,
                threshold=monthly_budget * self.thresholds["budget_critical"],
                recommendation="Immediate cost reduction required - implement emergency cost controls",
                timestamp=current_date
            ))
        
        elif budget_utilization >= self.thresholds["budget_warning"]:
            alerts.append(CostAlert(
                alert_id=f"budget_warning_{int(time.time())}",
                level=AlertLevel.WARNING,
                message=f"Budget warning: {budget_utilization:.1%} of monthly budget used",
                current_value=current_spending,
                threshold=monthly_budget * self.thresholds["budget_warning"],
                recommendation="Review and implement cost optimization recommendations",
                timestamp=current_date
            ))
        
        # Projected spending alerts
        if projected_budget_utilization > 1.1:  # Projected to exceed by 10%
            alerts.append(CostAlert(
                alert_id=f"projection_warning_{int(time.time())}",
                level=AlertLevel.WARNING,
                message=f"Projected to exceed budget by {(projected_budget_utilization - 1) * 100:.1f}%",
                current_value=projected_spending,
                threshold=monthly_budget,
                recommendation="Implement cost reduction measures to stay within budget",
                timestamp=current_date
            ))
        
        self.alerts.extend(alerts)
        return alerts
    
    def predict_costs(self, days_ahead: int = 30) -> Dict[str, float]:
        """Predict future costs based on trends"""
        cost_analysis = self.analyze_cost_trends()
        
        if "error" in cost_analysis:
            return {"error": "Insufficient data for cost prediction"}
        
        current_avg_hourly = cost_analysis["average_hourly_cost"]
        trend = cost_analysis["cost_trend"]
        
        # Simple linear prediction with trend
        hours_ahead = days_ahead * 24
        predicted_hourly_cost = current_avg_hourly + (trend * hours_ahead)
        
        # Ensure prediction doesn't go negative
        predicted_hourly_cost = max(predicted_hourly_cost, 0)
        
        return {
            "predicted_daily_cost": predicted_hourly_cost * 24,
            "predicted_weekly_cost": predicted_hourly_cost * 24 * 7,
            "predicted_monthly_cost": predicted_hourly_cost * 24 * 30,
            "prediction_confidence": "medium",  # Would be calculated from historical accuracy
            "trend_direction": "increasing" if trend > 0 else "decreasing" if trend < 0 else "stable"
        }
    
    def export_optimization_report(self, strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
                                 filename: str = None) -> str:
        """Export comprehensive optimization report"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prsm_optimization_report_{timestamp}.json"
        
        # Generate analysis
        utilization_analysis = self.analyze_utilization_patterns()
        cost_analysis = self.analyze_cost_trends()
        recommendations = self.generate_recommendations(strategy)
        cost_predictions = self.predict_costs()
        
        # Calculate total potential savings
        total_savings = sum(rec.potential_savings for rec in recommendations)
        
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "strategy": strategy.value,
                "analysis_period_hours": self.monitoring_window,
                "data_samples": len(self.usage_history)
            },
            "executive_summary": {
                "total_potential_savings": total_savings,
                "top_recommendation": recommendations[0].title if recommendations else None,
                "optimization_opportunities": len(recommendations),
                "current_efficiency_score": self._calculate_efficiency_score()
            },
            "utilization_analysis": utilization_analysis,
            "cost_analysis": cost_analysis,
            "recommendations": [asdict(rec) for rec in recommendations],
            "cost_predictions": cost_predictions,
            "alerts": [asdict(alert) for alert in self.alerts[-10:]]  # Last 10 alerts
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return filename
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate overall system efficiency score (0-100)"""
        if len(self.usage_history) < 5:
            return 50.0  # Default for insufficient data
        
        utilization_analysis = self.analyze_utilization_patterns()
        if "error" in utilization_analysis:
            return 50.0
        
        # Score based on resource utilization (target 60-80% utilization)
        cpu_score = self._utilization_score(utilization_analysis["statistics"]["cpu"]["avg"])
        memory_score = self._utilization_score(utilization_analysis["statistics"]["memory"]["avg"])
        gpu_score = self._utilization_score(utilization_analysis["statistics"]["gpu"]["avg"])
        
        # Weight GPU more heavily as it's typically most expensive
        efficiency_score = (cpu_score * 0.3 + memory_score * 0.3 + gpu_score * 0.4)
        
        return min(max(efficiency_score, 0), 100)
    
    def _utilization_score(self, utilization: float) -> float:
        """Calculate efficiency score for a single resource utilization"""
        # Optimal utilization is 60-80%
        if 60 <= utilization <= 80:
            return 100.0
        elif 40 <= utilization < 60:
            return 80.0 + (utilization - 40) * 1.0  # Linear from 80 to 100
        elif 80 < utilization <= 90:
            return 100.0 - (utilization - 80) * 2.0  # Linear from 100 to 80
        elif utilization > 90:
            return max(0, 80 - (utilization - 90) * 4.0)  # Penalty for over-utilization
        else:  # < 40%
            return utilization * 2.0  # Linear from 0 to 80


class OptimizationDashboard:
    """Real-time optimization monitoring dashboard"""
    
    def __init__(self, optimizer: CostOptimizer):
        self.optimizer = optimizer
        
    def display_dashboard(self):
        """Display real-time optimization dashboard"""
        print("🎯 PRSM Cost Optimization Dashboard")
        print("=" * 50)
        print()
        
        # Current efficiency score
        efficiency = self.optimizer._calculate_efficiency_score()
        efficiency_bar = "█" * int(efficiency / 5) + "░" * (20 - int(efficiency / 5))
        print(f"💡 System Efficiency: {efficiency:.1f}% [{efficiency_bar}]")
        print()
        
        # Current utilization
        if len(self.optimizer.usage_history) > 0:
            latest = self.optimizer.usage_history[-1]
            util = latest["utilization"]
            
            print("📊 Current Resource Utilization:")
            self._print_utilization_bar("CPU", util["cpu_utilization"])
            self._print_utilization_bar("Memory", util["memory_utilization"])
            self._print_utilization_bar("GPU", util["gpu_utilization"])
            print()
        
        # Cost trends
        cost_analysis = self.optimizer.analyze_cost_trends()
        if "error" not in cost_analysis:
            print("💰 Cost Analysis:")
            print(f"   Current hourly rate: ${cost_analysis['average_hourly_cost']:.2f}")
            print(f"   24h projection: ${cost_analysis['projection_24h']:.2f}")
            print(f"   Monthly projection: ${cost_analysis['projection_monthly']:.2f}")
            
            trend_icon = "📈" if cost_analysis['cost_trend'] > 0 else "📉" if cost_analysis['cost_trend'] < 0 else "➡️"
            print(f"   Trend: {trend_icon} ${cost_analysis['cost_trend']:.3f}/hour")
            print()
        
        # Top recommendations
        recommendations = self.optimizer.generate_recommendations()
        if recommendations:
            print("🎯 Top Optimization Opportunities:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec.title}")
                print(f"      💰 Potential savings: ${rec.potential_savings:.2f}/month")
                print(f"      ⚡ Impact score: {rec.impact_score:.0f}/100")
                print()
        
        # Recent alerts
        recent_alerts = [alert for alert in self.optimizer.alerts if 
                        (datetime.now() - alert.timestamp).total_seconds() < 3600]
        if recent_alerts:
            print("🚨 Recent Alerts:")
            for alert in recent_alerts[-3:]:
                icon = "🔴" if alert.level == AlertLevel.CRITICAL else "🟡" if alert.level == AlertLevel.WARNING else "🔵"
                print(f"   {icon} {alert.message}")
                print(f"      💡 {alert.recommendation}")
                print()
    
    def _print_utilization_bar(self, resource: str, utilization: float):
        """Print utilization bar chart"""
        bar_length = 20
        filled = int(utilization / 5)  # 5% per bar
        bar = "█" * filled + "░" * (bar_length - filled)
        
        color_icon = "🟢" if utilization < 60 else "🟡" if utilization < 80 else "🔴"
        print(f"   {color_icon} {resource:6}: {utilization:5.1f}% [{bar}]")


def main():
    """Main entry point for cost optimizer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PRSM Advanced Cost Optimizer")
    parser.add_argument("--demo", action="store_true", help="Run demo with sample data")
    parser.add_argument("--monitor", action="store_true", help="Start real-time monitoring")
    parser.add_argument("--budget", type=float, help="Monthly budget for monitoring")
    parser.add_argument("--strategy", choices=[s.value for s in OptimizationStrategy],
                       default="balanced", help="Optimization strategy")
    
    args = parser.parse_args()
    
    optimizer = CostOptimizer()
    
    if args.demo:
        # Generate sample data for demonstration
        print("🎮 Running PRSM Cost Optimizer Demo")
        print("=" * 40)
        print()
        
        # Simulate 24 hours of data
        for hour in range(24):
            for minute in range(0, 60, 5):  # Every 5 minutes
                # Simulate varying utilization
                base_cpu = 45 + 20 * np.sin(hour * np.pi / 12)  # Daily cycle
                base_memory = 55 + 15 * np.sin(hour * np.pi / 8)
                base_gpu = 30 + 25 * np.sin(hour * np.pi / 6)
                
                # Add some noise
                cpu_util = max(0, min(100, base_cpu + np.random.normal(0, 10)))
                memory_util = max(0, min(100, base_memory + np.random.normal(0, 8)))
                gpu_util = max(0, min(100, base_gpu + np.random.normal(0, 12)))
                
                utilization = ResourceUtilization(
                    cpu_utilization=cpu_util,
                    memory_utilization=memory_util,
                    gpu_utilization=gpu_util,
                    storage_utilization=75 + np.random.normal(0, 5),
                    network_utilization=40 + np.random.normal(0, 10),
                    avg_response_time=0.2 + np.random.exponential(0.1),
                    error_rate=np.random.exponential(1.0),
                    throughput=50 + np.random.normal(0, 10)
                )
                
                # Simulate costs
                costs = {
                    "infrastructure": 15.0 + np.random.normal(0, 2),
                    "ai_models": 25.0 + np.random.normal(0, 5),
                    "storage": 3.0 + np.random.normal(0, 0.5),
                    "network": 2.0 + np.random.normal(0, 0.3)
                }
                
                optimizer.add_usage_sample(utilization, costs)
        
        # Generate and display results
        dashboard = OptimizationDashboard(optimizer)
        dashboard.display_dashboard()
        
        # Export report
        strategy = OptimizationStrategy(args.strategy)
        report_file = optimizer.export_optimization_report(strategy)
        print(f"📄 Detailed report exported to: {report_file}")
        
        # Budget monitoring demo
        if args.budget:
            alerts = optimizer.monitor_budget(args.budget)
            if alerts:
                print("\n🚨 Budget Alerts:")
                for alert in alerts:
                    level_icon = "🔴" if alert.level == AlertLevel.CRITICAL else "🟡"
                    print(f"   {level_icon} {alert.message}")
                    print(f"      💡 {alert.recommendation}")
    
    elif args.monitor:
        print("📡 Starting real-time cost monitoring...")
        print("   Press Ctrl+C to stop")
        print()
        
        dashboard = OptimizationDashboard(optimizer)
        
        try:
            while True:
                # In real implementation, this would collect actual metrics
                # For demo, we'll simulate real-time data
                utilization = ResourceUtilization(
                    cpu_utilization=np.random.uniform(20, 80),
                    memory_utilization=np.random.uniform(30, 70),
                    gpu_utilization=np.random.uniform(10, 90),
                    storage_utilization=np.random.uniform(60, 85),
                    network_utilization=np.random.uniform(20, 60)
                )
                
                costs = {
                    "infrastructure": np.random.uniform(10, 20),
                    "ai_models": np.random.uniform(20, 40),
                    "storage": np.random.uniform(2, 5),
                    "network": np.random.uniform(1, 3)
                }
                
                optimizer.add_usage_sample(utilization, costs)
                
                # Clear screen and display dashboard
                print("\033[2J\033[H")  # Clear screen
                dashboard.display_dashboard()
                
                # Budget monitoring
                if args.budget:
                    alerts = optimizer.monitor_budget(args.budget)
                    if alerts:
                        print("🚨 New Budget Alerts:")
                        for alert in alerts[-3:]:  # Show last 3 alerts
                            level_icon = "🔴" if alert.level == AlertLevel.CRITICAL else "🟡"
                            print(f"   {level_icon} {alert.message}")
                
                time.sleep(10)  # Update every 10 seconds
                
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped.")
    
    else:
        print("🎯 PRSM Cost Optimizer")
        print("Use --demo to see sample analysis or --monitor for real-time monitoring")
        print("Run with --help for all options")


if __name__ == "__main__":
    main()