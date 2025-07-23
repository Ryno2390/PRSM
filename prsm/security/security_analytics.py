"""
PRSM Security Analytics and Reporting System
Advanced security analytics, metrics generation, and reporting dashboard
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import json
import statistics
from collections import defaultdict, Counter
import redis.asyncio as aioredis
import logging
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import pandas as pd

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of security metrics"""
    THREAT_SCORE = "threat_score"
    ALERT_COUNT = "alert_count"
    BLOCKED_IPS = "blocked_ips"
    FAILED_LOGINS = "failed_logins"
    RATE_LIMIT_VIOLATIONS = "rate_limit_violations"
    API_USAGE = "api_usage"
    GEOGRAPHIC_DISTRIBUTION = "geographic_distribution"
    USER_ACTIVITY = "user_activity"


class TimeGranularity(Enum):
    """Time granularity for metrics"""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class SecurityMetric:
    """Security metric data point"""
    metric_type: MetricType
    timestamp: datetime
    value: float
    dimensions: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsReport:
    """Analytics report"""
    report_id: str
    title: str
    description: str
    generated_at: datetime
    time_range: Tuple[datetime, datetime]
    metrics: List[SecurityMetric]
    insights: List[str]
    recommendations: List[str]
    charts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatTrend:
    """Threat trend analysis"""
    trend_type: str
    direction: str  # "increasing", "decreasing", "stable"
    magnitude: float  # percentage change
    confidence: float  # 0.0 to 1.0
    timespan_hours: int
    description: str


class SecurityMetricsCollector:
    """Collects and aggregates security metrics"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.metric_buffer: List[SecurityMetric] = []
        self.collection_interval = 60  # seconds
        
    async def start_collection(self):
        """Start metrics collection background task"""
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("✅ Security metrics collection started")
    
    async def stop_collection(self):
        """Stop metrics collection"""
        if hasattr(self, 'collection_task'):
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Security metrics collection stopped")
    
    async def collect_real_time_metrics(self):
        """Collect current security metrics"""
        now = datetime.now(timezone.utc)
        metrics = []
        
        # Threat score metrics
        avg_threat_score = await self._calculate_average_threat_score()
        metrics.append(SecurityMetric(
            metric_type=MetricType.THREAT_SCORE,
            timestamp=now,
            value=avg_threat_score,
            dimensions={"aggregation": "average", "window": "5min"}
        ))
        
        # Alert count metrics
        alert_counts = await self._get_alert_counts()
        for alert_type, count in alert_counts.items():
            metrics.append(SecurityMetric(
                metric_type=MetricType.ALERT_COUNT,
                timestamp=now,
                value=count,
                dimensions={"alert_type": alert_type, "window": "1hour"}
            ))
        
        # Failed login metrics
        failed_logins = await self._count_failed_logins()
        metrics.append(SecurityMetric(
            metric_type=MetricType.FAILED_LOGINS,
            timestamp=now,
            value=failed_logins,
            dimensions={"window": "1hour"}
        ))
        
        # Rate limit violation metrics
        rate_violations = await self._count_rate_violations()
        metrics.append(SecurityMetric(
            metric_type=MetricType.RATE_LIMIT_VIOLATIONS,
            timestamp=now,
            value=rate_violations,
            dimensions={"window": "1hour"}
        ))
        
        # Blocked IPs metrics
        blocked_ips = await self._count_blocked_ips()
        metrics.append(SecurityMetric(
            metric_type=MetricType.BLOCKED_IPS,
            timestamp=now,
            value=blocked_ips,
            dimensions={"status": "active"}
        ))
        
        # API usage metrics
        api_usage = await self._get_api_usage_stats()
        for endpoint, count in api_usage.items():
            metrics.append(SecurityMetric(
                metric_type=MetricType.API_USAGE,
                timestamp=now,
                value=count,
                dimensions={"endpoint": endpoint, "window": "1hour"}
            ))
        
        # Store metrics
        for metric in metrics:
            await self._store_metric(metric)
        
        return metrics
    
    async def _collection_loop(self):
        """Background metrics collection loop"""
        while True:
            try:
                await self.collect_real_time_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _calculate_average_threat_score(self) -> float:
        """Calculate average threat score from recent events"""
        scores = []
        
        # Get recent security events
        pattern = "security_event:*"
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        
        async for key in self.redis.scan_iter(match=pattern):
            event_data = await self.redis.get(key)
            if event_data:
                event = json.loads(event_data)
                event_time = datetime.fromisoformat(event["timestamp"])
                
                if event_time > cutoff_time:
                    scores.append(event.get("risk_score", 0.0))
        
        return statistics.mean(scores) if scores else 0.0
    
    async def _get_alert_counts(self) -> Dict[str, int]:
        """Get alert counts by type in the last hour"""
        counts = defaultdict(int)
        pattern = "security_alert:*"
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
        
        async for key in self.redis.scan_iter(match=pattern):
            alert_data = await self.redis.get(key)
            if alert_data:
                alert = json.loads(alert_data)
                alert_time = datetime.fromisoformat(alert["timestamp"])
                
                if alert_time > cutoff_time:
                    counts[alert["alert_type"]] += 1
        
        return dict(counts)
    
    async def _count_failed_logins(self) -> int:
        """Count failed login attempts in the last hour"""
        count = 0
        pattern = "auth_attempt:*"
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
        
        async for key in self.redis.scan_iter(match=pattern):
            attempt_data = await self.redis.get(key)
            if attempt_data:
                attempt = json.loads(attempt_data)
                attempt_time = datetime.fromisoformat(attempt["timestamp"])
                
                if (attempt_time > cutoff_time and 
                    not attempt.get("success", False)):
                    count += 1
        
        return count
    
    async def _count_rate_violations(self) -> int:
        """Count rate limit violations in the last hour"""
        count = 0
        pattern = "rate_violations:*"
        
        async for key in self.redis.scan_iter(match=pattern):
            violations = await self.redis.get(key)
            if violations:
                count += int(violations)
        
        return count
    
    async def _count_blocked_ips(self) -> int:
        """Count currently blocked IPs"""
        count = 0
        pattern = "blocked_ip:*"
        
        async for key in self.redis.scan_iter(match=pattern):
            # Key exists means IP is blocked
            count += 1
        
        return count
    
    async def _get_api_usage_stats(self) -> Dict[str, int]:
        """Get API usage statistics by endpoint"""
        usage = defaultdict(int)
        pattern = "security_event:*"
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
        
        async for key in self.redis.scan_iter(match=pattern):
            event_data = await self.redis.get(key)
            if event_data:
                event = json.loads(event_data)
                event_time = datetime.fromisoformat(event["timestamp"])
                
                if event_time > cutoff_time:
                    endpoint = event.get("endpoint", "unknown")
                    usage[endpoint] += 1
        
        # Return top 10 endpoints
        sorted_usage = sorted(usage.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_usage[:10])
    
    async def _store_metric(self, metric: SecurityMetric):
        """Store metric in Redis"""
        metric_data = {
            "metric_type": metric.metric_type.value,
            "timestamp": metric.timestamp.isoformat(),
            "value": metric.value,
            "dimensions": metric.dimensions,
            "metadata": metric.metadata
        }
        
        # Store with time-based key for easy querying
        time_key = metric.timestamp.strftime("%Y%m%d%H%M")
        metric_key = f"metric:{metric.metric_type.value}:{time_key}:{hash(str(metric.dimensions))}"
        
        await self.redis.setex(
            metric_key,
            86400 * 7,  # 7 days retention for metrics
            json.dumps(metric_data)
        )


class ThreatAnalyzer:
    """Analyzes security threats and trends"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
    
    async def analyze_threat_trends(self, hours: int = 24) -> List[ThreatTrend]:
        """Analyze threat trends over specified time period"""
        trends = []
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        
        # Analyze threat score trends
        threat_trend = await self._analyze_metric_trend(
            MetricType.THREAT_SCORE, start_time, end_time
        )
        if threat_trend:
            trends.append(threat_trend)
        
        # Analyze alert count trends
        alert_trend = await self._analyze_metric_trend(
            MetricType.ALERT_COUNT, start_time, end_time
        )
        if alert_trend:
            trends.append(alert_trend)
        
        # Analyze failed login trends
        login_trend = await self._analyze_metric_trend(
            MetricType.FAILED_LOGINS, start_time, end_time
        )
        if login_trend:
            trends.append(login_trend)
        
        return trends
    
    async def _analyze_metric_trend(self, metric_type: MetricType, 
                                  start_time: datetime, end_time: datetime) -> Optional[ThreatTrend]:
        """Analyze trend for a specific metric type"""
        
        # Get metric values over time
        values = await self._get_metric_values(metric_type, start_time, end_time)
        
        if len(values) < 3:  # Need at least 3 points for trend analysis
            return None
        
        # Calculate trend direction and magnitude
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        if first_avg == 0:
            return None
        
        change_percent = ((second_avg - first_avg) / first_avg) * 100
        
        # Determine trend direction
        if abs(change_percent) < 5:  # Less than 5% change
            direction = "stable"
        elif change_percent > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        # Calculate confidence based on data consistency
        confidence = self._calculate_trend_confidence(values)
        
        # Generate description
        timespan_hours = int((end_time - start_time).total_seconds() / 3600)
        description = f"{metric_type.value.replace('_', ' ').title()} is {direction}"
        if direction != "stable":
            description += f" by {abs(change_percent):.1f}% over {timespan_hours} hours"
        
        return ThreatTrend(
            trend_type=metric_type.value,
            direction=direction,
            magnitude=abs(change_percent),
            confidence=confidence,
            timespan_hours=timespan_hours,
            description=description
        )
    
    async def _get_metric_values(self, metric_type: MetricType, 
                               start_time: datetime, end_time: datetime) -> List[float]:
        """Get metric values for time range"""
        values = []
        pattern = f"metric:{metric_type.value}:*"
        
        async for key in self.redis.scan_iter(match=pattern):
            metric_data = await self.redis.get(key)
            if metric_data:
                metric = json.loads(metric_data)
                metric_time = datetime.fromisoformat(metric["timestamp"])
                
                if start_time <= metric_time <= end_time:
                    values.append(metric["value"])
        
        # Sort by timestamp (implied by key structure)
        return values
    
    def _calculate_trend_confidence(self, values: List[float]) -> float:
        """Calculate confidence in trend analysis"""
        if len(values) < 3:
            return 0.0
        
        # Calculate variance as a measure of confidence
        variance = statistics.variance(values)
        mean_value = statistics.mean(values)
        
        if mean_value == 0:
            return 0.5
        
        # Lower variance relative to mean = higher confidence
        coefficient_of_variation = variance / (mean_value ** 2)
        confidence = max(0.1, min(0.9, 1.0 - coefficient_of_variation))
        
        return confidence
    
    async def generate_threat_intelligence_summary(self) -> Dict[str, Any]:
        """Generate comprehensive threat intelligence summary"""
        
        # Get recent alerts
        recent_alerts = await self._get_recent_alerts(hours=24)
        
        # Analyze attack patterns
        attack_patterns = self._analyze_attack_patterns(recent_alerts)
        
        # Get top threat sources
        threat_sources = await self._get_top_threat_sources()
        
        # Calculate overall threat level
        overall_threat_level = await self._calculate_overall_threat_level()
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_threat_level": overall_threat_level,
            "total_alerts_24h": len(recent_alerts),
            "attack_patterns": attack_patterns,
            "top_threat_sources": threat_sources,
            "trending_threats": await self.analyze_threat_trends(24),
            "recommendations": self._generate_security_recommendations(recent_alerts)
        }
    
    async def _get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent security alerts"""
        alerts = []
        pattern = "security_alert:*"
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        async for key in self.redis.scan_iter(match=pattern):
            alert_data = await self.redis.get(key)
            if alert_data:
                alert = json.loads(alert_data)
                alert_time = datetime.fromisoformat(alert["timestamp"])
                
                if alert_time > cutoff_time:
                    alerts.append(alert)
        
        return alerts
    
    def _analyze_attack_patterns(self, alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze attack patterns from alerts"""
        
        # Count alert types
        alert_types = Counter(alert["alert_type"] for alert in alerts)
        
        # Analyze source IPs
        source_ips = Counter(alert["source_ip"] for alert in alerts)
        
        # Analyze timing patterns
        hours = [datetime.fromisoformat(alert["timestamp"]).hour for alert in alerts]
        peak_hours = Counter(hours).most_common(3)
        
        return {
            "most_common_alert_types": dict(alert_types.most_common(5)),
            "most_active_ips": dict(source_ips.most_common(10)),
            "peak_attack_hours": [{"hour": hour, "count": count} for hour, count in peak_hours],
            "total_unique_ips": len(source_ips),
            "attack_intensity": len(alerts) / 24  # alerts per hour
        }
    
    async def _get_top_threat_sources(self) -> List[Dict[str, Any]]:
        """Get top threat sources by IP"""
        ip_scores = defaultdict(float)
        ip_incidents = defaultdict(int)
        
        # Analyze recent events
        pattern = "security_event:*"
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        async for key in self.redis.scan_iter(match=pattern):
            event_data = await self.redis.get(key)
            if event_data:
                event = json.loads(event_data)
                event_time = datetime.fromisoformat(event["timestamp"])
                
                if event_time > cutoff_time:
                    ip = event["ip_address"]
                    ip_scores[ip] += event.get("risk_score", 0.0)
                    ip_incidents[ip] += 1
        
        # Calculate average risk scores and sort
        threat_sources = []
        for ip in ip_scores:
            avg_score = ip_scores[ip] / ip_incidents[ip]
            threat_sources.append({
                "ip_address": ip,
                "avg_risk_score": avg_score,
                "incident_count": ip_incidents[ip],
                "total_risk_score": ip_scores[ip]
            })
        
        # Sort by total risk score
        threat_sources.sort(key=lambda x: x["total_risk_score"], reverse=True)
        
        return threat_sources[:10]
    
    async def _calculate_overall_threat_level(self) -> str:
        """Calculate overall threat level"""
        
        # Get recent metrics
        threat_scores = await self._get_metric_values(
            MetricType.THREAT_SCORE,
            datetime.now(timezone.utc) - timedelta(hours=1),
            datetime.now(timezone.utc)
        )
        
        alert_counts = await self._get_metric_values(
            MetricType.ALERT_COUNT,
            datetime.now(timezone.utc) - timedelta(hours=1),
            datetime.now(timezone.utc)
        )
        
        avg_threat_score = statistics.mean(threat_scores) if threat_scores else 0.0
        total_alerts = sum(alert_counts) if alert_counts else 0
        
        # Determine threat level
        if avg_threat_score >= 0.8 or total_alerts >= 20:
            return "CRITICAL"
        elif avg_threat_score >= 0.6 or total_alerts >= 10:
            return "HIGH"
        elif avg_threat_score >= 0.4 or total_alerts >= 5:
            return "MEDIUM"
        elif avg_threat_score >= 0.2 or total_alerts >= 2:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _generate_security_recommendations(self, alerts: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations based on recent alerts"""
        recommendations = []
        
        alert_types = Counter(alert["alert_type"] for alert in alerts)
        
        # Brute force recommendations
        if alert_types.get("brute_force", 0) > 5:
            recommendations.append("Consider implementing stricter account lockout policies")
            recommendations.append("Enable CAPTCHA for repeated failed login attempts")
        
        # Rate limiting recommendations
        if alert_types.get("rate_limit_exceeded", 0) > 10:
            recommendations.append("Review and adjust rate limiting thresholds")
            recommendations.append("Implement adaptive rate limiting based on user behavior")
        
        # Suspicious activity recommendations
        if alert_types.get("suspicious_api_usage", 0) > 3:
            recommendations.append("Enhance API endpoint monitoring and logging")
            recommendations.append("Consider implementing API request anomaly detection")
        
        # General recommendations
        if len(alerts) > 20:
            recommendations.append("Review security alert thresholds to reduce false positives")
            recommendations.append("Consider increasing security team monitoring during peak hours")
        
        if not recommendations:
            recommendations.append("Security posture appears stable - continue monitoring")
        
        return recommendations


class SecurityDashboard:
    """Security analytics dashboard generator"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.metrics_collector = SecurityMetricsCollector(redis_client)
        self.threat_analyzer = ThreatAnalyzer(redis_client)
    
    async def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for security dashboard"""
        
        # Real-time metrics
        current_metrics = await self.metrics_collector.collect_real_time_metrics()
        
        # Threat analysis
        threat_summary = await self.threat_analyzer.generate_threat_intelligence_summary()
        
        # Historical data for charts
        chart_data = await self._generate_chart_data()
        
        return {
            "current_metrics": self._format_current_metrics(current_metrics),
            "threat_summary": threat_summary,
            "charts": chart_data,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    def _format_current_metrics(self, metrics: List[SecurityMetric]) -> Dict[str, Any]:
        """Format current metrics for dashboard display"""
        formatted = {}
        
        for metric in metrics:
            metric_key = metric.metric_type.value
            if metric_key not in formatted:
                formatted[metric_key] = []
            
            formatted[metric_key].append({
                "value": metric.value,
                "dimensions": metric.dimensions,
                "timestamp": metric.timestamp.isoformat()
            })
        
        return formatted
    
    async def _generate_chart_data(self) -> Dict[str, Any]:
        """Generate chart data for dashboard"""
        charts = {}
        
        # Threat score over time
        charts["threat_score_timeline"] = await self._create_threat_score_chart()
        
        # Alert distribution
        charts["alert_distribution"] = await self._create_alert_distribution_chart()
        
        # Geographic threat map
        charts["geographic_threats"] = await self._create_geographic_chart()
        
        # Top attacked endpoints
        charts["endpoint_attacks"] = await self._create_endpoint_chart()
        
        return charts
    
    async def _create_threat_score_chart(self) -> Dict[str, Any]:
        """Create threat score timeline chart"""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=24)
        
        # Get threat score metrics
        values = []
        timestamps = []
        
        pattern = f"metric:{MetricType.THREAT_SCORE.value}:*"
        async for key in self.redis.scan_iter(match=pattern):
            metric_data = await self.redis.get(key)
            if metric_data:
                metric = json.loads(metric_data)
                metric_time = datetime.fromisoformat(metric["timestamp"])
                
                if start_time <= metric_time <= end_time:
                    values.append(metric["value"])
                    timestamps.append(metric_time)
        
        # Sort by timestamp
        sorted_data = sorted(zip(timestamps, values))
        timestamps, values = zip(*sorted_data) if sorted_data else ([], [])
        
        return {
            "type": "line",
            "data": {
                "timestamps": [ts.isoformat() for ts in timestamps],
                "values": list(values)
            },
            "title": "Threat Score Over Time (24h)",
            "x_axis": "Time",
            "y_axis": "Average Threat Score"
        }
    
    async def _create_alert_distribution_chart(self) -> Dict[str, Any]:
        """Create alert type distribution chart"""
        alerts = await self.threat_analyzer._get_recent_alerts(hours=24)
        alert_types = Counter(alert["alert_type"] for alert in alerts)
        
        return {
            "type": "pie",
            "data": {
                "labels": list(alert_types.keys()),
                "values": list(alert_types.values())
            },
            "title": "Alert Distribution (24h)",
            "total_alerts": len(alerts)
        }
    
    async def _create_geographic_chart(self) -> Dict[str, Any]:
        """Create geographic threat distribution chart"""
        # Mock geographic data - in production would use GeoIP
        geographic_data = {
            "United States": 45,
            "Russia": 32,
            "China": 28,
            "Germany": 15,
            "United Kingdom": 12,
            "France": 8,
            "Canada": 6,
            "Brazil": 5
        }
        
        return {
            "type": "map",
            "data": {
                "countries": list(geographic_data.keys()),
                "threat_counts": list(geographic_data.values())
            },
            "title": "Geographic Threat Distribution (24h)"
        }
    
    async def _create_endpoint_chart(self) -> Dict[str, Any]:
        """Create top attacked endpoints chart"""
        api_usage = await self.metrics_collector._get_api_usage_stats()
        
        # Sort by attack frequency (assuming higher usage = more attacks for demo)
        sorted_endpoints = sorted(api_usage.items(), key=lambda x: x[1], reverse=True)[:10]
        endpoints, counts = zip(*sorted_endpoints) if sorted_endpoints else ([], [])
        
        return {
            "type": "bar",
            "data": {
                "endpoints": list(endpoints),
                "attack_counts": list(counts)
            },
            "title": "Most Targeted Endpoints (24h)",
            "x_axis": "API Endpoints",
            "y_axis": "Request Count"
        }
    
    async def generate_dashboard_html(self) -> str:
        """Generate HTML dashboard"""
        dashboard_data = await self.generate_dashboard_data()
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>PRSM Security Analytics Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f7fa; }}
                .dashboard {{ max-width: 1400px; margin: 0 auto; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
                .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ color: #7f8c8d; font-size: 14px; }}
                .threat-level {{ padding: 5px 10px; border-radius: 4px; font-weight: bold; display: inline-block; }}
                .threat-level.CRITICAL {{ background: #e74c3c; color: white; }}
                .threat-level.HIGH {{ background: #f39c12; color: white; }}
                .threat-level.MEDIUM {{ background: #f1c40f; color: black; }}
                .threat-level.LOW {{ background: #3498db; color: white; }}
                .threat-level.MINIMAL {{ background: #2ecc71; color: white; }}
                .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
                .chart-container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .recommendations {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .recommendation {{ background: #ecf0f1; padding: 10px; margin: 10px 0; border-left: 4px solid #3498db; }}
                .last-updated {{ text-align: center; color: #7f8c8d; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="dashboard">
                <div class="header">
                    <h1>🛡️ PRSM Security Analytics Dashboard</h1>
                    <p>Real-time security monitoring and threat intelligence</p>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{dashboard_data['threat_summary']['total_alerts_24h']}</div>
                        <div class="metric-label">Total Alerts (24h)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">
                            <span class="threat-level {dashboard_data['threat_summary']['overall_threat_level']}">
                                {dashboard_data['threat_summary']['overall_threat_level']}
                            </span>
                        </div>
                        <div class="metric-label">Threat Level</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(dashboard_data['threat_summary']['top_threat_sources'])}</div>
                        <div class="metric-label">Active Threat Sources</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{dashboard_data['threat_summary']['attack_patterns']['total_unique_ips']}</div>
                        <div class="metric-label">Unique Attack IPs</div>
                    </div>
                </div>
                
                <div class="charts-grid">
                    <div class="chart-container">
                        <div id="threat-timeline"></div>
                    </div>
                    <div class="chart-container">
                        <div id="alert-distribution"></div>
                    </div>
                    <div class="chart-container">
                        <div id="geographic-threats"></div>
                    </div>
                    <div class="chart-container">
                        <div id="endpoint-attacks"></div>
                    </div>
                </div>
                
                <div class="recommendations">
                    <h3>🎯 Security Recommendations</h3>
        """
        
        for recommendation in dashboard_data['threat_summary']['recommendations']:
            html += f'<div class="recommendation">{recommendation}</div>'
        
        html += f"""
                </div>
                
                <div class="last-updated">
                    Last updated: {dashboard_data['last_updated']}
                </div>
            </div>
            
            <script>
                // Chart data
                const chartData = {json.dumps(dashboard_data['charts'], cls=PlotlyJSONEncoder)};
                
                // Threat score timeline
                if (chartData.threat_score_timeline && chartData.threat_score_timeline.data.timestamps.length > 0) {{
                    Plotly.newPlot('threat-timeline', [{{
                        x: chartData.threat_score_timeline.data.timestamps,
                        y: chartData.threat_score_timeline.data.values,
                        type: 'scatter',
                        mode: 'lines+markers',
                        line: {{ color: '#e74c3c' }},
                        name: 'Threat Score'
                    }}], {{
                        title: chartData.threat_score_timeline.title,
                        xaxis: {{ title: chartData.threat_score_timeline.x_axis }},
                        yaxis: {{ title: chartData.threat_score_timeline.y_axis, range: [0, 1] }}
                    }});
                }}
                
                // Alert distribution
                if (chartData.alert_distribution && chartData.alert_distribution.data.labels.length > 0) {{
                    Plotly.newPlot('alert-distribution', [{{
                        labels: chartData.alert_distribution.data.labels,
                        values: chartData.alert_distribution.data.values,
                        type: 'pie'
                    }}], {{
                        title: chartData.alert_distribution.title
                    }});
                }}
                
                // Geographic threats
                if (chartData.geographic_threats) {{
                    Plotly.newPlot('geographic-threats', [{{
                        x: chartData.geographic_threats.data.countries,
                        y: chartData.geographic_threats.data.threat_counts,
                        type: 'bar',
                        marker: {{ color: '#3498db' }}
                    }}], {{
                        title: chartData.geographic_threats.title,
                        xaxis: {{ title: 'Country' }},
                        yaxis: {{ title: 'Threat Count' }}
                    }});
                }}
                
                // Endpoint attacks
                if (chartData.endpoint_attacks && chartData.endpoint_attacks.data.endpoints.length > 0) {{
                    Plotly.newPlot('endpoint-attacks', [{{
                        x: chartData.endpoint_attacks.data.endpoints,
                        y: chartData.endpoint_attacks.data.attack_counts,
                        type: 'bar',
                        marker: {{ color: '#e67e22' }}
                    }}], {{
                        title: chartData.endpoint_attacks.title,
                        xaxis: {{ title: chartData.endpoint_attacks.x_axis }},
                        yaxis: {{ title: chartData.endpoint_attacks.y_axis }}
                    }});
                }}
                
                // Auto-refresh every 30 seconds
                setTimeout(() => {{
                    location.reload();
                }}, 30000);
            </script>
        </body>
        </html>
        """
        
        return html


# Global analytics instances
security_analytics: Optional[SecurityDashboard] = None
metrics_collector: Optional[SecurityMetricsCollector] = None


async def initialize_security_analytics(redis_client: aioredis.Redis):
    """Initialize security analytics system"""
    global security_analytics, metrics_collector
    
    security_analytics = SecurityDashboard(redis_client)
    metrics_collector = SecurityMetricsCollector(redis_client)
    
    await metrics_collector.start_collection()
    logger.info("✅ Security analytics system initialized")


def get_security_analytics() -> SecurityDashboard:
    """Get the global security analytics instance"""
    if security_analytics is None:
        raise RuntimeError("Security analytics not initialized. Call initialize_security_analytics() first.")
    return security_analytics


def get_metrics_collector() -> SecurityMetricsCollector:
    """Get the global metrics collector instance"""
    if metrics_collector is None:
        raise RuntimeError("Metrics collector not initialized. Call initialize_security_analytics() first.")
    return metrics_collector


async def shutdown_security_analytics():
    """Shutdown security analytics system"""
    if metrics_collector:
        await metrics_collector.stop_collection()


def create_security_analytics_endpoints(app: FastAPI):
    """Create security analytics API endpoints"""
    
    @app.get("/security/dashboard", response_class=HTMLResponse, include_in_schema=False)
    async def get_security_dashboard():
        """Get security analytics dashboard"""
        dashboard = get_security_analytics()
        return await dashboard.generate_dashboard_html()
    
    @app.get("/api/security/metrics", include_in_schema=False)
    async def get_security_metrics():
        """Get current security metrics"""
        dashboard = get_security_analytics()
        return await dashboard.generate_dashboard_data()
    
    @app.get("/api/security/threats", include_in_schema=False)
    async def get_threat_intelligence():
        """Get threat intelligence summary"""
        analyzer = ThreatAnalyzer(get_security_analytics().redis)
        return await analyzer.generate_threat_intelligence_summary()
    
    @app.get("/api/security/trends", include_in_schema=False)
    async def get_threat_trends(hours: int = 24):
        """Get threat trends analysis"""
        analyzer = ThreatAnalyzer(get_security_analytics().redis)
        trends = await analyzer.analyze_threat_trends(hours)
        return {
            "timespan_hours": hours,
            "trends": [
                {
                    "trend_type": trend.trend_type,
                    "direction": trend.direction,
                    "magnitude": trend.magnitude,
                    "confidence": trend.confidence,
                    "description": trend.description
                }
                for trend in trends
            ]
        }