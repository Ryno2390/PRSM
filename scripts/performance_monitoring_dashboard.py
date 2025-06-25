#!/usr/bin/env python3
"""
PRSM Performance Monitoring Dashboard

Real-time performance monitoring and regression detection system that tracks:
- RLT component performance metrics
- System health indicators  
- Performance regression detection
- Historical performance trends
"""

import asyncio
import json
import time
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import threading
from dataclasses import dataclass, asdict
import argparse

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    commit_sha: str
    branch: str
    rlt_success_rate: float
    working_components: int
    total_components: int
    integration_gaps: int
    avg_component_performance: float
    min_component_performance: float
    max_component_performance: float
    execution_time: float
    memory_usage_mb: float
    cpu_percent: float
    overall_score: int


class PerformanceDatabase:
    """SQLite database for storing performance metrics"""
    
    def __init__(self, db_path: str = "performance_history.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the performance metrics database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                commit_sha TEXT,
                branch TEXT,
                rlt_success_rate REAL,
                working_components INTEGER,
                total_components INTEGER,
                integration_gaps INTEGER,
                avg_component_performance REAL,
                min_component_performance REAL,
                max_component_performance REAL,
                execution_time REAL,
                memory_usage_mb REAL,
                cpu_percent REAL,
                overall_score INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON performance_metrics(timestamp);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_commit_sha ON performance_metrics(commit_sha);
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Performance database initialized: {self.db_path}")
    
    def store_metrics(self, metrics: PerformanceMetrics):
        """Store performance metrics in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO performance_metrics (
                timestamp, commit_sha, branch, rlt_success_rate,
                working_components, total_components, integration_gaps,
                avg_component_performance, min_component_performance, max_component_performance,
                execution_time, memory_usage_mb, cpu_percent, overall_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.timestamp.isoformat(),
            metrics.commit_sha,
            metrics.branch,
            metrics.rlt_success_rate,
            metrics.working_components,
            metrics.total_components,
            metrics.integration_gaps,
            metrics.avg_component_performance,
            metrics.min_component_performance,
            metrics.max_component_performance,
            metrics.execution_time,
            metrics.memory_usage_mb,
            metrics.cpu_percent,
            metrics.overall_score
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored performance metrics for commit {metrics.commit_sha[:8]}")
    
    def get_recent_metrics(self, days: int = 7) -> List[Dict]:
        """Get performance metrics from the last N days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("""
            SELECT * FROM performance_metrics 
            WHERE timestamp >= ? 
            ORDER BY timestamp DESC
        """, (since_date,))
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_baseline_metrics(self, branch: str = "main") -> Optional[Dict]:
        """Get baseline performance metrics for a branch"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM performance_metrics 
            WHERE branch = ? AND rlt_success_rate = 1.0 AND integration_gaps = 0
            ORDER BY timestamp DESC 
            LIMIT 1
        """, (branch,))
        
        result = cursor.fetchone()
        if result:
            columns = [description[0] for description in cursor.description]
            baseline = dict(zip(columns, result))
        else:
            baseline = None
        
        conn.close()
        return baseline


class PerformanceMonitor:
    """Main performance monitoring and regression detection system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.db = PerformanceDatabase()
        self.monitoring = False
        
        # Performance thresholds
        self.rlt_success_threshold = 1.0  # 100% success rate required
        self.performance_regression_threshold = 0.1  # 10% performance drop
        self.memory_regression_threshold = 0.2  # 20% memory increase
        
        logger.info("Performance monitor initialized")
    
    async def run_performance_test(self) -> PerformanceMetrics:
        """Run comprehensive performance test and return metrics"""
        logger.info("🎯 Running comprehensive performance test...")
        
        start_time = time.time()
        
        # Get current git information
        try:
            commit_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                cwd=os.getcwd()
            ).decode().strip()
            
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], 
                cwd=os.getcwd()
            ).decode().strip()
        except:
            commit_sha = "unknown"
            branch = "unknown"
        
        # Run RLT integration test
        rlt_metrics = await self._run_rlt_performance_test()
        
        # Get system resource usage
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        execution_time = time.time() - start_time
        
        # Calculate overall performance score
        overall_score = self._calculate_performance_score(rlt_metrics, execution_time)
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            commit_sha=commit_sha,
            branch=branch,
            rlt_success_rate=rlt_metrics.get("success_rate", 0.0),
            working_components=rlt_metrics.get("working_components", 0),
            total_components=rlt_metrics.get("total_components", 11),
            integration_gaps=rlt_metrics.get("integration_gaps", 1),
            avg_component_performance=rlt_metrics.get("avg_performance", 0.0),
            min_component_performance=rlt_metrics.get("min_performance", 0.0),
            max_component_performance=rlt_metrics.get("max_performance", 0.0),
            execution_time=execution_time,
            memory_usage_mb=memory_usage_mb,
            cpu_percent=cpu_percent,
            overall_score=overall_score
        )
        
        # Store in database
        self.db.store_metrics(metrics)
        
        logger.info(f"Performance test completed in {execution_time:.2f}s")
        return metrics
    
    async def _run_rlt_performance_test(self) -> Dict[str, float]:
        """Run RLT integration test and extract performance metrics"""
        try:
            # Run the RLT system integration test
            env = os.environ.copy()
            env["PYTHONPATH"] = os.getcwd()
            
            result = subprocess.run(
                ["python3", "tests/test_rlt_system_integration.py"],
                capture_output=True,
                text=True,
                env=env,
                cwd=os.getcwd()
            )
            
            # Check for the integration report
            report_path = "rlt_system_integration_report.json"
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    report_data = json.load(f)
                
                summary = report_data.get("summary", {})
                components = report_data.get("components", {})
                
                # Extract component performance metrics
                performances = [
                    comp.get("performance", 0) 
                    for comp in components.values() 
                    if comp.get("performance", 0) > 0
                ]
                
                rlt_metrics = {
                    "success_rate": summary.get("success_rate", 0.0),
                    "working_components": summary.get("working_components", 0),
                    "total_components": summary.get("total_components", 11),
                    "integration_gaps": summary.get("gaps_found", 1),
                    "avg_performance": sum(performances) / len(performances) if performances else 0.0,
                    "min_performance": min(performances) if performances else 0.0,
                    "max_performance": max(performances) if performances else 0.0,
                    "test_output": result.stdout if result.returncode == 0 else result.stderr
                }
                
                logger.info(f"RLT test results: {rlt_metrics['success_rate']*100:.1f}% success rate")
                return rlt_metrics
            else:
                logger.warning("RLT integration report not found")
                return {"success_rate": 0.0, "error": "No report generated"}
                
        except Exception as e:
            logger.error(f"RLT performance test failed: {e}")
            return {"success_rate": 0.0, "error": str(e)}
    
    def _calculate_performance_score(self, rlt_metrics: Dict, execution_time: float) -> int:
        """Calculate overall performance score (0-100)"""
        score = 100
        
        # RLT success rate (50% weight)
        success_rate = rlt_metrics.get("success_rate", 0.0)
        if success_rate < 1.0:
            score -= (1.0 - success_rate) * 50
        
        # Integration gaps (20% weight)
        gaps = rlt_metrics.get("integration_gaps", 1)
        if gaps > 0:
            score -= min(gaps * 10, 20)
        
        # Performance (20% weight)
        avg_perf = rlt_metrics.get("avg_performance", 0.0)
        if avg_perf < 6000:  # Below 6K ops/sec
            score -= 20
        elif avg_perf < 5000:  # Below 5K ops/sec
            score -= 30
        
        # Execution time (10% weight)
        if execution_time > 10:  # Over 10 seconds
            score -= 10
        
        return max(0, int(score))
    
    def detect_regression(self, current: PerformanceMetrics) -> Dict[str, Any]:
        """Detect performance regression compared to baseline"""
        baseline = self.db.get_baseline_metrics(current.branch)
        
        if not baseline:
            return {
                "regression_detected": False,
                "reason": "No baseline metrics available",
                "recommendations": ["Establish baseline by running tests on stable commit"]
            }
        
        regressions = []
        
        # Check RLT success rate
        if current.rlt_success_rate < baseline["rlt_success_rate"]:
            regressions.append({
                "metric": "RLT Success Rate",
                "current": f"{current.rlt_success_rate*100:.1f}%",
                "baseline": f"{baseline['rlt_success_rate']*100:.1f}%",
                "severity": "CRITICAL"
            })
        
        # Check integration gaps
        if current.integration_gaps > baseline["integration_gaps"]:
            regressions.append({
                "metric": "Integration Gaps", 
                "current": current.integration_gaps,
                "baseline": baseline["integration_gaps"],
                "severity": "HIGH"
            })
        
        # Check performance regression
        baseline_perf = baseline["avg_component_performance"]
        if baseline_perf > 0:
            perf_change = (current.avg_component_performance - baseline_perf) / baseline_perf
            if perf_change < -self.performance_regression_threshold:
                regressions.append({
                    "metric": "Component Performance",
                    "current": f"{current.avg_component_performance:,.0f} ops/sec",
                    "baseline": f"{baseline_perf:,.0f} ops/sec",
                    "change": f"{perf_change*100:.1f}%",
                    "severity": "MEDIUM"
                })
        
        # Check memory regression
        baseline_memory = baseline["memory_usage_mb"]
        if baseline_memory > 0:
            memory_change = (current.memory_usage_mb - baseline_memory) / baseline_memory
            if memory_change > self.memory_regression_threshold:
                regressions.append({
                    "metric": "Memory Usage",
                    "current": f"{current.memory_usage_mb:.1f} MB",
                    "baseline": f"{baseline_memory:.1f} MB", 
                    "change": f"+{memory_change*100:.1f}%",
                    "severity": "MEDIUM"
                })
        
        # Generate recommendations
        recommendations = []
        if regressions:
            critical_regressions = [r for r in regressions if r["severity"] == "CRITICAL"]
            if critical_regressions:
                recommendations.append("🚨 CRITICAL: Fix RLT integration issues immediately")
            
            high_regressions = [r for r in regressions if r["severity"] == "HIGH"]
            if high_regressions:
                recommendations.append("⚠️ HIGH: Resolve integration gaps before deployment")
            
            medium_regressions = [r for r in regressions if r["severity"] == "MEDIUM"]
            if medium_regressions:
                recommendations.append("💡 MEDIUM: Investigate performance degradation")
        
        return {
            "regression_detected": len(regressions) > 0,
            "regressions": regressions,
            "baseline_commit": baseline.get("commit_sha", "unknown"),
            "recommendations": recommendations
        }
    
    def generate_performance_report(self, current: PerformanceMetrics) -> str:
        """Generate comprehensive performance report"""
        regression_analysis = self.detect_regression(current)
        
        report = []
        report.append("# 📊 PRSM Performance Report")
        report.append("")
        report.append(f"**Timestamp:** {current.timestamp}")
        report.append(f"**Commit:** {current.commit_sha[:8]}")
        report.append(f"**Branch:** {current.branch}")
        report.append("")
        
        # RLT Integration Status
        report.append("## 🎯 RLT Integration Status")
        report.append("")
        
        if current.rlt_success_rate >= 1.0:
            report.append("✅ **RLT Success Rate: 100%** - Perfect integration")
        else:
            report.append(f"⚠️ **RLT Success Rate: {current.rlt_success_rate*100:.1f}%** - Below target")
        
        report.append(f"- **Working Components:** {current.working_components}/{current.total_components}")
        
        if current.integration_gaps == 0:
            report.append("✅ **Integration Gaps: 0** - Air-tight system")
        else:
            report.append(f"⚠️ **Integration Gaps: {current.integration_gaps}** - System has gaps")
        
        report.append("")
        
        # Performance Metrics
        report.append("## ⚡ Performance Metrics")
        report.append("")
        report.append(f"- **Average Component Performance:** {current.avg_component_performance:,.0f} ops/sec")
        report.append(f"- **Performance Range:** {current.min_component_performance:,.0f} - {current.max_component_performance:,.0f} ops/sec")
        report.append(f"- **Execution Time:** {current.execution_time:.2f} seconds")
        report.append(f"- **Memory Usage:** {current.memory_usage_mb:.1f} MB")
        report.append(f"- **Overall Score:** {current.overall_score}/100")
        report.append("")
        
        # Regression Analysis
        if regression_analysis["regression_detected"]:
            report.append("## 🚨 Regression Analysis")
            report.append("")
            report.append("**Regressions Detected:**")
            
            for regression in regression_analysis["regressions"]:
                severity_icon = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "💡"}.get(regression["severity"], "ℹ️")
                report.append(f"- {severity_icon} **{regression['metric']}**")
                report.append(f"  - Current: {regression['current']}")
                report.append(f"  - Baseline: {regression['baseline']}")
                if "change" in regression:
                    report.append(f"  - Change: {regression['change']}")
            
            report.append("")
            report.append("**Recommendations:**")
            for rec in regression_analysis["recommendations"]:
                report.append(f"- {rec}")
        else:
            report.append("## ✅ No Regressions Detected")
            report.append("")
            report.append("Performance is stable compared to baseline metrics.")
        
        report.append("")
        
        # Overall Assessment
        if current.rlt_success_rate >= 1.0 and current.integration_gaps == 0:
            report.append("🎉 **OVERALL STATUS: EXCELLENT** - Production ready!")
        elif current.rlt_success_rate >= 0.9:
            report.append("⚠️ **OVERALL STATUS: GOOD** - Minor issues to address")
        else:
            report.append("❌ **OVERALL STATUS: NEEDS IMPROVEMENT** - Significant issues detected")
        
        return "\n".join(report)
    
    async def continuous_monitoring(self, interval: int = 300):
        """Run continuous performance monitoring"""
        logger.info(f"Starting continuous monitoring (interval: {interval}s)")
        self.monitoring = True
        
        while self.monitoring:
            try:
                metrics = await self.run_performance_test()
                report = self.generate_performance_report(metrics)
                
                # Save report to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_file = f"performance_report_{timestamp}.md"
                with open(report_file, 'w') as f:
                    f.write(report)
                
                logger.info(f"Performance report saved: {report_file}")
                
                # Check for critical regressions
                regression_analysis = self.detect_regression(metrics)
                if regression_analysis["regression_detected"]:
                    critical_issues = [r for r in regression_analysis["regressions"] if r["severity"] == "CRITICAL"]
                    if critical_issues:
                        logger.error("🚨 CRITICAL PERFORMANCE REGRESSION DETECTED!")
                        for issue in critical_issues:
                            logger.error(f"   - {issue['metric']}: {issue['current']} (was {issue['baseline']})")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring = False
        logger.info("Performance monitoring stopped")


async def main():
    """Main entry point for performance monitoring dashboard"""
    parser = argparse.ArgumentParser(description="PRSM Performance Monitoring Dashboard")
    parser.add_argument("--mode", choices=["single", "continuous"], default="single",
                       help="Run single test or continuous monitoring")
    parser.add_argument("--interval", type=int, default=300,
                       help="Monitoring interval in seconds (default: 300)")
    parser.add_argument("--output", type=str, default="performance_report.md",
                       help="Output file for performance report")
    
    args = parser.parse_args()
    
    monitor = PerformanceMonitor()
    
    if args.mode == "single":
        print("🎯 Running single performance test...")
        metrics = await monitor.run_performance_test()
        report = monitor.generate_performance_report(metrics)
        
        with open(args.output, 'w') as f:
            f.write(report)
        
        print(f"📊 Performance report saved to: {args.output}")
        print(f"📈 Overall Score: {metrics.overall_score}/100")
        print(f"🎯 RLT Success Rate: {metrics.rlt_success_rate*100:.1f}%")
        
    elif args.mode == "continuous":
        print(f"🔄 Starting continuous monitoring (interval: {args.interval}s)")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            await monitor.continuous_monitoring(args.interval)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            print("\n👋 Monitoring stopped by user")


if __name__ == "__main__":
    asyncio.run(main())