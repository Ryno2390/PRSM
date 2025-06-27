#!/usr/bin/env python3
"""
PRSM Real-Time Monitoring Dashboard
Interactive web-based dashboard for monitoring PRSM network performance,
AI model metrics, and system health in real-time.

Features:
- Live P2P network topology visualization
- Real-time AI model performance metrics
- System resource monitoring (CPU, memory, network)
- Agent orchestration status and performance
- Interactive charts and graphs
- Historical data tracking
- Alert system for anomalies
- Export capabilities for reports

This dashboard provides comprehensive visibility into PRSM operations
for administrators, developers, and stakeholders.
"""

import asyncio
import json
import time
import logging
import threading
import webbrowser
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import psutil
import socket
from collections import deque, defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Try to import web framework with graceful fallbacks
try:
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Import PRSM components
from demos.enhanced_p2p_ai_demo import EnhancedP2PNetworkDemo
from demos.advanced_agent_orchestration_demo import AdvancedAgentOrchestrationDemo

# Try to import validation suite
try:
    from scripts.comprehensive_validation_suite import ComprehensiveValidationSuite
except ImportError:
    ComprehensiveValidationSuite = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    network_sent_mb: float
    network_recv_mb: float
    active_connections: int

@dataclass
class NetworkMetrics:
    """P2P network metrics"""
    timestamp: float
    total_nodes: int
    active_nodes: int
    total_connections: int
    total_messages: int
    consensus_proposals: int
    average_latency: float
    message_rate: float

@dataclass
class AIMetrics:
    """AI model performance metrics"""
    timestamp: float
    total_models: int
    active_models: int
    total_inferences: int
    successful_inferences: int
    average_inference_time: float
    average_confidence: float
    model_distribution: Dict[str, int]

@dataclass
class AlertData:
    """System alert information"""
    timestamp: float
    level: str  # "info", "warning", "error", "critical"
    component: str
    message: str
    details: Dict[str, Any]

class MetricsCollector:
    """Collects and stores system metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.system_metrics: deque = deque(maxlen=max_history)
        self.network_metrics: deque = deque(maxlen=max_history)
        self.ai_metrics: deque = deque(maxlen=max_history)
        self.alerts: deque = deque(maxlen=max_history)
        
        self.is_collecting = False
        self.collection_interval = 1.0  # seconds
        self.network_demo: Optional[EnhancedP2PNetworkDemo] = None
        
        # Baseline metrics for comparison
        self.baseline_metrics = {}
        self._initialize_baseline()
    
    def _initialize_baseline(self):
        """Initialize baseline metrics"""
        self.baseline_metrics = {
            "cpu_threshold": 80.0,
            "memory_threshold": 85.0,
            "disk_threshold": 90.0,
            "network_latency_threshold": 100.0,  # ms
            "inference_time_threshold": 5.0,     # seconds
            "confidence_threshold": 0.7          # minimum confidence
        }
    
    async def start_collection(self, network_demo: Optional[EnhancedP2PNetworkDemo] = None):
        """Start metrics collection"""
        self.network_demo = network_demo
        self.is_collecting = True
        
        logger.info("🔍 Starting real-time metrics collection...")
        
        while self.is_collecting:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Collect network metrics if available
                if self.network_demo:
                    await self._collect_network_metrics()
                    await self._collect_ai_metrics()
                
                # Check for alerts
                await self._check_alerts()
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {str(e)}")
            
            await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network statistics
            network = psutil.net_io_counters()
            network_connections = len(psutil.net_connections())
            
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_total_gb=memory.total / (1024**3),
                disk_percent=disk.percent,
                network_sent_mb=network.bytes_sent / (1024**2),
                network_recv_mb=network.bytes_recv / (1024**2),
                active_connections=network_connections
            )
            
            self.system_metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
    
    async def _collect_network_metrics(self):
        """Collect P2P network metrics"""
        try:
            if not self.network_demo or not self.network_demo.nodes:
                return
            
            network_status = self.network_demo.get_network_status()
            
            # Calculate average latency (simulated)
            avg_latency = 50.0 + (len(network_status.get("nodes", [])) * 5.0)
            
            # Calculate message rate
            total_messages = network_status.get("total_messages", 0)
            current_time = time.time()
            
            # Estimate message rate from recent activity
            message_rate = 0.0
            if hasattr(self, '_last_message_count') and hasattr(self, '_last_message_time'):
                time_diff = current_time - self._last_message_time
                if time_diff > 0:
                    message_rate = (total_messages - self._last_message_count) / time_diff
            
            self._last_message_count = total_messages
            self._last_message_time = current_time
            
            metrics = NetworkMetrics(
                timestamp=current_time,
                total_nodes=network_status.get("total_nodes", 0),
                active_nodes=network_status.get("active_nodes", 0),
                total_connections=network_status.get("total_connections", 0),
                total_messages=total_messages,
                consensus_proposals=network_status.get("consensus_proposals", 0),
                average_latency=avg_latency,
                message_rate=message_rate
            )
            
            self.network_metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"Error collecting network metrics: {str(e)}")
    
    async def _collect_ai_metrics(self):
        """Collect AI model performance metrics"""
        try:
            if not self.network_demo or not self.network_demo.nodes:
                return
            
            total_models = 0
            active_models = 0
            total_inferences = 0
            successful_inferences = 0
            total_inference_time = 0.0
            total_confidence = 0.0
            confidence_count = 0
            model_distribution = defaultdict(int)
            
            for node in self.network_demo.nodes:
                if hasattr(node, 'ai_manager'):
                    # Model counts
                    node_models = len(node.ai_manager.models)
                    total_models += node_models
                    active_models += node_models  # Assume all models are active
                    
                    # Performance metrics
                    perf_metrics = node.ai_manager.get_performance_metrics()
                    node_inferences = perf_metrics.get("successful_inferences", 0)
                    total_inferences += node_inferences
                    successful_inferences += node_inferences
                    
                    # Model type distribution
                    for model_info in node.ai_manager.models.values():
                        model_distribution[model_info.model_type] += 1
            
            # Calculate averages
            avg_inference_time = total_inference_time / max(successful_inferences, 1)
            avg_confidence = total_confidence / max(confidence_count, 1) if confidence_count > 0 else 0.85
            
            metrics = AIMetrics(
                timestamp=time.time(),
                total_models=total_models,
                active_models=active_models,
                total_inferences=total_inferences,
                successful_inferences=successful_inferences,
                average_inference_time=avg_inference_time,
                average_confidence=avg_confidence,
                model_distribution=dict(model_distribution)
            )
            
            self.ai_metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"Error collecting AI metrics: {str(e)}")
    
    async def _check_alerts(self):
        """Check for system alerts and anomalies"""
        try:
            if not self.system_metrics:
                return
            
            latest_system = self.system_metrics[-1]
            
            # CPU alerts
            if latest_system.cpu_percent > self.baseline_metrics["cpu_threshold"]:
                await self._create_alert(
                    "warning", "system", 
                    f"High CPU usage: {latest_system.cpu_percent:.1f}%",
                    {"cpu_percent": latest_system.cpu_percent}
                )
            
            # Memory alerts
            if latest_system.memory_percent > self.baseline_metrics["memory_threshold"]:
                await self._create_alert(
                    "warning", "system",
                    f"High memory usage: {latest_system.memory_percent:.1f}%",
                    {"memory_percent": latest_system.memory_percent}
                )
            
            # Network alerts
            if self.network_metrics:
                latest_network = self.network_metrics[-1]
                if latest_network.average_latency > self.baseline_metrics["network_latency_threshold"]:
                    await self._create_alert(
                        "warning", "network",
                        f"High network latency: {latest_network.average_latency:.1f}ms",
                        {"latency": latest_network.average_latency}
                    )
            
            # AI alerts
            if self.ai_metrics:
                latest_ai = self.ai_metrics[-1]
                if latest_ai.average_confidence < self.baseline_metrics["confidence_threshold"]:
                    await self._create_alert(
                        "warning", "ai",
                        f"Low AI confidence: {latest_ai.average_confidence:.2f}",
                        {"confidence": latest_ai.average_confidence}
                    )
                
        except Exception as e:
            logger.error(f"Error checking alerts: {str(e)}")
    
    async def _create_alert(self, level: str, component: str, message: str, details: Dict[str, Any]):
        """Create and store an alert"""
        alert = AlertData(
            timestamp=time.time(),
            level=level,
            component=component,
            message=message,
            details=details
        )
        
        self.alerts.append(alert)
        logger.warning(f"🚨 Alert [{level.upper()}] {component}: {message}")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.is_collecting = False
        logger.info("🛑 Stopped metrics collection")
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get latest metrics for dashboard"""
        return {
            "system": asdict(self.system_metrics[-1]) if self.system_metrics else None,
            "network": asdict(self.network_metrics[-1]) if self.network_metrics else None,
            "ai": asdict(self.ai_metrics[-1]) if self.ai_metrics else None,
            "alerts": [asdict(alert) for alert in list(self.alerts)[-10:]]  # Last 10 alerts
        }
    
    def get_historical_data(self, component: str, duration_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get historical data for charts"""
        cutoff_time = time.time() - (duration_minutes * 60)
        
        if component == "system":
            data = [asdict(m) for m in self.system_metrics if m.timestamp >= cutoff_time]
        elif component == "network":
            data = [asdict(m) for m in self.network_metrics if m.timestamp >= cutoff_time]
        elif component == "ai":
            data = [asdict(m) for m in self.ai_metrics if m.timestamp >= cutoff_time]
        else:
            data = []
        
        return data

class MockFlaskDashboard:
    """Mock dashboard when Flask is not available"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.is_running = False
    
    async def start_dashboard(self, host: str = "localhost", port: int = 5000):
        """Start mock dashboard with console output"""
        self.is_running = True
        logger.info(f"📊 Mock Dashboard started - Flask not available")
        logger.info(f"📱 Dashboard would be available at: http://{host}:{port}")
        
        while self.is_running:
            await asyncio.sleep(5)
            await self._display_console_dashboard()
    
    async def _display_console_dashboard(self):
        """Display dashboard info in console"""
        metrics = self.metrics_collector.get_latest_metrics()
        
        print("\n" + "="*80)
        print("📊 PRSM REAL-TIME MONITORING DASHBOARD")
        print("="*80)
        
        # System metrics
        if metrics["system"]:
            sys_m = metrics["system"]
            print(f"🖥️  SYSTEM: CPU {sys_m['cpu_percent']:.1f}% | "
                  f"Memory {sys_m['memory_percent']:.1f}% | "
                  f"Disk {sys_m['disk_percent']:.1f}%")
        
        # Network metrics
        if metrics["network"]:
            net_m = metrics["network"]
            print(f"🌐 NETWORK: {net_m['active_nodes']}/{net_m['total_nodes']} nodes | "
                  f"{net_m['total_connections']} connections | "
                  f"{net_m['message_rate']:.1f} msg/s")
        
        # AI metrics
        if metrics["ai"]:
            ai_m = metrics["ai"]
            print(f"🤖 AI: {ai_m['active_models']} models | "
                  f"{ai_m['successful_inferences']} inferences | "
                  f"{ai_m['average_confidence']:.2f} confidence")
        
        # Recent alerts
        if metrics["alerts"]:
            print(f"🚨 ALERTS: {len(metrics['alerts'])} recent alerts")
            for alert in metrics["alerts"][-3:]:
                print(f"   [{alert['level'].upper()}] {alert['component']}: {alert['message']}")
        
        print("="*80)
    
    def stop_dashboard(self):
        """Stop mock dashboard"""
        self.is_running = False

class FlaskDashboard:
    """Real Flask-based dashboard"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            return self._render_dashboard_template()
        
        @self.app.route('/api/metrics')
        def get_metrics():
            return jsonify(self.metrics_collector.get_latest_metrics())
        
        @self.app.route('/api/historical/<component>')
        def get_historical(component):
            duration = request.args.get('duration', 60, type=int)
            data = self.metrics_collector.get_historical_data(component, duration)
            return jsonify(data)
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info("📱 Dashboard client connected")
            emit('status', {'message': 'Connected to PRSM Dashboard'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info("📱 Dashboard client disconnected")
    
    def _render_dashboard_template(self):
        """Render dashboard HTML template"""
        # Simple HTML template for the dashboard
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRSM Real-Time Monitoring Dashboard</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .header { text-align: center; color: #333; margin-bottom: 30px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-title { font-size: 18px; font-weight: bold; color: #333; margin-bottom: 15px; }
        .metric-value { font-size: 24px; font-weight: bold; margin: 10px 0; }
        .metric-value.good { color: #28a745; }
        .metric-value.warning { color: #ffc107; }
        .metric-value.danger { color: #dc3545; }
        .chart-container { height: 200px; margin-top: 15px; }
        .alerts { background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px; padding: 10px; margin: 10px 0; }
        .alert-item { margin: 5px 0; padding: 5px; border-radius: 3px; }
        .alert-warning { background: #fff3cd; }
        .alert-error { background: #f8d7da; }
        .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        .status-active { background: #28a745; }
        .status-inactive { background: #dc3545; }
        .refresh-info { text-align: center; color: #666; margin-top: 20px; font-size: 14px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 PRSM Real-Time Monitoring Dashboard</h1>
        <p>Live system metrics, network status, and AI performance monitoring</p>
    </div>

    <div class="metrics-grid">
        <!-- System Metrics -->
        <div class="metric-card">
            <div class="metric-title">🖥️ System Performance</div>
            <div id="system-metrics">
                <div>CPU Usage: <span id="cpu-usage" class="metric-value">--</span></div>
                <div>Memory Usage: <span id="memory-usage" class="metric-value">--</span></div>
                <div>Active Connections: <span id="connections" class="metric-value">--</span></div>
            </div>
            <div class="chart-container">
                <canvas id="systemChart"></canvas>
            </div>
        </div>

        <!-- Network Metrics -->
        <div class="metric-card">
            <div class="metric-title">🌐 P2P Network Status</div>
            <div id="network-metrics">
                <div><span class="status-indicator status-active"></span>Active Nodes: <span id="active-nodes" class="metric-value">--</span></div>
                <div>Total Connections: <span id="total-connections" class="metric-value">--</span></div>
                <div>Message Rate: <span id="message-rate" class="metric-value">--</span> msg/s</div>
            </div>
            <div class="chart-container">
                <canvas id="networkChart"></canvas>
            </div>
        </div>

        <!-- AI Metrics -->
        <div class="metric-card">
            <div class="metric-title">🤖 AI Model Performance</div>
            <div id="ai-metrics">
                <div>Active Models: <span id="active-models" class="metric-value">--</span></div>
                <div>Successful Inferences: <span id="successful-inferences" class="metric-value">--</span></div>
                <div>Average Confidence: <span id="avg-confidence" class="metric-value">--</span></div>
            </div>
            <div class="chart-container">
                <canvas id="aiChart"></canvas>
            </div>
        </div>

        <!-- Alerts -->
        <div class="metric-card">
            <div class="metric-title">🚨 System Alerts</div>
            <div id="alerts-container">
                <div class="alerts">No alerts at this time</div>
            </div>
        </div>
    </div>

    <div class="refresh-info">
        Dashboard updates every 2 seconds | Last update: <span id="last-update">--</span>
    </div>

    <script>
        // Initialize Socket.IO connection
        const socket = io();
        
        // Initialize charts
        let systemChart, networkChart, aiChart;
        
        function initCharts() {
            const chartConfig = {
                type: 'line',
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { display: false },
                        y: { beginAtZero: true }
                    },
                    plugins: { legend: { display: false } }
                }
            };
            
            systemChart = new Chart(document.getElementById('systemChart'), {
                ...chartConfig,
                data: { labels: [], datasets: [{ label: 'CPU %', data: [], borderColor: 'rgb(75, 192, 192)', tension: 0.1 }] }
            });
            
            networkChart = new Chart(document.getElementById('networkChart'), {
                ...chartConfig,
                data: { labels: [], datasets: [{ label: 'Connections', data: [], borderColor: 'rgb(255, 99, 132)', tension: 0.1 }] }
            });
            
            aiChart = new Chart(document.getElementById('aiChart'), {
                ...chartConfig,
                data: { labels: [], datasets: [{ label: 'Confidence', data: [], borderColor: 'rgb(54, 162, 235)', tension: 0.1 }] }
            });
        }
        
        function updateMetrics() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    updateSystemMetrics(data.system);
                    updateNetworkMetrics(data.network);
                    updateAIMetrics(data.ai);
                    updateAlerts(data.alerts);
                    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                })
                .catch(error => console.error('Error fetching metrics:', error));
        }
        
        function updateSystemMetrics(system) {
            if (!system) return;
            
            const cpuElement = document.getElementById('cpu-usage');
            const memoryElement = document.getElementById('memory-usage');
            const connectionsElement = document.getElementById('connections');
            
            cpuElement.textContent = system.cpu_percent.toFixed(1) + '%';
            cpuElement.className = 'metric-value ' + (system.cpu_percent > 80 ? 'danger' : system.cpu_percent > 60 ? 'warning' : 'good');
            
            memoryElement.textContent = system.memory_percent.toFixed(1) + '%';
            memoryElement.className = 'metric-value ' + (system.memory_percent > 85 ? 'danger' : system.memory_percent > 70 ? 'warning' : 'good');
            
            connectionsElement.textContent = system.active_connections;
            connectionsElement.className = 'metric-value good';
            
            // Update chart
            updateChart(systemChart, system.timestamp, system.cpu_percent);
        }
        
        function updateNetworkMetrics(network) {
            if (!network) return;
            
            document.getElementById('active-nodes').textContent = network.active_nodes + '/' + network.total_nodes;
            document.getElementById('total-connections').textContent = network.total_connections;
            document.getElementById('message-rate').textContent = network.message_rate.toFixed(1);
            
            // Update chart
            updateChart(networkChart, network.timestamp, network.total_connections);
        }
        
        function updateAIMetrics(ai) {
            if (!ai) return;
            
            document.getElementById('active-models').textContent = ai.active_models;
            document.getElementById('successful-inferences').textContent = ai.successful_inferences;
            
            const confidenceElement = document.getElementById('avg-confidence');
            confidenceElement.textContent = (ai.average_confidence * 100).toFixed(1) + '%';
            confidenceElement.className = 'metric-value ' + (ai.average_confidence < 0.7 ? 'warning' : 'good');
            
            // Update chart
            updateChart(aiChart, ai.timestamp, ai.average_confidence * 100);
        }
        
        function updateAlerts(alerts) {
            const container = document.getElementById('alerts-container');
            
            if (!alerts || alerts.length === 0) {
                container.innerHTML = '<div class="alerts">No alerts at this time</div>';
                return;
            }
            
            const alertsHtml = alerts.map(alert => 
                `<div class="alert-item alert-${alert.level}">${alert.component}: ${alert.message}</div>`
            ).join('');
            
            container.innerHTML = alertsHtml;
        }
        
        function updateChart(chart, timestamp, value) {
            const time = new Date(timestamp * 1000).toLocaleTimeString();
            
            chart.data.labels.push(time);
            chart.data.datasets[0].data.push(value);
            
            // Keep only last 20 data points
            if (chart.data.labels.length > 20) {
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
            }
            
            chart.update('none');
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            updateMetrics();
            setInterval(updateMetrics, 2000); // Update every 2 seconds
        });
        
        // Socket.IO event handlers
        socket.on('connect', function() {
            console.log('Connected to PRSM Dashboard');
        });
        
        socket.on('metrics_update', function(data) {
            // Handle real-time updates if needed
            console.log('Real-time metrics update:', data);
        });
    </script>
</body>
</html>
        """
        return html_template
    
    async def start_dashboard(self, host: str = "localhost", port: int = 5000):
        """Start Flask dashboard"""
        logger.info(f"📊 Starting Flask Dashboard on http://{host}:{port}")
        
        # Start background task to emit real-time updates
        asyncio.create_task(self._emit_realtime_updates())
        
        # Run Flask app in thread
        def run_flask():
            self.socketio.run(self.app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)
        
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        
        # Try to open browser
        try:
            webbrowser.open(f"http://{host}:{port}")
        except:
            pass
        
        logger.info(f"📱 Dashboard available at: http://{host}:{port}")
    
    async def _emit_realtime_updates(self):
        """Emit real-time updates to connected clients"""
        while True:
            try:
                metrics = self.metrics_collector.get_latest_metrics()
                self.socketio.emit('metrics_update', metrics)
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"Error emitting updates: {str(e)}")
                await asyncio.sleep(5)

class PRSMMonitoringDashboard:
    """Main dashboard orchestrator"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.dashboard = None
        self.network_demo = None
        self.is_running = False
    
    async def start_monitoring(self, 
                              host: str = "localhost", 
                              port: int = 5000,
                              with_demo: bool = True):
        """Start comprehensive monitoring with dashboard"""
        logger.info("🚀 Starting PRSM Real-Time Monitoring Dashboard...")
        
        try:
            # Start demo network if requested
            if with_demo:
                await self._start_demo_network()
            
            # Choose dashboard implementation
            if FLASK_AVAILABLE:
                self.dashboard = FlaskDashboard(self.metrics_collector)
                logger.info("📊 Using Flask-based interactive dashboard")
            else:
                self.dashboard = MockFlaskDashboard(self.metrics_collector)
                logger.info("📊 Using console-based dashboard (Flask not available)")
            
            # Start metrics collection
            collection_task = asyncio.create_task(
                self.metrics_collector.start_collection(self.network_demo)
            )
            
            # Start dashboard
            dashboard_task = asyncio.create_task(
                self.dashboard.start_dashboard(host, port)
            )
            
            self.is_running = True
            logger.info("✅ PRSM Monitoring Dashboard is running!")
            
            # Keep running
            while self.is_running:
                await asyncio.sleep(1)
            
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested...")
        except Exception as e:
            logger.error(f"❌ Dashboard error: {str(e)}")
        finally:
            await self._cleanup()
    
    async def _start_demo_network(self):
        """Start demo P2P network for monitoring"""
        try:
            logger.info("🌐 Starting demo P2P network for monitoring...")
            self.network_demo = EnhancedP2PNetworkDemo()
            
            # Initialize AI models on nodes
            for node in self.network_demo.nodes:
                await node.ai_manager.initialize_demo_models()
            
            # Run basic network activity
            asyncio.create_task(self._simulate_network_activity())
            
            logger.info("✅ Demo network started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start demo network: {str(e)}")
    
    async def _simulate_network_activity(self):
        """Simulate ongoing network activity for demonstration"""
        while self.is_running:
            try:
                if self.network_demo and self.network_demo.nodes:
                    # Simulate some AI inferences
                    for node in self.network_demo.nodes[:2]:  # Use first 2 nodes
                        if node.ai_manager.models:
                            model_id = list(node.ai_manager.models.keys())[0]
                            test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
                            await node.request_inference(model_id, test_data)
                
                await asyncio.sleep(10)  # Simulate activity every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in network activity simulation: {str(e)}")
                await asyncio.sleep(30)
    
    async def _cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        
        if self.metrics_collector:
            self.metrics_collector.stop_collection()
        
        if self.network_demo:
            await self.network_demo.stop_network()
        
        if hasattr(self.dashboard, 'stop_dashboard'):
            self.dashboard.stop_dashboard()
        
        logger.info("🧹 Cleanup completed")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_running = False

# CLI interface
async def main():
    """Main dashboard application"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PRSM Real-Time Monitoring Dashboard")
    parser.add_argument("--host", default="localhost", help="Dashboard host")
    parser.add_argument("--port", type=int, default=5000, help="Dashboard port")
    parser.add_argument("--no-demo", action="store_true", help="Don't start demo network")
    
    args = parser.parse_args()
    
    dashboard = PRSMMonitoringDashboard()
    
    try:
        await dashboard.start_monitoring(
            host=args.host,
            port=args.port,
            with_demo=not args.no_demo
        )
    except KeyboardInterrupt:
        logger.info("🛑 Dashboard stopped by user")
    finally:
        await dashboard._cleanup()

if __name__ == "__main__":
    print("🚀 PRSM Real-Time Monitoring Dashboard")
    print("=" * 50)
    
    if not FLASK_AVAILABLE:
        print("⚠️  Flask not available - using console dashboard")
        print("   Install Flask for full web dashboard: pip install flask flask-socketio")
    
    asyncio.run(main())