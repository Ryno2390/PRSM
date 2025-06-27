"""
PRSM Monitoring Dashboard
========================

Web-based monitoring dashboard for PRSM metrics and system health.
Provides real-time visualizations and interactive monitoring capabilities.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

try:
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = Request = WebSocket = WebSocketDisconnect = None
    HTMLResponse = JSONResponse = StaticFiles = Jinja2Templates = None
    uvicorn = None

from .metrics import MetricsCollector, MetricValue

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for the monitoring dashboard"""
    host: str = "127.0.0.1"  # Default to localhost for security
    port: int = 3000
    title: str = "PRSM Monitoring Dashboard"
    update_interval: int = 5  # seconds
    max_data_points: int = 1000
    enable_alerts: bool = True
    enable_export: bool = True
    theme: str = "light"  # light, dark


class WebSocketManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Connect a new WebSocket client"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket client"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, data: Dict[str, Any]):
        """Broadcast data to all connected clients"""
        if not self.active_connections:
            return
        
        message = json.dumps(data, default=str)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error sending to WebSocket client: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


class DashboardManager:
    """Main dashboard manager"""
    
    def __init__(self, metrics_collector: MetricsCollector, config: Optional[DashboardConfig] = None):
        self.metrics_collector = metrics_collector
        self.config = config or DashboardConfig()
        self.app: Optional[FastAPI] = None
        self.websocket_manager = WebSocketManager()
        self.is_running = False
        self._update_task: Optional[asyncio.Task] = None
        
        if FASTAPI_AVAILABLE:
            self._setup_app()
        else:
            logger.error("FastAPI not available, dashboard cannot be started")
    
    def _setup_app(self):
        """Setup FastAPI application"""
        self.app = FastAPI(
            title=self.config.title,
            description="PRSM Monitoring Dashboard",
            version="1.0.0"
        )
        
        # Setup templates
        templates_dir = Path(__file__).parent / "templates"
        static_dir = Path(__file__).parent / "static"
        
        # Create directories if they don't exist
        templates_dir.mkdir(exist_ok=True)
        static_dir.mkdir(exist_ok=True)
        
        # Mount static files
        if static_dir.exists():
            self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        
        # Setup Jinja2 templates
        self.templates = Jinja2Templates(directory=str(templates_dir))
        
        # Setup routes
        self._setup_routes()
        
        # Create default template if it doesn't exist
        self._create_default_template()
    
    def _setup_routes(self):
        """Setup dashboard routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Main dashboard page"""
            context = {
                "request": request,
                "title": self.config.title,
                "update_interval": self.config.update_interval * 1000,  # Convert to ms
                "theme": self.config.theme
            }
            return self.templates.TemplateResponse("dashboard.html", context)
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get current metrics"""
            try:
                metrics = await self.metrics_collector.registry.collect_all()
                summary = self.metrics_collector.get_metric_summary()
                
                return JSONResponse({
                    "metrics": [asdict(m) for m in metrics],
                    "summary": summary,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error getting metrics: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)
        
        @self.app.get("/api/metrics/history")
        async def get_metrics_history(minutes: int = 60):
            """Get metrics history"""
            try:
                recent_metrics = self.metrics_collector.get_recent_metrics(minutes)
                
                # Group metrics by name for easier visualization
                grouped_metrics = {}
                for metric in recent_metrics:
                    if metric.name not in grouped_metrics:
                        grouped_metrics[metric.name] = []
                    grouped_metrics[metric.name].append(asdict(metric))
                
                return JSONResponse({
                    "metrics": grouped_metrics,
                    "time_range": minutes,
                    "total_points": len(recent_metrics),
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error getting metrics history: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            return JSONResponse({
                "status": "healthy",
                "dashboard_running": self.is_running,
                "metrics_collecting": self.metrics_collector.is_collecting,
                "websocket_connections": len(self.websocket_manager.active_connections),
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.get("/api/prometheus")
        async def prometheus_metrics():
            """Prometheus metrics endpoint"""
            metrics_text = self.metrics_collector.get_prometheus_metrics()
            return Response(content=metrics_text, media_type="text/plain")
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await self.websocket_manager.connect(websocket)
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket)
    
    def _create_default_template(self):
        """Create default dashboard template"""
        templates_dir = Path(__file__).parent / "templates"
        template_file = templates_dir / "dashboard.html"
        
        if template_file.exists():
            return
        
        # Create default HTML template
        template_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: {% if theme == 'dark' %}#1a1a1a{% else %}#f5f5f5{% endif %};
            color: {% if theme == 'dark' %}#ffffff{% else %}#333333{% endif %};
        }
        
        .header {
            background: {% if theme == 'dark' %}#2d2d2d{% else %}#ffffff{% endif %};
            padding: 1rem 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-bottom: 1px solid {% if theme == 'dark' %}#404040{% else %}#e0e0e0{% endif %};
        }
        
        .header h1 {
            color: #4CAF50;
            font-size: 1.8rem;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-left: 10px;
        }
        
        .status-healthy { background-color: #4CAF50; }
        .status-warning { background-color: #FF9800; }
        .status-error { background-color: #f44336; }
        
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .card {
            background: {% if theme == 'dark' %}#2d2d2d{% else %}#ffffff{% endif %};
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border: 1px solid {% if theme == 'dark' %}#404040{% else %}#e0e0e0{% endif %};
        }
        
        .card h3 {
            margin-bottom: 1rem;
            color: #4CAF50;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 0.5rem;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #4CAF50;
            margin: 0.5rem 0;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: {% if theme == 'dark' %}#cccccc{% else %}#666666{% endif %};
            margin-bottom: 0.5rem;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 1rem;
        }
        
        .metrics-list {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .metric-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid {% if theme == 'dark' %}#404040{% else %}#f0f0f0{% endif %};
        }
        
        .metric-item:last-child {
            border-bottom: none;
        }
        
        .loading {
            text-align: center;
            color: {% if theme == 'dark' %}#cccccc{% else %}#666666{% endif %};
            font-style: italic;
        }
        
        .error {
            color: #f44336;
            text-align: center;
            padding: 1rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }} <span id="status-indicator" class="status-indicator status-healthy"></span></h1>
        <p>Real-time monitoring dashboard - Last updated: <span id="last-updated">Loading...</span></p>
    </div>
    
    <div class="dashboard">
        <!-- System Overview -->
        <div class="card">
            <h3>System Overview</h3>
            <div class="metric-item">
                <span class="metric-label">Total Queries</span>
                <span id="total-queries" class="metric-value">-</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Active Sessions</span>
                <span id="active-sessions" class="metric-value">-</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Total FTNS Used</span>
                <span id="total-ftns" class="metric-value">-</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Error Rate</span>
                <span id="error-rate" class="metric-value">-</span>
            </div>
        </div>
        
        <!-- Query Performance -->
        <div class="card">
            <h3>Query Performance</h3>
            <div class="chart-container">
                <canvas id="performance-chart"></canvas>
            </div>
        </div>
        
        <!-- FTNS Usage -->
        <div class="card">
            <h3>FTNS Usage</h3>
            <div class="chart-container">
                <canvas id="ftns-chart"></canvas>
            </div>
        </div>
        
        <!-- Recent Metrics -->
        <div class="card">
            <h3>Recent Metrics</h3>
            <div id="recent-metrics" class="metrics-list loading">
                Loading metrics...
            </div>
        </div>
        
        <!-- System Health -->
        <div class="card">
            <h3>System Health</h3>
            <div id="health-status" class="loading">
                Checking health...
            </div>
        </div>
        
        <!-- Alerts -->
        <div class="card">
            <h3>Active Alerts</h3>
            <div id="alerts-list">
                <p style="color: #4CAF50;">No active alerts</p>
            </div>
        </div>
    </div>
    
    <script>
        // WebSocket connection for real-time updates
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        // Chart instances
        let performanceChart, ftnsChart;
        
        // Initialize charts
        function initCharts() {
            const chartOptions = {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                },
                plugins: {
                    legend: {
                        display: true
                    }
                }
            };
            
            // Performance chart
            const perfCtx = document.getElementById('performance-chart').getContext('2d');
            performanceChart = new Chart(perfCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Queries per minute',
                        data: [],
                        borderColor: '#4CAF50',
                        backgroundColor: 'rgba(76, 175, 80, 0.1)',
                        tension: 0.4
                    }]
                },
                options: chartOptions
            });
            
            // FTNS chart
            const ftnsCtx = document.getElementById('ftns-chart').getContext('2d');
            ftnsChart = new Chart(ftnsCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'FTNS per minute',
                        data: [],
                        borderColor: '#2196F3',
                        backgroundColor: 'rgba(33, 150, 243, 0.1)',
                        tension: 0.4
                    }]
                },
                options: chartOptions
            });
        }
        
        // Update dashboard with new data
        function updateDashboard(data) {
            document.getElementById('last-updated').textContent = new Date().toLocaleTimeString();
            
            if (data.summary) {
                const summary = data.summary;
                
                // Update system overview
                document.getElementById('total-queries').textContent = summary.total_metrics_collected || '-';
                document.getElementById('active-sessions').textContent = summary.recent_metrics_count || '-';
                
                // Update status indicator
                const statusIndicator = document.getElementById('status-indicator');
                statusIndicator.className = 'status-indicator ' + 
                    (summary.collection_active ? 'status-healthy' : 'status-error');
            }
            
            if (data.metrics) {
                updateRecentMetrics(data.metrics);
            }
        }
        
        // Update recent metrics list
        function updateRecentMetrics(metrics) {
            const container = document.getElementById('recent-metrics');
            container.innerHTML = '';
            
            if (metrics.length === 0) {
                container.innerHTML = '<p class="loading">No recent metrics</p>';
                return;
            }
            
            metrics.slice(0, 10).forEach(metric => {
                const item = document.createElement('div');
                item.className = 'metric-item';
                item.innerHTML = `
                    <span class="metric-label">${metric.name}</span>
                    <span class="metric-value">${metric.value}</span>
                `;
                container.appendChild(item);
            });
        }
        
        // Fetch and update metrics
        async function fetchMetrics() {
            try {
                const response = await fetch('/api/metrics');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Error fetching metrics:', error);
                document.getElementById('status-indicator').className = 'status-indicator status-error';
            }
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            fetchMetrics();
            
            // Set up periodic updates
            setInterval(fetchMetrics, {{ update_interval }});
        });
        
        // WebSocket event handlers
        ws.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };
        
        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
            document.getElementById('status-indicator').className = 'status-indicator status-warning';
        };
        
        ws.onclose = function() {
            console.log('WebSocket connection closed');
            document.getElementById('status-indicator').className = 'status-indicator status-warning';
        };
    </script>
</body>
</html>
        '''
        
        with open(template_file, 'w') as f:
            f.write(template_content)
        
        logger.info(f"Created default dashboard template: {template_file}")
    
    async def start_server(self) -> None:
        """Start the dashboard server"""
        if not FASTAPI_AVAILABLE:
            logger.error("FastAPI not available, cannot start dashboard server")
            return
        
        if self.is_running:
            logger.warning("Dashboard server already running")
            return
        
        self.is_running = True
        
        # Start real-time update task
        self._update_task = asyncio.create_task(self._real_time_updates())
        
        # Start the server
        logger.info(f"Starting dashboard server on {self.config.host}:{self.config.port}")
        
        config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def stop_server(self) -> None:
        """Stop the dashboard server"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped dashboard server")
    
    async def _real_time_updates(self) -> None:
        """Send real-time updates to connected WebSocket clients"""
        while self.is_running:
            try:
                if self.websocket_manager.active_connections:
                    # Get current metrics
                    metrics = await self.metrics_collector.registry.collect_all()
                    summary = self.metrics_collector.get_metric_summary()
                    
                    # Send update to all connected clients
                    await self.websocket_manager.broadcast({
                        "type": "metrics_update",
                        "metrics": [asdict(m) for m in metrics],
                        "summary": summary,
                        "timestamp": datetime.now().isoformat()
                    })
                
                await asyncio.sleep(self.config.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in real-time updates: {e}")
                await asyncio.sleep(self.config.update_interval)
    
    def get_dashboard_url(self) -> str:
        """Get the dashboard URL"""
        return f"http://{self.config.host}:{self.config.port}"
