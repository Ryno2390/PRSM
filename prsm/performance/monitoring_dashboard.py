"""
PRSM Real-time Monitoring Dashboard
Advanced real-time monitoring dashboard with WebSocket streaming, interactive visualizations, and alerting
"""

from typing import Dict, Any, List, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import json
import time
import logging
from collections import defaultdict, deque
import redis.asyncio as aioredis
from contextlib import asynccontextmanager

# FastAPI and WebSocket imports
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)


class DashboardMetricType(Enum):
    """Types of dashboard metrics"""
    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    TIMELINE = "timeline"
    TABLE = "table"
    ALERT = "alert"


class DashboardUpdateType(Enum):
    """Types of dashboard updates"""
    METRIC_UPDATE = "metric_update"
    ALERT_UPDATE = "alert_update"
    SYSTEM_STATUS = "system_status"
    LOG_ENTRY = "log_entry"
    TRACE_UPDATE = "trace_update"


@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str
    title: str
    metric_type: DashboardMetricType
    data_source: str
    refresh_interval: int = 30  # seconds
    chart_config: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0, "w": 4, "h": 3})
    filters: Dict[str, Any] = field(default_factory=dict)
    alerts_enabled: bool = False
    alert_thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class DashboardLayout:
    """Dashboard layout configuration"""
    layout_id: str
    name: str
    description: str
    widgets: List[DashboardWidget] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DashboardUpdate:
    """Real-time dashboard update"""
    update_type: DashboardUpdateType
    widget_id: Optional[str]
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class WebSocketConnectionManager:
    """Manager for WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self.widget_subscribers: Dict[str, Set[str]] = defaultdict(set)
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        logger.info(f"WebSocket connection established: {connection_id}")
    
    def disconnect(self, connection_id: str):
        """Remove WebSocket connection"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            
            # Clean up subscriptions
            if connection_id in self.connection_subscriptions:
                for widget_id in self.connection_subscriptions[connection_id]:
                    self.widget_subscribers[widget_id].discard(connection_id)
                del self.connection_subscriptions[connection_id]
            
            logger.info(f"WebSocket connection closed: {connection_id}")
    
    def subscribe_to_widget(self, connection_id: str, widget_id: str):
        """Subscribe connection to widget updates"""
        self.connection_subscriptions[connection_id].add(widget_id)
        self.widget_subscribers[widget_id].add(connection_id)
    
    def unsubscribe_from_widget(self, connection_id: str, widget_id: str):
        """Unsubscribe connection from widget updates"""
        self.connection_subscriptions[connection_id].discard(widget_id)
        self.widget_subscribers[widget_id].discard(connection_id)
    
    async def send_to_connection(self, connection_id: str, data: Dict[str, Any]):
        """Send data to specific connection"""
        if connection_id in self.active_connections:
            try:
                await self.active_connections[connection_id].send_text(json.dumps(data))
            except Exception as e:
                logger.error(f"Error sending to connection {connection_id}: {e}")
                self.disconnect(connection_id)
    
    async def send_to_widget_subscribers(self, widget_id: str, data: Dict[str, Any]):
        """Send data to all subscribers of a widget"""
        subscribers = self.widget_subscribers.get(widget_id, set()).copy()
        
        for connection_id in subscribers:
            await self.send_to_connection(connection_id, data)
    
    async def broadcast(self, data: Dict[str, Any]):
        """Broadcast data to all connections"""
        disconnected = []
        
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(data))
            except Exception as e:
                logger.error(f"Error broadcasting to {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            self.disconnect(connection_id)
    
    @property
    def connection_count(self) -> int:
        return len(self.active_connections)


class MetricDataProvider:
    """Provider for dashboard metric data"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.cache = {}
        self.cache_ttl = {}
        self.cache_timeout = 10  # seconds
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            # Get from metrics system
            metrics_data = await self.redis.get("system_metrics_current")
            if metrics_data:
                return json.loads(metrics_data)
            
            # Fallback to basic system info
            return {
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "disk_usage": 0.0,
                "network_io": {"bytes_sent": 0, "bytes_recv": 0},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
    
    async def get_application_metrics(self) -> Dict[str, Any]:
        """Get application performance metrics"""
        try:
            # Get HTTP request metrics
            http_metrics = await self.redis.hgetall("http_metrics")
            
            # Get database metrics
            db_metrics = await self.redis.hgetall("database_metrics")
            
            # Get cache metrics
            cache_metrics = await self.redis.hgetall("cache_metrics")
            
            # Get task queue metrics
            task_metrics = await self.redis.hgetall("task_metrics")
            
            return {
                "http": {k.decode(): json.loads(v) for k, v in http_metrics.items()} if http_metrics else {},
                "database": {k.decode(): json.loads(v) for k, v in db_metrics.items()} if db_metrics else {},
                "cache": {k.decode(): json.loads(v) for k, v in cache_metrics.items()} if cache_metrics else {},
                "tasks": {k.decode(): json.loads(v) for k, v in task_metrics.items()} if task_metrics else {},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error getting application metrics: {e}")
            return {}
    
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active system alerts"""
        try:
            alerts = []
            
            # Get database alerts
            db_alerts = await self.redis.lrange("database_alerts", 0, 49)  # Last 50
            for alert_data in db_alerts:
                try:
                    alert = json.loads(alert_data)
                    if not alert.get("resolved", False):
                        alerts.append({
                            **alert,
                            "source": "database"
                        })
                except json.JSONDecodeError:
                    continue
            
            # Get task alerts
            task_alerts = await self.redis.lrange("task_alerts", 0, 49)  # Last 50
            for alert_data in task_alerts:
                try:
                    alert = json.loads(alert_data)
                    if not alert.get("resolved", False):
                        alerts.append({
                            **alert,
                            "source": "tasks"
                        })
                except json.JSONDecodeError:
                    continue
            
            # Sort by timestamp (newest first)
            alerts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return alerts[:20]  # Return top 20 active alerts
        
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
    
    async def get_recent_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent log entries"""
        try:
            # Get from structured logging system
            log_keys = await self.redis.zrevrange("logs:timeline", 0, limit - 1)
            
            logs = []
            for key in log_keys:
                log_data = await self.redis.get(key.decode())
                if log_data:
                    logs.append(json.loads(log_data))
            
            return logs
        
        except Exception as e:
            logger.error(f"Error getting recent logs: {e}")
            return []
    
    async def get_trace_analytics(self, hours: int = 1) -> Dict[str, Any]:
        """Get trace analytics summary"""
        try:
            # This would integrate with the tracing system
            # For now, return basic structure
            return {
                "total_traces": 0,
                "avg_duration_ms": 0,
                "error_rate": 0,
                "throughput_per_minute": 0,
                "services": {},
                "operations": {},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error getting trace analytics: {e}")
            return {}
    
    async def get_queue_statistics(self) -> Dict[str, Any]:
        """Get task queue statistics"""
        try:
            queue_stats = {}
            
            # Get stats for each queue
            queue_names = ["default", "high_priority", "low_priority", "background"]
            
            for queue_name in queue_names:
                pending_count = await self.redis.llen(f"queue:{queue_name}:pending")
                processing_count = await self.redis.llen(f"queue:{queue_name}:processing")
                
                queue_stats[queue_name] = {
                    "pending": pending_count,
                    "processing": processing_count,
                    "total": pending_count + processing_count
                }
            
            return queue_stats
        
        except Exception as e:
            logger.error(f"Error getting queue statistics: {e}")
            return {}


class MonitoringDashboard:
    """Real-time monitoring dashboard system"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.connection_manager = WebSocketConnectionManager()
        self.data_provider = MetricDataProvider(redis_client)
        
        # Dashboard configurations
        self.layouts: Dict[str, DashboardLayout] = {}
        self.widgets: Dict[str, DashboardWidget] = {}
        
        # Real-time updates
        self.update_tasks: Dict[str, asyncio.Task] = {}
        self.update_intervals: Dict[str, int] = {}
        
        # Statistics
        self.stats = {
            "connections": 0,
            "updates_sent": 0,
            "widgets_active": 0,
            "layouts_configured": 0
        }
        
        # Create default layout
        self._create_default_layout()
    
    def _create_default_layout(self):
        """Create default dashboard layout"""
        
        default_widgets = [
            DashboardWidget(
                widget_id="system_overview",
                title="System Overview",
                metric_type=DashboardMetricType.GAUGE,
                data_source="system_metrics",
                position={"x": 0, "y": 0, "w": 6, "h": 4},
                refresh_interval=10
            ),
            DashboardWidget(
                widget_id="active_alerts",
                title="Active Alerts",
                metric_type=DashboardMetricType.TABLE,
                data_source="alerts",
                position={"x": 6, "y": 0, "w": 6, "h": 4},
                refresh_interval=15
            ),
            DashboardWidget(
                widget_id="http_requests",
                title="HTTP Requests",
                metric_type=DashboardMetricType.TIMELINE,
                data_source="http_metrics",
                position={"x": 0, "y": 4, "w": 8, "h": 4},
                refresh_interval=20
            ),
            DashboardWidget(
                widget_id="database_performance",
                title="Database Performance",
                metric_type=DashboardMetricType.HISTOGRAM,
                data_source="database_metrics",
                position={"x": 8, "y": 4, "w": 4, "h": 4},
                refresh_interval=30
            ),
            DashboardWidget(
                widget_id="task_queues",
                title="Task Queues",
                metric_type=DashboardMetricType.GAUGE,
                data_source="queue_stats",
                position={"x": 0, "y": 8, "w": 6, "h": 3},
                refresh_interval=20
            ),
            DashboardWidget(
                widget_id="recent_logs",
                title="Recent Logs",
                metric_type=DashboardMetricType.TABLE,
                data_source="logs",
                position={"x": 6, "y": 8, "w": 6, "h": 3},
                refresh_interval=10
            )
        ]
        
        default_layout = DashboardLayout(
            layout_id="default",
            name="System Overview",
            description="Default system monitoring dashboard",
            widgets=default_widgets
        )
        
        self.layouts["default"] = default_layout
        
        # Register widgets
        for widget in default_widgets:
            self.widgets[widget.widget_id] = widget
        
        self.stats["layouts_configured"] = 1
        self.stats["widgets_active"] = len(default_widgets)
    
    async def start_monitoring(self):
        """Start real-time monitoring updates"""
        logger.info("Starting dashboard monitoring...")
        
        # Start update tasks for each widget
        for widget_id, widget in self.widgets.items():
            if widget_id not in self.update_tasks:
                task = asyncio.create_task(self._widget_update_loop(widget))
                self.update_tasks[widget_id] = task
        
        logger.info(f"✅ Dashboard monitoring started with {len(self.update_tasks)} widgets")
    
    async def stop_monitoring(self):
        """Stop monitoring updates"""
        logger.info("Stopping dashboard monitoring...")
        
        # Cancel all update tasks
        for task in self.update_tasks.values():
            if not task.done():
                task.cancel()
        
        if self.update_tasks:
            await asyncio.gather(*self.update_tasks.values(), return_exceptions=True)
        
        self.update_tasks.clear()
        logger.info("✅ Dashboard monitoring stopped")
    
    async def _widget_update_loop(self, widget: DashboardWidget):
        """Update loop for individual widget"""
        while True:
            try:
                # Get widget data
                data = await self._get_widget_data(widget)
                
                if data:
                    # Create update
                    update = DashboardUpdate(
                        update_type=DashboardUpdateType.METRIC_UPDATE,
                        widget_id=widget.widget_id,
                        data={
                            "widget_id": widget.widget_id,
                            "metric_type": widget.metric_type.value,
                            "data": data,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    )
                    
                    # Send to subscribers
                    await self.connection_manager.send_to_widget_subscribers(
                        widget.widget_id,
                        update.data
                    )
                    
                    self.stats["updates_sent"] += 1
                
                # Wait for next update
                await asyncio.sleep(widget.refresh_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in widget update loop for {widget.widget_id}: {e}")
                await asyncio.sleep(widget.refresh_interval)
    
    async def _get_widget_data(self, widget: DashboardWidget) -> Optional[Dict[str, Any]]:
        """Get data for specific widget"""
        
        try:
            if widget.data_source == "system_metrics":
                return await self.data_provider.get_system_metrics()
            
            elif widget.data_source == "alerts":
                alerts = await self.data_provider.get_active_alerts()
                return {"alerts": alerts, "count": len(alerts)}
            
            elif widget.data_source == "http_metrics":
                app_metrics = await self.data_provider.get_application_metrics()
                return app_metrics.get("http", {})
            
            elif widget.data_source == "database_metrics":
                app_metrics = await self.data_provider.get_application_metrics()
                return app_metrics.get("database", {})
            
            elif widget.data_source == "queue_stats":
                return await self.data_provider.get_queue_statistics()
            
            elif widget.data_source == "logs":
                logs = await self.data_provider.get_recent_logs(50)
                return {"logs": logs[-20:], "total": len(logs)}  # Last 20 logs
            
            elif widget.data_source == "trace_analytics":
                return await self.data_provider.get_trace_analytics()
            
            else:
                logger.warning(f"Unknown data source: {widget.data_source}")
                return None
        
        except Exception as e:
            logger.error(f"Error getting data for widget {widget.widget_id}: {e}")
            return None
    
    # WebSocket handlers
    
    async def handle_websocket_connection(self, websocket: WebSocket, connection_id: str):
        """Handle new WebSocket connection"""
        await self.connection_manager.connect(websocket, connection_id)
        self.stats["connections"] = self.connection_manager.connection_count
        
        try:
            # Send initial dashboard configuration
            await self._send_dashboard_config(connection_id)
            
            # Handle incoming messages
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                await self._handle_websocket_message(connection_id, message)
        
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"WebSocket error for {connection_id}: {e}")
        finally:
            self.connection_manager.disconnect(connection_id)
            self.stats["connections"] = self.connection_manager.connection_count
    
    async def _send_dashboard_config(self, connection_id: str):
        """Send dashboard configuration to connection"""
        
        # Send default layout
        default_layout = self.layouts.get("default")
        if default_layout:
            config_data = {
                "type": "dashboard_config",
                "layout": {
                    "layout_id": default_layout.layout_id,
                    "name": default_layout.name,
                    "description": default_layout.description,
                    "widgets": [
                        {
                            "widget_id": widget.widget_id,
                            "title": widget.title,
                            "metric_type": widget.metric_type.value,
                            "data_source": widget.data_source,
                            "position": widget.position,
                            "chart_config": widget.chart_config,
                            "refresh_interval": widget.refresh_interval
                        }
                        for widget in default_layout.widgets
                    ]
                }
            }
            
            await self.connection_manager.send_to_connection(connection_id, config_data)
    
    async def _handle_websocket_message(self, connection_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        
        message_type = message.get("type")
        
        if message_type == "subscribe_widget":
            widget_id = message.get("widget_id")
            if widget_id and widget_id in self.widgets:
                self.connection_manager.subscribe_to_widget(connection_id, widget_id)
                
                # Send current widget data
                widget = self.widgets[widget_id]
                data = await self._get_widget_data(widget)
                if data:
                    await self.connection_manager.send_to_connection(connection_id, {
                        "type": "widget_data",
                        "widget_id": widget_id,
                        "data": data,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
        
        elif message_type == "unsubscribe_widget":
            widget_id = message.get("widget_id")
            if widget_id:
                self.connection_manager.unsubscribe_from_widget(connection_id, widget_id)
        
        elif message_type == "get_widget_data":
            widget_id = message.get("widget_id")
            if widget_id and widget_id in self.widgets:
                widget = self.widgets[widget_id]
                data = await self._get_widget_data(widget)
                if data:
                    await self.connection_manager.send_to_connection(connection_id, {
                        "type": "widget_data",
                        "widget_id": widget_id,
                        "data": data,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
        
        else:
            logger.warning(f"Unknown message type: {message_type}")
    
    # Dashboard management
    
    def add_widget(self, widget: DashboardWidget) -> bool:
        """Add widget to dashboard"""
        try:
            self.widgets[widget.widget_id] = widget
            
            # Start update task
            if widget.widget_id not in self.update_tasks:
                task = asyncio.create_task(self._widget_update_loop(widget))
                self.update_tasks[widget.widget_id] = task
            
            self.stats["widgets_active"] = len(self.widgets)
            return True
        
        except Exception as e:
            logger.error(f"Error adding widget {widget.widget_id}: {e}")
            return False
    
    def remove_widget(self, widget_id: str) -> bool:
        """Remove widget from dashboard"""
        try:
            if widget_id in self.widgets:
                del self.widgets[widget_id]
            
            # Cancel update task
            if widget_id in self.update_tasks:
                self.update_tasks[widget_id].cancel()
                del self.update_tasks[widget_id]
            
            self.stats["widgets_active"] = len(self.widgets)
            return True
        
        except Exception as e:
            logger.error(f"Error removing widget {widget_id}: {e}")
            return False
    
    def get_layout(self, layout_id: str = "default") -> Optional[DashboardLayout]:
        """Get dashboard layout"""
        return self.layouts.get(layout_id)
    
    def save_layout(self, layout: DashboardLayout) -> bool:
        """Save dashboard layout"""
        try:
            layout.updated_at = datetime.now(timezone.utc)
            self.layouts[layout.layout_id] = layout
            
            # Update widgets
            for widget in layout.widgets:
                self.widgets[widget.widget_id] = widget
            
            self.stats["layouts_configured"] = len(self.layouts)
            self.stats["widgets_active"] = len(self.widgets)
            return True
        
        except Exception as e:
            logger.error(f"Error saving layout {layout.layout_id}: {e}")
            return False
    
    async def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics"""
        return {
            "dashboard_active": True,
            "connections": self.connection_manager.connection_count,
            "widgets": len(self.widgets),
            "layouts": len(self.layouts),
            "update_tasks": len(self.update_tasks),
            "statistics": self.stats.copy(),
            "system_overview": await self.data_provider.get_system_metrics()
        }


# FastAPI integration (if available)
if FASTAPI_AVAILABLE:
    
    def create_dashboard_app(redis_client: aioredis.Redis) -> FastAPI:
        """Create FastAPI app for monitoring dashboard"""
        
        app = FastAPI(title="PRSM Monitoring Dashboard", version="1.0.0")
        dashboard = MonitoringDashboard(redis_client)
        
        # Static files and templates
        # app.mount("/static", StaticFiles(directory="static"), name="static")
        # templates = Jinja2Templates(directory="templates")
        
        @app.on_event("startup")
        async def startup_event():
            await dashboard.start_monitoring()
        
        @app.on_event("shutdown")
        async def shutdown_event():
            await dashboard.stop_monitoring()
        
        @app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Serve dashboard home page"""
            # In production, this would serve the actual dashboard HTML
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>PRSM Monitoring Dashboard</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .widget { border: 1px solid #ddd; margin: 10px; padding: 15px; border-radius: 5px; }
                    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                    .alert { color: red; font-weight: bold; }
                    .status { color: green; }
                </style>
            </head>
            <body>
                <h1>PRSM Monitoring Dashboard</h1>
                <div id="dashboard">
                    <div class="widget">
                        <h3>System Status</h3>
                        <div id="system-status">Loading...</div>
                    </div>
                    <div class="widget">
                        <h3>Active Alerts</h3>
                        <div id="alerts">Loading...</div>
                    </div>
                    <div class="widget">
                        <h3>Performance Metrics</h3>
                        <div id="metrics">Loading...</div>
                    </div>
                </div>
                
                <script>
                    // WebSocket connection for real-time updates
                    const ws = new WebSocket('ws://localhost:8000/ws/dashboard');
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        console.log('Received:', data);
                        
                        // Update dashboard based on data type
                        if (data.type === 'widget_data') {
                            updateWidget(data.widget_id, data.data);
                        }
                    };
                    
                    function updateWidget(widgetId, data) {
                        // Update widget display with new data
                        console.log('Updating widget:', widgetId, data);
                    }
                    
                    // Subscribe to all widgets
                    ws.onopen = function() {
                        ws.send(JSON.stringify({type: 'subscribe_widget', widget_id: 'system_overview'}));
                        ws.send(JSON.stringify({type: 'subscribe_widget', widget_id: 'active_alerts'}));
                        ws.send(JSON.stringify({type: 'subscribe_widget', widget_id: 'http_requests'}));
                    };
                </script>
            </body>
            </html>
            """)
        
        @app.websocket("/ws/dashboard")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time dashboard updates"""
            import uuid
            connection_id = str(uuid.uuid4())
            await dashboard.handle_websocket_connection(websocket, connection_id)
        
        @app.get("/api/dashboard/stats")
        async def get_dashboard_stats():
            """Get dashboard statistics"""
            stats = await dashboard.get_dashboard_stats()
            return JSONResponse(stats)
        
        @app.get("/api/widgets/{widget_id}/data")
        async def get_widget_data(widget_id: str):
            """Get data for specific widget"""
            if widget_id in dashboard.widgets:
                widget = dashboard.widgets[widget_id]
                data = await dashboard._get_widget_data(widget)
                return JSONResponse(data or {})
            else:
                return JSONResponse({"error": "Widget not found"}, status_code=404)
        
        @app.get("/api/alerts")
        async def get_active_alerts():
            """Get active alerts"""
            alerts = await dashboard.data_provider.get_active_alerts()
            return JSONResponse({"alerts": alerts, "count": len(alerts)})
        
        @app.get("/api/logs")
        async def get_recent_logs(limit: int = 100):
            """Get recent log entries"""
            logs = await dashboard.data_provider.get_recent_logs(limit)
            return JSONResponse({"logs": logs, "count": len(logs)})
        
        @app.get("/prometheus")
        async def prometheus_metrics():
            """Prometheus metrics endpoint"""
            try:
                from .metrics import get_metrics_collector
                collector = get_metrics_collector()
                metrics_text = collector.get_prometheus_metrics()
                return Response(content=metrics_text, media_type="text/plain")
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)
        
        return app


# Global dashboard instance
dashboard_instance: Optional[MonitoringDashboard] = None


def initialize_dashboard(redis_client: aioredis.Redis) -> MonitoringDashboard:
    """Initialize monitoring dashboard"""
    global dashboard_instance
    
    dashboard_instance = MonitoringDashboard(redis_client)
    logger.info("✅ Monitoring dashboard initialized")
    return dashboard_instance


def get_dashboard() -> MonitoringDashboard:
    """Get global dashboard instance"""
    if dashboard_instance is None:
        raise RuntimeError("Dashboard not initialized")
    return dashboard_instance


async def start_dashboard_monitoring():
    """Start dashboard monitoring"""
    if dashboard_instance:
        await dashboard_instance.start_monitoring()


async def stop_dashboard_monitoring():
    """Stop dashboard monitoring"""
    if dashboard_instance:
        await dashboard_instance.stop_monitoring()