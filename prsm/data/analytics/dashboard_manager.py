#!/usr/bin/env python3
"""
Customizable Dashboard Framework
===============================

Advanced dashboard management system for creating, configuring, and
managing interactive analytics dashboards.
"""

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
import uuid

from .metrics_collector import MetricsCollector, MetricDefinition
from .visualization_engine import VisualizationEngine, ChartType, ChartConfiguration, DataSeries
from .bi_query_engine import BusinessIntelligenceEngine, BIQuery, QueryBuilder

from prsm.compute.plugins import require_optional, has_optional_dependency

logger = logging.getLogger(__name__)


class DashboardType(Enum):
    """Types of dashboards"""
    EXECUTIVE = "executive"           # High-level KPIs and summaries
    OPERATIONAL = "operational"      # Real-time operational metrics
    ANALYTICAL = "analytical"        # Deep dive analytics
    TECHNICAL = "technical"          # System and technical metrics
    CUSTOM = "custom"               # User-defined dashboards


class WidgetType(Enum):
    """Types of dashboard widgets"""
    CHART = "chart"
    TABLE = "table"
    METRIC_CARD = "metric_card"
    TEXT = "text"
    IMAGE = "image"
    IFRAME = "iframe"
    CUSTOM_HTML = "custom_html"
    FILTER_CONTROL = "filter_control"
    DATE_PICKER = "date_picker"


class RefreshMode(Enum):
    """Dashboard refresh modes"""
    MANUAL = "manual"
    AUTO = "auto"
    REAL_TIME = "real_time"
    SCHEDULED = "scheduled"


@dataclass
class WidgetConfiguration:
    """Configuration for a dashboard widget"""
    widget_id: str
    widget_type: WidgetType
    title: str
    position: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0, "width": 4, "height": 3})
    
    # Data configuration
    data_source: Optional[str] = None
    query: Optional[BIQuery] = None
    
    # Visual configuration
    chart_config: Optional[ChartConfiguration] = None
    chart_type: Optional[ChartType] = None
    
    # Behavior configuration
    refresh_interval: int = 300  # seconds
    auto_refresh: bool = True
    interactive: bool = True
    
    # Styling
    background_color: str = "white"
    border_color: str = "#e1e5e9"
    text_color: str = "#333333"
    custom_css: str = ""
    
    # Filters and parameters
    filters: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Custom content (for text, HTML widgets)
    content: str = ""
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert widget configuration to dictionary"""
        return {
            "widget_id": self.widget_id,
            "widget_type": self.widget_type.value,
            "title": self.title,
            "position": self.position,
            "data_source": self.data_source,
            "query": self.query.to_dict() if self.query else None,
            "chart_config": self.chart_config.to_dict() if self.chart_config else None,
            "chart_type": self.chart_type.value if self.chart_type else None,
            "refresh_interval": self.refresh_interval,
            "auto_refresh": self.auto_refresh,
            "interactive": self.interactive,
            "background_color": self.background_color,
            "border_color": self.border_color,
            "text_color": self.text_color,
            "custom_css": self.custom_css,
            "filters": self.filters,
            "parameters": self.parameters,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WidgetConfiguration':
        """Create widget configuration from dictionary"""
        widget_config = cls(
            widget_id=data["widget_id"],
            widget_type=WidgetType(data["widget_type"]),
            title=data["title"],
            position=data.get("position", {"x": 0, "y": 0, "width": 4, "height": 3}),
            data_source=data.get("data_source"),
            refresh_interval=data.get("refresh_interval", 300),
            auto_refresh=data.get("auto_refresh", True),
            interactive=data.get("interactive", True),
            background_color=data.get("background_color", "white"),
            border_color=data.get("border_color", "#e1e5e9"),
            text_color=data.get("text_color", "#333333"),
            custom_css=data.get("custom_css", ""),
            filters=data.get("filters", {}),
            parameters=data.get("parameters", {}),
            content=data.get("content", ""),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now(timezone.utc).isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now(timezone.utc).isoformat()))
        )
        
        # Reconstruct complex objects
        if data.get("query"):
            # This would need a proper BIQuery.from_dict method
            pass
        
        if data.get("chart_config"):
            # This would need a proper ChartConfiguration.from_dict method  
            pass
        
        if data.get("chart_type"):
            widget_config.chart_type = ChartType(data["chart_type"])
        
        return widget_config


@dataclass
class DashboardConfiguration:
    """Configuration for a complete dashboard"""
    dashboard_id: str
    name: str
    dashboard_type: DashboardType
    description: str = ""
    
    # Widgets
    widgets: List[WidgetConfiguration] = field(default_factory=list)
    
    # Layout configuration
    grid_columns: int = 12
    grid_row_height: int = 60
    margin: List[int] = field(default_factory=lambda: [10, 10])
    
    # Theme and styling
    theme: str = "light"  # light, dark, custom
    background_color: str = "#f5f5f5"
    custom_css: str = ""
    
    # Behavior
    refresh_mode: RefreshMode = RefreshMode.AUTO
    refresh_interval: int = 300  # seconds
    auto_layout: bool = False
    
    # Access control
    owner: Optional[str] = None
    shared_with: List[str] = field(default_factory=list)
    public: bool = False
    
    # Global filters
    global_filters: Dict[str, Any] = field(default_factory=dict)
    global_time_range: Optional[Dict[str, datetime]] = None
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dashboard configuration to dictionary"""
        return {
            "dashboard_id": self.dashboard_id,
            "name": self.name,
            "dashboard_type": self.dashboard_type.value,
            "description": self.description,
            "widgets": [w.to_dict() for w in self.widgets],
            "grid_columns": self.grid_columns,
            "grid_row_height": self.grid_row_height,
            "margin": self.margin,
            "theme": self.theme,
            "background_color": self.background_color,
            "custom_css": self.custom_css,
            "refresh_mode": self.refresh_mode.value,
            "refresh_interval": self.refresh_interval,
            "auto_layout": self.auto_layout,
            "owner": self.owner,
            "shared_with": self.shared_with,
            "public": self.public,
            "global_filters": self.global_filters,
            "global_time_range": {
                "start": self.global_time_range["start"].isoformat(),
                "end": self.global_time_range["end"].isoformat()
            } if self.global_time_range else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "access_count": self.access_count
        }


class Dashboard:
    """Runtime dashboard instance"""
    
    def __init__(self, config: DashboardConfiguration, 
                 metrics_collector: MetricsCollector,
                 visualization_engine: VisualizationEngine,
                 bi_engine: BusinessIntelligenceEngine):
        self.config = config
        self.metrics_collector = metrics_collector
        self.visualization_engine = visualization_engine
        self.bi_engine = bi_engine
        
        # Runtime state
        self.widget_data_cache: Dict[str, Any] = {}
        self.last_refresh: Dict[str, datetime] = {}
        self.refresh_tasks: Dict[str, asyncio.Task] = {}
        self.is_running = False
        
        # Statistics
        self.render_stats = {
            "total_renders": 0,
            "widget_renders": defaultdict(int),
            "avg_render_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def start(self):
        """Start the dashboard and begin auto-refresh if configured"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start auto-refresh tasks for widgets
        if self.config.refresh_mode in [RefreshMode.AUTO, RefreshMode.REAL_TIME]:
            for widget in self.config.widgets:
                if widget.auto_refresh:
                    task = asyncio.create_task(self._auto_refresh_widget(widget))
                    self.refresh_tasks[widget.widget_id] = task
        
        logger.info(f"Dashboard {self.config.name} started with {len(self.refresh_tasks)} auto-refresh tasks")
    
    async def stop(self):
        """Stop the dashboard and cancel refresh tasks"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all refresh tasks
        for task in self.refresh_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.refresh_tasks.clear()
        logger.info(f"Dashboard {self.config.name} stopped")
    
    async def render(self, format_type: str = "html") -> Dict[str, Any]:
        """Render the complete dashboard"""
        start_time = datetime.now()
        
        try:
            # Update access statistics
            self.config.last_accessed = datetime.now(timezone.utc)
            self.config.access_count += 1
            
            # Render all widgets
            rendered_widgets = []
            for widget in self.config.widgets:
                try:
                    widget_render = await self._render_widget(widget)
                    rendered_widgets.append(widget_render)
                except Exception as e:
                    logger.error(f"Error rendering widget {widget.widget_id}: {e}")
                    # Add error widget
                    rendered_widgets.append({
                        "widget_id": widget.widget_id,
                        "error": str(e),
                        "widget_type": widget.widget_type.value
                    })
            
            # Generate dashboard HTML
            dashboard_html = self._generate_dashboard_html(rendered_widgets)
            
            # Calculate render time
            render_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update statistics
            self._update_render_stats(render_time)
            
            return {
                "dashboard_id": self.config.dashboard_id,
                "name": self.config.name,
                "html": dashboard_html if format_type == "html" else None,
                "widgets": rendered_widgets,
                "config": self.config.to_dict(),
                "render_time_ms": render_time,
                "rendered_at": datetime.now(timezone.utc).isoformat(),
                "stats": self.render_stats
            }
            
        except Exception as e:
            logger.error(f"Error rendering dashboard {self.config.dashboard_id}: {e}")
            raise
    
    async def _render_widget(self, widget: WidgetConfiguration) -> Dict[str, Any]:
        """Render a single widget"""
        widget_start_time = datetime.now()
        
        # Check cache first
        cache_key = f"{widget.widget_id}_{hash(json.dumps(widget.filters, sort_keys=True))}"
        if cache_key in self.widget_data_cache:
            cached_data = self.widget_data_cache[cache_key]
            cache_age = (datetime.now() - cached_data["cached_at"]).total_seconds()
            if cache_age < widget.refresh_interval:
                self.render_stats["cache_hits"] += 1
                return cached_data["data"]
        
        self.render_stats["cache_misses"] += 1
        
        try:
            if widget.widget_type == WidgetType.CHART:
                result = await self._render_chart_widget(widget)
            elif widget.widget_type == WidgetType.TABLE:
                result = await self._render_table_widget(widget)
            elif widget.widget_type == WidgetType.METRIC_CARD:
                result = await self._render_metric_card_widget(widget)
            elif widget.widget_type == WidgetType.TEXT:
                result = self._render_text_widget(widget)
            elif widget.widget_type == WidgetType.CUSTOM_HTML:
                result = self._render_html_widget(widget)
            else:
                result = self._render_default_widget(widget)
            
            # Add common widget properties
            result.update({
                "widget_id": widget.widget_id,
                "widget_type": widget.widget_type.value,
                "title": widget.title,
                "position": widget.position,
                "render_time_ms": (datetime.now() - widget_start_time).total_seconds() * 1000
            })
            
            # Cache result
            self.widget_data_cache[cache_key] = {
                "data": result,
                "cached_at": datetime.now()
            }
            
            # Update statistics
            self.render_stats["widget_renders"][widget.widget_id] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error rendering widget {widget.widget_id}: {e}")
            return {
                "widget_id": widget.widget_id,
                "widget_type": widget.widget_type.value,
                "error": str(e),
                "position": widget.position
            }
    
    async def _render_chart_widget(self, widget: WidgetConfiguration) -> Dict[str, Any]:
        """Render a chart widget"""
        if not widget.query or not widget.chart_type:
            raise ValueError("Chart widget requires query and chart_type")
        
        # Execute query to get data
        query_result = await self.bi_engine.execute_query(widget.query)
        
        # Convert query result to data series
        if not query_result.data:
            return {"error": "No data available", "content": "<div>No data to display</div>"}
        
        # Create data series from query result
        data_series = []
        
        # Simple approach: use first non-id column as Y data
        data_columns = [col for col in query_result.columns if col.lower() not in ['id', 'timestamp']]
        
        if len(data_columns) >= 1:
            y_column = data_columns[0]
            x_column = 'timestamp' if 'timestamp' in query_result.columns else None
            
            x_data = [row.get(x_column) for row in query_result.data] if x_column else None
            y_data = [row.get(y_column) for row in query_result.data]
            
            series = DataSeries(
                name=y_column,
                data=y_data,
                x_data=x_data,
                chart_type=widget.chart_type
            )
            data_series.append(series)
        
        # Create chart
        chart_config = widget.chart_config or ChartConfiguration()
        chart_config.title = widget.title
        
        chart_result = self.visualization_engine.create_chart(
            widget.chart_type, data_series, chart_config
        )
        
        return {
            "content": chart_result.get("html", ""),
            "chart_data": chart_result,
            "query_result": query_result.to_dict()
        }
    
    async def _render_table_widget(self, widget: WidgetConfiguration) -> Dict[str, Any]:
        """Render a table widget"""
        if not widget.query:
            raise ValueError("Table widget requires query")
        
        # Execute query
        query_result = await self.bi_engine.execute_query(widget.query)
        
        # Generate HTML table
        if not query_result.data:
            html_content = "<div>No data available</div>"
        else:
            html_content = self._generate_html_table(query_result.data, query_result.columns)
        
        return {
            "content": html_content,
            "query_result": query_result.to_dict()
        }
    
    async def _render_metric_card_widget(self, widget: WidgetConfiguration) -> Dict[str, Any]:
        """Render a metric card widget"""
        if not widget.query:
            raise ValueError("Metric card widget requires query")
        
        # Execute query
        query_result = await self.bi_engine.execute_query(widget.query)
        
        if not query_result.data:
            value = "N/A"
        else:
            # Get first numeric value from first row
            first_row = query_result.data[0]
            numeric_values = [v for v in first_row.values() if isinstance(v, (int, float))]
            value = numeric_values[0] if numeric_values else "N/A"
        
        # Generate metric card HTML
        html_content = f"""
        <div class="metric-card" style="text-align: center; padding: 20px;">
            <h3 style="margin: 0; color: {widget.text_color};">{widget.title}</h3>
            <div style="font-size: 2em; font-weight: bold; margin: 10px 0; color: {widget.text_color};">
                {value}
            </div>
        </div>
        """
        
        return {
            "content": html_content,
            "value": value,
            "query_result": query_result.to_dict()
        }
    
    def _render_text_widget(self, widget: WidgetConfiguration) -> Dict[str, Any]:
        """Render a text widget"""
        html_content = f"""
        <div class="text-widget" style="padding: 15px; color: {widget.text_color};">
            <h3 style="margin-top: 0;">{widget.title}</h3>
            <div>{widget.content}</div>
        </div>
        """
        
        return {"content": html_content}
    
    def _render_html_widget(self, widget: WidgetConfiguration) -> Dict[str, Any]:
        """Render a custom HTML widget"""
        return {"content": widget.content}
    
    def _render_default_widget(self, widget: WidgetConfiguration) -> Dict[str, Any]:
        """Render a default placeholder widget"""
        html_content = f"""
        <div class="default-widget" style="padding: 20px; text-align: center; border: 2px dashed #ccc;">
            <h3>{widget.title}</h3>
            <p>Widget type: {widget.widget_type.value}</p>
            <p>This widget type is not yet implemented.</p>
        </div>
        """
        
        return {"content": html_content}
    
    def _generate_html_table(self, data: List[Dict[str, Any]], columns: List[str]) -> str:
        """Generate HTML table from data"""
        if not data:
            return "<div>No data available</div>"
        
        # Generate table HTML
        html_parts = ['<table class="data-table" style="width: 100%; border-collapse: collapse;">']
        
        # Header
        html_parts.append('<thead><tr>')
        for column in columns:
            html_parts.append(f'<th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">{column}</th>')
        html_parts.append('</tr></thead>')
        
        # Body
        html_parts.append('<tbody>')
        for row in data:
            html_parts.append('<tr>')
            for column in columns:
                value = row.get(column, '')
                html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px;">{value}</td>')
            html_parts.append('</tr>')
        html_parts.append('</tbody>')
        
        html_parts.append('</table>')
        
        return '\n'.join(html_parts)
    
    def _generate_dashboard_html(self, rendered_widgets: List[Dict[str, Any]]) -> str:
        """Generate complete dashboard HTML"""
        # Generate CSS
        css = f"""
        <style>
        .dashboard-container {{
            background-color: {self.config.background_color};
            padding: 20px;
            font-family: Arial, sans-serif;
        }}
        .dashboard-header {{
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e1e5e9;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat({self.config.grid_columns}, 1fr);
            gap: {self.config.margin[0]}px;
            min-height: 400px;
        }}
        .widget-container {{
            background-color: white;
            border: 1px solid #e1e5e9;
            border-radius: 6px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .widget-title {{
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }}
        .widget-error {{
            color: #d32f2f;
            background-color: #ffebee;
            padding: 10px;
            border-radius: 4px;
            border: 1px solid #ffcdd2;
        }}
        {self.config.custom_css}
        </style>
        """
        
        # Generate HTML structure
        html_parts = [
            '<!DOCTYPE html>',
            '<html>',
            '<head>',
            '<meta charset="utf-8">',
            f'<title>{self.config.name}</title>',
            css,
            '</head>',
            '<body>',
            '<div class="dashboard-container">',
            '<div class="dashboard-header">',
            f'<h1>{self.config.name}</h1>',
            f'<p>{self.config.description}</p>' if self.config.description else '',
            '</div>',
            '<div class="dashboard-grid">'
        ]
        
        # Add widgets
        for widget in rendered_widgets:
            position = widget.get("position", {"width": 4, "height": 3})
            
            widget_style = f"""
            grid-column: span {position.get('width', 4)};
            grid-row: span {max(1, position.get('height', 3) // 2)};
            """
            
            html_parts.append(f'<div class="widget-container" style="{widget_style}">')
            
            if widget.get("error"):
                html_parts.append(f'<div class="widget-error">Error: {widget["error"]}</div>')
            else:
                html_parts.append(widget.get("content", ""))
            
            html_parts.append('</div>')
        
        html_parts.extend([
            '</div>',  # dashboard-grid
            '</div>',  # dashboard-container
            '</body>',
            '</html>'
        ])
        
        return '\n'.join(html_parts)
    
    async def _auto_refresh_widget(self, widget: WidgetConfiguration):
        """Auto-refresh a widget at specified intervals"""
        while self.is_running:
            try:
                await asyncio.sleep(widget.refresh_interval)
                
                if not self.is_running:
                    break
                
                # Clear cache for this widget to force refresh
                cache_keys_to_remove = [k for k in self.widget_data_cache.keys() if k.startswith(widget.widget_id)]
                for key in cache_keys_to_remove:
                    del self.widget_data_cache[key]
                
                # Update last refresh time
                self.last_refresh[widget.widget_id] = datetime.now()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-refresh for widget {widget.widget_id}: {e}")
    
    def _update_render_stats(self, render_time_ms: float):
        """Update rendering statistics"""
        self.render_stats["total_renders"] += 1
        
        # Update average render time
        total_renders = self.render_stats["total_renders"]
        current_avg = self.render_stats["avg_render_time"]
        self.render_stats["avg_render_time"] = \
            (current_avg * (total_renders - 1) + render_time_ms) / total_renders


class DashboardManager:
    """Main dashboard management system"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./dashboards")
        self.storage_path.mkdir(exist_ok=True)
        
        # Core components (would be injected in real implementation)
        self.metrics_collector = MetricsCollector()
        self.visualization_engine = VisualizationEngine()
        self.bi_engine = BusinessIntelligenceEngine()
        
        # Dashboard storage
        self.dashboards: Dict[str, DashboardConfiguration] = {}
        self.active_dashboards: Dict[str, Dashboard] = {}
        
        # Templates
        self.dashboard_templates: Dict[str, DashboardConfiguration] = {}
        
        # Statistics
        self.manager_stats = {
            "total_dashboards": 0,
            "active_dashboards": 0,
            "total_renders": 0,
            "avg_render_time": 0.0
        }
        
        # Load existing dashboards
        self._load_dashboards()
        
        # Initialize templates
        self._initialize_templates()
        
        logger.info(f"Dashboard Manager initialized with {len(self.dashboards)} dashboards")
    
    def _load_dashboards(self):
        """Load existing dashboards from storage"""
        try:
            dashboard_files = list(self.storage_path.glob("*.json"))
            for dashboard_file in dashboard_files:
                try:
                    with open(dashboard_file, 'r') as f:
                        dashboard_data = json.load(f)
                    
                    # Convert back to DashboardConfiguration
                    # This would need proper from_dict implementation
                    dashboard_id = dashboard_data["dashboard_id"]
                    self.dashboards[dashboard_id] = dashboard_data
                    
                except Exception as e:
                    logger.error(f"Error loading dashboard {dashboard_file}: {e}")
        
        except Exception as e:
            logger.error(f"Error loading dashboards: {e}")
    
    def _initialize_templates(self):
        """Initialize built-in dashboard templates"""
        # Executive Dashboard Template
        executive_template = self._create_executive_template()
        self.dashboard_templates["executive"] = executive_template
        
        # Operational Dashboard Template
        operational_template = self._create_operational_template()
        self.dashboard_templates["operational"] = operational_template
        
        # Technical Dashboard Template
        technical_template = self._create_technical_template()
        self.dashboard_templates["technical"] = technical_template
    
    def _create_executive_template(self) -> DashboardConfiguration:
        """Create executive dashboard template"""
        template_id = f"exec_template_{uuid.uuid4().hex[:8]}"
        
        # Create sample widgets
        widgets = [
            WidgetConfiguration(
                widget_id="kpi_revenue",
                widget_type=WidgetType.METRIC_CARD,
                title="Revenue",
                position={"x": 0, "y": 0, "width": 3, "height": 2}
            ),
            WidgetConfiguration(
                widget_id="kpi_users",
                widget_type=WidgetType.METRIC_CARD,
                title="Active Users",
                position={"x": 3, "y": 0, "width": 3, "height": 2}
            ),
            WidgetConfiguration(
                widget_id="revenue_trend",
                widget_type=WidgetType.CHART,
                title="Revenue Trend",
                chart_type=ChartType.LINE,
                position={"x": 0, "y": 2, "width": 6, "height": 4}
            )
        ]
        
        return DashboardConfiguration(
            dashboard_id=template_id,
            name="Executive Dashboard Template",
            dashboard_type=DashboardType.EXECUTIVE,
            description="High-level KPIs and business metrics",
            widgets=widgets
        )
    
    def _create_operational_template(self) -> DashboardConfiguration:
        """Create operational dashboard template"""
        template_id = f"ops_template_{uuid.uuid4().hex[:8]}"
        
        widgets = [
            WidgetConfiguration(
                widget_id="system_health",
                widget_type=WidgetType.CHART,
                title="System Health",
                chart_type=ChartType.GAUGE,
                position={"x": 0, "y": 0, "width": 4, "height": 3}
            ),
            WidgetConfiguration(
                widget_id="error_rate",
                widget_type=WidgetType.CHART,
                title="Error Rate",
                chart_type=ChartType.LINE,
                position={"x": 4, "y": 0, "width": 8, "height": 3}
            ),
            WidgetConfiguration(
                widget_id="active_requests",
                widget_type=WidgetType.TABLE,
                title="Active Requests",
                position={"x": 0, "y": 3, "width": 12, "height": 4}
            )
        ]
        
        return DashboardConfiguration(
            dashboard_id=template_id,
            name="Operational Dashboard Template",
            dashboard_type=DashboardType.OPERATIONAL,
            description="Real-time operational metrics and monitoring",
            widgets=widgets
        )
    
    def _create_technical_template(self) -> DashboardConfiguration:
        """Create technical dashboard template"""
        template_id = f"tech_template_{uuid.uuid4().hex[:8]}"
        
        widgets = [
            WidgetConfiguration(
                widget_id="cpu_usage",
                widget_type=WidgetType.CHART,
                title="CPU Usage",
                chart_type=ChartType.LINE,
                position={"x": 0, "y": 0, "width": 6, "height": 3}
            ),
            WidgetConfiguration(
                widget_id="memory_usage",
                widget_type=WidgetType.CHART,
                title="Memory Usage",
                chart_type=ChartType.AREA,
                position={"x": 6, "y": 0, "width": 6, "height": 3}
            ),
            WidgetConfiguration(
                widget_id="network_traffic",
                widget_type=WidgetType.CHART,
                title="Network Traffic",
                chart_type=ChartType.LINE,
                position={"x": 0, "y": 3, "width": 12, "height": 4}
            )
        ]
        
        return DashboardConfiguration(
            dashboard_id=template_id,
            name="Technical Dashboard Template",
            dashboard_type=DashboardType.TECHNICAL,
            description="System performance and technical metrics",
            widgets=widgets
        )
    
    def create_dashboard(self, name: str, dashboard_type: DashboardType = DashboardType.CUSTOM,
                        template: Optional[str] = None) -> str:
        """Create a new dashboard"""
        dashboard_id = f"dash_{uuid.uuid4().hex[:8]}"
        
        if template and template in self.dashboard_templates:
            # Create from template
            template_config = self.dashboard_templates[template]
            config = DashboardConfiguration(
                dashboard_id=dashboard_id,
                name=name,
                dashboard_type=dashboard_type,
                description=template_config.description,
                widgets=[w for w in template_config.widgets],  # Copy widgets
                theme=template_config.theme,
                refresh_mode=template_config.refresh_mode
            )
        else:
            # Create empty dashboard
            config = DashboardConfiguration(
                dashboard_id=dashboard_id,
                name=name,
                dashboard_type=dashboard_type
            )
        
        self.dashboards[dashboard_id] = config
        self.manager_stats["total_dashboards"] += 1
        
        # Save to storage
        self._save_dashboard(config)
        
        logger.info(f"Created dashboard: {name} ({dashboard_id})")
        return dashboard_id
    
    def get_dashboard(self, dashboard_id: str) -> Optional[DashboardConfiguration]:
        """Get dashboard configuration"""
        return self.dashboards.get(dashboard_id)
    
    def update_dashboard(self, dashboard_id: str, updates: Dict[str, Any]) -> bool:
        """Update dashboard configuration"""
        if dashboard_id not in self.dashboards:
            return False
        
        config = self.dashboards[dashboard_id]
        
        # Update allowed fields
        for field, value in updates.items():
            if hasattr(config, field):
                setattr(config, field, value)
        
        config.updated_at = datetime.now(timezone.utc)
        
        # Save to storage
        self._save_dashboard(config)
        
        logger.info(f"Updated dashboard: {dashboard_id}")
        return True
    
    def delete_dashboard(self, dashboard_id: str) -> bool:
        """Delete a dashboard"""
        if dashboard_id not in self.dashboards:
            return False
        
        # Stop if running
        if dashboard_id in self.active_dashboards:
            asyncio.create_task(self.stop_dashboard(dashboard_id))
        
        # Remove from storage
        dashboard_file = self.storage_path / f"{dashboard_id}.json"
        if dashboard_file.exists():
            dashboard_file.unlink()
        
        # Remove from memory
        del self.dashboards[dashboard_id]
        self.manager_stats["total_dashboards"] -= 1
        
        logger.info(f"Deleted dashboard: {dashboard_id}")
        return True
    
    async def start_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Start a dashboard instance"""
        if dashboard_id not in self.dashboards:
            return None
        
        if dashboard_id in self.active_dashboards:
            return self.active_dashboards[dashboard_id]
        
        config = self.dashboards[dashboard_id]
        dashboard = Dashboard(config, self.metrics_collector, 
                            self.visualization_engine, self.bi_engine)
        
        await dashboard.start()
        
        self.active_dashboards[dashboard_id] = dashboard
        self.manager_stats["active_dashboards"] += 1
        
        logger.info(f"Started dashboard: {dashboard_id}")
        return dashboard
    
    async def stop_dashboard(self, dashboard_id: str) -> bool:
        """Stop a dashboard instance"""
        if dashboard_id not in self.active_dashboards:
            return False
        
        dashboard = self.active_dashboards[dashboard_id]
        await dashboard.stop()
        
        del self.active_dashboards[dashboard_id]
        self.manager_stats["active_dashboards"] -= 1
        
        logger.info(f"Stopped dashboard: {dashboard_id}")
        return True
    
    async def render_dashboard(self, dashboard_id: str, format_type: str = "html") -> Optional[Dict[str, Any]]:
        """Render a dashboard"""
        # Start dashboard if not running
        dashboard = await self.start_dashboard(dashboard_id)
        if not dashboard:
            return None
        
        # Render dashboard
        result = await dashboard.render(format_type)
        
        # Update manager statistics
        self.manager_stats["total_renders"] += 1
        render_time = result.get("render_time_ms", 0)
        current_avg = self.manager_stats["avg_render_time"]
        total_renders = self.manager_stats["total_renders"]
        self.manager_stats["avg_render_time"] = \
            (current_avg * (total_renders - 1) + render_time) / total_renders
        
        return result
    
    def list_dashboards(self) -> List[Dict[str, Any]]:
        """List all dashboards with metadata"""
        return [
            {
                "dashboard_id": config.dashboard_id,
                "name": config.name,
                "type": config.dashboard_type.value,
                "description": config.description,
                "widget_count": len(config.widgets),
                "created_at": config.created_at.isoformat(),
                "updated_at": config.updated_at.isoformat(),
                "last_accessed": config.last_accessed.isoformat() if config.last_accessed else None,
                "access_count": config.access_count,
                "active": config.dashboard_id in self.active_dashboards
            }
            for config in self.dashboards.values()
        ]
    
    def get_templates(self) -> List[Dict[str, Any]]:
        """Get available dashboard templates"""
        return [
            {
                "template_id": template_id,
                "name": config.name,
                "type": config.dashboard_type.value,
                "description": config.description,
                "widget_count": len(config.widgets)
            }
            for template_id, config in self.dashboard_templates.items()
        ]
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get dashboard manager statistics"""
        return {
            **self.manager_stats,
            "templates_available": len(self.dashboard_templates),
            "storage_path": str(self.storage_path)
        }
    
    def _save_dashboard(self, config: DashboardConfiguration):
        """Save dashboard configuration to storage"""
        try:
            dashboard_file = self.storage_path / f"{config.dashboard_id}.json"
            with open(dashboard_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving dashboard {config.dashboard_id}: {e}")


# Export main classes
__all__ = [
    'DashboardType',
    'WidgetType',
    'RefreshMode',
    'WidgetConfiguration',
    'DashboardConfiguration',
    'Dashboard',
    'DashboardManager'
]