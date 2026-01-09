#!/usr/bin/env python3
"""
Interactive Visualization Engine
===============================

Advanced visualization system for creating interactive charts, graphs,
and data visualizations for analytics dashboards.
"""

import logging
import json
import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import io

from prsm.compute.plugins import require_optional, has_optional_dependency

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Supported chart types"""
    LINE = "line"
    BAR = "bar" 
    SCATTER = "scatter"
    PIE = "pie"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BOX_PLOT = "box_plot"
    AREA = "area"
    GAUGE = "gauge"
    TABLE = "table"
    TREEMAP = "treemap"
    SANKEY = "sankey"
    RADAR = "radar"
    CANDLESTICK = "candlestick"
    WATERFALL = "waterfall"


class RenderFormat(Enum):
    """Output formats for visualizations"""
    HTML = "html"
    PNG = "png"
    SVG = "svg"
    PDF = "pdf"
    JSON = "json"
    INTERACTIVE = "interactive"


@dataclass
class ChartConfiguration:
    """Configuration for chart appearance and behavior"""
    title: str = ""
    subtitle: str = ""
    width: int = 800
    height: int = 600
    
    # Colors and styling
    color_scheme: str = "default"
    background_color: str = "white"
    grid_enabled: bool = True
    
    # Axes configuration
    x_axis_title: str = ""
    y_axis_title: str = ""
    x_axis_type: str = "linear"  # linear, log, category, datetime
    y_axis_type: str = "linear"
    
    # Interactive features
    zoom_enabled: bool = True
    pan_enabled: bool = True
    hover_enabled: bool = True
    selection_enabled: bool = False
    
    # Animation
    animation_enabled: bool = True
    animation_duration: int = 750
    
    # Legend
    legend_enabled: bool = True
    legend_position: str = "right"  # top, bottom, left, right
    
    # Custom styling
    custom_css: str = ""
    theme: str = "light"  # light, dark, custom
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "width": self.width,
            "height": self.height,
            "color_scheme": self.color_scheme,
            "background_color": self.background_color,
            "grid_enabled": self.grid_enabled,
            "x_axis_title": self.x_axis_title,
            "y_axis_title": self.y_axis_title,
            "x_axis_type": self.x_axis_type,
            "y_axis_type": self.y_axis_type,
            "zoom_enabled": self.zoom_enabled,
            "pan_enabled": self.pan_enabled, 
            "hover_enabled": self.hover_enabled,
            "selection_enabled": self.selection_enabled,
            "animation_enabled": self.animation_enabled,
            "animation_duration": self.animation_duration,
            "legend_enabled": self.legend_enabled,
            "legend_position": self.legend_position,
            "custom_css": self.custom_css,
            "theme": self.theme
        }


@dataclass
class DataSeries:
    """Data series for visualization"""
    name: str
    data: List[Any]
    x_data: Optional[List[Any]] = None
    chart_type: Optional[ChartType] = None
    color: Optional[str] = None
    line_style: str = "solid"  # solid, dashed, dotted
    marker_style: str = "circle"  # circle, square, triangle, diamond
    opacity: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert series to dictionary"""
        return {
            "name": self.name,
            "data": self.data,
            "x_data": self.x_data,
            "chart_type": self.chart_type.value if self.chart_type else None,
            "color": self.color,
            "line_style": self.line_style,
            "marker_style": self.marker_style,
            "opacity": self.opacity,
            "metadata": self.metadata
        }


class VisualizationEngine:
    """Main visualization engine for creating interactive charts"""
    
    def __init__(self):
        # Available rendering backends
        self.backends = self._initialize_backends()
        self.default_backend = self._get_default_backend()
        
        # Chart templates and themes
        self.chart_templates = {}
        self.color_schemes = self._initialize_color_schemes()
        
        # Cache for generated visualizations
        self.visualization_cache = {}
        self.cache_enabled = True
        
        # Statistics
        self.render_stats = {
            "total_renders": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "render_errors": 0
        }
        
        logger.info(f"Visualization engine initialized with backend: {self.default_backend}")
    
    def _initialize_backends(self) -> Dict[str, Any]:
        """Initialize available visualization backends"""
        backends = {}
        
        # Plotly backend (preferred for interactive charts)
        plotly = require_optional("plotly")
        if plotly:
            backends["plotly"] = {
                "module": plotly,
                "capabilities": [RenderFormat.HTML, RenderFormat.PNG, RenderFormat.SVG, 
                               RenderFormat.PDF, RenderFormat.JSON, RenderFormat.INTERACTIVE],
                "chart_types": list(ChartType),
                "interactive": True
            }
            logger.info("Plotly backend available - interactive visualizations enabled")
        
        # Matplotlib backend (for static charts)
        matplotlib = require_optional("matplotlib")
        if matplotlib:
            backends["matplotlib"] = {
                "module": matplotlib,
                "capabilities": [RenderFormat.PNG, RenderFormat.SVG, RenderFormat.PDF],
                "chart_types": [ChartType.LINE, ChartType.BAR, ChartType.SCATTER, 
                               ChartType.PIE, ChartType.HISTOGRAM, ChartType.HEATMAP],
                "interactive": False
            }
            logger.info("Matplotlib backend available - static visualizations enabled")
        
        # Seaborn backend (for statistical visualizations)
        seaborn = require_optional("seaborn")
        if seaborn and matplotlib:
            backends["seaborn"] = {
                "module": seaborn,
                "capabilities": [RenderFormat.PNG, RenderFormat.SVG, RenderFormat.PDF],
                "chart_types": [ChartType.HISTOGRAM, ChartType.HEATMAP, ChartType.BOX_PLOT],
                "interactive": False
            }
            logger.info("Seaborn backend available - statistical visualizations enabled")
        
        return backends
    
    def _get_default_backend(self) -> str:
        """Get the default backend based on availability"""
        if "plotly" in self.backends:
            return "plotly"
        elif "matplotlib" in self.backends:
            return "matplotlib" 
        elif "seaborn" in self.backends:
            return "seaborn"
        else:
            logger.warning("No visualization backends available")
            return "none"
    
    def _initialize_color_schemes(self) -> Dict[str, List[str]]:
        """Initialize color schemes for visualizations"""
        return {
            "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                       "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
            "corporate": ["#003f5c", "#2f4b7c", "#665191", "#a05195", "#d45087", 
                         "#f95d6a", "#ff7c43", "#ffa600"],
            "pastel": ["#fbb4ae", "#b3cde3", "#ccebc5", "#decbe4", "#fed9a6", 
                      "#ffffcc", "#e5d8bd", "#fddaec", "#f2f2f2"],
            "dark": ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", 
                    "#fdb462", "#b3de69", "#fccde5", "#d9d9d9"],
            "viridis": ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", 
                       "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"],
            "business": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#593E2C", 
                        "#8B8680", "#5A7F7A", "#D62839", "#BA1200", "#2F1B14"]
        }
    
    def create_chart(self, chart_type: ChartType, 
                    data_series: List[DataSeries],
                    config: Optional[ChartConfiguration] = None,
                    backend: Optional[str] = None) -> Dict[str, Any]:
        """Create a chart with specified type and data"""
        backend = backend or self.default_backend
        config = config or ChartConfiguration()
        
        if backend not in self.backends:
            raise ValueError(f"Backend '{backend}' not available")
        
        # Check if chart type is supported by backend
        if chart_type not in self.backends[backend]["chart_types"]:
            raise ValueError(f"Chart type '{chart_type.value}' not supported by backend '{backend}'")
        
        # Generate cache key
        cache_key = self._generate_cache_key(chart_type, data_series, config, backend)
        
        # Check cache
        if self.cache_enabled and cache_key in self.visualization_cache:
            self.render_stats["cache_hits"] += 1
            return self.visualization_cache[cache_key]
        
        self.render_stats["cache_misses"] += 1
        
        try:
            # Create visualization based on backend
            if backend == "plotly":
                result = self._create_plotly_chart(chart_type, data_series, config)
            elif backend == "matplotlib":
                result = self._create_matplotlib_chart(chart_type, data_series, config)
            elif backend == "seaborn":
                result = self._create_seaborn_chart(chart_type, data_series, config)
            else:
                raise ValueError(f"Unsupported backend: {backend}")
            
            # Cache result
            if self.cache_enabled:
                self.visualization_cache[cache_key] = result
            
            self.render_stats["total_renders"] += 1
            return result
            
        except Exception as e:
            self.render_stats["render_errors"] += 1
            logger.error(f"Error creating chart: {e}")
            raise
    
    def _create_plotly_chart(self, chart_type: ChartType, 
                           data_series: List[DataSeries],
                           config: ChartConfiguration) -> Dict[str, Any]:
        """Create chart using Plotly backend"""
        plotly = self.backends["plotly"]["module"]
        go = plotly.graph_objects
        
        fig = go.Figure()
        
        # Add data series to figure
        for i, series in enumerate(data_series):
            color = series.color or self._get_color_from_scheme(config.color_scheme, i)
            
            if chart_type == ChartType.LINE:
                fig.add_trace(go.Scatter(
                    x=series.x_data or list(range(len(series.data))),
                    y=series.data,
                    mode='lines+markers' if len(series.data) < 50 else 'lines',
                    name=series.name,
                    line=dict(color=color, dash=series.line_style),
                    opacity=series.opacity
                ))
            
            elif chart_type == ChartType.BAR:
                fig.add_trace(go.Bar(
                    x=series.x_data or list(range(len(series.data))),
                    y=series.data,
                    name=series.name,
                    marker_color=color,
                    opacity=series.opacity
                ))
            
            elif chart_type == ChartType.SCATTER:
                fig.add_trace(go.Scatter(
                    x=series.x_data or list(range(len(series.data))),
                    y=series.data,
                    mode='markers',
                    name=series.name,
                    marker=dict(color=color, symbol=series.marker_style),
                    opacity=series.opacity
                ))
            
            elif chart_type == ChartType.PIE:
                fig.add_trace(go.Pie(
                    labels=series.x_data or [f"Item {i+1}" for i in range(len(series.data))],
                    values=series.data,
                    name=series.name,
                    opacity=series.opacity
                ))
            
            elif chart_type == ChartType.HISTOGRAM:
                fig.add_trace(go.Histogram(
                    x=series.data,
                    name=series.name,
                    marker_color=color,
                    opacity=series.opacity
                ))
            
            elif chart_type == ChartType.HEATMAP:
                fig.add_trace(go.Heatmap(
                    z=series.data,
                    colorscale='Viridis',
                    name=series.name
                ))
            
            elif chart_type == ChartType.AREA:
                fig.add_trace(go.Scatter(
                    x=series.x_data or list(range(len(series.data))),
                    y=series.data,
                    fill='tonexty' if i > 0 else 'tozeroy',
                    mode='lines',
                    name=series.name,
                    line=dict(color=color),
                    opacity=series.opacity
                ))
            
            elif chart_type == ChartType.BOX_PLOT:
                fig.add_trace(go.Box(
                    y=series.data,
                    name=series.name,
                    marker_color=color
                ))
            
            elif chart_type == ChartType.GAUGE:
                fig.add_trace(go.Indicator(
                    mode="gauge+number+delta",
                    value=series.data[0] if series.data else 0,
                    title={'text': series.name},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': color},
                           'steps': [{'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 100], 'color': "gray"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75, 'value': 90}}
                ))
        
        # Configure layout
        layout_config = {
            'title': config.title,
            'width': config.width,
            'height': config.height,
            'plot_bgcolor': config.background_color,
            'paper_bgcolor': config.background_color,
            'showlegend': config.legend_enabled,
            'hovermode': 'closest' if config.hover_enabled else False,
        }
        
        if config.x_axis_title:
            layout_config['xaxis'] = {'title': config.x_axis_title}
        if config.y_axis_title:
            layout_config['yaxis'] = {'title': config.y_axis_title}
        
        # Apply theme
        if config.theme == "dark":
            layout_config.update({
                'plot_bgcolor': '#2e2e2e',
                'paper_bgcolor': '#1e1e1e',
                'font': {'color': 'white'}
            })
        
        fig.update_layout(**layout_config)
        
        # Return result with multiple formats
        return {
            "chart_type": chart_type.value,
            "backend": "plotly",
            "html": fig.to_html(include_plotlyjs='cdn'),
            "json": fig.to_json(),
            "config": config.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "interactive": True
        }
    
    def _create_matplotlib_chart(self, chart_type: ChartType,
                                data_series: List[DataSeries],
                                config: ChartConfiguration) -> Dict[str, Any]:
        """Create chart using Matplotlib backend"""
        matplotlib = self.backends["matplotlib"]["module"]
        plt = matplotlib.pyplot
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))
        
        # Add data series
        for i, series in enumerate(data_series):
            color = series.color or self._get_color_from_scheme(config.color_scheme, i)
            x_data = series.x_data or list(range(len(series.data)))
            
            if chart_type == ChartType.LINE:
                ax.plot(x_data, series.data, label=series.name, color=color,
                       linestyle=series.line_style, alpha=series.opacity)
            
            elif chart_type == ChartType.BAR:
                ax.bar(x_data, series.data, label=series.name, color=color,
                      alpha=series.opacity)
            
            elif chart_type == ChartType.SCATTER:
                ax.scatter(x_data, series.data, label=series.name, color=color,
                          alpha=series.opacity)
            
            elif chart_type == ChartType.HISTOGRAM:
                ax.hist(series.data, label=series.name, color=color,
                       alpha=series.opacity, bins=30)
        
        # Configure chart
        if config.title:
            ax.set_title(config.title)
        if config.x_axis_title:
            ax.set_xlabel(config.x_axis_title)
        if config.y_axis_title:
            ax.set_ylabel(config.y_axis_title)
        
        if config.legend_enabled and len(data_series) > 1:
            ax.legend(loc=config.legend_position)
        
        if config.grid_enabled:
            ax.grid(True, alpha=0.3)
        
        # Apply theme
        if config.theme == "dark":
            fig.patch.set_facecolor('#1e1e1e')
            ax.set_facecolor('#2e2e2e')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')  
            ax.spines['left'].set_color('white')
            ax.tick_params(colors='white')
            ax.yaxis.label.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.title.set_color('white')
        
        # Convert to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', 
                   facecolor=fig.get_facecolor(), dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        
        return {
            "chart_type": chart_type.value,
            "backend": "matplotlib", 
            "image": f"data:image/png;base64,{image_base64}",
            "config": config.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "interactive": False
        }
    
    def _create_seaborn_chart(self, chart_type: ChartType,
                             data_series: List[DataSeries], 
                             config: ChartConfiguration) -> Dict[str, Any]:
        """Create chart using Seaborn backend"""
        seaborn = self.backends["seaborn"]["module"]
        matplotlib = self.backends["matplotlib"]["module"]
        plt = matplotlib.pyplot
        
        # Set seaborn style
        seaborn.set_theme(style="whitegrid" if config.theme == "light" else "darkgrid")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))
        
        # Create chart based on type
        if chart_type == ChartType.HISTOGRAM:
            for series in data_series:
                seaborn.histplot(data=series.data, label=series.name, ax=ax, alpha=series.opacity)
        
        elif chart_type == ChartType.BOX_PLOT:
            if len(data_series) == 1:
                seaborn.boxplot(y=data_series[0].data, ax=ax)
            else:
                # Multiple series boxplot
                import pandas as pd
                df_data = []
                for series in data_series:
                    for value in series.data:
                        df_data.append({"value": value, "series": series.name})
                df = pd.DataFrame(df_data)
                seaborn.boxplot(data=df, x="series", y="value", ax=ax)
        
        # Configure chart
        if config.title:
            ax.set_title(config.title)
        if config.x_axis_title:
            ax.set_xlabel(config.x_axis_title)
        if config.y_axis_title:
            ax.set_ylabel(config.y_axis_title)
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        
        return {
            "chart_type": chart_type.value,
            "backend": "seaborn",
            "image": f"data:image/png;base64,{image_base64}",
            "config": config.to_dict(), 
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "interactive": False
        }
    
    def _get_color_from_scheme(self, scheme_name: str, index: int) -> str:
        """Get color from color scheme by index"""
        colors = self.color_schemes.get(scheme_name, self.color_schemes["default"])
        return colors[index % len(colors)]
    
    def _generate_cache_key(self, chart_type: ChartType, 
                           data_series: List[DataSeries],
                           config: ChartConfiguration,
                           backend: str) -> str:
        """Generate cache key for visualization"""
        import hashlib
        
        # Create a string representation of the inputs
        key_data = {
            "chart_type": chart_type.value,
            "backend": backend,
            "data_series": [series.to_dict() for series in data_series],
            "config": config.to_dict()
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def create_dashboard_layout(self, charts: List[Dict[str, Any]],
                               layout_type: str = "grid") -> Dict[str, Any]:
        """Create a dashboard layout with multiple charts"""
        if not charts:
            return {"error": "No charts provided"}
        
        dashboard = {
            "layout_type": layout_type,
            "charts": charts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_charts": len(charts)
        }
        
        # Generate HTML for dashboard
        if layout_type == "grid":
            dashboard["html"] = self._create_grid_layout(charts)
        elif layout_type == "tabs":
            dashboard["html"] = self._create_tabs_layout(charts)
        elif layout_type == "sidebar":
            dashboard["html"] = self._create_sidebar_layout(charts)
        else:
            dashboard["html"] = self._create_simple_layout(charts)
        
        return dashboard
    
    def _create_grid_layout(self, charts: List[Dict[str, Any]]) -> str:
        """Create grid layout HTML"""
        html_parts = ['<div class="dashboard-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; padding: 20px;">']
        
        for i, chart in enumerate(charts):
            chart_html = chart.get("html", "") or f'<img src="{chart.get("image", "")}" style="max-width: 100%;">'
            html_parts.append(f'<div class="chart-container" id="chart-{i}">{chart_html}</div>')
        
        html_parts.append('</div>')
        return '\n'.join(html_parts)
    
    def _create_tabs_layout(self, charts: List[Dict[str, Any]]) -> str:
        """Create tabs layout HTML"""
        tab_headers = []
        tab_contents = []
        
        for i, chart in enumerate(charts):
            title = chart.get("config", {}).get("title", f"Chart {i+1}")
            tab_headers.append(f'<button class="tab-button" onclick="showTab({i})">{title}</button>')
            
            chart_html = chart.get("html", "") or f'<img src="{chart.get("image", "")}" style="max-width: 100%;">'
            tab_contents.append(f'<div class="tab-content" id="tab-{i}" style="display: {"block" if i == 0 else "none"};">{chart_html}</div>')
        
        html = f'''
        <div class="dashboard-tabs">
            <div class="tab-headers">{''.join(tab_headers)}</div>
            <div class="tab-body">{''.join(tab_contents)}</div>
        </div>
        <script>
        function showTab(tabIndex) {{
            var contents = document.querySelectorAll('.tab-content');
            contents.forEach(function(content, index) {{
                content.style.display = index === tabIndex ? 'block' : 'none';
            }});
        }}
        </script>
        '''
        return html
    
    def _create_sidebar_layout(self, charts: List[Dict[str, Any]]) -> str:
        """Create sidebar layout HTML"""
        return self._create_simple_layout(charts)  # Simplified for now
    
    def _create_simple_layout(self, charts: List[Dict[str, Any]]) -> str:
        """Create simple vertical layout HTML"""
        html_parts = ['<div class="dashboard-simple" style="padding: 20px;">']
        
        for i, chart in enumerate(charts):
            chart_html = chart.get("html", "") or f'<img src="{chart.get("image", "")}" style="max-width: 100%;">'
            html_parts.append(f'<div class="chart-container" style="margin-bottom: 30px;">{chart_html}</div>')
        
        html_parts.append('</div>')
        return '\n'.join(html_parts)
    
    def get_available_backends(self) -> Dict[str, Any]:
        """Get information about available backends"""
        return {
            backend: {
                "available": True,
                "capabilities": info["capabilities"],
                "chart_types": [ct.value for ct in info["chart_types"]],
                "interactive": info["interactive"]
            }
            for backend, info in self.backends.items()
        }
    
    def get_render_statistics(self) -> Dict[str, Any]:
        """Get rendering statistics"""
        return {
            **self.render_stats,
            "cache_hit_rate": self.render_stats["cache_hits"] / max(1, self.render_stats["total_renders"]),
            "cache_size": len(self.visualization_cache),
            "available_backends": list(self.backends.keys()),
            "default_backend": self.default_backend
        }
    
    def clear_cache(self):
        """Clear visualization cache"""
        self.visualization_cache.clear()
        logger.info("Visualization cache cleared")


# Export main classes
__all__ = [
    'ChartType',
    'RenderFormat',
    'ChartConfiguration',
    'DataSeries', 
    'VisualizationEngine'
]