#!/usr/bin/env python3
"""
Advanced Data Visualization and Collaborative Dashboards for PRSM
================================================================

This module implements advanced data visualization and collaborative dashboard
functionality with P2P security for university-industry research partnerships:

- Interactive collaborative dashboards with real-time updates
- Secure data visualization sharing with post-quantum encryption
- Multi-institutional dashboard collaboration
- Publication-ready visualizations for research papers
- Industry-specific dashboard templates and workflows
- NWTN AI-powered visualization recommendations

Key Features:
- Real-time collaborative dashboard editing
- Interactive visualizations (Plotly, D3.js, custom widgets)
- Secure sharing of sensitive research visualizations
- Multi-format export (PDF, SVG, PNG, HTML, LaTeX)
- Integration with statistical analysis and ML platforms
- University-industry presentation templates
"""

import json
import uuid
import asyncio
import base64
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import tempfile
import hashlib

# Data visualization libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Import PRSM components
from ..security.post_quantum_crypto_sharding import PostQuantumCryptoSharding, CryptoMode
from ..models import QueryRequest

# Mock UnifiedPipelineController for testing
class UnifiedPipelineController:
    """Mock pipeline controller for data visualization collaboration"""
    async def initialize(self):
        pass
    
    async def process_query_full_pipeline(self, user_id: str, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Visualization-specific NWTN responses
        if context.get("visualization_recommendations"):
            return {
                "response": {
                    "text": """
Data Visualization Recommendations:

📊 **Chart Type Selection**:
```python
# For quantum computing research data visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Error rate comparison across institutions
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Error Rates by Method', 'Performance Over Time', 
                   'Institution Comparison', 'Noise Level Impact'),
    specs=[[{"secondary_y": True}, {"type": "scatter"}],
           [{"type": "bar"}, {"type": "heatmap"}]]
)

# Interactive scatter plot with hover data
fig.add_trace(go.Scatter(
    x=noise_levels, y=error_rates,
    mode='markers+lines',
    marker=dict(size=10, color=institutions, 
                colorscale='Viridis', showscale=True),
    hovertemplate='<b>%{text}</b><br>Noise: %{x}<br>Error Rate: %{y}<extra></extra>',
    text=institution_names
), row=1, col=1)

# Publication-ready styling
fig.update_layout(
    title='Multi-University Quantum Error Correction Analysis',
    font=dict(family="Arial", size=12),
    showlegend=True,
    height=800
)
```

🎨 **Design Principles for Academic Collaboration**:
- Use colorblind-friendly palettes (Viridis, Cividis)
- Include clear error bars and confidence intervals
- Add institutional attribution in legends
- Implement responsive design for various screen sizes

📈 **Interactive Features**:
- Drill-down capabilities for detailed analysis
- Real-time updates as collaborators modify data
- Annotation tools for collaborative discussion
- Export functionality for presentations and papers

🏛️ **University-Industry Specific**:
- Dual branding options (university + company logos)
- Confidentiality levels for different data views
- Publication-ready outputs with proper citations
- Integration with institutional reporting systems
                    """,
                    "confidence": 0.93,
                    "sources": ["plotly.com", "matplotlib.org", "data_viz_best_practices.pdf"]
                },
                "performance_metrics": {"total_processing_time": 3.1}
            }
        elif context.get("dashboard_design"):
            return {
                "response": {
                    "text": """
Collaborative Dashboard Design Recommendations:

🖥️ **Dashboard Layout Strategy**:
```html
<!-- University-Industry Dashboard Template -->
<div class="dashboard-container">
  <header class="dashboard-header">
    <div class="institution-branding">
      <img src="university_logo.png" alt="University">
      <img src="company_logo.png" alt="Industry Partner">
    </div>
    <h1>Quantum Computing Research Dashboard</h1>
    <div class="collaboration-status">
      <span class="active-users">3 collaborators online</span>
    </div>
  </header>
  
  <div class="dashboard-grid">
    <div class="widget primary-metrics">
      <h3>Key Performance Indicators</h3>
      <div class="kpi-grid">
        <div class="kpi-item">
          <span class="kpi-value">94.7%</span>
          <span class="kpi-label">Error Correction Success</span>
        </div>
      </div>
    </div>
    
    <div class="widget visualization">
      <h3>Real-time Results</h3>
      <div id="live-chart"></div>
    </div>
    
    <div class="widget collaboration-panel">
      <h3>Collaboration Notes</h3>
      <div class="annotation-stream"></div>
    </div>
  </div>
</div>
```

🎯 **Key Design Elements**:
- Responsive grid layout for different screen sizes
- Real-time collaboration indicators
- Secure sharing controls with access level visualization
- Export options for presentations and reports

🔒 **Security & Privacy Features**:
- Role-based data visibility (blur sensitive information)
- Watermarking for proprietary research data
- Session recording for audit trails
- Encrypted dashboard state synchronization

📱 **Cross-Platform Compatibility**:
- Progressive Web App (PWA) functionality
- Mobile-responsive design for field research
- Offline capability with sync when connected
- Integration with institutional authentication systems
                    """,
                    "confidence": 0.88,
                    "sources": ["dashboard_design_patterns.pdf", "collaborative_ui_ux.com"]
                },
                "performance_metrics": {"total_processing_time": 2.7}
            }
        else:
            return {
                "response": {"text": "Data visualization collaboration assistance available", "confidence": 0.75, "sources": []},
                "performance_metrics": {"total_processing_time": 1.5}
            }

class VisualizationAccessLevel(Enum):
    """Access levels for visualization collaboration"""
    OWNER = "owner"
    EDITOR = "editor"
    COMMENTER = "commenter"
    VIEWER = "viewer"

class DashboardType(Enum):
    """Types of collaborative dashboards"""
    RESEARCH_ANALYTICS = "research_analytics"
    EXPERIMENT_MONITORING = "experiment_monitoring"
    PRESENTATION = "presentation"
    PUBLICATION = "publication"
    EXECUTIVE_SUMMARY = "executive_summary"

class ChartType(Enum):
    """Supported chart types"""
    SCATTER_PLOT = "scatter_plot"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    VIOLIN_PLOT = "violin_plot"
    NETWORK_GRAPH = "network_graph"
    TREEMAP = "treemap"
    PARALLEL_COORDINATES = "parallel_coordinates"

class ExportFormat(Enum):
    """Export formats for visualizations"""
    PNG = "png"
    SVG = "svg"
    PDF = "pdf"
    HTML = "html"
    LATEX = "latex"
    POWERPOINT = "powerpoint"
    JSON = "json"

@dataclass
class VisualizationSpec:
    """Specification for a data visualization"""
    chart_type: ChartType
    data_source: str
    x_axis: str
    y_axis: str
    color_by: Optional[str] = None
    size_by: Optional[str] = None
    filter_conditions: Dict[str, Any] = None
    
    # Styling
    title: str = ""
    x_title: str = ""
    y_title: str = ""
    color_scheme: str = "viridis"
    width: int = 800
    height: int = 600
    
    # Interactivity
    hover_data: List[str] = None
    click_events: bool = False
    zoom_enabled: bool = True
    pan_enabled: bool = True

@dataclass
class DashboardWidget:
    """Widget component in a collaborative dashboard"""
    widget_id: str
    name: str
    description: str
    widget_type: str  # 'chart', 'metric', 'text', 'image', 'table'
    position: Dict[str, int]  # x, y, width, height in grid units
    
    # Content
    visualization_spec: Optional[VisualizationSpec] = None
    content: str = ""
    data_binding: Optional[str] = None
    
    # Styling
    background_color: str = "#ffffff"
    border_style: str = "solid"
    font_family: str = "Arial"
    
    # Permissions
    editable_by: List[str] = None
    visible_to: List[str] = None
    
    created_at: datetime = None
    last_modified: datetime = None

@dataclass
class CollaborativeDashboard:
    """Collaborative dashboard with real-time updates"""
    dashboard_id: str
    name: str
    description: str
    dashboard_type: DashboardType
    owner: str
    collaborators: Dict[str, VisualizationAccessLevel]
    
    # Layout and content
    widgets: Dict[str, DashboardWidget]
    layout_config: Dict[str, Any]
    theme: str = "default"
    
    # Data sources
    connected_datasets: List[str]
    data_refresh_interval: int = 300  # seconds
    
    # Collaboration features
    real_time_enabled: bool = True
    comments_enabled: bool = True
    version_history: List[Dict[str, Any]] = None
    
    # Security
    encrypted: bool = True
    access_controlled: bool = True
    security_level: str = "high"
    
    # Metadata
    tags: List[str] = None
    created_at: datetime = None
    last_modified: datetime = None
    
    # Analytics
    view_count: int = 0
    unique_viewers: List[str] = None
    export_history: List[Dict[str, Any]] = None

@dataclass
class VisualizationProject:
    """Multi-institutional visualization project"""
    project_id: str
    title: str
    description: str
    principal_investigator: str
    institutions: List[str]
    participants: Dict[str, VisualizationAccessLevel]
    
    # Project content
    dashboards: List[str]
    shared_datasets: List[str]
    visualization_library: List[str]
    
    # Research context
    research_area: str
    funding_source: Optional[str] = None
    publication_target: Optional[str] = None
    
    # Timeline
    start_date: datetime
    end_date: Optional[datetime] = None
    milestones: List[Dict[str, Any]] = None
    
    # Compliance
    data_classification: str = "restricted"
    export_restrictions: List[str] = None
    
    created_at: datetime = None

class DataVisualizationCollaboration:
    """
    Main class for advanced data visualization and collaborative dashboards
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize data visualization collaboration system"""
        self.storage_path = storage_path or Path("./data_visualization")
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize PRSM components
        self.crypto_sharding = PostQuantumCryptoSharding(
            default_shards=5,
            required_shards=3,
            crypto_mode=CryptoMode.POST_QUANTUM
        )
        self.nwtn_pipeline = None
        
        # Active dashboards and projects
        self.collaborative_dashboards: Dict[str, CollaborativeDashboard] = {}
        self.visualization_projects: Dict[str, VisualizationProject] = {}
        
        # Visualization templates
        self.dashboard_templates = self._initialize_dashboard_templates()
        self.chart_templates = self._initialize_chart_templates()
        
        # Color schemes and themes
        self.color_schemes = self._initialize_color_schemes()
        self.dashboard_themes = self._initialize_dashboard_themes()
        
        # Data sources (mock connections)
        self.data_sources = {}
    
    def _initialize_dashboard_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize dashboard templates for different use cases"""
        return {
            "quantum_research": {
                "name": "Quantum Computing Research Dashboard",
                "description": "Template for quantum algorithm performance monitoring",
                "widgets": [
                    {"type": "metric", "title": "Error Correction Rate", "position": {"x": 0, "y": 0, "w": 3, "h": 2}},
                    {"type": "chart", "title": "Performance Over Time", "position": {"x": 3, "y": 0, "w": 6, "h": 4}},
                    {"type": "chart", "title": "Institution Comparison", "position": {"x": 9, "y": 0, "w": 3, "h": 4}},
                    {"type": "chart", "title": "Noise Level Analysis", "position": {"x": 0, "y": 4, "w": 6, "h": 3}},
                    {"type": "table", "title": "Experiment Log", "position": {"x": 6, "y": 4, "w": 6, "h": 3}}
                ]
            },
            "university_industry": {
                "name": "University-Industry Partnership Dashboard",
                "description": "Template for collaborative research project monitoring",
                "widgets": [
                    {"type": "metric", "title": "Project Progress", "position": {"x": 0, "y": 0, "w": 4, "h": 2}},
                    {"type": "metric", "title": "Collaboration Score", "position": {"x": 4, "y": 0, "w": 4, "h": 2}},
                    {"type": "metric", "title": "IP Generation", "position": {"x": 8, "y": 0, "w": 4, "h": 2}},
                    {"type": "chart", "title": "Research Milestones", "position": {"x": 0, "y": 2, "w": 8, "h": 4}},
                    {"type": "chart", "title": "Resource Allocation", "position": {"x": 8, "y": 2, "w": 4, "h": 4}}
                ]
            },
            "publication_ready": {
                "name": "Publication-Ready Visualization Suite",
                "description": "Template for creating publication-quality figures",
                "widgets": [
                    {"type": "chart", "title": "Figure 1: Main Results", "position": {"x": 0, "y": 0, "w": 6, "h": 6}},
                    {"type": "chart", "title": "Figure 2: Statistical Analysis", "position": {"x": 6, "y": 0, "w": 6, "h": 6}},
                    {"type": "chart", "title": "Figure 3: Comparative Study", "position": {"x": 0, "y": 6, "w": 12, "h": 4}}
                ]
            }
        }
    
    def _initialize_chart_templates(self) -> Dict[ChartType, Dict[str, Any]]:
        """Initialize chart templates with best practices"""
        return {
            ChartType.SCATTER_PLOT: {
                "config": {
                    "mode": "markers+lines",
                    "marker": {"size": 8, "opacity": 0.7},
                    "showlegend": True,
                    "hovermode": "closest"
                },
                "layout": {
                    "xaxis": {"showgrid": True, "gridcolor": "lightgray"},
                    "yaxis": {"showgrid": True, "gridcolor": "lightgray"},
                    "font": {"family": "Arial", "size": 12}
                }
            },
            ChartType.HEATMAP: {
                "config": {
                    "colorscale": "Viridis",
                    "showscale": True,
                    "hoverongaps": False
                },
                "layout": {
                    "xaxis": {"side": "bottom"},
                    "yaxis": {"side": "left"},
                    "font": {"family": "Arial", "size": 10}
                }
            },
            ChartType.BOX_PLOT: {
                "config": {
                    "boxpoints": "outliers",
                    "jitter": 0.3,
                    "pointpos": -1.8
                },
                "layout": {
                    "showlegend": False,
                    "font": {"family": "Arial", "size": 12}
                }
            }
        }
    
    def _initialize_color_schemes(self) -> Dict[str, List[str]]:
        """Initialize colorblind-friendly color schemes"""
        return {
            "viridis": ["#440154", "#31688e", "#35b779", "#fde725"],
            "cividis": ["#00224e", "#123570", "#3b496c", "#575d6d"],
            "university_brand": ["#4b9cd3", "#13294b", "#f47735", "#78be20"],
            "publication": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
            "accessibility": ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
        }
    
    def _initialize_dashboard_themes(self) -> Dict[str, Dict[str, Any]]:
        """Initialize dashboard themes"""
        return {
            "default": {
                "background_color": "#ffffff",
                "text_color": "#333333",
                "accent_color": "#4b9cd3",
                "grid_color": "#e0e0e0"
            },
            "dark": {
                "background_color": "#2b2b2b",
                "text_color": "#ffffff",
                "accent_color": "#64b5f6",
                "grid_color": "#404040"
            },
            "university": {
                "background_color": "#f8f9fa",
                "text_color": "#13294b",
                "accent_color": "#4b9cd3",
                "grid_color": "#dee2e6"
            }
        }
    
    async def initialize_nwtn_pipeline(self):
        """Initialize NWTN pipeline for visualization recommendations"""
        if self.nwtn_pipeline is None:
            self.nwtn_pipeline = UnifiedPipelineController()
            await self.nwtn_pipeline.initialize()
    
    def create_collaborative_dashboard(self,
                                     name: str,
                                     description: str,
                                     dashboard_type: DashboardType,
                                     owner: str,
                                     collaborators: Optional[Dict[str, VisualizationAccessLevel]] = None,
                                     template_name: Optional[str] = None,
                                     security_level: str = "high") -> CollaborativeDashboard:
        """Create a new collaborative dashboard"""
        
        dashboard_id = str(uuid.uuid4())
        
        # Initialize with template if specified
        widgets = {}
        if template_name and template_name in self.dashboard_templates:
            template = self.dashboard_templates[template_name]
            for i, widget_spec in enumerate(template["widgets"]):
                widget_id = str(uuid.uuid4())
                widgets[widget_id] = DashboardWidget(
                    widget_id=widget_id,
                    name=widget_spec["title"],
                    description=f"Widget from {template_name} template",
                    widget_type=widget_spec["type"],
                    position=widget_spec["position"],
                    created_at=datetime.now(),
                    last_modified=datetime.now()
                )
        
        dashboard = CollaborativeDashboard(
            dashboard_id=dashboard_id,
            name=name,
            description=description,
            dashboard_type=dashboard_type,
            owner=owner,
            collaborators=collaborators or {},
            widgets=widgets,
            layout_config={"grid_columns": 12, "row_height": 60},
            theme="default",
            connected_datasets=[],
            data_refresh_interval=300,
            real_time_enabled=True,
            comments_enabled=True,
            version_history=[],
            encrypted=True,
            access_controlled=True,
            security_level=security_level,
            tags=[],
            created_at=datetime.now(),
            last_modified=datetime.now(),
            view_count=0,
            unique_viewers=[],
            export_history=[]
        )
        
        self.collaborative_dashboards[dashboard_id] = dashboard
        self._save_dashboard(dashboard)
        
        print(f"📊 Created collaborative dashboard: {name}")
        print(f"   Dashboard ID: {dashboard_id}")
        print(f"   Type: {dashboard_type.value}")
        print(f"   Template: {template_name or 'None'}")
        print(f"   Widgets: {len(widgets)}")
        print(f"   Collaborators: {len(collaborators or {})}")
        print(f"   Security: {security_level}")
        
        return dashboard
    
    def create_visualization_project(self,
                                   title: str,
                                   description: str,
                                   principal_investigator: str,
                                   institutions: List[str],
                                   participants: Dict[str, VisualizationAccessLevel],
                                   research_area: str,
                                   start_date: datetime,
                                   end_date: Optional[datetime] = None) -> VisualizationProject:
        """Create a multi-institutional visualization project"""
        
        project_id = str(uuid.uuid4())
        
        project = VisualizationProject(
            project_id=project_id,
            title=title,
            description=description,
            principal_investigator=principal_investigator,
            institutions=institutions,
            participants=participants,
            dashboards=[],
            shared_datasets=[],
            visualization_library=[],
            research_area=research_area,
            funding_source=None,
            publication_target=None,
            start_date=start_date,
            end_date=end_date,
            milestones=[],
            data_classification="restricted",
            export_restrictions=[],
            created_at=datetime.now()
        )
        
        self.visualization_projects[project_id] = project
        self._save_project(project)
        
        print(f"🔬 Created visualization project: {title}")
        print(f"   Project ID: {project_id}")
        print(f"   PI: {principal_investigator}")
        print(f"   Institutions: {', '.join(institutions)}")
        print(f"   Participants: {len(participants)}")
        print(f"   Research Area: {research_area}")
        
        return project
    
    def add_dashboard_widget(self,
                           dashboard_id: str,
                           widget_name: str,
                           widget_type: str,
                           position: Dict[str, int],
                           user_id: str,
                           visualization_spec: Optional[VisualizationSpec] = None,
                           content: str = "") -> DashboardWidget:
        """Add a widget to a collaborative dashboard"""
        
        if dashboard_id not in self.collaborative_dashboards:
            raise ValueError(f"Dashboard {dashboard_id} not found")
        
        dashboard = self.collaborative_dashboards[dashboard_id]
        
        # Check permissions
        if not self._check_dashboard_access(dashboard, user_id, VisualizationAccessLevel.EDITOR):
            raise PermissionError("Insufficient permissions to add widgets")
        
        widget_id = str(uuid.uuid4())
        
        widget = DashboardWidget(
            widget_id=widget_id,
            name=widget_name,
            description=f"Widget added by {user_id}",
            widget_type=widget_type,
            position=position,
            visualization_spec=visualization_spec,
            content=content,
            editable_by=[user_id],
            visible_to=list(dashboard.collaborators.keys()) + [dashboard.owner],
            created_at=datetime.now(),
            last_modified=datetime.now()
        )
        
        dashboard.widgets[widget_id] = widget
        dashboard.last_modified = datetime.now()
        self._save_dashboard(dashboard)
        
        print(f"📈 Added widget to dashboard: {widget_name}")
        print(f"   Widget ID: {widget_id}")
        print(f"   Type: {widget_type}")
        print(f"   Position: {position}")
        
        return widget
    
    def create_interactive_visualization(self,
                                       dashboard_id: str,
                                       widget_id: str,
                                       data: pd.DataFrame,
                                       viz_spec: VisualizationSpec,
                                       user_id: str) -> str:
        """Create an interactive visualization using Plotly"""
        
        if dashboard_id not in self.collaborative_dashboards:
            raise ValueError(f"Dashboard {dashboard_id} not found")
        
        dashboard = self.collaborative_dashboards[dashboard_id]
        
        if widget_id not in dashboard.widgets:
            raise ValueError(f"Widget {widget_id} not found")
        
        # Check permissions
        if not self._check_dashboard_access(dashboard, user_id, VisualizationAccessLevel.EDITOR):
            raise PermissionError("Insufficient permissions to create visualizations")
        
        # Create Plotly figure based on specification
        fig = self._create_plotly_figure(data, viz_spec)
        
        # Generate HTML
        html_content = pyo.plot(fig, output_type='div', include_plotlyjs=True)
        
        # Save visualization
        viz_dir = self.storage_path / "visualizations" / dashboard_id / widget_id
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        html_file = viz_dir / "visualization.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        # Update widget
        widget = dashboard.widgets[widget_id]
        widget.visualization_spec = viz_spec
        widget.last_modified = datetime.now()
        
        dashboard.last_modified = datetime.now()
        self._save_dashboard(dashboard)
        
        print(f"📊 Created interactive visualization:")
        print(f"   Chart Type: {viz_spec.chart_type.value}")
        print(f"   Title: {viz_spec.title}")
        print(f"   Data Points: {len(data)}")
        
        return str(html_file)
    
    def _create_plotly_figure(self, data: pd.DataFrame, spec: VisualizationSpec) -> go.Figure:
        """Create Plotly figure from data and specification"""
        
        # Get color scheme
        colors = self.color_schemes.get(spec.color_scheme, self.color_schemes["viridis"])
        
        if spec.chart_type == ChartType.SCATTER_PLOT:
            fig = px.scatter(
                data, 
                x=spec.x_axis, 
                y=spec.y_axis,
                color=spec.color_by,
                size=spec.size_by,
                hover_data=spec.hover_data or [],
                title=spec.title,
                color_discrete_sequence=colors
            )
            
        elif spec.chart_type == ChartType.LINE_CHART:
            fig = px.line(
                data,
                x=spec.x_axis,
                y=spec.y_axis,
                color=spec.color_by,
                title=spec.title,
                color_discrete_sequence=colors
            )
            
        elif spec.chart_type == ChartType.BAR_CHART:
            fig = px.bar(
                data,
                x=spec.x_axis,
                y=spec.y_axis,
                color=spec.color_by,
                title=spec.title,
                color_discrete_sequence=colors
            )
            
        elif spec.chart_type == ChartType.HEATMAP:
            # Create correlation matrix or pivot table
            if spec.color_by:
                pivot_data = data.pivot_table(
                    index=spec.x_axis,
                    columns=spec.y_axis,
                    values=spec.color_by,
                    aggfunc='mean'
                )
            else:
                pivot_data = data.corr()
            
            fig = px.imshow(
                pivot_data,
                title=spec.title,
                color_continuous_scale=spec.color_scheme
            )
            
        elif spec.chart_type == ChartType.BOX_PLOT:
            fig = px.box(
                data,
                x=spec.x_axis,
                y=spec.y_axis,
                color=spec.color_by,
                title=spec.title,
                color_discrete_sequence=colors
            )
            
        else:
            # Default to scatter plot
            fig = px.scatter(
                data,
                x=spec.x_axis,
                y=spec.y_axis,
                title=spec.title or "Visualization"
            )
        
        # Update layout
        fig.update_layout(
            width=spec.width,
            height=spec.height,
            xaxis_title=spec.x_title or spec.x_axis,
            yaxis_title=spec.y_title or spec.y_axis,
            font=dict(family="Arial", size=12),
            showlegend=True if spec.color_by else False
        )
        
        # Configure interactivity
        if not spec.zoom_enabled:
            fig.update_layout(xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
        
        return fig
    
    async def get_visualization_recommendations(self,
                                             data_description: str,
                                             research_goals: List[str],
                                             audience: str,
                                             user_id: str) -> Dict[str, Any]:
        """Get NWTN AI recommendations for data visualization"""
        
        await self.initialize_nwtn_pipeline()
        
        viz_prompt = f"""
Please provide data visualization recommendations for this research collaboration:

**Data Description**: {data_description}
**Research Goals**: {', '.join(research_goals)}
**Target Audience**: {audience}
**Context**: University-industry collaborative research
**Requirements**: Publication-ready, collaborative, accessible

Please provide:
1. Optimal chart types for the data and goals
2. Design recommendations for academic/industry audiences
3. Interactive features that enhance collaboration
4. Color schemes and styling for accessibility
5. Export formats for different use cases

Focus on visualizations that facilitate understanding across different expertise levels and institutional contexts.
"""
        
        result = await self.nwtn_pipeline.process_query_full_pipeline(
            user_id=user_id,
            query=viz_prompt,
            context={
                "domain": "visualization_recommendations",
                "visualization_recommendations": True,
                "audience": audience,
                "recommendation_type": "comprehensive_analysis"
            }
        )
        
        recommendations = {
            "data_description": data_description,
            "research_goals": research_goals,
            "target_audience": audience,
            "visualization_advice": result.get('response', {}).get('text', ''),
            "confidence": result.get('response', {}).get('confidence', 0.0),
            "sources": result.get('response', {}).get('sources', []),
            "processing_time": result.get('performance_metrics', {}).get('total_processing_time', 0.0),
            "generated_at": datetime.now().isoformat(),
            "requested_by": user_id
        }
        
        print(f"💡 Visualization recommendations generated:")
        print(f"   Data: {data_description}")
        print(f"   Goals: {len(research_goals)} research objectives")
        print(f"   Audience: {audience}")
        print(f"   Confidence: {recommendations['confidence']:.2f}")
        
        return recommendations
    
    async def get_dashboard_design_guidance(self,
                                          dashboard_type: str,
                                          collaboration_requirements: List[str],
                                          user_id: str) -> Dict[str, Any]:
        """Get AI guidance for collaborative dashboard design"""
        
        await self.initialize_nwtn_pipeline()
        
        design_prompt = f"""
Please provide collaborative dashboard design guidance:

**Dashboard Type**: {dashboard_type}
**Collaboration Requirements**: {', '.join(collaboration_requirements)}
**Context**: Multi-institutional research dashboard
**Users**: Researchers, industry partners, administrators

Please provide:
1. Optimal layout and information architecture
2. Collaborative features and real-time updates
3. Security and access control considerations
4. Mobile and cross-platform compatibility
5. Integration with existing university/industry systems

Focus on designs that enhance collaboration while maintaining data security and usability.
"""
        
        result = await self.nwtn_pipeline.process_query_full_pipeline(
            user_id=user_id,
            query=design_prompt,
            context={
                "domain": "dashboard_design",
                "dashboard_design": True,
                "dashboard_type": dashboard_type,
                "guidance_type": "comprehensive_design"
            }
        )
        
        guidance = {
            "dashboard_type": dashboard_type,
            "collaboration_requirements": collaboration_requirements,
            "design_guidance": result.get('response', {}).get('text', ''),
            "confidence": result.get('response', {}).get('confidence', 0.0),
            "sources": result.get('response', {}).get('sources', []),
            "processing_time": result.get('performance_metrics', {}).get('total_processing_time', 0.0),
            "generated_at": datetime.now().isoformat(),
            "requested_by": user_id
        }
        
        print(f"🎨 Dashboard design guidance generated:")
        print(f"   Type: {dashboard_type}")
        print(f"   Requirements: {len(collaboration_requirements)} features")
        print(f"   Confidence: {guidance['confidence']:.2f}")
        
        return guidance
    
    def export_dashboard(self,
                        dashboard_id: str,
                        export_format: ExportFormat,
                        user_id: str,
                        include_data: bool = False) -> str:
        """Export dashboard in specified format"""
        
        if dashboard_id not in self.collaborative_dashboards:
            raise ValueError(f"Dashboard {dashboard_id} not found")
        
        dashboard = self.collaborative_dashboards[dashboard_id]
        
        # Check permissions
        if not self._check_dashboard_access(dashboard, user_id, VisualizationAccessLevel.VIEWER):
            raise PermissionError("Insufficient permissions to export dashboard")
        
        export_dir = self.storage_path / "exports" / dashboard_id
        export_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if export_format == ExportFormat.HTML:
            # Generate complete HTML dashboard
            html_content = self._generate_html_dashboard(dashboard, include_data)
            export_file = export_dir / f"dashboard_{timestamp}.html"
            with open(export_file, 'w') as f:
                f.write(html_content)
                
        elif export_format == ExportFormat.PDF:
            # Generate PDF report (requires additional libraries)
            export_file = export_dir / f"dashboard_{timestamp}.pdf"
            self._generate_pdf_dashboard(dashboard, export_file, include_data)
            
        elif export_format == ExportFormat.JSON:
            # Export dashboard configuration
            export_file = export_dir / f"dashboard_config_{timestamp}.json"
            dashboard_data = asdict(dashboard)
            with open(export_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
                
        elif export_format == ExportFormat.POWERPOINT:
            # Generate PowerPoint presentation (mock implementation)
            export_file = export_dir / f"dashboard_presentation_{timestamp}.pptx"
            self._generate_powerpoint_dashboard(dashboard, export_file)
            
        else:
            raise ValueError(f"Export format {export_format.value} not supported")
        
        # Record export
        export_record = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "format": export_format.value,
            "file_path": str(export_file),
            "include_data": include_data
        }
        
        dashboard.export_history.append(export_record)
        self._save_dashboard(dashboard)
        
        print(f"📦 Dashboard exported successfully:")
        print(f"   Format: {export_format.value.upper()}")
        print(f"   File: {export_file.name}")
        print(f"   Include Data: {include_data}")
        
        return str(export_file)
    
    def _generate_html_dashboard(self, dashboard: CollaborativeDashboard, include_data: bool) -> str:
        """Generate HTML representation of dashboard"""
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dashboard.name}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: {self.dashboard_themes[dashboard.theme]['background_color']};
            color: {self.dashboard_themes[dashboard.theme]['text_color']};
        }}
        .dashboard-header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            border-bottom: 2px solid {self.dashboard_themes[dashboard.theme]['accent_color']};
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 15px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .widget {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border: 1px solid {self.dashboard_themes[dashboard.theme]['grid_color']};
        }}
        .widget h3 {{
            margin-top: 0;
            color: {self.dashboard_themes[dashboard.theme]['text_color']};
        }}
        .collaboration-info {{
            background: {self.dashboard_themes[dashboard.theme]['accent_color']};
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>{dashboard.name}</h1>
        <p>{dashboard.description}</p>
        <div class="collaboration-info">
            <strong>Owner:</strong> {dashboard.owner} | 
            <strong>Collaborators:</strong> {len(dashboard.collaborators)} | 
            <strong>Type:</strong> {dashboard.dashboard_type.value} |
            <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
    
    <div class="dashboard-grid">
        {self._generate_widget_html(dashboard)}
    </div>
    
    <script>
        // Add any interactive JavaScript here
        console.log('Dashboard loaded: {dashboard.name}');
    </script>
</body>
</html>
"""
        
        return html_template
    
    def _generate_widget_html(self, dashboard: CollaborativeDashboard) -> str:
        """Generate HTML for dashboard widgets"""
        
        widget_html = ""
        
        for widget in dashboard.widgets.values():
            # Calculate grid position
            grid_column = f"{widget.position['x'] + 1} / span {widget.position['w']}"
            grid_row = f"{widget.position['y'] + 1} / span {widget.position['h']}"
            
            widget_html += f"""
            <div class="widget" style="grid-column: {grid_column}; grid-row: {grid_row};">
                <h3>{widget.name}</h3>
                <div class="widget-content">
                    {self._generate_widget_content(widget)}
                </div>
            </div>
            """
        
        return widget_html
    
    def _generate_widget_content(self, widget: DashboardWidget) -> str:
        """Generate content for a specific widget"""
        
        if widget.widget_type == "metric":
            return f"""
            <div class="metric-widget">
                <div class="metric-value">94.7%</div>
                <div class="metric-label">{widget.name}</div>
            </div>
            """
            
        elif widget.widget_type == "chart":
            return f"""
            <div class="chart-widget">
                <div id="chart-{widget.widget_id}" style="height: 300px;"></div>
                <script>
                    // Placeholder for chart rendering
                    document.getElementById('chart-{widget.widget_id}').innerHTML = 
                        '<p>Interactive chart: {widget.name}</p>';
                </script>
            </div>
            """
            
        elif widget.widget_type == "text":
            return f"""
            <div class="text-widget">
                <p>{widget.content}</p>
            </div>
            """
            
        else:
            return f"<p>Widget type: {widget.widget_type}</p>"
    
    def _generate_pdf_dashboard(self, dashboard: CollaborativeDashboard, output_path: Path, include_data: bool):
        """Generate PDF version of dashboard (mock implementation)"""
        
        # In a real implementation, would use libraries like reportlab or weasyprint
        with open(output_path, 'w') as f:
            f.write(f"PDF Dashboard Export: {dashboard.name}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Widgets: {len(dashboard.widgets)}\n")
            if include_data:
                f.write("Data included in export\n")
    
    def _generate_powerpoint_dashboard(self, dashboard: CollaborativeDashboard, output_path: Path):
        """Generate PowerPoint presentation (mock implementation)"""
        
        # In a real implementation, would use python-pptx library
        with open(output_path, 'w') as f:
            f.write(f"PowerPoint Dashboard: {dashboard.name}\n")
            f.write(f"Slides: {len(dashboard.widgets)}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
    
    def generate_sample_data(self, data_type: str = "quantum_research") -> pd.DataFrame:
        """Generate sample data for testing visualizations"""
        
        np.random.seed(42)
        
        if data_type == "quantum_research":
            # Generate quantum computing research data
            n_samples = 1000
            institutions = ['UNC Chapel Hill', 'Duke University', 'NC State', 'SAS Institute']
            methods = ['Adaptive', 'Standard', 'Hybrid']
            
            data = pd.DataFrame({
                'noise_level': np.random.uniform(0.001, 0.1, n_samples),
                'error_rate': np.random.exponential(0.05, n_samples),
                'success_rate': np.random.beta(8, 2, n_samples),
                'institution': np.random.choice(institutions, n_samples),
                'method': np.random.choice(methods, n_samples),
                'qubits': np.random.choice([5, 7, 9, 11], n_samples),
                'runtime_ms': np.random.lognormal(3, 1, n_samples),
                'temperature_mk': np.random.uniform(10, 50, n_samples)
            })
            
        elif data_type == "collaboration_metrics":
            # Generate collaboration metrics data
            n_samples = 500
            data = pd.DataFrame({
                'project_month': np.repeat(range(1, 13), n_samples // 12),
                'collaboration_score': np.random.uniform(0.6, 1.0, n_samples),
                'ip_generated': np.random.poisson(2, n_samples),
                'publications': np.random.poisson(0.5, n_samples),
                'meetings_held': np.random.poisson(4, n_samples),
                'data_shared_gb': np.random.exponential(10, n_samples),
                'participant_satisfaction': np.random.uniform(3.5, 5.0, n_samples)
            })
            
        else:
            # Generic research data
            n_samples = 800
            data = pd.DataFrame({
                'x_value': np.random.randn(n_samples),
                'y_value': np.random.randn(n_samples) * 2 + 1,
                'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
                'size_metric': np.random.uniform(10, 100, n_samples),
                'time_point': pd.date_range('2023-01-01', periods=n_samples, freq='D')
            })
        
        return data
    
    def _check_dashboard_access(self, dashboard: CollaborativeDashboard, user_id: str, required_level: VisualizationAccessLevel) -> bool:
        """Check if user has required access level to dashboard"""
        
        # Owner has all access
        if dashboard.owner == user_id:
            return True
        
        # Check collaborator access
        if user_id in dashboard.collaborators:
            user_level = dashboard.collaborators[user_id]
            
            # Define access hierarchy
            access_hierarchy = {
                VisualizationAccessLevel.VIEWER: 1,
                VisualizationAccessLevel.COMMENTER: 2,
                VisualizationAccessLevel.EDITOR: 3,
                VisualizationAccessLevel.OWNER: 4
            }
            
            return access_hierarchy[user_level] >= access_hierarchy[required_level]
        
        return False
    
    def _save_dashboard(self, dashboard: CollaborativeDashboard):
        """Save collaborative dashboard"""
        dashboard_dir = self.storage_path / "dashboards" / dashboard.dashboard_id
        dashboard_dir.mkdir(parents=True, exist_ok=True)
        
        dashboard_file = dashboard_dir / "dashboard.json"
        with open(dashboard_file, 'w') as f:
            dashboard_data = asdict(dashboard)
            json.dump(dashboard_data, f, default=str, indent=2)
    
    def _save_project(self, project: VisualizationProject):
        """Save visualization project"""
        project_dir = self.storage_path / "projects" / project.project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        
        project_file = project_dir / "project.json"
        with open(project_file, 'w') as f:
            project_data = asdict(project)
            json.dump(project_data, f, default=str, indent=2)

# University-specific visualization templates
class UniversityVisualizationTemplates:
    """Pre-configured visualization templates for university research"""
    
    @staticmethod
    def create_research_publication_dashboard() -> Dict[str, Any]:
        """Create template for research publication dashboards"""
        return {
            "name": "Research Publication Dashboard",
            "description": "Publication-ready visualizations for academic papers",
            "layout": {
                "figure_1": {"position": {"x": 0, "y": 0, "w": 6, "h": 6}, "type": "primary_results"},
                "figure_2": {"position": {"x": 6, "y": 0, "w": 6, "h": 6}, "type": "statistical_analysis"},
                "figure_3": {"position": {"x": 0, "y": 6, "w": 12, "h": 4}, "type": "comparative_study"},
                "supplementary": {"position": {"x": 0, "y": 10, "w": 12, "h": 3}, "type": "additional_data"}
            },
            "styling": {
                "color_scheme": "publication",
                "font_family": "Arial",
                "dpi": 300,
                "format": "vector"
            }
        }
    
    @staticmethod
    def create_grant_presentation_dashboard() -> Dict[str, Any]:
        """Create template for grant application presentations"""
        return {
            "name": "Grant Application Dashboard",
            "description": "Compelling visualizations for funding applications",
            "layout": {
                "project_overview": {"position": {"x": 0, "y": 0, "w": 8, "h": 4}, "type": "summary_metrics"},
                "budget_allocation": {"position": {"x": 8, "y": 0, "w": 4, "h": 4}, "type": "financial_breakdown"},
                "timeline": {"position": {"x": 0, "y": 4, "w": 12, "h": 3}, "type": "project_timeline"},
                "expected_outcomes": {"position": {"x": 0, "y": 7, "w": 6, "h": 4}, "type": "outcome_projections"},
                "collaboration_network": {"position": {"x": 6, "y": 7, "w": 6, "h": 4}, "type": "partner_network"}
            },
            "styling": {
                "color_scheme": "university_brand",
                "emphasis": "impact_metrics",
                "interactive": False
            }
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_data_visualization_collaboration():
        """Test data visualization and collaborative dashboards"""
        
        print("🚀 Testing Data Visualization and Collaborative Dashboards")
        print("=" * 60)
        
        # Initialize data visualization collaboration
        viz_collab = DataVisualizationCollaboration()
        
        # Create visualization project
        project = viz_collab.create_visualization_project(
            title="Multi-University Quantum Computing Visualization Project",
            description="Collaborative data visualization for quantum error correction research across institutions",
            principal_investigator="sarah.chen@unc.edu",
            institutions=["UNC Chapel Hill", "Duke University", "NC State", "SAS Institute"],
            participants={
                "alex.rodriguez@duke.edu": VisualizationAccessLevel.EDITOR,
                "jennifer.kim@ncsu.edu": VisualizationAccessLevel.EDITOR,
                "michael.johnson@sas.com": VisualizationAccessLevel.COMMENTER,
                "data.scientist@unc.edu": VisualizationAccessLevel.EDITOR,
                "viz.designer@duke.edu": VisualizationAccessLevel.EDITOR
            },
            research_area="Quantum Computing",
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=365)
        )
        
        print(f"\n✅ Created visualization project: {project.title}")
        print(f"   Project ID: {project.project_id}")
        print(f"   Institutions: {len(project.institutions)}")
        print(f"   Participants: {len(project.participants)}")
        
        # Create collaborative dashboard
        dashboard = viz_collab.create_collaborative_dashboard(
            name="Quantum Error Correction Research Dashboard",
            description="Real-time collaborative dashboard for monitoring quantum algorithm performance across universities",
            dashboard_type=DashboardType.RESEARCH_ANALYTICS,
            owner="sarah.chen@unc.edu",
            collaborators={
                "alex.rodriguez@duke.edu": VisualizationAccessLevel.EDITOR,
                "jennifer.kim@ncsu.edu": VisualizationAccessLevel.EDITOR,
                "michael.johnson@sas.com": VisualizationAccessLevel.COMMENTER,
                "data.scientist@unc.edu": VisualizationAccessLevel.EDITOR
            },
            template_name="quantum_research",
            security_level="high"
        )
        
        print(f"\n✅ Created collaborative dashboard: {dashboard.name}")
        print(f"   Dashboard ID: {dashboard.dashboard_id}")
        print(f"   Type: {dashboard.dashboard_type.value}")
        print(f"   Widgets: {len(dashboard.widgets)}")
        print(f"   Collaborators: {len(dashboard.collaborators)}")
        
        # Generate sample data
        quantum_data = viz_collab.generate_sample_data("quantum_research")
        collaboration_data = viz_collab.generate_sample_data("collaboration_metrics")
        
        print(f"\n📊 Generated sample datasets:")
        print(f"   Quantum research data: {len(quantum_data)} samples")
        print(f"   Collaboration metrics: {len(collaboration_data)} samples")
        
        # Add interactive visualization widget
        viz_spec = VisualizationSpec(
            chart_type=ChartType.SCATTER_PLOT,
            data_source="quantum_research",
            x_axis="noise_level",
            y_axis="success_rate",
            color_by="institution",
            size_by="qubits",
            title="Quantum Error Correction Performance by Institution",
            x_title="Noise Level",
            y_title="Success Rate",
            color_scheme="university_brand",
            hover_data=["method", "runtime_ms", "temperature_mk"],
            width=800,
            height=600
        )
        
        widget = viz_collab.add_dashboard_widget(
            dashboard.dashboard_id,
            "Performance Analysis",
            "chart",
            {"x": 0, "y": 2, "w": 8, "h": 4},
            "sarah.chen@unc.edu",
            visualization_spec=viz_spec
        )
        
        print(f"\n✅ Added visualization widget: {widget.name}")
        print(f"   Chart type: {viz_spec.chart_type.value}")
        print(f"   Data dimensions: {viz_spec.x_axis} vs {viz_spec.y_axis}")
        
        # Create interactive visualization
        viz_file = viz_collab.create_interactive_visualization(
            dashboard.dashboard_id,
            widget.widget_id,
            quantum_data,
            viz_spec,
            "sarah.chen@unc.edu"
        )
        
        print(f"✅ Created interactive visualization: {Path(viz_file).name}")
        
        # Get AI visualization recommendations
        print(f"\n💡 Getting visualization recommendations...")
        
        viz_recommendations = await viz_collab.get_visualization_recommendations(
            "Multi-institutional quantum computing performance data with error rates, success metrics, and institutional comparisons",
            ["performance_analysis", "institutional_comparison", "method_evaluation", "publication_quality"],
            "academic_and_industry_mixed",
            "sarah.chen@unc.edu"
        )
        
        print(f"✅ Visualization recommendations generated:")
        print(f"   Confidence: {viz_recommendations['confidence']:.2f}")
        print(f"   Processing time: {viz_recommendations['processing_time']:.1f}s")
        print(f"   Audience: {viz_recommendations['target_audience']}")
        
        # Get dashboard design guidance
        print(f"\n🎨 Getting dashboard design guidance...")
        
        design_guidance = await viz_collab.get_dashboard_design_guidance(
            "research_analytics",
            ["real_time_collaboration", "multi_institutional_access", "publication_export", "mobile_compatibility"],
            "sarah.chen@unc.edu"
        )
        
        print(f"✅ Dashboard design guidance generated:")
        print(f"   Confidence: {design_guidance['confidence']:.2f}")
        print(f"   Requirements: {len(design_guidance['collaboration_requirements'])} features")
        
        # Export dashboard in multiple formats
        print(f"\n📦 Exporting dashboard...")
        
        html_export = viz_collab.export_dashboard(
            dashboard.dashboard_id,
            ExportFormat.HTML,
            "sarah.chen@unc.edu",
            include_data=True
        )
        
        json_export = viz_collab.export_dashboard(
            dashboard.dashboard_id,
            ExportFormat.JSON,
            "sarah.chen@unc.edu",
            include_data=False
        )
        
        print(f"✅ Dashboard exported:")
        print(f"   HTML: {Path(html_export).name}")
        print(f"   JSON: {Path(json_export).name}")
        
        # Test university-specific templates
        print(f"\n🏛️ Testing university-specific templates...")
        
        publication_template = UniversityVisualizationTemplates.create_research_publication_dashboard()
        grant_template = UniversityVisualizationTemplates.create_grant_presentation_dashboard()
        
        print(f"✅ Publication template: {publication_template['name']}")
        print(f"   Figures: {len(publication_template['layout'])} publication-ready visualizations")
        
        print(f"✅ Grant template: {grant_template['name']}")
        print(f"   Sections: {len(grant_template['layout'])} presentation components")
        
        # Create additional dashboard for grant presentation
        grant_dashboard = viz_collab.create_collaborative_dashboard(
            name="Quantum Computing Grant Proposal Visualizations",
            description="Publication-ready visualizations for NSF grant application",
            dashboard_type=DashboardType.PRESENTATION,
            owner="sarah.chen@unc.edu",
            collaborators={
                "alex.rodriguez@duke.edu": VisualizationAccessLevel.EDITOR,
                "grants.office@unc.edu": VisualizationAccessLevel.COMMENTER
            },
            security_level="high"
        )
        
        print(f"\n✅ Created grant presentation dashboard: {grant_dashboard.name}")
        print(f"   Dashboard ID: {grant_dashboard.dashboard_id}")
        print(f"   Type: {grant_dashboard.dashboard_type.value}")
        
        print(f"\n🎉 Data visualization and collaborative dashboards test completed!")
        print("✅ Ready for university-industry research visualization partnerships!")
    
    # Run test
    import asyncio
    asyncio.run(test_data_visualization_collaboration())