# PRSM Plugin Development Tutorial

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Environment Setup](#development-environment-setup)
3. [Creating Your First Plugin](#creating-your-first-plugin)
4. [Plugin Architecture](#plugin-architecture)
5. [Security and Validation](#security-and-validation)
6. [Testing and Debugging](#testing-and-debugging)
7. [Publishing to Marketplace](#publishing-to-marketplace)
8. [Advanced Topics](#advanced-topics)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Getting Started

### What is a PRSM Plugin?

A PRSM plugin is a modular extension that adds functionality to the PRSM platform. Plugins can:

- Provide specialized AI models or algorithms
- Add new data processing capabilities
- Integrate with external services and APIs
- Create custom user interfaces and dashboards
- Extend the marketplace ecosystem

### Plugin Types

PRSM supports several plugin types:

| Type | Description | Examples |
|------|-------------|----------|
| **AI Model** | Custom AI models and algorithms | Fine-tuned language models, specialized classifiers |
| **Data Processor** | Data transformation and analysis | ETL pipelines, data validators, formatters |
| **Integration** | External service connectors | CRM integrations, API wrappers, webhooks |
| **Analytics** | Visualization and reporting | Custom dashboards, charts, reports |
| **Utility** | Helper tools and utilities | Validators, converters, optimizers |

### Prerequisites

- Python 3.11 or higher
- Basic understanding of async/await programming
- Familiarity with PRSM architecture
- Development environment with PRSM SDK

## Development Environment Setup

### 1. Install PRSM SDK

```bash
pip install prsm-sdk
```

### 2. Set Up Development Environment

```bash
# Create plugin development directory
mkdir my-prsm-plugins
cd my-prsm-plugins

# Create virtual environment
python -m venv plugin-dev-env
source plugin-dev-env/bin/activate  # On Windows: plugin-dev-env\Scripts\activate

# Install development dependencies
pip install prsm-sdk pytest pytest-asyncio black isort mypy
```

### 3. Initialize Plugin Project

```bash
# Create plugin structure
mkdir analytics-dashboard-plugin
cd analytics-dashboard-plugin

# Create directory structure
mkdir src tests docs examples
touch README.md setup.py requirements.txt
```

### 4. Configure Development Tools

Create `pyproject.toml`:

```toml
[tool.black]
line-length = 88
target-version = ['py311']

[tool.isort]
profile = "black"

[tool.mypy]
python_version = "3.11"
strict = true
```

## Creating Your First Plugin

### Step 1: Define Plugin Manifest

Create `plugin_manifest.json`:

```json
{
  "name": "Analytics Dashboard Plugin",
  "version": "1.0.0",
  "description": "Custom analytics dashboard with real-time metrics",
  "author": "Your Name",
  "email": "your.email@example.com",
  "website": "https://your-website.com",
  "entry_point": "src.analytics_dashboard:AnalyticsDashboard",
  "capabilities": [
    "data_visualization",
    "real_time_analytics", 
    "custom_dashboards",
    "export_functionality"
  ],
  "permissions": [
    "read_analytics_data",
    "write_dashboard_config",
    "access_user_metrics"
  ],
  "dependencies": [
    "pandas>=1.5.0",
    "plotly>=5.0.0",
    "dash>=2.14.0"
  ],
  "resource_requirements": {
    "memory": "256MB",
    "cpu": "0.25 cores",
    "storage": "50MB"
  },
  "security_level": "sandbox",
  "category": "analytics",
  "tags": ["dashboard", "visualization", "analytics"],
  "min_prsm_version": "1.0.0",
  "documentation_url": "https://docs.example.com/analytics-dashboard",
  "support_url": "https://support.example.com"
}
```

### Step 2: Implement Plugin Base Class

Create `src/analytics_dashboard.py`:

```python
#!/usr/bin/env python3
"""
Analytics Dashboard Plugin for PRSM
==================================

A comprehensive analytics dashboard plugin that provides real-time metrics,
custom visualizations, and automated reporting capabilities.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, callback

from prsm_sdk.plugin_base import PRSMPlugin
from prsm_sdk.exceptions import PluginError, ValidationError
from prsm_sdk.types import PluginResponse, PluginConfig


class AnalyticsDashboard(PRSMPlugin):
    """
    Analytics Dashboard Plugin
    
    Provides comprehensive analytics capabilities including:
    - Real-time metric dashboards
    - Custom visualization creation
    - Automated report generation
    - Data export functionality
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Analytics Dashboard Plugin"
        self.version = "1.0.0"
        self.logger = logging.getLogger(__name__)
        self.dash_app = None
        self.data_cache = {}
        self.dashboard_configs = {}
        
    async def initialize(self, config: PluginConfig) -> bool:
        """
        Initialize the analytics dashboard plugin
        
        Args:
            config: Plugin configuration from PRSM
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Analytics Dashboard Plugin...")
            
            # Validate configuration
            await self._validate_config(config)
            
            # Initialize Dash application
            self.dash_app = Dash(__name__, suppress_callback_exceptions=True)
            self._setup_dashboard_layout()
            self._register_callbacks()
            
            # Initialize data cache
            self.data_cache = {
                'metrics': {},
                'last_updated': None,
                'refresh_interval': config.get('refresh_interval', 300)  # 5 minutes
            }
            
            # Load existing dashboard configurations
            await self._load_dashboard_configs()
            
            self.logger.info("Analytics Dashboard Plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize plugin: {str(e)}")
            return False
    
    async def _validate_config(self, config: PluginConfig):
        """Validate plugin configuration"""
        required_permissions = ['read_analytics_data', 'write_dashboard_config']
        
        for permission in required_permissions:
            if not config.has_permission(permission):
                raise ValidationError(f"Missing required permission: {permission}")
        
        # Validate resource limits
        if config.get_memory_limit() < 256 * 1024 * 1024:  # 256MB
            raise ValidationError("Plugin requires at least 256MB memory")
    
    def _setup_dashboard_layout(self):
        """Set up the main dashboard layout"""
        self.dash_app.layout = html.Div([
            # Header
            html.H1("PRSM Analytics Dashboard", 
                   style={'text-align': 'center', 'margin-bottom': '30px'}),
            
            # Control Panel
            html.Div([
                html.H3("Dashboard Controls"),
                dcc.Dropdown(
                    id='dashboard-selector',
                    options=[
                        {'label': 'System Metrics', 'value': 'system'},
                        {'label': 'User Analytics', 'value': 'users'},
                        {'label': 'Query Performance', 'value': 'queries'},
                        {'label': 'Revenue Metrics', 'value': 'revenue'}
                    ],
                    value='system',
                    style={'margin-bottom': '20px'}
                ),
                dcc.DatePickerRange(
                    id='date-range-picker',
                    start_date=datetime.now() - timedelta(days=7),
                    end_date=datetime.now(),
                    style={'margin-bottom': '20px'}
                ),
                html.Button('Refresh Data', id='refresh-button', 
                           style={'margin-bottom': '20px'})
            ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
            
            # Main Dashboard Area
            html.Div([
                dcc.Graph(id='main-chart'),
                html.Div(id='metrics-cards'),
                dcc.Graph(id='secondary-chart')
            ], style={'width': '70%', 'display': 'inline-block'}),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=5*60*1000,  # 5 minutes in milliseconds
                n_intervals=0
            )
        ])
    
    def _register_callbacks(self):
        """Register Dash callbacks for interactivity"""
        
        @self.dash_app.callback(
            [Output('main-chart', 'figure'),
             Output('metrics-cards', 'children'),
             Output('secondary-chart', 'figure')],
            [Input('dashboard-selector', 'value'),
             Input('date-range-picker', 'start_date'),
             Input('date-range-picker', 'end_date'),
             Input('refresh-button', 'n_clicks'),
             Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(dashboard_type, start_date, end_date, refresh_clicks, intervals):
            """Update dashboard based on user selections"""
            
            # Get data for selected dashboard type
            data = self._get_dashboard_data(dashboard_type, start_date, end_date)
            
            # Create main chart
            main_fig = self._create_main_chart(dashboard_type, data)
            
            # Create metrics cards
            metrics_cards = self._create_metrics_cards(dashboard_type, data)
            
            # Create secondary chart
            secondary_fig = self._create_secondary_chart(dashboard_type, data)
            
            return main_fig, metrics_cards, secondary_fig
    
    def _get_dashboard_data(self, dashboard_type: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get data for specific dashboard type"""
        
        # This would typically fetch real data from PRSM analytics APIs
        # For this example, we'll generate sample data
        
        if dashboard_type == 'system':
            return self._generate_system_metrics_data(start_date, end_date)
        elif dashboard_type == 'users':
            return self._generate_user_analytics_data(start_date, end_date)
        elif dashboard_type == 'queries':
            return self._generate_query_performance_data(start_date, end_date)
        elif dashboard_type == 'revenue':
            return self._generate_revenue_metrics_data(start_date, end_date)
        else:
            return {}
    
    def _generate_system_metrics_data(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Generate sample system metrics data"""
        dates = pd.date_range(start=start_date, end=end_date, freq='H')
        
        return {
            'timestamps': dates.tolist(),
            'cpu_usage': [50 + 20 * np.sin(i * 0.1) + np.random.normal(0, 5) for i in range(len(dates))],
            'memory_usage': [60 + 15 * np.cos(i * 0.1) + np.random.normal(0, 3) for i in range(len(dates))],
            'response_time': [200 + 50 * np.sin(i * 0.2) + np.random.normal(0, 10) for i in range(len(dates))],
            'active_connections': [100 + 50 * np.cos(i * 0.15) + np.random.normal(0, 8) for i in range(len(dates))]
        }
    
    def _create_main_chart(self, dashboard_type: str, data: Dict[str, Any]) -> go.Figure:
        """Create the main chart for the dashboard"""
        
        if dashboard_type == 'system':
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['timestamps'],
                y=data['cpu_usage'],
                mode='lines',
                name='CPU Usage (%)',
                line=dict(color='#1f77b4')
            ))
            fig.add_trace(go.Scatter(
                x=data['timestamps'],
                y=data['memory_usage'],
                mode='lines',
                name='Memory Usage (%)',
                line=dict(color='#ff7f0e')
            ))
            
            fig.update_layout(
                title='System Resource Usage Over Time',
                xaxis_title='Time',
                yaxis_title='Usage (%)',
                hovermode='x unified'
            )
            
        elif dashboard_type == 'queries':
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['timestamps'],
                y=data['response_time'],
                mode='lines+markers',
                name='Response Time (ms)',
                line=dict(color='#2ca02c')
            ))
            
            fig.update_layout(
                title='Query Response Time',
                xaxis_title='Time',
                yaxis_title='Response Time (ms)'
            )
            
        else:
            # Default empty chart
            fig = go.Figure()
            fig.update_layout(title=f'{dashboard_type.title()} Dashboard')
        
        return fig
    
    def _create_metrics_cards(self, dashboard_type: str, data: Dict[str, Any]) -> List[html.Div]:
        """Create metrics cards for the dashboard"""
        
        if dashboard_type == 'system':
            cards = [
                html.Div([
                    html.H4("Average CPU Usage"),
                    html.H2(f"{np.mean(data['cpu_usage']):.1f}%")
                ], style={'width': '23%', 'display': 'inline-block', 'text-align': 'center',
                         'border': '1px solid #ddd', 'margin': '1%', 'padding': '10px'}),
                
                html.Div([
                    html.H4("Average Memory Usage"),
                    html.H2(f"{np.mean(data['memory_usage']):.1f}%")
                ], style={'width': '23%', 'display': 'inline-block', 'text-align': 'center',
                         'border': '1px solid #ddd', 'margin': '1%', 'padding': '10px'}),
                
                html.Div([
                    html.H4("Average Response Time"),
                    html.H2(f"{np.mean(data['response_time']):.0f}ms")
                ], style={'width': '23%', 'display': 'inline-block', 'text-align': 'center',
                         'border': '1px solid #ddd', 'margin': '1%', 'padding': '10px'}),
                
                html.Div([
                    html.H4("Active Connections"),
                    html.H2(f"{int(np.mean(data['active_connections']))}")
                ], style={'width': '23%', 'display': 'inline-block', 'text-align': 'center',
                         'border': '1px solid #ddd', 'margin': '1%', 'padding': '10px'})
            ]
        else:
            cards = [html.Div(f"Metrics for {dashboard_type} dashboard")]
        
        return cards
    
    def _create_secondary_chart(self, dashboard_type: str, data: Dict[str, Any]) -> go.Figure:
        """Create secondary chart for the dashboard"""
        
        if dashboard_type == 'system':
            # Create a distribution chart
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=data['response_time'],
                name='Response Time Distribution',
                nbinsx=20
            ))
            
            fig.update_layout(
                title='Response Time Distribution',
                xaxis_title='Response Time (ms)',
                yaxis_title='Frequency'
            )
        else:
            fig = go.Figure()
            fig.update_layout(title=f'{dashboard_type.title()} Secondary Chart')
        
        return fig
    
    async def execute_task(self, task: Dict[str, Any]) -> PluginResponse:
        """
        Execute a plugin task
        
        Args:
            task: Task configuration and parameters
            
        Returns:
            PluginResponse with task results
        """
        try:
            task_type = task.get('type', 'unknown')
            
            if task_type == 'create_dashboard':
                return await self._create_custom_dashboard(task)
            elif task_type == 'generate_report':
                return await self._generate_report(task)
            elif task_type == 'export_data':
                return await self._export_data(task)
            elif task_type == 'get_metrics':
                return await self._get_metrics(task)
            else:
                raise PluginError(f"Unknown task type: {task_type}")
                
        except Exception as e:
            self.logger.error(f"Task execution failed: {str(e)}")
            return PluginResponse(
                success=False,
                error=str(e),
                data=None
            )
    
    async def _create_custom_dashboard(self, task: Dict[str, Any]) -> PluginResponse:
        """Create a custom dashboard configuration"""
        
        dashboard_config = task.get('config', {})
        dashboard_name = dashboard_config.get('name', 'Custom Dashboard')
        
        # Validate dashboard configuration
        if not self._validate_dashboard_config(dashboard_config):
            return PluginResponse(
                success=False,
                error="Invalid dashboard configuration",
                data=None
            )
        
        # Generate unique dashboard ID
        dashboard_id = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store dashboard configuration
        self.dashboard_configs[dashboard_id] = {
            'name': dashboard_name,
            'config': dashboard_config,
            'created_at': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat()
        }
        
        return PluginResponse(
            success=True,
            data={
                'dashboard_id': dashboard_id,
                'dashboard_name': dashboard_name,
                'url': f"/dashboard/{dashboard_id}"
            }
        )
    
    async def _generate_report(self, task: Dict[str, Any]) -> PluginResponse:
        """Generate an analytics report"""
        
        report_config = task.get('config', {})
        report_type = report_config.get('type', 'summary')
        time_range = report_config.get('time_range', '7d')
        
        # Generate report data
        report_data = await self._compile_report_data(report_type, time_range)
        
        # Format report
        report = {
            'title': f'Analytics Report - {report_type.title()}',
            'generated_at': datetime.now().isoformat(),
            'time_range': time_range,
            'summary': report_data.get('summary', {}),
            'charts': report_data.get('charts', []),
            'recommendations': report_data.get('recommendations', [])
        }
        
        return PluginResponse(
            success=True,
            data=report
        )
    
    async def _export_data(self, task: Dict[str, Any]) -> PluginResponse:
        """Export dashboard data in various formats"""
        
        export_config = task.get('config', {})
        export_format = export_config.get('format', 'csv')
        data_type = export_config.get('data_type', 'metrics')
        
        # Get data to export
        export_data = await self._get_export_data(data_type, export_config)
        
        # Format data based on requested format
        if export_format == 'csv':
            formatted_data = self._format_as_csv(export_data)
        elif export_format == 'json':
            formatted_data = json.dumps(export_data, indent=2)
        elif export_format == 'excel':
            formatted_data = self._format_as_excel(export_data)
        else:
            return PluginResponse(
                success=False,
                error=f"Unsupported export format: {export_format}",
                data=None
            )
        
        return PluginResponse(
            success=True,
            data={
                'format': export_format,
                'data': formatted_data,
                'filename': f"analytics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}"
            }
        )
    
    async def _get_metrics(self, task: Dict[str, Any]) -> PluginResponse:
        """Get current metrics data"""
        
        metrics_config = task.get('config', {})
        metric_types = metrics_config.get('types', ['system', 'users', 'queries'])
        
        metrics = {}
        for metric_type in metric_types:
            metrics[metric_type] = await self._fetch_metrics(metric_type)
        
        return PluginResponse(
            success=True,
            data={
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def _validate_dashboard_config(self, config: Dict[str, Any]) -> bool:
        """Validate dashboard configuration"""
        required_fields = ['name', 'widgets']
        
        for field in required_fields:
            if field not in config:
                return False
        
        # Validate widgets
        widgets = config.get('widgets', [])
        for widget in widgets:
            if 'type' not in widget or 'data_source' not in widget:
                return False
        
        return True
    
    async def _compile_report_data(self, report_type: str, time_range: str) -> Dict[str, Any]:
        """Compile data for report generation"""
        
        # This would typically fetch real data from PRSM APIs
        # For this example, we'll return sample data
        
        return {
            'summary': {
                'total_users': 1247,
                'total_queries': 15634,
                'success_rate': 0.96,
                'average_response_time': 18.4
            },
            'charts': [
                {
                    'type': 'line_chart',
                    'title': 'Query Volume Over Time',
                    'data': 'query_volume_data_here'
                },
                {
                    'type': 'pie_chart',
                    'title': 'Query Types Distribution',
                    'data': 'query_types_data_here'
                }
            ],
            'recommendations': [
                'Consider optimizing the 4% of failed queries',
                'Response time is within acceptable range',
                'User growth is trending positively'
            ]
        }
    
    async def _get_export_data(self, data_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for export"""
        
        # Fetch data based on type
        if data_type == 'metrics':
            return await self._fetch_metrics('all')
        elif data_type == 'users':
            return await self._fetch_user_data(config)
        elif data_type == 'queries':
            return await self._fetch_query_data(config)
        else:
            return {}
    
    def _format_as_csv(self, data: Dict[str, Any]) -> str:
        """Format data as CSV string"""
        if isinstance(data, dict) and 'timestamps' in data:
            df = pd.DataFrame(data)
            return df.to_csv(index=False)
        else:
            # Convert dict to CSV format
            return json.dumps(data)  # Simplified for example
    
    def _format_as_excel(self, data: Dict[str, Any]) -> bytes:
        """Format data as Excel bytes"""
        # This would use pandas or openpyxl to create Excel format
        # For this example, we'll return a placeholder
        return b"Excel data would go here"
    
    async def _fetch_metrics(self, metric_type: str) -> Dict[str, Any]:
        """Fetch metrics from PRSM system"""
        # This would make actual API calls to PRSM
        # For this example, return sample data
        
        return {
            'cpu_usage': 67.5,
            'memory_usage': 71.2,
            'active_users': 1247,
            'queries_per_minute': 45.3,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _load_dashboard_configs(self):
        """Load existing dashboard configurations"""
        # This would typically load from persistent storage
        self.dashboard_configs = {}
    
    async def shutdown(self):
        """Cleanup plugin resources"""
        try:
            if self.dash_app:
                # Stop Dash server if running
                pass
            
            # Clear caches
            self.data_cache.clear()
            self.dashboard_configs.clear()
            
            self.logger.info("Analytics Dashboard Plugin shut down successfully")
            
        except Exception as e:
            self.logger.error(f"Error during plugin shutdown: {str(e)}")


# Required: Plugin factory function
def create_plugin() -> AnalyticsDashboard:
    """Factory function to create plugin instance"""
    return AnalyticsDashboard()


# Required: Plugin metadata
PLUGIN_METADATA = {
    "name": "Analytics Dashboard Plugin",
    "version": "1.0.0",
    "description": "Comprehensive analytics dashboard with real-time metrics",
    "author": "Your Name",
    "capabilities": ["data_visualization", "real_time_analytics", "custom_dashboards"],
    "permissions": ["read_analytics_data", "write_dashboard_config"]
}
```

### Step 3: Create Requirements File

Create `requirements.txt`:

```
pandas>=1.5.0
plotly>=5.0.0
dash>=2.14.0
numpy>=1.24.0
prsm-sdk>=1.0.0
```

### Step 4: Create Setup Configuration

Create `setup.py`:

```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="analytics-dashboard-plugin",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Analytics Dashboard Plugin for PRSM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/analytics-dashboard-plugin",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "prsm.plugins": [
            "analytics_dashboard = src.analytics_dashboard:create_plugin",
        ],
    },
)
```

## Plugin Architecture

### Base Plugin Class

All PRSM plugins inherit from `PRSMPlugin`:

```python
from prsm_sdk.plugin_base import PRSMPlugin
from prsm_sdk.types import PluginConfig, PluginResponse

class MyPlugin(PRSMPlugin):
    async def initialize(self, config: PluginConfig) -> bool:
        """Initialize plugin with configuration"""
        pass
    
    async def execute_task(self, task: Dict[str, Any]) -> PluginResponse:
        """Execute plugin-specific task"""
        pass
    
    async def shutdown(self):
        """Cleanup plugin resources"""
        pass
```

### Plugin Lifecycle

1. **Registration**: Plugin is registered in marketplace
2. **Installation**: User installs plugin in their PRSM instance
3. **Initialization**: Plugin is initialized with configuration
4. **Execution**: Plugin executes tasks as requested
5. **Shutdown**: Plugin cleans up resources

### Configuration Management

Plugins receive configuration through `PluginConfig`:

```python
async def initialize(self, config: PluginConfig) -> bool:
    # Access configuration values
    refresh_interval = config.get('refresh_interval', 300)
    
    # Check permissions
    if not config.has_permission('read_analytics_data'):
        raise ValidationError("Missing required permission")
    
    # Get resource limits
    memory_limit = config.get_memory_limit()
    cpu_limit = config.get_cpu_limit()
```

## Security and Validation

### Security Levels

PRSM plugins operate at different security levels:

| Level | Description | Capabilities |
|-------|-------------|--------------|
| **TRUSTED** | Verified developers, full access | System APIs, file system, network |
| **SANDBOX** | Standard level, restricted access | Limited APIs, no file system |
| **ISOLATED** | High-risk plugins, minimal access | Basic APIs only |

### Security Best Practices

```python
class SecurePlugin(PRSMPlugin):
    async def initialize(self, config: PluginConfig) -> bool:
        # 1. Validate all inputs
        if not self._validate_config(config):
            return False
        
        # 2. Use parameterized queries for database access
        self.db_query = "SELECT * FROM table WHERE id = %s"
        
        # 3. Sanitize user inputs
        user_input = self._sanitize_input(raw_input)
        
        # 4. Handle errors gracefully
        try:
            result = await self._risky_operation()
        except Exception as e:
            self.logger.error(f"Operation failed: {str(e)}")
            return False
        
        return True
    
    def _validate_config(self, config: PluginConfig) -> bool:
        """Validate configuration parameters"""
        required_fields = ['api_key', 'endpoint_url']
        
        for field in required_fields:
            if not config.get(field):
                self.logger.error(f"Missing required field: {field}")
                return False
        
        return True
    
    def _sanitize_input(self, user_input: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        # Remove dangerous characters
        import re
        sanitized = re.sub(r'[<>"\'\&]', '', user_input)
        return sanitized.strip()
```

### Validation Rules

```python
class PluginValidator:
    """Validate plugin compliance"""
    
    def validate_manifest(self, manifest: Dict[str, Any]) -> List[str]:
        """Validate plugin manifest"""
        errors = []
        
        # Required fields
        required = ['name', 'version', 'entry_point', 'capabilities']
        for field in required:
            if field not in manifest:
                errors.append(f"Missing required field: {field}")
        
        # Version format
        if 'version' in manifest:
            if not re.match(r'^\d+\.\d+\.\d+$', manifest['version']):
                errors.append("Version must be in format X.Y.Z")
        
        # Capability validation
        if 'capabilities' in manifest:
            valid_capabilities = [
                'data_visualization', 'real_time_analytics', 
                'custom_dashboards', 'export_functionality'
            ]
            for cap in manifest['capabilities']:
                if cap not in valid_capabilities:
                    errors.append(f"Unknown capability: {cap}")
        
        return errors
```

## Testing and Debugging

### Unit Testing

Create `tests/test_analytics_dashboard.py`:

```python
import pytest
import asyncio
from unittest.mock import Mock, patch

from src.analytics_dashboard import AnalyticsDashboard
from prsm_sdk.types import PluginConfig, PluginResponse


class TestAnalyticsDashboard:
    
    @pytest.fixture
    def plugin(self):
        """Create plugin instance for testing"""
        return AnalyticsDashboard()
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        config = Mock(spec=PluginConfig)
        config.get.return_value = 300  # refresh_interval
        config.has_permission.return_value = True
        config.get_memory_limit.return_value = 256 * 1024 * 1024  # 256MB
        return config
    
    @pytest.mark.asyncio
    async def test_plugin_initialization(self, plugin, mock_config):
        """Test plugin initialization"""
        result = await plugin.initialize(mock_config)
        assert result is True
        assert plugin.dash_app is not None
        assert plugin.data_cache is not None
    
    @pytest.mark.asyncio
    async def test_create_dashboard_task(self, plugin, mock_config):
        """Test dashboard creation task"""
        await plugin.initialize(mock_config)
        
        task = {
            'type': 'create_dashboard',
            'config': {
                'name': 'Test Dashboard',
                'widgets': [
                    {'type': 'chart', 'data_source': 'metrics'}
                ]
            }
        }
        
        response = await plugin.execute_task(task)
        
        assert response.success is True
        assert 'dashboard_id' in response.data
        assert response.data['dashboard_name'] == 'Test Dashboard'
    
    @pytest.mark.asyncio
    async def test_generate_report_task(self, plugin, mock_config):
        """Test report generation task"""
        await plugin.initialize(mock_config)
        
        task = {
            'type': 'generate_report',
            'config': {
                'type': 'summary',
                'time_range': '7d'
            }
        }
        
        response = await plugin.execute_task(task)
        
        assert response.success is True
        assert 'title' in response.data
        assert 'summary' in response.data
    
    @pytest.mark.asyncio
    async def test_invalid_task_type(self, plugin, mock_config):
        """Test handling of invalid task type"""
        await plugin.initialize(mock_config)
        
        task = {'type': 'invalid_task'}
        response = await plugin.execute_task(task)
        
        assert response.success is False
        assert 'Unknown task type' in response.error
    
    def test_dashboard_config_validation(self, plugin):
        """Test dashboard configuration validation"""
        
        # Valid configuration
        valid_config = {
            'name': 'Test Dashboard',
            'widgets': [
                {'type': 'chart', 'data_source': 'metrics'}
            ]
        }
        assert plugin._validate_dashboard_config(valid_config) is True
        
        # Invalid configuration - missing name
        invalid_config = {
            'widgets': [
                {'type': 'chart', 'data_source': 'metrics'}
            ]
        }
        assert plugin._validate_dashboard_config(invalid_config) is False
        
        # Invalid configuration - missing widgets
        invalid_config2 = {
            'name': 'Test Dashboard'
        }
        assert plugin._validate_dashboard_config(invalid_config2) is False
```

### Integration Testing

Create `tests/test_integration.py`:

```python
import pytest
import asyncio
from unittest.mock import Mock, patch

from src.analytics_dashboard import AnalyticsDashboard


@pytest.mark.integration
class TestAnalyticsDashboardIntegration:
    
    @pytest.mark.asyncio
    async def test_full_dashboard_workflow(self):
        """Test complete dashboard creation and usage workflow"""
        
        plugin = AnalyticsDashboard()
        
        # Mock PRSM SDK components
        with patch('prsm_sdk.plugin_base.PRSMPlugin'):
            config = Mock()
            config.get.return_value = 300
            config.has_permission.return_value = True
            config.get_memory_limit.return_value = 256 * 1024 * 1024
            
            # Initialize plugin
            result = await plugin.initialize(config)
            assert result is True
            
            # Create dashboard
            create_task = {
                'type': 'create_dashboard',
                'config': {
                    'name': 'Integration Test Dashboard',
                    'widgets': [
                        {'type': 'chart', 'data_source': 'system_metrics'},
                        {'type': 'table', 'data_source': 'user_data'}
                    ]
                }
            }
            
            create_response = await plugin.execute_task(create_task)
            assert create_response.success is True
            
            dashboard_id = create_response.data['dashboard_id']
            
            # Generate report
            report_task = {
                'type': 'generate_report',
                'config': {
                    'type': 'dashboard_summary',
                    'dashboard_id': dashboard_id
                }
            }
            
            report_response = await plugin.execute_task(report_task)
            assert report_response.success is True
            
            # Export data
            export_task = {
                'type': 'export_data',
                'config': {
                    'format': 'csv',
                    'data_type': 'metrics'
                }
            }
            
            export_response = await plugin.execute_task(export_task)
            assert export_response.success is True
            assert export_response.data['format'] == 'csv'
            
            # Cleanup
            await plugin.shutdown()
```

### Debugging Tools

Create `debug/debug_runner.py`:

```python
#!/usr/bin/env python3
"""
Debug runner for Analytics Dashboard Plugin
"""

import asyncio
import logging
from unittest.mock import Mock

from src.analytics_dashboard import AnalyticsDashboard


async def debug_plugin():
    """Debug plugin functionality"""
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create plugin instance
    plugin = AnalyticsDashboard()
    
    # Create mock configuration
    config = Mock()
    config.get.return_value = 300
    config.has_permission.return_value = True
    config.get_memory_limit.return_value = 256 * 1024 * 1024
    
    print("🐛 Debugging Analytics Dashboard Plugin")
    
    try:
        # Test initialization
        print("1. Testing initialization...")
        result = await plugin.initialize(config)
        print(f"   Initialization result: {result}")
        
        # Test dashboard creation
        print("2. Testing dashboard creation...")
        task = {
            'type': 'create_dashboard',
            'config': {
                'name': 'Debug Dashboard',
                'widgets': [
                    {'type': 'chart', 'data_source': 'debug_metrics'}
                ]
            }
        }
        
        response = await plugin.execute_task(task)
        print(f"   Dashboard creation result: {response.success}")
        if response.success:
            print(f"   Dashboard ID: {response.data['dashboard_id']}")
        
        # Test metrics retrieval
        print("3. Testing metrics retrieval...")
        metrics_task = {
            'type': 'get_metrics',
            'config': {
                'types': ['system']
            }
        }
        
        metrics_response = await plugin.execute_task(metrics_task)
        print(f"   Metrics retrieval result: {metrics_response.success}")
        
        # Test shutdown
        print("4. Testing shutdown...")
        await plugin.shutdown()
        print("   Shutdown completed")
        
        print("✅ Debug session completed successfully")
        
    except Exception as e:
        print(f"❌ Debug session failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_plugin())
```

## Publishing to Marketplace

### Step 1: Package Plugin

```bash
# Create distribution package
python setup.py sdist bdist_wheel

# Verify package contents
tar -tzf dist/analytics-dashboard-plugin-1.0.0.tar.gz
```

### Step 2: Validate Plugin

```bash
# Run PRSM plugin validator
prsm-cli plugin validate analytics-dashboard-plugin/

# Run security scan
prsm-cli plugin security-scan analytics-dashboard-plugin/
```

### Step 3: Submit to Marketplace

```python
# marketplace_submission.py
import asyncio
from prsm_sdk.marketplace import MarketplaceClient

async def submit_plugin():
    client = MarketplaceClient(api_key="your_api_key")
    
    # Upload plugin package
    result = await client.upload_plugin(
        package_path="dist/analytics-dashboard-plugin-1.0.0.tar.gz",
        manifest_path="plugin_manifest.json"
    )
    
    print(f"Plugin submitted: {result.submission_id}")
    print(f"Status: {result.status}")
    print(f"Review URL: {result.review_url}")

if __name__ == "__main__":
    asyncio.run(submit_plugin())
```

### Step 4: Monitor Review Process

```bash
# Check submission status
prsm-cli marketplace status <submission_id>

# View reviewer feedback
prsm-cli marketplace feedback <submission_id>
```

## Advanced Topics

### Custom UI Components

```python
# Custom React components for advanced UI
def create_custom_component():
    return html.Div([
        dcc.Graph(
            id='custom-3d-plot',
            figure=create_3d_visualization()
        ),
        html.Div(id='custom-controls')
    ])

def create_3d_visualization():
    """Create custom 3D visualization"""
    import plotly.graph_objects as go
    
    fig = go.Figure(data=[go.Scatter3d(
        x=[1, 2, 3, 4],
        y=[10, 11, 12, 13],
        z=[2, 3, 4, 5],
        mode='markers+lines'
    )])
    
    return fig
```

### WebSocket Integration

```python
import websockets
import json

class WebSocketPlugin(PRSMPlugin):
    def __init__(self):
        super().__init__()
        self.websocket_server = None
        
    async def initialize(self, config):
        # Start WebSocket server for real-time updates
        self.websocket_server = await websockets.serve(
            self.handle_websocket, "localhost", 8765
        )
        return True
        
    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connections"""
        try:
            async for message in websocket:
                data = json.loads(message)
                response = await self.process_websocket_message(data)
                await websocket.send(json.dumps(response))
        except websockets.exceptions.ConnectionClosed:
            pass
```

### Machine Learning Integration

```python
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class MLAnalyticsPlugin(PRSMPlugin):
    def __init__(self):
        super().__init__()
        self.model = None
        
    async def initialize(self, config):
        # Load pre-trained model
        try:
            self.model = joblib.load('models/analytics_model.pkl')
        except FileNotFoundError:
            # Train new model if none exists
            self.model = await self._train_model()
            
        return True
        
    async def _train_model(self):
        """Train analytics prediction model"""
        # Get historical data
        X, y = await self._get_training_data()
        
        # Train model
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        
        # Save model
        joblib.dump(model, 'models/analytics_model.pkl')
        
        return model
        
    async def predict_metrics(self, features):
        """Predict future metrics"""
        if self.model is None:
            raise PluginError("Model not initialized")
            
        features_array = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features_array)
        
        return {
            'prediction': prediction[0],
            'confidence': self.model.score(features_array, [prediction[0]])
        }
```

## Best Practices

### Performance Optimization

```python
import asyncio
import functools
from typing import Dict, Any

class OptimizedPlugin(PRSMPlugin):
    def __init__(self):
        super().__init__()
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes
        
    @functools.lru_cache(maxsize=128)
    def expensive_computation(self, param: str) -> str:
        """Cache expensive computations"""
        # Expensive operation here
        return f"result_for_{param}"
        
    async def get_data_with_cache(self, key: str) -> Dict[str, Any]:
        """Get data with caching"""
        now = asyncio.get_event_loop().time()
        
        if key in self._cache:
            data, timestamp = self._cache[key]
            if now - timestamp < self._cache_timeout:
                return data
        
        # Fetch fresh data
        data = await self._fetch_data(key)
        self._cache[key] = (data, now)
        
        return data
        
    async def batch_process(self, items: List[Any]) -> List[Any]:
        """Process items in batches for better performance"""
        batch_size = 10
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = await asyncio.gather(*[
                self.process_item(item) for item in batch
            ])
            results.extend(batch_results)
            
        return results
```

### Error Handling

```python
import traceback
from enum import Enum

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RobustPlugin(PRSMPlugin):
    def __init__(self):
        super().__init__()
        self.error_count = 0
        self.max_errors = 10
        
    async def execute_task(self, task: Dict[str, Any]) -> PluginResponse:
        """Execute task with comprehensive error handling"""
        try:
            return await self._execute_task_safely(task)
            
        except ValidationError as e:
            return self._handle_validation_error(e)
        except TimeoutError as e:
            return self._handle_timeout_error(e)
        except ConnectionError as e:
            return self._handle_connection_error(e)
        except Exception as e:
            return self._handle_unexpected_error(e)
            
    async def _execute_task_safely(self, task: Dict[str, Any]) -> PluginResponse:
        """Execute task with timeout and retries"""
        max_retries = 3
        timeout = 30
        
        for attempt in range(max_retries):
            try:
                result = await asyncio.wait_for(
                    self._do_execute_task(task),
                    timeout=timeout
                )
                return result
                
            except asyncio.TimeoutError:
                if attempt == max_retries - 1:
                    raise TimeoutError(f"Task timeout after {timeout}s")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                await asyncio.sleep(1)
                
    def _handle_validation_error(self, error: ValidationError) -> PluginResponse:
        """Handle validation errors"""
        self.logger.error(f"Validation error: {str(error)}")
        return PluginResponse(
            success=False,
            error=f"Invalid input: {str(error)}",
            error_severity=ErrorSeverity.MEDIUM
        )
        
    def _handle_unexpected_error(self, error: Exception) -> PluginResponse:
        """Handle unexpected errors"""
        self.error_count += 1
        self.logger.error(f"Unexpected error: {str(error)}")
        self.logger.error(traceback.format_exc())
        
        if self.error_count >= self.max_errors:
            self.logger.critical("Too many errors, plugin may be unstable")
            
        return PluginResponse(
            success=False,
            error="Internal plugin error occurred",
            error_severity=ErrorSeverity.HIGH,
            error_code="PLUGIN_ERROR_001"
        )
```

### Configuration Management

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import json

@dataclass
class PluginSettings:
    refresh_interval: int = 300
    max_cache_size: int = 1000
    enable_debug: bool = False
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None

class ConfigurablePlugin(PRSMPlugin):
    def __init__(self):
        super().__init__()
        self.settings = PluginSettings()
        
    async def initialize(self, config: PluginConfig) -> bool:
        """Initialize with configuration"""
        # Load settings from multiple sources
        await self._load_settings(config)
        
        # Validate settings
        if not self._validate_settings():
            return False
            
        return True
        
    async def _load_settings(self, config: PluginConfig):
        """Load settings from multiple sources"""
        
        # 1. Default settings (already set in PluginSettings)
        
        # 2. Configuration file
        config_file = "plugin_config.json"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                self._update_settings_from_dict(file_config)
        
        # 3. Environment variables
        env_config = {
            'refresh_interval': os.getenv('PLUGIN_REFRESH_INTERVAL'),
            'api_endpoint': os.getenv('PLUGIN_API_ENDPOINT'),
            'api_key': os.getenv('PLUGIN_API_KEY'),
            'enable_debug': os.getenv('PLUGIN_DEBUG', '').lower() == 'true'
        }
        self._update_settings_from_dict(env_config)
        
        # 4. PRSM configuration (highest priority)
        prsm_config = {
            'refresh_interval': config.get('refresh_interval'),
            'max_cache_size': config.get('max_cache_size'),
            'enable_debug': config.get('debug_mode'),
            'api_endpoint': config.get('api_endpoint'),
            'api_key': config.get('api_key')
        }
        self._update_settings_from_dict(prsm_config)
        
    def _update_settings_from_dict(self, config_dict: Dict[str, Any]):
        """Update settings from dictionary, ignoring None values"""
        for key, value in config_dict.items():
            if value is not None and hasattr(self.settings, key):
                setattr(self.settings, key, value)
                
    def _validate_settings(self) -> bool:
        """Validate plugin settings"""
        if self.settings.refresh_interval <= 0:
            self.logger.error("refresh_interval must be positive")
            return False
            
        if self.settings.api_endpoint and not self.settings.api_key:
            self.logger.error("api_key required when api_endpoint is set")
            return False
            
        return True
```

## Troubleshooting

### Common Issues

#### Issue 1: Plugin Fails to Initialize

**Symptoms:**
```
❌ Plugin initialization failed: Missing required permission: read_analytics_data
```

**Solution:**
```python
# Check plugin manifest permissions
{
  "permissions": [
    "read_analytics_data",  # ✅ Add this permission
    "write_dashboard_config"
  ]
}

# Verify permission in code
async def initialize(self, config: PluginConfig) -> bool:
    if not config.has_permission('read_analytics_data'):
        self.logger.error("Missing required permission: read_analytics_data")
        return False
```

#### Issue 2: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'prsm_sdk'
```

**Solution:**
```bash
# Install PRSM SDK
pip install prsm-sdk

# Verify installation
python -c "import prsm_sdk; print(prsm_sdk.__version__)"

# Check requirements.txt includes SDK
echo "prsm-sdk>=1.0.0" >> requirements.txt
```

#### Issue 3: Task Execution Timeout

**Symptoms:**
```
⏰ Task execution timeout after 30s
```

**Solution:**
```python
async def execute_task(self, task: Dict[str, Any]) -> PluginResponse:
    # Add timeout handling
    try:
        result = await asyncio.wait_for(
            self._process_task(task),
            timeout=60  # Increase timeout
        )
        return result
    except asyncio.TimeoutError:
        return PluginResponse(
            success=False,
            error="Task execution timeout",
            data={"timeout": 60}
        )
```

#### Issue 4: Memory Usage Exceeds Limits

**Symptoms:**
```
🚫 Plugin memory usage (512MB) exceeds limit (256MB)
```

**Solution:**
```python
class MemoryEfficientPlugin(PRSMPlugin):
    def __init__(self):
        super().__init__()
        self._data_cache = {}
        self._max_cache_size = 100  # Limit cache size
        
    def _add_to_cache(self, key: str, data: Any):
        """Add data to cache with size limit"""
        if len(self._data_cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._data_cache))
            del self._data_cache[oldest_key]
            
        self._data_cache[key] = data
        
    async def process_large_dataset(self, data: List[Any]):
        """Process data in chunks to manage memory"""
        chunk_size = 1000
        results = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            chunk_result = await self._process_chunk(chunk)
            results.extend(chunk_result)
            
            # Clear chunk from memory
            del chunk
            
        return results
```

### Debugging Tips

#### Enable Debug Logging

```python
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DebuggablePlugin(PRSMPlugin):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    async def execute_task(self, task: Dict[str, Any]) -> PluginResponse:
        self.logger.debug(f"Executing task: {task.get('type', 'unknown')}")
        self.logger.debug(f"Task parameters: {task}")
        
        try:
            result = await self._do_execute_task(task)
            self.logger.debug(f"Task completed successfully: {result}")
            return result
        except Exception as e:
            self.logger.exception(f"Task failed: {str(e)}")
            raise
```

#### Performance Profiling

```python
import time
import functools

def profile_execution_time(func):
    """Decorator to profile function execution time"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            print(f"{func.__name__} executed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"{func.__name__} failed after {execution_time:.2f}s: {str(e)}")
            raise
    return wrapper

class ProfiledPlugin(PRSMPlugin):
    @profile_execution_time
    async def execute_task(self, task: Dict[str, Any]) -> PluginResponse:
        return await self._do_execute_task(task)
```

### Getting Help

- **Documentation**: Check the [PRSM Plugin SDK Documentation](https://docs.prsm.ai/sdk)
- **Community**: Join the [PRSM Developer Community](https://community.prsm.ai)
- **Support**: Contact plugin support at [plugin-support@prsm.ai](mailto:plugin-support@prsm.ai)
- **GitHub**: Report issues at [https://github.com/prsm/plugin-sdk/issues](https://github.com/prsm/plugin-sdk/issues)

## Next Steps

After completing this tutorial, you should be able to:

1. ✅ Create a basic PRSM plugin
2. ✅ Implement plugin security and validation
3. ✅ Test and debug your plugin
4. ✅ Package and publish to the marketplace

### Advanced Learning

- **Custom AI Models**: Learn to integrate custom AI models
- **Real-time Processing**: Implement WebSocket-based real-time features
- **Enterprise Features**: Build enterprise-grade plugins with advanced security
- **Plugin Monetization**: Set up pricing and billing for your plugins

### Community Contributions

Consider contributing to the PRSM plugin ecosystem:

- **Open Source Plugins**: Share useful plugins with the community
- **Documentation**: Improve plugin documentation and tutorials
- **Tools**: Create development tools and utilities for other plugin developers
- **Examples**: Share example plugins and best practices

Happy plugin development! 🚀