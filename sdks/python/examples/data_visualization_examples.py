#!/usr/bin/env python3
"""
PRSM Python SDK - Sophisticated Data Visualization Examples

This example demonstrates advanced data visualization capabilities using PRSM
for data analysis, model performance visualization, and interactive dashboards.

Features:
- Real-time model performance tracking
- Interactive cost analysis dashboards
- Multi-dimensional data exploration
- Advanced statistical visualizations
- Comparative model analysis
- Time-series prediction visualizations
- Network analysis and graph visualizations
"""

import asyncio
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Interactive widgets for Jupyter
try:
    import ipywidgets as widgets
    from IPython.display import display, HTML
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

# Network analysis
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from prsm_sdk import PRSMClient, PRSMError


@dataclass
class ModelPerformanceMetric:
    """Model performance tracking data"""
    timestamp: datetime
    model_name: str
    latency_ms: float
    cost_per_token: float
    quality_score: float
    tokens_processed: int
    success_rate: float
    error_count: int


@dataclass
class CostAnalysisData:
    """Cost analysis and optimization data"""
    timestamp: datetime
    total_cost: float
    cost_by_model: Dict[str, float]
    token_usage: Dict[str, int]
    efficiency_score: float
    budget_utilization: float


class PRSMVisualizationDashboard:
    """Sophisticated visualization dashboard for PRSM data"""
    
    def __init__(self, api_key: str):
        self.client = PRSMClient(api_key=api_key)
        self.performance_data: List[ModelPerformanceMetric] = []
        self.cost_data: List[CostAnalysisData] = []
        
        # Set up visualization styles
        self._setup_styles()
    
    def _setup_styles(self):
        """Configure visualization styles and themes"""
        # Matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Plotly theme
        self.plotly_theme = "plotly_white"
        
        # Custom color schemes
        self.model_colors = {
            'gpt-4': '#FF6B6B',
            'claude-3': '#4ECDC4', 
            'gpt-3.5-turbo': '#45B7D1',
            'gemini-pro': '#96CEB4',
            'llama-2': '#FECA57'
        }
    
    async def collect_performance_data(self, duration_hours: int = 24) -> None:
        """Collect model performance data for visualization"""
        print(f"📊 Collecting performance data for the last {duration_hours} hours...")
        
        try:
            # Get performance metrics from PRSM
            metrics = await self.client.analytics.get_performance_metrics(
                time_range=f"{duration_hours}h"
            )
            
            # Convert to our format
            for metric in metrics:
                perf_metric = ModelPerformanceMetric(
                    timestamp=datetime.fromisoformat(metric['timestamp']),
                    model_name=metric['model_name'],
                    latency_ms=metric['latency_ms'],
                    cost_per_token=metric['cost_per_token'],
                    quality_score=metric['quality_score'],
                    tokens_processed=metric['tokens_processed'],
                    success_rate=metric['success_rate'],
                    error_count=metric['error_count']
                )
                self.performance_data.append(perf_metric)
                
        except PRSMError as e:
            print(f"⚠️ Using simulated data due to API error: {e}")
            # Generate simulated data for demonstration
            self._generate_sample_data(duration_hours)
    
    def _generate_sample_data(self, duration_hours: int) -> None:
        """Generate realistic sample data for demonstration"""
        models = ['gpt-4', 'claude-3', 'gpt-3.5-turbo', 'gemini-pro', 'llama-2']
        start_time = datetime.now() - timedelta(hours=duration_hours)
        
        for i in range(duration_hours * 4):  # Every 15 minutes
            timestamp = start_time + timedelta(minutes=i * 15)
            
            for model in models:
                # Realistic performance characteristics per model
                base_latency = {'gpt-4': 2500, 'claude-3': 1800, 'gpt-3.5-turbo': 1200, 
                               'gemini-pro': 1500, 'llama-2': 800}[model]
                base_cost = {'gpt-4': 0.06, 'claude-3': 0.045, 'gpt-3.5-turbo': 0.002, 
                            'gemini-pro': 0.0025, 'llama-2': 0.0015}[model]
                base_quality = {'gpt-4': 0.95, 'claude-3': 0.92, 'gpt-3.5-turbo': 0.85, 
                               'gemini-pro': 0.88, 'llama-2': 0.82}[model]
                
                # Add realistic variance
                latency = base_latency + np.random.normal(0, base_latency * 0.2)
                cost = base_cost + np.random.normal(0, base_cost * 0.1)
                quality = min(1.0, max(0.0, base_quality + np.random.normal(0, 0.05)))
                tokens = np.random.randint(100, 2000)
                success_rate = min(1.0, max(0.8, np.random.normal(0.98, 0.02)))
                errors = np.random.poisson(1)
                
                metric = ModelPerformanceMetric(
                    timestamp=timestamp,
                    model_name=model,
                    latency_ms=latency,
                    cost_per_token=cost,
                    quality_score=quality,
                    tokens_processed=tokens,
                    success_rate=success_rate,
                    error_count=errors
                )
                self.performance_data.append(metric)
        
        # Generate cost data
        for i in range(duration_hours):
            timestamp = start_time + timedelta(hours=i)
            
            total_cost = np.random.uniform(50, 200)
            cost_by_model = {model: total_cost * np.random.uniform(0.1, 0.3) for model in models}
            token_usage = {model: np.random.randint(10000, 50000) for model in models}
            
            cost_data = CostAnalysisData(
                timestamp=timestamp,
                total_cost=total_cost,
                cost_by_model=cost_by_model,
                token_usage=token_usage,
                efficiency_score=np.random.uniform(0.7, 0.95),
                budget_utilization=np.random.uniform(0.3, 0.8)
            )
            self.cost_data.append(cost_data)
    
    def create_performance_dashboard(self) -> go.Figure:
        """Create comprehensive performance dashboard"""
        print("📈 Creating performance dashboard...")
        
        # Convert data to DataFrame
        df = pd.DataFrame([
            {
                'timestamp': metric.timestamp,
                'model': metric.model_name,
                'latency': metric.latency_ms,
                'cost': metric.cost_per_token,
                'quality': metric.quality_score,
                'tokens': metric.tokens_processed,
                'success_rate': metric.success_rate,
                'errors': metric.error_count
            }
            for metric in self.performance_data
        ])
        
        # Create subplot dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Model Latency Over Time',
                'Cost per Token Comparison', 
                'Quality Score Trends',
                'Token Usage Distribution',
                'Success Rate Monitoring',
                'Error Rate Analysis'
            ],
            specs=[
                [{"secondary_y": False}, {"type": "bar"}],
                [{"secondary_y": True}, {"type": "violin"}],
                [{"secondary_y": False}, {"type": "scatter"}]
            ]
        )
        
        # 1. Latency trends
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data['timestamp'],
                    y=model_data['latency'],
                    name=f'{model} Latency',
                    line=dict(color=self.model_colors.get(model, '#888888')),
                    mode='lines+markers'
                ),
                row=1, col=1
            )
        
        # 2. Cost comparison (bar chart)
        avg_cost_by_model = df.groupby('model')['cost'].mean()
        fig.add_trace(
            go.Bar(
                x=avg_cost_by_model.index,
                y=avg_cost_by_model.values,
                name='Avg Cost per Token',
                marker_color=[self.model_colors.get(model, '#888888') for model in avg_cost_by_model.index]
            ),
            row=1, col=2
        )
        
        # 3. Quality score trends with confidence bands
        for model in df['model'].unique():
            model_data = df[df['model'] == model].sort_values('timestamp')
            
            # Calculate rolling average and std
            window = 10
            model_data['quality_rolling'] = model_data['quality'].rolling(window=window, center=True).mean()
            model_data['quality_std'] = model_data['quality'].rolling(window=window, center=True).std()
            
            # Main line
            fig.add_trace(
                go.Scatter(
                    x=model_data['timestamp'],
                    y=model_data['quality_rolling'],
                    name=f'{model} Quality',
                    line=dict(color=self.model_colors.get(model, '#888888')),
                    mode='lines'
                ),
                row=2, col=1
            )
            
            # Confidence band
            fig.add_trace(
                go.Scatter(
                    x=list(model_data['timestamp']) + list(model_data['timestamp'])[::-1],
                    y=list(model_data['quality_rolling'] + model_data['quality_std']) + 
                      list(model_data['quality_rolling'] - model_data['quality_std'])[::-1],
                    fill='tonexty',
                    fillcolor=self.model_colors.get(model, '#888888').replace('#', 'rgba(') + ', 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. Token usage distribution (violin plot)
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            fig.add_trace(
                go.Violin(
                    y=model_data['tokens'],
                    name=model,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=self.model_colors.get(model, '#888888'),
                    opacity=0.6
                ),
                row=2, col=2
            )
        
        # 5. Success rate monitoring
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data['timestamp'],
                    y=model_data['success_rate'] * 100,
                    name=f'{model} Success Rate',
                    line=dict(color=self.model_colors.get(model, '#888888')),
                    mode='lines+markers'
                ),
                row=3, col=1
            )
        
        # 6. Error rate scatter plot
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data['latency'],
                    y=model_data['errors'],
                    name=f'{model} Errors',
                    mode='markers',
                    marker=dict(
                        color=self.model_colors.get(model, '#888888'),
                        size=model_data['tokens'] / 100,  # Size based on token usage
                        opacity=0.6
                    )
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="🚀 PRSM Model Performance Dashboard",
            height=1200,
            showlegend=True,
            template=self.plotly_theme
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Latency (ms)", row=1, col=1)
        fig.update_xaxes(title_text="Model", row=1, col=2)
        fig.update_yaxes(title_text="Cost per Token ($)", row=1, col=2)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Quality Score", row=2, col=1)
        fig.update_yaxes(title_text="Tokens Processed", row=2, col=2)
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="Success Rate (%)", row=3, col=1)
        fig.update_xaxes(title_text="Latency (ms)", row=3, col=2)
        fig.update_yaxes(title_text="Error Count", row=3, col=2)
        
        return fig
    
    def create_cost_optimization_analysis(self) -> go.Figure:
        """Create detailed cost optimization analysis"""
        print("💰 Creating cost optimization analysis...")
        
        # Convert cost data to DataFrame
        df = pd.DataFrame([
            {
                'timestamp': data.timestamp,
                'total_cost': data.total_cost,
                'efficiency': data.efficiency_score,
                'budget_util': data.budget_utilization,
                **{f'cost_{model}': cost for model, cost in data.cost_by_model.items()},
                **{f'tokens_{model}': tokens for model, tokens in data.token_usage.items()}
            }
            for data in self.cost_data
        ])
        
        # Create subplots for cost analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Cost Trends Over Time',
                'Cost Breakdown by Model',
                'Efficiency vs Budget Utilization',
                'Cost-Effectiveness Analysis'
            ],
            specs=[
                [{"secondary_y": True}, {"type": "pie"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        # 1. Cost trends with efficiency overlay
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['total_cost'],
                name='Total Cost',
                line=dict(color='#FF6B6B', width=3),
                mode='lines+markers'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['efficiency'] * 100,
                name='Efficiency %',
                line=dict(color='#4ECDC4', width=2),
                mode='lines',
                yaxis='y2'
            ),
            row=1, col=1
        )
        
        # 2. Cost breakdown pie chart (latest data)
        latest_costs = {
            model: df[f'cost_{model}'].iloc[-1] 
            for model in ['gpt-4', 'claude-3', 'gpt-3.5-turbo', 'gemini-pro', 'llama-2']
        }
        
        fig.add_trace(
            go.Pie(
                labels=list(latest_costs.keys()),
                values=list(latest_costs.values()),
                marker_colors=[self.model_colors.get(model, '#888888') for model in latest_costs.keys()],
                textinfo='label+percent',
                name='Cost Distribution'
            ),
            row=1, col=2
        )
        
        # 3. Efficiency vs Budget Utilization scatter
        fig.add_trace(
            go.Scatter(
                x=df['budget_util'] * 100,
                y=df['efficiency'] * 100,
                mode='markers',
                marker=dict(
                    size=df['total_cost'] / 5,  # Size based on total cost
                    color=df['total_cost'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Total Cost ($)")
                ),
                text=df['timestamp'].dt.strftime('%Y-%m-%d %H:%M'),
                textposition='middle center',
                name='Efficiency Analysis'
            ),
            row=2, col=1
        )
        
        # 4. Cost-effectiveness by model
        models = ['gpt-4', 'claude-3', 'gpt-3.5-turbo', 'gemini-pro', 'llama-2']
        cost_effectiveness = []
        
        for model in models:
            avg_cost = df[f'cost_{model}'].mean()
            avg_tokens = df[f'tokens_{model}'].mean()
            effectiveness = avg_tokens / avg_cost if avg_cost > 0 else 0
            cost_effectiveness.append(effectiveness)
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=cost_effectiveness,
                name='Tokens per Dollar',
                marker_color=[self.model_colors.get(model, '#888888') for model in models],
                text=[f'{val:.0f}' for val in cost_effectiveness],
                textposition='auto'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="💰 PRSM Cost Optimization Analysis",
            height=900,
            template=self.plotly_theme
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Cost ($)", row=1, col=1)
        fig.update_yaxes(title_text="Efficiency (%)", secondary_y=True, row=1, col=1)
        fig.update_xaxes(title_text="Budget Utilization (%)", row=2, col=1)
        fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
        fig.update_xaxes(title_text="Model", row=2, col=2)
        fig.update_yaxes(title_text="Tokens per Dollar", row=2, col=2)
        
        return fig
    
    def create_model_comparison_matrix(self) -> go.Figure:
        """Create comprehensive model comparison matrix"""
        print("🔍 Creating model comparison matrix...")
        
        # Calculate aggregate metrics per model
        df = pd.DataFrame([
            {
                'timestamp': metric.timestamp,
                'model': metric.model_name,
                'latency': metric.latency_ms,
                'cost': metric.cost_per_token,
                'quality': metric.quality_score,
                'tokens': metric.tokens_processed,
                'success_rate': metric.success_rate
            }
            for metric in self.performance_data
        ])
        
        models = df['model'].unique()
        metrics = ['latency', 'cost', 'quality', 'success_rate']
        
        # Create normalized comparison matrix
        comparison_data = []
        for model in models:
            model_data = df[df['model'] == model]
            model_metrics = []
            
            for metric in metrics:
                if metric in ['quality', 'success_rate']:
                    # Higher is better
                    normalized_score = model_data[metric].mean()
                else:
                    # Lower is better (invert and normalize)
                    max_val = df[metric].max()
                    normalized_score = (max_val - model_data[metric].mean()) / max_val
                
                model_metrics.append(normalized_score)
            
            comparison_data.append(model_metrics)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=comparison_data,
            x=['Latency', 'Cost', 'Quality', 'Success Rate'],
            y=models,
            colorscale='RdYlGn',
            text=[[f'{val:.3f}' for val in row] for row in comparison_data],
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="🏆 Model Performance Comparison Matrix<br><sub>Green = Better Performance, Red = Worse Performance</sub>",
            xaxis_title="Performance Metrics",
            yaxis_title="Models",
            height=600,
            template=self.plotly_theme
        )
        
        return fig
    
    def create_predictive_analysis(self) -> go.Figure:
        """Create predictive cost and usage analysis"""
        print("🔮 Creating predictive analysis...")
        
        # Use cost data for prediction
        df = pd.DataFrame([
            {
                'timestamp': data.timestamp,
                'total_cost': data.total_cost,
                'efficiency': data.efficiency_score
            }
            for data in self.cost_data
        ])
        
        # Simple linear prediction (in production, use more sophisticated models)
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        
        # Prepare data
        df['hours'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600
        X = df[['hours']].values
        y_cost = df['total_cost'].values
        y_efficiency = df['efficiency'].values
        
        # Fit polynomial regression
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X)
        
        cost_model = LinearRegression().fit(X_poly, y_cost)
        efficiency_model = LinearRegression().fit(X_poly, y_efficiency)
        
        # Predict future 24 hours
        future_hours = np.arange(df['hours'].max(), df['hours'].max() + 24, 0.5)
        future_X = future_hours.reshape(-1, 1)
        future_X_poly = poly_features.transform(future_X)
        
        cost_pred = cost_model.predict(future_X_poly)
        efficiency_pred = efficiency_model.predict(future_X_poly)
        
        # Create future timestamps
        base_time = df['timestamp'].max()
        future_timestamps = [base_time + timedelta(hours=h - df['hours'].max()) for h in future_hours]
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Cost Prediction', 'Efficiency Prediction'],
            shared_xaxes=True
        )
        
        # Historical data
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['total_cost'],
                name='Historical Cost',
                line=dict(color='#FF6B6B', width=2),
                mode='lines+markers'
            ),
            row=1, col=1
        )
        
        # Cost prediction
        fig.add_trace(
            go.Scatter(
                x=future_timestamps,
                y=cost_pred,
                name='Predicted Cost',
                line=dict(color='#FF6B6B', dash='dash', width=2),
                mode='lines'
            ),
            row=1, col=1
        )
        
        # Historical efficiency
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['efficiency'],
                name='Historical Efficiency',
                line=dict(color='#4ECDC4', width=2),
                mode='lines+markers'
            ),
            row=2, col=1
        )
        
        # Efficiency prediction
        fig.add_trace(
            go.Scatter(
                x=future_timestamps,
                y=efficiency_pred,
                name='Predicted Efficiency',
                line=dict(color='#4ECDC4', dash='dash', width=2),
                mode='lines'
            ),
            row=2, col=1
        )
        
        # Add confidence bands (simplified)
        cost_std = np.std(y_cost) * 0.1
        efficiency_std = np.std(y_efficiency) * 0.05
        
        # Cost confidence band
        fig.add_trace(
            go.Scatter(
                x=list(future_timestamps) + list(future_timestamps)[::-1],
                y=list(cost_pred + cost_std) + list(cost_pred - cost_std)[::-1],
                fill='tonexty',
                fillcolor='rgba(255, 107, 107, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name='Cost Confidence'
            ),
            row=1, col=1
        )
        
        # Efficiency confidence band
        fig.add_trace(
            go.Scatter(
                x=list(future_timestamps) + list(future_timestamps)[::-1],
                y=list(efficiency_pred + efficiency_std) + list(efficiency_pred - efficiency_std)[::-1],
                fill='tonexty',
                fillcolor='rgba(78, 205, 196, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name='Efficiency Confidence'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="🔮 Predictive Cost and Efficiency Analysis",
            height=800,
            template=self.plotly_theme
        )
        
        fig.update_yaxes(title_text="Cost ($)", row=1, col=1)
        fig.update_yaxes(title_text="Efficiency Score", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        
        return fig
    
    def create_network_analysis(self) -> Optional[go.Figure]:
        """Create network analysis of model relationships and usage patterns"""
        if not NETWORKX_AVAILABLE:
            print("⚠️ NetworkX not available, skipping network analysis")
            return None
        
        print("🕸️ Creating network analysis...")
        
        # Create a network graph of model relationships
        G = nx.Graph()
        
        # Add nodes for models
        models = ['gpt-4', 'claude-3', 'gpt-3.5-turbo', 'gemini-pro', 'llama-2']
        for model in models:
            G.add_node(model)
        
        # Add edges based on usage patterns and similarities
        # This is a simplified example - in practice, you'd use real correlation data
        usage_correlations = {
            ('gpt-4', 'claude-3'): 0.8,
            ('gpt-4', 'gpt-3.5-turbo'): 0.6,
            ('claude-3', 'gemini-pro'): 0.7,
            ('gpt-3.5-turbo', 'llama-2'): 0.5,
            ('gemini-pro', 'llama-2'): 0.4
        }
        
        for (model1, model2), weight in usage_correlations.items():
            G.add_edge(model1, model2, weight=weight)
        
        # Calculate layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(G[edge[0]][edge[1]]['weight'])
        
        # Create node traces
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_colors = [self.model_colors.get(node, '#888888') for node in G.nodes()]
        
        # Calculate node sizes based on usage
        df = pd.DataFrame([
            {'model': metric.model_name, 'tokens': metric.tokens_processed}
            for metric in self.performance_data
        ])
        
        usage_by_model = df.groupby('model')['tokens'].sum()
        node_sizes = [usage_by_model.get(node, 1000) / 100 for node in G.nodes()]
        
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            ),
            text=list(G.nodes()),
            textposition='middle center',
            textfont=dict(size=12, color='white'),
            hoverinfo='text',
            hovertext=[f'{node}<br>Total Tokens: {usage_by_model.get(node, 0):,}' for node in G.nodes()],
            showlegend=False
        ))
        
        fig.update_layout(
            title="🕸️ Model Usage Network Analysis<br><sub>Node size = Token usage, Edge thickness = Usage correlation</sub>",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Models with similar usage patterns are connected",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor="left", yanchor="bottom",
                font=dict(size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            template=self.plotly_theme
        )
        
        return fig
    
    def create_statistical_distribution_analysis(self) -> go.Figure:
        """Create statistical distribution analysis of model metrics"""
        print("📊 Creating statistical distribution analysis...")
        
        # Prepare data
        df = pd.DataFrame([
            {
                'model': metric.model_name,
                'latency': metric.latency_ms,
                'cost': metric.cost_per_token * 1000,  # Convert to cost per 1K tokens
                'quality': metric.quality_score,
                'tokens': metric.tokens_processed
            }
            for metric in self.performance_data
        ])
        
        # Create distribution plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Latency Distribution by Model',
                'Cost Distribution by Model',
                'Quality Score Distribution',
                'Token Usage Distribution'
            ],
            specs=[
                [{"type": "violin"}, {"type": "box"}],
                [{"type": "histogram"}, {"type": "histogram"}]
            ]
        )
        
        # 1. Latency violin plots
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            fig.add_trace(
                go.Violin(
                    y=model_data['latency'],
                    name=model,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=self.model_colors.get(model, '#888888'),
                    opacity=0.7,
                    side='positive',
                    width=0.8
                ),
                row=1, col=1
            )
        
        # 2. Cost box plots
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            fig.add_trace(
                go.Box(
                    y=model_data['cost'],
                    name=model,
                    marker_color=self.model_colors.get(model, '#888888'),
                    boxpoints='outliers'
                ),
                row=1, col=2
            )
        
        # 3. Quality histogram
        fig.add_trace(
            go.Histogram(
                x=df['quality'],
                nbinsx=20,
                name='Quality Distribution',
                marker_color='rgba(78, 205, 196, 0.7)',
                opacity=0.8
            ),
            row=2, col=1
        )
        
        # 4. Token usage histogram (log scale)
        fig.add_trace(
            go.Histogram(
                x=np.log10(df['tokens']),
                nbinsx=25,
                name='Log Token Usage',
                marker_color='rgba(255, 107, 107, 0.7)',
                opacity=0.8
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="📊 Statistical Distribution Analysis",
            height=800,
            showlegend=False,
            template=self.plotly_theme
        )
        
        # Update axes
        fig.update_yaxes(title_text="Latency (ms)", row=1, col=1)
        fig.update_yaxes(title_text="Cost per 1K Tokens ($)", row=1, col=2)
        fig.update_xaxes(title_text="Quality Score", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Log10(Token Count)", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        return fig
    
    def export_dashboard_report(self, filename: str = "prsm_dashboard_report.html") -> str:
        """Export complete dashboard as interactive HTML report"""
        print(f"📄 Exporting dashboard to {filename}...")
        
        # Create all visualizations
        performance_fig = self.create_performance_dashboard()
        cost_fig = self.create_cost_optimization_analysis()
        comparison_fig = self.create_model_comparison_matrix()
        prediction_fig = self.create_predictive_analysis()
        network_fig = self.create_network_analysis()
        distribution_fig = self.create_statistical_distribution_analysis()
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PRSM Analytics Dashboard Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .section {{ margin: 40px 0; }}
                .chart-container {{ margin: 20px 0; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 PRSM Analytics Dashboard Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <h2>📈 Executive Summary</h2>
                <p>This report provides comprehensive analysis of PRSM model performance, cost optimization, and usage patterns.</p>
                <ul>
                    <li><strong>Models Analyzed:</strong> {len(set(m.model_name for m in self.performance_data))}</li>
                    <li><strong>Data Points:</strong> {len(self.performance_data)}</li>
                    <li><strong>Total Cost Analyzed:</strong> ${sum(d.total_cost for d in self.cost_data):.2f}</li>
                    <li><strong>Average Efficiency:</strong> {np.mean([d.efficiency_score for d in self.cost_data]):.2%}</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>📊 Performance Dashboard</h2>
                <div class="chart-container" id="performance-chart"></div>
            </div>
            
            <div class="section">
                <h2>💰 Cost Analysis</h2>
                <div class="chart-container" id="cost-chart"></div>
            </div>
            
            <div class="section">
                <h2>🏆 Model Comparison</h2>
                <div class="chart-container" id="comparison-chart"></div>
            </div>
            
            <div class="section">
                <h2>🔮 Predictive Analysis</h2>
                <div class="chart-container" id="prediction-chart"></div>
            </div>
            
            <div class="section">
                <h2>📊 Statistical Distributions</h2>
                <div class="chart-container" id="distribution-chart"></div>
            </div>
        """
        
        if network_fig:
            html_content += """
            <div class="section">
                <h2>🕸️ Network Analysis</h2>
                <div class="chart-container" id="network-chart"></div>
            </div>
            """
        
        html_content += """
            <script>
        """
        
        # Add Plotly JSON data
        html_content += f"Plotly.newPlot('performance-chart', {performance_fig.to_json()});\n"
        html_content += f"Plotly.newPlot('cost-chart', {cost_fig.to_json()});\n"
        html_content += f"Plotly.newPlot('comparison-chart', {comparison_fig.to_json()});\n"
        html_content += f"Plotly.newPlot('prediction-chart', {prediction_fig.to_json()});\n"
        html_content += f"Plotly.newPlot('distribution-chart', {distribution_fig.to_json()});\n"
        
        if network_fig:
            html_content += f"Plotly.newPlot('network-chart', {network_fig.to_json()});\n"
        
        html_content += """
            </script>
        </body>
        </html>
        """
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Dashboard exported to {filename}")
        return filename


async def create_real_time_monitoring_example():
    """Example of real-time monitoring dashboard"""
    print("🔄 Creating real-time monitoring example...")
    
    # This would connect to PRSM's real-time data stream
    api_key = os.getenv("PRSM_API_KEY")
    if not api_key:
        print("⚠️ Using demo mode - set PRSM_API_KEY for live data")
        api_key = "demo_key"
    
    dashboard = PRSMVisualizationDashboard(api_key)
    
    # Collect performance data
    await dashboard.collect_performance_data(duration_hours=24)
    
    # Create visualizations
    print("📊 Creating performance dashboard...")
    perf_fig = dashboard.create_performance_dashboard()
    perf_fig.show()
    
    print("💰 Creating cost analysis...")
    cost_fig = dashboard.create_cost_optimization_analysis()
    cost_fig.show()
    
    print("🏆 Creating model comparison...")
    comp_fig = dashboard.create_model_comparison_matrix()
    comp_fig.show()
    
    return dashboard


async def create_advanced_analytics_example():
    """Example of advanced analytics and machine learning"""
    print("🧠 Creating advanced analytics example...")
    
    api_key = os.getenv("PRSM_API_KEY", "demo_key")
    dashboard = PRSMVisualizationDashboard(api_key)
    
    # Collect data
    await dashboard.collect_performance_data(duration_hours=48)
    
    # Advanced analytics
    print("🔮 Creating predictive analysis...")
    pred_fig = dashboard.create_predictive_analysis()
    pred_fig.show()
    
    print("📊 Creating statistical analysis...")
    stat_fig = dashboard.create_statistical_distribution_analysis()
    stat_fig.show()
    
    # Network analysis if available
    network_fig = dashboard.create_network_analysis()
    if network_fig:
        network_fig.show()
    
    return dashboard


async def create_business_intelligence_report():
    """Create comprehensive business intelligence report"""
    print("📈 Creating business intelligence report...")
    
    api_key = os.getenv("PRSM_API_KEY", "demo_key")
    dashboard = PRSMVisualizationDashboard(api_key)
    
    # Collect comprehensive data
    await dashboard.collect_performance_data(duration_hours=72)
    
    # Export comprehensive report
    report_file = dashboard.export_dashboard_report("prsm_bi_report.html")
    
    print(f"✅ Business intelligence report created: {report_file}")
    print("🌐 Open the HTML file in your browser to view the interactive dashboard")
    
    return dashboard, report_file


def create_jupyter_widgets_example():
    """Example of interactive Jupyter widgets for PRSM data exploration"""
    if not JUPYTER_AVAILABLE:
        print("⚠️ Jupyter widgets not available - install ipywidgets to use this feature")
        return
    
    print("📱 Creating Jupyter widgets example...")
    
    # Create interactive widgets
    model_selector = widgets.SelectMultiple(
        options=['gpt-4', 'claude-3', 'gpt-3.5-turbo', 'gemini-pro', 'llama-2'],
        value=['gpt-4', 'claude-3'],
        description='Models:',
        disabled=False
    )
    
    time_range_slider = widgets.IntRangeSlider(
        value=[1, 24],
        min=1,
        max=72,
        step=1,
        description='Hours:',
        continuous_update=False
    )
    
    metric_selector = widgets.Dropdown(
        options=['latency', 'cost', 'quality', 'tokens'],
        value='latency',
        description='Metric:',
    )
    
    # Interactive function
    def update_visualization(models, time_range, metric):
        print(f"Updating visualization for {models} over {time_range[0]}-{time_range[1]} hours, showing {metric}")
        # This would update the actual visualization
        return f"Chart updated: {len(models)} models, {time_range[1]-time_range[0]} hour range, {metric} metric"
    
    # Create interactive widget
    interactive_plot = widgets.interactive(
        update_visualization,
        models=model_selector,
        time_range=time_range_slider,
        metric=metric_selector
    )
    
    # Display widgets
    display(widgets.VBox([
        widgets.HTML("<h3>🎛️ Interactive PRSM Analytics Dashboard</h3>"),
        model_selector,
        time_range_slider,
        metric_selector,
        interactive_plot
    ]))
    
    print("✅ Interactive widgets created - use in Jupyter notebook")


async def main():
    """Run all data visualization examples"""
    print("🎨 PRSM Python SDK - Data Visualization Examples")
    print("=" * 70)
    
    try:
        # Check dependencies
        print("🔍 Checking dependencies...")
        required_packages = {
            'matplotlib': 'Basic plotting',
            'seaborn': 'Statistical visualizations', 
            'plotly': 'Interactive dashboards',
            'pandas': 'Data manipulation',
            'numpy': 'Numerical computing',
            'sklearn': 'Machine learning'
        }
        
        missing_packages = []
        for package, description in required_packages.items():
            try:
                __import__(package)
                print(f"✅ {package}: {description}")
            except ImportError:
                missing_packages.append(package)
                print(f"❌ {package}: {description} - MISSING")
        
        if missing_packages:
            print(f"\n⚠️  Install missing packages: pip install {' '.join(missing_packages)}")
            print("Continuing with available features...\n")
        
        # Run examples
        print("\n🔄 Real-time Monitoring Dashboard")
        print("-" * 40)
        dashboard1 = await create_real_time_monitoring_example()
        
        print("\n🧠 Advanced Analytics & Predictions")
        print("-" * 40)
        dashboard2 = await create_advanced_analytics_example()
        
        print("\n📈 Business Intelligence Report")
        print("-" * 40)
        dashboard3, report_file = await create_business_intelligence_report()
        
        print("\n📱 Interactive Jupyter Widgets")
        print("-" * 40)
        create_jupyter_widgets_example()
        
        print("\n" + "=" * 70)
        print("✅ All visualization examples completed!")
        print("\n💡 Key Features Demonstrated:")
        print("• Real-time performance monitoring")
        print("• Cost optimization analysis")
        print("• Predictive analytics and forecasting")
        print("• Statistical distribution analysis")
        print("• Network analysis of model relationships")
        print("• Interactive dashboards and reports")
        print("• Business intelligence reporting")
        print("• Jupyter notebook integration")
        
        print(f"\n📄 Reports generated:")
        print(f"• {report_file}")
        
        # Close clients
        for dashboard in [dashboard1, dashboard2, dashboard3]:
            if hasattr(dashboard, 'client'):
                await dashboard.client.close()
        
    except Exception as e:
        print(f"❌ Error in visualization examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Install required packages if running standalone
    print("📦 PRSM Data Visualization Examples")
    print("This example requires additional packages for full functionality:")
    print("pip install matplotlib seaborn plotly pandas numpy scikit-learn networkx ipywidgets")
    print("\nRunning examples...\n")
    
    asyncio.run(main())