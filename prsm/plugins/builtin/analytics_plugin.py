#!/usr/bin/env python3
"""
Analytics Plugin
===============

Plugin for advanced analytics and visualization capabilities.
"""

import logging
from typing import Dict, List, Any, Callable, Optional
from ..plugin_manager import Plugin, PluginMetadata
from ..optional_deps import require_optional, has_optional_dependency

logger = logging.getLogger(__name__)


class AnalyticsPlugin(Plugin):
    """Plugin for analytics and visualization"""
    
    def __init__(self):
        self._pandas = None
        self._numpy = None
        self._matplotlib = None
        self._plotly = None
        self._initialized = False
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="analytics",
            version="1.0.0",
            description="Advanced analytics and visualization capabilities",
            author="PRSM Core Team",
            dependencies=[],
            optional_dependencies=["pandas", "numpy", "matplotlib", "plotly"],
            entry_points={
                "data_analyzer": "analyze_data",
                "chart_generator": "generate_charts"
            }
        )
    
    def initialize(self, **kwargs) -> bool:
        """Initialize the analytics plugin"""
        try:
            # Try to import optional dependencies
            self._pandas = require_optional("pandas")
            self._numpy = require_optional("numpy")
            self._matplotlib = require_optional("matplotlib")
            self._plotly = require_optional("plotly")
            
            if self._pandas:
                logger.info("pandas available - advanced data analysis enabled")
            if self._numpy:
                logger.info("numpy available - numerical computing enabled")
            if self._matplotlib:
                logger.info("matplotlib available - static plotting enabled")
            if self._plotly:
                logger.info("plotly available - interactive plotting enabled")
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize analytics plugin: {e}")
            return False
    
    def cleanup(self) -> bool:
        """Cleanup analytics resources"""
        self._initialized = False
        return True
    
    def get_capabilities(self) -> List[str]:
        """Get analytics capabilities"""
        capabilities = ["basic_analytics"]
        
        if self._pandas:
            capabilities.extend([
                "data_analysis",
                "data_transformation",
                "statistical_analysis"
            ])
        
        if self._numpy:
            capabilities.extend([
                "numerical_computing",
                "array_operations",
                "mathematical_functions"
            ])
        
        if self._matplotlib:
            capabilities.extend([
                "static_plotting",
                "chart_generation",
                "data_visualization"
            ])
        
        if self._plotly:
            capabilities.extend([
                "interactive_plotting",
                "dashboard_creation",
                "web_visualization"
            ])
        
        return capabilities
    
    def get_hooks(self) -> Dict[str, Callable]:
        """Get analytics hook functions"""
        hooks = {
            "analyze_basic": self.analyze_basic_data,
            "generate_summary": self.generate_data_summary
        }
        
        if self._pandas:
            hooks.update({
                "analyze_dataframe": self.analyze_dataframe,
                "transform_data": self.transform_data,
                "statistical_analysis": self.statistical_analysis
            })
        
        if self._matplotlib or self._plotly:
            hooks.update({
                "create_chart": self.create_chart,
                "generate_plots": self.generate_plots
            })
        
        return hooks
    
    def analyze_basic_data(self, data: Any) -> Dict[str, Any]:
        """Perform basic data analysis"""
        analysis = {
            "data_type": type(data).__name__,
            "timestamp": self._get_timestamp()
        }
        
        if isinstance(data, (list, tuple)):
            analysis.update({
                "length": len(data),
                "sample": data[:5] if len(data) > 5 else data
            })
            
            # Basic statistics for numeric data
            if data and all(isinstance(x, (int, float)) for x in data):
                analysis.update({
                    "min": min(data),
                    "max": max(data),
                    "mean": sum(data) / len(data),
                    "sum": sum(data)
                })
        
        elif isinstance(data, dict):
            analysis.update({
                "keys": list(data.keys()),
                "length": len(data)
            })
        
        elif isinstance(data, str):
            analysis.update({
                "length": len(data),
                "word_count": len(data.split()) if data else 0
            })
        
        return analysis
    
    def generate_data_summary(self, data: Any, title: str = "Data Summary") -> Dict[str, Any]:
        """Generate a comprehensive data summary"""
        summary = {
            "title": title,
            "timestamp": self._get_timestamp(),
            "basic_analysis": self.analyze_basic_data(data)
        }
        
        # Add advanced analysis if pandas is available
        if self._pandas and hasattr(data, 'to_dict'):
            try:
                # Assume it's a pandas-like object
                df = self._pandas.DataFrame(data)
                summary["dataframe_analysis"] = self.analyze_dataframe(df)
            except Exception as e:
                logger.warning(f"Could not perform dataframe analysis: {e}")
        
        return summary
    
    def analyze_dataframe(self, df) -> Dict[str, Any]:
        """Analyze a pandas DataFrame"""
        if not self._pandas:
            return {"error": "pandas not available"}
        
        try:
            analysis = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "describe": df.describe().to_dict() if not df.empty else {}
            }
            
            # Add correlation matrix for numeric columns
            numeric_cols = df.select_dtypes(include=[self._numpy.number]).columns if self._numpy else []
            if len(numeric_cols) > 1:
                analysis["correlation"] = df[numeric_cols].corr().to_dict()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing dataframe: {e}")
            return {"error": str(e)}
    
    def transform_data(self, data: Any, transformation: str, **kwargs) -> Any:
        """Transform data using pandas operations"""
        if not self._pandas:
            return data
        
        try:
            df = self._pandas.DataFrame(data) if not hasattr(data, 'to_dict') else data
            
            if transformation == "normalize":
                return (df - df.mean()) / df.std()
            elif transformation == "standardize":
                return (df - df.min()) / (df.max() - df.min())
            elif transformation == "log":
                return self._numpy.log(df) if self._numpy else df
            elif transformation == "filter":
                condition = kwargs.get("condition")
                return df.query(condition) if condition else df
            elif transformation == "groupby":
                column = kwargs.get("column")
                agg_func = kwargs.get("agg", "mean")
                return df.groupby(column).agg(agg_func) if column else df
            else:
                logger.warning(f"Unknown transformation: {transformation}")
                return df
                
        except Exception as e:
            logger.error(f"Error transforming data: {e}")
            return data
    
    def statistical_analysis(self, data: Any) -> Dict[str, Any]:
        """Perform statistical analysis"""
        if not self._pandas and not self._numpy:
            return self.analyze_basic_data(data)
        
        try:
            if self._pandas and hasattr(data, 'to_dict'):
                df = data
            else:
                df = self._pandas.DataFrame(data) if self._pandas else None
            
            if df is None:
                return self.analyze_basic_data(data)
            
            numeric_cols = df.select_dtypes(include=[self._numpy.number]).columns if self._numpy else []
            
            stats = {}
            for col in numeric_cols:
                series = df[col].dropna()
                if len(series) > 0:
                    stats[col] = {
                        "count": len(series),
                        "mean": float(series.mean()),
                        "std": float(series.std()),
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "median": float(series.median()),
                        "q25": float(series.quantile(0.25)),
                        "q75": float(series.quantile(0.75))
                    }
                    
                    if self._numpy:
                        stats[col].update({
                            "skewness": float(series.skew()),
                            "kurtosis": float(series.kurtosis())
                        })
            
            return {
                "timestamp": self._get_timestamp(),
                "statistical_summary": stats,
                "total_columns": len(df.columns),
                "numeric_columns": len(numeric_cols)
            }
            
        except Exception as e:
            logger.error(f"Error in statistical analysis: {e}")
            return {"error": str(e)}
    
    def create_chart(self, data: Any, chart_type: str = "line", **kwargs) -> Optional[str]:
        """Create a chart from data"""
        if not (self._matplotlib or self._plotly):
            logger.warning("No plotting library available")
            return None
        
        try:
            # Prefer plotly for interactive charts
            if self._plotly and kwargs.get("interactive", True):
                return self._create_plotly_chart(data, chart_type, **kwargs)
            elif self._matplotlib:
                return self._create_matplotlib_chart(data, chart_type, **kwargs)
            else:
                logger.warning("No suitable plotting library for chart creation")
                return None
                
        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            return None
    
    def generate_plots(self, data: Any, plot_types: List[str] = None) -> Dict[str, Any]:
        """Generate multiple plots for data analysis"""
        if plot_types is None:
            plot_types = ["histogram", "scatter", "box"]
        
        plots = {}
        for plot_type in plot_types:
            try:
                plot_result = self.create_chart(data, plot_type)
                if plot_result:
                    plots[plot_type] = plot_result
            except Exception as e:
                logger.warning(f"Could not create {plot_type} plot: {e}")
        
        return {
            "timestamp": self._get_timestamp(),
            "plots": plots,
            "data_summary": self.analyze_basic_data(data)
        }
    
    def _create_plotly_chart(self, data: Any, chart_type: str, **kwargs) -> str:
        """Create a chart using plotly"""
        if not self._plotly:
            return None
        
        try:
            fig = None
            
            if chart_type == "line":
                fig = self._plotly.graph_objects.Figure()
                if isinstance(data, dict):
                    for key, values in data.items():
                        fig.add_trace(self._plotly.graph_objects.Scatter(
                            y=values, name=key, mode='lines'
                        ))
                elif isinstance(data, (list, tuple)):
                    fig.add_trace(self._plotly.graph_objects.Scatter(
                        y=data, mode='lines'
                    ))
            
            elif chart_type == "bar":
                if isinstance(data, dict):
                    fig = self._plotly.graph_objects.Figure(
                        data=[self._plotly.graph_objects.Bar(
                            x=list(data.keys()),
                            y=list(data.values())
                        )]
                    )
            
            elif chart_type == "histogram":
                if isinstance(data, (list, tuple)):
                    fig = self._plotly.graph_objects.Figure(
                        data=[self._plotly.graph_objects.Histogram(x=data)]
                    )
            
            if fig:
                fig.update_layout(
                    title=kwargs.get("title", f"{chart_type.title()} Chart"),
                    xaxis_title=kwargs.get("xlabel", "X"),
                    yaxis_title=kwargs.get("ylabel", "Y")
                )
                return fig.to_html()
            
        except Exception as e:
            logger.error(f"Error creating plotly chart: {e}")
        
        return None
    
    def _create_matplotlib_chart(self, data: Any, chart_type: str, **kwargs) -> str:
        """Create a chart using matplotlib"""
        if not self._matplotlib:
            return None
        
        try:
            import matplotlib.pyplot as plt
            import io
            import base64
            
            plt.figure(figsize=kwargs.get("figsize", (10, 6)))
            
            if chart_type == "line":
                if isinstance(data, dict):
                    for key, values in data.items():
                        plt.plot(values, label=key)
                    plt.legend()
                elif isinstance(data, (list, tuple)):
                    plt.plot(data)
            
            elif chart_type == "bar":
                if isinstance(data, dict):
                    plt.bar(list(data.keys()), list(data.values()))
            
            elif chart_type == "histogram":
                if isinstance(data, (list, tuple)):
                    plt.hist(data, bins=kwargs.get("bins", 20))
            
            plt.title(kwargs.get("title", f"{chart_type.title()} Chart"))
            plt.xlabel(kwargs.get("xlabel", "X"))
            plt.ylabel(kwargs.get("ylabel", "Y"))
            
            # Save to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Error creating matplotlib chart: {e}")
        
        return None
    
    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()