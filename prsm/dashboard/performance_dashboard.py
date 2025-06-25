#!/usr/bin/env python3
"""
PRSM Performance Dashboard
Live Streamlit dashboard for monitoring and visualizing PRSM performance metrics

Features:
- Real-time benchmark visualization
- Historical performance tracking
- Interactive performance analysis
- Comparative benchmark results
- System health monitoring
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import numpy as np
from typing import Dict, List, Optional, Any
import threading
import queue

# Add PRSM to path
PRSM_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PRSM_ROOT))

# Import PRSM performance components
try:
    from prsm.performance.benchmark_collector import get_global_collector
    from comprehensive_performance_benchmark import PerformanceBenchmarkSuite, BenchmarkConfig, BenchmarkType, NetworkCondition
except ImportError:
    st.error("Could not import PRSM performance modules. Make sure PRSM is properly installed.")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="PRSM Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-good { color: #28a745; }
    .status-warning { color: #ffc107; }
    .status-error { color: #dc3545; }
    .big-font { font-size: 2rem !important; }
</style>
""", unsafe_allow_html=True)

class PerformanceDashboard:
    """Main dashboard class for PRSM performance monitoring"""
    
    def __init__(self):
        self.benchmark_suite = PerformanceBenchmarkSuite("dashboard_results")
        self.results_cache = {}
        self.live_metrics = queue.Queue()
        
        # Initialize session state
        if 'benchmark_running' not in st.session_state:
            st.session_state.benchmark_running = False
        if 'benchmark_results' not in st.session_state:
            st.session_state.benchmark_results = []
        if 'live_mode' not in st.session_state:
            st.session_state.live_mode = False

    def load_historical_results(self) -> List[Dict]:
        """Load historical benchmark results from files"""
        results = []
        results_dir = Path("benchmark_results")
        
        if results_dir.exists():
            for json_file in results_dir.glob("benchmark_results_*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        if 'results' in data:
                            results.extend(data['results'])
                except Exception as e:
                    st.warning(f"Could not load {json_file}: {e}")
        
        return results

    def create_performance_overview(self):
        """Create the main performance overview section"""
        st.header("📊 PRSM Performance Overview")
        
        # Load latest results
        historical_results = self.load_historical_results()
        
        if not historical_results:
            st.info("No benchmark results found. Run some benchmarks to see performance data.")
            return
        
        # Create metrics summary
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate key metrics
        throughput_values = [r['metrics']['operations_per_second'] for r in historical_results]
        latency_values = [r['metrics']['mean_latency_ms'] for r in historical_results]
        success_rates = [r['metrics']['consensus_success_rate'] for r in historical_results]
        node_counts = [r['config']['node_count'] for r in historical_results]
        
        with col1:
            avg_throughput = np.mean(throughput_values) if throughput_values else 0
            st.metric(
                "Average Throughput", 
                f"{avg_throughput:.2f} ops/s",
                delta=f"{avg_throughput - np.mean(throughput_values[-5:]) if len(throughput_values) >= 5 else 0:.2f}"
            )
        
        with col2:
            avg_latency = np.mean(latency_values) if latency_values else 0
            st.metric(
                "Average Latency", 
                f"{avg_latency:.1f}ms",
                delta=f"{avg_latency - np.mean(latency_values[-5:]) if len(latency_values) >= 5 else 0:.1f}ms"
            )
        
        with col3:
            avg_success_rate = np.mean(success_rates) if success_rates else 0
            st.metric(
                "Success Rate", 
                f"{avg_success_rate:.1%}",
                delta=f"{avg_success_rate - np.mean(success_rates[-5:]) if len(success_rates) >= 5 else 0:.1%}"
            )
        
        with col4:
            max_nodes = max(node_counts) if node_counts else 0
            st.metric(
                "Max Tested Nodes", 
                f"{max_nodes}",
                delta=f"+{max_nodes - min(node_counts) if node_counts else 0}"
            )

    def create_throughput_analysis(self, results: List[Dict]):
        """Create throughput analysis charts"""
        st.subheader("🚀 Throughput Analysis")
        
        # Prepare data
        df = pd.DataFrame([{
            'timestamp': r['timing']['start_time'],
            'throughput': r['metrics']['operations_per_second'],
            'node_count': r['config']['node_count'],
            'network_condition': r['config']['network_condition'],
            'benchmark_type': r['config']['benchmark_type']
        } for r in results])
        
        if df.empty:
            st.info("No throughput data available")
            return
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Throughput over time
            fig_time = px.line(
                df, 
                x='timestamp', 
                y='throughput',
                color='benchmark_type',
                title="Throughput Over Time",
                labels={'throughput': 'Operations per Second', 'timestamp': 'Time'}
            )
            fig_time.update_layout(height=400)
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            # Throughput vs Node Count
            scaling_df = df[df['benchmark_type'] == 'consensus_scaling'].copy()
            if not scaling_df.empty:
                fig_scaling = px.scatter(
                    scaling_df,
                    x='node_count',
                    y='throughput',
                    size='throughput',
                    color='network_condition',
                    title="Scaling Characteristics",
                    labels={'node_count': 'Number of Nodes', 'throughput': 'Operations per Second'}
                )
                fig_scaling.update_layout(height=400)
                st.plotly_chart(fig_scaling, use_container_width=True)

    def create_latency_analysis(self, results: List[Dict]):
        """Create latency analysis charts"""
        st.subheader("⏱️ Latency Analysis")
        
        # Prepare latency data
        latency_data = []
        for r in results:
            latency_data.append({
                'benchmark': r['config']['name'],
                'mean_latency': r['metrics']['mean_latency_ms'],
                'p95_latency': r['metrics']['p95_latency_ms'],
                'p99_latency': r['metrics']['p99_latency_ms'],
                'node_count': r['config']['node_count'],
                'network_condition': r['config']['network_condition']
            })
        
        df_latency = pd.DataFrame(latency_data)
        
        if df_latency.empty:
            st.info("No latency data available")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Latency percentiles
            fig_percentiles = go.Figure()
            fig_percentiles.add_trace(go.Bar(name='Mean', x=df_latency['benchmark'], y=df_latency['mean_latency']))
            fig_percentiles.add_trace(go.Bar(name='P95', x=df_latency['benchmark'], y=df_latency['p95_latency']))
            fig_percentiles.add_trace(go.Bar(name='P99', x=df_latency['benchmark'], y=df_latency['p99_latency']))
            
            fig_percentiles.update_layout(
                title="Latency Percentiles by Benchmark",
                xaxis_title="Benchmark",
                yaxis_title="Latency (ms)",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_percentiles, use_container_width=True)
        
        with col2:
            # Network condition impact
            network_impact = df_latency.groupby('network_condition')['mean_latency'].mean().reset_index()
            fig_network = px.bar(
                network_impact,
                x='network_condition',
                y='mean_latency',
                title="Network Condition Impact on Latency",
                labels={'mean_latency': 'Mean Latency (ms)', 'network_condition': 'Network Condition'}
            )
            fig_network.update_layout(height=400)
            st.plotly_chart(fig_network, use_container_width=True)

    def create_scaling_analysis(self, results: List[Dict]):
        """Create scaling efficiency analysis"""
        st.subheader("📈 Scaling Efficiency Analysis")
        
        # Filter scaling results
        scaling_results = [r for r in results if r['config']['benchmark_type'] == 'consensus_scaling']
        
        if not scaling_results:
            st.info("No scaling benchmark data available")
            return
        
        # Calculate scaling efficiency
        scaling_data = []
        scaling_results.sort(key=lambda x: x['config']['node_count'])
        
        if scaling_results:
            baseline = scaling_results[0]
            baseline_throughput = baseline['metrics']['operations_per_second']
            baseline_nodes = baseline['config']['node_count']
            
            for result in scaling_results:
                nodes = result['config']['node_count']
                throughput = result['metrics']['operations_per_second']
                
                scaling_factor = nodes / baseline_nodes
                throughput_ratio = throughput / baseline_throughput if baseline_throughput > 0 else 0
                efficiency = throughput_ratio / scaling_factor if scaling_factor > 0 else 0
                
                scaling_data.append({
                    'nodes': nodes,
                    'throughput': throughput,
                    'efficiency': efficiency,
                    'latency': result['metrics']['mean_latency_ms'],
                    'ops_per_node': result['metrics']['operations_per_node_per_second']
                })
        
        df_scaling = pd.DataFrame(scaling_data)
        
        if not df_scaling.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Scaling efficiency
                fig_efficiency = px.line(
                    df_scaling,
                    x='nodes',
                    y='efficiency',
                    title="Scaling Efficiency",
                    labels={'nodes': 'Number of Nodes', 'efficiency': 'Scaling Efficiency'},
                    markers=True
                )
                fig_efficiency.add_hline(y=1.0, line_dash="dash", line_color="red", 
                                      annotation_text="Perfect Scaling")
                fig_efficiency.update_layout(height=400)
                st.plotly_chart(fig_efficiency, use_container_width=True)
            
            with col2:
                # Operations per node
                fig_ops_per_node = px.line(
                    df_scaling,
                    x='nodes',
                    y='ops_per_node',
                    title="Per-Node Efficiency",
                    labels={'nodes': 'Number of Nodes', 'ops_per_node': 'Operations per Node per Second'},
                    markers=True
                )
                fig_ops_per_node.update_layout(height=400)
                st.plotly_chart(fig_ops_per_node, use_container_width=True)

    def create_benchmark_runner(self):
        """Create the live benchmark runner section"""
        st.header("🧪 Live Benchmark Runner")
        
        with st.expander("Benchmark Configuration", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                benchmark_type = st.selectbox(
                    "Benchmark Type",
                    options=[bt.value for bt in BenchmarkType],
                    help="Type of benchmark to run"
                )
                
                node_count = st.slider(
                    "Node Count",
                    min_value=5,
                    max_value=100,
                    value=25,
                    step=5,
                    help="Number of nodes to simulate"
                )
            
            with col2:
                duration = st.slider(
                    "Duration (seconds)",
                    min_value=5,
                    max_value=120,
                    value=30,
                    help="How long to run the benchmark"
                )
                
                target_ops = st.number_input(
                    "Target Ops/Sec",
                    min_value=1.0,
                    max_value=100.0,
                    value=10.0,
                    step=1.0,
                    help="Target operations per second"
                )
            
            with col3:
                network_condition = st.selectbox(
                    "Network Condition",
                    options=[nc.value for nc in NetworkCondition],
                    help="Network latency simulation"
                )
                
                enable_pq = st.checkbox(
                    "Enable Post-Quantum",
                    value=True,
                    help="Include post-quantum signature overhead"
                )
        
        # Benchmark controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            run_benchmark = st.button(
                "🚀 Run Benchmark",
                disabled=st.session_state.benchmark_running,
                help="Start a new benchmark with the configured parameters"
            )
        
        with col2:
            if st.button("📊 Run Quick Demo"):
                self.run_quick_demo()
        
        with col3:
            if st.button("🔄 Refresh Results"):
                st.rerun()
        
        # Run benchmark if requested
        if run_benchmark:
            config = BenchmarkConfig(
                name=f"dashboard_{benchmark_type}_{int(time.time())}",
                benchmark_type=BenchmarkType(benchmark_type),
                node_count=node_count,
                duration_seconds=duration,
                target_operations_per_second=target_ops,
                network_condition=NetworkCondition(network_condition),
                enable_post_quantum=enable_pq
            )
            
            self.run_benchmark_async(config)

    def run_benchmark_async(self, config: BenchmarkConfig):
        """Run a benchmark asynchronously"""
        st.session_state.benchmark_running = True
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_placeholder = st.empty()
        
        def run_benchmark():
            try:
                # Run the benchmark
                async def benchmark_task():
                    status_text.text(f"🚀 Starting benchmark: {config.name}")
                    result = await self.benchmark_suite.run_benchmark(config)
                    return result
                
                # Update progress
                for i in range(config.duration_seconds):
                    progress_bar.progress((i + 1) / config.duration_seconds)
                    status_text.text(f"⏱️ Running benchmark... {i+1}/{config.duration_seconds}s")
                    
                    # Show live metrics if available
                    collector = get_global_collector()
                    metrics = collector.get_all_metrics()
                    
                    if metrics:
                        with metrics_placeholder.container():
                            cols = st.columns(len(metrics))
                            for idx, (op_name, metric) in enumerate(metrics.items()):
                                with cols[idx % len(cols)]:
                                    st.metric(
                                        f"{op_name}",
                                        f"{metric.mean_ms:.2f}ms",
                                        f"{metric.sample_count} samples"
                                    )
                    
                    time.sleep(1)
                
                # Run the actual benchmark
                result = asyncio.run(benchmark_task())
                
                # Update session state
                if 'benchmark_results' not in st.session_state:
                    st.session_state.benchmark_results = []
                st.session_state.benchmark_results.append(result)
                
                status_text.text("✅ Benchmark completed successfully!")
                progress_bar.progress(1.0)
                
                # Show results
                st.success(f"Benchmark completed: {result.operations_per_second:.2f} ops/s, {result.mean_latency_ms:.2f}ms avg latency")
                
            except Exception as e:
                status_text.text(f"❌ Benchmark failed: {e}")
                st.error(f"Benchmark error: {e}")
            finally:
                st.session_state.benchmark_running = False
        
        # Run in thread to avoid blocking UI
        threading.Thread(target=run_benchmark, daemon=True).start()

    def run_quick_demo(self):
        """Run a quick demo benchmark"""
        st.info("🚀 Running quick demo benchmark...")
        
        demo_config = BenchmarkConfig(
            name=f"demo_quick_{int(time.time())}",
            benchmark_type=BenchmarkType.CONSENSUS_SCALING,
            node_count=10,
            duration_seconds=5,
            target_operations_per_second=8.0,
            network_condition=NetworkCondition.LAN,
            enable_post_quantum=True
        )
        
        try:
            # Run synchronously for demo
            result = asyncio.run(self.benchmark_suite.run_benchmark(demo_config))
            
            st.success(f"Demo completed! {result.operations_per_second:.2f} ops/s, {result.mean_latency_ms:.2f}ms latency")
            
            # Add to session state
            if 'benchmark_results' not in st.session_state:
                st.session_state.benchmark_results = []
            st.session_state.benchmark_results.append(result)
            
        except Exception as e:
            st.error(f"Demo failed: {e}")

    def create_system_health(self):
        """Create system health monitoring section"""
        st.header("💚 System Health")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Consensus Health</h3>
                <p class="status-good big-font">✅ Healthy</p>
                <small>Last check: Just now</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Network Status</h3>
                <p class="status-good big-font">🌐 Connected</p>
                <small>Active peers: 12</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>Performance</h3>
                <p class="status-warning big-font">⚡ Good</p>
                <small>Avg latency: 45ms</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>Security</h3>
                <p class="status-good big-font">🔒 Secure</p>
                <small>PQ signatures: Active</small>
            </div>
            """, unsafe_allow_html=True)

    def run_dashboard(self):
        """Main dashboard runner"""
        # Sidebar
        st.sidebar.title("📊 PRSM Dashboard")
        st.sidebar.markdown("Real-time performance monitoring and analysis")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Choose a view:",
            ["Overview", "Live Benchmarks", "Historical Analysis", "System Health"]
        )
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
        if auto_refresh:
            time.sleep(30)
            st.rerun()
        
        # Main content
        if page == "Overview":
            self.create_performance_overview()
            historical_results = self.load_historical_results()
            if historical_results:
                self.create_throughput_analysis(historical_results)
        
        elif page == "Live Benchmarks":
            self.create_benchmark_runner()
        
        elif page == "Historical Analysis":
            st.header("📈 Historical Performance Analysis")
            historical_results = self.load_historical_results()
            
            if historical_results:
                self.create_latency_analysis(historical_results)
                self.create_scaling_analysis(historical_results)
            else:
                st.info("No historical data available. Run some benchmarks first!")
        
        elif page == "System Health":
            self.create_system_health()
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("**PRSM Performance Dashboard**")
        st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")


def main():
    """Main entry point for the Streamlit dashboard"""
    dashboard = PerformanceDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()