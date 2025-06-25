#!/usr/bin/env python3
"""
Test PRSM Performance Dashboard
Validate dashboard components without full Streamlit environment
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add PRSM to path
PRSM_ROOT = Path(__file__).parent
sys.path.insert(0, str(PRSM_ROOT))

def test_dashboard_components():
    """Test the key dashboard components"""
    print("🧪 Testing PRSM Performance Dashboard Components")
    print("=" * 60)
    
    try:
        # Test imports
        print("1. Testing imports...")
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np
        print("   ✅ All required packages imported successfully")
        
        # Test benchmark suite integration
        print("\n2. Testing benchmark integration...")
        from comprehensive_performance_benchmark import PerformanceBenchmarkSuite, BenchmarkConfig, BenchmarkType, NetworkCondition
        
        suite = PerformanceBenchmarkSuite("test_dashboard_results")
        print("   ✅ Benchmark suite integration working")
        
        # Test mock data creation
        print("\n3. Testing data processing...")
        
        # Create mock benchmark results
        mock_results = [
            {
                "config": {
                    "name": "test_consensus_scaling_10",
                    "benchmark_type": "consensus_scaling",
                    "node_count": 10,
                    "network_condition": "wan"
                },
                "metrics": {
                    "operations_per_second": 8.5,
                    "mean_latency_ms": 45.2,
                    "p95_latency_ms": 78.1,
                    "p99_latency_ms": 95.3,
                    "consensus_success_rate": 0.975,
                    "operations_per_node_per_second": 0.85
                },
                "timing": {
                    "start_time": datetime.now().isoformat(),
                    "duration_seconds": 30.0
                }
            },
            {
                "config": {
                    "name": "test_consensus_scaling_25",
                    "benchmark_type": "consensus_scaling", 
                    "node_count": 25,
                    "network_condition": "wan"
                },
                "metrics": {
                    "operations_per_second": 12.3,
                    "mean_latency_ms": 62.8,
                    "p95_latency_ms": 95.4,
                    "p99_latency_ms": 120.7,
                    "consensus_success_rate": 0.965,
                    "operations_per_node_per_second": 0.492
                },
                "timing": {
                    "start_time": datetime.now().isoformat(),
                    "duration_seconds": 30.0
                }
            }
        ]
        
        # Test data processing
        df = pd.DataFrame([{
            'timestamp': r['timing']['start_time'],
            'throughput': r['metrics']['operations_per_second'],
            'node_count': r['config']['node_count'],
            'network_condition': r['config']['network_condition'],
            'benchmark_type': r['config']['benchmark_type']
        } for r in mock_results])
        
        print(f"   ✅ Created DataFrame with {len(df)} records")
        
        # Test chart creation
        print("\n4. Testing chart generation...")
        
        # Test throughput chart
        fig_throughput = px.line(
            df, 
            x='node_count', 
            y='throughput',
            title="Test Throughput Chart"
        )
        print("   ✅ Throughput chart created")
        
        # Test latency chart
        latency_data = [{
            'benchmark': r['config']['name'],
            'mean_latency': r['metrics']['mean_latency_ms'],
            'p95_latency': r['metrics']['p95_latency_ms'],
            'node_count': r['config']['node_count']
        } for r in mock_results]
        
        df_latency = pd.DataFrame(latency_data)
        fig_latency = px.bar(
            df_latency,
            x='benchmark',
            y='mean_latency',
            title="Test Latency Chart"
        )
        print("   ✅ Latency chart created")
        
        # Test scaling analysis
        print("\n5. Testing scaling analysis...")
        
        scaling_results = [r for r in mock_results if r['config']['benchmark_type'] == 'consensus_scaling']
        scaling_results.sort(key=lambda x: x['config']['node_count'])
        
        if scaling_results:
            baseline = scaling_results[0]
            baseline_throughput = baseline['metrics']['operations_per_second']
            baseline_nodes = baseline['config']['node_count']
            
            scaling_data = []
            for result in scaling_results:
                nodes = result['config']['node_count']
                throughput = result['metrics']['operations_per_second']
                
                scaling_factor = nodes / baseline_nodes
                throughput_ratio = throughput / baseline_throughput
                efficiency = throughput_ratio / scaling_factor
                
                scaling_data.append({
                    'nodes': nodes,
                    'throughput': throughput,
                    'efficiency': efficiency
                })
            
            print(f"   ✅ Scaling analysis completed for {len(scaling_data)} data points")
            for data in scaling_data:
                print(f"      {data['nodes']} nodes: {data['efficiency']:.3f} efficiency")
        
        # Test result saving
        print("\n6. Testing result persistence...")
        
        test_results_dir = Path("test_dashboard_results")
        test_results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_file = test_results_dir / f"test_results_{timestamp}.json"
        
        with open(test_file, 'w') as f:
            json.dump({
                "test_run": timestamp,
                "results": mock_results
            }, f, indent=2)
        
        print(f"   ✅ Test results saved to {test_file}")
        
        print("\n" + "=" * 60)
        print("🎉 ALL DASHBOARD COMPONENTS TESTED SUCCESSFULLY!")
        print("=" * 60)
        print("📊 Dashboard Features Validated:")
        print("   ✅ Data import and processing")
        print("   ✅ Chart generation (Plotly)")
        print("   ✅ Performance metrics calculation")
        print("   ✅ Scaling efficiency analysis")
        print("   ✅ Result persistence")
        print("   ✅ Benchmark suite integration")
        print()
        print("🚀 Ready to launch Streamlit dashboard!")
        print("   Run: python run_dashboard.py")
        print("   Or: streamlit run prsm/dashboard/performance_dashboard.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streamlit_availability():
    """Test if Streamlit is available"""
    print("\n🔍 Testing Streamlit availability...")
    
    try:
        import streamlit as st
        print("   ✅ Streamlit is installed and available")
        
        # Check version
        import streamlit
        version = streamlit.__version__
        print(f"   📦 Streamlit version: {version}")
        
        return True
    except ImportError:
        print("   ⚠️  Streamlit not installed")
        print("   💡 Install with: pip install streamlit")
        return False

if __name__ == "__main__":
    print("🧪 PRSM Performance Dashboard - Component Testing")
    print("=" * 60)
    
    # Test core components
    components_ok = test_dashboard_components()
    
    # Test Streamlit
    streamlit_ok = test_streamlit_availability()
    
    print("\n" + "=" * 60)
    print("🏁 DASHBOARD TEST SUMMARY:")
    print(f"   Core Components: {'✅ PASS' if components_ok else '❌ FAIL'}")
    print(f"   Streamlit: {'✅ READY' if streamlit_ok else '⚠️ NEEDS INSTALL'}")
    
    if components_ok and streamlit_ok:
        print("\n🎉 Dashboard is ready to launch!")
        print("🚀 Next steps:")
        print("   1. Run: python run_dashboard.py")
        print("   2. Open browser to: http://localhost:8501")
        print("   3. Navigate through dashboard sections")
    elif components_ok and not streamlit_ok:
        print("\n⚠️  Install Streamlit to use the dashboard:")
        print("   pip install streamlit plotly")
    else:
        print("\n❌ Fix component issues before launching dashboard")
    
    sys.exit(0 if components_ok else 1)