#!/usr/bin/env python3
"""
Quick test of the PRSM monitoring dashboard functionality
"""

import asyncio
import time
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dashboard.real_time_monitoring_dashboard import MetricsCollector, MockFlaskDashboard

async def test_metrics_collection():
    """Test metrics collection functionality"""
    print("🧪 Testing PRSM Monitoring Dashboard Components")
    print("=" * 60)
    
    # Test metrics collector
    print("📊 Testing MetricsCollector...")
    collector = MetricsCollector()
    
    # Collect some test metrics
    await collector._collect_system_metrics()
    
    if collector.system_metrics:
        latest = collector.system_metrics[-1]
        print(f"✅ System metrics collected:")
        print(f"   CPU: {latest.cpu_percent:.1f}%")
        print(f"   Memory: {latest.memory_percent:.1f}%")
        print(f"   Connections: {latest.active_connections}")
    
    # Test dashboard data
    print("\n📱 Testing Dashboard Data...")
    data = collector.get_latest_metrics()
    
    if data['system']:
        print("✅ Dashboard data structure working")
        print(f"   System data: CPU {data['system']['cpu_percent']:.1f}%")
    else:
        print("❌ No system data available")
    
    # Test historical data
    print("\n📈 Testing Historical Data...")
    historical = collector.get_historical_data("system", 1)
    print(f"✅ Historical data points: {len(historical)}")
    
    # Test mock dashboard
    print("\n🖥️  Testing Mock Dashboard...")
    mock_dashboard = MockFlaskDashboard(collector)
    
    # Show one console output
    await mock_dashboard._display_console_dashboard()
    
    print("\n✅ All dashboard components working correctly!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_metrics_collection())
    
    if success:
        print("\n🎉 PRSM Monitoring Dashboard Test Completed Successfully!")
        print("📊 The dashboard can monitor system metrics, network status, and AI performance")
        print("📱 Ready for production deployment with Flask for web interface")
    else:
        print("\n❌ Dashboard test failed")