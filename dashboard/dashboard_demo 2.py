#!/usr/bin/env python3
"""
PRSM Dashboard Demo
Quick demonstration of the real-time monitoring dashboard capabilities.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dashboard.real_time_monitoring_dashboard import PRSMMonitoringDashboard

async def run_dashboard_demo():
    """Run a quick dashboard demonstration"""
    print("🚀 PRSM Real-Time Monitoring Dashboard Demo")
    print("=" * 50)
    print("📊 Starting dashboard with demo P2P network...")
    print("🌐 Dashboard will be available at: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the demo")
    print()
    
    dashboard = PRSMMonitoringDashboard()
    
    try:
        await dashboard.start_monitoring(
            host="localhost",
            port=5000,
            with_demo=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"❌ Demo error: {str(e)}")
        print("📊 Running console-only version...")
        
        # Fallback to console version
        try:
            await dashboard.start_monitoring(
                host="localhost",
                port=5000,
                with_demo=True
            )
        except KeyboardInterrupt:
            print("\n🛑 Demo stopped by user")

if __name__ == "__main__":
    asyncio.run(run_dashboard_demo())