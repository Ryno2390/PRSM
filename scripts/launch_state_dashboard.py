#!/usr/bin/env python3
"""
PRSM State of Network Dashboard Launcher
=======================================

Simple launcher for the public State of the Network dashboard.
Addresses Gemini's recommendation for user-facing experience.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from prsm.public.state_of_network_dashboard import StateOfNetworkDashboard, DashboardConfig
except ImportError as e:
    print(f"❌ Error importing dashboard: {e}")
    print("💡 Try: pip install fastapi uvicorn")
    sys.exit(1)


async def main():
    """Launch the State of Network dashboard"""
    
    print("🌐 PRSM State of the Network Dashboard")
    print("=" * 60)
    print("📊 Public transparency dashboard for stakeholders and investors")
    print("🎯 Addresses Gemini recommendation for user-facing experience")
    print("")
    
    # Configuration
    config = DashboardConfig(
        host="0.0.0.0",  # Allow external access
        port=8081,
        title="PRSM State of the Network",
        update_interval=30
    )
    
    # Create dashboard
    dashboard = StateOfNetworkDashboard(config)
    
    print(f"🔗 Dashboard URL: {dashboard.get_dashboard_url()}")
    print("📈 Investment readiness metrics")
    print("🏥 Real-time network health status")
    print("🔍 Evidence generation transparency")
    print("🏆 Recent achievements showcase")
    print("")
    print("🚀 Starting dashboard server...")
    print("📱 Access from any device on your network")
    print("⏹️  Press Ctrl+C to stop")
    print("")
    
    try:
        await dashboard.start_dashboard()
    except KeyboardInterrupt:
        print("\n🛑 Stopping State of Network dashboard...")
        await dashboard.stop_dashboard()
        print("✅ Dashboard stopped successfully")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check for FastAPI
    try:
        import fastapi
        import uvicorn
    except ImportError:
        print("❌ Missing dependencies for dashboard")
        print("💡 Install with: pip install fastapi uvicorn")
        sys.exit(1)
    
    asyncio.run(main())