#!/usr/bin/env python3
"""
PRSM Public Testnet Launcher
===========================

Easy launcher for the public testnet interface.
Addresses Gemini's recommendation for "Public Testnet and Governance Portal".
"""

import asyncio
import sys
import os
from pathlib import Path

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from prsm.public.testnet_interface import PRSMTestnetInterface, TestnetConfig
except ImportError as e:
    print(f"❌ Error importing testnet interface: {e}")
    print("💡 Try: pip install fastapi uvicorn")
    sys.exit(1)


async def main():
    """Launch the PRSM Public Testnet"""
    
    print("🌐 PRSM Public Testnet Launcher")
    print("=" * 60)
    print("🎯 Address Gemini recommendation: 'Public Testnet and Governance Portal'")
    print("🤖 Experience AI coordination and Recursive Learning Teachers")
    print("💰 Participate in FTNS token economy simulation")
    print("🚀 Free public access for community building")
    print("")
    
    # Configuration
    config = TestnetConfig(
        host="0.0.0.0",  # Allow external access
        port=8090,
        title="PRSM Public Testnet",
        max_queries_per_user=10,
        enable_demo_mode=True  # Safe simulated responses
    )
    
    # Create testnet
    testnet = PRSMTestnetInterface(config)
    
    print(f"🔗 Testnet URL: {testnet.get_testnet_url()}")
    print("🌟 Features Available:")
    print("   • Submit queries to PRSM AI coordination network")
    print("   • Experience Recursive Learning Teachers (RLT)")
    print("   • Multi-AI model coordination demonstration")
    print("   • FTNS token economy participation")
    print("   • Real-time network statistics")
    print("   • Interactive sample queries")
    print("")
    print("👥 Target Users:")
    print("   • Developers exploring AI coordination")
    print("   • Researchers interested in RLT systems")
    print("   • Community members testing PRSM capabilities")
    print("   • Investors evaluating technical functionality")
    print("")
    print("🚀 Starting public testnet interface...")
    print("📱 Access from any device on your network")
    print("⏹️  Press Ctrl+C to stop")
    print("")
    
    try:
        await testnet.start_testnet()
    except KeyboardInterrupt:
        print("\n🛑 Stopping PRSM Public Testnet...")
        await testnet.stop_testnet()
        print("✅ Testnet stopped successfully")
    except Exception as e:
        print(f"❌ Error running testnet: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check for FastAPI
    try:
        import fastapi
        import uvicorn
    except ImportError:
        print("❌ Missing dependencies for testnet interface")
        print("💡 Install with: pip install fastapi uvicorn")
        sys.exit(1)
    
    asyncio.run(main())