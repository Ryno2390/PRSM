#!/usr/bin/env python3
"""
PRSM Governance Portal Launcher
==============================

Easy launcher for the governance portal.
Completes Gemini's recommendation for "Public Testnet and Governance Portal".
"""

import asyncio
import sys
import os
from pathlib import Path

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from prsm.public.governance_portal import PRSMGovernancePortal, GovernanceConfig
except ImportError as e:
    print(f"❌ Error importing governance portal: {e}")
    print("💡 Try: pip install fastapi uvicorn")
    sys.exit(1)


async def main():
    """Launch the PRSM Governance Portal"""
    
    print("🏛️ PRSM Governance Portal Launcher")
    print("=" * 60)
    print("🎯 Complete Gemini recommendation: 'Public Testnet and Governance Portal'")
    print("🗳️ Democratic decision-making for PRSM development")
    print("📋 Community-driven proposal submission and voting")
    print("💰 FTNS token-weighted governance system")
    print("")
    
    # Configuration
    config = GovernanceConfig(
        host="0.0.0.0",  # Allow external access
        port=8095,
        title="PRSM Governance Portal",
        voting_period_days=7,
        quorum_threshold=0.1,
        passing_threshold=0.6
    )
    
    # Create governance portal
    governance = PRSMGovernancePortal(config)
    
    print(f"🔗 Governance URL: {governance.get_governance_url()}")
    print("🌟 Features Available:")
    print("   • Democratic proposal submission and voting")
    print("   • FTNS token-weighted governance system")
    print("   • Real-time voting results and statistics")
    print("   • Multiple proposal types (Technical, Economic, Governance, Community)")
    print("   • Transparent implementation tracking")
    print("   • Community participation metrics")
    print("")
    print("📊 Sample Proposals Included:")
    print("   • Technical: RLT Caching System Implementation")
    print("   • Economic: FTNS Token Economics Adjustment")
    print("   • Governance: Community Advisory Board Establishment")
    print("   • Implemented: Additional AI Models Integration (Completed)")
    print("")
    print("👥 Target Users:")
    print("   • Community members participating in governance")
    print("   • Developers proposing technical improvements")
    print("   • Token holders exercising voting rights")
    print("   • Stakeholders tracking development priorities")
    print("")
    print("🚀 Starting governance portal...")
    print("📱 Access from any device on your network")
    print("⏹️  Press Ctrl+C to stop")
    print("")
    
    try:
        await governance.start_governance()
    except KeyboardInterrupt:
        print("\n🛑 Stopping PRSM Governance Portal...")
        await governance.stop_governance()
        print("✅ Governance portal stopped successfully")
    except Exception as e:
        print(f"❌ Error running governance portal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check for FastAPI
    try:
        import fastapi
        import uvicorn
    except ImportError:
        print("❌ Missing dependencies for governance portal")
        print("💡 Install with: pip install fastapi uvicorn")
        sys.exit(1)
    
    asyncio.run(main())