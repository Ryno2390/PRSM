#!/usr/bin/env python3
"""
Hello PRSM - Your First PRSM Program
This example demonstrates basic PRSM setup and API usage.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

def main():
    """Main function demonstrating basic PRSM concepts"""
    print("🚀 Hello PRSM!")
    print("=" * 40)
    
    print("🔍 Basic PRSM Information:")
    print(f"   Project Root: {project_root}")
    print(f"   Python Version: {sys.version.split()[0]}")
    print(f"   Current Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📚 PRSM Core Concepts:")
    concepts = {
        "AI Agents": "Intelligent agents that can process tasks and make decisions",
        "P2P Network": "Distributed peer-to-peer network for collaboration",
        "Model Management": "Loading, running, and sharing AI models",
        "Orchestration": "Coordinating multiple agents for complex workflows",
        "Monitoring": "Real-time system and performance monitoring"
    }
    
    for concept, description in concepts.items():
        print(f"   🎯 {concept}: {description}")
    
    print("\n🛠️  Available PRSM Components:")
    try:
        # Try to import PRSM components
        from prsm.core.config import PRSMConfig
        print("   ✅ Core Configuration - Available")
    except ImportError:
        print("   ⚠️  Core Configuration - Install PRSM package")
    
    try:
        from prsm.agents.base import BaseAgent
        print("   ✅ AI Agents - Available")
    except ImportError:
        print("   ⚠️  AI Agents - Install PRSM package")
    
    try:
        from demos.p2p_network_demo import P2PNetworkDemo
        print("   ✅ P2P Network - Available")
    except ImportError:
        print("   ⚠️  P2P Network - Check demos directory")
    
    try:
        from demos.enhanced_p2p_ai_demo import EnhancedP2PNetworkDemo
        print("   ✅ Enhanced AI Network - Available")
    except ImportError:
        print("   ⚠️  Enhanced AI Network - Check demos directory")
    
    try:
        from dashboard.real_time_monitoring_dashboard import PRSMMonitoringDashboard
        print("   ✅ Monitoring Dashboard - Available")
    except ImportError:
        print("   ⚠️  Monitoring Dashboard - Check dashboard directory")
    
    print("\n🎯 Quick Start Guide:")
    print("   1. Explore the examples in playground/examples/")
    print("   2. Try the interactive tutorials")
    print("   3. Build your first AI agent")
    print("   4. Set up a P2P network")
    print("   5. Monitor your applications")
    
    print("\n💡 Next Steps:")
    print("   • Run: python playground_launcher.py --example basic/simple_agent")
    print("   • Try: python playground_launcher.py --tutorial getting-started")
    print("   • Interactive: python playground_launcher.py --interactive")
    
    print("\n🎉 Welcome to the PRSM Developer Community!")
    print("   Visit: https://github.com/your-org/PRSM")
    print("   Docs: https://docs.prsm.ai")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ Hello PRSM completed successfully!")
        print("🚀 Ready to explore more examples and tutorials!")
    else:
        print("\n❌ Something went wrong. Check the setup and try again.")
        sys.exit(1)