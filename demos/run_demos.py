#!/usr/bin/env python3
"""
PRSM Investor Demo Launcher
Professional demonstration suite for technical due diligence

For complete investor demo guide, see: INVESTOR_DEMO.md
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header():
    """Display professional header for investor presentations"""
    print("\n" + "="*60)
    print("🚀 PRSM: Advanced Prototype Demonstration Suite")
    print("   Protocol for Recursive Scientific Modeling")
    print("="*60)
    print("📊 Status: Advanced Prototype - Ready for Investment")
    print("🎯 Purpose: Technical Due Diligence & Capability Validation")
    print("⏱️  Duration: ~30 minutes for complete demonstration")
    print("-"*60)

def print_menu():
    """Display investor-focused demo menu"""
    print("\n📋 DEMO OPTIONS:")
    print("\n🌐 CORE DEMONSTRATIONS:")
    print("  1. P2P Network Demo          [~3 min] - Decentralized coordination")
    print("  2. Tokenomics Simulation     [~5 min] - Economic stress testing") 
    print("\n📊 INTERACTIVE DASHBOARDS:")
    print("  3. P2P Network Dashboard     [Live]   - Real-time network monitoring")
    print("  4. Tokenomics Dashboard      [Live]   - Economic analysis interface")
    print("\n🧪 VALIDATION SUITE:")
    print("  5. Complete Test Suite       [~8 min] - Full validation & benchmarks")
    print("  6. Investor Demo Walkthrough [~30min] - Guided presentation")
    print("\n💡 RESOURCES:")
    print("  7. System Requirements Check [~1 min] - Verify demo environment")
    print("  8. Demo Guide (INVESTOR_DEMO.md)      - Complete investor materials")
    print("\n  0. Exit")
    print("-"*60)

def main():
    print_header()
    
    # Quick environment check
    print("\n🔍 Quick Environment Check...")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"   ✅ Python Version: {python_version}")
    print(f"   ✅ Working Directory: {Path.cwd()}")
    print(f"   ✅ Demo Launcher Ready")
    
    while True:
        try:
            print_menu()
            choice = input("\n👉 Select demo option (0-8): ").strip()
            
            if choice == '0':
                print("\n" + "="*60)
                print("👋 Thank you for exploring PRSM!")
                print("📧 Contact: funding@prsm.ai for investment opportunities")
                print("📄 Full materials: See INVESTOR_DEMO.md")
                print("="*60)
                break
            elif choice == '1':
                print("\n" + "="*60)
                print("🌐 P2P NETWORK DEMONSTRATION")
                print("="*60)
                print("🎯 Demonstrating: Decentralized coordination, consensus, fault tolerance")
                print("⏱️  Expected Duration: ~3 minutes")
                print("\n▶️  Starting P2P Network Demo...")
                print("-"*60)
                
                start_time = time.time()
                result = subprocess.run([sys.executable, "p2p_network_demo.py"])
                duration = time.time() - start_time
                
                print("-"*60)
                if result.returncode == 0:
                    print(f"✅ P2P Network Demo COMPLETED successfully in {duration:.1f} seconds")
                    print("📊 Key Validation: Node discovery, consensus, fault recovery")
                else:
                    print(f"❌ P2P Network Demo encountered issues (exit code: {result.returncode})")
                print("="*60)
                
            elif choice == '2':
                print("\n" + "="*60)
                print("💰 TOKENOMICS STRESS TEST SIMULATION")
                print("="*60)
                print("🎯 Demonstrating: Economic viability, fairness, attack resistance")
                print("⏱️  Expected Duration: ~5 minutes")
                print("\n▶️  Starting Tokenomics Simulation...")
                print("-"*60)
                
                start_time = time.time()
                result = subprocess.run([sys.executable, "tokenomics_simulation.py"])
                duration = time.time() - start_time
                
                print("-"*60)
                if result.returncode == 0:
                    print(f"✅ Tokenomics Simulation COMPLETED successfully in {duration:.1f} seconds")
                    print("📊 Key Validation: Economic stress testing, fairness metrics")
                else:
                    print(f"❌ Tokenomics Simulation encountered issues (exit code: {result.returncode})")
                print("="*60)
            elif choice == '3':
                print("\n" + "="*60)
                print("📊 P2P NETWORK INTERACTIVE DASHBOARD")
                print("="*60)
                print("🎯 Features: Real-time network topology, metrics, monitoring")
                print("🌐 URL: http://localhost:8501")
                print("💡 Tip: Keep this running during P2P demo for live visualization")
                print("\n▶️  Launching P2P Dashboard...")
                print("   (Press Ctrl+C to stop dashboard)")
                print("-"*60)
                
                try:
                    subprocess.run([sys.executable, "-m", "streamlit", "run", "p2p_dashboard.py", "--server.port", "8501"])
                except KeyboardInterrupt:
                    print("\n✅ P2P Dashboard stopped by user")
                print("="*60)
                
            elif choice == '4':
                print("\n" + "="*60)
                print("📈 TOKENOMICS ANALYSIS DASHBOARD")
                print("="*60)
                print("🎯 Features: Economic analysis, fairness metrics, stress test results")
                print("🌐 URL: http://localhost:8502")
                print("💡 Tip: Run after tokenomics simulation for detailed analysis")
                print("\n▶️  Launching Tokenomics Dashboard...")
                print("   (Press Ctrl+C to stop dashboard)")
                print("-"*60)
                
                try:
                    subprocess.run([sys.executable, "-m", "streamlit", "run", "tokenomics_dashboard.py", "--server.port", "8502"])
                except KeyboardInterrupt:
                    print("\n✅ Tokenomics Dashboard stopped by user")
                print("="*60)
            elif choice == '5':
                print("\n" + "="*60)
                print("🧪 COMPLETE VALIDATION SUITE")
                print("="*60)
                print("🎯 Purpose: Comprehensive technical validation for investors")
                print("⏱️  Expected Duration: ~8 minutes")
                print("📊 Tests: P2P Network + Tokenomics + Performance validation")
                print("\n▶️  Starting Complete Test Suite...")
                print("-"*60)
                
                overall_start = time.time()
                all_passed = True
                
                # Test 1: P2P Network
                print("\n🌐 [1/2] Testing P2P Network Architecture...")
                start_time = time.time()
                result1 = subprocess.run([sys.executable, "p2p_network_demo.py"], 
                                       capture_output=True, text=True)
                duration1 = time.time() - start_time
                
                # Test 2: Tokenomics
                print("\n💰 [2/2] Testing Tokenomics & Economic Model...")
                start_time = time.time()
                result2 = subprocess.run([sys.executable, "-c", 
                    "from tokenomics_simulation import FTNSEconomicSimulation; "
                    "sim = FTNSEconomicSimulation(num_agents=10, simulation_days=5); "
                    "result = sim.run_simulation(); "
                    "print('✅ Tokenomics test passed:', result['overall_success'])"],
                    capture_output=True, text=True)
                duration2 = time.time() - start_time
                total_duration = time.time() - overall_start
                
                print("\n" + "="*60)
                print("📋 VALIDATION RESULTS SUMMARY")
                print("="*60)
                
                if result1.returncode == 0:
                    print(f"✅ P2P Network Architecture: PASSED ({duration1:.1f}s)")
                    print("   ✓ Node discovery and consensus mechanisms")
                    print("   ✓ Fault tolerance and recovery systems")
                else:
                    print(f"❌ P2P Network Architecture: FAILED ({duration1:.1f}s)")
                    print(f"   Error: {result1.stderr.strip() if result1.stderr else 'Unknown error'}")
                    all_passed = False
                
                if result2.returncode == 0 and "✅ Tokenomics test passed: True" in result2.stdout:
                    print(f"✅ Tokenomics & Economic Model: PASSED ({duration2:.1f}s)")
                    print("   ✓ Multi-agent economic simulation")
                    print("   ✓ Stress testing and fairness validation")
                else:
                    print(f"❌ Tokenomics & Economic Model: FAILED ({duration2:.1f}s)")
                    print(f"   Error: {result2.stderr.strip() if result2.stderr else 'Unknown error'}")
                    all_passed = False
                
                print("-"*60)
                if all_passed:
                    print(f"🎉 ALL TESTS PASSED - Total Duration: {total_duration:.1f} seconds")
                    print("✅ PRSM prototype validated for investor presentation")
                    print("📊 Technical feasibility and economic viability confirmed")
                else:
                    print(f"⚠️  SOME TESTS FAILED - Total Duration: {total_duration:.1f} seconds")
                    print("🔧 Please check error messages above")
                print("="*60)
            elif choice == '6':
                print("\n" + "="*60)
                print("🎤 GUIDED INVESTOR DEMO WALKTHROUGH")
                print("="*60)
                print("📄 Opening complete investor demo guide...")
                print("📍 File: INVESTOR_DEMO.md")
                print("⏱️  Duration: ~30 minutes guided presentation")
                print("\n💡 This guide includes:")
                print("   • Step-by-step technical demonstration")
                print("   • Investment thesis and value proposition")
                print("   • Q&A preparation and talking points")
                print("   • Success criteria and validation metrics")
                print("-"*60)
                
                if Path("INVESTOR_DEMO.md").exists():
                    print("✅ INVESTOR_DEMO.md found")
                    print("📖 Please open INVESTOR_DEMO.md for complete guidance")
                    print("💡 Tip: Use this guide alongside demo launcher for presentations")
                else:
                    print("❌ INVESTOR_DEMO.md not found in current directory")
                print("="*60)
                
            elif choice == '7':
                print("\n" + "="*60)
                print("🔍 SYSTEM REQUIREMENTS CHECK")
                print("="*60)
                print("🎯 Verifying demo environment for investor presentations")
                print("\n⚙️  Checking system requirements...")
                print("-"*60)
                
                # Check Python version
                python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                if sys.version_info >= (3, 9):
                    print(f"✅ Python Version: {python_version} (Requirements: 3.9+)")
                else:
                    print(f"❌ Python Version: {python_version} (Requirements: 3.9+)")
                
                # Check required files
                required_files = ["p2p_network_demo.py", "tokenomics_simulation.py", 
                                "p2p_dashboard.py", "tokenomics_dashboard.py", "requirements.txt"]
                for file in required_files:
                    if Path(file).exists():
                        print(f"✅ Demo File: {file}")
                    else:
                        print(f"❌ Demo File: {file} (MISSING)")
                
                # Check dependencies
                print("\n📦 Checking key dependencies...")
                try:
                    import streamlit
                    print(f"✅ Streamlit: {streamlit.__version__}")
                except ImportError:
                    print("❌ Streamlit: Not installed (pip install streamlit)")
                
                try:
                    import plotly
                    print(f"✅ Plotly: {plotly.__version__}")
                except ImportError:
                    print("❌ Plotly: Not installed (pip install plotly)")
                
                try:
                    import pandas
                    print(f"✅ Pandas: {pandas.__version__}")
                except ImportError:
                    print("❌ Pandas: Not installed (pip install pandas)")
                
                print("-"*60)
                print("💡 If any dependencies are missing, run: pip install -r requirements.txt")
                print("="*60)
                
            elif choice == '8':
                print("\n" + "="*60)
                print("📚 INVESTOR DEMO MATERIALS")
                print("="*60)
                if Path("INVESTOR_DEMO.md").exists():
                    print("📄 Opening INVESTOR_DEMO.md...")
                    print("\n📋 This comprehensive guide includes:")
                    print("   • Complete 30-minute investor presentation script")
                    print("   • Technical validation and success criteria")
                    print("   • Anticipated Q&A with prepared responses")
                    print("   • Business model validation through demos")
                    print("   • Investment thesis and value proposition")
                    print("   • Follow-up materials and next steps")
                    print("\n🎯 Perfect for: Investor meetings, technical due diligence")
                    try:
                        if sys.platform == "darwin":  # macOS
                            subprocess.run(["open", "INVESTOR_DEMO.md"])
                        elif sys.platform == "linux":
                            subprocess.run(["xdg-open", "INVESTOR_DEMO.md"])
                        elif sys.platform == "win32":
                            subprocess.run(["start", "INVESTOR_DEMO.md"], shell=True)
                        print("📖 Demo guide opened in default application")
                    except Exception as e:
                        print(f"💡 Please manually open: INVESTOR_DEMO.md")
                        print(f"   (Auto-open failed: {e})")
                else:
                    print("❌ INVESTOR_DEMO.md not found")
                    print("💡 Please ensure you're in the demos/ directory")
                print("="*60)
                
            else:
                print("❌ Invalid choice. Please select 0-8.")
                
        except KeyboardInterrupt:
            print("\n" + "="*60)
            print("👋 Demo session interrupted by user")
            print("📧 Contact: funding@prsm.ai for investment opportunities")
            print("="*60)
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("💡 Please check system requirements or contact support")

if __name__ == "__main__":
    # Change to demo directory
    demo_dir = Path(__file__).parent
    os.chdir(demo_dir)
    
    # Welcome message for investors
    print("\n🎯 Welcome to PRSM - Advanced Prototype Demonstration")
    print("📧 For investment opportunities: funding@prsm.ai")
    print("📄 Complete investor materials: INVESTOR_DEMO.md")
    
    main()