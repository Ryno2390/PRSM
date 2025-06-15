#!/usr/bin/env python3
"""
PRSM Demo Launcher
Quick launcher for both P2P Network and Tokenomics demonstrations
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🚀 PRSM Demo Suite Launcher")
    print("=" * 40)
    print("1. P2P Network Demo")
    print("2. Tokenomics Simulation") 
    print("3. P2P Dashboard (Streamlit)")
    print("4. Tokenomics Dashboard (Streamlit)")
    print("5. Run All Tests")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nSelect demo (0-5): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                print("\n🌐 Running P2P Network Demo...")
                subprocess.run([sys.executable, "p2p_network_demo.py"])
            elif choice == '2':
                print("\n💰 Running Tokenomics Simulation...")
                subprocess.run([sys.executable, "tokenomics_simulation.py"])
            elif choice == '3':
                print("\n📊 Launching P2P Dashboard...")
                print("Opening http://localhost:8501 in your browser...")
                subprocess.run([sys.executable, "-m", "streamlit", "run", "p2p_dashboard.py", "--server.port", "8501"])
            elif choice == '4':
                print("\n📈 Launching Tokenomics Dashboard...")
                print("Opening http://localhost:8502 in your browser...")
                subprocess.run([sys.executable, "-m", "streamlit", "run", "tokenomics_dashboard.py", "--server.port", "8502"])
            elif choice == '5':
                print("\n🧪 Running All Tests...")
                print("Testing P2P Network...")
                result1 = subprocess.run([sys.executable, "p2p_network_demo.py"], 
                                       capture_output=True, text=True)
                
                print("Testing Tokenomics...")
                result2 = subprocess.run([sys.executable, "-c", 
                    "from tokenomics_simulation import FTNSEconomicSimulation; "
                    "sim = FTNSEconomicSimulation(num_agents=10, simulation_days=5); "
                    "result = sim.run_simulation(); "
                    "print('✅ Tokenomics test passed:', result['overall_success'])"],
                    capture_output=True, text=True)
                
                print("\n📋 Test Results:")
                if result1.returncode == 0:
                    print("✅ P2P Network Demo: PASSED")
                else:
                    print("❌ P2P Network Demo: FAILED")
                    print(result1.stderr)
                
                if result2.returncode == 0 and "✅ Tokenomics test passed: True" in result2.stdout:
                    print("✅ Tokenomics Simulation: PASSED")
                else:
                    print("❌ Tokenomics Simulation: FAILED")
                    print(result2.stderr)
            else:
                print("❌ Invalid choice. Please select 0-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Change to demo directory
    demo_dir = Path(__file__).parent
    os.chdir(demo_dir)
    
    main()