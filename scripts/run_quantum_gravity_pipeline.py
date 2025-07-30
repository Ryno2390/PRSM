#!/usr/bin/env python3
"""
Run Quantum Gravity Query Through Existing NWTN Pipeline
========================================================

Uses the existing general-purpose NWTN pipeline (scripts/run_nwtn_pipeline.py)
with the quantum gravity query and appropriate parameters.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the existing pipeline runner
from scripts.run_nwtn_pipeline import run_nwtn_pipeline

async def main():
    """Run quantum gravity query through existing NWTN pipeline"""
    
    print("🔬 QUANTUM GRAVITY THROUGH EXISTING NWTN PIPELINE")
    print("=" * 60)
    print("📋 Using general-purpose pipeline infrastructure")
    print("🎯 Parameters: DEEP reasoning, STANDARD verbosity, REVOLUTIONARY mode")
    print("=" * 60)
    print()
    
    # Your quantum gravity query
    query = ("What are the most promising theoretical approaches to unifying quantum mechanics "
            "and general relativity, and what experimental evidence exists to support or refute these approaches?")
    
    # Parameters based on your stress testing requirements
    depth = "DEEP"              # 5,040-iteration deep reasoning
    verbosity = "STANDARD"      # Standard length response as requested
    breakthrough_mode = "REVOLUTIONARY"  # One of the extreme modes you wanted to test
    
    print(f"📝 Query: {query}")
    print(f"🔬 Reasoning Depth: {depth}")
    print(f"📄 Verbosity Level: {verbosity}")
    print(f"🚀 Breakthrough Mode: {breakthrough_mode}")
    print()
    
    print("🚀 Executing existing NWTN pipeline...")
    print("⏱️  Expected time: 2-3 hours for DEEP mode")
    print()
    
    # Run through the existing pipeline infrastructure
    success = await run_nwtn_pipeline(query, depth, verbosity, breakthrough_mode)
    
    if success:
        print("\n🎉 Quantum gravity analysis completed!")
        print("📄 Natural language response with citations generated")
        print("💾 Results automatically saved by existing pipeline")
    else:
        print("\n❌ Pipeline execution failed")
    
    # Now run Conservative mode for comparison
    print("\n" + "=" * 60)
    print("🎯 RUNNING CONSERVATIVE MODE FOR COMPARISON")
    print("=" * 60)
    
    conservative_success = await run_nwtn_pipeline(query, depth, verbosity, "CONSERVATIVE")
    
    if conservative_success:
        print("\n🎉 Both Conservative and Revolutionary analyses completed!")
        print("📊 You now have both extreme mode responses as requested")
    else:
        print("\n⚠️  Conservative mode failed")

if __name__ == "__main__":
    asyncio.run(main())