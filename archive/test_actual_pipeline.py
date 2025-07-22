#!/usr/bin/env python3
"""
Test Actual NWTN Pipeline Execution
===================================

This script runs the actual NWTN pipeline with the suggested prompt
to test real end-to-end functionality.
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_test_pipeline():
    """Run the actual pipeline with test parameters"""
    
    # Test parameters as if user selected them
    query = "What are the most promising approaches for developing room-temperature superconductors, and what are the current theoretical and experimental barriers?"
    depth = "STANDARD"
    verbosity = "DETAILED"
    
    print("🧠 NWTN PIPELINE TEST - ACTUAL EXECUTION")
    print("=" * 60)
    print(f"📝 Query: {query}")
    print(f"🔬 Depth: {depth}")
    print(f"📄 Verbosity: {verbosity}")
    print()
    
    try:
        # Import the pipeline runner function
        from run_nwtn_pipeline import run_nwtn_pipeline
        
        print("🚀 Starting actual pipeline execution...")
        print("⚠️  This will run the real NWTN system with:")
        print("   - Real search through 149,726 arXiv papers")
        print("   - Real STANDARD reasoning (5-6 engines)")
        print("   - Real Claude API synthesis")
        print("   - Real content grounding system")
        print()
        
        # Execute the pipeline
        success = await run_nwtn_pipeline(query, depth, verbosity)
        
        if success:
            print("\n🎉 SUCCESS! Pipeline executed smoothly!")
            print("✅ True 'prompt → answer' functionality achieved!")
        else:
            print("\n❌ Pipeline had issues - needs debugging")
            
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(run_test_pipeline())