#!/usr/bin/env python3
"""
Debug version of NWTN test to trace .lower() errors
"""

# Import the debug tracer first
import debug_lower_errors

# Now run the actual test
import asyncio
import time
from nwtn_focused_reasoning_test import NWTNFocusedReasoningTest

async def debug_test():
    print("🔍 Starting debug test with .lower() error tracing...")
    
    tester = NWTNFocusedReasoningTest()
    
    try:
        await tester.run_test()
        print("✅ Test completed successfully")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_test())