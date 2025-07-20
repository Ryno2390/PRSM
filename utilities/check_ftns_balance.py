#!/usr/bin/env python3
"""
Quick FTNS balance check
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def check_ftns_balance():
    """Check FTNS balance for test user"""
    print("🔍 Checking FTNS balance...")
    
    try:
        # Import services
        from prsm.tokenomics.ftns_service import get_ftns_service
        
        # Initialize FTNS service
        ftns_service = await get_ftns_service()
        
        # Add balance
        await ftns_service.reward_contribution("test_user_001", "data", 2000.0)
        
        # Check balance
        balance = await ftns_service.get_balance("test_user_001")
        print(f"✅ User test_user_001 balance: {balance} FTNS")
        
        # Check if balance is sufficient for query (37.5 FTNS required)
        if balance >= 37.5:
            print("✅ Balance is sufficient for query execution")
        else:
            print(f"❌ Balance insufficient: {balance} < 37.5 FTNS")
            
        return balance >= 37.5
        
    except Exception as e:
        print(f"❌ Error checking balance: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(check_ftns_balance())
    if success:
        print("\n🎉 FTNS balance check passed!")
    else:
        print("\n🚨 FTNS balance check failed!")