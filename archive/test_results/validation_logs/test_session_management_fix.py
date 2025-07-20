#!/usr/bin/env python3
"""
Test Session Management Fix
==========================

This test verifies that the session ID mismatch issue has been resolved
between the SystemIntegrator and AttributionUsageTracker.
"""

import asyncio
import sys
sys.path.append('/Users/ryneschultz/Documents/GitHub/PRSM')

from prsm.nwtn.system_integrator import SystemIntegrator
from prsm.nwtn.attribution_usage_tracker import AttributionUsageTracker
from prsm.tokenomics.ftns_service import FTNSService
from prsm.integrations.core.provenance_engine import ProvenanceEngine


class MockProvenanceEngine:
    """Mock provenance engine for testing"""
    
    async def initialize(self):
        return True
    
    async def get_creator_info(self, paper_id: str):
        return {
            'creator_id': f'creator_{paper_id}',
            'name': f'Test Creator for {paper_id}',
            'wallet_address': f'wallet_{paper_id}'
        }


class MockFTNSService:
    """Mock FTNS service for testing"""
    
    def __init__(self):
        self.user_balances = {'test_user_bob': 1000.0}
        self.transactions = []
    
    async def initialize(self):
        return True
    
    async def get_user_balance(self, user_id: str):
        class Balance:
            def __init__(self, balance):
                self.balance = balance
        return Balance(self.user_balances.get(user_id, 0.0))
    
    async def distribute_royalty(self, creator_id: str, amount: float, description: str):
        transaction_id = f'txn_{len(self.transactions)}'
        self.transactions.append({
            'id': transaction_id,
            'creator_id': creator_id,
            'amount': amount,
            'description': description
        })
        return transaction_id


async def test_session_management():
    """Test that session IDs are properly coordinated"""
    
    print("🔍 Testing Session Management Fix")
    print("=" * 50)
    
    try:
        # 1. Create mock services
        ftns_service = MockFTNSService()
        provenance_engine = MockProvenanceEngine()
        
        # 2. Create SystemIntegrator with mocked services
        system_integrator = SystemIntegrator(
            ftns_service=ftns_service,
            provenance_engine=provenance_engine,
            force_mock_retriever=True  # Force mock for testing
        )
        
        # 3. Initialize the system
        init_success = await system_integrator.initialize()
        if not init_success:
            print("❌ SystemIntegrator initialization failed")
            return False
        
        print("✅ SystemIntegrator initialized successfully")
        
        # 4. Process a test query
        print("\n📝 Processing test query...")
        result = await system_integrator.process_test_scenario()
        
        print(f"Session ID: {result.session_id}")
        print(f"Success: {result.success}")
        print(f"Query: {result.query[:50]}...")
        
        if not result.success:
            print(f"❌ Query processing failed: {result.error_message}")
            return False
        
        print("✅ Query processed successfully")
        
        # 5. Check that the session was properly tracked
        usage_tracker = system_integrator.usage_tracker
        session_details = usage_tracker.get_session_details(result.session_id)
        
        if session_details is None:
            print(f"❌ Session {result.session_id} not found in usage tracker")
            return False
        
        print(f"✅ Session found in usage tracker: {session_details.session_id}")
        print(f"   Sources tracked: {len(session_details.sources_used)}")
        print(f"   Payment calculations: {len(session_details.payment_calculations)}")
        
        # 6. Test payment distribution (this was the failing step)
        print("\n💰 Testing payment distribution...")
        try:
            payment_distributions = await usage_tracker.distribute_payments(result.session_id)
            print(f"✅ Payment distribution successful: {len(payment_distributions)} distributions")
            
            for dist in payment_distributions:
                print(f"   - Creator {dist.creator_id}: {dist.payment_amount} FTNS")
        
        except Exception as e:
            if "Session" in str(e) and "not found" in str(e):
                print(f"❌ Session not found error still occurring: {e}")
                return False
            else:
                print(f"❌ Payment distribution failed with different error: {e}")
                return False
        
        # 7. Verify session consistency
        print("\n🔍 Verifying session consistency...")
        
        # Check that all session IDs match
        session_id = result.session_id
        usage_session_id = session_details.session_id
        
        if session_id != usage_session_id:
            print(f"❌ Session ID mismatch!")
            print(f"   SystemIntegrator: {session_id}")
            print(f"   UsageTracker:     {usage_session_id}")
            return False
        
        print(f"✅ Session IDs are consistent: {session_id}")
        
        # 8. Generate final report
        print("\n📊 Session Statistics:")
        print(f"   Total cost: {result.total_cost} FTNS")
        print(f"   Processing time: {result.processing_time:.2f}s")
        print(f"   Quality score: {result.quality_score:.2f}")
        print(f"   Citations: {len(result.citations)}")
        print(f"   Payments distributed: {len(result.payment_distributions)}")
        
        usage_stats = usage_tracker.get_usage_statistics()
        print(f"\n📈 Usage Tracker Statistics:")
        print(f"   Total sessions: {usage_stats['total_sessions']}")
        print(f"   Active sessions: {usage_stats['active_sessions']}")
        print(f"   Total payments distributed: {usage_stats['total_payments_distributed']:.2f} FTNS")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run the session management test"""
    
    print("🧪 NWTN Session Management Fix Test")
    print("=" * 50)
    
    success = await test_session_management()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ SESSION MANAGEMENT FIX TEST PASSED")
        print("   No 'Session not found' errors occurred")
        print("   Session IDs are properly coordinated")
        print("   Payment distribution works correctly")
    else:
        print("❌ SESSION MANAGEMENT FIX TEST FAILED")
        print("   Issues still exist in session coordination")
    
    return success


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)