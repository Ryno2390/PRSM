#!/usr/bin/env python3
"""
Test script for PRSM FTNS Token Service
Tests Phase 1, Week 1, Task 3 - FTNS Token Integration
"""

import asyncio
import sys
from datetime import datetime, timezone
from typing import Dict, Any

async def test_ftns_service():
    """Test FTNS service functionality"""
    
    print("🪙 Testing PRSM FTNS Token Service...")
    print("=" * 50)
    
    try:
        # Import the FTNS service
        from prsm.economy.tokenomics.ftns_service import FTNSService, ftns_service
        from prsm.core.models import PRSMSession, FTNSBalance
        
        print("✅ FTNS service imports successful")
        
        # Create service instance
        service = FTNSService()
        
        # Test user setup
        test_user = "test_user_123"
        
        # === Test Balance Management ===
        
        # Check initial balance (should be 0)
        initial_balance = await service.get_user_balance(test_user)
        assert initial_balance.balance == 0.0
        print(f"✅ Initial balance correct: {initial_balance.balance} FTNS")
        
        # Give user some initial FTNS tokens
        await service.reward_contribution(test_user, "data", 2000.0)  # 2000MB data = 100 FTNS
        
        balance_after_reward = await service.get_user_balance(test_user)
        assert balance_after_reward.balance > 0
        print(f"✅ Reward credited: {balance_after_reward.balance} FTNS")
        
        # === Test Context Charging ===
        
        # Create test session
        session = PRSMSession(
            user_id=test_user,
            nwtn_context_allocation=0
        )
        
        # Calculate context cost
        context_units = 500
        cost = await service.calculate_context_cost(session, context_units)
        assert cost > 0
        print(f"✅ Context cost calculated: {cost} FTNS for {context_units} units")
        
        # Store balance before charge
        balance_before_charge = balance_after_reward.balance
        
        # Charge for context access
        success = await service.charge_context_access(test_user, context_units)
        assert success == True
        print(f"✅ Context charge successful")
        
        # Check balance after charge
        balance_after_charge = await service.get_user_balance(test_user)
        assert balance_after_charge.balance < balance_before_charge
        print(f"✅ Balance updated after charge: {balance_after_charge.balance} FTNS")
        
        # Test insufficient funds
        large_context = 10000  # Should cost more than available balance
        insufficient_success = await service.charge_context_access(test_user, large_context)
        assert insufficient_success == False
        print(f"✅ Insufficient funds correctly rejected")
        
        # === Test Reward Types ===
        
        # Test data contribution reward
        data_reward = await service.reward_contribution(test_user, "data", 50.0)  # 50MB
        assert data_reward == True
        print(f"✅ Data contribution reward successful")
        
        # Test model contribution reward
        model_reward = await service.reward_contribution(test_user, "model", 1.0)
        assert model_reward == True
        print(f"✅ Model contribution reward successful")
        
        # Test research publication reward
        research_metadata = {"citations": 10}
        research_reward = await service.reward_contribution(
            test_user, "research", 1.0, research_metadata
        )
        assert research_reward == True
        print(f"✅ Research publication reward successful")
        
        # Test governance participation reward
        governance_reward = await service.reward_contribution(test_user, "governance", 1.0)
        assert governance_reward == True
        print(f"✅ Governance participation reward successful")
        
        # === Test Teaching Rewards ===
        
        teacher_id = "teacher_001"
        improvement_score = 0.75  # 75% improvement
        teaching_reward = await service.reward_teaching_success(teacher_id, improvement_score)
        assert teaching_reward > 0
        print(f"✅ Teaching success reward: {teaching_reward} FTNS")
        
        # === Test Royalty Calculation ===
        
        content_hash = "QmTestHash123"
        access_count = 50
        royalties = await service.calculate_royalties(content_hash, access_count)
        # Should be 0 since no provenance record exists
        assert royalties == 0.0
        print(f"✅ Royalty calculation (no record): {royalties} FTNS")
        
        # === Test Dividend Distribution ===
        
        # Create multiple test users
        holders = [test_user, "user_2", "user_3"]
        
        # Give some balance to other users
        await service.reward_contribution("user_2", "data", 200.0)  # 10 FTNS
        await service.reward_contribution("user_3", "data", 300.0)  # 15 FTNS
        
        # Distribute dividends
        dividend_pool = 100.0  # 100 FTNS to distribute
        distributions = await service.distribute_dividends(holders, dividend_pool)
        
        assert len(distributions) > 0
        assert sum(distributions.values()) <= dividend_pool + 0.001  # Allow for rounding
        print(f"✅ Dividend distribution successful: {len(distributions)} recipients")
        
        # === Test Transaction History ===
        
        transactions = await service.get_transaction_history(test_user, limit=10)
        assert len(transactions) > 0
        print(f"✅ Transaction history retrieved: {len(transactions)} transactions")
        
        # Verify transaction types
        tx_types = {tx.transaction_type for tx in transactions}
        expected_types = {"reward", "charge", "dividend"}
        assert expected_types.issubset(tx_types)
        print(f"✅ Transaction types correct: {tx_types}")
        
        # === Test System Statistics ===
        
        stats = await service.get_system_stats()
        assert "total_supply" in stats
        assert "total_holders" in stats
        assert "total_transactions" in stats
        assert stats["total_supply"] > 0
        assert stats["total_holders"] >= 3  # At least our test users
        print(f"✅ System stats: {stats['total_supply']:.2f} FTNS total supply")
        
        # === Test Context Allocation ===
        
        # Test successful allocation
        required_context = 100
        allocation_success = await service.allocate_context(session, required_context)
        assert allocation_success == True
        print(f"✅ Context allocation successful")
        
        # Test allocation with insufficient funds
        large_session = PRSMSession(user_id="poor_user", nwtn_context_allocation=0)
        large_allocation = await service.allocate_context(large_session, 50000)
        assert large_allocation == False
        print(f"✅ Large allocation correctly rejected")
        
        # === Test Edge Cases ===
        
        # Test zero context charge
        zero_charge = await service.charge_context_access(test_user, 0)
        assert zero_charge == True
        print(f"✅ Zero context charge handled correctly")
        
        # Test zero reward
        zero_reward = await service.reward_contribution(test_user, "unknown", 0.0)
        assert zero_reward == True
        print(f"✅ Zero reward handled correctly")
        
        # Test empty dividend distribution
        empty_dividends = await service.distribute_dividends([], 100.0)
        assert len(empty_dividends) == 0
        print(f"✅ Empty dividend distribution handled correctly")
        
        print("\n" + "=" * 50)
        print("🎉 ALL FTNS SERVICE TESTS PASSED!")
        
        # Show final balances
        final_balance = await service.get_user_balance(test_user)
        print(f"📊 Final test user balance: {final_balance.balance:.8f} FTNS")
        
        final_stats = await service.get_system_stats()
        print(f"📊 Final system stats:")
        print(f"   - Total supply: {final_stats['total_supply']:.2f} FTNS")
        print(f"   - Total holders: {final_stats['total_holders']}")
        print(f"   - Total transactions: {final_stats['total_transactions']}")
        
        return True
        
    except Exception as e:
        print(f"❌ FTNS service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_ftns_service())
    sys.exit(0 if success else 1)