#!/usr/bin/env python3
"""
Sprint 3 Phase 3: Marketplace Concurrency Integration Tests

Tests for concurrent purchase attempts, balance consistency under load,
idempotency key handling, race condition prevention, and atomic deduction.
"""

import asyncio
import pytest
from datetime import datetime
from typing import Dict, List, Optional
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

# Import marketplace components
from prsm.economy.tokenomics.advanced_ftns import AtomicFTNSService
from prsm.economy.tokenomics.ftns_service import FTNSService
from prsm.economy.marketplace.core import MarketplaceCore


class TestMarketplaceConcurrency:
    """Test suite for marketplace concurrency scenarios"""
    
    @pytest.fixture
    async def ftns_service(self):
        """Create a fresh FTNS service for each test"""
        service = AtomicFTNSService()
        await service.initialize()
        yield service
        await service.shutdown()
    
    @pytest.fixture
    async def funded_user(self, ftns_service):
        """Create a user with initial funds"""
        user_id = "test_user_concurrent"
        await ftns_service.credit(user_id, 1000.0, "initial_funds")
        yield user_id
        # Cleanup handled by ftns_service fixture
    
    # =========================================================================
    # Concurrent Purchase Tests
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_concurrent_purchases_same_user(self, ftns_service, funded_user):
        """Test that concurrent purchases from the same user are handled correctly"""
        initial_balance = await ftns_service.get_balance(funded_user)
        assert initial_balance == 1000.0
        
        # Try to make 5 concurrent purchases of 300 each
        # User only has 1000, so at most 3 should succeed
        async def purchase(amount: float, purchase_id: str):
            try:
                success = await ftns_service.atomic_deduct(
                    funded_user,
                    amount,
                    f"purchase_{purchase_id}",
                    idempotency_key=f"purchase_{purchase_id}"
                )
                return {"success": success, "amount": amount, "id": purchase_id}
            except Exception as e:
                return {"success": False, "error": str(e), "id": purchase_id}
        
        # Execute concurrent purchases
        results = await asyncio.gather(
            purchase(300.0, "p1"),
            purchase(300.0, "p2"),
            purchase(300.0, "p3"),
            purchase(300.0, "p4"),
            purchase(300.0, "p5"),
            return_exceptions=True
        )
        
        # Count successes and failures
        successes = [r for r in results if isinstance(r, dict) and r.get("success")]
        failures = [r for r in results if isinstance(r, dict) and not r.get("success")]
        
        # At least some should fail due to insufficient balance
        assert len(failures) >= 2, "At least 2 purchases should fail due to insufficient balance"
        
        # Verify final balance
        final_balance = await ftns_service.get_balance(funded_user)
        assert final_balance >= 0, "Balance should never go negative"
        assert final_balance <= initial_balance, "Balance should not increase"
        
        # Verify total deductions match
        total_deducted = sum(s.get("amount", 0) for s in successes)
        assert initial_balance - final_balance == total_deducted, "Total deductions should match"
    
    @pytest.mark.asyncio
    async def test_concurrent_purchases_different_users(self, ftns_service):
        """Test that concurrent purchases from different users don't interfere"""
        # Create multiple users with funds
        users = []
        for i in range(5):
            user_id = f"concurrent_user_{i}"
            await ftns_service.credit(user_id, 100.0, "initial_funds")
            users.append(user_id)
        
        # Each user tries to purchase 50
        async def purchase(user_id: str, amount: float):
            success = await ftns_service.atomic_deduct(
                user_id,
                amount,
                f"purchase_by_{user_id}",
                idempotency_key=f"purchase_{user_id}"
            )
            return {"user": user_id, "success": success}
        
        # Execute concurrent purchases
        results = await asyncio.gather(
            *[purchase(user, 50.0) for user in users],
            return_exceptions=True
        )
        
        # All should succeed since each user has enough balance
        successes = [r for r in results if isinstance(r, dict) and r.get("success")]
        assert len(successes) == 5, "All purchases should succeed"
        
        # Verify each user's balance
        for user in users:
            balance = await ftns_service.get_balance(user)
            assert balance == 50.0, f"User {user} should have 50.0 remaining"
    
    # =========================================================================
    # Balance Consistency Tests
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_balance_consistency_under_load(self, ftns_service, funded_user):
        """Test that balance remains consistent under high load"""
        initial_balance = await ftns_service.get_balance(funded_user)
        
        # Perform many small transactions
        async def small_deduct(amount: float, tx_id: str):
            try:
                return await ftns_service.atomic_deduct(
                    funded_user,
                    amount,
                    tx_id,
                    idempotency_key=tx_id
                )
            except Exception:
                return False
        
        # Create 100 small deduction attempts
        tasks = [
            small_deduct(10.0, f"tx_{i}")
            for i in range(100)
        ]
        
        # Execute in batches to avoid overwhelming
        results = []
        for i in range(0, len(tasks), 10):
            batch = tasks[i:i+10]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            results.extend(batch_results)
        
        # Count successful deductions
        successful = sum(1 for r in results if r is True)
        
        # Verify balance
        final_balance = await ftns_service.get_balance(funded_user)
        expected_deduction = successful * 10.0
        
        assert initial_balance - final_balance == expected_deduction, \
            f"Balance mismatch: expected {expected_deduction} deducted, got {initial_balance - final_balance}"
        assert final_balance >= 0, "Balance should never be negative"
    
    @pytest.mark.asyncio
    async def test_credit_debit_consistency(self, ftns_service):
        """Test that credits and debits maintain consistency"""
        user_id = "credit_debit_user"
        
        # Perform alternating credits and debits
        for i in range(10):
            await ftns_service.credit(user_id, 100.0, f"credit_{i}")
            await ftns_service.atomic_deduct(
                user_id,
                50.0,
                f"debit_{i}",
                idempotency_key=f"debit_{i}"
            )
        
        # Final balance should be 10 * (100 - 50) = 500
        final_balance = await ftns_service.get_balance(user_id)
        assert final_balance == 500.0, f"Expected 500.0, got {final_balance}"
    
    # =========================================================================
    # Idempotency Key Tests
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_idempotency_prevents_double_deduction(self, ftns_service, funded_user):
        """Test that same idempotency key prevents double deduction"""
        initial_balance = await ftns_service.get_balance(funded_user)
        
        # First deduction
        result1 = await ftns_service.atomic_deduct(
            funded_user,
            100.0,
            "test_purchase",
            idempotency_key="idempotent_key_1"
        )
        assert result1 is True, "First deduction should succeed"
        
        # Second deduction with same idempotency key
        result2 = await ftns_service.atomic_deduct(
            funded_user,
            100.0,
            "test_purchase",
            idempotency_key="idempotent_key_1"  # Same key
        )
        # Should return True but not deduct again (idempotent)
        assert result2 is True, "Idempotent request should return success"
        
        # Verify only one deduction happened
        final_balance = await ftns_service.get_balance(funded_user)
        assert final_balance == initial_balance - 100.0, \
            "Only one deduction should have occurred"
    
    @pytest.mark.asyncio
    async def test_different_idempotency_keys_allow_multiple_deductions(self, ftns_service, funded_user):
        """Test that different idempotency keys allow multiple deductions"""
        initial_balance = await ftns_service.get_balance(funded_user)
        
        # First deduction
        await ftns_service.atomic_deduct(
            funded_user,
            100.0,
            "purchase_1",
            idempotency_key="key_1"
        )
        
        # Second deduction with different key
        await ftns_service.atomic_deduct(
            funded_user,
            100.0,
            "purchase_2",
            idempotency_key="key_2"
        )
        
        # Verify both deductions happened
        final_balance = await ftns_service.get_balance(funded_user)
        assert final_balance == initial_balance - 200.0, \
            "Both deductions should have occurred"
    
    @pytest.mark.asyncio
    async def test_idempotency_key_concurrent_requests(self, ftns_service, funded_user):
        """Test that concurrent requests with same idempotency key are handled correctly"""
        initial_balance = await ftns_service.get_balance(funded_user)
        
        # Make 5 concurrent requests with the same idempotency key
        async def deduct_with_key(key: str):
            return await ftns_service.atomic_deduct(
                funded_user,
                100.0,
                "concurrent_purchase",
                idempotency_key=key
            )
        
        results = await asyncio.gather(
            deduct_with_key("same_key"),
            deduct_with_key("same_key"),
            deduct_with_key("same_key"),
            deduct_with_key("same_key"),
            deduct_with_key("same_key"),
            return_exceptions=True
        )
        
        # All should return success (idempotent)
        successes = sum(1 for r in results if r is True)
        assert successes == 5, "All idempotent requests should return success"
        
        # But only one deduction should have occurred
        final_balance = await ftns_service.get_balance(funded_user)
        assert final_balance == initial_balance - 100.0, \
            "Only one deduction should have occurred despite concurrent requests"
    
    # =========================================================================
    # Race Condition Prevention Tests
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_race_condition_prevention_check_balance(self, ftns_service):
        """Test that race conditions in balance checking are prevented"""
        user_id = "race_condition_user"
        await ftns_service.credit(user_id, 100.0, "initial")
        
        # Simulate race condition: multiple operations checking balance simultaneously
        async def check_and_deduct():
            balance = await ftns_service.get_balance(user_id)
            if balance >= 60:
                # Try to deduct 60
                return await ftns_service.atomic_deduct(
                    user_id,
                    60.0,
                    "race_deduct",
                    idempotency_key=f"race_{id(asyncio.current_task())}"
                )
            return False
        
        # Run multiple check-and-deduct operations concurrently
        results = await asyncio.gather(
            check_and_deduct(),
            check_and_deduct(),
            check_and_deduct(),
            return_exceptions=True
        )
        
        # Count successes
        successes = sum(1 for r in results if r is True)
        
        # Only one should succeed (100 - 60 = 40, not enough for second 60)
        assert successes == 1, f"Only 1 deduction should succeed, got {successes}"
        
        # Verify final balance
        final_balance = await ftns_service.get_balance(user_id)
        assert final_balance == 40.0, f"Final balance should be 40.0, got {final_balance}"
    
    @pytest.mark.asyncio
    async def test_atomic_deduct_all_or_nothing(self, ftns_service):
        """Test that atomic deduct is all-or-nothing"""
        user_id = "atomic_user"
        await ftns_service.credit(user_id, 100.0, "initial")
        
        # Attempt to deduct more than balance
        result = await ftns_service.atomic_deduct(
            user_id,
            200.0,  # More than available
            "overdraft_attempt",
            idempotency_key="overdraft_1"
        )
        
        # Should fail
        assert result is False, "Overdraft should fail"
        
        # Balance should be unchanged
        balance = await ftns_service.get_balance(user_id)
        assert balance == 100.0, "Balance should be unchanged after failed deduct"
    
    # =========================================================================
    # Stress Tests
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_high_concurrency_stress(self, ftns_service):
        """Stress test with many concurrent operations"""
        num_users = 20
        operations_per_user = 10
        
        # Create users
        users = [f"stress_user_{i}" for i in range(num_users)]
        for user in users:
            await ftns_service.credit(user, 1000.0, "initial")
        
        # Perform many operations
        async def user_operations(user_id: str):
            for i in range(operations_per_user):
                await ftns_service.atomic_deduct(
                    user_id,
                    10.0,
                    f"stress_op_{i}",
                    idempotency_key=f"{user_id}_op_{i}"
                )
        
        # Run all user operations concurrently
        await asyncio.gather(
            *[user_operations(user) for user in users],
            return_exceptions=True
        )
        
        # Verify all balances
        for user in users:
            balance = await ftns_service.get_balance(user)
            expected = 1000.0 - (operations_per_user * 10.0)
            assert balance == expected, f"User {user} balance mismatch"


class TestMarketplaceIntegration:
    """Integration tests for marketplace with FTNS service"""
    
    @pytest.fixture
    async def marketplace_setup(self):
        """Set up marketplace with FTNS service"""
        ftns_service = AtomicFTNSService()
        await ftns_service.initialize()
        
        # Create test users
        buyer_id = "test_buyer"
        seller_id = "test_seller"
        
        await ftns_service.credit(buyer_id, 500.0, "initial_funds")
        await ftns_service.credit(seller_id, 100.0, "initial_funds")
        
        yield {
            "ftns": ftns_service,
            "buyer": buyer_id,
            "seller": seller_id
        }
        
        await ftns_service.shutdown()
    
    @pytest.mark.asyncio
    async def test_marketplace_purchase_flow(self, marketplace_setup):
        """Test complete marketplace purchase flow"""
        ftns = marketplace_setup["ftns"]
        buyer = marketplace_setup["buyer"]
        seller = marketplace_setup["seller"]
        
        buyer_initial = await ftns.get_balance(buyer)
        seller_initial = await ftns.get_balance(seller)
        
        purchase_amount = 100.0
        
        # Buyer makes purchase
        success = await ftns.atomic_deduct(
            buyer,
            purchase_amount,
            "marketplace_purchase",
            idempotency_key="purchase_1"
        )
        assert success is True
        
        # Seller receives payment (credit)
        await ftns.credit(seller, purchase_amount, "sale_proceeds")
        
        # Verify balances
        buyer_final = await ftns.get_balance(buyer)
        seller_final = await ftns.get_balance(seller)
        
        assert buyer_final == buyer_initial - purchase_amount
        assert seller_final == seller_initial + purchase_amount
    
    @pytest.mark.asyncio
    async def test_marketplace_refund_flow(self, marketplace_setup):
        """Test marketplace refund flow"""
        ftns = marketplace_setup["ftns"]
        buyer = marketplace_setup["buyer"]
        
        buyer_initial = await ftns.get_balance(buyer)
        
        # Make purchase
        await ftns.atomic_deduct(
            buyer,
            100.0,
            "purchase_for_refund",
            idempotency_key="refund_test_purchase"
        )
        
        # Verify deduction
        assert await ftns.get_balance(buyer) == buyer_initial - 100.0
        
        # Process refund
        await ftns.credit(buyer, 100.0, "refund")
        
        # Verify refund
        assert await ftns.get_balance(buyer) == buyer_initial


# =========================================================================
# Test Runner
# =========================================================================

async def run_marketplace_concurrency_tests():
    """Run all marketplace concurrency tests manually"""
    print("=" * 60)
    print("MARKETPLACE CONCURRENCY INTEGRATION TESTS")
    print("=" * 60)
    
    # Create service
    print("\n[SETUP] Creating FTNS service...")
    service = AtomicFTNSService()
    await service.initialize()
    
    test_instance = TestMarketplaceConcurrency()
    
    # Test 1: Idempotency
    print("\n[TEST 1] Idempotency key prevents double deduction...")
    try:
        user_id = "idempotency_test_user"
        await service.credit(user_id, 500.0, "initial")
        
        result1 = await service.atomic_deduct(user_id, 100.0, "purchase", idempotency_key="idem_1")
        result2 = await service.atomic_deduct(user_id, 100.0, "purchase", idempotency_key="idem_1")
        
        balance = await service.get_balance(user_id)
        assert balance == 400.0, f"Expected 400.0, got {balance}"
        print("  ✓ PASSED: Idempotency works correctly")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    
    # Test 2: Concurrent purchases
    print("\n[TEST 2] Concurrent purchases from same user...")
    try:
        user_id = "concurrent_test_user"
        await service.credit(user_id, 200.0, "initial")
        
        async def purchase(amount, key):
            return await service.atomic_deduct(user_id, amount, "purchase", idempotency_key=key)
        
        results = await asyncio.gather(
            purchase(150.0, "c1"),
            purchase(150.0, "c2"),
            return_exceptions=True
        )
        
        successes = sum(1 for r in results if r is True)
        assert successes == 1, f"Expected 1 success, got {successes}"
        
        balance = await service.get_balance(user_id)
        assert balance == 50.0, f"Expected 50.0, got {balance}"
        print("  ✓ PASSED: Concurrent purchases handled correctly")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    
    # Test 3: Balance consistency
    print("\n[TEST 3] Balance consistency under load...")
    try:
        user_id = "consistency_test_user"
        await service.credit(user_id, 100.0, "initial")
        
        # Many small deductions
        for i in range(10):
            await service.atomic_deduct(user_id, 5.0, f"deduct_{i}", idempotency_key=f"deduct_{i}")
        
        balance = await service.get_balance(user_id)
        assert balance == 50.0, f"Expected 50.0, got {balance}"
        print("  ✓ PASSED: Balance consistency maintained")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    
    await service.shutdown()
    
    print("\n" + "=" * 60)
    print("MARKETPLACE CONCURRENCY TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_marketplace_concurrency_tests())
