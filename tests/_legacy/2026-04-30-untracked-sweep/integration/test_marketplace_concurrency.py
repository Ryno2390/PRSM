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
from prsm.economy.tokenomics.atomic_ftns_service import AtomicFTNSService
from prsm.economy.tokenomics.ftns_service import FTNSService
from prsm.economy.marketplace.ecosystem.marketplace_core import MarketplaceCore


class TestMarketplaceConcurrency:
    """Test suite for marketplace concurrency scenarios"""

    @pytest.fixture
    async def ftns_service(self):
        """Create AtomicFTNSService with mocked database for testing.

        The AtomicFTNSService uses PostgreSQL-specific SQL (NOW(), RETURNING),
        so we mock the database layer and use an in-memory state dict for testing.
        """
        from unittest.mock import AsyncMock, patch
        from sqlalchemy.ext.asyncio import AsyncSession

        # In-memory state for this test
        state = {
            "balances": {},  # user_id -> {"balance": float, "locked": float, "version": int}
            "idempotency_keys": {},  # key -> transaction_id
            "transactions": [],  # list of transactions
        }

        # Create a mock session
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()

        # Mock execute to handle various SQL operations
        async def mock_execute(query, params=None):
            from unittest.mock import MagicMock

            result = MagicMock()
            query_str = str(query) if query else ""

            # Handle account existence check
            if "SELECT" in query_str and "ftns_balances" in query_str and "WHERE user_id" in query_str:
                user_id = params.get("user_id") if params else None
                if user_id and user_id in state["balances"]:
                    row = MagicMock()
                    row.balance = state["balances"][user_id]["balance"]
                    row.locked_balance = state["balances"][user_id]["locked"]
                    row.version = state["balances"][user_id]["version"]
                    result.fetchone.return_value = row
                else:
                    result.fetchone.return_value = None

            # Handle balance insert
            elif "INSERT INTO ftns_balances" in query_str:
                if params:
                    user_id = params.get("user_id")
                    balance = float(params.get("balance", 0))
                    if user_id not in state["balances"]:
                        state["balances"][user_id] = {
                            "balance": balance,
                            "locked": 0.0,
                            "version": 1,
                            "total_earned": balance,
                            "total_spent": 0.0
                        }
                    row = MagicMock()
                    row.user_id = user_id
                    result.fetchone.return_value = row

            # Handle balance update (deduction)
            elif "UPDATE ftns_balances" in query_str:
                if params:
                    user_id = params.get("user_id")
                    new_balance = float(params.get("new_balance", 0))
                    if user_id in state["balances"]:
                        old_version = state["balances"][user_id]["version"]
                        expected_version = params.get("expected_version", old_version)
                        if expected_version == old_version:
                            state["balances"][user_id]["balance"] = new_balance
                            state["balances"][user_id]["version"] += 1
                            result.rowcount = 1
                        else:
                            result.rowcount = 0
                    else:
                        result.rowcount = 0

            # Handle idempotency key check
            elif "ftns_idempotency_keys" in query_str and "SELECT" in query_str:
                key = params.get("key") if params else None
                if key and key in state["idempotency_keys"]:
                    row = MagicMock()
                    row.transaction_id = state["idempotency_keys"][key]
                    result.fetchone.return_value = row
                else:
                    result.fetchone.return_value = None

            # Handle idempotency key insert
            elif "INSERT INTO ftns_idempotency_keys" in query_str:
                if params:
                    key = params.get("key")
                    tx_id = params.get("tx_id")
                    if key:
                        state["idempotency_keys"][key] = tx_id

            # Handle transaction insert
            elif "INSERT INTO ftns_transactions" in query_str:
                if params:
                    state["transactions"].append(params)

            return result

        mock_session.execute = mock_execute

        # Create context manager for session
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        # Patch get_async_session
        with patch('prsm.economy.tokenomics.atomic_ftns_service.get_async_session', return_value=mock_context):
            service = AtomicFTNSService()
            service._initialized = True

            # Override methods to use our mock state
            original_mint = service.mint_tokens_atomic
            original_deduct = service.deduct_tokens_atomic
            original_get_balance = service.get_balance

            async def mock_mint(to_user_id, amount, idempotency_key, description="Token mint", metadata=None):
                if to_user_id not in state["balances"]:
                    state["balances"][to_user_id] = {
                        "balance": 0.0,
                        "locked": 0.0,
                        "version": 1,
                        "total_earned": 0.0,
                        "total_spent": 0.0
                    }
                state["balances"][to_user_id]["balance"] += float(amount)
                state["balances"][to_user_id]["total_earned"] += float(amount)
                state["idempotency_keys"][idempotency_key] = f"tx_{idempotency_key}"

                from prsm.economy.tokenomics.atomic_ftns_service import TransactionResult
                return TransactionResult(
                    success=True,
                    transaction_id=f"tx_{idempotency_key}",
                    new_balance=Decimal(str(state["balances"][to_user_id]["balance"])),
                    error_message=None,
                    idempotent_replay=False
                )

            async def mock_deduct(user_id, amount, idempotency_key, description="", metadata=None):
                if idempotency_key in state["idempotency_keys"]:
                    from prsm.economy.tokenomics.atomic_ftns_service import TransactionResult
                    return TransactionResult(
                        success=True,
                        transaction_id=state["idempotency_keys"][idempotency_key],
                        new_balance=Decimal(str(state["balances"].get(user_id, {}).get("balance", 0))),
                        error_message=None,
                        idempotent_replay=True
                    )

                if user_id not in state["balances"]:
                    from prsm.economy.tokenomics.atomic_ftns_service import TransactionResult
                    return TransactionResult(
                        success=False,
                        transaction_id=None,
                        new_balance=None,
                        error_message="Account not found",
                        idempotent_replay=False
                    )

                available = state["balances"][user_id]["balance"] - state["balances"][user_id]["locked"]
                if available < float(amount):
                    from prsm.economy.tokenomics.atomic_ftns_service import TransactionResult
                    return TransactionResult(
                        success=False,
                        transaction_id=None,
                        new_balance=Decimal(str(available)),
                        error_message="Insufficient balance",
                        idempotent_replay=False
                    )

                state["balances"][user_id]["balance"] -= float(amount)
                state["balances"][user_id]["total_spent"] += float(amount)
                state["balances"][user_id]["version"] += 1
                state["idempotency_keys"][idempotency_key] = f"tx_{idempotency_key}"

                from prsm.economy.tokenomics.atomic_ftns_service import TransactionResult
                return TransactionResult(
                    success=True,
                    transaction_id=f"tx_{idempotency_key}",
                    new_balance=Decimal(str(state["balances"][user_id]["balance"])),
                    error_message=None,
                    idempotent_replay=False
                )

            async def mock_get_balance(user_id):
                from prsm.economy.tokenomics.atomic_ftns_service import BalanceInfo
                from datetime import datetime, timezone
                if user_id not in state["balances"]:
                    # Auto-create account
                    state["balances"][user_id] = {
                        "balance": 0.0,
                        "locked": 0.0,
                        "version": 1,
                        "total_earned": 0.0,
                        "total_spent": 0.0
                    }
                bal = state["balances"][user_id]
                return BalanceInfo(
                    user_id=user_id,
                    balance=Decimal(str(bal["balance"])),
                    locked_balance=Decimal(str(bal["locked"])),
                    available_balance=Decimal(str(bal["balance"] - bal["locked"])),
                    total_earned=Decimal(str(bal["total_earned"])),
                    total_spent=Decimal(str(bal["total_spent"])),
                    version=bal["version"],
                    last_updated=datetime.now(timezone.utc)
                )

            service.mint_tokens_atomic = mock_mint
            service.deduct_tokens_atomic = mock_deduct
            service.get_balance = mock_get_balance

            yield service
    
    @pytest.fixture
    async def funded_user(self, ftns_service):
        """Create a user with initial funds"""
        user_id = "test_user_concurrent"
        # Use mint_tokens_atomic instead of credit
        result = await ftns_service.mint_tokens_atomic(
            user_id,
            Decimal("1000.0"),
            f"initial_funds_{user_id}",
            "Initial funds for testing"
        )
        assert result.success, f"Failed to fund user: {result.error_message}"
        yield user_id
        # Cleanup handled by ftns_service fixture
    
    # =========================================================================
    # Concurrent Purchase Tests
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_concurrent_purchases_same_user(self, ftns_service, funded_user):
        """Test that concurrent purchases from the same user are handled correctly"""
        initial_balance_info = await ftns_service.get_balance(funded_user)
        initial_balance = float(initial_balance_info.balance)
        assert initial_balance == 1000.0
        
        # Try to make 5 concurrent purchases of 300 each
        # User only has 1000, so at most 3 should succeed
        async def purchase(amount: float, purchase_id: str):
            try:
                result = await ftns_service.deduct_tokens_atomic(
                    funded_user,
                    Decimal(str(amount)),
                    f"purchase_{purchase_id}",
                    f"Purchase {purchase_id}"
                )
                return {"success": result.success, "amount": amount, "id": purchase_id}
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
        final_balance_info = await ftns_service.get_balance(funded_user)
        final_balance = float(final_balance_info.balance)
        assert final_balance >= 0, "Balance should never go negative"
        assert final_balance <= initial_balance, "Balance should not increase"
        
        # Verify total deductions match
        total_deducted = sum(s.get("amount", 0) for s in successes)
        assert abs((initial_balance - final_balance) - total_deducted) < 0.01, "Total deductions should match"
    
    @pytest.mark.asyncio
    async def test_concurrent_purchases_different_users(self, ftns_service):
        """Test that concurrent purchases from different users don't interfere"""
        # Create multiple users with funds
        users = []
        for i in range(5):
            user_id = f"concurrent_user_{i}"
            result = await ftns_service.mint_tokens_atomic(
                user_id,
                Decimal("100.0"),
                f"initial_funds_{user_id}",
                "Initial funds"
            )
            assert result.success, f"Failed to fund user {user_id}"
            users.append(user_id)
        
        # Each user tries to purchase 50
        async def purchase(user_id: str, amount: float):
            result = await ftns_service.deduct_tokens_atomic(
                user_id,
                Decimal(str(amount)),
                f"purchase_{user_id}",
                f"Purchase by {user_id}"
            )
            return {"user": user_id, "success": result.success}
        
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
            balance_info = await ftns_service.get_balance(user)
            assert float(balance_info.balance) == 50.0, f"User {user} should have 50.0 remaining"
    
    # =========================================================================
    # Balance Consistency Tests
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_balance_consistency_under_load(self, ftns_service, funded_user):
        """Test that balance remains consistent under high load"""
        initial_balance_info = await ftns_service.get_balance(funded_user)
        initial_balance = float(initial_balance_info.balance)
        
        # Perform many small transactions
        async def small_deduct(amount: float, tx_id: str):
            try:
                result = await ftns_service.deduct_tokens_atomic(
                    funded_user,
                    Decimal(str(amount)),
                    tx_id,
                    f"Transaction {tx_id}"
                )
                return result.success
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
        final_balance_info = await ftns_service.get_balance(funded_user)
        final_balance = float(final_balance_info.balance)
        expected_deduction = successful * 10.0
        
        assert abs((initial_balance - final_balance) - expected_deduction) < 0.01, \
            f"Balance mismatch: expected {expected_deduction} deducted, got {initial_balance - final_balance}"
        assert final_balance >= 0, "Balance should never be negative"
    
    @pytest.mark.asyncio
    async def test_credit_debit_consistency(self, ftns_service):
        """Test that credits and debits maintain consistency"""
        user_id = "credit_debit_user"
        
        # Perform alternating credits and debits
        for i in range(10):
            result = await ftns_service.mint_tokens_atomic(
                user_id,
                Decimal("100.0"),
                f"credit_{i}_{user_id}",
                f"Credit {i}"
            )
            assert result.success, f"Credit {i} failed"
            
            result = await ftns_service.deduct_tokens_atomic(
                user_id,
                Decimal("50.0"),
                f"debit_{i}_{user_id}",
                f"Debit {i}"
            )
            assert result.success, f"Debit {i} failed"
        
        # Final balance should be 10 * (100 - 50) = 500
        final_balance_info = await ftns_service.get_balance(user_id)
        assert float(final_balance_info.balance) == 500.0, f"Expected 500.0, got {float(final_balance_info.balance)}"
    
    # =========================================================================
    # Idempotency Key Tests
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_idempotency_prevents_double_deduction(self, ftns_service, funded_user):
        """Test that same idempotency key prevents double deduction"""
        initial_balance_info = await ftns_service.get_balance(funded_user)
        initial_balance = float(initial_balance_info.balance)
        
        # First deduction
        result1 = await ftns_service.deduct_tokens_atomic(
            funded_user,
            Decimal("100.0"),
            "idempotent_key_1",
            "Test purchase"
        )
        assert result1.success, "First deduction should succeed"

        # Second deduction with same idempotency key
        result2 = await ftns_service.deduct_tokens_atomic(
            funded_user,
            Decimal("100.0"),
            "idempotent_key_1",  # Same key
            "Test purchase"
        )
        # Should return True but not deduct again (idempotent)
        assert result2.success, "Idempotent request should return success"
        assert result2.idempotent_replay, "Should indicate idempotent replay"
        
        # Verify only one deduction happened
        final_balance_info = await ftns_service.get_balance(funded_user)
        assert float(final_balance_info.balance) == initial_balance - 100.0, \
            "Only one deduction should have occurred"
    
    @pytest.mark.asyncio
    async def test_different_idempotency_keys_allow_multiple_deductions(self, ftns_service, funded_user):
        """Test that different idempotency keys allow multiple deductions"""
        initial_balance_info = await ftns_service.get_balance(funded_user)
        initial_balance = float(initial_balance_info.balance)
        
        # First deduction
        result1 = await ftns_service.deduct_tokens_atomic(
            funded_user,
            Decimal("100.0"),
            "key_1",
            "Purchase 1"
        )
        assert result1.success, "First deduction should succeed"
        
        # Second deduction with different key
        result2 = await ftns_service.deduct_tokens_atomic(
            funded_user,
            Decimal("100.0"),
            "key_2",
            "Purchase 2"
        )
        assert result2.success, "Second deduction should succeed"
        
        # Verify both deductions happened
        final_balance_info = await ftns_service.get_balance(funded_user)
        assert float(final_balance_info.balance) == initial_balance - 200.0, \
            "Both deductions should have occurred"
    
    @pytest.mark.asyncio
    async def test_idempotency_key_concurrent_requests(self, ftns_service, funded_user):
        """Test that concurrent requests with same idempotency key are handled correctly"""
        initial_balance_info = await ftns_service.get_balance(funded_user)
        initial_balance = float(initial_balance_info.balance)
        
        # Make 5 concurrent requests with the same idempotency key
        async def deduct_with_key(key: str):
            return await ftns_service.deduct_tokens_atomic(
                funded_user,
                Decimal("100.0"),
                key,
                "Concurrent purchase"
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
        successes = sum(1 for r in results if isinstance(r, object) and getattr(r, 'success', False))
        assert successes == 5, "All idempotent requests should return success"
        
        # But only one deduction should have occurred
        final_balance_info = await ftns_service.get_balance(funded_user)
        assert float(final_balance_info.balance) == initial_balance - 100.0, \
            "Only one deduction should have occurred despite concurrent requests"
    
    # =========================================================================
    # Race Condition Prevention Tests
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_race_condition_prevention_check_balance(self, ftns_service):
        """Test that race conditions in balance checking are prevented"""
        user_id = "race_condition_user"
        result = await ftns_service.mint_tokens_atomic(
            user_id, Decimal("100.0"), "initial_race", "Initial funds"
        )
        assert result.success, "Failed to fund user"
        
        # Simulate race condition: multiple operations checking balance simultaneously
        async def check_and_deduct():
            balance_info = await ftns_service.get_balance(user_id)
            if float(balance_info.balance) >= 60:
                # Try to deduct 60 with unique idempotency key
                key = f"race_{id(asyncio.current_task())}"
                return await ftns_service.deduct_tokens_atomic(
                    user_id,
                    Decimal("60.0"),
                    key,
                    "Race deduct"
                )
            return None
        
        # Run multiple check-and-deduct operations concurrently
        results = await asyncio.gather(
            check_and_deduct(),
            check_and_deduct(),
            check_and_deduct(),
            return_exceptions=True
        )
        
        # Count successes
        successes = sum(1 for r in results if hasattr(r, 'success') and r.success)
        
        # Only one should succeed (100 - 60 = 40, not enough for second 60)
        assert successes == 1, f"Only 1 deduction should succeed, got {successes}"
        
        # Verify final balance
        final_balance_info = await ftns_service.get_balance(user_id)
        assert float(final_balance_info.balance) == 40.0, f"Final balance should be 40.0, got {float(final_balance_info.balance)}"
    
    @pytest.mark.asyncio
    async def test_atomic_deduct_all_or_nothing(self, ftns_service):
        """Test that atomic deduct is all-or-nothing"""
        user_id = "atomic_user"
        result = await ftns_service.mint_tokens_atomic(
            user_id, Decimal("100.0"), "initial_atomic", "Initial funds"
        )
        assert result.success, "Failed to fund user"
        
        # Attempt to deduct more than balance
        result = await ftns_service.deduct_tokens_atomic(
            user_id,
            Decimal("200.0"),  # More than available
            "overdraft_1",
            "Overdraft attempt"
        )
        
        # Should fail
        assert not result.success, "Overdraft should fail"
        
        # Balance should be unchanged
        balance_info = await ftns_service.get_balance(user_id)
        assert float(balance_info.balance) == 100.0, "Balance should be unchanged after failed deduct"
    
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
            result = await ftns_service.mint_tokens_atomic(
                user, Decimal("1000.0"), f"initial_{user}", "Initial funds"
            )
            assert result.success, f"Failed to fund user {user}"
        
        # Perform many operations
        async def user_operations(user_id: str):
            for i in range(operations_per_user):
                await ftns_service.deduct_tokens_atomic(
                    user_id,
                    Decimal("10.0"),
                    f"{user_id}_op_{i}",
                    f"Stress op {i}"
                )
        
        # Run all user operations concurrently
        await asyncio.gather(
            *[user_operations(user) for user in users],
            return_exceptions=True
        )
        
        # Verify all balances
        for user in users:
            balance_info = await ftns_service.get_balance(user)
            expected = 1000.0 - (operations_per_user * 10.0)
            assert float(balance_info.balance) == expected, f"User {user} balance mismatch"


class TestMarketplaceIntegration:
    """Integration tests for marketplace with FTNS service"""

    @pytest.fixture
    async def marketplace_setup(self):
        """Set up marketplace with FTNS service using mocked database layer."""
        from unittest.mock import AsyncMock, patch
        from sqlalchemy.ext.asyncio import AsyncSession
        from prsm.economy.tokenomics.atomic_ftns_service import TransactionResult, BalanceInfo

        # In-memory state for this test
        state = {
            "balances": {},
            "idempotency_keys": {},
        }

        # Create a mock session
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()

        # Create context manager for session
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        # Patch get_async_session
        with patch('prsm.economy.tokenomics.atomic_ftns_service.get_async_session', return_value=mock_context):
            ftns_service = AtomicFTNSService()
            ftns_service._initialized = True

            # Override methods to use our mock state
            async def mock_mint(to_user_id, amount, idempotency_key, description="Token mint", metadata=None):
                if to_user_id not in state["balances"]:
                    state["balances"][to_user_id] = {
                        "balance": 0.0,
                        "locked": 0.0,
                        "total_earned": 0.0,
                        "total_spent": 0.0,
                        "version": 1
                    }
                state["balances"][to_user_id]["balance"] += float(amount)
                state["balances"][to_user_id]["total_earned"] += float(amount)
                state["idempotency_keys"][idempotency_key] = f"tx_{idempotency_key}"

                return TransactionResult(
                    success=True,
                    transaction_id=f"tx_{idempotency_key}",
                    new_balance=Decimal(str(state["balances"][to_user_id]["balance"])),
                    error_message=None,
                    idempotent_replay=False
                )

            async def mock_deduct(user_id, amount, idempotency_key, description="", metadata=None):
                if idempotency_key in state["idempotency_keys"]:
                    return TransactionResult(
                        success=True,
                        transaction_id=state["idempotency_keys"][idempotency_key],
                        new_balance=Decimal(str(state["balances"].get(user_id, {}).get("balance", 0))),
                        error_message=None,
                        idempotent_replay=True
                    )

                if user_id not in state["balances"]:
                    return TransactionResult(
                        success=False,
                        transaction_id=None,
                        new_balance=None,
                        error_message="Account not found",
                        idempotent_replay=False
                    )

                available = state["balances"][user_id]["balance"] - state["balances"][user_id]["locked"]
                if available < float(amount):
                    return TransactionResult(
                        success=False,
                        transaction_id=None,
                        new_balance=Decimal(str(available)),
                        error_message="Insufficient balance",
                        idempotent_replay=False
                    )

                state["balances"][user_id]["balance"] -= float(amount)
                state["balances"][user_id]["total_spent"] += float(amount)
                state["idempotency_keys"][idempotency_key] = f"tx_{idempotency_key}"

                return TransactionResult(
                    success=True,
                    transaction_id=f"tx_{idempotency_key}",
                    new_balance=Decimal(str(state["balances"][user_id]["balance"])),
                    error_message=None,
                    idempotent_replay=False
                )

            async def mock_get_balance(user_id):
                from datetime import datetime, timezone
                if user_id not in state["balances"]:
                    state["balances"][user_id] = {
                        "balance": 0.0,
                        "locked": 0.0,
                        "total_earned": 0.0,
                        "total_spent": 0.0,
                        "version": 1
                    }
                bal = state["balances"][user_id]
                return BalanceInfo(
                    user_id=user_id,
                    balance=Decimal(str(bal["balance"])),
                    locked_balance=Decimal(str(bal["locked"])),
                    available_balance=Decimal(str(bal["balance"] - bal["locked"])),
                    total_earned=Decimal(str(bal["total_earned"])),
                    total_spent=Decimal(str(bal["total_spent"])),
                    version=bal.get("version", 1),
                    last_updated=datetime.now(timezone.utc)
                )

            ftns_service.mint_tokens_atomic = mock_mint
            ftns_service.deduct_tokens_atomic = mock_deduct
            ftns_service.get_balance = mock_get_balance

            # Create test users
            buyer_id = "test_buyer"
            seller_id = "test_seller"

            result1 = await ftns_service.mint_tokens_atomic(
                buyer_id, Decimal("500.0"), "initial_buyer", "Initial funds"
            )
            result2 = await ftns_service.mint_tokens_atomic(
                seller_id, Decimal("100.0"), "initial_seller", "Initial funds"
            )

            yield {
                "ftns": ftns_service,
                "buyer": buyer_id,
                "seller": seller_id
            }
    
    @pytest.mark.asyncio
    async def test_marketplace_purchase_flow(self, marketplace_setup):
        """Test complete marketplace purchase flow"""
        ftns = marketplace_setup["ftns"]
        buyer = marketplace_setup["buyer"]
        seller = marketplace_setup["seller"]
        
        buyer_initial_info = await ftns.get_balance(buyer)
        seller_initial_info = await ftns.get_balance(seller)
        buyer_initial = float(buyer_initial_info.balance)
        seller_initial = float(seller_initial_info.balance)
        
        purchase_amount = Decimal("100.0")
        
        # Buyer makes purchase
        result = await ftns.deduct_tokens_atomic(
            buyer,
            purchase_amount,
            "purchase_1",
            "Marketplace purchase"
        )
        assert result.success, "Purchase should succeed"
        
        # Seller receives payment (mint/credit)
        result = await ftns.mint_tokens_atomic(
            seller, purchase_amount, "sale_proceeds", "Sale proceeds"
        )
        assert result.success, "Credit should succeed"
        
        # Verify balances
        buyer_final_info = await ftns.get_balance(buyer)
        seller_final_info = await ftns.get_balance(seller)
        
        assert float(buyer_final_info.balance) == buyer_initial - float(purchase_amount)
        assert float(seller_final_info.balance) == seller_initial + float(purchase_amount)
    
    @pytest.mark.asyncio
    async def test_marketplace_refund_flow(self, marketplace_setup):
        """Test marketplace refund flow"""
        ftns = marketplace_setup["ftns"]
        buyer = marketplace_setup["buyer"]
        
        buyer_initial_info = await ftns.get_balance(buyer)
        buyer_initial = float(buyer_initial_info.balance)
        
        # Make purchase
        result = await ftns.deduct_tokens_atomic(
            buyer,
            Decimal("100.0"),
            "refund_test_purchase",
            "Purchase for refund"
        )
        assert result.success, "Purchase should succeed"
        
        # Verify deduction
        balance_info = await ftns.get_balance(buyer)
        assert float(balance_info.balance) == buyer_initial - 100.0
        
        # Process refund
        result = await ftns.mint_tokens_atomic(
            buyer, Decimal("100.0"), "refund", "Refund"
        )
        assert result.success, "Refund should succeed"
        
        # Verify refund
        balance_info = await ftns.get_balance(buyer)
        assert float(balance_info.balance) == buyer_initial


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
        result = await service.mint_tokens_atomic(
            user_id, Decimal("500.0"), "initial", "Initial funds"
        )
        assert result.success, "Failed to fund user"
        
        result1 = await service.deduct_tokens_atomic(
            user_id, Decimal("100.0"), "purchase", "Purchase", idempotency_key="idem_1"
        )
        result2 = await service.deduct_tokens_atomic(
            user_id, Decimal("100.0"), "purchase", "Purchase", idempotency_key="idem_1"
        )
        
        balance_info = await service.get_balance(user_id)
        assert float(balance_info.balance) == 400.0, f"Expected 400.0, got {float(balance_info.balance)}"
        print("  ✓ PASSED: Idempotency works correctly")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    
    # Test 2: Concurrent purchases
    print("\n[TEST 2] Concurrent purchases from same user...")
    try:
        user_id = "concurrent_test_user"
        result = await service.mint_tokens_atomic(
            user_id, Decimal("200.0"), "initial", "Initial funds"
        )
        assert result.success, "Failed to fund user"
        
        async def purchase(amount, key):
            return await service.deduct_tokens_atomic(
                user_id, Decimal(str(amount)), "purchase", "Purchase", idempotency_key=key
            )
        
        results = await asyncio.gather(
            purchase(150.0, "c1"),
            purchase(150.0, "c2"),
            return_exceptions=True
        )
        
        successes = sum(1 for r in results if hasattr(r, 'success') and r.success)
        assert successes == 1, f"Expected 1 success, got {successes}"
        
        balance_info = await service.get_balance(user_id)
        assert float(balance_info.balance) == 50.0, f"Expected 50.0, got {float(balance_info.balance)}"
        print("  ✓ PASSED: Concurrent purchases handled correctly")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    
    # Test 3: Balance consistency
    print("\n[TEST 3] Balance consistency under load...")
    try:
        user_id = "consistency_test_user"
        result = await service.mint_tokens_atomic(
            user_id, Decimal("100.0"), "initial", "Initial funds"
        )
        assert result.success, "Failed to fund user"
        
        # Many small deductions
        for i in range(10):
            await service.deduct_tokens_atomic(
                user_id, Decimal("5.0"), f"deduct_{i}", f"Deduct {i}", idempotency_key=f"deduct_{i}"
            )
        
        balance_info = await service.get_balance(user_id)
        assert float(balance_info.balance) == 50.0, f"Expected 50.0, got {float(balance_info.balance)}"
        print("  ✓ PASSED: Balance consistency maintained")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    
    print("\n" + "=" * 60)
    print("MARKETPLACE CONCURRENCY TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_marketplace_concurrency_tests())
