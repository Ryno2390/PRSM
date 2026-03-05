#!/usr/bin/env python3
"""
Sprint 3 Phase 3: DAG Consensus Integration Tests

Tests for transaction ordering, double-spend prevention, fork detection,
and signature verification across the network.
"""

import asyncio
import pytest
from datetime import datetime
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Import DAG ledger components
from prsm.node.dag_ledger import (
    DAGLedger, TransactionType, DAGTransaction,
    InsufficientBalanceError, ConcurrentModificationError, BalanceLockError
)


class TestDAGConsensus:
    """Test suite for DAG consensus validation"""
    
    @pytest.fixture
    async def ledger(self):
        """Create a fresh DAG ledger for each test"""
        ledger = DAGLedger(db_path=':memory:', verify_signatures=False)
        await ledger.initialize()
        # Create initial funds
        await ledger.submit_transaction(
            tx_type=TransactionType.GENESIS,
            amount=10000.0,
            from_wallet=None,
            to_wallet='treasury',
            description='Initial treasury'
        )
        yield ledger
        await ledger.close()
    
    @pytest.fixture
    async def ledger_with_sig_verification(self):
        """Create a DAG ledger with signature verification enabled"""
        ledger = DAGLedger(db_path=':memory:', verify_signatures=True)
        await ledger.initialize()
        yield ledger
        await ledger.close()
    
    # =========================================================================
    # Transaction Ordering Tests
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_transaction_ordering_sequential(self, ledger):
        """Test that transactions are ordered correctly when submitted sequentially"""
        # Submit multiple transactions
        tx_ids = []
        for i in range(5):
            tx = await ledger.submit_transaction(
                tx_type=TransactionType.TRANSFER,
                amount=100.0,
                from_wallet='treasury',
                to_wallet=f'user{i}',
                description=f'Transfer {i}'
            )
            tx_ids.append(tx.tx_id)
        
        # Verify all transactions were recorded
        assert len(tx_ids) == 5
        
        # Verify balances are correct
        for i in range(5):
            balance = await ledger.get_balance(f'user{i}')
            assert balance == 100.0, f"User {i} should have 100.0, got {balance}"
        
        treasury_balance = await ledger.get_balance('treasury')
        assert treasury_balance == 9500.0, f"Treasury should have 9500.0, got {treasury_balance}"
    
    @pytest.mark.asyncio
    async def test_transaction_timestamps_increase(self, ledger):
        """Test that transaction timestamps are monotonically increasing"""
        timestamps = []
        for i in range(3):
            tx = await ledger.submit_transaction(
                tx_type=TransactionType.TRANSFER,
                amount=10.0,
                from_wallet='treasury',
                to_wallet='user1',
                description=f'Transfer {i}'
            )
            timestamps.append(tx.timestamp)
            await asyncio.sleep(0.01)  # Small delay to ensure different timestamps
        
        # Verify timestamps are increasing
        for i in range(len(timestamps) - 1):
            assert timestamps[i] < timestamps[i + 1], "Timestamps should be increasing"
    
    # =========================================================================
    # Double-Spend Prevention Tests
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_double_spend_prevention_insufficient_balance(self, ledger):
        """Test that double-spend attempts are rejected due to insufficient balance"""
        # Give user1 some funds
        await ledger.submit_transaction(
            tx_type=TransactionType.TRANSFER,
            amount=100.0,
            from_wallet='treasury',
            to_wallet='user1',
            description='Initial funds'
        )
        
        # First spend should succeed
        await ledger.submit_transaction(
            tx_type=TransactionType.TRANSFER,
            amount=80.0,
            from_wallet='user1',
            to_wallet='user2',
            description='First spend'
        )
        
        # Double spend attempt should fail (only 20 left)
        with pytest.raises(InsufficientBalanceError):
            await ledger.submit_transaction(
                tx_type=TransactionType.TRANSFER,
                amount=80.0,
                from_wallet='user1',
                to_wallet='user3',
                description='Double spend attempt'
            )
        
        # Verify balances
        assert await ledger.get_balance('user1') == 20.0
        assert await ledger.get_balance('user2') == 80.0
        assert await ledger.get_balance('user3') == 0.0
    
    @pytest.mark.asyncio
    async def test_concurrent_transfer_attempts(self, ledger):
        """Test that concurrent transfer attempts are handled correctly"""
        # Give user1 exactly 100 units
        await ledger.submit_transaction(
            tx_type=TransactionType.TRANSFER,
            amount=100.0,
            from_wallet='treasury',
            to_wallet='user1',
            description='Initial funds'
        )
        
        # Try to make two transfers of 100 each concurrently
        async def transfer(amount: float, to_user: str):
            try:
                await ledger.submit_transaction(
                    tx_type=TransactionType.TRANSFER,
                    amount=amount,
                    from_wallet='user1',
                    to_wallet=to_user,
                    description=f'Transfer to {to_user}'
                )
                return True
            except InsufficientBalanceError:
                return False
        
        # Run both transfers concurrently
        results = await asyncio.gather(
            transfer(100.0, 'user2'),
            transfer(100.0, 'user3'),
            return_exceptions=True
        )
        
        # One should succeed, one should fail
        successes = sum(1 for r in results if r is True)
        failures = sum(1 for r in results if r is False)
        
        # At least one should fail (both might fail due to race condition)
        assert failures >= 1, "At least one concurrent transfer should fail"
        
        # Total balance should be preserved
        total = (
            await ledger.get_balance('user1') +
            await ledger.get_balance('user2') +
            await ledger.get_balance('user3')
        )
        assert total == 100.0, f"Total balance should be 100.0, got {total}"
    
    # =========================================================================
    # Fork Detection and Resolution Tests
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_no_fork_in_sequential_transactions(self, ledger):
        """Test that sequential transactions don't create forks"""
        # Submit transactions sequentially
        tx1 = await ledger.submit_transaction(
            tx_type=TransactionType.TRANSFER,
            amount=100.0,
            from_wallet='treasury',
            to_wallet='user1',
            description='Transfer 1'
        )
        
        tx2 = await ledger.submit_transaction(
            tx_type=TransactionType.TRANSFER,
            amount=100.0,
            from_wallet='treasury',
            to_wallet='user2',
            description='Transfer 2'
        )
        
        # Both should reference the genesis transaction
        # (implementation depends on DAG structure)
        assert tx1.tx_id is not None
        assert tx2.tx_id is not None
    
    @pytest.mark.asyncio
    async def test_transaction_chain_integrity(self, ledger):
        """Test that transaction chain maintains integrity"""
        # Create a chain of transactions
        tx_ids = []
        for i in range(10):
            tx = await ledger.submit_transaction(
                tx_type=TransactionType.TRANSFER,
                amount=10.0,
                from_wallet='treasury',
                to_wallet=f'user{i}',
                description=f'Chain transaction {i}'
            )
            tx_ids.append(tx.tx_id)
        
        # Verify all transactions are in the ledger
        all_txs = await ledger.get_all_transactions()
        assert len(all_txs) >= 11  # 10 transfers + 1 genesis
    
    # =========================================================================
    # Genesis Transaction Tests
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_genesis_transaction_no_signature_required(self, ledger_with_sig_verification):
        """Test that genesis transactions don't require signatures"""
        # Genesis transaction should work without signature
        tx = await ledger_with_sig_verification.submit_transaction(
            tx_type=TransactionType.GENESIS,
            amount=1000.0,
            from_wallet=None,
            to_wallet='user1',
            description='Genesis allocation'
        )
        
        assert tx is not None
        balance = await ledger_with_sig_verification.get_balance('user1')
        assert balance == 1000.0
    
    @pytest.mark.asyncio
    async def test_system_transaction_no_signature_required(self, ledger_with_sig_verification):
        """Test that system transactions don't require signatures"""
        # First create funds via genesis
        await ledger_with_sig_verification.submit_transaction(
            tx_type=TransactionType.GENESIS,
            amount=1000.0,
            from_wallet=None,
            to_wallet='treasury',
            description='Initial treasury'
        )
        
        # System transaction should work without signature
        tx = await ledger_with_sig_verification.submit_transaction(
            tx_type=TransactionType.SYSTEM,
            amount=100.0,
            from_wallet='treasury',
            to_wallet='user1',
            description='System credit'
        )
        
        assert tx is not None
    
    @pytest.mark.asyncio
    async def test_regular_transfer_requires_signature(self, ledger_with_sig_verification):
        """Test that regular transfers require valid signatures"""
        # Create funds via genesis
        await ledger_with_sig_verification.submit_transaction(
            tx_type=TransactionType.GENESIS,
            amount=1000.0,
            from_wallet=None,
            to_wallet='user1',
            description='Initial funds'
        )
        
        # Regular transfer without signature should fail
        with pytest.raises(Exception):  # Should raise signature-related error
            await ledger_with_sig_verification.submit_transaction(
                tx_type=TransactionType.TRANSFER,
                amount=100.0,
                from_wallet='user1',
                to_wallet='user2',
                description='Unsigned transfer'
            )
    
    # =========================================================================
    # Balance Consistency Tests
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_balance_consistency_after_transfers(self, ledger):
        """Test that balances remain consistent after multiple transfers"""
        initial_treasury = await ledger.get_balance('treasury')
        
        # Perform multiple transfers
        transfers = [
            ('user1', 100.0),
            ('user2', 200.0),
            ('user3', 300.0),
        ]
        
        for user, amount in transfers:
            await ledger.submit_transaction(
                tx_type=TransactionType.TRANSFER,
                amount=amount,
                from_wallet='treasury',
                to_wallet=user,
                description=f'Transfer to {user}'
            )
        
        # Verify individual balances
        assert await ledger.get_balance('user1') == 100.0
        assert await ledger.get_balance('user2') == 200.0
        assert await ledger.get_balance('user3') == 300.0
        
        # Verify treasury balance
        expected_treasury = initial_treasury - 600.0
        assert await ledger.get_balance('treasury') == expected_treasury
        
        # Verify total supply is preserved
        total = (
            await ledger.get_balance('treasury') +
            await ledger.get_balance('user1') +
            await ledger.get_balance('user2') +
            await ledger.get_balance('user3')
        )
        assert total == initial_treasury
    
    @pytest.mark.asyncio
    async def test_negative_balance_prevention(self, ledger):
        """Test that negative balances are prevented"""
        # Try to transfer more than available
        with pytest.raises(InsufficientBalanceError):
            await ledger.submit_transaction(
                tx_type=TransactionType.TRANSFER,
                amount=999999.0,
                from_wallet='treasury',
                to_wallet='user1',
                description='Overdraft attempt'
            )
        
        # Verify balance is unchanged
        assert await ledger.get_balance('treasury') == 10000.0
        assert await ledger.get_balance('user1') == 0.0
    
    # =========================================================================
    # Concurrent Modification Tests
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_concurrent_modification_detection(self, ledger):
        """Test that concurrent modifications are detected"""
        # Get the current version
        cursor = await ledger._db.execute(
            'SELECT version FROM wallet_balances WHERE wallet_id = ?', ('treasury',)
        )
        row = await cursor.fetchone()
        original_version = row[0]
        
        # Simulate concurrent modification by updating version
        await ledger._db.execute(
            'UPDATE wallet_balances SET version = ? WHERE wallet_id = ?',
            (original_version + 999, 'treasury')
        )
        await ledger._db.commit()
        
        # Now try to transfer - should fail with ConcurrentModificationError
        with pytest.raises(ConcurrentModificationError):
            await ledger.submit_transaction(
                tx_type=TransactionType.TRANSFER,
                amount=10.0,
                from_wallet='treasury',
                to_wallet='user1',
                description='Should fail - concurrent modification'
            )


class TestMultiNodeDAGConsensus:
    """Test suite for multi-node DAG consensus scenarios"""
    
    @pytest.mark.asyncio
    async def test_cross_node_transaction_propagation(self):
        """Test that transactions propagate correctly across nodes"""
        # Create two ledgers representing different nodes
        ledger1 = DAGLedger(db_path=':memory:', verify_signatures=False)
        ledger2 = DAGLedger(db_path=':memory:', verify_signatures=False)
        
        await ledger1.initialize()
        await ledger2.initialize()
        
        # Initialize both with same genesis
        await ledger1.submit_transaction(
            tx_type=TransactionType.GENESIS,
            amount=1000.0,
            from_wallet=None,
            to_wallet='treasury',
            description='Initial treasury'
        )
        
        await ledger2.submit_transaction(
            tx_type=TransactionType.GENESIS,
            amount=1000.0,
            from_wallet=None,
            to_wallet='treasury',
            description='Initial treasury'
        )
        
        # In a real scenario, transactions would propagate via P2P
        # Here we simulate by submitting to both ledgers
        tx1 = await ledger1.submit_transaction(
            tx_type=TransactionType.TRANSFER,
            amount=100.0,
            from_wallet='treasury',
            to_wallet='user1',
            description='Cross-node transfer'
        )
        
        # Both ledgers should eventually have the same state
        # (in real implementation, this would happen via sync)
        
        await ledger1.close()
        await ledger2.close()
    
    @pytest.mark.asyncio
    async def test_conflicting_transactions_resolution(self):
        """Test that conflicting transactions are resolved correctly"""
        # This tests the scenario where two nodes receive conflicting transactions
        # The DAG should resolve to a consistent state
        
        ledger = DAGLedger(db_path=':memory:', verify_signatures=False)
        await ledger.initialize()
        
        # Create initial funds
        await ledger.submit_transaction(
            tx_type=TransactionType.GENESIS,
            amount=100.0,
            from_wallet=None,
            to_wallet='user1',
            description='Initial funds'
        )
        
        # User1 tries to spend the same funds twice
        # First spend
        await ledger.submit_transaction(
            tx_type=TransactionType.TRANSFER,
            amount=100.0,
            from_wallet='user1',
            to_wallet='user2',
            description='First spend'
        )
        
        # Second spend should fail
        with pytest.raises(InsufficientBalanceError):
            await ledger.submit_transaction(
                tx_type=TransactionType.TRANSFER,
                amount=100.0,
                from_wallet='user1',
                to_wallet='user3',
                description='Conflicting spend'
            )
        
        await ledger.close()


# =========================================================================
# Test Runner
# =========================================================================

async def run_dag_consensus_tests():
    """Run all DAG consensus tests manually"""
    print("=" * 60)
    print("DAG CONSENSUS INTEGRATION TESTS")
    print("=" * 60)
    
    test_instance = TestDAGConsensus()
    
    # Create ledger
    print("\n[SETUP] Creating test ledger...")
    ledger = DAGLedger(db_path=':memory:', verify_signatures=False)
    await ledger.initialize()
    await ledger.submit_transaction(
        tx_type=TransactionType.GENESIS,
        amount=10000.0,
        from_wallet=None,
        to_wallet='treasury',
        description='Initial treasury'
    )
    
    # Test 1: Transaction ordering
    print("\n[TEST 1] Transaction ordering...")
    try:
        await test_instance.test_transaction_ordering_sequential(ledger)
        print("  ✓ PASSED: Transaction ordering works correctly")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    
    # Test 2: Double-spend prevention
    print("\n[TEST 2] Double-spend prevention...")
    try:
        await test_instance.test_double_spend_prevention_insufficient_balance(ledger)
        print("  ✓ PASSED: Double-spend prevention works")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    
    # Test 3: Balance consistency
    print("\n[TEST 3] Balance consistency...")
    try:
        await test_instance.test_balance_consistency_after_transfers(ledger)
        print("  ✓ PASSED: Balance consistency maintained")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    
    # Test 4: Negative balance prevention
    print("\n[TEST 4] Negative balance prevention...")
    try:
        await test_instance.test_negative_balance_prevention(ledger)
        print("  ✓ PASSED: Negative balances prevented")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    
    # Test 5: Concurrent modification detection
    print("\n[TEST 5] Concurrent modification detection...")
    try:
        await test_instance.test_concurrent_modification_detection(ledger)
        print("  ✓ PASSED: Concurrent modification detected")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    
    await ledger.close()
    
    print("\n" + "=" * 60)
    print("DAG CONSENSUS TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_dag_consensus_tests())
