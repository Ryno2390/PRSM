"""
FTNS Concurrency Integration Tests
==================================

Integration tests that verify double-spend prevention with a real PostgreSQL database.
These tests validate:
- SELECT FOR UPDATE row-level locking
- Optimistic concurrency control (OCC) with version columns
- Idempotency key enforcement
- Atomic transfer consistency

Requirements:
- PostgreSQL database (configured via DATABASE_URL env var)
- Test database with migrations applied

Run with: pytest tests/integration/test_ftns_concurrency_integration.py -v --tb=short
"""

import asyncio
import os
import pytest
from decimal import Decimal
from uuid import uuid4
from typing import List, Dict, Any
from contextlib import asynccontextmanager

try:
    import asyncpg
except ImportError:
    pytest.skip("asyncpg not installed", allow_module_level=True)


# Skip if no database URL configured
DATABASE_URL = os.environ.get("DATABASE_URL", os.environ.get("TEST_DATABASE_URL"))
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="DATABASE_URL or TEST_DATABASE_URL environment variable not set"
)


class TestDatabase:
    """Test database helper for managing connections and test data."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None

    async def connect(self):
        """Create connection pool."""
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=5,
            max_size=20
        )

    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()

    async def execute(self, query: str, *args) -> str:
        """Execute a query."""
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetchrow(self, query: str, *args) -> asyncpg.Record:
        """Fetch a single row."""
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetch(self, query: str, *args) -> List[asyncpg.Record]:
        """Fetch multiple rows."""
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    @asynccontextmanager
    async def transaction(self):
        """Get a connection with transaction."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                yield conn


@pytest.fixture(scope="module")
async def db():
    """Create database connection for tests."""
    test_db = TestDatabase(DATABASE_URL)
    await test_db.connect()
    yield test_db
    await test_db.close()


@pytest.fixture
async def test_users(db):
    """Create test user accounts for each test."""
    user_ids = [f"test_user_{uuid4().hex[:8]}" for _ in range(3)]

    # Create test accounts
    for user_id in user_ids:
        await db.execute("""
            INSERT INTO ftns_balances
            (user_id, balance, locked_balance, total_earned, total_spent, account_type, version)
            VALUES ($1, 1000, 0, 1000, 0, 'user', 1)
            ON CONFLICT (user_id) DO UPDATE SET
                balance = 1000,
                locked_balance = 0,
                version = 1
        """, user_id)

    yield user_ids

    # Cleanup
    for user_id in user_ids:
        await db.execute("""
            DELETE FROM ftns_transactions WHERE from_user_id = $1 OR to_user_id = $1
        """, user_id)
        await db.execute("""
            DELETE FROM ftns_idempotency_keys WHERE user_id = $1
        """, user_id)
        await db.execute("""
            DELETE FROM ftns_balances WHERE user_id = $1
        """, user_id)


class TestRowLevelLocking:
    """Test SELECT FOR UPDATE row-level locking prevents double-spend."""

    @pytest.mark.asyncio
    async def test_concurrent_deductions_serialize(self, db, test_users):
        """Verify concurrent deductions are properly serialized by row locks."""
        user_id = test_users[0]
        initial_balance = Decimal("1000")
        deduction_amount = Decimal("100")
        num_concurrent = 15  # Try 15 concurrent deductions of 100

        # Verify initial balance
        row = await db.fetchrow(
            "SELECT balance FROM ftns_balances WHERE user_id = $1",
            user_id
        )
        assert Decimal(str(row['balance'])) == initial_balance

        # Track results
        results = {"success": 0, "failed": 0, "lock_error": 0}

        async def attempt_deduction(attempt_id: int):
            """Attempt a deduction with FOR UPDATE locking."""
            idempotency_key = f"deduct_{user_id}_{attempt_id}_{uuid4().hex[:8]}"

            async with db.pool.acquire() as conn:
                try:
                    async with conn.transaction():
                        # Check idempotency first
                        existing = await conn.fetchrow("""
                            SELECT transaction_id FROM ftns_idempotency_keys
                            WHERE idempotency_key = $1
                        """, idempotency_key)

                        if existing:
                            return {"status": "duplicate", "attempt": attempt_id}

                        # Acquire row lock with NOWAIT to detect contention
                        try:
                            row = await conn.fetchrow("""
                                SELECT balance, locked_balance, version
                                FROM ftns_balances
                                WHERE user_id = $1
                                FOR UPDATE NOWAIT
                            """, user_id)
                        except asyncpg.exceptions.LockNotAvailableError:
                            # Lock contention - this is expected behavior
                            results["lock_error"] += 1
                            return {"status": "lock_contention", "attempt": attempt_id}

                        if not row:
                            return {"status": "not_found", "attempt": attempt_id}

                        current_balance = Decimal(str(row['balance']))
                        available = current_balance - Decimal(str(row['locked_balance']))

                        if available < deduction_amount:
                            results["failed"] += 1
                            return {
                                "status": "insufficient",
                                "attempt": attempt_id,
                                "available": float(available)
                            }

                        # Perform deduction with OCC
                        current_version = row['version']
                        result = await conn.execute("""
                            UPDATE ftns_balances
                            SET balance = balance - $1,
                                total_spent = total_spent + $1,
                                version = version + 1,
                                updated_at = NOW()
                            WHERE user_id = $2 AND version = $3
                        """, deduction_amount, user_id, current_version)

                        if result == "UPDATE 0":
                            results["failed"] += 1
                            return {"status": "version_conflict", "attempt": attempt_id}

                        # Record idempotency
                        tx_id = f"tx_{uuid4().hex[:12]}"
                        await conn.execute("""
                            INSERT INTO ftns_idempotency_keys
                            (idempotency_key, transaction_id, user_id, operation_type, amount, status)
                            VALUES ($1, $2, $3, 'deduction', $4, 'completed')
                        """, idempotency_key, tx_id, user_id, deduction_amount)

                        results["success"] += 1
                        return {"status": "success", "attempt": attempt_id, "tx_id": tx_id}

                except Exception as e:
                    return {"status": "error", "attempt": attempt_id, "error": str(e)}

        # Run concurrent deductions
        tasks = [attempt_deduction(i) for i in range(num_concurrent)]
        attempt_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify final balance
        row = await db.fetchrow(
            "SELECT balance, total_spent FROM ftns_balances WHERE user_id = $1",
            user_id
        )
        final_balance = Decimal(str(row['balance']))

        # With 1000 initial and 100 per deduction, max 10 can succeed
        # But with lock contention, some may fail to acquire locks
        successful_count = results["success"]
        expected_balance = initial_balance - (deduction_amount * successful_count)

        assert final_balance == expected_balance, \
            f"Balance mismatch: expected {expected_balance}, got {final_balance}"

        # Verify we never went negative
        assert final_balance >= 0, "Balance went negative - double-spend occurred!"

        # Log results for analysis
        print(f"\nConcurrency test results:")
        print(f"  Successful: {results['success']}")
        print(f"  Failed (insufficient): {results['failed']}")
        print(f"  Lock contention: {results['lock_error']}")
        print(f"  Final balance: {final_balance}")


class TestOptimisticConcurrencyControl:
    """Test version-based optimistic concurrency control."""

    @pytest.mark.asyncio
    async def test_version_mismatch_detection(self, db, test_users):
        """Verify version mismatch is detected and prevents update."""
        user_id = test_users[0]

        # Get current version
        row = await db.fetchrow(
            "SELECT balance, version FROM ftns_balances WHERE user_id = $1",
            user_id
        )
        current_version = row['version']
        current_balance = Decimal(str(row['balance']))

        # Update with correct version (should succeed)
        result = await db.execute("""
            UPDATE ftns_balances
            SET balance = balance - 50, version = version + 1
            WHERE user_id = $1 AND version = $2
        """, user_id, current_version)

        assert result == "UPDATE 1", "First update should succeed"

        # Try to update with old version (should fail)
        result = await db.execute("""
            UPDATE ftns_balances
            SET balance = balance - 50, version = version + 1
            WHERE user_id = $1 AND version = $2
        """, user_id, current_version)  # Using old version

        assert result == "UPDATE 0", "Update with stale version should fail"

        # Verify balance only changed once
        row = await db.fetchrow(
            "SELECT balance FROM ftns_balances WHERE user_id = $1",
            user_id
        )
        assert Decimal(str(row['balance'])) == current_balance - 50

    @pytest.mark.asyncio
    async def test_concurrent_version_updates(self, db, test_users):
        """Test that concurrent updates with same version result in only one success."""
        user_id = test_users[0]

        # Reset balance
        await db.execute("""
            UPDATE ftns_balances SET balance = 1000, version = 1 WHERE user_id = $1
        """, user_id)

        # Get initial version
        row = await db.fetchrow(
            "SELECT version FROM ftns_balances WHERE user_id = $1",
            user_id
        )
        initial_version = row['version']

        successful_updates = []

        async def attempt_update(update_id: int):
            async with db.pool.acquire() as conn:
                # Simulate reading version at same time
                await asyncio.sleep(0.01)  # Small delay to simulate concurrent read

                result = await conn.execute("""
                    UPDATE ftns_balances
                    SET balance = balance - 100, version = version + 1
                    WHERE user_id = $1 AND version = $2
                """, user_id, initial_version)

                if result == "UPDATE 1":
                    successful_updates.append(update_id)
                    return True
                return False

        # Run concurrent updates
        tasks = [attempt_update(i) for i in range(10)]
        await asyncio.gather(*tasks)

        # Only one should succeed due to OCC
        assert len(successful_updates) == 1, \
            f"Expected 1 successful update, got {len(successful_updates)}"


class TestIdempotencyKeys:
    """Test idempotency key enforcement."""

    @pytest.mark.asyncio
    async def test_duplicate_request_returns_same_result(self, db, test_users):
        """Verify duplicate idempotency key returns cached result."""
        user_id = test_users[0]
        idempotency_key = f"idem_test_{uuid4().hex}"

        # First request
        tx_id_1 = f"tx_{uuid4().hex[:12]}"
        await db.execute("""
            INSERT INTO ftns_idempotency_keys
            (idempotency_key, transaction_id, user_id, operation_type, amount, status)
            VALUES ($1, $2, $3, 'deduction', 100, 'completed')
        """, idempotency_key, tx_id_1, user_id)

        # Check for duplicate
        row = await db.fetchrow("""
            SELECT transaction_id FROM ftns_idempotency_keys
            WHERE idempotency_key = $1
        """, idempotency_key)

        assert row is not None
        assert row['transaction_id'] == tx_id_1

        # Try to insert duplicate (should fail or be ignored)
        try:
            await db.execute("""
                INSERT INTO ftns_idempotency_keys
                (idempotency_key, transaction_id, user_id, operation_type, amount, status)
                VALUES ($1, $2, $3, 'deduction', 100, 'completed')
            """, idempotency_key, f"tx_{uuid4().hex[:12]}", user_id)
        except asyncpg.exceptions.UniqueViolationError:
            pass  # Expected

        # Verify original remains
        row = await db.fetchrow("""
            SELECT transaction_id FROM ftns_idempotency_keys
            WHERE idempotency_key = $1
        """, idempotency_key)

        assert row['transaction_id'] == tx_id_1

    @pytest.mark.asyncio
    async def test_concurrent_same_idempotency_key(self, db, test_users):
        """Test concurrent requests with same idempotency key."""
        user_id = test_users[0]
        idempotency_key = f"concurrent_idem_{uuid4().hex}"

        successful_inserts = []
        failed_inserts = []

        async def try_insert(attempt_id: int):
            tx_id = f"tx_{attempt_id}_{uuid4().hex[:8]}"
            async with db.pool.acquire() as conn:
                try:
                    await conn.execute("""
                        INSERT INTO ftns_idempotency_keys
                        (idempotency_key, transaction_id, user_id, operation_type, amount, status)
                        VALUES ($1, $2, $3, 'deduction', 100, 'completed')
                    """, idempotency_key, tx_id, user_id)
                    successful_inserts.append((attempt_id, tx_id))
                except asyncpg.exceptions.UniqueViolationError:
                    failed_inserts.append(attempt_id)

        # Run concurrent inserts
        tasks = [try_insert(i) for i in range(10)]
        await asyncio.gather(*tasks)

        # Only one should succeed
        assert len(successful_inserts) == 1, \
            f"Expected 1 successful insert, got {len(successful_inserts)}"
        assert len(failed_inserts) == 9


class TestAtomicTransfers:
    """Test atomic transfers between accounts."""

    @pytest.mark.asyncio
    async def test_transfer_atomic_consistency(self, db, test_users):
        """Verify transfers are atomic - both accounts updated or neither."""
        sender_id = test_users[0]
        receiver_id = test_users[1]

        # Reset balances
        await db.execute("""
            UPDATE ftns_balances SET balance = 1000, version = 1 WHERE user_id = $1
        """, sender_id)
        await db.execute("""
            UPDATE ftns_balances SET balance = 500, version = 1 WHERE user_id = $1
        """, receiver_id)

        transfer_amount = Decimal("250")

        async with db.transaction() as conn:
            # Lock both accounts in consistent order
            user_ids = sorted([sender_id, receiver_id])
            rows = await conn.fetch("""
                SELECT user_id, balance, version
                FROM ftns_balances
                WHERE user_id = ANY($1)
                ORDER BY user_id
                FOR UPDATE
            """, user_ids)

            balances = {row['user_id']: row for row in rows}
            sender = balances[sender_id]
            receiver = balances[receiver_id]

            # Deduct from sender
            await conn.execute("""
                UPDATE ftns_balances
                SET balance = balance - $1, version = version + 1
                WHERE user_id = $2 AND version = $3
            """, transfer_amount, sender_id, sender['version'])

            # Credit to receiver
            await conn.execute("""
                UPDATE ftns_balances
                SET balance = balance + $1, version = version + 1
                WHERE user_id = $2 AND version = $3
            """, transfer_amount, receiver_id, receiver['version'])

        # Verify balances
        sender_row = await db.fetchrow(
            "SELECT balance FROM ftns_balances WHERE user_id = $1",
            sender_id
        )
        receiver_row = await db.fetchrow(
            "SELECT balance FROM ftns_balances WHERE user_id = $1",
            receiver_id
        )

        assert Decimal(str(sender_row['balance'])) == Decimal("750")
        assert Decimal(str(receiver_row['balance'])) == Decimal("750")

    @pytest.mark.asyncio
    async def test_transfer_rollback_on_insufficient_funds(self, db, test_users):
        """Verify transfer rolls back if sender has insufficient funds."""
        sender_id = test_users[0]
        receiver_id = test_users[1]

        # Set sender with low balance
        await db.execute("""
            UPDATE ftns_balances SET balance = 100, version = 1 WHERE user_id = $1
        """, sender_id)
        await db.execute("""
            UPDATE ftns_balances SET balance = 500, version = 1 WHERE user_id = $1
        """, receiver_id)

        transfer_amount = Decimal("500")  # More than sender has

        try:
            async with db.transaction() as conn:
                sender = await conn.fetchrow("""
                    SELECT balance, version FROM ftns_balances
                    WHERE user_id = $1 FOR UPDATE
                """, sender_id)

                if Decimal(str(sender['balance'])) < transfer_amount:
                    raise ValueError("Insufficient funds")

                # This shouldn't execute
                await conn.execute("""
                    UPDATE ftns_balances SET balance = balance - $1 WHERE user_id = $2
                """, transfer_amount, sender_id)

        except ValueError:
            pass  # Expected

        # Verify balances unchanged
        sender_row = await db.fetchrow(
            "SELECT balance FROM ftns_balances WHERE user_id = $1",
            sender_id
        )
        receiver_row = await db.fetchrow(
            "SELECT balance FROM ftns_balances WHERE user_id = $1",
            receiver_id
        )

        assert Decimal(str(sender_row['balance'])) == Decimal("100")
        assert Decimal(str(receiver_row['balance'])) == Decimal("500")


class TestRaceConditionSimulation:
    """Simulate real-world race condition scenarios."""

    @pytest.mark.asyncio
    async def test_double_spend_attack_prevention(self, db, test_users):
        """
        Simulate a double-spend attack where attacker tries to spend
        the same tokens multiple times simultaneously.
        """
        attacker_id = test_users[0]
        victim_ids = [test_users[1], test_users[2]]

        # Attacker has 500 tokens
        await db.execute("""
            UPDATE ftns_balances SET balance = 500, version = 1 WHERE user_id = $1
        """, attacker_id)

        # Each victim reset
        for victim_id in victim_ids:
            await db.execute("""
                UPDATE ftns_balances SET balance = 0, version = 1 WHERE user_id = $1
            """, victim_id)

        # Attacker tries to send 400 tokens to each victim simultaneously
        attack_amount = Decimal("400")
        results = []

        async def attack_transfer(victim_id: str, attack_id: int):
            idempotency_key = f"attack_{attack_id}_{uuid4().hex}"
            async with db.pool.acquire() as conn:
                try:
                    async with conn.transaction():
                        # Lock attacker account
                        attacker = await conn.fetchrow("""
                            SELECT balance, version FROM ftns_balances
                            WHERE user_id = $1 FOR UPDATE NOWAIT
                        """, attacker_id)

                        if not attacker:
                            return {"success": False, "reason": "not_found"}

                        available = Decimal(str(attacker['balance']))
                        if available < attack_amount:
                            return {"success": False, "reason": "insufficient"}

                        # Deduct from attacker
                        result = await conn.execute("""
                            UPDATE ftns_balances
                            SET balance = balance - $1, version = version + 1
                            WHERE user_id = $2 AND version = $3
                        """, attack_amount, attacker_id, attacker['version'])

                        if result == "UPDATE 0":
                            return {"success": False, "reason": "version_conflict"}

                        # Credit victim
                        await conn.execute("""
                            UPDATE ftns_balances
                            SET balance = balance + $1, version = version + 1
                            WHERE user_id = $2
                        """, attack_amount, victim_id)

                        return {"success": True, "victim": victim_id}

                except asyncpg.exceptions.LockNotAvailableError:
                    return {"success": False, "reason": "lock_contention"}
                except Exception as e:
                    return {"success": False, "reason": str(e)}

        # Launch simultaneous attacks
        tasks = [attack_transfer(victim_ids[i % 2], i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        successful = [r for r in results if r.get("success")]

        # Verify attacker balance
        attacker_row = await db.fetchrow(
            "SELECT balance FROM ftns_balances WHERE user_id = $1",
            attacker_id
        )
        attacker_balance = Decimal(str(attacker_row['balance']))

        # With 500 initial and 400 per transfer, only 1 can succeed
        assert len(successful) <= 1, \
            f"Double-spend detected! {len(successful)} transfers succeeded"

        if len(successful) == 1:
            assert attacker_balance == Decimal("100")
        else:
            assert attacker_balance == Decimal("500")

        # Verify total tokens in system unchanged (conservation)
        total = await db.fetchrow("""
            SELECT SUM(balance) as total FROM ftns_balances
            WHERE user_id = ANY($1)
        """, [attacker_id] + victim_ids)

        assert Decimal(str(total['total'])) == Decimal("500"), \
            "Token conservation violated!"

        print(f"\nDouble-spend attack results:")
        print(f"  Successful transfers: {len(successful)}")
        print(f"  Attacker balance: {attacker_balance}")


class TestDatabaseConstraints:
    """Test database-level constraints."""

    @pytest.mark.asyncio
    async def test_balance_cannot_go_negative(self, db, test_users):
        """Verify CHECK constraint prevents negative balance."""
        user_id = test_users[0]

        # Set balance to 100
        await db.execute("""
            UPDATE ftns_balances SET balance = 100 WHERE user_id = $1
        """, user_id)

        # Try to set negative balance directly
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await db.execute("""
                UPDATE ftns_balances SET balance = -50 WHERE user_id = $1
            """, user_id)

    @pytest.mark.asyncio
    async def test_locked_balance_constraint(self, db, test_users):
        """Verify locked balance cannot exceed total balance."""
        user_id = test_users[0]

        # Set balance to 100
        await db.execute("""
            UPDATE ftns_balances SET balance = 100, locked_balance = 0 WHERE user_id = $1
        """, user_id)

        # Try to lock more than balance
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await db.execute("""
                UPDATE ftns_balances SET locked_balance = 150 WHERE user_id = $1
            """, user_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
