"""
Tests for prsm.node.local_ledger — SQLite FTNS ledger.
"""

import pytest

from prsm.node.local_ledger import LocalLedger, TransactionType


@pytest.fixture
async def ledger():
    """Create an in-memory ledger for testing."""
    db = LocalLedger(":memory:")
    await db.initialize()
    yield db
    await db.close()


class TestWalletManagement:
    @pytest.mark.asyncio
    async def test_create_wallet(self, ledger):
        await ledger.create_wallet("node-1", "Test Node")
        assert await ledger.wallet_exists("node-1")

    @pytest.mark.asyncio
    async def test_wallet_not_exists(self, ledger):
        assert not await ledger.wallet_exists("nonexistent")

    @pytest.mark.asyncio
    async def test_create_wallet_idempotent(self, ledger):
        await ledger.create_wallet("node-1")
        await ledger.create_wallet("node-1")  # should not raise
        assert await ledger.wallet_exists("node-1")


class TestBalances:
    @pytest.mark.asyncio
    async def test_new_wallet_zero_balance(self, ledger):
        await ledger.create_wallet("node-1")
        balance = await ledger.get_balance("node-1")
        assert balance == 0.0

    @pytest.mark.asyncio
    async def test_credit_increases_balance(self, ledger):
        await ledger.create_wallet("node-1")
        await ledger.credit("node-1", 50.0, TransactionType.WELCOME_GRANT)
        balance = await ledger.get_balance("node-1")
        assert balance == 50.0

    @pytest.mark.asyncio
    async def test_multiple_credits(self, ledger):
        await ledger.create_wallet("node-1")
        await ledger.credit("node-1", 100.0, TransactionType.WELCOME_GRANT)
        await ledger.credit("node-1", 25.5, TransactionType.COMPUTE_EARNING)
        balance = await ledger.get_balance("node-1")
        assert balance == 125.5

    @pytest.mark.asyncio
    async def test_debit_decreases_balance(self, ledger):
        await ledger.create_wallet("node-1")
        await ledger.create_wallet("system")
        await ledger.credit("node-1", 100.0, TransactionType.WELCOME_GRANT)
        await ledger.debit("node-1", 30.0, TransactionType.COMPUTE_PAYMENT)
        balance = await ledger.get_balance("node-1")
        assert balance == 70.0

    @pytest.mark.asyncio
    async def test_debit_insufficient_balance_raises(self, ledger):
        await ledger.create_wallet("node-1")
        await ledger.create_wallet("system")
        await ledger.credit("node-1", 10.0, TransactionType.WELCOME_GRANT)
        with pytest.raises(ValueError, match="Insufficient balance"):
            await ledger.debit("node-1", 20.0, TransactionType.COMPUTE_PAYMENT)


class TestTransfers:
    @pytest.mark.asyncio
    async def test_transfer_between_wallets(self, ledger):
        await ledger.create_wallet("alice")
        await ledger.create_wallet("bob")
        await ledger.credit("alice", 100.0, TransactionType.WELCOME_GRANT)

        await ledger.transfer("alice", "bob", 40.0)
        assert await ledger.get_balance("alice") == 60.0
        assert await ledger.get_balance("bob") == 40.0

    @pytest.mark.asyncio
    async def test_transfer_insufficient_balance(self, ledger):
        await ledger.create_wallet("alice")
        await ledger.create_wallet("bob")
        await ledger.credit("alice", 10.0, TransactionType.WELCOME_GRANT)

        with pytest.raises(ValueError, match="Insufficient balance"):
            await ledger.transfer("alice", "bob", 20.0)

    @pytest.mark.asyncio
    async def test_transfer_preserves_total(self, ledger):
        await ledger.create_wallet("a")
        await ledger.create_wallet("b")
        await ledger.credit("a", 100.0, TransactionType.WELCOME_GRANT)

        await ledger.transfer("a", "b", 30.0)
        await ledger.transfer("b", "a", 10.0)

        total = await ledger.get_balance("a") + await ledger.get_balance("b")
        assert total == 100.0


class TestWelcomeGrant:
    @pytest.mark.asyncio
    async def test_welcome_grant(self, ledger):
        await ledger.create_wallet("node-1")
        tx = await ledger.issue_welcome_grant("node-1", 100.0)
        assert tx.amount == 100.0
        assert tx.tx_type == TransactionType.WELCOME_GRANT
        assert await ledger.get_balance("node-1") == 100.0

    @pytest.mark.asyncio
    async def test_welcome_grant_only_once(self, ledger):
        await ledger.create_wallet("node-1")
        await ledger.issue_welcome_grant("node-1")
        with pytest.raises(ValueError, match="already received"):
            await ledger.issue_welcome_grant("node-1")


class TestTransactionHistory:
    @pytest.mark.asyncio
    async def test_history_records_transactions(self, ledger):
        await ledger.create_wallet("node-1")
        await ledger.credit("node-1", 100.0, TransactionType.WELCOME_GRANT, "Welcome")
        await ledger.credit("node-1", 5.0, TransactionType.COMPUTE_EARNING, "Job done")

        history = await ledger.get_transaction_history("node-1")
        assert len(history) == 2
        # Most recent first
        assert history[0].description == "Job done"
        assert history[1].description == "Welcome"

    @pytest.mark.asyncio
    async def test_history_limit(self, ledger):
        await ledger.create_wallet("node-1")
        for i in range(10):
            await ledger.credit("node-1", 1.0, TransactionType.STORAGE_REWARD, f"tx-{i}")

        history = await ledger.get_transaction_history("node-1", limit=5)
        assert len(history) == 5

    @pytest.mark.asyncio
    async def test_transaction_count(self, ledger):
        await ledger.create_wallet("node-1")
        await ledger.credit("node-1", 10.0, TransactionType.WELCOME_GRANT)
        await ledger.credit("node-1", 5.0, TransactionType.COMPUTE_EARNING)
        count = await ledger.get_transaction_count("node-1")
        assert count == 2

    @pytest.mark.asyncio
    async def test_transaction_has_unique_ids(self, ledger):
        await ledger.create_wallet("node-1")
        tx1 = await ledger.credit("node-1", 10.0, TransactionType.WELCOME_GRANT)
        tx2 = await ledger.credit("node-1", 5.0, TransactionType.COMPUTE_EARNING)
        assert tx1.tx_id != tx2.tx_id
