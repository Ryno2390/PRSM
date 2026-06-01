"""Sprint 912 — governance system-mint DURABLE idempotency (Design A).

The money-path review's last deferred item. Sprint 908 closed the *critical*
authz hole (self-select CORE_TEAM) + the shared eval-once UUID. This closes
the remaining durability gap: both governance system-mint paths lacked any
cross-restart / retry idempotency.

  * ``distribute_contribution_rewards`` — NO dedup on
    ``(user, contribution_type, contribution_reference)`` → a retried
    contribution minted again, unbounded.
  * ``activate_governance_participation`` — guarded ONLY by the in-memory
    ``self.governance_activations`` dict, so a daemon RESTART re-minted the
    full tier allocation (COMMUNITY 1k … CORE_TEAM 100k FTNS).

Root cause: the mint's ``reference_id`` was ``str(distribution_id)`` —
per-instance (sp908) but RANDOM per call, so it could never anchor "have I
already done this logical mint?".

Design A (no schema change): the mint's ``reference_id`` becomes a
DETERMINISTIC key (``gov-contrib:{user}:{type}:{ref}`` /
``gov-activate:{user}``); each path consults
``DatabaseFTNSService.find_confirmed_transaction(reference_id,
transaction_type)`` under the EXISTING asyncio lock BEFORE minting, and
short-circuits to an idempotent no-op (returning the prior record) when a
confirmed mint with that key already exists.
"""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from prsm.economy.governance import token_distribution as td
from prsm.economy.governance.token_distribution import (
    ContributionType,
    GovernanceParticipantTier,
)
from prsm.economy.tokenomics.database_ftns_service import DatabaseFTNSService
from prsm.economy.tokenomics.models import Base, FTNSTransaction, TransactionStatus


# ── A faithful in-memory durable store (NOT a self-asserting mock) ───────
class _FakeDurableFtns:
    """In-memory implementation of the DatabaseFTNSService dedup contract.

    ``create_transaction`` persists a CONFIRMED row;
    ``find_confirmed_transaction`` returns the row matching
    ``(reference_id, transaction_type)``. This exercises the REAL distributor
    idempotency logic against a faithful durable stand-in (the SQL itself is
    covered separately below against a real in-memory SQLite).
    """

    def __init__(self):
        self.rows: list[SimpleNamespace] = []
        self.create_calls: list[SimpleNamespace] = []
        self.stake_for_governance = AsyncMock(return_value=(None, Decimal("0")))

    async def create_transaction(self, *, from_user_id, to_user_id, amount,
                                 transaction_type, description,
                                 reference_id=None, **kw):
        call = SimpleNamespace(reference_id=reference_id,
                               transaction_type=transaction_type,
                               amount=amount, to_user_id=to_user_id)
        self.create_calls.append(call)
        row = SimpleNamespace(reference_id=reference_id,
                              transaction_type=transaction_type,
                              amount=amount, status="confirmed",
                              to_user_id=to_user_id)
        self.rows.append(row)
        return row

    async def find_confirmed_transaction(self, reference_id, transaction_type):
        tt = getattr(transaction_type, "value", transaction_type)
        for r in self.rows:
            if (r.reference_id == reference_id and r.transaction_type == tt
                    and r.status == "confirmed"):
                return r
        return None


@pytest.fixture
def distributor(monkeypatch):
    fake = _FakeDurableFtns()
    monkeypatch.setattr(td, "DatabaseFTNSService", lambda *a, **k: fake)
    monkeypatch.setattr(td.auth_manager, "get_user_by_id",
                        AsyncMock(return_value={"id": "u1", "username": "u1"}),
                        raising=False)
    monkeypatch.setattr(td.audit_logger, "log_security_event", AsyncMock(),
                        raising=False)
    d = td.GovernanceTokenDistributor()
    d._check_tier_upgrade = AsyncMock()
    d._stake_tokens_for_voting = AsyncMock(return_value=Decimal("0"))
    d._fake = fake
    return d


# ── contribution-reward idempotency ─────────────────────────────────────


@pytest.mark.asyncio
async def test_contribution_reward_mints_once_on_retry(distributor):
    args = dict(user_id="u1", contribution_type=ContributionType.TESTING,
                contribution_reference="pr-42")
    first = await distributor.distribute_contribution_rewards(**args)
    second = await distributor.distribute_contribution_rewards(**args)  # retry

    assert len(distributor._fake.create_calls) == 1     # minted exactly ONCE
    assert second.metadata.get("idempotent_replay") is True
    assert second.amount == first.amount


@pytest.mark.asyncio
async def test_distinct_contributions_each_mint(distributor):
    await distributor.distribute_contribution_rewards(
        user_id="u1", contribution_type=ContributionType.TESTING,
        contribution_reference="pr-1")
    await distributor.distribute_contribution_rewards(
        user_id="u1", contribution_type=ContributionType.TESTING,
        contribution_reference="pr-2")
    assert len(distributor._fake.create_calls) == 2     # different ref → 2 mints


@pytest.mark.asyncio
async def test_contribution_mint_uses_deterministic_reference_id(distributor):
    await distributor.distribute_contribution_rewards(
        user_id="u7", contribution_type=ContributionType.DOCUMENTATION,
        contribution_reference="doc-9")
    assert (distributor._fake.create_calls[0].reference_id
            == "gov-contrib:u7:documentation:doc-9")


# ── activation idempotency (restart safety) ──────────────────────────────


@pytest.mark.asyncio
async def test_activation_no_remint_after_restart(distributor):
    await distributor.activate_governance_participation(
        "u1", GovernanceParticipantTier.COMMUNITY)
    # Simulate a daemon RESTART: the in-memory guard is gone; the DB persists.
    distributor.governance_activations.clear()
    distributor.participant_tiers.clear()
    again = await distributor.activate_governance_participation(
        "u1", GovernanceParticipantTier.COMMUNITY)

    assert len(distributor._fake.create_calls) == 1     # NOT re-minted
    assert again.initial_token_allocation == Decimal("1000")


@pytest.mark.asyncio
async def test_distinct_users_same_tier_each_mint(distributor):
    await distributor.activate_governance_participation(
        "u1", GovernanceParticipantTier.COMMUNITY)
    await distributor.activate_governance_participation(
        "u2", GovernanceParticipantTier.COMMUNITY)
    assert len(distributor._fake.create_calls) == 2     # different user → 2 mints


@pytest.mark.asyncio
async def test_activation_mint_uses_deterministic_reference_id(distributor):
    await distributor.activate_governance_participation(
        "ua", GovernanceParticipantTier.COMMUNITY)
    assert distributor._fake.create_calls[0].reference_id == "gov-activate:ua"


# ── find_confirmed_transaction against a real in-memory SQLite ───────────


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as s:
        yield s
    await engine.dispose()


def _svc_with(session):
    svc = DatabaseFTNSService()
    svc._session = session
    return svc


async def _insert_tx(session, *, reference_id, transaction_type, status):
    session.add(FTNSTransaction(
        to_wallet_id=uuid4(), amount=Decimal("1000"),
        transaction_type=transaction_type, status=status,
        description="x", reference_id=reference_id,
    ))
    await session.commit()


@pytest.mark.asyncio
async def test_find_confirmed_matches_ref_and_type(db_session):
    await _insert_tx(db_session, reference_id="gov-activate:u1",
                     transaction_type="governance_activation",
                     status=TransactionStatus.CONFIRMED.value)
    found = await _svc_with(db_session).find_confirmed_transaction(
        "gov-activate:u1", "governance_activation")
    assert found is not None
    assert found.reference_id == "gov-activate:u1"


@pytest.mark.asyncio
async def test_find_confirmed_ignores_pending(db_session):
    await _insert_tx(db_session, reference_id="gov-activate:u2",
                     transaction_type="governance_activation",
                     status=TransactionStatus.PENDING.value)
    assert await _svc_with(db_session).find_confirmed_transaction(
        "gov-activate:u2", "governance_activation") is None


@pytest.mark.asyncio
async def test_find_confirmed_ignores_other_type(db_session):
    await _insert_tx(db_session, reference_id="k", transaction_type="reward",
                     status=TransactionStatus.CONFIRMED.value)
    assert await _svc_with(db_session).find_confirmed_transaction(
        "k", "governance_activation") is None


@pytest.mark.asyncio
async def test_find_confirmed_absent_returns_none(db_session):
    assert await _svc_with(db_session).find_confirmed_transaction(
        "nope", "governance_activation") is None


@pytest.mark.asyncio
async def test_find_confirmed_empty_reference_returns_none(db_session):
    assert await _svc_with(db_session).find_confirmed_transaction(
        "", "governance_activation") is None
