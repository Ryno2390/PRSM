# PRSM Remediation & Hardening Master Plan
## Series A Technical Due Diligence Response
### Target: 4.5/10 → 8.5/10 Readiness Score in 12 Weeks

---

## Executive Summary

This document details a 12-week high-intensity implementation plan to address critical findings from our Series A technical due diligence audit. The plan is organized into four 3-week sprints, each with clear deliverables, success criteria, and "Definition of Done" metrics that map directly to audit requirements.

**Current State Assessment:**
- Security Score: 4.5/10 (Critical vulnerabilities in financial operations)
- Production Readiness: Low (In-memory state, stubbed cryptography, monolithic architecture)
- Scalability: Limited (2,204-line main.py, no Redis-backed rate limiting in core paths)

**Target State:**
- Security Score: 8.5/10 (Production-hardened financial and auth systems)
- Production Readiness: High (Persistent state, real cryptography, modular architecture)
- Scalability: Enterprise-ready (Microservice-ready routers, distributed rate limiting)

---

## Phase 0: "Stop the Bleed" (Week 1)
### Emergency Security Patches - IMMEDIATE ACTION REQUIRED

### 0.1 Double-Spend Vulnerability Fix

**Location:** `prsm/economy/tokenomics/ftns_service.py`

**Current Vulnerable Code (lines 204-249):**
```python
def deduct_tokens(self, user_id: str, amount: Decimal, ...) -> bool:
    current_balance = self.get_user_balance(user_id)  # RACE CONDITION WINDOW
    if current_balance < amount:
        return False
    # Between check and update, another request can deduct same tokens
    new_balance = current_balance - amount
    balance_record.balance = new_balance  # NOT ATOMIC
```

**Root Cause:** Time-of-Check to Time-of-Use (TOCTOU) vulnerability. The balance check and deduction are not atomic, allowing concurrent requests to pass the balance check before either completes the deduction.

**Remediation - Atomic Database Operations with Row Locking:**

```python
# FILE: prsm/economy/tokenomics/ftns_service.py
# REPLACE deduct_tokens method with atomic implementation

from sqlalchemy import select, update
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from prsm.core.database_service import get_database_service

async def deduct_tokens_atomic(
    self,
    session: AsyncSession,
    user_id: str,
    amount: Decimal,
    idempotency_key: str,  # NEW: Prevent duplicate operations
    description: str = "",
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Atomically deduct FTNS tokens with pessimistic locking.

    Uses SELECT FOR UPDATE to acquire row-level lock, preventing
    concurrent modifications during the transaction.

    Returns:
        Tuple of (success: bool, transaction_id: Optional[str])
    """
    try:
        # Step 1: Check idempotency to prevent duplicate deductions
        existing_tx = await session.execute(
            select(FTNSTransactionModel)
            .where(FTNSTransactionModel.idempotency_key == idempotency_key)
        )
        if existing_tx.scalar_one_or_none():
            logger.warning("Duplicate deduction attempt blocked",
                          idempotency_key=idempotency_key)
            return False, None

        # Step 2: Acquire exclusive lock on user's balance row
        # with_for_update() generates SELECT ... FOR UPDATE
        balance_query = (
            select(FTNSBalanceModel)
            .where(FTNSBalanceModel.user_id == user_id)
            .with_for_update(nowait=False)  # Block until lock acquired
        )

        result = await session.execute(balance_query)
        balance_record = result.scalar_one_or_none()

        if not balance_record:
            logger.warning("User balance record not found", user_id=user_id)
            return False, None

        # Step 3: Validate balance (now with exclusive lock held)
        available_balance = balance_record.balance - balance_record.locked_balance
        if available_balance < amount:
            logger.warning("Insufficient balance for deduction",
                          user_id=user_id,
                          requested=float(amount),
                          available=float(available_balance))
            return False, None

        # Step 4: Perform atomic update with optimistic concurrency control
        new_balance = balance_record.balance - amount
        new_version = balance_record.version + 1

        update_stmt = (
            update(FTNSBalanceModel)
            .where(
                FTNSBalanceModel.user_id == user_id,
                FTNSBalanceModel.version == balance_record.version  # OCC check
            )
            .values(
                balance=new_balance,
                total_spent=balance_record.total_spent + amount,
                version=new_version,
                updated_at=datetime.now(timezone.utc)
            )
        )

        update_result = await session.execute(update_stmt)

        if update_result.rowcount == 0:
            # Concurrent modification detected - fail safely
            logger.error("Concurrent modification detected during deduction",
                        user_id=user_id)
            raise OptimisticLockException("Balance was modified by another transaction")

        # Step 5: Create transaction record with idempotency key
        transaction_id = f"ftns_{uuid4().hex[:12]}"
        tx_record = FTNSTransactionModel(
            id=transaction_id,
            user_id=user_id,
            amount=-amount,
            transaction_type="deduction",
            description=description,
            idempotency_key=idempotency_key,
            balance_after=new_balance,
            metadata=metadata or {},
            created_at=datetime.now(timezone.utc)
        )
        session.add(tx_record)

        # Commit happens at context manager exit
        logger.info("FTNS tokens deducted atomically",
                   user_id=user_id,
                   amount=float(amount),
                   new_balance=float(new_balance),
                   transaction_id=transaction_id)

        return True, transaction_id

    except Exception as e:
        logger.error(f"Atomic deduction failed: {e}", user_id=user_id)
        raise  # Let caller handle rollback
```

**Database Migration Required:**

```sql
-- FILE: scripts/migrations/003_ftns_atomic_operations.sql

-- Add version column for optimistic concurrency control
ALTER TABLE ftns_balances
ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1 NOT NULL;

-- Add idempotency tracking
CREATE TABLE IF NOT EXISTS ftns_idempotency_keys (
    idempotency_key VARCHAR(255) PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(64) NOT NULL,
    operation_type VARCHAR(32) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '24 hours')
);

-- Index for fast lookups and cleanup
CREATE INDEX IF NOT EXISTS idx_idempotency_expires
ON ftns_idempotency_keys(expires_at);

CREATE INDEX IF NOT EXISTS idx_idempotency_user
ON ftns_idempotency_keys(user_id, created_at);

-- Add row-level locking advisory lock function
CREATE OR REPLACE FUNCTION acquire_balance_lock(p_user_id VARCHAR)
RETURNS BOOLEAN AS $$
BEGIN
    PERFORM pg_advisory_xact_lock(hashtext(p_user_id));
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Cleanup job for expired idempotency keys
CREATE OR REPLACE FUNCTION cleanup_expired_idempotency_keys()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM ftns_idempotency_keys
    WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;
```

---

### 0.2 JWT Verification Bypass Fix

**Location:** `prsm/core/auth/jwt_handler.py`

**Current Vulnerable Code (lines 388-396):**
```python
async def _is_token_revoked(self, token_hash: str) -> bool:
    """Check if token is revoked"""
    try:
        # Would check database for revoked status
        # For now, assume not revoked   <-- VULNERABILITY: Always returns False
        return False
    except Exception as e:
        return True  # Fail safe
```

**Additional Issue (line 221):**
The JWT decode allows algorithm confusion if not properly configured. The current implementation passes a list which is correct, but the `secret_key` is used for both HS256 and RS256 scenarios without proper key type validation.

**Remediation - Proper Token Revocation and Algorithm Enforcement:**

```python
# FILE: prsm/core/auth/jwt_handler.py
# REPLACE _is_token_revoked and enhance verify_token

from prsm.core.redis_client import get_redis_client
from prsm.core.database_service import get_database_service

class JWTHandler:
    """Enhanced JWT handler with real revocation checking"""

    def __init__(self):
        self.algorithm = settings.jwt_algorithm
        self.secret_key = settings.secret_key

        # SECURITY FIX: Enforce single algorithm to prevent algorithm confusion
        if self.algorithm not in ["HS256", "HS384", "HS512"]:
            raise ValueError(f"Unsupported JWT algorithm: {self.algorithm}")

        # Validate secret key strength
        if len(self.secret_key) < 32:
            raise ValueError("JWT secret key must be at least 32 characters")

        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7

        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto",
            bcrypt__rounds=12
        )

        self.db_service = None
        self.redis_client = None

        # Token revocation cache TTL
        self._revocation_cache_ttl = 300  # 5 minutes

    async def initialize(self):
        """Initialize database and cache connections"""
        try:
            self.db_service = get_database_service()
            self.redis_client = get_redis_client()
            logger.info("JWT handler initialized with revocation support")
        except Exception as e:
            logger.error("Failed to initialize JWT handler", error=str(e))
            raise

    async def verify_token(self, token: str) -> Optional[TokenData]:
        """
        Verify and decode JWT token with proper revocation checking.

        Security measures:
        1. Algorithm enforcement (no algorithm confusion)
        2. Signature verification
        3. Expiration checking
        4. Real revocation checking (Redis + Database)
        5. Token binding validation
        """
        try:
            # SECURITY: Explicitly specify allowed algorithm (singular)
            # This prevents algorithm confusion attacks
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],  # Explicit single algorithm
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "require": ["exp", "iat", "sub", "jti"]  # Required claims
                }
            )

            user_id = UUID(payload.get("sub"))
            jti = payload.get("jti")  # Token ID for revocation checking

            if not jti:
                logger.warning("Token missing JTI claim", user_id=str(user_id))
                return None

            # SECURITY FIX: Real revocation checking
            token_hash = self.generate_token_hash(token)
            if await self._is_token_revoked(token_hash, jti):
                logger.warning("Token revoked", user_id=str(user_id), jti=jti)
                return None

            # Extract and validate remaining claims
            username = payload.get("username")
            email = payload.get("email", "")
            role = payload.get("role", "")
            permissions = payload.get("permissions", [])
            token_type = payload.get("token_type", "access")

            issued_at = datetime.fromtimestamp(payload.get("iat"), tz=timezone.utc)
            expires_at = datetime.fromtimestamp(payload.get("exp"), tz=timezone.utc)

            # Additional expiration check (belt and suspenders)
            if expires_at <= datetime.now(timezone.utc):
                logger.warning("Token expired", user_id=str(user_id))
                return None

            # Update last used timestamp (async, fire-and-forget)
            asyncio.create_task(self._update_token_last_used(token_hash))

            return TokenData(
                user_id=user_id,
                username=username,
                email=email,
                role=role,
                permissions=permissions,
                token_type=token_type,
                issued_at=issued_at,
                expires_at=expires_at
            )

        except jwt.ExpiredSignatureError:
            logger.warning("Token signature expired")
            return None
        except jwt.InvalidAlgorithmError:
            logger.error("Invalid token algorithm - possible attack")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid token", error=str(e))
            return None
        except Exception as e:
            logger.error("Token verification error", error=str(e))
            return None

    async def _is_token_revoked(self, token_hash: str, jti: str) -> bool:
        """
        Check if token is revoked using tiered caching strategy.

        Lookup order:
        1. Redis cache (fast path)
        2. Database (authoritative source)
        3. Cache the result for subsequent lookups
        """
        try:
            # Tier 1: Check Redis cache (sub-millisecond)
            if self.redis_client:
                cache_key = f"token_revoked:{jti}"
                cached_status = await self.redis_client.get(cache_key)

                if cached_status is not None:
                    return cached_status == b"1" or cached_status == "1"

            # Tier 2: Check database (authoritative)
            if self.db_service:
                async with self.db_service.get_session() as session:
                    from sqlalchemy import select, text

                    # Check revoked_tokens table
                    query = text("""
                        SELECT 1 FROM revoked_tokens
                        WHERE token_hash = :token_hash OR jti = :jti
                        LIMIT 1
                    """)

                    result = await session.execute(
                        query,
                        {"token_hash": token_hash, "jti": jti}
                    )
                    is_revoked = result.scalar() is not None

                    # Cache the result in Redis
                    if self.redis_client:
                        await self.redis_client.setex(
                            cache_key,
                            self._revocation_cache_ttl,
                            "1" if is_revoked else "0"
                        )

                    return is_revoked

            # If no database available, fail open in dev, closed in prod
            if settings.is_production:
                logger.error("Cannot verify token revocation - failing closed")
                return True

            return False

        except Exception as e:
            logger.error("Token revocation check failed", error=str(e))
            # Fail closed in production
            return settings.is_production

    async def _revoke_token(self, token_hash: str, jti: str, reason: str = "") -> bool:
        """
        Revoke token by adding to revocation list.

        Revocation is stored in:
        1. Database (persistent, authoritative)
        2. Redis (cache for fast lookups)
        """
        try:
            if self.db_service:
                async with self.db_service.get_session() as session:
                    from sqlalchemy import text

                    insert_query = text("""
                        INSERT INTO revoked_tokens (token_hash, jti, reason, revoked_at)
                        VALUES (:token_hash, :jti, :reason, NOW())
                        ON CONFLICT (token_hash) DO NOTHING
                    """)

                    await session.execute(insert_query, {
                        "token_hash": token_hash,
                        "jti": jti,
                        "reason": reason
                    })
                    await session.commit()

            # Immediately update Redis cache
            if self.redis_client:
                cache_key = f"token_revoked:{jti}"
                await self.redis_client.setex(
                    cache_key,
                    86400,  # 24 hour TTL for revoked tokens
                    "1"
                )

            logger.info("Token revoked", jti=jti, reason=reason)
            return True

        except Exception as e:
            logger.error("Failed to revoke token", error=str(e))
            return False
```

**Database Migration for Token Revocation:**

```sql
-- FILE: scripts/migrations/004_token_revocation.sql

CREATE TABLE IF NOT EXISTS revoked_tokens (
    id SERIAL PRIMARY KEY,
    token_hash VARCHAR(64) UNIQUE NOT NULL,
    jti VARCHAR(64) NOT NULL,
    user_id UUID,
    reason VARCHAR(255),
    revoked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE  -- For cleanup
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_revoked_tokens_hash ON revoked_tokens(token_hash);
CREATE INDEX IF NOT EXISTS idx_revoked_tokens_jti ON revoked_tokens(jti);
CREATE INDEX IF NOT EXISTS idx_revoked_tokens_user ON revoked_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_revoked_tokens_expires ON revoked_tokens(expires_at);

-- Cleanup function for expired revocation records
CREATE OR REPLACE FUNCTION cleanup_expired_revoked_tokens()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM revoked_tokens
    WHERE expires_at IS NOT NULL AND expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;
```

---

## Sprint 1: Database & Financial Integrity (Weeks 1-3)
### "Trust, But Verify" - Persistent, Atomic FTNS Operations

### 1.1 Migrate In-Memory FTNS to PostgreSQL

**Objective:** Replace `ftns_service.py` in-memory implementation with the existing `production_ledger.py` as the default.

**Current State:**
- `ftns_service.py`: In-memory dictionaries, lost on restart
- `production_ledger.py`: PostgreSQL-backed, exists but not integrated

**Action Items:**

1. **Create unified FTNS service interface:**

```python
# FILE: prsm/economy/tokenomics/__init__.py

from typing import Protocol, runtime_checkable
from decimal import Decimal
from typing import Dict, Any, Optional

@runtime_checkable
class FTNSServiceProtocol(Protocol):
    """Protocol defining FTNS service contract"""

    async def get_balance(self, user_id: str) -> Decimal: ...
    async def transfer(
        self,
        from_user: str,
        to_user: str,
        amount: Decimal,
        idempotency_key: str,
        description: str
    ) -> str: ...
    async def mint(self, user_id: str, amount: Decimal, reason: str) -> str: ...
    async def burn(self, user_id: str, amount: Decimal, reason: str) -> str: ...
    async def get_transaction_history(self, user_id: str, limit: int) -> list: ...


# Factory function for service selection
def get_ftns_service() -> FTNSServiceProtocol:
    """
    Get the appropriate FTNS service based on environment.

    Production: PostgreSQL-backed ProductionFTNSLedger
    Development: In-memory FTNSService (with warning)
    Testing: Mock service
    """
    from prsm.core.config import get_settings
    settings = get_settings()

    if settings.is_production or settings.environment.value == "staging":
        from prsm.economy.tokenomics.production_ledger import get_production_ledger
        return get_production_ledger()
    elif settings.environment.value == "testing":
        from prsm.economy.tokenomics.mock_ftns_service import MockFTNSService
        return MockFTNSService()
    else:
        # Development: Use production ledger but with warnings
        import warnings
        warnings.warn(
            "Using production FTNS ledger in development. "
            "Ensure PostgreSQL is available.",
            RuntimeWarning
        )
        from prsm.economy.tokenomics.production_ledger import get_production_ledger
        return get_production_ledger()
```

2. **Enhanced Production Ledger Schema:**

```sql
-- FILE: scripts/migrations/005_enhanced_ftns_schema.sql

-- Enhanced balances table with audit trail
CREATE TABLE IF NOT EXISTS ftns_balances (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(64) UNIQUE NOT NULL,
    balance DECIMAL(28, 8) NOT NULL DEFAULT 0 CHECK (balance >= 0),
    locked_balance DECIMAL(28, 8) NOT NULL DEFAULT 0 CHECK (locked_balance >= 0),
    total_earned DECIMAL(28, 8) NOT NULL DEFAULT 0,
    total_spent DECIMAL(28, 8) NOT NULL DEFAULT 0,
    total_burned DECIMAL(28, 8) NOT NULL DEFAULT 0,
    account_type VARCHAR(32) NOT NULL DEFAULT 'user',
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT valid_available_balance CHECK (balance >= locked_balance)
);

-- Enhanced transactions table
CREATE TABLE IF NOT EXISTS ftns_transactions (
    id VARCHAR(64) PRIMARY KEY,
    from_user_id VARCHAR(64) NOT NULL,
    to_user_id VARCHAR(64),
    amount DECIMAL(28, 8) NOT NULL,
    transaction_type VARCHAR(32) NOT NULL,
    description TEXT,
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    error_message TEXT,
    idempotency_key VARCHAR(255) UNIQUE,
    reference_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    balance_before_sender DECIMAL(28, 8),
    balance_after_sender DECIMAL(28, 8),
    balance_before_receiver DECIMAL(28, 8),
    balance_after_receiver DECIMAL(28, 8),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT valid_amount CHECK (amount > 0)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_ftns_balances_user ON ftns_balances(user_id);
CREATE INDEX IF NOT EXISTS idx_ftns_tx_from ON ftns_transactions(from_user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_ftns_tx_to ON ftns_transactions(to_user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_ftns_tx_status ON ftns_transactions(status);
CREATE INDEX IF NOT EXISTS idx_ftns_tx_type ON ftns_transactions(transaction_type);
CREATE INDEX IF NOT EXISTS idx_ftns_tx_reference ON ftns_transactions(reference_id);
CREATE INDEX IF NOT EXISTS idx_ftns_tx_idempotency ON ftns_transactions(idempotency_key);

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_ftns_balances_updated_at
    BEFORE UPDATE ON ftns_balances
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ftns_transactions_updated_at
    BEFORE UPDATE ON ftns_transactions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

### 1.2 Implement Transaction Rollback and Recovery

```python
# FILE: prsm/economy/tokenomics/transaction_recovery.py

"""
FTNS Transaction Recovery System

Handles failed transaction recovery, orphaned lock cleanup,
and ledger reconciliation.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from decimal import Decimal
import structlog

from sqlalchemy import text, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from prsm.core.database_service import get_database_service

logger = structlog.get_logger(__name__)


class TransactionRecoveryService:
    """
    Automated recovery service for FTNS transactions.

    Responsibilities:
    1. Detect and recover stuck transactions
    2. Clean up orphaned locks
    3. Reconcile balance discrepancies
    4. Generate audit reports
    """

    def __init__(self):
        self.db_service = get_database_service()
        self.stuck_threshold = timedelta(minutes=5)
        self.lock_timeout = timedelta(minutes=10)

    async def recover_stuck_transactions(self) -> Dict[str, Any]:
        """
        Find and recover transactions stuck in 'pending' state.

        Recovery strategy:
        1. Identify transactions pending > threshold
        2. Check if balances were actually modified
        3. Either complete or rollback based on state
        """
        recovered = []
        failed = []

        async with self.db_service.get_session() as session:
            # Find stuck transactions
            stuck_query = text("""
                SELECT id, from_user_id, to_user_id, amount, transaction_type,
                       balance_before_sender, balance_after_sender, created_at
                FROM ftns_transactions
                WHERE status = 'pending'
                AND created_at < NOW() - INTERVAL ':threshold_minutes minutes'
                FOR UPDATE SKIP LOCKED
            """)

            result = await session.execute(
                stuck_query,
                {"threshold_minutes": int(self.stuck_threshold.total_seconds() / 60)}
            )

            for row in result.fetchall():
                try:
                    recovery_result = await self._recover_single_transaction(
                        session, row
                    )
                    recovered.append({
                        "transaction_id": row.id,
                        "action": recovery_result
                    })
                except Exception as e:
                    failed.append({
                        "transaction_id": row.id,
                        "error": str(e)
                    })

            await session.commit()

        return {
            "recovered": len(recovered),
            "failed": len(failed),
            "details": {
                "recovered_transactions": recovered,
                "failed_transactions": failed
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def _recover_single_transaction(
        self,
        session: AsyncSession,
        tx_row
    ) -> str:
        """Attempt recovery of a single stuck transaction"""

        # Check current balances
        sender_balance_query = text("""
            SELECT balance FROM ftns_balances WHERE user_id = :user_id
        """)

        sender_result = await session.execute(
            sender_balance_query,
            {"user_id": tx_row.from_user_id}
        )
        current_sender_balance = sender_result.scalar()

        # Determine if deduction happened
        if tx_row.balance_after_sender is not None:
            # Transaction was partially completed
            if current_sender_balance == tx_row.balance_after_sender:
                # Deduction completed, mark as complete
                await session.execute(
                    text("UPDATE ftns_transactions SET status = 'completed' WHERE id = :id"),
                    {"id": tx_row.id}
                )
                return "completed"
            else:
                # Inconsistent state - rollback
                await self._rollback_transaction(session, tx_row)
                return "rolled_back"
        else:
            # Transaction never started - mark as failed
            await session.execute(
                text("""
                    UPDATE ftns_transactions
                    SET status = 'failed', error_message = 'Timeout during processing'
                    WHERE id = :id
                """),
                {"id": tx_row.id}
            )
            return "marked_failed"

    async def _rollback_transaction(self, session: AsyncSession, tx_row) -> None:
        """Rollback a partially completed transaction"""
        # Restore sender balance
        await session.execute(
            text("""
                UPDATE ftns_balances
                SET balance = balance + :amount,
                    total_spent = total_spent - :amount,
                    version = version + 1
                WHERE user_id = :user_id
            """),
            {"amount": tx_row.amount, "user_id": tx_row.from_user_id}
        )

        # If transfer, reverse receiver credit
        if tx_row.to_user_id:
            await session.execute(
                text("""
                    UPDATE ftns_balances
                    SET balance = balance - :amount,
                        total_earned = total_earned - :amount,
                        version = version + 1
                    WHERE user_id = :user_id
                """),
                {"amount": tx_row.amount, "user_id": tx_row.to_user_id}
            )

        # Mark transaction as rolled back
        await session.execute(
            text("""
                UPDATE ftns_transactions
                SET status = 'rolled_back',
                    error_message = 'Automatic rollback due to timeout'
                WHERE id = :id
            """),
            {"id": tx_row.id}
        )

        logger.warning("Transaction rolled back", transaction_id=tx_row.id)

    async def reconcile_ledger(self) -> Dict[str, Any]:
        """
        Full ledger reconciliation.

        Verifies that:
        1. Sum of all balances equals (minted - burned)
        2. Each user's balance = earned - spent
        3. No negative balances exist
        """
        async with self.db_service.get_session() as session:
            # Calculate expected total from transactions
            totals_query = text("""
                SELECT
                    SUM(CASE WHEN transaction_type = 'mint' THEN amount ELSE 0 END) as total_minted,
                    SUM(CASE WHEN transaction_type = 'burn' THEN amount ELSE 0 END) as total_burned
                FROM ftns_transactions
                WHERE status = 'completed'
            """)

            totals_result = await session.execute(totals_query)
            totals = totals_result.fetchone()
            expected_supply = (totals.total_minted or 0) - (totals.total_burned or 0)

            # Calculate actual total from balances
            actual_query = text("SELECT SUM(balance) as total FROM ftns_balances")
            actual_result = await session.execute(actual_query)
            actual_supply = actual_result.scalar() or 0

            # Find individual discrepancies
            discrepancy_query = text("""
                SELECT
                    b.user_id,
                    b.balance as recorded_balance,
                    b.total_earned - b.total_spent as calculated_balance,
                    b.balance - (b.total_earned - b.total_spent) as discrepancy
                FROM ftns_balances b
                WHERE ABS(b.balance - (b.total_earned - b.total_spent)) > 0.00000001
            """)

            discrepancy_result = await session.execute(discrepancy_query)
            discrepancies = discrepancy_result.fetchall()

            # Check for negative balances
            negative_query = text("""
                SELECT user_id, balance
                FROM ftns_balances
                WHERE balance < 0
            """)

            negative_result = await session.execute(negative_query)
            negative_balances = negative_result.fetchall()

            return {
                "supply_reconciliation": {
                    "expected_supply": float(expected_supply),
                    "actual_supply": float(actual_supply),
                    "discrepancy": float(actual_supply - expected_supply),
                    "is_balanced": abs(actual_supply - expected_supply) < 0.00000001
                },
                "user_discrepancies": len(discrepancies),
                "negative_balances": len(negative_balances),
                "is_healthy": (
                    abs(actual_supply - expected_supply) < 0.00000001 and
                    len(discrepancies) == 0 and
                    len(negative_balances) == 0
                ),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }


# Background job for periodic recovery
async def start_recovery_background_job(interval_seconds: int = 300):
    """Start background job for transaction recovery"""
    recovery_service = TransactionRecoveryService()

    while True:
        try:
            result = await recovery_service.recover_stuck_transactions()
            if result["recovered"] > 0 or result["failed"] > 0:
                logger.info("Transaction recovery completed", **result)
        except Exception as e:
            logger.error("Transaction recovery failed", error=str(e))

        await asyncio.sleep(interval_seconds)
```

### Sprint 1 Definition of Done

| Criteria | Metric | Verification Method |
|----------|--------|---------------------|
| Double-spend prevented | 0 successful double-spend attacks | Concurrent load test with 1000 simultaneous deductions |
| Token revocation functional | 100% revoked tokens rejected | Automated test suite with revocation scenarios |
| FTNS persistence | 0 data loss on restart | Kill-restart test with balance verification |
| Idempotency working | Duplicate requests return same result | Replay attack test suite |
| Recovery automated | <5 min stuck transaction resolution | Chaos engineering test (kill mid-transaction) |

---

## Sprint 2: Post-Quantum Cryptography Resolution (Weeks 4-6)
### "Real or Deprecated" - No Marketing Misalignment

### 2.1 Current State Analysis

**Location:** `prsm/core/cryptography/post_quantum.py`

**Problem:** The PQC implementation has a `mock_mode` that silently falls back to fake signatures when the real `dilithium-py` library isn't available. This creates a false sense of security.

```python
# Current vulnerable pattern (lines 141-145)
if not DILITHIUM_AVAILABLE:
    logger.warning("dilithium-py not available. Using MOCK PQC for development.")
    self.mock_mode = True
else:
    self.mock_mode = False
```

### 2.2 Option A: Full PQC Implementation with liboqs

**If continuing with PQC as a feature:**

```python
# FILE: prsm/core/cryptography/post_quantum_production.py

"""
Production Post-Quantum Cryptography Implementation
Uses liboqs (Open Quantum Safe) library for real PQC operations

REQUIRES: pip install liboqs-python
System dependency: liboqs C library
"""

import os
import hashlib
import structlog
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

logger = structlog.get_logger(__name__)

# Attempt to import liboqs
try:
    import oqs
    LIBOQS_AVAILABLE = True

    # Verify the library works
    _test_sig = oqs.Signature("Dilithium3")
    _test_sig.generate_keypair()
    LIBOQS_FUNCTIONAL = True
    del _test_sig

except ImportError:
    LIBOQS_AVAILABLE = False
    LIBOQS_FUNCTIONAL = False
    oqs = None
except Exception as e:
    LIBOQS_AVAILABLE = True
    LIBOQS_FUNCTIONAL = False
    logger.error(f"liboqs import succeeded but functionality test failed: {e}")


class PQCSecurityLevel(str, Enum):
    """NIST Post-Quantum Security Levels"""
    LEVEL_2 = "Dilithium2"   # NIST Level 2 (~AES-128)
    LEVEL_3 = "Dilithium3"   # NIST Level 3 (~AES-192)
    LEVEL_5 = "Dilithium5"   # NIST Level 5 (~AES-256)


class PQCMode(str, Enum):
    """Operational modes for PQC system"""
    REAL = "real"           # Full PQC with liboqs
    HYBRID = "hybrid"       # PQC + traditional (recommended)
    DISABLED = "disabled"   # No PQC (explicit opt-out)
    # NO MOCK MODE - removed intentionally


@dataclass
class PQCKeyPair:
    """Post-quantum key pair"""
    public_key: bytes
    private_key: bytes
    algorithm: str
    security_level: PQCSecurityLevel
    key_id: str = field(default_factory=lambda: hashlib.sha256(os.urandom(32)).hexdigest()[:16])
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PQCSignature:
    """Post-quantum signature"""
    signature: bytes
    algorithm: str
    key_id: str
    message_hash: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ProductionPostQuantumCrypto:
    """
    Production-grade Post-Quantum Cryptography implementation.

    IMPORTANT: This class will FAIL HARD if PQC is enabled but
    the library is not available. No silent fallbacks.
    """

    def __init__(
        self,
        mode: PQCMode = PQCMode.DISABLED,
        default_level: PQCSecurityLevel = PQCSecurityLevel.LEVEL_3
    ):
        """
        Initialize PQC system.

        Args:
            mode: Operational mode (REAL, HYBRID, or DISABLED)
            default_level: Default security level for operations

        Raises:
            RuntimeError: If mode requires liboqs but it's not available
        """
        self.mode = mode
        self.default_level = default_level

        if mode in [PQCMode.REAL, PQCMode.HYBRID]:
            if not LIBOQS_AVAILABLE:
                raise RuntimeError(
                    f"PQC mode '{mode.value}' requires liboqs library. "
                    "Install with: pip install liboqs-python\n"
                    "System dependency: liboqs (https://github.com/open-quantum-safe/liboqs)"
                )
            if not LIBOQS_FUNCTIONAL:
                raise RuntimeError(
                    "liboqs library is installed but not functional. "
                    "Check system liboqs installation."
                )

        self._signature_instances: Dict[str, oqs.Signature] = {}

        logger.info(f"PQC system initialized",
                   mode=mode.value,
                   liboqs_available=LIBOQS_AVAILABLE)

    def generate_keypair(
        self,
        security_level: Optional[PQCSecurityLevel] = None
    ) -> PQCKeyPair:
        """Generate a post-quantum key pair"""
        if self.mode == PQCMode.DISABLED:
            raise RuntimeError("PQC is disabled. Cannot generate keys.")

        level = security_level or self.default_level
        algorithm = level.value

        sig = self._get_signature_instance(algorithm)
        public_key = sig.generate_keypair()
        private_key = sig.export_secret_key()

        keypair = PQCKeyPair(
            public_key=public_key,
            private_key=private_key,
            algorithm=algorithm,
            security_level=level
        )

        logger.info("PQC keypair generated",
                   algorithm=algorithm,
                   key_id=keypair.key_id)

        return keypair

    def sign(
        self,
        message: Union[str, bytes],
        private_key: bytes,
        algorithm: str = "Dilithium3"
    ) -> PQCSignature:
        """Sign a message with post-quantum cryptography"""
        if self.mode == PQCMode.DISABLED:
            raise RuntimeError("PQC is disabled. Cannot sign.")

        if isinstance(message, str):
            message = message.encode('utf-8')

        message_hash = hashlib.sha256(message).hexdigest()

        sig_instance = oqs.Signature(algorithm, private_key)
        signature_bytes = sig_instance.sign(message)

        return PQCSignature(
            signature=signature_bytes,
            algorithm=algorithm,
            key_id="",  # Would be provided in real usage
            message_hash=message_hash
        )

    def verify(
        self,
        message: Union[str, bytes],
        signature: bytes,
        public_key: bytes,
        algorithm: str = "Dilithium3"
    ) -> bool:
        """Verify a post-quantum signature"""
        if self.mode == PQCMode.DISABLED:
            raise RuntimeError("PQC is disabled. Cannot verify.")

        if isinstance(message, str):
            message = message.encode('utf-8')

        try:
            sig_instance = oqs.Signature(algorithm)
            return sig_instance.verify(message, signature, public_key)
        except Exception as e:
            logger.error(f"PQC verification failed: {e}")
            return False

    def _get_signature_instance(self, algorithm: str) -> oqs.Signature:
        """Get or create a signature instance for the algorithm"""
        if algorithm not in self._signature_instances:
            self._signature_instances[algorithm] = oqs.Signature(algorithm)
        return self._signature_instances[algorithm]

    @staticmethod
    def get_available_algorithms() -> Dict[str, Any]:
        """Get information about available PQC algorithms"""
        if not LIBOQS_AVAILABLE:
            return {"error": "liboqs not available", "algorithms": []}

        return {
            "signature_algorithms": oqs.get_enabled_sig_mechanisms(),
            "recommended": ["Dilithium3", "Dilithium5"],
            "nist_finalists": ["Dilithium2", "Dilithium3", "Dilithium5"]
        }


# Factory function with explicit mode selection
def get_pqc_system(mode: Optional[PQCMode] = None) -> ProductionPostQuantumCrypto:
    """
    Get PQC system with explicit mode selection.

    Environment variable PRSM_PQC_MODE can be:
    - "real": Full PQC (requires liboqs)
    - "hybrid": PQC + traditional
    - "disabled": No PQC (default for safety)
    """
    if mode is None:
        mode_str = os.environ.get("PRSM_PQC_MODE", "disabled").lower()
        mode = PQCMode(mode_str)

    return ProductionPostQuantumCrypto(mode=mode)
```

### 2.3 Option B: Graceful Deprecation Path

**If removing PQC from production:**

```python
# FILE: prsm/core/cryptography/post_quantum.py (modified)

"""
Post-Quantum Cryptography Module - EXPERIMENTAL

WARNING: This module is marked EXPERIMENTAL and is NOT enabled in production.
PQC signatures are for research and future-proofing only.

For production cryptographic operations, use the standard
prsm.core.cryptography.signatures module.
"""

import warnings
from functools import wraps

def experimental_pqc(func):
    """Decorator to mark PQC functions as experimental"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"Function {func.__name__} uses experimental post-quantum cryptography. "
            "This is not enabled in production builds. "
            "See docs/EXPERIMENTAL_FEATURES.md for details.",
            category=ExperimentalWarning,
            stacklevel=2
        )
        return func(*args, **kwargs)
    return wrapper


class ExperimentalWarning(UserWarning):
    """Warning for experimental features"""
    pass


# Mark all PQC functions as experimental
@experimental_pqc
def generate_pq_keypair(...):
    ...

@experimental_pqc
def sign_with_pq(...):
    ...
```

**Documentation update:**

```markdown
<!-- FILE: docs/EXPERIMENTAL_FEATURES.md -->

# Experimental Features

## Post-Quantum Cryptography (PQC)

**Status:** EXPERIMENTAL - Not enabled in production

### Current State

PRSM includes a post-quantum cryptography module based on CRYSTALS-Dilithium
(ML-DSA) for future-proofing against quantum computing threats. However, this
module is currently **experimental** and **disabled in production** for the
following reasons:

1. **Library Dependencies:** Requires liboqs system library installation
2. **Performance Impact:** PQC signatures are 10-100x larger than traditional
3. **Standardization:** NIST finalization ongoing (FIPS 204)
4. **Interoperability:** Limited ecosystem support

### Roadmap

| Milestone | Target Date | Status |
|-----------|-------------|--------|
| liboqs integration complete | Q2 2025 | In Progress |
| Hybrid mode testing | Q3 2025 | Planned |
| Production pilot (opt-in) | Q4 2025 | Planned |
| Full production support | Q1 2026 | Planned |

### Enabling PQC (Development Only)

```bash
# Install system dependency
brew install liboqs  # macOS
apt install liboqs-dev  # Ubuntu

# Install Python bindings
pip install liboqs-python

# Enable in development
export PRSM_PQC_MODE=hybrid
```
```

### Sprint 2 Definition of Done

| Criteria | Metric | Verification Method |
|----------|--------|---------------------|
| No silent mock mode | Build fails if PQC enabled without library | CI/CD pipeline check |
| Clear documentation | 100% of PQC APIs documented with status | Documentation review |
| Marketing aligned | No PQC claims in production materials | Marketing audit |
| Future-ready | liboqs integration tested | Integration test suite |

---

## Sprint 3: API Scalability Cleanup (Weeks 7-9)
### "Divide and Conquer" - Modular Architecture

### 3.1 Break Down api/main.py (2,204 lines → ~200 lines)

**Target Architecture:**

```
prsm/interface/api/
├── main.py                 # ~200 lines - FastAPI app setup only
├── app_factory.py          # Application factory pattern
├── middleware/
│   ├── __init__.py
│   ├── auth.py             # Auth middleware (extracted)
│   ├── rate_limiting.py    # Rate limiting middleware
│   ├── security_headers.py # Security headers
│   └── request_validation.py
├── routers/
│   ├── __init__.py
│   ├── core/
│   │   ├── health.py       # Health check endpoints
│   │   ├── root.py         # Root/info endpoints
│   │   └── query.py        # Main query endpoint
│   ├── auth/
│   │   ├── login.py
│   │   ├── register.py
│   │   └── tokens.py
│   ├── ftns/
│   │   ├── balance.py
│   │   ├── transactions.py
│   │   └── governance.py
│   └── ... (existing routers organized)
├── websocket/
│   ├── __init__.py
│   ├── manager.py          # WebSocket manager (extracted)
│   └── handlers.py         # WebSocket message handlers
├── lifecycle/
│   ├── __init__.py
│   ├── startup.py          # Startup sequence
│   └── shutdown.py         # Shutdown sequence
└── dependencies/
    ├── __init__.py
    ├── auth.py             # Auth dependencies
    └── database.py         # Database dependencies
```

**Refactored main.py:**

```python
# FILE: prsm/interface/api/main.py (REFACTORED - ~200 lines)

"""
PRSM FastAPI Application
Main entry point - delegates to modular components
"""

from fastapi import FastAPI
from contextlib import asynccontextmanager
import structlog

from prsm.core.config import get_settings
from prsm.interface.api.app_factory import create_app
from prsm.interface.api.lifecycle import startup_sequence, shutdown_sequence
from prsm.interface.api.middleware import configure_middleware_stack
from prsm.interface.api.routers import include_all_routers

logger = structlog.get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting PRSM API server", environment=settings.environment)

    await startup_sequence(app)

    yield

    await shutdown_sequence(app)

    logger.info("PRSM API server shutdown complete")


# Create application
app = create_app(lifespan=lifespan)

# Configure middleware
configure_middleware_stack(app)

# Include routers
include_all_routers(app)

# Export for uvicorn
__all__ = ["app"]
```

**App Factory:**

```python
# FILE: prsm/interface/api/app_factory.py

"""
FastAPI Application Factory
Creates and configures the FastAPI application instance
"""

from typing import Callable, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from prsm.core.config import get_settings
from prsm.interface.api.openapi_config import custom_openapi_schema, API_TAGS_METADATA

settings = get_settings()


def create_app(
    lifespan: Optional[Callable] = None,
    title: str = "PRSM API",
    version: str = "0.1.0"
) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        lifespan: Async context manager for startup/shutdown
        title: API title
        version: API version

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=title,
        description="Protocol for Recursive Scientific Modeling - API for decentralized AI collaboration",
        version=version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        openapi_tags=API_TAGS_METADATA,
        contact={
            "name": "PRSM API Support",
            "email": "api-support@prsm.org",
            "url": "https://developers.prsm.org"
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT"
        },
        servers=_get_servers()
    )

    # Apply custom OpenAPI schema
    app.openapi = lambda: custom_openapi_schema(app)

    return app


def _get_servers():
    """Get server list based on environment"""
    servers = [
        {"url": "http://localhost:8000", "description": "Development server"}
    ]

    if not settings.is_development:
        servers = [
            {"url": "https://api.prsm.org", "description": "Production server"},
            {"url": "https://staging-api.prsm.org", "description": "Staging server"},
        ] + servers

    return servers
```

**Router Registry:**

```python
# FILE: prsm/interface/api/routers/__init__.py

"""
Router Registry
Centralized router management with lazy loading
"""

from typing import Dict, Any, List, Tuple
from fastapi import FastAPI, APIRouter
import structlog

logger = structlog.get_logger(__name__)


# Router registry: (module_path, prefix, tags)
ROUTER_REGISTRY: List[Tuple[str, str, List[str]]] = [
    # Core routers
    ("prsm.interface.api.routers.core.health", "/health", ["Health"]),
    ("prsm.interface.api.routers.core.query", "/query", ["Query"]),

    # Auth routers
    ("prsm.interface.api.auth_api", "/api/v1/auth", ["Authentication"]),

    # FTNS routers
    ("prsm.interface.api.payment_api", "/api/v1/ftns", ["FTNS"]),
    ("prsm.interface.api.budget_api", "/api/v1/budget", ["Budget"]),

    # Marketplace
    ("prsm.interface.api.real_marketplace_api", "/api/v1/marketplace", ["Marketplace"]),

    # Governance
    ("prsm.interface.api.governance_api", "/api/v1/governance", ["Governance"]),

    # Security
    ("prsm.interface.api.security_status_api", "/api/v1/security", ["Security"]),
    ("prsm.interface.api.security_logging_api", "/api/v1/security/logs", ["Security Logging"]),
    ("prsm.interface.api.compliance_api", "/api/v1/compliance", ["Compliance"]),

    # Infrastructure
    ("prsm.interface.api.ipfs_api", "/api/v1/ipfs", ["IPFS"]),
    ("prsm.interface.api.session_api", "/api/v1/sessions", ["Sessions"]),
    ("prsm.interface.api.task_api", "/api/v1/tasks", ["Tasks"]),

    # Advanced features
    ("prsm.interface.api.cryptography_api", "/api/v1/crypto", ["Cryptography"]),
    ("prsm.interface.api.distillation_api", "/api/v1/distillation", ["Distillation"]),
    ("prsm.interface.api.monitoring_api", "/api/v1/monitoring", ["Monitoring"]),

    # UI
    ("prsm.interface.api.ui_api", "/api/v1/ui", ["UI"]),
]


def include_all_routers(app: FastAPI) -> None:
    """
    Include all registered routers in the application.
    Uses lazy loading to improve startup time.
    """
    loaded = 0
    failed = []

    for module_path, prefix, tags in ROUTER_REGISTRY:
        try:
            router = _import_router(module_path)
            app.include_router(router, prefix=prefix, tags=tags)
            loaded += 1
        except Exception as e:
            failed.append((module_path, str(e)))
            logger.warning(f"Failed to load router: {module_path}", error=str(e))

    logger.info(f"Routers loaded: {loaded}/{len(ROUTER_REGISTRY)}",
               failed_count=len(failed))

    if failed:
        logger.warning("Failed routers", routers=failed)


def _import_router(module_path: str) -> APIRouter:
    """Dynamically import router from module path"""
    import importlib

    module = importlib.import_module(module_path)

    # Try common router attribute names
    for attr_name in ["router", "api_router", "Router"]:
        if hasattr(module, attr_name):
            return getattr(module, attr_name)

    raise AttributeError(f"No router found in module: {module_path}")
```

### 3.2 Migrate Rate Limiting to Redis

```python
# FILE: prsm/core/security/redis_rate_limiter.py

"""
Redis-backed Rate Limiter
Production-grade rate limiting with sliding window algorithm
"""

import time
import asyncio
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog

from prsm.core.redis_client import get_redis_client

logger = structlog.get_logger(__name__)


class RateLimitTier(str, Enum):
    """Rate limit tiers based on user type"""
    ANONYMOUS = "anonymous"
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit tier"""
    requests_per_minute: int
    requests_per_hour: int
    burst_allowance: int  # Extra requests allowed in burst
    cooldown_seconds: int  # Cooldown after limit hit


# Tier configurations
TIER_CONFIGS: Dict[RateLimitTier, RateLimitConfig] = {
    RateLimitTier.ANONYMOUS: RateLimitConfig(
        requests_per_minute=10,
        requests_per_hour=100,
        burst_allowance=5,
        cooldown_seconds=60
    ),
    RateLimitTier.FREE: RateLimitConfig(
        requests_per_minute=60,
        requests_per_hour=1000,
        burst_allowance=20,
        cooldown_seconds=30
    ),
    RateLimitTier.PRO: RateLimitConfig(
        requests_per_minute=300,
        requests_per_hour=10000,
        burst_allowance=50,
        cooldown_seconds=10
    ),
    RateLimitTier.ENTERPRISE: RateLimitConfig(
        requests_per_minute=1000,
        requests_per_hour=50000,
        burst_allowance=200,
        cooldown_seconds=5
    ),
    RateLimitTier.ADMIN: RateLimitConfig(
        requests_per_minute=10000,
        requests_per_hour=500000,
        burst_allowance=1000,
        cooldown_seconds=0
    ),
}


class RedisRateLimiter:
    """
    Production rate limiter using Redis sliding window.

    Features:
    - Sliding window rate limiting (more accurate than fixed window)
    - Tier-based limits
    - Burst allowance for traffic spikes
    - Cooldown periods after limit exceeded
    - IP and user-based limiting
    """

    def __init__(self, redis_client=None):
        self.redis = redis_client or get_redis_client()
        self._lua_script = None

    async def initialize(self):
        """Initialize Lua script for atomic operations"""
        # Sliding window rate limit script
        self._lua_script = self.redis.register_script("""
            local key = KEYS[1]
            local now = tonumber(ARGV[1])
            local window = tonumber(ARGV[2])
            local limit = tonumber(ARGV[3])

            -- Remove old entries outside the window
            redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)

            -- Count current requests in window
            local count = redis.call('ZCARD', key)

            if count < limit then
                -- Add new request
                redis.call('ZADD', key, now, now .. '-' .. math.random())
                redis.call('EXPIRE', key, window)
                return {1, count + 1, limit - count - 1}  -- allowed, current, remaining
            else
                return {0, count, 0}  -- denied, current, remaining
            end
        """)

        logger.info("Redis rate limiter initialized")

    async def check_rate_limit(
        self,
        identifier: str,
        tier: RateLimitTier = RateLimitTier.ANONYMOUS,
        endpoint: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed under rate limits.

        Args:
            identifier: User ID or IP address
            tier: Rate limit tier
            endpoint: Optional endpoint for endpoint-specific limits

        Returns:
            Tuple of (allowed: bool, metadata: dict)
        """
        config = TIER_CONFIGS[tier]

        # Check minute limit
        minute_key = f"ratelimit:minute:{identifier}"
        minute_result = await self._check_window(
            minute_key,
            60,
            config.requests_per_minute + config.burst_allowance
        )

        if not minute_result[0]:
            return False, {
                "reason": "minute_limit_exceeded",
                "limit": config.requests_per_minute,
                "current": minute_result[1],
                "retry_after": 60 - (time.time() % 60),
                "tier": tier.value
            }

        # Check hour limit
        hour_key = f"ratelimit:hour:{identifier}"
        hour_result = await self._check_window(
            hour_key,
            3600,
            config.requests_per_hour
        )

        if not hour_result[0]:
            return False, {
                "reason": "hour_limit_exceeded",
                "limit": config.requests_per_hour,
                "current": hour_result[1],
                "retry_after": 3600 - (time.time() % 3600),
                "tier": tier.value
            }

        # Check endpoint-specific limits if applicable
        if endpoint:
            endpoint_key = f"ratelimit:endpoint:{identifier}:{endpoint}"
            endpoint_result = await self._check_window(
                endpoint_key,
                60,
                self._get_endpoint_limit(endpoint, tier)
            )

            if not endpoint_result[0]:
                return False, {
                    "reason": "endpoint_limit_exceeded",
                    "endpoint": endpoint,
                    "current": endpoint_result[1],
                    "retry_after": 60,
                    "tier": tier.value
                }

        return True, {
            "minute_remaining": minute_result[2],
            "hour_remaining": hour_result[2],
            "tier": tier.value
        }

    async def _check_window(
        self,
        key: str,
        window_seconds: int,
        limit: int
    ) -> Tuple[bool, int, int]:
        """Check rate limit for a specific window"""
        now = time.time()

        if self._lua_script:
            result = await self._lua_script(
                keys=[key],
                args=[now, window_seconds, limit]
            )
            return bool(result[0]), result[1], result[2]
        else:
            # Fallback to non-atomic (less accurate but functional)
            return await self._check_window_fallback(key, window_seconds, limit, now)

    async def _check_window_fallback(
        self,
        key: str,
        window_seconds: int,
        limit: int,
        now: float
    ) -> Tuple[bool, int, int]:
        """Non-atomic fallback for rate limiting"""
        pipe = self.redis.pipeline()

        pipe.zremrangebyscore(key, '-inf', now - window_seconds)
        pipe.zcard(key)

        results = await pipe.execute()
        count = results[1]

        if count < limit:
            await self.redis.zadd(key, {f"{now}-{id(now)}": now})
            await self.redis.expire(key, window_seconds)
            return True, count + 1, limit - count - 1

        return False, count, 0

    def _get_endpoint_limit(self, endpoint: str, tier: RateLimitTier) -> int:
        """Get endpoint-specific rate limit"""
        # Expensive endpoints get lower limits
        expensive_endpoints = {
            "/query": 0.1,      # 10% of normal
            "/generate": 0.2,   # 20% of normal
            "/train": 0.05,     # 5% of normal
        }

        config = TIER_CONFIGS[tier]
        base_limit = config.requests_per_minute

        for pattern, multiplier in expensive_endpoints.items():
            if pattern in endpoint:
                return int(base_limit * multiplier)

        return base_limit

    async def get_usage_stats(self, identifier: str) -> Dict[str, Any]:
        """Get current rate limit usage for identifier"""
        minute_key = f"ratelimit:minute:{identifier}"
        hour_key = f"ratelimit:hour:{identifier}"

        now = time.time()

        pipe = self.redis.pipeline()
        pipe.zcount(minute_key, now - 60, now)
        pipe.zcount(hour_key, now - 3600, now)

        results = await pipe.execute()

        return {
            "requests_last_minute": results[0],
            "requests_last_hour": results[1],
            "identifier": identifier
        }


# Global rate limiter instance
_rate_limiter: Optional[RedisRateLimiter] = None


async def get_rate_limiter() -> RedisRateLimiter:
    """Get or create global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RedisRateLimiter()
        await _rate_limiter.initialize()
    return _rate_limiter
```

### Sprint 3 Definition of Done

| Criteria | Metric | Verification Method |
|----------|--------|---------------------|
| main.py reduced | <250 lines | Line count |
| Router modularity | Each router <500 lines | Code review |
| Rate limiting in Redis | 0 in-memory rate limit state | Code audit |
| Startup time maintained | <5 seconds | Performance test |
| All endpoints functional | 100% endpoint test coverage | Integration tests |

---

## Sprint 4: Hardening & Documentation (Weeks 10-12)
### "Production Ready" - Enterprise-Grade Reliability

### 4.1 Comprehensive Test Suite

```python
# FILE: tests/security/test_double_spend.py

"""
Double-Spend Attack Prevention Test Suite

Tests the FTNS system's resistance to various double-spend attack vectors.
"""

import asyncio
import pytest
from decimal import Decimal
from unittest.mock import patch
import uuid

from prsm.economy.tokenomics import get_ftns_service
from prsm.economy.tokenomics.production_ledger import ProductionFTNSLedger


class TestDoubleSpendPrevention:
    """Test suite for double-spend attack prevention"""

    @pytest.fixture
    async def ledger(self, test_database):
        """Create test ledger instance"""
        return ProductionFTNSLedger()

    @pytest.fixture
    async def funded_user(self, ledger):
        """Create user with initial balance"""
        user_id = f"test_user_{uuid.uuid4().hex[:8]}"
        await ledger.mint_tokens(
            user_id,
            Decimal("1000.00"),
            "Test funding"
        )
        return user_id

    @pytest.mark.asyncio
    async def test_concurrent_deductions_atomic(self, ledger, funded_user):
        """
        Test that concurrent deductions cannot exceed balance.

        Attack vector: Send 10 concurrent requests each trying to
        deduct the full balance. Only one should succeed.
        """
        balance = await ledger.get_balance(funded_user)
        deduction_amount = balance.balance  # Try to deduct full balance

        # Create 10 concurrent deduction tasks
        async def attempt_deduction(i: int):
            try:
                return await ledger.transfer_tokens(
                    from_user_id=funded_user,
                    to_user_id="recipient",
                    amount=deduction_amount,
                    description=f"Concurrent deduction {i}",
                    idempotency_key=f"concurrent_{i}_{uuid.uuid4().hex}"
                )
            except ValueError:
                return None

        # Execute concurrently
        results = await asyncio.gather(
            *[attempt_deduction(i) for i in range(10)],
            return_exceptions=True
        )

        # Count successful deductions
        successful = [r for r in results if r and not isinstance(r, Exception)]

        # Verify only one succeeded
        assert len(successful) == 1, \
            f"Expected 1 successful deduction, got {len(successful)}"

        # Verify final balance is zero (not negative)
        final_balance = await ledger.get_balance(funded_user)
        assert final_balance.balance == Decimal("0"), \
            f"Expected 0 balance, got {final_balance.balance}"

    @pytest.mark.asyncio
    async def test_idempotency_prevents_replay(self, ledger, funded_user):
        """
        Test that idempotency keys prevent replay attacks.

        Attack vector: Replay the same transaction multiple times.
        """
        idempotency_key = f"idempotent_{uuid.uuid4().hex}"
        amount = Decimal("100.00")

        # First request should succeed
        tx1 = await ledger.transfer_tokens(
            from_user_id=funded_user,
            to_user_id="recipient",
            amount=amount,
            description="First transfer",
            idempotency_key=idempotency_key
        )
        assert tx1 is not None

        # Replay with same idempotency key should be rejected
        with pytest.raises(Exception) as exc_info:
            await ledger.transfer_tokens(
                from_user_id=funded_user,
                to_user_id="recipient",
                amount=amount,
                description="Replay attempt",
                idempotency_key=idempotency_key
            )

        # Verify balance only deducted once
        balance = await ledger.get_balance(funded_user)
        assert balance.balance == Decimal("900.00")

    @pytest.mark.asyncio
    async def test_race_condition_between_check_and_update(self, ledger, funded_user):
        """
        Test TOCTOU (Time-of-Check to Time-of-Use) prevention.

        Attack vector: Interleave balance checks and updates.
        """
        # This test simulates the race condition by using precise timing
        async def slow_deduction():
            # Add artificial delay between check and update
            # This would exploit TOCTOU in vulnerable implementations
            return await ledger.transfer_tokens(
                from_user_id=funded_user,
                to_user_id="recipient_1",
                amount=Decimal("600.00"),
                description="Slow deduction",
                idempotency_key=f"slow_{uuid.uuid4().hex}"
            )

        async def fast_deduction():
            return await ledger.transfer_tokens(
                from_user_id=funded_user,
                to_user_id="recipient_2",
                amount=Decimal("600.00"),
                description="Fast deduction",
                idempotency_key=f"fast_{uuid.uuid4().hex}"
            )

        # Run both concurrently
        results = await asyncio.gather(
            slow_deduction(),
            fast_deduction(),
            return_exceptions=True
        )

        # At most one should succeed (both request 600 from 1000 balance)
        # Both could succeed (600 + 600 < 1000 is false, so this tests
        # that balance check is atomic with update)
        successful = [r for r in results if r and not isinstance(r, Exception)]
        failed = [r for r in results if isinstance(r, Exception)]

        # Verify we didn't go negative
        final_balance = await ledger.get_balance(funded_user)
        assert final_balance.balance >= Decimal("0"), \
            "Balance went negative - TOCTOU vulnerability!"

        # Verify total deducted matches successful transactions
        expected_remaining = Decimal("1000.00") - (Decimal("600.00") * len(successful))
        assert final_balance.balance == expected_remaining


# FILE: tests/security/test_jwt_security.py

"""
JWT Security Test Suite

Tests the authentication system's resistance to common JWT attacks.
"""

import pytest
import jwt
import time
from datetime import datetime, timezone, timedelta

from prsm.core.auth.jwt_handler import JWTHandler


class TestJWTSecurity:
    """Test suite for JWT security"""

    @pytest.fixture
    def jwt_handler(self):
        """Create JWT handler instance"""
        return JWTHandler()

    @pytest.mark.asyncio
    async def test_algorithm_confusion_attack(self, jwt_handler):
        """
        Test resistance to algorithm confusion attack.

        Attack: Change algorithm from HS256 to 'none' or RS256
        """
        # Create valid token
        user_data = {
            "user_id": "test-user",
            "username": "testuser",
            "email": "test@example.com",
            "role": "user"
        }

        token, _ = await jwt_handler.create_access_token(user_data)

        # Decode without verification to get payload
        payload = jwt.decode(token, options={"verify_signature": False})

        # Create malicious token with 'none' algorithm
        malicious_token_none = jwt.encode(
            payload,
            "",  # Empty key for 'none' algorithm
            algorithm="none"
        )

        # Should be rejected
        result = await jwt_handler.verify_token(malicious_token_none)
        assert result is None, "Algorithm confusion with 'none' should be rejected"

        # Create malicious token with different algorithm
        try:
            malicious_token_rs256 = jwt.encode(
                payload,
                jwt_handler.secret_key,  # Using secret as public key
                algorithm="RS256"
            )
            result = await jwt_handler.verify_token(malicious_token_rs256)
            assert result is None, "Algorithm confusion with RS256 should be rejected"
        except jwt.exceptions.InvalidKeyError:
            # This is also acceptable - key format rejection
            pass

    @pytest.mark.asyncio
    async def test_expired_token_rejected(self, jwt_handler):
        """Test that expired tokens are rejected"""
        user_data = {
            "user_id": "test-user",
            "username": "testuser",
            "email": "test@example.com",
            "role": "user"
        }

        # Create token that's already expired
        expired_delta = timedelta(minutes=-1)
        token, _ = await jwt_handler.create_access_token(
            user_data,
            expires_delta=expired_delta
        )

        result = await jwt_handler.verify_token(token)
        assert result is None, "Expired token should be rejected"

    @pytest.mark.asyncio
    async def test_revoked_token_rejected(self, jwt_handler, redis_client):
        """Test that revoked tokens are rejected"""
        jwt_handler.redis_client = redis_client
        await jwt_handler.initialize()

        user_data = {
            "user_id": "test-user",
            "username": "testuser",
            "email": "test@example.com",
            "role": "user"
        }

        token, _ = await jwt_handler.create_access_token(user_data)

        # Token should initially be valid
        result = await jwt_handler.verify_token(token)
        assert result is not None, "Valid token should be accepted"

        # Revoke the token
        await jwt_handler.revoke_token(token)

        # Token should now be rejected
        result = await jwt_handler.verify_token(token)
        assert result is None, "Revoked token should be rejected"

    @pytest.mark.asyncio
    async def test_tampered_signature_rejected(self, jwt_handler):
        """Test that tokens with tampered signatures are rejected"""
        user_data = {
            "user_id": "test-user",
            "username": "testuser",
            "email": "test@example.com",
            "role": "user"
        }

        token, _ = await jwt_handler.create_access_token(user_data)

        # Tamper with the signature (last part of JWT)
        parts = token.split('.')
        tampered_signature = parts[2][:-4] + "XXXX"  # Change last 4 chars
        tampered_token = f"{parts[0]}.{parts[1]}.{tampered_signature}"

        result = await jwt_handler.verify_token(tampered_token)
        assert result is None, "Tampered token should be rejected"
```

### 4.2 Security Hardening Checklist

```markdown
## Pre-Production Security Checklist

### Authentication & Authorization
- [ ] JWT secret key is at least 256 bits (32 characters)
- [ ] JWT algorithm is explicitly specified (no algorithm confusion)
- [ ] Token revocation is functional and tested
- [ ] Refresh token rotation is implemented
- [ ] Failed login attempts are rate-limited
- [ ] Account lockout after repeated failures
- [ ] Password hashing uses bcrypt with >=12 rounds

### FTNS/Financial Operations
- [ ] All balance operations use database transactions
- [ ] SELECT FOR UPDATE used for concurrent access
- [ ] Idempotency keys prevent duplicate operations
- [ ] Balance constraints prevent negative values
- [ ] Transaction recovery system operational
- [ ] Audit trail captures all financial operations

### API Security
- [ ] Rate limiting enforced at all tiers
- [ ] Rate limit state in Redis (not in-memory)
- [ ] Input validation on all endpoints
- [ ] Output encoding prevents XSS
- [ ] SQL injection prevented (parameterized queries)
- [ ] CORS properly configured
- [ ] Security headers present on all responses

### Infrastructure
- [ ] Secrets not in code or config files
- [ ] Database credentials rotated
- [ ] Redis requires authentication
- [ ] TLS 1.3 required for all connections
- [ ] Health endpoints don't leak sensitive info
- [ ] Error messages don't reveal internals

### Monitoring & Response
- [ ] Security events logged to audit system
- [ ] Alerts configured for suspicious activity
- [ ] Incident response plan documented
- [ ] Recovery procedures tested
```

### Sprint 4 Definition of Done

| Criteria | Metric | Verification Method |
|----------|--------|---------------------|
| Security test coverage | 100% of critical paths | Coverage report |
| Penetration test pass | 0 critical/high findings | Third-party pentest |
| Documentation complete | All APIs documented | Doc review |
| Runbook created | <15 min incident response | Tabletop exercise |
| Load test passed | 1000 concurrent users | Load test |

---

## Summary: 12-Week Sprint Backlog

| Sprint | Weeks | Focus | Key Deliverables |
|--------|-------|-------|------------------|
| Phase 0 | Week 1 | Stop the Bleed | Double-spend fix, JWT revocation |
| Sprint 1 | Weeks 1-3 | Database Integrity | PostgreSQL FTNS, atomic operations, recovery |
| Sprint 2 | Weeks 4-6 | Cryptography | PQC resolution (implement or deprecate) |
| Sprint 3 | Weeks 7-9 | Scalability | API modularization, Redis rate limiting |
| Sprint 4 | Weeks 10-12 | Hardening | Security tests, documentation, audit prep |

## Success Metrics Mapping to Audit Criteria

| Audit Finding | Target Score | Remediation |
|--------------|--------------|-------------|
| Double-spend vulnerability | 2/10 → 9/10 | Atomic transactions + idempotency |
| JWT verification bypass | 3/10 → 9/10 | Real revocation + algorithm enforcement |
| In-memory financial state | 2/10 → 9/10 | PostgreSQL persistence |
| Mock cryptography | 3/10 → 8/10 | Real implementation or clear deprecation |
| Monolithic architecture | 5/10 → 8/10 | Modular routers |
| Rate limiting | 6/10 → 9/10 | Redis-backed sliding window |

**Expected Post-Remediation Score: 8.5/10**

---

## Appendix A: Migration Scripts

See `scripts/migrations/` for:
- `003_ftns_atomic_operations.sql`
- `004_token_revocation.sql`
- `005_enhanced_ftns_schema.sql`

## Appendix B: Configuration Changes

Required environment variables for production:
```bash
# Security
PRSM_SECRET_KEY="<256-bit-random-key>"
PRSM_JWT_ALGORITHM="HS256"

# Database
PRSM_DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/prsm"

# Redis
PRSM_REDIS_URL="redis://:password@host:6379/0"

# PQC (if enabled)
PRSM_PQC_MODE="disabled"  # or "hybrid" if liboqs installed
```

## Appendix C: Rollback Procedures

Each sprint includes rollback procedures in case of critical issues. See `docs/operations/ROLLBACK_PROCEDURES.md`.

---

*Document Version: 1.0*
*Last Updated: 2025-01-16*
*Author: PRSM Engineering Team*
*Classification: Internal - Board Presentation*
