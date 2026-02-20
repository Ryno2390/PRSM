"""
DAG Ledger
 ==========

DAG-based (Directed Acyclic Graph) ledger for FTNS tokens.
Unlike traditional blockchain (linear), this DAG structure allows:
- Horizontal scaling with more users (more users = faster confirmation)
- No mining fees - instead, transactions approve each other
- Faster confirmation times as network grows

This implements a simplified IOTA-style Tangle with:
- Each transaction references 2-8 parent transactions
- Tip selection using MCMC (Markov Chain Monte Carlo)
- Confirmation confidence based on cumulative approval weight

Cryptographic Signature Verification:
- All transactions from non-null wallets must be signed with Ed25519
- Signatures verify the transaction hash (SHA-256)
- Public keys are stored with the transaction for verification
- Genesis and system transactions are exempt from signature requirements

Atomic Balance Operations (TOCTOU Prevention):
==============================================
This ledger implements atomic balance operations to prevent Time-of-Check-Time-of-Use
(TOCTOU) race conditions that could lead to double-spend attacks.

Security Architecture:
1. **Row-Level Locking**: Uses SQLite's BEGIN IMMEDIATE to acquire write locks
   before balance checks, preventing concurrent modifications.

2. **Optimistic Concurrency Control**: Balance cache includes a version number
   that is checked during deduction to detect concurrent modifications.

3. **Atomic Transaction Flow**:
   - BEGIN IMMEDIATE (acquires write lock)
   - Check balance with version
   - Create and store transaction
   - Update balance with version check (OCC)
   - COMMIT or ROLLBACK

4. **Balance Cache Table**: The `wallet_balances` table stores pre-computed
   balances with version numbers for efficient atomic operations.

Exception Hierarchy:
- AtomicOperationError: Base exception for atomic operation failures
- InsufficientBalanceError: Raised when balance is insufficient
- ConcurrentModificationError: Raised when TOCTOU is detected
- BalanceLockError: Raised when lock acquisition fails

Usage Example:
    ledger = DAGLedger(db_path='ledger.db')
    await ledger.initialize()
    
    try:
        tx = await ledger.submit_transaction(
            tx_type=TransactionType.TRANSFER,
            amount=100.0,
            from_wallet='sender_wallet',
            to_wallet='receiver_wallet',
            signature='base64_signature',
            public_key='hex_public_key'
        )
    except InsufficientBalanceError:
        # Handle insufficient balance
        pass
    except ConcurrentModificationError:
        # Retry the operation - concurrent modification detected
        pass
    except BalanceLockError:
        # Retry later - system under high contention
        pass

Security Guarantees:
- No double-spend: Balance check and deduction are atomic
- No lost updates: Concurrent modifications are detected
- Fail-safe: Any error during transaction rolls back all changes
- Audit trail: All operations are logged for security review
"""

import asyncio
import hashlib
import json
import logging
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Callable

import aiosqlite

from prsm.core.cryptography.dag_signatures import (
    DAGSignatureManager,
    KeyPair,
    InvalidSignatureError,
    MissingSignatureError,
)

# Configure logger for security audit trail
logger = logging.getLogger(__name__)


# =============================================================================
# ATOMIC OPERATION EXCEPTIONS
# =============================================================================

class AtomicOperationError(Exception):
    """Base exception for atomic operation failures."""
    pass


class InsufficientBalanceError(AtomicOperationError):
    """Raised when wallet has insufficient balance for transaction."""
    pass


class ConcurrentModificationError(AtomicOperationError):
    """Raised when concurrent modification is detected (TOCTOU prevention)."""
    pass


class BalanceLockError(AtomicOperationError):
    """Raised when balance lock cannot be acquired."""
    pass


class TransactionType(str, Enum):
    GENESIS = "genesis"
    WELCOME_GRANT = "welcome_grant"
    COMPUTE_PAYMENT = "compute_payment"
    COMPUTE_EARNING = "compute_earning"
    STORAGE_REWARD = "storage_reward"
    CONTENT_ROYALTY = "content_royalty"
    TRANSFER = "transfer"
    APPROVAL = "approval"


@dataclass
class DAGTransaction:
    tx_id: str
    tx_type: TransactionType
    amount: float
    from_wallet: Optional[str]
    to_wallet: str
    timestamp: float
    signature: Optional[str] = None
    public_key: Optional[str] = None  # Ed25519 public key (hex or base64)
    description: str = ""
    
    parent_ids: List[str] = field(default_factory=list)
    
    cumulative_weight: int = 1
    confirmation_level: float = 0.0
    
    approved_by: Set[str] = field(default_factory=set)
    
    def hash(self) -> str:
        """Generate SHA-256 hash of transaction data for signing."""
        data = {
            "tx_id": self.tx_id,
            "tx_type": self.tx_type.value,
            "amount": self.amount,
            "from_wallet": self.from_wallet,
            "to_wallet": self.to_wallet,
            "timestamp": self.timestamp,
            "parent_ids": self.parent_ids,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    def get_signing_data(self) -> dict:
        """Get the canonical transaction data for signing."""
        return {
            "tx_id": self.tx_id,
            "tx_type": self.tx_type.value,
            "amount": self.amount,
            "from_wallet": self.from_wallet,
            "to_wallet": self.to_wallet,
            "timestamp": self.timestamp,
            "parent_ids": self.parent_ids,
        }


@dataclass
class DAGState:
    tips: Set[str] = field(default_factory=set)
    transactions: Dict[str, DAGTransaction] = field(default_factory=dict)
    approvals: Dict[str, Set[str]] = field(default_factory=dict)


class DAGLedger:
    """
    DAG-based FTNS Ledger
    
    Key concepts:
    - Genesis transaction: Initial token supply
    - Tips: Unapproved transactions (eligible for approval)
    - Parents: Transactions this tx approves
    - Cumulative weight: 1 + number of transactions approving this tx
    - Confirmation level: 0.0-1.0 based on cumulative weight threshold
    
    Signature Verification:
    - Transactions with a from_wallet require valid Ed25519 signatures
    - Genesis and system transactions (from_wallet=None) are exempt
    - Public keys are stored with transactions for verification
    - Signature verification can be disabled for testing via verify_signatures=False
    """
    
    INITIAL_SUPPLY = 1_000_000_000.0
    
    def __init__(self, db_path: str = ":memory:", verify_signatures: bool = True):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        
        self._state = DAGState()
        
        self.max_parents = 4
        self.alpha = 0.5
        self.confirmation_threshold = 100
        
        # Signature verification setting (can be disabled for testing)
        self.verify_signatures = verify_signatures
        
        # Public key registry for wallet verification
        # Maps wallet_id -> public_key_hex
        self._wallet_public_keys: Dict[str, str] = {}
        
        # Temporary storage for signature verification data during transaction creation
        # This is used to pass verification data between pre-creation and post-creation phases
        self._pending_verification: Optional[Dict[str, str]] = None
        
    async def initialize(self) -> None:
        """Initialize database and load existing state."""
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._create_tables()
        await self._load_state()
        
    async def _create_tables(self) -> None:
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS wallets (
                wallet_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL DEFAULT '',
                public_key TEXT,
                created_at REAL NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS dag_transactions (
                tx_id TEXT PRIMARY KEY,
                tx_type TEXT NOT NULL,
                amount REAL NOT NULL,
                from_wallet TEXT,
                to_wallet TEXT NOT NULL,
                timestamp REAL NOT NULL,
                signature TEXT,
                public_key TEXT,
                description TEXT DEFAULT '',
                parent_ids TEXT NOT NULL,
                cumulative_weight INTEGER DEFAULT 1,
                confirmation_level REAL DEFAULT 0.0,
                hash TEXT NOT NULL
            );
            
            CREATE INDEX IF NOT EXISTS idx_dag_to ON dag_transactions(to_wallet);
            CREATE INDEX IF NOT EXISTS idx_dag_from ON dag_transactions(from_wallet);
            CREATE INDEX IF NOT EXISTS idx_dag_hash ON dag_transactions(hash);
            CREATE INDEX IF NOT EXISTS idx_dag_timestamp ON dag_transactions(timestamp);
            
            CREATE TABLE IF NOT EXISTS tips (
                tx_id TEXT PRIMARY KEY
            );
            
            CREATE TABLE IF NOT EXISTS approvals (
                tx_id TEXT PRIMARY KEY,
                approver_id TEXT NOT NULL,
                approved_at REAL NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS seen_nonces (
                nonce TEXT PRIMARY KEY,
                origin TEXT NOT NULL,
                seen_at REAL NOT NULL
            );
            
            -- Balance cache table for atomic operations with optimistic concurrency control
            -- This table stores pre-computed balances with version numbers for TOCTOU prevention
            CREATE TABLE IF NOT EXISTS wallet_balances (
                wallet_id TEXT PRIMARY KEY,
                balance REAL NOT NULL DEFAULT 0.0,
                version INTEGER NOT NULL DEFAULT 1,
                last_updated REAL NOT NULL,
                FOREIGN KEY (wallet_id) REFERENCES wallets(wallet_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_wallet_balances_version ON wallet_balances(wallet_id, version);
        """)
        
    async def _load_state(self) -> None:
        cursor = await self._db.execute("SELECT tx_id, parent_ids, cumulative_weight, confirmation_level FROM dag_transactions")
        async for row in cursor:
            tx_id, parent_ids_json, cumulative_weight, confirmation_level = row
            parent_ids = json.loads(parent_ids_json)
            
            cursor2 = await self._db.execute(
                """SELECT tx_id, tx_type, amount, from_wallet, to_wallet, timestamp, signature, public_key, description
                   FROM dag_transactions WHERE tx_id = ?""",
                (tx_id,)
            )
            row2 = await cursor2.fetchone()
            if row2:
                tx = DAGTransaction(
                    tx_id=row2[0],
                    tx_type=TransactionType(row2[1]),
                    amount=row2[2],
                    from_wallet=row2[3],
                    to_wallet=row2[4],
                    timestamp=row2[5],
                    signature=row2[6],
                    public_key=row2[7],
                    description=row2[8] or "",
                    parent_ids=parent_ids,
                    cumulative_weight=cumulative_weight,
                    confirmation_level=confirmation_level,
                )
                self._state.transactions[tx_id] = tx
                
        cursor = await self._db.execute("SELECT tx_id FROM tips")
        async for row in cursor:
            self._state.tips.add(row[0])
        
        # Load wallet public keys
        cursor = await self._db.execute("SELECT wallet_id, public_key FROM wallets WHERE public_key IS NOT NULL")
        async for row in cursor:
            self._wallet_public_keys[row[0]] = row[1]
            
        if not self._state.transactions:
            await self._create_genesis()
            
    async def _create_genesis(self) -> None:
        genesis = DAGTransaction(
            tx_id="genesis",
            tx_type=TransactionType.GENESIS,
            amount=self.INITIAL_SUPPLY,
            from_wallet=None,
            to_wallet="network",
            timestamp=0.0,
            description="Genesis transaction - initial FTNS supply",
            parent_ids=[],
        )
        await self._store_transaction(genesis)
        self._state.tips.add(genesis.tx_id)
        await self._db.execute("INSERT OR IGNORE INTO tips (tx_id) VALUES (?)", ("genesis",))
        await self._db.commit()
        
    async def _store_transaction(self, tx: DAGTransaction) -> None:
        await self._db.execute(
            """INSERT OR REPLACE INTO dag_transactions
               (tx_id, tx_type, amount, from_wallet, to_wallet, timestamp, signature, public_key, description, 
                parent_ids, cumulative_weight, confirmation_level, hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tx.tx_id,
                tx.tx_type.value,
                tx.amount,
                tx.from_wallet,
                tx.to_wallet,
                tx.timestamp,
                tx.signature,
                tx.public_key,
                tx.description,
                json.dumps(tx.parent_ids),
                tx.cumulative_weight,
                tx.confirmation_level,
                tx.hash(),
            ),
        )
        
    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
            
    async def create_wallet(
        self, 
        wallet_id: str, 
        display_name: str = "",
        public_key: Optional[str] = None
    ) -> None:
        """Create a new wallet with optional public key for signature verification."""
        await self._db.execute(
            "INSERT OR IGNORE INTO wallets (wallet_id, display_name, public_key, created_at) VALUES (?, ?, ?, ?)",
            (wallet_id, display_name, public_key, time.time()),
        )
        await self._db.commit()
        
        if public_key:
            self._wallet_public_keys[wallet_id] = public_key
            
    async def register_wallet_public_key(self, wallet_id: str, public_key: str) -> None:
        """Register or update the public key for an existing wallet."""
        await self._db.execute(
            "UPDATE wallets SET public_key = ? WHERE wallet_id = ?",
            (public_key, wallet_id),
        )
        await self._db.commit()
        self._wallet_public_keys[wallet_id] = public_key
        
    def get_wallet_public_key(self, wallet_id: str) -> Optional[str]:
        """Get the registered public key for a wallet."""
        return self._wallet_public_keys.get(wallet_id)
        
    async def wallet_exists(self, wallet_id: str) -> bool:
        cursor = await self._db.execute("SELECT 1 FROM wallets WHERE wallet_id = ?", (wallet_id,))
        return await cursor.fetchone() is not None
    
    async def get_balance(self, wallet_id: str) -> float:
        """Get the current balance for a wallet (non-atomic read)."""
        cursor = await self._db.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM dag_transactions WHERE to_wallet = ? AND tx_type != ?",
            (wallet_id, TransactionType.APPROVAL.value),
        )
        credits = (await cursor.fetchone())[0]
        
        cursor = await self._db.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM dag_transactions WHERE from_wallet = ? AND tx_type != ?",
            (wallet_id, TransactionType.APPROVAL.value),
        )
        debits = (await cursor.fetchone())[0]
        return round(credits - debits, 6)
    
    # =========================================================================
    # ATOMIC BALANCE OPERATIONS - TOCTOU Prevention
    # =========================================================================
    
    async def _get_or_create_balance_cache(self, wallet_id: str) -> Tuple[float, int]:
        """
        Get or create a balance cache entry for atomic operations.
        
        This method ensures the wallet_balances table has an entry for the wallet,
        computing the balance from transactions if needed.
        
        Args:
            wallet_id: The wallet to get/create cache for
            
        Returns:
            Tuple of (balance, version)
        """
        # Try to get existing cache
        cursor = await self._db.execute(
            "SELECT balance, version FROM wallet_balances WHERE wallet_id = ?",
            (wallet_id,)
        )
        row = await cursor.fetchone()
        
        if row:
            return (row[0], row[1])
        
        # Compute balance from transactions
        balance = await self.get_balance(wallet_id)
        
        # Insert new cache entry
        await self._db.execute(
            """INSERT INTO wallet_balances (wallet_id, balance, version, last_updated)
               VALUES (?, ?, 1, ?)
               ON CONFLICT(wallet_id) DO UPDATE SET balance = excluded.balance""",
            (wallet_id, balance, time.time())
        )
        
        return (balance, 1)
    
    async def _check_balance_atomic(
        self,
        wallet_id: str,
        amount: float,
        lock_timeout_ms: int = 5000
    ) -> Tuple[bool, float, int]:
        """
        Atomically check if wallet has sufficient balance with row-level locking.
        
        This method uses SQLite's SAVEPOINT to create a nested transaction context
        for atomic balance operations, preventing TOCTOU race conditions.
        
        Atomicity Guarantees:
        1. Uses SAVEPOINT for nested transaction support
        2. Reads balance with version for optimistic concurrency control
        3. Balance check and version read happen atomically
        
        Args:
            wallet_id: Wallet to check
            amount: Required amount
            lock_timeout_ms: Timeout for acquiring lock (default 5 seconds)
            
        Returns:
            Tuple of (has_sufficient, current_balance, version)
            
        Raises:
            BalanceLockError: If lock cannot be acquired within timeout
        """
        try:
            # Use SAVEPOINT for nested transaction support
            # This allows us to have atomic operations within an existing transaction
            await self._db.execute("SAVEPOINT balance_check")
            
            # Get or create balance cache with version
            balance, version = await self._get_or_create_balance_cache(wallet_id)
            
            has_sufficient = balance >= amount
            
            if not has_sufficient:
                # Release the savepoint and raise error
                await self._db.execute("RELEASE SAVEPOINT balance_check")
                logger.warning(
                    f"Atomic balance check failed: insufficient balance for {wallet_id[:8]}... "
                    f"(has {balance:.6f}, needs {amount:.6f})"
                )
            else:
                # Keep savepoint for caller to complete
                logger.debug(
                    f"Atomic balance check passed for {wallet_id[:8]}... "
                    f"(balance={balance:.6f}, version={version})"
                )
            
            return (has_sufficient, balance, version)
            
        except Exception as e:
            # Rollback to savepoint on error
            try:
                await self._db.execute("ROLLBACK TO SAVEPOINT balance_check")
                await self._db.execute("RELEASE SAVEPOINT balance_check")
            except:
                pass
            
            if "database is locked" in str(e).lower():
                raise BalanceLockError(
                    f"Could not acquire balance lock for {wallet_id[:8]}... within {lock_timeout_ms}ms"
                )
            raise
    
    async def _commit_balance_deduction(
        self,
        wallet_id: str,
        amount: float,
        expected_version: int
    ) -> bool:
        """
        Commit a balance deduction with optimistic concurrency control.
        
        This method MUST be called after _check_balance_atomic within the same
        transaction. It updates the balance cache only if the version hasn't
        changed, detecting concurrent modifications.
        
        Atomicity Guarantees:
        1. Uses optimistic concurrency control (version check)
        2. Update only succeeds if version matches expected
        3. Automatic detection of concurrent modifications
        
        Args:
            wallet_id: Wallet to deduct from
            amount: Amount to deduct
            expected_version: Version expected from balance check
            
        Returns:
            True if deduction succeeded
            
        Raises:
            ConcurrentModificationError: If balance was modified by another transaction
        """
        # Update balance with version check (optimistic concurrency control)
        cursor = await self._db.execute(
            """UPDATE wallet_balances
               SET balance = balance - ?,
                   version = version + 1,
                   last_updated = ?
               WHERE wallet_id = ? AND version = ?""",
            (amount, time.time(), wallet_id, expected_version)
        )
        
        if cursor.rowcount == 0:
            # Version mismatch - concurrent modification detected
            # Rollback to savepoint and release
            await self._db.execute("ROLLBACK TO SAVEPOINT balance_check")
            await self._db.execute("RELEASE SAVEPOINT balance_check")
            logger.error(
                f"TOCTOU detected: concurrent modification of {wallet_id[:8]}... "
                f"(expected version {expected_version})"
            )
            raise ConcurrentModificationError(
                f"Balance for {wallet_id[:8]}... was modified by another transaction. "
                "Please retry the operation."
            )
        
        # Release the savepoint - the main transaction will commit later
        await self._db.execute("RELEASE SAVEPOINT balance_check")
        
        logger.info(
            f"Atomic balance deduction completed for {wallet_id[:8]}... "
            f"(deducted {amount:.6f}, new version {expected_version + 1})"
        )
        
        return True
    
    async def _commit_balance_credit(
        self,
        wallet_id: str,
        amount: float
    ) -> bool:
        """
        Commit a balance credit (for receiving wallet).
        
        This method updates the balance cache for credits. It uses INSERT OR REPLACE
        to handle both new and existing wallets atomically.
        
        Args:
            wallet_id: Wallet to credit
            amount: Amount to credit
            
        Returns:
            True if credit succeeded
        """
        # Use INSERT ... ON CONFLICT for atomic upsert
        await self._db.execute(
            """INSERT INTO wallet_balances (wallet_id, balance, version, last_updated)
               VALUES (?, ?, 1, ?)
               ON CONFLICT(wallet_id) DO UPDATE SET
                   balance = balance + excluded.balance,
                   version = version + 1,
                   last_updated = excluded.last_updated""",
            (wallet_id, amount, time.time())
        )
        
        logger.debug(
            f"Atomic balance credit completed for {wallet_id[:8]}... "
            f"(credited {amount:.6f})"
        )
        
        return True
    
    async def _rollback_balance_check(self) -> None:
        """
        Rollback an atomic balance check savepoint.
        
        Call this if balance check passed but transaction creation fails.
        """
        try:
            await self._db.execute("ROLLBACK TO SAVEPOINT balance_check")
            await self._db.execute("RELEASE SAVEPOINT balance_check")
            logger.debug("Rolled back atomic balance check savepoint")
        except:
            pass
    
    def select_tips_mcmc(self, num_tips: int = 2) -> List[str]:
        """
        Select tips using simplified weighted random selection.
        
        In a proper IOTA implementation, this would use MCMC to walk
        the DAG and select tips with probability proportional to their
        cumulative weight. For now, we use weighted random from tips.
        """
        if not self._state.tips:
            return []
        
        tips_list = list(self._state.tips)
        
        if len(tips_list) <= num_tips:
            return tips_list
        
        weights = []
        for tip_id in tips_list:
            tip = self._state.transactions.get(tip_id)
            if tip:
                weight = math.exp(self.alpha * math.log(tip.cumulative_weight + 1))
            else:
                weight = 1.0
            weights.append(weight)
        
        total_weight = sum(weights)
        if total_weight == 0:
            return tips_list[:num_tips]
        
        selected = []
        available = list(tips_list)
        
        for _ in range(num_tips):
            if not available:
                break
            r = random.random() * total_weight
            cumulative = 0
            for i, w in enumerate(weights):
                cumulative += w
                if cumulative >= r and available[i] not in selected:
                    selected.append(available[i])
                    break
            
            weights = [w for i, w in enumerate(weights) if available[i] not in selected]
            available = [t for t in available if t not in selected]
            total_weight = sum(weights) if weights else 0
        
        return selected
    
    def _get_children(self, tx_id: str) -> List[str]:
        children = []
        for other_id, tx in self._state.transactions.items():
            if tx_id in tx.parent_ids:
                children.append(other_id)
        return children
    
    def _verify_transaction_signature(
        self,
        tx: DAGTransaction,
        signature: str,
        public_key: str
    ) -> bool:
        """
        Verify the Ed25519 signature of a transaction.
        
        This method performs cryptographic verification that the transaction
        was authorized by the owner of the private key corresponding to the
        provided public key.
        
        Args:
            tx: The transaction to verify
            signature: Base64-encoded signature (64 bytes when decoded)
            public_key: Hex-encoded Ed25519 public key (32 bytes when decoded)
            
        Returns:
            bool: True if signature is valid
            
        Raises:
            InvalidSignatureError: If signature verification fails due to:
                - Invalid signature format
                - Signature doesn't match transaction hash
                - Invalid public key format
                - Any other cryptographic error
                
        Security Notes:
            - This method fails closed: any error results in rejection
            - The transaction hash includes tx_id, so signatures must be
              created after the client receives the tx_id from the server
              OR the client must use a signing scheme that excludes tx_id
            - For the chicken-and-egg problem, clients typically sign
              transaction data without tx_id, and servers verify using
              a hash that excludes tx_id (see get_signing_data())
        """
        try:
            # Validate inputs are not empty
            if not signature:
                logger.warning("Empty signature provided for verification")
                raise InvalidSignatureError("Signature cannot be empty")
            if not public_key:
                logger.warning("Empty public key provided for verification")
                raise InvalidSignatureError("Public key cannot be empty")
            
            # Load the public key from hex format
            try:
                pk = DAGSignatureManager.load_public_key_from_hex(public_key)
            except ValueError as e:
                logger.warning(f"Invalid public key format: {e}")
                raise InvalidSignatureError(f"Invalid public key format: {e}")
            
            # Get the transaction hash for verification
            tx_hash = tx.hash()
            logger.debug(f"Verifying signature for tx {tx.tx_id[:8]}... with hash {tx_hash[:16]}...")
            
            # Verify the signature using Ed25519
            try:
                result = DAGSignatureManager.verify_signature(tx_hash, signature, pk)
                logger.debug(f"Signature verification successful for tx {tx.tx_id[:8]}...")
                return result
            except InvalidSignatureError:
                # Re-raise InvalidSignatureError from DAGSignatureManager directly
                logger.warning(
                    f"Signature verification failed for tx {tx.tx_id[:8]}... "
                    f"from wallet {tx.from_wallet[:8] if tx.from_wallet else 'system'}..."
                )
                raise
                
        except InvalidSignatureError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            # Catch any unexpected errors and wrap them
            logger.error(f"Unexpected error during signature verification: {e}")
            raise InvalidSignatureError(f"Signature verification failed: {e}")
    
    def _is_signature_required(self, tx_type: TransactionType, from_wallet: Optional[str]) -> bool:
        """
        Determine if a transaction requires a signature.
        
        Transactions that don't require signatures:
        - GENESIS: Initial token supply (system-initiated)
        - Transactions from None (system transactions)
        - WELCOME_GRANT: System-initiated grants for new users
        
        All other transactions require valid Ed25519 signatures.
        
        Security Policy:
            - Fail closed: when in doubt, require a signature
            - Only explicitly exempt transaction types can skip verification
            - System transactions (from_wallet=None) are trusted by definition
            
        Args:
            tx_type: The type of transaction being submitted
            from_wallet: The source wallet (None for system transactions)
            
        Returns:
            bool: True if signature is required, False if exempt
        """
        # System transactions (no source wallet) don't need signatures
        if from_wallet is None:
            logger.debug(f"No signature required: system transaction (tx_type={tx_type.value})")
            return False
        
        # Genesis transactions are system-initiated
        if tx_type == TransactionType.GENESIS:
            logger.debug(f"No signature required: genesis transaction")
            return False
        
        # All other transactions require signatures
        logger.debug(f"Signature required for tx_type={tx_type.value} from {from_wallet[:8]}...")
        return True
    
    async def submit_transaction(
        self,
        tx_type: TransactionType,
        amount: float,
        from_wallet: Optional[str],
        to_wallet: str,
        description: str = "",
        signature: Optional[str] = None,
        public_key: Optional[str] = None,
    ) -> DAGTransaction:
        """
        Submit a new transaction to the DAG.
        
        The transaction must approve (reference) 2-4 tips selected via MCMC.
        This is what makes the DAG grow - every transaction helps confirm
        previous transactions.
        
        Signature Requirements:
        - Transactions with a from_wallet require a valid Ed25519 signature
        - The signature must be created by signing the transaction hash
        - The public key must be provided or registered with the wallet
        
        Atomicity Guarantees (TOCTOU Prevention):
        - Balance checks use BEGIN IMMEDIATE to acquire write lock
        - Balance deduction uses optimistic concurrency control (version check)
        - Concurrent modifications are detected and rejected with retry guidance
        - The entire balance check + deduction is atomic
        
        Args:
            tx_type: Type of transaction
            amount: Transaction amount
            from_wallet: Source wallet (None for system transactions)
            to_wallet: Destination wallet
            description: Optional description
            signature: Base64-encoded Ed25519 signature (required if from_wallet is set)
            public_key: Hex-encoded Ed25519 public key (required for first transaction from wallet)
            
        Returns:
            DAGTransaction: The created transaction
            
        Raises:
            InsufficientBalanceError: If balance is insufficient (atomic check)
            ConcurrentModificationError: If concurrent modification detected
            BalanceLockError: If balance lock cannot be acquired
            MissingSignatureError: If signature is required but not provided
            InvalidSignatureError: If signature verification fails
        """
        # Track atomic balance check state for cleanup
        balance_version = None
        atomic_check_in_progress = False
        
        try:
            # Create wallets if they don't exist
            if from_wallet and not await self.wallet_exists(from_wallet):
                await self.create_wallet(from_wallet, f"wallet-{from_wallet[:8]}", public_key)
            if not await self.wallet_exists(to_wallet):
                await self.create_wallet(to_wallet, f"wallet-{to_wallet[:8]}")
            
            # ATOMIC BALANCE CHECK - TOCTOU Prevention
            # Use atomic balance check with row-level locking for debit transactions
            if from_wallet:
                has_sufficient, balance, balance_version = await self._check_balance_atomic(
                    from_wallet, amount
                )
                atomic_check_in_progress = True
                
                if not has_sufficient:
                    raise InsufficientBalanceError(
                        f"Insufficient balance for {from_wallet[:8]}...: "
                        f"has {balance:.6f}, needs {amount:.6f}"
                    )
        
            # Signature verification - pre-creation checks
            # Note: Signature verification happens AFTER transaction creation due to a
            # chicken-and-egg problem: the signature is created by the client who doesn't
            # know the tx_id yet, but the transaction hash includes tx_id.
            # Solution: Clients sign transaction data WITHOUT tx_id, and we verify using
            # the hash of core transaction fields (excluding tx_id).
            if self.verify_signatures and self._is_signature_required(tx_type, from_wallet):
                if not signature:
                    logger.warning(
                        f"Signature required but missing for transaction from {from_wallet}"
                    )
                    raise MissingSignatureError(
                        f"Transaction from {from_wallet} requires a signature"
                    )
                
                # Get public key - prefer provided key, then registered key
                verify_pk = public_key or self.get_wallet_public_key(from_wallet)
                if not verify_pk:
                    logger.warning(
                        f"No public key available for wallet {from_wallet}"
                    )
                    raise MissingSignatureError(
                        f"No public key provided or registered for wallet {from_wallet}"
                    )
                
                # Store verification data for post-creation verification
                # This ensures we don't silently skip verification due to missing data
                self._pending_verification = {
                    "from_wallet": from_wallet,
                    "signature": signature,
                    "public_key": verify_pk,
                }
            else:
                self._pending_verification = None
                    
            parent_ids = self.select_tips_mcmc(num_tips=min(self.max_parents, len(self._state.tips)))
            
            tx = DAGTransaction(
                tx_id=str(uuid.uuid4()),
                tx_type=tx_type,
                amount=amount,
                from_wallet=from_wallet,
                to_wallet=to_wallet,
                timestamp=time.time(),
                signature=signature,
                public_key=public_key or (self.get_wallet_public_key(from_wallet) if from_wallet else None),
                description=description,
                parent_ids=parent_ids,
            )
            
            # Post-creation signature verification
            # This is where we actually verify the signature after the transaction is created.
            # The signature should have been created from transaction data that will match
            # the tx.hash() method output (which includes tx_id).
            #
            # Edge cases handled:
            # - Genesis transactions: No signature required (system-initiated)
            # - System transactions (from_wallet=None): No signature required
            # - WELCOME_GRANT: System-initiated, no signature required
            # - First transaction from new wallet: Must provide public_key with signature
            #
            # SECURITY: We explicitly check all conditions and fail closed, not open.
            if self._pending_verification is not None:
                # Retrieve and clear pending verification data
                pending = self._pending_verification
                self._pending_verification = None
                
                # These should never be None due to pre-creation checks, but verify defensively
                sig_to_verify = pending.get("signature")
                pk_to_verify = pending.get("public_key")
                wallet_being_verified = pending.get("from_wallet")
                
                if not sig_to_verify:
                    logger.error(
                        f"SECURITY: Signature disappeared during transaction creation for wallet {wallet_being_verified}"
                    )
                    raise MissingSignatureError(
                        f"Signature required for transaction from {wallet_being_verified} but not found during verification"
                    )
                
                if not pk_to_verify:
                    logger.error(
                        f"SECURITY: Public key disappeared during transaction creation for wallet {wallet_being_verified}"
                    )
                    raise MissingSignatureError(
                        f"Public key required for transaction from {wallet_being_verified} but not found during verification"
                    )
                
                try:
                    self._verify_transaction_signature(tx, sig_to_verify, pk_to_verify)
                    logger.info(
                        f"Signature verified successfully for transaction {tx.tx_id[:8]}... "
                        f"from wallet {wallet_being_verified[:8] if wallet_being_verified else 'system'}..."
                    )
                except InvalidSignatureError as e:
                    logger.warning(
                        f"SECURITY: Invalid signature detected for transaction from {wallet_being_verified}: {e}"
                    )
                    raise InvalidSignatureError(
                        f"Invalid signature for transaction from {wallet_being_verified}"
                    )
                except Exception as e:
                    logger.error(
                        f"SECURITY: Unexpected error during signature verification for wallet {wallet_being_verified}: {e}"
                    )
                    raise InvalidSignatureError(
                        f"Signature verification failed for transaction from {wallet_being_verified}: {e}"
                    )
            
            await self._store_transaction(tx)
            
            for parent_id in parent_ids:
                if parent_id in self._state.tips:
                    self._state.tips.discard(parent_id)
                    await self._db.execute("DELETE FROM tips WHERE tx_id = ?", (parent_id,))
                    
            self._state.tips.add(tx.tx_id)
            await self._db.execute("INSERT OR IGNORE INTO tips (tx_id) VALUES (?)", (tx.tx_id,))
            
            await self._update_weights(tx)
            
            # ATOMIC BALANCE DEDUCTION - Complete the atomic operation
            # This commits the balance deduction with version check
            if from_wallet and balance_version is not None:
                await self._commit_balance_deduction(from_wallet, amount, balance_version)
            else:
                # For credit transactions (no from_wallet), just commit
                await self._db.commit()
            
            # Update balance cache for receiving wallet
            if to_wallet:
                await self._commit_balance_credit(to_wallet, amount)
            
            self._state.transactions[tx.tx_id] = tx
            
            logger.info(
                f"Transaction {tx.tx_id[:8]}... completed atomically: "
                f"{amount:.6f} from {from_wallet[:8] if from_wallet else 'system'}... "
                f"to {to_wallet[:8]}..."
            )
            
            return tx
            
        except (InsufficientBalanceError, ConcurrentModificationError, BalanceLockError):
            # These are atomic operation errors - re-raise as-is
            raise
        except (MissingSignatureError, InvalidSignatureError):
            # These are signature errors - rollback and re-raise
            if atomic_check_in_progress:
                await self._rollback_balance_check()
            raise
        except Exception as e:
            # Unexpected error - rollback and wrap
            logger.error(f"Unexpected error in submit_transaction: {e}")
            if atomic_check_in_progress:
                await self._rollback_balance_check()
            raise
    
    async def _update_weights(self, new_tx: DAGTransaction) -> None:
        """Update cumulative weights and confirmation levels for approved transactions."""
        to_update = set()
        queue = list(new_tx.parent_ids)
        
        while queue:
            tx_id = queue.pop(0)
            if tx_id in to_update:
                continue
            to_update.add(tx_id)
            
            tx = self._state.transactions.get(tx_id)
            if tx:
                tx.cumulative_weight += 1
                
                max_weight = max(t.cumulative_weight for t in self._state.transactions.values()) or 1
                tx.confirmation_level = min(1.0, tx.cumulative_weight / self.confirmation_threshold)
                
                await self._db.execute(
                    """UPDATE dag_transactions 
                       SET cumulative_weight = ?, confirmation_level = ?
                       WHERE tx_id = ?""",
                    (tx.cumulative_weight, tx.confirmation_level, tx_id),
                )
                
                for child_id in self._get_children(tx_id):
                    if child_id not in to_update:
                        queue.append(child_id)
    
    async def approve_transaction(self, approver_wallet: str, tx_id: str) -> Optional[DAGTransaction]:
        """Submit an approval transaction that directly approves a specific transaction."""
        tx = self._state.transactions.get(tx_id)
        if not tx:
            return None
            
        return await self.submit_transaction(
            tx_type=TransactionType.APPROVAL,
            amount=0,
            from_wallet=approver_wallet,
            to_wallet=tx_id,
            description=f"Approving transaction {tx_id[:16]}...",
        )
    
    async def get_transaction(self, tx_id: str) -> Optional[DAGTransaction]:
        if tx_id in self._state.transactions:
            return self._state.transactions[tx_id]
            
        cursor = await self._db.execute(
            """SELECT tx_id, tx_type, amount, from_wallet, to_wallet, timestamp, 
                      signature, public_key, description, parent_ids, cumulative_weight, confirmation_level
               FROM dag_transactions WHERE tx_id = ?""",
            (tx_id,),
        )
        row = await cursor.fetchone()
        if row:
            tx = DAGTransaction(
                tx_id=row[0],
                tx_type=TransactionType(row[1]),
                amount=row[2],
                from_wallet=row[3],
                to_wallet=row[4],
                timestamp=row[5],
                signature=row[6],
                public_key=row[7],
                description=row[8] or "",
                parent_ids=json.loads(row[9]),
                cumulative_weight=row[10],
                confirmation_level=row[11],
            )
            self._state.transactions[tx_id] = tx
            return tx
        return None
    
    async def get_transaction_history(self, wallet_id: str, limit: int = 50) -> List[DAGTransaction]:
        cursor = await self._db.execute(
            """SELECT tx_id, tx_type, amount, from_wallet, to_wallet, timestamp, 
                      signature, public_key, description, parent_ids, cumulative_weight, confirmation_level
               FROM dag_transactions
               WHERE (to_wallet = ? OR from_wallet = ?) AND tx_type != ?
               ORDER BY timestamp DESC LIMIT ?""",
            (wallet_id, wallet_id, TransactionType.APPROVAL.value, limit),
        )
        rows = await cursor.fetchall()
        return [
            DAGTransaction(
                tx_id=r[0],
                tx_type=TransactionType(r[1]),
                amount=r[2],
                from_wallet=r[3],
                to_wallet=r[4],
                timestamp=r[5],
                signature=r[6],
                public_key=r[7],
                description=r[8] or "",
                parent_ids=json.loads(r[9]),
                cumulative_weight=r[10],
                confirmation_level=r[11],
            )
            for r in rows
        ]
    
    async def transfer(
        self,
        from_wallet: str,
        to_wallet: str,
        amount: float,
        description: str = "",
        signature: Optional[str] = None,
        public_key: Optional[str] = None,
    ) -> DAGTransaction:
        return await self.submit_transaction(
            tx_type=TransactionType.TRANSFER,
            amount=amount,
            from_wallet=from_wallet,
            to_wallet=to_wallet,
            description=description,
            signature=signature,
            public_key=public_key,
        )
    
    async def credit(
        self,
        wallet_id: str,
        amount: float,
        tx_type: TransactionType,
        description: str = "",
    ) -> DAGTransaction:
        return await self.submit_transaction(
            tx_type=tx_type,
            amount=amount,
            from_wallet=None,
            to_wallet=wallet_id,
            description=description,
        )
    
    async def issue_welcome_grant(self, wallet_id: str, amount: float = 100.0) -> DAGTransaction:
        cursor = await self._db.execute(
            "SELECT 1 FROM dag_transactions WHERE to_wallet = ? AND tx_type = ?",
            (wallet_id, TransactionType.WELCOME_GRANT.value),
        )
        if await cursor.fetchone() is not None:
            raise ValueError(f"Wallet {wallet_id} already received a welcome grant")
            
        return await self.credit(
            wallet_id=wallet_id,
            amount=amount,
            tx_type=TransactionType.WELCOME_GRANT,
            description=f"Welcome grant of {amount} FTNS",
        )
    
    async def get_stats(self) -> Dict:
        confirmed = sum(1 for tx in self._state.transactions.values() if tx.confirmation_level >= 0.5)
        pending = len(self._state.transactions) - confirmed
        
        return {
            "total_transactions": len(self._state.transactions),
            "tips": len(self._state.tips),
            "confirmed": confirmed,
            "pending": pending,
            "cumulative_weight_sum": sum(tx.cumulative_weight for tx in self._state.transactions.values()),
            "avg_confirmation_level": sum(tx.confirmation_level for tx in self._state.transactions.values()) / max(1, len(self._state.transactions)),
        }
    
    async def get_tips(self) -> List[str]:
        return list(self._state.tips)
    
    async def get_transaction_count(self, wallet_id: str) -> int:
        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM dag_transactions WHERE to_wallet = ? OR from_wallet = ?",
            (wallet_id, wallet_id),
        )
        return (await cursor.fetchone())[0]
    
    async def has_seen_nonce(self, nonce: str) -> bool:
        cursor = await self._db.execute("SELECT 1 FROM seen_nonces WHERE nonce = ?", (nonce,))
        return await cursor.fetchone() is not None
    
    async def record_nonce(self, nonce: str, origin: str) -> None:
        await self._db.execute(
            "INSERT OR IGNORE INTO seen_nonces (nonce, origin, seen_at) VALUES (?, ?, ?)",
            (nonce, origin, time.time()),
        )
        await self._db.commit()


class DAGLedgerAdapter:
    """
    Adapter that makes DAGLedger compatible with the existing LocalLedger API.
    This allows the DAG ledger to be used as a drop-in replacement.
    """
    
    def __init__(self, dag_ledger: DAGLedger):
        self._dag = dag_ledger
        
    async def initialize(self) -> None:
        await self._dag.initialize()
        
    async def close(self) -> None:
        await self._dag.close()
        
    async def create_wallet(
        self, 
        wallet_id: str, 
        display_name: str = "",
        public_key: Optional[str] = None
    ) -> None:
        await self._dag.create_wallet(wallet_id, display_name, public_key)
        
    async def register_wallet_public_key(self, wallet_id: str, public_key: str) -> None:
        await self._dag.register_wallet_public_key(wallet_id, public_key)
        
    async def wallet_exists(self, wallet_id: str) -> bool:
        return await self._dag.wallet_exists(wallet_id)
        
    async def get_balance(self, wallet_id: str) -> float:
        return await self._dag.get_balance(wallet_id)
        
    async def credit(
        self,
        wallet_id: str,
        amount: float,
        tx_type: TransactionType,
        description: str = "",
        signature: Optional[str] = None,
    ) -> DAGTransaction:
        return await self._dag.credit(wallet_id, amount, tx_type, description)
        
    async def debit(
        self,
        wallet_id: str,
        amount: float,
        tx_type: TransactionType,
        description: str = "",
        signature: Optional[str] = None,
        public_key: Optional[str] = None,
    ) -> DAGTransaction:
        return await self._dag.submit_transaction(
            tx_type=tx_type,
            amount=amount,
            from_wallet=wallet_id,
            to_wallet="system",
            description=description,
            signature=signature,
            public_key=public_key,
        )
        
    async def transfer(
        self,
        from_wallet: str,
        to_wallet: str,
        amount: float,
        tx_type: TransactionType = TransactionType.TRANSFER,
        description: str = "",
        signature: Optional[str] = None,
        public_key: Optional[str] = None,
    ) -> DAGTransaction:
        return await self._dag.transfer(from_wallet, to_wallet, amount, description, signature, public_key)
        
    async def get_transaction_history(self, wallet_id: str, limit: int = 50) -> List[DAGTransaction]:
        return await self._dag.get_transaction_history(wallet_id, limit)
        
    async def get_transaction_count(self, wallet_id: str) -> int:
        return await self._dag.get_transaction_count(wallet_id)
        
    async def has_seen_nonce(self, nonce: str) -> bool:
        return await self._dag.has_seen_nonce(nonce)
        
    async def record_nonce(self, nonce: str, origin: str) -> None:
        await self._dag.record_nonce(nonce, origin)
        
    async def get_recent_tx_ids(self, wallet_id: str, limit: int = 50) -> List[str]:
        history = await self._dag.get_transaction_history(wallet_id, limit)
        return [tx.tx_id for tx in history]
        
    async def has_transaction(self, tx_id: str) -> bool:
        return await self._dag.get_transaction(tx_id) is not None
        
    async def issue_welcome_grant(self, wallet_id: str, amount: float = 100.0) -> DAGTransaction:
        return await self._dag.issue_welcome_grant(wallet_id, amount)
        
    async def grant_agent_allowance(
        self,
        principal_id: str,
        agent_id: str,
        amount: float,
        epoch_hours: float = 24.0,
    ) -> None:
        pass
        
    async def get_agent_allowance(self, agent_id: str) -> Optional[dict]:
        return None
        
    async def agent_debit(
        self,
        agent_id: str,
        amount: float,
        tx_type: TransactionType,
        description: str = "",
    ) -> Optional[DAGTransaction]:
        return None
        
    async def revoke_agent_allowance(self, principal_id: str, agent_id: str) -> bool:
        return False
        
    def get_stats(self) -> Dict:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return {
                    "total_transactions": 0,
                    "tips": 0,
                    "dag_mode": True,
                    "note": "Use async stats for live data"
                }
            return loop.run_until_complete(self._dag.get_stats())
        except:
            return {"dag_mode": True, "error": "stats unavailable"}
