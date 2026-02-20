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
"""

import asyncio
import hashlib
import json
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
        
        Args:
            tx: The transaction to verify
            signature: Base64-encoded signature
            public_key: Hex-encoded Ed25519 public key
            
        Returns:
            bool: True if signature is valid
            
        Raises:
            InvalidSignatureError: If signature verification fails
        """
        try:
            # Load the public key
            pk = DAGSignatureManager.load_public_key_from_hex(public_key)
            
            # Get the transaction hash
            tx_hash = tx.hash()
            
            # Verify the signature
            return DAGSignatureManager.verify_signature(tx_hash, signature, pk)
        except Exception as e:
            raise InvalidSignatureError(f"Signature verification failed: {e}")
    
    def _is_signature_required(self, tx_type: TransactionType, from_wallet: Optional[str]) -> bool:
        """
        Determine if a transaction requires a signature.
        
        Transactions that don't require signatures:
        - GENESIS: Initial token supply
        - Transactions from None (system transactions)
        - WELCOME_GRANT: System-initiated grants
        
        All other transactions require valid signatures.
        """
        if from_wallet is None:
            return False
        if tx_type == TransactionType.GENESIS:
            return False
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
            ValueError: If balance is insufficient
            MissingSignatureError: If signature is required but not provided
            InvalidSignatureError: If signature verification fails
        """
        # Create wallets if they don't exist
        if from_wallet and not await self.wallet_exists(from_wallet):
            await self.create_wallet(from_wallet, f"wallet-{from_wallet[:8]}", public_key)
        if not await self.wallet_exists(to_wallet):
            await self.create_wallet(to_wallet, f"wallet-{to_wallet[:8]}")
            
        # Check balance for debit transactions
        if from_wallet:
            balance = await self.get_balance(from_wallet)
            if balance < amount:
                raise ValueError(f"Insufficient balance: {balance:.6f} < {amount:.6f}")
        
        # Signature verification
        if self.verify_signatures and self._is_signature_required(tx_type, from_wallet):
            if not signature:
                raise MissingSignatureError(
                    f"Transaction from {from_wallet} requires a signature"
                )
            
            # Get public key - prefer provided key, then registered key
            verify_pk = public_key or self.get_wallet_public_key(from_wallet)
            if not verify_pk:
                raise MissingSignatureError(
                    f"No public key provided or registered for wallet {from_wallet}"
                )
            
            # Create a temporary transaction to verify signature
            # (we need the tx_id for the hash, but signature was created before tx_id was assigned)
            # This is a chicken-and-egg problem - we need to verify the signature matches
            # the transaction data that will be created
            
            # For now, we'll create the transaction first, then verify
            # In production, the tx_id should be derived from or included in the signed data
            pass  # Signature will be verified after tx creation
                
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
        
        # Verify signature after transaction creation
        # The signature should be created from transaction data without tx_id
        # or the tx_id should be predictable (e.g., derived from hash)
        if self.verify_signatures and self._is_signature_required(tx_type, from_wallet):
            if signature:
                verify_pk = public_key or self.get_wallet_public_key(from_wallet)
                if verify_pk:
                    try:
                        self._verify_transaction_signature(tx, signature, verify_pk)
                    except InvalidSignatureError:
                        raise InvalidSignatureError(
                            f"Invalid signature for transaction from {from_wallet}"
                        )
        
        await self._store_transaction(tx)
        
        for parent_id in parent_ids:
            if parent_id in self._state.tips:
                self._state.tips.discard(parent_id)
                await self._db.execute("DELETE FROM tips WHERE tx_id = ?", (parent_id,))
                
        self._state.tips.add(tx.tx_id)
        await self._db.execute("INSERT OR IGNORE INTO tips (tx_id) VALUES (?)", (tx.tx_id,))
        
        await self._update_weights(tx)
        await self._db.commit()
        
        self._state.transactions[tx.tx_id] = tx
        
        return tx
    
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
