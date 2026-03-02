"""
Local FTNS Ledger
=================

SQLite-backed ledger for FTNS token accounting.
Zero-config — no PostgreSQL required. Each node maintains its own ledger
which is reconciled via gossip when connected to the network.
"""

import json
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import aiosqlite


class TransactionType(str, Enum):
    WELCOME_GRANT = "welcome_grant"
    COMPUTE_PAYMENT = "compute_payment"
    COMPUTE_EARNING = "compute_earning"
    STORAGE_REWARD = "storage_reward"
    CONTENT_ROYALTY = "content_royalty"
    TRANSFER = "transfer"


@dataclass
class Transaction:
    tx_id: str
    tx_type: TransactionType
    from_wallet: Optional[str]     # None for grants/rewards
    to_wallet: str
    amount: float
    description: str
    timestamp: float
    signature: Optional[str] = None


class LocalLedger:
    """SQLite-backed FTNS ledger for a single node.

    The ledger is append-only for auditability. Balances are derived from
    the transaction log.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Open database and create tables."""
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._create_tables()

    async def _create_tables(self) -> None:
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS wallets (
                wallet_id   TEXT PRIMARY KEY,
                display_name TEXT NOT NULL DEFAULT '',
                created_at  REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS transactions (
                tx_id       TEXT PRIMARY KEY,
                tx_type     TEXT NOT NULL,
                from_wallet TEXT,
                to_wallet   TEXT NOT NULL,
                amount      REAL NOT NULL CHECK(amount > 0),
                description TEXT NOT NULL DEFAULT '',
                timestamp   REAL NOT NULL,
                signature   TEXT,
                FOREIGN KEY (to_wallet) REFERENCES wallets(wallet_id)
            );

            CREATE INDEX IF NOT EXISTS idx_tx_to ON transactions(to_wallet);
            CREATE INDEX IF NOT EXISTS idx_tx_from ON transactions(from_wallet);
            CREATE INDEX IF NOT EXISTS idx_tx_time ON transactions(timestamp);

            CREATE TABLE IF NOT EXISTS seen_nonces (
                nonce       TEXT PRIMARY KEY,
                origin      TEXT NOT NULL,
                seen_at     REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agent_allowances (
                agent_id        TEXT PRIMARY KEY,
                principal_id    TEXT NOT NULL,
                allowance       REAL NOT NULL CHECK(allowance >= 0),
                spent           REAL NOT NULL DEFAULT 0 CHECK(spent >= 0),
                epoch_hours     REAL NOT NULL DEFAULT 24,
                epoch_start     REAL NOT NULL,
                created_at      REAL NOT NULL,
                revoked         INTEGER NOT NULL DEFAULT 0
            );

            -- Gossip log for persistence and catch-up
            CREATE TABLE IF NOT EXISTS gossip_log (
                nonce       TEXT PRIMARY KEY,
                subtype     TEXT NOT NULL,
                origin      TEXT NOT NULL,
                payload     TEXT NOT NULL,
                ttl         INTEGER NOT NULL DEFAULT 5,
                received_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_gossip_log_time ON gossip_log(received_at);
            CREATE INDEX IF NOT EXISTS idx_gossip_log_subtype ON gossip_log(subtype);

            -- Collaboration persistence tables
            CREATE TABLE IF NOT EXISTS collab_tasks (
                task_id                 TEXT PRIMARY KEY,
                requester_agent_id      TEXT NOT NULL,
                requester_node_id       TEXT NOT NULL,
                title                   TEXT NOT NULL DEFAULT '',
                description             TEXT NOT NULL DEFAULT '',
                required_capabilities   TEXT NOT NULL DEFAULT '[]',
                ftns_budget             REAL NOT NULL DEFAULT 0,
                deadline_seconds        REAL NOT NULL DEFAULT 3600,
                status                  TEXT NOT NULL DEFAULT 'open',
                assigned_agent_id       TEXT,
                bids                    TEXT NOT NULL DEFAULT '[]',
                result                  TEXT,
                created_at              REAL NOT NULL,
                escrow_tx_id            TEXT
            );

            CREATE TABLE IF NOT EXISTS collab_reviews (
                review_id               TEXT PRIMARY KEY,
                submitter_agent_id      TEXT NOT NULL,
                submitter_node_id       TEXT NOT NULL,
                content_cid             TEXT NOT NULL DEFAULT '',
                description             TEXT NOT NULL DEFAULT '',
                required_capabilities   TEXT NOT NULL DEFAULT '[]',
                ftns_per_review         REAL NOT NULL DEFAULT 0.1,
                max_reviewers           INTEGER NOT NULL DEFAULT 3,
                status                  TEXT NOT NULL DEFAULT 'pending',
                reviews                 TEXT NOT NULL DEFAULT '[]',
                created_at              REAL NOT NULL,
                escrow_tx_id            TEXT,
                paid_reviewers          TEXT NOT NULL DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS collab_queries (
                query_id                TEXT PRIMARY KEY,
                requester_agent_id      TEXT NOT NULL,
                requester_node_id       TEXT NOT NULL,
                topic                   TEXT NOT NULL DEFAULT '',
                question                TEXT NOT NULL DEFAULT '',
                ftns_per_response       REAL NOT NULL DEFAULT 0.05,
                max_responses           INTEGER NOT NULL DEFAULT 5,
                responses               TEXT NOT NULL DEFAULT '[]',
                created_at              REAL NOT NULL,
                escrow_tx_id            TEXT,
                paid_responders         TEXT NOT NULL DEFAULT '[]'
            );
        """)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # ── Wallet management ────────────────────────────────────────

    async def create_wallet(self, wallet_id: str, display_name: str = "") -> None:
        """Create a wallet if it doesn't already exist."""
        await self._db.execute(
            "INSERT OR IGNORE INTO wallets (wallet_id, display_name, created_at) VALUES (?, ?, ?)",
            (wallet_id, display_name, time.time()),
        )
        await self._db.commit()

    async def wallet_exists(self, wallet_id: str) -> bool:
        cursor = await self._db.execute(
            "SELECT 1 FROM wallets WHERE wallet_id = ?", (wallet_id,)
        )
        return await cursor.fetchone() is not None

    # ── Balance ──────────────────────────────────────────────────

    async def get_balance(self, wallet_id: str) -> float:
        """Derive balance from transaction log."""
        cursor = await self._db.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE to_wallet = ?",
            (wallet_id,),
        )
        credits = (await cursor.fetchone())[0]

        cursor = await self._db.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE from_wallet = ?",
            (wallet_id,),
        )
        debits = (await cursor.fetchone())[0]
        return round(credits - debits, 6)

    # ── Transactions ─────────────────────────────────────────────

    async def _ensure_wallet(self, wallet_id: str) -> None:
        """Create wallet if it doesn't exist (for remote node wallets)."""
        if not await self.wallet_exists(wallet_id):
            await self.create_wallet(wallet_id, f"remote-{wallet_id[:8]}")

    async def credit(
        self,
        wallet_id: str,
        amount: float,
        tx_type: TransactionType,
        description: str = "",
        signature: Optional[str] = None,
    ) -> Transaction:
        """Credit FTNS to a wallet (from_wallet is None for system grants)."""
        await self._ensure_wallet(wallet_id)
        tx = Transaction(
            tx_id=str(uuid.uuid4()),
            tx_type=tx_type,
            from_wallet=None,
            to_wallet=wallet_id,
            amount=amount,
            description=description,
            timestamp=time.time(),
            signature=signature,
        )
        await self._insert_tx(tx)
        return tx

    async def debit(
        self,
        wallet_id: str,
        amount: float,
        tx_type: TransactionType,
        description: str = "",
    ) -> Transaction:
        """Debit FTNS from a wallet (payment to system/network)."""
        balance = await self.get_balance(wallet_id)
        if balance < amount:
            raise ValueError(
                f"Insufficient balance: {balance:.6f} < {amount:.6f}"
            )
        # Record as transfer from wallet to a system sink
        tx = Transaction(
            tx_id=str(uuid.uuid4()),
            tx_type=tx_type,
            from_wallet=wallet_id,
            to_wallet="system",
            amount=amount,
            description=description,
            timestamp=time.time(),
        )
        await self._insert_tx(tx)
        return tx

    async def transfer(
        self,
        from_wallet: str,
        to_wallet: str,
        amount: float,
        tx_type: TransactionType = TransactionType.TRANSFER,
        description: str = "",
        signature: Optional[str] = None,
    ) -> Transaction:
        """Transfer FTNS between wallets."""
        await self._ensure_wallet(to_wallet)
        balance = await self.get_balance(from_wallet)
        if balance < amount:
            raise ValueError(
                f"Insufficient balance: {balance:.6f} < {amount:.6f}"
            )
        tx = Transaction(
            tx_id=str(uuid.uuid4()),
            tx_type=tx_type,
            from_wallet=from_wallet,
            to_wallet=to_wallet,
            amount=amount,
            description=description,
            timestamp=time.time(),
            signature=signature,
        )
        await self._insert_tx(tx)
        return tx

    async def get_transaction_history(
        self, wallet_id: str, limit: int = 50
    ) -> List[Transaction]:
        """Get recent transactions involving a wallet."""
        cursor = await self._db.execute(
            """SELECT tx_id, tx_type, from_wallet, to_wallet, amount,
                      description, timestamp, signature
               FROM transactions
               WHERE to_wallet = ? OR from_wallet = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (wallet_id, wallet_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            Transaction(
                tx_id=r[0],
                tx_type=TransactionType(r[1]),
                from_wallet=r[2],
                to_wallet=r[3],
                amount=r[4],
                description=r[5],
                timestamp=r[6],
                signature=r[7],
            )
            for r in rows
        ]

    async def get_transaction_count(self, wallet_id: str) -> int:
        """Count total transactions for a wallet."""
        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM transactions WHERE to_wallet = ? OR from_wallet = ?",
            (wallet_id, wallet_id),
        )
        return (await cursor.fetchone())[0]

    # ── Nonce tracking (double-spend prevention) ────────────────

    async def has_seen_nonce(self, nonce: str) -> bool:
        """Check if a transaction nonce has already been processed."""
        cursor = await self._db.execute(
            "SELECT 1 FROM seen_nonces WHERE nonce = ?", (nonce,)
        )
        return await cursor.fetchone() is not None

    async def record_nonce(self, nonce: str, origin: str) -> None:
        """Record a transaction nonce to prevent replay."""
        await self._db.execute(
            "INSERT OR IGNORE INTO seen_nonces (nonce, origin, seen_at) VALUES (?, ?, ?)",
            (nonce, origin, time.time()),
        )
        await self._db.commit()

    async def get_recent_tx_ids(self, wallet_id: str, limit: int = 50) -> List[str]:
        """Get recent transaction IDs for reconciliation."""
        cursor = await self._db.execute(
            """SELECT tx_id FROM transactions
               WHERE to_wallet = ? OR from_wallet = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (wallet_id, wallet_id, limit),
        )
        rows = await cursor.fetchall()
        return [r[0] for r in rows]

    async def has_transaction(self, tx_id: str) -> bool:
        """Check if a transaction already exists in the ledger."""
        cursor = await self._db.execute(
            "SELECT 1 FROM transactions WHERE tx_id = ?", (tx_id,)
        )
        return await cursor.fetchone() is not None

    # ── Agent allowances (delegated payments) ───────────────────

    async def grant_agent_allowance(
        self,
        principal_id: str,
        agent_id: str,
        amount: float,
        epoch_hours: float = 24.0,
    ) -> None:
        """Grant an agent a spending allowance from the principal's wallet.

        The allowance resets at each epoch boundary (epoch_hours interval).
        """
        now = time.time()
        await self._db.execute(
            """INSERT INTO agent_allowances
               (agent_id, principal_id, allowance, spent, epoch_hours, epoch_start, created_at, revoked)
               VALUES (?, ?, ?, 0, ?, ?, ?, 0)
               ON CONFLICT(agent_id) DO UPDATE SET
                   allowance = excluded.allowance,
                   spent = 0,
                   epoch_hours = excluded.epoch_hours,
                   epoch_start = excluded.epoch_start,
                   revoked = 0""",
            (agent_id, principal_id, amount, epoch_hours, now, now),
        )
        await self._db.commit()

    async def get_agent_allowance(self, agent_id: str) -> Optional[dict]:
        """Get an agent's current allowance state. Returns None if not found."""
        cursor = await self._db.execute(
            "SELECT principal_id, allowance, spent, epoch_hours, epoch_start, revoked "
            "FROM agent_allowances WHERE agent_id = ?",
            (agent_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        principal_id, allowance, spent, epoch_hours, epoch_start, revoked = row

        # Check if epoch has rolled over — if so, reset spent
        now = time.time()
        epoch_seconds = epoch_hours * 3600
        if now - epoch_start >= epoch_seconds:
            spent = 0.0
            epoch_start = now
            await self._db.execute(
                "UPDATE agent_allowances SET spent = 0, epoch_start = ? WHERE agent_id = ?",
                (now, agent_id),
            )
            await self._db.commit()

        return {
            "agent_id": agent_id,
            "principal_id": principal_id,
            "allowance": allowance,
            "spent": spent,
            "remaining": max(0.0, allowance - spent),
            "epoch_hours": epoch_hours,
            "epoch_start": epoch_start,
            "revoked": bool(revoked),
        }

    async def agent_debit(
        self,
        agent_id: str,
        amount: float,
        tx_type: TransactionType,
        description: str = "",
    ) -> Optional[Transaction]:
        """Debit from the principal's wallet on behalf of an agent.

        Checks the agent's remaining allowance and the principal's balance.
        Returns the transaction, or None if rejected.
        """
        allowance = await self.get_agent_allowance(agent_id)
        if not allowance or allowance["revoked"]:
            return None

        if amount > allowance["remaining"]:
            return None  # Exceeds allowance

        principal_id = allowance["principal_id"]
        balance = await self.get_balance(principal_id)
        if balance < amount:
            return None  # Insufficient principal balance

        # Debit from principal
        tx = await self.debit(
            wallet_id=principal_id,
            amount=amount,
            tx_type=tx_type,
            description=f"[agent:{agent_id[:12]}] {description}",
        )

        # Update spent
        await self._db.execute(
            "UPDATE agent_allowances SET spent = spent + ? WHERE agent_id = ?",
            (amount, agent_id),
        )
        await self._db.commit()
        return tx

    async def revoke_agent_allowance(self, principal_id: str, agent_id: str) -> bool:
        """Revoke an agent's spending authority. Returns True if found."""
        cursor = await self._db.execute(
            "UPDATE agent_allowances SET revoked = 1 "
            "WHERE agent_id = ? AND principal_id = ?",
            (agent_id, principal_id),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    # ── Gossip Log ────────────────────────────────────────────────

    async def log_gossip(
        self,
        nonce: str,
        subtype: str,
        origin: str,
        payload: Dict[str, Any],
        ttl: int = 5,
    ) -> None:
        """Persist a gossip message for catch-up replay."""
        await self._db.execute(
            """INSERT OR IGNORE INTO gossip_log
               (nonce, subtype, origin, payload, ttl, received_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (nonce, subtype, origin, json.dumps(payload), ttl, time.time()),
        )
        await self._db.commit()

    async def get_recent_gossip(
        self,
        since: float,
        subtypes: Optional[List[str]] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Retrieve gossip messages received after *since* timestamp."""
        if subtypes:
            placeholders = ",".join("?" for _ in subtypes)
            cursor = await self._db.execute(
                f"""SELECT nonce, subtype, origin, payload, ttl, received_at
                    FROM gossip_log
                    WHERE received_at > ? AND subtype IN ({placeholders})
                    ORDER BY received_at ASC LIMIT ?""",
                (since, *subtypes, limit),
            )
        else:
            cursor = await self._db.execute(
                """SELECT nonce, subtype, origin, payload, ttl, received_at
                   FROM gossip_log
                   WHERE received_at > ?
                   ORDER BY received_at ASC LIMIT ?""",
                (since, limit),
            )
        rows = await cursor.fetchall()
        return [
            {
                "nonce": r[0],
                "subtype": r[1],
                "origin": r[2],
                "payload": json.loads(r[3]),
                "ttl": r[4],
                "received_at": r[5],
            }
            for r in rows
        ]

    async def prune_gossip_log(self, max_age: float) -> int:
        """Delete gossip entries older than *max_age* seconds. Returns count deleted."""
        cutoff = time.time() - max_age
        cursor = await self._db.execute(
            "DELETE FROM gossip_log WHERE received_at < ?", (cutoff,)
        )
        await self._db.commit()
        return cursor.rowcount

    # ── Collaboration Persistence ────────────────────────────────

    async def save_task(self, data: Dict[str, Any]) -> None:
        """Upsert a collaboration task record."""
        await self._db.execute(
            """INSERT OR REPLACE INTO collab_tasks
               (task_id, requester_agent_id, requester_node_id, title, description,
                required_capabilities, ftns_budget, deadline_seconds, status,
                assigned_agent_id, bids, result, created_at, escrow_tx_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data["task_id"],
                data["requester_agent_id"],
                data["requester_node_id"],
                data.get("title", ""),
                data.get("description", ""),
                json.dumps(data.get("required_capabilities", [])),
                data.get("ftns_budget", 0),
                data.get("deadline_seconds", 3600),
                data.get("status", "open"),
                data.get("assigned_agent_id"),
                json.dumps(data.get("bids", [])),
                json.dumps(data.get("result")) if data.get("result") else None,
                data["created_at"],
                data.get("escrow_tx_id"),
            ),
        )
        await self._db.commit()

    async def save_review(self, data: Dict[str, Any]) -> None:
        """Upsert a collaboration review record."""
        await self._db.execute(
            """INSERT OR REPLACE INTO collab_reviews
               (review_id, submitter_agent_id, submitter_node_id, content_cid,
                description, required_capabilities, ftns_per_review, max_reviewers,
                status, reviews, created_at, escrow_tx_id, paid_reviewers)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data["review_id"],
                data["submitter_agent_id"],
                data["submitter_node_id"],
                data.get("content_cid", ""),
                data.get("description", ""),
                json.dumps(data.get("required_capabilities", [])),
                data.get("ftns_per_review", 0.1),
                data.get("max_reviewers", 3),
                data.get("status", "pending"),
                json.dumps(data.get("reviews", [])),
                data["created_at"],
                data.get("escrow_tx_id"),
                json.dumps(data.get("paid_reviewers", [])),
            ),
        )
        await self._db.commit()

    async def save_query(self, data: Dict[str, Any]) -> None:
        """Upsert a collaboration query record."""
        await self._db.execute(
            """INSERT OR REPLACE INTO collab_queries
               (query_id, requester_agent_id, requester_node_id, topic, question,
                ftns_per_response, max_responses, responses, created_at,
                escrow_tx_id, paid_responders)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data["query_id"],
                data["requester_agent_id"],
                data["requester_node_id"],
                data.get("topic", ""),
                data.get("question", ""),
                data.get("ftns_per_response", 0.05),
                data.get("max_responses", 5),
                json.dumps(data.get("responses", [])),
                data["created_at"],
                data.get("escrow_tx_id"),
                json.dumps(data.get("paid_responders", [])),
            ),
        )
        await self._db.commit()

    async def load_active_tasks(self) -> List[Dict[str, Any]]:
        """Load non-terminal collaboration tasks."""
        cursor = await self._db.execute(
            """SELECT task_id, requester_agent_id, requester_node_id, title,
                      description, required_capabilities, ftns_budget,
                      deadline_seconds, status, assigned_agent_id, bids,
                      result, created_at, escrow_tx_id
               FROM collab_tasks WHERE status NOT IN ('completed', 'cancelled')"""
        )
        rows = await cursor.fetchall()
        return [
            {
                "task_id": r[0], "requester_agent_id": r[1],
                "requester_node_id": r[2], "title": r[3],
                "description": r[4],
                "required_capabilities": json.loads(r[5]),
                "ftns_budget": r[6], "deadline_seconds": r[7],
                "status": r[8], "assigned_agent_id": r[9],
                "bids": json.loads(r[10]),
                "result": json.loads(r[11]) if r[11] else None,
                "created_at": r[12], "escrow_tx_id": r[13],
            }
            for r in rows
        ]

    async def load_active_reviews(self) -> List[Dict[str, Any]]:
        """Load non-terminal collaboration reviews."""
        cursor = await self._db.execute(
            """SELECT review_id, submitter_agent_id, submitter_node_id,
                      content_cid, description, required_capabilities,
                      ftns_per_review, max_reviewers, status, reviews,
                      created_at, escrow_tx_id, paid_reviewers
               FROM collab_reviews WHERE status NOT IN ('accepted', 'rejected')"""
        )
        rows = await cursor.fetchall()
        return [
            {
                "review_id": r[0], "submitter_agent_id": r[1],
                "submitter_node_id": r[2], "content_cid": r[3],
                "description": r[4],
                "required_capabilities": json.loads(r[5]),
                "ftns_per_review": r[6], "max_reviewers": r[7],
                "status": r[8], "reviews": json.loads(r[9]),
                "created_at": r[10], "escrow_tx_id": r[11],
                "paid_reviewers": json.loads(r[12]),
            }
            for r in rows
        ]

    async def load_active_queries(self) -> List[Dict[str, Any]]:
        """Load active collaboration queries (not yet at max responses)."""
        cursor = await self._db.execute(
            """SELECT query_id, requester_agent_id, requester_node_id,
                      topic, question, ftns_per_response, max_responses,
                      responses, created_at, escrow_tx_id, paid_responders
               FROM collab_queries"""
        )
        rows = await cursor.fetchall()
        results = []
        for r in rows:
            responses = json.loads(r[7])
            max_responses = r[6]
            if len(responses) < max_responses:
                results.append({
                    "query_id": r[0], "requester_agent_id": r[1],
                    "requester_node_id": r[2], "topic": r[3],
                    "question": r[4], "ftns_per_response": r[5],
                    "max_responses": max_responses,
                    "responses": responses,
                    "created_at": r[8], "escrow_tx_id": r[9],
                    "paid_responders": json.loads(r[10]),
                })
        return results

    async def delete_collab_record(self, table: str, id_column: str, record_id: str) -> None:
        """Delete a completed/archived collaboration record from SQLite."""
        allowed = {"collab_tasks": "task_id", "collab_reviews": "review_id", "collab_queries": "query_id"}
        if table not in allowed or allowed[table] != id_column:
            return
        await self._db.execute(f"DELETE FROM {table} WHERE {id_column} = ?", (record_id,))
        await self._db.commit()

    # ── Internal ─────────────────────────────────────────────────

    async def _insert_tx(self, tx: Transaction) -> None:
        await self._db.execute(
            """INSERT INTO transactions
               (tx_id, tx_type, from_wallet, to_wallet, amount, description, timestamp, signature)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tx.tx_id,
                tx.tx_type.value,
                tx.from_wallet,
                tx.to_wallet,
                tx.amount,
                tx.description,
                tx.timestamp,
                tx.signature,
            ),
        )
        await self._db.commit()

    async def issue_welcome_grant(self, wallet_id: str, amount: float = 100.0) -> Transaction:
        """Issue a one-time welcome grant to a new node."""
        # Check if this wallet already received a welcome grant
        cursor = await self._db.execute(
            "SELECT 1 FROM transactions WHERE to_wallet = ? AND tx_type = ?",
            (wallet_id, TransactionType.WELCOME_GRANT.value),
        )
        if await cursor.fetchone() is not None:
            raise ValueError(f"Wallet {wallet_id} already received a welcome grant")

        return await self.credit(
            wallet_id=wallet_id,
            amount=amount,
            tx_type=TransactionType.WELCOME_GRANT,
            description=f"Welcome grant of {amount} FTNS for new node",
        )
