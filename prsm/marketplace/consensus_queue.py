"""Phase 7.1x.next: SQLite-backed queue for CONSENSUS_MISMATCH challenges.

The orchestrator's in-process `consensus_minority_queue` survives a
single process lifetime only. This module is the persistence layer that
makes the submitter → on-chain challenge pipeline crash-safe.

Lifecycle of a pending challenge:

                    enqueue_from_drained (orchestrator drain → rows)
                             │
                             ▼
                      ┌─────────────────┐
                      │     PENDING     │   minority_batch_id = NULL
                      └─────────────────┘   majority_batch_id = NULL
                             │
                    record_batch_commit (×2)
                             ▼
                      ┌─────────────────┐
                      │   SUBMITTABLE   │   both batch_ids set
                      └─────────────────┘
                             │
                     list_submittable →
                     submitter.submit_one
                             ▼
                 ┌─────────────────────────┐
                 ▼                         ▼
          mark_submitted            mark_failed
         (tx_hash known)         (terminal=True)
                 │                         │
                 ▼                         ▼
          ┌──────────┐              ┌──────────┐
          │ SUBMITTED│              │  FAILED  │
          └──────────┘              └──────────┘

Each drained orchestrator entry may contain multiple minority receipts
(k=5 with 3-2 split → 2 minorities); `enqueue_from_drained` expands one
entry into one row per minority receipt. Each row pairs its minority
with ONE majority receipt chosen deterministically (lowest provider_id)
so audit replay is reproducible.

Non-goals for MVP:
  - Concurrent writer coordination (assume one submitter process per DB).
  - Retry / backoff policy (caller loops + decides).
  - Cross-shard grouping (one row = one challenge).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from prsm.compute.shard_receipt import ShardExecutionReceipt

logger = logging.getLogger(__name__)


STATUS_PENDING = "pending"       # awaiting one or both batch_ids
STATUS_SUBMITTABLE = "submittable"  # both batch_ids set; ready to send
STATUS_SUBMITTED = "submitted"   # challenge tx landed successfully
STATUS_FAILED = "failed"         # terminal failure (reverted or abandoned)


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS consensus_challenges (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    shard_index INTEGER NOT NULL,
    agreed_output_hash TEXT NOT NULL,
    value_ftns_per_provider_wei INTEGER NOT NULL,

    minority_provider_id TEXT NOT NULL,
    minority_receipt_json TEXT NOT NULL,
    majority_provider_id TEXT NOT NULL,
    majority_receipt_json TEXT NOT NULL,

    -- Populated by record_batch_commit once each batch commits on-chain.
    minority_batch_id BLOB,    -- bytes32 or NULL
    majority_batch_id BLOB,    -- bytes32 or NULL

    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    last_tx_hash TEXT,

    -- Phase 7.1x.next+ §5.3 audit follow-up: claim-lease for multi-runner
    -- safety. claimed_by holds an opaque token unique per claim call
    -- (typically "<runner_id>:<uuid4>"). lease_expires_at is a float
    -- epoch-seconds timestamp after which the claim is considered
    -- abandoned and reclaimable. A single-runner deployment never
    -- populates these columns; they remain NULL.
    claimed_by TEXT,
    claimed_at REAL,
    lease_expires_at REAL,

    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_consensus_status
    ON consensus_challenges(status);
CREATE INDEX IF NOT EXISTS idx_consensus_minority_provider
    ON consensus_challenges(minority_provider_id, status);
CREATE INDEX IF NOT EXISTS idx_consensus_majority_provider
    ON consensus_challenges(majority_provider_id, status);
CREATE INDEX IF NOT EXISTS idx_consensus_job_shard
    ON consensus_challenges(job_id, shard_index, status);
-- Note: idx_consensus_claim (over the lease columns) is created by
-- _migrate_schema AFTER the ALTER TABLE, so pre-existing DBs without
-- the lease columns don't blow up here on a column reference.
"""


# Default lease duration for claim_submittable. 5 minutes is generous
# for a single submit_one + mark_* cycle (typical: ~15s including a
# few RPC retries); re-lease is cheap. Callers can override per call.
DEFAULT_LEASE_SECONDS = 300.0


@dataclass(frozen=True)
class PendingChallenge:
    """View of a row in the queue, decoded into Python types."""
    row_id: int
    job_id: str
    shard_index: int
    agreed_output_hash: str
    value_ftns_per_provider_wei: int
    minority_provider_id: str
    minority_receipt: ShardExecutionReceipt
    majority_provider_id: str
    majority_receipt: ShardExecutionReceipt
    minority_batch_id: Optional[bytes]
    majority_batch_id: Optional[bytes]
    status: str
    attempts: int
    last_error: Optional[str]
    last_tx_hash: Optional[str]
    # §5.3 lease fields (None on single-runner / unclaimed rows).
    claimed_by: Optional[str]
    claimed_at: Optional[float]
    lease_expires_at: Optional[float]
    created_at: float
    updated_at: float


class ConsensusChallengeQueue:
    """SQLite-backed queue for CONSENSUS_MISMATCH challenges.

    One process owns one queue. All operations are thread-safe within
    the process via a write lock; cross-process coordination is NOT
    provided (if you need it, open an issue — the current assumption
    is one submitter runner per DB path).
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        # `check_same_thread=False` so the submitter can drive the DB
        # from its runner thread while tests / orchestration touch it
        # from the main thread. The internal lock serializes writes.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA_SQL)
        self._migrate_schema()
        self._conn.commit()

    def _migrate_schema(self) -> None:
        """Apply schema changes to pre-existing DBs.

        SQLite's `ADD COLUMN` doesn't support `IF NOT EXISTS`; check the
        current schema first. Idempotent — safe to call on every open.

        Commits after each ALTER because Python's sqlite3 default
        isolation_level wraps DDL in an implicit transaction; without
        an explicit commit, subsequent queries on the same connection
        may not see the new columns (the implicit BEGIN blocks
        schema reload).
        """
        cur = self._conn.cursor()
        cur.execute("PRAGMA table_info(consensus_challenges)")
        existing_columns = {row[1] for row in cur.fetchall()}
        lease_columns = [
            ("claimed_by", "TEXT"),
            ("claimed_at", "REAL"),
            ("lease_expires_at", "REAL"),
        ]
        added_any = False
        for col_name, col_type in lease_columns:
            if col_name not in existing_columns:
                cur.execute(
                    f"ALTER TABLE consensus_challenges "
                    f"ADD COLUMN {col_name} {col_type}"
                )
                added_any = True
        # Also ensure the claim-index exists (CREATE IF NOT EXISTS in
        # _SCHEMA_SQL covers fresh DBs; pre-existing DBs miss this
        # index because they had an older CREATE TABLE).
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_consensus_claim
                ON consensus_challenges(status, claimed_by, lease_expires_at)
        """)
        if added_any:
            self._conn.commit()

    # ── Enqueue ─────────────────────────────────────────────────

    def enqueue_from_drained(
        self, drained_entries: List[Dict[str, Any]],
    ) -> List[int]:
        """Accept a list of dicts from
        `MarketplaceOrchestrator.drain_consensus_minority_queue()` and
        expand each into rows — one per minority receipt, paired with
        a deterministically-chosen majority receipt.

        Returns the row_ids of the newly-inserted rows in insertion
        order. Idempotent at the SQL level in the sense that calling
        twice inserts twice (the caller is responsible for only
        draining once); the orchestrator's drain is crash-atomic on
        its side (list.clear happens in memory only — any drained-
        but-not-enqueued entries are lost, which is the acceptable
        MVP failure mode).
        """
        row_ids: List[int] = []
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            for entry in drained_entries:
                majority = self._canonical_majority(entry["majority_receipts"])
                if majority is None:
                    # No majority receipt to pair with — can't build a
                    # challenge. Log and skip so the drain doesn't stall.
                    logger.warning(
                        f"enqueue skipped entry job={entry['job_id']!r} "
                        f"shard={entry['shard_index']}: no majority receipt"
                    )
                    continue
                for minority in entry["minority_receipts"]:
                    cur.execute(
                        """
                        INSERT INTO consensus_challenges (
                            job_id, shard_index, agreed_output_hash,
                            value_ftns_per_provider_wei,
                            minority_provider_id, minority_receipt_json,
                            majority_provider_id, majority_receipt_json,
                            status, attempts, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                        """,
                        (
                            entry["job_id"],
                            entry["shard_index"],
                            entry["agreed_output_hash"],
                            int(entry["value_ftns_per_provider_wei"]),
                            minority.provider_id,
                            json.dumps(minority.to_dict()),
                            majority.provider_id,
                            json.dumps(majority.to_dict()),
                            STATUS_PENDING,
                            now, now,
                        ),
                    )
                    row_ids.append(cur.lastrowid)
            self._conn.commit()
        return row_ids

    @staticmethod
    def _canonical_majority(
        majority_receipts: List[ShardExecutionReceipt],
    ) -> Optional[ShardExecutionReceipt]:
        """Pick one majority receipt to pair with each minority in this
        entry. Lowest provider_id wins — gives audit replay a stable
        target even when the majority set has multiple members."""
        if not majority_receipts:
            return None
        return min(majority_receipts, key=lambda r: r.provider_id)

    # ── Batch-commit coordination ───────────────────────────────

    def record_batch_commit(
        self,
        provider_id: str,
        job_id: str,
        shard_index: int,
        batch_id: bytes,
    ) -> int:
        """Record that `provider_id`'s batch for (job_id, shard_index)
        has committed on-chain with the given batch_id.

        Updates any PENDING row where this provider is the minority OR
        the majority. Transitions status to SUBMITTABLE once both
        batch_ids are set on a row.

        Returns the number of rows touched. Call from the settlement
        accumulator's post-commit hook with the minority provider's
        batch first (most likely to be the committing side), then the
        majority — or in any order; both branches are idempotent.
        """
        if len(batch_id) != 32:
            raise ValueError(f"batch_id must be 32 bytes (got {len(batch_id)})")
        touched = 0
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            # Update rows where this provider is the minority.
            cur.execute(
                """
                UPDATE consensus_challenges
                SET minority_batch_id = ?, updated_at = ?
                WHERE minority_provider_id = ?
                  AND job_id = ? AND shard_index = ?
                  AND status = ?
                  AND minority_batch_id IS NULL
                """,
                (batch_id, now, provider_id, job_id, shard_index,
                 STATUS_PENDING),
            )
            touched += cur.rowcount
            # Update rows where this provider is the majority.
            cur.execute(
                """
                UPDATE consensus_challenges
                SET majority_batch_id = ?, updated_at = ?
                WHERE majority_provider_id = ?
                  AND job_id = ? AND shard_index = ?
                  AND status = ?
                  AND majority_batch_id IS NULL
                """,
                (batch_id, now, provider_id, job_id, shard_index,
                 STATUS_PENDING),
            )
            touched += cur.rowcount
            # Promote any PENDING rows where both batch_ids are now set.
            cur.execute(
                """
                UPDATE consensus_challenges
                SET status = ?, updated_at = ?
                WHERE status = ?
                  AND minority_batch_id IS NOT NULL
                  AND majority_batch_id IS NOT NULL
                """,
                (STATUS_SUBMITTABLE, now, STATUS_PENDING),
            )
            self._conn.commit()
        return touched

    # ── Submitter-runner API ────────────────────────────────────

    def list_submittable(self, limit: int = 100) -> List[PendingChallenge]:
        """Return all rows with status=SUBMITTABLE, oldest first.
        The submitter runner calls this, fires `submit_one` per row,
        and updates status via mark_submitted / mark_failed.

        This is the LEGACY / single-runner entry point — it does NOT
        claim a lease. For multi-runner safety use `claim_submittable`."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT * FROM consensus_challenges
                WHERE status = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (STATUS_SUBMITTABLE, limit),
            )
            rows = cur.fetchall()
            col_names = [d[0] for d in cur.description]
        return [self._row_to_dataclass(dict(zip(col_names, r))) for r in rows]

    def claim_submittable(
        self,
        claimant_id: str,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
        limit: int = 100,
    ) -> List[PendingChallenge]:
        """Phase 7.1x.next+ §5.3: atomic claim of SUBMITTABLE rows for
        a specific runner, with a time-bounded lease.

        Two runners pointed at the same DB cannot both claim the same
        row — the UPDATE-with-subquery is one atomic SQL statement, and
        the WHERE clause excludes rows that already hold a live lease.
        Expired leases (lease_expires_at < now) are reclaimable, so a
        runner crash doesn't strand rows forever.

        claimant_id is a human-readable identifier (e.g., hostname,
        pod name). The stored `claimed_by` combines it with a per-call
        uuid4 so concurrent claims from the same claimant don't
        collide on row lookup.

        Caller contract: after running `submit_one` on a claimed row,
        call `mark_submitted` or `mark_failed` — both clear the lease.
        If the runner crashes before marking, the row auto-reclaims
        after lease_seconds.
        """
        claim_token = f"{claimant_id}:{uuid.uuid4().hex}"
        now = time.time()
        lease_expires = now + lease_seconds
        with self._lock:
            cur = self._conn.cursor()
            # Atomic claim: single UPDATE with a subquery picks rows
            # that are SUBMITTABLE AND either unclaimed OR have an
            # expired lease. No two-step read-then-write race is
            # possible because the whole operation is one SQL
            # statement under the lock + SQLite transactional bounds.
            cur.execute(
                """
                UPDATE consensus_challenges
                SET claimed_by = ?, claimed_at = ?,
                    lease_expires_at = ?, updated_at = ?
                WHERE row_id IN (
                    SELECT row_id FROM consensus_challenges
                    WHERE status = ?
                      AND (claimed_by IS NULL OR lease_expires_at < ?)
                    ORDER BY created_at ASC
                    LIMIT ?
                )
                """,
                (
                    claim_token, now, lease_expires, now,
                    STATUS_SUBMITTABLE, now, limit,
                ),
            )
            # Fetch the just-claimed rows by their unique claim_token.
            cur.execute(
                """
                SELECT * FROM consensus_challenges
                WHERE claimed_by = ?
                ORDER BY created_at ASC
                """,
                (claim_token,),
            )
            rows = cur.fetchall()
            col_names = [d[0] for d in cur.description]
            self._conn.commit()
        return [
            self._row_to_dataclass(dict(zip(col_names, r))) for r in rows
        ]

    def list_pending(self, limit: int = 1000) -> List[PendingChallenge]:
        """All non-terminal rows (PENDING + SUBMITTABLE). Useful for
        operator introspection / reconciliation."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT * FROM consensus_challenges
                WHERE status IN (?, ?)
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (STATUS_PENDING, STATUS_SUBMITTABLE, limit),
            )
            rows = cur.fetchall()
            col_names = [d[0] for d in cur.description]
        return [self._row_to_dataclass(dict(zip(col_names, r))) for r in rows]

    def mark_submitted(self, row_id: int, tx_hash_hex: str) -> None:
        """Terminal success: challenge tx landed. No retries from here.
        Clears any active claim-lease so the row doesn't keep a
        claimant reference after it's done."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                UPDATE consensus_challenges
                SET status = ?, last_tx_hash = ?, last_error = NULL,
                    attempts = attempts + 1, updated_at = ?,
                    claimed_by = NULL, claimed_at = NULL,
                    lease_expires_at = NULL
                WHERE row_id = ?
                """,
                (STATUS_SUBMITTED, tx_hash_hex, now, row_id),
            )
            self._conn.commit()

    def mark_failed(
        self,
        row_id: int,
        error_type: str,
        error_message: str,
        terminal: bool,
    ) -> None:
        """Record a failed submit attempt.

        `terminal=True` moves the row to FAILED (no more retries —
        e.g., contract revert, abandoned after N attempts). `terminal=
        False` keeps it SUBMITTABLE so the runner retries on the next
        tick (e.g., transient RPC failure).

        Both branches clear the claim-lease so a terminal row isn't
        stuck pointing at its last claimant, and a retryable row is
        immediately reclaimable by any runner (including this one)
        rather than waiting for the lease to expire."""
        now = time.time()
        new_status = STATUS_FAILED if terminal else STATUS_SUBMITTABLE
        err = f"{error_type}: {error_message}"
        with self._lock:
            self._conn.execute(
                """
                UPDATE consensus_challenges
                SET status = ?, last_error = ?,
                    attempts = attempts + 1, updated_at = ?,
                    claimed_by = NULL, claimed_at = NULL,
                    lease_expires_at = NULL
                WHERE row_id = ?
                """,
                (new_status, err, now, row_id),
            )
            self._conn.commit()

    # ── Introspection + cleanup ─────────────────────────────────

    def get(self, row_id: int) -> Optional[PendingChallenge]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT * FROM consensus_challenges WHERE row_id = ?",
                (row_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            col_names = [d[0] for d in cur.description]
        return self._row_to_dataclass(dict(zip(col_names, row)))

    def count_by_status(self) -> Dict[str, int]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT status, COUNT(*) FROM consensus_challenges "
                "GROUP BY status"
            )
            return {status: n for status, n in cur.fetchall()}

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── Internals ──────────────────────────────────────────────

    @staticmethod
    def _row_to_dataclass(row: Dict[str, Any]) -> PendingChallenge:
        return PendingChallenge(
            row_id=row["row_id"],
            job_id=row["job_id"],
            shard_index=row["shard_index"],
            agreed_output_hash=row["agreed_output_hash"],
            value_ftns_per_provider_wei=row["value_ftns_per_provider_wei"],
            minority_provider_id=row["minority_provider_id"],
            minority_receipt=ShardExecutionReceipt.from_dict(
                json.loads(row["minority_receipt_json"])
            ),
            majority_provider_id=row["majority_provider_id"],
            majority_receipt=ShardExecutionReceipt.from_dict(
                json.loads(row["majority_receipt_json"])
            ),
            minority_batch_id=row["minority_batch_id"],
            majority_batch_id=row["majority_batch_id"],
            status=row["status"],
            attempts=row["attempts"],
            last_error=row["last_error"],
            last_tx_hash=row["last_tx_hash"],
            claimed_by=row.get("claimed_by"),
            claimed_at=row.get("claimed_at"),
            lease_expires_at=row.get("lease_expires_at"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


__all__ = [
    "ConsensusChallengeQueue",
    "PendingChallenge",
    "DEFAULT_LEASE_SECONDS",
    "STATUS_PENDING",
    "STATUS_SUBMITTABLE",
    "STATUS_SUBMITTED",
    "STATUS_FAILED",
]
