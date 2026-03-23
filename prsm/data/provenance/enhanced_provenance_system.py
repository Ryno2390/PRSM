#!/usr/bin/env python3
"""
Enhanced Provenance System

Provides comprehensive tracking and verification of data lineage,
reasoning chains, and computational provenance throughout the PRSM ecosystem.

Core Functions:
- Reasoning chain provenance tracking
- Data source verification and lineage
- Computational reproducibility assurance
- Trust and credibility scoring
"""

import structlog
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4, UUID
from abc import ABC, abstractmethod
from pathlib import Path
import asyncio
import hashlib
import json
import gzip
import pickle

logger = structlog.get_logger(__name__)


class ProvenanceType(Enum):
    """Types of provenance records"""
    DATA_SOURCE = "data_source"
    REASONING_STEP = "reasoning_step"
    MODEL_PREDICTION = "model_prediction"
    COMPUTATION = "computation"
    TRANSFORMATION = "transformation"
    CITATION = "citation"
    VALIDATION = "validation"


class TrustLevel(Enum):
    """Trust levels for provenance records"""
    VERIFIED = "verified"          # Fully verified and trusted
    TRUSTED = "trusted"            # High confidence, peer reviewed
    CREDIBLE = "credible"          # Good confidence, some verification
    PROVISIONAL = "provisional"    # Limited verification
    UNCERTAIN = "uncertain"        # Low confidence, needs verification


@dataclass
class ProvenanceRecord:
    """Individual provenance record"""
    record_id: str
    provenance_type: ProvenanceType
    timestamp: datetime
    source_entity: str  # What created this record
    target_entity: str  # What this record describes
    operation: str      # What operation was performed
    inputs: List[str]   # Input record IDs
    outputs: List[str]  # Output record IDs
    metadata: Dict[str, Any]
    trust_level: TrustLevel = TrustLevel.CREDIBLE
    verification_status: bool = False
    content_hash: str = ""


@dataclass
class ReasoningProvenance:
    """Provenance chain for reasoning operations"""
    reasoning_id: str
    query: str
    reasoning_steps: List[ProvenanceRecord]
    data_sources: List[ProvenanceRecord]
    model_calls: List[ProvenanceRecord]
    final_conclusion: str
    confidence_score: float
    total_steps: int
    verification_chain: List[str] = field(default_factory=list)


# =============================================================================
# Persistence Backend Classes
# =============================================================================

class ProvenancePersistenceBackend(ABC):
    """
    Abstract base class for provenance persistence backends.

    Implementations can use PostgreSQL, SQLite, or other storage systems.
    """

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the backend (create tables, verify connection)."""
        pass

    @abstractmethod
    async def save_record(self, record: ProvenanceRecord) -> bool:
        """Save a provenance record to persistent storage."""
        pass

    @abstractmethod
    async def load_record(self, record_id: str) -> Optional[ProvenanceRecord]:
        """Load a provenance record by ID."""
        pass

    @abstractmethod
    async def list_records(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ProvenanceRecord]:
        """List provenance records with optional filters."""
        pass

    @abstractmethod
    async def delete_record(self, record_id: str) -> bool:
        """Delete (soft delete) a provenance record."""
        pass

    @abstractmethod
    async def save_reasoning_chain(self, chain: ReasoningProvenance) -> bool:
        """Save a reasoning chain to persistent storage."""
        pass

    @abstractmethod
    async def load_reasoning_chain(self, chain_id: str) -> Optional[ReasoningProvenance]:
        """Load a reasoning chain by ID."""
        pass

    @abstractmethod
    async def list_chains(
        self,
        node_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ReasoningProvenance]:
        """List reasoning chains, optionally filtered by node."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close database connections."""
        pass


class PostgreSQLProvenanceBackend(ProvenancePersistenceBackend):
    """
    PostgreSQL backend for provenance persistence.

    Uses the existing PRSM database connection pool.
    """

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url
        self._engine = None
        self._session_factory = None

    async def initialize(self) -> bool:
        """Initialize PostgreSQL connection."""
        try:
            from prsm.core.database import get_async_engine, async_sessionmaker

            self._engine = await get_async_engine()
            if self._engine is None:
                logger.error("Failed to get database engine")
                return False

            # Create session factory
            from sqlalchemy.ext.asyncio import async_sessionmaker
            self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

            logger.info("PostgreSQL provenance backend initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL backend: {e}")
            return False

    async def save_record(self, record: ProvenanceRecord) -> bool:
        """Save a provenance record to PostgreSQL."""
        if not self._session_factory:
            return False

        try:
            from sqlalchemy import text

            async with self._session_factory() as session:
                # Check if record exists
                existing = await session.execute(
                    text("SELECT record_id FROM provenance_records WHERE record_id = :record_id"),
                    {"record_id": record.record_id}
                )
                if existing.fetchone():
                    # Update existing record
                    await session.execute(
                        text("""
                            UPDATE provenance_records SET
                                provenance_type = :provenance_type,
                                source_entity = :source_entity,
                                target_entity = :target_entity,
                                operation = :operation,
                                inputs = :inputs,
                                outputs = :outputs,
                                metadata = :metadata,
                                trust_level = :trust_level,
                                verification_status = :verification_status,
                                content_hash = :content_hash,
                                updated_at = NOW()
                            WHERE record_id = :record_id
                        """),
                        self._record_to_db(record)
                    )
                else:
                    # Insert new record
                    await session.execute(
                        text("""
                            INSERT INTO provenance_records (
                                record_id, provenance_type, source_entity, target_entity,
                                operation, inputs, outputs, metadata, trust_level,
                                verification_status, content_hash, timestamp, active
                            ) VALUES (
                                :record_id, :provenance_type, :source_entity, :target_entity,
                                :operation, :inputs, :outputs, :metadata, :trust_level,
                                :verification_status, :content_hash, :timestamp, TRUE
                            )
                        """),
                        self._record_to_db(record)
                    )
                await session.commit()

            logger.debug("Saved provenance record", record_id=record.record_id)
            return True
        except Exception as e:
            logger.error(f"Failed to save provenance record: {e}")
            return False

    async def load_record(self, record_id: str) -> Optional[ProvenanceRecord]:
        """Load a provenance record by ID."""
        if not self._session_factory:
            return None

        try:
            from sqlalchemy import text

            async with self._session_factory() as session:
                result = await session.execute(
                    text("SELECT * FROM provenance_records WHERE record_id = :record_id AND active = TRUE"),
                    {"record_id": record_id}
                )
                row = result.fetchone()
                if row:
                    return self._db_to_record(row)
                return None
        except Exception as e:
            logger.error(f"Failed to load provenance record: {e}")
            return None

    async def list_records(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ProvenanceRecord]:
        """List provenance records with optional filters."""
        if not self._session_factory:
            return []

        try:
            from sqlalchemy import text

            query = "SELECT * FROM provenance_records WHERE active = TRUE"
            params = {"limit": limit, "offset": offset}

            if filters:
                if "node_id" in filters:
                    query += " AND source_entity = :node_id"
                    params["node_id"] = filters["node_id"]
                if "trust_level" in filters:
                    query += " AND trust_level = :trust_level"
                    params["trust_level"] = filters["trust_level"]
                if "provenance_type" in filters:
                    query += " AND provenance_type = :provenance_type"
                    params["provenance_type"] = filters["provenance_type"]

            query += " ORDER BY timestamp DESC LIMIT :limit OFFSET :offset"

            async with self._session_factory() as session:
                result = await session.execute(text(query), params)
                rows = result.fetchall()
                return [self._db_to_record(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to list provenance records: {e}")
            return []

    async def delete_record(self, record_id: str) -> bool:
        """Soft delete a provenance record."""
        if not self._session_factory:
            return False

        try:
            from sqlalchemy import text

            async with self._session_factory() as session:
                await session.execute(
                    text("UPDATE provenance_records SET active = FALSE WHERE record_id = :record_id"),
                    {"record_id": record_id}
                )
                await session.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to delete provenance record: {e}")
            return False

    async def save_reasoning_chain(self, chain: ReasoningProvenance) -> bool:
        """Save a reasoning chain to PostgreSQL."""
        if not self._session_factory:
            return False

        try:
            from sqlalchemy import text

            async with self._session_factory() as session:
                # Get node_id from first step if available
                node_id = chain.reasoning_steps[0].source_entity if chain.reasoning_steps else "unknown"

                # Check if chain exists
                existing = await session.execute(
                    text("SELECT chain_id FROM reasoning_chains WHERE chain_id = :chain_id"),
                    {"chain_id": chain.reasoning_id}
                )

                chain_data = {
                    "chain_id": chain.reasoning_id,
                    "node_id": node_id,
                    "query": chain.query,
                    "final_conclusion": chain.final_conclusion,
                    "confidence_score": chain.confidence_score,
                    "total_steps": chain.total_steps,
                    "step_count": len(chain.reasoning_steps),
                    "finalized": bool(chain.final_conclusion),
                    "metadata": json.dumps({
                        "data_sources_count": len(chain.data_sources),
                        "model_calls_count": len(chain.model_calls),
                        "verification_chain": chain.verification_chain,
                    }),
                }

                if existing.fetchone():
                    await session.execute(
                        text("""
                            UPDATE reasoning_chains SET
                                node_id = :node_id,
                                query = :query,
                                final_conclusion = :final_conclusion,
                                confidence_score = :confidence_score,
                                total_steps = :total_steps,
                                step_count = :step_count,
                                finalized = :finalized,
                                metadata = :metadata,
                                finalized_at = CASE WHEN :finalized THEN NOW() ELSE finalized_at END
                            WHERE chain_id = :chain_id
                        """),
                        chain_data
                    )
                else:
                    await session.execute(
                        text("""
                            INSERT INTO reasoning_chains (
                                chain_id, node_id, query, final_conclusion,
                                confidence_score, total_steps, step_count, finalized, metadata
                            ) VALUES (
                                :chain_id, :node_id, :query, :final_conclusion,
                                :confidence_score, :total_steps, :step_count, :finalized, :metadata
                            )
                        """),
                        chain_data
                    )
                await session.commit()

            logger.debug("Saved reasoning chain", chain_id=chain.reasoning_id)
            return True
        except Exception as e:
            logger.error(f"Failed to save reasoning chain: {e}")
            return False

    async def load_reasoning_chain(self, chain_id: str) -> Optional[ReasoningProvenance]:
        """Load a reasoning chain by ID."""
        if not self._session_factory:
            return None

        try:
            from sqlalchemy import text

            async with self._session_factory() as session:
                result = await session.execute(
                    text("SELECT * FROM reasoning_chains WHERE chain_id = :chain_id"),
                    {"chain_id": chain_id}
                )
                row = result.fetchone()
                if not row:
                    return None

                # Load related records
                records_result = await session.execute(
                    text("""
                        SELECT * FROM provenance_records
                        WHERE target_entity = :chain_id AND active = TRUE
                        ORDER BY timestamp
                    """),
                    {"chain_id": chain_id}
                )
                records = [self._db_to_record(r) for r in records_result.fetchall()]

                metadata = json.loads(row.metadata) if hasattr(row, 'metadata') and row.metadata else {}

                return ReasoningProvenance(
                    reasoning_id=row.chain_id,
                    query=row.query,
                    reasoning_steps=records,
                    data_sources=[r for r in records if r.provenance_type == ProvenanceType.DATA_SOURCE],
                    model_calls=[r for r in records if r.provenance_type == ProvenanceType.MODEL_PREDICTION],
                    final_conclusion=row.final_conclusion or "",
                    confidence_score=float(row.confidence_score) if row.confidence_score else 0.0,
                    total_steps=row.total_steps or 0,
                    verification_chain=metadata.get("verification_chain", []),
                )
        except Exception as e:
            logger.error(f"Failed to load reasoning chain: {e}")
            return None

    async def list_chains(
        self,
        node_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ReasoningProvenance]:
        """List reasoning chains."""
        if not self._session_factory:
            return []

        try:
            from sqlalchemy import text

            query = "SELECT chain_id FROM reasoning_chains"
            params = {"limit": limit, "offset": offset}

            if node_id:
                query += " WHERE node_id = :node_id"
                params["node_id"] = node_id

            query += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"

            async with self._session_factory() as session:
                result = await session.execute(text(query), params)
                chain_ids = [row.chain_id for row in result.fetchall()]

                chains = []
                for chain_id in chain_ids:
                    chain = await self.load_reasoning_chain(chain_id)
                    if chain:
                        chains.append(chain)
                return chains
        except Exception as e:
            logger.error(f"Failed to list reasoning chains: {e}")
            return []

    async def close(self) -> None:
        """Close database connections."""
        # Engine is managed by prsm.core.database, no cleanup needed
        pass

    def _record_to_db(self, record: ProvenanceRecord) -> Dict[str, Any]:
        """Convert ProvenanceRecord to database dict."""
        return {
            "record_id": record.record_id,
            "provenance_type": record.provenance_type.value,
            "source_entity": record.source_entity,
            "target_entity": record.target_entity,
            "operation": record.operation,
            "inputs": json.dumps(record.inputs),
            "outputs": json.dumps(record.outputs),
            "metadata": json.dumps(record.metadata),
            "trust_level": record.trust_level.value,
            "verification_status": record.verification_status,
            "content_hash": record.content_hash,
            "timestamp": record.timestamp,
        }

    def _db_to_record(self, row) -> ProvenanceRecord:
        """Convert database row to ProvenanceRecord."""
        return ProvenanceRecord(
            record_id=row.record_id,
            provenance_type=ProvenanceType(row.provenance_type),
            timestamp=row.timestamp if hasattr(row, 'timestamp') else datetime.now(timezone.utc),
            source_entity=row.source_entity,
            target_entity=row.target_entity,
            operation=row.operation,
            inputs=json.loads(row.inputs) if row.inputs else [],
            outputs=json.loads(row.outputs) if row.outputs else [],
            metadata=json.loads(row.metadata) if row.metadata else {},
            trust_level=TrustLevel(row.trust_level),
            verification_status=row.verification_status if hasattr(row, 'verification_status') else False,
            content_hash=row.content_hash or "",
        )


class SQLiteProvenanceBackend(ProvenancePersistenceBackend):
    """
    SQLite backend for provenance persistence.

    Lightweight fallback for single-node / development use.
    Uses aiosqlite for async operations.
    """

    def __init__(self, db_path: str = "~/.prsm/provenance.db"):
        self.db_path = Path(db_path).expanduser()
        self._connection = None

    async def initialize(self) -> bool:
        """Initialize SQLite database."""
        try:
            import aiosqlite

            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = await aiosqlite.connect(self.db_path)

            # Create tables
            await self._connection.executescript("""
                CREATE TABLE IF NOT EXISTS provenance_records (
                    record_id TEXT PRIMARY KEY,
                    provenance_type TEXT NOT NULL,
                    source_entity TEXT NOT NULL,
                    target_entity TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    inputs TEXT,
                    outputs TEXT,
                    metadata TEXT,
                    trust_level TEXT NOT NULL,
                    verification_status INTEGER DEFAULT 0,
                    content_hash TEXT,
                    timestamp TEXT NOT NULL,
                    active INTEGER DEFAULT 1
                );

                CREATE INDEX IF NOT EXISTS idx_prov_source ON provenance_records(source_entity);
                CREATE INDEX IF NOT EXISTS idx_prov_target ON provenance_records(target_entity);
                CREATE INDEX IF NOT EXISTS idx_prov_trust ON provenance_records(trust_level);
                CREATE INDEX IF NOT EXISTS idx_prov_timestamp ON provenance_records(timestamp);

                CREATE TABLE IF NOT EXISTS reasoning_chains (
                    chain_id TEXT PRIMARY KEY,
                    node_id TEXT,
                    query TEXT NOT NULL,
                    final_conclusion TEXT,
                    confidence_score REAL,
                    total_steps INTEGER,
                    step_count INTEGER,
                    finalized INTEGER DEFAULT 0,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    finalized_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_chain_node ON reasoning_chains(node_id);
            """)
            await self._connection.commit()

            logger.info("SQLite provenance backend initialized", path=str(self.db_path))
            return True
        except Exception as e:
            logger.error(f"Failed to initialize SQLite backend: {e}")
            return False

    async def save_record(self, record: ProvenanceRecord) -> bool:
        """Save a provenance record to SQLite."""
        if not self._connection:
            return False

        try:
            await self._connection.execute("""
                INSERT OR REPLACE INTO provenance_records (
                    record_id, provenance_type, source_entity, target_entity,
                    operation, inputs, outputs, metadata, trust_level,
                    verification_status, content_hash, timestamp, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                record.record_id,
                record.provenance_type.value,
                record.source_entity,
                record.target_entity,
                record.operation,
                json.dumps(record.inputs),
                json.dumps(record.outputs),
                json.dumps(record.metadata),
                record.trust_level.value,
                int(record.verification_status),
                record.content_hash,
                record.timestamp.isoformat(),
            ))
            await self._connection.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save provenance record to SQLite: {e}")
            return False

    async def load_record(self, record_id: str) -> Optional[ProvenanceRecord]:
        """Load a provenance record by ID."""
        if not self._connection:
            return None

        try:
            cursor = await self._connection.execute(
                "SELECT * FROM provenance_records WHERE record_id = ? AND active = 1",
                (record_id,)
            )
            row = await cursor.fetchone()
            if row:
                return self._row_to_record(row)
            return None
        except Exception as e:
            logger.error(f"Failed to load provenance record from SQLite: {e}")
            return None

    async def list_records(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ProvenanceRecord]:
        """List provenance records."""
        if not self._connection:
            return []

        try:
            query = "SELECT * FROM provenance_records WHERE active = 1"
            params = []

            if filters:
                if "node_id" in filters:
                    query += " AND source_entity = ?"
                    params.append(filters["node_id"])
                if "trust_level" in filters:
                    query += " AND trust_level = ?"
                    params.append(filters["trust_level"])

            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor = await self._connection.execute(query, params)
            rows = await cursor.fetchall()
            return [self._row_to_record(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to list provenance records from SQLite: {e}")
            return []

    async def delete_record(self, record_id: str) -> bool:
        """Soft delete a provenance record."""
        if not self._connection:
            return False

        try:
            await self._connection.execute(
                "UPDATE provenance_records SET active = 0 WHERE record_id = ?",
                (record_id,)
            )
            await self._connection.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to delete provenance record from SQLite: {e}")
            return False

    async def save_reasoning_chain(self, chain: ReasoningProvenance) -> bool:
        """Save a reasoning chain to SQLite."""
        if not self._connection:
            return False

        try:
            node_id = chain.reasoning_steps[0].source_entity if chain.reasoning_steps else "unknown"

            await self._connection.execute("""
                INSERT OR REPLACE INTO reasoning_chains (
                    chain_id, node_id, query, final_conclusion,
                    confidence_score, total_steps, step_count, finalized, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (
                chain.reasoning_id,
                node_id,
                chain.query,
                chain.final_conclusion,
                chain.confidence_score,
                chain.total_steps,
                len(chain.reasoning_steps),
                int(bool(chain.final_conclusion)),
                json.dumps({
                    "verification_chain": chain.verification_chain,
                    "data_sources_count": len(chain.data_sources),
                    "model_calls_count": len(chain.model_calls),
                }),
            ))
            await self._connection.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save reasoning chain to SQLite: {e}")
            return False

    async def load_reasoning_chain(self, chain_id: str) -> Optional[ReasoningProvenance]:
        """Load a reasoning chain by ID."""
        if not self._connection:
            return None

        try:
            cursor = await self._connection.execute(
                "SELECT * FROM reasoning_chains WHERE chain_id = ?",
                (chain_id,)
            )
            row = await cursor.fetchone()
            if not row:
                return None

            # Load related records
            records_cursor = await self._connection.execute(
                "SELECT * FROM provenance_records WHERE target_entity = ? AND active = 1 ORDER BY timestamp",
                (chain_id,)
            )
            records = [self._row_to_record(r) for r in await records_cursor.fetchall()]

            metadata = json.loads(row[8]) if row[8] else {}

            return ReasoningProvenance(
                reasoning_id=row[0],
                query=row[2],
                reasoning_steps=records,
                data_sources=[r for r in records if r.provenance_type == ProvenanceType.DATA_SOURCE],
                model_calls=[r for r in records if r.provenance_type == ProvenanceType.MODEL_PREDICTION],
                final_conclusion=row[3] or "",
                confidence_score=row[4] or 0.0,
                total_steps=row[5] or 0,
                verification_chain=metadata.get("verification_chain", []),
            )
        except Exception as e:
            logger.error(f"Failed to load reasoning chain from SQLite: {e}")
            return None

    async def list_chains(
        self,
        node_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ReasoningProvenance]:
        """List reasoning chains."""
        if not self._connection:
            return []

        try:
            if node_id:
                cursor = await self._connection.execute(
                    "SELECT chain_id FROM reasoning_chains WHERE node_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (node_id, limit, offset)
                )
            else:
                cursor = await self._connection.execute(
                    "SELECT chain_id FROM reasoning_chains ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (limit, offset)
                )

            chain_ids = [row[0] for row in await cursor.fetchall()]
            chains = []
            for chain_id in chain_ids:
                chain = await self.load_reasoning_chain(chain_id)
                if chain:
                    chains.append(chain)
            return chains
        except Exception as e:
            logger.error(f"Failed to list reasoning chains from SQLite: {e}")
            return []

    async def close(self) -> None:
        """Close SQLite connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    def _row_to_record(self, row) -> ProvenanceRecord:
        """Convert SQLite row to ProvenanceRecord."""
        return ProvenanceRecord(
            record_id=row[0],
            provenance_type=ProvenanceType(row[1]),
            timestamp=datetime.fromisoformat(row[11]) if row[11] else datetime.now(timezone.utc),
            source_entity=row[2],
            target_entity=row[3],
            operation=row[4],
            inputs=json.loads(row[5]) if row[5] else [],
            outputs=json.loads(row[6]) if row[6] else [],
            metadata=json.loads(row[7]) if row[7] else {},
            trust_level=TrustLevel(row[8]),
            verification_status=bool(row[9]),
            content_hash=row[10] or "",
        )


class ProvenanceGossipBridge:
    """
    Connects EnhancedProvenanceSystem to GossipProtocol for cross-node verification.

    Enables nodes to broadcast provenance records and verify each other's chains.
    """

    def __init__(
        self,
        provenance_system: 'EnhancedProvenanceSystem',
        gossip: Any,
        node_id: str = "unknown",
    ):
        self.provenance = provenance_system
        self.gossip = gossip
        self.node_id = node_id
        self._running = False
        self._verification_queue: List[Dict[str, Any]] = []

    async def start(self) -> None:
        """Start listening for provenance gossip messages."""
        if self._running:
            return

        # Subscribe to provenance message types
        self.gossip.subscribe("provenance_broadcast", self._on_provenance_broadcast)
        self.gossip.subscribe("provenance_verify", self._on_provenance_verify)
        self.gossip.subscribe("provenance_verified", self._on_provenance_verified)

        self._running = True
        logger.info("Provenance gossip bridge started", node_id=self.node_id)

    async def stop(self) -> None:
        """Stop the gossip bridge."""
        self._running = False
        logger.info("Provenance gossip bridge stopped")

    async def broadcast_record(self, record: ProvenanceRecord) -> None:
        """Broadcast a provenance record to the network."""
        if not self._running:
            return

        try:
            payload = {
                "record_id": record.record_id,
                "provenance_type": record.provenance_type.value,
                "source_entity": record.source_entity,
                "target_entity": record.target_entity,
                "operation": record.operation,
                "content_hash": record.content_hash,
                "trust_level": record.trust_level.value,
                "timestamp": record.timestamp.isoformat(),
                "broadcaster_node_id": self.node_id,
            }

            await self.gossip.publish("provenance_broadcast", payload)

            logger.debug("Broadcast provenance record", record_id=record.record_id)
        except Exception as e:
            logger.error(f"Failed to broadcast provenance record: {e}")

    async def _on_provenance_broadcast(
        self,
        subtype: str,
        data: Dict[str, Any],
        origin: str
    ) -> None:
        """Handle incoming provenance broadcast from another node."""
        try:
            # Add to verification queue
            self._verification_queue.append({
                "record_data": data,
                "origin": origin,
                "received_at": datetime.now(timezone.utc).isoformat(),
            })

            logger.debug(
                "Received provenance broadcast",
                record_id=data.get("record_id"),
                origin=origin
            )
        except Exception as e:
            logger.error(f"Error handling provenance broadcast: {e}")

    async def _on_provenance_verify(
        self,
        subtype: str,
        data: Dict[str, Any],
        origin: str
    ) -> None:
        """Handle verification request from another node."""
        try:
            record_id = data.get("record_id")
            if not record_id:
                return

            # Verify the record
            is_valid = self.provenance.verify_provenance_chain(record_id)

            # Send verification response
            response = {
                "record_id": record_id,
                "is_valid": is_valid,
                "verifier_node_id": self.node_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            await self.gossip.publish("provenance_verified", response)

            logger.debug(
                "Responded to verification request",
                record_id=record_id,
                is_valid=is_valid
            )
        except Exception as e:
            logger.error(f"Error handling verification request: {e}")

    async def _on_provenance_verified(
        self,
        subtype: str,
        data: Dict[str, Any],
        origin: str
    ) -> None:
        """Handle verification response from another node."""
        try:
            record_id = data.get("record_id")
            is_valid = data.get("is_valid", False)
            verifier = data.get("verifier_node_id", "unknown")

            # Update trust score based on verification
            if record_id and record_id in self.provenance.provenance_records:
                record = self.provenance.provenance_records[record_id]
                if is_valid:
                    # Increase trust level if verified by peer
                    trust_upgrade = {
                        TrustLevel.UNCERTAIN: TrustLevel.PROVISIONAL,
                        TrustLevel.PROVISIONAL: TrustLevel.CREDIBLE,
                        TrustLevel.CREDIBLE: TrustLevel.TRUSTED,
                    }
                    record.trust_level = trust_upgrade.get(record.trust_level, record.trust_level)

                logger.debug(
                    "Received verification response",
                    record_id=record_id,
                    is_valid=is_valid,
                    verifier=verifier
                )
        except Exception as e:
            logger.error(f"Error handling verification response: {e}")


class EnhancedProvenanceSystem:
    """
    Enhanced Provenance System for PRSM
    
    Tracks and verifies the complete lineage of reasoning, data,
    and computational operations within the PRSM ecosystem.
    """
    
    def __init__(self):
        """Initialize enhanced provenance system"""
        self.provenance_records: Dict[str, ProvenanceRecord] = {}
        self.reasoning_chains: Dict[str, ReasoningProvenance] = {}
        self.trust_scores: Dict[str, float] = {}
        self.verification_cache: Dict[str, bool] = {}
        
        logger.info("EnhancedProvenanceSystem initialized")
    
    def create_provenance_record(self,
                               provenance_type: ProvenanceType,
                               source_entity: str,
                               target_entity: str,
                               operation: str,
                               inputs: Optional[List[str]] = None,
                               outputs: Optional[List[str]] = None,
                               metadata: Optional[Dict[str, Any]] = None,
                               trust_level: TrustLevel = TrustLevel.CREDIBLE) -> ProvenanceRecord:
        """Create new provenance record"""
        try:
            record_id = f"prov_{uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
            
            # Create content hash for integrity
            content_data = {
                'type': provenance_type.value,
                'source': source_entity,
                'target': target_entity,
                'operation': operation,
                'inputs': inputs or [],
                'outputs': outputs or [],
                'metadata': metadata or {}
            }
            content_hash = hashlib.sha256(json.dumps(content_data, sort_keys=True).encode()).hexdigest()[:16]
            
            record = ProvenanceRecord(
                record_id=record_id,
                provenance_type=provenance_type,
                timestamp=datetime.now(timezone.utc),
                source_entity=source_entity,
                target_entity=target_entity,
                operation=operation,
                inputs=inputs or [],
                outputs=outputs or [],
                metadata=metadata or {},
                trust_level=trust_level,
                content_hash=content_hash
            )
            
            self.provenance_records[record_id] = record
            
            logger.debug("Provenance record created",
                        record_id=record_id,
                        type=provenance_type.value,
                        operation=operation)
            
            return record
            
        except Exception as e:
            logger.error(f"Failed to create provenance record: {e}")
            raise
    
    def start_reasoning_chain(self, query: str) -> str:
        """Start tracking a reasoning chain"""
        reasoning_id = f"reasoning_{uuid4().hex[:8]}"
        
        # Create initial data source record
        initial_record = self.create_provenance_record(
            provenance_type=ProvenanceType.DATA_SOURCE,
            source_entity="user_query",
            target_entity=reasoning_id,
            operation="query_initiation",
            metadata={
                'query_text': query,
                'query_length': len(query),
                'reasoning_initiated': True
            }
        )
        
        reasoning_chain = ReasoningProvenance(
            reasoning_id=reasoning_id,
            query=query,
            reasoning_steps=[initial_record],
            data_sources=[initial_record],
            model_calls=[],
            final_conclusion="",
            confidence_score=0.0,
            total_steps=1
        )
        
        self.reasoning_chains[reasoning_id] = reasoning_chain
        
        logger.info("Reasoning chain started", reasoning_id=reasoning_id, query=query[:100])
        
        return reasoning_id
    
    def add_reasoning_step(self,
                          reasoning_id: str,
                          step_type: str,
                          operation: str,
                          inputs: List[str],
                          outputs: List[str],
                          metadata: Optional[Dict[str, Any]] = None) -> ProvenanceRecord:
        """Add step to reasoning chain"""
        if reasoning_id not in self.reasoning_chains:
            raise ValueError(f"Reasoning chain {reasoning_id} not found")
        
        reasoning_chain = self.reasoning_chains[reasoning_id]
        
        # Create reasoning step record
        step_record = self.create_provenance_record(
            provenance_type=ProvenanceType.REASONING_STEP,
            source_entity=step_type,
            target_entity=reasoning_id,
            operation=operation,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata or {}
        )
        
        reasoning_chain.reasoning_steps.append(step_record)
        reasoning_chain.total_steps = len(reasoning_chain.reasoning_steps)
        
        logger.debug("Reasoning step added",
                    reasoning_id=reasoning_id,
                    step_type=step_type,
                    operation=operation)
        
        return step_record
    
    def add_data_source(self,
                       reasoning_id: str,
                       source_type: str,
                       source_id: str,
                       metadata: Optional[Dict[str, Any]] = None) -> ProvenanceRecord:
        """Add data source to reasoning chain"""
        if reasoning_id not in self.reasoning_chains:
            raise ValueError(f"Reasoning chain {reasoning_id} not found")
        
        reasoning_chain = self.reasoning_chains[reasoning_id]
        
        # Create data source record
        source_record = self.create_provenance_record(
            provenance_type=ProvenanceType.DATA_SOURCE,
            source_entity=source_type,
            target_entity=reasoning_id,
            operation="data_retrieval",
            outputs=[source_id],
            metadata=metadata or {}
        )
        
        reasoning_chain.data_sources.append(source_record)
        
        logger.debug("Data source added",
                    reasoning_id=reasoning_id,
                    source_type=source_type,
                    source_id=source_id)
        
        return source_record
    
    def add_model_call(self,
                      reasoning_id: str,
                      model_id: str,
                      input_text: str,
                      output_text: str,
                      metadata: Optional[Dict[str, Any]] = None) -> ProvenanceRecord:
        """Add model call to reasoning chain"""
        if reasoning_id not in self.reasoning_chains:
            raise ValueError(f"Reasoning chain {reasoning_id} not found")
        
        reasoning_chain = self.reasoning_chains[reasoning_id]
        
        # Create model prediction record
        model_record = self.create_provenance_record(
            provenance_type=ProvenanceType.MODEL_PREDICTION,
            source_entity=model_id,
            target_entity=reasoning_id,
            operation="model_inference",
            metadata={
                'input_text': input_text[:500],  # Truncate for storage
                'output_text': output_text[:500],
                'input_length': len(input_text),
                'output_length': len(output_text),
                **(metadata or {})
            }
        )
        
        reasoning_chain.model_calls.append(model_record)
        
        logger.debug("Model call added",
                    reasoning_id=reasoning_id,
                    model_id=model_id,
                    input_length=len(input_text))
        
        return model_record
    
    def finalize_reasoning_chain(self,
                               reasoning_id: str,
                               final_conclusion: str,
                               confidence_score: float) -> ReasoningProvenance:
        """Finalize reasoning chain with conclusion"""
        if reasoning_id not in self.reasoning_chains:
            raise ValueError(f"Reasoning chain {reasoning_id} not found")
        
        reasoning_chain = self.reasoning_chains[reasoning_id]
        reasoning_chain.final_conclusion = final_conclusion
        reasoning_chain.confidence_score = confidence_score
        
        # Create final conclusion record
        conclusion_record = self.create_provenance_record(
            provenance_type=ProvenanceType.COMPUTATION,
            source_entity="reasoning_system",
            target_entity=reasoning_id,
            operation="reasoning_conclusion",
            inputs=[r.record_id for r in reasoning_chain.reasoning_steps],
            metadata={
                'conclusion': final_conclusion[:500],
                'confidence_score': confidence_score,
                'total_reasoning_steps': reasoning_chain.total_steps,
                'data_sources_count': len(reasoning_chain.data_sources),
                'model_calls_count': len(reasoning_chain.model_calls)
            },
            trust_level=TrustLevel.TRUSTED if confidence_score > 0.8 else TrustLevel.CREDIBLE
        )
        
        reasoning_chain.reasoning_steps.append(conclusion_record)
        
        logger.info("Reasoning chain finalized",
                   reasoning_id=reasoning_id,
                   confidence_score=confidence_score,
                   total_steps=reasoning_chain.total_steps)
        
        return reasoning_chain
    
    def verify_provenance_chain(self, record_id: str) -> bool:
        """Verify integrity of provenance chain"""
        try:
            if record_id in self.verification_cache:
                return self.verification_cache[record_id]
            
            record = self.provenance_records.get(record_id)
            if not record:
                return False
            
            # Check content hash integrity
            content_data = {
                'type': record.provenance_type.value,
                'source': record.source_entity,
                'target': record.target_entity,
                'operation': record.operation,
                'inputs': record.inputs,
                'outputs': record.outputs,
                'metadata': record.metadata
            }
            expected_hash = hashlib.sha256(json.dumps(content_data, sort_keys=True).encode()).hexdigest()[:16]
            
            if record.content_hash != expected_hash:
                logger.warning("Provenance record hash mismatch", record_id=record_id)
                self.verification_cache[record_id] = False
                return False
            
            # Verify input dependencies
            for input_id in record.inputs:
                if input_id in self.provenance_records:
                    if not self.verify_provenance_chain(input_id):
                        self.verification_cache[record_id] = False
                        return False
            
            record.verification_status = True
            self.verification_cache[record_id] = True
            
            logger.debug("Provenance record verified", record_id=record_id)
            return True
            
        except Exception as e:
            logger.error(f"Provenance verification failed for {record_id}: {e}")
            self.verification_cache[record_id] = False
            return False
    
    def calculate_trust_score(self, reasoning_id: str) -> float:
        """Calculate trust score for reasoning chain"""
        if reasoning_id not in self.reasoning_chains:
            return 0.0
        
        if reasoning_id in self.trust_scores:
            return self.trust_scores[reasoning_id]
        
        reasoning_chain = self.reasoning_chains[reasoning_id]
        
        # Trust factors
        verification_score = 0.0
        trust_level_score = 0.0
        completeness_score = 0.0
        
        total_records = len(reasoning_chain.reasoning_steps)
        if total_records == 0:
            return 0.0
        
        # Calculate verification score
        verified_records = sum(1 for step in reasoning_chain.reasoning_steps 
                             if self.verify_provenance_chain(step.record_id))
        verification_score = verified_records / total_records
        
        # Calculate trust level score
        trust_values = {
            TrustLevel.VERIFIED: 1.0,
            TrustLevel.TRUSTED: 0.9,
            TrustLevel.CREDIBLE: 0.7,
            TrustLevel.PROVISIONAL: 0.5,
            TrustLevel.UNCERTAIN: 0.3
        }
        
        avg_trust = sum(trust_values.get(step.trust_level, 0.5) 
                       for step in reasoning_chain.reasoning_steps) / total_records
        trust_level_score = avg_trust
        
        # Calculate completeness score
        has_data_sources = len(reasoning_chain.data_sources) > 0
        has_model_calls = len(reasoning_chain.model_calls) > 0
        has_conclusion = bool(reasoning_chain.final_conclusion)
        
        completeness_score = sum([has_data_sources, has_model_calls, has_conclusion]) / 3.0
        
        # Combine scores
        final_trust_score = (
            verification_score * 0.4 +
            trust_level_score * 0.4 +
            completeness_score * 0.2
        )
        
        self.trust_scores[reasoning_id] = final_trust_score
        
        logger.debug("Trust score calculated",
                    reasoning_id=reasoning_id,
                    trust_score=final_trust_score,
                    verification=verification_score,
                    trust_level=trust_level_score,
                    completeness=completeness_score)
        
        return final_trust_score
    
    def get_reasoning_provenance(self, reasoning_id: str) -> Optional[ReasoningProvenance]:
        """Get complete provenance for reasoning chain"""
        return self.reasoning_chains.get(reasoning_id)
    
    def get_provenance_summary(self, reasoning_id: str) -> Dict[str, Any]:
        """Get summary of provenance information"""
        if reasoning_id not in self.reasoning_chains:
            return {}
        
        reasoning_chain = self.reasoning_chains[reasoning_id]
        trust_score = self.calculate_trust_score(reasoning_id)
        
        return {
            'reasoning_id': reasoning_id,
            'query': reasoning_chain.query,
            'total_steps': reasoning_chain.total_steps,
            'data_sources_count': len(reasoning_chain.data_sources),
            'model_calls_count': len(reasoning_chain.model_calls),
            'confidence_score': reasoning_chain.confidence_score,
            'trust_score': trust_score,
            'verification_status': trust_score > 0.7,
            'final_conclusion_preview': reasoning_chain.final_conclusion[:200] + "..." if reasoning_chain.final_conclusion else "",
            'created_at': reasoning_chain.reasoning_steps[0].timestamp.isoformat() if reasoning_chain.reasoning_steps else None
        }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get provenance system statistics"""
        total_records = len(self.provenance_records)
        total_chains = len(self.reasoning_chains)
        verified_records = sum(1 for record_id in self.provenance_records 
                             if self.verify_provenance_chain(record_id))
        
        # Type distribution
        type_counts = {}
        for record in self.provenance_records.values():
            ptype = record.provenance_type.value
            type_counts[ptype] = type_counts.get(ptype, 0) + 1
        
        # Trust level distribution
        trust_counts = {}
        for record in self.provenance_records.values():
            trust = record.trust_level.value
            trust_counts[trust] = trust_counts.get(trust, 0) + 1
        
        return {
            'total_provenance_records': total_records,
            'total_reasoning_chains': total_chains,
            'verified_records': verified_records,
            'verification_rate': (verified_records / max(total_records, 1)) * 100,
            'record_types': type_counts,
            'trust_levels': trust_counts,
            'average_chain_length': sum(len(chain.reasoning_steps) for chain in self.reasoning_chains.values()) / max(total_chains, 1)
        }


# Global provenance system instance
_global_provenance_system: Optional[EnhancedProvenanceSystem] = None


def get_enhanced_provenance_system() -> EnhancedProvenanceSystem:
    """Get global enhanced provenance system instance"""
    global _global_provenance_system
    if _global_provenance_system is None:
        _global_provenance_system = EnhancedProvenanceSystem()
    return _global_provenance_system


# Convenience functions
def track_reasoning_step(reasoning_id: str, step_type: str, operation: str, 
                        inputs: List[str] = None, outputs: List[str] = None,
                        metadata: Dict[str, Any] = None) -> ProvenanceRecord:
    """Convenience function to track reasoning step"""
    system = get_enhanced_provenance_system()
    return system.add_reasoning_step(
        reasoning_id=reasoning_id,
        step_type=step_type,
        operation=operation,
        inputs=inputs or [],
        outputs=outputs or [],
        metadata=metadata
    )


def start_reasoning_tracking(query: str) -> str:
    """Start tracking a new reasoning chain"""
    system = get_enhanced_provenance_system()
    return system.start_reasoning_chain(query)
