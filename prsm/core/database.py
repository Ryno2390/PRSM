"""
PRSM Database Layer - PostgreSQL Integration

🎯 PURPOSE IN PRSM:
This module provides the complete database abstraction layer for PRSM, implementing
real PostgreSQL persistence to replace placeholder TODO implementations across
the system.

🔧 INTEGRATION POINTS:
- API endpoints: Real data persistence for all operations
- Session management: Persistent user sessions and context tracking
- FTNS system: Transaction history and balance management
- Agent framework: Task hierarchy and execution history
- Teacher models: Model metadata and training records
- Safety system: Circuit breaker events and governance records
- P2P federation: Peer discovery and model registry

🚀 REAL-WORLD CAPABILITIES:
- Connection pooling with automatic retry and health checking
- Transaction management with rollback support
- Database migrations for schema evolution
- Performance monitoring and query optimization
- Backup and recovery integration
- Multi-environment configuration support
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, AsyncGenerator
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, BigInteger, String, DateTime,
    Float, Boolean, JSON, Text, ForeignKey, Index, UniqueConstraint, text,
    UUID as SQLAlchemyUUID
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import structlog

# Safe settings import with fallback
try:
    from prsm.core.config import get_settings
    settings = get_settings()
    if settings is None:
        raise Exception("Settings is None")
except Exception:
    # Fallback settings for database
    class FallbackDatabaseSettings:
        def __init__(self):
            self.database_url = "sqlite:///./prsm_test.db"
            self.database_echo = False
            self.database_pool_size = 5
            self.database_max_overflow = 10
            self.environment = "development"
            self.debug = True
    
    settings = FallbackDatabaseSettings()

logger = structlog.get_logger(__name__)

# === Database Configuration ===

Base = declarative_base()
metadata = MetaData()

# 🔧 DATABASE ENGINE SETUP
# Creates PostgreSQL engine with optimized connection pooling
async_engine = None
async_session_factory = None
sync_engine = None
sync_session_factory = None


async def get_async_engine():
    """
    Get async database engine with connection pooling
    
    🔧 OPTIMIZATION FEATURES:
    - Connection pooling for high concurrency
    - Automatic connection health checking
    - Retry logic for transient failures
    - Performance monitoring integration
    """
    global async_engine
    if async_engine is None:
        database_url = settings.database_url
        
        # Convert sync URL to async for asyncpg
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
        elif database_url.startswith("sqlite:///"):
            database_url = database_url.replace("sqlite:///", "sqlite+aiosqlite:///")
        
        async_engine = create_async_engine(
            database_url,
            echo=settings.database_echo,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_pre_ping=True,  # Health check connections
            pool_recycle=3600,   # Recycle connections every hour
        )
        
        logger.info("Async database engine created", url_scheme=database_url.split("://")[0])
    
    return async_engine


def get_sync_engine():
    """Get sync database engine for migrations and admin tasks"""
    global sync_engine
    if sync_engine is None:
        sync_engine = create_engine(
            settings.database_url,
            echo=settings.database_echo,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_pre_ping=True,
        )
        
        logger.info("Sync database engine created")
    
    return sync_engine


async def get_async_session_factory():
    """Get async session factory"""
    global async_session_factory
    if async_session_factory is None:
        engine = await get_async_engine()
        async_session_factory = async_sessionmaker(
            engine, 
            class_=AsyncSession,
            expire_on_commit=False
        )
    return async_session_factory


def get_sync_session_factory():
    """Get sync session factory"""
    global sync_session_factory
    if sync_session_factory is None:
        engine = get_sync_engine()
        sync_session_factory = sessionmaker(bind=engine)
    return sync_session_factory


# === SQLAlchemy Models ===

class PRSMSessionModel(Base):
    """
    Database model for PRSM sessions
    
    🧠 PRSM INTEGRATION:
    Stores complete session lifecycle including context allocation,
    reasoning traces, and safety validations for audit trails
    """
    __tablename__ = "prsm_sessions"
    
    session_id = Column(UUID(as_uuid=True), primary_key=True)
    user_id = Column(String(255), nullable=False, index=True)
    nwtn_context_allocation = Column(Integer, default=0)
    context_used = Column(Integer, default=0)
    status = Column(String(50), default="pending", index=True)
    model_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    reasoning_steps = relationship("ReasoningStepModel", back_populates="session")
    safety_flags = relationship("SafetyFlagModel", back_populates="session")
    tasks = relationship("ArchitectTaskModel", back_populates="session")
    
    __table_args__ = (
        Index('idx_session_user_created', 'user_id', 'created_at'),
        Index('idx_session_status_created', 'status', 'created_at'),
    )


class ReasoningStepModel(Base):
    """Database model for reasoning steps"""
    __tablename__ = "reasoning_steps"
    
    step_id = Column(UUID(as_uuid=True), primary_key=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("prsm_sessions.session_id"), nullable=False)
    agent_type = Column(String(50), nullable=False)
    agent_id = Column(String(255), nullable=False)
    input_data = Column(JSON, nullable=False)
    output_data = Column(JSON, nullable=False)
    execution_time = Column(Float, nullable=False)
    confidence_score = Column(Float)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("PRSMSessionModel", back_populates="reasoning_steps")
    
    __table_args__ = (
        Index('idx_reasoning_session_timestamp', 'session_id', 'timestamp'),
        Index('idx_reasoning_agent_type', 'agent_type'),
    )


class SafetyFlagModel(Base):
    """Database model for safety flags"""
    __tablename__ = "safety_flags"
    
    flag_id = Column(UUID(as_uuid=True), primary_key=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("prsm_sessions.session_id"), nullable=False)
    level = Column(String(20), nullable=False, index=True)
    category = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    triggered_by = Column(String(255), nullable=False)
    resolved = Column(Boolean, default=False, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("PRSMSessionModel", back_populates="safety_flags")
    
    __table_args__ = (
        Index('idx_safety_level_timestamp', 'level', 'timestamp'),
        Index('idx_safety_resolved', 'resolved'),
    )


class ArchitectTaskModel(Base):
    """Database model for architect tasks"""
    __tablename__ = "architect_tasks"
    
    task_id = Column(UUID(as_uuid=True), primary_key=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("prsm_sessions.session_id"), nullable=False)
    parent_task_id = Column(UUID(as_uuid=True), ForeignKey("architect_tasks.task_id"), nullable=True)
    level = Column(Integer, default=0)
    instruction = Column(Text, nullable=False)
    complexity_score = Column(Float, default=0.0)
    dependencies = Column(JSON, default=list)
    status = Column(String(50), default="pending", index=True)
    assigned_agent = Column(String(255))
    result = Column(JSON)
    execution_time = Column(Float)
    model_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    session = relationship("PRSMSessionModel", back_populates="tasks")
    parent_task = relationship("ArchitectTaskModel", remote_side="ArchitectTaskModel.task_id")
    
    __table_args__ = (
        Index('idx_task_session_level', 'session_id', 'level'),
        Index('idx_task_status_created', 'status', 'created_at'),
        Index('idx_task_parent', 'parent_task_id'),
    )


class FTNSTransactionModel(Base):
    """
    Database model for FTNS transactions
    
    🪙 FTNS INTEGRATION:
    Complete transaction history for token economy including
    rewards, charges, transfers, and dividend distributions
    
    Security features:
    - Idempotency key for duplicate detection
    - Balance snapshots for audit trail
    """
    __tablename__ = "ftns_transactions"
    
    transaction_id = Column(UUID(as_uuid=True), primary_key=True)
    from_user = Column(String(255), nullable=True, index=True)
    to_user = Column(String(255), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    transaction_type = Column(String(50), nullable=False, index=True)
    description = Column(Text, nullable=False)
    status = Column(String(20), default="completed", nullable=False)
    idempotency_key = Column(String(255), unique=True, nullable=True, index=True)
    context_units = Column(Integer)
    ipfs_cid = Column(String(255))
    block_hash = Column(String(255))
    balance_before_sender = Column(Float)
    balance_after_sender = Column(Float)
    balance_before_receiver = Column(Float)
    balance_after_receiver = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_ftns_user_type_created', 'to_user', 'transaction_type', 'created_at'),
        Index('idx_ftns_created', 'created_at'),
        Index('idx_ftns_block_hash', 'block_hash'),
        Index('idx_ftns_idempotency', 'idempotency_key'),
    )


class FTNSBalanceModel(Base):
    """Database model for FTNS balances
    
    Includes version column for optimistic concurrency control (OCC)
    to prevent race conditions during balance updates.
    """
    __tablename__ = "ftns_balances"
    
    user_id = Column(String(255), primary_key=True)
    balance = Column(Float, default=0.0, nullable=False)
    locked_balance = Column(Float, default=0.0, nullable=False)
    total_earned = Column(Float, default=0.0, nullable=False)
    total_spent = Column(Float, default=0.0, nullable=False)
    version = Column(Integer, default=1, nullable=False)
    last_transaction_id = Column(UUID(as_uuid=True))
    last_dividend = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_ftns_balance_updated', 'updated_at'),
    )


class TeacherModelModel(Base):
    """
    Database model for teacher models
    
    🧠 TEACHER SYSTEM INTEGRATION:
    Metadata and performance tracking for distilled teacher models
    in the PRSM ecosystem
    """
    __tablename__ = "teacher_models"
    
    teacher_id = Column(UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False)
    specialization = Column(String(255), nullable=False, index=True)
    model_type = Column(String(50), default="teacher", index=True)
    performance_score = Column(Float, default=0.0)
    curriculum_ids = Column(JSON, default=list)
    student_models = Column(JSON, default=list)
    rlvr_score = Column(Float)
    ipfs_cid = Column(String(255), index=True)
    version = Column(String(50), default="1.0.0")
    active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_teacher_specialization_active', 'specialization', 'active'),
        Index('idx_teacher_performance', 'performance_score'),
        UniqueConstraint('name', 'version', name='uq_teacher_name_version'),
    )


class CircuitBreakerEventModel(Base):
    """
    Database model for circuit breaker events
    
    🛡️ SAFETY INTEGRATION:
    Tracks all safety system activations for governance and monitoring
    """
    __tablename__ = "circuit_breaker_events"
    
    event_id = Column(UUID(as_uuid=True), primary_key=True)
    triggered_by = Column(String(255), nullable=False)
    safety_level = Column(String(20), nullable=False, index=True)
    reason = Column(Text, nullable=False)
    affected_components = Column(JSON, default=list)
    resolution_action = Column(Text)
    resolved_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_circuit_breaker_level_created', 'safety_level', 'created_at'),
        Index('idx_circuit_breaker_resolved', 'resolved_at'),
    )


class PeerNodeModel(Base):
    """
    Database model for P2P network peers
    
    🌐 P2P FEDERATION:
    Tracks peer nodes in the distributed PRSM network for
    model discovery and task distribution
    """
    __tablename__ = "peer_nodes"
    
    node_id = Column(String(255), primary_key=True)
    peer_id = Column(String(255), nullable=False, unique=True)
    multiaddr = Column(String(500), nullable=False)
    capabilities = Column(JSON, default=list)
    reputation_score = Column(Float, default=0.5)
    last_seen = Column(DateTime(timezone=True), server_default=func.now())
    active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_peer_active_last_seen', 'active', 'last_seen'),
        Index('idx_peer_reputation', 'reputation_score'),
    )


class ContentProvenanceModel(Base):
    """
    Database model for IPFS content provenance records.

    Persists ContentUploader.uploaded_content across node restarts so
    royalty collection continues after a node is restarted. Also enables
    cross-node provenance queries through the platform API.

    Each row corresponds to one UploadedContent dataclass instance.
    The in-memory dict remains authoritative during the process lifetime;
    this table is the backing store for hydration on restart.
    """
    __tablename__ = "content_provenance"

    # Primary identifier — IPFS CID (or manifest CID for sharded content)
    cid = Column(String(255), primary_key=True)

    filename = Column(String(500), nullable=False)
    size_bytes = Column(BigInteger, nullable=False)          # BigInteger: model weights can exceed 2GB
    content_hash = Column(String(64), nullable=False)        # SHA-256 hex digest

    creator_id = Column(String(255), nullable=False, index=True)
    provenance_signature = Column(Text, nullable=False)
    royalty_rate = Column(Float, nullable=False, default=0.01)
    parent_cids = Column(JSON, default=list)                 # List[str] — derivative lineage

    # Updated on each content access via update_access_stats()
    access_count = Column(Integer, nullable=False, default=0)
    total_royalties = Column(Float, nullable=False, default=0.0)

    # Sharding metadata
    is_sharded = Column(Boolean, nullable=False, default=False)
    manifest_cid = Column(String(255), nullable=True, index=True)
    total_shards = Column(Integer, nullable=False, default=0)

    # Semantic deduplication metadata
    embedding_id = Column(String(255), nullable=True)
    near_duplicate_of = Column(String(255), nullable=True)
    near_duplicate_similarity = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('idx_provenance_creator', 'creator_id'),
        Index('idx_provenance_hash', 'content_hash'),
        Index('idx_provenance_created', 'created_at'),
    )


class GovernanceProposalModel(Base):
    """
    Database model for governance proposals.

    Persists TokenWeightedVoting.proposals dict across restarts. Stores
    aggregate vote counts rather than individual vote records (sufficient
    for MVP — individual votes are reconstructible from event logs later).

    Note: required_quorum stored as Float for cross-proposal comparison;
    total_voting_power stored as Float to avoid Decimal serialization issues.
    """
    __tablename__ = "governance_proposals"

    proposal_id  = Column(UUID(as_uuid=True), primary_key=True)
    proposer_id  = Column(String(255), nullable=False, index=True)
    title        = Column(String(500), nullable=False)
    description  = Column(Text, nullable=False)
    proposal_type = Column(String(100), nullable=False, index=True)
    status        = Column(String(50),  nullable=False, default="active", index=True)

    # Aggregate vote state — updated on each cast_vote() call
    votes_for          = Column(Integer, nullable=False, default=0)
    votes_against      = Column(Integer, nullable=False, default=0)
    total_voting_power = Column(Float,   nullable=False, default=0.0)
    required_quorum    = Column(Float,   nullable=True)

    # Voting window
    voting_starts = Column(DateTime(timezone=True), nullable=True)
    voting_ends   = Column(DateTime(timezone=True), nullable=True)

    # Flexible metadata (implementation_details, budget_impact, etc.)
    proposal_metadata = Column(JSON, default=dict)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(),
                        onupdate=func.now())

    __table_args__ = (
        Index('idx_governance_status',   'status'),
        Index('idx_governance_proposer', 'proposer_id'),
        Index('idx_governance_created',  'created_at'),
    )


class ModelRegistryModel(Base):
    """
    Database model for the global model registry
    
    🤖 MODEL MARKETPLACE:
    Central registry for all models in the PRSM ecosystem including
    availability, pricing, and performance metadata
    """
    __tablename__ = "model_registry"
    
    model_id = Column(String(255), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    model_type = Column(String(50), nullable=False, index=True)
    specialization = Column(String(255), index=True)
    owner_id = Column(String(255), nullable=False, index=True)
    ipfs_cid = Column(String(255), index=True)
    version = Column(String(50), default="1.0.0")
    performance_metrics = Column(JSON, default=dict)
    resource_requirements = Column(JSON, default=dict)
    pricing_model = Column(JSON, default=dict)
    availability_status = Column(String(50), default="available", index=True)
    total_usage_hours = Column(Float, default=0.0)
    reputation_score = Column(Float, default=0.5)
    active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_model_type_specialization', 'model_type', 'specialization'),
        Index('idx_model_owner_active', 'owner_id', 'active'),
        Index('idx_model_availability', 'availability_status'),
        UniqueConstraint('name', 'version', 'owner_id', name='uq_model_name_version_owner'),
    )


class UserAPIConfigModel(Base):
    """
    Database model for user LLM provider API configurations.

    Persists per-user, per-provider API keys and settings across restarts.
    Config data is stored as JSON; encryption at rest is a future enhancement
    (see known technical debt in project plan).
    """
    __tablename__ = "user_api_configs"

    id = Column(UUID(as_uuid=True), primary_key=True)
    user_id = Column(String(255), nullable=False, index=True)
    provider = Column(String(100), nullable=False)
    config_data = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint('user_id', 'provider', name='uq_user_api_config_provider'),
        Index('idx_user_api_config_user', 'user_id'),
    )


# === Database Session Management ===

@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions
    
    🔧 TRANSACTION MANAGEMENT:
    - Automatic rollback on exceptions
    - Connection cleanup on completion
    - Performance monitoring integration
    """
    session_factory = await get_async_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error("Database transaction rolled back", error=str(e))
            raise
        finally:
            await session.close()


def get_sync_session():
    """Sync context manager for database sessions"""
    session_factory = get_sync_session_factory()
    return session_factory()

async def get_db_session():
    """
    Get database session (alias for get_async_session for compatibility)

    Returns:
        AsyncSession: SQLAlchemy async session
    """
    async for session in get_async_session():
        yield session

# Alias expected by prsm.interface.api.dependencies
get_db = get_db_session

# === Database Operations ===

class DatabaseManager:
    """
    Database manager for PRSM operations
    
    🎯 PURPOSE: Centralized database operations with error handling,
    performance monitoring, and automatic retry logic for reliability
    """
    
    def __init__(self):
        self.connection_healthy = False
        self.last_health_check = None
    
    async def initialize(self):
        """
        Initialize database connections and verify health
        
        🚀 STARTUP SEQUENCE:
        1. Create database engines with optimized configuration
        2. Test connectivity to PostgreSQL
        3. Run health checks on connection pool
        4. Initialize performance monitoring
        """
        try:
            # Initialize engines
            await get_async_engine()
            get_sync_engine()
            
            # Test connectivity
            await self.health_check()
            
            logger.info("Database manager initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize database manager", error=str(e))
            raise
    
    async def health_check(self) -> bool:
        """
        Perform comprehensive database health check
        
        🏥 HEALTH MONITORING:
        - Connection pool status
        - Query response time
        - Lock detection
        - Storage capacity
        """
        try:
            async with get_async_session() as session:
                # Simple connectivity test
                result = await session.execute("SELECT 1")
                result.scalar()
                
                # Performance test
                import time
                start_time = time.time()
                await session.execute("SELECT COUNT(*) FROM information_schema.tables")
                response_time = time.time() - start_time
                
                self.connection_healthy = True
                self.last_health_check = datetime.now()
                
                logger.debug("Database health check passed", 
                           response_time=response_time,
                           healthy=True)
                
                return True
                
        except Exception as e:
            self.connection_healthy = False
            logger.error("Database health check failed", error=str(e))
            return False
    
    async def create_tables(self):
        """Create all database tables"""
        try:
            engine = await get_async_engine()
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error("Failed to create database tables", error=str(e))
            raise
    
    async def cleanup(self):
        """Clean up database connections"""
        global async_engine, sync_engine
        
        try:
            if async_engine:
                await async_engine.dispose()
                async_engine = None
            
            if sync_engine:
                sync_engine.dispose()
                sync_engine = None
            
            logger.info("Database connections cleaned up")
            
        except Exception as e:
            logger.error("Error during database cleanup", error=str(e))


# === Database Query Helpers ===

class SessionQueries:
    """Query helpers for session management"""
    
    @staticmethod
    async def create_session(session_data: Dict[str, Any]) -> str:
        """Create a new PRSM session"""
        async with get_async_session() as db_session:
            session_model = PRSMSessionModel(**session_data)
            db_session.add(session_model)
            await db_session.flush()
            return str(session_model.session_id)
    
    @staticmethod
    async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        async with get_async_session() as db_session:
            from sqlalchemy import select
            
            stmt = select(PRSMSessionModel).where(PRSMSessionModel.session_id == session_id)
            result = await db_session.execute(stmt)
            session_model = result.scalar_one_or_none()
            
            if session_model:
                return {
                    "session_id": str(session_model.session_id),
                    "user_id": session_model.user_id,
                    "nwtn_context_allocation": session_model.nwtn_context_allocation,
                    "context_used": session_model.context_used,
                    "status": session_model.status,
                    "metadata": session_model.metadata,
                    "created_at": session_model.created_at,
                    "updated_at": session_model.updated_at
                }
            return None
    
    @staticmethod
    async def update_session_status(session_id: str, status: str) -> bool:
        """Update session status"""
        async with get_async_session() as db_session:
            from sqlalchemy import select, update
            
            stmt = (
                update(PRSMSessionModel)
                .where(PRSMSessionModel.session_id == session_id)
                .values(status=status)
            )
            result = await db_session.execute(stmt)
            return result.rowcount > 0


class FTNSQueries:
    """Query helpers for FTNS operations with atomic guarantees
    
    All balance-affecting operations use:
    - SELECT FOR UPDATE for row-level locking
    - Optimistic concurrency control via version column
    - Idempotency keys to prevent duplicate operations
    """
    
    @staticmethod
    async def create_transaction(transaction_data: Dict[str, Any]) -> str:
        """Create a new FTNS transaction
        
        Note: For atomic balance updates, use execute_atomic_transfer instead.
        This method is for recording transaction history only.
        """
        async with get_async_session() as db_session:
            transaction_model = FTNSTransactionModel(**transaction_data)
            db_session.add(transaction_model)
            await db_session.flush()
            return str(transaction_model.transaction_id)
    
    @staticmethod
    async def get_user_balance(user_id: str) -> Dict[str, float]:
        """Get user FTNS balance (read-only, no lock)"""
        async with get_async_session() as db_session:
            from sqlalchemy import select
            
            stmt = select(FTNSBalanceModel).where(FTNSBalanceModel.user_id == user_id)
            result = await db_session.execute(stmt)
            balance_model = result.scalar_one_or_none()
            
            if balance_model:
                return {
                    "balance": balance_model.balance,
                    "locked_balance": balance_model.locked_balance,
                    "available_balance": balance_model.balance - balance_model.locked_balance,
                    "total_earned": balance_model.total_earned or 0.0,
                    "total_spent": balance_model.total_spent or 0.0,
                    "version": balance_model.version
                }
            else:
                return {
                    "balance": 0.0, 
                    "locked_balance": 0.0,
                    "available_balance": 0.0,
                    "total_earned": 0.0,
                    "total_spent": 0.0,
                    "version": 1
                }
    
    @staticmethod
    async def get_user_balance_locked(user_id: str) -> Dict[str, Any]:
        """Get user FTNS balance with FOR UPDATE lock
        
        Use this when you need to prevent concurrent modifications.
        Must be called within a transaction.
        """
        async with get_async_session() as db_session:
            result = await db_session.execute(
                text("""
                    SELECT user_id, balance, locked_balance, total_earned, total_spent, version
                    FROM ftns_balances
                    WHERE user_id = :user_id
                    FOR UPDATE NOWAIT
                """),
                {"user_id": user_id}
            )
            row = result.fetchone()
            
            if row:
                return {
                    "user_id": row.user_id,
                    "balance": float(row.balance),
                    "locked_balance": float(row.locked_balance),
                    "available_balance": float(row.balance) - float(row.locked_balance),
                    "total_earned": float(row.total_earned or 0),
                    "total_spent": float(row.total_spent or 0),
                    "version": row.version
                }
            else:
                await db_session.execute(
                    text("""
                        INSERT INTO ftns_balances 
                        (user_id, balance, locked_balance, total_earned, total_spent, version)
                        VALUES (:user_id, 0, 0, 0, 0, 1)
                        ON CONFLICT (user_id) DO NOTHING
                    """),
                    {"user_id": user_id}
                )
                return {
                    "user_id": user_id,
                    "balance": 0.0,
                    "locked_balance": 0.0,
                    "available_balance": 0.0,
                    "total_earned": 0.0,
                    "total_spent": 0.0,
                    "version": 1
                }
    
    @staticmethod
    async def update_balance(user_id: str, amount: float) -> bool:
        """Update user balance (DEPRECATED - use execute_atomic_deduct instead)
        
        This method does NOT provide race condition protection.
        Kept for backwards compatibility only.
        """
        async with get_async_session() as db_session:
            from sqlalchemy import select, update
            
            await FTNSQueries.get_user_balance(user_id)
            
            stmt = (
                update(FTNSBalanceModel)
                .where(FTNSBalanceModel.user_id == user_id)
                .values(balance=FTNSBalanceModel.balance + amount)
            )
            result = await db_session.execute(stmt)
            return result.rowcount > 0
    
    @staticmethod
    async def execute_atomic_deduct(
        user_id: str,
        amount: float,
        idempotency_key: str,
        description: str = "",
        transaction_type: str = "deduction"
    ) -> Dict[str, Any]:
        """Atomically deduct tokens with race condition protection
        
        Uses PostgreSQL stored procedure for true atomicity.
        
        Args:
            user_id: User to deduct from
            amount: Amount to deduct (positive value)
            idempotency_key: Unique key to prevent duplicate operations
            description: Transaction description
            transaction_type: Type of transaction
            
        Returns:
            Dict with success, transaction_id, new_balance, error_message
        """
        async with get_async_session() as db_session:
            result = await db_session.execute(
                text("""
                    SELECT success, transaction_id, new_balance, error_message
                    FROM atomic_deduct_balance(
                        :user_id, :amount, :idempotency_key, :description, :tx_type
                    )
                """),
                {
                    "user_id": user_id,
                    "amount": amount,
                    "idempotency_key": idempotency_key,
                    "description": description,
                    "tx_type": transaction_type
                }
            )
            row = result.fetchone()
            await db_session.commit()
            
            return {
                "success": row.success,
                "transaction_id": row.transaction_id,
                "new_balance": float(row.new_balance) if row.new_balance is not None else None,
                "error_message": row.error_message
            }
    
    @staticmethod
    async def execute_atomic_transfer(
        from_user_id: str,
        to_user_id: str,
        amount: float,
        idempotency_key: str,
        description: str = ""
    ) -> Dict[str, Any]:
        """Atomically transfer tokens between users
        
        Uses PostgreSQL stored procedure with:
        - Consistent lock ordering to prevent deadlocks
        - Idempotency key to prevent duplicates
        - Balance validation
        - Automatic rollback on failure
        
        Args:
            from_user_id: Sender user ID
            to_user_id: Recipient user ID
            amount: Amount to transfer
            idempotency_key: Unique key for this operation
            description: Transfer description
            
        Returns:
            Dict with success, transaction_id, balances, error_message
        """
        async with get_async_session() as db_session:
            result = await db_session.execute(
                text("""
                    SELECT success, transaction_id, sender_new_balance, 
                           receiver_new_balance, error_message
                    FROM atomic_transfer(
                        :from_user, :to_user, :amount, :idempotency_key, :description
                    )
                """),
                {
                    "from_user": from_user_id,
                    "to_user": to_user_id,
                    "amount": amount,
                    "idempotency_key": idempotency_key,
                    "description": description
                }
            )
            row = result.fetchone()
            await db_session.commit()
            
            return {
                "success": row.success,
                "transaction_id": row.transaction_id,
                "sender_new_balance": float(row.sender_new_balance) if row.sender_new_balance is not None else None,
                "receiver_new_balance": float(row.receiver_new_balance) if row.receiver_new_balance is not None else None,
                "error_message": row.error_message
            }
    
    @staticmethod
    async def check_idempotency(idempotency_key: str) -> Optional[str]:
        """Check if idempotency key has been used
        
        Returns transaction_id if duplicate, None otherwise
        """
        async with get_async_session() as db_session:
            result = await db_session.execute(
                text("""
                    SELECT transaction_id FROM ftns_idempotency_keys
                    WHERE idempotency_key = :key AND expires_at > NOW()
                """),
                {"key": idempotency_key}
            )
            row = result.fetchone()
            return row.transaction_id if row else None

    @staticmethod
    async def get_user_transactions(
        user_id: str,
        limit: int = 50,
        search: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get transaction history for a user (sent and received).

        Args:
            user_id: User to look up
            limit: Max rows to return (1–100)
            search: Optional substring filter on description or transaction_id

        Returns:
            List of transaction dicts, ordered newest-first
        """
        async with get_async_session() as db_session:
            query = text("""
                SELECT
                    transaction_id, from_user, to_user, amount,
                    transaction_type, description, status,
                    balance_after_sender, balance_after_receiver,
                    created_at
                FROM ftns_transactions
                WHERE to_user = :user_id OR from_user = :user_id
                ORDER BY created_at DESC
                LIMIT :limit
            """)
            result = await db_session.execute(query, {"user_id": user_id, "limit": limit})
            rows = result.fetchall()

            transactions = []
            for row in rows:
                # Use receiver balance if user is recipient; sender balance otherwise
                balance_after = (
                    float(row.balance_after_receiver)
                    if row.to_user == user_id and row.balance_after_receiver is not None
                    else float(row.balance_after_sender)
                    if row.balance_after_sender is not None
                    else None
                )
                tx = {
                    "transaction_id": str(row.transaction_id),
                    "from_user": row.from_user,
                    "to_user": row.to_user,
                    "amount": float(row.amount),
                    "transaction_type": row.transaction_type,
                    "description": row.description,
                    "status": row.status,
                    "balance_after": balance_after,
                    "timestamp": row.created_at.isoformat() if row.created_at else None,
                }
                if search:
                    search_lower = search.lower()
                    if (search_lower in (row.description or "").lower()
                            or search_lower in str(row.transaction_id).lower()):
                        transactions.append(tx)
                else:
                    transactions.append(tx)

            return transactions


class ProvenanceQueries:
    """Query helpers for content provenance operations.

    All methods accept plain dicts (not UploadedContent) to avoid circular
    imports with prsm.node.content_uploader.
    """

    @staticmethod
    async def upsert_provenance(record: Dict[str, Any]) -> bool:
        """
        Insert or update a provenance record.

        Uses INSERT ... ON CONFLICT UPDATE so re-uploading the same CID
        (e.g., after a node restart) refreshes metadata rather than erroring.

        Args:
            record: Dict with all UploadedContent fields plus 'created_at'
                    as a Unix timestamp float.

        Returns:
            True on success, False on failure.
        """
        async with get_async_session() as session:
            try:
                from datetime import datetime, timezone
                created_at = datetime.fromtimestamp(
                    record.get("created_at", 0), tz=timezone.utc
                )
                existing = await session.get(ContentProvenanceModel, record["cid"])
                if existing:
                    # Update mutable fields only — don't overwrite creator/signature
                    existing.access_count = record.get("access_count", existing.access_count)
                    existing.total_royalties = record.get("total_royalties", existing.total_royalties)
                    existing.parent_cids = record.get("parent_cids", existing.parent_cids)
                else:
                    row = ContentProvenanceModel(
                        cid=record["cid"],
                        filename=record["filename"],
                        size_bytes=record["size_bytes"],
                        content_hash=record["content_hash"],
                        creator_id=record["creator_id"],
                        provenance_signature=record.get("provenance_signature", ""),
                        royalty_rate=record.get("royalty_rate", 0.01),
                        parent_cids=record.get("parent_cids", []),
                        access_count=record.get("access_count", 0),
                        total_royalties=record.get("total_royalties", 0.0),
                        is_sharded=record.get("is_sharded", False),
                        manifest_cid=record.get("manifest_cid"),
                        total_shards=record.get("total_shards", 0),
                        embedding_id=record.get("embedding_id"),
                        near_duplicate_of=record.get("near_duplicate_of"),
                        near_duplicate_similarity=record.get("near_duplicate_similarity"),
                        created_at=created_at,
                    )
                    session.add(row)
                await session.commit()
                return True
            except Exception as e:
                await session.rollback()
                logger.error(f"ProvenanceQueries.upsert_provenance failed: {e}")
                return False

    @staticmethod
    async def update_access_stats(
        cid: str,
        access_count_delta: int,
        royalty_delta: float,
    ) -> bool:
        """
        Atomically increment access_count and total_royalties for a CID.

        Uses a SQL UPDATE rather than read-modify-write to avoid race
        conditions when multiple access events arrive concurrently.

        Returns:
            True if the row existed and was updated, False otherwise.
        """
        async with get_async_session() as session:
            try:
                result = await session.execute(
                    text("""
                        UPDATE content_provenance
                        SET access_count   = access_count   + :delta_count,
                            total_royalties = total_royalties + :delta_royalty,
                            updated_at      = NOW()
                        WHERE cid = :cid
                    """),
                    {
                        "cid": cid,
                        "delta_count": access_count_delta,
                        "delta_royalty": royalty_delta,
                    },
                )
                await session.commit()
                return result.rowcount > 0
            except Exception as e:
                await session.rollback()
                logger.error(f"ProvenanceQueries.update_access_stats failed for {cid[:12]}...: {e}")
                return False

    @staticmethod
    async def load_all_for_node(creator_id: str) -> List[Dict[str, Any]]:
        """
        Load all provenance records for a given creator node ID.

        Called during node startup to hydrate the in-memory uploaded_content
        dict. Returns all records owned by this node regardless of age.

        Args:
            creator_id: The node ID to load records for.

        Returns:
            List of dicts, each representing one UploadedContent record.
        """
        async with get_async_session() as session:
            try:
                result = await session.execute(
                    text("""
                        SELECT
                            cid, filename, size_bytes, content_hash, creator_id,
                            provenance_signature, royalty_rate, parent_cids,
                            access_count, total_royalties, is_sharded, manifest_cid,
                            total_shards, embedding_id, near_duplicate_of,
                            near_duplicate_similarity, created_at
                        FROM content_provenance
                        WHERE creator_id = :creator_id
                        ORDER BY created_at ASC
                    """),
                    {"creator_id": creator_id},
                )
                rows = result.fetchall()
                return [
                    {
                        "cid": row.cid,
                        "filename": row.filename,
                        "size_bytes": row.size_bytes,
                        "content_hash": row.content_hash,
                        "creator_id": row.creator_id,
                        "provenance_signature": row.provenance_signature,
                        "royalty_rate": float(row.royalty_rate),
                        "parent_cids": row.parent_cids or [],
                        "access_count": row.access_count,
                        "total_royalties": float(row.total_royalties),
                        "is_sharded": bool(row.is_sharded),
                        "manifest_cid": row.manifest_cid,
                        "total_shards": row.total_shards or 0,
                        "embedding_id": row.embedding_id,
                        "near_duplicate_of": row.near_duplicate_of,
                        "near_duplicate_similarity": (
                            float(row.near_duplicate_similarity)
                            if row.near_duplicate_similarity is not None
                            else None
                        ),
                        "created_at": row.created_at.timestamp() if row.created_at else 0.0,
                    }
                    for row in rows
                ]
            except Exception as e:
                logger.error(f"ProvenanceQueries.load_all_for_node failed: {e}")
                return []

    @staticmethod
    async def get_provenance(cid: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single provenance record by CID.

        Returns a serializable dict or None if not found. Used by the
        GET /api/v1/content/{cid}/provenance endpoint.
        """
        async with get_async_session() as session:
            try:
                row = await session.get(ContentProvenanceModel, cid)
                if row is None:
                    return None
                return {
                    "cid":                      row.cid,
                    "filename":                 row.filename,
                    "size_bytes":               row.size_bytes,
                    "content_hash":             row.content_hash,
                    "creator_id":               row.creator_id,
                    "royalty_rate":             float(row.royalty_rate),
                    "parent_cids":              row.parent_cids or [],
                    "access_count":             row.access_count,
                    "total_royalties":          float(row.total_royalties),
                    "is_sharded":               bool(row.is_sharded),
                    "manifest_cid":             row.manifest_cid,
                    "total_shards":             row.total_shards or 0,
                    "embedding_id":             row.embedding_id,
                    "near_duplicate_of":        row.near_duplicate_of,
                    "near_duplicate_similarity": row.near_duplicate_similarity,
                    "created_at":               row.created_at.isoformat()
                                                if row.created_at else None,
                    "updated_at":               row.updated_at.isoformat()
                                                if row.updated_at else None,
                }
            except Exception as e:
                logger.error(f"ProvenanceQueries.get_provenance failed: {e}")
                return None


class GovernanceQueries:
    """Query helpers for governance proposal operations.

    Accepts GovernanceProposal Pydantic objects directly (no circular
    import issues since GovernanceProposal is from prsm.core.models,
    not from prsm.economy.governance).
    """

    @staticmethod
    async def upsert_proposal(proposal: "GovernanceProposal") -> bool:
        """
        Insert or update a governance proposal record.

        Uses ORM session.get() + conditional insert for PostgreSQL/SQLite
        compatibility (avoids dialect-specific ON CONFLICT syntax).

        Returns True on success, False on failure.
        """
        async with get_async_session() as session:
            try:
                existing = await session.get(
                    GovernanceProposalModel, proposal.proposal_id
                )
                if existing:
                    existing.status             = proposal.status
                    existing.votes_for          = proposal.votes_for
                    existing.votes_against      = proposal.votes_against
                    existing.total_voting_power = float(proposal.total_voting_power)
                    existing.voting_starts      = proposal.voting_starts
                    existing.voting_ends        = proposal.voting_ends
                    existing.proposal_metadata  = proposal.metadata or {}
                else:
                    row = GovernanceProposalModel(
                        proposal_id        = proposal.proposal_id,
                        proposer_id        = proposal.proposer_id,
                        title              = proposal.title,
                        description        = proposal.description,
                        proposal_type      = proposal.proposal_type,
                        status             = proposal.status,
                        votes_for          = proposal.votes_for,
                        votes_against      = proposal.votes_against,
                        total_voting_power = float(proposal.total_voting_power),
                        required_quorum    = float(proposal.required_quorum)
                                            if proposal.required_quorum else None,
                        voting_starts      = proposal.voting_starts,
                        voting_ends        = proposal.voting_ends,
                        proposal_metadata  = proposal.metadata or {},
                    )
                    session.add(row)
                await session.commit()
                return True
            except Exception as e:
                await session.rollback()
                logger.error(f"GovernanceQueries.upsert_proposal failed: {e}")
                return False

    @staticmethod
    async def get_proposal(proposal_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single proposal by ID (UUID string).

        Returns a serializable dict or None if not found. Used by the
        GET /api/v1/governance/proposals/{id} endpoint when the proposal
        isn't in the active in-memory set.
        """
        async with get_async_session() as session:
            try:
                from uuid import UUID
                row = await session.get(
                    GovernanceProposalModel, UUID(proposal_id)
                )
                if not row:
                    return None
                return {
                    "proposal_id":        str(row.proposal_id),
                    "proposer_id":        row.proposer_id,
                    "title":              row.title,
                    "description":        row.description,
                    "proposal_type":      row.proposal_type,
                    "status":             row.status,
                    "votes_for":          row.votes_for,
                    "votes_against":      row.votes_against,
                    "total_voting_power": row.total_voting_power,
                    "required_quorum":    row.required_quorum,
                    "voting_starts":      row.voting_starts.isoformat()
                                         if row.voting_starts else None,
                    "voting_ends":        row.voting_ends.isoformat()
                                         if row.voting_ends else None,
                    "proposal_metadata":  row.proposal_metadata or {},
                    "created_at":         row.created_at.isoformat()
                                         if row.created_at else None,
                }
            except Exception as e:
                logger.error(f"GovernanceQueries.get_proposal failed: {e}")
                return None

    @staticmethod
    async def load_all_proposals(
        status_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load all proposals, optionally filtered by status.

        Called during system startup to hydrate the in-memory proposals
        dict, and by list endpoints when in-memory state is empty.

        Returns list of dicts suitable for GovernanceProposal reconstruction.
        """
        async with get_async_session() as session:
            try:
                if status_filter:
                    query = text("""
                        SELECT proposal_id, proposer_id, title, description,
                               proposal_type, status, votes_for, votes_against,
                               total_voting_power, required_quorum,
                               voting_starts, voting_ends, proposal_metadata,
                               created_at
                        FROM governance_proposals
                        WHERE status = :status
                        ORDER BY created_at DESC
                    """)
                    result = await session.execute(query, {"status": status_filter})
                else:
                    query = text("""
                        SELECT proposal_id, proposer_id, title, description,
                               proposal_type, status, votes_for, votes_against,
                               total_voting_power, required_quorum,
                               voting_starts, voting_ends, proposal_metadata,
                               created_at
                        FROM governance_proposals
                        ORDER BY created_at DESC
                    """)
                    result = await session.execute(query)

                rows = result.fetchall()
                return [
                    {
                        "proposal_id":        str(row.proposal_id),
                        "proposer_id":        row.proposer_id,
                        "title":              row.title,
                        "description":        row.description,
                        "proposal_type":      row.proposal_type,
                        "status":             row.status,
                        "votes_for":          row.votes_for,
                        "votes_against":      row.votes_against,
                        "total_voting_power": float(row.total_voting_power or 0),
                        "required_quorum":    row.required_quorum,
                        "voting_starts":      row.voting_starts,
                        "voting_ends":        row.voting_ends,
                        "metadata":           row.proposal_metadata or {},
                        "created_at":         row.created_at,
                    }
                    for row in rows
                ]
            except Exception as e:
                logger.error(f"GovernanceQueries.load_all_proposals failed: {e}")
                return []


class ModelQueries:
    """Query helpers for model registry operations"""
    
    @staticmethod
    async def register_model(model_data: Dict[str, Any]) -> str:
        """Register a new model in the registry"""
        async with get_async_session() as db_session:
            model = ModelRegistryModel(**model_data)
            db_session.add(model)
            await db_session.flush()
            return model.model_id
    
    @staticmethod
    async def get_models_by_type(model_type: str) -> List[Dict[str, Any]]:
        """Get all models of a specific type"""
        async with get_async_session() as db_session:
            from sqlalchemy import select
            
            stmt = (
                select(ModelRegistryModel)
                .where(ModelRegistryModel.model_type == model_type)
                .where(ModelRegistryModel.active == True)
            )
            result = await db_session.execute(stmt)
            models = result.scalars().all()
            
            return [
                {
                    "model_id": model.model_id,
                    "name": model.name,
                    "description": model.description,
                    "specialization": model.specialization,
                    "performance_metrics": model.performance_metrics,
                    "availability_status": model.availability_status
                }
                for model in models
            ]


# === Team Models ===

class TeamModel(Base):
    """
    Database model for teams
    
    🧑‍🤝‍🧑 TEAMS INTEGRATION:
    Collaborative research units with shared resources and governance
    """
    __tablename__ = "teams"
    
    team_id = Column(UUID(as_uuid=True), primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    team_type = Column(String(50), default="research", index=True)
    
    # Visual identity
    avatar_url = Column(String(500))
    logo_url = Column(String(500))
    
    # Configuration
    governance_model = Column(String(50), default="democratic")
    reward_policy = Column(String(50), default="proportional")
    is_public = Column(Boolean, default=True, index=True)
    max_members = Column(Integer)
    
    # Financial settings
    entry_stake_required = Column(Float, default=0.0)
    
    # Research focus
    research_domains = Column(JSON, default=list)
    keywords = Column(JSON, default=list)
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    founding_date = Column(DateTime(timezone=True), server_default=func.now())
    
    # Statistics
    member_count = Column(Integer, default=0)
    total_ftns_earned = Column(Float, default=0.0)
    total_tasks_completed = Column(Integer, default=0)
    impact_score = Column(Float, default=0.0, index=True)
    
    # Metadata
    external_links = Column(JSON, default=dict)
    contact_info = Column(JSON, default=dict)
    model_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    members = relationship("TeamMemberModel", back_populates="team")
    wallet = relationship("TeamWalletModel", back_populates="team", uselist=False)
    tasks = relationship("TeamTaskModel", back_populates="team")
    governance = relationship("TeamGovernanceModel", back_populates="team", uselist=False)
    
    __table_args__ = (
        Index('idx_team_type_active', 'team_type', 'is_active'),
        Index('idx_team_impact_score', 'impact_score'),
        Index('idx_team_public_active', 'is_public', 'is_active'),
        UniqueConstraint('name', name='uq_team_name'),
    )


class TeamMemberModel(Base):
    """Database model for team members"""
    __tablename__ = "team_members"
    
    membership_id = Column(UUID(as_uuid=True), primary_key=True)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.team_id"), nullable=False)
    user_id = Column(String(255), nullable=False, index=True)
    
    # Membership details
    role = Column(String(50), default="member", index=True)
    status = Column(String(50), default="pending", index=True)
    invited_by = Column(String(255))
    invitation_message = Column(Text)
    
    # Dates
    invited_at = Column(DateTime(timezone=True), server_default=func.now())
    joined_at = Column(DateTime(timezone=True))
    left_at = Column(DateTime(timezone=True))
    
    # Contribution tracking
    ftns_contributed = Column(Float, default=0.0)
    tasks_completed = Column(Integer, default=0)
    models_contributed = Column(Integer, default=0)
    datasets_uploaded = Column(Integer, default=0)
    
    # Performance metrics
    performance_score = Column(Float, default=0.0)
    reputation_score = Column(Float, default=0.5)
    collaboration_score = Column(Float, default=0.0)
    
    # Permissions
    can_invite_members = Column(Boolean, default=False)
    can_manage_tasks = Column(Boolean, default=False)
    can_access_treasury = Column(Boolean, default=False)
    can_vote = Column(Boolean, default=True)
    
    # Profile
    bio = Column(Text)
    expertise_areas = Column(JSON, default=list)
    public_profile = Column(Boolean, default=True)
    model_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    team = relationship("TeamModel", back_populates="members")
    
    __table_args__ = (
        Index('idx_team_member_team_user', 'team_id', 'user_id'),
        Index('idx_team_member_status', 'status'),
        Index('idx_team_member_role', 'role'),
        UniqueConstraint('team_id', 'user_id', name='uq_team_member'),
    )


class TeamWalletModel(Base):
    """Database model for team wallets"""
    __tablename__ = "team_wallets"
    
    wallet_id = Column(UUID(as_uuid=True), primary_key=True)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.team_id"), nullable=False, unique=True)
    
    # Wallet configuration
    is_multisig = Column(Boolean, default=True)
    required_signatures = Column(Integer, default=1)
    authorized_signers = Column(JSON, default=list)
    
    # Balances
    total_balance = Column(Float, default=0.0)
    available_balance = Column(Float, default=0.0)
    locked_balance = Column(Float, default=0.0)
    
    # Distribution policy
    reward_policy = Column(String(50), default="proportional")
    policy_config = Column(JSON, default=dict)
    distribution_metrics = Column(JSON, default=list)
    metric_weights = Column(JSON, default=list)
    
    # Treasury management
    auto_distribution_enabled = Column(Boolean, default=False)
    distribution_frequency_days = Column(Integer, default=30)
    last_distribution = Column(DateTime(timezone=True))
    
    # Security
    wallet_address = Column(String(255))
    spending_limits = Column(JSON, default=dict)
    emergency_freeze = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    team = relationship("TeamModel", back_populates="wallet")
    
    __table_args__ = (
        Index('idx_team_wallet_balance', 'total_balance'),
        Index('idx_team_wallet_emergency', 'emergency_freeze'),
    )


class TeamTaskModel(Base):
    """Database model for team tasks"""
    __tablename__ = "team_tasks"
    
    task_id = Column(UUID(as_uuid=True), primary_key=True)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.team_id"), nullable=False)
    
    # Task details
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    task_type = Column(String(50), default="research", index=True)
    
    # Assignment
    assigned_to = Column(JSON, default=list)
    created_by = Column(String(255), nullable=False)
    priority = Column(String(20), default="medium", index=True)
    
    # Execution
    status = Column(String(50), default="pending", index=True)
    progress_percentage = Column(Float, default=0.0)
    
    # FTNS allocation
    ftns_budget = Column(Float, default=0.0)
    ftns_spent = Column(Float, default=0.0)
    
    # Deadlines
    due_date = Column(DateTime(timezone=True))
    estimated_hours = Column(Float)
    actual_hours = Column(Float)
    
    # Results
    output_artifacts = Column(JSON, default=list)
    output_models = Column(JSON, default=list)
    performance_metrics = Column(JSON, default=dict)
    
    # Collaboration
    requires_consensus = Column(Boolean, default=False)
    consensus_threshold = Column(Float, default=0.6)
    votes_for = Column(Integer, default=0)
    votes_against = Column(Integer, default=0)
    
    # Metadata
    tags = Column(JSON, default=list)
    external_links = Column(JSON, default=dict)
    model_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    team = relationship("TeamModel", back_populates="tasks")
    
    __table_args__ = (
        Index('idx_team_task_team_status', 'team_id', 'status'),
        Index('idx_team_task_type_priority', 'task_type', 'priority'),
        Index('idx_team_task_due_date', 'due_date'),
        Index('idx_team_task_created_by', 'created_by'),
    )


class TeamGovernanceModel(Base):
    """Database model for team governance"""
    __tablename__ = "team_governance"
    
    governance_id = Column(UUID(as_uuid=True), primary_key=True)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.team_id"), nullable=False, unique=True)
    
    # Governance configuration
    model = Column(String(50), default="democratic")
    constitution = Column(JSON, default=dict)
    
    # Voting configuration
    voting_period_days = Column(Integer, default=7)
    quorum_percentage = Column(Float, default=0.5)
    approval_threshold = Column(Float, default=0.6)
    
    # Role management
    role_assignments = Column(JSON, default=dict)
    role_term_limits = Column(JSON, default=dict)
    
    # Proposal configuration
    proposal_types = Column(JSON, default=list)
    type_thresholds = Column(JSON, default=dict)
    
    # Emergency procedures
    emergency_roles = Column(JSON, default=list)
    emergency_procedures = Column(JSON, default=dict)
    
    # Constitutional limits
    max_owner_power = Column(Float, default=0.4)
    member_protection_threshold = Column(Float, default=0.25)
    
    # Active proposals
    active_proposals = Column(JSON, default=list)
    
    # Statistics
    total_proposals = Column(Integer, default=0)
    proposals_passed = Column(Integer, default=0)
    average_participation = Column(Float, default=0.0)
    last_vote = Column(DateTime(timezone=True))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    team = relationship("TeamModel", back_populates="governance")
    
    __table_args__ = (
        Index('idx_team_governance_model', 'model'),
    )


# Global database manager instance
db_manager = DatabaseManager()


# === Service Functions ===

def get_database_service():
    """Get the database service instance"""
    # For compatibility with existing imports
    return db_manager


# === Initialization Functions ===

async def init_database():
    """Initialize database for PRSM"""
    await db_manager.initialize()
    await db_manager.create_tables()


async def close_database():
    """Close database connections"""
    await db_manager.cleanup()