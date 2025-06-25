# PostgreSQL Integration Guide

Integrate PRSM with PostgreSQL for robust, scalable data persistence and advanced database features.

## 🎯 Overview

This guide covers integrating PRSM with PostgreSQL, including setup, connection management, schema design, performance optimization, and production best practices.

## 📋 Prerequisites

- PostgreSQL 12+ installed
- PRSM instance configured
- Basic knowledge of SQL and database concepts
- Python development environment

## 🚀 Quick Start

### 1. PostgreSQL Setup

```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt update
sudo apt install postgresql postgresql-contrib

# Install PostgreSQL (macOS with Homebrew)
brew install postgresql
brew services start postgresql

# Install PostgreSQL (CentOS/RHEL)
sudo yum install postgresql-server postgresql-contrib
sudo postgresql-setup initdb
sudo systemctl start postgresql
```

### 2. Database and User Creation

```sql
-- Connect as postgres user
sudo -u postgres psql

-- Create database
CREATE DATABASE prsm_production;

-- Create user
CREATE USER prsm_user WITH ENCRYPTED PASSWORD 'secure_password_here';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE prsm_production TO prsm_user;

-- Grant additional privileges for schema management
ALTER USER prsm_user CREATEDB;

-- Exit psql
\q
```

### 3. Basic Connection Test

```python
# test_connection.py
import psycopg2
from psycopg2 import sql

def test_connection():
    """Test PostgreSQL connection."""
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="prsm_production",
            user="prsm_user",
            password="secure_password_here"
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"PostgreSQL version: {version[0]}")
        
        cursor.close()
        conn.close()
        
        print("Connection successful!")
        return True
        
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()
```

## 🔧 PRSM Database Configuration

### SQLAlchemy Integration

```python
# prsm/core/database.py
import os
import logging
from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import asyncio
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

# Database configuration
class DatabaseConfig:
    def __init__(self):
        self.database_url = os.environ.get('DATABASE_URL', 
            'postgresql://prsm_user:secure_password_here@localhost/prsm_production')
        self.echo = os.environ.get('DB_ECHO', 'false').lower() == 'true'
        self.pool_size = int(os.environ.get('DB_POOL_SIZE', '20'))
        self.max_overflow = int(os.environ.get('DB_MAX_OVERFLOW', '30'))
        self.pool_timeout = int(os.environ.get('DB_POOL_TIMEOUT', '30'))
        self.pool_recycle = int(os.environ.get('DB_POOL_RECYCLE', '3600'))

# Base model class
Base = declarative_base()

class DatabaseManager:
    def __init__(self, config: DatabaseConfig = None):
        self.config = config or DatabaseConfig()
        self.engine = None
        self.async_engine = None
        self.session_factory = None
        self.async_session_factory = None
        self.logger = logging.getLogger(__name__)

    def create_engine(self):
        """Create synchronous database engine."""
        self.engine = create_engine(
            self.config.database_url,
            echo=self.config.echo,
            poolclass=QueuePool,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            pool_pre_ping=True,
            connect_args={
                "options": "-c timezone=utc",
                "application_name": "prsm-api"
            }
        )
        
        # Add connection event listeners
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            if 'postgresql' in self.config.database_url:
                with dbapi_connection.cursor() as cursor:
                    cursor.execute("SET timezone = 'UTC'")
                    cursor.execute("SET statement_timeout = '30s'")

        self.session_factory = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False
        )
        
        return self.engine

    def create_async_engine(self):
        """Create asynchronous database engine."""
        # Convert PostgreSQL URL for asyncpg
        async_url = self.config.database_url.replace(
            'postgresql://', 'postgresql+asyncpg://'
        )
        
        self.async_engine = create_async_engine(
            async_url,
            echo=self.config.echo,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            pool_pre_ping=True
        )
        
        self.async_session_factory = sessionmaker(
            self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        return self.async_engine

    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    async def get_async_session(self):
        """Get async database session."""
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                self.logger.error(f"Async database session error: {e}")
                raise

    def health_check(self):
        """Check database connectivity."""
        try:
            with self.get_session() as session:
                result = session.execute(text("SELECT 1")).scalar()
                return {"status": "healthy", "database": "connected"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def async_health_check(self):
        """Async health check."""
        try:
            async with self.get_async_session() as session:
                result = await session.execute(text("SELECT 1"))
                return {"status": "healthy", "database": "connected"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

# Global database manager instance
db_manager = DatabaseManager()
```

### Database Models

```python
# prsm/models/core.py
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, JSON, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime
import uuid
from prsm.core.database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True, nullable=False)
    metadata_ = Column(JSONB, default=dict)
    
    # Relationships
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    query_logs = relationship("QueryLog", back_populates="user")
    
    # Indexes
    __table_args__ = (
        Index('idx_users_email', 'email'),
        Index('idx_users_username', 'username'),
        Index('idx_users_created_at', 'created_at'),
    )

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    title = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_archived = Column(Boolean, default=False, nullable=False)
    metadata_ = Column(JSONB, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_conversations_user_id', 'user_id'),
        Index('idx_conversations_created_at', 'created_at'),
        Index('idx_conversations_updated_at', 'updated_at'),
    )

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    content = Column(Text, nullable=False)
    role = Column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    metadata_ = Column(JSONB, default=dict)
    
    # AI-specific fields
    confidence = Column(Float)
    tokens_used = Column(Integer)
    processing_time = Column(Float)
    model_version = Column(String(50))
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    # Indexes
    __table_args__ = (
        Index('idx_messages_conversation_id', 'conversation_id'),
        Index('idx_messages_created_at', 'created_at'),
        Index('idx_messages_role', 'role'),
    )

class QueryLog(Base):
    __tablename__ = "query_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    prompt = Column(Text, nullable=False)
    response = Column(Text)
    status = Column(String(20), nullable=False)  # 'success', 'error', 'timeout'
    error_message = Column(Text)
    
    # Performance metrics
    response_time = Column(Float)
    tokens_used = Column(Integer)
    cost = Column(Float)
    confidence = Column(Float)
    
    # Request context
    ip_address = Column(String(45))
    user_agent = Column(Text)
    request_id = Column(String(100))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Additional metadata
    metadata_ = Column(JSONB, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="query_logs")
    
    # Indexes
    __table_args__ = (
        Index('idx_query_logs_user_id', 'user_id'),
        Index('idx_query_logs_created_at', 'created_at'),
        Index('idx_query_logs_status', 'status'),
        Index('idx_query_logs_request_id', 'request_id'),
    )

class SystemMetrics(Base):
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_type = Column(String(20), nullable=False)  # 'counter', 'gauge', 'histogram'
    labels = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_system_metrics_name_created', 'metric_name', 'created_at'),
        Index('idx_system_metrics_created_at', 'created_at'),
    )
```

### Repository Pattern

```python
# prsm/repositories/base.py
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from prsm.core.database import db_manager

class BaseRepository:
    def __init__(self, model_class):
        self.model_class = model_class

    def create(self, session: Session, **kwargs) -> Any:
        """Create a new record."""
        instance = self.model_class(**kwargs)
        session.add(instance)
        session.flush()
        return instance

    def get_by_id(self, session: Session, id: Any) -> Optional[Any]:
        """Get record by ID."""
        return session.query(self.model_class).filter(
            self.model_class.id == id
        ).first()

    def get_all(self, session: Session, limit: int = 100, offset: int = 0) -> List[Any]:
        """Get all records with pagination."""
        return session.query(self.model_class)\
            .offset(offset)\
            .limit(limit)\
            .all()

    def update(self, session: Session, id: Any, **kwargs) -> Optional[Any]:
        """Update record by ID."""
        instance = self.get_by_id(session, id)
        if instance:
            for key, value in kwargs.items():
                setattr(instance, key, value)
            session.flush()
        return instance

    def delete(self, session: Session, id: Any) -> bool:
        """Delete record by ID."""
        instance = self.get_by_id(session, id)
        if instance:
            session.delete(instance)
            session.flush()
            return True
        return False

    def count(self, session: Session, **filters) -> int:
        """Count records with optional filters."""
        query = session.query(self.model_class)
        for key, value in filters.items():
            if hasattr(self.model_class, key):
                query = query.filter(getattr(self.model_class, key) == value)
        return query.count()

# prsm/repositories/conversation.py
from typing import List, Optional
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import desc
from datetime import datetime, timedelta
from prsm.models.core import Conversation, Message, User
from prsm.repositories.base import BaseRepository

class ConversationRepository(BaseRepository):
    def __init__(self):
        super().__init__(Conversation)

    def get_user_conversations(
        self, 
        session: Session, 
        user_id: str, 
        limit: int = 50,
        include_archived: bool = False
    ) -> List[Conversation]:
        """Get conversations for a specific user."""
        query = session.query(Conversation).filter(
            Conversation.user_id == user_id
        )
        
        if not include_archived:
            query = query.filter(Conversation.is_archived == False)
        
        return query.options(
            joinedload(Conversation.messages)
        ).order_by(
            desc(Conversation.updated_at)
        ).limit(limit).all()

    def get_conversation_with_messages(
        self, 
        session: Session, 
        conversation_id: str,
        message_limit: int = 100
    ) -> Optional[Conversation]:
        """Get conversation with its messages."""
        conversation = session.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if conversation:
            # Load messages separately for better control
            messages = session.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at).limit(message_limit).all()
            
            conversation.messages = messages
        
        return conversation

    def archive_conversation(self, session: Session, conversation_id: str) -> bool:
        """Archive a conversation."""
        conversation = self.get_by_id(session, conversation_id)
        if conversation:
            conversation.is_archived = True
            conversation.updated_at = datetime.utcnow()
            session.flush()
            return True
        return False

    def search_conversations(
        self,
        session: Session,
        user_id: str,
        search_term: str,
        limit: int = 20
    ) -> List[Conversation]:
        """Search conversations by title or message content."""
        # Search in conversation titles and message content
        return session.query(Conversation).join(Message).filter(
            and_(
                Conversation.user_id == user_id,
                or_(
                    Conversation.title.ilike(f'%{search_term}%'),
                    Message.content.ilike(f'%{search_term}%')
                )
            )
        ).distinct().order_by(
            desc(Conversation.updated_at)
        ).limit(limit).all()

# Initialize repositories
conversation_repo = ConversationRepository()
```

### Database Services

```python
# prsm/services/database_service.py
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError
from prsm.core.database import db_manager
from prsm.models.core import User, Conversation, Message, QueryLog
from prsm.repositories.conversation import conversation_repo

class DatabaseService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_user(self, username: str, email: str, **metadata) -> Optional[User]:
        """Create a new user."""
        try:
            with db_manager.get_session() as session:
                user = User(
                    username=username,
                    email=email,
                    metadata_=metadata
                )
                session.add(user)
                session.flush()
                return user
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to create user: {e}")
            return None

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        try:
            with db_manager.get_session() as session:
                return session.query(User).filter(
                    User.email == email
                ).first()
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to get user by email: {e}")
            return None

    def create_conversation(
        self, 
        user_id: str, 
        title: str = None,
        **metadata
    ) -> Optional[Conversation]:
        """Create a new conversation."""
        try:
            with db_manager.get_session() as session:
                conversation = Conversation(
                    user_id=user_id,
                    title=title or f"Conversation {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                    metadata_=metadata
                )
                session.add(conversation)
                session.flush()
                return conversation
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to create conversation: {e}")
            return None

    def add_message(
        self,
        conversation_id: str,
        content: str,
        role: str,
        confidence: float = None,
        tokens_used: int = None,
        processing_time: float = None,
        **metadata
    ) -> Optional[Message]:
        """Add a message to a conversation."""
        try:
            with db_manager.get_session() as session:
                message = Message(
                    conversation_id=conversation_id,
                    content=content,
                    role=role,
                    confidence=confidence,
                    tokens_used=tokens_used,
                    processing_time=processing_time,
                    metadata_=metadata
                )
                session.add(message)
                
                # Update conversation timestamp
                conversation = session.query(Conversation).filter(
                    Conversation.id == conversation_id
                ).first()
                if conversation:
                    conversation.updated_at = datetime.utcnow()
                
                session.flush()
                return message
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to add message: {e}")
            return None

    def log_query(
        self,
        user_id: str,
        prompt: str,
        response: str = None,
        status: str = "success",
        error_message: str = None,
        response_time: float = None,
        tokens_used: int = None,
        cost: float = None,
        confidence: float = None,
        **context
    ) -> Optional[QueryLog]:
        """Log a PRSM query."""
        try:
            with db_manager.get_session() as session:
                query_log = QueryLog(
                    user_id=user_id,
                    prompt=prompt,
                    response=response,
                    status=status,
                    error_message=error_message,
                    response_time=response_time,
                    tokens_used=tokens_used,
                    cost=cost,
                    confidence=confidence,
                    ip_address=context.get('ip_address'),
                    user_agent=context.get('user_agent'),
                    request_id=context.get('request_id'),
                    metadata_=context
                )
                session.add(query_log)
                session.flush()
                return query_log
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to log query: {e}")
            return None

    def get_user_conversations(
        self, 
        user_id: str, 
        limit: int = 50
    ) -> List[Conversation]:
        """Get conversations for a user."""
        try:
            with db_manager.get_session() as session:
                return conversation_repo.get_user_conversations(
                    session, user_id, limit
                )
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to get user conversations: {e}")
            return []

    def get_conversation_history(
        self, 
        conversation_id: str, 
        limit: int = 100
    ) -> List[Message]:
        """Get conversation message history."""
        try:
            with db_manager.get_session() as session:
                return session.query(Message).filter(
                    Message.conversation_id == conversation_id
                ).order_by(Message.created_at).limit(limit).all()
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to get conversation history: {e}")
            return []

    def get_user_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get user analytics data."""
        try:
            with db_manager.get_session() as session:
                from datetime import timedelta
                from sqlalchemy import func
                
                since_date = datetime.utcnow() - timedelta(days=days)
                
                # Query metrics
                query_stats = session.query(
                    func.count(QueryLog.id).label('total_queries'),
                    func.avg(QueryLog.response_time).label('avg_response_time'),
                    func.sum(QueryLog.tokens_used).label('total_tokens'),
                    func.sum(QueryLog.cost).label('total_cost'),
                    func.avg(QueryLog.confidence).label('avg_confidence')
                ).filter(
                    QueryLog.user_id == user_id,
                    QueryLog.created_at >= since_date
                ).first()
                
                # Success rate
                success_rate = session.query(
                    func.count(QueryLog.id)
                ).filter(
                    QueryLog.user_id == user_id,
                    QueryLog.status == 'success',
                    QueryLog.created_at >= since_date
                ).scalar() / max(query_stats.total_queries, 1) * 100
                
                return {
                    'total_queries': query_stats.total_queries or 0,
                    'avg_response_time': float(query_stats.avg_response_time or 0),
                    'total_tokens': query_stats.total_tokens or 0,
                    'total_cost': float(query_stats.total_cost or 0),
                    'avg_confidence': float(query_stats.avg_confidence or 0),
                    'success_rate': success_rate,
                    'period_days': days
                }
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to get user analytics: {e}")
            return {}

# Initialize service
database_service = DatabaseService()
```

## 🚀 Performance Optimization

### Connection Pooling

```python
# prsm/core/connection_pool.py
import psycopg2
from psycopg2 import pool
import threading
import logging
from contextlib import contextmanager

class PostgreSQLConnectionPool:
    def __init__(
        self,
        host="localhost",
        database="prsm_production",
        user="prsm_user",
        password="secure_password_here",
        min_connections=5,
        max_connections=20
    ):
        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()
        
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                min_connections,
                max_connections,
                host=host,
                database=database,
                user=user,
                password=password,
                cursor_factory=psycopg2.extras.RealDictCursor
            )
            self.logger.info(f"Connection pool created with {min_connections}-{max_connections} connections")
        except Exception as e:
            self.logger.error(f"Failed to create connection pool: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Get connection from pool with automatic cleanup."""
        connection = None
        try:
            with self.lock:
                connection = self.connection_pool.getconn()
            
            yield connection
            
        except Exception as e:
            if connection:
                connection.rollback()
            self.logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if connection:
                with self.lock:
                    self.connection_pool.putconn(connection)

    def close_all_connections(self):
        """Close all connections in the pool."""
        with self.lock:
            self.connection_pool.closeall()
        self.logger.info("All connections closed")

# Global connection pool
connection_pool = PostgreSQLConnectionPool()
```

### Query Optimization

```sql
-- Performance optimization queries

-- Create indexes for common queries
CREATE INDEX CONCURRENTLY idx_query_logs_user_created 
ON query_logs (user_id, created_at DESC);

CREATE INDEX CONCURRENTLY idx_messages_conversation_created 
ON messages (conversation_id, created_at);

CREATE INDEX CONCURRENTLY idx_conversations_user_updated 
ON conversations (user_id, updated_at DESC) 
WHERE is_archived = false;

-- Partial index for active users
CREATE INDEX CONCURRENTLY idx_users_active 
ON users (id) 
WHERE is_active = true;

-- GIN index for JSONB metadata searching
CREATE INDEX CONCURRENTLY idx_query_logs_metadata_gin 
ON query_logs USING GIN (metadata_);

CREATE INDEX CONCURRENTLY idx_messages_metadata_gin 
ON messages USING GIN (metadata_);

-- Analyze tables for query optimization
ANALYZE users;
ANALYZE conversations;
ANALYZE messages;
ANALYZE query_logs;
```

### Database Monitoring

```python
# prsm/monitoring/database_monitor.py
import psycopg2
import psycopg2.extras
import time
import logging
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class DatabaseMetrics:
    active_connections: int
    idle_connections: int
    waiting_connections: int
    total_connections: int
    database_size: str
    slow_queries: List[Dict]
    lock_waits: List[Dict]
    cache_hit_ratio: float

class DatabaseMonitor:
    def __init__(self, connection_params: Dict):
        self.connection_params = connection_params
        self.logger = logging.getLogger(__name__)

    def get_connection_stats(self) -> Dict:
        """Get connection statistics."""
        query = """
        SELECT 
            count(*) as total_connections,
            count(*) FILTER (WHERE state = 'active') as active_connections,
            count(*) FILTER (WHERE state = 'idle') as idle_connections,
            count(*) FILTER (WHERE wait_event IS NOT NULL) as waiting_connections
        FROM pg_stat_activity 
        WHERE datname = current_database();
        """
        
        try:
            with psycopg2.connect(**self.connection_params) as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(query)
                    return dict(cursor.fetchone())
        except Exception as e:
            self.logger.error(f"Failed to get connection stats: {e}")
            return {}

    def get_slow_queries(self, threshold_ms: int = 1000) -> List[Dict]:
        """Get currently running slow queries."""
        query = """
        SELECT 
            pid,
            now() - pg_stat_activity.query_start AS duration,
            query,
            state,
            wait_event,
            wait_event_type
        FROM pg_stat_activity 
        WHERE (now() - pg_stat_activity.query_start) > interval '%s milliseconds'
        AND state = 'active'
        AND query NOT LIKE '%%pg_stat_activity%%'
        ORDER BY duration DESC;
        """ % threshold_ms
        
        try:
            with psycopg2.connect(**self.connection_params) as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(query)
                    return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Failed to get slow queries: {e}")
            return []

    def get_database_size(self) -> str:
        """Get database size."""
        query = "SELECT pg_size_pretty(pg_database_size(current_database()));"
        
        try:
            with psycopg2.connect(**self.connection_params) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    return cursor.fetchone()[0]
        except Exception as e:
            self.logger.error(f"Failed to get database size: {e}")
            return "Unknown"

    def get_cache_hit_ratio(self) -> float:
        """Get cache hit ratio."""
        query = """
        SELECT 
            sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) as hit_ratio
        FROM pg_statio_user_tables
        WHERE schemaname = 'public';
        """
        
        try:
            with psycopg2.connect(**self.connection_params) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    result = cursor.fetchone()[0]
                    return float(result) if result else 0.0
        except Exception as e:
            self.logger.error(f"Failed to get cache hit ratio: {e}")
            return 0.0

    def get_lock_waits(self) -> List[Dict]:
        """Get current lock waits."""
        query = """
        SELECT 
            blocked_locks.pid AS blocked_pid,
            blocked_activity.usename AS blocked_user,
            blocking_locks.pid AS blocking_pid,
            blocking_activity.usename AS blocking_user,
            blocked_activity.query AS blocked_statement,
            blocking_activity.query AS current_statement_in_blocking_process,
            blocked_activity.application_name AS blocked_application,
            blocking_activity.application_name AS blocking_application
        FROM pg_catalog.pg_locks blocked_locks
        JOIN pg_catalog.pg_stat_activity blocked_activity 
            ON blocked_activity.pid = blocked_locks.pid
        JOIN pg_catalog.pg_locks blocking_locks 
            ON blocking_locks.locktype = blocked_locks.locktype
            AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
            AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
            AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
            AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
            AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
            AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
            AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
            AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
            AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
            AND blocking_locks.pid != blocked_locks.pid
        JOIN pg_catalog.pg_stat_activity blocking_activity 
            ON blocking_activity.pid = blocking_locks.pid
        WHERE NOT blocked_locks.granted;
        """
        
        try:
            with psycopg2.connect(**self.connection_params) as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(query)
                    return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Failed to get lock waits: {e}")
            return []

    def collect_metrics(self) -> DatabaseMetrics:
        """Collect all database metrics."""
        connection_stats = self.get_connection_stats()
        
        return DatabaseMetrics(
            active_connections=connection_stats.get('active_connections', 0),
            idle_connections=connection_stats.get('idle_connections', 0),
            waiting_connections=connection_stats.get('waiting_connections', 0),
            total_connections=connection_stats.get('total_connections', 0),
            database_size=self.get_database_size(),
            slow_queries=self.get_slow_queries(),
            lock_waits=self.get_lock_waits(),
            cache_hit_ratio=self.get_cache_hit_ratio()
        )

# Usage
monitor = DatabaseMonitor({
    'host': 'localhost',
    'database': 'prsm_production',
    'user': 'prsm_user',
    'password': 'secure_password_here'
})

metrics = monitor.collect_metrics()
print(f"Active connections: {metrics.active_connections}")
print(f"Cache hit ratio: {metrics.cache_hit_ratio:.2%}")
```

## 📊 Database Migrations

### Alembic Configuration

```python
# alembic.ini
[alembic]
script_location = migrations
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = postgresql://prsm_user:secure_password_here@localhost/prsm_production

[post_write_hooks]
hooks = black
black.type = console_scripts
black.entrypoint = black
black.options = -l 79 REVISION_SCRIPT_FILENAME

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

### Migration Script Example

```python
# migrations/versions/001_create_core_tables.py
"""Create core tables

Revision ID: 001
Revises: 
Create Date: 2024-01-01 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create users table
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('username', sa.String(length=50), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('metadata_', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_users_email', 'users', ['email'], unique=True)
    op.create_index('idx_users_username', 'users', ['username'], unique=True)
    op.create_index('idx_users_created_at', 'users', ['created_at'])

    # Create conversations table
    op.create_table('conversations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('is_archived', sa.Boolean(), nullable=False),
        sa.Column('metadata_', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_conversations_user_id', 'conversations', ['user_id'])
    op.create_index('idx_conversations_created_at', 'conversations', ['created_at'])
    op.create_index('idx_conversations_updated_at', 'conversations', ['updated_at'])

    # Create messages table
    op.create_table('messages',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('conversation_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('role', sa.String(length=20), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('metadata_', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('tokens_used', sa.Integer(), nullable=True),
        sa.Column('processing_time', sa.Float(), nullable=True),
        sa.Column('model_version', sa.String(length=50), nullable=True),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_messages_conversation_id', 'messages', ['conversation_id'])
    op.create_index('idx_messages_created_at', 'messages', ['created_at'])
    op.create_index('idx_messages_role', 'messages', ['role'])

    # Create query_logs table
    op.create_table('query_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('response', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('response_time', sa.Float(), nullable=True),
        sa.Column('tokens_used', sa.Integer(), nullable=True),
        sa.Column('cost', sa.Float(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('request_id', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('metadata_', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_query_logs_user_id', 'query_logs', ['user_id'])
    op.create_index('idx_query_logs_created_at', 'query_logs', ['created_at'])
    op.create_index('idx_query_logs_status', 'query_logs', ['status'])
    op.create_index('idx_query_logs_request_id', 'query_logs', ['request_id'])

def downgrade():
    op.drop_table('query_logs')
    op.drop_table('messages')
    op.drop_table('conversations')
    op.drop_table('users')
```

## 🔒 Security Best Practices

### Connection Security

```python
# prsm/security/database_security.py
import ssl
import os
from urllib.parse import urlparse, parse_qs

class DatabaseSecurity:
    @staticmethod
    def get_secure_connection_url():
        """Get secure database connection URL."""
        base_url = os.environ.get('DATABASE_URL')
        
        # Parse URL
        parsed = urlparse(base_url)
        
        # Add SSL parameters
        ssl_params = {
            'sslmode': 'require',
            'sslcert': os.environ.get('DB_SSL_CERT'),
            'sslkey': os.environ.get('DB_SSL_KEY'),
            'sslrootcert': os.environ.get('DB_SSL_ROOT_CERT')
        }
        
        # Filter out None values
        ssl_params = {k: v for k, v in ssl_params.items() if v}
        
        # Reconstruct URL with SSL parameters
        query_params = []
        for key, value in ssl_params.items():
            query_params.append(f"{key}={value}")
        
        query_string = "&".join(query_params)
        
        secure_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if query_string:
            secure_url += f"?{query_string}"
        
        return secure_url

    @staticmethod
    def create_ssl_context():
        """Create SSL context for database connections."""
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_REQUIRED
        
        # Load certificates if provided
        cert_file = os.environ.get('DB_SSL_CERT')
        key_file = os.environ.get('DB_SSL_KEY')
        ca_file = os.environ.get('DB_SSL_ROOT_CERT')
        
        if cert_file and key_file:
            context.load_cert_chain(cert_file, key_file)
        
        if ca_file:
            context.load_verify_locations(ca_file)
        
        return context
```

### Data Encryption

```sql
-- Enable pgcrypto extension for encryption
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Create encrypted columns example
ALTER TABLE users ADD COLUMN encrypted_data bytea;

-- Function to encrypt sensitive data
CREATE OR REPLACE FUNCTION encrypt_sensitive_data(data text, key text)
RETURNS bytea AS $$
BEGIN
    RETURN pgp_sym_encrypt(data, key);
END;
$$ LANGUAGE plpgsql;

-- Function to decrypt sensitive data
CREATE OR REPLACE FUNCTION decrypt_sensitive_data(encrypted_data bytea, key text)
RETURNS text AS $$
BEGIN
    RETURN pgp_sym_decrypt(encrypted_data, key);
END;
$$ LANGUAGE plpgsql;
```

## 📋 Best Practices

### Configuration

```python
# config/database.py
import os
from dataclasses import dataclass

@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    
    # Connection settings
    host: str = os.environ.get('DB_HOST', 'localhost')
    port: int = int(os.environ.get('DB_PORT', '5432'))
    database: str = os.environ.get('DB_NAME', 'prsm_production')
    username: str = os.environ.get('DB_USER', 'prsm_user')
    password: str = os.environ.get('DB_PASSWORD', '')
    
    # Pool settings
    pool_size: int = int(os.environ.get('DB_POOL_SIZE', '20'))
    max_overflow: int = int(os.environ.get('DB_MAX_OVERFLOW', '30'))
    pool_timeout: int = int(os.environ.get('DB_POOL_TIMEOUT', '30'))
    pool_recycle: int = int(os.environ.get('DB_POOL_RECYCLE', '3600'))
    
    # Query settings
    echo: bool = os.environ.get('DB_ECHO', 'false').lower() == 'true'
    statement_timeout: int = int(os.environ.get('DB_STATEMENT_TIMEOUT', '30'))
    
    # SSL settings
    ssl_mode: str = os.environ.get('DB_SSL_MODE', 'prefer')
    ssl_cert: str = os.environ.get('DB_SSL_CERT', '')
    ssl_key: str = os.environ.get('DB_SSL_KEY', '')
    ssl_root_cert: str = os.environ.get('DB_SSL_ROOT_CERT', '')
    
    @property
    def connection_url(self) -> str:
        """Get connection URL."""
        base_url = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        
        params = []
        if self.ssl_mode:
            params.append(f"sslmode={self.ssl_mode}")
        if self.ssl_cert:
            params.append(f"sslcert={self.ssl_cert}")
        if self.ssl_key:
            params.append(f"sslkey={self.ssl_key}")
        if self.ssl_root_cert:
            params.append(f"sslrootcert={self.ssl_root_cert}")
        
        if params:
            base_url += "?" + "&".join(params)
        
        return base_url
```

### Maintenance Scripts

```python
# scripts/db_maintenance.py
#!/usr/bin/env python3
"""Database maintenance scripts."""

import argparse
import psycopg2
import logging
from datetime import datetime, timedelta

def vacuum_analyze_tables(connection_params):
    """Run VACUUM ANALYZE on all tables."""
    tables = ['users', 'conversations', 'messages', 'query_logs', 'system_metrics']
    
    with psycopg2.connect(**connection_params) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            for table in tables:
                print(f"Running VACUUM ANALYZE on {table}...")
                cursor.execute(f"VACUUM ANALYZE {table};")
                print(f"Completed {table}")

def cleanup_old_logs(connection_params, days_to_keep=90):
    """Clean up old query logs."""
    cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
    
    with psycopg2.connect(**connection_params) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM query_logs WHERE created_at < %s",
                (cutoff_date,)
            )
            deleted_count = cursor.rowcount
            print(f"Deleted {deleted_count} old query logs")

def reindex_tables(connection_params):
    """Rebuild indexes for better performance."""
    indexes = [
        'idx_users_email',
        'idx_conversations_user_id',
        'idx_messages_conversation_id',
        'idx_query_logs_created_at'
    ]
    
    with psycopg2.connect(**connection_params) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            for index in indexes:
                print(f"Rebuilding index {index}...")
                cursor.execute(f"REINDEX INDEX CONCURRENTLY {index};")
                print(f"Completed {index}")

def main():
    parser = argparse.ArgumentParser(description='Database maintenance')
    parser.add_argument('--vacuum', action='store_true', help='Run VACUUM ANALYZE')
    parser.add_argument('--cleanup', action='store_true', help='Clean up old logs')
    parser.add_argument('--reindex', action='store_true', help='Rebuild indexes')
    parser.add_argument('--days', type=int, default=90, help='Days to keep logs')
    
    args = parser.parse_args()
    
    connection_params = {
        'host': 'localhost',
        'database': 'prsm_production',
        'user': 'prsm_user',
        'password': 'secure_password_here'
    }
    
    if args.vacuum:
        vacuum_analyze_tables(connection_params)
    
    if args.cleanup:
        cleanup_old_logs(connection_params, args.days)
    
    if args.reindex:
        reindex_tables(connection_params)

if __name__ == '__main__':
    main()
```

---

**Need help with PostgreSQL integration?** Join our [Discord community](https://discord.gg/prsm) or [open an issue](https://github.com/prsm-org/prsm/issues).