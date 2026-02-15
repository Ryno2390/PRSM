"""
Database Testing Fixtures
==========================

Comprehensive database fixtures for testing with transaction rollback,
test data factories, and database isolation.
"""

import pytest
import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator
from unittest.mock import AsyncMock, Mock
from decimal import Decimal
import uuid
from datetime import datetime, timezone

try:
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.pool import StaticPool
    import pytest_asyncio
    from prsm.core.database import get_db_session, Base
    from prsm.core.models import *
    from prsm.economy.tokenomics.models import FTNSTransaction
except ImportError:
    # If imports fail, we'll create mock fixtures
    Base = None
    Session = None


@pytest.fixture(scope="session")
def test_database_url():
    """Test database URL for in-memory SQLite"""
    return "sqlite:///:memory:"


@pytest.fixture(scope="session")
def test_engine(test_database_url):
    """Create test database engine with in-memory SQLite"""
    if Base is None:
        pytest.skip("Database dependencies not available")
    
    engine = create_engine(
        test_database_url,
        poolclass=StaticPool,
        connect_args={
            "check_same_thread": False,
            "isolation_level": None,
        },
        echo=False  # Set to True for SQL debugging
    )
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    yield engine
    
    # Cleanup
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(scope="function")
def test_session(test_engine):
    """Create test database session with automatic rollback"""
    if Session is None:
        pytest.skip("Database dependencies not available")
    
    # Create session factory
    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=test_engine
    )
    
    # Create session
    session = TestingSessionLocal()
    
    # Begin a transaction
    transaction = session.begin()
    
    try:
        yield session
    finally:
        # Always rollback to ensure test isolation
        transaction.rollback()
        session.close()


@pytest.fixture
def db_session(test_session):
    """Alias for test_session for compatibility"""
    return test_session


class DatabaseTestFactory:
    """Factory for creating test database objects"""
    
    @staticmethod
    def create_prsm_session(
        session_id: Optional[str] = None,
        user_id: str = "test_user",
        status: str = "pending",
        **kwargs
    ) -> 'PRSMSession':
        """Create test PRSM session"""
        if 'PRSMSession' not in globals():
            return Mock()
        
        return PRSMSession(
            session_id=session_id or str(uuid.uuid4()),
            user_id=user_id,
            status=status,
            created_at=datetime.now(timezone.utc),
            **kwargs
        )
    
    @staticmethod
    def create_ftns_transaction(
        transaction_id: Optional[str] = None,
        user_id: str = "test_user",
        amount: Decimal = Decimal("10.0"),
        transaction_type: str = "reward",
        **kwargs
    ) -> 'FTNSTransaction':
        """Create test FTNS transaction"""
        if 'FTNSTransaction' not in globals():
            return Mock()
        
        return FTNSTransaction(
            transaction_id=transaction_id or str(uuid.uuid4()),
            user_id=user_id,
            amount=amount,
            transaction_type=transaction_type,
            timestamp=datetime.now(timezone.utc),
            **kwargs
        )
    
    @staticmethod
    def create_user_input(
        input_id: Optional[str] = None,
        user_id: str = "test_user",
        content: str = "Test query",
        **kwargs
    ) -> 'UserInput':
        """Create test user input"""
        if 'UserInput' not in globals():
            return Mock()
        
        return UserInput(
            input_id=input_id or str(uuid.uuid4()),
            user_id=user_id,
            content=content,
            timestamp=datetime.now(timezone.utc),
            **kwargs
        )


@pytest.fixture
def db_factory():
    """Database test factory fixture"""
    return DatabaseTestFactory()


@pytest.fixture
def sample_db_data(test_session, db_factory):
    """Create sample database data for testing"""
    if test_session is None:
        return {}
    
    # Create sample data
    sessions = []
    transactions = []
    inputs = []
    
    for i in range(3):
        # Create PRSM sessions
        session = db_factory.create_prsm_session(
            user_id=f"user_{i}",
            status="pending" if i == 0 else "completed"
        )
        sessions.append(session)
        test_session.add(session)
        
        # Create transactions
        transaction = db_factory.create_ftns_transaction(
            user_id=f"user_{i}",
            amount=Decimal(f"{10 + i * 5}.0"),
            transaction_type="reward"
        )
        transactions.append(transaction)
        test_session.add(transaction)
        
        # Create user inputs
        user_input = db_factory.create_user_input(
            user_id=f"user_{i}",
            content=f"Test query {i}"
        )
        inputs.append(user_input)
        test_session.add(user_input)
    
    test_session.commit()
    
    return {
        "sessions": sessions,
        "transactions": transactions,
        "inputs": inputs
    }


@pytest.fixture
def empty_test_db(test_session):
    """Empty test database for clean slate testing"""
    if test_session is None:
        return None
    
    # Ensure database is empty
    test_session.execute("DELETE FROM ftns_transactions")
    test_session.execute("DELETE FROM user_inputs") 
    test_session.execute("DELETE FROM prsm_sessions")
    test_session.commit()
    
    return test_session


# Async database testing fixtures

@pytest_asyncio.fixture
async def async_test_session(test_engine):
    """Async database session for async testing"""
    if Session is None:
        pytest.skip("Database dependencies not available")
    
    # Note: This is a simplified async session
    # In a real async setup, you'd use async SQLAlchemy
    
    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=test_engine
    )
    
    session = TestingSessionLocal()
    transaction = session.begin()
    
    try:
        yield session
    finally:
        transaction.rollback()
        session.close()


@pytest.fixture
def mock_database_error():
    """Mock database error scenarios"""
    return {
        "connection_error": Exception("Database connection failed"),
        "timeout_error": Exception("Database operation timed out"),
        "constraint_error": Exception("Database constraint violation"),
        "transaction_error": Exception("Transaction failed")
    }


@pytest.fixture
def database_performance_monitor():
    """Monitor database performance during tests"""
    class DatabasePerformanceMonitor:
        def __init__(self):
            self.query_times = []
            self.query_counts = 0
            
        def start_monitoring(self, engine):
            @event.listens_for(engine, "before_cursor_execute")
            def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
                context._query_start_time = datetime.now()
                
            @event.listens_for(engine, "after_cursor_execute")
            def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
                if hasattr(context, '_query_start_time'):
                    query_time = (datetime.now() - context._query_start_time).total_seconds() * 1000
                    self.query_times.append(query_time)
                    self.query_counts += 1
        
        def get_stats(self):
            if not self.query_times:
                return {"avg_query_time": 0, "total_queries": 0, "max_query_time": 0}
            
            return {
                "avg_query_time": sum(self.query_times) / len(self.query_times),
                "total_queries": self.query_counts,
                "max_query_time": max(self.query_times),
                "min_query_time": min(self.query_times)
            }
    
    return DatabasePerformanceMonitor()


# Database cleanup utilities

class DatabaseTestCleaner:
    """Utility for cleaning up test database"""
    
    @staticmethod
    def clean_all_tables(session):
        """Clean all tables in the test database"""
        if session is None:
            return
        
        # Get all table names (simplified approach)
        tables = [
            "ftns_transactions",
            "user_inputs", 
            "prsm_sessions",
            "architect_tasks",
            "governance_proposals",
            "governance_votes"
        ]
        
        for table in tables:
            try:
                session.execute(f"DELETE FROM {table}")
            except Exception:
                # Table might not exist or be accessible
                pass
        
        session.commit()
    
    @staticmethod
    def verify_cleanup(session):
        """Verify database cleanup was successful"""
        if session is None:
            return True
        
        # Check if tables are empty
        tables_to_check = [
            "ftns_transactions",
            "user_inputs",
            "prsm_sessions"
        ]
        
        for table in tables_to_check:
            try:
                result = session.execute(f"SELECT COUNT(*) FROM {table}").scalar()
                if result > 0:
                    return False
            except Exception:
                # Table might not exist
                continue
        
        return True


@pytest.fixture
def db_cleaner():
    """Database cleaner fixture"""
    return DatabaseTestCleaner()


# Integration test database fixtures

@pytest.fixture(scope="session")
def integration_test_db():
    """Database setup for integration tests"""
    # This would typically use a containerized database
    # For now, we'll use in-memory SQLite
    return "sqlite:///:memory:"


@pytest.fixture
def isolated_db_test(test_session):
    """Complete database isolation for sensitive tests"""
    if test_session is None:
        return None
    
    # Create a fresh session with full isolation
    # This fixture ensures no data leakage between tests
    
    # Clean up any existing data
    test_session.execute("DELETE FROM ftns_transactions")
    test_session.execute("DELETE FROM user_inputs")
    test_session.execute("DELETE FROM prsm_sessions")
    test_session.commit()
    
    yield test_session
    
    # Clean up after test
    test_session.execute("DELETE FROM ftns_transactions")
    test_session.execute("DELETE FROM user_inputs")
    test_session.execute("DELETE FROM prsm_sessions")
    test_session.commit()