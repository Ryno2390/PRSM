"""
Optimized Database Configuration for PRSM
==========================================

Production-grade database connection pooling and optimization
settings to support high concurrent load and improve performance.
"""

import os
from typing import Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import structlog

logger = structlog.get_logger(__name__)

class OptimizedDatabaseConfig:
    """Optimized database configuration for production performance"""
    
    def __init__(self):
        self.database_url = os.getenv(
            "DATABASE_URL", 
            "postgresql+asyncpg://prsm_user:prsm_password@localhost:5432/prsm_db"
        )
        
        # Optimized connection pool settings for high concurrency
        self.engine = create_async_engine(
            self.database_url,
            
            # Connection pool optimization for 1000+ concurrent users
            poolclass=QueuePool,
            pool_size=20,                    # Base connections
            max_overflow=50,                 # Additional connections under load
            pool_pre_ping=True,             # Validate connections
            pool_recycle=3600,              # Recycle connections every hour
            
            # Performance tuning
            echo=False,                     # Disable SQL logging in production
            echo_pool=False,                # Disable pool logging
            future=True,                    # Use SQLAlchemy 2.0 style
            
            # Connection optimization
            connect_args={
                "server_settings": {
                    "application_name": "PRSM_Production",
                    "jit": "off",           # Disable JIT for predictable performance
                }
            }
        )
        
        # Optimized session factory
        self.async_session_factory = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False          # Keep objects accessible after commit
        )
    
    async def get_session(self) -> AsyncSession:
        """Get optimized database session"""
        async with self.async_session_factory() as session:
            yield session
    
    async def health_check(self) -> bool:
        """Check database connectivity and performance"""
        try:
            async with self.async_session_factory() as session:
                result = await session.execute("SELECT 1")
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def close(self):
        """Gracefully close database connections"""
        await self.engine.dispose()

# Global database instance
_db_config: Optional[OptimizedDatabaseConfig] = None

async def get_optimized_db_config() -> OptimizedDatabaseConfig:
    """Get singleton database configuration instance"""
    global _db_config
    if _db_config is None:
        _db_config = OptimizedDatabaseConfig()
    return _db_config

async def close_db_connections():
    """Close all database connections"""
    global _db_config
    if _db_config:
        await _db_config.close()
        _db_config = None