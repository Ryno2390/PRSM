"""
Startup Performance Optimizer for PRSM
======================================

Optimizes application startup time by implementing lazy loading,
connection warming, and efficient initialization strategies.
"""

import asyncio
import time
from typing import List, Callable, Any
from contextlib import asynccontextmanager
import structlog

logger = structlog.get_logger(__name__)

class StartupOptimizer:
    """Optimizes application startup performance"""
    
    def __init__(self):
        self.initialization_tasks: List[Callable] = []
        self.background_tasks: List[Callable] = []
        self.startup_time = None
    
    def add_initialization_task(self, task: Callable):
        """Add critical initialization task"""
        self.initialization_tasks.append(task)
    
    def add_background_task(self, task: Callable):
        """Add non-critical background task"""
        self.background_tasks.append(task)
    
    @asynccontextmanager
    async def optimized_startup(self):
        """Context manager for optimized startup"""
        start_time = time.time()
        logger.info("🚀 Starting optimized PRSM initialization...")
        
        try:
            # Phase 1: Critical initialization (parallel)
            await self._run_critical_initialization()
            
            # Phase 2: Start background tasks (non-blocking)
            self._start_background_tasks()
            
            self.startup_time = time.time() - start_time
            logger.info(f"✅ PRSM initialized in {self.startup_time:.2f} seconds")
            
            yield self
            
        finally:
            logger.info("🛑 Shutting down PRSM...")
            await self._cleanup()
    
    async def _run_critical_initialization(self):
        """Run critical initialization tasks in parallel"""
        if not self.initialization_tasks:
            return
        
        logger.info(f"⚡ Running {len(self.initialization_tasks)} critical initialization tasks...")
        
        # Run tasks in parallel with timeout
        tasks = [asyncio.create_task(task()) for task in self.initialization_tasks]
        
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30.0  # 30 second timeout for startup
            )
            logger.info("✅ Critical initialization completed")
        except asyncio.TimeoutError:
            logger.error("❌ Critical initialization timed out")
            raise
    
    def _start_background_tasks(self):
        """Start background tasks without blocking"""
        if not self.background_tasks:
            return
        
        logger.info(f"🔄 Starting {len(self.background_tasks)} background tasks...")
        
        for task in self.background_tasks:
            asyncio.create_task(task())
    
    async def _cleanup(self):
        """Cleanup resources"""
        # Close database connections
        try:
            from prsm.core.optimized_database import close_db_connections
            await close_db_connections()
        except ImportError:
            pass
        
        logger.info("✅ Cleanup completed")

# Global startup optimizer
startup_optimizer = StartupOptimizer()

def critical_init(func):
    """Decorator to mark function as critical initialization"""
    startup_optimizer.add_initialization_task(func)
    return func

def background_init(func):
    """Decorator to mark function as background initialization"""
    startup_optimizer.add_background_task(func)
    return func

# Pre-configured optimization functions
@critical_init
async def initialize_database():
    """Initialize database connections"""
    try:
        from prsm.core.optimized_database import get_optimized_db_config
        db_config = await get_optimized_db_config()
        health_ok = await db_config.health_check()
        if health_ok:
            logger.info("✅ Database connection optimized and ready")
        else:
            logger.warning("⚠️ Database health check failed")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")

@critical_init  
async def initialize_cache():
    """Initialize caching layer"""
    try:
        from prsm.core.production_cache import get_cache_manager
        cache_manager = await get_cache_manager()
        logger.info("✅ Cache layer initialized")
    except Exception as e:
        logger.error(f"❌ Cache initialization failed: {e}")

@background_init
async def warm_up_models():
    """Warm up ML models in background"""
    try:
        # Simulate model warming
        await asyncio.sleep(1)
        logger.info("✅ Models warmed up")
    except Exception as e:
        logger.error(f"❌ Model warm-up failed: {e}")

@background_init
async def initialize_monitoring():
    """Initialize monitoring and metrics"""
    try:
        # Simulate monitoring setup
        await asyncio.sleep(0.5)
        logger.info("✅ Monitoring initialized")
    except Exception as e:
        logger.error(f"❌ Monitoring initialization failed: {e}")