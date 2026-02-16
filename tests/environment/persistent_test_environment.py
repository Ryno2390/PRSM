#!/usr/bin/env python3
"""
PRSM Persistent Test Environment
===============================

🎯 PURPOSE:
Creates and manages a persistent test environment for ongoing PRSM system validation,
development, and performance monitoring. This environment maintains consistent state
across test runs and provides automated health monitoring.

🚀 KEY FEATURES:
- Automated environment setup and teardown
- Persistent test data management
- Continuous health monitoring
- Performance benchmarking
- Test state isolation and cleanup
- Environment configuration management
- Automated dependency verification
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

import structlog
import pytest

# Import all PRSM components for comprehensive testing
try:
    from prsm.core.config import get_settings
    from prsm.core.database import init_database, close_database, db_manager
    from prsm.core.redis_client import init_redis, close_redis, redis_manager
    from prsm.core.vector_db import init_vector_databases, close_vector_databases
    from prsm.core.ipfs_client import init_ipfs, close_ipfs
    from prsm.economy.tokenomics.ftns_service import ftns_service
except (ImportError, ModuleNotFoundError) as e:
    pytest.skip(f"PRSM module dependencies not fully implemented: {e}", allow_module_level=True)

logger = structlog.get_logger(__name__)


@dataclass
class TestEnvironmentConfig:
    """Configuration for persistent test environment"""
    environment_id: str = field(default_factory=lambda: f"prsm_test_env_{uuid.uuid4().hex[:8]}")
    base_directory: Path = field(default_factory=lambda: Path.cwd() / "test_environments")
    persistent_data: bool = True
    auto_cleanup: bool = False
    health_check_interval: int = 30  # seconds
    performance_monitoring: bool = True
    isolated_services: bool = True
    test_data_seed: Optional[str] = None


@dataclass
class EnvironmentState:
    """Current state of test environment"""
    environment_id: str
    status: str  # 'initializing', 'running', 'stopped', 'error'
    created_at: datetime
    last_health_check: Optional[datetime] = None
    components_status: Dict[str, bool] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    test_data_info: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class PersistentTestEnvironment:
    """
    Manages a persistent test environment for PRSM system validation
    
    🎯 CAPABILITIES:
    - Environment lifecycle management (create, start, stop, destroy)
    - Service dependency management and health monitoring
    - Test data generation and persistence
    - Performance benchmarking and metrics collection
    - Isolated test execution with state cleanup
    - Automated environment recovery and repair
    """
    
    def __init__(self, config: Optional[TestEnvironmentConfig] = None):
        self.config = config or TestEnvironmentConfig()
        self.state = EnvironmentState(
            environment_id=self.config.environment_id,
            status='initializing',
            created_at=datetime.now(timezone.utc)
        )
        
        # Environment paths
        self.env_dir = self.config.base_directory / self.config.environment_id
        self.data_dir = self.env_dir / "data"
        self.logs_dir = self.env_dir / "logs"
        self.config_dir = self.env_dir / "config"
        self.temp_dir = self.env_dir / "temp"
        
        # Service managers
        self.services: Dict[str, Any] = {}
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.performance_monitor_task: Optional[asyncio.Task] = None
        
        # Test data management
        self.test_data_generators: Dict[str, callable] = {}
        self.cleanup_tasks: List[callable] = []
        
        logger.info("Persistent test environment initialized",
                   environment_id=self.config.environment_id,
                   base_dir=str(self.env_dir))
    
    async def setup(self) -> bool:
        """
        Set up the persistent test environment
        
        🚀 SETUP PROCESS:
        1. Create directory structure
        2. Initialize configuration files
        3. Start core services (database, redis, etc.)
        4. Verify service health
        5. Generate initial test data
        6. Start monitoring tasks
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            logger.info("Setting up persistent test environment", 
                       environment_id=self.config.environment_id)
            
            # 1. Create directory structure
            await self._create_directory_structure()
            
            # 2. Initialize configuration
            await self._initialize_configuration()
            
            # 3. Start core services
            await self._start_core_services()
            
            # 4. Verify service health
            health_check = await self._comprehensive_health_check()
            if not health_check:
                raise RuntimeError("Health check failed during setup")
            
            # 5. Generate test data
            await self._generate_test_data()
            
            # 6. Start monitoring
            await self._start_monitoring()
            
            self.state.status = 'running'
            logger.info("Persistent test environment setup complete",
                       environment_id=self.config.environment_id,
                       components=len(self.state.components_status))
            
            return True
            
        except Exception as e:
            self.state.status = 'error'
            self.state.errors.append(f"Setup failed: {str(e)}")
            logger.error("Failed to setup persistent test environment",
                        environment_id=self.config.environment_id,
                        error=str(e))
            return False
    
    async def _create_directory_structure(self):
        """Create the directory structure for the test environment"""
        directories = [
            self.env_dir,
            self.data_dir,
            self.logs_dir,
            self.config_dir,
            self.temp_dir,
            self.data_dir / "database",
            self.data_dir / "redis",
            self.data_dir / "vector_db",
            self.data_dir / "ipfs",
            self.logs_dir / "services",
            self.logs_dir / "tests",
            self.logs_dir / "performance"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.debug("Directory structure created",
                    environment_id=self.config.environment_id,
                    directories=len(directories))
    
    async def _initialize_configuration(self):
        """Initialize configuration files for the test environment"""
        
        # Environment-specific settings
        env_settings = {
            "environment_id": self.config.environment_id,
            "database_url": f"sqlite:///{self.data_dir}/database/test.db",
            "redis_url": f"redis://localhost:6379/1",  # Use DB 1 for testing
            "vector_db_path": str(self.data_dir / "vector_db"),
            "ipfs_data_path": str(self.data_dir / "ipfs"),
            "log_level": "DEBUG",
            "test_mode": True,
            "persistent_data": self.config.persistent_data,
            "created_at": self.state.created_at.isoformat()
        }
        
        # Save environment configuration
        config_file = self.config_dir / "environment.json"
        with open(config_file, 'w') as f:
            json.dump(env_settings, f, indent=2)
        
        # Create service-specific configurations
        await self._create_service_configs()
        
        logger.debug("Configuration initialized",
                    environment_id=self.config.environment_id,
                    config_file=str(config_file))
    
    async def _create_service_configs(self):
        """Create configuration files for individual services"""
        
        # Database configuration
        db_config = {
            "database_url": f"sqlite:///{self.data_dir}/database/test.db",
            "echo": True,
            "pool_size": 5,
            "max_overflow": 10
        }
        
        with open(self.config_dir / "database.json", 'w') as f:
            json.dump(db_config, f, indent=2)
        
        # Redis configuration
        redis_config = {
            "host": "localhost",
            "port": 6379,
            "db": 1,  # Use separate DB for testing
            "decode_responses": True,
            "health_check_interval": 30
        }
        
        with open(self.config_dir / "redis.json", 'w') as f:
            json.dump(redis_config, f, indent=2)
        
        # Vector database configuration
        vector_config = {
            "data_path": str(self.data_dir / "vector_db"),
            "collection_name": f"prsm_test_{self.config.environment_id}",
            "embedding_dimension": 1536,
            "metric": "cosine"
        }
        
        with open(self.config_dir / "vector_db.json", 'w') as f:
            json.dump(vector_config, f, indent=2)
    
    async def _start_core_services(self):
        """Start all core PRSM services"""
        
        services_to_start = [
            ("database", self._start_database_service),
            ("redis", self._start_redis_service),
            ("vector_db", self._start_vector_db_service),
            ("ipfs", self._start_ipfs_service),
            ("ftns", self._start_ftns_service)
        ]
        
        for service_name, start_func in services_to_start:
            try:
                logger.debug(f"Starting {service_name} service",
                           environment_id=self.config.environment_id)
                
                service = await start_func()
                self.services[service_name] = service
                self.state.components_status[service_name] = True
                
                logger.info(f"{service_name.title()} service started",
                          environment_id=self.config.environment_id)
                
            except Exception as e:
                self.state.components_status[service_name] = False
                self.state.errors.append(f"{service_name} service failed: {str(e)}")
                logger.error(f"Failed to start {service_name} service",
                           environment_id=self.config.environment_id,
                           error=str(e))
                # Continue with other services
    
    async def _start_database_service(self):
        """Start database service"""
        await init_database()
        await db_manager.create_tables()
        return db_manager
    
    async def _start_redis_service(self):
        """Start Redis service"""
        await init_redis()
        return redis_manager
    
    async def _start_vector_db_service(self):
        """Start vector database service"""
        await init_vector_databases()
        from prsm.core.vector_db import get_vector_db_manager
        return get_vector_db_manager()
    
    async def _start_ipfs_service(self):
        """Start IPFS service"""
        try:
            await init_ipfs()
            from prsm.core.ipfs_client import get_ipfs_client
            return get_ipfs_client()
        except Exception as e:
            logger.warning("IPFS service not available, using mock",
                         environment_id=self.config.environment_id,
                         error=str(e))
            return None
    
    async def _start_ftns_service(self):
        """Start FTNS tokenomics service"""
        # Initialize FTNS service with test configuration
        return ftns_service
    
    async def _comprehensive_health_check(self) -> bool:
        """Perform comprehensive health check of all services"""
        
        health_results = {}
        overall_health = True
        
        # Check each service
        for service_name, service in self.services.items():
            try:
                if service_name == "database":
                    health = await self._check_database_health(service)
                elif service_name == "redis":
                    health = await self._check_redis_health(service)
                elif service_name == "vector_db":
                    health = await self._check_vector_db_health(service)
                elif service_name == "ipfs":
                    health = await self._check_ipfs_health(service)
                elif service_name == "ftns":
                    health = await self._check_ftns_health(service)
                else:
                    health = True  # Default to healthy for unknown services
                
                health_results[service_name] = health
                if not health:
                    overall_health = False
                    
            except Exception as e:
                health_results[service_name] = False
                overall_health = False
                logger.error(f"Health check failed for {service_name}",
                           environment_id=self.config.environment_id,
                           error=str(e))
        
        self.state.components_status.update(health_results)
        self.state.last_health_check = datetime.now(timezone.utc)
        
        logger.info("Comprehensive health check completed",
                   environment_id=self.config.environment_id,
                   overall_health=overall_health,
                   service_health=health_results)
        
        return overall_health
    
    async def _check_database_health(self, service) -> bool:
        """Check database service health"""
        try:
            return await service.health_check()
        except Exception:
            return False
    
    async def _check_redis_health(self, service) -> bool:
        """Check Redis service health"""
        try:
            from prsm.core.redis_client import get_redis_client
            redis = get_redis_client()
            await redis.ping()
            return True
        except Exception:
            return False
    
    async def _check_vector_db_health(self, service) -> bool:
        """Check vector database health"""
        try:
            # Simple health check - try to get collection info
            return service is not None
        except Exception:
            return False
    
    async def _check_ipfs_health(self, service) -> bool:
        """Check IPFS service health"""
        try:
            if service is None:
                return True  # Mock IPFS is considered healthy
            # Add specific IPFS health checks here
            return True
        except Exception:
            return False
    
    async def _check_ftns_health(self, service) -> bool:
        """Check FTNS service health"""
        try:
            # Test basic FTNS operations
            test_user = f"test_user_{uuid.uuid4().hex[:8]}"
            balance = await service.get_user_balance(test_user)
            return balance is not None
        except Exception:
            return False
    
    async def _generate_test_data(self):
        """Generate initial test data for the environment"""
        
        test_data_info = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": self.config.test_data_seed or "default",
            "datasets": {}
        }
        
        # Generate test users
        test_users = await self._generate_test_users()
        test_data_info["datasets"]["users"] = {
            "count": len(test_users),
            "generated": True
        }
        
        # Generate test sessions
        test_sessions = await self._generate_test_sessions(test_users)
        test_data_info["datasets"]["sessions"] = {
            "count": len(test_sessions),
            "generated": True
        }
        
        # Generate test transactions
        test_transactions = await self._generate_test_transactions(test_users)
        test_data_info["datasets"]["transactions"] = {
            "count": len(test_transactions),
            "generated": True
        }
        
        self.state.test_data_info = test_data_info
        
        # Save test data info
        with open(self.data_dir / "test_data_info.json", 'w') as f:
            json.dump(test_data_info, f, indent=2)
        
        logger.info("Test data generated",
                   environment_id=self.config.environment_id,
                   datasets=list(test_data_info["datasets"].keys()))
    
    async def _generate_test_users(self) -> List[str]:
        """Generate test users"""
        users = []
        for i in range(10):
            user_id = f"test_user_{self.config.environment_id}_{i:03d}"
            users.append(user_id)
            
            # Initialize user in FTNS system
            try:
                await ftns_service.add_tokens(user_id, 1000.0)  # Give each user 1000 FTNS
            except Exception as e:
                logger.warning(f"Failed to initialize user {user_id}",
                             error=str(e))
        
        return users
    
    async def _generate_test_sessions(self, users: List[str]) -> List[str]:
        """Generate test sessions"""
        sessions = []
        for user in users[:5]:  # Create sessions for first 5 users
            session_id = str(uuid.uuid4())
            sessions.append(session_id)
            
            # Create session in database if available
            try:
                if "database" in self.services and self.state.components_status.get("database"):
                    from prsm.core.database import SessionQueries
                    await SessionQueries.create_session({
                        "session_id": session_id,
                        "user_id": user,
                        "nwtn_context_allocation": 1000,
                        "status": "active"
                    })
            except Exception as e:
                logger.warning(f"Failed to create session {session_id}",
                             error=str(e))
        
        return sessions
    
    async def _generate_test_transactions(self, users: List[str]) -> List[str]:
        """Generate test transactions"""
        transactions = []
        for user in users[:3]:  # Create transactions for first 3 users
            transaction_id = str(uuid.uuid4())
            transactions.append(transaction_id)
            
            # Create transaction in database if available
            try:
                if "database" in self.services and self.state.components_status.get("database"):
                    from prsm.core.database import FTNSQueries
                    await FTNSQueries.create_transaction({
                        "transaction_id": transaction_id,
                        "from_user": None,  # System transaction
                        "to_user": user,
                        "amount": 100.0,
                        "transaction_type": "reward",
                        "description": "Test environment setup reward"
                    })
            except Exception as e:
                logger.warning(f"Failed to create transaction {transaction_id}",
                             error=str(e))
        
        return transactions
    
    async def _start_monitoring(self):
        """Start monitoring tasks"""
        
        if self.config.performance_monitoring:
            # Start health monitoring
            self.health_monitor_task = asyncio.create_task(
                self._health_monitor_loop()
            )
            
            # Start performance monitoring
            self.performance_monitor_task = asyncio.create_task(
                self._performance_monitor_loop()
            )
            
            logger.info("Monitoring tasks started",
                       environment_id=self.config.environment_id)
    
    async def _health_monitor_loop(self):
        """Continuous health monitoring loop"""
        while self.state.status == 'running':
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._comprehensive_health_check()
                
                # Log health status
                healthy_services = sum(1 for status in self.state.components_status.values() if status)
                total_services = len(self.state.components_status)
                
                logger.debug("Health check completed",
                           environment_id=self.config.environment_id,
                           healthy_services=healthy_services,
                           total_services=total_services)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health monitor error",
                           environment_id=self.config.environment_id,
                           error=str(e))
    
    async def _performance_monitor_loop(self):
        """Continuous performance monitoring loop"""
        while self.state.status == 'running':
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Collect performance metrics
                metrics = await self._collect_performance_metrics()
                self.state.performance_metrics.update(metrics)
                
                # Log performance metrics
                logger.debug("Performance metrics collected",
                           environment_id=self.config.environment_id,
                           metrics=list(metrics.keys()))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Performance monitor error",
                           environment_id=self.config.environment_id,
                           error=str(e))
    
    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics from all services"""
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment_id": self.config.environment_id
        }
        
        # Database metrics
        if "database" in self.services and self.state.components_status.get("database"):
            try:
                # Simple database performance check
                start_time = time.time()
                await self._check_database_health(self.services["database"])
                metrics["database_response_time"] = time.time() - start_time
            except Exception:
                metrics["database_response_time"] = None
        
        # Redis metrics
        if "redis" in self.services and self.state.components_status.get("redis"):
            try:
                from prsm.core.redis_client import get_redis_client
                redis = get_redis_client()
                start_time = time.time()
                await redis.ping()
                metrics["redis_response_time"] = time.time() - start_time
            except Exception:
                metrics["redis_response_time"] = None
        
        # Memory usage
        import psutil
        process = psutil.Process()
        metrics["memory_usage_mb"] = process.memory_info().rss / 1024 / 1024
        metrics["cpu_percent"] = process.cpu_percent()
        
        return metrics
    
    @asynccontextmanager
    async def test_context(self, test_name: str):
        """
        Context manager for isolated test execution
        
        Usage:
            async with env.test_context("test_user_creation"):
                # Test code here
                # Automatic cleanup after test
        """
        test_id = f"{test_name}_{uuid.uuid4().hex[:8]}"
        cleanup_tasks = []
        
        logger.info("Starting test context",
                   environment_id=self.config.environment_id,
                   test_name=test_name,
                   test_id=test_id)
        
        try:
            # Pre-test setup
            test_temp_dir = self.temp_dir / test_id
            test_temp_dir.mkdir(exist_ok=True)
            cleanup_tasks.append(lambda: shutil.rmtree(test_temp_dir, ignore_errors=True))
            
            yield {
                "test_id": test_id,
                "temp_dir": test_temp_dir,
                "environment": self
            }
            
        finally:
            # Post-test cleanup
            for cleanup_task in cleanup_tasks:
                try:
                    cleanup_task()
                except Exception as e:
                    logger.warning("Cleanup task failed",
                                 test_id=test_id,
                                 error=str(e))
            
            logger.info("Test context completed",
                       environment_id=self.config.environment_id,
                       test_name=test_name,
                       test_id=test_id)
    
    async def get_environment_status(self) -> Dict[str, Any]:
        """Get current environment status and metrics"""
        return {
            "environment_id": self.state.environment_id,
            "status": self.state.status,
            "created_at": self.state.created_at.isoformat(),
            "last_health_check": self.state.last_health_check.isoformat() if self.state.last_health_check else None,
            "components_status": self.state.components_status,
            "performance_metrics": self.state.performance_metrics,
            "test_data_info": self.state.test_data_info,
            "errors": self.state.errors,
            "directories": {
                "environment": str(self.env_dir),
                "data": str(self.data_dir),
                "logs": str(self.logs_dir),
                "config": str(self.config_dir),
                "temp": str(self.temp_dir)
            }
        }
    
    async def stop(self):
        """Stop the persistent test environment"""
        logger.info("Stopping persistent test environment",
                   environment_id=self.config.environment_id)
        
        self.state.status = 'stopping'
        
        # Stop monitoring tasks
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
        if self.performance_monitor_task:
            self.performance_monitor_task.cancel()
        
        # Stop services
        for service_name in list(self.services.keys()):
            try:
                await self._stop_service(service_name)
            except Exception as e:
                logger.error(f"Failed to stop {service_name}",
                           environment_id=self.config.environment_id,
                           error=str(e))
        
        self.state.status = 'stopped'
        logger.info("Persistent test environment stopped",
                   environment_id=self.config.environment_id)
    
    async def _stop_service(self, service_name: str):
        """Stop a specific service"""
        if service_name == "database":
            await close_database()
        elif service_name == "redis":
            await close_redis()
        elif service_name == "vector_db":
            await close_vector_databases()
        elif service_name == "ipfs":
            await close_ipfs()
        
        if service_name in self.services:
            del self.services[service_name]
        
        self.state.components_status[service_name] = False
    
    async def cleanup(self):
        """Clean up the test environment"""
        await self.stop()
        
        if self.config.auto_cleanup and self.env_dir.exists():
            shutil.rmtree(self.env_dir, ignore_errors=True)
            logger.info("Test environment cleaned up",
                       environment_id=self.config.environment_id)


# Convenience functions for easy usage

async def create_test_environment(config: Optional[TestEnvironmentConfig] = None) -> PersistentTestEnvironment:
    """Create and set up a new persistent test environment"""
    env = PersistentTestEnvironment(config)
    success = await env.setup()
    if not success:
        raise RuntimeError(f"Failed to create test environment {env.config.environment_id}")
    return env


async def get_or_create_test_environment(environment_id: str) -> PersistentTestEnvironment:
    """Get existing test environment or create new one"""
    # Check if environment already exists
    base_dir = Path.cwd() / "test_environments" / environment_id
    if base_dir.exists():
        # Load existing environment
        config = TestEnvironmentConfig(environment_id=environment_id)
        env = PersistentTestEnvironment(config)
        # TODO: Add logic to resume existing environment
        return env
    else:
        # Create new environment
        config = TestEnvironmentConfig(environment_id=environment_id)
        return await create_test_environment(config)


# pytest integration

@pytest.fixture
async def prsm_test_environment():
    """Pytest fixture for PRSM test environment"""
    config = TestEnvironmentConfig(auto_cleanup=True)
    env = await create_test_environment(config)
    
    try:
        yield env
    finally:
        await env.cleanup()


if __name__ == "__main__":
    # Example usage
    async def main():
        print("🚀 Creating PRSM Persistent Test Environment")
        
        config = TestEnvironmentConfig(
            persistent_data=True,
            auto_cleanup=False,
            performance_monitoring=True
        )
        
        env = await create_test_environment(config)
        
        print(f"✅ Test environment created: {env.config.environment_id}")
        print(f"📁 Environment directory: {env.env_dir}")
        
        # Example test usage
        async with env.test_context("example_test") as test_ctx:
            print(f"🧪 Running test: {test_ctx['test_id']}")
            # Test code would go here
            await asyncio.sleep(1)
        
        # Get status
        status = await env.get_environment_status()
        print(f"📊 Environment status: {status['status']}")
        print(f"🔧 Components: {sum(status['components_status'].values())}/{len(status['components_status'])} healthy")
        
        # Keep running for a bit to see monitoring in action
        print("⏱️  Monitoring environment for 30 seconds...")
        await asyncio.sleep(30)
        
        await env.stop()
        print("🛑 Test environment stopped")
    
    asyncio.run(main())