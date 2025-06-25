"""
PRSM Persistent Test Environment Package
=======================================

🎯 PURPOSE:
Provides persistent test environments for PRSM system validation, development,
and continuous integration. Supports isolated test execution, comprehensive
monitoring, and automated environment management.

🚀 KEY COMPONENTS:
- PersistentTestEnvironment: Core environment management
- TestRunner: Comprehensive test execution and reporting
- Environment lifecycle management and monitoring
- Automated test data generation and cleanup
- Performance benchmarking and health monitoring

📦 USAGE:
    from tests.environment import create_test_environment, TestRunner
    
    # Create persistent environment
    env = await create_test_environment()
    
    # Run comprehensive tests
    runner = TestRunner()
    results = await runner.run_comprehensive_tests(env.config.environment_id)
"""

from .persistent_test_environment import (
    PersistentTestEnvironment,
    TestEnvironmentConfig,
    EnvironmentState,
    create_test_environment,
    get_or_create_test_environment
)

from .test_runner import TestRunner

__all__ = [
    "PersistentTestEnvironment",
    "TestEnvironmentConfig", 
    "EnvironmentState",
    "create_test_environment",
    "get_or_create_test_environment",
    "TestRunner"
]

__version__ = "1.0.0"