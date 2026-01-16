"""
Integration Tests
=================

Integration tests that require real external services (PostgreSQL, Redis).

These tests are skipped by default if the required environment variables
are not set.

Required Environment Variables:
- DATABASE_URL or TEST_DATABASE_URL: PostgreSQL connection string
- REDIS_URL: Redis connection string (default: redis://localhost:6379/1)

Run integration tests:
    pytest tests/integration/ -v --tb=short

Skip integration tests:
    pytest tests/ --ignore=tests/integration/
"""
