"""
Python SDK integration tests — runs against a live PRSM server.

Prerequisites:
    - PRSM node running at http://localhost:8000 (or PRSM_TEST_URL env var)
    - PRSM_TEST_API_KEY set, or test user created via /auth/register

Skip gracefully if server is not reachable.

Usage:
    # Unit tests (no server needed):
    pytest tests/test_client.py tests/test_models.py

    # Integration tests (requires running node):
    prsm node start &
    PRSM_TEST_API_KEY=your_key pytest tests/test_integration.py -v
"""
import pytest
import asyncio
import os
import aiohttp
from typing import Optional

# Import SDK components
from prsm_sdk import PRSMClient
from prsm_sdk.exceptions import AuthenticationError, PRSMError, NetworkError

# Configuration from environment
PRSM_URL = os.getenv("PRSM_TEST_URL", "http://localhost:8000")
PRSM_KEY = os.getenv("PRSM_TEST_API_KEY", "")


def prsm_server_available() -> bool:
    """Check if a PRSM server is available for integration testing."""
    try:
        import aiohttp
        import asyncio

        async def check():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{PRSM_URL}/health",
                        timeout=aiohttp.ClientTimeout(total=2.0)
                    ) as resp:
                        return resp.status == 200
            except Exception:
                return False

        return asyncio.get_event_loop().run_until_complete(check())
    except Exception:
        return False


@pytest.mark.skipif(
    not prsm_server_available(),
    reason="PRSM server not running at localhost:8000 — start node with `prsm node start`"
)
class TestPRSMClientIntegration:
    """Integration tests that require a live PRSM server."""

    @pytest.fixture
    async def client(self):
        """Create an authenticated client for testing."""
        async with PRSMClient(api_key=PRSM_KEY, base_url=PRSM_URL) as c:
            yield c

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test that the health endpoint returns a valid response."""
        health = await client.health_check()
        assert health is not None
        assert "status" in health
        assert health["status"] in ("healthy", "degraded", "ok")

    @pytest.mark.asyncio
    async def test_get_ftns_balance(self, client):
        """Test retrieving FTNS balance."""
        balance = await client.ftns.get_balance()
        assert balance is not None
        assert hasattr(balance, "total_balance")
        assert hasattr(balance, "available_balance")
        assert float(balance.available_balance) >= 0

    @pytest.mark.asyncio
    async def test_list_marketplace_models(self, client):
        """Test listing models from the marketplace."""
        models = await client.marketplace.list_models(limit=5)
        assert isinstance(models, list)
        # Each model should have required fields
        for model in models[:3]:  # Check first 3
            assert hasattr(model, "id")
            assert hasattr(model, "name")

    @pytest.mark.asyncio
    async def test_search_marketplace(self, client):
        """Test searching the marketplace."""
        results = await client.marketplace.search_models(
            query="gpt",
            limit=5
        )
        assert results is not None
        assert hasattr(results, "models")
        assert isinstance(results.models, list)

    @pytest.mark.asyncio
    async def test_estimate_query_cost(self, client):
        """Test estimating the cost of a query."""
        cost = await client.estimate_cost(
            prompt="What is machine learning?",
            model_id=None
        )
        assert cost is not None
        assert isinstance(cost, (int, float))
        assert cost >= 0

    @pytest.mark.asyncio
    async def test_list_available_tools(self, client):
        """Test listing available MCP tools."""
        tools = await client.tools.list_available()
        assert isinstance(tools, list)
        # Each tool should have required fields
        for tool in tools[:3]:
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")

    @pytest.mark.asyncio
    async def test_storage_operations(self, client):
        """Test basic storage operations."""
        # Upload a small test file
        test_data = b"PRSM integration test data"
        result = await client.storage.upload_bytes(
            data=test_data,
            filename="test_integration.txt",
            content_type="document"
        )

        assert result is not None
        assert hasattr(result, "cid")
        assert result.cid != ""

        # Get info about uploaded content
        info = await client.storage.get_info(result.cid)
        assert info is not None
        assert info.cid == result.cid

        # Clean up - delete the test content
        deleted = await client.storage.delete(result.cid)
        assert deleted is True


@pytest.mark.skipif(
    not prsm_server_available(),
    reason="PRSM server not running at localhost:8000"
)
class TestPRSMClientAuthentication:
    """Tests for authentication behavior."""

    @pytest.mark.asyncio
    async def test_invalid_api_key_raises(self):
        """Test that an invalid API key raises an authentication error."""
        async with PRSMClient(api_key="invalid_test_key_12345", base_url=PRSM_URL) as client:
            with pytest.raises((AuthenticationError, PRSMError)):
                await client.ftns.get_balance()

    @pytest.mark.asyncio
    async def test_empty_api_key_handled(self):
        """Test that empty API key is handled gracefully."""
        async with PRSMClient(api_key="", base_url=PRSM_URL) as client:
            # This should either work with anonymous access or raise an error
            try:
                await client.health_check()
            except (AuthenticationError, PRSMError):
                pass  # Expected behavior for anonymous access


@pytest.mark.skipif(
    not prsm_server_available(),
    reason="PRSM server not running at localhost:8000"
)
class TestPRSMClientGovernance:
    """Tests for governance functionality."""

    @pytest.fixture
    async def client(self):
        """Create an authenticated client for testing."""
        async with PRSMClient(api_key=PRSM_KEY, base_url=PRSM_URL) as c:
            yield c

    @pytest.mark.asyncio
    async def test_list_proposals(self, client):
        """Test listing governance proposals."""
        proposals = await client.governance.list_proposals(limit=5)
        assert proposals is not None
        assert isinstance(proposals, list)
        # Each proposal should have required fields
        for proposal in proposals[:3]:
            assert hasattr(proposal, "id") or hasattr(proposal, "proposal_id")

    @pytest.mark.asyncio
    async def test_get_proposal(self, client):
        """Test getting a specific proposal if one exists."""
        proposals = await client.governance.list_proposals(limit=1)
        if proposals:
            proposal_id = proposals[0].id if hasattr(proposals[0], "id") else proposals[0].proposal_id
            proposal = await client.governance.get_proposal(proposal_id)
            assert proposal is not None


@pytest.mark.skipif(
    not prsm_server_available(),
    reason="PRSM server not running at localhost:8000"
)
class TestPRSMClientCompute:
    """Tests for compute job functionality."""

    @pytest.fixture
    async def client(self):
        """Create an authenticated client for testing."""
        async with PRSMClient(api_key=PRSM_KEY, base_url=PRSM_URL) as c:
            yield c

    @pytest.mark.asyncio
    async def test_submit_compute_job(self, client):
        """Test submitting a compute job."""
        job = await client.compute.submit_job(
            job_type="benchmark",
            parameters={"iterations": 1},
        )
        assert job is not None
        assert hasattr(job, "job_id")
        assert job.status in ("submitted", "queued", "running", "pending")

    @pytest.mark.asyncio
    async def test_list_compute_jobs(self, client):
        """Test listing compute jobs."""
        jobs = await client.compute.list_jobs(limit=5)
        assert jobs is not None
        assert isinstance(jobs, list)


# =============================================================================
# Test Runner
# =============================================================================

if __name__ == "__main__":
    # Run integration tests with verbose output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])
