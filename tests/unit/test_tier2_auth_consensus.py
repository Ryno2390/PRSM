"""Tests for Tier 2: API authentication and result consensus."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from prsm.api.auth_middleware import (
    generate_api_key,
    hash_api_key,
    NodeAuthMiddleware,
    PUBLIC_ENDPOINTS,
    PROTECTED_PREFIXES,
)


class TestAPIAuthentication:
    def test_generate_api_key(self):
        key = generate_api_key()
        assert key.startswith("prsm_")
        assert len(key) > 20

    def test_hash_api_key_deterministic(self):
        key = "prsm_test_key_123"
        h1 = hash_api_key(key)
        h2 = hash_api_key(key)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_hash_different_keys_differ(self):
        h1 = hash_api_key("key_a")
        h2 = hash_api_key("key_b")
        assert h1 != h2

    def test_public_endpoints_defined(self):
        assert "/" in PUBLIC_ENDPOINTS
        assert "/health" in PUBLIC_ENDPOINTS
        assert "/status" in PUBLIC_ENDPOINTS

    def test_protected_prefixes_defined(self):
        assert "/settler/" in PROTECTED_PREFIXES
        assert "/content/upload" in PROTECTED_PREFIXES
        assert "/compute/forge" in PROTECTED_PREFIXES

    def test_auth_disabled_when_no_key(self):
        """When no API key hash is provided, auth is disabled."""
        middleware = NodeAuthMiddleware(MagicMock(), api_key_hash="")
        assert not middleware.auth_enabled

    def test_auth_enabled_when_key_set(self):
        middleware = NodeAuthMiddleware(MagicMock(), api_key_hash="abc123")
        assert middleware.auth_enabled


class TestResultConsensusIntegration:
    @pytest.mark.asyncio
    async def test_swarm_coordinator_accepts_consensus_param(self):
        from prsm.compute.swarm.coordinator import SwarmCoordinator
        mock_dispatcher = AsyncMock()
        mock_consensus = MagicMock()

        coordinator = SwarmCoordinator(
            dispatcher=mock_dispatcher,
            result_consensus=mock_consensus,
        )
        assert coordinator.result_consensus is mock_consensus

    @pytest.mark.asyncio
    async def test_swarm_coordinator_works_without_consensus(self):
        from prsm.compute.swarm.coordinator import SwarmCoordinator
        mock_dispatcher = AsyncMock()

        coordinator = SwarmCoordinator(dispatcher=mock_dispatcher)
        assert coordinator.result_consensus is None

    @pytest.mark.asyncio
    async def test_swarm_execute_with_consensus(self):
        import base64
        from prsm.compute.swarm.coordinator import SwarmCoordinator
        from prsm.compute.agents.models import AgentManifest, DispatchStatus

        mock_dispatcher = AsyncMock()
        mock_dispatcher.create_agent = MagicMock(side_effect=lambda **kw: MagicMock(agent_id="a1"))
        mock_dispatcher.dispatch = AsyncMock(return_value=MagicMock(status=DispatchStatus.COMPLETED))
        mock_dispatcher.select_and_transfer = AsyncMock(return_value=True)
        mock_dispatcher.wait_for_result = AsyncMock(return_value={
            "status": "success",
            "output_b64": base64.b64encode(b'{"v": 1}').decode(),
            "pcu": 0.1,
        })

        mock_consensus = MagicMock()
        coordinator = SwarmCoordinator(dispatcher=mock_dispatcher, result_consensus=mock_consensus)

        WASM = b"\x00asm\x01\x00\x00\x00"
        job = coordinator.create_swarm_job(
            query="test",
            shard_content_ids=["QmA"],
            wasm_binary=WASM,
            manifest=AgentManifest(required_content_ids=[], min_hardware_tier="t1"),
            budget_ftns=5.0,
        )

        result = await coordinator.execute(job)
        assert result.shards_completed == 1
