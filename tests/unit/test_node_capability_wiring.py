"""Tests for node.py capability announcement wiring."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestCapabilityWiring:

    @pytest.mark.asyncio
    async def test_storage_provider_receives_transport_and_discovery(self):
        import inspect
        from prsm.node.storage_provider import StorageProvider
        sig = inspect.signature(StorageProvider.__init__)
        params = list(sig.parameters.keys())
        assert "transport" in params
        assert "discovery" in params

    @pytest.mark.asyncio
    async def test_compute_requester_receives_discovery(self):
        import inspect
        from prsm.node.compute_requester import ComputeRequester
        sig = inspect.signature(ComputeRequester.__init__)
        params = list(sig.parameters.keys())
        assert "discovery" in params
