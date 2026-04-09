"""
Unit tests for the UI chat endpoint (prsm/interface/api/ui_api.py).

NWTN reasoning (NeuroSymbolicOrchestrator / System 1+2) was removed in the
v1.6.0 scope-alignment sprint. The UI chat endpoint now returns HTTP 501
Not Implemented until an in-scope inference surface is re-introduced on
top of the Ring-9 training pipeline. These tests pin that contract.

See docs/2026-04-09-v1.6-scope-alignment-design.md.
"""
import pytest
from fastapi import HTTPException


class TestUiChat:
    """Pin the 501 Not Implemented contract for the UI chat endpoint."""

    @pytest.mark.asyncio
    async def test_send_message_returns_501(self):
        """send_message() raises HTTP 501 for any message body."""
        from prsm.interface.api.ui_api import send_message

        message_data = {
            "content": "What is quantum computing?",
            "user_id": "test-user-123",
        }

        with pytest.raises(HTTPException) as exc_info:
            await send_message("test-conv-123", message_data)

        assert exc_info.value.status_code == 501
        assert "v1.6.0" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_send_message_returns_501_for_anonymous(self):
        """Anonymous requests are also rejected with 501 (no special case)."""
        from prsm.interface.api.ui_api import send_message

        message_data = {
            "content": "Test from anonymous",
            "user_id": "anonymous",
        }

        with pytest.raises(HTTPException) as exc_info:
            await send_message("conv-456", message_data)

        assert exc_info.value.status_code == 501

    @pytest.mark.asyncio
    async def test_send_message_returns_501_without_user_id(self):
        """Missing user_id is still rejected with 501, not 400 or 500."""
        from prsm.interface.api.ui_api import send_message

        message_data = {"content": "Test"}

        with pytest.raises(HTTPException) as exc_info:
            await send_message("conv-789", message_data)

        assert exc_info.value.status_code == 501
