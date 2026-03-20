"""
Unit tests for UI chat endpoint (prsm/interface/api/ui_api.py).

Tests verify that send_message() routes through NeuroSymbolicOrchestrator
and handles FTNS deduction correctly.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import HTTPException


class TestUiChat:
    """Test suite for UI chat endpoint."""

    @pytest.mark.asyncio
    async def test_send_message_calls_nwtn_orchestrator(self):
        """send_message routes to NeuroSymbolicOrchestrator, not the hardcoded echo."""
        from prsm.interface.api.ui_api import send_message

        # Mock NeuroSymbolicOrchestrator.solve_task to return real-looking result
        mock_result = {
            "output": "Quantum computing uses qubits to perform calculations...",
            "tokens_used": 120,
            "reward": 0.82,
            "inference_source": "mock",
            "mode": "light",
            "verification_hash": "abc123",
            "trace": [],
        }

        with patch(
            "prsm.compute.nwtn.reasoning.s1_neuro_symbolic.NeuroSymbolicOrchestrator",
            autospec=True
        ) as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.solve_task = AsyncMock(return_value=mock_result)
            mock_orchestrator_class.return_value = mock_orchestrator

            with patch(
                "prsm.core.database.FTNSQueries"
            ) as mock_ftns:
                mock_ftns.execute_atomic_deduct = AsyncMock(
                    return_value={"success": True}
                )

                message_data = {
                    "content": "What is quantum computing?",
                    "user_id": "test-user-123",
                }

                result = await send_message("test-conv-123", message_data)

        # Assert the response structure
        assert result["success"] is True
        assert result["user_message"]["content"] == "What is quantum computing?"
        assert result["ai_response"]["content"] == "Quantum computing uses qubits to perform calculations..."
        # Assert it does NOT contain the old hardcoded echo
        assert not result["ai_response"]["content"].startswith("AI response to:")
        # Assert real metadata from orchestrator
        assert result["ai_response"]["metadata"]["tokens_used"] == 120
        assert result["ai_response"]["metadata"]["inference_source"] == "mock"
        assert result["ai_response"]["metadata"]["verification_hash"] == "abc123"

    @pytest.mark.asyncio
    async def test_send_message_charges_ftns_for_known_user(self):
        """FTNS deduction is triggered when user_id is not 'anonymous'."""
        from prsm.interface.api.ui_api import send_message

        mock_result = {
            "output": "Test response",
            "tokens_used": 100,
            "reward": 0.75,
            "inference_source": "mock",
            "mode": "light",
        }

        with patch(
            "prsm.compute.nwtn.reasoning.s1_neuro_symbolic.NeuroSymbolicOrchestrator",
            autospec=True
        ) as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.solve_task = AsyncMock(return_value=mock_result)
            mock_orchestrator_class.return_value = mock_orchestrator

            mock_deduct = AsyncMock(return_value={"success": True})
            with patch(
                "prsm.core.database.FTNSQueries"
            ) as mock_ftns:
                mock_ftns.execute_atomic_deduct = mock_deduct

                message_data = {
                    "content": "Test prompt",
                    "user_id": "real-user-123",
                }

                result = await send_message("conv-123", message_data)

        # Assert FTNS deduction was called
        assert mock_deduct.called
        call_args = mock_deduct.call_args
        assert call_args.kwargs["user_id"] == "real-user-123"
        assert call_args.kwargs["idempotency_key"].startswith("ui-query:real-user-123:")
        assert call_args.kwargs["transaction_type"] == "query_usage"
        # Assert FTNS charged is reflected in metadata
        assert result["ai_response"]["metadata"]["ftns_charged"] > 0

    @pytest.mark.asyncio
    async def test_send_message_skips_ftns_for_anonymous(self):
        """No FTNS deduction when user_id is 'anonymous'."""
        from prsm.interface.api.ui_api import send_message

        mock_result = {
            "output": "Anonymous response",
            "tokens_used": 100,
            "reward": 0.70,
            "inference_source": "mock",
            "mode": "light",
        }

        with patch(
            "prsm.compute.nwtn.reasoning.s1_neuro_symbolic.NeuroSymbolicOrchestrator",
            autospec=True
        ) as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.solve_task = AsyncMock(return_value=mock_result)
            mock_orchestrator_class.return_value = mock_orchestrator

            mock_deduct = AsyncMock(return_value={"success": True})
            with patch(
                "prsm.core.database.FTNSQueries"
            ) as mock_ftns:
                mock_ftns.execute_atomic_deduct = mock_deduct

                # Test with explicit anonymous user_id
                message_data = {
                    "content": "Test from anonymous",
                    "user_id": "anonymous",
                }

                result = await send_message("conv-456", message_data)

        # Assert FTNS deduction was NOT called
        assert not mock_deduct.called
        # Assert FTNS charged is 0
        assert result["ai_response"]["metadata"]["ftns_charged"] == 0.0

    @pytest.mark.asyncio
    async def test_send_message_non_blocking_on_ftns_failure(self):
        """FTNS failure does not block the chat response."""
        from prsm.interface.api.ui_api import send_message

        mock_result = {
            "output": "Real AI response content",
            "tokens_used": 100,
            "reward": 0.80,
            "inference_source": "mock",
            "mode": "light",
        }

        with patch(
            "prsm.compute.nwtn.reasoning.s1_neuro_symbolic.NeuroSymbolicOrchestrator",
            autospec=True
        ) as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.solve_task = AsyncMock(return_value=mock_result)
            mock_orchestrator_class.return_value = mock_orchestrator

            # Mock FTNS to raise an exception (database unavailable)
            with patch(
                "prsm.core.database.FTNSQueries"
            ) as mock_ftns:
                mock_ftns.execute_atomic_deduct = AsyncMock(
                    side_effect=Exception("Database connection failed")
                )

                message_data = {
                    "content": "Test prompt",
                    "user_id": "real-user-456",
                }

                # Response should still succeed despite FTNS failure
                result = await send_message("conv-789", message_data)

        # Assert response still delivered
        assert result["success"] is True
        assert result["ai_response"]["content"] == "Real AI response content"
        # Assert FTNS charged is 0 due to failure
        assert result["ai_response"]["metadata"]["ftns_charged"] == 0.0
