"""
Unit tests for FTNS integration in distillation and alpha user management.

These tests verify that the deprecated FTNSService has been properly replaced
with FTNSQueries and AtomicFTNSService throughout the codebase.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID
from decimal import Decimal


class TestDistillationFTNS:
    """Tests for FTNS operations in the distillation orchestrator."""

    @pytest.mark.asyncio
    async def test_reserve_ftns_calls_atomic_deduct(self):
        """
        _reserve_ftns calls FTNSQueries.execute_atomic_deduct (not FTNSService).
        
        Verifies:
        - execute_atomic_deduct is called with correct parameters
        - idempotency_key starts with "distillation-reserve:"
        - Returns the transaction_id on success
        """
        # Mock FTNSQueries.execute_atomic_deduct → success=True, transaction_id="tx-123"
        with patch('prsm.core.database.FTNSQueries.execute_atomic_deduct', new_callable=AsyncMock) as mock_deduct:
            mock_deduct.return_value = {"success": True, "transaction_id": "tx-123"}
            
            # Import orchestrator after patching
            from prsm.compute.distillation.orchestrator import DistillationOrchestrator
            
            # Create orchestrator with mocked dependencies
            mock_circuit_breaker = MagicMock()
            mock_model_registry = MagicMock()
            mock_ipfs_client = MagicMock()
            mock_proposal_manager = MagicMock()
            
            orchestrator = DistillationOrchestrator(
                circuit_breaker=mock_circuit_breaker,
                model_registry=mock_model_registry,
                ipfs_client=mock_ipfs_client,
                proposal_manager=mock_proposal_manager
            )
            
            # Call _reserve_ftns("user-1", 100, uuid4())
            job_id = uuid4()
            result = await orchestrator._reserve_ftns("user-1", 100, job_id)
            
            # Assert execute_atomic_deduct called with correct user_id, amount, idempotency_key
            mock_deduct.assert_called_once()
            call_kwargs = mock_deduct.call_args.kwargs
            assert call_kwargs["user_id"] == "user-1"
            assert call_kwargs["amount"] == 100.0
            assert call_kwargs["description"] == "Distillation job — upfront reservation"
            assert call_kwargs["transaction_type"] == "distillation_reservation"
            
            # Assert idempotency_key starts with "distillation-reserve:"
            assert call_kwargs["idempotency_key"].startswith("distillation-reserve:")
            
            # Assert returns "tx-123"
            assert result == "tx-123"

    @pytest.mark.asyncio
    async def test_finalize_charges_is_noop(self):
        """
        _finalize_ftns_charges no longer raises (was crashing every completed job).
        
        Verifies:
        - No exception is raised
        - Returns None
        - This was the CRITICAL bug: finalize_charge caused AttributeError
        """
        from prsm.compute.distillation.orchestrator import DistillationOrchestrator
        from prsm.compute.distillation.models import DistillationJob, DistillationStatus
        
        # Create orchestrator with mocked dependencies
        mock_circuit_breaker = MagicMock()
        mock_model_registry = MagicMock()
        mock_ipfs_client = MagicMock()
        mock_proposal_manager = MagicMock()
        
        orchestrator = DistillationOrchestrator(
            circuit_breaker=mock_circuit_breaker,
            model_registry=mock_model_registry,
            ipfs_client=mock_ipfs_client,
            proposal_manager=mock_proposal_manager
        )
        
        # Create a mock DistillationJob with ftns_spent=50, user_id="user-1"
        job = DistillationJob(
            job_id=uuid4(),
            user_id="user-1",
            status=DistillationStatus.COMPLETED,
            ftns_spent=50
        )
        
        # Call _finalize_ftns_charges(job)
        result = await orchestrator._finalize_ftns_charges(job)
        
        # Assert no exception raised
        # Assert returns None
        assert result is None

    @pytest.mark.asyncio
    async def test_revenue_sharing_uses_atomic_transfer(self):
        """
        _distribute_revenue_sharing calls FTNSQueries.execute_atomic_transfer.
        
        Verifies:
        - execute_atomic_transfer is called (not transfer_tokens)
        - from_user_id is the request.user_id
        - idempotency_key starts with "revenue-share:"
        """
        with patch('prsm.core.database.FTNSQueries.execute_atomic_transfer', new_callable=AsyncMock) as mock_transfer:
            mock_transfer.return_value = {"success": True, "transaction_id": "tx-share-123"}
            
            from prsm.compute.distillation.orchestrator import DistillationOrchestrator
            from prsm.compute.distillation.models import DistillationJob, DistillationRequest, DistillationStatus
            
            # Create orchestrator with mocked dependencies
            mock_circuit_breaker = MagicMock()
            mock_model_registry = MagicMock()
            mock_ipfs_client = MagicMock()
            mock_proposal_manager = MagicMock()
            
            orchestrator = DistillationOrchestrator(
                circuit_breaker=mock_circuit_breaker,
                model_registry=mock_model_registry,
                ipfs_client=mock_ipfs_client,
                proposal_manager=mock_proposal_manager
            )
            
            # Mock _get_teacher_model_owner → returns "teacher-owner-1"
            orchestrator._get_teacher_model_owner = AsyncMock(return_value="teacher-owner-1")
            
            # Build mock request with revenue_sharing=0.1 and teacher_models
            request = DistillationRequest(
                user_id="user-1",
                domain="test-domain",
                teacher_model="teacher-model-1",
                teacher_models=[{"model": "teacher-model-1", "weight": 1.0}],
                budget_ftns=1000,
                revenue_sharing=0.1
            )
            
            # Create mock job
            job = DistillationJob(
                job_id=uuid4(),
                user_id="user-1",
                status=DistillationStatus.COMPLETED,
                ftns_spent=100
            )
            
            # Call _distribute_revenue_sharing(job, request, share_amount=10)
            await orchestrator._distribute_revenue_sharing(job, request, share_amount=10)
            
            # Assert execute_atomic_transfer called with from_user_id=request.user_id
            mock_transfer.assert_called_once()
            call_kwargs = mock_transfer.call_args.kwargs
            assert call_kwargs["from_user_id"] == "user-1"
            assert call_kwargs["to_user_id"] == "teacher-owner-1"
            assert call_kwargs["amount"] == 10.0
            
            # Assert idempotency_key starts with "revenue-share:"
            assert call_kwargs["idempotency_key"].startswith("revenue-share:")

    @pytest.mark.asyncio
    async def test_alpha_user_account_creation_uses_ensure_account_exists(self):
        """
        register_alpha_user calls AtomicFTNSService.ensure_account_exists.
        
        Verifies:
        - ensure_account_exists is called (not create_user_account)
        - Called with user_id and initial_balance as Decimal
        """
        with patch('prsm.compute.alpha.user_management.AtomicFTNSService') as mock_atomic_service_class:
            # Mock AtomicFTNSService.ensure_account_exists → True
            mock_instance = MagicMock()
            mock_instance.ensure_account_exists = AsyncMock(return_value=True)
            mock_atomic_service_class.return_value = mock_instance
            
            from prsm.compute.alpha.user_management import AlphaUserManager
            
            # Create user manager
            user_manager = AlphaUserManager()
            
            # Mock other dependencies (safe_send_email etc.) - not needed for this test
            # Call user_manager.register_alpha_user(...)
            result = await user_manager.register_alpha_user(
                email="test@example.com",
                name="Test User",
                organization="Test Org",
                user_type_str="software_engineer",
                use_case="Testing FTNS integration",
                technical_background="Python development",
                experience_level="intermediate",
                collaboration_consent=True,
                marketing_consent=False
            )
            
            # Assert ensure_account_exists called with the user_id and initial_balance
            mock_instance.ensure_account_exists.assert_called_once()
            call_kwargs = mock_instance.ensure_account_exists.call_args.kwargs
            
            # Verify user_id is passed
            assert "user_id" in call_kwargs
            assert call_kwargs["user_id"].startswith("alpha_")
            
            # Verify initial_balance is passed as Decimal
            assert "initial_balance" in call_kwargs
            assert isinstance(call_kwargs["initial_balance"], Decimal)
            assert call_kwargs["initial_balance"] == Decimal(str(user_manager.initial_ftns_grant))
            
            # Verify account_type is "user"
            assert call_kwargs.get("account_type") == "user"
            
            # Verify registration was successful
            assert result["success"] is True
