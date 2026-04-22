"""
Tests for PRSM Python SDK
Comprehensive tests for all SDK modules
"""

import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from pathlib import Path

# Add SDK to path
sdk_path = Path(__file__).parent.parent.parent / "sdks" / "python"
sys.path.insert(0, str(sdk_path))

from prsm_sdk import (
    PRSMClient,
    PRSMResponse,
    QueryRequest,
    FTNSBalance,
    ModelInfo,
    SafetyLevel,
    ModelProvider,
    PRSMError,
    AuthenticationError,
    InsufficientFundsError,
    NetworkError,
)
from prsm_sdk.compute import ComputeClient, JobStatus, JobPriority, JobRequest, JobResponse, JobResult
from prsm_sdk.storage import StorageClient, StorageStatus, ContentType, StorageInfo
from prsm_sdk.governance import GovernanceClient, ProposalStatus, ProposalType, VoteChoice, Proposal, Vote
from prsm_sdk.ftns import FTNSManager, Transaction, StakeInfo
from prsm_sdk.marketplace import ModelMarketplace, ModelCategory
from prsm_sdk.tools import ToolExecutor, ToolCategory, ToolInfo


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_client():
    """Create a mock PRSM client"""
    client = MagicMock(spec=PRSMClient)
    client.base_url = "https://api.prsm.ai/v1"
    client._request = AsyncMock()
    return client


@pytest.fixture
def prsm_client():
    """Create a PRSM client instance"""
    return PRSMClient(
        api_key="test_api_key",
        base_url="https://api.prsm.ai/v1",
        timeout=30
    )


# ============================================================================
# PRSM CLIENT TESTS
# ============================================================================

class TestPRSMClient:
    """Tests for the main PRSMClient class"""
    
    def test_client_initialization(self):
        """Test client initializes correctly"""
        client = PRSMClient(
            api_key="test_key",
            base_url="https://api.prsm.ai/v1",
            timeout=60
        )
        
        assert client.base_url == "https://api.prsm.ai/v1"
        assert client.timeout == 60
        assert client.auth is not None
        assert client.ftns is not None
        assert client.marketplace is not None
        assert client.tools is not None
        assert client.compute is not None
        assert client.storage is not None
        assert client.governance is not None
    
    def test_client_default_values(self):
        """Test client uses correct defaults"""
        client = PRSMClient()
        
        assert client.base_url == "https://api.prsm.ai/v1"
        assert client.timeout == 60
        assert client.max_retries == 3
    
    @pytest.mark.asyncio
    async def test_query_method(self, prsm_client):
        """Test query method"""
        mock_response = {
            "content": "Test response",
            "model_id": "nwtn",
            "provider": "prsm",
            "execution_time": 1.5,
            "token_usage": {"prompt": 10, "completion": 20, "total": 30},
            "ftns_cost": 0.05,
            "safety_status": "moderate",
            "request_id": "req_123",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        prsm_client._request = AsyncMock(return_value=mock_response)
        
        response = await prsm_client.query("Test prompt")
        
        assert isinstance(response, PRSMResponse)
        assert response.content == "Test response"
        assert response.model_id == "nwtn"
    
    @pytest.mark.asyncio
    async def test_health_check(self, prsm_client):
        """Test health check method"""
        prsm_client._request = AsyncMock(return_value={"status": "healthy"})
        
        result = await prsm_client.health_check()
        
        assert result["status"] == "healthy"


# ============================================================================
# COMPUTE CLIENT TESTS
# ============================================================================

class TestComputeClient:
    """Tests for the ComputeClient class"""
    
    def test_compute_client_initialization(self, mock_client):
        """Test compute client initializes correctly"""
        compute = ComputeClient(mock_client)
        assert compute._client == mock_client
    
    @pytest.mark.asyncio
    async def test_submit_job(self, mock_client):
        """Test job submission"""
        mock_response = {
            "job_id": "job_123",
            "status": "queued",
            "created_at": datetime.utcnow().isoformat(),
            "estimated_cost": 0.5,
            "estimated_duration": 10.0
        }
        mock_client._request.return_value = mock_response
        
        compute = ComputeClient(mock_client)
        result = await compute.submit_job(
            prompt="Test prompt",
            model="nwtn",
            max_tokens=1000
        )
        
        assert result.job_id == "job_123"
        assert result.status == JobStatus.QUEUED
        mock_client._request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_job(self, mock_client):
        """Test getting job status"""
        mock_response = {
            "job_id": "job_123",
            "status": "running",
            "request": {
                "prompt": "Test",
                "model": "nwtn",
                "max_tokens": 1000,
                "temperature": 0.7,
                "budget": None,
                "priority": "normal",
                "timeout": 300,
                "context": None,
                "tools": None
            },
            "progress": 0.5,
            "created_at": datetime.utcnow().isoformat(),
            "started_at": datetime.utcnow().isoformat()
        }
        mock_client._request.return_value = mock_response
        
        compute = ComputeClient(mock_client)
        result = await compute.get_job("job_123")
        
        assert result.job_id == "job_123"
        assert result.status == JobStatus.RUNNING
        assert result.progress == 0.5
    
    @pytest.mark.asyncio
    async def test_cancel_job(self, mock_client):
        """Test cancelling a job"""
        mock_client._request.return_value = {"cancelled": True}
        
        compute = ComputeClient(mock_client)
        result = await compute.cancel_job("job_123")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_list_jobs(self, mock_client):
        """Test listing jobs"""
        mock_response = {
            "jobs": [],
            "total": 0,
            "offset": 0,
            "limit": 10
        }
        mock_client._request.return_value = mock_response
        
        compute = ComputeClient(mock_client)
        result = await compute.list_jobs()
        
        assert result.total == 0
        assert result.jobs == []


# ============================================================================
# STORAGE CLIENT TESTS
# ============================================================================

class TestStorageClient:
    """Tests for the StorageClient class"""
    
    def test_storage_client_initialization(self, mock_client):
        """Test storage client initializes correctly"""
        storage = StorageClient(mock_client)
        assert storage._client == mock_client
    
    @pytest.mark.asyncio
    async def test_get_info(self, mock_client):
        """Test getting storage info"""
        mock_response = {
            "cid": "QmTest123",
            "content_type": "file",
            "size": 1024,
            "tags": [],
            "status": "available",
            "is_public": False,
            "is_pinned": True,
            "replication": 3,
            "created_at": datetime.utcnow().isoformat(),
            "owner": "0x123",
            "access_count": 5
        }
        mock_client._request.return_value = mock_response
        
        storage = StorageClient(mock_client)
        result = await storage.get_info("QmTest123")
        
        assert result.cid == "QmTest123"
        assert result.status == StorageStatus.AVAILABLE
        assert result.is_pinned is True
    
    @pytest.mark.asyncio
    async def test_pin(self, mock_client):
        """Test pinning content"""
        mock_response = {
            "cid": "QmTest123",
            "pinned_at": datetime.utcnow().isoformat(),
            "size": 1024,
            "replication": 3,
            "monthly_cost": 0.1
        }
        mock_client._request.return_value = mock_response
        
        storage = StorageClient(mock_client)
        result = await storage.pin("QmTest123")
        
        assert result.cid == "QmTest123"
        assert result.replication == 3


# ============================================================================
# GOVERNANCE CLIENT TESTS
# ============================================================================

class TestGovernanceClient:
    """Tests for the GovernanceClient class"""
    
    def test_governance_client_initialization(self, mock_client):
        """Test governance client initializes correctly"""
        governance = GovernanceClient(mock_client)
        assert governance._client == mock_client
    
    @pytest.mark.asyncio
    async def test_create_proposal(self, mock_client):
        """Test creating a proposal"""
        mock_response = {
            "proposal_id": "prop_123",
            "title": "Test Proposal",
            "description": "This is a test proposal description that meets the minimum length requirement of 50 characters",
            "proposal_type": "parameter_change",
            "status": "draft",
            "proposer": "0x123",
            "created_at": datetime.utcnow().isoformat(),
            "voting_starts": datetime.utcnow().isoformat(),
            "voting_ends": datetime.utcnow().isoformat(),
            "quorum": 0.1,
            "threshold": 0.5,
            "votes_yes": 0,
            "votes_no": 0,
            "votes_abstain": 0,
            "total_voters": 0,
            "parameters": {},
            "metadata": {}
        }
        mock_client._request.return_value = mock_response
        
        governance = GovernanceClient(mock_client)
        result = await governance.create_proposal(
            title="Test Proposal",
            description="This is a test proposal description that meets the minimum length requirement of 50 characters",
            proposal_type=ProposalType.PARAMETER_CHANGE
        )
        
        assert result.proposal_id == "prop_123"
        assert result.title == "Test Proposal"
        assert result.status == ProposalStatus.DRAFT
    
    @pytest.mark.asyncio
    async def test_vote(self, mock_client):
        """Test casting a vote"""
        mock_response = {
            "vote_id": "vote_123",
            "proposal_id": "prop_123",
            "voter": "0x123",
            "choice": "yes",
            "voting_power": 100.0,
            "timestamp": datetime.utcnow().isoformat()
        }
        mock_client._request.return_value = mock_response
        
        governance = GovernanceClient(mock_client)
        result = await governance.vote("prop_123", VoteChoice.YES)
        
        assert result.vote_id == "vote_123"
        assert result.choice == VoteChoice.YES
        assert result.voting_power == 100.0
    
    @pytest.mark.asyncio
    async def test_list_proposals(self, mock_client):
        """Test listing proposals"""
        mock_response = {
            "proposals": []
        }
        mock_client._request.return_value = mock_response
        
        governance = GovernanceClient(mock_client)
        result = await governance.list_proposals()
        
        assert result == []


# ============================================================================
# FTNS MANAGER TESTS
# ============================================================================

class TestFTNSManager:
    """Tests for the FTNSManager class"""
    
    def test_ftns_manager_initialization(self, mock_client):
        """Test FTNS manager initializes correctly"""
        ftns = FTNSManager(mock_client)
        assert ftns._client == mock_client
    
    @pytest.mark.asyncio
    async def test_get_balance(self, mock_client):
        """Test getting balance"""
        mock_response = {
            "total_balance": 100.0,
            "available_balance": 80.0,
            "reserved_balance": 20.0,
            "staked_balance": 0.0,
            "earned_today": 5.0,
            "spent_today": 2.0,
            "last_updated": datetime.utcnow().isoformat()
        }
        mock_client._request.return_value = mock_response
        
        ftns = FTNSManager(mock_client)
        result = await ftns.get_balance()
        
        assert result.total_balance == 100.0
        assert result.available_balance == 80.0
    
    @pytest.mark.asyncio
    async def test_transfer_insufficient_funds(self, mock_client):
        """Test transfer with insufficient funds"""
        mock_response = {
            "total_balance": 10.0,
            "available_balance": 10.0,
            "reserved_balance": 0.0,
            "staked_balance": 0.0,
            "earned_today": 0.0,
            "spent_today": 0.0,
            "last_updated": datetime.utcnow().isoformat()
        }
        mock_client._request.return_value = mock_response
        
        ftns = FTNSManager(mock_client)
        
        with pytest.raises(InsufficientFundsError):
            await ftns.transfer("0xabc", 100.0)


# ============================================================================
# EXCEPTION TESTS
# ============================================================================

class TestExceptions:
    """Tests for SDK exceptions"""
    
    def test_prsm_error(self):
        """Test PRSMError"""
        error = PRSMError("Test error", "TEST_ERROR", {"detail": "test"})
        
        assert str(error) == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.details == {"detail": "test"}
    
    def test_authentication_error(self):
        """Test AuthenticationError"""
        error = AuthenticationError("Invalid API key")
        
        assert str(error) == "Invalid API key"
        assert error.error_code == "AUTH_ERROR"
    
    def test_insufficient_funds_error(self):
        """Test InsufficientFundsError"""
        error = InsufficientFundsError(100.0, 50.0)
        
        assert "100" in str(error)
        assert "50" in str(error)
        assert error.error_code == "INSUFFICIENT_FUNDS"
    
    def test_network_error(self):
        """Test NetworkError"""
        error = NetworkError("Connection failed")
        
        assert str(error) == "Connection failed"
        assert error.error_code == "NETWORK_ERROR"


# ============================================================================
# MODEL TESTS
# ============================================================================

class TestModels:
    """Tests for SDK data models"""
    
    def test_query_request(self):
        """Test QueryRequest model"""
        request = QueryRequest(
            prompt="Test prompt",
            model_id="nwtn",
            max_tokens=1000,
            temperature=0.7
        )
        
        assert request.prompt == "Test prompt"
        assert request.model_id == "nwtn"
        assert request.max_tokens == 1000
        assert request.temperature == 0.7
    
    def test_ftns_balance(self):
        """Test FTNSBalance model"""
        balance = FTNSBalance(
            total_balance=100.0,
            available_balance=80.0,
            reserved_balance=20.0,
            staked_balance=0.0,
            earned_today=5.0,
            spent_today=2.0,
            last_updated=datetime.utcnow()
        )
        
        assert balance.total_balance == 100.0
        assert balance.available_balance == 80.0
    
    def test_model_info(self):
        """Test ModelInfo model"""
        model = ModelInfo(
            id="nwtn",
            name="NWTN",
            provider=ModelProvider.PRSM,
            description="Test model",
            category=ModelCategory.LANGUAGE,
            capabilities=["reasoning"],
            cost_per_token=0.001,
            max_tokens=4096,
            context_window=8192,
            is_available=True,
            performance_rating=0.9,
            safety_rating=0.95,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        assert model.id == "nwtn"
        assert model.provider == ModelProvider.PRSM
        assert model.is_available is True


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])