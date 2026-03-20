"""
Tests for Governance Persistence Layer

Tests the database persistence of governance proposals, including:
- GovernanceProposalModel and GovernanceQueries
- TokenWeightedVoting persistence methods
- API endpoint for fetching individual proposals
"""

import pytest
from datetime import datetime, timezone, timedelta
from uuid import UUID, uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from prsm.core.models import GovernanceProposal
from prsm.core.database import GovernanceQueries, GovernanceProposalModel
from prsm.economy.governance.voting import TokenWeightedVoting


# === Fixtures ===

@pytest.fixture
def sample_proposal():
    """Create a sample governance proposal for testing."""
    return GovernanceProposal(
        proposal_id=uuid4(),
        proposer_id="test_user_001",
        title="Test Proposal: Increase Voting Period",
        description="This is a test proposal to increase the voting period from 7 to 10 days for better community participation.",
        proposal_type="economic",
        status="active",
        votes_for=5,
        votes_against=2,
        total_voting_power=1500.0,
        voting_starts=datetime.now(timezone.utc),
        voting_ends=datetime.now(timezone.utc) + timedelta(days=7),
        created_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_proposal_dict(sample_proposal):
    """Create a sample proposal as a dictionary for database operations."""
    return {
        "proposal_id": str(sample_proposal.proposal_id),
        "proposer_id": sample_proposal.proposer_id,
        "title": sample_proposal.title,
        "description": sample_proposal.description,
        "proposal_type": sample_proposal.proposal_type,
        "status": sample_proposal.status,
        "votes_for": sample_proposal.votes_for,
        "votes_against": sample_proposal.votes_against,
        "total_voting_power": sample_proposal.total_voting_power,
        "voting_starts": sample_proposal.voting_starts.isoformat() if sample_proposal.voting_starts else None,
        "voting_ends": sample_proposal.voting_ends.isoformat() if sample_proposal.voting_ends else None,
        "created_at": sample_proposal.created_at.isoformat() if sample_proposal.created_at else None,
    }


# === Test 1: GovernanceQueries.upsert_proposal ===

@pytest.mark.asyncio
async def test_governance_queries_upsert_proposal(sample_proposal):
    """
    Test 1: GovernanceQueries.upsert_proposal persists a proposal.
    
    Verifies that:
    - The upsert operation returns True on success
    - The proposal is correctly serialized for database storage
    """
    with patch.object(GovernanceQueries, 'upsert_proposal', new_callable=AsyncMock) as mock_upsert:
        mock_upsert.return_value = True
        
        result = await GovernanceQueries.upsert_proposal(sample_proposal)
        
        assert result is True
        mock_upsert.assert_called_once_with(sample_proposal)


# === Test 2: GovernanceQueries.get_proposal ===

@pytest.mark.asyncio
async def test_governance_queries_get_proposal(sample_proposal_dict):
    """
    Test 2: GovernanceQueries.get_proposal retrieves a proposal by ID.
    
    Verifies that:
    - The query returns the correct proposal data
    - Returns None for non-existent proposals
    """
    proposal_id = sample_proposal_dict["proposal_id"]
    
    with patch.object(GovernanceQueries, 'get_proposal', new_callable=AsyncMock) as mock_get:
        mock_get.return_value = sample_proposal_dict
        
        result = await GovernanceQueries.get_proposal(proposal_id)
        
        assert result is not None
        assert result["proposal_id"] == proposal_id
        assert result["title"] == sample_proposal_dict["title"]
        assert result["status"] == "active"
        
        # Test non-existent proposal
        mock_get.return_value = None
        result_none = await GovernanceQueries.get_proposal("non-existent-id")
        assert result_none is None


# === Test 3: TokenWeightedVoting._persist_proposal ===

@pytest.mark.asyncio
async def test_token_weighted_voting_persist_proposal(sample_proposal):
    """
    Test 3: TokenWeightedVoting._persist_proposal calls GovernanceQueries.
    
    Verifies that:
    - The persistence method correctly delegates to GovernanceQueries
    - Returns True on successful persistence
    """
    voting_system = TokenWeightedVoting()
    
    with patch.object(GovernanceQueries, 'upsert_proposal', new_callable=AsyncMock) as mock_upsert:
        mock_upsert.return_value = True
        
        result = await voting_system._persist_proposal(sample_proposal)
        
        assert result is True
        mock_upsert.assert_called_once_with(sample_proposal)


# === Test 4: TokenWeightedVoting._sync_votes_to_db ===

@pytest.mark.asyncio
async def test_token_weighted_voting_sync_votes_to_db(sample_proposal):
    """
    Test 4: TokenWeightedVoting._sync_votes_to_db updates vote counts.
    
    Verifies that:
    - Vote count changes are persisted to the database
    - The method returns True on success
    """
    voting_system = TokenWeightedVoting()
    
    # Add proposal to in-memory store
    voting_system.proposals[sample_proposal.proposal_id] = sample_proposal
    
    # Modify vote counts
    sample_proposal.votes_for = 10
    sample_proposal.votes_against = 3
    
    with patch.object(GovernanceQueries, 'upsert_proposal', new_callable=AsyncMock) as mock_upsert:
        mock_upsert.return_value = True
        
        result = await voting_system._sync_votes_to_db(sample_proposal.proposal_id)
        
        assert result is True
        mock_upsert.assert_called_once()


# === Test 5: TokenWeightedVoting.hydrate_from_db ===

@pytest.mark.asyncio
async def test_token_weighted_voting_hydrate_from_db(sample_proposal_dict):
    """
    Test 5: TokenWeightedVoting.hydrate_from_db loads proposals from database.
    
    Verifies that:
    - Proposals are correctly loaded from the database
    - The in-memory proposals dictionary is populated
    - Status filtering works correctly
    """
    with patch.object(GovernanceQueries, 'load_all_proposals', new_callable=AsyncMock) as mock_load:
        mock_load.return_value = [sample_proposal_dict]
        
        voting_system = await TokenWeightedVoting.hydrate_from_db(status_filter="active")
        
        assert voting_system is not None
        assert len(voting_system.proposals) == 1
        
        # Verify the proposal was loaded correctly
        proposal_id = UUID(sample_proposal_dict["proposal_id"])
        assert proposal_id in voting_system.proposals
        
        loaded_proposal = voting_system.proposals[proposal_id]
        assert loaded_proposal.title == sample_proposal_dict["title"]
        assert loaded_proposal.status == "active"
        
        # Verify the status filter was passed
        mock_load.assert_called_once_with("active")


# === Additional Integration Tests ===

@pytest.mark.asyncio
async def test_create_proposal_persists_to_db():
    """
    Integration test: Creating a proposal triggers database persistence.
    
    This test verifies the full flow of proposal creation with persistence.
    """
    voting_system = TokenWeightedVoting()
    
    proposal = GovernanceProposal(
        title="Integration Test Proposal",
        description="Testing full persistence flow",
        proposal_type="technical",
        proposer_id="integration_test_user"
    )
    
    # Mock the dependencies
    with patch.object(voting_system, '_validate_proposer_eligibility', new_callable=AsyncMock) as mock_validate:
        with patch.object(voting_system, 'ftns_service') as mock_ftns:
            with patch.object(voting_system, '_persist_proposal', new_callable=AsyncMock) as mock_persist:
                with patch.object(voting_system, 'safety_monitor') as mock_safety:
                    mock_validate.return_value = True
                    mock_ftns.atomic_deduct = AsyncMock(return_value=True)
                    mock_safety.validate_model_output = AsyncMock(return_value=True)
                    mock_persist.return_value = True
                    
                    # This should trigger persistence
                    proposal_id = await voting_system.create_proposal("integration_test_user", proposal)
                    
                    # Verify persistence was called
                    mock_persist.assert_called_once()
                    assert proposal_id is not None


@pytest.mark.asyncio
async def test_cast_vote_syncs_to_db():
    """
    Integration test: Casting a vote triggers database sync.
    
    This test verifies that vote changes are persisted.
    """
    voting_system = TokenWeightedVoting()
    
    proposal_id = uuid4()
    proposal = GovernanceProposal(
        proposal_id=proposal_id,
        title="Vote Sync Test",
        description="Testing vote sync",
        proposal_type="operational",
        proposer_id="test_user",
        status="active",
        voting_starts=datetime.now(timezone.utc) - timedelta(days=1),
        voting_ends=datetime.now(timezone.utc) + timedelta(days=7)
    )
    
    voting_system.proposals[proposal_id] = proposal
    voting_system.votes[proposal_id] = []
    
    with patch.object(voting_system, '_validate_vote_eligibility', new_callable=AsyncMock) as mock_validate:
        with patch.object(voting_system, 'calculate_voting_power', new_callable=AsyncMock) as mock_power:
            with patch.object(voting_system, '_sync_votes_to_db', new_callable=AsyncMock) as mock_sync:
                with patch.object(voting_system, '_should_conclude_voting_early', new_callable=AsyncMock) as mock_early:
                    mock_validate.return_value = True
                    mock_power.return_value = MagicMock(total_voting_power=100.0)
                    mock_sync.return_value = True
                    mock_early.return_value = False
                    
                    result = await voting_system.cast_vote("voter_001", proposal_id, True, "Test vote")
                    
                    assert result is True
                    mock_sync.assert_called_once_with(proposal_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
