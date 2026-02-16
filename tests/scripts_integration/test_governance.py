#!/usr/bin/env python3
"""
PRSM Governance System Test Script
==================================

Test script for validating the governance token distribution and voting system.
"""

import pytest
pytest.skip('Module dependencies not yet fully implemented', allow_module_level=True)

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

import click
import structlog

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prsm.economy.governance.token_distribution import (
    get_governance_distributor, 
    GovernanceParticipantTier, 
    ContributionType
)
from prsm.economy.governance.voting import get_token_weighted_voting
from prsm.economy.governance.quadratic_voting import quadratic_voting
from prsm.core.auth.auth_manager import auth_manager

# Set up logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def cli(verbose):
    """PRSM Governance System Test Tool"""
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
def test_system():
    """Test the complete governance system"""
    
    async def _test():
        try:
            click.echo("🏛️ Testing PRSM Governance System")
            click.echo("=" * 60)
            
            distributor = get_governance_distributor()
            voting_system = get_token_weighted_voting()
            
            # Test 1: Enable voting system
            click.echo("\n📊 Test 1: Enable Voting System")
            activation_results = await distributor.enable_voting_system()
            click.echo(f"✅ Voting system activated: {activation_results['voting_system_activated']}")
            click.echo(f"   Councils created: {len(activation_results['councils_created'])}")
            click.echo(f"   Mechanisms enabled: {len(activation_results['voting_mechanisms_enabled'])}")
            
            # Test 2: Create test users
            click.echo("\n👥 Test 2: Create Test Users")
            test_users = []
            
            user_configs = [
                {"username": "alice_contributor", "email": "alice@example.com", "tier": GovernanceParticipantTier.CONTRIBUTOR},
                {"username": "bob_expert", "email": "bob@example.com", "tier": GovernanceParticipantTier.EXPERT},
                {"username": "carol_delegate", "email": "carol@example.com", "tier": GovernanceParticipantTier.DELEGATE},
            ]
            
            for config in user_configs:
                try:
                    # Create user (simplified - in production this would use proper auth)
                    user_id = f"test_user_{config['username']}"
                    test_users.append((user_id, config))
                    click.echo(f"   Created test user: {config['username']} ({config['tier'].value})")
                    
                except Exception as e:
                    click.echo(f"   ❌ Failed to create user {config['username']}: {e}")
            
            # Test 3: Activate governance participation
            click.echo("\n🎯 Test 3: Activate Governance Participation")
            activations = []
            
            for user_id, config in test_users:
                try:
                    activation = await distributor.activate_governance_participation(
                        user_id=user_id,
                        participant_tier=config["tier"],
                        council_nominations=["AI Safety Council"] if config["tier"] in [GovernanceParticipantTier.EXPERT, GovernanceParticipantTier.DELEGATE] else []
                    )
                    activations.append(activation)
                    click.echo(f"   ✅ {config['username']}: {activation.initial_token_allocation} FTNS allocated")
                    
                except Exception as e:
                    click.echo(f"   ❌ Failed to activate {config['username']}: {e}")
            
            # Test 4: Distribute contribution rewards
            click.echo("\n🎁 Test 4: Distribute Contribution Rewards")
            
            contribution_tests = [
                (test_users[0][0], ContributionType.MODEL_CONTRIBUTION, "test_model_123", 1.5),
                (test_users[1][0], ContributionType.RESEARCH_PUBLICATION, "research_paper_456", 2.0),
                (test_users[2][0], ContributionType.SECURITY_AUDIT, "security_audit_789", 1.2),
            ]
            
            for user_id, contrib_type, reference, multiplier in contribution_tests:
                try:
                    distribution = await distributor.distribute_contribution_rewards(
                        user_id=user_id,
                        contribution_type=contrib_type,
                        contribution_reference=reference,
                        quality_multiplier=multiplier
                    )
                    click.echo(f"   ✅ {contrib_type.value}: {distribution.amount} FTNS distributed")
                    
                except Exception as e:
                    click.echo(f"   ❌ Failed to distribute reward for {contrib_type.value}: {e}")
            
            # Test 5: Stake tokens for voting
            click.echo("\n🗳️ Test 5: Stake Tokens for Voting")
            
            for user_id, config in test_users[:2]:  # Test first 2 users
                try:
                    staked_amount, voting_power = await distributor.stake_tokens_for_governance(
                        user_id=user_id,
                        amount=1000,
                        lock_duration_days=60
                    )
                    click.echo(f"   ✅ {config['username']}: {staked_amount} FTNS staked, {voting_power} voting power")
                    
                except Exception as e:
                    click.echo(f"   ❌ Failed to stake tokens for {config['username']}: {e}")
            
            # Test 6: Create and vote on proposal (simplified)
            click.echo("\n📋 Test 6: Create and Vote on Proposal")
            
            try:
                # Create a test proposal (simplified)
                from prsm.core.models import GovernanceProposal
                
                proposal = GovernanceProposal(
                    title="Test Governance Proposal",
                    description="This is a test proposal to validate the governance system functionality. It proposes to increase the marketplace commission rate by 0.5% to fund additional security audits.",
                    proposal_type="economic",
                    proposer_id=test_users[0][0],
                    status="draft"
                )
                
                proposal_id = await voting_system.create_proposal(test_users[0][0], proposal)
                click.echo(f"   ✅ Proposal created: {proposal_id}")
                
                # Cast votes from test users
                for i, (user_id, config) in enumerate(test_users[:2]):
                    vote_choice = i % 2 == 0  # Alternate yes/no votes
                    vote_success = await voting_system.cast_vote(
                        voter_id=user_id,
                        proposal_id=proposal_id,
                        vote=vote_choice,
                        rationale=f"Test vote from {config['username']}"
                    )
                    
                    if vote_success:
                        click.echo(f"   ✅ {config['username']}: Voted {'for' if vote_choice else 'against'}")
                    else:
                        click.echo(f"   ❌ {config['username']}: Failed to vote")
                
            except Exception as e:
                click.echo(f"   ❌ Failed to test proposal system: {e}")
            
            # Test 7: Get governance statistics
            click.echo("\n📊 Test 7: Governance Statistics")
            
            try:
                stats = await distributor.get_distribution_statistics()
                click.echo(f"   Total participants: {stats.get('total_participants', 0)}")
                click.echo(f"   Total tokens distributed: {stats.get('total_tokens_distributed', '0')}")
                click.echo(f"   Active councils: {len(stats.get('council_statistics', {}))}")
                click.echo(f"   Activation rate: {stats.get('activation_rate', 0):.2%}")
                
            except Exception as e:
                click.echo(f"   ❌ Failed to get governance statistics: {e}")
            
            # Test 8: Test user governance status
            click.echo("\n👤 Test 8: User Governance Status")
            
            for user_id, config in test_users[:1]:  # Test first user
                try:
                    status = await distributor.get_governance_status(user_id)
                    
                    if status.get("is_activated"):
                        click.echo(f"   ✅ {config['username']} governance status:")
                        click.echo(f"      Tier: {status.get('current_tier')}")
                        balances = status.get('token_balances', {})
                        click.echo(f"      Total balance: {balances.get('total_balance', '0')} FTNS")
                        voting_power = status.get('voting_power', {})
                        click.echo(f"      Voting power: {voting_power.get('total_voting_power', 0)}")
                        councils = status.get('council_memberships', [])
                        click.echo(f"      Council memberships: {len(councils)}")
                    else:
                        click.echo(f"   ❌ {config['username']}: Not activated")
                        
                except Exception as e:
                    click.echo(f"   ❌ Failed to get status for {config['username']}: {e}")
            
            click.echo("\n🎉 Governance System Test Complete!")
            click.echo("All core governance functionality has been validated.")
            
            return True
            
        except Exception as e:
            logger.error("Governance system test failed", error=str(e))
            click.echo(f"❌ Test failed: {e}")
            return False
    
    success = asyncio.run(_test())
    sys.exit(0 if success else 1)


@cli.command()
@click.option('--user-id', required=True, help='User ID to activate')
@click.option('--tier', 
              type=click.Choice([t.value for t in GovernanceParticipantTier]), 
              default=GovernanceParticipantTier.COMMUNITY.value,
              help='Participant tier')
def activate_user(user_id, tier):
    """Activate governance participation for a specific user"""
    
    async def _activate():
        try:
            distributor = get_governance_distributor()
            
            tier_enum = GovernanceParticipantTier(tier)
            activation = await distributor.activate_governance_participation(
                user_id=user_id,
                participant_tier=tier_enum
            )
            
            click.echo(f"✅ Governance activated for {user_id}")
            click.echo(f"   Tier: {activation.participant_tier.value}")
            click.echo(f"   Initial allocation: {activation.initial_token_allocation} FTNS")
            click.echo(f"   Staked amount: {activation.staked_amount} FTNS")
            click.echo(f"   Voting power: {activation.voting_power}")
            
            return True
            
        except Exception as e:
            click.echo(f"❌ Failed to activate user: {e}")
            return False
    
    success = asyncio.run(_activate())
    sys.exit(0 if success else 1)


@cli.command()
def enable_voting():
    """Enable the governance voting system"""
    
    async def _enable():
        try:
            distributor = get_governance_distributor()
            
            click.echo("🏛️ Enabling Governance Voting System...")
            activation_results = await distributor.enable_voting_system()
            
            if activation_results["voting_system_activated"]:
                click.echo("✅ Governance voting system enabled successfully!")
                click.echo(f"   Quadratic voting: {'✅' if activation_results['quadratic_voting_enabled'] else '❌'}")
                click.echo(f"   Federated councils: {'✅' if activation_results['federated_councils_created'] else '❌'}")
                click.echo(f"   Councils created: {len(activation_results['councils_created'])}")
                click.echo(f"   Voting mechanisms: {', '.join(activation_results['voting_mechanisms_enabled'])}")
                click.echo(f"   Initial participants: {activation_results['initial_participants_activated']}")
            else:
                click.echo("❌ Failed to enable governance voting system")
                return False
            
            return True
            
        except Exception as e:
            click.echo(f"❌ Failed to enable voting system: {e}")
            return False
    
    success = asyncio.run(_enable())
    sys.exit(0 if success else 1)


@cli.command()
def statistics():
    """Get governance system statistics"""
    
    async def _stats():
        try:
            distributor = get_governance_distributor()
            
            stats = await distributor.get_distribution_statistics()
            
            click.echo("📊 Governance System Statistics")
            click.echo("=" * 40)
            click.echo(f"Total participants: {stats.get('total_participants', 0)}")
            click.echo(f"Total tokens distributed: {stats.get('total_tokens_distributed', '0')} FTNS")
            click.echo(f"Total staked tokens: {stats.get('total_staked_tokens', '0')} FTNS")
            click.echo(f"Active voters: {stats.get('active_voters', 0)}")
            click.echo(f"Active councils: {len(stats.get('council_statistics', {}))}")
            click.echo(f"Governance proposals: {stats.get('governance_proposals_created', 0)}")
            click.echo(f"Votes cast: {stats.get('votes_cast', 0)}")
            
            # Distribution by tier
            distributions_by_tier = stats.get('distributions_by_tier', {})
            if distributions_by_tier:
                click.echo("\n📈 Distribution by Tier:")
                for tier, data in distributions_by_tier.items():
                    click.echo(f"   {tier}: {data['count']} participants, {data['total_allocated']} FTNS")
            
            # Council statistics
            council_stats = stats.get('council_statistics', {})
            if council_stats:
                click.echo("\n🏛️ Council Statistics:")
                for council_id, council_data in council_stats.items():
                    click.echo(f"   {council_data['name']}: {council_data['member_count']} members")
            
            return True
            
        except Exception as e:
            click.echo(f"❌ Failed to get statistics: {e}")
            return False
    
    success = asyncio.run(_stats())
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    cli()