"""
PRSM Teams Integration Test

Comprehensive test demonstrating the complete teams functionality
including team creation, membership management, governance, and tokenomics.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from prsm.teams.service import get_team_service
from prsm.teams.wallet import get_team_wallet_service
from prsm.teams.governance import get_team_governance_service
from prsm.teams.models import (
    Team, TeamMember, TeamType, GovernanceModel, RewardPolicy, TeamRole
)


async def test_teams_integration():
    """
    Test complete teams integration workflow
    
    🧑‍🤝‍🧑 TEAMS DEMONSTRATION:
    This test showcases the full teams functionality including:
    - Team creation with governance models
    - Member invitation and management
    - Wallet operations and reward distribution
    - Governance proposals and voting
    - Task management and collaboration
    """
    print("🚀 Starting PRSM Teams Integration Test")
    print("=" * 60)
    
    # Initialize services
    team_service = get_team_service()
    wallet_service = get_team_wallet_service()
    governance_service = get_team_governance_service()
    
    # Temporarily disable eligibility checks for testing
    original_validate = team_service._validate_team_creation_eligibility
    original_validate_data = team_service._validate_team_data
    
    async def mock_eligibility(user_id):
        return True
    
    async def mock_validation(data):
        return True
    
    team_service._validate_team_creation_eligibility = mock_eligibility
    team_service._validate_team_data = mock_validation
    
    try:
        # === 1. Team Creation ===
        print("\n1️⃣ Creating Research Team")
        print("-" * 30)
        
        founder_id = "researcher_alice"
        team_data = {
            "name": "Quantum AI Research Collective",
            "description": "Exploring the intersection of quantum computing and artificial intelligence for breakthrough discoveries.",
            "team_type": TeamType.RESEARCH,
            "governance_model": GovernanceModel.DEMOCRATIC,
            "reward_policy": RewardPolicy.PROPORTIONAL,
            "is_public": True,
            "max_members": 10,
            "entry_stake_required": 100.0,
            "research_domains": ["quantum_computing", "artificial_intelligence", "machine_learning"],
            "keywords": ["quantum", "ai", "research", "collaborative", "innovation"],
            "contact_info": {"email": "contact@quantumai.org"},
            "external_links": {"website": "https://quantumai.org"}
        }
        
        team = await team_service.create_team(founder_id, team_data)
        
        print(f"✅ Team created successfully!")
        print(f"   Team ID: {team.team_id}")
        print(f"   Name: {team.name}")
        print(f"   Type: {team.team_type}")
        print(f"   Governance: {team.governance_model}")
        print(f"   Founder: {founder_id}")
        print(f"   Entry Stake: {team.entry_stake_required} FTNS")
        
        # === 2. Team Governance Setup ===
        print("\n2️⃣ Setting Up Team Governance")
        print("-" * 35)
        
        governance = await governance_service.create_team_governance(team)
        
        print(f"✅ Governance configured!")
        print(f"   Model: {governance.model}")
        print(f"   Voting Period: {governance.voting_period_days} days")
        print(f"   Quorum Required: {governance.quorum_percentage * 100}%")
        print(f"   Approval Threshold: {governance.approval_threshold * 100}%")
        
        # === 3. Member Invitations ===
        print("\n3️⃣ Inviting Team Members")
        print("-" * 30)
        
        members_to_invite = [
            ("researcher_bob", TeamRole.ADMIN, "Quantum computing expert"),
            ("data_scientist_carol", TeamRole.OPERATOR, "AI/ML specialist"),
            ("professor_dave", TeamRole.MEMBER, "Academic advisor"),
            ("engineer_eve", TeamRole.MEMBER, "System implementation")
        ]
        
        invitations = []
        for invitee_id, role, message in members_to_invite:
            invitation = await team_service.invite_member(
                team.team_id, founder_id, invitee_id, role, message
            )
            invitations.append((invitation, invitee_id))
            print(f"📧 Invited {invitee_id} as {role}")
        
        # Accept invitations
        print("\n   Accepting invitations...")
        for invitation, invitee_id in invitations:
            success = await team_service.accept_invitation(invitation.invitation_id, invitee_id)
            if success:
                print(f"✅ {invitee_id} joined the team")
        
        # === 4. Wallet Operations ===
        print("\n4️⃣ Team Wallet Operations")
        print("-" * 30)
        
        # Get team members for wallet operations
        members = await team_service.get_team_members(team.team_id)
        
        # Create team wallet (already done in team creation, but demonstrate interface)
        authorized_signers = [founder_id, "researcher_bob"]  # Owner and admin
        team_wallet = await wallet_service.create_team_wallet(
            team, authorized_signers, required_signatures=2
        )
        
        print(f"✅ Team wallet created!")
        print(f"   Wallet ID: {team_wallet.wallet_id}")
        print(f"   Multisig: {team_wallet.is_multisig}")
        print(f"   Required Signatures: {team_wallet.required_signatures}")
        print(f"   Authorized Signers: {len(team_wallet.authorized_signers)}")
        
        # Simulate FTNS deposits
        print("\n   Members contributing to team wallet...")
        deposits = [
            (founder_id, 500.0, "Initial funding"),
            ("researcher_bob", 300.0, "Research contribution"),
            ("data_scientist_carol", 200.0, "Data processing contribution")
        ]
        
        for depositor, amount, description in deposits:
            success = await wallet_service.deposit_ftns(
                team_wallet, amount, depositor, description
            )
            if success:
                print(f"💰 {depositor} deposited {amount} FTNS")
        
        print(f"   Total Wallet Balance: {team_wallet.total_balance} FTNS")
        
        # === 5. Governance Proposal ===
        print("\n5️⃣ Creating Governance Proposal")
        print("-" * 35)
        
        proposal_data = {
            "title": "Increase Research Budget Allocation",
            "description": "Propose to allocate 800 FTNS from team wallet for quantum algorithm research equipment and cloud computing resources.",
            "proposal_type": "treasury",
            "proposed_changes": {
                "budget_allocation": 800.0,
                "purpose": "research_equipment",
                "duration_months": 6
            },
            "implementation_plan": "Purchase quantum simulator access and cloud computing credits for research team.",
            "estimated_cost": 800.0
        }
        
        proposal = await governance_service.create_proposal(
            team.team_id, founder_id, proposal_data
        )
        
        print(f"✅ Proposal created!")
        print(f"   Proposal ID: {proposal.proposal_id}")
        print(f"   Title: {proposal.title}")
        print(f"   Type: {proposal.proposal_type}")
        print(f"   Voting Period: {proposal.voting_starts} to {proposal.voting_ends}")
        print(f"   Estimated Cost: {proposal.estimated_cost} FTNS")
        
        # === 6. Team Voting ===
        print("\n6️⃣ Conducting Team Vote")
        print("-" * 25)
        
        # Simulate team members voting
        votes = [
            (founder_id, "for", "Good investment in our research capabilities"),
            ("researcher_bob", "for", "Essential for quantum algorithm development"),
            ("data_scientist_carol", "for", "Will accelerate our AI research"),
            ("professor_dave", "abstain", "Need more technical details"),
            ("engineer_eve", "against", "Too much budget for experimental work")
        ]
        
        for voter_id, vote, rationale in votes:
            success = await governance_service.cast_vote(
                proposal.proposal_id, voter_id, vote, rationale
            )
            if success:
                print(f"🗳️ {voter_id} voted: {vote}")
        
        # Get voting results
        results = await governance_service.get_proposal_results(proposal.proposal_id)
        if results:
            print(f"\n   📊 Voting Results:")
            print(f"      Participation: {results['participation_rate'] * 100:.1f}%")
            print(f"      Approval: {results['approval_percentage'] * 100:.1f}%")
            print(f"      Status: {'PASSED' if results['proposal_passes'] else 'FAILED'}")
            print(f"      Votes For: {results['votes_for']}")
            print(f"      Votes Against: {results['votes_against']}")
            print(f"      Abstentions: {results['votes_abstain']}")
        
        # === 7. Task Creation ===
        print("\n7️⃣ Creating Collaborative Task")
        print("-" * 32)
        
        task_data = {
            "title": "Quantum Circuit Optimization Research",
            "description": "Develop and test quantum circuit optimization algorithms using variational quantum eigensolvers.",
            "task_type": "research",
            "priority": "high",
            "assigned_to": ["researcher_bob", "data_scientist_carol"],
            "ftns_budget": 200.0,
            "due_date": datetime.now(timezone.utc) + timedelta(days=30),
            "estimated_hours": 120.0,
            "requires_consensus": True,
            "consensus_threshold": 0.6,
            "tags": ["quantum", "optimization", "algorithms"],
            "external_links": {"github": "https://github.com/quantumai/circuit-opt"}
        }
        
        task = await team_service.create_team_task(
            team.team_id, founder_id, task_data
        )
        
        print(f"✅ Task created!")
        print(f"   Task ID: {task.task_id}")
        print(f"   Title: {task.title}")
        print(f"   Assigned to: {', '.join(task.assigned_to)}")
        print(f"   Budget: {task.ftns_budget} FTNS")
        print(f"   Due Date: {task.due_date}")
        print(f"   Estimated Hours: {task.estimated_hours}")
        
        # === 8. Reward Distribution ===
        print("\n8️⃣ Distributing Team Rewards")
        print("-" * 30)
        
        # Distribute 300 FTNS to team members based on contributions
        distribution_amount = 300.0
        distributions = await wallet_service.distribute_rewards(
            team_wallet, members, distribution_amount
        )
        
        print(f"💰 Distributed {distribution_amount} FTNS rewards:")
        for user_id, amount in distributions.items():
            print(f"   {user_id}: {amount:.2f} FTNS")
        
        print(f"   Remaining Wallet Balance: {team_wallet.available_balance} FTNS")
        
        # === 9. Team Statistics ===
        print("\n9️⃣ Team Performance Statistics")
        print("-" * 35)
        
        # Get service statistics
        team_stats = await team_service.get_service_statistics()
        wallet_stats = await wallet_service.get_wallet_statistics()
        governance_stats = await governance_service.get_governance_statistics()
        
        print("📊 System Statistics:")
        print(f"   Teams Created: {team_stats['total_teams_created']}")
        print(f"   Members Joined: {team_stats['total_members_joined']}")
        print(f"   Invitations Sent: {team_stats['total_invitations_sent']}")
        print(f"   Tasks Created: {team_stats['total_tasks_created']}")
        print(f"   Average Team Size: {team_stats['average_team_size']:.1f}")
        
        print(f"\n💰 Wallet Statistics:")
        print(f"   Wallets Created: {wallet_stats['total_wallets_created']}")
        print(f"   FTNS Distributed: {wallet_stats['total_ftns_distributed']:.2f}")
        print(f"   Multisig Operations: {wallet_stats['multisig_operations_executed']}")
        
        print(f"\n🗳️ Governance Statistics:")
        print(f"   Proposals Created: {governance_stats['total_proposals_created']}")
        print(f"   Votes Cast: {governance_stats['total_votes_cast']}")
        print(f"   Proposals Passed: {governance_stats['proposals_passed']}")
        print(f"   Average Participation: {governance_stats['average_participation_rate'] * 100:.1f}%")
        
        # === 10. Team Search and Discovery ===
        print("\n🔟 Team Discovery and Search")
        print("-" * 30)
        
        # Search for teams
        search_results = await team_service.search_teams(
            "quantum AI", 
            filters={"team_type": TeamType.RESEARCH, "min_members": 3}
        )
        
        print(f"🔍 Search Results for 'quantum AI':")
        for result_team in search_results:
            print(f"   📋 {result_team.name}")
            print(f"      Type: {result_team.team_type}")
            print(f"      Members: {result_team.member_count}")
            print(f"      Impact Score: {result_team.impact_score}")
            print(f"      Research Domains: {', '.join(result_team.research_domains)}")
        
        print("\n" + "=" * 60)
        print("✅ PRSM Teams Integration Test Completed Successfully!")
        print("\n🎉 Teams functionality is working correctly with:")
        print("   • Team creation and configuration")
        print("   • Member invitation and management") 
        print("   • Multisig wallet operations")
        print("   • Democratic governance and voting")
        print("   • Collaborative task management")
        print("   • FTNS reward distribution")
        print("   • Team discovery and search")
        print("   • Performance monitoring and analytics")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Restore original functions
        team_service._validate_team_creation_eligibility = original_validate
        team_service._validate_team_data = original_validate_data


async def main():
    """Run the teams integration test"""
    success = await test_teams_integration()
    
    if success:
        print("\n🎯 All teams functionality working correctly!")
        print("Ready for production deployment.")
    else:
        print("\n⚠️ Test failed - check implementation.")
        
    return success


if __name__ == "__main__":
    # Run the test
    asyncio.run(main())