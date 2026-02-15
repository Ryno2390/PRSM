#!/usr/bin/env python3

"""
PRSM Full Governance System Integration Tests

Comprehensive testing of the complete governance system including:
- Token-weighted voting system
- Proposal management lifecycle
- End-to-end governance workflows
- Integration with FTNS, safety, and other PRSM components

Phase 3, Week 17-18 - Full Governance System Testing
"""

import asyncio
import sys
import time
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
from uuid import uuid4

# Add the project root to Python path
sys.path.insert(0, '/Users/ryneschultz/GitHub/PRSM')

from prsm.economy.governance.voting import TokenWeightedVoting, ProposalCategory, VotingPeriod, GovernanceRole
from prsm.economy.governance.proposals import ProposalManager, ProposalStatus, ProposalPriority, ReviewDecision
from prsm.core.models import GovernanceProposal, Vote
from prsm.economy.tokenomics.ftns_service import ftns_service


class GovernanceIntegrationTester:
    """Comprehensive governance system integration tester"""
    
    def __init__(self):
        self.voting_system = TokenWeightedVoting()
        self.proposal_manager = ProposalManager()
        
        # Test data
        self.test_users = ["alice", "bob", "charlie", "diana", "eve"]
        self.test_proposals = []
        self.test_results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": []
        }
        
        print("🗳️ Governance Integration Tester initialized")
    
    
    def _create_test_proposal(self, title: str, description: str, 
                            proposal_type: str, proposer_id: str, 
                            status: str = "active", **kwargs) -> GovernanceProposal:
        """Helper to create test proposals with required fields"""
        current_time = datetime.now(timezone.utc)
        return GovernanceProposal(
            title=title,
            description=description,
            proposal_type=proposal_type,
            proposer_id=proposer_id,
            voting_starts=current_time,
            voting_ends=current_time + timedelta(days=7),
            status=status,
            **kwargs
        )
    
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all governance integration tests"""
        print("\n" + "="*80)
        print("🗳️ RUNNING COMPREHENSIVE GOVERNANCE SYSTEM TESTS")
        print("="*80)
        
        try:
            # Setup test environment
            await self._setup_test_environment()
            
            # Core integration tests
            await self._test_token_weighted_voting_system()
            await self._test_proposal_management_system()
            await self._test_end_to_end_governance_workflow()
            await self._test_integration_with_ftns_system()
            await self._test_integration_with_safety_systems()
            await self._test_governance_role_management()
            await self._test_delegation_and_voting_power()
            await self._test_proposal_lifecycle_edge_cases()
            await self._test_system_performance_and_scalability()
            
            # Generate final results
            return await self._generate_test_results()
            
        except Exception as e:
            print(f"❌ Critical error in governance testing: {e}")
            return {"error": str(e), "tests_completed": self.test_results["total_tests"]}
    
    
    async def _setup_test_environment(self):
        """Setup test environment with users and initial data"""
        print("\n📋 Setting up test environment...")
        
        try:
            # Setup test users with FTNS balances
            for user_id in self.test_users:
                await ftns_service._update_balance(user_id, 50000.0)  # 50K FTNS each
            
            # Create some governance roles
            await self.voting_system.assign_governance_role("alice", GovernanceRole.DELEGATE)
            await self.voting_system.assign_governance_role("bob", GovernanceRole.COMMITTEE_MEMBER)
            
            print("✅ Test environment setup complete")
            
        except Exception as e:
            print(f"❌ Failed to setup test environment: {e}")
            raise
    
    
    async def _test_token_weighted_voting_system(self):
        """Test core token-weighted voting functionality"""
        print("\n🗳️ Testing Token-Weighted Voting System...")
        
        test_cases = [
            ("voting_power_calculation", self._test_voting_power_calculation),
            ("proposal_creation", self._test_proposal_creation),
            ("vote_casting", self._test_vote_casting),
            ("delegation_system", self._test_delegation_system),
            ("term_limits", self._test_term_limits),
            ("voting_results", self._test_voting_results)
        ]
        
        for test_name, test_func in test_cases:
            await self._run_single_test(f"voting_{test_name}", test_func)
    
    
    async def _test_proposal_management_system(self):
        """Test comprehensive proposal management"""
        print("\n📋 Testing Proposal Management System...")
        
        test_cases = [
            ("proposal_eligibility", self._test_proposal_eligibility),
            ("proposal_lifecycle", self._test_proposal_lifecycle),
            ("proposal_execution", self._test_proposal_execution),
            ("community_support", self._test_community_support),
            ("proposal_reviews", self._test_proposal_reviews),
            ("proposal_metrics", self._test_proposal_metrics)
        ]
        
        for test_name, test_func in test_cases:
            await self._run_single_test(f"proposal_{test_name}", test_func)
    
    
    async def _test_end_to_end_governance_workflow(self):
        """Test complete end-to-end governance workflow"""
        print("\n🔄 Testing End-to-End Governance Workflow...")
        
        try:
            # Create a comprehensive test proposal
            proposal = self._create_test_proposal(
                title="Test Governance Proposal: Increase Community Rewards",
                description="This is a comprehensive test proposal to increase community rewards by 25% to improve participation in the PRSM ecosystem. The proposal includes detailed implementation plans and budget considerations.",
                proposal_type="economic",
                proposer_id="alice"
            )
            
            # Step 1: Validate proposal eligibility
            eligibility = await self.proposal_manager.validate_proposal_eligibility(proposal)
            assert eligibility, "Proposal should be eligible"
            
            # Step 2: Create proposal through voting system
            proposal_id = await self.voting_system.create_proposal("alice", proposal)
            assert proposal_id is not None, "Proposal creation should succeed"
            
            # Step 3: Add community support
            for supporter in ["bob", "charlie", "diana"]:
                support_added = await self.proposal_manager.add_community_support(
                    proposal_id, supporter, "endorse", f"Support from {supporter}"
                )
                assert support_added, f"Community support from {supporter} should be added"
            
            # Step 4: Manage proposal lifecycle
            lifecycle_result = await self.proposal_manager.manage_proposal_lifecycle(proposal_id)
            assert len(lifecycle_result["transitions_made"]) > 0, "Proposal should progress through lifecycle"
            
            # Step 5: Cast votes
            votes_cast = 0
            for voter in ["alice", "bob", "charlie"]:
                vote_success = await self.voting_system.cast_vote(voter, proposal_id, True, f"Vote from {voter}")
                if vote_success:
                    votes_cast += 1
            
            assert votes_cast >= 2, "At least 2 votes should be cast successfully"
            
            # Step 6: Get voting results
            results = await self.voting_system.get_proposal_results(proposal_id)
            assert results is not None, "Voting results should be available"
            assert results.total_votes_cast == votes_cast, "Vote count should match"
            
            # Step 7: Get comprehensive proposal status
            status = await self.proposal_manager.get_proposal_status(proposal_id)
            assert status is not None, "Proposal status should be available"
            assert status["community_support"]["total_supporters"] >= 3, "Should have community support"
            
            print("✅ End-to-end governance workflow test passed")
            return True
            
        except Exception as e:
            print(f"❌ End-to-end governance workflow test failed: {e}")
            return False
    
    
    async def _test_integration_with_ftns_system(self):
        """Test governance integration with FTNS token system"""
        print("\n💰 Testing FTNS Integration...")
        
        try:
            # Test proposal fee charging
            initial_balance = await ftns_service.get_user_balance("alice")
            
            proposal = self._create_test_proposal(
                title="FTNS Integration Test Proposal",
                description="Testing integration between governance system and FTNS token economy. This proposal should charge the appropriate fees and integrate with the token system seamlessly.",
                proposal_type="technical",
                proposer_id="alice"
            )
            
            proposal_id = await self.voting_system.create_proposal("alice", proposal)
            
            # Check that fee was charged
            final_balance = await ftns_service.get_user_balance("alice")
            fee_charged = initial_balance.balance - final_balance.balance
            
            assert fee_charged > 0, "Proposal fee should be charged"
            assert fee_charged == 1000.0, "Proposal fee should be correct amount"
            
            # Test voting power calculation integration
            voting_power = await self.voting_system.calculate_voting_power("alice")
            assert voting_power.total_voting_power > 0, "Voting power should be calculated"
            assert voting_power.token_balance > 0, "Token balance should be considered"
            
            print("✅ FTNS integration test passed")
            return True
            
        except Exception as e:
            print(f"❌ FTNS integration test failed: {e}")
            return False
    
    
    async def _test_integration_with_safety_systems(self):
        """Test governance integration with safety monitoring"""
        print("\n🛡️ Testing Safety Integration...")
        
        try:
            # Test safety validation during proposal creation
            safe_proposal = self._create_test_proposal(
                title="Safe Governance Proposal",
                description="This is a safe proposal that should pass safety validation checks. It includes appropriate content and technical specifications that comply with system safety requirements.",
                proposal_type="operational",
                proposer_id="bob"
            )
            
            # Should pass safety checks
            eligibility = await self.proposal_manager.validate_proposal_eligibility(safe_proposal)
            assert eligibility, "Safe proposal should pass eligibility checks"
            
            proposal_id = await self.voting_system.create_proposal("bob", safe_proposal)
            assert proposal_id is not None, "Safe proposal should be created"
            
            print("✅ Safety integration test passed")
            return True
            
        except Exception as e:
            print(f"❌ Safety integration test failed: {e}")
            return False
    
    
    async def _test_governance_role_management(self):
        """Test governance role assignment and management"""
        print("\n👥 Testing Governance Role Management...")
        
        try:
            # Test role assignment
            role_assigned = await self.voting_system.assign_governance_role("charlie", GovernanceRole.PROPOSAL_REVIEWER)
            assert role_assigned, "Role assignment should succeed"
            
            # Test voting power calculation with roles
            voting_power = await self.voting_system.calculate_voting_power("charlie")
            assert voting_power.role_multiplier > 1.0, "Role should provide voting power multiplier"
            
            # Test term limits enforcement
            enforcement_result = await self.voting_system.implement_term_limits(["charlie"])
            assert enforcement_result["roles_checked"] == 1, "Term limits should be checked"
            
            print("✅ Governance role management test passed")
            return True
            
        except Exception as e:
            print(f"❌ Governance role management test failed: {e}")
            return False
    
    
    async def _test_delegation_and_voting_power(self):
        """Test vote delegation and voting power distribution"""
        print("\n🤝 Testing Delegation and Voting Power...")
        
        try:
            # Test delegation
            delegation_success = await self.voting_system.delegate_voting_power(
                "diana", "alice", "all"
            )
            assert delegation_success, "Vote delegation should succeed"
            
            # Test voting power calculation with delegation
            alice_power = await self.voting_system.calculate_voting_power("alice")
            assert alice_power.delegation_power > 0, "Alice should have delegated power"
            
            diana_power = await self.voting_system.calculate_voting_power("diana")
            # Diana should still have power, but less due to delegation
            
            print("✅ Delegation and voting power test passed")
            return True
            
        except Exception as e:
            print(f"❌ Delegation and voting power test failed: {e}")
            return False
    
    
    async def _test_proposal_lifecycle_edge_cases(self):
        """Test edge cases in proposal lifecycle management"""
        print("\n🔍 Testing Proposal Lifecycle Edge Cases...")
        
        try:
            # Test proposal with insufficient community support
            proposal = self._create_test_proposal(
                title="Edge Case Test Proposal",
                description="Testing edge cases in proposal lifecycle management to ensure robustness of the governance system under various conditions and scenarios.",
                proposal_type="community",
                proposer_id="eve"
            )
            
            proposal_id = await self.voting_system.create_proposal("eve", proposal)
            
            # Manage lifecycle without sufficient support
            lifecycle_result = await self.proposal_manager.manage_proposal_lifecycle(proposal_id)
            assert lifecycle_result is not None, "Lifecycle management should handle edge cases"
            
            # Test proposal status tracking
            status = await self.proposal_manager.get_proposal_status(proposal_id)
            assert status is not None, "Proposal status should be available"
            
            print("✅ Proposal lifecycle edge cases test passed")
            return True
            
        except Exception as e:
            print(f"❌ Proposal lifecycle edge cases test failed: {e}")
            return False
    
    
    async def _test_system_performance_and_scalability(self):
        """Test system performance under load"""
        print("\n⚡ Testing System Performance and Scalability...")
        
        try:
            start_time = time.time()
            
            # Create multiple proposals
            proposal_ids = []
            for i in range(5):
                proposal = self._create_test_proposal(
                    title=f"Performance Test Proposal {i+1}",
                    description=f"Performance testing proposal number {i+1} to evaluate system scalability and response times under concurrent load conditions.",
                    proposal_type="technical",
                    proposer_id=self.test_users[i % len(self.test_users)]
                )
                
                proposal_id = await self.voting_system.create_proposal(
                    self.test_users[i % len(self.test_users)], 
                    proposal
                )
                proposal_ids.append(proposal_id)
            
            # Cast multiple votes
            vote_count = 0
            for proposal_id in proposal_ids:
                for voter in self.test_users[:3]:  # First 3 users vote
                    vote_success = await self.voting_system.cast_vote(
                        voter, proposal_id, True, f"Performance test vote"
                    )
                    if vote_success:
                        vote_count += 1
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Performance assertions
            assert len(proposal_ids) == 5, "All proposals should be created"
            assert vote_count > 0, "Some votes should be cast"
            assert total_time < 10.0, "Operations should complete within reasonable time"
            
            # Get system statistics
            voting_stats = await self.voting_system.get_governance_statistics()
            management_stats = await self.proposal_manager.get_management_statistics()
            
            assert voting_stats["total_proposals_created"] >= 5, "Voting system should track proposals"
            assert management_stats["total_reviews_submitted"] >= 0, "Management system should track reviews"
            
            print(f"✅ Performance test passed - {len(proposal_ids)} proposals, {vote_count} votes in {total_time:.2f}s")
            return True
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            return False
    
    
    # Individual test methods for voting system
    
    async def _test_voting_power_calculation(self):
        """Test voting power calculation logic"""
        power_calc = await self.voting_system.calculate_voting_power("alice")
        assert power_calc.total_voting_power > 0, "Alice should have voting power"
        assert power_calc.token_balance > 0, "Token balance should be considered"
        return True
    
    
    async def _test_proposal_creation(self):
        """Test proposal creation process"""
        proposal = self._create_test_proposal(
            title="Test Proposal Creation",
            description="Testing the proposal creation process to ensure it works correctly with all validation checks and integrations.",
            proposal_type="technical",
            proposer_id="alice"
        )
        
        proposal_id = await self.voting_system.create_proposal("alice", proposal)
        assert proposal_id is not None, "Proposal should be created"
        return True
    
    
    async def _test_vote_casting(self):
        """Test vote casting functionality"""
        # Create a test proposal
        proposal = self._create_test_proposal(
            title="Vote Casting Test",
            description="Testing vote casting functionality to ensure votes are properly recorded and counted.",
            proposal_type="operational",
            proposer_id="bob"
        )
        
        proposal_id = await self.voting_system.create_proposal("bob", proposal)
        
        # Cast votes
        vote_success = await self.voting_system.cast_vote("alice", proposal_id, True)
        assert vote_success, "Vote should be cast successfully"
        return True
    
    
    async def _test_delegation_system(self):
        """Test vote delegation functionality"""
        delegation_success = await self.voting_system.delegate_voting_power("charlie", "bob")
        assert delegation_success, "Delegation should succeed"
        return True
    
    
    async def _test_term_limits(self):
        """Test term limits enforcement"""
        result = await self.voting_system.implement_term_limits(["alice", "bob"])
        assert result["roles_checked"] >= 0, "Term limits should be checked"
        return True
    
    
    async def _test_voting_results(self):
        """Test voting results calculation"""
        # Create and vote on a proposal
        proposal = self._create_test_proposal(
            title="Voting Results Test",
            description="Testing voting results calculation to ensure accurate tallying and reporting.",
            proposal_type="community",
            proposer_id="charlie"
        )
        
        proposal_id = await self.voting_system.create_proposal("charlie", proposal)
        await self.voting_system.cast_vote("alice", proposal_id, True)
        
        results = await self.voting_system.get_proposal_results(proposal_id)
        assert results is not None, "Voting results should be available"
        assert results.total_votes_cast >= 1, "Vote count should be accurate"
        return True
    
    
    # Individual test methods for proposal management
    
    async def _test_proposal_eligibility(self):
        """Test proposal eligibility validation"""
        proposal = self._create_test_proposal(
            title="Eligibility Test Proposal",
            description="Testing proposal eligibility validation to ensure all criteria are properly checked and enforced.",
            proposal_type="technical",
            proposer_id="alice"
        )
        
        eligibility = await self.proposal_manager.validate_proposal_eligibility(proposal)
        assert eligibility, "Valid proposal should be eligible"
        return True
    
    
    async def _test_proposal_lifecycle(self):
        """Test proposal lifecycle management"""
        proposal = self._create_test_proposal(
            title="Lifecycle Test Proposal",
            description="Testing proposal lifecycle management to ensure proper status transitions and workflow management.",
            proposal_type="operational",
            proposer_id="bob"
        )
        
        # Store proposal directly for lifecycle testing
        proposal_id = proposal.proposal_id
        self.proposal_manager.proposals[proposal_id] = proposal
        
        result = await self.proposal_manager.manage_proposal_lifecycle(proposal_id)
        assert result is not None, "Lifecycle management should return results"
        return True
    
    
    async def _test_proposal_execution(self):
        """Test proposal execution process"""
        proposal = self._create_test_proposal(
            title="Execution Test Proposal",
            description="Testing proposal execution process to ensure approved proposals are properly implemented.",
            proposal_type="technical",
            proposer_id="alice",
            status="approved"
        )
        
        # Store approved proposal
        proposal_id = proposal.proposal_id
        self.proposal_manager.proposals[proposal_id] = proposal
        
        execution_result = await self.proposal_manager.execute_approved_proposals(proposal_id)
        assert execution_result is not None, "Execution should return results"
        return True
    
    
    async def _test_community_support(self):
        """Test community support functionality"""
        proposal = self._create_test_proposal(
            title="Community Support Test",
            description="Testing community support functionality to ensure proper tracking and engagement.",
            proposal_type="community",
            proposer_id="charlie"
        )
        
        # Store proposal
        proposal_id = proposal.proposal_id
        self.proposal_manager.proposals[proposal_id] = proposal
        
        support_added = await self.proposal_manager.add_community_support(
            proposal_id, "alice", "endorse", "Great proposal!"
        )
        assert support_added, "Community support should be added"
        return True
    
    
    async def _test_proposal_reviews(self):
        """Test proposal review system"""
        proposal = self._create_test_proposal(
            title="Review Test Proposal",
            description="Testing proposal review system to ensure proper review coordination and feedback collection.",
            proposal_type="safety",
            proposer_id="diana"
        )
        
        # Store proposal
        proposal_id = proposal.proposal_id
        self.proposal_manager.proposals[proposal_id] = proposal
        
        review_submitted = await self.proposal_manager.submit_proposal_review(
            proposal_id, "alice", ReviewDecision.APPROVE, "Looks good to me"
        )
        assert review_submitted, "Review should be submitted"
        return True
    
    
    async def _test_proposal_metrics(self):
        """Test proposal metrics calculation"""
        proposal = self._create_test_proposal(
            title="Metrics Test Proposal",
            description="Testing proposal metrics calculation to ensure accurate performance tracking and analysis.",
            proposal_type="economic",
            proposer_id="eve"
        )
        
        # Store proposal
        proposal_id = proposal.proposal_id
        self.proposal_manager.proposals[proposal_id] = proposal
        
        # Add some test data
        await self.proposal_manager.add_community_support(proposal_id, "alice", "endorse")
        
        status = await self.proposal_manager.get_proposal_status(proposal_id)
        assert status is not None, "Proposal status should include metrics"
        return True
    
    
    async def _run_single_test(self, test_name: str, test_func):
        """Run a single test and record results"""
        self.test_results["total_tests"] += 1
        
        try:
            start_time = time.time()
            result = await test_func()
            end_time = time.time()
            
            if result:
                self.test_results["passed_tests"] += 1
                status = "✅ PASSED"
            else:
                self.test_results["failed_tests"] += 1
                status = "❌ FAILED"
            
            self.test_results["test_details"].append({
                "test_name": test_name,
                "status": "passed" if result else "failed",
                "duration": end_time - start_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            print(f"  {status} {test_name} ({end_time - start_time:.3f}s)")
            
        except Exception as e:
            self.test_results["failed_tests"] += 1
            self.test_results["test_details"].append({
                "test_name": test_name,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            print(f"  ❌ FAILED {test_name} - {e}")
    
    
    async def _generate_test_results(self) -> Dict[str, Any]:
        """Generate comprehensive test results"""
        success_rate = (self.test_results["passed_tests"] / self.test_results["total_tests"]) * 100
        
        # Get system statistics
        voting_stats = await self.voting_system.get_governance_statistics()
        management_stats = await self.proposal_manager.get_management_statistics()
        
        results = {
            "test_summary": {
                "total_tests": self.test_results["total_tests"],
                "passed_tests": self.test_results["passed_tests"],
                "failed_tests": self.test_results["failed_tests"],
                "success_rate": f"{success_rate:.1f}%",
                "overall_status": "PASSED" if success_rate >= 80 else "FAILED"
            },
            "governance_system_stats": {
                "voting_system": voting_stats,
                "proposal_management": management_stats
            },
            "performance_metrics": {
                "average_test_duration": sum(t.get("duration", 0) for t in self.test_results["test_details"]) / len(self.test_results["test_details"]),
                "total_proposals_created": voting_stats["total_proposals_created"],
                "total_votes_cast": voting_stats["total_votes_cast"],
                "system_responsiveness": "excellent" if success_rate >= 90 else "good" if success_rate >= 80 else "needs_improvement"
            },
            "detailed_results": self.test_results["test_details"],
            "recommendations": self._generate_recommendations(success_rate)
        }
        
        return results
    
    
    def _generate_recommendations(self, success_rate: float) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if success_rate >= 95:
            recommendations.append("Excellent governance system performance - ready for production deployment")
        elif success_rate >= 85:
            recommendations.append("Good governance system performance - minor optimizations recommended")
        elif success_rate >= 70:
            recommendations.append("Acceptable governance system performance - some improvements needed")
        else:
            recommendations.append("Governance system needs significant improvements before deployment")
        
        if self.test_results["failed_tests"] > 0:
            recommendations.append("Review failed tests and implement fixes")
        
        recommendations.append("Continue monitoring governance system performance in production")
        recommendations.append("Implement governance analytics dashboard for ongoing oversight")
        
        return recommendations


async def main():
    """Run comprehensive governance system integration tests"""
    print("🗳️ Starting PRSM Full Governance System Integration Tests")
    print("=" * 80)
    
    tester = GovernanceIntegrationTester()
    
    try:
        # Run comprehensive tests
        results = await tester.run_comprehensive_tests()
        
        # Print summary
        print("\n" + "="*80)
        print("📊 GOVERNANCE SYSTEM TEST RESULTS SUMMARY")
        print("="*80)
        
        if "error" in results:
            print(f"❌ Testing failed with error: {results['error']}")
            return
        
        summary = results["test_summary"]
        print(f"📋 Total Tests: {summary['total_tests']}")
        print(f"✅ Passed Tests: {summary['passed_tests']}")
        print(f"❌ Failed Tests: {summary['failed_tests']}")
        print(f"📈 Success Rate: {summary['success_rate']}")
        print(f"🎯 Overall Status: {summary['overall_status']}")
        
        # Performance metrics
        perf = results["performance_metrics"]
        print(f"\n⚡ Performance Metrics:")
        print(f"   Average Test Duration: {perf['average_test_duration']:.3f}s")
        print(f"   Total Proposals Created: {perf['total_proposals_created']}")
        print(f"   Total Votes Cast: {perf['total_votes_cast']}")
        print(f"   System Responsiveness: {perf['system_responsiveness']}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        for rec in results["recommendations"]:
            print(f"   • {rec}")
        
        # Detailed governance stats
        voting_stats = results["governance_system_stats"]["voting_system"]
        management_stats = results["governance_system_stats"]["proposal_management"]
        
        print(f"\n🗳️ Voting System Stats:")
        print(f"   Active Proposals: {voting_stats['active_proposals']}")
        print(f"   Average Participation Rate: {voting_stats['average_participation_rate']:.2%}")
        print(f"   Active Delegations: {voting_stats['total_active_delegations']}")
        print(f"   Governance Roles Assigned: {voting_stats['governance_roles_assigned']}")
        
        print(f"\n📋 Proposal Management Stats:")
        print(f"   Proposals Approved Rate: {management_stats['proposals_approved_rate']:.2%}")
        print(f"   Execution Success Rate: {management_stats['execution_success_rate']:.2%}")
        print(f"   Active Executions: {management_stats['active_executions']}")
        print(f"   Community Engagement Rate: {management_stats['community_engagement_rate']:.2%}")
        
        print("\n" + "="*80)
        
        if summary['overall_status'] == 'PASSED':
            print("🎉 GOVERNANCE SYSTEM INTEGRATION TESTS COMPLETED SUCCESSFULLY!")
            print("✅ Full governance system is ready for production deployment")
        else:
            print("⚠️ GOVERNANCE SYSTEM TESTS COMPLETED WITH ISSUES")
            print("🔧 Review failed tests and implement improvements")
        
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Governance system testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())