#!/usr/bin/env python3
"""
PRSM Governance Portal
=====================

Simple governance portal for community participation in PRSM's democratic decision-making.
Allows proposal submission, voting, and governance transparency.

Completes Gemini recommendation for "Public Testnet and Governance Portal".
"""

import asyncio
import json
import logging
import uuid
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum

try:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = Request = HTTPException = None
    HTMLResponse = JSONResponse = BaseModel = None
    uvicorn = None

logger = logging.getLogger(__name__)


class ProposalType(Enum):
    """Types of governance proposals"""
    TECHNICAL = "technical"
    ECONOMIC = "economic"
    GOVERNANCE = "governance"
    COMMUNITY = "community"


class ProposalStatus(Enum):
    """Status of governance proposals"""
    DRAFT = "draft"
    ACTIVE = "active"
    PASSED = "passed"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"


class VoteChoice(Enum):
    """Vote choices"""
    YES = "yes"
    NO = "no"
    ABSTAIN = "abstain"


class ProposalSubmission(BaseModel):
    """New proposal submission model"""
    title: str
    description: str
    proposal_type: str
    proposer_id: str
    implementation_details: Optional[str] = ""


class VoteSubmission(BaseModel):
    """Vote submission model"""
    proposal_id: str
    voter_id: str
    choice: str
    ftns_weight: float = 1.0


@dataclass
class Proposal:
    """Governance proposal"""
    id: str
    title: str
    description: str
    proposal_type: ProposalType
    proposer_id: str
    created_at: datetime
    voting_deadline: datetime
    status: ProposalStatus
    implementation_details: str
    
    # Voting results
    yes_votes: int = 0
    no_votes: int = 0
    abstain_votes: int = 0
    yes_ftns: float = 0.0
    no_ftns: float = 0.0
    abstain_ftns: float = 0.0
    total_voters: int = 0
    
    # Additional metadata
    discussion_url: Optional[str] = None
    implementation_status: str = "pending"


@dataclass
class Vote:
    """Individual vote record"""
    id: str
    proposal_id: str
    voter_id: str
    choice: VoteChoice
    ftns_weight: float
    timestamp: datetime
    voter_reputation: float = 1.0


@dataclass
class GovernanceStats:
    """Governance system statistics"""
    total_proposals: int
    active_proposals: int
    passed_proposals: int
    total_votes: int
    total_voters: int
    total_ftns_voted: float
    participation_rate: float
    avg_proposal_duration_days: float


@dataclass
class GovernanceConfig:
    """Governance portal configuration"""
    host: str = "0.0.0.0"
    port: int = 8095
    title: str = "PRSM Governance Portal"
    voting_period_days: int = 7
    quorum_threshold: float = 0.1  # 10% of FTNS must participate
    passing_threshold: float = 0.6  # 60% yes votes to pass


class PRSMGovernancePortal:
    """PRSM governance portal for community participation"""
    
    def __init__(self, config: Optional[GovernanceConfig] = None):
        self.config = config or GovernanceConfig()
        self.app: Optional[FastAPI] = None
        self.is_running = False
        
        # Governance state
        self.proposals: Dict[str, Proposal] = {}
        self.votes: Dict[str, Vote] = {}
        self.voter_registry: Dict[str, Dict] = {}  # voter_id -> info
        
        # Initialize with sample proposals
        self._create_sample_proposals()
        
        if FASTAPI_AVAILABLE:
            self._setup_app()
        else:
            logger.error("FastAPI not available, governance portal cannot be started")
    
    def _setup_app(self):
        """Setup FastAPI application"""
        self.app = FastAPI(
            title=self.config.title,
            description="PRSM Governance Portal - Democratic Decision Making",
            version="1.0.0"
        )
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup governance portal routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def governance_home(request: Request):
            """Main governance portal page"""
            return self._get_governance_html()
        
        @self.app.get("/api/proposals")
        async def get_proposals(status: Optional[str] = None):
            """Get all proposals or filter by status"""
            try:
                proposals = list(self.proposals.values())
                
                if status:
                    proposals = [p for p in proposals if p.status.value == status]
                
                # Sort by creation date (newest first)
                proposals.sort(key=lambda p: p.created_at, reverse=True)
                
                return JSONResponse({
                    "proposals": jsonable_encoder([asdict(p) for p in proposals]),
                    "count": len(proposals),
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error getting proposals: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)
        
        @self.app.get("/api/proposals/{proposal_id}")
        async def get_proposal(proposal_id: str):
            """Get specific proposal details"""
            try:
                if proposal_id not in self.proposals:
                    raise HTTPException(status_code=404, detail="Proposal not found")
                
                proposal = self.proposals[proposal_id]
                
                # Get related votes
                proposal_votes = [
                    asdict(vote) for vote in self.votes.values() 
                    if vote.proposal_id == proposal_id
                ]
                
                return JSONResponse(content=jsonable_encoder({
                    "proposal": asdict(proposal),
                    "votes": proposal_votes,
                    "vote_count": len(proposal_votes),
                    "timestamp": datetime.now().isoformat()
                }))
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting proposal {proposal_id}: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)
        
        @self.app.post("/api/proposals")
        async def submit_proposal(proposal: ProposalSubmission):
            """Submit new governance proposal"""
            try:
                # Create new proposal
                proposal_id = str(uuid.uuid4())
                new_proposal = Proposal(
                    id=proposal_id,
                    title=proposal.title,
                    description=proposal.description,
                    proposal_type=ProposalType(proposal.proposal_type),
                    proposer_id=proposal.proposer_id,
                    created_at=datetime.now(),
                    voting_deadline=datetime.now() + timedelta(days=self.config.voting_period_days),
                    status=ProposalStatus.ACTIVE,
                    implementation_details=proposal.implementation_details or ""
                )
                
                # Store proposal
                self.proposals[proposal_id] = new_proposal
                
                # Register proposer if new
                if proposal.proposer_id not in self.voter_registry:
                    self.voter_registry[proposal.proposer_id] = {
                        "id": proposal.proposer_id,
                        "joined_at": datetime.now().isoformat(),
                        "proposals_submitted": 1,
                        "votes_cast": 0,
                        "ftns_balance": 10.0  # Starter FTNS for demo
                    }
                else:
                    self.voter_registry[proposal.proposer_id]["proposals_submitted"] += 1
                
                return JSONResponse({
                    "proposal_id": proposal_id,
                    "status": "submitted",
                    "voting_deadline": new_proposal.voting_deadline.isoformat(),
                    "message": "Proposal submitted successfully"
                })
                
            except Exception as e:
                logger.error(f"Error submitting proposal: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)
        
        @self.app.post("/api/vote")
        async def submit_vote(vote: VoteSubmission):
            """Submit vote on proposal"""
            try:
                # Validate proposal exists and is active
                if vote.proposal_id not in self.proposals:
                    raise HTTPException(status_code=404, detail="Proposal not found")
                
                proposal = self.proposals[vote.proposal_id]
                if proposal.status != ProposalStatus.ACTIVE:
                    raise HTTPException(status_code=400, detail="Proposal is not active for voting")
                
                # Check if voting deadline has passed
                if datetime.now() > proposal.voting_deadline:
                    proposal.status = ProposalStatus.REJECTED  # Auto-expire
                    raise HTTPException(status_code=400, detail="Voting deadline has passed")
                
                # Check if user already voted
                existing_vote = None
                for vote_id, v in self.votes.items():
                    if v.proposal_id == vote.proposal_id and v.voter_id == vote.voter_id:
                        existing_vote = vote_id
                        break
                
                if existing_vote:
                    raise HTTPException(status_code=400, detail="User has already voted on this proposal")
                
                # Register voter if new
                if vote.voter_id not in self.voter_registry:
                    self.voter_registry[vote.voter_id] = {
                        "id": vote.voter_id,
                        "joined_at": datetime.now().isoformat(),
                        "proposals_submitted": 0,
                        "votes_cast": 1,
                        "ftns_balance": vote.ftns_weight
                    }
                else:
                    self.voter_registry[vote.voter_id]["votes_cast"] += 1
                
                # Create vote record
                vote_id = str(uuid.uuid4())
                new_vote = Vote(
                    id=vote_id,
                    proposal_id=vote.proposal_id,
                    voter_id=vote.voter_id,
                    choice=VoteChoice(vote.choice),
                    ftns_weight=vote.ftns_weight,
                    timestamp=datetime.now()
                )
                
                # Store vote
                self.votes[vote_id] = new_vote
                
                # Update proposal vote counts
                if new_vote.choice == VoteChoice.YES:
                    proposal.yes_votes += 1
                    proposal.yes_ftns += vote.ftns_weight
                elif new_vote.choice == VoteChoice.NO:
                    proposal.no_votes += 1
                    proposal.no_ftns += vote.ftns_weight
                else:  # ABSTAIN
                    proposal.abstain_votes += 1
                    proposal.abstain_ftns += vote.ftns_weight
                
                proposal.total_voters += 1
                
                # Check if proposal should pass/fail
                self._evaluate_proposal(proposal)
                
                return JSONResponse({
                    "vote_id": vote_id,
                    "status": "recorded",
                    "proposal_status": proposal.status.value,
                    "message": "Vote recorded successfully"
                })
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error submitting vote: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)
        
        @self.app.get("/api/governance-stats")
        async def get_governance_stats():
            """Get governance system statistics"""
            try:
                stats = self._calculate_governance_stats()
                return JSONResponse(content=jsonable_encoder({
                    "stats": asdict(stats),
                    "timestamp": datetime.now().isoformat()
                }))
            except Exception as e:
                logger.error(f"Error getting governance stats: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)
        
        @self.app.get("/api/voter/{voter_id}")
        async def get_voter_info(voter_id: str):
            """Get voter information and history"""
            try:
                if voter_id not in self.voter_registry:
                    raise HTTPException(status_code=404, detail="Voter not found")
                
                voter_info = self.voter_registry[voter_id]
                
                # Get voter's votes
                voter_votes = [
                    asdict(vote) for vote in self.votes.values() 
                    if vote.voter_id == voter_id
                ]
                
                # Get voter's proposals
                voter_proposals = [
                    asdict(proposal) for proposal in self.proposals.values()
                    if proposal.proposer_id == voter_id
                ]
                
                return JSONResponse(content=jsonable_encoder({
                    "voter": voter_info,
                    "votes": voter_votes,
                    "proposals": voter_proposals,
                    "participation_score": self._calculate_participation_score(voter_id),
                    "timestamp": datetime.now().isoformat()
                }))
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting voter info: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return JSONResponse({
                "status": "healthy",
                "governance_running": self.is_running,
                "total_proposals": len(self.proposals),
                "total_votes": len(self.votes),
                "timestamp": datetime.now().isoformat()
            })
    
    def _create_sample_proposals(self):
        """Create sample governance proposals for demonstration"""
        
        # Sample proposal 1: Technical improvement
        proposal1 = Proposal(
            id="prop_001",
            title="Implement Advanced RLT Caching System",
            description="""Proposal to implement a distributed caching system for RLT (Recursive Learning Teacher) components to improve response times by 20-30%.

**Background:**
Current RLT processing shows latency spikes during high-usage periods. A distributed caching layer would store frequently requested teaching optimizations and reduce redundant computations.

**Implementation Plan:**
1. Design Redis-based distributed cache architecture
2. Implement cache invalidation strategies for dynamic learning
3. Add cache hit/miss metrics to monitoring dashboard
4. Deploy across all RLT network nodes

**Expected Benefits:**
- 20-30% improvement in response times
- Reduced computational load on RLT teachers
- Better user experience during peak usage
- Lower FTNS costs for cached operations

**Resource Requirements:**
- 2 weeks development time
- Additional infrastructure costs (~$200/month)
- Testing across testnet environment

**Vote Outcome:**
This proposal requires community approval to allocate development resources and infrastructure budget.""",
            proposal_type=ProposalType.TECHNICAL,
            proposer_id="dev_alice_001",
            created_at=datetime.now() - timedelta(days=3),
            voting_deadline=datetime.now() + timedelta(days=4),
            status=ProposalStatus.ACTIVE,
            implementation_details="Redis cluster setup with 3 nodes, cache TTL of 1 hour for teaching optimizations"
        )
        
        # Add some sample votes for proposal 1
        proposal1.yes_votes = 8
        proposal1.no_votes = 2
        proposal1.abstain_votes = 1
        proposal1.yes_ftns = 125.5
        proposal1.no_ftns = 15.0
        proposal1.abstain_ftns = 5.0
        proposal1.total_voters = 11
        
        # Sample proposal 2: Economic policy
        proposal2 = Proposal(
            id="prop_002", 
            title="Adjust FTNS Token Economics for Better Participation",
            description="""Proposal to modify FTNS token economics to incentivize broader community participation and reduce barriers to entry.

**Current Issues:**
- High FTNS costs discourage new user experimentation
- Limited ways for users to earn FTNS through contribution
- Token distribution concentrated among early adopters

**Proposed Changes:**
1. **Reduce Query Costs:** Lower base FTNS cost by 40%
2. **Introduce Earning Mechanisms:**
   - 1 FTNS for submitting quality governance proposals
   - 0.5 FTNS for casting informed votes (with justification)
   - 2 FTNS for contributing to code/documentation
3. **Daily Free Allowance:** 5 free queries per day for all users
4. **Staking Rewards:** 5% annual yield for FTNS holders who participate in governance

**Impact Analysis:**
- Lower barrier to entry for new users
- Increased daily active users (projected +150%)
- More diverse governance participation
- Sustainable token distribution mechanism

**Implementation Timeline:**
Phase 1 (Month 1): Reduce costs and add daily allowance
Phase 2 (Month 2): Implement earning mechanisms
Phase 3 (Month 3): Launch staking program

**Budget Impact:**
Requires adjustment to token emission schedule and development of earning tracking systems.""",
            proposal_type=ProposalType.ECONOMIC,
            proposer_id="economist_bob_002",
            created_at=datetime.now() - timedelta(days=5),
            voting_deadline=datetime.now() + timedelta(days=2),
            status=ProposalStatus.ACTIVE,
            implementation_details="Smart contract updates for token mechanics, new earning calculation algorithms"
        )
        
        # Add sample votes for proposal 2
        proposal2.yes_votes = 12
        proposal2.no_votes = 3
        proposal2.abstain_votes = 2
        proposal2.yes_ftns = 180.0
        proposal2.no_ftns = 25.0
        proposal2.abstain_ftns = 10.0
        proposal2.total_voters = 17
        
        # Sample proposal 3: Governance improvement 
        proposal3 = Proposal(
            id="prop_003",
            title="Establish PRSM Community Advisory Board",
            description="""Proposal to create a Community Advisory Board to provide structured guidance for PRSM's development priorities and strategic direction.

**Motivation:**
As PRSM grows, we need formal mechanisms for community input on major decisions. An Advisory Board would bridge community needs with development priorities.

**Board Structure:**
- 7 members elected by community vote
- 2-year terms with staggered rotation
- Quarterly meetings with core development team
- Monthly community office hours

**Member Categories:**
1. Technical Representative (2 seats) - Developers/Engineers
2. Research Representative (2 seats) - Academic/Research community  
3. User Representative (2 seats) - Active PRSM users
4. Economic Representative (1 seat) - Token economics expertise

**Responsibilities:**
- Review and advise on major technical proposals
- Provide community feedback to development team
- Help prioritize feature development roadmap
- Facilitate communication between community and core team
- Quarterly transparency reports to community

**Election Process:**
- Nominations open for 2 weeks
- Candidates submit platform statements
- Community vote using FTNS-weighted voting
- Results published with full transparency

**Compensation:**
Board members receive 50 FTNS per month for their service, funded from community treasury.

**Benefits:**
- Structured community representation
- Better alignment between development and user needs
- Democratic decision-making process
- Increased community engagement and ownership""",
            proposal_type=ProposalType.GOVERNANCE,
            proposer_id="community_charlie_003",
            created_at=datetime.now() - timedelta(days=1),
            voting_deadline=datetime.now() + timedelta(days=6),
            status=ProposalStatus.ACTIVE,
            implementation_details="Election system setup, board meeting infrastructure, compensation mechanisms"
        )
        
        # Sample proposal 4: Passed proposal example
        proposal4 = Proposal(
            id="prop_004",
            title="Integrate Additional AI Models into RLT Network",
            description="""Add support for Claude, Gemini, and local Ollama models to expand PRSM's AI coordination capabilities.

This proposal was approved and successfully implemented in the latest release.""",
            proposal_type=ProposalType.TECHNICAL,
            proposer_id="dev_diana_004",
            created_at=datetime.now() - timedelta(days=14),
            voting_deadline=datetime.now() - timedelta(days=7),
            status=ProposalStatus.IMPLEMENTED,
            implementation_details="Multi-model API integration completed",
            yes_votes=15,
            no_votes=1,
            abstain_votes=0,
            yes_ftns=240.0,
            no_ftns=8.0,
            abstain_ftns=0.0,
            total_voters=16,
            implementation_status="completed"
        )
        
        # Store sample proposals
        self.proposals[proposal1.id] = proposal1
        self.proposals[proposal2.id] = proposal2
        self.proposals[proposal3.id] = proposal3
        self.proposals[proposal4.id] = proposal4
        
        # Create sample voter registry
        self.voter_registry = {
            "dev_alice_001": {"id": "dev_alice_001", "joined_at": "2025-06-01", "proposals_submitted": 1, "votes_cast": 3, "ftns_balance": 45.5},
            "economist_bob_002": {"id": "economist_bob_002", "joined_at": "2025-06-05", "proposals_submitted": 1, "votes_cast": 4, "ftns_balance": 32.0},
            "community_charlie_003": {"id": "community_charlie_003", "joined_at": "2025-06-10", "proposals_submitted": 1, "votes_cast": 2, "ftns_balance": 28.5},
            "dev_diana_004": {"id": "dev_diana_004", "joined_at": "2025-05-28", "proposals_submitted": 1, "votes_cast": 5, "ftns_balance": 65.0}
        }
    
    def _evaluate_proposal(self, proposal: Proposal):
        """Evaluate if proposal should pass or fail based on votes"""
        total_ftns = proposal.yes_ftns + proposal.no_ftns + proposal.abstain_ftns
        
        # Check if voting period ended
        if datetime.now() > proposal.voting_deadline:
            # Determine outcome
            if total_ftns == 0:
                proposal.status = ProposalStatus.REJECTED
            else:
                yes_percentage = proposal.yes_ftns / (proposal.yes_ftns + proposal.no_ftns) if (proposal.yes_ftns + proposal.no_ftns) > 0 else 0
                
                if yes_percentage >= self.config.passing_threshold:
                    proposal.status = ProposalStatus.PASSED
                else:
                    proposal.status = ProposalStatus.REJECTED
    
    def _calculate_governance_stats(self) -> GovernanceStats:
        """Calculate governance system statistics"""
        total_proposals = len(self.proposals)
        active_proposals = len([p for p in self.proposals.values() if p.status == ProposalStatus.ACTIVE])
        passed_proposals = len([p for p in self.proposals.values() if p.status == ProposalStatus.PASSED])
        total_votes = len(self.votes)
        total_voters = len(self.voter_registry)
        total_ftns_voted = sum(vote.ftns_weight for vote in self.votes.values())
        
        # Calculate participation rate (voters who have voted / total voters)
        active_voters = len(set(vote.voter_id for vote in self.votes.values()))
        participation_rate = active_voters / max(1, total_voters)
        
        # Calculate average proposal duration
        completed_proposals = [p for p in self.proposals.values() if p.status in [ProposalStatus.PASSED, ProposalStatus.REJECTED]]
        if completed_proposals:
            durations = [(p.voting_deadline - p.created_at).days for p in completed_proposals]
            avg_duration = sum(durations) / len(durations)
        else:
            avg_duration = self.config.voting_period_days
        
        return GovernanceStats(
            total_proposals=total_proposals,
            active_proposals=active_proposals,
            passed_proposals=passed_proposals,
            total_votes=total_votes,
            total_voters=total_voters,
            total_ftns_voted=total_ftns_voted,
            participation_rate=participation_rate,
            avg_proposal_duration_days=avg_duration
        )
    
    def _calculate_participation_score(self, voter_id: str) -> float:
        """Calculate participation score for a voter"""
        if voter_id not in self.voter_registry:
            return 0.0
        
        voter = self.voter_registry[voter_id]
        proposals_weight = voter["proposals_submitted"] * 2
        votes_weight = voter["votes_cast"] * 1
        
        return min(100.0, proposals_weight + votes_weight)
    
    def _get_governance_html(self) -> HTMLResponse:
        """Generate the governance portal HTML"""
        
        html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRSM Governance Portal - Democratic Decision Making</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 2rem;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            color: #4CAF50;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            font-weight: 700;
        }
        
        .header p {
            color: #666;
            font-size: 1.2rem;
            margin-bottom: 1rem;
        }
        
        .governance-badge {
            display: inline-block;
            background: linear-gradient(90deg, #9C27B0, #673AB7);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        
        .nav-tabs {
            background: rgba(255, 255, 255, 0.9);
            display: flex;
            justify-content: center;
            gap: 0;
            padding: 0;
            margin: 0;
        }
        
        .nav-tab {
            background: transparent;
            border: none;
            padding: 1rem 2rem;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            color: #666;
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
        }
        
        .nav-tab.active {
            color: #4CAF50;
            border-bottom-color: #4CAF50;
            background: rgba(76, 175, 80, 0.1);
        }
        
        .nav-tab:hover {
            background: rgba(76, 175, 80, 0.05);
        }
        
        .container {
            max-width: 1200px;
            margin: 2rem auto;
            padding: 0 2rem;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .card h3 {
            color: #4CAF50;
            font-size: 1.4rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .stat-item {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid #e0e0e0;
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            color: #4CAF50;
            margin-bottom: 0.5rem;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .proposal-list {
            display: grid;
            gap: 1.5rem;
        }
        
        .proposal-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #e0e0e0;
            transition: transform 0.3s ease;
        }
        
        .proposal-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        
        .proposal-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 1rem;
        }
        
        .proposal-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #333;
            margin-bottom: 0.5rem;
        }
        
        .proposal-type {
            padding: 0.25rem 0.75rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .type-technical { background: #e3f2fd; color: #1976d2; }
        .type-economic { background: #f3e5f5; color: #7b1fa2; }
        .type-governance { background: #e8f5e8; color: #388e3c; }
        .type-community { background: #fff3e0; color: #f57c00; }
        
        .proposal-meta {
            display: flex;
            gap: 1rem;
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 1rem;
        }
        
        .proposal-description {
            color: #555;
            line-height: 1.6;
            margin-bottom: 1rem;
            max-height: 100px;
            overflow: hidden;
            position: relative;
        }
        
        .voting-section {
            border-top: 1px solid #e0e0e0;
            padding-top: 1rem;
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 1rem;
            align-items: center;
        }
        
        .vote-results {
            display: flex;
            gap: 1rem;
            font-size: 0.9rem;
        }
        
        .vote-count {
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }
        
        .vote-yes { color: #4CAF50; }
        .vote-no { color: #f44336; }
        .vote-abstain { color: #ff9800; }
        
        .vote-buttons {
            display: flex;
            gap: 0.5rem;
            justify-content: flex-end;
        }
        
        .vote-btn {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 20px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .vote-yes-btn { background: #4CAF50; color: white; }
        .vote-no-btn { background: #f44336; color: white; }
        .vote-abstain-btn { background: #ff9800; color: white; }
        
        .vote-btn:hover {
            transform: scale(1.05);
        }
        
        .form-group {
            margin-bottom: 1.5rem;
        }
        
        .form-label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 600;
            color: #333;
        }
        
        .form-input, .form-textarea, .form-select {
            width: 100%;
            padding: 0.75rem;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1rem;
            font-family: inherit;
            transition: border-color 0.3s ease;
        }
        
        .form-input:focus, .form-textarea:focus, .form-select:focus {
            outline: none;
            border-color: #4CAF50;
        }
        
        .form-textarea {
            min-height: 120px;
            resize: vertical;
        }
        
        .submit-btn {
            background: linear-gradient(90deg, #4CAF50, #45a049);
            color: white;
            border: none;
            padding: 1rem 2rem;
            border-radius: 25px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.3s ease;
        }
        
        .submit-btn:hover {
            transform: translateY(-2px);
        }
        
        .loading {
            text-align: center;
            padding: 2rem;
            color: #666;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #4CAF50;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .status-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .status-active { background: #e8f5e8; color: #388e3c; }
        .status-passed { background: #e3f2fd; color: #1976d2; }
        .status-rejected { background: #ffebee; color: #d32f2f; }
        .status-implemented { background: #f3e5f5; color: #7b1fa2; }
        
        .footer {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 2rem;
            text-align: center;
            color: #666;
            margin-top: 2rem;
        }
        
        .footer a {
            color: #4CAF50;
            text-decoration: none;
        }
        
        .footer a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏛️ PRSM Governance Portal</h1>
        <p>Democratic Decision Making for AI Coordination Protocol</p>
        <div class="governance-badge">🗳️ Community-Driven Development</div>
    </div>
    
    <div class="nav-tabs">
        <button class="nav-tab active" onclick="showTab('overview')">📊 Overview</button>
        <button class="nav-tab" onclick="showTab('proposals')">📋 Proposals</button>
        <button class="nav-tab" onclick="showTab('vote')">🗳️ Vote</button>
        <button class="nav-tab" onclick="showTab('submit')">📝 Submit Proposal</button>
    </div>
    
    <div class="container">
        <!-- Overview Tab -->
        <div id="overview-tab" class="tab-content active">
            <div class="card">
                <h3>📊 Governance Statistics</h3>
                <div class="stats-grid" id="governance-stats">
                    <div class="stat-item">
                        <div class="stat-value" id="total-proposals">-</div>
                        <div class="stat-label">Total Proposals</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="active-proposals">-</div>
                        <div class="stat-label">Active Proposals</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="total-votes">-</div>
                        <div class="stat-label">Total Votes Cast</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="participation-rate">-</div>
                        <div class="stat-label">Participation Rate</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3>🎯 How Governance Works</h3>
                <p><strong>PRSM uses democratic, token-weighted voting for all major decisions:</strong></p>
                <br>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
                    <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                        <h4 style="color: #4CAF50; margin-bottom: 0.5rem;">📝 1. Proposal Submission</h4>
                        <p>Community members submit proposals for network improvements, covering technical, economic, and governance changes.</p>
                    </div>
                    <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                        <h4 style="color: #4CAF50; margin-bottom: 0.5rem;">🗳️ 2. Community Voting</h4>
                        <p>All FTNS token holders can vote Yes, No, or Abstain. Votes are weighted by FTNS balance to prevent sybil attacks.</p>
                    </div>
                    <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                        <h4 style="color: #4CAF50; margin-bottom: 0.5rem;">⏰ 3. Voting Period</h4>
                        <p>Proposals have a 7-day voting period. 60% Yes votes required to pass, with transparent result tracking.</p>
                    </div>
                    <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                        <h4 style="color: #4CAF50; margin-bottom: 0.5rem;">🚀 4. Implementation</h4>
                        <p>Passed proposals are implemented by the development team with community oversight and progress updates.</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Proposals Tab -->
        <div id="proposals-tab" class="tab-content">
            <div class="card">
                <h3>📋 All Governance Proposals</h3>
                <div id="proposals-list" class="proposal-list">
                    <div class="loading">
                        <div class="spinner"></div>
                        Loading proposals...
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Vote Tab -->
        <div id="vote-tab" class="tab-content">
            <div class="card">
                <h3>🗳️ Active Proposals - Cast Your Vote</h3>
                <div id="active-proposals-list" class="proposal-list">
                    <div class="loading">
                        <div class="spinner"></div>
                        Loading active proposals...
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Submit Proposal Tab -->
        <div id="submit-tab" class="tab-content">
            <div class="card">
                <h3>📝 Submit New Governance Proposal</h3>
                <form id="proposal-form">
                    <div class="form-group">
                        <label class="form-label" for="proposal-title">Proposal Title</label>
                        <input type="text" id="proposal-title" class="form-input" 
                               placeholder="Clear, descriptive title for your proposal" required>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label" for="proposal-type">Proposal Type</label>
                        <select id="proposal-type" class="form-select" required>
                            <option value="">Select proposal type</option>
                            <option value="technical">Technical - System improvements, features</option>
                            <option value="economic">Economic - Token economics, incentives</option>
                            <option value="governance">Governance - Decision-making processes</option>
                            <option value="community">Community - Events, outreach, partnerships</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label" for="proposal-description">Detailed Description</label>
                        <textarea id="proposal-description" class="form-textarea" 
                                  placeholder="Provide comprehensive details about your proposal including:
• Background and motivation
• Specific implementation plan
• Expected benefits and impact
• Resource requirements
• Timeline and milestones

Be thorough - this helps the community make informed decisions." required></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label" for="implementation-details">Implementation Details (Optional)</label>
                        <textarea id="implementation-details" class="form-textarea" 
                                  placeholder="Technical specifications, development tasks, or other implementation details"></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label" for="proposer-id">Your User ID</label>
                        <input type="text" id="proposer-id" class="form-input" 
                               placeholder="Your unique identifier (e.g., dev_alice_001)" required>
                    </div>
                    
                    <button type="submit" class="submit-btn">Submit Proposal for Community Vote</button>
                </form>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>
            <strong>PRSM Governance Portal</strong><br>
            Democratic decision-making for the Protocol for Recursive Scientific Modeling<br>
            <a href="https://github.com/Ryno2390/PRSM" target="_blank">GitHub</a> |
            <a href="../state_of_network_dashboard.py" target="_blank">Network Status</a> |
            <a href="../testnet_interface.py" target="_blank">Public Testnet</a>
        </p>
    </div>
    
    <script>
        let currentUserId = localStorage.getItem('prsm_governance_user_id') || generateUserId();
        
        function generateUserId() {
            const id = 'voter_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('prsm_governance_user_id', id);
            return id;
        }
        
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Remove active class from all nav tabs
            document.querySelectorAll('.nav-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
            
            // Load data for specific tabs
            if (tabName === 'proposals') {
                loadAllProposals();
            } else if (tabName === 'vote') {
                loadActiveProposals();
            } else if (tabName === 'overview') {
                loadGovernanceStats();
            }
        }
        
        async function loadGovernanceStats() {
            try {
                const response = await fetch('/api/governance-stats');
                const data = await response.json();
                const stats = data.stats;
                
                document.getElementById('total-proposals').textContent = stats.total_proposals;
                document.getElementById('active-proposals').textContent = stats.active_proposals;
                document.getElementById('total-votes').textContent = stats.total_votes;
                document.getElementById('participation-rate').textContent = (stats.participation_rate * 100).toFixed(1) + '%';
            } catch (error) {
                console.error('Error loading governance stats:', error);
            }
        }
        
        async function loadAllProposals() {
            try {
                const response = await fetch('/api/proposals');
                const data = await response.json();
                
                const container = document.getElementById('proposals-list');
                container.innerHTML = '';
                
                if (data.proposals.length === 0) {
                    container.innerHTML = '<p style="text-align: center; color: #666;">No proposals found.</p>';
                    return;
                }
                
                data.proposals.forEach(proposal => {
                    const proposalEl = createProposalElement(proposal, false);
                    container.appendChild(proposalEl);
                });
            } catch (error) {
                console.error('Error loading proposals:', error);
                document.getElementById('proposals-list').innerHTML = '<p style="color: #f44336;">Error loading proposals</p>';
            }
        }
        
        async function loadActiveProposals() {
            try {
                const response = await fetch('/api/proposals?status=active');
                const data = await response.json();
                
                const container = document.getElementById('active-proposals-list');
                container.innerHTML = '';
                
                if (data.proposals.length === 0) {
                    container.innerHTML = '<p style="text-align: center; color: #666;">No active proposals available for voting.</p>';
                    return;
                }
                
                data.proposals.forEach(proposal => {
                    const proposalEl = createProposalElement(proposal, true);
                    container.appendChild(proposalEl);
                });
            } catch (error) {
                console.error('Error loading active proposals:', error);
                document.getElementById('active-proposals-list').innerHTML = '<p style="color: #f44336;">Error loading active proposals</p>';
            }
        }
        
        function createProposalElement(proposal, showVoting) {
            const div = document.createElement('div');
            div.className = 'proposal-card';
            
            const statusClass = `status-${proposal.status}`;
            const typeClass = `type-${proposal.proposal_type}`;
            
            // Truncate description
            const description = proposal.description.length > 200 
                ? proposal.description.substring(0, 200) + '...'
                : proposal.description;
            
            div.innerHTML = `
                <div class="proposal-header">
                    <div>
                        <div class="proposal-title">${proposal.title}</div>
                        <div class="proposal-meta">
                            <span class="proposal-type ${typeClass}">${proposal.proposal_type}</span>
                            <span class="status-badge ${statusClass}">${proposal.status}</span>
                            <span>By: ${proposal.proposer_id}</span>
                            <span>Deadline: ${new Date(proposal.voting_deadline).toLocaleDateString()}</span>
                        </div>
                    </div>
                </div>
                <div class="proposal-description">${description}</div>
                ${showVoting && proposal.status === 'active' ? `
                    <div class="voting-section">
                        <div class="vote-results">
                            <div class="vote-count vote-yes">👍 ${proposal.yes_votes} (${proposal.yes_ftns} FTNS)</div>
                            <div class="vote-count vote-no">👎 ${proposal.no_votes} (${proposal.no_ftns} FTNS)</div>
                            <div class="vote-count vote-abstain">🤷 ${proposal.abstain_votes} (${proposal.abstain_ftns} FTNS)</div>
                        </div>
                        <div class="vote-buttons">
                            <button class="vote-btn vote-yes-btn" onclick="castVote('${proposal.id}', 'yes')">Vote Yes</button>
                            <button class="vote-btn vote-no-btn" onclick="castVote('${proposal.id}', 'no')">Vote No</button>
                            <button class="vote-btn vote-abstain-btn" onclick="castVote('${proposal.id}', 'abstain')">Abstain</button>
                        </div>
                    </div>
                ` : `
                    <div class="vote-results">
                        <div class="vote-count vote-yes">👍 ${proposal.yes_votes} (${proposal.yes_ftns} FTNS)</div>
                        <div class="vote-count vote-no">👎 ${proposal.no_votes} (${proposal.no_ftns} FTNS)</div>
                        <div class="vote-count vote-abstain">🤷 ${proposal.abstain_votes} (${proposal.abstain_ftns} FTNS)</div>
                    </div>
                `}
            `;
            
            return div;
        }
        
        async function castVote(proposalId, choice) {
            try {
                const ftnsWeight = parseFloat(prompt('Enter your FTNS token weight for this vote:', '1.0')) || 1.0;
                
                const response = await fetch('/api/vote', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        proposal_id: proposalId,
                        voter_id: currentUserId,
                        choice: choice,
                        ftns_weight: ftnsWeight
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    alert(`Vote recorded successfully! Status: ${data.status}`);
                    loadActiveProposals(); // Refresh the list
                    loadGovernanceStats(); // Update stats
                } else {
                    alert(`Error: ${data.error}`);
                }
            } catch (error) {
                console.error('Error casting vote:', error);
                alert('Error casting vote. Please try again.');
            }
        }
        
        // Proposal submission
        document.getElementById('proposal-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const title = document.getElementById('proposal-title').value;
            const type = document.getElementById('proposal-type').value;
            const description = document.getElementById('proposal-description').value;
            const implementation = document.getElementById('implementation-details').value;
            const proposerId = document.getElementById('proposer-id').value;
            
            try {
                const response = await fetch('/api/proposals', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        title: title,
                        description: description,
                        proposal_type: type,
                        proposer_id: proposerId,
                        implementation_details: implementation
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    alert(`Proposal submitted successfully! ID: ${data.proposal_id}`);
                    document.getElementById('proposal-form').reset();
                    showTab('proposals'); // Switch to proposals tab
                } else {
                    alert(`Error: ${data.error}`);
                }
            } catch (error) {
                console.error('Error submitting proposal:', error);
                alert('Error submitting proposal. Please try again.');
            }
        });
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            loadGovernanceStats();
            
            // Set proposer ID field to current user
            document.getElementById('proposer-id').value = currentUserId;
        });
    </script>
</body>
</html>
        '''
        
        return HTMLResponse(content=html_content)
    
    async def start_governance(self):
        """Start the governance portal"""
        if not FASTAPI_AVAILABLE:
            logger.error("FastAPI not available, cannot start governance portal")
            return
        
        if self.is_running:
            logger.warning("Governance portal already running")
            return
        
        self.is_running = True
        
        logger.info(f"Starting PRSM Governance Portal on {self.config.host}:{self.config.port}")
        
        config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def stop_governance(self):
        """Stop the governance portal"""
        self.is_running = False
        logger.info("Stopped PRSM Governance Portal")
    
    def get_governance_url(self) -> str:
        """Get the governance portal URL"""
        return f"http://{self.config.host}:{self.config.port}"


# CLI runner
async def main():
    """Run the PRSM Governance Portal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PRSM Governance Portal")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8095, help="Port to bind to")
    
    args = parser.parse_args()
    
    config = GovernanceConfig(host=args.host, port=args.port)
    governance = PRSMGovernancePortal(config)
    
    print("🏛️ PRSM Governance Portal")
    print("=" * 50)
    print(f"🔗 Portal URL: {governance.get_governance_url()}")
    print("🗳️ Democratic decision-making for PRSM development")
    print("📋 Submit proposals and participate in voting")
    print("💰 FTNS token-weighted governance system")
    print("")
    print("Features:")
    print("  • View and vote on governance proposals")
    print("  • Submit new proposals for community consideration")
    print("  • Track voting results and proposal implementation")
    print("  • Participate in PRSM's democratic development")
    print("  • FTNS token-weighted voting system")
    print("")
    print("Press Ctrl+C to stop...")
    
    try:
        await governance.start_governance()
    except KeyboardInterrupt:
        print("\n🛑 Stopping PRSM Governance Portal...")
        await governance.stop_governance()


if __name__ == "__main__":
    asyncio.run(main())