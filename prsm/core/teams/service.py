"""
PRSM Team Service

Core team management functionality including team formation,
membership management, and collaborative operations.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID, uuid4

import structlog

from prsm.core.models import FTNSTransaction
from prsm.economy.tokenomics.ftns_service import ftns_service
from prsm.core.safety.monitor import SafetyMonitor
from .models import (
    Team, TeamMember, TeamWallet, TeamTask, TeamInvitation,
    TeamRole, TeamMembershipStatus, TeamType, GovernanceModel, RewardPolicy
)
from .wallet import get_team_wallet_service

logger = structlog.get_logger(__name__)


class TeamService:
    """
    Core team management service for PRSM collaborative functionality
    
    Features:
    - Team creation and configuration
    - Membership management with role-based permissions
    - Invitation system with stake requirements
    - Team discovery and directory services
    - Integration with FTNS tokenomics and safety systems
    """
    
    def __init__(self):
        self.service_id = str(uuid4())
        self.logger = logger.bind(component="team_service", service_id=self.service_id)
        
        # Service state
        self.teams: Dict[UUID, Team] = {}
        self.team_members: Dict[UUID, List[TeamMember]] = {}  # team_id -> members
        self.user_teams: Dict[str, List[UUID]] = {}           # user_id -> team_ids
        self.pending_invitations: Dict[UUID, TeamInvitation] = {}
        
        # Team directory and discovery
        self.team_directory: Dict[UUID, Dict[str, Any]] = {}
        self.team_search_index: Dict[str, List[UUID]] = {}  # keyword -> team_ids
        
        # Services integration
        self.wallet_service = get_team_wallet_service()
        self.safety_monitor = SafetyMonitor()
        
        # Performance statistics
        self.team_stats = {
            "total_teams_created": 0,
            "total_members_joined": 0,
            "total_invitations_sent": 0,
            "total_tasks_created": 0,
            "active_teams": 0,
            "average_team_size": 0.0,
            "most_popular_team_type": None
        }
        
        # Synchronization
        self._team_lock = asyncio.Lock()
        self._member_lock = asyncio.Lock()
        self._invitation_lock = asyncio.Lock()
        
        print("🧑‍🤝‍🧑 TeamService initialized")
    
    
    async def create_team(self, founder_id: str, team_data: Dict[str, Any]) -> Team:
        """
        Create a new team with the specified configuration
        
        Args:
            founder_id: User ID of the team founder
            team_data: Team configuration data
            
        Returns:
            Created team instance
        """
        try:
            async with self._team_lock:
                # Validate founder eligibility
                if not await self._validate_team_creation_eligibility(founder_id):
                    raise ValueError("User not eligible to create teams")
                
                # Validate team data
                if not await self._validate_team_data(team_data):
                    raise ValueError("Invalid team configuration")
                
                # Check team name uniqueness
                if await self._is_team_name_taken(team_data.get("name", "")):
                    raise ValueError("Team name already taken")
                
                # Create team instance
                team = Team(
                    name=team_data["name"],
                    description=team_data["description"],
                    team_type=TeamType(team_data.get("team_type", "research")),
                    governance_model=GovernanceModel(team_data.get("governance_model", "democratic")),
                    reward_policy=RewardPolicy(team_data.get("reward_policy", "proportional")),
                    is_public=team_data.get("is_public", True),
                    max_members=team_data.get("max_members"),
                    entry_stake_required=team_data.get("entry_stake_required", 0.0),
                    research_domains=team_data.get("research_domains", []),
                    keywords=team_data.get("keywords", []),
                    external_links=team_data.get("external_links", {}),
                    contact_info=team_data.get("contact_info", {}),
                    metadata=team_data.get("metadata", {})
                )
                
                # Store team
                self.teams[team.team_id] = team
                self.team_members[team.team_id] = []
                
                # Create founder membership
                founder_member = await self._create_founder_membership(team.team_id, founder_id)
                self.team_members[team.team_id].append(founder_member)
                
                # Update user teams mapping
                if founder_id not in self.user_teams:
                    self.user_teams[founder_id] = []
                self.user_teams[founder_id].append(team.team_id)
                
                # Create team wallet
                team_wallet = await self.wallet_service.create_team_wallet(
                    team, [founder_id], required_signatures=1
                )
                
                # Add to team directory
                await self._add_to_directory(team)
                
                # Update team statistics
                team.member_count = 1
                
                # Update service statistics
                self.team_stats["total_teams_created"] += 1
                self.team_stats["active_teams"] += 1
                
                self.logger.info(
                    "Team created",
                    team_id=str(team.team_id),
                    name=team.name,
                    founder=founder_id,
                    team_type=team.team_type
                )
                
                return team
                
        except Exception as e:
            self.logger.error("Failed to create team", error=str(e))
            raise
    
    
    async def invite_member(self, team_id: UUID, inviter_id: str, invitee_id: str,
                           role: TeamRole = TeamRole.MEMBER, message: Optional[str] = None) -> TeamInvitation:
        """
        Invite a user to join a team
        
        Args:
            team_id: Team to invite user to
            inviter_id: User sending the invitation
            invitee_id: User being invited
            role: Role to assign to invited user
            message: Optional invitation message
            
        Returns:
            Created invitation
        """
        try:
            async with self._invitation_lock:
                # Validate team exists
                if team_id not in self.teams:
                    raise ValueError("Team not found")
                
                team = self.teams[team_id]
                
                # Validate inviter permissions
                if not await self._can_invite_members(team_id, inviter_id):
                    raise ValueError("User not authorized to invite members")
                
                # Check if user is already a member
                if await self._is_team_member(team_id, invitee_id):
                    raise ValueError("User is already a team member")
                
                # Check if invitation already exists
                existing_invitation = await self._get_pending_invitation(team_id, invitee_id)
                if existing_invitation:
                    raise ValueError("Invitation already pending")
                
                # Check team capacity
                if team.max_members and team.member_count >= team.max_members:
                    raise ValueError("Team is at maximum capacity")
                
                # Create invitation
                invitation = TeamInvitation(
                    team_id=team_id,
                    invited_user=invitee_id,
                    invited_by=inviter_id,
                    role=role,
                    message=message,
                    required_stake=team.entry_stake_required,
                    expires_at=datetime.now(timezone.utc) + timedelta(days=7)  # 7 day expiry
                )
                
                # Store invitation
                self.pending_invitations[invitation.invitation_id] = invitation
                
                # Update statistics
                self.team_stats["total_invitations_sent"] += 1
                
                self.logger.info(
                    "Team invitation sent",
                    team_id=str(team_id),
                    inviter=inviter_id,
                    invitee=invitee_id,
                    role=role,
                    invitation_id=str(invitation.invitation_id)
                )
                
                return invitation
                
        except Exception as e:
            self.logger.error("Failed to send team invitation", error=str(e))
            raise
    
    
    async def accept_invitation(self, invitation_id: UUID, user_id: str) -> bool:
        """
        Accept a team invitation
        
        Args:
            invitation_id: Invitation to accept
            user_id: User accepting the invitation
            
        Returns:
            True if invitation accepted successfully
        """
        try:
            async with self._member_lock:
                # Validate invitation exists
                if invitation_id not in self.pending_invitations:
                    raise ValueError("Invitation not found")
                
                invitation = self.pending_invitations[invitation_id]
                
                # Validate user is the invitee
                if invitation.invited_user != user_id:
                    raise ValueError("User not authorized to accept this invitation")
                
                # Check if invitation expired
                if datetime.now(timezone.utc) > invitation.expires_at:
                    invitation.status = "expired"
                    raise ValueError("Invitation has expired")
                
                # Check if invitation already responded to
                if invitation.status != "pending":
                    raise ValueError("Invitation already responded to")
                
                # Validate team still exists and has capacity
                team = self.teams.get(invitation.team_id)
                if not team:
                    raise ValueError("Team no longer exists")
                
                if team.max_members and team.member_count >= team.max_members:
                    raise ValueError("Team is now at maximum capacity")
                
                # Check entry stake requirement
                if invitation.required_stake > 0:
                    user_balance = await ftns_service.get_user_balance(user_id)
                    if user_balance.balance < invitation.required_stake:
                        raise ValueError("Insufficient FTNS for entry stake")
                    
                    # Charge entry stake
                    success = await ftns_service.charge_context_access(user_id, int(invitation.required_stake))
                    if not success:
                        raise ValueError("Failed to charge entry stake")
                
                # Create team membership
                member = TeamMember(
                    team_id=invitation.team_id,
                    user_id=user_id,
                    role=invitation.role,
                    status=TeamMembershipStatus.ACTIVE,
                    invited_by=invitation.invited_by,
                    joined_at=datetime.now(timezone.utc),
                    ftns_contributed=invitation.required_stake
                )
                
                # Add member to team
                self.team_members[invitation.team_id].append(member)
                
                # Update user teams mapping
                if user_id not in self.user_teams:
                    self.user_teams[user_id] = []
                self.user_teams[user_id].append(invitation.team_id)
                
                # Update team statistics
                team.member_count += 1
                
                # Update invitation status
                invitation.status = "accepted"
                invitation.responded_at = datetime.now(timezone.utc)
                
                # Update service statistics
                self.team_stats["total_members_joined"] += 1
                self._update_average_team_size()
                
                self.logger.info(
                    "Team invitation accepted",
                    team_id=str(invitation.team_id),
                    user_id=user_id,
                    role=invitation.role,
                    entry_stake=invitation.required_stake
                )
                
                return True
                
        except Exception as e:
            self.logger.error("Failed to accept team invitation", error=str(e))
            return False
    
    
    async def leave_team(self, team_id: UUID, user_id: str, reason: Optional[str] = None) -> bool:
        """
        Leave a team
        
        Args:
            team_id: Team to leave
            user_id: User leaving the team
            reason: Optional reason for leaving
            
        Returns:
            True if successfully left team
        """
        try:
            async with self._member_lock:
                # Find member record
                member = await self._get_team_member(team_id, user_id)
                if not member:
                    raise ValueError("User is not a member of this team")
                
                team = self.teams.get(team_id)
                if not team:
                    raise ValueError("Team not found")
                
                # Check if user is the only owner
                if member.role == TeamRole.OWNER:
                    owners = [m for m in self.team_members[team_id] 
                            if m.role == TeamRole.OWNER and m.status == TeamMembershipStatus.ACTIVE]
                    if len(owners) <= 1:
                        raise ValueError("Cannot leave team - must transfer ownership first")
                
                # Update member status
                member.status = TeamMembershipStatus.LEFT
                member.left_at = datetime.now(timezone.utc)
                
                # Update team statistics
                team.member_count -= 1
                
                # Remove from user teams mapping
                if user_id in self.user_teams:
                    self.user_teams[user_id] = [tid for tid in self.user_teams[user_id] if tid != team_id]
                
                # Update service statistics
                self._update_average_team_size()
                
                self.logger.info(
                    "User left team",
                    team_id=str(team_id),
                    user_id=user_id,
                    reason=reason
                )
                
                return True
                
        except Exception as e:
            self.logger.error("Failed to leave team", error=str(e))
            return False
    
    
    async def create_team_task(self, team_id: UUID, creator_id: str, task_data: Dict[str, Any]) -> TeamTask:
        """
        Create a collaborative task for the team
        
        Args:
            team_id: Team to create task for
            creator_id: User creating the task
            task_data: Task configuration data
            
        Returns:
            Created team task
        """
        try:
            # Validate team membership and permissions
            if not await self._can_manage_tasks(team_id, creator_id):
                raise ValueError("User not authorized to create tasks")
            
            # Validate task data
            if not task_data.get("title") or not task_data.get("description"):
                raise ValueError("Task title and description required")
            
            # Create task
            task = TeamTask(
                team_id=team_id,
                title=task_data["title"],
                description=task_data["description"],
                task_type=task_data.get("task_type", "research"),
                created_by=creator_id,
                priority=task_data.get("priority", "medium"),
                assigned_to=task_data.get("assigned_to", []),
                ftns_budget=task_data.get("ftns_budget", 0.0),
                due_date=task_data.get("due_date"),
                estimated_hours=task_data.get("estimated_hours"),
                requires_consensus=task_data.get("requires_consensus", False),
                consensus_threshold=task_data.get("consensus_threshold", 0.6),
                tags=task_data.get("tags", []),
                external_links=task_data.get("external_links", {}),
                metadata=task_data.get("metadata", {})
            )
            
            # Update team statistics
            team = self.teams[team_id]
            team.total_tasks_completed += 1  # Will be decremented when actually completed
            
            # Update service statistics
            self.team_stats["total_tasks_created"] += 1
            
            self.logger.info(
                "Team task created",
                team_id=str(team_id),
                task_id=str(task.task_id),
                creator=creator_id,
                title=task.title
            )
            
            return task
            
        except Exception as e:
            self.logger.error("Failed to create team task", error=str(e))
            raise
    
    
    async def get_team(self, team_id: UUID) -> Optional[Team]:
        """Get team by ID"""
        return self.teams.get(team_id)
    
    
    async def get_user_teams(self, user_id: str) -> List[Team]:
        """Get all teams a user is a member of"""
        team_ids = self.user_teams.get(user_id, [])
        return [self.teams[tid] for tid in team_ids if tid in self.teams]
    
    
    async def get_team_members(self, team_id: UUID) -> List[TeamMember]:
        """Get all members of a team"""
        return self.team_members.get(team_id, [])
    
    
    async def search_teams(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Team]:
        """
        Search for teams in the directory
        
        Args:
            query: Search query
            filters: Optional filters (team_type, research_domains, etc.)
            
        Returns:
            List of matching teams
        """
        try:
            results = []
            query_lower = query.lower()
            
            for team in self.teams.values():
                if not team.is_public or not team.is_active:
                    continue
                
                # Text search
                if (query_lower in team.name.lower() or 
                    query_lower in team.description.lower() or
                    any(query_lower in keyword.lower() for keyword in team.keywords)):
                    
                    # Apply filters
                    if filters:
                        if filters.get("team_type") and team.team_type != filters["team_type"]:
                            continue
                        if filters.get("research_domains"):
                            if not any(domain in team.research_domains for domain in filters["research_domains"]):
                                continue
                        if filters.get("min_members") and team.member_count < filters["min_members"]:
                            continue
                        if filters.get("max_members") and team.member_count > filters["max_members"]:
                            continue
                    
                    results.append(team)
            
            # Sort by relevance (simplified)
            results.sort(key=lambda t: t.impact_score, reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error("Failed to search teams", error=str(e))
            return []
    
    
    # === Private Helper Methods ===
    
    async def _validate_team_creation_eligibility(self, user_id: str) -> bool:
        """Validate if user is eligible to create teams"""
        # Check minimum FTNS balance
        user_balance = await ftns_service.get_user_balance(user_id)
        min_balance = 1000.0  # 1000 FTNS minimum for team creation
        
        if user_balance.balance < min_balance:
            return False
        
        # Check if user has reached maximum number of teams
        user_teams = self.user_teams.get(user_id, [])
        max_teams_per_user = 5  # Maximum 5 teams per user
        
        if len(user_teams) >= max_teams_per_user:
            return False
        
        return True
    
    
    async def _validate_team_data(self, team_data: Dict[str, Any]) -> bool:
        """Validate team creation data"""
        required_fields = ["name", "description"]
        
        for field in required_fields:
            if not team_data.get(field):
                return False
        
        # Validate name length
        if len(team_data["name"]) < 3 or len(team_data["name"]) > 100:
            return False
        
        # Validate description length
        if len(team_data["description"]) < 10 or len(team_data["description"]) > 1000:
            return False
        
        # Safety validation
        safety_check = await self.safety_monitor.validate_model_output(
            {"team_data": team_data},
            ["no_harmful_content", "content_appropriateness"]
        )
        
        return safety_check
    
    
    async def _is_team_name_taken(self, name: str) -> bool:
        """Check if team name is already taken"""
        name_lower = name.lower()
        return any(team.name.lower() == name_lower for team in self.teams.values())
    
    
    async def _create_founder_membership(self, team_id: UUID, founder_id: str) -> TeamMember:
        """Create founder membership record"""
        return TeamMember(
            team_id=team_id,
            user_id=founder_id,
            role=TeamRole.OWNER,
            status=TeamMembershipStatus.ACTIVE,
            joined_at=datetime.now(timezone.utc),
            can_invite_members=True,
            can_manage_tasks=True,
            can_access_treasury=True,
            can_vote=True
        )
    
    
    async def _add_to_directory(self, team: Team):
        """Add team to searchable directory"""
        directory_entry = {
            "team_id": team.team_id,
            "name": team.name,
            "description": team.description,
            "team_type": team.team_type,
            "member_count": team.member_count,
            "impact_score": team.impact_score,
            "research_domains": team.research_domains,
            "keywords": team.keywords,
            "is_recruiting": team.member_count < (team.max_members or 999),
            "contact_info": team.contact_info
        }
        
        self.team_directory[team.team_id] = directory_entry
        
        # Update search index
        for keyword in team.keywords:
            keyword_lower = keyword.lower()
            if keyword_lower not in self.team_search_index:
                self.team_search_index[keyword_lower] = []
            self.team_search_index[keyword_lower].append(team.team_id)
    
    
    async def _can_invite_members(self, team_id: UUID, user_id: str) -> bool:
        """Check if user can invite members to team"""
        member = await self._get_team_member(team_id, user_id)
        if not member or member.status != TeamMembershipStatus.ACTIVE:
            return False
        
        return member.can_invite_members or member.role in [TeamRole.OWNER, TeamRole.ADMIN]
    
    
    async def _can_manage_tasks(self, team_id: UUID, user_id: str) -> bool:
        """Check if user can manage tasks for team"""
        member = await self._get_team_member(team_id, user_id)
        if not member or member.status != TeamMembershipStatus.ACTIVE:
            return False
        
        return member.can_manage_tasks or member.role in [TeamRole.OWNER, TeamRole.ADMIN, TeamRole.OPERATOR]
    
    
    async def _is_team_member(self, team_id: UUID, user_id: str) -> bool:
        """Check if user is a team member"""
        members = self.team_members.get(team_id, [])
        return any(m.user_id == user_id and m.status in [TeamMembershipStatus.ACTIVE, TeamMembershipStatus.PENDING] 
                  for m in members)
    
    
    async def _get_team_member(self, team_id: UUID, user_id: str) -> Optional[TeamMember]:
        """Get team member record"""
        members = self.team_members.get(team_id, [])
        for member in members:
            if member.user_id == user_id:
                return member
        return None
    
    
    async def _get_pending_invitation(self, team_id: UUID, user_id: str) -> Optional[TeamInvitation]:
        """Get pending invitation for user to team"""
        for invitation in self.pending_invitations.values():
            if (invitation.team_id == team_id and 
                invitation.invited_user == user_id and 
                invitation.status == "pending"):
                return invitation
        return None
    
    
    def _update_average_team_size(self):
        """Update average team size statistic"""
        if self.teams:
            total_members = sum(team.member_count for team in self.teams.values())
            self.team_stats["average_team_size"] = total_members / len(self.teams)
    
    
    async def get_service_statistics(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        # Calculate most popular team type
        type_counts = {}
        for team in self.teams.values():
            type_counts[team.team_type] = type_counts.get(team.team_type, 0) + 1
        
        if type_counts:
            self.team_stats["most_popular_team_type"] = max(type_counts, key=type_counts.get)
        
        return {
            **self.team_stats,
            "pending_invitations": len([inv for inv in self.pending_invitations.values() 
                                      if inv.status == "pending"]),
            "total_directory_entries": len(self.team_directory),
            "search_index_size": len(self.team_search_index),
            "team_type_distribution": type_counts
        }


# === Global Service Instance ===

_team_service_instance: Optional[TeamService] = None

def get_team_service() -> TeamService:
    """Get or create the global team service instance"""
    global _team_service_instance
    if _team_service_instance is None:
        _team_service_instance = TeamService()
    return _team_service_instance