"""
PRSM Teams Module

Collaborative team functionality for decentralized research coordination.
Enables shared resource access, token pooling, and collaborative AI development.
"""

from .models import (
    Team, TeamMember, TeamWallet, TeamTask, TeamGovernance,
    TeamRole, TeamMembershipStatus, TeamType, GovernanceModel
)
from .service import TeamService
from .wallet import TeamWalletService
from .governance import TeamGovernanceService

__all__ = [
    "Team", "TeamMember", "TeamWallet", "TeamTask", "TeamGovernance",
    "TeamRole", "TeamMembershipStatus", "TeamType", "GovernanceModel",
    "TeamService", "TeamWalletService", "TeamGovernanceService"
]