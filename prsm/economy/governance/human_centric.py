"""
PRSM Human-Centric Governance
=============================

Implements:
1. Expert Oracle Layer (HITL): Human validation for high-stakes research.
2. Liquid Science Delegation: Expertise-based voting power delegation.
"""

import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class ExpertiseDomain(str, Enum):
    AI_ARCHITECTURE = "ai_architecture"
    BIOLOGY = "biology"
    PHYSICS = "physics"
    ETHICS = "ethics"
    CLIMATE = "climate"

@dataclass
class ExpertProfile:
    user_id: str
    domains: List[ExpertiseDomain]
    reputation_score: float = 1.0
    staked_tokens: float = 0.0

@dataclass
class ValidationRequest:
    request_id: UUID
    target_cid: str # Content ID of the model/result
    domain: ExpertiseDomain
    required_approvals: int = 3
    approvals: List[str] = field(default_factory=list)
    rejections: List[str] = field(default_factory=list)
    status: str = "pending" # pending, approved, rejected

class ExpertOracle:
    """
    Expert Oracle Layer (HITL).
    Verified humans provide qualitative 'stamp of approval' for research.
    """
    def __init__(self):
        self.experts: Dict[str, ExpertProfile] = {}
        self.requests: Dict[UUID, ValidationRequest] = {}

    def register_expert(self, user_id: str, domains: List[ExpertiseDomain], stake: float):
        self.experts[user_id] = ExpertProfile(user_id=user_id, domains=domains, staked_tokens=stake)
        logger.info(f"Expert {user_id} registered in domains: {[d.value for d in domains]}")

    def create_validation_request(self, target_cid: str, domain: ExpertiseDomain) -> UUID:
        request_id = uuid4()
        self.requests[request_id] = ValidationRequest(request_id=request_id, target_cid=target_cid, domain=domain)
        return request_id

    def submit_validation(self, expert_id: str, request_id: UUID, approved: bool):
        if expert_id not in self.experts or request_id not in self.requests:
            return False
        
        expert = self.experts[expert_id]
        request = self.requests[request_id]
        
        if request.domain not in expert.domains:
            raise ValueError(f"Expert {expert_id} is not qualified for domain {request.domain}")

        if approved:
            if expert_id not in request.approvals:
                request.approvals.append(expert_id)
        else:
            if expert_id not in request.rejections:
                request.rejections.append(expert_id)

        # Update Status
        if len(request.approvals) >= request.required_approvals:
            request.status = "approved"
        elif len(request.rejections) >= request.required_approvals:
            request.status = "rejected"
            
        return True

class LiquidScienceGovernance:
    """
    Liquid Democracy for Scientific Governance.
    Allows users to delegate voting power to domain experts.
    """
    def __init__(self):
        # Delegations: delegator -> domain -> delegate_id
        self.domain_delegations: Dict[str, Dict[ExpertiseDomain, str]] = {}

    def delegate_by_domain(self, delegator_id: str, delegate_id: str, domain: ExpertiseDomain):
        if delegator_id not in self.domain_delegations:
            self.domain_delegations[delegator_id] = {}
        self.domain_delegations[delegator_id][domain] = delegate_id
        logger.info(f"User {delegator_id} delegated {domain.value} power to {delegate_id}")

    def calculate_domain_power(self, user_id: str, domain: ExpertiseDomain, base_power: float, all_users_power: Dict[str, float]) -> float:
        """
        Calculates total power for a user in a specific domain, including incoming delegations.
        """
        total_power = base_power
        
        # Add power from people who delegated to this user for this domain
        for delegator, domains in self.domain_delegations.items():
            if domains.get(domain) == user_id:
                # Add the delegator's base power to the delegate
                total_power += all_users_power.get(delegator, 0.0)
                
        return total_power
