"""
PRSM Non-Human Identity (NHI) Framework
=======================================

Implements verifiable credentials and on-chain reputation for autonomous agents.
Allows the network to distinguish between trusted agents and malicious bots.
"""

import logging
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4

from prsm.core.cryptography.post_quantum import get_post_quantum_crypto, PostQuantumKeyPair

logger = logging.getLogger(__name__)

@dataclass
class NonHumanIdentity:
    """Verifiable credential for an autonomous agent"""
    agent_id: str
    owner_id: str
    agent_type: str # 'orchestrator', 'researcher', 'validator'
    reputation_score: float = 1.0
    staked_tokens: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    pq_public_key: Optional[bytes] = None
    
    def calculate_staking_requirement(self, base_requirement: float) -> float:
        """
        Meritocracy of Machines: High reputation agents require less stake.
        """
        # Minimum requirement is 20% of base if reputation is very high (e.g. 5.0)
        multiplier = max(0.2, 1.0 / (self.reputation_score ** 0.5))
        return base_requirement * multiplier

class NHIManager:
    """
    Manages NHI lifecycle and reputation updates.
    """
    def __init__(self):
        self.agents: Dict[str, NonHumanIdentity] = {}
        self.pq = get_post_quantum_crypto()

    def register_agent(self, owner_id: str, agent_type: str) -> NonHumanIdentity:
        agent_id = f"nhi_{uuid4().hex[:8]}"
        
        # Generate PQ keys for the agent
        keypair = self.pq.generate_keypair()
        
        agent = NonHumanIdentity(
            agent_id=agent_id,
            owner_id=owner_id,
            agent_type=agent_type,
            pq_public_key=keypair.public_key
        )
        
        self.agents[agent_id] = agent
        logger.info(f"Registered NHI: {agent_id} for owner {owner_id}")
        return agent, keypair

    def update_reputation(self, agent_id: str, breakthrough_level: int):
        """
        Increases reputation based on breakthrough value (verified by Expert Oracle).
        """
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            bonus = breakthrough_level * 0.1
            agent.reputation_score += bonus
            logger.info(f"NHI {agent_id} reputation increased to {agent.reputation_score:.2f}")

    def get_agent_credential(self, agent_id: str) -> Dict[str, Any]:
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            return {
                "agent_id": agent.agent_id,
                "reputation": agent.reputation_score,
                "type": agent.agent_type,
                "verified": True
            }
        return {"verified": False}
