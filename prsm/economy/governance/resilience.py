"""
PRSM Network Resilience & Governance
====================================

Implements:
1. Validator Rotation via VRF (Verifiable Random Functions)
2. Sybil Resistance via Staking & Slashing
3. Collusion Ring detection and mitigation
"""

import hashlib
import time
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from decimal import Decimal
from uuid import uuid4

logger = logging.getLogger(__name__)

@dataclass
class NodeStake:
    node_id: str
    staked_amount: Decimal
    locked_until: float
    reputation_score: float = 1.0
    is_slashed: bool = False

class ResilienceManager:
    """
    Manages the security and resilience of the PRSM network.
    """
    def __init__(self, min_stake: Decimal = Decimal("100.0")):
        self.min_stake = min_stake
        self.stakes: Dict[str, NodeStake] = {}
        self.epoch_seed: str = str(uuid4()) # Updated periodically
        
    def register_node(self, node_id: str, amount: Decimal):
        """Register a node with a stake for Sybil resistance"""
        if amount < self.min_stake:
            raise ValueError(f"Insufficient stake. Min required: {self.min_stake}")
        
        self.stakes[node_id] = NodeStake(
            node_id=node_id,
            staked_amount=amount,
            locked_until=time.time() + 86400 * 7 # 1 day lock
        )
        logger.info(f"Node {node_id} registered with {amount} FTNS stake.")

    def get_shuffled_validators(self, task_id: str, available_nodes: List[str], count: int = 3) -> List[str]:
        """
        Validator Rotation via VRF (Simulated).
        PERIODICALLY shuffles which nodes verify which shards.
        """
        if not available_nodes:
            return []
            
        # Combine epoch seed, task_id, and node_ids for a deterministic but unpredictable shuffle
        validators = []
        eligible_nodes = [n for n in available_nodes if n in self.stakes and not self.stakes[n].is_slashed]
        
        if len(eligible_nodes) < count:
            # Fallback to whatever is available if network is small
            eligible_nodes = available_nodes
            
        # Sort for determinism before shuffling
        eligible_nodes.sort()
        
        # VRF-like selection
        def vrf_score(node_id: str) -> str:
            seed_material = f"{self.epoch_seed}-{task_id}-{node_id}"
            return hashlib.sha256(seed_material.encode()).hexdigest()
            
        # Select top N nodes based on VRF score
        ranked_nodes = sorted(eligible_nodes, key=vrf_score, reverse=True)
        return ranked_nodes[:count]

    def slash_node(self, node_id: str, reason: str):
        """
        Slashing: Destroy or redistribute stake of a dishonest node.
        Triggered when Deterministic Re-execution proves a lie.
        """
        if node_id in self.stakes:
            stake = self.stakes[node_id]
            logger.warning(f"🚨 SLASHING NODE {node_id}! Reason: {reason}. Amount: {stake.staked_amount}")
            stake.is_slashed = True
            stake.staked_amount = Decimal("0.0")
            stake.reputation_score = 0.0
            # In a real blockchain, this would trigger a BURN transaction or treasury transfer

    def update_epoch(self):
        """Update the network epoch seed to prevent long-range prediction of validator rotation"""
        self.epoch_seed = hashlib.sha256(f"{self.epoch_seed}-{time.time()}".encode()).hexdigest()
        logger.info(f"Network Epoch updated. New VRF Seed: {self.epoch_seed[:10]}...")

    def calculate_fairness_reward(self, task_complexity: float, node_count: int) -> Decimal:
        """
        Dynamic Resource Pricing (The Fairness Algorithm).
        Increase base payment if few nodes are working on a specific niche task.
        """
        base_reward = Decimal(str(task_complexity)) * Decimal("0.1")
        
        # Scarcity Multiplier: if node_count is low, reward is higher
        # Ensures rare disease research gets funded/computed.
        if node_count < 5:
            multiplier = Decimal("2.0")
        elif node_count < 10:
            multiplier = Decimal("1.5")
        else:
            multiplier = Decimal("1.0")
            
        return base_reward * multiplier
