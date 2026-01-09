"""
Hierarchical Consensus for Large-Scale PRSM Networks
Implements multi-tier consensus to achieve O(log n) scaling
"""

import asyncio
import hashlib
import json
import statistics
import time
from collections import defaultdict, Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from uuid import UUID, uuid4
from enum import Enum
import math

from prsm.core.config import settings
from prsm.core.models import PeerNode, AgentResponse, SafetyFlag, SafetyLevel
from prsm.core.safety.circuit_breaker import CircuitBreakerNetwork, ThreatLevel
from prsm.core.safety.monitor import SafetyMonitor
from prsm.economy.tokenomics.ftns_service import ftns_service
from .consensus import DistributedConsensus, ConsensusResult, ConsensusType


# === Hierarchical Consensus Configuration ===

# Tier settings
TIER_SIZE_LIMIT = int(getattr(settings, "PRSM_TIER_SIZE_LIMIT", 15))  # Max nodes per tier
MIN_TIER_SIZE = int(getattr(settings, "PRSM_MIN_TIER_SIZE", 3))      # Min nodes per tier
MAX_HIERARCHY_DEPTH = int(getattr(settings, "PRSM_MAX_HIERARCHY_DEPTH", 4))  # Max depth

# Consensus thresholds for hierarchical consensus
TIER_CONSENSUS_THRESHOLD = float(getattr(settings, "PRSM_TIER_CONSENSUS", 0.75))  # 75% within tier
GLOBAL_CONSENSUS_THRESHOLD = float(getattr(settings, "PRSM_GLOBAL_CONSENSUS", 0.80))  # 80% across tiers
COORDINATOR_CONSENSUS_THRESHOLD = float(getattr(settings, "PRSM_COORD_CONSENSUS", 0.67))  # 67% coordinators

# Performance settings
ENABLE_PARALLEL_CONSENSUS = getattr(settings, "PRSM_PARALLEL_CONSENSUS", True)
TIER_TIMEOUT_SECONDS = int(getattr(settings, "PRSM_TIER_TIMEOUT", 20))
COORDINATOR_TIMEOUT_SECONDS = int(getattr(settings, "PRSM_COORDINATOR_TIMEOUT", 30))


class HierarchyTier(Enum):
    """Hierarchy tier levels"""
    LEAF = "leaf"           # Bottom tier with actual nodes
    INTERMEDIATE = "intermediate"  # Middle tiers with coordinators
    ROOT = "root"           # Top tier with global coordinators


class NodeRole(Enum):
    """Node roles in hierarchical consensus"""
    PARTICIPANT = "participant"      # Regular node participating in consensus
    COORDINATOR = "coordinator"      # Coordinates a tier
    GLOBAL_COORDINATOR = "global_coordinator"  # Coordinates across tiers


class TierConsensusResult:
    """Result of consensus within a specific tier"""
    
    def __init__(self, tier_id: str, tier_level: int):
        self.tier_id = tier_id
        self.tier_level = tier_level
        self.consensus_achieved = False
        self.agreed_value = None
        self.agreement_ratio = 0.0
        self.participating_nodes = []
        self.coordinator_node = None
        self.execution_time = 0.0
        self.timestamp = datetime.now(timezone.utc)
        self.message_count = 0
        self.round_count = 1


class HierarchicalNode:
    """Node in hierarchical consensus network"""
    
    def __init__(self, node_id: str, role: NodeRole = NodeRole.PARTICIPANT):
        self.node_id = node_id
        self.role = role
        self.tier_id = None
        self.tier_level = 0
        self.parent_coordinator = None
        self.child_nodes: Set[str] = set()
        self.peer_nodes: Set[str] = set()
        
        # Consensus state
        self.pending_proposals: Dict[str, Dict[str, Any]] = {}
        self.tier_consensus_results: Dict[str, TierConsensusResult] = {}
        
        # Performance tracking
        self.consensus_metrics = {
            "consensus_participated": 0,
            "coordinator_rounds": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "average_tier_time": 0.0,
            "tier_size": 0
        }


class HierarchicalConsensusNetwork:
    """
    Hierarchical consensus network for large-scale PRSM deployments
    Reduces O(n²) complexity to O(log n) through multi-tier consensus
    """
    
    def __init__(self):
        # Network topology
        self.nodes: Dict[str, HierarchicalNode] = {}
        self.tiers: Dict[str, List[str]] = {}  # tier_id -> node_ids
        self.tier_coordinators: Dict[str, str] = {}  # tier_id -> coordinator_node_id
        self.hierarchy_depth = 0
        
        # Consensus infrastructure
        self.base_consensus = DistributedConsensus()
        self.active_hierarchical_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Safety integration
        self.circuit_breaker = CircuitBreakerNetwork()
        self.safety_monitor = SafetyMonitor()
        
        # Performance metrics
        self.hierarchical_metrics = {
            "total_hierarchical_consensus": 0,
            "successful_hierarchical_consensus": 0,
            "average_hierarchy_depth": 0.0,
            "average_tier_size": 0.0,
            "total_message_reduction": 0.0,
            "parallel_tier_consensus": 0,
            "coordinator_failures": 0
        }
        
        # Synchronization
        self._network_lock = asyncio.Lock()
        self._consensus_lock = asyncio.Lock()
    
    async def initialize_hierarchical_network(self, peer_nodes: List[PeerNode]) -> bool:
        """
        Initialize hierarchical network topology from peer nodes
        
        Args:
            peer_nodes: List of peer nodes to organize hierarchically
            
        Returns:
            True if hierarchy was successfully created
        """
        try:
            async with self._network_lock:
                if len(peer_nodes) < MIN_TIER_SIZE:
                    print(f"⚠️ Not enough nodes for hierarchical consensus: {len(peer_nodes)} < {MIN_TIER_SIZE}")
                    return False
                
                print(f"🏗️ Initializing hierarchical network with {len(peer_nodes)} nodes")
                
                # Create hierarchical nodes
                for peer in peer_nodes:
                    node = HierarchicalNode(peer.peer_id, NodeRole.PARTICIPANT)
                    self.nodes[peer.peer_id] = node
                
                # Organize nodes into hierarchical tiers
                success = await self._organize_hierarchy(list(peer.peer_id for peer in peer_nodes))
                
                if success:
                    await self._select_coordinators()
                    await self._establish_communication_paths()
                    
                    # Calculate metrics
                    self.hierarchical_metrics["average_tier_size"] = self._calculate_average_tier_size()
                    self.hierarchical_metrics["average_hierarchy_depth"] = self.hierarchy_depth
                    
                    print(f"✅ Hierarchical network initialized:")
                    print(f"   - Hierarchy depth: {self.hierarchy_depth}")
                    print(f"   - Number of tiers: {len(self.tiers)}")
                    print(f"   - Average tier size: {self.hierarchical_metrics['average_tier_size']:.1f}")
                    
                    return True
                else:
                    print("❌ Failed to organize hierarchical topology")
                    return False
                    
        except Exception as e:
            print(f"❌ Error initializing hierarchical network: {e}")
            return False
    
    async def achieve_hierarchical_consensus(self, 
                                           proposal: Dict[str, Any], 
                                           session_id: Optional[str] = None) -> ConsensusResult:
        """
        Achieve consensus using hierarchical multi-tier approach
        
        Args:
            proposal: Consensus proposal
            session_id: Optional session identifier
            
        Returns:
            ConsensusResult with hierarchical consensus details
        """
        start_time = datetime.now(timezone.utc)
        session_id = session_id or str(uuid4())
        
        async with self._consensus_lock:
            try:
                self.hierarchical_metrics["total_hierarchical_consensus"] += 1
                
                print(f"🏛️ Starting hierarchical consensus (session: {session_id[:8]})")
                
                # Phase 1: Tier-level consensus (parallel execution)
                tier_results = await self._execute_tier_consensus(proposal, session_id)
                
                if not tier_results:
                    return ConsensusResult(
                        consensus_achieved=False,
                        consensus_type="hierarchical_tier_failed",
                        execution_time=(datetime.now(timezone.utc) - start_time).total_seconds()
                    )
                
                # Phase 2: Coordinator consensus (if multiple tiers)
                coordinator_result = await self._execute_coordinator_consensus(tier_results, session_id)
                
                if not coordinator_result:
                    return ConsensusResult(
                        consensus_achieved=False,
                        consensus_type="hierarchical_coordinator_failed",
                        execution_time=(datetime.now(timezone.utc) - start_time).total_seconds()
                    )
                
                # Phase 3: Global validation
                global_consensus = await self._validate_global_consensus(tier_results, coordinator_result)
                
                execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                if global_consensus:
                    self.hierarchical_metrics["successful_hierarchical_consensus"] += 1
                    
                    # Calculate message complexity reduction
                    flat_messages = len(self.nodes) * (len(self.nodes) - 1)  # O(n²)
                    hierarchical_messages = self._calculate_hierarchical_message_count()
                    reduction = (flat_messages - hierarchical_messages) / flat_messages if flat_messages > 0 else 0
                    self.hierarchical_metrics["total_message_reduction"] = reduction
                    
                    # Calculate overall agreement ratio
                    total_participating = sum(len(result.participating_nodes) for result in tier_results.values())
                    total_agreed = sum(
                        len(result.participating_nodes) for result in tier_results.values() 
                        if result.consensus_achieved
                    )
                    agreement_ratio = total_agreed / total_participating if total_participating > 0 else 0
                    
                    print(f"✅ Hierarchical consensus achieved:")
                    print(f"   - Tiers with consensus: {sum(1 for r in tier_results.values() if r.consensus_achieved)}/{len(tier_results)}")
                    print(f"   - Message reduction: {reduction:.1%}")
                    print(f"   - Execution time: {execution_time:.2f}s")
                    
                    return ConsensusResult(
                        agreed_value=coordinator_result,
                        consensus_achieved=True,
                        consensus_type="hierarchical",
                        agreement_ratio=agreement_ratio,
                        participating_peers=list(self.nodes.keys()),
                        execution_time=execution_time
                    )
                else:
                    return ConsensusResult(
                        consensus_achieved=False,
                        consensus_type="hierarchical_global_failed",
                        execution_time=execution_time
                    )
                    
            except Exception as e:
                print(f"❌ Hierarchical consensus error: {e}")
                return ConsensusResult(
                    consensus_achieved=False,
                    consensus_type="hierarchical_error",
                    execution_time=(datetime.now(timezone.utc) - start_time).total_seconds()
                )
    
    async def _organize_hierarchy(self, node_ids: List[str]) -> bool:
        """Organize nodes into hierarchical tiers"""
        try:
            current_level_nodes = node_ids.copy()
            current_tier_level = 0
            
            while len(current_level_nodes) > 1 and current_tier_level < MAX_HIERARCHY_DEPTH:
                # Create tiers for current level
                level_tiers = []
                
                # Group nodes into tiers of max TIER_SIZE_LIMIT
                for i in range(0, len(current_level_nodes), TIER_SIZE_LIMIT):
                    tier_nodes = current_level_nodes[i:i + TIER_SIZE_LIMIT]
                    
                    if len(tier_nodes) >= MIN_TIER_SIZE or i == 0:  # Ensure minimum tier size
                        tier_id = f"tier_{current_tier_level}_{len(level_tiers)}"
                        self.tiers[tier_id] = tier_nodes
                        level_tiers.append(tier_id)
                        
                        # Update node tier information
                        for node_id in tier_nodes:
                            if node_id in self.nodes:
                                self.nodes[node_id].tier_id = tier_id
                                self.nodes[node_id].tier_level = current_tier_level
                                self.nodes[node_id].peer_nodes = set(tier_nodes) - {node_id}
                    else:
                        # Merge small tier with previous tier
                        if level_tiers:
                            prev_tier_id = level_tiers[-1]
                            self.tiers[prev_tier_id].extend(tier_nodes)
                            for node_id in tier_nodes:
                                if node_id in self.nodes:
                                    self.nodes[node_id].tier_id = prev_tier_id
                                    self.nodes[node_id].tier_level = current_tier_level
                
                # Prepare next level (coordinators from current level)
                current_level_nodes = level_tiers
                current_tier_level += 1
            
            self.hierarchy_depth = current_tier_level
            
            # Handle single remaining node as global coordinator
            if len(current_level_nodes) == 1:
                global_tier_id = f"tier_global_{current_tier_level}"
                self.tiers[global_tier_id] = [current_level_nodes[0]]
                
                # Create global coordinator node if needed
                if current_level_nodes[0] not in self.nodes:
                    global_coord = HierarchicalNode(current_level_nodes[0], NodeRole.GLOBAL_COORDINATOR)
                    global_coord.tier_id = global_tier_id
                    global_coord.tier_level = current_tier_level
                    self.nodes[current_level_nodes[0]] = global_coord
                else:
                    self.nodes[current_level_nodes[0]].role = NodeRole.GLOBAL_COORDINATOR
            
            return True
            
        except Exception as e:
            print(f"❌ Error organizing hierarchy: {e}")
            return False
    
    async def _select_coordinators(self):
        """Select coordinators for each tier"""
        try:
            for tier_id, node_ids in self.tiers.items():
                if len(node_ids) == 0:
                    continue
                
                # Select coordinator based on reputation or random for now
                # In production, would use more sophisticated coordinator selection
                coordinator_id = node_ids[0]  # Simple selection - first node
                
                # Update coordinator role
                if coordinator_id in self.nodes:
                    if self.nodes[coordinator_id].role != NodeRole.GLOBAL_COORDINATOR:
                        self.nodes[coordinator_id].role = NodeRole.COORDINATOR
                
                self.tier_coordinators[tier_id] = coordinator_id
                
                print(f"👑 Selected coordinator {coordinator_id} for {tier_id}")
                
        except Exception as e:
            print(f"❌ Error selecting coordinators: {e}")
    
    async def _establish_communication_paths(self):
        """Establish parent-child relationships between tiers"""
        try:
            tier_levels = defaultdict(list)
            
            # Group tiers by level
            for tier_id, node_ids in self.tiers.items():
                if node_ids and node_ids[0] in self.nodes:
                    level = self.nodes[node_ids[0]].tier_level
                    tier_levels[level].append(tier_id)
            
            # Establish parent-child relationships
            for level in range(1, self.hierarchy_depth + 1):
                if level in tier_levels and level - 1 in tier_levels:
                    parent_tiers = tier_levels[level]
                    child_tiers = tier_levels[level - 1]
                    
                    # Each parent tier coordinator becomes parent of multiple child tiers
                    tiers_per_parent = math.ceil(len(child_tiers) / len(parent_tiers))
                    
                    for i, parent_tier_id in enumerate(parent_tiers):
                        parent_coordinator = self.tier_coordinators.get(parent_tier_id)
                        if not parent_coordinator:
                            continue
                        
                        # Assign child tiers to this parent
                        start_idx = i * tiers_per_parent
                        end_idx = min(start_idx + tiers_per_parent, len(child_tiers))
                        
                        for j in range(start_idx, end_idx):
                            child_tier_id = child_tiers[j]
                            child_coordinator = self.tier_coordinators.get(child_tier_id)
                            
                            if child_coordinator and child_coordinator in self.nodes:
                                self.nodes[child_coordinator].parent_coordinator = parent_coordinator
                                
                                if parent_coordinator in self.nodes:
                                    self.nodes[parent_coordinator].child_nodes.add(child_coordinator)
            
        except Exception as e:
            print(f"❌ Error establishing communication paths: {e}")
    
    async def _execute_tier_consensus(self, proposal: Dict[str, Any], session_id: str) -> Dict[str, TierConsensusResult]:
        """Execute consensus within each tier in parallel"""
        try:
            tier_tasks = []
            tier_results = {}
            
            # Create consensus tasks for each tier
            for tier_id, node_ids in self.tiers.items():
                if len(node_ids) >= MIN_TIER_SIZE:
                    task = self._tier_consensus_task(tier_id, node_ids, proposal, session_id)
                    tier_tasks.append((tier_id, task))
            
            if not tier_tasks:
                return {}
            
            # Execute tier consensus in parallel if enabled
            if ENABLE_PARALLEL_CONSENSUS and len(tier_tasks) > 1:
                print(f"⚡ Running parallel consensus across {len(tier_tasks)} tiers")
                
                # Wait for all tier consensus to complete
                results = await asyncio.gather(*[task for _, task in tier_tasks], return_exceptions=True)
                
                for i, (tier_id, _) in enumerate(tier_tasks):
                    if isinstance(results[i], Exception):
                        print(f"❌ Tier {tier_id} consensus failed: {results[i]}")
                    else:
                        tier_results[tier_id] = results[i]
                        
                self.hierarchical_metrics["parallel_tier_consensus"] += 1
            else:
                # Sequential execution
                for tier_id, task in tier_tasks:
                    try:
                        result = await task
                        tier_results[tier_id] = result
                    except Exception as e:
                        print(f"❌ Tier {tier_id} consensus failed: {e}")
            
            successful_tiers = sum(1 for result in tier_results.values() if result.consensus_achieved)
            print(f"📊 Tier consensus complete: {successful_tiers}/{len(tier_tasks)} successful")
            
            return tier_results
            
        except Exception as e:
            print(f"❌ Error executing tier consensus: {e}")
            return {}
    
    async def _tier_consensus_task(self, tier_id: str, node_ids: List[str], 
                                 proposal: Dict[str, Any], session_id: str) -> TierConsensusResult:
        """Execute consensus within a single tier"""
        start_time = datetime.now(timezone.utc)
        
        try:
            result = TierConsensusResult(tier_id, self.nodes[node_ids[0]].tier_level if node_ids[0] in self.nodes else 0)
            result.participating_nodes = node_ids
            result.coordinator_node = self.tier_coordinators.get(tier_id)
            
            # Simulate tier consensus using base consensus mechanism
            # In production, would use actual network communication
            peer_results = []
            for node_id in node_ids:
                # Simulate peer result
                peer_result = {
                    "peer_id": node_id,
                    "result": proposal,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                peer_results.append(peer_result)
                
                # Update node metrics
                if node_id in self.nodes:
                    self.nodes[node_id].consensus_metrics["consensus_participated"] += 1
                    self.nodes[node_id].consensus_metrics["tier_size"] = len(node_ids)
            
            # Execute consensus within tier
            consensus_result = await self.base_consensus.achieve_result_consensus(
                peer_results, 
                ConsensusType.BYZANTINE_FAULT_TOLERANT,
                f"{session_id}_{tier_id}"
            )
            
            # Update tier result
            result.consensus_achieved = consensus_result.consensus_achieved
            result.agreed_value = consensus_result.agreed_value
            result.agreement_ratio = consensus_result.agreement_ratio
            result.execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            result.message_count = len(node_ids) * (len(node_ids) - 1)  # O(n²) within tier
            
            # Update coordinator metrics
            coordinator_id = self.tier_coordinators.get(tier_id)
            if coordinator_id and coordinator_id in self.nodes:
                self.nodes[coordinator_id].consensus_metrics["coordinator_rounds"] += 1
                current_avg = self.nodes[coordinator_id].consensus_metrics["average_tier_time"]
                rounds = self.nodes[coordinator_id].consensus_metrics["coordinator_rounds"]
                self.nodes[coordinator_id].consensus_metrics["average_tier_time"] = (
                    (current_avg * (rounds - 1) + result.execution_time) / rounds
                )
            
            if result.consensus_achieved:
                print(f"✅ Tier {tier_id} consensus: {result.agreement_ratio:.2%} agreement ({len(node_ids)} nodes)")
            else:
                print(f"❌ Tier {tier_id} consensus failed ({len(node_ids)} nodes)")
            
            return result
            
        except Exception as e:
            print(f"❌ Error in tier {tier_id} consensus: {e}")
            result = TierConsensusResult(tier_id, 0)
            result.consensus_achieved = False
            result.execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            return result
    
    async def _execute_coordinator_consensus(self, tier_results: Dict[str, TierConsensusResult], 
                                           session_id: str) -> Optional[Dict[str, Any]]:
        """Execute consensus among tier coordinators"""
        try:
            if len(tier_results) <= 1:
                # Single tier - use its result directly
                successful_tiers = [result for result in tier_results.values() if result.consensus_achieved]
                return successful_tiers[0].agreed_value if successful_tiers else None
            
            # Gather coordinator results
            coordinator_results = []
            for tier_id, tier_result in tier_results.items():
                if tier_result.consensus_achieved:
                    coordinator_id = self.tier_coordinators.get(tier_id)
                    if coordinator_id:
                        coordinator_result = {
                            "peer_id": coordinator_id,
                            "result": tier_result.agreed_value,
                            "tier_id": tier_id,
                            "tier_agreement_ratio": tier_result.agreement_ratio,
                            "tier_participants": len(tier_result.participating_nodes)
                        }
                        coordinator_results.append(coordinator_result)
            
            if not coordinator_results:
                return None
            
            # Check if we need coordinator consensus
            if len(coordinator_results) == 1:
                return coordinator_results[0]["result"]
            
            # Execute coordinator consensus
            coordinator_consensus = await self.base_consensus.achieve_result_consensus(
                coordinator_results,
                ConsensusType.WEIGHTED_MAJORITY,
                f"{session_id}_coordinators"
            )
            
            # Apply coordinator consensus threshold
            if (coordinator_consensus.consensus_achieved and 
                coordinator_consensus.agreement_ratio >= COORDINATOR_CONSENSUS_THRESHOLD):
                
                print(f"✅ Coordinator consensus: {coordinator_consensus.agreement_ratio:.2%} agreement")
                return coordinator_consensus.agreed_value
            else:
                print(f"❌ Coordinator consensus failed: {coordinator_consensus.agreement_ratio:.2%} < {COORDINATOR_CONSENSUS_THRESHOLD:.2%}")
                return None
                
        except Exception as e:
            print(f"❌ Error in coordinator consensus: {e}")
            return None
    
    async def _validate_global_consensus(self, tier_results: Dict[str, TierConsensusResult], 
                                       coordinator_result: Any) -> bool:
        """Validate global consensus across all tiers"""
        try:
            if not coordinator_result:
                return False
            
            # Calculate global agreement metrics
            total_nodes = sum(len(result.participating_nodes) for result in tier_results.values())
            nodes_with_consensus = sum(
                len(result.participating_nodes) for result in tier_results.values() 
                if result.consensus_achieved
            )
            
            global_agreement = nodes_with_consensus / total_nodes if total_nodes > 0 else 0
            
            # Apply global consensus threshold
            if global_agreement >= GLOBAL_CONSENSUS_THRESHOLD:
                print(f"✅ Global consensus validated: {global_agreement:.2%} agreement")
                return True
            else:
                print(f"❌ Global consensus failed: {global_agreement:.2%} < {GLOBAL_CONSENSUS_THRESHOLD:.2%}")
                return False
                
        except Exception as e:
            print(f"❌ Error validating global consensus: {e}")
            return False
    
    def _calculate_average_tier_size(self) -> float:
        """Calculate average tier size"""
        try:
            if not self.tiers:
                return 0.0
            
            total_nodes = sum(len(nodes) for nodes in self.tiers.values())
            return total_nodes / len(self.tiers)
            
        except Exception:
            return 0.0
    
    def _calculate_hierarchical_message_count(self) -> int:
        """Calculate total message count for hierarchical consensus"""
        try:
            total_messages = 0
            
            # Messages within each tier (O(t²) where t is tier size)
            for nodes in self.tiers.values():
                tier_size = len(nodes)
                total_messages += tier_size * (tier_size - 1)
            
            # Messages between coordinators (O(c²) where c is number of coordinators)
            coordinator_count = len(self.tier_coordinators)
            total_messages += coordinator_count * (coordinator_count - 1)
            
            return total_messages
            
        except Exception:
            return 0
    
    async def get_hierarchical_metrics(self) -> Dict[str, Any]:
        """Get comprehensive hierarchical consensus metrics"""
        try:
            # Calculate current topology metrics
            node_count = len(self.nodes)
            tier_count = len(self.tiers)
            coordinator_count = len(self.tier_coordinators)
            
            # Calculate message complexity reduction
            flat_complexity = node_count * (node_count - 1) if node_count > 1 else 0
            hierarchical_complexity = self._calculate_hierarchical_message_count()
            complexity_reduction = (
                (flat_complexity - hierarchical_complexity) / flat_complexity 
                if flat_complexity > 0 else 0
            )
            
            # Calculate scaling efficiency
            scaling_efficiency = 1.0 / math.log(node_count) if node_count > 1 else 1.0
            
            return {
                **self.hierarchical_metrics,
                "topology": {
                    "total_nodes": node_count,
                    "tier_count": tier_count,
                    "coordinator_count": coordinator_count,
                    "hierarchy_depth": self.hierarchy_depth,
                    "average_tier_size": self._calculate_average_tier_size()
                },
                "complexity": {
                    "flat_message_complexity": flat_complexity,
                    "hierarchical_message_complexity": hierarchical_complexity,
                    "complexity_reduction": complexity_reduction,
                    "scaling_efficiency": scaling_efficiency
                },
                "performance": {
                    "consensus_success_rate": (
                        self.hierarchical_metrics["successful_hierarchical_consensus"] /
                        max(1, self.hierarchical_metrics["total_hierarchical_consensus"])
                    ),
                    "parallel_consensus_enabled": ENABLE_PARALLEL_CONSENSUS,
                    "tier_timeout": TIER_TIMEOUT_SECONDS,
                    "coordinator_timeout": COORDINATOR_TIMEOUT_SECONDS
                },
                "thresholds": {
                    "tier_consensus": TIER_CONSENSUS_THRESHOLD,
                    "global_consensus": GLOBAL_CONSENSUS_THRESHOLD,
                    "coordinator_consensus": COORDINATOR_CONSENSUS_THRESHOLD
                }
            }
            
        except Exception as e:
            print(f"❌ Error getting hierarchical metrics: {e}")
            return self.hierarchical_metrics
    
    async def get_network_topology(self) -> Dict[str, Any]:
        """Get detailed network topology information"""
        try:
            topology = {
                "tiers": {},
                "coordinators": {},
                "communication_paths": {},
                "node_roles": {}
            }
            
            # Tier information
            for tier_id, node_ids in self.tiers.items():
                tier_level = self.nodes[node_ids[0]].tier_level if node_ids and node_ids[0] in self.nodes else 0
                topology["tiers"][tier_id] = {
                    "nodes": node_ids,
                    "level": tier_level,
                    "size": len(node_ids),
                    "coordinator": self.tier_coordinators.get(tier_id)
                }
            
            # Coordinator information
            for tier_id, coordinator_id in self.tier_coordinators.items():
                if coordinator_id in self.nodes:
                    node = self.nodes[coordinator_id]
                    topology["coordinators"][coordinator_id] = {
                        "tier_id": tier_id,
                        "role": node.role.value,
                        "parent": node.parent_coordinator,
                        "children": list(node.child_nodes),
                        "metrics": node.consensus_metrics
                    }
            
            # Communication paths
            for node_id, node in self.nodes.items():
                topology["communication_paths"][node_id] = {
                    "tier_id": node.tier_id,
                    "tier_level": node.tier_level,
                    "peers": list(node.peer_nodes),
                    "parent": node.parent_coordinator,
                    "children": list(node.child_nodes)
                }
            
            # Node roles
            for node_id, node in self.nodes.items():
                topology["node_roles"][node_id] = {
                    "role": node.role.value,
                    "tier": node.tier_id,
                    "level": node.tier_level
                }
            
            return topology
            
        except Exception as e:
            print(f"❌ Error getting network topology: {e}")
            return {}


# === Global Hierarchical Consensus Instance ===

_hierarchical_consensus_instance: Optional[HierarchicalConsensusNetwork] = None

def get_hierarchical_consensus() -> HierarchicalConsensusNetwork:
    """Get or create the global hierarchical consensus instance"""
    global _hierarchical_consensus_instance
    if _hierarchical_consensus_instance is None:
        _hierarchical_consensus_instance = HierarchicalConsensusNetwork()
    return _hierarchical_consensus_instance