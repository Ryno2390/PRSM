"""
Consensus Sharding for PRSM
Implements parallel consensus across multiple shards for massive throughput scaling
"""

import asyncio
import hashlib
import json
import statistics
import time
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from uuid import UUID, uuid4
from enum import Enum
import math

from prsm.core.config import settings
from prsm.core.models import PeerNode, AgentResponse, SafetyFlag, SafetyLevel
from prsm.core.safety.circuit_breaker import CircuitBreakerNetwork, ThreatLevel
from prsm.core.safety.monitor import SafetyMonitor
from prsm.economy.tokenomics.ftns_service import get_ftns_service
from .consensus import DistributedConsensus, ConsensusResult, ConsensusType
from .hierarchical_consensus import HierarchicalConsensusNetwork
from .adaptive_consensus import AdaptiveConsensusEngine


# === Consensus Sharding Configuration ===

# Shard sizing parameters
MIN_SHARD_SIZE = int(getattr(settings, "PRSM_MIN_SHARD_SIZE", 3))
MAX_SHARD_SIZE = int(getattr(settings, "PRSM_MAX_SHARD_SIZE", 15))
OPTIMAL_SHARD_SIZE = int(getattr(settings, "PRSM_OPTIMAL_SHARD_SIZE", 10))
MAX_SHARDS = int(getattr(settings, "PRSM_MAX_SHARDS", 20))

# Cross-shard coordination
CROSS_SHARD_CONSENSUS_THRESHOLD = float(getattr(settings, "PRSM_CROSS_SHARD_THRESHOLD", 0.67))  # 67%
GLOBAL_COORDINATION_TIMEOUT = int(getattr(settings, "PRSM_GLOBAL_COORD_TIMEOUT", 60))  # seconds
SHARD_SYNCHRONIZATION_INTERVAL = int(getattr(settings, "PRSM_SHARD_SYNC_INTERVAL", 30))  # seconds

# Load balancing
LOAD_BALANCE_THRESHOLD = float(getattr(settings, "PRSM_LOAD_BALANCE_THRESHOLD", 0.8))  # 80%
SHARD_SPLIT_THRESHOLD = int(getattr(settings, "PRSM_SHARD_SPLIT_THRESHOLD", 12))  # nodes
SHARD_MERGE_THRESHOLD = int(getattr(settings, "PRSM_SHARD_MERGE_THRESHOLD", 5))   # nodes

# Performance settings
ENABLE_DYNAMIC_SHARDING = getattr(settings, "PRSM_DYNAMIC_SHARDING", True)
ENABLE_CROSS_SHARD_VALIDATION = getattr(settings, "PRSM_CROSS_SHARD_VALIDATION", True)
PARALLEL_SHARD_CONSENSUS = getattr(settings, "PRSM_PARALLEL_SHARD_CONSENSUS", True)


class ShardState(Enum):
    """Shard operational states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    OVERLOADED = "overloaded"
    UNDERLOADED = "underloaded"
    SPLITTING = "splitting"
    MERGING = "merging"
    FAILED = "failed"
    RECOVERING = "recovering"


class ShardingStrategy(Enum):
    """Sharding strategies for different workloads"""
    HASH_BASED = "hash_based"           # Consistent hashing
    GEOGRAPHIC = "geographic"           # Geographic distribution
    WORKLOAD_BASED = "workload_based"   # Based on transaction types
    ADAPTIVE = "adaptive"               # Dynamic based on performance
    HYBRID = "hybrid"                   # Combination of strategies


class CrossShardOperation(Enum):
    """Types of cross-shard operations"""
    COORDINATE = "coordinate"           # Cross-shard coordination
    VALIDATE = "validate"               # Cross-shard validation
    SYNCHRONIZE = "synchronize"         # State synchronization
    REBALANCE = "rebalance"            # Load rebalancing
    MERGE = "merge"                    # Shard merging
    SPLIT = "split"                    # Shard splitting


class ConsensusShard:
    """Individual consensus shard with dedicated consensus engine"""
    
    def __init__(self, shard_id: str, initial_nodes: List[PeerNode]):
        self.shard_id = shard_id
        self.nodes: Dict[str, PeerNode] = {node.peer_id: node for node in initial_nodes}
        self.state = ShardState.INITIALIZING
        
        # Consensus engines
        self.adaptive_consensus = AdaptiveConsensusEngine()
        self.hierarchical_consensus = HierarchicalConsensusNetwork()
        
        # Shard performance metrics
        self.throughput_samples = deque(maxlen=100)
        self.latency_samples = deque(maxlen=100)
        self.consensus_history = deque(maxlen=200)
        self.load_factor = 0.0
        
        # Cross-shard coordination
        self.coordinator_node: Optional[str] = None
        self.pending_cross_shard_ops: Dict[str, Dict[str, Any]] = {}
        self.cross_shard_state: Dict[str, Any] = {}
        
        # Synchronization
        self.last_sync = datetime.now(timezone.utc)
        self.sync_lock = asyncio.Lock()
        
        # Safety integration
        self.circuit_breaker = CircuitBreakerNetwork()
        self.safety_monitor = SafetyMonitor()
    
    async def initialize_shard(self) -> bool:
        """Initialize shard consensus engines"""
        try:
            print(f"🔧 Initializing shard {self.shard_id} with {len(self.nodes)} nodes")
            
            # Initialize adaptive consensus
            node_list = list(self.nodes.values())
            success = await self.adaptive_consensus.initialize_adaptive_consensus(node_list)
            
            if not success:
                print(f"❌ Failed to initialize adaptive consensus for shard {self.shard_id}")
                self.state = ShardState.FAILED
                return False
            
            # Initialize hierarchical consensus for larger shards
            if len(self.nodes) > MAX_SHARD_SIZE // 2:
                h_success = await self.hierarchical_consensus.initialize_hierarchical_network(node_list)
                if not h_success:
                    print(f"⚠️ Hierarchical consensus initialization failed for shard {self.shard_id}")
            
            # Select coordinator node (highest reputation)
            if self.nodes:
                self.coordinator_node = max(self.nodes.keys(), 
                                          key=lambda nid: self.nodes[nid].reputation_score)
            
            self.state = ShardState.ACTIVE
            print(f"✅ Shard {self.shard_id} initialized with coordinator {self.coordinator_node}")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing shard {self.shard_id}: {e}")
            self.state = ShardState.FAILED
            return False
    
    async def achieve_shard_consensus(self, proposal: Dict[str, Any], session_id: str) -> ConsensusResult:
        """Achieve consensus within the shard"""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Check shard state
            if self.state not in [ShardState.ACTIVE, ShardState.OVERLOADED]:
                return ConsensusResult(
                    consensus_achieved=False,
                    consensus_type=f"shard_{self.state.value}",
                    execution_time=(datetime.now(timezone.utc) - start_time).total_seconds()
                )
            
            # Add shard context to proposal
            shard_proposal = {
                **proposal,
                "shard_id": self.shard_id,
                "shard_nodes": len(self.nodes),
                "coordinator": self.coordinator_node
            }
            
            # Use adaptive consensus for optimal strategy selection
            result = await self.adaptive_consensus.achieve_adaptive_consensus(shard_proposal, session_id)
            
            # Update shard metrics
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._update_shard_metrics(execution_time, result.consensus_achieved)
            
            # Store consensus result
            self.consensus_history.append({
                'session_id': session_id,
                'success': result.consensus_achieved,
                'execution_time': execution_time,
                'timestamp': datetime.now(timezone.utc),
                'strategy': result.consensus_type
            })
            
            return result
            
        except Exception as e:
            print(f"❌ Shard {self.shard_id} consensus error: {e}")
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            return ConsensusResult(
                consensus_achieved=False,
                consensus_type="shard_error",
                execution_time=execution_time
            )
    
    def _update_shard_metrics(self, execution_time: float, success: bool):
        """Update shard performance metrics"""
        try:
            # Update latency and throughput
            self.latency_samples.append(execution_time)
            
            # Calculate throughput (operations per second over last minute)
            now = datetime.now(timezone.utc)
            recent_consensus = [c for c in self.consensus_history 
                              if (now - c['timestamp']).total_seconds() < 60]
            throughput = len(recent_consensus) / 60.0  # ops per second
            self.throughput_samples.append(throughput)
            
            # Calculate load factor (0.0 to 1.0)
            avg_latency = statistics.mean(self.latency_samples) if self.latency_samples else 0
            avg_throughput = statistics.mean(self.throughput_samples) if self.throughput_samples else 0
            
            # Normalize load factor (higher latency and lower throughput = higher load)
            latency_factor = min(1.0, avg_latency / 2.0)  # 2s = 100% load
            throughput_factor = 1.0 - min(1.0, avg_throughput / 10.0)  # 10 ops/s = 0% load
            self.load_factor = (latency_factor + throughput_factor) / 2.0
            
            # Update shard state based on load
            if self.load_factor > LOAD_BALANCE_THRESHOLD:
                if len(self.nodes) >= SHARD_SPLIT_THRESHOLD:
                    self.state = ShardState.OVERLOADED
                else:
                    self.state = ShardState.ACTIVE  # Can't split further
            elif len(self.nodes) <= SHARD_MERGE_THRESHOLD and self.load_factor < 0.3:
                self.state = ShardState.UNDERLOADED
            else:
                self.state = ShardState.ACTIVE
                
        except Exception as e:
            print(f"❌ Error updating shard metrics for {self.shard_id}: {e}")
    
    async def add_node(self, node: PeerNode) -> bool:
        """Add a node to the shard"""
        try:
            async with self.sync_lock:
                if node.peer_id not in self.nodes:
                    self.nodes[node.peer_id] = node
                    print(f"📥 Added node {node.peer_id} to shard {self.shard_id}")
                    
                    # Re-initialize if needed
                    if self.state == ShardState.ACTIVE:
                        await self._reinitialize_consensus()
                    
                    return True
                return False
                
        except Exception as e:
            print(f"❌ Error adding node to shard {self.shard_id}: {e}")
            return False
    
    async def remove_node(self, node_id: str) -> bool:
        """Remove a node from the shard"""
        try:
            async with self.sync_lock:
                if node_id in self.nodes:
                    del self.nodes[node_id]
                    print(f"📤 Removed node {node_id} from shard {self.shard_id}")
                    
                    # Update coordinator if needed
                    if self.coordinator_node == node_id and self.nodes:
                        self.coordinator_node = max(self.nodes.keys(),
                                                  key=lambda nid: self.nodes[nid].reputation_score)
                    
                    # Re-initialize if needed
                    if self.state == ShardState.ACTIVE and len(self.nodes) >= MIN_SHARD_SIZE:
                        await self._reinitialize_consensus()
                    elif len(self.nodes) < MIN_SHARD_SIZE:
                        self.state = ShardState.UNDERLOADED
                    
                    return True
                return False
                
        except Exception as e:
            print(f"❌ Error removing node from shard {self.shard_id}: {e}")
            return False
    
    async def _reinitialize_consensus(self):
        """Reinitialize consensus engines with current nodes"""
        try:
            node_list = list(self.nodes.values())
            await self.adaptive_consensus.initialize_adaptive_consensus(node_list)
            
            if len(self.nodes) > MAX_SHARD_SIZE // 2:
                await self.hierarchical_consensus.initialize_hierarchical_network(node_list)
                
        except Exception as e:
            print(f"❌ Error reinitializing consensus for shard {self.shard_id}: {e}")
    
    def get_shard_metrics(self) -> Dict[str, Any]:
        """Get comprehensive shard metrics"""
        try:
            recent_consensus = list(self.consensus_history)[-50:]  # Last 50 consensus
            
            return {
                'shard_id': self.shard_id,
                'state': self.state.value,
                'node_count': len(self.nodes),
                'coordinator': self.coordinator_node,
                'load_factor': self.load_factor,
                'avg_latency': statistics.mean(self.latency_samples) if self.latency_samples else 0,
                'avg_throughput': statistics.mean(self.throughput_samples) if self.throughput_samples else 0,
                'consensus_success_rate': (
                    sum(1 for c in recent_consensus if c['success']) / max(1, len(recent_consensus))
                ),
                'recent_consensus_count': len(recent_consensus),
                'total_consensus_history': len(self.consensus_history)
            }
            
        except Exception as e:
            print(f"❌ Error getting shard metrics: {e}")
            return {'shard_id': self.shard_id, 'error': str(e)}


class ConsensusShardingManager:
    """
    Manages multiple consensus shards for massive throughput scaling
    Provides cross-shard coordination, load balancing, and dynamic shard management
    """
    
    def __init__(self, sharding_strategy: ShardingStrategy = ShardingStrategy.ADAPTIVE):
        # Shard management
        self.shards: Dict[str, ConsensusShard] = {}
        self.sharding_strategy = sharding_strategy
        self.global_consensus = AdaptiveConsensusEngine()
        
        # Cross-shard coordination
        self.cross_shard_coordinator: Optional[str] = None
        self.pending_cross_shard_operations: Dict[str, Dict[str, Any]] = {}
        self.global_state: Dict[str, Any] = {}
        
        # Load balancing
        self.load_balancer_active = False
        self.rebalancing_in_progress = False
        self.last_rebalance = datetime.now(timezone.utc)
        
        # Performance tracking
        self.sharding_metrics = {
            "total_shards": 0,
            "active_shards": 0,
            "total_throughput": 0.0,
            "average_shard_load": 0.0,
            "cross_shard_operations": 0,
            "successful_cross_shard_operations": 0,
            "shard_splits": 0,
            "shard_merges": 0,
            "nodes_per_shard": {},
            "shard_performance": {}
        }
        
        # Safety integration
        self.circuit_breaker = CircuitBreakerNetwork()
        self.safety_monitor = SafetyMonitor()
        
        # Synchronization
        self._sharding_lock = asyncio.Lock()
        self._rebalance_lock = asyncio.Lock()
    
    async def initialize_sharding(self, peer_nodes: List[PeerNode]) -> bool:
        """Initialize consensus sharding with peer nodes"""
        try:
            async with self._sharding_lock:
                print(f"🌐 Initializing consensus sharding with {len(peer_nodes)} nodes")
                
                if len(peer_nodes) < MIN_SHARD_SIZE:
                    print(f"⚠️ Not enough nodes for sharding: {len(peer_nodes)} < {MIN_SHARD_SIZE}")
                    return False
                
                # Create initial shards based on strategy
                success = await self._create_initial_shards(peer_nodes)
                
                if success:
                    # Initialize global consensus with shard coordinators
                    coordinators = [shard.coordinator_node for shard in self.shards.values() 
                                  if shard.coordinator_node]
                    if coordinators:
                        # Create coordinator peer nodes for global consensus
                        coordinator_peers = []
                        for coordinator_id in coordinators:
                            # Find coordinator in original peer list
                            coordinator_peer = next((p for p in peer_nodes if p.peer_id == coordinator_id), None)
                            if coordinator_peer:
                                coordinator_peers.append(coordinator_peer)
                        
                        if coordinator_peers:
                            await self.global_consensus.initialize_adaptive_consensus(coordinator_peers)
                            self.cross_shard_coordinator = coordinator_peers[0].peer_id
                    
                    # Start load balancing if enabled
                    if ENABLE_DYNAMIC_SHARDING:
                        self.load_balancer_active = True
                        asyncio.create_task(self._load_balancing_loop())
                    
                    print(f"✅ Consensus sharding initialized:")
                    print(f"   - Shards created: {len(self.shards)}")
                    print(f"   - Strategy: {self.sharding_strategy.value}")
                    print(f"   - Global coordinator: {self.cross_shard_coordinator}")
                    
                    return True
                else:
                    print("❌ Failed to create initial shards")
                    return False
                    
        except Exception as e:
            print(f"❌ Error initializing consensus sharding: {e}")
            return False
    
    async def achieve_sharded_consensus(self, proposal: Dict[str, Any], 
                                      session_id: Optional[str] = None) -> ConsensusResult:
        """Achieve consensus across shards"""
        start_time = datetime.now(timezone.utc)
        session_id = session_id or str(uuid4())
        
        try:
            print(f"🌐 Starting sharded consensus (session: {session_id[:8]})")
            
            # Determine target shard(s) for proposal
            target_shards = await self._select_target_shards(proposal)
            
            if not target_shards:
                return ConsensusResult(
                    consensus_achieved=False,
                    consensus_type="sharded_no_targets",
                    execution_time=(datetime.now(timezone.utc) - start_time).total_seconds()
                )
            
            # Execute consensus in parallel across target shards
            if PARALLEL_SHARD_CONSENSUS and len(target_shards) > 1:
                shard_results = await self._parallel_shard_consensus(target_shards, proposal, session_id)
            else:
                shard_results = await self._sequential_shard_consensus(target_shards, proposal, session_id)
            
            # Cross-shard coordination if multiple shards involved
            if len(target_shards) > 1 and ENABLE_CROSS_SHARD_VALIDATION:
                coordination_result = await self._cross_shard_coordination(shard_results, session_id)
            else:
                # Single shard or no cross-shard validation
                coordination_result = shard_results[0] if shard_results else None
            
            # Calculate final result
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            if coordination_result and coordination_result.consensus_achieved:
                # Calculate aggregate metrics
                successful_shards = sum(1 for r in shard_results if r.consensus_achieved)
                total_throughput = self._calculate_total_throughput()
                
                print(f"✅ Sharded consensus achieved:")
                print(f"   - Successful shards: {successful_shards}/{len(target_shards)}")
                print(f"   - Total throughput: {total_throughput:.1f} ops/s")
                print(f"   - Execution time: {execution_time:.2f}s")
                
                # Update sharding metrics
                self.sharding_metrics["cross_shard_operations"] += 1
                self.sharding_metrics["successful_cross_shard_operations"] += 1
                
                return ConsensusResult(
                    agreed_value=coordination_result.agreed_value,
                    consensus_achieved=True,
                    consensus_type="sharded_consensus",
                    agreement_ratio=coordination_result.agreement_ratio,
                    participating_peers=coordination_result.participating_peers,
                    execution_time=execution_time
                )
            else:
                print(f"❌ Sharded consensus failed")
                self.sharding_metrics["cross_shard_operations"] += 1
                
                return ConsensusResult(
                    consensus_achieved=False,
                    consensus_type="sharded_coordination_failed",
                    execution_time=execution_time
                )
                
        except Exception as e:
            print(f"❌ Sharded consensus error: {e}")
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            return ConsensusResult(
                consensus_achieved=False,
                consensus_type="sharded_error",
                execution_time=execution_time
            )
    
    async def _create_initial_shards(self, peer_nodes: List[PeerNode]) -> bool:
        """Create initial shards based on sharding strategy"""
        try:
            if self.sharding_strategy == ShardingStrategy.HASH_BASED:
                return await self._create_hash_based_shards(peer_nodes)
            elif self.sharding_strategy == ShardingStrategy.ADAPTIVE:
                return await self._create_adaptive_shards(peer_nodes)
            else:
                # Default to hash-based
                return await self._create_hash_based_shards(peer_nodes)
                
        except Exception as e:
            print(f"❌ Error creating initial shards: {e}")
            return False
    
    async def _create_hash_based_shards(self, peer_nodes: List[PeerNode]) -> bool:
        """Create shards using consistent hashing"""
        try:
            # Calculate optimal number of shards
            total_nodes = len(peer_nodes)
            num_shards = min(MAX_SHARDS, max(1, total_nodes // OPTIMAL_SHARD_SIZE))
            
            print(f"📊 Creating {num_shards} hash-based shards for {total_nodes} nodes")
            
            # Sort nodes by hash for consistent distribution
            sorted_nodes = sorted(peer_nodes, key=lambda n: hashlib.sha256(n.peer_id.encode()).hexdigest())
            
            # Distribute nodes across shards
            nodes_per_shard = total_nodes // num_shards
            remainder = total_nodes % num_shards
            
            node_index = 0
            for shard_index in range(num_shards):
                shard_id = f"shard_{shard_index:03d}"
                
                # Calculate shard size (distribute remainder across first shards)
                shard_size = nodes_per_shard + (1 if shard_index < remainder else 0)
                shard_nodes = sorted_nodes[node_index:node_index + shard_size]
                node_index += shard_size
                
                # Create and initialize shard
                shard = ConsensusShard(shard_id, shard_nodes)
                success = await shard.initialize_shard()
                
                if success:
                    self.shards[shard_id] = shard
                    print(f"✅ Created shard {shard_id} with {len(shard_nodes)} nodes")
                else:
                    print(f"❌ Failed to initialize shard {shard_id}")
                    return False
            
            self.sharding_metrics["total_shards"] = len(self.shards)
            self.sharding_metrics["active_shards"] = sum(1 for s in self.shards.values() if s.state == ShardState.ACTIVE)
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating hash-based shards: {e}")
            return False
    
    async def _create_adaptive_shards(self, peer_nodes: List[PeerNode]) -> bool:
        """Create shards using adaptive strategy based on node characteristics"""
        try:
            # Group nodes by reputation and network characteristics
            high_rep_nodes = [n for n in peer_nodes if n.reputation_score >= 0.8]
            medium_rep_nodes = [n for n in peer_nodes if 0.5 <= n.reputation_score < 0.8]
            low_rep_nodes = [n for n in peer_nodes if n.reputation_score < 0.5]
            
            print(f"📊 Creating adaptive shards based on node characteristics:")
            print(f"   - High reputation nodes: {len(high_rep_nodes)}")
            print(f"   - Medium reputation nodes: {len(medium_rep_nodes)}")
            print(f"   - Low reputation nodes: {len(low_rep_nodes)}")
            
            # Create shards with mixed reputation for resilience
            node_groups = [high_rep_nodes, medium_rep_nodes, low_rep_nodes]
            all_nodes = []
            
            # Interleave nodes to ensure balanced reputation per shard
            max_group_size = max(len(group) for group in node_groups)
            for i in range(max_group_size):
                for group in node_groups:
                    if i < len(group):
                        all_nodes.append(group[i])
            
            # Create shards with balanced composition
            total_nodes = len(all_nodes)
            num_shards = min(MAX_SHARDS, max(1, total_nodes // OPTIMAL_SHARD_SIZE))
            nodes_per_shard = total_nodes // num_shards
            remainder = total_nodes % num_shards
            
            node_index = 0
            for shard_index in range(num_shards):
                shard_id = f"adaptive_shard_{shard_index:03d}"
                
                shard_size = nodes_per_shard + (1 if shard_index < remainder else 0)
                shard_nodes = all_nodes[node_index:node_index + shard_size]
                node_index += shard_size
                
                # Create and initialize shard
                shard = ConsensusShard(shard_id, shard_nodes)
                success = await shard.initialize_shard()
                
                if success:
                    self.shards[shard_id] = shard
                    avg_reputation = statistics.mean(n.reputation_score for n in shard_nodes)
                    print(f"✅ Created {shard_id} with {len(shard_nodes)} nodes (avg rep: {avg_reputation:.2f})")
                else:
                    print(f"❌ Failed to initialize {shard_id}")
                    return False
            
            self.sharding_metrics["total_shards"] = len(self.shards)
            self.sharding_metrics["active_shards"] = sum(1 for s in self.shards.values() if s.state == ShardState.ACTIVE)
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating adaptive shards: {e}")
            return False
    
    async def _select_target_shards(self, proposal: Dict[str, Any]) -> List[str]:
        """Select target shards for a proposal"""
        try:
            # Extract sharding key from proposal
            sharding_key = proposal.get("shard_key", proposal.get("id", str(uuid4())))
            
            if self.sharding_strategy == ShardingStrategy.HASH_BASED:
                # Hash-based selection
                proposal_hash = hashlib.sha256(str(sharding_key).encode()).hexdigest()
                hash_int = int(proposal_hash[:8], 16)
                shard_index = hash_int % len(self.shards)
                shard_ids = list(self.shards.keys())
                return [shard_ids[shard_index]]
                
            elif self.sharding_strategy == ShardingStrategy.ADAPTIVE:
                # Adaptive selection based on current load
                active_shards = [(sid, shard) for sid, shard in self.shards.items() 
                               if shard.state == ShardState.ACTIVE]
                
                if not active_shards:
                    return []
                
                # Select shard with lowest load factor
                selected_shard = min(active_shards, key=lambda x: x[1].load_factor)
                return [selected_shard[0]]
            
            else:
                # Default: use first active shard
                active_shards = [sid for sid, shard in self.shards.items() 
                               if shard.state == ShardState.ACTIVE]
                return [active_shards[0]] if active_shards else []
                
        except Exception as e:
            print(f"❌ Error selecting target shards: {e}")
            return []
    
    async def _parallel_shard_consensus(self, target_shards: List[str], 
                                      proposal: Dict[str, Any], session_id: str) -> List[ConsensusResult]:
        """Execute consensus in parallel across multiple shards"""
        try:
            print(f"⚡ Running parallel consensus across {len(target_shards)} shards")
            
            # Create consensus tasks
            tasks = []
            for shard_id in target_shards:
                if shard_id in self.shards:
                    task = self.shards[shard_id].achieve_shard_consensus(proposal, f"{session_id}_{shard_id}")
                    tasks.append(task)
            
            # Execute all tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            consensus_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"⚠️ Shard {target_shards[i]} consensus failed: {result}")
                else:
                    consensus_results.append(result)
            
            return consensus_results
            
        except Exception as e:
            print(f"❌ Error in parallel shard consensus: {e}")
            return []
    
    async def _sequential_shard_consensus(self, target_shards: List[str], 
                                        proposal: Dict[str, Any], session_id: str) -> List[ConsensusResult]:
        """Execute consensus sequentially across shards"""
        try:
            results = []
            
            for shard_id in target_shards:
                if shard_id in self.shards:
                    result = await self.shards[shard_id].achieve_shard_consensus(proposal, f"{session_id}_{shard_id}")
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Error in sequential shard consensus: {e}")
            return []
    
    async def _cross_shard_coordination(self, shard_results: List[ConsensusResult], 
                                      session_id: str) -> Optional[ConsensusResult]:
        """Coordinate consensus results across shards"""
        try:
            if not shard_results:
                return None
            
            print(f"🔗 Coordinating results across {len(shard_results)} shards")
            
            # Prepare coordination proposal
            coordination_proposal = {
                "action": "cross_shard_coordination",
                "session_id": session_id,
                "shard_results": [
                    {
                        "consensus_achieved": r.consensus_achieved,
                        "agreement_ratio": r.agreement_ratio,
                        "consensus_type": r.consensus_type,
                        "result_hash": hashlib.sha256(str(r.agreed_value).encode()).hexdigest() if r.agreed_value else None
                    }
                    for r in shard_results
                ],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Use global consensus among shard coordinators
            global_result = await self.global_consensus.achieve_adaptive_consensus(coordination_proposal)
            
            if global_result.consensus_achieved:
                # Determine final result based on shard consensus
                successful_shards = [r for r in shard_results if r.consensus_achieved]
                
                if len(successful_shards) / len(shard_results) >= CROSS_SHARD_CONSENSUS_THRESHOLD:
                    # Majority of shards achieved consensus
                    # Use the result with highest agreement ratio
                    best_result = max(successful_shards, key=lambda r: r.agreement_ratio)
                    
                    return ConsensusResult(
                        agreed_value=best_result.agreed_value,
                        consensus_achieved=True,
                        consensus_type="cross_shard_coordinated",
                        agreement_ratio=len(successful_shards) / len(shard_results),
                        participating_peers=global_result.participating_peers
                    )
                else:
                    print(f"❌ Insufficient shard consensus: {len(successful_shards)}/{len(shard_results)}")
                    return None
            else:
                print(f"❌ Global coordination consensus failed")
                return None
                
        except Exception as e:
            print(f"❌ Error in cross-shard coordination: {e}")
            return None
    
    async def _load_balancing_loop(self):
        """Continuous load balancing for dynamic shard management"""
        try:
            while self.load_balancer_active:
                await asyncio.sleep(SHARD_SYNCHRONIZATION_INTERVAL)
                
                async with self._rebalance_lock:
                    if not self.rebalancing_in_progress:
                        await self._check_and_rebalance_shards()
                        
        except Exception as e:
            print(f"❌ Error in load balancing loop: {e}")
    
    async def _check_and_rebalance_shards(self):
        """Check shard loads and rebalance if needed"""
        try:
            # Check for overloaded/underloaded shards
            overloaded_shards = [sid for sid, shard in self.shards.items() 
                               if shard.state == ShardState.OVERLOADED]
            underloaded_shards = [sid for sid, shard in self.shards.items() 
                                if shard.state == ShardState.UNDERLOADED]
            
            if overloaded_shards:
                print(f"⚖️ Found {len(overloaded_shards)} overloaded shards")
                for shard_id in overloaded_shards:
                    await self._split_shard(shard_id)
            
            if len(underloaded_shards) >= 2:
                print(f"⚖️ Found {len(underloaded_shards)} underloaded shards")
                # Merge pairs of underloaded shards
                for i in range(0, len(underloaded_shards) - 1, 2):
                    await self._merge_shards(underloaded_shards[i], underloaded_shards[i + 1])
            
            # Update metrics
            self._update_sharding_metrics()
            
        except Exception as e:
            print(f"❌ Error checking and rebalancing shards: {e}")
    
    async def _split_shard(self, shard_id: str) -> bool:
        """Split an overloaded shard into two shards"""
        try:
            if shard_id not in self.shards:
                return False
            
            shard = self.shards[shard_id]
            if len(shard.nodes) < SHARD_SPLIT_THRESHOLD:
                return False
            
            print(f"🔀 Splitting overloaded shard {shard_id}")
            self.rebalancing_in_progress = True
            
            # Split nodes into two groups
            node_list = list(shard.nodes.values())
            mid_point = len(node_list) // 2
            
            nodes_1 = node_list[:mid_point]
            nodes_2 = node_list[mid_point:]
            
            # Create new shard IDs
            new_shard_1_id = f"{shard_id}_split_1"
            new_shard_2_id = f"{shard_id}_split_2"
            
            # Create new shards
            new_shard_1 = ConsensusShard(new_shard_1_id, nodes_1)
            new_shard_2 = ConsensusShard(new_shard_2_id, nodes_2)
            
            # Initialize new shards
            success_1 = await new_shard_1.initialize_shard()
            success_2 = await new_shard_2.initialize_shard()
            
            if success_1 and success_2:
                # Replace old shard with new shards
                del self.shards[shard_id]
                self.shards[new_shard_1_id] = new_shard_1
                self.shards[new_shard_2_id] = new_shard_2
                
                self.sharding_metrics["shard_splits"] += 1
                print(f"✅ Split shard {shard_id} into {new_shard_1_id} and {new_shard_2_id}")
                return True
            else:
                print(f"❌ Failed to initialize split shards for {shard_id}")
                return False
                
        except Exception as e:
            print(f"❌ Error splitting shard {shard_id}: {e}")
            return False
        finally:
            self.rebalancing_in_progress = False
    
    async def _merge_shards(self, shard_id_1: str, shard_id_2: str) -> bool:
        """Merge two underloaded shards into one"""
        try:
            if shard_id_1 not in self.shards or shard_id_2 not in self.shards:
                return False
            
            shard_1 = self.shards[shard_id_1]
            shard_2 = self.shards[shard_id_2]
            
            total_nodes = len(shard_1.nodes) + len(shard_2.nodes)
            if total_nodes > MAX_SHARD_SIZE:
                return False
            
            print(f"🔗 Merging underloaded shards {shard_id_1} and {shard_id_2}")
            self.rebalancing_in_progress = True
            
            # Combine nodes
            merged_nodes = list(shard_1.nodes.values()) + list(shard_2.nodes.values())
            merged_shard_id = f"merged_{shard_id_1}_{shard_id_2}"
            
            # Create merged shard
            merged_shard = ConsensusShard(merged_shard_id, merged_nodes)
            success = await merged_shard.initialize_shard()
            
            if success:
                # Replace old shards with merged shard
                del self.shards[shard_id_1]
                del self.shards[shard_id_2]
                self.shards[merged_shard_id] = merged_shard
                
                self.sharding_metrics["shard_merges"] += 1
                print(f"✅ Merged {shard_id_1} and {shard_id_2} into {merged_shard_id}")
                return True
            else:
                print(f"❌ Failed to initialize merged shard")
                return False
                
        except Exception as e:
            print(f"❌ Error merging shards: {e}")
            return False
        finally:
            self.rebalancing_in_progress = False
    
    def _calculate_total_throughput(self) -> float:
        """Calculate total throughput across all shards"""
        try:
            total_throughput = 0.0
            for shard in self.shards.values():
                if shard.throughput_samples:
                    shard_throughput = statistics.mean(shard.throughput_samples)
                    total_throughput += shard_throughput
            return total_throughput
        except Exception:
            return 0.0
    
    def _update_sharding_metrics(self):
        """Update comprehensive sharding metrics"""
        try:
            self.sharding_metrics.update({
                "total_shards": len(self.shards),
                "active_shards": sum(1 for s in self.shards.values() if s.state == ShardState.ACTIVE),
                "total_throughput": self._calculate_total_throughput(),
                "average_shard_load": statistics.mean(s.load_factor for s in self.shards.values()) if self.shards else 0,
                "nodes_per_shard": {sid: len(shard.nodes) for sid, shard in self.shards.items()},
                "shard_performance": {sid: shard.get_shard_metrics() for sid, shard in self.shards.items()}
            })
        except Exception as e:
            print(f"❌ Error updating sharding metrics: {e}")
    
    async def get_sharding_metrics(self) -> Dict[str, Any]:
        """Get comprehensive sharding metrics"""
        try:
            self._update_sharding_metrics()
            
            # Calculate scaling metrics
            total_nodes = sum(len(shard.nodes) for shard in self.shards.values())
            if total_nodes > 0:
                scaling_efficiency = self.sharding_metrics["total_throughput"] / total_nodes
            else:
                scaling_efficiency = 0.0
            
            return {
                **self.sharding_metrics,
                "sharding_strategy": self.sharding_strategy.value,
                "cross_shard_coordinator": self.cross_shard_coordinator,
                "total_nodes": total_nodes,
                "scaling_efficiency": scaling_efficiency,
                "load_balancer_active": self.load_balancer_active,
                "rebalancing_in_progress": self.rebalancing_in_progress,
                "configuration": {
                    "min_shard_size": MIN_SHARD_SIZE,
                    "max_shard_size": MAX_SHARD_SIZE,
                    "optimal_shard_size": OPTIMAL_SHARD_SIZE,
                    "max_shards": MAX_SHARDS,
                    "cross_shard_threshold": CROSS_SHARD_CONSENSUS_THRESHOLD
                }
            }
            
        except Exception as e:
            print(f"❌ Error getting sharding metrics: {e}")
            return self.sharding_metrics


# === Global Consensus Sharding Instance ===

_consensus_sharding_instance: Optional[ConsensusShardingManager] = None

def get_consensus_sharding(strategy: ShardingStrategy = ShardingStrategy.ADAPTIVE) -> ConsensusShardingManager:
    """Get or create the global consensus sharding instance"""
    global _consensus_sharding_instance
    if _consensus_sharding_instance is None:
        _consensus_sharding_instance = ConsensusShardingManager(strategy)
    return _consensus_sharding_instance