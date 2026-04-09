"""
P2P Model Network for PRSM
Torrent-like model distribution with safety oversight and distributed execution
"""

import asyncio
import hashlib
import json
import logging
import random
import warnings
from concurrent.futures import Future
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any, Tuple
from uuid import UUID, uuid4

from prsm.core.config import settings
from prsm.core.models import (
    ArchitectTask, PeerNode, ModelShard, TeacherModel, ModelType,
    SafetyFlag, CircuitBreakerEvent, AgentResponse
)
from prsm.storage import get_content_store
from prsm.economy.tokenomics.ftns_service import get_ftns_service
# v1.6.0 scope alignment: prsm.core.safety deleted. Federation primitives
# no longer validate in-network model output.
from .consensus import get_consensus, ConsensusType
from ..performance.benchmark_collector import time_async_operation, get_global_collector


logger = logging.getLogger(__name__)

_CANONICAL_COLLABORATION_REDIRECT = (
    "Use canonical collaboration dispatch via "
    "prsm.collaboration.CollaborationManager.dispatch_session() "
    "with prsm.node.agent_collaboration.AgentCollaboration bridge."
)


def _emit_collaboration_compatibility_fence(entrypoint: str) -> None:
    """Emit additive compatibility-only fence for collaboration-like federation entrypoints."""
    message = (
        f"Compatibility-only collaboration entrypoint used: {entrypoint}. "
        f"{_CANONICAL_COLLABORATION_REDIRECT}"
    )
    logger.warning(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


# === P2P Configuration ===

# Network settings
DEFAULT_SHARD_SIZE_MB = int(getattr(settings, "PRSM_SHARD_SIZE_MB", 100))
MAX_SHARDS_PER_MODEL = int(getattr(settings, "PRSM_MAX_SHARDS", 20))
PEER_DISCOVERY_INTERVAL = int(getattr(settings, "PRSM_PEER_DISCOVERY_SECONDS", 30))
EXECUTION_TIMEOUT_SECONDS = int(getattr(settings, "PRSM_EXECUTION_TIMEOUT", 300))

# Redundancy settings
MIN_REPLICAS_PER_SHARD = int(getattr(settings, "PRSM_MIN_REPLICAS", 3))
MAX_REPLICAS_PER_SHARD = int(getattr(settings, "PRSM_MAX_REPLICAS", 7))
REPLICATION_FACTOR = float(getattr(settings, "PRSM_REPLICATION_FACTOR", 2.0))

# Safety settings
ENABLE_SAFETY_MONITORING = getattr(settings, "PRSM_SAFETY_MONITORING", True)
MAX_CONCURRENT_EXECUTIONS = int(getattr(settings, "PRSM_MAX_CONCURRENT_EXEC", 10))
PEER_REPUTATION_THRESHOLD = float(getattr(settings, "PRSM_MIN_PEER_REPUTATION", 0.3))


class P2PModelNetwork:
    """
    P2P model network with IPFS-based content-addressed distribution and safety oversight.
    Coordinates distributed model execution using sharded content delivery and safety monitoring.
    """
    
    def __init__(self):
        # Core network state
        self.active_peers: Dict[str, PeerNode] = {}
        self.model_shards: Dict[str, List[ModelShard]] = {}  # model_cid -> shards
        self.shard_locations: Dict[UUID, Set[str]] = {}  # shard_id -> peer_ids
        self.execution_queue: asyncio.Queue = asyncio.Queue()

        # Consensus integration
        self.consensus_engine = get_consensus()
        
        # Performance tracking
        self.peer_performance: Dict[str, Dict[str, Any]] = {}
        self.execution_metrics: Dict[str, Any] = {}
        
        # Synchronization
        self._peer_lock = asyncio.Lock()
        self._shard_lock = asyncio.Lock()
        self._execution_lock = asyncio.Lock()
        
    
    # v1.6.0 scope alignment: distribute_model_shards() and
    # coordinate_distributed_execution() were AGI-era model-execution entrypoints
    # that relied on SafetyMonitor.validate_model_output + CircuitBreakerNetwork.
    # Federation primitives no longer validate in-network model output; execution
    # flows through prsm.node.compute_provider + prsm.collaboration.CollaborationManager.

    async def validate_peer_contributions(self, peer_results: List[dict],
                                        consensus_type: str = ConsensusType.BYZANTINE_FAULT_TOLERANT) -> bool:
        """
        Validate contributions from peers using advanced consensus mechanisms
        
        Args:
            peer_results: List of results from different peers
            consensus_type: Type of consensus mechanism to use
            
        Returns:
            True if contributions are valid and consensus is achieved
        """
        if not peer_results:
            return False
        
        try:
            # Enhanced consensus-based validation
            consensus_result = await self.consensus_engine.achieve_result_consensus(
                peer_results, 
                consensus_type
            )
            
            if consensus_result.consensus_achieved:
                # Update peer reputations based on consensus participation
                for peer_id in consensus_result.participating_peers:
                    if peer_id in self.active_peers:
                        # Positive reputation for successful consensus participation
                        await self._update_peer_reputation(peer_id, 0.05)
                        
                        # Update consensus engine reputation
                        await self.consensus_engine.update_peer_reputation(peer_id, 0.02)
                
                # Handle Byzantine failures if detected
                if consensus_result.failed_peers:
                    await self._handle_consensus_failures(consensus_result.failed_peers)
                
                print(f"✅ Advanced peer consensus achieved: {consensus_result.agreement_ratio:.2%} agreement ({consensus_type})")
                return True
            else:
                # Handle consensus failure
                print(f"❌ Advanced consensus failed: {consensus_result.agreement_ratio:.2%} agreement ({consensus_type})")
                
                # Penalize all participants for failed consensus
                for peer_id in consensus_result.participating_peers:
                    if peer_id in self.active_peers:
                        await self._update_peer_reputation(peer_id, -0.02)
                
                return False
                
        except Exception as e:
            print(f"❌ Error in advanced peer validation: {str(e)}")
            return False
    
    
    async def add_peer(self, peer: PeerNode) -> bool:
        """Add a new peer to the network"""
        async with self._peer_lock:
            try:
                # Validate peer capabilities
                if not peer.capabilities:
                    peer.capabilities = ["model_execution", "data_storage"]
                
                # Initialize peer performance tracking
                self.peer_performance[peer.peer_id] = {
                    "total_executions": 0,
                    "successful_executions": 0,
                    "average_response_time": 0.0,
                    "last_updated": datetime.now(timezone.utc)
                }
                
                # Add to active peers
                self.active_peers[peer.peer_id] = peer
                
                print(f"➕ Added peer {peer.peer_id} with capabilities: {peer.capabilities}")
                return True
                
            except Exception as e:
                print(f"❌ Failed to add peer {peer.peer_id}: {str(e)}")
                return False
    
    
    async def remove_peer(self, peer_id: str) -> bool:
        """Remove a peer from the network"""
        async with self._peer_lock:
            try:
                if peer_id in self.active_peers:
                    # Update shard locations
                    for shard_id, peer_set in self.shard_locations.items():
                        peer_set.discard(peer_id)
                    
                    # Remove peer
                    del self.active_peers[peer_id]
                    if peer_id in self.peer_performance:
                        del self.peer_performance[peer_id]
                    
                    print(f"➖ Removed peer {peer_id}")
                    return True
                return False
                
            except Exception as e:
                print(f"❌ Failed to remove peer {peer_id}: {str(e)}")
                return False
    
    
    async def validate_execution_integrity(self, execution_log: List[dict]) -> bool:
        """
        Validate execution integrity using distributed consensus mechanisms
        
        Args:
            execution_log: List of execution log entries from peers
            
        Returns:
            True if execution integrity is validated through consensus
        """
        try:
            # Use consensus engine for execution integrity validation
            integrity_valid = await self.consensus_engine.validate_execution_integrity(execution_log)
            
            if integrity_valid:
                print(f"✅ Execution integrity validated through consensus for {len(execution_log)} log entries")
            else:
                print(f"❌ Execution integrity validation failed for {len(execution_log)} log entries")

            return integrity_valid
            
        except Exception as e:
            print(f"❌ Error validating execution integrity: {str(e)}")
            return False
    
    
    async def get_network_status(self) -> Dict[str, Any]:
        """Get current network status including consensus metrics"""
        consensus_metrics = await self.consensus_engine.get_consensus_metrics()
        
        return {
            "total_peers": len(self.active_peers),
            "active_peers": list(self.active_peers.keys()),
            "total_models": len(self.model_shards),
            "total_shards": sum(len(shards) for shards in self.model_shards.values()),
            "pending_executions": self.execution_queue.qsize(),
            "safety_monitoring": ENABLE_SAFETY_MONITORING,
            "consensus_metrics": consensus_metrics
        }
    
    
    # === Private Helper Methods ===
    
    async def _select_hosting_peers(self, shard_id: UUID) -> List[str]:
        """Select peers to host a model shard"""
        qualified_peers = [
            peer_id for peer_id, peer in self.active_peers.items()
            if "data_storage" in peer.capabilities and 
               peer.reputation_score >= PEER_REPUTATION_THRESHOLD
        ]
        
        if not qualified_peers:
            # Fallback to all available peers if none qualified
            qualified_peers = list(self.active_peers.keys())
        
        # Select random subset for redundancy
        replica_count = min(len(qualified_peers), MAX_REPLICAS_PER_SHARD)
        replica_count = max(replica_count, MIN_REPLICAS_PER_SHARD)
        
        return random.sample(qualified_peers, min(replica_count, len(qualified_peers)))
    
    
    async def _store_shard_on_peers(self, shard: ModelShard, shard_data: bytes, peer_ids: List[str]):
        """Store shard data on selected peers (simulated)"""
        # In a real implementation, this would distribute the actual shard data
        # For now, we simulate successful storage
        for peer_id in peer_ids:
            if peer_id in self.active_peers:
                # Update peer storage metrics (simulated)
                if peer_id not in self.peer_performance:
                    self.peer_performance[peer_id] = {
                        "total_executions": 0,
                        "successful_executions": 0,
                        "average_response_time": 0.0,
                        "last_updated": datetime.now(timezone.utc)
                    }
                
                # Simulate storage success
                print(f"📦 Stored shard {shard.shard_index} on peer {peer_id}")
    
    
    async def _select_execution_peers(self, task: ArchitectTask) -> List[str]:
        """Select qualified peers for task execution"""
        qualified_peers = []
        
        for peer_id, peer in self.active_peers.items():
            # Check capabilities
            if "model_execution" not in peer.capabilities:
                continue
            
            # Check reputation
            if peer.reputation_score < PEER_REPUTATION_THRESHOLD:
                continue
            
            # Check if peer is active
            if not peer.active:
                continue
            
            qualified_peers.append(peer_id)
        
        # Limit concurrent executions
        max_peers = min(len(qualified_peers), MAX_CONCURRENT_EXECUTIONS)
        return qualified_peers[:max_peers]
    
    
    async def _execute_on_peer(self, peer_id: str, task: ArchitectTask) -> dict:
        """Execute task on a specific peer"""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Real task execution with performance instrumentation
            async with time_async_operation(
                f"peer_execution_{peer_id}", 
                {"peer_id": peer_id, "task_id": str(task.task_id), "instruction": task.instruction[:50]}
            ):
                # Actual task execution logic would go here
                # For now, simulate with realistic timing based on task complexity
                base_time = 0.1  # Minimum execution time
                complexity_factor = len(task.instruction) / 1000.0  # Scale with task complexity
                execution_time = base_time + complexity_factor + random.uniform(0.05, 0.2)
                
                # This sleep represents actual computational work
                # In production, this would be replaced with real AI model execution
                await asyncio.sleep(execution_time)
            
            # Get performance metrics from the collector
            collector = get_global_collector()
            metrics = collector.get_metrics(f"peer_execution_{peer_id}")
            actual_execution_time = metrics.mean_ms / 1000.0 if metrics else execution_time
            
            # Generate result with real timing data
            result = {
                "peer_id": peer_id,
                "task_id": task.task_id,
                "result": f"Execution result from peer {peer_id}",
                "execution_time": actual_execution_time,
                "timestamp": datetime.now(timezone.utc),
                "success": True,
                "performance_metrics": {
                    "mean_ms": metrics.mean_ms if metrics else None,
                    "sample_count": metrics.sample_count if metrics else 1
                }
            }
            
            # Update peer performance with real timing
            await self._update_peer_performance(peer_id, actual_execution_time, True)
            
            return result
            
        except Exception as e:
            # Handle execution failure with real timing
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            await self._update_peer_performance(peer_id, execution_time, False)
            
            return {
                "peer_id": peer_id,
                "task_id": task.task_id,
                "error": str(e),
                "execution_time": execution_time,
                "timestamp": datetime.now(timezone.utc),
                "success": False
            }
    
    
    async def _update_peer_reputation(self, peer_id: str, delta: float):
        """Update peer reputation score"""
        if peer_id in self.active_peers:
            current_score = self.active_peers[peer_id].reputation_score
            new_score = max(0.0, min(1.0, current_score + delta))
            self.active_peers[peer_id].reputation_score = new_score
    
    
    async def _update_peer_performance(self, peer_id: str, execution_time: float, success: bool):
        """Update peer performance metrics"""
        if peer_id not in self.peer_performance:
            self.peer_performance[peer_id] = {
                "total_executions": 0,
                "successful_executions": 0,
                "average_response_time": 0.0,
                "last_updated": datetime.now(timezone.utc)
            }
        
        metrics = self.peer_performance[peer_id]
        metrics["total_executions"] += 1
        
        if success:
            metrics["successful_executions"] += 1
        
        # Update average response time
        total_executions = metrics["total_executions"]
        current_avg = metrics["average_response_time"]
        metrics["average_response_time"] = (
            (current_avg * (total_executions - 1) + execution_time) / total_executions
        )
        
        metrics["last_updated"] = datetime.now(timezone.utc)
    
    
    async def _handle_consensus_failures(self, failed_peers: List[str]):
        """Handle consensus failures by updating reputations."""
        for peer_id in failed_peers:
            if peer_id in self.active_peers:
                # Significant reputation penalty for consensus failures
                await self._update_peer_reputation(peer_id, -0.15)

                # Check if peer should be removed from network
                peer_reputation = self.active_peers[peer_id].reputation_score
                if peer_reputation < PEER_REPUTATION_THRESHOLD:
                    print(f"⚠️ Peer {peer_id} reputation below threshold, considering removal")


# === Global P2P Network Instance ===

_p2p_network_instance: Optional[P2PModelNetwork] = None

def get_p2p_network() -> P2PModelNetwork:
    """Get or create the global P2P network instance"""
    global _p2p_network_instance
    if _p2p_network_instance is None:
        _p2p_network_instance = P2PModelNetwork()
    return _p2p_network_instance
