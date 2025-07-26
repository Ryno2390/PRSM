"""
Intelligent Shard Distribution System for PRSM P2P Collaboration

This module handles the strategic placement and distribution of encrypted
file shards across the P2P network, implementing the "Coca Cola Recipe"
security model where no single node has access to complete files.

Key Features:
- Geographic diversity for shard placement
- Redundancy management (configurable replication)
- Bandwidth-aware distribution
- Load balancing across network nodes
- Fault tolerance and automatic re-sharding
- Performance optimization for retrieval
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from enum import Enum
import random
import math

from .node_discovery import NodeDiscovery, PeerNode

logger = logging.getLogger(__name__)


class ShardDistributionStrategy(Enum):
    """Available strategies for shard distribution"""
    GEOGRAPHIC_DIVERSITY = "geographic_diversity"
    BANDWIDTH_OPTIMIZED = "bandwidth_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"
    REDUNDANCY_FOCUSED = "redundancy_focused"
    BALANCED = "balanced"


@dataclass
class ShardInfo:
    """Information about a file shard"""
    shard_id: str
    file_id: str
    shard_index: int
    total_shards: int
    size_bytes: int
    checksum: str
    encryption_key_id: str
    created_at: float
    expires_at: Optional[float] = None


@dataclass
class ShardLocation:
    """Location information for a shard"""
    shard_id: str
    node_id: str
    node_address: str
    stored_at: float
    last_verified: float
    access_count: int = 0
    is_primary: bool = True
    
    @property
    def is_stale(self) -> bool:
        """Check if location info is stale (not verified recently)"""
        return (time.time() - self.last_verified) > 3600  # 1 hour


@dataclass
class DistributionPlan:
    """Plan for distributing shards across the network"""
    file_id: str
    shards: List[ShardInfo]
    distribution_map: Dict[str, List[str]]  # shard_id -> list of node_ids
    strategy: ShardDistributionStrategy
    redundancy_factor: int
    created_at: float
    estimated_retrieval_time: float
    
    def get_shard_nodes(self, shard_id: str) -> List[str]:
        """Get list of node IDs storing a specific shard"""
        return self.distribution_map.get(shard_id, [])
    
    def get_node_shards(self, node_id: str) -> List[str]:
        """Get list of shard IDs stored on a specific node"""
        node_shards = []
        for shard_id, node_list in self.distribution_map.items():
            if node_id in node_list:
                node_shards.append(shard_id)
        return node_shards


class GeographicOptimizer:
    """Optimizes shard placement for geographic diversity"""
    
    def __init__(self):
        self.region_priorities = {
            'us-east-1': 1.0,
            'us-west-2': 1.0,
            'eu-west-1': 0.9,
            'ap-southeast-1': 0.8,
            'unknown': 0.5
        }
    
    def calculate_diversity_score(self, selected_nodes: List[PeerNode]) -> float:
        """Calculate geographic diversity score for selected nodes"""
        if not selected_nodes:
            return 0.0
        
        regions = set()
        for node in selected_nodes:
            regions.add(node.geographic_region or 'unknown')
        
        # Score based on number of unique regions
        region_count = len(regions)
        max_possible_regions = min(len(self.region_priorities), len(selected_nodes))
        
        diversity_score = region_count / max_possible_regions if max_possible_regions > 0 else 0
        
        return diversity_score
    
    def select_diverse_nodes(self, available_nodes: List[PeerNode], 
                           count: int) -> List[PeerNode]:
        """Select nodes to maximize geographic diversity"""
        if len(available_nodes) <= count:
            return available_nodes
        
        # Group nodes by region
        regions = {}
        for node in available_nodes:
            region = node.geographic_region or 'unknown'
            if region not in regions:
                regions[region] = []
            regions[region].append(node)
        
        # Sort regions by priority
        sorted_regions = sorted(
            regions.keys(),
            key=lambda r: self.region_priorities.get(r, 0.5),
            reverse=True
        )
        
        selected_nodes = []
        nodes_per_region = max(1, count // len(sorted_regions))
        
        # Distribute selections across regions
        for region in sorted_regions:
            if len(selected_nodes) >= count:
                break
            
            region_nodes = regions[region]
            # Sort by reputation and latency
            region_nodes.sort(
                key=lambda n: (n.reputation_score, -(n.network_latency or 1.0)),
                reverse=True
            )
            
            take_count = min(nodes_per_region, len(region_nodes), count - len(selected_nodes))
            selected_nodes.extend(region_nodes[:take_count])
        
        # Fill remaining slots with best available nodes
        if len(selected_nodes) < count:
            remaining_nodes = [
                node for node in available_nodes
                if node not in selected_nodes
            ]
            remaining_nodes.sort(
                key=lambda n: (n.reputation_score, -(n.network_latency or 1.0)),
                reverse=True
            )
            
            needed = count - len(selected_nodes)
            selected_nodes.extend(remaining_nodes[:needed])
        
        return selected_nodes[:count]


class BandwidthOptimizer:
    """Optimizes shard placement for bandwidth efficiency"""
    
    def __init__(self):
        self.min_bandwidth_mbps = 10  # Minimum acceptable bandwidth
        self.bandwidth_weight = 0.7   # Weight for bandwidth in scoring
        self.latency_weight = 0.3     # Weight for latency in scoring
    
    def calculate_bandwidth_score(self, node: PeerNode) -> float:
        """Calculate bandwidth score for a node"""
        if not node.bandwidth_capacity:
            return 0.5  # Unknown bandwidth gets neutral score
        
        # Convert to Mbps if needed and normalize
        bandwidth_mbps = node.bandwidth_capacity / (1024 * 1024 / 8)  # bytes/s to Mbps
        
        if bandwidth_mbps < self.min_bandwidth_mbps:
            return 0.1  # Very low score for insufficient bandwidth
        
        # Logarithmic scoring for bandwidth (diminishing returns)
        score = min(1.0, math.log10(bandwidth_mbps / self.min_bandwidth_mbps + 1))
        
        return score
    
    def calculate_latency_score(self, node: PeerNode) -> float:
        """Calculate latency score for a node"""
        if not node.network_latency:
            return 0.5  # Unknown latency gets neutral score
        
        # Lower latency = higher score
        latency_ms = node.network_latency * 1000
        
        if latency_ms > 500:  # Very high latency
            return 0.1
        
        # Inverse scoring for latency
        score = max(0.1, 1.0 - (latency_ms / 500))
        
        return score
    
    def select_optimal_nodes(self, available_nodes: List[PeerNode], 
                           count: int) -> List[PeerNode]:
        """Select nodes optimized for bandwidth and latency"""
        if len(available_nodes) <= count:
            return available_nodes
        
        # Calculate combined scores
        scored_nodes = []
        for node in available_nodes:
            bandwidth_score = self.calculate_bandwidth_score(node)
            latency_score = self.calculate_latency_score(node)
            
            combined_score = (
                bandwidth_score * self.bandwidth_weight +
                latency_score * self.latency_weight +
                node.reputation_score * 0.2  # Small reputation factor
            )
            
            scored_nodes.append((combined_score, node))
        
        # Sort by score and select top nodes
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        
        return [node for _, node in scored_nodes[:count]]


class RedundancyManager:
    """Manages redundancy and fault tolerance for shard distribution"""
    
    def __init__(self, default_redundancy: int = 3):
        self.default_redundancy = default_redundancy
        self.max_redundancy = 7
        self.min_redundancy = 2
    
    def calculate_optimal_redundancy(self, file_size: int, 
                                   network_size: int,
                                   importance_level: str = "normal") -> int:
        """Calculate optimal redundancy factor"""
        base_redundancy = self.default_redundancy
        
        # Adjust based on file importance
        importance_multipliers = {
            "low": 0.7,
            "normal": 1.0,
            "high": 1.3,
            "critical": 1.6
        }
        
        multiplier = importance_multipliers.get(importance_level, 1.0)
        
        # Adjust based on network size
        if network_size < 10:
            multiplier *= 0.8  # Lower redundancy for small networks
        elif network_size > 100:
            multiplier *= 1.2  # Higher redundancy for large networks
        
        # Adjust based on file size (larger files get slightly more redundancy)
        if file_size > 100 * 1024 * 1024:  # > 100MB
            multiplier *= 1.1
        
        optimal_redundancy = int(base_redundancy * multiplier)
        
        return max(self.min_redundancy, min(self.max_redundancy, optimal_redundancy))
    
    def validate_distribution(self, distribution_map: Dict[str, List[str]],
                            required_redundancy: int) -> Tuple[bool, List[str]]:
        """Validate that distribution meets redundancy requirements"""
        issues = []
        
        for shard_id, node_list in distribution_map.items():
            if len(node_list) < required_redundancy:
                issues.append(
                    f"Shard {shard_id} has insufficient redundancy: "
                    f"{len(node_list)}/{required_redundancy}"
                )
        
        return len(issues) == 0, issues
    
    def suggest_replication_targets(self, current_nodes: List[str],
                                  available_nodes: List[PeerNode],
                                  target_redundancy: int) -> List[PeerNode]:
        """Suggest additional nodes for replication"""
        current_node_set = set(current_nodes)
        
        # Filter out nodes that already have the shard
        candidate_nodes = [
            node for node in available_nodes
            if node.node_id not in current_node_set
        ]
        
        needed_replicas = target_redundancy - len(current_nodes)
        if needed_replicas <= 0:
            return []
        
        # Sort candidates by quality
        candidate_nodes.sort(
            key=lambda n: (n.reputation_score, -(n.network_latency or 1.0)),
            reverse=True
        )
        
        return candidate_nodes[:needed_replicas]


class ShardDistributor:
    """
    Main shard distribution system
    
    Coordinates the intelligent placement of encrypted file shards
    across the P2P network using various optimization strategies.
    """
    
    def __init__(self, node_discovery: NodeDiscovery, config: Optional[Dict] = None):
        self.node_discovery = node_discovery
        self.config = config or {}
        
        # Initialize optimizers
        self.geo_optimizer = GeographicOptimizer()
        self.bandwidth_optimizer = BandwidthOptimizer()
        self.redundancy_manager = RedundancyManager(
            self.config.get('default_redundancy', 3)
        )
        
        # Distribution settings
        self.default_strategy = ShardDistributionStrategy(
            self.config.get('default_strategy', 'balanced')
        )
        self.max_shards_per_node = self.config.get('max_shards_per_node', 20)
        self.min_available_nodes = self.config.get('min_available_nodes', 5)
        
        # Storage for tracking distributions
        self.active_distributions: Dict[str, DistributionPlan] = {}
        self.shard_locations: Dict[str, List[ShardLocation]] = {}
        
        logger.info("ShardDistributor initialized")
    
    async def create_distribution_plan(self, file_id: str, 
                                     shards: List[ShardInfo],
                                     strategy: Optional[ShardDistributionStrategy] = None,
                                     redundancy: Optional[int] = None,
                                     constraints: Optional[Dict] = None) -> DistributionPlan:
        """
        Create an optimal distribution plan for file shards
        """
        strategy = strategy or self.default_strategy
        constraints = constraints or {}
        
        # Get available nodes
        available_nodes = self.node_discovery.get_optimal_peers(100)
        
        if len(available_nodes) < self.min_available_nodes:
            raise ValueError(
                f"Insufficient nodes available: {len(available_nodes)} < {self.min_available_nodes}"
            )
        
        # Calculate optimal redundancy
        total_size = sum(shard.size_bytes for shard in shards)
        redundancy = redundancy or self.redundancy_manager.calculate_optimal_redundancy(
            total_size, len(available_nodes), constraints.get('importance', 'normal')
        )
        
        # Create distribution map
        distribution_map = {}
        
        if strategy == ShardDistributionStrategy.GEOGRAPHIC_DIVERSITY:
            distribution_map = await self._create_geographic_distribution(
                shards, available_nodes, redundancy
            )
        elif strategy == ShardDistributionStrategy.BANDWIDTH_OPTIMIZED:
            distribution_map = await self._create_bandwidth_optimized_distribution(
                shards, available_nodes, redundancy
            )
        elif strategy == ShardDistributionStrategy.LATENCY_OPTIMIZED:
            distribution_map = await self._create_latency_optimized_distribution(
                shards, available_nodes, redundancy
            )
        elif strategy == ShardDistributionStrategy.REDUNDANCY_FOCUSED:
            distribution_map = await self._create_redundancy_focused_distribution(
                shards, available_nodes, redundancy + 1  # Extra redundancy
            )
        else:  # BALANCED strategy
            distribution_map = await self._create_balanced_distribution(
                shards, available_nodes, redundancy
            )
        
        # Validate distribution
        is_valid, issues = self.redundancy_manager.validate_distribution(
            distribution_map, redundancy
        )
        
        if not is_valid:
            logger.warning(f"Distribution validation issues: {issues}")
            # Try to fix issues by adding more replicas
            distribution_map = await self._fix_distribution_issues(
                distribution_map, available_nodes, redundancy
            )
        
        # Estimate retrieval time
        estimated_time = self._estimate_retrieval_time(
            distribution_map, available_nodes
        )
        
        # Create distribution plan
        plan = DistributionPlan(
            file_id=file_id,
            shards=shards,
            distribution_map=distribution_map,
            strategy=strategy,
            redundancy_factor=redundancy,
            created_at=time.time(),
            estimated_retrieval_time=estimated_time
        )
        
        # Store plan
        self.active_distributions[file_id] = plan
        
        # Initialize shard location tracking
        for shard in shards:
            self.shard_locations[shard.shard_id] = []
        
        logger.info(f"Created distribution plan for {file_id}: "
                   f"{len(shards)} shards, {redundancy}x redundancy, "
                   f"strategy={strategy.value}")
        
        return plan
    
    async def _create_geographic_distribution(self, shards: List[ShardInfo],
                                            available_nodes: List[PeerNode],
                                            redundancy: int) -> Dict[str, List[str]]:
        """Create distribution optimized for geographic diversity"""
        distribution_map = {}
        
        for shard in shards:
            # Select nodes with maximum geographic diversity
            selected_nodes = self.geo_optimizer.select_diverse_nodes(
                available_nodes, redundancy
            )
            
            distribution_map[shard.shard_id] = [node.node_id for node in selected_nodes]
        
        return distribution_map
    
    async def _create_bandwidth_optimized_distribution(self, shards: List[ShardInfo],
                                                     available_nodes: List[PeerNode],
                                                     redundancy: int) -> Dict[str, List[str]]:
        """Create distribution optimized for bandwidth efficiency"""
        distribution_map = {}
        
        for shard in shards:
            # Select nodes with best bandwidth characteristics
            selected_nodes = self.bandwidth_optimizer.select_optimal_nodes(
                available_nodes, redundancy
            )
            
            distribution_map[shard.shard_id] = [node.node_id for node in selected_nodes]
        
        return distribution_map
    
    async def _create_latency_optimized_distribution(self, shards: List[ShardInfo],
                                                   available_nodes: List[PeerNode],
                                                   redundancy: int) -> Dict[str, List[str]]:
        """Create distribution optimized for low latency retrieval"""
        # Sort nodes by latency
        sorted_nodes = sorted(
            available_nodes,
            key=lambda n: n.network_latency or 999.0
        )
        
        distribution_map = {}
        
        for shard in shards:
            # Select lowest latency nodes
            selected_nodes = sorted_nodes[:redundancy]
            distribution_map[shard.shard_id] = [node.node_id for node in selected_nodes]
        
        return distribution_map
    
    async def _create_redundancy_focused_distribution(self, shards: List[ShardInfo],
                                                    available_nodes: List[PeerNode],
                                                    redundancy: int) -> Dict[str, List[str]]:
        """Create distribution with maximum redundancy and fault tolerance"""
        distribution_map = {}
        
        # Use all available high-quality nodes
        high_quality_nodes = [
            node for node in available_nodes
            if node.reputation_score >= 0.7
        ]
        
        for shard in shards:
            # Use as many high-quality nodes as possible
            target_count = min(redundancy, len(high_quality_nodes))
            selected_nodes = high_quality_nodes[:target_count]
            
            distribution_map[shard.shard_id] = [node.node_id for node in selected_nodes]
        
        return distribution_map
    
    async def _create_balanced_distribution(self, shards: List[ShardInfo],
                                          available_nodes: List[PeerNode],
                                          redundancy: int) -> Dict[str, List[str]]:
        """Create balanced distribution considering all factors"""
        distribution_map = {}
        node_load = {node.node_id: 0 for node in available_nodes}
        
        for shard in shards:
            # Score nodes based on multiple factors
            scored_nodes = []
            
            for node in available_nodes:
                # Calculate composite score
                geo_score = 1.0  # Placeholder - would use actual geographic scoring
                bandwidth_score = self.bandwidth_optimizer.calculate_bandwidth_score(node)
                latency_score = self.bandwidth_optimizer.calculate_latency_score(node)
                reputation_score = node.reputation_score
                
                # Penalize nodes with high load
                load_penalty = node_load[node.node_id] / self.max_shards_per_node
                
                composite_score = (
                    geo_score * 0.2 +
                    bandwidth_score * 0.3 +
                    latency_score * 0.2 +
                    reputation_score * 0.2 +
                    (1.0 - load_penalty) * 0.1
                )
                
                scored_nodes.append((composite_score, node))
            
            # Sort by score and select top nodes
            scored_nodes.sort(key=lambda x: x[0], reverse=True)
            selected_nodes = [node for _, node in scored_nodes[:redundancy]]
            
            # Update node load tracking
            for node in selected_nodes:
                node_load[node.node_id] += 1
            
            distribution_map[shard.shard_id] = [node.node_id for node in selected_nodes]
        
        return distribution_map
    
    async def _fix_distribution_issues(self, distribution_map: Dict[str, List[str]],
                                     available_nodes: List[PeerNode],
                                     required_redundancy: int) -> Dict[str, List[str]]:
        """Fix issues in distribution plan"""
        fixed_map = distribution_map.copy()
        
        for shard_id, node_list in fixed_map.items():
            if len(node_list) < required_redundancy:
                # Find additional nodes
                current_nodes = set(node_list)
                additional_nodes = [
                    node for node in available_nodes
                    if node.node_id not in current_nodes
                ]
                
                needed = required_redundancy - len(node_list)
                
                # Sort by quality and select best available
                additional_nodes.sort(
                    key=lambda n: (n.reputation_score, -(n.network_latency or 1.0)),
                    reverse=True
                )
                
                for node in additional_nodes[:needed]:
                    fixed_map[shard_id].append(node.node_id)
        
        return fixed_map
    
    def _estimate_retrieval_time(self, distribution_map: Dict[str, List[str]],
                               available_nodes: List[PeerNode]) -> float:
        """Estimate time to retrieve all shards"""
        node_map = {node.node_id: node for node in available_nodes}
        
        shard_times = []
        
        for shard_id, node_list in distribution_map.items():
            shard_nodes = [node_map.get(nid) for nid in node_list if nid in node_map]
            
            if not shard_nodes:
                shard_times.append(60.0)  # Default high time for missing nodes
                continue
            
            # Estimate time as best latency among available nodes
            best_latency = min(
                node.network_latency or 1.0 for node in shard_nodes
            )
            
            shard_times.append(best_latency)
        
        # Return maximum time (bottleneck shard)
        return max(shard_times) if shard_times else 10.0
    
    async def execute_distribution(self, plan: DistributionPlan) -> bool:
        """Execute a distribution plan by sending shards to nodes"""
        logger.info(f"Executing distribution plan for {plan.file_id}")
        
        success_count = 0
        total_operations = sum(len(nodes) for nodes in plan.distribution_map.values())
        
        for shard in plan.shards:
            target_nodes = plan.distribution_map.get(shard.shard_id, [])
            
            for node_id in target_nodes:
                try:
                    # In a real implementation, this would send the shard data
                    # For now, we'll simulate successful distribution
                    await self._store_shard_on_node(shard, node_id)
                    
                    # Record successful storage location
                    location = ShardLocation(
                        shard_id=shard.shard_id,
                        node_id=node_id,
                        node_address=f"{node_id}:8467",  # Simplified
                        stored_at=time.time(),
                        last_verified=time.time()
                    )
                    
                    if shard.shard_id not in self.shard_locations:
                        self.shard_locations[shard.shard_id] = []
                    
                    self.shard_locations[shard.shard_id].append(location)
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to store shard {shard.shard_id} on {node_id}: {e}")
        
        success_rate = success_count / total_operations if total_operations > 0 else 0
        logger.info(f"Distribution execution completed: {success_rate:.2%} success rate")
        
        return success_rate > 0.8  # Consider successful if >80% operations succeeded
    
    async def _store_shard_on_node(self, shard: ShardInfo, node_id: str):
        """Store a shard on a specific node (placeholder implementation)"""
        # In a real implementation, this would:
        # 1. Connect to the target node
        # 2. Send the encrypted shard data
        # 3. Verify successful storage
        # 4. Update shard location tracking
        
        # Simulate network delay
        await asyncio.sleep(0.1)
        
        logger.debug(f"Stored shard {shard.shard_id} on node {node_id}")
    
    def get_distribution_plan(self, file_id: str) -> Optional[DistributionPlan]:
        """Get the distribution plan for a file"""
        return self.active_distributions.get(file_id)
    
    def get_shard_locations(self, shard_id: str) -> List[ShardLocation]:
        """Get all known locations for a shard"""
        return self.shard_locations.get(shard_id, [])
    
    def get_distribution_stats(self) -> Dict[str, Any]:
        """Get distribution system statistics"""
        total_files = len(self.active_distributions)
        total_shards = sum(len(plan.shards) for plan in self.active_distributions.values())
        total_locations = sum(len(locations) for locations in self.shard_locations.values())
        
        avg_redundancy = 0
        if self.active_distributions:
            avg_redundancy = sum(plan.redundancy_factor for plan in self.active_distributions.values()) / total_files
        
        return {
            'total_files': total_files,
            'total_shards': total_shards,
            'total_locations': total_locations,
            'average_redundancy': avg_redundancy,
            'active_distributions': list(self.active_distributions.keys())
        }


# Example usage
async def example_shard_distribution():
    """Example of shard distribution system usage"""
    from .node_discovery import NodeDiscovery
    
    # Initialize node discovery
    discovery = NodeDiscovery({'port': 8467})
    await discovery.start()
    
    # Initialize shard distributor
    distributor = ShardDistributor(discovery, {
        'default_redundancy': 3,
        'default_strategy': 'balanced'
    })
    
    # Create example shards
    shards = [
        ShardInfo(
            shard_id=f"shard_{i}",
            file_id="example_file_123",
            shard_index=i,
            total_shards=5,
            size_bytes=1024 * 1024,  # 1MB
            checksum=f"checksum_{i}",
            encryption_key_id="key_123",
            created_at=time.time()
        )
        for i in range(5)
    ]
    
    try:
        # Create distribution plan
        plan = await distributor.create_distribution_plan(
            "example_file_123",
            shards,
            strategy=ShardDistributionStrategy.BALANCED,
            redundancy=3
        )
        
        print(f"Created distribution plan:")
        print(f"  File ID: {plan.file_id}")
        print(f"  Shards: {len(plan.shards)}")
        print(f"  Redundancy: {plan.redundancy_factor}")
        print(f"  Strategy: {plan.strategy.value}")
        print(f"  Estimated retrieval time: {plan.estimated_retrieval_time:.2f}s")
        
        # Execute distribution
        success = await distributor.execute_distribution(plan)
        print(f"Distribution execution: {'Success' if success else 'Failed'}")
        
        # Get statistics
        stats = distributor.get_distribution_stats()
        print(f"Distribution stats: {json.dumps(stats, indent=2)}")
        
    finally:
        await discovery.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_shard_distribution())