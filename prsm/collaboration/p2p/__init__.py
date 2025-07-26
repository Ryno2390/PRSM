"""
PRSM P2P Network Layer

This module provides the core P2P networking infrastructure for PRSM's
secure collaboration platform, implementing the "Coca Cola Recipe"
security model through distributed file sharding.

Key Components:
- Node Discovery: DHT-based peer discovery and network topology management
- Shard Distribution: Intelligent placement of encrypted file shards
- Bandwidth Optimization: Adaptive bandwidth management and QoS
- Node Reputation: Trust-based peer selection and behavior tracking
- Fallback Storage: IPFS integration for reliability and redundancy

The P2P network layer ensures that no single node has access to complete
files, while maintaining high availability and performance through
intelligent distribution strategies.
"""

from .node_discovery import (
    NodeDiscovery,
    PeerNode,
    KademliaDHT
)

from .shard_distribution import (
    ShardDistributor,
    ShardInfo,
    ShardLocation,
    DistributionPlan,
    ShardDistributionStrategy,
    GeographicOptimizer,
    BandwidthOptimizer,
    RedundancyManager
)

from .bandwidth_optimization import (
    BandwidthOptimizer,
    NetworkMonitor,
    AdaptiveRateController,
    QoSManager,
    TransferScheduler,
    BandwidthMeasurement,
    TransferRequest,
    TrafficPriority
)

from .node_reputation import (
    ReputationSystem,
    ReputationScore,
    ReputationEvent,
    ReputationFactor,
    BehaviorType,
    NodeMetrics,
    ReputationCalculator,
    ReputationTracker
)

from .fallback_storage import (
    FallbackStorageManager,
    IPFSClient,
    IPFSNode,
    StoredContent,
    StorageStrategy,
    StorageStatus
)

# Version information
__version__ = "1.0.0"
__author__ = "PRSM Development Team"

# Export main classes for easy import
__all__ = [
    # Node Discovery
    'NodeDiscovery',
    'PeerNode',
    'KademliaDHT',
    
    # Shard Distribution
    'ShardDistributor',
    'ShardInfo',
    'ShardLocation',
    'DistributionPlan',
    'ShardDistributionStrategy',
    'GeographicOptimizer',
    'BandwidthOptimizer',
    'RedundancyManager',
    
    # Bandwidth Optimization
    'BandwidthOptimizer',
    'NetworkMonitor',
    'AdaptiveRateController',
    'QoSManager',
    'TransferScheduler',
    'BandwidthMeasurement',
    'TransferRequest',
    'TrafficPriority',
    
    # Node Reputation
    'ReputationSystem',
    'ReputationScore',
    'ReputationEvent',
    'ReputationFactor',
    'BehaviorType',
    'NodeMetrics',
    'ReputationCalculator',
    'ReputationTracker',
    
    # Fallback Storage
    'FallbackStorageManager',
    'IPFSClient',
    'IPFSNode',
    'StoredContent',
    'StorageStrategy',
    'StorageStatus'
]

# Module documentation
DESCRIPTION = """
PRSM P2P Network Layer

This comprehensive P2P networking module implements the foundational
infrastructure for PRSM's secure collaboration platform. It provides:

1. **Distributed Node Discovery**
   - Kademlia DHT for decentralized peer discovery
   - Geographic and network topology awareness
   - Automatic peer health monitoring and management

2. **Intelligent Shard Distribution**
   - Strategic placement of encrypted file shards
   - Multiple distribution strategies (geographic, bandwidth, latency)
   - Redundancy management and fault tolerance
   - Load balancing across network nodes

3. **Adaptive Bandwidth Optimization**
   - Real-time network condition monitoring
   - Quality of Service (QoS) management
   - Adaptive rate control and traffic prioritization
   - Intelligent transfer scheduling

4. **Reputation-Based Trust System**
   - Multi-factor reputation scoring
   - Behavior tracking and analysis
   - Misbehavior detection and penalties
   - Trust-based peer selection

5. **Resilient Fallback Storage**
   - IPFS integration for data redundancy
   - Hybrid storage strategies
   - Automatic failover and recovery
   - Content addressing and verification

The "Coca Cola Recipe" Security Model ensures that file contents are
cryptographically sharded and distributed such that no single node
can access complete files without proper authorization.
"""

# Configuration constants
DEFAULT_CONFIG = {
    'node_discovery': {
        'port': 8467,
        'max_peers': 50,
        'bootstrap_nodes': []
    },
    'shard_distribution': {
        'default_redundancy': 3,
        'default_strategy': 'balanced',
        'max_shards_per_node': 20
    },
    'bandwidth_optimization': {
        'total_bandwidth': 100 * 1024 * 1024,  # 100 MB/s
        'initial_rate': 10 * 1024 * 1024,      # 10 MB/s
        'monitor_interval': 5.0
    },
    'reputation_system': {
        'min_reputation': 0.3,
        'preferred_reputation': 0.7,
        'cache_ttl': 300
    },
    'fallback_storage': {
        'default_strategy': 'hybrid',
        'auto_pin_threshold': 10 * 1024 * 1024,  # 10MB
        'verification_interval': 3600
    }
}

def get_default_config():
    """Get default configuration for P2P network layer"""
    return DEFAULT_CONFIG.copy()

def create_p2p_network(config=None):
    """
    Factory function to create a complete P2P network instance
    
    This is a convenience function that initializes all P2P components
    with proper integration between them.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Dictionary containing initialized P2P components
    """
    import asyncio
    
    # Use provided config or defaults
    full_config = DEFAULT_CONFIG.copy()
    if config:
        for section, values in config.items():
            if section in full_config:
                full_config[section].update(values)
            else:
                full_config[section] = values
    
    # Initialize components
    components = {}
    
    # Node Discovery
    components['node_discovery'] = NodeDiscovery(full_config['node_discovery'])
    
    # Reputation System
    components['reputation_system'] = ReputationSystem(full_config['reputation_system'])
    
    # Shard Distributor (requires node discovery)
    components['shard_distributor'] = ShardDistributor(
        components['node_discovery'],
        full_config['shard_distribution']
    )
    
    # Bandwidth Optimizer
    components['bandwidth_optimizer'] = BandwidthOptimizer(
        full_config['bandwidth_optimization']
    )
    
    # Fallback Storage (if IPFS nodes configured)
    if 'ipfs_nodes' in full_config:
        ipfs_client = IPFSClient(full_config['ipfs_nodes'])
        components['fallback_storage'] = FallbackStorageManager(
            ipfs_client,
            full_config['fallback_storage']
        )
    
    return components

async def start_p2p_network(components):
    """Start all P2P network components"""
    start_tasks = []
    
    if 'node_discovery' in components:
        start_tasks.append(components['node_discovery'].start())
    
    if 'bandwidth_optimizer' in components:
        start_tasks.append(components['bandwidth_optimizer'].start())
    
    if 'fallback_storage' in components:
        start_tasks.append(components['fallback_storage'].start())
    
    if start_tasks:
        await asyncio.gather(*start_tasks, return_exceptions=True)

async def stop_p2p_network(components):
    """Stop all P2P network components"""
    stop_tasks = []
    
    if 'fallback_storage' in components:
        stop_tasks.append(components['fallback_storage'].stop())
    
    if 'bandwidth_optimizer' in components:
        stop_tasks.append(components['bandwidth_optimizer'].stop())
    
    if 'node_discovery' in components:
        stop_tasks.append(components['node_discovery'].stop())
    
    if stop_tasks:
        await asyncio.gather(*stop_tasks, return_exceptions=True)

# Example usage documentation
EXAMPLE_USAGE = """
Example Usage:

    import asyncio
    from prsm.collaboration.p2p import create_p2p_network, start_p2p_network

    async def main():
        # Create P2P network with custom config
        config = {
            'node_discovery': {
                'port': 8467,
                'max_peers': 30
            },
            'shard_distribution': {
                'default_redundancy': 5
            }
        }
        
        components = create_p2p_network(config)
        
        try:
            # Start the network
            await start_p2p_network(components)
            
            # Use the components
            discovery = components['node_discovery']
            distributor = components['shard_distributor']
            reputation = components['reputation_system']
            
            # Your application logic here
            
        finally:
            # Clean shutdown
            await stop_p2p_network(components)

    asyncio.run(main())
"""