"""
P2P Node Discovery System for PRSM Secure Collaboration

This module implements DHT-based peer discovery for the distributed
file sharding network, enabling nodes to find and connect to other
peers for secure collaboration.

Key Features:
- Kademlia DHT for decentralized peer discovery
- Bootstrap node management for network initialization
- Peer reputation scoring integration
- Geographic and network topology awareness
- Automatic peer health monitoring
"""

import asyncio
import hashlib
import json
import logging
import socket
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple, Any
from urllib.parse import urlparse
import struct
import random

import aiohttp
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption

logger = logging.getLogger(__name__)


@dataclass
class PeerNode:
    """Represents a peer node in the P2P network"""
    node_id: str
    ip_address: str
    port: int
    public_key: str
    last_seen: float
    reputation_score: float = 1.0
    geographic_region: Optional[str] = None
    network_latency: Optional[float] = None
    bandwidth_capacity: Optional[int] = None
    supported_protocols: List[str] = None
    is_bootstrap: bool = False
    
    def __post_init__(self):
        if self.supported_protocols is None:
            self.supported_protocols = ['prsm-v1']
    
    @property
    def address(self) -> str:
        """Return the network address of this peer"""
        return f"{self.ip_address}:{self.port}"
    
    @property
    def is_active(self) -> bool:
        """Check if peer has been seen recently (within 5 minutes)"""
        return (time.time() - self.last_seen) < 300
    
    def to_dict(self) -> dict:
        """Convert peer to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PeerNode':
        """Create peer from dictionary"""
        return cls(**data)


class KademliaDHT:
    """
    Simplified Kademlia DHT implementation for peer discovery
    
    This implements a distributed hash table for decentralized
    peer discovery and routing in the PRSM network.
    """
    
    def __init__(self, node_id: str, port: int = 8467):
        self.node_id = node_id
        self.port = port
        self.k_bucket_size = 20  # Maximum nodes per k-bucket
        self.alpha = 3  # Concurrency parameter
        self.routing_table: Dict[int, List[PeerNode]] = {}
        self.peer_cache: Dict[str, PeerNode] = {}
        self.bootstrap_nodes: List[PeerNode] = []
        self.running = False
        
        # Initialize k-buckets (160 buckets for 160-bit node IDs)
        for i in range(160):
            self.routing_table[i] = []
    
    def _distance(self, node_id1: str, node_id2: str) -> int:
        """Calculate XOR distance between two node IDs"""
        id1_bytes = bytes.fromhex(node_id1)
        id2_bytes = bytes.fromhex(node_id2)
        
        distance = 0
        for b1, b2 in zip(id1_bytes, id2_bytes):
            distance = (distance << 8) | (b1 ^ b2)
        
        return distance
    
    def _get_bucket_index(self, node_id: str) -> int:
        """Get the k-bucket index for a given node ID"""
        distance = self._distance(self.node_id, node_id)
        if distance == 0:
            return 0
        
        return 159 - distance.bit_length() + 1
    
    def add_peer(self, peer: PeerNode) -> bool:
        """Add a peer to the routing table"""
        if peer.node_id == self.node_id:
            return False
        
        bucket_index = self._get_bucket_index(peer.node_id)
        bucket = self.routing_table[bucket_index]
        
        # If peer already exists, update it
        for i, existing_peer in enumerate(bucket):
            if existing_peer.node_id == peer.node_id:
                bucket[i] = peer
                self.peer_cache[peer.node_id] = peer
                return True
        
        # If bucket has space, add peer
        if len(bucket) < self.k_bucket_size:
            bucket.append(peer)
            self.peer_cache[peer.node_id] = peer
            return True
        
        # Bucket is full - ping least recently seen peer
        # For simplicity, we'll just replace the oldest peer
        oldest_peer = min(bucket, key=lambda p: p.last_seen)
        if not oldest_peer.is_active:
            bucket.remove(oldest_peer)
            bucket.append(peer)
            if oldest_peer.node_id in self.peer_cache:
                del self.peer_cache[oldest_peer.node_id]
            self.peer_cache[peer.node_id] = peer
            return True
        
        return False
    
    def find_closest_peers(self, target_id: str, count: int = 20) -> List[PeerNode]:
        """Find the closest peers to a target node ID"""
        all_peers = []
        for bucket in self.routing_table.values():
            all_peers.extend(bucket)
        
        # Sort by distance to target
        all_peers.sort(key=lambda p: self._distance(target_id, p.node_id))
        
        return all_peers[:count]
    
    def get_peer(self, node_id: str) -> Optional[PeerNode]:
        """Get a specific peer by node ID"""
        return self.peer_cache.get(node_id)
    
    def remove_peer(self, node_id: str) -> bool:
        """Remove a peer from the routing table"""
        if node_id not in self.peer_cache:
            return False
        
        peer = self.peer_cache[node_id]
        bucket_index = self._get_bucket_index(node_id)
        bucket = self.routing_table[bucket_index]
        
        try:
            bucket.remove(peer)
            del self.peer_cache[node_id]
            return True
        except ValueError:
            return False
    
    def get_all_peers(self) -> List[PeerNode]:
        """Get all known peers"""
        return list(self.peer_cache.values())
    
    def get_active_peers(self) -> List[PeerNode]:
        """Get all active peers"""
        return [peer for peer in self.peer_cache.values() if peer.is_active]


class NodeDiscovery:
    """
    Main node discovery service for PRSM P2P network
    
    Manages peer discovery, maintains network topology awareness,
    and provides high-level peer management functions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.node_id = self._generate_node_id()
        self.port = self.config.get('port', 8467)
        self.dht = KademliaDHT(self.node_id, self.port)
        self.server = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.discovery_tasks: Set[asyncio.Task] = set()
        
        # Network configuration
        self.max_peers = self.config.get('max_peers', 50)
        self.bootstrap_urls = self.config.get('bootstrap_nodes', [
            'bootstrap1.prsm.network:8467',
            'bootstrap2.prsm.network:8467'
        ])
        
        # Geographic and network awareness
        self.geographic_region = self.config.get('region', 'unknown')
        self.preferred_regions = self.config.get('preferred_regions', [])
        
        logger.info(f"Initialized NodeDiscovery with ID: {self.node_id}")
    
    def _generate_node_id(self) -> str:
        """Generate a unique node ID based on system characteristics"""
        hostname = socket.gethostname()
        timestamp = str(time.time())
        random_bytes = random.randbytes(16)
        
        content = f"{hostname}:{timestamp}:{random_bytes.hex()}"
        node_hash = hashlib.sha1(content.encode()).hexdigest()
        
        return node_hash
    
    async def start(self):
        """Start the node discovery service"""
        if self.dht.running:
            return
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        
        # Start DHT
        self.dht.running = True
        
        # Bootstrap network connection
        await self._bootstrap_network()
        
        # Start periodic tasks
        self._schedule_periodic_tasks()
        
        logger.info("Node discovery service started")
    
    async def stop(self):
        """Stop the node discovery service"""
        self.dht.running = False
        
        # Cancel all discovery tasks
        for task in self.discovery_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.discovery_tasks:
            await asyncio.gather(*self.discovery_tasks, return_exceptions=True)
        
        if self.session:
            await self.session.close()
        
        logger.info("Node discovery service stopped")
    
    async def _bootstrap_network(self):
        """Bootstrap connection to the P2P network"""
        bootstrap_peers = await self._load_bootstrap_peers()
        
        if not bootstrap_peers:
            logger.warning("No bootstrap peers available - starting in isolated mode")
            return
        
        # Connect to bootstrap peers
        connected_count = 0
        for peer in bootstrap_peers:
            try:
                if await self._connect_to_peer(peer):
                    connected_count += 1
                    logger.info(f"Connected to bootstrap peer: {peer.address}")
            except Exception as e:
                logger.warning(f"Failed to connect to bootstrap peer {peer.address}: {e}")
        
        if connected_count > 0:
            logger.info(f"Successfully bootstrapped with {connected_count} peers")
            # Perform initial peer discovery
            await self._discover_peers()
        else:
            logger.error("Failed to connect to any bootstrap peers")
    
    async def _load_bootstrap_peers(self) -> List[PeerNode]:
        """Load bootstrap peers from configuration or discovery"""
        peers = []
        
        for url in self.bootstrap_urls:
            try:
                # Parse bootstrap URL
                if '://' not in url:
                    url = f'http://{url}'
                
                parsed = urlparse(url)
                host = parsed.hostname
                port = parsed.port or 8467
                
                # Create bootstrap peer (we'll get the real node ID when connecting)
                peer = PeerNode(
                    node_id=f"bootstrap_{host}_{port}",
                    ip_address=host,
                    port=port,
                    public_key="unknown",  # Will be updated on connection
                    last_seen=time.time(),
                    is_bootstrap=True
                )
                peers.append(peer)
                
            except Exception as e:
                logger.warning(f"Invalid bootstrap URL {url}: {e}")
        
        return peers
    
    async def _connect_to_peer(self, peer: PeerNode) -> bool:
        """Establish connection with a peer"""
        if not self.session:
            return False
        
        try:
            # Send peer discovery request
            url = f"http://{peer.address}/api/peer/handshake"
            payload = {
                'node_id': self.node_id,
                'port': self.port,
                'region': self.geographic_region
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    peer_info = await response.json()
                    
                    # Update peer information
                    peer.node_id = peer_info.get('node_id', peer.node_id)
                    peer.public_key = peer_info.get('public_key', peer.public_key)
                    peer.geographic_region = peer_info.get('region')
                    peer.last_seen = time.time()
                    
                    # Add to DHT
                    self.dht.add_peer(peer)
                    return True
        
        except Exception as e:
            logger.debug(f"Connection to {peer.address} failed: {e}")
        
        return False
    
    async def _discover_peers(self):
        """Discover new peers through the DHT network"""
        if not self.dht.get_active_peers():
            return
        
        # Generate random target IDs for peer discovery
        targets = [
            hashlib.sha1(f"{self.node_id}:{i}:{time.time()}".encode()).hexdigest()
            for i in range(5)
        ]
        
        discovered_peers = set()
        
        for target in targets:
            closest_peers = self.dht.find_closest_peers(target, 10)
            
            for peer in closest_peers:
                if len(discovered_peers) >= self.max_peers:
                    break
                
                try:
                    # Query peer for their closest nodes to target
                    if self.session:
                        url = f"http://{peer.address}/api/peer/find_node"
                        payload = {'target': target, 'count': 20}
                        
                        async with self.session.post(url, json=payload) as response:
                            if response.status == 200:
                                peer_list = await response.json()
                                
                                for peer_data in peer_list.get('peers', []):
                                    try:
                                        new_peer = PeerNode.from_dict(peer_data)
                                        if new_peer.node_id not in discovered_peers:
                                            discovered_peers.add(new_peer.node_id)
                                            
                                            # Try to connect to new peer
                                            if await self._connect_to_peer(new_peer):
                                                logger.debug(f"Discovered new peer: {new_peer.address}")
                                    
                                    except Exception as e:
                                        logger.debug(f"Invalid peer data received: {e}")
                
                except Exception as e:
                    logger.debug(f"Peer discovery query to {peer.address} failed: {e}")
        
        logger.info(f"Peer discovery completed. Total active peers: {len(self.dht.get_active_peers())}")
    
    def _schedule_periodic_tasks(self):
        """Schedule periodic maintenance tasks"""
        # Periodic peer discovery
        discovery_task = asyncio.create_task(self._periodic_peer_discovery())
        self.discovery_tasks.add(discovery_task)
        
        # Peer health monitoring
        health_task = asyncio.create_task(self._periodic_health_check())
        self.discovery_tasks.add(health_task)
        
        # Network maintenance
        maintenance_task = asyncio.create_task(self._periodic_maintenance())
        self.discovery_tasks.add(maintenance_task)
    
    async def _periodic_peer_discovery(self):
        """Periodic peer discovery to maintain network connectivity"""
        while self.dht.running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._discover_peers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic peer discovery error: {e}")
    
    async def _periodic_health_check(self):
        """Periodic health check of known peers"""
        while self.dht.running:
            try:
                await asyncio.sleep(120)  # Run every 2 minutes
                await self._check_peer_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic health check error: {e}")
    
    async def _check_peer_health(self):
        """Check health of all known peers"""
        if not self.session:
            return
        
        peers_to_check = self.dht.get_all_peers()
        unhealthy_peers = []
        
        for peer in peers_to_check:
            try:
                url = f"http://{peer.address}/api/peer/ping"
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        peer.last_seen = time.time()
                        # Update latency measurement
                        # (In a real implementation, measure round-trip time)
                        peer.network_latency = 0.1  # Placeholder
                    else:
                        unhealthy_peers.append(peer)
            
            except Exception:
                unhealthy_peers.append(peer)
        
        # Remove unhealthy peers
        for peer in unhealthy_peers:
            if not peer.is_active:
                self.dht.remove_peer(peer.node_id)
                logger.debug(f"Removed unhealthy peer: {peer.address}")
    
    async def _periodic_maintenance(self):
        """Periodic maintenance tasks"""
        while self.dht.running:
            try:
                await asyncio.sleep(600)  # Run every 10 minutes
                await self._perform_maintenance()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic maintenance error: {e}")
    
    async def _perform_maintenance(self):
        """Perform network maintenance tasks"""
        active_peers = self.dht.get_active_peers()
        
        # Log network statistics
        logger.info(f"Network status - Active peers: {len(active_peers)}, "
                   f"Total known peers: {len(self.dht.get_all_peers())}")
        
        # If we have too few peers, try to discover more
        if len(active_peers) < 5:
            logger.info("Low peer count - initiating discovery")
            await self._discover_peers()
    
    # Public API methods
    
    def get_peers_for_region(self, region: str) -> List[PeerNode]:
        """Get peers in a specific geographic region"""
        return [
            peer for peer in self.dht.get_active_peers()
            if peer.geographic_region == region
        ]
    
    def get_optimal_peers(self, count: int = 10, 
                         prefer_region: bool = True) -> List[PeerNode]:
        """Get optimal peers for file distribution"""
        active_peers = self.dht.get_active_peers()
        
        if prefer_region and self.geographic_region != 'unknown':
            # Prefer peers in the same region
            regional_peers = [
                peer for peer in active_peers
                if peer.geographic_region == self.geographic_region
            ]
            
            if len(regional_peers) >= count:
                # Sort by reputation and latency
                regional_peers.sort(
                    key=lambda p: (p.reputation_score, -(p.network_latency or 1.0)),
                    reverse=True
                )
                return regional_peers[:count]
        
        # Fall back to all peers sorted by quality
        active_peers.sort(
            key=lambda p: (p.reputation_score, -(p.network_latency or 1.0)),
            reverse=True
        )
        
        return active_peers[:count]
    
    def find_peer_by_id(self, node_id: str) -> Optional[PeerNode]:
        """Find a peer by their node ID"""
        return self.dht.get_peer(node_id)
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get current network statistics"""
        active_peers = self.dht.get_active_peers()
        all_peers = self.dht.get_all_peers()
        
        regions = {}
        for peer in active_peers:
            region = peer.geographic_region or 'unknown'
            regions[region] = regions.get(region, 0) + 1
        
        return {
            'node_id': self.node_id,
            'active_peers': len(active_peers),
            'total_known_peers': len(all_peers),
            'regions': regions,
            'avg_reputation': sum(p.reputation_score for p in active_peers) / len(active_peers) if active_peers else 0,
            'avg_latency': sum(p.network_latency or 0 for p in active_peers) / len(active_peers) if active_peers else 0
        }


# Example usage and testing functions

async def example_node_discovery():
    """Example of how to use the node discovery system"""
    config = {
        'port': 8467,
        'max_peers': 30,
        'region': 'us-east-1',
        'bootstrap_nodes': [
            'demo1.prsm.network:8467',
            'demo2.prsm.network:8467'
        ]
    }
    
    discovery = NodeDiscovery(config)
    
    try:
        await discovery.start()
        
        # Wait for network discovery
        await asyncio.sleep(30)
        
        # Get network statistics
        stats = discovery.get_network_stats()
        print(f"Network Stats: {json.dumps(stats, indent=2)}")
        
        # Find optimal peers for file distribution
        optimal_peers = discovery.get_optimal_peers(5)
        print(f"Optimal peers: {[p.address for p in optimal_peers]}")
        
    finally:
        await discovery.stop()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    asyncio.run(example_node_discovery())