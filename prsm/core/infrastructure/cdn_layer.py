"""
PRSM Decentralized CDN Layer
============================

Transforms PRSM infrastructure into a high-performance, incentive-aligned
Content Delivery Network for scientific data and AI models.

Key Features:
- FTNS-incentivized bandwidth and storage contribution
- Latency-aware request routing with geographic optimization
- Dynamic content pinning based on access patterns
- Sybil-resistant validation and reward distribution
- Integration with institutional dynamics for enterprise-grade performance
"""

import asyncio
import hashlib
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass
from decimal import Decimal

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Types of CDN nodes in the PRSM network"""
    CORE_PRSM = "core_prsm"              # Full PRSM nodes with compute/storage
    EDGE_NODE = "edge_node"              # Bandwidth-optimized delivery nodes
    RESEARCH_INSTITUTION = "research_institution"  # University/lab supernodes
    ENTERPRISE_GATEWAY = "enterprise_gateway"      # Enterprise edge caches
    MICRO_CACHE = "micro_cache"          # Consumer devices with spare bandwidth


class ContentPriority(str, Enum):
    """Content priority levels for caching decisions"""
    CRITICAL = "critical"        # Core AI models, safety-critical data
    HIGH = "high"               # Popular research datasets, active models
    NORMAL = "normal"           # Standard scientific content
    LOW = "low"                # Archive content, rarely accessed data
    SPECULATIVE = "speculative" # Predictively cached content


@dataclass
class GeographicLocation:
    """Geographic location for latency optimization"""
    continent: str
    country: str
    region: str
    latitude: float
    longitude: float
    
    def distance_to(self, other: 'GeographicLocation') -> float:
        """Calculate approximate distance in kilometers"""
        # Simplified haversine formula
        import math
        
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return 6371 * c  # Earth's radius in km


@dataclass
class BandwidthMetrics:
    """Bandwidth and performance metrics for CDN optimization"""
    download_mbps: float
    upload_mbps: float
    latency_ms: float
    packet_loss_rate: float
    uptime_percentage: float
    geographic_location: GeographicLocation


class ContentItem(BaseModel):
    """Content item in the PRSM CDN"""
    content_hash: str  # IPFS hash
    content_type: str  # "model", "dataset", "paper", "embedding"
    size_bytes: int
    priority: ContentPriority
    
    # Access patterns
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    access_pattern: List[Tuple[datetime, str]] = Field(default_factory=list)  # (time, region)
    
    # Metadata
    scientific_domain: Optional[str] = None
    model_architecture: Optional[str] = None
    research_institution: Optional[str] = None


class CDNNode(BaseModel):
    """CDN node in the PRSM network"""
    node_id: UUID = Field(default_factory=uuid4)
    node_type: NodeType
    operator_id: UUID
    
    # Technical capabilities
    storage_capacity_gb: float
    bandwidth_metrics: BandwidthMetrics
    
    # CDN-specific metrics
    cached_content: Dict[str, ContentItem] = Field(default_factory=dict)
    total_requests_served: int = 0
    total_bytes_served: int = 0
    
    # Performance tracking
    average_response_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    geographic_coverage_score: float = 0.0
    
    # Economic tracking
    ftns_earned_total: Decimal = Field(default=Decimal('0'))
    ftns_earned_last_period: Decimal = Field(default=Decimal('0'))
    
    # Status
    is_active: bool = True
    last_ping: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RequestRoute(BaseModel):
    """Optimal routing decision for content request"""
    content_hash: str
    client_location: GeographicLocation
    
    # Selected nodes (primary + fallbacks)
    primary_node: UUID
    fallback_nodes: List[UUID] = Field(default_factory=list)
    
    # Performance estimates
    estimated_latency_ms: float
    estimated_bandwidth_mbps: float
    cache_probability: float
    
    # Economic considerations
    ftns_cost: Decimal
    node_rewards: Dict[UUID, Decimal] = Field(default_factory=dict)


class PRSMCDNLayer:
    """
    Decentralized CDN layer that transforms PRSM infrastructure into
    a high-performance, incentive-aligned content delivery network.
    """
    
    def __init__(self):
        # Node registry
        self.cdn_nodes: Dict[UUID, CDNNode] = {}
        self.content_registry: Dict[str, ContentItem] = {}
        
        # Routing optimization
        self.geographic_clusters: Dict[str, List[UUID]] = {}  # region -> node_ids
        self.performance_cache: Dict[str, float] = {}  # node_id -> recent_performance
        
        # Economic parameters
        self.reward_rates = {
            "bandwidth_per_gb": Decimal('0.01'),    # FTNS per GB served
            "latency_bonus_per_ms": Decimal('0.001'), # Bonus for low latency
            "cache_hit_bonus": Decimal('0.05'),     # Bonus per cache hit
            "geographic_diversity_bonus": Decimal('0.1'), # Bonus for underserved regions
        }
        
        # Content caching parameters
        self.cache_policies = {
            ContentPriority.CRITICAL: {"min_replicas": 10, "eviction_protection": True},
            ContentPriority.HIGH: {"min_replicas": 5, "eviction_protection": False},
            ContentPriority.NORMAL: {"min_replicas": 3, "eviction_protection": False},
            ContentPriority.LOW: {"min_replicas": 1, "eviction_protection": False},
            ContentPriority.SPECULATIVE: {"min_replicas": 1, "eviction_protection": False},
        }
        
        print("🌐 PRSM CDN Layer initialized")
        print("   - Decentralized content delivery active")
        print("   - FTNS-incentivized bandwidth sharing enabled")
        print("   - Geographic optimization ready")
    
    async def register_cdn_node(self,
                               node_type: NodeType,
                               operator_id: UUID,
                               storage_capacity_gb: float,
                               bandwidth_metrics: BandwidthMetrics) -> CDNNode:
        """
        Register a new node in the PRSM CDN network.
        """
        
        # Create CDN node
        node = CDNNode(
            node_type=node_type,
            operator_id=operator_id,
            storage_capacity_gb=storage_capacity_gb,
            bandwidth_metrics=bandwidth_metrics
        )
        
        # Register node
        self.cdn_nodes[node.node_id] = node
        
        # Update geographic clustering
        region = bandwidth_metrics.geographic_location.region
        if region not in self.geographic_clusters:
            self.geographic_clusters[region] = []
        self.geographic_clusters[region].append(node.node_id)
        
        # Calculate initial geographic coverage score
        node.geographic_coverage_score = await self._calculate_geographic_coverage(node)
        
        print(f"🌐 CDN node registered: {node_type}")
        print(f"   - Node ID: {node.node_id}")
        print(f"   - Capacity: {storage_capacity_gb} GB")
        print(f"   - Bandwidth: {bandwidth_metrics.download_mbps} Mbps")
        print(f"   - Location: {bandwidth_metrics.geographic_location.region}")
        
        return node
    
    async def optimize_content_routing(self,
                                     content_hash: str,
                                     client_location: GeographicLocation,
                                     performance_requirements: Dict[str, Any] = None) -> RequestRoute:
        """
        Find optimal routing for content request based on latency, cost, and availability.
        """
        
        if content_hash not in self.content_registry:
            raise ValueError(f"Content {content_hash} not found in registry")
        
        content_item = self.content_registry[content_hash]
        
        # Find nodes that have this content cached
        available_nodes = []
        for node_id, node in self.cdn_nodes.items():
            if not node.is_active:
                continue
            
            if content_hash in node.cached_content:
                # Calculate routing score
                distance = node.bandwidth_metrics.geographic_location.distance_to(client_location)
                latency_estimate = max(50, distance * 0.1 + node.bandwidth_metrics.latency_ms)
                
                # Performance score (lower latency = higher score)
                performance_score = 1000 / (latency_estimate + 1)
                
                # Economic score (consider FTNS costs)
                economic_score = 100 / (float(self.reward_rates["bandwidth_per_gb"]) + 1)
                
                # Reliability score
                reliability_score = node.bandwidth_metrics.uptime_percentage * node.cache_hit_rate
                
                # Combined score
                total_score = performance_score + economic_score + reliability_score
                
                available_nodes.append({
                    "node_id": node_id,
                    "node": node,
                    "latency_estimate": latency_estimate,
                    "total_score": total_score,
                    "cache_probability": 1.0  # Already cached
                })
        
        # If no nodes have content cached, find best nodes to fetch and cache
        if not available_nodes:
            available_nodes = await self._find_optimal_caching_nodes(content_item, client_location)
        
        # Sort by total score (best first)
        available_nodes.sort(key=lambda x: x["total_score"], reverse=True)
        
        if not available_nodes:
            raise RuntimeError("No suitable CDN nodes available")
        
        # Select primary and fallback nodes
        primary = available_nodes[0]
        fallbacks = [node["node_id"] for node in available_nodes[1:3]]  # Top 2 fallbacks
        
        # Calculate costs
        content_size_gb = content_item.size_bytes / (1024**3)
        ftns_cost = self.reward_rates["bandwidth_per_gb"] * Decimal(str(content_size_gb))
        
        # Create routing decision
        route = RequestRoute(
            content_hash=content_hash,
            client_location=client_location,
            primary_node=primary["node_id"],
            fallback_nodes=fallbacks,
            estimated_latency_ms=primary["latency_estimate"],
            estimated_bandwidth_mbps=primary["node"].bandwidth_metrics.download_mbps,
            cache_probability=primary["cache_probability"],
            ftns_cost=ftns_cost,
            node_rewards={primary["node_id"]: ftns_cost}
        )
        
        return route
    
    async def serve_content_request(self,
                                  route: RequestRoute,
                                  actual_bytes_served: int,
                                  actual_latency_ms: float) -> Dict[str, Any]:
        """
        Record successful content serving and distribute FTNS rewards.
        """
        
        primary_node = self.cdn_nodes[route.primary_node]
        content_item = self.content_registry[route.content_hash]
        
        # Update node performance metrics
        primary_node.total_requests_served += 1
        primary_node.total_bytes_served += actual_bytes_served
        
        # Update average response time (exponential moving average)
        alpha = 0.1  # Smoothing factor
        primary_node.average_response_time_ms = (
            (1 - alpha) * primary_node.average_response_time_ms +
            alpha * actual_latency_ms
        )
        
        # Calculate FTNS rewards
        base_reward = self.reward_rates["bandwidth_per_gb"] * Decimal(str(actual_bytes_served / (1024**3)))
        
        # Performance bonuses
        latency_bonus = Decimal('0')
        if actual_latency_ms < route.estimated_latency_ms:
            improvement_ms = route.estimated_latency_ms - actual_latency_ms
            latency_bonus = self.reward_rates["latency_bonus_per_ms"] * Decimal(str(improvement_ms))
        
        # Cache hit bonus
        cache_bonus = Decimal('0')
        if route.cache_probability > 0.9:  # Was already cached
            cache_bonus = self.reward_rates["cache_hit_bonus"]
        
        # Geographic diversity bonus
        geo_bonus = Decimal('0')
        if primary_node.geographic_coverage_score > 0.8:  # Serving underrepresented region
            geo_bonus = self.reward_rates["geographic_diversity_bonus"]
        
        total_reward = base_reward + latency_bonus + cache_bonus + geo_bonus
        
        # Award FTNS
        primary_node.ftns_earned_total += total_reward
        primary_node.ftns_earned_last_period += total_reward
        
        # Update content access patterns
        content_item.access_count += 1
        content_item.last_accessed = datetime.now(timezone.utc)
        content_item.access_pattern.append((
            datetime.now(timezone.utc),
            route.client_location.region
        ))
        
        print(f"💰 CDN request served and rewarded")
        print(f"   - Node: {route.primary_node}")
        print(f"   - Bytes: {actual_bytes_served:,}")
        print(f"   - Latency: {actual_latency_ms:.1f}ms")
        print(f"   - FTNS reward: {total_reward:.4f}")
        
        return {
            "success": True,
            "node_id": route.primary_node,
            "bytes_served": actual_bytes_served,
            "latency_ms": actual_latency_ms,
            "ftns_reward": total_reward,
            "performance_bonuses": {
                "latency_bonus": latency_bonus,
                "cache_bonus": cache_bonus,
                "geographic_bonus": geo_bonus
            }
        }
    
    async def optimize_content_caching(self) -> Dict[str, Any]:
        """
        Optimize content caching across the network based on access patterns.
        """
        
        optimizations = {
            "cache_migrations": [],
            "new_replications": [],
            "evictions": [],
            "ftns_savings": Decimal('0')
        }
        
        # Analyze access patterns
        for content_hash, content_item in self.content_registry.items():
            
            # Determine optimal caching locations based on access pattern
            regional_access = {}
            for access_time, region in content_item.access_pattern[-100:]:  # Last 100 accesses
                regional_access[region] = regional_access.get(region, 0) + 1
            
            # Find underserved high-access regions
            for region, access_count in regional_access.items():
                if access_count > 10:  # High access region
                    
                    # Check if we have enough nodes in this region
                    regional_nodes = self.geographic_clusters.get(region, [])
                    cached_nodes = [
                        node_id for node_id in regional_nodes
                        if content_hash in self.cdn_nodes[node_id].cached_content
                    ]
                    
                    min_replicas = self.cache_policies[content_item.priority]["min_replicas"]
                    if len(cached_nodes) < min_replicas:
                        # Need more replicas in this region
                        target_nodes = [
                            node_id for node_id in regional_nodes
                            if (node_id not in cached_nodes and
                                self._has_storage_capacity(node_id, content_item))
                        ]
                        
                        if target_nodes:
                            optimizations["new_replications"].append({
                                "content_hash": content_hash,
                                "target_region": region,
                                "target_nodes": target_nodes[:min_replicas - len(cached_nodes)],
                                "reason": f"High access ({access_count}) in underserved region"
                            })
        
        # Identify candidates for eviction (low-access content in over-replicated regions)
        for content_hash, content_item in self.content_registry.items():
            if (content_item.last_accessed and
                (datetime.now(timezone.utc) - content_item.last_accessed).days > 30 and
                content_item.priority in [ContentPriority.LOW, ContentPriority.SPECULATIVE]):
                
                # Find nodes that could evict this content
                eviction_candidates = []
                for node_id, node in self.cdn_nodes.items():
                    if content_hash in node.cached_content:
                        eviction_candidates.append(node_id)
                
                # Keep minimum replicas, evict from others
                min_replicas = self.cache_policies[content_item.priority]["min_replicas"]
                if len(eviction_candidates) > min_replicas:
                    nodes_to_evict = eviction_candidates[min_replicas:]
                    optimizations["evictions"].append({
                        "content_hash": content_hash,
                        "nodes_to_evict": nodes_to_evict,
                        "storage_freed_gb": content_item.size_bytes / (1024**3) * len(nodes_to_evict),
                        "reason": "Low access, over-replicated"
                    })
        
        print(f"🔧 CDN caching optimization completed")
        print(f"   - New replications: {len(optimizations['new_replications'])}")
        print(f"   - Evictions: {len(optimizations['evictions'])}")
        
        return optimizations
    
    async def get_network_health(self) -> Dict[str, Any]:
        """
        Get comprehensive CDN network health metrics.
        """
        
        # Node distribution
        total_nodes = len(self.cdn_nodes)
        active_nodes = sum(1 for node in self.cdn_nodes.values() if node.is_active)
        
        # Geographic distribution
        regions = set()
        for node in self.cdn_nodes.values():
            regions.add(node.bandwidth_metrics.geographic_location.region)
        
        # Performance metrics
        total_requests = sum(node.total_requests_served for node in self.cdn_nodes.values())
        total_bytes = sum(node.total_bytes_served for node in self.cdn_nodes.values())
        avg_latency = sum(node.average_response_time_ms for node in self.cdn_nodes.values()) / max(total_nodes, 1)
        avg_cache_hit_rate = sum(node.cache_hit_rate for node in self.cdn_nodes.values()) / max(total_nodes, 1)
        
        # Economic metrics
        total_ftns_distributed = sum(node.ftns_earned_total for node in self.cdn_nodes.values())
        
        # Content metrics
        total_content_items = len(self.content_registry)
        total_content_size_gb = sum(item.size_bytes for item in self.content_registry.values()) / (1024**3)
        
        return {
            "network_health": {
                "total_nodes": total_nodes,
                "active_nodes": active_nodes,
                "geographic_regions": len(regions),
                "node_uptime_avg": sum(n.bandwidth_metrics.uptime_percentage for n in self.cdn_nodes.values()) / max(total_nodes, 1)
            },
            "performance_metrics": {
                "total_requests_served": total_requests,
                "total_bytes_served_gb": total_bytes / (1024**3),
                "average_latency_ms": avg_latency,
                "average_cache_hit_rate": avg_cache_hit_rate * 100
            },
            "economic_metrics": {
                "total_ftns_distributed": total_ftns_distributed,
                "avg_ftns_per_node": total_ftns_distributed / max(total_nodes, 1),
                "revenue_per_gb": total_ftns_distributed / max(total_bytes / (1024**3), 1)
            },
            "content_metrics": {
                "total_content_items": total_content_items,
                "total_content_size_gb": total_content_size_gb,
                "avg_content_replicas": self._calculate_avg_replicas()
            }
        }
    
    def _has_storage_capacity(self, node_id: UUID, content_item: ContentItem) -> bool:
        """Check if node has capacity for additional content"""
        node = self.cdn_nodes[node_id]
        current_usage = sum(item.size_bytes for item in node.cached_content.values())
        capacity_bytes = node.storage_capacity_gb * (1024**3)
        return (current_usage + content_item.size_bytes) < (capacity_bytes * 0.9)  # 90% max usage
    
    async def _calculate_geographic_coverage(self, node: CDNNode) -> float:
        """Calculate how well this node serves underrepresented geographic areas"""
        node_region = node.bandwidth_metrics.geographic_location.region
        regional_nodes = self.geographic_clusters.get(node_region, [])
        
        # Score based on how few nodes serve this region
        if len(regional_nodes) <= 2:
            return 1.0  # High value for underserved regions
        elif len(regional_nodes) <= 5:
            return 0.8
        elif len(regional_nodes) <= 10:
            return 0.5
        else:
            return 0.2  # Lower value for well-served regions
    
    async def _find_optimal_caching_nodes(self,
                                        content_item: ContentItem,
                                        client_location: GeographicLocation) -> List[Dict[str, Any]]:
        """Find best nodes to cache content that isn't currently cached"""
        
        candidates = []
        for node_id, node in self.cdn_nodes.items():
            if not node.is_active:
                continue
            
            if not self._has_storage_capacity(node_id, content_item):
                continue
            
            # Calculate metrics for uncached content
            distance = node.bandwidth_metrics.geographic_location.distance_to(client_location)
            latency_estimate = max(100, distance * 0.2 + node.bandwidth_metrics.latency_ms)  # Higher for fetch
            
            candidates.append({
                "node_id": node_id,
                "node": node,
                "latency_estimate": latency_estimate,
                "total_score": 500 / (latency_estimate + 1),  # Lower score for fetch operations
                "cache_probability": 0.0  # Will need to fetch
            })
        
        return candidates
    
    def _calculate_avg_replicas(self) -> float:
        """Calculate average number of replicas per content item"""
        if not self.content_registry:
            return 0.0
        
        total_replicas = 0
        for content_hash in self.content_registry:
            replica_count = sum(
                1 for node in self.cdn_nodes.values()
                if content_hash in node.cached_content
            )
            total_replicas += replica_count
        
        return total_replicas / len(self.content_registry)


# Global PRSM CDN layer instance
prsm_cdn = PRSMCDNLayer()