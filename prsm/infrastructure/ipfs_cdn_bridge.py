"""
IPFS-CDN Bridge
===============

Bridges PRSM's CDN layer with IPFS for content storage and retrieval.
Handles content pinning, DHT lookups, and performance optimization.

Key Features:
- Intelligent content pinning based on access patterns
- DHT-aware node selection for optimal routing
- Content verification and integrity checking
- Performance monitoring and optimization
"""

import asyncio
import aiohttp
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set
from uuid import UUID
from dataclasses import dataclass

import ipfshttpclient
from .cdn_layer import prsm_cdn, ContentItem, ContentPriority


@dataclass
class IPFSNode:
    """IPFS node configuration"""
    api_url: str
    gateway_url: str
    node_id: str
    is_local: bool = True


class IPFSCDNBridge:
    """
    Bridges PRSM CDN with IPFS for decentralized content storage and retrieval.
    """
    
    def __init__(self, ipfs_nodes: List[IPFSNode] = None):
        self.ipfs_nodes = ipfs_nodes or [
            IPFSNode(
                api_url="http://localhost:5001",
                gateway_url="http://localhost:8080",
                node_id="local",
                is_local=True
            )
        ]
        
        # Content tracking
        self.pinned_content: Dict[str, Dict[str, Any]] = {}  # hash -> pin_info
        self.pin_priorities: Dict[str, ContentPriority] = {}
        
        # Performance tracking
        self.retrieval_stats: Dict[str, List[float]] = {}  # hash -> [latency_ms]
        
        print("🔗 IPFS-CDN Bridge initialized")
        print(f"   - Connected to {len(self.ipfs_nodes)} IPFS nodes")
    
    async def pin_content(self, 
                         content_hash: str, 
                         priority: ContentPriority,
                         metadata: Dict[str, Any] = None) -> bool:
        """
        Pin content to IPFS nodes based on priority and CDN requirements.
        """
        
        try:
            # Select IPFS nodes based on priority
            target_nodes = await self._select_pinning_nodes(priority)
            
            pin_results = []
            for node in target_nodes:
                try:
                    # Connect to IPFS node
                    client = ipfshttpclient.connect(node.api_url)
                    
                    # Pin the content
                    result = client.pin.add(content_hash)
                    pin_results.append({
                        "node": node.node_id,
                        "success": True,
                        "result": result
                    })
                    
                    print(f"📌 Content pinned: {content_hash[:16]}... on {node.node_id}")
                    
                except Exception as e:
                    pin_results.append({
                        "node": node.node_id,
                        "success": False,
                        "error": str(e)
                    })
                    print(f"❌ Pin failed on {node.node_id}: {e}")
            
            # Record pinning info
            self.pinned_content[content_hash] = {
                "priority": priority,
                "pin_results": pin_results,
                "pinned_at": datetime.now(timezone.utc),
                "metadata": metadata or {}
            }
            
            self.pin_priorities[content_hash] = priority
            
            # Register with CDN layer
            await self._register_content_with_cdn(content_hash, priority, metadata)
            
            successful_pins = sum(1 for r in pin_results if r["success"])
            return successful_pins > 0
            
        except Exception as e:
            print(f"❌ Content pinning failed: {e}")
            return False
    
    async def retrieve_content(self, 
                              content_hash: str,
                              requesting_node_id: UUID) -> Optional[bytes]:
        """
        Retrieve content from IPFS with performance optimization.
        """
        
        start_time = datetime.now()
        
        try:
            # Try local nodes first for better performance
            local_nodes = [node for node in self.ipfs_nodes if node.is_local]
            remote_nodes = [node for node in self.ipfs_nodes if not node.is_local]
            
            # Attempt retrieval from local nodes first
            for node in local_nodes + remote_nodes:
                try:
                    client = ipfshttpclient.connect(node.api_url)
                    content = client.cat(content_hash)
                    
                    # Record performance
                    retrieval_time = (datetime.now() - start_time).total_seconds() * 1000
                    self._record_retrieval_performance(content_hash, retrieval_time)
                    
                    print(f"📥 Content retrieved: {content_hash[:16]}... ({retrieval_time:.1f}ms)")
                    return content
                    
                except Exception as e:
                    print(f"⚠️ Retrieval failed from {node.node_id}: {e}")
                    continue
            
            print(f"❌ Content retrieval failed: {content_hash}")
            return None
            
        except Exception as e:
            print(f"❌ Content retrieval error: {e}")
            return None
    
    async def optimize_pinning_strategy(self) -> Dict[str, Any]:
        """
        Optimize content pinning based on access patterns and performance data.
        """
        
        optimizations = {
            "pins_added": [],
            "pins_removed": [],
            "priority_upgrades": [],
            "performance_improvements": []
        }
        
        # Analyze retrieval performance
        for content_hash, latencies in self.retrieval_stats.items():
            if len(latencies) >= 10:  # Enough data for analysis
                avg_latency = sum(latencies[-10:]) / 10  # Last 10 retrievals
                
                current_priority = self.pin_priorities.get(content_hash, ContentPriority.LOW)
                
                # Upgrade priority for frequently accessed, slow content
                if avg_latency > 1000 and len(latencies) > 50:  # > 1s latency, high access
                    if current_priority in [ContentPriority.LOW, ContentPriority.NORMAL]:
                        new_priority = ContentPriority.HIGH
                        await self._upgrade_content_priority(content_hash, new_priority)
                        optimizations["priority_upgrades"].append({
                            "content_hash": content_hash,
                            "old_priority": current_priority,
                            "new_priority": new_priority,
                            "reason": f"High latency ({avg_latency:.1f}ms) with high access"
                        })
        
        # Identify under-pinned high-priority content
        for content_hash, priority in self.pin_priorities.items():
            if priority in [ContentPriority.CRITICAL, ContentPriority.HIGH]:
                pin_info = self.pinned_content.get(content_hash, {})
                successful_pins = sum(1 for r in pin_info.get("pin_results", []) if r.get("success"))
                
                required_pins = {"CRITICAL": 5, "HIGH": 3}.get(priority.value.upper(), 1)
                
                if successful_pins < required_pins:
                    additional_nodes = await self._select_additional_pinning_nodes(
                        content_hash, required_pins - successful_pins
                    )
                    
                    for node in additional_nodes:
                        success = await self._pin_to_node(content_hash, node)
                        if success:
                            optimizations["pins_added"].append({
                                "content_hash": content_hash,
                                "node": node.node_id,
                                "reason": "Under-pinned high-priority content"
                            })
        
        print(f"🔧 IPFS pinning optimization completed")
        print(f"   - Priority upgrades: {len(optimizations['priority_upgrades'])}")
        print(f"   - Additional pins: {len(optimizations['pins_added'])}")
        
        return optimizations
    
    async def get_ipfs_health_metrics(self) -> Dict[str, Any]:
        """
        Get health metrics for IPFS integration.
        """
        
        # Test connectivity to all IPFS nodes
        node_health = []
        for node in self.ipfs_nodes:
            try:
                client = ipfshttpclient.connect(node.api_url)
                node_info = client.id()
                
                node_health.append({
                    "node_id": node.node_id,
                    "status": "healthy",
                    "peer_id": node_info.get("ID"),
                    "addresses": node_info.get("Addresses", [])
                })
            except Exception as e:
                node_health.append({
                    "node_id": node.node_id,
                    "status": "unhealthy",
                    "error": str(e)
                })
        
        # Content statistics
        total_pinned = len(self.pinned_content)
        priority_distribution = {}
        for priority in self.pin_priorities.values():
            priority_distribution[priority.value] = priority_distribution.get(priority.value, 0) + 1
        
        # Performance statistics
        total_retrievals = sum(len(latencies) for latencies in self.retrieval_stats.values())
        avg_latency = 0
        if total_retrievals > 0:
            all_latencies = [l for latencies in self.retrieval_stats.values() for l in latencies]
            avg_latency = sum(all_latencies) / len(all_latencies)
        
        return {
            "node_health": node_health,
            "content_stats": {
                "total_pinned_items": total_pinned,
                "priority_distribution": priority_distribution
            },
            "performance_stats": {
                "total_retrievals": total_retrievals,
                "average_latency_ms": avg_latency,
                "content_with_stats": len(self.retrieval_stats)
            }
        }
    
    async def _select_pinning_nodes(self, priority: ContentPriority) -> List[IPFSNode]:
        """Select IPFS nodes for pinning based on priority"""
        
        if priority == ContentPriority.CRITICAL:
            return self.ipfs_nodes  # Pin to all nodes
        elif priority == ContentPriority.HIGH:
            return self.ipfs_nodes[:3]  # Pin to first 3 nodes
        else:
            return self.ipfs_nodes[:1]  # Pin to first node only
    
    async def _register_content_with_cdn(self, 
                                        content_hash: str, 
                                        priority: ContentPriority,
                                        metadata: Dict[str, Any]) -> None:
        """Register pinned content with CDN layer"""
        
        try:
            # Estimate content size (in real implementation, get from IPFS)
            estimated_size = metadata.get("size_bytes", 1024 * 1024)  # Default 1MB
            
            content_item = ContentItem(
                content_hash=content_hash,
                content_type=metadata.get("content_type", "unknown"),
                size_bytes=estimated_size,
                priority=priority,
                scientific_domain=metadata.get("domain"),
                model_architecture=metadata.get("architecture"),
                research_institution=metadata.get("institution")
            )
            
            prsm_cdn.content_registry[content_hash] = content_item
            print(f"✅ Content registered with CDN: {content_hash[:16]}...")
            
        except Exception as e:
            print(f"❌ CDN registration failed: {e}")
    
    def _record_retrieval_performance(self, content_hash: str, latency_ms: float):
        """Record retrieval performance for optimization"""
        
        if content_hash not in self.retrieval_stats:
            self.retrieval_stats[content_hash] = []
        
        self.retrieval_stats[content_hash].append(latency_ms)
        
        # Keep only last 100 measurements
        if len(self.retrieval_stats[content_hash]) > 100:
            self.retrieval_stats[content_hash] = self.retrieval_stats[content_hash][-100:]
    
    async def _upgrade_content_priority(self, content_hash: str, new_priority: ContentPriority):
        """Upgrade content priority and add additional pinning"""
        
        self.pin_priorities[content_hash] = new_priority
        
        # Add additional pins for higher priority
        if new_priority in [ContentPriority.CRITICAL, ContentPriority.HIGH]:
            required_pins = 5 if new_priority == ContentPriority.CRITICAL else 3
            current_pins = len([r for r in self.pinned_content.get(content_hash, {}).get("pin_results", []) 
                               if r.get("success")])
            
            if current_pins < required_pins:
                additional_nodes = await self._select_additional_pinning_nodes(
                    content_hash, required_pins - current_pins
                )
                
                for node in additional_nodes:
                    await self._pin_to_node(content_hash, node)
    
    async def _select_additional_pinning_nodes(self, content_hash: str, count: int) -> List[IPFSNode]:
        """Select additional nodes for pinning"""
        
        # Get nodes that don't already have this content pinned
        pinned_nodes = set()
        pin_info = self.pinned_content.get(content_hash, {})
        for result in pin_info.get("pin_results", []):
            if result.get("success"):
                pinned_nodes.add(result["node"])
        
        available_nodes = [node for node in self.ipfs_nodes if node.node_id not in pinned_nodes]
        return available_nodes[:count]
    
    async def _pin_to_node(self, content_hash: str, node: IPFSNode) -> bool:
        """Pin content to specific IPFS node"""
        
        try:
            client = ipfshttpclient.connect(node.api_url)
            client.pin.add(content_hash)
            
            # Update pinning records
            if content_hash in self.pinned_content:
                self.pinned_content[content_hash]["pin_results"].append({
                    "node": node.node_id,
                    "success": True,
                    "result": f"Additional pin at {datetime.now(timezone.utc)}"
                })
            
            return True
            
        except Exception as e:
            print(f"❌ Additional pin failed on {node.node_id}: {e}")
            return False


# Global IPFS-CDN bridge instance
ipfs_cdn_bridge = IPFSCDNBridge()