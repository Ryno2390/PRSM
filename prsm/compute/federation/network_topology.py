"""
Network Topology Optimization for PRSM
Implements intelligent network topologies for optimal consensus scaling efficiency
"""

import asyncio
import math
import statistics
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from uuid import UUID, uuid4
from enum import Enum
import networkx as nx

from prsm.core.config import settings
from prsm.core.models import PeerNode, AgentResponse, SafetyFlag, SafetyLevel
from prsm.core.safety.circuit_breaker import CircuitBreakerNetwork, ThreatLevel
from prsm.core.safety.monitor import SafetyMonitor
from .consensus import DistributedConsensus, ConsensusResult, ConsensusType


# === Network Topology Configuration ===

# Topology optimization settings
ENABLE_TOPOLOGY_OPTIMIZATION = getattr(settings, "PRSM_TOPOLOGY_OPTIMIZATION", True)
TOPOLOGY_UPDATE_INTERVAL = int(getattr(settings, "PRSM_TOPOLOGY_UPDATE_INTERVAL", 60))  # seconds
MIN_CONNECTIVITY = float(getattr(settings, "PRSM_MIN_CONNECTIVITY", 0.5))  # 50% connectivity
OPTIMAL_CONNECTIVITY = float(getattr(settings, "PRSM_OPTIMAL_CONNECTIVITY", 0.7))  # 70% connectivity

# Performance thresholds
LATENCY_THRESHOLD_MS = int(getattr(settings, "PRSM_LATENCY_THRESHOLD", 100))
BANDWIDTH_THRESHOLD_MBPS = int(getattr(settings, "PRSM_BANDWIDTH_THRESHOLD", 10))
RELIABILITY_THRESHOLD = float(getattr(settings, "PRSM_RELIABILITY_THRESHOLD", 0.95))

# Topology strategies
SMALL_NETWORK_THRESHOLD = int(getattr(settings, "PRSM_SMALL_NETWORK_THRESHOLD", 10))
MEDIUM_NETWORK_THRESHOLD = int(getattr(settings, "PRSM_MEDIUM_NETWORK_THRESHOLD", 50))
LARGE_NETWORK_THRESHOLD = int(getattr(settings, "PRSM_LARGE_NETWORK_THRESHOLD", 200))


class TopologyType(Enum):
    """Network topology types for different scaling scenarios"""
    FULL_MESH = "full_mesh"           # Complete connectivity (small networks)
    SMALL_WORLD = "small_world"       # Small-world properties (medium networks)
    SCALE_FREE = "scale_free"         # Hub-based distribution (large networks)
    HIERARCHICAL = "hierarchical"     # Tree-like structure (very large networks)
    ADAPTIVE = "adaptive"             # Dynamic topology optimization
    HYBRID = "hybrid"                 # Combination of strategies


class NetworkMetrics:
    """Network performance and topology metrics"""
    
    def __init__(self):
        # Connectivity metrics
        self.connectivity_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.node_degrees: Dict[str, int] = {}
        self.clustering_coefficients: Dict[str, float] = {}
        
        # Performance metrics
        self.latencies: Dict[Tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=100))
        self.bandwidths: Dict[Tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=100))
        self.reliabilities: Dict[Tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Topology properties
        self.diameter: Optional[int] = None
        self.average_path_length: Optional[float] = None
        self.global_clustering: Optional[float] = None
        self.assortativity: Optional[float] = None
        
        # Performance history
        self.performance_history: deque = deque(maxlen=1000)
        self.topology_changes: deque = deque(maxlen=200)
        
        self.last_update = datetime.now(timezone.utc)
    
    def add_connection_metric(self, node_a: str, node_b: str, 
                            latency_ms: float, bandwidth_mbps: float, reliability: float):
        """Add connection performance metrics"""
        edge = (min(node_a, node_b), max(node_a, node_b))
        
        self.latencies[edge].append(latency_ms)
        self.bandwidths[edge].append(bandwidth_mbps)
        self.reliabilities[edge].append(reliability)
        
        # Update connectivity matrix
        self.connectivity_matrix[node_a][node_b] = reliability
        self.connectivity_matrix[node_b][node_a] = reliability
    
    def get_connection_quality(self, node_a: str, node_b: str) -> Dict[str, float]:
        """Get connection quality metrics between two nodes"""
        edge = (min(node_a, node_b), max(node_a, node_b))
        
        if edge not in self.latencies:
            return {"latency_ms": float('inf'), "bandwidth_mbps": 0.0, "reliability": 0.0}
        
        return {
            "latency_ms": statistics.mean(self.latencies[edge]) if self.latencies[edge] else float('inf'),
            "bandwidth_mbps": statistics.mean(self.bandwidths[edge]) if self.bandwidths[edge] else 0.0,
            "reliability": statistics.mean(self.reliabilities[edge]) if self.reliabilities[edge] else 0.0
        }
    
    def calculate_topology_properties(self, graph: nx.Graph):
        """Calculate graph-theoretic topology properties"""
        try:
            if len(graph.nodes()) == 0:
                return
            
            # Basic connectivity
            self.node_degrees = dict(graph.degree())
            
            # Path-based metrics
            if nx.is_connected(graph):
                self.diameter = nx.diameter(graph)
                self.average_path_length = nx.average_shortest_path_length(graph)
            else:
                # For disconnected graphs, calculate for largest component
                largest_cc = max(nx.connected_components(graph), key=len)
                subgraph = graph.subgraph(largest_cc)
                if len(subgraph.nodes()) > 1:
                    self.diameter = nx.diameter(subgraph)
                    self.average_path_length = nx.average_shortest_path_length(subgraph)
            
            # Clustering
            clustering = nx.clustering(graph)
            self.clustering_coefficients = clustering
            self.global_clustering = nx.average_clustering(graph)
            
            # Assortativity
            try:
                self.assortativity = nx.degree_assortativity_coefficient(graph)
            except:
                self.assortativity = 0.0
                
        except Exception as e:
            print(f"❌ Error calculating topology properties: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive network performance summary"""
        try:
            # Calculate average metrics
            all_latencies = [lat for edge_lats in self.latencies.values() for lat in edge_lats]
            all_bandwidths = [bw for edge_bws in self.bandwidths.values() for bw in edge_bws]
            all_reliabilities = [rel for edge_rels in self.reliabilities.values() for rel in edge_rels]
            
            avg_latency = statistics.mean(all_latencies) if all_latencies else 0.0
            avg_bandwidth = statistics.mean(all_bandwidths) if all_bandwidths else 0.0
            avg_reliability = statistics.mean(all_reliabilities) if all_reliabilities else 0.0
            
            # Network connectivity
            total_possible_edges = len(self.node_degrees) * (len(self.node_degrees) - 1) // 2
            actual_edges = sum(self.node_degrees.values()) // 2
            connectivity_ratio = actual_edges / max(1, total_possible_edges)
            
            return {
                "network_size": len(self.node_degrees),
                "connectivity_ratio": connectivity_ratio,
                "average_degree": statistics.mean(self.node_degrees.values()) if self.node_degrees else 0.0,
                "diameter": self.diameter,
                "average_path_length": self.average_path_length,
                "global_clustering": self.global_clustering,
                "assortativity": self.assortativity,
                "average_latency_ms": avg_latency,
                "average_bandwidth_mbps": avg_bandwidth,
                "average_reliability": avg_reliability,
                "topology_changes": len(self.topology_changes),
                "last_update": self.last_update.isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error generating performance summary: {e}")
            return {"error": str(e)}


class TopologyOptimizer:
    """Intelligent network topology optimization for scaling efficiency"""
    
    def __init__(self):
        self.current_topology: Optional[nx.Graph] = None
        self.topology_type = TopologyType.ADAPTIVE
        self.network_metrics = NetworkMetrics()
        
        # Node management
        self.active_nodes: Dict[str, PeerNode] = {}
        self.node_positions: Dict[str, Tuple[float, float]] = {}  # Geographic/logical positions
        
        # Optimization state
        self.optimization_active = False
        self.last_optimization = datetime.now(timezone.utc)
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.baseline_performance: Optional[Dict[str, float]] = None
        self.current_performance: Dict[str, float] = {}
        
        # Safety integration
        self.circuit_breaker = CircuitBreakerNetwork()
        self.safety_monitor = SafetyMonitor()
    
    async def initialize_topology(self, peer_nodes: List[PeerNode], 
                                topology_type: TopologyType = TopologyType.ADAPTIVE) -> bool:
        """Initialize network topology with peer nodes"""
        try:
            print(f"🌐 Initializing network topology with {len(peer_nodes)} nodes")
            
            # Store active nodes
            self.active_nodes = {node.peer_id: node for node in peer_nodes}
            self.topology_type = topology_type
            
            # Determine optimal topology type if adaptive
            if topology_type == TopologyType.ADAPTIVE:
                self.topology_type = self._determine_optimal_topology_type(len(peer_nodes))
            
            # Create initial topology
            self.current_topology = await self._create_topology(peer_nodes, self.topology_type)
            
            if self.current_topology:
                # Calculate initial metrics
                self.network_metrics.calculate_topology_properties(self.current_topology)
                
                # Start optimization if enabled
                if ENABLE_TOPOLOGY_OPTIMIZATION:
                    self.optimization_active = True
                    asyncio.create_task(self._optimization_loop())
                
                print(f"✅ Network topology initialized:")
                print(f"   - Topology type: {self.topology_type.value}")
                print(f"   - Nodes: {len(self.current_topology.nodes())}")
                print(f"   - Edges: {len(self.current_topology.edges())}")
                print(f"   - Connectivity: {len(self.current_topology.edges()) / (len(peer_nodes) * (len(peer_nodes) - 1) / 2):.2%}")
                
                return True
            else:
                print(f"❌ Failed to create network topology")
                return False
                
        except Exception as e:
            print(f"❌ Error initializing topology: {e}")
            return False
    
    def _determine_optimal_topology_type(self, network_size: int) -> TopologyType:
        """Determine optimal topology type based on network size"""
        if network_size <= SMALL_NETWORK_THRESHOLD:
            return TopologyType.FULL_MESH
        elif network_size <= MEDIUM_NETWORK_THRESHOLD:
            return TopologyType.SMALL_WORLD
        elif network_size <= LARGE_NETWORK_THRESHOLD:
            return TopologyType.SCALE_FREE
        else:
            return TopologyType.HIERARCHICAL
    
    async def _create_topology(self, peer_nodes: List[PeerNode], topology_type: TopologyType) -> nx.Graph:
        """Create network topology based on specified type"""
        try:
            node_ids = [node.peer_id for node in peer_nodes]
            graph = nx.Graph()
            graph.add_nodes_from(node_ids)
            
            if topology_type == TopologyType.FULL_MESH:
                return self._create_full_mesh(graph, peer_nodes)
            elif topology_type == TopologyType.SMALL_WORLD:
                return self._create_small_world(graph, peer_nodes)
            elif topology_type == TopologyType.SCALE_FREE:
                return self._create_scale_free(graph, peer_nodes)
            elif topology_type == TopologyType.HIERARCHICAL:
                return self._create_hierarchical(graph, peer_nodes)
            else:
                # Default to small world for unknown types
                return self._create_small_world(graph, peer_nodes)
                
        except Exception as e:
            print(f"❌ Error creating topology: {e}")
            return nx.Graph()
    
    def _create_full_mesh(self, graph: nx.Graph, peer_nodes: List[PeerNode]) -> nx.Graph:
        """Create full mesh topology (every node connected to every other node)"""
        print(f"🕸️ Creating full mesh topology")
        
        node_ids = [node.peer_id for node in peer_nodes]
        
        # Connect every node to every other node
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                graph.add_edge(node_ids[i], node_ids[j])
        
        return graph
    
    def _create_small_world(self, graph: nx.Graph, peer_nodes: List[PeerNode]) -> nx.Graph:
        """Create small-world topology (high clustering, short path lengths)"""
        print(f"🌍 Creating small-world topology")
        
        node_ids = [node.peer_id for node in peer_nodes]
        n = len(node_ids)
        
        if n < 3:
            # Too small for small-world, use full mesh
            return self._create_full_mesh(graph, peer_nodes)
        
        # Start with ring lattice (each node connected to k nearest neighbors)
        k = min(6, n - 1)  # Each node connected to 6 neighbors (or all if n < 7)
        
        for i in range(n):
            for j in range(1, k // 2 + 1):
                # Connect to neighbors on both sides
                left_neighbor = (i - j) % n
                right_neighbor = (i + j) % n
                
                graph.add_edge(node_ids[i], node_ids[left_neighbor])
                graph.add_edge(node_ids[i], node_ids[right_neighbor])
        
        # Rewire edges with probability p to create small-world properties
        p = 0.3  # Rewiring probability
        edges_to_rewire = list(graph.edges())
        
        for edge in edges_to_rewire:
            if asyncio.get_event_loop().is_running():
                if hash(edge) % 100 < p * 100:  # Probability-based rewiring
                    u, v = edge
                    # Find a new random target
                    possible_targets = [node for node in node_ids if node != u and not graph.has_edge(u, node)]
                    if possible_targets:
                        new_target = possible_targets[hash(f"{u}{v}") % len(possible_targets)]
                        graph.remove_edge(u, v)
                        graph.add_edge(u, new_target)
        
        return graph
    
    def _create_scale_free(self, graph: nx.Graph, peer_nodes: List[PeerNode]) -> nx.Graph:
        """Create scale-free topology (hub-based, preferential attachment)"""
        print(f"📊 Creating scale-free topology")
        
        # Sort nodes by reputation (higher reputation = more likely to be hubs)
        sorted_nodes = sorted(peer_nodes, key=lambda n: n.reputation_score, reverse=True)
        node_ids = [node.peer_id for node in sorted_nodes]
        n = len(node_ids)
        
        if n < 3:
            return self._create_full_mesh(graph, peer_nodes)
        
        # Start with a small complete graph
        initial_size = min(3, n)
        for i in range(initial_size):
            for j in range(i + 1, initial_size):
                graph.add_edge(node_ids[i], node_ids[j])
        
        # Add remaining nodes with preferential attachment
        for i in range(initial_size, n):
            new_node = node_ids[i]
            current_nodes = node_ids[:i]
            
            # Number of edges to add (based on reputation)
            reputation = sorted_nodes[i].reputation_score
            m = max(1, min(3, int(reputation * 4)))  # 1-3 edges based on reputation
            
            # Preferential attachment: higher degree nodes more likely to be chosen
            degrees = [graph.degree(node) for node in current_nodes]
            total_degree = sum(degrees)
            
            if total_degree == 0:
                # Connect to first available node
                graph.add_edge(new_node, current_nodes[0])
            else:
                # Select nodes based on degree probability
                targets = []
                for _ in range(min(m, len(current_nodes))):
                    probabilities = [degree / total_degree for degree in degrees]
                    
                    # Select target based on cumulative probability
                    rand_val = (hash(f"{new_node}_{len(targets)}") % 1000) / 1000.0
                    cumulative = 0.0
                    
                    for j, prob in enumerate(probabilities):
                        cumulative += prob
                        if rand_val <= cumulative and current_nodes[j] not in targets:
                            targets.append(current_nodes[j])
                            break
                    
                    if not targets:
                        targets.append(current_nodes[0])
                
                # Add edges to selected targets
                for target in targets:
                    if not graph.has_edge(new_node, target):
                        graph.add_edge(new_node, target)
        
        return graph
    
    def _create_hierarchical(self, graph: nx.Graph, peer_nodes: List[PeerNode]) -> nx.Graph:
        """Create hierarchical topology (tree-like structure with cross-links)"""
        print(f"🌳 Creating hierarchical topology")
        
        # Sort nodes by reputation (higher reputation = higher in hierarchy)
        sorted_nodes = sorted(peer_nodes, key=lambda n: n.reputation_score, reverse=True)
        node_ids = [node.peer_id for node in sorted_nodes]
        n = len(node_ids)
        
        if n < 3:
            return self._create_full_mesh(graph, peer_nodes)
        
        # Create tree structure
        branching_factor = max(2, int(math.sqrt(n)))  # Dynamic branching based on network size
        
        # Build tree level by level
        levels = []
        current_level = [node_ids[0]]  # Root node (highest reputation)
        levels.append(current_level)
        
        remaining_nodes = node_ids[1:]
        
        while remaining_nodes:
            next_level = []
            
            for parent in current_level:
                # Each parent gets up to branching_factor children
                children_count = min(branching_factor, len(remaining_nodes))
                children = remaining_nodes[:children_count]
                remaining_nodes = remaining_nodes[children_count:]
                
                for child in children:
                    graph.add_edge(parent, child)
                    next_level.append(child)
                
                if not remaining_nodes:
                    break
            
            levels.append(next_level)
            current_level = next_level
        
        # Add cross-links for redundancy (connect some nodes at same level)
        for level in levels:
            if len(level) > 1:
                # Connect some nodes within the level
                for i in range(0, len(level) - 1, 2):
                    if i + 1 < len(level):
                        graph.add_edge(level[i], level[i + 1])
        
        # Add some upward cross-links for better connectivity
        for i in range(1, len(levels)):
            current_level = levels[i]
            parent_level = levels[i - 1]
            
            for j, node in enumerate(current_level):
                # Occasionally connect to non-parent nodes in parent level
                if j % 3 == 0 and len(parent_level) > 1:
                    # Find non-parent in parent level
                    parent_candidates = [p for p in parent_level if not graph.has_edge(node, p)]
                    if parent_candidates:
                        graph.add_edge(node, parent_candidates[0])
        
        return graph
    
    async def optimize_topology(self) -> bool:
        """Optimize current topology for better performance"""
        try:
            if not self.current_topology or not self.active_nodes:
                return False
            
            print(f"🔧 Optimizing network topology")
            
            # Get current performance baseline
            current_metrics = self.network_metrics.get_performance_summary()
            
            # Identify optimization opportunities
            optimizations = await self._identify_optimizations()
            
            if not optimizations:
                print(f"   ✅ Network topology already optimal")
                return True
            
            # Apply optimizations
            improvements = 0
            for optimization in optimizations:
                success = await self._apply_optimization(optimization)
                if success:
                    improvements += 1
            
            # Recalculate metrics after optimization
            self.network_metrics.calculate_topology_properties(self.current_topology)
            new_metrics = self.network_metrics.get_performance_summary()
            
            # Record optimization results
            optimization_result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "optimizations_attempted": len(optimizations),
                "optimizations_applied": improvements,
                "before_metrics": current_metrics,
                "after_metrics": new_metrics
            }
            
            self.optimization_history.append(optimization_result)
            self.last_optimization = datetime.now(timezone.utc)
            
            print(f"✅ Topology optimization complete:")
            print(f"   - Optimizations applied: {improvements}/{len(optimizations)}")
            print(f"   - Connectivity: {new_metrics.get('connectivity_ratio', 0):.2%}")
            print(f"   - Average path length: {new_metrics.get('average_path_length', 'N/A')}")
            
            return improvements > 0
            
        except Exception as e:
            print(f"❌ Error optimizing topology: {e}")
            return False
    
    async def _identify_optimizations(self) -> List[Dict[str, Any]]:
        """Identify potential topology optimizations"""
        optimizations = []
        
        try:
            if not self.current_topology:
                return optimizations
            
            metrics = self.network_metrics.get_performance_summary()
            
            # Check connectivity ratio
            connectivity_ratio = metrics.get("connectivity_ratio", 0)
            if connectivity_ratio < MIN_CONNECTIVITY:
                optimizations.append({
                    "type": "increase_connectivity",
                    "reason": f"Low connectivity: {connectivity_ratio:.2%} < {MIN_CONNECTIVITY:.2%}",
                    "priority": "high"
                })
            
            # Check path length efficiency
            avg_path_length = metrics.get("average_path_length")
            network_size = metrics.get("network_size", 1)
            theoretical_optimal = math.log(network_size) / math.log(2) + 1
            
            if avg_path_length and avg_path_length > theoretical_optimal * 1.5:
                optimizations.append({
                    "type": "reduce_path_length",
                    "reason": f"High path length: {avg_path_length:.2f} > {theoretical_optimal * 1.5:.2f}",
                    "priority": "medium"
                })
            
            # Check clustering vs path length trade-off
            clustering = metrics.get("global_clustering", 0)
            if clustering < 0.3 and network_size > 10:
                optimizations.append({
                    "type": "improve_clustering",
                    "reason": f"Low clustering: {clustering:.2f} < 0.3",
                    "priority": "low"
                })
            
            # Check for isolated or poorly connected nodes
            degrees = self.network_metrics.node_degrees
            if degrees:
                min_degree = min(degrees.values())
                avg_degree = statistics.mean(degrees.values())
                
                if min_degree < avg_degree * 0.3:
                    optimizations.append({
                        "type": "balance_connectivity",
                        "reason": f"Unbalanced connectivity: min degree {min_degree} << avg {avg_degree:.1f}",
                        "priority": "medium"
                    })
            
            return optimizations
            
        except Exception as e:
            print(f"❌ Error identifying optimizations: {e}")
            return []
    
    async def _apply_optimization(self, optimization: Dict[str, Any]) -> bool:
        """Apply a specific topology optimization"""
        try:
            optimization_type = optimization["type"]
            
            if optimization_type == "increase_connectivity":
                return await self._increase_connectivity()
            elif optimization_type == "reduce_path_length":
                return await self._reduce_path_length()
            elif optimization_type == "improve_clustering":
                return await self._improve_clustering()
            elif optimization_type == "balance_connectivity":
                return await self._balance_connectivity()
            else:
                print(f"⚠️ Unknown optimization type: {optimization_type}")
                return False
                
        except Exception as e:
            print(f"❌ Error applying optimization {optimization}: {e}")
            return False
    
    async def _increase_connectivity(self) -> bool:
        """Increase overall network connectivity"""
        try:
            if not self.current_topology:
                return False
            
            nodes = list(self.current_topology.nodes())
            edges_added = 0
            
            # Add edges between disconnected or poorly connected nodes
            for i, node_a in enumerate(nodes):
                for node_b in nodes[i + 1:]:
                    if not self.current_topology.has_edge(node_a, node_b):
                        # Check if adding this edge would improve connectivity
                        degree_a = self.current_topology.degree(node_a)
                        degree_b = self.current_topology.degree(node_b)
                        
                        avg_degree = sum(dict(self.current_topology.degree()).values()) / len(nodes)
                        
                        if degree_a < avg_degree * 0.7 or degree_b < avg_degree * 0.7:
                            self.current_topology.add_edge(node_a, node_b)
                            edges_added += 1
                            
                            # Don't add too many edges at once
                            if edges_added >= len(nodes) // 2:
                                break
                
                if edges_added >= len(nodes) // 2:
                    break
            
            print(f"   📈 Added {edges_added} edges to increase connectivity")
            return edges_added > 0
            
        except Exception as e:
            print(f"❌ Error increasing connectivity: {e}")
            return False
    
    async def _reduce_path_length(self) -> bool:
        """Add strategic edges to reduce average path length"""
        try:
            if not self.current_topology or not nx.is_connected(self.current_topology):
                return False
            
            nodes = list(self.current_topology.nodes())
            edges_added = 0
            
            # Calculate shortest paths
            path_lengths = dict(nx.all_pairs_shortest_path_length(self.current_topology))
            
            # Find node pairs with longest shortest paths
            long_paths = []
            for source in path_lengths:
                for target, length in path_lengths[source].items():
                    if source < target and length > 3:  # Only consider paths longer than 3
                        long_paths.append((source, target, length))
            
            # Sort by path length and add shortcuts for longest paths
            long_paths.sort(key=lambda x: x[2], reverse=True)
            
            for source, target, length in long_paths[:len(nodes) // 4]:  # Limit shortcuts
                if not self.current_topology.has_edge(source, target):
                    self.current_topology.add_edge(source, target)
                    edges_added += 1
            
            print(f"   🔗 Added {edges_added} shortcut edges to reduce path length")
            return edges_added > 0
            
        except Exception as e:
            print(f"❌ Error reducing path length: {e}")
            return False
    
    async def _improve_clustering(self) -> bool:
        """Improve network clustering by adding triangular connections"""
        try:
            if not self.current_topology:
                return False
            
            nodes = list(self.current_topology.nodes())
            edges_added = 0
            
            # Find potential triangles to close
            for node in nodes:
                neighbors = list(self.current_topology.neighbors(node))
                
                # Look for pairs of neighbors that aren't connected
                for i, neighbor_a in enumerate(neighbors):
                    for neighbor_b in neighbors[i + 1:]:
                        if not self.current_topology.has_edge(neighbor_a, neighbor_b):
                            # Add edge to close triangle
                            self.current_topology.add_edge(neighbor_a, neighbor_b)
                            edges_added += 1
                            
                            # Don't add too many at once
                            if edges_added >= len(nodes) // 3:
                                break
                    
                    if edges_added >= len(nodes) // 3:
                        break
                
                if edges_added >= len(nodes) // 3:
                    break
            
            print(f"   🔺 Added {edges_added} edges to improve clustering")
            return edges_added > 0
            
        except Exception as e:
            print(f"❌ Error improving clustering: {e}")
            return False
    
    async def _balance_connectivity(self) -> bool:
        """Balance connectivity by preferentially connecting low-degree nodes"""
        try:
            if not self.current_topology:
                return False
            
            degrees = dict(self.current_topology.degree())
            avg_degree = statistics.mean(degrees.values())
            
            # Identify low-degree and high-degree nodes
            low_degree_nodes = [node for node, degree in degrees.items() if degree < avg_degree * 0.6]
            high_degree_nodes = [node for node, degree in degrees.items() if degree > avg_degree * 1.2]
            
            edges_added = 0
            
            # Connect low-degree nodes to each other and to high-degree nodes
            for low_node in low_degree_nodes:
                # Connect to other low-degree nodes
                for other_low in low_degree_nodes:
                    if (low_node != other_low and 
                        not self.current_topology.has_edge(low_node, other_low)):
                        self.current_topology.add_edge(low_node, other_low)
                        edges_added += 1
                        break  # Only one connection per low-degree node
                
                # Connect to a high-degree node if available
                for high_node in high_degree_nodes:
                    if not self.current_topology.has_edge(low_node, high_node):
                        self.current_topology.add_edge(low_node, high_node)
                        edges_added += 1
                        break
            
            print(f"   ⚖️ Added {edges_added} edges to balance connectivity")
            return edges_added > 0
            
        except Exception as e:
            print(f"❌ Error balancing connectivity: {e}")
            return False
    
    async def _optimization_loop(self):
        """Continuous topology optimization loop"""
        try:
            while self.optimization_active:
                await asyncio.sleep(TOPOLOGY_UPDATE_INTERVAL)
                
                if (datetime.now(timezone.utc) - self.last_optimization).total_seconds() >= TOPOLOGY_UPDATE_INTERVAL:
                    await self.optimize_topology()
                    
        except Exception as e:
            print(f"❌ Error in optimization loop: {e}")
    
    async def add_node(self, node: PeerNode) -> bool:
        """Add a new node to the topology"""
        try:
            if not self.current_topology:
                return False
            
            self.active_nodes[node.peer_id] = node
            self.current_topology.add_node(node.peer_id)
            
            # Connect new node based on current topology type
            await self._integrate_new_node(node.peer_id)
            
            # Update metrics
            self.network_metrics.calculate_topology_properties(self.current_topology)
            
            print(f"📈 Added node {node.peer_id} to topology")
            return True
            
        except Exception as e:
            print(f"❌ Error adding node to topology: {e}")
            return False
    
    async def remove_node(self, node_id: str) -> bool:
        """Remove a node from the topology"""
        try:
            if not self.current_topology or node_id not in self.current_topology:
                return False
            
            self.current_topology.remove_node(node_id)
            if node_id in self.active_nodes:
                del self.active_nodes[node_id]
            
            # Update metrics
            self.network_metrics.calculate_topology_properties(self.current_topology)
            
            print(f"📉 Removed node {node_id} from topology")
            return True
            
        except Exception as e:
            print(f"❌ Error removing node from topology: {e}")
            return False
    
    async def _integrate_new_node(self, node_id: str):
        """Integrate a new node into the existing topology"""
        try:
            existing_nodes = [n for n in self.current_topology.nodes() if n != node_id]
            
            if not existing_nodes:
                return
            
            # Connect based on topology type
            if self.topology_type == TopologyType.FULL_MESH:
                # Connect to all existing nodes
                for existing_node in existing_nodes:
                    self.current_topology.add_edge(node_id, existing_node)
            
            elif self.topology_type == TopologyType.SMALL_WORLD:
                # Connect to a few nearby nodes
                connections = min(3, len(existing_nodes))
                targets = existing_nodes[:connections]
                for target in targets:
                    self.current_topology.add_edge(node_id, target)
            
            elif self.topology_type == TopologyType.SCALE_FREE:
                # Preferential attachment to high-degree nodes
                degrees = dict(self.current_topology.degree())
                sorted_nodes = sorted(existing_nodes, key=lambda n: degrees[n], reverse=True)
                connections = min(2, len(sorted_nodes))
                for target in sorted_nodes[:connections]:
                    self.current_topology.add_edge(node_id, target)
            
            elif self.topology_type == TopologyType.HIERARCHICAL:
                # Connect to parent(s) in hierarchy
                degrees = dict(self.current_topology.degree())
                # Find nodes with moderate degree (not leaves, not super-hubs)
                avg_degree = statistics.mean(degrees.values()) if degrees else 1
                potential_parents = [n for n in existing_nodes if degrees[n] >= avg_degree * 0.8]
                
                if potential_parents:
                    parent = potential_parents[0]
                    self.current_topology.add_edge(node_id, parent)
                else:
                    # Fallback: connect to first available node
                    self.current_topology.add_edge(node_id, existing_nodes[0])
                    
        except Exception as e:
            print(f"❌ Error integrating new node: {e}")
    
    async def get_topology_metrics(self) -> Dict[str, Any]:
        """Get comprehensive topology metrics"""
        try:
            if not self.current_topology:
                return {"error": "No topology initialized"}
            
            # Get basic network metrics
            base_metrics = self.network_metrics.get_performance_summary()
            
            # Add topology-specific metrics
            topology_metrics = {
                "topology_type": self.topology_type.value,
                "optimization_active": self.optimization_active,
                "last_optimization": self.last_optimization.isoformat(),
                "optimization_count": len(self.optimization_history),
                "nodes": len(self.current_topology.nodes()),
                "edges": len(self.current_topology.edges()),
                "is_connected": nx.is_connected(self.current_topology) if self.current_topology.nodes() else False
            }
            
            return {**base_metrics, **topology_metrics}
            
        except Exception as e:
            print(f"❌ Error getting topology metrics: {e}")
            return {"error": str(e)}
    
    def get_adjacency_matrix(self) -> Dict[str, Dict[str, bool]]:
        """Get network adjacency matrix"""
        try:
            if not self.current_topology:
                return {}
            
            nodes = list(self.current_topology.nodes())
            matrix = {}
            
            for node_a in nodes:
                matrix[node_a] = {}
                for node_b in nodes:
                    matrix[node_a][node_b] = self.current_topology.has_edge(node_a, node_b)
            
            return matrix
            
        except Exception as e:
            print(f"❌ Error generating adjacency matrix: {e}")
            return {}


# === Global Network Topology Instance ===

_network_topology_instance: Optional[TopologyOptimizer] = None

def get_network_topology() -> TopologyOptimizer:
    """Get or create the global network topology optimizer"""
    global _network_topology_instance
    if _network_topology_instance is None:
        _network_topology_instance = TopologyOptimizer()
    return _network_topology_instance