"""
Information Space Data Models

Core data structures for the Information Space visualization system.
Builds on NetworkX and integrates with PRSM's existing architecture.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import networkx as nx

class NodeType(str, Enum):
    """Types of nodes in the Information Space."""
    RESEARCH_AREA = "research_area"
    DOCUMENT = "document"
    RESEARCHER = "researcher"
    PROJECT = "project"
    DATASET = "dataset"
    MODEL = "model"
    CONCEPT = "concept"
    COLLABORATION = "collaboration"
    FUNDING_OPPORTUNITY = "funding_opportunity"


class EdgeType(str, Enum):
    """Types of relationships between nodes."""
    COLLABORATION = "collaboration"
    CITATION = "citation"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CO_AUTHORSHIP = "co_authorship"
    DATASET_USAGE = "dataset_usage"
    MODEL_TRAINING = "model_training"
    CONCEPT_RELATION = "concept_relation"
    PROJECT_CONTRIBUTION = "project_contribution"
    FUNDING_RELATION = "funding_relation"


class OpportunityType(str, Enum):
    """Types of research opportunities."""
    COLLABORATION = "collaboration"
    CROSS_DOMAIN = "cross_domain"
    FUNDING = "funding"
    DATASET_SHARING = "dataset_sharing"
    MODEL_IMPROVEMENT = "model_improvement"
    KNOWLEDGE_GAP = "knowledge_gap"


@dataclass
class InfoNode:
    """Information Space node representing an entity in the research ecosystem."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label: str = ""
    node_type: NodeType = NodeType.CONCEPT
    
    # Core properties
    description: str = ""
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # IPFS integration
    ipfs_hash: Optional[str] = None
    content_url: Optional[str] = None
    
    # Network metrics
    connections: int = 0
    centrality_score: float = 0.0
    influence_score: float = 0.0
    
    # Research metrics
    opportunity_score: float = 0.0
    collaboration_potential: float = 0.0
    research_activity: float = 0.0
    
    # FTNS integration
    ftns_value: Decimal = field(default_factory=lambda: Decimal('0'))
    contribution_rewards: Decimal = field(default_factory=lambda: Decimal('0'))
    
    # Temporal data
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: Optional[datetime] = None
    
    # Position for visualization
    position: Optional[Tuple[float, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for API responses."""
        return {
            "id": self.id,
            "label": self.label,
            "type": self.node_type.value,
            "description": self.description,
            "tags": list(self.tags),
            "metadata": self.metadata,
            "ipfs_hash": self.ipfs_hash,
            "content_url": self.content_url,
            "connections": self.connections,
            "centrality_score": self.centrality_score,
            "influence_score": self.influence_score,
            "opportunity_score": self.opportunity_score,
            "collaboration_potential": self.collaboration_potential,
            "research_activity": self.research_activity,
            "ftns_value": float(self.ftns_value),
            "contribution_rewards": float(self.contribution_rewards),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "position": self.position
        }


@dataclass
class InfoEdge:
    """Information Space edge representing relationships between entities."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""  # Source node ID
    target: str = ""  # Target node ID
    edge_type: EdgeType = EdgeType.SEMANTIC_SIMILARITY
    
    # Relationship strength
    weight: float = 1.0
    confidence: float = 1.0
    
    # Relationship details
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Evidence and provenance
    evidence: List[str] = field(default_factory=list)  # IPFS hashes of supporting evidence
    citations: List[str] = field(default_factory=list)
    
    # FTNS rewards for relationship discovery
    discovery_reward: Decimal = field(default_factory=lambda: Decimal('0'))
    validator_rewards: Dict[str, Decimal] = field(default_factory=dict)
    
    # Temporal data
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    validated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary for API responses."""
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "type": self.edge_type.value,
            "weight": self.weight,
            "confidence": self.confidence,
            "description": self.description,
            "metadata": self.metadata,
            "evidence": self.evidence,
            "citations": self.citations,
            "discovery_reward": float(self.discovery_reward),
            "validator_rewards": {k: float(v) for k, v in self.validator_rewards.items()},
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "validated_at": self.validated_at.isoformat() if self.validated_at else None
        }


@dataclass
class ResearchOpportunity:
    """Research opportunity identified through Information Space analysis."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    opportunity_type: OpportunityType = OpportunityType.COLLABORATION
    
    # Opportunity metrics
    confidence: float = 0.0
    impact_score: float = 0.0
    feasibility_score: float = 0.0
    urgency_score: float = 0.0
    
    # Related entities
    research_areas: List[str] = field(default_factory=list)  # Node IDs
    researchers: List[str] = field(default_factory=list)  # Node IDs
    projects: List[str] = field(default_factory=list)  # Node IDs
    
    # FTNS economics
    estimated_value: Decimal = field(default_factory=lambda: Decimal('0'))
    funding_required: Decimal = field(default_factory=lambda: Decimal('0'))
    expected_rewards: Decimal = field(default_factory=lambda: Decimal('0'))
    
    # Implementation details
    suggested_timeline: str = ""
    required_resources: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    
    # Temporal data
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert opportunity to dictionary for API responses."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "type": self.opportunity_type.value,
            "confidence": self.confidence,
            "impact_score": self.impact_score,
            "feasibility_score": self.feasibility_score,
            "urgency_score": self.urgency_score,
            "research_areas": self.research_areas,
            "researchers": self.researchers,
            "projects": self.projects,
            "estimated_value": float(self.estimated_value),
            "funding_required": float(self.funding_required),
            "expected_rewards": float(self.expected_rewards),
            "suggested_timeline": self.suggested_timeline,
            "required_resources": self.required_resources,
            "success_criteria": self.success_criteria,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }


@dataclass
class ContentAnalysis:
    """Analysis results for content processed through Information Space."""
    
    content_id: str = ""
    ipfs_hash: str = ""
    
    # Extracted metadata
    title: str = ""
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    
    # Semantic analysis
    concepts: List[str] = field(default_factory=list)
    topics: Dict[str, float] = field(default_factory=dict)  # Topic -> probability
    embeddings: Optional[List[float]] = None
    
    # Relationship extraction
    cited_works: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    collaboration_indicators: List[str] = field(default_factory=list)
    
    # Quality metrics
    citation_count: int = 0
    impact_factor: float = 0.0
    novelty_score: float = 0.0
    quality_score: float = 0.0
    
    # Analysis metadata
    analysis_version: str = "1.0"
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary."""
        return {
            "content_id": self.content_id,
            "ipfs_hash": self.ipfs_hash,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "keywords": self.keywords,
            "concepts": self.concepts,
            "topics": self.topics,
            "embeddings": self.embeddings,
            "cited_works": self.cited_works,
            "related_concepts": self.related_concepts,
            "collaboration_indicators": self.collaboration_indicators,
            "citation_count": self.citation_count,
            "impact_factor": self.impact_factor,
            "novelty_score": self.novelty_score,
            "quality_score": self.quality_score,
            "analysis_version": self.analysis_version,
            "analyzed_at": self.analyzed_at.isoformat(),
            "processing_time": self.processing_time
        }


class InformationGraph:
    """NetworkX-based graph for Information Space with PRSM integration."""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, InfoNode] = {}
        self.edges: Dict[str, InfoEdge] = {}
        self.opportunities: Dict[str, ResearchOpportunity] = {}
        
    def add_node(self, node: InfoNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, **node.to_dict())
        
    def add_edge(self, edge: InfoEdge) -> None:
        """Add an edge to the graph."""
        self.edges[edge.id] = edge
        self.graph.add_edge(
            edge.source, 
            edge.target, 
            key=edge.id,
            **edge.to_dict()
        )
        
    def add_opportunity(self, opportunity: ResearchOpportunity) -> None:
        """Add a research opportunity."""
        self.opportunities[opportunity.id] = opportunity
        
    def get_node(self, node_id: str) -> Optional[InfoNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
        
    def get_edge(self, edge_id: str) -> Optional[InfoEdge]:
        """Get an edge by ID."""
        return self.edges.get(edge_id)
        
    def get_opportunity(self, opportunity_id: str) -> Optional[ResearchOpportunity]:
        """Get an opportunity by ID."""
        return self.opportunities.get(opportunity_id)
        
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get neighboring node IDs."""
        if node_id not in self.graph:
            return []
        return list(self.graph.neighbors(node_id))
        
    def get_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Get shortest path between two nodes."""
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None
            
    def calculate_centrality_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate various centrality metrics for all nodes."""
        if not self.graph.nodes():
            return {}
            
        # Convert to simple graph for centrality calculations
        simple_graph = nx.Graph(self.graph)
        
        metrics = {}
        
        # Betweenness centrality
        betweenness = nx.betweenness_centrality(simple_graph)
        
        # Closeness centrality
        closeness = nx.closeness_centrality(simple_graph)
        
        # Degree centrality
        degree = nx.degree_centrality(simple_graph)
        
        # Eigenvector centrality (if possible)
        try:
            eigenvector = nx.eigenvector_centrality(simple_graph, max_iter=1000)
        except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
            eigenvector = {node: 0.0 for node in simple_graph.nodes()}
            
        # Combine metrics
        for node_id in simple_graph.nodes():
            metrics[node_id] = {
                "betweenness": betweenness.get(node_id, 0.0),
                "closeness": closeness.get(node_id, 0.0),
                "degree": degree.get(node_id, 0.0),
                "eigenvector": eigenvector.get(node_id, 0.0)
            }
            
        return metrics
        
    def update_node_metrics(self) -> None:
        """Update centrality and influence scores for all nodes."""
        metrics = self.calculate_centrality_metrics()
        
        for node_id, node_metrics in metrics.items():
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.centrality_score = node_metrics["betweenness"]
                node.influence_score = (
                    node_metrics["eigenvector"] * 0.4 +
                    node_metrics["degree"] * 0.3 +
                    node_metrics["closeness"] * 0.3
                )
                node.connections = self.graph.degree(node_id)
                node.updated_at = datetime.utcnow()
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary for API responses."""
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
            "opportunities": [opp.to_dict() for opp in self.opportunities.values()],
            "graph_metrics": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "total_opportunities": len(self.opportunities),
                "density": nx.density(self.graph) if self.graph.nodes() else 0.0,
                "connected_components": nx.number_connected_components(self.graph.to_undirected())
            }
        }