"""
Information Space Service

Core service that orchestrates content analysis, graph building, and API endpoints.
Integrates with PRSM's existing systems including IPFS, FTNS tokenomics, and P2P network.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from decimal import Decimal
import json

from .models import (
    InfoNode, InfoEdge, ResearchOpportunity, ContentAnalysis,
    InformationGraph, NodeType, EdgeType, OpportunityType
)
from .analyzer import ContentAnalyzer, SemanticAnalyzer, GraphBuilder

logger = logging.getLogger(__name__)


class InformationSpaceService:
    """Main service for Information Space functionality."""
    
    def __init__(self, ipfs_client=None, ftns_service=None, federation_client=None):
        self.ipfs_client = ipfs_client
        self.ftns_service = ftns_service
        self.federation_client = federation_client
        
        # Core components
        self.content_analyzer = ContentAnalyzer(ipfs_client)
        self.graph_builder = GraphBuilder(self.content_analyzer)
        
        # Main information graph
        self.graph = InformationGraph()
        
        # Content tracking
        self.analyzed_content: Set[str] = set()
        self.analysis_queue: List[str] = []
        
        # Service state
        self.last_update = datetime.utcnow()
        self.update_interval = timedelta(hours=1)  # Update every hour
        self.is_running = False
        
        # Configuration
        self.config = {
            'max_content_per_update': 50,
            'similarity_threshold': 0.7,
            'opportunity_refresh_hours': 6,
            'enable_real_time_updates': True,
            'enable_ftns_rewards': True,
            'min_quality_threshold': 0.5
        }
        
    async def initialize(self) -> bool:
        """Initialize the Information Space service."""
        try:
            logger.info("Initializing Information Space service...")
            
            # Initialize components
            if self.ipfs_client:
                await self.ipfs_client.initialize()
                
            # Load existing content from IPFS
            await self._load_existing_content()
            
            # Start background processing
            if self.config['enable_real_time_updates']:
                asyncio.create_task(self._background_processor())
                
            self.is_running = True
            logger.info("Information Space service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Information Space service: {e}")
            return False
            
    async def shutdown(self):
        """Shutdown the service gracefully."""
        logger.info("Shutting down Information Space service...")
        self.is_running = False
        
    async def get_graph_data(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get the current Information Space graph data."""
        
        # Apply filters if provided
        if filters:
            filtered_graph = await self._apply_filters(self.graph, filters)
        else:
            filtered_graph = self.graph
            
        # Update metrics before returning
        filtered_graph.update_node_metrics()
        
        return filtered_graph.to_dict()
        
    async def add_content(self, ipfs_hash: str, priority: bool = False) -> bool:
        """Add content to be analyzed and integrated into the Information Space."""
        
        if ipfs_hash in self.analyzed_content:
            logger.info(f"Content already analyzed: {ipfs_hash}")
            return True
            
        if priority:
            self.analysis_queue.insert(0, ipfs_hash)
        else:
            self.analysis_queue.append(ipfs_hash)
            
        logger.info(f"Added content to analysis queue: {ipfs_hash}")
        
        # Process immediately if real-time updates are enabled
        if self.config['enable_real_time_updates']:
            await self._process_content(ipfs_hash)
            
        return True
        
    async def search_opportunities(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for research opportunities based on query and filters."""
        
        opportunities = []
        
        for opportunity in self.graph.opportunities.values():
            # Simple text matching
            if (query.lower() in opportunity.title.lower() or 
                query.lower() in opportunity.description.lower()):
                
                # Apply filters
                if filters:
                    if not self._opportunity_matches_filters(opportunity, filters):
                        continue
                        
                opportunities.append(opportunity.to_dict())
                
        # Sort by impact score
        opportunities.sort(key=lambda x: x['impact_score'], reverse=True)
        
        return opportunities
        
    async def get_node_details(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific node."""
        
        node = self.graph.get_node(node_id)
        if not node:
            return None
            
        # Get connected nodes
        neighbors = self.graph.get_neighbors(node_id)
        neighbor_details = []
        
        for neighbor_id in neighbors[:10]:  # Limit to 10 neighbors
            neighbor = self.graph.get_node(neighbor_id)
            if neighbor:
                neighbor_details.append({
                    'id': neighbor.id,
                    'label': neighbor.label,
                    'type': neighbor.node_type.value,
                    'opportunity_score': neighbor.opportunity_score
                })
                
        # Get related opportunities
        related_opportunities = []
        for opportunity in self.graph.opportunities.values():
            if node_id in opportunity.research_areas:
                related_opportunities.append(opportunity.to_dict())
                
        # Build detailed response
        details = node.to_dict()
        details.update({
            'neighbors': neighbor_details,
            'related_opportunities': related_opportunities,
            'centrality_metrics': {
                'betweenness': node.centrality_score,
                'influence': node.influence_score,
                'connections': node.connections
            }
        })
        
        return details
        
    async def get_collaboration_suggestions(self, researcher_id: str = None, research_area: str = None) -> List[Dict[str, Any]]:
        """Get collaboration suggestions based on researcher or research area."""
        
        suggestions = []
        
        if researcher_id:
            # Find researcher node
            researcher_node = self.graph.get_node(researcher_id)
            if not researcher_node:
                return []
                
            # Find similar researchers
            for node in self.graph.nodes.values():
                if (node.node_type == NodeType.RESEARCHER and 
                    node.id != researcher_id and
                    self._calculate_collaboration_potential(researcher_node, node) > 0.7):
                    
                    suggestions.append({
                        'type': 'researcher_collaboration',
                        'target_id': node.id,
                        'target_label': node.label,
                        'collaboration_score': self._calculate_collaboration_potential(researcher_node, node),
                        'shared_interests': list(researcher_node.tags.intersection(node.tags))
                    })
                    
        if research_area:
            # Find cross-domain opportunities
            for opportunity in self.graph.opportunities.values():
                if (opportunity.opportunity_type == OpportunityType.CROSS_DOMAIN and
                    research_area.lower() in opportunity.title.lower()):
                    
                    suggestions.append({
                        'type': 'cross_domain_opportunity',
                        'opportunity_id': opportunity.id,
                        'title': opportunity.title,
                        'impact_score': opportunity.impact_score,
                        'research_areas': opportunity.research_areas
                    })
                    
        # Sort by relevance score
        suggestions.sort(key=lambda x: x.get('collaboration_score', x.get('impact_score', 0)), reverse=True)
        
        return suggestions[:10]
        
    async def update_node_contribution(self, node_id: str, contribution_value: Decimal, contributor: str) -> bool:
        """Update node with new contribution and FTNS rewards."""
        
        node = self.graph.get_node(node_id)
        if not node:
            return False
            
        # Update node metrics
        node.ftns_value += contribution_value
        node.contribution_rewards += contribution_value * Decimal('0.1')  # 10% reward
        node.research_activity = min(node.research_activity + 0.1, 1.0)
        node.updated_at = datetime.utcnow()
        node.last_activity = datetime.utcnow()
        
        # Award FTNS tokens if service is available
        if self.ftns_service and self.config['enable_ftns_rewards']:
            try:
                await self.ftns_service.award_tokens(
                    recipient=contributor,
                    amount=contribution_value * Decimal('0.05'),
                    reason=f"Information Space contribution to {node.label}"
                )
            except Exception as e:
                logger.error(f"Failed to award FTNS tokens: {e}")
                
        logger.info(f"Updated node {node_id} with contribution from {contributor}")
        return True
        
    async def _load_existing_content(self):
        """Load and analyze existing content from IPFS."""
        
        if not self.ipfs_client:
            logger.warning("IPFS client not available - cannot load existing content")
            return
            
        try:
            # Get recent content from IPFS
            recent_content = await self.ipfs_client.list_recent_content(limit=100)
            
            for content_hash in recent_content:
                if content_hash not in self.analyzed_content:
                    self.analysis_queue.append(content_hash)
                    
            logger.info(f"Loaded {len(recent_content)} existing content items")
            
        except Exception as e:
            logger.error(f"Error loading existing content: {e}")
            
    async def _background_processor(self):
        """Background task for processing content and updating the graph."""
        
        while self.is_running:
            try:
                # Process queued content
                if self.analysis_queue:
                    batch_size = min(self.config['max_content_per_update'], len(self.analysis_queue))
                    batch = self.analysis_queue[:batch_size]
                    self.analysis_queue = self.analysis_queue[batch_size:]
                    
                    for content_hash in batch:
                        await self._process_content(content_hash)
                        
                    logger.info(f"Processed batch of {len(batch)} content items")
                    
                # Refresh opportunities periodically
                if (datetime.utcnow() - self.last_update).total_seconds() > self.config['opportunity_refresh_hours'] * 3600:
                    await self._refresh_opportunities()
                    self.last_update = datetime.utcnow()
                    
                # Wait before next iteration
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in background processor: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
                
    async def _process_content(self, ipfs_hash: str):
        """Process a single piece of content."""
        
        try:
            # Analyze content
            analysis = await self.content_analyzer.analyze_content(ipfs_hash)
            if not analysis:
                logger.warning(f"Failed to analyze content: {ipfs_hash}")
                return
                
            # Skip low-quality content
            if analysis.quality_score < self.config['min_quality_threshold']:
                logger.info(f"Skipping low-quality content: {ipfs_hash}")
                return
                
            # Create node from analysis
            node = self.graph_builder._create_node_from_analysis(analysis)
            self.graph.add_node(node)
            
            # Create edges to existing nodes
            await self._create_edges_for_new_node(node, analysis)
            
            # Mark as processed
            self.analyzed_content.add(ipfs_hash)
            
            logger.info(f"Successfully processed content: {ipfs_hash}")
            
        except Exception as e:
            logger.error(f"Error processing content {ipfs_hash}: {e}")
            
    async def _create_edges_for_new_node(self, new_node: InfoNode, analysis: ContentAnalysis):
        """Create edges between new node and existing nodes."""
        
        # Check similarity with existing nodes
        for existing_node in self.graph.nodes.values():
            if existing_node.id == new_node.id:
                continue
                
            # Calculate similarity based on available data
            similarity = self._calculate_node_similarity(new_node, existing_node)
            
            if similarity > self.config['similarity_threshold']:
                edge = InfoEdge(
                    source=new_node.id,
                    target=existing_node.id,
                    edge_type=EdgeType.SEMANTIC_SIMILARITY,
                    weight=similarity,
                    confidence=similarity,
                    description=f"Semantic similarity: {similarity:.2f}"
                )
                self.graph.add_edge(edge)
                
    def _calculate_node_similarity(self, node1: InfoNode, node2: InfoNode) -> float:
        """Calculate similarity between two nodes."""
        
        # Tag similarity
        if node1.tags and node2.tags:
            tag_intersection = len(node1.tags.intersection(node2.tags))
            tag_union = len(node1.tags.union(node2.tags))
            tag_similarity = tag_intersection / tag_union if tag_union > 0 else 0
        else:
            tag_similarity = 0
            
        # Type similarity
        type_similarity = 1.0 if node1.node_type == node2.node_type else 0.5
        
        # Combined similarity
        similarity = (tag_similarity * 0.7 + type_similarity * 0.3)
        
        return similarity
        
    async def _refresh_opportunities(self):
        """Refresh research opportunities based on current graph state."""
        
        try:
            # Clear existing opportunities
            self.graph.opportunities.clear()
            
            # Generate new opportunities
            opportunities = await self.graph_builder._generate_opportunities(
                self.graph, 
                list(self.content_analyzer.analysis_cache.values())
            )
            
            for opportunity in opportunities:
                self.graph.add_opportunity(opportunity)
                
            logger.info(f"Refreshed {len(opportunities)} research opportunities")
            
        except Exception as e:
            logger.error(f"Error refreshing opportunities: {e}")
            
    async def _apply_filters(self, graph: InformationGraph, filters: Dict[str, Any]) -> InformationGraph:
        """Apply filters to create a filtered view of the graph."""
        
        filtered_graph = InformationGraph()
        
        # Filter nodes
        for node in graph.nodes.values():
            if self._node_matches_filters(node, filters):
                filtered_graph.add_node(node)
                
        # Filter edges (only include if both nodes are in filtered graph)
        for edge in graph.edges.values():
            if edge.source in filtered_graph.nodes and edge.target in filtered_graph.nodes:
                filtered_graph.add_edge(edge)
                
        # Filter opportunities
        for opportunity in graph.opportunities.values():
            if self._opportunity_matches_filters(opportunity, filters):
                filtered_graph.add_opportunity(opportunity)
                
        return filtered_graph
        
    def _node_matches_filters(self, node: InfoNode, filters: Dict[str, Any]) -> bool:
        """Check if node matches the given filters."""
        
        # Node type filter
        if 'node_types' in filters:
            if node.node_type.value not in filters['node_types']:
                return False
                
        # Tag filter
        if 'tags' in filters:
            if not any(tag in node.tags for tag in filters['tags']):
                return False
                
        # Opportunity score filter
        if 'min_opportunity_score' in filters:
            if node.opportunity_score < filters['min_opportunity_score']:
                return False
                
        # Date filter
        if 'created_after' in filters:
            if node.created_at < datetime.fromisoformat(filters['created_after']):
                return False
                
        return True
        
    def _opportunity_matches_filters(self, opportunity: ResearchOpportunity, filters: Dict[str, Any]) -> bool:
        """Check if opportunity matches the given filters."""
        
        # Opportunity type filter
        if 'opportunity_types' in filters:
            if opportunity.opportunity_type.value not in filters['opportunity_types']:
                return False
                
        # Impact score filter
        if 'min_impact_score' in filters:
            if opportunity.impact_score < filters['min_impact_score']:
                return False
                
        # Confidence filter
        if 'min_confidence' in filters:
            if opportunity.confidence < filters['min_confidence']:
                return False
                
        return True
        
    def _calculate_collaboration_potential(self, node1: InfoNode, node2: InfoNode) -> float:
        """Calculate collaboration potential between two nodes."""
        
        # Shared interests
        shared_tags = len(node1.tags.intersection(node2.tags))
        total_tags = len(node1.tags.union(node2.tags))
        interest_similarity = shared_tags / total_tags if total_tags > 0 else 0
        
        # Complementary strengths
        strength_complement = abs(node1.research_activity - node2.research_activity)
        
        # Opportunity alignment
        opportunity_alignment = (node1.opportunity_score + node2.opportunity_score) / 2
        
        # Combined score
        collaboration_score = (
            interest_similarity * 0.4 +
            (1 - strength_complement) * 0.3 +
            opportunity_alignment * 0.3
        )
        
        return collaboration_score