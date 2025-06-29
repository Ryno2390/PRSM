"""
Information Space API Endpoints

FastAPI endpoints for Information Space functionality.
Integrates with existing PRSM API patterns and provides comprehensive REST interface.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal

from fastapi import APIRouter, HTTPException, Query, Body, Depends
from pydantic import BaseModel, Field

from .service import InformationSpaceService
from .models import NodeType, EdgeType, OpportunityType
from .visualizer import GraphVisualizer

logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class NodeFilterRequest(BaseModel):
    node_types: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    min_opportunity_score: Optional[float] = None
    min_influence_score: Optional[float] = None
    created_after: Optional[str] = None


class VisualizationConfigRequest(BaseModel):
    layout: str = Field(default="force_directed", description="Layout algorithm: force_directed, hierarchical, circular, grid, cluster")
    color_by: str = Field(default="type", description="Color scheme: type, impact, activity")
    filters: Optional[NodeFilterRequest] = None


class ContentAddRequest(BaseModel):
    ipfs_hash: str = Field(..., description="IPFS hash of content to analyze")
    priority: bool = Field(default=False, description="Process with high priority")
    content_type: Optional[str] = None


class ContributionRequest(BaseModel):
    node_id: str = Field(..., description="Node ID to update")
    contribution_value: Decimal = Field(..., description="FTNS value of contribution")
    contributor: str = Field(..., description="Contributor identifier")
    description: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    filters: Optional[NodeFilterRequest] = None
    limit: int = Field(default=20, ge=1, le=100)


class CollaborationRequest(BaseModel):
    researcher_id: Optional[str] = None
    research_area: Optional[str] = None
    max_suggestions: int = Field(default=10, ge=1, le=50)


class InformationSpaceAPI:
    """Information Space API endpoints."""
    
    def __init__(self, service: InformationSpaceService):
        self.service = service
        self.visualizer = GraphVisualizer()
        self.router = APIRouter(prefix="/information-space", tags=["Information Space"])
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.router.get("/graph-data")
        async def get_graph_data(
            filters: Optional[str] = Query(None, description="JSON string of filters")
        ):
            """Get Information Space graph data with optional filters."""
            try:
                filter_dict = None
                if filters:
                    import json
                    filter_dict = json.loads(filters)
                    
                graph_data = await self.service.get_graph_data(filter_dict)
                return {
                    "status": "success",
                    "data": graph_data,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting graph data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.router.post("/visualization")
        async def get_visualization_data(config: VisualizationConfigRequest):
            """Get visualization data with specified configuration."""
            try:
                # Get current graph
                filters = config.filters.dict(exclude_none=True) if config.filters else None
                graph_data = await self.service.get_graph_data(filters)
                
                # Create visualization
                vis_config = {
                    'layout': config.layout,
                    'color_by': config.color_by,
                    'filters': filters
                }
                
                # Note: This would need the actual InformationGraph object
                # For now, we'll return the graph_data with visualization metadata
                visualization_data = {
                    **graph_data,
                    'visualization_config': vis_config,
                    'layout_algorithm': config.layout,
                    'color_scheme': config.color_by
                }
                
                return {
                    "status": "success",
                    "data": visualization_data,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error generating visualization: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.router.post("/add-content")
        async def add_content(request: ContentAddRequest):
            """Add content for analysis in Information Space."""
            try:
                success = await self.service.add_content(
                    request.ipfs_hash, 
                    request.priority
                )
                
                if success:
                    return {
                        "status": "success",
                        "message": f"Content {request.ipfs_hash} added to analysis queue",
                        "priority": request.priority
                    }
                else:
                    raise HTTPException(status_code=400, detail="Failed to add content")
                    
            except Exception as e:
                logger.error(f"Error adding content: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.router.get("/node/{node_id}")
        async def get_node_details(node_id: str):
            """Get detailed information about a specific node."""
            try:
                node_details = await self.service.get_node_details(node_id)
                
                if not node_details:
                    raise HTTPException(status_code=404, detail="Node not found")
                    
                return {
                    "status": "success",
                    "data": node_details,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting node details: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.router.post("/search-opportunities")
        async def search_opportunities(request: SearchRequest):
            """Search for research opportunities."""
            try:
                filters = request.filters.dict(exclude_none=True) if request.filters else None
                
                opportunities = await self.service.search_opportunities(
                    request.query, 
                    filters
                )
                
                # Limit results
                limited_opportunities = opportunities[:request.limit]
                
                return {
                    "status": "success",
                    "data": {
                        "opportunities": limited_opportunities,
                        "total_found": len(opportunities),
                        "returned": len(limited_opportunities),
                        "query": request.query
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error searching opportunities: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.router.post("/collaboration-suggestions")
        async def get_collaboration_suggestions(request: CollaborationRequest):
            """Get collaboration suggestions."""
            try:
                suggestions = await self.service.get_collaboration_suggestions(
                    request.researcher_id,
                    request.research_area
                )
                
                # Limit results
                limited_suggestions = suggestions[:request.max_suggestions]
                
                return {
                    "status": "success",
                    "data": {
                        "suggestions": limited_suggestions,
                        "total_found": len(suggestions),
                        "returned": len(limited_suggestions),
                        "criteria": {
                            "researcher_id": request.researcher_id,
                            "research_area": request.research_area
                        }
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting collaboration suggestions: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.router.post("/update-contribution")
        async def update_node_contribution(request: ContributionRequest):
            """Update node with new contribution."""
            try:
                success = await self.service.update_node_contribution(
                    request.node_id,
                    request.contribution_value,
                    request.contributor
                )
                
                if success:
                    return {
                        "status": "success",
                        "message": f"Updated node {request.node_id} with contribution from {request.contributor}",
                        "contribution_value": float(request.contribution_value)
                    }
                else:
                    raise HTTPException(status_code=404, detail="Node not found or update failed")
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error updating contribution: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.router.get("/statistics")
        async def get_statistics():
            """Get Information Space statistics."""
            try:
                graph_data = await self.service.get_graph_data()
                
                stats = {
                    "graph_metrics": graph_data.get("graph_metrics", {}),
                    "node_distribution": {},
                    "edge_distribution": {},
                    "opportunity_distribution": {},
                    "activity_metrics": {},
                    "ftns_metrics": {}
                }
                
                # Calculate additional statistics
                nodes = graph_data.get("nodes", [])
                edges = graph_data.get("edges", [])
                opportunities = graph_data.get("opportunities", [])
                
                if nodes:
                    # Node type distribution
                    node_types = {}
                    ftns_total = 0
                    activity_total = 0
                    
                    for node in nodes:
                        node_type = node.get("type", "unknown")
                        node_types[node_type] = node_types.get(node_type, 0) + 1
                        ftns_total += node.get("ftns_value", 0)
                        activity_total += node.get("research_activity", 0)
                        
                    stats["node_distribution"] = node_types
                    stats["ftns_metrics"] = {
                        "total_value": ftns_total,
                        "average_value": ftns_total / len(nodes),
                        "total_nodes": len(nodes)
                    }
                    stats["activity_metrics"] = {
                        "average_activity": activity_total / len(nodes),
                        "active_nodes": sum(1 for node in nodes if node.get("research_activity", 0) > 0.5)
                    }
                    
                if edges:
                    # Edge type distribution
                    edge_types = {}
                    for edge in edges:
                        edge_type = edge.get("type", "unknown")
                        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
                        
                    stats["edge_distribution"] = edge_types
                    
                if opportunities:
                    # Opportunity type distribution
                    opp_types = {}
                    for opp in opportunities:
                        opp_type = opp.get("type", "unknown")
                        opp_types[opp_type] = opp_types.get(opp_type, 0) + 1
                        
                    stats["opportunity_distribution"] = opp_types
                
                return {
                    "status": "success",
                    "data": stats,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting statistics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.router.get("/health")
        async def get_health():
            """Get Information Space service health."""
            try:
                return {
                    "status": "healthy" if self.service.is_running else "stopped",
                    "service_initialized": self.service.is_running,
                    "last_update": self.service.last_update.isoformat(),
                    "analysis_queue_size": len(self.service.analysis_queue),
                    "analyzed_content_count": len(self.service.analyzed_content),
                    "graph_size": {
                        "nodes": len(self.service.graph.nodes),
                        "edges": len(self.service.graph.edges),
                        "opportunities": len(self.service.graph.opportunities)
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting health status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.router.get("/schema")
        async def get_schema():
            """Get Information Space data schema information."""
            try:
                return {
                    "status": "success",
                    "data": {
                        "node_types": [nt.value for nt in NodeType],
                        "edge_types": [et.value for et in EdgeType],
                        "opportunity_types": [ot.value for ot in OpportunityType],
                        "visualization_layouts": ["force_directed", "hierarchical", "circular", "grid", "cluster"],
                        "color_schemes": ["type", "impact", "activity"],
                        "api_version": "1.0.0"
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting schema: {e}")
                raise HTTPException(status_code=500, detail=str(e))


def create_information_space_router(service: InformationSpaceService) -> APIRouter:
    """Create and configure Information Space API router."""
    
    api = InformationSpaceAPI(service)
    return api.router


# Enhanced endpoint for existing PRSM API integration
async def get_information_space_data_enhanced(
    layout: str = Query("force_directed", description="Layout algorithm"),
    color_by: str = Query("type", description="Color scheme"),
    filters: Optional[str] = Query(None, description="JSON filters"),
    limit: int = Query(100, ge=1, le=500)
) -> Dict[str, Any]:
    """
    Enhanced version of the existing /ui/information-space endpoint.
    
    This replaces the mock data with real Information Space functionality
    while maintaining backward compatibility with the existing frontend.
    """
    
    try:
        # This would be injected in the actual FastAPI app
        # For now, we'll create a basic service instance
        service = InformationSpaceService()
        
        # Parse filters if provided
        filter_dict = None
        if filters:
            import json
            filter_dict = json.loads(filters)
            
        # Get graph data
        graph_data = await service.get_graph_data(filter_dict)
        
        # Format for existing frontend expectations
        formatted_data = {
            "graph_data": {
                "nodes": graph_data.get("nodes", [])[:limit],
                "edges": graph_data.get("edges", [])
            },
            "opportunities": graph_data.get("opportunities", [])[:10],  # Limit opportunities
            "visualization_config": {
                "layout": layout,
                "color_scheme": color_by
            },
            "metadata": {
                "total_nodes": len(graph_data.get("nodes", [])),
                "total_edges": len(graph_data.get("edges", [])),
                "total_opportunities": len(graph_data.get("opportunities", [])),
                "last_updated": datetime.utcnow().isoformat()
            }
        }
        
        return formatted_data
        
    except Exception as e:
        logger.error(f"Error in enhanced information space endpoint: {e}")
        
        # Fallback to basic mock data for compatibility
        return {
            "graph_data": {
                "nodes": [
                    {
                        "id": "ai_research",
                        "label": "AI Research",
                        "type": "research_area",
                        "connections": 15,
                        "opportunity_score": 0.85,
                        "x": 0,
                        "y": 0,
                        "color": "#3498db",
                        "size": 20
                    }
                ],
                "edges": []
            },
            "opportunities": [
                {
                    "id": "ai_collab_1",
                    "title": "AI Collaboration Opportunity",
                    "confidence": 0.80,
                    "impact_score": 0.85,
                    "type": "collaboration"
                }
            ],
            "visualization_config": {
                "layout": layout,
                "color_scheme": color_by
            },
            "metadata": {
                "total_nodes": 1,
                "total_edges": 0,
                "total_opportunities": 1,
                "last_updated": datetime.utcnow().isoformat(),
                "error": "Service unavailable - using fallback data"
            }
        }