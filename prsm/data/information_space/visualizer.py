"""
Information Space Visualization Engine

Creates interactive visualizations for the Information Space graph.
Integrates with existing PRSM frontend patterns and provides rich interactive experience.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from decimal import Decimal
import math
import random

from .models import InformationGraph, InfoNode, InfoEdge, ResearchOpportunity, NodeType, EdgeType

logger = logging.getLogger(__name__)


class VisualizationEngine:
    """Core visualization engine for Information Space."""
    
    def __init__(self):
        self.layout_algorithms = {
            'force_directed': self._force_directed_layout,
            'hierarchical': self._hierarchical_layout,
            'circular': self._circular_layout,
            'grid': self._grid_layout,
            'cluster': self._cluster_layout
        }
        
        self.color_schemes = {
            'default': {
                NodeType.RESEARCH_AREA: '#3498db',
                NodeType.DOCUMENT: '#2ecc71',
                NodeType.RESEARCHER: '#e74c3c',
                NodeType.PROJECT: '#f39c12',
                NodeType.DATASET: '#9b59b6',
                NodeType.MODEL: '#1abc9c',
                NodeType.CONCEPT: '#34495e',
                NodeType.COLLABORATION: '#e67e22',
                NodeType.FUNDING_OPPORTUNITY: '#f1c40f'
            },
            'impact': {
                'high': '#e74c3c',
                'medium': '#f39c12',
                'low': '#95a5a6'
            },
            'activity': {
                'active': '#27ae60',
                'moderate': '#f39c12',
                'inactive': '#95a5a6'
            }
        }
        
    def generate_visualization_data(
        self, 
        graph: InformationGraph, 
        layout: str = 'force_directed',
        color_by: str = 'type',
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate complete visualization data for frontend rendering."""
        
        try:
            # Apply layout algorithm
            if layout in self.layout_algorithms:
                positioned_nodes = self.layout_algorithms[layout](graph)
            else:
                positioned_nodes = self._force_directed_layout(graph)
                
            # Apply color scheme
            colored_nodes = self._apply_color_scheme(positioned_nodes, color_by)
            
            # Prepare edges for visualization
            vis_edges = self._prepare_edges_for_visualization(graph, colored_nodes)
            
            # Prepare opportunities
            vis_opportunities = self._prepare_opportunities_for_visualization(graph)
            
            # Generate interaction data
            interaction_data = self._generate_interaction_data(graph)
            
            # Create complete visualization package
            visualization_data = {
                'nodes': colored_nodes,
                'edges': vis_edges,
                'opportunities': vis_opportunities,
                'interaction': interaction_data,
                'layout': {
                    'algorithm': layout,
                    'color_scheme': color_by,
                    'timestamp': datetime.utcnow().isoformat()
                },
                'statistics': self._calculate_visualization_statistics(graph),
                'legends': self._generate_legends(color_by)
            }
            
            return visualization_data
            
        except Exception as e:
            logger.error(f"Error generating visualization data: {e}")
            return self._generate_fallback_visualization()
            
    def _force_directed_layout(self, graph: InformationGraph) -> List[Dict[str, Any]]:
        """Generate force-directed layout positions for nodes."""
        
        nodes = list(graph.nodes.values())
        if not nodes:
            return []
            
        # Initialize random positions
        positions = {}
        for node in nodes:
            positions[node.id] = {
                'x': random.uniform(-500, 500),
                'y': random.uniform(-500, 500)
            }
            
        # Force-directed algorithm (simplified)
        iterations = 100
        for iteration in range(iterations):
            forces = {node.id: {'x': 0, 'y': 0} for node in nodes}
            
            # Repulsion forces between all nodes
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes[i+1:], i+1):
                    dx = positions[node2.id]['x'] - positions[node1.id]['x']
                    dy = positions[node2.id]['y'] - positions[node1.id]['y']
                    distance = math.sqrt(dx*dx + dy*dy) + 0.01  # Avoid division by zero
                    
                    repulsion = 1000 / (distance * distance)
                    fx = (dx / distance) * repulsion
                    fy = (dy / distance) * repulsion
                    
                    forces[node1.id]['x'] -= fx
                    forces[node1.id]['y'] -= fy
                    forces[node2.id]['x'] += fx
                    forces[node2.id]['y'] += fy
                    
            # Attraction forces for connected nodes
            for edge in graph.edges.values():
                if edge.source in positions and edge.target in positions:
                    dx = positions[edge.target]['x'] - positions[edge.source]['x']
                    dy = positions[edge.target]['y'] - positions[edge.source]['y']
                    distance = math.sqrt(dx*dx + dy*dy) + 0.01
                    
                    attraction = distance * 0.01 * edge.weight
                    fx = (dx / distance) * attraction
                    fy = (dy / distance) * attraction
                    
                    forces[edge.source]['x'] += fx
                    forces[edge.source]['y'] += fy
                    forces[edge.target]['x'] -= fx
                    forces[edge.target]['y'] -= fy
                    
            # Apply forces with damping
            damping = 0.9
            for node in nodes:
                positions[node.id]['x'] += forces[node.id]['x'] * damping
                positions[node.id]['y'] += forces[node.id]['y'] * damping
                
        # Convert to visualization format
        vis_nodes = []
        for node in nodes:
            vis_node = node.to_dict()
            vis_node.update(positions[node.id])
            vis_nodes.append(vis_node)
            
        return vis_nodes
        
    def _hierarchical_layout(self, graph: InformationGraph) -> List[Dict[str, Any]]:
        """Generate hierarchical layout based on node importance."""
        
        nodes = list(graph.nodes.values())
        if not nodes:
            return []
            
        # Sort nodes by influence score
        nodes.sort(key=lambda n: n.influence_score, reverse=True)
        
        # Create levels based on influence
        levels = []
        current_level = []
        current_score = None
        
        for node in nodes:
            if current_score is None or abs(node.influence_score - current_score) < 0.1:
                current_level.append(node)
                current_score = node.influence_score
            else:
                levels.append(current_level)
                current_level = [node]
                current_score = node.influence_score
                
        if current_level:
            levels.append(current_level)
            
        # Position nodes
        vis_nodes = []
        level_height = 150
        
        for level_idx, level in enumerate(levels):
            y = level_idx * level_height
            level_width = len(level) * 120
            start_x = -level_width / 2
            
            for node_idx, node in enumerate(level):
                x = start_x + (node_idx * 120)
                
                vis_node = node.to_dict()
                vis_node.update({'x': x, 'y': y})
                vis_nodes.append(vis_node)
                
        return vis_nodes
        
    def _circular_layout(self, graph: InformationGraph) -> List[Dict[str, Any]]:
        """Generate circular layout for nodes."""
        
        nodes = list(graph.nodes.values())
        if not nodes:
            return []
            
        vis_nodes = []
        radius = 300
        center_x, center_y = 0, 0
        
        for i, node in enumerate(nodes):
            angle = (2 * math.pi * i) / len(nodes)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            vis_node = node.to_dict()
            vis_node.update({'x': x, 'y': y})
            vis_nodes.append(vis_node)
            
        return vis_nodes
        
    def _grid_layout(self, graph: InformationGraph) -> List[Dict[str, Any]]:
        """Generate grid layout for nodes."""
        
        nodes = list(graph.nodes.values())
        if not nodes:
            return []
            
        # Calculate grid dimensions
        grid_size = math.ceil(math.sqrt(len(nodes)))
        cell_size = 100
        
        vis_nodes = []
        for i, node in enumerate(nodes):
            row = i // grid_size
            col = i % grid_size
            
            x = (col - grid_size/2) * cell_size
            y = (row - grid_size/2) * cell_size
            
            vis_node = node.to_dict()
            vis_node.update({'x': x, 'y': y})
            vis_nodes.append(vis_node)
            
        return vis_nodes
        
    def _cluster_layout(self, graph: InformationGraph) -> List[Dict[str, Any]]:
        """Generate cluster-based layout grouping similar nodes."""
        
        nodes = list(graph.nodes.values())
        if not nodes:
            return []
            
        # Group nodes by type
        clusters = {}
        for node in nodes:
            node_type = node.node_type.value
            if node_type not in clusters:
                clusters[node_type] = []
            clusters[node_type].append(node)
            
        # Position clusters
        vis_nodes = []
        cluster_radius = 200
        node_radius = 80
        
        cluster_positions = []
        for i, cluster_type in enumerate(clusters.keys()):
            angle = (2 * math.pi * i) / len(clusters)
            cluster_x = cluster_radius * math.cos(angle)
            cluster_y = cluster_radius * math.sin(angle)
            cluster_positions.append((cluster_x, cluster_y))
            
        # Position nodes within clusters
        for i, (cluster_type, cluster_nodes) in enumerate(clusters.items()):
            cluster_x, cluster_y = cluster_positions[i]
            
            for j, node in enumerate(cluster_nodes):
                if len(cluster_nodes) == 1:
                    x, y = cluster_x, cluster_y
                else:
                    angle = (2 * math.pi * j) / len(cluster_nodes)
                    x = cluster_x + node_radius * math.cos(angle)
                    y = cluster_y + node_radius * math.sin(angle)
                    
                vis_node = node.to_dict()
                vis_node.update({'x': x, 'y': y})
                vis_nodes.append(vis_node)
                
        return vis_nodes
        
    def _apply_color_scheme(self, nodes: List[Dict[str, Any]], color_by: str) -> List[Dict[str, Any]]:
        """Apply color scheme to nodes based on the specified criteria."""
        
        colored_nodes = []
        
        for node in nodes:
            if color_by == 'type':
                node_type = NodeType(node['type'])
                color = self.color_schemes['default'].get(node_type, '#95a5a6')
                
            elif color_by == 'impact':
                impact_score = node.get('opportunity_score', 0)
                if impact_score > 0.7:
                    color = self.color_schemes['impact']['high']
                elif impact_score > 0.4:
                    color = self.color_schemes['impact']['medium']
                else:
                    color = self.color_schemes['impact']['low']
                    
            elif color_by == 'activity':
                activity_score = node.get('research_activity', 0)
                if activity_score > 0.7:
                    color = self.color_schemes['activity']['active']
                elif activity_score > 0.3:
                    color = self.color_schemes['activity']['moderate']
                else:
                    color = self.color_schemes['activity']['inactive']
                    
            else:
                color = '#3498db'  # Default blue
                
            # Add visualization properties
            node.update({
                'color': color,
                'size': self._calculate_node_size(node),
                'opacity': self._calculate_node_opacity(node),
                'stroke': self._calculate_node_stroke(node)
            })
            
            colored_nodes.append(node)
            
        return colored_nodes
        
    def _calculate_node_size(self, node: Dict[str, Any]) -> int:
        """Calculate node size based on importance metrics."""
        
        base_size = 10
        influence_factor = node.get('influence_score', 0) * 20
        connection_factor = min(node.get('connections', 0) / 10, 5)
        
        size = base_size + influence_factor + connection_factor
        return max(8, min(50, int(size)))  # Clamp between 8 and 50
        
    def _calculate_node_opacity(self, node: Dict[str, Any]) -> float:
        """Calculate node opacity based on activity and quality."""
        
        activity = node.get('research_activity', 0.5)
        base_opacity = 0.6 + (activity * 0.4)
        
        return max(0.3, min(1.0, base_opacity))
        
    def _calculate_node_stroke(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate node stroke properties."""
        
        opportunity_score = node.get('opportunity_score', 0)
        
        if opportunity_score > 0.8:
            return {'color': '#f39c12', 'width': 3}
        elif opportunity_score > 0.6:
            return {'color': '#e67e22', 'width': 2}
        else:
            return {'color': '#bdc3c7', 'width': 1}
            
    def _prepare_edges_for_visualization(self, graph: InformationGraph, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare edges for visualization."""
        
        # Create node ID to position mapping
        node_positions = {node['id']: {'x': node['x'], 'y': node['y']} for node in nodes}
        
        vis_edges = []
        for edge in graph.edges.values():
            if edge.source in node_positions and edge.target in node_positions:
                vis_edge = edge.to_dict()
                
                # Add visual properties
                vis_edge.update({
                    'color': self._get_edge_color(edge.edge_type),
                    'width': self._calculate_edge_width(edge.weight),
                    'opacity': self._calculate_edge_opacity(edge.confidence),
                    'style': self._get_edge_style(edge.edge_type)
                })
                
                vis_edges.append(vis_edge)
                
        return vis_edges
        
    def _get_edge_color(self, edge_type: EdgeType) -> str:
        """Get color for edge based on type."""
        
        edge_colors = {
            EdgeType.COLLABORATION: '#e74c3c',
            EdgeType.CITATION: '#3498db',
            EdgeType.SEMANTIC_SIMILARITY: '#2ecc71',
            EdgeType.CO_AUTHORSHIP: '#e67e22',
            EdgeType.DATASET_USAGE: '#9b59b6',
            EdgeType.MODEL_TRAINING: '#1abc9c',
            EdgeType.CONCEPT_RELATION: '#34495e',
            EdgeType.PROJECT_CONTRIBUTION: '#f39c12',
            EdgeType.FUNDING_RELATION: '#f1c40f'
        }
        
        return edge_colors.get(edge_type, '#95a5a6')
        
    def _calculate_edge_width(self, weight: float) -> int:
        """Calculate edge width based on weight."""
        return max(1, min(5, int(weight * 3)))
        
    def _calculate_edge_opacity(self, confidence: float) -> float:
        """Calculate edge opacity based on confidence."""
        return max(0.2, min(1.0, confidence))
        
    def _get_edge_style(self, edge_type: EdgeType) -> str:
        """Get edge style based on type."""
        
        if edge_type in [EdgeType.CITATION, EdgeType.FUNDING_RELATION]:
            return 'dashed'
        elif edge_type == EdgeType.COLLABORATION:
            return 'solid'
        else:
            return 'solid'
            
    def _prepare_opportunities_for_visualization(self, graph: InformationGraph) -> List[Dict[str, Any]]:
        """Prepare research opportunities for visualization overlay."""
        
        vis_opportunities = []
        
        for opportunity in graph.opportunities.values():
            # Find related nodes
            related_nodes = []
            for area_id in opportunity.research_areas:
                if area_id in graph.nodes:
                    node = graph.nodes[area_id]
                    related_nodes.append({
                        'id': node.id,
                        'label': node.label,
                        'position': node.position
                    })
                    
            vis_opportunity = opportunity.to_dict()
            vis_opportunity.update({
                'related_nodes': related_nodes,
                'visualization': {
                    'color': self._get_opportunity_color(opportunity.opportunity_type),
                    'size': self._calculate_opportunity_size(opportunity.impact_score),
                    'opacity': opportunity.confidence
                }
            })
            
            vis_opportunities.append(vis_opportunity)
            
        return vis_opportunities
        
    def _get_opportunity_color(self, opportunity_type) -> str:
        """Get color for opportunity based on type."""
        
        opportunity_colors = {
            'collaboration': '#e74c3c',
            'cross_domain': '#9b59b6',
            'funding': '#f1c40f',
            'dataset_sharing': '#2ecc71',
            'model_improvement': '#1abc9c',
            'knowledge_gap': '#e67e22'
        }
        
        return opportunity_colors.get(opportunity_type.value if hasattr(opportunity_type, 'value') else opportunity_type, '#95a5a6')
        
    def _calculate_opportunity_size(self, impact_score: float) -> int:
        """Calculate opportunity visualization size."""
        return max(20, min(60, int(impact_score * 50)))
        
    def _generate_interaction_data(self, graph: InformationGraph) -> Dict[str, Any]:
        """Generate data for interactive features."""
        
        return {
            'node_details': {
                node.id: {
                    'detailed_metrics': {
                        'centrality_score': node.centrality_score,
                        'influence_score': node.influence_score,
                        'opportunity_score': node.opportunity_score,
                        'research_activity': node.research_activity,
                        'connections': node.connections
                    },
                    'ftns_data': {
                        'value': float(node.ftns_value),
                        'rewards': float(node.contribution_rewards)
                    },
                    'temporal_data': {
                        'created_at': node.created_at.isoformat(),
                        'updated_at': node.updated_at.isoformat(),
                        'last_activity': node.last_activity.isoformat() if node.last_activity else None
                    }
                }
                for node in graph.nodes.values()
            },
            'filters': {
                'available_types': list(set(node.node_type.value for node in graph.nodes.values())),
                'available_tags': list(set().union(*(node.tags for node in graph.nodes.values()))),
                'score_ranges': {
                    'opportunity_min': min((node.opportunity_score for node in graph.nodes.values()), default=0),
                    'opportunity_max': max((node.opportunity_score for node in graph.nodes.values()), default=1),
                    'influence_min': min((node.influence_score for node in graph.nodes.values()), default=0),
                    'influence_max': max((node.influence_score for node in graph.nodes.values()), default=1)
                }
            },
            'search_indices': {
                'node_labels': [node.label for node in graph.nodes.values()],
                'node_descriptions': [node.description for node in graph.nodes.values()],
                'opportunity_titles': [opp.title for opp in graph.opportunities.values()]
            }
        }
        
    def _calculate_visualization_statistics(self, graph: InformationGraph) -> Dict[str, Any]:
        """Calculate statistics for the visualization."""
        
        if not graph.nodes:
            return {}
            
        return {
            'total_nodes': len(graph.nodes),
            'total_edges': len(graph.edges),
            'total_opportunities': len(graph.opportunities),
            'node_type_distribution': {
                node_type.value: sum(1 for node in graph.nodes.values() if node.node_type == node_type)
                for node_type in NodeType
            },
            'edge_type_distribution': {
                edge_type.value: sum(1 for edge in graph.edges.values() if edge.edge_type == edge_type)
                for edge_type in EdgeType
            },
            'average_metrics': {
                'opportunity_score': sum(node.opportunity_score for node in graph.nodes.values()) / len(graph.nodes),
                'influence_score': sum(node.influence_score for node in graph.nodes.values()) / len(graph.nodes),
                'research_activity': sum(node.research_activity for node in graph.nodes.values()) / len(graph.nodes),
                'connections': sum(node.connections for node in graph.nodes.values()) / len(graph.nodes)
            }
        }
        
    def _generate_legends(self, color_by: str) -> Dict[str, Any]:
        """Generate legend data for the visualization."""
        
        legends = {
            'color_scheme': color_by,
            'items': []
        }
        
        if color_by == 'type':
            for node_type, color in self.color_schemes['default'].items():
                legends['items'].append({
                    'label': node_type.value.replace('_', ' ').title(),
                    'color': color,
                    'type': 'node_type'
                })
                
        elif color_by == 'impact':
            for level, color in self.color_schemes['impact'].items():
                legends['items'].append({
                    'label': f'{level.title()} Impact',
                    'color': color,
                    'type': 'impact_level'
                })
                
        elif color_by == 'activity':
            for level, color in self.color_schemes['activity'].items():
                legends['items'].append({
                    'label': f'{level.title()} Activity',
                    'color': color,
                    'type': 'activity_level'
                })
                
        return legends
        
    def _generate_fallback_visualization(self) -> Dict[str, Any]:
        """Generate fallback visualization for error cases."""
        
        return {
            'nodes': [],
            'edges': [],
            'opportunities': [],
            'interaction': {'node_details': {}, 'filters': {}, 'search_indices': {}},
            'layout': {
                'algorithm': 'force_directed',
                'color_scheme': 'type',
                'timestamp': datetime.utcnow().isoformat()
            },
            'statistics': {},
            'legends': {'color_scheme': 'type', 'items': []},
            'error': 'Failed to generate visualization data'
        }


class GraphVisualizer:
    """High-level interface for generating Information Space visualizations."""
    
    def __init__(self):
        self.engine = VisualizationEngine()
        
    def create_interactive_visualization(
        self, 
        graph: InformationGraph, 
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create interactive visualization suitable for web frontend."""
        
        config = config or {}
        
        layout = config.get('layout', 'force_directed')
        color_by = config.get('color_by', 'type')
        filters = config.get('filters')
        
        return self.engine.generate_visualization_data(graph, layout, color_by, filters)
        
    def export_visualization_json(self, graph: InformationGraph, output_path: str) -> bool:
        """Export visualization data to JSON file."""
        
        try:
            vis_data = self.engine.generate_visualization_data(graph)
            
            with open(output_path, 'w') as f:
                json.dump(vis_data, f, indent=2, default=str)
                
            logger.info(f"Visualization data exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export visualization: {e}")
            return False