"""
PRSM Information Space Module

Comprehensive knowledge visualization and research collaboration system.
Builds on existing PRSM infrastructure to provide rich information mapping.
"""

from .models import (
    InfoNode, InfoEdge, ResearchOpportunity, 
    InformationGraph, ContentAnalysis
)
from .analyzer import ContentAnalyzer, SemanticAnalyzer
from .visualizer import GraphVisualizer, VisualizationEngine
from .service import InformationSpaceService
from .api import InformationSpaceAPI

__all__ = [
    'InfoNode', 'InfoEdge', 'ResearchOpportunity', 'InformationGraph',
    'ContentAnalysis', 'ContentAnalyzer', 'SemanticAnalyzer',
    'GraphVisualizer', 'VisualizationEngine', 'InformationSpaceService',
    'InformationSpaceAPI'
]

__version__ = "1.0.0"