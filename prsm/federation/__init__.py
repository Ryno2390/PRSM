"""
PRSM Federation Module
Distributed model registry and P2P federation for AI model discovery
Includes both legacy simulation and production-ready implementations
"""

# Legacy simulation implementations (for compatibility)
from .consensus import get_consensus, ConsensusType
from .model_registry import ModelRegistry, model_registry
from .p2p_network import get_p2p_network

# Production-ready implementations
from .enhanced_p2p_network import get_production_p2p_network, ProductionP2PNetwork
from .production_consensus import get_production_consensus, ProductionConsensus
from .distributed_model_registry import get_production_model_registry, ProductionModelRegistry

__all__ = [
    # Legacy implementations
    "ModelRegistry", 
    "model_registry",
    "get_consensus",
    "ConsensusType", 
    "get_p2p_network",
    
    # Production implementations
    "get_production_p2p_network",
    "ProductionP2PNetwork",
    "get_production_consensus", 
    "ProductionConsensus",
    "get_production_model_registry",
    "ProductionModelRegistry"
]