"""
PRSM RLT Federation Framework

This module provides distributed network capabilities for RLT (Reinforcement Learning Teachers)
including teacher discovery, collaboration coordination, and federated learning.

Enhanced P2P Network Features:
- Scalable P2P networking for 50-1000+ nodes
- Production-grade Byzantine fault tolerance
- Comprehensive fault detection and recovery
- Real-time consensus mechanisms with cryptographic verification
"""

from .distributed_rlt_network import (
    DistributedRLTNetwork,
    TeacherNodeInfo,
    NetworkQualityMetrics,
    CollaborationRequest,
    CollaborationSession,
    TeacherDiscoveryQuery,
    NetworkConsensus,
    TeacherNodeStatus,
    NetworkMessageType,
    NetworkLoadBalancer,
    ReputationTracker,
    ConsensusManager,
    CollaborationCoordinator
)

# DGM Federated Evolution System
from .distributed_evolution import (
    FederatedEvolutionSystem,
    DistributedArchiveManager,
    FederatedEvolutionCoordinator,
    SynchronizationStrategy,
    NetworkEvolutionRole,
    NetworkEvolutionResult,
    SolutionSyncRequest,
    SolutionSyncResponse,
    NetworkEvolutionTask
)

from .knowledge_transfer import (
    CrossDomainKnowledgeTransferSystem,
    KnowledgeTransferRequest,
    KnowledgeTransferResult,
    KnowledgeTransferType,
    DomainType,
    AdaptedSolution,
    DomainKnowledgeExtractor,
    KnowledgeAdaptationEngine
)

# Enhanced P2P Network Components
from .scalable_p2p_network import (
    ScalableP2PNetwork,
    NetworkMetrics,
    PeerNodeInfo,
    NodeRole,
    NetworkTopology
)

from .production_fault_tolerance import (
    ProductionFaultTolerance,
    FaultEvent,
    NetworkHealth,
    FaultSeverity,
    FaultCategory,
    RecoveryAction
)

from .enhanced_consensus_system import (
    EnhancedConsensusSystem,
    EnhancedConsensusNode,
    ConsensusProposal,
    ConsensusResult,
    ConsensusMessage,
    ConsensusPhase,
    ConsensusStatus
)

__all__ = [
    # RLT Network Components
    "DistributedRLTNetwork",
    "TeacherNodeInfo",
    "NetworkQualityMetrics", 
    "CollaborationRequest",
    "CollaborationSession",
    "TeacherDiscoveryQuery",
    "NetworkConsensus",
    "TeacherNodeStatus",
    "NetworkMessageType",
    "NetworkLoadBalancer",
    "ReputationTracker", 
    "ConsensusManager",
    "CollaborationCoordinator",
    
    # DGM Federated Evolution Components
    "FederatedEvolutionSystem",
    "DistributedArchiveManager",
    "FederatedEvolutionCoordinator",
    "SynchronizationStrategy",
    "NetworkEvolutionRole",
    "NetworkEvolutionResult",
    "SolutionSyncRequest",
    "SolutionSyncResponse",
    "NetworkEvolutionTask",
    
    # Cross-Domain Knowledge Transfer Components
    "CrossDomainKnowledgeTransferSystem",
    "KnowledgeTransferRequest",
    "KnowledgeTransferResult",
    "KnowledgeTransferType",
    "DomainType",
    "AdaptedSolution",
    "DomainKnowledgeExtractor",
    "KnowledgeAdaptationEngine",
    
    # Enhanced P2P Network Components
    "ScalableP2PNetwork",
    "NetworkMetrics",
    "PeerNodeInfo",
    "NodeRole",
    "NetworkTopology",
    
    # Production Fault Tolerance Components
    "ProductionFaultTolerance",
    "FaultEvent",
    "NetworkHealth",
    "FaultSeverity",
    "FaultCategory",
    "RecoveryAction",
    
    # Enhanced Consensus System Components
    "EnhancedConsensusSystem",
    "EnhancedConsensusNode",
    "ConsensusProposal",
    "ConsensusResult",
    "ConsensusMessage",
    "ConsensusPhase",
    "ConsensusStatus"
]