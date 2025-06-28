"""
PRSM RLT Federation Framework

This module provides distributed network capabilities for RLT (Reinforcement Learning Teachers)
including teacher discovery, collaboration coordination, and federated learning.
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
    "KnowledgeAdaptationEngine"
]