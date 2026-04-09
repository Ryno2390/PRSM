"""
PRSM Federation Framework

This module provides distributed network capabilities for the PRSM P2P mesh,
including peer discovery, consensus, and fault tolerance.

Enhanced P2P Network Features:
- Scalable P2P networking for 50-1000+ nodes
- Production-grade Byzantine fault tolerance
- Comprehensive fault detection and recovery
- Real-time consensus mechanisms with cryptographic verification
"""

# Enhanced P2P Network Components (in-scope, kept through v1.6.0)
# Wrapped in try/except because these files import from prsm.core.safety which is
# being removed in PR 3; prsm.core.safety stubs will be replaced in PR 4/PR 5.
try:
    from .scalable_p2p_network import (
        ScalableP2PNetwork,
        NetworkMetrics,
        PeerNodeInfo,
        NodeRole,
        NetworkTopology
    )
except (ImportError, ModuleNotFoundError):
    ScalableP2PNetwork = None  # type: ignore[assignment,misc]
    NetworkMetrics = None  # type: ignore[assignment,misc]
    PeerNodeInfo = None  # type: ignore[assignment,misc]
    NodeRole = None  # type: ignore[assignment,misc]
    NetworkTopology = None  # type: ignore[assignment,misc]

try:
    from .production_fault_tolerance import (
        ProductionFaultTolerance,
        FaultEvent,
        NetworkHealth,
        FaultSeverity,
        FaultCategory,
        RecoveryAction
    )
except (ImportError, ModuleNotFoundError):
    ProductionFaultTolerance = None  # type: ignore[assignment,misc]
    FaultEvent = None  # type: ignore[assignment,misc]
    NetworkHealth = None  # type: ignore[assignment,misc]
    FaultSeverity = None  # type: ignore[assignment,misc]
    FaultCategory = None  # type: ignore[assignment,misc]
    RecoveryAction = None  # type: ignore[assignment,misc]

try:
    from .enhanced_consensus_system import (
        EnhancedConsensusSystem,
        EnhancedConsensusNode,
        ConsensusProposal,
        ConsensusResult,
        ConsensusMessage,
        ConsensusPhase,
        ConsensusStatus
    )
except (ImportError, ModuleNotFoundError):
    EnhancedConsensusSystem = None  # type: ignore[assignment,misc]
    EnhancedConsensusNode = None  # type: ignore[assignment,misc]
    ConsensusProposal = None  # type: ignore[assignment,misc]
    ConsensusResult = None  # type: ignore[assignment,misc]
    ConsensusMessage = None  # type: ignore[assignment,misc]
    ConsensusPhase = None  # type: ignore[assignment,misc]
    ConsensusStatus = None  # type: ignore[assignment,misc]

# Legacy RLT/evolution/knowledge-transfer imports (scheduled for deletion in PR 4)
# Wrapped in try/except to avoid breaking the package when these files have
# broken transitive dependencies on agents/executors/ (deleted in PR 1).
try:
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
except (ImportError, ModuleNotFoundError):
    DistributedRLTNetwork = None  # type: ignore[assignment,misc]
    TeacherNodeInfo = None  # type: ignore[assignment,misc]
    NetworkQualityMetrics = None  # type: ignore[assignment,misc]
    CollaborationRequest = None  # type: ignore[assignment,misc]
    CollaborationSession = None  # type: ignore[assignment,misc]
    TeacherDiscoveryQuery = None  # type: ignore[assignment,misc]
    NetworkConsensus = None  # type: ignore[assignment,misc]
    TeacherNodeStatus = None  # type: ignore[assignment,misc]
    NetworkMessageType = None  # type: ignore[assignment,misc]
    NetworkLoadBalancer = None  # type: ignore[assignment,misc]
    ReputationTracker = None  # type: ignore[assignment,misc]
    ConsensusManager = None  # type: ignore[assignment,misc]
    CollaborationCoordinator = None  # type: ignore[assignment,misc]

try:
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
except (ImportError, ModuleNotFoundError):
    FederatedEvolutionSystem = None  # type: ignore[assignment,misc]
    DistributedArchiveManager = None  # type: ignore[assignment,misc]
    FederatedEvolutionCoordinator = None  # type: ignore[assignment,misc]
    SynchronizationStrategy = None  # type: ignore[assignment,misc]
    NetworkEvolutionRole = None  # type: ignore[assignment,misc]
    NetworkEvolutionResult = None  # type: ignore[assignment,misc]
    SolutionSyncRequest = None  # type: ignore[assignment,misc]
    SolutionSyncResponse = None  # type: ignore[assignment,misc]
    NetworkEvolutionTask = None  # type: ignore[assignment,misc]

try:
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
except (ImportError, ModuleNotFoundError):
    CrossDomainKnowledgeTransferSystem = None  # type: ignore[assignment,misc]
    KnowledgeTransferRequest = None  # type: ignore[assignment,misc]
    KnowledgeTransferResult = None  # type: ignore[assignment,misc]
    KnowledgeTransferType = None  # type: ignore[assignment,misc]
    DomainType = None  # type: ignore[assignment,misc]
    AdaptedSolution = None  # type: ignore[assignment,misc]
    DomainKnowledgeExtractor = None  # type: ignore[assignment,misc]
    KnowledgeAdaptationEngine = None  # type: ignore[assignment,misc]

__all__ = [
    # Enhanced P2P Network Components (active)
    "ScalableP2PNetwork",
    "NetworkMetrics",
    "PeerNodeInfo",
    "NodeRole",
    "NetworkTopology",

    # Production Fault Tolerance Components (active)
    "ProductionFaultTolerance",
    "FaultEvent",
    "NetworkHealth",
    "FaultSeverity",
    "FaultCategory",
    "RecoveryAction",

    # Enhanced Consensus System Components (active)
    "EnhancedConsensusSystem",
    "EnhancedConsensusNode",
    "ConsensusProposal",
    "ConsensusResult",
    "ConsensusMessage",
    "ConsensusPhase",
    "ConsensusStatus",

    # RLT Network Components (legacy, scheduled for PR 4 deletion)
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

    # DGM Federated Evolution Components (legacy, scheduled for PR 4 deletion)
    "FederatedEvolutionSystem",
    "DistributedArchiveManager",
    "FederatedEvolutionCoordinator",
    "SynchronizationStrategy",
    "NetworkEvolutionRole",
    "NetworkEvolutionResult",
    "SolutionSyncRequest",
    "SolutionSyncResponse",
    "NetworkEvolutionTask",

    # Cross-Domain Knowledge Transfer Components (legacy, scheduled for PR 4 deletion)
    "CrossDomainKnowledgeTransferSystem",
    "KnowledgeTransferRequest",
    "KnowledgeTransferResult",
    "KnowledgeTransferType",
    "DomainType",
    "AdaptedSolution",
    "DomainKnowledgeExtractor",
    "KnowledgeAdaptationEngine",
]
