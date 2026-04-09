"""
PRSM Federation Framework

This module provides distributed network capabilities for the PRSM P2P mesh,
including peer discovery, consensus, and fault tolerance.

Enhanced P2P Network Features:
- Scalable P2P networking for 50-1000+ nodes
- Production-grade Byzantine fault tolerance
- Comprehensive fault detection and recovery
- Real-time consensus mechanisms with cryptographic verification

v1.6.0 scope alignment: legacy DGM evolution, RLT teacher network, and
cross-domain knowledge transfer components removed. AGI-era SafetyMonitor /
CircuitBreakerNetwork hooks inside the federation primitives have been
excised (third-party LLMs handle reasoning; there is no in-network model
output to validate anymore).
"""

from .scalable_p2p_network import (
    ScalableP2PNetwork,
    NetworkMetrics,
    PeerNodeInfo,
    NodeRole,
    NetworkTopology,
)

from .production_fault_tolerance import (
    ProductionFaultTolerance,
    FaultEvent,
    NetworkHealth,
    FaultSeverity,
    FaultCategory,
    RecoveryAction,
)

from .enhanced_consensus_system import (
    EnhancedConsensusSystem,
    EnhancedConsensusNode,
    ConsensusProposal,
    ConsensusResult,
    ConsensusMessage,
    ConsensusPhase,
    ConsensusStatus,
)

__all__ = [
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
    "ConsensusStatus",
]
