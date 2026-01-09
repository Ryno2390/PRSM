"""
NWTN (Newton) - Neural Web for Transformation Networking
Hybrid AI Architecture Implementation for PRSM

This module implements the complete hybrid AI architecture described in the
brainstorming document, featuring:

System 1 (Fast/Intuitive):
- Transformer-based pattern recognition
- Rapid SOC (Subject/Object/Concept) identification
- Multi-provider AI integration (OpenAI, Anthropic, Ollama)

System 2 (Slow/Logical):
- First-principles world model reasoning
- Causal relationship validation
- Logical consistency checking

Learning & Collaboration:
- Bayesian search with automated experimentation
- Knowledge sharing across PRSM network
- Hive mind updates for core knowledge
- Threshold-based SOC confidence management

Key Components:
- HybridNWTNEngine: Main hybrid reasoning engine
- WorldModelEngine: First-principles reasoning system
- BayesianSearchEngine: Automated experimentation
- HybridNWTNManager: Multi-agent coordination

Usage:
    from prsm.compute.nwtn import create_hybrid_agent, create_agent_team
    
    # Single agent
    agent = await create_hybrid_agent(domain="physics")
    result = await agent.process_query("What is energy conservation?")
    
    # Agent team
    team = await create_agent_team("chemistry", team_size=3)
    result = await process_team_query(team, "How do catalysts work?")
"""

# Core hybrid architecture components
from .architectures.hybrid_architecture import (
    HybridNWTNEngine,
    SOC,
    SOCType,
    ConfidenceLevel,
    ExperimentResult,
    ExperimentType,
    create_hybrid_nwtn_engine,
    create_specialized_agent_team,
    integrate_with_nwtn_orchestrator
)

# World model engine
from .engines.world_model_engine import (
    NeuroSymbolicEngine,
    ScientificConstraint,
    get_world_model
)

# Backward compatibility alias
WorldModelEngine = NeuroSymbolicEngine


# Note: BayesianSearchEngine components are archived as they are not currently used

# Note: HybridNWTNManager components are archived as they are not currently used

# Note: Convenience functions are archived as HybridNWTNManager is not currently used
# The current NWTN pipeline uses EnhancedNWTNOrchestrator instead


# Export all components
__all__ = [
    # Core architecture
    "HybridNWTNEngine",
    "SOC",
    "SOCType", 
    "ConfidenceLevel",
    "ExperimentResult",
    "ExperimentType",
    
    # World model
    "NeuroSymbolicEngine",
    "WorldModelEngine",
    "ScientificConstraint",
    "get_world_model",

    
    # Note: Bayesian search, integration, and convenience functions are archived
    
    # Factory functions (only those still available)
    "create_hybrid_nwtn_engine",
    "create_specialized_agent_team",
    "create_world_model_engine",
    "create_domain_specialized_engine",
    "create_base_world_model",
]