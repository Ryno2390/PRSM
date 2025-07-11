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
    from prsm.nwtn import create_hybrid_agent, create_agent_team
    
    # Single agent
    agent = await create_hybrid_agent(domain="physics")
    result = await agent.process_query("What is energy conservation?")
    
    # Agent team
    team = await create_agent_team("chemistry", team_size=3)
    result = await process_team_query(team, "How do catalysts work?")
"""

# Core hybrid architecture components
from .hybrid_architecture import (
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
from .world_model_engine import (
    WorldModelEngine,
    DomainType,
    CausalRelationType,
    ValidationResult,
    CausalRelation,
    DomainModel,
    create_world_model_engine,
    create_domain_specialized_engine,
    create_base_world_model
)

# Bayesian search engine
from .bayesian_search_engine import (
    BayesianSearchEngine,
    HypothesisType,
    ExperimentMethodType,
    ExperimentStatus,
    Hypothesis,
    ExperimentDesign,
    ExperimentExecution,
    KnowledgeUpdate,
    create_bayesian_search_engine,
    create_domain_specialized_search_engine
)

# Integration layer
from .hybrid_integration import (
    HybridNWTNManager,
    create_hybrid_nwtn_manager,
    create_demo_physics_team,
    create_demo_multi_domain_network,
    demo_single_agent_query,
    demo_team_collaboration,
    demo_automated_research
)

# Convenience functions for easy usage
async def create_hybrid_agent(
    domain: str = "general",
    temperature: float = 0.7,
    agent_id: str = None
) -> HybridNWTNEngine:
    """
    Create a single hybrid agent with System 1 + System 2 architecture
    
    Args:
        domain: Knowledge domain (physics, chemistry, biology, etc.)
        temperature: Exploration temperature (0.1-1.0)
        agent_id: Optional custom agent ID
        
    Returns:
        Configured hybrid agent ready for use
    """
    manager = create_hybrid_nwtn_manager()
    return await manager.create_single_agent(agent_id, domain, temperature)


async def create_agent_team(
    domain: str,
    team_size: int = 3,
    base_temperature: float = 0.5
) -> tuple[HybridNWTNManager, list[HybridNWTNEngine]]:
    """
    Create team of agents with diverse perspectives
    
    Args:
        domain: Knowledge domain for specialization
        team_size: Number of agents in team
        base_temperature: Base temperature for perspective variation
        
    Returns:
        Tuple of (manager, agent_list) for team coordination
    """
    manager = create_hybrid_nwtn_manager()
    team = await manager.create_agent_team(domain, team_size, base_temperature)
    return manager, team


async def process_team_query(
    team: list[HybridNWTNEngine],
    query: str,
    context: dict = None
) -> dict:
    """
    Process query using team collaboration
    
    Args:
        team: List of hybrid agents
        query: Query to process
        context: Optional context dictionary
        
    Returns:
        Synthesized team response with consensus analysis
    """
    if not team:
        raise ValueError("Empty team provided")
        
    # Get manager from first agent (assumes team created by manager)
    manager = HybridNWTNManager()
    manager.agents = {agent.agent_id: agent for agent in team}
    
    domain = team[0].domain
    return await manager.process_query_with_team(query, domain, len(team), context)


async def run_automated_research(
    domain: str,
    research_query: str,
    max_iterations: int = 5
) -> dict:
    """
    Run automated research cycle
    
    Args:
        domain: Research domain
        research_query: Research question
        max_iterations: Maximum research iterations
        
    Returns:
        Comprehensive research results
    """
    manager = create_hybrid_nwtn_manager()
    return await manager.run_automated_research_cycle(domain, research_query, max_iterations)


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
    "WorldModelEngine",
    "DomainType",
    "CausalRelationType",
    "ValidationResult",
    "CausalRelation",
    "DomainModel",
    
    # Bayesian search
    "BayesianSearchEngine",
    "HypothesisType",
    "ExperimentMethodType",
    "ExperimentStatus",
    "Hypothesis",
    "ExperimentDesign",
    "ExperimentExecution",
    "KnowledgeUpdate",
    
    # Integration
    "HybridNWTNManager",
    
    # Factory functions
    "create_hybrid_nwtn_engine",
    "create_specialized_agent_team",
    "create_world_model_engine",
    "create_domain_specialized_engine",
    "create_base_world_model",
    "create_bayesian_search_engine",
    "create_domain_specialized_search_engine",
    "create_hybrid_nwtn_manager",
    
    # Convenience functions
    "create_hybrid_agent",
    "create_agent_team",
    "process_team_query",
    "run_automated_research",
    
    # Demo functions
    "create_demo_physics_team",
    "create_demo_multi_domain_network",
    "demo_single_agent_query",
    "demo_team_collaboration",
    "demo_automated_research",
]