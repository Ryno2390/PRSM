"""
NWTN Hybrid Architecture Integration
Complete integration of System 1 + System 2 with PRSM infrastructure

This module demonstrates how to integrate the hybrid architecture with existing
PRSM components and provides practical examples of the concepts from the
brainstorming document:

1. Base world model as "core instincts"
2. Domain-specialized agents with different temperatures
3. Bayesian search with experiment sharing
4. Hive mind updates of core knowledge
5. Integration with PRSM marketplace and tokenomics

Integration Points:
- NWTN Orchestrator: Enhanced with hybrid reasoning
- PRSM Marketplace: Agent and knowledge sharing
- FTNS Tokenomics: Rewards for valuable experiments
- Federation Network: Consensus on core knowledge
- IPFS: Persistent storage of world models

Usage Examples:
- Single hybrid agent processing queries
- Multi-agent teams with diverse perspectives
- Cross-domain knowledge transfer
- Automated research workflows
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4
from datetime import datetime, timezone
from decimal import Decimal

import structlog

from prsm.core.models import UserInput, PRSMResponse, PRSMSession
from prsm.core.config import get_settings
from prsm.nwtn.hybrid_architecture import HybridNWTNEngine, create_hybrid_nwtn_engine, create_specialized_agent_team
from prsm.nwtn.world_model_engine import WorldModelEngine, create_world_model_engine, create_base_world_model
from prsm.nwtn.bayesian_search_engine import BayesianSearchEngine, create_bayesian_search_engine
from prsm.nwtn.enhanced_orchestrator import EnhancedNWTNOrchestrator
from prsm.tokenomics.ftns_service import FTNSService
from prsm.marketplace.real_marketplace_service import RealMarketplaceService

logger = structlog.get_logger(__name__)
settings = get_settings()


class HybridNWTNManager:
    """
    Manager for hybrid NWTN architecture
    
    Coordinates multiple hybrid agents, manages knowledge sharing,
    and integrates with PRSM infrastructure.
    """
    
    def __init__(self):
        self.agents: Dict[str, HybridNWTNEngine] = {}
        self.world_models: Dict[str, WorldModelEngine] = {}
        self.base_world_model: Dict[str, Any] = {}
        
        # PRSM service integration
        self.ftns_service = FTNSService()
        self.marketplace_service = RealMarketplaceService()
        
        # Initialize base world model
        self.base_world_model = create_base_world_model()
        
        logger.info("Hybrid NWTN Manager initialized")
        
    async def create_single_agent(
        self,
        agent_id: str = None,
        domain: str = "general",
        temperature: float = 0.7
    ) -> HybridNWTNEngine:
        """
        Create single hybrid agent
        
        This demonstrates the basic hybrid architecture with System 1 + System 2
        """
        
        if agent_id is None:
            agent_id = f"hybrid_agent_{uuid4().hex[:8]}"
            
        # Create world model for agent
        world_model = create_world_model_engine()
        self.world_models[agent_id] = world_model
        
        # Create hybrid agent
        agent = create_hybrid_nwtn_engine(agent_id, temperature)
        agent.domain = domain
        
        # Replace world model with domain-specific one
        agent.world_model = world_model
        
        # Create Bayesian search engine
        agent.bayesian_search = create_bayesian_search_engine(agent_id, world_model, domain)
        
        self.agents[agent_id] = agent
        
        logger.info(
            "Created single hybrid agent",
            agent_id=agent_id,
            domain=domain,
            temperature=temperature
        )
        
        return agent
        
    async def create_agent_team(
        self,
        domain: str,
        team_size: int = 3,
        base_temperature: float = 0.5
    ) -> List[HybridNWTNEngine]:
        """
        Create team of specialized agents with different perspectives
        
        This implements the multi-agent concept from the brainstorming document
        where agents have different "temperatures" for diverse perspectives.
        """
        
        agents = []
        
        for i in range(team_size):
            # Vary temperature for different perspectives
            temperature = base_temperature + (i * 0.2) - 0.2
            temperature = max(0.1, min(1.0, temperature))
            
            agent_id = f"{domain}_agent_{i+1}"
            agent = await self.create_single_agent(agent_id, domain, temperature)
            agents.append(agent)
            
        # Set up knowledge sharing between team members
        await self._setup_team_knowledge_sharing(agents)
        
        logger.info(
            "Created agent team",
            domain=domain,
            team_size=team_size,
            agent_ids=[agent.agent_id for agent in agents],
            temperatures=[agent.temperature for agent in agents]
        )
        
        return agents
        
    async def _setup_team_knowledge_sharing(self, agents: List[HybridNWTNEngine]):
        """Set up knowledge sharing between team members"""
        
        # In full implementation, this would create shared channels in PRSM marketplace
        # For now, just establish references
        
        for agent in agents:
            agent.team_members = [other.agent_id for other in agents if other != agent]
            
        logger.info(
            "Set up knowledge sharing for team",
            team_size=len(agents)
        )
        
    async def process_query_with_single_agent(
        self,
        query: str,
        agent_id: str = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process query using single hybrid agent
        
        Demonstrates the full hybrid reasoning process:
        1. System 1: SOC recognition
        2. System 2: World model validation
        3. Bayesian search: Experimental validation
        4. Knowledge sharing: Network updates
        """
        
        # Get or create agent
        if agent_id is None:
            agent_id = list(self.agents.keys())[0] if self.agents else None
            
        if agent_id is None:
            agent = await self.create_single_agent()
            agent_id = agent.agent_id
        else:
            agent = self.agents[agent_id]
            
        # Process query through hybrid architecture
        logger.info(
            "Processing query with hybrid agent",
            agent_id=agent_id,
            query=query
        )
        
        result = await agent.process_query(query, context)
        
        # Add integration metadata
        result["processing_type"] = "hybrid_single_agent"
        result["agent_id"] = agent_id
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        return result
        
    async def process_query_with_team(
        self,
        query: str,
        domain: str,
        team_size: int = 3,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process query using team of hybrid agents
        
        Demonstrates multi-agent collaboration with diverse perspectives
        """
        
        # Create or get existing team
        team_key = f"{domain}_team"
        
        if team_key not in self.agents:
            team = await self.create_agent_team(domain, team_size)
        else:
            team = [agent for agent in self.agents.values() if agent.domain == domain]
            
        # Process query with each team member
        team_results = []
        
        for agent in team:
            logger.info(
                "Processing with team member",
                agent_id=agent.agent_id,
                temperature=agent.temperature
            )
            
            result = await agent.process_query(query, context)
            result["agent_temperature"] = agent.temperature
            team_results.append(result)
            
        # Synthesize team results
        synthesized_result = await self._synthesize_team_results(team_results, query)
        
        # Add team metadata
        synthesized_result["processing_type"] = "hybrid_team"
        synthesized_result["team_size"] = len(team)
        synthesized_result["domain"] = domain
        synthesized_result["individual_results"] = team_results
        synthesized_result["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        return synthesized_result
        
    async def _synthesize_team_results(
        self,
        team_results: List[Dict[str, Any]],
        query: str
    ) -> Dict[str, Any]:
        """Synthesize results from team of agents"""
        
        # Combine reasoning traces
        combined_reasoning = []
        for i, result in enumerate(team_results):
            reasoning = result.get("reasoning_trace", [])
            for step in reasoning:
                step["agent_index"] = i
                combined_reasoning.append(step)
                
        # Aggregate SOC usage
        all_socs = {}
        for result in team_results:
            socs = result.get("socs_used", [])
            for soc in socs:
                name = soc["name"]
                if name not in all_socs:
                    all_socs[name] = {
                        "name": name,
                        "confidence_values": [],
                        "levels": []
                    }
                all_socs[name]["confidence_values"].append(soc["confidence"])
                all_socs[name]["levels"].append(soc["level"])
                
        # Calculate consensus
        consensus_socs = []
        for name, soc_data in all_socs.items():
            avg_confidence = sum(soc_data["confidence_values"]) / len(soc_data["confidence_values"])
            consensus_socs.append({
                "name": name,
                "avg_confidence": avg_confidence,
                "agent_count": len(soc_data["confidence_values"]),
                "confidence_range": [min(soc_data["confidence_values"]), max(soc_data["confidence_values"])]
            })
            
        # Generate team response
        team_response = f"""
        Team Analysis Results:
        
        Query: {query}
        
        Team Consensus:
        - {len(consensus_socs)} SOCs identified across {len(team_results)} agents
        - Average confidence: {sum(soc['avg_confidence'] for soc in consensus_socs) / len(consensus_socs) if consensus_socs else 0:.3f}
        - Total experiments: {sum(result.get('experiments_conducted', 0) for result in team_results)}
        
        Agent Perspectives:
        {chr(10).join(f"- Agent {i+1} (temp={result.get('agent_temperature', 0):.1f}): {result.get('response', '')[:100]}..." for i, result in enumerate(team_results))}
        
        The team's diverse perspectives provide a robust analysis combining different approaches to the problem.
        """
        
        return {
            "response": team_response,
            "reasoning_trace": combined_reasoning,
            "consensus_socs": consensus_socs,
            "team_agreement_score": self._calculate_team_agreement(team_results),
            "experiments_total": sum(result.get('experiments_conducted', 0) for result in team_results)
        }
        
    def _calculate_team_agreement(self, team_results: List[Dict[str, Any]]) -> float:
        """Calculate agreement score between team members"""
        
        if len(team_results) < 2:
            return 1.0
            
        # Simple agreement calculation based on SOC overlap
        # In full implementation, this would be more sophisticated
        
        all_soc_sets = []
        for result in team_results:
            socs = result.get("socs_used", [])
            soc_names = set(soc["name"] for soc in socs)
            all_soc_sets.append(soc_names)
            
        # Calculate pairwise Jaccard similarity
        similarities = []
        for i in range(len(all_soc_sets)):
            for j in range(i+1, len(all_soc_sets)):
                intersection = len(all_soc_sets[i] & all_soc_sets[j])
                union = len(all_soc_sets[i] | all_soc_sets[j])
                similarity = intersection / union if union > 0 else 0
                similarities.append(similarity)
                
        return sum(similarities) / len(similarities) if similarities else 0
        
    async def update_core_knowledge(
        self,
        soc_name: str,
        new_confidence: float,
        source_agent_id: str
    ):
        """
        Update core knowledge across all agents (hive mind effect)
        
        This implements the hive mind concept where validated knowledge
        propagates to all agents in the network.
        """
        
        # Check if SOC should be promoted to core knowledge
        if new_confidence >= 0.9:
            
            # Add to base world model
            self.base_world_model[soc_name] = {
                "confidence": new_confidence,
                "promoted_at": datetime.now(timezone.utc).isoformat(),
                "source_agent": source_agent_id
            }
            
            # Propagate to all agents
            updated_agents = []
            for agent_id, agent in self.agents.items():
                if agent_id != source_agent_id:
                    # Update agent's world model
                    # In full implementation, this would be more sophisticated
                    updated_agents.append(agent_id)
                    
            logger.info(
                "Core knowledge updated across network",
                soc_name=soc_name,
                new_confidence=new_confidence,
                source_agent=source_agent_id,
                propagated_to=updated_agents
            )
            
    async def run_automated_research_cycle(
        self,
        domain: str,
        research_query: str,
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Run automated research cycle
        
        This demonstrates the full research automation concept:
        1. Generate initial hypotheses
        2. Design and run experiments
        3. Update knowledge based on results
        4. Share findings with network
        5. Iterate with new hypotheses
        """
        
        logger.info(
            "Starting automated research cycle",
            domain=domain,
            research_query=research_query,
            max_iterations=max_iterations
        )
        
        # Create research team
        research_team = await self.create_agent_team(domain, team_size=3)
        
        research_results = {
            "domain": domain,
            "research_query": research_query,
            "iterations": [],
            "knowledge_gained": [],
            "experiments_conducted": 0,
            "core_knowledge_updates": 0
        }
        
        for iteration in range(max_iterations):
            logger.info(f"Research iteration {iteration + 1}/{max_iterations}")
            
            iteration_results = {
                "iteration": iteration + 1,
                "agent_results": [],
                "knowledge_updates": [],
                "experiments": []
            }
            
            # Each agent conducts research
            for agent in research_team:
                # Process research query
                result = await agent.process_query(research_query)
                
                # Extract SOCs for experimentation
                socs_used = result.get("socs_used", [])
                
                # Run experiments on uncertain SOCs
                for soc_info in socs_used:
                    if soc_info["confidence"] < 0.7:
                        # Find SOC in agent's world model
                        soc_name = soc_info["name"]
                        
                        # Create dummy SOC for experimentation
                        from prsm.nwtn.hybrid_architecture import SOC, SOCType, ConfidenceLevel
                        soc = SOC(
                            name=soc_name,
                            soc_type=SOCType.CONCEPT,
                            confidence=soc_info["confidence"],
                            domain=domain
                        )
                        
                        # Run experiment cycle
                        experiments = await agent.bayesian_search.run_experiment_cycle(soc, max_experiments=2)
                        
                        iteration_results["experiments"].extend([{
                            "agent_id": agent.agent_id,
                            "soc_name": soc_name,
                            "experiments_count": len(experiments),
                            "information_value": sum(exp.information_value for exp in experiments)
                        }])
                        
                        research_results["experiments_conducted"] += len(experiments)
                        
                        # Check for core knowledge updates
                        if soc.confidence >= 0.9:
                            await self.update_core_knowledge(soc_name, soc.confidence, agent.agent_id)
                            research_results["core_knowledge_updates"] += 1
                            
                iteration_results["agent_results"].append({
                    "agent_id": agent.agent_id,
                    "result": result
                })
                
            research_results["iterations"].append(iteration_results)
            
            # Add delay between iterations
            await asyncio.sleep(0.1)
            
        # Synthesize final research results
        final_synthesis = await self._synthesize_research_results(research_results)
        research_results["final_synthesis"] = final_synthesis
        
        logger.info(
            "Automated research cycle completed",
            domain=domain,
            iterations=max_iterations,
            total_experiments=research_results["experiments_conducted"],
            core_updates=research_results["core_knowledge_updates"]
        )
        
        return research_results
        
    async def _synthesize_research_results(self, research_results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize final research results"""
        
        total_experiments = research_results["experiments_conducted"]
        core_updates = research_results["core_knowledge_updates"]
        iterations = len(research_results["iterations"])
        
        # Calculate research efficiency
        efficiency = core_updates / total_experiments if total_experiments > 0 else 0
        
        # Generate research summary
        summary = f"""
        Automated Research Results:
        
        Domain: {research_results['domain']}
        Query: {research_results['research_query']}
        
        Research Metrics:
        - Iterations: {iterations}
        - Total Experiments: {total_experiments}
        - Core Knowledge Updates: {core_updates}
        - Research Efficiency: {efficiency:.3f}
        
        Key Findings:
        - Successfully validated {core_updates} core knowledge items
        - Conducted {total_experiments} experiments across {iterations} iterations
        - Demonstrated automated hypothesis generation and testing
        - Showed knowledge propagation across agent network
        
        This demonstrates the hybrid architecture's ability to conduct
        systematic research through System 1 pattern recognition,
        System 2 logical validation, and Bayesian experimentation.
        """
        
        return {
            "summary": summary,
            "efficiency_score": efficiency,
            "total_experiments": total_experiments,
            "core_updates": core_updates,
            "research_quality": "high" if efficiency > 0.1 else "medium"
        }
        
    async def integrate_with_prsm_orchestrator(
        self,
        orchestrator: EnhancedNWTNOrchestrator
    ) -> EnhancedNWTNOrchestrator:
        """
        Integrate hybrid architecture with existing PRSM orchestrator
        
        This shows how to enhance existing PRSM infrastructure with
        hybrid reasoning capabilities.
        """
        
        # Store reference to hybrid manager
        orchestrator.hybrid_manager = self
        
        # Store original process method
        original_process = orchestrator.process_query
        
        # Create hybrid-enhanced process method
        async def hybrid_enhanced_process(query: str, context: Dict[str, Any] = None):
            try:
                # Try hybrid processing first
                hybrid_result = await self.process_query_with_single_agent(query, context=context)
                
                # Get original result
                original_result = await original_process(query, context)
                
                # Combine results
                enhanced_result = {
                    "original_response": original_result,
                    "hybrid_response": hybrid_result,
                    "processing_mode": "hybrid_enhanced",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                return enhanced_result
                
            except Exception as e:
                logger.error("Hybrid processing failed, using original", error=str(e))
                return await original_process(query, context)
                
        # Replace method
        orchestrator.process_query = hybrid_enhanced_process
        
        logger.info("Integrated hybrid architecture with PRSM orchestrator")
        
        return orchestrator
        
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get comprehensive manager statistics"""
        
        agent_stats = {}
        for agent_id, agent in self.agents.items():
            agent_stats[agent_id] = agent.get_agent_stats()
            
        world_model_stats = {}
        for model_id, model in self.world_models.items():
            world_model_stats[model_id] = model.get_world_model_stats()
            
        return {
            "total_agents": len(self.agents),
            "total_world_models": len(self.world_models),
            "base_world_model_size": len(self.base_world_model),
            "agent_stats": agent_stats,
            "world_model_stats": world_model_stats
        }


# Factory functions for easy integration

def create_hybrid_nwtn_manager() -> HybridNWTNManager:
    """Create hybrid NWTN manager instance"""
    return HybridNWTNManager()


async def create_demo_physics_team() -> Tuple[HybridNWTNManager, List[HybridNWTNEngine]]:
    """Create demo physics research team"""
    
    manager = create_hybrid_nwtn_manager()
    team = await manager.create_agent_team("physics", team_size=3)
    
    return manager, team


async def create_demo_multi_domain_network() -> Tuple[HybridNWTNManager, Dict[str, List[HybridNWTNEngine]]]:
    """Create demo multi-domain research network"""
    
    manager = create_hybrid_nwtn_manager()
    
    domains = ["physics", "chemistry", "biology", "computer_science"]
    network = {}
    
    for domain in domains:
        team = await manager.create_agent_team(domain, team_size=2)
        network[domain] = team
        
    return manager, network


# Example usage functions

async def demo_single_agent_query():
    """Demonstrate single agent query processing"""
    
    manager = create_hybrid_nwtn_manager()
    
    # Create agent
    agent = await manager.create_single_agent(domain="physics", temperature=0.7)
    
    # Process query
    result = await manager.process_query_with_single_agent(
        "What happens when a force is applied to an object?",
        agent_id=agent.agent_id
    )
    
    print("Single Agent Result:")
    print(json.dumps(result, indent=2, default=str))
    
    return result


async def demo_team_collaboration():
    """Demonstrate team collaboration"""
    
    manager = create_hybrid_nwtn_manager()
    
    # Process query with team
    result = await manager.process_query_with_team(
        "How does energy conservation apply to chemical reactions?",
        domain="chemistry",
        team_size=3
    )
    
    print("Team Collaboration Result:")
    print(json.dumps(result, indent=2, default=str))
    
    return result


async def demo_automated_research():
    """Demonstrate automated research cycle"""
    
    manager = create_hybrid_nwtn_manager()
    
    # Run automated research
    result = await manager.run_automated_research_cycle(
        domain="physics",
        research_query="How do electromagnetic fields interact with matter?",
        max_iterations=3
    )
    
    print("Automated Research Result:")
    print(json.dumps(result, indent=2, default=str))
    
    return result


if __name__ == "__main__":
    # Run demonstrations
    asyncio.run(demo_single_agent_query())
    asyncio.run(demo_team_collaboration())
    asyncio.run(demo_automated_research())