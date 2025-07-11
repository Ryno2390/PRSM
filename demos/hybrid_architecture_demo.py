#!/usr/bin/env python3
"""
NWTN Hybrid Architecture Demo
Demonstrates the System 1 + System 2 hybrid AI architecture

This demo showcases the key concepts from the brainstorming document:
1. System 1: Fast transformer-based pattern recognition
2. System 2: Slow first-principles world model reasoning
3. Bayesian search with automated experimentation
4. Knowledge sharing and hive mind updates
5. Multi-agent teams with diverse perspectives

Usage:
    python demos/hybrid_architecture_demo.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from prsm.nwtn import (
    create_hybrid_agent,
    create_agent_team,
    process_team_query,
    run_automated_research,
    demo_single_agent_query,
    demo_team_collaboration,
    demo_automated_research
)


async def demonstrate_hybrid_architecture():
    """Main demonstration of hybrid architecture capabilities"""
    
    print("🧠 NWTN Hybrid Architecture Demo")
    print("=" * 50)
    print()
    
    # Demo 1: Single Agent Processing
    print("🔍 Demo 1: Single Agent Hybrid Processing")
    print("-" * 40)
    
    try:
        # Create a physics-specialized agent
        agent = await create_hybrid_agent(
            domain="physics",
            temperature=0.7
        )
        
        # Process a physics query
        result = await agent.process_query(
            "What happens when you apply a force to an object with mass?"
        )
        
        print(f"Agent ID: {agent.agent_id}")
        print(f"Domain: {agent.domain}")
        print(f"Temperature: {agent.temperature}")
        print()
        print("Query: What happens when you apply a force to an object with mass?")
        print()
        print("Response:", result.get("response", ""))
        print()
        print("Reasoning Trace:")
        for i, step in enumerate(result.get("reasoning_trace", []), 1):
            print(f"  {i}. {step.get('step', '')}: {step.get('description', '')}")
        print()
        print("SOCs Used:")
        for soc in result.get("socs_used", []):
            print(f"  - {soc.get('name', '')} (confidence: {soc.get('confidence', 0):.3f}, level: {soc.get('level', '')})")
        print()
        
    except Exception as e:
        print(f"Error in single agent demo: {e}")
    
    print("=" * 50)
    print()
    
    # Demo 2: Multi-Agent Team Collaboration
    print("👥 Demo 2: Multi-Agent Team Collaboration")
    print("-" * 40)
    
    try:
        # Create a chemistry team
        manager, team = await create_agent_team(
            domain="chemistry",
            team_size=3,
            base_temperature=0.5
        )
        
        # Process query with team
        result = await process_team_query(
            team,
            "How do catalysts speed up chemical reactions?"
        )
        
        print(f"Team Size: {len(team)}")
        print(f"Domain: chemistry")
        print(f"Agent Temperatures: {[agent.temperature for agent in team]}")
        print()
        print("Query: How do catalysts speed up chemical reactions?")
        print()
        print("Team Response:", result.get("response", ""))
        print()
        print("Team Statistics:")
        print(f"  - Agreement Score: {result.get('team_agreement_score', 0):.3f}")
        print(f"  - Consensus SOCs: {len(result.get('consensus_socs', []))}")
        print(f"  - Total Experiments: {result.get('experiments_total', 0)}")
        print()
        
    except Exception as e:
        print(f"Error in team collaboration demo: {e}")
    
    print("=" * 50)
    print()
    
    # Demo 3: Automated Research Cycle
    print("🔬 Demo 3: Automated Research Cycle")
    print("-" * 40)
    
    try:
        # Run automated research
        result = await run_automated_research(
            domain="physics",
            research_query="What is the relationship between energy and mass?",
            max_iterations=3
        )
        
        print(f"Research Domain: {result.get('domain', '')}")
        print(f"Research Query: {result.get('research_query', '')}")
        print(f"Iterations: {len(result.get('iterations', []))}")
        print(f"Total Experiments: {result.get('experiments_conducted', 0)}")
        print(f"Core Knowledge Updates: {result.get('core_knowledge_updates', 0)}")
        print()
        
        # Show final synthesis
        synthesis = result.get('final_synthesis', {})
        print("Research Summary:")
        print(synthesis.get('summary', 'No summary available'))
        print()
        
    except Exception as e:
        print(f"Error in automated research demo: {e}")
    
    print("=" * 50)
    print()
    
    # Demo 4: Architecture Explanation
    print("📚 Demo 4: Architecture Explanation")
    print("-" * 40)
    
    print("""
    The NWTN Hybrid Architecture implements key concepts from the brainstorming document:
    
    🧠 System 1 (Fast/Intuitive):
    - Uses transformer models for rapid pattern recognition
    - Identifies Subjects, Objects, and Concepts (SOCs) from input
    - Leverages PRSM's multi-provider AI integration
    
    🔍 System 2 (Slow/Logical):
    - Validates SOCs against first-principles world models
    - Checks logical consistency and causal relationships
    - Maintains domain-specific knowledge hierarchies
    
    🔬 Bayesian Search:
    - Generates hypotheses about uncertain SOCs
    - Conducts automated experiments to test hypotheses
    - Updates confidence using Bayesian methods
    - Values both successes AND failures for learning
    
    🌐 Hive Mind:
    - Shares validated knowledge across agent network
    - Propagates core knowledge updates to all agents
    - Creates collective intelligence through collaboration
    
    👥 Multi-Agent Teams:
    - Different 'temperatures' create diverse perspectives
    - Agents specialize in different domains
    - Team collaboration provides robust analysis
    
    This architecture addresses LLM limitations by:
    - Combining fast pattern matching with logical reasoning
    - Grounding knowledge in first principles
    - Enabling continuous learning through experimentation
    - Distributing intelligence across collaborative networks
    """)
    
    print("=" * 50)
    print()
    print("🎉 Demo completed! The hybrid architecture is ready for integration.")
    print("   Key features demonstrated:")
    print("   ✅ System 1 + System 2 coordination")
    print("   ✅ SOC-based knowledge representation")
    print("   ✅ Bayesian experimental learning")
    print("   ✅ Multi-agent collaboration")
    print("   ✅ Automated research workflows")
    print()


if __name__ == "__main__":
    asyncio.run(demonstrate_hybrid_architecture())