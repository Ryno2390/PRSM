#!/usr/bin/env python3
"""
PRSM Python SDK Quick Start Examples

This file contains practical examples demonstrating the core capabilities
of the PRSM platform using the Python SDK.

Examples included:
1. Basic agent creation and execution
2. Multi-model orchestration
3. P2P network integration
4. Cost optimization
5. Real-time streaming
6. Multi-agent collaboration
7. Custom workflows

Run this file to see PRSM in action!
"""

import asyncio
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Import PRSM SDK (mock implementation for demonstration)
# In practice, install with: pip install prsm-sdk
class PRSMClient:
    """Mock PRSM client for demonstration purposes"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or os.getenv("PRSM_API_KEY", "demo_key")
        self.base_url = base_url or "https://api.prsm.ai"
        self.agents = AgentManager(self)
        self.models = ModelManager(self)
        self.p2p = P2PManager(self)
        self.workflows = WorkflowManager(self)
        self.cost = CostManager(self)
        
    def authenticate(self):
        """Authenticate with PRSM network"""
        print(f"🔐 Authenticating with PRSM API...")
        print(f"✅ Connected to {self.base_url}")
        return True

class AgentManager:
    def __init__(self, client):
        self.client = client
    
    def create(self, **kwargs):
        return Agent(kwargs)
    
    def list(self, **kwargs):
        return [Agent({"id": f"agent_{i}", "name": f"demo_agent_{i}"}) for i in range(3)]

class ModelManager:
    def __init__(self, client):
        self.client = client
    
    def list_providers(self):
        return ["openai", "anthropic", "huggingface", "local"]
    
    def execute_inference(self, **kwargs):
        return {"output": "Sample inference result", "confidence": 0.95}

class P2PManager:
    def __init__(self, client):
        self.client = client
    
    def create_node(self, **kwargs):
        return P2PNode(kwargs)
    
    def discover_peers(self):
        return [{"id": f"peer_{i}", "capabilities": ["inference"]} for i in range(5)]

class WorkflowManager:
    def __init__(self, client):
        self.client = client
    
    def create(self, name, steps):
        return Workflow(name, steps)

class CostManager:
    def __init__(self, client):
        self.client = client
    
    def get_usage(self):
        return {"total_cost": 145.67, "tokens_used": 125000}

class Agent:
    def __init__(self, config):
        self.id = config.get("id", f"agent_{hash(str(config)) % 1000}")
        self.name = config.get("name", "demo_agent")
        self.type = config.get("type", "researcher")
        self.config = config
        
    def execute(self, prompt, **kwargs):
        return ExecutionResult({
            "execution_id": f"exec_{self.id}",
            "status": "completed",
            "output": f"Response to: {prompt[:50]}...",
            "confidence": 0.92,
            "tokens_used": 150
        })
    
    def stream(self, prompt, **kwargs):
        """Simulate streaming response"""
        import time
        stages = ["Starting analysis...", "Processing data...", "Generating insights...", "Finalizing results..."]
        for i, stage in enumerate(stages):
            yield {"progress": (i + 1) * 25, "status": stage, "partial_output": f"Partial result {i+1}"}
            time.sleep(0.5)

class ExecutionResult:
    def __init__(self, data):
        self.execution_id = data["execution_id"]
        self.status = data["status"]
        self.output = data["output"]
        self.confidence = data["confidence"]
        self.tokens_used = data["tokens_used"]

class P2PNode:
    def __init__(self, config):
        self.id = f"node_{hash(str(config)) % 1000}"
        self.config = config
    
    def start_services(self):
        print(f"🚀 Starting P2P services for node {self.id}")
        return True
    
    def get_earnings(self):
        return {"ftns_earned": 45.67, "usd_equivalent": 228.35}

class Workflow:
    def __init__(self, name, steps):
        self.name = name
        self.steps = steps
        self.id = f"workflow_{hash(name) % 1000}"
    
    def execute(self, data):
        return {"workflow_id": self.id, "status": "completed", "results": "Workflow completed successfully"}

# Initialize PRSM client
client = PRSMClient()

def example_1_basic_agent_creation():
    """Example 1: Create and use a basic AI agent"""
    print("\n" + "="*60)
    print("🤖 Example 1: Basic Agent Creation and Execution")
    print("="*60)
    
    # Authenticate
    client.authenticate()
    
    # Create a research agent
    researcher = client.agents.create(
        name="scientific_researcher",
        type="researcher",
        model_provider="openai",
        model_name="gpt-4",
        capabilities=[
            "literature_search",
            "data_analysis",
            "hypothesis_generation",
            "report_writing"
        ],
        specialized_knowledge="machine_learning,artificial_intelligence,distributed_systems"
    )
    
    print(f"✅ Created agent: {researcher.name} (ID: {researcher.id})")
    
    # Execute a research task
    research_prompt = """
    Analyze the current state of federated learning research.
    Focus on:
    1. Recent algorithmic improvements
    2. Privacy preservation techniques
    3. Challenges in heterogeneous environments
    4. Applications in healthcare and finance
    
    Provide a comprehensive summary with key insights.
    """
    
    print("🔄 Executing research task...")
    result = researcher.execute(
        prompt=research_prompt,
        context={
            "domain": "machine_learning",
            "priority": "high",
            "deadline": (datetime.now() + timedelta(hours=2)).isoformat()
        }
    )
    
    print(f"📊 Execution completed:")
    print(f"   Status: {result.status}")
    print(f"   Confidence: {result.confidence:.2%}")
    print(f"   Tokens used: {result.tokens_used}")
    print(f"   Output preview: {result.output[:100]}...")
    
    return researcher

def example_2_multi_model_orchestration():
    """Example 2: Intelligent multi-model orchestration"""
    print("\n" + "="*60)
    print("🎭 Example 2: Multi-Model Orchestration")
    print("="*60)
    
    # Create agents with different model providers
    agents = {
        "reasoning_agent": client.agents.create(
            name="deep_reasoner",
            model_provider="openai",
            model_name="gpt-4",
            capabilities=["complex_reasoning", "problem_solving"]
        ),
        "analysis_agent": client.agents.create(
            name="data_analyst",
            model_provider="anthropic", 
            model_name="claude-3-sonnet",
            capabilities=["data_analysis", "statistical_modeling"]
        ),
        "coding_agent": client.agents.create(
            name="code_generator",
            model_provider="huggingface",
            model_name="codellama-34b",
            capabilities=["code_generation", "debugging"]
        )
    }
    
    print(f"✅ Created {len(agents)} specialized agents")
    
    # Define task routing rules
    task_router = {
        "complex_reasoning": "reasoning_agent",
        "data_analysis": "analysis_agent", 
        "code_generation": "coding_agent"
    }
    
    # Execute different types of tasks
    tasks = [
        {
            "type": "complex_reasoning",
            "prompt": "Explain the philosophical implications of AGI development on society"
        },
        {
            "type": "data_analysis", 
            "prompt": "Analyze customer churn patterns in subscription businesses"
        },
        {
            "type": "code_generation",
            "prompt": "Create a Python class for real-time data streaming with error handling"
        }
    ]
    
    print("🔄 Executing tasks with optimal agent routing...")
    
    for i, task in enumerate(tasks, 1):
        agent_name = task_router[task["type"]]
        agent = agents[agent_name]
        
        print(f"\n📝 Task {i}: {task['type']}")
        print(f"   🤖 Assigned to: {agent.name}")
        
        result = agent.execute(task["prompt"])
        print(f"   ✅ Completed with {result.confidence:.1%} confidence")
        print(f"   📊 Tokens: {result.tokens_used}")
    
    return agents

def example_3_p2p_network_integration():
    """Example 3: Join and utilize P2P network"""
    print("\n" + "="*60)
    print("🌐 Example 3: P2P Network Integration")
    print("="*60)
    
    # Discover available peers
    print("🔍 Discovering P2P network peers...")
    peers = client.p2p.discover_peers()
    print(f"   Found {len(peers)} available peers")
    
    for peer in peers[:3]:
        print(f"   - Peer {peer['id']}: {peer['capabilities']}")
    
    # Create a P2P node
    print("\n🏗️ Creating P2P node...")
    node = client.p2p.create_node(
        node_type="inference_provider",
        capabilities=["llama2-7b", "stable-diffusion", "whisper"],
        resources={
            "gpu_memory": "24GB",
            "cpu_cores": 16,
            "storage": "1TB"
        },
        pricing={
            "inference_rate": 0.001,  # per token
            "availability": "24/7"
        }
    )
    
    print(f"✅ Created P2P node: {node.id}")
    
    # Start providing services
    node.start_services()
    print("🚀 Node is now providing inference services to the network")
    
    # Check earnings (simulated)
    earnings = node.get_earnings()
    print(f"\n💰 Node earnings:")
    print(f"   FTNS tokens earned: {earnings['ftns_earned']}")
    print(f"   USD equivalent: ${earnings['usd_equivalent']:.2f}")
    
    # Use P2P network for inference
    print("\n🔄 Using P2P network for distributed inference...")
    
    p2p_tasks = [
        "Generate a Python function for data encryption",
        "Create a marketing strategy for a tech startup",
        "Analyze sentiment in customer reviews"
    ]
    
    for task in p2p_tasks:
        print(f"   📤 Submitting: {task[:40]}...")
        # In real implementation, this would route to optimal P2P peer
        result = client.models.execute_inference(
            prompt=task,
            provider="p2p_network",
            routing="automatic"
        )
        print(f"   📥 Received result with {result['confidence']:.1%} confidence")
    
    return node

def example_4_cost_optimization():
    """Example 4: Cost analysis and optimization"""
    print("\n" + "="*60)
    print("💰 Example 4: Cost Optimization")
    print("="*60)
    
    # Get current usage statistics
    print("📊 Analyzing current usage...")
    usage = client.cost.get_usage()
    print(f"   Total cost this month: ${usage['total_cost']:.2f}")
    print(f"   Tokens used: {usage['tokens_used']:,}")
    
    # Cost optimization analysis
    print("\n🎯 Running cost optimization analysis...")
    
    # Simulate cost comparison across providers
    providers = ["openai", "anthropic", "huggingface", "p2p_network"]
    cost_analysis = {}
    
    for provider in providers:
        # Simulated cost calculation
        base_cost = {"openai": 0.002, "anthropic": 0.0015, "huggingface": 0.001, "p2p_network": 0.0005}
        monthly_cost = usage["tokens_used"] * base_cost[provider]
        cost_analysis[provider] = {
            "monthly_cost": monthly_cost,
            "cost_per_token": base_cost[provider],
            "savings_vs_current": usage["total_cost"] - monthly_cost
        }
    
    print("📋 Cost analysis by provider:")
    for provider, analysis in cost_analysis.items():
        savings = analysis["savings_vs_current"]
        savings_pct = (savings / usage["total_cost"]) * 100
        print(f"   {provider:12}: ${analysis['monthly_cost']:8.2f} ({savings_pct:+5.1f}%)")
    
    # Find best option
    best_provider = min(cost_analysis.keys(), key=lambda p: cost_analysis[p]["monthly_cost"])
    best_savings = cost_analysis[best_provider]["savings_vs_current"]
    
    print(f"\n💡 Optimization recommendation:")
    print(f"   Best provider: {best_provider}")
    print(f"   Potential savings: ${best_savings:.2f}/month ({(best_savings/usage['total_cost']*100):.1f}%)")
    
    # ROI calculation for P2P participation
    print(f"\n🌐 P2P Network ROI Analysis:")
    p2p_investment = 500  # Initial hardware investment
    monthly_earnings = 85.50  # Estimated monthly earnings
    payback_months = p2p_investment / monthly_earnings
    annual_roi = (monthly_earnings * 12 - p2p_investment) / p2p_investment * 100
    
    print(f"   Initial investment: ${p2p_investment}")
    print(f"   Monthly earnings: ${monthly_earnings}")
    print(f"   Payback period: {payback_months:.1f} months")
    print(f"   Annual ROI: {annual_roi:.1f}%")
    
    return cost_analysis

def example_5_real_time_streaming():
    """Example 5: Real-time streaming execution"""
    print("\n" + "="*60)
    print("📡 Example 5: Real-time Streaming")
    print("="*60)
    
    # Create a streaming-capable agent
    streaming_agent = client.agents.create(
        name="real_time_analyst",
        type="analyst",
        model_provider="anthropic",
        model_name="claude-3-haiku",  # Faster model for streaming
        capabilities=["real_time_analysis", "data_processing"]
    )
    
    print(f"✅ Created streaming agent: {streaming_agent.name}")
    
    # Execute task with streaming
    streaming_prompt = """
    Perform a comprehensive analysis of the current AI market trends.
    Include:
    1. Major players and their strategies
    2. Emerging technologies and applications
    3. Investment patterns and funding trends
    4. Regulatory developments
    5. Future predictions for 2025
    
    Provide detailed insights with supporting data.
    """
    
    print("🔄 Starting streaming execution...")
    print("📺 Real-time updates:")
    
    # Simulate streaming response
    for update in streaming_agent.stream(streaming_prompt):
        progress_bar = "█" * (update["progress"] // 5) + "░" * (20 - update["progress"] // 5)
        print(f"   [{progress_bar}] {update['progress']:3d}% - {update['status']}")
        
        if update.get("partial_output"):
            print(f"   💬 {update['partial_output']}")
    
    print("✅ Streaming execution completed!")
    
    return streaming_agent

def example_6_multi_agent_collaboration():
    """Example 6: Multi-agent team collaboration"""
    print("\n" + "="*60)
    print("👥 Example 6: Multi-Agent Collaboration")
    print("="*60)
    
    # Create a collaborative research team
    team_agents = {
        "lead_researcher": client.agents.create(
            name="project_leader",
            type="orchestrator",
            capabilities=["project_management", "research_coordination"]
        ),
        "data_scientist": client.agents.create(
            name="data_expert", 
            type="analyst",
            capabilities=["statistical_analysis", "machine_learning"]
        ),
        "domain_expert": client.agents.create(
            name="biology_specialist",
            type="specialist",
            specialized_knowledge="molecular_biology,genetics,bioinformatics"
        ),
        "technical_writer": client.agents.create(
            name="science_communicator",
            type="specialist",
            capabilities=["technical_writing", "visualization"]
        )
    }
    
    print(f"✅ Created research team with {len(team_agents)} agents")
    
    # Define collaborative research project
    research_project = {
        "title": "CRISPR-Cas9 Efficiency Analysis",
        "objective": "Analyze factors affecting CRISPR-Cas9 gene editing efficiency",
        "tasks": [
            {
                "agent": "data_scientist",
                "task": "Statistical analysis of editing efficiency data",
                "deliverable": "Statistical report with key findings"
            },
            {
                "agent": "domain_expert", 
                "task": "Biological interpretation of efficiency patterns",
                "deliverable": "Mechanistic insights and hypotheses"
            },
            {
                "agent": "technical_writer",
                "task": "Synthesize findings into comprehensive report",
                "deliverable": "Publication-ready manuscript"
            }
        ]
    }
    
    print(f"🎯 Project: {research_project['title']}")
    print(f"📋 Objective: {research_project['objective']}")
    
    # Execute collaborative workflow
    print("\n🔄 Executing collaborative research workflow...")
    
    project_results = {}
    
    for i, task in enumerate(research_project["tasks"], 1):
        agent = team_agents[task["agent"]]
        print(f"\n📝 Task {i}: {task['task']}")
        print(f"   👤 Assigned to: {agent.name}")
        
        # Execute task (with previous results as context)
        context = {"previous_results": project_results}
        result = agent.execute(task["task"], context=context)
        
        project_results[task["agent"]] = {
            "deliverable": task["deliverable"],
            "result": result.output,
            "confidence": result.confidence
        }
        
        print(f"   ✅ Completed: {task['deliverable']}")
        print(f"   📊 Confidence: {result.confidence:.1%}")
    
    # Lead researcher synthesizes final results
    print(f"\n🎯 Final synthesis by project leader...")
    final_synthesis = team_agents["lead_researcher"].execute(
        prompt="Synthesize the research findings into executive summary",
        context={"team_results": project_results}
    )
    
    print(f"📄 Final research summary generated")
    print(f"   Confidence: {final_synthesis.confidence:.1%}")
    print(f"   Length: {len(final_synthesis.output)} characters")
    
    return team_agents, project_results

def example_7_custom_workflow():
    """Example 7: Custom scientific workflow"""
    print("\n" + "="*60)
    print("🔬 Example 7: Custom Scientific Workflow")
    print("="*60)
    
    # Define a drug discovery workflow
    workflow = client.workflows.create(
        name="drug_discovery_pipeline",
        steps=[
            {
                "name": "target_identification",
                "agent_type": "bioinformatics_specialist",
                "input": "disease_pathway_data",
                "output": "potential_targets"
            },
            {
                "name": "compound_screening",
                "agent_type": "cheminformatics_expert", 
                "input": "potential_targets",
                "output": "candidate_compounds"
            },
            {
                "name": "admet_prediction",
                "agent_type": "pharmacokinetics_modeler",
                "input": "candidate_compounds",
                "output": "filtered_compounds"
            },
            {
                "name": "efficacy_modeling",
                "agent_type": "systems_biologist",
                "input": "filtered_compounds",
                "output": "efficacy_predictions"
            },
            {
                "name": "report_generation",
                "agent_type": "regulatory_writer",
                "input": "efficacy_predictions", 
                "output": "research_report"
            }
        ]
    )
    
    print(f"✅ Created workflow: {workflow.name}")
    print(f"   Pipeline steps: {len(workflow.steps)}")
    
    # Execute the workflow
    input_data = {
        "disease_pathway_data": "alzheimer_pathway_analysis.json",
        "compound_libraries": ["chembl_library.sdf", "zinc_database.sdf"],
        "experimental_constraints": {
            "blood_brain_barrier": True,
            "oral_bioavailability": ">30%",
            "toxicity_threshold": "low"
        }
    }
    
    print(f"\n🔄 Executing drug discovery workflow...")
    print(f"   Input data: {list(input_data.keys())}")
    
    workflow_result = workflow.execute(input_data)
    
    print(f"✅ Workflow execution completed:")
    print(f"   Status: {workflow_result['status']}")
    print(f"   Workflow ID: {workflow_result['workflow_id']}")
    print(f"   Results: {workflow_result['results']}")
    
    # Workflow metrics
    print(f"\n📊 Workflow Performance:")
    print(f"   Total execution time: 45.3 minutes")
    print(f"   Steps completed: 5/5")
    print(f"   Success rate: 100%")
    print(f"   Total cost: $12.45")
    
    return workflow, workflow_result

async def example_8_async_operations():
    """Example 8: Asynchronous operations for performance"""
    print("\n" + "="*60)
    print("⚡ Example 8: Asynchronous Operations")
    print("="*60)
    
    # Create multiple agents for parallel execution
    agents = [
        client.agents.create(name=f"async_agent_{i}", type="analyst")
        for i in range(5)
    ]
    
    print(f"✅ Created {len(agents)} agents for parallel processing")
    
    # Define parallel tasks
    tasks = [
        "Analyze market trends in renewable energy",
        "Evaluate investment opportunities in AI startups", 
        "Research developments in quantum computing",
        "Study impact of remote work on productivity",
        "Examine blockchain adoption in supply chain"
    ]
    
    print("🔄 Executing tasks in parallel...")
    
    # Simulate async execution
    async def execute_task(agent, task):
        # In real implementation, this would be truly async
        await asyncio.sleep(0.5)  # Simulate network delay
        result = agent.execute(task)
        return {
            "agent": agent.name,
            "task": task[:40] + "...",
            "result": result,
            "execution_time": 2.3
        }
    
    # Execute all tasks concurrently
    start_time = datetime.now()
    
    async_results = await asyncio.gather(*[
        execute_task(agent, task) 
        for agent, task in zip(agents, tasks)
    ])
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    print(f"✅ All tasks completed in {total_time:.1f} seconds")
    print(f"📊 Results summary:")
    
    for result in async_results:
        print(f"   {result['agent']}: {result['task']} ({result['execution_time']}s)")
    
    # Performance comparison
    sequential_time = sum(r["execution_time"] for r in async_results)
    speedup = sequential_time / total_time
    
    print(f"\n⚡ Performance improvement:")
    print(f"   Sequential execution: {sequential_time:.1f}s")
    print(f"   Parallel execution: {total_time:.1f}s")
    print(f"   Speedup: {speedup:.1f}x")
    
    return async_results

def main():
    """Run all PRSM examples"""
    print("🚀 PRSM Python SDK - Comprehensive Examples")
    print("=" * 60)
    print("This demonstration showcases the key capabilities of PRSM:")
    print("- AI agent management and orchestration")
    print("- Multi-model inference and routing")
    print("- P2P network participation")
    print("- Cost optimization and ROI analysis")
    print("- Real-time streaming and collaboration")
    print("- Custom scientific workflows")
    print("- Asynchronous operations")
    
    try:
        # Run all examples
        agent = example_1_basic_agent_creation()
        agents = example_2_multi_model_orchestration() 
        node = example_3_p2p_network_integration()
        cost_analysis = example_4_cost_optimization()
        streaming_agent = example_5_real_time_streaming()
        team_agents, project_results = example_6_multi_agent_collaboration()
        workflow, workflow_result = example_7_custom_workflow()
        
        # Run async example
        async_results = asyncio.run(example_8_async_operations())
        
        # Summary
        print("\n" + "="*60)
        print("🎉 All Examples Completed Successfully!")
        print("="*60)
        print("Summary of what we accomplished:")
        print(f"✅ Created {len(agents) + 1} AI agents")
        print(f"✅ Executed {len(cost_analysis)} cost optimization scenarios") 
        print(f"✅ Deployed P2P node: {node.id}")
        print(f"✅ Completed collaborative research project")
        print(f"✅ Executed scientific workflow: {workflow.name}")
        print(f"✅ Processed {len(async_results)} parallel tasks")
        
        print("\n🎯 Next Steps:")
        print("- Explore the interactive API documentation")
        print("- Join the PRSM community on Discord")
        print("- Contribute to open-source development")
        print("- Build your own AI-powered applications")
        
        print(f"\n📚 Learn more: https://docs.prsm.network")
        print(f"💬 Community: https://discord.gg/prsm")
        print(f"🐙 GitHub: https://github.com/Ryno2390/PRSM")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Please check your API key and network connection")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)