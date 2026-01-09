#!/usr/bin/env python3
"""
PRSM LangChain Integration Example
=================================

Comprehensive example demonstrating how to use PRSM with LangChain for
advanced AI workflows, agent systems, and conversational applications.
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from langchain.agents import initialize_agent, AgentType
    from langchain.chains import LLMChain, ConversationChain
    from langchain.prompts import PromptTemplate
    from langchain.schema import HumanMessage, AIMessage, SystemMessage
except ImportError:
    print("❌ LangChain is required for this example.")
    print("Install with: pip install langchain")
    sys.exit(1)

from prsm_sdk import PRSMClient
from prsm.integrations.mcp import MCPClient
from prsm.integrations.langchain import (
    PRSMLangChainLLM,
    PRSMChatModel,
    PRSMAgentTools,
    PRSMChain,
    PRSMConversationChain,
    PRSMChatMemory,
)


class LangChainIntegrationDemo:
    """Comprehensive demonstration of PRSM LangChain integration"""
    
    def __init__(self):
        # Initialize PRSM clients
        self.prsm_client = PRSMClient(
            base_url="http://localhost:8000",
            api_key="demo-api-key"
        )
        
        # Initialize MCP client (optional)
        try:
            self.mcp_client = MCPClient("http://localhost:3000")
        except Exception:
            self.mcp_client = None
            print("⚠️ MCP client not available, some features will be limited")
        
        # Initialize LangChain components
        self.prsm_llm = None
        self.prsm_chat = None
        self.agent_tools = None
    
    async def run_demo(self):
        """Run the complete LangChain integration demo"""
        print("🦜 PRSM LangChain Integration Demo")
        print("=" * 50)
        
        try:
            # Step 1: Basic LLM Integration
            await self.demo_basic_llm_integration()
            
            # Step 2: Chat Model Integration
            await self.demo_chat_model_integration()
            
            # Step 3: Agent Tools Integration
            await self.demo_agent_tools_integration()
            
            # Step 4: Custom Chains
            await self.demo_custom_chains()
            
            # Step 5: Memory Integration
            await self.demo_memory_integration()
            
            # Step 6: Advanced Agent Workflows
            await self.demo_advanced_agent_workflows()
            
            print("\n🎉 LangChain Integration Demo Complete!")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
        finally:
            await self.cleanup()
    
    async def demo_basic_llm_integration(self):
        """Demonstrate basic LLM integration"""
        print("\n🔧 1. Basic LLM Integration")
        print("-" * 30)
        
        # Initialize PRSM LLM
        self.prsm_llm = PRSMLangChainLLM(
            base_url="http://localhost:8000",
            api_key="demo-api-key",
            default_user_id="langchain-demo",
            context_allocation=75
        )
        
        print("✅ Initialized PRSM LangChain LLM")
        
        # Test basic query
        print("\n📝 Testing basic query...")
        try:
            # Simulate LLM call
            response = await self._simulate_llm_call(
                "Explain the concept of recursion in computer science with a simple example."
            )
            print(f"🤖 Response: {response[:100]}...")
            
        except Exception as e:
            print(f"⚠️ LLM call simulation: {e}")
        
        # Test with LangChain prompt template
        print("\n📋 Testing with LangChain prompt template...")
        try:
            prompt_template = PromptTemplate(
                input_variables=["topic", "audience"],
                template="Explain {topic} in simple terms for {audience}. Provide practical examples."
            )
            
            # Simulate chain execution
            formatted_prompt = prompt_template.format(
                topic="machine learning",
                audience="high school students"
            )
            
            response = await self._simulate_llm_call(formatted_prompt)
            print(f"🎯 Formatted response: {response[:100]}...")
            
        except Exception as e:
            print(f"⚠️ Template simulation: {e}")
    
    async def demo_chat_model_integration(self):
        """Demonstrate chat model integration"""
        print("\n💬 2. Chat Model Integration")
        print("-" * 30)
        
        # Initialize PRSM Chat Model
        self.prsm_chat = PRSMChatModel(
            base_url="http://localhost:8000",
            api_key="demo-api-key",
            default_user_id="chat-demo",
            maintain_session=True
        )
        
        print("✅ Initialized PRSM Chat Model")
        
        # Test conversation with history
        print("\n🗣️ Testing conversation with message history...")
        try:
            messages = [
                SystemMessage(content="You are a helpful AI assistant specializing in scientific explanations."),
                HumanMessage(content="What is quantum entanglement?"),
                AIMessage(content="Quantum entanglement is a quantum mechanical phenomenon where pairs or groups of particles become interconnected..."),
                HumanMessage(content="Can you give me a practical application of this phenomenon?")
            ]
            
            # Simulate chat response
            response = await self._simulate_chat_call(messages)
            print(f"🔬 Chat response: {response[:100]}...")
            
        except Exception as e:
            print(f"⚠️ Chat simulation: {e}")
        
        # Test streaming response
        print("\n🌊 Testing streaming response...")
        try:
            messages = [HumanMessage(content="Explain photosynthesis step by step.")]
            await self._simulate_streaming_chat(messages)
            
        except Exception as e:
            print(f"⚠️ Streaming simulation: {e}")
    
    async def demo_agent_tools_integration(self):
        """Demonstrate agent tools integration"""
        print("\n🛠️ 3. Agent Tools Integration")
        print("-" * 30)
        
        # Initialize PRSM Agent Tools
        self.agent_tools = PRSMAgentTools(
            prsm_client=self.prsm_client,
            mcp_client=self.mcp_client,
            default_user_id="agent-demo"
        )
        
        tools = self.agent_tools.get_tools()
        print(f"✅ Initialized {len(tools)} PRSM agent tools")
        
        # List available tools
        print("\n📋 Available tools:")
        for tool in tools:
            print(f"  • {tool.name}: {tool.description[:60]}...")
        
        # Test individual tools
        print("\n🔨 Testing PRSM Query Tool...")
        try:
            query_tool = self.agent_tools.get_tool_by_name("prsm_query")
            if query_tool:
                # Simulate tool execution
                result = await self._simulate_tool_execution(
                    query_tool,
                    {
                        "prompt": "Analyze the environmental impact of renewable energy adoption",
                        "context_allocation": 100
                    }
                )
                print(f"🌱 Tool result: {result[:100]}...")
        
        except Exception as e:
            print(f"⚠️ Tool simulation: {e}")
        
        # Test Analysis Tool
        print("\n📊 Testing PRSM Analysis Tool...")
        try:
            analysis_tool = self.agent_tools.get_tool_by_name("prsm_analysis")
            if analysis_tool:
                result = await self._simulate_tool_execution(
                    analysis_tool,
                    {
                        "data": "Global temperature data shows a 1.2°C increase since 1900. Arctic ice coverage has decreased by 13% per decade since 1979.",
                        "analysis_type": "scientific",
                        "depth": "deep"
                    }
                )
                print(f"🌡️ Analysis result: {result[:100]}...")
        
        except Exception as e:
            print(f"⚠️ Analysis simulation: {e}")
    
    async def demo_custom_chains(self):
        """Demonstrate custom PRSM chains"""
        print("\n⛓️ 4. Custom Chains Integration")
        print("-" * 30)
        
        # Test Research Chain
        print("\n🔬 Testing PRSM Research Chain...")
        try:
            research_chain = PRSMChain(
                prsm_client=self.prsm_client,
                workflow_type="research",
                context_allocation=150
            )
            
            result = await self._simulate_chain_execution(
                research_chain,
                {
                    "input": "Impact of artificial intelligence on healthcare diagnostics",
                    "workflow_params": {
                        "scope": "comprehensive",
                        "methodology": "systematic"
                    }
                }
            )
            
            print(f"📈 Research result: {result['output'][:100]}...")
            print(f"💰 Cost: {result['metadata']['cost']} FTNS")
            
        except Exception as e:
            print(f"⚠️ Research chain simulation: {e}")
        
        # Test Analysis Chain
        print("\n📋 Testing PRSM Analysis Chain...")
        try:
            analysis_chain = PRSMChain(
                prsm_client=self.prsm_client,
                workflow_type="analysis",
                context_allocation=100
            )
            
            result = await self._simulate_chain_execution(
                analysis_chain,
                {
                    "input": "The rise of remote work has fundamentally changed organizational structures and employee expectations.",
                    "workflow_params": {
                        "depth": "comprehensive",
                        "focus": "organizational"
                    }
                }
            )
            
            print(f"🏢 Analysis result: {result['output'][:100]}...")
            print(f"⚡ Processing time: {result['metadata']['processing_time']:.2f}s")
            
        except Exception as e:
            print(f"⚠️ Analysis chain simulation: {e}")
    
    async def demo_memory_integration(self):
        """Demonstrate memory integration"""
        print("\n🧠 5. Memory Integration")
        print("-" * 30)
        
        # Initialize PRSM Chat Memory
        try:
            memory = PRSMChatMemory(
                prsm_client=self.prsm_client,
                user_id="memory-demo",
                session_prefix="demo"
            )
            
            print("✅ Initialized PRSM Chat Memory")
            print(f"📊 Memory stats: {memory.get_memory_stats()}")
            
            # Test conversation with memory
            print("\n💭 Testing conversation with persistent memory...")
            
            conversation_chain = PRSMConversationChain(
                llm=self.prsm_llm,
                memory=memory,
                conversation_type="educational",
                prsm_client=self.prsm_client
            )
            
            # Simulate multi-turn conversation
            turns = [
                "Let's discuss machine learning algorithms.",
                "What's the difference between supervised and unsupervised learning?",
                "Can you give me examples of each type?",
                "Which approach would be better for customer segmentation?"
            ]
            
            for i, turn in enumerate(turns, 1):
                print(f"\n👤 Turn {i}: {turn}")
                try:
                    response = await self._simulate_conversation_turn(
                        conversation_chain, turn
                    )
                    print(f"🤖 Response: {response[:80]}...")
                except Exception as e:
                    print(f"⚠️ Turn simulation: {e}")
            
            # Show final memory stats
            print(f"\n📈 Final memory stats: {memory.get_memory_stats()}")
            
        except Exception as e:
            print(f"⚠️ Memory demo failed: {e}")
    
    async def demo_advanced_agent_workflows(self):
        """Demonstrate advanced agent workflows"""
        print("\n🤖 6. Advanced Agent Workflows")
        print("-" * 30)
        
        if not self.agent_tools:
            print("⚠️ Agent tools not available, skipping advanced workflows")
            return
        
        # Simulate agent with PRSM tools
        print("\n🎯 Testing multi-step agent workflow...")
        try:
            # This would typically use LangChain's agent framework
            # For demo purposes, we'll simulate the workflow
            
            workflow_steps = [
                {
                    "action": "prsm_query",
                    "input": {
                        "prompt": "What are the key challenges in sustainable energy storage?",
                        "context_allocation": 75
                    },
                    "description": "Initial research query"
                },
                {
                    "action": "prsm_analysis",
                    "input": {
                        "data": "Battery technology limitations, grid integration challenges, cost factors, environmental considerations",
                        "analysis_type": "technical",
                        "depth": "moderate"
                    },
                    "description": "Technical analysis of challenges"
                },
                {
                    "action": "prsm_query",
                    "input": {
                        "prompt": "Based on the analysis, what are the most promising emerging solutions for energy storage?",
                        "context_allocation": 100
                    },
                    "description": "Solution-focused follow-up query"
                }
            ]
            
            agent_results = []
            for i, step in enumerate(workflow_steps, 1):
                print(f"\n🔄 Step {i}: {step['description']}")
                
                tool = self.agent_tools.get_tool_by_name(step['action'])
                if tool:
                    try:
                        result = await self._simulate_tool_execution(tool, step['input'])
                        agent_results.append(result)
                        print(f"✅ Result: {result[:60]}...")
                    except Exception as e:
                        print(f"⚠️ Step failed: {e}")
            
            print(f"\n🏁 Agent workflow completed with {len(agent_results)} successful steps")
            
        except Exception as e:
            print(f"⚠️ Advanced workflow simulation: {e}")
    
    async def _simulate_llm_call(self, prompt: str) -> str:
        """Simulate LLM call for demo purposes"""
        # In a real implementation, this would call the actual LLM
        await asyncio.sleep(0.1)  # Simulate processing time
        return f"Simulated PRSM response to: '{prompt[:30]}...'"
    
    async def _simulate_chat_call(self, messages: List) -> str:
        """Simulate chat model call for demo purposes"""
        await asyncio.sleep(0.1)
        last_message = messages[-1].content if messages else "empty"
        return f"Simulated PRSM chat response to: '{last_message[:30]}...'"
    
    async def _simulate_streaming_chat(self, messages: List) -> None:
        """Simulate streaming chat response"""
        chunks = ["Photosynthesis ", "is the process ", "by which plants ", "convert sunlight..."]
        print("🌊 Streaming response: ", end="")
        for chunk in chunks:
            print(chunk, end="", flush=True)
            await asyncio.sleep(0.2)
        print("\n")
    
    async def _simulate_tool_execution(self, tool, inputs: Dict[str, Any]) -> str:
        """Simulate tool execution for demo purposes"""
        await asyncio.sleep(0.1)
        return f"Simulated {tool.name} result for input: {str(inputs)[:50]}..."
    
    async def _simulate_chain_execution(self, chain, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate chain execution for demo purposes"""
        await asyncio.sleep(0.2)
        return {
            "output": f"Simulated {chain.workflow_type} chain result for: {inputs.get('input', 'unknown')[:50]}...",
            "metadata": {
                "workflow_type": chain.workflow_type,
                "cost": 45.5,
                "processing_time": 2.3,
                "quality_score": 88
            }
        }
    
    async def _simulate_conversation_turn(self, chain, input_text: str) -> str:
        """Simulate conversation turn for demo purposes"""
        await asyncio.sleep(0.1)
        return f"Simulated conversation response to: '{input_text[:30]}...'"
    
    async def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        
        try:
            if self.prsm_llm:
                await self.prsm_llm.close()
                print("   Closed PRSM LLM")
            
            if self.prsm_chat:
                await self.prsm_chat.close()
                print("   Closed PRSM Chat Model")
            
            if self.prsm_client:
                await self.prsm_client.close()
                print("   Closed PRSM Client")
            
            if self.mcp_client:
                await self.mcp_client.disconnect()
                print("   Disconnected MCP Client")
                
        except Exception as e:
            print(f"   Cleanup warning: {e}")


def show_integration_overview():
    """Show overview of PRSM LangChain integration capabilities"""
    print("""
🦜 PRSM LangChain Integration Overview
=====================================

📦 Available Components:

1. PRSMLangChainLLM
   • Drop-in replacement for LangChain LLMs
   • Async and sync support
   • Streaming capabilities
   • Cost and quality tracking

2. PRSMChatModel  
   • Chat-based interface with conversation history
   • Session management integration
   • Message formatting and context preservation

3. PRSMAgentTools
   • Collection of PRSM-powered tools for agents
   • Query tool for complex reasoning
   • Analysis tool for data processing
   • MCP tool for external service integration

4. PRSMChain & PRSMConversationChain
   • Specialized workflows (research, analysis, problem-solving)
   • Enhanced conversation chains with context
   • Workflow-specific prompt engineering

5. PRSMChatMemory & PRSMSessionMemory
   • PRSM session-backed memory
   • Multi-context memory management
   • Persistent conversation history

🎯 Use Cases:
• AI-powered research assistants
• Multi-agent systems with specialized tools
• Conversational AI with long-term memory
• Complex reasoning and analysis workflows
• Integration with existing LangChain applications

🚀 Getting Started:
   from prsm.integrations.langchain import PRSMLangChainLLM
   
   llm = PRSMLangChainLLM(base_url="http://localhost:8000", api_key="your-key")
   response = llm("Your question here")
""")


async def main():
    """Main entry point"""
    # Show overview
    show_integration_overview()
    
    # Run demo
    demo = LangChainIntegrationDemo()
    await demo.run_demo()


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())
