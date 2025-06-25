# PRSM LangChain Integration Guide

Complete guide to using PRSM with LangChain for advanced AI workflows, agent systems, and conversational applications.

## Overview

The PRSM LangChain integration provides seamless compatibility with the LangChain ecosystem, enabling you to use PRSM's advanced AI capabilities in existing LangChain applications or build new ones from scratch.

## Key Benefits

- **Drop-in Replacement**: Use PRSM as a direct replacement for other LLMs in LangChain
- **Advanced Capabilities**: Leverage PRSM's recursive modeling and multi-agent systems
- **Session Management**: Persistent conversation history with context preservation
- **Agent Tools**: Specialized tools for research, analysis, and external integrations
- **Cost Tracking**: Built-in FTNS cost monitoring and quality metrics
- **Async Support**: Full async/await support for high-performance applications

## Architecture

```
┌────────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   LangChain Apps    │    │   PRSM LangChain  │    │   PRSM Core       │
│                    │◄──►│   Integration     │◄──►│                  │
│ • Agents           │    │                  │    │ • NWTN Orchestrator│
│ • Chains           │    │ • PRSMLangChainLLM │    │ • Teacher Models   │
│ • Tools            │    │ • PRSMChatModel    │    │ • Agent Systems    │
│ • Memory           │    │ • PRSMAgentTools   │    │ • Session Mgmt     │
│ • Conversations    │    │ • PRSMChains       │    │ • MCP Integration  │
└────────────────────┘    └──────────────────┘    └──────────────────┘
```

## Installation

```bash
# Install LangChain
pip install langchain

# PRSM LangChain integration is included with PRSM
pip install prsm-sdk
```

## Quick Start

### Basic LLM Usage

```python
from prsm.integrations.langchain import PRSMLangChainLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Initialize PRSM LLM
llm = PRSMLangChainLLM(
    base_url="http://localhost:8000",
    api_key="your-api-key",
    default_user_id="my-app"
)

# Use directly
response = llm("Explain quantum computing in simple terms")
print(response)

# Use in LangChain chains
prompt = PromptTemplate(
    template="Explain {topic} for {audience}",
    input_variables=["topic", "audience"]
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(
    topic="machine learning",
    audience="high school students"
)
print(result)
```

### Chat Model Usage

```python
from prsm.integrations.langchain import PRSMChatModel
from langchain.schema import HumanMessage, SystemMessage

# Initialize PRSM Chat Model
chat = PRSMChatModel(
    base_url="http://localhost:8000",
    api_key="your-api-key",
    maintain_session=True
)

# Single interaction
messages = [
    SystemMessage(content="You are a helpful science tutor."),
    HumanMessage(content="What is photosynthesis?")
]

response = chat(messages)
print(response.content)

# Conversation with history
messages.extend([
    response,
    HumanMessage(content="What role does chlorophyll play?")
])

follow_up = chat(messages)
print(follow_up.content)
```

## Core Components

### 1. PRSMLangChainLLM

LangChain-compatible LLM that uses PRSM as the backend.

**Features:**
- Drop-in replacement for any LangChain LLM
- Async and sync support
- Streaming responses
- Cost and quality tracking
- Configurable context allocation

**Example:**
```python
from prsm.integrations.langchain import PRSMLangChainLLM

llm = PRSMLangChainLLM(
    base_url="http://localhost:8000",
    api_key="your-api-key",
    context_allocation=100,  # FTNS tokens per query
    quality_threshold=80,    # Minimum quality score
    timeout=30              # Request timeout
)

# Sync usage
response = llm("Your question here")

# Async usage
response = await llm.agenerate(["Question 1", "Question 2"])

# Streaming
async for chunk in llm.astream("Tell me a story"):
    print(chunk, end="")
```

### 2. PRSMChatModel

Chat model implementation with conversation history and session management.

**Features:**
- Message history preservation
- PRSM session integration
- Context-aware conversations
- Streaming support

**Example:**
```python
from prsm.integrations.langchain import PRSMChatModel
from langchain.schema import HumanMessage, AIMessage, SystemMessage

chat = PRSMChatModel(
    base_url="http://localhost:8000",
    api_key="your-api-key",
    maintain_session=True,
    max_context_length=4000
)

# Multi-turn conversation
messages = [
    SystemMessage(content="You are an expert in renewable energy."),
    HumanMessage(content="What are the main types of solar panels?"),
    AIMessage(content="There are three main types..."),
    HumanMessage(content="Which type is most efficient?")
]

response = chat(messages)
print(response.content)

# Clear session to start fresh
chat.clear_session()
```

### 3. PRSMAgentTools

Collection of PRSM-powered tools for LangChain agents.

**Available Tools:**
- **PRSMQueryTool**: Complex reasoning and analysis
- **PRSMAnalysisTool**: Specialized data analysis
- **PRSMMCPTool**: External tool integration via MCP

**Example:**
```python
from prsm.integrations.langchain import PRSMAgentTools
from prsm_sdk import PRSMClient
from langchain.agents import initialize_agent, AgentType

# Initialize PRSM client
prsm_client = PRSMClient("http://localhost:8000", "your-api-key")

# Create agent tools
tools_manager = PRSMAgentTools(
    prsm_client=prsm_client,
    default_user_id="my-agent"
)

tools = tools_manager.get_tools()

# Use with LangChain agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run(
    "Analyze the environmental impact of electric vehicles "
    "and provide recommendations for policy makers."
)
```

### 4. Custom Chains

Specialized chains for different workflow types.

**PRSMChain:**
```python
from prsm.integrations.langchain import PRSMChain

# Research workflow
research_chain = PRSMChain(
    prsm_client=prsm_client,
    workflow_type="research",
    context_allocation=150
)

result = research_chain.run({
    "input": "Impact of AI on healthcare diagnostics",
    "workflow_params": {
        "scope": "comprehensive",
        "methodology": "systematic"
    }
})

print(result["output"])
print(f"Cost: {result['metadata']['cost']} FTNS")
```

**PRSMConversationChain:**
```python
from prsm.integrations.langchain import PRSMConversationChain, PRSMChatMemory

# Enhanced conversation with context
memory = PRSMChatMemory(
    prsm_client=prsm_client,
    user_id="conversation-user"
)

conversation = PRSMConversationChain(
    llm=prsm_llm,
    memory=memory,
    conversation_type="educational",  # research, technical, analytical, creative
    context_enhancement=True
)

response = conversation.predict(
    input="Let's discuss machine learning algorithms"
)
```

### 5. Memory Integration

Persistent memory using PRSM sessions.

**PRSMChatMemory:**
```python
from prsm.integrations.langchain import PRSMChatMemory
from langchain.chains import ConversationChain

# Session-backed memory
memory = PRSMChatMemory(
    prsm_client=prsm_client,
    user_id="memory-user",
    session_prefix="chat",
    max_token_limit=2000
)

# Use with any LangChain chain
conversation = ConversationChain(
    llm=llm,
    memory=memory
)

# Conversation history is preserved across sessions
response1 = conversation.predict(input="My name is Alice")
response2 = conversation.predict(input="What's my name?")

# Get memory statistics
stats = memory.get_memory_stats()
print(f"Session ID: {stats['session_id']}")
print(f"Message count: {stats['message_count']}")
```

**PRSMSessionMemory (Multi-Context):**
```python
from prsm.integrations.langchain import PRSMSessionMemory

# Multi-context memory management
memory = PRSMSessionMemory(
    prsm_client=prsm_client,
    user_id="multi-user"
)

# Set up different contexts
memory.set_context("work", {"domain": "software engineering"})
memory.set_context("research", {"domain": "AI research"})

# Switch between contexts
memory.switch_context("work")
# Work-related conversation...

memory.switch_context("research") 
# Research-related conversation...

# List all contexts
contexts = memory.list_contexts()
print(f"Available contexts: {contexts}")
```

## Advanced Usage

### Research Assistant Agent

```python
from prsm.integrations.langchain import (
    PRSMLangChainLLM, PRSMAgentTools, PRSMChatMemory
)
from langchain.agents import initialize_agent, AgentType

# Initialize components
prsm_client = PRSMClient("http://localhost:8000", "your-api-key")
llm = PRSMLangChainLLM(prsm_client=prsm_client, context_allocation=100)
tools = PRSMAgentTools(prsm_client=prsm_client).get_tools()
memory = PRSMChatMemory(prsm_client=prsm_client, user_id="researcher")

# Create research agent
research_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    max_iterations=5
)

# Multi-step research task
result = research_agent.run(
    "Research the latest developments in quantum computing, "
    "analyze their potential impact on cryptography, "
    "and provide recommendations for cybersecurity professionals."
)

print(result)
```

### Scientific Analysis Workflow

```python
from prsm.integrations.langchain import PRSMChain

# Multi-step analysis workflow
analysis_steps = [
    {
        "workflow_type": "research",
        "input": "Climate change impact on Arctic ecosystems",
        "params": {"scope": "comprehensive", "methodology": "systematic"}
    },
    {
        "workflow_type": "analysis", 
        "input": "{{previous_result}}",  # Would use actual result
        "params": {"depth": "deep", "focus": "environmental"}
    },
    {
        "workflow_type": "synthesis",
        "input": "{{all_previous_results}}",
        "params": {"perspective": "scientific", "sources": ["research", "analysis"]}
    }
]

results = []
for step in analysis_steps:
    chain = PRSMChain(
        prsm_client=prsm_client,
        workflow_type=step["workflow_type"],
        context_allocation=200
    )
    
    result = chain.run({
        "input": step["input"],
        "workflow_params": step["params"]
    })
    
    results.append(result)
    print(f"Completed {step['workflow_type']} step")

# Final synthesis
final_result = results[-1]
print(f"\nFinal Analysis: {final_result['output']}")
print(f"Total Cost: {sum(r['metadata']['cost'] for r in results)} FTNS")
```

### Streaming Conversation Interface

```python
import asyncio
from prsm.integrations.langchain import PRSMChatModel
from langchain.schema import HumanMessage

async def streaming_chat_interface():
    chat = PRSMChatModel(
        base_url="http://localhost:8000",
        api_key="your-api-key",
        maintain_session=True
    )
    
    print("Streaming Chat Interface (type 'quit' to exit)")
    print("-" * 50)
    
    conversation_history = []
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
        
        # Add user message to history
        conversation_history.append(HumanMessage(content=user_input))
        
        print("AI: ", end="", flush=True)
        
        # Stream response
        full_response = ""
        async for chunk in chat.astream(conversation_history):
            if chunk.message.content:
                print(chunk.message.content, end="", flush=True)
                full_response += chunk.message.content
        
        print()  # New line after response
        
        # Add AI response to history
        from langchain.schema import AIMessage
        conversation_history.append(AIMessage(content=full_response))
    
    await chat.close()

# Run the interface
# asyncio.run(streaming_chat_interface())
```

## Configuration

### Environment Variables

```bash
# PRSM Configuration
PRSM_API_URL=http://localhost:8000
PRSM_API_KEY=your-api-key
PRSM_DEFAULT_USER_ID=langchain-app
PRSM_DEFAULT_CONTEXT_ALLOCATION=50
PRSM_QUALITY_THRESHOLD=75
PRSM_TIMEOUT=30

# LangChain Integration Settings
LANGCHAIN_PRSM_MAINTAIN_SESSIONS=true
LANGCHAIN_PRSM_MAX_CONTEXT_LENGTH=4000
LANGCHAIN_PRSM_MEMORY_TTL=3600
```

### Programmatic Configuration

```python
from prsm.integrations.langchain import PRSMLangChainLLM

# Custom configuration
llm = PRSMLangChainLLM(
    base_url="http://localhost:8000",
    api_key="your-api-key",
    default_user_id="my-app",
    context_allocation=100,     # FTNS per query
    quality_threshold=85,       # Minimum quality score
    timeout=45,                # Request timeout
    max_retries=3              # Retry attempts
)

# Override per query
response = llm(
    "Your question",
    user_id="specific-user",
    context_allocation=150
)
```

## Best Practices

### 1. Resource Management

```python
# Always close clients when done
async def main():
    llm = PRSMLangChainLLM(...)
    try:
        # Use the LLM
        result = await llm.agenerate(["query"])
    finally:
        await llm.close()

# Or use context managers
class PRSMContext:
    def __init__(self, **kwargs):
        self.llm = PRSMLangChainLLM(**kwargs)
    
    async def __aenter__(self):
        return self.llm
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.llm.close()

# Usage
async with PRSMContext(base_url="...", api_key="...") as llm:
    result = await llm.agenerate(["query"])
```

### 2. Cost Optimization

```python
# Monitor and optimize FTNS usage
from prsm.integrations.langchain import PRSMLangChainLLM

class CostAwareLLM:
    def __init__(self, **kwargs):
        self.llm = PRSMLangChainLLM(**kwargs)
        self.total_cost = 0
        self.query_count = 0
    
    async def query_with_tracking(self, prompt, **kwargs):
        result = await self.llm.agenerate([prompt], **kwargs)
        
        # Extract cost from result metadata
        if hasattr(result, 'llm_output') and 'total_cost' in result.llm_output:
            cost = result.llm_output['total_cost']
            self.total_cost += cost
            self.query_count += 1
            
            print(f"Query cost: {cost} FTNS (Total: {self.total_cost} FTNS)")
            print(f"Average cost: {self.total_cost / self.query_count:.2f} FTNS")
        
        return result

# Usage
cost_aware_llm = CostAwareLLM(base_url="...", api_key="...")
result = await cost_aware_llm.query_with_tracking("Your question")
```

### 3. Error Handling

```python
from prsm.integrations.langchain import PRSMLangChainLLM
from prsm.core.exceptions import PRSMException
from langchain.schema import LLMResult

async def robust_llm_call(llm: PRSMLangChainLLM, prompts: list, **kwargs):
    """Robust LLM call with error handling and fallbacks"""
    max_retries = 3
    base_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            result = await llm.agenerate(prompts, **kwargs)
            return result
            
        except PRSMException as e:
            if "rate limit" in str(e).lower():
                # Handle rate limiting with exponential backoff
                delay = base_delay * (2 ** attempt)
                print(f"Rate limited, waiting {delay}s before retry...")
                await asyncio.sleep(delay)
                continue
            elif "quality" in str(e).lower():
                # Adjust parameters for quality issues
                kwargs["context_allocation"] = kwargs.get("context_allocation", 50) + 25
                print(f"Quality issue, increasing context allocation to {kwargs['context_allocation']}")
                continue
            else:
                raise
        
        except Exception as e:
            if attempt == max_retries - 1:
                # Last attempt, return error result
                return LLMResult(
                    generations=[[{"text": f"Error: {str(e)}", "generation_info": {"error": True}}]]
                )
            else:
                print(f"Attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(base_delay)
    
    raise Exception("Max retries exceeded")

# Usage
llm = PRSMLangChainLLM(...)
result = await robust_llm_call(llm, ["Your question"])
```

### 4. Performance Optimization

```python
# Batch processing for multiple queries
async def batch_process_queries(llm: PRSMLangChainLLM, queries: list, batch_size: int = 5):
    """Process queries in batches for better performance"""
    results = []
    
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        
        # Process batch concurrently
        batch_results = await llm.agenerate(batch)
        results.extend(batch_results.generations)
        
        # Optional: Add delay between batches to avoid rate limiting
        if i + batch_size < len(queries):
            await asyncio.sleep(0.1)
    
    return results

# Usage
queries = ["Question 1", "Question 2", "Question 3", ...]
results = await batch_process_queries(llm, queries)
```

## Troubleshooting

### Common Issues

**Connection Issues:**
```python
# Test PRSM connectivity
from prsm_sdk import PRSMClient

async def test_prsm_connection():
    try:
        client = PRSMClient("http://localhost:8000", "your-api-key")
        health = await client.ping()
        print(f"PRSM health: {health}")
        await client.close()
    except Exception as e:
        print(f"Connection failed: {e}")

await test_prsm_connection()
```

**Memory Issues:**
```python
# Debug memory problems
from prsm.integrations.langchain import PRSMChatMemory

memory = PRSMChatMemory(...)
stats = memory.get_memory_stats()
print(f"Memory stats: {stats}")

# Clear memory if needed
if stats['estimated_tokens'] > 1500:
    memory.clear()
    print("Memory cleared")
```

**Quality Issues:**
```python
# Monitor and adjust for quality
from prsm.integrations.langchain import PRSMLangChainLLM

llm = PRSMLangChainLLM(
    quality_threshold=90,  # Higher threshold
    context_allocation=150  # More resources
)

# Check quality in responses
result = await llm.agenerate(["complex question"])
if hasattr(result, 'llm_output'):
    quality = result.llm_output.get('quality_score', 0)
    if quality < 85:
        print(f"Low quality response: {quality}/100")
```

## Examples

See `examples/langchain_integration_example.py` for a comprehensive demonstration of all features.

Key examples include:
- Basic LLM integration
- Chat model with conversation history
- Agent tools and workflows
- Custom chains for specialized tasks
- Memory management and persistence
- Streaming interfaces
- Error handling and optimization

## API Reference

For detailed API documentation, see the module docstrings in:
- `prsm.integrations.langchain.llm`
- `prsm.integrations.langchain.chat_model`
- `prsm.integrations.langchain.tools`
- `prsm.integrations.langchain.chains`
- `prsm.integrations.langchain.memory`

---

**Next Steps:**
- Explore the [MCP Integration Guide](./MCP_INTEGRATION_GUIDE.md) for external tool integration
- Check out [Integration Guides](./integration-guides/) for framework-specific examples
- Review the [SDK Documentation](./SDK_DOCUMENTATION.md) for core PRSM functionality
