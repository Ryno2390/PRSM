#!/usr/bin/env python3
"""
PRSM Python SDK - Tool Execution Example
Demonstrates AI agent tool usage and execution capabilities
"""

import os
import json
import asyncio
from typing import List, Dict, Any

# Import PRSM SDK (assuming it's in parent directory)
import sys
sys.path.append('..')
from prsm_sdk import Client, Tool

async def tool_execution_example():
    """Demonstrate tool execution with PRSM Python SDK"""
    print("🚀 PRSM Python SDK - Tool Execution")
    
    # Initialize client
    client = Client(
        api_key=os.getenv('PRSM_API_KEY', 'demo-key'),
        endpoint=os.getenv('PRSM_ENDPOINT', 'http://localhost:8000')
    )
    
    # Define available tools
    tools = [
        Tool(
            name="calculate",
            description="Perform mathematical calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        ),
        Tool(
            name="search_knowledge",
            description="Search PRSM knowledge base",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for knowledge base"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_token_price",
            description="Get current FTNS token price",
            parameters={
                "type": "object",
                "properties": {},
            }
        )
    ]
    
    try:
        # Query with tool usage
        response = await client.query_with_tools(
            prompt="What is 25% of 10,000 FTNS tokens? Also, what's the current FTNS price and find information about staking rewards?",
            tools=tools,
            max_tokens=250
        )
        
        print(f"📝 AI Response: {response.text}")
        print(f"💰 Cost: {response.cost} FTNS")
        
        # Handle tool calls
        if response.tool_calls:
            print(f"🔧 Tool calls made: {len(response.tool_calls)}")
            
            for tool_call in response.tool_calls:
                print(f"  - Tool: {tool_call.name}")
                print(f"  - Args: {json.dumps(tool_call.arguments, indent=2)}")
                
                # Execute tool (mock implementation)
                result = await execute_tool(tool_call.name, tool_call.arguments)
                print(f"  - Result: {result}")
                
        # Get token usage statistics
        stats = await client.get_usage_stats()
        print(f"📊 Session stats: {stats.queries_made} queries, {stats.total_cost} FTNS spent")
        
    except Exception as error:
        print(f"❌ Tool execution error: {error}")

async def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Mock tool execution - in production, this would call actual tools"""
    
    if tool_name == "calculate":
        try:
            # Safe evaluation (in production, use a proper math parser)
            expression = arguments.get("expression", "")
            result = eval(expression)  # WARNING: eval is dangerous in production
            return f"Calculation result: {result}"
        except Exception as e:
            return f"Calculation error: {e}"
    
    elif tool_name == "search_knowledge":
        query = arguments.get("query", "")
        # Mock knowledge search
        return f"Found documentation about '{query}' in PRSM knowledge base: Staking rewards are distributed daily based on token holdings and network participation."
    
    elif tool_name == "get_token_price":
        # Mock price data
        return "Current FTNS price: $0.15 USD (24h change: +3.2%)"
    
    else:
        return f"Unknown tool: {tool_name}"

def run_sync_example():
    """Synchronous wrapper for the async example"""
    try:
        asyncio.run(tool_execution_example())
    except KeyboardInterrupt:
        print("\n👋 Example interrupted by user")
    except Exception as e:
        print(f"❌ Example failed: {e}")

if __name__ == "__main__":
    run_sync_example()