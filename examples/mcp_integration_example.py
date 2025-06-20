#!/usr/bin/env python3
"""
PRSM MCP Integration Example
===========================

Comprehensive example demonstrating how to use PRSM's MCP (Model Context Protocol) 
client to connect to external tool servers and execute tools.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prsm.integrations.mcp import (
    MCPClient, MCPError, ToolRegistry, SessionManager,
    MCPIntegrationConfig, SecurityLevel
)
from prsm.integrations.mcp.models import ToolDefinition, ToolParameter, ToolParameterType


class MCPIntegrationDemo:
    """Comprehensive MCP integration demonstration"""
    
    def __init__(self):
        self.clients: Dict[str, MCPClient] = {}
        self.tool_registry = ToolRegistry()
        self.session_manager = SessionManager()
        
    async def run_demo(self):
        """Run the complete MCP integration demo"""
        print("🔌 PRSM MCP Integration Demo")
        print("=" * 50)
        
        try:
            # Step 1: Setup mock MCP server connection
            await self.setup_mock_servers()
            
            # Step 2: Demonstrate tool discovery
            await self.demonstrate_tool_discovery()
            
            # Step 3: Demonstrate tool execution
            await self.demonstrate_tool_execution()
            
            # Step 4: Demonstrate session management
            await self.demonstrate_session_management()
            
            # Step 5: Demonstrate security features
            await self.demonstrate_security_features()
            
            # Step 6: Show statistics and monitoring
            await self.show_statistics()
            
            print("\n🎉 MCP Integration Demo Complete!")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
        finally:
            await self.cleanup()
    
    async def setup_mock_servers(self):
        """Setup connections to mock MCP servers"""
        print("\n📡 Setting up MCP server connections...")
        
        # In a real implementation, these would be actual MCP servers
        mock_servers = [
            {
                "name": "development-tools",
                "uri": "http://localhost:3001",
                "description": "Development and coding tools"
            },
            {
                "name": "data-analysis",
                "uri": "http://localhost:3002", 
                "description": "Data analysis and visualization tools"
            },
            {
                "name": "web-services",
                "uri": "ws://localhost:3003",
                "description": "Web API and service integration tools"
            }
        ]
        
        for server_info in mock_servers:
            try:
                # Create client with configuration
                config = MCPIntegrationConfig(
                    max_concurrent_calls=5,
                    default_timeout=30,
                    security_level=SecurityLevel.MEDIUM
                )
                
                client = MCPClient(server_info["uri"], config)
                
                # Since these are mock servers, we'll simulate the connection
                await self.simulate_connection(client, server_info)
                
                self.clients[server_info["name"]] = client
                print(f"✅ Connected to {server_info['name']}: {server_info['uri']}")
                
            except Exception as e:
                print(f"⚠️ Failed to connect to {server_info['name']}: {e}")
        
        print(f"📊 Successfully connected to {len(self.clients)} MCP servers")
    
    async def simulate_connection(self, client: MCPClient, server_info: Dict[str, Any]):
        """Simulate MCP server connection and tool discovery"""
        # Mock server connection
        client.connected = True
        
        # Create mock tools for demonstration
        mock_tools = self.create_mock_tools(server_info["name"])
        
        # Register tools in client's registry
        for tool_def in mock_tools:
            client.tool_registry.register_tool(tool_def)
            self.tool_registry.register_tool(tool_def)
    
    def create_mock_tools(self, server_name: str) -> list[ToolDefinition]:
        """Create mock tools for each server type"""
        tools = []
        
        if server_name == "development-tools":
            tools = [
                ToolDefinition(
                    name="code_formatter",
                    description="Format code in various languages",
                    parameters=[
                        ToolParameter(
                            name="code",
                            type=ToolParameterType.STRING,
                            description="Code to format",
                            required=True
                        ),
                        ToolParameter(
                            name="language",
                            type=ToolParameterType.STRING,
                            description="Programming language",
                            required=True,
                            enum=["python", "javascript", "go", "rust"]
                        )
                    ],
                    security_level=SecurityLevel.LOW,
                    category="development",
                    tags=["formatting", "code", "development"]
                ),
                ToolDefinition(
                    name="run_tests",
                    description="Execute unit tests",
                    parameters=[
                        ToolParameter(
                            name="test_path",
                            type=ToolParameterType.STRING,
                            description="Path to test files",
                            required=True
                        ),
                        ToolParameter(
                            name="framework",
                            type=ToolParameterType.STRING,
                            description="Testing framework",
                            required=False,
                            default="pytest"
                        )
                    ],
                    security_level=SecurityLevel.HIGH,
                    category="development",
                    tags=["testing", "development", "validation"]
                )
            ]
        
        elif server_name == "data-analysis":
            tools = [
                ToolDefinition(
                    name="analyze_dataset",
                    description="Perform statistical analysis on dataset",
                    parameters=[
                        ToolParameter(
                            name="data_url",
                            type=ToolParameterType.STRING,
                            description="URL or path to dataset",
                            required=True
                        ),
                        ToolParameter(
                            name="analysis_type",
                            type=ToolParameterType.STRING,
                            description="Type of analysis to perform",
                            required=True,
                            enum=["descriptive", "correlation", "regression"]
                        )
                    ],
                    security_level=SecurityLevel.MEDIUM,
                    category="data-science",
                    tags=["analysis", "statistics", "data"]
                ),
                ToolDefinition(
                    name="create_visualization",
                    description="Create data visualizations",
                    parameters=[
                        ToolParameter(
                            name="data",
                            type=ToolParameterType.OBJECT,
                            description="Data to visualize",
                            required=True
                        ),
                        ToolParameter(
                            name="chart_type",
                            type=ToolParameterType.STRING,
                            description="Type of chart",
                            required=True,
                            enum=["bar", "line", "scatter", "histogram"]
                        )
                    ],
                    security_level=SecurityLevel.LOW,
                    category="visualization",
                    tags=["charts", "visualization", "data"]
                )
            ]
        
        elif server_name == "web-services":
            tools = [
                ToolDefinition(
                    name="http_request",
                    description="Make HTTP requests to external APIs",
                    parameters=[
                        ToolParameter(
                            name="url",
                            type=ToolParameterType.STRING,
                            description="URL to request",
                            required=True
                        ),
                        ToolParameter(
                            name="method",
                            type=ToolParameterType.STRING,
                            description="HTTP method",
                            required=False,
                            default="GET",
                            enum=["GET", "POST", "PUT", "DELETE"]
                        ),
                        ToolParameter(
                            name="headers",
                            type=ToolParameterType.OBJECT,
                            description="HTTP headers",
                            required=False
                        )
                    ],
                    security_level=SecurityLevel.MEDIUM,
                    category="web",
                    tags=["http", "api", "web"]
                )
            ]
        
        return tools
    
    async def demonstrate_tool_discovery(self):
        """Demonstrate tool discovery across MCP servers"""
        print("\n🔍 Demonstrating Tool Discovery...")
        
        # List all available tools
        all_tools = self.tool_registry.list_tools()
        print(f"📋 Total tools discovered: {len(all_tools)}")
        
        # Show tools by category
        categories = self.tool_registry.get_categories()
        for category in categories:
            tools_in_category = self.tool_registry.list_tools(category=category)
            print(f"  📂 {category}: {len(tools_in_category)} tools")
            for tool in tools_in_category:
                print(f"    • {tool.name}: {tool.description}")
        
        # Demonstrate search functionality
        print("\n🔎 Tool search examples:")
        search_queries = ["format", "data", "test"]
        
        for query in search_queries:
            results = self.tool_registry.search_tools(query)
            print(f"  Search '{query}': {len(results)} results")
            for tool in results[:2]:  # Show first 2 results
                print(f"    • {tool.name}")
        
        # Show security level distribution
        print("\n🔒 Security level distribution:")
        for level in SecurityLevel:
            tools_at_level = self.tool_registry.list_tools(security_level=level)
            print(f"  {level.value}: {len(tools_at_level)} tools")
    
    async def demonstrate_tool_execution(self):
        """Demonstrate tool execution with validation"""
        print("\n⚙️ Demonstrating Tool Execution...")
        
        # Example 1: Simple tool call
        print("\n1. Simple tool execution:")
        try:
            # Find a low-security tool to execute
            low_security_tools = self.tool_registry.get_tools_by_security_level(SecurityLevel.LOW)
            if low_security_tools:
                tool = low_security_tools[0]
                print(f"   Executing tool: {tool.name}")
                
                # Create sample parameters
                params = self.create_sample_parameters(tool)
                
                # Validate parameters first
                validation = self.tool_registry.validate_tool_call(tool.name, params)
                if validation["valid"]:
                    # Simulate tool execution
                    result = await self.simulate_tool_execution(tool.name, params)
                    print(f"   ✅ Tool executed successfully: {result['success']}")
                    print(f"   ⏱️  Execution time: {result['execution_time']:.2f}s")
                else:
                    print(f"   ❌ Parameter validation failed: {validation['errors']}")
        
        except Exception as e:
            print(f"   ❌ Tool execution failed: {e}")
        
        # Example 2: Parameter validation demonstration
        print("\n2. Parameter validation examples:")
        
        # Valid parameters
        tool = self.tool_registry.get_tool("code_formatter")
        if tool:
            valid_params = {"code": "def hello(): print('world')", "language": "python"}
            validation = tool.validate_parameters(valid_params)
            print(f"   Valid params: {validation['valid']} (errors: {len(validation['errors'])})")
            
            # Invalid parameters
            invalid_params = {"code": "some code", "language": "invalid_lang"}
            validation = tool.validate_parameters(invalid_params)
            print(f"   Invalid params: {validation['valid']} (errors: {len(validation['errors'])})")
            if validation['errors']:
                print(f"     Errors: {validation['errors'][0]}")
    
    async def demonstrate_session_management(self):
        """Demonstrate MCP session management"""
        print("\n📋 Demonstrating Session Management...")
        
        # Create a session
        session = await self.session_manager.create_session(
            server_uri="http://localhost:3001",
            user_id="demo_user",
            initial_context={"project": "mcp_demo", "environment": "development"}
        )
        
        print(f"✅ Created session: {session.session_id}")
        print(f"   User: {session.user_id}")
        print(f"   Context keys: {list(session.context.keys())}")
        
        # Update session context
        await self.session_manager.update_session_context(
            session.session_id,
            {"last_tool": "code_formatter", "step": 1}
        )
        
        print("📝 Updated session context")
        
        # Simulate adding tool calls to session
        from prsm.integrations.mcp.models import ToolCall
        mock_tool_call = ToolCall(
            tool_name="code_formatter",
            parameters={"code": "print('hello')", "language": "python"},
            user_id="demo_user"
        )
        
        await self.session_manager.add_tool_call_to_session(session.session_id, mock_tool_call)
        print(f"🔧 Added tool call to session")
        
        # Get session statistics
        stats = self.session_manager.get_session_statistics()
        print(f"📊 Session stats: {stats['total_sessions']} total, {stats['active_sessions']} active")
    
    async def demonstrate_security_features(self):
        """Demonstrate MCP security features"""
        print("\n🔒 Demonstrating Security Features...")
        
        # Set up security manager from any client
        if self.clients:
            client = next(iter(self.clients.values()))
            security_manager = client.security_manager
            
            # Set user security levels
            security_manager.set_user_security_level("demo_user", SecurityLevel.MEDIUM)
            security_manager.set_user_security_level("admin_user", SecurityLevel.HIGH)
            
            print("👤 Set user security levels:")
            print(f"   demo_user: {SecurityLevel.MEDIUM.value}")
            print(f"   admin_user: {SecurityLevel.HIGH.value}")
            
            # Test security validation
            high_security_tool = self.tool_registry.get_tool("run_tests")
            if high_security_tool:
                from prsm.integrations.mcp.models import ToolCall
                
                # Test with medium security user (should fail)
                tool_call = ToolCall(
                    tool_name="run_tests",
                    parameters={"test_path": "/safe/tests"},
                    user_id="demo_user"
                )
                
                validation = await security_manager.validate_tool_call(tool_call, high_security_tool.definition)
                print(f"\n🧪 Medium user accessing high security tool:")
                print(f"   Approved: {validation.approved}")
                if not validation.approved:
                    print(f"   Reason: {validation.reason}")
                
                # Test with high security user (should pass)
                tool_call.user_id = "admin_user"
                validation = await security_manager.validate_tool_call(tool_call, high_security_tool.definition)
                print(f"\n🧪 High user accessing high security tool:")
                print(f"   Approved: {validation.approved}")
                print(f"   Risk level: {validation.risk_level.value}")
            
            # Show security statistics
            sec_stats = security_manager.get_security_statistics()
            print(f"\n📊 Security statistics:")
            print(f"   Users configured: {sec_stats['total_users_configured']}")
            print(f"   Security patterns: {sec_stats['blocked_patterns_count']}")
    
    async def show_statistics(self):
        """Show comprehensive statistics"""
        print("\n📊 System Statistics...")
        
        # Tool registry statistics
        tool_stats = self.tool_registry.get_statistics()
        print(f"\n🔧 Tool Registry:")
        print(f"   Total tools: {tool_stats['total_tools']}")
        print(f"   Categories: {tool_stats['total_categories']}")
        print(f"   Tags: {tool_stats['total_tags']}")
        print(f"   Total calls: {tool_stats['total_calls']}")
        print(f"   Success rate: {tool_stats['overall_success_rate']:.1f}%")
        
        # Session manager statistics
        session_stats = self.session_manager.get_session_statistics()
        print(f"\n📋 Session Manager:")
        print(f"   Total sessions: {session_stats['total_sessions']}")
        print(f"   Active sessions: {session_stats['active_sessions']}")
        print(f"   Users with sessions: {session_stats['users_with_sessions']}")
        
        # Client statistics
        print(f"\n📡 MCP Clients:")
        for name, client in self.clients.items():
            stats = client.get_statistics()
            print(f"   {name}:")
            print(f"     Connected: {stats['connected']}")
            print(f"     Total requests: {stats['total_requests']}")
            print(f"     Success rate: {stats['success_rate']:.1f}%")
            print(f"     Available tools: {stats['available_tools']}")
    
    def create_sample_parameters(self, tool) -> Dict[str, Any]:
        """Create sample parameters for a tool"""
        params = {}
        
        for param in tool.definition.parameters:
            if param.required:
                if param.type == ToolParameterType.STRING:
                    if param.enum:
                        params[param.name] = param.enum[0]
                    else:
                        params[param.name] = f"sample_{param.name}"
                elif param.type == ToolParameterType.INTEGER:
                    params[param.name] = 42
                elif param.type == ToolParameterType.BOOLEAN:
                    params[param.name] = True
                elif param.type == ToolParameterType.OBJECT:
                    params[param.name] = {"sample": "data"}
                elif param.type == ToolParameterType.ARRAY:
                    params[param.name] = ["item1", "item2"]
        
        return params
    
    async def simulate_tool_execution(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate tool execution for demo purposes"""
        import time
        
        # Simulate execution time
        execution_time = 0.1 + (len(str(parameters)) * 0.001)
        await asyncio.sleep(execution_time)
        
        # Simulate result
        return {
            "success": True,
            "result": f"Simulated result for {tool_name}",
            "execution_time": execution_time,
            "tool_name": tool_name,
            "parameters_count": len(parameters)
        }
    
    async def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        
        # Close all client connections
        for name, client in self.clients.items():
            try:
                await client.disconnect()
                print(f"   Disconnected from {name}")
            except Exception as e:
                print(f"   Failed to disconnect from {name}: {e}")
        
        # Close all sessions
        closed_sessions = await self.session_manager.cleanup_expired_sessions()
        if closed_sessions > 0:
            print(f"   Closed {closed_sessions} expired sessions")


async def main():
    """Main entry point"""
    demo = MCPIntegrationDemo()
    await demo.run_demo()


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())