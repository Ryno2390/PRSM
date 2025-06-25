# PRSM MCP Integration Guide

Complete guide to using PRSM's Model Context Protocol (MCP) client for tool integration.

## Overview

The Model Context Protocol (MCP) is a standardized protocol for connecting AI applications with external tools and data sources. PRSM's MCP integration provides:

- **Standardized Tool Integration**: Connect to any MCP-compliant tool server
- **Security & Sandboxing**: Comprehensive security validation for tool execution
- **Session Management**: Context preservation across tool interactions
- **Resource Management**: Access to external data sources via MCP
- **Rich Monitoring**: Detailed statistics and performance tracking

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PRSM Core     │    │   MCP Client     │    │  External Tools │
│                 │◄──►│                  │◄──►│                 │
│ • NWTN          │    │ • Tool Registry  │    │ • Code Tools    │
│ • Teachers      │    │ • Security Mgr   │    │ • Data Tools    │
│ • Agents        │    │ • Session Mgr    │    │ • Web APIs      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Basic MCP Client Setup

```python
from prsm.integrations.mcp import MCPClient, MCPIntegrationConfig, SecurityLevel

# Create client configuration
config = MCPIntegrationConfig(
    max_concurrent_calls=10,
    default_timeout=30,
    security_level=SecurityLevel.MEDIUM,
    sandbox_enabled=True
)

# Connect to MCP server
client = MCPClient("http://localhost:3000", config)
await client.connect()

# Discover available tools
tools = await client.list_tools()
print(f"Found {len(tools)} tools")
```

### 2. Execute Tools

```python
# Execute a tool with parameters
result = await client.call_tool(
    tool_name="code_formatter",
    parameters={
        "code": "def hello():print('world')",
        "language": "python"
    },
    user_id="your_user_id"
)

if result.success:
    print(f"Result: {result.result}")
    print(f"Execution time: {result.execution_time:.2f}s")
else:
    print(f"Error: {result.error}")
```

### 3. Session Management

```python
from prsm.integrations.mcp import SessionManager

session_manager = SessionManager()

# Create session with context
session = await session_manager.create_session(
    server_uri="http://localhost:3000",
    user_id="your_user_id",
    initial_context={"project": "my_project"}
)

# Use session for tool calls
result = await client.call_tool(
    tool_name="analyze_code",
    parameters={"file_path": "main.py"},
    user_id="your_user_id",
    session_id=session.session_id
)
```

## Core Components

### MCPClient

The main client for connecting to MCP servers.

**Features:**
- HTTP and WebSocket connection support
- Automatic tool discovery and registration
- Request/response handling with timeouts
- Connection health monitoring

**Key Methods:**
```python
# Connection management
await client.connect()
await client.disconnect()
await client.ping()

# Tool operations
tools = await client.list_tools()
result = await client.call_tool(name, params, user_id)

# Resource operations
resources = await client.list_resources()
```

### ToolRegistry

Manages discovered tools with validation and statistics.

**Features:**
- Tool registration and indexing
- Parameter validation with type checking
- Usage statistics and performance tracking
- Search and filtering capabilities

**Example Usage:**
```python
from prsm.integrations.mcp import ToolRegistry

registry = ToolRegistry()

# Search tools
results = registry.search_tools("format")

# Filter by security level
safe_tools = registry.get_tools_by_security_level(SecurityLevel.LOW)

# Validate tool call
validation = registry.validate_tool_call("tool_name", parameters)
if validation["valid"]:
    # Proceed with execution
    pass
```

### Security Manager

Provides comprehensive security validation.

**Security Features:**
- User-based security levels
- Parameter sanitization and validation
- Suspicious pattern detection
- Execution restrictions and sandboxing

**Example Usage:**
```python
from prsm.integrations.mcp import MCPSecurityManager

security = MCPSecurityManager()

# Set user security level
security.set_user_security_level("user_id", SecurityLevel.MEDIUM)

# Validate tool call
validation = await security.validate_tool_call(tool_call, tool_definition)
if validation.approved:
    print(f"Approved with risk level: {validation.risk_level}")
else:
    print(f"Blocked: {validation.reason}")
```

### Session Manager

Manages persistent sessions for context preservation.

**Features:**
- Session lifecycle management
- Context preservation across tool calls
- Automatic cleanup of expired sessions
- Multi-user session support

**Example Usage:**
```python
from prsm.integrations.mcp import SessionManager

session_mgr = SessionManager()

# Create session
session = await session_mgr.create_session(
    server_uri="http://localhost:3000",
    user_id="user_123",
    initial_context={"workspace": "/path/to/project"}
)

# Update context
await session_mgr.update_session_context(
    session.session_id,
    {"last_file": "main.py", "line_number": 42}
)

# Get context later
context = await session_mgr.get_session_context(session.session_id)
```

## Security Model

### Security Levels

| Level | Description | Capabilities |
|-------|-------------|--------------|
| **LOW** | Read-only operations | File reading, data queries |
| **MEDIUM** | Standard operations | File read/write, network access |
| **HIGH** | System operations | Process management, system calls |
| **CRITICAL** | Administrative | Full system access, destructive operations |

### Security Validation

The security manager performs multiple validation layers:

1. **User Permission Check**: Ensures user can access the tool's security level
2. **Parameter Validation**: Checks for malicious patterns and data
3. **Resource Restrictions**: Enforces file, network, and system access limits
4. **Pattern Detection**: Identifies suspicious code injection attempts

### Example Security Policies

```python
# Set restrictive user permissions
security.set_user_security_level("intern_user", SecurityLevel.LOW)
security.set_user_restrictions("intern_user", {
    "max_parameter_size": 1024,
    "network_access": False,
    "allowed_file_operations": {"read"}
})

# Set admin permissions
security.set_user_security_level("admin_user", SecurityLevel.HIGH)
```

## Advanced Usage

### Custom Tool Integration

```python
from prsm.integrations.mcp.models import ToolDefinition, ToolParameter, ToolParameterType

# Define custom tool
custom_tool = ToolDefinition(
    name="custom_analyzer",
    description="Custom data analysis tool",
    parameters=[
        ToolParameter(
            name="data_source",
            type=ToolParameterType.STRING,
            description="Path to data file",
            required=True
        ),
        ToolParameter(
            name="analysis_type",
            type=ToolParameterType.STRING,
            description="Type of analysis",
            enum=["summary", "correlation", "regression"],
            default="summary"
        )
    ],
    security_level=SecurityLevel.MEDIUM,
    category="data-analysis"
)

# Register with tool registry
registry.register_tool(custom_tool)
```

### Resource Management

```python
from prsm.integrations.mcp import ResourceManager

resource_mgr = ResourceManager()

# List available resources
resources = await resource_mgr.list_resources(
    user_id="user_123",
    category="datasets"
)

# Get resource content
resource = await resource_mgr.get_resource(
    uri="dataset://sales_data.csv",
    user_id="user_123"
)

if resource:
    print(f"Content: {resource.content}")
```

### Monitoring and Statistics

```python
# Get client statistics
stats = client.get_statistics()
print(f"Success rate: {stats['success_rate']:.1f}%")
print(f"Average response time: {stats['average_response_time']:.2f}s")

# Get tool usage statistics
tool_stats = registry.get_statistics()
print(f"Most used tool: {tool_stats['most_used_tool']}")
print(f"Overall success rate: {tool_stats['overall_success_rate']:.1f}%")

# Get session statistics
session_stats = session_mgr.get_session_statistics()
print(f"Active sessions: {session_stats['active_sessions']}")
```

## Integration with PRSM Core

### Using MCP Tools in NWTN Orchestrator

```python
from prsm.nwtn.orchestrator import NWTNOrchestrator
from prsm.integrations.mcp import MCPClient

# Initialize with MCP integration
orchestrator = NWTNOrchestrator()

# Register MCP client as tool provider
mcp_client = MCPClient("http://localhost:3000")
await mcp_client.connect()

# Use in NWTN workflow
async def enhanced_query_processing(user_input):
    # NWTN can now access MCP tools
    if "format code" in user_input.prompt:
        # Use MCP tool for code formatting
        result = await mcp_client.call_tool(
            tool_name="code_formatter",
            parameters={"code": extracted_code, "language": "python"},
            user_id=user_input.user_id
        )
        
        if result.success:
            # Integrate result into NWTN response
            return f"Formatted code: {result.result}"
    
    # Continue with normal NWTN processing
    return await orchestrator.process_query(user_input)
```

### Teacher Model Integration

```python
from prsm.teachers.teacher_model import TeacherModel

class MCPEnhancedTeacher(TeacherModel):
    def __init__(self, mcp_client: MCPClient):
        super().__init__()
        self.mcp_client = mcp_client
    
    async def process_specialized_request(self, request):
        # Use MCP tools for specialized processing
        if request.type == "code_analysis":
            result = await self.mcp_client.call_tool(
                tool_name="analyze_code",
                parameters={"code": request.code},
                user_id=request.user_id
            )
            
            return self.enhance_with_teacher_knowledge(result)
```

## Configuration

### Environment Variables

```bash
# MCP Configuration
MCP_ENABLED=true
MCP_DEFAULT_TIMEOUT=30
MCP_MAX_CONCURRENT_CALLS=10
MCP_SECURITY_LEVEL=medium
MCP_SANDBOX_ENABLED=true

# Server Connections
MCP_SERVERS='[
  {"name": "dev-tools", "uri": "http://localhost:3001"},
  {"name": "data-tools", "uri": "ws://localhost:3002"}
]'
```

### Programmatic Configuration

```python
from prsm.integrations.mcp.models import MCPIntegrationConfig, MCPServerConnection

config = MCPIntegrationConfig(
    enabled=True,
    max_concurrent_calls=20,
    default_timeout=45,
    security_level=SecurityLevel.HIGH,
    sandbox_enabled=True,
    servers=[
        MCPServerConnection(
            name="development-tools",
            uri="http://localhost:3001",
            auth_token="your_auth_token",
            enabled=True
        ),
        MCPServerConnection(
            name="data-analysis",
            uri="ws://localhost:3002",
            timeout=60,
            enabled=True
        )
    ]
)
```

## Best Practices

### 1. Security First

- Always set appropriate user security levels
- Validate tool parameters before execution
- Use sandboxing for untrusted tools
- Monitor for suspicious patterns

### 2. Performance Optimization

- Use connection pooling for multiple servers
- Implement caching for frequently used tools
- Set appropriate timeouts for different tool types
- Monitor and optimize based on statistics

### 3. Error Handling

```python
try:
    result = await client.call_tool(name, params, user_id)
    if result.success:
        return result.result
    else:
        logger.error(f"Tool execution failed: {result.error}")
        return fallback_result
except MCPConnectionError:
    logger.error("MCP server connection lost")
    return cached_result
except MCPToolError as e:
    logger.error(f"Tool error: {e}")
    return error_result
```

### 4. Session Management

- Create sessions for related tool calls
- Clean up expired sessions regularly
- Use context to maintain state across calls
- Implement session-based caching

## Troubleshooting

### Common Issues

#### Connection Problems
```python
# Check server status
is_healthy = await client.ping()
if not is_healthy:
    logger.error("MCP server not responding")
    
# Check connection statistics
stats = client.get_statistics()
if stats['success_rate'] < 90:
    logger.warning("High error rate detected")
```

#### Tool Execution Failures
```python
# Validate parameters first
validation = registry.validate_tool_call(tool_name, parameters)
if not validation["valid"]:
    logger.error(f"Invalid parameters: {validation['errors']}")

# Check security permissions
security_result = await security.validate_tool_call(tool_call, tool_def)
if not security_result.approved:
    logger.error(f"Security blocked: {security_result.reason}")
```

#### Performance Issues
```python
# Monitor execution times
if result.execution_time > 30:
    logger.warning(f"Slow tool execution: {result.execution_time}s")

# Check concurrent call limits
stats = client.get_statistics()
if stats['failed_requests'] > stats['successful_requests'] * 0.1:
    logger.warning("High failure rate - consider reducing concurrency")
```

## Examples

See `examples/mcp_integration_example.py` for a comprehensive demonstration of all MCP features.

## API Reference

For detailed API documentation, see the module docstrings in:
- `prsm.integrations.mcp.client`
- `prsm.integrations.mcp.tools`
- `prsm.integrations.mcp.session`
- `prsm.integrations.mcp.security`
- `prsm.integrations.mcp.resources`