# PRSM AI Integration

PRSM exposes an MCP (Model Context Protocol) server so AI assistants (Hermes,
OpenClaw, Claude, etc.) can interact with your node natively.

## How It Works

When your node is running (via `prsm daemon start` or `prsm node start`), an
MCP server is automatically started on `localhost:9100` (configurable). This
server exposes all installed PRSM skill packages as callable tools.

```
AI Assistant ← MCP (streamable-http) → PRSM MCP Server → PRSM Node
```

## Connecting an AI Assistant

Add the MCP server to your assistant's config:

```yaml
mcp_servers:
  prsm:
    transport: streamable-http
    url: http://localhost:9100/mcp
```

Once connected, the AI can call PRSM tools directly. For example:
- "Search the PRSM network for code training datasets"
- "Submit a fine-tuning compute job with 50 epochs"
- "Show me my connected peers"

## Configuring the MCP Server

```
prsm config get mcp_server_enabled     # check if enabled
prsm config get mcp_server_port        # check port
prsm config set mcp_server_enabled false   # disable
prsm config set mcp_server_port 9200       # change port
```

After changing ports, restart the daemon:
```
prsm daemon restart
```

## Skill Packages

Skills are the unit of capability on the PRSM network. Each package defines
tools, prompts, and workflows that an AI can use.

### Built-in Skills

PRSM ships with 3 built-in skill packages:

| Package | Tools | Description |
|---------|-------|-------------|
| `prsm-datasets` | search, curate, validate, publish | Dataset discovery and management |
| `prsm-compute` | submit, status, cancel, queue | Compute job lifecycle |
| `prsm-network` | peers, stats, health, discover | Network operations |

### Skill Package Structure

```
prsm-skill-name/
├── SKILL.yaml       # manifest (name, version, capabilities, tool defs)
├── README.md        # human docs
├── tools.json       # MCP tool definitions
├── prompts/         # system prompts for AI agents
│   └── agent.md
├── workflows/       # multi-step workflow definitions
│   └── process.yaml
└── examples/        # usage examples
    └── example.md
```

### Skill Manifest Format

```yaml
name: prsm-example
version: 1.0.0
description: Example skill description
author: Your Name
capabilities:
  - capability_one
  - capability_two
requires:
  - prsm-network >= 0.2.0
tools:
  - name: prsm_example_tool
    description: "Does something useful on the PRSM network"
    parameters:
      query: {type: string, description: "Search query"}
      limit: {type: number, description: "Max results", optional: true}
```

### Managing Skills

```
prsm skills list              # list installed skills
prsm skills search <query>    # search network for skills
prsm skills install <package> # install from network
prsm skills remove <package>  # uninstall
prsm skills info <package>    # show details
prsm skills export <package>  # export as MCP manifest
```

## Example MCP Interaction

Once connected, an AI assistant can:

1. **Search datasets:**
   AI calls `prsm_search_datasets(query="python code", min_quality=0.8)`
   PRSM queries the network → returns matching datasets

2. **Submit a compute job:**
   AI calls `prsm_submit_job(model="transformer", epochs=100)`
   PRSM queues the job → returns job ID and estimated cost

3. **Monitor progress:**
   AI calls `prsm_job_status(job_id="abc123")`
   PRSM returns current status and progress

4. **Check network health:**
   AI calls `prsm_network_health()`
   PRSM returns peer count, uptime, resource usage

## For Skill Authors

To create a custom skill package:

1. Create a directory with a `SKILL.yaml` following the format above
2. Add tool definitions in `tools.json`
3. Include system prompts in `prompts/`
4. Install it: `prsm skills install ./my-skill/`

The skill will be automatically discovered by the MCP server and exposed to
any connected AI assistant.
