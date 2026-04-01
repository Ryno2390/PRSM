# AI Integration Guide

PRSM exposes its capabilities to AI assistants through the **Model Context Protocol (MCP)**. This means tools like Claude Desktop, Hermes, OpenClaw, and any MCP-compatible agent can discover and invoke PRSM operations — searching datasets, submitting compute jobs, monitoring the network — without custom integration code.

---

## Overview

When you enable the MCP server (`mcp_server_enabled: true` in your config, which is the default), PRSM runs a **FastMCP** server alongside your node. This server advertises all installed skill tools as MCP-compatible endpoints that AI agents can call.

The flow looks like this:

```
AI Assistant  →  MCP Protocol  →  PRSM MCP Server  →  PRSM Network
   (Claude)       (HTTP)            (localhost:9100)     (P2P)
```

---

## Setup

### 1. Start the MCP server

The MCP server starts automatically with your node if `mcp_server_enabled` is `true`. You can also start it standalone:

```bash
prsm mcp start
```

By default it listens on `localhost:9100`.

### 2. Get the config snippet

Generate the configuration your AI client needs:

```bash
prsm mcp config-snippet
```

This prints:

```yaml
mcp_servers:
  prsm:
    transport: streamable-http
    url: http://localhost:9100/mcp
```

### 3. Add to your AI client

**Claude Desktop** — Add the snippet to your Claude Desktop MCP configuration file (usually `~/.claude/mcp_servers.yaml` or via Settings → MCP Servers).

**Hermes** — Add the snippet to your Hermes config under the `mcp_servers` section.

**OpenClaw** — Add the snippet to your OpenClaw agent configuration.

**Any MCP client** — Use the streamable-http transport pointed at `http://localhost:9100/mcp`.

Once configured, the AI assistant will discover PRSM's tools automatically and can use them in conversation.

### 4. Install FastMCP (if needed)

The MCP server requires the `fastmcp` package:

```bash
pip install fastmcp
```

This is an optional dependency — the rest of PRSM works without it.

---

## Available Skill Packages

PRSM ships with three built-in skill packages. Each package provides a set of tools that the MCP server exposes to AI agents.

### prsm-datasets

Dataset curation and management on the PRSM network.

| Tool | Description |
|---|---|
| `prsm_search_datasets` | Search for datasets by keywords, domain, quality score, or format |
| `prsm_curate_dataset` | Combine and filter dataset sources into a new curated dataset |
| `prsm_validate_dataset` | Run quality checks (schema, duplicates, bias, quality scoring) |
| `prsm_publish_dataset` | Publish a dataset to the network with metadata and pricing |

### prsm-compute

Compute job management on the PRSM network.

| Tool | Description |
|---|---|
| `prsm_submit_job` | Submit a compute job (training, inference, evaluation, preprocessing) |
| `prsm_job_status` | Check the status and progress of a submitted job |
| `prsm_cancel_job` | Cancel a running or queued job |
| `prsm_list_queue` | List jobs in the queue with status and resource usage |

### prsm-network

Network operations and monitoring.

| Tool | Description |
|---|---|
| `prsm_list_peers` | List connected peers with their capabilities and status |
| `prsm_network_stats` | Get aggregate network statistics (compute, storage, bandwidth, tokens) |
| `prsm_health_check` | Run health checks on your node and its connectivity |
| `prsm_discover_nodes` | Discover new nodes by capability, region, or reputation |

---

## Example Workflows

These examples show how an AI assistant might use PRSM tools in conversation.

### Search for code generation datasets

**You:** "Search PRSM for code generation datasets with quality above 0.8"

The assistant calls `prsm_search_datasets`:

```json
{
  "query": "code generation",
  "domain": "code",
  "min_quality": 0.8,
  "format": "jsonl",
  "limit": 10
}
```

**Result:** A list of matching datasets with IDs, names, quality scores, sizes, and descriptions. The assistant can then help you curate, validate, or download them.

### Submit a compute job for model fine-tuning

**You:** "Submit a fine-tuning job for CodeLlama on the dataset we found"

The assistant calls `prsm_submit_job`:

```json
{
  "name": "codellama-finetune-run1",
  "task_type": "training",
  "model": "codellama/CodeLlama-7b-hf",
  "dataset": "dataset-abc123",
  "config": {
    "epochs": 3,
    "batch_size": 8,
    "learning_rate": 2e-5,
    "gpu_required": true
  },
  "priority": "normal",
  "max_ftns": 50.0
}
```

**Result:** A job ID and submission confirmation. The assistant can then check progress with `prsm_job_status` and report back when the job completes.

### Check network health and peer status

**You:** "How's the PRSM network doing? Any issues with my node?"

The assistant calls `prsm_health_check` and `prsm_network_stats`:

```json
// Health check
{
  "checks": ["connectivity", "storage", "compute", "latency"],
  "verbose": true
}

// Network stats
{
  "metric": "overview",
  "timeframe": "24h"
}
```

**Result:** A summary of your node's health (all checks passing, latency to peers, resource usage) plus network-wide stats (total peers, aggregate compute capacity, jobs processed in the last 24 hours).

---

## Creating Custom Skill Packages

You can extend PRSM with custom skill packages. Each package is a directory containing a `SKILL.yaml` manifest that declares the package's tools.

### SKILL.yaml format

```yaml
name: my-custom-skill
version: 1.0.0
description: What this skill package does
author: Your Name
capabilities:
  - capability_one
  - capability_two
requires:
  - prsm-network >= 0.2.0
tools:
  - name: my_tool_name
    description: What this tool does — shown to AI agents
    parameters:
      param_one:
        type: string
        description: "What this parameter is for"
      param_two:
        type: integer
        description: "Optional numeric parameter"
        optional: true
      param_three:
        type: object
        description: "Complex configuration object"
        optional: true
```

### Field reference

| Field | Required | Description |
|---|---|---|
| `name` | yes | Unique package identifier (convention: `prsm-*` for official, anything else for community) |
| `version` | yes | Semantic version |
| `description` | yes | One-line description of the package |
| `author` | yes | Package author |
| `capabilities` | yes | List of capability identifiers this package provides |
| `requires` | no | Dependency list (package name + version constraint) |
| `tools` | yes | List of tool definitions (see below) |

### Tool definition

Each tool has:

| Field | Required | Description |
|---|---|---|
| `name` | yes | Tool identifier (snake_case, must be unique across all skills) |
| `description` | yes | What the tool does — this is shown to AI agents, so make it clear |
| `parameters` | yes | Map of parameter name → schema (type, description, optional flag) |

### Parameter types

Supported types: `string`, `integer`, `number`, `boolean`, `array`, `object`.

### Installing custom skills

Place your skill directory (containing `SKILL.yaml`) in the skills search path, then verify:

```bash
prsm skills list
```

Your custom skill should appear alongside the built-in packages.

---

## Skills CLI Commands

Manage and inspect skill packages from the command line.

### List installed skills

```bash
prsm skills list
```

```
Installed Skills:
  prsm-datasets  v1.0.0  Dataset curation and management (4 tools)
  prsm-compute   v1.0.0  Compute job management (4 tools)
  prsm-network   v1.0.0  Network operations and monitoring (4 tools)
```

### Get skill details

```bash
prsm skills info prsm-datasets
```

Shows the full skill manifest: description, version, capabilities, and all tools with their parameters.

### Search for skills

```bash
prsm skills search "compute"
```

Searches skill names and descriptions.

### Export a skill manifest

```bash
prsm skills export prsm-datasets > my-datasets-skill.yaml
```

Exports the SKILL.yaml for inspection or modification.

---

## Troubleshooting

**"fastmcp is required to run the PRSM MCP server"** — Install it with `pip install fastmcp`.

**AI client can't connect** — Verify the MCP server is running (`prsm status`) and the port matches your client config. Check that `mcp_server_enabled` is `true`.

**Tools not appearing in AI client** — Restart the MCP server after installing new skills. Some clients cache tool lists and may need to be restarted too.

**Port conflict** — Change the MCP port:

```bash
prsm config set mcp_server_port 9200
prsm mcp start
```
