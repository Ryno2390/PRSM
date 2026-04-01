# Configuration Reference

PRSM stores all node settings in a single YAML file. This document covers every config field, how to override settings with environment variables, and the full set of CLI config commands.

---

## Config File Location

```
~/.prsm/config.yaml
```

Created automatically by `prsm setup` or on first node start. API keys and secrets are stored separately in `~/.prsm/.env`.

---

## Config Fields

### Node Identity

| Field | Type | Default | Description |
|---|---|---|---|
| `display_name` | string | `"prsm-node"` | Human-readable name for your node |
| `node_role` | string | `"full"` | Node role: `"full"`, `"contributor"`, or `"consumer"` |

### Resource Allocation

| Field | Type | Default | Range | Description |
|---|---|---|---|---|
| `cpu_pct` | integer | `50` | 10–90 | Max CPU usage as percentage of available cores |
| `memory_pct` | integer | `50` | 10–90 | Max RAM usage as percentage of total memory |
| `gpu_pct` | integer | `80` | 0–100 | Max GPU memory usage (0 = no GPU) |
| `storage_gb` | float | `10.0` | ≥ 1.0 | Disk space allocated for PRSM data (GB) |
| `max_concurrent_jobs` | integer | `3` | 1–20 | Maximum simultaneous compute jobs |
| `upload_mbps_limit` | float | `0.0` | ≥ 0 | Upload bandwidth cap in Mbps (0 = unlimited) |

### Scheduling

| Field | Type | Default | Description |
|---|---|---|---|
| `active_hours_start` | integer or null | `null` | Hour to start accepting jobs (0–23). Null = always active. |
| `active_hours_end` | integer or null | `null` | Hour to stop accepting jobs (0–23). Null = always active. |
| `active_days` | list of strings | `[]` | Days the node is active (e.g., `["mon", "tue", "wed"]`). Empty = every day. |

**Example — run only on weekdays during business hours:**

```yaml
active_hours_start: 9
active_hours_end: 17
active_days: ["mon", "tue", "wed", "thu", "fri"]
```

### Network

| Field | Type | Default | Range | Description |
|---|---|---|---|---|
| `p2p_port` | integer | `9001` | 1024–65535 | Port for P2P peer connections |
| `api_port` | integer | `8000` | 1024–65535 | Port for the local management API |
| `bootstrap_nodes` | list of strings | `[]` | — | Bootstrap peer addresses for initial network connection |

**Default bootstrap server** is pre-configured at `wss://bootstrap1.prsm-network.com:8765`. Additional bootstrap nodes can be added:

```yaml
bootstrap_nodes:
  - "wss://bootstrap1.prsm-network.com:8765"
  - "ws://192.168.1.50:9001"
```

### API Keys

These booleans track whether keys are configured. Actual keys are stored in `~/.prsm/.env`.

| Field | Type | Default | Description |
|---|---|---|---|
| `has_openai_key` | boolean | `false` | Whether an OpenAI API key is configured |
| `has_anthropic_key` | boolean | `false` | Whether an Anthropic API key is configured |
| `has_huggingface_token` | boolean | `false` | Whether a HuggingFace token is configured |

### Wallet

| Field | Type | Default | Description |
|---|---|---|---|
| `wallet_address` | string or null | `null` | Base mainnet wallet address for FTNS token earnings |

### AI Integration (MCP)

| Field | Type | Default | Range | Description |
|---|---|---|---|---|
| `mcp_server_enabled` | boolean | `true` | — | Enable the MCP server for AI assistant integration |
| `mcp_server_port` | integer | `9100` | 1024–65535 | Port for the MCP server |

### Meta

| Field | Type | Default | Description |
|---|---|---|---|
| `setup_completed` | boolean | `false` | Whether the setup wizard has been run |
| `setup_version` | string | `"1.0.0"` | Schema version of the config (for migrations) |

---

## Full Example Config

```yaml
# Node identity
display_name: "my-research-node"
node_role: "full"

# Resources
cpu_pct: 60
memory_pct: 50
gpu_pct: 80
storage_gb: 50.0
max_concurrent_jobs: 5
upload_mbps_limit: 10.0

# Scheduling
active_hours_start: 8
active_hours_end: 22
active_days: []  # every day

# Network
p2p_port: 9001
api_port: 8000
bootstrap_nodes:
  - "wss://bootstrap1.prsm-network.com:8765"

# API keys (actual keys in ~/.prsm/.env)
has_openai_key: true
has_anthropic_key: true
has_huggingface_token: false

# Wallet
wallet_address: "0x1234567890abcdef1234567890abcdef12345678"

# AI Integration
mcp_server_enabled: true
mcp_server_port: 9100

# Meta
setup_completed: true
setup_version: "1.0.0"
```

---

## Environment Variable Overrides

Any config field can be overridden with an environment variable using the `PRSM_` prefix. The variable name is the field name in uppercase:

| Config Field | Environment Variable |
|---|---|
| `cpu_pct` | `PRSM_CPU_PCT` |
| `memory_pct` | `PRSM_MEMORY_PCT` |
| `gpu_pct` | `PRSM_GPU_PCT` |
| `storage_gb` | `PRSM_STORAGE_GB` |
| `p2p_port` | `PRSM_P2P_PORT` |
| `api_port` | `PRSM_API_PORT` |
| `node_role` | `PRSM_NODE_ROLE` |
| `display_name` | `PRSM_DISPLAY_NAME` |
| `mcp_server_enabled` | `PRSM_MCP_SERVER_ENABLED` |
| `mcp_server_port` | `PRSM_MCP_SERVER_PORT` |

Environment variables take precedence over values in `config.yaml`. This is useful for containerized deployments:

```bash
PRSM_CPU_PCT=80 PRSM_NODE_ROLE=contributor prsm start
```

---

## CLI Config Commands

### Show all settings

```bash
prsm config show
```

Displays the full current configuration with values and sources (file, environment, default).

### Get a single value

```bash
prsm config get cpu_pct
# 50
```

### Set a value

```bash
prsm config set cpu_pct 70
prsm config set display_name "gpu-beast"
prsm config set active_days '["mon","tue","wed","thu","fri"]'
```

Values are validated against the schema before saving. Out-of-range values are rejected with an error message.

### Reset to defaults

```bash
# Reset a single field
prsm config reset cpu_pct

# Reset everything
prsm config reset --all
```

### Export and import

```bash
# Export current config to a file
prsm config export my-config.yaml

# Import config from a file
prsm config import my-config.yaml
```

Useful for replicating settings across multiple nodes.

### Validate

```bash
prsm config validate
```

Checks the current config file against the schema and reports any issues.

### Show config file path

```bash
prsm config path
# /Users/you/.prsm/config.yaml
```

---

## Resource Allocation Guide

How much to allocate depends on your hardware and how you use your machine.

### Dedicated server (headless, PRSM-only)

```yaml
cpu_pct: 85
memory_pct: 80
gpu_pct: 95
storage_gb: 100.0
max_concurrent_jobs: 10
```

### Shared workstation (PRSM + daily use)

```yaml
cpu_pct: 40
memory_pct: 40
gpu_pct: 60
storage_gb: 20.0
max_concurrent_jobs: 3
```

### Laptop (conservative, battery-aware)

```yaml
cpu_pct: 20
memory_pct: 25
gpu_pct: 0
storage_gb: 5.0
max_concurrent_jobs: 1
upload_mbps_limit: 5.0
active_hours_start: 22
active_hours_end: 8
```

### GPU-focused contributor

```yaml
cpu_pct: 30
memory_pct: 50
gpu_pct: 90
storage_gb: 50.0
max_concurrent_jobs: 5
```

### General guidelines

- **CPU**: Leave at least 10–20% headroom for your OS and other applications.
- **Memory**: Going above 80% risks triggering OOM kills. Be conservative on machines with <16 GB RAM.
- **GPU**: GPU jobs are the highest-value compute on the network. Allocate generously if you have one.
- **Storage**: Cached datasets and model artifacts accumulate over time. Check usage periodically with `prsm status`.
- **Concurrent jobs**: More jobs earn more FTNS, but each consumes resources. Start low and increase if your machine handles it well.

---

## Network Configuration

### Ports

PRSM uses three ports by default:

| Port | Purpose | Protocol |
|---|---|---|
| `9001` | P2P peer connections | WebSocket |
| `8000` | Local management API | HTTP |
| `9100` | MCP server (AI integration) | HTTP |

If any port conflicts with another service, change it in your config or at startup:

```bash
prsm config set p2p_port 9002
prsm config set api_port 8001
```

### Firewall

For inbound peer connections, open your P2P port (default 9001). The API and MCP ports only need to be accessible locally unless you're exposing them intentionally.

### Bootstrap Nodes

Bootstrap nodes are the entry point to the PRSM network. Your node contacts them on startup to discover other peers.

The default bootstrap server is always available:

```
wss://bootstrap1.prsm-network.com:8765
```

To run a local test network, you can point nodes at each other:

```yaml
# Node A config
p2p_port: 9001
bootstrap_nodes: []

# Node B config
p2p_port: 9002
bootstrap_nodes:
  - "ws://localhost:9001"
```

---

## Active Hours & Days Scheduling

Schedule when your node accepts compute jobs. Outside active hours, the node stays connected to the network but won't pick up new work.

**Hours** use 24-hour format (0–23). **Days** use three-letter abbreviations: `mon`, `tue`, `wed`, `thu`, `fri`, `sat`, `sun`.

```yaml
# Active 9 AM to 11 PM, weekdays only
active_hours_start: 9
active_hours_end: 23
active_days: ["mon", "tue", "wed", "thu", "fri"]
```

```yaml
# Overnight only (e.g., for a workstation you use during the day)
active_hours_start: 22
active_hours_end: 7
active_days: []  # every day
```

Set both `active_hours_start` and `active_hours_end` to `null` (or omit them) to run 24/7. Leave `active_days` as an empty list to be active every day.
