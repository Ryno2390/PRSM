# PRSM Configuration Reference

All settings live in `~/.prsm/config.yaml` (YAML format). The schema is
enforced by Pydantic via `prsm.cli_modules.config_schema.PRSMConfig`.

## Config File Location

```
~/.prsm/
├── config.yaml            # unified configuration
├── daemon.pid             # background daemon PID
├── logs/
│   └── daemon.log         # daemon stdout/stderr
└── skills/                # installed skill packages
```

## Full Schema

```yaml
# === Node Identity ===
display_name: "prsm-node"        # human-readable name
node_role: "full"                # full | contributor | consumer

# === Resource Allocation ===
cpu_pct: 50                      # 10-90 (% of total CPU)
memory_pct: 50                   # 10-90 (% of total RAM)
gpu_pct: 80                      # 0-100 (% of GPU; 0 = disabled)
storage_gb: 10.0                 # minimum 1.0 GB
max_concurrent_jobs: 3           # 1-20 concurrent compute jobs
upload_mbps_limit: 0.0           # 0 = unlimited
active_hours_start: null         # null = always active
active_hours_end: null
active_days: []                  # [] = every day

# === Network ===
p2p_port: 9001                   # P2P listen port
api_port: 8000                   # REST API port
bootstrap_nodes: []              # list of ws/wss:// addresses

# === API Keys (flags; actual keys stored in ~/.prsm/.env) ===
has_openai_key: false
has_anthropic_key: false
has_huggingface_token: false

# === FTNS Wallet ===
wallet_address: null             # Base mainnet 0x... address

# === AI Integration ===
mcp_server_enabled: true         # expose MCP server for AI assistants
mcp_server_port: 9100

# === Meta ===
setup_completed: false           # true after first successful setup
setup_version: "1.0.0"
```

## CLI Commands

### View

```
prsm config              # re-run the interactive config wizard
prsm config show         # styled display of all settings
prsm config show --format json   # JSON output
prsm config get <key>    # print one value to stdout
prsm config path         # print config file path
prsm config export       # dump full config as YAML
prsm config validate     # check config for errors
```

### Modify

```
prsm config set <key> <value>     # update a single setting
prsm config reset                 # restore defaults (confirms first)
prsm config reset --yes           # skip confirmation
prsm config import <file.yaml>    # load config from file
```

### Keys You Can Set

```
prsm config set display_name "my-node"
prsm config set node_role full
prsm config set cpu_pct 60
prsm config set memory_pct 40
prsm config set gpu_pct 90
prsm config set storage_gb 50
prsm config set max_concurrent_jobs 5
prsm config set p2p_port 9002
prsm config set api_port 8001
prsm config set mcp_server_enabled true
prsm config set mcp_server_port 9200
prsm config set bootstrap_nodes "wss://node1:8000,wss://node2:8000"
```

## Backward Compatibility

Old versions used `~/.prsm/node_config.json` (the legacy `NodeConfig` dataclass).
The migration runs automatically:

- On startup: `migrate_if_needed()` converts `node_config.json` → `config.yaml`
- Old file backed up to `node_config.json.bak`
- `prsm setup --reset` also triggers migration

The legacy `prsm node configure` command still works but prints a deprecation
notice pointing you to `prsm config`.

## Programmatic API

```python
from prsm.cli_modules.config_schema import PRSMConfig

# Load (creates defaults if no file exists)
cfg = PRSMConfig.load()

# Check / modify
print(cfg.node_role)    # NodeRole.FULL
cfg.cpu_pct = 70

# Validate and save
cfg.save()

# Reset to defaults
if PRSMConfig.config_path().exists():
    PRSMConfig.config_path().unlink()
```
