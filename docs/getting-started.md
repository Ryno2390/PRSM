# PRSM Getting Started Guide

> PRSM — P2P infrastructure protocol for open-source collaboration.

## Quick Install

```
$ pip install prsm-network

# Check it works
$ prsm --version
0.3.2
```

## First Run: Setup Wizard

The first time you interact with PRSM, run the interactive setup:

```
$ prsm setup
```

This walks you through:
- System detection (CPU cores, RAM, GPUs, disk space)
- Role selection (Contribute, Consume, or Both)
- Resource allocation (CPU%, RAM%, storage pledge)
- Network config (ports, bootstrap nodes)
- AI assistant integration (MCP server for Hermes / OpenClaw / Claude)

### Quick Setup (Smart Defaults)

```
$ prsm setup --minimal
```

### Dry Run (Preview Without Saving)

```
$ prsm setup --dry-run
```

## Starting Your Node

### Background Daemon (Recommended)

```
$ prsm daemon start              # start in background
$ prsm daemon status             # check status
$ prsm daemon logs -f            # follow logs live
$ prsm daemon stop               # stop gracefully
```

### Foreground (Interactive)

```
$ prsm node start                 # full node with live dashboard
$ prsm node start --no-dashboard  # full node, static console view
```

### Auto-Start on Boot

```
$ prsm daemon install             # install launchd (macOS) or systemd (Linux)
$ prsm daemon uninstall           # remove service
```

## Configuration

```
$ prsm config show                # display all settings (human-readable)
$ prsm config show --format json  # machine-readable
$ prsm config set cpu_pct 60     # change a single setting
$ prsm config get p2p_port        # get one value
$ prsm config path               # path to config file (~/.prsm/config.yaml)
$ prsm config validate           # check config validity
$ prsm config export             # export current config as YAML
$ prsm config import file.yaml   # load a config file
$ prsm config reset              # reset to defaults (confirms first)
```

### Key Settings

Setting | Range | Default | Description
--- | --- | --- | ---
`cpu_pct` | 10-90 | 50 | CPU allocation for compute jobs
`memory_pct` | 10-90 | 50 | RAM allocation
`gpu_pct` | 0-100 | 80 | GPU allocation (0 = disabled)
`storage_gb` | 1+ | 10.0 | Storage pledge in GB
`max_concurrent_jobs` | 1-20 | 3 | Max parallel compute jobs
`p2p_port` | 1024-65535 | 9001 | P2P network port
`api_port` | 1024-65535 | 8000 | REST API port
`mcp_server_enabled` | true/false | true | Enable MCP server for AI assistants
`mcp_server_port` | 1024-65535 | 9100 | MCP server port
`node_role` | full/contributor/consumer | full | Node role

## Daemon Management

```
$ prsm daemon start               # start as background process
$ prsm daemon stop                # stop (SIGTERM then SIGKILL after 10s)
$ prsm daemon restart             # stop + start
$ prsm daemon status              # show status (running/stopped, PID, uptime)
$ prsm daemon status --format json
$ prsm daemon logs -n 100         # show last 100 lines
$ prsm daemon logs -f             # follow (tail -f style)
$ prsm daemon install             # install system service
$ prsm daemon install --dry-run   # print service file without installing
$ prsm daemon uninstall           # remove system service
```

## Skill Packages

PRSM ships with built-in skill packages that any AI can ingest:

```
$ prsm skills list              # list installed skills
$ prsm skills search <query>    # search the network for skills
$ prsm skills install <pkg>     # install a skill package
$ prsm skills remove <pkg>      # remove a skill package
$ prsm skills info <pkg>        # show detailed skill info
```

---

## What's Next?

- See `docs/configuration.md` for the full configuration reference
- See `docs/ai-integration.md` for connecting AI assistants via MCP
- Join the community: `prsm node start` connects you to the bootstrap network
