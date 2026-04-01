# PRSM Network Skill

Network operations and monitoring on the PRSM decentralized network.

## Overview

This skill enables AI agents to monitor network health, manage peer connections,
discover new nodes, and gather network-wide statistics. Essential for maintaining
a healthy presence on the PRSM network.

## Tools

| Tool | Description |
|------|-------------|
| `prsm_list_peers` | List connected peers with status and capability filtering |
| `prsm_network_stats` | Get aggregate network statistics and metrics |
| `prsm_health_check` | Run diagnostic health checks on local node and connectivity |
| `prsm_discover_nodes` | Find and connect to new nodes by capability and region |

## Prompts

- **operator** — System prompt for a network operations agent

## Example Usage

```
List active compute peers:
  prsm_list_peers(status="active", capability="compute")

Get 24-hour network overview:
  prsm_network_stats(metric="overview", timeframe="24h")

Run full health check:
  prsm_health_check(checks=["all"], verbose=true)

Discover GPU nodes in your region:
  prsm_discover_nodes(capability="gpu", min_reputation=0.8, connect=true)
```

## Network Health

A healthy PRSM node maintains:
- Active connections to 5+ peers
- Latency under 200ms to nearest peers
- Available storage for data caching
- Compute resources registered (if contributing)
