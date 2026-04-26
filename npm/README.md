# prsm-mcp

[![npm version](https://badge.fury.io/js/prsm-mcp.svg)](https://www.npmjs.com/package/prsm-mcp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MCP server wrapper for [PRSM](https://github.com/prsm-network/PRSM)** — Protocol for Research, Storage, and Modeling.

This npm package is a thin Node.js wrapper that lets MCP clients (Claude Desktop, Cursor, Continue, Cody, etc.) launch PRSM's MCP server with `npx prsm-mcp` instead of running `pip install prsm-network && prsm mcp-server` manually.

The wrapper auto-detects a Python interpreter, verifies the `prsm-network` package is installed, then spawns `python -m prsm.mcp_server` with stdio inherited so the MCP JSON-RPC stream flows directly to the underlying Python implementation.

## Quick Start

### MCP client configuration

**Claude Desktop** (`~/.claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "prsm": {
      "command": "npx",
      "args": ["prsm-mcp"]
    }
  }
}
```

**Cursor** (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "prsm": {
      "command": "npx",
      "args": ["prsm-mcp"]
    }
  }
}
```

After adding to your MCP client config, restart the client. PRSM's 17 tools (`prsm_analyze`, `prsm_inference`, `prsm_quote`, etc.) become available to the LLM automatically.

### Prerequisites

- **Node.js 18+** (the wrapper itself)
- **Python 3.10+** (the actual PRSM server)
- **`prsm-network` Python package** — auto-prompted on first run

If Python is not installed, the wrapper prints clear instructions pointing to python.org or your platform's package manager.

If `prsm-network` is not installed, the wrapper prints the install command. To opt into auto-install:

```bash
npx prsm-mcp --auto-install
# or
PRSM_AUTO_INSTALL=1 npx prsm-mcp
```

## What this wrapper is — and isn't

**It IS:**
- A convenience layer for users who already have Python but don't want to memorize PRSM's CLI
- A standard MCP-client integration path (`command: "npx"`, `args: ["prsm-mcp"]`)
- A first-run sanity check (right Python version, package installed)
- A signal forwarder (SIGINT/SIGTERM cleanly stop the Python child)

**It IS NOT:**
- A reimplementation of PRSM in Node.js
- An embedded Python distribution (the wrapper is ~5KB; embedded Python would be 50-100MB)
- A hosted-service client (PRSM is self-hosted; see [hosted-MCP-server-decision-memo](https://github.com/prsm-network/PRSM/blob/main/docs/2026-04-26-hosted-mcp-server-decision-memo.md))
- A wallet, account manager, or billing system (those live in the Python `prsm-network` package)

## Available MCP tools

PRSM exposes 17 MCP tools. See the [main README](https://github.com/prsm-network/PRSM#mcp-integration) for the full catalog. Highlights:

- `prsm_analyze` — full Ring 1-10 pipeline (query in, answer out)
- `prsm_inference` — TEE-attested model inference with verifiable receipts
- `prsm_quote` — cost estimate before committing FTNS
- `prsm_create_agent` / `prsm_dispatch_agent` — custom agent construction
- `prsm_upload_dataset` — publish data with on-chain royalties
- `prsm_yield_estimate` — predict node-operator earnings

## Environment variables

| Variable | Purpose |
|---|---|
| `PRSM_AUTO_INSTALL=1` | Equivalent to `--auto-install` flag |
| `PRSM_NODE_URL` | Override the default node API (`http://localhost:8000`) |
| `PRSM_NODE_API_KEY` | Bearer token for node API authentication |

## CLI flags

| Flag | Purpose |
|---|---|
| `--auto-install` | Install `prsm-network` via pip if missing |
| `--help`, `-h` | Print usage information (to stderr — does not corrupt MCP stdio) |
| `--version`, `-V` | Print wrapper version |

Any other flags are forwarded to the Python MCP server.

## MCP stdio purity

The MCP protocol requires that **stdout carry only JSON-RPC messages** from the server. This wrapper writes ALL diagnostic output (Python detection messages, install instructions, errors) to **stderr**, never stdout.

Test this yourself:

```bash
# Should produce zero bytes:
npx prsm-mcp --version 2>/dev/null

# Should produce wrapper diagnostic on stderr:
npx prsm-mcp --version 2>&1 >/dev/null
```

If you see anything other than valid JSON-RPC on stdout when the server is running, please file an issue at https://github.com/prsm-network/PRSM/issues — that's a bug.

## Platform support

| Platform | Status |
|---|---|
| macOS (Apple Silicon + Intel) | Supported |
| Linux (Ubuntu, Debian, Fedora) | Supported |
| Windows 10/11 | Supported (uses `py -3` if available) |

The wrapper itself is pure JavaScript and runs anywhere Node.js 18+ runs. Compatibility constraints come from the underlying Python `prsm-network` package.

## Development

The wrapper source lives in [PRSM/npm/](https://github.com/prsm-network/PRSM/tree/main/npm). Smoke tests:

```bash
cd npm
bash test.sh
```

Tests verify:
- `--help` and `--version` exit cleanly
- All wrapper output goes to stderr (MCP stdio purity)
- Python detection module parses versions correctly
- Argument forwarding strips wrapper-only flags
- Auto-install opt-in honors both env and argv

## Related distribution channels

| Channel | Use case |
|---|---|
| **`npx prsm-mcp`** (this package) | MCP client integration, fast onboarding |
| **`pip install prsm-network`** | Direct Python use, full CLI access |
| **`brew install prsm/tap/prsm`** (Phase 3.x.1 Task 10) | macOS ergonomics |

The Python package (`prsm-network` on PyPI) remains the canonical install path. This npm wrapper is a convenience layer atop it.

## Troubleshooting

**"no Python 3.10+ interpreter found"** — install Python 3.10 or newer from python.org or your platform's package manager.

**"prsm-network is not installed"** — run the suggested `pip install prsm-network` command, or use `--auto-install`.

**MCP client shows blank tool list** — restart the MCP client after editing the config file. Some clients cache tool listings.

**Connection refused / node not running** — the MCP server starts on demand from the MCP client. PRSM's node-side runtime (`prsm node start`) is a separate process that you'll want running for full functionality. See the [main Getting Started guide](https://github.com/prsm-network/PRSM/blob/main/docs/GETTING_STARTED.md).

## License

MIT, same as parent PRSM repo.

## Links

- **Main repo:** https://github.com/prsm-network/PRSM
- **Vision document:** [PRSM_Vision.md](https://github.com/prsm-network/PRSM/blob/main/PRSM_Vision.md)
- **Phase 3.x.1 design:** [`docs/2026-04-26-phase3.x.1-mcp-server-completion-design-plan.md`](https://github.com/prsm-network/PRSM/blob/main/docs/2026-04-26-phase3.x.1-mcp-server-completion-design-plan.md)
- **Discord:** https://discord.gg/R8dhCBCUp3
- **Issues:** https://github.com/prsm-network/PRSM/issues
