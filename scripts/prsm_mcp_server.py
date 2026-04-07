#!/usr/bin/env python3
"""
PRSM MCP Server — Clean Entry Point
====================================

This script lives OUTSIDE the prsm package so Python's -m resolution
doesn't trigger prsm/__init__.py imports before we can redirect stdout.

Usage:
    python scripts/prsm_mcp_server.py

Configure in Claude Desktop:
    {"mcpServers": {"prsm": {"command": "python", "args": ["scripts/prsm_mcp_server.py"]}}}
"""

import sys
import os
import logging

# 1. Redirect stdout to stderr BEFORE any prsm imports
_real_stdout = sys.stdout
sys.stdout = sys.stderr

# 2. Kill all logging
logging.basicConfig(level=logging.CRITICAL, stream=sys.stderr)
for name in ["prsm", "structlog", "httpx", "aiohttp", "mcp"]:
    logging.getLogger(name).setLevel(logging.CRITICAL)

# 3. Set env flag to suppress structlog output
os.environ["PRSM_QUIET"] = "1"

# 4. Add project root to path if running from repo
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 5. Now import prsm — all noise goes to stderr
from prsm.mcp_server import run_server

# 6. Restore stdout for MCP JSON-RPC protocol
sys.stdout = _real_stdout

# 7. Run
import asyncio
asyncio.run(run_server())
