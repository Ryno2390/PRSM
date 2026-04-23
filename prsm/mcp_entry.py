"""
MCP Server Entry Point (clean stdout)
======================================

Standalone entry point that suppresses all stdout noise
before importing PRSM modules. Use this instead of
``python -m prsm.mcp_server`` for clean MCP protocol.

Usage:
    python -m prsm.mcp_entry
"""

import sys

# IMMEDIATELY redirect stdout to stderr before ANY imports
# This catches structlog output from prsm/__init__.py
_real_stdout = sys.stdout
sys.stdout = sys.stderr

# Suppress all logging
import logging
logging.basicConfig(level=logging.CRITICAL, stream=sys.stderr)

# Now import prsm (all noise goes to stderr)
from prsm.mcp_server import run_server

# Restore stdout for MCP protocol
sys.stdout = _real_stdout

import asyncio

def main():
    asyncio.run(run_server())

if __name__ == "__main__":
    main()
