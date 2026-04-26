#!/usr/bin/env node
/**
 * prsm-mcp — npm entrypoint for the PRSM Model Context Protocol server.
 *
 * This wrapper exists so MCP clients (Claude Desktop, Cursor, Continue,
 * etc.) can launch PRSM via `npx prsm-mcp` without users needing to
 * install Python or invoke `prsm mcp-server` manually.
 *
 * The wrapper does three things:
 *   1. Detect a Python 3.10+ interpreter on the user's PATH.
 *   2. Verify the `prsm-network` package is importable; instruct the
 *      user to install it (or auto-install if opted in).
 *   3. Spawn `python -m prsm.mcp_server` with stdio inherited so the
 *      MCP JSON-RPC stream flows directly between the MCP client and
 *      the Python server.
 *
 * CRITICAL: this wrapper writes ONLY to stderr. The MCP protocol
 * requires that stdout carry only JSON-RPC messages from the Python
 * server. Any wrapper-side print to stdout would corrupt the stream.
 */

"use strict";

const { spawn } = require("child_process");
const path = require("path");

const { detectPython, formatVersion, MIN_PYTHON_MAJOR, MIN_PYTHON_MINOR } = require(
  path.join(__dirname, "..", "lib", "python-detect")
);
const { ensurePrsmNetwork, stripWrapperArgs } = require(
  path.join(__dirname, "..", "lib", "ensure-package")
);

function main() {
  const argv = process.argv.slice(2);

  // Help flag — handle here rather than passing to Python so users get
  // npm-wrapper-specific guidance.
  if (argv.includes("--help") || argv.includes("-h")) {
    printHelp();
    process.exit(0);
  }

  if (argv.includes("--version") || argv.includes("-V")) {
    const pkg = require("../package.json");
    process.stderr.write(`prsm-mcp ${pkg.version} (Node.js wrapper for PRSM MCP server)\n`);
    process.exit(0);
  }

  // Step 1 — detect Python
  const python = detectPython();
  if (!python) {
    process.stderr.write(
      `prsm-mcp: no Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ interpreter found on PATH.\n\n` +
        `PRSM requires Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR} or newer. Install from:\n\n` +
        `    https://www.python.org/downloads/\n\n` +
        `Or use a system package manager:\n` +
        `    macOS:   brew install python@3.12\n` +
        `    Ubuntu:  sudo apt install python3.12\n` +
        `    Windows: winget install Python.Python.3.12\n\n` +
        `After installing, re-run prsm-mcp.\n`
    );
    process.exit(1);
  }

  process.stderr.write(
    `prsm-mcp: using ${python.cmd}${python.args.length ? " " + python.args.join(" ") : ""} ` +
      `(Python ${formatVersion(python.version)})\n`
  );

  // Step 2 — verify prsm-network installed (or install on opt-in)
  if (!ensurePrsmNetwork(python, argv)) {
    process.exit(1);
  }

  // Step 3 — spawn the Python MCP server with stdio inherited
  const pythonArgs = stripWrapperArgs(argv);
  const fullArgs = [...python.args, "-m", "prsm.mcp_server", ...pythonArgs];

  const child = spawn(python.cmd, fullArgs, {
    // stdio inherited so MCP JSON-RPC flows: client → wrapper stdin →
    // python stdin, python stdout → wrapper stdout → MCP client.
    // stderr inherited so users see Python errors without having to dig.
    stdio: "inherit",
    shell: false,
    env: process.env,
  });

  // Forward signals so SIGINT/SIGTERM from the MCP client cleanly stop
  // the Python server.
  const forwardSignal = (signal) => {
    if (!child.killed) {
      child.kill(signal);
    }
  };
  process.on("SIGINT", () => forwardSignal("SIGINT"));
  process.on("SIGTERM", () => forwardSignal("SIGTERM"));
  process.on("SIGHUP", () => forwardSignal("SIGHUP"));

  child.on("error", (err) => {
    process.stderr.write(`prsm-mcp: failed to spawn Python: ${err.message}\n`);
    process.exit(1);
  });

  child.on("exit", (code, signal) => {
    if (signal) {
      // Re-raise signal so parent (MCP client) sees it
      process.kill(process.pid, signal);
    } else {
      process.exit(code ?? 0);
    }
  });
}

function printHelp() {
  process.stderr.write(
    `prsm-mcp — npm wrapper for the PRSM MCP server.\n\n` +
      `USAGE:\n` +
      `    npx prsm-mcp [options]\n\n` +
      `OPTIONS:\n` +
      `    --auto-install        Install prsm-network via pip if missing\n` +
      `    --help, -h            Print this message\n` +
      `    --version, -V         Print wrapper version\n\n` +
      `ENVIRONMENT:\n` +
      `    PRSM_AUTO_INSTALL=1   Equivalent to --auto-install\n` +
      `    PRSM_NODE_URL         Override default node API (http://localhost:8000)\n` +
      `    PRSM_NODE_API_KEY     Bearer token for node API auth\n\n` +
      `MCP CLIENT CONFIGURATION:\n` +
      `    Add to ~/.claude/claude_desktop_config.json (or equivalent):\n\n` +
      `      {\n` +
      `        "mcpServers": {\n` +
      `          "prsm": {\n` +
      `            "command": "npx",\n` +
      `            "args": ["prsm-mcp"]\n` +
      `          }\n` +
      `        }\n` +
      `      }\n\n` +
      `MORE INFO:\n` +
      `    https://github.com/prsm-network/PRSM\n` +
      `    https://github.com/prsm-network/PRSM/blob/main/docs/GETTING_STARTED.md\n`
  );
}

main();
