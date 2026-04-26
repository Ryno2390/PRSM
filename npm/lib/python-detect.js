/**
 * Python interpreter detection for the prsm-mcp wrapper.
 *
 * Tries platform-appropriate Python invocations and returns the first one
 * that reports a Python 3.10+ version. Returns null if no compatible Python
 * is found.
 *
 * Order:
 *   - Unix-like (darwin, linux): python3, python
 *   - Windows: py -3, python3, python
 *
 * Detection writes ONLY to stderr (never stdout), matching MCP stdio
 * purity requirements. The wrapper itself must not print anything to
 * stdout that isn't a JSON-RPC message from the underlying Python server.
 */

"use strict";

const { spawnSync } = require("child_process");

/**
 * Minimum supported Python version (matches `prsm-network` package
 * pyproject.toml requires-python).
 */
const MIN_PYTHON_MAJOR = 3;
const MIN_PYTHON_MINOR = 10;

/**
 * Candidate invocations per platform. Each entry is [command, args[]].
 * Args are passed verbatim before any version-check argv we add.
 */
function candidates() {
  if (process.platform === "win32") {
    return [
      ["py", ["-3"]],
      ["python3", []],
      ["python", []],
    ];
  }
  return [
    ["python3", []],
    ["python", []],
  ];
}

/**
 * Return the parsed [major, minor, patch] tuple from a Python `--version`
 * stderr/stdout line, or null if it can't be parsed.
 */
function parseVersion(output) {
  if (!output) return null;
  const match = output.match(/Python\s+(\d+)\.(\d+)\.(\d+)/);
  if (!match) return null;
  return [parseInt(match[1], 10), parseInt(match[2], 10), parseInt(match[3], 10)];
}

/**
 * Returns true iff the version tuple meets the minimum requirement.
 */
function meetsMinimum(version) {
  if (!version) return false;
  const [major, minor] = version;
  if (major > MIN_PYTHON_MAJOR) return true;
  if (major < MIN_PYTHON_MAJOR) return false;
  return minor >= MIN_PYTHON_MINOR;
}

/**
 * Detect a usable Python interpreter.
 *
 * @returns {{cmd: string, args: string[], version: number[]} | null}
 */
function detectPython() {
  for (const [cmd, prefixArgs] of candidates()) {
    const result = spawnSync(cmd, [...prefixArgs, "--version"], {
      encoding: "utf-8",
      shell: false,
    });
    if (result.error) continue;
    if (result.status !== 0) continue;

    // Python 2 prints version to stderr; Python 3 to stdout. Check both.
    const output = `${result.stdout || ""}${result.stderr || ""}`;
    const version = parseVersion(output);

    if (meetsMinimum(version)) {
      return { cmd, args: prefixArgs, version };
    }
  }
  return null;
}

/**
 * Format a Python version tuple as a human-readable string.
 */
function formatVersion(version) {
  if (!version) return "unknown";
  return version.join(".");
}

module.exports = {
  detectPython,
  parseVersion,
  meetsMinimum,
  formatVersion,
  MIN_PYTHON_MAJOR,
  MIN_PYTHON_MINOR,
};
