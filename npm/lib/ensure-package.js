/**
 * Ensure the prsm-network Python package is importable.
 *
 * On a fresh machine, the user has Python but not the prsm-network package.
 * We test for it; if missing, we either:
 *   - print a clear install command and exit non-zero (default — safe)
 *   - auto-install via pip if PRSM_AUTO_INSTALL=1 or --auto-install was passed
 *
 * Auto-install is opt-in because installing Python packages from a
 * Node.js wrapper is unexpected behavior; users who want it can opt in,
 * everyone else gets a clean instruction.
 *
 * All output goes to stderr to preserve MCP stdio purity on stdout.
 */

"use strict";

const { spawnSync } = require("child_process");

const REQUIRED_PACKAGE = "prsm-network";

/**
 * Test whether `prsm-network` is importable by the given Python interpreter.
 *
 * We import a module from the package rather than just checking pip; this
 * catches edge cases like "package installed but PYTHONPATH excludes it"
 * or partial install corruption.
 *
 * @param {{cmd: string, args: string[]}} python
 * @returns {boolean}
 */
function isPackageAvailable(python) {
  const result = spawnSync(
    python.cmd,
    [...python.args, "-c", "import prsm; import prsm.mcp_server"],
    {
      encoding: "utf-8",
      shell: false,
      // Suppress Python's own stderr from leaking to user — we'll print
      // our own diagnostic instead.
      stdio: ["ignore", "pipe", "pipe"],
    }
  );
  return result.status === 0;
}

/**
 * Attempt to install prsm-network via pip.
 *
 * Returns true on success, false on failure. Output is forwarded to stderr
 * so the user can see what's happening during the install.
 */
function autoInstall(python) {
  process.stderr.write(`prsm-mcp: installing ${REQUIRED_PACKAGE} via pip...\n`);
  const result = spawnSync(
    python.cmd,
    [...python.args, "-m", "pip", "install", "--user", REQUIRED_PACKAGE],
    {
      stdio: ["ignore", "inherit", "inherit"],
      shell: false,
    }
  );
  return result.status === 0;
}

/**
 * Whether the user has opted into auto-install via env or argv flag.
 */
function autoInstallRequested(argv) {
  if (process.env.PRSM_AUTO_INSTALL === "1") return true;
  if (Array.isArray(argv) && argv.includes("--auto-install")) return true;
  return false;
}

/**
 * Strip any wrapper-specific args before passing argv onward to Python.
 */
function stripWrapperArgs(argv) {
  return argv.filter((a) => a !== "--auto-install");
}

/**
 * Ensure the package is available; prompt + install or fail loud.
 *
 * @param {{cmd: string, args: string[]}} python
 * @param {string[]} argv  process argv (sliced past `node prsm-mcp.js`)
 * @returns {boolean}  true if package is available after this call
 */
function ensurePrsmNetwork(python, argv) {
  if (isPackageAvailable(python)) return true;

  if (autoInstallRequested(argv)) {
    if (autoInstall(python)) {
      // Verify the install actually worked
      if (isPackageAvailable(python)) return true;
      process.stderr.write(
        `prsm-mcp: pip install completed but ${REQUIRED_PACKAGE} still not importable.\n` +
          `         Check your Python environment and PYTHONPATH.\n`
      );
      return false;
    }
    process.stderr.write(
      `prsm-mcp: pip install of ${REQUIRED_PACKAGE} failed (see output above).\n`
    );
    return false;
  }

  // Default path: print clear instruction and fail
  process.stderr.write(
    `prsm-mcp: the Python package "${REQUIRED_PACKAGE}" is not installed.\n\n` +
      `To install it, run ONE of:\n\n` +
      `    ${python.cmd}${python.args.length ? " " + python.args.join(" ") : ""} -m pip install ${REQUIRED_PACKAGE}\n` +
      `    pipx install ${REQUIRED_PACKAGE}\n\n` +
      `Or re-run prsm-mcp with auto-install:\n\n` +
      `    PRSM_AUTO_INSTALL=1 npx prsm-mcp\n` +
      `    npx prsm-mcp --auto-install\n\n` +
      `For more setup help, see: https://github.com/prsm-network/PRSM/blob/main/docs/GETTING_STARTED.md\n`
  );
  return false;
}

module.exports = {
  isPackageAvailable,
  autoInstall,
  ensurePrsmNetwork,
  autoInstallRequested,
  stripWrapperArgs,
  REQUIRED_PACKAGE,
};
