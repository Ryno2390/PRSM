#!/usr/bin/env bash
#
# Smoke tests for the prsm-mcp npm wrapper.
#
# Tests are pure-Node where possible; the spawn-Python path is exercised
# via short-circuit help/version flags so we don't depend on the real
# Python package being installed.

set -euo pipefail

cd "$(dirname "$0")"

GREEN="\033[0;32m"
RED="\033[0;31m"
NC="\033[0m"

pass() { printf "${GREEN}PASS${NC}  %s\n" "$1"; }
fail() { printf "${RED}FAIL${NC}  %s\n" "$1"; exit 1; }

# 1. Help flag exits 0
if node bin/prsm-mcp.js --help >/dev/null 2>&1; then
  pass "--help exits 0"
else
  fail "--help should exit 0"
fi

# 2. -h alias works
if node bin/prsm-mcp.js -h >/dev/null 2>&1; then
  pass "-h exits 0"
else
  fail "-h should exit 0"
fi

# 3. Version flag exits 0 and prints something starting with "prsm-mcp"
HELP_OUT=$(node bin/prsm-mcp.js --version 2>&1 || true)
if echo "$HELP_OUT" | grep -q "prsm-mcp"; then
  pass "--version prints wrapper name"
else
  fail "--version should print 'prsm-mcp ...'"
fi

# 4. Help output goes to stderr, NOT stdout (MCP stdio purity)
STDOUT=$(node bin/prsm-mcp.js --help 2>/dev/null || true)
if [ -z "$STDOUT" ]; then
  pass "--help writes nothing to stdout"
else
  fail "--help leaked output to stdout (MCP stdio purity violation)"
fi

# 5. Version output goes to stderr only
STDOUT_V=$(node bin/prsm-mcp.js --version 2>/dev/null || true)
if [ -z "$STDOUT_V" ]; then
  pass "--version writes nothing to stdout"
else
  fail "--version leaked output to stdout"
fi

# 6. python-detect module exports work
if node -e "require('./lib/python-detect').detectPython()" 2>/dev/null; then
  pass "python-detect module loadable"
else
  fail "python-detect module not loadable"
fi

# 7. ensure-package module exports work
if node -e "require('./lib/ensure-package').isPackageAvailable" 2>/dev/null; then
  pass "ensure-package module loadable"
else
  fail "ensure-package module not loadable"
fi

# 8. parseVersion extracts version tuples correctly
node -e '
const { parseVersion, meetsMinimum } = require("./lib/python-detect");
const v = parseVersion("Python 3.12.4");
if (!v || v[0] !== 3 || v[1] !== 12 || v[2] !== 4) { process.exit(1); }
if (!meetsMinimum([3, 10, 0])) process.exit(1);
if (meetsMinimum([3, 9, 0])) process.exit(1);
if (meetsMinimum([2, 7, 0])) process.exit(1);
if (!meetsMinimum([3, 99, 0])) process.exit(1);
' && pass "version parsing + minimum check correct" || fail "version parsing/check broken"

# 9. stripWrapperArgs removes --auto-install
node -e '
const { stripWrapperArgs } = require("./lib/ensure-package");
const out = stripWrapperArgs(["--auto-install", "--other-flag", "value"]);
if (out.length !== 2 || out[0] !== "--other-flag" || out[1] !== "value") {
  console.error("Got:", JSON.stringify(out));
  process.exit(1);
}
' && pass "stripWrapperArgs filters --auto-install" || fail "stripWrapperArgs broken"

# 10. autoInstallRequested honors env + argv
node -e '
const { autoInstallRequested } = require("./lib/ensure-package");
if (autoInstallRequested([])) process.exit(1);
if (!autoInstallRequested(["--auto-install"])) process.exit(1);
process.env.PRSM_AUTO_INSTALL = "1";
if (!autoInstallRequested([])) process.exit(1);
' && pass "autoInstallRequested honors env + argv" || fail "autoInstallRequested broken"

# 11. package.json declares correct binary name
if grep -q '"prsm-mcp": "bin/prsm-mcp.js"' package.json; then
  pass "package.json bin entry correct"
else
  fail "package.json missing bin entry"
fi

# 12. bin script has shebang and exec bit
if [ -x "bin/prsm-mcp.js" ] && head -1 bin/prsm-mcp.js | grep -q "node"; then
  pass "bin script is executable with node shebang"
else
  fail "bin script missing exec bit or shebang"
fi

echo
echo "All smoke tests passed."
