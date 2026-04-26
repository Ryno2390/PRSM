#!/usr/bin/env bash
#
# Regenerate Formula/prsm.rb after a prsm-network PyPI publish.
#
# Replaces the placeholder sha256 values in the formula scaffold with real
# values computed against the actually-published wheels. Uses
# `homebrew-pypi-poet` to enumerate the full transitive Python dependency
# tree as Homebrew `resource` blocks.
#
# Prerequisites:
#   - prsm-network must already be published to PyPI at the target version
#   - homebrew-pypi-poet installed (pip install homebrew-pypi-poet)
#   - ruby (for parsing/formatting the generated formula)
#
# Usage:
#   bash ops/homebrew-tap/scripts/regenerate-formula.sh [VERSION]
#
# Default VERSION is read from the project's pyproject.toml.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# Resolve version: argv[1] OR parse pyproject.toml
VERSION="${1:-}"
if [ -z "$VERSION" ]; then
  VERSION=$(grep -E '^version\s*=' pyproject.toml | head -1 | sed -E 's/version\s*=\s*"([^"]+)"/\1/')
fi
echo "Target version: $VERSION"

# Step 1: Verify the version is published on PyPI
PYPI_URL="https://pypi.org/pypi/prsm-network/${VERSION}/json"
if ! curl -fsSL "$PYPI_URL" >/dev/null 2>&1; then
  echo "❌ prsm-network==${VERSION} is not published to PyPI."
  echo "   Run 'make publish-pypi' first; this script needs the published"
  echo "   sdist + wheels to compute real sha256 values."
  exit 1
fi
echo "✅ Verified prsm-network==${VERSION} is on PyPI."

# Step 2: Compute sha256 of the source tarball (the formula's main `url`)
echo "Computing sdist sha256..."
SDIST_URL="https://files.pythonhosted.org/packages/source/p/prsm-network/prsm-network-${VERSION}.tar.gz"
SDIST_SHA=$(curl -fsSL "$SDIST_URL" | shasum -a 256 | awk '{print $1}')
echo "  sdist sha256: $SDIST_SHA"

# Step 3: Generate full resource block via homebrew-pypi-poet
#         (creates a temporary venv, installs prsm-network, walks the dep tree)
if ! command -v poet >/dev/null 2>&1; then
  echo "❌ homebrew-pypi-poet not found. Install with: pip install homebrew-pypi-poet"
  exit 1
fi

echo "Generating resource blocks via homebrew-pypi-poet..."
TMPVENV=$(mktemp -d)
python3 -m venv "$TMPVENV"
"$TMPVENV/bin/pip" install --quiet "prsm-network==${VERSION}"
RESOURCES=$("$TMPVENV/bin/poet" prsm-network)
rm -rf "$TMPVENV"

# Step 4: Splice into Formula/prsm.rb
FORMULA="ops/homebrew-tap/Formula/prsm.rb"
TMPFORMULA=$(mktemp)

awk -v sha="$SDIST_SHA" -v version="$VERSION" -v resources="$RESOURCES" '
  /sha256 "0000000000000000000000000000000000000000000000000000000000000000"/ && !sha_replaced {
    print "  sha256 \"" sha "\""
    sha_replaced = 1
    next
  }
  /^  resource / { in_old_resources = 1 }
  /^  end$/ && in_old_resources { in_old_resources = 0; next }
  in_old_resources { next }
  /^  def install$/ {
    print resources
    print ""
    print $0
    next
  }
  { print }
' "$FORMULA" > "$TMPFORMULA"

mv "$TMPFORMULA" "$FORMULA"
echo "✅ Regenerated $FORMULA"

# Step 5: Verify with brew audit if available
if command -v brew >/dev/null 2>&1; then
  echo "Running brew audit..."
  brew audit --strict --formula "$FORMULA" || echo "⚠️  brew audit reported issues; review above."
else
  echo "⚠️  brew not installed; skipped audit step."
fi

echo
echo "Next steps:"
echo "  1. Review the diff: git diff $FORMULA"
echo "  2. Test locally: brew install --build-from-source $FORMULA"
echo "  3. Sync to tap repo: bash ops/homebrew-tap/scripts/sync-to-tap.sh"
