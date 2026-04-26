#!/usr/bin/env bash
#
# Sync ops/homebrew-tap/Formula/prsm.rb to the public tap repository
# at https://github.com/prsm-network/homebrew-tap.
#
# The tap repo is a separate GitHub repository (Homebrew convention:
# `<org>/homebrew-<name>`). Users install via `brew tap prsm-network/tap`
# which clones the tap repo, then `brew install prsm` resolves to
# Formula/prsm.rb in that repo.
#
# This script clones the tap repo to a temp directory, copies in the
# current Formula, commits with a clear message, and pushes. Requires
# write access to the tap repo on the user's GitHub credentials.
#
# Prerequisites:
#   - The tap repo prsm-network/homebrew-tap exists on GitHub
#     (created manually one time; can be empty)
#   - Current user has write access via SSH key or PAT
#   - ops/homebrew-tap/Formula/prsm.rb has been regenerated with real
#     sha256 values via regenerate-formula.sh
#
# Usage:
#   bash ops/homebrew-tap/scripts/sync-to-tap.sh [VERSION]

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

VERSION="${1:-$(grep -E '^version\s*=' pyproject.toml | head -1 | sed -E 's/version\s*=\s*"([^"]+)"/\1/')}"
TAP_REPO="git@github.com:prsm-network/homebrew-tap.git"
LOCAL_FORMULA="ops/homebrew-tap/Formula/prsm.rb"

# Sanity check: formula has real sha256 (not placeholder)
if grep -q 'sha256 "0000000000000000000000000000000000000000000000000000000000000000"' "$LOCAL_FORMULA"; then
  echo "❌ Formula still has placeholder sha256 values."
  echo "   Run regenerate-formula.sh first; pushing with placeholders would"
  echo "   break 'brew install prsm' for everyone."
  exit 1
fi
echo "✅ Formula has real sha256 values."

# Clone tap repo to scratch dir
SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH"' EXIT

echo "Cloning $TAP_REPO ..."
GIT_SSH_COMMAND="ssh -i ~/.ssh/id_github_gemini" git clone --depth 1 "$TAP_REPO" "$SCRATCH"

# Copy formula in
mkdir -p "$SCRATCH/Formula"
cp "$LOCAL_FORMULA" "$SCRATCH/Formula/prsm.rb"

# Commit + push
cd "$SCRATCH"
if git diff --quiet --exit-code; then
  echo "ℹ️  Tap formula already up-to-date — nothing to push."
  exit 0
fi

git add Formula/prsm.rb
GIT_SSH_COMMAND="ssh -i ~/.ssh/id_github_gemini" git commit -m "chore: bump prsm to ${VERSION}"
echo "Pushing to $TAP_REPO ..."
GIT_SSH_COMMAND="ssh -i ~/.ssh/id_github_gemini" git push origin main

echo
echo "✅ Synced. Users can now install via:"
echo "   brew tap prsm-network/tap"
echo "   brew install prsm"
