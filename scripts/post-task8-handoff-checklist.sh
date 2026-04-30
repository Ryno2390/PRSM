#!/usr/bin/env bash
#
# Phase 1.3 Task 8 — post-deploy ops-integration handoff checklist.
#
# Wires F3 from docs/2026-04-30-multisig-action-plan-engineering-audit.md
# into executable infrastructure. After Task 8 deploys ProvenanceRegistry +
# RoyaltyDistributor on Base mainnet, the new addresses must propagate
# into:
#   - production-config files (any hard-coded testnet addresses)
#   - Forta detection bots (re-target from sepolia to mainnet)
#   - pause-tx templates (mainnet RoyaltyDistributor swap-in)
#   - cumulative audit-prep doc (deploy-address section refresh + retag)
#   - MEMORY.md (project-memory entry recording mainnet addresses)
#
# This script does NOT auto-edit. It reads the mainnet manifest, finds
# every place in the repo that references the OLD (sepolia/placeholder)
# addresses, and prints a structured findings list the operator can
# walk through manually + commit/PR.
#
# Required env var:
#   MAINNET_MANIFEST  - path to provenance-base-*.json (mainnet deploy)
#
# Usage:
#   MAINNET_MANIFEST=contracts/deployments/provenance-base-1234.json \
#     ./scripts/post-task8-handoff-checklist.sh
#
# Exit codes:
#   0 = checklist generated; manual review pending
#   1 = manifest read/parse error

set -uo pipefail

MAINNET_MANIFEST="${MAINNET_MANIFEST:-}"
if [[ -z "${MAINNET_MANIFEST}" ]] || [[ ! -f "${MAINNET_MANIFEST}" ]]; then
  echo "❌ MAINNET_MANIFEST env var must point at the mainnet provenance manifest" >&2
  echo "   e.g. contracts/deployments/provenance-base-<ts>.json" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ── Parse mainnet manifest ──────────────────────────────────────────────
extract() {
  # $1 = JSON key path, e.g. ".contracts.ProvenanceRegistry"
  python3 -c "
import json, sys
m = json.load(open('${MAINNET_MANIFEST}'))
keys = '${1}'.lstrip('.').split('.')
v = m
for k in keys:
    v = v[k]
print(v)
" 2>/dev/null
}

NEW_NETWORK="$(extract '.network')"
NEW_CHAIN_ID="$(extract '.chainId')"
NEW_REGISTRY="$(extract '.contracts.ProvenanceRegistry')"
NEW_DISTRIBUTOR="$(extract '.contracts.RoyaltyDistributor')"
NEW_FTNS="$(extract '.contracts.FTNSToken')"
NEW_TREASURY="$(extract '.contracts.NetworkTreasury')"
NEW_DEPLOYER="$(extract '.deployer')"
NEW_TIMESTAMP="$(extract '.timestamp')"

if [[ -z "${NEW_REGISTRY}" ]] || [[ -z "${NEW_DISTRIBUTOR}" ]]; then
  echo "❌ could not extract addresses from manifest ${MAINNET_MANIFEST}" >&2
  exit 1
fi

if [[ "${NEW_NETWORK}" != "base" ]]; then
  echo "⚠️  manifest network=${NEW_NETWORK} (not 'base'); proceeding but verify this is the mainnet deploy" >&2
fi

# ── Find latest sepolia manifest for OLD addresses ─────────────────────
OLD_MANIFEST="$(ls -1t contracts/deployments/provenance-base-sepolia-*.json 2>/dev/null | head -1 || true)"
OLD_REGISTRY=""
OLD_DISTRIBUTOR=""
if [[ -n "${OLD_MANIFEST}" ]]; then
  OLD_REGISTRY="$(MAINNET_MANIFEST="${OLD_MANIFEST}" extract '.contracts.ProvenanceRegistry')"
  OLD_DISTRIBUTOR="$(MAINNET_MANIFEST="${OLD_MANIFEST}" extract '.contracts.RoyaltyDistributor')"
fi

# ── Print summary header ────────────────────────────────────────────────
cat <<EOF

=== Phase 1.3 Task 8 post-deploy ops-integration handoff ===

Mainnet manifest:    ${MAINNET_MANIFEST}
Network:             ${NEW_NETWORK} (chainId=${NEW_CHAIN_ID})
Deployer:            ${NEW_DEPLOYER}
Timestamp:           ${NEW_TIMESTAMP}

NEW addresses (mainnet):
  ProvenanceRegistry: ${NEW_REGISTRY}
  RoyaltyDistributor: ${NEW_DISTRIBUTOR}
  FTNSToken:          ${NEW_FTNS}
  NetworkTreasury:    ${NEW_TREASURY}

EOF

if [[ -n "${OLD_MANIFEST}" ]]; then
  cat <<EOF
OLD addresses (base-sepolia, from ${OLD_MANIFEST}):
  ProvenanceRegistry: ${OLD_REGISTRY}
  RoyaltyDistributor: ${OLD_DISTRIBUTOR}

EOF
fi

# ── Section 1: code/config references to old addresses ────────────────
echo "──── [1] Repo references to OLD sepolia addresses ────"
echo "(each match needs review — replace with mainnet address per task)"
echo

count=0
if [[ -n "${OLD_REGISTRY}" ]]; then
  echo "  Searching for OLD ProvenanceRegistry (${OLD_REGISTRY})…"
  matches="$(grep -rin "${OLD_REGISTRY}" \
    --include="*.py" --include="*.md" --include="*.json" \
    --include="*.ts" --include="*.js" --include="*.yml" --include="*.yaml" \
    --exclude-dir=node_modules --exclude-dir=artifacts \
    --exclude-dir=.git --exclude-dir=cache 2>/dev/null || true)"
  if [[ -n "${matches}" ]]; then
    echo "${matches}" | sed 's/^/    /'
    count=$((count + $(echo "${matches}" | wc -l | tr -d ' ')))
  else
    echo "    (none found — already clean)"
  fi
fi

if [[ -n "${OLD_DISTRIBUTOR}" ]]; then
  echo
  echo "  Searching for OLD RoyaltyDistributor (${OLD_DISTRIBUTOR})…"
  matches="$(grep -rin "${OLD_DISTRIBUTOR}" \
    --include="*.py" --include="*.md" --include="*.json" \
    --include="*.ts" --include="*.js" --include="*.yml" --include="*.yaml" \
    --exclude-dir=node_modules --exclude-dir=artifacts \
    --exclude-dir=.git --exclude-dir=cache 2>/dev/null || true)"
  if [[ -n "${matches}" ]]; then
    echo "${matches}" | sed 's/^/    /'
    count=$((count + $(echo "${matches}" | wc -l | tr -d ' ')))
  else
    echo "    (none found — already clean)"
  fi
fi
echo
echo "  Total OLD-address references: ${count}"

# ── Section 2: known integration touchpoints ──────────────────────────
echo
echo "──── [2] Known integration touchpoints ────"

check_path() {
  local label="$1"
  local path="$2"
  if [[ -e "${path}" ]]; then
    echo "  ✓ EXISTS — ${label}"
    echo "    ${path}"
  else
    echo "  ⊗ ABSENT  — ${label}"
    echo "    ${path}"
  fi
}

# Forta bot configs
echo
echo "  [2a] Forta detection bots:"
FORTA_DIR="$(find . -type d -name "forta*" -not -path "*/node_modules/*" 2>/dev/null | head -1 || true)"
if [[ -n "${FORTA_DIR}" ]]; then
  echo "    Found: ${FORTA_DIR}"
  echo "    → re-target Forta bot config from sepolia to mainnet addresses"
else
  # Search by content marker
  FORTA_FILES="$(grep -rln "forta\|Forta" \
    --include="*.py" --include="*.json" --include="*.yml" \
    --exclude-dir=node_modules --exclude-dir=.git 2>/dev/null \
    | grep -v "test" | head -5 || true)"
  if [[ -n "${FORTA_FILES}" ]]; then
    echo "    Forta-mentioning files:"
    echo "${FORTA_FILES}" | sed 's/^/      /'
  else
    echo "    (no Forta bot config detected — operator confirms with ops-monitoring runbook)"
  fi
fi

# Pause-tx templates (task #138 done)
echo
echo "  [2b] Pause-tx templates:"
PAUSE_TPL="$(find . -type f \( -name "pause-*.md" -o -name "pause-*.json" -o -name "*pause*template*" \) -not -path "*/node_modules/*" -not -path "*/.git/*" 2>/dev/null | head -3 || true)"
if [[ -n "${PAUSE_TPL}" ]]; then
  echo "${PAUSE_TPL}" | sed 's/^/    /'
  echo "    → swap RoyaltyDistributor address for new mainnet ${NEW_DISTRIBUTOR}"
else
  echo "    (no pause-tx template files matched; check repo task #138)"
fi

# Cumulative audit-prep
echo
check_path "[2c] Cumulative audit-prep doc (refresh deploy-address section + retag)" \
  "docs/2026-04-27-cumulative-audit-prep.md"

# MEMORY.md project memory
echo
check_path "[2d] MEMORY.md (add project-memory entry for mainnet deploy)" \
  "/Users/ryneschultz/.claude/projects/-Users-ryneschultz-Documents-GitHub-PRSM/memory/MEMORY.md"

# Production-config / .env templates
echo
echo "  [2e] Production .env / config files:"
ENV_TEMPLATES="$(find . -type f \( -name "*.env.example" -o -name "*.env.template" -o -name "config.production.*" \) -not -path "*/node_modules/*" -not -path "*/.git/*" 2>/dev/null | head -5 || true)"
if [[ -n "${ENV_TEMPLATES}" ]]; then
  echo "${ENV_TEMPLATES}" | sed 's/^/    /'
  echo "    → set ProvenanceRegistry / RoyaltyDistributor addresses if applicable"
else
  echo "    (no .env.example / config.production.* templates found)"
fi

# README / public docs
echo
check_path "[2f] README.md (update if deploy addresses are referenced)" \
  "README.md"
check_path "[2g] CHANGELOG.md (add Phase 1.3 Task 8 mainnet entry)" \
  "CHANGELOG.md"

# ── Section 3: PR body draft ──────────────────────────────────────────
PR_PATH="/tmp/task8-mainnet-handoff-pr-body-$(date +%s).md"
cat > "${PR_PATH}" <<EOF
# Phase 1.3 Task 8 — Mainnet deploy handoff

ProvenanceRegistry + RoyaltyDistributor deployed on Base mainnet ${NEW_TIMESTAMP}.

## Mainnet addresses
- **ProvenanceRegistry:** \`${NEW_REGISTRY}\`
- **RoyaltyDistributor:** \`${NEW_DISTRIBUTOR}\`
- **NetworkTreasury (Foundation Safe):** \`${NEW_TREASURY}\`
- **FTNSToken (canonical):** \`${NEW_FTNS}\`

Manifest: \`${MAINNET_MANIFEST}\`
Deployer (now retired): \`${NEW_DEPLOYER}\`

## What this PR does
- Swap sepolia testnet addresses for mainnet across:
  - production-config files
  - Forta detection bot config
  - pause-tx templates
  - audit-prep doc (deploy-address section)
  - README + CHANGELOG
- Refresh \`docs/2026-04-27-cumulative-audit-prep.md\` and re-tag (\`cumulative-audit-prep-mainnet-task8-$(date +%Y%m%d)\`).
- Save project-memory entry for this deploy.

## Verification
\`\`\`bash
PROVENANCE_MANIFEST=${MAINNET_MANIFEST} \\
  npx hardhat run scripts/verify-provenance-deployment.js --network base
\`\`\`
Expect: ✅ All on-chain state matches manifest.

## Out of scope
- FTNSToken DEFAULT_ADMIN_ROLE handoff (separate decision; not part of Task 8).
- Audit-bundle / Phase 8 emission / Phase 7-storage stack (post-external-audit ceremony; see \`docs/2026-04-30-post-audit-deploy-ceremony-runbook.md\`).
EOF

echo
echo "──── [3] PR body draft ────"
echo "  Saved: ${PR_PATH}"
echo "  → use \`gh pr create --body-file ${PR_PATH}\` after pushing the address-swap branch"

# ── Section 4: memory entry stub ──────────────────────────────────────
MEM_PATH="/tmp/task8-mainnet-deploy-memory-stub-$(date +%s).md"
cat > "${MEM_PATH}" <<EOF
---
name: Phase 1.3 Task 8 mainnet deploy — $(date +%Y-%m-%d)
description: ProvenanceRegistry + RoyaltyDistributor live on Base mainnet at known addresses; Foundation Safe wired as NETWORK_TREASURY; deployer key retired post-sweep
type: project
---
# Phase 1.3 Task 8 mainnet deploy — $(date +%Y-%m-%d)

Deploy completed ${NEW_TIMESTAMP} on Base mainnet (chainId 8453) via
the disposable deployer key per Multi-Sig_Action_Plan.md §5-6.

**Mainnet addresses:**
- ProvenanceRegistry: \`${NEW_REGISTRY}\`
- RoyaltyDistributor: \`${NEW_DISTRIBUTOR}\`
- NetworkTreasury (Foundation 2-of-3 Safe): \`${NEW_TREASURY}\`
- FTNSToken (canonical, pre-existing): \`${NEW_FTNS}\`

**Deployer (now retired):** \`${NEW_DEPLOYER}\`

**Why this matters:** the 2% network fee + 8% creator royalty router
is now live on mainnet. Every on-chain royalty payment flow goes
through these contracts; ops integration must point at the new
addresses (Forta bots, pause-tx templates, audit-prep doc, .env
defaults).

**How to apply:** when reviewing payment-flow code or asking about
mainnet addresses, treat these as canonical. The sepolia testnet
addresses (\`${OLD_REGISTRY:-N/A}\` / \`${OLD_DISTRIBUTOR:-N/A}\`) are
historical only.

**Honest scope:** this is Phase 1.3 Task 8 only — Provenance + Royalty.
The audit-bundle stack (audit-bundle + Phase 8 emission + Phase
7-storage) is a SEPARATE, post-external-audit ceremony tracked under
Phase 7 Task 9 + Phase 7.1 Task 9.
EOF

echo
echo "──── [4] Project-memory entry stub ────"
echo "  Saved: ${MEM_PATH}"
echo "  → review + move to ~/.claude/projects/-Users-ryneschultz-Documents-GitHub-PRSM/memory/"
echo "    project_phase1_3_task8_mainnet_deploy_$(date +%Y_%m_%d).md"
echo "  → also add a one-line entry to MEMORY.md index"

# ── Final summary ─────────────────────────────────────────────────────
echo
echo "=== Handoff checklist generated ==="
echo
echo "Next manual steps for the operator:"
echo "  1. Walk the Section [1] OLD-address references (${count} total)."
echo "     Replace each with the new mainnet address. Open a PR."
echo "  2. Update Section [2] integration touchpoints (Forta, pause-tx,"
echo "     audit-prep, MEMORY.md, .env, README, CHANGELOG)."
echo "  3. Use the PR body draft at ${PR_PATH}."
echo "  4. Move the memory stub at ${MEM_PATH} into the memory directory."
echo "  5. Tag: \`git tag phase1.3-task8-mainnet-handoff-\$(date +%Y%m%d)\`"
echo
exit 0
