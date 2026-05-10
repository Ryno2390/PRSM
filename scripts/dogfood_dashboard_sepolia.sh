#!/bin/bash
# Dogfood the operator dashboard suite (sprints 76-115).
# Targets Base Sepolia testnet — gas is free.
#
# Prerequisites:
#   - contracts/.env has PRIVATE_KEY = Sepolia testnet wallet
#     (with some Sepolia ETH for gas)
#   - WEBHOOK_URL set below to your webhook.site URL
#
# Usage:
#   bash scripts/dogfood_dashboard_sepolia.sh
#   (then in another terminal: prsm node earnings, etc.)
set -euo pipefail

# Load Sepolia private key from contracts/.env (PRIVATE_KEY var)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$REPO/contracts/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$REPO/contracts/.env"
    set +a
fi

# Normalize 0x prefix on PRIVATE_KEY (Hardhat is strict;
# FTNS_WALLET_PRIVATE_KEY here is forgiving but stay consistent)
if [ -n "${PRIVATE_KEY:-}" ] && [[ "$PRIVATE_KEY" != 0x* ]]; then
    PRIVATE_KEY="0x$PRIVATE_KEY"
fi

# ── Network selection (Base Sepolia) ──────────────────────────────
export PRSM_NETWORK=testnet
# Public Base Sepolia endpoints rotate by reliability. Prefer
# publicnode (more lax rate limit) over base.org (frequently 503).
# If you have an Alchemy/Infura key, set BASE_SEPOLIA_RPC_URL
# externally to override.
export PRSM_BASE_RPC_URL="${BASE_SEPOLIA_RPC_URL:-https://base-sepolia-rpc.publicnode.com}"

# ── Wallet wiring ─────────────────────────────────────────────────
export FTNS_WALLET_PRIVATE_KEY="${PRIVATE_KEY:?PRIVATE_KEY missing in contracts/.env}"

# ── Phase 7-storage / Phase 8 contract addresses (per networks.py) ─
export PRSM_STORAGE_SLASHING_ADDRESS="0x2ba1B361d2AD49f15F1131762fA3512d7824EB06"
export PRSM_COMPENSATION_DISTRIBUTOR_ADDRESS="0xFd730f8E513eD184F255cb1a62791e711B2e81b9"

# ── Watcher activation (sprints 78, 79, 86) ───────────────────────
export PRSM_STORAGE_SLASHING_WATCHER_ENABLED=1
export PRSM_COMPENSATION_DISTRIBUTOR_WATCHER_ENABLED=1
export PRSM_STORAGE_SLASHING_WATCHER_POLL_SECONDS=10
export PRSM_COMPENSATION_DISTRIBUTOR_WATCHER_POLL_SECONDS=10

# ── Webhook delivery (sprints 76, 85, 88, 89) ─────────────────────
export PRSM_WEBHOOK_URL="https://webhook.site/6fe9a4a6-999d-4c97-8dd3-0e484632a241"
export PRSM_DAEMON_WATCHDOG_INTERVAL_SEC=10

# ── Persistence dirs (sprints 91, 92) ─────────────────────────────
mkdir -p ~/.prsm/dogfood/{slash,heartbeat,distribution}
export PRSM_SLASH_EVENT_LOG_DIR=~/.prsm/dogfood/slash
export PRSM_HEARTBEAT_LOG_DIR=~/.prsm/dogfood/heartbeat
export PRSM_DISTRIBUTION_LOG_DIR=~/.prsm/dogfood/distribution

# ── Echo config (mask private key) ────────────────────────────────
echo "─── Dogfood config ──────────────────────────────────"
echo "  Network:        $PRSM_NETWORK ($PRSM_BASE_RPC_URL)"
echo "  Wallet:         ${FTNS_WALLET_PRIVATE_KEY:0:6}…${FTNS_WALLET_PRIVATE_KEY: -4}"
echo "  Slashing:       $PRSM_STORAGE_SLASHING_ADDRESS"
echo "  Compensation:   $PRSM_COMPENSATION_DISTRIBUTOR_ADDRESS"
echo "  Webhook:        $PRSM_WEBHOOK_URL"
echo "─────────────────────────────────────────────────────"
echo

# Run from REPO SOURCE (not pipx) so we get:
#   (a) the dylib at libp2p/build/libprsm_p2p_darwin_arm64.dylib
#   (b) all 40 of today's sprints (76-115) — the new CLI commands
#       like `prsm node earnings`, trigger-heartbeat, etc.
# pipx-installed prsm-network on PyPI lags behind by weeks.
cd "$REPO"
exec python3 -m prsm.cli node start
