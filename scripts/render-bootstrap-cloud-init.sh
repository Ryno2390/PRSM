#!/usr/bin/env bash
#
# Helper: render scripts/bootstrap-server-cloud-init.sh
# with the hostname + email pre-filled, ready to paste
# into a cloud provider's User Data field.
#
# Usage:
#   bash scripts/render-bootstrap-cloud-init.sh \
#     bootstrap-eu.prsm-network.com \
#     foundation-ops@prsm-network.com \
#     eu-central-1 \
#     > /tmp/cloud-init-eu.sh
#
# Then: copy /tmp/cloud-init-eu.sh contents into the
# launch wizard's User Data textbox, OR pass via OCI CLI
# --user-data-file flag in the retry loop.
#
# Region (arg 3) is optional but recommended; without it
# the rendered output keeps the REGION_UNSET placeholder
# so /health reports something noticeable instead of
# silently falling back to the BootstrapConfig default
# (us-east-1) — added sprint 398 after dogfood found
# bootstrap-eu in AWS Frankfurt reporting us-east-1.

set -euo pipefail

HOSTNAME="${1:-}"
EMAIL="${2:-foundation-ops@prsm-network.com}"
REGION="${3:-REGION_UNSET}"

if [ -z "$HOSTNAME" ]; then
    echo "usage: $0 <HOSTNAME> [ADMIN_EMAIL] [REGION]" >&2
    echo "  e.g.: $0 bootstrap-eu.prsm-network.com foundation-ops@prsm-network.com eu-central-1" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEMPLATE="$REPO_ROOT/scripts/bootstrap-server-cloud-init.sh"

if [ ! -f "$TEMPLATE" ]; then
    echo "template not found: $TEMPLATE" >&2
    exit 1
fi

sed \
    -e "s|^PRSM_HOSTNAME=\".*\"|PRSM_HOSTNAME=\"$HOSTNAME\"|" \
    -e "s|^ADMIN_EMAIL=\".*\"|ADMIN_EMAIL=\"$EMAIL\"|" \
    -e "s|^PRSM_REGION=\".*\"|PRSM_REGION=\"$REGION\"|" \
    "$TEMPLATE"
