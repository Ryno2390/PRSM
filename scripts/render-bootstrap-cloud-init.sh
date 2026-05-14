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
#     > /tmp/cloud-init-eu.sh
#
# Then: copy /tmp/cloud-init-eu.sh contents into the
# launch wizard's User Data textbox, OR pass via OCI CLI
# --user-data-file flag in the retry loop.

set -euo pipefail

HOSTNAME="${1:-}"
EMAIL="${2:-foundation-ops@prsm-network.com}"

if [ -z "$HOSTNAME" ]; then
    echo "usage: $0 <HOSTNAME> [ADMIN_EMAIL]" >&2
    echo "  e.g.: $0 bootstrap-eu.prsm-network.com" >&2
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
    "$TEMPLATE"
