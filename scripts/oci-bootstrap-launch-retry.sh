#!/usr/bin/env bash
#
# OCI Always Free A1 launch retry loop — keeps trying until
# Frankfurt frees up capacity. Closes the §7.29 SPOF gap
# operationally (sprint 382 deployment guide).
#
# Usage:  bash scripts/oci-bootstrap-launch-retry.sh
# Watch:  tail -f /tmp/oci-launch.log
# Stop:   Ctrl-C (or kill the bash PID)

set -u
export SUPPRESS_LABEL_WARNING=True

# Resources we provisioned earlier
COMPARTMENT="ocid1.compartment.oc1..aaaaaaaaklt7pkcw635bgq4oi6zeel4mhnuugjxzsyaldusxcjcn6rj66pyq"
SUBNET="ocid1.subnet.oc1.eu-frankfurt-1.aaaaaaaab6cadl5ipnv5335ytvmgwjt3umsywdspnz4arf5mjeg5kzzhtnja"
IMAGE="ocid1.image.oc1.eu-frankfurt-1.aaaaaaaaiclwz5vfkbnxddiyffjrycraavdu73axaswdsav2nmf3pxbnxv2a"
SSH_PUB_KEY_FILE="$HOME/.ssh/oci_prsm_bootstrap.pub"

# Sprint 384 — pre-render cloud-init userdata so when
# the launch finally succeeds, the instance auto-deploys
# the bootstrap server without operator intervention.
# (Pre-fix, we'd SSH in + run ~20 manual commands. Post-
# fix, the OCI launch carries userdata that does it all.)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
USERDATA_FILE="/tmp/oci-bootstrap-eu-userdata.sh"
bash "$SCRIPT_DIR/render-bootstrap-cloud-init.sh" \
    "bootstrap-eu.prsm-network.com" \
    > "$USERDATA_FILE"
echo "rendered userdata to $USERDATA_FILE ($(wc -l < "$USERDATA_FILE") lines)"

ADS=(
  "buef:EU-FRANKFURT-1-AD-3"
  "buef:EU-FRANKFURT-1-AD-2"
  "buef:EU-FRANKFURT-1-AD-1"
)

LOG=/tmp/oci-launch.log
echo "$(date) — retry loop starting" | tee -a "$LOG"

ATTEMPT=0
while true; do
  for AD in "${ADS[@]}"; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "$(date) — attempt #$ATTEMPT in $AD" | tee -a "$LOG"
    OUTPUT=$(oci compute instance launch \
      --compartment-id "$COMPARTMENT" \
      --availability-domain "$AD" \
      --shape "VM.Standard.A1.Flex" \
      --shape-config '{"ocpus":1,"memoryInGBs":6}' \
      --image-id "$IMAGE" \
      --subnet-id "$SUBNET" \
      --assign-public-ip true \
      --display-name "prsm-bootstrap-eu" \
      --ssh-authorized-keys-file "$SSH_PUB_KEY_FILE" \
      --user-data-file "$USERDATA_FILE" \
      --wait-for-state RUNNING \
      --max-wait-seconds 600 \
      2>&1)
    EXIT=$?
    echo "$OUTPUT" >> "$LOG"
    if [[ $EXIT -eq 0 ]]; then
      echo "" | tee -a "$LOG"
      echo "$(date) — SUCCESS in $AD" | tee -a "$LOG"
      # Extract + print the public IP
      INSTANCE_ID=$(echo "$OUTPUT" | python3 -c "
import sys, json, re
text = sys.stdin.read()
# Strip warnings, find first JSON object
m = re.search(r'\{.*\}', text, re.DOTALL)
if m:
    d = json.loads(m.group(0))['data']
    print(d['id'])
" 2>/dev/null)
      if [[ -n "$INSTANCE_ID" ]]; then
        echo "INSTANCE_ID=$INSTANCE_ID" | tee -a "$LOG"
        PUB_IP=$(oci compute instance list-vnics \
          --instance-id "$INSTANCE_ID" \
          --raw-output --query 'data[0]."public-ip"' 2>/dev/null)
        echo "PUBLIC_IP=$PUB_IP" | tee -a "$LOG"
      fi
      # macOS notification + audible bell
      osascript -e 'display notification "EU bootstrap instance is RUNNING" with title "PRSM OCI launch" sound name "Glass"' 2>/dev/null || true
      printf '\a'
      exit 0
    fi
    # Capacity-out error → fall through to next AD or backoff
    if echo "$OUTPUT" | grep -q "Out of host capacity"; then
      echo "$(date) — out of capacity, trying next AD" | tee -a "$LOG"
    else
      echo "$(date) — non-capacity failure (see log) — continuing retry" | tee -a "$LOG"
    fi
  done
  echo "$(date) — all 3 ADs out of capacity; sleeping 90s before next round" | tee -a "$LOG"
  sleep 90
done
