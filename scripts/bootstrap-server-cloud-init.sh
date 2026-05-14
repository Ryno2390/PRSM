#!/bin/bash
#
# PRSM Bootstrap Server — cloud-init userdata
#
# Paste this into the cloud provider's "User data" field
# during instance launch (AWS / OCI / DO / Azure / GCP all
# support it via cloud-init). The instance bootstraps
# itself on first boot — no SSH-in required.
#
# REQUIRED BEFORE BOOT:
#   - DNS A record for $PRSM_HOSTNAME must already resolve
#     to the instance's public IP. The script waits up to
#     10 min for DNS to propagate before invoking certbot;
#     if it doesn't resolve in that window, certbot fails
#     and you'll need to SSH in to finish manually.
#
# CUSTOMIZE BEFORE PASTING:
#   Set the two placeholder values at the top:
#     PRSM_HOSTNAME — e.g., bootstrap-eu.prsm-network.com
#     ADMIN_EMAIL   — Let's Encrypt registration email
#
# WATCH PROGRESS:
#   ssh -i ~/.ssh/oci_prsm_bootstrap ubuntu@<PUBLIC_IP>
#   sudo tail -f /var/log/cloud-init-output.log
#
# Closes the per-bootstrap-host manual-walkthrough friction
# left over from sprint 382 (OCI deployment guide). When
# Frankfurt's retry loop eventually succeeds, the instance
# auto-deploys without operator intervention.

set -euxo pipefail

# ────── CUSTOMIZE ──────
PRSM_HOSTNAME="CHANGEME.prsm-network.com"
ADMIN_EMAIL="foundation-ops@prsm-network.com"
# Sprint 398: surface the deploy region so /health reports
# accurately. Examples: us-east-1, eu-central-1, ap-
# northeast-1. Leave as REGION_UNSET to fail-loud rather
# than silently fall back to the BootstrapConfig default.
PRSM_REGION="REGION_UNSET"
# ───────────────────────

PRSM_REPO="https://github.com/prsm-network/PRSM.git"
INSTALL_DIR="/opt/prsm"
LOG_FILE="/var/log/prsm-bootstrap.log"
PEER_DB_DIR="/var/lib/prsm-bootstrap"

# ── 1. System prep ──────────────────────────────────────
timedatectl set-timezone UTC
hostnamectl set-hostname "${PRSM_HOSTNAME%%.*}"

export DEBIAN_FRONTEND=noninteractive

# Wait for unattended-upgrades to release apt lock (common
# on first-boot Ubuntu cloud images — apt is locked for
# the first 60-90s while the kernel package check runs).
for i in {1..30}; do
    if ! fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; then
        break
    fi
    sleep 5
done

apt-get update -qq
apt-get install -y -qq \
    python3 python3-venv python3-pip \
    git certbot ufw

# ── 2. Firewall ─────────────────────────────────────────
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 8765/tcp
# Sprint 398: port 8000 carries the bootstrap server's
# HTTP API + /metrics + /health/detailed surfaces. Without
# this rule the sprint-388-396 observability work is
# unreachable from the public internet (dogfood-discovered
# 2026-05-14 on the freshly-deployed bootstrap-eu droplet).
ufw allow 8000/tcp
ufw --force enable

# ── 3. PRSM install ─────────────────────────────────────
mkdir -p "$INSTALL_DIR"
chown ubuntu:ubuntu "$INSTALL_DIR"
sudo -u ubuntu git clone --depth 1 "$PRSM_REPO" "$INSTALL_DIR"
sudo -u ubuntu python3 -m venv "$INSTALL_DIR/.venv"
sudo -u ubuntu "$INSTALL_DIR/.venv/bin/pip" install --upgrade pip --quiet
sudo -u ubuntu "$INSTALL_DIR/.venv/bin/pip" install -e "$INSTALL_DIR" --quiet

# ── 4. Wait for DNS to resolve to OUR public IP ─────────
# Required for certbot's HTTP-01 challenge. If the operator
# hasn't created the A record yet, this waits up to 10 min
# before giving up and letting certbot fail.
PUBLIC_IP=$(curl -s --max-time 10 http://checkip.amazonaws.com || \
            curl -s --max-time 10 https://ifconfig.me || \
            echo "unknown")
echo "Instance public IP: $PUBLIC_IP"

DNS_RESOLVED=0
for i in {1..120}; do
    RESOLVED=$(dig +short "$PRSM_HOSTNAME" | tail -n1 || echo "")
    if [ "$RESOLVED" = "$PUBLIC_IP" ]; then
        echo "DNS resolves correctly: $PRSM_HOSTNAME -> $PUBLIC_IP"
        DNS_RESOLVED=1
        break
    fi
    echo "Waiting for DNS ($i/120) — currently resolves: '$RESOLVED' vs expected '$PUBLIC_IP'"
    sleep 5
done

if [ "$DNS_RESOLVED" = "0" ]; then
    echo "WARNING: DNS did not resolve to public IP within 10min — skipping certbot."
    echo "After fixing DNS, run on the instance:"
    echo "  sudo certbot certonly --standalone -d $PRSM_HOSTNAME --agree-tos -m $ADMIN_EMAIL --non-interactive"
    echo "  sudo systemctl restart prsm-bootstrap"
fi

# ── 5. Let's Encrypt cert ───────────────────────────────
if [ "$DNS_RESOLVED" = "1" ]; then
    certbot certonly --standalone \
        -d "$PRSM_HOSTNAME" \
        --agree-tos \
        -m "$ADMIN_EMAIL" \
        --non-interactive

    # Cert perms — let `ubuntu` read privkey while keeping
    # root ownership for the cert directory tree.
    chmod 750 /etc/letsencrypt/live /etc/letsencrypt/archive
    chown -R root:ubuntu /etc/letsencrypt/live /etc/letsencrypt/archive
    chmod 640 /etc/letsencrypt/archive/"$PRSM_HOSTNAME"/*.pem
fi

# ── 6. Peer DB directory (sprint 383) ───────────────────
mkdir -p "$PEER_DB_DIR"
chown ubuntu:ubuntu "$PEER_DB_DIR"

# ── 7. Env file ─────────────────────────────────────────
mkdir -p /etc/prsm
cat > /etc/prsm/bootstrap-server.env <<EOF
PRSM_BOOTSTRAP_HOST=0.0.0.0
PRSM_BOOTSTRAP_PORT=8765
PRSM_SSL_ENABLED=true
PRSM_SSL_CERT_PATH=/etc/letsencrypt/live/$PRSM_HOSTNAME/fullchain.pem
PRSM_SSL_KEY_PATH=/etc/letsencrypt/live/$PRSM_HOSTNAME/privkey.pem
PRSM_DOMAIN=$PRSM_HOSTNAME
PRSM_EXTERNAL_IP=$PUBLIC_IP
PRSM_MAX_PEERS=500
PRSM_PEER_TIMEOUT=300
PRSM_HEARTBEAT_INTERVAL=30
PRSM_PEER_DB_PATH=$PEER_DB_DIR/peers.db
PRSM_REGION=$PRSM_REGION
EOF
chmod 640 /etc/prsm/bootstrap-server.env
chown root:ubuntu /etc/prsm/bootstrap-server.env

# ── 8. systemd unit ─────────────────────────────────────
cat > /etc/systemd/system/prsm-bootstrap.service <<EOF
[Unit]
Description=PRSM Bootstrap Server ($PRSM_HOSTNAME)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=$INSTALL_DIR
EnvironmentFile=/etc/prsm/bootstrap-server.env
ExecStart=$INSTALL_DIR/.venv/bin/python -m prsm.bootstrap.server
Restart=on-failure
RestartSec=5s
StandardOutput=append:$LOG_FILE
StandardError=append:$LOG_FILE
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log $PEER_DB_DIR
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

touch "$LOG_FILE"
chown ubuntu:ubuntu "$LOG_FILE"

systemctl daemon-reload
if [ "$DNS_RESOLVED" = "1" ]; then
    systemctl enable --now prsm-bootstrap
    sleep 3
    systemctl status prsm-bootstrap --no-pager | head -10
fi

echo "=== PRSM bootstrap server cloud-init complete ==="
echo "Hostname:   $PRSM_HOSTNAME"
echo "Public IP:  $PUBLIC_IP"
echo "DNS OK:     $DNS_RESOLVED"
echo "Log:        $LOG_FILE"
echo "Service:    systemctl status prsm-bootstrap"
