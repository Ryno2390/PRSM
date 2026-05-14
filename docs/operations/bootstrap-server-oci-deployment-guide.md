# PRSM Bootstrap Server — OCI Always Free Deployment Guide

**Status:** Operator runbook for provisioning EU + APAC bootstrap servers on Oracle Cloud Infrastructure (OCI) Always Free Tier. Closes the §7.29 honest-scope item: "EU+APAC droplets aspirational" (the sprint-375 code path exists; this guide brings up the actual hosts).

**Audience:** Foundation operations (you). Sole-founder scope.

**Cost:** $0/month forever within OCI Always Free Tier limits (4 ARM Ampere A1 cores + 24 GB RAM total OR 2 AMD VM.Standard.E2.1.Micro instances; we use 1 ARM instance per region).

**Time:** ~60–90 minutes per region (90 the first time, 60 the second once you've done it once).

**Outcome:** `wss://bootstrap-eu.prsm-network.com:8765` and `wss://bootstrap-apac.prsm-network.com:8765` accept WebSocket peer-registration traffic, surfaced through the `Libp2pDiscovery.bootstrap_fallback_nodes` code path that's been waiting for them since sprint 375.

---

## 1. Pre-flight

### 1.1 Accounts you need

- **OCI tenancy** (free tier — requires credit card for identity verification, never billed within limits)
- **DNS control** for `prsm-network.com` (Cloudflare / Namecheap / wherever bootstrap1's DNS lives — verify by `dig bootstrap1.prsm-network.com NS` first)
- **GitHub SSH key** on the OCI instance, OR (simpler) `git clone https://github.com/...` over HTTPS

### 1.2 Regions to pick

OCI Always Free has these regions that match the target geography:

| Target | OCI Home Region recommendation |
|---|---|
| EU | Frankfurt (eu-frankfurt-1) or Amsterdam (eu-amsterdam-1) |
| APAC | Tokyo (ap-tokyo-1) or Singapore (ap-singapore-1) |

**Important caveat:** OCI Always Free Tier is per-tenancy, not per-region. You can only mark ONE home region during signup (this is permanent). To get instances in BOTH EU and APAC you need EITHER:
- (a) Two separate OCI accounts (each free-tier independent), OR
- (b) One tenancy with home in EU; Singapore instances will then be "paid-tier but free-eligible" subject to the global 4-core ARM Ampere A1 + 24 GB RAM Always Free pool which spans regions.

Option (b) is simpler. Pick your home region as the one that matters more (probably EU given the European compute scarcity discussion in Vision §11), then provision the APAC instance from the same tenancy.

### 1.3 Reference: bootstrap1 (the precedent)

The existing US bootstrap is at `wss://bootstrap1.prsm-network.com:8765`, hosted on a DigitalOcean Droplet. Operationally, EU + APAC mirror that shape: same `prsm.bootstrap.server` process, same port 8765, same TLS posture. The cloud provider differs (OCI vs DO) but the wire protocol does not — operators reach all three the same way.

---

## 2. OCI: provision the instance

### 2.1 Sign up + select home region

1. Go to `https://signup.cloud.oracle.com`
2. Fill the signup form; **select Frankfurt or Amsterdam as home region** (permanent choice)
3. Provide a payment card for identity verification — **OCI does not charge within Always Free limits**, but the card is mandatory
4. Wait for the activation email (~30–60 min)

### 2.2 Create a compartment (optional but tidy)

Compartments are OCI's namespace mechanism. Skip and use the root tenancy if you want to move fast.

If you want it tidy:
1. Identity → Compartments → Create
2. Name: `prsm-bootstrap`
3. Description: "PRSM EU + APAC bootstrap servers"

### 2.3 Generate an SSH key locally

```bash
ssh-keygen -t ed25519 -C "prsm-bootstrap-oci" \
  -f ~/.ssh/oci_prsm_bootstrap
# No passphrase = simpler systemd auto-restart; choose your trade
```

Capture the public key — you'll paste it into the instance creation form.

```bash
cat ~/.ssh/oci_prsm_bootstrap.pub
```

### 2.4 Create the ARM Ampere A1 instance

1. Compute → Instances → Create Instance
2. Name: `prsm-bootstrap-eu` (or `prsm-bootstrap-apac` for the second one)
3. Compartment: `prsm-bootstrap` (or root)
4. **Image:** Canonical Ubuntu 22.04 (or 24.04 when available)
5. **Shape:** "Change Shape" → Specialty and previous generation → **Ampere → VM.Standard.A1.Flex** → 1 OCPU + 6 GB RAM (well within free tier)
6. **Networking:** Use default VCN + subnet. Confirm public IPv4 is enabled.
7. **SSH key:** Paste the public key from §2.3
8. **Boot volume:** Default (47 GB free tier OK)
9. Create

Wait ~2 min for provisioning. Note the public IP that appears on the instance detail page.

### 2.5 Open the firewall port

OCI's default security list blocks everything except port 22. You need:
- TCP/8765 — bootstrap WebSocket
- TCP/443 — Let's Encrypt HTTP-01 challenge (and future operator API surface if added)
- TCP/9090 — Prometheus metrics (optional, only if you want external scraping; otherwise leave closed)

1. Networking → Virtual Cloud Networks → (your default VCN) → Security Lists → Default Security List
2. Add Ingress Rules:

| Source CIDR | Protocol | Dest Port | Description |
|---|---|---|---|
| `0.0.0.0/0` | TCP | 8765 | PRSM bootstrap WebSocket |
| `0.0.0.0/0` | TCP | 443 | TLS / Let's Encrypt |
| `0.0.0.0/0` | TCP | 80 | Let's Encrypt HTTP-01 (drop after cert issued) |

**Also configure Ubuntu's firewall** on the instance (iptables/ufw) — OCI's security list AND the local firewall both need the ports open:

```bash
# From the SSH session (see §2.6)
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8765/tcp
sudo ufw --force enable
```

### 2.6 SSH in

```bash
ssh -i ~/.ssh/oci_prsm_bootstrap ubuntu@<PUBLIC_IP>
```

First-login housekeeping:

```bash
sudo apt update && sudo apt upgrade -y
sudo timedatectl set-timezone UTC
sudo hostnamectl set-hostname prsm-bootstrap-eu  # or apac
```

---

## 3. DNS: point the canonical name at the new IP

This is the load-bearing step that makes the sprint-375 fallback path actually fire on operator nodes.

1. Go to your DNS control panel (Cloudflare / Namecheap / whoever holds `prsm-network.com`)
2. Add an **A record**:

| Type | Name | Value | TTL |
|---|---|---|---|
| A | `bootstrap-eu` (or `bootstrap-apac`) | `<PUBLIC_IP from §2.4>` | 300 |

3. Verify propagation:

```bash
dig bootstrap-eu.prsm-network.com +short
# Expect: <PUBLIC_IP>
```

DNS typically propagates in 1–5 minutes with TTL 300; can be up to 24 hours on stubbornly-cached resolvers.

---

## 4. Install + configure the bootstrap server

All commands run on the OCI instance (the `ubuntu@prsm-bootstrap-eu:~$` shell).

### 4.1 Install system packages

```bash
sudo apt install -y python3.11 python3.11-venv git certbot nginx
```

(Python 3.11 is the canonical PRSM target per `pyproject.toml`; verify with `python3.11 --version`.)

### 4.2 Clone PRSM

```bash
sudo mkdir -p /opt/prsm
sudo chown -R ubuntu:ubuntu /opt/prsm
cd /opt/prsm
git clone https://github.com/prsm-network/PRSM.git .
```

### 4.3 Create the venv + install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .  # editable install per pyproject.toml
```

(Editable install lets you `git pull` to update without re-running `pip install`.)

### 4.4 Issue the TLS certificate

The bootstrap server speaks WSS (TLS-wrapped WebSocket), so you need a real cert. Use Let's Encrypt:

```bash
# Temporarily stop anything binding port 80
sudo systemctl stop nginx 2>/dev/null || true

# Issue cert (standalone mode binds port 80 for HTTP-01)
sudo certbot certonly --standalone \
  -d bootstrap-eu.prsm-network.com \
  --agree-tos \
  -m foundation-ops@prsm-network.com \
  --non-interactive

# Cert lands at:
#   /etc/letsencrypt/live/bootstrap-eu.prsm-network.com/
#     fullchain.pem    ← cert + intermediates
#     privkey.pem      ← private key
```

**Cert auto-renewal:** certbot installs a systemd timer (`systemctl list-timers certbot.timer`); no further action needed unless renewal fails (check `sudo systemctl status certbot.timer` quarterly).

### 4.5 Configure the bootstrap server via env file

```bash
sudo mkdir -p /etc/prsm
sudo tee /etc/prsm/bootstrap-server.env > /dev/null <<'EOF'
PRSM_BOOTSTRAP_HOST=0.0.0.0
PRSM_BOOTSTRAP_PORT=8765
PRSM_SSL_ENABLED=true
PRSM_SSL_CERT_PATH=/etc/letsencrypt/live/bootstrap-eu.prsm-network.com/fullchain.pem
PRSM_SSL_KEY_PATH=/etc/letsencrypt/live/bootstrap-eu.prsm-network.com/privkey.pem
PRSM_DOMAIN=bootstrap-eu.prsm-network.com
PRSM_EXTERNAL_IP=<PUBLIC_IP>
PRSM_MAX_PEERS=500
PRSM_PEER_TIMEOUT=300
PRSM_HEARTBEAT_INTERVAL=30
# Sprint 383: peer-DB path override — without this the server
# tries the Docker-conventional /app/data/ default and emits
# spurious "Read-only file system: '/app'" errors on every
# peer state change.
PRSM_PEER_DB_PATH=/var/lib/prsm-bootstrap/peers.db
EOF
# Create the peer-DB directory (writable by the service user)
sudo mkdir -p /var/lib/prsm-bootstrap
sudo chown ubuntu:ubuntu /var/lib/prsm-bootstrap
sudo chmod 640 /etc/prsm/bootstrap-server.env
sudo chown root:ubuntu /etc/prsm/bootstrap-server.env
```

(Swap `bootstrap-eu` → `bootstrap-apac` and the corresponding cert paths for the APAC instance.)

### 4.6 Allow the prsm user to read the Let's Encrypt private key

Let's Encrypt creates `privkey.pem` with `root:root` 600 by default. The bootstrap server runs as `ubuntu`, so we need read access:

```bash
sudo chmod 750 /etc/letsencrypt/live /etc/letsencrypt/archive
sudo chown -R root:ubuntu /etc/letsencrypt/live /etc/letsencrypt/archive
sudo chmod 640 /etc/letsencrypt/archive/bootstrap-eu.prsm-network.com/*.pem
```

(Renewal preserves these perms because certbot's deploy hook copies the cert + adjusts; verify after first renewal via `sudo certbot renew --dry-run`.)

### 4.7 Create the systemd service

```bash
sudo tee /etc/systemd/system/prsm-bootstrap.service > /dev/null <<'EOF'
[Unit]
Description=PRSM Bootstrap Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/prsm
EnvironmentFile=/etc/prsm/bootstrap-server.env
ExecStart=/opt/prsm/.venv/bin/python -m prsm.bootstrap.server
Restart=on-failure
RestartSec=5s
StandardOutput=append:/var/log/prsm-bootstrap.log
StandardError=append:/var/log/prsm-bootstrap.log

# Hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log /var/lib/prsm-bootstrap
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

sudo touch /var/log/prsm-bootstrap.log
sudo chown ubuntu:ubuntu /var/log/prsm-bootstrap.log
sudo systemctl daemon-reload
sudo systemctl enable --now prsm-bootstrap
```

### 4.8 Verify

```bash
sudo systemctl status prsm-bootstrap
# Expect: active (running)

sudo tail -f /var/log/prsm-bootstrap.log
# Expect lines like:
#   INFO - prsm.bootstrap.server - Bootstrap server listening on 0.0.0.0:8765 (SSL=True)
```

---

## 5. End-to-end verification from a separate node

**Canonical one-shot (sprint 385).** From your laptop (or any PRSM-installed machine), run the fleet probe — TCP + TLS + WSS layers across every canonical bootstrap host in a single command:

```bash
prsm node bootstrap-test
# Expect (after this guide ships EU online):
#   PRSM Bootstrap Fleet Probe — ✓ all healthy (3/3 reachable)
#   ✓ bootstrap1.prsm-network.com:8765    TCP ✓  TLS ✓  WSS ✓   42ms
#   ✓ bootstrap-eu.prsm-network.com:8765  TCP ✓  TLS ✓  WSS ✓   88ms
#   ✓ bootstrap-apac.prsm-network.com:8765 TCP ✓  TLS ✓  WSS ✓ 150ms
```

JSON output for ops automation:

```bash
prsm node bootstrap-test --format json
# {"status": "all_healthy", "hosts": [...]}
```

AI-assisted triage from inside Claude (sprint 387 MCP tool):

```
prsm_bootstrap_test
# Same probe, rendered for the side panel, with per-host TCP/TLS/WSS markers + cert subject/issuer.
```

**Manual fallback** (only if `prsm` CLI isn't installed on the probe host):

```bash
# 1. TCP reachability
nc -zv bootstrap-eu.prsm-network.com 8765
# Expect: Connection succeeded

# 2. TLS handshake
openssl s_client -connect bootstrap-eu.prsm-network.com:8765 \
  -servername bootstrap-eu.prsm-network.com < /dev/null \
  | grep "Verify return code"
# Expect: Verify return code: 0 (ok)

# 3. WSS handshake (uses websocat if installed; otherwise a python one-liner)
python3 -c "
import asyncio, websockets, ssl
async def go():
    async with websockets.connect(
        'wss://bootstrap-eu.prsm-network.com:8765',
        ssl=ssl.create_default_context(),
    ) as ws:
        print('WSS handshake OK')
asyncio.run(go())
"
# Expect: WSS handshake OK
```

Then point a fresh PRSM node at the new bootstrap to confirm it actually registers:

```bash
# On the test node
export BOOTSTRAP_PRIMARY=wss://bootstrap-eu.prsm-network.com:8765
prsm node start
# In another terminal:
prsm node bootstrap
# Expect:
#   PRSM Bootstrap Status — ✓ healthy
#     client_state: connected
#     active URL:   bootstrap-eu.prsm-network.com:8765
```

---

## 6. Monitoring + observability

The sprint-377 Prometheus surface is already wired into every PRSM operator node — the bootstrap *server* itself doesn't run `/metrics` (it's a registration daemon, not a full node). Server observability is via:

- **systemd:** `sudo systemctl status prsm-bootstrap`
- **Log file:** `/var/log/prsm-bootstrap.log` (append-mode; rotate with logrotate if it grows)
- **Process metrics:** `ss -tnlp | grep 8765` to confirm the listener; `ps aux | grep prsm.bootstrap`

For fleet-side observability of "is the EU bootstrap being used":

- Run an instance of `prsm_bootstrap_status` MCP against any node — operators connecting via EU show `active_url: wss://bootstrap-eu.prsm-network.com:8765`
- Prometheus-side: `count by (url) (prsm_bootstrap_active)` across the operator fleet — EU shows up as a non-zero label when operators in EU regions cold-start

For external reachability monitoring (separate from the operator-side view):

- **Canonical CLI** (sprint 385): `prsm node bootstrap-test --format json` from a cron job on any PRSM-installed host. Stable JSON shape (`status`, `hosts[].{tcp_ok,tls_ok,wss_ok,latency_ms}`) suitable for piping into a monitoring agent — alert on `status != "all_healthy"` or any per-host `wss_ok: false`.
- **Cron snippet** (e.g., on a US-east probe host):
  ```cron
  */5 * * * * /usr/local/bin/prsm node bootstrap-test --format json \
      | /opt/prsm-ops/bin/alert-on-fleet-degradation.sh
  ```
- **AI triage** (sprint 387): when an alert fires, invoke `prsm_bootstrap_test` from Claude/Claude Code — the MCP tool surfaces per-layer breakdown (TCP / TLS / WSS) + cert subject/issuer per host so you can localize the failure (DNS? cert? listener?) without SSHing in.

### 6.1 Optional: external uptime monitor

Cheap-or-free options:
- **UptimeRobot** free tier — 50 monitors, 5-min check interval, HTTPS check against `https://bootstrap-eu.prsm-network.com:8765` (the WSS server speaks HTTPS handshake on the same port)
- **Better Uptime** free tier — similar
- **Hetrixtools** free tier

Set up TCP-check on port 8765 against both hosts; alert via Slack/email/Discord when down.

---

## 7. Maintenance

### 7.1 Software updates

```bash
ssh ubuntu@<PUBLIC_IP>
cd /opt/prsm
git pull
sudo systemctl restart prsm-bootstrap
sudo tail -20 /var/log/prsm-bootstrap.log
```

### 7.2 OS updates (monthly cadence recommended)

```bash
sudo apt update && sudo apt upgrade -y
sudo reboot
```

The systemd service auto-restarts on boot.

### 7.3 Cert renewal verification (quarterly)

```bash
sudo certbot renew --dry-run
# Expect: "Congratulations, all simulated renewals succeeded"
```

If real renewal fires (auto via certbot.timer), the bootstrap server picks up the new cert on its next restart — `sudo systemctl restart prsm-bootstrap` after renewal.

### 7.4 Disk pressure

ARM Ampere A1 free tier ships 47 GB boot volume. Logs grow:

```bash
# /var/log/prsm-bootstrap.log + journalctl + apt cache
sudo du -sh /var/log/* | sort -h | tail -5
sudo journalctl --vacuum-time=14d
sudo apt clean
```

Set up logrotate for the PRSM log:

```bash
sudo tee /etc/logrotate.d/prsm-bootstrap > /dev/null <<'EOF'
/var/log/prsm-bootstrap.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 644 ubuntu ubuntu
    postrotate
        systemctl reload prsm-bootstrap 2>/dev/null || true
    endscript
}
EOF
```

---

## 8. Troubleshooting

### 8.1 Service won't start

```bash
sudo systemctl status prsm-bootstrap
sudo journalctl -u prsm-bootstrap -n 100 --no-pager
```

Common causes:
- **Cert file unreadable** — re-check §4.6 chmod
- **Port 8765 in use** — `sudo ss -tnlp | grep 8765` (probably an old `python` you forgot to kill)
- **Python module path** — `cd /opt/prsm && .venv/bin/python -c "import prsm.bootstrap.server"`; ImportError means re-run `.venv/bin/pip install -e .`

### 8.2 TLS handshake fails from external node

- **OCI security list missing port 8765** — re-check §2.5
- **ufw blocking** — re-check §2.5 local firewall step
- **DNS pointing wrong** — `dig +short bootstrap-eu.prsm-network.com` vs OCI public IP
- **Cert/SNI mismatch** — `openssl s_client -connect ... -servername ...` should return `Verify return code: 0 (ok)`

### 8.3 WSS handshake succeeds but operators don't show up

- Operator's `prsm node bootstrap` shows `active_url: null` even with EU/APAC reachable → operator's node may not have sprint-375 code yet. Check operator's `prsm version` against the multi-bootstrap-fallback tag (sprint 375).
- Server-side log shows TLS connections but no `register` messages → client-side issue, not server.

---

## 9. Migration path (when you outgrow free tier)

OCI Always Free pool is 4 ARM Ampere A1 cores + 24 GB RAM across all your instances. Two `1 OCPU + 6 GB` bootstraps consume 2 cores + 12 GB — well within the pool. If usage grows (PRSM fleet hits 10K+ operators), migration paths in order of friction:

1. **Bump OCI instance shape within Always Free** — go to 2 OCPU + 12 GB on each, still free if you stay under 4 cores total
2. **Migrate to OCI paid tier same shape** — minor cost (~$10/mo per instance)
3. **Migrate to DigitalOcean Droplet matching bootstrap1's shape** — change DNS A record, the rest of the protocol doesn't care
4. **Hybrid: keep OCI for free EU/APAC, add a third DO bootstrap for redundancy** — sprint 375 fallback list scales to N

---

## 10. Cross-references

- `prsm/node/config.py:20-29` — `DEFAULT_BOOTSTRAP_NODES` + `FALLBACK_BOOTSTRAP_NODES` consume these URLs
- `prsm/node/libp2p_discovery.py` — sprint 375 fallback-iteration code path
- `prsm/cli_helpers/bootstrap_probe.py` — sprint 385 TCP+TLS+WSS layered probe (CLI + MCP shared backend)
- `prsm/cli.py` `node bootstrap-test` — sprint 385 operator CLI surface
- `prsm/mcp_server.py` `prsm_bootstrap_test` — sprint 387 AI-side-panel surface
- `docs/operations/fleet-kill-switch-operator-runbook.md` — sister operator runbook (§7.21 honest-scope closer)
- `docs/2026-04-27-cumulative-audit-prep.md` §7.29 + §7.35 — multi-bootstrap arc audit-prep entries
- `docs/governance/PRSM-CR-2026-05-13-2.md` §5 non-scope item 6 — explicit acknowledgment that EU+APAC droplets are operator-driven

---

## 11. Changelog

| Date | Sprint | Change |
|---|---|---|
| 2026-05-13 | post-381 | Initial guide. Closes the operator-side ops gap that PRSM-CR-2026-05-13-2 §5 non-scope item 6 flagged as "operator-driven, not engineering-driven." When the EU + APAC droplets land, the sprint-375 fallback code path immediately benefits — every cold-start operator in those regions reaches a host before falling back to US. |
| 2026-05-14 | 385/387 update | §5 + §6 now point at the canonical `prsm node bootstrap-test` CLI (sprint 385) + `prsm_bootstrap_test` MCP tool (sprint 387) as the single-command fleet probe. Manual nc/openssl/python triplet retained as fallback for hosts without the PRSM CLI installed. |
