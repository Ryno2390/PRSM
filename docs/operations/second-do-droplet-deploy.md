# Second DigitalOcean droplet — operator deployment

Sprint 675 — playbook for adding a 2nd always-on PRSM operator node on a
DigitalOcean droplet. Closes the sprint 673 multi-host live-attest gap with
a permanent (not transient) fleet member.

## Target configuration (this session's choices)

- **Tier**: Basic Regular 1GB / 1 vCPU / 25GB SSD — **$6/mo**
- **Region**: SFO3 (US-West)
- **OS**: Ubuntu 24.04 LTS
- **Role**: Operator-only (not a bootstrap server)
- **Connects to**: `wss://bootstrap-us.prsm-network.com:8765` for discovery

The existing bootstrap-us droplet at `159.203.129.218` (NYC3) co-locates
the bootstrap server + an operator daemon. The new droplet is operator-only;
it dials bootstrap-us to find peers + announce itself.

NYC ↔ SFO WAN latency: ~70ms — visible in multi-stage timing measurements
(real cross-region demonstration of PRSM's chain-of-trust handoff).

## 8-step deployment

### 1. Provision the droplet (you)

Via the DigitalOcean web UI:
1. Click "Create" → "Droplets"
2. Choose region: **SFO3 (San Francisco 3)**
3. Choose image: **Ubuntu 24.04 LTS**
4. Choose plan: **Basic** → CPU options: **Regular** → **$6/mo (1GB / 1 vCPU / 25GB)**
5. Add your SSH key (or create one)
6. Hostname: something memorable like `prsm-operator-sfo` (this is just
   for your DO console; PRSM doesn't care)
7. Click "Create Droplet"

Wait ~30s for provisioning. Note the public IPv4 from the DO console.

Tell me when the droplet is up + share the IPv4 address. I'll drive the
remaining 7 steps with one copy-paste command per step.

### 2. SSH baseline + system deps (I'll give the command)

```bash
ssh root@<DROPLET_IP> 'apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3.11 python3.11-venv python3-pip git build-essential \
        libssl-dev libffi-dev'
```

### 3. Clone PRSM + venv install (I'll give the command)

```bash
ssh root@<DROPLET_IP> 'cd /opt && \
    git clone https://github.com/prsm-network/PRSM.git prsm-operator && \
    cd prsm-operator && \
    python3.11 -m venv .venv && \
    .venv/bin/pip install --quiet --upgrade pip && \
    .venv/bin/pip install --quiet -e ".[blockchain]"'
```

Lazy-imports from sprints 460/462 mean we don't need full ML deps for the
daemon to boot. We'll add them in step 5.

### 4. Generate node identity (I'll give the command)

```bash
ssh root@<DROPLET_IP> 'cd /opt/prsm-operator && \
    .venv/bin/python -m prsm.cli node init'
```

This generates `/root/.prsm/identity.json` with a fresh Ed25519 keypair.
The `node_id` (32 hex chars) is what the anchor will record. The
`public_key_b64` is what we'll register on-chain.

I'll then pull both values from the file via SSH so the registration
script has them.

### 5. Install ML deps (I'll give the command)

```bash
ssh root@<DROPLET_IP> 'cd /opt/prsm-operator && \
    .venv/bin/pip install --quiet transformers torch'
```

Sprint 460's lazy-import refactor makes this safe to install AFTER
node init. Torch + transformers add ~2GB to disk; on this 25GB droplet
that's plenty of headroom.

### 6. Register pubkey on the live anchor (one mainnet TX — you authorize)

From YOUR LOCAL MACHINE (not the droplet — your deployer key never leaves
your machine):

```bash
cd /Users/ryneschultz/Documents/GitHub/PRSM

# I'll extract these values from the droplet for you:
OPERATOR_NODE_ID=<from-step-4>
OPERATOR_PUBKEY_B64=<from-step-4>
PRSM_DEPLOYER_PRIVATE_KEY=<your-funded-Base-EOA-key> \
    OPERATOR_NODE_ID=$OPERATOR_NODE_ID \
    OPERATOR_PUBKEY_B64=$OPERATOR_PUBKEY_B64 \
    python scripts/sprint_675_register_operator_pubkey.py
```

This submits ONE TX to the live PublisherKeyAnchor on Base mainnet.
Cost ~$0.01. **Irreversible once mined.**

After confirmation (~10s), the script verifies via anchor.lookup()
that the new operator's pubkey is on-chain.

### 7. Copy gpt2 model registry from bootstrap-us droplet (I'll give the command)

```bash
ssh root@bootstrap-us.prsm-network.com 'tar czf /tmp/gpt2_registry.tgz -C /var/lib/prsm-registry gpt2' && \
scp root@bootstrap-us.prsm-network.com:/tmp/gpt2_registry.tgz /tmp/ && \
scp /tmp/gpt2_registry.tgz root@<DROPLET_IP>:/tmp/ && \
ssh root@<DROPLET_IP> 'mkdir -p /var/lib/prsm-registry && \
    tar xzf /tmp/gpt2_registry.tgz -C /var/lib/prsm-registry'
```

The manifest was signed by Mac's NodeIdentity (sprint 626) — that
signature stays valid regardless of which operator holds the registry,
because the publisher_node_id is encoded in the signed payload.

### 8. Configure + start daemon via systemd (I'll give the unit file)

```bash
ssh root@<DROPLET_IP> 'cat > /etc/systemd/system/prsm-operator.service <<EOF
[Unit]
Description=PRSM Operator Node (sprint 675 — SFO3)
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/prsm-operator
Environment=PRSM_PARALLAX_TRUST_STACK_KIND=production
Environment=PRSM_PARALLAX_LAYER_SLICE_RUNNER_KIND=huggingface
Environment=PRSM_PARALLAX_HF_MODEL_ID=gpt2
Environment=PRSM_PARALLAX_HF_DEVICE=cpu
Environment=PRSM_PARALLAX_KV_CACHE_ENABLED=1
Environment=PRSM_MODEL_REGISTRY_ROOT=/var/lib/prsm-registry
Environment=PRSM_STAKE_BOND_ADDRESS=0xD4C6584BB69d1cc46B32502c57124Df12D8979Ed
ExecStart=/opt/prsm-operator/.venv/bin/python -m prsm.cli node start \\
    --no-dashboard --api-port 8002 \\
    --bootstrap wss://bootstrap-us.prsm-network.com:8765
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload && systemctl enable --now prsm-operator && \
    sleep 8 && systemctl status prsm-operator --no-pager | head -10'
```

### 9. Verify symmetric discovery + live-attest multi-host (I'll drive)

From your Mac:
```bash
prsm node info  # should show 2 connected peers now
```

Then a 2-stage live inference where stage 0 → existing droplet (NYC), stage
1 → new SFO droplet:
```bash
prsm node infer --prompt "The capital of France is" -n 5 \
    --stages "0-6:484f003c895ee02ac7ed01e570a6a51f" \
    --stages "6-12:<new-sfo-node-id>" \
    --save-receipts /tmp/multihost.jsonl

prsm node verify-receipts /tmp/multihost.jsonl
```

Expected: 5 tokens generated, each with 2 stage_chain entries (NYC droplet
+ SFO droplet) signed by distinct node_ids. The verify-receipts output
shows per-stage signatures all ✓ against the live mainnet anchor.

Multi-stage live-attest CLOSED on a true multi-host fleet.

## Monitoring + cost

- **Uptime**: systemd's `Restart=on-failure` keeps the daemon alive
  through transient crashes; DO's hypervisor reboots restart the unit
  automatically.
- **Cost**: $6/mo fixed. Add to existing $4/mo bootstrap-us droplet → total
  $10/mo for the 2-operator fleet.
- **Bandwidth**: PRSM ledger + bootstrap heartbeats are negligible; chain-
  exec-ping payloads are 1-3MB per inference token (activations). At
  typical demo cadence well within DO's 1TB/mo egress allotment.

## Follow-on (after this fleet is up)

- Sprint 676+: Tighten the bootstrap-mediated peer announcement for
  multi-host fleets (test with 3+ peers if/when we add bootstrap-eu or
  bootstrap-apac droplet siblings).
- Sprint 677+: Add `prsm node fleet-status` CLI that lists all
  anchor-registered peers + last-seen times.
- Future: Lambda GPU session as a 3rd PERIODIC operator (sprint 674
  playbook); same anchor, different cost pattern.
