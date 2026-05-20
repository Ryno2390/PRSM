# Lambda GPU operator deployment

Sprint 674 — playbook for spinning up a Lambda Labs Cloud GPU instance as a
second PRSM operator node. Closes two arcs simultaneously:

1. **Multi-stage chain live-attest** (sprint 673 gap): distinct operator
   identity at a distinct IP → true multi-host attestation, not just
   CI-mocked.
2. **GPU-accelerated stage forward** (foundation for sprint 631+ DHT GPU pool):
   real GPU available for the LayerStageServer's huggingface runner.

## Cost model

Lambda Cloud is hourly-billed:

| Instance | $/hr | GPU | RAM | Use case |
|----------|------|-----|-----|----------|
| `gpu_1x_a10` | $0.75 | A10 24GB | 30GB | gpt2 / llama-3.2-1b |
| `gpu_1x_a100_pcie_40gb` | $1.29 | A100 40GB | 200GB | llama-3.2-3b / 7b |
| `gpu_1x_h100_pcie` | $2.49 | H100 80GB | 200GB | llama-3.1-8b+ |

Sprint 674 use case (gpt2): `gpu_1x_a10` at $0.75/hr is sufficient. Plan to
spin up for testing sessions then tear down. A 2-hour session = $1.50.

Always-on operator membership would be ~$540/month at A10 rates — NOT
recommended; use the cheaper DO droplet pattern for always-on fleet, Lambda
for GPU-accelerated bursts.

## Prerequisites

- Lambda Cloud account (https://lambda.ai/cloud) with payment method
- SSH key registered on Lambda
- Local checkout of PRSM (this repo)
- Your `PRSM Mainnet Deployer` EOA private key (sprint 623) for the
  anchor registration mainnet TX

## Steps

### 1. Spin up the instance

Via Lambda Cloud console or CLI:
```
lambda-cloud instance launch \
    --instance-type gpu_1x_a10 \
    --region us-west-2 \
    --ssh-key-name <your-key>
```

Note the public IPv4 once it's running. Lambda dynamic IPs change between
sessions — record it for this run.

### 2. SSH in + install PRSM

```
ssh ubuntu@<lambda-public-ip>

# System deps
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip git

# Clone PRSM
git clone https://github.com/prsm-network/PRSM.git /opt/prsm-operator
cd /opt/prsm-operator

# Venv + dependencies (uses GPU torch already on Lambda)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[blockchain]"
pip install transformers torch  # torch is pre-installed on Lambda; this
                                # picks up the existing CUDA build
```

### 3. Generate node identity + register pubkey on anchor

On the Lambda instance:
```
.venv/bin/python -m prsm.cli node init  # generates identity at ~/.prsm/identity.json
```

Capture the node_id + public_key_b64 from `~/.prsm/identity.json`. You'll
need both to register on the anchor.

**On your local machine** (where the Mainnet Deployer EOA private key
lives — NEVER on the Lambda instance):

```
PRSM_DEPLOYER_PRIVATE_KEY=<your_deployer_key> \
LAMBDA_NODE_ID=<from-step-3> \
LAMBDA_PUBKEY_B64=<from-step-3> \
python scripts/sprint_674_register_lambda_pubkey.py
```

This submits a single `register()` TX to the live PublisherKeyAnchor at
`0xd811ad9986f44f404b0fd992168a7cc76206df03` on Base mainnet. Irreversible
once mined. Wait ~10s for confirmation; verify via:

```
.venv/bin/python -c "
from prsm.security.publisher_key_anchor.client import PublisherKeyAnchorClient
anchor = PublisherKeyAnchorClient(
    contract_address='0xd811ad9986f44f404b0fd992168a7cc76206df03',
    rpc_url='https://mainnet.base.org',
)
print(anchor.lookup('<LAMBDA_NODE_ID>'))
"
```

Should print the Lambda's pubkey_b64.

### 4. Copy gpt2 model registry to Lambda

```
scp -r /opt/prsm-operator/var/lib/prsm-registry/gpt2 \
    ubuntu@<lambda-ip>:/opt/prsm-operator/var/lib/prsm-registry/
```

Or pre-download gpt2 via huggingface-cli on Lambda directly.

### 5. Start the daemon

On the Lambda instance:
```
export PRSM_PARALLAX_TRUST_STACK_KIND=production
export PRSM_PARALLAX_LAYER_SLICE_RUNNER_KIND=huggingface
export PRSM_PARALLAX_HF_MODEL_ID=gpt2
export PRSM_PARALLAX_HF_DEVICE=cuda  # ← engages the GPU!
export PRSM_PARALLAX_KV_CACHE_ENABLED=1
export PRSM_MODEL_REGISTRY_ROOT=/opt/prsm-operator/var/lib/prsm-registry
export PRSM_STAKE_BOND_ADDRESS=0xD4C6584BB69d1cc46B32502c57124Df12D8979Ed

nohup .venv/bin/python -m prsm.cli node start \
    --no-dashboard --api-port 8002 \
    --bootstrap wss://bootstrap-us.prsm-network.com:8765 \
    > /tmp/prsm_lambda.log 2>&1 &
```

### 6. Verify symmetric discovery

From your Mac:
```
prsm node info | grep -A2 "Connected peers"
```

Should show TWO connected peers: the existing DigitalOcean droplet
(`484f003c...`) AND the new Lambda instance (`<lambda-node-id>`).

### 7. Live-attest multi-host multi-stage inference

```
prsm node infer --prompt "The capital of France is" -n 5 \
    --stages "0-6:484f003c895ee02ac7ed01e570a6a51f" \
    --stages "6-12:<lambda-node-id>" \
    --save-receipts /tmp/multihost.jsonl

prsm node verify-receipts /tmp/multihost.jsonl
```

Expected: each receipt's stage_chain has 2 entries with DIFFERENT
stage_node_ids (DO droplet + Lambda); both signatures verify against
the live mainnet anchor.

### 8. Tear down when done

```
lambda-cloud instance terminate <instance-id>
```

The node_id stays registered on the anchor — it's a permanent fact that
this identity was once active. Next session: spin up a new Lambda
instance, generate a FRESH node_id (different identity), register that.

## Live-attest target metrics

- Stage 0 (DO droplet CPU): ~1s/forward on gpt2 (current sprint 660
  baseline)
- Stage 1 (Lambda A10 GPU): ~50ms/forward on gpt2 (estimate; 20x speedup
  from CUDA)
- Total per-token: ~1.05s (dominated by the CPU stage)

Could route the WHOLE chain through Lambda for max speedup, but the
multi-host attestation requires BOTH to be involved per token.

## Follow-on (sprint 675+)

- Add `prsm node deploy-lambda` CLI that automates steps 1-5
- Add a `prsm node fleet-status` command that lists all anchor-registered
  peers + their last-seen times
- DHT-backed GPU pool provider (sprint 631 documented gap) can now
  live-attest against the Lambda GPU
