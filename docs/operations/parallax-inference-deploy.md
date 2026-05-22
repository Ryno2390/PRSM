# Parallax Inference — Operator deployment runbook

Sprint 697 (refreshed sprint 709) — single consolidated runbook for
standing up a PRSM operator node that participates in the §7
verifiable-inference pool via the `ParallaxScheduledExecutor` path.
Reflects the state of sprints 558-708, with all 23 PRSM_PARALLAX_* /
PRSM_INFERENCE_* env vars + the on-chain wiring + the multi-host
architecture (sprint 695) + the OOM gate (sprint 704) + the
DP-injection path (sprint 702) + Lambda GPU support (sprint 698).

After completing this runbook, the operator's node:

- Joins the DHT-backed GPU pool (sprint 682)
- Advertises hardware via DISCOVERY_ANNOUNCE (sprint 680)
- Serves `/compute/inference` requests with cryptographically-signed
  receipts (sprint 687)
- Serves `/compute/inference/stream` SSE with real per-token autoregressive
  output (sprint 693)
- Participates in cross-host 2-stage allocation (sprint 695)

Existing fleet members (use as canonical examples):

- `bootstrap-us.prsm-network.com:8765` — bootstrap server + co-located
  NYC operator daemon `484f003c895ee02ac7ed01e570a6a51f`
- `146.190.175.239` — SFO operator `d437aa67d99cff4a6a17179f5c731b77`

## Prerequisites

- Linux host with ≥ 2GB RAM (1GB is the hard floor — sprint 459 surfaced
  OOM on 1GB during `gpt2` cold-load; sprint 678 closed by upgrading
  SFO to 2GB)
- Python 3.10+, `pip`, `git`, `build-essential`
- Network reachable from `bootstrap-us.prsm-network.com:8765` (outbound
  HTTPS + WSS)
- A funded Base mainnet EOA for the **deployer** account (~0.0001 ETH for
  the anchor registration TX — ~$0.01)

If you have a separate Lambda Cloud GPU instance instead of a CPU
droplet, use `lambda-gpu-operator-deploy.md` for the provisioning steps,
then return here for the env config + readiness check.

## 1. Install PRSM

```bash
git clone https://github.com/prsm-network/PRSM.git /opt/prsm-operator && cd /opt/prsm-operator && python3 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -e .
```

After install completes, verify imports cleanly:

```bash
cd /opt/prsm-operator && .venv/bin/python -c "from prsm.node.node import PRSMNode; print('ok')"
```

## 2. Generate the node identity

The first daemon start creates `~/.prsm/identity.json` containing this
node's Ed25519 keypair + node_id. Generate it now without starting the
daemon:

```bash
cd /opt/prsm-operator && .venv/bin/python -c "from prsm.node.identity import NodeIdentity; i = NodeIdentity.load_or_create(); print('node_id:', i.node_id); print('pubkey_b64:', i.public_key_b64)"
```

Record both values. The `node_id` is 32 hex chars; the `pubkey_b64` is
the base64-encoded Ed25519 public key.

## 3. Register the pubkey on the live anchor

The Base-mainnet `PublisherKeyAnchor` at
`0xd811ad9986f44f404b0fd992168a7cc76206df03` is the canonical name-resolver
that other peers consult when verifying signed receipts from this node.
Until the pubkey is registered there, the node's signatures cannot be
verified by anyone — and the DHT pool's `AnchorVerifyAdapter` rejects
the node entirely.

Use sprint 675's parameterized registration script:

```bash
OPERATOR_NODE_ID=<your-node-id> OPERATOR_PUBKEY_B64=<your-pubkey-b64> PRSM_DEPLOYER_PRIVATE_KEY=0x<funded-base-eoa-key> .venv/bin/python scripts/sprint_675_register_operator_pubkey.py
```

Costs ~$0.01 in gas. The script is idempotent: if the node_id is already
registered, it skips with a warning.

## 4. Configure systemd

Create the unit file at `/etc/systemd/system/prsm-operator.service`. Mirror
the canonical bootstrap-us reference (see
`bootstrap-us-operator-systemd.service.reference`). For a fresh deploy:

```ini
[Unit]
Description=PRSM Operator Node
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=/opt/prsm-operator
Environment=HOME=/root
Environment=PRSM_TRANSPORT_BACKEND=websocket
Environment=PRSM_ADVERTISE_ADDRESS=<your-public-ipv4>
ExecStart=/opt/prsm-operator/.venv/bin/python -m prsm.cli node start --no-dashboard --api-port 8002 --p2p-port 9001 --bootstrap wss://bootstrap-us.prsm-network.com:8765
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Notes:

- `PRSM_ADVERTISE_ADDRESS` must be the externally-reachable IPv4 so remote
  peers can dial back. The bootstrap-server-supplied client_ip would be
  127.0.0.1 if you bootstrap via loopback (co-located deployment); the
  advertise override fixes that (sprint 566).
- For a node co-located with a bootstrap server, change the
  `--bootstrap` flag to `wss://127.0.0.1:8765` (sprint 460/565 fleet-
  coordination invariant — NAT-hairpin via the public hostname fails
  inside the same droplet).

## 5. Configure the Parallax env block

Add a systemd drop-in for the Parallax-specific env vars (keeps the main
unit small + lets you tune without re-editing):

```bash
mkdir -p /etc/systemd/system/prsm-operator.service.d && cat > /etc/systemd/system/prsm-operator.service.d/parallax.conf <<'EOF'
[Service]
# § Verifiable inference path (sprint 558-696)
Environment=PRSM_INFERENCE_EXECUTOR=parallax
Environment=PRSM_PARALLAX_GPU_POOL_KIND=dht-backed
Environment=PRSM_PARALLAX_TRUST_STACK_KIND=production
Environment=PRSM_PARALLAX_MODEL_CATALOG_FILE=/opt/prsm-operator/config/parallax/model_catalog.json
Environment=PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS=0xd811ad9986f44f404b0fd992168a7cc76206df03
Environment=PRSM_STAKE_BOND_ADDRESS=0xD4C6584BB69d1cc46B32502c57124Df12D8979Ed

# § Chain executor (sprint 598)
Environment=PRSM_PARALLAX_CHAIN_EXECUTOR_KIND=rpc

# § Stage executor + HF runner (sprint 607, 611)
Environment=PRSM_PARALLAX_STAGE_EXECUTOR_KIND=layer_stage
Environment=PRSM_PARALLAX_LAYER_SLICE_RUNNER_KIND=huggingface
Environment=PRSM_PARALLAX_HF_MODEL_ID=gpt2
Environment=PRSM_PARALLAX_HF_DEVICE=cpu
Environment=PRSM_MODEL_REGISTRY_ROOT=/var/lib/prsm-registry

# § Tokenizers (sprint 615, 616, 688)
Environment=PRSM_PARALLAX_PROMPT_ENCODER_KIND=huggingface
Environment=PRSM_PARALLAX_OUTPUT_DECODER_KIND=huggingface

# § Streaming (sprints 692, 693, 694)
Environment=PRSM_PARALLAX_STREAMING_RUNNER_KIND=embedder_backed
Environment=PRSM_PARALLAX_KV_CACHE_ENABLED=1

# § Pool tunables (sprint 686, 695)
# Use advisory until the operator has actually staked on-chain (sprint 690).
Environment=PRSM_PARALLAX_STAKE_ELIGIBILITY=advisory
# Sprint 686 — small model on small node needs explicit layer cap
Environment=PRSM_PARALLAX_LAYER_CAPACITY_OVERRIDE=12
# Sprint 695 — populates rtt_to_nodes for routing DP (inf without this)
Environment=PRSM_PARALLAX_DEFAULT_RTT_MS=100

# § Tier gate (sprint 702) — set to advisory ONLY for dev/test
# exercise of the DP-injection path on software-TEE operators.
# Production with real hardware-TEE keeps default enforced.
Environment=PRSM_PARALLAX_TIER_GATE=advisory

# § OOM gate (sprint 704) — REQUIRED on memory-tight nodes
# (2GB DO droplets). Limits concurrent inference so peak memory
# is bounded by a single request's working set. Without this,
# concurrent inferences OOM-kill the daemon under cold-load.
# Memory-rich nodes (Lambda A10 with 200GB RAM) can omit this
# OR set to 8+ for throughput.
Environment=PRSM_INFERENCE_CONCURRENCY_LIMIT=1

# § Stake-pool wiring (sprint 690) — operator's funded EOA. When
# combined with on-chain StakeBond bond, lets the daemon run
# PRSM_PARALLAX_STAKE_ELIGIBILITY=enforced for production posture.
# Until bonded, advisory mode above bypasses the check.
# Environment=PRSM_OPERATOR_ADDRESS=0x<your-EOA>

# § Timeouts (sprint 687)
Environment=PRSM_PARALLAX_SEND_MESSAGE_TIMEOUT_S=600

# § Multi-stage allocation overrides (sprint 695) — OPT-IN only
# for testing the 2-stage chain path on small homogeneous CPU
# fleets. Defaults work for production heterogeneous deployments
# after sprint 700's F46 fix (monotonic hardware_profile gossip).
# Set memory_gb=0.8 + tflops=30 ONLY when you want to force a
# 2-stage split on tiny droplets running gpt2; otherwise omit.
# Environment=PRSM_PARALLAX_MEMORY_GB_OVERRIDE=0.8
# Environment=PRSM_PARALLAX_TFLOPS_FP16_OVERRIDE=30.0
EOF
```

For Lambda GPU operators: change `PRSM_PARALLAX_HF_DEVICE=cpu` to `cuda`,
omit the `INFERENCE_CONCURRENCY_LIMIT` (200GB RAM doesn't need the
gate), and skip the optional multi-stage-allocation overrides at the
bottom (real GPU memory + tflops clear the Phase-1/Phase-2 thresholds
without them).

## 6. Pre-flight check (sprint 696)

Verify the env block is sound before starting the daemon:

```bash
systemctl daemon-reload && eval "$(systemctl show prsm-operator.service -p Environment --value | tr ' ' '\n' | grep ^PRSM | sed 's/^/export /')" && cd /opt/prsm-operator && .venv/bin/python -m prsm.cli node parallax-readiness
```

A clean `✓ ready` verdict + no `MISSING` or `INVALID` rows means the env
block is sound. Fix anything red before proceeding.

## 7. Start the daemon

```bash
systemctl enable prsm-operator.service && systemctl start prsm-operator.service && sleep 30 && systemctl is-active prsm-operator.service
```

Expected: `active`. If `failed`, check `journalctl -u prsm-operator.service --no-pager | tail -40` for the structured warning naming the missing piece.

## 8. Smoke test — pool snapshot + signed-receipt inference

After ~60-90s for DHT sync to find the other fleet members:

```bash
curl -s -m 5 http://127.0.0.1:8002/admin/parallax/pool/snapshot | python3 -m json.tool
```

Expected: `gpu_count: ≥ 2`, `pool_kind: "dht-backed"`, both NYC + your
new node visible.

Then a real inference call:

```bash
curl -s -m 120 -X POST http://127.0.0.1:8002/compute/inference -H "Content-Type: application/json" -d '{"prompt":"The capital of France is","model_id":"gpt2","budget_ftns":1.0,"privacy_tier":"none","content_tier":"A","max_tokens":1}' | python3 -m json.tool
```

Expected: `success: true`, `output: " the"` (matches HF gpt2 greedy reference exactly per sprint 688), and a `receipt` with `settler_signature` + `output_hash`.

## Troubleshooting (F-class bugs surfaced during sprint 685-708 live-attests)

| Symptom | Cause | Fix |
| --- | --- | --- |
| `/compute/inference` returns 503 "Inference executor not initialized" | Missing one of the 4 required env vars (sprint 696 parallax-readiness reports which) | Set `PRSM_INFERENCE_EXECUTOR=parallax` + the 3 _KIND vars + catalog file |
| `error: "no GPU passed pool gating"` | All peers filtered by AnchorVerifyAdapter (pubkey not on-chain) OR stake-eligibility (advisory mode disabled + no real stake) | Run `scripts/sprint_675_register_operator_pubkey.py` (step 3); use `PRSM_PARALLAX_STAKE_ELIGIBILITY=advisory` for pre-stake live-attest |
| `error: "tier gate refusal: no GPU in pool has hardware-TEE attestation; required for privacy_level=standard"` | Adapter B rejects software-TEE operators for tier≥standard (correct production behavior) | Set `PRSM_PARALLAX_TIER_GATE=advisory` for dev/test DP exercise on software TEE (sprint 702) |
| `error: "TIER_GATE — privacy_tier=standard requires hardware TEE; local runtime is software"` | Server-side runtime check separate from Adapter B | Same `PRSM_PARALLAX_TIER_GATE=advisory` covers both checks (sprint 702 F49) |
| `error: "insufficient capacity: region 'default': total layer_capacity=X < num_layers=12"` | Per-GPU `layer_capacity` heuristic too conservative for small models | Set `PRSM_PARALLAX_LAYER_CAPACITY_OVERRIDE=12` for gpt2 single-host (default in current runbook) |
| `error: "AllocationResult has no pipelines"` (multi-host attempts only) | Memory-vs-embedding math fails — gpt2's 154MB embedding cost eats too much per-node capacity | Calibrate `PRSM_PARALLAX_MEMORY_GB_OVERRIDE` (0.8 is the gpt2 sweet spot — see sprint 695 notes); only needed when forcing multi-stage |
| `error: "no chain in regions ['default'] covers all 12 layers"` | Phase-2 routing DP transition cost is inf because `rtt_to_nodes` is None | Set `PRSM_PARALLAX_DEFAULT_RTT_MS=100` (sprint 695 F44 fix) |
| `error: "transport.send_to_peer returned False for peer_id='1.2.3.4:9001'"` | Address resolver returning peer.address instead of node_id | Already fixed in sprint 695 F45 — pull main if older |
| `error: "chain executor does not support streaming: '_StubChainExecutor'"` | Chain executor falling back to stub because `node._loop` was None at construction time | Already fixed in sprint 686 F33 — pull main if older |
| `error: "Cannot copy out of meta tensor; no data!"` | Newer transformers defaults `low_cpu_mem_usage=True` which leaves weights on meta-tensor | Already fixed in sprint 702 F48 — pull main if older |
| `error: "Expected all tensors to be on the same device"` (GPU operators only) | Prompt encoder's `token_ids` on CPU, model on GPU | Already fixed in sprint 698 F47 — pull main if older |
| Heterogeneous CPU+GPU pool: NYC sees Lambda as CPU with low tflops | Gossip propagation overwrote authoritative profile (F46) | Already fixed in sprint 700 — pull main; monotonic-improvement gossip preserves direct-announce data |
| SSE returns 1 frame per word instead of per token | `PRSM_PARALLAX_STREAMING_RUNNER_KIND=synthetic` (sprint 692) | Switch to `embedder_backed` for real autoregressive (sprint 693, default in current runbook) |
| SSE output diverges from HF greedy reference | `SamplingDefaults.temperature` defaulting to 1.0 (sampling) | Sprint 694 fix shipped; pull main if older |
| Daemon hangs on first inference attempt | Sprint 687 F35 deadlock (sync chain executor on event-loop thread) | Already fixed in sprint 687 — pull main if older |
| Daemon OOM-cycles under concurrent inference load | Concurrent requests each load gpt2 (~500MB); peak exceeds 2GB droplet budget | Set `PRSM_INFERENCE_CONCURRENCY_LIMIT=1` (sprint 704; default in current runbook for ≤2GB nodes) |

## 9. Verify your operator's receipts externally (sprint 706+707+708)

Once the daemon serves requests, capture any signed receipt and verify
it with the standalone PRSM-import-free verifier from anywhere:

```bash
# Capture a receipt
curl -s -m 60 -X POST http://127.0.0.1:8002/compute/inference \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"Hello","model_id":"gpt2","budget_ftns":1.0,"privacy_tier":"none","content_tier":"A","max_tokens":1}' \
    | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin)['receipt'], indent=2))" \
    > my-receipt.json

# Verify it (works from any machine with web3 + cryptography installed)
pip install web3 cryptography
python3 scripts/verify_prsm_receipt.py my-receipt.json
# → ✓ VALID — receipt verifies cleanly
```

The verifier confirms (a) the settler_node_id resolves on the on-chain
`PublisherKeyAnchor`, (b) the `settler_signature` is a valid Ed25519
signature of the canonical bytes, (c) the attestation envelope parses
cleanly, and (d) `activation_noise_trace` (tier ≥ standard) integrity
holds. See `docs/sample-receipts/README.md` for 4 reference receipts
covering unary, streaming, DP, and multi-host modes.

## Related runbooks

- `bootstrap-server-oci-deployment-guide.md` — bootstrap server (not operator)
- `second-do-droplet-deploy.md` — DigitalOcean droplet provisioning (CPU)
- `lambda-gpu-operator-deploy.md` — Lambda Cloud GPU provisioning
- `bootstrap-us-operator-systemd.service.reference` — canonical co-located unit reference
- `phase-3-x-3-publisher-key-anchor-deploy-ceremony.md` — anchor contract reference

## Sprint references

| Sprint | What it shipped |
| --- | --- |
| 558 | `build_parallax_executor_or_none` + env-driven kinds |
| 560 | Production trust stack (anchor + tier gate) |
| 561 | `AnchorMediatedStakeLookup` (later superseded by sprint 690's `PoolBackedStakeLookup`) |
| 580 | `_build_anchor_or_none` shared helper |
| 585 | `prsm node section7-readiness` CLI |
| 595-606 | Phase 2 chain executor wiring (async-to-sync bridge, request handler) |
| 611-616 | Real HF model forward pass + tokenizer adapters |
| 651, 656 | Streaming wire path + KV cache |
| 675 | Generic operator pubkey registration script |
| 680-682 | DHT-backed GPU pool with hardware advertisement |
| 683, 690 | On-chain stake lookup (proper fix via pool-backed) |
| 685 | `/admin/parallax/pool/snapshot` endpoint |
| 686-694 | F30→F43 bug cluster (deadlock, dtype, position embeddings, etc.) |
| 695 | True multi-host 2-stage allocation (F44+F45) |
| 696 | `prsm node parallax-readiness` CLI |
| 697 | This runbook (initial 8-step deploy guide) |
| 698 | Lambda A10 GPU operator + F47 (prompt encoder device fix) |
| 700 | F46 monotonic hardware_profile gossip propagation |
| 702 | `PRSM_PARALLAX_TIER_GATE=advisory` env + F48/F49 (DP-injection path closure) |
| 703 | `scripts/verify_prsm_receipt.py` standalone PRSM-import-free verifier |
| 704 | `PRSM_INFERENCE_CONCURRENCY_LIMIT` semaphore (NYC OOM gate) |
| 705 | Sprint 704 live-validated under 4-concurrent load |
| 706-708 | Sample receipts in `docs/sample-receipts/` covering all 4 inference modes |
