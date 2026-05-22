# PRSM sample receipts

Real signed `InferenceReceipt` JSON files captured from the live PRSM
operator fleet during sprints 685–705. Each one was returned from
a real `/compute/inference` POST against a mainnet-anchored operator
(NYC `484f003c…` on `bootstrap-us.prsm-network.com`).

These receipts let an external party verify PRSM's chain-of-custody
claim end-to-end without spinning up any infrastructure: clone the
repo, run the standalone verifier, see `✓ VALID`. Tamper any byte,
re-run, see `✗ INVALID — settler_signature does NOT verify`.

## Files

| Receipt | Tier | Features exercised |
| --- | --- | --- |
| `tier-none-unary-2026-05-22.json` | none | Baseline unary inference; output_hash `fe0663fd…` is bit-identical to local HuggingFace `gpt2` greedy decode for the same prompt (sprint 688's semantic-correctness milestone) |
| `tier-none-streaming-2026-05-22.json` | none | SSE streaming with `streamed_output: true`; output_hash `1b46bc86…` covers 4-token autoregressive decode of "The capital of France is" → " the capital of the" (sprint 694's bit-identical-to-HF streaming) |
| `tier-standard-dp-2026-05-22.json` | standard (ε=8.0) | Activation-DP injection (`activation_noise_trace` field populated) + per-stage TEE attestation envelope + signed by NYC's mainnet-anchored Ed25519 key |
| `multi-host-2stage-2026-05-22.json` | none (multi-host) | **True 2-stage cross-WAN allocation** — NYC (`484f003c…`, layers 0-5) + SFO (`d437aa67…`, layers 6-11). `topology_assignment.stage_count=2`, attestation envelope carries per-stage hex for BOTH operators, settler_signature commits over the topology_assignment hash. Output `output_hash: cdb4ee2a…` matches sprint 695's milestone. |

## Verifying a sample

```bash
pip install web3 cryptography
python3 scripts/verify_prsm_receipt.py docs/sample-receipts/tier-standard-dp-2026-05-22.json
```

Expected output:

```
{
  "valid": true,
  "anchor_lookup": "vObS7F/i3NpC7ZkWNSwGARvHwLdBzBtNuwq7q6cQZJQ=",
  "signature_valid": true,
  "attestation_envelope": {"version": 1, "stage_count": 1, ...},
  "noise_trace": {"tier": "standard", "total_epsilon_spent": 8.0, ...},
  "findings": []
}
✓ VALID — receipt verifies cleanly
```

The `anchor_lookup` value is the base64-encoded Ed25519 public key
that the on-chain `PublisherKeyAnchor` contract (Base mainnet
`0xd811ad9986f44f404b0fd992168a7cc76206df03`) returned for the
settler's `node_id`. Verify the contract address on Basescan
independently if you want zero PRSM-team trust.

## Proving tamper-detection

```bash
# Copy + corrupt one byte of output_hash
python3 -c "
import json
r = json.load(open('docs/sample-receipts/tier-standard-dp-2026-05-22.json'))
old = r['output_hash']
r['output_hash'] = old[:-2] + ('00' if old[-2:] != '00' else 'ff')
json.dump(r, open('/tmp/tampered.json', 'w'))
print('flipped last byte of output_hash')
"

python3 scripts/verify_prsm_receipt.py /tmp/tampered.json
# Expected: ✗ INVALID — settler_signature does NOT verify
```

Tampering ANY field included in the canonical signing payload
invalidates the signature. See `scripts/verify_prsm_receipt.py:
_build_signing_payload` for the exact field order — it matches
`prsm/compute/inference/models.py:InferenceReceipt.signing_payload`
byte-for-byte (defended by sprint 703 pin tests).

## Capturing your own

To capture a fresh receipt from any PRSM operator node:

```bash
curl -s -m 600 -X POST http://<operator-host>:<port>/compute/inference \
    -H 'Content-Type: application/json' \
    -d '{
      "prompt": "Hello",
      "model_id": "gpt2",
      "budget_ftns": 1.0,
      "privacy_tier": "none",
      "content_tier": "A",
      "max_tokens": 1
    }' \
    | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin)['receipt'], indent=2))" \
    > my-receipt.json

python3 scripts/verify_prsm_receipt.py my-receipt.json
```

If the operator's pubkey is anchor-registered, you'll see
`✓ VALID`. If not (their pubkey isn't on-chain), the verifier
returns `settler_node_id … not registered on anchor`.
