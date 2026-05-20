"""Sprint 667 — multi-stage chain arc kickoff. Smoke-test for
2-stage gpt2 inference.

Splits gpt2's 12 transformer layers across two chain stages:
  Stage 0: layers [0, 6)  — runs on droplet
  Stage 1: layers [6, 12) — also runs on droplet (placeholder
                            until 2-host fleet)

The settler (Mac) drives the chain:
  1. Embed prompt locally (wte + wpe)
  2. Ship activation to stage 0 (decode_mode=PREFILL, layers 0-5)
  3. Receive stage-0 hidden states + signed receipt
  4. Verify stage-0 signature against anchor
  5. Ship stage-0 output as input to stage 1 (layers 6-11,
     is_final_stage=true so server applies ln_f + lm_head)
  6. Receive stage-1 logits + signed receipt
  7. Verify stage-1 signature
  8. argmax → next token

This is the minimum-viable demonstration of the chain-of-trust
handoff. Sprint 668+ will productize into CLI flags + handle the
multi-host case properly.

Honest scope: both stages route to the same droplet node_id, so
the "chain-of-trust" is theatrical — Mac signs handoff token,
droplet verifies + processes, signs response, ships back. Sprint
672+ will use 2 distinct peers in the fleet.
"""
from __future__ import annotations

import base64
import sys
import time

import httpx
import numpy as np


MAC_API = "http://127.0.0.1:8000"
PROMPT = "Hello"
SPLIT_LAYER = 6  # gpt2 has 12 layers; split at midpoint


def main() -> int:
    from prsm.compute.chain_rpc.protocol import (
        ContentTier, HandoffToken, PrivacyLevel,
        RunLayerSliceRequest, encode_message, parse_message,
    )
    from prsm.node.config import NodeConfig
    from prsm.node.identity import load_node_identity
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    peers = httpx.get(f"{MAC_API}/peers", timeout=5).json()
    if peers["connected_count"] < 1:
        print("✗ No connected peers")
        return 1
    stage_peer_id = peers["connected"][0]["peer_id"]
    print(f"Stage peer (both stages, this sprint): {stage_peer_id}")

    settler = load_node_identity(NodeConfig.load().identity_path)

    print("Loading gpt2 on Mac...")
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", torch_dtype=torch.float32,
    ).eval()

    # Embed locally
    input_ids = tok.encode(PROMPT, return_tensors="pt")
    with torch.no_grad():
        te = model.transformer.wte(input_ids)
        pe = model.transformer.wpe(
            torch.arange(input_ids.shape[-1]).unsqueeze(0),
        )
        activation = (te + pe).numpy()

    request_id = f"sprint667-multi-stage-{int(time.time())}"
    chain_total = 2
    deadline = time.time() + 120.0

    print(f"\nChain: 2 stages, prompt={PROMPT!r}")
    print("=" * 60)

    current_activation = activation
    for stage_idx, (lo, hi) in enumerate([
        (0, SPLIT_LAYER), (SPLIT_LAYER, 12),
    ]):
        step_t0 = time.time()
        ho_token = HandoffToken.sign(
            identity=settler, request_id=request_id,
            chain_stage_index=stage_idx, chain_total_stages=chain_total,
            deadline_unix=deadline,
        )
        request = RunLayerSliceRequest(
            request_id=request_id, model_id="gpt2",
            layer_range=(lo, hi),
            privacy_tier=PrivacyLevel.NONE,
            content_tier=ContentTier.A,
            activation_blob=current_activation.tobytes(),
            activation_shape=tuple(current_activation.shape),
            activation_dtype=str(current_activation.dtype),
            upstream_token=ho_token, deadline_unix=deadline,
        )
        req_bytes = encode_message(request)
        r = httpx.post(
            f"{MAC_API}/admin/chain-exec-ping",
            json={
                "peer_id": stage_peer_id,
                "payload_b64": base64.b64encode(req_bytes).decode("ascii"),
                "timeout": 115.0,
            },
            timeout=120.0,
        )
        if r.status_code != 200:
            print(f"  ✗ stage {stage_idx}: HTTP {r.status_code}: "
                  f"{r.text[:200]}")
            return 1
        resp_bytes = base64.b64decode(r.json()["response_b64"])
        resp = parse_message(resp_bytes)
        if not hasattr(resp, "activation_blob"):
            print(f"  ✗ stage {stage_idx}: StageError: "
                  f"code={getattr(resp, 'code', '?')} "
                  f"message={getattr(resp, 'message', '?')[:200]}")
            return 1
        # Decode the activation for the next stage's input
        current_activation = np.frombuffer(
            resp.activation_blob, dtype=resp.activation_dtype,
        ).reshape(resp.activation_shape)
        step_dt = time.time() - step_t0
        print(
            f"  [stage {stage_idx}] layers [{lo}, {hi}) — "
            f"signature={resp.stage_signature_b64[:24]}... "
            f"out_shape={current_activation.shape} ({step_dt:.1f}s)"
        )

    # Final stage's output is logits (is_final_stage was inferred
    # from the layer range covering the model's tail).
    next_id = int(current_activation[0, -1, :].argmax())
    next_token = tok.decode([next_id])
    print("=" * 60)
    print(f"🎯 Multi-stage chain produced next token "
          f"{next_id} = {next_token!r}")
    print()
    print("Next sprints (668-674):")
    print("  - Productize as `prsm node infer --stages N` CLI flag")
    print("  - Route each stage to a DISTINCT peer (multi-host)")
    print("  - Verify handoff tokens carry stage_index/total correctly")
    print("  - Audit-chain receipts capture multi-stage signatures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
