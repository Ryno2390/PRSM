"""Sprint 628 — multi-token generation loop through PRSM production trust-stack.

Real demo: prompt → 10 generated tokens via P2P inference. Each iteration
re-embeds the full text-so-far + ships to droplet + receives logits + samples
greedy argmax → next token → append. No KV-cache exploited (sprint 629+
could wire INCREMENTAL via the sprint-618 KVCacheManager).

Each iteration ~3-5s on warm droplet. 10 tokens ~30-60s total wall time.

Live proof-of-concept: a multi-token completion produced by a remote node
with on-chain anchor verification at every cryptographic gate.
"""
from __future__ import annotations

import base64
import sys
import time

import httpx
import numpy as np


MAC_API = "http://127.0.0.1:8000"
DROPLET_NODE_ID = "484f003c895ee02ac7ed01e570a6a51f"
PROMPT = "The capital of France is"
N_TOKENS = 10


def main() -> int:
    from prsm.node.config import NodeConfig
    from prsm.node.identity import load_node_identity
    from prsm.compute.chain_rpc.protocol import (
        HandoffToken, RunLayerSliceRequest, parse_message, encode_message,
        PrivacyLevel, ContentTier,
    )
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    settler = load_node_identity(NodeConfig.load().identity_path)
    print(f"Mac (settler) node_id: {settler.node_id}")
    print(f"Droplet (stage) node_id: {DROPLET_NODE_ID}")
    print(f"\nLoading gpt2 on Mac for tokenize + embed...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", torch_dtype=torch.float32,
    ).eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    text = PROMPT
    print(f"\n{'=' * 60}")
    print(f"INITIAL PROMPT: {text!r}")
    print(f"Generating {N_TOKENS} tokens via PRSM P2P inference...")
    print(f"{'=' * 60}\n")

    overall_t0 = time.time()
    for step in range(N_TOKENS):
        step_t0 = time.time()
        # Re-tokenize the full text-so-far
        input_ids = tok.encode(text, return_tensors="pt")
        with torch.no_grad():
            te = model.transformer.wte(input_ids)
            pe = model.transformer.wpe(
                torch.arange(input_ids.shape[-1]).unsqueeze(0),
            )
            activation = (te + pe).numpy()

        # Build signed request — fresh request_id + handoff token each step
        request_id = f"sprint628-step{step}-{int(time.time())}"
        deadline = time.time() + 120.0
        token = HandoffToken.sign(
            identity=settler, request_id=request_id,
            chain_stage_index=0, chain_total_stages=1,
            deadline_unix=deadline,
        )
        request = RunLayerSliceRequest(
            request_id=request_id, model_id="gpt2",
            layer_range=(0, 12),
            privacy_tier=PrivacyLevel.NONE,
            content_tier=ContentTier.A,
            activation_blob=activation.tobytes(),
            activation_shape=tuple(activation.shape),
            activation_dtype=str(activation.dtype),
            upstream_token=token, deadline_unix=deadline,
        )
        req_bytes = encode_message(request)

        # Ship via Mac's chain-exec-ping
        r = httpx.post(
            f"{MAC_API}/admin/chain-exec-ping",
            json={
                "peer_id": DROPLET_NODE_ID,
                "payload_b64": base64.b64encode(req_bytes).decode("ascii"),
                "timeout": 115.0,
            },
            timeout=120.0,
        )
        if r.status_code != 200:
            print(f"  [step {step}] HTTP {r.status_code}: {r.text[:200]}")
            return 1

        # Parse response, argmax, decode
        resp_bytes = base64.b64decode(r.json()["response_b64"])
        resp = parse_message(resp_bytes)
        if not hasattr(resp, "activation_blob"):
            print(f"  [step {step}] StageError: {getattr(resp, 'message', '?')}")
            return 1
        logits = np.frombuffer(
            resp.activation_blob, dtype=resp.activation_dtype,
        ).reshape(resp.activation_shape)
        next_id = int(logits[0, -1, :].argmax())
        next_token = tok.decode([next_id])
        text += next_token
        step_dt = time.time() - step_t0
        print(f"  [step {step:2d}] +token {next_id:5d} = {next_token!r:>12s}  "
              f"({step_dt:.1f}s, seq_len={input_ids.shape[-1]} → "
              f"{input_ids.shape[-1] + 1})")

    overall_dt = time.time() - overall_t0
    print(f"\n{'=' * 60}")
    print(f"🎯 GENERATED ({N_TOKENS} tokens in {overall_dt:.1f}s):")
    print(f"   {text!r}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
