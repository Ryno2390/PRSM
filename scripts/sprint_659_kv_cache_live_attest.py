"""Sprint 659 — KV-cache arc live-attestation.

Exercises sprints 654-658 end-to-end: stable request_id across
N iterations, INCREMENTAL decode_mode, server-side KVCacheManager
handles allocate (cold) then get (hot) so each iteration after
the first only forwards 1 new token.

Audit-chain note: this script reuses the same request_id across
tokens, which DOES trip sprint 637's C3 (DUPLICATE_REQUEST_ID)
under --check-chain. Full audit-chain integration (separate
cache_session_id field) is a follow-on sprint; this script is
the perf-measurement deliverable for the KV-cache arc.

Expected outcome: per-iteration latency drops from ~5s/token
(PREFILL full-prefix replay) to ~1s/token (INCREMENTAL single
token) once the cache is warm.
"""
from __future__ import annotations

import base64
import sys
import time

import httpx
import numpy as np


MAC_API = "http://127.0.0.1:8000"
PROMPT = "PRSM"
N_TOKENS = 5


def main() -> int:
    from prsm.compute.chain_rpc.protocol import (
        ContentTier, DecodeMode, HandoffToken, PrivacyLevel,
        RunLayerSliceRequest, encode_message, parse_message,
    )
    from prsm.node.config import NodeConfig
    from prsm.node.identity import load_node_identity
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Find stage peer
    peers = httpx.get(f"{MAC_API}/peers", timeout=5).json()
    if peers["connected_count"] < 1:
        print("✗ No connected peers")
        return 1
    stage_peer_id = peers["connected"][0]["peer_id"]
    print(f"Stage peer: {stage_peer_id}")

    settler = load_node_identity(NodeConfig.load().identity_path)

    print("Loading gpt2 on Mac...")
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", torch_dtype=torch.float32,
    ).eval()

    # Stable cache key across the run
    cache_request_id = f"sprint659-kvcache-{int(time.time())}"
    print(f"Cache request_id (stable): {cache_request_id}")

    text = PROMPT
    print(f"\nInitial prompt: {text!r}")
    print(f"Generating {N_TOKENS} tokens via INCREMENTAL decode mode")
    print("=" * 60)

    # Track token IDs explicitly to defend against BPE merges
    # (re-tokenizing the appended text doesn't always grow by +1).
    token_ids = tok.encode(text, return_tensors="pt").squeeze(0).tolist()
    total_t0 = time.time()
    for step in range(N_TOKENS):
        step_t0 = time.time()
        cur_token_count = len(token_ids)

        with torch.no_grad():
            if step == 0:
                send_ids = torch.tensor([token_ids])
                pos_offset = 0
            else:
                # Send only the last appended token
                send_ids = torch.tensor([[token_ids[-1]]])
                pos_offset = cur_token_count - 1
            te = model.transformer.wte(send_ids)
            positions = torch.arange(
                pos_offset, pos_offset + send_ids.shape[-1],
            ).unsqueeze(0)
            pe = model.transformer.wpe(positions)
            activation = (te + pe).numpy()

        deadline = time.time() + 120.0
        token = HandoffToken.sign(
            identity=settler, request_id=cache_request_id,
            chain_stage_index=0, chain_total_stages=1,
            deadline_unix=deadline,
        )
        request = RunLayerSliceRequest(
            request_id=cache_request_id,  # STABLE across the run
            model_id="gpt2",
            layer_range=(0, 12),
            privacy_tier=PrivacyLevel.NONE,
            content_tier=ContentTier.A,
            activation_blob=activation.tobytes(),
            activation_shape=tuple(activation.shape),
            activation_dtype=str(activation.dtype),
            upstream_token=token, deadline_unix=deadline,
            decode_mode=DecodeMode.INCREMENTAL,
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
            print(f"  ✗ step {step}: HTTP {r.status_code}: {r.text[:200]}")
            return 1
        resp_bytes = base64.b64decode(r.json()["response_b64"])
        resp = parse_message(resp_bytes)
        if not hasattr(resp, "activation_blob"):
            print(f"  ✗ step {step}: StageError: "
                  f"code={getattr(resp, 'code', '?')} "
                  f"message={getattr(resp, 'message', '?')[:200]}")
            return 1
        logits = np.frombuffer(
            resp.activation_blob, dtype=resp.activation_dtype,
        ).reshape(resp.activation_shape)
        next_id = int(logits[0, -1, :].argmax())
        next_token = tok.decode([next_id])
        text += next_token
        token_ids.append(next_id)
        step_dt = time.time() - step_t0
        sent_positions = send_ids.shape[-1]
        marker = "COLD" if step == 0 else "HOT "
        print(
            f"  [step {step:2d}] {marker} sent_seq_len={sent_positions:2d} "
            f"+token {next_id:6d} = {next_token!r:>12s}  ({step_dt:.2f}s)"
        )

    total_dt = time.time() - total_t0
    print("=" * 60)
    print(f"🎯 Generated ({N_TOKENS} tokens in {total_dt:.1f}s):")
    print(f"   {text!r}")
    print()
    print("Cold (step 0) vs hot (step 1+) latency comparison should")
    print("show the KV-cache speedup: hot steps forward 1 position;")
    print("cold step forwarded the full prefix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
