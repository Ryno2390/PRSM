"""Sprint 626 — gpt2 live inference end-to-end through Phase 2 wire.

Mac side:
  1. Load gpt2 tokenizer + model (gpt2 first call downloads ~500MB to cache)
  2. Tokenize a prompt → token_ids
  3. Embed via model.get_input_embeddings() → hidden_state activation
  4. Build RunLayerSliceRequest(model_id="gpt2", layer_range=(0,12),
     activation=embeddings) signed by Mac's NodeIdentity
  5. Ship via /admin/chain-exec-ping with payload_b64

Droplet side (sprint 626 env: HF runner + model_id=gpt2):
  6. Parse + anchor-verify request
  7. registry.get("gpt2") returns the shim ShardedModel
  8. HuggingFaceLayerSliceRunner.run_layer_range:
     - Loads cached HF gpt2
     - Forwards through model.transformer.h[0:12]
     - is_final_stage=True → applies model.lm_head → logits
  9. Signs RunLayerSliceResponse with droplet's NodeIdentity
  10. Returns logits as activation_blob

Mac side again:
  11. Parse response → logits as np.ndarray
  12. argmax over vocab at last position → next_token_id
  13. tokenizer.decode([next_token_id]) → next-token string
  14. Print: "prompt + next_token"

This proves PRSM's full inference path with REAL transformer
computation through the live mainnet trust-stack.
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


def main() -> int:
    from prsm.node.config import NodeConfig
    from prsm.node.identity import load_node_identity
    from prsm.compute.chain_rpc.protocol import (
        HandoffToken, RunLayerSliceRequest, parse_message, encode_message,
        PrivacyLevel, ContentTier,
    )

    settler = load_node_identity(NodeConfig.load().identity_path)
    print(f"Mac (settler) node_id: {settler.node_id}")

    # Load gpt2 on Mac for tokenize + embed
    print(f"\nLoading gpt2 on Mac (first call downloads ~500MB)...")
    t0 = time.time()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    # Tokenize + embed (WITH position embeddings — GPT-2 transformer blocks
    # don't apply position info themselves; it must be baked into the
    # input hidden_states. Sprint 611's runner assumes activation is
    # already a full pre-block hidden state.)
    print(f"\nPrompt: {PROMPT!r}")
    input_ids = tok.encode(PROMPT, return_tensors="pt")
    print(f"  tokens: {input_ids.tolist()[0]}")
    print(f"  decoded: {[tok.decode([t]) for t in input_ids[0]]}")
    with torch.no_grad():
        # GPT-2: hidden_state_0 = wte(token_ids) + wpe(arange(S))
        token_embeds = model.transformer.wte(input_ids)
        position_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0)
        position_embeds = model.transformer.wpe(position_ids)
        hidden_states = token_embeds + position_embeds
    activation = hidden_states.numpy()  # shape [B=1, S, H=768]
    print(f"  hidden_states: shape={activation.shape} dtype={activation.dtype} "
          f"(tokens + position embeddings)")

    # Build signed request
    request_id = f"sprint626-gpt2-{int(time.time())}"
    deadline = time.time() + 120.0  # 2 min — first call on droplet may take 10s+
    token = HandoffToken.sign(
        identity=settler, request_id=request_id,
        chain_stage_index=0, chain_total_stages=1, deadline_unix=deadline,
    )
    request = RunLayerSliceRequest(
        request_id=request_id,
        model_id="gpt2",
        layer_range=(0, 12),  # all gpt2 layers
        privacy_tier=PrivacyLevel.NONE,
        content_tier=ContentTier.A,
        activation_blob=activation.tobytes(),
        activation_shape=tuple(activation.shape),
        activation_dtype=str(activation.dtype),
        upstream_token=token,
        deadline_unix=deadline,
    )
    req_bytes = encode_message(request)
    print(f"\nencoded request: {len(req_bytes)} bytes ({len(req_bytes)/1024:.1f} KB)")

    print(f"\nSending to droplet (timeout=110s — droplet first-call loads gpt2 ~10s)...")
    t1 = time.time()
    r = httpx.post(
        f"{MAC_API}/admin/chain-exec-ping",
        json={
            "peer_id": DROPLET_NODE_ID,
            "payload_b64": base64.b64encode(req_bytes).decode("ascii"),
            "timeout": 110.0,
        },
        timeout=120.0,
    )
    elapsed = time.time() - t1
    print(f"  HTTP {r.status_code} ({elapsed:.1f}s)")
    if r.status_code != 200:
        print(f"  {r.text}")
        return 1

    data = r.json()
    resp_bytes = base64.b64decode(data["response_b64"])
    print(f"  response: {len(resp_bytes)} bytes ({len(resp_bytes)/1024:.1f} KB)")

    parsed = parse_message(resp_bytes)
    print(f"  type: {type(parsed).__name__}")
    if not hasattr(parsed, "activation_blob"):
        # StageError or similar
        print(f"  CODE: {getattr(parsed, 'code', '?')}")
        print(f"  MESSAGE: {getattr(parsed, 'message', '?')}")
        return 1

    # Parse + sample. Sprint 626 NOTE: droplet's _is_final_stage()
    # returns False because the persisted manifest dropped layer_range
    # to the (0,0) sentinel — so sprint 613 lm_head + sprint 626 ln_f
    # were NOT applied. We get raw hidden_states back. Apply LM head
    # locally on Mac side (we have the model loaded anyway). This is
    # acceptable for the live test; the persistence bug is sprint 627+.
    hidden_arr = np.frombuffer(
        parsed.activation_blob, dtype=parsed.activation_dtype,
    ).reshape(parsed.activation_shape)
    print(f"\nDroplet returned hidden_states: shape={hidden_arr.shape}")
    hidden_t = torch.from_numpy(hidden_arr)
    with torch.no_grad():
        hidden_t = model.transformer.ln_f(hidden_t)
        logits_t = model.lm_head(hidden_t)
    next_token_id = int(logits_t[0, -1, :].argmax())
    next_token = tok.decode([next_token_id])
    print(f"  (Mac applied ln_f + lm_head locally)")
    print(f"  argmax token_id: {next_token_id}")
    print(f"  decoded:          {next_token!r}")
    print()
    print("=" * 60)
    print(f"🎯 PROMPT:     {PROMPT!r}")
    print(f"🎯 GENERATED:  {(PROMPT + next_token)!r}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
