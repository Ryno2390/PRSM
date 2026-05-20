"""Sprint 624 Step A — send signed RunLayerSliceRequest from Mac to droplet
for an UNREGISTERED model_id. Expect: structured MODEL_NOT_FOUND
response from droplet's LayerStageServer.

This proves end-to-end:
  1. Mac constructs + signs HandoffToken with its NodeIdentity
  2. Mac builds RunLayerSliceRequest with the signed token
  3. Bytes ship via WS-P2P chain_executor_rpc wire (sprints 596-599)
  4. Droplet's LayerStageServer parses (sprint 606+ wrapper +
     chain_rpc.server)
  5. Droplet verifies upstream_token.settler_node_id pubkey via LIVE
     PublisherKeyAnchor on Base mainnet
  6. Droplet's registry.get("non-existent-model") raises ModelNotFoundError
  7. LayerStageServer signs the structured error response with its
     NodeIdentity
  8. Bytes ship back via WS-P2P
  9. Mac receives + parses the signed error response

If we see a MODEL_NOT_FOUND-style error in the response, the FULL
trust-stack path is exercised against the live Phase 3.x.3 anchor.
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys
import time

import httpx


MAC_API = "http://127.0.0.1:8000"
DROPLET_NODE_ID = "484f003c895ee02ac7ed01e570a6a51f"


def main() -> int:
    from cryptography.hazmat.primitives import serialization
    from prsm.node.config import NodeConfig
    from prsm.node.identity import load_node_identity
    from prsm.compute.chain_rpc.protocol import (
        HandoffToken, RunLayerSliceRequest, encode_message, parse_message,
        PrivacyLevel, ContentTier,
    )
    import numpy as np

    cfg = NodeConfig.load()
    settler = load_node_identity(cfg.identity_path)
    print(f"Mac (settler) node_id: {settler.node_id}")

    request_id = f"sprint624-step-a-{int(time.time())}"
    deadline = time.time() + 30.0
    print(f"request_id: {request_id}")

    # Sign HandoffToken — Mac is the settler at stage 0 of a 1-stage chain.
    token = HandoffToken.sign(
        identity=settler,
        request_id=request_id,
        chain_stage_index=0,
        chain_total_stages=1,
        deadline_unix=deadline,
    )

    # Fake 1-dim activation tensor: 4 floats, [1, 1, 4] shape.
    activation = np.zeros((1, 1, 4), dtype=np.float32)
    activation_blob = activation.tobytes()

    request = RunLayerSliceRequest(
        request_id=request_id,
        model_id="sprint625-identity-test",
        layer_range=(0, 1),
        privacy_tier=PrivacyLevel.NONE,
        content_tier=ContentTier.A,
        activation_blob=activation_blob,
        activation_shape=tuple(activation.shape),
        activation_dtype=str(activation.dtype),
        upstream_token=token,
        deadline_unix=deadline,
    )

    request_bytes = encode_message(request)
    print(f"encoded request: {len(request_bytes)} bytes")

    # Send via Mac's /admin/chain-exec-ping with payload_b64 (sprint 624
    # endpoint extension supports arbitrary binary payloads).
    response = httpx.post(
        f"{MAC_API}/admin/chain-exec-ping",
        json={
            "peer_id": DROPLET_NODE_ID,
            "payload_b64": base64.b64encode(request_bytes).decode("ascii"),
            "timeout": 25.0,
        },
        timeout=30.0,
    )

    if response.status_code != 200:
        print(f"HTTP {response.status_code}: {response.text}")
        return 1

    data = response.json()
    print(f"\n=== response ===")
    print(f"  size_bytes: {data.get('size_bytes')}")
    if data.get("response_b64"):
        resp_bytes = base64.b64decode(data["response_b64"])
        print(f"  raw bytes (first 100): {resp_bytes[:100]!r}")

        # Parse via chain_rpc.protocol
        try:
            parsed = parse_message(resp_bytes)
            print(f"\n  parsed type: {type(parsed).__name__}")
            if hasattr(parsed, "code"):
                print(f"  error code: {parsed.code}")
                print(f"  error message: {getattr(parsed,'message','-')}")
        except Exception as exc:
            print(f"  parse failed: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
