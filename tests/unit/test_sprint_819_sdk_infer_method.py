"""Sprint 819 — PRSMClient.infer() for verifiable inference.

The `prsm.sdk.client.PRSMClient` had `.query()` (forge pipeline)
+ `.prompt()` (legacy NWTN) but NO method for /compute/inference
(the verifiable-inference path that produces signed receipts).

Python developers wanting to integrate verifiable inference had
to either fall back to httpx + JSON or shell out to the CLI.
Sprint 819 ships a first-class SDK method.

  async def infer(
      self, prompt: str, *,
      model_id: str = "gpt2",
      max_tokens: int = 8,
      budget_ftns: float = 1.0,
      privacy_tier: str = "none",
      content_tier: str = "A",
      verify_pubkey_b64: Optional[str] = None,
  ) -> Dict[str, Any]:
      \"\"\"POST /compute/inference; return parsed payload.

      When verify_pubkey_b64 is set, the returned payload
      includes a `receipt_verified` boolean computed via
      sprint-706 verify_receipt against the supplied pubkey.
      \"\"\"

Pin tests:
- PRSMClient has .infer method.
- infer POSTs to /compute/inference (NOT /compute/query).
- Body has canonical 6-field shape.
- Default values match the CLI defaults.
- Overrides pass through.
- Returns parsed JSON dict.
- With verify_pubkey_b64 + valid receipt → receipt_verified=True.
- With verify_pubkey_b64 + wrong pubkey → receipt_verified=False.
- Without verify_pubkey_b64 → no receipt_verified field.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


pytestmark = pytest.mark.asyncio


def _make_signed_receipt_dict():
    """Build a real signed InferenceReceipt + return as dict +
    matching public_key_b64."""
    from prsm.compute.inference.models import (
        InferenceReceipt, ContentTier,
    )
    from prsm.compute.tee.models import PrivacyLevel, TEEType
    from prsm.compute.inference.receipt import sign_receipt
    from prsm.node.identity import generate_node_identity

    identity = generate_node_identity("sdk-test")
    r = InferenceReceipt(
        job_id="j1", request_id="r1", model_id="gpt2",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=b"att", output_hash=b"\xde\xad",
        duration_seconds=1.0, cost_ftns=Decimal("0.5"),
        settler_node_id="",
    )
    signed = sign_receipt(r, identity)
    return signed.to_dict(), identity.public_key_b64


# ---- Method exists -------------------------------------------


async def test_infer_method_exists():
    from prsm.sdk.client import PRSMClient
    client = PRSMClient()
    assert hasattr(client, "infer")
    assert callable(client.infer)


# ---- POSTs to /compute/inference -----------------------------


async def test_infer_posts_to_compute_inference():
    from prsm.sdk.client import PRSMClient
    client = PRSMClient()
    # Patch _post to a coroutine returning a stub payload
    expected_payload = {
        "success": True, "output": " the",
        "ftns_charged": "0.25", "receipt": {},
    }
    with patch.object(
        client, "_post",
        new=AsyncMock(return_value=expected_payload),
    ) as mp:
        result = await client.infer(prompt="Hi")
    mp.assert_called_once()
    path, body = mp.call_args.args[0], mp.call_args.args[1]
    assert path == "/compute/inference"
    assert result == expected_payload


# ---- Body shape ----------------------------------------------


async def test_infer_body_canonical_shape():
    from prsm.sdk.client import PRSMClient
    client = PRSMClient()
    with patch.object(
        client, "_post",
        new=AsyncMock(return_value={"success": True}),
    ) as mp:
        await client.infer(prompt="Hi")
    body = mp.call_args.args[1]
    assert "prompt" in body
    assert "model_id" in body
    assert "budget_ftns" in body
    assert "privacy_tier" in body
    assert "content_tier" in body
    assert "max_tokens" in body
    # Wrong field names (regression guard against typos)
    assert "model" not in body
    assert "budget" not in body
    assert "privacy" not in body


async def test_infer_defaults_match_cli():
    """SDK defaults match `prsm compute infer` (sprint 802) so
    users moving between CLI and SDK see consistent behavior."""
    from prsm.sdk.client import PRSMClient
    client = PRSMClient()
    with patch.object(
        client, "_post",
        new=AsyncMock(return_value={"success": True}),
    ) as mp:
        await client.infer(prompt="Hi")
    body = mp.call_args.args[1]
    assert body["model_id"] == "gpt2"
    assert body["content_tier"] == "A"
    assert body["privacy_tier"] == "none"
    assert body["budget_ftns"] == 1.0
    assert body["max_tokens"] == 8


async def test_infer_overrides_pass_through():
    from prsm.sdk.client import PRSMClient
    client = PRSMClient()
    with patch.object(
        client, "_post",
        new=AsyncMock(return_value={"success": True}),
    ) as mp:
        await client.infer(
            prompt="Capital of France?",
            model_id="llama-7b",
            max_tokens=50,
            budget_ftns=3.5,
            privacy_tier="standard",
            content_tier="B",
        )
    body = mp.call_args.args[1]
    assert body["prompt"] == "Capital of France?"
    assert body["model_id"] == "llama-7b"
    assert body["max_tokens"] == 50
    assert body["budget_ftns"] == 3.5
    assert body["privacy_tier"] == "standard"
    assert body["content_tier"] == "B"


# ---- Receipt verification (optional) -------------------------


async def test_infer_with_verify_pubkey_passes():
    from prsm.sdk.client import PRSMClient
    receipt_dict, pubkey_b64 = _make_signed_receipt_dict()
    client = PRSMClient()
    with patch.object(
        client, "_post",
        new=AsyncMock(return_value={
            "success": True, "output": " the",
            "receipt": receipt_dict,
        }),
    ):
        result = await client.infer(
            prompt="Hi", verify_pubkey_b64=pubkey_b64,
        )
    assert result.get("receipt_verified") is True


async def test_infer_with_wrong_pubkey_fails_verify():
    from prsm.sdk.client import PRSMClient
    from prsm.node.identity import generate_node_identity
    receipt_dict, _ = _make_signed_receipt_dict()
    wrong_pubkey = generate_node_identity("other").public_key_b64
    client = PRSMClient()
    with patch.object(
        client, "_post",
        new=AsyncMock(return_value={
            "success": True, "output": " the",
            "receipt": receipt_dict,
        }),
    ):
        result = await client.infer(
            prompt="Hi", verify_pubkey_b64=wrong_pubkey,
        )
    assert result.get("receipt_verified") is False


async def test_infer_without_verify_no_field_added():
    """Default path: no receipt_verified field added to payload."""
    from prsm.sdk.client import PRSMClient
    client = PRSMClient()
    with patch.object(
        client, "_post",
        new=AsyncMock(return_value={
            "success": True, "output": " the",
            "receipt": {"settler_signature": "0xabc"},
        }),
    ):
        result = await client.infer(prompt="Hi")
    assert "receipt_verified" not in result
