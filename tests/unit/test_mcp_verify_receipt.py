"""Sprint 241 — prsm_pubkey + prsm_verify_receipt MCP tools.

End-users running prsm_inference receive an InferenceReceipt with
settler_signature + settler_node_id. Pre-fix there was no MCP
path to verify the signature — and no HTTP endpoint to fetch
the settler's pubkey either. This sprint adds both.

prsm_pubkey: GET /node/identity/pubkey wrapper.
prsm_verify_receipt: takes a receipt dict + optional
public_key_b64, returns verified/not-verified with the verified
fields displayed. When public_key_b64 omitted, fetches from
the running node — useful when the running node IS the settler.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS,
    handle_prsm_pubkey,
    handle_prsm_verify_receipt,
)


class TestRegistration:
    def test_pubkey_registered(self):
        assert "prsm_pubkey" in TOOL_HANDLERS
        assert TOOL_HANDLERS["prsm_pubkey"] is handle_prsm_pubkey

    def test_verify_receipt_registered(self):
        assert "prsm_verify_receipt" in TOOL_HANDLERS
        assert TOOL_HANDLERS["prsm_verify_receipt"] is handle_prsm_verify_receipt


class TestPubkey:
    @pytest.mark.asyncio
    async def test_renders(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "node_id": "settler-7",
                "public_key_b64": "BTLPdMcuFdK2GuoMxxv+Z+G8bRPCYqyR6Bt/X8FldhE=",
            }),
        ) as mock_call:
            result = await handle_prsm_pubkey({})
        args, _ = mock_call.await_args
        assert args[0] == "GET"
        assert args[1] == "/node/identity/pubkey"
        assert "settler-7" in result
        assert "BTLPdMcuFd" in result


class TestVerifyReceiptValidation:
    @pytest.mark.asyncio
    async def test_missing_receipt_rejected(self):
        result = await handle_prsm_verify_receipt({})
        assert "receipt" in result.lower()

    @pytest.mark.asyncio
    async def test_receipt_must_be_dict(self):
        result = await handle_prsm_verify_receipt({
            "receipt": "not a dict",
        })
        assert "receipt" in result.lower()


class TestVerifyReceiptHappyPath:
    @pytest.mark.asyncio
    async def test_signature_verified_with_supplied_pubkey(self):
        """Build a real receipt + sign it + pass to the MCP tool
        with the matching pubkey. Expect SIGNATURE VALID."""
        from prsm.node.identity import generate_node_identity
        from prsm.compute.inference.models import (
            ContentTier, InferenceReceipt,
        )
        from prsm.compute.inference.receipt import sign_receipt
        from prsm.compute.tee.models import PrivacyLevel, TEEType
        from decimal import Decimal

        identity = generate_node_identity()
        unsigned = InferenceReceipt(
            job_id="job-1",
            request_id="req-1",
            model_id="mock-llama-3-8b",
            privacy_tier=PrivacyLevel.STANDARD,
            content_tier=ContentTier.A,
            tee_type=TEEType.SOFTWARE,
            epsilon_spent=8.0,
            cost_ftns=Decimal("0.10"),
            duration_seconds=1.0,
            output_hash=b"\x00" * 32,
            tee_attestation=b"",
        )
        signed = sign_receipt(unsigned, identity)

        result = await handle_prsm_verify_receipt({
            "receipt": signed.to_dict(),
            "public_key_b64": identity.public_key_b64,
        })
        assert "valid" in result.lower()
        assert "job-1" in result
        assert "mock-llama-3-8b" in result

    @pytest.mark.asyncio
    async def test_tampered_payload_rejected(self):
        from prsm.node.identity import generate_node_identity
        from prsm.compute.inference.models import (
            ContentTier, InferenceReceipt,
        )
        from prsm.compute.inference.receipt import sign_receipt
        from prsm.compute.tee.models import PrivacyLevel, TEEType
        from decimal import Decimal

        identity = generate_node_identity()
        signed = sign_receipt(
            InferenceReceipt(
                job_id="job-1",
                request_id="req-1",
                model_id="mock-llama-3-8b",
                privacy_tier=PrivacyLevel.STANDARD,
                content_tier=ContentTier.A,
                tee_type=TEEType.SOFTWARE,
                epsilon_spent=8.0,
                cost_ftns=Decimal("0.10"),
                duration_seconds=1.0,
                output_hash=b"\x00" * 32,
                tee_attestation=b"",
            ),
            identity,
        )
        tampered = signed.to_dict()
        tampered["model_id"] = "evil-impostor-model"

        result = await handle_prsm_verify_receipt({
            "receipt": tampered,
            "public_key_b64": identity.public_key_b64,
        })
        assert "invalid" in result.lower() or "failed" in result.lower()


class TestVerifyReceiptFetchPubkey:
    @pytest.mark.asyncio
    async def test_fetches_pubkey_from_node_when_omitted(self):
        """Without public_key_b64, the tool calls /node/identity/
        pubkey on the running node and verifies against that."""
        from prsm.node.identity import generate_node_identity
        from prsm.compute.inference.models import (
            ContentTier, InferenceReceipt,
        )
        from prsm.compute.inference.receipt import sign_receipt
        from prsm.compute.tee.models import PrivacyLevel, TEEType
        from decimal import Decimal

        identity = generate_node_identity()
        signed = sign_receipt(
            InferenceReceipt(
                job_id="job-1",
                request_id="req-1",
                model_id="m1",
                privacy_tier=PrivacyLevel.STANDARD,
                content_tier=ContentTier.A,
                tee_type=TEEType.SOFTWARE,
                epsilon_spent=0.0,
                cost_ftns=Decimal("0.10"),
                duration_seconds=1.0,
                output_hash=b"\x00" * 32,
                tee_attestation=b"",
            ),
            identity,
        )
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "node_id": signed.settler_node_id,
                "public_key_b64": identity.public_key_b64,
            }),
        ) as mock_call:
            result = await handle_prsm_verify_receipt({
                "receipt": signed.to_dict(),
            })
        args, _ = mock_call.await_args
        assert args[1] == "/node/identity/pubkey"
        assert "valid" in result.lower()
