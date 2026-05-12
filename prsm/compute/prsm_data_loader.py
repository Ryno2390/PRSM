"""Sprint 310 — PRSM content-layer DataLoader for FL.

Sprint 309's `DataLoader` Protocol let operators wire any
function `dataset_cid -> (features, labels)` into the
PyTorch training loop. The default `synthetic_data_loader`
produced reproducible test data; real deployments needed
operators to write their own integration with PRSM's
content + decryption layers.

This module ships that integration. The wire:
  - `content_provider.request_content(cid)` fetches the
    raw blob from PRSM's content layer (the same path
    `/content/retrieve/{cid}` uses)
  - The blob is an EncryptedPayload (sprint 304 shape)
  - Worker's X25519 recipient privkey decrypts inside
    the TEE
  - Plaintext is the JSON envelope produced by
    `serialize_training_data` — self-describing
    {features, labels} pair with shape, dtype, base64
    bytes

OR-decrypt only in v1 (one worker per shard is the
standard FL pattern). Threshold mode (sprint 307)
doesn't fit the FL DataLoader model and is deliberately
out of scope here — workers act unilaterally.

The data envelope is JSON-wrapped, not pickled. Pickle
on untrusted input is a remote-code-execution surface;
explicit shape + dtype + base64-bytes is auditable and
language-agnostic.
"""
from __future__ import annotations

import base64
import json
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

import torch

from prsm.enterprise.recipient_encryption import (
    EncryptedPayload,
    decrypt_for_recipient,
)


# ── Errors ──────────────────────────────────────────


class DataDeserializationError(Exception):
    """Raised when an EncryptedPayload decrypts to bytes
    that aren't a valid training-data envelope."""


# ── dtype mapping ───────────────────────────────────


# Supported torch dtypes for v1. Complex / quantized
# tensors aren't supported; refuse loud upfront so we
# don't produce a wire-compatible-looking blob that the
# consumer can't actually load.
_DTYPE_TO_STR: Dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}
_STR_TO_DTYPE: Dict[str, torch.dtype] = {
    v: k for k, v in _DTYPE_TO_STR.items()
}


# ── Serialize / deserialize ─────────────────────────


def _tensor_to_dict(t: torch.Tensor) -> Dict[str, Any]:
    if t.dtype not in _DTYPE_TO_STR:
        raise ValueError(
            f"tensor dtype {t.dtype} not supported for "
            f"FL data serialization; supported: "
            f"{sorted(_DTYPE_TO_STR.values())}"
        )
    # Contiguous so the byte view is well-defined
    contig = t.detach().cpu().contiguous()
    raw = contig.numpy().tobytes()
    return {
        "shape": list(contig.shape),
        "dtype": _DTYPE_TO_STR[contig.dtype],
        "data_b64": base64.b64encode(raw).decode("ascii"),
    }


def _tensor_from_dict(
    d: Dict[str, Any],
) -> torch.Tensor:
    if not isinstance(d, dict):
        raise DataDeserializationError(
            f"tensor envelope must be a dict, got "
            f"{type(d).__name__}"
        )
    shape = d.get("shape")
    dtype_str = d.get("dtype")
    data_b64 = d.get("data_b64")
    if (
        not isinstance(shape, list)
        or not all(isinstance(x, int) for x in shape)
    ):
        raise DataDeserializationError(
            f"shape must be a list of ints; got "
            f"{shape!r}"
        )
    if dtype_str not in _STR_TO_DTYPE:
        raise DataDeserializationError(
            f"unsupported dtype {dtype_str!r}; supported: "
            f"{sorted(_STR_TO_DTYPE)}"
        )
    if not isinstance(data_b64, str):
        raise DataDeserializationError(
            "data_b64 must be a base64 string"
        )
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except Exception as e:
        raise DataDeserializationError(
            f"data_b64 not valid base64: {e}"
        )
    dtype = _STR_TO_DTYPE[dtype_str]
    # torch.frombuffer needs a writable buffer; copy from
    # the raw bytes via numpy → torch
    import numpy as _np
    np_dtype = _np.dtype(dtype_str)
    arr = _np.frombuffer(raw, dtype=np_dtype).copy()
    tensor = torch.from_numpy(arr).to(dtype)
    if shape:
        try:
            tensor = tensor.reshape(*shape)
        except RuntimeError as e:
            raise DataDeserializationError(
                f"reshape to {shape} failed: {e}"
            )
    return tensor


def serialize_training_data(
    features: torch.Tensor, labels: torch.Tensor,
) -> bytes:
    """Pack (features, labels) into a JSON envelope. The
    output is plaintext bytes ready to be encrypted via
    sprint 304's encrypt_for_recipients."""
    if not isinstance(features, torch.Tensor):
        raise TypeError(
            f"features must be torch.Tensor, got "
            f"{type(features).__name__}"
        )
    if not isinstance(labels, torch.Tensor):
        raise TypeError(
            f"labels must be torch.Tensor, got "
            f"{type(labels).__name__}"
        )
    envelope = {
        "features": _tensor_to_dict(features),
        "labels": _tensor_to_dict(labels),
    }
    return json.dumps(envelope).encode("utf-8")


def deserialize_training_data(
    plaintext: bytes,
) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        envelope = json.loads(plaintext)
    except (json.JSONDecodeError, TypeError) as e:
        raise DataDeserializationError(
            f"plaintext is not valid JSON: {e}"
        )
    if not isinstance(envelope, dict):
        raise DataDeserializationError(
            f"envelope must be a JSON object; got "
            f"{type(envelope).__name__}"
        )
    if "features" not in envelope or "labels" not in envelope:
        raise DataDeserializationError(
            "envelope missing required 'features' or "
            "'labels' field"
        )
    features = _tensor_from_dict(envelope["features"])
    labels = _tensor_from_dict(envelope["labels"])
    return features, labels


# ── DataLoader factory ──────────────────────────────


class _ContentProviderProtocol:
    """Structural typing reference — anything with this
    method shape works. PRSM's content_provider (the same
    object /content/retrieve uses) satisfies it; tests
    can pass any stub that conforms."""

    async def request_content(
        self,
        *,
        cid: str,
        timeout: Optional[float] = None,
        verify_hash: bool = True,
    ) -> bytes: ...


def prsm_content_data_loader_async(
    *,
    content_provider: Any,
    recipient_privkey_b64: str,
    request_timeout: float = 30.0,
) -> Callable[[str], Awaitable[Tuple[torch.Tensor, torch.Tensor]]]:
    """Build an async DataLoader bound to
    (content_provider, recipient_privkey). Workers calling
    from a native-async context use this directly;
    sprint-309 PyTorch training (sync) uses
    `prsm_content_data_loader` (the sync wrapper) instead.
    """

    async def loader(
        dataset_cid: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Fetch raw bundle from PRSM content layer.
        # Exceptions propagate verbatim — callers learn
        # whether the shard is missing, the layer is down,
        # or hash verification failed.
        raw = await content_provider.request_content(
            cid=dataset_cid,
            timeout=request_timeout,
            verify_hash=True,
        )
        # Parse as EncryptedPayload. Sprint 304's wire
        # format is JSON-of-dict. If the blob isn't a
        # valid bundle, this raises — the layer above
        # surfaces the error.
        try:
            payload = EncryptedPayload.from_dict(
                json.loads(raw),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise DataDeserializationError(
                f"content at cid {dataset_cid!r} is not "
                f"a valid encrypted bundle: {e}"
            )
        # Decrypt via OR-decrypt (sprint 304). Threshold
        # mode out of scope for FL DataLoader.
        plaintext = decrypt_for_recipient(
            payload, recipient_privkey_b64,
        )
        # Deserialize to torch tensors.
        return deserialize_training_data(plaintext)

    return loader


def prsm_content_data_loader(
    *,
    content_provider: Any,
    recipient_privkey_b64: str,
    request_timeout: float = 30.0,
) -> Callable[[str], Tuple[torch.Tensor, torch.Tensor]]:
    """Sync DataLoader compatible with the sprint-309
    PyTorch TrainingFn (which calls data_loader
    synchronously inside the training loop). Internally
    drives the async fetch via a fresh event loop per call.

    If the caller is already in an event loop (e.g.,
    pytest-asyncio), use `prsm_content_data_loader_async`
    directly to avoid the can't-call-asyncio.run-from-loop
    error.
    """
    import asyncio

    async_loader = prsm_content_data_loader_async(
        content_provider=content_provider,
        recipient_privkey_b64=recipient_privkey_b64,
        request_timeout=request_timeout,
    )

    def sync_loader(
        dataset_cid: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Use asyncio.run for a fresh loop per call.
        # PyTorch training is the dominant cost, so the
        # per-call loop creation overhead is negligible.
        return asyncio.run(async_loader(dataset_cid))

    return sync_loader
