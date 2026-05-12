"""Sprint 314 — real PyTorch per-stage forward pass.

Wires PyTorch into the sprint 312 StageRunner Protocol.
The factory takes an `nn.Sequential` model factory; the
returned runner slices the model's children by
layer_indices for each stage call, runs the input tensor
through them, and returns the output activations as
bytes.

Activation wire format: JSON envelope {shape, dtype,
data_b64} — same shape semantics as sprint 310's
training-data envelope but for a SINGLE tensor (not a
features/labels pair). Self-describing, language-
agnostic, no pickle.

The model is constructed ONCE per StageRunner instance
(on the first call) and cached for subsequent stage
invocations — otherwise per-stage compute would repeat
model construction. Operators wanting to refresh weights
build a new runner.
"""
from __future__ import annotations

import base64
import json
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn


# ── dtype mapping (mirrors sprint 310 surface) ─────


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


# ── Activation serialization ───────────────────────


def serialize_activation(t: torch.Tensor) -> bytes:
    """Pack a single tensor into a JSON envelope. Output
    is plaintext bytes suitable for direct HTTP transport
    (sprint 313) or wrapping in sprint 304 recipient
    encryption."""
    if not isinstance(t, torch.Tensor):
        raise TypeError(
            f"activation must be torch.Tensor, got "
            f"{type(t).__name__}"
        )
    if t.dtype not in _DTYPE_TO_STR:
        raise ValueError(
            f"tensor dtype {t.dtype} not supported"
        )
    contig = t.detach().cpu().contiguous()
    raw = contig.numpy().tobytes()
    envelope = {
        "shape": list(contig.shape),
        "dtype": _DTYPE_TO_STR[contig.dtype],
        "data_b64": base64.b64encode(raw).decode("ascii"),
    }
    return json.dumps(envelope).encode("utf-8")


def deserialize_activation(blob: bytes) -> torch.Tensor:
    try:
        envelope = json.loads(blob)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(
            f"activation envelope not valid JSON: {e}"
        )
    if not isinstance(envelope, dict):
        raise ValueError(
            "activation envelope must be a JSON object"
        )
    shape = envelope.get("shape")
    dtype_str = envelope.get("dtype")
    data_b64 = envelope.get("data_b64")
    if not isinstance(shape, list) or not all(
        isinstance(x, int) for x in shape
    ):
        raise ValueError(
            f"shape must be a list of ints; got {shape!r}"
        )
    if dtype_str not in _STR_TO_DTYPE:
        raise ValueError(
            f"unsupported dtype {dtype_str!r}"
        )
    if not isinstance(data_b64, str):
        raise ValueError(
            "data_b64 must be a base64 string"
        )
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except Exception as e:
        raise ValueError(f"data_b64 not valid base64: {e}")
    import numpy as _np
    np_dtype = _np.dtype(dtype_str)
    arr = _np.frombuffer(raw, dtype=np_dtype).copy()
    tensor = torch.from_numpy(arr).to(
        _STR_TO_DTYPE[dtype_str],
    )
    if shape:
        tensor = tensor.reshape(*shape)
    return tensor


# ── pytorch_stage_runner factory ───────────────────


def pytorch_stage_runner(
    *,
    model_factory: Callable[[], nn.Module],
):
    """Build a StageRunner that runs a slice of the
    provided nn.Sequential model on each call. The model
    is constructed once on first invocation + cached for
    the lifetime of the returned runner.

    The runner's signature matches sprint 312's
    StageRunner Protocol: `(input_activations: bytes,
    stage_id: int, layer_indices: List[int]) -> bytes`.

    layer_indices selects which CHILDREN of the
    nn.Sequential are owned by this stage. The runner
    builds a sub-Sequential from those children and
    forwards the deserialized input tensor through it.
    """
    cached_model: Dict[str, nn.Module] = {}

    def _runner(
        *,
        input_activations: bytes,
        stage_id: int,
        layer_indices: List[int],
    ) -> bytes:
        if "model" not in cached_model:
            cached_model["model"] = model_factory()
        model = cached_model["model"]
        children = list(model.children())
        # Bounds check upfront so we raise a clear error
        # before reshaping or computing anything
        for idx in layer_indices:
            if idx < 0 or idx >= len(children):
                raise IndexError(
                    f"layer index {idx} out of range for "
                    f"model with {len(children)} children"
                )
        stage_layers = nn.Sequential(*[
            children[i] for i in layer_indices
        ])
        input_tensor = deserialize_activation(
            input_activations,
        )
        with torch.no_grad():
            output_tensor = stage_layers(input_tensor)
        return serialize_activation(output_tensor)

    return _runner
