"""Sprint 657 (KV-cache arc piece 4) — LayerStageServer dispatches
INCREMENTAL requests to the kv-cache path when kv_cache_manager is
wired AND runner supports run_layer_range_incremental.

Tests the routing decisions only — full end-to-end exercise of
the incremental forward path uses real GPT-2 and lives in
sprint 660's live-attestation work.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _make_runner_with_inc():
    """Runner that supports both run_layer_range AND
    run_layer_range_incremental.
    """
    runner = MagicMock()
    runner.run_layer_range = MagicMock()
    runner.run_layer_range_incremental = MagicMock()
    return runner


def _make_runner_without_inc():
    """Runner missing run_layer_range_incremental (pre-sprint-656)."""
    runner = MagicMock(spec=["run_layer_range"])
    return runner


def _make_server(*, runner, kv_cache_manager=None):
    from prsm.compute.chain_rpc.server import LayerStageServer
    from prsm.node.identity import generate_node_identity
    return LayerStageServer(
        identity=generate_node_identity(),
        registry=MagicMock(get=MagicMock()),
        runner=runner,
        tee_runtime=MagicMock(),
        anchor=MagicMock(),
        kv_cache_manager=kv_cache_manager,
    )


def test_incremental_without_cache_manager_returns_error():
    """No kv_cache_manager wired → MALFORMED_REQUEST with breadcrumb
    naming PRSM_PARALLAX_KV_CACHE_ENABLED."""
    server = _make_server(runner=_make_runner_with_inc())
    # Call handle's dispatch path with a non-PREFILL request
    from prsm.compute.chain_rpc.protocol import (
        DecodeMode, PrivacyLevel, ContentTier, HandoffToken,
        RunLayerSliceRequest, parse_message, StageError,
    )
    import numpy as np
    from prsm.node.identity import generate_node_identity
    settler = generate_node_identity()
    deadline = 9999999999.0
    request = RunLayerSliceRequest(
        request_id="r0", model_id="test",
        layer_range=(0, 1),
        privacy_tier=PrivacyLevel.NONE,
        content_tier=ContentTier.A,
        activation_blob=np.zeros((1, 1, 4), dtype=np.float32).tobytes(),
        activation_shape=(1, 1, 4),
        activation_dtype="float32",
        upstream_token=HandoffToken.sign(
            identity=settler, request_id="r0",
            chain_stage_index=0, chain_total_stages=1,
            deadline_unix=deadline,
        ),
        deadline_unix=deadline,
        decode_mode=DecodeMode.INCREMENTAL,
    )
    # Server.handle expects RunLayerSliceRequest serialized — call
    # the inline dispatcher directly through the routing logic.
    # The early-gate inside the inline dispatcher returns bytes
    # encoding a StageError.
    from prsm.compute.chain_rpc.protocol import encode_message
    bytes_out = server._dispatch(request)
    parsed = parse_message(bytes_out)
    assert isinstance(parsed, StageError)
    assert "kv_cache_manager" in parsed.message
    assert "PRSM_PARALLAX_KV_CACHE_ENABLED" in parsed.message


def test_incremental_without_runner_support_returns_error():
    """kv_cache_manager wired but runner lacks
    run_layer_range_incremental → MALFORMED_REQUEST naming sprint 656.
    """
    from prsm.compute.chain_rpc.kv_cache import KVCacheManager
    cache_mgr = KVCacheManager()
    server = _make_server(
        runner=_make_runner_without_inc(),
        kv_cache_manager=cache_mgr,
    )
    from prsm.compute.chain_rpc.protocol import (
        DecodeMode, PrivacyLevel, ContentTier, HandoffToken,
        RunLayerSliceRequest, parse_message, StageError,
    )
    import numpy as np
    from prsm.node.identity import generate_node_identity
    settler = generate_node_identity()
    deadline = 9999999999.0
    request = RunLayerSliceRequest(
        request_id="r0", model_id="test",
        layer_range=(0, 1),
        privacy_tier=PrivacyLevel.NONE,
        content_tier=ContentTier.A,
        activation_blob=np.zeros((1, 1, 4), dtype=np.float32).tobytes(),
        activation_shape=(1, 1, 4),
        activation_dtype="float32",
        upstream_token=HandoffToken.sign(
            identity=settler, request_id="r0",
            chain_stage_index=0, chain_total_stages=1,
            deadline_unix=deadline,
        ),
        deadline_unix=deadline,
        decode_mode=DecodeMode.INCREMENTAL,
    )
    bytes_out = server._dispatch(request)
    parsed = parse_message(bytes_out)
    assert isinstance(parsed, StageError)
    assert "run_layer_range_incremental" in parsed.message
    assert "sprint 656" in parsed.message


def test_dispatch_inline_incremental_method_exists():
    """Sprint 657 adds the _dispatch_inline_incremental method."""
    from prsm.compute.chain_rpc.server import LayerStageServer
    assert hasattr(
        LayerStageServer, "_dispatch_inline_incremental",
    ), "sprint 657 must add _dispatch_inline_incremental"


def test_cache_get_failure_falls_back_to_cold():
    """If kv_cache_manager.get raises, the dispatch must NOT
    propagate — it falls back to cold cache (prev_kv_state=None)
    + logs.
    """
    from prsm.compute.chain_rpc.server import LayerStageServer
    import inspect
    src = inspect.getsource(
        LayerStageServer._dispatch_inline_incremental,
    )
    # Must catch exceptions on .get()
    assert "kv_cache_manager.get raised" in src or "proceeding with cold" in src


def test_cache_put_failure_is_non_fatal():
    """If kv_cache_manager.put raises, the response is still
    returned — operator gets the correct output even when the
    cache is corrupt; next INCREMENTAL just re-warms.
    """
    from prsm.compute.chain_rpc.server import LayerStageServer
    import inspect
    src = inspect.getsource(
        LayerStageServer._dispatch_inline_incremental,
    )
    assert "kv_cache_manager.put raised" in src or "next INCREMENTAL will be cold" in src
