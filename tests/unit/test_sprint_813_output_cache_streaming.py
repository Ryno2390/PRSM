"""Sprint 813 — output cache integration on the streaming path.

Sprint 811 wired the cache into ParallaxScheduledExecutor.execute
(unary). Sprint 813 extends to execute_streaming.

The streamed receipt has `streamed_output=True` baked into the
signing-payload (Phase 3.x.8 conditional encoding). A cache hit
on the streaming path must produce a fresh receipt signed with
`streamed=True`, otherwise the signature would either be invalid
(if we tried to dataclasses.replace the unary one) or downgraded
(if we returned the unary receipt unchanged).

Sprint 813 changes:
- `_try_output_cache(*, request, cost, streamed=False)` —
  new kwarg threads through `_build_signed_receipt`. Default
  False preserves sprint 811 unary behavior.
- `_put_output_cache` stays storage-side identical (cached
  value is just `{output, tee_attestation}`, no streamed bit).
- execute_streaming:
  - On cache hit (after gates pass): yields one
    `InferenceTokenEvent` with the full cached output as
    text_delta (`finish_reason="cache_hit"`) and one terminal
    `InferenceResult` with the streamed-flagged receipt.
  - On full streaming success: synthesizes a
    `ChainExecutionResult`-shaped value from the final
    accumulated text + calls `_put_output_cache`.

Pin tests:
- `_try_output_cache` accepts `streamed` kwarg; default False.
- streamed=True → fresh receipt has `streamed_output=True`.
- streamed=True → receipt verifies under same identity (sanity
  check: the streamed flag was included in signing payload).
- streamed=False → receipt has streamed_output=False (sprint
  811 behavior preserved).
- Source-shape: execute_streaming calls
  `_try_output_cache(..., streamed=True)`.
- Source-shape: execute_streaming yields a token event then
  a terminal result on cache hit (yields BEFORE returning).
- Source-shape: execute_streaming calls `_put_output_cache`
  on the success path AFTER the chain-executor stream
  finishes.
"""
from __future__ import annotations

import inspect
from decimal import Decimal


def _make_executor_with_identity():
    from prsm.compute.inference.parallax_executor import (
        ParallaxScheduledExecutor,
    )
    from prsm.node.identity import generate_node_identity
    identity = generate_node_identity("stream-cache-test")
    exec_ = ParallaxScheduledExecutor.__new__(ParallaxScheduledExecutor)
    exec_._identity = identity
    exec_._output_cache = None
    return exec_


def _make_request(prompt="hello", privacy_tier="none", max_tokens=8):
    from prsm.compute.inference.models import (
        InferenceRequest, ContentTier,
    )
    from prsm.compute.tee.models import PrivacyLevel
    return InferenceRequest(
        request_id="r1", prompt=prompt, model_id="gpt2",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel(privacy_tier),
        budget_ftns=Decimal("1.0"), max_tokens=max_tokens,
    )


# ---- streamed kwarg on _try_output_cache ----------------------


def test_try_cache_accepts_streamed_kwarg():
    """The helper accepts streamed=True without TypeError."""
    exec_ = _make_executor_with_identity()
    # No cache wired; call with streamed=True must not raise.
    result = exec_._try_output_cache(
        request=_make_request(),
        cost=Decimal("0.5"),
        streamed=True,
    )
    assert result is None


def test_try_cache_streamed_true_returns_streamed_receipt():
    """Cache hit with streamed=True → receipt.streamed_output is True."""
    from prsm.compute.inference.output_cache import (
        OutputCache, make_cache_key,
    )
    exec_ = _make_executor_with_identity()
    exec_._output_cache = OutputCache(
        max_entries=10, ttl_seconds=3600,
    )
    req = _make_request(prompt="hi", max_tokens=8)
    key = make_cache_key("hi", "gpt2", 8, "none")
    exec_._output_cache.put(key, {
        "output": "streamed-cached",
        "tee_attestation": b"att",
    })
    result = exec_._try_output_cache(
        request=req, cost=Decimal("0.5"), streamed=True,
    )
    assert result is not None
    assert result.receipt.streamed_output is True


def test_try_cache_streamed_true_signature_verifies():
    """Receipt MUST verify — the streamed flag is part of the
    signing payload (Phase 3.x.8 conditional encoding)."""
    from prsm.compute.inference.output_cache import (
        OutputCache, make_cache_key,
    )
    from prsm.compute.inference.receipt import verify_receipt
    exec_ = _make_executor_with_identity()
    exec_._output_cache = OutputCache(
        max_entries=10, ttl_seconds=3600,
    )
    req = _make_request(prompt="hi", max_tokens=8)
    key = make_cache_key("hi", "gpt2", 8, "none")
    exec_._output_cache.put(key, {
        "output": "x", "tee_attestation": b"att",
    })
    result = exec_._try_output_cache(
        request=req, cost=Decimal("0.5"), streamed=True,
    )
    assert verify_receipt(
        result.receipt, identity=exec_._identity,
    )


def test_try_cache_default_streamed_false_preserved():
    """Sprint 811 contract preserved: default kwarg is False."""
    from prsm.compute.inference.output_cache import (
        OutputCache, make_cache_key,
    )
    exec_ = _make_executor_with_identity()
    exec_._output_cache = OutputCache(
        max_entries=10, ttl_seconds=3600,
    )
    req = _make_request(prompt="hi", max_tokens=8)
    key = make_cache_key("hi", "gpt2", 8, "none")
    exec_._output_cache.put(key, {
        "output": "x", "tee_attestation": b"att",
    })
    # No kwarg → defaults to streamed=False (sprint 811 path)
    result = exec_._try_output_cache(
        request=req, cost=Decimal("0.5"),
    )
    assert result.receipt.streamed_output is False


# ---- Source-shape on execute_streaming -----------------------


def test_execute_streaming_calls_try_cache_with_streamed_true():
    """execute_streaming must call _try_output_cache with
    streamed=True so the cached receipt sets the streamed flag
    in the signed payload (downgrade-resistance)."""
    from prsm.compute.inference.parallax_executor import (
        ParallaxScheduledExecutor,
    )
    src = inspect.getsource(
        ParallaxScheduledExecutor.execute_streaming,
    )
    # Heuristic: contains both `_try_output_cache` and
    # `streamed=True` in the same general region.
    assert "_try_output_cache" in src, (
        "Sprint 813: execute_streaming must consult output cache"
    )
    # Find the _try_output_cache call site
    idx = src.find("_try_output_cache(")
    assert idx > 0
    # streamed=True keyword must appear in the call's argument
    # block. Look in the 200 chars after the call site.
    call_block = src[idx:idx + 300]
    assert "streamed=True" in call_block, (
        "Sprint 813: execute_streaming must pass streamed=True"
    )


def test_execute_streaming_yields_token_then_result_on_hit():
    """Source-shape: the cache-hit branch yields a token event
    AND a terminal result before returning."""
    from prsm.compute.inference.parallax_executor import (
        ParallaxScheduledExecutor,
    )
    src = inspect.getsource(
        ParallaxScheduledExecutor.execute_streaming,
    )
    # Find the cache check; the lines that follow should
    # include `yield InferenceTokenEvent(` and `yield cached`
    # (or similar) before a return.
    idx = src.find("_try_output_cache")
    assert idx > 0
    cache_block = src[idx:idx + 1500]
    assert "yield InferenceTokenEvent(" in cache_block, (
        "Sprint 813: cache-hit branch must yield a token event "
        "carrying the cached output"
    )
    # And a result event
    assert (
        "yield cached" in cache_block
        or "yield InferenceResult" in cache_block
    ), (
        "Sprint 813: cache-hit branch must yield the cached "
        "result before returning"
    )


def test_execute_streaming_calls_put_cache_on_success():
    """Source-shape: after the chain-executor stream completes
    successfully, execute_streaming records the result so future
    identical requests hit cache."""
    from prsm.compute.inference.parallax_executor import (
        ParallaxScheduledExecutor,
    )
    src = inspect.getsource(
        ParallaxScheduledExecutor.execute_streaming,
    )
    assert "_put_output_cache" in src, (
        "Sprint 813: execute_streaming must call "
        "_put_output_cache on successful stream completion"
    )
