"""Sprint 828 — smoke-test auto-detects model from /compute/models.

Sprint 824 made the gpt2-vs-mock-catalog error message actionable
("Hint: pass --model <id> matching your daemon's catalog"), but
the underlying friction remained: every fresh mock-executor
operator hit the same wall + had to manually re-run with --model.
Sprint 828 prevents the wall by auto-detecting:

  - When --model is NOT passed, GET /compute/models first.
  - If gpt2 is in the catalog, use it (preserves prod default).
  - Else use the first listed model + console-log the
    substitution.
  - When --model IS passed, respect it verbatim (no auto-detect).

This composes with sprint 824: the auto-detect path produces a
valid model_id under normal conditions; sprint 824's error path
still fires on a genuinely-broken catalog response (e.g.
/compute/models 503 → resolved_model falls back to "gpt2" →
sprint 824 surfaces "Unknown model_id: gpt2" cleanly).

Pin tests:
- Default (no --model) + catalog with gpt2 → uses gpt2.
- Default (no --model) + catalog WITHOUT gpt2 → uses first
  listed model + logs the substitution.
- Default (no --model) + /compute/models unavailable → falls
  back to "gpt2" (sprint 824's error path catches it).
- Explicit --model bypasses auto-detect (catalog NOT queried).
- Explicit --model gpt2 against mock catalog still fires the
  sprint-824 actionable-error path (regression guard).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def _invoke(args=None):
    from prsm.cli import node as _node_group
    return CliRunner().invoke(
        _node_group, ["smoke-test"] + (args or []),
    )


def _success_pool_response():
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "gpu_count": 2,
        "pool_kind": "dht-backed",
    }
    return r


def _models_response(model_ids, *, as_dicts=False):
    """Live daemon returns bare strings; older fixtures used
    dicts. Default to live shape; pass as_dicts=True for the
    legacy shape."""
    r = MagicMock()
    r.status_code = 200
    if as_dicts:
        payload = [{"model_id": mid} for mid in model_ids]
    else:
        payload = list(model_ids)
    r.json.return_value = {
        "models": payload,
        "count": len(model_ids),
    }
    return r


def _success_inference_response(model_id):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "success": True,
        "output": " ok",
        "receipt": {
            "model_id": model_id,
            "output_hash": "abc",
            "settler_signature": "sig",
        },
    }
    return r


def _unknown_model_response():
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "success": False,
        "error": "Unknown model_id: gpt2",
    }
    return r


# ---- Auto-detect picks gpt2 when present ---------------------


def test_default_model_picks_gpt2_when_in_catalog():
    """No --model + catalog includes gpt2 → uses gpt2 (preserves
    prod default behavior). Auto-detection is transparent."""
    captured_bodies = []

    def fake_post(url, json=None, **kwargs):
        captured_bodies.append(json)
        return _success_inference_response("gpt2")

    def fake_get(url, **kwargs):
        if url.endswith("/admin/parallax/pool/snapshot"):
            return _success_pool_response()
        if url.endswith("/compute/models"):
            return _models_response(["gpt2", "mock-phi-3"])
        raise AssertionError(f"unexpected GET {url}")

    with patch("httpx.get", side_effect=fake_get), \
         patch("httpx.post", side_effect=fake_post):
        result = _invoke(["--format", "text"])

    assert result.exit_code == 0, result.output
    assert captured_bodies[0]["model_id"] == "gpt2"
    # No auto-detect notice when gpt2 was present
    assert "Auto-detected" not in result.output


# ---- Auto-detect picks first when gpt2 missing ---------------


def test_default_model_picks_first_when_gpt2_missing():
    """No --model + catalog missing gpt2 → picks first listed +
    logs the substitution. THE fresh-operator-mock-catalog
    fix."""
    captured_bodies = []

    def fake_post(url, json=None, **kwargs):
        captured_bodies.append(json)
        return _success_inference_response("mock-llama-3-8b")

    def fake_get(url, **kwargs):
        if url.endswith("/admin/parallax/pool/snapshot"):
            return _success_pool_response()
        if url.endswith("/compute/models"):
            return _models_response([
                "mock-llama-3-8b",
                "mock-mistral-7b",
                "mock-phi-3",
            ])
        raise AssertionError(f"unexpected GET {url}")

    with patch("httpx.get", side_effect=fake_get), \
         patch("httpx.post", side_effect=fake_post):
        result = _invoke(["--format", "text"])

    assert result.exit_code == 0, result.output
    assert captured_bodies[0]["model_id"] == "mock-llama-3-8b"
    assert "Auto-detected" in result.output
    assert "mock-llama-3-8b" in result.output


# ---- Catalog unavailable → falls back to gpt2 ----------------


def test_models_endpoint_unavailable_falls_back_to_gpt2():
    """If GET /compute/models fails, auto-detect falls back to
    gpt2 — sprint 824's error path catches a stale-catalog
    situation cleanly."""

    def fake_post(url, json=None, **kwargs):
        return _unknown_model_response()

    def fake_get(url, **kwargs):
        if url.endswith("/admin/parallax/pool/snapshot"):
            return _success_pool_response()
        if url.endswith("/compute/models"):
            raise RuntimeError("models endpoint blew up")
        raise AssertionError(f"unexpected GET {url}")

    with patch("httpx.get", side_effect=fake_get), \
         patch("httpx.post", side_effect=fake_post):
        result = _invoke(["--format", "text"])

    # Fallback model_id was gpt2 → daemon rejects → sprint 824
    # actionable-error path fires.
    assert result.exit_code == 1
    assert "Unknown model_id" in result.output


# ---- Explicit --model bypasses auto-detect -------------------


def test_explicit_model_bypasses_auto_detect():
    """When --model is explicitly passed, the smoke-test must
    NOT GET /compute/models (operator's intent is authoritative).
    """
    get_urls = []
    captured_bodies = []

    def fake_post(url, json=None, **kwargs):
        captured_bodies.append(json)
        return _success_inference_response("mock-phi-3")

    def fake_get(url, **kwargs):
        get_urls.append(url)
        if url.endswith("/admin/parallax/pool/snapshot"):
            return _success_pool_response()
        # If we hit /compute/models, that's a bug — bypass
        # should have prevented it.
        raise AssertionError(f"models lookup should be skipped {url}")

    with patch("httpx.get", side_effect=fake_get), \
         patch("httpx.post", side_effect=fake_post):
        result = _invoke([
            "--model", "mock-phi-3", "--format", "text",
        ])

    assert result.exit_code == 0, result.output
    assert captured_bodies[0]["model_id"] == "mock-phi-3"
    # /compute/models was NOT in the GET urls
    assert not any(
        "/compute/models" in u for u in get_urls
    )


# ---- Sprint 824 regression-guard composability ---------------


def test_auto_detect_handles_dict_shape_legacy():
    """Defense-in-depth: if some older daemon/test surface still
    returns models as list of dicts, auto-detect MUST still
    work. The live wire format is bare strings (sprint 828
    primary path); this is the legacy fallback."""
    captured_bodies = []

    def fake_post(url, json=None, **kwargs):
        captured_bodies.append(json)
        return _success_inference_response("mock-phi-3")

    def fake_get(url, **kwargs):
        if url.endswith("/admin/parallax/pool/snapshot"):
            return _success_pool_response()
        if url.endswith("/compute/models"):
            return _models_response(
                ["mock-phi-3", "mock-llama-3-8b"],
                as_dicts=True,
            )
        raise AssertionError(f"unexpected GET {url}")

    with patch("httpx.get", side_effect=fake_get), \
         patch("httpx.post", side_effect=fake_post):
        result = _invoke(["--format", "text"])

    assert result.exit_code == 0, result.output
    assert captured_bodies[0]["model_id"] == "mock-phi-3"


def test_explicit_gpt2_against_mock_still_surfaces_sprint_824():
    """When an operator explicitly passes --model gpt2 against a
    mock catalog, sprint-824's actionable-error path MUST still
    fire (regression guard). Sprint 828 doesn't change explicit
    paths, but pin the relationship so the two sprints compose
    intentionally."""

    def fake_post(url, json=None, **kwargs):
        return _unknown_model_response()

    def fake_get(url, **kwargs):
        if url.endswith("/admin/parallax/pool/snapshot"):
            return _success_pool_response()
        raise AssertionError(f"models lookup should be skipped {url}")

    with patch("httpx.get", side_effect=fake_get), \
         patch("httpx.post", side_effect=fake_post):
        result = _invoke([
            "--model", "gpt2", "--format", "text",
        ])

    assert result.exit_code == 1
    assert "Unknown model_id" in result.output
    # Sprint 824's --model hint also still surfaces
    assert "--model" in result.output
