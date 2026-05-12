"""Sprint 313 — HTTP stage runner factory.

Sprint 312 ran all pipeline stages in the orchestrator's
process. This module ships the bridge that lets the
orchestrator call REMOTE stage workers over HTTP. Each
remote stage worker exposes
`POST /compute/inference/pipeline/stage` (see api.py); the
factory here returns a sync `StageRunner` (sprint 312
Protocol) that POSTs to a configured stage worker URL.

v1 is orchestrator-driven: every stage's output round-trips
through the orchestrator. Worker-to-worker chaining (stages
directly call the next stage's URL) is a follow-on
sprint — saves bandwidth but adds coordination complexity
not warranted for v1.

The factory accepts an injected `http_post` callable so
tests can route through a FastAPI TestClient without
spinning up a real HTTP server. Production callers leave
it as the default (httpx-based).
"""
from __future__ import annotations

import base64
from typing import Any, Callable, List, Optional


class HTTPStageRunnerError(Exception):
    """Raised when the remote stage worker returns a
    non-2xx response or the response shape is malformed.
    The orchestrator's `execute()` propagates this up and
    marks the round FAILED."""


def _default_http_post(
    url: str, *, json: dict, timeout: float,
):
    """Default httpx-based POST. Imports httpx lazily so
    tests that inject their own http_post don't pay the
    import cost."""
    import httpx
    return httpx.post(url, json=json, timeout=timeout)


def http_stage_runner(
    *,
    stage_node_url: str,
    job_id: str,
    round_id: str,
    http_post: Optional[Callable[..., Any]] = None,
    request_timeout: float = 60.0,
):
    """Build a sprint-312 `StageRunner` that POSTs each
    stage call to a remote stage worker.

    The orchestrator calls the returned function with
    `(input_activations, stage_id, layer_indices)`; the
    function POSTs `{job_id, round_id, stage_id,
    layer_indices, input_activations_b64}` to
    `{stage_node_url}/compute/inference/pipeline/stage`
    and returns the decoded `output_activations_b64` from
    the response.

    http_post: injectable for testing (default httpx).
    """
    post = http_post if http_post is not None else (
        _default_http_post
    )

    def _runner(
        *,
        input_activations: bytes,
        stage_id: int,
        layer_indices: List[int],
    ) -> bytes:
        url = (
            stage_node_url.rstrip("/")
            + "/compute/inference/pipeline/stage"
        )
        body = {
            "job_id": job_id,
            "round_id": round_id,
            "stage_id": int(stage_id),
            "layer_indices": list(layer_indices),
            "input_activations_b64": base64.b64encode(
                input_activations,
            ).decode("ascii"),
        }
        try:
            response = post(
                url, json=body, timeout=request_timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPStageRunnerError(
                f"HTTP POST to stage worker "
                f"{stage_node_url!r} failed: "
                f"{type(exc).__name__}: {exc}"
            )
        # FastAPI TestClient and httpx both expose
        # status_code + .json() in the same shape
        status = getattr(response, "status_code", None)
        if status is None or status >= 400:
            try:
                detail = response.json().get(
                    "detail", "(no detail)",
                )
            except Exception:  # noqa: BLE001
                detail = "(no detail)"
            raise HTTPStageRunnerError(
                f"stage worker {stage_node_url!r} returned "
                f"{status}: {detail}"
            )
        try:
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPStageRunnerError(
                f"stage worker {stage_node_url!r} returned "
                f"non-JSON response: "
                f"{type(exc).__name__}: {exc}"
            )
        out_b64 = payload.get("output_activations_b64")
        if not isinstance(out_b64, str):
            raise HTTPStageRunnerError(
                f"stage worker {stage_node_url!r} response "
                f"missing 'output_activations_b64' field"
            )
        try:
            return base64.b64decode(out_b64, validate=True)
        except Exception as exc:  # noqa: BLE001
            raise HTTPStageRunnerError(
                f"stage worker {stage_node_url!r} returned "
                f"invalid base64: {exc}"
            )

    return _runner
