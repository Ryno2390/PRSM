"""Sprint 694 F43 fix — streaming defaults to greedy (deterministic).

Sprint 693 wired EmbedderBackedStreamingRunner with the default
SamplingDefaults() which has temperature=1.0 (sampling). Live-
attest showed 4-token output `" the capital of France"` where
the first 3 tokens matched HF reference byte-for-byte but the
4th diverged (` France` vs reference ` the`) — sampling, not a
wiring bug. The receipt's `output_hash` was non-deterministic
across runs.

Sprint 694 makes streaming default to greedy by constructing
`SamplingDefaults(temperature=0.0)` in the embedder_backed
branch. EmbedderBackedStreamingRunner already maps `temperature
== 0 → do_sample=False`. Operators can still override via
`request.temperature`.

This brings streaming receipt determinism to parity with the
unary path (sprint 688's bit-identical-to-reference win).
"""
from __future__ import annotations

import inspect

import pytest


def test_embedder_runner_constructed_with_greedy_default_in_source():
    """Pin: sprint 694 fix must construct
    SamplingDefaults(temperature=0.0) in the embedder_backed
    branch — without it, streaming output is non-deterministic
    (sprint 693 F43)."""
    from prsm.node.chain_executor_adapters import (
        build_layer_stage_server_executor,
    )
    src = inspect.getsource(build_layer_stage_server_executor)
    assert "temperature=0.0" in src, (
        "embedder_backed streaming must default to greedy "
        "(SamplingDefaults(temperature=0.0)) so receipts are "
        "deterministic and match HF reference byte-for-byte"
    )


def test_sampling_defaults_zero_temperature_triggers_greedy():
    """SamplingDefaults(temperature=0.0) is a valid construction
    and represents greedy — defensive check on the dataclass
    contract relied upon by sprint 694."""
    from prsm.compute.inference.autoregressive_runner import (
        SamplingDefaults,
    )
    s = SamplingDefaults(temperature=0.0)
    assert s.temperature == 0.0
    # Runner contract: temperature == 0 → do_sample=False
    # We don't run the runner here (needs HF model), just verify
    # the data passes through correctly. The runner's behavior
    # is covered by EmbedderBackedStreamingRunner tests directly.
    assert s.max_tokens == 512
    assert s.top_k == 50
    assert s.top_p == 0.95


def test_sprint_694_marker_present():
    """Pin sprint 694's fix marker in the source so future
    refactors that touch this region keep the F43 context."""
    from prsm.node.chain_executor_adapters import (
        build_layer_stage_server_executor,
    )
    src = inspect.getsource(build_layer_stage_server_executor)
    assert "Sprint 694" in src or "sprint 694" in src
    assert "F43" in src
