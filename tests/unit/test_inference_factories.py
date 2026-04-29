"""Phase 3.x.10.x Task 4 — unit tests for production factories.

Covers ``make_autoregressive_streaming_runner`` (closes M5
production-wiring deferral from Phase 3.x.10) +
``make_layer_stage_server(streaming_runner=...)`` extension.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pytest

from prsm.compute.inference import (
    AutoregressiveStreamingRunner,
    SamplingDefaults,
    SyntheticStreamingRunner,
    make_autoregressive_streaming_runner,
)
from prsm.compute.tee.models import PrivacyLevel, TEEType


# ──────────────────────────────────────────────────────────────────────────
# Minimal HF-shaped fakes (no real distilgpt2 needed for factory smoke).
# ──────────────────────────────────────────────────────────────────────────


class _FakeTokenizer:
    eos_token_id = 99

    def encode(self, text: str) -> List[int]:
        return [1, 2]

    def decode(self, ids, skip_special_tokens=True) -> str:
        return "".join({1: "h", 2: "i"}.get(int(i), "") for i in ids)


class _FakeModel:
    def generate(
        self, *, input_ids, streamer, max_new_tokens, temperature,
        do_sample, top_k, top_p, eos_token_id,
    ):
        if hasattr(input_ids, "tolist"):
            as_list = input_ids.tolist()
            if isinstance(as_list[0], list):
                as_list = as_list[0]
            input_ids = as_list
        # Mirror HF behavior: prompt's input_ids go through
        # streamer.put() first (Phase 3.x.10.y Task 1).
        if input_ids:
            streamer.put(list(input_ids))
        # Emit one token then end.
        streamer.put(1)
        streamer.end()
        return list(input_ids) + [1]


def _prompt_provider(
    layer_range: Tuple[int, int],
    activation: np.ndarray,
    privacy_tier: PrivacyLevel,
) -> str:
    return "smoke"


# ──────────────────────────────────────────────────────────────────────────
# make_autoregressive_streaming_runner
# ──────────────────────────────────────────────────────────────────────────


class TestMakeAutoregressiveStreamingRunner:
    def test_factory_builds_a_runner(self):
        runner = make_autoregressive_streaming_runner(
            model=_FakeModel(),
            tokenizer=_FakeTokenizer(),
            tee_attestation=b"\x05" * 32,
            prompt_provider=_prompt_provider,
        )
        assert isinstance(runner, AutoregressiveStreamingRunner)

    def test_factory_smoke_runs_one_token_decode(self):
        # End-to-end: construct via factory, dispatch one tail
        # decode, assert at least one StreamingChunk emitted.
        runner = make_autoregressive_streaming_runner(
            model=_FakeModel(),
            tokenizer=_FakeTokenizer(),
            tee_attestation=b"\x05" * 32,
            prompt_provider=_prompt_provider,
            sampling_defaults=SamplingDefaults(max_tokens=4),
        )
        chunks = list(runner.run_layer_slice_streaming(
            model=None,
            layer_range=(0, 1),
            activation=np.zeros((1, 1), dtype=np.float32),
            privacy_tier=PrivacyLevel.NONE,
            is_final_stage=True,
        ))
        assert len(chunks) >= 1
        terminal = chunks[-1]
        assert terminal.finish_reason in {"stop", "max_tokens"}
        assert terminal.tee_attestation == b"\x05" * 32
        assert terminal.tee_type == TEEType.SOFTWARE

    def test_factory_rejects_missing_model(self):
        with pytest.raises(RuntimeError, match="model"):
            make_autoregressive_streaming_runner(
                model=None,
                tokenizer=_FakeTokenizer(),
                tee_attestation=b"\x05" * 32,
                prompt_provider=_prompt_provider,
            )

    def test_factory_rejects_missing_tokenizer(self):
        with pytest.raises(RuntimeError, match="tokenizer"):
            make_autoregressive_streaming_runner(
                model=_FakeModel(),
                tokenizer=None,
                tee_attestation=b"\x05" * 32,
                prompt_provider=_prompt_provider,
            )

    def test_factory_rejects_non_bytes_attestation(self):
        with pytest.raises(RuntimeError, match="tee_attestation"):
            make_autoregressive_streaming_runner(
                model=_FakeModel(),
                tokenizer=_FakeTokenizer(),
                tee_attestation="not-bytes",  # type: ignore[arg-type]
                prompt_provider=_prompt_provider,
            )

    def test_factory_rejects_non_callable_prompt_provider(self):
        with pytest.raises(RuntimeError, match="prompt_provider"):
            make_autoregressive_streaming_runner(
                model=_FakeModel(),
                tokenizer=_FakeTokenizer(),
                tee_attestation=b"\x05" * 32,
                prompt_provider=None,  # type: ignore[arg-type]
            )

    def test_factory_passes_through_sampling_defaults(self):
        defaults = SamplingDefaults(
            max_tokens=8, temperature=0.5, top_k=20, top_p=0.85,
        )
        runner = make_autoregressive_streaming_runner(
            model=_FakeModel(),
            tokenizer=_FakeTokenizer(),
            tee_attestation=b"\x05" * 32,
            prompt_provider=_prompt_provider,
            sampling_defaults=defaults,
        )
        # Defaults are private; verify by exercising a dispatch
        # and checking the resolver fell back to them.
        assert runner._defaults is defaults  # noqa: SLF001

    def test_factory_passes_through_tee_type(self):
        runner = make_autoregressive_streaming_runner(
            model=_FakeModel(),
            tokenizer=_FakeTokenizer(),
            tee_attestation=b"\x05" * 32,
            prompt_provider=_prompt_provider,
            tee_type=TEEType.SGX,
        )
        chunks = list(runner.run_layer_slice_streaming(
            model=None,
            layer_range=(0, 1),
            activation=np.zeros((1, 1), dtype=np.float32),
            privacy_tier=PrivacyLevel.NONE,
            is_final_stage=True,
        ))
        # Terminal chunk carries the operator-supplied tee_type.
        assert chunks[-1].tee_type == TEEType.SGX


# ──────────────────────────────────────────────────────────────────────────
# make_layer_stage_server(streaming_runner=...) extension
# ──────────────────────────────────────────────────────────────────────────


class TestMakeLayerStageServerStreamingExtension:
    """The factory now accepts an optional streaming_runner kwarg.
    Default None preserves back-compat (server rejects token-stream
    requests with 'not configured for streaming'); set value wires
    the streaming runner in."""

    def _identity_anchor(self):
        from prsm.node.identity import generate_node_identity
        identity = generate_node_identity("stage")

        class _Anchor:
            def lookup(self, node_id):
                if node_id == identity.node_id:
                    return identity.public_key_b64
                return None

        return identity, _Anchor()

    def _stub_runner(self):
        # Minimal LayerSliceRunner stub.
        from prsm.compute.chain_rpc.server import (
            LayerSliceResult, LayerSliceRunner,
        )

        class _Stub(LayerSliceRunner):
            def run_layer_range(
                self, *, model, layer_range, activation, privacy_tier,
                is_final_stage,
            ) -> LayerSliceResult:
                return LayerSliceResult(
                    output=activation.copy(),
                    duration_seconds=0.001,
                    tee_attestation=b"\x07" * 32,
                    tee_type=TEEType.SOFTWARE,
                    epsilon_spent=0.0,
                )

        return _Stub()

    def test_factory_back_compat_no_streaming_runner(self):
        # Default: streaming_runner=None — back-compat preserved
        # for operators not opting into streaming.
        from prsm.compute.chain_rpc import make_layer_stage_server

        identity, anchor = self._identity_anchor()

        class _TEERuntime:
            tee_type = TEEType.SOFTWARE

        class _Registry:
            def get(self, model_id):
                raise KeyError(model_id)

        server = make_layer_stage_server(
            identity=identity,
            registry=_Registry(),
            runner=self._stub_runner(),
            tee_runtime=_TEERuntime(),
            anchor=anchor,
        )
        # Server constructed; no streaming_runner.
        assert server is not None
        assert server._streaming_runner is None  # noqa: SLF001

    def test_factory_with_streaming_runner_wires_it_in(self):
        from prsm.compute.chain_rpc import make_layer_stage_server

        identity, anchor = self._identity_anchor()
        runner = make_autoregressive_streaming_runner(
            model=_FakeModel(),
            tokenizer=_FakeTokenizer(),
            tee_attestation=b"\x05" * 32,
            prompt_provider=_prompt_provider,
        )

        class _TEERuntime:
            tee_type = TEEType.SOFTWARE

        class _Registry:
            def get(self, model_id):
                raise KeyError(model_id)

        server = make_layer_stage_server(
            identity=identity,
            registry=_Registry(),
            runner=self._stub_runner(),
            tee_runtime=_TEERuntime(),
            anchor=anchor,
            streaming_runner=runner,
        )
        # Confirm the streaming runner is wired through to the
        # server. Server-mediated dispatch is exercised E2E in
        # tests/integration/test_autoregressive_runner_server_wire.py
        # and the Phase 3.x.10.x Task 5 full-stack E2E.
        assert server._streaming_runner is runner  # noqa: SLF001

    def test_factory_with_synthetic_runner_also_wires_in(self):
        # Back-compat: SyntheticStreamingRunner from Phase 3.x.8
        # still wires through the same kwarg. Operators not yet on
        # the autoregressive runner can still opt into streaming.
        from prsm.compute.chain_rpc import make_layer_stage_server

        identity, anchor = self._identity_anchor()

        from prsm.compute.chain_rpc.server import (
            LayerSliceResult, LayerSliceRunner,
        )

        class _Stub(LayerSliceRunner):
            def run_layer_range(
                self, *, model, layer_range, activation, privacy_tier,
                is_final_stage,
            ) -> LayerSliceResult:
                return LayerSliceResult(
                    output=activation.copy(),
                    duration_seconds=0.001,
                    tee_attestation=b"\x07" * 32,
                    tee_type=TEEType.SOFTWARE,
                    epsilon_spent=0.0,
                )

        synth = SyntheticStreamingRunner(
            runner=_Stub(),
            output_decoder=lambda act: "ok",
        )

        class _TEERuntime:
            tee_type = TEEType.SOFTWARE

        class _Registry:
            def get(self, model_id):
                raise KeyError(model_id)

        server = make_layer_stage_server(
            identity=identity,
            registry=_Registry(),
            runner=self._stub_runner(),
            tee_runtime=_TEERuntime(),
            anchor=anchor,
            streaming_runner=synth,
        )
        assert server._streaming_runner is synth  # noqa: SLF001
