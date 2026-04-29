"""Phase 3.x.11 Task 7 — sharded autoregressive E2E with real distilgpt2.

Loads a real HuggingFace ``distilgpt2`` model and splits its 6
transformer layers into TWO chain stages:

  - Stage 1 (alice): layers 0-2 + token/position embeddings
  - Stage 2 (bob):   layers 3-5 + final layer norm + LM head

Drives a 4-token sharded autoregressive decode through the
``RpcChainExecutor.execute_chain_streaming(enable_sharded_decode=True)``
per-token chain loop. Verifies design plan §4 Task 7 acceptance:

  - Sharded output equals single-host distilgpt2 output for the
    same prompt + greedy decode (``temperature=0.0``,
    bit-identical token-by-token).
  - Per-token ``next_token_id`` is real (not synthetic) — i.e.,
    each iteration's sampled token comes from the actual LM-head
    projection of the boundary hidden state.
  - Cache survives across the 4-iteration loop and is evicted
    on terminal (verified via the manager's ``__contains__``
    state at each iteration + after the executor's terminal
    eviction broadcast).

All tests are ``@pytest.mark.slow`` (HF model load ~5-30s).
Default CI run excludes them; the production wiring is fully
exercised by the unit + sharded-runner suites; this E2E adds
the real-model + real-wire proof on top.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from prsm.compute.chain_rpc.client import (
    RpcChainExecutor,
    StreamToken,
)
from prsm.compute.chain_rpc.kv_cache import KVCacheManager
from prsm.compute.chain_rpc.protocol import (
    EvictCacheRequest,
    EvictCacheResponse,
    encode_message,
    parse_message,
)
from prsm.compute.chain_rpc.server import (
    LayerSliceResult,
    LayerSliceRunner,
    LayerStageServer,
)
from prsm.compute.inference.autoregressive_runner import SamplingDefaults
from prsm.compute.inference.models import ContentTier, InferenceRequest
from prsm.compute.inference.parallax_executor import ChainExecutionResult
from prsm.compute.inference.sharded_runner import (
    ShardedAutoregressiveRunner,
)
from prsm.compute.parallax_scheduling.prsm_request_router import GPUChain
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import generate_node_identity


pytestmark = pytest.mark.slow


# ──────────────────────────────────────────────────────────────────────────
# distilgpt2 adapter — implements ShardedLayerForward Protocol
# ──────────────────────────────────────────────────────────────────────────


class _DistilGPT2ShardedAdapter:
    """Wraps ``transformers.AutoModelForCausalLM`` (distilgpt2) for
    one chain stage's layer range.

    Uses HF transformers 5.x ``DynamicCache`` for per-stage KV
    storage. The cache is opaque to the runner; the model adapter
    creates one on PREFILL + threads the same instance through
    every INCREMENTAL.

    Stage 1 (layer_range[0]==0): receives ``input_ids`` (List[int]
    on the wire as np.int64), runs the embeddings + selected
    layer range; returns boundary hidden state + DynamicCache.

    Stage > 1 (layer_range[0]>0): receives float hidden state,
    runs the selected layer range with that as input; returns
    boundary hidden state + DynamicCache.

    Tail variant (apply_lm_head_and_sample): runs ``ln_f`` +
    ``lm_head`` on the boundary hidden state, samples the next
    token id (greedy when temperature == 0.0; temperature-scaled
    softmax + top-k/top-p otherwise).
    """

    def __init__(self, model: Any, tokenizer: Any) -> None:
        import torch
        from transformers import DynamicCache
        self._torch = torch
        self._DynamicCache = DynamicCache
        self._model = model
        self._tokenizer = tokenizer

    # ── ShardedLayerForward Protocol ──────────────────────────────────

    def forward_prefill(
        self,
        *,
        input_or_hidden: Any,
        layer_range: Tuple[int, int],
    ) -> Tuple[np.ndarray, Any]:
        torch = self._torch
        model = self._model
        start, end = layer_range
        cache = self._DynamicCache()
        with torch.no_grad():
            if start == 0:
                input_ids = self._coerce_input_ids(input_or_hidden)
                input_ids_tensor = torch.tensor(
                    [input_ids], dtype=torch.long,
                )
                seq_len = input_ids_tensor.shape[-1]
                position_ids = torch.arange(
                    0, seq_len, dtype=torch.long,
                ).unsqueeze(0)
                inputs_embeds = model.transformer.wte(input_ids_tensor)
                position_embeds = model.transformer.wpe(position_ids)
                hidden_states = inputs_embeds + position_embeds
                hidden_states = model.transformer.drop(hidden_states)
                cache_position = torch.arange(0, seq_len, dtype=torch.long)
            else:
                hidden_states = self._numpy_to_hidden(input_or_hidden)
                seq_len = hidden_states.shape[-2]
                cache_position = torch.arange(0, seq_len, dtype=torch.long)

            for i in range(start, end):
                block = model.transformer.h[i]
                outputs = block(
                    hidden_states,
                    past_key_values=cache,
                    cache_position=cache_position,
                    use_cache=True,
                )
                hidden_states = outputs[0] if isinstance(
                    outputs, tuple,
                ) else outputs

            return self._hidden_to_numpy(hidden_states), cache

    def forward_incremental(
        self,
        *,
        input_or_hidden: Any,
        layer_range: Tuple[int, int],
        kv_cache_payload: Any,
    ) -> Tuple[np.ndarray, Any]:
        torch = self._torch
        model = self._model
        start, end = layer_range
        cache = kv_cache_payload  # mutated in place by GPT2Block
        # past_length = how many tokens are already cached at this
        # stage's layer range. All layers in the range share the
        # same seq length post-PREFILL/INCREMENTAL, so peek any.
        past_length = cache.get_seq_length(start)
        with torch.no_grad():
            if start == 0:
                input_ids = self._coerce_input_ids(input_or_hidden)
                input_ids_tensor = torch.tensor(
                    [input_ids], dtype=torch.long,
                )
                position_ids = torch.tensor(
                    [[past_length]], dtype=torch.long,
                )
                inputs_embeds = model.transformer.wte(input_ids_tensor)
                position_embeds = model.transformer.wpe(position_ids)
                hidden_states = inputs_embeds + position_embeds
                hidden_states = model.transformer.drop(hidden_states)
            else:
                hidden_states = self._numpy_to_hidden(input_or_hidden)
            cache_position = torch.tensor(
                [past_length], dtype=torch.long,
            )

            for i in range(start, end):
                block = model.transformer.h[i]
                outputs = block(
                    hidden_states,
                    past_key_values=cache,
                    cache_position=cache_position,
                    use_cache=True,
                )
                hidden_states = outputs[0] if isinstance(
                    outputs, tuple,
                ) else outputs

            return self._hidden_to_numpy(hidden_states), cache

    def apply_lm_head_and_sample(
        self,
        *,
        hidden_state: np.ndarray,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> int:
        torch = self._torch
        model = self._model
        with torch.no_grad():
            h = self._numpy_to_hidden(hidden_state)
            h = model.transformer.ln_f(h)
            logits = model.lm_head(h)
            # Last position's logits (squeeze batch + seq dims).
            last_logits = logits[..., -1, :].squeeze()
            if temperature == 0.0:
                return int(last_logits.argmax().item())
            scaled = last_logits / temperature
            # Top-k filtering.
            if top_k > 0:
                topk_vals, _ = torch.topk(scaled, k=top_k)
                threshold = topk_vals[-1]
                scaled = torch.where(
                    scaled < threshold,
                    torch.full_like(scaled, float("-inf")),
                    scaled,
                )
            probs = torch.softmax(scaled, dim=-1)
            return int(torch.multinomial(probs, 1).item())

    # ── helpers ────────────────────────────────────────────────────────

    def _coerce_input_ids(self, value: Any) -> List[int]:
        if isinstance(value, list):
            return [int(v) for v in value]
        if isinstance(value, np.ndarray):
            return [int(v) for v in value.flatten().tolist()]
        raise TypeError(
            f"Stage 1 expected List[int] or np.ndarray of int64 ids, "
            f"got {type(value).__name__}"
        )

    def _numpy_to_hidden(self, arr: np.ndarray) -> Any:
        # Ensure 3-D [batch=1, seq_len, hidden_dim] for HF blocks.
        torch = self._torch
        np_arr = np.asarray(arr, dtype=np.float32)
        if np_arr.ndim == 2:
            np_arr = np_arr[np.newaxis, :, :]
        return torch.from_numpy(np_arr.copy()).to(
            next(self._model.parameters()).dtype,
        )

    def _hidden_to_numpy(self, hidden: Any) -> np.ndarray:
        # Always return as 3-D float32 [batch=1, seq, hidden].
        return hidden.detach().to(self._torch.float32).cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────
# Server-side scaffolding
# ──────────────────────────────────────────────────────────────────────────


class _Anchor:
    def __init__(self) -> None:
        self.registered: Dict[str, str] = {}

    def register(self, identity) -> None:
        self.registered[identity.node_id] = identity.public_key_b64

    def lookup(self, node_id: str) -> Optional[str]:
        return self.registered.get(node_id)


class _Shard:
    def __init__(self, layer_range: Tuple[int, int]) -> None:
        self.layer_range = layer_range


class _Model:
    def __init__(
        self, model_id: str, shard_ranges: List[Tuple[int, int]],
    ) -> None:
        self.model_id = model_id
        self.shards: List[_Shard] = [_Shard(lr) for lr in shard_ranges]


class _Registry:
    def __init__(
        self, *, model_id: str, shard_ranges: List[Tuple[int, int]],
    ) -> None:
        self._model = _Model(model_id, shard_ranges)
        self._model_id = model_id

    def get(self, model_id: str) -> _Model:
        if model_id != self._model_id:
            raise _ModelNotFoundError(model_id)
        return self._model


class _ModelNotFoundError(Exception):
    pass


class _TEERuntime:
    tee_type = TEEType.SOFTWARE

    def get_attestation_bytes(self) -> bytes:
        return b"\x07" * 32


class _PassthroughUnaryRunner(LayerSliceRunner):
    """Required by the LayerStageServer constructor — sharded
    decode never reaches this runner (the sharded path takes
    over for every dispatch when sharded_runner is wired)."""

    def run_layer_range(
        self,
        *,
        model,
        layer_range,
        activation,
        privacy_tier,
        is_final_stage,
    ) -> LayerSliceResult:
        return LayerSliceResult(
            output=activation.copy(),
            duration_seconds=0.001,
            tee_attestation=b"\x07" * 32,
            tee_type=TEEType.SOFTWARE,
            epsilon_spent=0.0,
        )


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hf_model_and_tokenizer():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "distilgpt2",
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "distilgpt2",
        )
        model.eval()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"distilgpt2 unavailable: {exc.__class__.__name__}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────
# Two-stage executor build
# ──────────────────────────────────────────────────────────────────────────


def _build_two_stage_sharded(
    model: Any,
    tokenizer: Any,
) -> Tuple[
    RpcChainExecutor,
    Any,  # alice identity
    Any,  # bob identity
    KVCacheManager,
    KVCacheManager,
    Dict[str, LayerStageServer],
]:
    alice_identity = generate_node_identity("alice")
    bob_identity = generate_node_identity("bob")
    settler_identity = generate_node_identity("settler")
    anchor = _Anchor()
    anchor.register(alice_identity)
    anchor.register(bob_identity)
    anchor.register(settler_identity)

    adapter = _DistilGPT2ShardedAdapter(model, tokenizer)

    alice_cache = KVCacheManager()
    bob_cache = KVCacheManager()

    alice_runner = ShardedAutoregressiveRunner(
        model=adapter,
        layer_range=(0, 3),
        kv_cache_manager=alice_cache,
        tee_attestation=b"\x07" * 32,
        tee_type=TEEType.SOFTWARE,
    )
    bob_runner = ShardedAutoregressiveRunner(
        model=adapter,
        layer_range=(3, 6),
        kv_cache_manager=bob_cache,
        tee_attestation=b"\x07" * 32,
        tee_type=TEEType.SOFTWARE,
        sampling_defaults=SamplingDefaults(
            max_tokens=4, temperature=0.0, top_k=50, top_p=0.95,
        ),
        eos_token_id=tokenizer.eos_token_id,
    )

    alice_server = LayerStageServer(
        identity=alice_identity,
        registry=_Registry(
            model_id="distilgpt2", shard_ranges=[(0, 3)],
        ),
        runner=_PassthroughUnaryRunner(),
        tee_runtime=_TEERuntime(),
        anchor=anchor,
        clock=lambda: 1000.0,
        kv_cache_manager=alice_cache,
        sharded_runner=alice_runner,
    )
    bob_server = LayerStageServer(
        identity=bob_identity,
        registry=_Registry(
            model_id="distilgpt2", shard_ranges=[(3, 6)],
        ),
        runner=_PassthroughUnaryRunner(),
        tee_runtime=_TEERuntime(),
        anchor=anchor,
        clock=lambda: 1000.0,
        kv_cache_manager=bob_cache,
        sharded_runner=bob_runner,
    )
    servers: Dict[str, LayerStageServer] = {
        alice_identity.node_id: alice_server,
        bob_identity.node_id: bob_server,
    }

    def send_message(address: str, request_bytes: bytes) -> bytes:
        srv = servers.get(address)
        if srv is None:
            raise ConnectionError(f"no server at {address}")
        return srv.handle(request_bytes)

    def cache_evict_send(address: str, request_bytes: bytes) -> bytes:
        srv = servers.get(address)
        if srv is None:
            raise ConnectionError(f"no server at {address}")
        return srv.handle(request_bytes)

    def prompt_encoder(prompt: str) -> np.ndarray:
        # Not used on the sharded path (executor uses tokenizer
        # directly), but RpcChainExecutor still requires a
        # callable here.
        return np.array([0], dtype=np.int32)

    def output_decoder(arr: np.ndarray) -> str:
        return ""

    executor = RpcChainExecutor(
        settler_identity=settler_identity,
        send_message=send_message,
        anchor=anchor,
        prompt_encoder=prompt_encoder,
        output_decoder=output_decoder,
        enable_sharded_decode=True,
        tokenizer=tokenizer,
        cache_evict_send_message=cache_evict_send,
        sharded_default_max_tokens=4,
    )
    return (
        executor,
        alice_identity,
        bob_identity,
        alice_cache,
        bob_cache,
        servers,
    )


def _make_chain(
    alice_id: str, bob_id: str,
) -> GPUChain:
    return GPUChain(
        request_id="req-sharded-e2e",
        region="us-east",
        stages=(alice_id, bob_id),
        layer_ranges=((0, 3), (3, 6)),
        total_latency_ms=10.0,
        stale_profile_count=0,
    )


def _make_request(
    *,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> InferenceRequest:
    return InferenceRequest(
        prompt=prompt,
        model_id="distilgpt2",
        budget_ftns=Decimal("10.0"),
        privacy_tier=PrivacyLevel.NONE,
        content_tier=ContentTier.A,
        request_id="req-sharded-e2e",
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _single_host_greedy_token_ids(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> List[int]:
    """Reference: drive distilgpt2 end-to-end on a single host with
    greedy decoding (do_sample=False, temperature=1.0 collapses to
    argmax). Returns the list of generated token ids (not the
    prompt). Used as the bit-identical determinism oracle."""
    import torch
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    prompt_len = input_ids.shape[-1]
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    full = out[0].tolist()
    return full[prompt_len:]


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────


class TestShardedE2EBitIdentical:
    def test_sharded_output_matches_single_host_greedy(
        self, hf_model_and_tokenizer,
    ):
        model, tokenizer = hf_model_and_tokenizer

        prompt = "The quick brown fox"
        max_tokens = 4

        # Reference: single-host greedy decode.
        reference_ids = _single_host_greedy_token_ids(
            model, tokenizer, prompt, max_new_tokens=max_tokens,
        )
        assert len(reference_ids) == max_tokens

        # Sharded chain via executor.
        (
            executor, alice_identity, bob_identity, _, _, _,
        ) = _build_two_stage_sharded(model, tokenizer)
        chain = _make_chain(alice_identity.node_id, bob_identity.node_id)

        events = list(executor.execute_chain_streaming(
            request=_make_request(
                prompt=prompt, max_tokens=max_tokens,
                temperature=0.0,
            ),
            chain=chain,
        ))
        stream_tokens = [
            e for e in events if isinstance(e, StreamToken)
        ]
        result = next(
            e for e in events if isinstance(e, ChainExecutionResult)
        )

        # Same number of tokens.
        assert len(stream_tokens) == max_tokens
        # Each sharded token id matches the single-host reference
        # token id at the same position (bit-identical greedy
        # determinism).
        sharded_ids = [t.token_id for t in stream_tokens]
        assert sharded_ids == reference_ids
        # ChainExecutionResult.output is the joined text deltas.
        assert isinstance(result.output, str)


class TestShardedE2EAutoregressive:
    def test_per_token_id_changes_across_iterations(
        self, hf_model_and_tokenizer,
    ):
        # If sampled tokens were synthetic placeholders, every
        # token would be the same id. Real autoregressive decode
        # produces non-trivial sequences. Confirm the 4 sampled
        # ids span at least 2 distinct values.
        model, tokenizer = hf_model_and_tokenizer
        (
            executor, alice_identity, bob_identity, _, _, _,
        ) = _build_two_stage_sharded(model, tokenizer)
        chain = _make_chain(alice_identity.node_id, bob_identity.node_id)
        events = list(executor.execute_chain_streaming(
            request=_make_request(
                prompt="The quick brown fox", max_tokens=4,
                temperature=0.0,
            ),
            chain=chain,
        ))
        ids = [
            e.token_id for e in events
            if isinstance(e, StreamToken)
        ]
        assert len(ids) == 4
        # Real model output is non-trivial; at least 2 distinct
        # token ids in 4 consecutive greedy samples.
        assert len(set(ids)) >= 2

    def test_token_id_is_real_int_not_synthetic_negative(
        self, hf_model_and_tokenizer,
    ):
        # Synthetic placeholders historically used token_id=-1 or
        # None; real autoregressive sampling produces non-negative
        # ids in vocab range.
        model, tokenizer = hf_model_and_tokenizer
        (
            executor, alice_identity, bob_identity, _, _, _,
        ) = _build_two_stage_sharded(model, tokenizer)
        chain = _make_chain(alice_identity.node_id, bob_identity.node_id)
        events = list(executor.execute_chain_streaming(
            request=_make_request(
                prompt="Hello", max_tokens=4, temperature=0.0,
            ),
            chain=chain,
        ))
        for e in events:
            if isinstance(e, StreamToken):
                assert e.token_id is not None
                assert e.token_id >= 0
                assert e.token_id < tokenizer.vocab_size


class TestShardedE2ECacheLifecycle:
    def test_cache_survives_loop_then_evicted_on_terminal(
        self, hf_model_and_tokenizer,
    ):
        # The two stages' caches must each hold ONE handle for
        # the request_id from PREFILL onwards through every
        # INCREMENTAL iteration, then be empty after terminal
        # eviction broadcast.
        model, tokenizer = hf_model_and_tokenizer
        (
            executor, alice_identity, bob_identity,
            alice_cache, bob_cache, _,
        ) = _build_two_stage_sharded(model, tokenizer)
        chain = _make_chain(alice_identity.node_id, bob_identity.node_id)

        gen = executor.execute_chain_streaming(
            request=_make_request(
                prompt="The quick brown fox", max_tokens=4,
                temperature=0.0,
            ),
            chain=chain,
        )
        # Pull tokens one at a time; assert cache is populated
        # at each step (PREFILL allocated; INCREMENTALs reuse).
        observed_token_count = 0
        for event in gen:
            if isinstance(event, StreamToken):
                observed_token_count += 1
                # Mid-stream — both caches must hold the request.
                assert "req-sharded-e2e" in alice_cache
                assert "req-sharded-e2e" in bob_cache
            elif isinstance(event, ChainExecutionResult):
                # ChainExecutionResult yields BEFORE the finally
                # block runs eviction. At this point caches still
                # hold the handle.
                assert "req-sharded-e2e" in alice_cache
                assert "req-sharded-e2e" in bob_cache
        assert observed_token_count == 4

        # Generator fully exhausted → finally block ran →
        # eviction broadcast went out → both caches empty.
        assert "req-sharded-e2e" not in alice_cache
        assert "req-sharded-e2e" not in bob_cache

    def test_cache_evicted_on_caller_close(
        self, hf_model_and_tokenizer,
    ):
        # Alternate exit path: caller closes generator mid-stream.
        # finally block still runs eviction broadcast.
        model, tokenizer = hf_model_and_tokenizer
        (
            executor, alice_identity, bob_identity,
            alice_cache, bob_cache, _,
        ) = _build_two_stage_sharded(model, tokenizer)
        chain = _make_chain(alice_identity.node_id, bob_identity.node_id)

        gen = executor.execute_chain_streaming(
            request=_make_request(
                prompt="The quick brown fox", max_tokens=4,
                temperature=0.0,
            ),
            chain=chain,
        )
        # Pull only ONE token then close.
        next(gen)
        assert "req-sharded-e2e" in alice_cache
        assert "req-sharded-e2e" in bob_cache
        gen.close()
        # finally → eviction broadcast → caches empty.
        assert "req-sharded-e2e" not in alice_cache
        assert "req-sharded-e2e" not in bob_cache


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.11.x Task 5 — chunked + sharded PREFILL E2E
# ──────────────────────────────────────────────────────────────────────────


def _build_two_stage_sharded_chunked(
    model: Any,
    tokenizer: Any,
    *,
    chunk_threshold_bytes: int = 10_000,
):
    """Phase 3.x.11.x: same as _build_two_stage_sharded but with
    streamed_send_message wired (routing through
    LayerStageServer.handle_streamed) and a low chunk_threshold
    that forces chunking on a normal-length prompt's Stage 1 →
    Stage 2 hidden-state handoff (768 hidden_dim × 4 bytes per
    fp32 × ~30+ positions easily exceeds the threshold)."""
    alice_identity = generate_node_identity("alice")
    bob_identity = generate_node_identity("bob")
    settler_identity = generate_node_identity("settler")
    anchor = _Anchor()
    anchor.register(alice_identity)
    anchor.register(bob_identity)
    anchor.register(settler_identity)

    adapter = _DistilGPT2ShardedAdapter(model, tokenizer)
    alice_cache = KVCacheManager()
    bob_cache = KVCacheManager()
    alice_runner = ShardedAutoregressiveRunner(
        model=adapter, layer_range=(0, 3),
        kv_cache_manager=alice_cache,
        tee_attestation=b"\x07" * 32, tee_type=TEEType.SOFTWARE,
    )
    bob_runner = ShardedAutoregressiveRunner(
        model=adapter, layer_range=(3, 6),
        kv_cache_manager=bob_cache,
        tee_attestation=b"\x07" * 32, tee_type=TEEType.SOFTWARE,
        sampling_defaults=SamplingDefaults(
            max_tokens=4, temperature=0.0, top_k=50, top_p=0.95,
        ),
        eos_token_id=tokenizer.eos_token_id,
    )

    alice_server = LayerStageServer(
        identity=alice_identity,
        registry=_Registry(
            model_id="distilgpt2", shard_ranges=[(0, 3)],
        ),
        runner=_PassthroughUnaryRunner(),
        tee_runtime=_TEERuntime(),
        anchor=anchor, clock=lambda: 1000.0,
        kv_cache_manager=alice_cache,
        sharded_runner=alice_runner,
    )
    bob_server = LayerStageServer(
        identity=bob_identity,
        registry=_Registry(
            model_id="distilgpt2", shard_ranges=[(3, 6)],
        ),
        runner=_PassthroughUnaryRunner(),
        tee_runtime=_TEERuntime(),
        anchor=anchor, clock=lambda: 1000.0,
        kv_cache_manager=bob_cache,
        sharded_runner=bob_runner,
    )
    servers = {
        alice_identity.node_id: alice_server,
        bob_identity.node_id: bob_server,
    }

    # Track every wire transport invocation so tests can assert
    # that the chunked path was actually used.
    delivery_log = {"unary": [], "streamed": []}

    def send_message(address: str, request_bytes: bytes) -> bytes:
        delivery_log["unary"].append(address)
        srv = servers.get(address)
        if srv is None:
            raise ConnectionError(f"no server at {address}")
        return srv.handle(request_bytes)

    def streamed_send_message(
        address: str,
        manifest_bytes: bytes,
        chunk_iter,
    ):
        delivery_log["streamed"].append(address)
        srv = servers.get(address)
        if srv is None:
            raise ConnectionError(f"no server at {address}")
        return srv.handle_streamed(manifest_bytes, chunk_iter)

    def cache_evict_send(address: str, request_bytes: bytes) -> bytes:
        srv = servers.get(address)
        if srv is None:
            raise ConnectionError(f"no server at {address}")
        return srv.handle(request_bytes)

    executor = RpcChainExecutor(
        settler_identity=settler_identity,
        send_message=send_message,
        streamed_send_message=streamed_send_message,
        anchor=anchor,
        prompt_encoder=lambda p: np.array([0], dtype=np.int32),
        output_decoder=lambda a: "",
        enable_sharded_decode=True,
        tokenizer=tokenizer,
        cache_evict_send_message=cache_evict_send,
        sharded_default_max_tokens=4,
        chunk_threshold_bytes=chunk_threshold_bytes,
    )
    return (
        executor, alice_identity, bob_identity,
        alice_cache, bob_cache, servers, delivery_log,
    )


class TestShardedE2EChunkedPrefill:
    """Phase 3.x.11.x Task 5 — chunked + sharded PREFILL
    composition E2E. Drives the full server-side streamed path
    against real distilgpt2 + verifies bit-identical output vs
    the inline-PREFILL Phase 3.x.11 Task 7 baseline."""

    def test_chunked_prefill_path_exercised(
        self, hf_model_and_tokenizer,
    ):
        # Stage 1 → Stage 2 hidden state at distilgpt2's
        # hidden_dim=768 fp32 = 3072 bytes/position.
        # 30+ positions of a long prompt exceeds the 10 KB
        # threshold and forces the chunked path.
        model, tokenizer = hf_model_and_tokenizer
        prompt = (
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump in the field. "
            "Sphinx of black quartz judge my vow swiftly today."
        )
        (
            executor, alice_id, bob_id, _, _, _, log,
        ) = _build_two_stage_sharded_chunked(
            model, tokenizer, chunk_threshold_bytes=10_000,
        )
        chain = _make_chain(alice_id.node_id, bob_id.node_id)
        events = list(executor.execute_chain_streaming(
            request=_make_request(
                prompt=prompt, max_tokens=4, temperature=0.0,
            ),
            chain=chain,
        ))
        # Streamed transport hit at least once during PREFILL
        # (Stage 2 input = Stage 1's hidden state → multi-KB).
        assert len(log["streamed"]) >= 1, (
            f"chunked PREFILL path NOT exercised; "
            f"streamed deliveries: {log['streamed']}"
        )
        # Chain still completed (4 tokens + ChainExecutionResult).
        stream_tokens = [
            e for e in events if isinstance(e, StreamToken)
        ]
        results = [
            e for e in events if isinstance(e, ChainExecutionResult)
        ]
        assert len(stream_tokens) == 4
        assert len(results) == 1

    def test_chunked_prefill_output_matches_single_host_greedy(
        self, hf_model_and_tokenizer,
    ):
        # Bit-identical determinism check (Phase 3.x.11 Task 7
        # invariant carries forward through the chunked path).
        model, tokenizer = hf_model_and_tokenizer
        prompt = (
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump in the field."
        )
        max_tokens = 4
        reference_ids = _single_host_greedy_token_ids(
            model, tokenizer, prompt, max_new_tokens=max_tokens,
        )
        (
            executor, alice_id, bob_id, _, _, _, log,
        ) = _build_two_stage_sharded_chunked(
            model, tokenizer, chunk_threshold_bytes=10_000,
        )
        chain = _make_chain(alice_id.node_id, bob_id.node_id)
        events = list(executor.execute_chain_streaming(
            request=_make_request(
                prompt=prompt, max_tokens=max_tokens,
                temperature=0.0,
            ),
            chain=chain,
        ))
        sharded_ids = [
            e.token_id for e in events
            if isinstance(e, StreamToken)
        ]
        # Streamed path was actually used (sanity).
        assert len(log["streamed"]) >= 1
        # AND output is bit-identical to single-host greedy.
        assert sharded_ids == reference_ids, (
            f"chunked sharded output {sharded_ids} != "
            f"single-host {reference_ids}"
        )

    def test_receipt_carries_iteration_attestation_envelope(
        self, hf_model_and_tokenizer,
    ):
        # Phase 3.x.11.x Task 2 envelope — receipt's
        # tee_attestation MUST decode as a multi-iteration
        # attestation chain (one entry per chain pass), even
        # when PREFILL went through the chunked path.
        from prsm.compute.inference.multi_stage_attestation import (
            decode_multi_iteration_attestation,
            is_multi_iteration_attestation,
        )
        model, tokenizer = hf_model_and_tokenizer
        prompt = (
            "The quick brown fox jumps over the lazy dog. "
            "Sphinx of black quartz judge my vow swiftly today. "
            "How vexingly quick daft zebras jump in the field."
        )
        (
            executor, alice_id, bob_id, _, _, _, log,
        ) = _build_two_stage_sharded_chunked(
            model, tokenizer, chunk_threshold_bytes=10_000,
        )
        chain = _make_chain(alice_id.node_id, bob_id.node_id)
        events = list(executor.execute_chain_streaming(
            request=_make_request(
                prompt=prompt, max_tokens=3, temperature=0.0,
            ),
            chain=chain,
        ))
        result = next(
            e for e in events if isinstance(e, ChainExecutionResult)
        )
        # Multi-iteration envelope (NOT the legacy multi-stage
        # envelope).
        assert is_multi_iteration_attestation(result.tee_attestation)
        iterations = decode_multi_iteration_attestation(
            result.tee_attestation,
        )
        assert iterations is not None
        # 1 PREFILL + 2 INCREMENTALs = 3 iterations.
        assert len(iterations) == 3
        # Each iteration carries 2 stage records (alice + bob).
        for it in iterations:
            assert len(it.stage_records) == 2
        # Streamed path was used during PREFILL (chunked).
        assert len(log["streamed"]) >= 1


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.11.q Task 5 — Tier C chain-level decorators E2E
# ──────────────────────────────────────────────────────────────────────────


class TestTierCShardedDecoratorsE2E:
    """Exercises the Phase 3.x.11.q chain-level decorators end-to-end
    against a real distilgpt2 2-stage chain. Confirms:
      - undecorated executor leaks per-token cadence (baseline)
      - M2 decorator emits a single trailing StreamToken
      - M1 decorator paces inter-frame intervals at ≥ cadence

    The timing observer records the wall-clock time of each
    StreamToken event; assertions are loose enough to handle CI
    timing jitter (±5% on the M1 cadence test isn't tractable on
    real distilgpt2 without longer cadences; we use ≥ cadence
    instead, which is the load-bearing invariant — the decorator
    must never YIELD faster than cadence)."""

    def test_undecorated_executor_emits_per_token_cadence(
        self, hf_model_and_tokenizer,
    ):
        # Baseline reference: without any decorator, each token
        # emits as soon as the chain produces it. Mostly a
        # smoke that the test scaffolding works; no timing
        # assertion since real-distilgpt2 native rate varies.
        from prsm.compute.chain_rpc.client import StreamToken
        model, tokenizer = hf_model_and_tokenizer
        prompt = "The quick brown fox"
        max_tokens = 3
        (
            executor, alice_id, bob_id, _, _, _,
        ) = _build_two_stage_sharded(model, tokenizer)
        chain = _make_chain(alice_id.node_id, bob_id.node_id)
        events = list(executor.execute_chain_streaming(
            request=_make_request(
                prompt=prompt, max_tokens=max_tokens,
                temperature=0.0,
            ),
            chain=chain,
        ))
        tokens = [e for e in events if isinstance(e, StreamToken)]
        # Baseline: undecorated emits exactly max_tokens tokens.
        assert len(tokens) == max_tokens

    def test_m2_decorator_emits_single_trailing_frame(
        self, hf_model_and_tokenizer,
    ):
        # M2 — BatchedTrailingShardedExecutor: regardless of how
        # many tokens the inner chain produces, the executor →
        # caller wire sees ONE StreamToken (joined text) + ONE
        # ChainExecutionResult. This is the load-bearing M2
        # invariant.
        from prsm.compute.chain_rpc.client import StreamToken
        from prsm.compute.chain_rpc import (
            BatchedTrailingShardedExecutor,
        )
        model, tokenizer = hf_model_and_tokenizer
        prompt = "The quick brown fox"
        max_tokens = 4
        (
            executor, alice_id, bob_id, _, _, _,
        ) = _build_two_stage_sharded(model, tokenizer)
        chain = _make_chain(alice_id.node_id, bob_id.node_id)
        # Wrap the real executor with M2.
        m2_executor = BatchedTrailingShardedExecutor(inner=executor)
        events = list(m2_executor.execute_chain_streaming(
            request=_make_request(
                prompt=prompt, max_tokens=max_tokens,
                temperature=0.0,
            ),
            chain=chain,
        ))
        tokens = [e for e in events if isinstance(e, StreamToken)]
        results = [
            e for e in events if isinstance(e, ChainExecutionResult)
        ]
        # Load-bearing: exactly ONE StreamToken regardless of how
        # many tokens the inner chain produced (max_tokens=4).
        assert len(tokens) == 1
        assert len(results) == 1
        # Joined text contains the cumulative output.
        assert len(tokens[0].text_delta) > 0
        # finish_reason from the inner's last token propagates.
        assert tokens[0].finish_reason is not None

    def test_m1_decorator_paces_inter_frame_intervals(
        self, hf_model_and_tokenizer,
    ):
        # M1 — FixedRateShardedExecutor: inter-StreamToken
        # intervals on the executor → caller wire are clamped
        # to ≥ cadence. We use a small cadence (50ms) and 4
        # tokens; the inter-token wall-clock observation must
        # satisfy interval >= cadence - small_tolerance for
        # each gap.
        import time as _time
        from prsm.compute.chain_rpc.client import StreamToken
        from prsm.compute.chain_rpc import FixedRateShardedExecutor

        model, tokenizer = hf_model_and_tokenizer
        prompt = "The quick brown fox"
        max_tokens = 4
        cadence = 0.05  # 50ms — large enough to dominate chain native
        (
            executor, alice_id, bob_id, _, _, _,
        ) = _build_two_stage_sharded(model, tokenizer)
        chain = _make_chain(alice_id.node_id, bob_id.node_id)
        m1_executor = FixedRateShardedExecutor(
            inner=executor, cadence_seconds=cadence,
        )

        emit_times: list = []
        for ev in m1_executor.execute_chain_streaming(
            request=_make_request(
                prompt=prompt, max_tokens=max_tokens,
                temperature=0.0,
            ),
            chain=chain,
        ):
            if isinstance(ev, StreamToken):
                emit_times.append(_time.monotonic())
        assert len(emit_times) == max_tokens
        # First token: no prior emit to gate against. Subsequent
        # tokens: interval ≥ cadence - tolerance. Use 5ms
        # tolerance to absorb scheduler jitter; the load-bearing
        # invariant is "decorator never yields faster than cadence".
        intervals = [
            emit_times[i] - emit_times[i - 1]
            for i in range(1, len(emit_times))
        ]
        for i, dt in enumerate(intervals):
            assert dt >= cadence - 0.005, (
                f"M1 decorator yielded faster than cadence at "
                f"interval {i}: dt={dt:.4f}s vs cadence={cadence}s"
            )
