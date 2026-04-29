"""Phase 3.x.11.y Task 7 — speculative-decoding E2E with real
distilgpt2 + ``HFDraftModel``.

Closes the speculative-decoding control-loop's real-model proof.
Loads a real HuggingFace distilgpt2, splits its 6 transformer
layers into TWO chain stages (alice 0-2 + bob 3-5), wires
``HFDraftModel`` as the draft using the SAME distilgpt2 as the
verifier (perfect-accept oracle — every draft proposal matches
the verifier's argmax under greedy), and drives an 8-token
speculative decode through
``RpcChainExecutor.execute_chain_streaming(draft_model=...)``.

Acceptance (design plan §4 Task 7):
  - At least one speculation round accepts ``>= 2`` draft tokens
    — proves the perf path actually fires (vs falling back to
    single-token-per-round emit).
  - Output is bit-identical to non-speculative single-host
    distilgpt2 greedy decode for the same prompt + temperature
    == 0.0 (speculation is a perf optimization, not a sampling
    change).
  - Per-iteration receipt records ``decode_mode=VERIFY`` for
    every speculation round.
  - Cancellation mid-speculation cleanly evicts both the chain-
    side cache (via EvictCacheRequest broadcast) and the draft
    state (via draft.evict).

All tests are ``@pytest.mark.slow`` (HF model load ~5-30s).
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
    DecodeMode,
    EvictCacheRequest,
    EvictCacheResponse,
    RollbackCacheRequest,
    RollbackCacheResponse,
    encode_message,
    parse_message,
)
from prsm.compute.chain_rpc.server import (
    LayerSliceResult,
    LayerSliceRunner,
    LayerStageServer,
)
from prsm.compute.inference.autoregressive_runner import SamplingDefaults
from prsm.compute.inference.draft_model import HFDraftModel
from prsm.compute.inference.models import ContentTier, InferenceRequest
from prsm.compute.inference.parallax_executor import ChainExecutionResult
from prsm.compute.inference.multi_stage_attestation import (
    decode_multi_iteration_attestation,
    is_multi_iteration_attestation,
)
from prsm.compute.inference.sharded_runner import (
    ShardedAutoregressiveRunner,
)
from prsm.compute.parallax_scheduling.prsm_request_router import GPUChain
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import generate_node_identity


pytestmark = pytest.mark.slow


# ──────────────────────────────────────────────────────────────────────────
# Speculation-capable distilgpt2 adapter
# ──────────────────────────────────────────────────────────────────────────


class _SpeculativeDistilGPT2Adapter:
    """Speculation-capable extension of the Phase 3.x.11 distilgpt2
    sharded adapter. Adds ``forward_verify``,
    ``apply_lm_head_and_sample_batch``, and ``truncate_cache``.

    All operations use HF's ``DynamicCache``. Truncation uses
    ``DynamicCache.crop`` (drops the last N positions in-place).
    """

    def __init__(self, model: Any, tokenizer: Any) -> None:
        import torch
        from transformers import DynamicCache
        self._torch = torch
        self._DynamicCache = DynamicCache
        self._model = model
        self._tokenizer = tokenizer

    # ── forward_prefill / forward_incremental (Phase 3.x.11) ──────────

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
        cache = kv_cache_payload
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

    # ── Phase 3.x.11.y — VERIFY / batch sampling / truncate ───────────

    def forward_verify(
        self,
        *,
        input_or_hidden: Any,
        layer_range: Tuple[int, int],
        kv_cache_payload: Any,
    ) -> Tuple[np.ndarray, Any]:
        """Batched K+1-position forward with cached KV. Builds an
        explicit 4D additive causal mask so each new-position
        query only attends to (cached past + own + earlier new
        positions). Without an explicit mask, HF GPT2's attention
        between K+1 query positions and (cached + K+1) key
        positions defaults to full attention across new tokens —
        that breaks greedy-equivalence vs single-token
        INCREMENTAL.
        """
        torch = self._torch
        model = self._model
        start, end = layer_range
        cache = kv_cache_payload
        past_length = cache.get_seq_length(start)
        with torch.no_grad():
            if start == 0:
                input_ids = self._coerce_input_ids(input_or_hidden)
                seq_len = len(input_ids)
                input_ids_tensor = torch.tensor(
                    [input_ids], dtype=torch.long,
                )
                position_ids = torch.arange(
                    past_length, past_length + seq_len, dtype=torch.long,
                ).unsqueeze(0)
                inputs_embeds = model.transformer.wte(input_ids_tensor)
                position_embeds = model.transformer.wpe(position_ids)
                hidden_states = inputs_embeds + position_embeds
                hidden_states = model.transformer.drop(hidden_states)
            else:
                hidden_states = self._numpy_to_hidden(input_or_hidden)
                seq_len = hidden_states.shape[-2]
            cache_position = torch.arange(
                past_length, past_length + seq_len, dtype=torch.long,
            )

            # Build 4D additive causal mask: shape
            # [batch=1, heads=1 (broadcast), q_len, kv_len].
            # Query at new position q (absolute = past_length + q)
            # attends to keys at absolute positions 0..past_length+q
            # (i.e., all cached past + new positions up to and
            # including itself).
            kv_len = past_length + seq_len
            q_abs = torch.arange(past_length, past_length + seq_len)
            k_abs = torch.arange(0, kv_len)
            attendable = k_abs[None, :] <= q_abs[:, None]  # [seq_len, kv_len]
            dtype = next(model.parameters()).dtype
            min_value = torch.finfo(dtype).min
            attention_mask = torch.where(
                attendable,
                torch.tensor(0.0, dtype=dtype),
                torch.tensor(min_value, dtype=dtype),
            ).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, kv_len]

            for i in range(start, end):
                block = model.transformer.h[i]
                outputs = block(
                    hidden_states,
                    attention_mask=attention_mask,
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
            last_logits = logits[..., -1, :].squeeze()
            if temperature == 0.0:
                return int(last_logits.argmax().item())
            scaled = last_logits / temperature
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

    def apply_lm_head_and_sample_batch(
        self,
        *,
        hidden_state_batch: np.ndarray,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> List[int]:
        """Project K+1 hidden states through ln_f + lm_head and
        argmax per position. v1 greedy-only: temperature == 0.0 →
        per-position argmax; non-zero raises (the runner's
        greedy-only gate should catch this earlier, but we fail
        loudly if reached).
        """
        torch = self._torch
        model = self._model
        if temperature != 0.0:
            raise RuntimeError(
                f"_SpeculativeDistilGPT2Adapter: greedy-only batch "
                f"sample at v1; got temperature={temperature}"
            )
        with torch.no_grad():
            h = self._numpy_to_hidden(hidden_state_batch)
            # h shape: [batch=1, seq=K+1, hidden] (or [seq, hidden]
            # → augmented to [1, seq, hidden] by _numpy_to_hidden).
            h = model.transformer.ln_f(h)
            logits = model.lm_head(h)
            # logits shape [1, K+1, vocab] → argmax along vocab axis.
            argmax_ids = logits.argmax(dim=-1).squeeze(0)
            return [int(t) for t in argmax_ids.tolist()]

    def truncate_cache(self, payload: Any, n_positions: int) -> Any:
        """Drop the last N positions from the DynamicCache. HF's
        ``DynamicCache.crop(max_length)`` truncates IN PLACE to the
        first ``max_length`` positions; we compute
        ``current - n_positions`` and call crop.

        Per-stage seq_length lookup: the adapter is layer-range-
        agnostic but a sharded runner only populates its own
        layer range. Calling ``get_seq_length()`` without args
        defaults to layer 0, which is empty on stages whose
        layer_range[0] > 0 (e.g., bob 3-6) — would crop to 0
        and corrupt the cache. Iterate to find any populated
        layer and use its length as the source-of-truth.
        """
        current = 0
        # DynamicCache supports __len__() returning layer count.
        try:
            n_layers = len(payload)
        except TypeError:
            n_layers = 32  # defensive upper bound
        for i in range(n_layers):
            try:
                seq_len = payload.get_seq_length(i)
            except Exception:  # noqa: BLE001
                continue
            if seq_len > 0:
                current = seq_len
                break
        new_length = max(0, current - n_positions)
        payload.crop(new_length)
        return payload

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
        torch = self._torch
        np_arr = np.asarray(arr, dtype=np.float32)
        if np_arr.ndim == 2:
            np_arr = np_arr[np.newaxis, :, :]
        return torch.from_numpy(np_arr.copy()).to(
            next(self._model.parameters()).dtype,
        )

    def _hidden_to_numpy(self, hidden: Any) -> np.ndarray:
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
# Speculative two-stage executor build
# ──────────────────────────────────────────────────────────────────────────


def _build_speculative_two_stage(
    model: Any,
    tokenizer: Any,
    *,
    speculation_depth: int = 4,
    sharded_default_max_tokens: int = 8,
) -> Tuple[
    RpcChainExecutor,
    Any,  # alice identity
    Any,  # bob identity
    KVCacheManager,
    KVCacheManager,
    Dict[str, LayerStageServer],
    HFDraftModel,
]:
    alice_identity = generate_node_identity("alice")
    bob_identity = generate_node_identity("bob")
    settler_identity = generate_node_identity("settler")
    anchor = _Anchor()
    anchor.register(alice_identity)
    anchor.register(bob_identity)
    anchor.register(settler_identity)

    adapter = _SpeculativeDistilGPT2Adapter(model, tokenizer)

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
            max_tokens=sharded_default_max_tokens,
            temperature=0.0, top_k=50, top_p=0.95,
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

    def rollback_cache_send(address: str, request_bytes: bytes) -> bytes:
        srv = servers.get(address)
        if srv is None:
            raise ConnectionError(f"no server at {address}")
        return srv.handle(request_bytes)

    def prompt_encoder(prompt: str) -> np.ndarray:
        return np.array([0], dtype=np.int32)

    def output_decoder(arr: np.ndarray) -> str:
        return ""

    # Draft model: same distilgpt2 (perfect-accept oracle for
    # greedy under temp=0; every draft proposal matches the
    # verifier's argmax).
    draft = HFDraftModel(
        model=model,
        eos_token_id=tokenizer.eos_token_id,
    )

    executor = RpcChainExecutor(
        settler_identity=settler_identity,
        send_message=send_message,
        anchor=anchor,
        prompt_encoder=prompt_encoder,
        output_decoder=output_decoder,
        enable_sharded_decode=True,
        tokenizer=tokenizer,
        cache_evict_send_message=cache_evict_send,
        rollback_cache_send_message=rollback_cache_send,
        sharded_default_max_tokens=sharded_default_max_tokens,
        draft_model=draft,
        speculation_depth=speculation_depth,
    )
    return (
        executor,
        alice_identity,
        bob_identity,
        alice_cache,
        bob_cache,
        servers,
        draft,
    )


def _make_chain(
    alice_id: str, bob_id: str,
) -> GPUChain:
    return GPUChain(
        request_id="req-spec-e2e",
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
        request_id="req-spec-e2e",
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _single_host_greedy_token_ids(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> List[int]:
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


class TestSpeculativeE2EGreedyEquivalence:
    def test_speculative_output_matches_single_host_greedy(
        self, hf_model_and_tokenizer,
    ):
        """Bit-identical token-by-token output between speculative
        sharded chain and non-speculative single-host greedy.
        Speculation is a perf optimization, not a sampling
        change — same prompt + same temp=0 → same tokens.
        """
        model, tokenizer = hf_model_and_tokenizer
        prompt = "The quick brown fox"
        max_tokens = 8

        reference_ids = _single_host_greedy_token_ids(
            model, tokenizer, prompt, max_new_tokens=max_tokens,
        )
        assert len(reference_ids) == max_tokens

        (
            executor, alice_identity, bob_identity, _, _, _, _,
        ) = _build_speculative_two_stage(
            model, tokenizer, speculation_depth=4,
            sharded_default_max_tokens=max_tokens,
        )
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

        assert len(stream_tokens) == max_tokens
        sharded_ids = [t.token_id for t in stream_tokens]
        assert sharded_ids == reference_ids, (
            f"speculative output {sharded_ids} != single-host "
            f"greedy {reference_ids} — speculation broke greedy "
            f"equivalence"
        )
        assert isinstance(result.output, str)


class TestSpeculativeE2EAcceptanceFires:
    def test_at_least_one_round_accepts_multi_token(
        self, hf_model_and_tokenizer,
    ):
        """With draft == verifier (same distilgpt2, same greedy
        argmax), every speculation round should accept ALL K
        drafts (perfect accept under matched models). Verify the
        speculation perf path actually fires by checking that
        the receipt's per-iteration envelope contains VERIFY
        iterations, not just PREFILL+INCREMENTALs.
        """
        model, tokenizer = hf_model_and_tokenizer
        (
            executor, alice_identity, bob_identity, _, _, _, _,
        ) = _build_speculative_two_stage(
            model, tokenizer, speculation_depth=4,
            sharded_default_max_tokens=8,
        )
        chain = _make_chain(alice_identity.node_id, bob_identity.node_id)
        events = list(executor.execute_chain_streaming(
            request=_make_request(
                prompt="The quick brown fox", max_tokens=8,
                temperature=0.0,
            ),
            chain=chain,
        ))
        result = next(
            e for e in events if isinstance(e, ChainExecutionResult)
        )
        assert is_multi_iteration_attestation(result.tee_attestation)
        iterations = decode_multi_iteration_attestation(
            result.tee_attestation,
        )
        # Iteration 0 must be PREFILL.
        assert iterations[0].decode_mode == DecodeMode.PREFILL
        # At least one VERIFY iteration must follow (proves the
        # speculation perf path engaged; if it had fallen back to
        # single-token decode we'd see INCREMENTAL only).
        verify_iterations = [
            it for it in iterations
            if it.decode_mode == DecodeMode.VERIFY
        ]
        assert len(verify_iterations) >= 1, (
            f"expected at least 1 VERIFY iteration; got "
            f"{[it.decode_mode.value for it in iterations]}"
        )
        # Every VERIFY iteration covered both stages.
        for it in verify_iterations:
            assert len(it.stage_records) == 2

    def test_speculation_emits_more_than_one_token_per_verify_round(
        self, hf_model_and_tokenizer,
    ):
        """With perfect-accept (draft==verifier under greedy),
        each VERIFY round emits K+1 tokens. 8 max_tokens at K=4
        = 1 PREFILL (1 tok) + ≥1 VERIFY round (5 toks).

        Counted iterations should be FEWER than tokens emitted
        — proving the speculation actually amortizes chain
        passes.
        """
        model, tokenizer = hf_model_and_tokenizer
        (
            executor, alice_identity, bob_identity, _, _, _, _,
        ) = _build_speculative_two_stage(
            model, tokenizer, speculation_depth=4,
            sharded_default_max_tokens=8,
        )
        chain = _make_chain(alice_identity.node_id, bob_identity.node_id)
        events = list(executor.execute_chain_streaming(
            request=_make_request(
                prompt="The quick brown fox", max_tokens=8,
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
        iterations = decode_multi_iteration_attestation(
            result.tee_attestation,
        )
        # Tokens emitted > iterations — speculation is amortizing.
        assert len(stream_tokens) > len(iterations), (
            f"speculation should emit more tokens than chain "
            f"iterations; got {len(stream_tokens)} tokens vs "
            f"{len(iterations)} iterations"
        )


class TestSpeculativeE2ECacheLifecycle:
    def test_cache_evicted_on_terminal(
        self, hf_model_and_tokenizer,
    ):
        """Both stages' caches + the draft state must be empty
        after the speculative loop terminates.
        """
        model, tokenizer = hf_model_and_tokenizer
        (
            executor, alice_identity, bob_identity,
            alice_cache, bob_cache, _, draft,
        ) = _build_speculative_two_stage(
            model, tokenizer, speculation_depth=4,
            sharded_default_max_tokens=8,
        )
        chain = _make_chain(alice_identity.node_id, bob_identity.node_id)

        events = list(executor.execute_chain_streaming(
            request=_make_request(
                prompt="The quick brown fox", max_tokens=8,
                temperature=0.0,
            ),
            chain=chain,
        ))
        # At least one StreamToken + a ChainExecutionResult.
        assert any(isinstance(e, StreamToken) for e in events)
        assert any(isinstance(e, ChainExecutionResult) for e in events)
        # Caches drained on terminal eviction broadcast.
        assert "req-spec-e2e" not in alice_cache
        assert "req-spec-e2e" not in bob_cache
        # Draft state evicted.
        assert "req-spec-e2e" not in draft

    def test_cache_evicted_on_caller_close_mid_speculation(
        self, hf_model_and_tokenizer,
    ):
        """Caller closes the generator mid-stream → finally block
        runs → both server caches + draft state cleaned up.
        """
        model, tokenizer = hf_model_and_tokenizer
        (
            executor, alice_identity, bob_identity,
            alice_cache, bob_cache, _, draft,
        ) = _build_speculative_two_stage(
            model, tokenizer, speculation_depth=4,
            sharded_default_max_tokens=8,
        )
        chain = _make_chain(alice_identity.node_id, bob_identity.node_id)

        gen = executor.execute_chain_streaming(
            request=_make_request(
                prompt="The quick brown fox", max_tokens=8,
                temperature=0.0,
            ),
            chain=chain,
        )
        # Pull the first token (PREFILL emit) then close mid-stream.
        first = next(gen)
        assert isinstance(first, StreamToken)
        # Mid-stream — caches are populated.
        assert "req-spec-e2e" in alice_cache
        assert "req-spec-e2e" in bob_cache
        # draft.reset is called before PREFILL, so by this point
        # it's also populated.
        assert "req-spec-e2e" in draft
        gen.close()
        # finally block ran — everything drained.
        assert "req-spec-e2e" not in alice_cache
        assert "req-spec-e2e" not in bob_cache
        assert "req-spec-e2e" not in draft


# ──────────────────────────────────────────────────────────────────────────
# Round-1 HIGH-1 regression coverage — partial-accept on real distilgpt2
# ──────────────────────────────────────────────────────────────────────────


class _DeliberateMismatchDraft:
    """Round-1 HIGH-1 regression coverage: a draft that proposes
    deliberately-wrong tokens to force partial-accept. Non-tail
    rollback against the cached_positions counter (NOT
    tokens_generated, which stays 0 on non-tail) is the load-
    bearing fix; this fake exercises that path.

    Implements the DraftModel Protocol with state-tracked
    history so commit/evict shape match HFDraftModel.
    """

    def __init__(self, k: int) -> None:
        self._k = k
        self._states: Dict[str, List[int]] = {}
        self.propose_calls: List[dict] = []
        self.commit_calls: List[dict] = []
        self.evict_calls: List[str] = []

    def reset(
        self, *, request_id: str, prompt_input_ids: List[int],
    ) -> None:
        self._states[request_id] = list(prompt_input_ids)

    def propose(
        self,
        *,
        request_id: str,
        parent_token_id: int,
        k: int,
        temperature: float,
    ) -> List[int]:
        self.propose_calls.append({
            "parent": parent_token_id, "k": k,
        })
        # Always propose K zeros — token_id 0 is "!" in distilgpt2,
        # essentially never the verifier's argmax for any natural
        # prompt → forces accepted_count = 0 every round.
        return [0] * k

    def commit(
        self, *, request_id: str, accepted_token_ids: List[int],
    ) -> None:
        self.commit_calls.append({
            "accepted": list(accepted_token_ids),
        })
        self._states[request_id].extend(accepted_token_ids)

    def evict(self, *, request_id: str) -> None:
        self.evict_calls.append(request_id)
        self._states.pop(request_id, None)

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._states


class TestSpeculativeE2EPartialAcceptRollback:
    """Round-1 HIGH-1 regression coverage. The pre-fix bug: rollback
    silently no-op'd on non-tail stages because the manager
    clamped on tail-only ``tokens_generated`` (stays 0 on non-
    tail). With deliberate-mismatch drafts forcing accepted_count=0
    every round, every chain stage gets a real RollbackCacheRequest
    that drops K positions; both stages' caches must end up
    correctly truncated AND the final output must still match
    single-host greedy (proves rollback didn't corrupt cache state).
    """

    def test_zero_accept_rolls_back_non_tail_cache(
        self, hf_model_and_tokenizer,
    ):
        model, tokenizer = hf_model_and_tokenizer
        prompt = "The quick brown fox"
        max_tokens = 4

        # Reference: single-host greedy.
        reference_ids = _single_host_greedy_token_ids(
            model, tokenizer, prompt, max_new_tokens=max_tokens,
        )

        # Build a 2-stage chain with our deliberate-mismatch draft.
        alice_identity = generate_node_identity("alice")
        bob_identity = generate_node_identity("bob")
        settler_identity = generate_node_identity("settler")
        anchor = _Anchor()
        anchor.register(alice_identity)
        anchor.register(bob_identity)
        anchor.register(settler_identity)

        adapter = _SpeculativeDistilGPT2Adapter(model, tokenizer)
        alice_cache = KVCacheManager()
        bob_cache = KVCacheManager()
        alice_runner = ShardedAutoregressiveRunner(
            model=adapter, layer_range=(0, 3),
            kv_cache_manager=alice_cache,
            tee_attestation=b"\x07" * 32,
            tee_type=TEEType.SOFTWARE,
        )
        bob_runner = ShardedAutoregressiveRunner(
            model=adapter, layer_range=(3, 6),
            kv_cache_manager=bob_cache,
            tee_attestation=b"\x07" * 32,
            tee_type=TEEType.SOFTWARE,
            sampling_defaults=SamplingDefaults(
                max_tokens=max_tokens, temperature=0.0,
                top_k=50, top_p=0.95,
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
            kv_cache_manager=alice_cache, sharded_runner=alice_runner,
        )
        bob_server = LayerStageServer(
            identity=bob_identity,
            registry=_Registry(
                model_id="distilgpt2", shard_ranges=[(3, 6)],
            ),
            runner=_PassthroughUnaryRunner(),
            tee_runtime=_TEERuntime(),
            anchor=anchor, clock=lambda: 1000.0,
            kv_cache_manager=bob_cache, sharded_runner=bob_runner,
        )
        servers = {
            alice_identity.node_id: alice_server,
            bob_identity.node_id: bob_server,
        }

        def send_message(addr, b):
            return servers[addr].handle(b)

        draft = _DeliberateMismatchDraft(k=4)
        executor = RpcChainExecutor(
            settler_identity=settler_identity,
            send_message=send_message,
            anchor=anchor,
            prompt_encoder=lambda p: np.array([0], dtype=np.int32),
            output_decoder=lambda a: "",
            enable_sharded_decode=True,
            tokenizer=tokenizer,
            cache_evict_send_message=send_message,
            rollback_cache_send_message=send_message,
            sharded_default_max_tokens=max_tokens,
            draft_model=draft,
            speculation_depth=4,
        )
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
        sharded_ids = [t.token_id for t in stream_tokens]

        # Greedy-equivalence holds even with all-rejected drafts —
        # accepted_count=0 emits the verifier's correction, which
        # matches reference. If non-tail rollback no-op'd (pre-fix
        # bug), Stage 1's cache would be inflated by 4 stale
        # positions and the next VERIFY would compute wrong
        # logits, breaking equivalence.
        assert sharded_ids == reference_ids, (
            f"partial-accept rollback corrupted non-tail cache: "
            f"sharded {sharded_ids} != reference {reference_ids}"
        )
        # Caches drained on terminal eviction.
        assert "req-spec-e2e" not in alice_cache
        assert "req-spec-e2e" not in bob_cache
        # Draft proposed every round (forced 0-accept means many
        # rounds for max_tokens=4). At least one commit happened.
        assert len(draft.propose_calls) >= 1
        assert len(draft.commit_calls) >= 1
