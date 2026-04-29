"""Phase 3.x.11.y Task 3 — ``DraftModel`` Protocol + ``HFDraftModel``
reference impl.

Speculative decoding requires a small, fast **draft model** that
proposes K candidate tokens per step. The full sharded chain
(verifier) verifies them in a single batched forward pass; the
executor accepts the longest matching prefix. On accept,
K+1 tokens delivered in one chain pass instead of K-separate
dispatches; on reject, fall back to standard one-token decode.

The draft model is co-located with the executor (or with the
tail stage; v1 co-locates with executor for simplicity).
Stateful — caller resets at PREFILL boundaries; calls
``propose`` per speculation round; calls ``commit`` to roll the
draft state forward to match the verified prefix; calls
``evict`` on terminal/cancellation.

**v1 ships greedy-only.** ``temperature == 0`` triggers
deterministic ``argmax`` proposal — output identical to
non-speculative greedy decode (speculation is a perf
optimization, not a sampling change). Sampling-correct
speculation under ``temperature > 0`` requires the
[Leviathan et al. 2023] rejection-sampling correction —
deferred to Phase 3.x.11.y.x.

Reference impl ``HFDraftModel`` wraps an HF
``AutoModelForCausalLM`` and drives ``model.generate()`` per
propose call. v1 is **stateless w.r.t. the model's KV cache** —
each propose runs a fresh prefill + K decode steps via
``generate()``. This is correct + testable + simple but trades
speed for clarity; operators wanting a perf-optimized draft can
implement the same Protocol against a stateful KV cache (e.g.,
HF ``DynamicCache``) without modifying the executor.

What this module does NOT do:
  - It does NOT verify draft proposals against the chain.
    That's the executor's job (Phase 3.x.11.y Task 5
    speculation loop).
  - It does NOT manage the chain-side cache. That's
    ``KVCacheManager`` (Phase 3.x.11 Task 2) +
    ``KVCacheManager.rollback`` (Phase 3.x.11.y Task 2).
  - It does NOT handle sampling under temperature > 0.
    Phase 3.x.11.y.x.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


__all__ = [
    "DraftModel",
    "HFDraftModel",
    "DraftModelError",
    "DraftStateNotFoundError",
]


class DraftModelError(RuntimeError):
    """Base for draft-model lifecycle failures."""


class DraftStateNotFoundError(DraftModelError):
    """Raised on ``propose`` / ``commit`` for a request_id that
    was never ``reset()``. Caller bug — the executor's
    speculation loop must call ``reset`` at PREFILL boundary
    before any ``propose``."""


# ──────────────────────────────────────────────────────────────────────────
# DraftModel Protocol
# ──────────────────────────────────────────────────────────────────────────


class DraftModel(Protocol):
    """Protocol for draft-and-verify speculative decoding's
    draft side.

    Lifecycle:
      reset(request_id, prompt_input_ids)
        ↓
      propose(request_id, parent_token_id, k, temperature) → List[int]
        ↓ (verifier accepts some prefix)
      commit(request_id, accepted_token_ids)
        ↓
      propose(request_id, parent_token_id=last_accepted_token,
              k, temperature) → ...
        ↓
      ... (repeat propose/commit until terminal) ...
        ↓
      evict(request_id)

    Every method MUST be safe to call concurrently across
    different ``request_id`` values; concurrent calls for the
    SAME ``request_id`` are caller-serialized (the executor's
    speculation loop is single-threaded per request).
    """

    def reset(
        self,
        *,
        request_id: str,
        prompt_input_ids: List[int],
    ) -> None:
        """Initialize per-request draft state from the prompt's
        input ids. The draft is now ready to ``propose``.
        Called once at PREFILL boundary."""
        ...

    def propose(
        self,
        *,
        request_id: str,
        parent_token_id: int,
        k: int,
        temperature: float,
    ) -> List[int]:
        """Propose up to K speculative tokens following
        ``parent_token_id``. Returns ``up to K`` token ids — may
        be fewer than K if the draft model emits EOS early.

        v1 (Phase 3.x.11.y) is greedy-only; ``temperature != 0.0``
        raises. v2 (Phase 3.x.11.y.x) supports ``temperature > 0``
        but callers should prefer ``propose_with_probs`` to also
        receive the per-token sampling probabilities required by
        the Leviathan-2023 rejection-sampling correction. Without
        the probs, the executor falls back to v1 greedy prefix-
        match logic regardless of the configured temperature.

        ``parent_token_id`` is the token whose successor we want
        to predict. On the first call after reset,
        ``parent_token_id`` is the FIRST token produced by the
        chain's PREFILL (the executor passes this in). On
        subsequent calls, ``parent_token_id`` is the LAST token
        the verifier emitted in the previous round (matches
        ``accepted_token_ids[-1]`` from the previous commit).
        """
        ...

    def propose_with_probs(
        self,
        *,
        request_id: str,
        parent_token_id: int,
        k: int,
        temperature: float,
    ) -> Tuple[List[int], List[float]]:
        """Phase 3.x.11.y.x — stochastic propose. Returns
        ``(proposed_ids, proposed_probs)`` where
        ``proposed_probs[i]`` is the probability ``q(d_i)`` the
        draft model assigned to ``proposed_ids[i]`` at the i-th
        sampling step. Greedy path (``temperature == 0.0``)
        returns probabilities of 1.0 per token (degenerate
        under argmax sampling). Stochastic path (``temperature
        > 0``) returns ``softmax(scaled_logits)[id]`` per
        position after any top-k / top-p filtering.

        ``len(proposed_probs) == len(proposed_ids)`` always.
        Each prob in ``[0, 1]``; greedy reports 1.0 exactly.

        Used by the executor's speculation loop to populate
        ``RunLayerSliceRequest.proposed_token_probs`` so the
        tail can run Leviathan-2023 rejection sampling against
        the verifier's distribution.
        """
        ...

    def commit(
        self,
        *,
        request_id: str,
        accepted_token_ids: List[int],
    ) -> None:
        """Caller indicates which tokens the verifier emitted
        from the previous propose round.
        ``accepted_token_ids`` length == accepted_count + 1
        (the +1 is the verifier's correction or bonus token).
        Draft state advances to "just consumed
        ``accepted_token_ids``"."""
        ...

    def evict(self, *, request_id: str) -> None:
        """Per-request cleanup. Idempotent — calling on an
        unknown ``request_id`` is a no-op."""
        ...


# ──────────────────────────────────────────────────────────────────────────
# HFDraftModel reference impl
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class _DraftState:
    """Per-request canonical history. The HFDraftModel keeps the
    full sequence of consumed tokens (prompt + first chain
    output + accepted tokens from each verify round); each
    ``propose`` re-runs ``model.generate`` starting from this
    history. v1 simplification — a stateful-KV-cache impl is a
    drop-in replacement that maintains an HF ``DynamicCache``
    instead.
    """

    full_history: List[int]


class HFDraftModel:
    """Reference ``DraftModel`` impl using HuggingFace's
    ``AutoModelForCausalLM.generate``. Wraps a small co-located
    model whose distribution should track the verifier's at the
    top-K positions for high accept-rate.

    Constructor:
      model              HF ``AutoModelForCausalLM`` (or
                         compatible). Must expose
                         ``.generate(input_ids, max_new_tokens,
                         do_sample, ...)``.
      eos_token_id       Optional; when set, ``propose`` stops
                         early if the draft emits this token
                         (returns the prefix INCLUDING the EOS
                         marker). Operators wire this from the
                         tokenizer.

    State management: each request_id maps to a ``_DraftState``
    holding the canonical history. Stateless w.r.t. the model's
    KV cache — each propose call re-runs prefill via
    ``model.generate``. v2 perf upgrade swaps in HF
    ``DynamicCache`` per-request without changing the
    Protocol-level public API.
    """

    def __init__(
        self,
        *,
        model: Any,
        eos_token_id: Optional[int] = None,
    ) -> None:
        if model is None or not callable(getattr(model, "generate", None)):
            raise DraftModelError(
                "HFDraftModel: model must expose .generate(...) — "
                "see HF AutoModelForCausalLM"
            )
        if eos_token_id is not None and (
            isinstance(eos_token_id, bool)
            or not isinstance(eos_token_id, int)
            or eos_token_id < 0
        ):
            raise DraftModelError(
                f"HFDraftModel: eos_token_id must be non-negative "
                f"int or None, got {eos_token_id!r}"
            )
        self._model = model
        self._eos_token_id = eos_token_id
        self._states: Dict[str, _DraftState] = {}

    # ── DraftModel Protocol ───────────────────────────────────────────

    def reset(
        self,
        *,
        request_id: str,
        prompt_input_ids: List[int],
    ) -> None:
        if not isinstance(request_id, str) or not request_id:
            raise DraftModelError(
                "HFDraftModel.reset: request_id must be non-empty "
                "string"
            )
        if (
            not isinstance(prompt_input_ids, list)
            or not prompt_input_ids
        ):
            raise DraftModelError(
                "HFDraftModel.reset: prompt_input_ids must be "
                "non-empty list of ints"
            )
        for tok in prompt_input_ids:
            if isinstance(tok, bool) or not isinstance(tok, int):
                raise DraftModelError(
                    f"HFDraftModel.reset: prompt_input_ids entries "
                    f"must be int, got {type(tok).__name__}"
                )
        self._states[request_id] = _DraftState(
            full_history=list(prompt_input_ids),
        )

    def propose(
        self,
        *,
        request_id: str,
        parent_token_id: int,
        k: int,
        temperature: float,
    ) -> List[int]:
        """Thin wrapper around ``propose_with_probs`` that
        discards the probabilities. Kept for v1 callers (Phase
        3.x.11.y greedy path); v2 callers (Phase 3.x.11.y.x
        stochastic path) should prefer ``propose_with_probs``
        directly.
        """
        ids, _probs = self.propose_with_probs(
            request_id=request_id,
            parent_token_id=parent_token_id,
            k=k,
            temperature=temperature,
        )
        return ids

    def propose_with_probs(
        self,
        *,
        request_id: str,
        parent_token_id: int,
        k: int,
        temperature: float,
    ) -> Tuple[List[int], List[float]]:
        """Phase 3.x.11.y.x reference impl. Returns
        ``(proposed_ids, proposed_probs)`` per the ``DraftModel``
        Protocol contract.

        Greedy (``temperature == 0.0``) — uses ``do_sample=False``
        to argmax-decode K tokens, reports probability 1.0 per
        token (degenerate under argmax). v1 callers via
        ``propose`` get an identical id sequence to Phase
        3.x.11.y semantics.

        Stochastic (``temperature > 0``) — uses
        ``do_sample=True`` with the requested temperature plus
        any top-k / top-p filtering configured on the model's
        generation_config (HF defaults preserved). Reports
        ``softmax(scaled_logits)[sampled_id]`` per step using
        ``output_scores=True``. The scores HF returns are
        post-temperature/top-k/top-p (``LogitsProcessor`` chain
        output), so the softmax + lookup is the exact sampling
        probability the model used.
        """
        if request_id not in self._states:
            raise DraftStateNotFoundError(
                f"HFDraftModel.propose_with_probs: no draft state "
                f"for request_id={request_id!r}; caller must "
                f"invoke reset() at PREFILL boundary first"
            )
        if (
            isinstance(parent_token_id, bool)
            or not isinstance(parent_token_id, int)
            or parent_token_id < 0
        ):
            raise DraftModelError(
                f"HFDraftModel.propose_with_probs: parent_token_id "
                f"must be non-negative int, got {parent_token_id!r}"
            )
        if (
            isinstance(k, bool)
            or not isinstance(k, int)
            or k <= 0
        ):
            raise DraftModelError(
                f"HFDraftModel.propose_with_probs: k must be "
                f"positive int, got {k!r}"
            )
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or temperature < 0.0
        ):
            raise DraftModelError(
                f"HFDraftModel.propose_with_probs: temperature "
                f"must be non-negative number, got {temperature!r}"
            )

        state = self._states[request_id]
        # Append parent_token_id to history if not already there
        # (same logic as v1 — handles the fresh-after-reset vs
        # after-commit cases).
        if (
            not state.full_history
            or state.full_history[-1] != parent_token_id
        ):
            state.full_history.append(parent_token_id)

        import torch
        do_sample = float(temperature) > 0.0
        with torch.no_grad():
            input_ids = torch.tensor(
                [state.full_history], dtype=torch.long,
            )
            gen_kwargs = dict(
                input_ids=input_ids,
                max_new_tokens=k,
                do_sample=do_sample,
                pad_token_id=(
                    self._eos_token_id
                    if self._eos_token_id is not None else 0
                ),
                output_scores=True,
                return_dict_in_generate=True,
            )
            if do_sample:
                gen_kwargs["temperature"] = float(temperature)
            else:
                # do_sample=False ignores temperature; pass 1.0
                # to avoid HF warnings about "temperature ignored
                # when do_sample=False".
                gen_kwargs["temperature"] = 1.0
            output = self._model.generate(**gen_kwargs)

        # GenerateOutput has .sequences (full-history + new) and
        # .scores (tuple of K logits tensors, post-LogitsProcessor).
        sequences = output.sequences[0].tolist()
        scores = output.scores  # tuple, len == k_actual <= k
        proposed_full = sequences[len(state.full_history):]

        # Compute per-token probabilities. For each step i:
        # - greedy: prob = 1.0 (argmax has all the mass under
        #   degenerate point-mass).
        # - stochastic: prob = softmax(scores[i])[proposed_full[i]]
        probs: List[float] = []
        if do_sample:
            for i, tok in enumerate(proposed_full):
                step_logits = scores[i][0]  # [vocab]
                step_probs = torch.softmax(step_logits, dim=-1)
                probs.append(float(step_probs[int(tok)].item()))
        else:
            probs = [1.0] * len(proposed_full)

        # Stop early on EOS (include the EOS marker; trim the
        # probs to match).
        if (
            self._eos_token_id is not None
            and self._eos_token_id in proposed_full
        ):
            eos_idx = proposed_full.index(self._eos_token_id)
            proposed_full = proposed_full[: eos_idx + 1]
            probs = probs[: eos_idx + 1]

        return list(proposed_full), probs

    def commit(
        self,
        *,
        request_id: str,
        accepted_token_ids: List[int],
    ) -> None:
        if request_id not in self._states:
            raise DraftStateNotFoundError(
                f"HFDraftModel.commit: no draft state for "
                f"request_id={request_id!r}"
            )
        if not isinstance(accepted_token_ids, list):
            raise DraftModelError(
                f"HFDraftModel.commit: accepted_token_ids must be "
                f"list, got {type(accepted_token_ids).__name__}"
            )
        if not accepted_token_ids:
            raise DraftModelError(
                "HFDraftModel.commit: accepted_token_ids must be "
                "non-empty (at minimum the verifier's correction "
                "token; never an empty list)"
            )
        for tok in accepted_token_ids:
            if isinstance(tok, bool) or not isinstance(tok, int):
                raise DraftModelError(
                    f"HFDraftModel.commit: accepted_token_ids "
                    f"entries must be int, got "
                    f"{type(tok).__name__}"
                )
        state = self._states[request_id]
        # Append accepted tokens to canonical history. On the
        # next propose, the draft starts from this extended
        # history. v1 simplification: no KV-cache rollback —
        # generate() will re-run prefill from the new history
        # length on the next propose call.
        state.full_history.extend(accepted_token_ids)

    def evict(self, *, request_id: str) -> None:
        # Idempotent — pop returns None when key absent.
        self._states.pop(request_id, None)

    # ── observability helpers (operator-side metrics) ─────────────────

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._states

    def __len__(self) -> int:
        return len(self._states)
