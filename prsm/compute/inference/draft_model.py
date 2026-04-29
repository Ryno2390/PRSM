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
        be fewer than K if the draft model emits EOS early. v1
        is greedy-only; ``temperature != 0.0`` raises.

        ``parent_token_id`` is the token whose successor we want
        to predict. On the first call after reset,
        ``parent_token_id`` is the FIRST token produced by the
        chain's PREFILL (the executor passes this in). On
        subsequent calls, ``parent_token_id`` is the LAST token
        the verifier emitted in the previous round (matches
        ``accepted_token_ids[-1]`` from the previous commit).
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
        if request_id not in self._states:
            raise DraftStateNotFoundError(
                f"HFDraftModel.propose: no draft state for "
                f"request_id={request_id!r}; caller must invoke "
                f"reset() at PREFILL boundary first"
            )
        if (
            isinstance(parent_token_id, bool)
            or not isinstance(parent_token_id, int)
            or parent_token_id < 0
        ):
            raise DraftModelError(
                f"HFDraftModel.propose: parent_token_id must be "
                f"non-negative int, got {parent_token_id!r}"
            )
        if (
            isinstance(k, bool)
            or not isinstance(k, int)
            or k <= 0
        ):
            raise DraftModelError(
                f"HFDraftModel.propose: k must be positive int, "
                f"got {k!r}"
            )
        if temperature != 0.0:
            raise DraftModelError(
                f"HFDraftModel.propose: v1 is greedy-only; "
                f"temperature must be 0.0, got {temperature!r}. "
                f"Sampling-correct speculation under temperature > 0 "
                f"requires the Leviathan-2023 rejection-sampling "
                f"correction — deferred to Phase 3.x.11.y.x."
            )

        state = self._states[request_id]
        # Append parent_token_id to history if not already there.
        # On the first propose after reset, history == [prompt]
        # and parent_token_id is the first chain-emitted token —
        # append. On subsequent proposes, history was extended by
        # commit() with accepted_token_ids ending in
        # parent_token_id — skip append (don't double-consume).
        if (
            not state.full_history
            or state.full_history[-1] != parent_token_id
        ):
            state.full_history.append(parent_token_id)

        # Generate K greedy tokens via HF.
        import torch
        with torch.no_grad():
            input_ids = torch.tensor(
                [state.full_history], dtype=torch.long,
            )
            output = self._model.generate(
                input_ids=input_ids,
                max_new_tokens=k,
                do_sample=False,
                temperature=1.0,  # ignored when do_sample=False
                pad_token_id=(
                    self._eos_token_id
                    if self._eos_token_id is not None else 0
                ),
            )
        full_out = output[0].tolist()
        proposed = full_out[len(state.full_history):]

        # Stop early on EOS (include the EOS marker in the
        # returned prefix so the executor knows speculation
        # truncated naturally).
        if (
            self._eos_token_id is not None
            and self._eos_token_id in proposed
        ):
            eos_idx = proposed.index(self._eos_token_id)
            proposed = proposed[: eos_idx + 1]

        return list(proposed)

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
