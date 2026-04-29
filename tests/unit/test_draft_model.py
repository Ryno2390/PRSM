"""Phase 3.x.11.y Task 3 — DraftModel Protocol + HFDraftModel tests.

Covers:
  - HFDraftModel constructor validation (model shape,
    eos_token_id range)
  - reset/propose/commit/evict lifecycle
  - propose returns up to K tokens
  - propose stops early on EOS
  - commit advances state (next propose resumes from accepted
    tokens)
  - commit-fewer-than-K rolls forward by accepted count only
  - evict idempotent
  - Greedy-only temperature gate raises on temperature > 0
  - DraftStateNotFoundError on propose/commit before reset

Unit tests use a deterministic ``_FakeHFModel`` whose
``.generate()`` returns scripted token sequences. The real
HFDraftModel + distilgpt2 E2E lives in Task 7's slow integration
test.
"""

from __future__ import annotations

from typing import Any, List

import pytest

from prsm.compute.inference.draft_model import (
    DraftModelError,
    DraftStateNotFoundError,
    HFDraftModel,
)


# ──────────────────────────────────────────────────────────────────────────
# Fake HF model
# ──────────────────────────────────────────────────────────────────────────


class _FakeRow:
    def __init__(self, vals):
        self._vals = vals

    def tolist(self):
        return list(self._vals)


class _FakeSequencesTensor:
    """[1, seq_len] sequences tensor mock; supports
    ``output.sequences[0].tolist()`` access pattern."""

    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, idx):
        return _FakeRow(self._rows[idx])


class _FakeScoreTensor:
    """[1, vocab] score tensor mock for one decoding step.
    Indexable as ``score[0]`` → [vocab] row; supports
    ``torch.softmax(...)[token_id].item()`` via the score's
    own ``softmax_at_id`` helper (the test path uses
    monkeypatched ``torch.softmax`` to invoke it).
    """

    def __init__(self, target_probs):
        # target_probs is dict[token_id, prob] for the tokens
        # we care about; missing tokens have prob ~ 0.
        self._target_probs = dict(target_probs)

    def __getitem__(self, idx):
        # score[0] → returns self (we represent the row directly).
        return self

    def softmax_prob(self, token_id):
        return float(self._target_probs.get(int(token_id), 0.0))


class _FakeGenerateOutput:
    """GenerateOutput mock with .sequences + .scores attributes."""

    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.scores = scores


class _FakeHFModel:
    """Deterministic HF-shaped stub. ``.generate()`` extends
    input_ids with a scripted per-position next-token map AND
    (Phase 3.x.11.y.x) optionally with a per-position score
    map for stochastic propose_with_probs.

    Greedy-only constructor ``_FakeHFModel(token_script)``: the
    script is ``(input_ids, k) -> List[int]`` returning the K
    new tokens. Probabilities are not modeled (greedy reports
    1.0 per token in the runner, doesn't call softmax).

    Stochastic constructor ``_FakeHFModel.stochastic(script)``:
    the script is ``(input_ids, k) -> List[Tuple[int, dict]]``
    returning `(token_id, target_probs_dict)` per step where
    `target_probs_dict[token_id]` is the probability the
    runner should observe via softmax-lookup. Other vocab ids
    in the dict are unused by the rejection-sampling test
    path but ride through faithfully.
    """

    def __init__(self, script: Any) -> None:
        self._script = script
        self._stochastic_script: Any = None
        self.generate_calls: List[List[int]] = []

    @classmethod
    def stochastic(cls, stochastic_script: Any) -> "_FakeHFModel":
        m = cls(lambda _hist, _k: [])  # placeholder; unused
        m._stochastic_script = stochastic_script
        return m

    def generate(
        self,
        *,
        input_ids,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float = 1.0,
        pad_token_id: int = 0,
        output_scores: bool = False,
        return_dict_in_generate: bool = False,
    ):
        # input_ids is a torch.Tensor of shape [1, seq_len];
        # convert to list-of-int for the script.
        in_list = input_ids[0].tolist()
        self.generate_calls.append(in_list)

        # Choose script branch by do_sample.
        if do_sample:
            if self._stochastic_script is None:
                raise AssertionError(
                    "_FakeHFModel: do_sample=True requires the "
                    "stochastic script — use _FakeHFModel.stochastic(...)"
                )
            script_out = self._stochastic_script(in_list, max_new_tokens)
            # script_out: List[Tuple[int, dict]]
            new_tokens = [t for (t, _p) in script_out[:max_new_tokens]]
            scores = tuple(
                _FakeScoreTensor(p) for (_t, p) in script_out[:max_new_tokens]
            )
        else:
            new_tokens = self._script(in_list, max_new_tokens)[:max_new_tokens]
            scores = tuple(_FakeScoreTensor({}) for _ in new_tokens)

        full = list(in_list) + list(new_tokens)
        if return_dict_in_generate:
            return _FakeGenerateOutput(
                sequences=_FakeSequencesTensor([full]),
                scores=scores,
            )
        # Legacy pre-3.x.11.y.x path (unused by current runner).
        return _FakeSequencesTensor([full])


class _FakeTorchTensor:
    """Stub for the test path's torch.tensor — wraps a list +
    exposes ``[0].tolist()`` for the FakeHFModel."""

    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, idx):
        class _Row:
            def __init__(self, vals):
                self._vals = vals

            def tolist(self):
                return list(self._vals)

        return _Row(self._rows[idx])


# ──────────────────────────────────────────────────────────────────────────
# torch.tensor + torch.no_grad shim
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _shim_torch(monkeypatch):
    """The HFDraftModel imports torch lazily inside propose. Tests
    use the fake model + don't need real torch; shim
    ``torch.tensor`` + ``torch.no_grad`` + ``torch.long`` to
    side-step the dependency."""
    import sys
    import types

    if "torch" in sys.modules:
        # Real torch available — most ops (torch.tensor,
        # torch.no_grad) work with the fake model because it
        # doesn't do tensor math. But ``torch.softmax`` rejects
        # ``_FakeScoreTensor`` since it isn't a real Tensor.
        # Patch only the softmax slot to the fake-aware shim
        # below; leave the rest of torch alone so other tests
        # that share the module are unaffected.
        real_torch = sys.modules["torch"]

        class _FakeSoftmaxResult:
            def __init__(self, score_tensor):
                self._score = score_tensor

            def __getitem__(self, idx):
                class _Scalar:
                    def __init__(self, val):
                        self._val = val

                    def item(self):
                        return self._val

                sp = getattr(self._score, "softmax_prob", None)
                if sp is None:
                    return _Scalar(0.0)
                return _Scalar(sp(int(idx)))

        def _fake_softmax(t, dim=-1):
            return _FakeSoftmaxResult(t)

        real_softmax = getattr(real_torch, "softmax", None)
        monkeypatch.setattr(real_torch, "softmax", _fake_softmax)
        yield
        return

    fake_torch = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _fake_tensor(data, dtype=None):
        return _FakeTorchTensor(data)

    class _FakeSoftmaxResult:
        """Result of torch.softmax — the runner indexes it as
        ``probs[id].item()``. We delegate to the FakeScoreTensor's
        own ``softmax_prob`` helper to look up the test's
        scripted probability for that token id.
        """

        def __init__(self, score_tensor):
            self._score = score_tensor

        def __getitem__(self, idx):
            class _Scalar:
                def __init__(self, val):
                    self._val = val

                def item(self):
                    return self._val

            # Use the score tensor's softmax_prob if available
            # (FakeScoreTensor); otherwise return 0.0 (greedy
            # path doesn't reach this code under do_sample=False).
            sp = getattr(self._score, "softmax_prob", None)
            if sp is None:
                return _Scalar(0.0)
            return _Scalar(sp(int(idx)))

    def _fake_softmax(t, dim=-1):
        return _FakeSoftmaxResult(t)

    fake_torch.no_grad = _NoGrad
    fake_torch.tensor = _fake_tensor
    fake_torch.long = "long"
    fake_torch.softmax = _fake_softmax
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    yield


# ──────────────────────────────────────────────────────────────────────────
# Constructor validation
# ──────────────────────────────────────────────────────────────────────────


class TestConstructorValidation:
    def test_rejects_none_model(self):
        with pytest.raises(DraftModelError, match="model must"):
            HFDraftModel(model=None)

    def test_rejects_model_missing_generate(self):
        class _Bad:
            pass
        with pytest.raises(DraftModelError, match="generate"):
            HFDraftModel(model=_Bad())

    def test_accepts_eos_none(self):
        m = HFDraftModel(
            model=_FakeHFModel(lambda _, k: [99] * k),
        )
        assert m._eos_token_id is None

    def test_accepts_eos_zero(self):
        m = HFDraftModel(
            model=_FakeHFModel(lambda _, k: [99] * k),
            eos_token_id=0,
        )
        assert m._eos_token_id == 0

    def test_rejects_negative_eos(self):
        with pytest.raises(DraftModelError, match="eos_token_id"):
            HFDraftModel(
                model=_FakeHFModel(lambda _, k: []),
                eos_token_id=-1,
            )

    def test_rejects_bool_eos(self):
        with pytest.raises(DraftModelError, match="eos_token_id"):
            HFDraftModel(
                model=_FakeHFModel(lambda _, k: []),
                eos_token_id=True,  # type: ignore[arg-type]
            )


# ──────────────────────────────────────────────────────────────────────────
# Lifecycle: reset
# ──────────────────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_initializes_state(self):
        m = HFDraftModel(model=_FakeHFModel(lambda _, k: [9] * k))
        m.reset(request_id="r1", prompt_input_ids=[1, 2, 3])
        assert "r1" in m
        assert len(m) == 1

    def test_reset_rejects_empty_request_id(self):
        m = HFDraftModel(model=_FakeHFModel(lambda _, k: []))
        with pytest.raises(DraftModelError, match="request_id"):
            m.reset(request_id="", prompt_input_ids=[1, 2])

    def test_reset_rejects_empty_prompt(self):
        m = HFDraftModel(model=_FakeHFModel(lambda _, k: []))
        with pytest.raises(DraftModelError, match="prompt_input_ids"):
            m.reset(request_id="r1", prompt_input_ids=[])

    def test_reset_rejects_non_int_prompt_token(self):
        m = HFDraftModel(model=_FakeHFModel(lambda _, k: []))
        with pytest.raises(DraftModelError, match="must be int"):
            m.reset(
                request_id="r1",
                prompt_input_ids=[1, "two", 3],  # type: ignore[list-item]
            )

    def test_reset_rejects_bool_prompt_token(self):
        m = HFDraftModel(model=_FakeHFModel(lambda _, k: []))
        with pytest.raises(DraftModelError, match="must be int"):
            m.reset(
                request_id="r1",
                prompt_input_ids=[1, True, 3],  # type: ignore[list-item]
            )


# ──────────────────────────────────────────────────────────────────────────
# Lifecycle: propose
# ──────────────────────────────────────────────────────────────────────────


class TestPropose:
    def test_propose_returns_k_tokens(self):
        # Script returns [10, 20, 30, 40] for K=4.
        m = HFDraftModel(
            model=_FakeHFModel(lambda _, k: [10, 20, 30, 40][:k]),
        )
        m.reset(request_id="r1", prompt_input_ids=[1, 2, 3])
        out = m.propose(
            request_id="r1",
            parent_token_id=5,
            k=4,
            temperature=0.0,
        )
        assert out == [10, 20, 30, 40]

    def test_propose_appends_parent_to_history_first_time(self):
        # On first propose after reset, parent_token_id is
        # appended to history before generate. Subsequent calls
        # to model.generate should see [prompt, parent].
        fake = _FakeHFModel(lambda hist, k: [99] * k)
        m = HFDraftModel(model=fake)
        m.reset(request_id="r1", prompt_input_ids=[1, 2, 3])
        m.propose(
            request_id="r1",
            parent_token_id=5,
            k=4,
            temperature=0.0,
        )
        # generate() called with [prompt, parent_token_id].
        assert fake.generate_calls[0] == [1, 2, 3, 5]

    def test_propose_stops_early_on_eos(self):
        # Script returns [10, 999, 20, 30] but EOS=999 → propose
        # returns [10, 999] (includes EOS marker).
        fake = _FakeHFModel(lambda _, k: [10, 999, 20, 30][:k])
        m = HFDraftModel(model=fake, eos_token_id=999)
        m.reset(request_id="r1", prompt_input_ids=[1, 2, 3])
        out = m.propose(
            request_id="r1",
            parent_token_id=5,
            k=4,
            temperature=0.0,
        )
        assert out == [10, 999]

    def test_propose_no_eos_when_eos_token_id_unset(self):
        # Same model returning 999 in output, but eos_token_id
        # is None — propose returns full list.
        fake = _FakeHFModel(lambda _, k: [10, 999, 20, 30][:k])
        m = HFDraftModel(model=fake, eos_token_id=None)
        m.reset(request_id="r1", prompt_input_ids=[1, 2, 3])
        out = m.propose(
            request_id="r1",
            parent_token_id=5,
            k=4,
            temperature=0.0,
        )
        assert out == [10, 999, 20, 30]

    def test_propose_raises_on_negative_temperature(self):
        # Phase 3.x.11.y.x lifted the greedy-only gate. The runner
        # now accepts ``temperature > 0`` (routes to do_sample=True
        # in propose_with_probs); negative temperatures remain
        # invalid and raise.
        m = HFDraftModel(model=_FakeHFModel(lambda _, k: []))
        m.reset(request_id="r1", prompt_input_ids=[1, 2, 3])
        with pytest.raises(DraftModelError, match="non-negative"):
            m.propose(
                request_id="r1",
                parent_token_id=5,
                k=4,
                temperature=-0.5,
            )

    def test_propose_raises_on_unknown_request_id(self):
        m = HFDraftModel(model=_FakeHFModel(lambda _, k: []))
        with pytest.raises(
            DraftStateNotFoundError, match="reset",
        ):
            m.propose(
                request_id="never-reset",
                parent_token_id=5,
                k=4,
                temperature=0.0,
            )

    def test_propose_rejects_non_positive_k(self):
        m = HFDraftModel(model=_FakeHFModel(lambda _, k: []))
        m.reset(request_id="r1", prompt_input_ids=[1, 2])
        with pytest.raises(DraftModelError, match="k must"):
            m.propose(
                request_id="r1",
                parent_token_id=5,
                k=0,
                temperature=0.0,
            )

    def test_propose_rejects_bool_k(self):
        m = HFDraftModel(model=_FakeHFModel(lambda _, k: []))
        m.reset(request_id="r1", prompt_input_ids=[1, 2])
        with pytest.raises(DraftModelError, match="k must"):
            m.propose(
                request_id="r1",
                parent_token_id=5,
                k=True,  # type: ignore[arg-type]
                temperature=0.0,
            )

    def test_propose_rejects_negative_parent(self):
        m = HFDraftModel(model=_FakeHFModel(lambda _, k: []))
        m.reset(request_id="r1", prompt_input_ids=[1, 2])
        with pytest.raises(DraftModelError, match="parent_token_id"):
            m.propose(
                request_id="r1",
                parent_token_id=-1,
                k=4,
                temperature=0.0,
            )


# ──────────────────────────────────────────────────────────────────────────
# Lifecycle: commit
# ──────────────────────────────────────────────────────────────────────────


class TestCommit:
    def test_commit_advances_state(self):
        # propose extends history with [parent, d_1..d_K];
        # commit replaces tail with accepted tokens.
        # Subsequent propose should generate from the
        # post-commit history.
        fake = _FakeHFModel(lambda _, k: [10, 20, 30, 40][:k])
        m = HFDraftModel(model=fake)
        m.reset(request_id="r1", prompt_input_ids=[1, 2, 3])
        m.propose(
            request_id="r1", parent_token_id=5, k=4,
            temperature=0.0,
        )
        # Commit accepts [10, 20, 99] (verifier's emitted tokens
        # = matching prefix [10, 20] + correction 99).
        m.commit(
            request_id="r1",
            accepted_token_ids=[10, 20, 99],
        )
        # Next propose — history should now be
        # [1, 2, 3, 5, 10, 20, 99]. parent_token_id=99 already
        # the last token in history → no double-append.
        m.propose(
            request_id="r1",
            parent_token_id=99,
            k=4,
            temperature=0.0,
        )
        assert fake.generate_calls[1] == [1, 2, 3, 5, 10, 20, 99]

    def test_commit_with_full_accept_plus_bonus(self):
        # All K accepted + bonus token = K+1 tokens emitted.
        fake = _FakeHFModel(lambda _, k: [10, 20, 30, 40][:k])
        m = HFDraftModel(model=fake)
        m.reset(request_id="r1", prompt_input_ids=[1, 2, 3])
        m.propose(
            request_id="r1", parent_token_id=5, k=4,
            temperature=0.0,
        )
        # Verifier accepted all 4 + emitted bonus 50 → 5 tokens.
        m.commit(
            request_id="r1",
            accepted_token_ids=[10, 20, 30, 40, 50],
        )
        m.propose(
            request_id="r1", parent_token_id=50, k=4,
            temperature=0.0,
        )
        assert fake.generate_calls[1] == [1, 2, 3, 5, 10, 20, 30, 40, 50]

    def test_commit_with_zero_accept(self):
        # Verifier rejected ALL drafts; emitted only its
        # correction token. accepted_token_ids has length 1.
        fake = _FakeHFModel(lambda _, k: [10, 20, 30, 40][:k])
        m = HFDraftModel(model=fake)
        m.reset(request_id="r1", prompt_input_ids=[1, 2, 3])
        m.propose(
            request_id="r1", parent_token_id=5, k=4,
            temperature=0.0,
        )
        # Single token = the verifier's correction.
        m.commit(
            request_id="r1",
            accepted_token_ids=[77],
        )
        m.propose(
            request_id="r1", parent_token_id=77, k=4,
            temperature=0.0,
        )
        # Post-commit history should be [prompt, parent_for_round_1, 77].
        # parent_for_round_1 was 5; the K drafts (10, 20, 30, 40)
        # were rejected; verifier emitted [77].
        # commit appends 77; propose's parent=77 matches last → no double.
        assert fake.generate_calls[1] == [1, 2, 3, 5, 77]

    def test_commit_rejects_empty_accepted(self):
        m = HFDraftModel(model=_FakeHFModel(lambda _, k: []))
        m.reset(request_id="r1", prompt_input_ids=[1, 2])
        with pytest.raises(DraftModelError, match="non-empty"):
            m.commit(request_id="r1", accepted_token_ids=[])

    def test_commit_rejects_unknown_request_id(self):
        m = HFDraftModel(model=_FakeHFModel(lambda _, k: []))
        with pytest.raises(DraftStateNotFoundError):
            m.commit(
                request_id="never-reset",
                accepted_token_ids=[10],
            )

    def test_commit_rejects_non_list(self):
        m = HFDraftModel(model=_FakeHFModel(lambda _, k: []))
        m.reset(request_id="r1", prompt_input_ids=[1, 2])
        with pytest.raises(DraftModelError, match="must be list"):
            m.commit(
                request_id="r1",
                accepted_token_ids="not a list",  # type: ignore[arg-type]
            )

    def test_commit_rejects_bool_token(self):
        m = HFDraftModel(model=_FakeHFModel(lambda _, k: []))
        m.reset(request_id="r1", prompt_input_ids=[1, 2])
        with pytest.raises(DraftModelError, match="entries must be int"):
            m.commit(
                request_id="r1",
                accepted_token_ids=[10, True, 30],  # type: ignore[list-item]
            )


# ──────────────────────────────────────────────────────────────────────────
# Lifecycle: evict
# ──────────────────────────────────────────────────────────────────────────


class TestEvict:
    def test_evict_removes_state(self):
        m = HFDraftModel(model=_FakeHFModel(lambda _, k: []))
        m.reset(request_id="r1", prompt_input_ids=[1, 2])
        assert "r1" in m
        m.evict(request_id="r1")
        assert "r1" not in m

    def test_evict_idempotent_unknown(self):
        m = HFDraftModel(model=_FakeHFModel(lambda _, k: []))
        # Should NOT raise on unknown request_id.
        m.evict(request_id="never-allocated")

    def test_evict_idempotent_double(self):
        m = HFDraftModel(model=_FakeHFModel(lambda _, k: []))
        m.reset(request_id="r1", prompt_input_ids=[1, 2])
        m.evict(request_id="r1")
        m.evict(request_id="r1")  # no-op the second time
        assert len(m) == 0

    def test_evict_then_propose_raises(self):
        m = HFDraftModel(model=_FakeHFModel(lambda _, k: []))
        m.reset(request_id="r1", prompt_input_ids=[1, 2])
        m.evict(request_id="r1")
        with pytest.raises(DraftStateNotFoundError):
            m.propose(
                request_id="r1", parent_token_id=5, k=4,
                temperature=0.0,
            )


# ──────────────────────────────────────────────────────────────────────────
# Cross-request isolation
# ──────────────────────────────────────────────────────────────────────────


class TestMultipleRequests:
    def test_independent_state_per_request_id(self):
        fake = _FakeHFModel(lambda _, k: [99] * k)
        m = HFDraftModel(model=fake)
        m.reset(request_id="r1", prompt_input_ids=[1, 2, 3])
        m.reset(request_id="r2", prompt_input_ids=[10, 20])
        m.propose(
            request_id="r1", parent_token_id=5, k=2,
            temperature=0.0,
        )
        m.propose(
            request_id="r2", parent_token_id=50, k=2,
            temperature=0.0,
        )
        # Two independent generate calls, each from its own
        # history.
        assert fake.generate_calls[0] == [1, 2, 3, 5]
        assert fake.generate_calls[1] == [10, 20, 50]
        # Eviction of one doesn't affect the other.
        m.evict(request_id="r1")
        assert "r1" not in m
        assert "r2" in m


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.11.y.x Task 2 — propose_with_probs
# ──────────────────────────────────────────────────────────────────────────


class TestProposeWithProbs:
    """Phase 3.x.11.y.x Task 2 — sampling-correct draft probs.

    Covers:
      - Greedy (temperature=0.0) returns 1.0 per token (degenerate)
      - Stochastic (temperature>0) returns softmax(scores)[id] per
        position via FakeScoreTensor.softmax_prob lookup
      - Length match: len(probs) == len(ids) always
      - EOS truncation trims both ids AND probs symmetrically
      - propose() thin-wrapper discards probs, returns identical
        ids to propose_with_probs's first element
      - Negative temperature raises (only validation now; positive
        no longer raises after Phase 3.x.11.y.x lifted greedy gate)
    """

    def test_greedy_returns_one_point_zero_per_token(self):
        m = HFDraftModel(model=_FakeHFModel(lambda _, k: [10, 20, 30]))
        m.reset(request_id="r1", prompt_input_ids=[1, 2, 3])
        ids, probs = m.propose_with_probs(
            request_id="r1", parent_token_id=5, k=3,
            temperature=0.0,
        )
        assert ids == [10, 20, 30]
        assert probs == [1.0, 1.0, 1.0]

    def test_stochastic_returns_softmax_probs(self):
        # Stochastic script returns scripted (token, target_probs)
        # per step. The runner's softmax-lookup recovers
        # target_probs[token].
        fake = _FakeHFModel.stochastic(
            lambda _hist, _k: [
                (10, {10: 0.7}),
                (20, {20: 0.4}),
                (30, {30: 0.55}),
            ]
        )
        m = HFDraftModel(model=fake)
        m.reset(request_id="r1", prompt_input_ids=[1, 2, 3])
        ids, probs = m.propose_with_probs(
            request_id="r1", parent_token_id=5, k=3,
            temperature=0.7,
        )
        assert ids == [10, 20, 30]
        assert probs == pytest.approx([0.7, 0.4, 0.55], rel=1e-6)

    def test_length_match_invariant(self):
        # K=4 but script returns only 2 (early-exit emulation).
        # propose_with_probs should still return matching len.
        fake = _FakeHFModel.stochastic(
            lambda _hist, _k: [(7, {7: 0.9}), (8, {8: 0.5})]
        )
        m = HFDraftModel(model=fake)
        m.reset(request_id="r1", prompt_input_ids=[1, 2, 3])
        ids, probs = m.propose_with_probs(
            request_id="r1", parent_token_id=5, k=4,
            temperature=0.5,
        )
        assert len(ids) == len(probs)

    def test_eos_trims_ids_and_probs_symmetrically(self):
        # EOS=999 mid-script. Tokens after EOS dropped; probs
        # truncated to match.
        fake = _FakeHFModel.stochastic(
            lambda _hist, _k: [
                (10, {10: 0.6}),
                (999, {999: 0.3}),  # EOS
                (20, {20: 0.5}),    # never observed
                (30, {30: 0.4}),    # never observed
            ]
        )
        m = HFDraftModel(model=fake, eos_token_id=999)
        m.reset(request_id="r1", prompt_input_ids=[1, 2, 3])
        ids, probs = m.propose_with_probs(
            request_id="r1", parent_token_id=5, k=4,
            temperature=0.7,
        )
        # EOS included; tokens after dropped.
        assert ids == [10, 999]
        assert probs == pytest.approx([0.6, 0.3], rel=1e-6)

    def test_propose_wrapper_discards_probs(self):
        # propose() returns just ids; propose_with_probs() returns
        # (ids, probs). Both should produce the same id sequence
        # under identical inputs (greedy path).
        m1 = HFDraftModel(model=_FakeHFModel(lambda _, k: [42, 43]))
        m1.reset(request_id="r1", prompt_input_ids=[1, 2, 3])
        ids1 = m1.propose(
            request_id="r1", parent_token_id=5, k=2,
            temperature=0.0,
        )

        m2 = HFDraftModel(model=_FakeHFModel(lambda _, k: [42, 43]))
        m2.reset(request_id="r2", prompt_input_ids=[1, 2, 3])
        ids2, _probs2 = m2.propose_with_probs(
            request_id="r2", parent_token_id=5, k=2,
            temperature=0.0,
        )
        assert ids1 == ids2 == [42, 43]

    def test_negative_temperature_rejected(self):
        # v2 lifts the greedy-only gate; positive temperatures
        # are now valid. Negative temps remain invalid.
        m = HFDraftModel(model=_FakeHFModel(lambda _, k: [1]))
        m.reset(request_id="r1", prompt_input_ids=[1, 2, 3])
        with pytest.raises(DraftModelError, match="non-negative"):
            m.propose_with_probs(
                request_id="r1", parent_token_id=5, k=1,
                temperature=-0.1,
            )

    def test_positive_temperature_does_not_raise(self):
        # Inverse of the negative-temp test. Confirms v1 greedy-
        # only gate has been lifted by Phase 3.x.11.y.x.
        fake = _FakeHFModel.stochastic(
            lambda _hist, _k: [(7, {7: 0.5})]
        )
        m = HFDraftModel(model=fake)
        m.reset(request_id="r1", prompt_input_ids=[1, 2, 3])
        # Should NOT raise.
        ids, probs = m.propose_with_probs(
            request_id="r1", parent_token_id=5, k=1,
            temperature=0.7,
        )
        assert ids == [7]
        assert probs == pytest.approx([0.5], rel=1e-6)

    def test_unknown_request_id_raises(self):
        m = HFDraftModel(model=_FakeHFModel(lambda _, k: []))
        with pytest.raises(DraftStateNotFoundError):
            m.propose_with_probs(
                request_id="never-reset",
                parent_token_id=5, k=1, temperature=0.0,
            )
