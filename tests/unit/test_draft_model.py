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


class _FakeHFModel:
    """Deterministic HF-shaped stub. ``.generate()`` extends
    input_ids with a scripted per-position next-token map.

    The script is a callable ``(input_ids: List[int]) -> List[int]``
    returning the K new tokens to append. Tests inject scripts
    that exercise specific propose/commit/EOS scenarios."""

    def __init__(self, script: Any) -> None:
        self._script = script
        self.generate_calls: List[List[int]] = []

    def generate(
        self,
        *,
        input_ids,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float = 1.0,
        pad_token_id: int = 0,
    ):
        # input_ids is a torch.Tensor of shape [1, seq_len];
        # convert to list-of-int for the script.
        in_list = input_ids[0].tolist()
        self.generate_calls.append(in_list)
        new_tokens = self._script(in_list, max_new_tokens)
        # Pad-or-truncate to max_new_tokens (mirrors HF's
        # contract under do_sample=False).
        new_tokens = new_tokens[:max_new_tokens]
        # Return a torch-tensor-like object with .tolist().
        full = list(in_list) + list(new_tokens)

        class _Output:
            def __init__(self, rows):
                self._rows = rows

            def __getitem__(self, idx):
                return _Row(self._rows[idx])

        class _Row:
            def __init__(self, vals):
                self._vals = vals

            def tolist(self):
                return list(self._vals)

        return _Output([full])


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
        # Real torch available — no shim needed; HFDraftModel
        # calls torch.tensor(...) which works with the fake model
        # because it doesn't do any tensor math.
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

    fake_torch.no_grad = _NoGrad
    fake_torch.tensor = _fake_tensor
    fake_torch.long = "long"
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

    def test_propose_raises_on_temperature_nonzero(self):
        m = HFDraftModel(model=_FakeHFModel(lambda _, k: []))
        m.reset(request_id="r1", prompt_input_ids=[1, 2, 3])
        with pytest.raises(DraftModelError, match="greedy-only"):
            m.propose(
                request_id="r1",
                parent_token_id=5,
                k=4,
                temperature=0.7,
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
