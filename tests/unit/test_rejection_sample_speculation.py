"""Phase 3.x.11.y.x Task 3 — unit tests for the
``rejection_sample_speculation`` helper.

Covers the Leviathan-2023 §2.2 rejection-sampling speculation
algorithm:
  - Greedy specialization (q is point mass on argmax → equivalent
    to v1 prefix-match)
  - All-accept (high p[d_i] / q(d_i) ratio → always accept) +
    bonus sampling
  - All-reject (q[d_i] >> p[d_i] → always reject + correction)
  - Partial accept (mixed accept/reject across K positions)
  - Length invariant (len(verified) == accepted_count + 1 always)
  - Numerical edge: q(d_i) == 0 → accept_prob clamped to 1.0
  - Distribution correctness: over many trials, the marginal
    output distribution converges to the target distribution p
    (the load-bearing Leviathan-2023 invariant)
  - Validation: K mismatch + target_distributions length mismatch
"""

from __future__ import annotations

import numpy as np
import pytest

from prsm.compute.inference.sharded_runner import (
    rejection_sample_speculation,
)


# ──────────────────────────────────────────────────────────────────────────
# Deterministic RNG fixtures — control u draws + choice samples
# ──────────────────────────────────────────────────────────────────────────


class _ScriptedRng:
    """Deterministic RNG with separate scripts for ``random()``
    (u draws — accept-check) and ``choice()`` (correction +
    bonus sampling). Tests inject specific scripts to exercise
    accept/reject branches deterministically."""

    def __init__(
        self, *, random_script=None, choice_script=None,
    ):
        self._random_script = list(random_script or [])
        self._choice_script = list(choice_script or [])
        self.random_calls = 0
        self.choice_calls = []

    def random(self) -> float:
        if self.random_calls >= len(self._random_script):
            raise AssertionError(
                f"_ScriptedRng: random() called {self.random_calls + 1} "
                f"times but script only has "
                f"{len(self._random_script)} entries"
            )
        out = self._random_script[self.random_calls]
        self.random_calls += 1
        return out

    def choice(self, n, p=None) -> int:
        self.choice_calls.append((n, p))
        if not self._choice_script:
            raise AssertionError(
                "_ScriptedRng: choice() called but no script entries "
                "remaining (test should pre-script every choice)"
            )
        return self._choice_script.pop(0)


def _make_dist(probs_dict, vocab_size=4):
    """Build a vocab-size NumPy distribution from a dict
    {token_id: prob}. Other tokens get the leftover mass divided
    evenly. Useful for keeping tests readable."""
    dist = np.zeros(vocab_size, dtype=np.float64)
    for tok, p in probs_dict.items():
        dist[tok] = p
    leftover = 1.0 - dist.sum()
    if leftover > 0:
        # Distribute over un-set tokens.
        unset_idxs = [i for i in range(vocab_size) if i not in probs_dict]
        if unset_idxs:
            dist[unset_idxs] = leftover / len(unset_idxs)
    return dist


# ──────────────────────────────────────────────────────────────────────────
# Algorithm correctness
# ──────────────────────────────────────────────────────────────────────────


class TestRejectionSampleSpeculation:
    def test_all_accept_emits_k_plus_one(self):
        # K=2; both drafts accepted (p[d_i] >> q[d_i]); bonus
        # sampled from p_K via choice script.
        # Scripted u=0.0 always < accept_prob = min(1, p/q).
        K = 2
        target = [
            _make_dist({1: 0.9}),   # p_0: most mass on token 1
            _make_dist({2: 0.9}),   # p_1: most mass on token 2
            _make_dist({3: 0.9}),   # p_2: bonus from token 3
        ]
        rng = _ScriptedRng(
            random_script=[0.0, 0.0],  # accept both drafts
            choice_script=[3],         # bonus = 3
        )
        ids, ac = rejection_sample_speculation(
            target_distributions=target,
            proposed_token_ids=[1, 2],
            proposed_token_probs=[0.5, 0.5],
            rng=rng,
        )
        assert ids == [1, 2, 3]
        assert ac == K
        # Only one choice call (bonus); accept-checks didn't trigger
        # residual sampling.
        assert len(rng.choice_calls) == 1

    def test_all_reject_at_position_zero_emits_correction(self):
        # K=2; first draft rejected immediately; correction sampled
        # from residual.
        target = [
            _make_dist({0: 0.6}, vocab_size=4),  # p_0
            _make_dist({2: 0.9}, vocab_size=4),  # p_1 unused
            _make_dist({3: 0.9}, vocab_size=4),  # p_2 unused
        ]
        rng = _ScriptedRng(
            random_script=[0.99],  # u >= accept_prob → reject
            choice_script=[2],     # correction = 2
        )
        # d_0 = 1; q(d_0) = 0.9; p_0(d_0) = (1 - 0.6) / 3 ≈ 0.133
        # accept_prob = min(1, 0.133/0.9) ≈ 0.148. u=0.99 > 0.148.
        ids, ac = rejection_sample_speculation(
            target_distributions=target,
            proposed_token_ids=[1, 2],
            proposed_token_probs=[0.9, 0.5],
            rng=rng,
        )
        assert ids == [2]   # only the correction emitted
        assert ac == 0      # zero drafts accepted
        # No bonus call; one residual choice.
        assert len(rng.choice_calls) == 1

    def test_partial_accept_one_of_two(self):
        # K=2; first draft accepted, second rejected → correction.
        target = [
            _make_dist({1: 0.9}, vocab_size=4),  # p_0
            _make_dist({0: 0.7}, vocab_size=4),  # p_1
            _make_dist({3: 0.9}, vocab_size=4),  # p_2 unused
        ]
        rng = _ScriptedRng(
            random_script=[0.0, 0.99],  # accept #0, reject #1
            choice_script=[3],          # correction
        )
        ids, ac = rejection_sample_speculation(
            target_distributions=target,
            proposed_token_ids=[1, 2],   # d_0=1, d_1=2
            proposed_token_probs=[0.5, 0.9],
            rng=rng,
        )
        assert ids == [1, 3]   # accepted d_0=1 + correction=3
        assert ac == 1

    def test_q_zero_always_accepts(self):
        # Numerical edge: q(d_i) == 0 → accept_prob clamped to 1.0.
        # u=0.99 still accepts.
        target = [
            _make_dist({1: 0.5}, vocab_size=4),
            _make_dist({2: 0.9}, vocab_size=4),
        ]
        rng = _ScriptedRng(
            random_script=[0.99],
            choice_script=[2],  # bonus
        )
        ids, ac = rejection_sample_speculation(
            target_distributions=target,
            proposed_token_ids=[1],
            proposed_token_probs=[0.0],   # q(d_0) = 0
            rng=rng,
        )
        # Accept (q=0 → clamp to 1.0 even though u=0.99); bonus.
        assert ids == [1, 2]
        assert ac == 1

    def test_length_invariant_holds_on_all_paths(self):
        # Three runs covering the three exit paths; each produces
        # accepted_count + 1 tokens.
        target_K1 = [
            _make_dist({1: 0.9}, vocab_size=3),
            _make_dist({2: 0.9}, vocab_size=3),
        ]
        # Reject path
        rng = _ScriptedRng(random_script=[0.99], choice_script=[2])
        ids, ac = rejection_sample_speculation(
            target_distributions=target_K1,
            proposed_token_ids=[1],
            proposed_token_probs=[0.99],
            rng=rng,
        )
        assert len(ids) == ac + 1

        # Accept path
        rng = _ScriptedRng(random_script=[0.0], choice_script=[2])
        ids, ac = rejection_sample_speculation(
            target_distributions=target_K1,
            proposed_token_ids=[1],
            proposed_token_probs=[0.5],
            rng=rng,
        )
        assert len(ids) == ac + 1

    def test_residual_sample_excludes_unmatched_d_i(self):
        # When d_i has q=1.0 (point mass) and p[d_i] < 1.0, the
        # residual r[d_i] = max(0, p[d_i] - q(d_i)) = max(0,
        # p[d_i] - 1.0) = 0. The residual distribution has zero
        # mass on d_i. Sample MUST come from t != d_i.
        target = [
            np.array([0.4, 0.3, 0.3], dtype=np.float64),  # p_0
            np.array([0.5, 0.3, 0.2], dtype=np.float64),  # p_1 unused
        ]
        rng = _ScriptedRng(
            random_script=[0.99],  # reject
            choice_script=[2],     # correction sampled outside d_i
        )
        ids, ac = rejection_sample_speculation(
            target_distributions=target,
            proposed_token_ids=[0],   # d_0 = 0
            proposed_token_probs=[1.0],  # q(d_0) = 1 (point mass)
            rng=rng,
        )
        # Residual on d_0=0 should be max(0, 0.4 - 1.0) = 0.
        # The choice was scripted to return 2; we verify the
        # ``p`` arg passed to choice has zero on d_0=0.
        assert ac == 0
        assert ids == [2]
        n, p_arg = rng.choice_calls[0]
        assert n == 3
        assert p_arg[0] == 0.0

    def test_validation_rejects_id_prob_length_mismatch(self):
        target = [_make_dist({1: 0.9}), _make_dist({2: 0.9})]
        rng = _ScriptedRng()
        with pytest.raises(RuntimeError, match="proposed_token_probs"):
            rejection_sample_speculation(
                target_distributions=target,
                proposed_token_ids=[1, 2, 3],
                proposed_token_probs=[0.5, 0.5],  # K=2 vs ids K=3
                rng=rng,
            )

    def test_validation_rejects_target_distribution_count_mismatch(self):
        # K drafts → expect K+1 target distributions.
        rng = _ScriptedRng()
        with pytest.raises(RuntimeError, match="target_distributions"):
            rejection_sample_speculation(
                target_distributions=[_make_dist({1: 0.9})],  # only 1
                proposed_token_ids=[1, 2],   # K=2 → expect 3
                proposed_token_probs=[0.5, 0.5],
                rng=rng,
            )

    def test_distribution_convergence_under_many_trials(self):
        """Load-bearing Leviathan-2023 invariant: the marginal
        distribution of emitted tokens equals the verifier's
        target distribution p.

        Setup: K=1 with target ``p_0 = [0.5, 0.3, 0.15, 0.05]``
        and draft as a point mass on token 0 with ``q(d_0) = 1.0``.
        Under this configuration:
          - Accept prob = min(1, p[d_0]/q(d_0)) = 0.5 → emit 0
            with prob 0.5.
          - Reject path: residual r[0] = max(0, 0.5 - 1.0) = 0;
            r = [0, 0.3, 0.15, 0.05] → renormalize to
            [0, 0.6, 0.3, 0.1]. Sample correction.
          - Marginal P(emit t) for t > 0 = 0.5 * (p[t] / 0.5) =
            p[t] ✓.
          - Marginal P(emit 0) = 0.5 ✓.

        5000 trials; compare empirical histogram to p_0 within
        ~3-sigma tolerance.
        """
        target = np.array(
            [0.5, 0.3, 0.15, 0.05], dtype=np.float64,
        )
        rng = np.random.default_rng(seed=42)
        N = 5000
        emitted_first_token = np.zeros(4, dtype=np.int64)
        for _ in range(N):
            ids, ac = rejection_sample_speculation(
                target_distributions=[target.copy(), target.copy()],
                proposed_token_ids=[0],
                proposed_token_probs=[1.0],   # q = point mass on d_0
                rng=rng,
            )
            emitted_first_token[ids[0]] += 1

        empirical = emitted_first_token / N
        np.testing.assert_allclose(empirical, target, atol=0.025), (
            f"empirical {empirical} vs target {target}"
        )

    def test_distribution_convergence_at_K4_first_emit(self):
        """Round-1 review M1 remediation (1/2). Convergence test at
        K=4 validating the §2.2 marginal-equals-target claim under
        Option C.1's degenerate-q regime (q a TRUE point mass with
        mass 1.0). The K=1 test above proved the per-position
        invariant; this test exercises the multi-position partial-
        accept paths.

        Setup: K=4 with target p = [0.5, 0.3, 0.15, 0.05] at every
        position 0..K and d_i = 0 with q(d_i) = 1.0 (degenerate
        point mass — Option C.1 is exact here). Per-position
        accept_prob = min(1, p(0)/q(0)) = 0.5. Residual on reject:
        r(0) = max(0, 0.5 - 1.0) = 0; r(t!=0) = p(t); normalize
        → [0, 0.6, 0.3, 0.1].

        Theorem (Leviathan-2023 §2.2 + Option C.1 degenerate-q):
          P(first emit = 0)   = P(accept_at_0) = 0.5 = p(0) ✓
          P(first emit = t!=0) = P(reject_at_0) * r0_norm(t)
                               = 0.5 * (p(t)/0.5) = p(t) ✓

        The marginal of ids[0] equals p REGARDLESS of K. K=4
        exercises the K=4 → K=3 → ... → K=0 partial-accept depth +
        all-accept bonus from p_K. Total emitted varies 1..K+1=5;
        we restrict the convergence assertion to ids[0] which is
        what the load-bearing claim covers.

        5000 trials; ~3σ tolerance.
        """
        target = np.array(
            [0.5, 0.3, 0.15, 0.05], dtype=np.float64,
        )
        rng = np.random.default_rng(seed=2026042901)
        N = 5000
        K = 4
        target_distributions = [target.copy() for _ in range(K + 1)]
        proposed_ids = [0] * K
        proposed_probs = [1.0] * K  # degenerate: Option C.1 exact
        emitted_first_token = np.zeros(4, dtype=np.int64)
        for _ in range(N):
            ids, ac = rejection_sample_speculation(
                target_distributions=target_distributions,
                proposed_token_ids=proposed_ids,
                proposed_token_probs=proposed_probs,
                rng=rng,
            )
            assert 0 <= ac <= K
            assert len(ids) == ac + 1
            emitted_first_token[ids[0]] += 1

        empirical = emitted_first_token / N
        np.testing.assert_allclose(empirical, target, atol=0.025), (
            f"K=4 first-emit empirical {empirical} vs target "
            f"{target}; §2.2 marginal claim violated under Option "
            f"C.1 degenerate-q regime"
        )

    def test_option_c1_drift_under_stochastic_q_documented(self):
        """Round-1 review M1 remediation (2/2) — honest-scope
        companion. Option C.1 treats q as a point mass on d_i with
        mass q(d_i); under stochastic drafting (q(d_i) < 1.0), the
        residual computation differs from the §2.2 exact construction
        because the true q has nonzero mass on non-d_i tokens that
        C.1 ignores.

        This test pins the C.1 marginal numerically — both
        documenting the drift for the audit trail AND catching any
        future change to the helper that would break C.1's
        determinism.

        Setup: K=1, target p = [0.5, 0.3, 0.15, 0.05], q(d_0) = 0.6.
        Under Option C.1:
          accept_prob = min(1, 0.5/0.6) = 0.8333...
          residual r = [max(0, 0.5 - 0.6), 0.3, 0.15, 0.05]
                     = [0, 0.3, 0.15, 0.05]; sum = 0.5;
                     normalized = [0, 0.6, 0.3, 0.1]
          P(emit 0)   = 0.8333
          P(emit t>0) = (1 - 0.8333) * r_norm(t)
                      = 0.1667 * [0, 0.6, 0.3, 0.1]
                      = [0, 0.1, 0.05, 0.0167]
          C.1 marginal = [0.8333, 0.1, 0.05, 0.0167]

        Under §2.2 EXACT (with the true stochastic q on the full
        vocab), the marginal would equal target p. C.1 differs.
        Phase 3.x.11.y.x' bumps to Option C.3 (full top-M draft
        distribution wire) if production telemetry shows the
        drift matters in real traffic. Until then, threat-model
        addendum §3.6 documents this as honest scope.
        """
        target = np.array(
            [0.5, 0.3, 0.15, 0.05], dtype=np.float64,
        )
        rng = np.random.default_rng(seed=2026042902)
        N = 5000
        accept_prob = min(1.0, target[0] / 0.6)
        r = target.copy()
        r[0] = max(0.0, target[0] - 0.6)
        r_norm = r / r.sum()
        c1_marginal = np.zeros_like(target)
        c1_marginal[0] = accept_prob
        c1_marginal[1:] = (1.0 - accept_prob) * r_norm[1:]
        assert abs(c1_marginal.sum() - 1.0) < 1e-9
        max_abs_drift = np.max(np.abs(c1_marginal - target))
        assert max_abs_drift > 0.05, (
            f"Test setup invalid: C.1 marginal {c1_marginal} should "
            f"differ from target {target} by > 0.05 under non-"
            f"degenerate q"
        )

        emitted = np.zeros(4, dtype=np.int64)
        for _ in range(N):
            ids, ac = rejection_sample_speculation(
                target_distributions=[target.copy(), target.copy()],
                proposed_token_ids=[0],
                proposed_token_probs=[0.6],
                rng=rng,
            )
            emitted[ids[0]] += 1

        empirical = emitted / N
        np.testing.assert_allclose(empirical, c1_marginal, atol=0.025), (
            f"empirical {empirical} vs C.1-analytical {c1_marginal} "
            f"(target was {target}, drift = "
            f"{np.abs(c1_marginal - target)}) — helper drifted "
            f"from documented Option C.1 algorithm"
        )
