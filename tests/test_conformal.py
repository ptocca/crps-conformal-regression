"""Tests for crpsconfreg.conformal."""

import numpy as np
import pytest

from crpsconfreg.conformal import conformal_pvalue_grid, conformal_interval


# ── conformal_pvalue_grid ─────────────────────────────────────────────────────

class TestConformalPvalueGrid:
    def test_pvalue_at_least_one_over_m_plus_one(self):
        # p(y_h) = count/(m+1) ≥ 1/(m+1) always (y_h itself is always counted)
        rng = np.random.default_rng(0)
        y_bin = rng.standard_normal(15)
        y_grid = rng.standard_normal(30)
        pvals = conformal_pvalue_grid(y_bin, y_grid)
        assert np.all(pvals >= 1 / (len(y_bin) + 1) - 1e-12)

    def test_pvalue_at_most_one(self):
        rng = np.random.default_rng(1)
        y_bin = rng.standard_normal(10)
        y_grid = np.linspace(-5, 5, 100)
        pvals = conformal_pvalue_grid(y_bin, y_grid)
        assert np.all(pvals <= 1.0 + 1e-12)

    def test_median_has_high_pvalue(self):
        # The median of y_bin is the least "strange" value → high p-value
        y_bin = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        median = np.array([np.median(y_bin)])
        p_median = conformal_pvalue_grid(y_bin, median)[0]
        p_extreme = conformal_pvalue_grid(y_bin, np.array([100.0]))[0]
        assert p_median > p_extreme

    def test_extreme_value_has_minimum_pvalue(self):
        # A very extreme y_h always has the largest α → p = 1/(m+1)
        y_bin = np.array([0.0, 1.0, 2.0, 3.0])
        p_extreme = conformal_pvalue_grid(y_bin, np.array([1e9]))[0]
        assert p_extreme == pytest.approx(1 / (len(y_bin) + 1))

    def test_grid_matches_scalar_evaluation(self):
        # Vectorised result matches calling on each point separately
        rng = np.random.default_rng(2)
        y_bin = rng.standard_normal(8)
        y_grid = np.array([-2.0, 0.0, 1.5, 3.0])
        pvals_vec = conformal_pvalue_grid(y_bin, y_grid)
        pvals_scalar = np.array([
            conformal_pvalue_grid(y_bin, np.array([yh]))[0] for yh in y_grid
        ])
        np.testing.assert_allclose(pvals_vec, pvals_scalar)

    def test_sum_of_scores_equals_augmented_bin_cost(self):
        # Σ_j α_j(y_h) = cost({y_bin, y_h}) = (m+1)/m^2 * W(augmented)
        from crpsconfreg.binning import pairwise_abs_sum
        y_bin = np.array([0.0, 1.0, 3.0])
        y_h = 2.0
        m = len(y_bin)
        augmented = np.append(y_bin, y_h)

        # Manually compute each α_j
        total = 0.0
        for j in range(m + 1):
            support = np.delete(augmented, j)
            obs = augmented[j]
            crps_j = np.mean(np.abs(support - obs)) - pairwise_abs_sum(support) / m ** 2
            total += crps_j

        W_aug = pairwise_abs_sum(augmented)
        expected = (m + 1) / m ** 2 * W_aug
        assert total == pytest.approx(expected, rel=1e-6)

    def test_finite_sample_coverage(self):
        # Statistical coverage test: P(p(y*) > ε) ≥ 1-ε
        # Under exact exchangeability within the bin this holds exactly.
        rng = np.random.default_rng(42)
        n_trials = 2000
        epsilon = 0.10

        # Draw iid observations; use first m as bin, last one as test
        m = 19  # bin size → p-values are multiples of 1/20
        covered = 0
        for _ in range(n_trials):
            obs = rng.standard_normal(m + 1)
            y_bin, y_star = obs[:m], obs[m]
            p = conformal_pvalue_grid(y_bin, np.array([y_star]))[0]
            covered += p > epsilon

        coverage = covered / n_trials
        # Allow 3 standard deviations of slack
        target = 1 - epsilon
        std = np.sqrt(target * epsilon / n_trials)
        assert coverage >= target - 3 * std


# ── conformal_interval ────────────────────────────────────────────────────────

class TestConformalInterval:
    def test_interval_contains_bin_median(self):
        y_bin = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        lo, hi = conformal_interval(y_bin, epsilon=0.10)
        median = np.median(y_bin)
        assert lo <= median <= hi

    def test_interval_grows_with_bin_spread(self):
        y_narrow = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        y_wide   = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        lo_n, hi_n = conformal_interval(y_narrow, epsilon=0.10)
        lo_w, hi_w = conformal_interval(y_wide,   epsilon=0.10)
        assert (hi_w - lo_w) > (hi_n - lo_n)

    def test_interval_grows_as_epsilon_shrinks(self):
        y_bin = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        lo1, hi1 = conformal_interval(y_bin, epsilon=0.20)
        lo2, hi2 = conformal_interval(y_bin, epsilon=0.05)
        assert (hi2 - lo2) >= (hi1 - lo1)

    def test_endpoints_are_in_grid_range(self):
        y_bin = np.array([1.0, 2.0, 3.0])
        std = np.std(y_bin)
        lo, hi = conformal_interval(y_bin, epsilon=0.10)
        assert lo >= y_bin.min() - 4 * std - 1e-6
        assert hi <= y_bin.max() + 4 * std + 1e-6

    def test_interval_is_connected(self):
        # For unimodal distributions the prediction set should be a single interval:
        # all p-values between lo and hi should exceed epsilon
        y_bin = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        epsilon = 0.10
        lo, hi = conformal_interval(y_bin, epsilon=epsilon)
        y_inside = np.linspace(lo + 0.01, hi - 0.01, 50)
        pvals = conformal_pvalue_grid(y_bin, y_inside)
        assert np.all(pvals > epsilon)
