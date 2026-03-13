"""Tests for crpsconfreg.selection."""

import numpy as np
import pytest

from crpsconfreg.selection import crps_empirical, select_K_cv


# ── crps_empirical ────────────────────────────────────────────────────────────

class TestCrpsEmpirical:
    def test_perfect_point_mass(self):
        # ECDF concentrated on y_obs: mean abs dev = 0, spread = 0 → CRPS = 0
        y_support = np.array([2.0, 2.0, 2.0])
        assert crps_empirical(y_support, 2.0) == pytest.approx(0.0)

    def test_nonnegative(self):
        rng = np.random.default_rng(0)
        y_support = rng.standard_normal(20)
        for y_obs in rng.standard_normal(10):
            assert crps_empirical(y_support, y_obs) >= -1e-10

    def test_empty_support_is_nan(self):
        assert np.isnan(crps_empirical(np.array([]), 1.0))

    def test_known_value_two_points(self):
        # ECDF on {0, 2}, obs at 1:
        # mean_abs_dev = (|0-1| + |2-1|)/2 = 1
        # spread = |0-2| / 4 = 0.5
        # CRPS = 1 - 0.5 = 0.5
        assert crps_empirical(np.array([0.0, 2.0]), 1.0) == pytest.approx(0.5)

    def test_increases_with_distance(self):
        y_support = np.array([0.0, 1.0, 2.0])
        c1 = crps_empirical(y_support, 1.0)
        c2 = crps_empirical(y_support, 5.0)
        assert c2 > c1

    def test_energy_score_formula(self):
        # CRPS(F, y) = E|X-y| - 0.5 E|X-X'|
        rng = np.random.default_rng(1)
        y_support = rng.standard_normal(50)
        y_obs = 0.5
        expected = np.mean(np.abs(y_support - y_obs)) - 0.5 * np.mean(
            np.abs(y_support[:, None] - y_support[None, :]))
        assert crps_empirical(y_support, y_obs) == pytest.approx(expected, rel=1e-6)


# ── select_K_cv ───────────────────────────────────────────────────────────────

class TestSelectKCv:
    def _make_data(self, seed=42, n=100):
        rng = np.random.default_rng(seed)
        x = np.sort(rng.uniform(0, 3, n))
        y = rng.normal(loc=x, scale=1 + x)
        return x, y

    def test_k_opt_in_range(self):
        x, y = self._make_data(n=60)
        K_opt, _ = select_K_cv(x, y, K_max=10)
        assert 1 <= K_opt <= 10

    def test_test_crps_length(self):
        x, y = self._make_data(n=60)
        _, tc = select_K_cv(x, y, K_max=8)
        assert len(tc) == 8

    def test_infeasible_k_is_inf(self):
        x, y = self._make_data(n=20)
        _, tc = select_K_cv(x, y, K_max=15)
        # Training set has ~10 obs; K > 5 should be inf
        assert tc[-1] == np.inf

    def test_k1_always_finite(self):
        x, y = self._make_data(n=40)
        _, tc = select_K_cv(x, y, K_max=5)
        assert np.isfinite(tc[0])

    def test_heteroscedastic_selects_multiple_bins(self):
        # Strong heteroscedasticity: K=1 should not be optimal
        x, y = self._make_data(seed=0, n=200)
        K_opt, _ = select_K_cv(x, y, K_max=10)
        assert K_opt > 1
