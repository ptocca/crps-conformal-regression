"""Tests for crpsconfreg.binning."""

import numpy as np
import pytest

from crpsconfreg.binning import (
    pairwise_abs_sum,
    precompute_costs,
    optimal_partition,
    bin_x_boundaries,
)


# ── pairwise_abs_sum ──────────────────────────────────────────────────────────

class TestPairwiseAbsSum:
    def test_two_points(self):
        assert pairwise_abs_sum(np.array([0.0, 1.0])) == pytest.approx(1.0)

    def test_three_points(self):
        # |0-1| + |0-3| + |1-3| = 1 + 3 + 2 = 6
        assert pairwise_abs_sum(np.array([0.0, 1.0, 3.0])) == pytest.approx(6.0)

    def test_singleton_is_zero(self):
        assert pairwise_abs_sum(np.array([5.0])) == 0.0

    def test_order_invariant(self):
        y = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        assert pairwise_abs_sum(y) == pytest.approx(pairwise_abs_sum(np.sort(y)))

    def test_constant_array(self):
        assert pairwise_abs_sum(np.ones(10)) == pytest.approx(0.0)

    def test_matches_naive(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(20)
        naive = sum(abs(y[i] - y[j]) for i in range(len(y)) for j in range(i + 1, len(y)))
        assert pairwise_abs_sum(y) == pytest.approx(naive)


# ── precompute_costs ──────────────────────────────────────────────────────────

class TestPrecomputeCosts:
    def test_singleton_is_inf(self):
        C = precompute_costs(np.array([1.0, 2.0, 3.0]))
        assert C[0, 0] == np.inf
        assert C[1, 1] == np.inf

    def test_two_point_bin(self):
        # cost(S) = m/(m-1)^2 * W = 2/1 * |y0-y1|
        y = np.array([0.0, 3.0])
        C = precompute_costs(y)
        assert C[0, 1] == pytest.approx(2.0 * 3.0)

    def test_upper_triangular_structure(self):
        y = np.arange(5, dtype=float)
        C = precompute_costs(y)
        # Below diagonal should be inf (unused)
        for i in range(len(y)):
            for j in range(i):
                assert C[i, j] == np.inf

    def test_cost_increases_with_spread(self):
        # Wider bins are more expensive
        y = np.array([0.0, 1.0, 2.0, 10.0])
        C = precompute_costs(y)
        assert C[0, 1] < C[0, 3]

    def test_formula_for_three_points(self):
        # cost = 3/(2^2) * W, W = sum of pairs
        y = np.array([0.0, 1.0, 3.0])
        W = pairwise_abs_sum(y)  # 1+3+2 = 6
        expected = 3 / 4 * W
        C = precompute_costs(y)
        assert C[0, 2] == pytest.approx(expected)


# ── optimal_partition ─────────────────────────────────────────────────────────

class TestOptimalPartition:
    def test_k1_is_single_bin(self):
        y = np.array([0.0, 1.0, 2.0, 3.0])
        bps, cost = optimal_partition(y, 1)
        assert bps == [0, 4]
        C = precompute_costs(y)
        assert cost == pytest.approx(C[0, 3])

    def test_k_equals_n_over_2_all_pairs(self):
        y = np.array([0.0, 1.0, 2.0, 3.0])
        bps, cost = optimal_partition(y, 2)
        assert len(bps) == 3
        assert bps[0] == 0 and bps[-1] == 4
        assert all(bps[k + 1] - bps[k] >= 2 for k in range(2))

    def test_optimal_cost_beats_equal_split(self):
        # The optimal partition must have cost ≤ any specific partition of the same K.
        # Use equal-sized bins as a baseline.
        rng = np.random.default_rng(42)
        n = 20
        y = rng.standard_normal(n)
        C = precompute_costs(y)
        K = 4
        _, opt_cost = optimal_partition(y, K, C=C)
        # Equal-split baseline: bins of size n//K = 5
        size = n // K
        baseline_cost = sum(C[k * size, (k + 1) * size - 1] for k in range(K))
        assert opt_cost <= baseline_cost + 1e-10

    def test_invalid_k_raises(self):
        y = np.arange(6, dtype=float)
        with pytest.raises(ValueError):
            optimal_partition(y, 0)
        with pytest.raises(ValueError):
            optimal_partition(y, 4)  # 4 > 6//2 = 3

    def test_breakpoints_cover_full_range(self):
        rng = np.random.default_rng(7)
        y = rng.standard_normal(30)
        for K in range(1, 6):
            bps, _ = optimal_partition(y, K)
            assert bps[0] == 0
            assert bps[-1] == len(y)
            assert len(bps) == K + 1

    def test_each_bin_has_at_least_two_obs(self):
        rng = np.random.default_rng(3)
        y = rng.standard_normal(24)
        for K in range(1, 7):
            bps, _ = optimal_partition(y, K)
            for k in range(K):
                assert bps[k + 1] - bps[k] >= 2

    def test_partition_minimises_cost_over_alternatives(self):
        # For n=4, K=2: only one valid split (2|2); verify it's optimal
        y = np.array([0.0, 0.1, 5.0, 5.1])  # two clear clusters
        bps, cost = optimal_partition(y, 2)
        assert bps == [0, 2, 4]  # split between the clusters


# ── bin_x_boundaries ──────────────────────────────────────────────────────────

class TestBinXBoundaries:
    def test_endpoints_are_infinite(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        bps = [0, 2, 4]
        edges = bin_x_boundaries(x, bps)
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf

    def test_interior_midpoint(self):
        x = np.array([0.0, 1.0, 2.0, 4.0])
        bps = [0, 2, 4]
        edges = bin_x_boundaries(x, bps)
        # Midpoint of x[1]=1.0 and x[2]=2.0 is 1.5
        assert edges[1] == pytest.approx(1.5)

    def test_single_bin_no_interior(self):
        x = np.array([0.0, 1.0, 2.0])
        bps = [0, 3]
        edges = bin_x_boundaries(x, bps)
        assert len(edges) == 2
        assert edges[0] == -np.inf
        assert edges[1] == np.inf
