"""Tests for crpsconfreg.predictor.BinningPredictor."""

import numpy as np
import pytest

from crpsconfreg import BinningPredictor


@pytest.fixture
def hetero_data():
    rng = np.random.default_rng(42)
    n = 100
    x = np.sort(rng.uniform(0, 3, n))
    y = rng.normal(loc=x, scale=1 + x)
    return x, y


class TestBinningPredictorFit:
    def test_fit_returns_self(self, hetero_data):
        x, y = hetero_data
        pred = BinningPredictor()
        assert pred.fit(x, y, K_max=5) is pred

    def test_k_opt_set_after_fit(self, hetero_data):
        x, y = hetero_data
        pred = BinningPredictor().fit(x, y, K_max=5)
        assert hasattr(pred, "K_")
        assert pred.K_ >= 1

    def test_fixed_k_bypasses_cv(self, hetero_data):
        x, y = hetero_data
        pred = BinningPredictor().fit(x, y, K=3)
        assert pred.K_ == 3
        assert pred.cv_test_crps_ is None

    def test_unsorted_x_raises(self):
        x = np.array([1.0, 0.0, 2.0])
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="sorted"):
            BinningPredictor().fit(x, y)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            BinningPredictor().fit(np.arange(5, dtype=float), np.arange(4, dtype=float))

    def test_predict_before_fit_raises(self):
        pred = BinningPredictor()
        with pytest.raises(RuntimeError):
            pred.predict_interval(np.array([1.0]))

    def test_bin_sizes_sum_to_n(self, hetero_data):
        x, y = hetero_data
        pred = BinningPredictor().fit(x, y, K_max=5)
        assert sum(pred.bin_sizes()) == len(x)

    def test_each_bin_at_least_size_2(self, hetero_data):
        x, y = hetero_data
        pred = BinningPredictor().fit(x, y, K_max=5)
        assert all(s >= 2 for s in pred.bin_sizes())


class TestBinningPredictorPredict:
    @pytest.fixture(autouse=True)
    def fitted(self, hetero_data):
        x, y = hetero_data
        self.pred = BinningPredictor().fit(x, y, K=3)
        self.x = x
        self.y = y

    def test_predict_ecdf_shape(self):
        t_grid = np.linspace(-5, 10, 50)
        cdf = self.pred.predict_ecdf(self.x[50], t_grid)
        assert cdf.shape == (50,)

    def test_predict_ecdf_monotone(self):
        t_grid = np.linspace(-5, 10, 200)
        cdf = self.pred.predict_ecdf(self.x[50], t_grid)
        assert np.all(np.diff(cdf) >= -1e-12)

    def test_predict_ecdf_bounded(self):
        t_grid = np.linspace(-5, 10, 100)
        cdf = self.pred.predict_ecdf(self.x[50], t_grid)
        assert np.all(cdf >= 0) and np.all(cdf <= 1)

    def test_predict_interval_shape_scalar(self):
        lo, hi = self.pred.predict_interval(1.5)
        assert lo.shape == (1,) and hi.shape == (1,)

    def test_predict_interval_shape_array(self):
        x_new = np.array([0.5, 1.0, 2.0])
        lo, hi = self.pred.predict_interval(x_new)
        assert lo.shape == (3,) and hi.shape == (3,)

    def test_predict_interval_lo_less_than_hi(self):
        x_new = np.linspace(self.x[5], self.x[-5], 20)
        lo, hi = self.pred.predict_interval(x_new, epsilon=0.10)
        assert np.all(lo < hi)

    def test_conformal_pvalue_shape(self):
        x_new = np.array([0.5, 1.0, 2.0])
        y_new = np.array([0.5, 1.0, 2.0])
        pvals = self.pred.conformal_pvalue(x_new, y_new)
        assert pvals.shape == (3,)

    def test_conformal_pvalue_in_range(self):
        x_new = self.x[:10]
        y_new = self.y[:10]
        pvals = self.pred.conformal_pvalue(x_new, y_new)
        assert np.all(pvals > 0) and np.all(pvals <= 1)

    def test_coverage_on_fresh_data(self):
        # Empirical coverage should be ≥ 1-ε (with slack for finite sample)
        rng = np.random.default_rng(99)
        n_te = 500
        x_te = np.sort(rng.uniform(0, 3, n_te))
        y_te = rng.normal(loc=x_te, scale=1 + x_te)
        pvals = self.pred.conformal_pvalue(x_te, y_te)
        epsilon = 0.10
        coverage = np.mean(pvals > epsilon)
        target = 1 - epsilon
        std = np.sqrt(target * epsilon / n_te)
        assert coverage >= target - 3 * std
