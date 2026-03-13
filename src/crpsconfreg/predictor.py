"""High-level predictor combining optimal binning and conformal inference."""

import numpy as np

from .binning import precompute_costs, optimal_partition, bin_x_boundaries
from .selection import crps_empirical, select_K_cv
from .conformal import conformal_pvalue_grid, conformal_interval


class BinningPredictor:
    """Optimal-binning predictor with full conformal prediction sets.

    Usage
    -----
    >>> pred = BinningPredictor()
    >>> pred.fit(x_train, y_train)
    >>> lo, hi = pred.predict_interval(x_new, epsilon=0.10)
    >>> pval = pred.conformal_pvalue(x_new, y_new)
    """

    def __init__(self) -> None:
        self._fitted = False

    # ── fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        K_max: int = 20,
        K: int | None = None,
    ) -> "BinningPredictor":
        """Fit the predictor on training data sorted by x.

        Parameters
        ----------
        x, y  : (n,) arrays. Must be sorted by x on entry.
        K_max : maximum number of bins to consider in CV selection.
                Ignored if K is provided.
        K     : fix the number of bins, bypassing CV selection.

        Returns
        -------
        self
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("x and y must be 1-D arrays of equal length")
        if not np.all(np.diff(x) >= 0):
            raise ValueError("x must be sorted in ascending order")

        self.x_train_ = x
        self.y_train_ = y
        self._C = precompute_costs(y)

        if K is not None:
            self.K_ = K
            self.cv_test_crps_ = None
        else:
            self.K_, self.cv_test_crps_ = select_K_cv(x, y, K_max)

        self.breakpoints_, self.total_cost_ = optimal_partition(y, self.K_, C=self._C)
        self.edges_ = bin_x_boundaries(x, self.breakpoints_)
        self._fitted = True
        return self

    # ── helpers ───────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before predicting")

    def _bin_index(self, xq: float) -> int:
        return int(np.clip(
            np.searchsorted(self.edges_, xq, "right") - 1,
            0, self.K_ - 1,
        ))

    def _bin_responses(self, xq: float) -> np.ndarray:
        idx = self._bin_index(xq)
        lo, hi = self.breakpoints_[idx], self.breakpoints_[idx + 1]
        return self.y_train_[lo:hi]

    # ── prediction ────────────────────────────────────────────────────────────

    def predict_ecdf(self, x_new: float, t_grid: np.ndarray) -> np.ndarray:
        """Within-bin ECDF evaluated at t_grid for a single test point.

        Parameters
        ----------
        x_new  : scalar test covariate.
        t_grid : (T,) array of threshold values.

        Returns
        -------
        cdf : (T,) array in [0, 1].
        """
        self._check_fitted()
        y_bin = self._bin_responses(float(x_new))
        return np.mean(y_bin[:, None] <= np.asarray(t_grid)[None, :], axis=0)

    def predict_interval(
        self,
        x_new: np.ndarray | float,
        epsilon: float = 0.10,
        n_grid: int = 2000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Conformal prediction interval at level epsilon for each test point.

        Parameters
        ----------
        x_new   : scalar or (n_test,) array of test covariates.
        epsilon : miscoverage level; coverage ≥ 1 - epsilon.
        n_grid  : grid size for p-value evaluation.

        Returns
        -------
        lower, upper : arrays of shape (n_test,).
        """
        self._check_fitted()
        x_new = np.atleast_1d(np.asarray(x_new, dtype=float))
        lower = np.empty(len(x_new))
        upper = np.empty(len(x_new))
        for q, xq in enumerate(x_new):
            y_bin = self._bin_responses(xq)
            lower[q], upper[q] = conformal_interval(y_bin, epsilon, n_grid)
        return lower, upper

    def conformal_pvalue(
        self,
        x_new: np.ndarray | float,
        y_new: np.ndarray | float,
    ) -> np.ndarray:
        """Conformal p-value p(y*) for observed test responses.

        Parameters
        ----------
        x_new, y_new : scalars or (n_test,) arrays.

        Returns
        -------
        pvals : (n_test,) array of p-values in (0, 1].
        """
        self._check_fitted()
        x_new = np.atleast_1d(np.asarray(x_new, dtype=float))
        y_new = np.atleast_1d(np.asarray(y_new, dtype=float))
        pvals = np.empty(len(x_new))
        for q, (xq, yq) in enumerate(zip(x_new, y_new)):
            y_bin = self._bin_responses(xq)
            pvals[q] = conformal_pvalue_grid(y_bin, np.array([yq]))[0]
        return pvals

    def bin_sizes(self) -> list[int]:
        """Sizes of the fitted bins."""
        self._check_fitted()
        return [
            self.breakpoints_[k + 1] - self.breakpoints_[k]
            for k in range(self.K_)
        ]
