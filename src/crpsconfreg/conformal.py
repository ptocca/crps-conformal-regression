"""Full conformal prediction with CRPS as the nonconformity score."""

import numpy as np

from .binning import pairwise_abs_sum


def conformal_pvalue_grid(
    y_bin: np.ndarray,
    y_h_grid: np.ndarray,
) -> np.ndarray:
    """Full conformal p-values for a grid of test candidates.

    For a test point x* in a bin with training responses y_bin = (y_1,…,y_m)
    and candidate label y_h, the nonconformity score is the LOO-CRPS of y_h
    in the augmented set {y_1,…,y_m, y_h}:

        α(y_h) = CRPS(F̂_m, y_h) = (1/m) Σ_i |y_i - y_h| - W/m²

    The training score for y_j in the same augmented set is:

        α_j(y_h) = (d_j + |y_h - y_j|)/m  -  (W - d_j)/m²  -  A_j(y_h)/m²

    where d_j = Σ_{i≠j}|y_i - y_j|, A_j(y_h) = Σ_{i≠j}|y_i - y_h|, and
    W = Σ_{i<j}|y_i - y_j|.

    The conformal p-value is:

        p(y_h) = #{j ∈ {1,…,m+1} : α_j(y_h) ≥ α(y_h)} / (m+1)

    where j=m+1 refers to y_h itself (always counted, since α(y_h) ≥ α(y_h)).

    Under exchangeability of (y_1,…,y_m, y*), P(y* ∈ {y_h: p(y_h) > ε}) ≥ 1−ε.

    Fully vectorised over y_h_grid. Time complexity O(m² + m·N).

    Parameters
    ----------
    y_bin    : (m,) training responses in the bin.
    y_h_grid : (N,) grid of test candidates.

    Returns
    -------
    pvals : (N,) conformal p-values in (0, 1].
    """
    m = len(y_bin)
    yh = np.asarray(y_h_grid, dtype=float)

    # |y_i - y_h| for all (i, h): shape (m, N)
    abs_diff = np.abs(y_bin[:, None] - yh[None, :])
    sum_abs = abs_diff.sum(axis=0)          # (N,)  Σ_i |y_i - y_h|

    W_m = pairwise_abs_sum(y_bin)

    # Test score α(y_h) = CRPS(F̂_m, y_h): shape (N,)
    alpha_h = sum_abs / m - W_m / m ** 2

    # d_j = Σ_{i≠j} |y_i - y_j|: shape (m,)  (diagonal of pairwise matrix is 0)
    d = np.abs(y_bin[:, None] - y_bin[None, :]).sum(axis=1)

    # A_j(y_h) = Σ_{i≠j} |y_i - y_h|: shape (m, N)
    A = sum_abs[None, :] - abs_diff

    # Training score α_j(y_h): shape (m, N)
    alpha_j = (
        (d[:, None] + abs_diff) / m
        - (W_m - d[:, None]) / m ** 2
        - A / m ** 2
    )

    # p-value: count training scores ≥ α_h, plus 1 for y_h itself
    count = (alpha_j >= alpha_h[None, :]).sum(axis=0) + 1
    return count / (m + 1)


def conformal_interval(
    y_bin: np.ndarray,
    epsilon: float,
    n_grid: int = 2000,
) -> tuple[float, float]:
    """Conformal prediction interval {y_h : p(y_h) > ε} at level epsilon.

    Evaluates p(y_h) on a grid spanning the bin range ± 4 standard deviations.
    Returns the lower and upper endpoints of the (typically single) interval.

    Parameters
    ----------
    y_bin   : (m,) training responses in the bin.
    epsilon : miscoverage level; the interval has coverage ≥ 1 - epsilon.
    n_grid  : number of grid points for evaluating p(y_h).

    Returns
    -------
    (lower, upper) : float pair. Returns (nan, nan) if the prediction set is empty
                     on the grid (only possible for very large epsilon).
    """
    std = float(np.std(y_bin)) if len(y_bin) > 1 else 1.0
    y_grid = np.linspace(y_bin.min() - 4 * std, y_bin.max() + 4 * std, n_grid)
    mask = conformal_pvalue_grid(y_bin, y_grid) > epsilon
    if not mask.any():
        return np.nan, np.nan
    return float(y_grid[mask][0]), float(y_grid[mask][-1])
