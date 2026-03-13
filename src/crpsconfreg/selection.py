"""Cross-validated selection of the number of bins K."""

import numpy as np

from .binning import pairwise_abs_sum, precompute_costs, optimal_partition, bin_x_boundaries


def crps_empirical(y_support: np.ndarray, y_obs: float) -> float:
    """CRPS of the empirical CDF of y_support evaluated at y_obs.

    CRPS(ECDF_m, y) = (1/m) Σ_i |y_i - y| - W / m²
    where W = Σ_{i<j} |y_i - y_j|.
    """
    m = len(y_support)
    if m == 0:
        return np.nan
    mad = float(np.mean(np.abs(y_support - y_obs)))
    spread = pairwise_abs_sum(y_support) / m ** 2
    return mad - spread


def select_K_cv(
    x: np.ndarray,
    y: np.ndarray,
    K_max: int,
) -> tuple[int, np.ndarray]:
    """Select K via an alternating train/test split of the x-sorted data.

    The training set consists of even-indexed observations (x[0], x[2], …)
    and the test set of odd-indexed observations (x[1], x[3], …); both
    preserve x-sorted order.

    For each K the optimal partition is found on the training half and
    evaluated by average CRPS on the test half.  K* = argmin test CRPS.
    Within-sample LOO-CRPS is gameable (the DP finds bins with near-zero
    LOO cost by placing nearby y-values together); the train/test split
    breaks this strategy because held-out observations are independent.

    Parameters
    ----------
    x, y  : (n,) arrays, sorted by x.
    K_max : maximum K to evaluate.

    Returns
    -------
    K_opt     : int, selected number of bins.
    test_crps : (K_max,) array; test_crps[K-1] = average test CRPS for K bins.
                Entries for infeasible K are inf.
    """
    n = len(x)
    tr = np.arange(0, n, 2)
    te = np.arange(1, n, 2)
    x_tr, y_tr = x[tr], y[tr]
    x_te, y_te = x[te], y[te]

    C_tr = precompute_costs(y_tr)
    test_crps = np.full(K_max, np.inf)

    for K in range(1, K_max + 1):
        if K > len(y_tr) // 2:
            break
        bp, _ = optimal_partition(y_tr, K, C=C_tr)
        edges = bin_x_boundaries(x_tr, bp)
        K_bins = len(bp) - 1

        total = 0.0
        for xq, yq in zip(x_te, y_te):
            idx = int(np.clip(np.searchsorted(edges, xq, "right") - 1, 0, K_bins - 1))
            total += crps_empirical(y_tr[bp[idx] : bp[idx + 1]], yq)
        test_crps[K - 1] = total / len(y_te)

    K_opt = int(np.argmin(test_crps)) + 1
    return K_opt, test_crps
