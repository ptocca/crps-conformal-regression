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


def _evaluate_fold(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    K_max: int,
) -> np.ndarray:
    """Evaluate test CRPS for K = 1, …, K_max on a single train/test fold.

    Returns
    -------
    fold_crps : (K_max,) array; fold_crps[K-1] = average test CRPS for K bins.
                Entries for infeasible K are inf.
    """
    C_tr = precompute_costs(y_tr)
    fold_crps = np.full(K_max, np.inf)

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
        fold_crps[K - 1] = total / len(y_te)

    return fold_crps


def select_K_cv(
    x: np.ndarray,
    y: np.ndarray,
    K_max: int,
    n_folds: int = 5,
) -> tuple[int, np.ndarray]:
    """Select K by interleaved K-fold cross-validation.

    Observations are assigned to folds by their position in the x-sorted
    order: observation i goes to fold (i mod n_folds).  This interleaved
    assignment ensures every fold has points spread evenly across the
    covariate range.

    For each fold, the optimal K-partition is fitted on the remaining
    observations and test CRPS is evaluated on the fold.  The test CRPS
    curves are averaged across folds and K* = argmin of the average.

    Parameters
    ----------
    x, y    : (n,) arrays, sorted by x.
    K_max   : maximum K to evaluate.
    n_folds : number of CV folds (default 5).

    Returns
    -------
    K_opt     : int, selected number of bins.
    test_crps : (K_max,) array; test_crps[K-1] = mean test CRPS for K bins,
                averaged across folds.  Entries for infeasible K are inf.
    """
    n = len(x)
    fold_ids = np.arange(n) % n_folds

    all_fold_crps = np.full((n_folds, K_max), np.inf)

    for f in range(n_folds):
        tr_mask = fold_ids != f
        te_mask = fold_ids == f
        x_tr, y_tr = x[tr_mask], y[tr_mask]
        x_te, y_te = x[te_mask], y[te_mask]
        if len(x_te) == 0 or len(x_tr) < 4:
            continue
        all_fold_crps[f] = _evaluate_fold(x_tr, y_tr, x_te, y_te, K_max)

    test_crps = np.mean(all_fold_crps, axis=0)
    K_opt = int(np.argmin(test_crps)) + 1
    return K_opt, test_crps
