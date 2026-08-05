"""Clean-room reimplementation of Conformalized Histogram Regression (CHR).

Sesia, M. & Romano, Y. (2021), "Conformal Prediction using Conditional
Histograms", NeurIPS 2021 (arXiv:2105.08747).

Reimplemented from the published algorithm and from reading (not vendoring)
the reference implementation at github.com/msesia/chr, which has no LICENSE
file. Only the "intervals" mode is implemented here: the shortest contiguous
run of histogram bins whose estimated mass is >= 1-epsilon. This is a
deliberate match to what the reference code actually runs -- its alternative
density-ranked "sets" mode (which can be disconnected) exists on the
accumulator class but is never wired into the reference's own CHR.calibrate()/
predict() (that branch is an unfinished `assert(1==2)` stub), and the paper's
own real-data experiments only ever exercise the intervals mode.

Grey-box: any regressor exposing quantile predictions works; here we always
use our own quantile_forest-based QRF, the same base learner already used for
CQR-QRF elsewhere in this project, so that CHR vs. CQR-QRF isolates the value
of the histogram/shortest-window construction from the choice of base learner.

Simplifications relative to the reference (disclosed, not hidden):
  - Tail smoothing in the reference's `_estim_dist` is a no-op for its own
    default quantile grid (0.01..0.99, tau=0.01: only one point lies beyond
    the tau cutoff on each side, so `np.linspace(..., num=1)` returns the
    point unchanged) -- omitted here since it does nothing on that grid.
  - The reference re-evaluates the CDF on a second, redundant fixed grid
    inside `_estim_dist` ("Standardize") before `compute_histogram` evaluates
    it again on `self.breaks` -- when the two grids coincide (the reference's
    own default), this is one interpolation, not two; we interpolate once.
  - No boundary-bin randomization (the reference's `randomize=True` default):
    kept deterministic for consistency with the other split-conformal
    competitors in this project (CQR, CQR-QRF, IDR), at the cost of being
    marginally more conservative than the reference's headline numbers.
  - Coarser histogram resolution than the reference's default (n_bins=200
    here vs. their 1000): the reference's default was tuned for real-dataset
    calibration sets of n_cal=2000, and is wasteful at our n ~ 100-150.
    delta_alpha=0.001 (their default) is kept, not coarsened: an earlier
    attempt at delta_alpha=0.01 (10x coarser, for speed) caused a handful of
    hard-to-predict calibration points -- ones whose true y fell outside the
    grey-box's predicted [1%, 99%] quantile range, which does happen on the
    heteroscedastic motorcycle dataset's sharp transition region -- to
    saturate at the grid's coarsest level and blow up the resulting interval
    to near the full data range on rare splits. At delta_alpha=0.001 the same
    splits resolve to sane, non-degenerate coverage/width.
"""
import numpy as np
from quantile_forest import RandomForestQuantileRegressor

DEFAULT_QUANTILE_LEVELS = np.round(np.arange(0.01, 1.0, 0.01), 2)  # matches the paper's own grid


def _conformal_quantile(scores: np.ndarray, epsilon: float) -> float:
    """Same order-statistic convention as experiments/competitors.py."""
    n = len(scores)
    k = min(int(np.ceil((1 - epsilon) * (n + 1))), n)
    return float(np.sort(scores)[k - 1])


def fit_qrf_greybox(
    x_tr: np.ndarray, y_tr: np.ndarray,
    n_estimators: int = 500, min_samples_leaf: int = 5, random_state: int = 42,
) -> RandomForestQuantileRegressor:
    model = RandomForestQuantileRegressor(
        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=random_state,
    )
    model.fit(x_tr.reshape(-1, 1), y_tr)
    return model


def _predicted_quantiles(qrf, x: np.ndarray, quantile_levels: np.ndarray) -> np.ndarray:
    """Returns (len(x), len(quantile_levels)) array, non-decreasing per row."""
    q = qrf.predict(x.reshape(-1, 1), quantiles=list(quantile_levels))
    q = np.asarray(q)
    if q.ndim == 1:
        q = q.reshape(-1, 1)
    return np.maximum.accumulate(q, axis=1)  # guard against quantile crossing


def _build_histogram(
    pred_quantiles: np.ndarray, quantile_levels: np.ndarray,
    y_min: float, y_max: float, n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Conditional histogram from predicted quantiles.

    Returns (masses, breaks): masses is (n, n_bins), breaks is (n_bins+1,)
    equal-width edges spanning [y_min, y_max]. masses[i, j] is the estimated
    probability mass of row i in (breaks[j], breaks[j+1]].
    """
    n = pred_quantiles.shape[0]
    breaks = np.linspace(y_min, y_max, n_bins + 1)
    levels = np.concatenate(([0.0], quantile_levels, [1.0]))
    masses = np.empty((n, n_bins))
    jitter = np.linspace(0.0, 1e-6, num=len(levels))  # break ties, keep strictly increasing
    for i in range(n):
        q = np.concatenate(([y_min], pred_quantiles[i], [y_max]))
        q = np.clip(q, y_min, y_max)
        q = np.maximum.accumulate(q) + jitter
        cdf_at_breaks = np.interp(breaks, q, levels, left=0.0, right=1.0)
        m = np.diff(cdf_at_breaks)
        m = np.clip(m, 0.0, None) + 1e-6
        masses[i] = m / m.sum()
    return masses, breaks


def _shortest_window(
    mass: np.ndarray, target: float, lo: int = 0, hi: int | None = None,
    include: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Shortest contiguous window (inclusive bin indices) of mass[lo:hi] with
    sum >= target. If `include=(a, b)` is given (absolute indices), the
    returned window is constrained to contain [a, b]. Falls back to the full
    [lo, hi) range if no window reaches `target` (can happen from floating-
    point mass loss near 1.0).

    Direct reimplementation of the sliding-window logic in the reference's
    `smallestSubWithSum` (github.com/msesia/chr, chr/grey_boxes.py).
    """
    if hi is None:
        hi = len(mass)
    if include is None:
        end_init, start_max = lo, hi - 1
    else:
        end_init, start_max = include[1], include[0]

    start_best, end_best, min_len = lo, hi - 1, hi - lo + 1
    start = lo
    curr = mass[lo:end_init].sum() if end_init > lo else 0.0
    for end in range(end_init, hi):
        curr += mass[end]
        while curr >= target and start <= end and start <= start_max:
            if end - start + 1 < min_len:
                min_len = end - start + 1
                start_best, end_best = start, end
            curr -= mass[start]
            start += 1
    return start_best, end_best


def _nested_window_family(
    mass_matrix: np.ndarray, target_epsilon: float, alpha_grid: np.ndarray,
) -> dict[float, np.ndarray]:
    """Monotonically nested family of shortest-window sets, one per alpha_grid
    level, each an (n, 2) array of inclusive (start, end) bin indices.

    Nesting (needed for the calibration score below to be well-defined) is
    enforced exactly as in the reference: sets for alpha > target_epsilon are
    computed by restricting the search to within the next-larger-alpha (i.e.
    previously computed, smaller) set; sets for alpha < target_epsilon are
    computed by requiring the search to contain the next-smaller-alpha set.
    """
    n = mass_matrix.shape[0]
    k_star = int(np.searchsorted(alpha_grid, target_epsilon))
    windows: dict[float, np.ndarray] = {}

    base = np.empty((n, 2), dtype=int)
    for i in range(n):
        base[i] = _shortest_window(mass_matrix[i], 1.0 - target_epsilon)
    windows[alpha_grid[k_star]] = base

    prev = base
    for k in range(k_star + 1, len(alpha_grid)):
        a = alpha_grid[k]
        cur = np.empty((n, 2), dtype=int)
        for i in range(n):
            cur[i] = _shortest_window(mass_matrix[i], 1.0 - a, lo=prev[i, 0], hi=prev[i, 1] + 1)
        windows[a] = cur
        prev = cur

    prev = base
    for k in range(k_star - 1, -1, -1):
        a = alpha_grid[k]
        cur = np.empty((n, 2), dtype=int)
        for i in range(n):
            cur[i] = _shortest_window(mass_matrix[i], 1.0 - a, include=tuple(prev[i]))
        windows[a] = cur
        prev = cur

    return windows


def _calibration_scores(
    mass_matrix: np.ndarray, y: np.ndarray, breaks: np.ndarray,
    target_epsilon: float, alpha_grid: np.ndarray,
) -> np.ndarray:
    n_bins = mass_matrix.shape[1]
    bin_idx = np.clip(np.searchsorted(breaks, y, side="right") - 1, 0, n_bins - 1)
    windows = _nested_window_family(mass_matrix, target_epsilon, alpha_grid)
    alpha_max = np.zeros(len(y))
    for a in alpha_grid:  # ascending: later (larger-alpha, smaller-set) hits overwrite
        w = windows[a]
        contains = (bin_idx >= w[:, 0]) & (bin_idx <= w[:, 1])
        alpha_max[contains] = a
    return 1.0 - alpha_max


def chr_fit_and_calibrate(
    x_tr: np.ndarray, y_tr: np.ndarray,
    x_cal: np.ndarray, y_cal: np.ndarray,
    epsilon: float,
    y_min: float, y_max: float,
    n_bins: int = 200, delta_alpha: float = 0.001,
    quantile_levels: np.ndarray = DEFAULT_QUANTILE_LEVELS,
    n_estimators: int = 500, min_samples_leaf: int = 5, random_state: int = 42,
) -> dict:
    qrf = fit_qrf_greybox(
        x_tr, y_tr,
        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=random_state,
    )
    q_cal = _predicted_quantiles(qrf, x_cal, quantile_levels)
    mass_cal, breaks = _build_histogram(q_cal, quantile_levels, y_min, y_max, n_bins)

    alpha_grid = np.round(np.arange(delta_alpha, 1.0, delta_alpha), 6)
    alpha_grid = np.unique(np.sort(np.concatenate([alpha_grid, [epsilon]])))

    scores = _calibration_scores(mass_cal, y_cal, breaks, epsilon, alpha_grid)
    q = _conformal_quantile(scores, epsilon)
    calibrated_alpha = float(np.clip(1.0 - q, 1e-6, 1.0 - 1e-6))

    return {
        "qrf": qrf, "breaks": breaks, "quantile_levels": quantile_levels,
        "y_min": y_min, "y_max": y_max, "n_bins": n_bins,
        "epsilon": epsilon, "calibrated_alpha": calibrated_alpha,
    }


def chr_predict(model: dict, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_test = _predicted_quantiles(model["qrf"], x_test, model["quantile_levels"])
    mass_test, breaks = _build_histogram(
        q_test, model["quantile_levels"], model["y_min"], model["y_max"], model["n_bins"],
    )
    target = 1.0 - model["calibrated_alpha"]
    n = mass_test.shape[0]
    lo = np.empty(n); hi = np.empty(n)
    for i in range(n):
        start, end = _shortest_window(mass_test[i], target)
        lo[i] = breaks[start]
        hi[i] = breaks[end + 1]
    return lo, hi
