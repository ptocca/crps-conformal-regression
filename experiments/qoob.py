"""QOOB (Quantile Out-Of-Bag conformal), Gupta, Kuchibhotla & Ramdas (2022,
Pattern Recognition; arXiv:1910.10562), "Nested conformal prediction and
quantile out-of-bag ensemble methods".

Faithful reimplementation of Algorithm 2 (QOOB) composed with Algorithm 1
(efficient cross-conformal aggregation over nested sets) from that paper.
This is the *full* (possibly non-convex, union-of-intervals) cross-conformal
variant that the paper reports as its headline result -- not the weaker
always-single-interval QOOB-JP relaxation (their Eq. 15).

No train/calibration split: every one of the n points is used both to grow
the bagged trees and (via out-of-bag aggregation) to calibrate. Public code
for QOOB is MATLAB-only (github.com/AIgen/QOOB); this is an independent
Python reimplementation from the paper's algorithm description, built on
sklearn regression trees (no dependency on skgarden, which the msesia/chr
codebase also had to route around).
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor


def fit_qoob_forest(
    x: np.ndarray, y: np.ndarray,
    T: int = 200, min_samples_leaf: int = 5, random_state: int = 42,
) -> dict:
    """Grow T bootstrap-bagged regression trees, tracking OOB membership.

    Each tree is a plain CART regressor (Meinshausen 2006: the same
    variance-reduction splits as a standard regression tree; only the
    *prediction* step differs -- quantiles of leaf-member y-values instead
    of the leaf mean).

    `x` may be 1-D (n,) -- treated as a single covariate -- or 2-D (n, d)
    for the general multivariate case (needed to replicate the paper's own
    real-dataset experiments, which all use several covariates).
    """
    n = len(y)
    rng = np.random.default_rng(random_state)
    X = np.asarray(x)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    in_bag = np.zeros((n, T), dtype=bool)
    trees = []
    leaf_to_ys: list[dict] = []       # per tree: leaf_id -> y-values of bag members in that leaf
    leaf_id_all = np.empty((n, T), dtype=np.int64)   # leaf id of every point i, in every tree j

    for j in range(T):
        boot_idx = rng.integers(0, n, size=n)
        in_bag[np.unique(boot_idx), j] = True
        tree = DecisionTreeRegressor(
            min_samples_leaf=min_samples_leaf,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        tree.fit(X[boot_idx], y[boot_idx])
        trees.append(tree)

        leaves_all = tree.apply(X)              # leaf id for every one of the n points
        leaf_id_all[:, j] = leaves_all
        leaves_bag = leaves_all[boot_idx]
        d: dict = {}
        for leaf_id in np.unique(leaves_bag):
            d[leaf_id] = y[boot_idx][leaves_bag == leaf_id]
        leaf_to_ys.append(d)

    oob_mask = ~in_bag                            # oob_mask[i, j]: tree j excludes point i
    n_oob = oob_mask.sum(axis=1)
    if np.any(n_oob == 0):
        raise RuntimeError(
            "some training point is in-bag for every tree; increase T"
        )

    return {
        "X": X, "y": y, "n": n, "T": T,
        "trees": trees, "leaf_to_ys": leaf_to_ys,
        "leaf_id_all": leaf_id_all, "oob_mask": oob_mask,
    }


def _query_pool_from_leaf_ids(
    model: dict, leaf_ids_row: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate leaf-member y-values across all T trees for one query point,
    given that point's per-tree leaf ids (leaf_ids_row[j] = leaf id in tree j).

    Returns (values, tree_id) parallel arrays: values[k] came from tree tree_id[k].
    """
    T = model["T"]
    leaf_to_ys = model["leaf_to_ys"]

    vals_list, tid_list = [], []
    for j in range(T):
        ys = leaf_to_ys[j].get(leaf_ids_row[j])
        if ys is not None and len(ys):
            vals_list.append(ys)
            tid_list.append(np.full(len(ys), j))
    if not vals_list:
        return np.array([]), np.array([], dtype=int)
    return np.concatenate(vals_list), np.concatenate(tid_list)


def leaf_ids_for_new_points(model: dict, X_query: np.ndarray) -> np.ndarray:
    """Per-tree leaf ids for arbitrary (not necessarily training) query points.

    Returns an (n_query, T) array, matching the layout of model["leaf_id_all"]
    but for points outside the training set.
    """
    X_query = np.asarray(X_query)
    if X_query.ndim == 1:
        X_query = X_query.reshape(-1, 1)
    T = model["T"]
    trees = model["trees"]
    leaf_ids = np.empty((X_query.shape[0], T), dtype=np.int64)
    for j in range(T):
        leaf_ids[:, j] = trees[j].apply(X_query)
    return leaf_ids


def _oob_quantiles_from_leaf_ids(
    model: dict, leaf_ids_row: np.ndarray, levels: list[float],
) -> np.ndarray:
    """Return an (n, len(levels)) array: row e = quantiles of the OOB-for-e
    pooled distribution, evaluated at the query point with these per-tree leaf
    ids, for each level. `n` here is the number of training points (each is a
    candidate "exclude" index e), not the number of query points.
    """
    n = model["n"]
    oob_mask = model["oob_mask"]
    vals, tid = _query_pool_from_leaf_ids(model, leaf_ids_row)
    out = np.empty((n, len(levels)))
    for e in range(n):
        mask = oob_mask[e, tid]
        pooled = vals[mask]
        if len(pooled) == 0:
            raise RuntimeError(f"empty OOB pool for exclude={e}")
        out[e, :] = np.quantile(pooled, levels)
    return out


def _oob_quantiles_for_query(
    model: dict, q_idx: int, levels: list[float],
) -> np.ndarray:
    """Same as `_oob_quantiles_from_leaf_ids`, but for query point q_idx from
    the training set (uses the precomputed `leaf_id_all` table).
    """
    return _oob_quantiles_from_leaf_ids(model, model["leaf_id_all"][q_idx], levels)


def _cross_conformal_union(
    los: np.ndarray, his: np.ndarray, threshold: float,
) -> list[tuple[float, float]]:
    """Algorithm 1 (Gupta, Kuchibhotla & Ramdas 2022): efficient cross-conformal
    aggregation of n nested-set intervals [los[i], his[i]] into the (possibly
    disjoint) prediction set {y : #{i : r_i(X_i,Y_i) < r_i(x,y)} < threshold+1},
    i.e. the union of y-values covered by more than `threshold` of the n
    intervals.
    """
    if threshold < 0:
        return [(-np.inf, np.inf)]

    n = len(los)
    events = [(los[i], True) for i in range(n)] + [(his[i], False) for i in range(n)]
    events.sort(key=lambda e: (e[0], 0 if e[1] else 1))  # left endpoints before right, at ties

    segments = []
    count = 0
    left_endpoint = None
    for val, is_left in events:
        if is_left:
            count += 1
            if count > threshold and count - 1 <= threshold:
                left_endpoint = val
        else:
            if count > threshold and count - 1 <= threshold:
                segments.append((left_endpoint, val))
            count -= 1
    return segments


def qoob_full_benchmark(
    x: np.ndarray, y: np.ndarray, epsilon: float,
    T: int = 200, min_samples_leaf: int = 5, random_state: int = 42,
) -> dict:
    """Full-data QOOB evaluation at every one of the n data points.

    Mirrors the "Our method (full n)" convention used elsewhere in this
    project's fair-comparison harness: QOOB is transductive (no train/cal
    split), so it is fit once on all n points; each of those n points is then
    also used as an evaluation location, exactly as "Our method (full n)"
    reuses its full-data partition to evaluate coverage/width at cal-half
    points drawn from the same n. Per-point coverage/width can then be
    aggregated over any subset of indices (e.g. a calibration half) without
    recomputation.

    Returns dict with:
      "covered": bool array (n,) -- Y_q inside the QOOB prediction set at X_q
      "width":   float array (n,) -- Lebesgue measure of that set (sum of
                 segment lengths if the set is a union of several intervals)
      "n_segments": int array (n,) -- number of disjoint intervals (>1 => non-convex)
    """
    n = len(x)
    beta = 2 * epsilon
    threshold = epsilon * (n + 1) - 1

    model = fit_qoob_forest(x, y, T=T, min_samples_leaf=min_samples_leaf, random_state=random_state)

    # r_i(X_i, Y_i) = max(q_beta^{-i}(X_i) - Y_i, Y_i - q_{1-beta}^{-i}(X_i))
    # -- the (e=i, q=i) diagonal of the general (exclude, query) quantile grid.
    r = np.empty(n)
    q_beta_diag = np.empty(n)
    q_1mbeta_diag = np.empty(n)
    for i in range(n):
        qs = _oob_quantiles_for_query(model, i, [beta, 1 - beta])
        q_beta_diag[i] = qs[i, 0]
        q_1mbeta_diag[i] = qs[i, 1]
        r[i] = max(q_beta_diag[i] - y[i], y[i] - q_1mbeta_diag[i])

    covered = np.empty(n, dtype=bool)
    width = np.empty(n)
    n_segments = np.empty(n, dtype=int)

    for q in range(n):
        qs = _oob_quantiles_for_query(model, q, [beta, 1 - beta])
        los = qs[:, 0] - r
        his = qs[:, 1] + r
        segments = _cross_conformal_union(los, his, threshold)
        width[q] = sum(hi - lo for lo, hi in segments)
        covered[q] = any(lo <= y[q] <= hi for lo, hi in segments)
        n_segments[q] = len(segments)

    return {"covered": covered, "width": width, "n_segments": n_segments}


def qoob_fit_and_calibrate(
    X_train: np.ndarray, y_train: np.ndarray, epsilon: float,
    T: int = 100, min_samples_leaf: int = 5, random_state: int = 42,
) -> dict:
    """Train/calibrate variant of QOOB (used to replicate the paper's own
    real-dataset protocol, Section 6: a separate held-out test set, distinct
    from the training set used to grow the forest and compute residuals).

    Fits the forest on X_train/y_train and precomputes the n_train
    calibration residuals r_i(X_i, Y_i) (the diagonal of the exclude/query
    quantile grid). Returns a dict ready for `qoob_predict`.
    """
    n = len(y_train)
    beta = 2 * epsilon
    model = fit_qoob_forest(
        X_train, y_train, T=T, min_samples_leaf=min_samples_leaf, random_state=random_state,
    )

    r = np.empty(n)
    for i in range(n):
        qs = _oob_quantiles_for_query(model, i, [beta, 1 - beta])
        r[i] = max(qs[i, 0] - y_train[i], y_train[i] - qs[i, 1])

    model["r"] = r
    model["epsilon"] = epsilon
    model["beta"] = beta
    return model


def qoob_predict(model: dict, X_test: np.ndarray) -> list[list[tuple[float, float]]]:
    """Prediction sets (list of disjoint (lo, hi) intervals) for each row of
    X_test, using the calibration residuals from `qoob_fit_and_calibrate`.
    """
    n = model["n"]
    epsilon = model["epsilon"]
    beta = model["beta"]
    r = model["r"]
    threshold = epsilon * (n + 1) - 1

    leaf_ids = leaf_ids_for_new_points(model, X_test)
    results = []
    for k in range(leaf_ids.shape[0]):
        qs = _oob_quantiles_from_leaf_ids(model, leaf_ids[k], [beta, 1 - beta])
        los = qs[:, 0] - r
        his = qs[:, 1] + r
        results.append(_cross_conformal_union(los, his, threshold))
    return results
