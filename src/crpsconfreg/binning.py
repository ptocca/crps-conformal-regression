"""Optimal K-partition of x-sorted observations under LOO-CRPS."""

import numpy as np


def pairwise_abs_sum(y: np.ndarray) -> float:
    """Sum of all pairwise absolute differences Σ_{i<j} |y_i - y_j|.

    Uses the sorted-array identity in O(m log m) time:
        Σ_{i<j} |y_i - y_j| = Σ_k (2k - (m-1)) · y_k   (y sorted ascending)
    """
    ys = np.sort(y)
    m = len(ys)
    if m < 2:
        return 0.0
    return float(np.dot(2 * np.arange(m) - (m - 1), ys))


def _bin_cost(W: float, m: int) -> float:
    """LOO-CRPS cost of a bin with pairwise sum W and size m."""
    if m < 2:
        return np.inf
    return m * W / (m - 1) ** 2


def precompute_costs(y: np.ndarray) -> np.ndarray:
    """Precompute C[i, j] = LOO-CRPS cost of bin y[i : j+1] for all i ≤ j.

    The LOO-CRPS cost of a bin S with m ≥ 2 observations is:

        cost(S) = m / (m-1)^2 · Σ_{l<r ∈ S} |y_l - y_r|

    Singletons (m=1) have undefined LOO cost and are stored as inf.

    Parameters
    ----------
    y : (n,) array, sorted by x (not necessarily by value).

    Returns
    -------
    C : (n, n) array with C[i, j] = cost(y[i:j+1]); inf for i > j or m=1.
    """
    n = len(y)
    C = np.full((n, n), np.inf)

    for i in range(n):
        W = 0.0
        sorted_vals: list[float] = []
        prefix: list[float] = [0.0]   # prefix[k] = sum(sorted_vals[:k])

        for j in range(i, n):
            val = float(y[j])

            # Binary-search insertion rank into sorted_vals
            lo, hi = 0, len(sorted_vals)
            while lo < hi:
                mid = (lo + hi) // 2
                if sorted_vals[mid] < val:
                    lo = mid + 1
                else:
                    hi = mid
            rank = lo

            S_le = prefix[rank]
            m_cur = len(sorted_vals)
            S_gt = prefix[m_cur] - S_le
            # Incremental update of the pairwise sum
            W += val * rank - S_le + S_gt - val * (m_cur - rank)

            sorted_vals.insert(rank, val)
            prefix.insert(rank + 1, S_le + val)
            for k in range(rank + 2, len(prefix)):
                prefix[k] += val

            m = j - i + 1
            C[i, j] = _bin_cost(W, m)

    return C


def optimal_partition(
    y: np.ndarray,
    K: int,
    C: np.ndarray | None = None,
) -> tuple[list[int], float]:
    """Find the K-partition of y minimising total LOO-CRPS via dynamic programming.

    Each bin must contain at least 2 observations (LOO requires m ≥ 2).

    Parameters
    ----------
    y : (n,) array, sorted by x.
    K : number of bins; must satisfy 1 ≤ K ≤ n // 2.
    C : optional precomputed cost matrix from `precompute_costs`. Computed
        internally if not supplied.

    Returns
    -------
    breakpoints : list of length K+1.
        breakpoints[0] = 0, breakpoints[-1] = n.
        Bin k contains y[breakpoints[k] : breakpoints[k+1]].
    total_cost : float, optimal total LOO-CRPS.
    """
    n = len(y)
    if not (1 <= K <= n // 2):
        raise ValueError(f"Need 1 ≤ K ≤ n//2; got K={K}, n={n}")
    if C is None:
        C = precompute_costs(y)

    # dp[k, j] = min cost for y[0:j] partitioned into k bins (each size ≥ 2)
    dp = np.full((K + 1, n + 1), np.inf)
    split = np.full((K + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0

    for k in range(1, K + 1):
        for j in range(2 * k, n + 1):
            for i in range(2 * (k - 1), j - 1):
                val = dp[k - 1, i] + C[i, j - 1]
                if val < dp[k, j]:
                    dp[k, j] = val
                    split[k, j] = i

    # Backtrack
    bps = [n]
    j = n
    for k in range(K, 0, -1):
        i = split[k, j]
        bps.append(i)
        j = i
    bps.reverse()

    return bps, float(dp[K, n])


def bin_x_boundaries(x_train: np.ndarray, breakpoints: list[int]) -> np.ndarray:
    """Partition the real line at midpoints between adjacent bin boundaries.

    Parameters
    ----------
    x_train    : (n,) x-values, sorted ascending.
    breakpoints: output of `optimal_partition`.

    Returns
    -------
    edges : (K+1,) array.
        edges[0] = -inf, edges[-1] = +inf.
        edges[k] = midpoint of x_train[breakpoints[k]-1] and x_train[breakpoints[k]]
        for 1 ≤ k ≤ K-1.
    """
    K = len(breakpoints) - 1
    edges = np.empty(K + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    for k in range(1, K):
        edges[k] = 0.5 * (x_train[breakpoints[k] - 1] + x_train[breakpoints[k]])
    return edges
