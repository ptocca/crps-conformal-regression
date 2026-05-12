"""Bimodal mixture DGP: fan plots comparing CRPS vs 1-NN conformal intervals.

DGP: Y | X=x  ~  0.5·N(x−1.5, 0.3²) + 0.5·N(x+1.5, 0.3²),   X ~ Uniform(0, 6)
The two conditional modes shift linearly with x, staying 3 units apart.

Produces figures/fig_bimodal_fan.pdf: side-by-side fan plots showing
  Left  — CRPS score: one wide interval spanning both modes per bin
  Right — 1-NN score: two narrow intervals tracking the two modes per bin
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from crpsconfreg import (
    conformal_interval,
    precompute_costs,
    optimal_partition,
    bin_x_boundaries,
    select_K_cv,
)

plt.rcParams.update({
    "text.usetex": False,
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
EPSILON = 0.10
N = 600
K_MAX = 20
COLORS = plt.cm.tab10.colors

# ── Generate data ──────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)
x_raw = rng.uniform(0, 6, N)
mode_shift = rng.choice([-1.5, 1.5], size=N)
y_raw = rng.normal(x_raw + mode_shift, 0.3)

order = np.argsort(x_raw)
x = x_raw[order]
y = y_raw[order]
n = len(x)

# ── K selection and optimal partition ──────────────────────────────────────────

K_opt, _ = select_K_cv(x, y, K_MAX)
C = precompute_costs(y)
bp, _ = optimal_partition(y, K_opt, C=C)
edges = bin_x_boundaries(x, bp)
K = len(bp) - 1

print(f"n={n}, K*={K_opt}, bin sizes: {[bp[k+1]-bp[k] for k in range(K)]}")

# ── 1-NN conformal interval computation ───────────────────────────────────────

def knn_conformal_intervals(y_bin: np.ndarray, epsilon: float, k: int = 1,
                             n_grid: int = 5000) -> list[tuple[float, float]]:
    """Transductive k-NN conformal prediction set (exact LOO construction).

    For each training point j, define
      α_j(y_h) = clamp(|y_j − y_h|, d_{k−1}^j, d_k^j)
    where d_r^j = r-th LOO nearest-neighbour distance of y_j in {y_i : i≠j}.
    For k=1 this reduces to min(d_1^j, |y_j − y_h|) — the manuscript formula.

    Test score: α_{m+1}(y_h) = k-th NN distance from y_h to {y_i}.
    p(y_h) = (1 + #{j : α_j(y_h) ≥ α_{m+1}(y_h)}) / (m+1).
    Prediction set = {y_h : p(y_h) > ε}.
    """
    m = len(y_bin)
    k_eff = min(k, m - 1)

    # Clamp bounds per training point: d_{k-1}^j and d_k^j
    D = np.abs(y_bin[:, None] - y_bin[None, :])  # (m, m)
    np.fill_diagonal(D, np.inf)
    D_sorted = np.sort(D, axis=1)                 # ascending per row
    calib_hi = D_sorted[:, k_eff - 1]             # d_k^j        (m,)
    calib_lo = D_sorted[:, k_eff - 2] if k_eff >= 2 else np.zeros(m)  # d_{k-1}^j

    # Grid spanning ±4σ beyond the bin extremes
    margin = max(4.0 * float(y_bin.std()), 2.0)
    grid = np.linspace(float(y_bin.min()) - margin, float(y_bin.max()) + margin, n_grid)

    dist_to_grid = np.abs(y_bin[:, None] - grid[None, :])  # (m, n_grid)

    # α_{m+1}(y_h): k-th NN distance from each grid point to training set
    test_alpha = np.sort(dist_to_grid, axis=0)[k_eff - 1, :]   # (n_grid,)

    # α_j(y_h) = clamp(|y_j − y_h|, d_{k-1}^j, d_k^j)
    alpha_j = np.clip(dist_to_grid, calib_lo[:, None], calib_hi[:, None])  # (m, n_grid)

    # p(y_h) = (1 + #{j : α_j ≥ test_alpha}) / (m+1)
    count = 1 + np.sum(alpha_j >= test_alpha[None, :], axis=0)  # (n_grid,)
    in_set = count / (m + 1) > epsilon

    # Extract contiguous segments
    diff = np.diff(in_set.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends   = np.where(diff == -1)[0]
    if in_set[0]:
        starts = np.concatenate([[0], starts])
    if in_set[-1]:
        ends = np.concatenate([ends, [len(grid) - 1]])

    return [(float(grid[s]), float(grid[e])) for s, e in zip(starts, ends)]


def select_knn_cv(y: np.ndarray, bp: list, epsilon: float,
                   k_max: int = 20, n_folds: int = 5) -> int:
    """Select k by n-fold CV: minimise average prediction-set length.

    Uses the same interleaved fold assignment as select_K_cv so that each
    fold trains on ~(n_folds-1)/n_folds of the bin, closer to the full-data
    regime than a 50/50 split.
    """
    n = len(y)
    K = len(bp) - 1
    fold_ids = np.arange(n) % n_folds
    avg_lengths = np.full(k_max, np.inf)

    for k in range(1, k_max + 1):
        total_len = 0.0
        total_te  = 0
        feasible  = True
        for fold in range(n_folds):
            for b in range(K):
                bin_all = np.arange(bp[b], bp[b + 1])
                tr = bin_all[fold_ids[bin_all] != fold]
                te = bin_all[fold_ids[bin_all] == fold]
                if len(tr) < k + 1 or len(te) == 0:
                    feasible = False
                    break
                segs = knn_conformal_intervals(y[tr], epsilon, k=k)
                total_len += sum(hi - lo for lo, hi in segs) * len(te)
                total_te  += len(te)
            if not feasible:
                break
        if feasible and total_te > 0:
            avg_lengths[k - 1] = total_len / total_te

    k_star = int(np.argmin(avg_lengths)) + 1
    return k_star

# ── CV k-selection ────────────────────────────────────────────────────────────

K_NN_MAX = 15
k_star = select_knn_cv(y, bp, EPSILON, k_max=K_NN_MAX)
print(f"CV-selected k* = {k_star}")

# ── Compute per-bin intervals ──────────────────────────────────────────────────

crps_segs = []   # list of (lo, hi)
knn_segs  = []   # list of list of (lo, hi)

for k in range(K):
    y_bin = y[bp[k]:bp[k + 1]]
    lo, hi = conformal_interval(y_bin, EPSILON, n_grid=5000)
    crps_segs.append((lo, hi))
    knn_segs.append(knn_conformal_intervals(y_bin, EPSILON, k=k_star))

n_knn_segs = [len(segs) for segs in knn_segs]
print(f"k-NN (k={k_star}) segments per bin: {n_knn_segs}")

# ── Figure ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
fig.subplots_adjust(wspace=0.05)

x_margin = 0.15
x_plot_lo = x.min() - x_margin
x_plot_hi = x.max() + x_margin
y_plot_lo  = y.min() - 0.4
y_plot_hi  = y.max() + 0.4


def bin_x_extent(k: int) -> tuple[float, float]:
    """Clip the infinite first/last bin edge to the data range."""
    lo = x_plot_lo if k == 0     else float(edges[k])
    hi = x_plot_hi if k == K - 1 else float(edges[k + 1])
    return lo, hi


panel_data = [
    (axes[0], crps_segs, f"CRPS score ($\\varepsilon={EPSILON}$): one interval per bin"),
    (axes[1], knn_segs,
     f"$k$-NN score ($k^*={k_star}$, $\\varepsilon={EPSILON}$): intervals track each mode"),
]

for ax, segs, title in panel_data:
    # Grey scatter of training data
    ax.scatter(x, y, s=5, color="grey", alpha=0.20, linewidths=0, zorder=1)

    # Coloured conformal bands
    for k in range(K):
        x_lo, x_hi = bin_x_extent(k)
        color = COLORS[k % len(COLORS)]
        # segs[k] is either (lo, hi) or a list of (lo, hi)
        segments = segs[k] if isinstance(segs[k], list) else [segs[k]]
        for seg_lo, seg_hi in segments:
            rect = mpatches.Rectangle(
                (x_lo, seg_lo), x_hi - x_lo, seg_hi - seg_lo,
                linewidth=0.5, edgecolor="white", facecolor=color, alpha=0.45,
                zorder=2,
            )
            ax.add_patch(rect)

    ax.set_xlim(x_plot_lo, x_plot_hi)
    ax.set_ylim(y_plot_lo, y_plot_hi)
    ax.set_xlabel("$x$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, zorder=0)

axes[0].set_ylabel("$y$")

plt.suptitle(
    f"Bimodal mixture DGP ($n={N}$, $K^*={K_opt}$, $k^*={k_star}$): "
    "CRPS spans the gap; $k$-NN tracks each mode separately",
    fontsize=10,
)

out_path = os.path.join(FIGDIR, "fig_bimodal_fan.pdf")
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_path}")
