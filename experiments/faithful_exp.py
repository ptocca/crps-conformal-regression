"""Old Faithful experiment: bimodal conditional distribution."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from experiments.data import load_faithful
from experiments.competitors import (
    fit_gaussian_split_conformal, predict_gaussian_interval,
    fit_cqr, predict_cqr_interval,
    fit_cqr_qrf, predict_cqr_qrf_interval,
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
COLORS = plt.cm.Set2.colors

# Inline copies of core functions (avoid import issues with save_figures.py)
from save_figures import (
    pairwise_abs_sum, precompute_costs, optimal_partition,
    bin_x_boundaries, crps_empirical, select_K_cv,
    conformal_pvalue_grid, conformal_interval,
)

# ── Load data ─────────────────────────────────────────────────────────────────

x, y = load_faithful()
n = len(x)
print(f"Faithful: n={n}, x=[{x.min():.0f},{x.max():.0f}] min, y=[{y.min():.2f},{y.max():.2f}] min")

# ── Fit our method (full data) ────────────────────────────────────────────────

K_max = n // 10
K_opt, test_crps_curve = select_K_cv(x, y, K_max)
C_all = precompute_costs(y)
bp, _ = optimal_partition(y, K_opt, C=C_all)
edges = bin_x_boundaries(x, bp)
print(f"K* = {K_opt}, boundaries at x = {[f'{e:.1f}' for e in edges[1:-1]]}")

# ── Train/cal split for competitors (alternating, matching CV convention) ─────

tr_idx = np.arange(0, n, 2)
cal_idx = np.arange(1, n, 2)
x_tr, y_tr = x[tr_idx], y[tr_idx]
x_cal, y_cal = x[cal_idx], y[cal_idx]

gauss_model    = fit_gaussian_split_conformal(x_tr, y_tr, x_cal, y_cal)
cqr_model      = fit_cqr(x_tr, y_tr, x_cal, y_cal, EPSILON, degree=3)
cqr_qrf_model  = fit_cqr_qrf(x_tr, y_tr, x_cal, y_cal, EPSILON)

# ── Coverage evaluation on calibration half ───────────────────────────────────

def our_interval(xq):
    bi = int(np.clip(np.searchsorted(edges, xq, "right") - 1, 0, K_opt - 1))
    return conformal_interval(y[bp[bi]:bp[bi + 1]], EPSILON, n_grid=3000)

def coverage_width(lo_arr, hi_arr, y_eval):
    covered = np.mean((y_eval >= lo_arr) & (y_eval <= hi_arr))
    width   = np.mean(hi_arr - lo_arr)
    return covered, width

# Our method evaluated on all n points (LOO)
our_lo = np.array([our_interval(xq)[0] for xq in x_cal])
our_hi = np.array([our_interval(xq)[1] for xq in x_cal])
gauss_lo, gauss_hi     = predict_gaussian_interval(gauss_model, x_cal, EPSILON)
cqr_lo,   cqr_hi       = predict_cqr_interval(cqr_model, x_cal)
cqr_qrf_lo, cqr_qrf_hi = predict_cqr_qrf_interval(cqr_qrf_model, x_cal)

results = {
    "Our method":         coverage_width(our_lo,      our_hi,      y_cal),
    "Gaussian conformal": coverage_width(gauss_lo,    gauss_hi,    y_cal),
    "CQR (cubic)":        coverage_width(cqr_lo,      cqr_hi,      y_cal),
    "CQR-QRF":            coverage_width(cqr_qrf_lo,  cqr_qrf_hi,  y_cal),
}

print(f"\nOld Faithful — nominal coverage {int(100*(1-EPSILON))}%")
print(f"{'Method':<22} {'Coverage':>10}  {'Mean width':>10}")
print("-" * 46)
for name, (cov, wid) in results.items():
    print(f"{name:<22} {100*cov:>9.1f}%  {wid:>10.3f}")

# ── LaTeX table ───────────────────────────────────────────────────────────────

def print_latex_table():
    print("\n% --- LaTeX table ---")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\begin{tabular}{lcc}")
    print(r"\hline")
    print(r"Method & Coverage (\%) & Mean width (min) \\")
    print(r"\hline")
    for name, (cov, wid) in results.items():
        bold_open  = r"\textbf{" if name == "Our method" else ""
        bold_close = r"}" if name == "Our method" else ""
        print(f"{bold_open}{name}{bold_close} & {100*cov:.1f} & {wid:.3f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Old Faithful: empirical coverage and mean prediction-interval width "
          r"at nominal level $1-\varepsilon = 0.90$. "
          r"Coverage evaluated on the held-out alternating split ($n_{\rm cal}=136$).}")
    print(r"\label{tab:faithful_coverage}")
    print(r"\end{table}")

print_latex_table()

# ── Figure 1: partition scatter + within-bin ECDFs ────────────────────────────

t_grid = np.linspace(y.min() - 0.1, y.max() + 0.1, 600)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: scatter with bin shading
ax = axes[0]
x_disp = np.clip(edges,
                 x[0]  - 0.02 * (x[-1] - x[0]),
                 x[-1] + 0.02 * (x[-1] - x[0]))
ax.scatter(x, y, s=14, alpha=0.55, color="steelblue", zorder=3)
for b in range(K_opt):
    ax.axvspan(x_disp[b], x_disp[b + 1], alpha=0.18, color=COLORS[b % len(COLORS)])
    if b > 0:
        ax.axvline(edges[b], color="grey", lw=0.9, ls=":")
ax.set_xlabel("Waiting time (min)")
ax.set_ylabel("Eruption duration (min)")
ax.set_title(f"Optimal {K_opt}-bin partition ($K^* = {K_opt}$, CV-selected)")
ax.grid(True, alpha=0.3)

# Right: within-bin ECDFs
ax = axes[1]
for b in range(K_opt):
    yb = y[bp[b]:bp[b + 1]]
    m  = len(yb)
    F  = np.mean(yb[:, None] <= t_grid[None, :], axis=0)
    ax.step(t_grid, F, where="post", color=COLORS[b % len(COLORS)], lw=2,
            label=f"Bin {b+1}  ($m={m}$)")
    ax.plot(np.sort(yb), np.zeros(m) + 0.01 * (b + 1),
            "|", color=COLORS[b % len(COLORS)], ms=6, alpha=0.7)
ax.set_xlabel("Eruption duration (min)")
ax.set_ylabel("$\\hat{F}(t)$")
ax.set_title("Within-bin empirical CDFs")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(f"{FIGDIR}/fig_faithful_partition.pdf", bbox_inches="tight")
plt.close(fig)
print("\nSaved fig_faithful_partition.pdf")

# ── Figure 2: prediction interval comparison ──────────────────────────────────

x_dense = np.linspace(x.min(), x.max(), 300)
our_lo_d  = np.array([our_interval(xq)[0] for xq in x_dense])
our_hi_d  = np.array([our_interval(xq)[1] for xq in x_dense])
g_lo_d, g_hi_d   = predict_gaussian_interval(gauss_model, x_dense, EPSILON)
c_lo_d, c_hi_d   = predict_cqr_interval(cqr_model, x_dense)
cq_lo_d, cq_hi_d = predict_cqr_qrf_interval(cqr_qrf_model, x_dense)

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.scatter(x, y, s=14, alpha=0.45, color="steelblue", zorder=4, label="Data")

ax.fill_between(x_dense, our_lo_d, our_hi_d, alpha=0.25, color="steelblue")
ax.plot(x_dense, our_lo_d, color="steelblue", lw=1.8, label="Our method (CRPS conformal)")
ax.plot(x_dense, our_hi_d, color="steelblue", lw=1.8)

ax.plot(x_dense, g_lo_d, "--", color="tomato",   lw=1.5, label="Gaussian split conformal")
ax.plot(x_dense, g_hi_d, "--", color="tomato",   lw=1.5)

ax.plot(x_dense, c_lo_d, "-.", color="darkorange", lw=1.5, label="CQR (cubic)")
ax.plot(x_dense, c_hi_d, "-.", color="darkorange", lw=1.5)

ax.plot(x_dense, cq_lo_d, ":", color="purple",    lw=2.0, label="CQR-QRF")
ax.plot(x_dense, cq_hi_d, ":", color="purple",    lw=2.0)

for edge in edges[1:-1]:
    ax.axvline(edge, color="grey", lw=0.8, ls=":")

ax.set_xlabel("Waiting time (min)")
ax.set_ylabel("Eruption duration (min)")
ax.set_title(f"90% prediction intervals — Old Faithful ($\\varepsilon = {EPSILON}$)")
ax.legend(loc="upper left", ncol=2, fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(f"{FIGDIR}/fig_faithful_intervals.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved fig_faithful_intervals.pdf")

print("\nDone.")
