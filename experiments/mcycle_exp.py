"""Motorcycle accident experiment: heteroscedastic benchmark."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from experiments.data import load_mcycle
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

from save_figures import (
    pairwise_abs_sum, precompute_costs, optimal_partition,
    bin_x_boundaries, crps_empirical, select_K_cv,
    conformal_pvalue_grid, conformal_interval,
)

# ── Load data ─────────────────────────────────────────────────────────────────

x, y = load_mcycle()
n = len(x)
print(f"Mcycle: n={n}, x=[{x.min():.1f},{x.max():.1f}] ms, y=[{y.min():.1f},{y.max():.1f}] g")

# ── Fit our method ────────────────────────────────────────────────────────────

K_max = n // 10
K_opt, test_crps_curve = select_K_cv(x, y, K_max)
C_all = precompute_costs(y)
bp, _ = optimal_partition(y, K_opt, C=C_all)
edges = bin_x_boundaries(x, bp)
print(f"K* = {K_opt}, boundaries at x = {[f'{e:.1f}' for e in edges[1:-1]]} ms")

# ── Train/cal split for competitors ──────────────────────────────────────────

tr_idx  = np.arange(0, n, 2)
cal_idx = np.arange(1, n, 2)
x_tr, y_tr   = x[tr_idx],  y[tr_idx]
x_cal, y_cal = x[cal_idx], y[cal_idx]

gauss_model    = fit_gaussian_split_conformal(x_tr, y_tr, x_cal, y_cal)
cqr_model      = fit_cqr(x_tr, y_tr, x_cal, y_cal, EPSILON, degree=3)
cqr_qrf_model  = fit_cqr_qrf(x_tr, y_tr, x_cal, y_cal, EPSILON)

# ── Coverage evaluation ───────────────────────────────────────────────────────

def our_interval(xq):
    bi = int(np.clip(np.searchsorted(edges, xq, "right") - 1, 0, K_opt - 1))
    return conformal_interval(y[bp[bi]:bp[bi + 1]], EPSILON, n_grid=3000)

def coverage_width(lo_arr, hi_arr, y_eval):
    covered = np.mean((y_eval >= lo_arr) & (y_eval <= hi_arr))
    width   = np.mean(hi_arr - lo_arr)
    return covered, width

our_lo  = np.array([our_interval(xq)[0] for xq in x_cal])
our_hi  = np.array([our_interval(xq)[1] for xq in x_cal])
g_lo, g_hi         = predict_gaussian_interval(gauss_model, x_cal, EPSILON)
c_lo, c_hi         = predict_cqr_interval(cqr_model, x_cal)
cq_lo, cq_hi       = predict_cqr_qrf_interval(cqr_qrf_model, x_cal)

results = {
    "Our method":         coverage_width(our_lo, our_hi,  y_cal),
    "Gaussian conformal": coverage_width(g_lo,   g_hi,    y_cal),
    "CQR (cubic)":        coverage_width(c_lo,   c_hi,    y_cal),
    "CQR-QRF":            coverage_width(cq_lo,  cq_hi,   y_cal),
}

print(f"\nMotorcycle — nominal coverage {int(100*(1-EPSILON))}%")
print(f"{'Method':<22} {'Coverage':>10}  {'Mean width':>10}")
print("-" * 46)
for name, (cov, wid) in results.items():
    print(f"{name:<22} {100*cov:>9.1f}%  {wid:>10.1f}")

# ── LaTeX table ───────────────────────────────────────────────────────────────

def print_latex_table():
    print("\n% --- LaTeX table ---")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\begin{tabular}{lcc}")
    print(r"\hline")
    print(r"Method & Coverage (\%) & Mean width (g) \\")
    print(r"\hline")
    for name, (cov, wid) in results.items():
        bold_open  = r"\textbf{" if name == "Our method" else ""
        bold_close = r"}" if name == "Our method" else ""
        print(f"{bold_open}{name}{bold_close} & {100*cov:.1f} & {wid:.1f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Motorcycle: empirical coverage and mean prediction-interval width "
          r"at nominal level $1-\varepsilon = 0.90$. "
          r"Coverage evaluated on the held-out alternating split ($n_{\rm cal}=67$).}")
    print(r"\label{tab:mcycle_coverage}")
    print(r"\end{table}")

print_latex_table()

# ── Figure 1: K-selection + partition scatter ─────────────────────────────────

Ks = list(range(1, K_max + 1))
loo = [optimal_partition(y, K_, C=C_all)[1] if K_ <= n // 2 else np.inf for K_ in Ks]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

ax = axes[0]
ax.plot(Ks, test_crps_curve, marker=".", markersize=5, color="tomato")
ax.axvline(K_opt, color="tomato", ls="--", lw=1.5, label=f"$K^* = {K_opt}$")
ax.set_xlabel("$K$")
ax.set_ylabel("Average test CRPS")
ax.set_title("Cross-validated test CRPS")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
x_disp = np.clip(edges,
                 x[0]  - 0.02 * (x[-1] - x[0]),
                 x[-1] + 0.02 * (x[-1] - x[0]))
ax.scatter(x, y, s=14, alpha=0.55, color="steelblue", zorder=3)
for b in range(K_opt):
    ax.axvspan(x_disp[b], x_disp[b + 1], alpha=0.18, color=COLORS[b % len(COLORS)])
    if b > 0:
        ax.axvline(edges[b], color="grey", lw=0.9, ls=":")
ax.set_xlabel("Time after impact (ms)")
ax.set_ylabel("Head acceleration (g)")
ax.set_title(f"Optimal {K_opt}-bin partition ($K^* = {K_opt}$)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(f"{FIGDIR}/fig_mcycle_partition.pdf", bbox_inches="tight")
plt.close(fig)
print("\nSaved fig_mcycle_partition.pdf")

# ── Figure 2: prediction interval fan ─────────────────────────────────────────

x_dense = np.linspace(x.min(), x.max(), 300)
our_lo_d = np.array([our_interval(xq)[0] for xq in x_dense])
our_hi_d = np.array([our_interval(xq)[1] for xq in x_dense])
g_lo_d,  g_hi_d   = predict_gaussian_interval(gauss_model, x_dense, EPSILON)
c_lo_d,  c_hi_d   = predict_cqr_interval(cqr_model, x_dense)
cq_lo_d, cq_hi_d  = predict_cqr_qrf_interval(cqr_qrf_model, x_dense)

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.scatter(x, y, s=14, alpha=0.45, color="steelblue", zorder=4, label="Data")

ax.fill_between(x_dense, our_lo_d, our_hi_d, alpha=0.25, color="steelblue")
ax.plot(x_dense, our_lo_d, color="steelblue",   lw=1.8, label="Our method (CRPS conformal)")
ax.plot(x_dense, our_hi_d, color="steelblue",   lw=1.8)

ax.plot(x_dense, g_lo_d, "--",  color="tomato",    lw=1.5, label="Gaussian split conformal")
ax.plot(x_dense, g_hi_d, "--",  color="tomato",    lw=1.5)

ax.plot(x_dense, c_lo_d, "-.",  color="darkorange", lw=1.5, label="CQR (cubic)")
ax.plot(x_dense, c_hi_d, "-.",  color="darkorange", lw=1.5)

ax.plot(x_dense, cq_lo_d, ":",  color="purple",    lw=2.0, label="CQR-QRF")
ax.plot(x_dense, cq_hi_d, ":",  color="purple",    lw=2.0)

for edge in edges[1:-1]:
    ax.axvline(edge, color="grey", lw=0.8, ls=":")

ax.set_xlabel("Time after impact (ms)")
ax.set_ylabel("Head acceleration (g)")
ax.set_title(f"90% prediction intervals — motorcycle ($\\varepsilon = {EPSILON}$)")
ax.legend(loc="upper right", ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(f"{FIGDIR}/fig_mcycle_intervals.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved fig_mcycle_intervals.pdf")

print("\nDone.")
