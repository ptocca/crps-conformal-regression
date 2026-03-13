"""Reproduce all figures from conformal_binning.ipynb and save as PDFs."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.rcParams.update({
    "text.usetex": False,
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

FIGDIR = "figures"

# ── core functions ────────────────────────────────────────────────────────────

def pairwise_abs_sum(y):
    ys = np.sort(y); m = len(ys)
    return float(np.dot(2 * np.arange(m) - (m - 1), ys))

def precompute_costs(y):
    n = len(y); C = np.full((n, n), np.inf)
    for i in range(n):
        W = 0.; sv = []; pf = [0.]
        for j in range(i, n):
            val = y[j]; lo, hi = 0, len(sv)
            while lo < hi:
                mid = (lo + hi) // 2
                if sv[mid] < val: lo = mid + 1
                else: hi = mid
            r = lo; Sle = pf[r]; mc = len(sv); Sgt = pf[mc] - Sle
            W += val * r - Sle + Sgt - val * (mc - r)
            sv.insert(r, val); pf.insert(r + 1, Sle + val)
            for k in range(r + 2, len(pf)): pf[k] += val
            m = j - i + 1; C[i, j] = np.inf if m < 2 else m * W / (m - 1) ** 2
    return C

def optimal_partition(y, K, C=None):
    n = len(y)
    if C is None: C = precompute_costs(y)
    dp = np.full((K + 1, n + 1), np.inf)
    sp = np.full((K + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.
    for k in range(1, K + 1):
        for j in range(2 * k, n + 1):
            for i in range(2 * (k - 1), j - 1):
                v = dp[k - 1, i] + C[i, j - 1]
                if v < dp[k, j]: dp[k, j] = v; sp[k, j] = i
    bps = [n]; j = n
    for k in range(K, 0, -1): i = sp[k, j]; bps.append(i); j = i
    bps.reverse(); return bps, dp[K, n]

def bin_x_boundaries(x, bp):
    K = len(bp) - 1; e = np.empty(K + 1)
    e[0] = -np.inf; e[-1] = np.inf
    for k in range(1, K): e[k] = 0.5 * (x[bp[k] - 1] + x[bp[k]])
    return e

def crps_empirical(ys, y):
    m = len(ys)
    if m == 0: return np.nan
    return np.mean(np.abs(ys - y)) - pairwise_abs_sum(ys) / m ** 2

def select_K_cv(x, y, K_max):
    n = len(x)
    tr = np.arange(0, n, 2); te = np.arange(1, n, 2)
    x_tr, y_tr = x[tr], y[tr]; x_te, y_te = x[te], y[te]
    C_tr = precompute_costs(y_tr)
    tc = np.full(K_max, np.inf)
    for K in range(1, K_max + 1):
        if K > len(y_tr) // 2: break
        bp, _ = optimal_partition(y_tr, K, C=C_tr)
        edges = bin_x_boundaries(x_tr, bp); Kb = len(bp) - 1
        total = sum(
            crps_empirical(
                y_tr[bp[int(np.clip(np.searchsorted(edges, xq, 'right') - 1, 0, Kb - 1))]:
                     bp[int(np.clip(np.searchsorted(edges, xq, 'right') - 1, 0, Kb - 1)) + 1]],
                yq)
            for xq, yq in zip(x_te, y_te))
        tc[K - 1] = total / len(y_te)
    return int(np.argmin(tc)) + 1, tc

def conformal_pvalue_grid(yb, yh):
    m = len(yb); yh = np.asarray(yh)
    ad = np.abs(yb[:, None] - yh[None, :]); sa = ad.sum(0)
    Wm = pairwise_abs_sum(yb); ah = sa / m - Wm / m ** 2
    d = np.abs(yb[:, None] - yb[None, :]).sum(1); A = sa[None, :] - ad
    aj = (d[:, None] + ad) / m - (Wm - d[:, None]) / m ** 2 - A / m ** 2
    return ((aj >= ah[None, :]).sum(0) + 1) / (m + 1)

def conformal_interval(yb, eps, n_grid=2000):
    std = np.std(yb) if len(yb) > 1 else 1.
    yg = np.linspace(yb.min() - 4 * std, yb.max() + 4 * std, n_grid)
    mask = conformal_pvalue_grid(yb, yg) > eps
    return (float(yg[mask][0]), float(yg[mask][-1])) if mask.any() else (np.nan, np.nan)

# ── data ──────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)
n = 1000
x = np.sort(rng.uniform(0, 3, n))
y = rng.normal(loc=3 * x, scale=1 + x)

K_max = 20
K_opt, test_crps = select_K_cv(x, y, K_max)
C_all = precompute_costs(y)
bp, _ = optimal_partition(y, K_opt, C=C_all)
edges = bin_x_boundaries(x, bp)
print(f"\n=== Synthetic example results (n={n}) ===")
print(f"K* = {K_opt}")
for k in range(K_opt):
    mk = bp[k+1] - bp[k]
    xlo, xhi = x[bp[k]], x[bp[k+1]-1]
    print(f"  B{k+1}: x in [{xlo:.2f}, {xhi:.2f}], m_{k+1} = {mk}, 1/(m+1) = {1/(mk+1):.4f}")
Ks = list(range(1, K_max + 1))
loo = [optimal_partition(y, K_, C=C_all)[1] if K_ <= n // 2 else np.inf for K_ in Ks]

epsilon = 0.10
colors = plt.cm.Set2.colors

# ── Fig 1: K selection ────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].plot(Ks, loo, marker='.', markersize=4, color='steelblue')
axes[0].set_xlabel('$K$'); axes[0].set_ylabel('Total LOO-CRPS')
axes[0].set_title('Within-sample LOO-CRPS')
axes[0].grid(True, alpha=0.3)

axes[1].plot(Ks, test_crps, marker='.', markersize=4, color='tomato')
axes[1].axvline(K_opt, color='tomato', ls='--', lw=1.5, label=f'$K^* = {K_opt}$')
axes[1].set_xlabel('$K$'); axes[1].set_ylabel('Average test CRPS')
axes[1].set_title('Cross-validated test CRPS (U-shaped)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{FIGDIR}/fig_kselect.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved fig_kselect.pdf")

# ── Fig 2: partition scatter ──────────────────────────────────────────────────

x_disp = np.clip(edges, x[0] - 0.05 * (x[-1] - x[0]), x[-1] + 0.05 * (x[-1] - x[0]))
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.scatter(x, y, s=15, alpha=0.5, color='steelblue', zorder=3)
for b in range(K_opt):
    ax.axvspan(x_disp[b], x_disp[b + 1], alpha=0.2, color=colors[b % len(colors)])
    if b > 0:
        ax.axvline(edges[b], color='grey', lw=0.8, ls=':')
ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
ax.set_title(f'Optimal {K_opt}-bin partition ($K^* = {K_opt}$ selected by cross-validation)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{FIGDIR}/fig_partition.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved fig_partition.pdf")

# ── Fig 3: Venn band ──────────────────────────────────────────────────────────

t_grid = np.linspace(-5, 25, 500)
fig, axes = plt.subplots(1, K_opt, figsize=(4 * K_opt, 3.8), sharey=True)
for b, ax in enumerate(axes):
    lo, hi = bp[b], bp[b + 1]; yb = y[lo:hi]; m = len(yb)
    x_mid = 0.5 * (x[lo] + x[hi - 1])
    F_m = np.mean(yb[:, None] <= t_grid[None, :], axis=0)
    F_lo = m / (m + 1) * F_m; F_hi = F_lo + 1 / (m + 1)
    true_cdf = norm.cdf(t_grid, loc=3 * x_mid, scale=1 + x_mid)
    ax.fill_between(t_grid, F_lo, F_hi, alpha=0.35, color='steelblue',
                    label=f'Venn band (width $1/{m+1}$)')
    ax.step(t_grid, F_m, where='post', color='steelblue', lw=1.5,
            label='Training ECDF $\\hat{F}_m$')
    ax.plot(t_grid, true_cdf, '--', color='tomato', lw=1.5,
            label=f'True CDF at $x = {x_mid:.2f}$')
    ax.set_title(f'Bin {b+1},  $m = {m}$')
    ax.set_xlabel('$t$'); ax.legend()
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel('$F(t)$')
fig.suptitle('Venn prediction band (shaded) around training ECDF', y=1.02)
plt.tight_layout()
fig.savefig(f"{FIGDIR}/fig_venn.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved fig_venn.pdf")

# ── Fig 4: p-value curves ─────────────────────────────────────────────────────

x_queries = [0.3, 1.5, 2.7]
fig, axes = plt.subplots(1, len(x_queries), figsize=(6 * len(x_queries), 3.8))
axes = np.atleast_1d(axes)
for ax, xq in zip(axes, x_queries):
    bin_idx = int(np.clip(np.searchsorted(edges, xq, 'right') - 1, 0, K_opt - 1))
    lo, hi = bp[bin_idx], bp[bin_idx + 1]; yb = y[lo:hi]; m = len(yb)
    std = np.std(yb)
    yg = np.linspace(yb.mean() - 3.5 * std, yb.mean() + 3.5 * std, 800)
    pv = conformal_pvalue_grid(yb, yg); mask = pv > epsilon
    ax.plot(yg, pv, color='steelblue', lw=1.5)
    ax.axhline(epsilon, color='grey', lw=1, ls='--', label=f'$\\varepsilon = {epsilon}$')
    ax.fill_between(yg, 0, pv, where=mask, alpha=0.25, color='steelblue',
                    label=f'$\\Gamma^{{{epsilon}}}$')
    tlo = norm.ppf(epsilon / 2, loc=3 * xq, scale=1 + xq)
    thi = norm.ppf(1 - epsilon / 2, loc=3 * xq, scale=1 + xq)
    ax.axvline(tlo, color='tomato', lw=1.5, label='True 90\\% interval')
    ax.axvline(thi, color='tomato', lw=1.5)
    ax.set_title(f'$x^* = {xq}$,  bin {bin_idx+1},  $m = {m}$')
    ax.set_xlabel('$y_h$'); ax.set_ylabel('$p(y_h)$')
    ax.legend(); ax.grid(True, alpha=0.3)
fig.suptitle(f'Conformal p-value as a function of $y_h$  ($\\varepsilon = {epsilon}$)', y=1.02)
plt.tight_layout()
fig.savefig(f"{FIGDIR}/fig_pvalue.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved fig_pvalue.pdf")

# ── Print conformal intervals for Table ───────────────────────────────────────
print("\n=== Conformal intervals (Table~\\ref{tab:ci}) ===")
for xq in x_queries:
    bin_idx = int(np.clip(np.searchsorted(edges, xq, 'right') - 1, 0, K_opt - 1))
    lo, hi = bp[bin_idx], bp[bin_idx + 1]; yb = y[lo:hi]
    clo, chi = conformal_interval(yb, epsilon)
    true_w = norm.ppf(1 - epsilon / 2, loc=3 * xq, scale=1 + xq) - norm.ppf(epsilon / 2, loc=3 * xq, scale=1 + xq)
    print(f"  x*={xq}: bin {bin_idx+1}, Gamma=[{clo:.2f}, {chi:.2f}], width={chi-clo:.2f}, true 90% width={true_w:.2f}")

# ── Fig 5: fan plot ───────────────────────────────────────────────────────────

x_test = np.linspace(x[2], x[-3], 100)
ci_lo = np.empty(len(x_test)); ci_hi = np.empty(len(x_test))
for q, xq in enumerate(x_test):
    bi = int(np.clip(np.searchsorted(edges, xq, 'right') - 1, 0, K_opt - 1))
    lo, hi = bp[bi], bp[bi + 1]
    ci_lo[q], ci_hi[q] = conformal_interval(y[lo:hi], epsilon)

true_lo = norm.ppf(epsilon / 2, loc=3 * x_test, scale=1 + x_test)
true_hi = norm.ppf(1 - epsilon / 2, loc=3 * x_test, scale=1 + x_test)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(x, y, s=12, alpha=0.4, color='steelblue', zorder=3, label='Training data')
ax.fill_between(x_test, ci_lo, ci_hi, alpha=0.3, color='steelblue',
                label=f'Conformal {int(100*(1-epsilon))}\\% interval')
ax.plot(x_test, ci_lo, color='steelblue', lw=1)
ax.plot(x_test, ci_hi, color='steelblue', lw=1)
ax.plot(x_test, true_lo, '--', color='tomato', lw=1.5,
        label=f'True {int(100*(1-epsilon))}\\% interval')
ax.plot(x_test, true_hi, '--', color='tomato', lw=1.5)
for edge in edges[1:-1]:
    ax.axvline(edge, color='grey', lw=0.8, ls=':')
ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
ax.set_title(f'Conformal prediction intervals — {K_opt}-bin partition,  '
             f'$\\varepsilon = {epsilon}$')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{FIGDIR}/fig_fan.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved fig_fan.pdf")

# ── Fig 6: coverage histogram ─────────────────────────────────────────────────

rng_te = np.random.default_rng(999)
n_te = 2000
x_te = np.sort(rng_te.uniform(0, 3, n_te))
y_te = rng_te.normal(loc=3 * x_te, scale=1 + x_te)
pvals_te = np.array([
    conformal_pvalue_grid(
        y[bp[int(np.clip(np.searchsorted(edges, xq, 'right') - 1, 0, K_opt - 1))]:
          bp[int(np.clip(np.searchsorted(edges, xq, 'right') - 1, 0, K_opt - 1)) + 1]],
        np.array([yq]))[0]
    for xq, yq in zip(x_te, y_te)
])

print("\n=== Empirical coverage on test set ===")
for eps in [0.05, 0.10, 0.20]:
    cov = np.mean(pvals_te > eps)
    print(f"  eps={eps}: coverage={cov*100:.1f}% (nominal {100*(1-eps):.0f}%)")

fig, ax = plt.subplots(figsize=(5, 3.8))
ax.hist(pvals_te, bins=20, density=True, color='steelblue', alpha=0.7, label='p-values')
ax.axhline(1.0, color='tomato', ls='--', lw=1.5, label='Uniform(0,1)')
ax.set_xlabel('$p(y^*)$'); ax.set_ylabel('Density')
ax.set_title(f'P-value distribution on test set ($n={n_te}$)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{FIGDIR}/fig_coverage.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved fig_coverage.pdf")

print("\nAll figures saved.")
