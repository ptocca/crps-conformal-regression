"""C2/C3: Fair comparison with standard errors over R random 50/50 splits.

For each of R=200 seeds on each dataset:
  - Random 50/50 split into train and calibration halves.
  - Competitors (Gaussian, CQR, CQR-QRF): fit on train, calibrate and evaluate on cal.
  - Our method (full n): fitted once on all n points (pre-computed); conformal interval
    uses full-bin ECDF; evaluated on the cal half.
  - Our method (n/2): fitted on train half only; conformal interval uses train-half ECDF;
    evaluated on cal half.

Reports mean coverage ± SE and mean interval width ± SE for all methods × datasets,
plus a LaTeX table fragment suitable for the manuscript.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from experiments.data import load_faithful, load_mcycle
from experiments.competitors import (
    fit_gaussian_split_conformal, predict_gaussian_interval,
    fit_cqr, predict_cqr_interval,
    fit_cqr_qrf, predict_cqr_qrf_interval,
)
from save_figures import (
    precompute_costs, optimal_partition, bin_x_boundaries,
    select_K_cv, conformal_interval,
)

# ── Parameters ────────────────────────────────────────────────────────────────

R       = 200
EPSILON = 0.10
SEED    = 0

DATASETS = [
    ("faithful", load_faithful, "min"),
    ("mcycle",   load_mcycle,   "g"),
]

METHODS = [
    "Our method (full n)",
    "Our method (n/2)",
    "Gaussian conformal",
    "CQR (cubic)",
    "CQR-QRF",
]

# ── Per-split runner ──────────────────────────────────────────────────────────

def run_split(x, y, tr_idx, cal_idx,
              bp_full, edges_full, K_full,
              seed):
    """Evaluate all methods on one random 50/50 split.

    x, y      — full dataset, sorted by x.
    tr_idx    — indices in the training half (unsorted; function will sort by x).
    cal_idx   — indices in the calibration half.
    bp_full, edges_full, K_full — pre-fitted full-data partition (same every seed).
    seed      — integer used as random_state for CQR-QRF.

    Returns dict: method_name -> (coverage, mean_width).
    """
    # Sort train by x (required for our DP)
    tr_sort    = tr_idx[np.argsort(x[tr_idx])]
    x_tr, y_tr = x[tr_sort], y[tr_sort]
    x_cal      = x[cal_idx]
    y_cal      = y[cal_idx]
    n_tr       = len(tr_idx)

    results = {}

    # ── Our method (full n) ───────────────────────────────────────────────────
    # Partition is pre-fitted on all n; only evaluation set changes.
    def full_interval(xq):
        bi = int(np.clip(np.searchsorted(edges_full, xq, "right") - 1, 0, K_full - 1))
        return conformal_interval(y[bp_full[bi]:bp_full[bi + 1]], EPSILON, n_grid=2000)

    lo = np.array([full_interval(xq)[0] for xq in x_cal])
    hi = np.array([full_interval(xq)[1] for xq in x_cal])
    results["Our method (full n)"] = (
        float(np.mean((y_cal >= lo) & (y_cal <= hi))),
        float(np.mean(hi - lo)),
    )

    # ── Our method (n/2) ──────────────────────────────────────────────────────
    K_max_half = max(1, n_tr // 10)
    K_half, _  = select_K_cv(x_tr, y_tr, K_max_half)
    C_half     = precompute_costs(y_tr)
    bp_half, _ = optimal_partition(y_tr, K_half, C=C_half)
    edges_half = bin_x_boundaries(x_tr, bp_half)

    def half_interval(xq):
        bi = int(np.clip(np.searchsorted(edges_half, xq, "right") - 1, 0, K_half - 1))
        return conformal_interval(y_tr[bp_half[bi]:bp_half[bi + 1]], EPSILON, n_grid=2000)

    lo = np.array([half_interval(xq)[0] for xq in x_cal])
    hi = np.array([half_interval(xq)[1] for xq in x_cal])
    results["Our method (n/2)"] = (
        float(np.mean((y_cal >= lo) & (y_cal <= hi))),
        float(np.mean(hi - lo)),
    )

    # ── Competitors (all use x_tr / y_tr for fitting) ─────────────────────────
    gauss = fit_gaussian_split_conformal(x_tr, y_tr, x_cal, y_cal)
    g_lo, g_hi = predict_gaussian_interval(gauss, x_cal, EPSILON)
    results["Gaussian conformal"] = (
        float(np.mean((y_cal >= g_lo) & (y_cal <= g_hi))),
        float(np.mean(g_hi - g_lo)),
    )

    cqr = fit_cqr(x_tr, y_tr, x_cal, y_cal, EPSILON, degree=3)
    c_lo, c_hi = predict_cqr_interval(cqr, x_cal)
    results["CQR (cubic)"] = (
        float(np.mean((y_cal >= c_lo) & (y_cal <= c_hi))),
        float(np.mean(c_hi - c_lo)),
    )

    cqr_qrf = fit_cqr_qrf(x_tr, y_tr, x_cal, y_cal, EPSILON, random_state=seed)
    cq_lo, cq_hi = predict_cqr_qrf_interval(cqr_qrf, x_cal)
    results["CQR-QRF"] = (
        float(np.mean((y_cal >= cq_lo) & (y_cal <= cq_hi))),
        float(np.mean(cq_hi - cq_lo)),
    )

    return results


# ── Main simulation ───────────────────────────────────────────────────────────

all_results = {}   # dataset_name -> {method: (cov_arr, wid_arr)}

for dsname, loader, width_unit in DATASETS:
    print(f"\nLoading {dsname}...", flush=True)
    x, y = loader()
    n = len(x)

    # Pre-fit full-data partition (deterministic, computed once)
    print(f"  Pre-fitting full partition (n={n})...", flush=True)
    K_max_full = n // 10
    K_full, _  = select_K_cv(x, y, K_max_full)
    C_full     = precompute_costs(y)
    bp_full, _ = optimal_partition(y, K_full, C=C_full)
    edges_full = bin_x_boundaries(x, bp_full)
    print(f"  K* (full) = {K_full}", flush=True)

    rng = np.random.default_rng(SEED)
    cov_arr = {m: np.empty(R) for m in METHODS}
    wid_arr = {m: np.empty(R) for m in METHODS}

    for r in range(R):
        if r % 50 == 0:
            print(f"  Seed {r}/{R}...", flush=True)
        perm    = rng.permutation(n)
        tr_idx  = perm[:n // 2]
        cal_idx = perm[n // 2:]

        res = run_split(x, y, tr_idx, cal_idx,
                        bp_full, edges_full, K_full,
                        seed=r)
        for m, (cov, wid) in res.items():
            cov_arr[m][r] = cov
            wid_arr[m][r] = wid

    all_results[dsname] = (cov_arr, wid_arr, width_unit, n)

# ── Print results ─────────────────────────────────────────────────────────────

for dsname, (cov_arr, wid_arr, width_unit, n) in all_results.items():
    print(f"\n{'='*70}")
    print(f"{dsname.upper()}  n={n}  R={R}  ε={EPSILON}  nominal={100*(1-EPSILON):.0f}%")
    print(f"{'='*70}")
    print(f"{'Method':<24}  {'Coverage (%)':>22}  {'Width ({})'.format(width_unit):>22}")
    print("-" * 72)
    for m in METHODS:
        cov_mu = 100 * cov_arr[m].mean()
        cov_se = 100 * cov_arr[m].std(ddof=1) / np.sqrt(R)
        wid_mu = wid_arr[m].mean()
        wid_se = wid_arr[m].std(ddof=1) / np.sqrt(R)
        marker = "  <-- our method" if "Our method" in m else ""
        print(f"{m:<24}  {cov_mu:>7.1f} ± {cov_se:<7.1f}  "
              f"{wid_mu:>8.3f} ± {wid_se:<8.3f}{marker}")

# ── LaTeX tables ──────────────────────────────────────────────────────────────

def latex_table(dsname, cov_arr, wid_arr, width_unit, n, label, caption):
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\hline")
    lines.append(f"Method & Coverage (\\%) & Mean width ({width_unit}) \\\\")
    lines.append(r"\hline")
    for m in METHODS:
        cov_mu = 100 * cov_arr[m].mean()
        cov_se = 100 * cov_arr[m].std(ddof=1) / np.sqrt(R)
        wid_mu = wid_arr[m].mean()
        wid_se = wid_arr[m].std(ddof=1) / np.sqrt(R)
        name_tex = m.replace("(", r"\textup{(}").replace(")", r"\textup{)}")
        if "full" in m:
            name_tex = r"\textbf{" + name_tex + r"}"
        elif "n/2" in m:
            name_tex = r"\textit{" + name_tex + r"}"
        cov_str = f"${cov_mu:.1f} \\pm {cov_se:.1f}$"
        wid_str = f"${wid_mu:.3f} \\pm {wid_se:.3f}$"
        lines.append(f"{name_tex} & {cov_str} & {wid_str} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


for dsname, (cov_arr, wid_arr, width_unit, n) in all_results.items():
    label   = f"tab:{dsname}_coverage"
    if dsname == "faithful":
        caption = (
            f"Old Faithful: empirical coverage and mean prediction-interval width "
            f"at nominal level $1-\\varepsilon = {1-EPSILON:.2f}$, "
            f"averaged over $R={R}$ random 50/50 splits ($n={n}$). "
            r"$\pm$ one standard error. "
            r"\textbf{Our method (full $n$)} uses all $n$ observations for fitting; "
            r"\textit{Our method ($n/2$)} uses the same training half as the competitors."
        )
    else:
        caption = (
            f"Motorcycle accident: empirical coverage and mean prediction-interval width "
            f"at nominal level $1-\\varepsilon = {1-EPSILON:.2f}$, "
            f"averaged over $R={R}$ random 50/50 splits ($n={n}$). "
            r"$\pm$ one standard error. "
            r"\textbf{Our method (full $n$)} uses all $n$ observations for fitting; "
            r"\textit{Our method ($n/2$)} uses the same training half as the competitors."
        )
    print(f"\n% ── LaTeX table: {dsname} ────────────────────────────────────────")
    print(latex_table(dsname, cov_arr, wid_arr, width_unit, n, label, caption))
