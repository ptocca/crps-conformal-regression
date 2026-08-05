"""CHR benchmark on the paper's two real datasets (Old Faithful, motorcycle).

CHR (Sesia & Romano 2021) is split-conformal, like CQR-QRF -- same train/
calibrate protocol, same R=200 random 50/50 splits as fair_comparison.py, same
base seed convention (SEED=0, one rng per dataset). The grey-box QRF uses the
same (n_estimators=500, min_samples_leaf=5) config as CQR-QRF's bbox in
experiments/competitors.py, so that CHR vs. CQR-QRF isolates the value of the
histogram/shortest-window interval construction from the choice of base
learner.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from experiments.data import load_faithful, load_mcycle
from experiments.chr import chr_fit_and_calibrate, chr_predict
from experiments.competitors import fit_cqr_qrf, predict_cqr_qrf_interval

R       = 200
EPSILON = 0.10
SEED    = 0
MSL     = 5
N_TREES = 500

DATASETS = [
    ("faithful", load_faithful, "min"),
    ("mcycle",   load_mcycle,   "g"),
]

for dsname, loader, width_unit in DATASETS:
    print(f"\n{'='*70}\n{dsname.upper()}\n{'='*70}", flush=True)
    x, y = loader()
    n = len(x)
    y_min = y.min() - 0.05 * (y.max() - y.min())
    y_max = y.max() + 0.05 * (y.max() - y.min())

    rng = np.random.default_rng(SEED)
    cov_chr = np.empty(R); wid_chr = np.empty(R)
    cov_cqrqrf = np.empty(R); wid_cqrqrf = np.empty(R)

    for r in range(R):
        perm = rng.permutation(n)
        tr_idx = perm[:n // 2]
        cal_idx = perm[n // 2:]
        x_tr, y_tr = x[tr_idx], y[tr_idx]
        x_cal, y_cal = x[cal_idx], y[cal_idx]

        model = chr_fit_and_calibrate(
            x_tr, y_tr, x_cal, y_cal, EPSILON, y_min, y_max,
            n_bins=200, min_samples_leaf=MSL, n_estimators=N_TREES, random_state=r,
        )
        lo, hi = chr_predict(model, x_cal)
        cov_chr[r] = np.mean((y_cal >= lo) & (y_cal <= hi))
        wid_chr[r] = np.mean(hi - lo)

        cqrqrf = fit_cqr_qrf(x_tr, y_tr, x_cal, y_cal, EPSILON, n_estimators=N_TREES, random_state=r)
        cq_lo, cq_hi = predict_cqr_qrf_interval(cqrqrf, x_cal)
        cov_cqrqrf[r] = np.mean((y_cal >= cq_lo) & (y_cal <= cq_hi))
        wid_cqrqrf[r] = np.mean(cq_hi - cq_lo)

        if r % 20 == 0:
            print(f"    split {r}/{R}...", flush=True)

    print(f"\n  {'Method':<16}{'Coverage (%)':>18}{'Width ({})'.format(width_unit):>18}")
    for name, cov, wid in [("CHR", cov_chr, wid_chr), ("CQR-QRF (x-check)", cov_cqrqrf, wid_cqrqrf)]:
        cmu, cse = 100 * cov.mean(), 100 * cov.std(ddof=1) / np.sqrt(R)
        wmu, wse = wid.mean(), wid.std(ddof=1) / np.sqrt(R)
        print(f"  {name:<16}{cmu:>10.1f} ± {cse:<5.1f}{wmu:>10.3f} ± {wse:<5.3f}")
