"""QOOB benchmark on the paper's two real datasets (Old Faithful, motorcycle).

QOOB is transductive (Gupta, Kuchibhotla & Ramdas 2022): no train/calibration
split, so -- exactly like "Our method (full n)" elsewhere in this project's
fair-comparison harness -- it is fit once on the full n points and its
per-point (covered, width) outcomes are precomputed once, then averaged over
the same R random calibration-half subsets used for the split-conformal
competitors, for a directly comparable row in the same table.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from experiments.data import load_faithful, load_mcycle
from experiments.qoob import qoob_full_benchmark
from experiments.competitors import (
    fit_cqr_qrf, predict_cqr_qrf_interval,
)

R       = 200
EPSILON = 0.10
SEED    = 0
T_QOOB  = 200

DATASETS = [
    ("faithful", load_faithful, "min"),
    ("mcycle",   load_mcycle,   "g"),
]

for dsname, loader, width_unit in DATASETS:
    print(f"\n{'='*70}\n{dsname.upper()}\n{'='*70}", flush=True)
    x, y = loader()
    n = len(x)

    min_samples_leaf = max(5, n // 10)
    print(f"  Fitting QOOB once on full data (n={n}, T={T_QOOB}, min_samples_leaf={min_samples_leaf})...", flush=True)
    qoob_res = qoob_full_benchmark(x, y, EPSILON, T=T_QOOB, min_samples_leaf=min_samples_leaf, random_state=1)
    print(f"  QOOB (all {n} points as eval locations): "
          f"coverage={qoob_res['covered'].mean():.3f}  "
          f"mean_width={qoob_res['width'].mean():.4f}  "
          f"max_segments={qoob_res['n_segments'].max()}  "
          f"frac_nonconvex={np.mean(qoob_res['n_segments'] > 1):.3f}", flush=True)

    rng = np.random.default_rng(SEED)
    cov_qoob = np.empty(R); wid_qoob = np.empty(R)
    cov_cqrqrf = np.empty(R); wid_cqrqrf = np.empty(R)

    for r in range(R):
        perm = rng.permutation(n)
        tr_idx = perm[:n // 2]
        cal_idx = perm[n // 2:]

        # QOOB: just look up the precomputed per-point outcomes at cal_idx.
        cov_qoob[r] = qoob_res["covered"][cal_idx].mean()
        wid_qoob[r] = qoob_res["width"][cal_idx].mean()

        # CQR-QRF: fit fresh on the train half, exactly as in fair_comparison.py.
        tr_sort = tr_idx[np.argsort(x[tr_idx])]
        x_tr, y_tr = x[tr_sort], y[tr_sort]
        x_cal, y_cal = x[cal_idx], y[cal_idx]
        model = fit_cqr_qrf(x_tr, y_tr, x_cal, y_cal, EPSILON, random_state=r)
        lo, hi = predict_cqr_qrf_interval(model, x_cal)
        cov_cqrqrf[r] = np.mean((y_cal >= lo) & (y_cal <= hi))
        wid_cqrqrf[r] = np.mean(hi - lo)

        if r % 50 == 0:
            print(f"    split {r}/{R}...", flush=True)

    print(f"\n  {'Method':<16}{'Coverage (%)':>18}{'Width ({})'.format(width_unit):>18}")
    for name, cov, wid in [("QOOB", cov_qoob, wid_qoob), ("CQR-QRF", cov_cqrqrf, wid_cqrqrf)]:
        cmu, cse = 100*cov.mean(), 100*cov.std(ddof=1)/np.sqrt(R)
        wmu, wse = wid.mean(), wid.std(ddof=1)/np.sqrt(R)
        print(f"  {name:<16}{cmu:>10.1f} ± {cse:<5.1f}{wmu:>10.3f} ± {wse:<5.3f}")
