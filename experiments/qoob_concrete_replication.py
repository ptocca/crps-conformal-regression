"""Replicate Gupta, Kuchibhotla & Ramdas (2022) Table 2/3 protocol on the
Concrete Compressive Strength UCI dataset, as an independent correctness
check of our QOOB reimplementation against the paper's own published numbers
(not just our synthetic sweep / brute-force aggregation check).

Paper protocol (Section 6, "Numerical comparisons"):
  - dataset: n=1030 (UCI "concrete strength" -- 8 covariates, compressive
    strength target)
  - 100 independent resamples of 1000 points drawn without replacement from
    the full dataset
  - each resample split into train (768) / test (232)
  - forest: T=100 bagged quantile regression trees
  - alpha=0.1, nominal quantile level beta=2*alpha (their recommended default)
  - report Ave-Mean-Width (16) and Ave-Mean-Coverage (17) across the 100
    resamples

Their Table 2/3 numbers for QOOB-100 (2*alpha) on concrete:
  mean-width = 18.19 (se 0.05), mean-coverage = 0.92

This script first runs a small pilot (few resamples) to check correctness
and measure wall-clock cost per resample, then scales up.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from experiments.qoob import qoob_fit_and_calibrate, qoob_predict

DATA_CACHE = os.path.join(os.path.dirname(__file__), "concrete_data.npz")


def load_concrete():
    if os.path.exists(DATA_CACHE):
        d = np.load(DATA_CACHE)
        return d["X"], d["y"]
    from sklearn.datasets import fetch_openml
    d = fetch_openml(name="Concrete_Compressive_Strength", version=7, as_frame=True)
    X = d.data.to_numpy(dtype=float)
    y = d.target.to_numpy(dtype=float)
    np.savez(DATA_CACHE, X=X, y=y)
    return X, y


def run_one_resample(X_full, y_full, epsilon, T, min_samples_leaf, seed):
    n_full = len(y_full)
    rng = np.random.default_rng(seed)
    idx1000 = rng.choice(n_full, size=1000, replace=False)
    perm = rng.permutation(1000)
    tr_idx = idx1000[perm[:768]]
    te_idx = idx1000[perm[768:]]

    X_tr, y_tr = X_full[tr_idx], y_full[tr_idx]
    X_te, y_te = X_full[te_idx], y_full[te_idx]

    model = qoob_fit_and_calibrate(
        X_tr, y_tr, epsilon, T=T, min_samples_leaf=min_samples_leaf, random_state=seed,
    )
    pred_sets = qoob_predict(model, X_te)

    widths = np.array([sum(hi - lo for lo, hi in segs) for segs in pred_sets])
    covered = np.array([
        any(lo <= y_te[k] <= hi for lo, hi in segs) for k, segs in enumerate(pred_sets)
    ])
    return covered.mean(), widths.mean()


if __name__ == "__main__":
    EPSILON = 0.10
    T = 100
    MIN_SAMPLES_LEAF = 5  # paper does not report tuning this; try their implicit default first

    X, y = load_concrete()
    print(f"Concrete dataset: n={len(y)}, d={X.shape[1]}", flush=True)

    N_PILOT = 3
    print(f"\nPilot: {N_PILOT} resamples, timing each...", flush=True)
    covs, wids = [], []
    for r in range(N_PILOT):
        t0 = time.time()
        cov, wid = run_one_resample(X, y, EPSILON, T, MIN_SAMPLES_LEAF, seed=r)
        dt = time.time() - t0
        covs.append(cov); wids.append(wid)
        print(f"  resample {r}: coverage={cov:.3f} width={wid:.3f}  ({dt:.1f}s)", flush=True)

    print(f"\nPilot mean over {N_PILOT}: coverage={np.mean(covs):.3f} width={np.mean(wids):.3f}")
    print("Paper's Table 2/3 for QOOB-100 (2a) on concrete: width=18.19 (se 0.05), coverage=0.92")
