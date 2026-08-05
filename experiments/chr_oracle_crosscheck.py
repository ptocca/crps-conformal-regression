"""Validate experiments/chr.py against the actual CHR reference implementation.

This is NOT a benchmark script (see chr_benchmark.py for that) -- it is a
correctness check. github.com/msesia/chr carries no LICENSE file, so we do not
vendor or redistribute it anywhere in this repository; instead this script
expects you to clone it yourself and points to it at runtime, using it purely
as a local oracle to compare against our independent reimplementation.

Setup:
    git clone https://github.com/msesia/chr /path/to/chr_upstream

The reference package transitively imports torch (via chr/utils.py) and,
through chr/__init__.py, two branches we don't need and that don't import
cleanly on a modern stack: chr/black_boxes.py (old torch API + the
long-abandoned `skgarden` package) and chr/others_r.py (rpy2 / R bridge). We
stub those three submodules out below -- they're irrelevant to the algorithm
under test, since we supply our own grey-box (the same quantile_forest QRF
used elsewhere in this project) rather than the reference's bundled ones.

Install (a throwaway venv is recommended; torch is only needed to satisfy the
reference's unused import chain, not for any actual computation here):
    pip install torch --index-url https://download.pytorch.org/whl/cpu \\
        pandas scipy tqdm matplotlib quantile-forest scikit-learn

Usage:
    python chr_oracle_crosscheck.py --chr-repo /path/to/chr_upstream [--r 15]
"""
import argparse
import os
import sys
import types

os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_reference(chr_repo_path: str):
    sys.path.insert(0, chr_repo_path)
    for name in ["chr.black_boxes", "chr.black_boxes_r", "chr.others_r"]:
        sys.modules[name] = types.ModuleType(name)
    from chr.methods import CHR as RefCHR
    return RefCHR


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chr-repo", required=True, help="path to a local clone of github.com/msesia/chr")
    p.add_argument("--r", type=int, default=15, help="number of random splits per dataset (default 15)")
    args = p.parse_args()

    RefCHR = load_reference(args.chr_repo)

    import numpy as np
    from experiments.data import load_faithful, load_mcycle
    from experiments.chr import (
        fit_qrf_greybox, _predicted_quantiles, DEFAULT_QUANTILE_LEVELS,
        chr_fit_and_calibrate, chr_predict,
    )

    EPSILON = 0.10
    N_BINS = 200
    N_TREES = 500
    MSL = 5

    class QRFBbox:
        """Wraps our quantile_forest QRF to satisfy the reference CHR's bbox
        interface: get_quantiles() / fit(X, Y) / predict(X) -> (n, n_quantiles).
        Using the identical grey-box in both implementations isolates the
        histogram/window-search logic as the only thing under comparison."""
        def __init__(self, quantile_levels, n_estimators, min_samples_leaf, random_state):
            self.quantile_levels = quantile_levels
            self.n_estimators, self.min_samples_leaf, self.random_state = (
                n_estimators, min_samples_leaf, random_state,
            )
            self.qrf = None

        def get_quantiles(self):
            return self.quantile_levels

        def fit(self, X, Y):
            self.qrf = fit_qrf_greybox(
                X.flatten(), Y.flatten(), n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf, random_state=self.random_state,
            )
            return 0

        def predict(self, X):
            return _predicted_quantiles(self.qrf, X.flatten(), self.quantile_levels)

    def run_one_split(x, y, tr_idx, cal_idx, y_min, y_max, seed):
        x_tr, y_tr = x[tr_idx], y[tr_idx]
        x_cal, y_cal = x[cal_idx], y[cal_idx]

        # Reference CHR. Note y_steps = N_BINS + 1: the reference's y_steps
        # counts break *points*, ours counts *bins* (n_bins + 1 edges) -- get
        # this wrong and every histogram bin is silently shifted by one.
        bbox_ref = QRFBbox(DEFAULT_QUANTILE_LEVELS, N_TREES, MSL, seed)
        ref = RefCHR(bbox=bbox_ref, ymin=y_min, ymax=y_max, y_steps=N_BINS + 1,
                     delta_alpha=0.001, intervals=True, randomize=False)
        ref.fit(x_tr.reshape(-1, 1), y_tr)
        ref.calibrate(x_cal.reshape(-1, 1), y_cal, EPSILON)
        ref_bands = ref.predict(x_cal.reshape(-1, 1))
        ref_lo, ref_hi = ref_bands[:, 0], ref_bands[:, 1]

        ours = chr_fit_and_calibrate(
            x_tr, y_tr, x_cal, y_cal, EPSILON, y_min, y_max,
            n_bins=N_BINS, min_samples_leaf=MSL, n_estimators=N_TREES, random_state=seed,
        )
        our_lo, our_hi = chr_predict(ours, x_cal)
        return ref_lo, ref_hi, our_lo, our_hi, y_cal

    for dsname, loader in [("faithful", load_faithful), ("mcycle", load_mcycle)]:
        print(f"\n{'='*70}\n{dsname.upper()}\n{'='*70}")
        x, y = loader()
        n = len(x)
        y_min = y.min() - 0.05 * (y.max() - y.min())
        y_max = y.max() + 0.05 * (y.max() - y.min())

        rng = np.random.default_rng(0)
        ref_cov = np.empty(args.r); ref_wid = np.empty(args.r)
        our_cov = np.empty(args.r); our_wid = np.empty(args.r)
        all_d_lo, all_d_hi = [], []

        for r in range(args.r):
            perm = rng.permutation(n)
            tr_idx, cal_idx = perm[:n // 2], perm[n // 2:]
            ref_lo, ref_hi, our_lo, our_hi, y_cal = run_one_split(x, y, tr_idx, cal_idx, y_min, y_max, seed=r)

            all_d_lo.append(ref_lo - our_lo)
            all_d_hi.append(ref_hi - our_hi)
            ref_cov[r] = np.mean((y_cal >= ref_lo) & (y_cal <= ref_hi))
            ref_wid[r] = np.mean(ref_hi - ref_lo)
            our_cov[r] = np.mean((y_cal >= our_lo) & (y_cal <= our_hi))
            our_wid[r] = np.mean(our_hi - our_lo)
            print(f"  split {r}: ref_cov={ref_cov[r]:.3f} our_cov={our_cov[r]:.3f}  "
                  f"ref_wid={ref_wid[r]:.4f} our_wid={our_wid[r]:.4f}")

        print(f"\n  Aggregate over {args.r} splits: ref  cov={100*ref_cov.mean():.2f}%  wid={ref_wid.mean():.4f}")
        print(f"                                 ours cov={100*our_cov.mean():.2f}%  wid={our_wid.mean():.4f}")

        all_d_lo = np.concatenate(all_d_lo)
        all_d_hi = np.concatenate(all_d_hi)
        print(f"\n  Pooled per-point differences (ref - ours), n={len(all_d_lo)} points:")
        print(f"    lo: mean={all_d_lo.mean():+.4f}  sd={all_d_lo.std(ddof=1):.4f}  "
              f"mean|.|={np.abs(all_d_lo).mean():.4f}  median|.|={np.median(np.abs(all_d_lo)):.4f}")
        print(f"    hi: mean={all_d_hi.mean():+.4f}  sd={all_d_hi.std(ddof=1):.4f}  "
              f"mean|.|={np.abs(all_d_hi).mean():.4f}  median|.|={np.median(np.abs(all_d_hi)):.4f}")


if __name__ == "__main__":
    main()
