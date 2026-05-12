"""
validate_competitors.py
=======================
Validates competitor implementations in experiments/competitors.py against
reference implementations and analytical results.

Checks
------
1. LP quantile regression vs statsmodels.QuantReg (cross-solver comparison)
2. LP quantile regression — oracle accuracy on correctly-specified polynomial DGPs
3. CQR conformal coverage on synthetic data with known conditional quantiles
4. IDR qpred output shape and quantile ordering
5. CQR-QRF predict output shape
6. CQR-IDR conformal coverage on synthetic data

Run with:
    uv run python experiments/validate_competitors.py
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from statsmodels.regression.quantile_regression import QuantReg

from experiments.competitors import (
    _fit_idr_model,
    _predict_idr_quantiles,
    fit_cqr,
    fit_cqr_idr,
    fit_linear_quantile,
    fit_qrf,
    predict_cqr_interval,
    predict_cqr_idr_interval,
    predict_quantile,
    predict_qrf_interval,
)
from experiments.data import load_faithful, load_mcycle

LOSS_TOL = 1e-5  # our quantile loss must be <= sm loss + LOSS_TOL (absolute)


def _section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def _pf(cond):
    return "OK  " if cond else "FAIL"


# ── 1. LP quantile regression vs statsmodels.QuantReg ────────────────────────

def _pinball(y, pred, tau):
    """Pinball (check-function) loss at quantile level tau."""
    r = y - pred
    return float(np.mean(tau * np.maximum(r, 0) + (1 - tau) * np.maximum(-r, 0)))


def check_lp_vs_statsmodels(label, x, y):
    """Compare fit_linear_quantile against statsmodels.QuantReg.

    The LP optimum may not be unique (degenerate quantile regression problems
    can have whole faces of optimal solutions).  Different solvers may return
    different vertices, so we compare achieved quantile loss rather than
    predictions.  Our implementation passes if it achieves loss no worse
    than statsmodels (within LOSS_TOL).  Prediction differences are reported
    for information.

    statsmodels max_iter is raised to 5000 to reduce Barrodale-Roberts
    convergence failures on ill-conditioned polynomial designs.
    """
    _section(f"LP vs statsmodels.QuantReg — {label}")
    taus    = [0.05, 0.25, 0.50, 0.75, 0.95]
    degrees = [1, 3]
    all_ok  = True

    for degree in degrees:
        X_train = np.column_stack([x ** k for k in range(degree + 1)])
        for tau in taus:
            our_coef  = fit_linear_quantile(x, y, tau, degree=degree)
            our_pred  = predict_quantile(our_coef, x)
            our_loss  = _pinball(y, our_pred, tau)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sm_params = QuantReg(y, X_train).fit(q=tau, max_iter=5000).params
            sm_pred  = X_train @ sm_params
            sm_loss  = _pinball(y, sm_pred, tau)

            diff_pred = float(np.max(np.abs(our_pred - sm_pred)))
            # Pass: our loss is no worse than statsmodels loss
            ok = our_loss <= sm_loss + LOSS_TOL
            if not ok:
                all_ok = False
            print(
                f"  deg={degree}  tau={tau:.2f}  "
                f"loss_ours={our_loss:.6f}  loss_sm={sm_loss:.6f}  "
                f"max|Δpred|={diff_pred:.2e}  [{_pf(ok)}]"
            )
    return all_ok


# ── 2. LP quantile regression — oracle accuracy ───────────────────────────────

def check_lp_oracle(n=5000, seed=99):
    """Verify that fit_linear_quantile recovers known oracle quantiles at large n.

    This is the width-level check missing from the statsmodels comparison: we
    fit the LP on data whose true conditional quantile is exactly a polynomial,
    then measure sup|ŷ(x) - q_true(x)| over a test grid.  A correct LP
    implementation must converge to the oracle; a broken one will be off by
    O(1) even at n=5000.

    Both DGPs use constant residual variance σ=1 so that the asymptotic SE of
    the quantile estimator is the same at every x.  For correctly-specified
    linear quantile regression the asymptotic SD at a test point x is
      SD ≈ sqrt(τ(1−τ)) / φ(Φ⁻¹(τ)) · sqrt(x'(X'X)⁻¹x)
    which, for X~U(0,3) degree-1 and x at the boundary (x=3), is ≈0.06 at
    τ=0.95, giving 3·SD ≈ 0.18.  The tolerance of 0.25 covers 4+ SEs and is
    tight enough that any O(1) bug in the LP formulation will be caught.

    Degree-1 DGP:  Y | X ~ N(3x, 1),  X ~ U(0, 3).
      True quantile: q(τ|x) = 3x + Φ⁻¹(τ)  [linear in x, correctly specified].

    Degree-3 DGP:  Y | X ~ N(1 + 2x − x² + 0.5x³,  1),  X ~ U(0, 1).
      True quantile: q(τ|x) = 1 + 2x − x² + 0.5x³ + Φ⁻¹(τ)  [cubic, correctly specified].
      X restricted to [0, 1] to avoid the ill-conditioned design matrix that
      causes statsmodels to fail on raw Old Faithful/Motorcycle scales.
    """
    from scipy.stats import norm

    _section(f"LP oracle accuracy — correctly-specified polynomial DGPs (n={n})")
    rng    = np.random.default_rng(seed)
    taus   = [0.05, 0.25, 0.50, 0.75, 0.95]
    TOL    = 0.25   # covers 4+ asymptotic SEs; any O(1) LP bug will exceed this
    all_ok = True

    # ── degree=1 ──────────────────────────────────────────────────────────────
    x_tr   = rng.uniform(0, 3, size=n)
    y_tr   = rng.normal(3 * x_tr, 1.0)
    x_grid = np.linspace(0, 3, 200)
    print("  Degree=1  [DGP: Y|X ~ N(3x, 1), X ~ U(0,3)]")
    for tau in taus:
        oracle = 3 * x_grid + norm.ppf(tau)
        coef   = fit_linear_quantile(x_tr, y_tr, tau, degree=1)
        pred   = predict_quantile(coef, x_grid)
        err    = float(np.max(np.abs(pred - oracle)))
        ok     = err < TOL
        if not ok:
            all_ok = False
        print(f"    tau={tau:.2f}  sup|ŷ − oracle|={err:.4f}  [{_pf(ok)}]")

    # ── degree=3 ──────────────────────────────────────────────────────────────
    x_tr3    = rng.uniform(0, 1, size=n)
    mu_tr3   = 1 + 2*x_tr3 - x_tr3**2 + 0.5*x_tr3**3
    y_tr3    = rng.normal(mu_tr3, 1.0)
    x_grid3  = np.linspace(0, 1, 200)
    mu_grid3 = 1 + 2*x_grid3 - x_grid3**2 + 0.5*x_grid3**3
    print("  Degree=3  [DGP: Y|X ~ N(1+2x-x²+0.5x³, 1), X ~ U(0,1)]")
    for tau in taus:
        oracle = mu_grid3 + norm.ppf(tau)
        coef   = fit_linear_quantile(x_tr3, y_tr3, tau, degree=3)
        pred   = predict_quantile(coef, x_grid3)
        err    = float(np.max(np.abs(pred - oracle)))
        ok     = err < TOL
        if not ok:
            all_ok = False
        print(f"    tau={tau:.2f}  sup|ŷ − oracle|={err:.4f}  [{_pf(ok)}]")

    return all_ok


# ── 3. CQR conformal coverage on synthetic data ───────────────────────────────

def check_cqr_coverage(n_trials=500, n=400, epsilon=0.10, seed=0):
    """Synthetic: Y | X ~ N(3X, (1+X)^2), X ~ U(0, 3).

    Linear quantiles q(tau | x) = 3x + (1+x) * Phi^{-1}(tau) are correctly
    specified by CQR with degree=1.  The conformal guarantee requires mean
    coverage >= 1 - epsilon.
    """
    _section(
        f"CQR (degree=1) conformal coverage — synthetic "
        f"(n={n}, epsilon={epsilon}, R={n_trials})"
    )
    rng = np.random.default_rng(seed)
    coverages = []
    for _ in range(n_trials):
        x    = rng.uniform(0, 3, size=n)
        y    = rng.normal(3 * x, 1 + x)
        half = n // 2
        perm = rng.permutation(n)
        tr, cal = perm[:half], perm[half:]

        model  = fit_cqr(x[tr], y[tr], x[cal], y[cal], epsilon, degree=1)
        lo, hi = predict_cqr_interval(model, x[cal])
        coverages.append(float(np.mean((y[cal] >= lo) & (y[cal] <= hi))))

    mu = float(np.mean(coverages))
    se = float(np.std(coverages, ddof=1) / np.sqrt(n_trials))
    ok = mu >= (1 - epsilon) - 3 * se
    print(f"  nominal={1-epsilon:.2f}  observed={mu:.4f} ± {se:.4f}  [{_pf(ok)}]")
    return ok


# ── 3. IDR qpred output shape and quantile ordering ──────────────────────────

def check_idr_shape():
    """Verify _predict_idr_quantiles returns 1-D arrays of the right length
    and that the lower quantile does not exceed the upper quantile."""
    _section("IDR qpred output shape and ordering")
    rng    = np.random.default_rng(42)
    n      = 120
    x      = np.sort(rng.uniform(0, 3, size=n))
    y      = rng.normal(3 * x, 1 + x)
    x_test = np.linspace(0, 3, 15)

    fit         = _fit_idr_model(x, y)
    q_lo, q_hi  = _predict_idr_quantiles(fit, x_test, epsilon=0.10)

    ok_lo_1d  = q_lo.ndim == 1
    ok_hi_1d  = q_hi.ndim == 1
    ok_lo_len = len(q_lo) == len(x_test)
    ok_hi_len = len(q_hi) == len(x_test)
    ok_order  = bool(np.all(q_lo <= q_hi))

    print(f"  q_lo.shape = {q_lo.shape}   expected ({len(x_test)},)  [{_pf(ok_lo_1d and ok_lo_len)}]")
    print(f"  q_hi.shape = {q_hi.shape}   expected ({len(x_test)},)  [{_pf(ok_hi_1d and ok_hi_len)}]")
    print(f"  q_lo <= q_hi everywhere  [{_pf(ok_order)}]")
    return ok_lo_1d and ok_hi_1d and ok_lo_len and ok_hi_len and ok_order


# ── 4. CQR-QRF predict output shape ──────────────────────────────────────────

def check_qrf_shape():
    """Verify predict_qrf_interval returns 1-D arrays of the right length
    and that lower <= upper quantile everywhere."""
    _section("CQR-QRF predict output shape")
    rng    = np.random.default_rng(7)
    n      = 100
    x      = rng.uniform(0, 3, size=n)
    y      = rng.normal(3 * x, 1 + x)
    x_test = np.linspace(0, 3, 20)

    qrf    = fit_qrf(x, y, n_estimators=50, random_state=7)
    lo, hi = predict_qrf_interval(qrf, x_test, epsilon=0.10)

    ok_lo_1d  = lo.ndim == 1
    ok_hi_1d  = hi.ndim == 1
    ok_lo_len = len(lo) == len(x_test)
    ok_hi_len = len(hi) == len(x_test)
    ok_order  = bool(np.all(lo <= hi))

    print(f"  lo.shape = {lo.shape}   expected ({len(x_test)},)  [{_pf(ok_lo_1d and ok_lo_len)}]")
    print(f"  hi.shape = {hi.shape}   expected ({len(x_test)},)  [{_pf(ok_hi_1d and ok_hi_len)}]")
    print(f"  lo <= hi everywhere  [{_pf(ok_order)}]")
    return ok_lo_1d and ok_hi_1d and ok_lo_len and ok_hi_len and ok_order


# ── 5. CQR-IDR conformal coverage on synthetic data ──────────────────────────

def check_cqr_idr_coverage(n_trials=100, n=400, epsilon=0.10, seed=1):
    """Same synthetic setting as check_cqr_coverage.

    Fewer trials because IDR fitting is slower.  If the IDR shape check
    (section 3) reveals a broadcasting bug, this test will also fail.
    """
    _section(
        f"CQR-IDR conformal coverage — synthetic "
        f"(n={n}, epsilon={epsilon}, R={n_trials})"
    )
    rng = np.random.default_rng(seed)
    coverages = []
    for _ in range(n_trials):
        x    = rng.uniform(0, 3, size=n)
        y    = rng.normal(3 * x, 1 + x)
        half = n // 2
        perm = rng.permutation(n)
        tr, cal = perm[:half], perm[half:]

        # Sort training set by x (mirrors usage in fair_comparison.py)
        sort_idx = np.argsort(x[tr])
        x_tr = x[tr][sort_idx]
        y_tr = y[tr][sort_idx]

        model  = fit_cqr_idr(x_tr, y_tr, x[cal], y[cal], epsilon)
        lo, hi = predict_cqr_idr_interval(model, x[cal])
        coverages.append(float(np.mean((y[cal] >= lo) & (y[cal] <= hi))))

    mu = float(np.mean(coverages))
    se = float(np.std(coverages, ddof=1) / np.sqrt(n_trials))
    ok = mu >= (1 - epsilon) - 3 * se
    print(f"  nominal={1-epsilon:.2f}  observed={mu:.4f} ± {se:.4f}  [{_pf(ok)}]")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading datasets...")
    x_faith, y_faith = load_faithful()
    x_cycle, y_cycle = load_mcycle()

    checks = [
        (
            "LP vs statsmodels (faithful)",
            lambda: check_lp_vs_statsmodels("Old Faithful", x_faith, y_faith),
        ),
        (
            "LP vs statsmodels (mcycle)",
            lambda: check_lp_vs_statsmodels("Motorcycle", x_cycle, y_cycle),
        ),
        ("LP oracle accuracy", check_lp_oracle),
        ("CQR coverage",       check_cqr_coverage),
        ("IDR shape",          check_idr_shape),
        ("QRF shape",          check_qrf_shape),
        ("CQR-IDR coverage",   check_cqr_idr_coverage),
    ]

    results = {}
    for name, fn in checks:
        results[name] = fn()

    _section("SUMMARY")
    all_passed = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}")
        all_passed = all_passed and ok
    print()
    sys.exit(0 if all_passed else 1)
