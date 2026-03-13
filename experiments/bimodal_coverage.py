"""Bimodal bin: empirical coverage and set-measure comparison, CRPS vs 1-NN.

DGP: 0.5·N(-3, 0.5²) + 0.5·N(3, 0.5²),  m=50 training,  m_test=500 test,  R=500 seeds.
Reports: mean coverage ± SE  and  mean Lebesgue set-measure ± SE  for each score.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

# ── Parameters ─────────────────────────────────────────────────────────────────
M       = 50        # training observations per seed
M_TEST  = 500       # test observations per seed
R       = 500       # independent seeds
EPSILON = 0.10
Y_LO, Y_HI, N_GRID = -8.0, 8.0, 10_000

rng_master = np.random.default_rng(0)
y_grid = np.linspace(Y_LO, Y_HI, N_GRID)
DY     = (Y_HI - Y_LO) / N_GRID   # grid spacing for Lebesgue measure

# ── Vectorised score functions ──────────────────────────────────────────────────

def crps_alpha_vec(y_train, yh_vec):
    """CRPS nonconformity score for a vector of candidates yh_vec."""
    m_ = len(y_train)
    W  = np.sum(np.abs(y_train[:, None] - y_train[None, :])) / 2
    return np.mean(np.abs(y_train[:, None] - yh_vec[None, :]), axis=0) - W / m_**2

def knn1_alpha_vec(y_train, yh_vec):
    """1-NN nonconformity score for a vector of candidates yh_vec."""
    return np.min(np.abs(y_train[:, None] - yh_vec[None, :]), axis=0)

# ── LOO calibration scores (O(m²) vectorised) ──────────────────────────────────

def loo_crps_scores(y_train):
    """LOO-CRPS calibration score for each training point.

    Closed form: score_j = (sum_j * m - W_full) / (m-1)^2
    where sum_j = sum_{i≠j} |y_j - y_i|  and  W_full = Σ_{i<r} |y_i - y_r|.
    """
    m_  = len(y_train)
    D   = np.abs(y_train[:, None] - y_train[None, :])   # (m, m), diagonal = 0
    sum_j  = D.sum(axis=1)                               # (m,)
    W_full = D.sum() / 2
    return (sum_j * m_ - W_full) / (m_ - 1)**2

def loo_knn1_scores(y_train):
    """1-NN LOO calibration score for each training point."""
    D = np.abs(y_train[:, None] - y_train[None, :])     # (m, m)
    np.fill_diagonal(D, np.inf)
    return D.min(axis=1)                                 # (m,)

def conformal_threshold(cal_scores, epsilon, m):
    level = min(np.ceil((1 - epsilon) * (m + 1)) / m, 1.0)
    return np.quantile(cal_scores, level)

# ── Simulation loop ─────────────────────────────────────────────────────────────

cov_crps  = np.empty(R)
cov_knn   = np.empty(R)
meas_crps = np.empty(R)
meas_knn  = np.empty(R)

for seed in range(R):
    rng = np.random.default_rng(rng_master.integers(1 << 31))

    # Training data from bimodal DGP
    y_train = np.concatenate([rng.normal(-3, 0.5, M // 2),
                               rng.normal( 3, 0.5, M // 2)])

    # Conformal thresholds
    c_crps = conformal_threshold(loo_crps_scores(y_train), EPSILON, M)
    c_knn  = conformal_threshold(loo_knn1_scores(y_train),  EPSILON, M)

    # Lebesgue measure of prediction sets (via grid)
    meas_crps[seed] = np.sum(crps_alpha_vec(y_train, y_grid) <= c_crps) * DY
    meas_knn[seed]  = np.sum(knn1_alpha_vec(y_train, y_grid) <= c_knn)  * DY

    # Test coverage
    y_test = np.concatenate([rng.normal(-3, 0.5, M_TEST // 2),
                              rng.normal( 3, 0.5, M_TEST // 2)])
    cov_crps[seed] = np.mean(crps_alpha_vec(y_train, y_test) <= c_crps)
    cov_knn[seed]  = np.mean(knn1_alpha_vec(y_train, y_test) <= c_knn)

# ── Results ─────────────────────────────────────────────────────────────────────

def fmt_pct(arr):
    mu, se = arr.mean(), arr.std(ddof=1) / np.sqrt(R)
    return f"{100*mu:.1f} ± {100*se:.1f} pp"

def fmt_val(arr):
    mu, se = arr.mean(), arr.std(ddof=1) / np.sqrt(R)
    return f"{mu:.2f} ± {se:.2f}"

print(f"R={R}, M={M}, M_TEST={M_TEST}, epsilon={EPSILON}")
print(f"{'Score':<6}  {'Coverage':<20}  {'Set measure':<20}")
print(f"{'CRPS':<6}  {fmt_pct(cov_crps):<20}  {fmt_val(meas_crps):<20}")
print(f"{'1-NN':<6}  {fmt_pct(cov_knn):<20}  {fmt_val(meas_knn):<20}")
print()
print("Ratio of mean set measures (CRPS / 1-NN):",
      f"{meas_crps.mean() / meas_knn.mean():.2f}x")
