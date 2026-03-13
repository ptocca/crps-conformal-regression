"""Competitor prediction-interval methods for comparison experiments."""

import numpy as np
from scipy.optimize import linprog
from quantile_forest import RandomForestQuantileRegressor


# ── Gaussian split-conformal ──────────────────────────────────────────────────

def fit_gaussian_split_conformal(
    x_tr: np.ndarray, y_tr: np.ndarray,
    x_cal: np.ndarray, y_cal: np.ndarray,
) -> dict:
    """OLS on (x_tr, y_tr); calibrate absolute residuals on (x_cal, y_cal)."""
    X_tr = np.column_stack([np.ones(len(x_tr)), x_tr])
    beta, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
    X_cal = np.column_stack([np.ones(len(x_cal)), x_cal])
    resid = np.abs(y_cal - X_cal @ beta)
    n_cal = len(y_cal)
    # standard split-conformal quantile: ceil((1-eps)(n+1))/n
    return {"beta": beta, "resid_cal": resid}


def predict_gaussian_interval(
    model: dict, x_test: np.ndarray, epsilon: float
) -> tuple[np.ndarray, np.ndarray]:
    n_cal = len(model["resid_cal"])
    level = np.ceil((1 - epsilon) * (n_cal + 1)) / n_cal
    level = min(level, 1.0)
    q = np.quantile(model["resid_cal"], level)
    X_test = np.column_stack([np.ones(len(x_test)), x_test])
    yhat = X_test @ model["beta"]
    return yhat - q, yhat + q


# ── Linear quantile regression (via LP) ──────────────────────────────────────

def fit_linear_quantile(
    x: np.ndarray, y: np.ndarray, tau: float, degree: int = 1
) -> np.ndarray:
    """Fit quantile regression at level tau on polynomial features of given degree.

    Uses the standard LP formulation with non-negative slack variables.
    Returns coefficient vector of length (degree+1).
    """
    n = len(y)
    d = degree + 1  # number of coefficients

    # Design matrix
    X = np.column_stack([x ** k for k in range(d)])

    # Variables: [beta_pos(d), beta_neg(d), u(n), v(n)]
    # y_i = X[i] @ (beta_pos - beta_neg) + u_i - v_i
    # min: tau*sum(u) + (1-tau)*sum(v)
    # s.t. X @ (beta_pos - beta_neg) + u - v = y, all >= 0

    c = np.concatenate([
        np.zeros(d), np.zeros(d),       # beta_pos, beta_neg (free in objective)
        tau * np.ones(n),                # u
        (1 - tau) * np.ones(n),          # v
    ])

    # Equality constraint: [X, -X, I, -I] @ z = y
    A_eq = np.hstack([X, -X, np.eye(n), -np.eye(n)])
    b_eq = y

    bounds = (
        [(0, None)] * d +   # beta_pos >= 0
        [(0, None)] * d +   # beta_neg >= 0
        [(0, None)] * n +   # u >= 0
        [(0, None)] * n     # v >= 0
    )

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    beta_pos = res.x[:d]
    beta_neg = res.x[d:2*d]
    return beta_pos - beta_neg


def predict_quantile(coef: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    degree = len(coef) - 1
    X = np.column_stack([x_test ** k for k in range(degree + 1)])
    return X @ coef


def fit_cqr(
    x_tr: np.ndarray, y_tr: np.ndarray,
    x_cal: np.ndarray, y_cal: np.ndarray,
    epsilon: float, degree: int = 1,
) -> dict:
    """Fit CQR: conformalized quantile regression.

    Fits lower (eps/2) and upper (1-eps/2) quantile regressors, calibrates
    the conformity score max(q_lo - y, y - q_hi) on the calibration set.
    """
    alpha_lo, alpha_hi = epsilon / 2, 1 - epsilon / 2
    coef_lo = fit_linear_quantile(x_tr, y_tr, alpha_lo, degree=degree)
    coef_hi = fit_linear_quantile(x_tr, y_tr, alpha_hi, degree=degree)
    q_lo_cal = predict_quantile(coef_lo, x_cal)
    q_hi_cal = predict_quantile(coef_hi, x_cal)
    scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    n_cal = len(y_cal)
    level = np.ceil((1 - epsilon) * (n_cal + 1)) / n_cal
    level = min(level, 1.0)
    q_hat = np.quantile(scores, level)
    return {"coef_lo": coef_lo, "coef_hi": coef_hi, "q_hat": q_hat, "degree": degree}


def predict_cqr_interval(
    model: dict, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    q_lo = predict_quantile(model["coef_lo"], x_test)
    q_hi = predict_quantile(model["coef_hi"], x_test)
    q = model["q_hat"]
    return q_lo - q, q_hi + q


# ── Quantile Regression Forest ────────────────────────────────────────────────

def fit_qrf(
    x_tr: np.ndarray, y_tr: np.ndarray,
    n_estimators: int = 500, random_state: int = 42,
) -> RandomForestQuantileRegressor:
    model = RandomForestQuantileRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        min_samples_leaf=5,
    )
    model.fit(x_tr.reshape(-1, 1), y_tr)
    return model


def predict_qrf_interval(
    model: RandomForestQuantileRegressor,
    x_test: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    lo = model.predict(x_test.reshape(-1, 1), quantiles=[epsilon / 2])
    hi = model.predict(x_test.reshape(-1, 1), quantiles=[1 - epsilon / 2])
    return lo.ravel(), hi.ravel()


# ── Conformalized QRF (CQR-QRF) ──────────────────────────────────────────────

def fit_cqr_qrf(
    x_tr: np.ndarray, y_tr: np.ndarray,
    x_cal: np.ndarray, y_cal: np.ndarray,
    epsilon: float,
    n_estimators: int = 500, random_state: int = 42,
) -> dict:
    """Conformalized QRF: fit QRF on training half, calibrate CQR score on cal half."""
    qrf = fit_qrf(x_tr, y_tr, n_estimators=n_estimators, random_state=random_state)
    alpha_lo, alpha_hi = epsilon / 2, 1 - epsilon / 2
    q_lo_cal = qrf.predict(x_cal.reshape(-1, 1), quantiles=[alpha_lo]).ravel()
    q_hi_cal = qrf.predict(x_cal.reshape(-1, 1), quantiles=[alpha_hi]).ravel()
    scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    n_cal = len(y_cal)
    level = np.ceil((1 - epsilon) * (n_cal + 1)) / n_cal
    level = min(level, 1.0)
    q_hat = np.quantile(scores, level)
    return {"qrf": qrf, "q_hat": q_hat, "epsilon": epsilon}


def predict_cqr_qrf_interval(
    model: dict, x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    epsilon = model["epsilon"]
    alpha_lo, alpha_hi = epsilon / 2, 1 - epsilon / 2
    q_lo = model["qrf"].predict(x_test.reshape(-1, 1), quantiles=[alpha_lo]).ravel()
    q_hi = model["qrf"].predict(x_test.reshape(-1, 1), quantiles=[alpha_hi]).ravel()
    q = model["q_hat"]
    return q_lo - q, q_hi + q
