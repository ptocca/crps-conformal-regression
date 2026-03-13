"""Synthetic bimodal figure: data distribution, CRPS interval, k-NN prediction set."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

plt.rcParams.update({
    "text.usetex": False,
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")

rng = np.random.default_rng(42)
m = 50  # training points in one bin
y = np.sort(np.concatenate([rng.normal(-3, 0.5, m // 2), rng.normal(3, 0.5, m // 2)]))
epsilon = 0.10

# ── Score functions ────────────────────────────────────────────────────────────

def crps_alpha(y_train, yh):
    """CRPS nonconformity score: (1/m) sum|y_i - yh| - W/m^2."""
    m = len(y_train)
    W = np.sum(np.abs(y_train[:, None] - y_train[None, :])) / 2
    return np.mean(np.abs(y_train - yh)) - W / m**2

def knn1_alpha(y_train, yh):
    """1-NN nonconformity score: min_i |y_i - yh|."""
    return np.min(np.abs(y_train - yh))

# ── Threshold via split-conformal: use y as calibration ───────────────────────
# Compute LOO calibration scores for each y_j

def crps_loo_score(y_train, j):
    """CRPS score of y_j under leave-j-out ECDF."""
    y_loo = np.delete(y_train, j)
    m_loo = len(y_loo)
    W_loo = np.sum(np.abs(y_loo[:, None] - y_loo[None, :])) / 2
    return np.mean(np.abs(y_loo - y_train[j])) - W_loo / m_loo**2

def knn1_loo_score(y_train, j):
    """1-NN LOO score: nearest neighbor of y_j in y_train without y_j."""
    y_loo = np.delete(y_train, j)
    return np.min(np.abs(y_loo - y_train[j]))

cal_crps = np.array([crps_loo_score(y, j) for j in range(m)])
cal_knn  = np.array([knn1_loo_score(y, j) for j in range(m)])

# Conformal threshold: (1-eps)(1 + 1/m) quantile of calibration scores
level = min(np.ceil((1 - epsilon) * (m + 1)) / m, 1.0)
c_crps = np.quantile(cal_crps, level)
c_knn  = np.quantile(cal_knn,  level)

# ── Score profiles on y grid ───────────────────────────────────────────────────

y_grid = np.linspace(-6.5, 6.5, 3000)
alpha_crps = np.array([crps_alpha(y, yh) for yh in y_grid])
alpha_knn  = np.array([knn1_alpha(y, yh) for yh in y_grid])

in_crps = alpha_crps <= c_crps
in_knn  = alpha_knn  <= c_knn

# ── Figure ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

# Panel 0: data distribution
ax0 = axes[0]
kde_grid = np.linspace(-6.5, 6.5, 500)
kde_density = gaussian_kde(y, bw_method=0.3)(kde_grid)
ax0.plot(kde_grid, kde_density, color="dimgrey", lw=1.8, label="KDE")
ax0.fill_between(kde_grid, 0, kde_density, alpha=0.18, color="dimgrey")
ax0.plot(y, np.zeros(m) - 0.003 * kde_density.max(), "|",
         color="grey", ms=8, alpha=0.7, label="Training $y_i$")
ax0.set_xlabel("$y$")
ax0.set_ylabel("Density")
ax0.set_title("Training data distribution")
ax0.legend(loc="upper center", fontsize=8.5)
ax0.grid(True, alpha=0.3)

for ax, alpha, c, in_set, label, color, title in [
    (axes[1], alpha_crps, c_crps, in_crps, "CRPS score $\\alpha(y_h)$",
     "steelblue", "CRPS nonconformity score"),
    (axes[2], alpha_knn,  c_knn,  in_knn,  "1-NN score $\\alpha^{(1)}(y_h)$",
     "tomato",    "1-NN nonconformity score"),
]:
    ax.plot(y_grid, alpha, color=color, lw=1.8, label=label)
    ax.axhline(c, color="black", ls="--", lw=1.2, label=f"Threshold $c_{{\\varepsilon}}={c:.2f}$")
    # Shade prediction set
    ax.fill_between(y_grid, 0, alpha.max() * 1.05,
                    where=in_set, alpha=0.18, color=color,
                    label=f"$\\Gamma^\\varepsilon$")
    # Rug plot of training data
    ax.plot(y, np.zeros(m) - 0.003 * alpha.max(), "|",
            color="grey", ms=8, alpha=0.7, label="Training $y_i$")
    ax.set_xlabel("$y_h$")
    ax.set_ylabel("Nonconformity score")
    ax.set_title(title)
    ax.set_ylim(bottom=-0.015 * alpha.max())
    ax.legend(loc="upper center", fontsize=8.5)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    f"Bimodal bin ($m={m}$, $\\varepsilon={epsilon}$, modes at $\\pm 3$): "
    "CRPS gives one wide interval; 1-NN gives two tight intervals",
    fontsize=10, y=1.02,
)
plt.tight_layout()
fig.savefig(f"{FIGDIR}/fig_bimodal_scores.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved fig_bimodal_scores.pdf")
