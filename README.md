# Optimal Binning for Regression via LOO-CRPS

Research project extending the Venn-ABERS calibrated prediction framework to regression (continuous response, real-valued covariate).

## Overview

Given training data $(x_i, y_i)$, we sort by $x$, partition the observations into $K$ contiguous bins, and use the within-bin empirical CDF as the predictive distribution. The three main contributions are:

1. **Optimal binning**: bin boundaries are chosen to minimise the total leave-one-out CRPS via dynamic programming in $O(n^2 K)$ time.
2. **Cross-validated K selection**: the number of bins $K$ is selected by test CRPS on a held-out split; within-sample LOO-CRPS is gameable and must not be used for model selection.
3. **Conformal prediction**: the bin ECDF is wrapped in a full conformal predictor (CRPS as nonconformity score) to obtain finite-sample coverage guarantees.

The Venn prediction band — the family of augmented ECDFs as the hypothetical label varies — is also formalised as the direct regression analog of the Venn-ABERS interval.

## Package usage

```python
import numpy as np
from crpsconfreg import BinningPredictor

# x and y must be sorted by x
pred = BinningPredictor().fit(x_train, y_train, K_max=20)  # CV selects K*

# Conformal prediction interval with coverage >= 1 - epsilon
lo, hi = pred.predict_interval(x_new, epsilon=0.10)

# Conformal p-value for an observed (x*, y*) pair
pvals = pred.conformal_pvalue(x_new, y_new)

# Within-bin empirical CDF
cdf = pred.predict_ecdf(x_new, t_grid)

print(f"K* = {pred.K_}, bin sizes = {pred.bin_sizes()}")
```

Pass `K=k` to `fit()` to fix the number of bins and bypass cross-validation.

## Installation

```bash
uv sync
# or
pip install -e .
```

## Repository structure

| Path | Description |
|---|---|
| `src/crpsconfreg/` | Python package |
| `src/crpsconfreg/binning.py` | LOO-CRPS cost, DP, bin boundaries |
| `src/crpsconfreg/selection.py` | CRPS scoring, CV K selection |
| `src/crpsconfreg/conformal.py` | Conformal p-values (vectorised), prediction intervals |
| `src/crpsconfreg/predictor.py` | `BinningPredictor` high-level class |
| `tests/` | 61 pytest tests |
| `dp_formulation.tex` / `.pdf` | Full write-up: derivations, CV K selection, Venn band, conformal prediction, numerical illustration |
| `conformal_binning.ipynb` | Clean notebook: CV K selection, Venn band, p-value curves, fan plot, coverage check |
| `optimal_binning.ipynb` | Original exploratory notebook (includes Bayesian regularisation exploration) |
| `save_figures.py` | Reproduces all figures in `figures/` as PDFs |
| `showcase.ipynb` | Interactive notebook: edit one config cell to plug in any DGP |
| `demo/app.py` | Panel + Pyodide static web demo (Python in the browser) |
| `demo-js/` | TypeScript + Vite + Plotly.js web demo (instant load, no WASM) |
| `docs/` | Generated static site served by GitHub Pages |

## Key results

- **LOO-CRPS cost of a bin** $S$ with $m \ge 2$ observations: $\mathrm{cost}(S) = \frac{m}{(m-1)^2} \sum_{\ell < r} |y_\ell - y_r|$
- **CV selects** $K^*=3$ on the heteroscedastic example ($n=200$, $Y|X=x \sim \mathcal{N}(x,(1+x)^2)$), with bin sizes 51/60/89
- **Empirical coverage** at $\varepsilon=0.10$: 91.0% (target $\ge$ 90%) on a 2000-point test set

## Interactive showcase

`showcase.ipynb` lets you plug in any data-generating process and immediately see
the full pipeline — K selection, partition, predictive CDFs, conformal p-value curves,
prediction bands, and empirical coverage. Edit only the configuration cell at the top:

```python
def ygiven_x(x, rng):          # sample Y | X = x
    return rng.normal(loc=x, scale=0.5 + x)

def true_quantile(x, p):       # oracle p-quantile (or set to None)
    from scipy.stats import norm
    return norm.ppf(p, loc=x, scale=0.5 + x)

n_train, x_lo, x_hi = 300, 0.0, 3.0
K_max, epsilon, seed = 20, 0.10, 42
```

Three alternatives are included as comments: skewed (gamma), bimodal mixture, and sinusoidal mean.

## Live demo

A static web demo runs entirely in the browser with no server and no Python runtime.
It is built with TypeScript, Vite, and Plotly.js (`demo-js/`), giving instant page
loads. Four DGP presets are included; you can also write a custom Data Generating
Process in the editor. All plots from the notebook are reproduced.

**Enable GitHub Pages** (Settings → Pages → Source: `docs/` on `js-demo`) to host it
at `https://ptocca.github.io/RegressionVenn/`.

To rebuild `docs/` after editing the demo source:

```bash
cd demo-js
npm install      # first time only
npm run build    # outputs to ../docs/
```

The earlier Panel + Pyodide demo (`demo/app.py`, branch `master`) is still available
and can be rebuilt with `bash demo/build.sh`.

## Running the tests

```bash
uv run pytest
```
