# CRPS-Optimal Conformal Regression

Python implementation of the method described in:

> [Author(s)]. *CRPS-Optimal Binning for Conformal Regression*. [arXiv preprint, 2026.]

Given training data $(x_i, y_i)$, we sort by $x$, partition the observations into $K$ contiguous bins, and use the within-bin empirical CDF as the predictive distribution. The three main contributions are:

1. **Optimal binning**: bin boundaries are chosen to minimise the total leave-one-out CRPS via dynamic programming in $O(n^2 K)$ time.
2. **Cross-validated K selection**: the number of bins $K$ is selected by test CRPS on a held-out split; within-sample LOO-CRPS is gameable and must not be used for model selection.
3. **Conformal prediction**: the bin ECDF is wrapped in a full conformal predictor (CRPS as nonconformity score) to obtain finite-sample coverage guarantees.

## Installation

```bash
pip install crpsconfreg
```

Or from source:

```bash
git clone https://github.com/ptocca/crps-conformal-regression.git
cd crps-conformal-regression
pip install -e .
```

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

## Repository structure

| Path | Description |
|---|---|
| `src/crpsconfreg/` | Python package |
| `src/crpsconfreg/binning.py` | LOO-CRPS cost, DP, bin boundaries |
| `src/crpsconfreg/selection.py` | CRPS scoring, CV K selection |
| `src/crpsconfreg/conformal.py` | Conformal p-values (vectorised), prediction intervals |
| `src/crpsconfreg/predictor.py` | `BinningPredictor` high-level class |
| `tests/` | 61 pytest tests |
| `experiments/` | Scripts reproducing all numerical results in the paper |
| `save_figures.py` | Reproduces all figures in `figures/` as PDFs |
| `figures/` | Pre-generated PDF figures |
| `demo-js/` | TypeScript + Vite + Plotly.js interactive demo (instant load, no WASM) |
| `docs/` | Generated static site served by GitHub Pages |

## Key results

- **LOO-CRPS cost of a bin** $S$ with $m \ge 2$ observations: $\mathrm{cost}(S) = \frac{m}{(m-1)^2} \sum_{\ell < r} |y_\ell - y_r|$
- **Empirical coverage** at $\varepsilon=0.10$: 90.3% on Old Faithful (target $\ge$ 90%), 91.0% on Motorcycle
- **Interval width**: 11–40% narrower than Gaussian conformal and CQR baselines on the two real datasets

## Live demo

A static web demo runs entirely in the browser with no server and no Python runtime.
It is built with TypeScript, Vite, and Plotly.js (`demo-js/`), giving instant page
loads. Four DGP presets are included; you can also write a custom Data Generating
Process in the editor. All plots from the paper are reproduced interactively.

**[Live demo](https://ptocca.github.io/crps-conformal-regression/)**

To rebuild `docs/` after editing the demo source:

```bash
cd demo-js
npm install      # first time only
npm run build    # outputs to ../docs/
```

## Running the tests

```bash
uv run pytest
# or
pip install pytest && pytest
```

## Reproducing paper results

```bash
uv run python -m experiments.fair_comparison   # Table 1–2: coverage and width
uv run python save_figures.py                  # All figures in figures/
```
