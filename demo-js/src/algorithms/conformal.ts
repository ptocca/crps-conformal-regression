import { pairwiseAbsSum } from './binning';

/**
 * Vectorized conformal p-values for a grid of hypothetical responses.
 *
 * For each y_h in yhGrid:
 *   α(y_h) = CRPS(y_h; yBin)
 * p(y_h) = (1 + #{j: α(y_j) >= α(y_h)}) / (m+1)
 */
export function conformalPvalueGrid(
  yBin: number[],
  yhGrid: number[]
): Float64Array {
  const m = yBin.length;
  const ng = yhGrid.length;
  const result = new Float64Array(ng);

  if (m === 0) {
    result.fill(NaN);
    return result;
  }

  const W_m = pairwiseAbsSum(yBin);

  // alpha_h[g] = CRPS of y_h[g] w.r.t. yBin
  const alphaH = new Float64Array(ng);
  for (let g = 0; g < ng; g++) {
    let sumAbs = 0;
    for (let i = 0; i < m; i++) sumAbs += Math.abs(yBin[i] - yhGrid[g]);
    alphaH[g] = sumAbs / m - W_m / (m * m);
  }

  // For each training point j, compute alpha_j[g] = CRPS when y_j is replaced by y_h[g]
  // alpha_j = (d_j + |y_j - y_h|) / m - (W_m - d_j) / m^2 - A[j,g] / m^2
  // where d_j = Σ_i |y_i - y_j| (row sums of abs diff matrix among yBin),
  // A[j,g] = sumAbs_g - |y_j - y_h[g]|  (sum of |y_i - y_h| for i≠j)

  // Precompute d_j for each j
  const d = new Float64Array(m);
  for (let j = 0; j < m; j++) {
    let dj = 0;
    for (let i = 0; i < m; i++) dj += Math.abs(yBin[i] - yBin[j]);
    d[j] = dj;
  }

  // For each grid point, count how many alpha_j >= alpha_h
  // We compute per grid point to keep memory O(m + ng)
  const sumAbsG = new Float64Array(ng);
  for (let g = 0; g < ng; g++) {
    let s = 0;
    for (let i = 0; i < m; i++) s += Math.abs(yBin[i] - yhGrid[g]);
    sumAbsG[g] = s;
  }

  for (let g = 0; g < ng; g++) {
    const aH = alphaH[g];
    let count = 1; // +1 for y_h itself
    const yh = yhGrid[g];
    for (let j = 0; j < m; j++) {
      const absJG = Math.abs(yBin[j] - yh);
      const A = sumAbsG[g] - absJG;
      const alphaJ = (d[j] + absJG) / m - (W_m - d[j]) / (m * m) - A / (m * m);
      if (alphaJ >= aH) count++;
    }
    result[g] = count / (m + 1);
  }

  return result;
}

/**
 * Conformal prediction interval for a single bin via grid search.
 * Returns [lo, hi] such that p(y_h) > epsilon for y_h in [lo, hi].
 */
export function conformalInterval(
  yBin: number[],
  epsilon: number,
  nGrid = 2000
): [number, number] {
  if (yBin.length === 0) return [NaN, NaN];

  const sorted = yBin.slice().sort((a, b) => a - b);
  const mn = sorted[0];
  const mx = sorted[sorted.length - 1];
  let std = 0;
  const mean = yBin.reduce((s, v) => s + v, 0) / yBin.length;
  for (const v of yBin) std += (v - mean) * (v - mean);
  std = yBin.length > 1 ? Math.sqrt(std / (yBin.length - 1)) : 1;

  const lo = mn - 4 * std;
  const hi = mx + 4 * std;

  const grid: number[] = new Array(nGrid);
  for (let i = 0; i < nGrid; i++) grid[i] = lo + (hi - lo) * (i / (nGrid - 1));

  const pvals = conformalPvalueGrid(yBin, grid);

  let first = -1, last = -1;
  for (let i = 0; i < nGrid; i++) {
    if (pvals[i] > epsilon) {
      if (first === -1) first = i;
      last = i;
    }
  }

  if (first === -1) return [NaN, NaN];
  return [grid[first], grid[last]];
}
