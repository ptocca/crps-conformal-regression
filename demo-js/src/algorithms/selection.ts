import { pairwiseAbsSum, precomputeCosts, optimalPartition, binXBoundaries } from './binning';

/**
 * Empirical CRPS of y_support at y_obs.
 * = (1/m) Σ|y_i - y_obs| - W / m²
 */
export function crpsEmpirical(ySupport: number[], yObs: number): number {
  const m = ySupport.length;
  if (m === 0) return NaN;
  let sumAbs = 0;
  for (let i = 0; i < m; i++) sumAbs += Math.abs(ySupport[i] - yObs);
  const W = pairwiseAbsSum(ySupport);
  return sumAbs / m - W / (m * m);
}

export interface CVResult {
  Kstar: number;
  /** LOO total cost per K (length Kmax) — within-sample, gameable */
  looPerK: number[];
  /** Average test CRPS per K (length Kmax) — cross-validated */
  testCrpsPerK: number[];
}

/**
 * Select K by 50/50 alternating CV split.
 * Even indices → training, odd indices → test.
 */
export function selectKCV(
  x: number[],
  y: number[],
  Kmax: number
): CVResult {
  const n = x.length;
  const trIdx: number[] = [];
  const teIdx: number[] = [];
  for (let i = 0; i < n; i++) {
    (i % 2 === 0 ? trIdx : teIdx).push(i);
  }
  const xTr = trIdx.map(i => x[i]);
  const yTr = trIdx.map(i => y[i]);
  const xTe = teIdx.map(i => x[i]);
  const yTe = teIdx.map(i => y[i]);
  const nTr = yTr.length;

  const C_tr = precomputeCosts(yTr);

  const testCrpsPerK: number[] = new Array(Kmax).fill(Infinity);
  const looPerK: number[] = new Array(Kmax).fill(Infinity);

  for (let K = 1; K <= Kmax; K++) {
    if (K > Math.floor(nTr / 2)) break;

    const bp = optimalPartition(nTr, K, C_tr);
    const edges = binXBoundaries(xTr, bp);
    const Kbins = bp.length - 1;

    // LOO: total DP cost (within-sample)
    let dpCost = 0;
    for (let k = 0; k < Kbins; k++) {
      const m = bp[k + 1] - bp[k];
      if (m < 2) { dpCost = Infinity; break; }
      let W = 0;
      const yBinSlice = yTr.slice(bp[k], bp[k + 1]);
      const sorted = yBinSlice.slice().sort((a, b) => a - b);
      for (let idx = 0; idx < m; idx++) W += sorted[idx] * (2 * idx - (m - 1));
      dpCost += (m * W) / ((m - 1) * (m - 1));
    }
    looPerK[K - 1] = dpCost;

    // Test CRPS
    let total = 0;
    for (let qi = 0; qi < xTe.length; qi++) {
      const xq = xTe[qi];
      const yq = yTe[qi];
      // searchsorted right
      let idx = upperBound(edges, xq) - 1;
      idx = Math.max(0, Math.min(Kbins - 1, idx));
      const yBin = yTr.slice(bp[idx], bp[idx + 1]);
      total += crpsEmpirical(yBin, yq);
    }
    testCrpsPerK[K - 1] = total / xTe.length;
  }

  // Find Kstar = argmin over finite entries
  let Kstar = 1;
  let best = Infinity;
  for (let K = 1; K <= Kmax; K++) {
    if (testCrpsPerK[K - 1] < best) {
      best = testCrpsPerK[K - 1];
      Kstar = K;
    }
  }

  return { Kstar, looPerK, testCrpsPerK };
}

/** searchsorted right: first index where edges[idx] > val */
function upperBound(arr: ArrayLike<number>, val: number): number {
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] <= val) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
