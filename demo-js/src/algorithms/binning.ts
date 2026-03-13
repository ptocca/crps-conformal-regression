/**
 * Pairwise absolute sum W(y) = Σ_{i<j} |y_i - y_j| * 2
 * = Σ_k y_sort[k] * (2k - (m-1))
 * O(m log m)
 */
export function pairwiseAbsSum(y: ArrayLike<number>): number {
  const m = y.length;
  if (m < 2) return 0;
  const s = Array.from(y).sort((a, b) => a - b);
  let w = 0;
  for (let k = 0; k < m; k++) {
    w += s[k] * (2 * k - (m - 1));
  }
  return w;
}

function binCost(W: number, m: number): number {
  if (m < 2) return Infinity;
  return (m * W) / ((m - 1) * (m - 1));
}

/**
 * Precompute cost matrix C[i,j] = cost of bin containing y[i..j] (inclusive).
 * Maintains an incrementally sorted array per row.
 * Returns a flat Float64Array of length n*n (row-major).
 * O(n² log n) time.
 */
export function precomputeCosts(y: number[]): Float64Array {
  const n = y.length;
  const C = new Float64Array(n * n).fill(Infinity);

  for (let i = 0; i < n; i++) {
    // sorted_vals and prefix sums maintained incrementally
    const sorted: number[] = [];
    const prefix: number[] = [0]; // prefix[k] = sum of sorted[0..k-1]
    let W = 0;

    for (let j = i; j < n; j++) {
      const val = y[j];
      // binary search for insertion rank
      let lo = 0, hi = sorted.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (sorted[mid] < val) lo = mid + 1;
        else hi = mid;
      }
      const rank = lo;
      const mCur = sorted.length;
      const S_le = prefix[rank];
      const S_gt = prefix[mCur] - S_le;
      W += val * rank - S_le + S_gt - val * (mCur - rank);

      // insert into sorted array and prefix sums
      sorted.splice(rank, 0, val);
      // rebuild prefix from rank+1 onward (insert new prefix entry)
      prefix.splice(rank + 1, 0, S_le + val);
      for (let k = rank + 2; k < prefix.length; k++) {
        prefix[k] += val;
      }

      C[i * n + j] = binCost(W, j - i + 1);
    }
  }
  return C;
}

/**
 * Optimal K-bin partition of y[0..n-1] using precomputed cost matrix C.
 * Returns breakpoints array of length K+1 with bp[0]=0, bp[K]=n.
 * O(n²K) time.
 * Enforces minimum bin size of 2.
 */
export function optimalPartition(n: number, K: number, C: Float64Array): number[] {
  // dp[k][j] = min cost for k bins over y[0..j-1]
  // We use 1D rolling approach but need full table for backtracking
  const dp = new Float64Array((K + 1) * (n + 1)).fill(Infinity);
  const split = new Int32Array((K + 1) * (n + 1)).fill(-1);
  dp[0] = 0; // dp[0][0] = 0

  for (let k = 1; k <= K; k++) {
    for (let j = 2 * k; j <= n; j++) {
      for (let i = 2 * (k - 1); i <= j - 2; i++) {
        const val = dp[(k - 1) * (n + 1) + i] + C[i * n + (j - 1)];
        if (val < dp[k * (n + 1) + j]) {
          dp[k * (n + 1) + j] = val;
          split[k * (n + 1) + j] = i;
        }
      }
    }
  }

  // backtrack
  const bps: number[] = [n];
  let j = n;
  for (let k = K; k > 0; k--) {
    const i = split[k * (n + 1) + j];
    bps.push(i);
    j = i;
  }
  bps.reverse();
  return bps;
}

/**
 * Compute x-axis bin edges (midpoints between adjacent bins' extremes).
 * Returns Float64Array of length K+1 with edges[0]=-Inf, edges[K]=+Inf.
 */
export function binXBoundaries(x: number[], bp: number[]): Float64Array {
  const K = bp.length - 1;
  const edges = new Float64Array(K + 1);
  edges[0] = -Infinity;
  edges[K] = Infinity;
  for (let k = 1; k < K; k++) {
    edges[k] = 0.5 * (x[bp[k] - 1] + x[bp[k]]);
  }
  return edges;
}
