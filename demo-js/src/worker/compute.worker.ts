import { precomputeCosts, optimalPartition, binXBoundaries } from '../algorithms/binning';
import { selectKCV } from '../algorithms/selection';
import { conformalInterval } from '../algorithms/conformal';
import type { RunParams, WorkerMessage, ComputeResult, BinData } from './protocol';

const COLORS = [
  '#4e79a7', '#f28e2b', '#e15759', '#76b7b2',
  '#59a14f', '#edc948', '#b07aa1', '#ff9da7',
];

function progress(step: string, message: string): void {
  const msg: WorkerMessage = { type: 'progress', step, message };
  self.postMessage(msg);
}

/** Yield control so progress messages can be flushed */
function yieldControl(): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, 0));
}

self.onmessage = async (event: MessageEvent<RunParams>) => {
  const { x, y, Kmax, epsilon } = event.data;
  const n = x.length;

  try {
    // ── Step 1: cost matrix on CV training half ─────────────────────────────
    progress('1/4', `Computing cost matrix for K-selection (n_train = ${Math.ceil(n / 2)})…`);
    await yieldControl();

    const { Kstar, looPerK, testCrpsPerK } = selectKCV(x, y, Kmax);

    // ── Step 2: progress for CV ──────────────────────────────────────────────
    progress('2/4', `Cross-validating K = 1…${Kmax} → K* = ${Kstar}…`);
    await yieldControl();

    // ── Step 3: cost matrix on full data + optimal partition ────────────────
    progress('3/4', `Fitting optimal ${Kstar}-bin partition on all ${n} points…`);
    await yieldControl();

    const C_full = precomputeCosts(y);
    const bp = optimalPartition(n, Kstar, C_full);
    const edges = binXBoundaries(x, bp);
    const K = bp.length - 1;

    // ── Step 4: per-bin conformal intervals ─────────────────────────────────
    progress('4/4', `Computing per-bin conformal intervals (ε = ${epsilon.toFixed(2)})…`);
    await yieldControl();

    const bins: BinData[] = [];
    for (let k = 0; k < K; k++) {
      const yBin = y.slice(bp[k], bp[k + 1]);
      const xBin = x.slice(bp[k], bp[k + 1]);
      const xMid = xBin[Math.floor(xBin.length / 2)];
      const interval = conformalInterval(yBin, epsilon, 1500);
      bins.push({
        yBin,
        xMid,
        size: yBin.length,
        color: COLORS[k % COLORS.length],
        interval,
      });
    }

    const result: ComputeResult = {
      x,
      y,
      breakpoints: bp,
      edges: Array.from(edges),
      K,
      looPerK,
      testCrpsPerK,
      Kstar,
      bins,
      epsilon,
    };

    const msg: WorkerMessage = { type: 'result', data: result };
    self.postMessage(msg);
  } catch (err) {
    const msg: WorkerMessage = {
      type: 'error',
      message: err instanceof Error ? err.message : String(err),
    };
    self.postMessage(msg);
  }
};
