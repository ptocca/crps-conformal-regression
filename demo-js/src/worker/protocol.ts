export interface RunParams {
  x: number[];
  y: number[];
  Kmax: number;
  epsilon: number;
  seed: number;
}

export interface BinData {
  /** y values in this bin */
  yBin: number[];
  /** median x of the training points in this bin */
  xMid: number;
  /** number of training points */
  size: number;
  /** CSS color string */
  color: string;
  /** conformal interval [lo, hi] */
  interval: [number, number];
}

export interface ComputeResult {
  x: number[];
  y: number[];
  /** breakpoints array, length K+1 */
  breakpoints: number[];
  /** x-axis bin edges, length K+1 */
  edges: number[];
  K: number;
  /** LOO total cost per K (gameable) */
  looPerK: number[];
  /** Cross-validated test CRPS per K */
  testCrpsPerK: number[];
  Kstar: number;
  bins: BinData[];
  epsilon: number;
}

export type WorkerMessage =
  | { type: 'progress'; step: string; message: string }
  | { type: 'result'; data: ComputeResult }
  | { type: 'error'; message: string };
