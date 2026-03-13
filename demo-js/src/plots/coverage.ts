import Plotly from 'plotly.js-dist-min';
import type { ComputeResult } from '../worker/protocol';
import { RNG } from '../rng';
import { conformalPvalueGrid } from '../algorithms/conformal';

function binIndex(xq: number, edges: number[], K: number): number {
  let idx = 0;
  for (let k = 1; k < K; k++) {
    if (xq > edges[k]) idx = k; else break;
  }
  return Math.max(0, Math.min(K - 1, idx));
}

export function plotCoverage(
  result: ComputeResult,
  elId: string,
  dgpFn: (x: number, rng: RNG) => number,
  xLo: number,
  xHi: number,
  seed: number
): void {
  const { breakpoints: bp, edges, K, epsilon } = result;

  const nTest = 1000;
  const rng = new RNG(seed + 999);

  // Generate fresh test data
  const xTe: number[] = [];
  const yTe: number[] = [];
  for (let i = 0; i < nTest; i++) xTe.push(rng.uniform(xLo, xHi));
  xTe.sort((a, b) => a - b);
  for (let i = 0; i < nTest; i++) yTe.push(dgpFn(xTe[i], rng));

  // Compute p-values
  const pvals: number[] = [];
  for (let i = 0; i < nTest; i++) {
    const k = binIndex(xTe[i], edges, K);
    const yBin = result.y.slice(bp[k], bp[k + 1]);
    const pv = conformalPvalueGrid(yBin, [yTe[i]]);
    pvals.push(pv[0]);
  }

  // Histogram of p-values (20 bins over [0,1])
  const nBins = 20;
  const hist = new Array(nBins).fill(0);
  for (const pv of pvals) {
    const b = Math.min(Math.floor(pv * nBins), nBins - 1);
    hist[b]++;
  }
  const histX = Array.from({ length: nBins }, (_, i) => (i + 0.5) / nBins);
  const histDensity = hist.map(c => (c / nTest) * nBins); // density

  // Coverage curve
  const nEps = 200;
  const epsGrid: number[] = [];
  const empCov: number[] = [];
  for (let i = 0; i < nEps; i++) {
    const e = 0.01 + 0.49 * (i / (nEps - 1));
    epsGrid.push(e);
    empCov.push(pvals.filter(pv => pv > e).length / nTest);
  }

  const empCovAtEps = pvals.filter(pv => pv > epsilon).length / nTest;

  // ── Traces ─────────────────────────────────────────────────────────────────
  // Left subplot: histogram
  const traceHist: Partial<Plotly.Data> = {
    x: histX,
    y: histDensity,
    type: 'bar',
    name: 'P-value density',
    marker: { color: '#4c78a8', opacity: 0.7 },
    width: new Array(nBins).fill(1 / nBins),
    xaxis: 'x',
    yaxis: 'y',
  };

  const traceUniform: Partial<Plotly.ScatterData> = {
    x: [0, 1],
    y: [1, 1],
    mode: 'lines',
    line: { color: '#e45756', dash: 'dash', width: 1.5 },
    name: 'Uniform(0,1)',
    xaxis: 'x',
    yaxis: 'y',
  };

  // Right subplot: coverage curve
  const traceNominal: Partial<Plotly.ScatterData> = {
    x: epsGrid,
    y: epsGrid.map(e => 1 - e),
    mode: 'lines',
    line: { color: '#e45756', dash: 'dash', width: 1.5 },
    name: 'Nominal 1−ε',
    xaxis: 'x2',
    yaxis: 'y2',
  };

  const traceEmpCov: Partial<Plotly.ScatterData> = {
    x: epsGrid,
    y: empCov,
    mode: 'lines',
    line: { color: '#4c78a8', width: 1.5 },
    name: 'Empirical coverage',
    xaxis: 'x2',
    yaxis: 'y2',
  };

  const layout: Partial<Plotly.Layout> = {
    grid: { rows: 1, columns: 2, pattern: 'independent' },
    xaxis: { title: 'p(y*)', range: [0, 1], domain: [0, 0.47] },
    yaxis: { title: 'Density', domain: [0, 1] },
    xaxis2: { title: 'ε', domain: [0.53, 1] },
    yaxis2: { title: 'Coverage', range: [0, 1] },
    annotations: [
      {
        text: `<b>P-value distribution on test set (n = ${nTest})</b>`,
        xref: 'x domain', yref: 'y domain',
        x: 0.5, y: 1.05, xanchor: 'center', yanchor: 'bottom',
        showarrow: false, font: { size: 12 },
      },
      {
        text: `<b>Empirical coverage = ${(empCovAtEps * 100).toFixed(1)}%</b><br>(target ≥ ${Math.round(100 * (1 - epsilon))}%,  ε = ${epsilon.toFixed(2)})`,
        xref: 'x2 domain', yref: 'y2 domain',
        x: 0.5, y: 1.05, xanchor: 'center', yanchor: 'bottom',
        showarrow: false, font: { size: 12 },
      },
    ],
    shapes: [
      {
        type: 'line',
        x0: epsilon, x1: epsilon,
        y0: 0, y1: 1,
        xref: 'x2', yref: 'y2 domain',
        line: { color: '#888', width: 1, dash: 'dot' },
      },
    ],
    margin: { t: 60, l: 55, r: 20, b: 50 },
    legend: { orientation: 'h', x: 0.5, y: -0.12, xanchor: 'center' },
    bargap: 0,
  };

  Plotly.react(elId, [traceHist, traceUniform, traceNominal, traceEmpCov], layout, {
    responsive: true,
  });
}
