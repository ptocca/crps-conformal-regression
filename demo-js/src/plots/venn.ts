import Plotly from 'plotly.js-dist-min';
import type { ComputeResult } from '../worker/protocol';

function ecdf(values: number[], grid: number[]): number[] {
  const sorted = values.slice().sort((a, b) => a - b);
  const m = sorted.length;
  return grid.map(t => {
    let count = 0;
    for (const v of sorted) { if (v <= t) count++; else break; }
    return count / m;
  });
}

export function plotVenn(result: ComputeResult, elId: string): void {
  const { x, y, breakpoints: bp, edges, K, bins } = result;
  const n = x.length;
  const n_queries = 4;

  // Pick 4 representative query x-values at 5%, 35%, 65%, 95% quantiles
  const xLo = x[Math.floor(0.05 * n)];
  const xHi = x[Math.floor(0.95 * n)];
  const queryXs: number[] = [];
  for (let q = 0; q < n_queries; q++) {
    queryXs.push(xLo + (xHi - xLo) * (q / (n_queries - 1)));
  }

  const yArr = y as number[];
  const yMin = Math.min(...yArr);
  const yMax = Math.max(...yArr);
  const yLo = yMin - 0.25 * (yMax - yMin);
  const yHi = yMax + 0.25 * (yMax - yMin);
  const nGrid = 300;
  const tFine: number[] = [];
  for (let i = 0; i < nGrid; i++) tFine.push(yLo + (yHi - yLo) * (i / (nGrid - 1)));

  // Figure out which bin each query falls into
  function getBinIndex(xq: number): number {
    let idx = 0;
    for (let k = 1; k < K; k++) {
      if (xq > edges[k]) idx = k; else break;
    }
    return idx;
  }

  const traces: Partial<Plotly.ScatterData>[] = [];
  const annotations: Partial<Plotly.Annotations>[] = [];
  const subplotFraction = 1 / n_queries;

  for (let qi = 0; qi < n_queries; qi++) {
    const xq = queryXs[qi];
    const binIdx = getBinIndex(xq);
    const yBin = bins[binIdx].yBin;
    const m = yBin.length;
    const color = bins[binIdx].color;

    const cdf = ecdf(yBin, tFine);
    const fLo = cdf.map(c => (m / (m + 1)) * c);
    const fHi = cdf.map(c => (m / (m + 1)) * c + 1 / (m + 1));

    const ax = qi === 0 ? '' : `${qi + 1}`;
    const xaxisKey = `x${ax}` as Plotly.ScatterData['xaxis'];
    const yaxisKey = `y${ax}` as Plotly.ScatterData['yaxis'];

    // Venn band fill (upper - lower)
    traces.push({
      x: tFine,
      y: fHi,
      mode: 'lines',
      line: { color, width: 0 },
      showlegend: false,
      xaxis: xaxisKey,
      yaxis: yaxisKey,
      fill: 'none',
    });
    traces.push({
      x: tFine,
      y: fLo,
      mode: 'lines',
      line: { color, width: 0 },
      fill: 'tonexty',
      fillcolor: color + '44',
      showlegend: qi === 0,
      name: `Venn band (w=1/${m + 1})`,
      xaxis: xaxisKey,
      yaxis: yaxisKey,
    });

    // ECDF step
    traces.push({
      x: tFine,
      y: cdf,
      mode: 'lines',
      line: { color, width: 1.5, shape: 'hv' },
      name: 'Bin ECDF',
      showlegend: qi === 0,
      xaxis: xaxisKey,
      yaxis: yaxisKey,
    });

    // Rug
    traces.push({
      x: yBin,
      y: new Array(m).fill(-0.04),
      mode: 'markers',
      marker: { symbol: 'line-ns-open', color, size: 8, opacity: 0.6 },
      showlegend: false,
      xaxis: xaxisKey,
      yaxis: yaxisKey,
    });

    // Subplot title
    annotations.push({
      text: `x* = ${xq.toFixed(2)},  m = ${m}`,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      xref: `x${ax} domain` as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      yref: `y${ax} domain` as any,
      x: 0.5, y: 1.05,
      xanchor: 'center', yanchor: 'bottom',
      showarrow: false,
      font: { size: 11 },
    });
  }

  // Build layout axes
  const layoutAxes: Record<string, unknown> = {};
  for (let qi = 0; qi < n_queries; qi++) {
    const ax = qi === 0 ? '' : `${qi + 1}`;
    const domain: [number, number] = [
      qi * subplotFraction + 0.01,
      (qi + 1) * subplotFraction - 0.01,
    ];
    layoutAxes[`xaxis${ax}`] = { title: 'y', domain, anchor: `y${ax}` };
    layoutAxes[`yaxis${ax}`] = {
      title: qi === 0 ? 'CDF' : '',
      range: [-0.09, 1.09],
      domain: [0, 1],
      anchor: `x${ax}`,
      showticklabels: qi === 0,
    };
  }

  const layout: Partial<Plotly.Layout> = {
    ...layoutAxes,
    annotations,
    margin: { t: 40, l: 55, r: 20, b: 60 },
    legend: {
      orientation: 'h',
      x: 0.5, xanchor: 'center',
      y: -0.12,
      bgcolor: 'rgba(255,255,255,0.85)',
      bordercolor: '#ddd',
      borderwidth: 1,
    },
  } as Partial<Plotly.Layout>;

  Plotly.react(elId, traces, layout, { responsive: true });
}
