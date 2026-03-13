import Plotly from 'plotly.js-dist-min';
import type { ComputeResult } from '../worker/protocol';
import { conformalPvalueGrid } from '../algorithms/conformal';

export function plotPvalue(result: ComputeResult, elId: string): void {
  const { bins, epsilon } = result;
  const K = bins.length;

  const ncols = Math.min(K, 4);
  const nrows = Math.ceil(K / ncols);
  const nGrid = 600;

  // Compute shared y_h range across all bins
  let yhMin = Infinity, yhMax = -Infinity;
  for (const bin of bins) {
    const yBin = bin.yBin;
    const mean = yBin.reduce((s, v) => s + v, 0) / yBin.length;
    let std = 0;
    for (const v of yBin) std += (v - mean) * (v - mean);
    std = yBin.length > 1 ? Math.sqrt(std / (yBin.length - 1)) : 0.01;
    std = Math.max(std, 0.01);
    yhMin = Math.min(yhMin, mean - 4 * std);
    yhMax = Math.max(yhMax, mean + 4 * std);
  }

  const yGrid: number[] = [];
  for (let i = 0; i < nGrid; i++) yGrid.push(yhMin + (yhMax - yhMin) * (i / (nGrid - 1)));

  const traces: Partial<Plotly.ScatterData>[] = [];
  const annotations: Partial<Plotly.Annotations>[] = [];

  for (let k = 0; k < K; k++) {
    const col = (k % ncols) + 1;
    const row = Math.floor(k / ncols) + 1;
    const panelIdx = (row - 1) * ncols + col;
    const ax = panelIdx === 1 ? '' : `${panelIdx}`;
    const xaxisKey = `x${ax}` as Plotly.ScatterData['xaxis'];
    const yaxisKey = `y${ax}` as Plotly.ScatterData['yaxis'];

    const bin = bins[k];
    const pvals = conformalPvalueGrid(bin.yBin, yGrid);
    const color = bin.color;

    // p-value curve
    traces.push({
      x: yGrid,
      y: Array.from(pvals),
      mode: 'lines',
      line: { color, width: 1.5 },
      name: `Bin ${k + 1}`,
      showlegend: false,
      xaxis: xaxisKey,
      yaxis: yaxisKey,
    });

    // Prediction set fill (p > epsilon)
    const fillX: number[] = [];
    const fillY: number[] = [];
    for (let i = 0; i < nGrid; i++) {
      if (pvals[i] > epsilon) {
        fillX.push(yGrid[i]);
        fillY.push(pvals[i]);
      }
    }
    if (fillX.length > 0) {
      traces.push({
        x: fillX,
        y: fillY,
        mode: 'lines',
        fill: 'tozeroy',
        fillcolor: color + '44',
        line: { width: 0, color },
        name: 'Prediction set',
        showlegend: false,
        xaxis: xaxisKey,
        yaxis: yaxisKey,
      });
    }

    // epsilon line (horizontal)
    traces.push({
      x: [yhMin, yhMax],
      y: [epsilon, epsilon],
      mode: 'lines',
      line: { color: '#888', width: 1, dash: 'dash' },
      showlegend: false,
      xaxis: xaxisKey,
      yaxis: yaxisKey,
    });

    annotations.push({
      text: `Bin ${k + 1}  (x̃ = ${bin.xMid.toFixed(2)},  m = ${bin.size})`,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      xref: `x${ax} domain` as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      yref: `y${ax} domain` as any,
      x: 0.5, y: 1.04,
      xanchor: 'center', yanchor: 'bottom',
      showarrow: false,
      font: { size: 10 },
    });
  }

  // Build subplot axes grid
  const layoutAxes: Record<string, unknown> = {};
  const colW = 1 / ncols;
  const rowH = 1 / nrows;
  for (let k = 0; k < K; k++) {
    const col = (k % ncols);
    const row = Math.floor(k / ncols);
    const panelIdx = row * ncols + col + 1;
    const ax = panelIdx === 1 ? '' : `${panelIdx}`;
    const xDomain: [number, number] = [col * colW + 0.02, (col + 1) * colW - 0.02];
    const yDomain: [number, number] = [
      (nrows - row - 1) * rowH + 0.04,
      (nrows - row) * rowH - 0.06,
    ];
    layoutAxes[`xaxis${ax}`] = { title: 'yₕ', domain: xDomain, range: [yhMin, yhMax] };
    layoutAxes[`yaxis${ax}`] = { title: col === 0 ? 'p-value' : '', domain: yDomain, range: [-0.02, 1.05] };
  }

  // Hide empty panels
  for (let k = K; k < ncols * nrows; k++) {
    const panelIdx = k + 1;
    const ax = panelIdx === 1 ? '' : `${panelIdx}`;
    layoutAxes[`xaxis${ax}`] = { visible: false };
    layoutAxes[`yaxis${ax}`] = { visible: false };
  }

  const layout: Partial<Plotly.Layout> = {
    title: { text: `Conformal p-value curves by bin  [ε = ${epsilon.toFixed(2)}]`, font: { size: 13 } },
    ...layoutAxes,
    annotations,
    margin: { t: 60, l: 55, r: 20, b: 50 },
  } as Partial<Plotly.Layout>;

  Plotly.react(elId, traces, layout, { responsive: true });
}
