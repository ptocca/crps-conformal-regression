import Plotly from 'plotly.js-dist-min';
import type { ComputeResult } from '../worker/protocol';

export function plotPartition(result: ComputeResult, elId: string): void {
  const { x, y, breakpoints: bp, edges, K, bins } = result;

  const scatter: Partial<Plotly.ScatterData> = {
    x,
    y,
    mode: 'markers',
    name: 'Training data',
    marker: { size: 5, color: '#4c78a8', opacity: 0.55 },
  };

  const xLo = x[0];
  const xHi = x[x.length - 1];
  const pad = 0.02 * (xHi - xLo);

  const shapes: Partial<Plotly.Shape>[] = [];
  const annotations: Partial<Plotly.Annotations>[] = [];

  const yVals = y as number[];
  const yMin = Math.min(...yVals);
  const yMax = Math.max(...yVals);
  const yLabelY = yMax + 0.04 * (yMax - yMin);

  for (let k = 0; k < K; k++) {
    const color = bins[k].color;
    const xl = Math.max(edges[k] === -Infinity ? xLo - pad : edges[k], xLo - pad);
    const xr = Math.min(edges[k + 1] === Infinity ? xHi + pad : edges[k + 1], xHi + pad);

    // Shaded band
    shapes.push({
      type: 'rect',
      x0: xl, x1: xr,
      y0: yMin - pad * 3, y1: yMax + pad * 3,
      yref: 'y',
      fillcolor: color,
      opacity: 0.12,
      line: { width: 0 },
      layer: 'below',
    });

    // Bin boundary line (skip first)
    if (k > 0) {
      shapes.push({
        type: 'line',
        x0: edges[k], x1: edges[k],
        y0: yMin - pad * 3, y1: yMax + pad * 3,
        yref: 'y',
        line: { color: '#aaa', width: 0.8, dash: 'dot' },
      });
    }

    // Bin label
    const xMid = 0.5 * (x[bp[k]] + x[bp[k + 1] - 1]);
    annotations.push({
      x: xMid,
      y: yLabelY,
      text: `Bin ${k + 1}<br>(m=${bins[k].size})`,
      showarrow: false,
      font: { size: 10, color },
      xanchor: 'center',
      yanchor: 'bottom',
    });
  }

  const layout: Partial<Plotly.Layout> = {
    title: { text: `Optimal ${K}-bin partition  (K* = ${K})`, font: { size: 13 } },
    xaxis: { title: 'x' },
    yaxis: { title: 'y' },
    shapes,
    annotations,
    margin: { t: 50, l: 55, r: 20, b: 50 },
    showlegend: false,
  };

  Plotly.react(elId, [scatter], layout, { responsive: true });
}
