import Plotly from 'plotly.js-dist-min';
import type { ComputeResult } from '../worker/protocol';

export function plotFan(
  result: ComputeResult,
  elId: string,
  trueQuantile: ((x: number, p: number) => number) | null
): void {
  const { x, y, edges, K, bins, epsilon } = result;

  const xPlotLo = x[3] ?? x[0];
  const xPlotHi = x[x.length - 4] ?? x[x.length - 1];

  // Build step-function arrays for the conformal interval band
  const xSteps: number[] = [];
  const loSteps: number[] = [];
  const hiSteps: number[] = [];

  for (let k = 0; k < K; k++) {
    const [lo, hi] = bins[k].interval;
    const xl = k === 0
      ? xPlotLo
      : Math.max(Math.min(edges[k], xPlotHi), xPlotLo);
    const xr = k === K - 1
      ? xPlotHi
      : Math.max(Math.min(edges[k + 1], xPlotHi), xPlotLo);

    if (k === 0) {
      xSteps.push(xl);
      loSteps.push(lo);
      hiSteps.push(hi);
    }
    xSteps.push(xr);
    loSteps.push(lo);
    hiSteps.push(hi);

    if (k < K - 1) {
      const [loNext, hiNext] = bins[k + 1].interval;
      xSteps.push(xr);
      loSteps.push(loNext);
      hiSteps.push(hiNext);
    }
  }

  const coveragePct = Math.round(100 * (1 - epsilon));

  const traces: Partial<Plotly.ScatterData>[] = [];

  // Training scatter
  traces.push({
    x: x,
    y: y,
    mode: 'markers',
    name: 'Training data',
    marker: { size: 4, color: '#4c78a8', opacity: 0.4 },
  });

  // Conformal band fill
  traces.push({
    x: xSteps,
    y: hiSteps,
    mode: 'lines',
    line: { color: '#4c78a8', width: 0 },
    showlegend: false,
    fill: 'none',
  });
  traces.push({
    x: xSteps,
    y: loSteps,
    mode: 'lines',
    line: { color: '#4c78a8', width: 0 },
    fill: 'tonexty',
    fillcolor: '#4c78a844',
    name: `Conformal ${coveragePct}% interval`,
  });

  // Upper/lower border lines
  traces.push({
    x: xSteps,
    y: loSteps,
    mode: 'lines',
    line: { color: '#4c78a8', width: 1.2 },
    showlegend: false,
  });
  traces.push({
    x: xSteps,
    y: hiSteps,
    mode: 'lines',
    line: { color: '#4c78a8', width: 1.2 },
    showlegend: false,
  });

  // Oracle quantiles (dashed red)
  if (trueQuantile) {
    try {
      const nSmooth = 200;
      const xSmooth: number[] = [];
      const qLo: number[] = [];
      const qHi: number[] = [];
      for (let i = 0; i < nSmooth; i++) {
        const xq = xPlotLo + (xPlotHi - xPlotLo) * (i / (nSmooth - 1));
        xSmooth.push(xq);
        qLo.push(trueQuantile(xq, epsilon / 2));
        qHi.push(trueQuantile(xq, 1 - epsilon / 2));
      }
      traces.push({
        x: xSmooth, y: qLo,
        mode: 'lines',
        line: { color: '#e45756', width: 1.5, dash: 'dash' },
        name: `True ${coveragePct}% interval`,
      });
      traces.push({
        x: xSmooth, y: qHi,
        mode: 'lines',
        line: { color: '#e45756', width: 1.5, dash: 'dash' },
        showlegend: false,
      });
    } catch {
      // silently skip if trueQuantile throws
    }
  }

  // Bin boundary lines
  const shapes: Partial<Plotly.Shape>[] = [];
  const allY = y as number[];
  const yMin = Math.min(...allY) - 0.5;
  const yMax = Math.max(...allY) + 0.5;
  for (let k = 1; k < K; k++) {
    shapes.push({
      type: 'line',
      x0: edges[k], x1: edges[k],
      y0: yMin, y1: yMax,
      line: { color: '#aaa', width: 0.8, dash: 'dot' },
    });
  }

  const layout: Partial<Plotly.Layout> = {
    title: {
      text: `Conformal prediction intervals — ${K} bins,  ε = ${epsilon.toFixed(2)}  (nominal coverage ≥ ${coveragePct}%)`,
      font: { size: 13 },
    },
    xaxis: { title: 'x' },
    yaxis: { title: 'y' },
    shapes,
    margin: { t: 55, l: 55, r: 20, b: 50 },
    legend: { orientation: 'h', x: 0, y: -0.12 },
  };

  Plotly.react(elId, traces, layout, { responsive: true });
}
