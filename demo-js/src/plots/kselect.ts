import Plotly from 'plotly.js-dist-min';
import type { ComputeResult } from '../worker/protocol';

export function plotKSelect(result: ComputeResult, elId: string): void {
  const { looPerK, testCrpsPerK, Kstar } = result;
  const Kmax = looPerK.length;
  const ks = Array.from({ length: Kmax }, (_, i) => i + 1);

  const finite = (arr: number[]) => arr.map(v => (isFinite(v) ? v : null));

  const traceLeft: Partial<Plotly.ScatterData> = {
    x: ks,
    y: finite(looPerK),
    mode: 'lines+markers',
    name: 'Within-sample LOO-CRPS',
    line: { color: '#4c78a8', width: 1.5 },
    marker: { size: 5, color: '#4c78a8' },
    xaxis: 'x',
    yaxis: 'y',
    connectgaps: false,
  };

  const traceRight: Partial<Plotly.ScatterData> = {
    x: ks,
    y: finite(testCrpsPerK),
    mode: 'lines+markers',
    name: 'CV test CRPS',
    line: { color: '#e45756', width: 1.5 },
    marker: { size: 5, color: '#e45756' },
    xaxis: 'x2',
    yaxis: 'y2',
    connectgaps: false,
  };

  const kstarY = testCrpsPerK[Kstar - 1];
  const kstarMarker: Partial<Plotly.ScatterData> = {
    x: [Kstar],
    y: [isFinite(kstarY) ? kstarY : null],
    mode: 'markers',
    name: `K* = ${Kstar}`,
    marker: { size: 10, color: '#e45756', symbol: 'circle' },
    xaxis: 'x2',
    yaxis: 'y2',
    showlegend: true,
  };

  const layout: Partial<Plotly.Layout> = {
    grid: { rows: 1, columns: 2, pattern: 'independent' },
    xaxis: { title: 'K', tickmode: 'linear', dtick: 1 },
    yaxis: { title: 'Total LOO-CRPS' },
    xaxis2: { title: 'K', tickmode: 'linear', dtick: 1 },
    yaxis2: { title: 'Average test CRPS' },
    shapes: [
      {
        type: 'line',
        x0: Kstar, x1: Kstar,
        y0: 0, y1: 1,
        yref: 'y2 domain',
        xref: 'x2',
        line: { color: '#e45756', width: 1.5, dash: 'dash' },
      },
    ],
    annotations: [
      {
        text: '<b>Within-sample LOO-CRPS</b><br>(gameable by DP)',
        xref: 'x domain', yref: 'y domain',
        x: 0.5, y: 1.05, xanchor: 'center', yanchor: 'bottom',
        showarrow: false, font: { size: 12 },
      },
      {
        text: '<b>Cross-validated test CRPS</b><br>(U-shaped — use this for K selection)',
        xref: 'x2 domain', yref: 'y2 domain',
        x: 0.5, y: 1.05, xanchor: 'center', yanchor: 'bottom',
        showarrow: false, font: { size: 12 },
      },
    ],
    margin: { t: 60, l: 55, r: 20, b: 50 },
    legend: { orientation: 'h', x: 0.75, y: -0.12 },
  };

  Plotly.react(elId, [traceLeft, traceRight, kstarMarker], layout, { responsive: true });
}
