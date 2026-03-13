import Plotly from 'plotly.js-dist-min';
import { PRESETS, makeDgpFn, makeQuantileFn, parseEditorCode } from './presets';
import { RNG } from './rng';
import type { RunParams, WorkerMessage, ComputeResult } from './worker/protocol';
import { plotKSelect } from './plots/kselect';
import { plotPartition } from './plots/partition';
import { plotVenn } from './plots/venn';
import { plotPvalue } from './plots/pvalue';
import { plotFan } from './plots/fan';
import { plotCoverage } from './plots/coverage';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const presetSelect = document.getElementById('preset-select') as HTMLSelectElement;
const codeEditor = document.getElementById('code-editor') as HTMLTextAreaElement;
const nSlider = document.getElementById('n-slider') as HTMLInputElement;
const nVal = document.getElementById('n-val') as HTMLSpanElement;
const xLoInput = document.getElementById('x-lo') as HTMLInputElement;
const xHiInput = document.getElementById('x-hi') as HTMLInputElement;
const kmaxSlider = document.getElementById('kmax-slider') as HTMLInputElement;
const kmaxVal = document.getElementById('kmax-val') as HTMLSpanElement;
const epsSlider = document.getElementById('eps-slider') as HTMLInputElement;
const epsVal = document.getElementById('eps-val') as HTMLSpanElement;
const seedInput = document.getElementById('seed-input') as HTMLInputElement;
const runBtn = document.getElementById('run-btn') as HTMLButtonElement;
const statusDiv = document.getElementById('status') as HTMLDivElement;

// ── Populate preset select ────────────────────────────────────────────────────
for (const p of PRESETS) {
  const opt = document.createElement('option');
  opt.value = p.label;
  opt.textContent = p.label;
  presetSelect.appendChild(opt);
}
codeEditor.value = PRESETS[0].code;

presetSelect.addEventListener('change', () => {
  const preset = PRESETS.find(p => p.label === presetSelect.value);
  if (preset) codeEditor.value = preset.code;
});

// ── Slider display ────────────────────────────────────────────────────────────
nSlider.addEventListener('input', () => { nVal.textContent = nSlider.value; });
kmaxSlider.addEventListener('input', () => { kmaxVal.textContent = kmaxSlider.value; });
epsSlider.addEventListener('input', () => { epsVal.textContent = parseFloat(epsSlider.value).toFixed(2); });

// ── Tab switching ─────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    const tabId = (btn as HTMLElement).dataset['tab']!;
    const panel = document.getElementById(tabId)!;
    panel.classList.add('active');
    // Resize any Plotly chart that was rendered while this panel was hidden
    const plotDiv = panel.querySelector<HTMLElement>('.js-plotly-plot');
    if (plotDiv) Plotly.Plots.resize(plotDiv);
  });
});

// ── Worker management ─────────────────────────────────────────────────────────
let currentWorker: Worker | null = null;

function setStatus(msg: string, isError = false): void {
  statusDiv.textContent = msg;
  statusDiv.className = isError ? 'error' : '';
}

// ── Run button ────────────────────────────────────────────────────────────────
runBtn.addEventListener('click', () => {
  // Terminate any running worker
  if (currentWorker) {
    currentWorker.terminate();
    currentWorker = null;
  }

  // Parse inputs
  const xLo = parseFloat(xLoInput.value);
  const xHi = parseFloat(xHiInput.value);
  if (isNaN(xLo) || isNaN(xHi) || xLo >= xHi) {
    setStatus('Error: x lo must be less than x hi', true);
    return;
  }

  const n = parseInt(nSlider.value);
  const Kmax = parseInt(kmaxSlider.value);
  const coverage = parseFloat(epsSlider.value);
  const epsilon = parseFloat((1 - coverage).toFixed(4));
  const seed = parseInt(seedInput.value) || 0;

  // Split editor into DGP body and optional quantile body
  const { dgpBody, quantileBody } = parseEditorCode(codeEditor.value);

  // Validate and compile DGP
  let dgpFn: (x: number, rng: RNG) => number;
  try {
    dgpFn = makeDgpFn(dgpBody);
    // Test it once
    dgpFn(xLo, new RNG(seed));
  } catch (e) {
    setStatus(`DGP error: ${e instanceof Error ? e.message : String(e)}`, true);
    return;
  }

  // Compile optional true-quantile function
  let trueQuantile: ((x: number, p: number) => number) | null = null;
  try {
    trueQuantile = makeQuantileFn(quantileBody);
  } catch {
    // ignore quantile errors — oracle overlay simply won't appear
  }

  // Generate data
  const rng = new RNG(seed);
  const rawX: number[] = [];
  for (let i = 0; i < n; i++) rawX.push(rng.uniform(xLo, xHi));
  rawX.sort((a, b) => a - b);
  const yArr: number[] = rawX.map(xi => dgpFn(xi, rng));

  const params: RunParams = { x: rawX, y: yArr, Kmax, epsilon, seed };

  // Disable run button
  runBtn.disabled = true;
  setStatus('[1/4] Starting computation…');

  // Spawn worker
  const worker = new Worker(new URL('./worker/compute.worker.ts', import.meta.url), {
    type: 'module',
  });
  currentWorker = worker;

  worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
    const msg = event.data;
    if (msg.type === 'progress') {
      setStatus(`[${msg.step}] ${msg.message}`);
    } else if (msg.type === 'result') {
      handleResult(msg.data, dgpFn, trueQuantile, xLo, xHi, seed);
      runBtn.disabled = false;
      currentWorker = null;
      worker.terminate();
    } else if (msg.type === 'error') {
      setStatus(`Error: ${msg.message}`, true);
      runBtn.disabled = false;
      currentWorker = null;
      worker.terminate();
    }
  };

  worker.onerror = (e) => {
    setStatus(`Worker error: ${e.message}`, true);
    runBtn.disabled = false;
    currentWorker = null;
  };

  worker.postMessage(params);
});

// ── Handle result ─────────────────────────────────────────────────────────────
function handleResult(
  result: ComputeResult,
  dgpFn: (x: number, rng: RNG) => number,
  trueQuantile: ((x: number, p: number) => number) | null,
  xLo: number,
  xHi: number,
  seed: number
): void {
  clearPlaceholders();

  plotKSelect(result, 'kselect');
  plotPartition(result, 'partition');
  plotVenn(result, 'venn');
  plotPvalue(result, 'pvalue');
  plotFan(result, 'fan', trueQuantile);
  plotCoverage(result, 'coverage', dgpFn, xLo, xHi, seed);

  const sizes = result.bins.map(b => b.size).join(' / ');
  setStatus(`Done — K* = ${result.Kstar}, bin sizes: ${sizes}`);
}

function clearPlaceholders(): void {
  document.querySelectorAll<HTMLElement>('.tab-panel .placeholder').forEach(el => el.remove());
}
