import { RNG } from './rng';

export interface Preset {
  label: string;
  /** Full editor code: DGP body, optionally followed by QUANTILE_SEP and quantile body */
  code: string;
}

/**
 * Separator line that divides the DGP body from the optional trueQuantile body.
 * Must appear alone on a line in the editor.
 */
export const QUANTILE_SEP = '// --- trueQuantile(x, p) ---';

export const PRESETS: Preset[] = [
  {
    label: 'Heteroscedastic normal',
    code: `// Y | X=x ~ Normal(x, 0.3*(1+x))
const std = 0.3 * (1 + x);
return rng.normal(x, std);

${QUANTILE_SEP}
const std = 0.3 * (1 + x);
const z = Math.sqrt(2) * erfInv(2 * p - 1);
return x + std * z;`,
  },
  {
    label: 'Skewed (gamma)',
    code: `// Y | X=x ~ Gamma(shape=2, scale=(x+1)/2)
// mean = x+1, pronounced right skew
return rng.gamma(2, (x + 1) / 2);

${QUANTILE_SEP}
// Wilson-Hilferty approximation for Gamma(2, scale)
const a = 2, scale = (x + 1) / 2;
const z = Math.sqrt(2) * erfInv(2 * p - 1);
const t = 1 - 1 / (9 * a) + z * Math.sqrt(1 / (9 * a));
return scale * a * Math.max(t * t * t, 0);`,
  },
  {
    label: 'Bimodal mixture',
    code: `// Equal-weight mixture: N(x-1.5, 0.3) or N(x+1.5, 0.3)
if (rng.random() < 0.5) {
  return rng.normal(x - 1.5, 0.3);
} else {
  return rng.normal(x + 1.5, 0.3);
}`,
    // No closed-form quantile for a mixture — no separator section
  },
  {
    label: 'Sinusoidal mean',
    code: `// Y | X=x ~ Normal(sin(2π x / 3), 0.3)
return rng.normal(Math.sin(2 * Math.PI * x / 3), 0.3);

${QUANTILE_SEP}
const mu = Math.sin(2 * Math.PI * x / 3);
const z = Math.sqrt(2) * erfInv(2 * p - 1);
return mu + 0.3 * z;`,
  },
];

/**
 * Split editor content into DGP body and optional quantile body.
 */
export function parseEditorCode(val: string): {
  dgpBody: string;
  quantileBody: string | null;
} {
  const idx = val.indexOf(QUANTILE_SEP);
  if (idx === -1) return { dgpBody: val.trim(), quantileBody: null };
  const dgpBody = val.slice(0, idx).trim();
  const after = val.slice(val.indexOf('\n', idx) + 1).trim();
  return { dgpBody, quantileBody: after || null };
}

/**
 * Inverse error function (for normal/gamma quantile helpers).
 */
export function erfInv(x: number): number {
  const a = 0.147;
  const s = x < 0 ? -1 : 1;
  const ln = Math.log(1 - x * x);
  const t = 2 / (Math.PI * a) + ln / 2;
  return s * Math.sqrt(Math.sqrt(t * t - ln / a) - t);
}

/**
 * Compile the DGP body string into a callable (x, rng) => number.
 */
export function makeDgpFn(dgpBody: string): (x: number, rng: RNG) => number {
  // eslint-disable-next-line @typescript-eslint/no-implied-eval
  return new Function('x', 'rng', dgpBody) as (x: number, rng: RNG) => number;
}

/**
 * Compile the quantile body string into a callable (x, p) => number.
 * Returns null if no body is provided.
 * erfInv is injected as a third parameter so quantile code can use it.
 */
export function makeQuantileFn(
  quantileBody: string | null | undefined
): ((x: number, p: number) => number) | null {
  if (!quantileBody) return null;
  type QFn = (x: number, p: number, erfInv: (x: number) => number) => number;
  // eslint-disable-next-line @typescript-eslint/no-implied-eval
  const raw = new Function('x', 'p', 'erfInv', quantileBody) as QFn;
  return (x: number, p: number) => raw(x, p, erfInv);
}
