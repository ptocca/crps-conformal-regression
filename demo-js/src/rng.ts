/** mulberry32 PRNG — fast 32-bit generator */
function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s += 0x6d2b79f5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** RNG class wrapping mulberry32 with typed distributions */
export class RNG {
  private _rand: () => number;

  constructor(seed: number) {
    this._rand = mulberry32(seed >>> 0);
  }

  /** Uniform(0,1) */
  random(): number {
    return this._rand();
  }

  /** Uniform(lo, hi) */
  uniform(lo = 0, hi = 1): number {
    return lo + (hi - lo) * this._rand();
  }

  /** Normal(mean, std) via Box-Muller */
  normal(mean = 0, std = 1): number {
    const u1 = Math.max(this._rand(), 1e-15);
    const u2 = this._rand();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mean + std * z;
  }

  /**
   * Gamma(shape, scale) — Marsaglia-Tsang method (shape ≥ 1).
   * For shape < 1: use boost trick Gamma(shape+1) * U^(1/shape).
   */
  gamma(shape: number, scale = 1): number {
    if (shape <= 0) throw new RangeError('Gamma shape must be > 0');
    if (shape < 1) {
      const g = this._gamma1plus(shape + 1);
      return scale * g * Math.pow(this._rand(), 1 / shape);
    }
    return scale * this._gamma1plus(shape);
  }

  private _gamma1plus(d: number): number {
    // Marsaglia-Tsang for d ≥ 1
    const c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x: number, v: number;
      do {
        x = this.normal(0, 1);
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const u = this._rand();
      const x2 = x * x;
      if (u < 1 - 0.0331 * (x2 * x2)) return d * v;
      if (Math.log(u) < 0.5 * x2 + d * (1 - v + Math.log(v))) return d * v;
    }
  }
}
