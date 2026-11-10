import type { Frame, HeatmapLayer, Overlay } from "../layers";
import type { LiqMap } from "./api";

export type ScaleMode = "sqrt" | "log" | "linear";
export interface LiqOpts { scale: ScaleMode; thr: number }

const STOPS = [[68, 1, 84], [71, 44, 122], [59, 81, 139], [44, 113, 142], [33, 144, 141], [39, 173, 129], [92, 200, 99], [170, 220, 50], [253, 231, 37]];
/** Viridis, 256 entries of [r,g,b]. */
export const VIRIDIS: number[][] = Array.from({ length: 256 }, (_, i) => {
  const x = (i / 255) * (STOPS.length - 1), k = Math.min(STOPS.length - 2, Math.floor(x)), f = x - k;
  return STOPS[k].map((a, c) => Math.round(a + (STOPS[k + 1][c] - a) * f));
});

/** Normalised intensity 0..1 of a quantised cell under the chosen scaling. */
export function norm(q: number, mode: ScaleMode): number {
  const s = q / 255;
  if (mode === "sqrt") return s;
  const lin = s * s;
  return mode === "linear" ? lin : Math.log1p(1000 * lin) / Math.log1p(1000);
}
export const usdOf = (q: number, scale: number) => scale * (q / 255) ** 2;
export const qOf = (usd: number, scale: number) => Math.round(255 * Math.sqrt(Math.min(usd / scale, 1)));

/** Default: hide the weakest 60% of non-zero cells (percentile threshold). */
export const DEFAULT_THR = 0.6;
const GAMMA = 2.2;

export interface Hit { time: number; price: number; usd: number; side: "long" | "short"; future: boolean }

/**
 * Liquidation heatmap layer (Coinglass style). A column is drawn from its time to the next one; a
 * consumed level is simply absent from later columns. Active levels (last column) extend to the
 * right edge as future blocks.
 */
export class LiqLayer implements HeatmapLayer {
  rev = 0;
  private m: LiqMap | null = null;
  private o: LiqOpts = { scale: "sqrt", thr: DEFAULT_THR };
  private map: { xl: number; dx: number; hr: number } | null = null;
  constructor(public id = "liq") {}
  get data() { return this.m; }
  get opts() { return this.o; }
  /** Cumulative share of non-zero cells at or below each quantised level (percentile rank). */
  private cdf = new Float32Array(256);
  /** Quantised level of the 99th percentile of non-zero cells: the normalisation ceiling. */
  private q99 = 255;
  setData(m: LiqMap | null) {
    this.m = m; this.map = null; this.rev++;
    if (!m) return;
    const h = new Float64Array(256);
    for (let i = 0; i < m.q.length; i++) if (m.q[i]) h[m.q[i]]++;
    const tot = h.reduce((a, b) => a + b, 0) || 1;
    let acc = 0, q99 = 255;
    for (let q = 1; q < 256; q++) { acc += h[q]; this.cdf[q] = acc / tot; if (q99 === 255 && acc / tot >= 0.99) q99 = q; }
    this.q99 = Math.max(1, q99);
  }
  setOpts(o: Partial<LiqOpts>) { this.o = { ...this.o, ...o }; this.rev++; }
  /** USD value at the normalisation ceiling (p99), for the colour bar. */
  get ceiling() { return this.m ? usdOf(this.q99, this.m.scale) : 0; }
  /** Percentile rank (0..1) of a quantised level among non-zero cells. */
  rank(q: number) { return this.cdf[q]; }
  /** Display intensity 0..1: USD / p99 USD, scaled (sqrt/log/linear), then gamma so mid values stay dark. */
  intensity(q: number) {
    const r = Math.min(1, usdOf(q, 1) / usdOf(this.q99, 1));
    const s = this.o.scale === "linear" ? r : this.o.scale === "sqrt" ? Math.sqrt(r) : Math.log1p(1000 * r) / Math.log1p(1000);
    return Math.pow(s, GAMMA);
  }

  lut(): Uint32Array {
    const l = new Uint32Array(256);
    for (let q = 1; q < 256; q++) {
      if (this.cdf[q] < this.o.thr) continue;   // weakest share of cells -> chart background
      const n = this.intensity(q);
      const [r, g, b] = VIRIDIS[Math.round(n * 255)];
      const a = Math.round((0.18 + 0.8 * Math.sqrt(n)) * 255);
      l[q] = ((a << 24) | (b << 16) | (g << 8) | r) >>> 0;
    }
    return l;
  }

  paint(f: Frame): Overlay | void {
    const m = this.m;
    if (!m || m.T < 2 || m.P < 2) return;
    const { T, P, times, prices, step } = m;
    const tl = times[T - 1], tp = times[T - 2];
    const xl = f.xOf(tl), xp = f.xOf(tp);
    if (xl == null || xp == null || xl <= xp) return;
    const dx = ((xl - xp) * step) / (tl - tp);
    this.map = { xl, dx, hr: f.hr };
    const xAt = (t: number) => xl + ((t - tl) / step) * dx;
    const sr = Math.sqrt(prices[1] / prices[0]);
    // Row edges at the price-bin boundaries (geometric midpoints), so adjacent rows touch with no seams.
    const ye = new Float64Array(P + 1);
    let rowsOk = true;
    for (let k = 0; k <= P; k++) {
      const pe = k === 0 ? prices[0] / sr : k === P ? prices[P - 1] * sr : Math.sqrt(prices[k - 1] * prices[k]);
      const y = f.yOf(pe);
      if (y == null) { rowsOk = false; break; }
      ye[k] = y;
    }
    if (!rowsOk) return;
    const y0 = new Int32Array(P), y1 = new Int32Array(P);
    for (let k = 0; k < P; k++) {
      // price grows upward: bin k spans from edge k+1 (top) to edge k (bottom)
      const a = Math.max(0, Math.round(ye[k + 1])), b = Math.min(f.H, Math.round(ye[k]));
      y0[k] = a; y1[k] = b > a ? b : Math.min(f.H, a + 1);
    }
    const lut = this.lut();
    const fill = (xa: number, xb: number, k: number, c: number) => {
      const a = c >>> 24;
      for (let yy = y0[k]; yy < y1[k]; yy++) {
        const o = yy * f.W;
        for (let xx = xa; xx < xb; xx++) if ((f.px[o + xx] >>> 24) <= a) f.px[o + xx] = c;
      }
    };
    // Column edges: each column spans from its own x to the next column's x (contiguous, no gaps).
    const xe = new Int32Array(T + 1);
    for (let i = 0; i <= T; i++) xe[i] = Math.round(i < T ? xAt(times[i]) : xAt(tl) + dx);
    const q = m.q;
    for (let i = 0; i < T; i++) {
      // 1px overlap with the next column: no seams from rounding
      const xa = Math.max(0, xe[i]), xb = Math.min(f.W, Math.max(xe[i + 1] + 1, xe[i] + 1));
      if (xb <= 0 || xa >= f.W || xb <= xa) continue;
      const base = i * P;
      for (let k = 0; k < P; k++) {
        let v = q[base + k];
        // fill single-column holes: 0 between two non-zero cells of the same row
        if (!v && i > 0 && i < T - 1) { const l = q[base - P + k], r = q[base + P + k]; if (l && r) v = Math.min(l, r); }
        const c = lut[v];
        if (c) fill(xa, xb, k, c);
      }
    }
    // Future region = the last grid column's non-zero rows, same rows and intensity, extended to the right edge
    // (the grid is the source of truth; `active` is only used for tooltip USD).
    const fx = Math.max(0, xe[T] - 1);
    if (fx < f.W) {
      const last = (T - 1) * P;
      for (let k = 0; k < P; k++) { const c = lut[q[last + k]]; if (c) fill(fx, f.W, k, c); }
    }
    const evMax = m.events.reduce((x, e) => Math.max(x, e.usd), 1);
    return (ctx) => {
      ctx.fillStyle = "rgba(255,255,255,0.55)";
      for (const e of m.events) {
        if (e.usd < evMax * 0.12) continue;
        const x = xAt(e.t), y = f.yOf(e.p);
        if (y == null || x < 0 || x > f.W) continue;
        const r = Math.max(1.5, Math.min(5, 5 * Math.sqrt(e.usd / evMax))) * f.hr;
        ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill();
      }
    };
  }

  /** Hover lookup; xCss in CSS px, price in chart units. */
  lookup(xCss: number, price: number): Hit | null {
    const m = this.m, mp = this.map;
    if (!m || !mp || !(price > 0)) return null;
    const { T, P, times, prices, step } = m;
    const t = times[T - 1] + ((xCss * mp.hr - mp.xl) / mp.dx) * step;
    if (t < times[0]) return null;
    const future = t >= times[T - 1] + step;
    const i = Math.min(T - 1, Math.floor((t - times[0]) / step));
    const lr = Math.log(prices[1] / prices[0]);
    const j0 = Math.round(Math.log(price / prices[0]) / lr);
    let best = 0, bj = -1;
    for (let j = Math.max(0, j0 - 1); j <= Math.min(P - 1, j0 + 1); j++) {
      const q = m.q[i * P + j];
      if (q > best) { best = q; bj = j; }
    }
    if (bj < 0 || this.cdf[best] < this.o.thr) return null;
    const c = m.candles[i].close;
    return { time: times[i], price: prices[bj], usd: usdOf(best, m.scale), side: prices[bj] < c ? "long" : "short", future };
  }
}
