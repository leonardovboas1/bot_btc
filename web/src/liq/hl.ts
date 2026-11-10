import type { Frame, HeatmapLayer, Overlay } from "../layers";

export interface HlLevel { p: number; lo: number; hi: number; long_usd: number; short_usd: number; n_long: number; n_short: number; age_long_s: number | null; age_short_s: number | null }
export interface HlCoverage { addresses: number; polled: number; positions: number; notional: number; oi_usd: number | null; oi_share: number | null; median_age_s: number | null }
export interface HlLiq { coin: string; empty: boolean; mark: number; levels: HlLevel[]; coverage?: HlCoverage }
export interface HlStopLevel { p: number; lo: number; hi: number; buy_usd: number; sell_usd: number; n_buy: number; n_sell: number; age_buy_s: number | null; age_sell_s: number | null }
export interface HlStops { coin: string; empty: boolean; mark: number; levels: HlStopLevel[] }

export const HL_COINS = ["BTC", "ETH", "SOL", "XRP", "BNB", "NEAR", "FET", "ADA"];
export const hlCoin = (symbol: string) => symbol.toUpperCase().replace(/(USDT|USDC|USD)$/, "");

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(`/api${path}`, { signal });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}
export const getHlLiq = (coin: string, signal?: AbortSignal) => get<HlLiq>(`/hl/liqlevels?coin=${coin}&bins=300`, signal);
export const getHlStops = (coin: string, signal?: AbortSignal) => get<HlStops>(`/hl/stops?coin=${coin}&bins=300`, signal);

export interface HlHit { price: number; usd: number; n: number; age: number | null; label: string }
interface Row { lo: number; hi: number; p: number; usd: number; n: number; age: number | null; color: string; label: string }

/** Thin bright horizontal markers at exact price bins; length ~ sqrt(USD / max USD). Sits on top of the modelled heatmap. */
export class HlLayer implements HeatmapLayer {
  rev = 0;
  private rows: Row[] = [];
  constructor(public id: string, private kind: "liq" | "stops") {}
  setData(d: HlLiq | HlStops | null) {
    this.rev++;
    this.rows = [];
    if (!d || d.empty) return;
    if (this.kind === "liq") {
      for (const l of (d as HlLiq).levels) {
        if (l.long_usd > 0) this.rows.push({ lo: l.lo, hi: l.hi, p: l.p, usd: l.long_usd, n: l.n_long, age: l.age_long_s, color: "#2de2c8", label: "Liquidação long (exata)" });
        if (l.short_usd > 0) this.rows.push({ lo: l.lo, hi: l.hi, p: l.p, usd: l.short_usd, n: l.n_short, age: l.age_short_s, color: "#ff7a70", label: "Liquidação short (exata)" });
      }
    } else {
      for (const l of (d as HlStops).levels) {
        if (l.sell_usd > 0) this.rows.push({ lo: l.lo, hi: l.hi, p: l.p, usd: l.sell_usd, n: l.n_sell, age: l.age_sell_s, color: "#ffb020", label: "Stop/TP de venda" });
        if (l.buy_usd > 0) this.rows.push({ lo: l.lo, hi: l.hi, p: l.p, usd: l.buy_usd, n: l.n_buy, age: l.age_buy_s, color: "#c46bff", label: "Stop/TP de compra" });
      }
    }
  }
  private max() { return this.rows.reduce((m, r) => Math.max(m, r.usd), 1); }
  paint(f: Frame): Overlay | void {
    if (!this.rows.length) return;
    const mx = this.max(), rows = this.rows, stops = this.kind === "stops";
    return (ctx) => {
      for (const r of rows) {
        const y = f.yOf(r.p);
        if (y == null || y < 0 || y > f.H) continue;
        const w = f.W * (0.06 + 0.34 * Math.sqrt(r.usd / mx));
        ctx.fillStyle = r.color;
        ctx.globalAlpha = 0.45 + 0.55 * Math.sqrt(r.usd / mx);
        // liquidation markers hug the right edge; stops use the left edge so both stay readable together
        ctx.fillRect(stops ? 0 : f.W - w, Math.round(y), w, Math.max(1, Math.round(f.vr)));
      }
      ctx.globalAlpha = 1;
    };
  }
  /** Nearest marker within tolPx (CSS px) of the cursor price. */
  lookup(price: number, yOf: (p: number) => number | null, y: number, tol = 5): HlHit | null {
    let best: Row | null = null, bd = tol;
    for (const r of this.rows) {
      const yy = yOf(r.p);
      if (yy == null) continue;
      const d = Math.abs(yy - y);
      if (d <= bd) { bd = d; best = r; }
    }
    void price;
    return best ? { price: best.p, usd: best.usd, n: best.n, age: best.age, label: best.label } : null;
  }
}
