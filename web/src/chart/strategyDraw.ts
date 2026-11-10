/** TradingView-style strategy drawing: entry/exit markers, trade connectors, indicator overlays. */
import type { SeriesMarker, Time, UTCTimestamp } from "lightweight-charts";
import type { Backtest, Bar, Pt } from "../types";

export const SCOL = { long: "#4c8dff", short: "#ff7a45", exit: "#c46bff", win: "rgba(14,168,148,0.85)", loss: "rgba(242,92,84,0.85)" };
const REASON: Record<string, string> = {
  signal: "sinal", stop_loss: "stop", take_profit: "alvo", trailing_stop: "trailing", time_stop: "tempo", reverse: "reversão", end: "fim do período",
};
export const reasonPt = (r: string) => REASON[r] ?? r;

export interface DrawTrade {
  entryT: number; exitT: number | null; entryP: number; exitP: number | null; long: boolean;
  win: boolean | null; pnl: number | null; reason: string; hold: boolean; n: number;
}

const idxAt = (bars: Bar[], t: number) => {
  let lo = 0, hi = bars.length - 1, r = -1;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (bars[m].time <= t) { r = m; lo = m + 1; } else hi = m - 1; }
  return r;
};

/** In-sample trades from the trade list; holdout trades paired from holdout markers (prices from bar closes). */
export function buildTrades(bt: Backtest, bars: Bar[]): DrawTrade[] {
  const out: DrawTrade[] = bt.trades.map((t, i) => ({
    entryT: t.entry_time, exitT: t.exit_time, entryP: t.entry_price, exitP: t.exit_price, long: t.side === "LONG",
    win: t.pnl_pct > 0, pnl: t.pnl_pct, reason: t.exit_reason, hold: false, n: i + 1,
  }));
  const px = (t: number) => { const i = idxAt(bars, t); return i >= 0 && bars[i].time === t ? bars[i].close : null; };
  let open: DrawTrade | null = null, n = out.length;
  for (const m of [...bt.holdout_markers].sort((a, b) => a.time - b.time)) {
    if (m.shape !== "circle") {
      open = { entryT: m.time, exitT: null, entryP: px(m.time) ?? NaN, exitP: null, long: m.shape === "arrowUp", win: null, pnl: null, reason: "", hold: true, n: ++n };
      out.push(open);
    } else if (open) {
      open.exitT = m.time; open.exitP = px(m.time);
      if (open.exitP != null && isFinite(open.entryP)) {
        open.pnl = (open.long ? 1 : -1) * (open.exitP / open.entryP - 1);
        open.win = open.pnl > 0;
      }
      open.reason = "signal";
      open = null;
    }
  }
  return out;
}

/** Markers snapped to displayed bars; text labels only when there is room (zoomed in) or forced. */
export function buildMarkers(trades: DrawTrade[], shown: Bar[], labels: boolean): SeriesMarker<Time>[] {
  if (!shown.length) return [];
  const t0 = shown[0].time, t1 = shown[shown.length - 1].time;
  const snap = (t: number) => { if (t < t0 || t > t1 + 1) return null; const i = idxAt(shown, t); return i < 0 ? null : (shown[i].time as UTCTimestamp); };
  const list: SeriesMarker<Time>[] = [];
  for (const tr of trades) {
    const a = snap(tr.entryT);
    const fade = tr.hold ? "bb" : "";
    if (a != null) list.push({
      time: a, position: tr.long ? "belowBar" : "aboveBar", shape: tr.long ? "arrowUp" : "arrowDown",
      color: (tr.long ? SCOL.long : SCOL.short) + fade, size: 1, text: labels ? (tr.long ? "long" : "short") : undefined,
    });
    if (tr.exitT != null) {
      const b = snap(tr.exitT);
      if (b != null) list.push({
        time: b, position: tr.long ? "aboveBar" : "belowBar", shape: tr.long ? "arrowDown" : "arrowUp",
        color: SCOL.exit + fade, size: 0.8, text: labels ? `Fechar ${tr.long ? "long" : "short"} · ${reasonPt(tr.reason)}` : undefined,
      });
    }
  }
  return list.sort((x, y) => (x.time as number) - (y.time as number));
}

// ---------------------------------------------------------------- indicators
export interface IndSeries { key: string; data: Pt[]; kind: "overlay" | "osc" }
export interface IndData { name: string; series: IndSeries[]; regime: Pt[] | null }

const cache = new Map<string, Promise<IndData>>();
/** Tfs whose /indicator payload stays small (the endpoint returns full history: 5m is ~430 MB). */
export const IND_TFS = new Set(["5m", "15m", "1h", "4h", "1d", "1w"]);

const median = (a: number[]) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1] ?? 0; };

export function getIndicator(name: string, params: object, symbol: string, tf: string, closeMedian: number): Promise<IndData> {
  const key = [name, JSON.stringify(params), symbol, tf].join("|");
  let p = cache.get(key);
  if (!p) {
    p = fetch(`/api/indicator/${encodeURIComponent(name)}?tf=${tf}&symbol=${symbol}&params=${encodeURIComponent(JSON.stringify(params))}`)
      .then((r) => { if (!r.ok) throw new Error(String(r.status)); return r.json(); })
      .then((d: { series: Record<string, Pt[]> }) => {
        const series: IndSeries[] = [];
        let regime: Pt[] | null = null;
        for (const [k, v] of Object.entries(d.series)) {
          if (!v.length) continue;
          const sample = v.slice(-500).map((x) => x.value);
          const binary = sample.every((x) => x === 0 || x === 1);
          if (binary) { if (/rising|bull|up/.test(k) && !regime) regime = v; continue; }
          const med = median(sample);
          if (closeMedian > 0 && med / closeMedian > 0.5 && med / closeMedian < 2) series.push({ key: k, data: v, kind: "overlay" });
          else if (!/_tr$|atr|range/.test(k)) series.push({ key: k, data: v, kind: "osc" });
        }
        return { name, series, regime };
      });
    p.catch(() => cache.delete(key));
    cache.set(key, p);
  }
  return p;
}
