/**
 * Strategy-tester statistics computed client-side from /backtest (trades + equity), TradingView style.
 * Sizing model (matches the engine): 100% of equity per trade, compounded; pnl_pct is net of costs.
 */
import type { Backtest, Bar, Trade } from "../types";

export interface TT extends Trade {
  n: number; long: boolean; eqBefore: number; contracts: number; profit: number; cum: number; commission: number;
  bars: number; runup: number | null; runupPct: number | null; dd: number | null; ddPct: number | null;
}
export interface SideStats {
  net: number; netPct: number; grossP: number; grossPPct: number; grossL: number; grossLPct: number; pf: number | null;
  maxDD: number; maxDDPct: number; n: number; wins: number; losses: number; even: number; winRate: number | null;
  avg: number; avgPct: number; avgWin: number; avgWinPct: number; avgLoss: number; avgLossPct: number; ratio: number | null;
  bigWin: number; bigWinPct: number; bigLoss: number; bigLossPct: number; avgBars: number; avgBarsWin: number; avgBarsLoss: number;
  commission: number; sharpe: number | null; sortino: number | null; maxRunup: number; expectancy: number; expectancyPct: number;
  maxConsecWins: number; maxConsecLosses: number;
}
export interface Stats {
  C: number; trades: TT[]; all: SideStats; long: SideStats; short: SideStats;
  curve: { time: number; pnl: number; bh: number; dd: number; ddPct: number }[];
  maxDD: number; maxDDPct: number; maxRunup: number; maxRunupPct: number; bhReturn: number; bhPnl: number;
  start: number; end: number; years: number; cagr: number | null; outlier: number; outlierPct: number;
  sharpe: number | null; sortino: number | null; exposure: number | null;
}

const BE = 0.0005; // |return| below 0.05% counts as breakeven
const mean = (a: number[]) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
const sd = (a: number[]) => { const m = mean(a); return a.length > 1 ? Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)) : 0; };
const num = (v: unknown) => (typeof v === "number" && isFinite(v) ? v : null);

function side(ts: TT[], C: number, years: number): SideStats {
  const w = ts.filter((t) => t.pnl_pct > BE), l = ts.filter((t) => t.pnl_pct < -BE);
  const grossP = w.reduce((s, t) => s + t.profit, 0), grossL = -l.reduce((s, t) => s + t.profit, 0);
  const net = ts.reduce((s, t) => s + t.profit, 0);
  // drawdown / run-up on the closed-trade cumulative P&L of this subset
  let cum = 0, peak = 0, trough = 0, maxDD = 0, maxDDPct = 0, maxRunup = 0;
  for (const t of ts) {
    cum += t.profit;
    if (cum > peak) peak = cum;
    if (cum < trough) trough = cum;
    const dd = peak - cum;
    if (dd > maxDD) { maxDD = dd; maxDDPct = dd / (C + peak); }
    maxRunup = Math.max(maxRunup, cum - trough);
  }
  let cw = 0, cl = 0, mw = 0, ml = 0;
  for (const t of ts) {
    if (t.pnl_pct > BE) { cw++; cl = 0; } else if (t.pnl_pct < -BE) { cl++; cw = 0; } else { cw = 0; cl = 0; }
    mw = Math.max(mw, cw); ml = Math.max(ml, cl);
  }
  const r = ts.map((t) => t.pnl_pct);
  const perYear = years > 0 ? ts.length / years : 0;
  const s = sd(r), dn = Math.sqrt(mean(r.map((x) => Math.min(0, x) ** 2)));
  const avgWin = mean(w.map((t) => t.profit)), avgLoss = -mean(l.map((t) => t.profit));
  const big = ts.reduce((a, t) => (t.profit > a.profit ? t : a), ts[0] ?? null);
  const small = ts.reduce((a, t) => (t.profit < a.profit ? t : a), ts[0] ?? null);
  return {
    net, netPct: net / C, grossP, grossPPct: grossP / C, grossL, grossLPct: grossL / C, pf: grossL > 0 ? grossP / grossL : null,
    maxDD, maxDDPct, n: ts.length, wins: w.length, losses: l.length, even: ts.length - w.length - l.length,
    winRate: ts.length ? w.length / ts.length : null,
    avg: ts.length ? net / ts.length : 0, avgPct: mean(r),
    avgWin, avgWinPct: mean(w.map((t) => t.pnl_pct)), avgLoss, avgLossPct: -mean(l.map((t) => t.pnl_pct)),
    ratio: avgLoss > 0 ? avgWin / avgLoss : null,
    bigWin: big && big.profit > 0 ? big.profit : 0, bigWinPct: big && big.profit > 0 ? big.pnl_pct : 0,
    bigLoss: small && small.profit < 0 ? -small.profit : 0, bigLossPct: small && small.profit < 0 ? -small.pnl_pct : 0,
    avgBars: mean(ts.map((t) => t.bars)), avgBarsWin: mean(w.map((t) => t.bars)), avgBarsLoss: mean(l.map((t) => t.bars)),
    commission: ts.reduce((a, t) => a + t.commission, 0),
    sharpe: s > 0 && perYear ? (mean(r) / s) * Math.sqrt(perYear) : null,
    sortino: dn > 0 && perYear ? (mean(r) / dn) * Math.sqrt(perYear) : null,
    maxRunup, expectancy: ts.length ? net / ts.length : 0, expectancyPct: mean(r),
    maxConsecWins: mw, maxConsecLosses: ml,
  };
}

/** Index of the last bar with time <= t (binary search), -1 if none. */
function at(bars: Bar[], t: number) {
  let lo = 0, hi = bars.length - 1, r = -1;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (bars[m].time <= t) { r = m; lo = m + 1; } else hi = m - 1; }
  return r;
}

export function computeStats(bt: Backtest, C: number, feeRate: number, tfSec: number, bars?: Bar[] | null): Stats {
  let eq = C, cum = 0;
  const trades: TT[] = bt.trades.map((t, i) => {
    const long = t.side === "LONG";
    const eqBefore = eq;
    const contracts = eqBefore / t.entry_price;
    const profit = eqBefore * t.pnl_pct;
    eq += profit; cum += profit;
    let runup: number | null = null, dd: number | null = null;
    if (bars && bars.length) {
      const a = at(bars, t.entry_time), b = at(bars, t.exit_time);
      if (a >= 0 && b >= a && bars[a].time === t.entry_time) {
        let hi = -Infinity, lo = Infinity;
        for (let k = a; k <= b; k++) { hi = Math.max(hi, bars[k].high); lo = Math.min(lo, bars[k].low); }
        runup = Math.max(0, (long ? hi - t.entry_price : t.entry_price - lo) * contracts);
        dd = Math.max(0, (long ? t.entry_price - lo : hi - t.entry_price) * contracts);
      }
    }
    return {
      ...t, n: i + 1, long, eqBefore, contracts, profit, cum,
      commission: contracts * (t.entry_price + t.exit_price) * feeRate,
      bars: Math.max(1, Math.round((t.exit_time - t.entry_time) / tfSec)),
      runup, runupPct: runup != null ? runup / eqBefore : null, dd, ddPct: dd != null ? dd / eqBefore : null,
    };
  });

  // exact daily curve when the backend provides it (equity/buy_hold are downsampled for drawing)
  const daily = bt.equity_daily;
  const e = daily?.length ? daily.map((d) => ({ time: d.time, value: d.equity })) : bt.equity;
  const bh = daily?.length ? daily.map((d) => ({ time: d.time, value: d.buy_hold })) : bt.buy_hold;
  const e0 = e[0]?.value || 1, b0 = bh[0]?.value || 1;
  const start = e[0]?.time ?? 0, end = e[e.length - 1]?.time ?? 0;
  const years = Math.max(1e-9, (end - start) / (365.25 * 86400));
  let peak = -Infinity, maxDD = 0, maxDDPct = 0, trough = Infinity, maxRunup = 0, maxRunupPct = 0;
  const curve = e.map((p, i) => {
    const v = (p.value / e0) * C;
    if (v > peak) peak = v;
    if (v < trough) trough = v;
    const dd = peak - v, ddPct = dd / peak;
    if (dd > maxDD) maxDD = dd;
    if (ddPct > maxDDPct) maxDDPct = ddPct;
    if (v - trough > maxRunup) { maxRunup = v - trough; maxRunupPct = (v - trough) / trough; }
    return { time: p.time, pnl: v - C, bh: ((bh[i]?.value ?? b0) / b0) * C - C, dd: -dd, ddPct: -ddPct };
  });
  const finalEq = curve.length ? curve[curve.length - 1].pnl + C : C;
  const bhReturn = bh.length ? bh[bh.length - 1].value / b0 - 1 : 0;

  // outliers: trades outside Tukey fences (1.5 IQR) on % return
  const rs = trades.map((t) => t.pnl_pct).sort((a, b) => a - b);
  const q = (p: number) => rs[Math.min(rs.length - 1, Math.max(0, Math.floor(p * (rs.length - 1))))] ?? 0;
  const iqr = q(0.75) - q(0.25), lo = q(0.25) - 1.5 * iqr, hi = q(0.75) + 1.5 * iqr;
  const outlier = trades.filter((t) => t.pnl_pct < lo || t.pnl_pct > hi).reduce((s, t) => s + t.profit, 0);

  const m = bt.metrics;
  return {
    C, trades, all: side(trades, C, years), long: side(trades.filter((t) => t.long), C, years), short: side(trades.filter((t) => !t.long), C, years),
    curve, maxDD, maxDDPct, maxRunup, maxRunupPct, bhReturn, bhPnl: bhReturn * C, start, end, years,
    cagr: num(m.cagr) ?? (finalEq > 0 ? (finalEq / C) ** (1 / years) - 1 : null),
    outlier, outlierPct: outlier / C, sharpe: num(m.sharpe), sortino: num(m.sortino), exposure: num(m.exposure),
  };
}

export type Period = "D" | "W" | "Q" | "Y";
export const PERIODS: [Period, string][] = [["D", "Diário"], ["W", "Semanal"], ["Q", "Trimestral"], ["Y", "Anual"]];

/** Bucket start (UTC seconds) for a timestamp. */
export function bucket(t: number, p: Period) {
  const d = new Date(t * 1000);
  if (p === "D") return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()) / 1000;
  if (p === "W") { const day = (d.getUTCDay() + 6) % 7; return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate() - day) / 1000; }
  if (p === "Q") return Date.UTC(d.getUTCFullYear(), Math.floor(d.getUTCMonth() / 3) * 3, 1) / 1000;
  return Date.UTC(d.getUTCFullYear(), 0, 1) / 1000;
}

/** Realized P&L per period (by exit time) split in profit / loss, plus favourable / adverse excursion. */
export function realizedByPeriod(s: Stats, p: Period) {
  const map = new Map<number, { t: number; win: number; loss: number; mfe: number; mae: number }>();
  for (const t of s.trades) {
    const k = bucket(t.exit_time, p);
    const r = map.get(k) ?? { t: k, win: 0, loss: 0, mfe: 0, mae: 0 };
    if (t.profit >= 0) r.win += t.profit; else r.loss += t.profit;
    r.mfe += t.runup ?? 0; r.mae -= t.dd ?? 0;
    map.set(k, r);
  }
  return [...map.values()].sort((a, b) => a.t - b.t);
}

/** Strategy vs buy & hold P&L per period (mark-to-market from the equity curves) and their correlation. */
export function benchByPeriod(s: Stats, p: Period) {
  const out: { t: number; strat: number; bh: number }[] = [];
  let cur = -1, s0 = 0, b0 = 0, sl = 0, bl = 0;
  for (const c of s.curve) {
    const k = bucket(c.time, p);
    if (k !== cur) {
      if (cur >= 0) out.push({ t: cur, strat: sl - s0, bh: bl - b0 });
      cur = k; s0 = sl || c.pnl; b0 = bl || c.bh;
    }
    sl = c.pnl; bl = c.bh;
  }
  if (cur >= 0) out.push({ t: cur, strat: sl - s0, bh: bl - b0 });
  const a = out.map((x) => x.strat), b = out.map((x) => x.bh), ma = mean(a), mb = mean(b);
  let num_ = 0, da = 0, db = 0;
  for (let i = 0; i < a.length; i++) { num_ += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) ** 2; db += (b[i] - mb) ** 2; }
  return { rows: out, corr: da > 0 && db > 0 ? num_ / Math.sqrt(da * db) : null };
}

/** Histogram of trade % returns with 1% (or adaptive) bins. */
export function histogram(s: Stats) {
  const r = s.trades.map((t) => t.pnl_pct * 100);
  if (!r.length) return { bins: [], width: 1, meanWin: null as number | null, meanLoss: null as number | null };
  const lo = Math.floor(Math.max(Math.min(...r), -50)), hi = Math.ceil(Math.min(Math.max(...r), 100));
  const width = hi - lo > 40 ? Math.ceil((hi - lo) / 40) : 1;
  const bins: { x: number; n: number; win: boolean }[] = [];
  for (let x = lo; x < hi + width; x += width) bins.push({ x, n: 0, win: x >= 0 });
  for (const v of r) { const i = Math.min(bins.length - 1, Math.max(0, Math.floor((Math.min(Math.max(v, lo), hi) - lo) / width))); bins[i].n++; }
  const w = r.filter((x) => x > BE * 100), l = r.filter((x) => x < -BE * 100);
  return { bins, width, meanWin: w.length ? mean(w) : null, meanLoss: l.length ? mean(l) : null };
}
