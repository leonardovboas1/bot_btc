import type { Backtest, Bar, Signal, Strategy, SymbolInfo } from "./types";

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}
const get = <T>(url: string, signal?: AbortSignal) => fetch(url, { signal }).then((r) => j<T>(r));
const iso = (t: number) => new Date(t * 1000).toISOString().slice(0, 19);
const sp = (s: string) => `symbol=${encodeURIComponent(s)}`;

export const FALLBACK_SYMBOLS: SymbolInfo[] = [{ symbol: "BTCUSDT", base: "BTC", tfs: ["5m", "15m", "1h", "4h", "1d", "1w"], has_liq: true }];
/** Symbols announced but not yet ingested by the backend: shown in the watchlist (live price only). */
export const COMING = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "NEARUSDT", "FETUSDT", "ADAUSDT", "XRPUSDT"];
export const getSymbols = async (): Promise<SymbolInfo[]> => {
  try {
    const r = await fetch("/api/symbols");
    if (!r.ok) return FALLBACK_SYMBOLS;
    const d = (await r.json()) as SymbolInfo[];
    return Array.isArray(d) && d.length ? d : FALLBACK_SYMBOLS;
  } catch { return FALLBACK_SYMBOLS; }
};
export const getStrategies = () => get<Strategy[]>("/api/strategies");
export const getOhlcv = (symbol: string, tf: string, start?: number, end?: number, signal?: AbortSignal) =>
  get<{ holdout_start: number; bars: Bar[] }>(
    `/api/ohlcv?market=perp&tf=${tf}&${sp(symbol)}${start != null ? `&start=${iso(start)}` : ""}${end != null ? `&end=${iso(end)}` : ""}`, signal);
export type Source = "real" | "ha";
export const getSignal = (symbol: string, strategy: string, tf: string, signal?: AbortSignal, source: Source = "real") =>
  get<Signal>(`/api/signal/current?strategy=${encodeURIComponent(strategy)}&tf=${tf}&${sp(symbol)}${source === "ha" ? "&source=ha" : ""}`, signal);

const btCache = new Map<string, Backtest>();
export const getBacktest = async (symbol: string, strategy: string, tf: string, fee: string, signal?: AbortSignal, source: Source = "real") => {
  const key = [symbol, strategy, tf, fee, source].join("|");
  const hit = btCache.get(key);
  if (hit) return hit;
  const b = await fetch("/api/backtest", {
    method: "POST", headers: { "content-type": "application/json" }, signal,
    body: JSON.stringify(source === "ha" ? { symbol, strategy, tf, fee, source } : { symbol, strategy, tf, fee }),
  }).then((r) => j<Backtest>(r));
  btCache.set(key, b);
  return b;
};

export interface Fee { name: string; fee_rate: number; slippage_bp: number; allow_short: boolean; verified: boolean; note: string }
export const getFees = () => get<Fee[]>("/api/fees");
export interface GridResp { empty: boolean; available_from?: number; times: number[]; prices: number[]; bid?: number[][]; ask?: number[][]; long?: number[][]; short?: number[][] }
export interface VpResp { prices: number[]; volume: number[]; poc?: number }
const q = (symbol: string, tf: string, s: number, e: number) => `tf=${tf}&start=${iso(s)}&end=${iso(e)}&${sp(symbol)}`;
export const getBook = (sym: string, tf: string, s: number, e: number, sig?: AbortSignal) => get<GridResp>(`/api/heatmap/book?${q(sym, tf, s, e)}`, sig);
export const getLiq = (sym: string, tf: string, s: number, e: number, sig?: AbortSignal) => get<GridResp>(`/api/heatmap/liq?${q(sym, tf, s, e)}`, sig);
export const getVp = (sym: string, tf: string, s: number, e: number, sig?: AbortSignal) => get<VpResp>(`/api/volume_profile?${q(sym, tf, s, e)}`, sig);

// ---------------------------------------------------------------- bar cache (LRU) + idle prefetch
const ALL = ["5m", "15m", "1h", "4h", "1d", "1w"];
const barLru = new Map<string, { bars: Bar[]; holdout: number | null }>();
const LRU_MAX = 24;
export function cachedBars(symbol: string, tf: string) {
  const k = symbol + "|" + tf, v = barLru.get(k);
  if (v) { barLru.delete(k); barLru.set(k, v); }
  return v ?? null;
}
export function putBars(symbol: string, tf: string, bars: Bar[], holdout: number | null) {
  const k = symbol + "|" + tf;
  barLru.delete(k);
  barLru.set(k, { bars, holdout });
  while (barLru.size > LRU_MAX) barLru.delete(barLru.keys().next().value as string);
}
/** Last 5000 bars incl. the live tail (no start/end: the backend appends Binance REST bars). */
export const getRecent = (symbol: string, tf: string, signal?: AbortSignal) =>
  get<{ holdout_start: number; bars: Bar[] }>(`/api/ohlcv?market=perp&tf=${tf}&${sp(symbol)}`, signal);
const inflight = new Set<string>();
/** Warm the cache for the neighbouring timeframes when the browser is idle. */
export function prefetchNear(symbol: string, tf: string) {
  const i = ALL.indexOf(tf);
  const near = [ALL[i - 1], ALL[i + 1]].filter(Boolean) as string[];
  const idle = (window as unknown as { requestIdleCallback?: (f: () => void) => void }).requestIdleCallback ?? ((f: () => void) => setTimeout(f, 300));
  for (const t of near) {
    const k = symbol + "|" + t;
    if (barLru.has(k) || inflight.has(k)) continue;
    idle(() => {
      inflight.add(k);
      getRecent(symbol, t).then((r) => { if (!barLru.has(k)) putBars(symbol, t, r.bars, r.holdout_start ?? null); })
        .catch(() => {}).finally(() => inflight.delete(k));
    });
  }
}
