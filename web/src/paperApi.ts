/** Forward paper trading endpoints (/paper/*). */
const get = async <T,>(url: string, signal?: AbortSignal): Promise<T> => {
  const r = await fetch(url, { signal });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
};

export interface PaperStrategy {
  id: string; name: string; symbol: string; tf: string; source: "real" | "ha"; start_ts: number; first_bar: number | null;
  position: "LONG" | "SHORT" | "FLAT"; qty: number; entry_px: number; unrealized: number; n_trades: number; n_entries: number;
  equity: number | null; equity_start: number | null; price: number | null; return: number | null; bh_return: number | null;
  mdd: number | null; bh_mdd: number | null; wins: number; closed: number; days: number;
  last_signal_time: number | null; last_signal_close_time: number | null;
  last_signal: { long: number; short: number; exit_long: number; exit_short: number } | null;
}
export interface PaperStatus {
  running: boolean; last_run: number | null; lag_s: number | null; current_errors?: string[];
  errors: { time: number; error: string }[]; last_bars: Record<string, number | null>;
}
/** One simulated fill: drawn on the chart as a hollow marker. */
export interface PaperFill { time: number; price: number; buy: boolean; long: boolean; entry: boolean }
export interface PaperPoint { time: number; equity: number; buy_hold: number; position: number }

export const getPaperStrategies = (signal?: AbortSignal) => get<PaperStrategy[]>("/api/paper/strategies", signal);
export const getPaperStatus = (signal?: AbortSignal) => get<PaperStatus>("/api/paper/status", signal);
export const getPaperEquity = (id: string, signal?: AbortSignal) =>
  get<{ start: number | null; points: PaperPoint[] }>(`/api/paper/equity?strategy=${encodeURIComponent(id)}`, signal);
export const getPaperFills = async (id: string, signal?: AbortSignal): Promise<PaperFill[]> => {
  const r = await get<{ fills: { bar_time: number; price: number; side: string; position_side: number; leg: string }[] }>(
    `/api/paper/ledger?strategy=${encodeURIComponent(id)}&limit=1`, signal);
  return r.fills.map((f) => ({ time: f.bar_time, price: f.price, buy: f.side === "buy", long: f.position_side === 1, entry: f.leg === "entry" }));
};
