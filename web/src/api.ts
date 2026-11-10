import type { Backtest, Bar, Signal, Strategy } from "./types";

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}
export const getStrategies = () => fetch("/api/strategies").then((r) => j<Strategy[]>(r));
export const getOhlcv = (tf: string) =>
  fetch(`/api/ohlcv?market=perp&tf=${tf}`).then((r) => j<{ holdout_start: number; bars: Bar[] }>(r));
export const getSignal = (strategy: string, tf: string) =>
  fetch(`/api/signal/current?strategy=${strategy}&tf=${tf}`).then((r) => j<Signal>(r));
export const getBacktest = (strategy: string, tf: string, fee: string) =>
  fetch("/api/backtest", {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ strategy, tf, fee }),
  }).then((r) => j<Backtest>(r));

export interface Fee { name: string; fee_rate: number; slippage_bp: number; allow_short: boolean; verified: boolean; note: string }
export const getFees = () => fetch("/api/fees").then((r) => j<Fee[]>(r));
export interface GridResp { empty: boolean; available_from?: number; times: number[]; prices: number[]; bid?: number[][]; ask?: number[][]; long?: number[][]; short?: number[][] }
export interface VpResp { prices: number[]; volume: number[]; poc?: number }
const q = (tf: string, start: number, end: number) =>
  `tf=${tf}&start=${new Date(start * 1000).toISOString().slice(0, 19)}&end=${new Date(end * 1000).toISOString().slice(0, 19)}`;
export const getBook = (tf: string, s: number, e: number) => fetch(`/api/heatmap/book?${q(tf, s, e)}`).then((r) => j<GridResp>(r));
export const getLiq = (tf: string, s: number, e: number) => fetch(`/api/heatmap/liq?${q(tf, s, e)}`).then((r) => j<GridResp>(r));
export const getVp = (tf: string, s: number, e: number) => fetch(`/api/volume_profile?${q(tf, s, e)}`).then((r) => j<VpResp>(r));
