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
