import type { Bar } from "../types";

export const RANGES = ["1d", "3d", "1w", "2w", "1m", "3m", "6m"] as const;
export type LiqRange = (typeof RANGES)[number];

export interface LiqMap {
  symbol: string; range: string; resolution: string; step: number; start: number; end: number;
  times: number[]; prices: number[]; scale: number;
  /** T x P row-major, u8 = round(255*sqrt(min(usd/scale,1))) */
  q: Uint8Array; T: number; P: number;
  candles: Bar[];
  active: { p: number; usd: number; side: "long" | "short" }[];
  events: { t: number; p: number; usd: number; side: "long" | "short" }[];
  note: string; venues: string[];
}

interface Raw {
  symbol: string; range: string; resolution: string; step: number; start: number; end: number;
  times: number[]; prices: number[]; scale: number; shape: [number, number]; grid: string;
  candles: { o: number[]; h: number[]; l: number[]; c: number[]; v: number[] };
  active: LiqMap["active"]; events: LiqMap["events"]; note: string; venues: string[];
}

const cache = new Map<string, { at: number; v: LiqMap }>();

export async function getLiqMap(symbol: string, range: string, venues: string[], signal?: AbortSignal): Promise<LiqMap> {
  const key = `${symbol}|${range}|${venues.join(",")}`;
  const hit = cache.get(key);
  if (hit && Date.now() - hit.at < 5 * 60_000) return hit.v;
  const r = await fetch(`/api/heatmap/liq?symbol=${encodeURIComponent(symbol)}&range=${range}&venues=${venues.join(",")}`, { signal });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  const d = (await r.json()) as Raw;
  const bin = atob(d.grid);
  const q = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) q[i] = bin.charCodeAt(i);
  const c = d.candles;
  const v: LiqMap = {
    ...d, q, T: d.shape[0], P: d.shape[1],
    candles: d.times.map((t, i) => ({ time: t, open: c.o[i], high: c.h[i], low: c.l[i], close: c.c[i], volume: c.v[i] })),
  };
  cache.set(key, { at: Date.now(), v });
  return v;
}
