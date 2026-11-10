/**
 * Live market data from Binance USDT-M public WebSockets (no key).
 *
 * One shared socket multiplexes every stream the UI needs (per-pane @kline_<tf> + @aggTrade, and the
 * all-market !miniTicker@arr); streams are (un)subscribed on the fly, the socket reconnects with
 * exponential backoff, and a "resync" event tells panes to backfill gaps from /api/ohlcv after a
 * reconnect or tab refocus.
 *
 * Endpoint (verified 2026-11-10): only wss://fstream.binance.com/market/stream delivers kline + aggTrade;
 * /stream, /public/stream and /ws accept the socket but stay silent. A silent socket rotates to the next URL.
 *
 * Latency: Binance event time E is compared against a server-synchronised clock (local clocks drift; this
 * machine was 25 s behind), giving the true wire+render lag shown next to the live dot.
 */
import type { Bar } from "./types";

export interface LiveKline extends Bar { closed: boolean; E: number }
export interface LiveTrade { p: number; q: number; T: number; E: number }
export interface Tick { symbol: string; price: number; open: number; pct: number; vol: number; ts: number }
export type LiveStatus = "connecting" | "live" | "stale" | "offline";

const VERSION = 3;
const URLS = ["wss://fstream.binance.com/market/stream", "wss://fstream.binance.com/stream"];
const TICKER = "!miniTicker@arr";
const STALE_MS = 15000;
const DEV = import.meta.env.DEV;

type Cb = (data: unknown) => void;

class LiveHub {
  readonly version = VERSION;
  private ws: WebSocket | null = null;
  private subs = new Map<string, Set<Cb>>();
  private tickSubs = new Set<(t: Map<string, Tick>) => void>();
  private statusSubs = new Set<(s: LiveStatus) => void>();
  private resyncSubs = new Set<() => void>();
  private lagSubs = new Set<(ms: number | null) => void>();
  private active = new Set<string>();
  private ticks = new Map<string, Tick>();
  private retry = 0;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private iv: ReturnType<typeof setInterval> | null = null;
  private lastMsg = 0;
  private reqId = 1;
  private hadOpen = false;
  private urlIdx = 0;
  private gotData = false;
  private closed = false;
  /** serverTime - localTime (ms). */
  private offset: number | null = null;
  private minRaw = Infinity;            // min(local - E) fallback when the time endpoint is unreachable
  private lags: number[] = [];          // render lags (ms) since last report
  lag: number | null = null;
  status: LiveStatus = "connecting";

  constructor() {
    if (typeof window === "undefined") return;
    this.iv = setInterval(() => this.housekeep(), 1000);
    document.addEventListener("visibilitychange", this.onVis);
    window.addEventListener("online", this.onOnline);
    this.syncClock();
    this.connect();
  }

  private onVis = () => {
    if (this.closed || document.visibilityState !== "visible") return;
    if (!this.ws || this.ws.readyState > 1) this.connect();
    this.resyncSubs.forEach((f) => f());
  };
  private onOnline = () => this.connect();

  /** Close everything (used when HMR swaps in a newer hub). */
  close() {
    this.closed = true;
    if (this.iv) clearInterval(this.iv);
    if (this.timer) clearTimeout(this.timer);
    document.removeEventListener("visibilitychange", this.onVis);
    window.removeEventListener("online", this.onOnline);
    const ws = this.ws; this.ws = null; ws?.close();
  }

  private async syncClock() {
    try {
      const t0 = Date.now();
      const r = await fetch("https://fapi.binance.com/fapi/v1/time");
      const t1 = Date.now();
      const { serverTime } = (await r.json()) as { serverTime: number };
      this.offset = serverTime - (t0 + t1) / 2;
      if (DEV) console.info(`[live] clock offset vs Binance: ${Math.round(this.offset)} ms (rtt ${t1 - t0} ms)`);
    } catch { /* fall back to min-latency estimate */ }
    setTimeout(() => !this.closed && this.syncClock(), 10 * 60_000);
  }
  /** Server-synchronised "now" in ms. */
  now() { return Date.now() + (this.offset ?? -(isFinite(this.minRaw) ? this.minRaw : 0)); }

  /** Panes call this right after series.update() with the event time of the newest data drawn. */
  noteRender(E: number) { this.lags.push(this.now() - E); }

  private housekeep() {
    const ws = this.ws, t = Date.now();
    if (ws && ws.readyState === 1 && !this.gotData && t - this.lastMsg > 6000) {
      this.urlIdx = (this.urlIdx + 1) % URLS.length; ws.close(); return;
    }
    if (this.status === "live" && t - this.lastMsg > STALE_MS) this.setStatus("stale");
    if (this.status === "stale" && t - this.lastMsg > STALE_MS * 2) this.ws?.close();
    if (this.lags.length) {
      const a = this.lags.sort((x, y) => x - y);
      this.lag = Math.max(0, Math.round(a[Math.floor(a.length / 2)]));
      if (DEV && Math.random() < 0.2) console.debug(`[live] render lag p50 ${this.lag} ms, p90 ${Math.round(a[Math.floor(a.length * 0.9)])} ms, n=${a.length}/s`);
      this.lags = [];
    } else if (this.status !== "live") this.lag = null;
    this.lagSubs.forEach((f) => f(this.lag));
  }

  private setStatus(s: LiveStatus) {
    if (s === this.status) return;
    this.status = s;
    this.statusSubs.forEach((f) => f(s));
  }

  private wanted() { return [TICKER, ...this.subs.keys()]; }

  private connect() {
    if (this.closed) return;
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
    if (this.ws && this.ws.readyState <= 1) return;
    this.setStatus("connecting");
    let ws: WebSocket;
    try { ws = new WebSocket(`${URLS[this.urlIdx]}?streams=${this.wanted().join("/")}`); } catch { this.schedule(); return; }
    this.ws = ws;
    this.active = new Set(this.wanted());
    ws.onopen = () => {
      this.lastMsg = Date.now();
      this.gotData = false;
      this.sync();
      if (this.hadOpen) this.resyncSubs.forEach((f) => f());
      this.hadOpen = true;
    };
    ws.onmessage = (ev) => this.onMsg(ev.data as string);
    ws.onclose = () => { if (this.ws === ws) { this.ws = null; this.setStatus("offline"); this.schedule(); } };
    ws.onerror = () => ws.close();
  }

  private schedule() {
    if (this.timer || this.closed) return;
    const ms = Math.min(30000, 500 * 2 ** this.retry++) * (0.8 + Math.random() * 0.4);
    this.timer = setTimeout(() => { this.timer = null; this.connect(); }, ms);
  }

  private sync() {
    const ws = this.ws;
    if (!ws || ws.readyState !== 1) return;
    const want = new Set(this.wanted());
    const add = [...want].filter((s) => !this.active.has(s));
    const del = [...this.active].filter((s) => !want.has(s));
    if (add.length) ws.send(JSON.stringify({ method: "SUBSCRIBE", params: add, id: this.reqId++ }));
    if (del.length) ws.send(JSON.stringify({ method: "UNSUBSCRIBE", params: del, id: this.reqId++ }));
    this.active = want;
  }

  private onMsg(raw: string) {
    let m: { stream?: string; data?: { E?: number } };
    try { m = JSON.parse(raw); } catch { return; }
    if (!m.stream || !m.data) return;
    this.lastMsg = Date.now();
    if (!this.gotData) { this.gotData = true; this.retry = 0; }
    if (this.status !== "live") this.setStatus("live");
    if (m.stream === TICKER) {
      for (const t of m.data as unknown as { s: string; c: string; o: string; q: string; E: number }[]) {
        const price = +t.c, open = +t.o;
        this.ticks.set(t.s, { symbol: t.s, price, open, pct: open ? price / open - 1 : 0, vol: +t.q, ts: t.E });
      }
      this.tickSubs.forEach((f) => f(this.ticks));
      return;
    }
    if (m.data.E) this.minRaw = Math.min(this.minRaw, Date.now() - m.data.E);
    this.subs.get(m.stream)?.forEach((f) => f(m.data));
  }

  private sub(stream: string, cb: Cb) {
    let s = this.subs.get(stream);
    if (!s) { s = new Set(); this.subs.set(stream, s); }
    s.add(cb);
    this.sync();
    return () => {
      const set = this.subs.get(stream);
      if (!set) return;
      set.delete(cb);
      if (!set.size) { this.subs.delete(stream); this.sync(); }
    };
  }

  /** Kline stream: bar open/close boundaries and volume truth. */
  kline(symbol: string, tf: string, cb: (k: LiveKline) => void) {
    return this.sub(`${symbol.toLowerCase()}@kline_${tf}`, (d) => {
      const { k, E } = d as { E: number; k: { t: number; o: string; h: string; l: string; c: string; v: string; x: boolean } };
      cb({ time: Math.floor(k.t / 1000), open: +k.o, high: +k.h, low: +k.l, close: +k.c, volume: +k.v, closed: k.x, E });
    });
  }
  /** Aggregated trades: tick-by-tick price for the forming candle. */
  trades(symbol: string, cb: (t: LiveTrade) => void) {
    return this.sub(`${symbol.toLowerCase()}@aggTrade`, (d) => {
      const t = d as { p: string; q: string; T: number; E: number };
      cb({ p: +t.p, q: +t.q, T: t.T, E: t.E });
    });
  }
  ticker(cb: (t: Map<string, Tick>) => void) { this.tickSubs.add(cb); if (this.ticks.size) cb(this.ticks); return () => { this.tickSubs.delete(cb); }; }
  onStatus(cb: (s: LiveStatus) => void) { this.statusSubs.add(cb); return () => { this.statusSubs.delete(cb); }; }
  onResync(cb: () => void) { this.resyncSubs.add(cb); return () => { this.resyncSubs.delete(cb); }; }
  onLag(cb: (ms: number | null) => void) { this.lagSubs.add(cb); return () => { this.lagSubs.delete(cb); }; }
}

/** Bar open time (s) containing timestamp t (s) for a timeframe; weeks open Monday 00:00 UTC like Binance. */
export function barOpen(t: number, tfSec: number) {
  if (tfSec === 604800) return Math.floor((t - 345600) / 604800) * 604800 + 345600;
  return Math.floor(t / tfSec) * tfSec;
}

const g = globalThis as unknown as { __btcLive?: LiveHub };
// Singleton across HMR; a newer module version replaces (and closes) an older hub.
if (g.__btcLive && g.__btcLive.version !== VERSION) { try { g.__btcLive.close?.(); } catch { /* old hub */ } g.__btcLive = undefined; }
export const live: LiveHub = g.__btcLive ?? (g.__btcLive = new LiveHub());
