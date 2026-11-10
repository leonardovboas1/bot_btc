import { memo, useEffect, useRef, useState } from "react";
import {
  CandlestickSeries, HistogramSeries, LineSeries, ColorType, CrosshairMode, LineStyle, PriceScaleMode, createChart, createSeriesMarkers, createTextWatermark,
  type IChartApi, type ISeriesApi, type ITextWatermarkPluginApi, type ISeriesMarkersPluginApi, type MouseEventParams, type SeriesMarker, type Time, type UTCTimestamp,
} from "lightweight-charts";
import type { Backtest, Bar, Strategy } from "../types";
import { IND_TFS, SCOL, buildMarkers, buildTrades, getIndicator, reasonPt, type DrawTrade, type IndData } from "./strategyDraw";
import { GridLayer, LayersPrimitive, VpLayer } from "../layers";
import { cachedBars, getBacktest, getBook, getOhlcv, getRecent, getVp, prefetchNear, putBars } from "../api";
import { barOpen, live } from "../live";
import { ALL_TFS, TF_LABEL, TF_SEC, decimals, fmtDate, fmtPct, fmtPrice, fmtVol, isAbort } from "../util";
import { OverlayPrimitive } from "./overlay";
import { drawings, newId, type Drawing, type Tool } from "./drawings";
import type { SyncBus } from "./sync";
import LiqMode, { type LiqHost } from "../liq/LiqControls";

const T = (n: number) => n as UTCTimestamp;
const N_BARS = 1200;
export const COL = { up: "#0ea894", dn: "#f25c54", eq: "#4c8dff", bh: "#5d6676", text: "#5d6676", grid: "#000000", line: "#1c222d", bg: "#000000" };

const chartOpts = {
  layout: {
    background: { type: ColorType.Solid, color: COL.bg }, textColor: "#7c8596", fontSize: 11,
    fontFamily: "'JetBrains Mono Variable', ui-monospace, monospace", attributionLogo: false,
    panes: { separatorColor: COL.line, separatorHoverColor: "rgba(255,176,32,0.25)", enableResize: true },
  },
  grid: { vertLines: { visible: false }, horzLines: { visible: false } },  // pure black, no grid (also avoids black lines cutting the heatmap)
  crosshair: {
    mode: CrosshairMode.Normal,
    vertLine: { color: "rgba(230,233,239,0.28)", width: 1 as const, style: LineStyle.Dashed, labelBackgroundColor: "#262d3a" },
    horzLine: { color: "rgba(230,233,239,0.28)", width: 1 as const, style: LineStyle.Dashed, labelBackgroundColor: "#262d3a" },
  },
  timeScale: { timeVisible: true, borderColor: COL.line, rightOffset: 8, barSpacing: 7, minBarSpacing: 0.5 },
  rightPriceScale: { borderColor: COL.line, scaleMargins: { top: 0.08, bottom: 0.2 } },
  localization: { locale: "pt-BR" },
  autoSize: true,
};

export type PaneMode = "candles" | "liq";
export interface PaneProps {
  idx: number; symbol: string; tf: string; tfs: string[]; mode: PaneMode; strategy: string; fee: string;
  active: boolean; multi: boolean; ha: boolean; log?: boolean; source: "real" | "ha"; showEq: boolean; details: boolean; showBook: boolean; showVp: boolean; hasLiq: boolean;
  tool: Tool; bus: SyncBus; spec: Strategy | null; focus: { n: number; t0: number; t1: number; seq: number } | null;
  onActivate(i: number): void; onBt(i: number, b: Backtest | null): void; onErr(e: string): void;
  onTf(i: number, tf: string): void; onMode(i: number, m: PaneMode): void; onTool(t: Tool): void;
  onBarClose(symbol: string, tf: string): void;
}

function Pane(props: PaneProps) {
  const { idx, symbol, tf, mode, strategy, fee, active, multi, showEq, details, showBook, showVp, hasLiq, tool } = props;
  const pr = useRef(props);
  pr.current = props;
  const el = useRef<HTMLDivElement>(null);
  const legend = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const candle = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const vol = useRef<ISeriesApi<"Histogram"> | null>(null);
  const eqS = useRef<ISeriesApi<"Line"> | null>(null);
  const bhS = useRef<ISeriesApi<"Line"> | null>(null);
  const markers = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const prim = useRef<LayersPrimitive | null>(null);
  const wm = useRef<ITextWatermarkPluginApi<Time> | null>(null);
  const ovl = useRef<OverlayPrimitive | null>(null);
  const bars = useRef<Bar[]>([]);
  const shown = useRef<Bar[]>([]);         // what the chart displays (bars, or liq-mode candles)
  const frozen = useRef(false);            // liq mode owns the candles
  const exhausted = useRef(false);
  const extending = useRef(false);
  const barsAc = useRef<AbortController | null>(null);
  const hover = useRef(false);
  const draft = useRef<Drawing | null>(null);
  const layerObj = useRef({ book: new GridLayer("book"), vp: new VpLayer("vp") });
  const cov = useRef<Record<string, { key: string; from: number; to: number } | undefined>>({});
  const layerAc = useRef<Record<string, AbortController | undefined>>({});
  const bt = useRef<Backtest | null>(null);
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(true);
  const [note, setNote] = useState("");
  const [btState, setBtState] = useState<"idle" | "busy" | "ok" | "err">("idle");
  const [loadedKey, setLoadedKey] = useState("");  // candles painted for symbol|tf: gates the backtest request
  const [tfOpen, setTfOpen] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [eqTop, setEqTop] = useState<number | null>(null);

  // ---- time <-> logical index over the displayed bars (extrapolates past both ends)
  const logicalOf = (t: number) => {
    const b = shown.current, n = b.length, step = TF_SEC[pr.current.tf] || 3600;
    if (!n) return null;
    if (t <= b[0].time) return (t - b[0].time) / step;
    if (t >= b[n - 1].time) return n - 1 + (t - b[n - 1].time) / step;
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (b[m].time <= t) lo = m; else hi = m; }
    return lo + (t - b[lo].time) / (b[hi].time - b[lo].time);
  };
  const timeOf = (l: number) => {
    const b = shown.current, n = b.length, step = TF_SEC[pr.current.tf] || 3600;
    if (!n) return null;
    if (l <= 0) return b[0].time + l * step;
    if (l >= n - 1) return b[n - 1].time + (l - (n - 1)) * step;
    const i = Math.floor(l);
    return b[i].time + (l - i) * (b[i + 1].time - b[i].time);
  };
  const xOf = (t: number) => {
    const l = logicalOf(t);
    const ts = chart.current?.timeScale();
    if (l == null || !ts) return null;
    // LWC's logicalToCoordinate only handles integer indices (fractions return 0): interpolate ourselves
    const i = Math.floor(l), f = l - i;
    const a = ts.logicalToCoordinate(i as never), b = ts.logicalToCoordinate((i + 1) as never);
    if (a == null || b == null) return null;
    return a + (b - a) * f;
  };

  const setLegend = (b?: Bar | null, prev?: Bar | null) => {
    const e = legend.current;
    if (!e) return;
    if (!b) { e.innerHTML = ""; return; }
    const d = decimals(b.close);
    const ch = prev ? b.close / prev.close - 1 : b.close / b.open - 1;
    const cls = b.close >= b.open ? "up" : "dn";
    e.innerHTML =
      `<span><i>A</i>${fmtPrice(b.open, d)}</span><span><i>M</i>${fmtPrice(b.high, d)}</span>` +
      `<span><i>m</i>${fmtPrice(b.low, d)}</span><span><i>F</i><b class="${cls}">${fmtPrice(b.close, d)}</b></span>` +
      `<span class="${ch >= 0 ? "up" : "dn"}">${fmtPct(ch, true, 2)}</span><span class="vol"><i>Vol</i>${fmtVol(b.volume)}</span>`;
  };
  const legendLast = () => { const b = shown.current; setLegend(b[b.length - 1], b[b.length - 2]); };

  const volPt = (x: Bar) => ({ time: T(x.time), value: x.volume, color: x.close >= x.open ? "rgba(14,168,148,0.2)" : "rgba(242,92,84,0.2)" });
  const candPt = (x: Bar) => ({ time: T(x.time), open: x.open, high: x.high, low: x.low, close: x.close });
  // Heikin Ashi rendering (client-side): haArr mirrors shown candles when HA is on
  const haArr = useRef<Bar[]>([]);
  const trades = useRef<DrawTrade[]>([]);
  const labelsOn = useRef(false);
  const ind = useRef<IndData[]>([]);
  const indS = useRef<ISeriesApi<"Line">[]>([]);
  const indLeg = useRef<HTMLDivElement>(null);
  const tip = useRef<HTMLDivElement>(null);
  const haOf = (x: Bar, prev: Bar | undefined): Bar => {
    const c = (x.open + x.high + x.low + x.close) / 4;
    const o = prev ? (prev.open + prev.close) / 2 : (x.open + x.close) / 2;
    return { time: x.time, open: o, high: Math.max(x.high, o, c), low: Math.min(x.low, o, c), close: c, volume: x.volume };
  };
  const useHa = () => pr.current.ha && !frozen.current;

  const fmtKey = useRef(-1);
  const show = (b: Bar[], keepRange: boolean) => {
    const c = chart.current;
    const d = b.length ? decimals(b[b.length - 1].close) : 2;
    if (d !== fmtKey.current && candle.current) {
      fmtKey.current = d;
      candle.current.applyOptions({ priceFormat: { type: "custom", minMove: 10 ** -d, formatter: (p: number) => fmtPrice(p, d) } });
    }
    // keep the viewport anchored on the same bars when older data is prepended
    const lr = keepRange ? c?.timeScale().getVisibleLogicalRange() : null;
    const shift = keepRange && shown.current.length ? b.length - shown.current.length : 0;
    shown.current = b;
    if (useHa()) {
      const h: Bar[] = [];
      for (let i = 0; i < b.length; i++) h.push(haOf(b[i], h[i - 1]));
      haArr.current = h;
      candle.current?.setData(h.map(candPt));
    } else { haArr.current = []; candle.current?.setData(b.map(candPt)); }
    vol.current?.setData(b.map(volPt));
    if (lr && c) c.timeScale().setVisibleLogicalRange({ from: lr.from + shift, to: lr.to + shift });
    ovl.current?.update();
    prim.current?.invalidate();
  };

  const extendLeft = async () => {
    if (frozen.current || extending.current || exhausted.current || !bars.current.length) return;
    const { symbol: s, tf: f } = pr.current;
    const first = bars.current[0].time;
    const ac = barsAc.current;
    extending.current = true;
    try {
      const r = await getOhlcv(s, f, first - N_BARS * TF_SEC[f], first, ac?.signal);
      if (ac?.signal.aborted) return;
      const older = r.bars.filter((b) => b.time < first);
      if (!older.length) { exhausted.current = true; return; }
      bars.current = [...older, ...bars.current];
      if (!frozen.current) { show(bars.current, true); drawBt(); drawInd(); }
    } catch (e) { if (!isAbort(e)) pr.current.onErr(String(e)); }
    finally { extending.current = false; }
  };

  /** Merge bars newer than (or equal to) the last one in place: live ticks and gap backfill. */
  const mergeTail = (nb: Bar[]) => {
    const b = bars.current;
    if (!b.length) return;
    for (const x of nb) {
      const last = b[b.length - 1];
      if (x.time < last.time) continue;
      if (x.time === last.time) b[b.length - 1] = x; else b.push(x);
      if (!frozen.current) {
        let cp = x;
        if (useHa() && haArr.current.length) {
          const h = haArr.current, same = h[h.length - 1].time === x.time;
          cp = haOf(x, same ? h[h.length - 2] : h[h.length - 1]);
          if (same) h[h.length - 1] = cp; else h.push(cp);
        }
        candle.current?.update(candPt(cp)); vol.current?.update(volPt(x));
      }
    }
    if (!frozen.current && !hover.current) legendLast();
  };

  /** HTTP backfill: insert missing bars (gap after reconnect/refocus), never overwrite the live forming bar. */
  const mergeGap = (nb: Bar[]) => {
    const b = bars.current;
    if (!b.length) return;
    const last = b[b.length - 1].time;
    const have = new Set(b.slice(-nb.length - 5).map((x) => x.time));
    const mid = nb.filter((x) => x.time < last && !have.has(x.time));
    const after = nb.filter((x) => x.time > last);
    if (mid.length) {
      bars.current = [...b, ...mid].sort((x, y) => x.time - y.time);
      if (!frozen.current) show(bars.current, true);
    }
    if (after.length) mergeTail(after);
  };

  const refreshLayers = () => {
    const c = chart.current, p = prim.current;
    if (!c || !p) return;
    const { symbol: s, tf: f, showBook: sb, showVp: sv } = pr.current;
    const want: Record<string, boolean> = { book: sb && !frozen.current, vp: sv && !frozen.current };
    const vr = c.timeScale().getVisibleRange();
    const key = s + "|" + f;
    for (const k of Object.keys(want)) {
      const cv = cov.current[k];
      if (!want[k] || (cv && cv.key !== key)) {
        layerAc.current[k]?.abort(); cov.current[k] = undefined; p.removeLayer(k);
        if (k === "book") setNote("");
      }
    }
    if (!vr) return;
    const from = Number(vr.from), to = Number(vr.to), span = Math.max(to - from, TF_SEC[f] * 20);
    for (const k of Object.keys(want)) {
      if (!want[k]) continue;
      const cv = cov.current[k];
      if (cv && cv.key === key && from >= cv.from && to <= cv.to) continue;
      layerAc.current[k]?.abort();
      const ac = new AbortController();
      layerAc.current[k] = ac;
      const s0 = Math.floor(from - span * 0.5), e0 = Math.ceil(to + span * 0.5);
      cov.current[k] = { key, from: s0, to: e0 };
      const L = layerObj.current;
      const fail = (e: unknown) => { if (!isAbort(e)) { cov.current[k] = undefined; setNote(k === "book" ? "Livro: indisponível neste TF" : "Volume profile: indisponível neste TF"); } };
      if (k === "book") {
        getBook(s, f, s0, e0, ac.signal).then((r) => {
          if (r.empty || !r.bid || !r.ask) { p.removeLayer("book"); setNote("Livro de ofertas sem dados neste período"); return; }
          setNote(""); L.book.setData({ times: r.times, prices: r.prices, a: r.bid, b: r.ask, colA: COL.up, colB: COL.dn }); p.setLayer(L.book);
        }).catch(fail);
      } else {
        getVp(s, f, s0, e0, ac.signal).then((r) => { L.vp.setData(r); p.setLayer(L.vp); }).catch(fail);
      }
    }
  };

  /** Strategy indicator overlays: lines on the price pane + regime-coloured band fill (clipped to loaded bars). */
  const drawInd = () => {
    const m = chart.current;
    if (!m) return;
    for (const x of indS.current) m.removeSeries(x);
    indS.current = [];
    if (ovl.current) ovl.current.bands = [];
    if (frozen.current || !shown.current.length) { ovl.current?.update(); indLegend(null); return; }
    const t0 = shown.current[0].time;
    for (const d of ind.current) {
      const ov = d.series.filter((x) => x.kind === "overlay");
      const reg = d.regime ? new Map(d.regime.map((p) => [p.time, p.value > 0])) : null;
      for (const sr of ov) {
        const main = /filt|mid|basis|ma$|ema|sma/.test(sr.key);
        const band = !main;
        const ls = m.addSeries(LineSeries, {
          color: main ? "rgba(230,233,239,0.92)" : "rgba(154,163,178,0.6)", lineWidth: main ? 2 : 1,
          priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
        });
        ls.setData(sr.data.filter((p) => p.time >= t0).map((p) => ({
          time: T(p.time), value: p.value,
          ...(band && reg ? { color: reg.get(p.time) ? "rgba(14,168,148,0.85)" : "rgba(242,92,84,0.8)" } : {}),
        })));
        indS.current.push(ls);
      }
      const hi = ov.find((x) => /hband|upper/.test(x.key)), lo = ov.find((x) => /lband|lower/.test(x.key));
      if (hi && lo && ovl.current) {
        const lm = new Map(lo.data.map((p) => [p.time, p.value]));
        ovl.current.bands = hi.data.filter((p) => p.time >= t0 && lm.has(p.time))
          .map((p) => ({ time: p.time, hi: p.value, lo: lm.get(p.time)!, up: reg ? !!reg.get(p.time) : true }));
      }
    }
    ovl.current?.update();
    indLegend(null);
  };
  /** Legend rows (top-left) with each indicator's values at the crosshair time (or the latest). */
  const indLegend = (t: number | null) => {
    const e = indLeg.current;
    if (!e) return;
    const P = pr.current.spec;
    if (frozen.current || !P) { e.innerHTML = ""; return; }
    const val = (a: { time: number; value: number }[]) => {
      if (!a.length) return null;
      if (t == null) return a[a.length - 1].value;
      let lo = 0, hi = a.length - 1, r = -1;
      while (lo <= hi) { const mm = (lo + hi) >> 1; if (a[mm].time <= t) { r = mm; lo = mm + 1; } else hi = mm - 1; }
      return r < 0 ? null : a[r].value;
    };
    const rows = ind.current.map((d) => {
      const params = (P.indicators as Record<string, Record<string, unknown>> | undefined)?.[d.name];
      const ps = params && Object.keys(params).length ? "<em>" + Object.values(params).join(" ") + "</em>" : "";
      const vals = d.series.map((x) => {
        const v = val(x.data);
        return v == null ? "" : '<span class="' + x.kind + '"><i>' + x.key + "</i>" + (x.kind === "osc" ? v.toFixed(1) : fmtPrice(v)) + "</span>";
      }).join("");
      return "<div><b>" + d.name + "</b>" + ps + vals + "</div>";
    }).join("");
    const note = IND_TFS.has(pr.current.tf) ? "" : "<em>indicadores em 4H/1D/1S</em>";
    e.innerHTML = '<div class="st"><b>' + P.name + "</b><em>estratégia · " + TF_LABEL[pr.current.tf] + (pr.current.source === "ha" ? " · sinais HA" : "") + "</em>" + note + "</div>" + rows;
  };

  const drawBt = () => {
    const m = chart.current, b = bt.current, d = pr.current;
    if (!m) return;
    trades.current = b ? buildTrades(b, bars.current) : [];
    labelsOn.current = d.details || (m.timeScale().options().barSpacing ?? 0) >= 9;
    const list = buildMarkers(trades.current, shown.current, labelsOn.current);
    if (ovl.current) { ovl.current.trades = frozen.current ? [] : trades.current; ovl.current.update(); }
    markers.current?.setMarkers(list);
    if (d.showEq && b && !frozen.current) {
      if (!eqS.current) {
        eqS.current = m.addSeries(LineSeries, { color: COL.eq, lineWidth: 2, title: "Estratégia", priceLineVisible: false, crosshairMarkerRadius: 3 }, 1);
        bhS.current = m.addSeries(LineSeries, { color: COL.bh, lineWidth: 1, title: "Buy & hold", priceLineVisible: false, lineStyle: LineStyle.Dotted }, 1);
        m.panes()[1]?.setStretchFactor(0.32);
        m.panes()[0]?.setStretchFactor(1);
      }
      const b0 = b.equity[0]?.value || 1;
      const pf = { type: "custom" as const, minMove: 0.0001, formatter: (v: number) => fmtPct(v / b0 - 1, true, 0) };
      eqS.current.applyOptions({ priceFormat: pf }); bhS.current!.applyOptions({ priceFormat: pf });
      // clip to loaded bars: extra time points would shift the shared logical index space
      const t0 = shown.current[0]?.time ?? Infinity;
      const clip = (a: { time: number; value: number }[]) => a.filter((p) => p.time >= t0).map((p) => ({ time: T(p.time), value: p.value }));
      eqS.current.setData(clip(b.equity));
      bhS.current!.setData(clip(b.buy_hold));
      requestAnimationFrame(() => { const h = chart.current?.panes()[0]?.getHeight(); setEqTop(h != null && chart.current?.panes()[1] ? h + 1 : null); });
    } else if (eqS.current) {
      m.removeSeries(eqS.current); m.removeSeries(bhS.current!);
      eqS.current = null; bhS.current = null;
      setEqTop(null);
    }
  };

  const liqRange = useRef<[number, number] | null>(null);
  const showRange = async (t0: number, t1: number) => {
    liqRange.current = [t0, t1];
    const f = TF_SEC[pr.current.tf];
    if (!bars.current.length) return;   // re-applied once the bars are painted
    if (bars.current.length && t0 < bars.current[0].time && (bars.current[0].time - t0) / f < 60000) {
      const r = await getOhlcv(pr.current.symbol, pr.current.tf, t0 - 20 * f, bars.current[0].time, barsAc.current?.signal).catch(() => null);
      if (!r) return;
      const older = r.bars.filter((x) => x.time < bars.current[0].time);
      if (older.length) { bars.current = [...older, ...bars.current]; show(bars.current, true); drawBt(); drawInd(); }
    }
    let a = logicalOf(t0), b = logicalOf(t1);
    if (a == null || b == null) return;
    if (b - a < 30) a = b - 30;   // window shorter than a few bars (e.g. 1d on 1D): keep context, cells merge per bar
    chart.current?.timeScale().setVisibleLogicalRange({ from: a, to: b + (b - a) * 0.15 });
    prim.current?.invalidate();
  };

  const fns = useRef({ showRange, drawInd, indLegend, extendLeft, refreshLayers, mergeTail, mergeGap, drawBt, logicalOf, timeOf, xOf, show, legendLast });
  fns.current = { showRange, drawInd, indLegend, extendLeft, refreshLayers, mergeTail, mergeGap, drawBt, logicalOf, timeOf, xOf, show, legendLast };

  // ---- chart lifecycle: created exactly once per pane
  useEffect(() => {
    const m = createChart(el.current!, chartOpts);
    chart.current = m;
    candle.current = m.addSeries(CandlestickSeries, {
      upColor: COL.up, downColor: COL.dn, borderVisible: false, wickUpColor: COL.up, wickDownColor: COL.dn,
      priceLineVisible: true, priceLineStyle: LineStyle.Dotted, priceLineWidth: 1,
    });
    vol.current = m.addSeries(HistogramSeries, { priceFormat: { type: "volume" }, priceScaleId: "vol", lastValueVisible: false, priceLineVisible: false });
    m.priceScale("vol").applyOptions({ scaleMargins: { top: 0.87, bottom: 0 } });
    prim.current = new LayersPrimitive();
    prim.current.timeToX = (t) => fns.current.xOf(t);
    prim.current.tfSec = TF_SEC[pr.current.tf];
    if (import.meta.env.DEV && idx === 0) Object.assign(window, { __prim: prim.current, __pane0: { fns, m, shown } });
    candle.current.attachPrimitive(prim.current);
    ovl.current = new OverlayPrimitive();
    ovl.current.xOf = (t) => fns.current.xOf(t);
    candle.current.attachPrimitive(ovl.current);
    markers.current = createSeriesMarkers<Time>(candle.current, []);
    wm.current = createTextWatermark(m.panes()[0], { horzAlign: "center", vertAlign: "center", lines: [] });

    const bus = pr.current.bus;
    const nearest = (t: number) => {
      const b = shown.current;
      let lo = 0, hi = b.length - 1, r = -1;
      while (lo <= hi) { const mid = (lo + hi) >> 1; if (b[mid].time <= t) { r = mid; lo = mid + 1; } else hi = mid - 1; }
      return r < 0 ? null : b[r];
    };
    let applying = 0;
    bus.peers.set(idx, {
      cross: (t) => {
        if (hover.current) return;
        if (t == null) { m.clearCrosshairPosition(); fns.current.legendLast(); return; }
        const b = nearest(t);
        if (b && candle.current) m.setCrosshairPosition(b.close, T(b.time), candle.current);
        else m.clearCrosshairPosition();
      },
      range: (a, b) => {
        if (hover.current) return;
        const la = fns.current.logicalOf(a), lb = fns.current.logicalOf(b);
        if (la == null || lb == null) return;
        applying = performance.now();
        m.timeScale().setVisibleLogicalRange({ from: la, to: lb });
      },
    });

    const pointPos = (p: { point?: { x: number; y: number } }) => {
      if (!p.point || !candle.current) return null;
      const l = m.timeScale().coordinateToLogical(p.point.x);
      const price = candle.current.coordinateToPrice(p.point.y);
      if (l == null || price == null) return null;
      const t = fns.current.timeOf(l);
      return t == null ? null : { t: Math.round(t), p: price };
    };

    m.subscribeCrosshairMove((p) => {
      const d = p.seriesData.get(candle.current!) as Bar | undefined;
      if (d) {
        const b = shown.current, i = b.findIndex((x) => x.time === d.time);
        setLegend({ ...d, volume: b[i]?.volume ?? 0 }, b[i - 1]);
      } else if (hover.current || p.time == null) fns.current.legendLast();
      if (hover.current) bus.emitCross(idx, p.time == null ? null : (p.time as number));
      fns.current.indLegend(p.time == null ? null : (p.time as number));
      // trade tooltip when hovering an entry/exit bar (labels collapse when zoomed out)
      const tp = tip.current;
      if (tp) {
        const tt = p.time as number | undefined;
        const hit = tt == null || !p.point ? null : trades.current.find((x) => x.entryT === tt || x.exitT === tt);
        if (!hit) tp.style.display = "none";
        else {
          const isEntry = hit.entryT === tt;
          tp.style.display = "block";
          tp.style.left = Math.min(p.point!.x + 14, (el.current?.clientWidth ?? 400) - 210) + "px";
          tp.style.top = Math.max(4, p.point!.y - 70) + "px";
          const head = isEntry ? (hit.long ? "▲ Entrada long" : "▼ Entrada short") : "Fechar " + (hit.long ? "long" : "short") + " · " + reasonPt(hit.reason);
          const col = isEntry ? (hit.long ? SCOL.long : SCOL.short) : SCOL.exit;
          tp.innerHTML = '<b style="color:' + col + '">' + head + "</b>" +
            "<div>Negociação #" + hit.n + (hit.hold ? " · holdout" : "") + "</div>" +
            "<div>Entrada " + fmtPrice(hit.entryP) + (hit.exitP != null ? " → saída " + fmtPrice(hit.exitP) : " · em aberto") + "</div>" +
            (hit.pnl != null ? '<div class="' + (hit.pnl >= 0 ? "up" : "dn") + '">' + fmtPct(hit.pnl, true, 2) + "</div>" : "");
        }
      }
      // drawing preview
      const dr = draft.current, o = ovl.current;
      if (dr && o) {
        const pos = pointPos(p);
        if (pos) { o.preview = { ...dr, b: pos }; o.update(); }
      }
    });

    // Own pointer handling (LWC folds a fast second click into a double-click and drops it)
    const onPick = (p: { point: { x: number; y: number } }) => {
      const o = ovl.current, P = pr.current, sym = P.symbol;
      if (!o) return;
      const t = P.tool;
      if (t === "cursor" || t === "erase") {
        const h = o.hit(p.point.x, p.point.y, el.current!.clientWidth);
        if (t === "erase") { if (h) drawings.remove(sym, h.id); return; }
        o.selected = h?.id ?? null; setSelected(o.selected); o.update();
        return;
      }
      const pos = pointPos(p);
      if (!pos) return;
      if (t === "hline") {
        drawings.add(sym, { id: newId(), kind: "hline", a: pos });
        P.onTool("cursor");
        return;
      }
      if (!draft.current) { draft.current = { id: newId(), kind: t, a: pos }; return; }
      const d = { ...draft.current, b: pos };
      draft.current = null; o.preview = null;
      drawings.add(sym, d);
      P.onTool("cursor");
    };
    const host = el.current!;
    let down: { x: number; y: number } | null = null;
    const rel = (e: PointerEvent) => { const r = host.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
    const pd = (e: PointerEvent) => { down = e.button === 0 ? rel(e) : null; };
    const pu = (e: PointerEvent) => {
      const u = rel(e), d0 = down;
      down = null;
      if (!d0 || Math.hypot(u.x - d0.x, u.y - d0.y) > 4) return;
      if (u.x > m.timeScale().width() || u.y > (m.panes()[0]?.getHeight() ?? 0)) return;
      onPick({ point: u });
    };
    host.addEventListener("pointerdown", pd);
    host.addEventListener("pointerup", pu);

    let timer: ReturnType<typeof setTimeout>;
    m.timeScale().subscribeVisibleLogicalRangeChange((r) => {
      const want = pr.current.details || (m.timeScale().options().barSpacing ?? 0) >= 9;
      if (want !== labelsOn.current) fns.current.drawBt();
      if (r && hover.current && performance.now() - applying > 60) {
        const a = fns.current.timeOf(r.from), b = fns.current.timeOf(r.to);
        if (a != null && b != null) bus.emitRange(idx, a, b);
      }
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (r && r.from < 40) fns.current.extendLeft();
        fns.current.refreshLayers();
      }, 200);
    });

    const unDraw = drawings.subscribe((s) => {
      if (s !== pr.current.symbol || !ovl.current) return;
      ovl.current.drawings = drawings.get(s);
      if (ovl.current.selected && !ovl.current.drawings.some((x) => x.id === ovl.current!.selected)) { ovl.current.selected = null; setSelected(null); }
      ovl.current.update();
    });
    const ro = new ResizeObserver(() => requestAnimationFrame(() => {
      const ps = chart.current?.panes();
      setEqTop(ps && ps.length > 1 ? ps[0].getHeight() + 1 : null);
    }));
    ro.observe(el.current!);
    if (import.meta.env.DEV && idx === 0) (window as unknown as { __barSpacing?: () => number }).__barSpacing = () => { const r = m.timeScale().getVisibleLogicalRange(); return r ? Math.round((m.timeScale().width() / (r.to - r.from)) * 100) / 100 : 0; };
    setReady(true);
    return () => {
      host.removeEventListener("pointerdown", pd);
      host.removeEventListener("pointerup", pu);
      ro.disconnect();
      clearTimeout(timer);
      unDraw();
      bus.peers.delete(idx);
      barsAc.current?.abort();
      Object.values(layerAc.current).forEach((a) => a?.abort());
      m.remove();
      chart.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- bars: reload on symbol/tf change (stale requests aborted)
  useEffect(() => {
    const ac = new AbortController();
    barsAc.current = ac;
    bars.current = []; exhausted.current = false; extending.current = false;
    if (!frozen.current) { shown.current = []; candle.current?.setData([]); vol.current?.setData([]); }
    if (ovl.current) { ovl.current.drawings = drawings.get(symbol); ovl.current.selected = null; ovl.current.update(); }
    setSelected(null);
    setLegend(null); setLoading(true);
    wm.current?.applyOptions({ lines: [
      { text: symbol.replace("USDT", ""), color: "rgba(230,233,239,0.035)", fontSize: 120, fontStyle: "800", fontFamily: "Inter Variable, sans-serif", lineHeight: 120 },
      { text: `${TF_LABEL[tf]} · USDT PERP`, color: "rgba(230,233,239,0.05)", fontSize: 16, fontStyle: "600", fontFamily: "JetBrains Mono Variable, monospace", lineHeight: 24 },
    ] });
    // Instant paint from the in-memory LRU (warm switch), then refresh from the API (last 5000 bars + live tail).
    const tStart = performance.now();
    let painted = false, earlyK: Bar | null = null;
    const paint = (b: Bar[], holdout: number | null, src: string) => {
      if (ac.signal.aborted) return;
      const first = !painted;
      if (painted && bars.current.length) {
        // refresh: keep the live forming bar, insert anything new/missing
        fns.current.mergeGap(b);
      } else {
        bars.current = b.slice();
        if (earlyK) { const k = earlyK; earlyK = null; const lt = b[b.length - 1]?.time ?? 0; if (k.time > lt) bars.current.push(k); else if (k.time === lt) bars.current[bars.current.length - 1] = k; }
        if (ovl.current) ovl.current.holdout = holdout;
        if (!frozen.current) {
          show(bars.current, false);
          const n = bars.current.length;
          if (n) chart.current?.timeScale().setVisibleLogicalRange({ from: n - 160, to: n + 6 });
          if (prim.current) { prim.current.tfSec = TF_SEC[tf]; prim.current.invalidate(); }
          if (pr.current.mode === "liq" && liqRange.current) void fns.current.showRange(...liqRange.current);
          legendLast();
          fns.current.drawBt();
          fns.current.drawInd();
        }
        painted = true;
        setLoadedKey(symbol + "|" + tf);
        setLoading(false);
        fns.current.refreshLayers();
      }
      if (first) requestAnimationFrame(() => {
        const ms = Math.round(performance.now() - tStart);
        (window as unknown as { __tfSwitch?: unknown[] }).__tfSwitch = [...((window as unknown as { __tfSwitch?: unknown[] }).__tfSwitch ?? []), { tf, symbol, ms, src }];
        if (import.meta.env.DEV) console.info(`[pane ${idx}] ${symbol} ${tf}: candles painted in ${ms} ms (${src})`);
      });
      putBars(symbol, tf, bars.current, holdout);
    };
    const hit = cachedBars(symbol, tf);
    if (hit) paint(hit.bars, hit.holdout, "cache");
    getRecent(symbol, tf, ac.signal)
      .then((r) => paint(r.bars, r.holdout_start ?? null, "rede"))
      .then(() => { if (!ac.signal.aborted) prefetchNear(symbol, tf); })
      .catch((e) => { if (!isAbort(e)) { pr.current.onErr(String(e)); setLoading(false); } });

    // live stream for the forming candle + gap backfill on reconnect / refocus
    const backfill = () => {
      const b = bars.current;
      if (!b.length) return;
      getOhlcv(symbol, tf, b[b.length - 1].time - TF_SEC[tf], undefined, ac.signal)
        .then((r) => { if (!ac.signal.aborted) fns.current.mergeGap(r.bars); }).catch(() => {});
    };
    // Forming candle: aggTrade drives price tick-by-tick, kline supplies open/close boundaries and volume
    // truth. All updates coalesce into one series.update per animation frame (refs only, no React state).
    const tfS = TF_SEC[tf];
    let pend: Bar | null = null, pendE = 0, raf = 0, lastTradeE = 0;
    const flush = () => {
      raf = 0;
      if (!pend || ac.signal.aborted) return;
      const b = pend, E = pendE;
      pend = null;
      fns.current.mergeTail([b]);
      live.noteRender(E);
    };
    const queue = (b: Bar, E: number) => { pend = b; pendE = Math.max(pendE, E); if (!raf) raf = requestAnimationFrame(flush); };
    const cur = () => pend ?? bars.current[bars.current.length - 1];
    const unT = live.trades(symbol, (t) => {
      if (ac.signal.aborted || !bars.current.length) return;
      const c = cur(), bt0 = barOpen(Math.floor(t.T / 1000), tfS);
      if (bt0 < c.time) return;
      let nb: Bar;
      if (bt0 === c.time) nb = { ...c, high: Math.max(c.high, t.p), low: Math.min(c.low, t.p), close: t.p, volume: c.volume + t.q };
      else {
        if (bt0 > c.time + tfS * 1.5) backfill();
        nb = { time: bt0, open: t.p, high: t.p, low: t.p, close: t.p, volume: t.q };
      }
      lastTradeE = t.E;
      queue(nb, t.E);
    });
    const unK = live.kline(symbol, tf, (k) => {
      if (ac.signal.aborted) return;
      if (!bars.current.length) { earlyK = k; return; }  // subscribed before HTTP returns: hold the newest bar
      const c = cur();
      if (k.time < c.time) return;
      const fresh = k.E >= lastTradeE;
      const nb: Bar = k.time === c.time
        ? { time: k.time, open: k.open, high: Math.max(k.high, c.high), low: Math.min(k.low, c.low), close: fresh ? k.close : c.close, volume: Math.max(k.volume, c.volume) }
        : { time: k.time, open: k.open, high: k.high, low: k.low, close: k.close, volume: k.volume };
      if (k.time > c.time + tfS * 1.5) backfill();
      queue(nb, k.E);
      if (k.closed) pr.current.onBarClose(symbol, tf);
    });
    const unR = live.onResync(backfill);
    return () => { ac.abort(); cancelAnimationFrame(raf); unT(); unK(); unR(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol, tf]);

  // ---- backtest (memoized in api.ts)
  useEffect(() => {
    if (!strategy || !fee || loadedKey !== symbol + "|" + tf) return;  // never compete with the candle request
    const ac = new AbortController();
    setBtState("busy");
    getBacktest(symbol, strategy, tf, fee, ac.signal, props.source)
      .then((b) => { bt.current = b; props.onBt(idx, b); fns.current.drawBt(); setBtState("ok"); })
      .catch((e) => { if (!isAbort(e)) { bt.current = null; props.onBt(idx, null); fns.current.drawBt(); setBtState("err"); } });
    return () => ac.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol, strategy, tf, fee, props.source, loadedKey]);

  useEffect(() => fns.current.drawBt(), [showEq, details, mode]);

  // strategy indicators (only on tfs where /indicator stays light)
  const spec = props.spec;
  useEffect(() => {
    ind.current = []; fns.current.drawInd();
    if (!spec?.indicators || !IND_TFS.has(tf)) return;
    let dead = false;
    const go = () => {
      const b = bars.current;
      if (!b.length) { setTimeout(() => { if (!dead) go(); }, 300); return; }
      const med = b[b.length >> 1].close;
      Promise.all(Object.entries(spec.indicators as Record<string, object>).map(([n, prm]) => getIndicator(n, prm, symbol, tf, med)))
        .then((r) => { if (!dead) { ind.current = r; fns.current.drawInd(); } }).catch(() => {});
    };
    go();
    return () => { dead = true; };
  }, [spec, symbol, tf]);
  useEffect(() => fns.current.indLegend(null), [props.source]);

  // scroll to a trade picked in the strategy tester (loads older bars if needed)
  const focus = props.focus;
  useEffect(() => {
    if (!focus || (multi && !active)) return;
    let dead = false;
    (async () => {
      const f = TF_SEC[pr.current.tf];
      if (bars.current.length && focus.t0 < bars.current[0].time) {
        const r = await getOhlcv(pr.current.symbol, pr.current.tf, focus.t0 - 200 * f, bars.current[0].time);
        if (dead) return;
        const older = r.bars.filter((x) => x.time < bars.current[0].time);
        bars.current = [...older, ...bars.current];
        fns.current.show(bars.current, false); fns.current.drawBt(); fns.current.drawInd();
      }
      const a = fns.current.logicalOf(focus.t0), b = fns.current.logicalOf(focus.t1);
      if (a == null || b == null) return;
      const span = Math.max(40, (b - a) * 3);
      chart.current?.timeScale().setVisibleLogicalRange({ from: a - span / 2, to: b + span / 2 });
      if (ovl.current) { ovl.current.focus = focus.n; ovl.current.update(); }
    })().catch(() => {});
    return () => { dead = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focus?.seq]);
  useEffect(() => { if (!frozen.current && bars.current.length) fns.current.show(bars.current, true); }, [props.ha]);
  // Log / linear price scale (heatmaps and overlays map prices through the series, so they follow it)
  useEffect(() => {
    chart.current?.priceScale("right").applyOptions({ mode: props.log ? PriceScaleMode.Logarithmic : PriceScaleMode.Normal });
  }, [props.log, ready]);
  useEffect(() => { fns.current.refreshLayers(); }, [symbol, tf, showBook, showVp, mode]);

  // tool change: drop any half-made drawing
  useEffect(() => {
    if (draft.current && draft.current.kind === tool) return;
    draft.current = null;
    if (ovl.current) { ovl.current.preview = null; ovl.current.update(); }
  }, [tool]);

  // delete key removes the selected drawing
  useEffect(() => {
    if (!selected) return;
    const k = (e: KeyboardEvent) => {
      if ((e.key === "Delete" || e.key === "Backspace") && !(e.target as HTMLElement).closest("input,textarea")) {
        drawings.remove(pr.current.symbol, selected);
      }
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [selected]);

  const liqOn = mode === "liq" && hasLiq;
  const host: LiqHost | null = ready && chart.current && candle.current && prim.current ? {
    chart: chart.current, candle: candle.current, prim: prim.current,
    // The heatmap overlays the pane's own TF; the range buttons only move the viewport (older bars loaded on demand).
    showRange: (t0: number, t1: number) => { void fns.current.showRange(t0, t1); },
  } : null;

  const m = bt.current?.metrics;
  const ret = typeof m?.total_return === "number" ? m.total_return : null;
  const sh = typeof m?.sharpe === "number" ? m.sharpe : null;

  return (
    <div className={"pane" + (active ? " active" : "") + (tool !== "cursor" ? " drawing" : "")}
      onMouseDown={() => props.onActivate(idx)}
      onMouseEnter={() => { hover.current = true; }} onMouseLeave={() => { hover.current = false; props.bus.emitCross(idx, null); }}>
      <div className="phead">
        <span className="psym">{symbol.replace("USDT", "")}<em>/USDT</em><small>PERP</small></span>
        {multi ? (
          <div className="ptf">
            <button className="chip" onClick={() => setTfOpen((v) => !v)} onBlur={() => setTimeout(() => setTfOpen(false), 150)}>{TF_LABEL[tf]}<svg width="8" height="8" viewBox="0 0 8 8"><path d="M1 3l3 3 3-3" stroke="currentColor" fill="none" /></svg></button>
            {tfOpen && (
              <div className="menu">
                {ALL_TFS.map((t) => (
                  <button key={t} className={t === tf ? "on" : ""} disabled={!props.tfs.includes(t)} onMouseDown={(e) => { e.preventDefault(); props.onTf(idx, t); setTfOpen(false); }}>{TF_LABEL[t]}</button>
                ))}
              </div>
            )}
          </div>
        ) : null}
        <div className="pmode">
          <button className={mode === "candles" ? "on" : ""} onClick={() => props.onMode(idx, "candles")}>Velas</button>
          <button className={liqOn ? "on" : ""} disabled={!hasLiq} title={hasLiq ? "Heatmap de liquidações" : "Sem dados de liquidação para este ativo"} onClick={() => props.onMode(idx, "liq")}>Liquidações</button>
        </div>
        <div ref={legend} className="legend" />
        <div className="pright">
          {note && <span className="pnote">{note}</span>}
          {btState === "busy" && <span className="pnote"><i className="spin" />backtest</span>}
          {btState === "ok" && ret != null && !liqOn && (
            <span className="pbt" title="Backtest in-sample">
              <span className={ret >= 0 ? "up" : "dn"}>{fmtPct(ret, true, 0)}</span>
              <span>SR {sh?.toFixed(2) ?? "—"}</span>
            </span>
          )}
        </div>
      </div>
      <div className="pbody">
        <div className="pchart" ref={el} />
        <div className="indleg" ref={indLeg} />
        <div className="ttip" ref={tip} />
        {eqTop != null && bt.current && (
          <div className="eqleg" style={{ top: eqTop + 6 }}>
            <span><i style={{ background: COL.eq }} />Estratégia</span><span><i className="dash" />Buy &amp; hold</span>
            <span className="mut">equity in-sample · até {fmtDate(bt.current.holdout_start)}</span>
          </div>
        )}
        {liqOn && host && <div className="liq-host"><LiqMode symbol={symbol} host={host} /></div>}
        {loading && <div className="ploading"><i /></div>}
      </div>
    </div>
  );
}

export default memo(Pane);
