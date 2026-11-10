import { useEffect, useRef, useState } from "react";
import {
  CandlestickSeries, HistogramSeries, LineSeries, ColorType, createChart, createSeriesMarkers,
  type IChartApi, type ISeriesApi, type ISeriesMarkersPluginApi, type MouseEventParams, type Time, type UTCTimestamp,
} from "lightweight-charts";
import type { Backtest, Bar } from "./types";

type Tool = "none" | "hline" | "trend";
const T = (n: number) => n as UTCTimestamp;
const opts = {
  layout: { background: { type: ColorType.Solid, color: "#0f1218" }, textColor: "#aab2c3", fontSize: 13 },
  grid: { vertLines: { color: "#1c2230" }, horzLines: { color: "#1c2230" } },
  timeScale: { timeVisible: true, borderColor: "#2a3140" },
  rightPriceScale: { borderColor: "#2a3140" },
  autoSize: true,
};

export default function Charts({ bars, bt }: { bars: Bar[]; bt: Backtest | null }) {
  const mainEl = useRef<HTMLDivElement>(null);
  const eqEl = useRef<HTMLDivElement>(null);
  const main = useRef<IChartApi | null>(null);
  const eq = useRef<IChartApi | null>(null);
  const candle = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const vol = useRef<ISeriesApi<"Histogram"> | null>(null);
  const eqS = useRef<ISeriesApi<"Line"> | null>(null);
  const bhS = useRef<ISeriesApi<"Line"> | null>(null);
  const markersApi = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const [tool, setTool] = useState<Tool>("none");
  const toolRef = useRef<Tool>("none");
  toolRef.current = tool;
  const pending = useRef<{ time: number; price: number } | null>(null);
  const drawn = useRef<{ lines: unknown[]; series: ISeriesApi<"Line">[] }>({ lines: [], series: [] });
  const [hold, setHold] = useState<{ left: number; width: number } | null>(null);
  const holdRef = useRef(0);
  holdRef.current = bt?.holdout_start ?? 0;

  const updateOverlay = () => {
    const c = main.current;
    if (!c || !holdRef.current) return setHold(null);
    const ts = c.timeScale();
    const w = ts.width();
    let x: number | null = ts.timeToCoordinate(T(holdRef.current));
    if (x == null) {
      const r = ts.getVisibleRange();
      x = r && holdRef.current < (r.from as number) ? 0 : null;
    }
    if (x == null) return setHold(null);
    const left = Math.max(0, x);
    setHold({ left, width: Math.max(0, w - left) });
  };

  useEffect(() => {
    const m = createChart(mainEl.current!, opts);
    const e = createChart(eqEl.current!, opts);
    main.current = m; eq.current = e;
    candle.current = m.addSeries(CandlestickSeries, {
      upColor: "#26a69a", downColor: "#ef5350", borderVisible: false, wickUpColor: "#26a69a", wickDownColor: "#ef5350",
    });
    vol.current = m.addSeries(HistogramSeries, { priceFormat: { type: "volume" }, priceScaleId: "vol" });
    m.priceScale("vol").applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
    markersApi.current = createSeriesMarkers<Time>(candle.current, []);
    eqS.current = e.addSeries(LineSeries, { color: "#4c8dff", lineWidth: 2, title: "Estratégia" });
    bhS.current = e.addSeries(LineSeries, { color: "#b0b7c6", lineWidth: 2, title: "Buy & hold" });
    let busy = false;
    const sync = (from: IChartApi, to: IChartApi) =>
      from.timeScale().subscribeVisibleTimeRangeChange((r) => {
        if (busy || !r) return;
        busy = true;
        try { to.timeScale().setVisibleRange(r); } catch { /* no data yet */ }
        busy = false;
      });
    sync(m, e); sync(e, m);
    m.timeScale().subscribeVisibleTimeRangeChange(updateOverlay);
    m.subscribeClick((p: MouseEventParams<Time>) => {
      const cs = candle.current;
      if (toolRef.current === "none" || !p.point || p.time == null || !cs) return;
      const price = cs.coordinateToPrice(p.point.y);
      if (price == null) return;
      const time = p.time as number;
      if (toolRef.current === "hline") {
        drawn.current.lines.push(cs.createPriceLine({ price, color: "#e0b53a", lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: "" }));
        setTool("none");
      } else if (!pending.current) {
        pending.current = { time, price };
      } else {
        const a = pending.current;
        pending.current = null;
        const s = m.addSeries(LineSeries, { color: "#e0b53a", lineWidth: 2, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false });
        const pts = [{ time: T(a.time), value: a.price }, { time: T(time), value: price }]
          .sort((x, y) => (x.time as number) - (y.time as number));
        if (pts[0].time !== pts[1].time) s.setData(pts);
        drawn.current.series.push(s);
        setTool("none");
      }
    });
    return () => { m.remove(); e.remove(); };
  }, []);

  useEffect(() => {
    candle.current?.setData(bars.map((b) => ({ time: T(b.time), open: b.open, high: b.high, low: b.low, close: b.close })));
    vol.current?.setData(bars.map((b) => ({ time: T(b.time), value: b.volume, color: b.close >= b.open ? "#26a69a55" : "#ef535055" })));
    main.current?.timeScale().fitContent();
    updateOverlay();
  }, [bars]);

  useEffect(() => {
    if (!bt) return;
    const hm = bt.holdout_markers.map((m) => ({ ...m, color: "#6b7280", text: m.text === "EXIT" ? "" : "H " + m.text }));
    const all = [...bt.markers, ...hm].sort((a, b) => a.time - b.time).map((m) => ({ ...m, time: T(m.time) }));
    markersApi.current?.setMarkers(all);
    eqS.current?.setData(bt.equity.map((p) => ({ time: T(p.time), value: p.value })));
    bhS.current?.setData(bt.buy_hold.map((p) => ({ time: T(p.time), value: p.value })));
    updateOverlay();
  }, [bt]);

  const clear = () => {
    drawn.current.lines.forEach((l) => candle.current?.removePriceLine(l as never));
    drawn.current.series.forEach((s) => main.current?.removeSeries(s));
    drawn.current = { lines: [], series: [] };
    pending.current = null;
    setTool("none");
  };
  const pick = (t: Tool) => { pending.current = null; setTool(tool === t ? "none" : t); };

  return (
    <div className="charts">
      <div className="cwrap">
        <div className="c" ref={mainEl} />
        {hold && <div className="hold" style={{ left: hold.left, width: hold.width }}><span>Holdout (intocado)</span></div>}
        <div className="tools">
          <button className={tool === "hline" ? "on" : ""} onClick={() => pick("hline")}>Linha horizontal</button>
          <button className={tool === "trend" ? "on" : ""} onClick={() => pick("trend")}>Tendência (2 cliques)</button>
          <button onClick={clear}>Limpar</button>
        </div>
      </div>
      <div className="cwrap">
        <div className="c" ref={eqEl} />
        <div className="lbl">Equity: estratégia (azul) vs buy &amp; hold (cinza), in-sample</div>
      </div>
    </div>
  );
}
