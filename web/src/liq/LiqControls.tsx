import { useEffect, useRef, useState } from "react";
import type { IChartApi, ISeriesApi } from "lightweight-charts";
import type { Bar } from "../types";
import type { LayersPrimitive } from "../layers";
import { getLiqMap, RANGES, type LiqMap } from "./api";
import { DEFAULT_THR, LiqLayer, VIRIDIS, type Hit, type ScaleMode } from "./LiqHeatmap";
import { HL_COINS, HlLayer, getHlLiq, getHlStops, hlCoin, type HlCoverage, type HlHit } from "./hl";
import "./liq.css";

/** Everything LiqMode needs from the host pane. See README.md. */
export interface LiqHost {
  chart: IChartApi;
  candle: ISeriesApi<"Candlestick">;
  prim: LayersPrimitive;
  /** Legacy: replace the pane candles (no longer used: the heatmap overlays the pane's own TF). */
  setCandles?(bars: Bar[] | null): void;
  /** Scroll/zoom the pane to a time window (seconds), loading older bars if needed. */
  showRange?(t0: number, t1: number): void;
}

const usd = (v: number) => (v >= 1e9 ? `US$ ${(v / 1e9).toFixed(2)} bi` : v >= 1e6 ? `US$ ${(v / 1e6).toFixed(1)} mi` : `US$ ${(v / 1e3).toFixed(0)} mil`);
const SCALES: [ScaleMode, string][] = [["sqrt", "Raiz"], ["log", "Log"], ["linear", "Linear"]];

function Colorbar({ layer, scale, mode, thr }: { layer: LiqLayer; scale: number; mode: ScaleMode; thr: number }) {
  const cv = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = cv.current!.getContext("2d")!;
    for (let x = 0; x < 160; x++) {
      const q = Math.max(1, Math.round((x / 159) * 255));
      const n = layer.intensity(q);
      const [r, g, b] = VIRIDIS[Math.round(n * 255)];
      c.fillStyle = layer.rank(q) < thr ? "#10141b" : `rgb(${r},${g},${b})`;
      c.fillRect(x, 0, 1, 10);
    }
  }, [layer, mode, thr, scale]);
  return (
    <span className="liq-cb">
      <span>0</span><canvas ref={cv} width={160} height={10} /><span>≥ {usd(layer.ceiling || scale)}</span>
    </span>
  );
}

/** Liquidation mode: range buttons, scale, threshold slider, colorbar, hover tooltip. Mount while active. */
export default function LiqMode({ symbol, host }: { symbol: string; host: LiqHost }) {
  const layer = useRef(new LiqLayer("liq")).current;
  const [range, setRange] = useState<string>("1w");
  const [mode, setMode] = useState<ScaleMode>("sqrt");
  const [thr, setThr] = useState(DEFAULT_THR);
  const [bybit, setBybit] = useState(false);
  const [data, setData] = useState<LiqMap | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [tip, setTip] = useState<{ x: number; y: number; h: Hit } | null>(null);
  const hostRef = useRef(host);
  hostRef.current = host;
  const coin = hlCoin(symbol), hlOk = HL_COINS.includes(coin);
  const hlLiq = useRef(new HlLayer("hl-liq", "liq")).current;
  const hlStp = useRef(new HlLayer("hl-stops", "stops")).current;
  const [showHl, setShowHl] = useState(false);
  const [showStops, setShowStops] = useState(false);
  const [cov, setCov] = useState<HlCoverage | null>(null);
  const [hlTip, setHlTip] = useState<{ x: number; y: number; h: HlHit } | null>(null);
  const showRef = useRef({ showHl, showStops });
  showRef.current = { showHl, showStops };

  // exact on-chain layers: refetch every 60 s while a toggle is on
  useEffect(() => {
    const on = hlOk && (showHl || showStops);
    const h = hostRef.current;
    if (!on) { hlLiq.setData(null); hlStp.setData(null); h.prim.removeLayer("hl-liq"); h.prim.removeLayer("hl-stops"); setCov(null); return; }
    const ac = new AbortController();
    const load = () => {
      if (showHl) getHlLiq(coin, ac.signal).then((d) => { hlLiq.setData(d); h.prim.setLayer(hlLiq); setCov(d.coverage ?? null); }).catch(() => {});
      else { hlLiq.setData(null); h.prim.removeLayer("hl-liq"); }
      if (showStops) getHlStops(coin, ac.signal).then((d) => { hlStp.setData(d); h.prim.setLayer(hlStp); }).catch(() => {});
      else { hlStp.setData(null); h.prim.removeLayer("hl-stops"); }
    };
    load();
    const id = setInterval(load, 60_000);
    return () => { ac.abort(); clearInterval(id); };
  }, [coin, hlOk, showHl, showStops, hlLiq, hlStp]);

  useEffect(() => {
    const ac = new AbortController();
    setBusy(true); setErr("");
    getLiqMap(symbol, range, bybit ? ["binance", "bybit"] : ["binance"], ac.signal).then((m) => {
      const h = hostRef.current;
      layer.setData(m);
      h.prim.setLayer(layer);
      h.chart.timeScale().applyOptions({ minBarSpacing: 0.02 });
      // candles keep the pane TF; the range only sets the heatmap window and scrolls the chart to it
      h.showRange?.(m.start, m.end);
      setData(m); setBusy(false);
    }).catch((e) => { if (e?.name !== "AbortError") { setErr(String(e)); setBusy(false); } });
    return () => ac.abort();
  }, [symbol, range, bybit, layer]);

  useEffect(() => {
    layer.setOpts({ scale: mode, thr });
    hostRef.current.prim.setLayer(layer);
  }, [mode, thr, layer]);

  useEffect(() => {
    const { chart, candle } = hostRef.current;
    const fn = (p: { point?: { x: number; y: number } }) => {
      if (!p.point) { setTip(null); return; }
      const price = candle.coordinateToPrice(p.point.y);
      const h = price == null ? null : layer.lookup(p.point.x, price);
      setTip(h ? { x: p.point.x, y: p.point.y, h } : null);
      const sr = showRef.current;
      const x = hlLiq.lookup(price ?? 0, (q) => candle.priceToCoordinate(q), p.point.y);
      const hh = (sr.showHl ? x : null) ?? (sr.showStops ? hlStp.lookup(price ?? 0, (q) => candle.priceToCoordinate(q), p.point.y) : null);
      setHlTip(hh ? { x: p.point.x, y: p.point.y, h: hh } : null);
    };
    chart.subscribeCrosshairMove(fn);
    return () => { try { chart.unsubscribeCrosshairMove(fn); } catch { /* chart disposed */ } };
  }, [layer, hlLiq, hlStp]);

  useEffect(() => () => {
    const h = hostRef.current;
    try {
      h.prim.removeLayer("liq");
      h.prim.removeLayer("hl-liq");
      h.prim.removeLayer("hl-stops");
      h.chart.timeScale().applyOptions({ minBarSpacing: 0.5 });
    } catch { /* chart disposed */ }
    h.setCandles?.(null);
  }, []);

  return (
    <>
      <div className="liq-bar">
        {RANGES.map((r) => (
          <button key={r} className={r === range ? "on" : ""} onClick={() => setRange(r)}>{r}</button>
        ))}
        <span className="liq-sep" />
        {SCALES.map(([k, l]) => <button key={k} className={k === mode ? "on" : ""} onClick={() => setMode(k)}>{l}</button>)}
        <label className="liq-thr" title="Ocultar a fração mais fraca das células (percentil)">
          Limiar <input type="range" min={0} max={0.95} step={0.01} value={thr} onChange={(e) => setThr(Number(e.target.value))} />
        </label>
        <label className="liq-thr"><input type="checkbox" checked={bybit} onChange={(e) => setBybit(e.target.checked)} /> Bybit</label>
        <span className="liq-sep" />
        <label className="liq-thr" title={hlOk ? "Níveis de liquidação exatos (posições on-chain da Hyperliquid) sobre o modelo" : "Sem dados Hyperliquid para este ativo"}>
          <input type="checkbox" disabled={!hlOk} checked={showHl} onChange={(e) => setShowHl(e.target.checked)} /> Hyperliquid exato
        </label>
        <label className="liq-thr" title="Ordens stop/take-profit abertas na Hyperliquid">
          <input type="checkbox" disabled={!hlOk} checked={showStops} onChange={(e) => setShowStops(e.target.checked)} /> Stops HL
        </label>
        {cov && showHl && (
          <span className="liq-note hl-badge" title="Cobertura da amostra de carteiras vs. open interest total da Hyperliquid">
            HL · {cov.addresses.toLocaleString("pt-BR")} carteiras · {cov.oi_share == null ? "?" : (cov.oi_share * 100).toFixed(1)}% do OI · idade {cov.median_age_s == null ? "?" : Math.max(1, Math.round(cov.median_age_s / 60))}m
          </span>
        )}
        {data && <Colorbar layer={layer} scale={data.scale} mode={mode} thr={thr} />}
        {busy && <span className="liq-note">carregando…</span>}
        {!busy && data && (data.T < 2 || !data.q.some((v) => v > 0)) && <span className="liq-note err">Sem liquidações estimadas nesta janela</span>}
        {err && <span className="liq-note err">{err}</span>}
      </div>
      {data && <div className="liq-foot">{data.note} Resolução {data.resolution}.</div>}
      {hlTip && (
        <div className="liq-tip" style={{ left: Math.min(hlTip.x + 14, 9999), top: hlTip.y - 44 }}>
          <b>{hlTip.h.price.toLocaleString("pt-BR", { maximumFractionDigits: hlTip.h.price < 10 ? 4 : 0 })}</b> · {usd(hlTip.h.usd)}
          <div>{hlTip.h.label} · {hlTip.h.n} {hlTip.h.n === 1 ? "posição" : "posições"} · idade {hlTip.h.age == null ? "?" : `${Math.max(1, Math.round(hlTip.h.age / 60))}m`}</div>
        </div>
      )}
      {tip && !hlTip && (
        <div className="liq-tip" style={{ left: Math.min(tip.x + 14, 9999), top: tip.y + 14 }}>
          <b>{tip.h.price.toLocaleString("pt-BR", { maximumFractionDigits: 0 })}</b> · {usd(tip.h.usd)}
          <div>{tip.h.side === "long" ? "Liquidações de compradas" : "Liquidações de vendidas"} (est.){tip.h.future ? " · ativo" : ""}</div>
        </div>
      )}
    </>
  );
}
