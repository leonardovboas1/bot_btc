import { useEffect, useState } from "react";
import { getBacktest, getBook, getFees, getLiq, getOhlcv, getSignal, getStrategies, getVp } from "./api";
import type { LayerData } from "./layers";
import type { Backtest, Bar, Signal, Strategy } from "./types";
import Charts from "./Charts";
import SidePanel from "./SidePanel";
import { fmtTime } from "./util";

const TF_SEC: Record<string, number> = { "1h": 3600, "4h": 14400, "1d": 86400 };
const TFS = ["1h", "4h", "1d"];

export default function App() {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [strategy, setStrategy] = useState("");
  const [tf, setTf] = useState("4h");
  const [fee, setFee] = useState("");
  const [fees, setFees] = useState<string[]>([]);
  const [showBook, setShowBook] = useState(false);
  const [showLiq, setShowLiq] = useState(false);
  const [showVp, setShowVp] = useState(false);
  const [layers, setLayers] = useState<LayerData>({ book: null, liq: null, vp: null });
  const [bookNote, setBookNote] = useState("");
  const [bars, setBars] = useState<Bar[]>([]);
  const [bt, setBt] = useState<Backtest | null>(null);
  const [sig, setSig] = useState<Signal | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    getStrategies()
      .then((s) => { setStrategies(s); if (s[0]) { setStrategy(s[0].name); setTf(s[0].tf); } })
      .catch((e) => setErr(String(e)));
  }, []);

  useEffect(() => {
    getFees().then((f) => { const n = f.map((x) => x.name); setFees(n); setFee((c) => c || n[0] || ""); })
      .catch((e) => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (!bars.length) return;
    let dead = false;
    const s = bars[Math.max(0, bars.length - 400)].time;
    const e = bars[bars.length - 1].time + (TF_SEC[tf] ?? 3600);
    const patch = (p: Partial<LayerData>) => !dead && setLayers((l) => ({ ...l, ...p }));
    const fail = (x: unknown) => !dead && setErr(String(x));
    setBookNote("");
    if (showBook) {
      getBook(tf, s, e).then((r) => {
        if (r.empty || !r.bid || !r.ask) {
          patch({ book: null });
          if (!dead) setBookNote("Livro de ofertas só existe a partir de 2023: sem dados neste período.");
        } else patch({ book: { times: r.times, prices: r.prices, a: r.bid, b: r.ask, colA: "#26a69a", colB: "#ef5350" } });
      }).catch(fail);
    } else patch({ book: null });
    if (showLiq) {
      getLiq(tf, s, e).then((r) => patch(r.empty || !r.long || !r.short ? { liq: null }
        : { liq: { times: r.times, prices: r.prices, a: r.long, b: r.short, colA: "#ff9800", colB: "#b36bff" } })).catch(fail);
    } else patch({ liq: null });
    if (showVp) getVp(tf, s, e).then((r) => patch({ vp: r })).catch(fail);
    else patch({ vp: null });
    return () => { dead = true; };
  }, [bars, tf, showBook, showLiq, showVp]);

  useEffect(() => {
    if (!strategy || !fee) return;
    let dead = false;
    setErr("");
    getOhlcv(tf).then((o) => !dead && setBars(o.bars)).catch((e) => !dead && setErr(String(e)));
    getBacktest(strategy, tf, fee).then((b) => !dead && setBt(b)).catch((e) => !dead && setErr(String(e)));
    getSignal(strategy, tf).then((s) => !dead && setSig(s)).catch((e) => !dead && setErr(String(e)));
    return () => { dead = true; };
  }, [strategy, tf, fee]);

  return (
    <div className="app">
      <div className="top">
        <b style={{ fontSize: 20 }}>BTC Signal Lab</b>
        <label>Estratégia
          <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
            {strategies.map((s) => <option key={s.name}>{s.name}</option>)}
          </select></label>
        <label>Timeframe
          <select value={tf} onChange={(e) => setTf(e.target.value)}>
            {TFS.map((t) => <option key={t}>{t}</option>)}
          </select></label>
        <label>Taxas
          <select value={fee} onChange={(e) => setFee(e.target.value)}>
            {fees.map((f) => <option key={f}>{f}</option>)}
          </select></label>
        <div className="toggles">
          <button className={showBook ? "on" : ""} onClick={() => setShowBook(!showBook)}>Livro de ofertas</button>
          <button className={showLiq ? "on" : ""} onClick={() => setShowLiq(!showLiq)}>Liquidações estimadas</button>
          <button className={showVp ? "on" : ""} onClick={() => setShowVp(!showVp)}>Volume profile</button>
        </div>
        {bookNote && <span className="note">{bookNote}</span>}
        <div className="spacer" />
        {err && <span className="neg" title={err}>Erro na API</span>}
        <div className={`badge ${sig?.state ?? "FLAT"}`}>
          <div><small>Sinal atual</small><b>{sig ? sig.state : "..."}</b></div>
          {sig && (
            <small>${sig.price.toLocaleString("pt-BR", { maximumFractionDigits: 2 })}<br />{fmtTime(sig.time)}</small>
          )}
        </div>
      </div>
      <div className="body">
        <Charts bars={bars} bt={bt} layers={layers} />
        <SidePanel bt={bt} />
      </div>
    </div>
  );
}
