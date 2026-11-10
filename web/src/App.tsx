import { useEffect, useState } from "react";
import { getBacktest, getOhlcv, getSignal, getStrategies } from "./api";
import type { Backtest, Bar, Signal, Strategy } from "./types";
import Charts from "./Charts";
import SidePanel from "./SidePanel";
import { fmtTime } from "./util";

const FEES = ["binance_perp_taker", "hyperliquid_taker", "user_tv_default", "spot"];
const TFS = ["1h", "4h", "1d"];

export default function App() {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [strategy, setStrategy] = useState("");
  const [tf, setTf] = useState("4h");
  const [fee, setFee] = useState(FEES[0]);
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
    if (!strategy) return;
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
            {FEES.map((f) => <option key={f}>{f}</option>)}
          </select></label>
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
        <Charts bars={bars} bt={bt} />
        <SidePanel bt={bt} />
      </div>
    </div>
  );
}
