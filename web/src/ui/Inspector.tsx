import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Backtest, Signal, SymbolInfo } from "../types";
import type { Tick } from "../live";
import { TF_LABEL, fmtDate, fmtNum, fmtPct, fmtPrice, fmtTime, base } from "../util";
import Price from "./Price";

type Tab = "resumo" | "detalhes" | "trades";
export interface InspectorProps {
  symbols: SymbolInfo[]; coming: string[]; symbol: string; onSymbol(s: string): void; ticks: Map<string, Tick>;
  sig: Signal | null; sigTf: string; strategy: string; bt: Backtest | null; btTf: string;
  tab: Tab; setTab(t: Tab): void;
  fees: { name: string; note: string }[]; fee: string; setFee(f: string): void;
  book: boolean; vp: boolean; exits: boolean; setFlag(k: "book" | "vp" | "details", v: boolean): void;
}

const num = (m: Backtest["metrics"] | undefined, k: string) => { const v = m?.[k]; return typeof v === "number" ? v : null; };

function Sparkline({ bt }: { bt: Backtest }) {
  const W = 260, H = 64;
  const d = useMemo(() => {
    const e = bt.equity, b = bt.buy_hold;
    if (e.length < 2) return null;
    const step = Math.max(1, Math.floor(e.length / 240));
    const es = e.filter((_, i) => i % step === 0), bs = b.filter((_, i) => i % step === 0);
    const all = [...es, ...bs].map((p) => Math.log(p.value));
    const lo = Math.min(...all), hi = Math.max(...all);
    const t0 = e[0].time, t1 = e[e.length - 1].time;
    const x = (t: number) => ((t - t0) / (t1 - t0 || 1)) * W;
    const y = (v: number) => H - 3 - ((Math.log(v) - lo) / (hi - lo || 1)) * (H - 6);
    const path = (s: typeof es) => s.map((p, i) => `${i ? "L" : "M"}${x(p.time).toFixed(1)},${y(p.value).toFixed(1)}`).join("");
    const ep = path(es);
    return { ep, bp: path(bs), area: `${ep}L${W},${H}L0,${H}Z`, base: y(e[0].value) };
  }, [bt]);
  if (!d) return null;
  return (
    <svg className="spark" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label="Equity da estratégia vs buy & hold (escala log)">
      <defs>
        <linearGradient id="eqg" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#4c8dff" stopOpacity="0.28" /><stop offset="1" stopColor="#4c8dff" stopOpacity="0" /></linearGradient>
      </defs>
      <line x1="0" x2={W} y1={d.base} y2={d.base} stroke="#1c222d" strokeDasharray="2 3" />
      <path d={d.area} fill="url(#eqg)" />
      <path d={d.bp} stroke="#5d6676" strokeWidth="1" fill="none" strokeDasharray="2 2" vectorEffect="non-scaling-stroke" />
      <path d={d.ep} stroke="#4c8dff" strokeWidth="1.6" fill="none" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function Watchlist(p: InspectorProps) {
  const ORDER = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "NEARUSDT", "FETUSDT"];
  const rank = (s: string) => { const i = ORDER.indexOf(s); return i < 0 ? 99 : i; };
  const rows = [...p.symbols.map((s) => ({ s: s.symbol, soon: false })), ...p.coming.map((s) => ({ s, soon: true }))]
    .sort((a, b) => Number(a.soon) - Number(b.soon) || rank(a.s) - rank(b.s));
  return (
    <section className="card watch">
      <header><h4>Watchlist</h4><span className="hint">Binance perp · 24h</span></header>
      <div className="wl">
        {rows.map(({ s, soon }) => {
          const t = p.ticks.get(s);
          return (
            <button key={s} className={"wrow" + (s === p.symbol ? " on" : "") + (soon ? " soon" : "")} disabled={soon}
              onClick={() => p.onSymbol(s)} title={soon ? "Em breve: dados ainda não ingeridos" : `Abrir ${s}`}>
              <span className="ws">{base(s)}{soon && <em>em breve</em>}</span>
              <Price value={t?.price} className="wp" />
              <span className={"wc " + (t ? (t.pct >= 0 ? "up" : "dn") : "")}>{t ? fmtPct(t.pct, true, 2) : "—"}</span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function SignalCard({ sig, strategy, tf }: { sig: Signal | null; strategy: string; tf: string }) {
  const st = sig?.state ?? null;
  return (
    <section className={"card sigcard " + (st ?? "none")}>
      <header><h4>Sinal atual</h4><span className="hint">{strategy || "—"} · {TF_LABEL[tf]}</span></header>
      <div className="sigbody">
        <div className="sigstate">
          <i className="sigdot" />
          <b>{st ?? "…"}</b>
          <span>{st === "LONG" ? "comprado" : st === "SHORT" ? "vendido" : st === "FLAT" ? "fora do mercado" : "calculando"}</span>
        </div>
        <div className="sigmeta">
          <span><i>Preço</i>{sig ? fmtPrice(sig.price) : "—"}</span>
          <span><i>Candle</i>{sig ? fmtTime(sig.time).replace(" UTC", "") : "—"}</span>
        </div>
      </div>
      <footer>Calculado em candles fechados · atualiza a cada fechamento</footer>
    </section>
  );
}

function Stats({ bt, tf, onMore }: { bt: Backtest | null; tf: string; onMore(): void }) {
  const m = bt?.metrics;
  const tiles: [string, string, string][] = [
    ["Retorno", fmtPct(num(m, "total_return"), true, 0), (num(m, "total_return") ?? 0) >= 0 ? "up" : "dn"],
    ["Sharpe", fmtNum(num(m, "sharpe")), ""],
    ["Max DD", fmtPct(num(m, "max_drawdown"), true, 1), "dn"],
    ["Win rate", fmtPct(num(m, "win_rate"), false, 0), ""],
    ["Trades", num(m, "n_trades")?.toString() ?? "—", ""],
  ];
  return (
    <section className="card stats">
      <header><h4>Backtest</h4><span className="hint">in-sample · {TF_LABEL[tf]}{bt ? ` · até ${fmtDate(bt.holdout_start)}` : ""}</span></header>
      <div className="tiles">
        {tiles.map(([l, v, c], i) => (
          <div key={l} className={"tile" + (i === 0 ? " hero" : "")}><span>{l}</span><b className={bt ? c : ""}>{bt ? v : "—"}</b></div>
        ))}
      </div>
      {bt && <Sparkline bt={bt} />}
      <div className="sparkleg">
        <span><i style={{ background: "#4c8dff" }} />Estratégia</span><span><i className="dash" />Buy &amp; hold</span><span className="mut">log</span>
        <button className="link" onClick={onMore}>Detalhes →</button>
      </div>
    </section>
  );
}

function Details(p: InspectorProps) {
  const m = p.bt?.metrics;
  const rows: [string, string][] = [
    ["CAGR", fmtPct(num(m, "cagr"))], ["Sortino", fmtNum(num(m, "sortino"))], ["Calmar", fmtNum(num(m, "calmar"))],
    ["Profit factor", fmtNum(num(m, "profit_factor"))], ["Expectativa", num(m, "expectancy_bp") != null ? fmtNum(num(m, "expectancy_bp"), 1) + " bp" : "—"],
    ["Exposição", fmtPct(num(m, "exposure"), false, 0)], ["Duração média", num(m, "avg_trade_hours") != null ? fmtNum(num(m, "avg_trade_hours"), 0) + " h" : "—"],
    ["DD máx. (dias)", fmtNum(num(m, "max_drawdown_days"), 0)], ["Long / Short", `${num(m, "n_long") ?? "—"} / ${num(m, "n_short") ?? "—"}`],
    ["Taxas pagas", fmtPct(num(m, "total_fees"), false, 1)], ["Funding", fmtPct(num(m, "total_funding"), true, 1)], ["Anos", fmtNum(num(m, "years"), 1)],
  ];
  const yr = (m?.yearly_returns ?? {}) as Record<string, number | null>;
  const years = Object.entries(yr).filter(([, v]) => v != null) as [string, number][];
  const mx = Math.max(0.01, ...years.map(([, v]) => Math.abs(v)));
  return (
    <div className="details">
      <section className="card">
        <header><h4>Métricas</h4><span className="hint">in-sample</span></header>
        <div className="kvgrid">{rows.map(([k, v]) => <div key={k} className="kv"><span>{k}</span><b>{v}</b></div>)}</div>
      </section>
      {years.length > 0 && (
        <section className="card">
          <header><h4>Retorno anual</h4></header>
          <div className="years">
            {years.map(([y, v]) => (
              <div key={y} className="yr" title={`${y}: ${fmtPct(v)}`}>
                <span>{y}</span>
                <div className="ybar"><i className={v >= 0 ? "pos" : "neg"} style={{ width: `${(Math.abs(v) / mx) * 50}%` }} /></div>
                <b className={v >= 0 ? "up" : "dn"}>{fmtPct(v, true, 0)}</b>
              </div>
            ))}
          </div>
        </section>
      )}
      <section className="card">
        <header><h4>Camadas e custos</h4></header>
        <label className="tg"><input type="checkbox" checked={p.book} onChange={(e) => p.setFlag("book", e.target.checked)} /><i />Livro de ofertas (heatmap)</label>
        <label className="tg"><input type="checkbox" checked={p.vp} onChange={(e) => p.setFlag("vp", e.target.checked)} /><i />Volume profile</label>
        <label className="tg"><input type="checkbox" checked={p.exits} onChange={(e) => p.setFlag("details", e.target.checked)} /><i />Saídas e rótulos L/S no gráfico</label>
        <label className="selrow">Custos
          <select value={p.fee} onChange={(e) => p.setFee(e.target.value)}>{p.fees.map((f) => <option key={f.name} value={f.name} title={f.note}>{f.name}</option>)}</select>
        </label>
      </section>
    </div>
  );
}

function Trades({ bt, compact }: { bt: Backtest | null; compact?: boolean }) {
  const box = useRef<HTMLDivElement>(null);
  const [per, setPer] = useState(12);
  const [page, setPage] = useState(0);
  useLayoutEffect(() => {
    const e = box.current;
    if (!e) return;
    const ro = new ResizeObserver(() => setPer(Math.max(1, Math.floor((e.clientHeight - 26) / 23))));
    ro.observe(e);
    return () => ro.disconnect();
  }, []);
  const trades = useMemo(() => (bt ? [...bt.trades].reverse() : []), [bt]);
  useEffect(() => setPage(0), [bt]);
  const pages = Math.max(1, Math.ceil(trades.length / per));
  const pg = Math.min(page, pages - 1);
  return (
    <section className={"card trades" + (compact ? " compact" : "")}>
      <header><h4>{compact ? "Últimos trades" : "Trades"}</h4><span className="hint">{trades.length} · mais recentes primeiro</span></header>
      <div className="tbox" ref={box}>
        <table>
          <thead><tr><th>Entrada</th><th>Lado</th><th className="r">Preço</th><th className="r">P&amp;L</th></tr></thead>
          <tbody>
            {trades.slice(pg * per, pg * per + per).map((t, i) => (
              <tr key={i} title={`Saída ${fmtTime(t.exit_time)} · ${t.exit_reason}`}>
                <td>{fmtTime(t.entry_time).slice(0, 10)}</td>
                <td><span className={"side " + (t.side === "LONG" ? "up" : "dn")}>{t.side === "LONG" ? "▲ L" : "▼ S"}</span></td>
                <td className="r">{fmtPrice(t.entry_price)}</td>
                <td className={"r " + (t.pnl_pct >= 0 ? "up" : "dn")}>{fmtPct(t.pnl_pct, true, 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!compact && <div className="pager">
        <button disabled={pg === 0} onClick={() => setPage(pg - 1)}>‹</button>
        <span>{pg + 1} / {pages}</span>
        <button disabled={pg >= pages - 1} onClick={() => setPage(pg + 1)}>›</button>
      </div>}
    </section>
  );
}

export default function Inspector(p: InspectorProps) {
  const tabs: [Tab, string][] = [["resumo", "Resumo"], ["detalhes", "Detalhes"], ["trades", "Trades"]];
  return (
    <aside className="inspector">
      <nav className="tabs">
        {tabs.map(([k, l]) => <button key={k} className={p.tab === k ? "on" : ""} onClick={() => p.setTab(k)}>{l}</button>)}
      </nav>
      <div className="ibody">
        {p.tab === "resumo" && (<>
          <Watchlist {...p} />
          <SignalCard sig={p.sig} strategy={p.strategy} tf={p.sigTf} />
          <Stats bt={p.bt} tf={p.btTf} onMore={() => p.setTab("detalhes")} />
          <Trades bt={p.bt} compact />
        </>)}
        {p.tab === "detalhes" && <Details {...p} />}
        {p.tab === "trades" && <Trades bt={p.bt} />}
      </div>
    </aside>
  );
}
