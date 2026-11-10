import { memo, useEffect, useMemo, useRef, useState } from "react";
import { AreaSeries, BaselineSeries, ColorType, CrosshairMode, LineSeries, LineStyle, createChart, type IChartApi, type ISeriesApi, type UTCTimestamp } from "lightweight-charts";
import type { Backtest, Bar, Strategy } from "../types";
import type { Fee } from "../api";
import { getOhlcv } from "../api";
import { TF_LABEL, TF_SEC, fmtDate, fmtPrice, fmtTime } from "../util";
import { PERIODS, benchByPeriod, bucket, computeStats, histogram, realizedByPeriod, type Period, type SideStats, type Stats } from "./stats";
import { BarChart, Donut, Pills, Seg, Stat, kfmt } from "./charts";
import { reasonPt } from "../chart/strategyDraw";

export type TTab = "overview" | "analysis" | "summary" | "trades" | "props";
const TABS: [TTab, string][] = [["overview", "Visão geral"], ["analysis", "Análise de desempenho"], ["summary", "Resumo de desempenho"], ["trades", "Lista de negociações"], ["props", "Propriedades"]];
const CAPS = [1000, 10000, 60000, 100000, 1000000];
const UP = "#0ea894", DN = "#f25c54", EQ = "#4c8dff", BH = "#5d6676";

const money = (v: number | null | undefined, signed = false, d = 2) =>
  v == null || !isFinite(v) ? "—" : (signed && v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toLocaleString("pt-BR", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v: number | null | undefined, signed = false, d = 2) =>
  v == null || !isFinite(v) ? "—" : (signed && v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v * 100).toLocaleString("pt-BR", { minimumFractionDigits: d, maximumFractionDigits: d }) + "%";
const n2 = (v: number | null | undefined, d = 3) => (v == null || !isFinite(v) ? "—" : v.toLocaleString("pt-BR", { minimumFractionDigits: d, maximumFractionDigits: d }));
const cls = (v: number | null | undefined) => (v == null ? "" : v > 0 ? "up" : v < 0 ? "dn" : "");
const U = () => <small className="u">USDT</small>;
const Money = ({ v, p, signed = false, color = false }: { v: number | null; p?: number | null; signed?: boolean; color?: boolean }) => (
  <span className={color ? cls(v) : ""}>{money(v, signed)}<U />{p !== undefined && <> {pct(p, signed)}</>}</span>
);

export interface TesterProps {
  bt: Backtest | null; spec: Strategy | null; symbol: string; tf: string; fee: Fee | null; source: "real" | "ha";
  setSource(s: "real" | "ha"): void; tab: TTab; setTab(t: TTab): void; collapsed: boolean; setCollapsed(c: boolean): void;
  capital: number; setCapital(c: number): void; onFocusTrade(n: number, t0: number, t1: number): void; max: boolean; setMax(m: boolean): void;
}

/** Full OHLCV over the backtest window, for per-trade run-up / drawdown (skipped when too many bars). */
function useBtBars(bt: Backtest | null, symbol: string, tf: string) {
  const [bars, setBars] = useState<Bar[] | null>(null);
  useEffect(() => {
    setBars(null);
    if (!bt || !bt.trades.length) return;
    const t0 = bt.trades[0].entry_time, t1 = bt.holdout_start;
    if ((t1 - t0) / TF_SEC[tf] > 60000) return;
    const ac = new AbortController();
    getOhlcv(symbol, tf, t0 - TF_SEC[tf], t1, ac.signal).then((r) => setBars(r.bars)).catch(() => {});
    return () => ac.abort();
  }, [bt, symbol, tf]);
  return bars;
}

// ------------------------------------------------------------------ Overview
function Overview({ s, C }: { s: Stats; C: number }) {
  const el = useRef<HTMLDivElement>(null);
  const ch = useRef<IChartApi | null>(null);
  const ser = useRef<{ pnl: ISeriesApi<"Baseline">; bh: ISeriesApi<"Line">; dd: ISeriesApi<"Area"> } | null>(null);
  const [show, setShow] = useState({ pnl: true, bh: false, dd: true });
  const leg = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const m = createChart(el.current!, {
      autoSize: true, layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#7c8596", fontSize: 10.5, fontFamily: "'JetBrains Mono Variable', monospace", attributionLogo: false, panes: { separatorColor: "#1c222d" } },
      grid: { vertLines: { visible: false }, horzLines: { color: "#11161e" } }, crosshair: { mode: CrosshairMode.Magnet },
      timeScale: { borderColor: "#1c222d", rightOffset: 2 }, rightPriceScale: { borderColor: "#1c222d" }, localization: { locale: "pt-BR" },
      handleScroll: false, handleScale: false,
    });
    const pf = { type: "custom" as const, minMove: 0.01, formatter: (v: number) => money(v) };
    const pnl = m.addSeries(BaselineSeries, {
      baseValue: { type: "price", price: 0 }, lineWidth: 2, priceFormat: pf,
      topLineColor: UP, topFillColor1: "rgba(14,168,148,0.28)", topFillColor2: "rgba(14,168,148,0.02)",
      bottomLineColor: DN, bottomFillColor1: "rgba(242,92,84,0.02)", bottomFillColor2: "rgba(242,92,84,0.25)", title: "",
    });
    const bh = m.addSeries(LineSeries, { color: BH, lineWidth: 1, lineStyle: LineStyle.Dotted, priceFormat: pf, priceLineVisible: false });
    const dd = m.addSeries(AreaSeries, { lineColor: "rgba(242,92,84,0.8)", topColor: "rgba(242,92,84,0.05)", bottomColor: "rgba(242,92,84,0.35)", lineWidth: 1, priceFormat: { type: "custom", minMove: 0.0001, formatter: (v: number) => pct(v, true) }, priceLineVisible: false, invertFilledArea: true }, 1);
    m.panes()[1]?.setStretchFactor(0.35);
    ser.current = { pnl, bh, dd };
    ch.current = m;
    m.subscribeCrosshairMove((p) => {
      const e = leg.current;
      if (!e) return;
      const a = p.seriesData.get(pnl) as { value?: number } | undefined, b = p.seriesData.get(bh) as { value?: number } | undefined, c = p.seriesData.get(dd) as { value?: number } | undefined;
      e.dataset.t = p.time ? fmtTime(p.time as number) : "";
      e.querySelectorAll<HTMLElement>("[data-k]").forEach((x) => {
        const v = x.dataset.k === "pnl" ? a?.value : x.dataset.k === "bh" ? b?.value : c?.value;
        x.textContent = v == null ? "" : x.dataset.k === "dd" ? pct(v, true) : money(v, true);
      });
    });
    return () => { m.remove(); ch.current = null; };
  }, []);
  useEffect(() => {
    const r = ser.current;
    if (!r) return;
    const T = (t: number) => t as UTCTimestamp;
    // decimate long curves for speed (keeps last point exact)
    const step = Math.max(1, Math.floor(s.curve.length / 3000));
    const c = s.curve.filter((_, i) => i % step === 0 || i === s.curve.length - 1);
    r.pnl.setData(c.map((p) => ({ time: T(p.time), value: p.pnl })));
    r.bh.setData(c.map((p) => ({ time: T(p.time), value: p.bh })));
    r.dd.setData(c.map((p) => ({ time: T(p.time), value: p.ddPct })));
    ch.current?.timeScale().fitContent();
  }, [s]);
  useEffect(() => {
    const r = ser.current;
    if (!r) return;
    r.pnl.applyOptions({ visible: show.pnl }); r.bh.applyOptions({ visible: show.bh }); r.dd.applyOptions({ visible: show.dd });
  }, [show]);
  const a = s.all;
  const last = s.curve[s.curve.length - 1];
  return (
    <div className="ov">
      <h3>Principais estatísticas</h3>
      <div className="kstrip">
        <Stat label="L&P total"><span className={cls(a.net)}>{money(a.net, true)}<U /> {pct(a.netPct, true)}</span></Stat>
        <Stat label="Drawdown máximo"><span>{money(s.maxDD)}<U /> {pct(s.maxDDPct)}</span></Stat>
        <Stat label="Negociações lucrativas"><span>{pct(a.winRate)} <b className="mut">{a.wins}/{a.n}</b></span></Stat>
        <Stat label="Fator de lucro"><span>{n2(a.pf)}</span></Stat>
        <Stat label="Sharpe"><span>{n2(s.sharpe)}</span></Stat>
        <Stat label="Buy e hold"><span className={cls(s.bhReturn)}>{pct(s.bhReturn, true)}</span></Stat>
      </div>
      <div className="ovchart">
        <div className="ovhead">
          <h4>Desempenho</h4>
          <div className="ovleg" ref={leg}>
            <button className={show.pnl ? "on" : ""} onClick={() => setShow({ ...show, pnl: !show.pnl })}><i style={{ background: UP }} />L&amp;P acumulado <b data-k="pnl" className="up">{last ? money(last.pnl, true) : ""}</b></button>
            <button className={show.bh ? "on" : ""} onClick={() => setShow({ ...show, bh: !show.bh })}><i style={{ background: BH }} />Buy e hold <b data-k="bh">{last ? money(last.bh, true) : ""}</b></button>
            <button className={show.dd ? "on" : ""} onClick={() => setShow({ ...show, dd: !show.dd })}><i style={{ background: DN }} />Drawdown <b data-k="dd" className="dn">{last ? pct(last.ddPct, true) : ""}</b></button>
            <button disabled title="Sem alavancagem: nenhuma chamada de margem possível"><i style={{ background: "#ff4d6d" }} />Chamadas de margem <b>0</b></button>
          </div>
        </div>
        <div className="ovplot" ref={el} />
      </div>
      <span className="hidden">{C}</span>
    </div>
  );
}

// ------------------------------------------------------------------ Performance analysis
type APill = "det" | "per" | "bench" | "margin" | "growth";
type NPill = "dist" | "seq" | "time";
function Analysis({ s, fee }: { s: Stats; fee: Fee | null }) {
  const [pill, setPill] = useState<APill>("det");
  const [np, setNp] = useState<NPill>("dist");
  const [per, setPer] = useState<Period>("W");
  const [bySide, setBySide] = useState<"sig" | "dir">("sig");
  const a = s.all;
  const fmtP = (t: number) => {
    const d = new Date(t * 1000);
    if (per === "Y") return String(d.getUTCFullYear());
    if (per === "Q") return `T${Math.floor(d.getUTCMonth() / 3) + 1} ${String(d.getUTCFullYear()).slice(2)}`;
    return d.toLocaleDateString("pt-BR", { timeZone: "UTC", day: "numeric", month: "short" }).replace(".", "");
  };
  const realized = useMemo(() => realizedByPeriod(s, per), [s, per]);
  const bench = useMemo(() => benchByPeriod(s, per), [s, per]);
  const hist = useMemo(() => histogram(s), [s]);
  const maxPL = Math.max(1, a.grossP, a.grossL);
  const rows: [string, SideStats][] = [["Todos os sinais", s.all], ["long", s.long], ["short", s.short]];
  const dow = useMemo(() => {
    const names = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"];
    const r = names.map((x) => ({ x, v: { win: 0, loss: 0 } }));
    for (const t of s.trades) { const d = (new Date(t.entry_time * 1000).getUTCDay() + 6) % 7; if (t.profit >= 0) r[d].v.win += t.profit; else r[d].v.loss += t.profit; }
    return r;
  }, [s]);
  const hours = useMemo(() => {
    const r = Array.from({ length: 24 }, (_, h) => ({ x: String(h).padStart(2, "0") + "h", v: { win: 0, loss: 0 } }));
    for (const t of s.trades) { const h = new Date(t.entry_time * 1000).getUTCHours(); if (t.profit >= 0) r[h].v.win += t.profit; else r[h].v.loss += t.profit; }
    return r.filter((x) => x.v.win || x.v.loss);
  }, [s]);
  // drawdown episodes (underwater periods) for "Crescimento e declínio"
  const episodes = useMemo(() => {
    const out: { start: number; end: number; depth: number; days: number }[] = [];
    let st = 0, depth = 0;
    for (const c of s.curve) {
      if (c.dd < 0) { if (!st) st = c.time; depth = Math.min(depth, c.ddPct); }
      else if (st) { out.push({ start: st, end: c.time, depth, days: (c.time - st) / 86400 }); st = 0; depth = 0; }
    }
    if (st) out.push({ start: st, end: s.curve[s.curve.length - 1].time, depth, days: (s.curve[s.curve.length - 1].time - st) / 86400 });
    return out.sort((x, y) => x.depth - y.depth).slice(0, 5);
  }, [s]);
  return (
    <div className="an">
      <h3>Análise de desempenho</h3>
      <Pills<APill> items={[["det", "Detalhamento"], ["per", "Periódico"], ["bench", "Benchmarking"], ["margin", "Uso de margem"], ["growth", "Crescimento e declínio"]]} value={pill} onChange={setPill} />
      {pill === "det" && (<>
        <div className="kstrip">
          <Stat label="Lucro bruto"><Money v={a.grossP} p={a.grossPPct} /></Stat>
          <Stat label="Prejuízo bruto"><Money v={a.grossL} p={a.grossLPct} /></Stat>
          <Stat label="Fator de lucro"><span>{n2(a.pf)}</span></Stat>
          <Stat label="Taxa de comissão"><Money v={a.commission} /></Stat>
        </div>
        <div className="sechead"><h4>Lucros e Prejuízos</h4><Seg<"sig" | "dir"> items={[["sig", "Por sinais"], ["dir", "Por direção"]]} value={bySide} onChange={setBySide} /></div>
        <div className="plbars">
          {(bySide === "sig" ? rows : rows.slice(1)).map(([l, x]) => (
            <div key={l} className="plrow">
              <span>{l}</span>
              <div className="plbar">
                <i className="loss" style={{ width: `${(x.grossL / maxPL) * 100}%` }} title={`Prejuízo bruto ${money(x.grossL)}`} />
                <i className="prof" style={{ width: `${(x.grossP / maxPL) * 100}%` }} title={`Lucro bruto ${money(x.grossP)}`} />
              </div>
              <b className={cls(x.net)}>{money(x.net, true)}<U /></b>
            </div>
          ))}
        </div>
        <h3 className="mt">Análise de negociações</h3>
        <Pills<NPill> items={[["dist", "Distribuição"], ["seq", "Sequências"], ["time", "Padrões de tempo"]]} value={np} onChange={setNp} />
        {np === "dist" && (<>
          <div className="kstrip">
            <Stat label="Expectativa"><Money v={a.expectancy} p={a.expectancyPct} /></Stat>
            <Stat label="P&L de valores atípicos" hint="Soma do L&P das negociações fora de 1,5×IQR"><Money v={s.outlier} p={s.outlierPct} /></Stat>
            <Stat label="Maior lucro"><Money v={a.bigWin} p={a.bigWinPct} /></Stat>
            <Stat label="Maior perda"><Money v={a.bigLoss} p={a.bigLossPct} /></Stat>
          </div>
          <div className="duo">
            <div>
              <h4>Distribuição dos retornos</h4>
              <BarChart height={170} minBar={8} rows={hist.bins.map((b) => ({ x: `${b.x.toLocaleString("pt-BR")}%`, tip: `${b.x}% a ${b.x + hist.width}%`, v: { n: b.n } }))}
                series={[{ key: "n", label: "Negociações", color: UP }]} colorOf={(r) => (r.x.startsWith("-") || r.x.startsWith("−") ? DN : UP)} yFmt={(v) => String(Math.round(v))} legend={false}
                refLines={[...(hist.meanLoss != null ? [{ x: hist.meanLoss, color: DN, label: `Perda média ${hist.meanLoss.toFixed(2)}%` }] : []), ...(hist.meanWin != null ? [{ x: hist.meanWin, color: UP, label: `Lucro médio ${hist.meanWin.toFixed(2)}%` }] : [])]} />
              <div className="blegend"><span><i style={{ background: DN }} />Perdedores</span><span><i style={{ background: UP }} />Vencedores</span>
                {hist.meanLoss != null && <span><i className="dash" style={{ borderColor: DN }} />Perda média {hist.meanLoss.toFixed(2).replace(".", ",")}%</span>}
                {hist.meanWin != null && <span><i className="dash" style={{ borderColor: UP }} />Lucro médio {hist.meanWin.toFixed(2).replace(".", ",")}%</span>}</div>
            </div>
            <div>
              <h4>Distribuição de negociações</h4>
              <Donut total={a.n} label="Total de negociações" parts={[{ label: "Vencedores", n: a.wins, color: UP }, { label: "Perdedores", n: a.losses, color: DN }, { label: "Breakevens", n: a.even, color: "#ffb020" }]} />
            </div>
          </div>
        </>)}
        {np === "seq" && (
          <div className="kstrip">
            <Stat label="Máx. vitórias seguidas"><span>{a.maxConsecWins}</span></Stat>
            <Stat label="Máx. perdas seguidas"><span>{a.maxConsecLosses}</span></Stat>
            <Stat label="Média de barras (vencedoras)"><span>{n2(a.avgBarsWin, 0)}</span></Stat>
            <Stat label="Média de barras (perdedoras)"><span>{n2(a.avgBarsLoss, 0)}</span></Stat>
          </div>
        )}
        {np === "time" && (
          <div className="duo">
            <div><h4>L&amp;P por dia da semana (entrada)</h4><BarChart height={160} rows={dow} series={[{ key: "win", label: "Lucro realizado", color: UP }, { key: "loss", label: "Prejuízo realizado", color: DN }]} /></div>
            <div><h4>L&amp;P por hora UTC (entrada)</h4><BarChart height={160} minBar={5} rows={hours} series={[{ key: "win", label: "Lucro realizado", color: UP }, { key: "loss", label: "Prejuízo realizado", color: DN }]} /></div>
          </div>
        )}
      </>)}
      {pill === "per" && (<>
        <div className="kstrip">
          <Stat label="Retorno anualizado (CAGR)"><span className={cls(s.cagr)}>{pct(s.cagr, true)}</span></Stat>
          <Stat label="Retorno total"><span className={cls(a.netPct)}>{pct(a.netPct, true)}</span></Stat>
          <Stat label="Razão de Sharpe"><span>{n2(s.sharpe)}</span></Stat>
          <Stat label="Razão de Sortino"><span>{n2(s.sortino)}</span></Stat>
        </div>
        <div className="sechead"><h4>L&amp;P {PERIODS.find((p) => p[0] === per)![1].toLowerCase()}</h4><Seg<Period> items={PERIODS} value={per} onChange={setPer} /></div>
        <BarChart height={190} minBar={7} rows={realized.map((r) => ({ x: fmtP(r.t), tip: fmtP(r.t) + " " + new Date(r.t * 1000).getUTCFullYear(), v: { win: r.win, loss: r.loss, mfe: r.mfe, mae: r.mae } }))}
          series={[{ key: "win", label: "Lucro realizado", color: UP }, { key: "loss", label: "Prejuízo realizado", color: DN }, { key: "mfe", label: "Excursão favorável", color: "rgba(14,168,148,0.35)" }, { key: "mae", label: "Excursão adversa", color: "rgba(242,92,84,0.35)" }]} />
      </>)}
      {pill === "bench" && (<>
        <div className="kstrip">
          <Stat label="Retorno da estratégia"><span className={cls(a.netPct)}>{pct(a.netPct, true)}</span></Stat>
          <Stat label="Retorno buy e hold"><span className={cls(s.bhReturn)}>{pct(s.bhReturn, true)}</span></Stat>
          <Stat label="Desempenho superior da estratégia"><span className={cls(a.netPct - s.bhReturn)}>{pct(a.netPct - s.bhReturn, true)}</span></Stat>
          <Stat label="Correlação" hint={`Correlação dos L&P por período (${PERIODS.find((p) => p[0] === per)![1].toLowerCase()})`}><span>{n2(bench.corr)}</span></Stat>
        </div>
        <div className="sechead"><h4>Estratégia vs. benchmark</h4><Seg<Period> items={PERIODS} value={per} onChange={setPer} /></div>
        <BarChart height={190} minBar={6} rows={bench.rows.map((r) => ({ x: fmtP(r.t), tip: fmtP(r.t) + " " + new Date(r.t * 1000).getUTCFullYear(), v: { strat: r.strat, bh: r.bh } }))}
          series={[{ key: "strat", label: "L&P da estratégia", color: EQ }, { key: "bh", label: "L&P de buy e hold", color: BH }]} />
      </>)}
      {pill === "margin" && (
        <div className="kstrip">
          <Stat label="Alavancagem"><span>1×</span></Stat>
          <Stat label="Margem por operação"><span>100% do patrimônio</span></Stat>
          <Stat label="Exposição (tempo posicionado)"><span>{pct(s.exposure, false, 1)}</span></Stat>
          <Stat label="Chamadas de margem"><span>0</span></Stat>
        </div>
      )}
      {pill === "growth" && (<>
        <div className="kstrip">
          <Stat label="Run-up máximo"><Money v={s.maxRunup} p={s.maxRunupPct} /></Stat>
          <Stat label="Drawdown máximo"><Money v={s.maxDD} p={s.maxDDPct} /></Stat>
          <Stat label="Retorno / drawdown"><span>{n2(s.maxDDPct ? (s.cagr ?? 0) / s.maxDDPct : null, 2)}</span></Stat>
          <Stat label="Taxas" hint={fee?.note}><span>{fee ? `${(fee.fee_rate * 100).toFixed(3)}% + ${fee.slippage_bp} bp` : "—"}</span></Stat>
        </div>
        <h4>Piores drawdowns</h4>
        <table className="ttable compact">
          <thead><tr><th>Início</th><th>Recuperação</th><th className="r">Profundidade</th><th className="r">Duração</th></tr></thead>
          <tbody>{episodes.map((e) => <tr key={e.start}><td>{fmtDate(e.start)}</td><td>{fmtDate(e.end)}</td><td className="r dn">{pct(e.depth, true)}</td><td className="r">{Math.round(e.days)} dias</td></tr>)}</tbody>
        </table>
      </>)}
      <span className="hidden">{bucket(0, per)}</span>
    </div>
  );
}

// ------------------------------------------------------------------ Performance summary table
function Summary({ s }: { s: Stats }) {
  const R: [string, (x: SideStats) => React.ReactNode][] = [
    ["Lucro líquido", (x) => <><Money v={x.net} signed color /><em>{pct(x.netPct, true)}</em></>],
    ["Lucro bruto", (x) => <><Money v={x.grossP} /><em>{pct(x.grossPPct)}</em></>],
    ["Prejuízo bruto", (x) => <><Money v={x.grossL} /><em>{pct(x.grossLPct)}</em></>],
    ["Fator de lucro", (x) => n2(x.pf)],
    ["Drawdown máximo (fechado)", (x) => <><Money v={x.maxDD} /><em>{pct(x.maxDDPct)}</em></>],
    ["Run-up máximo (fechado)", (x) => <Money v={x.maxRunup} />],
    ["Comissão paga", (x) => <Money v={x.commission} />],
    ["Razão de Sharpe", (x) => n2(x.sharpe)],
    ["Razão de Sortino", (x) => n2(x.sortino)],
    ["Total de negociações", (x) => String(x.n)],
    ["Negociações vencedoras", (x) => String(x.wins)],
    ["Negociações perdedoras", (x) => String(x.losses)],
    ["Percentual lucrativo", (x) => pct(x.winRate)],
    ["Média de L&P", (x) => <><Money v={x.avg} signed color /><em>{pct(x.avgPct, true)}</em></>],
    ["Média de negociação vencedora", (x) => <><Money v={x.avgWin} /><em>{pct(x.avgWinPct)}</em></>],
    ["Média de negociação perdedora", (x) => <><Money v={x.avgLoss} /><em>{pct(x.avgLossPct)}</em></>],
    ["Razão média ganho / média perda", (x) => n2(x.ratio)],
    ["Maior negociação vencedora", (x) => <><Money v={x.bigWin} /><em>{pct(x.bigWinPct)}</em></>],
    ["Maior negociação perdedora", (x) => <><Money v={x.bigLoss} /><em>{pct(x.bigLossPct)}</em></>],
    ["Média de barras nas negociações", (x) => n2(x.avgBars, 0)],
    ["Média de barras (vencedoras)", (x) => n2(x.avgBarsWin, 0)],
    ["Média de barras (perdedoras)", (x) => n2(x.avgBarsLoss, 0)],
  ];
  return (
    <table className="ttable sum">
      <thead><tr><th>Métrica</th><th className="r">Todos</th><th className="r">Long</th><th className="r">Short</th></tr></thead>
      <tbody>
        <tr><td>Retorno buy e hold</td><td className="r"><Money v={s.bhPnl} signed color /><em>{pct(s.bhReturn, true)}</em></td><td /><td /></tr>
        {R.map(([l, f]) => <tr key={l}><td>{l}</td><td className="r">{f(s.all)}</td><td className="r">{s.long.n ? f(s.long) : "—"}</td><td className="r">{s.short.n ? f(s.short) : "—"}</td></tr>)}
      </tbody>
    </table>
  );
}

// ------------------------------------------------------------------ Trades list
function Trades({ s, onFocus }: { s: Stats; onFocus(n: number, t0: number, t1: number): void }) {
  const [desc, setDesc] = useState(true);
  const [sel, setSel] = useState<number | null>(null);
  const list = desc ? [...s.trades].reverse() : s.trades;
  return (
    <table className="ttable trl">
      <thead><tr>
        <th className="clk" onClick={() => setDesc(!desc)}>Negociação # {desc ? "↓" : "↑"}</th><th>Tipo</th><th>Sinal</th><th>Data/Hora</th><th className="r">Preço</th>
        <th className="r">Contratos</th><th className="r">Lucro</th><th className="r">Lucro cumulativo</th><th className="r">Run-up</th><th className="r">Drawdown</th>
      </tr></thead>
      <tbody>
        {list.map((t) => (
          <tr key={t.n} className={sel === t.n ? "sel" : ""} onClick={() => { setSel(t.n); onFocus(t.n, t.entry_time, t.exit_time); }} title="Clique para mostrar no gráfico">
            <td><b>{t.n}</b> <span className={"side " + (t.long ? "up" : "dn")}>{t.long ? "Long" : "Short"}</span></td>
            <td><div>Saída</div><div className="mut">Entrada</div></td>
            <td><div>{reasonPt(t.exit_reason)}</div><div className="mut">{t.long ? "long" : "short"}</div></td>
            <td><div>{fmtTime(t.exit_time).replace(" UTC", "")}</div><div className="mut">{fmtTime(t.entry_time).replace(" UTC", "")}</div></td>
            <td className="r"><div>{fmtPrice(t.exit_price)}</div><div className="mut">{fmtPrice(t.entry_price)}</div></td>
            <td className="r">{n2(t.contracts, 4)}</td>
            <td className="r"><div className={cls(t.profit)}>{money(t.profit, true)}<U /></div><div className={"mut " + cls(t.pnl_pct)}>{pct(t.pnl_pct, true)}</div></td>
            <td className="r"><div className={cls(t.cum)}>{money(t.cum, true)}<U /></div><div className="mut">{pct(t.cum / s.C, true)}</div></td>
            <td className="r"><div>{t.runup != null ? money(t.runup) : "—"}</div><div className="mut">{pct(t.runupPct)}</div></td>
            <td className="r"><div>{t.dd != null ? money(-t.dd, true) : "—"}</div><div className="mut">{t.ddPct != null ? pct(-t.ddPct, true) : "—"}</div></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ------------------------------------------------------------------ Properties
function Props({ s, bt, spec, fee, tf, symbol, source }: { s: Stats; bt: Backtest; spec: Strategy | null; fee: Fee | null; tf: string; symbol: string; source: string }) {
  const rows: [string, React.ReactNode][] = [
    ["Símbolo", `${symbol} perp (Binance USDT-M)`], ["Timeframe", TF_LABEL[tf]],
    ["Período in-sample", `${fmtDate(s.start)} — ${fmtDate(s.end)} (${s.years.toFixed(1).replace(".", ",")} anos)`],
    ["Holdout (fora da amostra)", `a partir de ${fmtDate(bt.holdout_start)}: apenas sinais, sem estatísticas`],
    ["Capital inicial", <Money v={s.C} />], ["Tamanho da ordem", "100% do patrimônio, composto"], ["Pirâmide", "não (1 posição por vez)"],
    ["Custos", fee ? `${fee.name}: ${(fee.fee_rate * 100).toFixed(3)}% por lado + ${fee.slippage_bp} bp de slippage${fee.allow_short ? "" : " · somente long"}` : "—"],
    ["Funding", "incluído (perp)"],
    ["Fonte dos sinais", source === "ha" ? "Heikin Ashi (sinais calculados sobre velas HA)" : "Velas normais"],
    ["Execução", "entradas, saídas e stops sempre a preços reais (OHLC normal), mesmo com sinais em Heikin Ashi"],
  ];
  const sp = spec as unknown as Record<string, unknown> | null;
  return (
    <div className="props">
      <table className="ttable compact">
        <tbody>{rows.map(([k, v]) => <tr key={k}><td className="mut">{k}</td><td>{v}</td></tr>)}</tbody>
      </table>
      {sp && (
        <table className="ttable compact">
          <thead><tr><th colSpan={2}>Parâmetros da estratégia · {String(sp.name)}</th></tr></thead>
          <tbody>
            {Object.entries(sp).filter(([k]) => k !== "name").map(([k, v]) => (
              <tr key={k}><td className="mut">{k}</td><td><code>{typeof v === "string" ? v : JSON.stringify(v)}</code></td></tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ shell
function Tester(p: TesterProps) {
  const bars = useBtBars(p.bt, p.symbol, p.tf);
  const s = useMemo(() => (p.bt ? computeStats(p.bt, p.capital, p.fee?.fee_rate ?? 0, TF_SEC[p.tf], bars) : null), [p.bt, p.capital, p.fee, p.tf, bars]);
  const [capOpen, setCapOpen] = useState(false);
  return (
    <section className={"tester" + (p.collapsed ? " collapsed" : "")}>
      <header className="thead">
        <h2>{p.spec?.name ?? "Estratégia"}</h2>
        <nav className="ttabs">{TABS.map(([k, l]) => <button key={k} className={p.tab === k && !p.collapsed ? "on" : ""} onClick={() => { p.setTab(k); p.setCollapsed(false); }}>{l}</button>)}</nav>
        <div className="spacer" />
        {s && !p.collapsed && (<>
          <span className="tchip" title="Período in-sample">▦ {fmtDate(s.start)} — {fmtDate(s.end)}</span>
          <div className="tcap">
            <button className="tchip" onClick={() => setCapOpen(!capOpen)} onBlur={() => setTimeout(() => setCapOpen(false), 150)}>$ {kfmt(p.capital)} <small>USDT</small></button>
            {capOpen && <div className="menu up">{CAPS.map((c) => <button key={c} className={c === p.capital ? "on" : ""} onMouseDown={(e) => { e.preventDefault(); p.setCapital(c); setCapOpen(false); }}>{kfmt(c)}</button>)}</div>}
          </div>
          <span className="tchip" title="Estatísticas apenas in-sample; holdout só mostra sinais">Detalhes padrão · in-sample</span>
        </>)}
        <label className="tg mini" title="Sinais calculados sobre velas Heikin Ashi (execução a preços reais)">
          <input type="checkbox" checked={p.source === "ha"} onChange={(e) => p.setSource(e.target.checked ? "ha" : "real")} /><i />Sinais em Heikin Ashi
        </label>
        <button className="ticon" onClick={() => p.setMax(!p.max)} title={p.max ? "Restaurar" : "Maximizar"}>{p.max ? "▭" : "⤢"}</button>
        <button className="ticon" onClick={() => p.setCollapsed(!p.collapsed)} title={p.collapsed ? "Expandir" : "Recolher"}>{p.collapsed ? "▴" : "▾"}</button>
      </header>
      {!p.collapsed && (
        <div className="tbody">
          {!p.bt || !s ? <div className="tempty">{p.bt === null ? "Carregando backtest…" : "Sem dados"}</div> : (<>
            {p.tab === "overview" && <Overview s={s} C={p.capital} />}
            {p.tab === "analysis" && <Analysis s={s} fee={p.fee} />}
            {p.tab === "summary" && <Summary s={s} />}
            {p.tab === "trades" && <Trades s={s} onFocus={p.onFocusTrade} />}
            {p.tab === "props" && <Props s={s} bt={p.bt} spec={p.spec} fee={p.fee} tf={p.tf} symbol={p.symbol} source={p.source} />}
          </>)}
        </div>
      )}
    </section>
  );
}
export default memo(Tester);
