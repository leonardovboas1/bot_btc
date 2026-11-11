import { useEffect, useRef, useState } from "react";
import { ColorType, CrosshairMode, LineSeries, LineStyle, createChart, type IChartApi, type ISeriesApi, type UTCTimestamp } from "lightweight-charts";
import { getPaperEquity, getPaperStatus, getPaperStrategies, type PaperPoint, type PaperStatus, type PaperStrategy } from "../paperApi";
import { TF_LABEL, TF_SEC, fmtPrice, fmtTime } from "../util";

const EQ = "#4c8dff", BH = "#5d6676";
const pct = (v: number | null | undefined) =>
  v == null || !isFinite(v) ? "—" : (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v * 100).toLocaleString("pt-BR", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + "%";
const money = (v: number | null | undefined) => (v == null || !isFinite(v) ? "—" : v.toLocaleString("pt-BR", { minimumFractionDigits: 2, maximumFractionDigits: 2 }));
const cls = (v: number | null | undefined) => (v == null ? "" : v > 0 ? "up" : v < 0 ? "dn" : "");
/** First bar close at/after the strategy start: nothing is recorded before it. */
const firstClose = (s: PaperStrategy) => Math.ceil(s.start_ts / TF_SEC[s.tf]) * TF_SEC[s.tf];

function sigText(s: PaperStrategy) {
  const g = s.last_signal;
  if (!g) return "—";
  const f = [g.long && "long", g.short && "short", g.exit_long && "saída long", g.exit_short && "saída short"].filter(Boolean);
  return f.length ? f.join(" + ") : "sem sinal";
}

function EquityChart({ pts }: { pts: PaperPoint[] }) {
  const el = useRef<HTMLDivElement>(null);
  const ser = useRef<{ eq: ISeriesApi<"Line">; bh: ISeriesApi<"Line"> } | null>(null);
  const ch = useRef<IChartApi | null>(null);
  useEffect(() => {
    const m = createChart(el.current!, {
      autoSize: true, layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#7c8596", fontSize: 10.5, fontFamily: "-apple-system, BlinkMacSystemFont, 'Trebuchet MS', Roboto, Ubuntu, sans-serif", attributionLogo: false },
      grid: { vertLines: { visible: false }, horzLines: { color: "#11161e" } }, crosshair: { mode: CrosshairMode.Magnet },
      timeScale: { borderColor: "#1c222d", rightOffset: 2, timeVisible: true }, rightPriceScale: { borderColor: "#1c222d" }, localization: { locale: "pt-BR" },
    });
    const pf = { type: "custom" as const, minMove: 0.01, formatter: (v: number) => money(v) };
    ser.current = {
      eq: m.addSeries(LineSeries, { color: EQ, lineWidth: 2, priceFormat: pf, priceLineVisible: false, title: "Paper" }),
      bh: m.addSeries(LineSeries, { color: BH, lineWidth: 1, lineStyle: LineStyle.Dotted, priceFormat: pf, priceLineVisible: false, title: "B&H" }),
    };
    ch.current = m;
    return () => { m.remove(); ch.current = null; };
  }, []);
  useEffect(() => {
    const r = ser.current;
    if (!r) return;
    r.eq.setData(pts.map((p) => ({ time: p.time as UTCTimestamp, value: p.equity })));
    r.bh.setData(pts.map((p) => ({ time: p.time as UTCTimestamp, value: p.buy_hold })));
    ch.current?.timeScale().fitContent();
  }, [pts]);
  return <div className="ovplot" ref={el} style={{ height: 150, marginTop: 10 }} />;
}

interface Row { name: string; members: PaperStrategy[]; ret: number | null; alpha: number | null; mdd: number | null; bhMdd: number | null; wins: number; closed: number; open: number; days: number }
const mean = (xs: (number | null)[]) => { const v = xs.filter((x): x is number => x != null && isFinite(x)); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };

/** Families aggregated over their coins, ranked by mean excess return over buy & hold, then by drawdown. */
function rankRows(list: PaperStrategy[], by: "family" | "each"): Row[] {
  const groups = new Map<string, PaperStrategy[]>();
  for (const s of list) {
    const k = by === "family" ? `${s.name} · ${TF_LABEL[s.tf] ?? s.tf} · ${s.source === "ha" ? "HA" : "real"}` : `${s.name} · ${s.symbol.replace("USDT", "")} · ${TF_LABEL[s.tf] ?? s.tf} · ${s.source === "ha" ? "HA" : "real"}`;
    groups.set(k, [...(groups.get(k) ?? []), s]);
  }
  const rows: Row[] = [...groups].map(([name, m]) => ({
    name, members: m,
    ret: mean(m.map((s) => s.return)),
    alpha: mean(m.map((s) => (s.return == null || s.bh_return == null ? null : s.return - s.bh_return))),
    mdd: m.some((s) => s.mdd != null) ? Math.max(...m.map((s) => s.mdd ?? 0)) : null,
    bhMdd: m.some((s) => s.bh_mdd != null) ? Math.max(...m.map((s) => s.bh_mdd ?? 0)) : null,
    wins: m.reduce((a, s) => a + (s.wins ?? 0), 0), closed: m.reduce((a, s) => a + (s.closed ?? 0), 0),
    open: m.filter((s) => s.position !== "FLAT").length,
    days: Math.max(0, ...m.map((s) => s.days ?? 0)),
  }));
  const k = (x: number | null) => (x == null ? -Infinity : x);
  return rows.sort((a, b) => k(b.alpha) - k(a.alpha) || k(-(a.mdd ?? 1)) - k(-(b.mdd ?? 1)) || a.name.localeCompare(b.name));
}

function Ranking({ list, onPick }: { list: PaperStrategy[]; onPick(id: string): void }) {
  const [by, setBy] = useState<"family" | "each">("family");
  const rows = rankRows(list, by);
  const days = Math.max(0, ...list.map((s) => s.days ?? 0));
  const closed = list.reduce((a, s) => a + (s.closed ?? 0), 0);
  return (
    <>
      <div className="paperst">
        <span className="seg">
          <button className={by === "family" ? "on" : ""} onClick={() => setBy("family")}>Por estratégia</button>
          <button className={by === "each" ? "on" : ""} onClick={() => setBy("each")}>Por moeda</button>
        </span>
        <span className="mut">Ordenado pelo excesso sobre buy &amp; hold (média das moedas), desempate pelo menor drawdown · {days.toLocaleString("pt-BR", { maximumFractionDigits: 1 })} dias de dados · {closed} trades fechados</span>
        {(days < 60 || closed < 30) && <span className="warn">amostra pequena: ranking ainda é ruído, não conclusão</span>}
      </div>
      <table className="ttable trl">
        <thead><tr><th className="r">#</th><th>Estratégia</th><th className="r">Moedas</th><th className="r">Retorno</th><th className="r">vs B&amp;H</th><th className="r">Max DD</th><th className="r">DD B&amp;H</th><th className="r">Acerto</th><th className="r">Trades</th><th className="r">Abertas</th></tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.name} onClick={() => onPick(r.members[0].id)} title="Abrir detalhe e mostrar fills no gráfico">
              <td className="r mut">{i + 1}</td>
              <td><b>{r.name}</b></td>
              <td className="r">{r.members.map((s) => s.symbol.replace("USDT", "")).join(" ")}</td>
              <td className={"r " + cls(r.ret)}>{pct(r.ret)}</td>
              <td className={"r " + cls(r.alpha)}>{pct(r.alpha)}</td>
              <td className="r">{r.mdd == null ? "—" : pct(-r.mdd)}</td>
              <td className="r mut">{r.bhMdd == null ? "—" : pct(-r.bhMdd)}</td>
              <td className="r">{r.closed ? `${Math.round((r.wins / r.closed) * 100)}% (${r.wins}/${r.closed})` : "—"}</td>
              <td className="r">{r.closed}</td>
              <td className="r">{r.open ? r.open : <span className="mut">0</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

/** "Paper trading" tab: forward (out-of-sample) ledger per tracked strategy. */
export default function Paper({ sel, setSel }: { sel: string | null; setSel(id: string | null): void }) {
  const [list, setList] = useState<PaperStrategy[] | null>(null);
  const [st, setSt] = useState<PaperStatus | null>(null);
  const [pts, setPts] = useState<PaperPoint[]>([]);
  const [err, setErr] = useState("");
  const [view, setView] = useState<"rank" | "detail">("rank");
  useEffect(() => {
    let dead = false;
    const ac = new AbortController();
    const load = () => {
      getPaperStrategies(ac.signal).then((r) => { if (!dead) { setList(r); setErr(""); } }).catch((e) => { if (!dead && e?.name !== "AbortError") setErr(String(e.message ?? e)); });
      getPaperStatus(ac.signal).then((r) => { if (!dead) setSt(r); }).catch(() => {});
    };
    load();
    const t = setInterval(load, 20000);
    return () => { dead = true; ac.abort(); clearInterval(t); };
  }, []);
  useEffect(() => {
    if (!sel) { setPts([]); return; }
    let dead = false;
    const ac = new AbortController();
    const load = () => getPaperEquity(sel, ac.signal).then((r) => { if (!dead) setPts(r.points); }).catch(() => {});
    load();
    const t = setInterval(load, 30000);
    return () => { dead = true; ac.abort(); clearInterval(t); };
  }, [sel]);
  const cur = list?.find((s) => s.id === sel) ?? null;
  return (
    <div className="paper">
      <div className="paperst">
        <span className="seg">
          <button className={view === "rank" ? "on" : ""} onClick={() => setView("rank")}>Ranking</button>
          <button className={view === "detail" ? "on" : ""} onClick={() => setView("detail")}>Detalhe</button>
        </span>
        <span className={"st " + (st?.running ? "ok" : st ? "bad" : "")}><i />{st == null ? "serviço …" : st.running ? "serviço ativo" : "serviço parado"}</span>
        {st?.lag_s != null && <span className="mut">última execução há {Math.round(st.lag_s)} s</span>}
        {st?.current_errors?.length ? <span className="dn" title={st.current_errors.join("\n")}>{st.current_errors.length} erro(s): {st.current_errors[0].slice(0, 90)}</span> : null}
        <span className="mut">Ledger forward fora da amostra · fills no open do candle seguinte · {money(10000)} USDT iniciais · clique numa linha para ver os fills no gráfico</span>
      </div>
      {err && <div className="tempty">Erro: {err}</div>}
      {list && !list.length && <div className="tempty">Nenhuma estratégia registrada ainda. Inicie scripts/start_paper.ps1.</div>}
      {list && list.length > 0 && view === "rank" && <Ranking list={list} onPick={(id) => { setSel(id); setView("detail"); }} />}
      {list && list.length > 0 && view === "detail" && (
        <table className="ttable">
          <thead><tr><th>Estratégia</th><th>Início</th><th>Posição</th><th className="r">Negociações</th><th className="r">Equity (USDT)</th><th className="r">Paper</th><th className="r">Buy &amp; hold</th><th>Último sinal</th></tr></thead>
          <tbody>
            {list.map((s) => (
              <tr key={s.id} className={sel === s.id ? "sel" : ""} onClick={() => setSel(sel === s.id ? null : s.id)} title="Mostrar fills paper no gráfico">
                <td><b>{s.name}</b> <span className="mut">{s.symbol} · {TF_LABEL[s.tf] ?? s.tf} · {s.source === "ha" ? "HA" : "real"}</span></td>
                <td>{fmtTime(s.start_ts)}</td>
                <td>{s.position === "FLAT" ? <span className="mut">FLAT</span> : <span className={"side " + (s.position === "LONG" ? "up" : "dn")}>{s.position}</span>}
                  {s.position !== "FLAT" && <span className="mut"> @ {fmtPrice(s.entry_px)}</span>}</td>
                <td className="r">{s.n_trades}{s.n_entries > s.n_trades ? " +1 aberta" : ""}</td>
                <td className="r">{s.equity == null ? <span className="mut">1º candle {fmtTime(firstClose(s))}</span> : money(s.equity)}</td>
                <td className={"r " + cls(s.return)}>{pct(s.return)}</td>
                <td className={"r " + cls(s.bh_return)}>{pct(s.bh_return)}</td>
                <td>{s.last_signal_time ? <>{fmtTime(s.last_signal_time)}<em>{sigText(s)}</em></> : <span className="mut">—</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {view === "detail" && cur && (pts.length > 1 ? <EquityChart pts={pts} /> : <div className="tempty">Equity aparece após o fechamento do primeiro candle depois do início ({fmtTime(firstClose(cur))}).</div>)}
    </div>
  );
}
