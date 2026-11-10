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

/** "Paper trading" tab: forward (out-of-sample) ledger per tracked strategy. */
export default function Paper({ sel, setSel }: { sel: string | null; setSel(id: string | null): void }) {
  const [list, setList] = useState<PaperStrategy[] | null>(null);
  const [st, setSt] = useState<PaperStatus | null>(null);
  const [pts, setPts] = useState<PaperPoint[]>([]);
  const [err, setErr] = useState("");
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
        <span className={"st " + (st?.running ? "ok" : st ? "bad" : "")}><i />{st == null ? "serviço …" : st.running ? "serviço ativo" : "serviço parado"}</span>
        {st?.lag_s != null && <span className="mut">última execução há {Math.round(st.lag_s)} s</span>}
        {st?.current_errors?.length ? <span className="dn" title={st.current_errors.join("\n")}>{st.current_errors.length} erro(s): {st.current_errors[0].slice(0, 90)}</span> : null}
        <span className="mut">Ledger forward fora da amostra · fills no open do candle seguinte · {money(10000)} USDT iniciais · clique numa linha para ver os fills no gráfico</span>
      </div>
      {err && <div className="tempty">Erro: {err}</div>}
      {list && !list.length && <div className="tempty">Nenhuma estratégia registrada ainda. Inicie scripts/start_paper.ps1.</div>}
      {list && list.length > 0 && (
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
      {cur && (pts.length > 1 ? <EquityChart pts={pts} /> : <div className="tempty">Equity aparece após o fechamento do primeiro candle depois do início ({fmtTime(firstClose(cur))}).</div>)}
    </div>
  );
}
