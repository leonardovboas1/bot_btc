import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { COMING, FALLBACK_SYMBOLS, getFees, getSignal, getStrategies, getSymbols, type Fee } from "./api";
import type { Backtest, Signal, Strategy, SymbolInfo } from "./types";
import Pane, { type PaneMode } from "./chart/Pane";
import { SyncBus } from "./chart/sync";
import { drawings, type Tool } from "./chart/drawings";
import Inspector from "./ui/Inspector";
import Tester, { type TTab } from "./strategy/Tester";
import Palette, { type Cmd } from "./ui/Palette";
import Price from "./ui/Price";
import { I, Logo } from "./ui/icons";
import { live, type LiveStatus, type Tick } from "./live";
import { ALL_TFS, TF_LABEL, fmtDate, fmtPct, base } from "./util";

type Layout = 1 | 2 | 4;
type Tab = "resumo" | "detalhes" | "trades";
interface Ui {
  layout: Layout; symbol: string; tfs: string[]; modes: PaneMode[]; strategy: string; fee: string;
  inspector: boolean; tab: Tab; details: boolean; eq: boolean; book: boolean; vp: boolean; syncCross: boolean; syncRange: boolean;
  candles: "real" | "ha"; log: boolean; source: "real" | "ha"; testerH: number; testerOpen: boolean; testerMax: boolean; testerTab: TTab; capital: number;
}
const DEF: Ui = {
  layout: 1, symbol: "BTCUSDT", tfs: ["4h", "1h", "1d", "15m"], modes: ["candles", "candles", "candles", "candles"], strategy: "", fee: "",
  inspector: true, tab: "resumo", details: false, eq: false, book: false, vp: false, syncCross: true, syncRange: false,
  candles: "real", log: false, source: "real", testerH: 300, testerOpen: true, testerMax: false, testerTab: "overview", capital: 10000,
};
const KEY = "btclab.ui.v2";
const load = (): Ui => {
  try {
    const u = { ...DEF, ...JSON.parse(localStorage.getItem(KEY) || "{}") } as Ui;
    if (!Array.isArray(u.modes) || u.modes.length < 4) u.modes = DEF.modes;
    if (!Array.isArray(u.tfs) || u.tfs.length < 4) u.tfs = DEF.tfs;
    return u;
  } catch { return DEF; }
};
const TOOLS: [Tool, string, string][] = [
  ["cursor", "Cursor / selecionar", "Esc"], ["trend", "Linha de tendência", "Alt+T"], ["hline", "Linha horizontal", "Alt+H"],
  ["rect", "Retângulo", "Alt+R"], ["erase", "Apagar desenho (clique)", "Alt+E"],
];

export default function App() {
  const [ui, setUi] = useState<Ui>(load);
  const set = useCallback((p: Partial<Ui>) => setUi((u) => ({ ...u, ...p })), []);
  useEffect(() => { try { localStorage.setItem(KEY, JSON.stringify(ui)); } catch { /* ignore */ } }, [ui]);

  const [symbols, setSymbols] = useState<SymbolInfo[]>(FALLBACK_SYMBOLS);
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [fees, setFees] = useState<Fee[]>([]);
  const [active, setActive] = useState(0);
  const [bts, setBts] = useState<Record<number, Backtest | null>>({});
  const [sig, setSig] = useState<Signal | null>(null);
  const [err, setErr] = useState("");
  const [apiOk, setApiOk] = useState<boolean | null>(null);
  const [tool, setTool] = useState<Tool>("cursor");
  const [pal, setPal] = useState(false);
  const [ticks, setTicks] = useState<Map<string, Tick>>(new Map());
  const [ws, setWs] = useState<LiveStatus>(live.status);
  const [lag, setLag] = useState<number | null>(null);
  const [now, setNow] = useState(() => new Date());
  const [sigRev, setSigRev] = useState(0);
  const [focus, setFocus] = useState<{ n: number; t0: number; t1: number; seq: number } | null>(null);
  const bus = useRef(new SyncBus()).current;
  bus.cross = ui.syncCross;
  bus.range = ui.syncRange;

  useEffect(() => {
    getSymbols().then((s) => { setSymbols(s); setUi((u) => (s.some((x) => x.symbol === u.symbol) ? u : { ...u, symbol: s[0].symbol })); });
    getStrategies().then((s) => { setApiOk(true); setStrategies(s); setUi((u) => (s.some((x) => x.name === u.strategy) ? u : { ...u, strategy: s[0]?.name ?? "" })); })
      .catch((e) => { setApiOk(false); setErr(String(e)); });
    getFees().then((f) => { setFees(f); setUi((u) => (f.some((x) => x.name === u.fee) ? u : { ...u, fee: f[0]?.name ?? "" })); }).catch(() => {});
  }, []);

  // live ticker (throttled to the 1 s miniTicker cadence) + socket status + clock
  useEffect(() => {
    const a = live.ticker((t) => setTicks(new Map(t)));
    const b = live.onStatus(setWs);
    const d = live.onLag(setLag);
    const c = setInterval(() => setNow(new Date()), 1000);
    return () => { a(); b(); d(); clearInterval(c); };
  }, []);

  const sym = useMemo(() => symbols.find((s) => s.symbol === ui.symbol) ?? symbols[0], [symbols, ui.symbol]);
  // a persisted liquidation mode is meaningless without liquidation data: fall back to candles
  useEffect(() => {
    if (sym && !sym.has_liq && ui.modes.some((m) => m === "liq")) setUi((u) => ({ ...u, modes: u.modes.map(() => "candles" as PaneMode) }));
  }, [sym, ui.modes]);
  const coming = useMemo(() => COMING.filter((s) => !symbols.some((x) => x.symbol === s)), [symbols]);
  const ai = Math.min(active, ui.layout - 1);
  const activeTf = ui.tfs[ai];

  useEffect(() => {
    if (!ui.strategy) return;
    const ac = new AbortController();
    if (sigRev === 0) setSig(null);
    getSignal(ui.symbol, ui.strategy, activeTf, ac.signal, ui.source).then((s) => { setSig(s); setApiOk(true); }).catch(() => {});
    return () => ac.abort();
  }, [ui.symbol, ui.strategy, activeTf, sigRev, ui.source]);

  const closeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const curSig = useRef({ symbol: ui.symbol, tf: activeTf });
  curSig.current = { symbol: ui.symbol, tf: activeTf };
  const onBarClose = useCallback((s: string, tf: string) => {
    if (s !== curSig.current.symbol || tf !== curSig.current.tf) return;
    if (closeTimer.current) clearTimeout(closeTimer.current);
    // give the backend a moment to ingest the closed candle, then recompute the signal
    closeTimer.current = setTimeout(() => setSigRev((r) => r + 1), 2500);
  }, []);

  const setTf = useCallback((i: number, tf: string) => setUi((u) => ({ ...u, tfs: u.tfs.map((t, k) => (k === i ? tf : t)) })), []);
  const setMode = useCallback((i: number, m: PaneMode) => setUi((u) => ({ ...u, modes: u.modes.map((t, k) => (k === i ? m : t)) })), []);
  const onBt = useCallback((i: number, b: Backtest | null) => setBts((x) => ({ ...x, [i]: b })), []);
  const onErr = useCallback((e: string) => setErr(e), []);
  const onActivate = useCallback((i: number) => setActive(i), []);

  // keyboard: Ctrl+K palette, Alt+letter tools, Esc cursor
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); setPal((v) => !v); return; }
      if ((e.target as HTMLElement).closest("input,select,textarea")) return;
      if (e.key === "Escape") setTool("cursor");
      if (e.altKey && e.key.toLowerCase() === "l") { e.preventDefault(); setUi((u) => ({ ...u, log: !u.log })); return; }
      if (e.altKey) {
        const m: Record<string, Tool> = { t: "trend", h: "hline", r: "rect", e: "erase" };
        const t = m[e.key.toLowerCase()];
        if (t) { e.preventDefault(); setTool(t); }
      }
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, []);

  const cmds = useMemo<Cmd[]>(() => [
    ...[...symbols.map((s) => s.symbol), ...coming].map((s) => ({
      id: "s" + s, group: "Símbolo", label: `${base(s)}/USDT perp`, hint: coming.includes(s) ? "em breve" : undefined,
      disabled: coming.includes(s), run: () => set({ symbol: s }),
    })),
    ...ALL_TFS.map((t) => ({ id: "t" + t, group: "Timeframe", label: `${TF_LABEL[t]} no gráfico ativo`, disabled: !!sym && !sym.tfs.includes(t), run: () => setTf(ai, t) })),
    ...strategies.map((s) => ({ id: "st" + s.name, group: "Estratégia", label: s.name, hint: TF_LABEL[s.tf] ?? s.tf, run: () => set({ strategy: s.name }) })),
    ...([1, 2, 4] as Layout[]).map((n) => ({ id: "l" + n, group: "Layout", label: `${n} gráfico${n > 1 ? "s" : ""}`, run: () => set({ layout: n }) })),
    ...TOOLS.map(([t, l, h]) => ({ id: "x" + t, group: "Ferramenta", label: l, hint: h, run: () => setTool(t) })),
    { id: "xclr", group: "Ferramenta", label: "Limpar todos os desenhos deste símbolo", run: () => drawings.clear(ui.symbol) },
    { id: "veq", group: "Exibir", label: (ui.eq ? "Ocultar" : "Mostrar") + " painel de equity", run: () => set({ eq: !ui.eq }) },
    { id: "vliq", group: "Exibir", label: "Heatmap de liquidações no gráfico ativo", disabled: !sym?.has_liq, run: () => setMode(ai, "liq") },
    { id: "vcan", group: "Exibir", label: "Velas no gráfico ativo", run: () => setMode(ai, "candles") },
    { id: "vsc", group: "Exibir", label: (ui.syncCross ? "Desligar" : "Ligar") + " crosshair sincronizado", run: () => set({ syncCross: !ui.syncCross }) },
    { id: "vsr", group: "Exibir", label: (ui.syncRange ? "Desligar" : "Ligar") + " scroll/zoom sincronizado", run: () => set({ syncRange: !ui.syncRange }) },
    { id: "vin", group: "Exibir", label: (ui.inspector ? "Ocultar" : "Mostrar") + " painel lateral", run: () => set({ inspector: !ui.inspector }) },
    { id: "vtr", group: "Exibir", label: "Lista de negociações (testador)", run: () => set({ testerOpen: true, testerTab: "trades" }) },
    { id: "vov", group: "Exibir", label: "Testador de estratégia: visão geral", run: () => set({ testerOpen: true, testerTab: "overview" }) },
    { id: "van", group: "Exibir", label: "Testador: análise de desempenho", run: () => set({ testerOpen: true, testerTab: "analysis" }) },
    { id: "vlog", group: "Exibir", label: ui.log ? "Escala linear" : "Escala logarítmica", run: () => set({ log: !ui.log }) },
    { id: "vha", group: "Exibir", label: ui.candles === "ha" ? "Velas normais" : "Velas Heikin Ashi", run: () => set({ candles: ui.candles === "ha" ? "real" : "ha" }) },
    { id: "vhs", group: "Estratégia", label: ui.source === "ha" ? "Sinais em velas normais" : "Sinais em Heikin Ashi", run: () => set({ source: ui.source === "ha" ? "real" : "ha" }) },
  ], [symbols, coming, strategies, sym, ai, ui.symbol, ui.eq, ui.candles, ui.source, ui.syncCross, ui.syncRange, ui.inspector, set, setTf, setMode]);

  const spec = useMemo(() => strategies.find((x) => x.name === ui.strategy) ?? null, [strategies, ui.strategy]);
  const feeObj = useMemo(() => fees.find((f) => f.name === ui.fee) ?? null, [fees, ui.fee]);
  const onFocusTrade = useCallback((n: number, t0: number, t1: number) => setFocus((f) => ({ n, t0, t1, seq: (f?.seq ?? 0) + 1 })), []);
  const setTesterTab = useCallback((t: TTab) => set({ testerTab: t }), [set]);
  const setTesterOpen = useCallback((c: boolean) => set({ testerOpen: !c }), [set]);
  const setCapital = useCallback((c: number) => set({ capital: c }), [set]);
  const setSource = useCallback((v: "real" | "ha") => set({ source: v }), [set]);
  const setTesterMax = useCallback((v: boolean) => set({ testerMax: v, testerOpen: true }), [set]);
  // drag handle to resize the tester
  const startDrag = (e: React.PointerEvent) => {
    e.preventDefault();
    const y0 = e.clientY, h0 = ui.testerH;
    const mv = (ev: PointerEvent) => set({ testerH: Math.max(140, Math.min(window.innerHeight - 220, h0 + (y0 - ev.clientY))), testerMax: false, testerOpen: true });
    const up = () => { window.removeEventListener("pointermove", mv); window.removeEventListener("pointerup", up); };
    window.addEventListener("pointermove", mv); window.addEventListener("pointerup", up);
  };
  const testerH = !ui.testerOpen ? 34 : ui.testerMax ? "calc(100vh - var(--top) - var(--stat) - 160px)" : `min(${ui.testerH}px, calc(100vh - var(--top) - var(--stat) - 200px))`;
  const tick = ticks.get(ui.symbol);
  const bt = bts[ai] ?? null;
  const holdout = bt?.holdout_start;
  const panes = Array.from({ length: ui.layout }, (_, i) => i);
  const wsLabel: Record<LiveStatus, string> = { live: "ao vivo", connecting: "conectando", stale: "atrasado", offline: "offline" };
  const st = sig?.state;

  return (
    <div className={"app" + (ui.inspector ? "" : " noinsp")} style={{ ["--tester" as string]: typeof testerH === "number" ? testerH + "px" : testerH }}>
      <header className="top">
        <div className="brand"><Logo /><span>Signal<b>Lab</b></span></div>
        <button className="symbtn" onClick={() => setPal(true)} title="Trocar símbolo (Ctrl+K)">
          <i className={"dot " + ws} title={`Binance WS: ${ws}`} />
          <span className={"lag" + (lag != null && lag > 1000 ? " slow" : "")} title="Atraso de ponta a ponta (evento Binance → candle desenhado)">{lag != null ? `${lag}ms` : "—"}</span>
          <span className="sb">{base(ui.symbol)}<em>/USDT</em></span>
          <Price value={tick?.price} className="sp" />
          <span className={"sc " + (tick ? (tick.pct >= 0 ? "up" : "dn") : "")}>{tick ? fmtPct(tick.pct, true, 2) : "—"}</span>
          {I.chev}
        </button>
        <div className="seg tfseg" role="group" aria-label="Timeframe do gráfico ativo">
          {ALL_TFS.map((t) => (
            <button key={t} className={t === activeTf ? "on" : ""} disabled={!!sym && !sym.tfs.includes(t)} onClick={() => setTf(ai, t)}>{TF_LABEL[t]}</button>
          ))}
        </div>
        <div className="seg ha" role="group" aria-label="Tipo de vela" title="Velas: normais ou Heikin Ashi (apenas visual)">
          <button className={ui.candles === "real" ? "on" : ""} onClick={() => set({ candles: "real" })}>Normais</button>
          <button className={ui.candles === "ha" ? "on" : ""} onClick={() => set({ candles: "ha" })}>Heikin Ashi</button>
        </div>
        <div className="seg" role="group" aria-label="Escala de preço" title="Escala de preço logarítmica (Alt+L)">
          <button className={ui.log ? "on" : ""} aria-pressed={ui.log} onClick={() => set({ log: !ui.log })}>Log</button>
        </div>
        <label className="stsel" title="Estratégia">
          <span>Estratégia</span>
          <select value={ui.strategy} onChange={(e) => set({ strategy: e.target.value })}>
            {strategies.map((s) => <option key={s.name} value={s.name}>{s.name}</option>)}
          </select>
        </label>
        <div className={"sigpill " + (st ?? "none")} title={sig ? `Sinal em ${TF_LABEL[activeTf]} · candle fechado` : "Calculando sinal"}>
          <i />{st === "LONG" ? "▲ LONG" : st === "SHORT" ? "▼ SHORT" : st === "FLAT" ? "— FLAT" : "…"}
        </div>
        <div className="spacer" />
        <div className="seg icon" role="group" aria-label="Layout">
          {([1, 2, 4] as Layout[]).map((n) => (
            <button key={n} className={ui.layout === n ? "on" : ""} onClick={() => set({ layout: n })} title={`${n} gráfico${n > 1 ? "s" : ""}`}>{n === 1 ? I.l1 : n === 2 ? I.l2 : I.l4}</button>
          ))}
        </div>
        <div className="seg icon" role="group" aria-label="Sincronização">
          <button className={ui.syncCross ? "on" : ""} onClick={() => set({ syncCross: !ui.syncCross })} title="Crosshair sincronizado">{I.cross}</button>
          <button className={ui.syncRange ? "on" : ""} onClick={() => set({ syncRange: !ui.syncRange })} title="Scroll/zoom sincronizado">{I.link}</button>
        </div>
        <button className={"tgl" + (ui.eq ? " on" : "")} onClick={() => set({ eq: !ui.eq })} title="Painel de equity">{I.equity}<span>Equity</span></button>
        <button className="kbtn" onClick={() => setPal(true)} title="Paleta de comandos">{I.search}<span>Comandos</span><kbd>Ctrl K</kbd></button>
        <button className={"tgl icon" + (ui.inspector ? " on" : "")} onClick={() => set({ inspector: !ui.inspector })} title="Painel lateral">{I.panel}</button>
      </header>

      <nav className="rail" aria-label="Ferramentas de desenho">
        {TOOLS.map(([t, l, h]) => (
          <button key={t} className={tool === t ? "on" : ""} onClick={() => setTool(t)} title={`${l} (${h})`}>{I[t]}</button>
        ))}
        <div className="rsep" />
        <button onClick={() => drawings.clear(ui.symbol)} title="Limpar desenhos deste símbolo">{I.trash}</button>
      </nav>

      <main className={`grid g${ui.layout}`}>
        {panes.map((i) => (
          <Pane key={i} idx={i} symbol={ui.symbol} tf={ui.tfs[i]} tfs={sym?.tfs ?? []} mode={ui.modes[i]} strategy={ui.strategy} fee={ui.fee}
            active={i === ai && ui.layout > 1} multi={ui.layout > 1} ha={ui.candles === "ha"} log={ui.log} source={ui.source} spec={spec} focus={i === ai ? focus : null} showEq={ui.eq} details={ui.details} showBook={ui.book} showVp={ui.vp}
            hasLiq={!!sym?.has_liq} tool={tool} bus={bus} onActivate={onActivate} onBt={onBt} onErr={onErr}
            onTf={setTf} onMode={setMode} onTool={setTool} onBarClose={onBarClose} />
        ))}
      </main>

      <div className="tester-wrap">
        {ui.testerOpen && <div className="tdrag" onPointerDown={startDrag} title="Arraste para redimensionar" />}
        <Tester bt={bt} spec={spec} symbol={ui.symbol} tf={activeTf} fee={feeObj} source={ui.source} setSource={setSource}
          tab={ui.testerTab} setTab={setTesterTab} collapsed={!ui.testerOpen} setCollapsed={setTesterOpen}
          capital={ui.capital} setCapital={setCapital} onFocusTrade={onFocusTrade} max={ui.testerMax} setMax={setTesterMax} />
      </div>

      {ui.inspector && (
        <Inspector symbols={symbols} coming={coming} symbol={ui.symbol} onSymbol={(s) => set({ symbol: s })} ticks={ticks}
          sig={sig} sigTf={activeTf} strategy={ui.strategy} bt={bt} btTf={activeTf} tab={ui.tab} setTab={(tab) => set({ tab })}
          fees={fees} fee={ui.fee} setFee={(fee) => set({ fee })} book={ui.book} vp={ui.vp} exits={ui.details}
          setFlag={(k, v) => set({ [k]: v } as Partial<Ui>)} />
      )}

      <footer className="status">
        <span className={"st " + (apiOk === false ? "bad" : apiOk ? "ok" : "")}><i />API {apiOk === false ? "indisponível" : apiOk ? "ok" : "…"}</span>
        <span className={"st " + (ws === "live" ? "ok" : ws === "connecting" ? "" : "bad")}><i />Binance WS {wsLabel[ws]}{lag != null ? ` · ${lag} ms` : ""}</span>
        {holdout != null && <span className="st"><b className="is">In-sample</b> até {fmtDate(holdout)} <b className="ho">Holdout</b> depois</span>}
        <span className="st mut">{ui.strategy} · {ui.fee}</span>
        {err && <button className="st bad errbtn" title={err} onClick={() => setErr("")}>Erro: {err.slice(0, 60)} ✕</button>}
        <span className="spacer" />
        <span className="st mut">{tool !== "cursor" ? `Ferramenta: ${TOOLS.find((x) => x[0] === tool)?.[1]} · Esc cancela` : "Ctrl+K comandos · Alt+T/H/R desenhar"}</span>
        <span className="st clock">{now.toISOString().slice(11, 19)} UTC</span>
      </footer>

      {pal && <Palette cmds={cmds} onClose={() => setPal(false)} />}
    </div>
  );
}
