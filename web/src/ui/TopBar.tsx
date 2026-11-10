import { useEffect, useRef, useState, type ReactNode } from "react";
import type { Strategy } from "../types";
import type { Fee } from "../api";
import { ALL_TFS, TF_TV, TF_LABEL } from "../util";
import Btn from "./Btn";
import { I, Logo } from "./icons";

type Layout = 1 | 2 | 4;
export interface Flags { eq: boolean; book: boolean; vp: boolean; details: boolean; haSig: boolean; syncCross: boolean; syncRange: boolean }

function Dd({ trigger, tip, kbd, children, align = "left" }: { trigger: ReactNode; tip: string; kbd?: string; children: (close: () => void) => ReactNode; align?: "left" | "right" }) {
  const [open, setOpen] = useState(false);
  const r = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const f = (e: MouseEvent) => { if (!r.current?.contains(e.target as Node)) setOpen(false); };
    const k = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", f); window.addEventListener("keydown", k);
    return () => { document.removeEventListener("mousedown", f); window.removeEventListener("keydown", k); };
  }, [open]);
  return (
    <div className="dd" ref={r}>
      <Btn tip={tip} kbd={kbd} on={open} className="wide" onClick={() => setOpen((v) => !v)}>{trigger}</Btn>
      {open && <div className={"ddm " + align}>{children(() => setOpen(false))}</div>}
    </div>
  );
}
const Item = ({ on, onClick, children, icon }: { on?: boolean; onClick(): void; children: ReactNode; icon?: ReactNode }) => (
  <button className={"mi" + (on ? " on" : "")} onClick={onClick}><span className="mchk">{on ? I.check : icon}</span>{children}</button>
);

export interface TopProps {
  symbol: string; onSearch(): void; tf: string; tfs: string[]; onTf(t: string): void;
  candles: "real" | "ha"; onCandles(c: "real" | "ha"): void;
  strategies: Strategy[]; strategy: string; onStrategy(s: string): void; fees: Fee[]; fee: string; onFee(f: string): void;
  flags: Flags; onFlag(k: keyof Flags, v: boolean): void;
  layout: Layout; onLayout(n: Layout): void;
  inspector: boolean; onInspector(): void; err: string; onErr(): void; apiOk: boolean | null;
}

export default function TopBar(p: TopProps) {
  const F = (k: keyof Flags, label: string) => <Item key={k} on={p.flags[k]} onClick={() => p.onFlag(k, !p.flags[k])}>{label}</Item>;
  return (
    <header className="top">
      <span className="brand" data-tip="SignalLab" data-pos="b"><Logo /></span>
      <i className="sep" />
      <button className="symchip" onClick={p.onSearch} data-tip="Buscar símbolo   Ctrl+K" data-pos="b">{p.symbol}.P</button>
      <i className="sep" />
      {ALL_TFS.map((t) => (
        <button key={t} className={"tfb" + (t === p.tf ? " on" : "")} disabled={!p.tfs.includes(t)} onClick={() => p.onTf(t)}
          data-tip={`Intervalo ${TF_LABEL[t]}`} data-pos="b">{TF_TV[t]}</button>
      ))}
      <i className="sep" />
      <Dd tip="Tipo de vela" trigger={<>{p.candles === "ha" ? I.ha : I.candles}{I.chev}</>}>
        {(close) => (<>
          <Item on={p.candles === "real"} icon={I.candles} onClick={() => { p.onCandles("real"); close(); }}>Velas</Item>
          <Item on={p.candles === "ha"} icon={I.ha} onClick={() => { p.onCandles("ha"); close(); }}>Heikin Ashi</Item>
        </>)}
      </Dd>
      <i className="sep" />
      <Dd tip="Indicadores e estratégias" trigger={<>{I.ind}<span>Indicadores</span>{I.chev}</>}>
        {() => (<>
          <div className="mh">Estratégia</div>
          {p.strategies.map((s) => <Item key={s.name} on={s.name === p.strategy} onClick={() => p.onStrategy(s.name)}>{s.name}<em>{TF_LABEL[s.tf] ?? s.tf}</em></Item>)}
          <div className="mh">Exibir</div>
          {F("eq", "Painel de equity")}{F("details", "Saídas e rótulos L/S")}{F("book", "Livro de ofertas (heatmap)")}{F("vp", "Volume profile")}{F("haSig", "Sinais em Heikin Ashi")}
          <div className="mh">Custos</div>
          {p.fees.map((f) => <Item key={f.name} on={f.name === p.fee} onClick={() => p.onFee(f.name)}>{f.name}</Item>)}
        </>)}
      </Dd>
      <i className="sep" />
      <Dd tip="Layout dos gráficos" trigger={<>{I.grid}{I.chev}</>}>
        {(close) => (<>
          {([1, 2, 4] as Layout[]).map((n) => <Item key={n} on={p.layout === n} icon={n === 1 ? I.l1 : n === 2 ? I.l2 : I.l4} onClick={() => { p.onLayout(n); close(); }}>{n} gráfico{n > 1 ? "s" : ""}</Item>)}
          <div className="mh">Sincronizar</div>
          {F("syncCross", "Crosshair")}{F("syncRange", "Scroll e zoom")}
        </>)}
      </Dd>
      <span className="spacer" />
      {(p.err || p.apiOk === false) && <button className="errbtn" title={p.err} onClick={p.onErr}>{p.apiOk === false ? "API indisponível" : "Erro: " + p.err.slice(0, 50)} {I.x}</button>}
      <Btn tip="Buscar e comandos" kbd="Ctrl+K" pos="l" onClick={p.onSearch}>{I.search}</Btn>
      <Btn tip="Lista de selecionados" pos="l" on={p.inspector} onClick={p.onInspector}>{I.panel}</Btn>
    </header>
  );
}
