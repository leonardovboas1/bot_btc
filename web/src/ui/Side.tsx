import { memo, useEffect, useMemo, useRef } from "react";
import type { SymbolInfo } from "../types";
import { live, type Tick } from "../live";
import { base, fmtPct, fmtPrice, symName } from "../util";
import Coin from "./Coin";

const ORDER = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "NEARUSDT", "FETUSDT"];
const rank = (s: string) => { const i = ORDER.indexOf(s); return i < 0 ? 99 : i; };

/** "73.639,6" -> big part + small last digit. */
const split = (v: string) => `${v.slice(0, -1)}<s>${v.slice(-1)}</s>`;
const flash = (e: HTMLElement, up: boolean) => { e.classList.remove("fu", "fd"); void e.offsetWidth; e.classList.add(up ? "fu" : "fd"); };
const sgn = (v: number, d: number) => (v > 0 ? "+" : v < 0 ? "−" : "") + fmtPrice(Math.abs(v), d);

interface Props { symbols: SymbolInfo[]; coming: string[]; symbol: string; onSymbol(s: string): void }

/** Watchlist + symbol details. Live prices are written straight to the DOM (no React renders per tick). */
function Side({ symbols, coming, symbol, onSymbol }: Props) {
  const rows = useMemo(() => [...symbols.map((s) => ({ s: s.symbol, soon: false })), ...coming.map((s) => ({ s, soon: true }))]
    .sort((a, b) => Number(a.soon) - Number(b.soon) || rank(a.s) - rank(b.s)), [symbols, coming]);
  const root = useRef<HTMLElement>(null);
  const sel = useRef(symbol);
  sel.current = symbol;
  const last = useRef(new Map<string, number>());

  useEffect(() => {
    let raf = 0, data: Map<string, Tick> | null = null;
    const paint = () => {
      raf = 0;
      const r = root.current;
      if (!r || !data) return;
      const prev = last.current;
      r.querySelectorAll<HTMLElement>("[data-sym]").forEach((row) => {
        const t = data!.get(row.dataset.sym!);
        if (!t) return;
        const d = t.price >= 1000 ? 1 : t.price >= 100 ? 2 : t.price >= 1 ? 3 : 4;
        const cls = t.pct >= 0 ? "up" : "dn", p = row.querySelector<HTMLElement>(".wp")!;
        p.innerHTML = split(fmtPrice(t.price, d));
        const o = prev.get(row.dataset.sym!);
        if (o != null && o !== t.price) flash(p, t.price > o);
        prev.set(row.dataset.sym!, t.price);
        const c = row.querySelector<HTMLElement>(".wv")!, q = row.querySelector<HTMLElement>(".wc")!;
        c.textContent = sgn(t.price - t.open, d); q.textContent = fmtPct(t.pct, true, 2);
        c.className = "wv " + cls; q.className = "wc " + cls;
      });
      const t = data.get(sel.current), box = r.querySelector<HTMLElement>(".det");
      if (t && box) {
        const d = t.price >= 1000 ? 1 : t.price >= 100 ? 2 : t.price >= 1 ? 3 : 4, cls = t.pct >= 0 ? "up" : "dn";
        const big = box.querySelector<HTMLElement>(".dp")!;
        big.innerHTML = split(fmtPrice(t.price, d));
        big.className = "dp " + cls;
        const ch = box.querySelector<HTMLElement>(".dc")!;
        ch.textContent = `${sgn(t.price - t.open, d)}  ${fmtPct(t.pct, true, 2)}`;
        ch.className = "dc " + cls;
      }
    };
    const off = live.ticker((m) => { data = m; if (!raf) raf = requestAnimationFrame(paint); });
    return () => { off(); cancelAnimationFrame(raf); };
  }, []);

  return (
    <aside className="side" ref={root}>
      <div className="sh"><span>Lista de selecionados</span></div>
      <div className="wh"><span>Símbolo</span><span>Preço</span><span>Var</span><span>Var%</span></div>
      <div className="wl">
        {rows.map(({ s, soon }) => (
          <button key={s} data-sym={s} className={"wrow" + (s === symbol ? " on" : "") + (soon ? " soon" : "")} disabled={soon}
            onClick={() => onSymbol(s)} title={soon ? "Em breve: dados ainda não ingeridos" : s}>
            <span className="ws"><Coin symbol={s} size={18} />{s}</span>
            <span className="wp">—</span><span className="wv">—</span><span className="wc">—</span>
          </button>
        ))}
      </div>
      <section className="det">
        <div className="dh"><Coin symbol={symbol} size={26} /><b>{symbol}.P</b></div>
        <div className="dn2">{symName(symbol)} PERPETUAL CONTRACT · BINANCE</div>
        <div className="dn2">Swap · Cripto</div>
        <div className="dl"><span className="dp">—</span><span className="du">USDT</span><span className="dc">—</span></div>
        <div className="dn2 mkt"><i className="dot live" /> Mercado aberto · {base(symbol)}</div>
      </section>
    </aside>
  );
}
export default memo(Side);
