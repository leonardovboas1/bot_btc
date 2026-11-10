import { useEffect, useMemo, useRef, useState } from "react";
import { I } from "./icons";

export interface Cmd { id: string; group: string; label: string; hint?: string; disabled?: boolean; run(): void }

const norm = (s: string) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
/** Subsequence fuzzy score; higher is better, -1 = no match. */
function score(q: string, s: string) {
  if (!q) return 0;
  const a = norm(q), b = norm(s);
  const i = b.indexOf(a);
  if (i >= 0) return 100 - i;
  let j = 0, sc = 0;
  for (let k = 0; k < b.length && j < a.length; k++) if (b[k] === a[j]) { j++; sc += 1; }
  return j === a.length ? sc : -1;
}

export default function Palette({ cmds, onClose }: { cmds: Cmd[]; onClose(): void }) {
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(0);
  const inp = useRef<HTMLInputElement>(null);
  const list = useMemo(() => {
    const r = cmds.map((c) => ({ c, s: Math.max(score(q, c.label), score(q, c.group + " " + c.label) - 5) })).filter((x) => x.s >= 0 && !x.c.disabled);
    if (q) r.sort((a, b) => b.s - a.s);
    return r.slice(0, 9).map((x) => x.c);
  }, [cmds, q]);
  useEffect(() => { inp.current?.focus(); }, []);
  useEffect(() => setSel(0), [q]);
  const run = (c?: Cmd) => { if (!c) return; c.run(); onClose(); };
  let lastGroup = "";
  return (
    <div className="pal-back" onMouseDown={onClose}>
      <div className="pal" onMouseDown={(e) => e.stopPropagation()} role="dialog" aria-label="Paleta de comandos">
        <div className="pal-in">
          {I.search}
          <input ref={inp} value={q} placeholder="Buscar símbolo, timeframe, estratégia, layout, ferramenta…" onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "ArrowDown") { e.preventDefault(); setSel((s) => Math.min(list.length - 1, s + 1)); }
              else if (e.key === "ArrowUp") { e.preventDefault(); setSel((s) => Math.max(0, s - 1)); }
              else if (e.key === "Enter") run(list[sel]);
              else if (e.key === "Escape") onClose();
            }} />
          <kbd>Esc</kbd>
        </div>
        <div className="pal-list">
          {list.length === 0 && <div className="pal-empty">Nenhum comando encontrado</div>}
          {list.map((c, i) => {
            const head = !q && c.group !== lastGroup ? c.group : null;
            lastGroup = c.group;
            return (
              <div key={c.id}>
                {head && <div className="pal-group">{head}</div>}
                <button className={"pal-item" + (i === sel ? " on" : "")} onMouseEnter={() => setSel(i)} onClick={() => run(c)}>
                  <span className="pal-g">{c.group}</span><span className="pal-l">{c.label}</span>{c.hint && <kbd>{c.hint}</kbd>}
                </button>
              </div>
            );
          })}
        </div>
        <div className="pal-foot"><span><kbd>↑</kbd><kbd>↓</kbd> navegar</span><span><kbd>Enter</kbd> executar</span><span><kbd>Ctrl</kbd><kbd>K</kbd> abrir</span></div>
      </div>
    </div>
  );
}
