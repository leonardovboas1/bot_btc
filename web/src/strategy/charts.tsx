import { useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";

/** Width/height of an element, live. */
export function useSize<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [s, setS] = useState({ w: 0, h: 0 });
  useLayoutEffect(() => {
    const e = ref.current;
    if (!e) return;
    const ro = new ResizeObserver(() => setS({ w: e.clientWidth, h: e.clientHeight }));
    ro.observe(e);
    return () => ro.disconnect();
  }, []);
  return [ref, s] as const;
}

const niceStep = (span: number, n: number) => {
  const raw = span / Math.max(1, n), p = 10 ** Math.floor(Math.log10(raw || 1)), f = raw / p;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
};
export const kfmt = (v: number) => {
  const a = Math.abs(v), s = v < 0 ? "−" : "";
  if (a >= 1e6) return s + (a / 1e6).toLocaleString("pt-BR", { maximumFractionDigits: 2 }) + "M";
  if (a >= 1e3) return s + (a / 1e3).toLocaleString("pt-BR", { maximumFractionDigits: 2 }) + "K";
  return s + a.toLocaleString("pt-BR", { maximumFractionDigits: 2 });
};

export interface BarSeries { key: string; label: string; color: string }
export interface BarRow { x: string; tip?: string; v: Record<string, number> }

/**
 * Grouped / stacked-by-sign bar chart in SVG with right-hand y axis, zero line, hover tooltip and paging
 * (shows the most recent rows that fit; ‹ › pages back in time).
 */
export function BarChart({ rows, series, height = 180, minBar = 6, yFmt = kfmt, refLines = [], legend = true, colorOf }: {
  rows: BarRow[]; series: BarSeries[]; height?: number; minBar?: number; yFmt?: (v: number) => string;
  refLines?: { x: number; color: string; label: string }[]; legend?: boolean; colorOf?: (row: BarRow, s: BarSeries) => string;
}) {
  const [ref, { w }] = useSize<HTMLDivElement>();
  const [page, setPage] = useState(0);
  const [hov, setHov] = useState<number | null>(null);
  const AX = 64, PADB = 22, H = height;
  const plotW = Math.max(0, w - AX - 8);
  const per = Math.max(1, Math.floor(plotW / (minBar * series.length + 4)));
  const pages = Math.max(1, Math.ceil(rows.length / per));
  const pg = Math.min(page, pages - 1);
  const view = rows.slice(Math.max(0, rows.length - per * (pg + 1)), rows.length - per * pg);
  const { lo, hi, step } = useMemo(() => {
    let lo = 0, hi = 0;
    for (const r of view) for (const s of series) { const v = r.v[s.key] ?? 0; lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (lo === hi) hi = lo + 1;
    const step = niceStep(hi - lo, 4);
    return { lo: Math.floor(lo / step) * step, hi: Math.ceil(hi / step) * step, step };
  }, [view, series]);
  const y = (v: number) => 8 + ((hi - v) / (hi - lo)) * (H - PADB - 8);
  const slot = view.length ? plotW / Math.max(view.length, per > view.length ? view.length : per) : 0;
  const bw = Math.max(2, Math.min(28, (slot - 3) / series.length - 1));
  const ticks: number[] = [];
  for (let v = lo; v <= hi + 1e-9; v += step) ticks.push(v);
  const labEvery = Math.max(1, Math.ceil(70 / Math.max(1, slot)));
  return (
    <div className="bchart" ref={ref}>
      {w > 0 && (
        <svg width={w} height={H} onMouseLeave={() => setHov(null)}>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={0} x2={plotW} y1={y(t)} y2={y(t)} className={t === 0 ? "zero" : "gl"} />
              <text x={plotW + 8} y={y(t) + 3.5} className="yl">{yFmt(t)}</text>
            </g>
          ))}
          {view.map((r, i) => {
            const cx = i * slot + slot / 2;
            return (
              <g key={r.x + i} onMouseEnter={() => setHov(i)}>
                <rect x={i * slot} y={0} width={slot} height={H - PADB} fill="transparent" />
                {hov === i && <rect x={i * slot} y={4} width={slot} height={H - PADB - 4} className="hovband" />}
                {series.map((s, k) => {
                  const v = r.v[s.key] ?? 0;
                  const x = cx - (series.length * (bw + 1)) / 2 + k * (bw + 1);
                  const y0 = y(Math.max(0, v)), h = Math.max(v === 0 ? 0 : 1, Math.abs(y(v) - y(0)));
                  return <rect key={s.key} x={x} y={y0} width={bw} height={h} rx={Math.min(2, bw / 3)} fill={colorOf ? colorOf(r, s) : s.color} />;
                })}
                {i % labEvery === 0 && <text x={cx} y={H - 6} className="xl">{r.x}</text>}
              </g>
            );
          })}
          {refLines.map((l) => {
            const i = view.findIndex((r) => Number(r.x.replace("%", "").replace(",", ".").replace("−", "-")) >= l.x);
            if (i < 0) return null;
            const x = i * slot;
            return <line key={l.label} x1={x} x2={x} y1={6} y2={H - PADB} stroke={l.color} strokeDasharray="3 3" strokeWidth={1.2} />;
          })}
        </svg>
      )}
      {hov != null && view[hov] && (
        <div className="btip" style={{ left: Math.min(w - 170, Math.max(0, hov * slot + slot / 2 - 80)) }}>
          <b>{view[hov].tip ?? view[hov].x}</b>
          {series.map((s) => <div key={s.key}><i style={{ background: s.color }} />{s.label}<span>{yFmt(view[hov].v[s.key] ?? 0)}</span></div>)}
        </div>
      )}
      {pages > 1 && (<>
        <button className="bnav l" disabled={pg >= pages - 1} onClick={() => setPage(pg + 1)} aria-label="Anterior">‹</button>
        <button className="bnav r" disabled={pg === 0} onClick={() => setPage(pg - 1)} aria-label="Próximo">›</button>
      </>)}
      {legend && (
        <div className="blegend">{series.map((s) => <span key={s.key}><i style={{ background: s.color }} />{s.label}</span>)}{refLines.map((l) => <span key={l.label}><i className="dash" style={{ borderColor: l.color }} />{l.label}</span>)}</div>
      )}
    </div>
  );
}

export function Donut({ parts, total, label }: { parts: { label: string; n: number; color: string }[]; total: number; label: string }) {
  const R = 58, r = 44, C = 2 * Math.PI * ((R + r) / 2);
  let off = 0;
  return (
    <div className="donut">
      <svg width={140} height={140} viewBox="0 0 140 140">
        <circle cx={70} cy={70} r={(R + r) / 2} fill="none" stroke="#151a23" strokeWidth={R - r} />
        {parts.map((p) => {
          const len = total ? (p.n / total) * C : 0;
          const el = len > 0 ? (
            <circle key={p.label} cx={70} cy={70} r={(R + r) / 2} fill="none" stroke={p.color} strokeWidth={R - r}
              strokeDasharray={`${Math.max(0, len - 2)} ${C}`} strokeDashoffset={-off} transform="rotate(-90 70 70)" />
          ) : null;
          off += len;
          return el;
        })}
        <text x={70} y={68} textAnchor="middle" className="dn-n">{total}</text>
        <text x={70} y={84} textAnchor="middle" className="dn-l">{label}</text>
      </svg>
      <div className="dlegend">
        {parts.map((p) => (
          <div key={p.label}><i style={{ background: p.color }} /><span>{p.label}</span><b>{p.n} negociações</b><b>{total ? ((p.n / total) * 100).toLocaleString("pt-BR", { maximumFractionDigits: 2, minimumFractionDigits: 2 }) : "0,00"}%</b></div>
        ))}
      </div>
    </div>
  );
}

export function Pills<T extends string>({ items, value, onChange }: { items: [T, string][]; value: T; onChange(v: T): void }) {
  return <div className="pills">{items.map(([k, l]) => <button key={k} className={k === value ? "on" : ""} onClick={() => onChange(k)}>{l}</button>)}</div>;
}
export function Seg<T extends string>({ items, value, onChange }: { items: [T, string][]; value: T; onChange(v: T): void }) {
  return <div className="tseg">{items.map(([k, l]) => <button key={k} className={k === value ? "on" : ""} onClick={() => onChange(k)}>{l}</button>)}</div>;
}
export function Stat({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return <div className="tstat" title={hint}><span>{label}</span><div>{children}</div></div>;
}
