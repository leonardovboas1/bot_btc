import { memo, useEffect, useRef } from "react";
import { tzLabel } from "../util";

export const RANGES = ["1D", "5D", "1M", "3M", "6M", "YTD", "1A", "5A", "Todos"];

/** Range presets (apply to the active pane) + live clock written straight to the DOM. */
function RangeBar({ onRange }: { onRange(k: string): void }) {
  const clk = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    const tz = tzLabel();
    const tick = () => { if (clk.current) clk.current.textContent = new Date().toLocaleTimeString("pt-BR", { hour12: false }) + " " + tz; };
    tick();
    const t = setInterval(tick, 1000);
    return () => clearInterval(t);
  }, []);
  return (
    <div className="rb">
      {RANGES.map((k) => <button key={k} className="rbb" onClick={() => onRange(k)} title={k === "Todos" ? "Todo o histórico carregado" : `Último período: ${k}`}>{k}</button>)}
      <span className="spacer" />
      <span className="clock" ref={clk} />
    </div>
  );
}
export default memo(RangeBar);
