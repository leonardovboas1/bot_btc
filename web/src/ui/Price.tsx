import { useEffect, useRef } from "react";
import { fmtPrice } from "../util";

/** Price that flashes up/down on change (DOM-only animation restart, no re-render churn). */
export default function Price({ value, className = "" }: { value: number | null | undefined; className?: string }) {
  const el = useRef<HTMLSpanElement>(null);
  const prev = useRef<number | null>(null);
  useEffect(() => {
    const e = el.current, p = prev.current;
    prev.current = value ?? null;
    if (!e || value == null || p == null || value === p) return;
    e.classList.remove("flash-up", "flash-dn");
    void e.offsetWidth;
    e.classList.add(value > p ? "flash-up" : "flash-dn");
  }, [value]);
  return <span ref={el} className={"price " + className}>{fmtPrice(value)}</span>;
}
