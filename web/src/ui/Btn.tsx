import type { ReactNode } from "react";

/** Flat icon-first button with a CSS tooltip (label + shortcut). */
export default function Btn({ tip, kbd, on, disabled, pos = "b", className = "", onClick, children }: {
  tip: string; kbd?: string; on?: boolean; disabled?: boolean; pos?: "b" | "r" | "l" | "t"; className?: string; onClick?(e: React.MouseEvent): void; children: ReactNode;
}) {
  return (
    <button className={"ib " + className + (on ? " on" : "")} data-tip={tip + (kbd ? "   " + kbd : "")} data-pos={pos} aria-label={tip} aria-pressed={on} disabled={disabled} onClick={onClick}>
      {children}
    </button>
  );
}
