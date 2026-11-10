import { base } from "../util";

// Bundled CC0 coin logos (cryptocurrency-icons); tickers without a logo fall back to a coloured initial.
const files = import.meta.glob("../../node_modules/cryptocurrency-icons/svg/color/*.svg", { eager: true, query: "?url", import: "default" }) as Record<string, string>;
const url: Record<string, string> = {};
for (const [k, v] of Object.entries(files)) url[k.split("/").pop()!.replace(".svg", "").toUpperCase()] = v;
const HUE: Record<string, string> = { NEAR: "#00a67e", FET: "#1d2e5d" };
const hue = (b: string) => HUE[b] ?? `hsl(${[...b].reduce((a, c) => a + c.charCodeAt(0) * 37, 0) % 360} 55% 42%)`;

export default function Coin({ symbol, size = 18 }: { symbol: string; size?: number }) {
  const b = base(symbol);
  const u = url[b];
  if (u) return <img className="coin" src={u} width={size} height={size} alt={b} draggable={false} />;
  return <span className="coin" style={{ width: size, height: size, background: hue(b), fontSize: size * 0.55 }}>{b[0]}</span>;
}
