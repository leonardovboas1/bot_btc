/** Per-symbol drawing store, persisted in localStorage and shared by every pane of that symbol. */
export type Tool = "cursor" | "trend" | "hline" | "rect" | "erase";
export interface DPoint { t: number; p: number }
export interface Drawing { id: string; kind: "trend" | "hline" | "rect"; a: DPoint; b?: DPoint }

const KEY = (s: string) => "btclab.draw." + s;
const cache = new Map<string, Drawing[]>();
const subs = new Set<(symbol: string) => void>();

export const drawings = {
  get(symbol: string): Drawing[] {
    let d = cache.get(symbol);
    if (!d) {
      try { d = JSON.parse(localStorage.getItem(KEY(symbol)) || "[]") as Drawing[]; } catch { d = []; }
      cache.set(symbol, d);
    }
    return d;
  },
  set(symbol: string, list: Drawing[]) {
    cache.set(symbol, list);
    try { localStorage.setItem(KEY(symbol), JSON.stringify(list)); } catch { /* quota */ }
    subs.forEach((f) => f(symbol));
  },
  add(symbol: string, d: Drawing) { this.set(symbol, [...this.get(symbol), d]); },
  remove(symbol: string, id: string) { this.set(symbol, this.get(symbol).filter((x) => x.id !== id)); },
  clear(symbol: string) { this.set(symbol, []); },
  subscribe(f: (symbol: string) => void) { subs.add(f); return () => { subs.delete(f); }; },
};
export const newId = () => Math.random().toString(36).slice(2, 10);
