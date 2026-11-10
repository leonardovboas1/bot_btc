export interface Strategy { name: string; tf: string; [k: string]: unknown }
export interface Bar { time: number; open: number; high: number; low: number; close: number; volume: number }
export interface Marker { time: number; position: "aboveBar" | "belowBar"; shape: "arrowUp" | "arrowDown" | "circle"; color: string; text: string }
export interface Pt { time: number; value: number }
export interface Trade { entry_time: number; exit_time: number; side: string; entry_price: number; exit_price: number; pnl_pct: number; exit_reason: string }
export type Metrics = Record<string, number | null | Record<string, number | null>>;
export interface Backtest {
  holdout_start: number; trades: Trade[]; markers: Marker[]; holdout_markers: Marker[];
  equity: Pt[]; buy_hold: Pt[]; metrics: Metrics;
  equity_daily?: { time: number; equity: number; buy_hold: number }[]; tf?: string; strategy?: string;
}
export interface Signal { state: "LONG" | "SHORT" | "FLAT"; time: number; price: number; tf?: string }
export interface SymbolInfo { symbol: string; base: string; tfs: string[]; has_liq: boolean; soon?: boolean }
