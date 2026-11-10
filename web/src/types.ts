export interface Strategy { name: string; tf: string }
export interface Bar { time: number; open: number; high: number; low: number; close: number; volume: number }
export interface Marker { time: number; position: "aboveBar" | "belowBar"; shape: "arrowUp" | "arrowDown" | "circle"; color: string; text: string }
export interface Pt { time: number; value: number }
export interface Trade { entry_time: number; exit_time: number; side: string; entry_price: number; exit_price: number; pnl_pct: number; exit_reason: string }
export interface Backtest {
  holdout_start: number; trades: Trade[]; markers: Marker[]; holdout_markers: Marker[];
  equity: Pt[]; buy_hold: Pt[]; metrics: Record<string, number | null>;
}
export interface Signal { state: "LONG" | "SHORT" | "FLAT"; time: number; price: number }
