import {
  MousePointer2, TrendingUp, Minus, RectangleHorizontal, Eraser, Trash2, Search, LayoutGrid, Square, Columns2, Grid2x2, ChartCandlestick,
  ChartLine, Activity, PanelRight, ChevronDown, ChevronUp, List, PanelBottom, Flame, Link2, Crosshair, Check, ChartNoAxesColumn, X, ChevronLeft, ChevronRight, ChevronsUpDown,
} from "lucide-react";

const P = { size: 18, strokeWidth: 1.5 };
const mk = (C: typeof Search, size = P.size) => <C size={size} strokeWidth={P.strokeWidth} />;
export const I = {
  cursor: mk(MousePointer2), trend: mk(TrendingUp), hline: mk(Minus), rect: mk(RectangleHorizontal), erase: mk(Eraser), trash: mk(Trash2),
  search: mk(Search), grid: mk(LayoutGrid), l1: mk(Square), l2: mk(Columns2), l4: mk(Grid2x2), candles: mk(ChartCandlestick), ha: mk(ChartNoAxesColumn), line: mk(ChartLine),
  ind: mk(Activity), panel: mk(PanelRight), chev: mk(ChevronDown, 12), up: mk(ChevronUp, 14), down: mk(ChevronDown, 14),
  list: mk(List), tester: mk(PanelBottom), liq: mk(Flame), link: mk(Link2), cross: mk(Crosshair), check: mk(Check, 14), x: mk(X, 14),
  left: mk(ChevronLeft, 14), right: mk(ChevronRight, 14), sel: mk(ChevronsUpDown, 14),
};

/** Brand mark: flat candle pair (SignalLab). */
export const Logo = () => (
  <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
    <rect x="1" y="1" width="20" height="20" rx="5" fill="#2962ff" />
    <path d="M7 5v12M15 4v14" stroke="#fff" strokeWidth="1.5" strokeLinecap="round" />
    <rect x="5" y="8" width="4" height="6" rx="1" fill="#fff" />
    <rect x="13" y="6.5" width="4" height="8" rx="1" fill="#fff" />
  </svg>
);
