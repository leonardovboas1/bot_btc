# Liquidation heatmap (web/src/liq)

Self-contained Coinglass-style liquidation heatmap. Backend: `GET /api/heatmap/liq?symbol&range&venues&end`
(`btclab/liquidity/liq_map.py`). Ranges 1d..6m; resolution is set by the range (5m / 15m / 1h), not the chart.

## Interface

```tsx
import LiqMode, { type LiqHost } from "./liq/LiqControls";
<LiqMode symbol="BTCUSDT" host={{ chart, candle, prim, setCandles }} />
```

`LiqHost`:
- `chart: IChartApi`, `candle: ISeriesApi<"Candlestick">`, `prim: LayersPrimitive` (from `layers.ts`, the pane primitive).
- `setCandles(bars | null)`: called with the range-resolution candles (replace pane candles and freeze lazy
  loading) and with `null` on unmount (restore the normal candles).

Mount `LiqMode` only after the chart exists (render it once `chart` is created) and only while the mode is on;
unmount removes the layer and restores the timescale. `LiqLayer` (LiqHeatmap.ts) implements `HeatmapLayer`
from `layers.ts` and also exposes `lookup(x, price)` for tooltips. Styles are in `liq.css`; UI text is pt-BR.

## Rendering
Grid is `u8 = round(255*sqrt(min(usd/scale,1)))`, decoded in `api.ts`. Viridis LUT, sqrt/log/linear scaling,
threshold slider hides weak levels, active final-column levels extend to the right edge, consumed levels
are absent from later columns (white dots mark large consumptions).

## Mount recipe (pane component, ~10 lines)
```tsx
const [ready, setReady] = useState(false);          // set true in the effect that creates the chart
const liqOn = useRef(false);                          // bars effect: `if (liqOn.current) return;`
{ready && showLiq && (
  <LiqMode symbol={symbol} host={{ chart, candle, prim, setCandles: (b) => {
    if (b) { liqOn.current = true; bars.current = b; applyBars(false); }   // freeze left-extension
    else { liqOn.current = false; reloadBars(); }                          // restore normal candles
  } }} />
)}
```
Parent must be `position: relative` (the controls and tooltip are absolutely positioned).
