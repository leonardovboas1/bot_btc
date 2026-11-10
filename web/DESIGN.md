# SignalLab — design brief ("TradingView dark")

Direction: a flat, minimalist clone of TradingView's dark chart page. Chart first, chrome as thin as
possible: black surfaces, 1px `#2a2e39` separators, icon-first buttons, no gradients, glows, shadows (except
menus), badges, uppercase or letter-spaced labels. Not a copy of TradingView assets: icons come from
lucide-react (MIT), coin logos from `cryptocurrency-icons` (CC0, bundled; coloured-initial circle as fallback).

## Tokens
| Role | Hex |
|---|---|
| background / chart | `#000000` (no grid lines) |
| panels, menus | `#0f0f0f` / `#1a1a1a` |
| borders, separators, hover bg | `#2a2e39` |
| text / secondary / muted | `#d1d4dc` / `#b2b5be` / `#787b86` |
| up / down | `#26a69a` / `#ef5350` |
| accent (active, selection, equity) | `#2962ff` |
| holdout | `#8b7cf6` dashed line (unchanged) |

## Type
Stack `-apple-system, BlinkMacSystemFont, "Trebuchet MS", Roboto, Ubuntu, sans-serif` everywhere.
Weight 400; 600 only for the symbol chip and panel headers. UI 13-14px, axes/legend values/range bar 12px,
tabular-nums for every price. Last price digit rendered smaller in watchlist and details (like TV).

## Layout (CSS grid in `style.css`)
- Top bar 38px: logo, symbol chip `BTCUSDT.P` (opens Ctrl+K palette), TFs `5m 15m 1h 4h D S`, candle-type
  dropdown (Velas / Heikin Ashi), `Indicadores` dropdown (strategy picker, equity, exits, book, volume profile,
  HA signals, costs), layout dropdown (1/2/4 + crosshair / range sync), right: search and panel toggle.
- Left rail 48px: drawing tools (cursor, trend, horizontal, rectangle, eraser, clear).
- Chart panes: no header; the legend overlays top-left: line 1 `Bitcoin / TetherUS PERPETUAL CONTRACT · 4h · Binance`
  + WS dot + lag + Abr/Máx/Mín/Fch + change coloured by candle direction; line 2+ strategy name, position text
  ("long/short · em posição" / "fora" replaces the old signal card) and indicator values; chevron collapses.
  Multi-pane: the timeframe in line 1 is a dropdown per pane.
- Range bar 32px below the charts: `1D 5D 1M 3M 6M YTD 1A 5A Todos` (apply to the active pane) and a live clock in
  the browser timezone (`16:29:30 UTC-3`). Time axis uses the browser timezone intraday, UTC for 1D/1S bars.
- Strategy tester: bottom panel (unchanged features: overview, analysis, summary, trades, properties, Paper trading).
- Right panel 340px: "Lista de selecionados" (Símbolo / Preço / Var / Var%, selected row outlined) over a symbol
  details box (name, big last price, change). Far-right 46px icon rail toggles watchlist, tester, liquidations.
- No footer. Errors / API-down appear as a red chip in the top bar.

## Buttons
Flat, no borders or pills; 34px square icon hit areas; `#d1d4dc` icons, white on hover with `#2a2e39` rounded
background; active = `#2962ff`; CSS tooltips (`data-tip`: label + shortcut); 1px vertical separators; small chevrons.

## Live data performance
Live ticks never go through React state: pane candles update via refs + one `requestAnimationFrame` flush,
the legend, WS dot / lag, watchlist, details box and clock are written straight to the DOM. App does not
re-render per tick (it used to re-render every second on the ticker map, clock and lag state).
