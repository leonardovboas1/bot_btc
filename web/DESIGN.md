# BTC Signal Lab — design brief ("Obsidian Terminal")

Direction: a research terminal, not a retail app. Bloomberg-level density, but built on a
modern near-black surface with hairline structure, one warm brand accent, and colour reserved
for meaning (long / short / equity). The chart is the hero; chrome recedes. Every number is
monospaced and tabular so columns never jitter during live updates.

References: TradingView (fixed chart centre, permanent tool edges, left drawing rail),
Hyperliquid (flat panels, tight 1px borders, mono numerals, live price flash),
Coinglass (heatmap pane with compact control strip), Bloomberg (amber accent, keyboard-first).

## Palette (dark only, validated with dataviz `validate_palette.js`, surface #0a0d12: all PASS)
| Role | Hex | Use |
|---|---|---|
| bg-0 / bg-1 | `#000000` | app background and chart surface (pure black, TradingView-like; no grid) |
| bg-2 | `#10141b` | panels, top bar |
| bg-3 | `#171c25` | hover, inputs |
| line | `#1c222d` | hairlines / grid (grid at `#121720`) |
| text-1 / 2 / 3 | `#e6e9ef` / `#9aa3b2` / `#5d6676` | primary / secondary / muted |
| accent | `#ffb020` | brand, focus, active pane, selection (never data) |
| long / up | `#0ea894` | long markers, up candles, positive values (+ sign / ▲) |
| short / down | `#f25c54` | short markers, down candles, negative values (− sign / ▼) |
| equity | `#4c8dff` | strategy equity; buy & hold in text-3 grey |
| holdout | `#8b7cf6` dashed cutoff line | in-sample / holdout boundary; labels "◂ IN-SAMPLE" / "HOLDOUT ▸" |
| strategy entry | long `#4c8dff`, short `#ff7a45` | entry arrows ("long"/"short" labels when zoomed in) |
| strategy exit | `#c46bff` | exit arrows, label "Fechar long · motivo"; entry→exit dashed line green/red by P&L |
Long/short CVD ΔE 11 (deutan) — still always paired with shape (▲▼ arrows) and text.

## Type
- UI: Inter Variable. Numbers: JetBrains Mono Variable, `tabular-nums`.
- Scale (px): 10 (micro labels, uppercase +0.08em tracking) · 11 (table/body) · 12 (controls) ·
  13 (pane titles) · 15 (price in top bar) · 22 (stat values) · 28 (signal state).

## Space & shape
- 4px base grid. Bars: top 44px, status 24px, rail 44px, inspector 300px (260px < 1500px wide).
- Radius 4px controls, 6px cards; 1px borders; no shadows except palette/menus.
- Panes separated by 1px bg-0 gaps; active pane: 1px accent inset ring + accent tick in header.

## Components
- Top bar: brand mark · symbol button (live dot, last price with flash, 24h %) · TF segmented ·
  strategy picker · signal pill (LONG/SHORT/FLAT, pulsing) · layout 1/2/4 icons · sync toggles · ⌘K.
- Left rail: cursor, trend line, horizontal line, rectangle, delete, clear.
- Pane header: symbol, per-pane TF, mode (Velas / Liquidações), OHLCV legend, live dot.
- Inspector: watchlist · signal card · stats card (5 KPIs + equity sparkline) · Detalhes on demand
  (extra metrics, annual returns, trades, layers, fees).
- Status bar: API/WS health, in-sample → holdout cutoff, data range, UTC clock, shortcuts.
- Command palette (Ctrl+K): fuzzy list, grouped (Símbolo, Timeframe, Estratégia, Layout, Ferramentas).

## Motion
120–180ms ease-out for hovers/menus; price flash 600ms fade; signal pill 2s pulse; no layout
animation on the chart itself (60fps panning first).

## Strategy tester (bottom panel, TradingView "Testador de estratégia" layout)
Tabs: Visão geral · Análise de desempenho (pills Detalhamento / Periódico / Benchmarking / Uso de margem /
Crescimento e declínio; sub-pills Distribuição / Sequências / Padrões de tempo) · Resumo de desempenho
(Todos / Long / Short) · Lista de negociações (click = scroll chart) · Propriedades. Metric strips: label 11.5px
text-2 over 13px mono value; pills white-on-black when active; charts in SVG with right-hand axis.
Resizable (drag top edge), collapsible to its 34px header, maximisable. All metrics computed client-side
(strategy/stats.ts) from /backtest trades + exact `equity_daily`.
