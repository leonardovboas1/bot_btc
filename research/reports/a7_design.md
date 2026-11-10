# A7 design-phase run (design set only)

Set: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, ADAUSDT, NEARUSDT, FETUSDT; daily bars up to 2026-03-10 (exclusive; holdout cutoff 2026-03-10 14:00 UTC, loaders' default enforcement). Evaluation window 2020-01-06 -> 2026-03-09 (first 120 days of the panel excluded for strategies and benchmarks alike).

Costs: perp 5 bp fee + 2 bp slippage per side, spot 10 bp + 2 bp, funding at settlement (daily sum). Weights decided at close t earn day t+1. Spot hedge leg for C3 uses spot klines.

Design-phase look only: it informs nothing outside the registered A7 grids (no config was added or changed).

| name | Sharpe | CAGR | max DD | Calmar | turnover/yr | % pos quarters | days |
|---|---|---|---|---|---|---|---|
| C1_LO_s20 | 1.42 | 92.0% | 59.6% | 1.54 | 51.9 | 48.0% | 2255 |
| C1_LO_s40 | 1.39 | 131.2% | 74.0% | 1.77 | 94.4 | 48.0% | 2255 |
| C1_LS_s20 | 0.98 | 58.1% | 76.3% | 0.76 | 104.9 | 56.0% | 2255 |
| C1_LS_s40 | 1.07 | 74.9% | 84.5% | 0.89 | 167.3 | 52.0% | 2255 |
| C2_LO_L14 | 1.35 | 123.2% | 80.7% | 1.53 | 49.1 | 56.0% | 2255 |
| C2_LS_L14 | 0.78 | 24.0% | 43.6% | 0.55 | 50.5 | 56.0% | 2255 |
| C2_LO_L28 | 1.01 | 64.4% | 81.3% | 0.79 | 38.2 | 52.0% | 2255 |
| C2_LS_L28 | 0.67 | 18.7% | 31.5% | 0.59 | 39.0 | 52.0% | 2255 |
| C2_LO_L56 | 1.06 | 71.5% | 85.2% | 0.84 | 26.7 | 56.0% | 2255 |
| C2_LS_L56 | 0.11 | -2.4% | 45.7% | -0.05 | 30.4 | 36.0% | 2255 |
| C3_th10 | 4.82 | 5.1% | 1.7% | 2.92 | 26.9 | 64.0% | 2255 |
| C3_th20 | 4.26 | 5.8% | 1.3% | 4.31 | 15.5 | 64.0% | 2255 |
| EW_BH | 0.95 | 53.6% | 79.9% | 0.67 | 7.5 | 56.0% | 2255 |
| BTC_BH | 0.70 | 26.6% | 78.9% | 0.34 | 0.1 | 52.0% | 2255 |
| BRAYVB | 1.51 | 53.9% | 33.4% | 1.62 | n/a | 64.0% | 2255 |

Highest design Sharpe among the 12 configs: **C3_th10** (4.82) vs EW B&H 0.95, brayvb 1.51. Success rules are applied only on the validation sets.