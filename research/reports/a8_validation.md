# A8 validation (V1 = 25 fresh coins, V2 = 8 design coins after 2026-03-10 14:00 UTC)

**VERDICT: 0 of 10 candidates pass V1 and V2** (nothing validated; the top-3 by V1 score are marked 'nao validada')
- a8c1 (T1:ich:qqe:4h:real): V1 FAIL, V2 FAIL => **REJEITADA**
- a8c2 (T1:ema:qqe:4h:ha): V1 FAIL, V2 FAIL => **REJEITADA**
- a8c3 (T2:gc:psar:4h:real): V1 FAIL, V2 FAIL => **REJEITADA**
- a8c4 (T1:ich:qqe:4h:ha): V1 FAIL, V2 FAIL => **REJEITADA**
- a8c5 (T2:ema:gc:1d:real): V1 FAIL, V2 FAIL => **REJEITADA**
- a8c6 (T2:ema:hull:1d:ha): V1 FAIL, V2 FAIL => **REJEITADA**
- a8c7 (T2:None:gc:4h:real): V1 FAIL, V2 FAIL => **REJEITADA**
- a8c8 (T2:st:ssl:4h:ha): V1 FAIL, V2 FAIL => **REJEITADA**
- a8c9 (T2:adx:gc:1d:real): V1 FAIL, V2 PASS => **REJEITADA**
- a8c10 (T1:ema:qqe:4h:real): V1 FAIL, V2 FAIL => **REJEITADA**

DSR deflated at N = 213906 (all rows of the trials table incl. a8 design, T3 and these validation rows); trial-Sharpe variance from the 10 candidates on that set. Rules (A8): median-over-coins Sharpe > 0.5; Sharpe > 0 on >= 60% of coins; pooled profit factor > 1.15; EW portfolio maxDD <= 0.7 x EW B&H maxDD; on V1 also EW portfolio Sharpe >= EW B&H Sharpe. V1 starts 120 days after each coin's first bar; V2 evaluates bars >= cutoff with continuous signal state. Costs BINANCE_PERP_TAKER + funding; fills next-bar open. B&H = perp close-to-close incl. funding, equal weight, daily rebalance.

## Benchmarks

| set | EW B&H Sharpe | EW B&H maxDD | EW B&H CAGR | brayvb_v2 default 4h EW Sharpe | brayvb maxDD | brayvb CAGR | brayvb PF | brayvb med coin Sharpe |
|---|---|---|---|---|---|---|---|---|
| V1 (25 coins) | 0.66 | 84.9% | 22.6% | 0.79 | 39.3% | 20.7% | 1.18 | 0.31 |
| V2 (8 coins) | 0.48 | 28.5% | 12.3% | 0.38 | 17.9% | 6.6% | 0.90 | -0.49 |

## V1 per candidate

| cand | combo | med coin Sharpe | % coins>0 | PF | EW Sharpe | EW maxDD | EW CAGR | EW B&H Sharpe | B&H maxDD | short pnl | long pnl | trades | DSR p | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a8c1 | T1:ich:qqe:4h:real | 0.17 | 64.0% | 1.17 | 0.60 | 26.0% | 13.8% | 0.66 | 84.9% | 13.58 | 29.88 | 5314 | 0.26 | N |
| a8c2 | T1:ema:qqe:4h:ha | 0.15 | 68.0% | 1.05 | 0.38 | 33.7% | 7.5% | 0.66 | 84.9% | -0.16 | 20.33 | 7646 | 0.12 | N |
| a8c3 | T2:gc:psar:4h:real | 0.21 | 72.0% | 1.05 | 0.64 | 35.1% | 21.3% | 0.66 | 84.9% | 57.45 | 0.42 | 8154 | 0.30 | N |
| a8c4 | T1:ich:qqe:4h:ha | 0.20 | 68.0% | 1.11 | 0.50 | 30.6% | 11.1% | 0.66 | 84.9% | 20.64 | 17.93 | 6383 | 0.19 | N |
| a8c5 | T2:ema:gc:1d:real | 0.27 | 84.0% | 1.17 | 0.88 | 34.9% | 22.1% | 0.67 | 84.8% | -5.79 | 30.64 | 2281 | 0.54 | N |
| a8c6 | T2:ema:hull:1d:ha | 0.51 | 88.0% | 1.12 | 0.96 | 31.8% | 29.1% | 0.67 | 84.8% | -6.71 | 47.37 | 4258 | 0.63 | N |
| a8c7 | T2:None:gc:4h:real | 0.40 | 76.0% | 1.05 | 0.67 | 54.9% | 25.0% | 0.66 | 84.9% | -3.00 | 76.67 | 17802 | 0.33 | N |
| a8c8 | T2:st:ssl:4h:ha | 0.37 | 72.0% | 1.20 | 0.81 | 30.3% | 20.9% | 0.66 | 84.9% | 46.50 | 24.81 | 3052 | 0.47 | N |
| a8c9 | T2:adx:gc:1d:real | 0.44 | 84.0% | 1.35 | 0.78 | 39.0% | 26.0% | 0.67 | 84.8% | 72.49 | 71.64 | 2780 | 0.43 | N |
| a8c10 | T1:ema:qqe:4h:real | 0.23 | 68.0% | 1.08 | 0.50 | 32.7% | 12.3% | 0.66 | 84.9% | 8.15 | 43.13 | 8950 | 0.19 | N |

## V2 per candidate

| cand | combo | med coin Sharpe | % coins>0 | PF | EW Sharpe | EW maxDD | EW CAGR | EW B&H Sharpe | B&H maxDD | short pnl | long pnl | trades | DSR p | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a8c1 | T1:ich:qqe:4h:real | -0.49 | 25.0% | 0.78 | -0.39 | 20.1% | -8.7% | 0.48 | 28.5% | -0.38 | -4.90 | 224 | 0.00 | N |
| a8c2 | T1:ema:qqe:4h:ha | -0.49 | 25.0% | 0.77 | -0.72 | 21.0% | -16.6% | 0.48 | 28.5% | -0.15 | -3.82 | 309 | 0.00 | N |
| a8c3 | T2:gc:psar:4h:real | -0.06 | 50.0% | 1.11 | 0.38 | 25.7% | 7.5% | 0.48 | 28.5% | 5.30 | 6.89 | 362 | 0.01 | N |
| a8c4 | T1:ich:qqe:4h:ha | 0.07 | 50.0% | 0.91 | -0.02 | 20.8% | -2.5% | 0.48 | 28.5% | 0.58 | -2.68 | 265 | 0.00 | N |
| a8c5 | T2:ema:gc:1d:real | 0.21 | 50.0% | 0.95 | 0.52 | 8.3% | 7.8% | 0.60 | 27.4% | 0.68 | -1.37 | 91 | 0.01 | N |
| a8c6 | T2:ema:hull:1d:ha | -0.77 | 25.0% | 1.01 | -0.18 | 18.4% | -7.0% | 0.60 | 27.4% | 4.87 | -4.53 | 190 | 0.00 | N |
| a8c7 | T2:None:gc:4h:real | -0.10 | 50.0% | 0.86 | -0.36 | 36.0% | -21.8% | 0.48 | 28.5% | -1.73 | -6.43 | 806 | 0.00 | N |
| a8c8 | T2:st:ssl:4h:ha | 0.41 | 87.5% | 1.47 | 1.53 | 13.3% | 33.7% | 0.48 | 28.5% | 12.79 | -1.89 | 125 | 0.05 | N |
| a8c9 | T2:adx:gc:1d:real | 0.95 | 62.5% | 1.32 | 1.38 | 15.6% | 45.0% | 0.60 | 27.4% | 17.97 | 12.64 | 117 | 0.04 | Y |
| a8c10 | T1:ema:qqe:4h:real | 0.12 | 62.5% | 1.07 | 0.31 | 24.7% | 4.6% | 0.48 | 28.5% | 4.36 | -2.28 | 357 | 0.01 | N |

## Best vs brayvb_v2 and EW B&H

- V1: a8c6 EW Sharpe 0.96 / maxDD 31.8% vs brayvb 0.79 / 39.3% vs EW B&H 0.67 / 84.8%
- V2: a8c6 EW Sharpe -0.18 / maxDD 18.4% vs brayvb 0.38 / 17.9% vs EW B&H 0.60 / 27.4%

Runtime of validation: 15s.