# Liquidation model validation (observed Gate liquidations vs modeled bands)

Generated 2026-11-10 21:13 UTC by `research/validate_liq_model.py`.

## Method

- Data: Gate.io USDT-perp `contract_stats` 5m (<=180 d): OI USD and long/short liquidation USD. Prices: Binance 5m klines. Model OI input = Gate OI USD / mark (coin units), default leverage mix, causal `liq_columns`, first 5 days discarded as warm-up.
- Spike = bar where observed liquidation USD on a side is >= the 95th percentile of that symbol/side (and above the median positive value). Long and short tested separately: long spike vs modeled LONG levels consumed.
- Touch = model consumed levels of that side in the bar. Hit = touch within +-1 bar of the spike. 'big touch' = consumed USD in the top quartile of touch bars.
- Lift = hit rate / unconditional touch rate. Placebo: spike series circularly shifted 1000x by >=1 day (preserves clustering); `plift` = hit / placebo mean; p = share of placebo >= observed.
- Caveat: Gate is one venue (a fraction of market liquidations); a touch of a level in a 5m high/low is mechanically correlated with volatility, and spikes happen in volatile bars, so raw/placebo lift mostly reflects volatility. The vol-matched null compares the hit rate with the touch rate among bars in the same bar-range decile (high-low)/close as the spikes; it is the honest measure of model-specific information.

## any touch

| symbol | side | spikes | hit | base | lift | lift vs placebo | p | vol-matched base | vol-matched lift | spearman |
|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | long | 2519 | 71.5% | 11.9% | 6.03 | 6.06 | 0.001 | 27.9% | 2.57 | 0.465 |
| BTCUSDT | short | 2519 | 66.2% | 11.3% | 5.88 | 5.90 | 0.001 | 25.0% | 2.64 | 0.454 |
| ETHUSDT | long | 2519 | 77.3% | 15.2% | 5.08 | 5.10 | 0.001 | 36.1% | 2.14 | 0.471 |
| ETHUSDT | short | 2519 | 75.1% | 15.0% | 5.02 | 5.04 | 0.001 | 32.8% | 2.29 | 0.457 |
| SOLUSDT | long | 2038 | 83.5% | 20.0% | 4.17 | 4.17 | 0.001 | 38.8% | 2.15 | 0.510 |
| SOLUSDT | short | 1571 | 81.8% | 19.8% | 4.13 | 4.14 | 0.001 | 37.5% | 2.18 | 0.461 |
| BNBUSDT | long | 538 | 84.6% | 12.4% | 6.83 | 6.84 | 0.001 | 31.2% | 2.71 | 0.363 |
| BNBUSDT | short | 375 | 76.8% | 13.3% | 5.79 | 5.84 | 0.001 | 27.3% | 2.81 | 0.285 |
| NEARUSDT | long | 1277 | 93.3% | 41.5% | 2.25 | 2.25 | 0.001 | 59.7% | 1.56 | 0.333 |
| NEARUSDT | short | 998 | 91.0% | 41.1% | 2.21 | 2.23 | 0.001 | 62.0% | 1.47 | 0.300 |
| FETUSDT | long | 463 | 92.9% | 36.5% | 2.54 | 2.53 | 0.001 | 56.5% | 1.64 | 0.241 |
| FETUSDT | short | 146 | 87.7% | 34.0% | 2.58 | 2.59 | 0.001 | 53.6% | 1.64 | 0.132 |
| ADAUSDT | long | 814 | 88.7% | 28.3% | 3.13 | 3.13 | 0.001 | 49.7% | 1.79 | 0.324 |
| ADAUSDT | short | 437 | 86.7% | 26.9% | 3.23 | 3.24 | 0.001 | 45.5% | 1.91 | 0.250 |
| XRPUSDT | long | 1546 | 86.4% | 20.2% | 4.28 | 4.31 | 0.001 | 40.1% | 2.15 | 0.492 |
| XRPUSDT | short | 897 | 84.9% | 19.1% | 4.45 | 4.48 | 0.001 | 40.6% | 2.09 | 0.403 |

Pooled (spike-weighted): hit 79.8%, lift 4.60, lift vs placebo 4.62, volatility-matched lift 2.20; 16/16 symbol-sides with p<0.05.

## big touch (top 25% USD)

| symbol | side | spikes | hit | base | lift | lift vs placebo | p | vol-matched base | vol-matched lift | spearman |
|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | long | 2519 | 35.1% | 3.1% | 11.40 | 11.41 | 0.001 | 10.3% | 3.42 | 0.465 |
| BTCUSDT | short | 2519 | 31.0% | 3.0% | 10.44 | 10.47 | 0.001 | 8.9% | 3.47 | 0.454 |
| ETHUSDT | long | 2519 | 41.3% | 3.9% | 10.49 | 10.51 | 0.001 | 14.1% | 2.93 | 0.471 |
| ETHUSDT | short | 2519 | 38.5% | 3.9% | 9.83 | 9.89 | 0.001 | 12.3% | 3.12 | 0.457 |
| SOLUSDT | long | 2038 | 50.6% | 5.2% | 9.73 | 9.78 | 0.001 | 14.0% | 3.61 | 0.510 |
| SOLUSDT | short | 1571 | 50.7% | 5.1% | 9.89 | 9.96 | 0.001 | 13.8% | 3.68 | 0.461 |
| BNBUSDT | long | 538 | 59.3% | 3.3% | 18.14 | 18.21 | 0.001 | 12.2% | 4.85 | 0.363 |
| BNBUSDT | short | 375 | 53.1% | 3.4% | 15.43 | 15.72 | 0.001 | 10.0% | 5.32 | 0.285 |
| NEARUSDT | long | 1277 | 71.3% | 11.4% | 6.25 | 6.28 | 0.001 | 23.2% | 3.07 | 0.333 |
| NEARUSDT | short | 998 | 67.8% | 10.9% | 6.20 | 6.20 | 0.001 | 25.6% | 2.65 | 0.300 |
| FETUSDT | long | 463 | 79.5% | 10.0% | 7.96 | 7.97 | 0.001 | 22.8% | 3.49 | 0.241 |
| FETUSDT | short | 146 | 65.1% | 9.1% | 7.17 | 7.26 | 0.001 | 20.9% | 3.11 | 0.132 |
| ADAUSDT | long | 814 | 66.3% | 7.6% | 8.77 | 8.79 | 0.001 | 19.1% | 3.47 | 0.324 |
| ADAUSDT | short | 437 | 67.7% | 7.0% | 9.64 | 9.80 | 0.001 | 17.6% | 3.86 | 0.250 |
| XRPUSDT | long | 1546 | 57.1% | 5.4% | 10.48 | 10.53 | 0.001 | 15.1% | 3.78 | 0.492 |
| XRPUSDT | short | 897 | 55.5% | 5.0% | 11.13 | 11.32 | 0.001 | 15.4% | 3.60 | 0.403 |

Pooled (spike-weighted): hit 48.6%, lift 10.08, lift vs placebo 10.14, volatility-matched lift 3.43; 16/16 symbol-sides with p<0.05.

## Verdict

Versus shuffled-time placebo the model shows large lift (pooled 4.62, 16/16 symbol-sides p<0.05), but that is mostly volatility: liquidation spikes occur in wide-range bars, which also cross more levels. Against a volatility-matched null the pooled lift is 2.20. Only that last number measures model-specific information; it is above 1, so the levels carry some information beyond volatility.
