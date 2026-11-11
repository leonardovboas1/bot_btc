# A8 design: indicator long/short strategies, one shared parameter set across the 8 design coins

Combos run: 280 (4h and 1d; 1h skipped, see below); Optuna trials: 22400 (all recorded in `research/trials.duckdb`, stage `a8`); combos pruned after 20 trials (best training Sharpe < 0): 0; combos passing all gates: 92.

Selection per combo: trial with best training objective (median over coins of mean in-sample fold Sharpe). Score = median over coins of stitched OOS Sharpe of that trial. Gates: OOS Sharpe > 0 on >= 6/8 coins; pooled OOS profit factor > 1.1; short-leg pooled PnL >= 0 or its loss <= 20% of the long leg's gross loss; >= 30 OOS trades on every coin (strict, also on 1d). Costs BINANCE_PERP_TAKER + funding; 6 WF folds per coin (coins with < 12 months training dropped from that fold, as in A5). 1h was not run: the 25 fresh V1 coins have no 1h data on disk, so 1h candidates could not be validated; the T1/T2 grids were run on 4h and 1d only.

## Indicator mapping (named indicator -> registry primitive)

```
named indicator -> registry primitive actually used)
- EMA/SMA 50/100/200 price-above  -> macross (kind ema|sma, slow in 50/100/200): close_above_slow / close_below_slow
  (no slope primitive exists: the 'slope' variant is mapped to fast_above_slow / fast_below_slow, fast in 10/20)
- Supertrend filter               -> supertrend.up / supertrend.down
- Ichimoku cloud filter           -> ichimoku.above_cloud / below_cloud
- ADX>theta filter                -> dmi.bull_strong / bear_strong (DI+ > DI- and ADX > thr)
- Gaussian Channel filter         -> gc.filter_rising / filter_falling
- MACD trigger                    -> macd.cross_up/cross_down ("cross") or macd.hist_rising / not:hist_rising ("hist")
- RSI cross of levels             -> NO rsi indicator in registry; closest = mfi (volume-weighted RSI): cross_up_os / cross_down_ob
- Stoch RSI K/D cross in zones    -> stochrsi.k_cross_up_d / k_cross_down_d ("kd") or bull_cross_os / bear_cross_ob ("zone")
- Cipher B WT cross               -> cipherb.wt_cross_up / wt_cross_down
- QQE cross                       -> qqe.cross_up / cross_down
- Bollinger re-entry              -> no re-entry primitive; closest = bb.cross_above_basis/cross_below_basis ("basis")
                                     or bb.cross_above_upper/cross_below_lower ("breakout")
- T2 SAR indicators: supertrend (flip_up/flip_down, state up/down), utbot (buy/sell, long_pos/short_pos),
  psar (flip_long/flip_short, long/short), ssl (flip_up/flip_down, bull/bear), rangefilter (buy/sell, bull/bear),
  gc (cross_above_filt/cross_below_filt, close_above_filt/close_below_filt), hull (close_cross_up/down, rising/falling),
  donchian (breakout_up/down; exit when close crosses mid: close_below_mid / close_above_mid).
```

## Top 30 by score (all combos; gates column: pos6/8, PF, side, trades)

| # | combo | score (med OOS Sharpe) | train obj | coins>0 | PF | short pnl | long pnl | min trades | gates |
|---|---|---|---|---|---|---|---|---|---|
| 1 | T1:gc:qqe:4h:real | 1.10 | 0.75 | 5/8 | 1.56 | 24.96 | 39.56 | 97 | pos fail |
| 2 | T1:ich:qqe:4h:real | 1.06 | 0.90 | 6/8 | 1.42 | 5.93 | 22.56 | 74 | ALL |
| 3 | T1:ema:mfi:4h:real | 0.94 | 0.98 | 7/8 | 0.15 | 0.00 | -4.48 | 0 | pf,trades fail |
| 4 | T1:gc:mfi:4h:real | 0.91 | 0.97 | 7/8 | 0.04 | 0.00 | -77.19 | 0 | pf,trades fail |
| 5 | T1:ema:qqe:4h:ha | 0.90 | 0.51 | 6/8 | 1.22 | 4.22 | 8.38 | 103 | ALL |
| 6 | T2:ema:gc:1d:ha | 0.86 | 0.82 | 6/8 | 2.36 | 19.49 | 90.33 | 17 | trades fail |
| 7 | T1:ich:qqe:1d:real | 0.86 | 0.90 | 7/8 | 1.10 | -13.65 | 16.94 | 12 | pf,side,trades fail |
| 8 | T2:gc:psar:4h:real | 0.85 | 0.65 | 8/8 | 1.23 | 44.28 | 24.92 | 124 | ALL |
| 9 | T1:ich:qqe:1d:ha | 0.83 | 0.88 | 6/8 | 1.44 | 1.26 | 15.06 | 14 | trades fail |
| 10 | T1:ich:qqe:4h:ha | 0.83 | 0.81 | 7/8 | 1.25 | 5.17 | 15.18 | 80 | ALL |
| 11 | T1:gc:macd:1d:ha | 0.82 | 0.99 | 7/8 | 1.12 | -15.09 | 31.79 | 34 | side fail |
| 12 | T1:gc:mfi:4h:ha | 0.82 | 0.79 | 6/8 | 5.81 | 0.87 | 0.00 | 0 | trades fail |
| 13 | T2:adx:gc:1d:ha | 0.81 | 0.95 | 7/8 | 1.64 | 11.85 | 9.29 | 19 | trades fail |
| 14 | T1:ich:macd:1d:real | 0.79 | 1.05 | 6/8 | 1.50 | -9.10 | 35.32 | 20 | side,trades fail |
| 15 | T2:ema:gc:1d:real | 0.79 | 1.06 | 7/8 | 1.54 | 4.91 | 18.28 | 30 | ALL |
| 16 | T1:ema:macd:1d:real | 0.78 | 1.14 | 7/8 | 1.36 | -2.78 | 28.51 | 23 | trades fail |
| 17 | T2:ema:hull:1d:ha | 0.78 | 0.97 | 7/8 | 1.21 | 0.57 | 19.21 | 62 | ALL |
| 18 | T2:None:gc:4h:real | 0.76 | 0.82 | 8/8 | 1.11 | -2.59 | 27.95 | 256 | ALL |
| 19 | T2:st:ssl:4h:ha | 0.75 | 0.93 | 7/8 | 1.36 | -5.34 | 40.89 | 38 | ALL |
| 20 | T2:ich:gc:1d:real | 0.75 | 0.75 | 7/8 | 1.52 | 0.92 | 14.21 | 16 | trades fail |
| 21 | T2:ich:hull:1d:real | 0.74 | 0.52 | 6/8 | 1.78 | 3.30 | 14.69 | 12 | trades fail |
| 22 | T1:ich:macd:1d:ha | 0.73 | 1.07 | 7/8 | 1.29 | -5.07 | 23.10 | 23 | trades fail |
| 23 | T2:adx:gc:1d:real | 0.73 | 0.64 | 8/8 | 1.71 | 36.62 | 164.97 | 41 | ALL |
| 24 | T1:ema:qqe:4h:real | 0.72 | 0.62 | 7/8 | 1.19 | 8.88 | 9.24 | 121 | ALL |
| 25 | T1:adx:mfi:1d:real | 0.71 | 0.64 | 8/8 | 3.66 | 1.14 | 3.02 | 1 | trades fail |
| 26 | T1:ich:macd:4h:real | 0.70 | 1.16 | 7/8 | 1.15 | -5.71 | 43.10 | 94 | ALL |
| 27 | T1:gc:macd:4h:ha | 0.70 | 1.05 | 8/8 | 1.14 | 5.23 | 54.24 | 127 | ALL |
| 28 | T2:ema:utbot:4h:ha | 0.70 | 0.99 | 7/8 | 1.28 | 17.25 | 31.22 | 43 | ALL |
| 29 | T1:gc:stochrsi:1d:ha | 0.70 | 0.96 | 8/8 | 1.15 | 24.15 | 20.38 | 81 | ALL |
| 30 | T2:ema:psar:4h:real | 0.69 | 0.54 | 8/8 | 1.24 | 30.05 | 25.26 | 82 | ALL |

## Advancing candidates (10: top by score among combos passing all gates)

### 1. a8c1: T1:ich:qqe:4h:real

- Regra: COMPRA (long) quando: preco acima da nuvem Ichimoku E QQE cruza para cima; VENDE a descoberto (short) quando: preco abaixo da nuvem Ichimoku E QQE cruza para baixo; sai do long quando: (QQE cruza para baixo OU preco abaixo da nuvem Ichimoku); sai do short quando: (QQE cruza para cima OU preco acima da nuvem Ichimoku).
- Parametros: `{"f": {"conv": 18, "base": 28, "span_b": 62, "disp": 28}, "t": {"rsi_len": 20, "smooth": 10, "factor": 4.25}, "exit": "opp+flip", "risk": {}}`
- Design: score 1.06, treino 0.90, 6/8 moedas > 0, PF 1.42, short pnl 5.93 / long pnl 22.56, Sharpe EW OOS 1.50, maxDD EW 17%. Por moeda: {'BTCUSDT': -0.273, 'ETHUSDT': -0.082, 'SOLUSDT': 1.199, 'BNBUSDT': 0.005, 'XRPUSDT': 1.101, 'ADAUSDT': 1.289, 'NEARUSDT': 1.021, 'FETUSDT': 1.549}
- T3 (so long / so short, OOS med. Sharpe): long 0.55 (PF 1.60), short 0.34 (PF 1.41)

### 2. a8c2: T1:ema:qqe:4h:ha

- Regra: COMPRA (long) quando: preco fecha acima da media movel lenta E QQE cruza para cima; VENDE a descoberto (short) quando: preco fecha abaixo da media movel lenta E QQE cruza para baixo; sai do long quando: QQE cruza para baixo; sai do short quando: QQE cruza para cima; stop: sl_atr=2.0 (multiplos de ATR).
- Parametros: `{"f": {"kind": "ema", "slow": 50, "fast": 10, "mode": "price"}, "t": {"rsi_len": 17, "smooth": 7, "factor": 5.0}, "exit": "opp+atr", "risk": {"sl_atr": 2.0, "trail_atr": 0.0}}`
- Design: score 0.90, treino 0.51, 6/8 moedas > 0, PF 1.22, short pnl 4.22 / long pnl 8.38, Sharpe EW OOS 1.21, maxDD EW 20%. Por moeda: {'BTCUSDT': -0.065, 'ETHUSDT': 0.101, 'SOLUSDT': 0.99, 'BNBUSDT': -0.162, 'XRPUSDT': 1.12, 'ADAUSDT': 0.858, 'NEARUSDT': 1.126, 'FETUSDT': 0.938}
- T3 (so long / so short, OOS med. Sharpe): long 0.48 (PF 1.23), short 0.34 (PF 1.18)

### 3. a8c3: T2:gc:psar:4h:real

- Regra: COMPRA (long) quando: psar.flip long E Gaussian Channel com filtro subindo; VENDE a descoberto (short) quando: psar.flip short E Gaussian Channel com filtro caindo; sai do long quando: psar.short; sai do short quando: psar.long.
- Parametros: `{"s": {"start": 0.01, "inc": 0.005, "maximum": 0.1}, "exit": "sar", "risk": {}, "f": {"poles": 1, "period": 90, "mult": 1.7, "mode_lag": false, "mode_fast": true}}`
- Design: score 0.85, treino 0.65, 8/8 moedas > 0, PF 1.23, short pnl 44.28 / long pnl 24.92, Sharpe EW OOS 1.27, maxDD EW 33%. Por moeda: {'BTCUSDT': 0.305, 'ETHUSDT': 0.581, 'SOLUSDT': 1.301, 'BNBUSDT': 0.006, 'XRPUSDT': 1.073, 'ADAUSDT': 0.901, 'NEARUSDT': 0.804, 'FETUSDT': 1.057}
- T3 (so long / so short, OOS med. Sharpe): long 0.72 (PF 1.25), short 0.40 (PF 1.11)

### 4. a8c4: T1:ich:qqe:4h:ha

- Regra: COMPRA (long) quando: preco acima da nuvem Ichimoku E QQE cruza para cima; VENDE a descoberto (short) quando: preco abaixo da nuvem Ichimoku E QQE cruza para baixo; sai do long quando: (QQE cruza para baixo OU preco abaixo da nuvem Ichimoku); sai do short quando: (QQE cruza para cima OU preco acima da nuvem Ichimoku).
- Parametros: `{"f": {"conv": 20, "base": 40, "span_b": 64, "disp": 28}, "t": {"rsi_len": 21, "smooth": 9, "factor": 3.25}, "exit": "opp+flip", "risk": {}}`
- Design: score 0.83, treino 0.81, 7/8 moedas > 0, PF 1.25, short pnl 5.17 / long pnl 15.18, Sharpe EW OOS 1.33, maxDD EW 16%. Por moeda: {'BTCUSDT': 0.212, 'ETHUSDT': 0.361, 'SOLUSDT': 0.787, 'BNBUSDT': -0.07, 'XRPUSDT': 0.979, 'ADAUSDT': 0.91, 'NEARUSDT': 0.867, 'FETUSDT': 1.456}
- T3 (so long / so short, OOS med. Sharpe): long 0.69 (PF 1.32), short 0.06 (PF 1.42)

### 5. a8c5: T2:ema:gc:1d:real

- Regra: COMPRA (long) quando: gc.cross above filt E preco fecha acima da media movel lenta; VENDE a descoberto (short) quando: gc.cross below filt E preco fecha abaixo da media movel lenta; sai do long quando: gc.close below filt; sai do short quando: gc.close above filt.
- Parametros: `{"s": {"poles": 2, "period": 100, "mult": 2.3, "mode_lag": true, "mode_fast": true}, "exit": "sar", "risk": {}, "f": {"kind": "ema", "slow": 100, "fast": 20, "mode": "price"}}`
- Design: score 0.79, treino 1.06, 7/8 moedas > 0, PF 1.54, short pnl 4.91 / long pnl 18.28, Sharpe EW OOS 1.23, maxDD EW 28%. Por moeda: {'BTCUSDT': 0.878, 'ETHUSDT': 0.221, 'SOLUSDT': 0.935, 'BNBUSDT': 0.996, 'XRPUSDT': -0.229, 'ADAUSDT': 1.139, 'NEARUSDT': 0.702, 'FETUSDT': 0.606}
- T3 (so long / so short, OOS med. Sharpe): long 0.53 (PF 1.56), short 0.46 (PF 1.56)

### 6. a8c6: T2:ema:hull:1d:ha

- Regra: COMPRA (long) quando: hull.close cross up E media rapida acima da lenta; VENDE a descoberto (short) quando: hull.close cross down E media rapida abaixo da lenta; sai do long quando: hull.falling; sai do short quando: hull.rising.
- Parametros: `{"s": {"length": 15}, "exit": "sar", "risk": {}, "f": {"kind": "sma", "slow": 100, "fast": 20, "mode": "fastslow"}}`
- Design: score 0.78, treino 0.97, 7/8 moedas > 0, PF 1.21, short pnl 0.57 / long pnl 19.21, Sharpe EW OOS 0.96, maxDD EW 28%. Por moeda: {'BTCUSDT': 1.211, 'ETHUSDT': 1.199, 'SOLUSDT': 0.646, 'BNBUSDT': 0.949, 'XRPUSDT': -0.546, 'ADAUSDT': 0.409, 'NEARUSDT': 0.292, 'FETUSDT': 0.907}
- T3 (so long / so short, OOS med. Sharpe): long 0.75 (PF 1.54), short 0.16 (PF 1.03)

### 7. a8c7: T2:None:gc:4h:real

- Regra: COMPRA (long) quando: gc.cross above filt; VENDE a descoberto (short) quando: gc.cross below filt; sai do long quando: gc.close below filt; sai do short quando: gc.close above filt.
- Parametros: `{"s": {"poles": 2, "period": 194, "mult": 1.2, "mode_lag": true, "mode_fast": false}, "exit": "sar", "risk": {}}`
- Design: score 0.76, treino 0.82, 8/8 moedas > 0, PF 1.11, short pnl -2.59 / long pnl 27.95, Sharpe EW OOS 0.90, maxDD EW 32%. Por moeda: {'BTCUSDT': 0.81, 'ETHUSDT': 0.417, 'SOLUSDT': 0.238, 'BNBUSDT': 0.863, 'XRPUSDT': 0.711, 'ADAUSDT': 0.111, 'NEARUSDT': 0.966, 'FETUSDT': 0.828}
- T3 (so long / so short, OOS med. Sharpe): long 0.78 (PF 1.23), short -0.03 (PF 0.93)

### 8. a8c8: T2:st:ssl:4h:ha

- Regra: COMPRA (long) quando: ssl.flip up E Supertrend em alta; VENDE a descoberto (short) quando: ssl.flip down E Supertrend em baixa; sai do long quando: ssl.bear; sai do short quando: ssl.bull.
- Parametros: `{"s": {"length": 44}, "exit": "sar", "risk": {}, "f": {"atr_len": 23, "factor": 3.25}}`
- Design: score 0.75, treino 0.93, 7/8 moedas > 0, PF 1.36, short pnl -5.34 / long pnl 40.89, Sharpe EW OOS 1.24, maxDD EW 28%. Por moeda: {'BTCUSDT': 0.736, 'ETHUSDT': 0.159, 'SOLUSDT': 0.847, 'BNBUSDT': 0.764, 'XRPUSDT': 1.356, 'ADAUSDT': 1.216, 'NEARUSDT': 0.152, 'FETUSDT': -0.209}
- T3 (so long / so short, OOS med. Sharpe): long 1.13 (PF 1.82), short 0.02 (PF 0.89)

### 9. a8c9: T2:adx:gc:1d:real

- Regra: COMPRA (long) quando: gc.cross above filt E ADX forte com DI+ > DI-; VENDE a descoberto (short) quando: gc.cross below filt E ADX forte com DI- > DI+; sai do long quando: gc.close below filt; sai do short quando: gc.close above filt.
- Parametros: `{"s": {"poles": 5, "period": 96, "mult": 2.3, "mode_lag": false, "mode_fast": true}, "exit": "sar", "risk": {}, "f": {"di_len": 9, "adx_len": 30, "adx_thr": 17.0}}`
- Design: score 0.73, treino 0.64, 8/8 moedas > 0, PF 1.71, short pnl 36.62 / long pnl 164.97, Sharpe EW OOS 1.17, maxDD EW 28%. Por moeda: {'BTCUSDT': 0.859, 'ETHUSDT': 0.926, 'SOLUSDT': 1.219, 'BNBUSDT': 0.316, 'XRPUSDT': 0.028, 'ADAUSDT': 0.632, 'NEARUSDT': 0.674, 'FETUSDT': 0.778}
- T3 (so long / so short, OOS med. Sharpe): long 0.78 (PF 2.13), short 0.04 (PF 1.12)

### 10. a8c10: T1:ema:qqe:4h:real

- Regra: COMPRA (long) quando: preco fecha acima da media movel lenta E QQE cruza para cima; VENDE a descoberto (short) quando: preco fecha abaixo da media movel lenta E QQE cruza para baixo; sai do long quando: QQE cruza para baixo; sai do short quando: QQE cruza para cima.
- Parametros: `{"f": {"kind": "ema", "slow": 100, "fast": 10, "mode": "price"}, "t": {"rsi_len": 21, "smooth": 10, "factor": 2.5}, "exit": "opp", "risk": {}}`
- Design: score 0.72, treino 0.62, 7/8 moedas > 0, PF 1.19, short pnl 8.88 / long pnl 9.24, Sharpe EW OOS 1.06, maxDD EW 21%. Por moeda: {'BTCUSDT': 0.075, 'ETHUSDT': 0.796, 'SOLUSDT': 0.776, 'BNBUSDT': 0.142, 'XRPUSDT': -0.04, 'ADAUSDT': 0.687, 'NEARUSDT': 0.762, 'FETUSDT': 1.55}
- T3 (so long / so short, OOS med. Sharpe): long 0.45 (PF 1.29), short -0.07 (PF 1.31)
