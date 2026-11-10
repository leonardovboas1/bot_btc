# H1-H6 pre-registered hypothesis evaluation

Generated 2026-11-10 16:21 by `research/h1_h6.py`; preregistration sha256 `0c5d002b111982e71935484c359524d487d30c46acf92f923c2fac0b6b4f7aa2`; holdout never loaded (data cut at the cutoff day 2026-03-10, daily bars through 2026-03-09).

## Verdicts

- **H1: REJECTED** - OOS Sharpe > 0 in only 3/6 folds; beats B&H Calmar in only 1/6 folds. S2.1 eligible for combination: **NO**. [DSR p=0.405 (<= 0.95), WF 3/6 folds>0, A2 trade minimums FAIL (IS 121, per-fold OOS min 5), S3 (>=50 bp & funding<=15%/yr) FAIL]
- **H2: REJECTED** - DSR-adjusted OOS Sharpe above H1 alone in only 0/6 folds. S2.1 eligible for combination: **NO**. [DSR p=0.002 (<= 0.95), WF 4/6 folds>0, A2 trade minimums FAIL (IS 98, per-fold OOS min 5), S3 (>=50 bp & funding<=15%/yr) ok]
- **H3: PASS** - all registered falsification criteria survived. S2.1 eligible for combination: **NO**. [DSR p=0.819 (<= 0.95), WF 4/6 folds>0, A2 trade minimums ok (IS 263, per-fold OOS min 15), S3 (>=50 bp & funding<=15%/yr) FAIL]
- **H4: REJECTED** - beats unfiltered H1 OOS Sharpe in only 3/6 folds; OOS precision of taken trades 35.7% (<= 52%) on 28 taken events. S2.1 eligible for combination: **NO**. [DSR p=0.021 (<= 0.95), WF 2/6 folds>0, A2 trade minimums FAIL (IS 28, per-fold OOS min 0), S3 (>=50 bp & funding<=15%/yr) FAIL]
- **H5-filter: UNTESTABLE** - legs with < 20 independent events: fragile longs (2), fragile shorts (3). S2.1 eligible for combination: **NO**. [DSR p=0.758 (<= 0.95), WF 4/6 folds>0, A2 trade minimums ok (IS 196, per-fold OOS min 9), S3 (>=50 bp & funding<=15%/yr) FAIL]
- **H5-standalone: UNTESTABLE** - legs with < 20 independent events: fragile longs (2), fragile shorts (2) (the individual criteria would also fail: net expectancy 45.3 bp per round trip < 50 bp (28 trades)). S2.1 eligible for combination: **NO**. [DSR p=0.005 (<= 0.95), WF 5/6 folds>0, A2 trade minimums FAIL (IS 59, per-fold OOS min 1), S3 (>=50 bp & funding<=15%/yr) FAIL]
- **H6: REJECTED** - beats raw H1 OOS Sharpe in only 2/4 folds (needs 3). S2.1 eligible for combination: **NO**. [DSR p=0.332 (<= 0.95), WF 3/4 folds>0, A2 trade minimums ok (IS 171, per-fold OOS min 9), S3 (>=50 bp & funding<=15%/yr) FAIL] Depth window: 4 folds [2, 3, 4, 5], 24 OOS months, SE of Sharpe x1.22 vs full window; applied threshold 3/4.

**Bottom line:** 1 of 7 hypothesis variants survive their own falsification criteria; 0 are eligible for the S2 combination step (falsification pass + WF + family DSR p > 0.95 at N = 191,598).

## Trial accounting (S5.1, A1.3)

- Trials table before this run: 191,589 rows (excluding earlier h1-h6 rows: 190,884). New rows from this run: 705 (of which H5/H6 additions under A1: 420). Amendment add-on: 9 (A1 +1, A2 +1, A3 +1, A4 +6).
- **DSR N with A1 additions: 191,598; without A1 additions: 191,177** (inflation 421).

## Decisions taken before the run (ambiguities in the pre-registration)

```
OPERATIONAL DECISIONS (fixed BEFORE the first run; repeated at the top of the report)
 D1  Simulator: the engine (btclab/engine/sim.py) only supports discrete entry/exit, not continuous vol-scaled
     positions, so a daily vectorised position simulator is used with the engine's cost model
     BINANCE_PERP_TAKER (fee 0.05% + slippage 2 bp = 7 bp per side on |delta position|) and Binance 8h funding
     (positive rate: longs pay).  Signal on close t, fill at open t+1.  Equity compounding is ignored inside the
     position scale (position in equity units); max |position| = L_max = 2 (S3).  The S3 turnover guard is applied
     to every series: rebalance only if |target - current| > 0.10, always on exit / sign change.  The S3 kill switch,
     soft brake and daily loss limit are NOT applied to hypothesis tests (they are "final model" rules).
 D2  Grids (fixed here; "median parameters" = middle of each registered range):
       H1 sigma-window {20,40,60}, sigma_target {0.2,0.3,0.4}, short_scale {0,0.5,1}, deadband {0,0.125,0.25} (81)
       H1 median = (40, 0.30, 0.5, 0.125) is used as "H1 at median parameters" and as "raw H1" everywhere.
       H2 n {10,25,40}, theta {0.2,0.3,0.4}, z {1.5,2,2.5}, short_scale {0,0.5,1} (81)
       H3 z-window {30,60,90}, tau {1.5,2,2.5,3}, h {0,0.5}, hold {1,3,5,7} (96)
       H4 C {0.01,0.1,1}, p_min {0.50,0.55,0.60}, vertical barrier {5,10,15} (27)
       H5 k {1,3}, w {30,60,90}, tau {1.5,2,2.5,3}, hold {1,3,5,7} (96 per mode; each mode also re-run with
          sum_open_interest_value for the registered value check (96 more each))
       H6 b {1%,2%}, w {20,40,60}, gamma {0.5,1,1.5} (18; placebo 18 more)
 D3  Walk-forward = the standard 6 folds of build_context (18m train / 6m test, 5D embargo, 200-bar purge, end =
     cutoff day).  "WF OOS series" of a hypothesis = per fold the configuration with the best IN-SAMPLE Sharpe on that
     fold's train window (nested selection), stitched over the six test windows.  Raw H1 / B&H / brayvb_v2 are
     compared on the same folds.  B&H = spot-style close-to-close returns (no funding).
 D4  "Trade" = round trip: a maximal run of constant-sign non-zero position (entry from flat or sign flip).  Rebalance
     fills inside a run are not trades (counts of fills are reported as information).  Trade return = run net P&L
     (fees, slippage, funding) / mean |position| of the run.  A2 minimums (40 in-sample trades over the union of train
     windows, 8 per OOS fold) are reported as a flag; S2.1 eligibility = falsification passed + WF + family DSR gate;
     A2 and S3 (>=50 bp net expectancy per round trip, funding bleed <= 15 %/yr) are reported next to it and are
     required for a final model, not for the hypothesis verdict (except H5-standalone, where the 50 bp rule is in the
     registered falsification text).
 D5  DSR: p of the stitched WF OOS series with N = all rows of the trials table (+ these trials + amendments), trial
     Sharpe variance = variance over the hypothesis family matrix (as btclab.research.stats.stages.family_stats);
     also shown: N = family size only, and the all-trials variance.  PBO = CSCV 16 blocks on the family OOS matrix;
     SPA (Hansen) on the family matrix vs B&H.  "DSR-adjusted OOS Sharpe" (H2) = fold Sharpe minus the expected-max
     Sharpe of the family (N = 81, fold-level trial variance), H1 alone has no haircut (N = 1).
 D6  H2: sigma_20 = rolling std of closes (20); RSI(3) Wilder; MR size = 0.5 * sigma_target/sigma_realized of H1
     median; "no position within 2 bars of a regime flip" = flat on the flip bar and the next one; the trend leg and the
     MR leg are two separately guarded books; exit at close >= SMA20 (long) / <= SMA20 (short) or after 5 closes.
     MR expectancy = mean net round-trip return of MR trades in OOS (no MR trades = fails).
 D7  H3: funding z = rolling z of the 8h prints over w days (3w prints, min 0.8), last print <= day end; "close above
     its 5-day low" = close > min of the last 5 daily lows (incl. today); add-on = +0.5 * vol-scaled size while a
     trigger occurred in the last `hold` days; added to H1 (sum).  Untestable if the independent extremes (|z|>tau,
     >=5 d apart) summed over the OOS windows with each fold's selected (w, tau) are < 20.
 D8  H4: primary event = every H1 order that opens or increases exposure (the sign-run entries alone are too few
     to fit 8 features); label y=1 if TP (2 ATR) is hit first or the vertical barrier ends with a positive directional
     return, y=0 otherwise (SL first / same-day both = SL / negative vertical); class-balanced logit so p=0.5 is the
     no-information point; features signed by trade direction (r1, r7, r30, funding z, taker ratio-0.5, Donchian
     distance to the breakout side in ATR); vol z = 20d vol vs trailing 180d.  Training events: t <= test_start - 5d -
     max(10d, V).  Selection of (C, p_min, V) per fold uses in-sample Sharpe with out-of-fold probabilities (3 blocks,
     15-day purge).  Meta size multiplier is set at each event and kept until the run ends or the next event.
     Leak rule: rejected if the mean over folds of the largest |coef| share exceeds 60 %.
 D9  H5: metrics as-of rule of A1.4 (create_time <= daily close - 5 min, max age 12 h, never interpolate); a day is
     "metrics_gap" if any used series is NaN.  Filter mode: fragile longs/shorts halve H1's long/short while the
     condition holds, the flush leg is ADDED (sum) to H1.  Standalone: three legs (fragile longs -> short, fragile shorts
     -> long, flush -> long), 0.5 * vol-scaled size, no pyramiding, OI exit and flush-low stop as written.  Value check:
     "result" = OOS Sharpe gain over raw H1 (filter) or OOS Sharpe (standalone); rejected if it changes sign AND the
     value version is better.  Untestable if any leg has < 20 independent events (>=5 d apart) summed over the OOS
     windows at each fold's selected parameters.
 D10 H6: depth window = folds whose NOMINAL training window (fold train_start .. test_start - 5D) contains >= 12 months
     of depth data (>= 2023-01-01); in-sample selection for H6 uses that nominal window.  Depth values = `depth` (base
     units) at +-1 % / +-2 %; day invalid if < 1,440 snapshots.  Vetoed entry = entry or flip blocked (distinct
     episodes), counted over the depth-window OOS folds at each fold's selected parameters.  Placebo uses the same
     per-fold parameters on the z-score of total depth; "captures >= 80 %" = placebo OOS Sharpe gain >= 0.8 x H6 gain.
 D11 Trials go to research/trials.duckdb (stages h1..h6, tf '1d-h' so existing 1d matrices are not disturbed).
     Amendment add-on to N: A1 +1, A2 +1, A3 +1, A4 +6 (disclosed runs) = 9.
 D12 S2 combination step runs only for components that pass individually AND the DSR gate.  S4 holdout rows are not
     evaluated (holdout never loaded; S2.6).
```

## Benchmarks (same folds)

| fold | OOS window | raw H1 Sharpe | B&H Sharpe | brayvb_v2 Sharpe | raw H1 ret | B&H ret | brayvb ret |
|---|---|---|---|---|---|---|---|
| 0 | 2023-03-10..2023-09-09 | -0.58 | 1.31 | 0.60 | -4.9% | 27.2% | 6.0% |
| 1 | 2023-09-10..2024-03-09 | 3.25 | 4.39 | 2.76 | 58.1% | 164.0% | 58.2% |
| 2 | 2024-03-10..2024-09-09 | -2.51 | -0.38 | -1.09 | -16.9% | -16.6% | -12.9% |
| 3 | 2024-09-10..2025-03-09 | 0.83 | 1.66 | 0.82 | 8.5% | 41.5% | 10.1% |
| 4 | 2025-03-10..2025-09-09 | 0.53 | 1.86 | 0.68 | 4.0% | 38.2% | 6.4% |
| 5 | 2025-09-10..2026-03-09 | 0.51 | -1.73 | -1.27 | 3.8% | -38.7% | -8.2% |

brayvb_v2 default (4h preset, BINANCE_PERP_TAKER + funding): 198 trades over the whole history. Raw H1 = H1 at median parameters {'sw': 40, 'tgt': 0.3, 'ss': 0.5, 'db': 0.125}.

## A1.4 gap flags (days flagged per WF test fold)

Metrics series (H5):

| fold | OOS days | metrics_gap days | share | status |
|---|---|---|---|---|
| 0 | 184 | 0 | 0.0% | ok |
| 1 | 182 | 0 | 0.0% | ok |
| 2 | 184 | 0 | 0.0% | ok |
| 3 | 181 | 0 | 0.0% | ok |
| 4 | 184 | 0 | 0.0% | ok |
| 5 | 181 | 0 | 0.0% | ok |

Depth series (H6; days before 2023-01-01 count as gap, folds outside the depth window are not used):

| fold | OOS days | depth_gap days | share | status |
|---|---|---|---|---|
| 0 | 184 | 1 | 0.5% | ok |
| 1 | 182 | 0 | 0.0% | ok |
| 2 | 184 | 2 | 1.1% | ok |
| 3 | 181 | 0 | 0.0% | ok |
| 4 | 184 | 0 | 0.0% | ok |
| 5 | 181 | 0 | 0.0% | ok |

## H1 - Volatility-managed multi-horizon TSMOM: REJECTED

Failing criterion: OOS Sharpe > 0 in only 3/6 folds; beats B&H Calmar in only 1/6 folds

| fold | OOS window | selected (IS-best) params | Sharpe | raw H1 | B&H | brayvb_v2 | return | max DD | Calmar | B&H Calmar | round trips |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2023-03-10..2023-09-09 | sw=40, tgt=0.2, ss=1.0, db=0.0 | -1.10 | -0.58 | 1.31 | 0.60 | -6.3% | 11.5% | -1.06 | 3.38 | 20 |
| 1 | 2023-09-10..2024-03-09 | sw=40, tgt=0.2, ss=0.5, db=0.0 | 3.25 | 3.25 | 4.39 | 2.76 | 36.3% | 5.7% | 15.14 | 38.11 | 13 |
| 2 | 2024-03-10..2024-09-09 | sw=60, tgt=0.2, ss=0.0, db=0.0 | -2.70 | -2.51 | -0.38 | -1.09 | -11.8% | 13.8% | -1.60 | -1.15 | 11 |
| 3 | 2024-09-10..2025-03-09 | sw=20, tgt=0.3, ss=0.0, db=0.0 | 0.84 | 0.83 | 1.66 | 0.82 | 8.4% | 12.0% | 1.47 | 4.24 | 8 |
| 4 | 2025-03-10..2025-09-09 | sw=20, tgt=0.2, ss=0.0, db=0.0 | 0.89 | 0.53 | 1.86 | 0.68 | 5.5% | 6.0% | 1.85 | 7.05 | 9 |
| 5 | 2025-09-10..2026-03-09 | sw=60, tgt=0.2, ss=0.0, db=0.0 | -1.15 | 0.51 | -1.73 | -1.27 | -4.6% | 7.3% | -1.24 | -1.26 | 5 |

- Median fold Sharpe -0.13 (raw H1 0.52, B&H 1.49, brayvb_v2 0.64); pooled OOS Sharpe 0.53 (IS mean 0.75), OOS return 22.8%, max DD 17.8%; B&H pooled Sharpe 1.08, max DD 49.6%; raw H1 pooled 0.73.
- DSR p (N=191,598, family var): **0.4055** (SR* = 0.67 annualised vs candidate 0.53); with the all-trials Sharpe variance (as stages.family_stats over the broad pool): 0.0003; with N = family size only (81): 0.6191; PSR vs 0: 0.828.
- PBO (CSCV, 16 blocks, 81 configs): **0.479** (gate < 0.15); SPA p vs B&H on the family: 1.000 (RC 0.959); stitched candidate alone vs B&H: 1.000.
- Round trips: 66 OOS, mean net -30.5 bp each (cost per round trip 14 bp); A2: IS 121 (min 40), per-fold OOS [20, 13, 11, 8, 9, 5] (min 8); funding paid 2.3%/yr on the OOS windows.

Falsification tests:

| criterion | result | met |
|---|---|---|
| OOS Sharpe>0 in >=4/6 folds | 3/6 | NO |
| OOS Calmar > B&H in >=4/6 folds | 1/6 | NO |

## H2 - Regime switch (ER): trend / mean reversion: REJECTED

Failing criterion: DSR-adjusted OOS Sharpe above H1 alone in only 0/6 folds

| fold | OOS window | selected (IS-best) params | Sharpe | raw H1 | B&H | brayvb_v2 | return | max DD | Calmar | B&H Calmar | round trips |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2023-03-10..2023-09-09 | n=10, theta=0.4, z=2.0, ss=0.0 | -0.52 | -0.58 | 1.31 | 0.60 | -2.6% | 5.2% | -1.00 | 3.38 | 7 |
| 1 | 2023-09-10..2024-03-09 | n=10, theta=0.4, z=1.5, ss=0.5 | 1.87 | 3.25 | 4.39 | 2.76 | 20.3% | 5.6% | 8.00 | 38.11 | 12 |
| 2 | 2024-03-10..2024-09-09 | n=10, theta=0.4, z=1.5, ss=1.0 | -1.81 | -2.51 | -0.38 | -1.09 | -10.6% | 11.5% | -1.73 | -1.15 | 13 |
| 3 | 2024-09-10..2025-03-09 | n=10, theta=0.2, z=1.5, ss=0.0 | 0.56 | 0.83 | 1.66 | 0.82 | 3.8% | 9.0% | 0.87 | 4.24 | 15 |
| 4 | 2025-03-10..2025-09-09 | n=25, theta=0.3, z=2.5, ss=1.0 | 1.40 | 0.53 | 1.86 | 0.68 | 8.3% | 6.2% | 2.78 | 7.05 | 5 |
| 5 | 2025-09-10..2026-03-09 | n=25, theta=0.3, z=2.0, ss=1.0 | 1.27 | 0.51 | -1.73 | -1.27 | 11.8% | 7.3% | 3.49 | -1.26 | 11 |

- Median fold Sharpe 0.91 (raw H1 0.52, B&H 1.49, brayvb_v2 0.64); pooled OOS Sharpe 0.67 (IS mean 1.54), OOS return 31.7%, max DD 13.2%; B&H pooled Sharpe 1.08, max DD 49.6%; raw H1 pooled 0.73.
- DSR p (N=191,598, family var): **0.0021** (SR* = 2.25 annualised vs candidate 0.67); with the all-trials Sharpe variance (as stages.family_stats over the broad pool): 0.0005; with N = family size only (81): 0.1600; PSR vs 0: 0.888.
- PBO (CSCV, 16 blocks, 81 configs): **0.090** (gate < 0.15); SPA p vs B&H on the family: 1.000 (RC 0.933); stitched candidate alone vs B&H: 1.000.
- Round trips: 63 OOS, mean net 73.0 bp each (cost per round trip 14 bp); A2: IS 98 (min 40), per-fold OOS [7, 12, 13, 15, 5, 11] (min 8); funding paid 1.3%/yr on the OOS windows.
- DSR-adjusted fold Sharpe (fold Sharpe - family expected max): -2.91, -0.12, -3.72, -1.86, -1.54, -1.08; haircuts 2.40, 1.99, 1.91, 2.42, 2.93, 2.34; raw H1 -0.58, 3.25, -2.51, 0.83, 0.53, 0.51.
- MR-leg OOS expectancy 32.2 bp over 24 trades.

Falsification tests:

| criterion | result | met |
|---|---|---|
| DSR-adj OOS Sharpe > H1 alone in >=4/6 folds | 0/6 | NO |
| MR-leg OOS net expectancy > 0 | 32.2 bp (24 trades) | yes |

## H3 - Funding-rate crowding filter on H1: PASS

| fold | OOS window | selected (IS-best) params | Sharpe | raw H1 | B&H | brayvb_v2 | return | max DD | Calmar | B&H Calmar | round trips |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2023-03-10..2023-09-09 | w=60, tau=1.5, h=0.0, hold=5 | 0.31 | -0.58 | 1.31 | 0.60 | 2.2% | 17.0% | 0.27 | 3.38 | 21 |
| 1 | 2023-09-10..2024-03-09 | w=30, tau=1.5, h=0.0, hold=3 | 4.64 | 3.25 | 4.39 | 2.76 | 84.9% | 7.7% | 31.39 | 38.11 | 31 |
| 2 | 2024-03-10..2024-09-09 | w=30, tau=1.5, h=0.0, hold=5 | -1.73 | -2.51 | -0.38 | -1.09 | -14.1% | 19.6% | -1.32 | -1.15 | 31 |
| 3 | 2024-09-10..2025-03-09 | w=60, tau=2.0, h=0.0, hold=1 | 0.32 | 0.83 | 1.66 | 0.82 | 2.3% | 13.9% | 0.35 | 4.24 | 23 |
| 4 | 2025-03-10..2025-09-09 | w=30, tau=1.5, h=0.0, hold=7 | 0.78 | 0.53 | 1.86 | 0.68 | 7.5% | 10.5% | 1.48 | 7.05 | 15 |
| 5 | 2025-09-10..2026-03-09 | w=60, tau=1.5, h=0.5, hold=5 | -0.03 | 0.51 | -1.73 | -1.27 | -1.0% | 10.8% | -0.18 | -1.26 | 16 |

- Median fold Sharpe 0.32 (raw H1 0.52, B&H 1.49, brayvb_v2 0.64); pooled OOS Sharpe 1.00 (IS mean 1.33), OOS return 77.0%, max DD 23.0%; B&H pooled Sharpe 1.08, max DD 49.6%; raw H1 pooled 0.73.
- DSR p (N=191,598, family var): **0.8186** (SR* = 0.49 annualised vs candidate 1.00); with the all-trials Sharpe variance (as stages.family_stats over the broad pool): 0.0040; with N = family size only (96): 0.9037; PSR vs 0: 0.964.
- PBO (CSCV, 16 blocks, 96 configs): **0.625** (gate < 0.15); SPA p vs B&H on the family: 1.000 (RC 0.962); stitched candidate alone vs B&H: 1.000.
- Round trips: 137 OOS, mean net -33.5 bp each (cost per round trip 14 bp); A2: IS 263 (min 40), per-fold OOS [21, 31, 31, 23, 15, 16] (min 8); funding paid 2.6%/yr on the OOS windows.
- Independent extremes summed over OOS windows at selected params: 69. In the OOS windows at every grid point: w30/tau1.5: 80; w30/tau2.0: 41; w30/tau2.5: 22; w30/tau3.0: 18; w60/tau1.5: 73; w60/tau2.0: 36; w60/tau2.5: 23; w60/tau3.0: 16; w90/tau1.5: 66; w90/tau2.0: 32; w90/tau2.5: 23; w90/tau3.0: 15

Falsification tests:

| criterion | result | met |
|---|---|---|
| independent extremes >= 20 | 69 | yes |
| beats raw H1 OOS Sharpe in >=4/6 folds | 4/6 | yes |

## H4 - Meta-labeling on H1 (L2 logit): REJECTED

Failing criterion: beats unfiltered H1 OOS Sharpe in only 3/6 folds; OOS precision of taken trades 35.7% (<= 52%) on 28 taken events

| fold | OOS window | selected (IS-best) params | Sharpe | raw H1 | B&H | brayvb_v2 | return | max DD | Calmar | B&H Calmar | round trips |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2023-03-10..2023-09-09 | C=1.0, p_min=0.6, V=5 | 0.05 | -0.58 | 1.31 | 0.60 | -0.0% | 8.8% | -0.01 | 3.38 | 8 |
| 1 | 2023-09-10..2024-03-09 | C=0.01, p_min=0.55, V=5 | -1.97 | 3.25 | 4.39 | 2.76 | -0.1% | 0.1% | -2.00 | 38.11 | 2 |
| 2 | 2024-03-10..2024-09-09 | C=0.01, p_min=0.6, V=5 | 0.00 | -2.51 | -0.38 | -1.09 | 0.0% | 0.0% | 0.00 | -1.15 | 0 |
| 3 | 2024-09-10..2025-03-09 | C=1.0, p_min=0.6, V=15 | -1.35 | 0.83 | 1.66 | 0.82 | -3.7% | 4.9% | -1.48 | 4.24 | 6 |
| 4 | 2025-03-10..2025-09-09 | C=0.1, p_min=0.6, V=10 | 0.00 | 0.53 | 1.86 | 0.68 | 0.0% | 0.0% | 0.00 | 7.05 | 0 |
| 5 | 2025-09-10..2026-03-09 | C=0.01, p_min=0.55, V=5 | 0.69 | 0.51 | -1.73 | -1.27 | 0.2% | 0.4% | 1.02 | -1.26 | 4 |

- Median fold Sharpe 0.00 (raw H1 0.52, B&H 1.49, brayvb_v2 0.64); pooled OOS Sharpe -0.21 (IS mean 1.23), OOS return -3.6%, max DD 12.0%; B&H pooled Sharpe 1.08, max DD 49.6%; raw H1 pooled 0.73.
- DSR p (N=191,598, family var): **0.0209** (SR* = 0.98 annualised vs candidate -0.21); with the all-trials Sharpe variance (as stages.family_stats over the broad pool): 0.0000; with N = family size only (27): 0.1340; PSR vs 0: 0.362.
- PBO (CSCV, 16 blocks, 24 configs): **0.611** (gate < 0.15); SPA p vs B&H on the family: 1.000 (RC 0.974); stitched candidate alone vs B&H: 1.000.
- Round trips: 20 OOS, mean net -146.9 bp each (cost per round trip 14 bp); A2: IS 28 (min 40), per-fold OOS [8, 2, 0, 6, 0, 4] (min 8); funding paid 0.2%/yr on the OOS windows.
- Taken OOS events 28, precision 35.7% vs base rate of all OOS events 39.9%; max |coef| share per fold [0.2583874663324852, 0.23941237697367024, 0.28998526189186336, 0.3334217121168339, 0.2811649459240444, 0.3503477890154925].

Falsification tests:

| criterion | result | met |
|---|---|---|
| OOS Sharpe > unfiltered H1 in >=4/6 folds | 3/6 | NO |
| OOS precision on taken trades > 52% | 35.7% (28 taken; base rate of all events 39.9%) | NO |
| no feature > 60% of |coef| mass (mean of per-fold max) | 29%; per fold 26%, 24%, 29%, 33%, 28%, 35% | yes |

## H5-filter - OI/price divergence filter on H1 (+ flush leg added): UNTESTABLE

Failing criterion: legs with < 20 independent events: fragile longs (2), fragile shorts (3)

| fold | OOS window | selected (IS-best) params | Sharpe | raw H1 | B&H | brayvb_v2 | return | max DD | Calmar | B&H Calmar | round trips |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2023-03-10..2023-09-09 | k=3, w=30, tau=2.5, hold=3 | 0.21 | -0.58 | 1.31 | 0.60 | 1.1% | 13.5% | 0.16 | 3.38 | 22 |
| 1 | 2023-09-10..2024-03-09 | k=1, w=60, tau=1.5, hold=3 | 3.59 | 3.25 | 4.39 | 2.76 | 67.6% | 8.0% | 22.70 | 38.11 | 13 |
| 2 | 2024-03-10..2024-09-09 | k=3, w=60, tau=2.0, hold=7 | -1.64 | -2.51 | -0.38 | -1.09 | -12.1% | 16.6% | -1.36 | -1.15 | 25 |
| 3 | 2024-09-10..2025-03-09 | k=1, w=30, tau=1.5, hold=7 | 0.90 | 0.83 | 1.66 | 0.82 | 10.1% | 12.9% | 1.66 | 4.24 | 20 |
| 4 | 2025-03-10..2025-09-09 | k=1, w=30, tau=1.5, hold=5 | 0.82 | 0.53 | 1.86 | 0.68 | 6.4% | 7.3% | 1.80 | 7.05 | 17 |
| 5 | 2025-09-10..2026-03-09 | k=1, w=30, tau=1.5, hold=5 | -0.51 | 0.51 | -1.73 | -1.27 | -6.1% | 14.1% | -0.84 | -1.26 | 9 |

- Median fold Sharpe 0.52 (raw H1 0.52, B&H 1.49, brayvb_v2 0.64); pooled OOS Sharpe 0.88 (IS mean 1.05), OOS return 64.0%, max DD 20.1%; B&H pooled Sharpe 1.08, max DD 49.6%; raw H1 pooled 0.73.
- DSR p (N=191,598, family var): **0.7576** (SR* = 0.48 annualised vs candidate 0.88); with the all-trials Sharpe variance (as stages.family_stats over the broad pool): 0.0025; with N = family size only (96): 0.8581; PSR vs 0: 0.938.
- PBO (CSCV, 16 blocks, 96 configs): **0.242** (gate < 0.15); SPA p vs B&H on the family: 1.000 (RC 0.929); stitched candidate alone vs B&H: 1.000.
- Round trips: 106 OOS, mean net -25.6 bp each (cost per round trip 14 bp); A2: IS 196 (min 40), per-fold OOS [22, 13, 25, 20, 17, 9] (min 8); funding paid 3.2%/yr on the OOS windows.
- Independent events per leg (OOS windows, selected params): {'fragile longs': 2, 'fragile shorts': 3, 'flush': 31}. Value-variant OOS Sharpe 0.79 (pos folds 4/6).

Falsification tests:

| criterion | result | met |
|---|---|---|
| beats raw H1 OOS Sharpe in >=4/6 folds | 5/6 | yes |
| value-variant check (sign reversal AND value better => reject) | result OI +0.15 vs OI-value +0.06 | yes |
| independent events per leg >= 20 (OOS windows, selected params) | fragile longs: 2, fragile shorts: 3, flush: 31 | NO |

## H5-standalone - OI/price divergence standalone engine: UNTESTABLE

Failing criterion: legs with < 20 independent events: fragile longs (2), fragile shorts (2) (the individual criteria would also fail: net expectancy 45.3 bp per round trip < 50 bp (28 trades))

| fold | OOS window | selected (IS-best) params | Sharpe | raw H1 | B&H | brayvb_v2 | return | max DD | Calmar | B&H Calmar | round trips |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2023-03-10..2023-09-09 | k=3, w=30, tau=3.0, hold=1 | 1.48 | -0.58 | 1.31 | 0.60 | 0.2% | 0.0% | 10.98 | 3.38 | 1 |
| 1 | 2023-09-10..2024-03-09 | k=1, w=30, tau=2.0, hold=1 | 2.09 | 3.25 | 4.39 | 2.76 | 1.8% | 0.0% | 166.83 | 38.11 | 3 |
| 2 | 2024-03-10..2024-09-09 | k=3, w=90, tau=2.0, hold=7 | 2.46 | -2.51 | -0.38 | -1.09 | 8.9% | 1.5% | 11.97 | -1.15 | 5 |
| 3 | 2024-09-10..2025-03-09 | k=1, w=30, tau=1.5, hold=7 | 0.05 | 0.83 | 1.66 | 0.82 | 0.0% | 7.0% | 0.01 | 4.24 | 7 |
| 4 | 2025-03-10..2025-09-09 | k=1, w=30, tau=1.5, hold=5 | 0.18 | 0.53 | 1.86 | 0.68 | 0.4% | 2.5% | 0.31 | 7.05 | 5 |
| 5 | 2025-09-10..2026-03-09 | k=1, w=30, tau=1.5, hold=5 | -2.18 | 0.51 | -1.73 | -1.27 | -9.1% | 10.0% | -1.75 | -1.26 | 7 |

- Median fold Sharpe 0.83 (raw H1 0.52, B&H 1.49, brayvb_v2 0.64); pooled OOS Sharpe 0.11 (IS mean 2.27), OOS return 1.4%, max DD 15.1%; B&H pooled Sharpe 1.08, max DD 49.6%; raw H1 pooled 0.73.
- DSR p (N=191,598, family var): **0.0047** (SR* = 1.61 annualised vs candidate 0.11); with the all-trials Sharpe variance (as stages.family_stats over the broad pool): 0.0000; with N = family size only (96): 0.0871; PSR vs 0: 0.574.
- PBO (CSCV, 16 blocks, 96 configs): **0.567** (gate < 0.15); SPA p vs B&H on the family: 1.000 (RC 0.953); stitched candidate alone vs B&H: 1.000.
- Round trips: 28 OOS, mean net 45.3 bp each (cost per round trip 14 bp); A2: IS 59 (min 40), per-fold OOS [1, 3, 5, 7, 5, 7] (min 8); funding paid 0.2%/yr on the OOS windows.
- Independent events per leg (OOS windows, selected params): {'fragile longs': 2, 'fragile shorts': 2, 'flush': 27}. Value-variant OOS Sharpe -0.03 (pos folds 3/6).

Falsification tests:

| criterion | result | met |
|---|---|---|
| OOS Sharpe > 0 in >=4/6 folds | 5/6 | yes |
| net expectancy per round trip >= 50 bp | 45.3 bp (28 trades) | NO |
| value-variant check (sign reversal AND value better => reject) | result OI +0.11 vs OI-value -0.03 | yes |
| independent events per leg >= 20 (OOS windows, selected params) | fragile longs: 2, fragile shorts: 2, flush: 27 | NO |

## H6 - Book-depth imbalance entry filter on H1 (depth window): REJECTED

Failing criterion: beats raw H1 OOS Sharpe in only 2/4 folds (needs 3)

| fold | OOS window | selected (IS-best) params | Sharpe | raw H1 | B&H | brayvb_v2 | return | max DD | Calmar | B&H Calmar | round trips |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 2024-03-10..2024-09-09 | b=1, w=60, gamma=1.5 | -2.40 | -2.51 | -0.38 | -1.09 | -16.2% | 19.2% | -1.53 | -1.15 | 22 |
| 3 | 2024-09-10..2025-03-09 | b=1, w=40, gamma=1.5 | 0.88 | 0.83 | 1.66 | 0.82 | 9.1% | 12.2% | 1.57 | 4.24 | 15 |
| 4 | 2025-03-10..2025-09-09 | b=2, w=60, gamma=0.5 | 0.35 | 0.53 | 1.86 | 0.68 | 2.2% | 8.6% | 0.51 | 7.05 | 13 |
| 5 | 2025-09-10..2026-03-09 | b=2, w=60, gamma=0.5 | 0.26 | 0.51 | -1.73 | -1.27 | 1.5% | 10.2% | 0.30 | -1.26 | 9 |

- Median fold Sharpe 0.30 (raw H1 0.52, B&H 0.64, brayvb_v2 -0.20); pooled OOS Sharpe -0.06 (IS mean 1.00), OOS return -5.1%, max DD 24.3%; B&H pooled Sharpe 0.24, max DD 49.6%; raw H1 pooled 0.02.
- DSR p (N=191,598, family var): **0.3320** (SR* = 0.25 annualised vs candidate -0.06); with the all-trials Sharpe variance (as stages.family_stats over the broad pool): 0.0002; with N = family size only (18): 0.4106; PSR vs 0: 0.467.
- PBO (CSCV, 16 blocks, 18 configs): **0.701** (gate < 0.15); SPA p vs B&H on the family: 1.000 (RC 0.629); stitched candidate alone vs B&H: 1.000.
- Round trips: 59 OOS, mean net -96.6 bp each (cost per round trip 14 bp); A2: IS 171 (min 40), per-fold OOS [22, 15, 13, 9] (min 8); funding paid 2.2%/yr on the OOS windows.
- Depth window: 4 folds kept [(2, 14.1), (3, 18.1), (4, 18.0), (5, 18.0)] (id, months of depth training data), dropped [(0, 2.1), (1, 8.1)]; 24 OOS months; SE factor x1.22; vetoed entry episodes 23; placebo OOS Sharpe -0.00.

Falsification tests:

| criterion | result | met |
|---|---|---|
| beats raw H1 (same depth window) OOS Sharpe in >= 3/4 folds | 2/4 | NO |
| placebo captures < 80% of H6 Sharpe gain | H6 gain -0.08, placebo gain -0.02 (H6 gain <= 0) | yes |
| vetoed entries >= 20 in depth OOS | 23 | yes |

## S2 combination step

Not run: no hypothesis is eligible (S2.1 requires an individual WF pass AND the family DSR gate p > 0.95). `research/runs/combinations.json` is therefore not written and no combination trials are added to N.

## S4 success criteria - best candidate vs brayvb_v2 default and B&H

Best candidate = highest DSR p among the seven variants: **H3** (NOT eligible - shown for transparency only). The holdout was never unlocked: holdout rows cannot be evaluated here (S2.6, only a final model passing everything earlier may run on it).

| criterion | H3 (candidate) | brayvb_v2 default | B&H | met by candidate |
|---|---|---|---|---|
| Deflated Sharpe p >= 0.95 (N per S2.4) | 0.8186 (N=191,598) | n/a | n/a | NO |
| PBO < 0.15 | 0.625 | n/a (single strategy) | n/a | NO |
| WF OOS Sharpe > 0 in >= 4/6 folds | 4/6 | 4/6 | 4/6 | YES |
| ... and above brayvb_v2 default in >= 4/6 folds | 3/6 | - | - | NO |
| OOS/IS Sharpe degradation <= 50% | IS 1.33 -> OOS 1.00 (25%) | n/a | n/a | YES |
| Pooled OOS Sharpe / return / max DD | 1.00 / 77.0% / 23.0% | 0.72 / 57.0% / 21.8% | 1.08 / 236.1% / 49.6% | info |
| Holdout net Sharpe >= B&H + 0.3 and >= best catalog | not evaluated (holdout locked) | - | - | n/a |
| Holdout MaxDD <= 0.6 x B&H MaxDD | not evaluated | - | - | n/a |
| Holdout inside 5-95% bootstrap band of WF OOS Sharpe | band [-0.07, 2.03] prepared; holdout not evaluated | - | - | n/a |

**Verdict:** no candidate meets the pre-holdout S4 criteria. The custom model is NOT better than the catalog strategy; the catalog strategy is preferred and nothing goes to the holdout or to live.

_Runtime 6 s._