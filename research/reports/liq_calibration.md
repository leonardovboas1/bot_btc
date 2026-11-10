# Liquidation model calibration (gross openings, Gate-observed liquidations)

Generated 2026-11-10 by `research/calibrate_liq.py` (+ `research/validate_liq_model.py`). Fitted params: `btclab/liquidity/liq_calib.json`.

## Model change
new notional per bar = `scale * (max(dOI,0)*close + kappa*turnover)`, turnover = Binance 5m volume*close (taker-volume proxy;
Gate/OKX use Binance turnover x venue-OI/Binance-OI share). Leverage mix, MMR=0.4%, consumed-on-touch and shrink-on-OI-drop unchanged.
`liq_columns` kept backward compatible (new kwargs default to legacy behavior). Warm-up now max(30 d, 2 x window).
Venues: binance (default), gate, bybit, okx (rubik 1H OI history, <=~50 d, coin units); each venue's OI is in coin units x close = USD.
Endpoint: `calibrated=true` default (`calibrated=false` = legacy net-dOI model).

## Fitted parameters (4 free, BTCUSDT on Gate)
| parameter | value |
|---|---|
| kappa (turnover to gross opening) | 0.02 |
| half-life | 720 h (grid edge: 6/24/72/240/720 h; flat above 240 h) |
| leverage preset | 3/5/10/25/50/100x = 5/10/25/30/22/8 % (preset chosen from 3 candidates) |
| global scale (closed form, train totals) | 0.0414 |
Grid: 6 kappa x 5 half-lives x 3 presets = 90 fits on train only (objective: Pearson of log1p USD per 1h bar); `liq_calib_grid.csv`.
The grid is very flat (best 0.652 vs median ~0.64), so individual parameter values are weakly identified.
Scale is applied to all venues (only Gate has observed liquidations; the recorder has just hours of data, unusable).

## Fit / eval (model consumed USD per bar, long+short, vs Gate observed)
Train = first 120 d (scored after 30 d warm-up), eval = last 60 d (state carried, params frozen).
| | corr 5m (log) | Spearman 5m | corr 1h (log) | Spearman 1h | ratio of totals |
|---|---|---|---|---|---|
| calibrated fit | 0.544 | 0.501 | 0.653 | 0.745 | 1.00 |
| calibrated eval | 0.548 | 0.503 | 0.657 | 0.753 | 1.09 |
| legacy (scale refit) fit | 0.519 | 0.473 | 0.636 | 0.732 | 1.00 |
| legacy (scale refit) eval | 0.524 | 0.475 | 0.634 | 0.734 | 1.09 |
Unscaled legacy model consumes 24x (fit) / 26x (eval) the observed Gate liquidation USD.

## Validation lift (research/validate_liq_model.py, 8 symbols, same 30 d warm-up, "any touch", spike-weighted)
| model | vol-matched lift | placebo lift | spearman (mean) |
|---|---|---|---|
| legacy | 2.93 | 5.62 | 0.366 |
| calibrated | 2.67 | 5.15 | 0.367 |
(Earlier report had 2.2 with a 5 d warm-up.) Params fitted on BTC only, applied to all symbols.

## BTC active-level totals vs OI (Binance OI now US$7.43bn)
| range | legacy | calibrated (scale 0.041) |
|---|---|---|
| 1w | 1.59bn (21% of OI) | 0.087bn (1.2%) |
| 1m | 1.52bn | 0.096bn |
| 6m | 2.12bn | 0.153bn |
Cold compute 6m: 0.9 s legacy / 1.0 s calibrated (binance only). Multi-venue incl. Bybit REST paging is slow cold (~30 s), cached 5 min.

## Honest verdict
- Gross-opening term improves the match to observed liquidations modestly (1h log-corr 0.634 to 0.657 eval; stable out of sample) and the
  total is matched by construction, but the improvement is small and the grid is flat.
- The key finding runs against the goal: observed Gate liquidations are only ~4% of what modeled levels would consume. Matching observed
  liquidated USD therefore SHRINKS active totals (to ~1-2% of OI), the opposite of the Coinglass-like "more USD" target. The
  gap vs other platforms reflects a different quantity (position notional resting at liq prices vs USD actually liquidated).
  If the display should show exposure, use `calibrated=false` (or kappa-only without scale); calibrated amounts are an estimate of liquidatable-and-liquidated USD.
- Validation lift did not improve (2.93 to 2.67, a drop within noise): the new params add USD fidelity, not better level locations.
- Caveats: one venue, one symbol fit, Gate's own OI used as model input in the fit, turnover is a volume proxy (no taker split), 1h bars net
  openings inside the hour, OKX history short.
