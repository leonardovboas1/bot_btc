# BTC Signal Lab — Pre-registration of the custom strategy (v1, frozen 2026-11-10)

**Status:** frozen before any backtest has been run; nobody has looked at `data/` or `research/runs/`. This text is never edited after the search starts. Changes go into the dated *Amendments* section with a reason, and each amendment counts as extra trials in the Deflated Sharpe (DSR) `N`.

**Scope.** Target: Hyperliquid BTC perpetual (taker 0.045%/side, hourly funding). Research data: Binance perp OHLCV and funding from 2019-09 (spot earlier only for warm-up). Daily and 4h bars; sizing changes at most once per 4h bar. Walk-forward (WF) = the 40-month, 6-fold purged/embargoed scheme of `plan.md` §C; holdout = the untouched last 8 months.

## 1. Hypotheses

Conventions: signal on bar close, fill next open; shorts allowed everywhere, with a `short_scale` ∈ {0, 0.5, 1} so the data decides whether shorting adds anything. Parameter ranges fixed here are the only values the search may visit.

### H1 — Volatility-managed multi-horizon time-series momentum (anchor)

*Rationale.* BTC has no cash-flow anchor; price is set by flows (late retail, ETF/treasury buying, leverage cascades) that arrive slowly and overshoot. The other side is the holder who sells into strength early and the leveraged trader liquidated into weakness. The edge persists because the payoff (many small whipsaw losses, few large gains) is behaviourally hard to hold, and BTC momentum is documented across the post-2015 sample (Liu & Tsyvinski 2021; Zarattini et al. 2025).

*Rules.* Lookbacks fixed, not searched: L ∈ {10, 20, 40, 80, 160} days. `s_L = sign(close_t − close_{t−L})`, `S = mean(s_L)` ∈ [−1, 1]. Position = `S · σ_target / σ_realized` (EWMA std of daily log returns). Exit when S crosses zero; no stops.

*Parameters (4).* σ window 20–60 d; σ_target 20–40% annualized; short_scale; rebalance deadband 0–0.25 on |ΔS|.

*Data.* Daily OHLCV.

*Falsification.* Rejected if WF OOS net Sharpe is not > 0 in ≥ 4/6 folds, **or** it does not beat buy-and-hold (B&H) on OOS Calmar in ≥ 4/6 folds.

### H2 — Regime switching: trend when trending, mean reversion in ranges

*Rationale.* In ranges the other side of a breakout is the market maker and the stop-hunt: breakouts fail and price returns to value. In trends the other side is H1's. The switch has an edge only if the detector is slower than the whipsaw it avoids, so it is causal and unsmoothed.

*Rules.* Kaufman efficiency ratio `ER = |close_t − close_{t−n}| / Σ|Δclose|` on daily bars. `ER > θ` ⇒ run H1 at its median parameters (not re-tuned). `ER ≤ θ` ⇒ mean reversion: long when close < SMA(20) − z·σ₂₀ and RSI(3) < 20, short mirror; exit at SMA(20) touch or after 5 bars; size 0.5 × H1's vol-scaled size. No position within 2 bars of a regime flip.

*Parameters (4).* n 10–40 d; θ 0.20–0.40; z 1.5–2.5; short_scale.

*Data.* Daily OHLCV.

*Falsification.* Rejected if DSR-adjusted OOS Sharpe is not above H1 alone in ≥ 4/6 folds (the switch must add value), **or** if the mean-reversion leg's OOS net expectancy is ≤ 0.

### H3 — Funding-rate crowding as a filter on H1

*Rationale.* Extreme positive funding means crowded leveraged longs paying to stay long; extreme negative funding after a flush means forced sellers are exhausted. The other side is the structurally long-biased retail perp trader. The edge persists because liquidation mechanics are mechanical, and the carry trader who collects funding hedges in spot and takes no direction.

*Rules.* `z_f` = rolling z-score of the 8h funding print. If `z_f > τ` and S > 0, multiply the long by `h`; if `z_f < −τ` and S < 0, multiply the short by `h`. Contrarian add-on: if `z_f < −τ` and close is above its 5-day low, open a 0.5-size long for `hold` days.

*Parameters (4).* z window 30–90 d; τ 1.5–3.0; h ∈ {0, 0.5}; hold 1–7 d.

*Data.* Binance funding 2019-09 onward (the only derivatives series with ≥ 4 years); daily OHLCV.

*Falsification.* Rejected if it does not beat raw H1 on OOS Sharpe in ≥ 4/6 folds. Declared **untestable** (neither pass nor fail) if the WF window holds < 20 independent extremes (|z_f| > τ, ≥ 5 days apart).

### H4 — Meta-labeling on the primary signal

*Rationale.* Simple rules are right about direction more often than about *when*. A secondary model that skips primary signals likely to pay fees for nothing (high vol, exhaustion, crowding) adds cost avoidance, not new alpha. It persists because it is trained on the strategy's own trade outcomes, not on the price series everybody fits.

*Rules.* Primary = H1 (later, the top single survivor by DSR). Label = triple-barrier outcome per primary trade: TP 2 ATR(14), SL 1 ATR(14), vertical barrier 10 d. Secondary = L2 logistic regression on 8 fixed features: 20d vol z-score, ER(20), funding z (30 d), 1d/7d/30d returns, distance to 20d Donchian high/low in ATR, 7d taker-buy ratio. Trained **only** on each fold's training window (purge 10 d, embargo 5 d), features standardized with training statistics. Take the trade if `p > p_min`; size = H1 size × min(1, (p − 0.5)/0.25).

*Parameters (3).* C ∈ {0.01, 0.1, 1}; p_min 0.50–0.60; vertical barrier 5–15 d.

*Data.* Daily OHLCV, funding, taker-buy volume.

*Falsification.* Rejected if OOS Sharpe does not beat the unfiltered primary in ≥ 4/6 folds, **or** OOS precision on taken trades ≤ 52%, **or** one feature carries > 60% of |coefficient| mass (treated as a leak).

### Not registered

Volume-profile/liquidity nodes (aggTrades only from 2022), OI, long/short ratio, liquidations (30-day history). They may become v2 hypotheses once ≥ 3 years of self-recorded data exist, never v1 components.

## 2. The "best parts" protocol

1. **Eligibility.** A component may be combined only if it individually passed the WF stage (OOS Sharpe > 0 in ≥ 4/6 folds) *and* the family DSR gate (p > 0.95). Component types: direction engine (H1, H2, catalog survivor), filter/overlay (H3, H4, regime filter), exit rule.
2. **Maximum 3 components:** at most one engine, one filter, one exit. Engines stack only through the ensemble operators below.
3. **Operators, fixed now:** (a) AND-gate filter on entry; (b) equal-weight vote, position = mean of vol-scaled positions; (c) inverse-vol risk parity across component P&L. Weights are never optimized. Engines join an ensemble only if pairwise OOS-return correlation < 0.5.
4. **Trial accounting.** The combination set is enumerated and written to `research/runs/combinations.json` *before* any combination runs, capped at 200 (if larger, keep the 200 lowest-correlation ones). Each combination is one row in the DuckDB `trials` table. DSR `N` = all S0–S4 trials + combinations + amendments.
5. **Selection.** Highest WF OOS DSR wins; ties go to fewer parameters. No component is re-tuned after combination.
6. **Holdout.** The final model only (plus best catalog strategy and B&H as benchmarks) runs once under `HOLDOUT_UNLOCK=1`. No re-tuning afterwards. If §4 fails, the report says so and the custom model does not go live.

## 3. Position sizing and risk (any final model)

- **Vol targeting:** notional = equity × min(σ_target/σ_realized, L_max); σ_realized = EWMA(30 d) daily log returns, updated each 4h bar.
- **Max leverage:** L_max = 2.0×, hard cap in simulator and live risk layer.
- **Turnover guard:** rebalance only when target differs from current by > 10% of equity. Net expectancy per round trip must be ≥ 4× round-trip cost (2 × 0.045% + 2 × 2 bp ≈ 13 bp ⇒ ≥ 50 bp), else rejected regardless of Sharpe.
- **Funding:** hourly at the Hyperliquid rate live; in backtest Binance 8h funding spread over 8 hours. Paying > 15%/yr in funding during WF ⇒ rejected as carry bleed.
- **Kill switch:** 25% equity drawdown from peak ⇒ flatten and stop; re-entry only after a written post-mortem and an amendment. Soft brake: at 15% drawdown halve σ_target until a new equity high.
- **Daily loss limit:** 5% of equity in a UTC day ⇒ flat until next daily close.

## 4. Success criteria (decided now)

The custom model is **"better" than the best catalog strategy** only if *all* hold:

| Criterion | Threshold |
|---|---|
| Deflated Sharpe (N per §2.4) | p ≥ 0.95 |
| PBO (CSCV, 16 blocks) | < 0.15 (stricter than the 0.20 gate; combinations add freedom) |
| WF OOS Sharpe | > 0 in ≥ 4/6 folds **and** above the best catalog strategy in ≥ 4/6 folds |
| OOS/IS Sharpe degradation | ≤ 50% |
| Holdout net Sharpe | ≥ B&H + 0.3 **and** ≥ best catalog strategy |
| Holdout MaxDD | ≤ 0.6 × B&H MaxDD |
| Holdout consistency | inside the 5–95% block-bootstrap band of WF OOS Sharpe |

If B&H is negative in the holdout, the Sharpe row becomes "custom > 0 and > B&H". Honesty note: 8 months cannot separate a Sharpe of 1.0 from 0.5 (standard error ≈ 1.2), so the holdout is a direction-and-consistency check that can only *refute*; the claim of superiority rests on DSR and PBO. Anything else is reported as "not better" and the catalog strategy is preferred.

## 5. Order of work once results arrive

1. Record this file's hash and the DuckDB trial count `N` in the first results report.
2. Run H1–H4 exactly as written through WF and family gates; report pass/fail/untestable per criterion before anything else.
3. List catalog survivors; compute the pairwise OOS correlation matrix with the passing hypotheses.
4. Enumerate combinations per §2, write `combinations.json`, run and log them.
5. Select per §2.5; run the top-50 gauntlet (SPA, bootstrap, stability, regimes) on the final model and the best catalog strategy.
6. Unlock the holdout once; run final model, best catalog strategy, B&H; apply §4; write the verdict.
7. Only on a §4 pass: Pine export for TradingView verification, then ≥ 8 weeks of Hyperliquid paper trading under §3 before real capital.

## Amendments

*(none)*

## Amendment 1 (2026-11-10, before any backtest result)

**Status at the time of writing.** No backtest has run, `research/runs/` has not been opened and no strategy result exists. The only inspection performed for this amendment was of column names, dtypes and date coverage of the files listed in A1.1 (parquet schema and the time index); no series was joined to returns. The `*(none)*` placeholder above is superseded by this section but left in place so the original text is byte-identical to the frozen commit. Per the header rule this amendment counts as extra trials in DSR `N` (A1.3).

### A1.1 What changed, and why an amendment before results is legitimate

*What changed.* The "Not registered" paragraph excluded open interest (OI), long/short ratios and liquidity data for one reason only: the Binance API serves 30 days of history. That reason is gone. Free archival history exists:

| Source | File / location | Columns (dtype) | Coverage | Cadence |
|---|---|---|---|---|
| Binance UM `metrics` archive | `data/raw/um_futures/metrics_archive_5m.parquet` | `sum_open_interest`, `sum_open_interest_value`, `count_toptrader_long_short_ratio` (top-trader *accounts*), `sum_toptrader_long_short_ratio` (top-trader *positions*), `count_long_short_ratio` (global accounts), `sum_taker_long_short_vol_ratio`; all `double`; index `create_time` (UTC, ns) | 2021-01-01 00:00 to 2026-11-09 23:55 UTC, 615,431 rows | 5 min; 601 bars missing in 152 runs, longest gap 10 h 30 min |
| Binance UM `bookDepth` daily archives | being downloaded to `data/raw/um_futures/` (not yet on disk at the time of writing) | cumulative bid/ask depth at ±1, 2, 3, 4, 5 % from mid per snapshot | 2023-01-01 to present | ~30 s snapshots |
| Funding | `data/raw/um_futures/funding_rate.parquet` (`fundingRate`, `markPrice`; index `funding_time`) | `double` | 2019-09-10 to 2026-11-10, 7,855 prints, no gaps | 8 h |
| Perp klines | `klines_{15m,1h,4h,1d}.parquet` (`open, high, low, close, volume, quote_volume, trades, taker_buy_base, taker_buy_quote`; index `open_time`) | `double` | 2019-09-08 to 2026-11-10, no gaps at 1d/4h | 15m / 1h / 4h / 1d |

The API pollers (`openInterestHist_*.parquet`, `*LongShort*Ratio_*.parquet`, `takerlongshortRatio_*.parquet`, from 2026-10-12) are **not** backtest inputs; the archive is the sole research source for these series, and the pollers serve live trading only. No stitching of the two sources in research.

*Why this is legitimate.* Pre-registration protects one thing: that hypotheses and their falsification criteria are fixed before the data can speak. This amendment changes the hypothesis set, not any decision made after seeing outcomes. Nothing in it is conditioned on a return series, a Sharpe, a fold result or a plot; the only new information is that the data exist and what their coverage is. The amendment is committed with a git timestamp, the frozen v1 text is unchanged, and the added hypotheses enter DSR `N` like every other trial, so a reader can verify that the search space grew *before* the search started. Had the same change been made after the first run, it would be a result-driven expansion of the search and would have to be reported as such; made now, it is a change of plan and its cost is paid in `N`.

*Coverage limit (disclosed now).* The search window is 40 months of WF plus the 8-month holdout, about 48 months ending at the last data date (2026-11-09): the WF window opens around 2022-11 and the holdout around 2026-03.

- **OI, long/short and taker ratios (from 2021-01-01)** cover the full WF window and the holdout with at least 22 months of warm-up before fold 1's 18-month training window. H5 therefore runs on the standard 6-fold scheme and its results are comparable with H1 to H4 without caveat.
- **bookDepth (from 2023-01-01)** does *not* cover the start of fold 1's training window and has no warm-up before it. Anything that uses depth (H6 and any combination containing H6) is validated on its own **depth window**: the standard fold boundaries are kept, the series starts 2023-01-01, every fold whose training window would be shorter than 12 months is dropped, and the fold-pass threshold becomes at least ceil(2k/3) of the k remaining folds (6 → 4, 5 → 4, 4 → 3). Its benchmarks (raw H1, B&H, best catalog strategy) are re-evaluated on the same depth window for the comparison; a depth result is never compared against a full-window number. The shorter window carries **lower statistical power**: with OOS months T_d < T, the standard error of the OOS Sharpe grows by about sqrt(T / T_d), and the window contains a single bull cycle (2023 to 2025) and one drawdown, so regime diversity is smaller than in the full window. Every report that mentions H6 states the depth-window fold count, the OOS months and this SE factor next to the number. If the final model contains H6, §4 is applied on the depth window and the report says so in its first line.

Still **not registered** (unchanged): the liquidation feed (30-day history) and volume-profile/liquidity nodes from aggTrades (2022 onward). H4's eight features are **not** extended with OI or depth; H4 stays as written.

### H5 — OI/price divergence and leverage flush (filter on H1, or standalone)

*Rationale.* OI counts open leveraged positions; its change tells whether a move is being *opened into* (new leverage chasing price) or *closed out of* (positions forced shut). Price up with OI rising fast and the top-trader position ratio at an extreme means the move is carried by late leveraged longs with little buffer: fragile longs. Price down with OI falling sharply means positions are being closed at market by the liquidation engine, not by anyone's opinion: forced sellers who are exhausted once the cascade stops. The other side of the fragile-longs leg is the late leveraged long who will become the forced seller; the other side of the flush leg is the liquidation engine selling that long's position at market, plus the momentum follower who shorts the breakdown. The edge persists because liquidation is mechanical (margin, not view, closes the position), because each trader sees only their own account while the archive aggregates all of them, and because cross-exchange arbitrage absorbs *price*, not *positioning*. OI is used in BTC units (`sum_open_interest`), never in USD (`sum_open_interest_value` = OI × price, which is mechanically correlated with returns and would make the signal a disguised return feature).

*Rules.* Daily bars; metrics aggregated to daily by as-of join (A1.4). `dOI_k = log(OI_t / OI_{t−k})`, `r_k = log(close_t / close_{t−k})`. `z_OI` = rolling z-score of `dOI_k` over `w` days; `z_top` = rolling z-score of `sum_toptrader_long_short_ratio` over `w` days.
- **Fragile longs:** `r_k > 0` and `z_OI > τ` and `z_top > τ`. Filter mode: multiply H1's long by 0.5 (fixed, not searched) while the condition holds. Standalone mode: short 0.5 × H1's vol-scaled size for `hold` days; exit earlier if OI falls below its entry value (the leverage has left). Mirror for fragile shorts: `r_k < 0`, `z_OI > τ`, `z_top < −τ`.
- **Leverage flush:** `r_k < 0` and `z_OI < −τ`. Both modes: long 0.5 × vol-scaled size for `hold` days; exit at `hold` or if close falls below the flush bar's low. (This leg has a stop, unlike H1, because the thesis is "the forced selling is over"; a new low falsifies it bar by bar.)
- In filter mode the H5 position is added to H1's per §2.3(b); in standalone mode H5 is an engine. The two modes are two registered variants, each a separate trial family in `N`. H1 runs at its median parameters, not re-tuned.

*Parameters (4).* `k` ∈ {1, 3} d; `w` 30–90 d; `τ` 1.5–3.0; `hold` 1–7 d. Size multipliers (0.5) and the OI exit are fixed.

*Data.* `metrics_archive_5m.parquet` (`sum_open_interest`, `sum_toptrader_long_short_ratio`) from 2021-01-01; daily OHLCV. Standard 6-fold WF.

*Falsification.* Filter mode rejected if it does not beat raw H1 on OOS Sharpe in ≥ 4/6 folds. Standalone mode rejected if OOS net Sharpe is not > 0 in ≥ 4/6 folds, **or** net expectancy per round trip is < 50 bp (§3). Either mode declared **untestable** if the WF window holds < 20 independent events per leg (condition true, ≥ 5 days apart), reported per leg. Also rejected if the result reverses sign when `sum_open_interest` is replaced by `sum_open_interest_value` *and* the value version is the better one (that would mean the signal is price, not positioning).

### H6 — Book-depth imbalance as an entry filter (depth window only, 2023 onward)

*Rationale.* Resting depth within ±1 % of mid is the inventory that market makers and limit buyers are willing to absorb right now. When bid depth is abnormally thin relative to ask depth, a long entry has to be filled against a book that will not hold, and the breakout is more likely to be a stop-hunt that reverses (the H2 story); when bids are thick, the entry is confirmed. The other side is the market maker, who pulls quotes ahead of moves that hurt their inventory, and the spoofer, who posts depth they intend to cancel. Spoofed depth is fleeting and a single 30-second snapshot can be anything, so the rule uses the **median** of a full day's snapshots and a z-score over weeks; no intraday use. The edge, if any, is confirmation rather than prediction, and it is expected to be small: depth within 1 % is also a volatility proxy (makers widen in high vol), which is why a placebo on total depth is part of the falsification.

*Rules.* Per daily bar, per snapshot: `I = (bid_b − ask_b) / (bid_b + ask_b)` with `bid_b`, `ask_b` the cumulative depth at band `b` from mid, `I` ∈ [−1, 1]. Daily `I_d` = median of the day's snapshots (as of the daily close; snapshots after the close belong to the next day). `z_I` = rolling z-score of `I_d` over `w` days. AND-gate on H1 entries (§2.3a): a long entry or add is taken only if `z_I > −γ` (veto when bids are abnormally thin); a short entry only if `z_I < γ`. The gate acts on entries only; it never closes an open position. A day with no valid `I_d` (A1.4) is **no veto** (no information is not a signal). H1 at median parameters, not re-tuned.

*Parameters (3).* `b` ∈ {1 %, 2 %}; `w` 20–60 d; `γ` 0.5–1.5. The aggregator (median) and the entries-only scope are fixed.

*Data.* `bookDepth` daily archives from 2023-01-01; daily OHLCV. Validated on the **depth window** of A1.1 with its reduced fold count and threshold ceil(2k/3).

*Falsification.* Rejected if it does not beat raw H1 (evaluated on the same depth window) on OOS Sharpe in at least ceil(2k/3) of k folds, **or** if the placebo filter (the same rule and parameters applied to the z-score of *total* depth `bid_b + ask_b` instead of the imbalance) captures ≥ 80 % of H6's OOS Sharpe gain over raw H1; in that case the information is liquidity level, that is volatility, not direction, and H6 is redundant with a vol filter. Declared **untestable** if fewer than 20 entries are vetoed across the depth-window OOS folds.

### A1.3 Eligible components and trial accounting (changes to §2.1 and §2.4)

- **§2.1 component types, amended:** direction engine = H1, H2, H5-standalone, catalog survivor; filter/overlay = H3, H4, H5-filter, H6, regime filter; exit rule unchanged. §2.2 stands: at most one engine, one filter, one exit, so H5-filter and H6 cannot both appear in one combination, and H5-standalone is an engine that may be filtered by H3, H4 or H6 but not by H5-filter (same signal twice). The eligibility rule is unchanged: individual WF pass and family DSR p > 0.95 first. For H6 and any combination containing H6, "WF pass" means the depth-window pass of A1.1.
- **§2.4 combination cap:** raised from 200 to **300** (an increase of 100, the maximum for one amendment; if the enumeration exceeds 300, keep the 300 lowest-correlation ones). Combinations containing H6 are enumerated in the same `combinations.json`, flagged `window: "depth"`, and run only on the depth window.
- **DSR `N`, amended:** `N` = all S0–S4 trials + every H5 and H6 parameter trial in both windows (the two H5 modes are separate families) + combinations (now ≤ 300) + amendments, this one included (+1). Depth-window trials go into the same `trials` table and the same `N`, not a separate ledger; there is one search, not two. The first results report records `N` with and without the A1 additions so the inflation is visible.

### A1.4 Causal handling of missing metrics and depth bars

The 5-minute archive has 601 missing bars in 152 runs (longest gap 10 h 30 min); depth days may have partial snapshot coverage. Rules, fixed now:

1. **Observation time.** A metrics row is treated as observed at its `create_time`; a 5-minute publication lag is assumed, so a daily feature uses only rows with `create_time ≤ daily close − 5 min`. Depth snapshots are observed at their snapshot time.
2. **Daily aggregation = as-of join, backward only.** The daily OI, ratio and taker values are the last observation at or before the bar close (`merge_asof`, direction `backward`). The depth value is the median of the snapshots inside the bar.
3. **Forward-fill with a maximum age.** The as-of value is used only if its observation is ≤ 12 h old at the bar close; older means the daily value is `NaN`. For depth, a day with < 50 % of its expected snapshots (fewer than 1,440 of about 2,880 at 30 s) means `I_d = NaN`.
4. **Never interpolate.** No linear, spline or mean fill across a gap, in the 5-minute series or the daily series. `dOI_k` is `NaN` if either endpoint is `NaN`. Rolling z-scores skip `NaN` with `min_periods = 0.8 · w`; a window with fewer valid points is `NaN`.
5. **`NaN` is "no signal, no veto".** H5 conditions evaluate to false (H1 unchanged, no standalone entry); H6 does not veto. Open H5 positions are managed by their `hold`/stop rules, which depend only on price, so a gap never freezes a trade.
6. **Logged, not hidden.** Each daily bar carries `metrics_gap` and `depth_gap` flags; the results report states the number of flagged bars per series inside every WF fold and the holdout. If more than 5 % of a fold's bars are flagged for a series, that fold is reported as degraded for the hypotheses using it.
7. **Future gaps.** If the archive is later found to have more gaps than stated here, the count is corrected in a further amendment; the rules above do not change.

## Amendment 2 (2026-11-10, before any full-search result was seen)
Daily (1d) bars cannot reach the plan's minimum of 100 in-sample / 30 per-OOS-fold trades in 40 months. For 1d only, the minimums are 40 in-sample trades and 8 per OOS fold. 1h and 4h keep 100 / 30. S3 Optuna tuning optimises in-sample Sharpe only (OOS folds never enter the objective). Recorded after a 50-config smoke run whose results were not used for any decision.

## Amendment 3 (2026-11-10) — H7, multi-family intraday fusion with an ML gate

*Disclosure.* Before this amendment, S1 (1h, 1d) and S2 (1h) single/pair-component summaries had been generated. H7's rules, features and grids below were written from the literature and from a third-party product description ("Chimera"), not from those results; no parameter is taken from them. H7 is a new trial family and every configuration below is added to the DSR `N`.

### H7 — Five signal families fused by a meta-labeling gate, multi-coin, short holds

*Rationale.* Each family alone is weak and well known; the hypothesis is that their *conjunction* at the moment of an event carries information a single rule misses, and that a secondary model can tell which events are worth paying fees for (H4's logic, at intraday scale and across coins). Universe BTC, ETH, SOL, XRP (USDT perpetuals, Binance).

*Feature families (all causal, A1.4 rules apply, 1h bars, published-lag aware):*
1. Order flow — taker buy/sell ratio z (24h, 7d), signed taker volume burst.
2. Liquidation map — distance (ATR) and size of the nearest modeled long/short liquidation cluster above/below (OI-delta model of `btclab/liquidity/liq_model.py`, a model, not observed liquidations), cluster consumed in the last bar (yes/no).
3. Derivatives positioning — ΔOI 1h/4h/24h z, funding z (30d), global and top-trader long/short ratio z.
4. Spot–perp basis — (perp − spot)/spot z (7d), its 4h change.
5. Session — hour-of-day and weekday (cyclical encoding), Asia/Europe/US session flag.
Plus 4 context features: realized vol z, 24h return, ER(24), coin id (one-hot).

*Primary events.* A bar is an event if any family crosses a fixed extreme: |taker z| > 2, nearest cluster consumed, |ΔOI 4h z| > 2, |basis z| > 2. Direction by fixed rule per trigger (flow/basis: with the move; consumed cluster or OI flush: fade). No event, no trade.

*Label.* Triple barrier per event: TP 1.5 ATR(24, 1h), SL 1 ATR, vertical barrier h ∈ {2, 4, 6} h.

*Gate.* Two registered variants, separate families in `N`: (a) L2 logistic regression; (b) gradient-boosted trees (max depth 3, ≤ 200 trees, learning rate 0.05, fixed). Pooled across the 4 coins, trained only on each WF fold's training window, purge = h + 24h, embargo 24h, scaling fitted on training data. Take if p > p_min; size ∝ min(1, (p − 0.5)/0.25), equal risk per coin, portfolio gross exposure ≤ 1.

*Parameters (3).* h ∈ {2, 4, 6}; p_min ∈ {0.52, 0.55, 0.58, 0.60}; gate ∈ {logit, GBT} → 24 configurations, all counted.

*Costs.* Taker fee both sides + slippage per `btclab/engine/costs.py` "realistic" preset + funding at settlement; reported also at 2× costs.

*Falsification.* Rejected if any holds: (i) OOS Sharpe after costs does not beat the ungated primary in ≥ 4/6 folds; (ii) OOS precision of taken trades ≤ 53 %; (iii) the GBT gate does not beat the logistic gate OOS (then only the logistic variant may proceed); (iv) a shuffled-label placebo gate reaches ≥ 80 % of the real gate's OOS Sharpe gain; (v) Sharpe at 2× costs ≤ 0; (vi) one family carries > 60 % of permutation importance (treated as a leak or a single-signal strategy and re-evaluated as that). Results are reported per coin and for the portfolio; the portfolio is the registered claim. Same holdout cutoff for all coins; holdout judged once.

## Amendment 4 (2026-11-10) — Heikin Ashi as a signal source

*Disclosure.* Before this amendment, brayvb_v2 was run in-sample on 1h/4h/1d with real and HA signal candles (6 runs: HA better on 4h, worse on 1h and 1d). Those 6 runs are added to `N`.

*Rule.* A spec may set `source: "ha"`: indicators and signal rules read causal Heikin Ashi candles; entries, exits, stops, ATR, funding and costs always use real OHLC (next-bar open fills, as for every spec). Executing at HA prices is forbidden (it is TradingView's "non-standard chart" bias). HA is a variant, not a free parameter: for any catalog strategy or search survivor evaluated with HA, the HA run is a separate trial family in the same `trials` table and in `N`. A strategy may switch to HA only if the HA variant itself passes every gate (WF, DSR, PBO) on its own; picking the better of real/HA per strategy without that counts as a selection and is reported as such.

## Amendment 5 (2026-11-10, before any cross-asset result was seen) — Cross-asset robustness of the brayvb_v2 default preset

*Disclosure.* brayvb_v2 was tuned and evaluated only on BTCUSDT (Amendment 4 and `research/brayvb_tune.py`). No ETH, SOL, BNB, XRP, ADA, NEAR or FET result of this strategy had been generated before this amendment.

*Test.* The unmodified `brayvb_v2` DEFAULT preset (`btclab/api/presets.py`; indicator defaults, no stop, no trail, long only, `BINANCE_PERP_TAKER` costs plus funding, next-bar-open fills) is run with ZERO parameter changes. No tuning, no selection among configurations. Universe: alts ETHUSDT SOLUSDT BNBUSDT XRPUSDT ADAUSDT NEARUSDT FETUSDT (7); BTCUSDT is run as a reference only and is not counted in any pass rule. Timeframes 1d and 4h; sources `real` and `ha` (Amendment 4). 4 configs (tf x source) x 8 assets = 32 trials. Data via `btclab/data/loaders.py` with `symbol=`; the global holdout cutoff (2026-03-10 14:00 UTC) is enforced for all assets and never unlocked.

*Metric / folds.* The same 6 walk-forward folds per asset: test windows are the six 6-month tiles ending at the holdout cutoff (identical calendar windows for every asset), train = 18 months ending 5 days before test start (purge 200 bars), as in the search infrastructure (`walk_forward_folds` geometry). For alts with shorter history the train window is truncated at the asset's first bar; folds whose train window has < 12 months of data are dropped and reported (the strategy has no fitted parameters, so this is a data-maturity filter, not a fitting requirement). Per asset-config: OOS Sharpe, CAGR and max drawdown are computed on the stitched daily OOS returns of the retained folds; B&H (spot-style close-to-close, no funding, as in D3) is computed on the same days; Calmar = CAGR / maxDD.

*Pass rule "robust across assets"* (per config, over the 7 alts): ALL of (i) median-over-alts OOS Sharpe > 0; (ii) OOS Sharpe > 0 on >= 5 of 7 alts; (iii) strategy Calmar > B&H Calmar on >= 4 of 7 alts. *Secondary question "loses less than B&H"*: strategy OOS max drawdown <= 0.6 x B&H max drawdown on >= 5 of 7 alts. Both are reported per config; no config is promoted by this test alone.

*Trial accounting.* The 32 trials enter the `trials` table (stage `xasset`, family `xasset:<tf>:<source>`, tag = symbol) and the DSR `N`: N = rows of the main trials table before insertion + 32 + 1 (this amendment). A pooled-across-alts DSR is added per config: the pooled series is the equal-weight mean of the 7 alts' OOS daily returns on each day (available alts only), deflated at that updated N, with trial-Sharpe variance taken over the 28 alt trials. Holdout is not evaluated.
