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
