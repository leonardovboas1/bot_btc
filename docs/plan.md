# BTC Signal Lab — Implementation Plan (v1, 2026-11-10, architect: Fable)

## Principle
Monorepo. Python does all numerics; the browser only renders. Every indicator is a pure function `(ohlcv arrays, params) -> arrays`, shared by backtester, live bot and chart API (no backtest/live drift).

## A. Stack
**Data** (free, Binance): spot klines 2017-08→now and perp klines 2019-09→now (bulk `data.binance.vision` + REST gap-fill); funding 2019-09→now; OI / long-short / taker ratio (REST only ~30d, then accrue by polling every 5m; check `metrics` archive on data.binance.vision for longer OI history); aggTrades bulk 2022+ (volume profile/CVD); depth WS recorded going forward. Storage: Parquet, UTC ms, queried with **DuckDB**; higher TFs derived deterministically.

**Backtest engine:** own **numpy + numba** engine. (1) Signal library: each indicator computed once per (TF, params), cached as arrays keyed by hash. (2) `@njit` portfolio simulator: entries long/short, exits, fees, slippage, funding at 8h boundaries, SL/TP/trailing → trades + equity. vectorbt (OSS) only in tests for parity. backtesting.py rejected (too slow).

**Backend:** FastAPI: `/ohlcv`, `/indicator/{name}`, `/signals/{id}`, `/backtest`, `/heatmap/liq`, `/heatmap/book`, WS `/live`.
**Frontend:** TradingView **Lightweight Charts v5** (Apache-2.0) + React + Vite + TS. Panes for oscillators, markers for signals, custom primitives for heatmap and 4 drawing tools (trendline, horizontal ray, rectangle, fib). Full TradingView Charting Library needs a licence — not used.
**Pine:** hand-port to Python with Pine-faithful semantics (`ta.ema` seeding, `request.security` as HTF resample with lookahead off, crossover). Fixture tests vs TradingView CSV exports. Reverse: `pine/emitter.py` generates Pine v6 `strategy()` from a JSON strategy DSL so the winner can be verified in TradingView.

## B. Liquidity heatmap
| Layer | Source | Backtestable? |
|---|---|---|
| (a) Live order-book heatmap | Binance depth WS, 1s snapshots, $10 buckets | Only from recording start (useful after ~6 months) |
| (b) Estimated liquidation heatmap | OI increases × leverage mix (10/25/50/100x) → liq prices, cleared when traded through | Last 30d + accrued; earlier only via labelled low-confidence proxy |
| (c) Volume-profile proxy | aggTrades volume-at-price, HVN/LVN | Yes, full history |

## C. Statistical methodology
- **Untouched holdout:** last 8 months; loader refuses dates past cutoff unless `HOLDOUT_UNLOCK=1`. Used once on top ≤5.
- **Walk-forward, purged + embargoed:** remaining 40 months, 6 folds (train 18m / test 6m rolling), purge = max lookback, embargo 5 days.
- **CSCV/PBO:** S=16 blocks, C(16,8)=12,870 splits.
- **Costs (perp):** taker 0.05%/side (also test user's 0.1%), slippage 2bp + ATR-scaled, funding every 8h, signal on close → fill next open. Spot variant: 0.1%, long only.
- **Objective:** DSR-adjusted Sharpe (daily returns) subject to ≥100 IS trades, ≥30 per OOS fold, MaxDD ≤35%, PF >1.2, positive in ≥4/6 OOS folds. Also report CAGR, Sortino, Calmar, OOS/IS degradation, expectancy (bp), exposure, turnover.

| Stage | Test | Pass |
|---|---|---|
| Every candidate | WF OOS Sharpe | >0 in ≥4/6 folds |
| Family | Deflated Sharpe (N = all trials) | p >0.95 |
| Family | PBO (CSCV) | <0.2 |
| Top 50 | White RC + Hansen SPA (stationary bootstrap 2,000) vs B&H and zero | p <0.05 |
| Top 50 | Block bootstrap of trades (10k) | 5th pct equity >0 and > B&H 5th pct |
| Top 50 | Param stability ±30% | neighbour median ≥70% of peak |
| Top 50 | Regimes (200d slope × vol terciles) | no regime worse than B&H |
| Top 5 | Holdout | report only, no re-tuning |

**Staged search:** S0 signal library (~300 configs/TF) → S1 single primitives × 3 exits (~5k/TF, keep top 15%) → S2 pairs entry+filter (~100k) → S3 triples from survivors + Optuna TPE (200 trials each) → S4 gauntlet. Every trial logged in DuckDB `trials` table (honest N). Benchmarks: B&H, 50/50 rebalanced, random entries with same trade count.

## D. Layout
```
btclab/ data/ indicators/ signals/(primitives, dsl, combiner) engine/(sim, costs, metrics)
        research/(stages, optuna_search, stats/{dsr,cscv,spa,bootstrap,stability,regimes})
        liquidity/(book_recorder, liq_model, volume_profile) pine/(emitter) live/(signal_bot) api/
web/ data/ research/(runs, cache, reports) docs/ tests/
```

| M | Deliverable | Verification |
|---|---|---|
| M1 Data | download, funding, OI poller | no gaps/dupes; resampled 1h == Binance 1h |
| M2 Indicators | catalog ported | fixtures vs TradingView for ≥10 incl. Cipher B, Gaussian |
| M3 Engine | sim, costs, metrics, DSL | parity vs vectorbt; user's Gaussian+StochRSI matches TV trade count ±2 |
| M4 Research | staged search + gauntlet | noise strategies → PBO≈0.5 & DSR fail; planted edge passes |
| M5 Platform | API + chart + backtest panel + heatmaps | markers match trades table |
| M6 Live bot | signal on bar close + Pine export | Pine reproduces last 20 signals in TV |

Tasks/tier: T1 scaffold (sonnet) · T2 downloader (sonnet) · T3 pollers (sonnet) · T4 Pine-semantics base (opus) · T5 catalog ports ×5 (sonnet; Cipher B opus) · T6 fixtures (haiku) · T7 numba sim (fable) · T8 vectorbt parity (sonnet) · T9 DSL+combiner+cache (opus) · T10 stats module (fable) · T11 search orchestrator (opus) · T12 null tests (sonnet) · T13 API (sonnet) · T14 web chart (opus) · T15 drawings+heatmap primitives (opus) · T16 liquidity (fable model / sonnet recorder) · T17 Pine emitter (opus) · T18 live bot+notifier (sonnet) · T19 docs/reports (haiku).
Waves: {T1} → {T2,T3,T4,T7} → {T5,T8,T9,T10,T16} → {T6,T11,T12,T13,T17} → {T14,T18,T19} → {T15}.

## E. Risks
Backtest ≠ forecast; expect modest regime-dependent edges. OI history thin (don't leak proxies into 4-year claims). Binance geo/API risk → exchange-abstracted layer, Bybit fallback. Pine fidelity in recursive filters → fixtures mandatory. DSR N must count every trial. Heatmap (a) needs months of recording; (b) is a model. Windows+numba: pin versions, cache kernels. Live signals actionable at next bar open; auto-execution needs a separate risk layer.
