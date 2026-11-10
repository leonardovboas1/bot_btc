**HOLDOUT VERDICT (Amendment 6): FAIL: candidate rejected; holdout spent**

# brayvb_v2 default, 4h, Heikin Ashi signals: one-shot holdout judgment

Window: 2026-03-10 16:00:00+00:00 → 2026-11-10 16:00:00+00:00 (246 days). Signals on the full series; equity re-based at the cutoff. Costs BINANCE_PERP_TAKER + funding; B&H = long perp incl. funding. Position open at cutoff (SOL): False.

## Primary: SOLUSDT

| asset | src | Sharpe | total | CAGR | max DD | Calmar | trades | win | B&H Sharpe | B&H total | B&H max DD | B&H Calmar | (a) | (b) | (c) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SOLUSDT | ha | 0.47 | 6.7% | 10.2% | 23.9% | 0.42 | 18 | 22.2% | 0.08 | -6.8% | 36.0% | -0.27 | ✓ | ✗ | ✓ |

Primary pass (a ∧ b ∧ c): **FAIL**. Consistency (reported, not gating): holdout Sharpe 0.47 vs WF-OOS block-bootstrap 5–95% band [-0.90, 2.96] → inside.

## Control: same config on 7 other assets

| asset | src | Sharpe | total | CAGR | max DD | Calmar | trades | win | B&H Sharpe | B&H total | B&H max DD | B&H Calmar | (a) | (b) | (c) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | ha | -0.31 | -5.4% | -7.9% | 16.3% | -0.48 | 25 | 20.0% | 0.26 | 1.8% | 29.0% | 0.09 | ✗ | ✓ | ✗ |
| ETHUSDT | ha | -0.55 | -11.6% | -16.8% | 29.4% | -0.57 | 26 | 19.2% | 0.16 | -3.4% | 35.3% | -0.14 | ✗ | ✗ | ✗ |
| BNBUSDT | ha | -1.49 | -18.1% | -25.7% | 22.4% | -1.15 | 23 | 21.7% | 0.26 | 1.7% | 24.0% | 0.11 | ✗ | ✗ | ✗ |
| XRPUSDT | ha | -0.48 | -10.1% | -14.6% | 24.6% | -0.59 | 15 | 13.3% | -0.55 | -26.3% | 35.5% | -1.03 | ✗ | ✗ | ✓ |
| ADAUSDT | ha | -1.40 | -26.6% | -36.8% | 28.7% | -1.28 | 11 | 27.3% | -0.64 | -36.9% | 50.5% | -0.98 | ✗ | ✓ | ✓ |
| NEARUSDT | ha | 1.45 | 72.0% | 123.6% | 40.0% | 3.09 | 22 | 27.3% | 1.88 | 165.2% | 44.2% | 7.35 | ✓ | ✗ | ✗ |
| FETUSDT | ha | 1.03 | 35.3% | 56.6% | 23.7% | 2.39 | 12 | 41.7% | 0.49 | 0.4% | 56.3% | 0.01 | ✓ | ✓ | ✓ |

Generalises ((a) and (b) on ≥ 4/7): **NO** (1/7).

## Comparison (not gating): SOLUSDT with real-candle signals

| asset | src | Sharpe | total | CAGR | max DD | Calmar | trades | win | B&H Sharpe | B&H total | B&H max DD | B&H Calmar | (a) | (b) | (c) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SOLUSDT | real | 0.34 | 4.1% | 6.1% | 28.6% | 0.21 | 19 | 21.1% | 0.08 | -6.8% | 36.0% | -0.27 | ✓ | ✗ | ✓ |

Honesty note: ~8 months cannot separate Sharpe 1.0 from 0.5 (SE ≈ 1.2); this run can refute, not prove. The holdout is now spent; it must not be used for further selection.