# A7 validation (Amendment 7)

## Verdict per config (must pass on V1 and V2, and V3 for C1/C2)

| config | V1 | V2 | V3 | overall |
|---|---|---|---|---|
| C1_LO_s20 | FAIL | FAIL | FAIL | **FAIL** |
| C1_LO_s40 | FAIL | FAIL | FAIL | **FAIL** |
| C1_LS_s20 | FAIL | FAIL | FAIL | **FAIL** |
| C1_LS_s40 | FAIL | FAIL | FAIL | **FAIL** |
| C2_LO_L14 | FAIL | FAIL | FAIL | **FAIL** |
| C2_LS_L14 | FAIL | PASS | FAIL | **FAIL** |
| C2_LO_L28 | FAIL | FAIL | FAIL | **FAIL** |
| C2_LS_L28 | FAIL | FAIL | FAIL | **FAIL** |
| C2_LO_L56 | FAIL | FAIL | FAIL | **FAIL** |
| C2_LS_L56 | FAIL | FAIL | FAIL | **FAIL** |
| C3_th10 | FAIL | FAIL | n/a | **FAIL** |
| C3_th20 | PASS | FAIL | n/a | **FAIL** |

Configs passing: 0/12. None passes: that is the result.

Rule detail:

- C1_LO_s20 V1: FAIL (Sharpe 0.92>=EW 0.69+0.3:False; DD 89.2%<=0.7*85.3%:False; Sharpe>=brayvb 1.07:False)
- C1_LO_s20 V2: FAIL (Sharpe 0.66>=EW 0.60+0.3:False; DD 29.8%<=0.7*27.4%:False; Sharpe>=brayvb 0.42:True)
- C1_LO_s20 V3: FAIL (Sharpe 0.98>=EW 0.62+0.3:True; DD 19.0%<=0.7*88.0%:True; Sharpe>=brayvb 1.93:False)
- C1_LO_s40 V1: FAIL (Sharpe 0.99>=EW 0.69+0.3:True; DD 96.5%<=0.7*85.3%:False; Sharpe>=brayvb 1.07:False)
- C1_LO_s40 V2: FAIL (Sharpe 0.73>=EW 0.60+0.3:False; DD 41.8%<=0.7*27.4%:False; Sharpe>=brayvb 0.42:True)
- C1_LO_s40 V3: FAIL (Sharpe 0.98>=EW 0.62+0.3:True; DD 34.8%<=0.7*88.0%:True; Sharpe>=brayvb 1.93:False)
- C1_LS_s20 V1: FAIL (Sharpe 0.48>=EW 0.69+0.3:False; DD 98.3%<=0.7*85.3%:False; Sharpe>=brayvb 1.07:False)
- C1_LS_s20 V2: FAIL (Sharpe 0.41>=EW 0.60+0.3:False; DD 29.5%<=0.7*27.4%:False; Sharpe>=brayvb 0.42:False)
- C1_LS_s20 V3: FAIL (Sharpe 1.11>=EW 0.62+0.3:True; DD 20.1%<=0.7*88.0%:True; Sharpe>=brayvb 1.93:False)
- C1_LS_s40 V1: FAIL (Sharpe 0.55>=EW 0.69+0.3:False; DD 99.2%<=0.7*85.3%:False; Sharpe>=brayvb 1.07:False)
- C1_LS_s40 V2: FAIL (Sharpe 0.25>=EW 0.60+0.3:False; DD 38.3%<=0.7*27.4%:False; Sharpe>=brayvb 0.42:False)
- C1_LS_s40 V3: FAIL (Sharpe 1.09>=EW 0.62+0.3:True; DD 38.3%<=0.7*88.0%:True; Sharpe>=brayvb 1.93:False)
- C2_LO_L14 V1: FAIL (Sharpe 0.63>=EW 0.69+0.3:False; DD 93.1%<=0.7*85.3%:False; Sharpe>=brayvb 1.07:False)
- C2_LO_L14 V2: FAIL (Sharpe 1.42>=EW 0.60+0.3:True; DD 26.8%<=0.7*27.4%:False; Sharpe>=brayvb 0.42:True)
- C2_LO_L14 V3: FAIL (Sharpe 1.03>=EW 0.62+0.3:True; DD 77.3%<=0.7*88.0%:False; Sharpe>=brayvb 1.93:False)
- C2_LS_L14 V1: FAIL (Sharpe 0.00>=EW 0.69+0.3:False; DD 65.0%<=0.7*85.3%:False; Sharpe>=brayvb 1.07:False)
- C2_LS_L14 V2: PASS (Sharpe 1.20>=EW 0.60+0.3:True; DD 16.4%<=0.7*27.4%:True; Sharpe>=brayvb 0.42:True)
- C2_LS_L14 V3: FAIL (Sharpe 1.22>=EW 0.62+0.3:True; DD 36.3%<=0.7*88.0%:True; Sharpe>=brayvb 1.93:False)
- C2_LO_L28 V1: FAIL (Sharpe 0.70>=EW 0.69+0.3:False; DD 87.5%<=0.7*85.3%:False; Sharpe>=brayvb 1.07:False)
- C2_LO_L28 V2: FAIL (Sharpe 0.62>=EW 0.60+0.3:False; DD 28.8%<=0.7*27.4%:False; Sharpe>=brayvb 0.42:True)
- C2_LO_L28 V3: FAIL (Sharpe 0.55>=EW 0.62+0.3:False; DD 81.9%<=0.7*88.0%:False; Sharpe>=brayvb 1.93:False)
- C2_LS_L28 V1: FAIL (Sharpe 0.12>=EW 0.69+0.3:False; DD 57.5%<=0.7*85.3%:True; Sharpe>=brayvb 1.07:False)
- C2_LS_L28 V2: FAIL (Sharpe -0.49>=EW 0.60+0.3:False; DD 23.2%<=0.7*27.4%:False; Sharpe>=brayvb 0.42:False)
- C2_LS_L28 V3: FAIL (Sharpe 0.01>=EW 0.62+0.3:False; DD 57.6%<=0.7*88.0%:True; Sharpe>=brayvb 1.93:False)
- C2_LO_L56 V1: FAIL (Sharpe 0.64>=EW 0.69+0.3:False; DD 91.9%<=0.7*85.3%:False; Sharpe>=brayvb 1.07:False)
- C2_LO_L56 V2: FAIL (Sharpe -0.26>=EW 0.60+0.3:False; DD 30.3%<=0.7*27.4%:False; Sharpe>=brayvb 0.42:False)
- C2_LO_L56 V3: FAIL (Sharpe 0.78>=EW 0.62+0.3:False; DD 86.6%<=0.7*88.0%:False; Sharpe>=brayvb 1.93:False)
- C2_LS_L56 V1: FAIL (Sharpe -0.09>=EW 0.69+0.3:False; DD 61.9%<=0.7*85.3%:False; Sharpe>=brayvb 1.07:False)
- C2_LS_L56 V2: FAIL (Sharpe -3.37>=EW 0.60+0.3:False; DD 37.6%<=0.7*27.4%:False; Sharpe>=brayvb 0.42:False)
- C2_LS_L56 V3: FAIL (Sharpe 0.59>=EW 0.62+0.3:False; DD 37.7%<=0.7*88.0%:True; Sharpe>=brayvb 1.93:False)
- C3_th10 V1: FAIL (Sharpe 2.88>=1.0:True; DD 1.9%<=15%:True; posQ 66.7%>=70%:False)
- C3_th10 V2: FAIL (Sharpe -1.19>=1.0:False; DD 0.3%<=15%:True; posQ 0.0%>=70%:False)
- C3_th20 V1: PASS (Sharpe 2.52>=1.0:True; DD 1.7%<=15%:True; posQ 74.1%>=70%:True)
- C3_th20 V2: FAIL (Sharpe n/a>=1.0:False; DD 0.0%<=15%:True; posQ 0.0%>=70%:False)

## Best config (selection across classes, by V1 Sharpe): C3_th10

- V1: Sharpe 2.88, max DD 1.9%; EW B&H Sharpe 0.69 / max DD 85.3%; brayvb Sharpe 1.07.
- V2: Sharpe -1.19, max DD 0.3%; EW B&H Sharpe 0.60 / max DD 27.4%; brayvb Sharpe 0.42.

## Deflated Sharpe

N = 191630 prior trials + 12 A7 configs = **191642**. Trial-Sharpe variance (a): across the 12 configs on the same set (per-day Sharpes); (b) ledger variance of the prior trials' Sharpes (0.283, assumed annualised, /365 per day). E[max SR] annualised. DSR = probability the true Sharpe exceeds E[max SR]; > 0.95 passes. Selection of the best config is judged at this N.

| set | config | Sharpe | DSR (a) | E[maxSR] (a) | DSR (b) | E[maxSR] (b) |
|---|---|---|---|---|---|---|
| V1 | C1_LO_s20 | 0.92 | 0.000 | 4.19 | 0.000 | 2.41 |
| V1 | C1_LO_s40 | 0.99 | 0.000 | 4.19 | 0.000 | 2.41 |
| V1 | C1_LS_s20 | 0.48 | 0.000 | 4.19 | 0.000 | 2.41 |
| V1 | C1_LS_s40 | 0.55 | 0.000 | 4.19 | 0.000 | 2.41 |
| V1 | C2_LO_L14 | 0.63 | 0.000 | 4.19 | 0.000 | 2.41 |
| V1 | C2_LS_L14 | 0.00 | 0.000 | 4.19 | 0.000 | 2.41 |
| V1 | C2_LO_L28 | 0.70 | 0.000 | 4.19 | 0.000 | 2.41 |
| V1 | C2_LS_L28 | 0.12 | 0.000 | 4.19 | 0.000 | 2.41 |
| V1 | C2_LO_L56 | 0.64 | 0.000 | 4.19 | 0.000 | 2.41 |
| V1 | C2_LS_L56 | -0.09 | 0.000 | 4.19 | 0.000 | 2.41 |
| V1 | C3_th10 (best) | 2.88 | 0.001 | 4.19 | 0.875 | 2.41 |
| V1 | C3_th20 | 2.52 | 0.000 | 4.19 | 0.598 | 2.41 |
| V2 | C1_LO_s20 | 0.66 | 0.000 | 6.10 | 0.076 | 2.41 |
| V2 | C1_LO_s40 | 0.73 | 0.000 | 6.10 | 0.085 | 2.41 |
| V2 | C1_LS_s20 | 0.41 | 0.000 | 6.10 | 0.051 | 2.41 |
| V2 | C1_LS_s40 | 0.25 | 0.000 | 6.10 | 0.039 | 2.41 |
| V2 | C2_LO_L14 | 1.42 | 0.000 | 6.10 | 0.207 | 2.41 |
| V2 | C2_LS_L14 | 1.20 | 0.000 | 6.10 | 0.166 | 2.41 |
| V2 | C2_LO_L28 | 0.62 | 0.000 | 6.10 | 0.071 | 2.41 |
| V2 | C2_LS_L28 | -0.49 | 0.000 | 6.10 | 0.009 | 2.41 |
| V2 | C2_LO_L56 | -0.26 | 0.000 | 6.10 | 0.014 | 2.41 |
| V2 | C2_LS_L56 | -3.37 | 0.000 | 6.10 | 0.000 | 2.41 |
| V2 | C3_th10 (best) | -1.19 | 0.000 | 6.10 | 0.001 | 2.41 |
| V2 | C3_th20 | n/a | n/a | 6.10 | n/a | 2.41 |
| V3 | C1_LO_s20 | 0.98 | 0.163 | 1.65 | 0.018 | 2.41 |
| V3 | C1_LO_s40 | 0.98 | 0.164 | 1.65 | 0.018 | 2.41 |
| V3 | C1_LS_s20 | 1.11 | 0.212 | 1.65 | 0.028 | 2.41 |
| V3 | C1_LS_s40 | 1.09 | 0.206 | 1.65 | 0.027 | 2.41 |
| V3 | C2_LO_L14 | 1.03 | 0.186 | 1.65 | 0.023 | 2.41 |
| V3 | C2_LS_L14 | 1.22 | 0.272 | 1.65 | 0.045 | 2.41 |
| V3 | C2_LO_L28 | 0.55 | 0.058 | 1.65 | 0.004 | 2.41 |
| V3 | C2_LS_L28 | 0.01 | 0.009 | 1.65 | 0.000 | 2.41 |
| V3 | C2_LO_L56 | 0.78 | 0.107 | 1.65 | 0.010 | 2.41 |
| V3 | C2_LS_L56 | 0.59 | 0.064 | 1.65 | 0.005 | 2.41 |

Trials table: 46 stage-'a7' rows now stored (12 configs x design/V1/V2 + 10 x V3); total ledger 191676 rows.

## Notes

- V1: 25/25 coins loaded; panel 2019-12-19 -> 2026-11-09; evaluation skips the first 120 days. C3 excludes coins without a spot pair: 1000PEPEUSDT, XMRUSDT. Coins: ZECUSDT, 1000PEPEUSDT, DOGEUSDT, UNIUSDT, SUIUSDT, LINKUSDT, AVAXUSDT, FILUSDT, INJUSDT, AAVEUSDT, QNTUSDT, ARBUSDT, BCHUSDT, XMRUSDT, GTCUSDT, LTCUSDT, TRXUSDT, IOSTUSDT, DOTUSDT, ONEUSDT, APTUSDT, OPUSDT, XLMUSDT, BATUSDT, RSRUSDT.
- V2: design coins, evaluated 2026-03-11 -> 2026-11-09; signals/weights run on the full series (design history = warm-up only), returns sliced at the start.
- V3: spot daily, full window 2017-08-17 -> 2019-09-08 (strategies are flat during the 120-day signal warm-up; no skip). Spot fee 10 bp + 2 bp slippage, no funding; long-short configs are simulated with synthetic shorts at spot cost (historical signal test). brayvb benchmark on spot 4h, SPOT costs, long-only.

## V1: fresh universe

| name | Sharpe | CAGR | max DD | Calmar | turnover/yr | % pos quarters | days |
|---|---|---|---|---|---|---|---|
| C1_LO_s20 | 0.92 | 50.4% | 89.2% | 0.56 | 103.5 | 40.7% | 2398 |
| C1_LO_s40 | 0.99 | 52.1% | 96.5% | 0.54 | 149.4 | 44.4% | 2398 |
| C1_LS_s20 | 0.48 | -10.8% | 98.3% | -0.11 | 174.9 | 44.4% | 2398 |
| C1_LS_s40 | 0.55 | -12.2% | 99.2% | -0.12 | 193.0 | 40.7% | 2398 |
| C2_LO_L14 | 0.63 | 19.8% | 93.1% | 0.21 | 51.0 | 44.4% | 2398 |
| C2_LS_L14 | 0.00 | -5.6% | 65.0% | -0.09 | 54.5 | 40.7% | 2398 |
| C2_LO_L28 | 0.70 | 26.3% | 87.5% | 0.30 | 37.2 | 51.9% | 2398 |
| C2_LS_L28 | 0.12 | -1.0% | 57.5% | -0.02 | 39.7 | 63.0% | 2398 |
| C2_LO_L56 | 0.64 | 19.9% | 91.9% | 0.22 | 30.4 | 55.6% | 2398 |
| C2_LS_L56 | -0.09 | -8.0% | 61.9% | -0.13 | 32.1 | 40.7% | 2398 |
| C3_th10 | 2.88 | 4.7% | 1.9% | 2.43 | 33.1 | 66.7% | 2398 |
| C3_th20 | 2.52 | 5.1% | 1.7% | 2.94 | 20.4 | 74.1% | 2398 |
| EW_BH | 0.69 | 25.6% | 85.3% | 0.30 | 8.8 | 51.9% | 2398 |
| BTC_BH | 0.72 | 28.0% | 78.9% | 0.35 | 0.1 | 51.9% | 2398 |
| BRAYVB | 1.07 | 30.2% | 34.3% | 0.88 | n/a | 51.9% | 2398 |

## V2: design coins after the cutoff

| name | Sharpe | CAGR | max DD | Calmar | turnover/yr | % pos quarters | days |
|---|---|---|---|---|---|---|---|
| C1_LO_s20 | 0.66 | 22.9% | 29.8% | 0.77 | 54.9 | 25.0% | 244 |
| C1_LO_s40 | 0.73 | 30.6% | 41.8% | 0.73 | 93.8 | 25.0% | 244 |
| C1_LS_s20 | 0.41 | 6.3% | 29.5% | 0.21 | 145.4 | 50.0% | 244 |
| C1_LS_s40 | 0.25 | -8.4% | 38.3% | -0.22 | 196.9 | 50.0% | 244 |
| C2_LO_L14 | 1.42 | 94.4% | 26.8% | 3.53 | 45.9 | 50.0% | 244 |
| C2_LS_L14 | 1.20 | 26.3% | 16.4% | 1.60 | 47.0 | 100.0% | 244 |
| C2_LO_L28 | 0.62 | 20.9% | 28.8% | 0.73 | 35.1 | 50.0% | 244 |
| C2_LS_L28 | -0.49 | -12.5% | 23.2% | -0.54 | 41.3 | 25.0% | 244 |
| C2_LO_L56 | -0.26 | -24.4% | 30.3% | -0.81 | 30.2 | 25.0% | 244 |
| C2_LS_L56 | -3.37 | -48.1% | 37.6% | -1.28 | 35.0 | 25.0% | 244 |
| C3_th10 | -1.19 | -0.4% | 0.3% | -1.13 | 13.9 | 0.0% | 244 |
| C3_th20 | n/a | 0.0% | 0.0% | n/a | 0.0 | 0.0% | 244 |
| EW_BH | 0.60 | 19.4% | 27.4% | 0.71 | 5.3 | 50.0% | 244 |
| BTC_BH | 0.39 | 7.8% | 29.0% | 0.27 | 0.0 | 25.0% | 244 |
| BRAYVB | 0.42 | 7.5% | 16.8% | 0.45 | n/a | 50.0% | 244 |

## V3: BTC+ETH spot 2017-08-17 -> 2019-09-08

| name | Sharpe | CAGR | max DD | Calmar | turnover/yr | % pos quarters | days |
|---|---|---|---|---|---|---|---|
| C1_LO_s20 | 0.98 | 18.1% | 19.0% | 0.95 | 9.0 | 33.3% | 753 |
| C1_LO_s40 | 0.98 | 34.7% | 34.8% | 1.00 | 17.9 | 22.2% | 753 |
| C1_LS_s20 | 1.11 | 29.9% | 20.1% | 1.49 | 21.0 | 33.3% | 753 |
| C1_LS_s40 | 1.09 | 55.8% | 38.3% | 1.46 | 42.7 | 33.3% | 753 |
| C2_LO_L14 | 1.03 | 68.3% | 77.3% | 0.88 | 21.8 | 44.4% | 753 |
| C2_LS_L14 | 1.22 | 47.4% | 36.3% | 1.31 | 26.3 | 55.6% | 753 |
| C2_LO_L28 | 0.55 | 10.2% | 81.9% | 0.12 | 16.0 | 44.4% | 753 |
| C2_LS_L28 | 0.01 | -7.3% | 57.6% | -0.13 | 21.5 | 44.4% | 753 |
| C2_LO_L56 | 0.78 | 35.3% | 86.6% | 0.41 | 8.2 | 44.4% | 753 |
| C2_LS_L56 | 0.59 | 16.2% | 37.7% | 0.43 | 13.6 | 44.4% | 753 |
| EW_BH | 0.62 | 16.2% | 88.0% | 0.18 | 5.1 | 55.6% | 753 |
| BTC_BH | 0.93 | 53.5% | 83.2% | 0.64 | 0.5 | 55.6% | 753 |
| BRAYVB | 1.93 | 108.5% | 20.9% | 5.19 | n/a | 66.7% | 753 |