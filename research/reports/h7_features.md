# H7 data report (features, events, labels)

Data side only (Amendment 3). No strategy performance is computed here. Holdout cutoff 2026-03-10T14:00:00+00:00 (exclusive) is enforced; no row, feature or label end time reaches it.

## Rows and range

| symbol | rows | first bar | last bar | metrics_gap | spot_gap |
|---|---|---|---|---|---|
| BTCUSDT | 44726 | 2021-02-01 00:00:00+00:00 | 2026-03-10 13:00:00+00:00 | 7.09% | 0.03% |
| ETHUSDT | 36710 | 2022-01-01 00:00:00+00:00 | 2026-03-10 13:00:00+00:00 | 8.41% | 0.00% |
| SOLUSDT | 36710 | 2022-01-01 00:00:00+00:00 | 2026-03-10 13:00:00+00:00 | 8.41% | 0.00% |
| XRPUSDT | 36710 | 2022-01-01 00:00:00+00:00 | 2026-03-10 13:00:00+00:00 | 8.34% | 0.00% |

Alt metrics start 2021-12-01 (BTC 2021-01-01); tables start 31 days after the first metrics row (warm-up for 30 d z-scores and the liquidation model).

## Events per trigger and direction (count, labelled count at h=4 in parentheses)

| symbol | trigger | long | short |
|---|---|---|---|
| BTCUSDT | taker | 986 (986) | 1014 (1014) |
| BTCUSDT | basis | 1496 (1495) | 1181 (1181) |
| BTCUSDT | oi | 1105 (1105) | 1176 (1176) |
| BTCUSDT | liq | 2550 (2550) | 2414 (2414) |
| ETHUSDT | taker | 774 (774) | 777 (777) |
| ETHUSDT | basis | 1230 (1230) | 990 (990) |
| ETHUSDT | oi | 986 (986) | 1073 (1073) |
| ETHUSDT | liq | 2440 (2440) | 2217 (2217) |
| SOLUSDT | taker | 781 (781) | 780 (780) |
| SOLUSDT | basis | 1077 (1077) | 1096 (1096) |
| SOLUSDT | oi | 990 (990) | 993 (993) |
| SOLUSDT | liq | 3593 (3593) | 3207 (3206) |
| XRPUSDT | taker | 820 (820) | 819 (818) |
| XRPUSDT | basis | 1068 (1068) | 951 (951) |
| XRPUSDT | oi | 837 (837) | 920 (920) |
| XRPUSDT | liq | 2709 (2709) | 2044 (2044) |

## Label balance (share of events by first barrier hit; labelled events only)

| symbol | h=2 TP / SL / vertical | h=4 TP / SL / vertical | h=6 TP / SL / vertical |
|---|---|---|---|
| BTCUSDT | 10.1% / 28.8% / 61.0% (n=11922) | 18.3% / 41.2% / 40.5% (n=11921) | 23.9% / 48.2% / 27.9% (n=11920) |
| ETHUSDT | 10.5% / 28.3% / 61.2% (n=10487) | 19.7% / 41.6% / 38.7% (n=10487) | 25.3% / 48.7% / 26.0% (n=10486) |
| SOLUSDT | 10.6% / 28.0% / 61.5% (n=12517) | 20.2% / 41.3% / 38.6% (n=12516) | 26.7% / 48.5% / 24.7% (n=12515) |
| XRPUSDT | 11.2% / 28.5% / 60.2% (n=10168) | 20.5% / 40.8% / 38.7% (n=10167) | 26.6% / 47.5% / 25.9% (n=10165) |

## NaN fraction per feature

| feature | BTCUSDT | ETHUSDT | SOLUSDT | XRPUSDT |
|---|---|---|---|---|
| taker_z24 | 7.3% | 8.6% | 8.6% | 8.6% |
| taker_z168 | 7.4% | 8.8% | 8.8% | 8.8% |
| taker_burst | 0.0% | 0.0% | 0.0% | 0.0% |
| liq_long_dist_atr | 0.0% | 0.0% | 0.0% | 0.0% |
| liq_short_dist_atr | 0.0% | 0.0% | 0.0% | 0.0% |
| liq_long_size | 0.1% | 0.0% | 0.1% | 0.1% |
| liq_short_size | 0.1% | 0.0% | 0.0% | 0.1% |
| liq_long_consumed | 0.0% | 0.0% | 0.0% | 0.0% |
| liq_short_consumed | 0.0% | 0.0% | 0.0% | 0.0% |
| doi1_z | 0.2% | 0.1% | 0.1% | 0.1% |
| doi4_z | 0.2% | 0.1% | 0.1% | 0.1% |
| doi24_z | 0.2% | 0.1% | 0.1% | 0.1% |
| funding_z30d | 0.0% | 0.0% | 0.4% | 0.0% |
| ls_global_z | 2.3% | 2.8% | 2.8% | 2.8% |
| ls_top_z | 19.7% | 24.0% | 24.0% | 24.0% |
| basis_z7d | 0.0% | 0.0% | 0.0% | 0.0% |
| basis_chg4h | 0.1% | 0.0% | 0.0% | 0.0% |
| hour_sin | 0.0% | 0.0% | 0.0% | 0.0% |
| hour_cos | 0.0% | 0.0% | 0.0% | 0.0% |
| wd_sin | 0.0% | 0.0% | 0.0% | 0.0% |
| wd_cos | 0.0% | 0.0% | 0.0% | 0.0% |
| sess_asia | 0.0% | 0.0% | 0.0% | 0.0% |
| sess_europe | 0.0% | 0.0% | 0.0% | 0.0% |
| sess_us | 0.0% | 0.0% | 0.0% | 0.0% |
| rvol_z | 0.0% | 0.0% | 0.0% | 0.0% |
| ret_24h | 0.0% | 0.0% | 0.0% | 0.0% |
| er24 | 0.0% | 0.0% | 0.0% | 0.0% |

## Notes

- Features are known at bar close (index = bar open); entry is the next bar's open; barriers use real prices.
- `liq_*` features are a modeled map (OI-delta heuristic), not observed liquidations.
- A bar can appear under several triggers (one row per bar and trigger); events lacking h future bars are unlabelled (NaN).
- NaN is kept as 'no signal' (A1.4); no interpolation or imputation was applied.
