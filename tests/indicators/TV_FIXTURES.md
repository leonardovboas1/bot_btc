# TradingView fixtures — how to export reference values

TradingView values cannot be downloaded automatically. Export them once by hand so that
tests can assert exact equality between `btclab.indicators` and TradingView.

Target chart: **BINANCE:BTCUSDT.P** (Binance USDT-M perpetual), timeframe **4h**, timezone **UTC**
(Chart settings → Symbol → Timezone = UTC).

## 1. Load the full history

Recursive indicators (EMA/RMA/RSI, the Gaussian filter, STC, Supertrend, PSAR) depend on the
first bar the script sees. TradingView starts each script on the first *loaded* bar.

1. Open the chart and press `Alt+G` (Go to). Jump to **2019-09-08**, the first BTCUSDT.P bar, so that
   all history from the listing is loaded. If your plan limits the number of bars, note the first bar
   time shown in the exported CSV. The test will then run our code from that same bar.
2. Do not change the session or extended-hours settings. Crypto has a 24/7 session.

## 2. Add the fixture script

Pine Editor → New → paste the script below → **Save** → **Add to chart**. It plots the raw
`ta.*` values and the two user scripts' key series. `display.data_window` keeps them off the chart,
but they are still included in the export.

```pine
//@version=6
indicator("btclab fixtures", overlay=false, max_bars_back=5000)
dw = display.data_window
plot(ta.ema(close, 9),      "ema9",   display=dw)
plot(ta.ema(close, 200),    "ema200", display=dw)
plot(ta.rma(close, 14),     "rma14",  display=dw)
plot(ta.sma(close, 20),     "sma20",  display=dw)
plot(ta.wma(close, 14),     "wma14",  display=dw)
plot(ta.hma(close, 21),     "hma21",  display=dw)
plot(ta.rsi(close, 14),     "rsi14",  display=dw)
plot(ta.atr(14),            "atr14",  display=dw)
plot(ta.tr(true),           "tr",     display=dw)
plot(ta.stoch(close, high, low, 14), "stoch14", display=dw)
plot(ta.cci(hlc3, 20),      "cci20",  display=dw)
plot(ta.mfi(hlc3, 14),      "mfi14",  display=dw)
[m, s, h] = ta.macd(close, 12, 26, 9)
plot(m, "macd", display=dw), plot(s, "macd_signal", display=dw)
[st, dir] = ta.supertrend(3, 10)
plot(st, "supertrend", display=dw), plot(dir, "st_dir", display=dw)
[dip, dim, adx] = ta.dmi(14, 14)
plot(dip, "plus_di", display=dw), plot(dim, "minus_di", display=dw), plot(adx, "adx", display=dw)
plot(ta.sar(0.02, 0.02, 0.2), "psar", display=dw)
plot(ta.pivothigh(high, 5, 5), "pivothigh55", display=dw)   // reported at the confirmation bar
plot(ta.pivotlow(low, 5, 5),   "pivotlow55",  display=dw)
r = ta.rsi(close, 14)
k = ta.sma(ta.stoch(r, r, r, 14), 3)
plot(k, "stochrsi_k", display=dw), plot(ta.sma(k, 3), "stochrsi_d", display=dw)
// higher timeframe with lookahead OFF; compare with request_security_htf(df, "D", ...)
plot(request.security(syminfo.tickerid, "D", ta.ema(close, 10), lookahead=barmerge.lookahead_off), "htf_d_ema10", display=dw)
```

Then add the two original user indicators **unchanged** and with default inputs:

* **Gaussian Channel [DW]** (poles 4, period 144, mult 1.414, both modes off). Its plots are
  exported as "Filter", "Filtered True Range High Band" and "Low Band" (names depend on the version).
  Export a second time with *Reduced Lag* and *Fast Response* switched on, if possible.
* **VuManChu Cipher B + Divergences**. Enable *Show Stoch*, *Show RSI*, *Show MFI*, *Show Schaff TC*
  and *Show Sommi flag/diamond* so that those series are plotted, and therefore exported.

## 3. Export

Chart menu (top-left `☰` / the layout name) → **Export chart data…** → choose the layout,
**Time format: UNIX timestamp** → *Export*. A paid plan is needed for this menu.

Save the files as follows:

```
tests/indicators/fixtures/tv_BTCUSDTP_4h_base.csv        # btclab fixtures script
tests/indicators/fixtures/tv_BTCUSDTP_4h_gaussian.csv     # Gaussian Channel defaults
tests/indicators/fixtures/tv_BTCUSDTP_4h_gaussian_lag_fast.csv
tests/indicators/fixtures/tv_BTCUSDTP_4h_cipherb.csv
```

Each CSV has `time,open,high,low,close,Volume,<plot titles...>`. Leave the column titles as they are.
The fixture test maps them by name, and you only need to list them in `fixtures/README` if you
renamed anything.

## 4. What the tests will assert (once the files exist)

* OHLC in the CSV must equal `data/raw/um_futures/klines_4h.parquet`. This proves both use the same
  feed and timestamps (TradingView time = bar open, UTC).
* Run our indicator from the CSV's first bar. Pure-window indicators (SMA, WMA, stoch, CCI, highest,
  pivots) must match exactly (`rtol 1e-12`). Recursive ones must match to `rtol 1e-9`. Pine
  prints only about 8 significant digits per value, so the effective tolerance is the export
  precision.
* Cipher B: wt1/wt2/RSI/MFI/Stoch/STC series must match. Signal shapes (circles, divergences) are
  plotted by TradingView at the pivot bar (`offset=-2`), and ours are on the confirmation bar, so
  the test shifts them by 2. The **Sommi diamond is expected to differ**: the original requests
  Heikin-Ashi HTF candles with `barmerge.lookahead_on`, which leaks future data on historical bars,
  and our port uses lookahead OFF on purpose.
* The last bar of the export is usually still open, so drop it.
