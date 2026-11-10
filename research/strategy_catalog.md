# BTC Signal Catalog (free tools only)

Compiled 2026-11-10. 66 entries. Web research was limited (a handful of searches plus one fetched Cipher B source); formulas for classic indicators are from standard definitions (Wilder, Appel, Bollinger, etc.) and should be unit-tested against TA-Lib / pandas-ta before use. Where a URL was not directly verified in this session it is marked `(unverified)`.

## 0. Global conventions and traps

Notation: `EMA(x,n)` Wilder/standard exponential (alpha=2/(n+1)); `RMA(x,n)` Wilder (alpha=1/n); `SMA`; `TR=max(H-L,|H-C1|,|L-C1|)`; `ATR=RMA(TR,n)`; `hlc3=(H+L+C)/3`; `[k]` = k bars ago; `cross_up(a,b)`: a[1]<=b[1] and a>b.

**Execution rules for a non-repainting backtest (apply to every entry):**
1. Signal computed on bar t close -> fill at open of bar t+1 (or close t with explicit slippage). Never fill at the close that created the signal if the indicator uses that close... (it is only acceptable if you truly trade the close).
2. Pivot/fractal/divergence signals are confirmed `right` bars late (Cipher B uses 2 right bars; ta.pivothigh(L,R) is confirmed at bar t+R but plotted at t). Shift the signal to the confirmation bar.
3. Higher-timeframe data (Pine `request.security`): with `lookahead_on` and no `[1]` offset, the HTF bar's final value leaks into earlier LTF bars. In Python, resample HTF, compute, shift by one HTF bar, then forward-fill.
4. Kernel regression / Nadaraya-Watson envelopes (non-causal "endpoint" mode) redraw history. Use only the causal one-sided estimate and trade it as-is.
5. Zigzag, swing-structure, order-block and FVG "mitigation" logic often labels bars retroactively. Only the confirmation bar is tradable.
6. Renko / Heikin Ashi / Range bars: HA is deterministic from OHLC (OK), but HA close is not a tradable price; fill on real open.
7. Cross-exchange: tick-size, perp vs spot, funding timestamps (00/08/16 UTC) matter. Daily bars: use UTC close consistently.
8. Multiple testing: every parameter grid multiplies false-discovery risk. Use walk-forward, deflated Sharpe (Bailey & López de Prado), and report the number of trials. BTC has essentially ~1 regime-rich 15-year path; N effective is small.
9. Costs: assume >= 5-10 bp taker fee plus 2-5 bp slippage per side on majors; funding cost for perps (about 0.01% / 8h baseline, can be much higher).

**Family legend:** T trend, M momentum, MR mean reversion, V volatility, VOL volume, D derivatives, OC on-chain, P pattern, ML machine learning.

---

## 1. Summary table

Data: **O** = OHLCV only (free via Binance/Bybit/OKX/Kraken REST, yfinance, ccxt). Repaint risk: **L** low / **M** medium / **H** high.

| # | Name | Family | Data | Repaint/Lookahead risk | Best use |
|---|------|--------|------|-----|------|
| 1 | VuManChu Cipher B (+Divergences) | M/MR/composite | O | H (HTF security, fractal divs) | Reversal confluence |
| 2 | WaveTrend (LazyBear) | M/MR | O | L | OB/OS cross |
| 3 | RSI levels | M/MR | O | L | MR or trend-regime |
| 4 | RSI divergence | MR/P | O | H (late pivots) | Reversal |
| 5 | Stochastic RSI | MR | O | L | Timing |
| 6 | Gaussian Channel (DonovanWall) | T | O | L (Pine v4 original uses security option) | Trend |
| 7 | MACD | T/M | O | L | Trend/momentum |
| 8 | Supertrend | T | O | L | Trend flip |
| 9 | Ichimoku | T | O | M (Chikou/forward shift) | Trend |
| 10 | Bollinger Bands | V/MR | O | L | MR / breakout |
| 11 | Keltner Channel | V/T | O | L | Breakout |
| 12 | BB/KC TTM Squeeze | V | O | L | Vol expansion |
| 13 | ADX/DMI | T | O | L | Trend strength + direction |
| 14 | Parabolic SAR | T | O | L | Stop-and-reverse |
| 15 | EMA/SMA crosses | T | O | L | Trend |
| 16 | EMA ribbon | T | O | L | Trend |
| 17 | Hull MA | T | O | L | Fast trend |
| 18 | VWAP (session) | VOL/MR | O | L | Intraday |
| 19 | Anchored VWAP | VOL/T | O | L (anchor choice is hindsight) | Support/resistance |
| 20 | Donchian / Turtle | T | O | L | Breakout trend |
| 21 | Schaff Trend Cycle | M | O | L | Cycle timing |
| 22 | CCI | M/MR | O | L | OB/OS |
| 23 | Williams %R | MR | O | L | OB/OS |
| 24 | MFI | VOL/MR | O | L | Volume RSI |
| 25 | OBV | VOL | O | L | Confirmation |
| 26 | CMF | VOL | O | L | Accumulation |
| 27 | Heikin Ashi | T | O | L | Filter |
| 28 | Chandelier Exit | T/V | O | L | Trailing stop/flip |
| 29 | UT Bot Alerts | T | O | L | ATR trailing flip |
| 30 | SSL Channel | T | O | L | Trend |
| 31 | QQE | M | O | L | RSI smoothing/trend |
| 32 | Squeeze Momentum (LazyBear) | V/M | O | L | Squeeze fire |
| 33 | Range Filter | T | O | L | Trend |
| 34 | Nadaraya-Watson envelope | MR/ML | O | H | Do not use endpoint variant |
| 35 | Lorentzian Classification | ML | O | M | kNN direction |
| 36 | Fair Value Gaps | P/SMC | O | M | Retest entries |
| 37 | Order Blocks | P/SMC | O | H | Retest entries |
| 38 | BOS/CHoCH market structure | P/SMC | O | H | Structure |
| 39 | Volume Profile (POC/VAH/VAL) | VOL | O (better with tick) | L/M | Levels |
| 40 | Funding rate | D | funding (free) | L | Contrarian |
| 41 | Open interest | D | OI (free, limited history) | L | Confirmation |
| 42 | Long/short ratio | D | LS ratio (free, 30d-1y history) | L | Contrarian |
| 43 | Pi Cycle Top/Bottom | OC-ish | O (daily) | L | Macro top |
| 44 | MVRV / MVRV-Z | OC | on-chain (partly free) | L | Macro |
| 45 | Time-series momentum | M | O | L | Core trend |
| 46 | Stochastic oscillator | MR | O | L | OB/OS |
| 47 | Aroon | T | O | L | Trend |
| 48 | Vortex | T | O | L | Trend |
| 49 | ROC / Momentum | M | O | L | TSMOM variant |
| 50 | Fisher Transform | MR | O | L | Turning points |
| 51 | Elder Ray / Force Index | M/VOL | O | L | Pressure |
| 52 | Connors RSI / RSI(2) | MR | O | L | Short-term MR |
| 53 | KAMA / adaptive MA | T | O | L | Adaptive trend |
| 54 | Volatility breakout (Larry Williams) | V/T | O | L | Intraday/daily |
| 55 | Opening range / session breakout | T | O (intraday) | L | Intraday |
| 56 | Mayer Multiple / 200DMA | T/MR | O | L | Macro |
| 57 | SOPR / NUPL / Puell / Realized price | OC | on-chain (mostly paid) | L (revisions) | Macro |
| 58 | Liquidation cascades | D | liquidations (free live only) | L | Reversal |
| 59 | Futures basis / premium | D | spot+futures (free) | L | Sentiment |
| 60 | Taker buy/sell ratio, CVD | VOL/D | taker volume (free from Binance klines) | L | Flow |
| 61 | Candlestick patterns | P | O | L | Weak |
| 62 | Fibonacci retracement / pivots | P | O | H (swing definition) | Levels |
| 63 | Trendline / chart patterns | P | O | H | Discretionary |
| 64 | Ehlers filters (Super Smoother, ITrend, Cyber Cycle) | T/M | O | L | DSP trend |
| 65 | Vol-target / regime filter (HMM, vol percentile) | V/ML | O | M (HMM smoothing) | Overlay |
| 66 | Halving-cycle seasonality & calendar effects | OC/P | O | L (n=4 cycles!) | Narrative |
| 67 | Fear & Greed Index | sentiment | free API (alternative.me) | L | Contrarian |
| 68 | Gradient-boosted / ML on feature stack | ML | O (+optional) | M | Meta-model |

---

## 2. Detailed entries

Format: **Formula** / **Params** (default; search range) / **Rules** / **Data** / **Sources** / **Evidence** / **Repaint**.

### 1. VuManChu Cipher B + Divergences (open-source re-creation of paid Market Cipher B)
- **Family:** momentum/mean-reversion composite (oscillator panel).
- **Formula** (from a community port of the TradingView script, https://gist.github.com/itsXactlY/a7ddc99055e942006dd945d5c04dd759, fetched; it self-describes as an imitation of Market Cipher B so treat as approximate):
  ```
  ap  = hlc3
  esa = EMA(ap, 9);  de = EMA(|ap-esa|, 9)
  ci  = (ap-esa)/(0.015*de)
  wt1 = EMA(ci, 12);  wt2 = SMA(wt1, 3);  wtVwap = wt1 - wt2
  mfi_area = SMA(((close-open)/(high-low))*175, 60) - 2.5   # "RSI+MFI" green/red area (period 60, mult 175, yoffset 2.5)
  rsi = RSI(close,14);  stochrsi: k = SMA(stoch(RSI,RSI,RSI,14),3); d = SMA(k,3)
  WT levels: OB 53/60/100, OS -53/-60/-100
  ```
- **Signals:**
  - Buy circle: `cross_up(wt1,wt2) and wt2 <= -53`. Sell circle: `cross_down and wt2 >= 53`.
  - Divergence (WT or RSI): fractal pivots (top: `src[4]<src[2], src[3]<src[2], src[2]>src[1], src[2]>src[0]`, confirmed 2 bars late). Regular bull: price lower low, indicator higher low; bear mirror. Hidden divergences optional. Min WT level for bull div -65 / bear 45; RSI 30/60.
  - Gold buy: bullish (WT or RSI) divergence AND prior WT low <= -100 AND current wt2 > -100 AND RSI at the earlier fractal < 30.
  - Sommi flag (bull): `mfi_area>0 and wt2<0 and cross_up and HTF(720m) vwap-of-WT > 0`; bear mirror (`mfi<0, wt2>0, cross_down, HTF vwap<0`).
  - Sommi diamond (bull): `wt2<=0 and cross_up and both HTF (60m,240m) Heikin Ashi candles green`; bear mirror.
- **Params:** WT chan 9 (6-14), avg 12 (9-21), malen 3; OB 53 (45-70), OS -53; MFI period 60 (30-100), mult 175 (scale only); RSI 14. HTF values 60/240/720 min.
- **Rules (typical):** Long on buy circle / gold / bull flag, ideally with MFI area green; short on mirror. Exit on opposite circle, WT cross back through zero, or fixed ATR stop (1.5-3 ATR). Many users treat green dot at wt2<-53 on 4h as the base long.
- **Data:** OHLCV, free.
- **Sources:** TradingView "VuManChu Cipher B + Divergences" (MPL-2.0); https://gainium.io/blog/vumanchu-cipher-b-indicator-explained; https://www.luxalgo.com/library/indicator/T85iFvQj-vumanchu-cipher-b-divergences-strategy ; https://community.gainium.io/t/vumanchu-cipher-b/2668. Exact original Pine differs in details between versions; verify against source.
- **Evidence:** None peer-reviewed. Vendor/blog claims only. Strong skepticism: composite of 5+ discretionary signals invites forking-paths; there are many signal combinations to cherry-pick. Underlying WT is just a smoothed CCI.
- **Repaint:** HIGH. (a) `security(..., lookahead_on)` for Sommi HTF components leaks future HTF data (a commenter in the fetched source notes heavy repainting). (b) Fractal divergence is confirmed 2 bars after the pivot but plotted at the pivot. (c) HA candles from HTF are unfinished on current bar. In Python: compute HTF with shift(1), delay divergences by 2 bars.

### 2. WaveTrend Oscillator (LazyBear)
- **Family:** momentum / mean reversion.
- **Formula:** as in #1 (n1=10 channel, n2=21 average in LazyBear defaults; Cipher uses 9/12). wt1 = EMA(ci,n2); wt2 = SMA(wt1,4).
- **Params:** n1 10 (6-20), n2 21 (9-34), signal MA 4 (2-6), OB +60/+53, OS -60/-53 (search 40-80).
- **Rules:** Long: wt1 crosses above wt2 while wt2 < OS. Short: cross down while wt2 > OB. Exit: opposite cross, or wt crossing zero. Trend variant: take only cross in direction of 200 EMA.
- **Data:** OHLCV.
- **Source:** TradingView "WaveTrend Oscillator [WT]" by LazyBear (search name); https://www.tradingview.com/script/2KE8wTuF-Indicator-WaveTrend-Oscillator-WT/ (unverified ID).
- **Evidence:** No rigorous BTC evidence found. Like CCI-based OB/OS; on trending BTC, OS cross longs in downtrends lose.
- **Repaint:** Low (causal EMAs). Cross confirmed at bar close only.

### 3. RSI (levels)
- **Family:** momentum / mean reversion.
- **Formula:** `U=max(C-C1,0); D=max(C1-C,0); RS=RMA(U,n)/RMA(D,n); RSI=100-100/(1+RS)`.
- **Params:** n 14 (2-30); OS 30 (10-40), OB 70 (60-90). Regime form: bull regime RSI range 40-80, bear 20-60 (Cardwell).
- **Rules:** MR: long when RSI crosses up through OS, short when crosses down through OB; exit at 50. Trend/regime: long if RSI>50 (or >55) and rising; BTC daily RSI(14)>70 often persists.
- **Data:** OHLCV.
- **Source:** Wilder, New Concepts in Technical Trading Systems (1978); https://www.investopedia.com/terms/r/rsi.asp.
- **Evidence:** Mixed. Classic MR OB/OS shorts on BTC usually lose in bull trends; Hudson & Urquhart tested RSI-class rules inside ~15k-rule universe (some significant, but not specifically isolated). Treat as unproven.
- **Repaint:** Low.

### 4. RSI Divergence (regular / hidden)
- **Family:** mean-reversion pattern.
- **Algorithm:** find pivot lows in price (`pivotlow(L=5,R=5)`) and in RSI; bullish regular if price LL & RSI HL within lookback (5-60 bars) between consecutive pivots; bearish mirror; hidden bull = price HL & RSI LL (trend continuation).
- **Params:** pivot L/R 3-10 (default 5/5 in TradingView built-in "RSI Divergence Indicator"); max lookback 60 (20-100); RSI 14; require RSI<30-45 for bull.
- **Rules:** Long at confirmation bar of bull div; stop below the lower low; exit on RSI>60 or opposite div. Short mirror.
- **Data:** OHLCV.
- **Source:** TradingView built-in "RSI Divergence Indicator"; Cipher B code (#1).
- **Evidence:** Anecdotal. Because signal lags R bars, expected edge is smaller than charts suggest.
- **Repaint/lookahead:** HIGH if plotted at pivot bar; signal only knowable R bars later. Enter at bar t+R open.

### 5. Stochastic RSI
- **Family:** mean reversion.
- **Formula:** `stoch = (RSI - min(RSI,m))/(max(RSI,m)-min(RSI,m))*100; K=SMA(stoch,3); D=SMA(K,3)`.
- **Params:** RSI 14 (7-21), stoch len 14 (7-21), K 3, D 3; levels 20/80 (10-30 / 70-90).
- **Rules:** Long: K crosses above D below 20; short: K crosses below D above 80; Exit: opposite cross or K at 50. Combined with trend filter (EMA200).
- **Data:** OHLCV. **Source:** Chande & Kroll, The New Technical Trader (1994); TradingView built-in.
- **Evidence:** None specific. Very noisy (oscillates 0-100 constantly) -> high turnover, fee drag.
- **Repaint:** Low.

### 6. Gaussian Channel (DonovanWall)
- **Family:** trend.
- **Formula:** N-pole Gaussian low-pass filter on hlc3 with cutoff set by period; `alpha` from `beta=(1-cos(2pi/period))/(2^(1/poles)-1); alpha=-beta+sqrt(beta^2+2beta)`; filter: `f_n = alpha^n*src + sum_{k=1..n}(-1)^(k+1) C(n,k)(1-alpha)^k f[k]` (poles 1-9, default 4). Channel = filter +/- mult*filtered TR (default mult 1.414, same filter applied to TR). Optional "reduced lag" (shift by lag=(period-1)/(2*poles)) and "fast response" (average of N-pole and 1-pole).
- **Params:** poles 4 (1-6), period 144 (30-200), mult 1.414 (1-2.5), reduced lag off, fast response off. Designed for daily.
- **Rules:** Long when price closes above upper band (filter green, rising); exit when close drops below the filter (or below upper band). Short mirror (rarely used; mostly long/flat for BTC).
- **Data:** OHLCV.
- **Source:** TradingView "Gaussian Channel [DW]" by DonovanWall; Ehlers, Rocket Science for Traders (Gaussian filters).
- **Evidence:** Popular in BTC community (often quoted as holding through bull runs); no peer-reviewed support. Equivalent to a slow vol-band trend system, so expectation is similar to Donchian/MA trend models (see #20, #45).
- **Repaint:** Low if you use `reduced lag` as a causal formula; no lookahead, but `security` timeframe option in the original can leak when set to HTF.

### 7. MACD
- **Family:** trend / momentum.
- **Formula:** `macd=EMA(C,12)-EMA(C,26); signal=EMA(macd,9); hist=macd-signal`.
- **Params:** fast 12 (5-20), slow 26 (20-60), signal 9 (3-15); crypto users use 8/21/5 or 34/144.
- **Rules:** Long: macd crosses above signal (stronger if below zero); or macd > 0. Short mirror. Exit on opposite cross. Zero-line trend filter.
- **Data:** OHLCV. **Source:** Appel (1979); https://www.investopedia.com/terms/m/macd.asp.
- **Evidence:** Included in Hudson & Urquhart / other rule universes; mixed. Whipsaw heavy in range.
- **Repaint:** Low.

### 8. Supertrend
- **Family:** trend.
- **Formula:** `mid=hl2; ub=mid+m*ATR(n); lb=mid-m*ATR(n)`; final_ub = ub if ub<final_ub[1] or C[1]>final_ub[1] else final_ub[1]; final_lb mirrored; trend flips up when C > final_ub[1], down when C < final_lb[1]; supertrend line = final_lb in up trend else final_ub.
- **Params:** ATR n 10 (7-30), m 3.0 (1.5-5). ATR via RMA.
- **Rules:** Long on flip to up; short on flip to down (stop-and-reverse). Exit = flip.
- **Data:** OHLCV. **Source:** Olivier Seban; TradingView built-in "Supertrend".
- **Evidence:** Retail backtests only. Behaves like a volatility-adaptive trailing stop; expect profitable on strong trends, losing in chop.
- **Repaint:** Low. Note TradingView's `ta.supertrend` ATR uses RMA; some ports use SMA (differences).

### 9. Ichimoku Kinko Hyo
- **Family:** trend.
- **Formula:** Tenkan=(HH9+LL9)/2; Kijun=(HH26+LL26)/2; SenkouA=(Tenkan+Kijun)/2 shifted +26; SenkouB=(HH52+LL52)/2 shifted +26; Chikou=close shifted -26.
- **Params:** 9/26/52 (original, designed for 6-day week). Crypto variants 20/60/120 (7-day). Search tenkan 5-20, kijun 20-60, senkouB 40-130.
- **Rules:** Long: close above cloud (max(SenkouA,SenkouB) as plotted at t, i.e., values computed 26 bars ago), Tenkan>Kijun, optionally Chikou above price 26 bars ago. Short mirror. Exit: close back into cloud / Tenkan-Kijun cross.
- **Data:** OHLCV. **Source:** Hosoda; https://www.investopedia.com/terms/i/ichimokuchart.asp.
- **Evidence:** Some academic equity/FX studies, little BTC-specific rigorous evidence found.
- **Repaint/lookahead:** MEDIUM. Cloud at bar t must be taken from values computed at t-26 (shift forward). Using the unshifted Senkou spans leaks the future. Chikou is plotted in the past; using it as a signal at t requires comparing close[t] with close[t-26] only.

### 10. Bollinger Bands
- **Family:** volatility / mean reversion or breakout.
- **Formula:** `mid=SMA(C,n); sd=stdev(C,n) (population); up=mid+k*sd; lo=mid-k*sd; %B=(C-lo)/(up-lo); BW=(up-lo)/mid`.
- **Params:** n 20 (10-50), k 2.0 (1.5-3).
- **Rules:** MR: long when close < lo then closes back inside; short at up; exit at mid. Breakout: long close>up (with BW expansion), exit at mid cross.
- **Data:** OHLCV. **Source:** Bollinger, Bollinger on Bollinger Bands (2001).
- **Evidence:** Crypto papers on BB are mostly weak/mixed; BTC trends "walk the band" so MR shorts can be lethal.
- **Repaint:** Low.

### 11. Keltner Channel
- **Family:** volatility / trend.
- **Formula:** `mid=EMA(C,n); up=mid+m*ATR(n2); lo=mid-m*ATR(n2)` (original Keltner used SMA of typical price and daily range).
- **Params:** n 20 (10-50), ATR n 10 (10-20), m 2.0 (1-3).
- **Rules:** Breakout long: close>up; short close<lo; exit at mid. Pullback: trade toward mid in trend.
- **Data:** OHLCV. **Source:** Chester Keltner (1960); Linda Raschke version.
- **Evidence:** None BTC-specific found. **Repaint:** Low.

### 12. TTM Squeeze (BB inside KC)
- **Family:** volatility.
- **Formula:** squeeze_on = `BB_lo > KC_lo and BB_up < KC_up` (BB 20/2.0, KC 20/1.5 ATR). Momentum = linear-regression value of `close - mean(avg(HH20,LL20), SMA20)`. See #32.
- **Rules:** Long when squeeze turns off (fires) and momentum >0 (and rising); short when momentum <0. Exit: momentum color change or ATR trail.
- **Data:** OHLCV. **Source:** John Carter, Mastering the Trade; https://school.stockcharts.com/doku.php?id=technical_indicators:ttm_squeeze (unverified path).
- **Evidence:** Anecdotal. **Repaint:** Low (regression is on the window ending at t).

### 13. ADX / DMI
- **Family:** trend strength + direction.
- **Formula:** `upMove=H-H1; dnMove=L1-L; +DM = upMove if upMove>dnMove and upMove>0 else 0; -DM mirror; +DI=100*RMA(+DM,n)/ATR; -DI likewise; DX=100*|+DI - -DI|/(+DI + -DI); ADX=RMA(DX,n)`.
- **Params:** n 14 (7-30); ADX threshold 20-25 (15-35).
- **Rules:** Long: +DI crosses above -DI and ADX>25 (or rising); short mirror. Exit: opposite DI cross or ADX falling from >40. As a filter: trade trend rules only when ADX>thr, MR rules when below.
- **Data:** OHLCV. **Source:** Wilder (1978).
- **Evidence:** Used as filter in many studies; value-add unverified in BTC. **Repaint:** Low.

### 14. Parabolic SAR
- **Family:** trend. **Formula:** `SAR_t = SAR_{t-1} + AF*(EP - SAR_{t-1})`; AF starts 0.02, +0.02 each new extreme to max 0.2; flips when price crosses SAR; SAR cannot enter last two bars' range.
- **Params:** step 0.02 (0.005-0.05), max 0.2 (0.1-0.4).
- **Rules:** Stop-and-reverse: long when SAR flips below price, short when above.
- **Data:** OHLCV. **Source:** Wilder (1978).
- **Evidence:** Poor in ranges; crypto 24/7 gives many whipsaws. **Repaint:** Low.

### 15. EMA / SMA crosses
- **Family:** trend. **Formula:** `fast=MA(C,a); slow=MA(C,b)`.
- **Params:** classics 50/200 (golden/death cross), 20/50, 9/21; search a 5-100, b 20-300 (a<b), EMA vs SMA.
- **Rules:** Long when fast>slow (cross up); short (or flat) when fast<slow. Exit on reverse cross. Variants: price vs MA(200), MA with % band filter (e.g. 1-3%) to cut whipsaws.
- **Data:** OHLCV. **Source:** Brock, Lakonishok & LeBaron (1992); Hudson & Urquhart (2021) tested these across ~15k rules.
- **Evidence:** Among the best-supported classes for BTC: Detzel et al. (2018, "Bitcoin: Learning, Predictability and Profitability via Technical Analysis") report 5-100 day MA rules predictive in and out of sample (as summarized by search results; verify). Recent skeptic study 2022-23 (Li 2023) finds none profitable post data-snooping adjustments. Regime dependent.
- **Repaint:** Low.

### 16. EMA Ribbon
- **Family:** trend. **Formula:** set of EMAs, e.g. 8,13,21,34,55,89 (Fibonacci) or 20-200 by 20.
- **Rules:** Long when all EMAs ordered (short above long) and price above fastest; ribbon expansion = strong trend. Short mirror. Exit when fast EMA crosses mid EMA or ordering breaks. Score = fraction of EMAs ordered.
- **Params:** count 6-10, base lengths 5-200. **Data:** OHLCV. **Source:** Guppy Multiple Moving Average (Daryl Guppy). **Evidence:** anecdotal. **Repaint:** Low.

### 17. Hull Moving Average
- **Family:** trend. **Formula:** `HMA(n)=WMA(2*WMA(C,n/2) - WMA(C,n), sqrt(n))`.
- **Params:** n 21 (9-100). Variants: EHMA, THMA.
- **Rules:** Long when HMA slope turns up (HMA > HMA[2]); short when slope turns down; or HMA/EMA cross. Lower lag, more overshoot.
- **Data:** OHLCV. **Source:** Alan Hull (2005), alanhull.com. **Evidence:** none specific. **Repaint:** Low (slope flip is lagging but causal).

### 18. VWAP (session)
- **Family:** volume / mean reversion. **Formula:** `VWAP=cumsum(hlc3*V)/cumsum(V)` reset at session start (00:00 UTC for crypto; define consistently). Bands +/- k*stdev.
- **Params:** reset daily/weekly/monthly; bands 1-3 sigma.
- **Rules:** Intraday MR: long at -2 sigma band reclaim; trend: long when price holds above VWAP with rising slope; short mirror. Exit at VWAP.
- **Data:** OHLCV (use 1-5 min bars). **Source:** TradingView built-in VWAP. **Evidence:** none for BTC specific; crypto 24/7 has no natural session.
- **Repaint:** Low, but resets at arbitrary boundary change results.

### 19. Anchored VWAP
- **Family:** volume / trend support. **Formula:** same cumulative VWAP from an anchor bar (swing low/high, halving date, ATH, FOMC day, weekly open).
- **Rules:** Long on reclaim/bounce of AVWAP from a major swing low; short on rejection of AVWAP from swing high. Exit: close back beyond AVWAP by ATR.
- **Params:** anchors are discretionary; systematic version: anchor to most recent confirmed pivot (L=R=10-50) or to rolling N-day low/high.
- **Data:** OHLCV. **Source:** Brian Shannon, Technical Analysis Using Multiple Timeframes. **Evidence:** anecdotal.
- **Repaint/lookahead:** anchor selection in hindsight is the main trap (choosing the "obvious" bottom). Use only anchors confirmed at time t.

### 20. Donchian Channel / Turtle breakout
- **Family:** trend. **Formula:** `up=max(H[1..n]); lo=min(L[1..n])` (exclude current bar).
- **Params:** entry n 20 (5-360), exit n 10 (5-180); Turtle System 1 20/10, System 2 55/20. ATR(20) position sizing (N-unit).
- **Rules:** Long when close (or high) > up_entry; short when < lo_entry; exit long at lower exit-channel break. Optional vol targeting.
- **Data:** OHLCV. **Source:** Richard Dennis/Turtles; Zarattini, Pagani, Barbon (2025) "Catching Crypto Trends", SSRN 5209907 (https://abarbon.com/papers/catching-crypto-trends ; https://concretumgroup.com/catching-crypto-trends-a-tactical-approach-for-bitcoin-and-altcoins/).
- **Evidence:** Ensemble of Donchian models (lookbacks 5,10,20,30,60,90,150,250,360 days) with 25% vol targeting: reported BTC Sharpe ~1.58, CAGR ~30%, alpha ~14%/yr vs BTC, Jan 2015-Mar 2025 (from secondary summaries; not verified against the paper). Skepticism: 2015-2025 contains the main BTC secular bull; ensemble smooths lookback overfitting but sample is short. Probably the best-documented BTC rule family.
- **Repaint:** Low. Make sure channel excludes current bar; fill at breakout level only if intrabar stop orders are simulated with realistic slippage, else next open.

### 21. Schaff Trend Cycle (STC)
- **Family:** momentum. **Formula:** `m=EMA(C,fast)-EMA(C,slow)`; stoch1 of m over cycle with smoothing factor 0.5; stoch2 of smoothed result; output 0-100.
- **Params:** cycle 10 (5-20), fast 23 (12-30), slow 50 (26-80), factor 0.5. Levels 25/75.
- **Rules:** Long when STC crosses up through 25; short when crosses down through 75. Cipher B includes STC panel (off by default).
- **Data:** OHLCV. **Source:** Doug Schaff (1999); TradingView STC scripts. **Evidence:** anecdotal. **Repaint:** Low.

### 22. CCI
- **Family:** momentum / MR. **Formula:** `CCI=(TP-SMA(TP,n))/(0.015*MD)`, TP=hlc3, MD=mean absolute deviation.
- **Params:** n 20 (10-50); levels +/-100 (80-200).
- **Rules:** Long on cross up through -100 (MR) or through +100 (trend breakout); short on cross down through +100 / -100. Exit at 0.
- **Data:** OHLCV. **Source:** Donald Lambert (1980). **Evidence:** none specific. **Repaint:** Low.

### 23. Williams %R
- **Family:** MR. **Formula:** `%R=-100*(HH_n - C)/(HH_n - LL_n)`.
- **Params:** n 14 (5-30); OS -80, OB -20.
- **Rules:** Long cross up through -80; short cross down through -20; exit at -50. Inverted Stochastic. **Data:** OHLCV. **Source:** Larry Williams. **Evidence:** none. **Repaint:** Low.

### 24. Money Flow Index (MFI)
- **Family:** volume / MR. **Formula:** `TP=hlc3; raw=TP*V; pos=sum(raw where TP>TP1, n); neg=sum(raw where TP<TP1, n); MFI=100-100/(1+pos/neg)`.
- **Params:** n 14 (7-30); 20/80 (10-30/70-90).
- **Rules:** like RSI; long cross up from <20, short cross down from >80; divergence version. Note Cipher B's "MFI" is a different formula (#1).
- **Data:** OHLCV (volume exchange-specific; use aggregated if possible). **Source:** Quong & Soudack. **Evidence:** none. **Repaint:** Low.

### 25. On-Balance Volume (OBV)
- **Family:** volume. **Formula:** `OBV_t=OBV_{t-1}+sign(C-C1)*V`.
- **Params:** smoothing MA 20 (10-100).
- **Rules:** Long when OBV crosses above its MA / breaks N-bar high before price (accumulation); short mirror. Confirmation filter for breakouts. **Data:** OHLCV. **Source:** Granville (1963). **Evidence:** Brock-type studies include volume rules; BTC-specific unclear. Wash/fake volume on some exchanges is a concern (use Binance/Coinbase/Kraken). **Repaint:** Low.

### 26. Chaikin Money Flow (CMF)
- **Family:** volume. **Formula:** `MFM=((C-L)-(H-C))/(H-L); CMF=sum(MFM*V,n)/sum(V,n)`.
- **Params:** n 20 (10-50); threshold +/-0.05 (0-0.15).
- **Rules:** Long when CMF>+0.05 and rising; short < -0.05. Filter or divergence. **Data:** OHLCV. **Source:** Marc Chaikin. **Evidence:** none. **Repaint:** Low.

### 27. Heikin Ashi
- **Family:** trend filter. **Formula:** `HA_C=(O+H+L+C)/4; HA_O=(HA_O1+HA_C1)/2; HA_H=max(H,HA_O,HA_C); HA_L=min(L,HA_O,HA_C)`.
- **Rules:** Long when HA candle green with no lower wick (strong) for k consecutive; exit on first red or on wicked candle. Used as HTF filter in Cipher B diamonds.
- **Params:** k 1-3. **Data:** OHLCV. **Source:** Japanese technique; Dan Valcu. **Evidence:** smoothing without real edge; backtests on HA prices are biased since HA close is not tradable. **Repaint:** Low, fill-price bias HIGH (use real next open).

### 28. Chandelier Exit
- **Family:** trend / volatility stop. **Formula:** `long_stop=HH_n - m*ATR(n)`; `short_stop=LL_n + m*ATR(n)`; direction flips when close crosses the opposite stop (ratcheting: long_stop=max(long_stop, long_stop[1]) while close[1]>long_stop[1]).
- **Params:** n 22 (10-50), m 3.0 (1.5-5).
- **Rules:** Buy when close crosses above short_stop; sell when below long_stop. **Data:** OHLCV. **Source:** Chuck LeBeau; TradingView "Chandelier Exit" (everget). **Evidence:** none specific. **Repaint:** Low (TradingView version uses `close` stop flips; some versions use highest(high) incl. current bar).

### 29. UT Bot Alerts
- **Family:** trend (ATR trailing stop). **Formula:** `nLoss=a*ATR(c)`; `xATRTrailingStop_t = C>stop1 and C1>stop1 ? max(stop1, C-nLoss) : C<stop1 and C1<stop1 ? min(stop1, C+nLoss) : C>stop1 ? C-nLoss : C+nLoss`; pos flips on cross of source with stop. Optional Heikin Ashi source.
- **Params:** key value a 1 (0.5-4), ATR period c 10 (5-30).
- **Rules:** Buy: close crosses above trailing stop; sell: below. Stop-and-reverse. **Data:** OHLCV. **Source:** TradingView "UT Bot Alerts" by QuantNomad (adapted from Yo_adriiiiaan); https://www.tradingview.com/script/n8ss8BID-UT-Bot-Alerts/ (unverified ID). **Evidence:** retail only. **Repaint:** Low, but HA-source option gives non-tradable prices.

### 30. SSL Channel
- **Family:** trend. **Formula:** `smaHigh=SMA(H,n); smaLow=SMA(L,n); Hlv = C>smaHigh ? 1 : C<smaLow ? -1 : Hlv[1]; ssl_down = Hlv<0 ? smaHigh : smaLow; ssl_up = Hlv<0 ? smaLow : smaHigh`.
- **Params:** n 10-14 (5-50). **Rules:** Long when ssl_up crosses above ssl_down (close above SMA(H)); short opposite; often paired with baseline & ATR stop (NNFX). **Data:** OHLCV. **Source:** Mladen/ TradingView SSL by ErwinBeckers. **Evidence:** none. **Repaint:** Low.

### 31. QQE (Quantitative Qualitative Estimation)
- **Family:** momentum. **Formula:** `RsiMa=EMA(RSI(C,14),5); ATRRsi=|RsiMa-RsiMa1|; MaAtrRsi=EMA(ATRRsi, 2*14-1); dar=EMA(MaAtrRsi,27)*4.236`; trailing line `FastAtrRsiTL` = ratcheting RsiMa +/- dar.
- **Params:** RSI 14 (7-21), SF 5 (3-10), factor 4.236 (2-5). 
- **Rules:** Long when RsiMa crosses above trailing line (and >50); short opposite. **Data:** OHLCV. **Source:** Roman Ignatov (Mladen); TradingView QQE variants. **Evidence:** anecdotal. **Repaint:** Low (some variants repaint via security).

### 32. Squeeze Momentum Indicator (LazyBear)
- **Family:** volatility / momentum. **Formula:** BB(20, 2.0) vs KC(20, 1.5, TR-based, SMA); `val=linreg(close - avg(avg(HH20,LL20), SMA20), 20, 0)`. Squeeze on = BB inside KC; off = BB outside; "no squeeze" otherwise.
- **Params:** length 20 (10-40), BB mult 2.0 (1.5-2.5), KC mult 1.5 (1-2).
- **Rules:** Long: first bar after squeeze release with val>0 and val > val[1]; short: val<0 and falling. Exit when val crosses zero or momentum bars lose color. **Data:** OHLCV. **Source:** LazyBear TradingView (derived from Carter's TTM). **Evidence:** none rigorous. **Repaint:** Low.

### 33. Range Filter (DonovanWall)
- **Family:** trend. **Formula:** `r = EMA(EMA(|C-C1|, per), per*2-1) * mult` (smoothed average change); filter: `f = C>f1 ? max(f1, C-r) : min(f1, C+r)`; `upward = f>f1` count; bands hi=f+r, lo=f-r.
- **Params:** period 100 (14-200), mult 3.0 (1.5-6).
- **Rules:** Long when filter turns up (price > filter and upward>0) and price above hi band; short on turns down. **Data:** OHLCV. **Source:** TradingView "Range Filter Buy and Sell 5min" by guikroth, Range Filter by DonovanWall. **Evidence:** anecdotal. **Repaint:** Low.

### 34. Nadaraya-Watson Envelope (LuxAlgo)
- **Family:** mean reversion / non-parametric regression.
- **Formula:** `yhat_t = sum_i K(i,h)*C[i] / sum_i K(i,h)` over window of 0..499 bars, Gaussian kernel `K=exp(-(i^2)/(2h^2))`, h~8; envelope = yhat +/- 3*MAE(C-yhat, 499).
- **Params:** h 8 (3-20), mult 3 (2-4), window 500.
- **Rules:** Long when close < lower band (MR), short at upper band; exit at yhat.
- **Data:** OHLCV. **Source:** TradingView "Nadaraya-Watson Envelope" by LuxAlgo (and jdehorty's Nadaraya-Watson Estimator, which has "repaint" toggle). 
- **Evidence:** none; widely noted as repainting.
- **Repaint:** HIGH (documented). The original script computes the kernel using future bars (non-causal) to redraw the last ~500 bars at each update; the arrows appear but disappear. The backtest MUST use a one-sided kernel (only i>=0 backward weights), in which case it's an exponentially/Gaussian-weighted moving average whose edge is just a Bollinger-style band.

### 35. Lorentzian Classification (jdehorty)
- **Family:** ML (approximate k-nearest-neighbors).
- **Algorithm:** features f_1..f_5 each normalized to 0-1 (default: RSI(14,1), WT(10,11), CCI(20,1), ADX(20,2), RSI(9,1)). Label y_t = sign(C[t+4]-C[t]) (4 bars ahead). For each new bar compute Lorentzian distance `d(x,q)=sum_j ln(1+|x_j-q_j|)` to historical bars sampled every 4 bars (chronological spacing), keep k=8 nearest neighbors (largest k hist), sum their labels; prediction >0 => long, <0 => short. Filters: volatility, regime (kernel slope), ADX threshold, optional EMA200/SMA200, kernel regression (Rational Quadratic, lookback 8, rel. weight 8) smoothing.
- **Params:** neighbors 8 (4-30), max bars back 2000 (1000-5000), horizon 4 (2-16), feature set/lengths, kernel lookback 8 (3-20).
- **Rules:** enter on direction change of prediction (after filters); exit after 4 bars (fixed horizon) or on opposite signal.
- **Data:** OHLCV. **Source:** https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/ (unverified ID; the search showed a derived version at /script/m97duZjW); Python port `advanced-ta` on PyPI (pip: advanced-ta, https://pypi.org/project/advanced-ta/). 
- **Evidence:** only the author's TV screenshots. Training label uses the future (4 bars) so only samples >=4 bars old are legit; the Pine code respects this but naive ports leak. In-sample "accuracy" displays are not out-of-sample.
- **Repaint:** MEDIUM. The kernel-regression smoothing line and label building can leak; the "Show Trade Stats" win rate displays are in-sample. Implement with strict `t-4` cutoff.

### 36. Fair Value Gaps (FVG)
- **Family:** pattern / SMC. **Formula:** 3-candle imbalance: bullish FVG at bar t if `L[t] > H[t-2]` (gap zone [H[t-2], L[t]]), bearish if `H[t] < L[t-2]`. Min size filter e.g. > 0.1-0.5 ATR.
- **Rules:** Long when price returns into an untested bullish FVG zone and a bullish reaction candle closes (or limit at zone mid/50%) in an uptrend (HTF structure up); stop below zone; target: prior high / 2R. Short mirror. Mitigated when price closes through.
- **Params:** min size 0.25 ATR (0-1), expiry 50-200 bars.
- **Data:** OHLCV. **Source:** ICT (Michael Huddleston) teachings; TradingView LuxAlgo "Fair Value Gap" scripts. **Evidence:** none peer-reviewed; gaps in 24/7 crypto are intrabar imbalances not true gaps. **Repaint:** medium; the FVG is only known at bar t (3rd candle close); safe if entries start after t.

### 37. Order Blocks
- **Family:** pattern / SMC. **Algorithm (systematic):** after a break of structure (#38) up, the last down candle before the impulse that caused the break is the bullish OB (zone [L,H] or [open,low]). Demand OB invalidated if price closes below its low.
- **Rules:** Long on first retest of bullish OB (limit at upper edge or 50%); stop below OB low; mirror for bear. 
- **Params:** swing length 5-20, max OB count, mitigation by close vs wick.
- **Data:** OHLCV. **Source:** ICT; LuxAlgo "Order Blocks & Breaker Blocks" / "Smart Money Concepts [LuxAlgo]" (open source MPL). **Evidence:** none. **Repaint/lookahead:** HIGH: OB identity depends on later pivot/BOS; only mark it at BOS confirmation bar.

### 38. Market Structure: BOS / CHoCH
- **Family:** pattern / SMC. **Algorithm:** swing highs/lows via `pivot(L,R)` (R-bar confirmation). Uptrend = HH & HL. BOS (continuation): close above last confirmed swing high in uptrend. CHoCH (reversal): close below last confirmed swing low in an uptrend (or above last swing high in downtrend). Internal vs swing structure (e.g. 5 vs 50 length).
- **Rules:** Long on bullish CHoCH, or after BOS retest (to OB/FVG); stop under last swing low; target next swing high/liquidity. Short mirror.
- **Params:** swing length 5-50. **Data:** OHLCV. **Source:** LuxAlgo "Smart Money Concepts"; ICT. **Evidence:** none. **Repaint:** HIGH if swing points are used at their pivot bar; use confirmation bar t+R; labels drawn retroactively in TV.

### 39. Volume Profile (POC / VAH / VAL)
- **Family:** volume. **Algorithm:** bin price into B buckets (e.g. 0.25% or 100 bins) over a window (session/week/composite); volume per bin via distributing bar volume across its H-L range (or using 1m data); POC = max-volume bin; value area = contiguous bins around POC holding 70% of volume.
- **Rules:** MR: price outside value area re-enters => target POC (long from VAL, short from VAH); acceptance outside VA => breakout trade toward next HVN. LVN rejection entries.
- **Params:** bins 50-200, VA 70% (68-80), window day/week/month.
- **Data:** OHLCV (1m for good approximation), free. Aggregated tape profile requires trade data (free from Binance data.binance.vision daily trade dumps). **Source:** Market Profile (J. Peter Steidlmayer), https://www.tradingview.com/support/solutions/43000502040/ (unverified). **Evidence:** none for BTC. **Repaint:** developing profile shifts with each bar; use completed prior-period levels only.

### 40. Funding Rate signals
- **Family:** derivatives / contrarian. **Formula:** perp funding rate (Binance every 8h; 3 prints/day), annualized `f*3*365`. Features: level, z-score over 30-90 d, cumulative 3d sum, sign flips, extremes (e.g. > +0.05%/8h or < -0.02%).
- **Rules:** Contrarian short/reduce long when funding z>2 (or > +0.05%) and price extended; long when funding strongly negative (< -0.02%) after capitulation. Trend variant: sustained positive funding as bullish regime filter. Cash-and-carry: not a directional signal.
- **Params:** z window 30 (14-180), threshold z 1.5-3, hold 1-7 days.
- **Data:** Binance/Bybit/OKX public REST `fapi/v1/fundingRate` (history back to 2019, free), also CoinGlass (limited free), Bybit. 
- **Source:** Exchange docs https://binance-docs.github.io/apidocs/futures/en/ ; Glassnode/CryptoQuant commentary; academic: He, Manela, Ross, von Wachter "Fundamentals of perpetual futures" (arXiv 2212.06888, no-arbitrage pricing; funding deviations) (unverified details). 
- **Evidence:** Practitioner claims of contrarian predictability; no strong peer-reviewed BTC evidence found in my search. Extremes are rare (few independent events => weak statistics).
- **Repaint:** Low (funding is published at settlement; predicted funding is not final). Use only settled values and timestamp precisely.

### 41. Open Interest signals
- **Family:** derivatives. **Formula:** OI (contracts or USD), `dOI`, OI/market cap, OI z-score. Matrix: price up + OI up = new longs (trend confirm); price up + OI down = short covering (weak); price down + OI up = new shorts; price down + OI down = long liquidation (capitulation -> reversal long).
- **Rules:** Long on price breakout + OI rising (<extreme); reversal long after large OI drop (>X% in 24h) with negative funding (flush); short on OI at ATH with price stalling and high funding.
- **Params:** OI change window 4h-7d, threshold z 2.
- **Data:** Binance `futures/data/openInterestHist` (only last 30 days at 5m-1d granularity via REST), Bybit/OKX limited; longer history needs CoinGlass/Coinalyze (free tier limited) or self-collection. Aggregated OI history is a data-acquisition problem: start logging now.
- **Source:** Exchange docs; TradingView "Open Interest" tickers (`BTCUSDT.P_OI`).
- **Evidence:** blogs only. **Repaint:** low, but OI revisions/exchange-specific quirks; Binance OI history is short.

### 42. Long/Short Ratio (contrarian)
- **Family:** derivatives / sentiment. **Formula:** Binance "global long/short account ratio", "top trader long/short ratio (accounts / positions)", taker buy/sell volume ratio. `LSR=long_accounts/short_accounts`.
- **Rules:** Contrarian: short when global LSR z>2 (retail crowd long), long when z<-2; "smart money" variant: follow top trader position ratio. Combine with funding.
- **Params:** period 5m-1d; z window 30d; threshold 1.5-2.5.
- **Data:** Binance `futures/data/globalLongShortAccountRatio` / `topLongShortPositionRatio` — free but history limited to ~30 days (some endpoints 1 year); CoinGlass free tier partial. Retaining a history requires you to snapshot daily.
- **Source:** https://binance-docs.github.io/apidocs/futures/en/#long-short-ratio. **Evidence:** none robust; Binance top-trader ratios are exchange-specific samples. **Repaint:** low.

### 43. Pi Cycle Top / Bottom
- **Family:** on-chain-like (price-only) macro timing. **Formula:** Top: `SMA(C,111)` crosses above `2*SMA(C,350)` (daily). Pi Cycle Bottom variant: `EMA(C,150)` crosses below `SMA(C,471)` (approx.).
- **Rules:** Sell/short at top cross (called 2013, 2017, 2021 tops within days; failed to trigger for the 2024-25 peak at least partly — verify with the data); exit/long at bottom cross.
- **Params:** 111/350 x2 (fixed; any search would be curve-fit). 
- **Data:** daily OHLCV only (free; Bitstamp history from 2011 via CryptoDataDownload; yfinance BTC-USD from 2014). **Source:** Philip Swift (@ApolloCrypto) 2019; https://www.lookintobitcoin.com/charts/pi-cycle-top-indicator/. 
- **Evidence:** 3 events; cannot be statistically tested (n=3). Reliability should be treated as anecdote and potentially already arbitraged. **Repaint:** low.

### 44. MVRV / MVRV Z-Score
- **Family:** on-chain. **Formula:** `MVRV=MarketCap/RealizedCap`; `Z=(MarketCap - RealizedCap)/stdev(MarketCap over full history)`.
- **Rules:** Buy when Z < 0 (or MVRV<1); reduce/short when Z > 7 (cycle top zone). Systematic: percentile-based thresholds.
- **Params:** Z top 6-8, bottom -0.5 to 0.5; MVRV 3.7 top / 1 bottom.
- **Data:** On-chain (realized cap = sum of UTXO value at last-moved price). Free: CoinMetrics Community API (`CapMVRVCur`, daily, free tier, https://docs.coinmetrics.io/), Blockchain-data on bitcoin-data.com, LookIntoBitcoin charts (view only), TradingView `BTC MVRV` via CoinMetrics symbols (e.g. `COINMETRICS:BTC_MVRV`?). Glassnode free tier gives only delayed/low-resolution (24h) basic metrics; most granular metrics paid. 
- **Source:** Mahmudov & Puell (2018), "Bitcoin MVRV"; https://coinmetrics.io/community-network-data/. **Evidence:** historically timed major tops/bottoms; again n~4 cycles, so no significance; Z-score's stdev uses the full sample (lookahead if computed over full history, use expanding window).
- **Repaint/lookahead:** Z-score stdev over full history is future-leaking; use expanding window. On-chain data are sometimes revised (entity clustering).

### 45. Time-Series Momentum (TSMOM)
- **Family:** momentum. **Formula:** `signal_t = sign(R_{t-L,t})` (L-day return), position = signal * (target_vol/realized_vol). Variants: log MA ratio `log(MA_short/MA_long)`, t-stat of returns (Moskowitz, Ooi, Pedersen 2012).
- **Params:** L 7-365 d (common: 28, 56, 90, 180, 365); vol window 20-60 d, target vol 20-60%; rebalance daily/weekly. Ensemble across L is more robust.
- **Rules:** Long if past L-day return>0 else short (or flat); exit at sign flip.
- **Data:** daily OHLCV. **Source:** Moskowitz, Ooi, Pedersen (2012) JFE; Liu & Tsyvinski (2021, RFS, "Risks and Returns of Cryptocurrency", https://www.nber.org/papers/w24877) find strong time-series momentum in crypto (1-4 weeks horizon); Zarattini et al. (2025) for Donchian ensemble.
- **Evidence:** One of the most robust documented effects, but gains concentrated in 2013-2021 bulls; short side usually lowers returns. **Repaint:** low.

### 46. Stochastic Oscillator
- **Family:** MR. **Formula:** `%K=100*(C-LL_n)/(HH_n-LL_n)`; smooth K 3; D=SMA(K,3).
- **Params:** n 14 (5-30), 20/80. **Rules:** Long on K cross D below 20; short above 80; exit at 50. **Data:** OHLCV. **Source:** George Lane. **Evidence:** none. **Repaint:** Low.

### 47. Aroon
- **Family:** trend. **Formula:** `AroonUp=100*(n-barsSinceHH_n)/n; AroonDn=100*(n-barsSinceLL_n)/n`; osc = Up-Dn.
- **Params:** n 25 (10-50). **Rules:** Long when Up crosses above Dn (or osc>50); short mirror. **Data:** OHLCV. **Source:** Tushar Chande (1995). **Evidence:** none. **Repaint:** Low.

### 48. Vortex Indicator
- **Family:** trend. **Formula:** `VM+=|H-L1|; VM-=|L-H1|; VI+=sum(VM+,n)/sum(TR,n); VI-=sum(VM-,n)/sum(TR,n)`.
- **Params:** n 14 (7-30). **Rules:** Long VI+ crosses above VI-; short opposite. **Data:** OHLCV. **Source:** Botes & Siepman (2010, TASC). **Evidence:** none. **Repaint:** Low.

### 49. Rate of Change / Momentum
- **Family:** momentum. **Formula:** `ROC=100*(C/C[n]-1)`.
- **Params:** n 10-200. **Rules:** Long ROC>0 (or > threshold) short <0; or ROC rank (percentile) filter. Equivalent to TSMOM (#45). **Data:** OHLCV. **Source:** classic. **Evidence:** see #45. **Repaint:** Low.

### 50. Fisher Transform
- **Family:** MR / turning point. **Formula:** `x=2*((hl2-LL_n)/(HH_n-LL_n)-0.5)`, smoothed `v=0.33*x+0.67*v1` clipped to (-0.999,0.999); `Fisher=0.5*ln((1+v)/(1-v))+0.5*Fisher1`; trigger=Fisher[1].
- **Params:** n 10 (5-20). **Rules:** Long when Fisher crosses above trigger below -1.5; short above +1.5. **Data:** OHLCV. **Source:** John Ehlers, Cybernetic Analysis for Stocks and Futures (2004). **Evidence:** none. **Repaint:** Low.

### 51. Elder Ray / Force Index
- **Family:** momentum / volume. **Formula:** Bull Power=H-EMA13; Bear Power=L-EMA13; Force=(C-C1)*V, smoothed EMA(13).
- **Rules (Elder triple screen):** Long when EMA13 rising, Bear Power negative but rising; short mirror. Force Index zero cross. **Data:** OHLCV. **Source:** Alexander Elder. **Evidence:** none. **Repaint:** Low.

### 52. Connors RSI / RSI(2)
- **Family:** short-term MR. **Formula:** `CRSI=(RSI(C,3)+RSI(streak,2)+PercentRank(ROC(1),100))/3`; RSI(2) variant: long when RSI(2)<10 and C>SMA200.
- **Params:** thresholds 5-15 / 85-95; exit C>SMA5.
- **Rules:** Long deeply oversold in uptrend; short symmetrical (weaker). **Data:** daily OHLCV. **Source:** Connors & Alvarez, Short Term Trading Strategies That Work (2008). **Evidence:** equities; crypto evidence thin and BTC daily MR edge has weakened. **Repaint:** Low.

### 53. KAMA / Adaptive MAs (KAMA, VIDYA, FRAMA, ALMA, T3, ZLEMA)
- **Family:** trend. **KAMA formula:** `ER=|C-C[n]|/sum|C-C1|(n); sc=(ER*(2/3-2/31)+2/31)^2; KAMA=KAMA1+sc*(C-KAMA1)`.
- **Params:** n 10 (5-30), fast 2, slow 30. **Rules:** Long when price > KAMA and KAMA rising; short opposite; use ER<thr to stand aside. **Data:** OHLCV. **Source:** Perry Kaufman (1995). **Evidence:** none specific. **Repaint:** Low.

### 54. Volatility Breakout (Larry Williams / Dual-thrust)
- **Family:** volatility/trend. **Formula:** `long_trigger = Open_t + k*(H[1]-L[1])` (k 0.5, range 0.3-1.0); short_trigger = Open - k*range.
- **Rules:** Long when price trades above trigger; exit at next open/close. **Params:** k 0.5; range window 1-5 d. **Data:** OHLCV (use 1h bars to detect intrabar touch order). **Source:** Larry Williams; popular in Korean BTC bots. **Evidence:** blogs. **Repaint:** low but intrabar fill order ambiguity (high vs low first) inflates backtests; use finer bars.

### 55. Opening Range / Session Breakout
- **Family:** intraday trend. **Formula:** range of first N minutes after anchor (00:00 UTC, 08:00 UTC London, 13:30 UTC NY); `OR_high/OR_low`.
- **Rules:** Long on break above OR_high with volume filter; stop at OR mid/low; time exit at session end.
- **Params:** N 15-120 min; anchors. **Data:** 1-15m OHLCV. **Source:** Toby Crabel (ORB); Concretum "Seasonality in Bitcoin intraday trend trading" (https://concretumgroup.com/seasonality-in-bitcoin-intraday-trend-trading/) reports gross Sharpe ~1.6 for an intraday ensemble 2018-2025 (not verified, gross-of-fees; a high-frequency fee drag is the key risk). **Repaint:** Low.

### 56. Mayer Multiple / 200-day MA regime
- **Family:** trend/MR macro. **Formula:** `MM=C/SMA(C,200)`.
- **Rules:** Trend: long above SMA200 (or 200 EMA), flat below. Value: accumulate when MM<0.8-1.0, de-risk MM>2.4. **Params:** 150-250d; band 1-3%. **Data:** daily OHLCV. **Source:** Trace Mayer; https://mayermultiple.info/. **Evidence:** price vs 200d MA is a basic rule inside Hudson & Urquhart class MA; effective at reducing drawdown. **Repaint:** Low.

### 57. SOPR / NUPL / Puell / Realized Price / Reserve Risk
- **Family:** on-chain. **Formulas:** SOPR = value at spend / value at creation of spent outputs (aSOPR excludes <1h). NUPL=(MC-RC)/MC. Puell=daily issuance USD / MA365 of issuance USD. Realized price = RC/supply.
- **Rules:** Long when SOPR<1 then recovers to >1 (capitulation end); NUPL<0 buy / >0.75 sell; Puell<0.5 buy / >4 sell; price crossing below realized price = bear regime.
- **Data:** Puell is computable free from price + known issuance schedule (OHLCV only!). Others need UTXO-based data: CoinMetrics community free (realized cap, NUPL derivation possible; SOPR not in community set I believe, verify), bitcoin-data.com free API (rate-limited), Glassnode paid (free tier restricted). **Source:** Glassnode Academy; https://studio.glassnode.com; https://docs.coinmetrics.io/ . **Evidence:** retrospective timing; low n. **Repaint:** revisions & entity-adjusted variants change history; use first-print vintage if possible.

### 58. Liquidation cascades
- **Family:** derivatives / reversal. **Formula:** aggregate forced-liquidation USD per window (e.g. 1h) vs rolling z-score; long-liquidation spikes suggest local capitulation.
- **Rules:** Long after extreme long-liquidation spike (z>3) once price stops making new lows (first higher low 5-15 min); short after short squeeze exhaustion. Stop beyond cascade extreme.
- **Data:** Binance `forceOrder` websocket (live, not historical via free REST); history needs CoinGlass/Coinalyze paid or self-recording. **Source:** exchange docs. **Evidence:** blogs. **Repaint:** low.

### 59. Futures basis / premium
- **Family:** derivatives. **Formula:** `basis=(F-S)/S*365/days_to_expiry` for quarterly futures; perp premium index = (perp-spot)/spot.
- **Rules:** Contrarian at extremes (annualized basis >20-30% = overheated; <0 backwardation = fear); trend filter if basis positive. **Data:** Binance/Deribit/OKX quarterlies (free, history via REST of delivery contracts) or CME (paid). **Source:** exchange docs; He et al. 2022 (arXiv 2212.06888). **Evidence:** limited. **Repaint:** low.

### 60. Taker buy/sell ratio and Cumulative Volume Delta (CVD)
- **Family:** volume/flow. **Formula:** Binance klines include taker buy base volume; `delta=2*takerBuyVol - Vol`; `CVD=cumsum(delta)`; ratio=takerBuy/(Vol-takerBuy).
- **Rules:** Long on price breakout with rising CVD; divergence (price HH, CVD LH) = short. 
- **Params:** CVD MA 20-100. **Data:** free (kline fields 9-10 on Binance; data.binance.vision). **Source:** Binance docs. **Evidence:** blogs; order-flow edges at tick-level are short-lived. **Repaint:** low.

### 61. Candlestick patterns
- **Family:** pattern. **Algorithm:** engulfing (`C>O and C1<O1 and C>=O1 and O<=C1`), hammer (lower shadow >= 2*body, small upper shadow), doji, morning/evening star (TA-Lib `CDL*`).
- **Rules:** Long on bullish reversal pattern at support/oversold (RSI<30); short mirror; stop beyond pattern extreme.
- **Data:** OHLCV. **Source:** Nison; TA-Lib. **Evidence:** academic tests generally find little robust predictive power in equities/FX; crypto minimal. **Repaint:** low (confirm after close).

### 62. Fibonacci retracement / Pivot points
- **Family:** pattern/levels. **Formula:** retracements 0.382/0.5/0.618/0.786 of last confirmed swing; classic pivots `P=(H+L+C)/3 (prior day), R1=2P-L, S1=2P-H`; Camarilla etc.
- **Rules:** Long on bounce at 0.618 in uptrend with reversal candle; Short mirror. **Data:** OHLCV. **Source:** textbook. **Evidence:** none robust. **Repaint:** HIGH for swing-based Fibs (swing is hindsight); prior-day pivots safe.

### 63. Trendline / chart patterns (triangles, H&S, flags, wedges)
- **Family:** pattern. **Algorithm:** fit lines through pivot highs/lows (RANSAC / `scipy.signal.find_peaks`), detect breakout bar.
- **Rules:** Long on close above descending trendline / triangle top with volume; target = pattern height; stop inside pattern.
- **Data:** OHLCV. **Source:** Lo, Mamaysky & Wang (2000), "Foundations of Technical Analysis" (kernel regression pattern detection, JF); Bulkowski. **Evidence:** equities only. **Repaint:** HIGH, since lines are fit in hindsight; kernel smoothing is non-causal.

### 64. Ehlers DSP filters (Super Smoother, Instantaneous Trendline, Cyber Cycle, MAMA/FAMA, Roofing filter)
- **Family:** trend/cycle. **Super Smoother:** `a1=exp(-1.414*pi/n); b1=2*a1*cos(1.414*pi/n); c2=b1; c3=-a1^2; c1=1-c2-c3; SS=c1*(x+x1)/2+c2*SS1+c3*SS2`.
- **Params:** n 10-100. **Rules:** Long when SS slope positive / price crosses SS; MAMA crosses FAMA for trend flips. Roofing filter + Stoch for cycle. **Data:** OHLCV. **Source:** Ehlers books; https://www.mesasoftware.com/ . **Evidence:** vendor. **Repaint:** low (Ehlers Hilbert-based items are causal but need warm-up).

### 65. Volatility / regime overlay (vol-targeting, HMM, vol percentile, GARCH)
- **Family:** volatility/ML overlay. **Algorithm:** realized vol `sigma=stdev(r,n)*sqrt(365)`; position size = `target/sigma` (cap 2x); regime filter: trade trend rules only when ATR percentile > 30%, MR rules when low; Gaussian HMM (2-3 states) on returns & vol via `hmmlearn`.
- **Rules:** Scale/long-short rules from other entries by regime. **Params:** vol window 20-60, target 20-60%, HMM states 2-3. **Data:** OHLCV. **Source:** Moskowitz et al. (2012); Harvey et al. (2018) "The Impact of Volatility Targeting" (JPM). **Evidence:** vol targeting generally improves Sharpe/drawdown for risk assets; BTC specific positive in Zarattini et al. **Repaint/lookahead:** HMM smoothed state probabilities (`predict`, Viterbi) use the full sequence = lookahead; use filtered (forward-only) probabilities and refit expanding-window.

### 66. Halving cycle / seasonality / calendar effects
- **Family:** on-chain/calendar narrative. **Algorithm:** days since halving (Nov 2012, Jul 2016, May 2020, Apr 2024); position by phase (e.g., long from ~6-12 mo post-halving to ~18 mo, flat after). Day-of-week/hour-of-day effects (e.g., Concretum intraday seasonality).
- **Data:** OHLCV. **Evidence:** n = 4 cycles; essentially untestable and a classic overfit. Treat as prior, not signal. **Repaint:** none, but hindsight in choosing phase windows.

### 67. Fear & Greed Index (and social/search sentiment)
- **Family:** sentiment. **Formula:** alternative.me index 0-100 (volatility, momentum/volume, social, dominance, trends).
- **Rules:** Contrarian: long at <20-25 (extreme fear), reduce/short at >80. 
- **Data:** free API https://api.alternative.me/fng/?limit=0 (daily since Feb 2018). Google Trends via `pytrends` free. **Evidence:** blog-level; contrarian works at some extremes, but overlaps with momentum (index is partly built from price). Methodology changed over time. **Repaint:** low; check historical revisions.

### 68. ML meta-model on feature stack (GBM / logistic / LSTM)
- **Family:** ML. **Algorithm:** features = returns at multiple horizons, RSI, MACD hist, ATR%, BB %B, vol z, funding z, OI change; target = sign (or triple-barrier label) of forward return; model = LightGBM / logistic regression; purged & embargoed K-fold / walk-forward (López de Prado, Advances in Financial Machine Learning, 2018); meta-labeling to filter primary signals.
- **Params:** horizon 1-24h or 1-7d; depth 3-6, 100-500 trees (strong regularization).
- **Rules:** Long if P(up)>0.55, short if <0.45; size by confidence; retrain rolling. **Data:** OHLCV (+funding/OI optional). **Evidence:** academic results are mixed; leakage is endemic. **Lookahead risk:** MEDIUM-HIGH (label overlap, scaler fit on full sample, feature shift errors).

---

## 3. Academic and credible references

| # | Paper | One-line finding |
|---|-------|------------------|
| 1 | Hudson, R. & Urquhart, A. (2021). "Technical trading and cryptocurrencies." *Annals of Operations Research* 297, 191-220. https://link.springer.com/article/10.1007/s10479-019-03357-1 | ~15k rules from 5 families (filters, MA, SR, channel breakouts, momentum/oscillators) on BTC (two markets) and 3 other coins; many rules remain significant after data-snooping correction, and rules reduce drawdowns vs buy-and-hold (sample ends 2018). |
| 2 | Bailey, Borwein, López de Prado & Zhu (2014/2017). "The Probability of Backtest Overfitting." *J. Computational Finance* (SSRN 2326253). | Picking the best of many backtests gives an inflated IS Sharpe; CSCV estimates PBO and shows more trials -> higher overfit probability (recalled from memory; not fetched). |
| 3 | Bailey & López de Prado (2014). "The Deflated Sharpe Ratio." *J. Portfolio Management* (SSRN 2460551). | Adjusts the Sharpe for number of trials, skew and kurtosis; use to judge strategy selected from a grid (memory). |
| 4 | Harvey, Liu & Zhu (2016). "...and the Cross-Section of Expected Returns." *RFS*; Harvey & Liu (2015) "Backtesting" *JPM*. | Multiple-testing requires t-stat > ~3 and Sharpe haircuts of ~50% for backtested strategies (memory). |
| 5 | Liu, Y. & Tsyvinski, A. (2021). "Risks and Returns of Cryptocurrency." *RFS* 34(6), 2689-2727. https://www.nber.org/papers/w24877 | Strong time-series momentum and attention proxies forecast crypto returns; only crypto-specific factors (momentum/attention) explain returns. |
| 6 | Zarattini, Pagani & Barbon (2025). "Catching Crypto Trends." SSRN 5209907 / SFI RP 25-80. https://abarbon.com/papers/catching-crypto-trends | Donchian-ensemble with vol-sizing on BTC reported Sharpe ~1.6 and ~14% alpha net of costs (2015-2025); secondary summary, verify. |
| 7 | Detzel, Liu, Strauss, Zhou & Zhu (2021). "Learning and predictability via technical analysis: Evidence from bitcoin and stocks with hard-to-value fundamentals." *Financial Management*. | 5-100 day moving-average signals predict BTC returns in- and out-of-sample (as per search summary). |
| 8 | Bakker (2018), Erasmus Univ. thesis (https://thesis.eur.nl/pub/41546/). | 3,312 intraday rules on 5-min BTC/USD 2013-17: many significant after data-snooping and costs (thesis-level quality). |
| 9 | Li (2023), *J. Accounting and Finance*, 2022-23 BTC/ETH-USDT. | Skeptical: no profitable technical strategies in-sample or out-of-sample after data-snooping adjustment; shows regime/sample dependence. |
| 10 | Corbet, Eraslan, Lucey & Sensoy (2019). "The effectiveness of technical trading rules in cryptocurrency markets." *Finance Research Letters* 31 (https://ideas.repec.org/a/eee/finlet/v32y2020ics1544612319308852.html is the related 2020 "Technical trading rules in the cryptocurrency market") | Reports effects of simple MA/ momentum rules; mixed; also shows sensitivity to exchange and period (summary uncertain; verify). |
| 11 | Brock, Lakonishok & LeBaron (1992), *J. Finance* 47; Sullivan, Timmermann & White (1999), *J. Finance* 54; White (2000) Reality Check *Econometrica*; Hansen (2005) SPA. | Foundations: TA rules appeared profitable on DJIA until data-snooping (reality check) adjustments; use these tests (e.g. `arch` package `SPA`, `StepM`) for crypto. |
| 12 | Lo, Mamaysky & Wang (2000). "Foundations of Technical Analysis." *J. Finance* 55. | Kernel-regression pattern recognition: some patterns carry incremental info in US equities (note the non-causal smoothing risk). |
| 13 | Moskowitz, Ooi & Pedersen (2012). "Time Series Momentum." *JFE* 104. | Positive 1-12 month TSMOM in 58 futures; basis for #45. |
| 14 | López de Prado (2018), *Advances in Financial Machine Learning*, Wiley. | Purged k-fold, embargo, triple barrier, meta-labeling for ML signals (#35, #68). |

Caveat: the first three entries are anchored by searches in this session; items 2-4, 10-14 are from memory and need URL/volume check before citing in any write-up.

---

## 4. Data availability summary

| Need | Source | Free? | History |
|------|--------|-------|---------|
| OHLCV spot/perp | Binance (`/api/v3/klines`, data.binance.vision bulk), Bybit, OKX, Kraken, Coinbase via `ccxt`; yfinance BTC-USD daily | Yes | Binance spot 2017+, Bitstamp 2011+, perp 2019+ |
| Taker buy volume | Binance klines fields | Yes | 2017+ |
| Funding rate | Binance `fapi/v1/fundingRate`, Bybit, OKX | Yes | 2019+ |
| Open interest | Binance `futures/data/openInterestHist` | Yes but only ~30 days | Needs self-logging or CoinGlass/Coinalyze (partial) |
| Long/short ratio | Binance `futures/data/*LongShort*` | Yes but ~30 days (some ~1 year) | Needs self-logging |
| Liquidations | Binance WS `forceOrder` | Live only | Self-record; history paid |
| Futures basis | Binance/Deribit/OKX quarterlies | Yes | Contract-by-contract |
| MVRV, realized cap | CoinMetrics community API; bitcoin-data.com | Yes (limits) | 2010+ daily |
| SOPR, NUPL, entity-adjusted metrics, exchange flows | Glassnode, CryptoQuant | Mostly PAID (free tier delayed/daily-limited) | |
| Fear & Greed | alternative.me API | Yes | 2018+ |
| Puell Multiple, Pi Cycle, Mayer, 200DMA | Price only (+ issuance schedule) | Yes | |

## 5. Non-OHLCV dependencies (shortlist)
Funding (#40, free), OI (#41, free short history), L/S ratio (#42, free short history), liquidations (#58, live only), basis (#59, free), taker CVD (#60, free via klines), MVRV/NUPL/SOPR (#44, #57; MVRV partly free via CoinMetrics, SOPR mostly paid), Fear & Greed (#67, free), ML (#68, optional).

## 6. Top repaint/lookahead traps (checklist)
1. `request.security` with `lookahead_on` and no `[1]` offset (Cipher B Sommi flags/diamonds, any MTF script).
2. Pivot/fractal-based logic (divergences, BOS/CHoCH, order blocks, Fibs, anchored VWAP anchors): shift by right-bars R and fill after confirmation.
3. Nadaraya-Watson (non-causal kernel) and Lorentzian's kernel-smoothed and in-sample stats.
4. Ichimoku cloud/Chikou shifting; Z-scores with full-sample stdev (MVRV-Z); HMM smoothed states; ML scalers/labels fitted on the whole sample.
5. Intrabar fill ordering (breakout/stop and take-profit in the same bar); HA/Renko prices as fill prices.
6. Parameter-grid snooping: report number of trials; use deflated Sharpe and White/Hansen SPA test; prefer ensembles over single optimized parameters.
