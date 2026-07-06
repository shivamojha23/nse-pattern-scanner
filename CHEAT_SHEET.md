# 📊 Chart Pattern Cheat Sheet — 7 Patterns at a Glance

A quick-reference guide for all patterns detected by the NSE Multi-Pattern Scanner v3.0.

---

## 1. ☕ Cup and Handle — Bullish Continuation

```
  Left Rim ────╮                  ╭──── Right Rim
                ╲                ╱
                 ╲              ╱         ╮── Handle
                  ╲            ╱          │   (small dip)
                   ╰──────────╯           ╯
                     Cup Bottom       → Breakout ↑
```

| Property | Details |
|----------|---------|
| Signal | **Bullish continuation** — price resumes uptrend after breakout |
| Key Levels | Left Rim, Cup Bottom, Right Rim, Handle Low |
| What to Look For | Cup drop 10–35%, rounded bottom (not V-shaped), Right Rim near Left Rim, handle dip ≤32% of cup depth, breakout above rim on high volume |
| Volume | Low during decline, increasing during recovery, spike on breakout |

---

## 2. 🟢 Bull Flag — Bullish Continuation

```
                                        ╱ Breakout ↑
                            ╲──────╱
                           ╱ Flag ╲
                          ╱  (tight)
            Pole ↑       ╱
           ╱            ╱
          ╱
         ╱
```

| Property | Details |
|----------|---------|
| Signal | **Bullish continuation** — sharp rally continues after brief pause |
| Key Levels | Pole Start, Pole Top, Flag Low, Breakout Point |
| What to Look For | Pole: strong ≥8% rise in ≤20 candles (R²>0.8), Flag: tight ≤5% range drifting down/flat, Breakout above pole top on high volume |
| Volume | High during pole, contracts during flag, spikes on breakout |

---

## 3. 🔴 Bear Flag — Bearish Continuation

```
         ╲
          ╲
           ╲
            ╲       Pole ↓
             ╲
              ╲
                ╲──────╲
                ╱ Flag  ╲
               ╱ (tight) ╲
                           ╲ Breakdown ↓
```

| Property | Details |
|----------|---------|
| Signal | **Bearish continuation** — sharp selloff continues after weak bounce |
| Key Levels | Pole Start, Pole Bottom, Flag High, Breakdown Point |
| What to Look For | Pole: ≥8% drop in ≤20 candles (R²>0.8), Flag: tight ≤5% range drifting up/flat, Breakdown below pole bottom on high volume |
| Volume | High during pole, contracts during flag, spikes on breakdown |

---

## 4. 🔺 Pennant — Continuation (Both Directions)

```
                         ╱╲
            Pole ↑     ╱   ╲      ╱ Breakout ↑
           ╱          ╱ Pennant ╲╱
          ╱            ╲       ╱
         ╱               ╲  ╱
                           ╲╱
                    (converging triangle)
```

| Property | Details |
|----------|---------|
| Signal | **Continuation** — breakout in the same direction as the pole |
| Key Levels | Pole Start, Pole End, Pennant Start/End, Breakout Point |
| What to Look For | Strong pole (≥8%), then converging highs+lows (triangle), second half range < first half range, breakout in pole direction |
| Volume | High during pole, contracts in pennant, spikes on breakout |

---

## 5. 👤 Head and Shoulders — Bearish Reversal

```
                   ╱╲    Head
                  ╱  ╲
         ╱╲      ╱    ╲      ╱╲
        ╱  ╲    ╱      ╲    ╱  ╲
  LS   ╱    ╲  ╱        ╲  ╱    ╲  RS
      ╱      ╲╱  Left    ╲╱      ╲
             Neckline   Neckline   ╲ Breakdown ↓
```

| Property | Details |
|----------|---------|
| Signal | **Bearish reversal** — uptrend exhaustion, shift to downtrend |
| Key Levels | Left Shoulder, Head, Right Shoulder, Left/Right Neckline, Breakdown |
| What to Look For | Head ≥3% higher than both shoulders, shoulders within 5% of each other, neckline relatively flat, span ≥20 candles, breakdown below neckline |
| Volume | Highest on left shoulder, moderate on head, lightest on right shoulder, spike on breakdown |

---

## 6. 🔝 Double Top — Bearish Reversal

```
         ╱╲            ╱╲
        ╱  ╲   Top 1  ╱  ╲  Top 2
       ╱    ╲        ╱    ╲
      ╱      ╲      ╱      ╲
              ╲    ╱         ╲  Breakdown ↓
               ╲  ╱
                ╲╱  Valley (neckline)
```

| Property | Details |
|----------|---------|
| Signal | **Bearish reversal** — shaped like "M", buyers fail twice at same level |
| Key Levels | First Top, Valley Low, Second Top, Breakdown Point |
| What to Look For | Two peaks within 3% of each other, valley ≥3% below peaks, separation ≥10 candles, breakdown below valley |
| Volume | First top has higher volume than second (declining buying interest) |

---

## 7. 🔽 Double Bottom — Bullish Reversal

```
                ╱╲  Peak (neckline)
               ╱  ╲
              ╱    ╲
             ╱      ╲                  ╱ Breakout ↑
        ╲   ╱        ╲   ╱           ╱
         ╲ ╱          ╲ ╱
          V  Bot 1     V  Bot 2
```

| Property | Details |
|----------|---------|
| Signal | **Bullish reversal** — shaped like "W", sellers fail twice at same level |
| Key Levels | First Bottom, Peak High, Second Bottom, Breakout Point |
| What to Look For | Two troughs within 3% of each other, peak ≥3% above bottoms, separation ≥10 candles, breakout above peak |
| Volume | Second bottom has lower volume (declining selling pressure), breakout on high volume |

---

## Quick Comparison Table

| Pattern | Type | Signal | Smoothing Used | Direction |
|---------|------|--------|----------------|-----------|
| Cup & Handle | Continuation | Bullish ↑ | ✅ Yes (find_peaks) | Up |
| Bull Flag | Continuation | Bullish ↑ | ❌ No (linregress) | Up |
| Bear Flag | Continuation | Bearish ↓ | ❌ No (linregress) | Down |
| Pennant | Continuation | Pole direction | ❌ No (linregress) | Pole |
| Head & Shoulders | Reversal | Bearish ↓ | ✅ Yes (find_peaks) | Down |
| Double Top | Reversal | Bearish ↓ | ✅ Yes (find_peaks) | Down |
| Double Bottom | Reversal | Bullish ↑ | ✅ Yes (find_peaks) | Up |

---

## How to Verify a Pattern Manually

When the scanner flags a stock, verify on a chart:

1. **Open the chart** on TradingView, Zerodha Kite, or any charting tool
2. **Set the same interval** the scanner used (15m, 1d, etc.)
3. **Draw the key levels** from the scanner output on the chart
4. **Check volume bars** — do they match the expected pattern?
5. **Confirm the breakout/breakdown** candle exists and has elevated volume
6. **Look at the broader context** — is the overall trend supportive?

> ⚠ **No pattern is 100% reliable.** Always use stop-losses and position sizing. Scanner flags are *candidates* for further analysis, not trade signals.
