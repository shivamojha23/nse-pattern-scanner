import numpy as np
import pandas as pd
import math
import datetime
import os
import sys
from scipy.signal import find_peaks
from scipy.stats import linregress
from .core import *

def detect_bull_flag(prices, highs=None, lows=None, volumes=None, ticker="UNKNOWN", dates=None,
                     interval="1d", verbose=False):
    """
    Detects Bull Flag patterns using strict wick boundaries and slanted trendline breakout detection.
    """
    from scipy.stats import linregress
    import numpy as np
    import pandas as pd
    
    prices = np.array(prices, dtype=float)
    if highs is None: highs = prices
    if lows is None: lows = prices
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)

    if volumes is not None:
        volumes = np.array(volumes, dtype=float)

    n = len(prices)
    if n < 30:
        return []

    patterns_found = []

    # Slow Path: Full Detection
    rolling_min = pd.Series(lows).rolling(window=MAX_POLE_CANDLES, min_periods=5).min().values
    potential_return = np.zeros(n)
    for idx in range(MAX_POLE_CANDLES, n):
        if rolling_min[idx] > 0:
            potential_return[idx] = (highs[idx] - rolling_min[idx]) / rolling_min[idx] * 100
    potential_pole_ends = np.where(potential_return >= MIN_POLE_RISE_PCT)[0]

    valid_structures = []

    for pole_end_idx in potential_pole_ends:
        if pole_end_idx >= n - MIN_FLAG_CANDLES - 1:
            continue

        best_pole = None
        for pole_len in range(5, min(MAX_POLE_CANDLES + 1, pole_end_idx + 1)):
            pole_start_idx = pole_end_idx - pole_len
            if pole_start_idx < 0:
                continue

            rise_pct = (highs[pole_end_idx] - lows[pole_start_idx]) / lows[pole_start_idx] * 100
            if rise_pct < MIN_POLE_RISE_PCT:
                continue

            rise_per_day = rise_pct / pole_len
            if rise_per_day < MIN_POLE_SLOPE_PCT_PER_DAY:
                continue

            pole_prices = prices[pole_start_idx:pole_end_idx + 1]
            x = np.arange(len(pole_prices))
            res = linregress(x, pole_prices)
            r_squared = res.rvalue ** 2

            if res.slope <= 0 or r_squared < POLE_R_SQUARED_MIN:
                continue

            if best_pole is None or rise_pct > best_pole["rise_pct"]:
                best_pole = {
                    "start_idx": pole_start_idx,
                    "end_idx": pole_end_idx,
                    "rise_pct": rise_pct,
                    "r_squared": r_squared,
                    "slope": res.slope,
                }

        if best_pole is None:
            continue

        pole_top_price = highs[pole_end_idx]

        for flag_len in range(MIN_FLAG_CANDLES, min(MAX_FLAG_CANDLES + 1, n - pole_end_idx)):
            flag_start = pole_end_idx + 1
            flag_end = pole_end_idx + flag_len
            if flag_end >= n:
                break

            flag_prices = prices[flag_start:flag_end + 1]
            flag_highs = highs[flag_start:flag_end + 1]
            flag_lows = lows[flag_start:flag_end + 1]
            if len(flag_prices) < MIN_FLAG_CANDLES:
                continue

            flag_high = float(np.max(flag_highs))
            flag_low = float(np.min(flag_lows))

            if flag_high > pole_top_price * 1.001:
                continue

            flag_range_pct = (flag_high - flag_low) / pole_top_price * 100
            if flag_range_pct > FLAG_CHANNEL_PCT:
                continue

            pole_size = pole_top_price - lows[best_pole["start_idx"]]
            retracement_pct = (pole_top_price - flag_low) / pole_size * 100 if pole_size > 0 else 0
            if retracement_pct > MAX_RETRACEMENT_PCT:
                continue

            x = np.arange(len(flag_prices))
            flag_res = linregress(x, flag_prices)
            flag_total_drift_pct = (flag_res.slope * len(flag_prices)) / pole_top_price * 100
            if flag_total_drift_pct > 0.5:
                continue

            # Calculate upper and lower bounds for the channel based on residuals of highs and lows
            line_vals = flag_res.intercept + flag_res.slope * x
            upper_intercept = flag_res.intercept + np.max(flag_highs - line_vals)
            lower_intercept = flag_res.intercept + np.min(flag_lows - line_vals)

            pattern = {
                "pattern_type":       "bull_flag",
                "ticker":             ticker,
                "verdict":            "VALID",
                "reject_reason":      "",
                "pole_start_price":   round(float(lows[best_pole["start_idx"]]), 2),
                "pole_top_price":     round(float(pole_top_price), 2),
                "flag_low_price":     round(float(flag_low), 2),
                "flag_high_price":    round(float(flag_high), 2),
                "pole_rise_pct":      round(float(best_pole["rise_pct"]), 2),
                "pole_r_squared":     round(float(best_pole["r_squared"]), 3),
                "flag_range_pct":     round(float(flag_range_pct), 2),
                "flag_slope":         round(float(flag_res.slope), 4),
                "pole_start_idx":     best_pole["start_idx"],
                "pole_end_idx":       pole_end_idx,
                "flag_start_idx":     flag_start,
                "flag_end_idx":       flag_end,
                "retracement_pct":    round(float(retracement_pct), 2),
                "flag_upper_intercept": float(upper_intercept),
                "flag_lower_intercept": float(lower_intercept),
            }

            if dates is not None:
                pattern["pole_start_date"] = str(dates[best_pole["start_idx"]])
                pattern["pole_top_date"]   = str(dates[pole_end_idx])
                pattern["flag_start_date"] = str(dates[flag_start])
                pattern["flag_end_date"]   = str(dates[flag_end])
            
            valid_structures.append(pattern)

    # Now evaluate breakouts on all valid structures
    forming_candidates = {}
    
    for pat in valid_structures:
        flag_end = pat["flag_end_idx"]
        flag_start = pat["flag_start_idx"]
        flag_slope = pat["flag_slope"]
        upper_intercept = pat["flag_upper_intercept"]
        
        breakout_idx = None
        for k in range(flag_end + 1, min(n, flag_end + 6)):
            rel_k = k - flag_start
            upper_tl_val = upper_intercept + flag_slope * rel_k
            if prices[k] > upper_tl_val:
                breakout_idx = k
                break

        is_forming = False
        if breakout_idx is None:
            # It's a forming pattern only if it's currently active at the right edge
            if flag_end >= n - 6:
                is_forming = True
            else:
                if verbose:
                    print(f"  ❌ {ticker}: Bull Flag pole found (idx {pat['pole_start_idx']}-"
                          f"{pat['pole_end_idx']}, +{pat['pole_rise_pct']:.1f}%) but no breakout and flag expired.")
                continue
            
        vol_sma50 = 0
        vol_pole_pass = None
        vol_flag_pass = None
        vol_breakout_pass = None

        if volumes is not None and len(volumes) == n:
            vol_series = pd.Series(volumes)
            vol_sma = vol_series.rolling(window=VOLUME_SMA_PERIOD, min_periods=10).mean()
            sma_idx = min(pat["pole_end_idx"], len(vol_sma) - 1)
            vol_sma50 = float(vol_sma.iloc[sma_idx]) if not pd.isna(vol_sma.iloc[sma_idx]) else 0

            if vol_sma50 > 0:
                pole_vol = float(np.mean(volumes[pat["pole_start_idx"]:pat["pole_end_idx"] + 1]))
                vol_pole_pass = pole_vol > vol_sma50

                flag_vol = float(np.mean(volumes[pat["pole_end_idx"] + 1:pat["flag_end_idx"] + 1]))
                vol_flag_pass = flag_vol < vol_sma50

                if not is_forming:
                    breakout_vol = float(volumes[breakout_idx])
                    vol_breakout_pass = breakout_vol > (BREAKOUT_VOLUME_MULTIPLIER * vol_sma50)

        if not is_forming and REQUIRE_BREAKOUT_VOLUME and vol_breakout_pass is False:
            continue
            
        quality = compute_quality_score("bull_flag", {
            "pole_change_pct": pat["pole_rise_pct"],
            "r_squared": pat["pole_r_squared"],
            "flag_range_pct": pat["flag_range_pct"],
            "vol_pole_pass": vol_pole_pass,
            "vol_flag_pass": vol_flag_pass,
            "vol_breakout_pass": vol_breakout_pass,
        })
        
        pat = pat.copy()
        pat["status"] = "forming" if is_forming else "confirmed"
        pat["vol_pole_pass"] = vol_pole_pass
        pat["vol_flag_pass"] = vol_flag_pass
        pat["vol_breakout_pass"] = vol_breakout_pass
        pat["quality_score"] = quality
        
        if is_forming:
            pat["breakout_idx"] = None
            pat["breakout_price"] = None
            if dates is not None:
                pat["signal_date"] = None
                
            pole_key = pat["pole_start_idx"]
            if pole_key not in forming_candidates or pat["flag_end_idx"] > forming_candidates[pole_key]["flag_end_idx"]:
                forming_candidates[pole_key] = pat
        else:
            pat["breakout_idx"] = breakout_idx
            pat["breakout_price"] = round(float(prices[breakout_idx]), 2)
            if dates is not None:
                pat["signal_date"] = str(dates[breakout_idx])
            patterns_found.append(pat)

    # Add the single best forming pattern per pole
    confirmed_poles = {p["pole_start_idx"] for p in patterns_found}
    for fpat in forming_candidates.values():
        if fpat["pole_start_idx"] not in confirmed_poles:
            patterns_found.append(fpat)

    patterns_found.sort(key=lambda p: p.get("quality_score", 0), reverse=True)
    
    # Deduplication
    final_patterns = []
    claimed = []
    for c in patterns_found:
        is_dup = False
        for (cs, ce) in claimed:
            ce_c = c.get("breakout_idx")
            ce_s = ce
            
            if ce_c is not None and ce_s is not None:
                if abs(c["pole_start_idx"] - cs) <= 10 and abs(ce_c - ce_s) <= 10:
                    is_dup = True
                    break
            else:
                if abs(c["pole_start_idx"] - cs) <= 10:
                    is_dup = True
                    break
                    
        if not is_dup:
            final_patterns.append(c)
            claimed.append((c["pole_start_idx"], c.get("breakout_idx")))
            
    return final_patterns
