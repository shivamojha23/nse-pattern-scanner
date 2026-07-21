import numpy as np
import pandas as pd
import math
import datetime
import os
import sys
from scipy.signal import find_peaks
from scipy.stats import linregress
from .core import *

def detect_bear_flag(prices, highs=None, lows=None, volumes=None, ticker="UNKNOWN", dates=None,
                     interval="1d", verbose=False):
    """
    Detects Bear Flag patterns using strict wick boundaries and slanted trendline breakdown detection.
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
    rolling_max = pd.Series(highs).rolling(window=MAX_POLE_CANDLES, min_periods=5).max().values
    potential_drop = np.zeros(n)
    for idx in range(MAX_POLE_CANDLES, n):
        if rolling_max[idx] > 0:
            potential_drop[idx] = (rolling_max[idx] - lows[idx]) / rolling_max[idx] * 100
    potential_pole_ends = np.where(potential_drop >= MIN_POLE_DROP_PCT)[0]

    valid_structures = []

    for pole_end_idx in potential_pole_ends:
        if pole_end_idx >= n - MIN_FLAG_CANDLES - 1:
            continue

        best_pole = None
        for pole_len in range(5, min(MAX_POLE_CANDLES + 1, pole_end_idx + 1)):
            pole_start_idx = pole_end_idx - pole_len
            if pole_start_idx < 0:
                continue

            drop_pct = (highs[pole_start_idx] - lows[pole_end_idx]) / highs[pole_start_idx] * 100
            if drop_pct < MIN_POLE_DROP_PCT:
                continue

            drop_per_day = drop_pct / pole_len
            if drop_per_day < MIN_POLE_SLOPE_PCT_PER_DAY:
                continue

            pole_prices = prices[pole_start_idx:pole_end_idx + 1]
            x = np.arange(len(pole_prices))
            res = linregress(x, pole_prices)
            r_squared = res.rvalue ** 2

            if res.slope >= 0 or r_squared < POLE_R_SQUARED_MIN:
                continue

            if best_pole is None or drop_pct > best_pole["drop_pct"]:
                best_pole = {
                    "start_idx": pole_start_idx,
                    "end_idx": pole_end_idx,
                    "drop_pct": drop_pct,
                    "r_squared": r_squared,
                    "slope": res.slope,
                }

        if best_pole is None:
            continue

        pole_bottom_price = lows[pole_end_idx]

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

            if flag_low < pole_bottom_price * 0.999:
                continue

            flag_range_pct = (flag_high - flag_low) / pole_bottom_price * 100
            if flag_range_pct > FLAG_CHANNEL_PCT:
                continue

            pole_size = highs[best_pole["start_idx"]] - pole_bottom_price
            retracement_pct = (flag_high - pole_bottom_price) / pole_size * 100 if pole_size > 0 else 0
            if retracement_pct > MAX_RETRACEMENT_PCT:
                continue

            x = np.arange(len(flag_prices))
            flag_res = linregress(x, flag_prices)
            flag_total_drift_pct = (flag_res.slope * len(flag_prices)) / pole_bottom_price * 100
            if flag_total_drift_pct < -0.5:
                continue

            # Calculate upper and lower bounds for the channel based on residuals of highs and lows
            line_vals = flag_res.intercept + flag_res.slope * x
            upper_intercept = flag_res.intercept + np.max(flag_highs - line_vals)
            lower_intercept = flag_res.intercept + np.min(flag_lows - line_vals)

            pattern = {
                "pattern_type":       "bear_flag",
                "ticker":             ticker,
                "verdict":            "VALID",
                "reject_reason":      "",
                "pole_start_price":   round(float(highs[best_pole["start_idx"]]), 2),
                "pole_bottom_price":  round(float(pole_bottom_price), 2),
                "flag_high_price":    round(float(flag_high), 2),
                "flag_low_price":     round(float(flag_low), 2),
                "pole_drop_pct":      round(float(best_pole["drop_pct"]), 2),
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
                pattern["pole_start_date"]  = str(dates[best_pole["start_idx"]])
                pattern["pole_bottom_date"] = str(dates[pole_end_idx])
                pattern["flag_start_date"]  = str(dates[flag_start])
                pattern["flag_end_date"]    = str(dates[flag_end])
                
            valid_structures.append(pattern)

    forming_candidates = {}

    # Evaluate breakdowns on all valid structures
    for pat in valid_structures:
        flag_end = pat["flag_end_idx"]
        flag_start = pat["flag_start_idx"]
        flag_slope = pat["flag_slope"]
        lower_intercept = pat["flag_lower_intercept"]
        
        breakdown_idx = None
        for k in range(flag_end + 1, min(n, flag_end + 6)):
            rel_k = k - flag_start
            lower_tl_val = lower_intercept + flag_slope * rel_k
            if prices[k] < lower_tl_val:
                breakdown_idx = k
                break

        is_forming = False
        if breakdown_idx is None:
            if flag_end >= n - 6:
                is_forming = True
            else:
                continue
            
        vol_sma50 = 0
        vol_pole_pass = None
        vol_flag_pass = None
        vol_breakdown_pass = None

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
                    bd_vol = float(volumes[breakdown_idx])
                    vol_breakdown_pass = bd_vol > (BREAKOUT_VOLUME_MULTIPLIER * vol_sma50)

        if not is_forming and REQUIRE_BREAKOUT_VOLUME and vol_breakdown_pass is False:
            continue
            
        quality = compute_quality_score("bear_flag", {
            "pole_change_pct": pat["pole_drop_pct"],
            "r_squared": pat["pole_r_squared"],
            "flag_range_pct": pat["flag_range_pct"],
            "vol_pole_pass": vol_pole_pass,
            "vol_flag_pass": vol_flag_pass,
            "vol_breakout_pass": vol_breakdown_pass,
        })
        
        pat = pat.copy()
        pat["status"] = "forming" if is_forming else "confirmed"
        pat["vol_pole_pass"] = vol_pole_pass
        pat["vol_flag_pass"] = vol_flag_pass
        pat["vol_breakdown_pass"] = vol_breakdown_pass
        pat["quality_score"] = quality
        
        if is_forming:
            pat["breakdown_idx"] = None
            pat["breakdown_price"] = None
            if dates is not None:
                pat["signal_date"] = None
                
            pole_key = pat["pole_start_idx"]
            if pole_key not in forming_candidates or pat["flag_end_idx"] > forming_candidates[pole_key]["flag_end_idx"]:
                forming_candidates[pole_key] = pat
        else:
            pat["breakdown_idx"] = breakdown_idx
            pat["breakdown_price"] = round(float(prices[breakdown_idx]), 2)
            if dates is not None:
                pat["signal_date"] = str(dates[breakdown_idx])
            patterns_found.append(pat)

    # Add the single best forming pattern per pole
    confirmed_poles = {p["pole_start_idx"] for p in patterns_found}
    for fpat in forming_candidates.values():
        if fpat["pole_start_idx"] not in confirmed_poles:
            patterns_found.append(fpat)

    # Deduplication
    patterns_found.sort(key=lambda p: p["quality_score"], reverse=True)
    final_patterns = []
    claimed = []
    for c in patterns_found:
        is_dup = False
        for (cs, ce) in claimed:
            # Handle None breakdown_idx for forming patterns
            ce_c = c.get("breakdown_idx")
            ce_s = ce
            
            if ce_c is not None and ce_s is not None:
                if abs(c["pole_start_idx"] - cs) <= 10 and abs(ce_c - ce_s) <= 10:
                    is_dup = True
                    break
            else:
                # If either is forming, we just check pole start proximity
                if abs(c["pole_start_idx"] - cs) <= 10:
                    is_dup = True
                    break
                    
        if not is_dup:
            final_patterns.append(c)
            claimed.append((c["pole_start_idx"], c.get("breakdown_idx")))

    return final_patterns
