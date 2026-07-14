import numpy as np
import pandas as pd
import math
import datetime
import os
import sys
from scipy.signal import find_peaks
from scipy.stats import linregress
from .core import *

def detect_bear_flag(prices, volumes=None, ticker="UNKNOWN", dates=None,
                     interval="1d", verbose=False):
    """
    Detects Bear Flag patterns using structural_cache.
    """
    from scipy.stats import linregress
    import numpy as np
    import pandas as pd
    
    prices = np.array(prices, dtype=float)
    if volumes is not None:
        volumes = np.array(volumes, dtype=float)

    n = len(prices)
    if n < 30:
        return []

    patterns_found = []

    # Slow Path: Full Detection
    rolling_max = pd.Series(prices).rolling(window=MAX_POLE_CANDLES, min_periods=5).max().values
    potential_drop = np.zeros(n)
    for idx in range(MAX_POLE_CANDLES, n):
        if rolling_max[idx] > 0:
            potential_drop[idx] = (rolling_max[idx] - prices[idx]) / rolling_max[idx] * 100
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

            drop_pct = (prices[pole_start_idx] - prices[pole_end_idx]) / prices[pole_start_idx] * 100
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

        pole_bottom_price = prices[pole_end_idx]

        for flag_len in range(MIN_FLAG_CANDLES, min(MAX_FLAG_CANDLES + 1, n - pole_end_idx)):
            flag_start = pole_end_idx + 1
            flag_end = pole_end_idx + flag_len
            if flag_end >= n:
                break

            flag_prices = prices[flag_start:flag_end + 1]
            if len(flag_prices) < MIN_FLAG_CANDLES:
                continue

            flag_high = float(np.max(flag_prices))
            flag_low = float(np.min(flag_prices))

            if flag_low < pole_bottom_price * 0.999:
                continue

            flag_range_pct = (flag_high - flag_low) / pole_bottom_price * 100
            if flag_range_pct > FLAG_CHANNEL_PCT:
                continue

            pole_size = prices[best_pole["start_idx"]] - pole_bottom_price
            retracement_pct = (flag_high - pole_bottom_price) / pole_size * 100 if pole_size > 0 else 0
            if retracement_pct > MAX_RETRACEMENT_PCT:
                continue

            x = np.arange(len(flag_prices))
            flag_res = linregress(x, flag_prices)
            flag_total_drift_pct = (flag_res.slope * len(flag_prices)) / pole_bottom_price * 100
            if flag_total_drift_pct < -0.5:
                continue

            pattern = {
                "pattern_type":       "bear_flag",
                "ticker":             ticker,
                "verdict":            "VALID",
                "reject_reason":      "",
                "pole_start_price":   round(float(prices[best_pole["start_idx"]]), 2),
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
            }

            if dates is not None:
                pattern["pole_start_date"]  = str(dates[best_pole["start_idx"]])
                pattern["pole_bottom_date"] = str(dates[pole_end_idx])
                pattern["flag_start_date"]  = str(dates[flag_start])
                pattern["flag_end_date"]    = str(dates[flag_end])
                
            valid_structures.append(pattern)
            break

    # Evaluate breakdowns on all valid structures
    for pat in valid_structures:
        flag_end = pat["flag_end_idx"]
        pole_bottom_price = pat["pole_bottom_price"]
        breakdown_idx = None
        for k in range(flag_end + 1, min(n, flag_end + 6)):
            if prices[k] < pole_bottom_price:
                breakdown_idx = k
                break

        if breakdown_idx is None:
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

                bd_vol = float(volumes[breakdown_idx])
                vol_breakdown_pass = bd_vol > (BREAKOUT_VOLUME_MULTIPLIER * vol_sma50)

        if REQUIRE_BREAKOUT_VOLUME and vol_breakdown_pass is False:
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
        pat["breakdown_idx"] = breakdown_idx
        pat["breakdown_price"] = round(float(prices[breakdown_idx]), 2)
        pat["vol_pole_pass"] = vol_pole_pass
        pat["vol_flag_pass"] = vol_flag_pass
        pat["vol_breakdown_pass"] = vol_breakdown_pass
        pat["quality_score"] = quality
        
        if dates is not None:
            pat["signal_date"] = str(dates[breakdown_idx])
            
        patterns_found.append(pat)

    # Deduplication
    patterns_found.sort(key=lambda p: p["quality_score"], reverse=True)
    final_patterns = []
    claimed = []
    for c in patterns_found:
        is_dup = False
        for (cs, ce) in claimed:
            if abs(c["pole_start_idx"] - cs) <= 10 and abs(c["breakdown_idx"] - ce) <= 10:
                is_dup = True
                break
        if not is_dup:
            final_patterns.append(c)
            claimed.append((c["pole_start_idx"], c["breakdown_idx"]))

    return final_patterns
