import numpy as np
import pandas as pd
import math
import datetime
import os
import sys
from scipy.signal import find_peaks
from scipy.stats import linregress
from .core import *

def detect_pennant(prices, highs=None, lows=None, volumes=None, ticker="UNKNOWN", dates=None,
                   interval="1d", verbose=False):
    """
    Detects Pennant patterns with strict algorithmic constraints.
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
        
    vol_sma20 = None
    if volumes is not None:
        vol_series = pd.Series(volumes)
        vol_sma20 = vol_series.rolling(window=20, min_periods=5).mean().values

    
    # Slow Path: Full Detection
    adx_arr = compute_adx(highs, lows, prices, period=14)
    valid_structures = []
        
    for pole_end_idx in range(PENNANT_POLE_PERIODS, n - MIN_PENNANT_CANDLES - 1):
        pole_start_idx = pole_end_idx - PENNANT_POLE_PERIODS
        change_pct = (prices[pole_end_idx] - prices[pole_start_idx]) / prices[pole_start_idx] * 100
        
        is_bullish = change_pct >= PENNANT_POLE_CHANGE_PCT
        is_bearish = change_pct <= -PENNANT_POLE_CHANGE_PCT
        
        if not is_bullish and not is_bearish:
            continue
            
        pole_adx = adx_arr[pole_end_idx]
        if np.isnan(pole_adx) or pole_adx <= PENNANT_ADX_THRESHOLD:
            continue
            
        pole_direction = "bullish" if is_bullish else "bearish"
        
        max_pennant_len = min(MAX_PENNANT_CANDLES, n - pole_end_idx - 2)
        if max_pennant_len < MIN_PENNANT_CANDLES:
            continue
            
        for plen in range(MIN_PENNANT_CANDLES, max_pennant_len + 1):
            pennant_start = pole_end_idx + 1
            pennant_end = pole_end_idx + plen
            
            pennant_highs = highs[pennant_start:pennant_end + 1]
            pennant_lows = lows[pennant_start:pennant_end + 1]
            
            x = np.arange(len(pennant_highs))
            res_high = linregress(x, pennant_highs)
            res_low = linregress(x, pennant_lows)
            
            upper_slope = res_high.slope
            lower_slope = res_low.slope
            
            if upper_slope >= 0 or lower_slope <= 0:
                continue
                
            dist_start = res_high.intercept - res_low.intercept
            dist_end = (res_high.intercept + upper_slope * len(x)) - (res_low.intercept + lower_slope * len(x))
            
            if dist_end >= dist_start:
                continue
                
            vol_pennant_pass = False
            if volumes is not None and vol_sma20 is not None:
                pennant_vols = volumes[pennant_start:pennant_end + 1]
                res_vol = linregress(x, pennant_vols)
                avg_pennant_vol = np.mean(pennant_vols)
                sma20_at_end = vol_sma20[pennant_end]
                
                if res_vol.slope < 0 and avg_pennant_vol < sma20_at_end:
                    vol_pennant_pass = True
            else:
                vol_pennant_pass = True 
                
            if not vol_pennant_pass:
                continue
                
            pattern = {
                "pattern_type":       "pennant",
                "ticker":             ticker,
                "verdict":            "VALID",
                "reject_reason":      "",
                "direction":          pole_direction,
                "pole_start_price":   round(float(prices[pole_start_idx]), 2),
                "pole_end_price":     round(float(prices[pole_end_idx]), 2),
                "pennant_high":       round(float(np.max(pennant_highs)), 2),
                "pennant_low":        round(float(np.min(pennant_lows)), 2),
                "pole_change_pct":    round(float(change_pct), 2),
                "adx_value":          round(float(pole_adx), 2),
                "upper_slope":        float(upper_slope),
                "lower_slope":        float(lower_slope),
                "res_high_intercept": float(res_high.intercept),
                "res_low_intercept":  float(res_low.intercept),
                "convergence_ratio":  round(float(dist_end / dist_start if dist_start > 0 else 1.0), 3),
                "r_squared":          round(float(res_high.rvalue**2), 3),
                "vol_pennant_pass":   vol_pennant_pass,
                "pole_start_idx":     pole_start_idx,
                "pole_end_idx":       pole_end_idx,
                "pennant_start_idx":  pennant_start,
                "pennant_end_idx":    pennant_end,
            }

            if dates is not None:
                pattern["pole_start_date"]    = str(dates[pole_start_idx])
                pattern["pole_end_date"]      = str(dates[pole_end_idx])
                pattern["pennant_start_date"] = str(dates[pennant_start])
                pattern["pennant_end_date"]   = str(dates[pennant_end])
                
            valid_structures.append(pattern)
            break 

    # Evaluate breakouts on all valid structures
    for pat in valid_structures:
        pennant_end = pat["pennant_end_idx"]
        pennant_start = pat["pennant_start_idx"]
        pole_direction = pat["direction"]
        res_high_intercept = pat["res_high_intercept"]
        res_low_intercept = pat["res_low_intercept"]
        upper_slope = pat["upper_slope"]
        lower_slope = pat["lower_slope"]

        breakout_idx = None
        vol_breakout_pass = False

        for k in range(pennant_end + 1, min(n, pennant_end + 6)):
            rel_k = k - pennant_start
            upper_tl_val = res_high_intercept + upper_slope * rel_k
            lower_tl_val = res_low_intercept + lower_slope * rel_k
            
            if pole_direction == "bullish" and prices[k] > upper_tl_val:
                if volumes is not None and vol_sma20 is not None:
                    if volumes[k] > PENNANT_BREAKOUT_VOL_MULT * vol_sma20[k-1]:
                        breakout_idx = k
                        vol_breakout_pass = True
                        break
                else:
                    breakout_idx = k
                    vol_breakout_pass = True
                    break
                    
            elif pole_direction == "bearish" and prices[k] < lower_tl_val:
                if volumes is not None and vol_sma20 is not None:
                    if volumes[k] > PENNANT_BREAKOUT_VOL_MULT * vol_sma20[k-1]:
                        breakout_idx = k
                        vol_breakout_pass = True
                        break
                else:
                    breakout_idx = k
                    vol_breakout_pass = True
                    break

        if breakout_idx is None:
            continue
            
        quality = compute_quality_score("pennant", {
            "pole_change_pct": pat["pole_change_pct"],
            "convergence_ratio": pat["convergence_ratio"],
            "r_squared": pat["r_squared"],
            "vol_pole_pass": True,
            "vol_pennant_pass": pat["vol_pennant_pass"],
            "vol_breakout_pass": vol_breakout_pass,
        })
        
        pat = pat.copy()
        pat["breakout_idx"] = breakout_idx
        pat["breakout_price"] = round(float(prices[breakout_idx]), 2)
        pat["vol_breakout_pass"] = vol_breakout_pass
        pat["quality_score"] = quality
        
        if dates is not None:
            pat["signal_date"] = str(dates[breakout_idx])
            pat["breakout_date"] = str(dates[breakout_idx])
            
        patterns_found.append(pat)

    # Deduplication
    patterns_found.sort(key=lambda p: p["quality_score"], reverse=True)
    final_patterns = []
    claimed = []

    for c in patterns_found:
        is_dup = False
        for (cs, ce) in claimed:
            if abs(c["pole_start_idx"] - cs) <= 10 and abs(c["breakout_idx"] - ce) <= 10:
                is_dup = True
                break
        if not is_dup:
            final_patterns.append(c)
            claimed.append((c["pole_start_idx"], c["breakout_idx"]))

    return final_patterns
