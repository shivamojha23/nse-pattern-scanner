import numpy as np
import pandas as pd
import math
import datetime
import os
import sys
from scipy.signal import find_peaks
from scipy.stats import linregress
from .core import *

def detect_pennant(prices, highs=None, lows=None, volumes=None, ticker="UNKNOWN", dates=None, interval="1d", verbose=False):
    """
    Detects Pennant patterns with strict algorithmic constraints (Advanced Quant Implementation).
    """
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
    
    vol_sma20 = None
    if volumes is not None:
        vol_series = pd.Series(volumes)
        vol_sma20 = vol_series.rolling(window=20, min_periods=5).mean().values

    adx_arr = compute_adx(highs, lows, prices, period=14)
    valid_structures = []
        
    # Search for pole ends
    for pole_end_idx in range(20, n - MIN_PENNANT_CANDLES - 1):
        # Dynamic pole lookback (5 to 20 candles)
        best_pole_start = None
        best_change_pct = 0
        best_pole_dir = None
        
        # Calculate ATR at pole_end for dynamic threshold
        atr = compute_atr(highs[:pole_end_idx+1], lows[:pole_end_idx+1], prices[:pole_end_idx+1], period=14)
        atr_threshold = max(PENNANT_POLE_CHANGE_PCT, (3 * atr / prices[pole_end_idx]) * 100) if atr > 0 else PENNANT_POLE_CHANGE_PCT

        for lookback in range(5, 21): 
            pole_start_idx = pole_end_idx - lookback
            change_pct = (prices[pole_end_idx] - prices[pole_start_idx]) / prices[pole_start_idx] * 100
            
            if abs(change_pct) >= atr_threshold:
                if abs(change_pct) > abs(best_change_pct):
                    best_change_pct = change_pct
                    best_pole_start = pole_start_idx
                    best_pole_dir = "bullish" if change_pct > 0 else "bearish"
                    
        if best_pole_start is None:
            continue
            
        pole_start_idx = best_pole_start
        pole_direction = best_pole_dir
        change_pct = best_change_pct
        pole_len = pole_end_idx - pole_start_idx
        
        pole_adx = adx_arr[pole_end_idx]
        if np.isnan(pole_adx) or pole_adx <= PENNANT_ADX_THRESHOLD:
            continue
            
        max_pennant_len = min(MAX_PENNANT_CANDLES, n - pole_end_idx - 2)
        # Apply proportionality rule: pennant length <= 1.5 * pole_len
        max_pennant_len = min(max_pennant_len, int(1.5 * pole_len))
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
                
            # Shift trendlines to create true bounding boxes touching max wicks
            upper_line_vals = res_high.intercept + upper_slope * x
            max_pos_residual = np.max(pennant_highs - upper_line_vals)
            shifted_upper_intercept = res_high.intercept + max_pos_residual
            
            lower_line_vals = res_low.intercept + lower_slope * x
            max_neg_residual = np.min(pennant_lows - lower_line_vals)
            shifted_lower_intercept = res_low.intercept + max_neg_residual
            
            # Strict Symmetry Check (40% max diff to start slightly loose)
            slope_diff = abs(abs(upper_slope) - abs(lower_slope)) / max(abs(upper_slope), abs(lower_slope))
            if slope_diff > 0.40: 
                continue
                
            vol_pennant_pass = False
            vol_slope = 0.0
            if volumes is not None and vol_sma20 is not None:
                pennant_vols = volumes[pennant_start:pennant_end + 1]
                res_vol = linregress(x, pennant_vols)
                vol_slope = res_vol.slope
                avg_pennant_vol = np.mean(pennant_vols)
                sma20_at_end = vol_sma20[pennant_end]
                
                if res_vol.slope < 0 and avg_pennant_vol < sma20_at_end:
                    vol_pennant_pass = True
            else:
                vol_pennant_pass = True 
                
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
                "res_high_intercept": float(shifted_upper_intercept),
                "res_low_intercept":  float(shifted_lower_intercept),
                "symmetry_diff":      round(float(slope_diff), 3),
                "vol_slope":          float(vol_slope),
                "r_squared":          round(float(res_high.rvalue**2), 3),
                "vol_pennant_pass":   vol_pennant_pass,
                "pole_start_idx":     pole_start_idx,
                "pole_end_idx":       pole_end_idx,
                "pennant_start_idx":  pennant_start,
                "pennant_end_idx":    pennant_end,
                "pole_len":           pole_len,
            }

            if dates is not None:
                pattern["pole_start_date"]    = str(dates[pole_start_idx])
                pattern["pole_end_date"]      = str(dates[pole_end_idx])
                pattern["pennant_start_date"] = str(dates[pennant_start])
                pattern["pennant_end_date"]   = str(dates[pennant_end])
                
            valid_structures.append(pattern)

    forming_candidates = {}

    # Evaluate breakouts and apex on all valid structures
    for pat in valid_structures:
        pennant_end = pat["pennant_end_idx"]
        pennant_start = pat["pennant_start_idx"]
        pole_direction = pat["direction"]
        upper_intercept = pat["res_high_intercept"]
        lower_intercept = pat["res_low_intercept"]
        upper_slope = pat["upper_slope"]
        lower_slope = pat["lower_slope"]

        # Calculate Apex (Intersection of trendlines)
        x_apex = (lower_intercept - upper_intercept) / (upper_slope - lower_slope) if upper_slope != lower_slope else 9999

        breakout_idx = None
        vol_breakout_pass = False
        apex_ratio = 0.0

        for k in range(pennant_end + 1, min(n, pennant_end + 6)):
            rel_k = k - pennant_start
            upper_tl_val = upper_intercept + upper_slope * rel_k
            lower_tl_val = lower_intercept + lower_slope * rel_k
            
            is_breakout = False
            if pole_direction == "bullish" and prices[k] > upper_tl_val:
                is_breakout = True
            elif pole_direction == "bearish" and prices[k] < lower_tl_val:
                is_breakout = True
                
            if is_breakout:
                breakout_idx = k
                apex_ratio = rel_k / x_apex if x_apex > 0 else 0
                
                # Check volume breakout - price breakout is valid regardless
                if volumes is not None and vol_sma20 is not None:
                    if volumes[k] > PENNANT_BREAKOUT_VOL_MULT * vol_sma20[k-1]:
                        vol_breakout_pass = True
                else:
                    vol_breakout_pass = True
                break
                
        is_forming = False
        if breakout_idx is None:
            if pennant_end >= n - 6:
                # Reject forming if it's already exhausted into the apex
                current_rel_k = n - 1 - pennant_start
                current_apex_ratio = current_rel_k / x_apex if x_apex > 0 else 0
                if current_apex_ratio <= 0.85:
                    is_forming = True
                else:
                    continue
            else:
                continue

        # Check Apex Trap: Breakout should be between 40% and 85% of apex distance
        if breakout_idx is not None:
            if apex_ratio < 0.40 or apex_ratio > 0.85:
                continue 

        pat["apex_ratio"] = round(float(apex_ratio), 3)

        quality = compute_quality_score("pennant", {
            "pole_change_pct": pat["pole_change_pct"],
            "symmetry_diff": pat["symmetry_diff"],
            "vol_slope": pat["vol_slope"],
            "vol_pole_pass": True,
            "vol_pennant_pass": pat["vol_pennant_pass"],
            "vol_breakout_pass": vol_breakout_pass if not is_forming else None,
        })
        
        pat = pat.copy()
        pat["status"] = "forming" if is_forming else "confirmed"
        pat["vol_breakout_pass"] = vol_breakout_pass if not is_forming else None
        pat["quality_score"] = quality
        
        if is_forming:
            pat["breakout_idx"] = None
            pat["breakout_price"] = None
            if dates is not None:
                pat["signal_date"] = None
                pat["breakout_date"] = None
                
            pole_key = pat["pole_start_idx"]
            if pole_key not in forming_candidates or pat["quality_score"] > forming_candidates[pole_key]["quality_score"]:
                forming_candidates[pole_key] = pat
        else:
            pat["breakout_idx"] = breakout_idx
            pat["breakout_price"] = round(float(prices[breakout_idx]), 2)
            if dates is not None:
                pat["signal_date"] = str(dates[breakout_idx])
                pat["breakout_date"] = str(dates[breakout_idx])
                
            patterns_found.append(pat)

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
        if c.get("status") == "forming":
            for (cs, ce) in claimed:
                if abs(c["pole_start_idx"] - cs) <= 10:
                    is_dup = True
                    break
        else:
            for (cs, ce) in claimed:
                if ce is not None and abs(c["pole_start_idx"] - cs) <= 10 and abs(c["breakout_idx"] - ce) <= 10:
                    is_dup = True
                    break
        
        if not is_dup:
            final_patterns.append(c)
            claimed.append((c["pole_start_idx"], c.get("breakout_idx")))

    return final_patterns
