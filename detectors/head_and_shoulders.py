import numpy as np
import pandas as pd
import math
import datetime
import os
import sys
from scipy.signal import find_peaks

from .core import *

def detect_head_and_shoulders(prices, highs=None, lows=None, volumes=None, ticker="UNKNOWN",
                               dates=None, interval="1d", verbose=False, refine_window=4):
    """
    Detects Head and Shoulders reversal patterns.
    """
    import numpy as np
    import pandas as pd
    from scipy.signal import find_peaks
    
    prices = np.array(prices, dtype=float)
    if volumes is not None:
        volumes = np.array(volumes, dtype=float)

    n = len(prices)
    if n < 40:
        return []

    patterns_found = []

    # Slow Path: Full Detection
    smoothed = smooth_prices(prices, window=SMOOTHING_WINDOW)
    min_prominence = np.median(smoothed) * 0.01

    peak_indices, _ = find_peaks(smoothed, distance=8, prominence=min_prominence)
    trough_indices, _ = find_peaks(-smoothed, distance=8, prominence=min_prominence)

    if len(peak_indices) < 3 or len(trough_indices) < 2:
        return []

    valid_structures = []

    for i in range(len(peak_indices) - 2):
        for j in range(i + 1, len(peak_indices) - 1):
            for k in range(j + 1, len(peak_indices)):
                ls_approx = peak_indices[i]
                head_approx = peak_indices[j]
                rs_approx = peak_indices[k]
                
                ls_idx = refine_peak(ls_approx, prices, window=refine_window)
                head_idx = refine_peak(head_approx, prices, window=refine_window)
                rs_idx = refine_peak(rs_approx, prices, window=refine_window)
                
                if not (ls_idx < head_idx < rs_idx):
                    continue

                ls_price = prices[ls_idx]
                head_price = prices[head_idx]
                rs_price = prices[rs_idx]

                span = rs_idx - ls_idx
                if span < MIN_HS_CANDLES:
                    continue

                ls_to_head = head_idx - ls_idx
                head_to_rs = rs_idx - head_idx
        
                if ls_to_head < MIN_LS_TO_HEAD_CANDLES or ls_to_head > MAX_LS_TO_HEAD_CANDLES:
                    continue
                if head_to_rs < MIN_HEAD_TO_RS_CANDLES or head_to_rs > MAX_HEAD_TO_RS_CANDLES:
                    continue
            
                duration_ratio = ls_to_head / head_to_rs if head_to_rs > 0 else 0
                if not (0.6 <= duration_ratio <= 1.66):
                    continue

                ls_prominence = (head_price - ls_price) / ls_price * 100
                rs_prominence = (head_price - rs_price) / rs_price * 100

                if ls_prominence < HEAD_SHOULDER_RATIO or rs_prominence < HEAD_SHOULDER_RATIO:
                    continue

                shoulder_diff_pct = abs(ls_price - rs_price) / ls_price * 100
                if shoulder_diff_pct > SHOULDER_SYMMETRY_PCT:
                    continue

                left_troughs = [t for t in trough_indices if ls_idx < t < head_idx]
                right_troughs = [t for t in trough_indices if head_idx < t < rs_idx]

                if not left_troughs or not right_troughs:
                    continue

                left_trough_approx = min(left_troughs, key=lambda t: prices[t])
                right_trough_approx = min(right_troughs, key=lambda t: prices[t])
        
                left_neck_idx = refine_trough(left_trough_approx, prices, window=refine_window)
                right_neck_idx = refine_trough(right_trough_approx, prices, window=refine_window)
        
                left_neck_price = prices[left_neck_idx]
                right_neck_price = prices[right_neck_idx]

                if head_idx - left_neck_idx < 3 or right_neck_idx - head_idx < 3:
                    continue

                trough_diff_pct = abs(left_neck_price - right_neck_price) / max(left_neck_price, right_neck_price) * 100
                if trough_diff_pct > TROUGH_MAX_DIFF_PCT:
                    continue

                if highs is not None and lows is not None:
                    atr = compute_atr(highs[:rs_idx+1], lows[:rs_idx+1], prices[:rs_idx+1])
                    if atr > 0:
                        head_depth = head_price - min(left_neck_price, right_neck_price)
                        if head_depth < ATR_HS_MULTIPLIER * atr:
                            continue

                lower_shoulder = min(ls_price, rs_price)
                left_trough_depth_pct = (lower_shoulder - left_neck_price) / lower_shoulder
                right_trough_depth_pct = (lower_shoulder - right_neck_price) / lower_shoulder
        
                if left_trough_depth_pct > TROUGH_MAX_DEPTH_PCT or right_trough_depth_pct > TROUGH_MAX_DEPTH_PCT:
                    continue

                neck_span = right_neck_idx - left_neck_idx
                if neck_span <= 0:
                    continue
                neckline_slope = (right_neck_price - left_neck_price) / neck_span
                neckline_slope_pct = abs(right_neck_price - left_neck_price) / left_neck_price * 100

                vol_sma50 = 0
                vol_ls_avg = 0
                vol_head_avg = 0
                vol_rs_avg = 0
                vol_progression_pass = None

                if volumes is not None and len(volumes) == n:
                    vol_series = pd.Series(volumes)
                    vol_sma = vol_series.rolling(window=VOLUME_SMA_PERIOD, min_periods=10).mean()
                    sma_idx = min(rs_idx, len(vol_sma) - 1)
                    vol_sma50 = float(vol_sma.iloc[sma_idx]) if not pd.isna(vol_sma.iloc[sma_idx]) else 0

                    def avg_vol_around(idx, radius=3):
                        start = max(0, idx - radius)
                        end = min(n, idx + radius + 1)
                        return float(np.mean(volumes[start:end]))

                    vol_ls_avg = avg_vol_around(ls_idx)
                    vol_head_avg = avg_vol_around(head_idx)
                    vol_rs_avg = avg_vol_around(rs_idx)

                    vol_progression_pass = (vol_ls_avg > vol_head_avg > vol_rs_avg)

                head_prom_pct = (ls_prominence + rs_prominence) / 2

                pattern = {
                    "pattern_type":          "head_and_shoulders",
                    "ticker":                ticker,
                    "verdict":               "VALID",
                    "reject_reason":         "",
                    "left_shoulder_price":   round(float(ls_price), 2),
                    "head_price":            round(float(head_price), 2),
                    "right_shoulder_price":  round(float(rs_price), 2),
                    "left_neckline_price":   round(float(left_neck_price), 2),
                    "right_neckline_price":  round(float(right_neck_price), 2),
                    "head_vs_ls_pct":        round(float(ls_prominence), 2),
                    "head_vs_rs_pct":        round(float(rs_prominence), 2),
                    "shoulder_symmetry_pct": round(float(shoulder_diff_pct), 2),
                    "neckline_slope":        float(neckline_slope),
                    "neckline_slope_pct":    round(float(neckline_slope_pct), 2),
                    "span_candles":          int(span),
                    "vol_ls_avg":            round(float(vol_ls_avg), 0),
                    "vol_head_avg":          round(float(vol_head_avg), 0),
                    "vol_rs_avg":            round(float(vol_rs_avg), 0),
                    "vol_progression_pass":  vol_progression_pass,
                    "vol_sma50":             vol_sma50,
                    "head_prom_pct":         head_prom_pct,
                    "left_shoulder_idx":     int(ls_idx),
                    "head_idx":              int(head_idx),
                    "right_shoulder_idx":    int(rs_idx),
                    "left_neckline_idx":     int(left_neck_idx),
                    "right_neckline_idx":    int(right_neck_idx),
                }

                if dates is not None:
                    pattern["left_shoulder_date"]  = str(dates[ls_idx])
                    pattern["head_date"]           = str(dates[head_idx])
                    pattern["right_shoulder_date"] = str(dates[rs_idx])
                    pattern["left_neckline_date"]  = str(dates[left_neck_idx])
                    pattern["right_neckline_date"] = str(dates[right_neck_idx])

                valid_structures.append(pattern)

    forming_candidates = {}

    # Evaluate breakdowns on all valid structures
    for pat in valid_structures:
        rs_idx = pat["right_shoulder_idx"]
        left_neck_idx = pat["left_neckline_idx"]
        left_neck_price = pat["left_neckline_price"]
        neckline_slope = pat["neckline_slope"]

        breakdown_idx = None
        for k in range(rs_idx + 1, min(n, rs_idx + 30)):
            neckline_at_k = left_neck_price + neckline_slope * (k - left_neck_idx)
            if prices[k] < neckline_at_k:
                breakdown_idx = k
                break

        is_forming = False
        if breakdown_idx is None:
            if rs_idx >= n - 30:
                is_forming = True
            else:
                continue

        vol_breakdown_pass = None
        if volumes is not None and breakdown_idx is not None:
            vol_sma50 = pat.get("vol_sma50", 0)
            if vol_sma50 > 0:
                bd_vol = float(volumes[breakdown_idx])
                vol_breakdown_pass = bd_vol > (BREAKOUT_VOLUME_MULTIPLIER * vol_sma50)

        quality = compute_quality_score("head_and_shoulders", {
            "head_prominence_pct": pat["head_prom_pct"],
            "shoulder_symmetry_pct": pat["shoulder_symmetry_pct"],
            "neckline_slope_pct": pat["neckline_slope_pct"],
            "vol_progression_pass": pat["vol_progression_pass"],
            "vol_breakdown_pass": vol_breakdown_pass,
            "right_trough_lower": pat["right_neckline_price"] < pat["left_neckline_price"],
        })

        pat = pat.copy()
        pat["status"] = "forming" if is_forming else "confirmed"
        pat["vol_breakdown_pass"] = vol_breakdown_pass if not is_forming else None
        pat["quality_score"] = quality

        if is_forming:
            pat["breakdown_idx"] = None
            pat["breakdown_price"] = None
            if dates is not None:
                pat["signal_date"] = None
                pat["breakdown_date"] = None
                
            head_key = pat["head_idx"]
            if head_key not in forming_candidates or pat["right_shoulder_idx"] > forming_candidates[head_key]["right_shoulder_idx"]:
                forming_candidates[head_key] = pat
        else:
            pat["breakdown_idx"] = breakdown_idx
            pat["breakdown_price"] = round(float(prices[breakdown_idx]), 2)
            if dates is not None:
                pat["signal_date"] = str(dates[breakdown_idx])
                pat["breakdown_date"] = str(dates[breakdown_idx])
                
            patterns_found.append(pat)

    confirmed_heads = {p["head_idx"] for p in patterns_found}
    for fpat in forming_candidates.values():
        if fpat["head_idx"] not in confirmed_heads:
            patterns_found.append(fpat)

    # Deduplication
    patterns_found.sort(key=lambda p: (p["head_price"], p["quality_score"]), reverse=True)
    final_patterns = []
    claimed = []

    for c in patterns_found:
        is_dup = False
        for (ch, cr) in claimed:
            if abs(c["head_idx"] - ch) <= 10 and abs(c["right_shoulder_idx"] - cr) <= 10:
                is_dup = True
                break
        if not is_dup:
            final_patterns.append(c)
            claimed.append((c["head_idx"], c["right_shoulder_idx"]))

    return final_patterns
