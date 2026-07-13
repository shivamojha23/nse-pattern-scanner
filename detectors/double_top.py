import numpy as np
import pandas as pd
import math
import datetime
import os
import sys
from scipy.signal import find_peaks
from scipy.stats import linregress
from .core import *

def detect_double_top(prices, volumes=None, ticker="UNKNOWN", dates=None, interval="1d", verbose=False):
    """
    Detects Double Top reversal patterns.
    """
    import numpy as np
    import pandas as pd
    from scipy.signal import find_peaks

    prices = np.array(prices, dtype=float)
    if volumes is not None:
        volumes = np.array(volumes, dtype=float)

    n = len(prices)
    if n < 30:
        return []

    
    # Slow Path: Full Detection
    smoothed = smooth_prices(prices, window=SMOOTHING_WINDOW)
    min_prominence = np.median(smoothed) * 0.01
    peak_indices, _ = find_peaks(smoothed, distance=8, prominence=min_prominence)

    if len(peak_indices) < 2:
        return []

    valid_structures = []
    candidates = []

    for i in range(len(peak_indices)):
        top1_idx = peak_indices[i]
        top2_candidates = []
        if i + 1 < len(peak_indices):
            top2_candidates.append((peak_indices[i + 1], False))
        if top1_idx >= n - 40:
            top2_candidates.append((n - 1, True))
        for top2_idx, is_stage_b in top2_candidates:

            top1_price = prices[top1_idx]
            top2_price = prices[top2_idx]

            separation = top2_idx - top1_idx
            if separation < MIN_DOUBLE_CANDLES:
                continue

            similarity_pct = abs(top1_price - top2_price) / top1_price * 100
            if similarity_pct > DOUBLE_TOP_SIMILARITY_PCT:
                if is_stage_b and top2_price < top1_price and similarity_pct <= DOUBLE_TOP_SIMILARITY_PCT + 3:
                    pass 
                else:
                    continue

            valley_prices = prices[top1_idx:top2_idx + 1]
            valley_rel_idx = int(np.argmin(valley_prices))
            valley_idx = top1_idx + valley_rel_idx
            valley_price = prices[valley_idx]

            avg_top = (top1_price + top2_price) / 2
            valley_drop_pct = (avg_top - valley_price) / avg_top * 100
            if valley_drop_pct < MIN_VALLEY_DROP_PCT:
                continue

            breakdown_idx = None
            for k in range(top2_idx + 1, min(n, top2_idx + 30)):
                if prices[k] < valley_price:
                    breakdown_idx = k
                    break

            status = "confirmed"
            stage = ""
            if breakdown_idx is None:
                if is_stage_b and similarity_pct > DOUBLE_TOP_SIMILARITY_PCT:
                    status = "forming"
                    stage = "Stage B - Approaching second peak: Price rallying back up toward first peak"
                    breakdown_idx = n - 1
                elif not is_stage_b or (is_stage_b and similarity_pct <= DOUBLE_TOP_SIMILARITY_PCT):
                    status = "forming"
                    stage = "Stage C - Second peak formed, awaiting confirmation: Price touched first peak, hasn't broken neckline"
                    breakdown_idx = n - 1
                else:
                    continue

            vol_sma50 = 0
            vol_top1_avg = 0
            vol_top2_avg = 0
            vol_pattern_pass = None
            vol_breakdown_pass = None

            if volumes is not None and len(volumes) == n:
                vol_series = pd.Series(volumes)
                vol_sma = vol_series.rolling(window=VOLUME_SMA_PERIOD, min_periods=10).mean()
                sma_idx = min(top2_idx, len(vol_sma) - 1)
                vol_sma50 = float(vol_sma.iloc[sma_idx]) if not pd.isna(vol_sma.iloc[sma_idx]) else 0

                def avg_vol(idx, radius=3):
                    s = max(0, idx - radius)
                    e = min(n, idx + radius + 1)
                    return float(np.mean(volumes[s:e]))

                vol_top1_avg = avg_vol(top1_idx)
                vol_top2_avg = avg_vol(top2_idx)

                vol_pattern_pass = vol_top1_avg > vol_top2_avg

                if vol_sma50 > 0 and status == "confirmed":
                    bd_vol = float(volumes[breakdown_idx])
                    vol_breakdown_pass = bd_vol > (BREAKOUT_VOLUME_MULTIPLIER * vol_sma50)

            quality = compute_quality_score("double_top", {
                "similarity_pct": similarity_pct,
                "depth_pct": valley_drop_pct,
                "vol_pattern_pass": vol_pattern_pass,
                "vol_breakout_pass": vol_breakdown_pass,
            })

            pattern = {
                "pattern_type": "double_top",
                "ticker": ticker,
                "verdict": "VALID",
                "reject_reason": "",
                "status": status,
                "stage": stage,
                "first_top_price": round(float(top1_price), 2),
                "valley_price": round(float(valley_price), 2),
                "second_top_price": round(float(top2_price), 2),
                "breakdown_price": round(float(prices[breakdown_idx]), 2) if status == "confirmed" else None,
                "similarity_pct": round(float(similarity_pct), 2),
                "valley_drop_pct": round(float(valley_drop_pct), 2),
                "separation": int(separation),
                "vol_top1_avg": round(float(vol_top1_avg), 0),
                "vol_top2_avg": round(float(vol_top2_avg), 0),
                "vol_pattern_pass": vol_pattern_pass,
                "vol_breakdown_pass": vol_breakdown_pass,
                "vol_sma50": vol_sma50,
                "quality_score": quality,
                "first_top_idx": int(top1_idx),
                "valley_idx": int(valley_idx),
                "second_top_idx": int(top2_idx),
                "breakdown_idx": int(breakdown_idx),
            }

            if dates is not None:
                pattern["first_top_date"] = str(dates[top1_idx])
                pattern["valley_date"] = str(dates[valley_idx])
                pattern["second_top_date"] = str(dates[top2_idx])
                if status == "confirmed":
                    pattern["breakdown_date"] = str(dates[breakdown_idx])
                    pattern["signal_date"] = str(dates[breakdown_idx])

            candidates.append(pattern)
            
            # Cache completed, non-forming structures
            if not is_stage_b:
                valid_structures.append(pattern)

    # Deduplication for slow path output
    candidates.sort(key=lambda p: p["quality_score"], reverse=True)
    patterns_found = []
    claimed = []

    for c in candidates:
        is_dup = False
        for (c1, c2) in claimed:
            if abs(c["first_top_idx"] - c1) <= 10 and abs(c["second_top_idx"] - c2) <= 10:
                is_dup = True
                break
        if not is_dup:
            patterns_found.append(c)
            claimed.append((c["first_top_idx"], c["second_top_idx"]))

    return patterns_found
