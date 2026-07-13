import numpy as np
import pandas as pd
import math
import datetime
import os
import sys
from scipy.signal import find_peaks
from scipy.stats import linregress
from .core import *

def detect_double_bottom(prices, volumes=None, ticker="UNKNOWN", dates=None, interval="1d", verbose=False):
    """
    Detects Double Bottom reversal patterns.
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
    trough_indices, _ = find_peaks(-smoothed, distance=8, prominence=min_prominence)

    if len(trough_indices) < 2:
        return []

    valid_structures = []
    candidates = []

    for i in range(len(trough_indices)):
        bot1_idx = trough_indices[i]
        bot2_candidates = []
        if i + 1 < len(trough_indices):
            bot2_candidates.append((trough_indices[i + 1], False))
        if bot1_idx >= n - 40:
            bot2_candidates.append((n - 1, True))
        for bot2_idx, is_stage_b in bot2_candidates:

            bot1_price = prices[bot1_idx]
            bot2_price = prices[bot2_idx]

            separation = bot2_idx - bot1_idx
            if separation < MIN_DOUBLE_CANDLES:
                continue

            similarity_pct = abs(bot1_price - bot2_price) / bot1_price * 100
            if similarity_pct > DOUBLE_BOTTOM_SIMILARITY_PCT:
                if is_stage_b and bot2_price > bot1_price and similarity_pct <= DOUBLE_BOTTOM_SIMILARITY_PCT + 3:
                    pass 
                else:
                    continue

            between_prices = prices[bot1_idx:bot2_idx + 1]
            peak_rel_idx = int(np.argmax(between_prices))
            peak_idx = bot1_idx + peak_rel_idx
            peak_price = prices[peak_idx]

            avg_bot = (bot1_price + bot2_price) / 2
            peak_rise_pct = (peak_price - avg_bot) / avg_bot * 100
            if peak_rise_pct < MIN_PEAK_RISE_PCT:
                continue

            breakout_idx = None
            for k in range(bot2_idx + 1, min(n, bot2_idx + 30)):
                if prices[k] > peak_price:
                    breakout_idx = k
                    break

            status = "confirmed"
            stage = ""
            if breakout_idx is None:
                if is_stage_b and similarity_pct > DOUBLE_BOTTOM_SIMILARITY_PCT:
                    status = "forming"
                    stage = "Stage B - Approaching second bottom: Price dropping back down toward first bottom"
                    breakout_idx = n - 1
                elif not is_stage_b or (is_stage_b and similarity_pct <= DOUBLE_BOTTOM_SIMILARITY_PCT):
                    status = "forming"
                    stage = "Stage C - Second bottom formed, awaiting confirmation: Price touched first bottom, hasn't broken neckline"
                    breakout_idx = n - 1
                else:
                    continue

            vol_sma50 = 0
            vol_bot1_avg = 0
            vol_bot2_avg = 0
            vol_pattern_pass = None
            vol_breakout_pass = None

            if volumes is not None and len(volumes) == n:
                vol_series = pd.Series(volumes)
                vol_sma = vol_series.rolling(window=VOLUME_SMA_PERIOD, min_periods=10).mean()
                sma_idx = min(bot2_idx, len(vol_sma) - 1)
                vol_sma50 = float(vol_sma.iloc[sma_idx]) if not pd.isna(vol_sma.iloc[sma_idx]) else 0

                def avg_vol(idx, radius=3):
                    s = max(0, idx - radius)
                    e = min(n, idx + radius + 1)
                    return float(np.mean(volumes[s:e]))

                vol_bot1_avg = avg_vol(bot1_idx)
                vol_bot2_avg = avg_vol(bot2_idx)

                vol_pattern_pass = vol_bot1_avg > vol_bot2_avg

                if vol_sma50 > 0 and status == "confirmed":
                    bo_vol = float(volumes[breakout_idx])
                    vol_breakout_pass = bo_vol > (BREAKOUT_VOLUME_MULTIPLIER * vol_sma50)

            quality = compute_quality_score("double_bottom", {
                "similarity_pct": similarity_pct,
                "depth_pct": peak_rise_pct,
                "vol_pattern_pass": vol_pattern_pass,
                "vol_breakout_pass": vol_breakout_pass,
            })

            pattern = {
                "pattern_type": "double_bottom",
                "ticker": ticker,
                "verdict": "VALID",
                "reject_reason": "",
                "status": status,
                "stage": stage,
                "first_bottom_price": round(float(bot1_price), 2),
                "peak_price": round(float(peak_price), 2),
                "second_bottom_price": round(float(bot2_price), 2),
                "breakout_price": round(float(prices[breakout_idx]), 2) if status == "confirmed" else None,
                "similarity_pct": round(float(similarity_pct), 2),
                "peak_rise_pct": round(float(peak_rise_pct), 2),
                "separation": int(separation),
                "vol_bot1_avg": round(float(vol_bot1_avg), 0),
                "vol_bot2_avg": round(float(vol_bot2_avg), 0),
                "vol_pattern_pass": vol_pattern_pass,
                "vol_breakout_pass": vol_breakout_pass,
                "vol_sma50": vol_sma50,
                "quality_score": quality,
                "first_bottom_idx": int(bot1_idx),
                "peak_idx": int(peak_idx),
                "second_bottom_idx": int(bot2_idx),
                "breakout_idx": int(breakout_idx),
            }

            if dates is not None:
                pattern["first_bottom_date"] = str(dates[bot1_idx])
                pattern["peak_date"] = str(dates[peak_idx])
                pattern["second_bottom_date"] = str(dates[bot2_idx])
                if status == "confirmed":
                    pattern["breakout_date"] = str(dates[breakout_idx])
                    pattern["signal_date"] = str(dates[breakout_idx])

            candidates.append(pattern)
            
            if not is_stage_b:
                valid_structures.append(pattern)

    # Deduplication for slow path output
    candidates.sort(key=lambda p: p["quality_score"], reverse=True)
    patterns_found = []
    claimed = []

    for c in candidates:
        is_dup = False
        for (c1, c2) in claimed:
            if abs(c["first_bottom_idx"] - c1) <= 10 and abs(c["second_bottom_idx"] - c2) <= 10:
                is_dup = True
                break
        if not is_dup:
            patterns_found.append(c)
            claimed.append((c["first_bottom_idx"], c["second_bottom_idx"]))

    return patterns_found
