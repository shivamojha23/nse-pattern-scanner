import numpy as np
import pandas as pd
import math
import datetime
import os
import sys
from scipy.signal import find_peaks
from scipy.stats import linregress
from .core import *

def detect_cup_and_handle(prices, highs=None, volumes=None,
                          ticker="UNKNOWN", dates=None,
                          interval="1d", verbose=False):
    """
    Scans a 1-D array of closing prices for Cup-and-Handle patterns using
    verified mathematical geometry filters, price smoothing, volume
    confirmation, and all bug-fix rules.

    Smoothing: YES — uses smooth_prices() + find_peaks() for peak detection.
    """
    prices = np.array(prices, dtype=float)

    if highs is not None:
        highs = np.array(highs, dtype=float)
    else:
        highs = prices.copy()

    if volumes is not None:
        volumes = np.array(volumes, dtype=float)

    if len(prices) < 30:
        return []

    # ── Layer 4b Structural Cache Bypass ──
    valid_cups = []
    # ── Step 1: Smooth prices and find peaks/troughs ──
    smoothed = smooth_prices(prices, window=SMOOTHING_WINDOW)

    min_prominence = np.median(smoothed) * 0.01
    peak_indices, _ = find_peaks(smoothed, distance=10, prominence=min_prominence)
    trough_indices, _ = find_peaks(-smoothed, distance=10, prominence=min_prominence)

    if len(peak_indices) < 2 or len(trough_indices) < 1:

        return []

    # ── Step 2: Try every combination (Left Rim, Right Rim) ──
    for i in range(len(peak_indices)):
        for j in range(i + 1, len(peak_indices)):
            left_rim_idx = peak_indices[i]
            right_rim_idx = peak_indices[j]

            left_rim_price = prices[left_rim_idx]
            right_rim_price = prices[right_rim_idx]

            if right_rim_price == 0:
                continue

            is_macro = interval.endswith('d') or interval.endswith('w') or interval.endswith('wk')
            if is_macro:
                window_start = max(0, left_rim_idx - 20)
                window_end = min(len(prices), left_rim_idx + 21)
                if np.max(prices[window_start:window_end]) > left_rim_price:
                    continue
            else:
                if left_rim_idx < 10:
                    continue
                if prices[left_rim_idx - 10] >= left_rim_price:
                    continue
                window_start = max(0, left_rim_idx - 5)
                window_end = min(len(prices), left_rim_idx + 6)
                if np.max(prices[window_start:window_end]) > left_rim_price:
                    continue

            cup_width = right_rim_idx - left_rim_idx
            if cup_width < 15 or cup_width > len(prices) * 0.8:
                continue

            troughs_in_cup = [t for t in trough_indices if left_rim_idx < t < right_rim_idx]
            if not troughs_in_cup:
                continue

            raw_prices_in_cup = prices[left_rim_idx:right_rim_idx]
            cup_bottom_idx = left_rim_idx + int(np.argmin(raw_prices_in_cup))
            cup_bottom_price = prices[cup_bottom_idx]

            cup_left_duration = cup_bottom_idx - left_rim_idx
            cup_right_duration = right_rim_idx - cup_bottom_idx
            if cup_left_duration < MIN_CUP_CANDLES or cup_right_duration < MIN_CUP_CANDLES:
                continue

            prices_after_bottom = prices[cup_bottom_idx:right_rim_idx + 1]
            if len(prices_after_bottom) > 0 and np.min(prices_after_bottom) < cup_bottom_price:
                continue

            left_margin = left_rim_idx + cup_width * 0.15
            right_margin = right_rim_idx - cup_width * 0.15
            if not (left_margin <= cup_bottom_idx <= right_margin):
                continue

            cup_drop_pct = ((left_rim_price - cup_bottom_price) / left_rim_price) * 100
            if not (MIN_CUP_DROP_PCT <= cup_drop_pct <= MAX_CUP_DROP_PCT):
                continue

            recovery_pct = abs(right_rim_price - left_rim_price) / left_rim_price * 100
            if recovery_pct > MAX_RECOVERY_GAP_PCT:
                continue

            valid_cups.append((left_rim_idx, cup_bottom_idx, right_rim_idx))

    candidates = []
    
    # Evaluate handle and breakout for each valid cup structure
    for left_rim_idx, cup_bottom_idx, right_rim_idx in valid_cups:
        left_rim_price = prices[left_rim_idx]
        right_rim_price = prices[right_rim_idx]
        cup_bottom_price = prices[cup_bottom_idx]
        cup_width = right_rim_idx - left_rim_idx
        cup_left_duration = cup_bottom_idx - left_rim_idx
        cup_right_duration = right_rim_idx - cup_bottom_idx
        cup_drop_pct = ((left_rim_price - cup_bottom_price) / left_rim_price) * 100
        recovery_pct = abs(right_rim_price - left_rim_price) / left_rim_price * 100

        # ── Handle detection (Bug 2/3 Fix) ──
        if right_rim_idx + 1 >= len(prices):
            continue

        old_handle_zone_end = min(right_rim_idx + max(1, cup_width // 4), len(prices) - 1)
        old_handle_prices = prices[right_rim_idx : old_handle_zone_end + 1]
        old_handle_low_price = np.min(old_handle_prices) if len(old_handle_prices) > 0 else right_rim_price

        current_handle_low = prices[right_rim_idx]
        current_handle_low_idx = right_rim_idx
        breakout_count = 0
        breakout_confirmed = False
        breakout_start_idx = -1

        for k in range(right_rim_idx + 1,
                       min(len(prices), right_rim_idx + 1 + MAX_HANDLE_LOOKFORWARD_CANDLES)):
            curr_price = prices[k]

            if curr_price < current_handle_low:
                current_handle_low = curr_price
                current_handle_low_idx = k

            if curr_price > right_rim_price:
                breakout_count += 1
            else:
                breakout_count = 0

            if breakout_count >= BREAKOUT_CONFIRM_CANDLES:
                breakout_confirmed = True
                breakout_start_idx = k - BREAKOUT_CONFIRM_CANDLES + 1
                break

        handle_low_price = current_handle_low
        handle_low_idx = current_handle_low_idx

        # ── Pause Before Breakout (Bug 3) ──
        pause_duration = (breakout_start_idx - right_rim_idx) if breakout_confirmed else 0

        handle_slope = 0
        if breakout_confirmed and pause_duration > 1:
            pause_prices = prices[right_rim_idx : breakout_start_idx + 1]
            res = linregress(np.arange(len(pause_prices)), pause_prices)
            handle_slope = res.slope

        # ── Validation ──
        base_zone_max = cup_bottom_price * (1 + BASE_ZONE_PCT)
        cup_candles = prices[left_rim_idx:right_rim_idx]
        base_candles_count = int(np.sum(cup_candles <= base_zone_max))
        roundedness_pct = (base_candles_count / len(cup_candles)) if len(cup_candles) > 0 else 0

        cup_depth = left_rim_price - cup_bottom_price
        max_handle_dip = HANDLE_MAX_RETRACE_RATIO * cup_depth
        handle_pullback = right_rim_price - handle_low_price
        handle_pullback_pct = (handle_pullback / right_rim_price) * 100
        min_handle_pullback = right_rim_price * 0.01
        handle_duration = handle_low_idx - right_rim_idx

        rim_ceiling = highs[left_rim_idx]
        highs_inside_cup = highs[left_rim_idx + 1 : right_rim_idx]
        closes_inside_cup = prices[left_rim_idx + 1 : right_rim_idx]
        max_internal_high = np.max(highs_inside_cup) if len(highs_inside_cup) > 0 else 0
        max_internal_close = np.max(closes_inside_cup) if len(closes_inside_cup) > 0 else 0

        cup_closes_for_diff = prices[left_rim_idx : right_rim_idx + 1]
        if len(cup_closes_for_diff) > 1:
            daily_returns = np.abs(np.diff(cup_closes_for_diff) / cup_closes_for_diff[:-1])
            max_daily_jump = float(np.max(daily_returns))
        else:
            max_daily_jump = 0.0

        # ── Volume Confirmation ──
        vol_decline_pass = None
        vol_decline_avg = 0
        vol_sma50 = 0
        vol_recovery_pass = None
        vol_recovery_ratio = 0
        vol_breakout_pass = None
        vol_breakout_avg = 0

        if volumes is not None and len(volumes) == len(prices):
            vol_series = pd.Series(volumes)
            vol_sma = vol_series.rolling(window=VOLUME_SMA_PERIOD, min_periods=10).mean()

            sma_idx = min(right_rim_idx, len(vol_sma) - 1)
            vol_sma50 = vol_sma.iloc[sma_idx] if not pd.isna(vol_sma.iloc[sma_idx]) else 0

            if cup_bottom_idx > left_rim_idx and vol_sma50 > 0:
                decline_vols = volumes[left_rim_idx:cup_bottom_idx + 1]
                vol_decline_avg = float(np.mean(decline_vols))
                vol_decline_pass = vol_decline_avg < vol_sma50

            if right_rim_idx > cup_bottom_idx + 1:
                recovery_closes = prices[cup_bottom_idx:right_rim_idx + 1]
                recovery_vols = volumes[cup_bottom_idx:right_rim_idx + 1]
                up_vol = 0.0
                down_vol = 0.0
                for rv_i in range(1, len(recovery_closes)):
                    if recovery_closes[rv_i] > recovery_closes[rv_i - 1]:
                        up_vol += recovery_vols[rv_i]
                    else:
                        down_vol += recovery_vols[rv_i]
                vol_recovery_ratio = (up_vol / down_vol) if down_vol > 0 else (2.0 if up_vol > 0 else 0)
                vol_recovery_pass = vol_recovery_ratio > 1.0

            if breakout_confirmed and breakout_start_idx > 0 and vol_sma50 > 0:
                bo_end = min(breakout_start_idx + BREAKOUT_CONFIRM_CANDLES, len(volumes))
                breakout_vols = volumes[breakout_start_idx:bo_end]
                if len(breakout_vols) > 0:
                    vol_breakout_avg = float(np.mean(breakout_vols))
                    vol_breakout_pass = vol_breakout_avg > (BREAKOUT_VOLUME_MULTIPLIER * vol_sma50)

        # ── Quality score ──
        quality_score = compute_quality_score("cup_and_handle", {
            "cup_drop_pct": cup_drop_pct,
            "recovery_vol_ratio": vol_recovery_ratio if vol_recovery_ratio else 0,
            "breakout_vol_ratio": (vol_breakout_avg / vol_sma50) if vol_sma50 > 0 else 0,
        })

        # ── Determine validity ──
        is_valid = True
        reject_reason = ""

        if max_internal_close > left_rim_price:
            is_valid = False
            reject_reason = (f"Internal close (₹{max_internal_close:.2f}) exceeded "
                             f"Left Rim close (₹{left_rim_price:.2f}).")
        elif max_internal_high > rim_ceiling:
            is_valid = False
            reject_reason = (f"Internal high (₹{max_internal_high:.2f}) exceeded "
                             f"Left Rim High (₹{rim_ceiling:.2f}).")
        elif max_daily_jump > MAX_DISCONTINUITY_PCT:
            is_valid = False
            reject_reason = (f"Excessive price discontinuity "
                             f"({max_daily_jump*100:.1f}% > "
                             f"{MAX_DISCONTINUITY_PCT*100}% limit).")
        elif roundedness_pct < MIN_BASE_CANDLES_PCT:
            is_valid = False
            reject_reason = (f"V-shape recovery. Roundedness "
                             f"{base_candles_count}/{len(cup_candles)} "
                             f"({roundedness_pct*100:.1f}%) < "
                             f"{MIN_BASE_CANDLES_PCT*100}% required.")
        elif handle_pullback < min_handle_pullback:
            is_valid = False
            reject_reason = "Handle pullback too small (<1%)."
        elif not breakout_confirmed:
            # ── FORMING STATE (Track A) ──
            # Cup geometry is valid + handle exists, but breakout not yet confirmed.
            # Of the 5 post-breakout checks, only handle_pullback depth is also a
            # structural-validity constraint (values fully available for forming).
            # The other 4 (pause_duration, handle_slope, handle_duration, cup:handle
            # ratio) depend on breakout timing or incomplete handle data — skip those.
            if handle_pullback > max_handle_dip:
                is_valid = False
                reject_reason = (f"Handle pullback (₹{handle_pullback:.2f}) "
                                 f"exceeded 0.32× depth (₹{max_handle_dip:.2f}).")
            else:
                rim_window_start = max(0, right_rim_idx - 2)
                rim_window_end = min(len(prices), right_rim_idx + 3)
                rim_window_prices = prices[rim_window_start:rim_window_end]
                if np.mean(rim_window_prices) < right_rim_price * (1 - RIGHT_RIM_STABILITY_PCT / 100):
                    is_valid = False
                    reject_reason = "Right rim is too spiky (failed stability check)."
            # If both pass, this is a valid forming pattern — is_valid stays True
        elif pause_duration < MIN_PAUSE_CANDLES:
            is_valid = False
            reject_reason = (f"No real handle — immediate continuation "
                             f"(breakout in {pause_duration} candles < "
                             f"{MIN_PAUSE_CANDLES}).")
        elif handle_slope > 0:
            is_valid = False
            reject_reason = (f"Handle is not a genuine pause (slope "
                             f"{handle_slope:.2f} > 0).")
        elif handle_pullback > max_handle_dip:
            is_valid = False
            reject_reason = (f"Handle pullback (₹{handle_pullback:.2f}) "
                             f"exceeded 0.32× depth (₹{max_handle_dip:.2f}).")
        elif handle_duration < MIN_HANDLE_CANDLES:
            is_valid = False
            reject_reason = f"Handle formed too quickly (<{MIN_HANDLE_CANDLES} candles)."
        elif handle_duration > 0 and cup_width < 3 * handle_duration:
            is_valid = False
            reject_reason = "Cup duration must be at least 3x handle duration."
        else:
            rim_window_start = max(0, right_rim_idx - 2)
            rim_window_end = min(len(prices), right_rim_idx + 3)
            rim_window_prices = prices[rim_window_start:rim_window_end]
            if np.mean(rim_window_prices) < right_rim_price * (1 - RIGHT_RIM_STABILITY_PCT / 100):
                is_valid = False
                reject_reason = "Right rim is too spiky (failed stability check)."

        if not is_valid and not verbose:
            continue

        # ── Build pattern dict ──
        pattern = {
            "pattern_type":        "cup_and_handle",
            "ticker":              ticker,
            "left_rim_price":      round(float(left_rim_price), 2),
            "cup_bottom_price":    round(float(cup_bottom_price), 2),
            "right_rim_price":     round(float(right_rim_price), 2),
            "handle_low_price":    round(float(handle_low_price), 2),
            "cup_drop_pct":        round(float(cup_drop_pct), 2),
            "recovery_pct":        round(float(recovery_pct), 2),
            "cup_depth":           round(float(cup_depth), 2),
            "max_handle_dip":      round(float(max_handle_dip), 2),
            "handle_pullback":     round(float(handle_pullback), 2),
            "handle_pullback_pct": round(float(handle_pullback_pct), 2),
            "left_rim_idx":        int(left_rim_idx),
            "cup_bottom_idx":      int(cup_bottom_idx),
            "right_rim_idx":       int(right_rim_idx),
            "handle_low_idx":      int(handle_low_idx),
            "cup_left_duration":   int(cup_left_duration),
            "cup_right_duration":  int(cup_right_duration),
            "handle_duration":     int(handle_duration),
            "double_dip_passed":   True,
            "roundedness_pct":     round(float(roundedness_pct * 100), 1),
            "base_candles":        int(base_candles_count),
            "cup_width":           int(len(cup_candles)),
            "old_handle_low":      round(float(old_handle_low_price), 2),
            "breakout_confirmed":  breakout_confirmed,
            "pause_duration":      int(pause_duration),
            "handle_slope":        round(float(handle_slope), 4),
            "reject_reason":       reject_reason,
            "quality_score":       quality_score if is_valid else 0,
            "vol_decline_avg":     round(float(vol_decline_avg), 0),
            "vol_sma50":           round(float(vol_sma50), 0),
            "vol_decline_pass":    vol_decline_pass,
            "vol_recovery_ratio":  round(float(vol_recovery_ratio), 2),
            "vol_recovery_pass":   vol_recovery_pass,
            "vol_breakout_avg":    round(float(vol_breakout_avg), 0),
            "vol_breakout_pass":   vol_breakout_pass,
            "smoothing_method":    f"SMA({SMOOTHING_WINDOW})",
        }

        if dates is not None:
            pattern["left_rim_date"]   = str(dates[left_rim_idx])
            pattern["cup_bottom_date"] = str(dates[cup_bottom_idx])
            pattern["right_rim_date"]  = str(dates[right_rim_idx])
            pattern["handle_low_date"] = str(dates[handle_low_idx])
            if breakout_confirmed and breakout_start_idx > 0:
                pattern["signal_date"] = str(dates[breakout_start_idx])

        if not is_valid and verbose:
            print_cup_and_handle(pattern, index="REJECTED")
            continue

        candidates.append(pattern)

    # ── Overlap Deduplication ──
    OVERLAP_TOLERANCE = 10
    candidates.sort(key=lambda p: p["cup_drop_pct"], reverse=True)

    patterns_found = []
    claimed_regions = []

    for candidate in candidates:
        cb_idx = candidate["cup_bottom_idx"]
        rr_idx = candidate["right_rim_idx"]

        is_duplicate = False
        for claimed_cb, claimed_rr in claimed_regions:
            if (abs(cb_idx - claimed_cb) <= OVERLAP_TOLERANCE and
                    abs(rr_idx - claimed_rr) <= OVERLAP_TOLERANCE):
                is_duplicate = True
                break

        if not is_duplicate:
            patterns_found.append(candidate)
            claimed_regions.append((cb_idx, rr_idx))

    patterns_found.sort(key=lambda p: p.get("quality_score", 0), reverse=True)
    return patterns_found
