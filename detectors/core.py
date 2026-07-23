import numpy as np
import pandas as pd
import math
import datetime
import os
import sys
from scipy.signal import find_peaks
from scipy.stats import linregress


SMOOTHING_WINDOW = 5
BREAKOUT_VOLUME_MULTIPLIER = 1.2
VOLUME_SMA_PERIOD = 50
BATCH_SIZE = 25
BATCH_SLEEP = 2
SCAN_INTERVAL_MINUTES = 15
IST_OFFSET = datetime.timedelta(hours=5, minutes=30)
MIN_CUP_DROP_PCT = 10
MAX_CUP_DROP_PCT = 45
MAX_RECOVERY_GAP_PCT = 3
HANDLE_MAX_RETRACE_RATIO = 0.50
MIN_CUP_CANDLES = 10
MIN_HANDLE_CANDLES = 5
RIGHT_RIM_STABILITY_PCT = 3.0
BASE_ZONE_PCT = 0.05
MIN_BASE_CANDLES_PCT = 0.08
BREAKOUT_CONFIRM_CANDLES = 3
MAX_HANDLE_LOOKFORWARD_CANDLES = 30
MIN_PAUSE_CANDLES = 5
MAX_DISCONTINUITY_PCT = 0.15
MIN_POLE_RISE_PCT = 8
MIN_POLE_DROP_PCT = 8
MAX_POLE_CANDLES = 20
MIN_POLE_SLOPE_PCT_PER_DAY = 0.6
FLAG_CHANNEL_PCT = 5
MIN_FLAG_CANDLES = 5
MAX_FLAG_CANDLES = 20
MAX_RETRACEMENT_PCT = 50
POLE_R_SQUARED_MIN = 0.8
REQUIRE_BREAKOUT_VOLUME = False
MIN_PENNANT_CANDLES = 5
MAX_PENNANT_CANDLES = 20
PENNANT_ADX_THRESHOLD = 30
PENNANT_POLE_CHANGE_PCT = 10.0
PENNANT_POLE_PERIODS = 20
PENNANT_BREAKOUT_VOL_MULT = 1.5
HEAD_SHOULDER_RATIO = 3
SHOULDER_SYMMETRY_PCT = 8
MIN_HS_CANDLES = 20
NECKLINE_SLOPE_WARN_PCT = 5
TROUGH_MAX_DEPTH_PCT = 0.22
TROUGH_MAX_DIFF_PCT = 8.0
MIN_LS_TO_HEAD_CANDLES = 7
MIN_HEAD_TO_RS_CANDLES = 7
MAX_LS_TO_HEAD_CANDLES = 45
MAX_HEAD_TO_RS_CANDLES = 45
ATR_HS_MULTIPLIER = 1.5
DOUBLE_TOP_SIMILARITY_PCT = 5
DOUBLE_BOTTOM_SIMILARITY_PCT = 5
MIN_VALLEY_DROP_PCT = 5
MIN_PEAK_RISE_PCT = 5
MIN_DOUBLE_CANDLES = 10
MAX_DOUBLE_CANDLES = 60
BREAKDOWN_CONFIRM_PCT = 0.005
BREAKOUT_CONFIRM_PCT_DB = 0.005
BREAKDOWN_CONFIRM_CANDLES_DT = 2
BREAKOUT_CONFIRM_CANDLES_DB = 2
MAX_BREAKOUT_WAIT_CANDLES = 10
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 1.5
ALL_PATTERNS = [
    "cup_and_handle",
    "bull_flag",
    "bear_flag",
    "pennant",
    "head_and_shoulders",
    "double_top",
    "double_bottom",
]
PATTERN_NAMES = {
    "cup_and_handle":     "Cup and Handle",
    "bull_flag":          "Bull Flag",
    "bear_flag":          "Bear Flag",
    "pennant":            "Pennant",
    "head_and_shoulders": "Head and Shoulders",
    "double_top":         "Double Top",
    "double_bottom":      "Double Bottom",
}
PATTERN_SIGNALS = {
    "cup_and_handle":     "Bullish Continuation",
    "bull_flag":          "Bullish Continuation",
    "bear_flag":          "Bearish Continuation",
    "pennant":            "Continuation (pole direction)",
    "head_and_shoulders": "Bearish Reversal",
    "double_top":         "Bearish Reversal",
    "double_bottom":      "Bullish Reversal",
}

def smooth_prices(prices, window=SMOOTHING_WINDOW):
    """
    Applies a Simple Moving Average (SMA) to the price series for noise
    reduction before peak/trough detection via find_peaks().

    ONLY used by: Cup & Handle, Head & Shoulders, Double Top, Double Bottom.
    NOT used by: Bull Flag, Bear Flag, Pennant (these use linregress on raw data).

    The smoothed series is ONLY used for find_peaks() to identify candidate
    dates. All actual price values come from RAW (unsmoothed) prices.
    """
    series = pd.Series(prices)
    smoothed = series.rolling(window=window, min_periods=1, center=True).mean()
    return smoothed.values

def compute_quality_score(pattern_type, metrics):
    """
    Computes a quality score for any pattern type.

    Parameters
    ----------
    pattern_type : str
        One of ALL_PATTERNS (e.g., "cup_and_handle", "bull_flag", etc.)
    metrics : dict
        Pattern-specific values used for scoring.

    Returns
    -------
    float
        Quality score.  Cup & Handle uses the original v2 formula (range ~5–80).
        All other patterns use a 0–10 scale.
    """
    if pattern_type == "cup_and_handle":
        # ── Granular 100-Point Quality Matrix (Normalized to 0.0 - 10.0) ──
        raw_score = 0.0

        # 1. Handle Quality (Max 30 pts)
        retrace = metrics.get("handle_retrace_ratio", 0.5)
        if retrace <= 0.25:
            raw_score += 20.0
        elif retrace <= 0.35:
            raw_score += 15.0
        elif retrace <= 0.45:
            raw_score += 10.0
        else:
            raw_score += 4.0

        slope = metrics.get("handle_slope", 0)
        if slope <= 0:
            raw_score += 10.0
        elif slope <= 1.0:
            raw_score += 5.0

        # 2. Cup Shape & Geometry (Max 30 pts)
        drop = metrics.get("cup_drop_pct", 0)
        if 15.0 <= drop <= 30.0:
            raw_score += 10.0
        elif 10.0 <= drop <= 45.0:
            raw_score += 6.0

        roundedness = metrics.get("roundedness_pct", 0)
        if roundedness >= 0.20:
            raw_score += 10.0
        elif roundedness >= 0.12:
            raw_score += 7.0
        elif roundedness >= 0.08:
            raw_score += 4.0

        rec_gap = metrics.get("recovery_pct", 3.0)
        if rec_gap <= 1.5:
            raw_score += 10.0
        elif rec_gap <= 3.0:
            raw_score += 5.0

        # 3. Volume Confirmation (Max 25 pts)
        if metrics.get("vol_decline_pass"):
            raw_score += 7.0

        rv = max(metrics.get("recovery_vol_ratio", 0), 0)
        if rv >= 1.5:
            raw_score += 10.0
        elif rv >= 1.0:
            raw_score += 6.0

        bv = max(metrics.get("breakout_vol_ratio", 0), 0)
        if bv >= 1.5:
            raw_score += 8.0
        elif bv >= 1.0:
            raw_score += 4.0

        # 4. Trend Context (Max 15 pts)
        if metrics.get("has_uptrend"):
            raw_score += 15.0
        else:
            raw_score += 5.0

        # Normalize 0-100 pts down to 0.0-10.0 scale
        normalized_score = round(min(100.0, max(0.0, raw_score)) / 10.0, 1)
        return normalized_score

    elif pattern_type in ("bull_flag", "bear_flag"):
        # Pole strength (0–3): 8% rise → 0, 20% → 3
        pole_pct = abs(metrics.get("pole_change_pct", 0))
        pole_score = min(3.0, max(0, (pole_pct - 8) / 4))
        # R² quality (0–2): 0.8 → 0, 1.0 → 2
        r2_score = min(2.0, max(0, (metrics.get("r_squared", 0.8) - 0.8) * 10))
        # Flag tightness (0–2): 5% range → 0, 0% → 2
        flag_range = metrics.get("flag_range_pct", 5)
        flag_score = max(0, 2.0 * (1 - flag_range / 5.0))
        # Volume checks (0–3): 1 point per passing check
        vol_score = 0
        if metrics.get("vol_pole_pass"):
            vol_score += 1
        if metrics.get("vol_flag_pass"):
            vol_score += 1
        if metrics.get("vol_breakout_pass"):
            vol_score += 1
        return round(min(10.0, max(0, pole_score + r2_score + flag_score + vol_score)), 1)

    elif pattern_type == "pennant":
        # Pole strength (0–3)
        pole_pct = abs(metrics.get("pole_change_pct", 0))
        pole_score = min(3.0, max(0, (pole_pct - 8) / 4))
        # Symmetry quality (0–3): 0% diff → 3, 40% diff → 0
        sym_diff = metrics.get("symmetry_diff", 0.40)
        sym_score = max(0, 3.0 * (1 - sym_diff / 0.40))
        # Volume breakdown (0-4)
        vol_score = 0
        if metrics.get("vol_pole_pass"): vol_score += 1
        if metrics.get("vol_pennant_pass"): vol_score += 1
        if metrics.get("vol_slope", 0) < 0: vol_score += 1
        if metrics.get("vol_breakout_pass"): vol_score += 1
        return round(min(10.0, max(0, pole_score + sym_score + vol_score)), 1)

    elif pattern_type == "head_and_shoulders":
        # Head prominence (0–3): 3% → 1, 9% → 3
        head_prom = metrics.get("head_prominence_pct", 0)
        head_score = min(3.0, max(0, head_prom / 3))
        # Shoulder symmetry (0–2): 0% diff → 2, 8% → 0
        sym = metrics.get("shoulder_symmetry_pct", 8)
        sym_score = max(0, 2.0 * (1 - sym / 8))
        # Neckline flatness (0–2): 0% slope → 2, 8% → 0
        neck_slope = abs(metrics.get("neckline_slope_pct", 0))
        neck_score = max(0, 2.0 * (1 - neck_slope / 8))
        
        # Downward slope bonus: +1 point if right trough is lower than left trough
        if metrics.get("right_trough_lower"):
            neck_score += 1.0
        # Volume (0–3)
        vol_score = 0
        if metrics.get("vol_progression_pass"):
            vol_score += 1.5
        if metrics.get("vol_breakdown_pass"):
            vol_score += 1.5
        return round(min(10.0, max(0, head_score + sym_score + neck_score + vol_score)), 1)

    elif pattern_type in ("double_top", "double_bottom"):
        # Similarity (0–3): 0% → 3, 3% → 0
        sim = metrics.get("similarity_pct", 3)
        sim_score = max(0, 3.0 * (1 - sim / 3))
        # Depth/height (0–3): 3% → 1, 9% → 3
        depth = metrics.get("depth_pct", 3)
        depth_score = min(3.0, max(0, depth / 3))
        # Volume (0–4)
        vol_score = 0
        if metrics.get("vol_pattern_pass"):
            vol_score += 2
        if metrics.get("vol_breakout_pass"):
            vol_score += 2
        return round(min(10.0, max(0, sim_score + depth_score + vol_score)), 1)

    return 0.0

def refine_peak(idx, prices, window=4):
    start = max(0, idx - window)
    end = min(len(prices), idx + window + 1)
    local_idx = int(np.argmax(prices[start:end]))
    return start + local_idx

def refine_trough(idx, prices, window=4):
    start = max(0, idx - window)
    end = min(len(prices), idx + window + 1)
    local_idx = int(np.argmin(prices[start:end]))
    return start + local_idx

def compute_atr(highs, lows, closes, period=ATR_PERIOD):
    """
    Computes the Average True Range (ATR) over the given period.

    True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    ATR = Simple Moving Average of True Range over `period` bars.

    Uses only numpy/pandas — no TA-Lib needed.

    Parameters
    ----------
    highs, lows, closes : array-like
        High, Low, Close price arrays (same length).
    period : int
        ATR averaging period (default 14).

    Returns
    -------
    float
        The most recent ATR value, or 0.0 if insufficient data.
    """
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)
    if len(highs) < period + 1:
        return 0.0
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    if len(tr) < period:
        return float(np.mean(tr)) if len(tr) > 0 else 0.0
    return float(np.mean(tr[-period:]))

def compute_adx(highs, lows, closes, period=14):
    """
    Computes the Average Directional Index (ADX) over the given period.
    Uses Wilder's Smoothing.
    
    Parameters
    ----------
    highs, lows, closes : array-like
        High, Low, Close price arrays (same length).
    period : int
        ADX averaging period (default 14).
        
    Returns
    -------
    adx : np.ndarray
        Array of ADX values corresponding to the inputs. Values are padded
        with np.nan at the beginning where ADX cannot be computed.
    """
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)
    n = len(closes)
    adx = np.full(n, np.nan)
    if n < period * 2:
        return adx
    tr1 = highs[1:] - lows[1:]
    tr2 = np.abs(highs[1:] - closes[:-1])
    tr3 = np.abs(lows[1:] - closes[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.insert(tr, 0, np.nan)
    up_move = highs[1:] - highs[:-1]
    down_move = lows[:-1] - lows[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.insert(plus_dm, 0, np.nan)
    minus_dm = np.insert(minus_dm, 0, np.nan)

    def wilders_smoothing(data, period):
        smoothed = np.full_like(data, np.nan)
        first_valid = period
        if len(data) <= first_valid:
            return smoothed
        smoothed[first_valid] = np.sum(data[1:first_valid + 1])
        for i in range(first_valid + 1, len(data)):
            smoothed[i] = smoothed[i - 1] - smoothed[i - 1] / period + data[i]
        return smoothed
    atr = wilders_smoothing(tr, period)
    plus_di_smooth = wilders_smoothing(plus_dm, period)
    minus_di_smooth = wilders_smoothing(minus_dm, period)
    plus_di = np.full_like(atr, np.nan)
    minus_di = np.full_like(atr, np.nan)
    valid_atr = atr > 0
    plus_di[valid_atr] = 100 * (plus_di_smooth[valid_atr] / atr[valid_atr])
    minus_di[valid_atr] = 100 * (minus_di_smooth[valid_atr] / atr[valid_atr])
    dx = np.full_like(atr, np.nan)
    di_sum = plus_di + minus_di
    valid_di = di_sum > 0
    dx[valid_di] = 100 * np.abs(plus_di[valid_di] - minus_di[valid_di]) / di_sum[valid_di]
    first_dx = -1
    for i in range(n):
        if not np.isnan(dx[i]):
            first_dx = i
            break
    if first_dx == -1 or first_dx + period > n:
        return adx
    adx[first_dx + period - 1] = np.mean(dx[first_dx:first_dx + period])
    for i in range(first_dx + period, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx
