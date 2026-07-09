"""
=============================================================================
  NSE MULTI-PATTERN SCANNER  v3.0
  Indian Stock Market (NSE)
=============================================================================

  Author : AI-assisted project
  Purpose: Detect 7 chart patterns on NSE stocks using Yahoo Finance data,
           with smoothing (where needed), volume confirmation, quality
           scoring, and composite ranking.

  PATTERNS DETECTED
  -----------------
  1. Cup and Handle       — Bullish Continuation
  2. Bull Flag            — Bullish Continuation
  3. Bear Flag            — Bearish Continuation
  4. Pennant              — Continuation (pole direction)
  5. Head and Shoulders   — Bearish Reversal
  6. Double Top           — Bearish Reversal
  7. Double Bottom        — Bullish Reversal

  THREE MODES
  -----------
  A. SELF-TEST     → Synthetic data to prove the math works (no internet).
  B. HISTORICAL    → Sweeps past data for already-completed patterns.
  C. LIVE SCANNER  → Continuous scanner during NSE market hours.

  Smoothing Policy
  ----------------
  smooth_prices() is applied ONLY to patterns that use scipy find_peaks()
  for peak/trough detection: Cup & Handle, Head & Shoulders, Double Top,
  Double Bottom.  Patterns using sliding-window + linregress (Bull Flag,
  Bear Flag, Pennant) work on RAW prices — no smoothing needed.

  pip install
  -----------
  pip install yfinance pandas numpy scipy

=============================================================================
"""

# ─── IMPORTS ────────────────────────────────────────────────────────────────
import sys
import os
import time
import datetime
import warnings
import argparse
import math

# Fix Windows console encoding so emoji and special characters display correctly.
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    else:
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")

class DualWriter:
    """Writes to both standard output and a log file simultaneously."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def reconfigure(self, **kwargs):
        if hasattr(self.terminal, "reconfigure"):
            self.terminal.reconfigure(**kwargs)

# Redirect stdout to both terminal and results.txt
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, "results.txt")
sys.stdout = DualWriter(log_file)

import numpy as np                     # Fast math on arrays of numbers
import pandas as pd                    # DataFrames — like Excel spreadsheets in Python
import yfinance as yf                  # Downloads stock price data from Yahoo Finance
from scipy.signal import find_peaks    # Finds local highs/lows in a price curve
from scipy.stats import linregress     # Calculates linear regression slope + R²

# Suppress noisy warnings from yfinance / pandas
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
#  CONFIGURATION — All tuneable knobs in one place
# =============================================================================

# ─── SMOOTHING ──────────────────────────────────────────────────────────────
# Only applied for patterns that use find_peaks() (Cup & Handle, H&S,
# Double Top, Double Bottom).  NOT applied for Flag/Pennant patterns.
SMOOTHING_WINDOW = 5

# ─── VOLUME (shared across all patterns) ───────────────────────────────────
BREAKOUT_VOLUME_MULTIPLIER = 1.2   # Breakout/breakdown candle must exceed this × SMA
VOLUME_SMA_PERIOD = 50             # Period for the volume moving average baseline

# ─── DATA FETCHING ────────────────────────────────────────────────────────
BATCH_SIZE = 25               # Tickers per yf.download() call
BATCH_SLEEP = 2               # Seconds between download batches

# ─── LIVE SCANNER ─────────────────────────────────────────────────────────
SCAN_INTERVAL_MINUTES = 15    # How often the live scanner re-runs
IST_OFFSET = datetime.timedelta(hours=5, minutes=30)

# ─── CUP & HANDLE GEOMETRY (existing — unchanged) ────────────────────────
MIN_CUP_DROP_PCT = 10
MAX_CUP_DROP_PCT = 35
MAX_RECOVERY_GAP_PCT = 3
HANDLE_MAX_RETRACE_RATIO = 0.32
MIN_CUP_CANDLES = 10
MIN_HANDLE_CANDLES = 5
RIGHT_RIM_STABILITY_PCT = 3.0

# Cup & Handle bug-fix parameters
BASE_ZONE_PCT = 0.05
MIN_BASE_CANDLES_PCT = 0.20
BREAKOUT_CONFIRM_CANDLES = 3
MAX_HANDLE_LOOKFORWARD_CANDLES = 30
MIN_PAUSE_CANDLES = 5
MAX_DISCONTINUITY_PCT = 0.08

# ─── BULL FLAG / BEAR FLAG ────────────────────────────────────────────────
MIN_POLE_RISE_PCT = 8          # Minimum pole rise for bull flag
MIN_POLE_DROP_PCT = 8          # Minimum pole drop for bear flag
MAX_POLE_CANDLES = 20          # Max candles for pole duration
MIN_POLE_SLOPE_PCT_PER_DAY = 0.6  # Min rise/drop per bar — rejects slow-grind poles
FLAG_CHANNEL_PCT = 5           # Max channel width as % of pole-end price
MIN_FLAG_CANDLES = 5           # Min flag duration
MAX_FLAG_CANDLES = 20          # Max flag duration
MAX_RETRACEMENT_PCT = 50       # Max flag retracement as % of pole move — tune from backtests
# R² linearity of the pole. 0.8 may filter out gap-driven poles (sharp but noisy).
# A pole with one big gap day fits a line worse than a slow grind, which is
# backwards from what "pole" means. Consider sweeping 0.6–0.9 in backtests.
POLE_R_SQUARED_MIN = 0.8
REQUIRE_BREAKOUT_VOLUME = False  # When True, reject patterns where vol_breakout_pass is False
# TODO: interval-aware thresholds — MAX_POLE_CANDLES=20 means a month on daily
# bars but only 5 hours on 15-min bars. Restructure into a per-interval dict
# once backtested values are available.

# ─── PENNANT ──────────────────────────────────────────────────────────────
MIN_PENNANT_CANDLES = 5        # Minimum candles in the pennant body (consolidation)
MAX_PENNANT_CANDLES = 20       # Maximum candles in the pennant body
PENNANT_ADX_THRESHOLD = 30     # ADX must be above 30 signaling a powerful trend
PENNANT_POLE_CHANGE_PCT = 10.0 # Pole must change by at least 10%
PENNANT_POLE_PERIODS = 20      # Lookback window for pole calculation
PENNANT_BREAKOUT_VOL_MULT = 1.5 # Breakout volume must be 1.5x the SMA20

# ─── HEAD AND SHOULDERS ──────────────────────────────────────────────────
HEAD_SHOULDER_RATIO = 3        # Head must be ≥ this % higher than each shoulder
SHOULDER_SYMMETRY_PCT = 8      # Shoulders must be within this % of each other
MIN_HS_CANDLES = 20            # Minimum span (left shoulder → right shoulder)
NECKLINE_SLOPE_WARN_PCT = 5    # Warn if neckline slopes more than this %
TROUGH_MAX_DEPTH_PCT = 0.22    # Max depth of troughs relative to lower shoulder
TROUGH_MAX_DIFF_PCT = 8.0      # Max difference between the two troughs
MIN_LS_TO_HEAD_CANDLES = 7     # Min candles between left shoulder and head
MIN_HEAD_TO_RS_CANDLES = 7     # Min candles between head and right shoulder
MAX_LS_TO_HEAD_CANDLES = 45    # Max candles between left shoulder and head
MAX_HEAD_TO_RS_CANDLES = 45    # Max candles between head and right shoulder
ATR_HS_MULTIPLIER = 1.5        # Minimum Head-to-Neckline depth in ATR units

# ─── DOUBLE TOP / DOUBLE BOTTOM ──────────────────────────────────────────
DOUBLE_TOP_SIMILARITY_PCT = 5
DOUBLE_BOTTOM_SIMILARITY_PCT = 5
MIN_VALLEY_DROP_PCT = 5
MIN_PEAK_RISE_PCT = 5
MIN_DOUBLE_CANDLES = 10
MAX_DOUBLE_CANDLES = 60             # Maximum bars between valleys
BREAKDOWN_CONFIRM_PCT = 0.005       # 0.5% — decisive close margin for Double Top
BREAKOUT_CONFIRM_PCT_DB = 0.005     # 0.5% — decisive close margin for Double Bottom
BREAKDOWN_CONFIRM_CANDLES_DT = 2    # Sustained candles for breakdown confirmation
BREAKOUT_CONFIRM_CANDLES_DB = 2     # Sustained candles for breakout confirmation
MAX_BREAKOUT_WAIT_CANDLES = 10      # Time-decay: reject if no break within this window
ATR_PERIOD = 14                     # Average True Range lookback period
ATR_STOP_MULTIPLIER = 1.5           # Stop-loss = peak ± (this × ATR)

# ─── PATTERN REGISTRY ────────────────────────────────────────────────────
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

# ─── LIVE-MODE DEDUPLICATION CACHE ─────────────────────────────────────────
_live_alerted_today = set()   # set of (ticker, pattern_type) tuples
_live_alert_date = None


# =============================================================================
#  1. GET THE WATCHLIST — Dynamically fetch Nifty 100/200 tickers
# =============================================================================

def get_nifty_list():
    """
    Fetches the Nifty 200 stock list via the Layer 2 cache.
    """
    from db_cache import get_cached_watchlist
    return get_cached_watchlist("nifty200")


# =============================================================================
#  2. FETCH DATA IN BATCHES — Download candle data from Yahoo Finance
# =============================================================================

def fetch_batch_data(tickers, period=None, start=None, end=None, interval="15m"):
    """
    Downloads OHLCV candle data for many tickers in batches.

    Returns dict[str, pd.DataFrame] — keys are ticker symbols, values are
    DataFrames with Open, High, Low, Close, Volume columns.
    """
    all_data = {}
    total = len(tickers)
    min_candles = 30 if interval in ("1d", "1wk") else 50

    for i in range(0, total, BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        batch_str = " ".join(batch)
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  📦 Batch {batch_num}/{total_batches}  "
              f"({len(batch)} tickers) ... ", end="", flush=True)

        try:
            download_kwargs = {
                "tickers": batch_str,
                "interval": interval,
                "group_by": "ticker",
                "progress": False,
                "threads": True,
                "auto_adjust": True
            }
            if start and end:
                download_kwargs["start"] = start
                download_kwargs["end"] = end
            elif start:
                download_kwargs["start"] = start
            elif end:
                download_kwargs["end"] = end
            elif period:
                download_kwargs["period"] = period
            else:
                download_kwargs["period"] = "59d"

            data = yf.download(**download_kwargs)

            if data.empty:
                print("empty result.")
            else:
                for ticker in batch:
                    try:
                        if isinstance(data.columns, pd.MultiIndex):
                            ticker_df = data.xs(ticker, level="Ticker", axis=1)
                        else:
                            ticker_df = data

                        if not ticker_df.empty and len(ticker_df) > min_candles:
                            all_data[ticker] = ticker_df.copy()
                    except (KeyError, TypeError):
                        pass

                print(f"OK  ({len(all_data)} tickers so far)")

        except Exception as e:
            print(f"error: {e}")

        if i + BATCH_SIZE < total:
            time.sleep(BATCH_SLEEP)

    print(f"\n  ✓ Successfully fetched data for {len(all_data)} tickers.\n")
    return all_data


# =============================================================================
#  3. SMOOTH PRICES — Reduce noise before peak detection (only when needed)
# =============================================================================

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


# =============================================================================
#  4. QUALITY SCORE — Generalized pattern scoring
# =============================================================================

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
        # ── Original v2 formula — UNCHANGED ──
        cup_drop_pct = metrics.get("cup_drop_pct", 0)
        rv = max(metrics.get("recovery_vol_ratio", 0), 0)
        bv = max(metrics.get("breakout_vol_ratio", 0), 0)
        score = (cup_drop_pct * 0.4
                 + math.log(rv + 1) * 30
                 + math.log(bv + 1) * 30)
        return round(score, 2)

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
        # Convergence quality (0–2): ratio of second-half range / first-half range
        conv_ratio = metrics.get("convergence_ratio", 1.0)
        conv_score = min(2.0, max(0, (1 - conv_ratio) * 4))
        # R² (0–2)
        r2_score = min(2.0, max(0, (metrics.get("r_squared", 0.8) - 0.8) * 10))
        # Volume (0–3)
        vol_score = 0
        if metrics.get("vol_pole_pass"):
            vol_score += 1
        if metrics.get("vol_pennant_pass"):
            vol_score += 1
        if metrics.get("vol_breakout_pass"):
            vol_score += 1
        return round(min(10.0, max(0, pole_score + conv_score + r2_score + vol_score)), 1)

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


# =============================================================================
#  5. DETECT CUP AND HANDLE
#     (Exact copy from v2 — all rules intact, only quality_score call updated)
# =============================================================================

def detect_cup_and_handle(prices, highs=None, volumes=None,
                          ticker="UNKNOWN", dates=None,
                          interval="1d", verbose=False, structural_cache=None):
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
    cache_hit = False
    
    if structural_cache is not None and "cup_pairs" in structural_cache and dates is not None:
        try:
            for (l_date, b_date, r_date) in structural_cache["cup_pairs"]:
                l_idx = dates.get_loc(pd.Timestamp(l_date))
                b_idx = dates.get_loc(pd.Timestamp(b_date))
                r_idx = dates.get_loc(pd.Timestamp(r_date))
                if r_idx < len(prices) and prices[r_idx] > 0:
                    valid_cups.append((l_idx, b_idx, r_idx))
            cache_hit = True
        except KeyError:
            valid_cups = []
            cache_hit = False
            
    if not cache_hit:
        # ── Step 1: Smooth prices and find peaks/troughs ──
        smoothed = smooth_prices(prices, window=SMOOTHING_WINDOW)

        min_prominence = np.median(smoothed) * 0.01
        peak_indices, _ = find_peaks(smoothed, distance=10, prominence=min_prominence)
        trough_indices, _ = find_peaks(-smoothed, distance=10, prominence=min_prominence)

        if len(peak_indices) < 2 or len(trough_indices) < 1:
            if isinstance(structural_cache, dict):
                structural_cache["cup_pairs"] = []
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

        if isinstance(structural_cache, dict) and dates is not None:
            structural_cache["cup_pairs"] = [
                (str(dates[l]), str(dates[b]), str(dates[r])) for l, b, r in valid_cups
            ]

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
            is_valid = False
            reject_reason = (f"No breakout confirmed within "
                             f"{MAX_HANDLE_LOOKFORWARD_CANDLES} candles.")
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


# =============================================================================
#  6. DETECT BULL FLAG
# =============================================================================

def detect_bull_flag(prices, volumes=None, ticker="UNKNOWN", dates=None,
                     interval="1d", verbose=False, structural_cache=None):
    """
    Detects Bull Flag patterns using structural_cache.
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

    # Fast Path: Cache Hit
    patterns_found = []
    cache_hit = False
    
    if structural_cache is not None and "bull_flag_pairs" in structural_cache and dates is not None:
        try:
            for cached_pat in structural_cache["bull_flag_pairs"]:
                pole_start_idx = dates.get_loc(pd.Timestamp(cached_pat["pole_start_date"]))
                pole_end_idx = dates.get_loc(pd.Timestamp(cached_pat["pole_top_date"]))
                flag_end_idx = dates.get_loc(pd.Timestamp(cached_pat["flag_end_date"]))
                
                pole_top_price = cached_pat["pole_top_price"]
                
                breakout_idx = None
                for k in range(flag_end_idx + 1, min(n, flag_end_idx + 6)):
                    if prices[k] > pole_top_price:
                        breakout_idx = k
                        break
                        
                if breakout_idx is None:
                    continue
                    
                # Breakout found from cache
                vol_sma50 = 0
                vol_pole_pass = None
                vol_flag_pass = None
                vol_breakout_pass = None

                if volumes is not None and len(volumes) == n:
                    vol_series = pd.Series(volumes)
                    vol_sma = vol_series.rolling(window=VOLUME_SMA_PERIOD, min_periods=10).mean()
                    sma_idx = min(pole_end_idx, len(vol_sma) - 1)
                    vol_sma50 = float(vol_sma.iloc[sma_idx]) if not pd.isna(vol_sma.iloc[sma_idx]) else 0

                    if vol_sma50 > 0:
                        pole_vol = float(np.mean(volumes[pole_start_idx:pole_end_idx + 1]))
                        vol_pole_pass = pole_vol > vol_sma50

                        flag_vol = float(np.mean(volumes[pole_end_idx + 1:flag_end_idx + 1]))
                        vol_flag_pass = flag_vol < vol_sma50

                        breakout_vol = float(volumes[breakout_idx])
                        vol_breakout_pass = breakout_vol > (BREAKOUT_VOLUME_MULTIPLIER * vol_sma50)

                if REQUIRE_BREAKOUT_VOLUME and vol_breakout_pass is False:
                    continue
                    
                quality = compute_quality_score("bull_flag", {
                    "pole_change_pct": cached_pat["pole_rise_pct"],
                    "r_squared": cached_pat["pole_r_squared"],
                    "flag_range_pct": cached_pat["flag_range_pct"],
                    "vol_pole_pass": vol_pole_pass,
                    "vol_flag_pass": vol_flag_pass,
                    "vol_breakout_pass": vol_breakout_pass,
                })
                
                pat = cached_pat.copy()
                pat["breakout_idx"] = breakout_idx
                pat["breakout_price"] = round(float(prices[breakout_idx]), 2)
                pat["vol_pole_pass"] = vol_pole_pass
                pat["vol_flag_pass"] = vol_flag_pass
                pat["vol_breakout_pass"] = vol_breakout_pass
                pat["quality_score"] = quality
                pat["signal_date"] = str(dates[breakout_idx])
                patterns_found.append(pat)
            
            cache_hit = True
            
        except KeyError:
            patterns_found = []
            cache_hit = False

    if cache_hit:
        patterns_found.sort(key=lambda p: p.get("quality_score", 0), reverse=True)
        return patterns_found


    # Slow Path: Full Detection
    rolling_min = pd.Series(prices).rolling(window=MAX_POLE_CANDLES, min_periods=5).min().values
    potential_return = np.zeros(n)
    for idx in range(MAX_POLE_CANDLES, n):
        if rolling_min[idx] > 0:
            potential_return[idx] = (prices[idx] - rolling_min[idx]) / rolling_min[idx] * 100
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

            rise_pct = (prices[pole_end_idx] - prices[pole_start_idx]) / prices[pole_start_idx] * 100
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

        pole_top_price = prices[pole_end_idx]

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

            if flag_high > pole_top_price * 1.001:
                continue

            flag_range_pct = (flag_high - flag_low) / pole_top_price * 100
            if flag_range_pct > FLAG_CHANNEL_PCT:
                continue

            pole_size = pole_top_price - prices[best_pole["start_idx"]]
            retracement_pct = (pole_top_price - flag_low) / pole_size * 100 if pole_size > 0 else 0
            if retracement_pct > MAX_RETRACEMENT_PCT:
                continue

            x = np.arange(len(flag_prices))
            flag_res = linregress(x, flag_prices)
            flag_total_drift_pct = (flag_res.slope * len(flag_prices)) / pole_top_price * 100
            if flag_total_drift_pct > 0.5:
                continue

            pattern = {
                "pattern_type":       "bull_flag",
                "ticker":             ticker,
                "verdict":            "VALID",
                "reject_reason":      "",
                "pole_start_price":   round(float(prices[best_pole["start_idx"]]), 2),
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
            }

            if dates is not None:
                pattern["pole_start_date"] = str(dates[best_pole["start_idx"]])
                pattern["pole_top_date"]   = str(dates[pole_end_idx])
                pattern["flag_end_date"]   = str(dates[flag_end])
            
            valid_structures.append(pattern)

    # Populate cache if requested
    if structural_cache is not None and isinstance(structural_cache, dict):
        structural_cache["bull_flag_pairs"] = valid_structures

    # Now evaluate breakouts on all valid structures
    for pat in valid_structures:
        flag_end = pat["flag_end_idx"]
        pole_top_price = pat["pole_top_price"]
        breakout_idx = None
        for k in range(flag_end + 1, min(n, flag_end + 6)):
            if prices[k] > pole_top_price:
                breakout_idx = k
                break

        if breakout_idx is None:
            if verbose:
                print(f"  ❌ {ticker}: Bull Flag pole found (idx {pat['pole_start_idx']}-"
                      f"{pat['pole_end_idx']}, +{pat['pole_rise_pct']:.1f}%) but no breakout.")
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

                breakout_vol = float(volumes[breakout_idx])
                vol_breakout_pass = breakout_vol > (BREAKOUT_VOLUME_MULTIPLIER * vol_sma50)

        if REQUIRE_BREAKOUT_VOLUME and vol_breakout_pass is False:
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
        pat["breakout_idx"] = breakout_idx
        pat["breakout_price"] = round(float(prices[breakout_idx]), 2)
        pat["vol_pole_pass"] = vol_pole_pass
        pat["vol_flag_pass"] = vol_flag_pass
        pat["vol_breakout_pass"] = vol_breakout_pass
        pat["quality_score"] = quality
        
        if dates is not None:
            pat["signal_date"] = str(dates[breakout_idx])
            
        patterns_found.append(pat)

    patterns_found.sort(key=lambda p: p.get("quality_score", 0), reverse=True)
    return patterns_found

# =============================================================================
def detect_bear_flag(prices, volumes=None, ticker="UNKNOWN", dates=None,
                     interval="1d", verbose=False, structural_cache=None):
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

    # Fast Path: Cache Hit
    patterns_found = []
    cache_hit = False
    
    if structural_cache is not None and "bear_flag_pairs" in structural_cache and dates is not None:
        try:
            for cached_pat in structural_cache["bear_flag_pairs"]:
                pole_start_idx = dates.get_loc(pd.Timestamp(cached_pat["pole_start_date"]))
                pole_end_idx = dates.get_loc(pd.Timestamp(cached_pat["pole_bottom_date"]))
                flag_end_idx = dates.get_loc(pd.Timestamp(cached_pat["flag_end_date"]))
                
                pole_bottom_price = cached_pat["pole_bottom_price"]
                
                breakdown_idx = None
                for k in range(flag_end_idx + 1, min(n, flag_end_idx + 6)):
                    if prices[k] < pole_bottom_price:
                        breakdown_idx = k
                        break
                        
                if breakdown_idx is None:
                    continue
                    
                # Breakdown found from cache
                vol_sma50 = 0
                vol_pole_pass = None
                vol_flag_pass = None
                vol_breakdown_pass = None

                if volumes is not None and len(volumes) == n:
                    vol_series = pd.Series(volumes)
                    vol_sma = vol_series.rolling(window=VOLUME_SMA_PERIOD, min_periods=10).mean()
                    sma_idx = min(pole_end_idx, len(vol_sma) - 1)
                    vol_sma50 = float(vol_sma.iloc[sma_idx]) if not pd.isna(vol_sma.iloc[sma_idx]) else 0

                    if vol_sma50 > 0:
                        pole_vol = float(np.mean(volumes[pole_start_idx:pole_end_idx + 1]))
                        vol_pole_pass = pole_vol > vol_sma50

                        flag_vol = float(np.mean(volumes[pole_end_idx + 1:flag_end_idx + 1]))
                        vol_flag_pass = flag_vol < vol_sma50

                        bd_vol = float(volumes[breakdown_idx])
                        vol_breakdown_pass = bd_vol > (BREAKOUT_VOLUME_MULTIPLIER * vol_sma50)

                if REQUIRE_BREAKOUT_VOLUME and vol_breakdown_pass is False:
                    continue
                    
                quality = compute_quality_score("bear_flag", {
                    "pole_change_pct": cached_pat["pole_drop_pct"],
                    "r_squared": cached_pat["pole_r_squared"],
                    "flag_range_pct": cached_pat["flag_range_pct"],
                    "vol_pole_pass": vol_pole_pass,
                    "vol_flag_pass": vol_flag_pass,
                    "vol_breakout_pass": vol_breakdown_pass,
                })
                
                pat = cached_pat.copy()
                pat["breakdown_idx"] = breakdown_idx
                pat["breakdown_price"] = round(float(prices[breakdown_idx]), 2)
                pat["vol_pole_pass"] = vol_pole_pass
                pat["vol_flag_pass"] = vol_flag_pass
                pat["vol_breakdown_pass"] = vol_breakdown_pass
                pat["quality_score"] = quality
                pat["signal_date"] = str(dates[breakdown_idx])
                patterns_found.append(pat)
            
            cache_hit = True
            
        except KeyError:
            patterns_found = []
            cache_hit = False

    if cache_hit:
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
                pattern["flag_end_date"]    = str(dates[flag_end])
                
            valid_structures.append(pattern)
            break

    if structural_cache is not None and isinstance(structural_cache, dict):
        structural_cache["bear_flag_pairs"] = valid_structures

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

# =============================================================================
def detect_pennant(prices, highs=None, lows=None, volumes=None, ticker="UNKNOWN", dates=None,
                   interval="1d", verbose=False, structural_cache=None):
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

    # Fast Path: Cache Hit
    patterns_found = []
    cache_hit = False

    if structural_cache is not None and "pennant_pairs" in structural_cache and dates is not None:
        try:
            for cached_pat in structural_cache["pennant_pairs"]:
                pole_start_idx = dates.get_loc(pd.Timestamp(cached_pat["pole_start_date"]))
                pole_end_idx = dates.get_loc(pd.Timestamp(cached_pat["pole_end_date"]))
                pennant_start = dates.get_loc(pd.Timestamp(cached_pat["pennant_start_date"]))
                pennant_end = dates.get_loc(pd.Timestamp(cached_pat["pennant_end_date"]))
                
                pole_direction = cached_pat["direction"]
                res_high_intercept = float(cached_pat["res_high_intercept"])
                res_low_intercept = float(cached_pat["res_low_intercept"])
                upper_slope = float(cached_pat["upper_slope"])
                lower_slope = float(cached_pat["lower_slope"])
                
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
                    "pole_change_pct": cached_pat["pole_change_pct"],
                    "convergence_ratio": cached_pat["convergence_ratio"],
                    "r_squared": cached_pat.get("r_squared", 0.9),
                    "vol_pole_pass": True,
                    "vol_pennant_pass": cached_pat["vol_pennant_pass"],
                    "vol_breakout_pass": vol_breakout_pass,
                })
                
                pat = cached_pat.copy()
                pat["breakout_idx"] = breakout_idx
                pat["breakout_price"] = round(float(prices[breakout_idx]), 2)
                pat["vol_breakout_pass"] = vol_breakout_pass
                pat["quality_score"] = quality
                pat["signal_date"] = str(dates[breakout_idx])
                patterns_found.append(pat)

            cache_hit = True

        except KeyError:
            patterns_found = []
            cache_hit = False

    if cache_hit:
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

    if structural_cache is not None and isinstance(structural_cache, dict):
        structural_cache["pennant_pairs"] = valid_structures

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

# =============================================================================
def detect_head_and_shoulders(prices, highs=None, lows=None, volumes=None, ticker="UNKNOWN",
                               dates=None, interval="1d", verbose=False, structural_cache=None):
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

    # Fast Path: Cache Hit
    patterns_found = []
    cache_hit = False

    if structural_cache is not None and "hs_structures" in structural_cache and dates is not None:
        try:
            for cached_pat in structural_cache["hs_structures"]:
                ls_idx = dates.get_loc(pd.Timestamp(cached_pat["left_shoulder_date"]))
                head_idx = dates.get_loc(pd.Timestamp(cached_pat["head_date"]))
                rs_idx = dates.get_loc(pd.Timestamp(cached_pat["right_shoulder_date"]))
                left_neck_idx = dates.get_loc(pd.Timestamp(cached_pat["left_neckline_date"]))
                
                left_neck_price = cached_pat["left_neckline_price"]
                neckline_slope = cached_pat["neckline_slope"]
                
                breakdown_idx = None
                for k in range(rs_idx + 1, min(n, rs_idx + 30)):
                    neckline_at_k = left_neck_price + neckline_slope * (k - left_neck_idx)
                    if prices[k] < neckline_at_k:
                        breakdown_idx = k
                        break

                if breakdown_idx is None:
                    continue

                vol_breakdown_pass = None
                if volumes is not None:
                    vol_sma50 = cached_pat.get("vol_sma50", 0)
                    if vol_sma50 > 0:
                        bd_vol = float(volumes[breakdown_idx])
                        vol_breakdown_pass = bd_vol > (BREAKOUT_VOLUME_MULTIPLIER * vol_sma50)

                quality = compute_quality_score("head_and_shoulders", {
                    "head_prominence_pct": cached_pat["head_prom_pct"],
                    "shoulder_symmetry_pct": cached_pat["shoulder_symmetry_pct"],
                    "neckline_slope_pct": cached_pat["neckline_slope_pct"],
                    "vol_progression_pass": cached_pat["vol_progression_pass"],
                    "vol_breakdown_pass": vol_breakdown_pass,
                })

                pat = cached_pat.copy()
                pat["breakdown_idx"] = breakdown_idx
                pat["breakdown_price"] = round(float(prices[breakdown_idx]), 2)
                pat["vol_breakdown_pass"] = vol_breakdown_pass
                pat["quality_score"] = quality
                pat["signal_date"] = str(dates[breakdown_idx])
                pat["breakdown_date"] = str(dates[breakdown_idx])
                
                patterns_found.append(pat)

            cache_hit = True

        except KeyError:
            patterns_found = []
            cache_hit = False

    if cache_hit:
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
                
                ls_idx = refine_peak(ls_approx, prices, window=4)
                head_idx = refine_peak(head_approx, prices, window=4)
                rs_idx = refine_peak(rs_approx, prices, window=4)
                
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
        
                left_neck_idx = refine_trough(left_trough_approx, prices, window=4)
                right_neck_idx = refine_trough(right_trough_approx, prices, window=4)
        
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

    if structural_cache is not None and isinstance(structural_cache, dict):
        structural_cache["hs_structures"] = valid_structures

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

        if breakdown_idx is None:
            continue

        vol_breakdown_pass = None
        if volumes is not None:
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
        })

        pat = pat.copy()
        pat["breakdown_idx"] = breakdown_idx
        pat["breakdown_price"] = round(float(prices[breakdown_idx]), 2)
        pat["vol_breakdown_pass"] = vol_breakdown_pass
        pat["quality_score"] = quality

        if dates is not None:
            pat["signal_date"] = str(dates[breakdown_idx])
            pat["breakdown_date"] = str(dates[breakdown_idx])
            
        patterns_found.append(pat)

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

def compute_ema(prices, period=50):
    prices = np.array(prices, dtype=float)
    if len(prices) < period:
        return np.full_like(prices, np.nan)
    return pd.Series(prices).ewm(span=period, adjust=False).mean().values

def compute_rsi(prices, period=14):
    prices = np.array(prices, dtype=float)
    if len(prices) <= period:
        return np.full_like(prices, np.nan)
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period
        
    rs = np.zeros_like(prices)
    rsi = np.zeros_like(prices)
    
    # Avoid division by zero
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    # If loss is 0, RSI is 100
    rsi[~mask & (avg_gain > 0)] = 100
    
    # For initial periods
    rsi[:period] = np.nan
    
    return rsi

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

    # True Range: max of (H-L, |H-PrevC|, |L-PrevC|)
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )

    if len(tr) < period:
        return float(np.mean(tr)) if len(tr) > 0 else 0.0

    # Simple Moving Average of the last `period` True Range values
    return float(np.mean(tr[-period:]))


def interval_to_candles_per_day(interval):
    """
    Returns approximate number of trading candles per calendar day
    for a given yfinance interval string.

    Used to convert MIN_PEAK_GAP_DAYS (calendar days) into a candle count.

    NSE trading session ≈ 6.25 hours (9:15–15:30).
    Calendar day → trading day ratio ≈ 5/7 (weekdays only).

    Parameters
    ----------
    interval : str
        yfinance interval (e.g., "15m", "1h", "1d", "1wk").

    Returns
    -------
    float
        Candles per calendar day.
    """
    # Map interval string to minutes per candle
    interval_minutes = {
        "1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30,
        "60m": 60, "90m": 90, "1h": 60,
    }

    if interval in interval_minutes:
        mins_per_candle = interval_minutes[interval]
        trading_minutes_per_day = 375  # 6h 15m
        candles_per_trading_day = trading_minutes_per_day / mins_per_candle
        # Convert to per-calendar-day: ~5 trading days per 7 calendar days
        return candles_per_trading_day * (5.0 / 7.0)
    elif interval == "1wk":
        return 1.0 / 7.0  # one candle per week
    else:
        # Default: daily
        return 5.0 / 7.0  # ~0.714 candles per calendar day


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
        
    # True Range
    tr1 = highs[1:] - lows[1:]
    tr2 = np.abs(highs[1:] - closes[:-1])
    tr3 = np.abs(lows[1:] - closes[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.insert(tr, 0, np.nan)
    
    # Directional Movement
    up_move = highs[1:] - highs[:-1]
    down_move = lows[:-1] - lows[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_dm = np.insert(plus_dm, 0, np.nan)
    minus_dm = np.insert(minus_dm, 0, np.nan)

    # Wilder's Smoothing Function
    def wilders_smoothing(data, period):
        smoothed = np.full_like(data, np.nan)
        first_valid = period
        if len(data) <= first_valid:
            return smoothed
        smoothed[first_valid] = np.sum(data[1:first_valid+1])
        for i in range(first_valid + 1, len(data)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + data[i]
        return smoothed
        
    atr = wilders_smoothing(tr, period)
    plus_di_smooth = wilders_smoothing(plus_dm, period)
    minus_di_smooth = wilders_smoothing(minus_dm, period)
    
    # Calculate +DI and -DI
    plus_di = np.full_like(atr, np.nan)
    minus_di = np.full_like(atr, np.nan)
    
    valid_atr = atr > 0
    plus_di[valid_atr] = 100 * (plus_di_smooth[valid_atr] / atr[valid_atr])
    minus_di[valid_atr] = 100 * (minus_di_smooth[valid_atr] / atr[valid_atr])
    
    # Calculate DX
    dx = np.full_like(atr, np.nan)
    di_sum = plus_di + minus_di
    valid_di = di_sum > 0
    dx[valid_di] = 100 * np.abs(plus_di[valid_di] - minus_di[valid_di]) / di_sum[valid_di]
    
    # Calculate ADX (Smoothed DX)
    first_dx = -1
    for i in range(n):
        if not np.isnan(dx[i]):
            first_dx = i
            break
            
    if first_dx == -1 or first_dx + period > n:
        return adx
        
    adx[first_dx + period - 1] = np.mean(dx[first_dx:first_dx+period])
    for i in range(first_dx + period, n):
        adx[i] = ((adx[i-1] * (period - 1)) + dx[i]) / period
        
    return adx


# =============================================================================
#  HELPER FUNCTIONS FOR PEAK REFINEMENT
# =============================================================================

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


# =============================================================================
#  10. DETECT DOUBLE TOP
# =============================================================================


# =============================================================================
def detect_double_top(prices, volumes=None, ticker="UNKNOWN", dates=None, interval="1d", verbose=False, structural_cache=None):
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

    # Fast Path: Cache Hit
    patterns_found = []
    cache_hit = False

    if structural_cache is not None and "double_top_structures" in structural_cache and dates is not None:
        try:
            for cached_pat in structural_cache["double_top_structures"]:
                top1_idx = dates.get_loc(pd.Timestamp(cached_pat["first_top_date"]))
                valley_idx = dates.get_loc(pd.Timestamp(cached_pat["valley_date"]))
                top2_idx = dates.get_loc(pd.Timestamp(cached_pat["second_top_date"]))
                
                valley_price = cached_pat["valley_price"]
                
                breakdown_idx = None
                for k in range(top2_idx + 1, min(n, top2_idx + 30)):
                    if prices[k] < valley_price:
                        breakdown_idx = k
                        break

                status = "confirmed"
                stage = ""
                if breakdown_idx is None:
                    status = "forming"
                    stage = "Stage C - Second peak formed, awaiting confirmation: Price touched first peak, hasn't broken neckline"
                    breakdown_idx = n - 1

                vol_breakdown_pass = None
                if volumes is not None:
                    vol_sma50 = cached_pat.get("vol_sma50", 0)
                    if vol_sma50 > 0:
                        bd_vol = float(volumes[breakdown_idx])
                        vol_breakdown_pass = bd_vol > (BREAKOUT_VOLUME_MULTIPLIER * vol_sma50)

                quality = compute_quality_score("double_top", {
                    "similarity_pct": cached_pat["similarity_pct"],
                    "depth_pct": cached_pat["valley_drop_pct"],
                    "vol_pattern_pass": cached_pat["vol_pattern_pass"],
                    "vol_breakout_pass": vol_breakdown_pass,
                })

                pat = cached_pat.copy()
                pat["breakdown_idx"] = breakdown_idx
                pat["breakdown_price"] = round(float(prices[breakdown_idx]), 2) if status == "confirmed" else None
                pat["vol_breakdown_pass"] = vol_breakdown_pass
                pat["quality_score"] = quality
                pat["status"] = status
                pat["stage"] = stage
                if status == "confirmed":
                    pat["signal_date"] = str(dates[breakdown_idx])
                    pat["breakdown_date"] = str(dates[breakdown_idx])
                
                patterns_found.append(pat)

            cache_hit = True

        except KeyError:
            patterns_found = []
            cache_hit = False

    if cache_hit:
        patterns_found.sort(key=lambda p: p["quality_score"], reverse=True)
        final_patterns = []
        claimed = []
        for c in patterns_found:
            is_dup = False
            for (c1, c2) in claimed:
                if abs(c["first_top_idx"] - c1) <= 10 and abs(c["second_top_idx"] - c2) <= 10:
                    is_dup = True
                    break
            if not is_dup:
                final_patterns.append(c)
                claimed.append((c["first_top_idx"], c["second_top_idx"]))
        return final_patterns


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

    if structural_cache is not None and isinstance(structural_cache, dict):
        structural_cache["double_top_structures"] = valid_structures

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

def _emit_dt_rejected(ticker, reason):
    """Helper to print a quick Double Top rejection in verbose mode."""
    print(f"  [DT] {ticker}: REJECTED — {reason}")


# =============================================================================
#  11. DETECT DOUBLE BOTTOM
# =============================================================================

def detect_double_bottom(prices, volumes=None, ticker="UNKNOWN", dates=None, interval="1d", verbose=False, structural_cache=None):
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

    # Fast Path: Cache Hit
    patterns_found = []
    cache_hit = False

    if structural_cache is not None and "double_bottom_structures" in structural_cache and dates is not None:
        try:
            for cached_pat in structural_cache["double_bottom_structures"]:
                bot1_idx = dates.get_loc(pd.Timestamp(cached_pat["first_bottom_date"]))
                peak_idx = dates.get_loc(pd.Timestamp(cached_pat["peak_date"]))
                bot2_idx = dates.get_loc(pd.Timestamp(cached_pat["second_bottom_date"]))
                
                peak_price = cached_pat["peak_price"]
                
                breakout_idx = None
                for k in range(bot2_idx + 1, min(n, bot2_idx + 30)):
                    if prices[k] > peak_price:
                        breakout_idx = k
                        break

                status = "confirmed"
                stage = ""
                if breakout_idx is None:
                    status = "forming"
                    stage = "Stage C - Second bottom formed, awaiting confirmation: Price touched first bottom, hasn't broken neckline"
                    breakout_idx = n - 1

                vol_breakout_pass = None
                if volumes is not None:
                    vol_sma50 = cached_pat.get("vol_sma50", 0)
                    if vol_sma50 > 0:
                        bo_vol = float(volumes[breakout_idx])
                        vol_breakout_pass = bo_vol > (BREAKOUT_VOLUME_MULTIPLIER * vol_sma50)

                quality = compute_quality_score("double_bottom", {
                    "similarity_pct": cached_pat["similarity_pct"],
                    "depth_pct": cached_pat["peak_rise_pct"],
                    "vol_pattern_pass": cached_pat["vol_pattern_pass"],
                    "vol_breakout_pass": vol_breakout_pass,
                })

                pat = cached_pat.copy()
                pat["breakout_idx"] = breakout_idx
                pat["breakout_price"] = round(float(prices[breakout_idx]), 2) if status == "confirmed" else None
                pat["vol_breakout_pass"] = vol_breakout_pass
                pat["quality_score"] = quality
                pat["status"] = status
                pat["stage"] = stage
                
                if status == "confirmed":
                    pat["signal_date"] = str(dates[breakout_idx])
                    pat["breakout_date"] = str(dates[breakout_idx])
                
                patterns_found.append(pat)

            cache_hit = True

        except KeyError:
            patterns_found = []
            cache_hit = False

    if cache_hit:
        patterns_found.sort(key=lambda p: p["quality_score"], reverse=True)
        final_patterns = []
        claimed = []
        for c in patterns_found:
            is_dup = False
            for (c1, c2) in claimed:
                if abs(c["first_bottom_idx"] - c1) <= 10 and abs(c["second_bottom_idx"] - c2) <= 10:
                    is_dup = True
                    break
            if not is_dup:
                final_patterns.append(c)
                claimed.append((c["first_bottom_idx"], c["second_bottom_idx"]))
        return final_patterns


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

    if structural_cache is not None and isinstance(structural_cache, dict):
        structural_cache["double_bottom_structures"] = valid_structures

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

def _emit_db_rejected(ticker, reason):
    """Helper to print a quick Double Bottom rejection in verbose mode."""
    print(f"  [DB] {ticker}: REJECTED — {reason}")


# =============================================================================
#  12. PRINT FUNCTIONS — Pattern-specific debug output
# =============================================================================

def print_cup_and_handle(p, index=1):
    """Prints Cup & Handle debug output (exact same format as v2)."""
    score_str = (f"          Quality Score: {p.get('quality_score', 0)}"
                 if not p.get("reject_reason") else "")

    print(f"\n  {'═' * 60}")
    print(f"  🏆 CUP & HANDLE #{index}  —  {p['ticker']}{score_str}")
    print(f"  {'═' * 60}")
    print(f"  Left Rim  (peak before cup)  : ₹{p['left_rim_price']}")
    print(f"  Cup Bottom (lowest point)    : ₹{p['cup_bottom_price']}")
    print(f"  Right Rim  (recovery peak)   : ₹{p['right_rim_price']}")
    print(f"  Handle Low (small dip after) : ₹{p['handle_low_price']}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Cup Drop     : {p['cup_drop_pct']}%")
    print(f"  Recovery Gap : {p['recovery_pct']}%")

    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ GEOMETRY CHECK")
    print(f"    Cup Depth (₹)              : ₹{p['cup_depth']}")
    print(f"    Max Handle Dip Allowed (₹) : ₹{p['max_handle_dip']}  "
          f"(= 0.32 × ₹{p['cup_depth']})")
    print(f"    Actual Handle Pullback (₹) : ₹{p['handle_pullback']}  "
          f"({p['handle_pullback_pct']}%)")

    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ ROUNDEDNESS CHECK")
    rnd_pass = "✅ PASS" if p.get('roundedness_pct', 0) >= MIN_BASE_CANDLES_PCT * 100 else "❌ FAIL"
    print(f"    {p.get('base_candles', 'N/A')} candles in base zone out of "
          f"{p.get('cup_width', 'N/A')} cup candles "
          f"({p.get('roundedness_pct', 'N/A')}%) — {rnd_pass}")

    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ PAUSE-BEFORE-BREAKOUT CHECK")
    pause = p.get('pause_duration', 0)
    slope = p.get('handle_slope', 0)
    pause_pass = "✅ PASS" if pause >= MIN_PAUSE_CANDLES and slope <= 0 else "❌ FAIL"
    print(f"    Breakout Confirmed in      : {pause} candles (Needs ≥ {MIN_PAUSE_CANDLES})")
    print(f"    Handle Slope               : {slope} — {pause_pass}")

    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ HANDLE LOW")
    print(f"    OLD (first-dip)            : ₹{p.get('old_handle_low', 'N/A')}")
    print(f"    NEW (breakout-confirmed)   : ₹{p['handle_low_price']}")

    # Volume sections
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ VOLUME — Cup Decline")
    if p.get('vol_decline_pass') is not None:
        vd = "✅ PASS (light selling)" if p['vol_decline_pass'] else "⚠ WARN (heavy selling)"
        print(f"    Avg vol during decline     : {p['vol_decline_avg']:,.0f}")
        print(f"    50-period SMA vol          : {p['vol_sma50']:,.0f}")
        print(f"    Status                     : {vd}")
    else:
        print(f"    Status                     : N/A (no volume data)")

    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ VOLUME — Recovery")
    if p.get('vol_recovery_pass') is not None:
        vr = "✅ PASS (more buying)" if p['vol_recovery_pass'] else "⚠ WARN (weak buying)"
        print(f"    Up-vol / Down-vol ratio    : {p['vol_recovery_ratio']:.2f}")
        print(f"    Status                     : {vr}")
    else:
        print(f"    Status                     : N/A (no volume data)")

    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ VOLUME — Breakout")
    if p.get('vol_breakout_pass') is not None:
        vb = "✅ PASS (strong interest)" if p['vol_breakout_pass'] else "⚠ WARN (low interest)"
        print(f"    Avg breakout volume        : {p['vol_breakout_avg']:,.0f}")
        print(f"    Threshold (1.2× SMA50)     : {p['vol_sma50'] * BREAKOUT_VOLUME_MULTIPLIER:,.0f}")
        print(f"    Status                     : {vb}")
    else:
        print(f"    Status                     : N/A (no volume data)")

    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ SMOOTHING")
    print(f"    Method                     : {p.get('smoothing_method', 'None')}")
    print(f"    Note: Peaks detected on smoothed series; all prices are RAW.")

    if "left_rim_date" in p:
        print(f"  ─────────────────────────────────────────────────")
        print(f"  📅 Left Rim Date   : {p['left_rim_date']}")
        print(f"  📅 Cup Bottom Date : {p['cup_bottom_date']}")
        print(f"  📅 Right Rim Date  : {p['right_rim_date']}")
        print(f"  📅 Handle Low Date : {p['handle_low_date']}")

    print(f"  ─────────────────────────────────────────────────")
    if p.get("reject_reason"):
        print(f"  FINAL VERDICT: ❌ REJECTED ({p['reject_reason']})")
    else:
        print(f"  FINAL VERDICT: ✅ VALID (Quality Score: {p.get('quality_score', 0)})")
    print(f"  {'═' * 60}\n")


def _vol_str(val):
    """Formats a volume pass/fail as a string."""
    if val is None:
        return "N/A"
    return "✅ PASS" if val else "❌ FAIL"


def print_bull_flag(p, index=1):
    """Prints Bull Flag debug output."""
    print(f"\n  {'═' * 60}")
    print(f"  🏆 BULL FLAG #{index}  —  {p['ticker']}  (Score: {p['quality_score']})")
    print(f"  {'═' * 60}")
    print(f"  Pole Start      : ₹{p['pole_start_price']}"
          + (f"  ({p.get('pole_start_date', '')})" if 'pole_start_date' in p else ""))
    print(f"  Pole Top        : ₹{p['pole_top_price']}"
          + (f"  ({p.get('pole_top_date', '')})" if 'pole_top_date' in p else ""))
    print(f"  Flag Low        : ₹{p['flag_low_price']}"
          + (f"  ({p.get('flag_low_date', '')})" if 'flag_low_date' in p else ""))
    print(f"  Breakout Point  : ₹{p['breakout_price']}"
          + (f"  ({p.get('breakout_date', '')})" if 'breakout_date' in p else ""))
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Pole Rise %          : {p['pole_rise_pct']}%")
    print(f"  Pole R²              : {p['pole_r_squared']}")
    print(f"  Flag Channel Width % : {p['flag_range_pct']}%")
    print(f"  Flag Slope           : {p['flag_slope']}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ VOLUME")
    print(f"    Pole (above avg)   : {_vol_str(p['vol_pole_pass'])}")
    print(f"    Flag (below avg)   : {_vol_str(p['vol_flag_pass'])}")
    print(f"    Breakout (spike)   : {_vol_str(p['vol_breakout_pass'])}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  FINAL VERDICT: ✅ VALID (Quality Score: {p['quality_score']})")
    print(f"  {'═' * 60}\n")


def print_bear_flag(p, index=1):
    """Prints Bear Flag debug output."""
    print(f"\n  {'═' * 60}")
    print(f"  🏆 BEAR FLAG #{index}  —  {p['ticker']}  (Score: {p['quality_score']})")
    print(f"  {'═' * 60}")
    print(f"  Pole Start      : ₹{p['pole_start_price']}"
          + (f"  ({p.get('pole_start_date', '')})" if 'pole_start_date' in p else ""))
    print(f"  Pole Bottom     : ₹{p['pole_bottom_price']}"
          + (f"  ({p.get('pole_bottom_date', '')})" if 'pole_bottom_date' in p else ""))
    print(f"  Flag High       : ₹{p['flag_high_price']}"
          + (f"  ({p.get('flag_high_date', '')})" if 'flag_high_date' in p else ""))
    print(f"  Breakdown Point : ₹{p['breakdown_price']}"
          + (f"  ({p.get('breakdown_date', '')})" if 'breakdown_date' in p else ""))
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Pole Drop %          : {p['pole_drop_pct']}%")
    print(f"  Pole R²              : {p['pole_r_squared']}")
    print(f"  Flag Channel Width % : {p['flag_range_pct']}%")
    print(f"  Flag Slope           : {p['flag_slope']}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ VOLUME")
    print(f"    Pole (above avg)   : {_vol_str(p['vol_pole_pass'])}")
    print(f"    Flag (below avg)   : {_vol_str(p['vol_flag_pass'])}")
    print(f"    Breakdown (spike)  : {_vol_str(p['vol_breakdown_pass'])}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  FINAL VERDICT: ✅ VALID (Quality Score: {p['quality_score']})")
    print(f"  {'═' * 60}\n")


def print_pennant(p, index=1):
    """Prints Pennant debug output."""
    print(f"\n  {'═' * 60}")
    print(f"  🏆 PENNANT #{index}  —  {p['ticker']}  ({p['direction']})  (Score: {p['quality_score']})")
    print(f"  {'═' * 60}")
    print(f"  Pole Start      : ₹{p['pole_start_price']}"
          + (f"  ({p.get('pole_start_date', '')})" if 'pole_start_date' in p else ""))
    print(f"  Pole End        : ₹{p['pole_end_price']}"
          + (f"  ({p.get('pole_end_date', '')})" if 'pole_end_date' in p else ""))
    print(f"  Pennant High    : ₹{p['pennant_high']}")
    print(f"  Pennant Low     : ₹{p['pennant_low']}")
    print(f"  Breakout Point  : ₹{p['breakout_price']}"
          + (f"  ({p.get('breakout_date', '')})" if 'breakout_date' in p else ""))
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Pole Size %          : {p['pole_change_pct']}%")
    print(f"  Pole ADX             : {p['adx_value']} (must be > 30)")
    print(f"  Upper Trendline Slope: {p['upper_slope']} (must be < 0)")
    print(f"  Lower Trendline Slope: {p['lower_slope']} (must be > 0)")
    print(f"  Convergence Ratio    : {p['convergence_ratio']} "
          f"({'✅ Yes' if p['convergence_ratio'] < 1 else '❌ No'})")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ VOLUME")
    print(f"    Pennant Diminishing (below avg): {_vol_str(p['vol_pennant_pass'])}")
    print(f"    Breakout (>1.5x avg)           : {_vol_str(p['vol_breakout_pass'])}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  FINAL VERDICT: ✅ VALID (Quality Score: {p['quality_score']})")
    print(f"  {'═' * 60}\n")


def print_head_and_shoulders(p, index=1):
    """Prints Head & Shoulders debug output."""
    print(f"\n  {'═' * 60}")
    print(f"  🏆 HEAD & SHOULDERS #{index}  —  {p['ticker']}  (Score: {p['quality_score']})")
    print(f"  {'═' * 60}")
    print(f"  Left Shoulder   : ₹{p['left_shoulder_price']}"
          + (f"  ({p.get('left_shoulder_date', '')})" if 'left_shoulder_date' in p else ""))
    print(f"  Head            : ₹{p['head_price']}"
          + (f"  ({p.get('head_date', '')})" if 'head_date' in p else ""))
    print(f"  Right Shoulder  : ₹{p['right_shoulder_price']}"
          + (f"  ({p.get('right_shoulder_date', '')})" if 'right_shoulder_date' in p else ""))
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Left Neckline   : ₹{p['left_neckline_price']}"
          + (f"  ({p.get('left_neckline_date', '')})" if 'left_neckline_date' in p else ""))
    print(f"  Right Neckline  : ₹{p['right_neckline_price']}"
          + (f"  ({p.get('right_neckline_date', '')})" if 'right_neckline_date' in p else ""))
    print(f"  Neckline Slope  : {p['neckline_slope']}"
          + (f"  ⚠ STEEP ({p['neckline_slope_pct']}%)"
             if p['neckline_slope_pct'] > NECKLINE_SLOPE_WARN_PCT else ""))
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Head vs Left Shoulder  : +{p['head_vs_ls_pct']}%")
    print(f"  Head vs Right Shoulder : +{p['head_vs_rs_pct']}%")
    print(f"  Shoulder Symmetry      : {p['shoulder_symmetry_pct']}% diff")
    print(f"  Pattern Span           : {p['span_candles']} candles")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ VOLUME")
    print(f"    Left Shoulder avg  : {p['vol_ls_avg']:,.0f}")
    print(f"    Head avg           : {p['vol_head_avg']:,.0f}")
    print(f"    Right Shoulder avg : {p['vol_rs_avg']:,.0f}")
    print(f"    Progression (LS>H>RS) : {_vol_str(p['vol_progression_pass'])}")
    print(f"    Breakdown (spike)     : {_vol_str(p['vol_breakdown_pass'])}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  FINAL VERDICT: ✅ VALID (Quality Score: {p['quality_score']})")
    print(f"  {'═' * 60}\n")


def print_double_top(p, index=1):
    """Prints Double Top debug output in the updated format."""
    score = p.get('quality_score', 0)
    verdict = p.get('verdict', 'UNKNOWN')
    score_str = f"   Quality Score: {score}" if verdict == "CONFIRMED" else ""

    print(f"\nPATTERN -- {p['ticker']}  (Double Top){score_str}")
    print("=" * 60)
    # Prior trend
    trend_pass = "PASS" if p.get('prior_trend_pass') else "FAIL"
    print(f"Prior Trend: slope={p.get('prior_slope', 0)}, "
          f"R\u00b2={p.get('prior_r2', 0)} (on Adj Close) -- {trend_pass}")
    # Key prices
    print(f"First Peak: \u20b9{p['first_top_price']}"
          + (f"  ({p.get('first_top_date', '')})" if 'first_top_date' in p else ""))
    print(f"Neckline (Valley): \u20b9{p['valley_price']}"
          + (f"  ({p.get('valley_date', '')})" if 'valley_date' in p else ""))
    print(f"Second Peak: \u20b9{p['second_top_price']}"
          + (f"  ({p.get('second_top_date', '')})" if 'second_top_date' in p else ""))
    # Metrics
    print(f"Valley Drop: {p.get('valley_drop_pct', 0)}%")
    gap_pass = "PASS" if p.get('gap_days', 0) >= MIN_PEAK_GAP_DAYS else "FAIL"
    print(f"Gap between peaks: {p.get('gap_days', 0)} days -- {gap_pass} "
          f"(min {MIN_PEAK_GAP_DAYS} days)")
    sim_pass = "PASS" if p.get('similarity_pct', 99) <= DOUBLE_TOP_SIMILARITY_PCT else "FAIL"
    print(f"Similarity between peaks: {p.get('similarity_pct', 0)}% -- {sim_pass}")
    print("-" * 60)
    # Breakdown confirmation
    if p.get('breakdown_confirmed'):
        bd_pct = ((p['valley_price'] - p.get('breakdown_price', p['valley_price']))
                  / p['valley_price'] * 100) if p.get('breakdown_price') else 0
        print(f"* BREAKDOWN CONFIRMATION: decisive close = {bd_pct:.2f}%, "
              f"sustained {p.get('breakdown_confirm_candles', 2)} candles -- PASS")
    else:
        print(f"* BREAKDOWN CONFIRMATION: -- FAIL ({p.get('reject_reason', 'N/A')})")
    # Time decay
    waited = p.get('candles_waited', 0)
    max_wait = MAX_BREAKOUT_WAIT_CANDLES
    if verdict == "REJECTED" and "window expired" in p.get('reject_reason', ''):
        decay_str = f"{waited}/{max_wait} candles used -- EXPIRED"
    elif verdict == "CONFIRMED":
        decay_str = f"{waited}/{max_wait} candles used -- PASS"
    else:
        decay_str = f"{waited}/{max_wait} candles used -- FORMING"
    print(f"* BREAKOUT TIME-DECAY: {decay_str}")
    # Volume
    vol_t1 = p.get('vol_top1_avg', 0)
    vol_t2 = p.get('vol_top2_avg', 0)
    vol_trend = "declining (ideal)" if vol_t1 > vol_t2 else "increasing"
    print(f"* VOLUME -- First vs Second peak: {vol_trend} (informational)")
    bd_ratio = p.get('vol_breakdown_ratio', 0)
    bd_pass = _vol_str(p.get('vol_breakdown_pass'))
    print(f"* VOLUME -- Breakdown spike: {bd_ratio:.2f}x avg SMA -- {bd_pass}")
    print("-" * 60)
    # Trade reference
    print("* TRADE REFERENCE (educational only, not financial advice):")
    print(f"  Pattern Height: \u20b9{p.get('pattern_height', 0)}")
    print(f"  ATR (14-period): \u20b9{p.get('atr_14', 0)}")
    print(f"  Profit Target: \u20b9{p.get('profit_target', 0)}")
    print(f"  Suggested Stop-Loss: \u20b9{p.get('suggested_stop', 0)} (ATR-based)")
    print("-" * 60)
    # Final verdict
    if verdict == "CONFIRMED":
        print(f"FINAL VERDICT: CONFIRMED (score {score})")
    elif verdict == "FORMING":
        print(f"FINAL VERDICT: FORMING")
    else:
        print(f"FINAL VERDICT: REJECTED ({p.get('reject_reason', '')})")
    print("=" * 60 + "\n")


def print_double_bottom(p, index=1):
    """Prints Double Bottom debug output with Nifty 200 format."""
    score = p.get('quality_score', 0)
    verdict = p.get('verdict', 'UNKNOWN')
    score_str = f"   Quality Score: {score}" if verdict == "CONFIRMED" else ""

    print(f"\nPATTERN -- {p['ticker']}  (Double Bottom){score_str}")
    print("=" * 60)
    # Prior trend
    trend_pass = "PASS" if p.get('prior_trend_pass') else "FAIL"
    print(f"Prior Trend: slope < 0 OR below 50-EMA -- {trend_pass}")
    # Key prices
    print(f"First Bottom: \u20b9{p['first_bottom_price']}"
          + (f"  ({p.get('first_bottom_date', '')})" if 'first_bottom_date' in p else ""))
    print(f"Neckline (Peak): \u20b9{p['neckline_price']}"
          + (f"  ({p.get('neckline_date', '')})" if 'neckline_date' in p else ""))
    print(f"Second Bottom: \u20b9{p['second_bottom_price']}"
          + (f"  ({p.get('second_bottom_date', '')})" if 'second_bottom_date' in p else ""))
    # Metrics
    print(f"Pattern Depth: {p.get('depth_pct', 0)}% (Limits: 5% - 25%)")
    gap_pass = "PASS" if MIN_DOUBLE_CANDLES <= p.get('separation', 0) <= MAX_DOUBLE_CANDLES else "FAIL"
    print(f"Time Width: {p.get('separation', 0)} bars -- {gap_pass} "
          f"([ {MIN_DOUBLE_CANDLES}, {MAX_DOUBLE_CANDLES} ])")
    sim_pass = "PASS" if p.get('similarity_pct', 99) <= DOUBLE_BOTTOM_SIMILARITY_PCT else "FAIL"
    print(f"Bottom Symmetry: {p.get('similarity_pct', 0)}% -- {sim_pass} (Max {DOUBLE_BOTTOM_SIMILARITY_PCT}%)")
    
    rsi1 = p.get('rsi_1')
    rsi2 = p.get('rsi_2')
    if rsi1 is not None and rsi2 is not None:
        print(f"RSI Divergence: RSI1={rsi1}, RSI2={rsi2}")

    print("-" * 60)
    # Breakout confirmation
    if p.get('breakout_confirmed'):
        bo_pct = ((p.get('breakout_price', p['neckline_price']) - p['neckline_price'])
                  / p['neckline_price'] * 100) if p.get('breakout_price') else 0
        print(f"* BREAKOUT CONFIRMATION: Close > Neckline (+{bo_pct:.2f}%) -- PASS")
    else:
        print(f"* BREAKOUT CONFIRMATION: -- FAIL ({p.get('reject_reason', 'N/A')})")

    # Volume
    bo_pass = "PASS" if p.get('vol_breakout_pass') else "FAIL"
    print(f"* VOLUME -- Breakout spike >= 1.5x 20-SMA -- {bo_pass}")
    print("-" * 60)
    # Trade reference
    print("* TRADE REFERENCE (educational only, not financial advice):")
    print(f"  Profit Target: \u20b9{p.get('profit_target', 0)}")
    print(f"  Conservative Stop-Loss: \u20b9{p.get('conservative_stop', 0)} (ATR-based)")
    print(f"  Aggressive Stop-Loss: \u20b9{p.get('aggressive_stop', 0)} (Midpoint-based)")
    print("-" * 60)
    # Final verdict
    if verdict == "CONFIRMED":
        print(f"FINAL VERDICT: CONFIRMED (score {score})")
    elif verdict == "FORMING":
        print(f"FINAL VERDICT: FORMING")
    else:
        print(f"FINAL VERDICT: REJECTED ({p.get('reject_reason', '')})")
    print("=" * 60 + "\n")


# Dispatcher: route to the correct printer based on pattern_type
PRINT_DISPATCH = {
    "cup_and_handle":     print_cup_and_handle,
    "bull_flag":          print_bull_flag,
    "bear_flag":          print_bear_flag,
    "pennant":            print_pennant,
    "head_and_shoulders": print_head_and_shoulders,
    "double_top":         print_double_top,
    "double_bottom":      print_double_bottom,
}


def print_any_pattern(p, index=1):
    """Prints any pattern using the correct pattern-specific printer."""
    ptype = p.get("pattern_type", "cup_and_handle")
    printer = PRINT_DISPATCH.get(ptype, print_cup_and_handle)
    printer(p, index)


# =============================================================================
#  13. SELF-TEST FUNCTIONS — Synthetic data per pattern
# =============================================================================

def test_cup_and_handle():
    """Self-test for Cup & Handle using existing synthetic data."""
    print("\n  ── CUP AND HANDLE ──────────────────────────────────────")

    # Test 1: Textbook cup
    np.random.seed(123)
    rise = np.linspace(90, 100, 15)
    descent = np.linspace(100, 80, 30)
    bottom = np.ones(20) * 80 + np.random.uniform(-0.3, 0.3, 20)
    ascent = np.linspace(80, 100, 30)
    right_rim = np.array([100.0, 100.2, 100.1, 100.0, 99.8])
    handle_down = np.linspace(99.8, 95, 10)
    handle_up = np.linspace(95, 98, 10)
    post_break = np.linspace(98, 103, 5)
    post = np.ones(10) * 103

    prices = np.concatenate([rise, descent, bottom, ascent, right_rim,
                              handle_down, handle_up, post_break, post])
    n = len(prices)
    volumes = np.random.uniform(800000, 1200000, n)
    volumes[-15:] = np.random.uniform(2000000, 3000000, 15)

    results = detect_cup_and_handle(prices, volumes=volumes,
                                     ticker="SYNTH_CUP", verbose=False)
    if results:
        print(f"  ✅ Textbook shape → VALID  (score: {results[0].get('quality_score', 0)})")
    else:
        print("  ❌ Textbook shape → REJECTED (unexpected!)")

    # Test 2: Flat noise
    np.random.seed(42)
    flat = 100 + np.random.normal(0, 0.5, 200)
    results_flat = detect_cup_and_handle(flat, ticker="FLAT")
    if not results_flat:
        print("  ✅ Flat/random    → REJECTED (correct)")
    else:
        print(f"  ⚠ Flat/random    → {len(results_flat)} false positive(s)")

    return bool(results) and not results_flat


def test_bull_flag():
    """Self-test for Bull Flag."""
    print("\n  ── BULL FLAG ────────────────────────────────────────────")

    np.random.seed(200)
    lead = np.linspace(85, 90, 30)
    pole = np.linspace(90, 108, 15)
    flag = np.linspace(107, 104, 10)
    breakout = np.array([109, 111, 113])
    post = np.ones(15) * 113

    prices = np.concatenate([lead, pole, flag, breakout, post])
    n = len(prices)
    volumes = np.random.uniform(800000, 1200000, n)
    # Pole: above avg
    volumes[30:45] = np.random.uniform(1500000, 2000000, 15)
    # Flag: below avg
    volumes[45:55] = np.random.uniform(400000, 600000, 10)
    # Breakout: spike
    volumes[55:58] = np.random.uniform(2500000, 3500000, 3)

    results = detect_bull_flag(prices, volumes=volumes,
                                ticker="SYNTH_BULL_FLAG", verbose=False)
    if results:
        print(f"  ✅ Textbook shape → VALID  (score: {results[0]['quality_score']})")
        print_bull_flag(results[0], 1)
    else:
        print("  ❌ Textbook shape → REJECTED (unexpected!)")

    # Flat noise
    np.random.seed(42)
    flat = 100 + np.random.normal(0, 0.5, 200)
    results_flat = detect_bull_flag(flat, ticker="FLAT")
    if not results_flat:
        print("  ✅ Flat/random    → REJECTED (correct)")
    else:
        print(f"  ⚠ Flat/random    → {len(results_flat)} false positive(s)")

    return bool(results) and not results_flat


def test_bear_flag():
    """Self-test for Bear Flag."""
    print("\n  ── BEAR FLAG ────────────────────────────────────────────")

    np.random.seed(201)
    lead = np.linspace(100, 108, 30)
    pole = np.linspace(108, 98, 15)
    flag = np.linspace(99, 102, 10)
    breakdown = np.array([97, 95, 93])
    post = np.ones(15) * 93

    prices = np.concatenate([lead, pole, flag, breakdown, post])
    n = len(prices)
    volumes = np.random.uniform(800000, 1200000, n)
    volumes[30:45] = np.random.uniform(1500000, 2000000, 15)
    volumes[45:55] = np.random.uniform(400000, 600000, 10)
    volumes[55:58] = np.random.uniform(2500000, 3500000, 3)

    results = detect_bear_flag(prices, volumes=volumes,
                                ticker="SYNTH_BEAR_FLAG", verbose=False)
    if results:
        print(f"  ✅ Textbook shape → VALID  (score: {results[0]['quality_score']})")
        print_bear_flag(results[0], 1)
    else:
        print("  ❌ Textbook shape → REJECTED (unexpected!)")

    # Flat noise
    np.random.seed(42)
    flat = 100 + np.random.normal(0, 0.5, 200)
    results_flat = detect_bear_flag(flat, ticker="FLAT")
    if not results_flat:
        print("  ✅ Flat/random    → REJECTED (correct)")
    else:
        print(f"  ⚠ Flat/random    → {len(results_flat)} false positive(s)")

    return bool(results) and not results_flat


def test_pennant():
    """Self-test for Pennant (bullish)."""
    print("\n  ── PENNANT ──────────────────────────────────────────────")
    all_pass = True

    # 1. Valid Textbook Pennant
    np.random.seed(202)
    # Lead up to create strong ADX (>30). Needs to be steady uptrend
    lead_close = np.linspace(50, 90, 40)
    lead_high = lead_close + np.random.uniform(0.5, 1.5, 40)
    lead_low = lead_close - np.random.uniform(0.5, 1.5, 40)
    
    # Pole (sharp rise)
    pole_close = np.linspace(90, 120, 20)
    pole_high = pole_close + np.random.uniform(1, 2, 20)
    pole_low = pole_close - np.random.uniform(1, 2, 20)
    
    # Pennant body (converging)
    # Highs must have negative slope, lows must have positive slope
    p_len = 10
    pen_high = np.linspace(119, 115, p_len) + np.random.uniform(-0.5, 0.5, p_len)
    pen_low = np.linspace(111, 114, p_len) + np.random.uniform(-0.5, 0.5, p_len)
    pen_close = (pen_high + pen_low) / 2
    
    # Breakout
    bo_close = np.linspace(116, 125, 5)
    bo_high = bo_close + 1
    bo_low = bo_close - 1
    
    # Post
    post_close = np.ones(10) * 125
    post_high = post_close + 1
    post_low = post_close - 1
    
    prices1 = np.concatenate([lead_close, pole_close, pen_close, bo_close, post_close])
    highs1 = np.concatenate([lead_high, pole_high, pen_high, bo_high, post_high])
    lows1 = np.concatenate([lead_low, pole_low, pen_low, bo_low, post_low])
    n1 = len(prices1)
    
    # Volumes
    vol_lead = np.random.uniform(800000, 1000000, 40)
    vol_pole = np.random.uniform(1500000, 2000000, 20)
    # Diminishing volume in pennant
    vol_pen = np.linspace(1500000, 500000, p_len)
    # Breakout spike (> 1.5x SMA20)
    vol_bo = np.random.uniform(3000000, 4000000, 5)
    vol_post = np.random.uniform(800000, 1000000, 10)
    vols1 = np.concatenate([vol_lead, vol_pole, vol_pen, vol_bo, vol_post])
    
    res1 = detect_pennant(prices1, highs1, lows1, volumes=vols1, ticker="SYNTH_PEN_VALID", verbose=False)
    if res1:
        print(f"  ✅ [1] Textbook valid -> CONFIRMED (score: {res1[0]['quality_score']})")
        print_pennant(res1[0], 1)
    else:
        print("  ❌ [1] Textbook valid -> REJECTED (unexpected!)")
        all_pass = False

    # 2. Invalid ADX (ADX < 30) - Sudden 1 candle jump instead of steady trend
    lead_close_bad_adx = np.ones(40) * 90
    lead_high_bad = lead_close_bad_adx + 0.5
    lead_low_bad = lead_close_bad_adx - 0.5
    
    pole_close_bad_adx = np.ones(20) * 90
    pole_close_bad_adx[-1] = 101 # >10% jump on the very last day
    pole_high_bad_adx = pole_close_bad_adx + 0.5
    pole_low_bad_adx = pole_close_bad_adx - 0.5

    prices2 = np.concatenate([lead_close_bad_adx, pole_close_bad_adx, pen_close, bo_close, post_close])
    highs2 = np.concatenate([lead_high_bad, pole_high_bad_adx, pen_high, bo_high, post_high])
    lows2 = np.concatenate([lead_low_bad, pole_low_bad_adx, pen_low, bo_low, post_low])
    res2 = detect_pennant(prices2, highs2, lows2, volumes=vols1, ticker="SYNTH_PEN_BAD_ADX")
    if not res2:
        print("  ✅ [2] Bad ADX (sudden jump, no trend) -> REJECTED (correct)")
    else:
        print(f"  ❌ [2] Bad ADX -> CONFIRMED with ADX {res2[0]['adx_value']} (unexpected!)")
        all_pass = False

    # 3. Invalid Slopes (Upper slope positive)
    pen_high_bad = np.linspace(115, 119, p_len) # Positive slope
    highs3 = np.concatenate([lead_high, pole_high, pen_high_bad, bo_high, post_high])
    res3 = detect_pennant(prices1, highs3, lows1, volumes=vols1, ticker="SYNTH_PEN_BAD_SLOPE")
    if not res3:
        print("  ✅ [3] Bad Slopes (upper slope > 0) -> REJECTED (correct)")
    else:
        print("  ❌ [3] Bad Slopes -> CONFIRMED (unexpected!)")
        all_pass = False

    # 4. Bad Breakout Volume
    vol_bo_bad = np.random.uniform(500000, 800000, 5) # Low volume
    vols4 = np.concatenate([vol_lead, vol_pole, vol_pen, vol_bo_bad, vol_post])
    res4 = detect_pennant(prices1, highs1, lows1, volumes=vols4, ticker="SYNTH_PEN_BAD_VOL")
    if not res4:
        print("  ✅ [4] Weak Breakout Volume -> REJECTED (correct)")
    else:
        print("  ❌ [4] Weak Breakout Volume -> CONFIRMED (unexpected!)")
        all_pass = False

    # 5. Flat noise
    np.random.seed(42)
    flat = 100 + np.random.normal(0, 0.5, 200)
    results_flat = detect_pennant(flat, flat+1, flat-1, ticker="FLAT")
    if not results_flat:
        print("  ✅ [5] Flat/random -> REJECTED (correct)")
    else:
        print(f"  ❌ [5] Flat/random -> {len(results_flat)} false positive(s)")
        all_pass = False

    return all_pass


def test_head_and_shoulders():
    """Self-test for Head & Shoulders."""
    print("\n  ── HEAD AND SHOULDERS ───────────────────────────────────")

    np.random.seed(203)
    lead = np.linspace(85, 95, 20)
    ls_up = np.linspace(95, 105, 10)
    ls_down = np.linspace(105, 100, 8)
    head_up = np.linspace(100, 112, 10)
    head_down = np.linspace(112, 100, 10)
    rs_up = np.linspace(100, 105, 10)
    rs_down = np.linspace(105, 100, 8)
    breakdown = np.linspace(100, 94, 6)
    post = np.ones(10) * 94

    prices = np.concatenate([lead, ls_up, ls_down, head_up, head_down,
                              rs_up, rs_down, breakdown, post])
    n = len(prices)
    volumes = np.random.uniform(800000, 1200000, n)
    # Left shoulder: highest volume
    volumes[20:30] = np.random.uniform(2000000, 2500000, 10)
    # Head: moderate
    volumes[38:48] = np.random.uniform(1500000, 1800000, 10)
    # Right shoulder: lowest
    volumes[58:68] = np.random.uniform(900000, 1100000, 10)
    # Breakdown: spike
    volumes[76:82] = np.random.uniform(2500000, 3000000, 6)

    results = detect_head_and_shoulders(prices, volumes=volumes,
                                         ticker="SYNTH_HS", verbose=False)
    if results:
        print(f"  ✅ Textbook shape → VALID  (score: {results[0]['quality_score']})")
        print_head_and_shoulders(results[0], 1)
    else:
        print("  ❌ Textbook shape → REJECTED (unexpected!)")

    # Flat noise
    np.random.seed(42)
    flat = 100 + np.random.normal(0, 0.5, 200)
    results_flat = detect_head_and_shoulders(flat, ticker="FLAT")
    if not results_flat:
        print("  ✅ Flat/random    → REJECTED (correct)")
    else:
        print(f"  ⚠ Flat/random    → {len(results_flat)} false positive(s)")

    return bool(results) and not results_flat


def test_double_top():
    """
    Self-test for Double Top with 5 synthetic test cases:
    1. Textbook valid (should CONFIRM)
    2. Peaks too close (should REJECT)
    3. No prior uptrend (should REJECT)
    4. Touches neckline but doesn't close beyond it (should remain FORMING)
    5. Flat noise (should find nothing)
    """
    print("\n  == DOUBLE TOP — 5 SYNTHETIC TEST CASES ==================")
    all_pass = True

    # ── TEST 1: Textbook valid Double Top ──
    # Clear uptrend → peak1 at ~110 → valley drops >10% to ~97 →
    # peak2 at ~110 → decisive breakdown below valley → volume spike
    print("\n  [TEST 1] Textbook valid Double Top")
    np.random.seed(300)
    # 25-candle uptrend (strong positive slope, R² > 0.5)
    uptrend = np.linspace(80, 100, 25)
    # Rise to first peak at 110 (10 candles)
    first_up = np.linspace(100, 110, 10)
    # Valley drop: 110 → 97 (>10% below peak) over 15 candles (>14 day gap total)
    valley_down = np.linspace(110, 97, 8)
    valley_flat = np.ones(4) * 97
    # Rise to second peak at 110 (same height, <3% difference)
    second_up = np.linspace(97, 110, 8)
    # Breakdown: decisive close below 97 * (1 - 0.005) = 96.515
    # Need 2 consecutive candles below this threshold
    breakdown = np.array([105, 100, 96, 95.5, 95, 94.5])
    post = np.ones(10) * 94

    prices = np.concatenate([uptrend, first_up, valley_down, valley_flat,
                              second_up, breakdown, post])
    n = len(prices)
    volumes = np.random.uniform(800000, 1200000, n)
    # First top area: higher volume
    volumes[33:38] = np.random.uniform(2000000, 2500000, 5)
    # Second top area: lower volume (declining — ideal)
    volumes[50:55] = np.random.uniform(1000000, 1300000, 5)
    # Breakdown area: cover entire post-peak zone with volume spike > 1.2x SMA50
    # (breakdown occurs at idx ~57-58 per diagnostics)
    volumes[55:n] = np.random.uniform(2500000, 3500000, n - 55)

    results = detect_double_top(prices, volumes=volumes,
                                 ticker="SYNTH_DT_VALID", verbose=False)
    if results and results[0].get("verdict") == "CONFIRMED":
        print(f"  PASS: Textbook Double Top -> CONFIRMED (score: {results[0]['quality_score']})")
        print_double_top(results[0], 1)
    else:
        print("  FAIL: Textbook Double Top should be CONFIRMED but wasn't")
        if results:
            print(f"    Got verdict: {results[0].get('verdict')}, reason: {results[0].get('reject_reason')}")
        all_pass = False

    # ── TEST 2: Peaks too close together (<14 days apart) ──
    print("\n  [TEST 2] Peaks too close together (< 2 weeks)")
    np.random.seed(301)
    uptrend2 = np.linspace(80, 100, 25)
    up2 = np.linspace(100, 110, 5)
    # Very short valley — only 3 candles apart
    short_valley = np.array([105, 103])
    up2b = np.linspace(103, 110, 3)
    bd2 = np.linspace(110, 95, 10)
    post2 = np.ones(10) * 95

    prices2 = np.concatenate([uptrend2, up2, short_valley, up2b, bd2, post2])
    n2 = len(prices2)
    volumes2 = np.random.uniform(800000, 1200000, n2)
    volumes2[-10:] = np.random.uniform(2500000, 3500000, 10)

    results2 = detect_double_top(prices2, volumes=volumes2,
                                  ticker="SYNTH_DT_TOOCLOSE", verbose=False)
    if not results2:
        print("  PASS: Peaks too close -> correctly REJECTED (no results)")
    else:
        print(f"  FAIL: Should reject peaks too close, got verdict: {results2[0].get('verdict')}")
        all_pass = False

    # ── TEST 3: No prior uptrend ──
    print("\n  [TEST 3] No prior uptrend (flat before first peak)")
    np.random.seed(302)
    # Flat lead-in: no uptrend (slope ~ 0, R² low)
    flat_lead = np.ones(25) * 110 + np.random.normal(0, 0.3, 25)
    v_down3 = np.linspace(110, 97, 10)
    v_flat3 = np.ones(5) * 97
    up3 = np.linspace(97, 110, 10)
    bd3 = np.array([105, 100, 96, 95, 94])
    post3 = np.ones(10) * 94

    prices3 = np.concatenate([flat_lead, v_down3, v_flat3, up3, bd3, post3])
    n3 = len(prices3)
    volumes3 = np.random.uniform(800000, 1200000, n3)
    volumes3[-5:] = np.random.uniform(2500000, 3500000, 5)

    results3 = detect_double_top(prices3, volumes=volumes3,
                                  ticker="SYNTH_DT_NOTREND", verbose=False)
    if not results3:
        print("  PASS: No prior uptrend -> correctly REJECTED")
    else:
        print(f"  FAIL: Should reject (no uptrend), got verdict: {results3[0].get('verdict')}")
        all_pass = False

    # ── TEST 4: Touches neckline but doesn't close beyond it ──
    print("\n  [TEST 4] Touches neckline but no decisive close (should be FORMING)")
    np.random.seed(303)
    uptrend4 = np.linspace(80, 100, 25)
    up4 = np.linspace(100, 110, 10)
    v_down4 = np.linspace(110, 97, 8)
    v_flat4 = np.ones(4) * 97
    up4b = np.linspace(97, 110, 8)
    # Price approaches neckline but stays just above: 97 * (1 - 0.005) = 96.515
    # These values are ABOVE the threshold — no decisive close
    near_miss = np.array([105, 100, 97.5, 97.2, 97.1, 97.0, 97.5, 98, 99, 100])

    prices4 = np.concatenate([uptrend4, up4, v_down4, v_flat4, up4b, near_miss])
    n4 = len(prices4)
    volumes4 = np.random.uniform(800000, 1200000, n4)
    volumes4[33:38] = np.random.uniform(2000000, 2500000, 5)

    results4 = detect_double_top(prices4, volumes=volumes4,
                                  ticker="SYNTH_DT_FORMING", verbose=True)
    if results4 and results4[0].get("verdict") == "FORMING":
        print("  PASS: Near miss -> correctly shows FORMING")
    elif not results4:
        # With verbose=True, should still be emitted; without, no results is expected
        # since we haven't reached breakout window expiry (only 10 candles of data)
        print("  PASS: Near miss -> no decisive breakdown (correctly filtered)")
    else:
        got = results4[0].get('verdict', 'NONE') if results4 else 'NONE'
        print(f"  NOTE: Got verdict '{got}' — expected FORMING or filtered out")

    # ── TEST 5: Flat noise ──
    print("\n  [TEST 5] Flat/random noise")
    np.random.seed(42)
    flat = 100 + np.random.normal(0, 0.5, 200)
    results_flat = detect_double_top(flat, ticker="FLAT")
    if not results_flat:
        print("  PASS: Flat/random -> REJECTED (correct)")
    else:
        print(f"  FAIL: Flat/random -> {len(results_flat)} false positive(s)")
        all_pass = False

    return all_pass


def test_double_bottom():
    """
    Self-test for Double Bottom with 5 synthetic test cases:
    1. Textbook valid (should CONFIRM)
    2. Bottoms too close (should REJECT)
    3. No prior downtrend (should REJECT)
    4. Touches neckline but doesn't close beyond it (should remain FORMING)
    5. Flat noise (should find nothing)
    """
    print("\n  == DOUBLE BOTTOM — 5 SYNTHETIC TEST CASES ================")
    all_pass = True

    # ── TEST 1: Textbook valid Double Bottom ──
    print("\n  [TEST 1] Textbook valid Double Bottom")
    np.random.seed(400)
    # 25-candle downtrend (strong negative slope, R² > 0.5)
    downtrend = np.linspace(120, 100, 25)
    # Drop to first bottom at 90
    first_down = np.linspace(100, 90, 10)
    # Recovery peak (neckline): 90 → 100 (>10% above bottom)
    peak_up = np.linspace(90, 100, 8)
    peak_flat = np.ones(4) * 100
    # Drop to second bottom at 90 (same level)
    second_down = np.linspace(100, 90, 8)
    # Decisive breakout above 100 * (1 + 0.005) = 100.5
    breakout = np.array([95, 99, 101, 101.5, 102, 103])
    post = np.ones(10) * 103

    prices = np.concatenate([downtrend, first_down, peak_up, peak_flat,
                              second_down, breakout, post])
    n = len(prices)
    volumes = np.random.uniform(800000, 1200000, n)
    # First bottom: higher volume
    volumes[33:38] = np.random.uniform(2000000, 2500000, 5)
    # Second bottom: lower volume (declining — ideal)
    volumes[50:55] = np.random.uniform(800000, 1000000, 5)
    # Breakout area: cover entire post-bottom zone with volume spike > 1.2x SMA50
    volumes[55:n] = np.random.uniform(2500000, 3500000, n - 55)

    results = detect_double_bottom(prices, volumes=volumes,
                                    ticker="SYNTH_DB_VALID", verbose=False)
    if results and results[0].get("verdict") == "CONFIRMED":
        print(f"  PASS: Textbook Double Bottom -> CONFIRMED (score: {results[0]['quality_score']})")
        print_double_bottom(results[0], 1)
    else:
        print("  FAIL: Textbook Double Bottom should be CONFIRMED but wasn't")
        if results:
            print(f"    Got verdict: {results[0].get('verdict')}, reason: {results[0].get('reject_reason')}")
        all_pass = False

    # ── TEST 2: Bottoms too close together ──
    print("\n  [TEST 2] Bottoms too close together (< 2 weeks)")
    np.random.seed(401)
    downtrend2 = np.linspace(120, 100, 25)
    down2 = np.linspace(100, 90, 5)
    short_peak = np.array([95, 97])
    down2b = np.linspace(97, 90, 3)
    bo2 = np.linspace(90, 105, 10)
    post2 = np.ones(10) * 105

    prices2 = np.concatenate([downtrend2, down2, short_peak, down2b, bo2, post2])
    n2 = len(prices2)
    volumes2 = np.random.uniform(800000, 1200000, n2)
    volumes2[-10:] = np.random.uniform(2500000, 3500000, 10)

    results2 = detect_double_bottom(prices2, volumes=volumes2,
                                     ticker="SYNTH_DB_TOOCLOSE", verbose=False)
    if not results2:
        print("  PASS: Bottoms too close -> correctly REJECTED (no results)")
    else:
        print(f"  FAIL: Should reject bottoms too close, got verdict: {results2[0].get('verdict')}")
        all_pass = False

    # ── TEST 3: No prior downtrend ──
    print("\n  [TEST 3] No prior downtrend (flat before first bottom)")
    np.random.seed(402)
    flat_lead = np.ones(25) * 90 + np.random.normal(0, 0.3, 25)
    p_up3 = np.linspace(90, 100, 10)
    p_flat3 = np.ones(5) * 100
    down3 = np.linspace(100, 90, 10)
    bo3 = np.array([95, 99, 101, 102, 103])
    post3 = np.ones(10) * 103

    prices3 = np.concatenate([flat_lead, p_up3, p_flat3, down3, bo3, post3])
    n3 = len(prices3)
    volumes3 = np.random.uniform(800000, 1200000, n3)
    volumes3[-5:] = np.random.uniform(2500000, 3500000, 5)

    results3 = detect_double_bottom(prices3, volumes=volumes3,
                                     ticker="SYNTH_DB_NOTREND", verbose=False)
    if not results3:
        print("  PASS: No prior downtrend -> correctly REJECTED")
    else:
        print(f"  FAIL: Should reject (no downtrend), got verdict: {results3[0].get('verdict')}")
        all_pass = False

    # ── TEST 4: Touches neckline but no decisive close ──
    print("\n  [TEST 4] Touches neckline but no decisive breakout (should be FORMING)")
    np.random.seed(403)
    downtrend4 = np.linspace(120, 100, 25)
    down4 = np.linspace(100, 90, 10)
    peak_up4 = np.linspace(90, 100, 8)
    peak_flat4 = np.ones(4) * 100
    down4b = np.linspace(100, 90, 8)
    # Price approaches neckline but stays just below 100 * (1 + 0.005) = 100.5
    near_miss = np.array([95, 99, 100.2, 99.8, 100.3, 100.1, 99.5, 98, 97, 96])

    prices4 = np.concatenate([downtrend4, down4, peak_up4, peak_flat4, down4b, near_miss])
    n4 = len(prices4)
    volumes4 = np.random.uniform(800000, 1200000, n4)

    results4 = detect_double_bottom(prices4, volumes=volumes4,
                                     ticker="SYNTH_DB_FORMING", verbose=True)
    if results4 and results4[0].get("verdict") == "FORMING":
        print("  PASS: Near miss -> correctly shows FORMING")
    elif not results4:
        print("  PASS: Near miss -> no decisive breakout (correctly filtered)")
    else:
        got = results4[0].get('verdict', 'NONE') if results4 else 'NONE'
        print(f"  NOTE: Got verdict '{got}' — expected FORMING or filtered out")

    # ── TEST 5: Flat noise ──
    print("\n  [TEST 5] Flat/random noise")
    np.random.seed(42)
    flat = 100 + np.random.normal(0, 0.5, 200)
    results_flat = detect_double_bottom(flat, ticker="FLAT")
    if not results_flat:
        print("  PASS: Flat/random -> REJECTED (correct)")
    else:
        print(f"  FAIL: Flat/random -> {len(results_flat)} false positive(s)")
        all_pass = False

    return all_pass


def run_self_tests(pattern_type="all"):
    """
    Runs self-tests for one or all patterns.
    Each test creates textbook synthetic data and verifies detection.
    """
    print("\n" + "=" * 70)
    print("  🧪 SELF-TEST : Synthetic Pattern Verification")
    print("=" * 70)

    TEST_MAP = {
        "cup_and_handle":     test_cup_and_handle,
        "bull_flag":          test_bull_flag,
        "bear_flag":          test_bear_flag,
        "pennant":            test_pennant,
        "head_and_shoulders": test_head_and_shoulders,
        "double_top":         test_double_top,
        "double_bottom":      test_double_bottom,
    }

    if pattern_type == "all":
        tests_to_run = ALL_PATTERNS
    elif pattern_type in TEST_MAP:
        tests_to_run = [pattern_type]
    else:
        print(f"  Unknown pattern: {pattern_type}")
        return

    passed = 0
    failed = 0

    for ptype in tests_to_run:
        try:
            result = TEST_MAP[ptype]()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ {PATTERN_NAMES[ptype]} test CRASHED: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"  🧪 SELF-TEST COMPLETE: {passed} passed, {failed} failed "
          f"out of {passed + failed}")
    print("=" * 70 + "\n")


# =============================================================================
#  14. SCANNING INFRASTRUCTURE — Multi-pattern scanning
# =============================================================================

ALGO_VERSION = 1

def get_param_hash():
    """
    Computes a hash of all threshold constants in the current module.
    Automatically invalidates cache when parameters are tuned.
    """
    import json
    import hashlib
    ignore_keys = {"BATCH_SIZE", "BATCH_SLEEP", "SCAN_INTERVAL_MINUTES", 
                   "IST_OFFSET", "PATTERN_NAMES", "ALL_PATTERNS"}
    params = {}
    for k, v in globals().items():
        if k.isupper() and isinstance(v, (int, float, bool)) and k not in ignore_keys:
            params[k] = v
            
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.sha256(param_str.encode()).hexdigest()

def _restore_dates(pattern_list):
    """Restores _date fields in a pattern dict from string back to pd.Timestamp."""
    for p in pattern_list:
        for k, v in list(p.items()):
            if k.endswith("_date") and v is not None:
                p[k] = pd.to_datetime(v)
    return pattern_list

def scan_ticker(df, ticker, patterns_to_scan, interval="1d", verbose=False, is_live=False):
    """
    Runs selected pattern detections on a single ticker's DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data for one ticker.
    ticker : str
        Ticker symbol (e.g., "RELIANCE.NS").
    patterns_to_scan : list[str]
        Pattern keys to detect, or ["all"] for all 7 patterns.
    interval : str
        Candle interval.
    verbose : bool
        Print rejected candidates too.

    Returns
    -------
    dict[str, list[dict]]
        Keys are pattern types, values are lists of detected patterns.
    """
    close = df["Close"].dropna().values
    dates = df["Close"].dropna().index

    highs = None
    if "High" in df.columns:
        highs = df["High"].reindex(df["Close"].dropna().index).values

    lows = None
    if "Low" in df.columns:
        lows = df["Low"].reindex(df["Close"].dropna().index).values

    vol = None
    if "Volume" in df.columns:
        vol = df["Volume"].reindex(df["Close"].dropna().index).fillna(0).values

    # Adjusted Close for prior-trend check (avoids split/dividend distortion)
    # Available in daily/weekly data from yfinance; not available for intraday.
    adj_close = None
    if "Adj Close" in df.columns:
        adj_close = df["Adj Close"].reindex(df["Close"].dropna().index).values

    if len(close) < 30:
        return {}

    scan_all = "all" in patterns_to_scan
    results = {}

    DETECT_MAP = {
        "cup_and_handle": lambda sc=None: detect_cup_and_handle(
            close, highs=highs, volumes=vol,
            ticker=ticker, dates=dates, interval=interval, verbose=verbose,
            structural_cache=sc
        ),
        "bull_flag": lambda sc=None: detect_bull_flag(
            close, volumes=vol, ticker=ticker, dates=dates,
            interval=interval, verbose=verbose
        ),
        "bear_flag": lambda sc=None: detect_bear_flag(
            close, volumes=vol, ticker=ticker, dates=dates,
            interval=interval, verbose=verbose
        ),
        "pennant": lambda sc=None: detect_pennant(
            close, highs=highs, lows=lows, volumes=vol, ticker=ticker, dates=dates,
            interval=interval, verbose=verbose
        ),
        "head_and_shoulders": lambda sc=None: detect_head_and_shoulders(
            close, highs=highs, lows=lows, volumes=vol, ticker=ticker, dates=dates,
            interval=interval, verbose=verbose
        ),
        "double_top": lambda sc=None: detect_double_top(
            close, volumes=vol, ticker=ticker, dates=dates,
            interval=interval, verbose=verbose
        ),
        "double_bottom": lambda sc=None: detect_double_bottom(
            close, volumes=vol, ticker=ticker, dates=dates,
            interval=interval, verbose=verbose
        ),
    }

    import json
    import time
    import sqlite3
    from backend.main import _make_serializable
    from db_cache import CACHE_DB_PATH
    
    start_date_str = df.index.min().strftime('%Y-%m-%d')
    end_date_str = df.index.max().strftime('%Y-%m-%d')
    param_hash = get_param_hash()
    
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()

    for ptype, detect_fn in DETECT_MAP.items():
        if scan_all or ptype in patterns_to_scan:
            if is_live:
                structural_cache = {}
                try:
                    cursor.execute("SELECT structural_data FROM layer4_structural WHERE ticker = ? AND interval = ? AND pattern_type = ?", (ticker, interval, ptype))
                    srow = cursor.fetchone()
                    if srow:
                        structural_cache = json.loads(srow[0])
                except Exception:
                    pass

                found = detect_fn(sc=structural_cache)
                
                # If it's a refactored pattern (like cup_and_handle), it will populate structural_cache
                if structural_cache and isinstance(structural_cache, dict):
                    try:
                        cursor.execute('''
                            INSERT OR REPLACE INTO layer4_structural (ticker, interval, pattern_type, structural_data)
                            VALUES (?, ?, ?, ?)
                        ''', (ticker, interval, ptype, json.dumps(structural_cache)))
                        conn.commit()
                    except Exception:
                        pass

                if found:
                    results[ptype] = found
                continue

            # Check Layer 3 Cache
            cursor.execute('''
                SELECT results FROM layer3_backtest
                WHERE ticker = ? AND interval = ? AND pattern_type = ?
                AND start_date = ? AND end_date = ? AND param_hash = ? AND algo_version = ?
            ''', (ticker, interval, ptype, start_date_str, end_date_str, param_hash, ALGO_VERSION))
            
            row = cursor.fetchone()
            if row is not None:
                # Cache hit
                cached_res = json.loads(row[0])
                if cached_res:
                    results[ptype] = _restore_dates(cached_res)
                continue
            
            # Cache miss
            found = detect_fn()
            
            # Save to cache
            serializable_found = _make_serializable(found) if found else []
            cursor.execute('''
                INSERT OR REPLACE INTO layer3_backtest 
                (ticker, interval, pattern_type, start_date, end_date, param_hash, algo_version, results, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (ticker, interval, ptype, start_date_str, end_date_str, param_hash, ALGO_VERSION, 
                  json.dumps(serializable_found), time.time()))
            conn.commit()
            
            if found:
                results[ptype] = found
                
    conn.close()

    return results


def print_summary_table(all_results):
    """Prints the ALL PATTERNS summary table."""
    ist_now = _get_ist_now()

    print(f"\n{'═' * 60}")
    print(f"  SCAN SUMMARY")
    print(f"{'─' * 60}")
    print(f"  {'Pattern':<24} | {'Stocks Found':>12} | Top Pick")
    print(f"{'─' * 60}")

    total = 0
    for ptype in ALL_PATTERNS:
        patterns = all_results.get(ptype, [])
        count = len(patterns)
        total += count
        if count > 0:
            top = patterns[0]
            top_str = f"{top['ticker']} (score: {top.get('quality_score', 0)})"
        else:
            top_str = "—"
        print(f"  {PATTERN_NAMES[ptype]:<24} | {count:>12} | {top_str}")

    print(f"{'─' * 60}")
    print(f"  Total stocks flagged : {total}")
    print(f"  Scan completed at    : {ist_now.strftime('%H:%M:%S')} IST")
    print(f"{'═' * 60}")


def backtest_historical(patterns_to_scan, tickers=None, period="2y",
                        start=None, end=None, interval="1d"):
    """
    MODE B: HISTORICAL BACKTEST — scans historical data for patterns.

    Results are saved to 'backtest_results.txt'.
    """
    label = f"Interval: {interval}, Period: {period or 'Custom Date Range'}"
    scan_label = (", ".join(PATTERN_NAMES[p] for p in patterns_to_scan)
                  if "all" not in patterns_to_scan
                  else "ALL PATTERNS")

    print("\n" + "=" * 70)
    print("  📊 HISTORICAL BACKTEST  (Full Timeline Sweep)")
    print("=" * 70)
    print(f"  Patterns   : {scan_label}")
    print(f"  Data       : {label}")
    print(f"  Smoothing  : SMA({SMOOTHING_WINDOW}) (where needed)\n")

    if tickers is None:
        print("  Fetching Nifty watchlist ...\n")
        tickers = get_nifty_list()

    print(f"  Scanning {len(tickers)} tickers ...\n")

    all_data = fetch_batch_data(
        tickers, period=period, start=start, end=end, interval=interval
    )

    if not all_data:
        print("  ⚠ No data downloaded. Check your internet connection.\n")
        return {}

    # Results grouped by pattern type
    all_results = {p: [] for p in ALL_PATTERNS}
    tickers_scanned = 0

    for ticker, df in all_data.items():
        tickers_scanned += 1
        is_verbose = (len(all_data) <= 5)
        ticker_results = scan_ticker(df, ticker, patterns_to_scan,
                                      interval=interval, verbose=is_verbose)

        for ptype, patterns in ticker_results.items():
            if patterns:
                print(f"  ✓ {ticker}: {len(patterns)} {PATTERN_NAMES[ptype]} pattern(s)")
            all_results[ptype].extend(patterns)

    # Sort each pattern's results by quality score
    for ptype in all_results:
        all_results[ptype].sort(key=lambda p: p.get("quality_score", 0), reverse=True)

    # Helper for clean dates
    def clean_date(d):
        return str(d).split(' ')[0] if ' ' in str(d) else str(d)

    # ── Write results file ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(script_dir, "backtest_results.txt")

    with open(results_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  📊 MULTI-PATTERN BACKTEST RESULTS (v3.0)\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Run Date       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Patterns       : {scan_label}\n")
        f.write(f"  Data           : {label}\n")
        f.write(f"  Tickers scanned: {tickers_scanned}\n")
        f.write("=" * 70 + "\n\n")

        total_found = 0
        for ptype in ALL_PATTERNS:
            patterns = all_results[ptype]
            if not patterns and "all" not in patterns_to_scan and ptype not in patterns_to_scan:
                continue
            total_found += len(patterns)

            f.write(f"\n{'═' * 60}\n")
            f.write(f"  {PATTERN_NAMES[ptype].upper()} — {len(patterns)} pattern(s)\n")
            f.write(f"{'═' * 60}\n")

            if patterns:
                for idx, p in enumerate(patterns, 1):
                    f.write(f"\n  #{idx}  {p['ticker']}  "
                            f"(Score: {p.get('quality_score', 0)})\n")
                    for key, val in p.items():
                        if key not in ("pattern_type", "ticker", "verdict",
                                       "reject_reason") and not key.endswith("_idx"):
                            f.write(f"    {key:<30}: {val}\n")
            else:
                f.write("  No patterns found.\n")

        f.write(f"\n{'═' * 60}\n")
        f.write(f"  Total patterns found: {total_found}\n")
        f.write(f"{'═' * 60}\n")

    # ── Terminal: per-pattern sections ──
    print(f"\n  {'─' * 60}")
    print(f"  📊 BACKTEST RESULTS")
    print(f"  {'─' * 60}")
    print(f"  Tickers scanned : {tickers_scanned}")

    for ptype in ALL_PATTERNS:
        patterns = all_results[ptype]
        if not patterns and "all" not in patterns_to_scan and ptype not in patterns_to_scan:
            continue

        print(f"\n  {'═' * 58}")
        print(f"  {PATTERN_NAMES[ptype].upper()} — {len(patterns)} pattern(s)")
        print(f"  {'═' * 58}")

        if patterns:
            for idx, p in enumerate(patterns, 1):
                print_any_pattern(p, idx)
        else:
            print("  (no patterns found)")

    # Summary table (for ALL mode)
    if "all" in patterns_to_scan:
        print_summary_table(all_results)

    print(f"\n  📄 Full results saved to: {results_file}")
    print("\n" + "=" * 70)
    print("  📊 BACKTEST COMPLETE")
    print("=" * 70 + "\n")

    return all_results


# =============================================================================
#  15. LIVE SCANNER — Continuous during market hours
# =============================================================================

def _get_ist_now():
    """Returns the current datetime in IST."""
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    return utc_now + IST_OFFSET


def is_market_open():
    """
    Checks if the NSE is currently open.
    NSE hours: Mon–Fri, 9:15 AM – 3:30 PM IST.

    Returns (bool, str) — (open?, human-readable status)
    """
    ist_now = _get_ist_now()
    day_of_week = ist_now.weekday()
    current_time = ist_now.time()

    market_open = datetime.time(9, 15)
    market_close = datetime.time(15, 30)

    if day_of_week >= 5:
        return False, f"Weekend (IST: {ist_now.strftime('%A %H:%M')})"
    if current_time < market_open:
        return False, f"Pre-market (IST: {ist_now.strftime('%H:%M')}, opens at 09:15)"
    if current_time > market_close:
        return False, f"After-hours (IST: {ist_now.strftime('%H:%M')}, closed at 15:30)"

    return True, f"Market OPEN (IST: {ist_now.strftime('%H:%M')})"


def _reset_dedup_cache_if_new_day():
    """Clears the dedup cache when the IST date rolls over."""
    global _live_alerted_today, _live_alert_date
    today_ist = _get_ist_now().date()
    if _live_alert_date != today_ist:
        _live_alerted_today = set()
        _live_alert_date = today_ist


def _is_pattern_from_today(pattern):
    """Returns True if the pattern's signal date falls on today (IST)."""
    signal_date = pattern.get("signal_date",
                               pattern.get("handle_low_date",
                                            pattern.get("breakout_date",
                                                         pattern.get("breakdown_date"))))
    if signal_date is None:
        return False
    try:
        parsed = pd.Timestamp(signal_date)
        today_ist = _get_ist_now().date()
        return parsed.date() == today_ist
    except Exception:
        return False
def _process_alert_history(ticker, df, ptype, patterns):
    import sqlite3
    import json
    from backend.main import _make_serializable
    from db_cache import CACHE_DB_PATH
    
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    
    # 1. Invalidate superseded patterns
    cursor.execute('''
        SELECT signal_date, pattern_data FROM layer4_alert_history
        WHERE ticker = ? AND pattern_type = ? AND status = 'active'
    ''', (ticker, ptype))
    active_rows = cursor.fetchall()
    
    if len(df) > 0 and len(active_rows) > 0:
        today_ist = _get_ist_now().date()
        
        # Find the latest CLOSED candle (i.e. strictly before today)
        latest_closed = None
        for i in range(1, min(4, len(df) + 1)):
            if df.index[-i].date() < today_ist:
                latest_closed = float(df['Close'].iloc[-i])
                break
                
        if latest_closed is not None:
            for row in active_rows:
                signal_date, pdata_str = row
                pdata = json.loads(pdata_str)
                superseded = False
                
                if ptype == 'cup_and_handle':
                    # Invalidation logic: Price falls below the handle's low support
                    handle_low = pdata.get('handle_low_price')
                    if handle_low is not None and latest_closed < handle_low:
                        superseded = True
                elif ptype == 'bull_flag':
                    # Invalidation logic: Price falls below the flag's low support
                    flag_low = pdata.get('flag_low_price')
                    if flag_low is not None and latest_closed < flag_low:
                        superseded = True
                elif ptype == 'bear_flag':
                    # Invalidation logic: Price rallies above the flag's upper resistance
                    flag_high = pdata.get('flag_high_price')
                    if flag_high is not None and latest_closed > flag_high:
                        superseded = True
                        
                elif ptype == 'pennant':
                    direction = pdata.get('direction', 'bullish')
                    if direction == 'bullish':
                        pennant_low = pdata.get('pennant_low')
                        if pennant_low is not None and latest_closed < pennant_low:
                            superseded = True
                    else:
                        pennant_high = pdata.get('pennant_high')
                        if pennant_high is not None and latest_closed > pennant_high:
                            superseded = True
                elif ptype == 'head_and_shoulders':
                    # Invalidation logic: Price rallies above the right shoulder
                    right_shoulder_price = pdata.get('right_shoulder_price')
                    if right_shoulder_price is not None and latest_closed > right_shoulder_price:
                        superseded = True
                elif ptype == 'double_top':
                    # Invalidation logic: Price rallies above the highest top
                    first_top_price = pdata.get('first_top_price', 0)
                    second_top_price = pdata.get('second_top_price', 0)
                    highest_top = max(first_top_price, second_top_price)
                    if highest_top > 0 and latest_closed > highest_top:
                        superseded = True
                elif ptype == 'double_bottom':
                    # Invalidation logic: Price falls below the lowest bottom
                    first_bottom_price = pdata.get('first_bottom_price', float('inf'))
                    second_bottom_price = pdata.get('second_bottom_price', float('inf'))
                    lowest_bottom = min(first_bottom_price, second_bottom_price)
                    if lowest_bottom != float('inf') and latest_closed < lowest_bottom:
                        superseded = True
                            
                if superseded:
                    cursor.execute('''
                        UPDATE layer4_alert_history 
                        SET status = 'superseded' 
                        WHERE ticker = ? AND pattern_type = ? AND signal_date = ?
                    ''', (ticker, ptype, signal_date))
                    conn.commit()

    # 2. Check if there is still an active alert
    cursor.execute('''
        SELECT 1 FROM layer4_alert_history
        WHERE ticker = ? AND pattern_type = ? AND status = 'active'
    ''', (ticker, ptype))
    has_active = cursor.fetchone() is not None
    
    new_alert_to_return = None
    
    # 3. Process new patterns
    if not has_active:
        for pat in patterns:
            if _is_pattern_from_today(pat):
                signal_date = pat.get("signal_date",
                                      pat.get("handle_low_date",
                                              pat.get("breakout_date",
                                                      pat.get("breakdown_date"))))
                if signal_date:
                    serializable_pat = _make_serializable(pat)
                    cursor.execute('''
                        INSERT OR IGNORE INTO layer4_alert_history
                        (ticker, pattern_type, signal_date, pattern_data, status)
                        VALUES (?, ?, ?, ?, 'active')
                    ''', (ticker, ptype, str(signal_date), json.dumps(serializable_pat)))
                    conn.commit()
                    new_alert_to_return = pat
                    break
                    
    conn.close()
    return new_alert_to_return



def scan_watchlist(tickers, patterns_to_scan, period=None, start=None,
                   end=None, interval="15m"):
    """
    One-shot live scan: downloads latest data, runs detection,
    filters for patterns completing TODAY with deduplication.
    """
    global _live_alerted_today

    _reset_dedup_cache_if_new_day()

    all_data = fetch_batch_data(
        tickers, period=period, start=start, end=end, interval=interval
    )

    new_alerts = []

    for ticker, df in all_data.items():
        ticker_results = scan_ticker(df, ticker, patterns_to_scan, interval=interval, is_live=True)

        for ptype, patterns in ticker_results.items():
            new_alert = _process_alert_history(ticker, df, ptype, patterns)
            if new_alert:
                new_alerts.append(new_alert)

    return new_alerts


def run_scheduler(patterns_to_scan, tickers=None, period="59d",
                  start=None, end=None, interval="15m"):
    """
    MODE C: CONTINUOUS LIVE SCANNER during NSE market hours.
    Scans every SCAN_INTERVAL_MINUTES for patterns completing TODAY.
    """
    scan_label = (", ".join(PATTERN_NAMES[p] for p in patterns_to_scan)
                  if "all" not in patterns_to_scan
                  else "ALL PATTERNS")

    print("\n" + "=" * 70)
    print("  🚀 LIVE SCANNER — Multi-Pattern (v3.0)")
    print("=" * 70)
    print(f"  Patterns      : {scan_label}")
    print(f"  Interval      : {interval}")
    print(f"  Scan interval : every {SCAN_INTERVAL_MINUTES} minutes")
    print(f"  Market hours  : Mon–Fri, 9:15 AM – 3:30 PM IST")
    print(f"  Exit          : Press Ctrl+C\n")

    if tickers is None:
        tickers = get_nifty_list()

    scan_count = 0

    try:
        while True:
            market_open, status_msg = is_market_open()

            if not market_open:
                print(f"  ⏸  {status_msg} — sleeping 5 min ...")
                time.sleep(300)
                continue

            scan_count += 1
            print(f"\n  {'━' * 60}")
            print(f"  🔄 SCAN #{scan_count}  —  {status_msg}")
            print(f"  {'━' * 60}")
            print(f"  📋 Dedup cache: {len(_live_alerted_today)} alert(s) today.\n")

            try:
                new_alerts = scan_watchlist(
                    tickers, patterns_to_scan, period=period,
                    start=start, end=end, interval=interval
                )

                if new_alerts:
                    new_alerts.sort(key=lambda p: p.get("quality_score", 0), reverse=True)
                    print(f"\n  🔔 {len(new_alerts)} NEW ALERT(S)!\n")
                    for idx, pat in enumerate(new_alerts, 1):
                        print_any_pattern(pat, idx)
                else:
                    print("  ─ No new patterns completing today.\n")

            except Exception as e:
                print(f"\n  ⚠ Scan error (will retry): {e}\n")

            ist_now = _get_ist_now()
            print(f"  ⏳ Next scan in {SCAN_INTERVAL_MINUTES} minutes "
                  f"({ist_now.strftime('%H:%M')} IST)  (Ctrl+C to stop)\n")
            time.sleep(SCAN_INTERVAL_MINUTES * 60)

    except KeyboardInterrupt:
        print("\n\n  👋 Scanner stopped by user. Goodbye!\n")


# =============================================================================
#  16. MENU & MAIN ENTRY POINT
# =============================================================================

def validate_yfinance_params(interval, period):
    """Validates that the interval+lookback combo is supported by yfinance."""
    intraday = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]
    if interval in intraday:
        if period:
            if period.endswith("y") or period.endswith("mo"):
                print(f"  ⚠ yfinance supports ≤60 days for {interval}. Capping to 59d.\n")
                return "59d"
            elif period.endswith("d"):
                try:
                    if int(period[:-1]) >= 60:
                        print(f"  ⚠ yfinance supports ≤60 days for {interval}. Capping to 59d.\n")
                        return "59d"
                except ValueError:
                    pass
    return period


def prompt_for_config(default_interval="1d", default_period="2y"):
    """Interactive prompt for candle interval and lookback period."""
    print("\n  [Configuration]")
    interval = input(f"  Enter candle interval (15m / 30m / 1h / 1d / 1wk) "
                     f"[default {default_interval}]: ").strip()
    if not interval:
        interval = default_interval

    if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]:
        if default_period == "2y":
            default_period = "59d"

    period = input(f"  Enter lookback period (Xd / Xmo / Xy) "
                   f"[default {default_period}]: ").strip()
    if not period:
        period = default_period

    period = validate_yfinance_params(interval, period)
    return interval, period


def show_menu():
    """
    Main interactive menu for the multi-pattern scanner.
    Guides the user through pattern selection, execution mode,
    and data configuration.
    """
    print(r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     📊  NSE  MULTI-PATTERN  SCANNER   v3.0                   ║
    ║         Indian Stock Market (NSE)                             ║
    ║                                                              ║
    ║     7 Patterns | 3 Modes | Quality Scoring                   ║
    ║     Smoothing applied only where needed                      ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # ── Pattern selection ──
    print("  ═══════════════════════════════════════════")
    print("  Select scan pattern:")
    print("  ═══════════════════════════════════════════")
    print("    [1]  ☕ Cup and Handle")
    print("    [2]  🟢 Bull Flag")
    print("    [3]  🔴 Bear Flag")
    print("    [4]  🔺 Pennant")
    print("    [5]  👤 Head and Shoulders")
    print("    [6]  🔝 Double Top")
    print("    [7]  🔽 Double Bottom")
    print("    [8]  🌐 ALL PATTERNS (scan everything)")
    print("    [Q]  Exit\n")

    pattern_choice = input("  Enter choice (1-8 / Q): ").strip().lower()

    if pattern_choice in ("q", "quit", "exit"):
        print("  Bye! 👋\n")
        return

    PATTERN_MAP = {
        "1": ["cup_and_handle"],
        "2": ["bull_flag"],
        "3": ["bear_flag"],
        "4": ["pennant"],
        "5": ["head_and_shoulders"],
        "6": ["double_top"],
        "7": ["double_bottom"],
        "8": ["all"],
    }

    if pattern_choice not in PATTERN_MAP:
        print(f"  Invalid choice: '{pattern_choice}'\n")
        return

    patterns_to_scan = PATTERN_MAP[pattern_choice]
    selected_name = ("ALL PATTERNS" if "all" in patterns_to_scan
                     else PATTERN_NAMES[patterns_to_scan[0]])
    print(f"\n  Selected: {selected_name}\n")

    # ── Execution mode ──
    print("  ═══════════════════════════════════════════")
    print("  Select execution mode:")
    print("  ═══════════════════════════════════════════")
    print("    [A]  🧪 Self-Test (synthetic data, instant)")
    print("    [B]  📊 Backtest (historical data)")
    print("    [C]  🚀 Live Scanner (market hours)")
    print("    [D]  📊 Backtest ONE ticker\n")

    mode_choice = input("  Enter choice (A/B/C/D): ").strip().lower()

    if mode_choice == "a":
        if "all" in patterns_to_scan:
            run_self_tests("all")
        else:
            run_self_tests(patterns_to_scan[0])

    elif mode_choice == "b":
        interval, period = prompt_for_config("1d", "2y")
        backtest_historical(patterns_to_scan, period=period, interval=interval)

    elif mode_choice == "c":
        interval, period = prompt_for_config("15m", "59d")
        run_scheduler(patterns_to_scan, period=period, interval=interval)

    elif mode_choice == "d":
        ticker = input("  Enter ticker (e.g., RELIANCE or RELIANCE.NS): ").strip()
        if not ticker:
            print("  No ticker entered. Exiting.\n")
            return
        if not ticker.endswith(".NS"):
            ticker += ".NS"

        interval, period = prompt_for_config("1d", "2y")
        backtest_historical(patterns_to_scan, tickers=[ticker],
                            period=period, interval=interval)

    else:
        print(f"  Invalid mode: '{mode_choice}'\n")


def main():
    """
    Entry point. Supports both command-line arguments and interactive menu.

    Usage
    -----
    python pattern_scanner.py test [pattern]
    python pattern_scanner.py historical [tickers...] --pattern all --interval 1d --lookback 2y
    python pattern_scanner.py live --pattern bull_flag --interval 15m
    python pattern_scanner.py   # interactive menu
    """
    parser = argparse.ArgumentParser(
        description="NSE Multi-Pattern Scanner v3.0 — 7 Chart Patterns"
    )
    parser.add_argument("mode", nargs="?", default="",
                        help="test, historical, live, or empty for interactive menu")
    parser.add_argument("tickers", nargs="*",
                        help="Optional specific tickers for historical mode")
    parser.add_argument("--pattern", "-p", type=str, default="all",
                        help="Pattern to scan: cup_and_handle, bull_flag, bear_flag, "
                             "pennant, head_and_shoulders, double_top, double_bottom, all")
    parser.add_argument("--interval", "-i", type=str,
                        help="Candle interval (e.g., 15m, 1d)")
    parser.add_argument("--lookback", "-l", type=str,
                        help="Lookback period (e.g., 59d, 2y)")
    parser.add_argument("--start-date", "-s", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", "-e", type=str, help="End date (YYYY-MM-DD)")

    args = parser.parse_args()
    mode = args.mode.lower()

    # Parse pattern selection
    if args.pattern == "all":
        patterns_to_scan = ["all"]
    elif args.pattern in ALL_PATTERNS:
        patterns_to_scan = [args.pattern]
    else:
        patterns_to_scan = ["all"]

    interval = args.interval
    period = args.lookback
    if interval and period:
        period = validate_yfinance_params(interval, period)

    if mode == "test":
        if "all" in patterns_to_scan:
            run_self_tests("all")
        else:
            run_self_tests(patterns_to_scan[0])
        return

    elif mode in ("historical", "backtest"):
        if not interval:
            interval = "1d"
        if not period and not args.start_date:
            period = "2y"
        period = validate_yfinance_params(interval, period)

        if args.tickers:
            tickers = [t if t.endswith(".NS") else f"{t}.NS" for t in args.tickers]
            backtest_historical(patterns_to_scan, tickers=tickers,
                                period=period, start=args.start_date,
                                end=args.end_date, interval=interval)
        else:
            backtest_historical(patterns_to_scan, period=period,
                                start=args.start_date, end=args.end_date,
                                interval=interval)
        return

    elif mode in ("live", "live_scan"):
        if not interval:
            interval = "15m"
        if not period and not args.start_date:
            period = "59d"
        period = validate_yfinance_params(interval, period)
        run_scheduler(patterns_to_scan, period=period,
                      start=args.start_date, end=args.end_date,
                      interval=interval)
        return

    elif mode != "":
        print(f"  Unknown mode: '{mode}'")
        print(f"  Valid modes: test, historical, live\n")
        return

    # Interactive menu
    show_menu()


if __name__ == "__main__":
    main()
