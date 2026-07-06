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
MIN_PENNANT_CANDLES = 6        # Minimum candles in the pennant body

# ─── HEAD AND SHOULDERS ──────────────────────────────────────────────────
HEAD_SHOULDER_RATIO = 3        # Head must be ≥ this % higher than each shoulder
SHOULDER_SYMMETRY_PCT = 5      # Shoulders must be within this % of each other
MIN_HS_CANDLES = 20            # Minimum span (left shoulder → right shoulder)
NECKLINE_SLOPE_WARN_PCT = 5    # Warn if neckline slopes more than this %

# ─── DOUBLE TOP / DOUBLE BOTTOM ──────────────────────────────────────────
DOUBLE_TOP_SIMILARITY_PCT = 3
DOUBLE_BOTTOM_SIMILARITY_PCT = 3
MIN_VALLEY_DROP_PCT = 3        # Valley must be ≥ this % below tops
MIN_PEAK_RISE_PCT = 3          # Peak must be ≥ this % above bottoms
MIN_DOUBLE_CANDLES = 10        # Minimum separation between the two tops/bottoms

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
    Downloads the Nifty 200 stock list from a public CSV hosted by NSE India.

    Returns
    -------
    list[str]
        Ticker symbols with ".NS" suffix (e.g., ["RELIANCE.NS", "TCS.NS"]).
        Falls back to a curated Nifty 50 list if the download fails.
    """
    csv_urls = [
        "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
        "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
        "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    ]

    for url in csv_urls:
        try:
            print(f"  ↳ Trying to fetch watchlist from:\n    {url}")
            df = pd.read_csv(url)
            if "Symbol" in df.columns:
                symbols = [f"{sym.strip()}.NS" for sym in df["Symbol"].tolist()]
                print(f"  ✓ Loaded {len(symbols)} tickers from NSE index CSV.\n")
                return symbols
            else:
                print(f"  ✗ CSV downloaded but 'Symbol' column not found.")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    # Fallback — hardcoded Nifty 50
    print("  ⚠ All CSV sources failed. Using hardcoded Nifty 50 fallback.\n")
    fallback = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
        "LT", "AXISBANK", "ASIANPAINT", "HCLTECH", "MARUTI",
        "SUNPHARMA", "TITAN", "BAJFINANCE", "WIPRO", "ULTRACEMCO",
        "NESTLEIND", "ONGC", "NTPC", "POWERGRID", "M&M",
        "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "ADANIENT", "ADANIPORTS",
        "BAJAJFINSV", "TECHM", "HDFCLIFE", "SBILIFE", "DIVISLAB",
        "DRREDDY", "CIPLA", "APOLLOHOSP", "EICHERMOT", "HEROMOTOCO",
        "COALINDIA", "BPCL", "GRASIM", "INDUSINDBK", "BRITANNIA",
        "TATACONSUM", "HINDALCO", "UPL", "BAJAJ-AUTO", "LTIM",
    ]
    return [f"{s}.NS" for s in fallback]


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
                "threads": True
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
        # Shoulder symmetry (0–2): 0% diff → 2, 5% → 0
        sym = metrics.get("shoulder_symmetry_pct", 5)
        sym_score = max(0, 2.0 * (1 - sym / 5))
        # Neckline flatness (0–2): 0% slope → 2, 5% → 0
        neck_slope = abs(metrics.get("neckline_slope_pct", 0))
        neck_score = max(0, 2.0 * (1 - neck_slope / 5))
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

    # ── Step 1: Smooth prices and find peaks/troughs ──
    smoothed = smooth_prices(prices, window=SMOOTHING_WINDOW)

    min_prominence = np.median(smoothed) * 0.01
    peak_indices, _ = find_peaks(smoothed, distance=10, prominence=min_prominence)
    trough_indices, _ = find_peaks(-smoothed, distance=10, prominence=min_prominence)

    if len(peak_indices) < 2 or len(trough_indices) < 1:
        return []

    candidates = []

    # ── Step 2: Try every combination of (Left Rim, Right Rim) ──
    for i in range(len(peak_indices)):
        for j in range(i + 1, len(peak_indices)):
            left_rim_idx = peak_indices[i]
            right_rim_idx = peak_indices[j]

            left_rim_price = prices[left_rim_idx]
            right_rim_price = prices[right_rim_idx]

            if right_rim_price == 0:
                continue

            # ── Rule 1: Dynamic Trend Check ──
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

            # ── Width check ──
            cup_width = right_rim_idx - left_rim_idx
            if cup_width < 15 or cup_width > len(prices) * 0.8:
                continue

            # ── Find deepest trough between rims ──
            troughs_in_cup = [t for t in trough_indices
                              if left_rim_idx < t < right_rim_idx]
            if not troughs_in_cup:
                continue

            raw_prices_in_cup = prices[left_rim_idx:right_rim_idx]
            cup_bottom_idx = left_rim_idx + int(np.argmin(raw_prices_in_cup))
            cup_bottom_price = prices[cup_bottom_idx]

            # ── Minimum Cup Duration ──
            cup_left_duration = cup_bottom_idx - left_rim_idx
            cup_right_duration = right_rim_idx - cup_bottom_idx
            if cup_left_duration < MIN_CUP_CANDLES or cup_right_duration < MIN_CUP_CANDLES:
                continue

            # ── No Double-Dip ──
            prices_after_bottom = prices[cup_bottom_idx:right_rim_idx + 1]
            if len(prices_after_bottom) > 0 and np.min(prices_after_bottom) < cup_bottom_price:
                continue

            # ── Cup Symmetry ──
            left_margin = left_rim_idx + cup_width * 0.15
            right_margin = right_rim_idx - cup_width * 0.15
            if not (left_margin <= cup_bottom_idx <= right_margin):
                continue

            # ── Cup depth check ──
            cup_drop_pct = ((left_rim_price - cup_bottom_price)
                            / left_rim_price) * 100
            if not (MIN_CUP_DROP_PCT <= cup_drop_pct <= MAX_CUP_DROP_PCT):
                continue

            # ── Recovery check ──
            recovery_pct = abs(right_rim_price - left_rim_price) / left_rim_price * 100
            if recovery_pct > MAX_RECOVERY_GAP_PCT:
                continue

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
                     interval="1d", verbose=False):
    """
    Detects Bull Flag patterns: a strong upward pole → tight downward/flat
    consolidation (flag) → breakout above the pole top.

    Smoothing: NO — uses linregress directly on raw prices.

    Theory
    ------
    A bull flag is a continuation pattern. After a sharp rally (the "pole"),
    the stock pauses in a tight, slightly downward-drifting channel (the
    "flag"). Sellers can't push prices down much, and when buyers return,
    the stock breaks out above the pole top and continues rising.
    """
    prices = np.array(prices, dtype=float)
    if volumes is not None:
        volumes = np.array(volumes, dtype=float)

    n = len(prices)
    if n < 30:
        return []

    # Pre-filter: find candidate pole-end positions using rolling min
    # This avoids testing every single index as a pole end.
    rolling_min = pd.Series(prices).rolling(window=MAX_POLE_CANDLES, min_periods=5).min().values
    potential_return = np.zeros(n)
    for idx in range(MAX_POLE_CANDLES, n):
        if rolling_min[idx] > 0:
            potential_return[idx] = (prices[idx] - rolling_min[idx]) / rolling_min[idx] * 100
    potential_pole_ends = np.where(potential_return >= MIN_POLE_RISE_PCT)[0]

    candidates = []

    for pole_end_idx in potential_pole_ends:
        if pole_end_idx >= n - MIN_FLAG_CANDLES - 1:
            continue

        # Find the best pole for this pole_end position
        best_pole = None
        for pole_len in range(5, min(MAX_POLE_CANDLES + 1, pole_end_idx + 1)):
            pole_start_idx = pole_end_idx - pole_len
            if pole_start_idx < 0:
                continue

            rise_pct = (prices[pole_end_idx] - prices[pole_start_idx]) / prices[pole_start_idx] * 100
            if rise_pct < MIN_POLE_RISE_PCT:
                continue

            # Rec A: reject slow-grind poles — must be a sharp thrust
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

        # Look for flag after pole
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

            # Fix 1: reject if any flag bar already broke above the pole top
            if flag_high > pole_top_price * 1.001:   # small tolerance for float noise
                continue

            flag_range_pct = (flag_high - flag_low) / pole_top_price * 100

            if flag_range_pct > FLAG_CHANNEL_PCT:
                continue

            # Fix 2: flag must not retrace more than MAX_RETRACEMENT_PCT of the pole
            pole_size = pole_top_price - prices[best_pole["start_idx"]]
            retracement_pct = (pole_top_price - flag_low) / pole_size * 100 if pole_size > 0 else 0
            if retracement_pct > MAX_RETRACEMENT_PCT:
                continue

            # Flag should slope downward or be essentially flat
            x = np.arange(len(flag_prices))
            flag_res = linregress(x, flag_prices)
            flag_total_drift_pct = (flag_res.slope * len(flag_prices)) / pole_top_price * 100
            if flag_total_drift_pct > 0.5:  # allow up to 0.5% total upward drift
                continue

            # Check for breakout: price closes above pole top
            breakout_idx = None
            for k in range(flag_end + 1, min(n, flag_end + 6)):
                if prices[k] > pole_top_price:
                    breakout_idx = k
                    break

            if breakout_idx is None:
                if verbose:
                    print(f"  ❌ {ticker}: Bull Flag pole found (idx {best_pole['start_idx']}-"
                          f"{pole_end_idx}, +{best_pole['rise_pct']:.1f}%) but no breakout.")
                continue

            # Volume checks
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
                    pole_vol = float(np.mean(volumes[best_pole["start_idx"]:pole_end_idx + 1]))
                    vol_pole_pass = pole_vol > vol_sma50

                    flag_vol = float(np.mean(volumes[flag_start:flag_end + 1]))
                    vol_flag_pass = flag_vol < vol_sma50

                    breakout_vol = float(volumes[breakout_idx])
                    vol_breakout_pass = breakout_vol > (BREAKOUT_VOLUME_MULTIPLIER * vol_sma50)

            # Rec D: optional hard gate on breakout volume
            if REQUIRE_BREAKOUT_VOLUME and vol_breakout_pass is False:
                continue

            # Quality score
            quality = compute_quality_score("bull_flag", {
                "pole_change_pct": best_pole["rise_pct"],
                "r_squared": best_pole["r_squared"],
                "flag_range_pct": flag_range_pct,
                "vol_pole_pass": vol_pole_pass,
                "vol_flag_pass": vol_flag_pass,
                "vol_breakout_pass": vol_breakout_pass,
            })

            pattern = {
                "pattern_type":       "bull_flag",
                "ticker":             ticker,
                "verdict":            "VALID",
                "reject_reason":      "",
                "pole_start_price":   round(float(prices[best_pole["start_idx"]]), 2),
                "pole_top_price":     round(float(pole_top_price), 2),
                "flag_low_price":     round(float(flag_low), 2),
                "flag_high_price":    round(float(flag_high), 2),
                "breakout_price":     round(float(prices[breakout_idx]), 2),
                "pole_rise_pct":      round(float(best_pole["rise_pct"]), 2),
                "pole_r_squared":     round(float(best_pole["r_squared"]), 3),
                "flag_range_pct":     round(float(flag_range_pct), 2),
                "flag_slope":         round(float(flag_res.slope), 4),
                "vol_pole_pass":      vol_pole_pass,
                "vol_flag_pass":      vol_flag_pass,
                "vol_breakout_pass":  vol_breakout_pass,
                "quality_score":      quality,
                "pole_start_idx":     best_pole["start_idx"],
                "pole_end_idx":       pole_end_idx,
                "flag_start_idx":     flag_start,
                "flag_end_idx":       flag_end,
                "breakout_idx":       breakout_idx,
                "retracement_pct":    round(float(retracement_pct), 2),
            }

            if dates is not None:
                pattern["pole_start_date"] = str(dates[best_pole["start_idx"]])
                pattern["pole_top_date"]   = str(dates[pole_end_idx])
                pattern["flag_low_date"]   = str(dates[flag_start + int(np.argmin(flag_prices))])
                pattern["breakout_date"]   = str(dates[breakout_idx])
                pattern["signal_date"]     = str(dates[breakout_idx])

            candidates.append(pattern)
            break  # Found valid flag for this pole, move on

    # Deduplication
    candidates.sort(key=lambda p: p["quality_score"], reverse=True)
    patterns_found = []
    claimed = []

    for c in candidates:
        is_dup = False
        for (cs, ce) in claimed:
            if abs(c["pole_start_idx"] - cs) <= 10 and abs(c["breakout_idx"] - ce) <= 10:
                is_dup = True
                break
        if not is_dup:
            patterns_found.append(c)
            claimed.append((c["pole_start_idx"], c["breakout_idx"]))

    return patterns_found


# =============================================================================
#  7. DETECT BEAR FLAG
# =============================================================================

def detect_bear_flag(prices, volumes=None, ticker="UNKNOWN", dates=None,
                     interval="1d", verbose=False):
    """
    Detects Bear Flag patterns: a strong downward pole → tight upward/flat
    consolidation (flag) → breakdown below the pole bottom.

    Smoothing: NO — uses linregress directly on raw prices.

    Theory
    ------
    A bear flag is the mirror image of a bull flag. After a sharp selloff
    (the pole), the stock drifts upward slightly in a tight channel (the
    flag — a weak relief bounce). When sellers return, the stock breaks
    down below the pole bottom and continues falling.
    """
    prices = np.array(prices, dtype=float)
    if volumes is not None:
        volumes = np.array(volumes, dtype=float)

    n = len(prices)
    if n < 30:
        return []

    # Pre-filter: find candidate pole-end positions using rolling max
    rolling_max = pd.Series(prices).rolling(window=MAX_POLE_CANDLES, min_periods=5).max().values
    potential_drop = np.zeros(n)
    for idx in range(MAX_POLE_CANDLES, n):
        if rolling_max[idx] > 0:
            potential_drop[idx] = (rolling_max[idx] - prices[idx]) / rolling_max[idx] * 100
    potential_pole_ends = np.where(potential_drop >= MIN_POLE_DROP_PCT)[0]

    candidates = []

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

            # Rec A: reject slow-grind poles — must be a sharp drop
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

            # Fix 1: reject if any flag bar already broke below the pole bottom
            if flag_low < pole_bottom_price * 0.999:   # small tolerance for float noise
                continue

            flag_range_pct = (flag_high - flag_low) / pole_bottom_price * 100

            if flag_range_pct > FLAG_CHANNEL_PCT:
                continue

            # Fix 2: flag must not retrace more than MAX_RETRACEMENT_PCT of the pole
            pole_size = prices[best_pole["start_idx"]] - pole_bottom_price
            retracement_pct = (flag_high - pole_bottom_price) / pole_size * 100 if pole_size > 0 else 0
            if retracement_pct > MAX_RETRACEMENT_PCT:
                continue

            # Flag should slope upward or be flat (bear flag relief bounce)
            x = np.arange(len(flag_prices))
            flag_res = linregress(x, flag_prices)
            flag_total_drift_pct = (flag_res.slope * len(flag_prices)) / pole_bottom_price * 100
            if flag_total_drift_pct < -0.5:  # too much downward drift
                continue

            # Breakdown: price closes below pole bottom
            breakdown_idx = None
            for k in range(flag_end + 1, min(n, flag_end + 6)):
                if prices[k] < pole_bottom_price:
                    breakdown_idx = k
                    break

            if breakdown_idx is None:
                continue

            # Volume checks
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
                    pole_vol = float(np.mean(volumes[best_pole["start_idx"]:pole_end_idx + 1]))
                    vol_pole_pass = pole_vol > vol_sma50

                    flag_vol = float(np.mean(volumes[flag_start:flag_end + 1]))
                    vol_flag_pass = flag_vol < vol_sma50

                    bd_vol = float(volumes[breakdown_idx])
                    vol_breakdown_pass = bd_vol > (BREAKOUT_VOLUME_MULTIPLIER * vol_sma50)

            # Rec D: optional hard gate on breakdown volume
            if REQUIRE_BREAKOUT_VOLUME and vol_breakdown_pass is False:
                continue

            quality = compute_quality_score("bear_flag", {
                "pole_change_pct": best_pole["drop_pct"],
                "r_squared": best_pole["r_squared"],
                "flag_range_pct": flag_range_pct,
                "vol_pole_pass": vol_pole_pass,
                "vol_flag_pass": vol_flag_pass,
                "vol_breakout_pass": vol_breakdown_pass,
            })

            pattern = {
                "pattern_type":       "bear_flag",
                "ticker":             ticker,
                "verdict":            "VALID",
                "reject_reason":      "",
                "pole_start_price":   round(float(prices[best_pole["start_idx"]]), 2),
                "pole_bottom_price":  round(float(pole_bottom_price), 2),
                "flag_high_price":    round(float(flag_high), 2),
                "flag_low_price":     round(float(flag_low), 2),
                "breakdown_price":    round(float(prices[breakdown_idx]), 2),
                "pole_drop_pct":      round(float(best_pole["drop_pct"]), 2),
                "pole_r_squared":     round(float(best_pole["r_squared"]), 3),
                "flag_range_pct":     round(float(flag_range_pct), 2),
                "flag_slope":         round(float(flag_res.slope), 4),
                "vol_pole_pass":      vol_pole_pass,
                "vol_flag_pass":      vol_flag_pass,
                "vol_breakdown_pass": vol_breakdown_pass,
                "quality_score":      quality,
                "pole_start_idx":     best_pole["start_idx"],
                "pole_end_idx":       pole_end_idx,
                "flag_start_idx":     flag_start,
                "flag_end_idx":       flag_end,
                "breakdown_idx":      breakdown_idx,
                "retracement_pct":    round(float(retracement_pct), 2),
            }

            if dates is not None:
                pattern["pole_start_date"]  = str(dates[best_pole["start_idx"]])
                pattern["pole_bottom_date"] = str(dates[pole_end_idx])
                pattern["flag_high_date"]   = str(dates[flag_start + int(np.argmax(flag_prices))])
                pattern["breakdown_date"]   = str(dates[breakdown_idx])
                pattern["signal_date"]      = str(dates[breakdown_idx])

            candidates.append(pattern)
            break

    # Deduplication
    candidates.sort(key=lambda p: p["quality_score"], reverse=True)
    patterns_found = []
    claimed = []

    for c in candidates:
        is_dup = False
        for (cs, ce) in claimed:
            if abs(c["pole_start_idx"] - cs) <= 10 and abs(c["breakdown_idx"] - ce) <= 10:
                is_dup = True
                break
        if not is_dup:
            patterns_found.append(c)
            claimed.append((c["pole_start_idx"], c["breakdown_idx"]))

    return patterns_found


# =============================================================================
#  8. DETECT PENNANT
# =============================================================================

def detect_pennant(prices, volumes=None, ticker="UNKNOWN", dates=None,
                   interval="1d", verbose=False):
    """
    Detects Pennant patterns: a strong pole (up or down) → converging
    triangle consolidation (higher lows + lower highs) → breakout in
    the pole's direction.

    Smoothing: NO — uses linregress on raw prices for pole, and half-range
    comparison for convergence detection.

    Theory
    ------
    A pennant is similar to a flag, but instead of a channel, the
    consolidation forms a small symmetrical triangle — the highs get
    lower and the lows get higher, squeezing toward a point. The
    breakout comes in the same direction as the original pole.
    """
    prices = np.array(prices, dtype=float)
    if volumes is not None:
        volumes = np.array(volumes, dtype=float)

    n = len(prices)
    if n < 30:
        return []

    candidates = []

    for pole_end_idx in range(MAX_POLE_CANDLES, n - MIN_PENNANT_CANDLES - 1):
        # Try both bullish and bearish poles
        best_pole = None
        pole_direction = None  # "bullish" or "bearish"

        for pole_len in range(5, min(MAX_POLE_CANDLES + 1, pole_end_idx + 1)):
            pole_start_idx = pole_end_idx - pole_len
            if pole_start_idx < 0:
                continue

            change_pct = (prices[pole_end_idx] - prices[pole_start_idx]) / prices[pole_start_idx] * 100

            is_bullish = change_pct >= MIN_POLE_RISE_PCT
            is_bearish = change_pct <= -MIN_POLE_DROP_PCT

            if not is_bullish and not is_bearish:
                continue

            pole_prices = prices[pole_start_idx:pole_end_idx + 1]
            x = np.arange(len(pole_prices))
            res = linregress(x, pole_prices)
            r_squared = res.rvalue ** 2

            if r_squared < POLE_R_SQUARED_MIN:
                continue
            if is_bullish and res.slope <= 0:
                continue
            if is_bearish and res.slope >= 0:
                continue

            abs_change = abs(change_pct)
            if best_pole is None or abs_change > abs(best_pole["change_pct"]):
                best_pole = {
                    "start_idx": pole_start_idx,
                    "end_idx": pole_end_idx,
                    "change_pct": change_pct,
                    "r_squared": r_squared,
                    "slope": res.slope,
                }
                pole_direction = "bullish" if is_bullish else "bearish"

        if best_pole is None:
            continue

        # Look for pennant body after pole
        max_pennant_len = min(MAX_FLAG_CANDLES, n - pole_end_idx - 2)
        if max_pennant_len < MIN_PENNANT_CANDLES:
            continue

        for plen in range(MIN_PENNANT_CANDLES, max_pennant_len + 1):
            pennant_start = pole_end_idx + 1
            pennant_end = pole_end_idx + plen
            if pennant_end >= n:
                break

            pennant_prices = prices[pennant_start:pennant_end + 1]
            if len(pennant_prices) < MIN_PENNANT_CANDLES:
                continue

            # Convergence check using half-range comparison
            half = len(pennant_prices) // 2
            if half < 2:
                continue

            first_half = pennant_prices[:half]
            second_half = pennant_prices[half:]

            first_range = float(np.max(first_half) - np.min(first_half))
            second_range = float(np.max(second_half) - np.min(second_half))

            if first_range <= 0:
                continue

            # Second half range must be smaller (converging)
            if second_range >= first_range:
                continue

            # Highs decreasing, lows increasing
            if np.max(second_half) >= np.max(first_half):
                continue
            if np.min(second_half) <= np.min(first_half):
                continue

            convergence_ratio = second_range / first_range

            # Compute trendline slopes for debug output
            upper_slope = (float(np.max(second_half)) - float(np.max(first_half))) / half
            lower_slope = (float(np.min(second_half)) - float(np.min(first_half))) / half

            # Check breakout in pole direction
            breakout_idx = None
            for k in range(pennant_end + 1, min(n, pennant_end + 6)):
                if pole_direction == "bullish" and prices[k] > np.max(pennant_prices):
                    breakout_idx = k
                    break
                elif pole_direction == "bearish" and prices[k] < np.min(pennant_prices):
                    breakout_idx = k
                    break

            if breakout_idx is None:
                continue

            # Volume checks
            vol_sma50 = 0
            vol_pole_pass = None
            vol_pennant_pass = None
            vol_breakout_pass = None

            if volumes is not None and len(volumes) == n:
                vol_series = pd.Series(volumes)
                vol_sma = vol_series.rolling(window=VOLUME_SMA_PERIOD, min_periods=10).mean()
                sma_idx = min(pole_end_idx, len(vol_sma) - 1)
                vol_sma50 = float(vol_sma.iloc[sma_idx]) if not pd.isna(vol_sma.iloc[sma_idx]) else 0

                if vol_sma50 > 0:
                    pole_vol = float(np.mean(volumes[best_pole["start_idx"]:pole_end_idx + 1]))
                    vol_pole_pass = pole_vol > vol_sma50

                    pennant_vol = float(np.mean(volumes[pennant_start:pennant_end + 1]))
                    vol_pennant_pass = pennant_vol < vol_sma50

                    bo_vol = float(volumes[breakout_idx])
                    vol_breakout_pass = bo_vol > (BREAKOUT_VOLUME_MULTIPLIER * vol_sma50)

            quality = compute_quality_score("pennant", {
                "pole_change_pct": best_pole["change_pct"],
                "convergence_ratio": convergence_ratio,
                "r_squared": best_pole["r_squared"],
                "vol_pole_pass": vol_pole_pass,
                "vol_pennant_pass": vol_pennant_pass,
                "vol_breakout_pass": vol_breakout_pass,
            })

            pattern = {
                "pattern_type":       "pennant",
                "ticker":             ticker,
                "verdict":            "VALID",
                "reject_reason":      "",
                "direction":          pole_direction,
                "pole_start_price":   round(float(prices[best_pole["start_idx"]]), 2),
                "pole_end_price":     round(float(prices[pole_end_idx]), 2),
                "pennant_high":       round(float(np.max(pennant_prices)), 2),
                "pennant_low":        round(float(np.min(pennant_prices)), 2),
                "breakout_price":     round(float(prices[breakout_idx]), 2),
                "pole_change_pct":    round(float(best_pole["change_pct"]), 2),
                "pole_r_squared":     round(float(best_pole["r_squared"]), 3),
                "upper_slope":        round(float(upper_slope), 4),
                "lower_slope":        round(float(lower_slope), 4),
                "convergence_ratio":  round(float(convergence_ratio), 3),
                "vol_pole_pass":      vol_pole_pass,
                "vol_pennant_pass":   vol_pennant_pass,
                "vol_breakout_pass":  vol_breakout_pass,
                "quality_score":      quality,
                "pole_start_idx":     best_pole["start_idx"],
                "pole_end_idx":       pole_end_idx,
                "pennant_start_idx":  pennant_start,
                "pennant_end_idx":    pennant_end,
                "breakout_idx":       breakout_idx,
            }

            if dates is not None:
                pattern["pole_start_date"]    = str(dates[best_pole["start_idx"]])
                pattern["pole_end_date"]      = str(dates[pole_end_idx])
                pattern["pennant_start_date"] = str(dates[pennant_start])
                pattern["pennant_end_date"]   = str(dates[pennant_end])
                pattern["breakout_date"]      = str(dates[breakout_idx])
                pattern["signal_date"]        = str(dates[breakout_idx])

            candidates.append(pattern)
            break

    # Deduplication
    candidates.sort(key=lambda p: p["quality_score"], reverse=True)
    patterns_found = []
    claimed = []

    for c in candidates:
        is_dup = False
        for (cs, ce) in claimed:
            if abs(c["pole_start_idx"] - cs) <= 10 and abs(c["breakout_idx"] - ce) <= 10:
                is_dup = True
                break
        if not is_dup:
            patterns_found.append(c)
            claimed.append((c["pole_start_idx"], c["breakout_idx"]))

    return patterns_found


# =============================================================================
#  9. DETECT HEAD AND SHOULDERS
# =============================================================================

def detect_head_and_shoulders(prices, volumes=None, ticker="UNKNOWN",
                               dates=None, interval="1d", verbose=False):
    """
    Detects Head and Shoulders reversal patterns: three successive peaks
    where the middle (head) is higher than the two outer ones (shoulders),
    with a neckline connecting the two troughs. Breakdown below the
    neckline signals a trend reversal from uptrend to downtrend.

    Smoothing: YES — uses smooth_prices() + find_peaks() to detect peaks
    and troughs. All reported prices are RAW.

    Theory
    ------
    The H&S pattern forms at the TOP of an uptrend:
    - Left Shoulder: buyers push price to a high, then it pulls back
    - Head: buyers push even higher, but it pulls back again
    - Right Shoulder: buyers try again but can't reach the head's height
    The declining buying power (shoulder → head → shoulder with weakening
    volume) shows distribution. When price breaks below the neckline
    connecting the two pullback lows, sellers take control.
    """
    prices = np.array(prices, dtype=float)
    if volumes is not None:
        volumes = np.array(volumes, dtype=float)

    n = len(prices)
    if n < 40:
        return []

    # ── Smooth and find peaks/troughs ──
    smoothed = smooth_prices(prices, window=SMOOTHING_WINDOW)
    min_prominence = np.median(smoothed) * 0.01

    peak_indices, _ = find_peaks(smoothed, distance=8, prominence=min_prominence)
    trough_indices, _ = find_peaks(-smoothed, distance=8, prominence=min_prominence)

    if len(peak_indices) < 3 or len(trough_indices) < 2:
        return []

    candidates = []

    # Try every triplet of consecutive peaks
    for i in range(len(peak_indices) - 2):
        ls_idx = peak_indices[i]       # Left Shoulder
        head_idx = peak_indices[i + 1]  # Head
        rs_idx = peak_indices[i + 2]    # Right Shoulder

        ls_price = prices[ls_idx]
        head_price = prices[head_idx]
        rs_price = prices[rs_idx]

        # Duration check
        span = rs_idx - ls_idx
        if span < MIN_HS_CANDLES:
            continue

        # Head must be higher than both shoulders
        ls_prominence = (head_price - ls_price) / ls_price * 100
        rs_prominence = (head_price - rs_price) / rs_price * 100

        if ls_prominence < HEAD_SHOULDER_RATIO or rs_prominence < HEAD_SHOULDER_RATIO:
            continue

        # Shoulder symmetry
        shoulder_diff_pct = abs(ls_price - rs_price) / ls_price * 100
        if shoulder_diff_pct > SHOULDER_SYMMETRY_PCT:
            continue

        # Find neckline troughs
        left_troughs = [t for t in trough_indices if ls_idx < t < head_idx]
        right_troughs = [t for t in trough_indices if head_idx < t < rs_idx]

        if not left_troughs or not right_troughs:
            continue

        # Use deepest trough in each gap
        left_neck_idx = min(left_troughs, key=lambda t: prices[t])
        right_neck_idx = min(right_troughs, key=lambda t: prices[t])
        left_neck_price = prices[left_neck_idx]
        right_neck_price = prices[right_neck_idx]

        # Neckline slope
        neck_span = right_neck_idx - left_neck_idx
        if neck_span <= 0:
            continue
        neckline_slope = (right_neck_price - left_neck_price) / neck_span
        neckline_slope_pct = abs(right_neck_price - left_neck_price) / left_neck_price * 100

        # Breakdown check: price closes below neckline after right shoulder
        breakdown_idx = None
        for k in range(rs_idx + 1, min(n, rs_idx + 30)):
            # Projected neckline level at position k
            neckline_at_k = left_neck_price + neckline_slope * (k - left_neck_idx)
            if prices[k] < neckline_at_k:
                breakdown_idx = k
                break

        if breakdown_idx is None:
            continue

        # Volume checks
        vol_sma50 = 0
        vol_ls_avg = 0
        vol_head_avg = 0
        vol_rs_avg = 0
        vol_progression_pass = None
        vol_breakdown_pass = None

        if volumes is not None and len(volumes) == n:
            vol_series = pd.Series(volumes)
            vol_sma = vol_series.rolling(window=VOLUME_SMA_PERIOD, min_periods=10).mean()
            sma_idx = min(rs_idx, len(vol_sma) - 1)
            vol_sma50 = float(vol_sma.iloc[sma_idx]) if not pd.isna(vol_sma.iloc[sma_idx]) else 0

            # Volume around each peak (±3 candles)
            def avg_vol_around(idx, radius=3):
                start = max(0, idx - radius)
                end = min(n, idx + radius + 1)
                return float(np.mean(volumes[start:end]))

            vol_ls_avg = avg_vol_around(ls_idx)
            vol_head_avg = avg_vol_around(head_idx)
            vol_rs_avg = avg_vol_around(rs_idx)

            # Volume progression: LS > Head > RS (declining) is ideal
            vol_progression_pass = (vol_ls_avg > vol_head_avg > vol_rs_avg)

            if vol_sma50 > 0:
                bd_vol = float(volumes[breakdown_idx])
                vol_breakdown_pass = bd_vol > (BREAKOUT_VOLUME_MULTIPLIER * vol_sma50)

        # Head prominence for scoring (average of both sides)
        head_prom_pct = (ls_prominence + rs_prominence) / 2

        quality = compute_quality_score("head_and_shoulders", {
            "head_prominence_pct": head_prom_pct,
            "shoulder_symmetry_pct": shoulder_diff_pct,
            "neckline_slope_pct": neckline_slope_pct,
            "vol_progression_pass": vol_progression_pass,
            "vol_breakdown_pass": vol_breakdown_pass,
        })

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
            "breakdown_price":       round(float(prices[breakdown_idx]), 2),
            "head_vs_ls_pct":        round(float(ls_prominence), 2),
            "head_vs_rs_pct":        round(float(rs_prominence), 2),
            "shoulder_symmetry_pct": round(float(shoulder_diff_pct), 2),
            "neckline_slope":        round(float(neckline_slope), 4),
            "neckline_slope_pct":    round(float(neckline_slope_pct), 2),
            "span_candles":          int(span),
            "vol_ls_avg":            round(float(vol_ls_avg), 0),
            "vol_head_avg":          round(float(vol_head_avg), 0),
            "vol_rs_avg":            round(float(vol_rs_avg), 0),
            "vol_progression_pass":  vol_progression_pass,
            "vol_breakdown_pass":    vol_breakdown_pass,
            "quality_score":         quality,
            "left_shoulder_idx":     int(ls_idx),
            "head_idx":              int(head_idx),
            "right_shoulder_idx":    int(rs_idx),
            "left_neckline_idx":     int(left_neck_idx),
            "right_neckline_idx":    int(right_neck_idx),
            "breakdown_idx":         int(breakdown_idx),
        }

        if dates is not None:
            pattern["left_shoulder_date"]  = str(dates[ls_idx])
            pattern["head_date"]           = str(dates[head_idx])
            pattern["right_shoulder_date"] = str(dates[rs_idx])
            pattern["left_neckline_date"]  = str(dates[left_neck_idx])
            pattern["right_neckline_date"] = str(dates[right_neck_idx])
            pattern["breakdown_date"]      = str(dates[breakdown_idx])
            pattern["signal_date"]         = str(dates[breakdown_idx])

        candidates.append(pattern)

    # Deduplication
    candidates.sort(key=lambda p: p["quality_score"], reverse=True)
    patterns_found = []
    claimed = []

    for c in candidates:
        is_dup = False
        for (ch, cr) in claimed:
            if abs(c["head_idx"] - ch) <= 10 and abs(c["right_shoulder_idx"] - cr) <= 10:
                is_dup = True
                break
        if not is_dup:
            patterns_found.append(c)
            claimed.append((c["head_idx"], c["right_shoulder_idx"]))

    return patterns_found


# =============================================================================
#  10. DETECT DOUBLE TOP
# =============================================================================

def detect_double_top(prices, volumes=None, ticker="UNKNOWN", dates=None,
                      interval="1d", verbose=False):
    """
    Detects Double Top reversal patterns: two consecutive peaks at roughly
    the same height, with a valley between them. Breakdown below the
    valley (neckline) confirms a bearish reversal.

    Smoothing: YES — uses smooth_prices() + find_peaks() for peak detection.

    Theory
    ------
    The Double Top is shaped like the letter "M":
    - Price rallies to a HIGH (first top)
    - Price pulls back (creating the valley/neckline)
    - Price rallies again to approximately the SAME high (second top)
    - Price fails to go higher → buyers are exhausted
    - When price breaks below the valley, it confirms a reversal downward
    """
    prices = np.array(prices, dtype=float)
    if volumes is not None:
        volumes = np.array(volumes, dtype=float)

    n = len(prices)
    if n < 30:
        return []

    # ── Smooth and find peaks ──
    smoothed = smooth_prices(prices, window=SMOOTHING_WINDOW)
    min_prominence = np.median(smoothed) * 0.01
    peak_indices, _ = find_peaks(smoothed, distance=8, prominence=min_prominence)

    if len(peak_indices) < 2:
        return []

    candidates = []

    for i in range(len(peak_indices) - 1):
        top1_idx = peak_indices[i]
        top2_idx = peak_indices[i + 1]

        top1_price = prices[top1_idx]
        top2_price = prices[top2_idx]

        # Separation check
        separation = top2_idx - top1_idx
        if separation < MIN_DOUBLE_CANDLES:
            continue

        # Similarity check: tops within DOUBLE_TOP_SIMILARITY_PCT
        similarity_pct = abs(top1_price - top2_price) / top1_price * 100
        if similarity_pct > DOUBLE_TOP_SIMILARITY_PCT:
            continue

        # Find valley (lowest point between the two tops)
        valley_prices = prices[top1_idx:top2_idx + 1]
        valley_rel_idx = int(np.argmin(valley_prices))
        valley_idx = top1_idx + valley_rel_idx
        valley_price = prices[valley_idx]

        # Valley depth check
        avg_top = (top1_price + top2_price) / 2
        valley_drop_pct = (avg_top - valley_price) / avg_top * 100
        if valley_drop_pct < MIN_VALLEY_DROP_PCT:
            continue

        # Breakdown confirmation: price closes below valley after second top
        breakdown_idx = None
        for k in range(top2_idx + 1, min(n, top2_idx + 30)):
            if prices[k] < valley_price:
                breakdown_idx = k
                break

        if breakdown_idx is None:
            continue

        # Volume checks
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

            # Volume around each top (±3 candles)
            def avg_vol(idx, radius=3):
                s = max(0, idx - radius)
                e = min(n, idx + radius + 1)
                return float(np.mean(volumes[s:e]))

            vol_top1_avg = avg_vol(top1_idx)
            vol_top2_avg = avg_vol(top2_idx)

            # First top should have higher volume than second (weakening)
            vol_pattern_pass = vol_top1_avg > vol_top2_avg

            if vol_sma50 > 0:
                bd_vol = float(volumes[breakdown_idx])
                vol_breakdown_pass = bd_vol > (BREAKOUT_VOLUME_MULTIPLIER * vol_sma50)

        quality = compute_quality_score("double_top", {
            "similarity_pct": similarity_pct,
            "depth_pct": valley_drop_pct,
            "vol_pattern_pass": vol_pattern_pass,
            "vol_breakout_pass": vol_breakdown_pass,
        })

        pattern = {
            "pattern_type":       "double_top",
            "ticker":             ticker,
            "verdict":            "VALID",
            "reject_reason":      "",
            "first_top_price":    round(float(top1_price), 2),
            "valley_price":       round(float(valley_price), 2),
            "second_top_price":   round(float(top2_price), 2),
            "breakdown_price":    round(float(prices[breakdown_idx]), 2),
            "similarity_pct":     round(float(similarity_pct), 2),
            "valley_drop_pct":    round(float(valley_drop_pct), 2),
            "separation":         int(separation),
            "vol_top1_avg":       round(float(vol_top1_avg), 0),
            "vol_top2_avg":       round(float(vol_top2_avg), 0),
            "vol_pattern_pass":   vol_pattern_pass,
            "vol_breakdown_pass": vol_breakdown_pass,
            "quality_score":      quality,
            "first_top_idx":      int(top1_idx),
            "valley_idx":         int(valley_idx),
            "second_top_idx":     int(top2_idx),
            "breakdown_idx":      int(breakdown_idx),
        }

        if dates is not None:
            pattern["first_top_date"]  = str(dates[top1_idx])
            pattern["valley_date"]     = str(dates[valley_idx])
            pattern["second_top_date"] = str(dates[top2_idx])
            pattern["breakdown_date"]  = str(dates[breakdown_idx])
            pattern["signal_date"]     = str(dates[breakdown_idx])

        candidates.append(pattern)

    # Deduplication
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


# =============================================================================
#  11. DETECT DOUBLE BOTTOM
# =============================================================================

def detect_double_bottom(prices, volumes=None, ticker="UNKNOWN", dates=None,
                         interval="1d", verbose=False):
    """
    Detects Double Bottom reversal patterns: two consecutive troughs at
    roughly the same depth, with a peak between them. Breakout above
    the peak (neckline) confirms a bullish reversal.

    Smoothing: YES — uses smooth_prices() + find_peaks(-x) for trough detection.

    Theory
    ------
    The Double Bottom is shaped like the letter "W":
    - Price drops to a LOW (first bottom)
    - Price bounces up (creating the peak/neckline)
    - Price drops again to approximately the SAME low (second bottom)
    - Price holds → sellers are exhausted
    - When price breaks above the peak, it confirms a reversal upward
    """
    prices = np.array(prices, dtype=float)
    if volumes is not None:
        volumes = np.array(volumes, dtype=float)

    n = len(prices)
    if n < 30:
        return []

    # ── Smooth and find troughs ──
    smoothed = smooth_prices(prices, window=SMOOTHING_WINDOW)
    min_prominence = np.median(smoothed) * 0.01
    trough_indices, _ = find_peaks(-smoothed, distance=8, prominence=min_prominence)

    if len(trough_indices) < 2:
        return []

    candidates = []

    for i in range(len(trough_indices) - 1):
        bot1_idx = trough_indices[i]
        bot2_idx = trough_indices[i + 1]

        bot1_price = prices[bot1_idx]
        bot2_price = prices[bot2_idx]

        # Separation check
        separation = bot2_idx - bot1_idx
        if separation < MIN_DOUBLE_CANDLES:
            continue

        # Similarity check
        similarity_pct = abs(bot1_price - bot2_price) / bot1_price * 100
        if similarity_pct > DOUBLE_BOTTOM_SIMILARITY_PCT:
            continue

        # Find peak (highest point between the two bottoms)
        between_prices = prices[bot1_idx:bot2_idx + 1]
        peak_rel_idx = int(np.argmax(between_prices))
        peak_idx = bot1_idx + peak_rel_idx
        peak_price = prices[peak_idx]

        # Peak height check
        avg_bot = (bot1_price + bot2_price) / 2
        peak_rise_pct = (peak_price - avg_bot) / avg_bot * 100
        if peak_rise_pct < MIN_PEAK_RISE_PCT:
            continue

        # Breakout confirmation: price closes above peak after second bottom
        breakout_idx = None
        for k in range(bot2_idx + 1, min(n, bot2_idx + 30)):
            if prices[k] > peak_price:
                breakout_idx = k
                break

        if breakout_idx is None:
            continue

        # Volume checks
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

            # Second bottom should have lower volume (declining selling pressure)
            vol_pattern_pass = vol_bot2_avg < vol_bot1_avg

            if vol_sma50 > 0:
                bo_vol = float(volumes[breakout_idx])
                vol_breakout_pass = bo_vol > (BREAKOUT_VOLUME_MULTIPLIER * vol_sma50)

        quality = compute_quality_score("double_bottom", {
            "similarity_pct": similarity_pct,
            "depth_pct": peak_rise_pct,
            "vol_pattern_pass": vol_pattern_pass,
            "vol_breakout_pass": vol_breakout_pass,
        })

        pattern = {
            "pattern_type":       "double_bottom",
            "ticker":             ticker,
            "verdict":            "VALID",
            "reject_reason":      "",
            "first_bottom_price": round(float(bot1_price), 2),
            "peak_price":         round(float(peak_price), 2),
            "second_bottom_price":round(float(bot2_price), 2),
            "breakout_price":     round(float(prices[breakout_idx]), 2),
            "similarity_pct":     round(float(similarity_pct), 2),
            "peak_rise_pct":      round(float(peak_rise_pct), 2),
            "separation":         int(separation),
            "vol_bot1_avg":       round(float(vol_bot1_avg), 0),
            "vol_bot2_avg":       round(float(vol_bot2_avg), 0),
            "vol_pattern_pass":   vol_pattern_pass,
            "vol_breakout_pass":  vol_breakout_pass,
            "quality_score":      quality,
            "first_bottom_idx":   int(bot1_idx),
            "peak_idx":           int(peak_idx),
            "second_bottom_idx":  int(bot2_idx),
            "breakout_idx":       int(breakout_idx),
        }

        if dates is not None:
            pattern["first_bottom_date"]  = str(dates[bot1_idx])
            pattern["peak_date"]          = str(dates[peak_idx])
            pattern["second_bottom_date"] = str(dates[bot2_idx])
            pattern["breakout_date"]      = str(dates[breakout_idx])
            pattern["signal_date"]        = str(dates[breakout_idx])

        candidates.append(pattern)

    # Deduplication
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
    print(f"  Pole R²              : {p['pole_r_squared']}")
    print(f"  Upper Trendline Slope: {p['upper_slope']}")
    print(f"  Lower Trendline Slope: {p['lower_slope']}")
    print(f"  Convergence Ratio    : {p['convergence_ratio']} "
          f"({'✅ Yes' if p['convergence_ratio'] < 1 else '❌ No'})")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ VOLUME")
    print(f"    Pole (above avg)   : {_vol_str(p['vol_pole_pass'])}")
    print(f"    Pennant (below avg): {_vol_str(p['vol_pennant_pass'])}")
    print(f"    Breakout (spike)   : {_vol_str(p['vol_breakout_pass'])}")
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
    """Prints Double Top debug output."""
    print(f"\n  {'═' * 60}")
    print(f"  🏆 DOUBLE TOP #{index}  —  {p['ticker']}  (Score: {p['quality_score']})")
    print(f"  {'═' * 60}")
    print(f"  First Top       : ₹{p['first_top_price']}"
          + (f"  ({p.get('first_top_date', '')})" if 'first_top_date' in p else ""))
    print(f"  Valley Low      : ₹{p['valley_price']}"
          + (f"  ({p.get('valley_date', '')})" if 'valley_date' in p else ""))
    print(f"  Second Top      : ₹{p['second_top_price']}"
          + (f"  ({p.get('second_top_date', '')})" if 'second_top_date' in p else ""))
    print(f"  Breakdown Point : ₹{p['breakdown_price']}"
          + (f"  ({p.get('breakdown_date', '')})" if 'breakdown_date' in p else ""))
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Top Similarity %     : {p['similarity_pct']}%")
    print(f"  Valley Depth %       : {p['valley_drop_pct']}%")
    print(f"  Separation           : {p['separation']} candles")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ VOLUME")
    print(f"    First Top avg      : {p['vol_top1_avg']:,.0f}")
    print(f"    Second Top avg     : {p['vol_top2_avg']:,.0f}")
    print(f"    Pattern (T1>T2)    : {_vol_str(p['vol_pattern_pass'])}")
    print(f"    Breakdown (spike)  : {_vol_str(p['vol_breakdown_pass'])}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  FINAL VERDICT: ✅ VALID (Quality Score: {p['quality_score']})")
    print(f"  {'═' * 60}\n")


def print_double_bottom(p, index=1):
    """Prints Double Bottom debug output."""
    print(f"\n  {'═' * 60}")
    print(f"  🏆 DOUBLE BOTTOM #{index}  —  {p['ticker']}  (Score: {p['quality_score']})")
    print(f"  {'═' * 60}")
    print(f"  First Bottom    : ₹{p['first_bottom_price']}"
          + (f"  ({p.get('first_bottom_date', '')})" if 'first_bottom_date' in p else ""))
    print(f"  Peak High       : ₹{p['peak_price']}"
          + (f"  ({p.get('peak_date', '')})" if 'peak_date' in p else ""))
    print(f"  Second Bottom   : ₹{p['second_bottom_price']}"
          + (f"  ({p.get('second_bottom_date', '')})" if 'second_bottom_date' in p else ""))
    print(f"  Breakout Point  : ₹{p['breakout_price']}"
          + (f"  ({p.get('breakout_date', '')})" if 'breakout_date' in p else ""))
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Bottom Similarity %  : {p['similarity_pct']}%")
    print(f"  Peak Rise %          : {p['peak_rise_pct']}%")
    print(f"  Separation           : {p['separation']} candles")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ VOLUME")
    print(f"    First Bottom avg   : {p['vol_bot1_avg']:,.0f}")
    print(f"    Second Bottom avg  : {p['vol_bot2_avg']:,.0f}")
    print(f"    Pattern (B2<B1)    : {_vol_str(p['vol_pattern_pass'])}")
    print(f"    Breakout (spike)   : {_vol_str(p['vol_breakout_pass'])}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  FINAL VERDICT: ✅ VALID (Quality Score: {p['quality_score']})")
    print(f"  {'═' * 60}\n")


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

    np.random.seed(202)
    lead = np.linspace(85, 90, 30)
    pole = np.linspace(90, 108, 15)  # bullish pole

    # Pennant body: oscillating with decreasing amplitude
    pennant = np.array([108.0, 104.0, 107.5, 104.5, 107.0, 105.0, 106.5, 105.5, 106.2, 105.8])
    breakout = np.array([109, 111, 113])
    post = np.ones(15) * 113

    prices = np.concatenate([lead, pole, pennant, breakout, post])
    n = len(prices)
    volumes = np.random.uniform(800000, 1200000, n)
    volumes[30:45] = np.random.uniform(1500000, 2000000, 15)
    volumes[45:55] = np.random.uniform(400000, 600000, 10)
    volumes[55:58] = np.random.uniform(2500000, 3500000, 3)

    results = detect_pennant(prices, volumes=volumes,
                              ticker="SYNTH_PENNANT", verbose=False)
    if results:
        print(f"  ✅ Textbook shape → VALID  (score: {results[0]['quality_score']})")
        print_pennant(results[0], 1)
    else:
        print("  ❌ Textbook shape → REJECTED (unexpected!)")

    # Flat noise
    np.random.seed(42)
    flat = 100 + np.random.normal(0, 0.5, 200)
    results_flat = detect_pennant(flat, ticker="FLAT")
    if not results_flat:
        print("  ✅ Flat/random    → REJECTED (correct)")
    else:
        print(f"  ⚠ Flat/random    → {len(results_flat)} false positive(s)")

    return bool(results) and not results_flat


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
    """Self-test for Double Top."""
    print("\n  ── DOUBLE TOP ───────────────────────────────────────────")

    np.random.seed(204)
    lead = np.linspace(85, 95, 20)
    first_up = np.linspace(95, 110, 15)
    valley_down = np.linspace(110, 104, 10)
    second_up = np.linspace(104, 110, 12)
    breakdown = np.linspace(110, 101, 10)
    post = np.ones(10) * 101

    prices = np.concatenate([lead, first_up, valley_down, second_up,
                              breakdown, post])
    n = len(prices)
    volumes = np.random.uniform(800000, 1200000, n)
    # First top: higher volume
    volumes[32:38] = np.random.uniform(2000000, 2500000, 6)
    # Second top: lower volume
    volumes[55:61] = np.random.uniform(1000000, 1300000, 6)
    # Breakdown: spike
    volumes[61:71] = np.random.uniform(2500000, 3000000, 10)

    results = detect_double_top(prices, volumes=volumes,
                                 ticker="SYNTH_DT", verbose=False)
    if results:
        print(f"  ✅ Textbook shape → VALID  (score: {results[0]['quality_score']})")
        print_double_top(results[0], 1)
    else:
        print("  ❌ Textbook shape → REJECTED (unexpected!)")

    # Flat noise
    np.random.seed(42)
    flat = 100 + np.random.normal(0, 0.5, 200)
    results_flat = detect_double_top(flat, ticker="FLAT")
    if not results_flat:
        print("  ✅ Flat/random    → REJECTED (correct)")
    else:
        print(f"  ⚠ Flat/random    → {len(results_flat)} false positive(s)")

    return bool(results) and not results_flat


def test_double_bottom():
    """Self-test for Double Bottom."""
    print("\n  ── DOUBLE BOTTOM ────────────────────────────────────────")

    np.random.seed(205)
    lead = np.linspace(115, 105, 20)
    first_down = np.linspace(105, 90, 15)
    peak_up = np.linspace(90, 96, 10)
    second_down = np.linspace(96, 90, 12)
    breakout = np.linspace(90, 100, 10)
    post = np.ones(10) * 100

    prices = np.concatenate([lead, first_down, peak_up, second_down,
                              breakout, post])
    n = len(prices)
    volumes = np.random.uniform(800000, 1200000, n)
    # First bottom: higher volume
    volumes[32:38] = np.random.uniform(2000000, 2500000, 6)
    # Second bottom: lower volume
    volumes[55:61] = np.random.uniform(800000, 1000000, 6)
    # Breakout: spike
    volumes[61:71] = np.random.uniform(2500000, 3000000, 10)

    results = detect_double_bottom(prices, volumes=volumes,
                                    ticker="SYNTH_DB", verbose=False)
    if results:
        print(f"  ✅ Textbook shape → VALID  (score: {results[0]['quality_score']})")
        print_double_bottom(results[0], 1)
    else:
        print("  ❌ Textbook shape → REJECTED (unexpected!)")

    # Flat noise
    np.random.seed(42)
    flat = 100 + np.random.normal(0, 0.5, 200)
    results_flat = detect_double_bottom(flat, ticker="FLAT")
    if not results_flat:
        print("  ✅ Flat/random    → REJECTED (correct)")
    else:
        print(f"  ⚠ Flat/random    → {len(results_flat)} false positive(s)")

    return bool(results) and not results_flat


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

def scan_ticker(df, ticker, patterns_to_scan, interval="1d", verbose=False):
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

    vol = None
    if "Volume" in df.columns:
        vol = df["Volume"].reindex(df["Close"].dropna().index).fillna(0).values

    if len(close) < 30:
        return {}

    scan_all = "all" in patterns_to_scan
    results = {}

    DETECT_MAP = {
        "cup_and_handle": lambda: detect_cup_and_handle(
            close, highs=highs, volumes=vol,
            ticker=ticker, dates=dates, interval=interval, verbose=verbose
        ),
        "bull_flag": lambda: detect_bull_flag(
            close, volumes=vol, ticker=ticker, dates=dates,
            interval=interval, verbose=verbose
        ),
        "bear_flag": lambda: detect_bear_flag(
            close, volumes=vol, ticker=ticker, dates=dates,
            interval=interval, verbose=verbose
        ),
        "pennant": lambda: detect_pennant(
            close, volumes=vol, ticker=ticker, dates=dates,
            interval=interval, verbose=verbose
        ),
        "head_and_shoulders": lambda: detect_head_and_shoulders(
            close, volumes=vol, ticker=ticker, dates=dates,
            interval=interval, verbose=verbose
        ),
        "double_top": lambda: detect_double_top(
            close, volumes=vol, ticker=ticker, dates=dates,
            interval=interval, verbose=verbose
        ),
        "double_bottom": lambda: detect_double_bottom(
            close, volumes=vol, ticker=ticker, dates=dates,
            interval=interval, verbose=verbose
        ),
    }

    for ptype, detect_fn in DETECT_MAP.items():
        if scan_all or ptype in patterns_to_scan:
            found = detect_fn()
            if found:
                results[ptype] = found

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
        ticker_results = scan_ticker(df, ticker, patterns_to_scan, interval=interval)

        for ptype, patterns in ticker_results.items():
            dedup_key = (ticker, ptype)
            if dedup_key in _live_alerted_today:
                continue

            for pat in patterns:
                if _is_pattern_from_today(pat):
                    _live_alerted_today.add(dedup_key)
                    new_alerts.append(pat)
                    break

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
