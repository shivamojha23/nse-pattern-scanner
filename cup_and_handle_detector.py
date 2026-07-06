"""
=============================================================================
  CUP AND HANDLE PATTERN DETECTOR  v2.0
  Indian Stock Market (NSE)
=============================================================================

  Author : AI-assisted project
  Purpose: Detect "Cup and Handle" chart patterns on NSE stocks using
           Yahoo Finance data, with verified mathematical geometry filters,
           price smoothing, volume confirmation, and composite quality scoring.

  THREE MODES
  -----------
  1. SELF-TEST     → Synthetic data to prove the math works (no internet).
  2. HISTORICAL    → Sweeps past data for already-completed patterns.
  3. LIVE SCANNER  → Continuous scanner during NSE market hours.

  What is a "Cup and Handle" pattern?
  ------------------------------------
  Imagine the cross-section of a coffee cup:

      Left Rim ────╮                  ╭──── Right Rim
                    ╲                ╱
                     ╲              ╱         ╮── Handle
                      ╲            ╱          │   (small dip)
                       ╰──────────╯           ╯
                         Cup Bottom

  1. Price rises to a HIGH (Left Rim).
  2. Price DROPS significantly (the "cup" — 10-35% decline).
  3. Price RECOVERS back near the original high (Right Rim, within 3%).
  4. Price dips SLIGHTLY again (the "handle" — must stay within 32% of
     cup depth).
  5. A breakout above the rim signals a potential upward move.

  Bug-Fix History (all incorporated in this v2 rewrite)
  -----------------------------------------------------
  Bug 1: V-shaped bottoms — now requires "roundedness" (base zone check).
  Bug 2: Handle low locked too early — now tracks true low until breakout.
  Bug 3: Noise mistaken for handle — requires genuine pause before breakout.
  Bug 4: Internal price > Left Rim — strict structural ceiling check.
  Bug 5: Discontinuous/jagged cups — max single-day gap filter.

  New in v2
  ---------
  • Price smoothing (SMA) for peak/trough detection (raw prices for math).
  • Volume confirmation (3 checks: decline, recovery, breakout).
  • Composite quality score for ranking patterns.

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

import numpy as np                     # Fast math on arrays of numbers
import pandas as pd                    # DataFrames — like Excel spreadsheets in Python
import yfinance as yf                  # Downloads stock price data from Yahoo Finance
from scipy.signal import find_peaks    # Finds local highs/lows in a price curve
from scipy.stats import linregress     # Calculates linear regression slope

# Suppress noisy warnings from yfinance / pandas
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
#  CONFIGURATION — All tuneable knobs in one place
# =============================================================================

# ─── SMOOTHING ──────────────────────────────────────────────────────────────
# Before finding peaks/troughs, we smooth the closing prices with a simple
# moving average (SMA) to reduce noise. The smoothed series is ONLY used
# for find_peaks() — all actual price values come from RAW data.
SMOOTHING_WINDOW = 5

# ─── CUP GEOMETRY ──────────────────────────────────────────────────────────
# Minimum cup drop from Left Rim to Cup Bottom (percentage).
MIN_CUP_DROP_PCT = 10
# Maximum cup drop from Left Rim to Cup Bottom (percentage).
MAX_CUP_DROP_PCT = 35
# Maximum gap between Right Rim and Left Rim (percentage).
# 0% = identical height; 3% allows a small difference.
MAX_RECOVERY_GAP_PCT = 3

# ─── HANDLE GEOMETRY ───────────────────────────────────────────────────────
# Handle pullback must be ≤ this fraction of cup depth.
# 0.32 means the handle can retrace at most 32% of the cup's height.
HANDLE_MAX_RETRACE_RATIO = 0.32
# Minimum candles for each side of the cup (left-to-bottom, bottom-to-right).
MIN_CUP_CANDLES = 10
# Minimum candles for the handle pullback phase.
MIN_HANDLE_CANDLES = 5
# Right Rim stability: average price in a ±2 candle window must be within
# this percentage of the Right Rim peak.
RIGHT_RIM_STABILITY_PCT = 3.0

# ─── BUG FIX PARAMETERS ───────────────────────────────────────────────────
# Bug 1 — Bottom Roundedness
BASE_ZONE_PCT = 0.05          # Prices within 5% of cup bottom = "base zone"
MIN_BASE_CANDLES_PCT = 0.20   # At least 20% of cup candles must be in base zone

# Bug 2 — Handle Low Lock
BREAKOUT_CONFIRM_CANDLES = 3          # Consecutive closes above Right Rim
MAX_HANDLE_LOOKFORWARD_CANDLES = 30   # Max candles to search for breakout


# Bug 3 — Genuine Pause Before Breakout
MIN_PAUSE_CANDLES = 5

# Bug 5 — Discontinuity / Smoothness
MAX_DISCONTINUITY_PCT = 0.08  # Max single-day % change inside the cup

# ─── VOLUME CONFIRMATION ──────────────────────────────────────────────────
# Breakout volume must exceed this multiplier × 50-period SMA of volume.
BREAKOUT_VOLUME_MULTIPLIER = 1.2
# Volume SMA period for baseline comparison.
VOLUME_SMA_PERIOD = 50

# ─── DATA FETCHING ────────────────────────────────────────────────────────
BATCH_SIZE = 25               # Tickers per yf.download() call
BATCH_SLEEP = 2               # Seconds between download batches

# ─── LIVE SCANNER ─────────────────────────────────────────────────────────
SCAN_INTERVAL_MINUTES = 15    # How often the live scanner re-runs
IST_OFFSET = datetime.timedelta(hours=5, minutes=30)

# ─── LIVE-MODE DEDUPLICATION CACHE ─────────────────────────────────────────
_live_alerted_today = set()
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

    How it works
    ------------
    1. Tries multiple public CSV URLs containing Nifty index constituents.
    2. Reads them with pandas, looks for a "Symbol" column.
    3. Appends ".NS" so Yahoo Finance knows it's an NSE stock.
    4. Falls back to hardcoded Nifty 50 if all URLs fail.
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

    Parameters
    ----------
    tickers : list[str]
        e.g., ["RELIANCE.NS", "TCS.NS", ...]
    period : str
        How far back to look (e.g., "2y", "59d").
    start / end : str
        Date range (YYYY-MM-DD). Overrides period if provided.
    interval : str
        Candle size (e.g., "15m", "1d").

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are ticker symbols; values are DataFrames with columns:
        Open, High, Low, Close, Volume.
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
#  3. SMOOTH PRICES — Reduce noise before peak detection
# =============================================================================

def smooth_prices(prices, window=SMOOTHING_WINDOW):
    """
    Applies a Simple Moving Average (SMA) to the price series for noise
    reduction before peak/trough detection.

    Why we smooth
    -------------
    Raw stock prices have tiny daily wiggles that create false peaks/troughs.
    By averaging over a small window (default 5 candles), we eliminate
    1-2 candle noise while preserving the real structural shape of the cup.

    IMPORTANT: The smoothed series is ONLY used for find_peaks() to identify
    candidate dates. All actual price values used in calculations come from
    the RAW (unsmoothed) prices at those same dates.

    Parameters
    ----------
    prices : np.ndarray
        Raw closing prices.
    window : int
        Number of candles to average over (default 5).

    Returns
    -------
    np.ndarray
        Smoothed price series (same length as input).
    """
    # pd.Series.rolling with min_periods=1 handles the edges gracefully —
    # the first few values use whatever data is available instead of NaN.
    series = pd.Series(prices)
    smoothed = series.rolling(window=window, min_periods=1, center=True).mean()
    return smoothed.values


# =============================================================================
#  4. COMPUTE QUALITY SCORE — Rank patterns by strength
# =============================================================================

def compute_quality_score(cup_drop_pct, recovery_vol_ratio, breakout_vol_ratio):
    """
    Computes a single number that ranks how "strong" a Cup & Handle pattern is.

    Why a score?
    ------------
    When scanning 200 stocks, you might find 10 valid patterns. The score
    tells you which ones are the MOST convincing — deeper cups with strong
    volume confirmation rank higher.

    Formula (documented for beginners)
    -----------------------------------
    score = (cup_drop_pct × 0.4)
          + (log(recovery_vol_ratio + 1) × 30)
          + (log(breakout_vol_ratio + 1) × 30)

    • cup_drop_pct × 0.4: Deeper cups score higher (a 20% drop scores
      8 points, a 10% drop scores 4 points). We use 0.4 weight because
      cup depth is the most basic quality signal.

    • log(recovery_vol_ratio + 1) × 30: If the recovery had 2× more buying
      volume than selling volume, log(3) ≈ 1.1, so this adds ~33 points.
      The log prevents a single crazy volume spike from dominating.

    • log(breakout_vol_ratio + 1) × 30: Same logic for breakout volume.
      Higher breakout volume = more institutional interest.

    Parameters
    ----------
    cup_drop_pct : float
        How much the price dropped (e.g., 15.5 for 15.5%).
    recovery_vol_ratio : float
        Up-volume / down-volume ratio during recovery (> 1 = bullish).
    breakout_vol_ratio : float
        Breakout volume / 50-SMA volume ratio (> 1.2 = strong).

    Returns
    -------
    float
        Composite score (higher is better, typically 5–80 range).
    """
    # Clamp ratios to avoid negative log issues (ratio could be 0 if no data)
    rv = max(recovery_vol_ratio, 0)
    bv = max(breakout_vol_ratio, 0)

    score = (cup_drop_pct * 0.4
             + math.log(rv + 1) * 30
             + math.log(bv + 1) * 30)

    return round(score, 2)


# =============================================================================
#  5. PATTERN DETECTION — Find Cup and Handle shapes (core engine)
# =============================================================================

def detect_cup_and_handle(prices, highs=None, volumes=None,
                          ticker="UNKNOWN", dates=None,
                          interval="1d", verbose=False):
    """
    Scans a 1-D array of closing prices for Cup-and-Handle patterns using
    verified mathematical geometry filters, price smoothing, volume
    confirmation, and all bug-fix rules.

    Parameters
    ----------
    prices : array-like
        Sequence of CLOSING prices (raw, not smoothed).
    highs : array-like or None
        Corresponding HIGH prices (for structural ceiling check).
        Falls back to Close prices if not provided.
    volumes : array-like or None
        Corresponding VOLUME data (for volume confirmation checks).
        If None, volume checks are skipped (marked as N/A).
    ticker : str
        Stock symbol (used in print messages).
    dates : array-like or None
        Corresponding timestamps for each price bar.
    interval : str
        Candle interval (e.g., "15m", "1d", "1wk").
    verbose : bool
        If True, prints debug output for ALL candidates (even rejected ones).

    Returns
    -------
    list[dict]
        Each dict describes one detected pattern with all metrics.

    Algorithm (plain English)
    -------------------------
    1. Smooth prices with SMA(5). Run find_peaks on SMOOTHED series to find
       candidate peaks (Left/Right Rim) and troughs (Cup Bottom). All actual
       price values are read from RAW data at those indices.
    2. For each (Left Rim, Right Rim) pair, validate:
       - Cup depth (10–35% drop)
       - Recovery (Right Rim within 3% of Left Rim)
       - No double-dip below cup bottom
       - Cup symmetry (bottom in middle 70% of cup width)
       - Bottom roundedness (≥20% of candles in base zone)
       - No internal price exceeding Left Rim (structural ceiling)
       - No excessive discontinuity (max 8% single-day gap)
       - Handle detection with corrected low (Bug 2 fix)
       - Genuine pause before breakout (Bug 3 fix)
       - Handle geometry (≤32% of cup depth)
       - Right Rim stability
    3. Volume confirmation (3 checks — soft, don't reject):
       a. Cup decline volume vs 50-SMA
       b. Recovery up-vol / down-vol ratio
       c. Breakout volume vs 1.2× 50-SMA
    4. Compute composite quality score.
    5. Overlap deduplication (keep deepest cup per region).
    """
    prices = np.array(prices, dtype=float)

    # High prices for structural ceiling check (fallback to Close)
    if highs is not None:
        highs = np.array(highs, dtype=float)
    else:
        highs = prices.copy()

    # Volume data (None = skip volume checks)
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

    # Collect ALL candidate patterns first, then deduplicate.
    candidates = []

    # ── Step 2: Try every combination of (Left Rim, Right Rim) ──
    for i in range(len(peak_indices)):
        for j in range(i + 1, len(peak_indices)):
            left_rim_idx = peak_indices[i]
            right_rim_idx = peak_indices[j]

            # Read RAW prices at the smoothed-detected indices
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

            # Refine the cup bottom to be the absolute minimum RAW price between the rims.
            # This prevents noise from failing the double-dip check, as the true structural
            # bottom is the absolute lowest traded price.
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
                if verbose:
                    print(f"  ❌ REJECTED {ticker}: Double dip below cup bottom.")
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

            # OLD handle logic (for debug comparison)
            old_handle_zone_end = min(right_rim_idx + max(1, cup_width // 4), len(prices) - 1)
            old_handle_prices = prices[right_rim_idx : old_handle_zone_end + 1]
            old_handle_low_price = np.min(old_handle_prices) if len(old_handle_prices) > 0 else right_rim_price

            # NEW: Scan forward tracking true low until breakout confirms
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

            # ══════════════════════════════════════════════════════════
            # ── Validation: compute all metrics, then decide ──
            # ══════════════════════════════════════════════════════════

            # Roundedness (Bug 1)
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

            # Intermediate High / Close check (Bug 4)
            rim_ceiling = highs[left_rim_idx]
            highs_inside_cup = highs[left_rim_idx + 1 : right_rim_idx]
            closes_inside_cup = prices[left_rim_idx + 1 : right_rim_idx]
            max_internal_high = np.max(highs_inside_cup) if len(highs_inside_cup) > 0 else 0
            max_internal_close = np.max(closes_inside_cup) if len(closes_inside_cup) > 0 else 0

            # Discontinuity check (Bug 5)
            cup_closes_for_diff = prices[left_rim_idx : right_rim_idx + 1]
            if len(cup_closes_for_diff) > 1:
                daily_returns = np.abs(np.diff(cup_closes_for_diff) / cup_closes_for_diff[:-1])
                max_daily_jump = float(np.max(daily_returns))
            else:
                max_daily_jump = 0.0

            # ── Volume Confirmation (NEW in v2) ──
            vol_decline_pass = None   # None = no volume data
            vol_decline_avg = 0
            vol_sma50 = 0
            vol_recovery_pass = None
            vol_recovery_ratio = 0
            vol_breakout_pass = None
            vol_breakout_avg = 0

            if volumes is not None and len(volumes) == len(prices):
                # Compute 50-period volume SMA at the Right Rim position
                vol_series = pd.Series(volumes)
                vol_sma = vol_series.rolling(window=VOLUME_SMA_PERIOD, min_periods=10).mean()

                # Use the SMA value at the right rim (or nearest valid)
                sma_idx = min(right_rim_idx, len(vol_sma) - 1)
                vol_sma50 = vol_sma.iloc[sma_idx] if not pd.isna(vol_sma.iloc[sma_idx]) else 0

                # (a) Cup Decline Volume: avg volume from Left Rim to Bottom
                if cup_bottom_idx > left_rim_idx and vol_sma50 > 0:
                    decline_vols = volumes[left_rim_idx:cup_bottom_idx + 1]
                    vol_decline_avg = float(np.mean(decline_vols))
                    vol_decline_pass = vol_decline_avg < vol_sma50

                # (b) Recovery Volume: up-days vs down-days from Bottom to Right Rim
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

                # (c) Breakout Volume: avg volume on breakout-confirming candles
                if breakout_confirmed and breakout_start_idx > 0 and vol_sma50 > 0:
                    bo_end = min(breakout_start_idx + BREAKOUT_CONFIRM_CANDLES, len(volumes))
                    breakout_vols = volumes[breakout_start_idx:bo_end]
                    if len(breakout_vols) > 0:
                        vol_breakout_avg = float(np.mean(breakout_vols))
                        vol_breakout_pass = vol_breakout_avg > (BREAKOUT_VOLUME_MULTIPLIER * vol_sma50)

            # ── Compute quality score ──
            quality_score = compute_quality_score(
                cup_drop_pct,
                vol_recovery_ratio if vol_recovery_ratio else 0,
                (vol_breakout_avg / vol_sma50) if vol_sma50 > 0 else 0
            )

            # ── Determine validity ──
            is_valid = True
            reject_reason = ""

            if max_internal_close > left_rim_price:
                is_valid = False
                reject_reason = (f"Internal close (₹{max_internal_close:.2f}) exceeded "
                                 f"Left Rim close (₹{left_rim_price:.2f}) — Structural Violation.")
            elif max_internal_high > rim_ceiling:
                is_valid = False
                reject_reason = (f"Internal high (₹{max_internal_high:.2f}) exceeded "
                                 f"Left Rim High (₹{rim_ceiling:.2f}) — Structural Violation.")
            elif max_daily_jump > MAX_DISCONTINUITY_PCT:
                is_valid = False
                reject_reason = (f"Excessive price discontinuity "
                                 f"({max_daily_jump*100:.1f}% single-day jump > "
                                 f"{MAX_DISCONTINUITY_PCT*100}% limit).")
            elif roundedness_pct < MIN_BASE_CANDLES_PCT:
                is_valid = False
                reject_reason = (f"V-shape recovery. Bottom roundedness "
                                 f"{base_candles_count}/{len(cup_candles)} "
                                 f"({roundedness_pct*100:.1f}%) < "
                                 f"{MIN_BASE_CANDLES_PCT*100}% required.")
            elif handle_pullback < min_handle_pullback:
                is_valid = False
                reject_reason = "Handle pullback too small (<1%)."
            elif not breakout_confirmed:
                is_valid = False
                reject_reason = (f"Handle incomplete / no breakout confirmed within "
                                 f"{MAX_HANDLE_LOOKFORWARD_CANDLES} candles.")
            elif pause_duration < MIN_PAUSE_CANDLES:
                is_valid = False
                reject_reason = (f"No real handle — immediate continuation "
                                 f"(breakout in {pause_duration} candles < "
                                 f"{MIN_PAUSE_CANDLES}), not Cup & Handle.")
            elif handle_slope > 0:
                is_valid = False
                reject_reason = (f"Handle is not a genuine pause (slope "
                                 f"{handle_slope:.2f} > 0). Resumed upward momentum.")
            elif handle_pullback > max_handle_dip:
                is_valid = False
                reject_reason = (f"Corrected handle pullback (₹{handle_pullback:.2f}) "
                                 f"exceeded 0.32× depth (₹{max_handle_dip:.2f}).")
            elif handle_duration < MIN_HANDLE_CANDLES:
                is_valid = False
                reject_reason = f"Handle formed too quickly (<{MIN_HANDLE_CANDLES} candles)."
            elif handle_duration > 0 and cup_width < 3 * handle_duration:
                is_valid = False
                reject_reason = "Cup duration must be at least 3x handle duration."
            else:
                # Right Rim Stability check
                rim_window_start = max(0, right_rim_idx - 2)
                rim_window_end = min(len(prices), right_rim_idx + 3)
                rim_window_prices = prices[rim_window_start:rim_window_end]
                if np.mean(rim_window_prices) < right_rim_price * (1 - RIGHT_RIM_STABILITY_PCT / 100):
                    is_valid = False
                    reject_reason = "Right rim is too spiky (failed stability check)."

            # If invalid and not verbose, skip
            if not is_valid and not verbose:
                continue

            # ── Build pattern dict ──
            pattern = {
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
                # Volume metrics
                "vol_decline_avg":     round(float(vol_decline_avg), 0),
                "vol_sma50":           round(float(vol_sma50), 0),
                "vol_decline_pass":    vol_decline_pass,
                "vol_recovery_ratio":  round(float(vol_recovery_ratio), 2),
                "vol_recovery_pass":   vol_recovery_pass,
                "vol_breakout_avg":    round(float(vol_breakout_avg), 0),
                "vol_breakout_pass":   vol_breakout_pass,
                # Smoothing metadata
                "smoothing_method":    f"SMA({SMOOTHING_WINDOW})",
            }

            # Attach dates if available
            if dates is not None:
                pattern["left_rim_date"]   = str(dates[left_rim_idx])
                pattern["cup_bottom_date"] = str(dates[cup_bottom_idx])
                pattern["right_rim_date"]  = str(dates[right_rim_idx])
                pattern["handle_low_date"] = str(dates[handle_low_idx])

            if not is_valid and verbose:
                print_pattern(pattern, index="REJECTED")
                continue

            candidates.append(pattern)

    # ── Step 5: Overlap Deduplication ──
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

    # Sort by quality score (highest first)
    patterns_found.sort(key=lambda p: p.get("quality_score", 0), reverse=True)

    return patterns_found


# =============================================================================
#  6. PRETTY PRINT — Show pattern details in the requested debug format
# =============================================================================

def print_pattern(p, index=1):
    """
    Prints one detected pattern with ALL diagnostic details so a beginner
    can sanity-check every rule, even without stock market knowledge.

    Output format matches the user's requested specification exactly.
    """
    score_str = f"          Quality Score: {p.get('quality_score', 0)}" if not p.get("reject_reason") else ""

    print(f"\n  {'═' * 60}")
    print(f"  🏆 PATTERN #{index}  —  {p['ticker']}{score_str}")
    print(f"  {'═' * 60}")
    print(f"  Left Rim  (peak before cup)  : ₹{p['left_rim_price']}")
    print(f"  Cup Bottom (lowest point)    : ₹{p['cup_bottom_price']}")
    print(f"  Right Rim  (recovery peak)   : ₹{p['right_rim_price']}")
    print(f"  Handle Low (small dip after) : ₹{p['handle_low_price']}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Cup Drop     : {p['cup_drop_pct']}%  "
          f"(price fell this much from Left Rim to Bottom)")
    print(f"  Recovery Gap : {p['recovery_pct']}%  "
          f"(how close Right Rim is to Left Rim; <3% = good)")

    # ── Geometry Check ──
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ GEOMETRY CHECK")
    print(f"    Cup Depth (₹)              : ₹{p['cup_depth']}")
    print(f"    Max Handle Dip Allowed (₹) : ₹{p['max_handle_dip']}  "
          f"(= 0.32 × ₹{p['cup_depth']})")
    print(f"    Actual Handle Pullback (₹) : ₹{p['handle_pullback']}  "
          f"({p['handle_pullback_pct']}%)")

    # ── Roundedness Check ──
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ ROUNDEDNESS CHECK")
    rnd_pass = "✅ PASS" if p.get('roundedness_pct', 0) >= MIN_BASE_CANDLES_PCT * 100 else "❌ FAIL"
    print(f"    {p.get('base_candles', 'N/A')} candles in base zone out of "
          f"{p.get('cup_width', 'N/A')} cup candles "
          f"({p.get('roundedness_pct', 'N/A')}%) — {rnd_pass}")

    # ── Pause Before Breakout ──
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ PAUSE-BEFORE-BREAKOUT CHECK")
    pause = p.get('pause_duration', 0)
    slope = p.get('handle_slope', 0)
    pause_pass = "✅ PASS" if pause >= MIN_PAUSE_CANDLES and slope <= 0 else "❌ FAIL"
    print(f"    Breakout Confirmed in      : {pause} candles (Needs ≥ {MIN_PAUSE_CANDLES})")
    print(f"    Handle Slope               : {slope} — {pause_pass}")

    # ── Handle Low (old vs new) ──
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ HANDLE LOW")
    print(f"    OLD (first-dip)            : ₹{p.get('old_handle_low', 'N/A')}")
    print(f"    NEW (breakout-confirmed)   : ₹{p['handle_low_price']}")

    # ── Volume — Cup Decline ──
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ VOLUME — Cup Decline")
    if p.get('vol_decline_pass') is not None:
        vd_status = "✅ PASS (light selling)" if p['vol_decline_pass'] else "⚠ WARN (heavy selling)"
        print(f"    Avg vol during decline     : {p['vol_decline_avg']:,.0f}")
        print(f"    50-period SMA vol          : {p['vol_sma50']:,.0f}")
        print(f"    Status                     : {vd_status}")
    else:
        print(f"    Status                     : N/A (no volume data)")

    # ── Volume — Recovery ──
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ VOLUME — Recovery")
    if p.get('vol_recovery_pass') is not None:
        vr_status = "✅ PASS (more buying)" if p['vol_recovery_pass'] else "⚠ WARN (weak buying)"
        print(f"    Up-vol / Down-vol ratio    : {p['vol_recovery_ratio']:.2f}")
        print(f"    Status                     : {vr_status}")
    else:
        print(f"    Status                     : N/A (no volume data)")

    # ── Volume — Breakout ──
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ VOLUME — Breakout")
    if p.get('vol_breakout_pass') is not None:
        vb_status = "✅ PASS (strong interest)" if p['vol_breakout_pass'] else "⚠ WARN (low interest)"
        print(f"    Avg breakout volume        : {p['vol_breakout_avg']:,.0f}")
        print(f"    Threshold (1.2× SMA50)     : {p['vol_sma50'] * BREAKOUT_VOLUME_MULTIPLIER:,.0f}")
        print(f"    Status                     : {vb_status}")
    else:
        print(f"    Status                     : N/A (no volume data)")

    # ── Smoothing ──
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ SMOOTHING")
    print(f"    Method                     : {p.get('smoothing_method', 'None')}")
    print(f"    Note: Peaks detected on smoothed series; all prices are RAW.")

    # ── Dates ──
    if "left_rim_date" in p:
        print(f"  ─────────────────────────────────────────────────")
        print(f"  📅 Left Rim Date   : {p['left_rim_date']}")
        print(f"  📅 Cup Bottom Date : {p['cup_bottom_date']}")
        print(f"  📅 Right Rim Date  : {p['right_rim_date']}")
        print(f"  📅 Handle Low Date : {p['handle_low_date']}")

    # ── Final Verdict ──
    print(f"  ─────────────────────────────────────────────────")
    if p.get("reject_reason"):
        print(f"  FINAL VERDICT: ❌ REJECTED ({p['reject_reason']})")
    else:
        print(f"  FINAL VERDICT: ✅ VALID (Quality Score: {p.get('quality_score', 0)})")

    print(f"  {'═' * 60}\n")


# =============================================================================
#  7. SELF-TEST — Synthetic data to prove detection math works
# =============================================================================

def test_with_sample_data():
    """
    Creates FAKE price + volume data shaped like textbook patterns,
    runs detection, and verifies results. ZERO internet needed.

    Tests
    -----
    1. Textbook Cup & Handle → should PASS (VALID).
    2. Flat/random series   → should find NOTHING.
    3. V-shaped recovery    → should REJECT (roundedness failure).
    4. Handle too deep      → should REJECT (geometry failure).
    """
    print("\n" + "=" * 70)
    print("  🧪 SELF-TEST : Synthetic Cup & Handle Verification")
    print("=" * 70)

    # ── Test 1: Textbook Cup and Handle (should PASS) ────────────────────
    print("\n  TEST 1: Textbook Cup & Handle shape")
    print("  " + "-" * 50)

    np.random.seed(123)

    # Build each segment:
    rise_to_left_rim = np.linspace(90, 100, 15)
    descent = np.linspace(100, 80, 30)
    bottom = np.ones(20) * 80 + np.random.uniform(-0.3, 0.3, 20)
    ascent = np.linspace(80, 100, 30)
    right_rim = np.array([100.0, 100.2, 100.1, 100.0, 99.8])
    handle_down = np.linspace(99.8, 95, 10)
    handle_up = np.linspace(95, 98, 10)
    post_handle_breakout = np.linspace(98, 103, 5)
    post_breakout = np.ones(10) * 103

    synthetic_prices = np.concatenate([
        rise_to_left_rim, descent, bottom, ascent, right_rim,
        handle_down, handle_up, post_handle_breakout, post_breakout
    ])

    # Generate synthetic volume data:
    # Decline: low volume; Recovery: up-day heavy; Breakout: spike
    n = len(synthetic_prices)
    synthetic_volumes = np.random.uniform(800000, 1200000, n)
    # Breakout candles get a volume spike
    synthetic_volumes[-15:] = np.random.uniform(2000000, 3000000, 15)

    print(f"  Synthetic series length : {len(synthetic_prices)} candles")
    print(f"  Price range             : ₹{synthetic_prices.min():.1f}"
          f" – ₹{synthetic_prices.max():.1f}")
    print(f"  Expected                : Cup drop ≈ 20%, Handle dip ≈ ₹5")
    print(f"  Geometry                : Cup depth = ₹20, "
          f"Max handle = ₹{0.32 * 20:.1f}, Actual = ₹5 → PASS")

    results = detect_cup_and_handle(
        synthetic_prices, volumes=synthetic_volumes,
        ticker="SYNTHETIC_CUP", verbose=True
    )

    if results:
        print(f"\n  ✅ PASS — {len(results)} pattern(s) detected (expected ≥ 1)")
        for idx, pat in enumerate(results, 1):
            print_pattern(pat, idx)
    else:
        print("\n  ❌ FAIL — No pattern detected on textbook cup shape!")
        print("  (This means the detection parameters may need tuning.)")

    # ── Test 2: Flat / Random series (should NOT detect anything) ────────
    print("\n  TEST 2: Flat/random series (should find NOTHING)")
    print("  " + "-" * 50)

    np.random.seed(42)
    flat_prices = 100 + np.random.normal(0, 0.5, 200)

    print(f"  Flat series length : {len(flat_prices)} candles")
    print(f"  Price range        : ₹{flat_prices.min():.1f}"
          f" – ₹{flat_prices.max():.1f}")

    results_flat = detect_cup_and_handle(flat_prices, ticker="FLAT_NOISE")

    if not results_flat:
        print("\n  ✅ PASS — No patterns detected (correct for random data)")
    else:
        print(f"\n  ⚠️  {len(results_flat)} false positive(s) detected.")

    # ── Test 3: V-shaped recovery (should REJECT — roundedness) ──────────
    print("\n  TEST 3: V-shaped recovery (should REJECT — no rounded bottom)")
    print("  " + "-" * 50)

    # Sharp V: steep descent, no basing, steep ascent (30% drop)
    v_rise = np.linspace(85, 100, 15)
    v_descent = np.linspace(100, 70, 15)     # fast down to 70
    v_bottom = np.array([70.0])              # single candle at bottom
    v_ascent = np.linspace(70, 100, 15)      # fast up to 100
    v_rim = np.array([100.0, 100.2, 100.1])
    v_handle = np.linspace(100, 96, 8)
    v_handle_up = np.linspace(96, 103, 10)
    v_post = np.ones(10) * 103

    v_prices = np.concatenate([v_rise, v_descent, v_bottom, v_ascent,
                                v_rim, v_handle, v_handle_up, v_post])

    print(f"  Series length : {len(v_prices)} candles")
    print(f"  Shape         : Sharp V (1 candle at bottom)")

    results_v = detect_cup_and_handle(v_prices, ticker="V_SHAPE", verbose=True)

    if not results_v:
        print("\n  ✅ PASS — V-shape correctly REJECTED")
    else:
        print(f"\n  ⚠️  {len(results_v)} pattern(s) found (check roundedness filter)")

    # ── Test 4: Handle Too Deep (should REJECT — geometry) ───────────────
    print("\n  TEST 4: Handle dips TOO DEEP (should be REJECTED by 0.32× rule)")
    print("  " + "-" * 50)

    deep_handle_down = np.linspace(99.8, 90, 10)
    deep_handle_up = np.linspace(90, 95, 10)
    deep_post_breakout = np.linspace(95, 103, 5)
    deep_post = np.ones(10) * 103

    deep_handle_prices = np.concatenate([
        rise_to_left_rim, descent, bottom, ascent, right_rim,
        deep_handle_down, deep_handle_up, deep_post_breakout, deep_post
    ])

    print(f"  Synthetic series length : {len(deep_handle_prices)} candles")
    print(f"  Handle dips to          : ₹90")
    print(f"  Geometry                : Cup depth = ₹20, "
          f"Max handle = ₹{0.32 * 20:.1f}, Actual = ₹10 → REJECT")

    results_deep = detect_cup_and_handle(
        deep_handle_prices, ticker="DEEP_HANDLE"
    )

    if not results_deep:
        print("\n  ✅ PASS — Pattern correctly REJECTED "
              "(handle exceeded 0.32× cup depth)")
    else:
        print(f"\n  ❌ FAIL — {len(results_deep)} pattern(s) detected but "
              f"should have been rejected!")

    print("\n" + "=" * 70)
    print("  🧪 SELF-TEST COMPLETE")
    print("=" * 70 + "\n")


# =============================================================================
#  8. HISTORICAL BACKTEST — Full-sweep mode
# =============================================================================

def backtest_historical(tickers=None, period="2y", start=None, end=None, interval="1d"):
    """
    MODE B: HISTORICAL BACKTEST
    ===========================
    Downloads historical data and sweeps the ENTIRE timeline for
    cup-and-handle patterns. Prints full debug output for each candidate,
    ranks valid patterns by quality score.

    Results are saved to 'backtest_results.txt'.
    """
    label = f"Interval: {interval}, Period: {period or 'Custom Date Range'}"

    print("\n" + "=" * 70)
    print("  📊 HISTORICAL BACKTEST  (Full Timeline Sweep)")
    print("=" * 70)
    print(f"  Data       : {label}")
    print(f"  Smoothing  : SMA({SMOOTHING_WINDOW})")
    print(f"  Filter     : NONE — every valid pattern is printed")
    print(f"  Geometry   : Handle ≤ {HANDLE_MAX_RETRACE_RATIO * 100:.0f}% "
          f"of cup depth\n")

    if tickers is None:
        print("  Fetching Nifty watchlist ...\n")
        tickers = get_nifty_list()

    print(f"  Scanning {len(tickers)} tickers over {label}\n")

    all_data = fetch_batch_data(
        tickers, period=period, start=start, end=end, interval=interval
    )

    if not all_data:
        print("  ⚠ No data downloaded. Check your internet connection.\n")
        return []

    all_patterns = []
    tickers_scanned = 0
    tickers_with_patterns = 0

    for ticker, df in all_data.items():
        tickers_scanned += 1

        close_prices = df["Close"].dropna().values
        high_prices  = df["High"].reindex(df["Close"].dropna().index).values
        dates = df["Close"].dropna().index

        # Volume data (may have NaN — fill with 0)
        if "Volume" in df.columns:
            vol_data = df["Volume"].reindex(df["Close"].dropna().index).fillna(0).values
        else:
            vol_data = None

        if len(close_prices) < 30:
            continue

        is_verbose = (len(all_data) <= 5)

        patterns = detect_cup_and_handle(
            close_prices, highs=high_prices, volumes=vol_data,
            ticker=ticker, dates=dates, interval=interval, verbose=is_verbose
        )

        if patterns:
            tickers_with_patterns += 1
            print(f"  ✓ {ticker}: {len(patterns)} pattern(s) found")
        all_patterns.extend(patterns)

    # Sort all patterns by quality score
    all_patterns.sort(key=lambda p: p.get("quality_score", 0), reverse=True)

    # Helper for clean dates
    def clean_date(d):
        return str(d).split(' ')[0] if ' ' in str(d) else str(d)

    # ── Write results file ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(script_dir, "backtest_results.txt")

    with open(results_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  📊 HISTORICAL BACKTEST RESULTS (v2.0)\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Run Date       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Data           : {label}\n")
        f.write(f"  Smoothing      : SMA({SMOOTHING_WINDOW})\n")
        f.write(f"  Tickers scanned: {tickers_scanned}\n")
        f.write(f"  Patterns found : {len(all_patterns)}\n")
        f.write(f"  Geometry       : Handle ≤ {HANDLE_MAX_RETRACE_RATIO * 100:.0f}% "
                f"of cup depth\n")
        f.write("=" * 70 + "\n\n")

        if all_patterns:
            # Summary Table
            f.write("─" * 70 + "\n")
            f.write("  SUMMARY TABLE (ranked by Quality Score)\n")
            f.write("─" * 70 + "\n\n")

            df_rows = []
            for p in all_patterns:
                df_rows.append({
                    "#": len(df_rows) + 1,
                    "Ticker": p["ticker"],
                    "Score": p.get("quality_score", 0),
                    "Left Rim Date": clean_date(p.get("left_rim_date", "N/A")),
                    "Left Rim": p["left_rim_price"],
                    "Bottom Date": clean_date(p.get("cup_bottom_date", "N/A")),
                    "Bottom": p["cup_bottom_price"],
                    "Right Rim Date": clean_date(p.get("right_rim_date", "N/A")),
                    "Right Rim": p["right_rim_price"],
                    "Handle Date": clean_date(p.get("handle_low_date", "N/A")),
                    "Handle Low": p["handle_low_price"],
                    "Cup Drop %": p["cup_drop_pct"],
                    "Handle Dip %": p["handle_pullback_pct"],
                })

            df_out = pd.DataFrame(df_rows)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            f.write(df_out.to_string(index=False))
            f.write("\n\n")

            # Detailed Pattern Blocks
            f.write("─" * 70 + "\n")
            f.write("  DETAILED PATTERN BREAKDOWN\n")
            f.write("─" * 70 + "\n")

            for idx, p in enumerate(all_patterns, 1):
                f.write(f"\n  {'═' * 60}\n")
                f.write(f"  🏆 PATTERN #{idx}  —  {p['ticker']}  "
                        f"(Score: {p.get('quality_score', 0)})\n")
                f.write(f"  {'═' * 60}\n")
                f.write(f"  Left Rim  : ₹{p['left_rim_price']}  "
                        f"({clean_date(p.get('left_rim_date', 'N/A'))})\n")
                f.write(f"  Cup Bottom: ₹{p['cup_bottom_price']}  "
                        f"({clean_date(p.get('cup_bottom_date', 'N/A'))})\n")
                f.write(f"  Right Rim : ₹{p['right_rim_price']}  "
                        f"({clean_date(p.get('right_rim_date', 'N/A'))})\n")
                f.write(f"  Handle Low: ₹{p['handle_low_price']}  "
                        f"({clean_date(p.get('handle_low_date', 'N/A'))})\n")
                f.write(f"  ─────────────────────────────────────────────────\n")
                f.write(f"  Cup Drop: {p['cup_drop_pct']}% | "
                        f"Recovery Gap: {p['recovery_pct']}% | "
                        f"Handle Dip: {p['handle_pullback_pct']}%\n")
                f.write(f"  ─────────────────────────────────────────────────\n")
                f.write(f"  Geometry: Cup Depth ₹{p['cup_depth']} | "
                        f"Max Handle ₹{p['max_handle_dip']} | "
                        f"Actual ₹{p['handle_pullback']}\n")
                f.write(f"  Roundedness: {p.get('roundedness_pct', 0)}% "
                        f"({p.get('base_candles', 0)}/{p.get('cup_width', 0)} candles)\n")
                f.write(f"  Pause: {p.get('pause_duration', 0)} candles | "
                        f"Slope: {p.get('handle_slope', 0)}\n")

                # Volume info
                if p.get('vol_decline_pass') is not None:
                    vd = "PASS" if p['vol_decline_pass'] else "WARN"
                    vr = "PASS" if p.get('vol_recovery_pass') else "WARN"
                    vb = "PASS" if p.get('vol_breakout_pass') else "WARN"
                    f.write(f"  Volume: Decline={vd} | "
                            f"Recovery ratio={p.get('vol_recovery_ratio', 0):.2f} ({vr}) | "
                            f"Breakout={vb}\n")

                f.write(f"  Smoothing: {p.get('smoothing_method', 'None')}\n")
                f.write(f"  {'═' * 60}\n")

        else:
            f.write("  No patterns found in the historical data.\n")
            f.write("  This is normal — cup-and-handle patterns are relatively rare.\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("  📊 BACKTEST COMPLETE\n")
        f.write("=" * 70 + "\n")

    # ── Terminal summary ──
    print(f"\n  {'─' * 60}")
    print(f"  📊 BACKTEST RESULTS")
    print(f"  {'─' * 60}")
    print(f"  Tickers scanned       : {tickers_scanned}")
    print(f"  Tickers with patterns : {tickers_with_patterns}")
    print(f"  Total patterns found  : {len(all_patterns)}")

    if all_patterns:
        print(f"\n  {'─' * 60}")
        print(f"  SUMMARY TABLE (ranked by Quality Score)")
        print(f"  {'─' * 60}\n")

        header = (f"  {'#':>3}  {'Ticker':<16} {'Score':>6} {'Left Rim':>12} "
                  f"{'Bottom':>12} {'Right Rim':>12} {'Handle':>12} "
                  f"{'Drop%':>7} {'Hdl%':>6}")
        print(header)
        print("  " + "─" * (len(header) - 2))

        for idx, p in enumerate(all_patterns, 1):
            print(
                f"  {idx:>3}  {p['ticker']:<16} "
                f"{p.get('quality_score', 0):>6} "
                f"{p['left_rim_price']:>12} "
                f"{p['cup_bottom_price']:>12} "
                f"{p['right_rim_price']:>12} "
                f"{p['handle_low_price']:>12} "
                f"{p['cup_drop_pct']:>6}% "
                f"{p['handle_pullback_pct']:>5}%"
            )
        print()
    else:
        print("\n  No patterns found in the historical data.")
        print("  This is normal — cup-and-handle patterns are relatively rare.\n")

    print(f"  📄 Full results saved to: {results_file}")
    print()
    print("=" * 70)
    print("  📊 BACKTEST COMPLETE")
    print("=" * 70 + "\n")

    return all_patterns


# =============================================================================
#  9. LIVE SCANNER — Continuous scanning during NSE market hours
# =============================================================================

def _get_ist_now():
    """Returns the current datetime in IST (Indian Standard Time)."""
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    return utc_now + IST_OFFSET


def is_market_open():
    """
    Checks if the NSE is currently open.
    NSE trading hours: Monday–Friday, 9:15 AM – 3:30 PM IST.

    Returns
    -------
    tuple(bool, str)
        (True/False, human-readable status message)
    """
    ist_now = _get_ist_now()
    day_of_week = ist_now.weekday()
    current_time = ist_now.time()

    market_open = datetime.time(9, 15)
    market_close = datetime.time(15, 30)

    if day_of_week >= 5:
        return False, f"Weekend (IST: {ist_now.strftime('%A %H:%M')})"
    if current_time < market_open:
        return False, (f"Pre-market (IST: {ist_now.strftime('%H:%M')},"
                       f" opens at 09:15)")
    if current_time > market_close:
        return False, (f"After-hours (IST: {ist_now.strftime('%H:%M')},"
                       f" closed at 15:30)")

    return True, f"Market OPEN (IST: {ist_now.strftime('%H:%M')})"


def _reset_dedup_cache_if_new_day():
    """Clears the dedup cache when the IST date rolls over."""
    global _live_alerted_today, _live_alert_date
    today_ist = _get_ist_now().date()
    if _live_alert_date != today_ist:
        _live_alerted_today = set()
        _live_alert_date = today_ist


def _is_pattern_from_today(pattern):
    """Returns True if the pattern's handle low date falls on today (IST)."""
    if "handle_low_date" not in pattern:
        return False
    try:
        handle_date = pd.Timestamp(pattern["handle_low_date"])
        today_ist = _get_ist_now().date()
        return handle_date.date() == today_ist
    except Exception:
        return False


def scan_watchlist(tickers, period=None, start=None, end=None, interval="15m"):
    """
    One-shot live scan: downloads latest data, runs detection,
    filters for patterns completing TODAY only with deduplication.
    """
    global _live_alerted_today

    _reset_dedup_cache_if_new_day()

    all_data = fetch_batch_data(
        tickers, period=period, start=start, end=end, interval=interval
    )

    new_alerts = []

    for ticker, df in all_data.items():
        if ticker in _live_alerted_today:
            continue

        close_prices = df["Close"].dropna().values
        high_prices  = df["High"].reindex(df["Close"].dropna().index).values
        dates = df["Close"].dropna().index

        if "Volume" in df.columns:
            vol_data = df["Volume"].reindex(df["Close"].dropna().index).fillna(0).values
        else:
            vol_data = None

        if len(close_prices) < 50:
            continue

        patterns = detect_cup_and_handle(
            close_prices, highs=high_prices, volumes=vol_data,
            ticker=ticker, dates=dates, interval=interval
        )

        for pat in patterns:
            if _is_pattern_from_today(pat):
                _live_alerted_today.add(ticker)
                new_alerts.append(pat)
                break

    return new_alerts


def run_scheduler(tickers=None, period="59d", start=None, end=None, interval="15m"):
    """
    MODE C: CONTINUOUS LIVE SCANNER
    ================================
    Scans the watchlist every SCAN_INTERVAL_MINUTES during NSE market hours.

    • During market hours: scans for patterns completing TODAY.
    • Deduplication: each ticker alerts at most ONCE per day.
    • Outside market hours: prints a "waiting" message and sleeps.
    • Press Ctrl+C at any time to exit cleanly.
    """
    label = f"Interval: {interval}, Period: {period or 'Custom Date Range'}"

    print("\n" + "=" * 70)
    print("  🚀 LIVE SCANNER — Current-Day Patterns Only (v2.0)")
    print("=" * 70)
    print(f"  Data          : {label}")
    print(f"  Smoothing     : SMA({SMOOTHING_WINDOW})")
    print(f"  Scan interval : every {SCAN_INTERVAL_MINUTES} minutes")
    print(f"  Market hours  : Mon–Fri, 9:15 AM – 3:30 PM IST")
    print(f"  Dedup         : Each ticker alerts at most ONCE per day")
    print(f"  Geometry      : Handle ≤ {HANDLE_MAX_RETRACE_RATIO * 100:.0f}%"
          f" of cup depth")
    print(f"  Exit          : Press Ctrl+C\n")

    if tickers is None:
        tickers = get_nifty_list()

    scan_count = 0

    try:
        while True:
            market_open, status_msg = is_market_open()

            if not market_open:
                print(f"  ⏸  {status_msg} — sleeping 5 min before re-check ...")
                time.sleep(300)
                continue

            scan_count += 1
            print(f"\n  {'━' * 60}")
            print(f"  🔄 SCAN #{scan_count}  —  {status_msg}")
            print(f"  {'━' * 60}")
            print(f"  📋 Dedup cache: {len(_live_alerted_today)} ticker(s) "
                  f"already alerted today.\n")

            try:
                new_alerts = scan_watchlist(
                    tickers, period=period, start=start,
                    end=end, interval=interval
                )

                if new_alerts:
                    # Sort by quality score
                    new_alerts.sort(key=lambda p: p.get("quality_score", 0), reverse=True)
                    print(f"\n  🔔 {len(new_alerts)} NEW ALERT(S)!\n")
                    for idx, pat in enumerate(new_alerts, 1):
                        print_pattern(pat, idx)
                else:
                    print("  ─ No new patterns completing today "
                          "(or already alerted).\n")

            except Exception as e:
                print(f"\n  ⚠ Scan error (will retry next cycle): {e}\n")

            print(f"  ⏳ Next scan in {SCAN_INTERVAL_MINUTES} minutes ..."
                  f"  (Ctrl+C to stop)\n")
            time.sleep(SCAN_INTERVAL_MINUTES * 60)

    except KeyboardInterrupt:
        print("\n\n  👋 Scanner stopped by user (Ctrl+C). Goodbye!\n")


# =============================================================================
#  10. MAIN ENTRY POINT
# =============================================================================

def validate_yfinance_params(interval, period):
    """Validates that the interval+lookback combo is supported by yfinance."""
    intraday_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]

    if interval in intraday_intervals:
        if period:
            if period.endswith("y") or period.endswith("mo"):
                print(f"  ⚠ WARNING: yfinance only supports up to 60 days "
                      f"for {interval} intraday data.")
                print(f"  Capping lookback to '59d'.\n")
                return "59d"
            elif period.endswith("d"):
                try:
                    days = int(period[:-1])
                    if days >= 60:
                        print(f"  ⚠ WARNING: yfinance only supports up to 60 days "
                              f"for {interval} intraday data.")
                        print(f"  Capping lookback to '59d'.\n")
                        return "59d"
                except ValueError:
                    pass
    return period


def prompt_for_config(default_interval="1d", default_period="2y"):
    """Interactive prompt for candle interval and lookback period."""
    print("\n  [Configuration]")
    interval = input(f"  Enter candle interval (e.g., 15m, 30m, 1h, 1d, 1wk) "
                     f"[default {default_interval}]: ").strip()
    if not interval:
        interval = default_interval

    if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]:
        if default_period == "2y":
            default_period = "59d"

    period = input(f"  Enter lookback period (e.g., 59d, 6mo, 2y) "
                   f"[default {default_period}]: ").strip()
    if not period:
        period = default_period

    period = validate_yfinance_params(interval, period)
    return interval, period


def main():
    """
    Entry point. Choose a mode via command-line arguments or interactive menu.

    Usage
    -----
    python cup_and_handle_detector.py test
    python cup_and_handle_detector.py historical RELIANCE.NS TCS.NS --interval 1d --lookback 2y
    python cup_and_handle_detector.py historical --start-date 2024-01-01 --end-date 2024-06-01
    python cup_and_handle_detector.py live --interval 15m --lookback 59d
    python cup_and_handle_detector.py           # interactive menu
    """
    parser = argparse.ArgumentParser(
        description="Cup and Handle Pattern Detector v2.0 — NSE (India)"
    )
    parser.add_argument("mode", nargs="?", default="",
                        help="test, historical, live, or empty for interactive menu")
    parser.add_argument("tickers", nargs="*",
                        help="Optional specific tickers to scan in historical mode")
    parser.add_argument("--interval", "-i", type=str,
                        help="Candle interval (e.g., 15m, 1d)")
    parser.add_argument("--lookback", "-l", type=str,
                        help="Lookback period (e.g., 59d, 2y)")
    parser.add_argument("--start-date", "-s", type=str,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", "-e", type=str,
                        help="End date (YYYY-MM-DD)")

    args = parser.parse_args()

    print(r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     ☕  CUP & HANDLE  PATTERN  DETECTOR   v2.0               ║
    ║         Indian Stock Market (NSE)                            ║
    ║                                                              ║
    ║     Smoothing : SMA(5) for peak detection                    ║
    ║     Geometry  : Handle ≤ 32% of Cup Depth                    ║
    ║     Volume    : 3 confirmation checks                        ║
    ║     Scoring   : Composite quality ranking                    ║
    ║     Modes     : TEST  |  HISTORICAL  |  LIVE                 ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    mode = args.mode.lower()

    interval = args.interval
    period = args.lookback

    if interval and period:
        period = validate_yfinance_params(interval, period)

    if mode == "test":
        test_with_sample_data()
        return

    elif mode in ("historical", "backtest"):
        if not interval:
            interval = "1d"
        if not period and not args.start_date:
            period = "2y"

        period = validate_yfinance_params(interval, period)

        if args.tickers:
            specific_tickers = [t if t.endswith(".NS") else f"{t}.NS"
                                for t in args.tickers]
            backtest_historical(
                tickers=specific_tickers, period=period,
                start=args.start_date, end=args.end_date, interval=interval
            )
        else:
            backtest_historical(
                period=period, start=args.start_date,
                end=args.end_date, interval=interval
            )
        return

    elif mode in ("live", "live_scan"):
        if not interval:
            interval = "15m"
        if not period and not args.start_date:
            period = "59d"

        period = validate_yfinance_params(interval, period)
        run_scheduler(
            period=period, start=args.start_date,
            end=args.end_date, interval=interval
        )
        return

    elif mode != "":
        print(f"  Unknown mode: '{mode}'")
        print(f"  Valid modes: test, historical, live\n")
        return

    # ── Interactive menu ──
    print("  Choose a mode:\n")
    print("    [1]  🧪 Self-Test         — Run on synthetic data (instant, no internet)")
    print("    [2]  📊 Historical Backtest — Full timeline sweep")
    print("    [3]  🚀 Live Scanner       — Current-day patterns only")
    print("    [4]  📊 Backtest ONE       — Historical scan for a specific stock")
    print("    [Q]  Exit\n")

    choice = input("  Enter choice (1/2/3/4/Q): ").strip().lower()

    if choice == "1":
        test_with_sample_data()

    elif choice == "2":
        interval, period = prompt_for_config("1d", "2y")
        backtest_historical(period=period, interval=interval)

    elif choice == "3":
        interval, period = prompt_for_config("15m", "59d")
        run_scheduler(period=period, interval=interval)

    elif choice == "4":
        ticker = input("  Enter ticker (e.g., RELIANCE or RELIANCE.NS): ").strip()
        if not ticker:
            print("  No ticker entered. Exiting.\n")
            return
        if not ticker.endswith(".NS"):
            ticker += ".NS"

        interval, period = prompt_for_config("1d", "2y")
        backtest_historical(tickers=[ticker], period=period, interval=interval)

    elif choice in ("q", "quit", "exit"):
        print("  Bye! 👋\n")

    else:
        print(f"  Invalid choice: '{choice}'\n")


# ── Run the script ──
if __name__ == "__main__":
    main()
