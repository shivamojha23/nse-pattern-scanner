"""
=============================================================================
  CUP AND HANDLE PATTERN DETECTOR — Indian Stock Market (NSE)
=============================================================================

  Author : AI-assisted project
  Purpose: Detect "Cup and Handle" chart patterns on NSE stocks using
           Yahoo Finance data with verified mathematical geometry filters.

  Two Primary Modes  (toggle via RUN_MODE at the top)
  ---------------------------------------------------
  1. LIVE_SCAN   → Scans for patterns completing TODAY. Deduplicates alerts
                   so each ticker fires at most once per session-day.
  2. HISTORICAL  → Sweeps the full timeline (up to 2 years daily candles).
                   Prints EVERY valid historical pattern with no date filter
                   and no output cap.

  Additionally:
  3. Self-Test   → Synthetic data to prove the math works (no internet).

  Verified Geometry
  -----------------
      Cup Depth      = Left_Rim_Price − Cup_Bottom_Price
      Max Handle Dip = 0.32 × Cup Depth
      Rule: handle pullback (Right Rim − Handle Low) must be ≤ Max Handle Dip.

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
  2. Price DROPS significantly (the "cup" — 10-30% decline).
  3. Price RECOVERS back near the original high (Right Rim, within 3%).
  4. Price dips SLIGHTLY again (the "handle" — must stay within 32% of
     cup depth).
  5. A breakout above the rim signals a potential upward move.

=============================================================================
"""

# ─── IMPORTS ────────────────────────────────────────────────────────────────
import sys
import os
import time
import datetime
import warnings
import argparse

# Fix Windows console encoding so emoji and special characters display correctly.
# Without this, Windows cmd.exe (which uses cp1252) crashes on characters like ☕ or ═.
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
#  MODE SELECTION  —  Change this ONE variable to switch behaviour
# =============================================================================
# "LIVE"       → Only prints patterns completing TODAY (IST).
#                Deduplicates: each ticker alerts at most once per day.
# "HISTORICAL" → Sweeps the full date range. Prints EVERY valid pattern
#                found across the entire timeline. No date filter, no cap.
RUN_MODE = "LIVE"


# ─── PER-MODE DATA CONFIGURATION ──────────────────────────────────────────
# These control what yfinance downloads for each mode.
# LIVE       : 15-minute intraday candles over the last ~59 days.
# HISTORICAL : Daily candles over the last 2 years (yfinance intraday limit
#              is ~60 days, so daily is required for longer lookbacks).
MODE_CONFIG = {
    "LIVE": {
        "period":   "59d",
        "interval": "15m",
        "label":    "15-min intraday (last 59 days)",
    },
    "HISTORICAL": {
        "period":   "2y",
        "interval": "1d",
        "label":    "Daily candles (last 2 years)",
    },
}


# ─── GENERAL CONFIGURATION ────────────────────────────────────────────────

# How many tickers to download in one yf.download() call.
# Larger = faster, but too large may get throttled by Yahoo.
BATCH_SIZE = 25

# Seconds to pause between download batches (avoids rate-limiting).
BATCH_SLEEP = 2

# How often (in minutes) the live scanner re-runs.
SCAN_INTERVAL_MINUTES = 15

# Indian Standard Time offset from UTC (UTC + 5:30).
IST_OFFSET = datetime.timedelta(hours=5, minutes=30)


# ─── DETECTION PARAMETERS ─────────────────────────────────────────────────
# These are tuneable knobs that control pattern sensitivity.

# Minimum cup drop from Left Rim to Cup Bottom (percentage).
MIN_CUP_DROP_PCT = 10

# Maximum cup drop from Left Rim to Cup Bottom (percentage).
MAX_CUP_DROP_PCT = 35

# Maximum gap between Right Rim and Left Rim (percentage).
# 0% = identical height; 3% allows a small difference.
MAX_RECOVERY_GAP_PCT = 3

# Handle geometry: the handle pullback must be ≤ this fraction of cup depth.
# 0.32 means the handle can retrace at most 32% of the cup's height.
HANDLE_MAX_RETRACE_RATIO = 0.32

# Minimum number of candles for each side of the cup (left-to-bottom, bottom-to-right).
MIN_CUP_CANDLES = 10

# Minimum number of candles for the handle pullback phase.
MIN_HANDLE_CANDLES = 5

# Maximum allowed average deviation (in %) for the right rim peak stability check.
RIGHT_RIM_STABILITY_PCT = 3.0


# ─── NEW DETECTION PARAMETERS (Bug Fixes) ─────────────────────────────────
# For Bug 1 (Bottom Roundedness)
# Percentage above cup bottom to be considered the "base zone"
BASE_ZONE_PCT = 0.05
# Minimum percentage of cup candles that must be in the base zone
MIN_BASE_CANDLES_PCT = 0.20

# For Bug 2 (Handle Low Lock)
# Consecutive candles closing above the right rim required to confirm breakout
BREAKOUT_CONFIRM_CANDLES = 3
# Maximum candles to look forward for a breakout before giving up on the handle
MAX_HANDLE_LOOKFORWARD_CANDLES = 30

# For Bug 3 (Genuine Pause Before Breakout)
# Minimum number of candles the price must pause/consolidate before breaking out
MIN_PAUSE_CANDLES = 5
# For Bug 5 (Discontinuity / Smoothness)
# Maximum allowed single-day percentage change inside the cup (to prevent gappy/jagged cups)
MAX_DISCONTINUITY_PCT = 0.08


# ─── LIVE-MODE DEDUPLICATION CACHE ─────────────────────────────────────────
# Tracks tickers that already generated a live alert TODAY so they don't
# print repeatedly on each scan loop. Auto-clears when the date changes.
_live_alerted_today = set()
_live_alert_date = None           # the IST date the cache belongs to


# =============================================================================
#  1.  GET THE WATCHLIST  — Dynamically fetch Nifty 100/200 tickers
# =============================================================================

def get_nifty_list():
    """
    Downloads the Nifty 200 stock list from a public CSV hosted by
    the NSE India index provider.

    Returns
    -------
    list[str]
        Ticker symbols with ".NS" suffix (e.g., ["RELIANCE.NS", "TCS.NS"]).
        Falls back to a curated Nifty 50 list if the download fails.

    How it works (step by step)
    ---------------------------
    1. We try multiple public CSV URLs that contain the Nifty index
       constituents. These are plain-text files anyone can download.
    2. We read them with pandas (pd.read_csv), which turns the CSV rows
       into a DataFrame (think: an Excel table in Python).
    3. We look for a column named "Symbol" (the ticker code) and append
       ".NS" so Yahoo Finance knows it's an NSE stock.
    4. If ALL URLs fail (network down, URL changed, etc.), we fall back
       to a hardcoded list of major Nifty 50 stocks so the script
       never crashes just because a CSV link broke.
    """
    # Multiple source URLs — if one breaks, the next is tried.
    csv_urls = [
        # Nifty 200 from NSE India index data
        "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
        # Nifty 100 (smaller but still good coverage)
        "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
        # Nifty 50 (smallest, most liquid stocks)
        "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    ]

    for url in csv_urls:
        try:
            print(f"  ↳ Trying to fetch watchlist from:\n    {url}")
            # pd.read_csv downloads the file and turns it into a table.
            df = pd.read_csv(url)

            # The CSV has a column called "Symbol" with values like
            # "RELIANCE", "TCS", etc. We need to add ".NS" for Yahoo Finance.
            if "Symbol" in df.columns:
                symbols = [f"{sym.strip()}.NS" for sym in df["Symbol"].tolist()]
                print(f"  ✓ Loaded {len(symbols)} tickers from NSE index CSV.\n")
                return symbols
            else:
                print(f"  ✗ CSV downloaded but 'Symbol' column not found.")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    # ── FALLBACK ── hardcoded Nifty 50 (so the script still works offline) ──
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
#  2.  FETCH DATA IN BATCHES  — Download candle data from Yahoo Finance
# =============================================================================

def fetch_batch_data(tickers, period=None, start=None, end=None, interval="15m"):
    """
    Downloads candle data for many tickers in batches.

    Parameters
    ----------
    tickers : list[str]
        e.g., ["RELIANCE.NS", "TCS.NS", ...]
    period : str
        How far back to look.
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD).
    interval : str
        Candle size.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are ticker symbols; values are DataFrames with columns
        like Open, High, Low, Close, Volume.

    How it works
    ------------
    • yf.download() can accept a list of tickers (space-separated string).
    • We split the full list into chunks of BATCH_SIZE (default 25).
    • Between chunks, we pause (time.sleep) so Yahoo doesn't block us.
    • The result is a multi-level DataFrame; we split it per ticker.
    """
    all_data = {}
    total = len(tickers)

    # Minimum candle count to consider data usable.
    # For daily candles a cup can form in fewer bars, so lower the bar.
    min_candles = 30 if interval == "1d" else 50

    for i in range(0, total, BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        batch_str = " ".join(batch)
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  📦 Batch {batch_num}/{total_batches}  "
              f"({len(batch)} tickers) ... ", end="", flush=True)

        try:
            # Download data for the entire batch at once.
            # group_by="ticker" means the result is organized per stock.
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
                # ── Single ticker vs. multiple tickers ──
                # yfinance now returns a MultiIndex even for a single ticker
                # if group_by='ticker' is used.
                for ticker in batch:
                    try:
                        # .xs (cross-section) extracts one ticker's data.
                        # Sometimes yfinance might NOT return a multi-index if there's only 1 ticker
                        # and group_by is ignored in older versions, so we handle both.
                        if isinstance(data.columns, pd.MultiIndex):
                            ticker_df = data.xs(ticker, level="Ticker", axis=1)
                        else:
                            ticker_df = data
                            
                        if not ticker_df.empty and len(ticker_df) > min_candles:
                            all_data[ticker] = ticker_df.copy()
                    except (KeyError, TypeError):
                        pass   # ticker had no data — skip silently

                print(f"OK  ({len(all_data)} tickers so far)")

        except Exception as e:
            print(f"error: {e}")

        # Pause between batches to be polite to Yahoo's servers.
        if i + BATCH_SIZE < total:
            time.sleep(BATCH_SLEEP)

    print(f"\n  ✓ Successfully fetched data for {len(all_data)} tickers.\n")
    return all_data


# =============================================================================
#  3.  PATTERN DETECTION  — Find Cup and Handle shapes
# =============================================================================

def detect_cup_and_handle(prices, ticker="UNKNOWN", dates=None, interval="15m", highs=None, verbose=False):
    """
    Scans a 1-D array of closing prices for Cup-and-Handle patterns
    using verified mathematical geometry filters.

    Includes overlap-deduplication: if multiple peak-combinations produce
    patterns that share the same cup-bottom or right-rim region, only the
    BEST one (deepest cup with valid handle) is kept.

    Parameters
    ----------
    prices : array-like
        Sequence of closing prices (e.g., 15-min candles or daily candles).
    ticker : str
        Stock symbol (used in print messages only).
    dates : array-like or None
        Corresponding timestamps for each price (for date reporting).

    Returns
    -------
    list[dict]
        Each dict describes one detected pattern with fields:
        - ticker, left_rim_price, cup_bottom_price, right_rim_price,
          handle_low_price, cup_drop_pct, recovery_pct,
          cup_depth, max_handle_dip, handle_pullback,
          handle_pullback_pct,
          left_rim_idx, cup_bottom_idx, right_rim_idx, handle_low_idx,
          (and dates if provided).

    Algorithm (plain English)
    -------------------------
    1. Convert prices to a NumPy array (fast math).
    2. Use find_peaks() to locate LOCAL HIGHS (peaks) and LOCAL LOWS (troughs).
    3. For each PAIR of peaks (potential Left Rim → Right Rim):
       a. Find the deepest trough between them → Cup Bottom.
       b. Cup Symmetry: bottom must sit in the middle 70% of the cup width
          (not hugging either rim — that's a V-shape, not a U-shape).
       c. Check: is the cup deep enough? (10-30% drop from Left Rim)
       d. Check: did price recover? (Right Rim within 3% of Left Rim)
       e. Look for a mild pullback AFTER the Right Rim → Handle.
       f. GEOMETRY CHECK (verified formula):
            cup_depth      = Left_Rim_Price − Cup_Bottom_Price
            max_handle_dip = 0.32 × cup_depth
            handle_pullback = Right_Rim_Price − Handle_Low_Price
            Rule: handle_pullback must be ≥ 1% of Right Rim AND ≤ max_handle_dip.
    4. Overlap deduplication: if two patterns share similar cup-bottom and
       right-rim indices (within 10 candles), keep only the one with the
       deeper cup — it's the "cleanest" version of the same pattern.
    """
    prices = np.array(prices, dtype=float)

    # High prices for the strict Left Rim ceiling check.
    # If no OHLC High data is provided (e.g., synthetic tests), fall back
    # to Close prices — Close is always ≤ High, so the check still works
    # directionally; it just can't catch intraday spikes above the rim.
    if highs is not None:
        highs = np.array(highs, dtype=float)
    else:
        highs = prices  # fallback: treat Close as the High

    # Need enough candles to have a meaningful pattern.
    if len(prices) < 30:
        return []

    # ── Step 2a: Find local highs (peaks) ──
    # `distance` = peaks must be at least this many candles apart.
    # `prominence` = peak must stand out from its neighbors by at least 1%
    # of the median price (filters out tiny wiggles).
    min_prominence = np.median(prices) * 0.01
    peak_indices, _ = find_peaks(prices, distance=10, prominence=min_prominence)

    # ── Step 2b: Find local lows (troughs) ──
    # Trick: negate the prices, then find peaks on the negated version.
    trough_indices, _ = find_peaks(-prices, distance=10, prominence=min_prominence)

    # We need at least 2 peaks and 1 trough to form a cup.
    if len(peak_indices) < 2 or len(trough_indices) < 1:
        if verbose: print(f"DEBUG: Not enough peaks/troughs: peaks={peak_indices} troughs={trough_indices}")
        return []

    # Collect ALL candidate patterns first, then deduplicate.
    candidates = []

    # ── Step 3: Try every combination of (Left Rim, Right Rim) ──
    for i in range(len(peak_indices)):
        for j in range(i + 1, len(peak_indices)):
            left_rim_idx = peak_indices[i]
            right_rim_idx = peak_indices[j]
            left_rim_price = prices[left_rim_idx]
            right_rim_price = prices[right_rim_idx]
            if right_rim_price == 0:
                continue

            # ── Rule 1: Dynamic Trend Check (Adaptive Left Rim) ──
            # Determine if we're dealing with macro data (Daily/Weekly)
            is_macro = interval.endswith('d') or interval.endswith('w')
            
            if is_macro:
                # Macro mode: Left rim just needs to be the highest peak within a broader 20-candle window.
                # No absolute 10-candle positive return check, as macro trends have natural pullbacks.
                window_start = max(0, left_rim_idx - 20)
                window_end = min(len(prices), left_rim_idx + 21)
                if np.max(prices[window_start:window_end]) > left_rim_price:
                    continue
            else:
                # Intraday mode: Net positive return over the 10 candles preceding the Left Rim.
                if left_rim_idx < 10:
                    continue
                if prices[left_rim_idx - 10] >= left_rim_price:
                    continue
                    
                # Left Rim must be a true local maximum within a tight window of 5 candles before and after it.
                window_start = max(0, left_rim_idx - 5)
                window_end = min(len(prices), left_rim_idx + 6)
                if np.max(prices[window_start:window_end]) > left_rim_price:
                    continue

            # ── Width check ──
            # The cup should span a reasonable number of candles.
            cup_width = right_rim_idx - left_rim_idx
            if cup_width < 15 or cup_width > len(prices) * 0.8:
                continue   # too narrow or absurdly wide — skip

            # ── Step 3a: Find deepest trough between the two rims ──
            troughs_in_cup = [t for t in trough_indices
                              if left_rim_idx < t < right_rim_idx]
            if not troughs_in_cup:
                continue   # no trough between the rims — not a cup

            cup_bottom_idx = min(troughs_in_cup, key=lambda t: prices[t])
            cup_bottom_price = prices[cup_bottom_idx]

            # ── NEW RULE: Minimum Cup Duration ──
            # Both sides of the cup must be at least MIN_CUP_CANDLES.
            cup_left_duration = cup_bottom_idx - left_rim_idx
            cup_right_duration = right_rim_idx - cup_bottom_idx
            if cup_left_duration < MIN_CUP_CANDLES or cup_right_duration < MIN_CUP_CANDLES:
                continue   # a cup side was formed too quickly

            # ── NEW RULE: No Double-Dip ──
            # The price must not drop below the cup bottom between the bottom and right rim.
            # (In other words, the recovery must be clean).
            prices_after_bottom = prices[cup_bottom_idx:right_rim_idx + 1]
            if len(prices_after_bottom) > 0 and np.min(prices_after_bottom) < cup_bottom_price:
                if verbose:
                    print(f"  ❌ REJECTED {ticker}: Double dip detected below cup bottom.")
                continue   # double dip occurred

            # ── Step 3b-extra: Cup SYMMETRY check ──
            # The bottom should sit roughly in the middle of the cup,
            # not right next to one of the rims.
            # We require the bottom to be in the middle 70% of the cup width.
            # Example: if cup spans candles 10–110 (width=100), bottom must
            #          be between candle 25 and candle 95.
            left_margin = left_rim_idx + cup_width * 0.15
            right_margin = right_rim_idx - cup_width * 0.15
            if not (left_margin <= cup_bottom_idx <= right_margin):
                continue   # bottom hugs one rim — V-shape, not a cup


            # ── Step 3b: Cup depth check ──
            # Drop % = how much the price fell from the Left Rim to the Bottom.
            cup_drop_pct = ((left_rim_price - cup_bottom_price)
                            / left_rim_price) * 100

            if not (MIN_CUP_DROP_PCT <= cup_drop_pct <= MAX_CUP_DROP_PCT):
                continue   # drop too small or too large

            # ── Step 3c: Recovery check ──
            # Recovery % = how close the Right Rim is to the Left Rim.
            # 0% means identical height; negative means Right Rim is lower.
            recovery_pct = abs(right_rim_price - left_rim_price) / left_rim_price * 100

            if recovery_pct > MAX_RECOVERY_GAP_PCT:
                continue   # Right Rim is too far from Left Rim

            # ── Step 3d: Handle detection & BUG 2/3 FIX ──
            if right_rim_idx + 1 >= len(prices):
                continue
                
            # Calculate the OLD handle logic (first dip in 25% window) strictly for debug reporting
            old_handle_zone_end = min(right_rim_idx + max(1, cup_width // 4), len(prices) - 1)
            old_handle_prices = prices[right_rim_idx : old_handle_zone_end + 1]
            old_handle_low_price = np.min(old_handle_prices) if len(old_handle_prices) > 0 else right_rim_price

            # Scan forward to find true handle low
            current_handle_low = prices[right_rim_idx]
            current_handle_low_idx = right_rim_idx
            breakout_count = 0
            breakout_confirmed = False
            breakout_start_idx = -1
            
            for k in range(right_rim_idx + 1, min(len(prices), right_rim_idx + 1 + MAX_HANDLE_LOOKFORWARD_CANDLES)):
                curr_price = prices[k]
                
                # Update current handle low
                if curr_price < current_handle_low:
                    current_handle_low = curr_price
                    current_handle_low_idx = k
                    
                # Breakout check (closing above Right Rim)
                if curr_price > right_rim_price:
                    breakout_count += 1
                else:
                    breakout_count = 0  # must be consecutive
                    
                if breakout_count >= BREAKOUT_CONFIRM_CANDLES:
                    breakout_confirmed = True
                    breakout_start_idx = k - BREAKOUT_CONFIRM_CANDLES + 1
                    break

            handle_low_price = current_handle_low
            handle_low_idx = current_handle_low_idx
            
            # BUG 3 FIX: Pause Before Breakout check
            pause_duration = (breakout_start_idx - right_rim_idx) if breakout_confirmed else 0
            
            handle_slope = 0
            if breakout_confirmed and pause_duration > 1:
                # Linear regression slope of prices in the pause window
                pause_prices = prices[right_rim_idx : breakout_start_idx + 1]
                res = linregress(np.arange(len(pause_prices)), pause_prices)
                handle_slope = res.slope

            # ══════════════════════════════════════════════════════════
            # ── Step 3e: Handle validation tracking ──
            # ══════════════════════════════════════════════════════════
            
            # BUG 1 FIX: Bottom Roundedness Check (Calculate here for full pattern reporting)
            base_zone_max = cup_bottom_price * (1 + BASE_ZONE_PCT)
            cup_candles = prices[left_rim_idx:right_rim_idx]
            base_candles_count = np.sum(cup_candles <= base_zone_max)
            roundedness_pct = (base_candles_count / len(cup_candles)) if len(cup_candles) > 0 else 0

            cup_depth = left_rim_price - cup_bottom_price
            max_handle_dip = HANDLE_MAX_RETRACE_RATIO * cup_depth
            handle_pullback = right_rim_price - handle_low_price
            handle_pullback_pct = (handle_pullback / right_rim_price) * 100
            
            min_handle_pullback = right_rim_price * 0.01
            handle_duration = handle_low_idx - right_rim_idx
            
            # BUG 4 FIX: Intermediate High / Internal Close Violation Check
            rim_ceiling = highs[left_rim_idx]
            highs_inside_cup = highs[left_rim_idx + 1 : right_rim_idx]
            closes_inside_cup = prices[left_rim_idx + 1 : right_rim_idx]
            
            max_internal_high = np.max(highs_inside_cup) if len(highs_inside_cup) > 0 else 0
            max_internal_close = np.max(closes_inside_cup) if len(closes_inside_cup) > 0 else 0

            # BUG 5 FIX: Discontinuity Check
            cup_closes_for_diff = prices[left_rim_idx : right_rim_idx + 1]
            if len(cup_closes_for_diff) > 1:
                daily_returns = np.abs(np.diff(cup_closes_for_diff) / cup_closes_for_diff[:-1])
                max_daily_jump = np.max(daily_returns)
            else:
                max_daily_jump = 0
            
            is_valid = True
            reject_reason = ""
            
            # Evaluate rejection reasons in order of priority
            if max_internal_close > left_rim_price:
                is_valid = False
                reject_reason = f"Internal close (₹{max_internal_close:.2f}) exceeded Left Rim close (₹{left_rim_price:.2f}) — Structural Violation."
            elif max_internal_high > rim_ceiling:
                is_valid = False
                reject_reason = f"Internal high (₹{max_internal_high:.2f}) exceeded Left Rim High (₹{rim_ceiling:.2f}) — Structural Violation."
            elif max_daily_jump > MAX_DISCONTINUITY_PCT:
                is_valid = False
                reject_reason = f"Excessive price discontinuity ({max_daily_jump*100:.1f}% single-day jump > {MAX_DISCONTINUITY_PCT*100}% limit)."
            elif roundedness_pct < MIN_BASE_CANDLES_PCT:
                is_valid = False
                reject_reason = f"V-shape recovery. Bottom roundedness {base_candles_count}/{len(cup_candles)} ({roundedness_pct*100:.1f}%) < {MIN_BASE_CANDLES_PCT*100}% required."
            elif handle_pullback < min_handle_pullback:
                is_valid = False
                reject_reason = "Handle pullback too small (<1%)."
            elif not breakout_confirmed:
                is_valid = False
                reject_reason = f"Handle incomplete / no breakout confirmed within {MAX_HANDLE_LOOKFORWARD_CANDLES} candles."
            elif pause_duration < MIN_PAUSE_CANDLES:
                is_valid = False
                reject_reason = f"No real handle — immediate continuation (breakout in {pause_duration} candles < {MIN_PAUSE_CANDLES}), not Cup & Handle."
            elif handle_slope > 0:
                is_valid = False
                reject_reason = f"Handle is not a genuine pause (slope {handle_slope:.2f} > 0). Resumed upward momentum before breakout."
            elif handle_pullback > max_handle_dip:
                is_valid = False
                reject_reason = f"Corrected handle pullback (₹{handle_pullback:.2f}) exceeded 0.32× depth (₹{max_handle_dip:.2f})."
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

            # If invalid and we aren't verbose, drop it.
            if not is_valid and not verbose:
                continue

            # Scan forward to find true handle low
            current_handle_low = prices[right_rim_idx]
            current_handle_low_idx = right_rim_idx
            breakout_count = 0
            breakout_confirmed = False
            
            for k in range(right_rim_idx + 1, min(len(prices), right_rim_idx + 1 + MAX_HANDLE_LOOKFORWARD_CANDLES)):
                curr_price = prices[k]
                
                # Update current handle low
                if curr_price < current_handle_low:
                    current_handle_low = curr_price
                    current_handle_low_idx = k
                    
                # Breakout check (closing above Right Rim)
                if curr_price > right_rim_price:
                    breakout_count += 1
                else:
                    breakout_count = 0  # must be consecutive
                    
                if breakout_count >= BREAKOUT_CONFIRM_CANDLES:
                    breakout_confirmed = True
                    break

            handle_low_price = current_handle_low
            handle_low_idx = current_handle_low_idx




            # ════════════════ CANDIDATE PATTERN ════════════════
            pattern = {
                "ticker":              ticker,
                "left_rim_price":      round(left_rim_price, 2),
                "cup_bottom_price":    round(cup_bottom_price, 2),
                "right_rim_price":     round(right_rim_price, 2),
                "handle_low_price":    round(handle_low_price, 2),
                "cup_drop_pct":        round(cup_drop_pct, 2),
                "recovery_pct":        round(recovery_pct, 2),
                "cup_depth":           round(cup_depth, 2),
                "max_handle_dip":      round(max_handle_dip, 2),
                "handle_pullback":     round(handle_pullback, 2),
                "handle_pullback_pct": round(handle_pullback_pct, 2),
                "left_rim_idx":        int(left_rim_idx),
                "cup_bottom_idx":      int(cup_bottom_idx),
                "right_rim_idx":       int(right_rim_idx),
                "handle_low_idx":      int(handle_low_idx),
                "cup_left_duration":   int(cup_left_duration),
                "cup_right_duration":  int(cup_right_duration),
                "handle_duration":     int(handle_duration),
                "double_dip_passed":   True,
                "roundedness_pct":     round(roundedness_pct * 100, 1),
                "base_candles":        int(base_candles_count),
                "cup_width":           int(len(cup_candles)),
                "old_handle_low":      round(old_handle_low_price, 2),
                "breakout_confirmed":  breakout_confirmed,
                "pause_duration":      int(pause_duration),
                "handle_slope":        round(handle_slope, 4),
                "reject_reason":       reject_reason,
            }

            # Attach dates if available (for reporting).
            if dates is not None:
                pattern["left_rim_date"]   = str(dates[left_rim_idx])
                pattern["cup_bottom_date"] = str(dates[cup_bottom_idx])
                pattern["right_rim_date"]  = str(dates[right_rim_idx])
                pattern["handle_low_date"] = str(dates[handle_low_idx])

            # If we have a pattern that wasn't valid, but we are verbose, print it!
            if not is_valid and verbose:
                print_pattern(pattern, index="REJECTED")
                continue
                
            # Store valid candidates for deduplication
            candidates.append(pattern)

    # ══════════════════════════════════════════════════════════════════════
    # ── Step 4: OVERLAP DEDUPLICATION ──
    # ══════════════════════════════════════════════════════════════════════
    # Problem: the loop above tries EVERY (Left Rim, Right Rim) combo.
    # This means the same physical cup (same bottom, same general region)
    # can appear multiple times with slightly different rim peaks.
    #
    # Solution: if two patterns share a similar cup_bottom_idx AND a similar
    # right_rim_idx (within OVERLAP_TOLERANCE candles), they are the "same"
    # pattern. Keep only the one with the deepest cup (highest cup_drop_pct)
    # — it represents the clearest, most textbook version.
    # ──────────────────────────────────────────────────────────────────────

    OVERLAP_TOLERANCE = 10  # candles — two indices within this range = "same"

    # Sort candidates by cup_drop_pct descending (best/deepest first).
    candidates.sort(key=lambda p: p["cup_drop_pct"], reverse=True)

    patterns_found = []
    claimed_regions = []  # list of (cup_bottom_idx, right_rim_idx) already used

    for candidate in candidates:
        cb_idx = candidate["cup_bottom_idx"]
        rr_idx = candidate["right_rim_idx"]

        # Deduplication: check if this pattern overlaps with one already found.
        is_duplicate = False
        for claimed_cb, claimed_rr in claimed_regions:
            if (abs(cb_idx - claimed_cb) <= OVERLAP_TOLERANCE and
                    abs(rr_idx - claimed_rr) <= OVERLAP_TOLERANCE):
                is_duplicate = True
                break

        if not is_duplicate:
            patterns_found.append(candidate)
            claimed_regions.append((cb_idx, rr_idx))

    return patterns_found


# =============================================================================
#  4.  PRETTY PRINT  — Show pattern details in a human-readable way
# =============================================================================

def print_pattern(p, index=1):
    """
    Prints one detected pattern with all the 'WHY' details so a beginner
    can sanity-check it without knowing anything about stocks.

    Includes the verified geometry values so you can manually verify
    the 0.32× rule.
    """
    print(f"\n  {'═' * 60}")
    print(f"  🏆 PATTERN #{index}  —  {p['ticker']}")
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
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ GEOMETRY CHECK")
    print(f"    Cup Depth (₹)              : ₹{p['cup_depth']}")
    print(f"    Max Handle Dip Allowed (₹) : ₹{p['max_handle_dip']}  "
          f"(= 0.32 × ₹{p['cup_depth']})")
    print(f"    Actual Handle Pullback (₹) : ₹{p['handle_pullback']}  "
          f"({p['handle_pullback_pct']}%)")
    
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ ROUNDEDNESS CHECK")
    roundedness_pass = "✅ PASS" if p.get('roundedness_pct', 0) >= MIN_BASE_CANDLES_PCT * 100 else "❌ FAIL"
    print(f"    {p.get('base_candles', 'N/A')} candles in base zone out of {p.get('cup_width', 'N/A')} cup candles ({p.get('roundedness_pct', 'N/A')}%) — {roundedness_pass}")

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

    # Show dates if available.
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
        print(f"  FINAL VERDICT: ✅ VALID")

    print(f"  {'═' * 60}\n")


# =============================================================================
#  5.  SELF-TEST  — Synthetic data to prove the detection math works
# =============================================================================

def test_with_sample_data():
    """
    Creates FAKE price data shaped like a textbook Cup and Handle,
    runs detection on it, and verifies the result.

    Also tests rejection cases (flat data, downtrend, handle-too-deep)
    to confirm the geometry filter works correctly.

    This function needs ZERO internet and ZERO market data.
    It proves the detection algorithm is mathematically correct.

    How we build the synthetic cup shape
    -------------------------------------
    Think of the cup as a smooth U-shape:

       100 ─╮                              ╭─ 100  (Right Rim)
             ╲                            ╱
              ╲                          ╱
               ╲                        ╱
                ╰──── 80 ──────────────╯       (Cup Bottom = 20% drop)
                                               Then a handle: dips to ~95

    We use numpy to build this piece by piece:
    - Start high (100)             → Left Rim
    - Smoothly drop to 80          → Cup descent
    - Stay near 80 briefly         → Cup bottom
    - Smoothly rise back to 100    → Cup ascent / Right Rim
    - Dip to 95 then recover to 98 → Handle (within 32% of cup depth)
    """
    print("\n" + "=" * 70)
    print("  🧪 SELF-TEST : Synthetic Cup & Handle Verification")
    print("=" * 70)

    # ── Test 1: Textbook Cup and Handle (should PASS) ───────────────────
    print("\n  TEST 1: Textbook Cup & Handle shape")
    print("  " + "-" * 50)

    # Build each segment of the cup:
    # Key insight: find_peaks needs the price to RISE then FALL to see a peak.
    # A flat plateau is NOT a peak. So we build a realistic shape:
    #   rise → Left Rim peak → descent → bottom → ascent → Right Rim peak
    #   → handle dip → recovery

    # 1. Gradual rise from 90 to 100 (creates a clear Left Rim peak)
    rise_to_left_rim = np.linspace(90, 100, 15)

    # 2. Smooth descent from 100 → 80 (the left side of the cup)
    descent = np.linspace(100, 80, 30)

    # 3. Cup bottom: stay near 80 for a while (with tiny noise for realism)
    np.random.seed(123)  # reproducible
    bottom = np.ones(20) * 80 + np.random.uniform(-0.3, 0.3, 20)

    # 4. Smooth ascent from 80 → 100 (right side of the cup)
    ascent = np.linspace(80, 100, 30)

    # 5. Right Rim: brief peak at 100, then start the handle
    right_rim = np.array([100.0, 100.2, 100.1, 100.0, 99.8])

    # 6. Handle: dip to 95, then recover to 98
    #    Cup depth = 100 − 80 = 20.  Max handle dip = 0.32 × 20 = 6.4.
    #    Handle pullback = 100 − 95 = 5.  5 ≤ 6.4 → PASS ✅
    handle_down = np.linspace(99.8, 95, 10)
    handle_up = np.linspace(95, 98, 10)

    # 7. Post-handle (breakout above Right Rim to confirm the pattern)
    post_handle_breakout = np.linspace(98, 103, 5)
    post_breakout = np.ones(10) * 103

    # Concatenate all segments into one price series.
    synthetic_prices = np.concatenate([
        rise_to_left_rim, descent, bottom, ascent, right_rim,
        handle_down, handle_up, post_handle_breakout, post_breakout
    ])

    print(f"  Synthetic series length : {len(synthetic_prices)} candles")
    print(f"  Price range             : ₹{synthetic_prices.min():.1f}"
          f" – ₹{synthetic_prices.max():.1f}")
    print(f"  Expected                : Cup drop ≈ 20%, Handle dip ≈ ₹5")
    print(f"  Geometry                : Cup depth = ₹20, "
          f"Max handle = ₹{0.32 * 20:.1f}, Actual = ₹5 → PASS")

    # Run detection
    results = detect_cup_and_handle(synthetic_prices, ticker="SYNTHETIC_CUP", verbose=True)

    if results:
        print(f"\n  ✅ PASS — {len(results)} pattern(s) detected (expected ≥ 1)")
        for idx, pat in enumerate(results, 1):
            print_pattern(pat, idx)
    else:
        print("\n  ❌ FAIL — No pattern detected on textbook cup shape!")
        print("  (This means the detection parameters may need tuning.)")

    # ── Test 2: Flat / Random series (should NOT detect anything) ───────
    print("\n  TEST 2: Flat/random series (should find NOTHING)")
    print("  " + "-" * 50)

    # Create 200 candles of roughly flat prices with small noise.
    np.random.seed(42)  # reproducible randomness
    flat_prices = 100 + np.random.normal(0, 0.5, 200)  # very small wiggles

    print(f"  Flat series length : {len(flat_prices)} candles")
    print(f"  Price range        : ₹{flat_prices.min():.1f}"
          f" – ₹{flat_prices.max():.1f}")

    results_flat = detect_cup_and_handle(flat_prices, ticker="FLAT_NOISE")

    if not results_flat:
        print("\n  ✅ PASS — No patterns detected (correct for random data)")
    else:
        print(f"\n  ⚠️  {len(results_flat)} false positive(s) detected.")
        print("  This may be acceptable if the noise happened to form a shape.")

    # ── Test 3: Downtrend (should NOT detect) ───────────────────────────
    print("\n  TEST 3: Steady downtrend (should find NOTHING)")
    print("  " + "-" * 50)

    downtrend = np.linspace(150, 80, 200)  # just goes down, no cup shape

    results_down = detect_cup_and_handle(downtrend, ticker="DOWNTREND")

    if not results_down:
        print("  ✅ PASS — No patterns detected (correct for a straight decline)")
    else:
        print(f"  ⚠️  {len(results_down)} false positive(s) on downtrend.")

    # ── Test 4: Handle Too Deep — MUST BE REJECTED by geometry ──────────
    print("\n  TEST 4: Handle dips TOO DEEP (should be REJECTED by 0.32× rule)")
    print("  " + "-" * 50)

    # Same cup as Test 1, but the handle drops to ₹90 instead of ₹95.
    #   Cup depth = 100 − 80 = 20.  Max handle = 0.32 × 20 = 6.4.
    #   Handle pullback = 100 − 90 = 10.  10 > 6.4 → MUST REJECT ❌
    deep_handle_down = np.linspace(99.8, 90, 10)
    deep_handle_up = np.linspace(90, 95, 10)
    deep_post_handle_breakout = np.linspace(95, 103, 5)
    deep_post_breakout = np.ones(10) * 103

    deep_handle_prices = np.concatenate([
        rise_to_left_rim, descent, bottom, ascent, right_rim,
        deep_handle_down, deep_handle_up, deep_post_handle_breakout, deep_post_breakout
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
        for idx, pat in enumerate(results_deep, 1):
            print_pattern(pat, idx)

    print("\n" + "=" * 70)
    print("  🧪 SELF-TEST COMPLETE")
    print("=" * 70 + "\n")


# =============================================================================
#  6.  HISTORICAL BACKTEST  — Full-sweep mode (MODE B)
# =============================================================================

def run_historical_backtest(tickers=None, period="2y", start=None, end=None, interval="1d"):
    """
    MODE B: HISTORICAL BACKTEST
    ===========================
    Downloads historical data and sweeps the ENTIRE timeline for
    cup-and-handle patterns. No date restriction, no output cap.

    Results are saved to 'backtest_results.txt' in the script directory
    with full pattern details. A clean summary table is also printed
    to the terminal.

    Parameters
    ----------
    tickers : list[str] or None
        Specific tickers to scan. If None, fetches the full Nifty watchlist.
    """
    label = f"Interval: {interval}, Period: {period or 'Custom Date Range'}"

    print("\n" + "=" * 70)
    print("  📊 HISTORICAL BACKTEST  (Full Timeline Sweep)")
    print("=" * 70)
    print(f"  Data   : {label}")
    print(f"  Filter : NONE — every valid pattern is printed")
    print(f"  Geometry: Handle ≤ {HANDLE_MAX_RETRACE_RATIO * 100:.0f}% "
          f"of cup depth\n")

    # If no tickers specified, fetch the full Nifty watchlist.
    if tickers is None:
        print("  Fetching Nifty watchlist ...\n")
        tickers = get_nifty_list()

    print(f"  Scanning {len(tickers)} tickers over {label}\n")

    # Download all data in batches.
    all_data = fetch_batch_data(
        tickers, period=period, start=start, end=end, interval=interval
    )

    if not all_data:
        print("  ⚠ No data downloaded. Check your internet connection.\n")
        return []

    # ── Scan each ticker ──
    all_patterns = []
    tickers_scanned = 0
    tickers_with_patterns = 0

    for ticker, df in all_data.items():
        tickers_scanned += 1

        # Extract closing prices, high prices, and their timestamps.
        close_prices = df["Close"].dropna().values
        high_prices  = df["High"].reindex(df["Close"].dropna().index).values
        dates = df["Close"].dropna().index

        if len(close_prices) < 30:
            continue  # not enough data points

        # If there are only a few tickers, turn on verbose mode to show rejections
        is_verbose = (len(all_data) <= 5)

        # Full sweep — no date restriction.
        patterns = detect_cup_and_handle(
            close_prices, ticker=ticker, dates=dates,
            interval=interval, highs=high_prices, verbose=is_verbose
        )

        if patterns:
            tickers_with_patterns += 1
            print(f"  ✓ {ticker}: {len(patterns)} pattern(s) found")
        all_patterns.extend(patterns)

    # ── Build results file and terminal output ──
    # Helper to clean timestamps to date-only strings.
    def clean_date(d):
        return str(d).split(' ')[0] if ' ' in str(d) else str(d)

    # Determine the output file path (same directory as the script).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(script_dir, "backtest_results.txt")

    # ── Write the results file ──
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  📊 HISTORICAL BACKTEST RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Run Date       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Data           : {label}\n")
        f.write(f"  Tickers scanned: {tickers_scanned}\n")
        f.write(f"  Patterns found : {len(all_patterns)}\n")
        f.write(f"  Geometry       : Handle ≤ {HANDLE_MAX_RETRACE_RATIO * 100:.0f}% "
                f"of cup depth\n")
        f.write("=" * 70 + "\n\n")

        if all_patterns:
            # ── Summary Table ──
            f.write("─" * 70 + "\n")
            f.write("  SUMMARY TABLE\n")
            f.write("─" * 70 + "\n\n")

            df_rows = []
            for p in all_patterns:
                df_rows.append({
                    "#": len(df_rows) + 1,
                    "Ticker": p["ticker"],
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

            # ── Detailed Pattern Blocks ──
            f.write("─" * 70 + "\n")
            f.write("  DETAILED PATTERN BREAKDOWN\n")
            f.write("─" * 70 + "\n")

            for idx, p in enumerate(all_patterns, 1):
                f.write(f"\n  {'═' * 60}\n")
                f.write(f"  🏆 PATTERN #{idx}  —  {p['ticker']}\n")
                f.write(f"  {'═' * 60}\n")
                f.write(f"  Left Rim  (peak before cup)  : ₹{p['left_rim_price']}\n")
                f.write(f"  Cup Bottom (lowest point)    : ₹{p['cup_bottom_price']}\n")
                f.write(f"  Right Rim  (recovery peak)   : ₹{p['right_rim_price']}\n")
                f.write(f"  Handle Low (small dip after) : ₹{p['handle_low_price']}\n")
                f.write(f"  ─────────────────────────────────────────────────\n")
                f.write(f"  Cup Drop     : {p['cup_drop_pct']}%\n")
                f.write(f"  Recovery Gap : {p['recovery_pct']}%\n")
                f.write(f"  ─────────────────────────────────────────────────\n")
                f.write(f"  ✦ GEOMETRY CHECK\n")
                f.write(f"    Cup Depth (₹)              : ₹{p['cup_depth']}\n")
                f.write(f"    Max Handle Dip Allowed (₹) : ₹{p['max_handle_dip']}  "
                        f"(= 0.32 × ₹{p['cup_depth']})\n")
                f.write(f"    Actual Handle Pullback (₹) : ₹{p['handle_pullback']}  "
                        f"({p['handle_pullback_pct']}%)\n")
                status = "✅ PASS" if p['handle_pullback'] <= p['max_handle_dip'] else "❌ FAIL"
                f.write(f"    Status                     : {status}\n")
                if "left_rim_date" in p:
                    f.write(f"  ─────────────────────────────────────────────────\n")
                    f.write(f"  📅 Left Rim Date   : {p['left_rim_date']}\n")
                    f.write(f"  📅 Cup Bottom Date : {p['cup_bottom_date']}\n")
                    f.write(f"  📅 Right Rim Date  : {p['right_rim_date']}\n")
                    f.write(f"  📅 Handle Low Date : {p['handle_low_date']}\n")
                f.write(f"  {'═' * 60}\n")

        else:
            f.write("  No patterns found in the historical data.\n")
            f.write("  This is normal — cup-and-handle patterns are relatively rare.\n")
            f.write("  Try a different set of tickers or a wider lookback.\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("  📊 BACKTEST COMPLETE\n")
        f.write("=" * 70 + "\n")

    # ── Clean terminal summary ──
    print(f"\n  {'─' * 60}")
    print(f"  📊 BACKTEST RESULTS")
    print(f"  {'─' * 60}")
    print(f"  Tickers scanned       : {tickers_scanned}")
    print(f"  Tickers with patterns : {tickers_with_patterns}")
    print(f"  Total patterns found  : {len(all_patterns)}")

    if all_patterns:
        # Print a compact summary table to terminal.
        print(f"\n  {'─' * 60}")
        print(f"  SUMMARY TABLE")
        print(f"  {'─' * 60}\n")

        # Print a simplified, narrower table that fits in the terminal.
        header = f"  {'#':>3}  {'Ticker':<16} {'Left Rim':>12} {'Bottom':>12} {'Right Rim':>12} {'Handle':>12} {'Drop%':>7} {'Hdl%':>6}"
        print(header)
        print("  " + "─" * (len(header) - 2))

        for idx, p in enumerate(all_patterns, 1):
            lr_date = clean_date(p.get('left_rim_date', ''))
            print(
                f"  {idx:>3}  {p['ticker']:<16} "
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
#  7.  LIVE SCANNER  — Current-day patterns only, with deduplication (MODE A)
# =============================================================================

def _get_ist_now():
    """Returns the current datetime in IST (Indian Standard Time)."""
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    return utc_now + IST_OFFSET


def is_market_open():
    """
    Checks if the NSE (National Stock Exchange of India) is currently open.

    NSE trading hours: Monday–Friday, 9:15 AM – 3:30 PM IST.

    Returns
    -------
    tuple(bool, str)
        (True/False, human-readable status message)
    """
    ist_now = _get_ist_now()
    day_of_week = ist_now.weekday()   # 0=Monday, 6=Sunday
    current_time = ist_now.time()

    market_open = datetime.time(9, 15)
    market_close = datetime.time(15, 30)

    if day_of_week >= 5:  # Saturday or Sunday
        return False, f"Weekend (IST: {ist_now.strftime('%A %H:%M')})"

    if current_time < market_open:
        return False, (f"Pre-market (IST: {ist_now.strftime('%H:%M')},"
                       f" opens at 09:15)")

    if current_time > market_close:
        return False, (f"After-hours (IST: {ist_now.strftime('%H:%M')},"
                       f" closed at 15:30)")

    return True, f"Market OPEN (IST: {ist_now.strftime('%H:%M')})"


def _reset_dedup_cache_if_new_day():
    """
    Checks if the IST date has changed since the last alert was cached.
    If it has, clears the dedup cache so tickers can alert again on the
    new day.
    """
    global _live_alerted_today, _live_alert_date
    today_ist = _get_ist_now().date()
    if _live_alert_date != today_ist:
        _live_alerted_today = set()
        _live_alert_date = today_ist


def _is_pattern_from_today(pattern):
    """
    Returns True if the pattern's handle low date falls on the current
    IST calendar date.

    The handle low is used as the 'completion timestamp' because it is
    the last structural point of the pattern before a potential breakout.
    """
    if "handle_low_date" not in pattern:
        return False
    try:
        handle_date = pd.Timestamp(pattern["handle_low_date"])
        today_ist = _get_ist_now().date()
        return handle_date.date() == today_ist
    except Exception:
        return False


def scan_watchlist_live(tickers, period=None, start=None, end=None, interval="15m"):
    """
    One-shot live scan: downloads the latest intraday data, runs detection,
    and filters for patterns completing TODAY only.

    Deduplication
    -------------
    Uses the module-level `_live_alerted_today` set to track which tickers
    have already fired an alert today. If a ticker is already in the set,
    it is silently skipped (no repeat alert).

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to scan.

    Returns
    -------
    list[dict]
        Patterns that are (a) completing today and (b) not already alerted.
    """
    global _live_alerted_today

    # Step 1: Clear the dedup cache if the date rolled over.
    _reset_dedup_cache_if_new_day()

    # Step 2: Download latest intraday data.
    all_data = fetch_batch_data(
        tickers, period=period, start=start, end=end, interval=interval
    )

    new_alerts = []

    # Step 3: Scan each ticker for patterns.
    for ticker, df in all_data.items():
        # Dedup check — skip if this ticker already alerted today.
        if ticker in _live_alerted_today:
            continue

        close_prices = df["Close"].dropna().values
        high_prices  = df["High"].reindex(df["Close"].dropna().index).values
        dates = df["Close"].dropna().index

        if len(close_prices) < 50:
            continue

        patterns = detect_cup_and_handle(
            close_prices, ticker=ticker, dates=dates,
            interval=interval, highs=high_prices
        )

        # Step 4: Filter — only keep patterns completing TODAY.
        for pat in patterns:
            if _is_pattern_from_today(pat):
                # This ticker has a fresh, today-only pattern.
                # Add to dedup cache so it won't fire again this session-day.
                _live_alerted_today.add(ticker)
                new_alerts.append(pat)
                break  # one alert per ticker per day is enough

    return new_alerts


def run_live_scanner(tickers=None, period="59d", start=None, end=None, interval="15m"):
    """
    MODE A: LIVE SCANNER
    ====================
    Continuously scans the watchlist every SCAN_INTERVAL_MINUTES during
    NSE market hours.

    Behaviour
    ---------
    • During market hours: scans for patterns completing TODAY.
    • Deduplication: each ticker alerts at most ONCE per day.
    • Outside market hours: prints a "waiting" message and sleeps.
    • Press Ctrl+C at any time to exit cleanly.

    Parameters
    ----------
    tickers : list[str] or None
        If None, fetches the Nifty watchlist automatically.
    """
    label = f"Interval: {interval}, Period: {period or 'Custom Date Range'}"

    print("\n" + "=" * 70)
    print("  🚀 LIVE SCANNER — Current-Day Patterns Only")
    print("=" * 70)
    print(f"  Data          : {label}")
    print(f"  Scan interval : every {SCAN_INTERVAL_MINUTES} minutes")
    print(f"  Market hours  : Mon–Fri, 9:15 AM – 3:30 PM IST")
    print(f"  Dedup         : Each ticker alerts at most ONCE per day")
    print(f"  Geometry      : Handle ≤ {HANDLE_MAX_RETRACE_RATIO * 100:.0f}%"
          f" of cup depth")
    print(f"  Exit          : Press Ctrl+C\n")

    # Fetch the watchlist ONCE (reused across scans).
    if tickers is None:
        tickers = get_nifty_list()

    scan_count = 0

    try:
        while True:
            market_open, status_msg = is_market_open()

            if not market_open:
                print(f"  ⏸  {status_msg} — sleeping 5 min before re-check ...")
                time.sleep(300)  # check every 5 minutes if market opened
                continue

            scan_count += 1
            print(f"\n  {'━' * 60}")
            print(f"  🔄 SCAN #{scan_count}  —  {status_msg}")
            print(f"  {'━' * 60}")
            print(f"  📋 Dedup cache: {len(_live_alerted_today)} ticker(s) "
                  f"already alerted today.\n")

            try:
                new_alerts = scan_watchlist_live(tickers, period=period, start=start, end=end, interval=interval)

                if new_alerts:
                    print(f"\n  🔔 {len(new_alerts)} NEW ALERT(S)!\n")
                    for idx, pat in enumerate(new_alerts, 1):
                        print_pattern(pat, idx)
                else:
                    print("  ─ No new patterns completing today "
                          "(or already alerted).\n")

            except Exception as e:
                # One bad scan shouldn't kill the entire loop.
                print(f"\n  ⚠ Scan error (will retry next cycle): {e}\n")

            print(f"  ⏳ Next scan in {SCAN_INTERVAL_MINUTES} minutes ..."
                  f"  (Ctrl+C to stop)\n")
            time.sleep(SCAN_INTERVAL_MINUTES * 60)

    except KeyboardInterrupt:
        print("\n\n  👋 Scanner stopped by user (Ctrl+C). Goodbye!\n")


# =============================================================================
#  8.  MAIN ENTRY POINT  — Dispatches based on RUN_MODE
# =============================================================================

def validate_yfinance_params(interval, period):
    intraday_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]
    
    if interval in intraday_intervals:
        if period:
            if period.endswith("y") or period.endswith("mo"):
                print(f"  ⚠ WARNING: yfinance only supports up to 60 days for {interval} intraday data.")
                print(f"  Capping lookback to '59d'.\n")
                return "59d"
            elif period.endswith("d"):
                try:
                    days = int(period[:-1])
                    if days >= 60:
                        print(f"  ⚠ WARNING: yfinance only supports up to 60 days for {interval} intraday data.")
                        print(f"  Capping lookback to '59d'.\n")
                        return "59d"
                except:
                    pass
    return period

def prompt_for_config(default_interval="1d", default_period="2y"):
    print("\n  [Configuration]")
    interval = input(f"  Enter candle interval (e.g., 15m, 1h, 1d) [default {default_interval}]: ").strip()
    if not interval:
        interval = default_interval
        
    if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]:
        if default_period == "2y":
            default_period = "59d"
        
    period = input(f"  Enter lookback period (e.g., 59d, 6mo, 2y) [default {default_period}]: ").strip()
    if not period:
        period = default_period
        
    period = validate_yfinance_params(interval, period)
    return interval, period

def main():
    """
    Entry point. Reads the RUN_MODE variable and dispatches accordingly.

    Usage
    -----
    # Toggle at top of file:
    RUN_MODE = "LIVE"     # or "HISTORICAL"

    # Or override via command-line:
    python cup_and_handle_detector.py test
    python cup_and_handle_detector.py live --interval 15m --lookback 59d
    python cup_and_handle_detector.py historical RELIANCE.NS TCS.NS --interval 1d --lookback 2y
    python cup_and_handle_detector.py historical --start-date 2024-01-01 --end-date 2024-06-01

    # Or use the interactive menu:
    python cup_and_handle_detector.py
    """
    parser = argparse.ArgumentParser(description="Cup and Handle Pattern Detector")
    parser.add_argument("mode", nargs="?", default="", help="test, historical, live, or empty for interactive menu")
    parser.add_argument("tickers", nargs="*", help="Optional specific tickers to scan in historical mode")
    parser.add_argument("--interval", "-i", type=str, help="Candle interval (e.g., 15m, 1d)")
    parser.add_argument("--lookback", "-l", type=str, help="Lookback period (e.g., 59d, 2y)")
    parser.add_argument("--start-date", "-s", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", "-e", type=str, help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()

    print(r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     ☕  CUP & HANDLE  PATTERN  DETECTOR                      ║
    ║         Indian Stock Market (NSE)                            ║
    ║                                                              ║
    ║     Geometry : Handle ≤ 32% of Cup Depth                     ║
    ║     Modes    : LIVE       |  HISTORICAL                      ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    mode = args.mode.lower()
    
    # Process args if provided
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
            specific_tickers = [t if t.endswith(".NS") else f"{t}.NS" for t in args.tickers]
            run_historical_backtest(tickers=specific_tickers, period=period, start=args.start_date, end=args.end_date, interval=interval)
        else:
            run_historical_backtest(period=period, start=args.start_date, end=args.end_date, interval=interval)
        return

    elif mode in ("live", "live_scan"):
        if not interval:
            interval = "15m"
        if not period and not args.start_date:
            period = "59d"
            
        period = validate_yfinance_params(interval, period)
        run_live_scanner(period=period, start=args.start_date, end=args.end_date, interval=interval)
        return
        
    elif mode != "":
        print(f"  Unknown mode: '{mode}'")
        print(f"  Valid modes: test, historical, live\n")
        return

    # ── Interactive menu ──
    print(f"  ⚙ Active RUN_MODE = \"{RUN_MODE}\"\n")
    print("  Choose a mode:\n")
    print("    [1]  🧪 Self-Test         — Run on synthetic data (instant, no internet)")
    print("    [2]  📊 Historical Backtest — Full timeline sweep")
    print("    [3]  🚀 Live Scanner       — Current-day patterns only")
    print("    [4]  📊 Backtest ONE       — Historical scan for a specific stock")
    print(f"    [R]  ▶ Run RUN_MODE       — Execute the configured mode (\"{RUN_MODE}\")")
    print("    [Q]  Exit\n")

    choice = input("  Enter choice (1/2/3/4/R/Q): ").strip().lower()

    if choice == "1":
        test_with_sample_data()

    elif choice == "2":
        interval, period = prompt_for_config("1d", "2y")
        run_historical_backtest(period=period, interval=interval)

    elif choice == "3":
        interval, period = prompt_for_config("15m", "59d")
        run_live_scanner(period=period, interval=interval)

    elif choice == "4":
        ticker = input("  Enter ticker (e.g., RELIANCE or RELIANCE.NS): ").strip()
        if not ticker:
            print("  No ticker entered. Exiting.\n")
            return
        if not ticker.endswith(".NS"):
            ticker += ".NS"
            
        interval, period = prompt_for_config("1d", "2y")
        run_historical_backtest(tickers=[ticker], period=period, interval=interval)

    elif choice == "r":
        if RUN_MODE == "LIVE":
            interval, period = prompt_for_config("15m", "59d")
            run_live_scanner(period=period, interval=interval)
        elif RUN_MODE == "HISTORICAL":
            interval, period = prompt_for_config("1d", "2y")
            run_historical_backtest(period=period, interval=interval)
        else:
            print(f"  ⚠ Unknown RUN_MODE: '{RUN_MODE}'")
            print(f"  Set it to 'LIVE' or 'HISTORICAL' at the top of the file.\n")

    elif choice in ("q", "quit", "exit"):
        print("  Bye! 👋\n")

    else:
        print(f"  Invalid choice: '{choice}'\n")

# ── Run the script ──
if __name__ == "__main__":
    main()
