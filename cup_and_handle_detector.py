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

def fetch_batch_data(tickers, period="59d", interval="15m"):
    """
    Downloads candle data for many tickers in batches.

    Parameters
    ----------
    tickers : list[str]
        e.g., ["RELIANCE.NS", "TCS.NS", ...]
    period : str
        How far back to look. "59d" = last 59 calendar days,
        "2y" = last 2 years (requires interval="1d").
    interval : str
        Candle size. "15m" = each row is 15 minutes of trading.
        "1d" = each row is one trading day.

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
            data = yf.download(
                batch_str,
                period=period,
                interval=interval,
                group_by="ticker",
                progress=False,     # suppress yfinance's own progress bar
                threads=True,       # use parallel downloads internally
            )

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

            # ══════════════════════════════════════════════════════════
            # ── Step 3b-extra2: LEFT RIM ABSOLUTE CEILING CHECK ──
            # ══════════════════════════════════════════════════════════
            #
            #   "Intermediate High Violation" filter
            #   ------------------------------------
            #   After selecting a Left Rim peak, NO candle inside the cup
            #   (between Left Rim and Right Rim) may have a High price that
            #   exceeds the Left Rim candle's own High. If it does, the
            #   Left Rim was never the true structural ceiling — the cup
            #   geometry is fake.
            #
            #   We compare internal Highs against the Left Rim's own High
            #   (not Close) because the rim's intraday wick tip is the
            #   absolute structural maximum. Any candle whose High exceeds
            #   that is a clear intermediate breakout.
            #
            #   ZERO tolerance — even a ₹1 breach means the Left Rim was
            #   not the true peak. The old 0.5% buffer let through cases
            #   like ABB.NS where a ₹7490 High slipped under a ₹7495
            #   ceiling despite being ₹32 above the Left Rim Close.
            # ──────────────────────────────────────────────────────────
            rim_ceiling = highs[left_rim_idx]              # Left Rim's own High
            highs_inside_cup = highs[left_rim_idx + 1 : right_rim_idx]

            if len(highs_inside_cup) > 0:
                max_internal_high = np.max(highs_inside_cup)
                if max_internal_high > rim_ceiling:
                    if verbose:
                        breach_local_idx = np.argmax(highs_inside_cup)
                        breach_abs_idx   = left_rim_idx + 1 + breach_local_idx
                        breach_date_str  = (
                            str(dates[breach_abs_idx]) if dates is not None
                            else f"idx {breach_abs_idx}"
                        )
                        print(
                            f"  ❌ REJECTED {ticker}: Internal high "
                            f"₹{max_internal_high:.2f} on {breach_date_str} "
                            f"exceeded Left Rim High ₹{rim_ceiling:.2f} — "
                            f"Intermediate High Violation"
                        )
                    continue   # FAKE cup — reject immediately

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

            # ── Step 3d: Handle detection ──
            # Look for a mild pullback in the candles AFTER the Right Rim.
            # Handle search window strictly scales up to 25% of the total cup duration.
            handle_zone_start = right_rim_idx
            handle_zone_end = min(right_rim_idx + max(1, cup_width // 4),
                                  len(prices) - 1)

            if handle_zone_end <= handle_zone_start + 3:
                continue   # not enough data after Right Rim for a handle

            handle_prices = prices[handle_zone_start : handle_zone_end + 1]
            handle_low_price = np.min(handle_prices)
            handle_low_local_idx = np.argmin(handle_prices)
            handle_low_idx = handle_zone_start + handle_low_local_idx

            # ══════════════════════════════════════════════════════════
            # ── Step 3e: Handle depth check  (VERIFIED GEOMETRY) ──
            # ══════════════════════════════════════════════════════════
            #
            #   Cup Depth      = Left_Rim_Price − Cup_Bottom_Price
            #   Max Handle Dip = HANDLE_MAX_RETRACE_RATIO × Cup Depth
            #                  = 0.32 × Cup Depth  (by default)
            #
            #   Handle Pullback = Right_Rim_Price − Handle_Low_Price
            #
            #   RULE: Handle Pullback must be ≥ 1% of Right Rim (real dip)
            #         AND ≤ Max Handle Dip         (dip is shallow enough).
            #   If it exceeds Max Handle Dip → IMMEDIATELY REJECT.
            #
            cup_depth = left_rim_price - cup_bottom_price
            max_handle_dip = HANDLE_MAX_RETRACE_RATIO * cup_depth
            handle_pullback = right_rim_price - handle_low_price

            # No meaningful dip — handle must pull back at least 1% of
            # the right rim price to count as a real dip (not noise).
            min_handle_pullback = right_rim_price * 0.01
            if handle_pullback < min_handle_pullback:
                continue

            # Handle dipped deeper than the allowed 32% of cup depth — reject.
            if handle_pullback > max_handle_dip:
                continue

            # Compute percentage for display purposes.
            handle_pullback_pct = (handle_pullback / right_rim_price) * 100

            # ── Rule 5: Time Proportions ──
            # The Cup must take significantly longer to form than the Handle.
            # Cup Width must be >= 3 * Handle Width.
            handle_width = handle_low_idx - right_rim_idx
            if handle_width > 0 and cup_width < 3 * handle_width:
                continue

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
            }

            # Attach dates if available (for reporting).
            if dates is not None:
                pattern["left_rim_date"]   = str(dates[left_rim_idx])
                pattern["cup_bottom_date"] = str(dates[cup_bottom_idx])
                pattern["right_rim_date"]  = str(dates[right_rim_idx])
                pattern["handle_low_date"] = str(dates[handle_low_idx])

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

        # Check if this candidate overlaps with any already-accepted pattern.
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
    print(f"  ✦ GEOMETRY CHECK (verified formula)")
    print(f"    Cup Depth (₹)              : ₹{p['cup_depth']}")
    print(f"    Max Handle Dip Allowed (₹) : ₹{p['max_handle_dip']}  "
          f"(= 0.32 × ₹{p['cup_depth']})")
    print(f"    Actual Handle Pullback (₹) : ₹{p['handle_pullback']}  "
          f"({p['handle_pullback_pct']}%)")
    if p['handle_pullback'] <= p['max_handle_dip']:
        print(f"    Status                     : ✅ PASS  "
              f"(₹{p['handle_pullback']} ≤ ₹{p['max_handle_dip']})")
    else:
        print(f"    Status                     : ❌ FAIL  "
              f"(₹{p['handle_pullback']} > ₹{p['max_handle_dip']})")

    # Show dates if available.
    if "left_rim_date" in p:
        print(f"  ─────────────────────────────────────────────────")
        print(f"  📅 Left Rim Date   : {p['left_rim_date']}")
        print(f"  📅 Cup Bottom Date : {p['cup_bottom_date']}")
        print(f"  📅 Right Rim Date  : {p['right_rim_date']}")
        print(f"  📅 Handle Low Date : {p['handle_low_date']}")

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

    # 7. Post-handle (a few more candles so the pattern is "complete")
    post_handle = np.ones(15) * 98

    # Concatenate all segments into one price series.
    synthetic_prices = np.concatenate([
        rise_to_left_rim, descent, bottom, ascent, right_rim,
        handle_down, handle_up, post_handle
    ])

    print(f"  Synthetic series length : {len(synthetic_prices)} candles")
    print(f"  Price range             : ₹{synthetic_prices.min():.1f}"
          f" – ₹{synthetic_prices.max():.1f}")
    print(f"  Expected                : Cup drop ≈ 20%, Handle dip ≈ ₹5")
    print(f"  Geometry                : Cup depth = ₹20, "
          f"Max handle = ₹{0.32 * 20:.1f}, Actual = ₹5 → PASS")

    # Run detection
    results = detect_cup_and_handle(synthetic_prices, ticker="SYNTHETIC_CUP")

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
    deep_post_handle = np.ones(15) * 95

    deep_handle_prices = np.concatenate([
        rise_to_left_rim, descent, bottom, ascent, right_rim,
        deep_handle_down, deep_handle_up, deep_post_handle
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

def run_historical_backtest(tickers=None):
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
    cfg = MODE_CONFIG["HISTORICAL"]

    print("\n" + "=" * 70)
    print("  📊 HISTORICAL BACKTEST  (Full Timeline Sweep)")
    print("=" * 70)
    print(f"  Data   : {cfg['label']}")
    print(f"  Filter : NONE — every valid pattern is printed")
    print(f"  Geometry: Handle ≤ {HANDLE_MAX_RETRACE_RATIO * 100:.0f}% "
          f"of cup depth\n")

    # If no tickers specified, fetch the full Nifty watchlist.
    if tickers is None:
        print("  Fetching Nifty watchlist ...\n")
        tickers = get_nifty_list()

    print(f"  Scanning {len(tickers)} tickers over {cfg['label']}\n")

    # Download all data in batches.
    all_data = fetch_batch_data(
        tickers, period=cfg["period"], interval=cfg["interval"]
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

        # Full sweep — no date restriction.
        # verbose=False suppresses per-rejection print messages.
        patterns = detect_cup_and_handle(
            close_prices, ticker=ticker, dates=dates,
            interval=cfg["interval"], highs=high_prices, verbose=False
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
        f.write(f"  Data           : {cfg['label']}\n")
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


def scan_watchlist_live(tickers):
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

    cfg = MODE_CONFIG["LIVE"]

    # Step 1: Clear the dedup cache if the date rolled over.
    _reset_dedup_cache_if_new_day()

    # Step 2: Download latest intraday data.
    all_data = fetch_batch_data(
        tickers, period=cfg["period"], interval=cfg["interval"]
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
            interval=cfg["interval"], highs=high_prices
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


def run_live_scanner(tickers=None):
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
    cfg = MODE_CONFIG["LIVE"]

    print("\n" + "=" * 70)
    print("  🚀 LIVE SCANNER — Current-Day Patterns Only")
    print("=" * 70)
    print(f"  Data          : {cfg['label']}")
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
                new_alerts = scan_watchlist_live(tickers)

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

def main():
    """
    Entry point. Reads the RUN_MODE variable and dispatches accordingly.

    Usage
    -----
    # Toggle at top of file:
    RUN_MODE = "LIVE_SCAN"     # or "HISTORICAL"

    # Or override via command-line:
    python cup_and_handle_detector.py test         → self-test
    python cup_and_handle_detector.py live         → live scanner (overrides RUN_MODE)
    python cup_and_handle_detector.py historical   → historical backtest
    python cup_and_handle_detector.py historical RELIANCE.NS TCS.NS
                                                   → backtest specific stocks
    # Or use the interactive menu:
    python cup_and_handle_detector.py              → menu
    """
    print(r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     ☕  CUP & HANDLE  PATTERN  DETECTOR                      ║
    ║         Indian Stock Market (NSE)                             ║
    ║                                                              ║
    ║     Geometry : Handle ≤ 32% of Cup Depth                     ║
    ║     Modes    : LIVE       |  HISTORICAL                      ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    print(f"  ⚙ Active RUN_MODE = \"{RUN_MODE}\"\n")

    # ── Command-line argument mode (overrides RUN_MODE) ──
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == "test":
            test_with_sample_data()
            return

        elif mode in ("historical", "backtest"):
            # Optional: specific tickers after the mode keyword.
            if len(sys.argv) > 2:
                specific_tickers = sys.argv[2:]
                # Ensure .NS suffix
                specific_tickers = [
                    t if t.endswith(".NS") else f"{t}.NS"
                    for t in specific_tickers
                ]
                run_historical_backtest(tickers=specific_tickers)
            else:
                run_historical_backtest()
            return

        elif mode in ("live", "live_scan"):
            run_live_scanner()
            return

        else:
            print(f"  Unknown mode: '{mode}'")
            print(f"  Valid modes: test, historical, live\n")
            return

    # ── Interactive menu ──
    print("  Choose a mode:\n")
    print("    [1]  🧪 Self-Test         — Run on synthetic data "
          "(instant, no internet)")
    print("    [2]  📊 Historical Backtest — Full timeline sweep "
          f"({MODE_CONFIG['HISTORICAL']['label']})")
    print("    [3]  🚀 Live Scanner       — Current-day patterns only "
          f"({MODE_CONFIG['LIVE']['label']})")
    print("    [4]  📊 Backtest ONE       — Historical scan for a "
          "specific stock")
    print(f"    [R]  ▶ Run RUN_MODE       — Execute the configured mode "
          f"(\"{RUN_MODE}\")")
    print("    [Q]  Exit\n")

    choice = input("  Enter choice (1/2/3/4/R/Q): ").strip().lower()

    if choice == "1":
        test_with_sample_data()

    elif choice == "2":
        run_historical_backtest()

    elif choice == "3":
        run_live_scanner()

    elif choice == "4":
        ticker = input(
            "  Enter ticker (e.g., RELIANCE or RELIANCE.NS): "
        ).strip()
        if not ticker:
            print("  No ticker entered. Exiting.\n")
            return
        if not ticker.endswith(".NS"):
            ticker += ".NS"
        run_historical_backtest(tickers=[ticker])

    elif choice == "r":
        # Dispatch based on the RUN_MODE variable set at the top.
        if RUN_MODE == "LIVE":
            run_live_scanner()
        elif RUN_MODE == "HISTORICAL":
            run_historical_backtest()
        else:
            print(f"  ⚠ Unknown RUN_MODE: '{RUN_MODE}'")
            print(f"  Set it to 'LIVE' or 'HISTORICAL' at the top "
                  f"of the file.\n")

    elif choice in ("q", "quit", "exit"):
        print("  Bye! 👋\n")

    else:
        print(f"  Invalid choice: '{choice}'\n")


# ── Run the script ──
if __name__ == "__main__":
    main()
