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
from scipy.signal import find_peaks

ALGO_VERSION = "3.1"

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
from scipy.stats import linregress     # Calculates linear regression slope + R²

# Suppress noisy warnings from yfinance / pandas
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
#  DETECT ENGINE IMPORTS
# =============================================================================
from detectors.core import *
from detectors.cup_and_handle import detect_cup_and_handle
from detectors.bull_flag import detect_bull_flag
from detectors.bear_flag import detect_bear_flag
from detectors.pennant import detect_pennant
from detectors.head_and_shoulders import detect_head_and_shoulders
from detectors.double_top import detect_double_top
from detectors.double_bottom import detect_double_bottom

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

def fetch_batch_data(tickers, period=None, start=None, end=None, interval="15m", errors_dict=None):
    """
    Downloads OHLCV candle data for many tickers in batches.
    Returns dict[str, pd.DataFrame]. If errors_dict is provided, it populates
    it with short reasons for skipped tickers.
    """
    import db_cache
    import concurrent.futures

    all_data = {}
    min_candles = 30 if interval in ("1d", "1wk") else 50
    
    print(f"  📦 Fetching data for {len(tickers)} tickers via Layer 1 cache...")

    def fetch_single(t):
        try:
            # get_stock_data implements Layer 1 cache + delta fetching
            df = db_cache.get_stock_data(t, interval=interval, period=period or "59d")
            if df is None or df.empty:
                return t, None, "No data available"
            if len(df) < min_candles:
                return t, None, f"Insufficient history (<{min_candles} candles)"
            return t, df, None
        except Exception as e:
            print(f"Error fetching {t}: {e}")
            return t, None, "Fetch error or timeout"

    with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        futures = {executor.submit(fetch_single, t): t for t in tickers}
        for future in concurrent.futures.as_completed(futures):
            t, df, err = future.result()
            if df is not None:
                all_data[t] = df
            elif errors_dict is not None and err is not None:
                errors_dict[t] = err

    print(f"  ✓ Successfully fetched data for {len(all_data)} tickers.")
    return all_data


# =============================================================================
#  3. SMOOTH PRICES — Reduce noise before peak detection (only when needed)
# =============================================================================




# =============================================================================
#  4. QUALITY SCORE — Generalized pattern scoring
# =============================================================================




# =============================================================================
#  5. DETECT CUP AND HANDLE
#     (Exact copy from v2 — all rules intact, only quality_score call updated)
# =============================================================================




# =============================================================================
#  6. DETECT BULL FLAG
# =============================================================================



# =============================================================================


# =============================================================================


# =============================================================================










# =============================================================================
#  HELPER FUNCTIONS FOR PEAK REFINEMENT
# =============================================================================







# =============================================================================
#  10. DETECT DOUBLE TOP
# =============================================================================


# =============================================================================




# =============================================================================
#  11. DETECT DOUBLE BOTTOM
# =============================================================================





# =============================================================================
#  12. PRINT FUNCTIONS — Pattern-specific debug output
# =============================================================================

















# Dispatcher: route to the correct printer based on pattern_type




# =============================================================================
#  13. SELF-TEST FUNCTIONS — Synthetic data per pattern
# =============================================================================

















# =============================================================================
#  14. SCANNING INFRASTRUCTURE — Multi-pattern scanning
# =============================================================================


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
            close, highs=highs, lows=lows, volumes=vol, ticker=ticker, dates=dates,
            interval=interval, verbose=verbose
        ),
        "head_and_shoulders": lambda: detect_head_and_shoulders(
            close, highs=highs, lows=lows, volumes=vol, ticker=ticker, dates=dates,
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

    import json
    import time
    import sqlite3
    from core.serialization import _make_serializable
    from db_cache import CACHE_DB_PATH
    
    start_date_str = df.index.min().strftime('%Y-%m-%d')
    end_date_str = df.index.max().strftime('%Y-%m-%d')
    param_hash = get_param_hash()
    
    conn = sqlite3.connect(CACHE_DB_PATH, timeout=30.0)
    cursor = conn.cursor()

    for ptype, detect_fn in DETECT_MAP.items():
        if scan_all or ptype in patterns_to_scan:
            if is_live:
                found = detect_fn()
                
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


_live_alerted_today = set()
_live_alert_date = None

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
    from core.serialization import _make_serializable
    from db_cache import CACHE_DB_PATH
    
    conn = sqlite3.connect(CACHE_DB_PATH, timeout=30.0)
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
                   end=None, interval="15m", return_scanned_tickers=False):
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

    if return_scanned_tickers:
        return new_alerts, list(all_data.keys())
    return new_alerts




# =============================================================================
#  16. MENU & MAIN ENTRY POINT
# =============================================================================

def validate_yfinance_params(interval, period):
    """Validates that the interval+lookback combo is supported by yfinance."""
    import re
    
    if period:
        valid_periods = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
        if period not in valid_periods and not re.match(r"^\d+d$", period):
            raise ValueError(f"Invalid lookback period: {period}")

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








