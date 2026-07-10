"""
=============================================================================
  FASTAPI BACKEND — Trading Terminal API
  Wraps the existing multi-pattern scanner as HTTP endpoints
=============================================================================

This is a thin API layer. All detection logic (7 patterns, all checks,
quality scores) comes from the existing pattern_scanner.py — unchanged.

ENDPOINTS
---------
  GET /api/health         → Simple health check
  GET /api/market_status  → NSE market open/closed + time info
  GET /api/watchlist      → Nifty 100/200 ticker list
  GET /api/scan           → Run pattern detection across watchlist
  GET /api/candles        → OHLCV data for one ticker (TradingView format)

HOW TO RUN
----------
  pip install fastapi uvicorn yfinance pandas scipy numpy
  uvicorn backend.main:app --reload --port 8000
"""

import datetime
import os
import time
import traceback

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Import our scanner adapter and cache
from backend.scanner import (
    get_nifty_list,
    fetch_batch_data,
    scan_ticker,
    is_market_open,
    validate_yfinance_params,
    ALL_PATTERNS,
    PATTERN_NAMES,
    PATTERN_SIGNALS,
    scan_watchlist,
)
from backend.cache import ScanCache

# =============================================================================
#  APP SETUP
# =============================================================================

app = FastAPI(
    title="NSE Pattern Scanner API",
    description="Trading terminal API — 7 chart patterns on NSE stocks",
    version="1.0.0",
)

# CORS — allow the frontend (opened as a local file or from a different port)
# to call the API without browser CORS errors.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow all origins (fine for local development)
    allow_credentials=True,
    allow_methods=["*"],       # Allow GET, POST, etc.
    allow_headers=["*"],       # Allow all headers
)

# In-memory cache — results stay valid for 15 minutes
scan_cache = ScanCache(ttl_seconds=900)

# Serve the frontend — detect the project root and frontend directory
_backend_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_backend_dir)
_frontend_dir = os.path.join(_project_root, "frontend")


# =============================================================================
#  STARTUP EVENT — Print available endpoints
# =============================================================================

@app.on_event("startup")
async def startup_message():
    """Prints a startup banner showing the local URL and all endpoints."""
    print("\n" + "=" * 60)
    print("  🚀  NSE Pattern Scanner API — RUNNING")
    print("=" * 60)
    print("  Local URL : http://localhost:8000")
    print("")
    print("  Available Endpoints:")
    print("  ────────────────────────────────────────")
    print("  GET  /api/health         → Health check")
    print("  GET  /api/market_status  → NSE market open/closed")
    print("  GET  /api/watchlist      → Nifty ticker list")
    print("  GET  /api/scan           → Run pattern scan")
    print("  GET  /api/candles        → OHLCV candle data")
    print("  GET  /docs               → Interactive API docs (Swagger)")
    print("  GET  /                   → Trading Terminal UI")
    print("  ────────────────────────────────────────")
    print("  ✨ Open http://localhost:8000 in your browser to use the UI.")
    print("=" * 60 + "\n")


# =============================================================================
#  HELPER FUNCTIONS
# =============================================================================

def _format_date(ts):
    """
    Convert a pandas Timestamp/datetime to a clean date string.
    Handles various formats that come from yfinance / pattern dicts.
    """
    if ts is None:
        return None
    s = str(ts)
    # Remove timezone info (e.g. "+05:30") but keep the time for intraday
    return s.split("+")[0].replace("T", " ").strip()


def _make_serializable(obj):
    """
    Recursively convert numpy types and other non-JSON-serializable types
    to Python built-in types so FastAPI can serialize them to JSON.
    """
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        return str(obj)


def _extract_pattern_key_levels(pattern_dict):
    """
    Extracts the key price levels and dates from a pattern dict
    into a standardized format for the frontend.

    Each pattern type has different field names (left_rim, pole_start, etc.)
    so we normalize them into a list of {label, price, date, color} objects
    that the frontend can use to draw markers on the chart.
    """
    ptype = pattern_dict.get("pattern_type", "")
    markers = []

    if ptype == "cup_and_handle":
        markers = [
            {"label": "Left Rim", "price": pattern_dict.get("left_rim_price"),
             "date": _format_date(pattern_dict.get("left_rim_date")),
             "color": "#2196F3"},  # Blue
            {"label": "Cup Bottom", "price": pattern_dict.get("cup_bottom_price"),
             "date": _format_date(pattern_dict.get("cup_bottom_date")),
             "color": "#F44336"},  # Red
            {"label": "Right Rim", "price": pattern_dict.get("right_rim_price"),
             "date": _format_date(pattern_dict.get("right_rim_date")),
             "color": "#4CAF50"},  # Green
            {"label": "Handle Low", "price": pattern_dict.get("handle_low_price"),
             "date": _format_date(pattern_dict.get("handle_low_date")),
             "color": "#FF9800"},  # Orange
        ]

    elif ptype == "bull_flag":
        markers = [
            {"label": "Pole Start", "price": pattern_dict.get("pole_start_price"),
             "date": _format_date(pattern_dict.get("pole_start_date")),
             "color": "#2196F3"},
            {"label": "Pole Top", "price": pattern_dict.get("pole_top_price"),
             "date": _format_date(pattern_dict.get("pole_top_date")),
             "color": "#F44336"},
            {"label": "Flag Low", "price": pattern_dict.get("flag_low_price"),
             "date": _format_date(pattern_dict.get("flag_end_date")),
             "color": "#4CAF50"},
            {"label": "Breakout", "price": pattern_dict.get("breakout_price"),
             "date": _format_date(pattern_dict.get("signal_date")),
             "color": "#FF9800"},
        ]

    elif ptype == "bear_flag":
        markers = [
            {"label": "Pole Start", "price": pattern_dict.get("pole_start_price"),
             "date": _format_date(pattern_dict.get("pole_start_date")),
             "color": "#2196F3"},
            {"label": "Pole Bottom", "price": pattern_dict.get("pole_bottom_price"),
             "date": _format_date(pattern_dict.get("pole_bottom_date")),
             "color": "#F44336"},
            {"label": "Flag High", "price": pattern_dict.get("flag_high_price"),
             "date": _format_date(pattern_dict.get("flag_end_date")),
             "color": "#4CAF50"},
            {"label": "Breakdown", "price": pattern_dict.get("breakdown_price"),
             "date": _format_date(pattern_dict.get("signal_date")),
             "color": "#FF9800"},
        ]

    elif ptype == "pennant":
        markers = [
            {"label": "Pole Start", "price": pattern_dict.get("pole_start_price"),
             "date": _format_date(pattern_dict.get("pole_start_date")),
             "color": "#2196F3"},
            {"label": "Pole End", "price": pattern_dict.get("pole_end_price"),
             "date": _format_date(pattern_dict.get("pole_end_date")),
             "color": "#F44336"},
            {"label": "Pennant High", "price": pattern_dict.get("pennant_high"),
             "date": _format_date(pattern_dict.get("pennant_start_date")),
             "color": "#4CAF50"},
            {"label": "Breakout", "price": pattern_dict.get("breakout_price"),
             "date": _format_date(pattern_dict.get("breakout_date")),
             "color": "#FF9800"},
        ]

    elif ptype == "head_and_shoulders":
        markers = [
            {"label": "Left Shoulder", "price": pattern_dict.get("left_shoulder_price"),
             "date": _format_date(pattern_dict.get("left_shoulder_date")),
             "color": "#2196F3"},
            {"label": "Head", "price": pattern_dict.get("head_price"),
             "date": _format_date(pattern_dict.get("head_date")),
             "color": "#F44336"},
            {"label": "Right Shoulder", "price": pattern_dict.get("right_shoulder_price"),
             "date": _format_date(pattern_dict.get("right_shoulder_date")),
             "color": "#4CAF50"},
            {"label": "Breakdown", "price": pattern_dict.get("breakdown_price"),
             "date": _format_date(pattern_dict.get("breakdown_date")),
             "color": "#FF9800"},
        ]

    elif ptype == "double_top":
        markers = [
            {"label": "First Top", "price": pattern_dict.get("first_top_price"),
             "date": _format_date(pattern_dict.get("first_top_date")),
             "color": "#2196F3"},
            {"label": "Valley", "price": pattern_dict.get("valley_price"),
             "date": _format_date(pattern_dict.get("valley_date")),
             "color": "#F44336"},
            {"label": "Second Top", "price": pattern_dict.get("second_top_price"),
             "date": _format_date(pattern_dict.get("second_top_date")),
             "color": "#4CAF50"},
            {"label": "Breakdown", "price": pattern_dict.get("breakdown_price"),
             "date": _format_date(pattern_dict.get("breakdown_date")),
             "color": "#FF9800"},
        ]

    elif ptype == "double_bottom":
        markers = [
            {"label": "First Bottom", "price": pattern_dict.get("first_bottom_price"),
             "date": _format_date(pattern_dict.get("first_bottom_date")),
             "color": "#2196F3"},
            {"label": "Peak", "price": pattern_dict.get("peak_price"),
             "date": _format_date(pattern_dict.get("peak_date")),
             "color": "#F44336"},
            {"label": "Second Bottom", "price": pattern_dict.get("second_bottom_price"),
             "date": _format_date(pattern_dict.get("second_bottom_date")),
             "color": "#4CAF50"},
            {"label": "Breakout", "price": pattern_dict.get("breakout_price"),
             "date": _format_date(pattern_dict.get("breakout_date")),
             "color": "#FF9800"},
        ]

    # Filter out markers with missing data
    return [m for m in markers if m.get("price") is not None and m.get("date") is not None]


def _extract_checks(pattern_dict):
    """
    Extracts PASS/FAIL check results from a pattern dict into a
    standardized list for the frontend to display as badges.
    """
    ptype = pattern_dict.get("pattern_type", "")
    checks = []

    if ptype == "cup_and_handle":
        # Geometry
        checks.append({
            "name": "Geometry",
            "status": "PASS" if not pattern_dict.get("reject_reason") else "FAIL",
            "detail": f"Cup drop {pattern_dict.get('cup_drop_pct', 0)}%, Recovery gap {pattern_dict.get('recovery_pct', 0)}%"
        })
        # Roundedness
        rnd = pattern_dict.get("roundedness_pct", 0)
        checks.append({
            "name": "Roundedness",
            "status": "PASS" if rnd >= 20 else "FAIL",
            "detail": f"{rnd}% of candles in base zone (need ≥20%)"
        })
        # Pause before breakout
        pause = pattern_dict.get("pause_duration", 0)
        slope = pattern_dict.get("handle_slope", 0)
        checks.append({
            "name": "Pause Before Breakout",
            "status": "PASS" if pause >= 5 and slope <= 0 else "FAIL",
            "detail": f"{pause} candles, slope {slope}"
        })
        # Volume checks
        for vol_key, vol_name in [
            ("vol_decline_pass", "Volume Decline"),
            ("vol_recovery_pass", "Volume Recovery"),
            ("vol_breakout_pass", "Volume Breakout"),
        ]:
            val = pattern_dict.get(vol_key)
            checks.append({
                "name": vol_name,
                "status": "PASS" if val is True else ("FAIL" if val is False else "N/A"),
                "detail": ""
            })

    elif ptype in ("bull_flag", "bear_flag"):
        for vol_key, vol_name in [
            ("vol_pole_pass", "Pole Volume"),
            ("vol_flag_pass", "Flag Volume"),
            ("vol_breakout_pass" if ptype == "bull_flag" else "vol_breakdown_pass",
             "Breakout Volume" if ptype == "bull_flag" else "Breakdown Volume"),
        ]:
            val = pattern_dict.get(vol_key)
            checks.append({
                "name": vol_name,
                "status": "PASS" if val is True else ("FAIL" if val is False else "N/A"),
                "detail": ""
            })
        checks.append({
            "name": "Pole Linearity (R²)",
            "status": "PASS" if pattern_dict.get("pole_r_squared", 0) >= 0.8 else "FAIL",
            "detail": f"R² = {pattern_dict.get('pole_r_squared', 0)}"
        })
        key = "pole_rise_pct" if ptype == "bull_flag" else "pole_drop_pct"
        checks.append({
            "name": "Pole Strength",
            "status": "PASS",
            "detail": f"{pattern_dict.get(key, 0)}%"
        })

    elif ptype == "pennant":
        for vol_key, vol_name in [
            ("vol_pole_pass", "Pole Volume"),
            ("vol_pennant_pass", "Pennant Volume"),
            ("vol_breakout_pass", "Breakout Volume"),
        ]:
            val = pattern_dict.get(vol_key)
            checks.append({
                "name": vol_name,
                "status": "PASS" if val is True else ("FAIL" if val is False else "N/A"),
                "detail": ""
            })
        cr = pattern_dict.get("convergence_ratio", 1)
        checks.append({
            "name": "Convergence",
            "status": "PASS" if cr < 1 else "FAIL",
            "detail": f"Ratio = {cr}"
        })

    elif ptype == "head_and_shoulders":
        checks.append({
            "name": "Shoulder Symmetry",
            "status": "PASS" if pattern_dict.get("shoulder_symmetry_pct", 99) <= 5 else "FAIL",
            "detail": f"{pattern_dict.get('shoulder_symmetry_pct', 0)}% diff"
        })
        checks.append({
            "name": "Head Prominence",
            "status": "PASS",
            "detail": f"LS +{pattern_dict.get('head_vs_ls_pct', 0)}%, RS +{pattern_dict.get('head_vs_rs_pct', 0)}%"
        })
        val = pattern_dict.get("vol_progression_pass")
        checks.append({
            "name": "Volume Progression",
            "status": "PASS" if val is True else ("FAIL" if val is False else "N/A"),
            "detail": "LS > Head > RS (declining)"
        })
        val = pattern_dict.get("vol_breakdown_pass")
        checks.append({
            "name": "Breakdown Volume",
            "status": "PASS" if val is True else ("FAIL" if val is False else "N/A"),
            "detail": ""
        })

    elif ptype in ("double_top", "double_bottom"):
        checks.append({
            "name": "Peak/Bottom Similarity",
            "status": "PASS" if pattern_dict.get("similarity_pct", 99) <= 6 else "FAIL",
            "detail": f"{pattern_dict.get('similarity_pct', 0)}% diff (≤6% required)"
        })
        depth_key = "valley_drop_pct" if ptype == "double_top" else "peak_rise_pct"
        checks.append({
            "name": "Depth/Height",
            "status": "PASS",
            "detail": f"{pattern_dict.get(depth_key, 0)}%"
        })
        val = pattern_dict.get("vol_pattern_pass")
        checks.append({
            "name": "Volume Pattern",
            "status": "PASS" if val is True else ("FAIL" if val is False else "N/A"),
            "detail": ""
        })
        bk = "vol_breakdown_pass" if ptype == "double_top" else "vol_breakout_pass"
        val = pattern_dict.get(bk)
        checks.append({
            "name": "Breakout/Breakdown Volume",
            "status": "PASS" if val is True else ("FAIL" if val is False else "N/A"),
            "detail": ""
        })

    return checks


# =============================================================================
#  ENDPOINT 1: /api/health
# =============================================================================

@app.get("/api/health")
async def health_check():
    """Simple health check — returns {"status": "ok"} if the server is running."""
    return {"status": "ok"}


# =============================================================================
#  ENDPOINT 2: /api/market_status
# =============================================================================

@app.get("/api/market_status")
async def market_status():
    """
    Returns whether the NSE market is currently open or closed,
    plus the current IST time and next open/close time.
    """
    try:
        is_open, status_msg = is_market_open()

        # Calculate time until next event
        ist_offset = datetime.timedelta(hours=5, minutes=30)
        utc_now = datetime.datetime.now(datetime.timezone.utc)
        ist_now = utc_now + ist_offset

        market_open_time = datetime.time(9, 15)
        market_close_time = datetime.time(15, 30)

        if is_open:
            # Market is open — calculate time until close
            close_dt = ist_now.replace(
                hour=market_close_time.hour,
                minute=market_close_time.minute,
                second=0, microsecond=0
            )
            remaining = close_dt - ist_now
            next_event = f"Closes in {remaining.seconds // 3600}h {(remaining.seconds % 3600) // 60}m"
        else:
            # Market is closed — calculate time until next open
            next_open = ist_now.replace(
                hour=market_open_time.hour,
                minute=market_open_time.minute,
                second=0, microsecond=0
            )
            if ist_now.time() >= market_close_time or ist_now.weekday() >= 5:
                # After market close or weekend — next open is tomorrow/Monday
                days_ahead = 1
                if ist_now.weekday() == 4:  # Friday after close
                    days_ahead = 3
                elif ist_now.weekday() == 5:  # Saturday
                    days_ahead = 2
                elif ist_now.weekday() == 6:  # Sunday
                    days_ahead = 1
                next_open += datetime.timedelta(days=days_ahead)
            remaining = next_open - ist_now
            hours = remaining.seconds // 3600
            minutes = (remaining.seconds % 3600) // 60
            next_event = f"Opens in {remaining.days}d {hours}h {minutes}m" if remaining.days > 0 else f"Opens in {hours}h {minutes}m"

        return {
            "is_open": is_open,
            "status": status_msg,
            "ist_time": ist_now.strftime("%H:%M:%S"),
            "ist_date": ist_now.strftime("%Y-%m-%d"),
            "next_event": next_event,
        }

    except Exception as e:
        return {
            "is_open": False,
            "status": f"Error checking market status: {str(e)}",
            "ist_time": "",
            "ist_date": "",
            "next_event": "",
        }


# =============================================================================
#  ENDPOINT 3: /api/watchlist
# =============================================================================

@app.get("/api/watchlist")
async def watchlist():
    """Returns the current Nifty 100/200 ticker list."""
    try:
        tickers = get_nifty_list()
        return {
            "count": len(tickers),
            "tickers": tickers,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch watchlist: {str(e)}")


# =============================================================================
#  ENDPOINT 4: /api/scan
# =============================================================================

@app.get("/api/scan")
async def run_scan(
    pattern: str = Query("all", description="Pattern to scan: cup_handle, bull_flag, bear_flag, pennant, head_shoulders, double_top, double_bottom, or 'all'"),
    interval: str = Query("1d", description="Candle interval: 15m, 1h, 1d, 1wk"),
    lookback: str = Query("3mo", description="Lookback period: 1mo, 3mo, 6mo, 1y, 2y"),
    live_mode: bool = Query(False, description="Include forming patterns (Live Mode)"),
):
    """
    Runs the pattern scanner across the Nifty watchlist.

    Returns all detected patterns with key price levels, dates,
    check results, and quality scores.
    """
    # ── Normalize pattern name ──
    pattern_map = {
        "cup_handle": "cup_and_handle",
        "cup_and_handle": "cup_and_handle",
        "bull_flag": "bull_flag",
        "bear_flag": "bear_flag",
        "pennant": "pennant",
        "head_shoulders": "head_and_shoulders",
        "head_and_shoulders": "head_and_shoulders",
        "double_top": "double_top",
        "double_bottom": "double_bottom",
        "all": "all",
    }

    normalized = pattern_map.get(pattern.lower(), pattern.lower())
    if normalized != "all" and normalized not in ALL_PATTERNS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown pattern '{pattern}'. Valid options: {', '.join(list(pattern_map.keys()))}"
        )

    # ── Validate Interval ──
    valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
    if interval not in valid_intervals:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid interval '{interval}'. Must be one of: {', '.join(valid_intervals)}"
        )

    # ── Validate yfinance params ──
    validated_lookback = validate_yfinance_params(interval, lookback)

    # ── Check cache ──
    cache_key = f"{normalized}:{interval}:{validated_lookback}:{live_mode}"
    cached = scan_cache.get(cache_key)
    if cached is not None:
        cached["from_cache"] = True
        return cached

    # ── Run the scan ──
    scan_start = time.time()

    try:
        # Get watchlist
        tickers = get_nifty_list()
        total_scanned = len(tickers)

        # Download data
        all_data = fetch_batch_data(
            tickers, period=validated_lookback, interval=interval
        )

        if not all_data:
            return {
                "scan_time": datetime.datetime.now().isoformat(),
                "interval": interval,
                "lookback": validated_lookback,
                "pattern": normalized,
                "total_scanned": total_scanned,
                "total_with_data": 0,
                "matches": [],
                "scan_duration_seconds": round(time.time() - scan_start, 1),
                "from_cache": False,
                "error": "No data downloaded. Check your internet connection.",
            }

        # Determine which patterns to scan
        patterns_to_scan = ["all"] if normalized == "all" else [normalized]

        # Run detection on each ticker
        all_matches = []

        for ticker, df in all_data.items():
            try:
                ticker_results = scan_ticker(
                    df, ticker, patterns_to_scan, interval=interval
                )

                for ptype, patterns in ticker_results.items():
                    # Deduplicate by signal_date
                    best_patterns = {}
                    for pat in patterns:
                        # Filter out forming if not in live_mode
                        if not live_mode and pat.get("status") == "forming":
                            continue
                            
                        # Deduplicate by timeline (signal_date)
                        sig_date = pat.get("signal_date")
                        if not sig_date:
                            sig_date = pat.get("breakout_date", pat.get("breakdown_date", pat.get("status", "")))
                            
                        key = (ptype, sig_date)
                        if key not in best_patterns or pat.get("quality_score", 0) > best_patterns[key].get("quality_score", 0):
                            best_patterns[key] = pat
                            
                    for pat in best_patterns.values():
                        # Build a clean match object for the API
                        match = {
                            "ticker": pat.get("ticker", ticker),
                            "pattern": PATTERN_NAMES.get(ptype, ptype),
                            "pattern_type": ptype,
                            "signal": PATTERN_SIGNALS.get(ptype, ""),
                            "quality_score": pat.get("quality_score", 0),
                            "key_levels": _extract_pattern_key_levels(pat),
                            "checks": _extract_checks(pat),
                            "raw": _make_serializable(pat),
                        }
                        all_matches.append(match)

            except Exception as e:
                # If one ticker fails, log it and continue with the rest
                print(f"  ⚠ Error scanning {ticker}: {e}")
                continue

        # Sort by quality score (highest first)
        all_matches.sort(key=lambda m: m["quality_score"], reverse=True)

        result = {
            "scan_time": datetime.datetime.now().isoformat(),
            "interval": interval,
            "lookback": validated_lookback,
            "pattern": normalized,
            "total_scanned": total_scanned,
            "total_with_data": len(all_data),
            "matches": all_matches,
            "scan_duration_seconds": round(time.time() - scan_start, 1),
            "from_cache": False,
        }

        # Cache the result
        scan_cache.set(cache_key, result)

        return result

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Scan failed: {str(e)}"
        )


# =============================================================================
#  ENDPOINT 5: /api/candles
# =============================================================================

@app.get("/api/candles")
async def get_candles(
    ticker: str = Query(..., description="Ticker symbol (e.g. RELIANCE.NS)"),
    interval: str = Query("1d", description="Candle interval: 15m, 1h, 1d, 1wk"),
    lookback: str = Query("3mo", description="Lookback period: 1mo, 3mo, 6mo, 1y, 2y"),
):
    """
    Fetches OHLCV candle data for a single ticker via yfinance.

    Returns data in the exact format TradingView Lightweight Charts expects:
    - Intraday (15m, 1h): time as UNIX timestamp (integer seconds)
    - Daily/weekly: time as "YYYY-MM-DD" string
    """
    # Validate params
    validated_lookback = validate_yfinance_params(interval, lookback)

    # Ensure .NS suffix
    if not ticker.endswith(".NS"):
        ticker = ticker + ".NS"

    try:
        # Fetch data via Layer 1 Cache
        from db_cache import get_stock_data
        df = get_stock_data(
            ticker=ticker,
            period=validated_lookback,
            interval=interval
        )

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for {ticker}. It may be delisted or the ticker may be wrong."
            )

        # Handle MultiIndex columns (yfinance sometimes returns these)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel("Ticker")

        # Determine if we need UNIX timestamps (intraday) or date strings (daily)
        is_intraday = interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]

        candles = []
        for idx, row in df.iterrows():
            try:
                open_val = float(row["Open"])
                high_val = float(row["High"])
                low_val = float(row["Low"])
                close_val = float(row["Close"])
                volume_val = int(row["Volume"]) if not pd.isna(row["Volume"]) else 0

                # Skip rows with NaN prices
                if any(pd.isna([open_val, high_val, low_val, close_val])):
                    continue

                if is_intraday:
                    # UNIX timestamp (integer seconds)
                    ts = int(pd.Timestamp(idx).timestamp())
                    time_val = ts
                else:
                    # Date string "YYYY-MM-DD"
                    time_val = pd.Timestamp(idx).strftime("%Y-%m-%d")

                candles.append({
                    "time": time_val,
                    "open": round(open_val, 2),
                    "high": round(high_val, 2),
                    "low": round(low_val, 2),
                    "close": round(close_val, 2),
                    "volume": volume_val,
                })
            except (ValueError, TypeError):
                continue

        if not candles:
            raise HTTPException(
                status_code=404,
                detail=f"Downloaded data for {ticker} but all rows were invalid."
            )

        return {
            "ticker": ticker,
            "interval": interval,
            "lookback": validated_lookback,
            "count": len(candles),
            "candles": candles,
        }

    except HTTPException:
        raise  # Re-raise our own HTTPExceptions
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch candles for {ticker}: {str(e)}"
        )


# =============================================================================
#  LIVE SCAN ENDPOINT
# =============================================================================

@app.get("/api/live_scan")
async def api_live_scan(
    patterns: str = Query("all", description="Comma-separated pattern list"),
    interval: str = Query("15m", description="Candle interval"),
    lookback: str = Query("59d", description="Lookback period"),
):
    """
    Runs a live scan across Nifty 200 to find NEW alerts completing TODAY.
    Uses the stateful deduplication cache in pattern_scanner.py to avoid duplicates.
    """
    # Parse patterns
    pattern_list = [p.strip().lower() for p in patterns.split(",")]
    if "all" in pattern_list:
        patterns_to_scan = ["all"]
    else:
        patterns_to_scan = [p for p in pattern_list if p in ALL_PATTERNS]
        if not patterns_to_scan:
            patterns_to_scan = ["all"]

    # Get tickers
    tickers = get_nifty_list()
    
    # Validate Interval
    valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
    if interval not in valid_intervals:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid interval '{interval}'. Must be one of: {', '.join(valid_intervals)}"
        )

    # Validate yfinance lookback
    validated_lookback = validate_yfinance_params(interval, lookback)

    # Run scan_watchlist
    new_alerts = scan_watchlist(
        tickers=tickers,
        patterns_to_scan=patterns_to_scan,
        period=validated_lookback,
        interval=interval
    )
    
    # Format results
    results = []
    for pat in new_alerts:
        ptype = pat.get("pattern_type", "")
        serializable_pat = _make_serializable(pat)
        results.append({
            "ticker": pat.get("ticker", "UNKNOWN"),
            "pattern": PATTERN_NAMES.get(ptype, ptype.upper()),
            "pattern_type": ptype,
            "signal": PATTERN_SIGNALS.get(ptype, "NEUTRAL"),
            "verdict": pat.get("verdict", "UNKNOWN"),
            "quality_score": pat.get("quality_score", 0),
            "key_levels": _extract_pattern_key_levels(pat),
            "checks": _extract_checks(pat),
            "raw": serializable_pat,
            
            # Keep original keys for backward compatibility
            "pattern_id": ptype,
            "quality": pat.get("quality_score", 0),
            "signal_type": PATTERN_SIGNALS.get(ptype, "NEUTRAL"),
            "markers": _extract_pattern_key_levels(pat),
            "raw_data": serializable_pat
        })
        
    return {
        "count": len(results),
        "alerts": results,
    }


# =============================================================================
#  SERVE FRONTEND — Serve index.html at the root URL
# =============================================================================

@app.get("/")
async def serve_frontend():
    """Serve the trading terminal frontend at the root URL."""
    index_path = os.path.join(_frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"message": "Frontend not found. Place index.html in the frontend/ directory."}

