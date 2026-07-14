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
from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import sqlite3
import json
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
from backend.api_models import (
    HealthResponse, MarketStatusResponse, WatchlistResponse, ScanResponse, 
    CandlesResponse, LiveScanResponse, LiveAlertsResponse, DismissResponse, DismissRequest,
    _format_date, _make_serializable, _extract_pattern_key_levels, _extract_checks
)

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
scan_cache = ScanCache(ttl_seconds=900, max_size=100)

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
# =============================================================================
#  ENDPOINT 1: /api/health
# =============================================================================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Simple health check — returns {"status": "ok"} if the server is running."""
    return {"status": "ok"}


# =============================================================================
#  ENDPOINT 2: /api/market_status
# =============================================================================

@app.get("/api/market_status", response_model=MarketStatusResponse)
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

@app.get("/api/watchlist", response_model=WatchlistResponse)
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

@app.get("/api/scan", response_model=ScanResponse)
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
        errors_dict = {}
        all_data = fetch_batch_data(
            tickers, period=validated_lookback, interval=interval, errors_dict=errors_dict
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
                "skipped_tickers": len(errors_dict),
                "errors": errors_dict,
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
                # If one ticker fails, log the real error and trace to console
                print(f"  ⚠ Error scanning {ticker}: {e}")
                traceback.print_exc()
                # Return generic error to the API
                errors_dict[ticker] = "Internal detection error"
                continue

        # Sort by quality score (highest first), then ticker, then pattern, then signal date
        all_matches.sort(
            key=lambda m: (
                -m.get("quality_score", 0.0),
                m.get("ticker", ""),
                m.get("pattern_type", ""),
                m.get("raw", {}).get("signal_date", "")
            )
        )

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
            "skipped_tickers": len(errors_dict),
            "errors": errors_dict,
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

@app.get("/api/candles", response_model=CandlesResponse)
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

@app.get("/api/live_scan", response_model=LiveScanResponse)
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
    
    # Connect to DB to deduplicate and persist
    from db_cache import CACHE_DB_PATH
    conn = sqlite3.connect(CACHE_DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    
    now_str = datetime.datetime.now().isoformat()
    valid_new_results = []
    
    for pat in new_alerts:
        ptype = pat.get("pattern_type", "")
        ticker = pat.get("ticker", "UNKNOWN")
        # Identity logic
        sig_date = pat.get("signal_date", "")
        if not sig_date:
            sig_date = pat.get("breakout_date", pat.get("breakdown_date", ""))
            
        alert_id = f"{ticker}_{ptype}_{interval}_{sig_date}"
        
        # Check if exists
        cursor.execute("SELECT 1 FROM live_alerts WHERE alert_id = ?", (alert_id,))
        if cursor.fetchone():
            continue  # Skip, already exists (whether dismissed or active)
            
        serializable_pat = _make_serializable(pat)
        result_obj = {
            "alert_id": alert_id,
            "ticker": ticker,
            "pattern": PATTERN_NAMES.get(ptype, ptype.upper()),
            "pattern_type": ptype,
            "signal": PATTERN_SIGNALS.get(ptype, "NEUTRAL"),
            "verdict": pat.get("verdict", "UNKNOWN"),
            "quality_score": pat.get("quality_score", 0),
            "key_levels": _extract_pattern_key_levels(pat),
            "checks": _extract_checks(pat),
            "raw": serializable_pat,
            "detected_at": now_str,
            "breakout_timestamp": sig_date,
            "timeframe": interval
        }
        
        # Insert new alert
        cursor.execute('''
            INSERT INTO live_alerts (alert_id, ticker, pattern_type, timeframe, breakout_timestamp, detected_at, pattern_data, dismissed)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0)
        ''', (alert_id, ticker, ptype, interval, sig_date, now_str, json.dumps(result_obj)))
        
        valid_new_results.append(result_obj)
        
    conn.commit()
    conn.close()

    return {
        "count": len(valid_new_results),
        "alerts": valid_new_results,
    }


@app.get("/api/live_alerts", response_model=LiveAlertsResponse)
async def get_live_alerts():
    from db_cache import CACHE_DB_PATH
    conn = sqlite3.connect(CACHE_DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT pattern_data FROM live_alerts
        WHERE dismissed = 0
        ORDER BY detected_at DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    
    alerts = []
    for row in rows:
        try:
            alerts.append(json.loads(row[0]))
        except:
            pass
    return {"alerts": alerts}

@app.post("/api/live_alerts/dismiss", response_model=DismissResponse)
async def dismiss_live_alert(req: DismissRequest):
    from db_cache import CACHE_DB_PATH
    conn = sqlite3.connect(CACHE_DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    now_str = datetime.datetime.now().isoformat()
    cursor.execute('''
        UPDATE live_alerts SET dismissed = 1, dismissed_at = ?
        WHERE alert_id = ?
    ''', (now_str, req.alert_id))
    conn.commit()
    conn.close()
    return {"status": "success"}


# =============================================================================
#  SERVE FRONTEND — Serve static files and index.html
# =============================================================================

app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")

@app.get("/")
async def serve_frontend():
    """Serve the trading terminal frontend at the root URL."""
    index_path = os.path.join(_frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"message": "Frontend not found. Place index.html in the frontend/ directory."}

