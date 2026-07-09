import sqlite3
import pandas as pd
import json
import yfinance as yf
import datetime
import os
import time
import logging
import re
from zoneinfo import ZoneInfo

def get_ist_today():
    return datetime.datetime.now(ZoneInfo("Asia/Kolkata")).date()

CACHE_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache.db')

def init_db():
    """Initialize the SQLite database and create tables for Layer 1."""
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    
    # Layer 1: Raw Candle Data Cache
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS layer1_candles (
        ticker TEXT,
        interval TEXT,
        timestamp INTEGER,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        PRIMARY KEY (ticker, interval, timestamp)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS layer1_meta (
        ticker TEXT,
        interval TEXT,
        earliest_date TEXT,
        last_updated_date TEXT,
        last_reconciled_date TEXT,
        PRIMARY KEY (ticker, interval)
    )
    ''')
    
    # Layer 2: Watchlist Cache
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS layer2_watchlist (
        list_name TEXT PRIMARY KEY,
        tickers TEXT,
        fetched_at TEXT
    )
    ''')
    # Layer 3: Backtest Result Cache
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS layer3_backtest (
        ticker TEXT,
        interval TEXT,
        pattern_type TEXT,
        start_date TEXT,
        end_date TEXT,
        param_hash TEXT,
        algo_version INTEGER,
        results TEXT,
        computed_at REAL,
        PRIMARY KEY (ticker, interval, pattern_type, start_date, end_date, param_hash, algo_version)
    )
    ''')
    
    # Layer 4: Live Scan Cache
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS layer4_dedup (
        ticker TEXT,
        pattern_type TEXT,
        alert_date TEXT,
        PRIMARY KEY (ticker, pattern_type, alert_date)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS layer4_structural (
        ticker TEXT,
        interval TEXT,
        pattern_type TEXT,
        structural_data TEXT,
        PRIMARY KEY (ticker, interval, pattern_type)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS layer4_alert_history (
        ticker TEXT,
        pattern_type TEXT,
        signal_date TEXT,
        pattern_data TEXT,
        status TEXT,
        PRIMARY KEY (ticker, pattern_type, signal_date)
    )
    ''')

    conn.commit()
    
    # TTL Cleanup for Layer 3 (Delete rows older than 3600 seconds)
    current_time = time.time()
    cursor.execute('DELETE FROM layer3_backtest WHERE computed_at < ?', (current_time - 3600,))
    
    conn.commit()
    conn.close()

init_db()

def _parse_lookback_to_days(lookback: str) -> int:
    """Converts a yfinance lookback string (e.g., '3mo', '1y') into approximate days."""
    if lookback.endswith('mo'):
        return int(lookback[:-2]) * 30
    elif lookback.endswith('y'):
        return int(lookback[:-1]) * 365
    elif lookback.endswith('d'):
        return int(lookback[:-1])
    elif lookback.endswith('wk'):
        return int(lookback[:-2]) * 7
    elif lookback == 'max':
        return 365 * 100
    return 365 # Default 1y

def _df_to_db(conn, df, ticker, interval):
    """Inserts a yfinance DataFrame into the layer1_candles table."""
    if df is None or df.empty:
        return
        
    # Handle multiindex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel('Ticker')
        
    records = []
    for idx, row in df.iterrows():
        try:
            # Convert timestamp to UTC UNIX integer
            ts = int(pd.Timestamp(idx).timestamp())
            
            # yfinance column names
            open_val = float(row['Open'])
            high_val = float(row['High'])
            low_val = float(row['Low'])
            close_val = float(row['Close'])
            vol_val = int(row['Volume']) if not pd.isna(row['Volume']) else 0
            
            # Skip invalid rows
            if any(pd.isna([open_val, high_val, low_val, close_val])):
                continue
                
            records.append((ticker, interval, ts, open_val, high_val, low_val, close_val, vol_val))
        except (ValueError, KeyError, TypeError):
            continue
            
    if records:
        cursor = conn.cursor()
        cursor.executemany('''
            INSERT OR REPLACE INTO layer1_candles 
            (ticker, interval, timestamp, open, high, low, close, volume) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', records)
        conn.commit()

def _reconcile_corporate_actions(conn, ticker, interval, today_str):
    """
    Weekly reconciliation: Re-fetches the last 30 days of data and overwrites the cache
    to capture retroactive splits, dividends, or corrections from yfinance.
    """
    cursor = conn.cursor()
    logging.info(f"Reconciling {ticker} ({interval}) for corporate actions...")
    
    # Fetch last 30 days
    df = yf.download(tickers=ticker, period='1mo', interval=interval, progress=False)
    if not df.empty:
        _df_to_db(conn, df, ticker, interval)
        
        # Clear Layer 4 structural cache for this ticker because prices changed
        cursor.execute("DELETE FROM layer4_structural WHERE ticker = ? AND interval = ?", (ticker, interval))
        
        # Update reconciliation date
        cursor.execute('''
            UPDATE layer1_meta 
            SET last_reconciled_date = ? 
            WHERE ticker = ? AND interval = ?
        ''', (today_str, ticker, interval))
        conn.commit()

def get_stock_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    """
    Layer 1 Cache Wrapper. Replaces `yf.download`.
    Fetches data from SQLite cache, pulling delta updates or full history from yfinance if necessary.
    """
    conn = sqlite3.connect(CACHE_DB_PATH)
    
    today_dt = get_ist_today()
    today_str = today_dt.strftime('%Y-%m-%d')
    
    # 1. Check cache meta
    cursor = conn.cursor()
    cursor.execute("SELECT earliest_date, last_updated_date, last_reconciled_date FROM layer1_meta WHERE ticker = ? AND interval = ?", (ticker, interval))
    row = cursor.fetchone()
    
    lookback_days = _parse_lookback_to_days(period)
    requested_start_dt = today_dt - datetime.timedelta(days=lookback_days)
    
    needs_full_fetch = False
    
    if row is None:
        needs_full_fetch = True
    else:
        earliest_date_str, last_updated_str, last_reconciled_str = row
        earliest_date = datetime.datetime.strptime(earliest_date_str, '%Y-%m-%d').date()
        
        if earliest_date > requested_start_dt + datetime.timedelta(days=5):
            # We don't have enough history in cache. (Added 5 days leeway for weekends/holidays)
            needs_full_fetch = True
        else:
            # We have enough history. Do a delta fetch if we haven't updated today.
            # To handle the provisional window, we fetch from (last_updated_date - 2 days).
            last_updated_dt = datetime.datetime.strptime(last_updated_str, '%Y-%m-%d').date()
            if last_updated_dt <= today_dt:
                delta_start = last_updated_dt - datetime.timedelta(days=2)
                delta_start_str = delta_start.strftime('%Y-%m-%d')
                
                # Fetch delta
                df_delta = yf.download(tickers=ticker, start=delta_start_str, interval=interval, progress=False)
                _df_to_db(conn, df_delta, ticker, interval)
                
                # Update last_updated_date (we don't change earliest_date)
                cursor.execute("UPDATE layer1_meta SET last_updated_date = ? WHERE ticker = ? AND interval = ?", (today_str, ticker, interval))
                conn.commit()
                
            # Check for corporate action reconciliation (every 7 days)
            last_recon_dt = datetime.datetime.strptime(last_reconciled_str, '%Y-%m-%d').date() if last_reconciled_str else None
            if last_recon_dt is None or (today_dt - last_recon_dt).days >= 7:
                _reconcile_corporate_actions(conn, ticker, interval, today_str)

    if needs_full_fetch:
        # Fetch full requested period
        df_full = yf.download(tickers=ticker, period=period, interval=interval, progress=False)
        _df_to_db(conn, df_full, ticker, interval)
        
        # Calculate earliest date actually received
        if not df_full.empty:
            actual_earliest_dt = df_full.index.min().date()
            actual_earliest_str = actual_earliest_dt.strftime('%Y-%m-%d')
            
            cursor.execute('''
                INSERT OR REPLACE INTO layer1_meta 
                (ticker, interval, earliest_date, last_updated_date, last_reconciled_date)
                VALUES (?, ?, ?, ?, ?)
            ''', (ticker, interval, actual_earliest_str, today_str, today_str))
            conn.commit()
            
    # 2. Retrieve data from cache to return as DataFrame
    cutoff_ts = int(datetime.datetime.combine(requested_start_dt, datetime.time.min).replace(tzinfo=datetime.timezone.utc).timestamp())
    
    query = '''
        SELECT timestamp, open as Open, high as High, low as Low, close as Close, volume as Volume
        FROM layer1_candles
        WHERE ticker = ? AND interval = ? AND timestamp >= ?
        ORDER BY timestamp ASC
    '''
    df_cached = pd.read_sql_query(query, conn, params=(ticker, interval, cutoff_ts))
    
    if not df_cached.empty:
        df_cached['timestamp'] = pd.to_datetime(df_cached['timestamp'], unit='s', utc=True)
        df_cached.set_index('timestamp', inplace=True)
        df_cached.index.name = 'Date' if interval.endswith('d') or interval.endswith('wk') or interval.endswith('mo') else 'Datetime'
        
    conn.close()
    return df_cached

def validate_watchlist(tickers, list_name):
    if not tickers:
        return False, "Empty list"
    
    # Check bounds
    if list_name == "nifty100":
        if not (90 <= len(tickers) <= 110):
            return False, f"expected ~100 tickers, got {len(tickers)}"
    elif list_name == "nifty200":
        if not (190 <= len(tickers) <= 210):
            return False, f"expected ~200 tickers, got {len(tickers)}"
    elif list_name == "nifty50":
        if not (45 <= len(tickers) <= 55):
            return False, f"expected ~50 tickers, got {len(tickers)}"
    
    # Check format (simple sanity check on first few items)
    valid_pattern = re.compile(r'^[A-Z0-9\-&]+\.NS$')
    for t in tickers[:5]:
        if not valid_pattern.match(t):
            return False, f"invalid ticker format detected: {t}"
            
    return True, ""

def get_cached_watchlist(list_name="nifty200") -> list:
    """
    Layer 2 Cache. Fetches Nifty watchlist and caches it.
    """
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    
    today_dt = get_ist_today()
    today_str = today_dt.strftime('%Y-%m-%d')
    
    cursor.execute("SELECT tickers, fetched_at FROM layer2_watchlist WHERE list_name = ?", (list_name,))
    row = cursor.fetchone()
    
    if row is not None:
        cached_tickers, fetched_at = row
        if fetched_at == today_str:
            conn.close()
            return json.loads(cached_tickers)
            
    # Need to fetch
    csv_urls = [
        "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
        "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
        "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    ]
    
    fetched_list = None
    for url in csv_urls:
        try:
            logging.info(f"Trying to fetch watchlist from {url}")
            df = pd.read_csv(url)
            if "Symbol" in df.columns:
                fetched_list = [f"{sym.strip()}.NS" for sym in df["Symbol"].tolist()]
                break
        except Exception as e:
            logging.warning(f"Failed to fetch {url}: {e}")
            
    if fetched_list is not None:
        is_valid, err_msg = validate_watchlist(fetched_list, list_name)
        if not is_valid:
            logging.warning(f"Watchlist validation failed: {err_msg}")
            fetched_list = None
        else:
            # Save to cache
            cursor.execute('''
                INSERT OR REPLACE INTO layer2_watchlist (list_name, tickers, fetched_at)
                VALUES (?, ?, ?)
            ''', (list_name, json.dumps(fetched_list), today_str))
            conn.commit()
            conn.close()
            return fetched_list
        
    # Fallback if fetch failed
    if row is not None:
        logging.warning("Watchlist fetch failed. Returning stale cached list.")
        conn.close()
        return json.loads(row[0])
        
    logging.warning("All web fetches failed and no cache exists. Using hardcoded Nifty 50 fallback.")
    fallback = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
        "LT", "AXISBANK", "ASIANPAINT", "HCLTECH", "MARUTI",
        "TITAN", "BAJFINANCE", "SUNPHARMA", "TATASTEEL", "M&M",
        "ULTRACEMCO", "POWERGRID", "NTPC", "TATAMOTORS", "TECHM",
        "WIPRO", "NESTLEIND", "ONGC", "JSWSTEEL", "HDFCLIFE",
        "GRASIM", "HINDALCO", "BAJAJFINSV", "ADANIPORTS", "ADANIENT",
        "DIVISLAB", "CIPLA", "BRITANNIA", "APOLLOHOSP", "HEROMOTOCO",
        "DRREDDY", "BAJAJ-AUTO", "EICHERMOT", "TATACONSUM", "COALINDIA",
        "UPL", "INDUSINDBK", "SBILIFE", "LTIM", "BPCL"
    ]
    conn.close()
    return [f"{sym}.NS" for sym in fallback]
