import pytest
from core.orchestrator import scan_ticker, scan_watchlist, fetch_batch_data

# Our chosen sample tickers representing standard, sparse, and recent IPO edge cases
SAMPLE_TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS",   # Standard
    "ALKEM.NS", "MRF.NS", "TITAN.NS", # Sparse (few patterns)
    "JIOFIN.NS", "IREDA.NS"           # Recent IPO (limited history)
]

def test_fetch_batch_data(snapshot):
    # Fetch data for our samples (3 months lookback for speed, ensures > 30 candles)
    df_dict = fetch_batch_data(SAMPLE_TICKERS, period="3mo", interval="1d")
    
    # We don't snapshot the entire data frame contents because numbers change over time
    # Instead, we snapshot the structural keys (tickers that returned data)
    keys = sorted(list(df_dict.keys()))
    snapshot("core_fetch_batch_keys", keys)

def test_scan_ticker(snapshot):
    # We will test scan_ticker on RELIANCE for 3mo lookback
    df_dict = fetch_batch_data(["RELIANCE.NS"], period="3mo", interval="1d")
    df = df_dict.get("RELIANCE.NS")
    
    # We can't guarantee patterns will be found in exactly 3mo, but the structural output
    # will be a dictionary of pattern lists. We snapshot that structure.
    # Note: if it returns 0 patterns, the snapshot will be {} or empty lists, which is fine!
    results = scan_ticker(df, "RELIANCE.NS", ["all"], interval="1d")
    snapshot("core_scan_ticker_reliance_3mo", results)

def test_scan_watchlist(snapshot):
    # scan_watchlist uses live mode filtering, returning a list of new alerts
    # We will pass the full SAMPLE_TICKERS. It will probably return [] because
    # patterns rarely complete exactly on "today". That's fine, the snapshot will just be [].
    # But if one does complete, we capture it.
    results = scan_watchlist(SAMPLE_TICKERS, ["all"], period="3mo", interval="1d")
    snapshot("core_scan_watchlist_3mo_sample", results)
