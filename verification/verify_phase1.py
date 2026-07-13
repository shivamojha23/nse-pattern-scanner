import sqlite3
import pandas as pd
import json
import os
import sys

from db_cache import get_stock_data, get_cached_watchlist, CACHE_DB_PATH

def save_snapshot(name):
    print(f"Generating snapshot: {name}")
    
    # 1. Standard test (Daily)
    df_rel = get_stock_data("RELIANCE.NS", "1d", "3mo")
    df_rel.to_pickle(f"{name}_rel.pkl")
    
    # 2. IST-midnight boundary test (Intraday)
    # Using 15m to catch multiple candles, including the day transition at midnight IST
    df_tcs = get_stock_data("TCS.NS", "15m", "5d")
    df_tcs.to_pickle(f"{name}_tcs.pkl")
    
    # 3. Watchlist test
    wl = get_cached_watchlist("nifty50")
    with open(f"{name}_wl.json", "w") as f:
        json.dump(wl, f)
        
    # 4. Raw DB query test
    conn = sqlite3.connect(CACHE_DB_PATH)
    candles = pd.read_sql_query("SELECT * FROM layer1_candles WHERE ticker='TCS.NS'", conn)
    candles.to_pickle(f"{name}_raw_tcs.pkl")
    conn.close()

def compare_snapshots():
    print("Comparing snapshots...")
    
    try:
        # 1. Standard test
        df_rel_b = pd.read_pickle("before_rel.pkl")
        df_rel_a = pd.read_pickle("after_rel.pkl")
        pd.testing.assert_frame_equal(df_rel_b, df_rel_a)
        print("[PASS] RELIANCE standard test")
        
        # 2. IST-boundary test
        df_tcs_b = pd.read_pickle("before_tcs.pkl")
        df_tcs_a = pd.read_pickle("after_tcs.pkl")
        pd.testing.assert_frame_equal(df_tcs_b, df_tcs_a)
        print("[PASS] TCS IST-boundary test")
        
        # 3. Watchlist test
        with open("before_wl.json") as f: wl_b = json.load(f)
        with open("after_wl.json") as f: wl_a = json.load(f)
        assert wl_b == wl_a, "Watchlists do not match!"
        print("[PASS] Watchlist test")
        
        # 4. Raw DB query test
        raw_b = pd.read_pickle("before_raw_tcs.pkl")
        raw_a = pd.read_pickle("after_raw_tcs.pkl")
        pd.testing.assert_frame_equal(raw_b, raw_a)
        print("[PASS] Raw DB query test (layer1_candles)")
        
        print("\nAll Phase 1 verifications passed!")
    except Exception as e:
        print(f"\n[FAIL] Verification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "before":
        save_snapshot("before")
    elif len(sys.argv) > 1 and sys.argv[1] == "after":
        save_snapshot("after")
    elif len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_snapshots()
    else:
        print("Usage: python verify_phase1.py [before|after|compare]")
