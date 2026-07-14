import json
import sys
import numpy as np
import pandas as pd
import db_cache
from pattern_scanner import scan_ticker, ALL_PATTERNS

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return super(NpEncoder, self).default(obj)

def generate_snapshot(phase_name):
    print(f"Generating Phase 3b snapshot: {phase_name}")
    
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    
    all_results = {}
    for t in tickers:
        print(f"Scanning {t}...")
        df = db_cache.get_stock_data(t, "1d", "1y")
        patterns = scan_ticker(df, t, ALL_PATTERNS, interval="1d")
        all_results[t] = patterns
        
    with open(f"{phase_name}_phase3b.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)
    print("Snapshot saved.")

def compare_snapshots():
    print("\n--- Comparing Phase 3b Snapshots ---")
    
    with open("Before_phase3b.json") as f: before = json.load(f)
    with open("After_phase3b.json") as f: after = json.load(f)
    
    if before == after:
        print("[PASS] The output is 100% identical. The logic flattening and cache removal did not alter behavior.")
    else:
        print("[FAIL] The output changed! This should be identity preserving.")
        # Find differences
        for t in before:
            if before[t] != after[t]:
                print(f"  -> Differences found in ticker: {t}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "before":
        generate_snapshot("Before")
    elif len(sys.argv) > 1 and sys.argv[1] == "after":
        generate_snapshot("After")
    elif len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_snapshots()
