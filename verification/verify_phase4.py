import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from pattern_scanner import scan_watchlist

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
    print(f"Generating Phase 4 snapshot: {phase_name}")
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    
    from pattern_scanner import ALL_PATTERNS
    results = scan_watchlist(tickers, ALL_PATTERNS, period="1y", interval="1d")
    
    with open(f"verification/{phase_name}_phase4.json", "w") as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print("Snapshot saved.")

def compare_snapshots():
    print("\n--- Comparing Phase 4 Snapshots ---")
    
    with open("verification/Before_phase4.json") as f: before = json.load(f)
    with open("verification/After_phase4.json") as f: after = json.load(f)
    
    if before == after:
        print("[PASS] The output is 100% identical. Orchestration extraction did not alter behavior.")
    else:
        print("[FAIL] The output changed! This should be identity preserving.")
        # Find differences
        for t in before:
            if t not in after or before[t] != after.get(t):
                print(f"  -> Differences found in ticker: {t}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "before":
        generate_snapshot("Before")
    elif len(sys.argv) > 1 and sys.argv[1] == "after":
        generate_snapshot("After")
    elif len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_snapshots()
