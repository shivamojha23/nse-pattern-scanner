import json
import sys
import os
import db_cache
from pattern_scanner import detect_cup_and_handle, detect_head_and_shoulders

def generate_snapshot(phase_name):
    print(f"Generating Phase 2 snapshot: {phase_name}")
    df = db_cache.get_stock_data("RELIANCE.NS", "1d", "1y")
    prices = df['Close'].values
    highs = df['High'].values
    volumes = df['Volume'].values
    lows = df['Low'].values
    dates = df.index
    
    # 1. Cup and Handle (Behavior Change Expected)
    ch_patterns = detect_cup_and_handle(prices, highs, volumes, "RELIANCE.NS", dates, "1d")
    with open(f"{phase_name}_ch.json", "w") as f:
        json.dump(ch_patterns, f, indent=2)
        
    # 2. Head & Shoulders (Strict Identity Expected)
    # The only place refine_peak/trough is called!
    hs_patterns = detect_head_and_shoulders(prices, highs, lows, volumes, "RELIANCE.NS", dates, "1d")
    with open(f"{phase_name}_hs.json", "w") as f:
        json.dump(hs_patterns, f, indent=2)

def compare_snapshots():
    print("\n--- Comparing Phase 2 Snapshots ---")
    
    with open("Before_hs.json") as f: hs_before = json.load(f)
    with open("After_hs.json") as f: hs_after = json.load(f)
    
    if hs_before == hs_after:
        print("[PASS] Head & Shoulders output is 100% identical. Parameterizing refine_peak did not alter default behavior.")
    else:
        print("[FAIL] Head & Shoulders output changed! This should be identity preserving.")
        sys.exit(1)
        
    with open("Before_ch.json") as f: ch_before = json.load(f)
    with open("After_ch.json") as f: ch_after = json.load(f)
    
    print("\n--- Cup & Handle Scoring Diff ---")
    for i in range(min(len(ch_before), len(ch_after))):
        p_b = ch_before[i]
        p_a = ch_after[i]
        score_b = p_b['quality_score']
        score_a = p_a['quality_score']
        
        del p_b['quality_score']
        del p_a['quality_score']
        
        if p_b == p_a:
            print(f"Pattern {i+1}: Output exactly matches EXCEPT score changed from {score_b} -> {score_a}")
        else:
            print(f"[FAIL] Pattern {i+1} has unexpected structural changes!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "before":
        generate_snapshot("Before")
    elif len(sys.argv) > 1 and sys.argv[1] == "after":
        generate_snapshot("After")
    elif len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_snapshots()
