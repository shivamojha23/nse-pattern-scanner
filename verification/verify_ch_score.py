from pattern_scanner import compute_quality_score
import json

metrics = {
    "cup_drop_pct": 20.0,
    "recovery_vol_ratio": 1.5,
    "breakout_vol_ratio": 2.0
}

score = compute_quality_score("cup_and_handle", metrics)
print(f"[{'AFTER' if score <= 10 else 'BEFORE'}] Cup & Handle Score for metrics {json.dumps(metrics)}: {score}")
