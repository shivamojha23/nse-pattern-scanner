import sqlite3
import pytest
import os
import json
import difflib
import sys
from fastapi.testclient import TestClient
from fastapi.encoders import jsonable_encoder
from backend.api_models import _make_serializable

# Must add project root to path before imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.main import app
from db_cache import get_db_connection

def pytest_addoption(parser):
    parser.addoption("--update-snapshots", action="store_true", help="Update golden snapshots")

@pytest.fixture
def client():
    return TestClient(app)

def _sanitize_data(data):
    """
    Recursively removes highly dynamic fields like 'scan_time' or 'scan_duration_seconds' 
    so they don't break snapshots on every run.
    """
    if isinstance(data, dict):
        return {k: _sanitize_data(v) for k, v in data.items() if k not in ("scan_time", "scan_duration_seconds", "detected_at")}
    elif isinstance(data, list):
        return [_sanitize_data(item) for item in data]
    return data

@pytest.fixture
def snapshot(request):
    update = request.config.getoption("--update-snapshots")
    snapshots_dir = os.path.join(os.path.dirname(__file__), "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)
    
    def _assert_match(name, raw_data):
        filepath = os.path.join(snapshots_dir, f"{name}.json")
        
        # Strip dynamic timestamps and handle Numpy types
        clean_data = _sanitize_data(raw_data)
        serializable_data = _make_serializable(clean_data)
        current_str = json.dumps(serializable_data, indent=2, sort_keys=True)
        
        if not os.path.exists(filepath):
            if update:
                print(f"\n[SNAPSHOT] Creating NEW snapshot: {name}")
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(current_str)
                return
            else:
                pytest.fail(f"Snapshot {name} does not exist. Run pytest -s --update-snapshots to create it.")
                
        with open(filepath, "r", encoding="utf-8") as f:
            old_str = f.read()
            
        if current_str != old_str:
            if update:
                diff = "\n".join(difflib.unified_diff(
                    old_str.splitlines(),
                    current_str.splitlines(),
                    fromfile="old",
                    tofile="new"
                ))
                print(f"\n[SNAPSHOT] Mismatch in {name}. Diff:\n{diff}\n")
                ans = input(f"Update snapshot {name}? (y/N): ")
                if ans.lower() == 'y':
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(current_str)
                    print("Updated.")
                else:
                    pytest.fail(f"Snapshot {name} update rejected by user.")
            else:
                pytest.fail(f"Snapshot mismatch for {name}. Run with -s --update-snapshots to review and update.")
                
    return _assert_match

@pytest.fixture
def mock_live_alert():
    """
    Setup and Teardown for POST /api/live_alerts/dismiss tests.
    Ensures complete isolation and self-healing.
    """
    alert_id = "mock_test_alert_123"
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # SETUP: Self-heal by proactively deleting any orphaned row
    cursor.execute("DELETE FROM live_alerts WHERE alert_id = ?", (alert_id,))
    conn.commit()
    
    # Insert the fresh mock alert
    mock_pattern_data = {
        "alert_id": alert_id,
        "ticker": "RELIANCE.NS",
        "pattern": "CUP AND HANDLE",
        "pattern_type": "cup_and_handle",
        "signal": "BULLISH",
        "verdict": "UNKNOWN",
        "quality_score": 0.0,
        "key_levels": [],
        "checks": [],
        "raw": {},
        "detected_at": "2023-01-02",
        "breakout_timestamp": "2023-01-01",
        "timeframe": "1d"
    }
    cursor.execute('''
        INSERT INTO live_alerts (alert_id, ticker, pattern_type, timeframe, breakout_timestamp, detected_at, pattern_data, dismissed)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0)
    ''', (alert_id, "RELIANCE.NS", "cup_and_handle", "1d", "2023-01-01", "2023-01-02", json.dumps(mock_pattern_data)))
    conn.commit()
    
    try:
        yield alert_id
    finally:
        # TEARDOWN: Try/finally cleanup to guarantee execution
        cursor.execute("DELETE FROM live_alerts WHERE alert_id = ?", (alert_id,))
        conn.commit()
