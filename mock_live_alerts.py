import sqlite3
import json
import datetime
from db_cache import CACHE_DB_PATH

conn = sqlite3.connect(CACHE_DB_PATH, timeout=30.0)
cursor = conn.cursor()

now_str = datetime.datetime.now().isoformat()

# Clear existing to get a clean screenshot
cursor.execute("DELETE FROM live_alerts")

confirmed_alert = {
    "alert_id": "RELIANCE_bull_flag_1d_2024-05-15",
    "ticker": "RELIANCE.NS",
    "pattern": "Bull Flag",
    "pattern_type": "bull_flag",
    "signal": "BULLISH",
    "status": "confirmed",
    "verdict": "Confirmed Breakout",
    "quality_score": 9.5,
    "key_levels": [],
    "checks": [],
    "raw": {},
    "detected_at": now_str,
    "breakout_timestamp": "2024-05-15",
    "timeframe": "1d"
}

forming_alert = {
    "alert_id": "TCS_head_and_shoulders_1d_2024-05-15",
    "ticker": "TCS.NS",
    "pattern": "Head & Shoulders",
    "pattern_type": "head_and_shoulders",
    "signal": "BEARISH",
    "status": "forming",
    "verdict": "Right Shoulder Forming",
    "quality_score": 7.2,
    "key_levels": [],
    "checks": [],
    "raw": {},
    "detected_at": now_str,
    "breakout_timestamp": "2024-05-15",
    "timeframe": "1d"
}

cursor.execute('''
    INSERT INTO live_alerts (alert_id, ticker, pattern_type, timeframe, breakout_timestamp, detected_at, pattern_data, dismissed)
    VALUES (?, ?, ?, ?, ?, ?, ?, 0)
''', (confirmed_alert["alert_id"], confirmed_alert["ticker"], confirmed_alert["pattern_type"], confirmed_alert["timeframe"], confirmed_alert["breakout_timestamp"], confirmed_alert["detected_at"], json.dumps(confirmed_alert)))

cursor.execute('''
    INSERT INTO live_alerts (alert_id, ticker, pattern_type, timeframe, breakout_timestamp, detected_at, pattern_data, dismissed)
    VALUES (?, ?, ?, ?, ?, ?, ?, 0)
''', (forming_alert["alert_id"], forming_alert["ticker"], forming_alert["pattern_type"], forming_alert["timeframe"], forming_alert["breakout_timestamp"], forming_alert["detected_at"], json.dumps(forming_alert)))

conn.commit()
conn.close()
print("Mock live alerts inserted successfully.")
