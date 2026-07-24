import pytest
from unittest.mock import patch

# Our chosen sample tickers representing standard, sparse, and recent IPO edge cases
SAMPLE_TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS",   # Standard
    "ALKEM.NS", "MRF.NS", "TITAN.NS", # Sparse (few patterns)
    "JIOFIN.NS", "IREDA.NS"           # Recent IPO (limited history)
]

def test_api_health(client, snapshot):
    response = client.get("/api/health")
    assert response.status_code == 200
    snapshot("api_health", response.json())

def test_api_market_status(client, snapshot):
    response = client.get("/api/market_status")
    assert response.status_code == 200
    # market status is dynamic (time/is_open changes). We snapshot keys only, or a sanitized version.
    data = response.json()
    # Mask dynamic values to prevent flaky snapshots
    for key in ["ist_time", "ist_date", "next_event", "status"]:
        if key in data:
            data[key] = "<DYNAMIC>"
    snapshot("api_market_status", data)

def test_api_watchlist(client, snapshot):
    response = client.get("/api/watchlist")
    assert response.status_code == 200
    snapshot("api_watchlist", response.json())

@patch('backend.main.get_nifty_list')
def test_api_scan(mock_get_list, client, snapshot):
    mock_get_list.return_value = SAMPLE_TICKERS
    # lookback 1y to ensure JIOFIN/IREDA hit the "not enough data" edge case if applicable
    response = client.get("/api/scan?pattern=all&lookback=1y")
    assert response.status_code == 200
    snapshot("api_scan_1y_sample", response.json())

def test_api_candles_standard(client, snapshot):
    response = client.get("/api/candles?ticker=RELIANCE.NS&interval=1d&lookback=1mo")
    assert response.status_code == 200
    snapshot("api_candles_reliance_1mo", response.json())

def test_api_candles_recent_ipo(client, snapshot):
    response = client.get("/api/candles?ticker=JIOFIN.NS&interval=1d&lookback=1mo")
    assert response.status_code == 200
    snapshot("api_candles_jiofin_1mo", response.json())

@patch('backend.main.get_nifty_list')
def test_api_live_scan(mock_get_list, client, snapshot):
    mock_get_list.return_value = SAMPLE_TICKERS
    response = client.get("/api/live_scan?patterns=all&lookback=1mo")
    assert response.status_code == 200
    snapshot("api_live_scan_1mo_sample", response.json())

def test_api_live_alerts(client, mock_live_alert, snapshot):
    # mock_live_alert fixture inserts our test alert into the DB
    response = client.get("/api/live_alerts")
    assert response.status_code == 200
    
    # We expect our mock_test_alert_123 to be in the response.
    # Other alerts might be present if the DB has real data, so we filter to only snapshot our mock alert.
    data = response.json()
    my_alert = [a for a in data.get("alerts", []) if a.get("alert_id") == mock_live_alert]
    
    snapshot("api_live_alerts", {"mock_alert": my_alert})

def test_api_live_alerts_dismiss(client, mock_live_alert, snapshot):
    # Send dismiss request
    response = client.post("/api/live_alerts/dismiss", json={"alert_id": mock_live_alert})
    assert response.status_code == 200
    snapshot("api_live_alerts_dismiss", response.json())
    
    # Verify it was dismissed by fetching again
    fetch_response = client.get("/api/live_alerts")
    data = fetch_response.json()
    my_alert = [a for a in data.get("alerts", []) if a.get("alert_id") == mock_live_alert]
    assert len(my_alert) == 0, "Alert should have been omitted from active list after dismissal"

def test_frontend_ui(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<html" in response.text.lower()
    # We don't snapshot the entire HTML, just verify it serves successfully
