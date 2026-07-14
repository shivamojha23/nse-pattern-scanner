import sys
import os

sys.path.insert(0, os.path.abspath('.'))

import core.orchestrator as orch
from fastapi.testclient import TestClient
from backend.main import app

def run_smoke_test():
    print("--- Running Smoke Test ---")
    
    # 1. get_nifty_list
    print("1. Testing get_nifty_list()...")
    tickers = orch.get_nifty_list()
    assert isinstance(tickers, list) and len(tickers) > 0
    print(f"   ✓ Success (found {len(tickers)} tickers)")
    
    # 2. fetch_batch_data (using just 1 ticker for speed)
    print("2. Testing fetch_batch_data()...")
    test_ticker = [tickers[0]]
    data = orch.fetch_batch_data(test_ticker, period="1y", interval="1d")
    if test_ticker[0] not in data:
        print(f"   ⚠ Data fetch failed for {test_ticker[0]}. Trying RELIANCE.NS instead.")
        test_ticker = ["RELIANCE.NS"]
        data = orch.fetch_batch_data(test_ticker, period="1y", interval="1d")
    
    assert test_ticker[0] in data
    print(f"   ✓ Success (fetched data for {test_ticker[0]})")
    
    # 3. scan_ticker
    print("3. Testing scan_ticker()...")
    df = data[test_ticker[0]]
    res_ticker = orch.scan_ticker(df, test_ticker[0], ["cup_and_handle"], interval="1d")
    assert isinstance(res_ticker, dict)
    print("   ✓ Success")
    
    # 4. scan_watchlist
    print("4. Testing scan_watchlist()...")
    res_watchlist = orch.scan_watchlist(test_ticker, ["cup_and_handle"], period="1y", interval="1d")
    assert isinstance(res_watchlist, list)
    print("   ✓ Success")
    
    # 5. FastAPI Endpoints
    print("5. Testing FastAPI Endpoints...")
    client = TestClient(app)
    
    # /api/health
    print("   -> /api/health")
    response = client.get("/api/health")
    assert response.status_code == 200
    
    # /api/market_status
    print("   -> /api/market_status")
    response = client.get("/api/market_status")
    assert response.status_code == 200
    
    # /api/watchlist
    print("   -> /api/watchlist")
    response = client.get("/api/watchlist")
    assert response.status_code == 200
    
    # /api/scan
    print("   -> /api/scan")
    response = client.get(f"/api/scan?pattern=cup_and_handle&interval=1d&lookback=1y")
    assert response.status_code == 200
    
    # /api/candles
    print("   -> /api/candles")
    response = client.get(f"/api/candles?ticker={test_ticker[0]}&interval=1d&lookback=1y")
    assert response.status_code == 200
    
    print("   ✓ All FastAPI endpoints responded successfully!")
    print("\n✅ Smoke test completed without NameErrors or crashes!")

if __name__ == "__main__":
    run_smoke_test()
