# NSE Multi-Pattern Scanner & Trading Terminal v3.0 📊☕

A production-quality Python, FastAPI, and HTML/JS trading suite for detecting **7 chart patterns** in the Indian Stock Market (NSE). 

This project downloads data dynamically from Yahoo Finance, applies price smoothing for peak detection, performs volume confirmation, calculates quality scores, and displays all patterns in a web-based **Trading Terminal UI** using TradingView-style interactive charts.

---

## 🚀 Key Features

* **🧩 7 Chart Patterns Detected:**
  1. **Cup and Handle** (Bullish Continuation)
  2. **Bull Flag** (Bullish Continuation)
  3. **Bear Flag** (Bearish Continuation)
  4. **Pennant** (Continuation)
  5. **Head and Shoulders** (Bearish Reversal)
  6. **Double Top** (Bearish Reversal)
  7. **Double Bottom** (Bullish Reversal)
* **🔌 FastAPI Web Backend:** Provides clean HTTP JSON endpoints for market status, watchlist fetching, live/historical scans, and historical candles.
* **📈 Interactive Frontend Dashboard:** A modern UI with full TradingView charts (Lightweight Charts), drawing pattern markers (rims, bottoms, flags, poles, breakouts) directly on the candles.
* **⚡ Three Scanner Modes:**
  * **Self-Test:** Test pattern detection math instantly with mock/synthetic data (no internet required).
  * **Historical Backtest:** Scan the Nifty watchlist over custom time frames (e.g. 2 years) for already completed patterns.
  * **Live Scanner:** Continuously scan during NSE market hours (Mon-Fri, 9:15 AM – 3:30 PM IST) with custom scan intervals (e.g., every 15 minutes).
* **💾 Local Data Caching:** Uses a local SQLite database to cache historical stock data, significantly speeding up backtests and minimizing Yahoo Finance API rate limits.
* **🎯 Strict Geometric Validation & Volume Confirmation:** Eliminates spiky false positives using volume analysis and strict price relationships.

---

## 🛠️ Project Structure

```
├── backend/
│   ├── __init__.py
│   ├── cache.py          # 15-minute in-memory caching system
│   ├── main.py           # FastAPI application & entry points
│   └── scanner.py        # Adapter connecting backend to detector engine
├── frontend/
│   ├── index.html        # Interactive Trading Terminal dashboard
│   ├── script.js         # Charts initialization and API consumption
│   └── styles.css        # Clean UI styling
├── cup_and_handle_detector.py # Legacy standalone detector
├── pattern_scanner.py         # Main multi-pattern scanner engine (v3.0)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .gitignore                 # Files excluded from git tracking
```

---

## 📦 Setup & Installation

### Prerequisites
* Python 3.8+
* Active internet connection (to fetch data from Yahoo Finance)

### Installation
1. Clone the repository to your local system:
   ```bash
   git clone <your-repository-url>
   cd "Cup and Handle pattern detector"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ How to Run

You can run the suite either as a Command Line Interface (CLI) tool or as a Web Application with a graphical dashboard.

### Option A: Interactive Web UI (Highly Recommended)
Launch the FastAPI web server to run scans, view watchlists, and see interactive charts:
```bash
uvicorn backend.main:app --reload --port 8000
```
Open **`http://localhost:8000`** in your web browser. 

* *Features:* Interactive list of Nifty tickers, instant scanner configurations, custom lookbacks/intervals, and interactive TradingView charts showing exact pattern markers.

---

### Option B: CLI Scanner & Backtester
Run the core scanner from your terminal.

#### 1. Interactive Menu
Guides you step-by-step through configuration:
```bash
python pattern_scanner.py
```

#### 2. Self-Test
Runs instant validation using synthetic data:
```bash
python pattern_scanner.py test [pattern_name]
```

#### 3. Historical Backtest
Scan NSE index watchlists:
```bash
# Scan Nifty index for all patterns over the last 2 years (daily candles)
python pattern_scanner.py historical --pattern all --interval 1d --lookback 2y

# Scan specific tickers only
python pattern_scanner.py historical RELIANCE.NS TCS.NS --pattern cup_and_handle --interval 1d
```

#### 4. Live Continuous Scanner
Run during Indian stock market hours to alert on breakouts:
```bash
python pattern_scanner.py live --pattern all --interval 15m --lookback 59d
```

---

## 📊 Geometric Parameters & Math

The detection engine uses the following parameters for validation:

| Pattern | Detection Principle | Validation Details |
|---|---|---|
| **Cup & Handle** | Smoothed peaks/troughs + Handle retrace | Cup drop 10-35%, Handle retrace ≤32% of cup depth, roundedness check (≥20% of candles in bottom zone). |
| **Bull/Bear Flags** | Sliding-window + linear regression | Pole change ≥8%, channel width ≤5% of pole price, flag duration 5-20 candles, pole $R^2 \ge 0.8$. |
| **Pennants** | Symmetrical converging triangles | Converging price ranges during flag phase, volume decreases during flag formation. |
| **Double Tops/Bottoms** | find_peaks similarity checks | Two peaks/troughs within 3% similarity, separated by at least 10 candles. |
| **Head & Shoulders** | Left/Right shoulders + Head peak | Head must be at least 3% higher than shoulders, shoulder symmetry within 5%. Neckline slope limit 5%. |

---

## 📈 Quality Scoring

Every detected pattern is scored dynamically to help you prioritize the strongest setups:
* **Cup & Handle (Scored ~5-80+):** Based on cup depth, recovery volume ratio, and breakout volume ratio.
* **Flags, Pennants, Reversals (Scored 0.1-10.0):** Standardized scores based on pole strength, channel tightness, $R^2$ linear fit, and volume confirmations.

---

## 💾 Output Logs
* **`results.txt`:** Standard output and scan details are logged locally here.
* **`backtest_results.txt`:** Generated automatically to store detailed results from historical backtest sweeps.

---

## ⚖️ License
This project is licensed under the MIT License.
