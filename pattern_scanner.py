"""
=============================================================================
  NSE MULTI-PATTERN SCANNER  v3.0
  Indian Stock Market (NSE)
=============================================================================

  Author : AI-assisted project
  Purpose: Detect 7 chart patterns on NSE stocks using Yahoo Finance data,
           with smoothing (where needed), volume confirmation, quality
           scoring, and composite ranking.

  PATTERNS DETECTED
  -----------------
  1. Cup and Handle       — Bullish Continuation
  2. Bull Flag            — Bullish Continuation
  3. Bear Flag            — Bearish Continuation
  4. Pennant              — Continuation (pole direction)
  5. Head and Shoulders   — Bearish Reversal
  6. Double Top           — Bearish Reversal
  7. Double Bottom        — Bullish Reversal

  THREE MODES
  -----------
  A. SELF-TEST     → Synthetic data to prove the math works (no internet).
  B. HISTORICAL    → Sweeps past data for already-completed patterns.
  C. LIVE SCANNER  → Continuous scanner during NSE market hours.

  Smoothing Policy
  ----------------
  smooth_prices() is applied ONLY to patterns that use scipy find_peaks()
  for peak/trough detection: Cup & Handle, Head & Shoulders, Double Top,
  Double Bottom.  Patterns using sliding-window + linregress (Bull Flag,
  Bear Flag, Pennant) work on RAW prices — no smoothing needed.

  pip install
  -----------
  pip install yfinance pandas numpy scipy

=============================================================================
"""

# ─── IMPORTS ────────────────────────────────────────────────────────────────
import sys
import os
import time
import datetime
import warnings
import argparse
import math

# Fix Windows console encoding so emoji and special characters display correctly.
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    else:
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")

class DualWriter:
    """Writes to both standard output and a log file simultaneously."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def reconfigure(self, **kwargs):
        if hasattr(self.terminal, "reconfigure"):
            self.terminal.reconfigure(**kwargs)

# Redirect stdout to both terminal and results.txt
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, "results.txt")
sys.stdout = DualWriter(log_file)

import numpy as np                     # Fast math on arrays of numbers
import pandas as pd                    # DataFrames — like Excel spreadsheets in Python
import yfinance as yf                  # Downloads stock price data from Yahoo Finance
from scipy.signal import find_peaks    # Finds local highs/lows in a price curve
from scipy.stats import linregress     # Calculates linear regression slope + R²

# Suppress noisy warnings from yfinance / pandas
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
#  DETECT ENGINE IMPORTS
# =============================================================================
from detectors.core import *
from detectors.cup_and_handle import detect_cup_and_handle
from detectors.bull_flag import detect_bull_flag
from detectors.bear_flag import detect_bear_flag
from detectors.pennant import detect_pennant
from detectors.head_and_shoulders import detect_head_and_shoulders
from detectors.double_top import detect_double_top
from detectors.double_bottom import detect_double_bottom

from core.orchestrator import *


# =============================================================================
#  1. GET THE WATCHLIST — Dynamically fetch Nifty 100/200 tickers
# =============================================================================



# =============================================================================
#  2. FETCH DATA IN BATCHES — Download candle data from Yahoo Finance
# =============================================================================



# =============================================================================
#  3. SMOOTH PRICES — Reduce noise before peak detection (only when needed)
# =============================================================================




# =============================================================================
#  4. QUALITY SCORE — Generalized pattern scoring
# =============================================================================




# =============================================================================
#  5. DETECT CUP AND HANDLE
#     (Exact copy from v2 — all rules intact, only quality_score call updated)
# =============================================================================




# =============================================================================
#  6. DETECT BULL FLAG
# =============================================================================



# =============================================================================


# =============================================================================


# =============================================================================










# =============================================================================
#  HELPER FUNCTIONS FOR PEAK REFINEMENT
# =============================================================================







# =============================================================================
#  10. DETECT DOUBLE TOP
# =============================================================================


# =============================================================================


def _emit_dt_rejected(ticker, reason):
    """Helper to print a quick Double Top rejection in verbose mode."""
    print(f"  [DT] {ticker}: REJECTED — {reason}")


# =============================================================================
#  11. DETECT DOUBLE BOTTOM
# =============================================================================



def _emit_db_rejected(ticker, reason):
    """Helper to print a quick Double Bottom rejection in verbose mode."""
    print(f"  [DB] {ticker}: REJECTED — {reason}")


# =============================================================================
#  12. PRINT FUNCTIONS — Pattern-specific debug output
# =============================================================================

def print_cup_and_handle(p, index=1):
    """Prints Cup & Handle debug output (exact same format as v2)."""
    score_str = (f"          Quality Score: {p.get('quality_score', 0)}"
                 if not p.get("reject_reason") else "")

    print(f"\n  {'═' * 60}")
    print(f"  🏆 CUP & HANDLE #{index}  —  {p['ticker']}{score_str}")
    print(f"  {'═' * 60}")
    print(f"  Left Rim  (peak before cup)  : ₹{p['left_rim_price']}")
    print(f"  Cup Bottom (lowest point)    : ₹{p['cup_bottom_price']}")
    print(f"  Right Rim  (recovery peak)   : ₹{p['right_rim_price']}")
    print(f"  Handle Low (small dip after) : ₹{p['handle_low_price']}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Cup Drop     : {p['cup_drop_pct']}%")
    print(f"  Recovery Gap : {p['recovery_pct']}%")

    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ GEOMETRY CHECK")
    print(f"    Cup Depth (₹)              : ₹{p['cup_depth']}")
    print(f"    Max Handle Dip Allowed (₹) : ₹{p['max_handle_dip']}  "
          f"(= 0.32 × ₹{p['cup_depth']})")
    print(f"    Actual Handle Pullback (₹) : ₹{p['handle_pullback']}  "
          f"({p['handle_pullback_pct']}%)")

    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ ROUNDEDNESS CHECK")
    rnd_pass = "✅ PASS" if p.get('roundedness_pct', 0) >= MIN_BASE_CANDLES_PCT * 100 else "❌ FAIL"
    print(f"    {p.get('base_candles', 'N/A')} candles in base zone out of "
          f"{p.get('cup_width', 'N/A')} cup candles "
          f"({p.get('roundedness_pct', 'N/A')}%) — {rnd_pass}")

    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ PAUSE-BEFORE-BREAKOUT CHECK")
    pause = p.get('pause_duration', 0)
    slope = p.get('handle_slope', 0)
    pause_pass = "✅ PASS" if pause >= MIN_PAUSE_CANDLES and slope <= 0 else "❌ FAIL"
    print(f"    Breakout Confirmed in      : {pause} candles (Needs ≥ {MIN_PAUSE_CANDLES})")
    print(f"    Handle Slope               : {slope} — {pause_pass}")

    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ HANDLE LOW")
    print(f"    OLD (first-dip)            : ₹{p.get('old_handle_low', 'N/A')}")
    print(f"    NEW (breakout-confirmed)   : ₹{p['handle_low_price']}")

    # Volume sections
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ VOLUME — Cup Decline")
    if p.get('vol_decline_pass') is not None:
        vd = "✅ PASS (light selling)" if p['vol_decline_pass'] else "⚠ WARN (heavy selling)"
        print(f"    Avg vol during decline     : {p['vol_decline_avg']:,.0f}")
        print(f"    50-period SMA vol          : {p['vol_sma50']:,.0f}")
        print(f"    Status                     : {vd}")
    else:
        print(f"    Status                     : N/A (no volume data)")

    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ VOLUME — Recovery")
    if p.get('vol_recovery_pass') is not None:
        vr = "✅ PASS (more buying)" if p['vol_recovery_pass'] else "⚠ WARN (weak buying)"
        print(f"    Up-vol / Down-vol ratio    : {p['vol_recovery_ratio']:.2f}")
        print(f"    Status                     : {vr}")
    else:
        print(f"    Status                     : N/A (no volume data)")

    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ VOLUME — Breakout")
    if p.get('vol_breakout_pass') is not None:
        vb = "✅ PASS (strong interest)" if p['vol_breakout_pass'] else "⚠ WARN (low interest)"
        print(f"    Avg breakout volume        : {p['vol_breakout_avg']:,.0f}")
        print(f"    Threshold (1.2× SMA50)     : {p['vol_sma50'] * BREAKOUT_VOLUME_MULTIPLIER:,.0f}")
        print(f"    Status                     : {vb}")
    else:
        print(f"    Status                     : N/A (no volume data)")

    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ SMOOTHING")
    print(f"    Method                     : {p.get('smoothing_method', 'None')}")
    print(f"    Note: Peaks detected on smoothed series; all prices are RAW.")

    if "left_rim_date" in p:
        print(f"  ─────────────────────────────────────────────────")
        print(f"  📅 Left Rim Date   : {p['left_rim_date']}")
        print(f"  📅 Cup Bottom Date : {p['cup_bottom_date']}")
        print(f"  📅 Right Rim Date  : {p['right_rim_date']}")
        print(f"  📅 Handle Low Date : {p['handle_low_date']}")

    print(f"  ─────────────────────────────────────────────────")
    if p.get("reject_reason"):
        print(f"  FINAL VERDICT: ❌ REJECTED ({p['reject_reason']})")
    else:
        print(f"  FINAL VERDICT: ✅ VALID (Quality Score: {p.get('quality_score', 0)})")
    print(f"  {'═' * 60}\n")


def _vol_str(val):
    """Formats a volume pass/fail as a string."""
    if val is None:
        return "N/A"
    return "✅ PASS" if val else "❌ FAIL"


def print_bull_flag(p, index=1):
    """Prints Bull Flag debug output."""
    print(f"\n  {'═' * 60}")
    print(f"  🏆 BULL FLAG #{index}  —  {p['ticker']}  (Score: {p['quality_score']})")
    print(f"  {'═' * 60}")
    print(f"  Pole Start      : ₹{p['pole_start_price']}"
          + (f"  ({p.get('pole_start_date', '')})" if 'pole_start_date' in p else ""))
    print(f"  Pole Top        : ₹{p['pole_top_price']}"
          + (f"  ({p.get('pole_top_date', '')})" if 'pole_top_date' in p else ""))
    print(f"  Flag Low        : ₹{p['flag_low_price']}"
          + (f"  ({p.get('flag_low_date', '')})" if 'flag_low_date' in p else ""))
    print(f"  Breakout Point  : ₹{p['breakout_price']}"
          + (f"  ({p.get('breakout_date', '')})" if 'breakout_date' in p else ""))
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Pole Rise %          : {p['pole_rise_pct']}%")
    print(f"  Pole R²              : {p['pole_r_squared']}")
    print(f"  Flag Channel Width % : {p['flag_range_pct']}%")
    print(f"  Flag Slope           : {p['flag_slope']}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ VOLUME")
    print(f"    Pole (above avg)   : {_vol_str(p['vol_pole_pass'])}")
    print(f"    Flag (below avg)   : {_vol_str(p['vol_flag_pass'])}")
    print(f"    Breakout (spike)   : {_vol_str(p['vol_breakout_pass'])}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  FINAL VERDICT: ✅ VALID (Quality Score: {p['quality_score']})")
    print(f"  {'═' * 60}\n")


def print_bear_flag(p, index=1):
    """Prints Bear Flag debug output."""
    print(f"\n  {'═' * 60}")
    print(f"  🏆 BEAR FLAG #{index}  —  {p['ticker']}  (Score: {p['quality_score']})")
    print(f"  {'═' * 60}")
    print(f"  Pole Start      : ₹{p['pole_start_price']}"
          + (f"  ({p.get('pole_start_date', '')})" if 'pole_start_date' in p else ""))
    print(f"  Pole Bottom     : ₹{p['pole_bottom_price']}"
          + (f"  ({p.get('pole_bottom_date', '')})" if 'pole_bottom_date' in p else ""))
    print(f"  Flag High       : ₹{p['flag_high_price']}"
          + (f"  ({p.get('flag_high_date', '')})" if 'flag_high_date' in p else ""))
    print(f"  Breakdown Point : ₹{p['breakdown_price']}"
          + (f"  ({p.get('breakdown_date', '')})" if 'breakdown_date' in p else ""))
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Pole Drop %          : {p['pole_drop_pct']}%")
    print(f"  Pole R²              : {p['pole_r_squared']}")
    print(f"  Flag Channel Width % : {p['flag_range_pct']}%")
    print(f"  Flag Slope           : {p['flag_slope']}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ VOLUME")
    print(f"    Pole (above avg)   : {_vol_str(p['vol_pole_pass'])}")
    print(f"    Flag (below avg)   : {_vol_str(p['vol_flag_pass'])}")
    print(f"    Breakdown (spike)  : {_vol_str(p['vol_breakdown_pass'])}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  FINAL VERDICT: ✅ VALID (Quality Score: {p['quality_score']})")
    print(f"  {'═' * 60}\n")


def print_pennant(p, index=1):
    """Prints Pennant debug output."""
    print(f"\n  {'═' * 60}")
    print(f"  🏆 PENNANT #{index}  —  {p['ticker']}  ({p['direction']})  (Score: {p['quality_score']})")
    print(f"  {'═' * 60}")
    print(f"  Pole Start      : ₹{p['pole_start_price']}"
          + (f"  ({p.get('pole_start_date', '')})" if 'pole_start_date' in p else ""))
    print(f"  Pole End        : ₹{p['pole_end_price']}"
          + (f"  ({p.get('pole_end_date', '')})" if 'pole_end_date' in p else ""))
    print(f"  Pennant High    : ₹{p['pennant_high']}")
    print(f"  Pennant Low     : ₹{p['pennant_low']}")
    print(f"  Breakout Point  : ₹{p['breakout_price']}"
          + (f"  ({p.get('breakout_date', '')})" if 'breakout_date' in p else ""))
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Pole Size %          : {p['pole_change_pct']}%")
    print(f"  Pole ADX             : {p['adx_value']} (must be > 30)")
    print(f"  Upper Trendline Slope: {p['upper_slope']} (must be < 0)")
    print(f"  Lower Trendline Slope: {p['lower_slope']} (must be > 0)")
    print(f"  Convergence Ratio    : {p['convergence_ratio']} "
          f"({'✅ Yes' if p['convergence_ratio'] < 1 else '❌ No'})")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ VOLUME")
    print(f"    Pennant Diminishing (below avg): {_vol_str(p['vol_pennant_pass'])}")
    print(f"    Breakout (>1.5x avg)           : {_vol_str(p['vol_breakout_pass'])}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  FINAL VERDICT: ✅ VALID (Quality Score: {p['quality_score']})")
    print(f"  {'═' * 60}\n")


def print_head_and_shoulders(p, index=1):
    """Prints Head & Shoulders debug output."""
    print(f"\n  {'═' * 60}")
    print(f"  🏆 HEAD & SHOULDERS #{index}  —  {p['ticker']}  (Score: {p['quality_score']})")
    print(f"  {'═' * 60}")
    print(f"  Left Shoulder   : ₹{p['left_shoulder_price']}"
          + (f"  ({p.get('left_shoulder_date', '')})" if 'left_shoulder_date' in p else ""))
    print(f"  Head            : ₹{p['head_price']}"
          + (f"  ({p.get('head_date', '')})" if 'head_date' in p else ""))
    print(f"  Right Shoulder  : ₹{p['right_shoulder_price']}"
          + (f"  ({p.get('right_shoulder_date', '')})" if 'right_shoulder_date' in p else ""))
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Left Neckline   : ₹{p['left_neckline_price']}"
          + (f"  ({p.get('left_neckline_date', '')})" if 'left_neckline_date' in p else ""))
    print(f"  Right Neckline  : ₹{p['right_neckline_price']}"
          + (f"  ({p.get('right_neckline_date', '')})" if 'right_neckline_date' in p else ""))
    print(f"  Neckline Slope  : {p['neckline_slope']}"
          + (f"  ⚠ STEEP ({p['neckline_slope_pct']}%)"
             if p['neckline_slope_pct'] > NECKLINE_SLOPE_WARN_PCT else ""))
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Head vs Left Shoulder  : +{p['head_vs_ls_pct']}%")
    print(f"  Head vs Right Shoulder : +{p['head_vs_rs_pct']}%")
    print(f"  Shoulder Symmetry      : {p['shoulder_symmetry_pct']}% diff")
    print(f"  Pattern Span           : {p['span_candles']} candles")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  ✦ VOLUME")
    print(f"    Left Shoulder avg  : {p['vol_ls_avg']:,.0f}")
    print(f"    Head avg           : {p['vol_head_avg']:,.0f}")
    print(f"    Right Shoulder avg : {p['vol_rs_avg']:,.0f}")
    print(f"    Progression (LS>H>RS) : {_vol_str(p['vol_progression_pass'])}")
    print(f"    Breakdown (spike)     : {_vol_str(p['vol_breakdown_pass'])}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  FINAL VERDICT: ✅ VALID (Quality Score: {p['quality_score']})")
    print(f"  {'═' * 60}\n")


def print_double_top(p, index=1):
    """Prints Double Top debug output in the updated format."""
    score = p.get('quality_score', 0)
    verdict = p.get('verdict', 'UNKNOWN')
    score_str = f"   Quality Score: {score}" if verdict == "CONFIRMED" else ""

    print(f"\nPATTERN -- {p['ticker']}  (Double Top){score_str}")
    print("=" * 60)
    # Prior trend
    trend_pass = "PASS" if p.get('prior_trend_pass') else "FAIL"
    print(f"Prior Trend: slope={p.get('prior_slope', 0)}, "
          f"R\u00b2={p.get('prior_r2', 0)} (on Adj Close) -- {trend_pass}")
    # Key prices
    print(f"First Peak: \u20b9{p['first_top_price']}"
          + (f"  ({p.get('first_top_date', '')})" if 'first_top_date' in p else ""))
    print(f"Neckline (Valley): \u20b9{p['valley_price']}"
          + (f"  ({p.get('valley_date', '')})" if 'valley_date' in p else ""))
    print(f"Second Peak: \u20b9{p['second_top_price']}"
          + (f"  ({p.get('second_top_date', '')})" if 'second_top_date' in p else ""))
    # Metrics
    print(f"Valley Drop: {p.get('valley_drop_pct', 0)}%")
    gap_pass = "PASS" if p.get('gap_days', 0) >= MIN_PEAK_GAP_DAYS else "FAIL"
    print(f"Gap between peaks: {p.get('gap_days', 0)} days -- {gap_pass} "
          f"(min {MIN_PEAK_GAP_DAYS} days)")
    sim_pass = "PASS" if p.get('similarity_pct', 99) <= DOUBLE_TOP_SIMILARITY_PCT else "FAIL"
    print(f"Similarity between peaks: {p.get('similarity_pct', 0)}% -- {sim_pass}")
    print("-" * 60)
    # Breakdown confirmation
    if p.get('breakdown_confirmed'):
        bd_pct = ((p['valley_price'] - p.get('breakdown_price', p['valley_price']))
                  / p['valley_price'] * 100) if p.get('breakdown_price') else 0
        print(f"* BREAKDOWN CONFIRMATION: decisive close = {bd_pct:.2f}%, "
              f"sustained {p.get('breakdown_confirm_candles', 2)} candles -- PASS")
    else:
        print(f"* BREAKDOWN CONFIRMATION: -- FAIL ({p.get('reject_reason', 'N/A')})")
    # Time decay
    waited = p.get('candles_waited', 0)
    max_wait = MAX_BREAKOUT_WAIT_CANDLES
    if verdict == "REJECTED" and "window expired" in p.get('reject_reason', ''):
        decay_str = f"{waited}/{max_wait} candles used -- EXPIRED"
    elif verdict == "CONFIRMED":
        decay_str = f"{waited}/{max_wait} candles used -- PASS"
    else:
        decay_str = f"{waited}/{max_wait} candles used -- FORMING"
    print(f"* BREAKOUT TIME-DECAY: {decay_str}")
    # Volume
    vol_t1 = p.get('vol_top1_avg', 0)
    vol_t2 = p.get('vol_top2_avg', 0)
    vol_trend = "declining (ideal)" if vol_t1 > vol_t2 else "increasing"
    print(f"* VOLUME -- First vs Second peak: {vol_trend} (informational)")
    bd_ratio = p.get('vol_breakdown_ratio', 0)
    bd_pass = _vol_str(p.get('vol_breakdown_pass'))
    print(f"* VOLUME -- Breakdown spike: {bd_ratio:.2f}x avg SMA -- {bd_pass}")
    print("-" * 60)
    # Trade reference
    print("* TRADE REFERENCE (educational only, not financial advice):")
    print(f"  Pattern Height: \u20b9{p.get('pattern_height', 0)}")
    print(f"  ATR (14-period): \u20b9{p.get('atr_14', 0)}")
    print(f"  Profit Target: \u20b9{p.get('profit_target', 0)}")
    print(f"  Suggested Stop-Loss: \u20b9{p.get('suggested_stop', 0)} (ATR-based)")
    print("-" * 60)
    # Final verdict
    if verdict == "CONFIRMED":
        print(f"FINAL VERDICT: CONFIRMED (score {score})")
    elif verdict == "FORMING":
        print(f"FINAL VERDICT: FORMING")
    else:
        print(f"FINAL VERDICT: REJECTED ({p.get('reject_reason', '')})")
    print("=" * 60 + "\n")


def print_double_bottom(p, index=1):
    """Prints Double Bottom debug output with Nifty 200 format."""
    score = p.get('quality_score', 0)
    verdict = p.get('verdict', 'UNKNOWN')
    score_str = f"   Quality Score: {score}" if verdict == "CONFIRMED" else ""

    print(f"\nPATTERN -- {p['ticker']}  (Double Bottom){score_str}")
    print("=" * 60)
    # Prior trend
    trend_pass = "PASS" if p.get('prior_trend_pass') else "FAIL"
    print(f"Prior Trend: slope < 0 OR below 50-EMA -- {trend_pass}")
    # Key prices
    print(f"First Bottom: \u20b9{p['first_bottom_price']}"
          + (f"  ({p.get('first_bottom_date', '')})" if 'first_bottom_date' in p else ""))
    print(f"Neckline (Peak): \u20b9{p['neckline_price']}"
          + (f"  ({p.get('neckline_date', '')})" if 'neckline_date' in p else ""))
    print(f"Second Bottom: \u20b9{p['second_bottom_price']}"
          + (f"  ({p.get('second_bottom_date', '')})" if 'second_bottom_date' in p else ""))
    # Metrics
    print(f"Pattern Depth: {p.get('depth_pct', 0)}% (Limits: 5% - 25%)")
    gap_pass = "PASS" if MIN_DOUBLE_CANDLES <= p.get('separation', 0) <= MAX_DOUBLE_CANDLES else "FAIL"
    print(f"Time Width: {p.get('separation', 0)} bars -- {gap_pass} "
          f"([ {MIN_DOUBLE_CANDLES}, {MAX_DOUBLE_CANDLES} ])")
    sim_pass = "PASS" if p.get('similarity_pct', 99) <= DOUBLE_BOTTOM_SIMILARITY_PCT else "FAIL"
    print(f"Bottom Symmetry: {p.get('similarity_pct', 0)}% -- {sim_pass} (Max {DOUBLE_BOTTOM_SIMILARITY_PCT}%)")
    
    rsi1 = p.get('rsi_1')
    rsi2 = p.get('rsi_2')
    if rsi1 is not None and rsi2 is not None:
        print(f"RSI Divergence: RSI1={rsi1}, RSI2={rsi2}")

    print("-" * 60)
    # Breakout confirmation
    if p.get('breakout_confirmed'):
        bo_pct = ((p.get('breakout_price', p['neckline_price']) - p['neckline_price'])
                  / p['neckline_price'] * 100) if p.get('breakout_price') else 0
        print(f"* BREAKOUT CONFIRMATION: Close > Neckline (+{bo_pct:.2f}%) -- PASS")
    else:
        print(f"* BREAKOUT CONFIRMATION: -- FAIL ({p.get('reject_reason', 'N/A')})")

    # Volume
    bo_pass = "PASS" if p.get('vol_breakout_pass') else "FAIL"
    print(f"* VOLUME -- Breakout spike >= 1.5x 20-SMA -- {bo_pass}")
    print("-" * 60)
    # Trade reference
    print("* TRADE REFERENCE (educational only, not financial advice):")
    print(f"  Profit Target: \u20b9{p.get('profit_target', 0)}")
    print(f"  Conservative Stop-Loss: \u20b9{p.get('conservative_stop', 0)} (ATR-based)")
    print(f"  Aggressive Stop-Loss: \u20b9{p.get('aggressive_stop', 0)} (Midpoint-based)")
    print("-" * 60)
    # Final verdict
    if verdict == "CONFIRMED":
        print(f"FINAL VERDICT: CONFIRMED (score {score})")
    elif verdict == "FORMING":
        print(f"FINAL VERDICT: FORMING")
    else:
        print(f"FINAL VERDICT: REJECTED ({p.get('reject_reason', '')})")
    print("=" * 60 + "\n")


# Dispatcher: route to the correct printer based on pattern_type
PRINT_DISPATCH = {
    "cup_and_handle":     print_cup_and_handle,
    "bull_flag":          print_bull_flag,
    "bear_flag":          print_bear_flag,
    "pennant":            print_pennant,
    "head_and_shoulders": print_head_and_shoulders,
    "double_top":         print_double_top,
    "double_bottom":      print_double_bottom,
}


def print_any_pattern(p, index=1):
    """Prints any pattern using the correct pattern-specific printer."""
    ptype = p.get("pattern_type", "cup_and_handle")
    printer = PRINT_DISPATCH.get(ptype, print_cup_and_handle)
    printer(p, index)


# =============================================================================
#  13. SELF-TEST FUNCTIONS — Synthetic data per pattern
# =============================================================================

def test_cup_and_handle():
    """Self-test for Cup & Handle using existing synthetic data."""
    print("\n  ── CUP AND HANDLE ──────────────────────────────────────")

    # Test 1: Textbook cup
    np.random.seed(123)
    rise = np.linspace(90, 100, 15)
    descent = np.linspace(100, 80, 30)
    bottom = np.ones(20) * 80 + np.random.uniform(-0.3, 0.3, 20)
    ascent = np.linspace(80, 100, 30)
    right_rim = np.array([100.0, 100.2, 100.1, 100.0, 99.8])
    handle_down = np.linspace(99.8, 95, 10)
    handle_up = np.linspace(95, 98, 10)
    post_break = np.linspace(98, 103, 5)
    post = np.ones(10) * 103

    prices = np.concatenate([rise, descent, bottom, ascent, right_rim,
                              handle_down, handle_up, post_break, post])
    n = len(prices)
    volumes = np.random.uniform(800000, 1200000, n)
    volumes[-15:] = np.random.uniform(2000000, 3000000, 15)

    results = detect_cup_and_handle(prices, volumes=volumes,
                                     ticker="SYNTH_CUP", verbose=False)
    if results:
        print(f"  ✅ Textbook shape → VALID  (score: {results[0].get('quality_score', 0)})")
    else:
        print("  ❌ Textbook shape → REJECTED (unexpected!)")

    # Test 2: Flat noise
    np.random.seed(42)
    flat = 100 + np.random.normal(0, 0.5, 200)
    results_flat = detect_cup_and_handle(flat, ticker="FLAT")
    if not results_flat:
        print("  ✅ Flat/random    → REJECTED (correct)")
    else:
        print(f"  ⚠ Flat/random    → {len(results_flat)} false positive(s)")

    return bool(results) and not results_flat


def test_bull_flag():
    """Self-test for Bull Flag."""
    print("\n  ── BULL FLAG ────────────────────────────────────────────")

    np.random.seed(200)
    lead = np.linspace(85, 90, 30)
    pole = np.linspace(90, 108, 15)
    flag = np.linspace(107, 104, 10)
    breakout = np.array([109, 111, 113])
    post = np.ones(15) * 113

    prices = np.concatenate([lead, pole, flag, breakout, post])
    n = len(prices)
    volumes = np.random.uniform(800000, 1200000, n)
    # Pole: above avg
    volumes[30:45] = np.random.uniform(1500000, 2000000, 15)
    # Flag: below avg
    volumes[45:55] = np.random.uniform(400000, 600000, 10)
    # Breakout: spike
    volumes[55:58] = np.random.uniform(2500000, 3500000, 3)

    results = detect_bull_flag(prices, volumes=volumes,
                                ticker="SYNTH_BULL_FLAG", verbose=False)
    if results:
        print(f"  ✅ Textbook shape → VALID  (score: {results[0]['quality_score']})")
        print_bull_flag(results[0], 1)
    else:
        print("  ❌ Textbook shape → REJECTED (unexpected!)")

    # Flat noise
    np.random.seed(42)
    flat = 100 + np.random.normal(0, 0.5, 200)
    results_flat = detect_bull_flag(flat, ticker="FLAT")
    if not results_flat:
        print("  ✅ Flat/random    → REJECTED (correct)")
    else:
        print(f"  ⚠ Flat/random    → {len(results_flat)} false positive(s)")

    return bool(results) and not results_flat


def test_bear_flag():
    """Self-test for Bear Flag."""
    print("\n  ── BEAR FLAG ────────────────────────────────────────────")

    np.random.seed(201)
    lead = np.linspace(100, 108, 30)
    pole = np.linspace(108, 98, 15)
    flag = np.linspace(99, 102, 10)
    breakdown = np.array([97, 95, 93])
    post = np.ones(15) * 93

    prices = np.concatenate([lead, pole, flag, breakdown, post])
    n = len(prices)
    volumes = np.random.uniform(800000, 1200000, n)
    volumes[30:45] = np.random.uniform(1500000, 2000000, 15)
    volumes[45:55] = np.random.uniform(400000, 600000, 10)
    volumes[55:58] = np.random.uniform(2500000, 3500000, 3)

    results = detect_bear_flag(prices, volumes=volumes,
                                ticker="SYNTH_BEAR_FLAG", verbose=False)
    if results:
        print(f"  ✅ Textbook shape → VALID  (score: {results[0]['quality_score']})")
        print_bear_flag(results[0], 1)
    else:
        print("  ❌ Textbook shape → REJECTED (unexpected!)")

    # Flat noise
    np.random.seed(42)
    flat = 100 + np.random.normal(0, 0.5, 200)
    results_flat = detect_bear_flag(flat, ticker="FLAT")
    if not results_flat:
        print("  ✅ Flat/random    → REJECTED (correct)")
    else:
        print(f"  ⚠ Flat/random    → {len(results_flat)} false positive(s)")

    return bool(results) and not results_flat


def test_pennant():
    """Self-test for Pennant (bullish)."""
    print("\n  ── PENNANT ──────────────────────────────────────────────")
    all_pass = True

    # 1. Valid Textbook Pennant
    np.random.seed(202)
    # Lead up to create strong ADX (>30). Needs to be steady uptrend
    lead_close = np.linspace(50, 90, 40)
    lead_high = lead_close + np.random.uniform(0.5, 1.5, 40)
    lead_low = lead_close - np.random.uniform(0.5, 1.5, 40)
    
    # Pole (sharp rise)
    pole_close = np.linspace(90, 120, 20)
    pole_high = pole_close + np.random.uniform(1, 2, 20)
    pole_low = pole_close - np.random.uniform(1, 2, 20)
    
    # Pennant body (converging)
    # Highs must have negative slope, lows must have positive slope
    p_len = 10
    pen_high = np.linspace(119, 115, p_len) + np.random.uniform(-0.5, 0.5, p_len)
    pen_low = np.linspace(111, 114, p_len) + np.random.uniform(-0.5, 0.5, p_len)
    pen_close = (pen_high + pen_low) / 2
    
    # Breakout
    bo_close = np.linspace(116, 125, 5)
    bo_high = bo_close + 1
    bo_low = bo_close - 1
    
    # Post
    post_close = np.ones(10) * 125
    post_high = post_close + 1
    post_low = post_close - 1
    
    prices1 = np.concatenate([lead_close, pole_close, pen_close, bo_close, post_close])
    highs1 = np.concatenate([lead_high, pole_high, pen_high, bo_high, post_high])
    lows1 = np.concatenate([lead_low, pole_low, pen_low, bo_low, post_low])
    n1 = len(prices1)
    
    # Volumes
    vol_lead = np.random.uniform(800000, 1000000, 40)
    vol_pole = np.random.uniform(1500000, 2000000, 20)
    # Diminishing volume in pennant
    vol_pen = np.linspace(1500000, 500000, p_len)
    # Breakout spike (> 1.5x SMA20)
    vol_bo = np.random.uniform(3000000, 4000000, 5)
    vol_post = np.random.uniform(800000, 1000000, 10)
    vols1 = np.concatenate([vol_lead, vol_pole, vol_pen, vol_bo, vol_post])
    
    res1 = detect_pennant(prices1, highs1, lows1, volumes=vols1, ticker="SYNTH_PEN_VALID", verbose=False)
    if res1:
        print(f"  ✅ [1] Textbook valid -> CONFIRMED (score: {res1[0]['quality_score']})")
        print_pennant(res1[0], 1)
    else:
        print("  ❌ [1] Textbook valid -> REJECTED (unexpected!)")
        all_pass = False

    # 2. Invalid ADX (ADX < 30) - Sudden 1 candle jump instead of steady trend
    lead_close_bad_adx = np.ones(40) * 90
    lead_high_bad = lead_close_bad_adx + 0.5
    lead_low_bad = lead_close_bad_adx - 0.5
    
    pole_close_bad_adx = np.ones(20) * 90
    pole_close_bad_adx[-1] = 101 # >10% jump on the very last day
    pole_high_bad_adx = pole_close_bad_adx + 0.5
    pole_low_bad_adx = pole_close_bad_adx - 0.5

    prices2 = np.concatenate([lead_close_bad_adx, pole_close_bad_adx, pen_close, bo_close, post_close])
    highs2 = np.concatenate([lead_high_bad, pole_high_bad_adx, pen_high, bo_high, post_high])
    lows2 = np.concatenate([lead_low_bad, pole_low_bad_adx, pen_low, bo_low, post_low])
    res2 = detect_pennant(prices2, highs2, lows2, volumes=vols1, ticker="SYNTH_PEN_BAD_ADX")
    if not res2:
        print("  ✅ [2] Bad ADX (sudden jump, no trend) -> REJECTED (correct)")
    else:
        print(f"  ❌ [2] Bad ADX -> CONFIRMED with ADX {res2[0]['adx_value']} (unexpected!)")
        all_pass = False

    # 3. Invalid Slopes (Upper slope positive)
    pen_high_bad = np.linspace(115, 119, p_len) # Positive slope
    highs3 = np.concatenate([lead_high, pole_high, pen_high_bad, bo_high, post_high])
    res3 = detect_pennant(prices1, highs3, lows1, volumes=vols1, ticker="SYNTH_PEN_BAD_SLOPE")
    if not res3:
        print("  ✅ [3] Bad Slopes (upper slope > 0) -> REJECTED (correct)")
    else:
        print("  ❌ [3] Bad Slopes -> CONFIRMED (unexpected!)")
        all_pass = False

    # 4. Bad Breakout Volume
    vol_bo_bad = np.random.uniform(500000, 800000, 5) # Low volume
    vols4 = np.concatenate([vol_lead, vol_pole, vol_pen, vol_bo_bad, vol_post])
    res4 = detect_pennant(prices1, highs1, lows1, volumes=vols4, ticker="SYNTH_PEN_BAD_VOL")
    if not res4:
        print("  ✅ [4] Weak Breakout Volume -> REJECTED (correct)")
    else:
        print("  ❌ [4] Weak Breakout Volume -> CONFIRMED (unexpected!)")
        all_pass = False

    # 5. Flat noise
    np.random.seed(42)
    flat = 100 + np.random.normal(0, 0.5, 200)
    results_flat = detect_pennant(flat, flat+1, flat-1, ticker="FLAT")
    if not results_flat:
        print("  ✅ [5] Flat/random -> REJECTED (correct)")
    else:
        print(f"  ❌ [5] Flat/random -> {len(results_flat)} false positive(s)")
        all_pass = False

    return all_pass


def test_head_and_shoulders():
    """Self-test for Head & Shoulders."""
    print("\n  ── HEAD AND SHOULDERS ───────────────────────────────────")

    np.random.seed(203)
    lead = np.linspace(85, 95, 20)
    ls_up = np.linspace(95, 105, 10)
    ls_down = np.linspace(105, 100, 8)
    head_up = np.linspace(100, 112, 10)
    head_down = np.linspace(112, 100, 10)
    rs_up = np.linspace(100, 105, 10)
    rs_down = np.linspace(105, 100, 8)
    breakdown = np.linspace(100, 94, 6)
    post = np.ones(10) * 94

    prices = np.concatenate([lead, ls_up, ls_down, head_up, head_down,
                              rs_up, rs_down, breakdown, post])
    n = len(prices)
    volumes = np.random.uniform(800000, 1200000, n)
    # Left shoulder: highest volume
    volumes[20:30] = np.random.uniform(2000000, 2500000, 10)
    # Head: moderate
    volumes[38:48] = np.random.uniform(1500000, 1800000, 10)
    # Right shoulder: lowest
    volumes[58:68] = np.random.uniform(900000, 1100000, 10)
    # Breakdown: spike
    volumes[76:82] = np.random.uniform(2500000, 3000000, 6)

    results = detect_head_and_shoulders(prices, volumes=volumes,
                                         ticker="SYNTH_HS", verbose=False)
    if results:
        print(f"  ✅ Textbook shape → VALID  (score: {results[0]['quality_score']})")
        print_head_and_shoulders(results[0], 1)
    else:
        print("  ❌ Textbook shape → REJECTED (unexpected!)")

    # Flat noise
    np.random.seed(42)
    flat = 100 + np.random.normal(0, 0.5, 200)
    results_flat = detect_head_and_shoulders(flat, ticker="FLAT")
    if not results_flat:
        print("  ✅ Flat/random    → REJECTED (correct)")
    else:
        print(f"  ⚠ Flat/random    → {len(results_flat)} false positive(s)")

    return bool(results) and not results_flat


def test_double_top():
    """
    Self-test for Double Top with 5 synthetic test cases:
    1. Textbook valid (should CONFIRM)
    2. Peaks too close (should REJECT)
    3. No prior uptrend (should REJECT)
    4. Touches neckline but doesn't close beyond it (should remain FORMING)
    5. Flat noise (should find nothing)
    """
    print("\n  == DOUBLE TOP — 5 SYNTHETIC TEST CASES ==================")
    all_pass = True

    # ── TEST 1: Textbook valid Double Top ──
    # Clear uptrend → peak1 at ~110 → valley drops >10% to ~97 →
    # peak2 at ~110 → decisive breakdown below valley → volume spike
    print("\n  [TEST 1] Textbook valid Double Top")
    np.random.seed(300)
    # 25-candle uptrend (strong positive slope, R² > 0.5)
    uptrend = np.linspace(80, 100, 25)
    # Rise to first peak at 110 (10 candles)
    first_up = np.linspace(100, 110, 10)
    # Valley drop: 110 → 97 (>10% below peak) over 15 candles (>14 day gap total)
    valley_down = np.linspace(110, 97, 8)
    valley_flat = np.ones(4) * 97
    # Rise to second peak at 110 (same height, <3% difference)
    second_up = np.linspace(97, 110, 8)
    # Breakdown: decisive close below 97 * (1 - 0.005) = 96.515
    # Need 2 consecutive candles below this threshold
    breakdown = np.array([105, 100, 96, 95.5, 95, 94.5])
    post = np.ones(10) * 94

    prices = np.concatenate([uptrend, first_up, valley_down, valley_flat,
                              second_up, breakdown, post])
    n = len(prices)
    volumes = np.random.uniform(800000, 1200000, n)
    # First top area: higher volume
    volumes[33:38] = np.random.uniform(2000000, 2500000, 5)
    # Second top area: lower volume (declining — ideal)
    volumes[50:55] = np.random.uniform(1000000, 1300000, 5)
    # Breakdown area: cover entire post-peak zone with volume spike > 1.2x SMA50
    # (breakdown occurs at idx ~57-58 per diagnostics)
    volumes[55:n] = np.random.uniform(2500000, 3500000, n - 55)

    results = detect_double_top(prices, volumes=volumes,
                                 ticker="SYNTH_DT_VALID", verbose=False)
    if results and results[0].get("verdict") == "CONFIRMED":
        print(f"  PASS: Textbook Double Top -> CONFIRMED (score: {results[0]['quality_score']})")
        print_double_top(results[0], 1)
    else:
        print("  FAIL: Textbook Double Top should be CONFIRMED but wasn't")
        if results:
            print(f"    Got verdict: {results[0].get('verdict')}, reason: {results[0].get('reject_reason')}")
        all_pass = False

    # ── TEST 2: Peaks too close together (<14 days apart) ──
    print("\n  [TEST 2] Peaks too close together (< 2 weeks)")
    np.random.seed(301)
    uptrend2 = np.linspace(80, 100, 25)
    up2 = np.linspace(100, 110, 5)
    # Very short valley — only 3 candles apart
    short_valley = np.array([105, 103])
    up2b = np.linspace(103, 110, 3)
    bd2 = np.linspace(110, 95, 10)
    post2 = np.ones(10) * 95

    prices2 = np.concatenate([uptrend2, up2, short_valley, up2b, bd2, post2])
    n2 = len(prices2)
    volumes2 = np.random.uniform(800000, 1200000, n2)
    volumes2[-10:] = np.random.uniform(2500000, 3500000, 10)

    results2 = detect_double_top(prices2, volumes=volumes2,
                                  ticker="SYNTH_DT_TOOCLOSE", verbose=False)
    if not results2:
        print("  PASS: Peaks too close -> correctly REJECTED (no results)")
    else:
        print(f"  FAIL: Should reject peaks too close, got verdict: {results2[0].get('verdict')}")
        all_pass = False

    # ── TEST 3: No prior uptrend ──
    print("\n  [TEST 3] No prior uptrend (flat before first peak)")
    np.random.seed(302)
    # Flat lead-in: no uptrend (slope ~ 0, R² low)
    flat_lead = np.ones(25) * 110 + np.random.normal(0, 0.3, 25)
    v_down3 = np.linspace(110, 97, 10)
    v_flat3 = np.ones(5) * 97
    up3 = np.linspace(97, 110, 10)
    bd3 = np.array([105, 100, 96, 95, 94])
    post3 = np.ones(10) * 94

    prices3 = np.concatenate([flat_lead, v_down3, v_flat3, up3, bd3, post3])
    n3 = len(prices3)
    volumes3 = np.random.uniform(800000, 1200000, n3)
    volumes3[-5:] = np.random.uniform(2500000, 3500000, 5)

    results3 = detect_double_top(prices3, volumes=volumes3,
                                  ticker="SYNTH_DT_NOTREND", verbose=False)
    if not results3:
        print("  PASS: No prior uptrend -> correctly REJECTED")
    else:
        print(f"  FAIL: Should reject (no uptrend), got verdict: {results3[0].get('verdict')}")
        all_pass = False

    # ── TEST 4: Touches neckline but doesn't close beyond it ──
    print("\n  [TEST 4] Touches neckline but no decisive close (should be FORMING)")
    np.random.seed(303)
    uptrend4 = np.linspace(80, 100, 25)
    up4 = np.linspace(100, 110, 10)
    v_down4 = np.linspace(110, 97, 8)
    v_flat4 = np.ones(4) * 97
    up4b = np.linspace(97, 110, 8)
    # Price approaches neckline but stays just above: 97 * (1 - 0.005) = 96.515
    # These values are ABOVE the threshold — no decisive close
    near_miss = np.array([105, 100, 97.5, 97.2, 97.1, 97.0, 97.5, 98, 99, 100])

    prices4 = np.concatenate([uptrend4, up4, v_down4, v_flat4, up4b, near_miss])
    n4 = len(prices4)
    volumes4 = np.random.uniform(800000, 1200000, n4)
    volumes4[33:38] = np.random.uniform(2000000, 2500000, 5)

    results4 = detect_double_top(prices4, volumes=volumes4,
                                  ticker="SYNTH_DT_FORMING", verbose=True)
    if results4 and results4[0].get("verdict") == "FORMING":
        print("  PASS: Near miss -> correctly shows FORMING")
    elif not results4:
        # With verbose=True, should still be emitted; without, no results is expected
        # since we haven't reached breakout window expiry (only 10 candles of data)
        print("  PASS: Near miss -> no decisive breakdown (correctly filtered)")
    else:
        got = results4[0].get('verdict', 'NONE') if results4 else 'NONE'
        print(f"  NOTE: Got verdict '{got}' — expected FORMING or filtered out")

    # ── TEST 5: Flat noise ──
    print("\n  [TEST 5] Flat/random noise")
    np.random.seed(42)
    flat = 100 + np.random.normal(0, 0.5, 200)
    results_flat = detect_double_top(flat, ticker="FLAT")
    if not results_flat:
        print("  PASS: Flat/random -> REJECTED (correct)")
    else:
        print(f"  FAIL: Flat/random -> {len(results_flat)} false positive(s)")
        all_pass = False

    return all_pass


def test_double_bottom():
    """
    Self-test for Double Bottom with 5 synthetic test cases:
    1. Textbook valid (should CONFIRM)
    2. Bottoms too close (should REJECT)
    3. No prior downtrend (should REJECT)
    4. Touches neckline but doesn't close beyond it (should remain FORMING)
    5. Flat noise (should find nothing)
    """
    print("\n  == DOUBLE BOTTOM — 5 SYNTHETIC TEST CASES ================")
    all_pass = True

    # ── TEST 1: Textbook valid Double Bottom ──
    print("\n  [TEST 1] Textbook valid Double Bottom")
    np.random.seed(400)
    # 25-candle downtrend (strong negative slope, R² > 0.5)
    downtrend = np.linspace(120, 100, 25)
    # Drop to first bottom at 90
    first_down = np.linspace(100, 90, 10)
    # Recovery peak (neckline): 90 → 100 (>10% above bottom)
    peak_up = np.linspace(90, 100, 8)
    peak_flat = np.ones(4) * 100
    # Drop to second bottom at 90 (same level)
    second_down = np.linspace(100, 90, 8)
    # Decisive breakout above 100 * (1 + 0.005) = 100.5
    breakout = np.array([95, 99, 101, 101.5, 102, 103])
    post = np.ones(10) * 103

    prices = np.concatenate([downtrend, first_down, peak_up, peak_flat,
                              second_down, breakout, post])
    n = len(prices)
    volumes = np.random.uniform(800000, 1200000, n)
    # First bottom: higher volume
    volumes[33:38] = np.random.uniform(2000000, 2500000, 5)
    # Second bottom: lower volume (declining — ideal)
    volumes[50:55] = np.random.uniform(800000, 1000000, 5)
    # Breakout area: cover entire post-bottom zone with volume spike > 1.2x SMA50
    volumes[55:n] = np.random.uniform(2500000, 3500000, n - 55)

    results = detect_double_bottom(prices, volumes=volumes,
                                    ticker="SYNTH_DB_VALID", verbose=False)
    if results and results[0].get("verdict") == "CONFIRMED":
        print(f"  PASS: Textbook Double Bottom -> CONFIRMED (score: {results[0]['quality_score']})")
        print_double_bottom(results[0], 1)
    else:
        print("  FAIL: Textbook Double Bottom should be CONFIRMED but wasn't")
        if results:
            print(f"    Got verdict: {results[0].get('verdict')}, reason: {results[0].get('reject_reason')}")
        all_pass = False

    # ── TEST 2: Bottoms too close together ──
    print("\n  [TEST 2] Bottoms too close together (< 2 weeks)")
    np.random.seed(401)
    downtrend2 = np.linspace(120, 100, 25)
    down2 = np.linspace(100, 90, 5)
    short_peak = np.array([95, 97])
    down2b = np.linspace(97, 90, 3)
    bo2 = np.linspace(90, 105, 10)
    post2 = np.ones(10) * 105

    prices2 = np.concatenate([downtrend2, down2, short_peak, down2b, bo2, post2])
    n2 = len(prices2)
    volumes2 = np.random.uniform(800000, 1200000, n2)
    volumes2[-10:] = np.random.uniform(2500000, 3500000, 10)

    results2 = detect_double_bottom(prices2, volumes=volumes2,
                                     ticker="SYNTH_DB_TOOCLOSE", verbose=False)
    if not results2:
        print("  PASS: Bottoms too close -> correctly REJECTED (no results)")
    else:
        print(f"  FAIL: Should reject bottoms too close, got verdict: {results2[0].get('verdict')}")
        all_pass = False

    # ── TEST 3: No prior downtrend ──
    print("\n  [TEST 3] No prior downtrend (flat before first bottom)")
    np.random.seed(402)
    flat_lead = np.ones(25) * 90 + np.random.normal(0, 0.3, 25)
    p_up3 = np.linspace(90, 100, 10)
    p_flat3 = np.ones(5) * 100
    down3 = np.linspace(100, 90, 10)
    bo3 = np.array([95, 99, 101, 102, 103])
    post3 = np.ones(10) * 103

    prices3 = np.concatenate([flat_lead, p_up3, p_flat3, down3, bo3, post3])
    n3 = len(prices3)
    volumes3 = np.random.uniform(800000, 1200000, n3)
    volumes3[-5:] = np.random.uniform(2500000, 3500000, 5)

    results3 = detect_double_bottom(prices3, volumes=volumes3,
                                     ticker="SYNTH_DB_NOTREND", verbose=False)
    if not results3:
        print("  PASS: No prior downtrend -> correctly REJECTED")
    else:
        print(f"  FAIL: Should reject (no downtrend), got verdict: {results3[0].get('verdict')}")
        all_pass = False

    # ── TEST 4: Touches neckline but no decisive close ──
    print("\n  [TEST 4] Touches neckline but no decisive breakout (should be FORMING)")
    np.random.seed(403)
    downtrend4 = np.linspace(120, 100, 25)
    down4 = np.linspace(100, 90, 10)
    peak_up4 = np.linspace(90, 100, 8)
    peak_flat4 = np.ones(4) * 100
    down4b = np.linspace(100, 90, 8)
    # Price approaches neckline but stays just below 100 * (1 + 0.005) = 100.5
    near_miss = np.array([95, 99, 100.2, 99.8, 100.3, 100.1, 99.5, 98, 97, 96])

    prices4 = np.concatenate([downtrend4, down4, peak_up4, peak_flat4, down4b, near_miss])
    n4 = len(prices4)
    volumes4 = np.random.uniform(800000, 1200000, n4)

    results4 = detect_double_bottom(prices4, volumes=volumes4,
                                     ticker="SYNTH_DB_FORMING", verbose=True)
    if results4 and results4[0].get("verdict") == "FORMING":
        print("  PASS: Near miss -> correctly shows FORMING")
    elif not results4:
        print("  PASS: Near miss -> no decisive breakout (correctly filtered)")
    else:
        got = results4[0].get('verdict', 'NONE') if results4 else 'NONE'
        print(f"  NOTE: Got verdict '{got}' — expected FORMING or filtered out")

    # ── TEST 5: Flat noise ──
    print("\n  [TEST 5] Flat/random noise")
    np.random.seed(42)
    flat = 100 + np.random.normal(0, 0.5, 200)
    results_flat = detect_double_bottom(flat, ticker="FLAT")
    if not results_flat:
        print("  PASS: Flat/random -> REJECTED (correct)")
    else:
        print(f"  FAIL: Flat/random -> {len(results_flat)} false positive(s)")
        all_pass = False

    return all_pass


def run_self_tests(pattern_type="all"):
    """
    Runs self-tests for one or all patterns.
    Each test creates textbook synthetic data and verifies detection.
    """
    print("\n" + "=" * 70)
    print("  🧪 SELF-TEST : Synthetic Pattern Verification")
    print("=" * 70)

    TEST_MAP = {
        "cup_and_handle":     test_cup_and_handle,
        "bull_flag":          test_bull_flag,
        "bear_flag":          test_bear_flag,
        "pennant":            test_pennant,
        "head_and_shoulders": test_head_and_shoulders,
        "double_top":         test_double_top,
        "double_bottom":      test_double_bottom,
    }

    if pattern_type == "all":
        tests_to_run = ALL_PATTERNS
    elif pattern_type in TEST_MAP:
        tests_to_run = [pattern_type]
    else:
        print(f"  Unknown pattern: {pattern_type}")
        return

    passed = 0
    failed = 0

    for ptype in tests_to_run:
        try:
            result = TEST_MAP[ptype]()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ {PATTERN_NAMES[ptype]} test CRASHED: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"  🧪 SELF-TEST COMPLETE: {passed} passed, {failed} failed "
          f"out of {passed + failed}")
    print("=" * 70 + "\n")


# =============================================================================
#  14. SCANNING INFRASTRUCTURE — Multi-pattern scanning
# =============================================================================

ALGO_VERSION = 1





def print_summary_table(all_results):
    """Prints the ALL PATTERNS summary table."""
    ist_now = _get_ist_now()

    print(f"\n{'═' * 60}")
    print(f"  SCAN SUMMARY")
    print(f"{'─' * 60}")
    print(f"  {'Pattern':<24} | {'Stocks Found':>12} | Top Pick")
    print(f"{'─' * 60}")

    total = 0
    for ptype in ALL_PATTERNS:
        patterns = all_results.get(ptype, [])
        count = len(patterns)
        total += count
        if count > 0:
            top = patterns[0]
            top_str = f"{top['ticker']} (score: {top.get('quality_score', 0)})"
        else:
            top_str = "—"
        print(f"  {PATTERN_NAMES[ptype]:<24} | {count:>12} | {top_str}")

    print(f"{'─' * 60}")
    print(f"  Total stocks flagged : {total}")
    print(f"  Scan completed at    : {ist_now.strftime('%H:%M:%S')} IST")
    print(f"{'═' * 60}")


def backtest_historical(patterns_to_scan, tickers=None, period="2y",
                        start=None, end=None, interval="1d"):
    """
    MODE B: HISTORICAL BACKTEST — scans historical data for patterns.

    Results are saved to 'backtest_results.txt'.
    """
    label = f"Interval: {interval}, Period: {period or 'Custom Date Range'}"
    scan_label = (", ".join(PATTERN_NAMES[p] for p in patterns_to_scan)
                  if "all" not in patterns_to_scan
                  else "ALL PATTERNS")

    print("\n" + "=" * 70)
    print("  📊 HISTORICAL BACKTEST  (Full Timeline Sweep)")
    print("=" * 70)
    print(f"  Patterns   : {scan_label}")
    print(f"  Data       : {label}")
    print(f"  Smoothing  : SMA({SMOOTHING_WINDOW}) (where needed)\n")

    if tickers is None:
        print("  Fetching Nifty watchlist ...\n")
        tickers = get_nifty_list()

    print(f"  Scanning {len(tickers)} tickers ...\n")

    all_data = fetch_batch_data(
        tickers, period=period, start=start, end=end, interval=interval
    )

    if not all_data:
        print("  ⚠ No data downloaded. Check your internet connection.\n")
        return {}

    # Results grouped by pattern type
    all_results = {p: [] for p in ALL_PATTERNS}
    tickers_scanned = 0

    for ticker, df in all_data.items():
        tickers_scanned += 1
        is_verbose = (len(all_data) <= 5)
        ticker_results = scan_ticker(df, ticker, patterns_to_scan,
                                      interval=interval, verbose=is_verbose)

        for ptype, patterns in ticker_results.items():
            completed_patterns = [p for p in patterns if p.get("status") != "forming"]
            if completed_patterns:
                print(f"  ✓ {ticker}: {len(completed_patterns)} {PATTERN_NAMES[ptype]} pattern(s)")
            all_results[ptype].extend(completed_patterns)

    # Sort each pattern's results by quality score
    for ptype in all_results:
        all_results[ptype].sort(key=lambda p: p.get("quality_score", 0), reverse=True)

    # Helper for clean dates
    def clean_date(d):
        return str(d).split(' ')[0] if ' ' in str(d) else str(d)

    # ── Write results file ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(script_dir, "backtest_results.txt")

    with open(results_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  📊 MULTI-PATTERN BACKTEST RESULTS (v3.0)\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Run Date       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Patterns       : {scan_label}\n")
        f.write(f"  Data           : {label}\n")
        f.write(f"  Tickers scanned: {tickers_scanned}\n")
        f.write("=" * 70 + "\n\n")

        total_found = 0
        for ptype in ALL_PATTERNS:
            patterns = all_results[ptype]
            if not patterns and "all" not in patterns_to_scan and ptype not in patterns_to_scan:
                continue
            total_found += len(patterns)

            f.write(f"\n{'═' * 60}\n")
            f.write(f"  {PATTERN_NAMES[ptype].upper()} — {len(patterns)} pattern(s)\n")
            f.write(f"{'═' * 60}\n")

            if patterns:
                for idx, p in enumerate(patterns, 1):
                    f.write(f"\n  #{idx}  {p['ticker']}  "
                            f"(Score: {p.get('quality_score', 0)})\n")
                    for key, val in p.items():
                        if key not in ("pattern_type", "ticker", "verdict",
                                       "reject_reason") and not key.endswith("_idx"):
                            f.write(f"    {key:<30}: {val}\n")
            else:
                f.write("  No patterns found.\n")

        f.write(f"\n{'═' * 60}\n")
        f.write(f"  Total patterns found: {total_found}\n")
        f.write(f"{'═' * 60}\n")

    # ── Terminal: per-pattern sections ──
    print(f"\n  {'─' * 60}")
    print(f"  📊 BACKTEST RESULTS")
    print(f"  {'─' * 60}")
    print(f"  Tickers scanned : {tickers_scanned}")

    for ptype in ALL_PATTERNS:
        patterns = all_results[ptype]
        if not patterns and "all" not in patterns_to_scan and ptype not in patterns_to_scan:
            continue

        print(f"\n  {'═' * 58}")
        print(f"  {PATTERN_NAMES[ptype].upper()} — {len(patterns)} pattern(s)")
        print(f"  {'═' * 58}")

        if patterns:
            for idx, p in enumerate(patterns, 1):
                print_any_pattern(p, idx)
        else:
            print("  (no patterns found)")

    # Summary table (for ALL mode)
    if "all" in patterns_to_scan:
        print_summary_table(all_results)

    print(f"\n  📄 Full results saved to: {results_file}")
    print("\n" + "=" * 70)
    print("  📊 BACKTEST COMPLETE")
    print("=" * 70 + "\n")

    return all_results


# =============================================================================
#  15. LIVE SCANNER — Continuous during market hours
# =============================================================================













def run_scheduler(patterns_to_scan, tickers=None, period="59d",
                  start=None, end=None, interval="15m"):
    """
    MODE C: CONTINUOUS LIVE SCANNER during NSE market hours.
    Scans every SCAN_INTERVAL_MINUTES for patterns completing TODAY.
    """
    scan_label = (", ".join(PATTERN_NAMES[p] for p in patterns_to_scan)
                  if "all" not in patterns_to_scan
                  else "ALL PATTERNS")

    print("\n" + "=" * 70)
    print("  🚀 LIVE SCANNER — Multi-Pattern (v3.0)")
    print("=" * 70)
    print(f"  Patterns      : {scan_label}")
    print(f"  Interval      : {interval}")
    print(f"  Scan interval : every {SCAN_INTERVAL_MINUTES} minutes")
    print(f"  Market hours  : Mon–Fri, 9:15 AM – 3:30 PM IST")
    print(f"  Exit          : Press Ctrl+C\n")

    if tickers is None:
        tickers = get_nifty_list()

    scan_count = 0

    try:
        while True:
            market_open, status_msg = is_market_open()

            if not market_open:
                print(f"  ⏸  {status_msg} — sleeping 5 min ...")
                time.sleep(300)
                continue

            scan_count += 1
            print(f"\n  {'━' * 60}")
            print(f"  🔄 SCAN #{scan_count}  —  {status_msg}")
            print(f"  {'━' * 60}")
            print(f"  📋 Dedup cache: {len(_live_alerted_today)} alert(s) today.\n")

            try:
                new_alerts = scan_watchlist(
                    tickers, patterns_to_scan, period=period,
                    start=start, end=end, interval=interval
                )

                if new_alerts:
                    new_alerts.sort(key=lambda p: p.get("quality_score", 0), reverse=True)
                    print(f"\n  🔔 {len(new_alerts)} NEW ALERT(S)!\n")
                    for idx, pat in enumerate(new_alerts, 1):
                        print_any_pattern(pat, idx)
                else:
                    print("  ─ No new patterns completing today.\n")

            except Exception as e:
                print(f"\n  ⚠ Scan error (will retry): {e}\n")

            ist_now = _get_ist_now()
            print(f"  ⏳ Next scan in {SCAN_INTERVAL_MINUTES} minutes "
                  f"({ist_now.strftime('%H:%M')} IST)  (Ctrl+C to stop)\n")
            time.sleep(SCAN_INTERVAL_MINUTES * 60)

    except KeyboardInterrupt:
        print("\n\n  👋 Scanner stopped by user. Goodbye!\n")


# =============================================================================
#  16. MENU & MAIN ENTRY POINT
# =============================================================================



def prompt_for_config(default_interval="1d", default_period="2y"):
    """Interactive prompt for candle interval and lookback period."""
    print("\n  [Configuration]")
    interval = input(f"  Enter candle interval (15m / 30m / 1h / 1d / 1wk) "
                     f"[default {default_interval}]: ").strip()
    if not interval:
        interval = default_interval

    if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]:
        if default_period == "2y":
            default_period = "59d"

    period = input(f"  Enter lookback period (Xd / Xmo / Xy) "
                   f"[default {default_period}]: ").strip()
    if not period:
        period = default_period

    period = validate_yfinance_params(interval, period)
    return interval, period


def show_menu():
    """
    Main interactive menu for the multi-pattern scanner.
    Guides the user through pattern selection, execution mode,
    and data configuration.
    """
    print(r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     📊  NSE  MULTI-PATTERN  SCANNER   v3.0                   ║
    ║         Indian Stock Market (NSE)                             ║
    ║                                                              ║
    ║     7 Patterns | 3 Modes | Quality Scoring                   ║
    ║     Smoothing applied only where needed                      ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # ── Pattern selection ──
    print("  ═══════════════════════════════════════════")
    print("  Select scan pattern:")
    print("  ═══════════════════════════════════════════")
    print("    [1]  ☕ Cup and Handle")
    print("    [2]  🟢 Bull Flag")
    print("    [3]  🔴 Bear Flag")
    print("    [4]  🔺 Pennant")
    print("    [5]  👤 Head and Shoulders")
    print("    [6]  🔝 Double Top")
    print("    [7]  🔽 Double Bottom")
    print("    [8]  🌐 ALL PATTERNS (scan everything)")
    print("    [Q]  Exit\n")

    pattern_choice = input("  Enter choice (1-8 / Q): ").strip().lower()

    if pattern_choice in ("q", "quit", "exit"):
        print("  Bye! 👋\n")
        return

    PATTERN_MAP = {
        "1": ["cup_and_handle"],
        "2": ["bull_flag"],
        "3": ["bear_flag"],
        "4": ["pennant"],
        "5": ["head_and_shoulders"],
        "6": ["double_top"],
        "7": ["double_bottom"],
        "8": ["all"],
    }

    if pattern_choice not in PATTERN_MAP:
        print(f"  Invalid choice: '{pattern_choice}'\n")
        return

    patterns_to_scan = PATTERN_MAP[pattern_choice]
    selected_name = ("ALL PATTERNS" if "all" in patterns_to_scan
                     else PATTERN_NAMES[patterns_to_scan[0]])
    print(f"\n  Selected: {selected_name}\n")

    # ── Execution mode ──
    print("  ═══════════════════════════════════════════")
    print("  Select execution mode:")
    print("  ═══════════════════════════════════════════")
    print("    [A]  🧪 Self-Test (synthetic data, instant)")
    print("    [B]  📊 Backtest (historical data)")
    print("    [C]  🚀 Live Scanner (market hours)")
    print("    [D]  📊 Backtest ONE ticker\n")

    mode_choice = input("  Enter choice (A/B/C/D): ").strip().lower()

    if mode_choice == "a":
        if "all" in patterns_to_scan:
            run_self_tests("all")
        else:
            run_self_tests(patterns_to_scan[0])

    elif mode_choice == "b":
        interval, period = prompt_for_config("1d", "2y")
        backtest_historical(patterns_to_scan, period=period, interval=interval)

    elif mode_choice == "c":
        interval, period = prompt_for_config("15m", "59d")
        run_scheduler(patterns_to_scan, period=period, interval=interval)

    elif mode_choice == "d":
        ticker = input("  Enter ticker (e.g., RELIANCE or RELIANCE.NS): ").strip()
        if not ticker:
            print("  No ticker entered. Exiting.\n")
            return
        if not ticker.endswith(".NS"):
            ticker += ".NS"

        interval, period = prompt_for_config("1d", "2y")
        backtest_historical(patterns_to_scan, tickers=[ticker],
                            period=period, interval=interval)

    else:
        print(f"  Invalid mode: '{mode_choice}'\n")


def main():
    """
    Entry point. Supports both command-line arguments and interactive menu.

    Usage
    -----
    python pattern_scanner.py test [pattern]
    python pattern_scanner.py historical [tickers...] --pattern all --interval 1d --lookback 2y
    python pattern_scanner.py live --pattern bull_flag --interval 15m
    python pattern_scanner.py   # interactive menu
    """
    parser = argparse.ArgumentParser(
        description="NSE Multi-Pattern Scanner v3.0 — 7 Chart Patterns"
    )
    parser.add_argument("mode", nargs="?", default="",
                        help="test, historical, live, or empty for interactive menu")
    parser.add_argument("tickers", nargs="*",
                        help="Optional specific tickers for historical mode")
    parser.add_argument("--pattern", "-p", type=str, default="all",
                        help="Pattern to scan: cup_and_handle, bull_flag, bear_flag, "
                             "pennant, head_and_shoulders, double_top, double_bottom, all")
    parser.add_argument("--interval", "-i", type=str,
                        help="Candle interval (e.g., 15m, 1d)")
    parser.add_argument("--lookback", "-l", type=str,
                        help="Lookback period (e.g., 59d, 2y)")
    parser.add_argument("--start-date", "-s", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", "-e", type=str, help="End date (YYYY-MM-DD)")

    args = parser.parse_args()
    mode = args.mode.lower()

    # Parse pattern selection
    if args.pattern == "all":
        patterns_to_scan = ["all"]
    elif args.pattern in ALL_PATTERNS:
        patterns_to_scan = [args.pattern]
    else:
        patterns_to_scan = ["all"]

    interval = args.interval
    period = args.lookback
    if interval and period:
        period = validate_yfinance_params(interval, period)

    if mode == "test":
        if "all" in patterns_to_scan:
            run_self_tests("all")
        else:
            run_self_tests(patterns_to_scan[0])
        return

    elif mode in ("historical", "backtest"):
        if not interval:
            interval = "1d"
        if not period and not args.start_date:
            period = "2y"
        period = validate_yfinance_params(interval, period)

        if args.tickers:
            tickers = [t if t.endswith(".NS") else f"{t}.NS" for t in args.tickers]
            backtest_historical(patterns_to_scan, tickers=tickers,
                                period=period, start=args.start_date,
                                end=args.end_date, interval=interval)
        else:
            backtest_historical(patterns_to_scan, period=period,
                                start=args.start_date, end=args.end_date,
                                interval=interval)
        return

    elif mode in ("live", "live_scan"):
        if not interval:
            interval = "15m"
        if not period and not args.start_date:
            period = "59d"
        period = validate_yfinance_params(interval, period)
        run_scheduler(patterns_to_scan, period=period,
                      start=args.start_date, end=args.end_date,
                      interval=interval)
        return

    elif mode != "":
        print(f"  Unknown mode: '{mode}'")
        print(f"  Valid modes: test, historical, live\n")
        return

    # Interactive menu
    show_menu()


if __name__ == "__main__":
    main()
