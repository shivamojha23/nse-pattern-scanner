"""
Scanner adapter — imports and wraps the existing pattern_scanner.py functions
for use by the FastAPI backend.

WHY THIS FILE EXISTS
--------------------
The existing pattern_scanner.py lives in the project root and has a module-level
side effect: it redirects sys.stdout to a DualWriter that logs everything to
results.txt. That's great for the console scanner, but would break our API
server's output.

This adapter:
1. Temporarily saves sys.stdout before importing pattern_scanner.
2. Restores sys.stdout after the import, so FastAPI output stays on the console.
3. Re-exports only the functions the API needs, keeping the import clean.

IMPORTANT: This file does NOT rewrite any detection logic. All 7 pattern
detectors, quality scoring, and validation rules come directly from
pattern_scanner.py — unchanged.
"""

import sys
import os

# ── Step 1: Add the project root to Python's import path ──
# pattern_scanner.py lives one level up from backend/, so we need to tell
# Python where to find it.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ── Step 2: Import pattern_scanner while protecting sys.stdout ──
# pattern_scanner.py replaces sys.stdout with a DualWriter on lines 79-81.
# We save the real stdout, let the import happen, then restore it.
_saved_stdout = sys.stdout

try:
    import pattern_scanner
finally:
    # Restore the original stdout so FastAPI logs go to the console, not results.txt
    sys.stdout = _saved_stdout

# ── Step 3: Re-export the functions and constants the API needs ──

# Watchlist
get_nifty_list = pattern_scanner.get_nifty_list

# Data fetching
fetch_batch_data = pattern_scanner.fetch_batch_data

# Individual pattern detectors (all 7)
detect_cup_and_handle = pattern_scanner.detect_cup_and_handle
detect_bull_flag = pattern_scanner.detect_bull_flag
detect_bear_flag = pattern_scanner.detect_bear_flag
detect_pennant = pattern_scanner.detect_pennant
detect_head_and_shoulders = pattern_scanner.detect_head_and_shoulders
detect_double_top = pattern_scanner.detect_double_top
detect_double_bottom = pattern_scanner.detect_double_bottom

# Multi-pattern scanning
scan_ticker = pattern_scanner.scan_ticker

# Market status
is_market_open = pattern_scanner.is_market_open

# Smoothing (used for any custom pre-processing)
smooth_prices = pattern_scanner.smooth_prices

# Quality scoring
compute_quality_score = pattern_scanner.compute_quality_score

# Constants
ALL_PATTERNS = pattern_scanner.ALL_PATTERNS
PATTERN_NAMES = pattern_scanner.PATTERN_NAMES
PATTERN_SIGNALS = pattern_scanner.PATTERN_SIGNALS


def validate_yfinance_params(interval, period):
    """
    Validates interval + lookback combo for yfinance limitations.
    Re-exported from pattern_scanner.
    """
    return pattern_scanner.validate_yfinance_params(interval, period)
