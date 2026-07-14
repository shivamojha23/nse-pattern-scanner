"""
Scanner adapter — imports and wraps the core orchestration and detection functions
for use by the FastAPI backend.

This adapter now cleanly imports from `core.orchestrator` and `detectors.core`.
The previous sys.stdout hack is removed, as the orchestration logic no longer
resides in the CLI entrypoint (pattern_scanner.py) which caused side effects.
"""

import sys
import os

# ── Step 1: Add the project root to Python's import path ──
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ── Step 2: Import from core orchestrator and detectors ──
import core.orchestrator as orchestrator
from detectors.cup_and_handle import detect_cup_and_handle
from detectors.bull_flag import detect_bull_flag
from detectors.bear_flag import detect_bear_flag
from detectors.pennant import detect_pennant
from detectors.head_and_shoulders import detect_head_and_shoulders
from detectors.double_top import detect_double_top
from detectors.double_bottom import detect_double_bottom
import detectors.core as detectors_core

# ── Step 3: Re-export the functions and constants the API needs ──

# Watchlist
get_nifty_list = orchestrator.get_nifty_list

# Data fetching
fetch_batch_data = orchestrator.fetch_batch_data

# Multi-pattern scanning
scan_ticker = orchestrator.scan_ticker
scan_watchlist = orchestrator.scan_watchlist

# Market status
is_market_open = orchestrator.is_market_open

# Smoothing (used for any custom pre-processing)
smooth_prices = detectors_core.smooth_prices

# Quality scoring
compute_quality_score = detectors_core.compute_quality_score

# Constants
ALL_PATTERNS = orchestrator.ALL_PATTERNS
PATTERN_NAMES = orchestrator.PATTERN_NAMES
PATTERN_SIGNALS = orchestrator.PATTERN_SIGNALS

def validate_yfinance_params(interval, period):
    """
    Validates interval + lookback combo for yfinance limitations.
    Re-exported from core.orchestrator.
    """
    return orchestrator.validate_yfinance_params(interval, period)
