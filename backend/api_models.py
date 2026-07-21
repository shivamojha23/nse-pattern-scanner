import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union

# =============================================================================
#  PYDANTIC MODELS
# =============================================================================

class HealthResponse(BaseModel):
    status: str

class MarketStatusResponse(BaseModel):
    is_open: bool
    status: str
    ist_time: str
    ist_date: str
    next_event: str

class WatchlistResponse(BaseModel):
    count: int
    tickers: List[str]

class KeyLevel(BaseModel):
    label: str
    price: float
    date: str
    color: str

class Point(BaseModel):
    time: str
    value: float

class LineSegment(BaseModel):
    points: List[Point]
    color: str
    lineWidth: int = 2
    lineStyle: int = 0

class PatternCheck(BaseModel):
    name: str
    status: str
    detail: str

class ScanMatch(BaseModel):
    ticker: str
    pattern: str
    pattern_type: str
    signal: str
    status: str = "confirmed"
    quality_score: float
    key_levels: List[KeyLevel]
    line_segments: List[LineSegment] = []
    checks: List[PatternCheck]
    raw: Dict[str, Any]

class ScanResponse(BaseModel):
    scan_time: str
    interval: str
    lookback: str
    pattern: str
    total_scanned: int
    total_with_data: int
    matches: List[ScanMatch]
    scan_duration_seconds: float
    from_cache: bool
    skipped_tickers: int = 0
    errors: Dict[str, str] = Field(default_factory=dict)
    error: str = "" # Optional top level error

class Candle(BaseModel):
    time: Union[int, str]
    open: float
    high: float
    low: float
    close: float
    volume: int

class CandlesResponse(BaseModel):
    ticker: str
    interval: str
    lookback: str
    count: int
    candles: List[Candle]

class LiveAlertItem(BaseModel):
    alert_id: str
    ticker: str
    pattern: str
    pattern_type: str
    signal: str
    status: str = "confirmed"
    verdict: str
    quality_score: float
    key_levels: List[KeyLevel]
    line_segments: List[LineSegment] = []
    checks: List[PatternCheck]
    raw: Dict[str, Any]
    detected_at: str
    breakout_timestamp: str
    timeframe: str

class LiveScanResponse(BaseModel):
    count: int
    alerts: List[LiveAlertItem]

class LiveAlertsResponse(BaseModel):
    alerts: List[LiveAlertItem]

class DismissRequest(BaseModel):
    alert_id: str

class DismissResponse(BaseModel):
    status: str


# =============================================================================
#  FORMATTING HELPERS
# =============================================================================

def _format_date(ts):
    """
    Convert a pandas Timestamp/datetime to a clean date string.
    Handles various formats that come from yfinance / pattern dicts.
    """
    if ts is None:
        return None
    s = str(ts)
    # Remove timezone info (e.g. "+05:30") but keep the time for intraday
    return s.split("+")[0].replace("T", " ").strip()


from core.serialization import _make_serializable


def _extract_pattern_key_levels(pattern_dict):
    """
    Extracts the key price levels and dates from a pattern dict
    into a standardized format for the frontend.
    """
    ptype = pattern_dict.get("pattern_type", "")
    markers = []

    if ptype == "cup_and_handle":
        markers = [
            {"label": "Left Rim", "price": pattern_dict.get("left_rim_price"),
             "date": _format_date(pattern_dict.get("left_rim_date")),
             "color": "#2196F3"},  # Blue
            {"label": "Cup Bottom", "price": pattern_dict.get("cup_bottom_price"),
             "date": _format_date(pattern_dict.get("cup_bottom_date")),
             "color": "#F44336"},  # Red
            {"label": "Right Rim", "price": pattern_dict.get("right_rim_price"),
             "date": _format_date(pattern_dict.get("right_rim_date")),
             "color": "#4CAF50"},  # Green
            {"label": "Handle Low", "price": pattern_dict.get("handle_low_price"),
             "date": _format_date(pattern_dict.get("handle_low_date")),
             "color": "#FF9800"},  # Orange
        ]

    elif ptype == "bull_flag":
        markers = [
            {"label": "Pole Start", "price": pattern_dict.get("pole_start_price"),
             "date": _format_date(pattern_dict.get("pole_start_date")),
             "color": "#2196F3"},
            {"label": "Pole Top", "price": pattern_dict.get("pole_top_price"),
             "date": _format_date(pattern_dict.get("pole_top_date")),
             "color": "#F44336"},
            {"label": "Flag Low", "price": pattern_dict.get("flag_low_price"),
             "date": _format_date(pattern_dict.get("flag_end_date")),
             "color": "#4CAF50"},
            {"label": "Breakout", "price": pattern_dict.get("breakout_price"),
             "date": _format_date(pattern_dict.get("signal_date")),
             "color": "#FF9800"},
        ]

    elif ptype == "bear_flag":
        markers = [
            {"label": "Pole Start", "price": pattern_dict.get("pole_start_price"),
             "date": _format_date(pattern_dict.get("pole_start_date")),
             "color": "#2196F3"},
            {"label": "Pole Bottom", "price": pattern_dict.get("pole_bottom_price"),
             "date": _format_date(pattern_dict.get("pole_bottom_date")),
             "color": "#F44336"},
            {"label": "Flag High", "price": pattern_dict.get("flag_high_price"),
             "date": _format_date(pattern_dict.get("flag_end_date")),
             "color": "#4CAF50"},
            {"label": "Breakdown", "price": pattern_dict.get("breakdown_price"),
             "date": _format_date(pattern_dict.get("signal_date")),
             "color": "#FF9800"},
        ]

    elif ptype == "pennant":
        markers = [
            {"label": "Pole Start", "price": pattern_dict.get("pole_start_price"),
             "date": _format_date(pattern_dict.get("pole_start_date")),
             "color": "#2196F3"},
            {"label": "Pole End", "price": pattern_dict.get("pole_end_price"),
             "date": _format_date(pattern_dict.get("pole_end_date")),
             "color": "#F44336"},
            {"label": "Pennant High", "price": pattern_dict.get("pennant_high"),
             "date": _format_date(pattern_dict.get("pennant_start_date")),
             "color": "#4CAF50"},
            {"label": "Breakout", "price": pattern_dict.get("breakout_price"),
             "date": _format_date(pattern_dict.get("breakout_date")),
             "color": "#FF9800"},
        ]

    elif ptype == "head_and_shoulders":
        markers = [
            {"label": "Left Shoulder", "price": pattern_dict.get("left_shoulder_price"),
             "date": _format_date(pattern_dict.get("left_shoulder_date")),
             "color": "#2196F3"},
            {"label": "Head", "price": pattern_dict.get("head_price"),
             "date": _format_date(pattern_dict.get("head_date")),
             "color": "#F44336"},
            {"label": "Right Shoulder", "price": pattern_dict.get("right_shoulder_price"),
             "date": _format_date(pattern_dict.get("right_shoulder_date")),
             "color": "#4CAF50"},
            {"label": "Breakdown", "price": pattern_dict.get("breakdown_price"),
             "date": _format_date(pattern_dict.get("breakdown_date")),
             "color": "#FF9800"},
        ]

    elif ptype == "double_top":
        markers = [
            {"label": "First Top", "price": pattern_dict.get("first_top_price"),
             "date": _format_date(pattern_dict.get("first_top_date")),
             "color": "#2196F3"},
            {"label": "Valley", "price": pattern_dict.get("valley_price"),
             "date": _format_date(pattern_dict.get("valley_date")),
             "color": "#F44336"},
            {"label": "Second Top", "price": pattern_dict.get("second_top_price"),
             "date": _format_date(pattern_dict.get("second_top_date")),
             "color": "#4CAF50"},
            {"label": "Breakdown", "price": pattern_dict.get("breakdown_price"),
             "date": _format_date(pattern_dict.get("breakdown_date")),
             "color": "#FF9800"},
        ]

    elif ptype == "double_bottom":
        markers = [
            {"label": "First Bottom", "price": pattern_dict.get("first_bottom_price"),
             "date": _format_date(pattern_dict.get("first_bottom_date")),
             "color": "#2196F3"},
            {"label": "Peak", "price": pattern_dict.get("peak_price"),
             "date": _format_date(pattern_dict.get("peak_date")),
             "color": "#F44336"},
            {"label": "Second Bottom", "price": pattern_dict.get("second_bottom_price"),
             "date": _format_date(pattern_dict.get("second_bottom_date")),
             "color": "#4CAF50"},
            {"label": "Breakout", "price": pattern_dict.get("breakout_price"),
             "date": _format_date(pattern_dict.get("breakout_date")),
             "color": "#FF9800"},
        ]

    # Filter out markers with missing data
    return [m for m in markers if m.get("price") is not None and m.get("date") is not None]


def _extract_line_segments(pattern_dict):
    ptype = pattern_dict.get("pattern_type", "")
    segments = []

    if ptype == "pennant":
        # The Pole
        if pattern_dict.get("pole_start_date") and pattern_dict.get("pole_end_date"):
            segments.append({
                "points": [
                    {"time": _format_date(pattern_dict["pole_start_date"]), "value": pattern_dict["pole_start_price"]},
                    {"time": _format_date(pattern_dict["pole_end_date"]), "value": pattern_dict["pole_end_price"]}
                ],
                "color": "#2196F3", "lineWidth": 2, "lineStyle": 0
            })
        
        # Upper and Lower Trendlines
        p_start_date = pattern_dict.get("pennant_start_date")
        p_end_date = pattern_dict.get("pennant_end_date")
        # We need the prices at start and end for both lines.
        # res_high_intercept is the price at x=0 (which is pennant_start).
        start_idx = pattern_dict.get("pennant_start_idx")
        end_idx = pattern_dict.get("pennant_end_idx")
        upper_slope = pattern_dict.get("upper_slope")
        lower_slope = pattern_dict.get("lower_slope")
        upper_int = pattern_dict.get("res_high_intercept")
        lower_int = pattern_dict.get("res_low_intercept")
        
        if all(v is not None for v in [start_idx, end_idx, upper_slope, lower_slope, upper_int, lower_int, p_start_date, p_end_date]):
            p_len = end_idx - start_idx
            
            segments.append({
                "points": [
                    {"time": _format_date(p_start_date), "value": upper_int},
                    {"time": _format_date(p_end_date), "value": upper_int + upper_slope * p_len}
                ],
                "color": "#F44336", "lineWidth": 2, "lineStyle": 2
            })
            
            segments.append({
                "points": [
                    {"time": _format_date(p_start_date), "value": lower_int},
                    {"time": _format_date(p_end_date), "value": lower_int + lower_slope * p_len}
                ],
                "color": "#4CAF50", "lineWidth": 2, "lineStyle": 2
            })

    elif ptype in ("bull_flag", "bear_flag"):
        # The Pole
        if pattern_dict.get("pole_start_date") and pattern_dict.get("pole_bottom_date" if ptype == "bear_flag" else "pole_top_date"):
            segments.append({
                "points": [
                    {"time": _format_date(pattern_dict["pole_start_date"]), "value": pattern_dict["pole_start_price"]},
                    {"time": _format_date(pattern_dict["pole_bottom_date" if ptype == "bear_flag" else "pole_top_date"]), "value": pattern_dict["pole_bottom_price" if ptype == "bear_flag" else "pole_top_price"]}
                ],
                "color": "#2196F3", "lineWidth": 2, "lineStyle": 0
            })
        
        f_start_date = pattern_dict.get("flag_start_date")
        f_end_date = pattern_dict.get("flag_end_date")
        start_idx = pattern_dict.get("flag_start_idx")
        end_idx = pattern_dict.get("flag_end_idx")
        slope = pattern_dict.get("flag_slope")
        upper_int = pattern_dict.get("flag_upper_intercept")
        lower_int = pattern_dict.get("flag_lower_intercept")

        if all(v is not None for v in [start_idx, end_idx, slope, upper_int, lower_int, f_start_date, f_end_date]):
            f_len = end_idx - start_idx
            
            segments.append({
                "points": [
                    {"time": _format_date(f_start_date), "value": upper_int},
                    {"time": _format_date(f_end_date), "value": upper_int + slope * f_len}
                ],
                "color": "#F44336", "lineWidth": 2, "lineStyle": 2
            })
            
            segments.append({
                "points": [
                    {"time": _format_date(f_start_date), "value": lower_int},
                    {"time": _format_date(f_end_date), "value": lower_int + slope * f_len}
                ],
                "color": "#4CAF50", "lineWidth": 2, "lineStyle": 2
            })
            
    elif ptype == "head_and_shoulders":
        if pattern_dict.get("left_neckline_date") and pattern_dict.get("right_neckline_date"):
            segments.append({
                "points": [
                    {"time": _format_date(pattern_dict["left_neckline_date"]), "value": pattern_dict["left_neckline_price"]},
                    {"time": _format_date(pattern_dict["right_neckline_date"]), "value": pattern_dict["right_neckline_price"]}
                ],
                "color": "#F44336", "lineWidth": 2, "lineStyle": 2
            })

    elif ptype == "double_top":
        if pattern_dict.get("first_top_date") and pattern_dict.get("second_top_date"):
            segments.append({
                "points": [
                    {"time": _format_date(pattern_dict["first_top_date"]), "value": pattern_dict["valley_price"]},
                    {"time": _format_date(pattern_dict["second_top_date"]), "value": pattern_dict["valley_price"]}
                ],
                "color": "#F44336", "lineWidth": 2, "lineStyle": 2
            })
            
    elif ptype == "double_bottom":
        if pattern_dict.get("first_bottom_date") and pattern_dict.get("second_bottom_date"):
            segments.append({
                "points": [
                    {"time": _format_date(pattern_dict["first_bottom_date"]), "value": pattern_dict["peak_price"]},
                    {"time": _format_date(pattern_dict["second_bottom_date"]), "value": pattern_dict["peak_price"]}
                ],
                "color": "#4CAF50", "lineWidth": 2, "lineStyle": 2
            })

    return segments


def _extract_checks(pattern_dict):
    """
    Extracts PASS/FAIL check results from a pattern dict into a
    standardized list for the frontend to display as badges.
    """
    ptype = pattern_dict.get("pattern_type", "")
    checks = []

    if ptype == "cup_and_handle":
        # Geometry
        checks.append({
            "name": "Geometry",
            "status": "PASS" if not pattern_dict.get("reject_reason") else "FAIL",
            "detail": f"Cup drop {pattern_dict.get('cup_drop_pct', 0)}%, Recovery gap {pattern_dict.get('recovery_pct', 0)}%"
        })
        # Roundedness
        rnd = pattern_dict.get("roundedness_pct", 0)
        checks.append({
            "name": "Roundedness",
            "status": "PASS" if rnd >= 20 else "FAIL",
            "detail": f"{rnd}% of candles in base zone (need ≥20%)"
        })
        # Pause before breakout
        pause = pattern_dict.get("pause_duration", 0)
        slope = pattern_dict.get("handle_slope", 0)
        checks.append({
            "name": "Pause Before Breakout",
            "status": "PASS" if pause >= 5 and slope <= 0 else "FAIL",
            "detail": f"{pause} candles, slope {slope}"
        })
        # Volume checks
        for vol_key, vol_name in [
            ("vol_decline_pass", "Volume Decline"),
            ("vol_recovery_pass", "Volume Recovery"),
            ("vol_breakout_pass", "Volume Breakout"),
        ]:
            val = pattern_dict.get(vol_key)
            checks.append({
                "name": vol_name,
                "status": "PASS" if val is True else ("FAIL" if val is False else "N/A"),
                "detail": ""
            })

    elif ptype in ("bull_flag", "bear_flag"):
        for vol_key, vol_name in [
            ("vol_pole_pass", "Pole Volume"),
            ("vol_flag_pass", "Flag Volume"),
            ("vol_breakout_pass" if ptype == "bull_flag" else "vol_breakdown_pass",
             "Breakout Volume" if ptype == "bull_flag" else "Breakdown Volume"),
        ]:
            val = pattern_dict.get(vol_key)
            checks.append({
                "name": vol_name,
                "status": "PASS" if val is True else ("FAIL" if val is False else "N/A"),
                "detail": ""
            })
        checks.append({
            "name": "Pole Linearity (R²)",
            "status": "PASS" if pattern_dict.get("pole_r_squared", 0) >= 0.8 else "FAIL",
            "detail": f"R² = {pattern_dict.get('pole_r_squared', 0)}"
        })
        key = "pole_rise_pct" if ptype == "bull_flag" else "pole_drop_pct"
        checks.append({
            "name": "Pole Strength",
            "status": "PASS",
            "detail": f"{pattern_dict.get(key, 0)}%"
        })

    elif ptype == "pennant":
        for vol_key, vol_name in [
            ("vol_pole_pass", "Pole Volume"),
            ("vol_pennant_pass", "Pennant Volume"),
            ("vol_breakout_pass", "Breakout Volume"),
        ]:
            val = pattern_dict.get(vol_key)
            checks.append({
                "name": vol_name,
                "status": "PASS" if val is True else ("FAIL" if val is False else "N/A"),
                "detail": ""
            })
        cr = pattern_dict.get("convergence_ratio", 1)
        checks.append({
            "name": "Convergence",
            "status": "PASS" if cr < 1 else "FAIL",
            "detail": f"Ratio = {cr}"
        })

    elif ptype == "head_and_shoulders":
        checks.append({
            "name": "Shoulder Symmetry",
            "status": "PASS" if pattern_dict.get("shoulder_symmetry_pct", 99) <= 5 else "FAIL",
            "detail": f"{pattern_dict.get('shoulder_symmetry_pct', 0)}% diff"
        })
        checks.append({
            "name": "Head Prominence",
            "status": "PASS",
            "detail": f"LS +{pattern_dict.get('head_vs_ls_pct', 0)}%, RS +{pattern_dict.get('head_vs_rs_pct', 0)}%"
        })
        val = pattern_dict.get("vol_progression_pass")
        checks.append({
            "name": "Volume Progression",
            "status": "PASS" if val is True else ("FAIL" if val is False else "N/A"),
            "detail": "LS > Head > RS (declining)"
        })
        val = pattern_dict.get("vol_breakdown_pass")
        checks.append({
            "name": "Breakdown Volume",
            "status": "PASS" if val is True else ("FAIL" if val is False else "N/A"),
            "detail": ""
        })

    elif ptype in ("double_top", "double_bottom"):
        checks.append({
            "name": "Peak/Bottom Similarity",
            "status": "PASS" if pattern_dict.get("similarity_pct", 99) <= 6 else "FAIL",
            "detail": f"{pattern_dict.get('similarity_pct', 0)}% diff (≤6% required)"
        })
        depth_key = "valley_drop_pct" if ptype == "double_top" else "peak_rise_pct"
        checks.append({
            "name": "Depth/Height",
            "status": "PASS",
            "detail": f"{pattern_dict.get(depth_key, 0)}%"
        })
        val = pattern_dict.get("vol_pattern_pass")
        checks.append({
            "name": "Volume Pattern",
            "status": "PASS" if val is True else ("FAIL" if val is False else "N/A"),
            "detail": ""
        })
        bk = "vol_breakdown_pass" if ptype == "double_top" else "vol_breakout_pass"
        val = pattern_dict.get(bk)
        checks.append({
            "name": "Breakout/Breakdown Volume",
            "status": "PASS" if val is True else ("FAIL" if val is False else "N/A"),
            "detail": ""
        })

    return checks


def _extract_pattern_status(pattern_dict):
    """
    Derives "confirmed" or "forming" from detector output.

    - double_top / double_bottom / pennant: have explicit `status` field
    - cup_and_handle: has `breakout_confirmed` boolean
    - All others: only emit on breakout -> always "confirmed" (until Track B)
    """
    ptype = pattern_dict.get("pattern_type", "")

    # If the detector explicitly provided a status string, use it
    if "status" in pattern_dict:
        return pattern_dict["status"]

    # Cup and handle uses breakout_confirmed boolean
    if ptype == "cup_and_handle":
        return "confirmed" if pattern_dict.get("breakout_confirmed", True) else "forming"

    # Track B detectors - always confirmed until forming logic is added
    return "confirmed"
