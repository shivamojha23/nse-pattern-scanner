import numpy as np
import pandas as pd
from typing import Any

def _make_serializable(obj: Any) -> Any:
    """
    Recursively converts pandas/numpy data types into standard Python types
    so that FastAPI can JSON-serialize them without errors.
    """
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [_make_serializable(i) for i in obj]
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _make_serializable(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj
