import numpy as np
import pandas as pd
from backend.api_models import _make_serializable

def test_make_serializable_handles_numpy_types():
    """
    Independent test to verify our custom serialization function actually
    flattens numpy types to Python built-ins, ensuring our API responses
    don't crash with JSON encoding errors.
    """
    raw_data = {
        "int_val": np.int64(42),
        "float_val": np.float64(3.14159),
        "array_val": np.array([1, 2, 3]),
        "bool_val": np.bool_(True),
        "timestamp_val": pd.Timestamp("2026-07-14 10:00:00+05:30")
    }

    result = _make_serializable(raw_data)

    assert isinstance(result["int_val"], int)
    assert result["int_val"] == 42
    
    assert isinstance(result["float_val"], float)
    assert result["float_val"] == 3.14159
    
    assert isinstance(result["array_val"], list)
    assert result["array_val"] == [1, 2, 3]
    
    assert isinstance(result["bool_val"], bool)
    assert result["bool_val"] is True
    
    assert isinstance(result["timestamp_val"], str)
    assert result["timestamp_val"] == "2026-07-14 10:00:00+05:30"
