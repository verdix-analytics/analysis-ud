# tests/conftest.py
import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add project root to path so stock_analysis can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def dummy_df_large():
    """
    200 rows of fake 1H OHLCV data.
    Large enough for all sliding windows:
    PRE(20) + window(30) + POST(20) = 70 minimum
    """
    np.random.seed(42)
    n      = 200
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    data   = []
    for close in closes:
        open_  = close  + np.random.randn() * 0.2
        high   = max(open_, close) + abs(np.random.randn() * 0.3)
        low    = min(open_, close) - abs(np.random.randn() * 0.3)
        volume = int(abs(np.random.randn() * 100000) + 500000)
        data.append({
            'open':   round(open_,  4),
            'high':   round(high,   4),
            'low':    round(low,    4),
            'close':  round(close,  4),
            'volume': volume,
        })
    return pd.DataFrame(data)


@pytest.fixture
def dummy_df_small():
    """
    5 rows — too small for any pattern detection.
    Used to test min_candles guard.
    """
    return pd.DataFrame({
        'open':   [100.0, 101.0, 102.0, 101.5, 100.5],
        'high':   [101.0, 102.0, 103.0, 102.5, 101.5],
        'low':    [99.0,  100.0, 101.0, 100.5, 99.5 ],
        'close':  [100.5, 101.5, 102.5, 101.0, 100.0],
        'volume': [500000, 600000, 700000, 550000, 480000],
    })