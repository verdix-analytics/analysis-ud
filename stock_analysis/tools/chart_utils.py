import pandas as pd
import numpy as np


def safe_ratio(numerator: float, denominator: float) -> float:
    """
    Safe division to avoid division by zero.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator


def pct_diff(a: float, b: float) -> float:
    """
    Percentage difference between two values.
    Example:
        a=100, b=105 -> 0.05
    """
    if a == 0 and b == 0:
        return 0.0
    base = (abs(a) + abs(b)) / 2
    if base == 0:
        return 0.0
    return abs(a - b) / base


def is_close_pct(a: float, b: float, tol: float = 0.03) -> bool:
    """
    Check whether two values are close within percentage tolerance.
    Example:
        tol=0.03 means within 3%
    """
    return pct_diff(a, b) <= tol


def detect_pivots(
    df: pd.DataFrame,
    lookback: int = 2,
    min_move_pct: float = 0.0
) -> pd.DataFrame:
    """
    Detect local pivot highs and lows inside the input window.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
        ['Datetime', 'Open', 'High', 'Low', 'Close']
    lookback : int
        Number of candles to compare on both left and right side.
    min_move_pct : float
        Optional minimum percentage move required between consecutive pivots.
        Example: 0.01 means 1%.

    Returns
    -------
    pd.DataFrame
        Columns:
        ['Datetime', 'pivot_type', 'pivot_price', 'candle_index']
    """

    if len(df) < 2 * lookback + 1:
        return pd.DataFrame(columns=["Datetime", "pivot_type", "pivot_price", "candle_index"])

    df = df.reset_index(drop=True).copy()

    highs = df["High"].values
    lows = df["Low"].values
    times = df["Datetime"].values

    raw_pivots = []

    for i in range(lookback, len(df) - lookback):
        left_highs = highs[i - lookback:i]
        right_highs = highs[i + 1:i + 1 + lookback]

        left_lows = lows[i - lookback:i]
        right_lows = lows[i + 1:i + 1 + lookback]

        is_pivot_high = (highs[i] > left_highs.max()) and (highs[i] > right_highs.max())
        is_pivot_low = (lows[i] < left_lows.min()) and (lows[i] < right_lows.min())

        if is_pivot_high:
            raw_pivots.append({
                "Datetime": times[i],
                "pivot_type": "H",
                "pivot_price": highs[i],
                "candle_index": i,
            })
        elif is_pivot_low:
            raw_pivots.append({
                "Datetime": times[i],
                "pivot_type": "L",
                "pivot_price": lows[i],
                "candle_index": i,
            })

    if not raw_pivots:
        return pd.DataFrame(columns=["Datetime", "pivot_type", "pivot_price", "candle_index"])

    pivots = pd.DataFrame(raw_pivots).sort_values("candle_index").reset_index(drop=True)

    # Step 1: remove consecutive same-type pivots
    # keep only the stronger one
    cleaned = []

    for _, row in pivots.iterrows():
        if not cleaned:
            cleaned.append(row.to_dict())
            continue

        last = cleaned[-1]

        if row["pivot_type"] != last["pivot_type"]:
            cleaned.append(row.to_dict())
        else:
            # Same pivot type: keep stronger extreme
            if row["pivot_type"] == "H":
                if row["pivot_price"] > last["pivot_price"]:
                    cleaned[-1] = row.to_dict()
            else:  # "L"
                if row["pivot_price"] < last["pivot_price"]:
                    cleaned[-1] = row.to_dict()

    pivots = pd.DataFrame(cleaned)

    # Step 2: optional min move filter
    if min_move_pct > 0 and len(pivots) > 1:
        filtered = [pivots.iloc[0].to_dict()]

        for i in range(1, len(pivots)):
            prev_price = filtered[-1]["pivot_price"]
            curr_row = pivots.iloc[i].to_dict()
            move_pct = safe_ratio(abs(curr_row["pivot_price"] - prev_price), abs(prev_price))

            if move_pct >= min_move_pct:
                filtered.append(curr_row)

        pivots = pd.DataFrame(filtered)

    return pivots.reset_index(drop=True)


def get_last_n_pivots(pivots: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Get last n pivots from pivot dataframe.
    """
    if pivots.empty:
        return pivots.copy()
    return pivots.tail(n).reset_index(drop=True)


def calc_slope(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate slope between two points.
    """
    dx = x2 - x1
    if dx == 0:
        return 0.0
    return (y2 - y1) / dx


def fit_line_slope(x: list, y: list) -> float:
    """
    Fit a simple linear regression line and return slope.
    Requires at least 2 points.
    """
    slope, _ = fit_line_params(x, y)
    return slope


def fit_line_params(x: list, y: list) -> tuple[float, float]:
    """
    Fit a simple linear regression line and return (slope, intercept).
    Requires at least 2 points.
    """
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return 0.0, 0.0

    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)

    # degree 1 polynomial fit
    slope, intercept = np.polyfit(x_arr, y_arr, 1)
    return float(slope), float(intercept)


def fit_line_pct_slope(x: list, y: list) -> float:
    """
    Fit a line and normalize its slope by the average absolute price level.
    This makes slope thresholds more stable across instruments.
    """
    slope, _ = fit_line_params(x, y)
    if len(y) < 2:
        return 0.0

    scale = float(np.mean(np.abs(np.array(y, dtype=float))))
    return safe_ratio(slope, max(scale, 1e-9))


def line_value(slope: float, intercept: float, x: float) -> float:
    """
    Evaluate a fitted line at x.
    """
    return slope * x + intercept


def count_line_touches(
    x: list,
    y: list,
    slope: float,
    intercept: float,
    tol_pct: float = 0.01,
) -> int:
    """
    Count how many points lie near a fitted line within a percentage tolerance.
    """
    touches = 0
    for xi, yi in zip(x, y):
        expected = line_value(slope, intercept, xi)
        if is_close_pct(yi, expected, tol=tol_pct):
            touches += 1
    return touches


def pivots_to_lists(pivots: pd.DataFrame) -> dict:
    """
    Convert pivot dataframe to simple lists for convenience.

    Returns
    -------
    dict:
        {
            "types": [...],
            "prices": [...],
            "indices": [...],
            "times": [...]
        }
    """
    if pivots.empty:
        return {
            "types": [],
            "prices": [],
            "indices": [],
            "times": [],
        }

    return {
        "types": pivots["pivot_type"].tolist(),
        "prices": pivots["pivot_price"].tolist(),
        "indices": pivots["candle_index"].tolist(),
        "times": pivots["Datetime"].tolist(),
    }