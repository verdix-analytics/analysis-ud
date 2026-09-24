from __future__ import annotations
import math
import logging
import numpy as np
import pandas as pd
from pydantic import BaseModel
logger = logging.getLogger(__name__)
# Internal helpers
def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and title-case column names (mirrors trendanalysis.py)."""
    return df.rename(columns=lambda c: c.strip().title())

def _require_columns(df: pd.DataFrame, *cols: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame is missing required column(s): {missing}")

def _ema(series: pd.Series, period: int) -> pd.Series:
    """Standard EMA using span-based ewm (adjust=False)."""
    return series.ewm(span=period, adjust=False).mean()

def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

def _wma(series: pd.Series, period: int) -> pd.Series:
    """Linearly-weighted moving average."""
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )
# 1. Relative Strength Index
def rsi(df: pd.DataFrame, period: int = 14, column: str = "Close") -> pd.Series:
    """
    df     : Normalised OHLCV DataFrame.
    period : Look-back length (default 14).
    column : Price column (default 'Close').

    Returns
    -------
    pd.Series  labelled 'RSI_<period>', values in [0, 100].
    """
    df = _normalise(df)
    _require_columns(df, column)

    delta = df[column].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rs     = avg_gain / avg_loss.replace(0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    result.name = f"RSI_{period}"
    return result
# 2. Awesome Oscillator
def awesome_oscillator(
    df: pd.DataFrame,
    fast: int = 5,
    slow: int = 34,
) -> pd.Series:
    """
    pd.Series  labelled 'AO'.
    """
    df = _normalise(df)
    _require_columns(df, "High", "Low")

    median = (df["High"] + df["Low"]) / 2.0
    result = _sma(median, fast) - _sma(median, slow)
    result.name = "AO"
    return result
# 3. MACD
def macd(
    df: pd.DataFrame,
    fast: int   = 12,
    slow: int   = 26,
    signal: int = 9,
    column: str = "Close",
) -> pd.DataFrame:
    """
    pd.DataFrame with columns:
        'MACD'        - MACD line  (fast EMA - slow EMA)
        'MACD_Signal' - Signal line (EMA of MACD)
        'MACD_Hist'   - Histogram   (MACD - Signal)
    """
    df = _normalise(df)
    _require_columns(df, column)

    ema_fast    = _ema(df[column], fast)
    ema_slow    = _ema(df[column], slow)
    macd_line   = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram   = macd_line - signal_line

    return pd.DataFrame(
        {"MACD": macd_line, "MACD_Signal": signal_line, "MACD_Hist": histogram},
        index=df.index,
    )
# 4. Bear / Bull Power  (Elder Ray Index)
def bear_bull_power(
    df: pd.DataFrame,
    period: int = 13,
    column: str = "Close",
) -> pd.DataFrame:
    """
    Bear Power & Bull Power (Elder Ray Index).

    Bull Power = High - EMA(Close, period)
    Bear Power = Low  - EMA(Close, period)
    """
    df = _normalise(df)
    _require_columns(df, "High", "Low", column)

    ema_line = _ema(df[column], period)
    return pd.DataFrame(
        {"Bull_Power": df["High"] - ema_line, "Bear_Power": df["Low"] - ema_line},
        index=df.index,
    )
# 5. Average Directional Index
def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Average Directional Index (ADX) with +DI and -DI.

    Uses Wilder's smoothing (ewm com = period - 1) throughout, matching the
    original Wilder definition.
    """
    df = _normalise(df)
    _require_columns(df, "High", "Low", "Close")

    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    up_move   =  high.diff()
    down_move = -low.diff()

    plus_dm  = pd.Series(
        np.where((up_move > down_move)   & (up_move   > 0), up_move,   0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move)   & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    smoothed_tr  = tr.ewm(com=period - 1, adjust=False).mean()
    plus_di      = 100.0 * plus_dm.ewm(com=period - 1, adjust=False).mean() / smoothed_tr
    minus_di     = 100.0 * minus_dm.ewm(com=period - 1, adjust=False).mean() / smoothed_tr
    di_sum       = (plus_di + minus_di).replace(0, np.nan)
    dx           = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx_values   = dx.ewm(com=period - 1, adjust=False).mean()

    return pd.DataFrame(
        {"ADX": adx_values, "+DI": plus_di, "-DI": minus_di},
        index=df.index,
    )
# 6 & 7.  EMA / SMA - 10-period
def ema_10(df: pd.DataFrame, column: str = "Close") -> pd.Series:
    """Exponential Moving Average - 10-period."""
    df = _normalise(df)
    _require_columns(df, column)
    result = _ema(df[column], 10)
    result.name = "EMA_10"
    return result


def sma_10(df: pd.DataFrame, column: str = "Close") -> pd.Series:
    """Simple Moving Average - 10-period."""
    df = _normalise(df)
    _require_columns(df, column)
    result = _sma(df[column], 10)
    result.name = "SMA_10"
    return result
# 8 & 9.  EMA / SMA - 20-period
def ema_20(df: pd.DataFrame, column: str = "Close") -> pd.Series:
    """Exponential Moving Average - 20-period."""
    df = _normalise(df)
    _require_columns(df, column)
    result = _ema(df[column], 20)
    result.name = "EMA_20"
    return result


def sma_20(df: pd.DataFrame, column: str = "Close") -> pd.Series:
    """Simple Moving Average - 20-period."""
    df = _normalise(df)
    _require_columns(df, column)
    result = _sma(df[column], 20)
    result.name = "SMA_20"
    return result
# 10. Hull Moving Average - 9-period
def hma_9(df: pd.DataFrame, column: str = "Close") -> pd.Series:
    """
    Hull Moving Average - 9-period.

    HMA(n) = WMA( 2*WMA(n/2) - WMA(n),  floor(sqrt(n)) )
    """
    df = _normalise(df)
    _require_columns(df, column)

    n      = 9
    half   = max(int(n / 2), 1)
    sqrt_n = max(int(math.floor(math.sqrt(n))), 1)

    raw    = 2.0 * _wma(df[column], half) - _wma(df[column], n)
    result = _wma(raw, sqrt_n)
    result.name = "HMA_9"
    return result
# Orchestrator class  (mirrors TrendAnalysis interface from trendanalysis.py)
class TechnicalAnalysis(BaseModel):
    """
    Thin orchestrator for technical indicators.

    Follows the same Pydantic BaseModel pattern as TrendAnalysis so it can be
    instantiated at module level and reused across calls, just like:

        trend_analysis    = TrendAnalysis()       # already in engine.py
        technical_analysis = TechnicalAnalysis()  # drop-in companion
    """
    # Optional cache - mirrors history pattern in TrendAnalysis
    last_indicators: dict | None = None

    # ── public entry-point ──────────────────────────────────────────────────
    def compute_all(
        self,
        df: pd.DataFrame,
        column: str = "Close",
    ) -> pd.DataFrame:
        """
        Compute every indicator and return them appended to the original data.

        Indicator columns added
        ───────────────────────
        RSI_14, AO, MACD, MACD_Signal, MACD_Hist,
        Bull_Power, Bear_Power,
        ADX, +DI, -DI,
        EMA_10, SMA_10, EMA_20, SMA_20, HMA_9
        """
        try:
            result = _normalise(df).copy()

            result["RSI_14"] = rsi(df, column=column)
            result["AO"]     = awesome_oscillator(df)

            for col, series in macd(df, column=column).items():
                result[col] = series

            for col, series in bear_bull_power(df, column=column).items():
                result[col] = series

            for col, series in adx(df).items():
                result[col] = series

            result["EMA_10"] = ema_10(df, column=column)
            result["SMA_10"] = sma_10(df, column=column)
            result["EMA_20"] = ema_20(df, column=column)
            result["SMA_20"] = sma_20(df, column=column)
            result["HMA_9"]  = hma_9(df, column=column)

            # Cache a lightweight summary for optional downstream inspection
            self.last_indicators = {
                "rsi_last":        _safe_last(result, "RSI_14"),
                "ao_last":         _safe_last(result, "AO"),
                "macd_last":       _safe_last(result, "MACD"),
                "macd_hist_last":  _safe_last(result, "MACD_Hist"),
                "adx_last":        _safe_last(result, "ADX"),
                "bull_power_last": _safe_last(result, "Bull_Power"),
                "bear_power_last": _safe_last(result, "Bear_Power"),
                "ema10_last":      _safe_last(result, "EMA_10"),
                "sma10_last":      _safe_last(result, "SMA_10"),
                "ema20_last":      _safe_last(result, "EMA_20"),
                "sma20_last":      _safe_last(result, "SMA_20"),
                "hma9_last":       _safe_last(result, "HMA_9"),
            }

            return result

        except Exception as exc:
            logger.error("TechnicalAnalysis.compute_all failed: %s", exc)
            return _normalise(df).copy()

    # ── convenience: summary dict of the latest values ──────────────────────
    def latest_snapshot(self, df: pd.DataFrame, column: str = "Close") -> dict:
        """
        Return a flat dict of the most-recent value for every indicator.
        """
        enriched = self.compute_all(df, column=column)
        cols_map = {
            "rsi":         "RSI_14",
            "ao":          "AO",
            "macd":        "MACD",
            "macd_signal": "MACD_Signal",
            "macd_hist":   "MACD_Hist",
            "bull_power":  "Bull_Power",
            "bear_power":  "Bear_Power",
            "adx":         "ADX",
            "plus_di":     "+DI",
            "minus_di":    "-DI",
            "ema_10":      "EMA_10",
            "sma_10":      "SMA_10",
            "ema_20":      "EMA_20",
            "sma_20":      "SMA_20",
            "hma_9":       "HMA_9",
        }
        return {
            label: _safe_last(enriched, col)
            for label, col in cols_map.items()
        }
# Internal utility
def _safe_last(df: pd.DataFrame, col: str) -> float | None:
    """Return the last non-NaN value of a column, or None."""
    try:
        series = df[col].dropna()
        return round(float(series.iloc[-1]), 6) if not series.empty else None
    except Exception:
        return None

