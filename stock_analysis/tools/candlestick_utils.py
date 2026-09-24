import pandas as pd
import numpy as np


def add_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic candlestick features to dataframe.

    Required columns:
        ['Open', 'High', 'Low', 'Close']

    Returns:
        DataFrame with added columns:
        - body
        - range
        - upper_wick
        - lower_wick
        - direction
    """

    df_feat = df.copy()

    # --- Candle body ---
    df_feat["body"] = (df_feat["Close"] - df_feat["Open"]).abs()

    # --- Candle range ---
    df_feat["range"] = df_feat["High"] - df_feat["Low"]

    # --- Upper wick ---
    df_feat["upper_wick"] = (
        df_feat["High"] - df_feat[["Open", "Close"]].max(axis=1)
    )

    # --- Lower wick ---
    df_feat["lower_wick"] = (
        df_feat[["Open", "Close"]].min(axis=1) - df_feat["Low"]
    )

    # --- Direction ---
    df_feat["direction"] = np.where(
        df_feat["Close"] > df_feat["Open"],
        "bullish",
        np.where(
            df_feat["Close"] < df_feat["Open"],
            "bearish",
            "neutral",
        ),
    )

    return df_feat


def safe_ratio(numerator: float, denominator: float) -> float:
    """
    Safe division to avoid division by zero.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator

def get_last_candle(df: pd.DataFrame) -> pd.Series:
    """
    Get the last candle (row) from dataframe.
    """
    return df.iloc[-1]


def get_last_n_candles(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Get last n candles.
    """
    return df.iloc[-n:]


def body_top_ratio(candle: pd.Series) -> float:
    """
    Position of the top of the candle body within the full candle range.
    1.0 means the body top touches the high, 0.0 means it touches the low.
    Falls back to computing range from High - Low when the precomputed
    candlestick feature is not present.
    """
    range_ = candle["range"] if "range" in candle else candle["High"] - candle["Low"]
    return safe_ratio(max(candle["Open"], candle["Close"]) - candle["Low"], range_)


def body_bottom_ratio(candle: pd.Series) -> float:
    """
    Position of the bottom of the candle body within the full candle range.
    1.0 means the body bottom touches the high, 0.0 means it touches the low.
    Falls back to computing range from High - Low when the precomputed
    candlestick feature is not present.
    """
    range_ = candle["range"] if "range" in candle else candle["High"] - candle["Low"]
    return safe_ratio(min(candle["Open"], candle["Close"]) - candle["Low"], range_)