import pandas as pd
import numpy as np
import pytest

from stock_analysis.tools.candlestick_utils import (
    add_candlestick_features,
    safe_ratio,
    get_last_candle,
    get_last_n_candles,
    body_top_ratio,
    body_bottom_ratio,
)


# =========================
# Fixtures
# =========================

@pytest.fixture
def basic_ohlc_df():
    return pd.DataFrame({
        "Open":  [10, 12, 15],
        "High":  [15, 14, 18],
        "Low":   [ 8, 11, 14],
        "Close": [14, 11, 15],
    })


@pytest.fixture
def single_row_df():
    return pd.DataFrame({
        "Open":  [10],
        "High":  [15],
        "Low":   [8],
        "Close": [14],
    })


@pytest.fixture
def empty_ohlc_df():
    return pd.DataFrame(columns=["Open", "High", "Low", "Close"])


@pytest.fixture
def zero_range_df():
    return pd.DataFrame({
        "Open":  [10],
        "High":  [10],
        "Low":   [10],
        "Close": [10],
    })


# =========================
# Tests for add_candlestick_features
# =========================

def test_add_candlestick_features_adds_expected_columns(basic_ohlc_df):
    result = add_candlestick_features(basic_ohlc_df)

    expected_cols = {
        "Open", "High", "Low", "Close",
        "body", "range", "upper_wick", "lower_wick", "direction"
    }

    assert isinstance(result, pd.DataFrame)
    assert expected_cols.issubset(set(result.columns))


def test_add_candlestick_features_does_not_modify_original_df(basic_ohlc_df):
    original = basic_ohlc_df.copy(deep=True)
    _ = add_candlestick_features(basic_ohlc_df)

    pd.testing.assert_frame_equal(basic_ohlc_df, original)


def test_add_candlestick_features_computes_correct_values():
    df = pd.DataFrame({
        "Open":  [10],
        "High":  [15],
        "Low":   [8],
        "Close": [14],
    })

    result = add_candlestick_features(df)
    row = result.iloc[0]

    assert row["body"] == 4            # |14 - 10|
    assert row["range"] == 7           # 15 - 8
    assert row["upper_wick"] == 1      # 15 - max(10,14)
    assert row["lower_wick"] == 2      # min(10,14) - 8
    assert row["direction"] == "bullish"


def test_add_candlestick_features_direction_bullish():
    df = pd.DataFrame({
        "Open":  [10],
        "High":  [12],
        "Low":   [9],
        "Close": [11],
    })

    result = add_candlestick_features(df)
    assert result.iloc[0]["direction"] == "bullish"


def test_add_candlestick_features_direction_bearish():
    df = pd.DataFrame({
        "Open":  [10],
        "High":  [12],
        "Low":   [8],
        "Close": [9],
    })

    result = add_candlestick_features(df)
    assert result.iloc[0]["direction"] == "bearish"


def test_add_candlestick_features_direction_neutral():
    df = pd.DataFrame({
        "Open":  [10],
        "High":  [12],
        "Low":   [8],
        "Close": [10],
    })

    result = add_candlestick_features(df)
    assert result.iloc[0]["direction"] == "neutral"


def test_add_candlestick_features_zero_range_case(zero_range_df):
    result = add_candlestick_features(zero_range_df)
    row = result.iloc[0]

    assert row["body"] == 0
    assert row["range"] == 0
    assert row["upper_wick"] == 0
    assert row["lower_wick"] == 0
    assert row["direction"] == "neutral"


def test_add_candlestick_features_empty_dataframe(empty_ohlc_df):
    result = add_candlestick_features(empty_ohlc_df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    expected_new_cols = ["body", "range", "upper_wick", "lower_wick", "direction"]
    for col in expected_new_cols:
        assert col in result.columns


def test_add_candlestick_features_missing_required_column_raises():
    df = pd.DataFrame({
        "Open": [10],
        "High": [15],
        "Low": [8],
        # "Close" missing
    })

    with pytest.raises(KeyError):
        add_candlestick_features(df)


def test_add_candlestick_features_with_nan_values():
    df = pd.DataFrame({
        "Open":  [10, np.nan],
        "High":  [15, 16],
        "Low":   [8,  9],
        "Close": [14, 11],
    })

    result = add_candlestick_features(df)

    assert pd.isna(result.iloc[1]["body"])
    assert result.iloc[0]["direction"] == "bullish"
    # close 11 vs open nan -> comparison is False, falls to neutral in np.where chain
    assert result.iloc[1]["direction"] == "neutral"


# =========================
# Tests for safe_ratio
# =========================

def test_safe_ratio_normal_case():
    assert safe_ratio(10, 2) == 5


def test_safe_ratio_zero_denominator():
    assert safe_ratio(10, 0) == 0.0


def test_safe_ratio_zero_numerator():
    assert safe_ratio(0, 5) == 0


def test_safe_ratio_negative_values():
    assert safe_ratio(-10, 2) == -5
    assert safe_ratio(10, -2) == -5
    assert safe_ratio(-10, -2) == 5


def test_safe_ratio_float_values():
    assert safe_ratio(2.5, 0.5) == 5.0


# =========================
# Tests for get_last_candle
# =========================

def test_get_last_candle_returns_last_row(basic_ohlc_df):
    result = get_last_candle(basic_ohlc_df)

    assert isinstance(result, pd.Series)
    assert result["Open"] == 15
    assert result["High"] == 18
    assert result["Low"] == 14
    assert result["Close"] == 15


def test_get_last_candle_single_row(single_row_df):
    result = get_last_candle(single_row_df)

    assert result["Open"] == 10
    assert result["Close"] == 14


def test_get_last_candle_empty_dataframe_raises(empty_ohlc_df):
    with pytest.raises(IndexError):
        get_last_candle(empty_ohlc_df)


# =========================
# Tests for get_last_n_candles
# =========================

def test_get_last_n_candles_normal_case(basic_ohlc_df):
    result = get_last_n_candles(basic_ohlc_df, 2)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert list(result["Open"]) == [12, 15]


def test_get_last_n_candles_n_equals_len(basic_ohlc_df):
    result = get_last_n_candles(basic_ohlc_df, 3)

    pd.testing.assert_frame_equal(result, basic_ohlc_df)


def test_get_last_n_candles_n_greater_than_len(basic_ohlc_df):
    result = get_last_n_candles(basic_ohlc_df, 10)

    pd.testing.assert_frame_equal(result, basic_ohlc_df)


def test_get_last_n_candles_empty_dataframe(empty_ohlc_df):
    result = get_last_n_candles(empty_ohlc_df, 3)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_get_last_n_candles_n_zero_returns_full_dataframe_current_behavior(basic_ohlc_df):
    result = get_last_n_candles(basic_ohlc_df, 0)

    # Current behavior of df.iloc[-0:] == df.iloc[0:]
    pd.testing.assert_frame_equal(result, basic_ohlc_df)


# =========================
# Tests for body_top_ratio
# =========================

def test_body_top_ratio_with_precomputed_range():
    candle = pd.Series({
        "Open": 10,
        "Close": 14,
        "Low": 8,
        "High": 15,
        "range": 7,
    })

    result = body_top_ratio(candle)

    # max(10,14)-8 = 6 ; 6/7
    assert result == pytest.approx(6 / 7)


def test_body_top_ratio_without_precomputed_range():
    candle = pd.Series({
        "Open": 10,
        "Close": 14,
        "Low": 8,
        "High": 15,
    })

    result = body_top_ratio(candle)

    assert result == pytest.approx(6 / 7)


def test_body_top_ratio_zero_range():
    candle = pd.Series({
        "Open": 10,
        "Close": 10,
        "Low": 10,
        "High": 10,
    })

    result = body_top_ratio(candle)
    assert result == 0.0


def test_body_top_ratio_bearish_candle():
    candle = pd.Series({
        "Open": 14,
        "Close": 10,
        "Low": 8,
        "High": 15,
    })

    result = body_top_ratio(candle)

    # body top = max(14,10)=14 -> (14-8)/(15-8)=6/7
    assert result == pytest.approx(6 / 7)


def test_body_top_ratio_missing_required_field_raises():
    candle = pd.Series({
        "Open": 10,
        "Close": 14,
        "Low": 8,
        # High missing if no range provided
    })

    with pytest.raises(KeyError):
        body_top_ratio(candle)


# =========================
# Tests for body_bottom_ratio
# =========================

def test_body_bottom_ratio_with_precomputed_range():
    candle = pd.Series({
        "Open": 10,
        "Close": 14,
        "Low": 8,
        "High": 15,
        "range": 7,
    })

    result = body_bottom_ratio(candle)

    # min(10,14)-8 = 2 ; 2/7
    assert result == pytest.approx(2 / 7)


def test_body_bottom_ratio_without_precomputed_range():
    candle = pd.Series({
        "Open": 10,
        "Close": 14,
        "Low": 8,
        "High": 15,
    })

    result = body_bottom_ratio(candle)

    assert result == pytest.approx(2 / 7)


def test_body_bottom_ratio_zero_range():
    candle = pd.Series({
        "Open": 10,
        "Close": 10,
        "Low": 10,
        "High": 10,
    })

    result = body_bottom_ratio(candle)
    assert result == 0.0


def test_body_bottom_ratio_bearish_candle():
    candle = pd.Series({
        "Open": 14,
        "Close": 10,
        "Low": 8,
        "High": 15,
    })

    result = body_bottom_ratio(candle)

    # body bottom = min(14,10)=10 -> (10-8)/(15-8)=2/7
    assert result == pytest.approx(2 / 7)


def test_body_bottom_ratio_missing_required_field_raises():
    candle = pd.Series({
        "Open": 10,
        "Close": 14,
        "Low": 8,
        # High missing if no range provided
    })

    with pytest.raises(KeyError):
        body_bottom_ratio(candle)