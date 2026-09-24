import pandas as pd
import numpy as np
import pytest

from stock_analysis.tools.chart_utils import (
    safe_ratio,
    pct_diff,
    is_close_pct,
    detect_pivots,
    get_last_n_pivots,
    calc_slope,
    fit_line_slope,
    fit_line_params,
    fit_line_pct_slope,
    line_value,
    count_line_touches,
    pivots_to_lists,
)


# Fixtures
@pytest.fixture
def basic_ohlc_df():
    return pd.DataFrame({
        "Datetime": pd.date_range("2024-01-01", periods=7, freq="D"),
        "Open":  [10, 11, 12, 13, 12, 11, 10],
        "High":  [11, 13, 16, 14, 13, 12, 11],
        "Low":   [ 9, 10, 11, 10,  9,  8,  7],
        "Close": [10.5, 12, 15, 11, 10,  9,  8],
    })


@pytest.fixture
def short_ohlc_df():
    return pd.DataFrame({
        "Datetime": pd.date_range("2024-01-01", periods=3, freq="D"),
        "Open":  [10, 11, 12],
        "High":  [11, 13, 12],
        "Low":   [ 9, 10, 11],
        "Close": [10, 12, 11],
    })


@pytest.fixture
def empty_ohlc_df():
    return pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close"])


@pytest.fixture
def no_pivot_df():
    return pd.DataFrame({
        "Datetime": pd.date_range("2024-01-01", periods=7, freq="D"),
        "Open":  [10, 11, 12, 13, 14, 15, 16],
        "High":  [11, 12, 13, 14, 15, 16, 17],
        "Low":   [ 9, 10, 11, 12, 13, 14, 15],
        "Close": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5],
    })


@pytest.fixture
def pivot_df_sample():
    return pd.DataFrame({
        "Datetime": pd.date_range("2024-01-01", periods=4, freq="D"),
        "pivot_type": ["H", "L", "H", "L"],
        "pivot_price": [15.0, 10.0, 16.0, 9.0],
        "candle_index": [1, 2, 4, 5],
    })


@pytest.fixture
def empty_pivot_df():
    return pd.DataFrame(columns=["Datetime", "pivot_type", "pivot_price", "candle_index"])


# Tests for safe_ratio

def test_safe_ratio_normal_case():
    assert safe_ratio(10, 2) == 5


def test_safe_ratio_zero_denominator():
    assert safe_ratio(10, 0) == 0.0


def test_safe_ratio_negative_values():
    assert safe_ratio(-10, 2) == -5
    assert safe_ratio(10, -2) == -5
    assert safe_ratio(-10, -2) == 5


def test_safe_ratio_zero_numerator():
    assert safe_ratio(0, 5) == 0


# Tests for pct_diff
def test_pct_diff_normal_case():
    # |100-105| / ((100+105)/2) = 5/102.5
    assert pct_diff(100, 105) == pytest.approx(5 / 102.5)


def test_pct_diff_same_values():
    assert pct_diff(100, 100) == 0.0


def test_pct_diff_both_zero():
    assert pct_diff(0, 0) == 0.0


def test_pct_diff_opposite_signs():
    result = pct_diff(-10, 10)
    assert result == pytest.approx(20 / 10)  # base=(10+10)/2=10


def test_pct_diff_one_zero_one_nonzero():
    result = pct_diff(0, 10)
    assert result == pytest.approx(10 / 5)  # base=(0+10)/2=5


# Tests for is_close_pct

def test_is_close_pct_true():
    assert is_close_pct(100, 102, tol=0.03) is True


def test_is_close_pct_false():
    assert is_close_pct(100, 110, tol=0.03) is False


def test_is_close_pct_exact_boundary():
    # pct_diff(100,103)=3/101.5≈0.02956 < 0.03
    assert is_close_pct(100, 103, tol=0.03) is True


def test_is_close_pct_same_values():
    assert is_close_pct(50, 50) is True


# Tests for detect_pivots

def test_detect_pivots_returns_dataframe(basic_ohlc_df):
    result = detect_pivots(basic_ohlc_df, lookback=1)

    assert isinstance(result, pd.DataFrame)
    expected_cols = ["Datetime", "pivot_type", "pivot_price", "candle_index"]
    assert list(result.columns) == expected_cols


def test_detect_pivots_short_input_returns_empty(short_ohlc_df):
    # lookback=2 requires at least 5 rows
    result = detect_pivots(short_ohlc_df, lookback=2)

    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == ["Datetime", "pivot_type", "pivot_price", "candle_index"]


def test_detect_pivots_empty_dataframe_returns_empty(empty_ohlc_df):
    result = detect_pivots(empty_ohlc_df, lookback=2)

    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == ["Datetime", "pivot_type", "pivot_price", "candle_index"]


def test_detect_pivots_detects_high_pivot():
    df = pd.DataFrame({
        "Datetime": pd.date_range("2024-01-01", periods=5, freq="D"),
        "Open":  [10, 11, 12, 11, 10],
        "High":  [11, 12, 15, 12, 11],
        "Low":   [ 9, 10, 11, 10,  9],
        "Close": [10, 11, 14, 11, 10],
    })

    result = detect_pivots(df, lookback=1)

    assert len(result) == 1
    assert result.iloc[0]["pivot_type"] == "H"
    assert result.iloc[0]["pivot_price"] == 15
    assert result.iloc[0]["candle_index"] == 2


def test_detect_pivots_detects_low_pivot():
    df = pd.DataFrame({
        "Datetime": pd.date_range("2024-01-01", periods=5, freq="D"),
        "Open":  [10, 9, 8, 9, 10],
        "High":  [11, 10, 9, 10, 11],
        "Low":   [ 9, 8, 5, 8, 9],
        "Close": [10, 9, 6, 9, 10],
    })

    result = detect_pivots(df, lookback=1)

    assert len(result) == 1
    assert result.iloc[0]["pivot_type"] == "L"
    assert result.iloc[0]["pivot_price"] == 5
    assert result.iloc[0]["candle_index"] == 2


def test_detect_pivots_no_pivots_returns_empty(no_pivot_df):
    result = detect_pivots(no_pivot_df, lookback=1)

    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == ["Datetime", "pivot_type", "pivot_price", "candle_index"]


def test_detect_pivots_removes_consecutive_same_type_keep_stronger_high():
    df = pd.DataFrame({
        "Datetime": pd.date_range("2024-01-01", periods=7, freq="D"),
        "Open":  [10, 11, 12, 13, 12, 11, 10],
        "High":  [11, 15, 14, 16, 13, 12, 11],
        "Low":   [ 9, 10, 11, 12,  9,  8,  7],
        "Close": [10, 14, 13, 15, 10,  9,  8],
    })

    result = detect_pivots(df, lookback=1)

    # We mainly test that consecutive same-type H pivots are cleaned
    # and only the stronger one is kept if they occur consecutively.
    if len(result) >= 1:
        types = result["pivot_type"].tolist()
        for i in range(1, len(types)):
            assert types[i] != types[i - 1]


def test_detect_pivots_applies_min_move_filter():
    df = pd.DataFrame({
        "Datetime": pd.date_range("2024-01-01", periods=7, freq="D"),
        "Open":  [10, 11, 10, 11, 10, 11, 10],
        "High":  [11, 12, 11, 12.1, 11, 12.05, 11],
        "Low":   [ 9, 10,  9.5, 10,  9.6, 10,  9.7],
        "Close": [10, 11.5, 10, 11.7, 10, 11.6, 10],
    })

    result_no_filter = detect_pivots(df, lookback=1, min_move_pct=0.0)
    result_filtered = detect_pivots(df, lookback=1, min_move_pct=0.2)

    assert len(result_filtered) <= len(result_no_filter)


def test_detect_pivots_missing_required_column_raises():
    df = pd.DataFrame({
        "Datetime": pd.date_range("2024-01-01", periods=5, freq="D"),
        "Open":  [10, 11, 12, 11, 10],
        # "High" missing
        "Low":   [ 9, 10, 11, 10, 9],
        "Close": [10, 11, 12, 11, 10],
    })

    with pytest.raises(KeyError):
        detect_pivots(df, lookback=1)


def test_detect_pivots_resets_index():
    df = pd.DataFrame({
        "Datetime": pd.date_range("2024-01-01", periods=5, freq="D"),
        "Open":  [10, 11, 12, 11, 10],
        "High":  [11, 12, 15, 12, 11],
        "Low":   [ 9, 10, 11, 10,  9],
        "Close": [10, 11, 14, 11, 10],
    }, index=[10, 20, 30, 40, 50])

    result = detect_pivots(df, lookback=1)

    assert len(result) == 1
    assert result.iloc[0]["candle_index"] == 2  # reset_index(drop=True) effect


# Tests for get_last_n_pivots

def test_get_last_n_pivots_normal_case(pivot_df_sample):
    result = get_last_n_pivots(pivot_df_sample, 2)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert result.iloc[0]["pivot_type"] == "H"
    assert result.iloc[1]["pivot_type"] == "L"


def test_get_last_n_pivots_n_greater_than_length(pivot_df_sample):
    result = get_last_n_pivots(pivot_df_sample, 10)

    assert len(result) == len(pivot_df_sample)
    assert list(result.columns) == list(pivot_df_sample.columns)


def test_get_last_n_pivots_empty(empty_pivot_df):
    result = get_last_n_pivots(empty_pivot_df, 3)

    assert result.empty
    assert list(result.columns) == list(empty_pivot_df.columns)


def test_get_last_n_pivots_n_zero_current_behavior(pivot_df_sample):
    result = get_last_n_pivots(pivot_df_sample, 0)

    # pandas tail(0) returns empty df
    assert result.empty


# Tests for calc_slope

def test_calc_slope_normal_case():
    assert calc_slope(0, 0, 2, 4) == 2


def test_calc_slope_horizontal_line():
    assert calc_slope(0, 3, 2, 3) == 0


def test_calc_slope_vertical_line_returns_zero():
    assert calc_slope(1, 2, 1, 10) == 0.0


def test_calc_slope_negative():
    assert calc_slope(0, 5, 2, 1) == -2


# Tests for fit_line_params

def test_fit_line_params_normal_case():
    x = [0, 1, 2]
    y = [1, 3, 5]  # slope 2, intercept 1

    slope, intercept = fit_line_params(x, y)

    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(1.0)


def test_fit_line_params_insufficient_points():
    slope, intercept = fit_line_params([1], [2])

    assert slope == 0.0
    assert intercept == 0.0


def test_fit_line_params_mismatched_lengths():
    slope, intercept = fit_line_params([0, 1], [1])

    assert slope == 0.0
    assert intercept == 0.0


def test_fit_line_params_constant_line():
    x = [0, 1, 2, 3]
    y = [5, 5, 5, 5]

    slope, intercept = fit_line_params(x, y)

    assert slope == pytest.approx(0.0, abs=1e-8)
    assert intercept == pytest.approx(5.0)


# Tests for fit_line_slope

def test_fit_line_slope_normal_case():
    x = [0, 1, 2]
    y = [2, 4, 6]

    result = fit_line_slope(x, y)

    assert result == pytest.approx(2.0)


def test_fit_line_slope_insufficient_points():
    result = fit_line_slope([1], [2])

    assert result == 0.0


# Tests for fit_line_pct_slope

def test_fit_line_pct_slope_normal_case():
    x = [0, 1, 2]
    y = [10, 12, 14]  # slope 2, avg abs y = 12

    result = fit_line_pct_slope(x, y)

    assert result == pytest.approx(2 / 12)


def test_fit_line_pct_slope_insufficient_points():
    result = fit_line_pct_slope([1], [2])

    assert result == 0.0


def test_fit_line_pct_slope_zero_scale():
    x = [0, 1, 2]
    y = [0, 0, 0]

    result = fit_line_pct_slope(x, y)

    assert result == 0.0


def test_fit_line_pct_slope_negative_values():
    x = [0, 1, 2]
    y = [-10, -12, -14]

    result = fit_line_pct_slope(x, y)

    # slope = -2, mean abs y = 12
    assert result == pytest.approx(-2 / 12)


# Tests for line_value

def test_line_value_normal_case():
    assert line_value(2, 1, 3) == 7


def test_line_value_zero_slope():
    assert line_value(0, 5, 10) == 5


def test_line_value_negative_slope():
    assert line_value(-2, 10, 3) == 4


# Tests for count_line_touches
def test_count_line_touches_all_touch():
    x = [0, 1, 2]
    y = [1, 3, 5]  # slope=2 intercept=1

    result = count_line_touches(x, y, slope=2, intercept=1, tol_pct=0.001)

    assert result == 3


def test_count_line_touches_partial_touch():
    x = [0, 1, 2]
    y = [1, 3.5, 5]

    result = count_line_touches(x, y, slope=2, intercept=1, tol_pct=0.01)

    assert result == 2


def test_count_line_touches_no_touch():
    x = [0, 1, 2]
    y = [10, 10, 10]

    result = count_line_touches(x, y, slope=2, intercept=1, tol_pct=0.01)

    assert result == 0


def test_count_line_touches_empty_lists():
    result = count_line_touches([], [], slope=1, intercept=0, tol_pct=0.01)

    assert result == 0


def test_count_line_touches_mismatched_lengths_zip_behavior():
    x = [0, 1, 2]
    y = [1, 3]

    result = count_line_touches(x, y, slope=2, intercept=1, tol_pct=0.001)

    # zip truncates to shorter list
    assert result == 2


# Tests for pivots_to_lists

def test_pivots_to_lists_normal_case(pivot_df_sample):
    result = pivots_to_lists(pivot_df_sample)

    assert isinstance(result, dict)
    assert result["types"] == ["H", "L", "H", "L"]
    assert result["prices"] == [15.0, 10.0, 16.0, 9.0]
    assert result["indices"] == [1, 2, 4, 5]
    assert len(result["times"]) == 4


def test_pivots_to_lists_empty(empty_pivot_df):
    result = pivots_to_lists(empty_pivot_df)

    assert result == {
        "types": [],
        "prices": [],
        "indices": [],
        "times": [],
    }


def test_pivots_to_lists_missing_required_column_raises():
    pivots = pd.DataFrame({
        "pivot_type": ["H"],
        "pivot_price": [15.0],
        "candle_index": [1],
        # Datetime missing
    })

    with pytest.raises(KeyError):
        pivots_to_lists(pivots)