import pandas as pd
import numpy as np
import pytest

from stock_analysis.tools.harmonic_utils import (
    safe_ratio,
    pct_diff,
    is_close_pct,
    in_range,
    is_fib_match,
    fib_score,
    range_score,
    get_last_n_pivots,
    extract_xabcd_from_pivots,
    extract_recent_xabcd_candidates,
    compute_xabcd_moves,
    compute_xabcd_ratios,
    compute_extended_xabcd_ratios,
    compute_shark_ratios,
    is_valid_xabcd_alternating,
    is_bullish_xabcd,
    is_bearish_xabcd,
    structure_direction,
    is_potential_shark_structure,
    average_scores,
    clip_confidence,
    build_harmonic_context,
    build_extended_harmonic_context,
    build_shark_context,
)


# Fixtures
@pytest.fixture
def pivot_df_5_hlhlh():
    return pd.DataFrame({
        "Datetime": pd.date_range("2024-01-01", periods=5, freq="D"),
        "pivot_type": ["H", "L", "H", "L", "H"],
        "pivot_price": [10.0, 5.0, 8.0, 6.0, 9.0],
        "candle_index": [0, 1, 2, 3, 4],
    })


@pytest.fixture
def pivot_df_6():
    return pd.DataFrame({
        "Datetime": pd.date_range("2024-01-01", periods=6, freq="D"),
        "pivot_type": ["H", "L", "H", "L", "H", "L"],
        "pivot_price": [10.0, 5.0, 8.0, 6.0, 9.0, 7.0],
        "candle_index": [0, 1, 2, 3, 4, 5],
    })


@pytest.fixture
def empty_pivot_df():
    return pd.DataFrame(columns=["Datetime", "pivot_type", "pivot_price", "candle_index"])


@pytest.fixture
def bullish_candidate():
    # XA down, AB up, BC down, CD up
    return {
        "types": ["H", "L", "H", "L", "H"],
        "prices": [10.0, 5.0, 8.0, 6.0, 9.0],
        "indices": [0, 1, 2, 3, 4],
        "times": list(pd.date_range("2024-01-01", periods=5, freq="D")),
    }


@pytest.fixture
def bearish_candidate():
    # XA up, AB down, BC up, CD down
    return {
        "types": ["L", "H", "L", "H", "L"],
        "prices": [5.0, 10.0, 7.0, 9.0, 6.0],
        "indices": [0, 1, 2, 3, 4],
        "times": list(pd.date_range("2024-01-01", periods=5, freq="D")),
    }


@pytest.fixture
def invalid_candidate_non_alternating():
    return {
        "types": ["H", "H", "L", "L", "H"],
        "prices": [10.0, 8.0, 7.0, 6.0, 9.0],
        "indices": [0, 1, 2, 3, 4],
        "times": list(pd.date_range("2024-01-01", periods=5, freq="D")),
    }



# Basic numeric helpers
def test_safe_ratio_normal_case():
    assert safe_ratio(10, 2) == 5


def test_safe_ratio_zero_denominator():
    assert safe_ratio(10, 0) == 0.0


def test_safe_ratio_negative_values():
    assert safe_ratio(-10, 2) == -5
    assert safe_ratio(10, -2) == -5
    assert safe_ratio(-10, -2) == 5


def test_pct_diff_normal_case():
    assert pct_diff(100, 105) == pytest.approx(5 / 102.5)


def test_pct_diff_same_values():
    assert pct_diff(100, 100) == 0.0


def test_pct_diff_both_zero():
    assert pct_diff(0, 0) == 0.0


def test_is_close_pct_true():
    assert is_close_pct(100, 102, tol=0.03) is True


def test_is_close_pct_false():
    assert is_close_pct(100, 110, tol=0.03) is False


def test_in_range_inside():
    assert in_range(5, 1, 10) is True


def test_in_range_on_boundary():
    assert in_range(1, 1, 10) is True
    assert in_range(10, 1, 10) is True


def test_in_range_outside():
    assert in_range(0, 1, 10) is False
    assert in_range(11, 1, 10) is False


# Fibonacci helpers

def test_is_fib_match_true():
    assert is_fib_match(0.62, 0.618, tol=0.05) is True


def test_is_fib_match_false():
    assert is_fib_match(0.75, 0.618, tol=0.05) is False


def test_fib_score_exact_match():
    assert fib_score(0.618, 0.618, tol=0.05) == 1.0


def test_fib_score_inside_tolerance():
    result = fib_score(0.628, 0.618, tol=0.05)
    assert result == pytest.approx(1 - 0.01 / 0.05)


def test_fib_score_outside_tolerance():
    assert fib_score(0.80, 0.618, tol=0.05) == 0.0


def test_fib_score_zero_tolerance_exact_match():
    result = fib_score(0.618, 0.618, tol=0.0)
    assert result == 1.0


def test_range_score_midpoint():
    assert range_score(5, 0, 10) == 1.0


def test_range_score_boundary():
    assert range_score(0, 0, 10) == 0.0
    assert range_score(10, 0, 10) == 0.0


def test_range_score_inside_interval():
    assert range_score(7.5, 0, 10) == pytest.approx(0.5)


def test_range_score_outside_interval():
    assert range_score(11, 0, 10) == 0.0


def test_range_score_zero_width_interval_equal_value():
    assert range_score(5, 5, 5) == 1.0


def test_range_score_zero_width_interval_different_value():
    assert range_score(4, 5, 5) == 0.0


# Pivot extraction helpers

def test_get_last_n_pivots_normal_case(pivot_df_6):
    result = get_last_n_pivots(pivot_df_6, 3)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert result.iloc[0]["candle_index"] == 3


def test_get_last_n_pivots_empty_dataframe(empty_pivot_df):
    result = get_last_n_pivots(empty_pivot_df, 3)

    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == list(empty_pivot_df.columns)


def test_get_last_n_pivots_none_input():
    result = get_last_n_pivots(None, 3)

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_extract_xabcd_from_pivots_normal_case(pivot_df_5_hlhlh):
    result = extract_xabcd_from_pivots(pivot_df_5_hlhlh)

    assert isinstance(result, dict)
    assert result["types"] == ["H", "L", "H", "L", "H"]
    assert result["prices"] == [10.0, 5.0, 8.0, 6.0, 9.0]
    assert result["indices"] == [0, 1, 2, 3, 4]
    assert len(result["times"]) == 5


def test_extract_xabcd_from_pivots_wrong_length_returns_none(pivot_df_6):
    result = extract_xabcd_from_pivots(pivot_df_6)
    assert result is None


def test_extract_xabcd_from_pivots_none_returns_none():
    assert extract_xabcd_from_pivots(None) is None


def test_extract_recent_xabcd_candidates_normal_case(pivot_df_6):
    result = extract_recent_xabcd_candidates(pivot_df_6, max_candidates=10)

    assert isinstance(result, list)
    assert len(result) == 2
    for candidate in result:
        assert isinstance(candidate, dict)
        assert len(candidate["prices"]) == 5


def test_extract_recent_xabcd_candidates_respects_max_candidates():
    pivots = pd.DataFrame({
        "Datetime": pd.date_range("2024-01-01", periods=12, freq="D"),
        "pivot_type": ["H", "L"] * 6,
        "pivot_price": list(range(12)),
        "candle_index": list(range(12)),
    })

    result = extract_recent_xabcd_candidates(pivots, max_candidates=3)
    assert len(result) <= 3


def test_extract_recent_xabcd_candidates_too_short_returns_empty(pivot_df_5_hlhlh):
    short_df = pivot_df_5_hlhlh.iloc[:4]
    result = extract_recent_xabcd_candidates(short_df)
    assert result == []


def test_extract_recent_xabcd_candidates_none_returns_empty():
    assert extract_recent_xabcd_candidates(None) == []


# Move and ratio computation
def test_compute_xabcd_moves_bullish_candidate(bullish_candidate):
    result = compute_xabcd_moves(bullish_candidate)

    assert isinstance(result, dict)
    assert result["XA"] == -5.0
    assert result["AB"] == 3.0
    assert result["BC"] == -2.0
    assert result["CD"] == 3.0
    assert result["XD"] == -1.0

    assert result["XA_len"] == 5.0
    assert result["AB_len"] == 3.0
    assert result["BC_len"] == 2.0
    assert result["CD_len"] == 3.0
    assert result["XD_len"] == 1.0


def test_compute_xabcd_moves_invalid_candidate_returns_none():
    assert compute_xabcd_moves(None) is None
    assert compute_xabcd_moves({}) is None
    assert compute_xabcd_moves({"prices": [1, 2, 3, 4]}) is None


def test_compute_xabcd_ratios_normal_case(bullish_candidate):
    moves = compute_xabcd_moves(bullish_candidate)
    result = compute_xabcd_ratios(moves)

    assert isinstance(result, dict)
    assert result["AB_XA"] == pytest.approx(3 / 5)
    assert result["BC_AB"] == pytest.approx(2 / 3)
    assert result["CD_BC"] == pytest.approx(3 / 2)
    assert result["XD_XA"] == pytest.approx(1 / 5)


def test_compute_xabcd_ratios_none_returns_none():
    assert compute_xabcd_ratios(None) is None


def test_compute_xabcd_ratios_zero_denominator():
    moves = {
        "XA_len": 0.0,
        "AB_len": 3.0,
        "BC_len": 2.0,
        "CD_len": 1.0,
        "XD_len": 4.0,
    }
    result = compute_xabcd_ratios(moves)

    assert result["AB_XA"] == 0.0
    assert result["XD_XA"] == 0.0
    assert result["BC_AB"] == pytest.approx(2 / 3)
    assert result["CD_BC"] == pytest.approx(1 / 2)


def test_compute_extended_xabcd_ratios_normal_case(bullish_candidate):
    moves = compute_xabcd_moves(bullish_candidate)
    result = compute_extended_xabcd_ratios(moves)

    assert isinstance(result, dict)
    assert "AB_XA" in result
    assert "BC_AB" in result
    assert "CD_BC" in result
    assert "XD_XA" in result
    assert "XC_XA" in result
    assert "XD_XC" in result
    assert "AC_XA" in result


def test_compute_extended_xabcd_ratios_none_returns_none():
    assert compute_extended_xabcd_ratios(None) is None


def test_compute_shark_ratios_normal_case(bullish_candidate):
    result = compute_shark_ratios(bullish_candidate)

    assert isinstance(result, dict)
    assert "AB_XA" in result
    assert "BC_AB" in result
    assert "XC_XA" in result
    assert "XD_XC" in result
    assert "XD_XA" in result
    assert "CD_BC" in result


def test_compute_shark_ratios_invalid_candidate_returns_none():
    assert compute_shark_ratios(None) is None
    assert compute_shark_ratios({}) is None


# Structure and direction helpers
def test_is_valid_xabcd_alternating_true_hlhlh(bullish_candidate):
    assert is_valid_xabcd_alternating(bullish_candidate) is True


def test_is_valid_xabcd_alternating_true_lhlhl(bearish_candidate):
    assert is_valid_xabcd_alternating(bearish_candidate) is True


def test_is_valid_xabcd_alternating_false(invalid_candidate_non_alternating):
    assert is_valid_xabcd_alternating(invalid_candidate_non_alternating) is False
    assert is_valid_xabcd_alternating(None) is False
    assert is_valid_xabcd_alternating({}) is False


def test_is_bullish_xabcd_true(bullish_candidate):
    moves = compute_xabcd_moves(bullish_candidate)
    assert is_bullish_xabcd(moves) is True


def test_is_bullish_xabcd_false(bearish_candidate):
    moves = compute_xabcd_moves(bearish_candidate)
    assert is_bullish_xabcd(moves) is False
    assert is_bullish_xabcd(None) is False


def test_is_bearish_xabcd_true(bearish_candidate):
    moves = compute_xabcd_moves(bearish_candidate)
    assert is_bearish_xabcd(moves) is True


def test_is_bearish_xabcd_false(bullish_candidate):
    moves = compute_xabcd_moves(bullish_candidate)
    assert is_bearish_xabcd(moves) is False
    assert is_bearish_xabcd(None) is False


def test_structure_direction_bullish(bullish_candidate):
    moves = compute_xabcd_moves(bullish_candidate)
    assert structure_direction(bullish_candidate, moves) == "bullish"


def test_structure_direction_bearish(bearish_candidate):
    moves = compute_xabcd_moves(bearish_candidate)
    assert structure_direction(bearish_candidate, moves) == "bearish"


def test_structure_direction_unknown_for_invalid_types(invalid_candidate_non_alternating):
    moves = compute_xabcd_moves(invalid_candidate_non_alternating)
    assert structure_direction(invalid_candidate_non_alternating, moves) == "unknown"


def test_structure_direction_unknown_for_inconsistent_moves():
    candidate = {
        "types": ["H", "L", "H", "L", "H"],
        "prices": [10.0, 5.0, 7.0, 8.0, 9.0],  # BC positive, breaks bullish/bearish
        "indices": [0, 1, 2, 3, 4],
        "times": list(pd.date_range("2024-01-01", periods=5, freq="D")),
    }
    moves = compute_xabcd_moves(candidate)
    assert structure_direction(candidate, moves) == "unknown"


def test_is_potential_shark_structure_true_bullish(bullish_candidate):
    assert is_potential_shark_structure(bullish_candidate) is True


def test_is_potential_shark_structure_true_bearish(bearish_candidate):
    assert is_potential_shark_structure(bearish_candidate) is True


def test_is_potential_shark_structure_false_cases(invalid_candidate_non_alternating):
    assert is_potential_shark_structure(None) is False
    assert is_potential_shark_structure(invalid_candidate_non_alternating) is False


# Scoring helpers

def test_average_scores_normal_case():
    assert average_scores([0.5, 1.0, 0.0]) == pytest.approx(0.5)


def test_average_scores_empty():
    assert average_scores([]) == 0.0


def test_clip_confidence_normal_case():
    assert clip_confidence(0.75) == 75.0


def test_clip_confidence_below_zero():
    assert clip_confidence(-0.5) == 0.0


def test_clip_confidence_above_one():
    assert clip_confidence(1.5) == 100.0


# Context builders

def test_build_harmonic_context_normal_case(bullish_candidate):
    result = build_harmonic_context(bullish_candidate)

    assert isinstance(result, dict)
    assert result["candidate"] == bullish_candidate
    assert "moves" in result
    assert "ratios" in result
    assert result["direction"] == "bullish"


def test_build_harmonic_context_none_returns_none():
    assert build_harmonic_context(None) is None


def test_build_extended_harmonic_context_normal_case(bullish_candidate):
    result = build_extended_harmonic_context(bullish_candidate)

    assert isinstance(result, dict)
    assert result["candidate"] == bullish_candidate
    assert "moves" in result
    assert "ratios" in result
    assert result["direction"] == "bullish"
    assert "XC_XA" in result["ratios"]


def test_build_extended_harmonic_context_none_returns_none():
    assert build_extended_harmonic_context(None) is None


def test_build_shark_context_normal_case(bullish_candidate):
    result = build_shark_context(bullish_candidate)

    assert isinstance(result, dict)
    assert result["candidate"] == bullish_candidate
    assert "moves" in result
    assert "ratios" in result
    assert result["direction"] == "bullish"
    assert "XD_XC" in result["ratios"]


def test_build_shark_context_none_returns_none():
    assert build_shark_context(None) is None