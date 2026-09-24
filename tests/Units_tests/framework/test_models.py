import pytest
from pydantic import ValidationError

from stock_analysis.models import PatternResult


# =========================================================
# Tests for valid construction
# =========================================================

def test_pattern_result_minimal_valid():
    result = PatternResult(
        confirmed=True,
        confidence=80.0,
    )

    assert result.confirmed is True
    assert result.confidence == 80.0
    assert result.confidence_breakdown is None
    assert result.confidence_metadata is None
    assert result.pivots is None
    assert result.direction == "neutral"


def test_pattern_result_with_all_fields():
    result = PatternResult(
        confirmed=False,
        confidence=55.5,
        confidence_breakdown={"base_score": 55.5, "adjustment": -5.0},
        confidence_metadata={"pretrend_label": "uptrend", "expected_pretrend": "downtrend"},
        pivots=[{"pivot_type": "H", "pivot_price": 100.0}],
        direction="bullish",
    )

    assert result.confirmed is False
    assert result.confidence == 55.5
    assert result.confidence_breakdown == {"base_score": 55.5, "adjustment": -5.0}
    assert result.confidence_metadata == {
        "pretrend_label": "uptrend",
        "expected_pretrend": "downtrend",
    }
    assert result.pivots == [{"pivot_type": "H", "pivot_price": 100.0}]
    assert result.direction == "bullish"


# =========================================================
# Tests for confidence bounds
# =========================================================

def test_pattern_result_confidence_lower_bound():
    result = PatternResult(
        confirmed=True,
        confidence=0.0,
    )
    assert result.confidence == 0.0


def test_pattern_result_confidence_upper_bound():
    result = PatternResult(
        confirmed=True,
        confidence=100.0,
    )
    assert result.confidence == 100.0


def test_pattern_result_confidence_below_zero_raises():
    with pytest.raises(ValidationError):
        PatternResult(
            confirmed=True,
            confidence=-0.01,
        )


def test_pattern_result_confidence_above_hundred_raises():
    with pytest.raises(ValidationError):
        PatternResult(
            confirmed=True,
            confidence=100.01,
        )


# =========================================================
# Tests for required fields
# =========================================================

def test_pattern_result_missing_confirmed_raises():
    with pytest.raises(ValidationError):
        PatternResult(
            confidence=50.0,
        )


def test_pattern_result_missing_confidence_uses_default():
    result = PatternResult(
        confirmed=True,
    )

    assert result.confirmed is True
    assert result.confidence == 0


# =========================================================
# Tests for optional fields
# =========================================================

def test_pattern_result_confidence_breakdown_accepts_none():
    result = PatternResult(
        confirmed=True,
        confidence=50.0,
        confidence_breakdown=None,
    )

    assert result.confidence_breakdown is None


def test_pattern_result_confidence_breakdown_accepts_valid_dict():
    result = PatternResult(
        confirmed=True,
        confidence=50.0,
        confidence_breakdown={"base_score": 50.0, "bonus": 10.0},
    )

    assert result.confidence_breakdown == {"base_score": 50.0, "bonus": 10.0}


def test_pattern_result_confidence_breakdown_invalid_value_type_raises():
    with pytest.raises(ValidationError):
        PatternResult(
            confirmed=True,
            confidence=50.0,
            confidence_breakdown={"base_score": "high"},
        )


def test_pattern_result_confidence_metadata_accepts_none():
    result = PatternResult(
        confirmed=True,
        confidence=50.0,
        confidence_metadata=None,
    )

    assert result.confidence_metadata is None


def test_pattern_result_confidence_metadata_accepts_valid_dict():
    result = PatternResult(
        confirmed=True,
        confidence=50.0,
        confidence_metadata={"pretrend_label": "uptrend", "expected_pretrend": None},
    )

    assert result.confidence_metadata == {
        "pretrend_label": "uptrend",
        "expected_pretrend": None,
    }


def test_pattern_result_confidence_metadata_invalid_value_type_raises():
    with pytest.raises(ValidationError):
        PatternResult(
            confirmed=True,
            confidence=50.0,
            confidence_metadata={"pretrend_label": 123},
        )


def test_pattern_result_pivots_accepts_none():
    result = PatternResult(
        confirmed=True,
        confidence=50.0,
        pivots=None,
    )

    assert result.pivots is None


def test_pattern_result_pivots_accepts_valid_list_of_dicts():
    pivots = [
        {"pivot_type": "H", "pivot_price": 100.0},
        {"pivot_type": "L", "pivot_price": 90.0},
    ]

    result = PatternResult(
        confirmed=True,
        confidence=50.0,
        pivots=pivots,
    )

    assert result.pivots == pivots


def test_pattern_result_pivots_invalid_type_raises():
    with pytest.raises(ValidationError):
        PatternResult(
            confirmed=True,
            confidence=50.0,
            pivots="not_a_list",
        )


# =========================================================
# Tests for direction
# =========================================================

def test_pattern_result_direction_default():
    result = PatternResult(
        confirmed=True,
        confidence=50.0,
    )

    assert result.direction == "neutral"


def test_pattern_result_direction_custom_value():
    result = PatternResult(
        confirmed=True,
        confidence=50.0,
        direction="bullish",
    )

    assert result.direction == "bullish"


def test_pattern_result_direction_accepts_arbitrary_string():
    result = PatternResult(
        confirmed=True,
        confidence=50.0,
        direction="unknown",
    )

    assert result.direction == "unknown"


# =========================================================
# Tests for type validation / coercion behavior
# =========================================================

def test_pattern_result_confirmed_invalid_type_raises():
    with pytest.raises(ValidationError):
        PatternResult(
            confirmed=None,
            confidence=50.0,
        )


def test_pattern_result_confidence_string_number_is_parsed():
    result = PatternResult(
        confirmed=True,
        confidence="75.5",
    )

    assert result.confidence == 75.5


def test_pattern_result_confirmed_string_true_is_parsed():
    result = PatternResult(
        confirmed="true",
        confidence=50.0,
    )

    assert result.confirmed is True