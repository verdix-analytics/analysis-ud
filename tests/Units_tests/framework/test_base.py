import pandas as pd
import pytest

from stock_analysis.base import BasePattern
from stock_analysis.models import PatternResult
import stock_analysis.config as config



# Dummy pattern classes for testing
class DummyBullishChartPattern(BasePattern):
    category = "chart"
    expected_pretrend = "uptrend"

    def _run(self, df: pd.DataFrame) -> dict:
        return {
            "confirmed": True,
            "structure_detected": True,
            "confidence": 70.0,
        }


class DummyCandlestickPattern(BasePattern):
    category = "candlestick"
    expected_pretrend = "downtrend"

    def _run(self, df: pd.DataFrame) -> dict:
        return {
            "confirmed": True,
            "structure_detected": True,
            "confidence": 55.0,
        }


class DummyNeutralPattern(BasePattern):
    category = "chart"
    expected_pretrend = None

    def _run(self, df: pd.DataFrame) -> dict:
        return {
            "confirmed": True,
            "structure_detected": True,
            "confidence": 65.0,
        }


class DummyInvalidOutputPattern(BasePattern):
    category = "chart"

    def _run(self, df: pd.DataFrame) -> dict:
        # Missing required fields for PatternResult
        return {}


class DummyExceptionPattern(BasePattern):
    category = "chart"

    def _run(self, df: pd.DataFrame) -> dict:
        raise RuntimeError("simulated failure")


# Fixtures
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Open": [10, 11, 12],
        "High": [11, 12, 13],
        "Low": [9, 10, 11],
        "Close": [10.5, 11.5, 12.5],
    })


@pytest.fixture
def patched_config(monkeypatch):
    monkeypatch.setattr(config, "DETECTION_CONFIDENCE_THRESHOLD", 60.0, raising=False)
    monkeypatch.setattr(config, "TREND_STRENGTH_REFERENCE", 20.0, raising=False)
    monkeypatch.setattr(config, "TREND_MATCH_BONUS", 20.0, raising=False)
    monkeypatch.setattr(config, "TREND_SIDEWAYS_PENALTY", 10.0, raising=False)
    monkeypatch.setattr(config, "TREND_MISMATCH_PENALTY", 15.0, raising=False)


# Tests for simple base behavior
def test_get_expected_pretrend_returns_value():
    pattern = DummyBullishChartPattern()
    assert pattern.get_expected_pretrend() == "uptrend"


def test_get_expected_pretrend_returns_none():
    pattern = DummyNeutralPattern()
    assert pattern.get_expected_pretrend() is None


def test_is_confidence_sufficient_true(patched_config):
    pattern = DummyBullishChartPattern()
    assert pattern.is_confidence_sufficient(60.0) is True
    assert pattern.is_confidence_sufficient(80.0) is True


def test_is_confidence_sufficient_false(patched_config):
    pattern = DummyBullishChartPattern()
    assert pattern.is_confidence_sufficient(59.99) is False


# Tests for _trend_strength_scale

def test_trend_strength_scale_none_input(patched_config):
    pattern = DummyBullishChartPattern()
    assert pattern._trend_strength_scale(None) == 0.0


def test_trend_strength_scale_empty_dict(patched_config):
    pattern = DummyBullishChartPattern()
    assert pattern._trend_strength_scale({}) == 0.0


def test_trend_strength_scale_zero_strength(patched_config):
    pattern = DummyBullishChartPattern()
    assert pattern._trend_strength_scale({"strength": 0}) == 0.0


def test_trend_strength_scale_negative_strength(patched_config):
    pattern = DummyBullishChartPattern()
    assert pattern._trend_strength_scale({"strength": -5}) == 0.0


def test_trend_strength_scale_positive_strength(patched_config):
    pattern = DummyBullishChartPattern()
    result = pattern._trend_strength_scale({"strength": 20})
    # 20 / (20 + 20) = 0.5
    assert result == pytest.approx(0.5)


def test_trend_strength_scale_large_strength_capped_with_formula(patched_config):
    pattern = DummyBullishChartPattern()
    result = pattern._trend_strength_scale({"strength": 1000})
    assert 0.0 <= result <= 1.0
    assert result == pytest.approx(1000 / 1020)


# Tests for _apply_pretrend_adjustment
def test_apply_pretrend_adjustment_no_pretrend_clips_confidence(patched_config):
    pattern = DummyBullishChartPattern()
    result = PatternResult(
        confirmed=True,
        structure_detected=True,
        confidence=120.0,
    )

    adjusted = pattern._apply_pretrend_adjustment(result, pre_trend=None)

    assert adjusted.confidence == 100.0
    assert adjusted.confidence_breakdown["base_score"] == 120.0
    assert adjusted.confidence_metadata == {}


def test_apply_pretrend_adjustment_expected_trend_none_uses_legacy_logic(patched_config):
    pattern = DummyNeutralPattern()
    result = PatternResult(
        confirmed=True,
        structure_detected=True,
        confidence=65.0,
    )

    adjusted = pattern._apply_pretrend_adjustment(
        result,
        pre_trend={"label": "uptrend", "strength": 50},
    )

    assert adjusted.confidence == 65.0
    assert adjusted.confidence_metadata["pretrend_label"] == "uptrend"
    assert adjusted.confidence_metadata["expected_pretrend"] is None
    assert adjusted.confidence_breakdown["base_score"] == 65.0


def test_apply_pretrend_adjustment_unknown_trend_label_uses_legacy_logic(patched_config):
    pattern = DummyBullishChartPattern()
    result = PatternResult(
        confirmed=True,
        structure_detected=True,
        confidence=70.0,
    )

    adjusted = pattern._apply_pretrend_adjustment(
        result,
        pre_trend={"label": "unknown", "strength": 50},
    )

    assert adjusted.confidence == 70.0
    assert adjusted.confidence_metadata["pretrend_label"] == "unknown"
    assert adjusted.confidence_metadata["expected_pretrend"] == "uptrend"


def test_apply_pretrend_adjustment_matching_trend_adds_bonus(patched_config):
    pattern = DummyBullishChartPattern()
    result = PatternResult(
        confirmed=True,
        structure_detected=True,
        confidence=70.0,
    )

    adjusted = pattern._apply_pretrend_adjustment(
        result,
        pre_trend={"label": "uptrend", "strength": 20},
    )

    # scale = 20/(20+20)=0.5 ; bonus = 20 ; adjustment = 10
    assert adjusted.confidence == 80.0
    assert adjusted.confidence_breakdown["pretrend_base_adjustment"] == 20.0
    assert adjusted.confidence_breakdown["pretrend_adjustment"] == 10.0
    assert adjusted.confidence_breakdown["pretrend_strength"] == 20.0
    assert adjusted.confidence_breakdown["pretrend_strength_scale"] == 0.5
    assert adjusted.confidence_breakdown["final_score"] == 80.0
    assert adjusted.confidence_metadata["pretrend_label"] == "uptrend"
    assert adjusted.confidence_metadata["expected_pretrend"] == "uptrend"


def test_apply_pretrend_adjustment_sideways_applies_penalty(patched_config):
    pattern = DummyBullishChartPattern()
    result = PatternResult(
        confirmed=True,
        structure_detected=True,
        confidence=70.0,
    )

    adjusted = pattern._apply_pretrend_adjustment(
        result,
        pre_trend={"label": "sideways", "strength": 20},
    )

    # sideways penalty = -10 * 0.5 = -5
    assert adjusted.confidence == 65.0
    assert adjusted.confidence_breakdown["pretrend_base_adjustment"] == -10.0
    assert adjusted.confidence_breakdown["pretrend_adjustment"] == -5.0


def test_apply_pretrend_adjustment_mismatch_applies_penalty(patched_config):
    pattern = DummyBullishChartPattern()
    result = PatternResult(
        confirmed=True,
        structure_detected=True,
        confidence=70.0,
    )

    adjusted = pattern._apply_pretrend_adjustment(
        result,
        pre_trend={"label": "downtrend", "strength": 20},
    )

    # mismatch penalty = -15 * 0.5 = -7.5
    assert adjusted.confidence == 62.5
    assert adjusted.confidence_breakdown["pretrend_base_adjustment"] == -15.0
    assert adjusted.confidence_breakdown["pretrend_adjustment"] == -7.5


def test_apply_pretrend_adjustment_confidence_clipped_to_zero(patched_config):
    pattern = DummyBullishChartPattern()
    result = PatternResult(
        confirmed=True,
        structure_detected=True,
        confidence=5.0,
    )

    adjusted = pattern._apply_pretrend_adjustment(
        result,
        pre_trend={"label": "downtrend", "strength": 1000},
    )

    assert adjusted.confidence >= 0.0
    assert adjusted.confidence == 0.0


def test_apply_pretrend_adjustment_structure_detected_but_low_confidence_sets_unconfirmed(patched_config):
    pattern = DummyBullishChartPattern()
    result = PatternResult(
        confirmed=True,
        structure_detected=True,
        confidence=50.0,
    )

    adjusted = pattern._apply_pretrend_adjustment(
        result,
        pre_trend={"label": "downtrend", "strength": 20},
    )

    assert adjusted.confidence < config.DETECTION_CONFIDENCE_THRESHOLD
    assert adjusted.confirmed is False


def test_apply_pretrend_adjustment_candlestick_recomputes_confirmation_true(patched_config):
    pattern = DummyCandlestickPattern()
    result = PatternResult(
        confirmed=False,
        structure_detected=True,
        confidence=70.0,
    )

    adjusted = pattern._apply_pretrend_adjustment(
        result,
        pre_trend={"label": "downtrend", "strength": 20},
    )

    assert adjusted.confidence >= config.DETECTION_CONFIDENCE_THRESHOLD
    assert adjusted.confirmed is True


def test_apply_pretrend_adjustment_candlestick_recomputes_confirmation_false_when_no_structure(patched_config):
    pattern = DummyCandlestickPattern()
    result = PatternResult(
        confirmed=True,
        structure_detected=False,
        confidence=90.0,
    )

    adjusted = pattern._apply_pretrend_adjustment(
        result,
        pre_trend={"label": "downtrend", "strength": 20},
    )

    assert adjusted.confirmed is False


def test_apply_pretrend_adjustment_non_candlestick_keeps_true_only_if_sufficient(patched_config):
    pattern = DummyBullishChartPattern()
    result = PatternResult(
        confirmed=True,
        structure_detected=False,
        confidence=55.0,
    )

    adjusted = pattern._apply_pretrend_adjustment(
        result,
        pre_trend={"label": "unknown", "strength": 20},
    )

    assert adjusted.confirmed is False



# Tests for execute
def test_execute_success_without_pretrend(sample_df, patched_config):
    pattern = DummyBullishChartPattern()

    result = pattern.execute(sample_df)

    assert isinstance(result, PatternResult)
    assert result.confirmed is True
    assert result.confidence == 70.0
    assert result.confidence_breakdown["base_score"] == 70.0


def test_execute_success_with_pretrend(sample_df, patched_config):
    pattern = DummyBullishChartPattern()

    result = pattern.execute(
        sample_df,
        pre_trend={"label": "uptrend", "strength": 20},
    )

    assert isinstance(result, PatternResult)
    assert result.confirmed is True
    assert result.confidence == 80.0
    assert result.confidence_metadata["pretrend_label"] == "uptrend"


def test_execute_returns_default_pattern_result_on_validation_error(sample_df, patched_config):
    pattern = DummyInvalidOutputPattern()

    result = pattern.execute(sample_df)

    assert isinstance(result, PatternResult)
    assert result.confirmed is False
    assert result.confidence == 0


def test_execute_returns_default_pattern_result_on_exception(sample_df, patched_config):
    pattern = DummyExceptionPattern()

    result = pattern.execute(sample_df)

    assert isinstance(result, PatternResult)
    assert result.confirmed is False
    assert result.confidence == 0