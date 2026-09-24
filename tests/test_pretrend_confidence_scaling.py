import pandas as pd

from stock_analysis.base import BasePattern


class DummyCandlestickPattern(BasePattern):
    category = "candlestick"
    expected_pretrend = "uptrend"

    def _run(self, df: pd.DataFrame) -> dict:
        return {
            "confirmed": True,
            "structure_detected": True,
            "confidence": 58.0,
            "confidence_breakdown": {},
        }


class DummyNeutralPattern(BasePattern):
    category = "chart"
    expected_pretrend = None

    def _run(self, df: pd.DataFrame) -> dict:
        return {
            "confirmed": True,
            "structure_detected": True,
            "confidence": 10.0,
            "confidence_breakdown": {},
        }


def _dummy_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Open": 100.0, "High": 101.0, "Low": 99.5, "Close": 100.8},
            {"Open": 100.8, "High": 101.4, "Low": 100.2, "Close": 101.0},
            {"Open": 101.0, "High": 101.6, "Low": 100.7, "Close": 101.2},
        ]
    )


def test_trend_strength_scales_match_bonus():
    pattern = DummyCandlestickPattern()
    df = _dummy_df()

    weak = pattern.execute(df, pre_trend={"label": "uptrend", "strength": 2.0})
    strong = pattern.execute(df, pre_trend={"label": "uptrend", "strength": 24.0})

    assert strong.confidence > weak.confidence
    assert strong.confidence_breakdown["pretrend_adjustment"] > weak.confidence_breakdown["pretrend_adjustment"]


def test_trend_strength_scales_mismatch_penalty():
    pattern = DummyCandlestickPattern()
    df = _dummy_df()

    weak = pattern.execute(df, pre_trend={"label": "downtrend", "strength": 2.0})
    strong = pattern.execute(df, pre_trend={"label": "downtrend", "strength": 24.0})

    assert strong.confidence < weak.confidence
    assert abs(strong.confidence_breakdown["pretrend_adjustment"]) > abs(
        weak.confidence_breakdown["pretrend_adjustment"]
    )


def test_confidence_remains_on_0_to_100_scale():
    pattern = DummyCandlestickPattern()
    df = _dummy_df()

    strong_match = pattern.execute(df, pre_trend={"label": "uptrend", "strength": 1000.0})
    strong_mismatch = pattern.execute(df, pre_trend={"label": "downtrend", "strength": 1000.0})

    assert 0 <= strong_match.confidence <= 100
    assert 0 <= strong_mismatch.confidence <= 100


def test_candlestick_confirmation_depends_on_adjusted_score():
    pattern = DummyCandlestickPattern()
    df = _dummy_df()

    weak = pattern.execute(df, pre_trend={"label": "uptrend", "strength": 1.0})
    strong = pattern.execute(df, pre_trend={"label": "uptrend", "strength": 24.0})

    assert weak.structure_detected is True
    assert weak.confirmed is False
    assert strong.confirmed is True


def test_confidence_breakdown_is_numeric_and_labels_are_metadata():
    pattern = DummyCandlestickPattern()
    df = _dummy_df()

    result = pattern.execute(df, pre_trend={"label": "uptrend", "strength": 10.0})

    assert all(isinstance(value, float) for value in result.confidence_breakdown.values())
    assert result.confidence_metadata["pretrend_label"] == "uptrend"
    assert result.confidence_metadata["expected_pretrend"] == "uptrend"


def test_neutral_patterns_do_not_apply_threshold_logic_from_pretrend():
    pattern = DummyNeutralPattern()
    df = _dummy_df()

    result = pattern.execute(df, pre_trend={"label": "downtrend", "strength": 50.0})

    assert result.confirmed is True
    assert result.confidence == 10.0
    assert result.confidence_breakdown == {"base_score": 10.0}


def test_unknown_pretrend_label_does_not_override_confirmation():
    pattern = DummyCandlestickPattern()
    df = _dummy_df()

    result = pattern.execute(df, pre_trend={"label": "unknown", "strength": 50.0})

    assert result.confirmed is True
    assert result.confidence == 58.0
    assert result.confidence_breakdown == {"base_score": 58.0}
