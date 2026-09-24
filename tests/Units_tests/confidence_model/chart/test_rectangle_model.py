import numpy as np
import pandas as pd
import pytest

import stock_analysis.confidence_models.chart_patterns.rectangle_model as rm


# Tests for clamp_0_100

def test_clamp_0_100_inside_range():
    assert rm.clamp_0_100(50) == 50.0
    assert rm.clamp_0_100(0) == 0.0
    assert rm.clamp_0_100(100) == 100.0


def test_clamp_0_100_below_zero():
    assert rm.clamp_0_100(-10) == 0.0


def test_clamp_0_100_above_hundred():
    assert rm.clamp_0_100(120) == 100.0


# Tests for rectangle_score

def test_rectangle_score_ideal_case_high_score():
    score = rm.rectangle_score(
        high_consistency_score=0.95,
        low_consistency_score=0.95,
        touch_balance_ratio=0.95,
        touch_richness_score=0.95,
        range_reasonable_score=0.90,
        midline_stability_score=0.95,
    )
    assert 90 <= score <= 100


def test_rectangle_score_bad_case_low_score():
    score = rm.rectangle_score(
        high_consistency_score=0.10,
        low_consistency_score=0.10,
        touch_balance_ratio=0.10,
        touch_richness_score=0.10,
        range_reasonable_score=0.10,
        midline_stability_score=0.10,
    )
    assert 0 <= score <= 20


def test_rectangle_score_boundary_high_levels():
    score1 = rm.rectangle_score(0.95, 0.95, 0.95, 0.95, 0.95, 0.95)
    score2 = rm.rectangle_score(0.80, 0.95, 0.95, 0.95, 0.95, 0.95)
    score3 = rm.rectangle_score(0.60, 0.95, 0.95, 0.95, 0.95, 0.95)
    score4 = rm.rectangle_score(0.40, 0.95, 0.95, 0.95, 0.95, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_rectangle_score_boundary_low_levels():
    score1 = rm.rectangle_score(0.95, 0.95, 0.95, 0.95, 0.95, 0.95)
    score2 = rm.rectangle_score(0.95, 0.80, 0.95, 0.95, 0.95, 0.95)
    score3 = rm.rectangle_score(0.95, 0.60, 0.95, 0.95, 0.95, 0.95)
    score4 = rm.rectangle_score(0.95, 0.40, 0.95, 0.95, 0.95, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_rectangle_score_boundary_touch_balance_levels():
    score1 = rm.rectangle_score(0.95, 0.95, 0.95, 0.95, 0.95, 0.95)
    score2 = rm.rectangle_score(0.95, 0.95, 0.80, 0.95, 0.95, 0.95)
    score3 = rm.rectangle_score(0.95, 0.95, 0.65, 0.95, 0.95, 0.95)
    score4 = rm.rectangle_score(0.95, 0.95, 0.45, 0.95, 0.95, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_rectangle_score_boundary_touch_richness_levels():
    score1 = rm.rectangle_score(0.95, 0.95, 0.95, 0.95, 0.95, 0.95)
    score2 = rm.rectangle_score(0.95, 0.95, 0.95, 0.80, 0.95, 0.95)
    score3 = rm.rectangle_score(0.95, 0.95, 0.95, 0.60, 0.95, 0.95)
    score4 = rm.rectangle_score(0.95, 0.95, 0.95, 0.40, 0.95, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_rectangle_score_boundary_range_levels():
    score1 = rm.rectangle_score(0.95, 0.95, 0.95, 0.95, 0.95, 0.95)
    score2 = rm.rectangle_score(0.95, 0.95, 0.95, 0.95, 0.80, 0.95)
    score3 = rm.rectangle_score(0.95, 0.95, 0.95, 0.95, 0.60, 0.95)
    score4 = rm.rectangle_score(0.95, 0.95, 0.95, 0.95, 0.40, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_rectangle_score_boundary_midline_levels():
    score1 = rm.rectangle_score(0.95, 0.95, 0.95, 0.95, 0.95, 0.95)
    score2 = rm.rectangle_score(0.95, 0.95, 0.95, 0.95, 0.95, 0.80)
    score3 = rm.rectangle_score(0.95, 0.95, 0.95, 0.95, 0.95, 0.60)
    score4 = rm.rectangle_score(0.95, 0.95, 0.95, 0.95, 0.95, 0.40)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


# Tests for _sample

@pytest.mark.parametrize("region", [
    "ideal", "good", "borderline", "messy_highs",
    "messy_lows", "imbalanced_touches", "few_touches",
    "bad_range", "drifting_box", "random", "outside"
])
def test_sample_returns_valid_tuple(region):
    result = rm._sample(region)

    assert isinstance(result, tuple)
    assert len(result) == 6

    high_consistency_score, low_consistency_score, touch_balance_ratio, touch_richness_score, range_reasonable_score, midline_stability_score = result

    assert isinstance(high_consistency_score, float)
    assert isinstance(low_consistency_score, float)
    assert isinstance(touch_balance_ratio, float)
    assert isinstance(touch_richness_score, float)
    assert isinstance(range_reasonable_score, float)
    assert isinstance(midline_stability_score, float)


def test_sample_unknown_region_falls_to_outside_branch():
    result = rm._sample("unknown_region")
    assert isinstance(result, tuple)
    assert len(result) == 6


def test_sample_ranges_are_reasonable_for_outside():
    high_consistency_score, low_consistency_score, touch_balance_ratio, touch_richness_score, range_reasonable_score, midline_stability_score = rm._sample("outside")

    assert 0.00 <= high_consistency_score <= 0.25
    assert 0.00 <= low_consistency_score <= 0.25
    assert 0.00 <= touch_balance_ratio <= 0.25
    assert 0.00 <= touch_richness_score <= 0.25
    assert 0.00 <= range_reasonable_score <= 0.25
    assert 0.00 <= midline_stability_score <= 0.25


# Tests for build_dataset

def test_build_dataset_returns_dataframe():
    df = rm.build_dataset()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_build_dataset_has_expected_columns():
    df = rm.build_dataset()
    expected_columns = {
        "high_consistency_score",
        "low_consistency_score",
        "touch_balance_ratio",
        "touch_richness_score",
        "range_reasonable_score",
        "midline_stability_score",
        "quality",
        "region",
    }
    assert expected_columns.issubset(set(df.columns))


def test_build_dataset_expected_row_count():
    df = rm.build_dataset()
    expected_count = sum(rm.REGION_COUNTS.values())
    assert len(df) == expected_count


def test_build_dataset_quality_in_valid_range():
    df = rm.build_dataset()
    assert df["quality"].between(0, 100).all()


def test_build_dataset_regions_match_region_counts():
    df = rm.build_dataset()
    counts = df["region"].value_counts().to_dict()

    for region, expected_count in rm.REGION_COUNTS.items():
        assert counts.get(region, 0) == expected_count


# Tests for train

def test_train_returns_expected_outputs():
    df = rm.build_dataset().head(500)

    model, X_test, y_test, y_pred, mae, r2 = rm.train(df)

    assert model is not None
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(mae, float)
    assert isinstance(r2, float)

    assert len(X_test) == len(y_test) == len(y_pred)


def test_train_predictions_reasonable_range():
    df = rm.build_dataset().head(500)

    model, X_test, y_test, y_pred, mae, r2 = rm.train(df)

    assert np.isfinite(y_pred).all()


# Tests for load_model

def test_load_model_file_not_found_raises(monkeypatch):
    monkeypatch.setattr(rm.os.path, "exists", lambda path: False)

    with pytest.raises(FileNotFoundError):
        rm.load_model()


def test_load_model_success(monkeypatch):
    class DummyModel:
        pass

    dummy_model = DummyModel()

    monkeypatch.setattr(rm.os.path, "exists", lambda path: True)
    monkeypatch.setattr(rm.joblib, "load", lambda path: dummy_model)

    model = rm.load_model()
    assert model is dummy_model


# Tests for learned_confidence

def test_learned_confidence_returns_clamped_value():
    class DummyModel:
        def predict(self, x):
            return np.array([85.5])

    score = rm.learned_confidence(
        high_consistency_score=0.82,
        low_consistency_score=0.80,
        touch_balance_ratio=0.80,
        touch_richness_score=0.78,
        range_reasonable_score=0.80,
        midline_stability_score=0.82,
        model=DummyModel(),
    )

    assert score == 85.5


def test_learned_confidence_clamps_below_zero():
    class DummyModel:
        def predict(self, x):
            return np.array([-10.0])

    score = rm.learned_confidence(
        high_consistency_score=0.82,
        low_consistency_score=0.80,
        touch_balance_ratio=0.80,
        touch_richness_score=0.78,
        range_reasonable_score=0.80,
        midline_stability_score=0.82,
        model=DummyModel(),
    )

    assert score == 0.0


def test_learned_confidence_clamps_above_hundred():
    class DummyModel:
        def predict(self, x):
            return np.array([120.0])

    score = rm.learned_confidence(
        high_consistency_score=0.82,
        low_consistency_score=0.80,
        touch_balance_ratio=0.80,
        touch_richness_score=0.78,
        range_reasonable_score=0.80,
        midline_stability_score=0.82,
        model=DummyModel(),
    )

    assert score == 100.0


def test_learned_confidence_calls_model_with_correct_shape():
    captured = {}

    class DummyModel:
        def predict(self, x):
            captured["shape"] = x.shape
            captured["value"] = x
            return np.array([50.0])

    score = rm.learned_confidence(
        high_consistency_score=0.72,
        low_consistency_score=0.70,
        touch_balance_ratio=0.68,
        touch_richness_score=0.66,
        range_reasonable_score=0.74,
        midline_stability_score=0.62,
        model=DummyModel(),
    )

    assert score == 50.0
    assert captured["shape"] == (1, 6)
    assert captured["value"][0, 0] == 0.72
    assert captured["value"][0, 1] == 0.70
    assert captured["value"][0, 2] == 0.68
    assert captured["value"][0, 3] == 0.66
    assert captured["value"][0, 4] == 0.74
    assert captured["value"][0, 5] == 0.62