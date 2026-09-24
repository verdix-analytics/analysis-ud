import numpy as np
import pandas as pd
import pytest

import stock_analysis.confidence_models.chart_patterns.falling_wedge_model as fwm


# Tests for clamp_0_100

def test_clamp_0_100_inside_range():
    assert fwm.clamp_0_100(50) == 50.0
    assert fwm.clamp_0_100(0) == 0.0
    assert fwm.clamp_0_100(100) == 100.0


def test_clamp_0_100_below_zero():
    assert fwm.clamp_0_100(-10) == 0.0


def test_clamp_0_100_above_hundred():
    assert fwm.clamp_0_100(120) == 100.0


# Tests for falling_wedge_score

def test_falling_wedge_score_ideal_case_high_score():
    score = fwm.falling_wedge_score(
        high_neg_pct_slope=0.0012,
        low_neg_pct_slope=0.0005,
        slope_divergence_score=0.95,
        convergence_score=0.95,
        touch_richness_score=0.95,
        touch_balance_ratio=0.95,
    )
    assert 90 <= score <= 100


def test_falling_wedge_score_bad_case_low_score():
    score = fwm.falling_wedge_score(
        high_neg_pct_slope=0.00001,
        low_neg_pct_slope=0.00001,
        slope_divergence_score=0.10,
        convergence_score=0.10,
        touch_richness_score=0.10,
        touch_balance_ratio=0.10,
    )
    assert 0 <= score <= 20


def test_falling_wedge_score_boundary_drop_levels():
    # avg_drop thresholds: 0.0010, 0.0006, 0.0003, 0.00015
    score1 = fwm.falling_wedge_score(0.0010, 0.0010, 0.95, 0.95, 0.95, 0.95)
    score2 = fwm.falling_wedge_score(0.0006, 0.0006, 0.95, 0.95, 0.95, 0.95)
    score3 = fwm.falling_wedge_score(0.0003, 0.0003, 0.95, 0.95, 0.95, 0.95)
    score4 = fwm.falling_wedge_score(0.00015, 0.00015, 0.95, 0.95, 0.95, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_falling_wedge_score_boundary_divergence_levels():
    score1 = fwm.falling_wedge_score(0.0012, 0.0005, 0.95, 0.95, 0.95, 0.95)
    score2 = fwm.falling_wedge_score(0.0012, 0.0005, 0.80, 0.95, 0.95, 0.95)
    score3 = fwm.falling_wedge_score(0.0012, 0.0005, 0.60, 0.95, 0.95, 0.95)
    score4 = fwm.falling_wedge_score(0.0012, 0.0005, 0.40, 0.95, 0.95, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_falling_wedge_score_boundary_convergence_levels():
    score1 = fwm.falling_wedge_score(0.0012, 0.0005, 0.95, 0.95, 0.95, 0.95)
    score2 = fwm.falling_wedge_score(0.0012, 0.0005, 0.95, 0.80, 0.95, 0.95)
    score3 = fwm.falling_wedge_score(0.0012, 0.0005, 0.95, 0.60, 0.95, 0.95)
    score4 = fwm.falling_wedge_score(0.0012, 0.0005, 0.95, 0.40, 0.95, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_falling_wedge_score_boundary_touch_richness_levels():
    score1 = fwm.falling_wedge_score(0.0012, 0.0005, 0.95, 0.95, 0.95, 0.95)
    score2 = fwm.falling_wedge_score(0.0012, 0.0005, 0.95, 0.95, 0.80, 0.95)
    score3 = fwm.falling_wedge_score(0.0012, 0.0005, 0.95, 0.95, 0.60, 0.95)
    score4 = fwm.falling_wedge_score(0.0012, 0.0005, 0.95, 0.95, 0.40, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_falling_wedge_score_boundary_touch_balance_levels():
    score1 = fwm.falling_wedge_score(0.0012, 0.0005, 0.95, 0.95, 0.95, 0.95)
    score2 = fwm.falling_wedge_score(0.0012, 0.0005, 0.95, 0.95, 0.95, 0.80)
    score3 = fwm.falling_wedge_score(0.0012, 0.0005, 0.95, 0.95, 0.95, 0.65)
    score4 = fwm.falling_wedge_score(0.0012, 0.0005, 0.95, 0.95, 0.95, 0.45)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


# Tests for _sample

@pytest.mark.parametrize("region", [
    "ideal", "good", "borderline", "weak_drop",
    "bad_divergence", "non_converging",
    "few_touches", "imbalanced_touches",
    "random", "outside"
])
def test_sample_returns_valid_tuple(region):
    result = fwm._sample(region)

    assert isinstance(result, tuple)
    assert len(result) == 6

    high_neg_pct_slope, low_neg_pct_slope, slope_divergence_score, convergence_score, touch_richness_score, touch_balance_ratio = result

    assert isinstance(high_neg_pct_slope, float)
    assert isinstance(low_neg_pct_slope, float)
    assert isinstance(slope_divergence_score, float)
    assert isinstance(convergence_score, float)
    assert isinstance(touch_richness_score, float)
    assert isinstance(touch_balance_ratio, float)


def test_sample_unknown_region_falls_to_outside_branch():
    result = fwm._sample("unknown_region")
    assert isinstance(result, tuple)
    assert len(result) == 6


def test_sample_ranges_are_reasonable_for_outside():
    high_neg_pct_slope, low_neg_pct_slope, slope_divergence_score, convergence_score, touch_richness_score, touch_balance_ratio = fwm._sample("outside")

    assert 0.0000 <= high_neg_pct_slope <= 0.00005
    assert 0.0000 <= low_neg_pct_slope <= 0.00005
    assert 0.00 <= slope_divergence_score <= 0.25
    assert 0.00 <= convergence_score <= 0.25
    assert 0.00 <= touch_richness_score <= 0.25
    assert 0.00 <= touch_balance_ratio <= 0.25


# Tests for build_dataset

def test_build_dataset_returns_dataframe():
    df = fwm.build_dataset()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_build_dataset_has_expected_columns():
    df = fwm.build_dataset()
    expected_columns = {
        "high_neg_pct_slope",
        "low_neg_pct_slope",
        "slope_divergence_score",
        "convergence_score",
        "touch_richness_score",
        "touch_balance_ratio",
        "quality",
        "region",
    }
    assert expected_columns.issubset(set(df.columns))


def test_build_dataset_expected_row_count():
    df = fwm.build_dataset()
    expected_count = sum(fwm.REGION_COUNTS.values())
    assert len(df) == expected_count


def test_build_dataset_quality_in_valid_range():
    df = fwm.build_dataset()
    assert df["quality"].between(0, 100).all()


def test_build_dataset_regions_match_region_counts():
    df = fwm.build_dataset()
    counts = df["region"].value_counts().to_dict()

    for region, expected_count in fwm.REGION_COUNTS.items():
        assert counts.get(region, 0) == expected_count


# Tests for train

def test_train_returns_expected_outputs():
    df = fwm.build_dataset().head(500)

    model, X_test, y_test, y_pred, mae, r2 = fwm.train(df)

    assert model is not None
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(mae, float)
    assert isinstance(r2, float)

    assert len(X_test) == len(y_test) == len(y_pred)


def test_train_predictions_reasonable_range():
    df = fwm.build_dataset().head(500)

    model, X_test, y_test, y_pred, mae, r2 = fwm.train(df)

    assert np.isfinite(y_pred).all()


# Tests for load_model

def test_load_model_file_not_found_raises(monkeypatch):
    monkeypatch.setattr(fwm.os.path, "exists", lambda path: False)

    with pytest.raises(FileNotFoundError):
        fwm.load_model()


def test_load_model_success(monkeypatch):
    class DummyModel:
        pass

    dummy_model = DummyModel()

    monkeypatch.setattr(fwm.os.path, "exists", lambda path: True)
    monkeypatch.setattr(fwm.joblib, "load", lambda path: dummy_model)

    model = fwm.load_model()
    assert model is dummy_model


# Tests for learned_confidence

def test_learned_confidence_returns_clamped_value():
    class DummyModel:
        def predict(self, x):
            return np.array([85.5])

    score = fwm.learned_confidence(
        high_neg_pct_slope=0.0008,
        low_neg_pct_slope=0.00035,
        slope_divergence_score=0.82,
        convergence_score=0.82,
        touch_richness_score=0.80,
        touch_balance_ratio=0.80,
        model=DummyModel(),
    )

    assert score == 85.5


def test_learned_confidence_clamps_below_zero():
    class DummyModel:
        def predict(self, x):
            return np.array([-10.0])

    score = fwm.learned_confidence(
        high_neg_pct_slope=0.0008,
        low_neg_pct_slope=0.00035,
        slope_divergence_score=0.82,
        convergence_score=0.82,
        touch_richness_score=0.80,
        touch_balance_ratio=0.80,
        model=DummyModel(),
    )

    assert score == 0.0


def test_learned_confidence_clamps_above_hundred():
    class DummyModel:
        def predict(self, x):
            return np.array([120.0])

    score = fwm.learned_confidence(
        high_neg_pct_slope=0.0008,
        low_neg_pct_slope=0.00035,
        slope_divergence_score=0.82,
        convergence_score=0.82,
        touch_richness_score=0.80,
        touch_balance_ratio=0.80,
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

    score = fwm.learned_confidence(
        high_neg_pct_slope=0.00071,
        low_neg_pct_slope=0.00032,
        slope_divergence_score=0.72,
        convergence_score=0.70,
        touch_richness_score=0.68,
        touch_balance_ratio=0.66,
        model=DummyModel(),
    )

    assert score == 50.0
    assert captured["shape"] == (1, 6)
    assert captured["value"][0, 0] == 0.00071
    assert captured["value"][0, 1] == 0.00032
    assert captured["value"][0, 2] == 0.72
    assert captured["value"][0, 3] == 0.70
    assert captured["value"][0, 4] == 0.68
    assert captured["value"][0, 5] == 0.66