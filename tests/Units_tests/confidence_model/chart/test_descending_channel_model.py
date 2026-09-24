import numpy as np
import pandas as pd
import pytest

import stock_analysis.confidence_models.chart_patterns.descending_channel_model as dcm


# Tests for clamp_0_100

def test_clamp_0_100_inside_range():
    assert dcm.clamp_0_100(50) == 50.0
    assert dcm.clamp_0_100(0) == 0.0
    assert dcm.clamp_0_100(100) == 100.0


def test_clamp_0_100_below_zero():
    assert dcm.clamp_0_100(-10) == 0.0


def test_clamp_0_100_above_hundred():
    assert dcm.clamp_0_100(120) == 100.0


# Tests for descending_channel_score

def test_descending_channel_score_ideal_case_high_score():
    score = dcm.descending_channel_score(
        upper_neg_pct_slope=0.0012,
        lower_neg_pct_slope=0.0011,
        slope_parallel_score=0.95,
        touch_balance_ratio=0.95,
        width_stability_score=0.95,
        touch_richness_score=0.95,
    )
    assert 90 <= score <= 100


def test_descending_channel_score_bad_case_low_score():
    score = dcm.descending_channel_score(
        upper_neg_pct_slope=0.00001,
        lower_neg_pct_slope=0.00001,
        slope_parallel_score=0.10,
        touch_balance_ratio=0.10,
        width_stability_score=0.10,
        touch_richness_score=0.10,
    )
    assert 0 <= score <= 20


def test_descending_channel_score_boundary_drop_levels():
    # avg_drop thresholds: 0.0010, 0.0006, 0.0003, 0.00015
    score1 = dcm.descending_channel_score(0.0010, 0.0010, 0.95, 0.95, 0.95, 0.95)
    score2 = dcm.descending_channel_score(0.0006, 0.0006, 0.95, 0.95, 0.95, 0.95)
    score3 = dcm.descending_channel_score(0.0003, 0.0003, 0.95, 0.95, 0.95, 0.95)
    score4 = dcm.descending_channel_score(0.00015, 0.00015, 0.95, 0.95, 0.95, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_descending_channel_score_boundary_parallel_levels():
    score1 = dcm.descending_channel_score(0.0010, 0.0010, 0.95, 0.95, 0.95, 0.95)
    score2 = dcm.descending_channel_score(0.0010, 0.0010, 0.80, 0.95, 0.95, 0.95)
    score3 = dcm.descending_channel_score(0.0010, 0.0010, 0.60, 0.95, 0.95, 0.95)
    score4 = dcm.descending_channel_score(0.0010, 0.0010, 0.40, 0.95, 0.95, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_descending_channel_score_boundary_touch_balance_levels():
    score1 = dcm.descending_channel_score(0.0010, 0.0010, 0.95, 0.95, 0.95, 0.95)
    score2 = dcm.descending_channel_score(0.0010, 0.0010, 0.95, 0.80, 0.95, 0.95)
    score3 = dcm.descending_channel_score(0.0010, 0.0010, 0.95, 0.65, 0.95, 0.95)
    score4 = dcm.descending_channel_score(0.0010, 0.0010, 0.95, 0.45, 0.95, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_descending_channel_score_boundary_width_levels():
    score1 = dcm.descending_channel_score(0.0010, 0.0010, 0.95, 0.95, 0.95, 0.95)
    score2 = dcm.descending_channel_score(0.0010, 0.0010, 0.95, 0.95, 0.80, 0.95)
    score3 = dcm.descending_channel_score(0.0010, 0.0010, 0.95, 0.95, 0.60, 0.95)
    score4 = dcm.descending_channel_score(0.0010, 0.0010, 0.95, 0.95, 0.40, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_descending_channel_score_boundary_touch_richness_levels():
    score1 = dcm.descending_channel_score(0.0010, 0.0010, 0.95, 0.95, 0.95, 0.95)
    score2 = dcm.descending_channel_score(0.0010, 0.0010, 0.95, 0.95, 0.95, 0.80)
    score3 = dcm.descending_channel_score(0.0010, 0.0010, 0.95, 0.95, 0.95, 0.60)
    score4 = dcm.descending_channel_score(0.0010, 0.0010, 0.95, 0.95, 0.95, 0.40)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


# Tests for _sample

@pytest.mark.parametrize("region", [
    "ideal", "good", "borderline", "weak_drop",
    "non_parallel", "imbalanced_touches",
    "unstable_width", "few_touches", "random", "outside"
])
def test_sample_returns_valid_tuple(region):
    result = dcm._sample(region)

    assert isinstance(result, tuple)
    assert len(result) == 6

    upper_neg_pct_slope, lower_neg_pct_slope, slope_parallel_score, touch_balance_ratio, width_stability_score, touch_richness_score = result

    assert isinstance(upper_neg_pct_slope, float)
    assert isinstance(lower_neg_pct_slope, float)
    assert isinstance(slope_parallel_score, float)
    assert isinstance(touch_balance_ratio, float)
    assert isinstance(width_stability_score, float)
    assert isinstance(touch_richness_score, float)


def test_sample_unknown_region_falls_to_outside_branch():
    result = dcm._sample("unknown_region")
    assert isinstance(result, tuple)
    assert len(result) == 6


def test_sample_ranges_are_reasonable_for_outside():
    upper_neg_pct_slope, lower_neg_pct_slope, slope_parallel_score, touch_balance_ratio, width_stability_score, touch_richness_score = dcm._sample("outside")

    assert 0.0000 <= upper_neg_pct_slope <= 0.00005
    assert 0.0000 <= lower_neg_pct_slope <= 0.00005
    assert 0.00 <= slope_parallel_score <= 0.25
    assert 0.00 <= touch_balance_ratio <= 0.25
    assert 0.00 <= width_stability_score <= 0.25
    assert 0.00 <= touch_richness_score <= 0.25


# Tests for build_dataset

def test_build_dataset_returns_dataframe():
    df = dcm.build_dataset()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_build_dataset_has_expected_columns():
    df = dcm.build_dataset()
    expected_columns = {
        "upper_neg_pct_slope",
        "lower_neg_pct_slope",
        "slope_parallel_score",
        "touch_balance_ratio",
        "width_stability_score",
        "touch_richness_score",
        "quality",
        "region",
    }
    assert expected_columns.issubset(set(df.columns))


def test_build_dataset_expected_row_count():
    df = dcm.build_dataset()
    expected_count = sum(dcm.REGION_COUNTS.values())
    assert len(df) == expected_count


def test_build_dataset_quality_in_valid_range():
    df = dcm.build_dataset()
    assert df["quality"].between(0, 100).all()


def test_build_dataset_regions_match_region_counts():
    df = dcm.build_dataset()
    counts = df["region"].value_counts().to_dict()

    for region, expected_count in dcm.REGION_COUNTS.items():
        assert counts.get(region, 0) == expected_count


# Tests for train

def test_train_returns_expected_outputs():
    df = dcm.build_dataset().head(500)

    model, X_test, y_test, y_pred, mae, r2 = dcm.train(df)

    assert model is not None
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(mae, float)
    assert isinstance(r2, float)

    assert len(X_test) == len(y_test) == len(y_pred)


def test_train_predictions_reasonable_range():
    df = dcm.build_dataset().head(500)

    model, X_test, y_test, y_pred, mae, r2 = dcm.train(df)

    assert np.isfinite(y_pred).all()


# Tests for load_model

def test_load_model_file_not_found_raises(monkeypatch):
    monkeypatch.setattr(dcm.os.path, "exists", lambda path: False)

    with pytest.raises(FileNotFoundError):
        dcm.load_model()


def test_load_model_success(monkeypatch):
    class DummyModel:
        pass

    dummy_model = DummyModel()

    monkeypatch.setattr(dcm.os.path, "exists", lambda path: True)
    monkeypatch.setattr(dcm.joblib, "load", lambda path: dummy_model)

    model = dcm.load_model()
    assert model is dummy_model


# Tests for learned_confidence

def test_learned_confidence_returns_clamped_value():
    class DummyModel:
        def predict(self, x):
            return np.array([85.5])

    score = dcm.learned_confidence(
        upper_neg_pct_slope=0.0007,
        lower_neg_pct_slope=0.00065,
        slope_parallel_score=0.82,
        touch_balance_ratio=0.80,
        width_stability_score=0.82,
        touch_richness_score=0.80,
        model=DummyModel(),
    )

    assert score == 85.5


def test_learned_confidence_clamps_below_zero():
    class DummyModel:
        def predict(self, x):
            return np.array([-10.0])

    score = dcm.learned_confidence(
        upper_neg_pct_slope=0.0007,
        lower_neg_pct_slope=0.00065,
        slope_parallel_score=0.82,
        touch_balance_ratio=0.80,
        width_stability_score=0.82,
        touch_richness_score=0.80,
        model=DummyModel(),
    )

    assert score == 0.0


def test_learned_confidence_clamps_above_hundred():
    class DummyModel:
        def predict(self, x):
            return np.array([120.0])

    score = dcm.learned_confidence(
        upper_neg_pct_slope=0.0007,
        lower_neg_pct_slope=0.00065,
        slope_parallel_score=0.82,
        touch_balance_ratio=0.80,
        width_stability_score=0.82,
        touch_richness_score=0.80,
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

    score = dcm.learned_confidence(
        upper_neg_pct_slope=0.00052,
        lower_neg_pct_slope=0.00049,
        slope_parallel_score=0.72,
        touch_balance_ratio=0.68,
        width_stability_score=0.74,
        touch_richness_score=0.66,
        model=DummyModel(),
    )

    assert score == 50.0
    assert captured["shape"] == (1, 6)
    assert captured["value"][0, 0] == 0.00052
    assert captured["value"][0, 1] == 0.00049
    assert captured["value"][0, 2] == 0.72
    assert captured["value"][0, 3] == 0.68
    assert captured["value"][0, 4] == 0.74
    assert captured["value"][0, 5] == 0.66