import numpy as np
import pandas as pd
import pytest

import stock_analysis.confidence_models.chart_patterns.flag_model as fm


# Tests for clamp_0_100

def test_clamp_0_100_inside_range():
    assert fm.clamp_0_100(50) == 50.0
    assert fm.clamp_0_100(0) == 0.0
    assert fm.clamp_0_100(100) == 100.0


def test_clamp_0_100_below_zero():
    assert fm.clamp_0_100(-10) == 0.0


def test_clamp_0_100_above_hundred():
    assert fm.clamp_0_100(120) == 100.0


# Tests for flag_score

def test_flag_score_ideal_case_high_score():
    score = fm.flag_score(
        pole_move_pct=0.050,
        range_ratio=0.30,
        slope_strength_ratio=0.15,
        avg_range_ratio=0.55,
        flag_len_ratio=0.70,
        flag_direction_score=1.00,
    )
    assert 90 <= score <= 100


def test_flag_score_bad_case_low_score():
    score = fm.flag_score(
        pole_move_pct=0.005,
        range_ratio=1.50,
        slope_strength_ratio=2.00,
        avg_range_ratio=2.00,
        flag_len_ratio=4.50,
        flag_direction_score=0.00,
    )
    assert 0 <= score <= 20


def test_flag_score_boundary_pole_levels():
    score1 = fm.flag_score(0.040, 0.30, 0.15, 0.55, 0.70, 1.00)
    score2 = fm.flag_score(0.030, 0.30, 0.15, 0.55, 0.70, 1.00)
    score3 = fm.flag_score(0.020, 0.30, 0.15, 0.55, 0.70, 1.00)
    score4 = fm.flag_score(0.015, 0.30, 0.15, 0.55, 0.70, 1.00)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_flag_score_boundary_range_levels():
    score1 = fm.flag_score(0.040, 0.35, 0.15, 0.55, 0.70, 1.00)
    score2 = fm.flag_score(0.040, 0.50, 0.15, 0.55, 0.70, 1.00)
    score3 = fm.flag_score(0.040, 0.70, 0.15, 0.55, 0.70, 1.00)
    score4 = fm.flag_score(0.040, 0.90, 0.15, 0.55, 0.70, 1.00)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_flag_score_boundary_slope_strength_levels():
    score1 = fm.flag_score(0.040, 0.30, 0.20, 0.55, 0.70, 1.00)
    score2 = fm.flag_score(0.040, 0.30, 0.35, 0.55, 0.70, 1.00)
    score3 = fm.flag_score(0.040, 0.30, 0.50, 0.55, 0.70, 1.00)
    score4 = fm.flag_score(0.040, 0.30, 0.70, 0.55, 0.70, 1.00)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_flag_score_boundary_compression_levels():
    score1 = fm.flag_score(0.040, 0.30, 0.15, 0.60, 0.70, 1.00)
    score2 = fm.flag_score(0.040, 0.30, 0.15, 0.80, 0.70, 1.00)
    score3 = fm.flag_score(0.040, 0.30, 0.15, 1.00, 0.70, 1.00)
    score4 = fm.flag_score(0.040, 0.30, 0.15, 1.20, 0.70, 1.00)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_flag_score_boundary_duration_levels():
    score1 = fm.flag_score(0.040, 0.30, 0.15, 0.55, 0.70, 1.00)   # ideal
    score2 = fm.flag_score(0.040, 0.30, 0.15, 0.55, 1.40, 1.00)   # good
    score3 = fm.flag_score(0.040, 0.30, 0.15, 0.55, 1.80, 1.00)   # borderline
    score4 = fm.flag_score(0.040, 0.30, 0.15, 0.55, 3.00, 1.00)   # fallback 25

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_flag_score_boundary_direction_levels():
    score1 = fm.flag_score(0.040, 0.30, 0.15, 0.55, 0.70, 0.95)
    score2 = fm.flag_score(0.040, 0.30, 0.15, 0.55, 0.70, 0.70)
    score3 = fm.flag_score(0.040, 0.30, 0.15, 0.55, 0.70, 0.45)
    score4 = fm.flag_score(0.040, 0.30, 0.15, 0.55, 0.70, 0.20)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


# Tests for _sample

@pytest.mark.parametrize("region", [
    "ideal", "good", "borderline", "weak_pole",
    "wide_flag", "strong_flag_slope", "no_compression",
    "bad_duration", "same_direction_flag", "random", "outside"
])
def test_sample_returns_valid_tuple(region):
    result = fm._sample(region)

    assert isinstance(result, tuple)
    assert len(result) == 6

    pole_move_pct, range_ratio, slope_strength_ratio, avg_range_ratio, flag_len_ratio, flag_direction_score = result

    assert isinstance(pole_move_pct, float)
    assert isinstance(range_ratio, float)
    assert isinstance(slope_strength_ratio, float)
    assert isinstance(avg_range_ratio, float)
    assert isinstance(flag_len_ratio, float)
    assert isinstance(flag_direction_score, float)


def test_sample_unknown_region_falls_to_outside_branch():
    result = fm._sample("unknown_region")
    assert isinstance(result, tuple)
    assert len(result) == 6


def test_sample_ranges_are_reasonable_for_outside():
    pole_move_pct, range_ratio, slope_strength_ratio, avg_range_ratio, flag_len_ratio, flag_direction_score = fm._sample("outside")

    assert 0.000 <= pole_move_pct <= 0.010
    assert 1.00 <= range_ratio <= 2.50
    assert 1.00 <= slope_strength_ratio <= 3.00
    assert 1.20 <= avg_range_ratio <= 2.50
    assert 0.05 <= flag_len_ratio <= 5.00
    assert 0.00 <= flag_direction_score <= 0.20


# Tests for build_dataset

def test_build_dataset_returns_dataframe():
    df = fm.build_dataset()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_build_dataset_has_expected_columns():
    df = fm.build_dataset()
    expected_columns = {
        "pole_move_pct",
        "range_ratio",
        "slope_strength_ratio",
        "avg_range_ratio",
        "flag_len_ratio",
        "flag_direction_score",
        "quality",
        "region",
    }
    assert expected_columns.issubset(set(df.columns))


def test_build_dataset_expected_row_count():
    df = fm.build_dataset()
    expected_count = sum(fm.REGION_COUNTS.values())
    assert len(df) == expected_count


def test_build_dataset_quality_in_valid_range():
    df = fm.build_dataset()
    assert df["quality"].between(0, 100).all()


def test_build_dataset_regions_match_region_counts():
    df = fm.build_dataset()
    counts = df["region"].value_counts().to_dict()

    for region, expected_count in fm.REGION_COUNTS.items():
        assert counts.get(region, 0) == expected_count


# Tests for train

def test_train_returns_expected_outputs():
    df = fm.build_dataset().head(500)

    model, X_test, y_test, y_pred, mae, r2 = fm.train(df)

    assert model is not None
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(mae, float)
    assert isinstance(r2, float)

    assert len(X_test) == len(y_test) == len(y_pred)


def test_train_predictions_reasonable_range():
    df = fm.build_dataset().head(500)

    model, X_test, y_test, y_pred, mae, r2 = fm.train(df)

    assert np.isfinite(y_pred).all()

# Tests for load_model
def test_load_model_file_not_found_raises(monkeypatch):
    monkeypatch.setattr(fm.os.path, "exists", lambda path: False)

    with pytest.raises(FileNotFoundError):
        fm.load_model()


def test_load_model_success(monkeypatch):
    class DummyModel:
        pass

    dummy_model = DummyModel()

    monkeypatch.setattr(fm.os.path, "exists", lambda path: True)
    monkeypatch.setattr(fm.joblib, "load", lambda path: dummy_model)

    model = fm.load_model()
    assert model is dummy_model


# Tests for learned_confidence

def test_learned_confidence_returns_clamped_value():
    class DummyModel:
        def predict(self, x):
            return np.array([85.5])

    score = fm.learned_confidence(
        pole_move_pct=0.030,
        range_ratio=0.45,
        slope_strength_ratio=0.30,
        avg_range_ratio=0.75,
        flag_len_ratio=0.90,
        flag_direction_score=0.85,
        model=DummyModel(),
    )

    assert score == 85.5


def test_learned_confidence_clamps_below_zero():
    class DummyModel:
        def predict(self, x):
            return np.array([-10.0])

    score = fm.learned_confidence(
        pole_move_pct=0.030,
        range_ratio=0.45,
        slope_strength_ratio=0.30,
        avg_range_ratio=0.75,
        flag_len_ratio=0.90,
        flag_direction_score=0.85,
        model=DummyModel(),
    )

    assert score == 0.0


def test_learned_confidence_clamps_above_hundred():
    class DummyModel:
        def predict(self, x):
            return np.array([120.0])

    score = fm.learned_confidence(
        pole_move_pct=0.030,
        range_ratio=0.45,
        slope_strength_ratio=0.30,
        avg_range_ratio=0.75,
        flag_len_ratio=0.90,
        flag_direction_score=0.85,
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

    score = fm.learned_confidence(
        pole_move_pct=0.024,
        range_ratio=0.52,
        slope_strength_ratio=0.42,
        avg_range_ratio=0.88,
        flag_len_ratio=1.10,
        flag_direction_score=0.70,
        model=DummyModel(),
    )

    assert score == 50.0
    assert captured["shape"] == (1, 6)
    assert captured["value"][0, 0] == 0.024
    assert captured["value"][0, 1] == 0.52
    assert captured["value"][0, 2] == 0.42
    assert captured["value"][0, 3] == 0.88
    assert captured["value"][0, 4] == 1.10
    assert captured["value"][0, 5] == 0.70