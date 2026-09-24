import numpy as np
import pandas as pd
import pytest

import stock_analysis.confidence_models.chart_patterns.cup_and_handle_model as chm


# Tests for clamp_0_100

def test_clamp_0_100_inside_range():
    assert chm.clamp_0_100(50) == 50.0
    assert chm.clamp_0_100(0) == 0.0
    assert chm.clamp_0_100(100) == 100.0


def test_clamp_0_100_below_zero():
    assert chm.clamp_0_100(-10) == 0.0


def test_clamp_0_100_above_hundred():
    assert chm.clamp_0_100(120) == 100.0


# Tests for cup_and_handle_score

def test_cup_and_handle_score_ideal_case_high_score():
    score = chm.cup_and_handle_score(
        rim_similarity_score=0.95,
        cup_depth_ratio=0.12,
        handle_shallow_score=0.95,
        handle_len_score=0.95,
        cup_roundness_score=0.95,
        handle_slope_score=0.95,
    )
    assert 90 <= score <= 100


def test_cup_and_handle_score_bad_case_low_score():
    score = chm.cup_and_handle_score(
        rim_similarity_score=0.10,
        cup_depth_ratio=0.01,
        handle_shallow_score=0.10,
        handle_len_score=0.10,
        cup_roundness_score=0.10,
        handle_slope_score=0.10,
    )
    assert 0 <= score <= 20


def test_cup_and_handle_score_boundary_rim_levels():
    score1 = chm.cup_and_handle_score(0.95, 0.12, 0.95, 0.95, 0.95, 0.95)
    score2 = chm.cup_and_handle_score(0.80, 0.12, 0.95, 0.95, 0.95, 0.95)
    score3 = chm.cup_and_handle_score(0.60, 0.12, 0.95, 0.95, 0.95, 0.95)
    score4 = chm.cup_and_handle_score(0.40, 0.12, 0.95, 0.95, 0.95, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_cup_and_handle_score_boundary_depth_levels():
    score1 = chm.cup_and_handle_score(0.95, 0.12, 0.95, 0.95, 0.95, 0.95)
    score2 = chm.cup_and_handle_score(0.95, 0.08, 0.95, 0.95, 0.95, 0.95)
    score3 = chm.cup_and_handle_score(0.95, 0.05, 0.95, 0.95, 0.95, 0.95)
    score4 = chm.cup_and_handle_score(0.95, 0.03, 0.95, 0.95, 0.95, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_cup_and_handle_score_boundary_handle_shallow_levels():
    score1 = chm.cup_and_handle_score(0.95, 0.12, 0.95, 0.95, 0.95, 0.95)
    score2 = chm.cup_and_handle_score(0.95, 0.12, 0.80, 0.95, 0.95, 0.95)
    score3 = chm.cup_and_handle_score(0.95, 0.12, 0.60, 0.95, 0.95, 0.95)
    score4 = chm.cup_and_handle_score(0.95, 0.12, 0.40, 0.95, 0.95, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_cup_and_handle_score_boundary_handle_len_levels():
    score1 = chm.cup_and_handle_score(0.95, 0.12, 0.95, 0.95, 0.95, 0.95)
    score2 = chm.cup_and_handle_score(0.95, 0.12, 0.95, 0.80, 0.95, 0.95)
    score3 = chm.cup_and_handle_score(0.95, 0.12, 0.95, 0.60, 0.95, 0.95)
    score4 = chm.cup_and_handle_score(0.95, 0.12, 0.95, 0.40, 0.95, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_cup_and_handle_score_boundary_roundness_levels():
    score1 = chm.cup_and_handle_score(0.95, 0.12, 0.95, 0.95, 0.95, 0.95)
    score2 = chm.cup_and_handle_score(0.95, 0.12, 0.95, 0.95, 0.80, 0.95)
    score3 = chm.cup_and_handle_score(0.95, 0.12, 0.95, 0.95, 0.60, 0.95)
    score4 = chm.cup_and_handle_score(0.95, 0.12, 0.95, 0.95, 0.40, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_cup_and_handle_score_boundary_handle_slope_levels():
    score1 = chm.cup_and_handle_score(0.95, 0.12, 0.95, 0.95, 0.95, 0.95)
    score2 = chm.cup_and_handle_score(0.95, 0.12, 0.95, 0.95, 0.95, 0.80)
    score3 = chm.cup_and_handle_score(0.95, 0.12, 0.95, 0.95, 0.95, 0.60)
    score4 = chm.cup_and_handle_score(0.95, 0.12, 0.95, 0.95, 0.95, 0.40)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


# Tests for _sample

@pytest.mark.parametrize("region", [
    "ideal", "good", "borderline", "bad_rims",
    "shallow_cup", "deep_handle", "long_handle",
    "flat_bottom_issue", "bad_handle_slope", "random", "outside"
])
def test_sample_returns_valid_tuple(region):
    result = chm._sample(region)

    assert isinstance(result, tuple)
    assert len(result) == 6

    rim_similarity_score, cup_depth_ratio, handle_shallow_score, handle_len_score, cup_roundness_score, handle_slope_score = result

    assert isinstance(rim_similarity_score, float)
    assert isinstance(cup_depth_ratio, float)
    assert isinstance(handle_shallow_score, float)
    assert isinstance(handle_len_score, float)
    assert isinstance(cup_roundness_score, float)
    assert isinstance(handle_slope_score, float)


def test_sample_unknown_region_falls_to_outside_branch():
    result = chm._sample("unknown_region")
    assert isinstance(result, tuple)
    assert len(result) == 6


def test_sample_ranges_are_reasonable_for_outside():
    rim_similarity_score, cup_depth_ratio, handle_shallow_score, handle_len_score, cup_roundness_score, handle_slope_score = chm._sample("outside")

    assert 0.00 <= rim_similarity_score <= 0.25
    assert 0.00 <= cup_depth_ratio <= 0.03
    assert 0.00 <= handle_shallow_score <= 0.25
    assert 0.00 <= handle_len_score <= 0.25
    assert 0.00 <= cup_roundness_score <= 0.25
    assert 0.00 <= handle_slope_score <= 0.25


# Tests for build_dataset

def test_build_dataset_returns_dataframe():
    df = chm.build_dataset()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_build_dataset_has_expected_columns():
    df = chm.build_dataset()
    expected_columns = {
        "rim_similarity_score",
        "cup_depth_ratio",
        "handle_shallow_score",
        "handle_len_score",
        "cup_roundness_score",
        "handle_slope_score",
        "quality",
        "region",
    }
    assert expected_columns.issubset(set(df.columns))


def test_build_dataset_expected_row_count():
    df = chm.build_dataset()
    expected_count = sum(chm.REGION_COUNTS.values())
    assert len(df) == expected_count


def test_build_dataset_quality_in_valid_range():
    df = chm.build_dataset()
    assert df["quality"].between(0, 100).all()


def test_build_dataset_regions_match_region_counts():
    df = chm.build_dataset()
    counts = df["region"].value_counts().to_dict()

    for region, expected_count in chm.REGION_COUNTS.items():
        assert counts.get(region, 0) == expected_count


# Tests for train

def test_train_returns_expected_outputs():
    df = chm.build_dataset().head(500)

    model, X_test, y_test, y_pred, mae, r2 = chm.train(df)

    assert model is not None
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(mae, float)
    assert isinstance(r2, float)

    assert len(X_test) == len(y_test) == len(y_pred)


def test_train_predictions_reasonable_range():
    df = chm.build_dataset().head(500)

    model, X_test, y_test, y_pred, mae, r2 = chm.train(df)

    assert np.isfinite(y_pred).all()


# Tests for load_model

def test_load_model_file_not_found_raises(monkeypatch):
    monkeypatch.setattr(chm.os.path, "exists", lambda path: False)

    with pytest.raises(FileNotFoundError):
        chm.load_model()


def test_load_model_success(monkeypatch):
    class DummyModel:
        pass

    dummy_model = DummyModel()

    monkeypatch.setattr(chm.os.path, "exists", lambda path: True)
    monkeypatch.setattr(chm.joblib, "load", lambda path: dummy_model)

    model = chm.load_model()
    assert model is dummy_model


# Tests for learned_confidence

def test_learned_confidence_returns_clamped_value():
    class DummyModel:
        def predict(self, x):
            return np.array([85.5])

    score = chm.learned_confidence(
        rim_similarity_score=0.82,
        cup_depth_ratio=0.08,
        handle_shallow_score=0.82,
        handle_len_score=0.80,
        cup_roundness_score=0.82,
        handle_slope_score=0.82,
        model=DummyModel(),
    )

    assert score == 85.5


def test_learned_confidence_clamps_below_zero():
    class DummyModel:
        def predict(self, x):
            return np.array([-10.0])

    score = chm.learned_confidence(
        rim_similarity_score=0.82,
        cup_depth_ratio=0.08,
        handle_shallow_score=0.82,
        handle_len_score=0.80,
        cup_roundness_score=0.82,
        handle_slope_score=0.82,
        model=DummyModel(),
    )

    assert score == 0.0


def test_learned_confidence_clamps_above_hundred():
    class DummyModel:
        def predict(self, x):
            return np.array([120.0])

    score = chm.learned_confidence(
        rim_similarity_score=0.82,
        cup_depth_ratio=0.08,
        handle_shallow_score=0.82,
        handle_len_score=0.80,
        cup_roundness_score=0.82,
        handle_slope_score=0.82,
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

    score = chm.learned_confidence(
        rim_similarity_score=0.72,
        cup_depth_ratio=0.07,
        handle_shallow_score=0.74,
        handle_len_score=0.68,
        cup_roundness_score=0.70,
        handle_slope_score=0.66,
        model=DummyModel(),
    )

    assert score == 50.0
    assert captured["shape"] == (1, 6)
    assert captured["value"][0, 0] == 0.72
    assert captured["value"][0, 1] == 0.07
    assert captured["value"][0, 2] == 0.74
    assert captured["value"][0, 3] == 0.68
    assert captured["value"][0, 4] == 0.70
    assert captured["value"][0, 5] == 0.66