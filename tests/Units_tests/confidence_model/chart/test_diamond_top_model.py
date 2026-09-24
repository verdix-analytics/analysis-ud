import numpy as np
import pandas as pd
import pytest

import stock_analysis.confidence_models.chart_patterns.diamond_top_model as dtm


# Tests for clamp_0_100

def test_clamp_0_100_inside_range():
    assert dtm.clamp_0_100(50) == 50.0
    assert dtm.clamp_0_100(0) == 0.0
    assert dtm.clamp_0_100(100) == 100.0


def test_clamp_0_100_below_zero():
    assert dtm.clamp_0_100(-10) == 0.0


def test_clamp_0_100_above_hundred():
    assert dtm.clamp_0_100(120) == 100.0


# Tests for diamond_top_score

def test_diamond_top_score_ideal_case_high_score():
    score = dtm.diamond_top_score(
        center_proximity_score=0.98,
        left_expansion_score=0.95,
        right_contraction_score=0.95,
        span_symmetry_ratio=0.92,
        boundary_tightness=0.92,
        pivot_richness_score=0.95,
    )
    assert 90 <= score <= 100


def test_diamond_top_score_bad_case_low_score():
    score = dtm.diamond_top_score(
        center_proximity_score=0.10,
        left_expansion_score=0.10,
        right_contraction_score=0.10,
        span_symmetry_ratio=0.10,
        boundary_tightness=0.10,
        pivot_richness_score=0.10,
    )
    assert 0 <= score <= 20


def test_diamond_top_score_boundary_center_levels():
    score1 = dtm.diamond_top_score(0.95, 0.95, 0.95, 0.90, 0.90, 0.95)
    score2 = dtm.diamond_top_score(0.75, 0.95, 0.95, 0.90, 0.90, 0.95)
    score3 = dtm.diamond_top_score(0.55, 0.95, 0.95, 0.90, 0.90, 0.95)
    score4 = dtm.diamond_top_score(0.35, 0.95, 0.95, 0.90, 0.90, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_diamond_top_score_boundary_left_levels():
    score1 = dtm.diamond_top_score(0.95, 0.95, 0.95, 0.90, 0.90, 0.95)
    score2 = dtm.diamond_top_score(0.95, 0.75, 0.95, 0.90, 0.90, 0.95)
    score3 = dtm.diamond_top_score(0.95, 0.55, 0.95, 0.90, 0.90, 0.95)
    score4 = dtm.diamond_top_score(0.95, 0.35, 0.95, 0.90, 0.90, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_diamond_top_score_boundary_right_levels():
    score1 = dtm.diamond_top_score(0.95, 0.95, 0.95, 0.90, 0.90, 0.95)
    score2 = dtm.diamond_top_score(0.95, 0.95, 0.75, 0.90, 0.90, 0.95)
    score3 = dtm.diamond_top_score(0.95, 0.95, 0.55, 0.90, 0.90, 0.95)
    score4 = dtm.diamond_top_score(0.95, 0.95, 0.35, 0.90, 0.90, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_diamond_top_score_boundary_symmetry_levels():
    score1 = dtm.diamond_top_score(0.95, 0.95, 0.95, 0.90, 0.90, 0.95)
    score2 = dtm.diamond_top_score(0.95, 0.95, 0.95, 0.75, 0.90, 0.95)
    score3 = dtm.diamond_top_score(0.95, 0.95, 0.95, 0.60, 0.90, 0.95)
    score4 = dtm.diamond_top_score(0.95, 0.95, 0.95, 0.40, 0.90, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_diamond_top_score_boundary_tightness_levels():
    score1 = dtm.diamond_top_score(0.95, 0.95, 0.95, 0.90, 0.90, 0.95)
    score2 = dtm.diamond_top_score(0.95, 0.95, 0.95, 0.90, 0.75, 0.95)
    score3 = dtm.diamond_top_score(0.95, 0.95, 0.95, 0.90, 0.55, 0.95)
    score4 = dtm.diamond_top_score(0.95, 0.95, 0.95, 0.90, 0.35, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_diamond_top_score_boundary_pivot_richness_levels():
    score1 = dtm.diamond_top_score(0.95, 0.95, 0.95, 0.90, 0.90, 0.95)
    score2 = dtm.diamond_top_score(0.95, 0.95, 0.95, 0.90, 0.90, 0.75)
    score3 = dtm.diamond_top_score(0.95, 0.95, 0.95, 0.90, 0.90, 0.55)
    score4 = dtm.diamond_top_score(0.95, 0.95, 0.95, 0.90, 0.90, 0.35)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


# Tests for _sample
@pytest.mark.parametrize("region", [
    "ideal", "good", "borderline", "bad_center", "bad_left",
    "bad_right", "asymmetric", "loose_boundary",
    "few_pivots", "random", "outside"
])
def test_sample_returns_valid_tuple(region):
    result = dtm._sample(region)

    assert isinstance(result, tuple)
    assert len(result) == 6

    center_proximity_score, left_expansion_score, right_contraction_score, span_symmetry_ratio, boundary_tightness, pivot_richness_score = result

    assert isinstance(center_proximity_score, float)
    assert isinstance(left_expansion_score, float)
    assert isinstance(right_contraction_score, float)
    assert isinstance(span_symmetry_ratio, float)
    assert isinstance(boundary_tightness, float)
    assert isinstance(pivot_richness_score, float)


def test_sample_unknown_region_falls_to_outside_branch():
    result = dtm._sample("unknown_region")
    assert isinstance(result, tuple)
    assert len(result) == 6


def test_sample_ranges_are_reasonable_for_outside():
    center_proximity_score, left_expansion_score, right_contraction_score, span_symmetry_ratio, boundary_tightness, pivot_richness_score = dtm._sample("outside")

    assert 0.00 <= center_proximity_score <= 0.20
    assert 0.00 <= left_expansion_score <= 0.25
    assert 0.00 <= right_contraction_score <= 0.25
    assert 0.00 <= span_symmetry_ratio <= 0.25
    assert 0.00 <= boundary_tightness <= 0.25
    assert 0.00 <= pivot_richness_score <= 0.25


# Tests for build_dataset

def test_build_dataset_returns_dataframe():
    df = dtm.build_dataset()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_build_dataset_has_expected_columns():
    df = dtm.build_dataset()
    expected_columns = {
        "center_proximity_score",
        "left_expansion_score",
        "right_contraction_score",
        "span_symmetry_ratio",
        "boundary_tightness",
        "pivot_richness_score",
        "quality",
        "region",
    }
    assert expected_columns.issubset(set(df.columns))


def test_build_dataset_expected_row_count():
    df = dtm.build_dataset()
    expected_count = sum(dtm.REGION_COUNTS.values())
    assert len(df) == expected_count


def test_build_dataset_quality_in_valid_range():
    df = dtm.build_dataset()
    assert df["quality"].between(0, 100).all()


def test_build_dataset_regions_match_region_counts():
    df = dtm.build_dataset()
    counts = df["region"].value_counts().to_dict()

    for region, expected_count in dtm.REGION_COUNTS.items():
        assert counts.get(region, 0) == expected_count


# Tests for train

def test_train_returns_expected_outputs():
    df = dtm.build_dataset().head(500)

    model, X_test, y_test, y_pred, mae, r2 = dtm.train(df)

    assert model is not None
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(mae, float)
    assert isinstance(r2, float)

    assert len(X_test) == len(y_test) == len(y_pred)


def test_train_predictions_reasonable_range():
    df = dtm.build_dataset().head(500)

    model, X_test, y_test, y_pred, mae, r2 = dtm.train(df)

    assert np.isfinite(y_pred).all()


# Tests for load_model

def test_load_model_file_not_found_raises(monkeypatch):
    monkeypatch.setattr(dtm.os.path, "exists", lambda path: False)

    with pytest.raises(FileNotFoundError):
        dtm.load_model()


def test_load_model_success(monkeypatch):
    class DummyModel:
        pass

    dummy_model = DummyModel()

    monkeypatch.setattr(dtm.os.path, "exists", lambda path: True)
    monkeypatch.setattr(dtm.joblib, "load", lambda path: dummy_model)

    model = dtm.load_model()
    assert model is dummy_model


# Tests for learned_confidence

def test_learned_confidence_returns_clamped_value():
    class DummyModel:
        def predict(self, x):
            return np.array([85.5])

    score = dtm.learned_confidence(
        center_proximity_score=0.78,
        left_expansion_score=0.80,
        right_contraction_score=0.82,
        span_symmetry_ratio=0.74,
        boundary_tightness=0.72,
        pivot_richness_score=0.80,
        model=DummyModel(),
    )

    assert score == 85.5


def test_learned_confidence_clamps_below_zero():
    class DummyModel:
        def predict(self, x):
            return np.array([-10.0])

    score = dtm.learned_confidence(
        center_proximity_score=0.78,
        left_expansion_score=0.80,
        right_contraction_score=0.82,
        span_symmetry_ratio=0.74,
        boundary_tightness=0.72,
        pivot_richness_score=0.80,
        model=DummyModel(),
    )

    assert score == 0.0


def test_learned_confidence_clamps_above_hundred():
    class DummyModel:
        def predict(self, x):
            return np.array([120.0])

    score = dtm.learned_confidence(
        center_proximity_score=0.78,
        left_expansion_score=0.80,
        right_contraction_score=0.82,
        span_symmetry_ratio=0.74,
        boundary_tightness=0.72,
        pivot_richness_score=0.80,
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

    score = dtm.learned_confidence(
        center_proximity_score=0.74,
        left_expansion_score=0.68,
        right_contraction_score=0.71,
        span_symmetry_ratio=0.62,
        boundary_tightness=0.58,
        pivot_richness_score=0.64,
        model=DummyModel(),
    )

    assert score == 50.0
    assert captured["shape"] == (1, 6)
    assert captured["value"][0, 0] == 0.74
    assert captured["value"][0, 1] == 0.68
    assert captured["value"][0, 2] == 0.71
    assert captured["value"][0, 3] == 0.62
    assert captured["value"][0, 4] == 0.58
    assert captured["value"][0, 5] == 0.64