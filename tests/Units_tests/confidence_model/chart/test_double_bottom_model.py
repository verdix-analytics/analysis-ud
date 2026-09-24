import numpy as np
import pandas as pd
import pytest

import stock_analysis.confidence_models.chart_patterns.double_bottom_model as dbm


# =========================================================
# Tests for clamp_0_100
# =========================================================

def test_clamp_0_100_inside_range():
    assert dbm.clamp_0_100(50) == 50.0
    assert dbm.clamp_0_100(0) == 0.0
    assert dbm.clamp_0_100(100) == 100.0


def test_clamp_0_100_below_zero():
    assert dbm.clamp_0_100(-10) == 0.0


def test_clamp_0_100_above_hundred():
    assert dbm.clamp_0_100(120) == 100.0


# =========================================================
# Tests for double_bottom_score
# =========================================================

def test_double_bottom_score_ideal_case_high_score():
    score = dbm.double_bottom_score(
        bottom_diff_pct=0.005,
        rebound_ratio=0.070,
        left_span=4,
        right_span=4,
        total_span=8,
        symmetry_ratio=1.0,
    )
    assert 90 <= score <= 100


def test_double_bottom_score_bad_case_low_score():
    score = dbm.double_bottom_score(
        bottom_diff_pct=0.10,
        rebound_ratio=0.005,
        left_span=1,
        right_span=1,
        total_span=2,
        symmetry_ratio=0.2,
    )
    assert 0 <= score <= 20


def test_double_bottom_score_boundary_bottom_levels():
    score1 = dbm.double_bottom_score(0.008, 0.060, 4, 4, 8, 1.0)
    score2 = dbm.double_bottom_score(0.015, 0.060, 4, 4, 8, 1.0)
    score3 = dbm.double_bottom_score(0.025, 0.060, 4, 4, 8, 1.0)
    score4 = dbm.double_bottom_score(0.035, 0.060, 4, 4, 8, 1.0)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_double_bottom_score_boundary_rebound_levels():
    score1 = dbm.double_bottom_score(0.005, 0.060, 4, 4, 8, 1.0)
    score2 = dbm.double_bottom_score(0.005, 0.045, 4, 4, 8, 1.0)
    score3 = dbm.double_bottom_score(0.005, 0.030, 4, 4, 8, 1.0)
    score4 = dbm.double_bottom_score(0.005, 0.020, 4, 4, 8, 1.0)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_double_bottom_score_boundary_symmetry_levels():
    score1 = dbm.double_bottom_score(0.005, 0.060, 4, 4, 8, 0.90)
    score2 = dbm.double_bottom_score(0.005, 0.060, 4, 4, 8, 0.75)
    score3 = dbm.double_bottom_score(0.005, 0.060, 4, 4, 8, 0.60)
    score4 = dbm.double_bottom_score(0.005, 0.060, 4, 4, 8, 0.45)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_double_bottom_score_boundary_span_levels():
    score1 = dbm.double_bottom_score(0.005, 0.060, 4, 4, 8, 1.0)
    score2 = dbm.double_bottom_score(0.005, 0.060, 3, 3, 6, 1.0)
    score3 = dbm.double_bottom_score(0.005, 0.060, 2, 2, 4, 1.0)
    score4 = dbm.double_bottom_score(0.005, 0.060, 1, 2, 3, 0.5)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


# =========================================================
# Tests for _sample
# =========================================================

@pytest.mark.parametrize("region", [
    "ideal", "good", "borderline", "weak_rebound",
    "bad_bottom", "asymmetric", "compressed",
    "random", "outside"
])
def test_sample_returns_valid_tuple(region):
    result = dbm._sample(region)

    assert isinstance(result, tuple)
    assert len(result) == 6

    bottom_diff_pct, rebound_ratio, left_span, right_span, total_span, symmetry_ratio = result

    assert isinstance(bottom_diff_pct, float)
    assert isinstance(rebound_ratio, float)
    assert isinstance(left_span, float)
    assert isinstance(right_span, float)
    assert isinstance(total_span, float)
    assert isinstance(symmetry_ratio, float)

    assert total_span == left_span + right_span
    assert 0 <= symmetry_ratio <= 1


def test_sample_unknown_region_falls_to_outside_branch():
    result = dbm._sample("unknown_region")
    assert isinstance(result, tuple)
    assert len(result) == 6


def test_sample_compressed_region_has_span_1_1():
    _, _, left_span, right_span, total_span, symmetry_ratio = dbm._sample("compressed")
    assert left_span == 1.0
    assert right_span == 1.0
    assert total_span == 2.0
    assert symmetry_ratio == 1.0


# =========================================================
# Tests for build_dataset
# =========================================================

def test_build_dataset_returns_dataframe():
    df = dbm.build_dataset()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_build_dataset_has_expected_columns():
    df = dbm.build_dataset()
    expected_columns = {
        "bottom_diff_pct",
        "rebound_ratio",
        "left_span",
        "right_span",
        "total_span",
        "symmetry_ratio",
        "quality",
        "region",
    }
    assert expected_columns.issubset(set(df.columns))


def test_build_dataset_expected_row_count():
    df = dbm.build_dataset()
    expected_count = sum(dbm.REGION_COUNTS.values())
    assert len(df) == expected_count


def test_build_dataset_quality_in_valid_range():
    df = dbm.build_dataset()
    assert df["quality"].between(0, 100).all()


def test_build_dataset_regions_match_region_counts():
    df = dbm.build_dataset()
    counts = df["region"].value_counts().to_dict()

    for region, expected_count in dbm.REGION_COUNTS.items():
        assert counts.get(region, 0) == expected_count


# =========================================================
# Tests for train
# =========================================================

def test_train_returns_expected_outputs():
    df = dbm.build_dataset().head(500)

    model, X_test, y_test, y_pred, mae, r2 = dbm.train(df)

    assert model is not None
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(mae, float)
    assert isinstance(r2, float)

    assert len(X_test) == len(y_test) == len(y_pred)


def test_train_predictions_reasonable_range():
    df = dbm.build_dataset().head(500)

    model, X_test, y_test, y_pred, mae, r2 = dbm.train(df)

    assert np.isfinite(y_pred).all()


# =========================================================
# Tests for load_model
# =========================================================

def test_load_model_file_not_found_raises(monkeypatch):
    monkeypatch.setattr(dbm.os.path, "exists", lambda path: False)

    with pytest.raises(FileNotFoundError):
        dbm.load_model()


def test_load_model_success(monkeypatch):
    class DummyModel:
        pass

    dummy_model = DummyModel()

    monkeypatch.setattr(dbm.os.path, "exists", lambda path: True)
    monkeypatch.setattr(dbm.joblib, "load", lambda path: dummy_model)

    model = dbm.load_model()
    assert model is dummy_model


# =========================================================
# Tests for learned_confidence
# =========================================================

def test_learned_confidence_returns_clamped_value():
    class DummyModel:
        def predict(self, x):
            return np.array([85.5])

    score = dbm.learned_confidence(
        bottom_diff_pct=0.01,
        rebound_ratio=0.05,
        left_span=4,
        right_span=4,
        total_span=8,
        symmetry_ratio=1.0,
        model=DummyModel(),
    )

    assert score == 85.5


def test_learned_confidence_clamps_below_zero():
    class DummyModel:
        def predict(self, x):
            return np.array([-10.0])

    score = dbm.learned_confidence(
        bottom_diff_pct=0.01,
        rebound_ratio=0.05,
        left_span=4,
        right_span=4,
        total_span=8,
        symmetry_ratio=1.0,
        model=DummyModel(),
    )

    assert score == 0.0


def test_learned_confidence_clamps_above_hundred():
    class DummyModel:
        def predict(self, x):
            return np.array([120.0])

    score = dbm.learned_confidence(
        bottom_diff_pct=0.01,
        rebound_ratio=0.05,
        left_span=4,
        right_span=4,
        total_span=8,
        symmetry_ratio=1.0,
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

    score = dbm.learned_confidence(
        bottom_diff_pct=0.02,
        rebound_ratio=0.04,
        left_span=3,
        right_span=5,
        total_span=8,
        symmetry_ratio=0.6,
        model=DummyModel(),
    )

    assert score == 50.0
    assert captured["shape"] == (1, 6)
    assert captured["value"][0, 0] == 0.02
    assert captured["value"][0, 1] == 0.04
    assert captured["value"][0, 2] == 3
    assert captured["value"][0, 3] == 5
    assert captured["value"][0, 4] == 8
    assert captured["value"][0, 5] == 0.6