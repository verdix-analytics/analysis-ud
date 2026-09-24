import numpy as np
import pandas as pd
import pytest

import stock_analysis.confidence_models.chart_patterns.inverse_head_and_shoulders_model as ihsm


# =========================================================
# Tests for clamp_0_100
# =========================================================

def test_clamp_0_100_inside_range():
    assert ihsm.clamp_0_100(50) == 50.0
    assert ihsm.clamp_0_100(0) == 0.0
    assert ihsm.clamp_0_100(100) == 100.0


def test_clamp_0_100_below_zero():
    assert ihsm.clamp_0_100(-10) == 0.0


def test_clamp_0_100_above_hundred():
    assert ihsm.clamp_0_100(120) == 100.0


# =========================================================
# Tests for inverse_head_and_shoulders_score
# =========================================================

def test_inverse_head_and_shoulders_score_ideal_case_high_score():
    score = ihsm.inverse_head_and_shoulders_score(
        shoulder_vs_head_min=0.05,
        shoulder_diff_pct=0.008,
        neckline_diff_pct=0.010,
        left_span=4,
        right_span=4,
        symmetry_ratio=1.0,
    )
    assert 90 <= score <= 100


def test_inverse_head_and_shoulders_score_bad_case_low_score():
    score = ihsm.inverse_head_and_shoulders_score(
        shoulder_vs_head_min=0.005,
        shoulder_diff_pct=0.10,
        neckline_diff_pct=0.15,
        left_span=1,
        right_span=1,
        symmetry_ratio=0.2,
    )
    assert 0 <= score <= 20


def test_inverse_head_and_shoulders_score_boundary_head_levels():
    score1 = ihsm.inverse_head_and_shoulders_score(0.040, 0.010, 0.010, 4, 4, 1.0)
    score2 = ihsm.inverse_head_and_shoulders_score(0.030, 0.010, 0.010, 4, 4, 1.0)
    score3 = ihsm.inverse_head_and_shoulders_score(0.020, 0.010, 0.010, 4, 4, 1.0)
    score4 = ihsm.inverse_head_and_shoulders_score(0.010, 0.010, 0.010, 4, 4, 1.0)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_inverse_head_and_shoulders_score_boundary_shoulder_levels():
    score1 = ihsm.inverse_head_and_shoulders_score(0.050, 0.010, 0.010, 4, 4, 1.0)
    score2 = ihsm.inverse_head_and_shoulders_score(0.050, 0.020, 0.010, 4, 4, 1.0)
    score3 = ihsm.inverse_head_and_shoulders_score(0.050, 0.030, 0.010, 4, 4, 1.0)
    score4 = ihsm.inverse_head_and_shoulders_score(0.050, 0.040, 0.010, 4, 4, 1.0)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_inverse_head_and_shoulders_score_boundary_neckline_levels():
    score1 = ihsm.inverse_head_and_shoulders_score(0.050, 0.010, 0.010, 4, 4, 1.0)
    score2 = ihsm.inverse_head_and_shoulders_score(0.050, 0.010, 0.020, 4, 4, 1.0)
    score3 = ihsm.inverse_head_and_shoulders_score(0.050, 0.010, 0.040, 4, 4, 1.0)
    score4 = ihsm.inverse_head_and_shoulders_score(0.050, 0.010, 0.060, 4, 4, 1.0)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_inverse_head_and_shoulders_score_boundary_symmetry_levels():
    score1 = ihsm.inverse_head_and_shoulders_score(0.050, 0.010, 0.010, 4, 4, 0.90)
    score2 = ihsm.inverse_head_and_shoulders_score(0.050, 0.010, 0.010, 4, 4, 0.75)
    score3 = ihsm.inverse_head_and_shoulders_score(0.050, 0.010, 0.010, 4, 4, 0.60)
    score4 = ihsm.inverse_head_and_shoulders_score(0.050, 0.010, 0.010, 4, 4, 0.45)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


# =========================================================
# Tests for _sample
# =========================================================

@pytest.mark.parametrize("region", [
    "ideal", "good", "borderline", "weak_head",
    "bad_shoulders", "bad_neckline", "asymmetric",
    "compressed", "random", "outside"
])
def test_sample_returns_valid_tuple(region):
    result = ihsm._sample(region)

    assert isinstance(result, tuple)
    assert len(result) == 6

    shoulder_vs_head_min, shoulder_diff_pct, neckline_diff_pct, left_span, right_span, symmetry_ratio = result

    assert isinstance(shoulder_vs_head_min, float)
    assert isinstance(shoulder_diff_pct, float)
    assert isinstance(neckline_diff_pct, float)
    assert isinstance(left_span, float)
    assert isinstance(right_span, float)
    assert isinstance(symmetry_ratio, float)

    assert 0 <= symmetry_ratio <= 1


def test_sample_unknown_region_falls_to_outside_branch():
    result = ihsm._sample("unknown_region")
    assert isinstance(result, tuple)
    assert len(result) == 6


def test_sample_compressed_region_has_span_1_1():
    _, _, _, left_span, right_span, symmetry_ratio = ihsm._sample("compressed")
    assert left_span == 1.0
    assert right_span == 1.0
    assert symmetry_ratio == 1.0


# =========================================================
# Tests for build_dataset
# =========================================================

def test_build_dataset_returns_dataframe():
    df = ihsm.build_dataset()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_build_dataset_has_expected_columns():
    df = ihsm.build_dataset()
    expected_columns = {
        "shoulder_vs_head_min",
        "shoulder_diff_pct",
        "neckline_diff_pct",
        "left_span",
        "right_span",
        "symmetry_ratio",
        "quality",
        "region",
    }
    assert expected_columns.issubset(set(df.columns))


def test_build_dataset_expected_row_count():
    df = ihsm.build_dataset()
    expected_count = sum(ihsm.REGION_COUNTS.values())
    assert len(df) == expected_count


def test_build_dataset_quality_in_valid_range():
    df = ihsm.build_dataset()
    assert df["quality"].between(0, 100).all()


def test_build_dataset_regions_match_region_counts():
    df = ihsm.build_dataset()
    counts = df["region"].value_counts().to_dict()

    for region, expected_count in ihsm.REGION_COUNTS.items():
        assert counts.get(region, 0) == expected_count


# =========================================================
# Tests for train
# =========================================================

def test_train_returns_expected_outputs():
    df = ihsm.build_dataset().head(500)

    model, X_test, y_test, y_pred, mae, r2 = ihsm.train(df)

    assert model is not None
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(mae, float)
    assert isinstance(r2, float)

    assert len(X_test) == len(y_test) == len(y_pred)


def test_train_predictions_reasonable_range():
    df = ihsm.build_dataset().head(500)

    model, X_test, y_test, y_pred, mae, r2 = ihsm.train(df)

    assert np.isfinite(y_pred).all()


# =========================================================
# Tests for load_model
# =========================================================

def test_load_model_file_not_found_raises(monkeypatch):
    monkeypatch.setattr(ihsm.os.path, "exists", lambda path: False)

    with pytest.raises(FileNotFoundError):
        ihsm.load_model()


def test_load_model_success(monkeypatch):
    class DummyModel:
        pass

    dummy_model = DummyModel()

    monkeypatch.setattr(ihsm.os.path, "exists", lambda path: True)
    monkeypatch.setattr(ihsm.joblib, "load", lambda path: dummy_model)

    model = ihsm.load_model()
    assert model is dummy_model


# =========================================================
# Tests for learned_confidence
# =========================================================

def test_learned_confidence_returns_clamped_value():
    class DummyModel:
        def predict(self, x):
            return np.array([85.5])

    score = ihsm.learned_confidence(
        shoulder_vs_head_min=0.03,
        shoulder_diff_pct=0.015,
        neckline_diff_pct=0.020,
        left_span=4,
        right_span=4,
        symmetry_ratio=1.0,
        model=DummyModel(),
    )

    assert score == 85.5


def test_learned_confidence_clamps_below_zero():
    class DummyModel:
        def predict(self, x):
            return np.array([-10.0])

    score = ihsm.learned_confidence(
        shoulder_vs_head_min=0.03,
        shoulder_diff_pct=0.015,
        neckline_diff_pct=0.020,
        left_span=4,
        right_span=4,
        symmetry_ratio=1.0,
        model=DummyModel(),
    )

    assert score == 0.0


def test_learned_confidence_clamps_above_hundred():
    class DummyModel:
        def predict(self, x):
            return np.array([120.0])

    score = ihsm.learned_confidence(
        shoulder_vs_head_min=0.03,
        shoulder_diff_pct=0.015,
        neckline_diff_pct=0.020,
        left_span=4,
        right_span=4,
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

    score = ihsm.learned_confidence(
        shoulder_vs_head_min=0.018,
        shoulder_diff_pct=0.026,
        neckline_diff_pct=0.031,
        left_span=5,
        right_span=3,
        symmetry_ratio=0.60,
        model=DummyModel(),
    )

    assert score == 50.0
    assert captured["shape"] == (1, 6)
    assert captured["value"][0, 0] == 0.018
    assert captured["value"][0, 1] == 0.026
    assert captured["value"][0, 2] == 0.031
    assert captured["value"][0, 3] == 5
    assert captured["value"][0, 4] == 3
    assert captured["value"][0, 5] == 0.60