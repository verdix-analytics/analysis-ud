import numpy as np
import pandas as pd
import pytest

import stock_analysis.confidence_models.chart_patterns.triple_top_model as ttm


# =========================================================
# Tests for clamp_0_100
# =========================================================

def test_clamp_0_100_inside_range():
    assert ttm.clamp_0_100(50) == 50.0
    assert ttm.clamp_0_100(0) == 0.0
    assert ttm.clamp_0_100(100) == 100.0


def test_clamp_0_100_below_zero():
    assert ttm.clamp_0_100(-10) == 0.0


def test_clamp_0_100_above_hundred():
    assert ttm.clamp_0_100(120) == 100.0


# =========================================================
# Tests for triple_top_score
# =========================================================

def test_triple_top_score_ideal_case_high_score():
    score = ttm.triple_top_score(
        peak_spread_pct=0.005,
        valley_depth=0.070,
        span_score=0.95,
        symmetry_ratio=0.95,
        valley_balance_ratio=0.95,
        total_span=10,
    )
    assert 90 <= score <= 100


def test_triple_top_score_bad_case_low_score():
    score = ttm.triple_top_score(
        peak_spread_pct=0.10,
        valley_depth=0.005,
        span_score=0.10,
        symmetry_ratio=0.20,
        valley_balance_ratio=0.20,
        total_span=3,
    )
    assert 0 <= score <= 20


def test_triple_top_score_boundary_peak_levels():
    score1 = ttm.triple_top_score(0.008, 0.060, 0.95, 0.95, 0.95, 10)
    score2 = ttm.triple_top_score(0.015, 0.060, 0.95, 0.95, 0.95, 10)
    score3 = ttm.triple_top_score(0.025, 0.060, 0.95, 0.95, 0.95, 10)
    score4 = ttm.triple_top_score(0.035, 0.060, 0.95, 0.95, 0.95, 10)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_triple_top_score_boundary_valley_levels():
    score1 = ttm.triple_top_score(0.005, 0.060, 0.95, 0.95, 0.95, 10)
    score2 = ttm.triple_top_score(0.005, 0.045, 0.95, 0.95, 0.95, 10)
    score3 = ttm.triple_top_score(0.005, 0.030, 0.95, 0.95, 0.95, 10)
    score4 = ttm.triple_top_score(0.005, 0.020, 0.95, 0.95, 0.95, 10)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_triple_top_score_boundary_span_score_levels():
    score1 = ttm.triple_top_score(0.005, 0.060, 0.95, 0.95, 0.95, 10)
    score2 = ttm.triple_top_score(0.005, 0.060, 0.75, 0.95, 0.95, 10)
    score3 = ttm.triple_top_score(0.005, 0.060, 0.55, 0.95, 0.95, 10)
    score4 = ttm.triple_top_score(0.005, 0.060, 0.35, 0.95, 0.95, 10)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_triple_top_score_boundary_symmetry_levels():
    score1 = ttm.triple_top_score(0.005, 0.060, 0.95, 0.90, 0.95, 10)
    score2 = ttm.triple_top_score(0.005, 0.060, 0.95, 0.75, 0.95, 10)
    score3 = ttm.triple_top_score(0.005, 0.060, 0.95, 0.60, 0.95, 10)
    score4 = ttm.triple_top_score(0.005, 0.060, 0.95, 0.40, 0.95, 10)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_triple_top_score_boundary_valley_balance_levels():
    score1 = ttm.triple_top_score(0.005, 0.060, 0.95, 0.95, 0.95, 10)
    score2 = ttm.triple_top_score(0.005, 0.060, 0.95, 0.95, 0.80, 10)
    score3 = ttm.triple_top_score(0.005, 0.060, 0.95, 0.95, 0.65, 10)
    score4 = ttm.triple_top_score(0.005, 0.060, 0.95, 0.95, 0.45, 10)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_triple_top_score_boundary_total_span_levels():
    score1 = ttm.triple_top_score(0.005, 0.060, 0.95, 0.95, 0.95, 10)
    score2 = ttm.triple_top_score(0.005, 0.060, 0.95, 0.95, 0.95, 8)
    score3 = ttm.triple_top_score(0.005, 0.060, 0.95, 0.95, 0.95, 6)
    score4 = ttm.triple_top_score(0.005, 0.060, 0.95, 0.95, 0.95, 4)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


# =========================================================
# Tests for _sample
# =========================================================

@pytest.mark.parametrize("region", [
    "ideal", "good", "borderline", "bad_peaks",
    "shallow_valleys", "asymmetric", "uneven_valleys",
    "compressed", "random", "outside"
])
def test_sample_returns_valid_tuple(region):
    result = ttm._sample(region)

    assert isinstance(result, tuple)
    assert len(result) == 6

    peak_spread_pct, valley_depth, span_score, symmetry_ratio, valley_balance_ratio, total_span = result

    assert isinstance(peak_spread_pct, float)
    assert isinstance(valley_depth, float)
    assert isinstance(span_score, float)
    assert isinstance(symmetry_ratio, float)
    assert isinstance(valley_balance_ratio, float)
    assert isinstance(total_span, float)


def test_sample_unknown_region_falls_to_outside_branch():
    result = ttm._sample("unknown_region")
    assert isinstance(result, tuple)
    assert len(result) == 6


def test_sample_ranges_are_reasonable_for_compressed():
    peak_spread_pct, valley_depth, span_score, symmetry_ratio, valley_balance_ratio, total_span = ttm._sample("compressed")

    assert 0.005 <= peak_spread_pct <= 0.020
    assert 0.020 <= valley_depth <= 0.050
    assert 0.10 <= span_score <= 0.35
    assert 0.55 <= symmetry_ratio <= 0.95
    assert 0.55 <= valley_balance_ratio <= 0.95
    assert 3 <= total_span <= 5


# =========================================================
# Tests for build_dataset
# =========================================================

def test_build_dataset_returns_dataframe():
    df = ttm.build_dataset()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_build_dataset_has_expected_columns():
    df = ttm.build_dataset()
    expected_columns = {
        "peak_spread_pct",
        "valley_depth",
        "span_score",
        "symmetry_ratio",
        "valley_balance_ratio",
        "total_span",
        "quality",
        "region",
    }
    assert expected_columns.issubset(set(df.columns))


def test_build_dataset_expected_row_count():
    df = ttm.build_dataset()
    expected_count = sum(ttm.REGION_COUNTS.values())
    assert len(df) == expected_count


def test_build_dataset_quality_in_valid_range():
    df = ttm.build_dataset()
    assert df["quality"].between(0, 100).all()


def test_build_dataset_regions_match_region_counts():
    df = ttm.build_dataset()
    counts = df["region"].value_counts().to_dict()

    for region, expected_count in ttm.REGION_COUNTS.items():
        assert counts.get(region, 0) == expected_count


# =========================================================
# Tests for train
# =========================================================

def test_train_returns_expected_outputs():
    df = ttm.build_dataset().head(500)

    model, X_test, y_test, y_pred, mae, r2 = ttm.train(df)

    assert model is not None
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(mae, float)
    assert isinstance(r2, float)

    assert len(X_test) == len(y_test) == len(y_pred)


def test_train_predictions_reasonable_range():
    df = ttm.build_dataset().head(500)

    model, X_test, y_test, y_pred, mae, r2 = ttm.train(df)

    assert np.isfinite(y_pred).all()


# =========================================================
# Tests for load_model
# =========================================================

def test_load_model_file_not_found_raises(monkeypatch):
    monkeypatch.setattr(ttm.os.path, "exists", lambda path: False)

    with pytest.raises(FileNotFoundError):
        ttm.load_model()


def test_load_model_success(monkeypatch):
    class DummyModel:
        pass

    dummy_model = DummyModel()

    monkeypatch.setattr(ttm.os.path, "exists", lambda path: True)
    monkeypatch.setattr(ttm.joblib, "load", lambda path: dummy_model)

    model = ttm.load_model()
    assert model is dummy_model


# =========================================================
# Tests for learned_confidence
# =========================================================

def test_learned_confidence_returns_clamped_value():
    class DummyModel:
        def predict(self, x):
            return np.array([85.5])

    score = ttm.learned_confidence(
        peak_spread_pct=0.01,
        valley_depth=0.05,
        span_score=0.80,
        symmetry_ratio=0.85,
        valley_balance_ratio=0.82,
        total_span=8,
        model=DummyModel(),
    )

    assert score == 85.5


def test_learned_confidence_clamps_below_zero():
    class DummyModel:
        def predict(self, x):
            return np.array([-10.0])

    score = ttm.learned_confidence(
        peak_spread_pct=0.01,
        valley_depth=0.05,
        span_score=0.80,
        symmetry_ratio=0.85,
        valley_balance_ratio=0.82,
        total_span=8,
        model=DummyModel(),
    )

    assert score == 0.0


def test_learned_confidence_clamps_above_hundred():
    class DummyModel:
        def predict(self, x):
            return np.array([120.0])

    score = ttm.learned_confidence(
        peak_spread_pct=0.01,
        valley_depth=0.05,
        span_score=0.80,
        symmetry_ratio=0.85,
        valley_balance_ratio=0.82,
        total_span=8,
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

    score = ttm.learned_confidence(
        peak_spread_pct=0.018,
        valley_depth=0.032,
        span_score=0.62,
        symmetry_ratio=0.58,
        valley_balance_ratio=0.70,
        total_span=7,
        model=DummyModel(),
    )

    assert score == 50.0
    assert captured["shape"] == (1, 6)
    assert captured["value"][0, 0] == 0.018
    assert captured["value"][0, 1] == 0.032
    assert captured["value"][0, 2] == 0.62
    assert captured["value"][0, 3] == 0.58
    assert captured["value"][0, 4] == 0.70
    assert captured["value"][0, 5] == 7