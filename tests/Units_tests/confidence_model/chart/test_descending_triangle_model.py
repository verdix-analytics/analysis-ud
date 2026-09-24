import numpy as np
import pandas as pd
import pytest

import stock_analysis.confidence_models.chart_patterns.descending_triangle_model as dtm


# =========================================================
# Tests for clamp_0_100
# =========================================================

def test_clamp_0_100_inside_range():
    assert dtm.clamp_0_100(50) == 50.0
    assert dtm.clamp_0_100(0) == 0.0
    assert dtm.clamp_0_100(100) == 100.0


def test_clamp_0_100_below_zero():
    assert dtm.clamp_0_100(-10) == 0.0


def test_clamp_0_100_above_hundred():
    assert dtm.clamp_0_100(120) == 100.0


# =========================================================
# Tests for descending_triangle_score
# =========================================================

def test_descending_triangle_score_ideal_case_high_score():
    score = dtm.descending_triangle_score(
        high_neg_pct_slope=0.0012,
        low_abs_pct_slope=0.0002,
        low_deviation=0.0015,
        gap_pct_now=0.010,
        pivot_count=8,
        touch_balance_ratio=1.0,
    )
    assert 90 <= score <= 100


def test_descending_triangle_score_bad_case_low_score():
    score = dtm.descending_triangle_score(
        high_neg_pct_slope=0.00001,
        low_abs_pct_slope=0.01,
        low_deviation=0.03,
        gap_pct_now=0.12,
        pivot_count=3,
        touch_balance_ratio=0.1,
    )
    assert 0 <= score <= 20


def test_descending_triangle_score_boundary_high_drop_levels():
    score1 = dtm.descending_triangle_score(0.0008, 0.0002, 0.0020, 0.010, 8, 1.0)
    score2 = dtm.descending_triangle_score(0.0005, 0.0002, 0.0020, 0.010, 8, 1.0)
    score3 = dtm.descending_triangle_score(0.00025, 0.0002, 0.0020, 0.010, 8, 1.0)
    score4 = dtm.descending_triangle_score(0.00010, 0.0002, 0.0020, 0.010, 8, 1.0)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_descending_triangle_score_boundary_low_flat_levels():
    score1 = dtm.descending_triangle_score(0.0010, 0.0004, 0.0020, 0.010, 8, 1.0)
    score2 = dtm.descending_triangle_score(0.0010, 0.0008, 0.0020, 0.010, 8, 1.0)
    score3 = dtm.descending_triangle_score(0.0010, 0.0015, 0.0020, 0.010, 8, 1.0)
    score4 = dtm.descending_triangle_score(0.0010, 0.0025, 0.0020, 0.010, 8, 1.0)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_descending_triangle_score_boundary_low_consistency_levels():
    score1 = dtm.descending_triangle_score(0.0010, 0.0002, 0.0025, 0.010, 8, 1.0)
    score2 = dtm.descending_triangle_score(0.0010, 0.0002, 0.0045, 0.010, 8, 1.0)
    score3 = dtm.descending_triangle_score(0.0010, 0.0002, 0.0080, 0.010, 8, 1.0)
    score4 = dtm.descending_triangle_score(0.0010, 0.0002, 0.0120, 0.010, 8, 1.0)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_descending_triangle_score_boundary_gap_levels():
    score1 = dtm.descending_triangle_score(0.0010, 0.0002, 0.0020, 0.015, 8, 1.0)
    score2 = dtm.descending_triangle_score(0.0010, 0.0002, 0.0020, 0.025, 8, 1.0)
    score3 = dtm.descending_triangle_score(0.0010, 0.0002, 0.0020, 0.040, 8, 1.0)
    score4 = dtm.descending_triangle_score(0.0010, 0.0002, 0.0020, 0.060, 8, 1.0)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_descending_triangle_score_boundary_pivot_count_levels():
    score1 = dtm.descending_triangle_score(0.0010, 0.0002, 0.0020, 0.010, 8, 1.0)
    score2 = dtm.descending_triangle_score(0.0010, 0.0002, 0.0020, 0.010, 6, 1.0)
    score3 = dtm.descending_triangle_score(0.0010, 0.0002, 0.0020, 0.010, 5, 1.0)
    score4 = dtm.descending_triangle_score(0.0010, 0.0002, 0.0020, 0.010, 4, 1.0)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_descending_triangle_score_boundary_touch_balance_levels():
    score1 = dtm.descending_triangle_score(0.0010, 0.0002, 0.0020, 0.010, 8, 0.90)
    score2 = dtm.descending_triangle_score(0.0010, 0.0002, 0.0020, 0.010, 8, 0.75)
    score3 = dtm.descending_triangle_score(0.0010, 0.0002, 0.0020, 0.010, 8, 0.60)
    score4 = dtm.descending_triangle_score(0.0010, 0.0002, 0.0020, 0.010, 8, 0.40)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


# =========================================================
# Tests for _sample
# =========================================================

@pytest.mark.parametrize("region", [
    "ideal", "good", "borderline", "flat_or_weak_highs",
    "sloping_bottom", "messy_lows", "wide_gap",
    "imbalanced_touches", "compressed", "random", "outside"
])
def test_sample_returns_valid_tuple(region):
    result = dtm._sample(region)

    assert isinstance(result, tuple)
    assert len(result) == 6

    high_neg_pct_slope, low_abs_pct_slope, low_deviation, gap_pct_now, pivot_count, touch_balance_ratio = result

    assert isinstance(high_neg_pct_slope, float)
    assert isinstance(low_abs_pct_slope, float)
    assert isinstance(low_deviation, float)
    assert isinstance(gap_pct_now, float)
    assert isinstance(pivot_count, float)
    assert isinstance(touch_balance_ratio, float)


def test_sample_unknown_region_falls_to_outside_branch():
    result = dtm._sample("unknown_region")
    assert isinstance(result, tuple)
    assert len(result) == 6


def test_sample_ranges_are_reasonable_for_compressed():
    high_neg_pct_slope, low_abs_pct_slope, low_deviation, gap_pct_now, pivot_count, touch_balance_ratio = dtm._sample("compressed")

    assert 0.0003 <= high_neg_pct_slope <= 0.0012
    assert 0.0000 <= low_abs_pct_slope <= 0.0012
    assert 0.0020 <= low_deviation <= 0.0060
    assert 0.003 <= gap_pct_now <= 0.012
    assert pivot_count == 4.0
    assert touch_balance_ratio == 1.0


# =========================================================
# Tests for build_dataset
# =========================================================

def test_build_dataset_returns_dataframe():
    df = dtm.build_dataset()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_build_dataset_has_expected_columns():
    df = dtm.build_dataset()
    expected_columns = {
        "high_neg_pct_slope",
        "low_abs_pct_slope",
        "low_deviation",
        "gap_pct_now",
        "pivot_count",
        "touch_balance_ratio",
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


# =========================================================
# Tests for train
# =========================================================

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


# =========================================================
# Tests for load_model
# =========================================================

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


# =========================================================
# Tests for learned_confidence
# =========================================================

def test_learned_confidence_returns_clamped_value():
    class DummyModel:
        def predict(self, x):
            return np.array([85.5])

    score = dtm.learned_confidence(
        high_neg_pct_slope=0.0007,
        low_abs_pct_slope=0.0007,
        low_deviation=0.0035,
        gap_pct_now=0.020,
        pivot_count=6,
        touch_balance_ratio=0.75,
        model=DummyModel(),
    )

    assert score == 85.5


def test_learned_confidence_clamps_below_zero():
    class DummyModel:
        def predict(self, x):
            return np.array([-10.0])

    score = dtm.learned_confidence(
        high_neg_pct_slope=0.0007,
        low_abs_pct_slope=0.0007,
        low_deviation=0.0035,
        gap_pct_now=0.020,
        pivot_count=6,
        touch_balance_ratio=0.75,
        model=DummyModel(),
    )

    assert score == 0.0


def test_learned_confidence_clamps_above_hundred():
    class DummyModel:
        def predict(self, x):
            return np.array([120.0])

    score = dtm.learned_confidence(
        high_neg_pct_slope=0.0007,
        low_abs_pct_slope=0.0007,
        low_deviation=0.0035,
        gap_pct_now=0.020,
        pivot_count=6,
        touch_balance_ratio=0.75,
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
        high_neg_pct_slope=0.00042,
        low_abs_pct_slope=0.0013,
        low_deviation=0.0058,
        gap_pct_now=0.028,
        pivot_count=6,
        touch_balance_ratio=0.60,
        model=DummyModel(),
    )

    assert score == 50.0
    assert captured["shape"] == (1, 6)
    assert captured["value"][0, 0] == 0.00042
    assert captured["value"][0, 1] == 0.0013
    assert captured["value"][0, 2] == 0.0058
    assert captured["value"][0, 3] == 0.028
    assert captured["value"][0, 4] == 6
    assert captured["value"][0, 5] == 0.60