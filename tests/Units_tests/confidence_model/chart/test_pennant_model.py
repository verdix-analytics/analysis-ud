import numpy as np
import pandas as pd
import pytest

import stock_analysis.confidence_models.chart_patterns.pennant_model as pm


# Tests for clamp_0_100

def test_clamp_0_100_inside_range():
    assert pm.clamp_0_100(50) == 50.0
    assert pm.clamp_0_100(0) == 0.0
    assert pm.clamp_0_100(100) == 100.0


def test_clamp_0_100_below_zero():
    assert pm.clamp_0_100(-10) == 0.0


def test_clamp_0_100_above_hundred():
    assert pm.clamp_0_100(120) == 100.0


# Tests for pennant_score

def test_pennant_score_ideal_case_high_score():
    score = pm.pennant_score(
        pole_strength_score=0.95,
        convergence_score=0.95,
        compactness_score=0.95,
        pivot_richness_score=0.95,
        balance_score=0.95,
        symmetry_score=0.95,
    )
    assert 90 <= score <= 100


def test_pennant_score_bad_case_low_score():
    score = pm.pennant_score(
        pole_strength_score=0.10,
        convergence_score=0.10,
        compactness_score=0.10,
        pivot_richness_score=0.10,
        balance_score=0.10,
        symmetry_score=0.10,
    )
    assert 0 <= score <= 20


def test_pennant_score_boundary_pole_levels():
    score1 = pm.pennant_score(0.95, 0.95, 0.95, 0.95, 0.95, 0.95)
    score2 = pm.pennant_score(0.80, 0.95, 0.95, 0.95, 0.95, 0.95)
    score3 = pm.pennant_score(0.60, 0.95, 0.95, 0.95, 0.95, 0.95)
    score4 = pm.pennant_score(0.40, 0.95, 0.95, 0.95, 0.95, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_pennant_score_boundary_convergence_levels():
    score1 = pm.pennant_score(0.95, 0.95, 0.95, 0.95, 0.95, 0.95)
    score2 = pm.pennant_score(0.95, 0.80, 0.95, 0.95, 0.95, 0.95)
    score3 = pm.pennant_score(0.95, 0.60, 0.95, 0.95, 0.95, 0.95)
    score4 = pm.pennant_score(0.95, 0.40, 0.95, 0.95, 0.95, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_pennant_score_boundary_compactness_levels():
    score1 = pm.pennant_score(0.95, 0.95, 0.95, 0.95, 0.95, 0.95)
    score2 = pm.pennant_score(0.95, 0.95, 0.80, 0.95, 0.95, 0.95)
    score3 = pm.pennant_score(0.95, 0.95, 0.60, 0.95, 0.95, 0.95)
    score4 = pm.pennant_score(0.95, 0.95, 0.40, 0.95, 0.95, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_pennant_score_boundary_pivot_levels():
    score1 = pm.pennant_score(0.95, 0.95, 0.95, 0.95, 0.95, 0.95)
    score2 = pm.pennant_score(0.95, 0.95, 0.95, 0.80, 0.95, 0.95)
    score3 = pm.pennant_score(0.95, 0.95, 0.95, 0.60, 0.95, 0.95)
    score4 = pm.pennant_score(0.95, 0.95, 0.95, 0.40, 0.95, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_pennant_score_boundary_balance_levels():
    score1 = pm.pennant_score(0.95, 0.95, 0.95, 0.95, 0.95, 0.95)
    score2 = pm.pennant_score(0.95, 0.95, 0.95, 0.95, 0.80, 0.95)
    score3 = pm.pennant_score(0.95, 0.95, 0.95, 0.95, 0.60, 0.95)
    score4 = pm.pennant_score(0.95, 0.95, 0.95, 0.95, 0.40, 0.95)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


def test_pennant_score_boundary_symmetry_levels():
    score1 = pm.pennant_score(0.95, 0.95, 0.95, 0.95, 0.95, 0.95)
    score2 = pm.pennant_score(0.95, 0.95, 0.95, 0.95, 0.95, 0.80)
    score3 = pm.pennant_score(0.95, 0.95, 0.95, 0.95, 0.95, 0.60)
    score4 = pm.pennant_score(0.95, 0.95, 0.95, 0.95, 0.95, 0.40)

    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    assert 0 <= score3 <= 100
    assert 0 <= score4 <= 100


# Tests for _sample

@pytest.mark.parametrize("region", [
    "ideal", "good", "borderline", "weak_pole",
    "bad_convergence", "wide_pennant", "few_pivots",
    "imbalanced", "asymmetric", "random", "outside"
])
def test_sample_returns_valid_tuple(region):
    result = pm._sample(region)

    assert isinstance(result, tuple)
    assert len(result) == 6

    pole_strength_score, convergence_score, compactness_score, pivot_richness_score, balance_score, symmetry_score = result

    assert isinstance(pole_strength_score, float)
    assert isinstance(convergence_score, float)
    assert isinstance(compactness_score, float)
    assert isinstance(pivot_richness_score, float)
    assert isinstance(balance_score, float)
    assert isinstance(symmetry_score, float)


def test_sample_unknown_region_falls_to_outside_branch():
    result = pm._sample("unknown_region")
    assert isinstance(result, tuple)
    assert len(result) == 6


def test_sample_ranges_are_reasonable_for_outside():
    pole_strength_score, convergence_score, compactness_score, pivot_richness_score, balance_score, symmetry_score = pm._sample("outside")

    assert 0.00 <= pole_strength_score <= 0.25
    assert 0.00 <= convergence_score <= 0.25
    assert 0.00 <= compactness_score <= 0.25
    assert 0.00 <= pivot_richness_score <= 0.25
    assert 0.00 <= balance_score <= 0.25
    assert 0.00 <= symmetry_score <= 0.25


# Tests for build_dataset

def test_build_dataset_returns_dataframe():
    df = pm.build_dataset()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_build_dataset_has_expected_columns():
    df = pm.build_dataset()
    expected_columns = {
        "pole_strength_score",
        "convergence_score",
        "compactness_score",
        "pivot_richness_score",
        "balance_score",
        "symmetry_score",
        "quality",
        "region",
    }
    assert expected_columns.issubset(set(df.columns))


def test_build_dataset_expected_row_count():
    df = pm.build_dataset()
    expected_count = sum(pm.REGION_COUNTS.values())
    assert len(df) == expected_count


def test_build_dataset_quality_in_valid_range():
    df = pm.build_dataset()
    assert df["quality"].between(0, 100).all()


def test_build_dataset_regions_match_region_counts():
    df = pm.build_dataset()
    counts = df["region"].value_counts().to_dict()

    for region, expected_count in pm.REGION_COUNTS.items():
        assert counts.get(region, 0) == expected_count


# Tests for train

def test_train_returns_expected_outputs():
    df = pm.build_dataset().head(500)

    model, X_test, y_test, y_pred, mae, r2 = pm.train(df)

    assert model is not None
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(mae, float)
    assert isinstance(r2, float)

    assert len(X_test) == len(y_test) == len(y_pred)


def test_train_predictions_reasonable_range():
    df = pm.build_dataset().head(500)

    model, X_test, y_test, y_pred, mae, r2 = pm.train(df)

    assert np.isfinite(y_pred).all()


# Tests for load_model

def test_load_model_file_not_found_raises(monkeypatch):
    monkeypatch.setattr(pm.os.path, "exists", lambda path: False)

    with pytest.raises(FileNotFoundError):
        pm.load_model()


def test_load_model_success(monkeypatch):
    class DummyModel:
        pass

    dummy_model = DummyModel()

    monkeypatch.setattr(pm.os.path, "exists", lambda path: True)
    monkeypatch.setattr(pm.joblib, "load", lambda path: dummy_model)

    model = pm.load_model()
    assert model is dummy_model


# Tests for learned_confidence

def test_learned_confidence_returns_clamped_value():
    class DummyModel:
        def predict(self, x):
            return np.array([85.5])

    score = pm.learned_confidence(
        pole_strength_score=0.82,
        convergence_score=0.82,
        compactness_score=0.82,
        pivot_richness_score=0.80,
        balance_score=0.80,
        symmetry_score=0.80,
        model=DummyModel(),
    )

    assert score == 85.5


def test_learned_confidence_clamps_below_zero():
    class DummyModel:
        def predict(self, x):
            return np.array([-10.0])

    score = pm.learned_confidence(
        pole_strength_score=0.82,
        convergence_score=0.82,
        compactness_score=0.82,
        pivot_richness_score=0.80,
        balance_score=0.80,
        symmetry_score=0.80,
        model=DummyModel(),
    )

    assert score == 0.0


def test_learned_confidence_clamps_above_hundred():
    class DummyModel:
        def predict(self, x):
            return np.array([120.0])

    score = pm.learned_confidence(
        pole_strength_score=0.82,
        convergence_score=0.82,
        compactness_score=0.82,
        pivot_richness_score=0.80,
        balance_score=0.80,
        symmetry_score=0.80,
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

    score = pm.learned_confidence(
        pole_strength_score=0.72,
        convergence_score=0.70,
        compactness_score=0.68,
        pivot_richness_score=0.66,
        balance_score=0.74,
        symmetry_score=0.62,
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