import numpy as np
import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


np.random.seed(42)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "pkl_files")
MODEL_PATH = os.path.join(MODEL_DIR, "diamond_bottom_model.pkl")

FEATURES = [
    "center_proximity_score",
    "left_expansion_score",
    "right_contraction_score",
    "span_symmetry_ratio",
    "boundary_tightness",
    "pivot_richness_score",
]


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def diamond_bottom_score(
    center_proximity_score: float,
    left_expansion_score: float,
    right_contraction_score: float,
    span_symmetry_ratio: float,
    boundary_tightness: float,
    pivot_richness_score: float,
) -> float:
    """
    Strict rule-based quality score for a textbook-style Diamond Bottom.
    Output: [0, 100]
    """

    if center_proximity_score >= 0.95:
        score_center = 100
    elif center_proximity_score >= 0.75:
        score_center = 80
    elif center_proximity_score >= 0.55:
        score_center = 55
    elif center_proximity_score >= 0.35:
        score_center = 25
    else:
        score_center = 0

    if left_expansion_score >= 0.95:
        score_left = 100
    elif left_expansion_score >= 0.75:
        score_left = 80
    elif left_expansion_score >= 0.55:
        score_left = 55
    elif left_expansion_score >= 0.35:
        score_left = 25
    else:
        score_left = 0

    if right_contraction_score >= 0.95:
        score_right = 100
    elif right_contraction_score >= 0.75:
        score_right = 80
    elif right_contraction_score >= 0.55:
        score_right = 55
    elif right_contraction_score >= 0.35:
        score_right = 25
    else:
        score_right = 0

    if span_symmetry_ratio >= 0.90:
        score_sym = 100
    elif span_symmetry_ratio >= 0.75:
        score_sym = 80
    elif span_symmetry_ratio >= 0.60:
        score_sym = 55
    elif span_symmetry_ratio >= 0.40:
        score_sym = 25
    else:
        score_sym = 0

    if boundary_tightness >= 0.90:
        score_tight = 100
    elif boundary_tightness >= 0.75:
        score_tight = 80
    elif boundary_tightness >= 0.55:
        score_tight = 55
    elif boundary_tightness >= 0.35:
        score_tight = 25
    else:
        score_tight = 0

    if pivot_richness_score >= 0.95:
        score_pivots = 100
    elif pivot_richness_score >= 0.75:
        score_pivots = 80
    elif pivot_richness_score >= 0.55:
        score_pivots = 55
    elif pivot_richness_score >= 0.35:
        score_pivots = 25
    else:
        score_pivots = 0

    overall = (
        0.20 * score_center +
        0.22 * score_left +
        0.22 * score_right +
        0.14 * score_sym +
        0.12 * score_tight +
        0.10 * score_pivots
    )

    return clamp_0_100(overall)


def _sample(region: str):
    if region == "ideal":
        center_proximity_score = np.random.uniform(0.90, 1.00)
        left_expansion_score = np.random.uniform(0.88, 1.00)
        right_contraction_score = np.random.uniform(0.88, 1.00)
        span_symmetry_ratio = np.random.uniform(0.80, 1.00)
        boundary_tightness = np.random.uniform(0.80, 1.00)
        pivot_richness_score = np.random.uniform(0.85, 1.00)

    elif region == "good":
        center_proximity_score = np.random.uniform(0.70, 0.95)
        left_expansion_score = np.random.uniform(0.70, 0.95)
        right_contraction_score = np.random.uniform(0.70, 0.95)
        span_symmetry_ratio = np.random.uniform(0.65, 0.90)
        boundary_tightness = np.random.uniform(0.65, 0.90)
        pivot_richness_score = np.random.uniform(0.65, 0.90)

    elif region == "borderline":
        center_proximity_score = np.random.uniform(0.45, 0.75)
        left_expansion_score = np.random.uniform(0.45, 0.75)
        right_contraction_score = np.random.uniform(0.45, 0.75)
        span_symmetry_ratio = np.random.uniform(0.45, 0.75)
        boundary_tightness = np.random.uniform(0.45, 0.75)
        pivot_richness_score = np.random.uniform(0.45, 0.75)

    elif region == "bad_center":
        center_proximity_score = np.random.uniform(0.00, 0.30)
        left_expansion_score = np.random.uniform(0.70, 1.00)
        right_contraction_score = np.random.uniform(0.70, 1.00)
        span_symmetry_ratio = np.random.uniform(0.55, 0.90)
        boundary_tightness = np.random.uniform(0.55, 0.90)
        pivot_richness_score = np.random.uniform(0.65, 0.95)

    elif region == "bad_left":
        center_proximity_score = np.random.uniform(0.60, 0.95)
        left_expansion_score = np.random.uniform(0.00, 0.35)
        right_contraction_score = np.random.uniform(0.70, 1.00)
        span_symmetry_ratio = np.random.uniform(0.55, 0.90)
        boundary_tightness = np.random.uniform(0.55, 0.90)
        pivot_richness_score = np.random.uniform(0.65, 0.95)

    elif region == "bad_right":
        center_proximity_score = np.random.uniform(0.60, 0.95)
        left_expansion_score = np.random.uniform(0.70, 1.00)
        right_contraction_score = np.random.uniform(0.00, 0.35)
        span_symmetry_ratio = np.random.uniform(0.55, 0.90)
        boundary_tightness = np.random.uniform(0.20, 0.60)
        pivot_richness_score = np.random.uniform(0.65, 0.95)

    elif region == "asymmetric":
        center_proximity_score = np.random.uniform(0.55, 0.90)
        left_expansion_score = np.random.uniform(0.60, 0.95)
        right_contraction_score = np.random.uniform(0.60, 0.95)
        span_symmetry_ratio = np.random.uniform(0.10, 0.45)
        boundary_tightness = np.random.uniform(0.55, 0.90)
        pivot_richness_score = np.random.uniform(0.65, 0.95)

    elif region == "loose_boundary":
        center_proximity_score = np.random.uniform(0.55, 0.90)
        left_expansion_score = np.random.uniform(0.60, 0.95)
        right_contraction_score = np.random.uniform(0.60, 0.95)
        span_symmetry_ratio = np.random.uniform(0.55, 0.90)
        boundary_tightness = np.random.uniform(0.00, 0.40)
        pivot_richness_score = np.random.uniform(0.65, 0.95)

    elif region == "few_pivots":
        center_proximity_score = np.random.uniform(0.55, 0.90)
        left_expansion_score = np.random.uniform(0.55, 0.90)
        right_contraction_score = np.random.uniform(0.55, 0.90)
        span_symmetry_ratio = np.random.uniform(0.50, 0.85)
        boundary_tightness = np.random.uniform(0.50, 0.85)
        pivot_richness_score = np.random.uniform(0.00, 0.35)

    elif region == "random":
        center_proximity_score = np.random.uniform(0.00, 1.00)
        left_expansion_score = np.random.uniform(0.00, 1.00)
        right_contraction_score = np.random.uniform(0.00, 1.00)
        span_symmetry_ratio = np.random.uniform(0.00, 1.00)
        boundary_tightness = np.random.uniform(0.00, 1.00)
        pivot_richness_score = np.random.uniform(0.00, 1.00)

    else:  # outside
        center_proximity_score = np.random.uniform(0.00, 0.20)
        left_expansion_score = np.random.uniform(0.00, 0.25)
        right_contraction_score = np.random.uniform(0.00, 0.25)
        span_symmetry_ratio = np.random.uniform(0.00, 0.25)
        boundary_tightness = np.random.uniform(0.00, 0.25)
        pivot_richness_score = np.random.uniform(0.00, 0.25)

    return (
        float(center_proximity_score),
        float(left_expansion_score),
        float(right_contraction_score),
        float(span_symmetry_ratio),
        float(boundary_tightness),
        float(pivot_richness_score),
    )


REGION_COUNTS = {
    "ideal": 1400,
    "good": 1200,
    "borderline": 1000,
    "bad_center": 900,
    "bad_left": 900,
    "bad_right": 900,
    "asymmetric": 700,
    "loose_boundary": 700,
    "few_pivots": 600,
    "random": 700,
    "outside": 300,
}


def build_dataset() -> pd.DataFrame:
    rows = []

    for region, n in REGION_COUNTS.items():
        for _ in range(n):
            (
                center_proximity_score,
                left_expansion_score,
                right_contraction_score,
                span_symmetry_ratio,
                boundary_tightness,
                pivot_richness_score,
            ) = _sample(region)

            base_score = diamond_bottom_score(
                center_proximity_score=center_proximity_score,
                left_expansion_score=left_expansion_score,
                right_contraction_score=right_contraction_score,
                span_symmetry_ratio=span_symmetry_ratio,
                boundary_tightness=boundary_tightness,
                pivot_richness_score=pivot_richness_score,
            )

            noise = np.random.normal(0, 2.0)
            quality = clamp_0_100(base_score + noise)

            rows.append({
                "center_proximity_score": center_proximity_score,
                "left_expansion_score": left_expansion_score,
                "right_contraction_score": right_contraction_score,
                "span_symmetry_ratio": span_symmetry_ratio,
                "boundary_tightness": boundary_tightness,
                "pivot_richness_score": pivot_richness_score,
                "quality": quality,
                "region": region,
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def train(df: pd.DataFrame):
    X = df[FEATURES].values
    y = df["quality"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, X_test, y_test, y_pred, mae, r2


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Diamond Bottom regression model not found. Run "
            "`python -m stock_analysis.confidence_models.chart_patterns.diamond_bottom_model` "
            "to train and save it first."
        )
    model = joblib.load(MODEL_PATH)
    return model


def learned_confidence(
    center_proximity_score: float,
    left_expansion_score: float,
    right_contraction_score: float,
    span_symmetry_ratio: float,
    boundary_tightness: float,
    pivot_richness_score: float,
    model: RandomForestRegressor,
) -> float:
    x = np.array([[
        center_proximity_score,
        left_expansion_score,
        right_contraction_score,
        span_symmetry_ratio,
        boundary_tightness,
        pivot_richness_score,
    ]])

    pred = float(model.predict(x)[0])
    return clamp_0_100(pred)


if __name__ == "__main__":
    df = build_dataset()

    print(f"\nTotal samples: {len(df)}")
    print("\nQuality distribution:")
    print(df["quality"].describe())

    print("\nUnique rounded quality values (sample):")
    print(sorted(df["quality"].round(2).unique()[:20]))
    print(f"Total unique rounded qualities: {len(np.unique(df['quality'].round(2)))}")

    model, X_test, y_test, y_pred, mae, r2 = train(df)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"\nTest MAE: {mae:.3f}")
    print(f"Test R^2: {r2:.3f}")

    test_cases = [
        ("Ideal",          1.00, 0.95, 0.95, 0.90, 0.90, 0.95),
        ("Good",           0.82, 0.80, 0.78, 0.76, 0.74, 0.80),
        ("Borderline",     0.60, 0.58, 0.55, 0.55, 0.52, 0.55),
        ("Bad center",     0.15, 0.85, 0.80, 0.75, 0.70, 0.85),
        ("Bad left",       0.80, 0.20, 0.82, 0.75, 0.72, 0.82),
        ("Bad right",      0.82, 0.84, 0.20, 0.72, 0.30, 0.82),
        ("Asymmetric",     0.78, 0.80, 0.82, 0.20, 0.72, 0.82),
        ("Loose bound",    0.78, 0.82, 0.78, 0.74, 0.18, 0.80),
        ("Few pivots",     0.78, 0.75, 0.72, 0.70, 0.72, 0.18),
        ("User example",   0.74, 0.68, 0.71, 0.62, 0.58, 0.64),
    ]

    print(f"\n{'Pattern':<15} {'RuleScore':>10} {'Predicted':>10} {'Delta':>10}")
    print("-" * 52)

    for desc, center_proximity_score, left_expansion_score, right_contraction_score, span_symmetry_ratio, boundary_tightness, pivot_richness_score in test_cases:
        rule_score = diamond_bottom_score(
            center_proximity_score=center_proximity_score,
            left_expansion_score=left_expansion_score,
            right_contraction_score=right_contraction_score,
            span_symmetry_ratio=span_symmetry_ratio,
            boundary_tightness=boundary_tightness,
            pivot_richness_score=pivot_richness_score,
        )
        pred_score = learned_confidence(
            center_proximity_score=center_proximity_score,
            left_expansion_score=left_expansion_score,
            right_contraction_score=right_contraction_score,
            span_symmetry_ratio=span_symmetry_ratio,
            boundary_tightness=boundary_tightness,
            pivot_richness_score=pivot_richness_score,
            model=model,
        )
        delta = pred_score - rule_score
        print(f"{desc:<15} {rule_score:>10.2f} {pred_score:>10.2f} {delta:>10.2f}")