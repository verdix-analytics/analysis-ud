import numpy as np
import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


np.random.seed(42)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "pkl_files")
MODEL_PATH = os.path.join(MODEL_DIR, "double_bottom_model.pkl")

FEATURES = [
    "bottom_diff_pct",
    "rebound_ratio",
    "left_span",
    "right_span",
    "total_span",
    "symmetry_ratio",
]


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def double_bottom_score(
    bottom_diff_pct: float,
    rebound_ratio: float,
    left_span: float,
    right_span: float,
    total_span: float,
    symmetry_ratio: float,
) -> float:
    """
    Strict rule-based quality score for a textbook-style Double Bottom.
    Output: [0, 100]
    """

    # 1) Bottom similarity: most important
    if bottom_diff_pct <= 0.008:
        score_bottom = 100
    elif bottom_diff_pct <= 0.015:
        score_bottom = 80
    elif bottom_diff_pct <= 0.025:
        score_bottom = 55
    elif bottom_diff_pct <= 0.035:
        score_bottom = 25
    else:
        score_bottom = 0

    # 2) Rebound strength: stronger middle peak is better
    if rebound_ratio >= 0.060:
        score_rebound = 100
    elif rebound_ratio >= 0.045:
        score_rebound = 80
    elif rebound_ratio >= 0.030:
        score_rebound = 55
    elif rebound_ratio >= 0.020:
        score_rebound = 25
    else:
        score_rebound = 0

    # 3) Symmetry: two sides should be reasonably balanced
    if symmetry_ratio >= 0.90:
        score_sym = 100
    elif symmetry_ratio >= 0.75:
        score_sym = 80
    elif symmetry_ratio >= 0.60:
        score_sym = 55
    elif symmetry_ratio >= 0.45:
        score_sym = 25
    else:
        score_sym = 0

    # 4) Span: not too compressed
    if total_span >= 8:
        score_span = 100
    elif total_span >= 6:
        score_span = 80
    elif total_span >= 4:
        score_span = 55
    elif total_span >= 3:
        score_span = 25
    else:
        score_span = 0

    overall = (
        0.42 * score_bottom +
        0.24 * score_rebound +
        0.24 * score_sym +
        0.10 * score_span
    )

    return clamp_0_100(overall)


def _sample(region: str):
    if region == "ideal":
        bottom_diff_pct = np.random.uniform(0.000, 0.008)
        rebound_ratio = np.random.uniform(0.050, 0.080)
        left_span = np.random.randint(3, 6)
        right_span = np.random.randint(3, 6)

    elif region == "good":
        bottom_diff_pct = np.random.uniform(0.008, 0.020)
        rebound_ratio = np.random.uniform(0.035, 0.060)
        left_span = np.random.randint(2, 6)
        right_span = np.random.randint(2, 6)

    elif region == "borderline":
        bottom_diff_pct = np.random.uniform(0.020, 0.040)
        rebound_ratio = np.random.uniform(0.020, 0.040)
        left_span = np.random.randint(2, 5)
        right_span = np.random.randint(2, 5)

    elif region == "weak_rebound":
        bottom_diff_pct = np.random.uniform(0.005, 0.025)
        rebound_ratio = np.random.uniform(0.005, 0.020)
        left_span = np.random.randint(2, 6)
        right_span = np.random.randint(2, 6)

    elif region == "bad_bottom":
        bottom_diff_pct = np.random.uniform(0.040, 0.090)
        rebound_ratio = np.random.uniform(0.025, 0.060)
        left_span = np.random.randint(2, 6)
        right_span = np.random.randint(2, 6)

    elif region == "asymmetric":
        bottom_diff_pct = np.random.uniform(0.005, 0.030)
        rebound_ratio = np.random.uniform(0.020, 0.060)
        left_span = np.random.randint(1, 3)
        right_span = np.random.randint(5, 10)

    elif region == "compressed":
        bottom_diff_pct = np.random.uniform(0.005, 0.030)
        rebound_ratio = np.random.uniform(0.020, 0.060)
        left_span = 1
        right_span = 1

    elif region == "random":
        bottom_diff_pct = np.random.uniform(0.000, 0.120)
        rebound_ratio = np.random.uniform(0.000, 0.100)
        left_span = np.random.randint(1, 10)
        right_span = np.random.randint(1, 10)

    else:  # outside
        bottom_diff_pct = np.random.uniform(0.080, 0.200)
        rebound_ratio = np.random.uniform(0.000, 0.010)
        left_span = np.random.randint(1, 3)
        right_span = np.random.randint(1, 3)

    total_span = left_span + right_span
    symmetry_ratio = min(left_span, right_span) / max(left_span, right_span)

    return (
        float(bottom_diff_pct),
        float(rebound_ratio),
        float(left_span),
        float(right_span),
        float(total_span),
        float(symmetry_ratio),
    )


REGION_COUNTS = {
    "ideal": 1400,
    "good": 1200,
    "borderline": 1000,
    "weak_rebound": 900,
    "bad_bottom": 900,
    "asymmetric": 700,
    "compressed": 600,
    "random": 700,
    "outside": 300,
}


def build_dataset() -> pd.DataFrame:
    rows = []

    for region, n in REGION_COUNTS.items():
        for _ in range(n):
            (
                bottom_diff_pct,
                rebound_ratio,
                left_span,
                right_span,
                total_span,
                symmetry_ratio,
            ) = _sample(region)

            base_score = double_bottom_score(
                bottom_diff_pct=bottom_diff_pct,
                rebound_ratio=rebound_ratio,
                left_span=left_span,
                right_span=right_span,
                total_span=total_span,
                symmetry_ratio=symmetry_ratio,
            )

            noise = np.random.normal(0, 2.0)
            quality = clamp_0_100(base_score + noise)

            rows.append({
                "bottom_diff_pct": bottom_diff_pct,
                "rebound_ratio": rebound_ratio,
                "left_span": left_span,
                "right_span": right_span,
                "total_span": total_span,
                "symmetry_ratio": symmetry_ratio,
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
            "Double Bottom regression model not found. Run "
            "`python -m stock_analysis.confidence_models.chart_patterns.double_bottom_model` "
            "to train and save it first."
        )

    model = joblib.load(MODEL_PATH)
    return model


def learned_confidence(
    bottom_diff_pct: float,
    rebound_ratio: float,
    left_span: float,
    right_span: float,
    total_span: float,
    symmetry_ratio: float,
    model: RandomForestRegressor,
) -> float:
    x = np.array([[
        bottom_diff_pct,
        rebound_ratio,
        left_span,
        right_span,
        total_span,
        symmetry_ratio,
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
        ("Ideal",         0.005, 0.065, 4, 4, 8, 1.00),
        ("Good",          0.012, 0.045, 3, 4, 7, 0.75),
        ("Borderline",    0.028, 0.030, 3, 3, 6, 1.00),
        ("Weak rebound",  0.012, 0.012, 3, 3, 6, 1.00),
        ("Bad bottom",    0.055, 0.040, 3, 3, 6, 1.00),
        ("Asymmetric",    0.015, 0.040, 2, 6, 8, 0.33),
        ("Compressed",    0.012, 0.040, 1, 1, 2, 1.00),
        ("User example",  0.03512, 0.031205, 10, 5, 15, 0.50),
    ]

    print(f"\n{'Pattern':<15} {'RuleScore':>10} {'Predicted':>10} {'Delta':>10}")
    print("-" * 52)

    for desc, bottom_diff_pct, rebound_ratio, left_span, right_span, total_span, symmetry_ratio in test_cases:
        rule_score = double_bottom_score(
            bottom_diff_pct=bottom_diff_pct,
            rebound_ratio=rebound_ratio,
            left_span=left_span,
            right_span=right_span,
            total_span=total_span,
            symmetry_ratio=symmetry_ratio,
        )
        pred_score = learned_confidence(
            bottom_diff_pct=bottom_diff_pct,
            rebound_ratio=rebound_ratio,
            left_span=left_span,
            right_span=right_span,
            total_span=total_span,
            symmetry_ratio=symmetry_ratio,
            model=model,
        )
        delta = pred_score - rule_score
        print(f"{desc:<15} {rule_score:>10.2f} {pred_score:>10.2f} {delta:>10.2f}")