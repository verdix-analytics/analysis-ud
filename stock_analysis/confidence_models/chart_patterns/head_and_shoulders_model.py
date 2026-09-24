import numpy as np
import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


np.random.seed(42)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "pkl_files")
MODEL_PATH = os.path.join(MODEL_DIR, "head_and_shoulders_model.pkl")

FEATURES = [
    "head_vs_shoulder_min",
    "shoulder_diff_pct",
    "neckline_diff_pct",
    "left_span",
    "right_span",
    "symmetry_ratio",
]


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def head_and_shoulders_score(
    head_vs_shoulder_min: float,
    shoulder_diff_pct: float,
    neckline_diff_pct: float,
    left_span: float,
    right_span: float,
    symmetry_ratio: float,
) -> float:
    """
    Strict rule-based quality score for a textbook-style Head and Shoulders.
    Output: [0, 100]
    """

    # 1) Head prominence: most important
    if head_vs_shoulder_min >= 0.040:
        score_head = 100
    elif head_vs_shoulder_min >= 0.030:
        score_head = 80
    elif head_vs_shoulder_min >= 0.020:
        score_head = 55
    elif head_vs_shoulder_min >= 0.010:
        score_head = 25
    else:
        score_head = 0

    # 2) Shoulder similarity: smaller difference is better
    if shoulder_diff_pct <= 0.010:
        score_shoulders = 100
    elif shoulder_diff_pct <= 0.020:
        score_shoulders = 80
    elif shoulder_diff_pct <= 0.030:
        score_shoulders = 55
    elif shoulder_diff_pct <= 0.040:
        score_shoulders = 25
    else:
        score_shoulders = 0

    # 3) Neckline consistency
    if neckline_diff_pct <= 0.010:
        score_neck = 100
    elif neckline_diff_pct <= 0.020:
        score_neck = 80
    elif neckline_diff_pct <= 0.040:
        score_neck = 55
    elif neckline_diff_pct <= 0.060:
        score_neck = 25
    else:
        score_neck = 0

    # 4) Structural symmetry
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

    overall = (
        0.36 * score_head +
        0.28 * score_shoulders +
        0.18 * score_neck +
        0.18 * score_sym
    )

    return clamp_0_100(overall)


def _sample(region: str):
    if region == "ideal":
        head_vs_shoulder_min = np.random.uniform(0.035, 0.070)
        shoulder_diff_pct = np.random.uniform(0.000, 0.010)
        neckline_diff_pct = np.random.uniform(0.000, 0.015)
        left_span = np.random.randint(3, 6)
        right_span = np.random.randint(3, 6)

    elif region == "good":
        head_vs_shoulder_min = np.random.uniform(0.025, 0.045)
        shoulder_diff_pct = np.random.uniform(0.010, 0.020)
        neckline_diff_pct = np.random.uniform(0.010, 0.025)
        left_span = np.random.randint(2, 6)
        right_span = np.random.randint(2, 6)

    elif region == "borderline":
        head_vs_shoulder_min = np.random.uniform(0.015, 0.030)
        shoulder_diff_pct = np.random.uniform(0.020, 0.035)
        neckline_diff_pct = np.random.uniform(0.020, 0.050)
        left_span = np.random.randint(2, 5)
        right_span = np.random.randint(2, 5)

    elif region == "weak_head":
        head_vs_shoulder_min = np.random.uniform(0.003, 0.015)
        shoulder_diff_pct = np.random.uniform(0.005, 0.025)
        neckline_diff_pct = np.random.uniform(0.005, 0.030)
        left_span = np.random.randint(2, 6)
        right_span = np.random.randint(2, 6)

    elif region == "bad_shoulders":
        head_vs_shoulder_min = np.random.uniform(0.020, 0.050)
        shoulder_diff_pct = np.random.uniform(0.035, 0.090)
        neckline_diff_pct = np.random.uniform(0.005, 0.030)
        left_span = np.random.randint(2, 6)
        right_span = np.random.randint(2, 6)

    elif region == "bad_neckline":
        head_vs_shoulder_min = np.random.uniform(0.020, 0.050)
        shoulder_diff_pct = np.random.uniform(0.005, 0.025)
        neckline_diff_pct = np.random.uniform(0.050, 0.120)
        left_span = np.random.randint(2, 6)
        right_span = np.random.randint(2, 6)

    elif region == "asymmetric":
        head_vs_shoulder_min = np.random.uniform(0.020, 0.050)
        shoulder_diff_pct = np.random.uniform(0.005, 0.030)
        neckline_diff_pct = np.random.uniform(0.005, 0.030)
        left_span = np.random.randint(1, 3)
        right_span = np.random.randint(5, 10)

    elif region == "compressed":
        head_vs_shoulder_min = np.random.uniform(0.020, 0.050)
        shoulder_diff_pct = np.random.uniform(0.005, 0.030)
        neckline_diff_pct = np.random.uniform(0.005, 0.030)
        left_span = 1
        right_span = 1

    elif region == "random":
        head_vs_shoulder_min = np.random.uniform(0.000, 0.080)
        shoulder_diff_pct = np.random.uniform(0.000, 0.120)
        neckline_diff_pct = np.random.uniform(0.000, 0.150)
        left_span = np.random.randint(1, 10)
        right_span = np.random.randint(1, 10)

    else:  # outside
        head_vs_shoulder_min = np.random.uniform(0.000, 0.008)
        shoulder_diff_pct = np.random.uniform(0.060, 0.180)
        neckline_diff_pct = np.random.uniform(0.080, 0.200)
        left_span = np.random.randint(1, 3)
        right_span = np.random.randint(1, 3)

    symmetry_ratio = min(left_span, right_span) / max(left_span, right_span)

    return (
        float(head_vs_shoulder_min),
        float(shoulder_diff_pct),
        float(neckline_diff_pct),
        float(left_span),
        float(right_span),
        float(symmetry_ratio),
    )


REGION_COUNTS = {
    "ideal": 1400,
    "good": 1200,
    "borderline": 1000,
    "weak_head": 900,
    "bad_shoulders": 900,
    "bad_neckline": 700,
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
                head_vs_shoulder_min,
                shoulder_diff_pct,
                neckline_diff_pct,
                left_span,
                right_span,
                symmetry_ratio,
            ) = _sample(region)

            base_score = head_and_shoulders_score(
                head_vs_shoulder_min=head_vs_shoulder_min,
                shoulder_diff_pct=shoulder_diff_pct,
                neckline_diff_pct=neckline_diff_pct,
                left_span=left_span,
                right_span=right_span,
                symmetry_ratio=symmetry_ratio,
            )

            noise = np.random.normal(0, 2.0)
            quality = clamp_0_100(base_score + noise)

            rows.append({
                "head_vs_shoulder_min": head_vs_shoulder_min,
                "shoulder_diff_pct": shoulder_diff_pct,
                "neckline_diff_pct": neckline_diff_pct,
                "left_span": left_span,
                "right_span": right_span,
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
            "Head and Shoulders regression model not found. Run "
            "`python -m stock_analysis.confidence_models.chart_patterns.head_and_shoulders_model` "
            "to train and save it first."
        )

    model = joblib.load(MODEL_PATH)
    return model


def learned_confidence(
    head_vs_shoulder_min: float,
    shoulder_diff_pct: float,
    neckline_diff_pct: float,
    left_span: float,
    right_span: float,
    symmetry_ratio: float,
    model: RandomForestRegressor,
) -> float:
    x = np.array([[
        head_vs_shoulder_min,
        shoulder_diff_pct,
        neckline_diff_pct,
        left_span,
        right_span,
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
        ("Ideal",         0.050, 0.008, 0.010, 4, 4, 1.00),
        ("Good",          0.032, 0.015, 0.020, 3, 4, 0.75),
        ("Borderline",    0.020, 0.028, 0.040, 3, 3, 1.00),
        ("Weak head",     0.008, 0.015, 0.020, 3, 3, 1.00),
        ("Bad shoulders", 0.030, 0.060, 0.020, 3, 3, 1.00),
        ("Bad neckline",  0.030, 0.015, 0.090, 3, 3, 1.00),
        ("Asymmetric",    0.030, 0.018, 0.020, 2, 6, 0.33),
        ("Compressed",    0.030, 0.018, 0.020, 1, 1, 1.00),
        ("User example",  0.018, 0.026, 0.031, 5, 3, 0.60),
    ]

    print(f"\n{'Pattern':<15} {'RuleScore':>10} {'Predicted':>10} {'Delta':>10}")
    print("-" * 52)

    for desc, head_vs_shoulder_min, shoulder_diff_pct, neckline_diff_pct, left_span, right_span, symmetry_ratio in test_cases:
        rule_score = head_and_shoulders_score(
            head_vs_shoulder_min=head_vs_shoulder_min,
            shoulder_diff_pct=shoulder_diff_pct,
            neckline_diff_pct=neckline_diff_pct,
            left_span=left_span,
            right_span=right_span,
            symmetry_ratio=symmetry_ratio,
        )
        pred_score = learned_confidence(
            head_vs_shoulder_min=head_vs_shoulder_min,
            shoulder_diff_pct=shoulder_diff_pct,
            neckline_diff_pct=neckline_diff_pct,
            left_span=left_span,
            right_span=right_span,
            symmetry_ratio=symmetry_ratio,
            model=model,
        )
        delta = pred_score - rule_score
        print(f"{desc:<15} {rule_score:>10.2f} {pred_score:>10.2f} {delta:>10.2f}")