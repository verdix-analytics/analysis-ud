import numpy as np
import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


np.random.seed(42)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "pkl_files")
MODEL_PATH = os.path.join(MODEL_DIR, "rectangle_model.pkl")

FEATURES = [
    "high_consistency_score",
    "low_consistency_score",
    "touch_balance_ratio",
    "touch_richness_score",
    "range_reasonable_score",
    "midline_stability_score",
]


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def rectangle_score(
    high_consistency_score: float,
    low_consistency_score: float,
    touch_balance_ratio: float,
    touch_richness_score: float,
    range_reasonable_score: float,
    midline_stability_score: float,
) -> float:
    """
    Strict rule-based quality score for a textbook-style Rectangle.
    Output: [0, 100]
    """

    if high_consistency_score >= 0.95:
        score_high = 100
    elif high_consistency_score >= 0.80:
        score_high = 80
    elif high_consistency_score >= 0.60:
        score_high = 55
    elif high_consistency_score >= 0.40:
        score_high = 25
    else:
        score_high = 0

    if low_consistency_score >= 0.95:
        score_low = 100
    elif low_consistency_score >= 0.80:
        score_low = 80
    elif low_consistency_score >= 0.60:
        score_low = 55
    elif low_consistency_score >= 0.40:
        score_low = 25
    else:
        score_low = 0

    if touch_balance_ratio >= 0.95:
        score_balance = 100
    elif touch_balance_ratio >= 0.80:
        score_balance = 80
    elif touch_balance_ratio >= 0.65:
        score_balance = 55
    elif touch_balance_ratio >= 0.45:
        score_balance = 25
    else:
        score_balance = 0

    if touch_richness_score >= 0.95:
        score_touches = 100
    elif touch_richness_score >= 0.80:
        score_touches = 80
    elif touch_richness_score >= 0.60:
        score_touches = 55
    elif touch_richness_score >= 0.40:
        score_touches = 25
    else:
        score_touches = 0

    if range_reasonable_score >= 0.95:
        score_range = 100
    elif range_reasonable_score >= 0.80:
        score_range = 80
    elif range_reasonable_score >= 0.60:
        score_range = 55
    elif range_reasonable_score >= 0.40:
        score_range = 25
    else:
        score_range = 0

    if midline_stability_score >= 0.95:
        score_mid = 100
    elif midline_stability_score >= 0.80:
        score_mid = 80
    elif midline_stability_score >= 0.60:
        score_mid = 55
    elif midline_stability_score >= 0.40:
        score_mid = 25
    else:
        score_mid = 0

    overall = (
        0.26 * score_high +
        0.26 * score_low +
        0.12 * score_balance +
        0.12 * score_touches +
        0.12 * score_range +
        0.12 * score_mid
    )

    return clamp_0_100(overall)


def _sample(region: str):
    if region == "ideal":
        high_consistency_score = np.random.uniform(0.88, 1.00)
        low_consistency_score = np.random.uniform(0.88, 1.00)
        touch_balance_ratio = np.random.uniform(0.85, 1.00)
        touch_richness_score = np.random.uniform(0.85, 1.00)
        range_reasonable_score = np.random.uniform(0.80, 1.00)
        midline_stability_score = np.random.uniform(0.85, 1.00)

    elif region == "good":
        high_consistency_score = np.random.uniform(0.70, 0.92)
        low_consistency_score = np.random.uniform(0.70, 0.92)
        touch_balance_ratio = np.random.uniform(0.70, 0.92)
        touch_richness_score = np.random.uniform(0.70, 0.92)
        range_reasonable_score = np.random.uniform(0.65, 0.92)
        midline_stability_score = np.random.uniform(0.70, 0.92)

    elif region == "borderline":
        high_consistency_score = np.random.uniform(0.45, 0.75)
        low_consistency_score = np.random.uniform(0.45, 0.75)
        touch_balance_ratio = np.random.uniform(0.45, 0.75)
        touch_richness_score = np.random.uniform(0.45, 0.75)
        range_reasonable_score = np.random.uniform(0.45, 0.75)
        midline_stability_score = np.random.uniform(0.45, 0.75)

    elif region == "messy_highs":
        high_consistency_score = np.random.uniform(0.00, 0.35)
        low_consistency_score = np.random.uniform(0.65, 0.95)
        touch_balance_ratio = np.random.uniform(0.65, 0.95)
        touch_richness_score = np.random.uniform(0.60, 0.95)
        range_reasonable_score = np.random.uniform(0.60, 0.95)
        midline_stability_score = np.random.uniform(0.60, 0.95)

    elif region == "messy_lows":
        high_consistency_score = np.random.uniform(0.65, 0.95)
        low_consistency_score = np.random.uniform(0.00, 0.35)
        touch_balance_ratio = np.random.uniform(0.65, 0.95)
        touch_richness_score = np.random.uniform(0.60, 0.95)
        range_reasonable_score = np.random.uniform(0.60, 0.95)
        midline_stability_score = np.random.uniform(0.60, 0.95)

    elif region == "imbalanced_touches":
        high_consistency_score = np.random.uniform(0.60, 0.95)
        low_consistency_score = np.random.uniform(0.60, 0.95)
        touch_balance_ratio = np.random.uniform(0.00, 0.35)
        touch_richness_score = np.random.uniform(0.55, 0.90)
        range_reasonable_score = np.random.uniform(0.60, 0.95)
        midline_stability_score = np.random.uniform(0.60, 0.95)

    elif region == "few_touches":
        high_consistency_score = np.random.uniform(0.60, 0.95)
        low_consistency_score = np.random.uniform(0.60, 0.95)
        touch_balance_ratio = np.random.uniform(0.60, 0.95)
        touch_richness_score = np.random.uniform(0.00, 0.35)
        range_reasonable_score = np.random.uniform(0.60, 0.95)
        midline_stability_score = np.random.uniform(0.60, 0.95)

    elif region == "bad_range":
        high_consistency_score = np.random.uniform(0.60, 0.95)
        low_consistency_score = np.random.uniform(0.60, 0.95)
        touch_balance_ratio = np.random.uniform(0.60, 0.95)
        touch_richness_score = np.random.uniform(0.60, 0.95)
        range_reasonable_score = np.random.uniform(0.00, 0.35)
        midline_stability_score = np.random.uniform(0.55, 0.90)

    elif region == "drifting_box":
        high_consistency_score = np.random.uniform(0.55, 0.90)
        low_consistency_score = np.random.uniform(0.55, 0.90)
        touch_balance_ratio = np.random.uniform(0.55, 0.90)
        touch_richness_score = np.random.uniform(0.55, 0.90)
        range_reasonable_score = np.random.uniform(0.55, 0.90)
        midline_stability_score = np.random.uniform(0.00, 0.35)

    elif region == "random":
        high_consistency_score = np.random.uniform(0.00, 1.00)
        low_consistency_score = np.random.uniform(0.00, 1.00)
        touch_balance_ratio = np.random.uniform(0.00, 1.00)
        touch_richness_score = np.random.uniform(0.00, 1.00)
        range_reasonable_score = np.random.uniform(0.00, 1.00)
        midline_stability_score = np.random.uniform(0.00, 1.00)

    else:  # outside
        high_consistency_score = np.random.uniform(0.00, 0.25)
        low_consistency_score = np.random.uniform(0.00, 0.25)
        touch_balance_ratio = np.random.uniform(0.00, 0.25)
        touch_richness_score = np.random.uniform(0.00, 0.25)
        range_reasonable_score = np.random.uniform(0.00, 0.25)
        midline_stability_score = np.random.uniform(0.00, 0.25)

    return (
        float(high_consistency_score),
        float(low_consistency_score),
        float(touch_balance_ratio),
        float(touch_richness_score),
        float(range_reasonable_score),
        float(midline_stability_score),
    )


REGION_COUNTS = {
    "ideal": 1400,
    "good": 1200,
    "borderline": 1000,
    "messy_highs": 900,
    "messy_lows": 900,
    "imbalanced_touches": 800,
    "few_touches": 800,
    "bad_range": 700,
    "drifting_box": 700,
    "random": 700,
    "outside": 300,
}


def build_dataset() -> pd.DataFrame:
    rows = []

    for region, n in REGION_COUNTS.items():
        for _ in range(n):
            (
                high_consistency_score,
                low_consistency_score,
                touch_balance_ratio,
                touch_richness_score,
                range_reasonable_score,
                midline_stability_score,
            ) = _sample(region)

            base_score = rectangle_score(
                high_consistency_score=high_consistency_score,
                low_consistency_score=low_consistency_score,
                touch_balance_ratio=touch_balance_ratio,
                touch_richness_score=touch_richness_score,
                range_reasonable_score=range_reasonable_score,
                midline_stability_score=midline_stability_score,
            )

            noise = np.random.normal(0, 2.0)
            quality = clamp_0_100(base_score + noise)

            rows.append({
                "high_consistency_score": high_consistency_score,
                "low_consistency_score": low_consistency_score,
                "touch_balance_ratio": touch_balance_ratio,
                "touch_richness_score": touch_richness_score,
                "range_reasonable_score": range_reasonable_score,
                "midline_stability_score": midline_stability_score,
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
            "Rectangle regression model not found. Run "
            "`python -m stock_analysis.confidence_models.chart_patterns.rectangle_model` "
            "to train and save it first."
        )
    model = joblib.load(MODEL_PATH)
    return model


def learned_confidence(
    high_consistency_score: float,
    low_consistency_score: float,
    touch_balance_ratio: float,
    touch_richness_score: float,
    range_reasonable_score: float,
    midline_stability_score: float,
    model: RandomForestRegressor,
) -> float:
    x = np.array([[
        high_consistency_score,
        low_consistency_score,
        touch_balance_ratio,
        touch_richness_score,
        range_reasonable_score,
        midline_stability_score,
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
        ("Ideal",            0.95, 0.95, 0.95, 0.95, 0.90, 0.95),
        ("Good",             0.82, 0.80, 0.80, 0.78, 0.80, 0.82),
        ("Borderline",       0.60, 0.62, 0.62, 0.58, 0.60, 0.58),
        ("Messy highs",      0.20, 0.85, 0.80, 0.80, 0.75, 0.75),
        ("Messy lows",       0.85, 0.20, 0.80, 0.80, 0.75, 0.75),
        ("Bad balance",      0.80, 0.80, 0.20, 0.80, 0.75, 0.75),
        ("Few touches",      0.80, 0.80, 0.80, 0.20, 0.75, 0.75),
        ("Bad range",        0.80, 0.80, 0.80, 0.80, 0.20, 0.75),
        ("Drifting box",     0.75, 0.75, 0.75, 0.75, 0.70, 0.20),
        ("User example",     0.72, 0.70, 0.68, 0.66, 0.74, 0.62),
    ]

    print(f"\n{'Pattern':<16} {'RuleScore':>10} {'Predicted':>10} {'Delta':>10}")
    print("-" * 54)

    for desc, high_consistency_score, low_consistency_score, touch_balance_ratio, touch_richness_score, range_reasonable_score, midline_stability_score in test_cases:
        rule_score = rectangle_score(
            high_consistency_score=high_consistency_score,
            low_consistency_score=low_consistency_score,
            touch_balance_ratio=touch_balance_ratio,
            touch_richness_score=touch_richness_score,
            range_reasonable_score=range_reasonable_score,
            midline_stability_score=midline_stability_score,
        )
        pred_score = learned_confidence(
            high_consistency_score=high_consistency_score,
            low_consistency_score=low_consistency_score,
            touch_balance_ratio=touch_balance_ratio,
            touch_richness_score=touch_richness_score,
            range_reasonable_score=range_reasonable_score,
            midline_stability_score=midline_stability_score,
            model=model,
        )
        delta = pred_score - rule_score
        print(f"{desc:<16} {rule_score:>10.2f} {pred_score:>10.2f} {delta:>10.2f}")